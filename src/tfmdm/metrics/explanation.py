"""Explanation multiplicity over a set of models trained by the same procedure.

Predictive multiplicity asks whether equally-accurate models *decide* differently
(``metrics.multiplicity``). Explanation multiplicity asks whether they *explain* the
same decision differently -- the failure mode that matters when an additive model is
deployed because it is interpretable. Four views of it, all measured pairwise over the
model set and averaged over the pairs:

  global      how much the ranking of term importances moves (Spearman correlation --
              high means agreement, so unlike the others this is a *stability* score)
  functional  how far apart two models' shape functions are, normalised per term (FED)
  local       how far apart two models' per-row attributions are (LAD)
  agreement   how often the top-k attributed terms for a row coincide (Jaccard)

Every function here takes aligned arrays and nothing else. The alignment -- which term
of model A corresponds to which term of model B, and what to do when a term exists in
one model and not the other -- is the caller's job; ``analysis.explanations`` does it
by term name over the union of the model set's terms.

Two conventions, both deliberate and both departures from a naive implementation:

*Contributions are centred per term before comparison.* An additive constant shared by
every row is absorbed by the model's intercept and changes no explanation; leaving it
in would report the intercept's arbitrary split between terms as disagreement. EBM
term scores are already centred, so this only binds for NAM.

*A term absent from a model contributes exactly zero*, rather than being dropped from
the comparison. An EBM that selected an interaction its neighbour did not is genuinely
explaining differently, and intersecting the term sets would hide precisely that.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import spearmanr


def _pairs(n_models: int):
    if n_models < 2:
        raise ValueError("Explanation multiplicity is undefined for fewer than two models")
    return combinations(range(n_models), 2)


def center_contributions(contributions: np.ndarray) -> np.ndarray:
    """Subtract each term's mean over the evaluation rows, per model.

    Shape (n_models, n_points, n_terms) in and out. See the module docstring for why.
    """
    contributions = np.asarray(contributions, dtype=float)
    if contributions.ndim != 3:
        raise ValueError(
            f"Expected (n_models, n_points, n_terms), got shape {contributions.shape}"
        )
    return contributions - contributions.mean(axis=1, keepdims=True)


def global_explanation_multiplicity(importances: np.ndarray) -> dict[str, float]:
    """Mean pairwise Spearman correlation of the global term-importance ranking.

    ``importances`` is (n_models, n_terms), already aligned term-by-term. This is the
    one metric where *high is good*: 1.0 means every model ranks the terms identically.
    """
    importances = np.asarray(importances, dtype=float)
    correlations = []
    for a, b in _pairs(importances.shape[0]):
        corr, _ = spearmanr(importances[a], importances[b])
        if not np.isnan(corr):
            correlations.append(float(corr))
    return {
        "mean_spearman_correlation": float(np.mean(correlations)) if correlations else np.nan,
        "n_importance_pairs": len(correlations),
    }


def functional_multiplicity(
    contributions: np.ndarray, orders: np.ndarray | None = None
) -> dict[str, float]:
    """Mean pairwise Functional Explanation Discrepancy (FED), normalised per term.

    For a pair of models and a term j, FED is the mean absolute gap between the two
    shape functions over the evaluation rows, divided by the wider of the two terms'
    ranges. The normalisation makes a term whose contribution spans 4 logits and one
    that spans 0.04 count the same, so the average over terms is not dominated by
    whichever term happens to be on the largest scale.

    Terms whose contribution is flat in *both* models (range below 1e-12) carry no
    shape to disagree about and are skipped rather than counted as agreement.

    ``orders`` gives each term's arity (1 = main effect, 2 = pairwise interaction);
    when supplied, FED is also broken out per order, because a model set can agree
    completely on its main effects and still disagree on which interactions exist.
    """
    contributions = np.asarray(contributions, dtype=float)
    n_models, _, n_terms = contributions.shape
    if orders is not None:
        orders = np.asarray(orders, dtype=int)
        if orders.size != n_terms:
            raise ValueError(f"orders has {orders.size} entries for {n_terms} terms")

    pair_means: list[float] = []
    by_order: dict[int, list[float]] = {}

    for a, b in _pairs(n_models):
        first, second = contributions[a], contributions[b]
        spans = np.maximum(np.ptp(first, axis=0), np.ptp(second, axis=0))
        gaps = np.abs(first - second).mean(axis=0)

        live = spans > 1e-12
        if not live.any():
            continue
        normalised = gaps[live] / spans[live]
        pair_means.append(float(normalised.mean()))

        if orders is not None:
            for order in np.unique(orders[live]):
                by_order.setdefault(int(order), []).extend(
                    normalised[orders[live] == order].tolist()
                )

    result = {
        "mean_normalized_fed": float(np.mean(pair_means)) if pair_means else np.nan,
        "n_fed_pairs": len(pair_means),
    }
    for order, values in sorted(by_order.items()):
        result[f"mean_fed_order{order}"] = float(np.mean(values))
    return result


def local_explanation_multiplicity(contributions: np.ndarray) -> dict[str, float]:
    """Mean pairwise Local Attribution Discrepancy (LAD).

    The mean absolute difference between two models' per-row, per-term attributions,
    averaged over pairs. Reported in logits, unnormalised -- it answers "by how much
    does the reason given for a row move", which is only meaningful on the scale the
    contributions live on.
    """
    contributions = np.asarray(contributions, dtype=float)
    scores = [
        float(np.abs(contributions[a] - contributions[b]).mean())
        for a, b in _pairs(contributions.shape[0])
    ]
    return {
        "mean_local_attribution_discrepancy": float(np.mean(scores)) if scores else np.nan,
        "n_lad_pairs": len(scores),
    }


def _top_k_masks(contributions: np.ndarray, k: int) -> np.ndarray:
    """(n_models, n_points, n_terms) boolean: is term j among row i's top-k for model m?"""
    magnitudes = np.abs(contributions)
    cut = magnitudes.shape[2] - k
    order = np.argpartition(magnitudes, cut, axis=2)[:, :, cut:]
    masks = np.zeros(magnitudes.shape, dtype=bool)
    np.put_along_axis(masks, order, True, axis=2)
    return masks


def explanation_agreement(contributions: np.ndarray, k: int = 3) -> dict[str, float]:
    """Mean Jaccard similarity between the top-k attributed terms, per row and pair.

    "Do two models point at the same reasons for this particular person?" -- the
    question a subject asking for an explanation actually cares about, and the one a
    correlation over global importances cannot answer. High is good.

    Computed on masks rather than per-instance Python sets: the naive loop is
    O(pairs x rows) set operations, which for 30 models over 8k rows is 3.5M of them.
    """
    contributions = np.asarray(contributions, dtype=float)
    n_models, _, n_terms = contributions.shape
    k = min(int(k), n_terms)
    if k < 1:
        raise ValueError("k must be at least 1")

    masks = _top_k_masks(contributions, k)
    sizes = masks.sum(axis=2)
    scores = []
    for a, b in _pairs(n_models):
        intersection = (masks[a] & masks[b]).sum(axis=1)
        union = sizes[a] + sizes[b] - intersection
        valid = union > 0
        if valid.any():
            scores.append(float((intersection[valid] / union[valid]).mean()))
    return {
        f"mean_jaccard_top{k}": float(np.mean(scores)) if scores else np.nan,
        f"n_jaccard_top{k}_pairs": len(scores),
    }


def term_set_agreement(term_names: list[list[str]]) -> dict[str, float]:
    """Mean pairwise Jaccard similarity of the *sets of terms* the models selected.

    Not in the original metric set, but it is what makes the others readable for EBM:
    the interaction terms are chosen per seed, so two models in the same arm can carry
    quite different term sets. When this is well below 1.0, part of every other
    explanation-multiplicity number here is term selection rather than term shape.
    """
    sets = [set(names) for names in term_names]
    scores = []
    for a, b in _pairs(len(sets)):
        union = sets[a] | sets[b]
        if union:
            scores.append(len(sets[a] & sets[b]) / len(union))
    return {
        "mean_term_set_jaccard": float(np.mean(scores)) if scores else np.nan,
        "mean_n_terms": float(np.mean([len(s) for s in sets])),
        "n_terms_union": len(set().union(*sets)) if sets else 0,
    }

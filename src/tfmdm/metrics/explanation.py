"""Explanation multiplicity over a set of models trained by the same procedure.

Predictive multiplicity asks whether equally-accurate models *decide* differently
(``metrics.multiplicity``). Explanation multiplicity asks whether they *explain* the
same decision differently -- the failure mode that matters when an additive model is
deployed because it is interpretable.

The module holds two generations of metric, and the split matters when reading a
results table.

The *pairwise magnitude* set below (Spearman over global importances, FED, LAD, top-k
agreement) came first. Each averages a distance over all model pairs. They stay because
they answer "by how much do two explanations differ", but none of them can carry the
headline comparison: a distance on the logit scale is not comparable between a
hard-label arm and a distilled arm, which trains on softer targets and so produces
systematically smaller contributions.

The *rank and sign* set at the bottom of the module is what the decision rule reads.
It is built on the order and the sign of the attributions rather than their size, which
makes it exactly invariant to per-model rescaling, and it is constructed against a
reference model exactly as ``metrics.multiplicity`` is -- so explanation ambiguity and
predictive ambiguity land on the same axis and can be compared point for point.

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


# ---------------------------------------------------------------------------------
# Plan step 3.1 -- the rank-based metric set the decision rule actually reads.
#
# Everything above compares two models by *how far apart* their contributions are.
# That is a magnitude, and a magnitude on the logit scale is not comparable between a
# hard-label arm and a distilled arm, which train on softer targets and so produce
# systematically smaller contributions. The metrics below are built on the *order* and
# the *sign* of the attributions instead, which makes them exactly invariant to any
# per-model rescaling -- multiplying one seed's contributions by a positive constant
# cannot change which term is largest or which way it points (decision D2, asserted as
# a bit-for-bit equality in the tests, not as a tolerance).
#
# The primary metric is the direct analogue of Marx et al.'s ambiguity, one axis over:
# "could this individual have been handed a different *reason* by an equally accurate
# model", rather than a different decision. It therefore shares the reference-model
# construction, and the closed-form jackknives, of ``metrics.multiplicity``.
# ---------------------------------------------------------------------------------

from dataclasses import asdict, dataclass

from . import multiplicity as mult


@dataclass(frozen=True)
class ExplanationMultiplicityResult:
    """The rank/sign metric set for one model set, on one set of rows."""

    attribution_ambiguity: float
    attribution_discrepancy: float
    sign_flip_rate: float
    mean_margin: float
    median_margin: float
    n_models: int
    n_points: int
    n_terms: int
    reference_index: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _as_tensor(contributions: np.ndarray) -> np.ndarray:
    """Shape check only -- for the reshaping helpers, which are happy with one model."""
    contributions = np.asarray(contributions, dtype=float)
    if contributions.ndim != 3:
        raise ValueError(
            f"Expected (n_models, n_points, n_terms), got shape {contributions.shape}"
        )
    return contributions


def _validate_tensor(contributions: np.ndarray) -> np.ndarray:
    """Shape check plus the model-set requirement, for the metrics themselves."""
    contributions = _as_tensor(contributions)
    if contributions.shape[0] < 2:
        raise ValueError("Explanation multiplicity is undefined for fewer than two models")
    return contributions


def group_terms(
    contributions: np.ndarray, names: list[str], mapping: dict[str, str]
) -> tuple[np.ndarray, list[str]]:
    """Sum contributions onto the units an explanation is read at (decision D4).

    ``mapping`` sends each term name to its group; groups come out in first-seen order
    so the column layout stays deterministic across cells. Summing is the only correct
    reduction here: the model is additive, so a parent feature's contribution to the
    logit *is* the sum of its columns' contributions, and any other aggregate would no
    longer reconstruct the prediction.
    """
    contributions = _as_tensor(contributions)
    if contributions.shape[2] != len(names):
        raise ValueError(f"{len(names)} names for {contributions.shape[2]} terms")

    groups: list[str] = []
    index: dict[str, int] = {}
    for name in names:
        group = mapping[name]
        if group not in index:
            index[group] = len(groups)
            groups.append(group)

    grouped = np.zeros((contributions.shape[0], contributions.shape[1], len(groups)))
    np.add.at(
        grouped.transpose(2, 0, 1),
        np.array([index[mapping[name]] for name in names]),
        contributions.transpose(2, 0, 1),
    )
    return grouped, groups


def top1_terms(contributions: np.ndarray) -> np.ndarray:
    """(n_models, n_points) index of each model's largest-magnitude term per row.

    Magnitude, not signed value: the term that most moved a decision is the reason
    given for it whether it pushed the prediction up or down.
    """
    return np.argmax(np.abs(_as_tensor(contributions)), axis=2)


def attribution_disagreement(
    contributions: np.ndarray, reference_index: int = 0
) -> np.ndarray:
    """Boolean (n_points, n_models): does model j attribute row i to a different term?

    Laid out exactly like ``multiplicity.disagreement_matrix`` -- reference column
    included and all-False -- so the two families of metric share their aggregators
    and their jackknives, and so E1 compares like with like.
    """
    top1 = top1_terms(contributions)
    return (top1 != top1[reference_index]).T


def attribution_ambiguity(disagree: np.ndarray) -> float:
    """Fraction of rows where *some* model gives a different top-1 reason. Primary metric."""
    return mult.ambiguity(disagree)


def attribution_discrepancy(disagree: np.ndarray) -> float:
    """Largest fraction of rows any single model re-attributes."""
    return mult.discrepancy(disagree)


def per_point_sign_flips(contributions: np.ndarray, reference_index: int = 0) -> np.ndarray:
    """(n_points,) share of (model, term) pairs whose sign differs from the reference.

    Terms the reference model leaves at exactly zero are excluded: they have no sign to
    flip, and counting them would report the EBM's block of unselected pair terms as
    perfect agreement, diluting every real flip.
    """
    contributions = _validate_tensor(contributions)
    reference = contributions[reference_index]
    live = np.sign(reference) != 0
    others = np.delete(np.sign(contributions), reference_index, axis=0)
    flips = (others != np.sign(reference)) & live
    denominator = live.sum(axis=1) * others.shape[0]
    total = flips.sum(axis=(0, 2))
    return np.divide(total, denominator, out=np.zeros(total.shape, dtype=float),
                     where=denominator > 0)


def sign_flip_rate(contributions: np.ndarray, reference_index: int = 0) -> float:
    return float(per_point_sign_flips(contributions, reference_index).mean())


def attribution_margin(contributions: np.ndarray, reference_index: int = 0) -> np.ndarray:
    """(n_points,) gap between the reference model's top-1 and top-2 |attribution| (D2).

    The guard on every rank metric here. A row whose two largest attributions are
    nearly tied will be re-attributed by arbitrarily small seed-to-seed noise, and
    counts as ambiguous for a reason with no interpretive content. Comparing the two
    arms' margin distributions separates "explains differently" from "has a flatter
    importance profile", which the primary metric alone cannot tell apart.
    """
    magnitudes = np.abs(_validate_tensor(contributions)[reference_index])
    if magnitudes.shape[1] < 2:
        return np.full(magnitudes.shape[0], np.inf)
    top2 = np.partition(magnitudes, -2, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def margin_sweep(
    disagree: np.ndarray, margins: np.ndarray, epsilons: np.ndarray
) -> list[dict[str, float]]:
    """Ambiguity recomputed over only the rows whose margin exceeds each epsilon.

    If the hard-vs-distilled gap in the primary metric survives the sweep, it is a
    statement about explanations; if it vanishes once near-ties are dropped, it was a
    statement about profile flatness (the plan's E2 caveat).
    """
    margins = np.asarray(margins, dtype=float)
    rows = []
    for eps in np.asarray(epsilons, dtype=float):
        keep = margins > eps
        rows.append({
            "epsilon": float(eps),
            "n_points_kept": int(keep.sum()),
            "share_kept": float(keep.mean()),
            "ambiguity": float(disagree[keep].any(axis=1).mean()) if keep.any() else np.nan,
        })
    return rows


def top_k_jaccard(importances: np.ndarray, k: int = 5) -> dict[str, float]:
    """Mean pairwise Jaccard of the top-k globally most important terms.

    Replaces the Spearman correlation above as the global stability metric. Spearman
    ranks *every* term, which over the NAM's ~80 near-zero one-hot columns or the EBM's
    block of exactly-zero unselected pair terms is dominated by the arbitrary ordering
    of ties -- two near-identical models can score near zero. A top-k set ignores the
    tied tail entirely and asks only whether the terms a reader would actually look at
    are the same ones.
    """
    importances = np.asarray(importances, dtype=float)
    if importances.ndim != 2:
        raise ValueError(f"Expected (n_models, n_terms), got shape {importances.shape}")
    k = min(int(k), importances.shape[1])
    if k < 1:
        raise ValueError("k must be at least 1")

    tops = [set(np.argsort(-row, kind="stable")[:k].tolist()) for row in importances]
    scores = [
        len(tops[a] & tops[b]) / len(tops[a] | tops[b])
        for a, b in _pairs(len(tops))
    ]
    return {
        f"mean_top{k}_jaccard": float(np.mean(scores)) if scores else np.nan,
        f"n_top{k}_jaccard_pairs": len(scores),
    }


def explanation_multiplicity(
    contributions: np.ndarray, reference_index: int = 0
) -> ExplanationMultiplicityResult:
    contributions = _validate_tensor(contributions)
    disagree = attribution_disagreement(contributions, reference_index)
    margins = attribution_margin(contributions, reference_index)
    return ExplanationMultiplicityResult(
        attribution_ambiguity=attribution_ambiguity(disagree),
        attribution_discrepancy=attribution_discrepancy(disagree),
        sign_flip_rate=sign_flip_rate(contributions, reference_index),
        mean_margin=float(np.mean(margins)),
        median_margin=float(np.median(margins)),
        n_models=int(contributions.shape[0]),
        n_points=int(contributions.shape[1]),
        n_terms=int(contributions.shape[2]),
        reference_index=int(reference_index),
    )


def sign_flip_jackknife(per_point: np.ndarray) -> np.ndarray:
    """Leave-one-point-out values of the sign-flip rate, in closed form.

    Like ambiguity, it is a mean over points, so deleting one just removes its term.
    """
    per_point = np.asarray(per_point, dtype=float)
    n = per_point.size
    return (per_point.sum() - per_point) / (n - 1)

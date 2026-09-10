"""Explanation multiplicity over the trained model sets (plan phases 1-4).

One cell is one (dataset, model, arm, split): the 30 models that arm trained, compared
against each other on how they *explain* the shared test set. Nothing is retrained --
this reads the ``.joblib`` artifacts the sweep already wrote.

Two hypotheses are served, and each is a *within-family* comparison. The three families
-- EBM, NAM and the linear baseline -- are not compared to each other here: EBM and NAM
differ in feature space, in structure (GA2M with interactions against a strictly
univariate GAM) and in internal ensembling, so a gap between them would be a statement
about those three choices rather than about the families. Both arms of a within-family
delta share all three exactly, so none of them can bias it. (NAM against logreg *is*
matched on all three -- same view, same univariate structure, no ensembling -- and is
a legitimate comparison, but it answers a different question, about flexibility rather
than about distillation; see plans/logreg.md.)

  E1  explanation multiplicity exceeds *predictive* multiplicity on the same model set
      -- seeds agree on decisions while disagreeing on reasons
  E2  distillation moves explanation multiplicity in the same direction it moves
      predictive multiplicity

The pipeline each cell goes through, and why each step is not optional:

  reload      every seed's fitted model, and check that its per-term contributions plus
              its intercept reproduce its own logit. An attribution that does not add
              back up to the prediction is not an explanation of that prediction, so
              this gates everything downstream (plan 1.1)
  centre      on the *train* marginal, folding what is removed into an effective
              intercept (D3). An additive model identifies its shape functions only up
              to a constant, and the NAM's one-hot level nets emit f(0) on every row
              where that level is absent -- an arbitrary per-seed constant with no
              per-point meaning. Rank metrics are invariant to rescaling but *not* to a
              shift, so leaving these in can make one feature win top-1 on every row
  group       to the unit the explanation is read at (D4): NAM one-hot columns summed
              back to their parent feature, EBM terms left alone. See ``grouping``
  measure     rank- and sign-based metrics, which are exactly invariant to the logit
              shrinkage distillation induces, plus the one magnitude-based metric
              (shape distance) reported both raw and normalised

Alignment across seeds is by term name over the *union* of the cell's terms, with a
term absent from a model contributing exactly zero. An EBM that selected an interaction
its neighbour did not is genuinely explaining differently; intersecting the term sets
would hide precisely that.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import paths, progress
from ..config import load, model_view, uses_encoded_view
from ..metrics import bootstrap as boot
from ..metrics import explanation as expl
from ..metrics import multiplicity as mult
from ..models.explain import reconstruction_error, term_contributions
from ..models.io import load_learner
from ..stages import train as train_stage
from . import grouping, shapes

TOP_K = (3, 5)

# Rows the additivity identity is re-checked on, per model. A handful is enough: the
# identity is exact or it is broken, and checking every row of every seed would triple
# the cost of the stage to re-answer a question already answered.
RESIDUAL_CHECK_ROWS = 256

# Tolerance on the *probability*-scale reconstruction. The plan states the check in
# logits, which is the right statement of the identity but not a usable test at the
# edges -- see ``models.explain.reconstruction_error``. 1e-5 sits two orders above the
# float32 noise a NAM's forward pass carries and far below any difference that could
# change an attribution.
RESIDUAL_TOLERANCE = 1e-5

# Quantiles of the reference model's top-1-minus-top-2 attribution gap, used as the
# epsilon grid of the margin sweep (3.4). Expressed as quantiles rather than absolute
# logits so the sweep means the same thing for arms on different logit scales.
MARGIN_QUANTILES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# The metric family Holm corrects over for E2. The primary metric is pre-declared and
# listed first; it is corrected alongside the rest rather than exempted.
E2_METRICS = ("attribution_ambiguity", "attribution_discrepancy", "sign_flip_rate")


@dataclass
class CellExplanation:
    """Everything one (dataset, model, arm, split) contributes to the analysis."""

    dataset: str
    model: str
    arm: str
    split_seed: int
    seeds: list[int]
    groups: list[str]
    contributions: np.ndarray          # (n_models, n_points, n_groups), centred + grouped
    eval_positions: np.ndarray         # positions into the split's sorted test rows
    per_model_terms: list[list[str]]   # raw term names, before grouping
    effective_intercepts: list[float]
    max_reconstruction_error: float
    shape_curves: list[dict[str, np.ndarray]] = field(default_factory=list)
    shape_scales: list[float] = field(default_factory=list)

    @property
    def importances(self) -> np.ndarray:
        """(n_models, n_groups) mean absolute contribution -- global importance.

        Computed on the evaluation rows rather than read from each learner's own
        ``term_importances``: the EBM defines those over its training bins and the NAM
        over its one-hot columns, which would put the two families on different
        footings and, for the NAM, at the wrong granularity entirely.
        """
        return np.abs(self.contributions).mean(axis=1)


def _evaluation_rows(x_test: pd.DataFrame, max_rows: int | None, seed: int = 7):
    """The rows every model in the cell is explained on.

    All of them by default. ``max_rows`` draws a fixed random subsample instead, for
    when the test set is large enough that a (n_models, n_rows, n_terms) tensor stops
    being comfortable. The positions are returned as well as the rows: E1 pairs each
    explanation metric against the *predictive* metric on the very same points, which
    is only possible if the subsample can be applied to the prediction matrix too.
    """
    if max_rows is None or len(x_test) <= max_rows:
        return x_test, np.arange(len(x_test))
    positions = np.sort(np.random.default_rng(seed).choice(len(x_test), max_rows, replace=False))
    return x_test.iloc[positions].reset_index(drop=True), positions


def _align(per_model: list[tuple[list[str], np.ndarray]]) -> tuple[np.ndarray, list[str]]:
    """Stack per-model (names, values) onto one tensor over the union of names."""
    names: list[str] = []
    seen: dict[str, int] = {}
    for model_names, _ in per_model:
        for name in model_names:
            if name not in seen:
                seen[name] = len(names)
                names.append(name)

    n_rows = per_model[0][1].shape[0]
    aligned = np.zeros((len(per_model), n_rows, len(names)), dtype=float)
    for m, (model_names, values) in enumerate(per_model):
        aligned[m][:, [seen[name] for name in model_names]] = values
    return aligned, names


def collect_cell(
    dataset: str, model: str, arm: str, split_seed: int,
    seed_list: Iterable[int], max_rows: int | None = None,
    with_shapes: bool = True,
) -> CellExplanation:
    """Load a cell's models and reduce them to one centred, grouped tensor."""
    ctx = train_stage.prepare(dataset, model, arm, split_seed)
    x_eval, positions = _evaluation_rows(ctx.x_test, max_rows)
    x_check = x_eval.head(RESIDUAL_CHECK_ROWS)

    shape_columns: list[str] = []
    grids: dict[str, np.ndarray] = {}
    if with_shapes:
        column_groups = grouping.group_map(dataset, model, split_seed, list(ctx.x_train.columns)) \
            if uses_encoded_view(model) else {}
        shape_columns = shapes.numeric_columns(ctx.x_train, model, column_groups)
        grids = {c: shapes.quantile_grid(ctx.x_train[c].to_numpy()) for c in shape_columns}

    progress.log(
        f"{dataset}/{model}/{arm} split{split_seed}: explaining {len(x_eval)} test rows"
        f"{' (+ shape curves)' if with_shapes else ''}"
    )

    per_model: list[tuple[list[str], np.ndarray]] = []
    per_model_terms: list[list[str]] = []
    intercepts: list[float] = []
    curves: list[dict[str, np.ndarray]] = []
    scales: list[float] = []
    residual = 0.0
    found: list[int] = []

    seed_list = list(seed_list)
    for seed in progress.track(seed_list, "models scored", total=len(seed_list)):
        path = paths.model_artifact(dataset, model, arm, seed, split_seed)
        if not path.exists():
            continue
        learner = load_learner(path)

        terms = term_contributions(learner, x_eval)
        # Plan 1.1: the decomposition is checked, not assumed -- before centring, when
        # a broken intercept would still show up, and again after (the fold-in makes
        # centring exactly identity-preserving, so the second check is the real one).
        terms_check = term_contributions(learner, x_check)
        residual = max(residual,
                       reconstruction_error(learner, x_check, terms_check)["max_prob_error"])

        # D3/1.2: centre on the *train* marginal, not on these rows. Centring on the
        # evaluation rows would make each model's zero depend on which rows are being
        # explained, so the same model would be centred differently in a subsample.
        train_means = term_contributions(learner, ctx.x_train).values.mean(axis=0)
        terms = terms.centred(train_means)
        residual = max(residual, reconstruction_error(
            learner, x_check, terms_check.centred(train_means))["max_prob_error"])

        # D4: group before alignment. For the EBM this is the identity; for the NAM it
        # collapses ~80 one-hot columns onto ~12 parent features, and is lossless only
        # because the inactive levels' constants were just removed.
        mapping = grouping.group_map(dataset, model, split_seed, terms.names)
        grouped, group_names = expl.group_terms(terms.values[None, :, :], terms.names, mapping)

        per_model.append((group_names, grouped[0]))
        per_model_terms.append(list(terms.names))
        intercepts.append(terms.intercept)
        if with_shapes:
            curves.append(shapes.curves(learner, ctx.x_train, shape_columns, grids))
            scales.append(shapes.total_scale(learner, x_eval))
        found.append(int(seed))

    if len(found) < 2:
        raise FileNotFoundError(
            f"Found {len(found)} fitted model(s) for {dataset}/{model}/{arm} under "
            f"split{split_seed}; explanation multiplicity needs at least two. Has the "
            "sweep run, and were the .joblib artifacts kept?"
        )
    if residual > RESIDUAL_TOLERANCE:
        raise AssertionError(
            f"{dataset}/{model}/{arm} split{split_seed}: term contributions plus intercept "
            f"miss the model's own prediction by up to {residual:.2e} (tolerance "
            f"{RESIDUAL_TOLERANCE:.0e}). The decomposition is not the model's, so every "
            "explanation metric below it would be meaningless. Plan step 1.1 stops here."
        )

    contributions, groups = _align(per_model)
    return CellExplanation(
        dataset=dataset, model=model, arm=arm, split_seed=split_seed, seeds=found,
        groups=groups, contributions=contributions, eval_positions=positions,
        per_model_terms=per_model_terms, effective_intercepts=intercepts,
        max_reconstruction_error=residual, shape_curves=curves, shape_scales=scales,
    )


# --- per-cell measurement ---------------------------------------------------------

def cell_row(cell: CellExplanation) -> dict:
    """The wide diagnostic row for one cell: every metric, no intervals."""
    result = expl.explanation_multiplicity(cell.contributions)
    margins = expl.attribution_margin(cell.contributions)

    row = {
        "dataset": cell.dataset, "model": cell.model, "arm": cell.arm,
        "split_seed": cell.split_seed, "n_models": len(cell.seeds),
        "n_eval_rows": int(cell.contributions.shape[1]),
        "n_groups": len(cell.groups),
        "n_terms_union": len(set().union(*(set(n) for n in cell.per_model_terms))),
        "max_reconstruction_error": cell.max_reconstruction_error,
        "effective_intercept_mean": float(np.mean(cell.effective_intercepts)),
        "effective_intercept_std": float(np.std(cell.effective_intercepts, ddof=1)),
        # n_points/n_terms restate n_eval_rows/n_groups, which the row already carries.
        **{k: v for k, v in result.as_dict().items() if k not in ("n_points", "n_terms")},
    }

    # 3.4/D2: the guard on every rank metric above. A flatter importance profile makes
    # ranks less determinate at identical underlying disagreement, and these columns
    # are what tells the two readings apart.
    for quantile in (0.1, 0.25, 0.5, 0.75, 0.9):
        row[f"margin_q{int(quantile * 100)}"] = float(np.quantile(margins, quantile))
    row["top1_share_of_most_frequent_term"] = float(
        np.bincount(expl.top1_terms(cell.contributions)[0],
                    minlength=len(cell.groups)).max() / cell.contributions.shape[1]
    )

    for k in TOP_K:
        row.update(expl.top_k_jaccard(cell.importances, k))
    row.update(expl.term_set_agreement(cell.per_model_terms))

    # The pairwise magnitude metrics, kept as secondary reading. They answer "by how
    # much", which the rank metrics deliberately cannot.
    orders = np.ones(len(cell.groups), dtype=int)
    row.update(expl.functional_multiplicity(cell.contributions, orders))
    row.update(expl.local_explanation_multiplicity(cell.contributions))
    row.update(expl.global_explanation_multiplicity(cell.importances))

    if cell.shape_curves:
        row.update(shapes.pairwise_distance(cell.shape_curves, cell.shape_scales))
    return row


def cell_intervals(cell: CellExplanation, n_boot: int) -> list[dict]:
    """One long-format row per metric, with a BCa interval over test points.

    Resampling is over *points*, not seeds: these are already statistics of the whole
    model set, so their sampling variability comes from which points are in the test
    set. The jackknife each interval needs is available in closed form because every
    metric here is a mean or a max-of-means over points.
    """
    disagree = expl.attribution_disagreement(cell.contributions)
    flips = expl.per_point_sign_flips(cell.contributions)
    n = disagree.shape[0]

    specs = [
        ("attribution_ambiguity",
         lambda idx: float(disagree[idx].any(axis=1).mean()),
         mult.ambiguity_jackknife(disagree), 21),
        ("attribution_discrepancy",
         lambda idx: float(disagree[idx].mean(axis=0).max()),
         mult.discrepancy_jackknife(disagree), 22),
        ("sign_flip_rate",
         lambda idx: float(flips[idx].mean()),
         expl.sign_flip_jackknife(flips), 23),
    ]

    rows = []
    for metric, stat, jackknife, seed in specs:
        interval = boot.bca_ci(stat, n, jackknife, n_boot=n_boot, seed=seed)
        rows.append({
            "dataset": cell.dataset, "model": cell.model, "arm": cell.arm,
            "split_seed": cell.split_seed, "metric": metric,
            "n_models": len(cell.seeds), "n_eval_rows": n,
            **interval.as_dict(),
        })
    return rows


def margin_sweep_rows(cell: CellExplanation) -> list[dict]:
    """3.4 -- ambiguity restricted to rows whose attribution margin exceeds epsilon."""
    disagree = expl.attribution_disagreement(cell.contributions)
    margins = expl.attribution_margin(cell.contributions)
    epsilons = np.quantile(margins, MARGIN_QUANTILES)
    return [
        {"dataset": cell.dataset, "model": cell.model, "arm": cell.arm,
         "split_seed": cell.split_seed, "margin_quantile": q, **row}
        for q, row in zip(MARGIN_QUANTILES, expl.margin_sweep(disagree, margins, epsilons))
    ]


# --- comparisons ------------------------------------------------------------------

def _predictive_disagreement(cell: CellExplanation, cfg) -> np.ndarray | None:
    """The predictive disagreement matrix on exactly the rows ``cell`` was explained on.

    Returns None when the cell's model set and the prediction files do not line up --
    E1 is then simply not reported for that cell rather than being reported against a
    mismatched baseline.
    """
    from .aggregate import collect_arm

    try:
        arm_result = collect_arm(cell.dataset, cell.model, cell.arm, cell.split_seed, cell.seeds)
    except FileNotFoundError:
        return None
    if list(arm_result.seeds) != list(cell.seeds):
        return None
    disagree = mult.disagreement_matrix(arm_result.test_probs, float(cfg.eval.threshold))
    return disagree[cell.eval_positions]


def compare_to_predictive(cell: CellExplanation, cfg) -> list[dict]:
    """E1 -- explanation ambiguity minus predictive ambiguity, on shared points.

    Paired, because both are computed on the identical rows: the difference has far
    less variance than the two marginals suggest, and pairing is the whole reason the
    explanation metric was built with the same reference-model construction as the
    predictive one.
    """
    predictive = _predictive_disagreement(cell, cfg)
    if predictive is None:
        return []

    explanation = expl.attribution_disagreement(cell.contributions)
    n = explanation.shape[0]
    interval, p_value = boot.paired_bootstrap(
        lambda idx: float(explanation[idx].any(axis=1).mean()),
        lambda idx: float(predictive[idx].any(axis=1).mean()),
        n, n_boot=int(cfg.eval.n_boot), seed=31,
    )
    base = float(predictive.any(axis=1).mean())
    return [{
        "dataset": cell.dataset, "model": cell.model, "split_seed": cell.split_seed,
        "hypothesis": "E1", "arm_a": cell.arm, "arm_b": cell.arm,
        "metric": "attribution_ambiguity_minus_predictive_ambiguity",
        "p_value": p_value,
        "baseline": base,
        "relative_change": (interval.point / base) if base > 0 else np.nan,
        **interval.as_dict("delta_"),
    }]


def compare_arms(a: CellExplanation, b: CellExplanation, cfg) -> list[dict]:
    """E2 -- distilled minus hard, per metric, paired over the shared test points."""
    if not np.array_equal(a.eval_positions, b.eval_positions):
        raise AssertionError(
            f"{a.arm} and {b.arm} were explained on different rows of "
            f"{a.dataset}/{a.model} split{a.split_seed}"
        )

    n_boot = int(cfg.eval.n_boot)
    da, db = (expl.attribution_disagreement(c.contributions) for c in (a, b))
    fa, fb = (expl.per_point_sign_flips(c.contributions) for c in (a, b))
    n = da.shape[0]

    statistics = {
        "attribution_ambiguity": (lambda d: (lambda idx: float(d[idx].any(axis=1).mean()))),
        "attribution_discrepancy": (lambda d: (lambda idx: float(d[idx].mean(axis=0).max()))),
    }
    common = {"dataset": a.dataset, "model": a.model, "split_seed": a.split_seed,
              "hypothesis": "E2", "arm_a": a.arm, "arm_b": b.arm}

    rows = []
    for seed, (metric, factory) in enumerate(statistics.items()):
        interval, p_value = boot.paired_bootstrap(
            factory(da), factory(db), n, n_boot=n_boot, seed=41 + seed
        )
        base = factory(db)(np.arange(n))
        rows.append({**common, "metric": metric, "p_value": p_value, "baseline": base,
                     "relative_change": (interval.point / base) if base > 0 else np.nan,
                     **interval.as_dict("delta_")})

    interval, p_value = boot.paired_bootstrap(
        lambda idx: float(fa[idx].mean()), lambda idx: float(fb[idx].mean()),
        n, n_boot=n_boot, seed=43,
    )
    base = float(fb.mean())
    rows.append({**common, "metric": "sign_flip_rate", "p_value": p_value, "baseline": base,
                 "relative_change": (interval.point / base) if base > 0 else np.nan,
                 **interval.as_dict("delta_")})

    rows += _epsilon_trimmed_delta(a, b, da, db, cfg)
    rows += _shape_delta(a, b)
    return rows


def _epsilon_trimmed_delta(a: CellExplanation, b: CellExplanation,
                           da: np.ndarray, db: np.ndarray, cfg) -> list[dict]:
    """The E2 primary delta recomputed away from the near-tie rows (3.4, D2).

    If distillation mainly *flattens* the importance profile, the arms' margins differ
    and the raw ambiguity delta is partly an artifact of rank indeterminacy. Trimming
    both arms at a common epsilon -- the median of the two arms' pooled margins, so the
    same absolute gap is demanded of each -- says whether the delta survives.
    """
    margins_a = expl.attribution_margin(a.contributions)
    margins_b = expl.attribution_margin(b.contributions)
    epsilon = float(np.median(np.concatenate([margins_a, margins_b])))
    keep = (margins_a > epsilon) & (margins_b > epsilon)
    if keep.sum() < 50:
        return []

    trimmed_a, trimmed_b = da[keep], db[keep]
    interval, p_value = boot.paired_bootstrap(
        lambda idx: float(trimmed_a[idx].any(axis=1).mean()),
        lambda idx: float(trimmed_b[idx].any(axis=1).mean()),
        int(keep.sum()), n_boot=int(cfg.eval.n_boot), seed=44,
    )
    base = float(trimmed_b.any(axis=1).mean())
    return [{
        "dataset": a.dataset, "model": a.model, "split_seed": a.split_seed,
        "hypothesis": "E2", "arm_a": a.arm, "arm_b": b.arm,
        "metric": "attribution_ambiguity_margin_trimmed", "p_value": p_value,
        "baseline": base, "relative_change": (interval.point / base) if base > 0 else np.nan,
        "epsilon": epsilon, "share_kept": float(keep.mean()),
        **interval.as_dict("delta_"),
    }]


def _shape_delta(a: CellExplanation, b: CellExplanation) -> list[dict]:
    """The magnitude-based metric, raw and normalised, as a plain difference.

    No interval: the shape distance is an average over model *pairs* on a fixed
    quantile grid, not a mean over test points, so the point bootstrap the other
    metrics use does not apply to it. Reported for the raw-versus-normalised
    comparison the risk table asks for -- if the two disagree, the arms differ in logit
    scale rather than in shape.
    """
    if not (a.shape_curves and b.shape_curves):
        return []
    da = shapes.pairwise_distance(a.shape_curves, a.shape_scales)
    db = shapes.pairwise_distance(b.shape_curves, b.shape_scales)
    common = {"dataset": a.dataset, "model": a.model, "split_seed": a.split_seed,
              "hypothesis": "E2", "arm_a": a.arm, "arm_b": b.arm,
              "p_value": np.nan, "delta_ci_method": "none"}
    return [
        {**common, "metric": key, "baseline": db[key], "delta_point": da[key] - db[key],
         "relative_change": ((da[key] - db[key]) / db[key]) if db[key] else np.nan}
        for key in ("mean_shape_distance", "mean_shape_distance_normalized")
        if key in da and key in db
    ]


# --- drivers ----------------------------------------------------------------------

def run(datasets: list[str], models: list[str], arms: list[str], split_seed: int,
        max_rows: int | None = None, with_shapes: bool = True) -> dict:
    """Analyse one split replicate; writes the explanation_* tables under results/."""
    paths.ensure_dirs(split_seed)
    cells: list[dict] = []
    summaries: list[dict] = []
    sweeps: list[dict] = []
    comparisons: list[dict] = []

    for dataset in datasets:
        cfg = load(dataset, split_seed=split_seed)
        seed_list = [int(s) for s in cfg.model_seeds]
        n_boot = int(cfg.eval.n_boot)

        for model in models:
            collected: dict[str, CellExplanation] = {}
            for arm in arms:
                label = f"{dataset}/{model}/{arm}"
                try:
                    with progress.phase(f"{label}: loading and reducing model set"):
                        cell = collect_cell(dataset, model, arm, split_seed, seed_list,
                                            max_rows, with_shapes)
                except (FileNotFoundError, NotImplementedError) as exc:
                    progress.log(f"{label}: SKIPPED -- {exc}")
                    cells.append({"dataset": dataset, "model": model, "arm": arm,
                                  "split_seed": split_seed, "error": str(exc)})
                    continue

                collected[arm] = cell
                row = cell_row(cell)
                cells.append(row)
                progress.log(
                    f"{label}: attribution ambiguity {row['attribution_ambiguity']:.3f}, "
                    f"discrepancy {row['attribution_discrepancy']:.3f}, "
                    f"sign flips {row['sign_flip_rate']:.3f} "
                    f"over {row['n_groups']} terms"
                )
                with progress.phase(f"{label}: BCa intervals ({n_boot} resamples)"):
                    summaries += cell_intervals(cell, n_boot)
                sweeps += margin_sweep_rows(cell)
                with progress.phase(f"{label}: E1 against predictive multiplicity"):
                    comparisons += compare_to_predictive(cell, cfg)

            # Plan 1.3: the grouping map is an artifact in its own right -- it records
            # the granularity every number in these tables was measured at. Written once
            # per family, over the union of the terms every arm produced, because an
            # EBM's interaction set differs between arms as well as between seeds.
            if collected:
                terms = sorted(set().union(*(set(n) for cell in collected.values()
                                             for n in cell.per_model_terms)))
                grouping.write(dataset, model_view(model), split_seed,
                               grouping.group_map(dataset, model, split_seed, terms))

            if "distilled" in collected and "hard" in collected:
                with progress.phase(f"{dataset}/{model}: E2 distilled vs hard"):
                    comparisons += compare_arms(collected["distilled"], collected["hard"], cfg)

    _holm_within(comparisons, "E1", lambda c: True)
    for metric in E2_METRICS:
        _holm_within(comparisons, "E2", lambda c, m=metric: c["metric"] == m)

    out = paths.results_dir(split_seed)
    out.mkdir(parents=True, exist_ok=True)

    # Written by merge, not by overwrite: a run restricted to one family (``--models
    # nam``) must not delete the other family's rows. The cost of getting this wrong is
    # silent -- the table still looks well-formed, it is just missing half the study.
    merged = {
        "explanation_multiplicity.csv": _merge_into(out / "explanation_multiplicity.csv",
                                                    cells, CELL_KEYS),
        "explanation_summaries.csv": _merge_into(out / "explanation_summaries.csv",
                                                 summaries, CELL_KEYS),
        "explanation_margin_sweep.csv": _merge_into(out / "explanation_margin_sweep.csv",
                                                    sweeps, CELL_KEYS),
        "explanation_comparisons.csv": _merge_into(out / "explanation_comparisons.csv",
                                                   comparisons, COMPARISON_KEYS),
    }
    for name, frame in merged.items():
        frame.to_csv(out / name, index=False)
        progress.log(f"wrote {out / name} ({len(frame)} rows)")

    (out / "explanation_multiplicity.json").write_text(json.dumps(
        {"split_seed": split_seed,
         "cells": merged["explanation_multiplicity.csv"].to_dict(orient="records"),
         "comparisons": merged["explanation_comparisons.csv"].to_dict(orient="records")},
        indent=2, default=float,
    ))

    return {"split_seed": split_seed, "n_cells": len(cells),
            "n_errors": int(sum("error" in c for c in cells)),
            "n_summaries": len(summaries), "n_comparisons": len(comparisons)}


# What identifies a row for the purpose of replacing it. A cell table is keyed by the
# cell; the comparisons table has no ``arm`` column (it spans arms via arm_a/arm_b), so
# recomputing a family replaces all of that family's comparisons.
CELL_KEYS = ("dataset", "model", "arm", "split_seed")
COMPARISON_KEYS = ("dataset", "model", "split_seed")


def _merge_into(path, rows: list[dict], keys: tuple[str, ...]) -> pd.DataFrame:
    """Combine freshly computed rows with the ones already on disk.

    Rows whose key this run recomputed are replaced; every other row is kept, so
    ``--models nam`` today and ``--models ebm`` tomorrow build one complete table
    instead of each erasing the other.

    A file written by an *older* metric set is discarded rather than merged. Those rows
    cannot be aligned with these -- they were produced by different definitions -- and
    concatenating them would fill the new columns with NaN for half the table, which
    reads as "this cell has no attribution ambiguity" rather than as "this cell is
    stale". Losing them is safe: they have to be recomputed either way.
    """
    new = pd.DataFrame(rows)
    if new.empty or not path.exists():
        return new

    old = pd.read_csv(path)
    if old.empty:
        return new

    absent = sorted(set(new.columns) - set(old.columns))
    if absent:
        progress.log(
            f"{path.name}: discarding {len(old)} row(s) written by an older metric set "
            f"(missing {absent[:3]}{'...' if len(absent) > 3 else ''}); rerun the other "
            "cells to restore them"
        )
        return new

    keys = [k for k in keys if k in new.columns and k in old.columns]
    if not keys:
        return new

    recomputed = set(map(tuple, new[keys].astype(str).itertuples(index=False, name=None)))
    kept = old[~old[keys].astype(str).apply(tuple, axis=1).isin(recomputed)]
    if len(kept):
        progress.log(f"{path.name}: keeping {len(kept)} row(s) from earlier runs")
    return pd.concat([kept, new], ignore_index=True).sort_values(keys).reset_index(drop=True)


def _holm_within(comparisons: list[dict], hypothesis: str, predicate) -> None:
    """Holm-correct one metric family, in place.

    The family is the set of cells tested on *one* metric within one split: every
    (dataset, model family) this invocation covered -- 2 datasets x 3 families under
    the defaults. Splits are replicates of the same experiment rather than extra
    hypotheses, so pooling them into one correction would penalise the design for being
    repeated -- cross-split agreement is reported descriptively by ``combine``.

    Note that the family is what *this run* computed, not what ends up in the merged
    file. Running ``--models logreg`` on its own therefore corrects it against nothing
    and leaves the rows already on disk corrected over the narrower family they were
    computed in, which understates the correction for the table as a whole. Recompute
    all families in one invocation before reading the ``holm_reject`` column.
    """
    family = [c for c in comparisons
              if c["hypothesis"] == hypothesis and predicate(c) and np.isfinite(c["p_value"])]
    if not family:
        return
    for comparison, reject in zip(family, boot.holm([c["p_value"] for c in family])):
        comparison["holm_reject"] = bool(reject)


COMPILED_PREDICTIVE = [
    "auroc_mean", "auroc_std", "mean_auroc_point", "mean_auroc_ci_low",
    "mean_auroc_ci_high", "ambiguity", "discrepancy",
]

COMPILED_EXPLANATION = [
    "attribution_ambiguity", "attribution_discrepancy", "sign_flip_rate",
    "mean_margin", "median_margin", "top1_share_of_most_frequent_term",
    "mean_top3_jaccard", "mean_top5_jaccard", "mean_shape_distance",
    "mean_shape_distance_normalized", "mean_term_set_jaccard", "mean_n_terms",
    "mean_local_attribution_discrepancy", "mean_normalized_fed",
    "mean_spearman_correlation", "max_reconstruction_error",
]


def combine(split_seeds: list[int]) -> dict:
    """Pool the per-split files and build the two compiled reads.

    ``explanation_metrics.csv`` is one row per (dataset, model, arm, split_seed): the
    explanation metrics beside the AUROC they were bought at and the predictive
    multiplicity of the same model set. Multiplicity is never read without its
    accuracy -- a model set that always explains identically because every member is
    the same constant predictor would score perfectly here.

    ``across_splits.csv`` is the decision rule made readable: one row per (dataset,
    model, hypothesis, metric), carrying how many of the splits produced an interval
    clear of zero and in which direction. The plan's rule -- "in at least 3 of 5
    splits" -- is that column.
    """
    from .aggregate import _pool

    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    explanations = _pool(split_seeds, "explanation_multiplicity.csv")
    explanations.to_csv(paths.RESULTS / "all_explanation_multiplicity.csv", index=False)

    for name in ("explanation_summaries.csv", "explanation_margin_sweep.csv",
                 "explanation_comparisons.csv"):
        pooled = _pool(split_seeds, name)
        pooled.to_csv(paths.RESULTS / f"all_{name}", index=False)
        progress.log(f"pooled {len(pooled)} rows -> results/all_{name}")

    keys = ["dataset", "model", "arm", "split_seed"]
    summaries = _pool(split_seeds, "arm_summaries.csv")
    available = [c for c in COMPILED_PREDICTIVE if c in summaries.columns]

    compiled = explanations.merge(summaries[keys + available], on=keys, how="left")
    columns = keys + [c for c in ["n_models", "n_eval_rows", "n_groups", "n_terms_union"]
                      if c in compiled.columns]
    columns += [c for c in COMPILED_EXPLANATION if c in compiled.columns]
    columns += available
    compiled = compiled[columns + [c for c in compiled.columns if c not in columns]]
    compiled.to_csv(paths.RESULTS / "explanation_metrics.csv", index=False)
    progress.log(f"wrote results/explanation_metrics.csv ({len(compiled)} rows)")

    across = across_splits(_pool(split_seeds, "explanation_comparisons.csv"), len(split_seeds))
    across.to_csv(paths.RESULTS / "across_splits.csv", index=False)
    progress.log(
        f"wrote results/across_splits.csv ({len(across)} rows, "
        f"{int(across['meets_majority_rule'].sum())} meeting the majority rule)"
    )

    return {"n_splits": len(split_seeds), "n_rows": len(compiled),
            "n_across_split_rows": len(across),
            "n_missing_auroc": int(compiled["auroc_mean"].isna().sum())
            if "auroc_mean" in compiled.columns else len(compiled)}


MAJORITY_OF_SPLITS = 3


def across_splits(comparisons: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    """Collapse the per-split comparisons onto the decision rule (plan 4.3).

    A split "supports" a hypothesis when its interval excludes zero. The sign is
    counted separately from the significance because a hypothesis is only supported by
    splits pointing the *same* way -- two splits clear of zero in opposite directions
    are evidence against, not two thirds of the way to a result.
    """
    rows = []
    # arm_a is part of the key: an E1 row exists per *arm* (the hard arm's explanations
    # against the hard arm's decisions, and likewise distilled), and pooling the two
    # would average two different claims into one.
    for (dataset, model, hypothesis, metric, arm_a, arm_b), group in comparisons.groupby(
        ["dataset", "model", "hypothesis", "metric", "arm_a", "arm_b"], dropna=False
    ):
        clear = (group["delta_ci_low"] > 0) | (group["delta_ci_high"] < 0)
        positive = int((clear & (group["delta_point"] > 0)).sum())
        negative = int((clear & (group["delta_point"] < 0)).sum())
        # The direction is the effect's own sign, taken from the median across splits;
        # only the splits clear of zero *in that direction* count as support. Splits
        # significant the other way are evidence against, not partial support, so they
        # must not be able to satisfy the rule by sheer count.
        median = float(group["delta_point"].median())
        direction = "positive" if median > 0 else "negative" if median < 0 else "zero"
        supported = positive if direction == "positive" else negative
        rows.append({
            "dataset": dataset, "model": model, "hypothesis": hypothesis, "metric": metric,
            "arm_a": arm_a, "arm_b": arm_b,
            "n_splits": len(group), "n_splits_expected": n_splits,
            "n_clear_of_zero": int(clear.sum()),
            "n_positive": positive, "n_negative": negative,
            "mean_delta": float(group["delta_point"].mean()),
            "median_delta": median,
            "mean_relative_change": float(group["relative_change"].mean()),
            "direction": direction,
            "meets_majority_rule": bool(supported >= MAJORITY_OF_SPLITS),
        })
    return pd.DataFrame(rows).sort_values(
        ["hypothesis", "dataset", "model", "metric", "arm_a"]
    ).reset_index(drop=True)

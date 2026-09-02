"""Phase 5.4 -- explanation multiplicity over the trained model sets.

One row of output is one (dataset, model, arm, split) cell: the 30 models that arm
trained, compared against each other on how they *explain* the shared test set. This
reuses the fitted artifacts ``train`` already wrote, so it never retrains anything.

The alignment problem this module exists to solve: an EBM chooses its interaction
terms per seed, so two seeds of the same arm routinely carry different term sets --
36 terms for one, 12 for another. Metrics are therefore computed over the *union* of
the arm's term names, with a term missing from a model contributing zero (which is
what a missing term does). ``mean_term_set_jaccard`` reports how much of the measured
disagreement is term selection rather than term shape.

As with the predictive metrics, every model in a cell is evaluated on the same test
rows in the same order -- ``train.prepare`` rebuilds them from the frozen split, and
the seed loop asserts nothing moved.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

import joblib
import numpy as np
import pandas as pd

from .. import paths
from ..config import load
from ..metrics import explanation as expl
from ..models.explain import term_contributions
from ..stages import train as train_stage

TOP_K = (3, 5)


def _evaluation_rows(x_test: pd.DataFrame, max_rows: int | None, seed: int = 7):
    """The rows every model in the cell is explained on.

    All of them by default. ``max_rows`` draws a fixed random subsample instead, for
    when the test set is large enough that a (n_models, n_rows, n_terms) tensor stops
    being comfortable -- the metrics are means over rows, so a subsample estimates the
    same quantity, just noisier.
    """
    if max_rows is None or len(x_test) <= max_rows:
        return x_test, np.arange(len(x_test))
    index = np.sort(np.random.default_rng(seed).choice(len(x_test), max_rows, replace=False))
    return x_test.iloc[index].reset_index(drop=True), index


def collect_contributions(
    dataset: str, model: str, arm: str, split_seed: int,
    seed_list: Iterable[int], max_rows: int | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray, list[list[str]], list[int], int]:
    """Load the cell's models and align their term contributions onto one tensor.

    Returns (contributions, term names, term orders, per-model term names, seeds found,
    number of evaluation rows), where ``contributions`` is
    (n_models, n_rows, n_terms_union) and already centred per term.
    """
    ctx = train_stage.prepare(dataset, model, arm, split_seed)
    x_eval, _ = _evaluation_rows(ctx.x_test, max_rows)

    per_model: list = []
    found: list[int] = []
    for seed in seed_list:
        path = paths.model_artifact(dataset, model, arm, seed, split_seed)
        if not path.exists():
            continue
        per_model.append(term_contributions(joblib.load(path), x_eval))
        found.append(int(seed))

    if len(found) < 2:
        raise FileNotFoundError(
            f"Found {len(found)} fitted model(s) for {dataset}/{model}/{arm} under "
            f"split{split_seed}; explanation multiplicity needs at least two. Has the "
            "sweep run, and were the .joblib artifacts kept?"
        )

    # Union of term names, in first-seen order so the column layout is deterministic.
    names: list[str] = []
    orders: dict[str, int] = {}
    for tc in per_model:
        for name, order in zip(tc.names, tc.orders):
            if name not in orders:
                names.append(name)
                orders[name] = order

    lookup = {name: j for j, name in enumerate(names)}
    aligned = np.zeros((len(per_model), len(x_eval), len(names)), dtype=float)
    for m, tc in enumerate(per_model):
        aligned[m][:, [lookup[name] for name in tc.names]] = tc.values

    return (
        expl.center_contributions(aligned),
        names,
        np.array([orders[name] for name in names], dtype=int),
        [tc.names for tc in per_model],
        found,
        len(x_eval),
    )


def summarise_cell(
    dataset: str, model: str, arm: str, split_seed: int,
    seed_list: Iterable[int], max_rows: int | None = None,
) -> dict:
    """Every explanation-multiplicity metric for one (dataset, model, arm, split)."""
    contributions, names, orders, per_model_names, found, n_rows = collect_contributions(
        dataset, model, arm, split_seed, seed_list, max_rows
    )

    # Global importance is the mean absolute contribution per term, computed on the
    # same rows as everything else. EBM's own term_importances() is defined over its
    # training bins, which would put the two families on different footings.
    importances = np.abs(contributions).mean(axis=1)

    row = {
        "dataset": dataset, "model": model, "arm": arm, "split_seed": split_seed,
        "n_models": len(found), "n_eval_rows": n_rows, "n_terms_union": len(names),
        "n_interaction_terms_union": int((orders > 1).sum()),
    }
    row.update(expl.global_explanation_multiplicity(importances))
    row.update(expl.functional_multiplicity(contributions, orders))
    row.update(expl.local_explanation_multiplicity(contributions))
    for k in TOP_K:
        row.update(expl.explanation_agreement(contributions, k))
    row.update(expl.term_set_agreement(per_model_names))
    return row


def run(datasets: list[str], models: list[str], arms: list[str], split_seed: int,
        max_rows: int | None = None) -> dict:
    """Analyse one split replicate; writes results/split{K}/explanation_multiplicity.csv."""
    paths.ensure_dirs(split_seed)
    rows: list[dict] = []

    for dataset in datasets:
        seed_list = [int(s) for s in load(dataset, split_seed=split_seed).model_seeds]
        for model in models:
            for arm in arms:
                try:
                    rows.append(
                        summarise_cell(dataset, model, arm, split_seed, seed_list, max_rows)
                    )
                except (FileNotFoundError, NotImplementedError) as exc:
                    rows.append({"dataset": dataset, "model": model, "arm": arm,
                                 "split_seed": split_seed, "error": str(exc)})

    out = paths.results_dir(split_seed)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "explanation_multiplicity.csv", index=False)
    (out / "explanation_multiplicity.json").write_text(
        json.dumps({"split_seed": split_seed, "cells": rows}, indent=2, default=float)
    )
    return {"split_seed": split_seed, "n_cells": len(rows),
            "n_errors": int(sum("error" in r for r in rows))}


COMPILED_PREDICTIVE = [
    "auroc_mean", "auroc_std", "mean_auroc_point", "mean_auroc_ci_low",
    "mean_auroc_ci_high", "ambiguity", "discrepancy",
]

COMPILED_EXPLANATION = [
    "mean_spearman_correlation", "mean_normalized_fed", "mean_fed_order1",
    "mean_fed_order2", "mean_local_attribution_discrepancy", "mean_jaccard_top3",
    "mean_jaccard_top5", "mean_term_set_jaccard", "mean_n_terms",
]


def combine(split_seeds: list[int]) -> dict:
    """Pool the per-split files and build the compiled table.

    ``all_explanation_multiplicity.csv`` is the per-split files stacked, nothing
    dropped. ``explanation_metrics.csv`` is the compiled read: the explanation metrics
    beside the AUROC they were bought at and the predictive multiplicity of the same
    model set, one row per (dataset, model, arm, split_seed). Multiplicity is never
    read without its accuracy -- a model set that always explains identically because
    every member is the same constant predictor would score perfectly here.
    """
    from .aggregate import _pool

    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    explanations = _pool(split_seeds, "explanation_multiplicity.csv")
    explanations.to_csv(paths.RESULTS / "all_explanation_multiplicity.csv", index=False)

    keys = ["dataset", "model", "arm", "split_seed"]
    summaries = _pool(split_seeds, "arm_summaries.csv")
    available = [c for c in COMPILED_PREDICTIVE if c in summaries.columns]

    compiled = explanations.merge(summaries[keys + available], on=keys, how="left")
    columns = keys + [c for c in ["n_models", "n_eval_rows", "n_terms_union",
                                  "n_interaction_terms_union"] if c in compiled.columns]
    columns += [c for c in COMPILED_EXPLANATION if c in compiled.columns]
    columns += available
    compiled = compiled[columns + [c for c in compiled.columns if c not in columns]]
    compiled.to_csv(paths.RESULTS / "explanation_metrics.csv", index=False)

    return {"n_splits": len(split_seeds), "n_rows": len(compiled),
            "n_missing_auroc": int(compiled["auroc_mean"].isna().sum())
            if "auroc_mean" in compiled.columns else len(compiled)}

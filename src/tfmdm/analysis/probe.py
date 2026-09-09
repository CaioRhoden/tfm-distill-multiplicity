"""Plan step 1.1 -- the cheapest falsifying step, run on its own.

Every explanation metric in this package rests on one claim: that a fitted model's
per-term contributions, plus its intercept, reproduce that model's own logit. If they
do not, the "explanation" being measured is not an explanation of the model, and no
number downstream means anything -- so the claim is checked against real artifacts
before any of the analysis is run, and it is checked again per model inside
``analysis.explanations``.

Two family-specific risks this exists to catch, named in the plan's assumptions:

  NAM   a fitted NAM pickles the ``torch.device`` it was trained on. Reloading a
        CUDA-trained model on a CPU-only machine is the failure mode the plan flags,
        and this is where it surfaces as a message rather than as a stack trace in the
        middle of a sweep. The contributions also omit the bias, and ``feature_dropout``
        must be inactive under ``eval()`` for the identity to hold at all
  EBM   ``eval_terms`` must return every term's contribution including the pair terms,
        on the logit scale, with the model's own intercept left out

The identity is checked in *probability* space rather than in logits. See
``models.explain.reconstruction_error``: a NAM predicts in float32, so a confident row
arrives as p = 1.0 exactly and inverting the sigmoid there fabricates an error of
several logits out of a reconstruction correct to 1e-7. The logit residual is still
reported, over the rows where that inversion is conditioned.

It also reports the inactive-level offset the NAM carries before centring (D3): the
constant each one-hot level net emits on rows where that level is *absent*. That number
is the size of the shift D3 removes, and a rank metric is not shift-invariant.
"""

from __future__ import annotations

import numpy as np

from .. import paths, progress
from ..config import load
from ..models.explain import reconstruction_error, term_contributions
from ..models.io import load_learner
from ..stages import train as train_stage

TOLERANCE = 1e-5
PROBE_ROWS = 512


def probe_cell(dataset: str, model: str, arm: str, split_seed: int, seed: int) -> dict:
    """Reload one fitted model and check its decomposition against its own predictions."""
    path = paths.model_artifact(dataset, model, arm, seed, split_seed)
    if not path.exists():
        return {"dataset": dataset, "model": model, "arm": arm, "split_seed": split_seed,
                "seed": seed, "status": "missing", "path": str(path)}

    ctx = train_stage.prepare(dataset, model, arm, split_seed)
    x = ctx.x_test.head(PROBE_ROWS)
    learner = load_learner(path)

    terms = term_contributions(learner, x)
    raw = reconstruction_error(learner, x, terms)

    train_means = term_contributions(learner, ctx.x_train).values.mean(axis=0)
    centred = terms.centred(train_means)
    after = reconstruction_error(learner, x, centred)

    # How much of each row's logit was sitting in per-term constants rather than in the
    # row's own features. For the NAM this is dominated by the inactive one-hot levels.
    return {
        "dataset": dataset, "model": model, "arm": arm, "split_seed": split_seed,
        "seed": seed, "status": "ok",
        "n_terms": len(terms.names),
        "n_probe_rows": len(x),
        "device": str(getattr(learner, "device", "n/a")),
        "intercept": terms.intercept,
        "effective_intercept": centred.intercept,
        "offset_removed": centred.intercept - terms.intercept,
        "max_abs_train_mean": float(np.abs(train_means).max()),
        "max_prob_error": raw["max_prob_error"],
        "max_prob_error_centred": after["max_prob_error"],
        "max_logit_error_unsaturated": after["max_logit_error"],
        "n_saturated_rows": after["n_saturated"],
        "passes": bool(max(raw["max_prob_error"], after["max_prob_error"]) <= TOLERANCE),
    }


def run(datasets: list[str], models: list[str], arms: list[str], split_seed: int,
        seed: int = 0) -> dict:
    """One model per (dataset, family, arm). Non-zero exit if any decomposition fails."""
    rows = []
    for dataset in datasets:
        seed_list = [int(s) for s in load(dataset, split_seed=split_seed).model_seeds]
        probe_seed = seed if seed in seed_list else seed_list[0]
        for model in models:
            for arm in arms:
                row = probe_cell(dataset, model, arm, split_seed, probe_seed)
                if row["status"] == "ok":
                    progress.log(
                        f"{dataset}/{model}/{arm} s{probe_seed}: device={row['device']}, "
                        f"{row['n_terms']} terms, offset removed {row['offset_removed']:+.4f}, "
                        f"prob error {row['max_prob_error_centred']:.2e} "
                        f"-> {'PASS' if row['passes'] else 'FAIL'}"
                    )
                else:
                    progress.log(f"{dataset}/{model}/{arm}: {row['status']} ({row['path']})")
                rows.append(row)

    checked = [r for r in rows if r["status"] == "ok"]
    return {
        "split_seed": split_seed,
        "tolerance": TOLERANCE,
        "n_checked": len(checked),
        "n_missing": len(rows) - len(checked),
        "all_passed": bool(checked) and all(r["passes"] for r in checked),
        "worst_prob_error": max((r["max_prob_error_centred"] for r in checked), default=None),
        "worst_logit_error_unsaturated": max(
            (r["max_logit_error_unsaturated"] for r in checked), default=None),
        "probes": rows,
    }

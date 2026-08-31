"""Phase 4 -- training, either one cell or a whole (dataset, model, arm) group.

A *cell* is one (dataset, model, arm, seed): one trained model, one prediction file.
A *group* is all 30 run seeds of one cell family. Groups exist because the expensive
part of a cell is not the fit -- it is loading the feature view, the split and the soft
labels, which are identical for all 30 seeds. ``run_group`` pays that once.

The arms differ in exactly one place: the target vector handed to the learner, and the
matching validation objective. Everything else -- split, resample protocol,
hyperparameters, evaluation -- is held fixed, so any difference in the results is
attributable to the target and nothing else.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from .. import paths, provenance, seeds, wandb_logger
from ..config import apply_tuned, load, to_dict
from ..data import features as features_mod
from ..data import load_splits
from ..metrics import performance
from ..models import build, val_objective
from ..models.registry import importances


def _soft_targets(dataset: str, split_seed: int, split) -> tuple[np.ndarray, np.ndarray]:
    train = pd.read_parquet(paths.soft_train(dataset, split_seed))
    val = pd.read_parquet(paths.soft_val(dataset, split_seed))
    if not np.array_equal(train["row_index"].to_numpy(), split.train):
        raise AssertionError(
            f"Soft labels for {dataset} under split{split_seed} were generated against a "
            "different partition. Regenerate them (`task tabicl:softlabels`) before training."
        )
    if not np.array_equal(val["row_index"].to_numpy(), split.val):
        raise AssertionError(
            f"Validation soft labels for {dataset} under split{split_seed} are stale."
        )
    return train["prob"].to_numpy(), val["prob"].to_numpy()


def build_targets(dataset: str, arm: str, split_seed: int, split, y):
    """Return (train target, val target) for the arm.

    Hard labels stay 0/1; the distilled arm replaces them entirely -- decision D6 keeps
    the two from being mixed.
    """
    if arm == "hard":
        return y[split.train].astype(float), y[split.val].astype(float)
    if arm == "distilled":
        return _soft_targets(dataset, split_seed, split)
    raise ValueError(f"Unknown arm {arm!r}")


@dataclass
class GroupContext:
    """Everything shared by the 30 seeds of one (dataset, model, arm, split)."""

    cfg: object
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    t_train: np.ndarray
    t_val: np.ndarray
    split: object


def prepare(dataset: str, model: str, arm: str, split_seed: int) -> GroupContext:
    cfg = apply_tuned(load(dataset, model, split_seed=split_seed),
                      dataset, model, arm, split_seed)
    frame = features_mod.load_view(dataset, cfg.model.view, split_seed)
    split = load_splits(dataset, split_seed)
    x, y = features_mod.xy(frame)
    t_train, t_val = build_targets(dataset, arm, split_seed, split, y)
    return GroupContext(
        cfg=cfg,
        x_train=x.iloc[split.train].reset_index(drop=True),
        x_val=x.iloc[split.val].reset_index(drop=True),
        x_test=x.iloc[split.test].reset_index(drop=True),
        y_train=y[split.train], y_val=y[split.val], y_test=y[split.test],
        t_train=t_train, t_val=t_val, split=split,
    )


def _train_one(ctx: GroupContext, dataset: str, model: str, arm: str,
               seed: int, split_seed: int) -> dict:
    cfg = ctx.cfg
    seeds.seed_everything(seed)

    # D3: the same stratified bootstrap perturbation for every method and arm.
    if bool(cfg.resample.bootstrap):
        boot = seeds.stratified_bootstrap_indices(ctx.y_train, seed)
        x_fit, t_fit = ctx.x_train.iloc[boot].reset_index(drop=True), ctx.t_train[boot]
    else:
        x_fit, t_fit = ctx.x_train, ctx.t_train

    started = time.time()
    learner = build(model, seed=seed, params=to_dict(cfg.model.params))
    learner.fit(x_fit, t_fit, ctx.x_val, ctx.t_val, arm=arm)
    fit_seconds = time.time() - started

    p_val = learner.predict_proba(ctx.x_val)
    p_test = learner.predict_proba(ctx.x_test)

    # Test is always scored against the true labels, in every arm.
    metrics = {f"test_{k}": v for k, v in performance(ctx.y_test, p_test).items()}
    metrics.update({f"val_{k}": v for k, v in performance(ctx.y_val, p_val).items()})
    metrics["val_objective"] = val_objective(arm, ctx.t_val, p_val)
    metrics["fit_seconds"] = fit_seconds

    pd.DataFrame({
        "row_index": np.concatenate([ctx.split.val, ctx.split.test]),
        "split": ["val"] * ctx.split.val.size + ["test"] * ctx.split.test.size,
        "prob": np.concatenate([p_val, p_test]),
        "y_true": np.concatenate([ctx.y_val, ctx.y_test]),
    }).to_parquet(paths.preds(dataset, model, arm, seed, split_seed), index=False)

    joblib.dump(learner, paths.model_artifact(dataset, model, arm, seed, split_seed))

    try:
        paths.importances(dataset, model, arm, seed, split_seed).write_text(
            json.dumps(importances(learner, ctx.x_test), indent=2)
        )
    except NotImplementedError:
        pass

    run_config = {
        "dataset": dataset, "model": model, "arm": arm, "seed": seed,
        "split_seed": split_seed, "view": cfg.model.view,
        "params": to_dict(cfg.model.params), "tuned_from": cfg.model.get("tuned_from"),
        "bootstrap": bool(cfg.resample.bootstrap),
        **provenance.collect(paths.processed(dataset)),
    }
    with wandb_logger.run(
        name=f"{dataset}-{model}-{arm}-sp{split_seed}-s{seed}",
        group=f"{dataset}-{model}-sp{split_seed}",
        job_type=arm,
        config=run_config,
        tags=["phase4", dataset, model, arm, f"split{split_seed}"],
    ) as handle:
        handle.log(metrics)

    return {"status": "ok", "seed": seed, **metrics}


def run_group(
    dataset: str,
    model: str,
    arm: str,
    split_seed: int,
    run_seeds: Iterable[int] | None = None,
    *,
    allow_dirty: bool = False,
    overwrite: bool = False,
) -> dict:
    """Train every seed of one (dataset, model, arm, split), sharing the loaded data."""
    paths.ensure_dirs(split_seed)
    provenance.guard_clean_tree(allow_dirty)

    if run_seeds is None:
        run_seeds = [int(s) for s in load(dataset).model_seeds]
    run_seeds = list(run_seeds)

    todo = [s for s in run_seeds
            if overwrite or not paths.preds(dataset, model, arm, s, split_seed).exists()]
    if not todo:
        return {"status": "skipped", "dataset": dataset, "model": model, "arm": arm,
                "split_seed": split_seed, "n_seeds": len(run_seeds), "trained": 0}

    # Loading the view, split and soft labels is the expensive part of a cell, and it
    # is identical for every seed -- so it happens once, after the skip check.
    ctx = prepare(dataset, model, arm, split_seed)
    results = [_train_one(ctx, dataset, model, arm, seed, split_seed) for seed in todo]

    return {
        "status": "ok", "dataset": dataset, "model": model, "arm": arm,
        "split_seed": split_seed, "n_seeds": len(run_seeds), "trained": len(results),
        "skipped": len(run_seeds) - len(results),
        "total_fit_seconds": float(sum(r["fit_seconds"] for r in results)),
        "mean_test_auroc": float(np.mean([r["test_auroc"] for r in results])),
    }


def run(
    dataset: str,
    model: str,
    arm: str,
    seed: int,
    split_seed: int,
    *,
    allow_dirty: bool = False,
    overwrite: bool = False,
) -> dict:
    """One cell. Kept for debugging a single seed; the sweep uses run_group."""
    return run_group(dataset, model, arm, split_seed, [seed],
                     allow_dirty=allow_dirty, overwrite=overwrite)

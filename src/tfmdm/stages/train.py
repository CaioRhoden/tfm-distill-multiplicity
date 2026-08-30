"""Phase 4 -- one (dataset, model, arm, seed) cell of the sweep.

The arms differ in exactly one place: the target vector handed to the learner, and the
matching validation objective. Everything else -- split, resample protocol,
hyperparameters, evaluation -- is held fixed, so any difference in the results is
attributable to the target and nothing else.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .. import paths, provenance, seeds, wandb_logger
from ..config import SOFT_ARMS, apply_tuned, load, to_dict
from ..data import features as features_mod
from ..data import load_splits
from ..metrics import performance
from ..models import build, val_objective
from ..models.registry import importances


def _soft_targets(dataset: str, teacher: str, split) -> tuple[np.ndarray, np.ndarray]:
    train = pd.read_parquet(paths.soft_train(dataset, teacher))
    val = pd.read_parquet(paths.soft_val(dataset, teacher))
    if not np.array_equal(train["row_index"].to_numpy(), split.train):
        raise AssertionError(
            f"Soft labels for {dataset}/{teacher} were generated against a different split. "
            "Regenerate them (`task softlabels`) before training."
        )
    if not np.array_equal(val["row_index"].to_numpy(), split.val):
        raise AssertionError(f"Validation soft labels for {dataset}/{teacher} are stale.")
    return train["prob"].to_numpy(), val["prob"].to_numpy()


def build_targets(dataset: str, model: str, arm: str, seed: int, split, y):
    """Return (train target, val target) for the arm. Hard labels stay 0/1; soft arms
    replace them entirely -- decision D6 keeps the two from being mixed."""
    y_train, y_val = y[split.train].astype(float), y[split.val].astype(float)

    if arm == "hard":
        return y_train, y_val
    if arm == "distilled":
        return _soft_targets(dataset, "tabicl", split)
    raise ValueError(f"Unknown arm {arm!r}")


def run(
    dataset: str,
    model: str,
    arm: str,
    seed: int,
    *,
    allow_dirty: bool = False,
    overwrite: bool = False,
) -> dict:
    out_path = paths.preds(dataset, model, arm, seed)
    if out_path.exists() and not overwrite:
        # Idempotent skip: a resubmitted SLURM array reruns only the missing cells.
        return {"status": "skipped", "path": str(out_path)}

    paths.ensure_dirs()
    provenance.guard_clean_tree(allow_dirty)
    seeds.seed_everything(seed)

    cfg = apply_tuned(load(dataset, model), dataset, model, arm)
    frame = features_mod.load_view(dataset, cfg.model.view)
    split = load_splits(dataset)
    x, y = features_mod.xy(frame)

    t_train, t_val = build_targets(dataset, model, arm, seed, split, y)

    x_train = x.iloc[split.train].reset_index(drop=True)
    x_val = x.iloc[split.val].reset_index(drop=True)
    x_test = x.iloc[split.test].reset_index(drop=True)

    # D3: the same stratified bootstrap perturbation for every method and arm.
    if bool(cfg.resample.bootstrap):
        boot = seeds.stratified_bootstrap_indices(y[split.train], seed)
        x_fit, t_fit = x_train.iloc[boot].reset_index(drop=True), t_train[boot]
    else:
        x_fit, t_fit = x_train, t_train

    started = time.time()
    learner = build(model, seed=seed, params=to_dict(cfg.model.params))
    learner.fit(x_fit, t_fit, x_val, t_val, arm=arm)
    fit_seconds = time.time() - started

    p_val = learner.predict_proba(x_val)
    p_test = learner.predict_proba(x_test)

    # Test is always scored against the true labels, in every arm.
    metrics = {f"test_{k}": v for k, v in performance(y[split.test], p_test,
                                                     float(cfg.eval.threshold)).items()}
    metrics.update({f"val_{k}": v for k, v in performance(y[split.val], p_val,
                                                          float(cfg.eval.threshold)).items()})
    metrics["val_objective"] = val_objective(arm, t_val, p_val)
    metrics["fit_seconds"] = fit_seconds

    pd.DataFrame({
        "row_index": np.concatenate([split.val, split.test]),
        "split": ["val"] * split.val.size + ["test"] * split.test.size,
        "prob": np.concatenate([p_val, p_test]),
        "y_true": np.concatenate([y[split.val], y[split.test]]),
    }).to_parquet(out_path, index=False)

    try:
        imp = importances(learner, x_test)
        (paths.PREDS / f"{dataset}_{model}_{arm}_s{seed}_importances.json").write_text(
            json.dumps(imp, indent=2)
        )
    except NotImplementedError:
        pass

    run_config = {
        "dataset": dataset, "model": model, "arm": arm, "seed": seed,
        "view": cfg.model.view, "params": to_dict(cfg.model.params),
        "tuned_from": cfg.model.get("tuned_from"),
        "bootstrap": bool(cfg.resample.bootstrap),
        **provenance.collect(paths.processed(dataset)),
    }
    with wandb_logger.run(
        name=f"{dataset}-{model}-{arm}-s{seed}",
        group=f"{dataset}-{model}",
        job_type=arm,
        config=run_config,
        tags=["phase4", dataset, model, arm],
    ) as handle:
        handle.log(metrics)

    return {"status": "ok", "path": str(out_path), **metrics}

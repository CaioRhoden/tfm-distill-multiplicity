"""Phase 2.4 / R1 -- the TabICLv2 model set (baseline B3).

TabICLv2 does not train, so it has no initialisation seed and would show exactly zero
multiplicity if left alone. That is a fact about the API, not about the method, and
comparing it to a stochastically-trained EBM on that basis would be meaningless.

So TabICLv2 is perturbed the same way every other method is (decision D3): for seed s
its context is a stratified bootstrap of the training set, seeded with s. The 30
resulting prediction vectors form its model set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import paths, provenance, seeds
from ..config import load
from ..data import features as features_mod
from ..data import load_splits
from ..metrics import performance
from ..softlabels import get_backend

MODEL_NAME = "tabicl"
ARM_NAME = "incontext"


def run(dataset: str, seed: int, split_seed: int, *, allow_dirty: bool = False,
        overwrite: bool = False) -> dict:
    out_path = paths.preds(dataset, MODEL_NAME, ARM_NAME, seed, split_seed)
    if out_path.exists() and not overwrite:
        return {"status": "skipped", "path": str(out_path)}

    paths.ensure_dirs(split_seed)
    provenance.guard_clean_tree(allow_dirty)

    cfg = load(dataset, split_seed=split_seed)
    frame = features_mod.load_view(dataset, "raw", split_seed)
    split = load_splits(dataset, split_seed)
    x, y = features_mod.xy(frame)

    x_train, y_train = x.iloc[split.train].reset_index(drop=True), y[split.train]
    boot = seeds.stratified_bootstrap_indices(y_train, seed)
    ctx_x, ctx_y = x_train.iloc[boot].reset_index(drop=True), y_train[boot]

    backend = get_backend(max_context_rows=cfg.get("max_context_rows"))
    p_val = backend.fit_predict(ctx_x, ctx_y, x.iloc[split.val].reset_index(drop=True), seed)
    p_test = backend.fit_predict(ctx_x, ctx_y, x.iloc[split.test].reset_index(drop=True), seed)

    pd.DataFrame({
        "row_index": np.concatenate([split.val, split.test]),
        "split": ["val"] * split.val.size + ["test"] * split.test.size,
        "prob": np.concatenate([p_val, p_test]),
        "y_true": np.concatenate([y[split.val], y[split.test]]),
    }).to_parquet(out_path, index=False)

    metrics = {f"test_{k}": v for k, v in performance(y[split.test], p_test).items()}
    metrics["context_rows"] = int(len(ctx_x))

    return {"status": "ok", "path": str(out_path), **metrics}


def probe(dataset: str, split_seed: int, fraction: float = 0.05, seed: int = 0) -> dict:
    """Phase 2.1 feasibility probe: time and size a small run before committing the grid."""
    import time

    frame = features_mod.load_view(dataset, "raw", split_seed)
    split = load_splits(dataset, split_seed)
    x, y = features_mod.xy(frame)

    rng = np.random.default_rng(seed)
    keep = rng.choice(split.train, size=max(50, int(len(split.train) * fraction)), replace=False)
    query = rng.choice(split.test, size=max(50, int(len(split.test) * fraction)), replace=False)

    backend = get_backend()
    started = time.time()
    backend.fit_predict(x.iloc[keep].reset_index(drop=True), y[keep],
                        x.iloc[query].reset_index(drop=True), seed)
    elapsed = time.time() - started

    report = {"dataset": dataset, "split_seed": split_seed, "backend": backend.name,
              "context_rows": int(keep.size),
              "query_rows": int(query.size), "seconds": elapsed,
              "extrapolated_full_context_seconds": elapsed / fraction}
    try:
        import torch

        if torch.cuda.is_available():
            report["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    except ImportError:
        pass
    return report

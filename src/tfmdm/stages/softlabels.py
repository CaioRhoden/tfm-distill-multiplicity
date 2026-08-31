"""Phase 2 -- cross-fitted TabICLv2 probabilities for the distilled arm (R0)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import paths, provenance
from ..config import load
from ..data import features as features_mod
from ..data import load_splits
from ..softlabels import cross_fit, get_backend
from ..softlabels.crossfit import assert_honest


TEACHER = "tabicl"


def _teacher_fn(cfg):
    backend = get_backend(max_context_rows=cfg.get("max_context_rows"))

    def fn(ctx_x, ctx_y, query_x, seed):
        return backend.fit_predict(ctx_x, ctx_y, query_x, seed)

    return fn


def run(dataset: str, split_seed: int, allow_dirty: bool = False) -> dict:
    paths.ensure_dirs(split_seed)
    provenance.guard_clean_tree(allow_dirty)

    cfg = load(dataset, split_seed=split_seed)
    frame = features_mod.load_view(dataset, "raw", split_seed)
    split = load_splits(dataset, split_seed)
    x, y = features_mod.xy(frame)

    x_train, y_train = x.iloc[split.train].reset_index(drop=True), y[split.train]
    x_val = x.iloc[split.val].reset_index(drop=True)

    teacher_fn = _teacher_fn(cfg)

    result = cross_fit(
        teacher_fn, x_train, y_train,
        n_folds=int(cfg.softlabels.n_folds), seed=int(cfg.split.seed),
        compute_in_context=bool(cfg.softlabels.entropy_guard),
    )
    if cfg.softlabels.entropy_guard:
        assert_honest(result)

    # Phase 2.3: validation probabilities, full training set as context. No validation
    # row is in that context, so no cross-fitting is needed here.
    val_probs = np.asarray(teacher_fn(x_train, y_train, x_val, int(cfg.split.seed)), dtype=float)

    pd.DataFrame({
        "row_index": split.train,
        "prob": result.oof_probs,
        "fold": result.fold_ids,
        "hard_label": y_train,
    }).to_parquet(paths.soft_train(dataset, split_seed, TEACHER), index=False)

    pd.DataFrame({
        "row_index": split.val,
        "prob": val_probs,
        "hard_label": y[split.val],
    }).to_parquet(paths.soft_val(dataset, split_seed, TEACHER), index=False)

    from ..metrics import performance

    diagnostics = dict(result.diagnostics)
    diagnostics.update({f"val_{k}": v for k, v in performance(y[split.val], val_probs).items()})
    diagnostics.update({f"oof_{k}": v for k, v in performance(y_train, result.oof_probs).items()})
    diagnostics["n_train"] = int(len(x_train))
    diagnostics["teacher"] = TEACHER
    diagnostics["split_seed"] = split_seed

    paths.soft_diagnostics(dataset, split_seed, TEACHER).write_text(
        json.dumps(diagnostics, indent=2, default=float)
    )

    return diagnostics

"""Phase 3 -- two gates that run before any real sweep.

Both are cheap, so they run first: a broken metric or a broken reimplementation
discovered here costs minutes, and discovered in Phase 5 costs the whole sweep.
"""

from __future__ import annotations

import json

import numpy as np

from .. import paths, seeds
from ..config import load, to_dict
from ..data import features as features_mod
from ..data import load_splits
from ..metrics import performance
from ..metrics.multiplicity import multiplicity
from ..models import build

# Published references for the Phase 3.2 reproduction gate.
REFERENCE_AUROC = {("adult", "ebm"): 0.927, ("adult", "nam"): 0.907}
REPRODUCTION_TOLERANCE = 0.01


def _load(dataset: str, model: str, split_seed: int):
    cfg = load(dataset, model, split_seed=split_seed)
    frame = features_mod.load_view(dataset, cfg.model.view, split_seed)
    split = load_splits(dataset, split_seed)
    x, y = features_mod.xy(frame)
    return cfg, x, y, split


def degenerate_check(dataset: str = "adult", model: str = "ebm", split_seed: int = 0,
                     n_models: int = 5) -> dict:
    """3.1 -- identical seed, no resampling: multiplicity must be exactly zero.

    A metric implementation that cannot return 0 here will not return a trustworthy
    number anywhere else.
    """
    cfg, x, y, split = _load(dataset, model, split_seed)
    x_train = x.iloc[split.train].reset_index(drop=True)
    x_val = x.iloc[split.val].reset_index(drop=True)
    x_test = x.iloc[split.test].reset_index(drop=True)
    t_train, t_val = y[split.train].astype(float), y[split.val].astype(float)

    columns = []
    for _ in range(n_models):
        seeds.seed_everything(0)
        learner = build(model, seed=0, params=to_dict(cfg.model.params))
        learner.fit(x_train, t_train, x_val, t_val, arm="hard")
        columns.append(learner.predict_proba(x_test))

    result = multiplicity(np.column_stack(columns))
    passed = result.ambiguity == 0.0 and result.discrepancy == 0.0
    return {"check": "degenerate", "passed": bool(passed), **result.as_dict()}


def reproduction_check(dataset: str, model: str, split_seed: int = 0, seed: int = 0) -> dict:
    """3.2 -- hard-label single-seed AUROC against published numbers."""
    cfg, x, y, split = _load(dataset, model, split_seed)
    learner = build(model, seed=seed, params=to_dict(cfg.model.params))
    learner.fit(x.iloc[split.train].reset_index(drop=True), y[split.train].astype(float),
                x.iloc[split.val].reset_index(drop=True), y[split.val].astype(float), arm="hard")
    scores = performance(y[split.test], learner.predict_proba(x.iloc[split.test]))
    reference = REFERENCE_AUROC.get((dataset, model))
    passed = reference is None or abs(scores["auroc"] - reference) <= REPRODUCTION_TOLERANCE
    return {"check": "reproduction", "dataset": dataset, "model": model,
            "split_seed": split_seed, "reference_auroc": reference,
            "passed": bool(passed), **scores}


def run(datasets: list[str], models: list[str], split_seed: int) -> dict:
    paths.ensure_dirs(split_seed)
    results = [degenerate_check(split_seed=split_seed)]
    results += [reproduction_check(d, m, split_seed) for d in datasets for m in models]
    report = {"split_seed": split_seed,
              "all_passed": all(r["passed"] for r in results), "checks": results}
    out = paths.results_dir(split_seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sanity.json").write_text(json.dumps(report, indent=2, default=float))
    return report

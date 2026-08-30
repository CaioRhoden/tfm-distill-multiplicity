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
from ..metrics import multiplicity, performance
from ..models import build

# Published references for the Phase 3.3 reproduction gate.
REFERENCE_AUROC = {("adult", "ebm"): 0.927, ("adult", "nam"): 0.907}
REPRODUCTION_TOLERANCE = 0.01


def _load(dataset: str, model: str):
    cfg = load(dataset, model)
    frame = features_mod.load_view(dataset, cfg.model.view)
    split = load_splits(dataset)
    x, y = features_mod.xy(frame)
    return cfg, x, y, split


def degenerate_check(dataset: str = "adult", model: str = "ebm", n_models: int = 5) -> dict:
    """3.1 -- identical seed, no resampling: multiplicity must be exactly zero.

    A metric implementation that cannot return 0 here will not return a trustworthy
    number anywhere else.
    """
    cfg, x, y, split = _load(dataset, model)
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


def reproduction_check(dataset: str, model: str, seed: int = 0) -> dict:
    """3.3 -- hard-label single-seed AUROC against published numbers."""
    cfg, x, y, split = _load(dataset, model)
    learner = build(model, seed=seed, params=to_dict(cfg.model.params))
    learner.fit(x.iloc[split.train].reset_index(drop=True), y[split.train].astype(float),
                x.iloc[split.val].reset_index(drop=True), y[split.val].astype(float), arm="hard")
    scores = performance(y[split.test], learner.predict_proba(x.iloc[split.test]))
    reference = REFERENCE_AUROC.get((dataset, model))
    passed = reference is None or abs(scores["auroc"] - reference) <= REPRODUCTION_TOLERANCE
    return {"check": "reproduction", "dataset": dataset, "model": model,
            "reference_auroc": reference, "passed": bool(passed), **scores}


def run(datasets: list[str], models: list[str]) -> dict:
    paths.ensure_dirs()
    results = [degenerate_check()]
    results += [reproduction_check(d, m) for d in datasets for m in models]
    report = {"all_passed": all(r["passed"] for r in results), "checks": results}
    (paths.RESULTS / "sanity.json").write_text(json.dumps(report, indent=2, default=float))
    return report

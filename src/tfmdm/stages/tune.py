"""Phase 4.2 -- one random search per (dataset, model, arm), fixed across all seeds.

Tuning per seed would fold hyperparameter selection into the measured multiplicity and
confound the whole study: some of the seed-to-seed disagreement would be search noise
rather than training randomness. So the search runs once, at a dedicated tuning seed,
and the resulting config is committed and referenced by every run in that cell.
"""

from __future__ import annotations

import json

import numpy as np
from omegaconf import OmegaConf

from .. import paths
from ..config import load, to_dict
from ..data import features as features_mod
from ..data import load_splits
from ..models import build, val_objective
from .train import build_targets


def _sample(space: dict, rng: np.random.Generator) -> dict:
    return {key: values[int(rng.integers(len(values)))] for key, values in space.items()}


def run(dataset: str, model: str, arm: str, n_configs: int | None = None) -> dict:
    paths.ensure_dirs()
    cfg = load(dataset, model)
    n_configs = int(n_configs or cfg.tune.n_configs)
    rng = np.random.default_rng(int(cfg.tune.seed))

    frame = features_mod.load_view(dataset, cfg.model.view)
    split = load_splits(dataset)
    x, y = features_mod.xy(frame)
    t_train, t_val = build_targets(dataset, model, arm, int(cfg.tune.seed), split, y)

    x_train = x.iloc[split.train].reset_index(drop=True)
    x_val = x.iloc[split.val].reset_index(drop=True)

    space = to_dict(cfg.model.search_space)
    base = to_dict(cfg.model.params)
    trials: list[dict] = []

    for trial in range(n_configs):
        params = {**base, **_sample(space, rng)}
        learner = build(model, seed=int(cfg.tune.seed), params=params)
        learner.fit(x_train, t_train, x_val, t_val, arm=arm)
        score = val_objective(arm, t_val, learner.predict_proba(x_val))
        trials.append({"trial": trial, "params": params, "val_objective": float(score)})

    best = min(trials, key=lambda t: t["val_objective"])
    OmegaConf.save(
        OmegaConf.create({"dataset": dataset, "model": model, "arm": arm,
                          "tune_seed": int(cfg.tune.seed), "params": best["params"],
                          "val_objective": best["val_objective"]}),
        paths.tuned_config(dataset, model, arm),
    )
    (paths.TUNED / f"{dataset}_{model}_{arm}_trials.json").write_text(json.dumps(trials, indent=2))
    return best

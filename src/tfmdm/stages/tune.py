"""Phase 4.2 -- one random search per (dataset, model, arm, split), fixed across seeds.

Tuning per run seed would fold hyperparameter selection into the measured multiplicity
and confound the whole study: some of the seed-to-seed disagreement would be search
noise rather than training randomness. So the search runs once per cell family, at a
dedicated tuning seed, and the resulting config is committed and referenced by all 30
runs of that cell.

It does run once *per split*, though: the validation set moves with the partition, so
reusing one split's winner elsewhere would select hyperparameters on rows that are test
data in the other replicate.
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


def run(dataset: str, model: str, arm: str, split_seed: int,
        n_configs: int | None = None) -> dict:
    paths.ensure_dirs(split_seed)
    cfg = load(dataset, model, split_seed=split_seed)
    n_configs = int(n_configs or cfg.tune.n_configs)
    rng = np.random.default_rng(int(cfg.tune.seed))

    frame = features_mod.load_view(dataset, cfg.model.view, split_seed)
    split = load_splits(dataset, split_seed)
    x, y = features_mod.xy(frame)
    t_train, t_val = build_targets(dataset, arm, split_seed, split, y)

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
    out = paths.tuned_config(dataset, model, arm, split_seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        OmegaConf.create({"dataset": dataset, "model": model, "arm": arm,
                          "split_seed": split_seed, "tune_seed": int(cfg.tune.seed),
                          "params": best["params"], "val_objective": best["val_objective"]}),
        out,
    )
    out.with_name(f"{dataset}_{model}_{arm}_trials.json").write_text(json.dumps(trials, indent=2))
    return best

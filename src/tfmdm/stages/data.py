"""Phase 1 -- raw CSV to a frozen split and the two feature views, for one split seed.

Cleaning is split-independent and produces the same file every time; the split, the
views and the fitted encoder are per split.
"""

from __future__ import annotations

import json

from omegaconf import DictConfig

from .. import paths
from ..config import load
from ..data import clean as clean_mod
from ..data import features as features_mod
from ..data import loaders, split as split_mod


def run(dataset: str, split_seed: int) -> dict:
    cfg: DictConfig = load(dataset, split_seed=split_seed)
    paths.ensure_dirs(split_seed)

    frame = loaders.load_raw(cfg)
    loaders.write_interim(cfg, frame)

    cleaned, report = clean_mod.clean(cfg, frame)
    clean_mod.write(cfg, cleaned, report)

    split, balance = split_mod.make_splits(cfg, cleaned)
    split_mod.write(cfg, split, balance)

    views = features_mod.build_views(cfg, cleaned, split)
    features_mod.write(cfg, views)

    summary = {
        "dataset": dataset,
        "split_seed": split_seed,
        "cleaning": report,
        "balance": balance,
        "views": {name: list(frame.shape) for name, frame in views.items()},
    }
    (paths.split_root(split_seed) / f"{dataset}_data_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary

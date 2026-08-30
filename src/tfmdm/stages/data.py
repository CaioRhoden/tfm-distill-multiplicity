"""Phase 1 -- raw CSV to frozen splits and the two feature views."""

from __future__ import annotations

import json

from omegaconf import DictConfig

from .. import paths
from ..config import load
from ..data import clean as clean_mod
from ..data import features as features_mod
from ..data import loaders, split as split_mod


def run(dataset: str) -> dict:
    cfg: DictConfig = load(dataset)
    paths.ensure_dirs()

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
        "cleaning": report,
        "balance": balance,
        "views": {name: list(frame.shape) for name, frame in views.items()},
    }
    (paths.DATA_PROCESSED / f"{dataset}_data_summary.json").write_text(json.dumps(summary, indent=2))
    return summary

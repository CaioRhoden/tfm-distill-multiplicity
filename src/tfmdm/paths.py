"""Canonical filesystem layout. Every artifact path is derived here, never inlined."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("TFMDM_ROOT", Path(__file__).resolve().parents[2]))

CONFIGS = ROOT / "configs"
TUNED = CONFIGS / "tuned"

DATA_RAW = ROOT / "data"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"

ARTIFACTS = ROOT / "artifacts"
SOFTLABELS = ARTIFACTS / "softlabels"
PREDS = ARTIFACTS / "preds"
CKPT = ARTIFACTS / "ckpt"

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

WANDB_DIR = ROOT / "wandb"


def ensure_dirs() -> None:
    for path in (
        DATA_INTERIM, DATA_PROCESSED, ARTIFACTS, SOFTLABELS, PREDS, CKPT,
        RESULTS, FIGURES, WANDB_DIR, TUNED,
    ):
        path.mkdir(parents=True, exist_ok=True)


def interim(dataset: str) -> Path:
    return DATA_INTERIM / f"{dataset}.parquet"


def processed(dataset: str) -> Path:
    return DATA_PROCESSED / f"{dataset}.parquet"


def cleaning_report(dataset: str) -> Path:
    return DATA_PROCESSED / f"{dataset}_cleaning_report.json"


def splits(dataset: str) -> Path:
    return DATA_PROCESSED / f"{dataset}_splits.json"


def view(dataset: str, name: str) -> Path:
    return DATA_PROCESSED / f"{dataset}_{name}.parquet"


def transformer(dataset: str) -> Path:
    return DATA_PROCESSED / f"{dataset}_encoder.joblib"


def soft_train(dataset: str, source: str = "tabicl") -> Path:
    return SOFTLABELS / f"{dataset}_{source}_train_oof.parquet"


def soft_val(dataset: str, source: str = "tabicl") -> Path:
    return SOFTLABELS / f"{dataset}_{source}_val.parquet"


def preds(dataset: str, model: str, arm: str, seed: int) -> Path:
    return PREDS / f"{dataset}_{model}_{arm}_s{seed}.parquet"


def tuned_config(dataset: str, model: str, arm: str) -> Path:
    return TUNED / f"{dataset}_{model}_{arm}.yaml"

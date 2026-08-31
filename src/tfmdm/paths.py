"""Canonical filesystem layout. Every artifact path is derived here, never inlined.

The split seed is a first-class dimension: everything downstream of the partition
lives under ``artifacts/split{K}/``, so one split is one directory and deleting it is
one ``rm -rf``. Only the cleaned frame -- which is produced before any split exists --
is shared across splits.
"""

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
RESULTS = ROOT / "results"
WANDB_DIR = ROOT / "wandb"


# --- split-independent ------------------------------------------------------------

def interim(dataset: str) -> Path:
    return DATA_INTERIM / f"{dataset}.parquet"


def processed(dataset: str) -> Path:
    return DATA_PROCESSED / f"{dataset}.parquet"


def cleaning_report(dataset: str) -> Path:
    return DATA_PROCESSED / f"{dataset}_cleaning_report.json"


# --- per split --------------------------------------------------------------------

def split_root(split_seed: int) -> Path:
    return ARTIFACTS / f"split{split_seed}"


def splits(dataset: str, split_seed: int) -> Path:
    return split_root(split_seed) / "splits" / f"{dataset}.json"


def view(dataset: str, name: str, split_seed: int) -> Path:
    return split_root(split_seed) / "views" / f"{dataset}_{name}.parquet"


def transformer(dataset: str, split_seed: int) -> Path:
    return split_root(split_seed) / "views" / f"{dataset}_encoder.joblib"


def soft_train(dataset: str, split_seed: int, source: str = "tabicl") -> Path:
    return split_root(split_seed) / "softlabels" / f"{dataset}_{source}_train_oof.parquet"


def soft_val(dataset: str, split_seed: int, source: str = "tabicl") -> Path:
    return split_root(split_seed) / "softlabels" / f"{dataset}_{source}_val.parquet"


def soft_diagnostics(dataset: str, split_seed: int, source: str = "tabicl") -> Path:
    return split_root(split_seed) / "softlabels" / f"{dataset}_{source}_diagnostics.json"


def preds(dataset: str, model: str, arm: str, seed: int, split_seed: int) -> Path:
    return split_root(split_seed) / "preds" / f"{dataset}_{model}_{arm}_s{seed}.parquet"


def importances(dataset: str, model: str, arm: str, seed: int, split_seed: int) -> Path:
    return (split_root(split_seed) / "preds"
            / f"{dataset}_{model}_{arm}_s{seed}_importances.json")


def model_artifact(dataset: str, model: str, arm: str, seed: int, split_seed: int) -> Path:
    return (split_root(split_seed) / "models"
            / f"{dataset}_{model}_{arm}_s{seed}.joblib")


def tuned_config(dataset: str, model: str, arm: str, split_seed: int) -> Path:
    """Tuned hyperparameters live under configs/ so they can be committed.

    They are per split: the validation set moves with the partition, so reusing one
    split's search would select hyperparameters on rows that are test data elsewhere.
    """
    return TUNED / f"split{split_seed}" / f"{dataset}_{model}_{arm}.yaml"


def results_dir(split_seed: int) -> Path:
    return RESULTS / f"split{split_seed}"


def figures_dir(split_seed: int) -> Path:
    return results_dir(split_seed) / "figures"


def ensure_dirs(split_seed: int | None = None) -> None:
    for path in (DATA_INTERIM, DATA_PROCESSED, ARTIFACTS, RESULTS, WANDB_DIR, TUNED):
        path.mkdir(parents=True, exist_ok=True)
    if split_seed is None:
        return
    root = split_root(split_seed)
    for name in ("splits", "views", "softlabels", "preds", "models"):
        (root / name).mkdir(parents=True, exist_ok=True)
    figures_dir(split_seed).mkdir(parents=True, exist_ok=True)
    (TUNED / f"split{split_seed}").mkdir(parents=True, exist_ok=True)

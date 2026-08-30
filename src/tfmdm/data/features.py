"""Phase 1.4 -- two feature views, both fit on train only.

``raw``      : native dtypes, categoricals as strings. Consumed by TabICLv2 and EBM,
               which handle categoricals internally.
``encoded``  : numerics standardised, categoricals one-hot. Consumed by NAM and
               logistic regression.

Every transformer is fit on the training rows alone, so the encoding of a test row
never depends on any statistic drawn from the test set.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .. import paths
from .split import SplitIndex

VIEWS = ("raw", "encoded")


def _columns(cfg: DictConfig, frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in OmegaConf.to_container(cfg.dataset.numeric, resolve=True)
               if c in frame.columns]
    categorical = [c for c in OmegaConf.to_container(cfg.dataset.categorical, resolve=True)
                   if c in frame.columns]
    return numeric, categorical


def build_views(
    cfg: DictConfig, frame: pd.DataFrame, split: SplitIndex
) -> dict[str, pd.DataFrame]:
    numeric, categorical = _columns(cfg, frame)
    y = frame["target"]

    raw = frame.copy()
    for column in categorical:
        raw[column] = raw[column].astype(str)

    encoder = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                  min_frequency=20), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    encoder.fit(raw.iloc[split.train].drop(columns=["target"]))

    matrix = encoder.transform(raw.drop(columns=["target"]))
    encoded = pd.DataFrame(matrix, columns=list(encoder.get_feature_names_out()))
    encoded["target"] = y.to_numpy()

    paths.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    joblib.dump({"encoder": encoder, "numeric": numeric, "categorical": categorical},
                paths.transformer(cfg.dataset.name))
    return {"raw": raw, "encoded": encoded}


def write(cfg: DictConfig, views: dict[str, pd.DataFrame]) -> None:
    for name, frame in views.items():
        frame.to_parquet(paths.view(cfg.dataset.name, name), index=False)


def load_view(dataset: str, name: str) -> pd.DataFrame:
    if name not in VIEWS:
        raise ValueError(f"Unknown view {name!r}; expected one of {VIEWS}")
    return pd.read_parquet(paths.view(dataset, name))


def xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return frame.drop(columns=["target"]), frame["target"].to_numpy().astype(np.int8)

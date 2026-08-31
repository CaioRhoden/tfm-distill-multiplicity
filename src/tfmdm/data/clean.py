"""Phase 1.2 -- deduplicate, drop non-features, fold undocumented categories.

Deduplication happens *before* splitting, so a row that appears twice in Adult cannot
land in two different splits.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .. import paths

MISSING = "Missing"


def _coerce_keys(mapping: dict, series: pd.Series) -> dict:
    """YAML mapping keys arrive as strings; match them to the column's own dtype."""
    if pd.api.types.is_numeric_dtype(series):
        return {int(k): v for k, v in mapping.items()}
    return {str(k): v for k, v in mapping.items()}


def clean(cfg: DictConfig, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    ds = cfg.dataset
    report: dict[str, Any] = {"dataset": ds.name, "rows_in": int(len(frame))}

    drop = [c for c in OmegaConf.to_container(ds.drop_columns, resolve=True) if c in frame.columns]
    frame = frame.drop(columns=drop)
    report["dropped_columns"] = drop

    before = len(frame)
    frame = frame.drop_duplicates(keep="first").reset_index(drop=True)
    report["duplicate_rows_dropped"] = int(before - len(frame))

    recode_cfg = ds.get("recode", None)
    recode = OmegaConf.to_container(recode_cfg, resolve=True) if recode_cfg else {}
    applied: dict[str, int] = {}
    for column, mapping in recode.items():  # type: ignore[union-attr]
        if column not in frame.columns:
            continue
        mapping = _coerce_keys(mapping, frame[column])
        hits = int(frame[column].isin(mapping.keys()).sum())
        frame[column] = frame[column].replace(mapping)
        applied[column] = hits
    report["recoded_values"] = applied

    # Missingness is information here (Adult's '?' clusters in workclass/occupation),
    # so it becomes an explicit level rather than being imputed away.
    cat_cols = [c for c in OmegaConf.to_container(ds.categorical, resolve=True) if c in frame.columns]
    na_counts = {c: int(frame[c].isna().sum()) for c in cat_cols}
    for column in cat_cols:
        frame[column] = frame[column].astype("object").fillna(MISSING).astype(str).str.strip()
    report["categorical_missing_filled"] = {k: v for k, v in na_counts.items() if v}

    num_cols = [c for c in OmegaConf.to_container(ds.numeric, resolve=True) if c in frame.columns]
    num_na = {c: int(frame[c].isna().sum()) for c in num_cols}
    if any(num_na.values()):
        raise ValueError(
            f"{ds.name}: unexpected NaNs in numeric columns {num_na}. "
            "Neither dataset should have any; investigate before imputing."
        )

    keep = num_cols + cat_cols + ["target"]
    frame = frame[keep]

    report["rows_out"] = int(len(frame))
    report["n_numeric"] = len(num_cols)
    report["n_categorical"] = len(cat_cols)
    report["positive_rate"] = float(frame["target"].mean())
    return frame, report


def write(cfg: DictConfig, frame: pd.DataFrame, report: dict[str, Any]) -> None:
    paths.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(paths.processed(cfg.dataset.name), index=False)
    paths.cleaning_report(cfg.dataset.name).write_text(json.dumps(report, indent=2))

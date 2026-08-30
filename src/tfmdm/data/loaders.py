"""Phase 1.1 -- read the two raw CSVs into a common schema."""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .. import paths

EXPECTED_SHAPE = {"adult": (48842, 15), "taiwan": (30000, 24)}


def load_raw(cfg: DictConfig) -> pd.DataFrame:
    """Load a dataset's raw CSV and normalise its target column to int {0, 1}.

    Taiwan's CSV carries a two-row header -- generic ``X1..X23`` codes followed by
    the real column names -- so its config passes ``header: 1``.
    """
    name = cfg.dataset.name
    path = paths.ROOT / cfg.dataset.raw_path
    kwargs = OmegaConf.to_container(cfg.dataset.get("read_kwargs", {}), resolve=True)
    frame = pd.read_csv(path, **kwargs)  # type: ignore[arg-type]

    expected = EXPECTED_SHAPE.get(name)
    if expected is not None and frame.shape != expected:
        raise ValueError(
            f"{name}: expected raw shape {expected}, got {frame.shape}. "
            "The raw file changed, or the header offset in the dataset config is wrong."
        )

    target = cfg.dataset.target
    if target not in frame.columns:
        raise KeyError(f"{name}: target column {target!r} not in {list(frame.columns)}")

    positive = cfg.dataset.positive_label
    raw_target = frame[target]
    if raw_target.dtype == object:
        y = (raw_target.astype(str).str.strip() == str(positive)).astype("int8")
    else:
        y = (raw_target == positive).astype("int8")

    if y.nunique() != 2:
        raise ValueError(f"{name}: target is not binary after mapping (values={y.unique()})")

    frame = frame.drop(columns=[target])
    frame["target"] = y
    return frame.reset_index(drop=True)


def write_interim(cfg: DictConfig, frame: pd.DataFrame) -> None:
    paths.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(paths.interim(cfg.dataset.name), index=False)

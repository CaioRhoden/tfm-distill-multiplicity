"""Phase 1.3 -- one frozen stratified 60/20/20 partition (decision D1).

Ambiguity and discrepancy are disagreement rates over a *common* set of test points,
so the test set must be identical across all 30 seeds and all arms. Seed-to-seed
variation enters through the bootstrap resample of train (seeds.py), not here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from .. import paths

BALANCE_TOLERANCE_PP = 0.5


@dataclass(frozen=True)
class SplitIndex:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def frames(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (
            frame.iloc[self.train].reset_index(drop=True),
            frame.iloc[self.val].reset_index(drop=True),
            frame.iloc[self.test].reset_index(drop=True),
        )


def make_splits(cfg: DictConfig, frame: pd.DataFrame) -> tuple[SplitIndex, dict]:
    y = frame["target"].to_numpy()
    idx = np.arange(len(frame))
    seed = int(cfg.split.seed)

    test_frac = float(cfg.split.test)
    val_frac = float(cfg.split.val)
    trainval_idx, test_idx = train_test_split(
        idx, test_size=test_frac, stratify=y, random_state=seed, shuffle=True
    )
    # val_frac is expressed against the whole dataset, so rescale it for the sub-split.
    rel_val = val_frac / (1.0 - test_frac)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=rel_val, stratify=y[trainval_idx], random_state=seed, shuffle=True
    )

    split = SplitIndex(np.sort(train_idx), np.sort(val_idx), np.sort(test_idx))
    _assert_disjoint(split, len(frame))
    report = _balance_report(y, split)
    _assert_balanced(report)
    return split, report


def _assert_disjoint(split: SplitIndex, n_rows: int) -> None:
    sets = [set(split.train.tolist()), set(split.val.tolist()), set(split.test.tolist())]
    if sum(len(s) for s in sets) != n_rows or len(set.union(*sets)) != n_rows:
        raise AssertionError("Splits are not a disjoint cover of the dataset")


def _balance_report(y: np.ndarray, split: SplitIndex) -> dict:
    pooled = float(y.mean())
    out = {"pooled_positive_rate": pooled, "n_rows": int(y.size)}
    for name in ("train", "val", "test"):
        part = getattr(split, name)
        out[name] = {
            "n": int(part.size),
            "positive_rate": float(y[part].mean()),
            "delta_pp": float(abs(y[part].mean() - pooled) * 100.0),
        }
    return out


def _assert_balanced(report: dict) -> None:
    bad = {k: report[k]["delta_pp"] for k in ("train", "val", "test")
           if report[k]["delta_pp"] > BALANCE_TOLERANCE_PP}
    if bad:
        raise AssertionError(f"Split class balance drifted beyond {BALANCE_TOLERANCE_PP}pp: {bad}")


def write(cfg: DictConfig, split: SplitIndex, report: dict) -> None:
    payload = {
        "split_seed": int(cfg.split.seed),
        "fractions": {"train": float(cfg.split.train), "val": float(cfg.split.val),
                      "test": float(cfg.split.test)},
        "balance": report,
        "train": split.train.tolist(),
        "val": split.val.tolist(),
        "test": split.test.tolist(),
    }
    paths.splits(cfg.dataset.name).write_text(json.dumps(payload))


def load_splits(dataset: str) -> SplitIndex:
    payload = json.loads(paths.splits(dataset).read_text())
    return SplitIndex(
        np.asarray(payload["train"], dtype=int),
        np.asarray(payload["val"], dtype=int),
        np.asarray(payload["test"], dtype=int),
    )

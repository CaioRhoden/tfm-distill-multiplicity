"""Preprocessing correctness: the split, the encoder fit, and the cleaning rules."""

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from tfmdm.data import clean as clean_mod
from tfmdm.data import features as features_mod
from tfmdm.data import split as split_mod


def _cfg(split_seed: int = 0):
    return OmegaConf.create({
        "split": {"seed": split_seed, "train": 0.6, "val": 0.2, "test": 0.2},
        "dataset": {
            "name": "synthetic",
            "numeric": ["x1", "x2"],
            "categorical": ["c1"],
            "drop_columns": ["junk"],
            "recode": {"c1": {"weird": "other"}},
        },
    })


def _frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(loc=50.0, scale=10.0, size=n),
        "c1": rng.choice(["a", "b", "weird"], size=n),
        "junk": rng.normal(size=n),
        "target": rng.binomial(1, 0.25, size=n).astype("int8"),
    })


def test_splits_are_a_disjoint_cover_with_matched_class_balance():
    frame = _frame()
    split, report = split_mod.make_splits(_cfg(), frame)
    assert len(set(split.train) & set(split.test)) == 0
    assert len(set(split.val) & set(split.test)) == 0
    assert split.train.size + split.val.size + split.test.size == len(frame)
    for part in ("train", "val", "test"):
        assert report[part]["delta_pp"] <= split_mod.BALANCE_TOLERANCE_PP + 1e-9


def test_split_is_deterministic_across_calls():
    frame = _frame()
    first, _ = split_mod.make_splits(_cfg(), frame)
    second, _ = split_mod.make_splits(_cfg(), frame)
    assert np.array_equal(first.test, second.test)


def test_different_split_seeds_give_different_partitions():
    """The outer robustness loop is only meaningful if the replicates actually differ."""
    frame = _frame(n=900)
    a, _ = split_mod.make_splits(_cfg(0), frame)
    b, _ = split_mod.make_splits(_cfg(1), frame)
    assert not np.array_equal(a.test, b.test)
    # ...but each is still a valid, balanced partition of the same rows.
    assert a.test.size == b.test.size
    assert set(a.train) | set(a.val) | set(a.test) == set(b.train) | set(b.val) | set(b.test)


def test_encoder_is_fit_on_train_only():
    """The check has teeth because the test rows are shifted far from the train rows:
    a pooled fit would centre on the shifted mean, a train-only fit will not."""
    frame = _frame(n=600)
    split, _ = split_mod.make_splits(_cfg(), frame)
    frame.loc[split.test, "x2"] += 1000.0

    views = features_mod.build_views(_cfg(), frame, split)
    encoded = views["encoded"]
    train_x2 = encoded.iloc[split.train]["x2"]

    # Standardised training values must be centred on zero, unaffected by the shift.
    assert abs(float(train_x2.mean())) < 1e-6
    assert float(encoded.iloc[split.test]["x2"].mean()) > 10.0


def test_duplicates_are_removed_before_splitting():
    frame = _frame(n=100)
    duplicated = pd.concat([frame, frame.iloc[:10]], ignore_index=True)
    cleaned, report = clean_mod.clean(_cfg(), duplicated)
    assert report["duplicate_rows_dropped"] == 10
    assert report["rows_out"] == report["rows_in"] - 10
    assert not cleaned.duplicated().any()
    assert "junk" not in cleaned.columns


def test_undocumented_categories_are_folded():
    cleaned, report = clean_mod.clean(_cfg(), _frame())
    assert "weird" not in set(cleaned["c1"])
    assert report["recoded_values"]["c1"] > 0


def test_numeric_nans_are_refused_rather_than_imputed():
    frame = _frame()
    frame.loc[0, "x1"] = np.nan
    with pytest.raises(ValueError, match="NaNs in numeric"):
        clean_mod.clean(_cfg(), frame)

"""The weighted-duplication trick must be an exact restatement of soft cross-entropy."""

import numpy as np
import pandas as pd
import pytest

from tfmdm.models.base import soft_cross_entropy
from tfmdm.models.base import expand_soft_targets


def test_expansion_preserves_total_weight():
    x = pd.DataFrame({"a": np.arange(5.0)})
    p = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    _, _, weights = expand_soft_targets(x, p)
    assert weights.sum() == pytest.approx(len(x))


def test_expansion_reproduces_soft_cross_entropy():
    x = pd.DataFrame({"a": np.arange(4.0)})
    p = np.array([0.1, 0.4, 0.6, 0.9])
    q = np.array([0.2, 0.5, 0.5, 0.8])

    _, labels, weights = expand_soft_targets(x, p)
    # The duplicated rows carry the prediction of their source row.
    preds = np.concatenate([q, q])[weights_mask(p)]
    per_row = -(labels * np.log(preds) + (1 - labels) * np.log(1 - preds))
    weighted = float(np.sum(weights * per_row) / len(x))

    assert weighted == pytest.approx(soft_cross_entropy(p, q))


def weights_mask(p: np.ndarray) -> np.ndarray:
    """Mirror the zero-weight pruning that expand_soft_targets applies."""
    return np.concatenate([p, 1.0 - p]) > 1e-8


def test_hard_targets_round_trip_to_log_loss():
    x = pd.DataFrame({"a": np.arange(3.0)})
    p = np.array([0.0, 1.0, 1.0])
    _, labels, weights = expand_soft_targets(x, p)
    # Every zero-weight duplicate is pruned, so a hard target leaves one row each.
    assert len(labels) == 3
    assert np.allclose(weights, 1.0)
    assert np.array_equal(labels, np.array([1, 1, 0]))

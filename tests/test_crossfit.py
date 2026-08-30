"""Cross-fitting is what keeps decision D2 honest, so its guards are tested directly."""

import numpy as np
import pandas as pd
import pytest

from tfmdm.softlabels.crossfit import CrossFitResult, assert_honest, cross_fit, entropy


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (x["a"] + rng.normal(scale=0.5, size=n) > 0).astype(int).to_numpy()
    return x, y


def _memorising_teacher(ctx_x, ctx_y, query_x, seed):
    """Returns the true label for any row it has seen, 0.5 otherwise.

    This is a caricature of an in-context learner, and it is exactly the failure mode
    cross-fitting exists to prevent.
    """
    lookup = {tuple(row): float(label) for row, label in zip(ctx_x.to_numpy(), ctx_y)}
    return np.array([lookup.get(tuple(row), 0.5) for row in query_x.to_numpy()])


def test_every_training_row_receives_an_out_of_fold_label():
    x, y = _data()
    result = cross_fit(_memorising_teacher, x, y, n_folds=5, compute_in_context=False)
    assert not np.isnan(result.oof_probs).any()
    assert set(np.unique(result.fold_ids)) == set(range(5))


def test_cross_fitting_defeats_a_memorising_teacher():
    x, y = _data()
    result = cross_fit(_memorising_teacher, x, y, n_folds=5, compute_in_context=True)
    # Out of fold the teacher has never seen the row, so it falls back to 0.5.
    assert np.allclose(result.oof_probs, 0.5)
    # In context it reproduces the label exactly -- entropy zero.
    assert result.diagnostics["in_context_mean_entropy"] == pytest.approx(0.0, abs=1e-9)
    assert result.diagnostics["entropy_gain"] > 0.0


def test_assert_honest_rejects_labels_that_are_no_better_than_in_context():
    result = CrossFitResult(
        oof_probs=np.array([0.0, 1.0, 0.0, 1.0]),
        fold_ids=np.zeros(4, dtype=int),
        in_context_probs=np.array([0.5, 0.5, 0.5, 0.5]),
        diagnostics={"oof_mean": 0.5, "train_positive_rate": 0.5,
                     "oof_mean_entropy": 0.0, "in_context_mean_entropy": 0.69,
                     "entropy_gain": -0.69},
    )
    with pytest.raises(AssertionError, match="no higher-entropy"):
        assert_honest(result)


def test_assert_honest_rejects_a_miscalibrated_teacher():
    result = CrossFitResult(
        oof_probs=np.full(10, 0.9), fold_ids=np.zeros(10, dtype=int), in_context_probs=None,
        diagnostics={"oof_mean": 0.9, "train_positive_rate": 0.25,
                     "oof_mean_entropy": 0.3},
    )
    with pytest.raises(AssertionError, match="miscalibrated"):
        assert_honest(result)


def test_entropy_is_maximal_at_one_half():
    assert entropy(np.array([0.5]))[0] == pytest.approx(np.log(2))
    assert entropy(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-9)

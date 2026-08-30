import numpy as np

from tfmdm.seeds import stratified_bootstrap_indices


def test_bootstrap_preserves_class_counts():
    y = np.array([0] * 80 + [1] * 20)
    idx = stratified_bootstrap_indices(y, seed=3)
    assert idx.size == y.size
    assert int((y[idx] == 1).sum()) == 20


def test_bootstrap_is_seed_deterministic_and_seed_sensitive():
    y = np.array([0] * 50 + [1] * 50)
    assert np.array_equal(stratified_bootstrap_indices(y, 7), stratified_bootstrap_indices(y, 7))
    assert not np.array_equal(stratified_bootstrap_indices(y, 7),
                              stratified_bootstrap_indices(y, 8))


def test_bootstrap_actually_resamples():
    y = np.array([0] * 50 + [1] * 50)
    idx = stratified_bootstrap_indices(y, seed=0)
    assert np.unique(idx).size < y.size  # with replacement, so some rows repeat

import numpy as np
import pytest

from tfmdm.metrics import bootstrap as boot


def test_holm_is_step_down_and_order_preserving():
    # With m=3 and alpha=0.05 the thresholds are 0.0167, 0.025, 0.05.
    assert boot.holm([0.001, 0.02, 0.9]) == [True, True, False]
    # Step-down operates on sorted p-value order, not array position: 0.02 is the
    # second-smallest here and still clears its rank-2 threshold (0.025).
    assert boot.holm([0.001, 0.30, 0.02]) == [True, False, True]


def test_holm_with_a_single_test_reduces_to_alpha():
    assert boot.holm([0.049]) == [True]
    assert boot.holm([0.051]) == [False]


def test_percentile_ci_brackets_a_known_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=2.0, scale=1.0, size=500)
    interval = boot.percentile_ci(lambda idx: float(values[idx].mean()), values.size,
                                  n_boot=500, seed=0)
    assert interval.low < 2.0 < interval.high
    assert interval.point == pytest.approx(values.mean())


def test_paired_bootstrap_detects_a_real_difference_and_ignores_a_null_one():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 1.0, size=400)
    b = a - 0.8  # a constant, obvious shift
    interval, p_value = boot.paired_bootstrap(
        lambda idx: float(a[idx].mean()), lambda idx: float(b[idx].mean()),
        a.size, n_boot=400, seed=0,
    )
    assert interval.excludes_zero()
    assert p_value < 0.05

    same, p_same = boot.paired_bootstrap(
        lambda idx: float(a[idx].mean()), lambda idx: float(a[idx].mean()),
        a.size, n_boot=200, seed=0,
    )
    assert same.point == pytest.approx(0.0)
    assert p_same == pytest.approx(1.0)

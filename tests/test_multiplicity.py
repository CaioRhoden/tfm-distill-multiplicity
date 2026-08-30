"""Known-answer tests for the multiplicity metrics (plan step 5.1).

These are the cases where the correct value is knowable without running anything. A
metric implementation that misses them will not be caught by any later check, because
every downstream number would simply be wrong together.
"""

import numpy as np
import pytest

from tfmdm.metrics import multiplicity as mult


def test_identical_models_have_zero_multiplicity():
    probs = np.tile(np.linspace(0.01, 0.99, 100)[:, None], (1, 30))
    result = mult.multiplicity(probs)
    assert result.ambiguity == 0.0
    assert result.discrepancy == 0.0
    assert result.max_pairwise_discrepancy == 0.0


def test_fully_opposed_models_saturate():
    reference = np.full((50, 1), 0.9)
    opposed = np.full((50, 1), 0.1)
    probs = np.hstack([reference, opposed])
    result = mult.multiplicity(probs)
    assert result.ambiguity == 1.0
    assert result.discrepancy == 1.0
    assert result.max_pairwise_discrepancy == 1.0


def test_ambiguity_counts_points_not_disagreements():
    # Point 0: models 1 and 2 both flip. Point 1: only model 1 flips. Point 2: none.
    probs = np.array([
        [0.9, 0.1, 0.1],
        [0.9, 0.1, 0.9],
        [0.9, 0.9, 0.9],
    ])
    result = mult.multiplicity(probs)
    assert result.ambiguity == pytest.approx(2 / 3)
    # Model 1 disagrees on 2 of 3 points; model 2 on 1 of 3. Discrepancy takes the max.
    assert result.discrepancy == pytest.approx(2 / 3)


def test_discrepancy_is_relative_to_the_reference_model():
    probs = np.array([[0.9, 0.9, 0.1], [0.9, 0.9, 0.1]])
    assert mult.multiplicity(probs, reference_index=0).discrepancy == pytest.approx(1.0)
    # Against model 2 as reference, models 0 and 1 both disagree everywhere.
    assert mult.multiplicity(probs, reference_index=2).discrepancy == pytest.approx(1.0)


def test_jackknife_matches_bruteforce_recomputation():
    rng = np.random.default_rng(0)
    probs = rng.uniform(size=(40, 6))
    disagree = mult.disagreement_matrix(probs)

    fast_amb = mult.ambiguity_jackknife(disagree)
    fast_disc = mult.discrepancy_jackknife(disagree)
    for i in range(probs.shape[0]):
        kept = np.delete(disagree, i, axis=0)
        assert fast_amb[i] == pytest.approx(mult.ambiguity(kept))
        assert fast_disc[i] == pytest.approx(mult.discrepancy(kept))


def test_single_model_is_rejected():
    with pytest.raises(ValueError):
        mult.multiplicity(np.full((10, 1), 0.5))


def test_threshold_curve_covers_the_requested_grid():
    rng = np.random.default_rng(1)
    curve = mult.threshold_curve(rng.uniform(size=(30, 4)), np.linspace(0.1, 0.9, 9))
    assert curve["ambiguity"].shape == (9,)
    assert np.all((curve["ambiguity"] >= 0) & (curve["ambiguity"] <= 1))

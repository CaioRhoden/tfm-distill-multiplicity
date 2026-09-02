"""Known-answer tests for the explanation-multiplicity metrics (plan step 5.4).

Same principle as ``test_multiplicity``: every case here has an answer that can be
worked out on paper, because a metric that is silently wrong produces a full results
table that looks entirely plausible.
"""

import numpy as np
import pytest

from tfmdm.metrics import explanation as expl


def _stack(*models: np.ndarray) -> np.ndarray:
    return np.stack(models, axis=0)


def test_identical_models_show_no_explanation_multiplicity():
    rng = np.random.default_rng(0)
    one = rng.normal(size=(40, 6))
    contributions = _stack(one, one.copy(), one.copy())

    assert expl.global_explanation_multiplicity(np.abs(contributions).mean(axis=1))[
        "mean_spearman_correlation"
    ] == pytest.approx(1.0)
    assert expl.functional_multiplicity(contributions)["mean_normalized_fed"] == pytest.approx(0.0)
    assert expl.local_explanation_multiplicity(contributions)[
        "mean_local_attribution_discrepancy"
    ] == pytest.approx(0.0)
    assert expl.explanation_agreement(contributions, k=3)["mean_jaccard_top3"] == pytest.approx(1.0)


def test_reversed_importance_ranking_gives_spearman_minus_one():
    importances = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    result = expl.global_explanation_multiplicity(importances)
    assert result["mean_spearman_correlation"] == pytest.approx(-1.0)
    assert result["n_importance_pairs"] == 1


def test_fed_is_normalised_by_the_wider_term_range():
    # One term, ramping 0..1 in model A and 0..2 in model B. The mean gap is the mean
    # of |t - 2t| = t over t in [0, 1], i.e. 0.5; the wider range is 2, so FED = 0.25.
    ramp = np.linspace(0.0, 1.0, 101)[:, None]
    contributions = _stack(ramp, 2 * ramp)
    assert expl.functional_multiplicity(contributions)["mean_normalized_fed"] == pytest.approx(
        0.25, abs=1e-3
    )


def test_fed_ignores_terms_that_are_flat_in_both_models():
    ramp = np.linspace(0.0, 1.0, 51)
    flat = np.zeros(51)
    a = np.column_stack([ramp, flat])
    b = np.column_stack([ramp, flat])
    # The live term agrees exactly, so a flat term counted as agreement would be
    # invisible here -- what this pins is that it is not counted as *disagreement*.
    assert expl.functional_multiplicity(_stack(a, b))["mean_normalized_fed"] == pytest.approx(0.0)


def test_fed_breaks_out_by_term_order():
    ramp = np.linspace(0.0, 1.0, 101)
    a = np.column_stack([ramp, ramp])
    b = np.column_stack([ramp, 2 * ramp])
    result = expl.functional_multiplicity(_stack(a, b), orders=np.array([1, 2]))
    assert result["mean_fed_order1"] == pytest.approx(0.0)
    assert result["mean_fed_order2"] == pytest.approx(0.25, abs=1e-3)


def test_lad_is_the_mean_absolute_attribution_gap():
    a = np.zeros((10, 3))
    b = np.full((10, 3), 0.4)
    result = expl.local_explanation_multiplicity(_stack(a, b))
    assert result["mean_local_attribution_discrepancy"] == pytest.approx(0.4)


def test_disjoint_top_k_sets_give_zero_jaccard():
    # Model A always attributes to terms 0-1, model B always to terms 2-3.
    a = np.tile([3.0, 2.0, 0.1, 0.0], (20, 1))
    b = np.tile([0.0, 0.1, 2.0, 3.0], (20, 1))
    assert expl.explanation_agreement(_stack(a, b), k=2)["mean_jaccard_top2"] == pytest.approx(0.0)


def test_partially_overlapping_top_k_sets():
    # Top-2 is {0, 1} for A and {1, 2} for B: one shared of three distinct -> 1/3.
    a = np.tile([3.0, 2.0, 1.0], (5, 1))
    b = np.tile([1.0, 2.0, 3.0], (5, 1))
    assert expl.explanation_agreement(_stack(a, b), k=2)["mean_jaccard_top2"] == pytest.approx(
        1 / 3
    )


def test_agreement_uses_absolute_attribution_not_signed():
    # The strongest attribution in B is negative; ignoring the sign would rank it last.
    a = np.tile([5.0, 1.0, 0.0], (5, 1))
    b = np.tile([-5.0, 1.0, 0.0], (5, 1))
    assert expl.explanation_agreement(_stack(a, b), k=1)["mean_jaccard_top1"] == pytest.approx(1.0)


def test_k_is_capped_at_the_number_of_terms():
    contributions = np.random.default_rng(1).normal(size=(3, 10, 2))
    result = expl.explanation_agreement(contributions, k=5)
    assert result["mean_jaccard_top2"] == pytest.approx(1.0)  # every term is in the top 2


def test_centering_removes_a_constant_offset_but_not_shape():
    ramp = np.linspace(0.0, 1.0, 20)[:, None]
    contributions = _stack(ramp, ramp + 7.0)
    centred = expl.center_contributions(contributions)
    assert expl.local_explanation_multiplicity(centred)[
        "mean_local_attribution_discrepancy"
    ] == pytest.approx(0.0)
    assert expl.local_explanation_multiplicity(contributions)[
        "mean_local_attribution_discrepancy"
    ] == pytest.approx(7.0)


def test_term_set_agreement_counts_selection_not_shape():
    names = [["a", "b", "c"], ["a", "b", "d"]]
    result = expl.term_set_agreement(names)
    assert result["mean_term_set_jaccard"] == pytest.approx(0.5)  # 2 shared of 4 distinct
    assert result["n_terms_union"] == 4
    assert result["mean_n_terms"] == pytest.approx(3.0)


def test_a_single_model_is_rejected():
    with pytest.raises(ValueError):
        expl.local_explanation_multiplicity(np.zeros((1, 5, 3)))


def test_centering_rejects_a_non_tensor():
    with pytest.raises(ValueError):
        expl.center_contributions(np.zeros((5, 3)))

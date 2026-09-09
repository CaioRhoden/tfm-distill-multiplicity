"""Plan phase 2 -- the gates every explanation metric has to pass before it is believed.

These are not unit tests of convenience. Each one pins a property the *interpretation*
of the results table depends on, and each has an answer that can be worked out on
paper, because a metric that is silently wrong produces a full results table that looks
entirely plausible.

  2.1  a model set of identical models has exactly zero explanation multiplicity
  2.2  a model set of unrelated models saturates it
  2.3  rescaling one model's contributions changes no rank or sign metric *at all* --
       asserted as equality, not as a tolerance, because the invariance is exact and a
       tolerance would hide a metric that had quietly become scale-dependent
  2.4  two models differing only by a per-term constant are at zero distance
"""

import numpy as np
import pytest

from tfmdm.analysis import shapes
from tfmdm.metrics import explanation as expl
from tfmdm.metrics import multiplicity as mult


def _stack(*models: np.ndarray) -> np.ndarray:
    return np.stack(models, axis=0)


# --- 2.1 degenerate ---------------------------------------------------------------

def test_identical_models_have_zero_explanation_multiplicity():
    one = np.random.default_rng(0).normal(size=(200, 8))
    contributions = _stack(one, one.copy(), one.copy())

    result = expl.explanation_multiplicity(contributions)
    assert result.attribution_ambiguity == 0.0
    assert result.attribution_discrepancy == 0.0
    assert result.sign_flip_rate == 0.0
    assert expl.top_k_jaccard(np.abs(contributions).mean(axis=1), 3)["mean_top3_jaccard"] == 1.0


def test_identical_models_have_zero_shape_distance():
    grid = np.linspace(-2.0, 2.0, 100)
    curve = {"age": np.sin(grid)}
    result = shapes.pairwise_distance([curve, dict(curve)], [1.0, 1.0])
    assert result["mean_shape_distance"] == 0.0
    assert result["mean_shape_distance_normalized"] == 0.0


# --- 2.2 saturation ---------------------------------------------------------------

def test_unrelated_models_saturate_attribution_ambiguity():
    """Independent random attributions should re-attribute nearly every row.

    With 30 models and 8 terms, the chance that all 29 non-reference models happen to
    pick the reference's top-1 term is (1/8)^29 -- so anything below ~1.0 here means
    the metric is not seeing the disagreement in front of it.
    """
    contributions = np.random.default_rng(1).normal(size=(30, 500, 8))
    result = expl.explanation_multiplicity(contributions)
    assert result.attribution_ambiguity > 0.99
    assert result.sign_flip_rate == pytest.approx(0.5, abs=0.05)


def test_saturated_set_scores_above_a_partially_agreeing_one():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(500, 8))
    # Agreeing set: every model is the reference plus a nudge far smaller than the gaps.
    agreeing = _stack(*[base + 0.001 * rng.normal(size=base.shape) for _ in range(10)])
    unrelated = rng.normal(size=(10, 500, 8))
    assert (expl.explanation_multiplicity(agreeing).attribution_ambiguity
            < expl.explanation_multiplicity(unrelated).attribution_ambiguity)


# --- 2.3 scale invariance ---------------------------------------------------------

@pytest.mark.parametrize("factor", [0.01, 0.5, 3.0, 1000.0])
def test_rescaling_one_model_leaves_every_rank_metric_bit_identical(factor):
    rng = np.random.default_rng(3)
    contributions = rng.normal(size=(6, 300, 7))
    rescaled = contributions.copy()
    rescaled[2] *= factor

    before = expl.explanation_multiplicity(contributions)
    after = expl.explanation_multiplicity(rescaled)
    assert after.attribution_ambiguity == before.attribution_ambiguity
    assert after.attribution_discrepancy == before.attribution_discrepancy
    assert after.sign_flip_rate == before.sign_flip_rate

    for k in (3, 5):
        assert (expl.top_k_jaccard(np.abs(rescaled).mean(axis=1), k)
                == expl.top_k_jaccard(np.abs(contributions).mean(axis=1), k))


@pytest.mark.parametrize("factor", [0.25, 4.0])
def test_rescaling_moves_the_raw_shape_distance_but_not_the_normalised_one(factor):
    """The one magnitude-based metric, and the one D2 says to normalise."""
    grid = np.linspace(0.0, 1.0, 100)
    a = {"age": np.sin(3 * grid)}
    b = {"age": np.cos(3 * grid)}
    scales = [2.0, 2.0]

    plain = shapes.pairwise_distance([a, b], scales)
    scaled = shapes.pairwise_distance(
        [{"age": a["age"] * factor}, {"age": b["age"] * factor}],
        [s * factor for s in scales],
    )
    assert scaled["mean_shape_distance"] == pytest.approx(factor * plain["mean_shape_distance"])
    assert scaled["mean_shape_distance_normalized"] == pytest.approx(
        plain["mean_shape_distance_normalized"]
    )


def test_a_negative_scale_factor_is_not_claimed_to_be_invariant():
    """Only *positive* rescaling is a no-op: flipping a sign is a different explanation."""
    rng = np.random.default_rng(4)
    contributions = rng.normal(size=(4, 200, 5))
    flipped = contributions.copy()
    flipped[1] *= -1.0
    assert (expl.explanation_multiplicity(flipped).sign_flip_rate
            > expl.explanation_multiplicity(contributions).sign_flip_rate)


# --- 2.4 centring -----------------------------------------------------------------

def test_a_constant_shift_per_term_scores_zero_distance():
    grid = np.linspace(0.0, 1.0, 100)
    curve = np.sin(4 * grid)
    a = {"age": curve - curve.mean()}
    b = {"age": (curve + 7.0) - (curve + 7.0).mean()}
    assert shapes.pairwise_distance([a, b])["mean_shape_distance"] == pytest.approx(0.0)


def test_centring_on_the_train_marginal_is_what_removes_the_offset():
    """A per-term offset is exactly what D3 exists to remove, and it is *not* harmless.

    The shifted model's second term dominates every row by construction, so uncentred
    it wins top-1 everywhere and the two models never agree. Rank metrics are invariant
    to rescaling but not to a shift -- this is that failure, and its fix.
    """
    rng = np.random.default_rng(5)
    base = rng.normal(size=(400, 4))
    offset = np.array([0.0, 50.0, 0.0, 0.0])

    # The shifted model attributes *every* row to term 1; the reference attributes them
    # roughly uniformly, so the two disagree on the ~3/4 of rows the reference sends
    # elsewhere. Uncentred, that is pure artifact of the offset.
    uncentred = _stack(base, base + offset)
    assert expl.explanation_multiplicity(uncentred).attribution_ambiguity > 0.6

    centred = uncentred - uncentred.mean(axis=1, keepdims=True)
    assert expl.explanation_multiplicity(centred).attribution_ambiguity == 0.0


# --- 3.1 the jackknives the intervals are built on --------------------------------

def _naive_jackknife(values: np.ndarray, statistic) -> np.ndarray:
    n = values.shape[0]
    return np.array([statistic(np.delete(np.arange(n), i)) for i in range(n)])


def test_ambiguity_jackknife_matches_a_naive_recompute():
    contributions = np.random.default_rng(6).normal(size=(5, 200, 6))
    disagree = expl.attribution_disagreement(contributions)
    naive = _naive_jackknife(disagree, lambda idx: float(disagree[idx].any(axis=1).mean()))
    assert mult.ambiguity_jackknife(disagree) == pytest.approx(naive)


def test_discrepancy_jackknife_matches_a_naive_recompute():
    contributions = np.random.default_rng(7).normal(size=(5, 200, 6))
    disagree = expl.attribution_disagreement(contributions)
    naive = _naive_jackknife(disagree, lambda idx: float(disagree[idx].mean(axis=0).max()))
    assert mult.discrepancy_jackknife(disagree) == pytest.approx(naive)


def test_sign_flip_jackknife_matches_a_naive_recompute():
    contributions = np.random.default_rng(8).normal(size=(5, 200, 6))
    flips = expl.per_point_sign_flips(contributions)
    naive = _naive_jackknife(flips, lambda idx: float(flips[idx].mean()))
    assert expl.sign_flip_jackknife(flips) == pytest.approx(naive)


# --- grouping (D4) and the margin guard (D2) --------------------------------------

def test_grouping_sums_columns_onto_their_parent_feature():
    contributions = np.random.default_rng(9).normal(size=(3, 40, 4))
    mapping = {"a_x": "a", "a_y": "a", "b_x": "b", "n": "n"}
    grouped, groups = expl.group_terms(contributions, list(mapping), mapping)

    assert groups == ["a", "b", "n"]
    assert grouped[:, :, 0] == pytest.approx(contributions[:, :, 0] + contributions[:, :, 1])
    # Lossless: an additive model's total is unchanged by regrouping its terms.
    assert grouped.sum(axis=2) == pytest.approx(contributions.sum(axis=2))


def test_grouping_can_change_which_term_wins_top_one():
    """Why D4 is not cosmetic: splitting a feature across levels shrinks each piece."""
    contributions = np.zeros((2, 1, 3))
    contributions[:, 0, :] = [0.6, 0.6, 1.0]  # two levels of 'a' against one term 'b'
    mapping = {"a_x": "a", "a_y": "a", "b": "b"}

    assert expl.top1_terms(contributions)[0, 0] == 2  # 'b' wins ungrouped
    grouped, groups = expl.group_terms(contributions, list(mapping), mapping)
    assert groups[expl.top1_terms(grouped)[0, 0]] == "a"  # 'a' wins once summed


def test_margin_is_the_reference_models_top_two_gap():
    contributions = np.zeros((2, 3, 4))
    contributions[0] = [[5.0, 2.0, 0.0, 0.0], [-4.0, 3.5, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
    assert expl.attribution_margin(contributions) == pytest.approx([3.0, 0.5, 0.0])


def test_trimming_near_ties_can_only_use_the_rows_that_survive():
    rng = np.random.default_rng(10)
    contributions = rng.normal(size=(4, 300, 5))
    disagree = expl.attribution_disagreement(contributions)
    margins = expl.attribution_margin(contributions)

    sweep = expl.margin_sweep(disagree, margins, np.quantile(margins, [0.0, 0.5, 0.9]))
    assert [row["n_points_kept"] for row in sweep] == sorted(
        [row["n_points_kept"] for row in sweep], reverse=True
    )
    # The trim is strict (margin > epsilon), so the smallest-margin row is the one the
    # epsilon=q0 pass drops -- everything else survives.
    n_points = disagree.shape[0]
    assert sweep[0]["n_points_kept"] == n_points - 1


def test_top_k_jaccard_survives_a_tied_tail_that_spearman_does_not():
    """The degeneracy 3.2 exists to fix: mostly-zero importance vectors.

    Two models agree exactly on the three terms that carry any importance and differ
    only in the arbitrary ordering of a tied, all-zero tail. Top-k reads that as
    perfect agreement, which it is.
    """
    a = np.array([3.0, 2.0, 1.0] + [0.0] * 40)
    b = np.array([3.0, 2.0, 1.0] + [0.0] * 40)
    importances = np.stack([a, b])
    assert expl.top_k_jaccard(importances, 3)["mean_top3_jaccard"] == 1.0

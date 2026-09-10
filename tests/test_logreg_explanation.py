"""Plan phase 2 for the linear family (plans/logreg.md).

The linear model is the one family whose additive decomposition is exact algebra
rather than an extraction, so its gates are stated as equalities wherever the identity
is exact. That makes it the cheapest possible check on the *harness* -- column order,
centring, grouping -- since any residual seen here cannot be the model's fault.

  2.1  a set of identically-fitted models has exactly zero explanation multiplicity
  2.3  contributions plus intercept reproduce the model's own logit
  2.4  grouping one-hot columns back to their parent feature is lossless
"""

import numpy as np
import pandas as pd
import pytest

from tfmdm.metrics import explanation as expl
from tfmdm.models.explain import term_contributions
from tfmdm.models.logreg import LogRegModel


def _frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A standardised-numeric + one-hot frame, the shape of the ``encoded`` view."""
    rng = np.random.default_rng(seed)
    level = rng.integers(0, 3, size=n)
    return pd.DataFrame({
        "age": rng.normal(size=n),
        "hours": rng.normal(size=n),
        "workclass_a": (level == 0).astype(float),
        "workclass_b": (level == 1).astype(float),
        "workclass_c": (level == 2).astype(float),
    })


def _labels(x: pd.DataFrame, seed: int = 0) -> np.ndarray:
    logit = 1.5 * x["age"] - 0.8 * x["hours"] + 2.0 * x["workclass_a"]
    noise = np.random.default_rng(seed).normal(scale=0.5, size=len(x))
    return (logit + noise > 0).astype(float).to_numpy()


def _fit(x: pd.DataFrame, y: np.ndarray, seed: int = 0, C: float = 1.0, **params) -> LogRegModel:
    return LogRegModel(seed=seed, C=C, max_iter=5000, **params).fit(x, y, x, y, arm="hard")


# --- 2.3 the decomposition is the model's own -------------------------------------

def test_contributions_reconstruct_the_logit_exactly():
    """Stated as a near-exact equality, not the 1e-5 the NAM needs.

    ``coef_ @ x + intercept_`` *is* what sklearn computes to predict, so the only way
    this drifts is float64 rounding -- or a genuine bug in column alignment, which is
    what the test is really guarding.
    """
    x, = (_frame(),)
    model = _fit(x, _labels(x))

    terms = term_contributions(model, x)
    probabilities = model.predict_proba(x)
    recovered = np.log(probabilities / (1 - probabilities))

    assert terms.values.shape == (len(x), len(x.columns))
    assert np.abs(terms.logits() - recovered).max() < 1e-9


def test_contributions_follow_the_fitted_column_order_not_the_frames():
    """A reordered frame must not pair a coefficient with the wrong feature."""
    x = _frame()
    model = _fit(x, _labels(x))

    shuffled = x[list(reversed(x.columns))]
    terms = term_contributions(model, shuffled)

    assert terms.names == model.columns
    assert np.abs(terms.values - term_contributions(model, x).values).max() == 0.0


def test_centring_preserves_the_logit():
    """D3 folds the removed constants into the intercept, so the identity survives."""
    x = _frame()
    model = _fit(x, _labels(x))
    terms = term_contributions(model, x)

    centred = terms.centred(terms.values.mean(axis=0))

    assert np.abs(centred.logits() - terms.logits()).max() < 1e-9
    assert np.abs(centred.values.mean(axis=0)).max() < 1e-12


# --- 2.1 degenerate ---------------------------------------------------------------

def test_identically_fitted_models_explain_identically():
    """Stronger here than for the other families.

    A logistic regression is deterministic given its data, so without the stratified
    bootstrap *every* seed must collapse onto the same explanation. Anything above zero
    would mean the seeding protocol is perturbing something it should not be.
    """
    x = _frame()
    y = _labels(x)
    contributions = np.stack(
        [term_contributions(_fit(x, y, seed=seed), x).values for seed in range(5)]
    )

    result = expl.explanation_multiplicity(contributions)
    assert result.attribution_ambiguity == 0.0
    assert result.attribution_discrepancy == 0.0
    assert result.sign_flip_rate == 0.0


def test_bootstrapped_models_do_disagree():
    """The companion to the check above: perturb the data and the metric must move.

    Without this, a metric hard-wired to zero would pass the degenerate gate.
    """
    x = _frame()
    y = _labels(x)
    rng = np.random.default_rng(3)
    contributions = np.stack([
        term_contributions(_fit(x.iloc[rows].reset_index(drop=True), y[rows], seed=seed), x).values
        for seed, rows in enumerate(rng.integers(0, len(x), size=(5, len(x))))
    ])

    assert expl.explanation_multiplicity(contributions).attribution_discrepancy > 0.0


# --- 2.4 grouping -----------------------------------------------------------------

def test_grouping_one_hot_columns_is_lossless():
    """Summing a parent's levels cannot change the total contribution of a row.

    Exactly one level is active per row, but the *inactive* ones still contribute
    ``coef * 0 = 0`` before centring and a non-zero constant after it -- so this is a
    real check, not a tautology, and it is what makes the grouped tensor still
    reconstruct the prediction.
    """
    x = _frame()
    model = _fit(x, _labels(x))
    terms = term_contributions(model, x).centred(term_contributions(model, x).values.mean(axis=0))

    mapping = {"age": "age", "hours": "hours",
               "workclass_a": "workclass", "workclass_b": "workclass",
               "workclass_c": "workclass"}
    grouped, names = expl.group_terms(terms.values[None, :, :], terms.names, mapping)

    assert names == ["age", "hours", "workclass"]
    assert np.abs(grouped[0].sum(axis=1) - terms.values.sum(axis=1)).max() < 1e-12


# --- D-IMP ------------------------------------------------------------------------

def test_importance_is_data_weighted_not_the_bare_coefficient():
    """A rare level carries a large coefficient and almost no contribution.

    Ranking by ``|coef|`` would put it on top for every seed and read as stability;
    ranking by mean absolute contribution puts it where its influence actually is.

    ``C`` is loosened here so the penalty does not itself shrink the rare level's
    coefficient below the common one -- that would make the two rankings agree for a
    reason that has nothing to do with the effect being demonstrated.
    """
    rng = np.random.default_rng(5)
    n = 2000
    rare = (rng.random(n) < 0.01).astype(float)
    x = pd.DataFrame({"common": rng.normal(size=n), "rare": rare})
    y = ((6.0 * x["rare"] + 0.3 * x["common"] + rng.normal(scale=0.3, size=n)) > 0)

    model = _fit(x, y.astype(float).to_numpy(), C=10.0)
    bare = model.feature_importances()
    weighted = model.feature_importances_on(x)

    assert bare["rare"] > bare["common"]
    assert weighted["rare"] < weighted["common"]


# --- convergence guard ------------------------------------------------------------

def test_unconverged_fit_is_an_error_not_a_warning():
    """An lbfgs stopped at max_iter differs by seed for no reason the study measures."""
    x = _frame()
    with pytest.raises(AssertionError, match="did not converge"):
        LogRegModel(seed=0, C=1e6, max_iter=1, tol=1e-12).fit(
            x, _labels(x), x, _labels(x), arm="hard"
        )

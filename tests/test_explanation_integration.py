"""Gates 2.1 and 1.1 run against the artifacts on disk, not against synthetic arrays.

``test_explanation_gates`` pins the metrics' behaviour on arrays that were constructed
to have a known answer. That is necessary and not sufficient: it says nothing about
whether the *pipeline* -- reload, reconstruct, centre, group, align -- preserves those
properties on real fitted models. These tests close that gap, and skip rather than fail
when the sweep has not been run, so a fresh clone still gets a green suite.
"""

import joblib
import numpy as np
import pytest

from tfmdm import paths
from tfmdm.analysis import explanations as expl_stage
from tfmdm.analysis import grouping, shapes
from tfmdm.metrics import explanation as expl
from tfmdm.models.explain import reconstruction_error, term_contributions
from tfmdm.stages import train as train_stage

DATASET, SPLIT_SEED, SEED = "adult", 0, 0
FAMILIES = ("ebm", "nam")
PROBE_ROWS = 300


def _artifact(model: str, arm: str = "hard"):
    path = paths.model_artifact(DATASET, model, arm, SEED, SPLIT_SEED)
    if not path.exists():
        pytest.skip(f"{path} not present; run the sweep first")
    return joblib.load(path)


def _context(model: str, arm: str = "hard"):
    if not paths.splits(DATASET, SPLIT_SEED).exists():
        pytest.skip("split artifacts not present")
    return train_stage.prepare(DATASET, model, arm, SPLIT_SEED)


# --- 1.1 the decomposition is the model's -----------------------------------------

@pytest.mark.parametrize("model", FAMILIES)
def test_contributions_reconstruct_the_models_own_prediction(model):
    learner = _artifact(model)
    x = _context(model).x_test.head(PROBE_ROWS)
    error = reconstruction_error(learner, x, term_contributions(learner, x))
    assert error["max_prob_error"] <= expl_stage.RESIDUAL_TOLERANCE


@pytest.mark.parametrize("model", FAMILIES)
def test_centring_on_the_train_marginal_preserves_the_identity(model):
    """D3 removes a constant per term and folds it into the intercept -- exactly.

    If the fold-in were dropped, the centred decomposition would no longer reproduce
    the model, and every metric would be measuring a model nobody trained.
    """
    learner = _artifact(model)
    ctx = _context(model)
    x = ctx.x_test.head(PROBE_ROWS)

    train_means = term_contributions(learner, ctx.x_train).values.mean(axis=0)
    centred = term_contributions(learner, x).centred(train_means)
    error = reconstruction_error(learner, x, centred)
    assert error["max_prob_error"] <= expl_stage.RESIDUAL_TOLERANCE


def test_the_nam_carries_a_larger_pre_centring_offset_than_the_ebm():
    """The asymmetry D3 exists for, measured rather than asserted.

    ``interpret`` already centres an EBM's graphs on the train set, so its per-term
    means are near zero. A NAM centres nothing, and its one-hot level nets emit f(0) on
    every row where that level is absent -- so the constant D3 removes is orders of
    magnitude larger. That is why the step is not optional for the NAM.
    """
    offsets = {}
    for model in FAMILIES:
        learner = _artifact(model)
        ctx = _context(model)
        offsets[model] = float(
            abs(term_contributions(learner, ctx.x_train).values.mean(axis=0).sum())
        )
    assert offsets["nam"] > 10 * offsets["ebm"]


# --- 2.1 the degenerate model set --------------------------------------------------

@pytest.mark.parametrize("model", FAMILIES)
def test_a_model_set_of_one_repeated_model_scores_exactly_zero(model):
    """Gate 2.1, through the real pipeline: same seed, same model, no multiplicity.

    Exact equality, not a tolerance. Every metric here is a count or a max of counts
    over discrete comparisons, so a duplicated model cannot produce a small non-zero
    value -- if it does, something in the alignment or grouping is not deterministic.
    """
    learner = _artifact(model)
    ctx = _context(model)
    x = ctx.x_test.head(PROBE_ROWS)

    train_means = term_contributions(learner, ctx.x_train).values.mean(axis=0)
    terms = term_contributions(learner, x).centred(train_means)
    mapping = grouping.group_map(DATASET, model, SPLIT_SEED, terms.names)
    grouped, groups = expl.group_terms(terms.values[None, :, :], terms.names, mapping)

    contributions = np.repeat(grouped, 3, axis=0)
    result = expl.explanation_multiplicity(contributions)
    assert result.attribution_ambiguity == 0.0
    assert result.attribution_discrepancy == 0.0
    assert result.sign_flip_rate == 0.0
    for k in (3, 5):
        assert expl.top_k_jaccard(np.abs(contributions).mean(axis=1), k)[
            f"mean_top{k}_jaccard"
        ] == 1.0
    assert len(groups) == contributions.shape[2]


@pytest.mark.parametrize("model", FAMILIES)
def test_a_repeated_model_has_zero_shape_distance(model):
    learner = _artifact(model)
    ctx = _context(model)
    mapping = (grouping.group_map(DATASET, model, SPLIT_SEED, list(ctx.x_train.columns))
               if model == "nam" else {})
    columns = shapes.numeric_columns(ctx.x_train, model, mapping)
    grids = {c: shapes.quantile_grid(ctx.x_train[c].to_numpy()) for c in columns}

    curve = shapes.curves(learner, ctx.x_train, columns, grids)
    assert shapes.pairwise_distance([curve, dict(curve)])["mean_shape_distance"] == 0.0


# --- D4: the grouping map matches the encoder that produced the columns ------------

def test_every_nam_column_maps_to_exactly_one_parent_feature():
    if not paths.transformer(DATASET, SPLIT_SEED).exists():
        pytest.skip("encoder artifact not present")
    ctx = _context("nam")
    mapping = grouping.encoded_groups(DATASET, SPLIT_SEED)
    assert set(mapping) == set(ctx.x_train.columns)
    # Every parent has at least one column, and no column belongs to two parents.
    assert len(set(mapping.values())) < len(mapping)


def test_grouping_a_nam_cell_is_lossless():
    """Summing a parent's level nets must not change the total the model predicts."""
    learner = _artifact("nam")
    ctx = _context("nam")
    x = ctx.x_test.head(PROBE_ROWS)
    terms = term_contributions(learner, x)
    mapping = grouping.group_map(DATASET, "nam", SPLIT_SEED, terms.names)
    grouped, _ = expl.group_terms(terms.values[None, :, :], terms.names, mapping)
    assert grouped[0].sum(axis=1) == pytest.approx(terms.values.sum(axis=1))


# --- the artifacts must outlive the machine that produced them --------------------

def test_a_cuda_trained_nam_loads_where_cuda_is_unavailable(monkeypatch):
    """The reload risk named in the plan's assumptions, reproduced rather than assumed.

    A NAM is fit on a GPU node and re-scored wherever there is capacity. torch records
    the device each storage lived on and refuses to restore a CUDA storage when
    ``torch.cuda.is_available()`` is False -- which is true both on a CPU-only node and
    on a node whose driver is too old for the installed torch build. Faking that
    condition is the only way to test the CPU path on a machine that has a working GPU.
    """
    import torch

    from tfmdm.models.io import load_learner

    path = paths.model_artifact(DATASET, "nam", "hard", SEED, SPLIT_SEED)
    if not path.exists():
        pytest.skip(f"{path} not present; run the sweep first")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    learner = load_learner(path)

    # Loading is only half of it: the model must then actually run. ``models.explain``
    # re-resolves the pickled device attribute, which still says whatever it trained on.
    x = _context("nam").x_test.head(50)
    terms = term_contributions(learner, x)
    assert terms.values.shape[0] == len(x)
    assert reconstruction_error(learner, x, terms)["max_prob_error"] <= (
        expl_stage.RESIDUAL_TOLERANCE
    )


def test_the_cpu_mapping_does_not_leak_out_of_the_load(monkeypatch):
    """torch's deserialisation must be left exactly as it was found, even on failure."""
    import torch

    from tfmdm.models.io import _storages_on_cpu

    original = torch.storage._load_from_bytes
    with pytest.raises(RuntimeError), _storages_on_cpu():
        assert torch.storage._load_from_bytes is not original
        raise RuntimeError("boom")
    assert torch.storage._load_from_bytes is original

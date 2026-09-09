"""Uniform access to a fitted model's *terms*, so EBM and NAM can be compared.

An explanation-multiplicity metric needs three things from a trained model, and both
learners here can supply them even though their internals share nothing:

  names   what each additive term is, as a stable string -- the key the model set is
          aligned on, because EBM chooses its interaction terms per seed and two seeds
          in the same arm therefore do not carry the same terms in the same order
  orders  each term's arity: 1 for a main effect, 2 for a pairwise interaction
  values  (n_points, n_terms) contribution of each term to each row's logit

``values`` is deliberately evaluated on data rather than read out of the model's own
representation. EBM stores its shape functions as per-bin score vectors whose length
depends on the bins that seed's bootstrap produced, so two seeds' vectors are often
not even the same shape; a NAM has no tabular representation at all. Evaluating every
model on one shared set of rows sidesteps both problems and makes the two families
measurable with the same code.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch


class TermContributions:
    """Per-term contributions of one fitted model, evaluated on a fixed set of rows.

    ``intercept`` is the additive constant the terms do *not* carry, so that
    ``values.sum(axis=1) + intercept`` is the model's own logit. Keeping it is what
    makes the decomposition checkable (:func:`logit_residual`) rather than merely
    plausible -- an attribution that does not reconstruct the prediction is not an
    explanation of that prediction.
    """

    def __init__(
        self, names: list[str], orders: list[int], values: np.ndarray, intercept: float = 0.0
    ) -> None:
        if len(names) != len(orders) or values.shape[1] != len(names):
            raise ValueError(
                f"Inconsistent term data: {len(names)} names, {len(orders)} orders, "
                f"{values.shape[1]} columns"
            )
        self.names = names
        self.orders = orders
        self.values = values
        self.intercept = float(intercept)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.values, columns=self.names)

    def logits(self) -> np.ndarray:
        return self.values.sum(axis=1) + self.intercept

    def centred(self, means: np.ndarray | None = None) -> TermContributions:
        """Subtract a per-term constant, folding what is removed into the intercept.

        An additive model identifies its shape functions only up to a constant that the
        intercept absorbs, so two behaviourally identical models can differ arbitrarily
        in raw contributions (decision D3). Folding the removed constants back into the
        intercept keeps :func:`logits` exact, which is what lets the centred
        decomposition be re-checked rather than taken on trust.

        ``means`` defaults to the mean over *these* rows; pass the train-set means to
        centre on the training marginal, which is the reference the plan specifies and
        the only one that is independent of the rows being explained.
        """
        means = self.values.mean(axis=0) if means is None else np.asarray(means, dtype=float)
        if means.shape != (len(self.names),):
            raise ValueError(f"Expected {len(self.names)} term means, got {means.shape}")
        return TermContributions(
            list(self.names), list(self.orders), self.values - means,
            self.intercept + float(means.sum()),
        )


def _ebm_terms(learner: Any, x: pd.DataFrame) -> TermContributions:
    ebm = learner.model
    assert ebm is not None, "EBM was never fitted"
    # eval_terms returns the additive contribution of every term, in term order --
    # the same decomposition explain_local reports, without building an explanation
    # object per row (which is minutes rather than milliseconds over a test set).
    values = np.asarray(ebm.eval_terms(x), dtype=float)
    names = [str(name) for name in ebm.term_names_]
    orders = [len(features) for features in ebm.term_features_]
    intercept = float(np.asarray(ebm.intercept_, dtype=float).ravel()[0])
    return TermContributions(names, orders, values, intercept)


def _usable_device(learner: Any) -> torch.device:
    """The device this NAM can actually run on now, not the one it was trained on.

    A fitted NAM pickles the ``torch.device`` it was trained on. Reloading a
    CUDA-trained model on a machine without CUDA would otherwise fail inside
    ``_tensor``, which is exactly the reload risk the plan flags -- so the device is
    re-resolved against the current machine rather than trusted from the pickle.
    """
    device = getattr(learner, "device", None)
    if device is not None and torch.device(device).type == "cuda" and torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")


def _nam_terms(learner: Any, x: pd.DataFrame) -> TermContributions:
    assert learner.net is not None, "NAM was never fitted"
    learner.device = _usable_device(learner)
    learner.net.to(learner.device)
    learner.net.eval()
    with torch.no_grad():
        values = learner.net.contributions(learner._tensor(x)).cpu().numpy().astype(float)
        intercept = float(learner.net.bias.detach().cpu().numpy().ravel()[0])
    names = [str(name) for name in learner.columns]
    # feature_dropout is the identity under eval(), so the sum of contributions plus
    # the bias is exactly the logit -- asserted by logit_residual, not assumed.
    return TermContributions(names, [1] * len(names), values, intercept)


# A probability this close to 0 or 1 carries no recoverable logit: a NAM predicts in
# float32, whose ~1e-7 resolution at p = 1 - 1e-8 spans several logits. Rows outside
# this band are excluded from the logit-scale reading, never from the probability one.
SATURATION = 1e-6


def reconstruction_error(learner: Any, x: pd.DataFrame, terms: TermContributions) -> dict:
    """How far the term decomposition is from the model it claims to decompose.

    The gate of plan step 1.1. The decomposition is only an explanation of the model if
    it adds back up to the model, so this is checked rather than assumed -- once here,
    and again per model inside ``analysis.explanations``.

    Measured in **probability** space, which is where ``predict_proba`` actually carries
    information. The plan states the check on the logit scale, and that is the right
    statement of the identity, but it is not a usable test at the edges: a NAM computes
    in float32, so a row it is confident about arrives as p = 1.0 exactly, whose logit
    is +inf and whose recovered value is whatever the clip produced. Inverting the
    sigmoid there manufactures an error of *several logits* out of a reconstruction that
    is correct to 1e-7 -- which is precisely what a first run of this check reported.

    The logit residual is still returned, restricted to the rows where the inversion is
    conditioned at all, because on those rows it is the more legible number.
    """
    probs = np.asarray(learner.predict_proba(x), dtype=float)
    reconstructed = terms.logits()
    errors = {
        "max_prob_error": float(np.abs(_sigmoid(reconstructed) - probs).max()),
        "n_saturated": int(((probs <= SATURATION) | (probs >= 1 - SATURATION)).sum()),
    }

    usable = (probs > SATURATION) & (probs < 1 - SATURATION)
    errors["max_logit_error"] = (
        float(np.abs(reconstructed[usable] - np.log(probs[usable] / (1 - probs[usable]))).max())
        if usable.any() else 0.0
    )
    return errors


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # The stable branch form: exp of a large positive z overflows, exp of a large
    # negative one does not.
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exponential = np.exp(z[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out


def term_contributions(learner: Any, x: pd.DataFrame) -> TermContributions:
    """Dispatch on what the learner can do, not on its class name.

    Mirrors ``registry.importances``: the CLI and the analysis stage stay ignorant of
    which learner they hold.
    """
    if hasattr(getattr(learner, "model", None), "eval_terms"):
        return _ebm_terms(learner, x)
    if hasattr(learner, "net") and hasattr(learner, "columns"):
        return _nam_terms(learner, x)
    raise NotImplementedError(
        f"{type(learner).__name__} exposes no additive term decomposition; explanation "
        "multiplicity is only defined for the additive families (EBM, NAM)."
    )

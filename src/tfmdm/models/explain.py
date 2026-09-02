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
    """Per-term contributions of one fitted model, evaluated on a fixed set of rows."""

    def __init__(self, names: list[str], orders: list[int], values: np.ndarray) -> None:
        if len(names) != len(orders) or values.shape[1] != len(names):
            raise ValueError(
                f"Inconsistent term data: {len(names)} names, {len(orders)} orders, "
                f"{values.shape[1]} columns"
            )
        self.names = names
        self.orders = orders
        self.values = values

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.values, columns=self.names)


def _ebm_terms(learner: Any, x: pd.DataFrame) -> TermContributions:
    ebm = learner.model
    assert ebm is not None, "EBM was never fitted"
    # eval_terms returns the additive contribution of every term, in term order --
    # the same decomposition explain_local reports, without building an explanation
    # object per row (which is minutes rather than milliseconds over a test set).
    values = np.asarray(ebm.eval_terms(x), dtype=float)
    names = [str(name) for name in ebm.term_names_]
    orders = [len(features) for features in ebm.term_features_]
    return TermContributions(names, orders, values)


def _nam_terms(learner: Any, x: pd.DataFrame) -> TermContributions:
    assert learner.net is not None, "NAM was never fitted"
    learner.net.eval()
    with torch.no_grad():
        values = learner.net.contributions(learner._tensor(x)).cpu().numpy().astype(float)
    names = [str(name) for name in learner.columns]
    return TermContributions(names, [1] * len(names), values)


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

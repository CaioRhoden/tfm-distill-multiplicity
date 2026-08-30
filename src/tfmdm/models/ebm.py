"""Explainable Boosting Machine, with soft targets via the weighted-duplication trick.

EBM has no soft-label API, but it does accept ``sample_weight``, which is all that
``base.expand_soft_targets`` needs to turn soft cross-entropy into an equivalent
weighted log-loss problem. The distilled arm therefore needs no change to the learner.

Interactions are switched off so the model stays purely additive and therefore
comparable to the NAM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier

from ..config import SOFT_ARMS
from .base import expand_soft_targets


class EBMModel:
    def __init__(self, seed: int, **params: object) -> None:
        self.seed = seed
        self.params = dict(params)
        self.model: ExplainableBoostingClassifier | None = None

    def fit(self, x_train, t_train, x_val, t_val, *, arm: str) -> "EBMModel":
        if arm in SOFT_ARMS:
            x_fit, y_fit, w_fit = expand_soft_targets(x_train, t_train)
        else:
            x_fit, y_fit, w_fit = x_train, np.asarray(t_train).astype(int), None

        self.model = ExplainableBoostingClassifier(random_state=self.seed, **self.params)
        self.model.fit(x_fit, y_fit, sample_weight=w_fit)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "fit() must be called before predict_proba()"
        return self.model.predict_proba(x)[:, 1]

    def feature_importances(self) -> dict[str, float]:
        assert self.model is not None
        names = list(self.model.term_names_)
        scores = np.asarray(self.model.term_importances(), dtype=float)
        return {name: float(score) for name, score in zip(names, scores)}

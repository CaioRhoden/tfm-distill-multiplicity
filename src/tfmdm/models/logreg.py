"""Trivial baseline B0. If it matches a NAM, the dataset is not exercising the method."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..config import SOFT_ARMS
from .base import expand_soft_targets


class LogRegModel:
    def __init__(self, seed: int, **params: object) -> None:
        self.seed = seed
        self.params = dict(params)
        self.model: LogisticRegression | None = None
        self.columns: list[str] = []

    def fit(self, x_train, t_train, x_val, t_val, *, arm: str) -> "LogRegModel":
        self.columns = list(x_train.columns)
        if arm in SOFT_ARMS:
            x_fit, y_fit, w_fit = expand_soft_targets(x_train, t_train)
        else:
            x_fit, y_fit, w_fit = x_train, np.asarray(t_train).astype(int), None
        self.model = LogisticRegression(random_state=self.seed, **self.params)
        self.model.fit(x_fit, y_fit, sample_weight=w_fit)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        return self.model.predict_proba(x)[:, 1]

    def feature_importances(self) -> dict[str, float]:
        assert self.model is not None
        coefs = np.abs(self.model.coef_.ravel())
        return {name: float(value) for name, value in zip(self.columns, coefs)}

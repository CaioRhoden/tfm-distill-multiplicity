"""Baseline B0, and a measured family in its own right.

Two jobs. As a baseline it is the floor the flexible families have to clear: if a NAM
matches a linear model on both accuracy and multiplicity, the dataset is not exercising
the method. As a family it carries the same hard-vs-distilled comparison the others do,
and it is the one family whose additive decomposition is exact by construction rather
than extracted -- a logistic regression *is* a GAM whose shape functions are straight
lines, so ``models.explain`` reads its terms off the coefficients directly.

It runs on the ``encoded`` view, so its terms are one-hot *columns* and are grouped back
to parent features before any metric sees them, exactly as the NAM's are.
"""

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
        self._assert_converged()
        return self

    def _assert_converged(self) -> None:
        """A truncated solver is the one way this family fabricates multiplicity.

        Every other source of seed-to-seed disagreement here is real: the bootstrap
        moves the data, and the fit is deterministic given the data. An lbfgs that hits
        ``max_iter`` instead stops at wherever it happened to be, which differs by seed
        for no reason the study is measuring -- and it bites the distilled arm hardest,
        since ``expand_soft_targets`` doubles the rows, so it would show up as an
        arm-vs-arm difference. sklearn reports this as a warning, which a sweep of 300
        fits swallows; here it is an error.
        """
        assert self.model is not None
        n_iter = int(np.max(np.asarray(self.model.n_iter_)))
        max_iter = int(self.model.max_iter)
        if n_iter >= max_iter:
            raise AssertionError(
                f"LogisticRegression did not converge in {max_iter} iterations "
                f"(seed {self.seed}). Raise max_iter or loosen tol -- an unconverged fit "
                "makes seed-to-seed differences an artifact of the solver."
            )

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        return self.model.predict_proba(x[self.columns])[:, 1]

    def feature_importances_on(self, x: pd.DataFrame) -> dict[str, float]:
        """Mean absolute contribution per column, on the rows handed in (D-IMP).

        Not ``|coef|``. The other families report a data-weighted mean absolute
        contribution, and figure F4 compares the resulting *rankings* -- so the two have
        to mean the same thing. A one-hot level present in 1% of rows can carry a large
        coefficient while contributing nothing to almost every row; ranking by the
        coefficient would park it at the top for all 30 seeds and make the metric read
        as stable for a reason with no interpretive content.
        """
        assert self.model is not None
        coefficients = np.asarray(self.model.coef_, dtype=float).ravel()
        contributions = np.asarray(x[self.columns], dtype=float) * coefficients
        return {name: float(value)
                for name, value in zip(self.columns, np.abs(contributions).mean(axis=0))}

    def feature_importances(self) -> dict[str, float]:
        """Data-free fallback: the coefficients themselves.

        Only reached when no reference rows are available. ``registry.importances``
        prefers ``feature_importances_on``, which is the number that enters the metrics.
        """
        assert self.model is not None
        coefs = np.abs(self.model.coef_.ravel())
        return {name: float(value) for name, value in zip(self.columns, coefs)}

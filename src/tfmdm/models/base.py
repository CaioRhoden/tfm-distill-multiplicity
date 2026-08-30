"""Shared model interface and the objectives that separate the arms."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EPS = 1e-7


def soft_cross_entropy(target: np.ndarray, pred: np.ndarray) -> float:
    """-[p log q + (1-p) log(1-q)], averaged. Reduces to log loss when p is 0/1."""
    p = np.clip(np.asarray(target, dtype=float), 0.0, 1.0)
    q = np.clip(np.asarray(pred, dtype=float), EPS, 1 - EPS)
    return float(-np.mean(p * np.log(q) + (1 - p) * np.log(1 - q)))


def val_objective(arm: str, val_target: np.ndarray, val_pred: np.ndarray) -> float:
    """The quantity model selection minimises on the validation split.

    The hard arm selects on validation AUROC against true labels (negated so that
    lower is always better). The distilled arms select on soft cross-entropy against
    the teacher's validation probabilities -- this is the validation step that keeps
    the distilled arms from ever touching a hard label, which is what decision D6
    requires for the comparison to be attributable.
    """
    if arm == "hard":
        return -float(roc_auc_score(val_target.astype(int), val_pred))
    return soft_cross_entropy(val_target, val_pred)


def expand_soft_targets(
    x: pd.DataFrame, target: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Restate soft cross-entropy as weighted log loss, for learners without a soft-label API.

    Minimising -[p log q + (1-p) log(1-q)] on a row is exactly equivalent to minimising
    ordinary log loss on two copies of that row -- one labelled 1 with weight p, one
    labelled 0 with weight (1-p). Any learner that accepts ``sample_weight`` therefore
    needs no modification at all to be distilled.
    """
    p = np.clip(np.asarray(target, dtype=float), 0.0, 1.0)
    x_doubled = pd.concat([x, x], axis=0, ignore_index=True)
    y_doubled = np.concatenate([np.ones(len(x), dtype=int), np.zeros(len(x), dtype=int)])
    w_doubled = np.concatenate([p, 1.0 - p])
    keep = w_doubled > 1e-8  # zero-weight copies only slow the fit down
    return x_doubled.loc[keep].reset_index(drop=True), y_doubled[keep], w_doubled[keep]


class Model(Protocol):
    """Every learner exposes the same three calls, whatever its target is."""

    def fit(
        self,
        x_train: pd.DataFrame,
        t_train: np.ndarray,
        x_val: pd.DataFrame,
        t_val: np.ndarray,
        *,
        arm: str,
    ) -> "Model":
        ...

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        ...

    def feature_importances(self) -> dict[str, float]:
        """Per-feature global importance, used by figure F4 (explanation stability)."""
        ...

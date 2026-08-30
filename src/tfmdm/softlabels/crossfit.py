"""Phase 2.2 -- cross-fitted soft labels (decision D2).

An in-context learner that can see a row in its own context reproduces that row's
label almost exactly. Predicting on the same rows used as context therefore yields
near-degenerate 0/1 probabilities, distillation quietly collapses into hard-label
training, and H2 becomes untestable through no fault of the hypothesis.

So training-set soft labels are produced out-of-fold: for fold k, the context is
train minus fold k and the queries are fold k. Validation soft labels use the full
training set as context, which is legitimate because no validation row is ever in it.
Test rows never receive soft labels at all -- test is always scored against truth.

``teacher`` is any callable with the in-context signature, so the same machinery would
serve a different teacher unchanged -- which is what makes reinstating the smoothing
control cut in D4 a config change rather than a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

EPS = 1e-12

TeacherFn = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame, int], np.ndarray]


def entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return -(q * np.log(q) + (1 - q) * np.log(1 - q))


@dataclass
class CrossFitResult:
    oof_probs: np.ndarray
    fold_ids: np.ndarray
    in_context_probs: np.ndarray | None
    diagnostics: dict[str, float]


def cross_fit(
    teacher: TeacherFn,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    n_folds: int = 5,
    seed: int = 0,
    compute_in_context: bool = True,
) -> CrossFitResult:
    y_train = np.asarray(y_train).astype(int)
    oof = np.full(len(x_train), np.nan)
    fold_ids = np.full(len(x_train), -1, dtype=int)

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (ctx_idx, query_idx) in enumerate(splitter.split(x_train, y_train)):
        probs = teacher(
            x_train.iloc[ctx_idx].reset_index(drop=True),
            y_train[ctx_idx],
            x_train.iloc[query_idx].reset_index(drop=True),
            seed,
        )
        oof[query_idx] = np.asarray(probs, dtype=float)
        fold_ids[query_idx] = fold

    if np.isnan(oof).any():
        raise AssertionError("Cross-fitting left some training rows without a soft label")

    in_context = None
    if compute_in_context:
        in_context = np.asarray(
            teacher(x_train, y_train, x_train, seed), dtype=float
        )

    diagnostics = {
        "oof_mean": float(oof.mean()),
        "train_positive_rate": float(y_train.mean()),
        "oof_mean_entropy": float(entropy(oof).mean()),
        "oof_frac_extreme": float(np.mean((oof < 0.01) | (oof > 0.99))),
    }
    if in_context is not None:
        diagnostics["in_context_mean_entropy"] = float(entropy(in_context).mean())
        diagnostics["entropy_gain"] = (
            diagnostics["oof_mean_entropy"] - diagnostics["in_context_mean_entropy"]
        )
    return CrossFitResult(oof, fold_ids, in_context, diagnostics)


def assert_honest(result: CrossFitResult, mean_tolerance: float = 0.01) -> None:
    """The Phase 2.2 done-when checks, enforced in code rather than in a notebook.

    Two conditions. The soft labels must be calibrated in aggregate -- their mean sits
    within one point of the training positive rate. And they must be strictly
    higher-entropy than in-context predictions on the same rows, which is the direct
    evidence that cross-fitting removed the memorisation.
    """
    diag = result.diagnostics
    drift = abs(diag["oof_mean"] - diag["train_positive_rate"])
    if drift > mean_tolerance:
        raise AssertionError(
            f"Out-of-fold soft labels average {diag['oof_mean']:.4f} against a training "
            f"positive rate of {diag['train_positive_rate']:.4f} (drift {drift:.4f} > "
            f"{mean_tolerance}). The teacher is miscalibrated; distillation would inherit it."
        )
    if "entropy_gain" in diag and diag["entropy_gain"] <= 0.0:
        raise AssertionError(
            "Out-of-fold soft labels are no higher-entropy than in-context predictions "
            f"(gain {diag['entropy_gain']:.5f}). Either cross-fitting is not working, or "
            "the teacher does not memorise its context -- check before trusting D2."
        )

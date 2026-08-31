"""TabICLv2 adapter.

The distribution name and API surface of TabICLv2 are an open [DECIDE] item in the
plan, so the real backend is isolated behind one class with a single method. Set
``TFMDM_TABICL_BACKEND=mock`` to exercise the whole pipeline -- splits, cross-fitting,
distillation, metrics, figures -- without the package installed; the mock is a plain
logistic regression and is never a substitute for a real run.

TabICLv2 is an in-context learner: it does not train. ``fit_predict`` hands it a
context (the labelled rows it may condition on) and a set of query rows.
"""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np
import pandas as pd


class TabICLBackend(Protocol):
    name: str

    def fit_predict(
        self, context_x: pd.DataFrame, context_y: np.ndarray, query_x: pd.DataFrame, seed: int
    ) -> np.ndarray:
        """Return P(y=1) for each query row, conditioned on the context."""


class RealTabICL:
    """Thin wrapper over the installed TabICLv2 classifier."""

    name = "tabicl"

    def __init__(self, max_context_rows: int | None = None, **kwargs: object) -> None:
        self.max_context_rows = max_context_rows
        self.kwargs = kwargs

    def _subsample(
        self, x: pd.DataFrame, y: np.ndarray, seed: int
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Stratified context truncation when the table exceeds the model's window.

        Whether this fires at all is settled by the Phase 2.1 feasibility probe. If it
        does, it is a documented limitation of the run, not a silent fallback -- the
        caller logs ``context_rows`` for every prediction pass.
        """
        if self.max_context_rows is None or len(x) <= self.max_context_rows:
            return x, y
        rng = np.random.default_rng(seed)
        keep: list[np.ndarray] = []
        for value in np.unique(y):
            pool = np.flatnonzero(y == value)
            share = int(round(self.max_context_rows * pool.size / y.size))
            keep.append(rng.choice(pool, size=min(share, pool.size), replace=False))
        idx = np.sort(np.concatenate(keep))
        return x.iloc[idx].reset_index(drop=True), y[idx]

    def fit_predict(self, context_x, context_y, query_x, seed):
        try:
            from tabicl import TabICLClassifier  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "TabICLv2 is not installed. Pin the correct distribution in "
                "pyproject.toml's [tabicl] extra and run `uv sync --extra tabicl`, or "
                "set TFMDM_TABICL_BACKEND=mock to smoke-test the pipeline."
            ) from exc

        ctx_x, ctx_y = self._subsample(context_x, np.asarray(context_y), seed)
        clf = TabICLClassifier(random_state=seed, **self.kwargs)
        clf.fit(ctx_x, ctx_y)
        return np.asarray(clf.predict_proba(query_x))[:, 1]


class MockTabICL:
    """Deterministic stand-in with the same call signature. Not a scientific result."""

    name = "mock"

    def __init__(self, **_kwargs: object) -> None:
        pass

    def fit_predict(self, context_x, context_y, query_x, seed):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OrdinalEncoder, StandardScaler
        from sklearn.compose import ColumnTransformer

        num = [c for c in context_x.columns if pd.api.types.is_numeric_dtype(context_x[c])]
        cat = [c for c in context_x.columns if c not in num]
        pre = ColumnTransformer(
            [("num", StandardScaler(), num),
             ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat)]
        )
        pipe = make_pipeline(pre, LogisticRegression(max_iter=1000, random_state=seed))
        pipe.fit(context_x, np.asarray(context_y))
        return pipe.predict_proba(query_x)[:, 1]


def get_backend(**kwargs: object) -> TabICLBackend:
    choice = os.environ.get("TFMDM_TABICL_BACKEND", "tabicl").lower()
    if choice == "mock":
        return MockTabICL(**kwargs)
    if choice == "tabicl":
        return RealTabICL(**kwargs)
    raise ValueError(f"Unknown TFMDM_TABICL_BACKEND={choice!r}; expected 'tabicl' or 'mock'")

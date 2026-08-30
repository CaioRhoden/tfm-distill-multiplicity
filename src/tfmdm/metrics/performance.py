"""Point-prediction quality.

AUROC is the primary metric (decision D5). Log loss is kept alongside it but is not a
reported metric: it is the quantity the Rashomon filter uses to decide which models
are within epsilon of the best, and the quantity model selection minimises.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def performance(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """AUROC plus the log loss used internally for selection and Rashomon filtering.

    ``threshold`` is accepted and ignored -- AUROC is threshold-free, and it is kept in
    the signature so callers do not have to know that.
    """
    del threshold
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }

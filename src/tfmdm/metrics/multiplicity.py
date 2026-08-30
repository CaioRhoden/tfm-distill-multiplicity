"""Predictive multiplicity over a set of models trained by the same procedure.

Definitions follow Marx, Calmon & Ustun (2020), "Predictive Multiplicity in
Classification". Given a reference model h0 and a model set H:

  ambiguity   = fraction of test points on which *some* h in H disagrees with h0
  discrepancy = max over h in H of the fraction of points where h disagrees with h0

``max_pairwise_discrepancy`` is reported alongside as a reference-free companion: the
largest disagreement rate over all ordered model pairs. It answers "how far apart can
two equally-defensible models be" without privileging a particular h0.

All of these are computed on a test set that is identical across arms, models and seeds
(decision D1) -- they are not comparable otherwise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MultiplicityResult:
    ambiguity: float
    discrepancy: float
    max_pairwise_discrepancy: float
    n_models: int
    n_points: int
    reference_index: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _validate(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2:
        raise ValueError(f"Expected a (n_points, n_models) matrix, got shape {probs.shape}")
    if probs.shape[1] < 2:
        raise ValueError("Multiplicity is undefined for fewer than two models")
    if not np.all((probs >= 0.0) & (probs <= 1.0)):
        raise ValueError("Probabilities outside [0, 1]")
    return probs


def disagreement_matrix(
    probs: np.ndarray, threshold: float = 0.5, reference_index: int = 0
) -> np.ndarray:
    """Boolean (n_points, n_models): does model j disagree with the reference on point i?

    The reference column is included and is all-False by construction, which keeps the
    matrix's column indices aligned with the model set.
    """
    probs = _validate(probs)
    decisions = (probs >= threshold)
    return decisions != decisions[:, [reference_index]]


def ambiguity(disagree: np.ndarray) -> float:
    return float(disagree.any(axis=1).mean())


def discrepancy(disagree: np.ndarray) -> float:
    return float(disagree.mean(axis=0).max())


def max_pairwise_discrepancy(probs: np.ndarray, threshold: float = 0.5) -> float:
    decisions = (_validate(probs) >= threshold)
    n_points, n_models = decisions.shape
    best = 0.0
    for j in range(n_models):
        # Broadcast one column against all others; O(n_models) passes of O(n_points * n_models).
        rate = (decisions != decisions[:, [j]]).mean(axis=0).max()
        best = max(best, float(rate))
    return best


def multiplicity(
    probs: np.ndarray, threshold: float = 0.5, reference_index: int = 0
) -> MultiplicityResult:
    probs = _validate(probs)
    disagree = disagreement_matrix(probs, threshold, reference_index)
    return MultiplicityResult(
        ambiguity=ambiguity(disagree),
        discrepancy=discrepancy(disagree),
        max_pairwise_discrepancy=max_pairwise_discrepancy(probs, threshold),
        n_models=int(probs.shape[1]),
        n_points=int(probs.shape[0]),
        reference_index=int(reference_index),
    )


def ambiguity_jackknife(disagree: np.ndarray) -> np.ndarray:
    """Leave-one-test-point-out values of ambiguity, in closed form.

    Ambiguity is a mean over points, so deleting point i just removes its indicator.
    This makes BCa acceleration affordable on a 10k-point test set, where a naive
    recompute-per-point jackknife would not be.
    """
    per_point = disagree.any(axis=1).astype(float)
    n = per_point.size
    return (per_point.sum() - per_point) / (n - 1)


def discrepancy_jackknife(disagree: np.ndarray) -> np.ndarray:
    """Leave-one-out values of discrepancy, in closed form.

    Discrepancy is a max over models of a per-model mean, so deleting point i only
    shifts each model's count by that model's indicator before the max is retaken.
    """
    counts = disagree.sum(axis=0).astype(float)
    n = disagree.shape[0]
    return ((counts[None, :] - disagree.astype(float)) / (n - 1)).max(axis=1)


def threshold_curve(
    probs: np.ndarray, thresholds: np.ndarray, reference_index: int = 0
) -> dict[str, np.ndarray]:
    """Ambiguity and discrepancy as a function of the decision threshold (figure F4).

    Guards against a result that only exists at 0.5.
    """
    amb, disc = [], []
    for t in thresholds:
        d = disagreement_matrix(probs, float(t), reference_index)
        amb.append(ambiguity(d))
        disc.append(discrepancy(d))
    return {"threshold": np.asarray(thresholds, dtype=float),
            "ambiguity": np.asarray(amb), "discrepancy": np.asarray(disc)}

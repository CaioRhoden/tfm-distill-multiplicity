"""Uncertainty over test points (Phase 5.3).

Resampling is over *test points*, not over the 30 seeds: ambiguity and discrepancy are
already statistics of the whole model set, so their sampling variability comes from
which points happen to be in the test set.

BCa is used where a closed-form jackknife is available (the multiplicity metrics, via
``metrics.multiplicity``); elsewhere we fall back to the percentile interval rather
than pay for a 10,000-fold AUROC jackknife.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    method: str

    def excludes_zero(self) -> bool:
        return (self.low > 0.0) or (self.high < 0.0)

    def as_dict(self, prefix: str = "") -> dict[str, float | str]:
        return {f"{prefix}point": self.point, f"{prefix}ci_low": self.low,
                f"{prefix}ci_high": self.high, f"{prefix}ci_method": self.method}


def _boot_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def percentile_ci(
    stat_fn: Callable[[np.ndarray], float],
    n: int,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    point = stat_fn(np.arange(n))
    draws = np.array([stat_fn(idx) for idx in _boot_indices(n, n_boot, seed)])
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return Interval(float(point), float(low), float(high), "percentile")


def bca_ci(
    stat_fn: Callable[[np.ndarray], float],
    n: int,
    jackknife_values: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Bias-corrected and accelerated interval, given precomputed leave-one-out values."""
    point = stat_fn(np.arange(n))
    draws = np.array([stat_fn(idx) for idx in _boot_indices(n, n_boot, seed)])

    prop_below = float(np.mean(draws < point))
    if prop_below in (0.0, 1.0):  # degenerate; BCa's z0 is undefined
        low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
        return Interval(float(point), float(low), float(high), "percentile-fallback")
    z0 = stats.norm.ppf(prop_below)

    jack = np.asarray(jackknife_values, dtype=float)
    centred = jack.mean() - jack
    denom = 6.0 * (np.sum(centred ** 2) ** 1.5)
    acc = 0.0 if denom == 0 else float(np.sum(centred ** 3) / denom)

    z_lo, z_hi = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
    def adjust(z: float) -> float:
        return float(stats.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z))))

    low, high = np.quantile(draws, [adjust(z_lo), adjust(z_hi)])
    return Interval(float(point), float(low), float(high), "bca")


def paired_bootstrap(
    stat_a: Callable[[np.ndarray], float],
    stat_b: Callable[[np.ndarray], float],
    n: int,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[Interval, float]:
    """CI and two-sided p-value for ``stat_a - stat_b``, resampling the same points for both.

    Pairing matters: both arms are evaluated on the identical test set, so the
    difference has far less variance than the two marginals suggest.
    """
    point = stat_a(np.arange(n)) - stat_b(np.arange(n))
    draws = np.array([stat_a(idx) - stat_b(idx) for idx in _boot_indices(n, n_boot, seed)])
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    # Bootstrap p-value: how often the resampled difference crosses zero, two-sided.
    tail = min(float(np.mean(draws <= 0.0)), float(np.mean(draws >= 0.0)))
    p_value = min(1.0, 2.0 * tail)
    return Interval(float(point), float(low), float(high), "percentile-paired"), p_value


def holm(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down. Returns a rejection flag per input, in input order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = p.size
    reject = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        if p[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break  # step-down: once one fails, all larger p-values fail too
    return reject.tolist()

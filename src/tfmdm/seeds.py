"""Seeding and the shared resampling protocol (decision D3)."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # torch is optional for the EBM-only path
        pass


def stratified_bootstrap_indices(y: np.ndarray, seed: int) -> np.ndarray:
    """Draw a stratified bootstrap resample of ``y``'s index positions.

    Every method in the study -- TabICLv2 included -- builds its 30-model set from
    this same perturbation. Without it, a deterministic in-context model would show
    zero multiplicity by construction and H1 would be untestable (decision D3).

    Stratification uses the *hard* label even for the distilled arms, so that the
    class balance of a resample is identical across arms for a given seed.
    """
    rng = np.random.default_rng(seed)
    out = []
    for value in np.unique(y):
        pool = np.flatnonzero(y == value)
        out.append(rng.choice(pool, size=pool.size, replace=True))
    idx = np.concatenate(out)
    rng.shuffle(idx)
    return idx

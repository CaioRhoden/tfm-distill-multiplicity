"""Loading a fitted learner back, on a machine that need not resemble the one that fit it.

``joblib.dump`` of a NAM pickles a live ``torch`` module, and torch records the *device*
each tensor storage lived on. Restoring one on a machine without a working CUDA device
therefore fails inside the unpickler, before any of our code gets a chance to move the
model anywhere:

    RuntimeError: Attempting to deserialize object on a CUDA device but
    torch.cuda.is_available() is False.

This is not a hypothetical portability worry. A model set is fit once on a GPU node and
then re-scored many times -- by the explanation analysis, which is inference-only and has
no use for a GPU at all -- and those re-scorings routinely land on a CPU-only node, or on
a node whose driver is too old for the installed torch build, which makes CUDA
unavailable just as surely as having no GPU. The artifacts must outlive the machine that
produced them, so the load path maps storages to CPU whenever CUDA cannot be used.

``NAMModel._tensor`` reads ``self.device`` from the unpickled object, which still says
``cuda``; ``models.explain`` re-resolves that against the current machine. The two halves
are separate on purpose -- this one gets the bytes into memory, that one decides where the
forward pass runs.
"""

from __future__ import annotations

import io
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import joblib
import torch


@contextmanager
def _storages_on_cpu():
    """Force every torch storage unpickled inside the block onto the CPU.

    joblib drives the outer unpickling itself and calls ``torch.storage._load_from_bytes``
    for each tensor, with no ``map_location`` to pass through -- so redirecting that one
    function is the only seam available. It is restored on the way out, including on
    error, so a failed load cannot leave the process with torch's deserialisation
    quietly rewired.
    """
    original = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda data: torch.load(
        io.BytesIO(data), map_location="cpu", weights_only=False
    )
    try:
        yield
    finally:
        torch.storage._load_from_bytes = original


def load_learner(path: str | Path) -> Any:
    """Load a fitted model artifact, whatever device it was trained on.

    On a machine with working CUDA this is a plain ``joblib.load``: a GPU-trained model
    is restored to the GPU and scores there. Only when CUDA is unavailable are storages
    remapped, so the fast path is never paid for by the common case.
    """
    if torch.cuda.is_available():
        return joblib.load(path)
    with _storages_on_cpu():
        return joblib.load(path)

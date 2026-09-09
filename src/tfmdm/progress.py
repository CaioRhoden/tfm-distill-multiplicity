"""Human-readable progress, on stderr.

Every stage in this package prints a machine-readable JSON payload to *stdout* when it
finishes, and ``tfmdm groups`` emits one JSON object per line for SLURM to consume. So
progress reporting cannot go to stdout without corrupting that contract -- it goes to
stderr, which an interactive shell interleaves and a redirect keeps separate.

The stages this exists for are the slow, silent ones: ``explanations`` reloads 30 fitted
models per cell and then runs a few thousand bootstrap resamples over them, which is
minutes of no output at all. The question a progress line has to answer is not "is it
alive" but "which cell, how far in, and how long has that taken" -- because when a run
is too slow the answer is usually a specific cell, not the stage as a whole.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TypeVar

T = TypeVar("T")

# Off by default: a stage's stdout contract is its output, and every existing caller --
# the Taskfile loops, the SLURM scripts -- was written against a silent run. Verbosity
# is opt-in via ``tfmdm <stage> --verbose``.
_enabled = False
_started = time.perf_counter()


def set_enabled(enabled: bool) -> None:
    """Turn progress output on or off, and reset the elapsed clock."""
    global _enabled, _started
    _enabled = bool(enabled)
    _started = time.perf_counter()


def enabled() -> bool:
    return _enabled


def log(message: str) -> None:
    """One timestamped line on stderr, elapsed since the stage began."""
    if not _enabled:
        return
    print(f"[{time.perf_counter() - _started:7.1f}s] {message}", file=sys.stderr, flush=True)


@contextmanager
def phase(label: str):
    """Log a step's start and how long it took, even if it raises.

    The duration is the point: it is what tells you whether a slow run is slow in the
    model loading, the intervals, or the comparisons -- three stages with very
    different fixes.
    """
    log(f"{label} ...")
    started = time.perf_counter()
    try:
        yield
    except BaseException:
        log(f"{label} FAILED after {time.perf_counter() - started:.1f}s")
        raise
    log(f"{label} done in {time.perf_counter() - started:.1f}s")


def track(iterable: Iterable[T], label: str, total: int | None = None,
          every: int = 5) -> Iterator[T]:
    """Yield from ``iterable``, logging every ``every`` items and at the end.

    Deliberately periodic lines rather than a redrawing progress bar: this runs under
    SLURM as often as it runs in a terminal, and a bar's carriage returns turn a
    captured job log into one unreadable line.
    """
    if not _enabled:
        yield from iterable
        return

    started = time.perf_counter()
    count = 0
    for count, item in enumerate(iterable, start=1):
        yield item
        if count % every == 0:
            elapsed = time.perf_counter() - started
            suffix = f"/{total}" if total else ""
            rate = elapsed / count
            eta = f", eta {rate * (total - count):.0f}s" if total and count < total else ""
            log(f"  {label}: {count}{suffix} ({elapsed:.0f}s elapsed{eta})")
    log(f"  {label}: {count} done in {time.perf_counter() - started:.1f}s")

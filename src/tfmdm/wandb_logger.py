"""W&B wrapper: offline-first, because cluster compute nodes usually have no network.

WANDB_MODE is read from the environment (online | offline | disabled). In offline
mode runs land in ``wandb/offline-run-*`` and are pushed later with ``task wandb:sync``.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from . import paths


def mode() -> str:
    return os.environ.get("WANDB_MODE", "offline").lower()


@contextmanager
def run(
    *,
    name: str,
    group: str,
    job_type: str,
    config: dict[str, Any],
    tags: list[str] | None = None,
) -> Iterator[Any]:
    """Open a W&B run, or a no-op stand-in when WANDB_MODE=disabled."""
    if mode() == "disabled":
        yield _NullRun()
        return

    import wandb

    paths.WANDB_DIR.mkdir(parents=True, exist_ok=True)
    handle = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "tfm-distill-multiplicity"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=name,
        group=group,
        job_type=job_type,
        tags=tags or [],
        config=config,
        dir=str(paths.ROOT),
        mode=mode(),
        reinit=True,
    )
    try:
        yield handle
    finally:
        handle.finish()


class _NullRun:
    """Stand-in with the slice of the wandb.Run surface this project uses."""

    def log(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def summary_update(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    @property
    def summary(self) -> dict[str, Any]:
        return {}

    def finish(self) -> None:
        return None

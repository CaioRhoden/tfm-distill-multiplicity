"""Run provenance: git commit, lockfile hash, dataset digest, environment."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from . import paths


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=paths.ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_commit() -> str | None:
    return _git("rev-parse", "HEAD")


def git_dirty() -> bool:
    status = _git("status", "--porcelain")
    return bool(status)


def sha256(path: Path, chunk: int = 1 << 20) -> str | None:
    if not Path(path).exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def collect(dataset_path: Path | None = None) -> dict[str, object]:
    return {
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "uv_lock_sha256": sha256(paths.ROOT / "uv.lock"),
        "python": sys.version.split()[0],
        "dataset_sha256": sha256(dataset_path) if dataset_path else None,
    }


class DirtyWorkingTree(RuntimeError):
    pass


def guard_clean_tree(allow_dirty: bool) -> None:
    """Refuse to start a logged run from a dirty tree unless explicitly allowed.

    A figure that cannot be traced back to an exact commit is not a result.
    """
    if git_dirty() and not allow_dirty:
        raise DirtyWorkingTree(
            "Working tree has uncommitted changes, so this run would not be reproducible. "
            "Commit them, or re-run with ALLOW_DIRTY=1 to override."
        )

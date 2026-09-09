"""Config loading: base + dataset + model, merged with OmegaConf."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from . import paths

# The two training arms.
ARMS = ("hard", "distilled")

# Arms whose targets are soft probabilities rather than 0/1 labels.
SOFT_ARMS = ("distilled",)

# Control arms. Not part of any hypothesis -- they exist to put a known answer at each
# end of a metric's range, so a number in the middle can be read. ``shuffled`` trains
# on permuted labels, which leaves the model set nothing to agree about and should
# therefore saturate explanation multiplicity (plan gate 2.2).
CONTROL_ARMS = ("shuffled",)

# Arms whose targets are 0/1 labels, and which therefore select on validation AUROC.
HARD_ARMS = ("hard",) + CONTROL_ARMS

ALL_ARMS = ARMS + CONTROL_ARMS


def load(dataset: str, model: str | None = None, split_seed: int | None = None) -> DictConfig:
    """Compose the config for one (dataset, model, split) combination.

    ``split_seed`` overrides ``split.seed`` from base.yaml, which is what makes the
    partition a run-time dimension rather than a constant. The base value stays as the
    default so a plain ``load(dataset)`` still resolves to the primary split.
    """
    cfg = OmegaConf.load(paths.CONFIGS / "base.yaml")
    cfg.dataset = OmegaConf.load(paths.CONFIGS / "dataset" / f"{dataset}.yaml")
    if model is not None:
        cfg.model = OmegaConf.load(paths.CONFIGS / "model" / f"{model}.yaml")
    if split_seed is not None:
        cfg.split.seed = int(split_seed)
    return cfg  # type: ignore[return-value]


def split_seeds(cfg: DictConfig) -> list[int]:
    """Every split seed the experiment covers, primary split first."""
    return [int(s) for s in cfg.split.seeds]


def apply_tuned(cfg: DictConfig, dataset: str, model: str, arm: str, split_seed: int) -> DictConfig:
    """Overlay the hyperparameters chosen once per (dataset, model, arm, split).

    Missing tuned files are not an error: they mean the sweep is running on the
    defaults in configs/model/, which is what the sanity phase does.
    """
    path = paths.tuned_config(dataset, model, arm, split_seed)
    if path.exists():
        tuned = OmegaConf.load(path)
        cfg.model.params = OmegaConf.merge(cfg.model.params, tuned.get("params", {}))
        cfg.model.tuned_from = str(path)
    else:
        cfg.model.tuned_from = None
    return cfg


def to_dict(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]

"""Config loading: base + dataset + model, merged with OmegaConf."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from . import paths

# The two training arms.
ARMS = ("hard", "distilled")

# Arms whose targets are soft probabilities rather than 0/1 labels.
SOFT_ARMS = ("distilled",)


def load(dataset: str, model: str | None = None) -> DictConfig:
    cfg = OmegaConf.load(paths.CONFIGS / "base.yaml")
    cfg.dataset = OmegaConf.load(paths.CONFIGS / "dataset" / f"{dataset}.yaml")
    if model is not None:
        cfg.model = OmegaConf.load(paths.CONFIGS / "model" / f"{model}.yaml")
    return cfg  # type: ignore[return-value]


def apply_tuned(cfg: DictConfig, dataset: str, model: str, arm: str) -> DictConfig:
    """Overlay the hyperparameters chosen once per (dataset, model, arm) in Phase 4.2.

    Missing tuned files are not an error: they mean the sweep is running on the
    defaults in configs/model/, which is what the sanity phase does.
    """
    path = paths.tuned_config(dataset, model, arm)
    if path.exists():
        tuned = OmegaConf.load(path)
        cfg.model.params = OmegaConf.merge(cfg.model.params, tuned.get("params", {}))
        cfg.model.tuned_from = str(path)
    else:
        cfg.model.tuned_from = None
    return cfg


def to_dict(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]

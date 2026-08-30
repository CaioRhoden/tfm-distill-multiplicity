"""Name -> constructor, so the CLI never imports a model module directly."""

from __future__ import annotations

from typing import Any, Callable

MODELS: dict[str, str] = {"ebm": "EBM", "nam": "NAM", "logreg": "LogisticRegression"}


def build(name: str, seed: int, params: dict[str, Any]) -> Any:
    factories: dict[str, Callable[..., Any]] = {}

    if name == "ebm":
        from .ebm import EBMModel

        factories["ebm"] = EBMModel
    elif name == "nam":
        from .nam import NAMModel

        factories["nam"] = NAMModel
    elif name == "logreg":
        from .logreg import LogRegModel

        factories["logreg"] = LogRegModel
    else:
        raise KeyError(f"Unknown model {name!r}; expected one of {sorted(MODELS)}")

    return factories[name](seed=seed, **params)


def importances(model: Any, reference_x: Any) -> dict[str, float]:
    """Uniform access to global importances across learners (figure F4)."""
    if hasattr(model, "feature_importances_on"):
        return model.feature_importances_on(reference_x)
    return model.feature_importances()

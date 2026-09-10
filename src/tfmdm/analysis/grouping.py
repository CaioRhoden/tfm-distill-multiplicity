"""Plan step 1.3 -- the unit an explanation is read at, per family (decision D4).

A NAM trained on the ``encoded`` view has one feature net per *one-hot column*, so it
reports ~80 terms for adult's 12 features: ``occupation_Sales`` rather than
``occupation``. Nobody reads an explanation at that granularity, and worse, splitting
one feature across 15 columns systematically shrinks each column's attribution and so
changes which term wins a top-1 comparison. Contributions are therefore summed back to
the parent feature before any metric sees them.

The EBM is left alone: it already works in native features, and its pair terms are
genuinely read as pairs, so every term -- main and interaction -- is its own unit.

Nothing is compared across the two families (the plan's scope decision), so the two
maps need not agree, and neither is coerced towards the other.

The grouping is *lossless* only in combination with the centring of D3: exactly one
level of a parent is active per row, but the inactive levels' nets still emit f(0),
so an uncentred sum would carry the other 14 levels' constants. See
``analysis.explanations`` for where the two steps meet.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

import joblib

from .. import paths
from ..config import uses_encoded_view


def _one_hot_names(encoder, categorical: list[str]) -> dict[str, list[str]]:
    """Output column names of the one-hot block, grouped by the feature that made them.

    Rebuilt from the encoder's own ``categories_``/``infrequent_categories_`` rather
    than by splitting names on ``_``: feature names here contain underscores and
    hyphens (``marital-status``, ``educational-num``) and levels do too, so prefix
    matching is ambiguous in principle. The caller checks the reconstruction against
    ``get_feature_names_out()``, which turns any drift in sklearn's naming into a loud
    failure instead of a silent mis-grouping.
    """
    infrequent = getattr(encoder, "infrequent_categories_", None)
    grouped: dict[str, list[str]] = {}
    for i, feature in enumerate(categorical):
        dropped = set() if infrequent is None or infrequent[i] is None else set(infrequent[i])
        names = [f"{feature}_{level}" for level in encoder.categories_[i] if level not in dropped]
        if dropped:
            names.append(f"{feature}_infrequent_sklearn")
        grouped[feature] = names
    return grouped


def encoded_groups(dataset: str, split_seed: int) -> dict[str, str]:
    """One-hot column -> parent feature, for the NAM's ``encoded`` view.

    Numeric columns map to themselves: they are already the unit they are read at.
    """
    bundle = joblib.load(paths.transformer(dataset, split_seed))
    encoder, numeric, categorical = bundle["encoder"], bundle["numeric"], bundle["categorical"]

    mapping = {column: column for column in numeric}
    for feature, columns in _one_hot_names(encoder.named_transformers_["cat"], categorical).items():
        for column in columns:
            mapping[column] = feature

    produced = list(encoder.get_feature_names_out())
    if sorted(mapping) != sorted(produced):
        missing = sorted(set(produced) - set(mapping))
        extra = sorted(set(mapping) - set(produced))
        raise AssertionError(
            f"Reconstructed one-hot names for {dataset} do not match the encoder's own: "
            f"{len(missing)} unmapped ({missing[:5]}), {len(extra)} invented ({extra[:5]}). "
            "sklearn's naming convention has changed; fix _one_hot_names rather than "
            "falling back to prefix matching, which would mis-group silently."
        )
    # Column order matters downstream: the map is keyed by name, but callers index by
    # position, so return it in the encoder's order.
    return {column: mapping[column] for column in produced}


def identity_groups(names: Iterable[str]) -> dict[str, str]:
    """Every term is its own unit -- the EBM case, including pair terms."""
    return {str(name): str(name) for name in names}


def group_map(dataset: str, model: str, split_seed: int, names: Iterable[str]) -> dict[str, str]:
    """The map for one family, over the term names that family actually produced.

    Every family on the ``encoded`` view -- the NAM and the linear baseline alike --
    carries one term per one-hot column and is grouped back to parent features. The
    EBM's native view is left alone.
    """
    if uses_encoded_view(model):
        mapping = encoded_groups(dataset, split_seed)
        unknown = [n for n in names if n not in mapping]
        if unknown:
            raise KeyError(
                f"{model} terms absent from {dataset}'s split{split_seed} encoder: "
                f"{unknown[:5]}. The model was fitted against a different feature view "
                "than the one on disk."
            )
        return {name: mapping[name] for name in names}
    return identity_groups(names)


def write(dataset: str, view: str, split_seed: int, mapping: dict[str, str]) -> str:
    path = paths.term_groups(dataset, view, split_seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    return str(path)

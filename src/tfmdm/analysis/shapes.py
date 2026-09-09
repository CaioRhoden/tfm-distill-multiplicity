"""Plan step 3.3 -- shape-function distance for numeric features, within a family.

The one *magnitude*-based metric in the set, and so the only one logit-scale drift can
bite (risk table; decision D2). It is reported both raw and normalised by the model's
own mean absolute total contribution: if the two disagree, the arms differ in scale
rather than in shape.

Two mechanics make this harder than "read the shape function off the model".

*The models do not share an x-axis.* Every EBM seed learns its own ``max_bins`` cut
points from its own bootstrap, so two seeds' step functions live on different
breakpoints and cannot be subtracted as stored. Curves are therefore *evaluated*, on a
grid shared by every seed in the cell, rather than read out of the model.

*The grid must sit where the data is.* NAM shape functions are wild where the training
marginal is thin, and ExU units sharpen that, so a grid spread evenly over the observed
range would measure multiplicity mostly in regions holding almost no test mass. The
grid is 100 quantiles of the *train* marginal between its 1st and 99th percentile,
which puts grid points where the rows are and trims both tails.

One-hot columns are excluded. A "shape function" over {0, 1} is a coefficient, not a
curve; the global top-k Jaccard already covers those.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.explain import TermContributions, term_contributions

N_GRID = 100
LOW_PERCENTILE, HIGH_PERCENTILE = 1.0, 99.0


def numeric_columns(x: pd.DataFrame, model: str, groups: dict[str, str]) -> list[str]:
    """The columns a shape function is defined on, per family.

    For the NAM the ``encoded`` view has already one-hot expanded the categoricals, so
    a numeric column is exactly one that maps to itself under the D4 grouping. For the
    EBM the view is native, so numeric dtype is the test -- and pair terms are excluded
    here regardless, being surfaces rather than curves.
    """
    if model == "nam":
        return [c for c in x.columns if groups.get(c) == c]
    return [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]


def quantile_grid(values: np.ndarray, n_grid: int = N_GRID) -> np.ndarray:
    """``n_grid`` quantiles of the train marginal, 1st to 99th percentile.

    Duplicates are kept rather than uniqued away: a feature that is constant over most
    of its mass should contribute grid points in proportion to that mass, which is what
    makes the averaged distance density-weighted rather than range-weighted.
    """
    probabilities = np.linspace(LOW_PERCENTILE / 100.0, HIGH_PERCENTILE / 100.0, n_grid)
    return np.quantile(np.asarray(values, dtype=float), probabilities)


def _background_row(x_train: pd.DataFrame) -> pd.DataFrame:
    """A single representative row, used only as a carrier for the varied feature.

    Both families are additive in the varied feature's own term, so the values of the
    other columns cannot affect the extracted curve -- ``curves`` asserts exactly that
    rather than leaving it as a claim. The row still has to be *valid* (an EBM will
    reject an unseen category), so it is the train median for numerics and the mode
    for everything else.
    """
    row = {}
    for column in x_train.columns:
        if pd.api.types.is_numeric_dtype(x_train[column]):
            row[column] = x_train[column].median()
        else:
            row[column] = x_train[column].mode().iloc[0]
    return pd.DataFrame([row])[x_train.columns]


def _probe_frame(background: pd.DataFrame, column: str, grid: np.ndarray) -> pd.DataFrame:
    frame = pd.concat([background] * len(grid), ignore_index=True)
    frame[column] = np.asarray(grid, dtype=float)
    return frame


def curves(
    learner, x_train: pd.DataFrame, columns: list[str], grids: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """One model's centred shape function for each requested column.

    Centred on the mean over the grid, which -- the grid being quantiles of the train
    marginal -- is the train-marginal mean of the term, i.e. exactly the centring D3
    specifies. Without it two behaviourally identical models differ by whatever
    constant each happened to park in its intercept.
    """
    background = _background_row(x_train)
    out: dict[str, np.ndarray] = {}
    for column in columns:
        grid = grids[column]
        terms: TermContributions = term_contributions(learner, _probe_frame(background, column, grid))
        if column not in terms.names:
            continue
        curve = terms.values[:, terms.names.index(column)]
        out[column] = curve - curve.mean()
    return out


def total_scale(learner, x: pd.DataFrame) -> float:
    """Mean absolute total contribution -- the normaliser for the distance (D2).

    Distilled models train on softer targets and produce smaller logits throughout;
    dividing by a model's own scale asks whether its *shape* moved, not whether its
    amplitude did.
    """
    terms = term_contributions(learner, x)
    return float(np.abs(terms.values.sum(axis=1)).mean())


def pairwise_distance(model_curves: list[dict[str, np.ndarray]], scales: list[float] | None = None
                      ) -> dict[str, float]:
    """Mean pairwise root-mean-square distance between centred curves.

    Averaged over columns first and pairs second, so a cell with many features is not
    dominated by whichever one happens to live on the widest logit range. When
    ``scales`` is given, each pair's distance is additionally divided by the mean of
    the two models' scales, giving the normalised companion.
    """
    n_models = len(model_curves)
    if n_models < 2:
        raise ValueError("A shape distance needs at least two models")

    raw: list[float] = []
    normalised: list[float] = []
    per_column: dict[str, list[float]] = {}

    for a in range(n_models):
        for b in range(a + 1, n_models):
            shared = sorted(set(model_curves[a]) & set(model_curves[b]))
            if not shared:
                continue
            distances = [
                float(np.sqrt(np.mean((model_curves[a][c] - model_curves[b][c]) ** 2)))
                for c in shared
            ]
            for column, distance in zip(shared, distances):
                per_column.setdefault(column, []).append(distance)
            raw.append(float(np.mean(distances)))
            if scales is not None:
                scale = 0.5 * (scales[a] + scales[b])
                if scale > 0:
                    normalised.append(raw[-1] / scale)

    result = {
        "mean_shape_distance": float(np.mean(raw)) if raw else np.nan,
        "n_shape_pairs": len(raw),
        "n_shape_columns": len(per_column),
    }
    if scales is not None:
        result["mean_shape_distance_normalized"] = (
            float(np.mean(normalised)) if normalised else np.nan
        )
    if per_column:
        worst = max(per_column, key=lambda c: float(np.mean(per_column[c])))
        result["most_unstable_shape"] = worst
        result["most_unstable_shape_distance"] = float(np.mean(per_column[worst]))
    return result

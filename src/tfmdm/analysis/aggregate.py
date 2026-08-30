"""Phase 5 -- turn 360 prediction files into the numbers the decision rule reads.

Two things this module is careful about.

First, every arm is evaluated on the *same* test rows in the *same* order. The
multiplicity metrics are disagreement rates over shared points; a silently reordered
or subsetted test set would produce numbers that look fine and mean nothing, so the
row indices are asserted equal across every file that gets combined.

Second, multiplicity is never reported on its own. A constant predictor has zero
ambiguity, so a drop is only evidence for the hypothesis when it is paired with the
AUROC it was bought at -- which is why every arm result carries both.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import log_loss

from .. import paths
from ..config import load
from ..metrics import bootstrap as boot
from ..metrics import multiplicity as mult
from ..metrics.performance import performance

PRIMARY_PAIRS = (("distilled", "hard"),)


def auroc_columns(y: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """AUROC for every column of ``probs`` at once, via the rank identity.

    A loop of ``roc_auc_score`` calls costs 30 sorts per bootstrap draw; ranking the
    whole matrix once costs one. That is the difference between a 2,000-draw interval
    finishing in seconds and in an hour.
    """
    y = np.asarray(y).astype(int)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.full(probs.shape[1], np.nan)
    ranks = rankdata(probs, axis=0)
    pos_rank_sum = ranks[y == 1].sum(axis=0)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@dataclass
class ArmResult:
    dataset: str
    model: str
    arm: str
    seeds: list[int]
    row_index: np.ndarray
    y_true: np.ndarray
    test_probs: np.ndarray          # (n_test, n_seeds)
    val_log_loss: np.ndarray        # (n_seeds,)
    metrics: dict = field(default_factory=dict)

    def rashomon_mask(self, eps: float) -> np.ndarray:
        return self.val_log_loss <= self.val_log_loss.min() + eps


def collect_arm(dataset: str, model: str, arm: str, seed_list: list[int]) -> ArmResult:
    test_columns, val_losses, found = [], [], []
    reference_rows: np.ndarray | None = None
    y_true: np.ndarray | None = None

    for seed in seed_list:
        path = paths.preds(dataset, model, arm, seed)
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        test = frame[frame["split"] == "test"].sort_values("row_index")
        val = frame[frame["split"] == "val"].sort_values("row_index")

        rows = test["row_index"].to_numpy()
        if reference_rows is None:
            reference_rows, y_true = rows, test["y_true"].to_numpy().astype(int)
        elif not np.array_equal(rows, reference_rows):
            raise AssertionError(
                f"{path.name} was evaluated on different test rows than seed {found[0]}. "
                "Multiplicity across a moving test set is not defined."
            )

        test_columns.append(test["prob"].to_numpy())
        val_losses.append(
            log_loss(val["y_true"].to_numpy().astype(int),
                     np.clip(val["prob"].to_numpy(), 1e-12, 1 - 1e-12), labels=[0, 1])
        )
        found.append(seed)

    if len(found) < 2:
        raise FileNotFoundError(
            f"Found {len(found)} prediction file(s) for {dataset}/{model}/{arm}; "
            "multiplicity needs at least two. Has the sweep run?"
        )

    return ArmResult(dataset, model, arm, found, reference_rows, y_true,  # type: ignore[arg-type]
                     np.column_stack(test_columns), np.asarray(val_losses))


def summarise_arm(result: ArmResult, cfg) -> dict:
    threshold = float(cfg.eval.threshold)
    n_boot = int(cfg.eval.n_boot)
    probs, y = result.test_probs, result.y_true
    n = probs.shape[0]

    full = mult.multiplicity(probs, threshold)
    disagree = mult.disagreement_matrix(probs, threshold)

    amb_ci = boot.bca_ci(
        lambda idx: float(disagree[idx].any(axis=1).mean()), n,
        mult.ambiguity_jackknife(disagree), n_boot=n_boot, seed=1,
    )
    disc_ci = boot.bca_ci(
        lambda idx: float(disagree[idx].mean(axis=0).max()), n,
        mult.discrepancy_jackknife(disagree), n_boot=n_boot, seed=2,
    )
    auroc_ci = boot.percentile_ci(
        lambda idx: float(np.mean(auroc_columns(y[idx], probs[idx]))), n,
        n_boot=min(n_boot, 500), seed=3,
    )

    per_seed = [performance(y, probs[:, j], threshold) for j in range(probs.shape[1])]
    summary = {
        "dataset": result.dataset, "model": result.model, "arm": result.arm,
        "n_seeds": len(result.seeds), "n_test": n,
        **full.as_dict(),
        **amb_ci.as_dict("ambiguity_"),
        **disc_ci.as_dict("discrepancy_"),
        **auroc_ci.as_dict("mean_auroc_"),
    }
    for key in ("auroc", "log_loss"):
        values = np.array([m[key] for m in per_seed])
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1))

    # Rashomon-filtered variant: only models within eps of the best validation loss.
    keep = result.rashomon_mask(float(cfg.eval.rashomon_eps))
    if keep.sum() >= 2:
        filtered = mult.multiplicity(probs[:, keep], threshold)
        summary["rashomon_n_models"] = int(keep.sum())
        summary["rashomon_ambiguity"] = filtered.ambiguity
        summary["rashomon_discrepancy"] = filtered.discrepancy
    return summary


def compare_arms(a: ArmResult, b: ArmResult, cfg) -> list[dict]:
    """Paired differences a - b for ambiguity and mean AUROC, on shared test points."""
    if not np.array_equal(a.row_index, b.row_index):
        raise AssertionError(f"{a.arm} and {b.arm} were evaluated on different test rows")

    threshold = float(cfg.eval.threshold)
    n_boot = int(cfg.eval.n_boot)
    n = a.test_probs.shape[0]
    da = mult.disagreement_matrix(a.test_probs, threshold)
    db = mult.disagreement_matrix(b.test_probs, threshold)
    y = a.y_true

    amb_interval, amb_p = boot.paired_bootstrap(
        lambda idx: float(da[idx].any(axis=1).mean()),
        lambda idx: float(db[idx].any(axis=1).mean()),
        n, n_boot=n_boot, seed=11,
    )
    auc_interval, auc_p = boot.paired_bootstrap(
        lambda idx: float(np.mean(auroc_columns(y[idx], a.test_probs[idx]))),
        lambda idx: float(np.mean(auroc_columns(y[idx], b.test_probs[idx]))),
        n, n_boot=min(n_boot, 500), seed=12,
    )

    base_amb = float(db.any(axis=1).mean())
    common = {"dataset": a.dataset, "model": a.model, "arm_a": a.arm, "arm_b": b.arm}
    return [
        {**common, "metric": "ambiguity", "p_value": amb_p,
         "relative_change": (amb_interval.point / base_amb) if base_amb > 0 else np.nan,
         **amb_interval.as_dict("delta_")},
        {**common, "metric": "mean_auroc", "p_value": auc_p, "relative_change": np.nan,
         **auc_interval.as_dict("delta_")},
    ]


def aggregate(datasets: list[str], models: list[str], arms: list[str]) -> dict:
    paths.ensure_dirs()
    summaries: list[dict] = []
    comparisons: list[dict] = []

    for dataset in datasets:
        cfg = load(dataset)
        seed_list = [int(s) for s in cfg.seeds]
        for model in models:
            collected: dict[str, ArmResult] = {}
            for arm in arms:
                try:
                    collected[arm] = collect_arm(dataset, model, arm, seed_list)
                except FileNotFoundError as exc:
                    summaries.append({"dataset": dataset, "model": model, "arm": arm,
                                      "error": str(exc)})
                    continue
                summaries.append(summarise_arm(collected[arm], cfg))

            for arm_a, arm_b in PRIMARY_PAIRS:
                if arm_a in collected and arm_b in collected:
                    comparisons += compare_arms(collected[arm_a], collected[arm_b], cfg)

    # Holm across the primary family: the four ambiguity differences (2 datasets x 2 models).
    primary = [c for c in comparisons if c["metric"] == "ambiguity"]
    if primary:
        flags = boot.holm([c["p_value"] for c in primary])
        for comparison, reject in zip(primary, flags):
            comparison["holm_reject"] = bool(reject)

    summary_frame = pd.DataFrame(summaries)
    comparison_frame = pd.DataFrame(comparisons)
    summary_frame.to_csv(paths.RESULTS / "arm_summaries.csv", index=False)
    comparison_frame.to_csv(paths.RESULTS / "comparisons.csv", index=False)
    (paths.RESULTS / "aggregate.json").write_text(
        json.dumps({"summaries": summaries, "comparisons": comparisons}, indent=2, default=float)
    )
    return {"summaries": summaries, "comparisons": comparisons}

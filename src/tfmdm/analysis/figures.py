"""Figures F1-F4 from the plan.

One rule runs through all of them: multiplicity never appears without the accuracy it
was traded against. F1 is the headline for exactly that reason -- it is the only view
in which "we reduced ambiguity" and "we did not pay for it" can be read at once.
"""

from __future__ import annotations

import itertools
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from .. import paths  # noqa: E402
from ..config import load  # noqa: E402
from ..metrics import multiplicity as mult  # noqa: E402
from .aggregate import collect_arm  # noqa: E402

ARM_COLOR = {"hard": "#4C72B0", "distilled": "#DD8452", "tabicl": "#8172B3"}
ARM_LABEL = {"hard": "Hard labels", "distilled": "Distilled (TabICLv2)", "tabicl": "TabICLv2"}
MARKER = {"ebm": "o", "nam": "s", "logreg": "^", "tabicl": "D"}


def _save(fig: plt.Figure, name: str, split_seed: int) -> str:
    out = paths.figures_dir(split_seed)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return str(path)


def f1_pareto(summaries: pd.DataFrame, split_seed: int) -> str:
    """Ambiguity against AUROC. A point that moves left without moving down supports H2."""
    datasets = sorted(summaries["dataset"].dropna().unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        subset = summaries[summaries["dataset"] == dataset]
        for _, row in subset.iterrows():
            if pd.isna(row.get("ambiguity")):
                continue
            ax.errorbar(
                row["ambiguity"], row["auroc_mean"],
                xerr=[[row["ambiguity"] - row["ambiguity_ci_low"]],
                      [row["ambiguity_ci_high"] - row["ambiguity"]]],
                yerr=[[row["auroc_mean"] - row["mean_auroc_ci_low"]],
                      [row["mean_auroc_ci_high"] - row["auroc_mean"]]],
                fmt=MARKER.get(row["model"], "o"), markersize=9, capsize=3,
                color=ARM_COLOR.get(row["arm"], "#888888"),
                label=f"{row['model'].upper()} · {ARM_LABEL.get(row['arm'], row['arm'])}",
            )
        ax.set_xlabel("Ambiguity (lower is better)")
        ax.set_ylabel("Mean test AUROC")
        ax.set_title(dataset)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(f"F1 — multiplicity against the accuracy it costs (split {split_seed})")
    return _save(fig, "F1_pareto", split_seed)


def f2_bars(summaries: pd.DataFrame, split_seed: int) -> str:
    datasets = sorted(summaries["dataset"].dropna().unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.5), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        subset = summaries[(summaries["dataset"] == dataset) & summaries["ambiguity"].notna()]
        subset = subset.sort_values(["model", "arm"])
        labels = [f"{r['model']}\n{r['arm']}" for _, r in subset.iterrows()]
        positions = np.arange(len(subset))
        ax.bar(positions - 0.2, subset["ambiguity"], width=0.4, label="Ambiguity",
               color=[ARM_COLOR.get(a, "#888") for a in subset["arm"]])
        ax.bar(positions + 0.2, subset["discrepancy"], width=0.4, label="Discrepancy",
               color=[ARM_COLOR.get(a, "#888") for a in subset["arm"]], alpha=0.55)
        tabicl = summaries[(summaries["dataset"] == dataset) & (summaries["model"] == "tabicl")]
        if not tabicl.empty:
            ax.axhline(float(tabicl.iloc[0]["ambiguity"]), ls="--", color=ARM_COLOR["tabicl"],
                       label="TabICLv2 ambiguity")
        ax.set_xticks(positions, labels, fontsize=8)
        ax.set_ylabel("Rate")
        ax.set_title(dataset)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"F2 — multiplicity by arm, TabICLv2 as reference (split {split_seed})")
    return _save(fig, "F2_multiplicity_bars", split_seed)


def f3_threshold(datasets: list[str], models: list[str], split_seed: int) -> str:
    thresholds = np.linspace(0.05, 0.95, 37)
    fig, axes = plt.subplots(len(datasets), len(models),
                             figsize=(5 * len(models), 3.5 * len(datasets)), squeeze=False)
    for i, dataset in enumerate(datasets):
        cfg = load(dataset, split_seed=split_seed)
        seed_list = [int(s) for s in cfg.model_seeds]
        for j, model in enumerate(models):
            ax = axes[i][j]
            for arm in ("hard", "distilled"):
                try:
                    result = collect_arm(dataset, model, arm, split_seed, seed_list)
                except FileNotFoundError:
                    continue
                curve = mult.threshold_curve(result.test_probs, thresholds)
                ax.plot(curve["threshold"], curve["ambiguity"],
                        color=ARM_COLOR.get(arm), label=ARM_LABEL.get(arm, arm))
            ax.axvline(0.5, ls=":", color="grey")
            ax.set_xlabel("Decision threshold")
            ax.set_ylabel("Ambiguity")
            ax.set_title(f"{dataset} · {model.upper()}")
            ax.legend(fontsize=8)
    fig.suptitle("F3 — does the effect survive away from threshold 0.5?")
    fig.tight_layout()
    return _save(fig, "F3_threshold_curve", split_seed)


def _importance_vectors(dataset: str, model: str, arm: str, split_seed: int,
                        seed_list: list[int]) -> list[dict]:
    vectors = []
    for seed in seed_list:
        path = paths.importances(dataset, model, arm, seed, split_seed)
        if path.exists():
            vectors.append(json.loads(path.read_text()))
    return vectors


def f4_explanation_stability(datasets: list[str], models: list[str], split_seed: int) -> str:
    """Do the explanations stabilise, not just the predictions?

    This is the figure that speaks to the 'natively provides explanations' half of the
    thesis: a model set can agree on decisions while disagreeing on why.
    """
    records = []
    for dataset in datasets:
        cfg = load(dataset, split_seed=split_seed)
        seed_list = [int(s) for s in cfg.model_seeds]
        for model in models:
            for arm in ("hard", "distilled"):
                vectors = _importance_vectors(dataset, model, arm, split_seed, seed_list)
                if len(vectors) < 2:
                    continue
                keys = sorted(set.intersection(*(set(v) for v in vectors)))
                matrix = np.array([[v[k] for k in keys] for v in vectors])
                for a, b in itertools.combinations(range(len(vectors)), 2):
                    rho = spearmanr(matrix[a], matrix[b]).statistic
                    records.append({"dataset": dataset, "model": model, "arm": arm,
                                    "spearman": float(rho)})

    frame = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * max(len(frame.groupby(['dataset','model','arm'])), 1)), 4.5))
    if frame.empty:
        ax.text(0.5, 0.5, "No importance files found", ha="center", va="center")
    else:
        groups = list(frame.groupby(["dataset", "model", "arm"]))
        ax.boxplot([g["spearman"].to_numpy() for _, g in groups], showfliers=False)
        ax.set_xticks(range(1, len(groups) + 1),
                      [f"{d}\n{m}·{a}" for (d, m, a), _ in groups], fontsize=7)
        ax.set_ylabel("Spearman ρ between seed pairs")
        ax.grid(axis="y", alpha=0.3)
        frame.to_csv(paths.results_dir(split_seed) / "explanation_stability.csv",
                     index=False)
    ax.set_title("F4 — stability of global feature importances across seeds")
    return _save(fig, "F4_explanation_stability", split_seed)


def run(datasets: list[str], models: list[str], split_seed: int) -> list[str]:
    paths.ensure_dirs(split_seed)
    summaries = pd.read_csv(paths.results_dir(split_seed) / "arm_summaries.csv")
    return [
        f1_pareto(summaries, split_seed),
        f2_bars(summaries, split_seed),
        f3_threshold(datasets, models, split_seed),
        f4_explanation_stability(datasets, models, split_seed),
    ]

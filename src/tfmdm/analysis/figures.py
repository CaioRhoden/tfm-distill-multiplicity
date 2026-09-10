"""Figures F1-F5 and F8 from the plan.

One rule runs through all of them: multiplicity never appears without the accuracy it
was traded against. F1 is the headline for exactly that reason -- it is the only view
in which "we reduced ambiguity" and "we did not pay for it" can be read at once.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import paths
from ..config import load
from ..metrics import explanation as expl
from ..metrics import multiplicity as mult
from . import grouping
from .aggregate import collect_arm

TOP_K = (3, 5)
# Arms F4 looks for on disk; the control is included so it can be read as the ceiling.
STABILITY_ARMS = ("hard", "distilled", "shuffled")

ARM_COLOR = {"hard": "#4C72B0", "distilled": "#DD8452", "tabicl": "#8172B3",
             "shuffled": "#937860"}
ARM_LABEL = {"hard": "Hard labels", "distilled": "Distilled (TabICLv2)", "tabicl": "TabICLv2",
             "shuffled": "Shuffled labels (control)"}
MARKER = {"ebm": "o", "nam": "s", "logreg": "^", "tabicl": "D"}
LINESTYLE = {"ebm": "-", "nam": "--", "logreg": ":", "tabicl": "-."}


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
    """Do the explanations stabilise, not just the predictions? (plan 3.2)

    Measured as the mean pairwise Jaccard of the top-k globally most important terms,
    not as a Spearman correlation over the full importance vector. Spearman ranks every
    term, which over the NAM's ~80 near-zero one-hot columns and the EBM's block of
    exactly-zero unselected pair terms is dominated by the arbitrary ordering of ties --
    two near-identical models can score near zero for no reason a reader would accept.
    A top-k set ignores the tied tail and asks only whether the terms someone would
    actually look at are the same ones.

    NAM importances are grouped to parent features first (D4). The on-disk vectors are
    per one-hot column, which is neither the granularity a NAM is read at nor the one
    the rest of the analysis uses.
    """
    records = []
    for dataset in datasets:
        cfg = load(dataset, split_seed=split_seed)
        seed_list = [int(s) for s in cfg.model_seeds]
        for model in models:
            for arm in STABILITY_ARMS:
                vectors = _importance_vectors(dataset, model, arm, split_seed, seed_list)
                if len(vectors) < 2:
                    continue
                keys = sorted(set.intersection(*(set(v) for v in vectors)))
                matrix = np.array([[v[k] for k in keys] for v in vectors])
                matrix, groups = _group_importances(dataset, model, split_seed, matrix, keys)

                row = {"dataset": dataset, "model": model, "arm": arm,
                       "split_seed": split_seed, "n_models": len(vectors),
                       "n_groups": len(groups)}
                for k in TOP_K:
                    row.update(expl.top_k_jaccard(matrix, k))
                records.append(row)

    frame = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * max(len(frame), 1)), 4.5))
    if frame.empty:
        ax.text(0.5, 0.5, "No importance files found", ha="center", va="center")
    else:
        frame.to_csv(paths.results_dir(split_seed) / "explanation_stability.csv", index=False)
        positions = np.arange(len(frame))
        for offset, k in zip((-0.2, 0.2), TOP_K):
            ax.bar(positions + offset, frame[f"mean_top{k}_jaccard"], width=0.4,
                   label=f"top-{k}", alpha=0.85 if offset < 0 else 0.55,
                   color=[ARM_COLOR.get(a, "#888") for a in frame["arm"]])
        ax.set_xticks(positions,
                      [f"{r.dataset}\n{r.model}·{r.arm}" for r in frame.itertuples()],
                      fontsize=7)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Mean pairwise top-k Jaccard (1 = identical)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    ax.set_title("F4 — stability of the globally most important terms across seeds")
    return _save(fig, "F4_explanation_stability", split_seed)


def _group_importances(dataset: str, model: str, split_seed: int,
                       matrix: np.ndarray, keys: list[str]) -> tuple[np.ndarray, list[str]]:
    """Sum a (n_models, n_terms) importance matrix onto the D4 units.

    Importance is a mean of *absolute* contributions, so summing a parent's levels is
    an upper bound on the parent's own mean-absolute contribution rather than equal to
    it. That is the right reduction here anyway: this figure ranks features against
    each other, and a feature's claim on a reader's attention is the total magnitude it
    moves the logit by, spread across its levels or not.
    """
    mapping = grouping.group_map(dataset, model, split_seed, keys)
    grouped, groups = expl.group_terms(matrix[:, None, :], keys, mapping)
    return grouped[:, 0, :], groups


def f5_explanation_against_prediction(split_seed: int) -> str:
    """E1 — is a model set less settled about *why* than about *what*? (plan hypothesis E1)

    Each point is one cell. The diagonal is the claim being tested: a point above it
    means the seeds re-attribute more rows than they re-decide, i.e. they agree on the
    decision while disagreeing on the reason. Predictive ambiguity is the floor the
    explanation metric has to clear, which is why the two share an axis rather than
    appearing in separate panels.
    """
    results = paths.results_dir(split_seed)
    explanations = pd.read_csv(results / "explanation_multiplicity.csv")
    summaries = pd.read_csv(results / "arm_summaries.csv")
    keys = ["dataset", "model", "arm", "split_seed"]
    merged = explanations.merge(summaries[keys + ["ambiguity"]], on=keys, how="inner")

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    merged = merged.dropna(subset=["ambiguity", "attribution_ambiguity"])
    if merged.empty:
        ax.text(0.5, 0.5, "No paired cells found", ha="center", va="center")
    else:
        for row in merged.itertuples():
            ax.scatter(row.ambiguity, row.attribution_ambiguity, s=90,
                       marker=MARKER.get(row.model, "o"),
                       color=ARM_COLOR.get(row.arm, "#888"),
                       label=f"{row.model.upper()} · {ARM_LABEL.get(row.arm, row.arm)}")
            ax.annotate(row.dataset, (row.ambiguity, row.attribution_ambiguity),
                        fontsize=7, xytext=(4, -8), textcoords="offset points")
        limit = max(1e-3, float(merged[["ambiguity", "attribution_ambiguity"]].max().max()) * 1.15)
        ax.plot([0, limit], [0, limit], ls="--", color="grey", lw=1)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=7, loc="upper left")
    ax.set_xlabel("Predictive ambiguity (disagree on the decision)")
    ax.set_ylabel("Attribution ambiguity (disagree on the top-1 reason)")
    ax.grid(alpha=0.3)
    ax.set_title(f"F5 — explanations against decisions (split {split_seed})")
    return _save(fig, "F5_explanation_vs_prediction", split_seed)


def f8_margin_sweep(split_seed: int) -> str:
    """3.4/D2 — does the arm difference survive away from the near-tie rows?

    Left: how much attribution margin the reference model actually has, as the share of
    rows surviving each epsilon. If the distilled arm's curve falls away faster, its
    importance profile is flatter, and *that alone* raises any rank metric without any
    increase in genuine disagreement.

    Right: ambiguity recomputed over only the surviving rows. If the gap between the
    arms closes as epsilon grows, the effect lived in the near-ties and is not a
    statement about explanations -- which is the E2 caveat in the plan's decision rule.
    """
    path = paths.results_dir(split_seed) / "explanation_margin_sweep.csv"
    sweep = pd.read_csv(path) if path.exists() else pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if sweep.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No margin sweep found", ha="center", va="center")
    else:
        for (dataset, model, arm), group in sweep.groupby(["dataset", "model", "arm"]):
            group = group.sort_values("epsilon")
            style = {"color": ARM_COLOR.get(arm, "#888"),
                     "ls": LINESTYLE.get(model, "--"),
                     "marker": MARKER.get(model, "o"), "ms": 4,
                     "label": f"{dataset}·{model.upper()}·{arm}"}
            axes[0].plot(group["epsilon"], group["share_kept"], **style)
            axes[1].plot(group["epsilon"], group["ambiguity"], **style)
        axes[0].set_ylabel("Share of rows with margin > ε")
        axes[1].set_ylabel("Attribution ambiguity on surviving rows")
        for ax in axes:
            ax.set_xlabel("ε (top-1 minus top-2 attribution, logits)")
            ax.set_xscale("symlog", linthresh=1e-3)
            ax.grid(alpha=0.3)
        axes[1].legend(fontsize=6, ncol=2)
    fig.suptitle(f"F8 — attribution margin and the ε-trimmed effect (split {split_seed})")
    fig.tight_layout()
    return _save(fig, "F8_attribution_margin", split_seed)


def run(datasets: list[str], models: list[str], split_seed: int) -> list[str]:
    """Every figure for one split.

    F5 and F8 read the explanation tables, so they are skipped -- not failed -- when
    ``tfmdm explanations`` has not been run for this split yet. The predictive figures
    do not depend on them and should still render.
    """
    paths.ensure_dirs(split_seed)
    summaries = pd.read_csv(paths.results_dir(split_seed) / "arm_summaries.csv")
    rendered = [
        f1_pareto(summaries, split_seed),
        f2_bars(summaries, split_seed),
        f3_threshold(datasets, models, split_seed),
        f4_explanation_stability(datasets, models, split_seed),
    ]
    if (paths.results_dir(split_seed) / "explanation_multiplicity.csv").exists():
        rendered.append(f5_explanation_against_prediction(split_seed))
        rendered.append(f8_margin_sweep(split_seed))
    return rendered

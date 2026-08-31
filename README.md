# tfm-distill-multiplicity

Does distilling a tabular foundation model into interpretable models reduce **predictive
multiplicity** without costing accuracy?

Two hypotheses:

- **H1** — TabICLv2 exhibits lower multiplicity than EBM/NAM trained on hard labels, at equal
  or better AUROC.
- **H2** — Replacing hard labels with TabICLv2's soft probabilities transfers that stability
  to the interpretable models.

The full design, decision rule, risk review and technical decisions live in
[plans/2026-08-30-tfm-distillation-multiplicity.md](plans/2026-08-30-tfm-distillation-multiplicity.md).
Read that first; this file only covers how to run things.

## Three things that shape the code

**The split seed is a dimension, not a constant.** Within one split the partition is frozen —
ambiguity and discrepancy are disagreement rates over a *common* test set, so it must not move.
The whole experiment is then repeated at five split seeds, and every replicate's artifacts live
under `artifacts/split{K}/` so they can never be mixed. `results/across_splits.csv` reports
whether the effect survives re-drawing the partition.

**Multiplicity is never reported alone.** A constant predictor has zero ambiguity. Every arm
result carries both its multiplicity and the AUROC it was bought at, and figure F1 plots them
against each other.

**TabICLv2 is deterministic, so it is perturbed like everything else.** For seed *s*, every
method — TabICLv2 included — sees a stratified bootstrap of the training set seeded with *s*.
Without this, "the foundation model has less multiplicity" would be a statement about the API
rather than the method.

**Soft labels are cross-fitted.** An in-context learner that can see a row in its own context
reproduces that row's label. Training-set soft labels are therefore produced out-of-fold
(5-fold, context = train minus fold), and an assertion fails the stage if they are not
strictly higher-entropy than in-context predictions on the same rows.

## Setup

```bash
uv sync --locked --all-groups        # or: task setup
uv sync --locked --extra tabicl      # once the TabICLv2 distribution is pinned
cp .env.example .env
```

Everything runs through [Taskfile.yml](Taskfile.yml); `task --list` shows the targets.

## Running it

```bash
task data              # Phase 1: clean, split 60/20/20, build feature views, per replicate
task sanity            # Phase 3 gates: degenerate metric check, reproduction
task tabicl:probe      # Phase 2.1: TabICLv2 feasibility and runtime probe
task tabicl:softlabels # Phase 2.2-2.3: cross-fitted TabICLv2 probabilities
task tabicl:preds      # Phase 2.4: the 30-model TabICLv2 set (baseline B3)
task tune              # Phase 4.2: one random search per (dataset, model, arm, split)
task slurm             # Phase 4.1: submit the sweep as an idempotent SLURM array
task analyze           # Phase 5: multiplicity, bootstrap intervals, Holm-corrected tests
task combine           # Pool the replicates into results/across_splits.csv
task figures           # F1-F4 into results/split{K}/figures/
```

Every target covers all five split replicates by default. Narrow it per invocation:

```bash
SPLIT_SEEDS='0' task data          # just the primary split
SPLIT_SEEDS='0 1 2' task all       # three replicates
SPLIT=3 task clean:split           # drop replicate 3 entirely
```

`task all` runs the whole chain in order. To try the pipeline end to end without TabICLv2
installed, set `TFMDM_TABICL_BACKEND=mock` — it substitutes a logistic regression teacher, and
is for plumbing checks only, never for a result.

## Inputs and outputs per task

Everything split-dependent lives under `artifacts/split{K}/` and `results/split{K}/`, keyed by
`SPLIT_SEEDS`. `{ds}` is the dataset, `{K}` the split seed.

**`data`** — Phase 1, one dataset end to end.
- reads `data/{dataset}/{dataset}.csv` — the raw survey/census extract, untouched by anything else.
- writes `data/interim/{ds}.parquet` — the raw CSV loaded and typed, before any cleaning.
- writes `data/processed/{ds}.parquet` + `_cleaning_report.json` — the cleaned frame (missing/duplicate handling applied) and what was done to it; split-independent, so every replicate reuses it.
- writes `artifacts/split{K}/splits/{ds}.json` — the frozen 60/20/20 train/val/test row indices for this replicate; this is the partition every downstream stage must agree on.
- writes `artifacts/split{K}/views/{ds}_{view}.parquet` + `_encoder.joblib` — the feature matrices each model family consumes (e.g. raw vs. one-hot) and the fitted encoder that produced them, so test-time rows can be transformed identically.
- writes `artifacts/split{K}/{ds}_data_summary.json` — row counts and class balance for this run, a quick sanity readout.

**`sanity`** — Phase 3 gates, run before any real sweep so a broken metric or model wiring is caught in minutes, not after a full sweep.
- reads `artifacts/split{K}/views/`, `splits/` — same feature views and partition `data` produced.
- writes `results/split{K}/sanity.json` — pass/fail for the degenerate check (identical seed ⇒ zero multiplicity) and the reproduction check (hard-label AUROC against published reference numbers).

**`tabicl:probe`** — Phase 2.1, a cheap dry run before committing to the full TabICLv2 grid.
- reads `artifacts/split{K}/views/{ds}_raw.parquet` — only 5% of train/test, primary split only.
- writes nothing; prints elapsed time and context size to stdout, to size the real jobs.

**`tabicl:softlabels`** — Phase 2.2–2.3, the teacher signal the `distilled` arm trains on.
- reads `artifacts/split{K}/views/{ds}_raw.parquet`, `splits/{ds}.json`.
- writes `artifacts/split{K}/softlabels/{ds}_tabicl_train_oof.parquet` — out-of-fold TabICLv2 probabilities for every training row (cross-fitted so a row's own label never leaks into its own prediction).
- writes `..._val.parquet` — TabICLv2 probabilities for the validation rows, context = full training set.
- writes `..._diagnostics.json` — the entropy-guard check (soft labels must be less confident than in-context predictions) plus OOF/val performance.

**`tabicl:preds`** — Phase 2.4, the TabICLv2 baseline (B3): 30 perturbed "model set" members, one per seed.
- reads `artifacts/split{K}/views/{ds}_raw.parquet`, `splits/{ds}.json`.
- writes `artifacts/split{K}/preds/{ds}_tabicl_incontext_s{seed}.parquet` — val/test predicted probabilities for that seed's bootstrapped context, in the same schema every trained model's predictions use, so `analyze` treats all methods uniformly.

**`tune`** — Phase 4.2, one hyperparameter search per (dataset, model, arm, split), shared by all 30 run seeds so search noise doesn't get counted as training-seed multiplicity.
- reads `artifacts/split{K}/views/`, `splits/`, and softlabels (only for the `distilled` arm's target).
- writes `configs/tuned/split{K}/{ds}_{model}_{arm}.yaml` — the winning hyperparameters, committed to the repo and read by every `train` run of that cell family.
- writes `..._trials.json` — every trial tried, for auditing the search.

**`train` / `train:group` / `sweep` / `slurm`** — Phase 4.1, fitting the interpretable models (EBM/NAM) on either hard labels or distilled soft labels.
- reads `artifacts/split{K}/views/`, `splits/`, softlabels (for `distilled`), `configs/tuned/split{K}/` (if a tuned config exists).
- writes `artifacts/split{K}/preds/{ds}_{model}_{arm}_s{seed}.parquet` — that seed's val/test predicted probabilities, the input `analyze` scores.
- writes `artifacts/split{K}/models/{ds}_{model}_{arm}_s{seed}.joblib` — the fitted model itself, for later inspection.
- writes `..._importances.json` — per-feature importances, when the model type supports them; used by figure F4 (explanation stability).

**`analyze`** — Phase 5.1–5.3, turns raw predictions into the study's actual measurements.
- reads `artifacts/split{K}/preds/*` — every arm's predictions for this split.
- writes `results/split{K}/arm_summaries.csv` — per (dataset, model, arm) multiplicity (ambiguity/discrepancy) and AUROC, with bootstrap intervals.
- writes `comparisons.csv` — hard-vs-distilled and interpretable-vs-TabICLv2 deltas, Holm-corrected p-values, and the H1/H2 decision flags.
- writes `aggregate.json` — the same content as one file, for programmatic reuse.

**`combine`** — pools replicates into the robustness readout: does the effect survive re-drawing the partition, or was it an artifact of one split?
- reads `results/split{K}/comparisons.csv` for every split in `SPLIT_SEEDS`.
- writes `results/across_splits.csv` — median/min/max delta and how many splits' intervals clear zero, per (dataset, model, metric).
- writes `results/all_comparisons.csv` — every split's comparisons concatenated, for custom slicing.

**`figures`** — renders the paper's headline plots for one split.
- reads `results/split{K}/arm_summaries.csv`, `artifacts/split{K}/preds/*_importances.json`.
- writes `results/split{K}/figures/` — F1 (multiplicity vs. AUROC pareto), F2 (bar comparison), F3 (threshold sensitivity), F4 (explanation stability) as PNGs, plus `explanation_stability.csv` backing F4.

## The sweep

One SLURM job is a **group** — all 30 run seeds of one (dataset, model, arm, split) — because
the expensive part of a cell is loading the view, the split and the soft labels, which every
seed shares. With five replicates that is 40 jobs covering 1,200 models.

```bash
task groups                    # inspect the grid the array will cover
task slurm                     # submit it
CHUNK=10 task slurm            # 120 smaller jobs instead, when the queue has room
task train:group -- --dataset adult --model ebm --arm hard   # one group, locally
task train -- --dataset adult --model ebm --arm hard --seed 0   # one cell, for debugging
```

The array index *is* the grid coordinate: `tfmdm groups --index $SLURM_ARRAY_TASK_ID` is called
by both the submitting shell and each worker, so there is no manifest that can drift. Any run
seed whose prediction file already exists is skipped, so a partially-failed array is resubmitted
with the identical command and resumes at the gaps.

## Layout

```
configs/         base.yaml, dataset/, model/, and tuned/split{K}/ (written by `task tune`)
src/tfmdm/
  data/          loaders, cleaning, the frozen split, the raw and encoded feature views
  softlabels/    the TabICLv2 adapter and the cross-fitting machinery
  models/        EBM, NAM (PyTorch port), logistic regression, behind one interface
  metrics/       performance, multiplicity (ambiguity/discrepancy), bootstrap + Holm
  stages/        one module per pipeline phase; each is a Taskfile target and a SLURM job
  analysis/      aggregation and figures F1-F4
tests/           metric identities, preprocessing correctness, cross-fitting, the grid
scripts/         sweep.slurm
```

Artifacts:

```
data/processed/{ds}.parquet        cleaned frame — shared, produced before any split exists
artifacts/split{K}/
  splits/      views/      softlabels/      preds/
results/split{K}/  arm_summaries.csv  comparisons.csv  figures/
results/across_splits.csv          the cross-replicate robustness readout
```

## Tests

```bash
task test
```

They cover the parts where being wrong is silent: the multiplicity metrics against
known-answer cases, the equivalence behind the weighted-duplication distillation trick, the
split and encoder behaviour, and that cross-fitting defeats a memorising teacher.

## Data

`data/adult/adult.csv` (48,842 × 14) and `data/Taiwan/Taiwan.csv` (30,000 × 23; note the
two-row header, handled in the dataset config). Raw files are never modified — `task clean`
removes only derived artifacts.

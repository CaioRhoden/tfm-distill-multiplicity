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

## Weights & Biases

Offline is the default, because cluster compute nodes usually have no outbound network and an
online-only logger either fails the job or blocks it. Runs land in `wandb/offline-run-*`.

```bash
task wandb:sync                    # push every offline run, from a node with network
WANDB_MODE=online task data        # or log live
task wandb:online -- analyze       # same, for any target
```

Every run records its git commit, `uv.lock` hash, dataset SHA256, seed and resolved config. A
run started from a dirty working tree is refused unless `ALLOW_DIRTY=1`.

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

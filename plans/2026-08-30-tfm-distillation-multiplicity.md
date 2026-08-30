# Experiment: Does distilling a tabular foundation model into interpretable models reduce predictive multiplicity?

## Research question
Do TabICLv2's predictions exhibit lower predictive multiplicity than EBM/NAM trained on
hard labels, and does replacing hard labels with TabICLv2's soft probabilities transfer that
stability to the interpretable models — **without losing AUROC**?

**Hypothesis:** (H1) TabICLv2 has lower ambiguity/discrepancy than hard-label EBM/NAM at equal
or better AUROC. (H2) EBM/NAM distilled from TabICLv2 soft probs land strictly inside the
(multiplicity, AUROC) Pareto frontier traced by their hard-label counterparts.

**Decision rule** (pre-registered here; applied by hand to `results/comparisons.csv`, not by
the pipeline). Significance only — no minimum effect size (see D8).
*Supports:* on **both** datasets and **both** model families, the paired 95% CI on
ΔAmbiguity = distilled − hard lies entirely below zero and survives Holm correction across the
four comparisons, while test AUROC is non-inferior — CI lower bound of ΔAUROC above −0.005.
*Refutes:* the ΔAmbiguity CI includes zero or lies above it, OR AUROC falls below the
non-inferiority margin.
*Inconclusive:* the CI clears zero in some cells but not others → report as dataset- or
model-dependent, do not claim H2.
*Reporting requirement:* every ΔAmbiguity interval is quoted with its relative change. No
sentence claims that distillation "reduces multiplicity" without the size of the reduction in
the same sentence.

## Setup
| | |
|---|---|
| Data | `data/adult/adult.csv` (48,842 × 14, binary income); `data/Taiwan/Taiwan.csv` (30,000 × 23, binary default, ~22% positive) |
| Method(s) | TabICLv2 (in-context, no gradient training); EBM (`interpretml/interpret`); NAM (PyTorch port of `google-research/neural_additive_models`) |
| Baselines | B0 logistic regression (trivial); B1 hard-label EBM; B2 hard-label NAM; B3 TabICLv2 itself |
| Primary metric | Test **AUROC** |
| Secondary metrics | **Ambiguity** and **discrepancy** (Marx et al. 2020) over the 30-model set. Log loss is computed but not reported: it is what model selection and the Rashomon filter use |
| Compute budget | SLURM cluster, GPU partition. Est. ≈ 8 GPU-h per split replicate, ≈ 40 GPU-h for all five (see Runs) |
| Seeds | 30 run seeds (`0..29`) per cell; 5 split replicates (`split_seed 0..4`) as the outer robustness loop |
| Loss | Baselines/controls: binary cross-entropy on hard labels. Distilled: **soft cross-entropy** `−[p·log q + (1−p)·log(1−q)]` against the TabICLv2 probability `p`; hard labels discarded entirely |

## Assumptions
- [ ] TabICLv2 is available as a pip-installable package with a scikit-learn-style
      `predict_proba`, and its context window admits ≥30k training rows on the target GPU.
      If not, contexts are subsampled to the largest feasible size and this becomes a
      documented limitation. **[DECIDE]** confirm package name/version before Phase 0.
- [ ] The NAM reference implementation is TF1; we reimplement in PyTorch (ExU + feature-net
      dropout + output penalty) and validate against published Adult AUROC ≈ 0.907 ± 0.005.
- [ ] `fnlwgt` (Adult) is a census sampling weight, not a predictive feature → dropped.
- [ ] Taiwan `EDUCATION ∈ {0,5,6}` and `MARRIAGE = 0` are undocumented → folded into "others".
- [ ] Sensitive attributes (`race`, `gender`, `SEX`) are kept as features; this study measures
      multiplicity, not fairness. **[DECIDE]** confirm — dropping them changes every run.

## Plan

### Phase 0 — Environment and scaffolding
- [ ] **0.1** `uv init`; pin Python 3.11; add `interpret`, `torch`, `scikit-learn`, `pandas`,
      `pyarrow`, `wandb`, `omegaconf`, `numpy`, `scipy`, `matplotlib`, TabICLv2. · *produces:*
      `pyproject.toml`, `uv.lock` · *done when:* `uv sync --locked` is clean on a login node
      and a GPU node.
- [ ] **0.2** `Taskfile.yml` with `setup`, `lock`, `data`, `softlabels`, `train`, `eval`,
      `figures`, `all`, `wandb:sync`, `clean`. Every task shells into `uv run`. · *done when:*
      `task --list` shows all targets and `task setup` succeeds from a clean clone.
- [ ] **0.3** W&B wrapper reading `WANDB_MODE ∈ {online, offline, disabled}`, project
      `tfm-distill-multiplicity`, run name `{dataset}-{model}-{arm}-s{seed}`, group `{dataset}-{model}`,
      job_type `{arm}`; offline dir `wandb/offline-run-*`. `task wandb:sync` runs
      `uv run wandb sync --sync-all wandb/` and, on success, marks synced runs. · *done when:*
      an offline smoke run appears in the web UI after `task wandb:sync`.
- [ ] **0.4** Every run logs config hash, `git rev-parse HEAD`, `uv.lock` hash, dataset SHA256,
      seed. · *done when:* a run refuses to start on a dirty working tree unless `ALLOW_DIRTY=1`.

### Phase 1 — Data (cheap, blocking for everything)
- [ ] **1.1** Loaders. Taiwan: the CSV carries a **two-row header** (`X1..X23` then real names)
      — read with `header=1`, drop `ID`. Adult: treat `?` as NA. · *produces:*
      `data/interim/{adult,taiwan}.parquet` · *done when:* shapes are (48842, 15) and (30000, 24)
      and a schema assertion passes.
- [ ] **1.2** Cleaning: drop exact duplicate rows **before** splitting (Adult has ~50 known
      duplicates → otherwise the same row can land in two splits); drop `fnlwgt`; drop `education` (redundant with
      `educational-num`); collapse the undocumented Taiwan categories; keep NA as an explicit
      `"Missing"` category rather than imputing. · *produces:* `data/processed/{ds}.parquet` +
      `data/processed/{ds}_cleaning_report.json` · *done when:* zero duplicate rows and the
      report's dropped-row count is logged.
- [ ] **1.3** Stratified 60/20/20 split → train/val/test, **frozen within a split seed** and
      identical across all 30 run seeds and both arms (ambiguity/discrepancy are defined
      pointwise on a common test set). Repeat for `split_seed ∈ 0..4`; each replicate's
      artifacts live under `artifacts/split{K}/` so replicates cannot be mixed. · *produces:*
      `artifacts/split{K}/splits/{ds}.json` · *done when:* per-split positive rate is within
      0.5pp of the pooled rate, the three index sets are provably disjoint, and two different
      split seeds produce different test sets.
- [ ] **1.4** Two feature views built with transformers **fit on train only**: `raw` (TabICLv2,
      categoricals as strings), `encoded` (NAM: standardised numerics + one-hot per categorical).
      EBM consumes `raw` directly. Both are written per split, since the encoder is fit on that
      split's training rows. · *produces:* `artifacts/split{K}/views/{ds}_{view}.parquet` +
      pickled fitted transformer · *done when:* a test asserts the scaler's mean equals the
      train mean and does **not** equal the pooled mean.

### Phase 2 — TabICLv2 soft labels (cross-fitted)
- [ ] **2.1** Feasibility probe on 5% of Adult: run TabICLv2 end-to-end, record wall-clock and
      peak GPU memory. · *done when:* extrapolated full-context runtime fits one SLURM job; if
      not, record the maximum feasible context size and proceed with subsampling.
- [ ] **2.2** **Cross-fitted** train soft labels: 5-fold stratified split *inside train*; for fold
      k, context = train∖k, predict on fold k. This is the step that prevents in-context
      memorisation from producing degenerate near-0/1 targets (see D2). · *produces:*
      `artifacts/split{K}/softlabels/{ds}_tabicl_train_oof.parquet` · *done when:* the mean of the soft labels is
      within 1pp of the train positive rate, and their entropy is **strictly greater** than that
      of in-context (non-cross-fitted) probs on the same rows — assert this explicitly.
- [ ] **2.3** Val soft labels: context = full train, predict on val. Test is **never** given soft
      labels; test is always scored against true labels. · *produces:*
      `artifacts/split{K}/softlabels/{ds}_tabicl_val.parquet` · *done when:* val AUROC is
      logged and is ≥ the trivial baseline.
- [ ] **2.4** TabICLv2 model set for B3: for `seed ∈ 0..29`, context = stratified bootstrap of
      train (same resampling protocol as every other method, see D3), predict on test. ·
      *produces:* `artifacts/split{K}/preds/{ds}_tabicl_incontext_s{seed}.parquet` ·
      *done when:* 30 files exist per split and
      pairwise-identical prediction vectors number 0.

### Phase 3 — Sanity phase (run before the real sweeps)
- [ ] **3.1** Degenerate check: 30 EBM fits with resampling disabled and seed fixed → ambiguity
      and discrepancy must both be exactly 0. · *done when:* metric code returns 0.0.
- [ ] **3.2** Single-seed reproduction: hard-label EBM and NAM on Adult hit AUROC within 0.01 of
      published values (EBM ≈ 0.927, NAM ≈ 0.907). · *done when:* both match, else stop and debug.

### Phase 4 — Main sweep
- [ ] **4.1** For each `(dataset, model ∈ {EBM, NAM}, arm ∈ {hard, distilled}, seed ∈ 0..29)`:
      resample train per D3, train, early-stop/select on **val** (val AUROC for `hard`; val soft
      cross-entropy against TabICLv2 probs for `distilled` — this is the "one validation step on
      the probs" you asked for), predict on the frozen test set. · *produces:*
      `artifacts/split{K}/preds/{ds}_{model}_{arm}_s{seed}.parquet`, one W&B run each ·
      *done when:* 240 prediction files exist per split (1,200 across five) and each logs its
      selected hyperparameters.
- [ ] **4.2** Hyperparameters fixed across run seeds and tuned once per (dataset, model, arm,
      split) on val with a small random search (≤20 configs, `tune_seed=1000`), so seed-to-seed
      spread reflects training randomness only. Retuned per split because the validation set
      moves with the partition. · *produces:* `configs/tuned/split{K}/*.yaml` · *done when:*
      configs are committed and referenced by path in every run.
- [ ] **4.3** The SLURM array submits one job per **group** — all 30 run seeds of one
      (dataset, model, arm, split) — so the view, split and soft labels load once rather than
      thirty times; `--chunk` trades that back for parallelism. Each run seed is skipped if its
      prediction file already exists. · *done when:* a killed and resubmitted array re-runs only
      the missing seeds.

### Phase 5 — Analysis
- [ ] **5.1** Metric module: ambiguity = fraction of test points where ≥1 of the 30 models
      disagrees with the reference (seed-0) model at threshold 0.5; discrepancy = max over models
      of the disagreement rate against that reference, reported alongside a reference-free max
      over all 435 model pairs. · *produces:*
      `src/metrics/multiplicity.py` + unit tests on hand-built toy sets with known answers ·
      *done when:* tests pass, including the all-identical (0.0) and all-opposite (1.0) cases.
- [ ] **5.2** Rashomon filter: report metrics both over all 30 models and over the subset within
      ε = 0.01 val log-loss of the best. · *done when:* both variants are in the results table.
- [ ] **5.3** Uncertainty: BCa bootstrap (2,000 resamples) over **test points** for each metric,
      and paired bootstrap for arm differences. Holm correction over the 4 primary comparisons
      (2 datasets × 2 models × 1 arm-pair). · *done when:* every headline number carries a CI.
- [ ] **5.4** Apply the decision rule above to `results/split{K}/comparisons.csv` **before** any figure
      polishing, and record the reading in the write-up. This is done by hand — the pipeline
      produces the numbers and stops there. · *done when:* each of the four cells has a stated
      supports/refutes/inconclusive reading, quoting `delta_point`, the paired CI,
      `relative_change` and `holm_reject`.
- [ ] **5.5** Cross-split robustness: pool the five replicates into `results/across_splits.csv`
      — per (dataset, model), the median and range of ΔAmbiguity and how many splits cleared
      zero. Splits are replicates, not extra hypotheses, so Holm is applied *within* a split and
      cross-split agreement is reported descriptively. · *done when:* the table exists and the
      `consistent_direction` column is quoted in the write-up.

## Runs
Per split replicate; multiply by 5 for the full outer loop.

| Run | Condition | Varies | Seeds | Est. time |
|---|---|---|---|---|
| R0 | Soft-label generation (2.2–2.3) | dataset × fold | — | 12 passes, minutes |
| R1 | B3 TabICLv2 on test | dataset × bootstrap | 30 | 60 passes, ~1 GPU-h |
| R2 | B1/B2 hard-label EBM, NAM | dataset × model | 30 | 4 groups; EBM ~1 CPU-h, NAM ~2 GPU-h |
| R3 | Distilled EBM, NAM (soft-only) | dataset × model | 30 | 4 groups, ~3 GPU-h |
| R4 | B0 logistic regression | dataset | 30 | 2 groups, minutes |

## Figures
| # | Question it answers | Type | x / y | Notes |
|---|---|---|---|---|
| F1 | Is distillation a free lunch? | Scatter, Pareto frontier | ambiguity / test AUROC | One point per (model, arm, dataset); 95% CI crosshairs; the headline figure |
| F2 | How large is the multiplicity gap? | Grouped bar + CI | arm / ambiguity & discrepancy | Facet by dataset; TabICLv2 as a horizontal reference line |
| F3 | Does it hold across the accuracy range? | Line | decision threshold / ambiguity | Guards against a threshold-0.5 artifact |
| F4 | Do explanations stabilise too? | Box | arm / rank correlation of feature importances between seed pairs | Supports the "natively provides explanations" claim in the README |

## Risks and mitigations
| Risk | Likelihood | Impact | Mitigation | Detect by |
|---|---|---|---|---|
| TabICLv2 is deterministic, so "TFM has less multiplicity" is trivially true | High | Fatal to H1 | Apply the identical bootstrap-resampling protocol to every method (D3); state plainly that a deterministic model has zero multiplicity by construction | Step 2.4's zero-identical-vectors check |
| Soft labels reduce multiplicity merely by smoothing the loss surface, not because they came from TabICLv2 | High | The observed effect is real but misattributed | **None in this design** — the control that would isolate this was cut (D4). Report the effect as "soft distillation from TabICLv2 reduces multiplicity" and never as "TabICLv2's knowledge does" | Not detectable within this experiment |
| Multiplicity traded for accuracy (a constant model has zero ambiguity) | Med | Invalidates the claim | Never report multiplicity alone — F1 Pareto plane and the AUROC non-inferiority clause in the decision rule | ΔAUROC CI |
| In-context memorisation makes train soft labels near-0/1 and useless | High | Distillation collapses to hard labels | 5-fold cross-fitting (2.2) + the explicit entropy assertion | Entropy check in 2.2 |
| 30 seeds too few for a small ambiguity difference | Med | Underpowered | Ambiguity is a per-test-point statistic over ~6–10k points, so CIs come from the point bootstrap, not from n=30; if CIs still straddle 0, raise to 50 seeds for the affected cell | Width of the 5.3 CIs |
| NAM reimplementation is subtly wrong | Med | Silences one model family | Phase 3.3 reproduces published AUROC before any sweep | 3.3 gate |
| Cluster quota / GPU queue delays | Med | Schedule | SLURM array with idempotent skip (4.3); EBM arm runs CPU-only and can proceed independently | Queue wait times |

## Technical decisions

### D1 — Partition frozen within a replicate, repeated across five replicates
**Chose:** a stratified 60/20/20 split held identical across all 30 run seeds and both arms
*within* a split seed, with the whole experiment repeated at `split_seed ∈ 0..4`.
**Over:** resampling the split per run seed (which would destroy the metric), or a single
partition with no outer loop (which would leave split sensitivity unmeasured).
**Because:** ambiguity and discrepancy are disagreement rates over a *common* set of test
points, so within a replicate the test set must not move — that is what isolates multiplicity
arising from the training procedure, which is the quantity distillation is claimed to reduce.
Across replicates, re-drawing the partition answers the separate and equally necessary
question of whether the effect is a property of the method or of which rows happened to land
in test. The two are kept apart by construction: each replicate's artifacts live in
`artifacts/split{K}/`, and nothing in the analysis pools predictions across replicates.
**Revisit if:** the five replicates agree closely — then the primary split alone suffices for
follow-up work, at a fifth of the cost.

### D2 — Cross-fitted (out-of-fold) soft labels
**Chose:** 5-fold cross-fitting inside the training set to produce TabICLv2 probabilities.
**Over:** a single pass with the full training set as context, predicting on those same rows.
**Because:** an in-context learner that has the target row in its context reproduces its label
almost exactly. The resulting probabilities would be near-degenerate, distillation would
degenerate into hard-label training, and H2 would be untestable through no fault of the
hypothesis. Cross-fitting costs 5× inference and buys honest targets.
**Revisit if:** the entropy assertion in 2.2 shows non-cross-fitted probs are already
well-calibrated — then a single pass is 5× cheaper.

### D3 — One randomness protocol for all methods
**Chose:** for seed *s*, every method sees a stratified bootstrap resample of the training set,
seeded with *s*, plus its own internal seed *s*.
**Over:** letting each method use whatever randomness it natively has (EBM bagging, NAM init,
TabICLv2 none).
**Because:** the H1 comparison is only meaningful if the model sets are generated by the same
perturbation. Otherwise "TabICLv2 has less multiplicity" reduces to "TabICLv2 is deterministic",
which is a statement about the API, not about the method.
**Revisit if:** the target claim shifts to init-only multiplicity — then hold the resample fixed
and vary initialisation alone, and report both.

### D4 — No smoothing control
**Chose:** compare the distilled arm against hard-label training only.
**Over:** a self-distillation control, in which each interpretable model is distilled from
cross-fitted probabilities produced by its own family.
**Because:** the two-arm design is a third smaller and simpler to run and to write up. The cost
is specific and must be stated wherever the result is: soft targets flatten the loss landscape
and shrink the effective hypothesis space regardless of where they come from, so this
experiment cannot separate "TabICLv2's knowledge stabilises the student" from "any soft target
stabilises the student". The defensible claim is therefore about the *pipeline* — distilling
from TabICLv2 reduces multiplicity — and not about the foundation model's knowledge being what
does the work.
**Revisit if:** a reviewer asks what the soft labels contribute beyond smoothing, or the
preliminary results become a full paper. Reinstating it costs one extra arm (120 runs,
≈17 GPU-h) and no new code paths beyond a second cross-fitting teacher.

### D5 — AUROC as the primary metric
**Chose:** test AUROC, with a −0.005 non-inferiority margin.
**Over:** accuracy@0.5 (distorted by Taiwan's ~22% positive rate) and log loss (a proper scoring
rule, but it is what the distilled arm directly optimises, which would bias the comparison
toward distillation).
**Because:** threshold-free, standard on both datasets, and independent of the training
objective. Nothing else is reported: a wide secondary panel invites reading whichever metric
happens to favour the hypothesis, and the decision rule only ever consults AUROC. Log loss is
still computed, but purely as machinery — model selection and the Rashomon filter need it.
**Revisit if:** the deployment framing becomes cost-sensitive — then switch to AUPRC or expected
cost at a fixed operating point; or if a reviewer challenges calibration, in which case add
Brier and ECE as an explicit follow-up rather than folding them into the primary result.

### D6 — Soft-probability-only distillation objective
**Chose:** train the student purely on TabICLv2 probabilities; discard hard labels.
**Over:** the standard blended `α·soft + (1−α)·hard` KD loss.
**Because:** a blend introduces α as a free knob that trades the two hypotheses against each
other and makes any observed multiplicity reduction unattributable. Soft-only is the clean test.
**Revisit if:** soft-only loses more than the non-inferiority margin on AUROC — then sweep
α ∈ {0.25, 0.5, 0.75} as a documented follow-up, reported separately from the primary result.

### D7 — uv + Taskfile + W&B offline-first
**Chose:** `uv` with a committed `uv.lock`; every entry point behind a Taskfile target; W&B in
offline mode on compute nodes with a separate `task wandb:sync` step.
**Over:** conda, and online-only logging.
**Because:** GPU nodes on most SLURM clusters have no outbound network, so an online-only logger
either fails the job or blocks it. Offline-first with an explicit sync keeps runs identical
whether or not the node is connected. The lockfile plus per-run git and lock hashes make any
figure traceable to an exact environment.
**Revisit if:** the cluster provides network on compute nodes — `WANDB_MODE=online` already works.

### D8 — Significance-only decision rule, with mandatory effect-size reporting
**Chose:** H2 turns on whether the paired CI for ΔAmbiguity clears zero after Holm correction,
with no minimum effect size attached.
**Over:** a pre-registered effect-size floor — either a fixed relative drop, or the fraction of
the hard-to-TabICLv2 ambiguity gap that distillation closes.
**Because:** any floor we could name would be invented rather than derived, and an arbitrary
threshold is harder to defend than none at all. The cost is real and is mitigated by the
reporting requirement rather than by the rule: with 9,758 test points on Adult the standard
error of an ambiguity near 0.20 is ≈0.004, and the paired bootstrap is tighter still, so a drop
of roughly 4% relative already clears zero. Statistical significance is close to free at this
sample size and does not by itself establish that the effect matters to anyone. Quoting the
relative change beside every interval keeps a small effect legible as a small effect.
**Revisit if:** the observed drops are large and unambiguous, in which case the choice is moot;
or a reviewer asks for a practical-relevance criterion, in which case the fraction-of-gap-closed
anchor is the one to adopt, since it is derived from B3 rather than asserted.

## Out of scope
- Pooling the five split replicates into a single significance test. They are reported as
  agreeing or disagreeing, not combined into one interval.
- Attributing the effect to TabICLv2 specifically rather than to soft targets in general;
  see D4. This experiment measures the pipeline, not the teacher's knowledge.
- Leakage auditing. Deduplication before splitting and train-only transformer fitting are
  done as a matter of correct preprocessing; the experiment runs no checks to confirm the
  absence of leakage and makes no claim about it.
- Calibration quality (Brier, ECE) and threshold-dependent performance (accuracy, AUPRC).
- Per-point score-level multiplicity, e.g. prediction variance and Rashomon Capacity —
  multiplicity here is decision-level only.
- Fairness or disparate-impact analysis of the multiplicity findings.
- Datasets beyond Adult and Taiwan; no claim of generality across tabular tasks.
- Comparison against non-interpretable strong baselines (XGBoost, CatBoost) — TabICLv2 stands in
  as the performance ceiling.
- Rashomon-set enumeration over hyperparameters; multiplicity here is seed/resample multiplicity
  at fixed hyperparameters.
- Human-subject evaluation of whether the resulting explanations are actually more useful.

# Experiment: Explanation multiplicity under distillation, within EBM and within NAM

## Research question
Do the *explanations* produced by a model set vary across equally-accurate members as much as
its *predictions* do, and does training on TabICLv2 soft labels reduce that variation the way
it reduces predictive multiplicity?

**Hypothesis:** (E1) explanation multiplicity is strictly larger than predictive multiplicity —
seed sets agree on decisions while disagreeing on attributions; (E2) distillation reduces
explanation multiplicity in the same direction it reduces predictive multiplicity.

**Scope decision:** EBM and NAM are **not** compared to each other. Each family is measured
against itself, hard arm vs distilled arm. This removes the feature-space mismatch, the
`outer_bags` ensembling confound, and the GA²M-vs-GAM structural mismatch in one stroke — none
of them can bias a within-family delta, because both arms of a delta share them exactly.

**Decision rule:** E1 holds if, per family and dataset, local-attribution ambiguity exceeds
prediction ambiguity by a bootstrap interval clear of zero in ≥3 of 5 splits. E2 holds if the
hard→distilled delta on the primary explanation metric has a Holm-corrected interval clear of
zero with the same sign as the predictive delta. E2 is **not reportable as a stability result** if the
attribution-margin diagnostic (D2) shows the arms differ mainly in how flat their importance
profiles are — a flatter profile makes ranks less determinate at identical disagreement.

## Setup
| | |
|---|---|
| Data | adult, Taiwan; frozen 60/20/20 per split seed, test set common to all arms (D1 of the parent plan) |
| Method(s) | Re-derive per-term contributions from the 30 saved models per cell in `artifacts/split{K}/models/`; no retraining except the shuffled-label control |
| Baselines | Predictive multiplicity of the same model set (the floor E1 must clear); identical-seed pair (must give exactly 0); shuffled-label NAM (must give near-maximal) |
| Primary metric | **Local attribution ambiguity** — fraction of test points where some seed disagrees with the reference on the top-1 attributed term. Rank-based, so exactly invariant to any per-model rescaling of contributions |
| Secondary metrics | Attribution discrepancy (max over seeds) and sign-flip rate — also rank/sign-based, also scale-invariant; global importance instability (top-k Jaccard, likewise); centred shape-function L2 distance (numeric features only) — the **only** magnitude-based metric, and the only one D2 applies to |
| Compute budget | Inference-only over 30 models × 4 cells × 2 datasets × 5 splits; one extra sweep cell for the shuffled-label control |

## Already on disk
- **Saved models are confirmed present**: `artifacts/split{0..4}/models/` each hold 240 `.joblib`
  files (2 datasets × 2 arms × 30 seeds × {ebm,nam}), matching the Setup table's compute path.
  Reload success is still unverified (1.1), but the files the plan depends on exist for every
  split.
- **Global term importances already computed for both families**: `models/registry.py:importances()`
  is called during training (`stages/train.py:130-134`) and written to
  `artifacts/split{K}/preds/{dataset}_{model}_{arm}_s{seed}_importances.json` for every
  (dataset, model, arm, split, seed) — 30 files per cell, all 4 cells, all 5 splits, both
  datasets. For the **EBM** this is `interpret`'s own `term_importances()`: mean-|contribution|
  per term, main and pair terms both present by name, and already centred by `interpret`'s own
  convention — i.e. already at exactly D4's EBM granularity, no grouping step needed. For the
  **NAM** it is `feature_importances_on`'s per-one-hot-column mean-|contribution| (`nam.py:195-199`)
  — **not** grouped to parent feature, so D4's grouping still has to be applied before this feeds
  any metric.
- **A first cut at the global metric already exists and is already known to be the wrong one**:
  `figures.py:f4_explanation_stability` (`_importance_vectors` + pairwise `spearmanr`) reads
  exactly these JSONs and computes the seed-pairwise Spearman correlation the plan's Risk table
  and 3.2 already flag as degenerate over the NAM's ~100 mostly-zero one-hot columns and the
  EBM's zero unselected-pair block. Nothing has been run end-to-end yet — `results/split{K}/`
  contains only the predictive-multiplicity artifacts (`arm_summaries.csv`, `comparisons.csv`,
  `seed_metrics.csv`, `aggregate.json`, `sanity.json`) plus an **empty** `figures/`; no
  `explanation_stability.csv`, `explanation_summaries.csv`, `explanation_comparisons.csv`, or
  F4/F8 output has been produced.
- **Not yet present, and Phase 1 is still the true bottleneck**: no per-point contribution
  matrices, `eval_terms`/`explain_local` dumps, or shape-function grids exist anywhere. The
  primary metric (D1: local top-1 attribution ambiguity) and the shape-function distance (3.3)
  both need those, and only those — the global-importance shortcut above does not touch them.
  Phase 1's CPU-reload check is correspondingly lower-risk for the EBM (a plain scikit-learn-style
  estimator, already reloading fine for `feature_importances()` calls at train time) than for the
  NAM (the actual `torch`/device risk named in the Assumptions section remains open).

## Assumptions
- [ ] Saved `.joblib` models reload with working `contributions()` / `eval_terms()` — the NAM
      pickles a `torch` module carrying a device attribute; CPU reload must be verified first.
- [ ] `interpret` 0.7.8's `eval_terms(X)` returns per-term logit contributions including pair
      terms; otherwise fall back to `explain_local`.
- [ ] Contributions are additive in **logit** space for both families, so a model's terms sum
      to its own logit. This is exact for the NAM (`nam.py` sums `parts` plus a bias, and
      `feature_dropout` is the identity under `eval()`); it must be checked for the EBM.
- [ ] `min_frequency=20` in the one-hot encoder means every NAM column maps to exactly one
      parent feature (including the `infrequent_sklearn` bucket).

## Plan
### Phase 1 — Establish the decomposition *(cheapest falsifying step)*
- [ ] **1.1** Reload one saved NAM and one saved EBM per dataset on CPU and extract a
      (n_test, n_terms) contribution matrix from each · *produces:* a probe module under
      `src/tfmdm/analysis/` · *done when:* for both, row sums of contributions plus the
      intercept reproduce `predict_proba` logits to 1e-4. If this fails, no metric below is
      meaningful and the plan stops here. The EBM side is lower-risk: `ebm.py:feature_importances`
      already reloads the saved `.joblib` and calls into `interpret` successfully at train time
      (that's how the on-disk importance JSONs got produced), so this step is really about
      swapping `term_importances()` for the per-point `eval_terms`/`explain_local` call, not about
      reload itself. The NAM's `torch`-module-on-CPU risk named in the Assumptions section is
      still fully open and unverified.
- [ ] **1.2** Handle the **inactive-level offset**: a NAM feature net for one-hot level ℓ is
      evaluated at x=0 on rows where ℓ is absent, and f_ℓ(0) ≠ 0 — every inactive level
      contributes a constant that is silently absorbed into an effective intercept ·
      *done when:* each column's contribution is centred on its train-set mean, the removed
      constants are folded into a reported effective intercept, and 1.1's identity still holds.
- [ ] **1.3** Define the **term grouping map** per family · *produces:*
      `views/{ds}_{view}_groups.json` · *done when:* every NAM one-hot column maps to its parent
      feature (grouping sums over *all* the parent's level nets, active and inactive, so it stays
      lossless), and every EBM term — main and pair — is listed as its own unit. The two maps
      need not agree; nothing compares across them.

### Phase 2 — Sanity gates
- [ ] **2.1** Degenerate check: two models fit with the identical seed give exactly zero on
      every explanation metric · *done when:* asserted in `tests/`.
- [ ] **2.2** Saturation check: a NAM trained on shuffled labels gives near-maximal attribution
      ambiguity · *done when:* value exceeds the hard-label cell by a visible margin.
- [ ] **2.3** Scale-invariance check: multiplying one seed's contributions by any positive
      constant must leave the primary metric, the discrepancy, the sign-flip rate and the top-k
      Jaccard *bit-identical*, and must leave the normalised shape distance unchanged while
      moving the raw one · *done when:* asserted in `tests/` as an equality, not a tolerance.
- [ ] **2.4** Centring check: two models differing only by a constant shift in one shape
      function score zero distance · *done when:* asserted in `tests/`.

### Phase 3 — Metric implementation
- [ ] **3.1** `metrics/explanation.py` mirroring `multiplicity.py`'s shape: a
      (n_points, n_terms, n_models) attribution tensor in, an `ExplanationMultiplicityResult`
      out · *done when:* 2.1–2.4 pass and the jackknife closed forms match a naive recompute on
      a 200-point sample.
- [ ] **3.2** Global metric: top-k Jaccard over grouped importances, k ∈ {3, 5}, replacing the
      `spearmanr` call inside `figures.py:f4_explanation_stability` · reuse `_importance_vectors`
      (it already reads the on-disk `*_importances.json` for every seed) as the input, insert the
      D4 parent-feature grouping for the NAM branch only (the EBM vectors are already at the
      right granularity), and swap the Spearman loop for top-k Jaccard over the (grouped) top-k
      sets · *done when:* it no longer degenerates on the NAM's ~100 near-zero one-hot columns,
      nor on the EBM's block of exactly-zero unselected pair terms — in both cases the tied
      ordering is arbitrary and Spearman reads it as signal · *done when:* the value differs from
      the Spearman one for both families and `explanation_stability.csv` is finally written for
      all 5 splits.
- [ ] **3.3** Shape-function distance for **numeric** features, per family: 100 quantile points
      of the train marginal between the 1st and 99th percentile. For the NAM the grid is mapped
      onto the standardised scale (the scaler is fit per split and shared by all 30 seeds, so it
      is constant within a cell). For the EBM the grid matters more: each seed learns its own
      `max_bins` cut points from its own bootstrap, so seeds' step functions live on *different*
      breakpoints and must be resampled onto the common grid before any distance is taken ·
      *done when:* 30 seeds' `age` curves plot on one axis for both families, and the EBM curves
      show visible step edges at differing positions. One-hot columns are excluded — a two-point
      "shape function" is a coefficient, covered by the global metric instead.
- [ ] **3.4** Attribution-margin diagnostic: per test point, the gap between the reference
      model's top-1 and top-2 absolute attributions; and ambiguity recomputed over only those
      points whose margin exceeds ε, swept over ε · *produces:* a column in
      `explanation_summaries.csv` plus figure F8 · *done when:* the hard and distilled arms'
      margin distributions can be compared directly, and the ε-sweep shows whether the E2 delta
      survives away from the near-tie points.

### Phase 4 — Measurement
- [ ] **4.1** Run over all cells and splits, writing `results/split{K}/explanation_summaries.csv`
      · *done when:* one row per (dataset, model, arm, metric) with BCa intervals.
- [ ] **4.2** Paired comparison against predictive multiplicity on the same model set (E1) and
      hard-vs-distilled deltas (E2), Holm-corrected over the metric family · *produces:*
      `explanation_comparisons.csv` with both raw and normalised deltas side by side.
- [ ] **4.3** Pool across splits into `results/across_splits.csv`'s existing schema ·
      *done when:* the ≥3-of-5 clause of the decision rule can be read off directly.

## Runs
| Run | Condition | Varies | Seeds | Est. time |
|---|---|---|---|---|
| R1 | Re-score saved EBM/NAM × {hard, distilled} × 2 datasets × 5 splits | family, arm, split | 30 | inference only |


## Risks and mitigations
| Risk | Likelihood | Impact | Mitigation | Detect by |
|---|---|---|---|---|
| **Flat importance profiles**: if distillation compresses the gaps between features, rank-based metrics get less determinate at identical underlying disagreement | High | Ambiguous E2 — "less stable" and "flatter" give the same number | Margin diagnostic and ε-sweep (3.4, D2, F8) | Margin CDFs differ between arms while the ε-trimmed delta vanishes |
| **Logit-scale drift**: distilled models train on softer targets and produce smaller contributions | High | None for the primary metric or any rank/sign metric — all are exactly invariant to per-model rescaling. Affects **only** the shape-function distance | Normalise that one metric by the model's mean absolute total contribution (D2) | Raw and normalised shape distances disagree |
| **Inactive-level offset (NAM only)**: f_ℓ(0) ≠ 0 for every absent one-hot level, adding arbitrary per-seed constants with no per-point meaning. The EBM is pre-centred by `interpret`, so this is asymmetric | High | **Corrupts the primary metric**, not just the distances: rank metrics are scale-invariant but *not* shift-invariant, so a large summed offset can make one parent feature win top-1 on every row regardless of the data | Centre every column on its train-set mean before grouping (1.2, D3) | A parent feature is top-1 on a near-100% share of points; the share moves when centring is toggled |
| **Rank-metric degeneracy**: Spearman over ~100 mostly-zero importances is dominated by tie-breaking noise | High | The current F4 is likely already misleading | Replace with top-k Jaccard (3.2) | Spearman near zero for two near-identical models |
| **Extrapolation noise**: NAM shape functions are wild where data is sparse, and ExU units sharpen this | Medium | Multiplicity dominated by regions with no test mass | 1st–99th percentile grid, density-weighted | Metric drops sharply when the grid is trimmed |
| **Early-stopping variance**: seeds stop at different epochs, so effective capacity varies | Medium | None — this is genuine procedure multiplicity, not noise | Record stopping epoch per seed; report its spread alongside | Correlation between stopping epoch and attribution distance |
| **Pair-term churn in the EBM**: the top-`2x` interaction set is re-ranked per bootstrapped seed, so seeds can select different pairs | Medium | Within-EBM only; a term absent from one seed has no counterpart to compare | Treat an unselected pair as a zero contribution, not a missing value | Selected-pair Jaccard across seeds well below 1 |
| **Multiple comparisons**: several metrics × 2 arms × 2 families × 2 datasets | Certain | Inflated significance | One pre-declared primary metric; Holm over the rest | — |
| **NAM unpickling** breaks on a different device or torch version | Medium | Phase 1 blocks | Probe first (1.1); fall back to re-fitting split 0 | 1.1 |

## Technical decisions
### D1 — Primary metric is local top-1 attribution ambiguity, not shape-function distance
**Chose:** the fraction of test points where some member of the model set attributes the
decision primarily to a different term. **Over:** an averaged distance between shape functions.
**Because:** it is the direct analogue of Marx et al.'s ambiguity — a per-point count of
"could this individual have been handed a different explanation by an equally good model" — so
it lands on the same axis as the predictive metric the study already reports, which is what
makes figure F5 and hypothesis E1 possible at all. A curve distance is an aggregate with no
individual-level reading. **Revisit if:** one term dominates top-1 in every seed, in which case
move to top-k set disagreement.

### D2 — Guard the rank metrics with an attribution margin, and normalise only the distance metric
**Chose:** report, alongside every rank-based metric, the distribution of the reference model's
top-1-minus-top-2 attribution gap, plus ambiguity restricted to points whose margin exceeds ε.
Separately, normalise the shape-function distance by each model's mean absolute total
contribution. **Over:** normalising everything, or normalising nothing.

**Because:** the primary metric and its rank/sign-based companions are *exactly* invariant to a
per-model rescaling — multiplying one seed's contributions by any positive constant cannot
change which term is largest, the full importance ordering, or any sign. Normalising them is a
no-op, and treating logit shrinkage as their main threat would be guarding the wrong flank. The
real sensitivity of a rank metric is to the *gaps*, not the scale: when two terms are nearly
tied, arbitrarily small seed-to-seed noise flips the winner, and a point counts as ambiguous
for a reason with no interpretive content. That matters here because distillation plausibly
flattens importance profiles, which would raise rank instability without any increase in genuine
disagreement. The margin diagnostic separates the two readings; the ε-sweep is the same guard
F3 already applies to the decision threshold. The magnitude-based shape distance is the one
metric where logit shrinkage does bite, and normalisation is the right fix there.

**Revisit if:** margins turn out to be large everywhere, in which case the near-tie concern is
empty and the diagnostic can be reported once and dropped.

### D3 — Centre every shape function on the train marginal
**Chose:** subtract each term's train-set mean contribution before any comparison, folding the
constants into a reported effective intercept. **Over:** comparing contributions as returned.
**Because:** an additive model identifies its shape functions only up to a constant absorbed by
the intercept, so two behaviourally identical models can differ arbitrarily in raw
contributions. This is the decision the primary metric actually rests on, and it is asymmetric
across the two families. `interpret` already centres the EBM — each graph is centred so that
the average prediction on the train set is zero, with the per-term means folded into the
intercept — so its terms arrive usable. The NAM centres nothing, and its one-hot level nets
contribute f_ℓ(0) on every row where that level is *absent*. Crucially this is not confined to
the magnitude-based metrics: a rank metric is invariant to rescaling but not to a shift, so an
uncentred parent feature carrying a large summed offset can rank top-1 on every test point for
no data-driven reason. **Revisit if:** a NAM variant is adopted that constrains each feature net
to zero mean on the training data, which would make the step a no-op — but verify rather than
assume, as no rank metric is shift-invariant by construction.

### D4 — Group NAM one-hot levels to parent features; keep EBM terms as-is
**Chose:** for the NAM, sum contributions over all of a parent's level columns; for the EBM,
treat each term including pairs as its own unit. **Over:** forcing a common granularity.
**Because:** nothing is compared across families any more, so each family should be measured at
the granularity its own explanation is read at — a NAM user reads "occupation", not
"occupation_Sales", while an EBM user genuinely reads the pair heatmaps. The NAM grouping is
lossless because exactly one level is active per row and the inactive constants are removed by
D3. **Revisit if:** the question becomes which *level* drives a decision.

### D5 — Explanation metrics use the same frozen test set as the predictive ones
**Chose:** reuse the split's test rows. **Over:** a fresh evaluation sample. **Because:** F5
pairs the two metrics point-for-point; a different sample would make that comparison approximate
for no gain. **Revisit if:** the test set is deemed over-used, in which case both metrics move
to a held-out portion together.

## Out of scope
- **Any EBM-vs-NAM comparison.** The two differ in feature space (12 native terms vs ~100
  one-hot columns), in structure (GA²M with `interactions: 2x` vs a strictly univariate GAM),
  and in internal ensembling (`outer_bags=8` vs a single network). A gap between them would be
  a statement about those three choices, not about the model families.
- SHAP/LIME-style post-hoc attributions. Both families are exactly additive, so their own
  decomposition is the ground truth; an approximator would add its own multiplicity.
- Faithfulness or human-usefulness of the explanations — only their stability under an
  arbitrary choice among equally-accurate models.
- TabICLv2 explanation multiplicity: no additive decomposition, so it cannot enter without a
  post-hoc attributor.
- Counterfactual or recourse multiplicity.

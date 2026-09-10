# Implementation: Logistic Regression as a measured family (B0)

## What exists and what is missing

`logreg` is already half-wired. It has a learner (`src/tfmdm/models/logreg.py`), a config
(`configs/model/logreg.yaml`), a registry entry, and it is an accepted `--model` value for
`tune`, `train` and `train-group`. Nothing downstream of training knows about it:

| Stage | State |
|---|---|
| tune | works as-is; search space is thin (`C` only) |
| train / train-group | works as-is; models and preds are written |
| `analyze` (predictive multiplicity) | **model-agnostic** — needs only the name passed in |
| `explain-probe` / `explanations` | **fails**: `models.explain.term_contributions` raises `NotImplementedError`, since a `LogRegModel` has neither `model.eval_terms` nor `.net` |
| `grouping.group_map` | returns identity for anything that is not `nam` — wrong for `logreg`, which is on the `encoded` view |
| `shapes.numeric_columns` | same: the `model == "nam"` branch is really an *encoded-view* branch |
| CLI / Taskfile / figures defaults | `logreg` absent from `INTERPRETABLE`, `MODELS`, F4 line styles |
| `feature_importances` | returns `|coef|`, which is not the mean-|contribution| every other family reports |

So the work is: one new decomposition branch, three `model == "nam"` predicates
generalised to a view predicate, a richer search space, and plumbing.

## Scope decision (extends the parent plan's)

`logreg` is measured **against itself** (hard vs distilled), exactly like EBM and NAM. That
is hypothesis E2 for a third family and needs no new machinery.

One *cross-family* comparison is additionally legitimate here and is not for EBM-vs-NAM:
**logreg vs NAM share the `encoded` view, the same one-hot columns, the same scaler, the
same strictly-univariate additive structure and the same single-model (non-ensembled)
fitting**. The only thing that differs is whether each feature's shape function is a
straight line or a learned curve. A logreg-vs-NAM delta is therefore attributable to
*flexibility*, which is a real reading — and the reason the parent plan bans EBM-vs-NAM
(feature space, structure, `outer_bags`) does not apply. This comparison is **secondary and
declared in advance**; the primary logreg result stays the within-family hard→distilled
delta. Nothing is compared between logreg and EBM (different feature space).

logreg also keeps its original role as baseline B0: if a NAM's multiplicity matches a
linear model's, the dataset is not exercising the method.

## Assumptions to verify before anything else
- [ ] A fitted `LogRegModel` reloads from `.joblib` and its per-column contribution
      `coef_j * x_j` plus `intercept_` reproduces `predict_proba` logits to well under
      `RESIDUAL_TOLERANCE` (1e-5, probability scale). This is exact in float64 and is the
      cheapest possible version of plan step 1.1 — if it fails, something is wrong with the
      *harness*, not the model.
- [ ] `OneHotEncoder` is built without `drop=`, so the dummy block is collinear. With
      L2 (`penalty="l2"`, the default) the optimum is still unique, so coefficients are
      determinate; **with `penalty=None` it is not** and seed-to-seed coefficient
      multiplicity would be an artifact of the parameterisation, not of the data.
      Consequence: `penalty=None` must stay out of the search space (see D-L2).
- [ ] `max_iter: 2000` actually converges on both datasets in both arms — the distilled arm
      doubles the row count via `expand_soft_targets`, and lbfgs failing to converge would
      inject optimiser noise that reads as multiplicity.

## Phase 1 — Decomposition (gates everything)
- [x] **1.1** Add `_logreg_terms(learner, x)` to `src/tfmdm/models/explain.py`:
      `values = x[columns].to_numpy() * coef_.ravel()`, `names = learner.columns`,
      `orders = [1] * n`, `intercept = float(intercept_[0])`. Dispatch in
      `term_contributions` on `hasattr(learner.model, "coef_")`, placed **after** the
      `eval_terms` check (an EBM has no `coef_`, but order the checks defensively) ·
      *done when:* `tfmdm explain-probe --models logreg` passes on both datasets, both
      arms.
- [x] **1.2** Generalise the three `model == "nam"` predicates to a **view** predicate.
      Introduce one helper — `config.model_view(model)` reading `configs/model/{m}.yaml`, or
      simply an `ENCODED_VIEW_MODELS = {"nam", "logreg"}` constant in `grouping` — and use
      it in `grouping.group_map:88`, `shapes.numeric_columns:44`,
      `explanations.collect_cell:161-162` and `explanations.run:538` (the `"encoded" if
      model == "nam" else "raw"` group-file name) · *done when:* logreg's ~80 one-hot
      columns group to ~12 parent features, and `views/{ds}_encoded_groups.json` is written
      once per view, not once per model.
- [x] **1.3** Replace `LogRegModel.feature_importances` with
      `feature_importances_on(x)` returning `mean(|coef_j * x_j|)` per column, matching what
      `registry.importances` already prefers and what the NAM does. `|coef|` alone is not
      comparable across columns with different marginals even after standardisation (a
      rare one-hot level has tiny variance and a large coefficient) · *done when:* the
      on-disk `*_importances.json` for logreg is on the same footing as the other families,
      and the top-k Jaccard in F4 is not dominated by rare levels.

## Phase 2 — Sanity gates (mirror plan gates 2.1–2.4, in `tests/`)
- [x] **2.1** Degenerate: two logreg fits at the identical seed give exactly 0 on every
      explanation metric. Stronger here than for the other families — logreg is
      deterministic given its data, so with `resample.bootstrap` off, *all 30 seeds* must
      collapse to zero multiplicity. Assert that as a separate case; it is a free check on
      the whole seeding protocol.
- [ ] **2.2** Saturation (needs the trained `shuffled` arm on disk): the `shuffled` arm gives near-maximal attribution ambiguity.
- [x] **2.3** Reconstruction: `values.sum(1) + intercept == logit(predict_proba)` to 1e-10
      on a synthetic frame — an equality, not a tolerance, since it is exact algebra.
- [x] **2.4** Grouping: summing one-hot contributions to the parent feature is lossless,
      i.e. the grouped tensor's row sums equal the ungrouped ones.

Note on **centring (D3)**: for logreg, `E[coef_j * x_j]` over the train set is `coef_j *
mean(x_j)`, which is ~0 for standardised numerics but *not* for one-hot columns (mean =
level frequency). Centring is therefore not a no-op and must not be skipped; the existing
`collect_cell` already applies it on the train marginal, so no code change — just do not
"optimise it away" for the linear case.

## Phase 3 — Hyperparameter tuning
`stages/tune.py` needs **no change**: it random-samples `n_configs` from `search_space`
per (dataset, model, arm, split) at the fixed `tune.seed`, exactly as required. Only
`configs/model/logreg.yaml` changes.

```yaml
name: logreg
view: encoded
params:
  C: 1.0
  penalty: l2
  solver: lbfgs
  max_iter: 5000          # raised: the distilled arm fits on 2x rows
  tol: 0.0001
search_space:
  C: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
  # class_weight deliberately absent -- see the risk table
```

- [x] **3.1** Widen the space as above · *done when:* `configs/tuned/split{K}/adult_logreg_{arm}.yaml`
      exists for both arms and its `val_objective` is finite.
- [x] **3.2** `tune.n_configs` is 20 against a space of 12 combinations, sampled **with
      replacement** — so ~10 distinct configs get evaluated and 10 fits are wasted. Either
      pass `--n-configs 12` for logreg, or (better, and it helps every family) dedupe
      sampled configs inside `_sample`/the trial loop and stop at `n_configs` *distinct*
      draws · *done when:* the trials JSON has no duplicate `params`.
- [x] **3.3** Assert convergence: after each fit, check `n_iter_ < max_iter` and fail loudly
      otherwise. A silently-truncated lbfgs is the one way this family can manufacture
      multiplicity out of nothing · *done when:* logged in the tune trials and asserted in
      `_train_one` (or in `LogRegModel.fit`, which is where the information is).

## Phase 4 — Training and saving
No code change. `train.run_group` is already generic: it applies the tuned config, runs the
stratified bootstrap (`seeds.stratified_bootstrap_indices`), fits, writes
`preds/…_s{seed}.parquet`, dumps `models/….joblib`, and writes the importances JSON via
`registry.importances` (which will now pick up `feature_importances_on` from 1.3).

- [ ] **4.1** `tfmdm train-group --model logreg --arm {hard,distilled,shuffled}` × 2 datasets
      × 5 splits · *est.:* seconds per fit, minutes for the whole grid — negligible next to
      EBM/NAM, so it can run locally and does not need the SLURM array.
- [x] **4.2** Add `logreg` to the sweep grid so it is not forgotten: `MODELS: ebm nam logreg`
      in `Taskfile.yml`, and to `cli.INTERPRETABLE` (rename it `MEASURED` — logreg is
      interpretable in the same sense, so the name is fine either way, but the three
      `INTERPRETABLE + ["logreg"]` spellings in `cli.py:96,103,112` collapse to one list).

## Phase 5 — Evaluation and joining the existing results
- [ ] **5.1 Predictive multiplicity.** `analysis.aggregate` is already model-agnostic; it
      only needs `logreg` in the `--models` default. Rows land in the same
      `results/split{K}/seed_metrics.csv`, `arm_summaries.csv` and `comparisons.csv`
      keyed by `model`, so nothing merges or collides · *done when:* `arm_summaries.csv`
      has `model == "logreg"` rows for both arms with ambiguity, discrepancy and AUROC.
- [ ] **5.2 Explanation multiplicity.** After Phase 1, `tfmdm explanations --models logreg`
      runs unchanged and merges into `explanation_summaries.csv` /
      `explanation_comparisons.csv` via the existing `_merge_into` keys · *done when:*
      one row per (dataset, logreg, arm, metric) with BCa intervals.
- [ ] **5.3 Multiple-comparison bookkeeping.** Adding a family adds a third hard→distilled
      delta per (dataset, metric). `_holm_within` corrects within a declared family of
      hypotheses — confirm logreg's deltas enter the same Holm family as EBM's and NAM's
      (they should: same hypothesis E2, three families now) rather than being corrected
      separately, which would understate the correction · *done when:* the Holm rank
      denominators in `explanation_comparisons.csv` increase accordingly.
- [ ] **5.4 The logreg-vs-NAM secondary comparison.** `compare_arms` compares two cells;
      the model-matched, arm-matched version is the same call with `model` varying instead
      of `arm`. Declare it secondary, Holm-correct it in its own family, and report it with
      the flexibility reading above · *done when:* it appears in `explanation_comparisons.csv`
      under a distinct `hypothesis` label, not mixed into E2.
- [ ] **5.5 Figures.** `MARKER` already has `"logreg": "^"`. `figures.py:273` hardcodes
      `"-" if model == "ebm" else "--"` for two families — make it a `LINESTYLE` dict
      keyed like `MARKER`. Add `logreg` to the `figures` and `analyze` CLI defaults and to
      `Taskfile.yml`'s `MODELS`.
- [ ] **5.6 Sanity reference.** `sanity.REFERENCE_AUROC` has entries for adult/ebm and
      adult/nam. Add `("adult", "logreg")` once a first run establishes the value (~0.90 on
      adult is the usual figure for a regularised linear model on the one-hot view), so a
      regression in the feature pipeline is caught for this family too.

## Risks specific to this family

| Risk | Likelihood | Impact | Mitigation | Detect by |
|---|---|---|---|---|
| **Shape-function metric degenerates.** A logreg shape function is a straight line, so the centred curve on the quantile grid is `coef_j * (grid - mean)` and the RMS distance between two seeds collapses to `|Δcoef_j| * sd(grid)` | Certain | Not a bug — it is the correct value — but it makes the shape distance and the global importance metric near-redundant for this family | Report it, do not treat the agreement between the two as corroboration | The two metrics correlate ~1 within logreg cells and not within NAM cells |
| **Collinear one-hot block.** No `drop=` in the encoder; with an unpenalised fit the coefficient vector is not identified | Low (default is L2) | Would fabricate explanation multiplicity with zero predictive multiplicity — the exact failure mode the study is about | Keep `penalty=None` out of the search space; keep `C` bounded above | Huge coefficient multiplicity alongside ~0 prediction disagreement |
| **Non-convergence under the distilled arm.** `expand_soft_targets` doubles the rows and weights them, which slows lbfgs | Medium | Optimiser noise reads as multiplicity, and asymmetrically across arms | `max_iter: 5000` plus the `n_iter_` assertion (3.3) | `n_iter_` at the cap for some seeds |
| **Floor is too low to be informative.** logreg may show *higher* multiplicity than the NAM (a high-bias model can still be coefficient-unstable under collinearity), inverting the intended "baseline" reading | Medium | The B0 framing ("if the NAM matches it, the dataset is not exercising the method") stops applying | Report it as a finding, not as a gate; the E2 within-family delta is unaffected | logreg ambiguity above NAM ambiguity in the same cell |
| **`class_weight: balanced` interacts with the soft arm.** It computes weights from the *doubled, relabelled* frame in `expand_soft_targets`, where the class balance is by construction near 50/50 | Medium | The tuned hyperparameters would mean different things in the two arms | **Resolved:** `class_weight` is kept out of the search space entirely | — |

## Technical decisions

### D-L2 — Keep the fit penalised, always
**Chose:** `penalty: l2` fixed, `C` swept over six decades. **Over:** including `penalty:
None` or `l1`. **Because:** the one-hot block is collinear (no `drop=` in the encoder), and
an unpenalised logistic fit on a collinear design has a non-unique coefficient vector — two
seeds could then differ arbitrarily in their explanations while making identical
predictions. That is not the multiplicity this study measures; it is a parameterisation
artifact, and it would contaminate exactly the metric logreg is being added to inform.
`l1` is excluded for a different reason: it does its own feature selection, so seeds would
carry *different term sets*, which the alignment code handles (absent term = zero) but
which changes what the family is. **Revisit if:** an explicit "sparse linear model"
condition is wanted, in which case it is a fourth family, not a variant of this one.

### D-IMP — Global importance is mean-|contribution|, not |coef|
**Chose:** `mean(|coef_j * x_j|)` over the evaluation rows. **Over:** `|coef_j|`.
**Because:** every other family reports a data-weighted mean absolute contribution, and the
top-k Jaccard in F4 compares across families' *rankings*. `|coef|` on a one-hot column with
1% prevalence is large while its contribution to almost every row is zero; ranking by it
would put rare levels at the top for all 30 seeds and make the metric read as stable for a
reason with no interpretive content. **Revisit if:** the question becomes the stability of
the coefficients themselves, which is a different (and worth reporting separately) quantity.

### D-VIEW — Branch on the feature view, not on the model name
**Chose:** replace the three `model == "nam"` tests with an encoded-view test. **Over:**
adding `or model == "logreg"` in three places. **Because:** every one of those branches is
really asking "did this model consume one-hot columns?", and the name test was only ever a
proxy for it. A fourth encoded-view family would otherwise need the same three edits again,
and the failure mode of forgetting one is silent mis-grouping, not an error.

## Out of scope
- Comparing logreg to the EBM (different feature space — the parent plan's ban applies).
- Coefficient-level inference (standard errors, Wald tests). Multiplicity here is
  disagreement across an equally-accurate model *set*, not sampling uncertainty of one fit.
- L1 / elastic-net sparse linear models as a separate family.
- Calibration of the linear model against the teacher beyond what the distilled arm's soft
  cross-entropy already imposes.

## Order of work
1.1 → 1.2 → 1.3 → 3.1/3.3 → tune → 4.1 → 5.1 (predictive lands first, cheaply) → 2.x tests
→ 5.2 → 5.3/5.4/5.5/5.6.

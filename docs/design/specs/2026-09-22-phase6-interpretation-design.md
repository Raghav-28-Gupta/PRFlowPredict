# Phase 6 Design — SHAP Attribution, Error Analysis, Fairness

Written for: whoever implements Phase 6 and whoever writes Phase 8's report.

**Status:** approved in design review 2026-09-22; awaiting spec review.
**Depends on:** Phase 4 complete and merged — 24 boosters in `data/models/`, 24 prediction
files in `data/predictions/`, `data/phase4_runs.json`, gate 5/5 PASS.
**Produces:** the mechanism behind Phase 4's cold-start finding, 50 inspected failures with
GitHub links, the blueprint's §4 fairness check closed with confidence intervals, and a
validity gate.

---

## 1. Purpose

Phase 4 established *that* cold-start transfer fails: on Scenario B, removing the four
label-replay features drops the model to AUC-PR 0.763 against the baseline's 0.821 (at or
below in 4 of 5 folds), while on Scenario A the same ablation still wins (0.902 vs 0.887).
Phase 6 establishes *why* — by showing where each model's attribution mass actually sits.

Blueprint §3, Phase 6: "SHAP feature attribution, manual error analysis on worst 50
predictions." Blueprint §4 also asks whether predictions "differ systematically for
first-time contributors vs. repeat contributors, and for PRs authored outside the
maintainers' typical active hours" — Phase 4 measured the gaps (+0.088 A, +0.053 B) but
reported no interval, so it cannot yet be claimed the newcomer gap is real. Phase 6
closes that.

No model is retrained. Every booster and every per-row prediction already exists on disk.

## 2. Decisions locked in design review

| Decision | Choice | Why |
|---|---|---|
| SHAP focus | **A-vs-B attribution shift on the FULL model** | Directly shows the mechanism behind the cold-start finding rather than asserting it |
| Error analysis | **Both tails on Scenario A: 25 confident false positives + 25 confident false negatives** | A false positive wastes a lead's attention; a false negative is a PR that silently rots. Both matter for a triage tool |
| Fairness | **Cluster-bootstrap CI on each slice gap, plus the first-timer-minus-repeat gap difference with its own CI, plus a SHAP dependence plot for `is_first_pr_here`** | Turns Phase 4's suggestive number into a defensible claim, with a mechanism |
| Compute | **Exact TreeSHAP on a seeded 5,000-row sample per model**; the 50 error rows get exact SHAP individually | `shap.TreeExplainer` is exact for trees; 5k is ample for stable mean-\|SHAP\| rankings and costs seconds |
| Scenario B folds | **Pooled row-wise before ranking** | Five boosters share the same 37 columns; pooling is what makes "what does the B model lean on" a single answer |
| Gate | **Validity, not success** | Consistent with Phases 3–4. SHAP additivity is the hard stop |

## 3. Module map

```
attribution.py   explain(); sample_rows(); importance(); shift(); label_replay_share()
errors.py        worst_rows(); error_patterns()
fairness.py      slice_gaps(); gap_difference()
report6.py       docs/phase6_interpretation.md + figures/; Phase 6 gate -> data/phase6_gate.json
tests/test_attribution.py, test_errors.py, test_fairness.py, test_report6.py
```

Consumed unchanged: `featuresets.FEATURE_SETS`, `model.load/predict`, `metrics.cluster_bootstrap`,
`experiment.load_table`, `splits.SEED`. New dependency: `shap` (pinned in `requirements.txt`).

## 4. Attribution (`attribution.py`)

### 4.1 Sampling and explanation

`sample_rows(df, n=5000, seed=SEED) -> pd.DataFrame` — seeded sample; a frame with fewer
than `n` rows is returned whole.

`explain(booster, X) -> tuple[np.ndarray, float]` — `shap.TreeExplainer(booster)` then
`.shap_values(X)`; returns the value matrix and the explainer's `expected_value`. Exact
for tree ensembles: no background set, no sampling approximation, and the additivity
property is what gate #1 checks.

### 4.2 Importance and the headline shift

`importance(shap_values, cols) -> DataFrame` with `feature, mean_abs_shap, share`, sorted
descending. `share = mean_abs_shap / total_mean_abs_shap`, which makes two scenarios
comparable despite different base rates and different absolute margins.

`shift(imp_a, imp_b) -> DataFrame` — outer join on `feature`; columns
`feature, share_a, share_b, delta` where `delta = share_b − share_a`; sorted by `|delta|`
descending. **This is the phase's headline artifact.**

`label_replay_share(imp) -> float` — sums `share` over
`featuresets.LABEL_REPLAY` (`trailing_90d_slow_rate, trailing_n,
author_prior_slow_rate_here, author_prior_n`). One number per scenario. If B's is
materially larger than A's, the cold-start finding has its mechanism: the B model leans on
the repo's own history — the baseline's signal — rather than on transferable PR-level
structure.

For Scenario B, the five folds' SHAP matrices are concatenated row-wise before
`importance` is called. Each fold contributes its own sample.

## 5. Error analysis (`errors.py`)

`worst_rows(pred, shap_values, cols, feature_frame, n=25) -> DataFrame` on Scenario A's
FULL predictions:

- 25 **false positives**: `is_slow == False`, highest `p_hat`.
- 25 **false negatives**: `is_slow == True`, lowest `p_hat`.

Per row: `kind` (fp/fn), `repo`, `number`, `url` (`https://github.com/{repo}/pull/{number}`),
`p_hat`, `is_slow`, `wait_h`, then the five largest-\|SHAP\| features with their signed SHAP
value and the row's own feature value, as `top1_feature, top1_shap, top1_value, … top5_*`.

`error_patterns(worst, feature_frame, cols) -> DataFrame` — per feature, the z-score of the
50 worst rows' mean against the full test set's mean and standard deviation; columns
`feature, mean_worst, mean_all, z`. Sorted by `|z|`. This is a first pass at "what do the
failures have in common", so the written analysis starts from evidence rather than a blank
page. It does not replace reading the rows.

## 6. Fairness (`fairness.py`)

`slice_gaps(pred, seed=SEED) -> DataFrame` — for `is_first_pr_here` ∈ {first-time, repeat}
and `created_hour_utc` buckets {00–06, 06–12, 12–18, 18–24}: `slice, level, n, n_repos,
actual_rate, mean_p_hat, gap` where `gap = mean_p_hat − actual_rate`, plus `gap_ci_lo,
gap_ci_hi` from `metrics.cluster_bootstrap` resampling **repos**, matching every other
interval in this project.

`gap_difference(pred, seed=SEED) -> dict` — the quantity that actually bears on fairness:
`first_time_gap − repeat_gap`, with its own cluster-bootstrap CI over repos, and a
`ci_excludes_zero` boolean. A positive difference whose CI excludes zero means the model is
*differentially* more pessimistic about newcomers than about repeat contributors — a triage
tool that would systematically deprioritise first-time contributors, which the blueprint
flags as compounding an existing open-source problem.

Both functions run on Scenario A and on pooled Scenario B predictions.

## 7. Report (`report6.py`)

Writes `docs/phase6_interpretation.md` and figures into `figures/`:

1. **Attribution shift** — the `shift` table (top 20) and `figures/phase6_shift.png`, a
   diverging bar chart of the top 15 by `|delta|`.
2. **The cold-start mechanism** — `label_replay_share` for A and B, with a plain-language
   reading tied back to Phase 4's ablation numbers.
3. **Beeswarms** — `figures/phase6_beeswarm_A.png`, `..._B.png` (`shap.summary_plot`).
4. **Dependence** — `figures/phase6_dep_is_first_pr_here.png` and one for the top-shift
   feature.
5. **The 50 worst predictions** — the table with URLs, `error_patterns`, and written
   analysis of what the failures share.
6. **Fairness** — slice table with CIs for both scenarios, and the gap-difference verdict
   stated plainly in either direction.
7. **Gate** table.

Artifacts also written as data: `data/phase6_shift.csv`, `data/phase6_importance_{A,B}.csv`,
`data/phase6_worst50.csv`, `data/phase6_fairness.csv`, `data/phase6_gate.json`.

## 8. Phase 6 gate — pre-registered, validity only

| # | Check | Pass | On fail |
|---|---|---|---|
| 1 | **SHAP additivity**: for 100 seeded rows per explained model, `shap_values.sum(axis=1) + expected_value` equals the booster's raw-margin prediction | max \|Δ\| < 1e-6 | **HARD STOP** — the attributions are not the model's |
| 2 | Sampling stability: two different seeds' top-10 feature rankings on the same model | Spearman ≥ 0.9 | raise the sample size and re-run |
| 3 | Error rows: every `pr_id` appears in Scenario A's FULL prediction file; exactly 25 fp and 25 fn; every `is_slow` matches that file | exact | the error table is not describing the model's real failures |
| 4 | Every fairness CI is finite and computed from ≥ 2 repos | exact | the interval is not meaningful |
| 5 | All five data artifacts written and non-empty | exact | rerun |

**Not gated:** what the attributions say. A shift table showing B leaning on label-replay
features and one showing it not are both valid outcomes; §7 item 2 reports whichever occurs.

## 9. Testing

Toy fixtures with known answers, following the Phase 3–4 pattern:

- `test_attribution.py`: a 2-feature, 400-row toy where feature `a` drives the label and `b`
  is noise — assert `importance` ranks `a` first and its `share` exceeds `b`'s; `shift` on
  two hand-built importance frames produces the expected `delta` signs and ordering;
  `label_replay_share` sums exactly the four named columns and returns 0.0 when none are
  present; `sample_rows` is seeded (same seed → same rows) and returns all rows when
  `n > len(df)`; additivity holds on the toy booster.
- `test_errors.py`: a hand-built pred frame where the fp/fn extremes are known by
  construction — assert `worst_rows` picks exactly those, labels `kind` correctly, builds
  the right URL, and orders `top1..top5` by descending `|SHAP|`; `error_patterns` returns a
  large positive `z` for a feature deliberately inflated among the worst rows.
- `test_fairness.py`: a frame with a planted first-timer gap — assert `slice_gaps` recovers
  it, all six slice rows are present, and `gap_difference`'s `ci_excludes_zero` is True for a
  large planted gap and False for no gap.
- `test_report6.py`: `gate_checks` as a pure function on synthetic inputs — each failure mode
  fails exactly its own check.

## 10. Out of scope

Retraining anything. The survival model (Phase 5, nice-to-have). The Streamlit demo
(Phase 7). The write-up itself (Phase 8) — Phase 6 produces its evidence, not its prose.

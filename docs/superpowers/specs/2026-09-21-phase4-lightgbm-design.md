# Phase 4 Design — LightGBM Classifier, Both Scenarios, Against the Baseline

Written for: whoever implements Phase 4 and whoever reads its results in Phases 5–8.

**Status:** approved in design review 2026-09-21; awaiting spec review.
**Depends on:** Phase 3 complete — `data/features/features.parquet` (38,462 × 50),
`data/phase3_gate.json` all five checks PASS, `docs/feature_dictionary.md`.
**Produces:** tuned parameters, 24 trained models with per-row predictions, the results
document with the blueprint's headline comparison, and a validity gate.

---

## 1. Purpose

Blueprint §2, model 2: "**Primary model — LightGBM binary classifier** predicting `is_slow`
… Every subsequent model is measured against [the baseline]." Blueprint §1: "The baseline
this must beat: predict `is_slow` using only the repo's trailing 90-day slow-rate. If the
trained model can't beat that, PR-level signal isn't adding value, and that is a
legitimate, reportable finding — not a failure to hide."

Phase 2 established that baseline on real data: **AUC-PR 0.887** on Scenario A (it ranks
*repos* extremely well) but **within-repo P@10 0.58 against a 0.64 base rate** — a random
draw, because the score is nearly constant within a repo over short spans. Phase 4 asks
the blueprint's actual question: does PR-level signal let you rank *within* a repo?

The phase is designed so that a *negative* answer is a clean result, not a broken run:
the gate tests validity, the report states the comparison either way, and the
NO_LABEL_REPLAY ablation asks whether the model beats the baseline without being handed
the baseline's own feature.

## 2. Decisions locked in design review

| Decision | Choice | Why |
|---|---|---|
| Tuning | Optuna on Scenario A training rows only (3-fold `TimeSeriesSplit`); params **frozen and reused** for Scenario B and for every ablation | B's held-out repos never touch tuning, so the cold-start claim stays clean; ablations compared under identical params is the fair comparison |
| Ablations | FULL, NO_SNAPSHOT, NO_LABEL_REPLAY, PR_ONLY — all on both scenarios | NO_LABEL_REPLAY answers the blueprint's question; NO_SNAPSHOT tests the feature dictionary's caveat that HEAD-at-collection repo facts may carry the review culture |
| Bias checks | Minimal slice table in Phase 4 (first-time vs repeat; hour-of-day buckets); depth in Phase 6 | Free once predictions are saved |
| Gate | **Validity**, not success | Failing to beat the baseline is a reportable finding, not a blocker |
| Figures | P@10 bars with CIs per scenario; PR curves FULL vs baseline per scenario | Write-up material |
| Early stopping | None on test; `n_estimators` is tuned | Never touch test rows during fitting |
| Truncated rows | The 18 `timeline_may_be_truncated` rows are dropped | Feature dictionary caveat #1: they are biased toward current state |

## 3. Module map

```
featuresets.py   FEATURE_SETS: dict[name -> list[str]] derived from features.COLUMN_SPEC
model.py         fit(X, y, params) -> Booster; predict(booster, X) -> p_hat; categorical + seeds
tune.py          Optuna on Scenario A train (FULL) -> data/models/params.json
experiment.py    run(scenario, fold, featureset) -> metrics dict; saves model + predictions; logs
report4.py       docs/phase4_results.md + figures/; Phase 4 gate -> data/phase4_gate.json
tests/test_featuresets.py, test_model.py, test_experiment.py, test_report4.py
```

Consumed unchanged: `features.py` (`COLUMN_SPEC`, `feature_columns`), `splits.py`
(`scenario_a`, `scenario_b`, `CUTOFF_A`, `SEED`), `metrics.py` (`precision_at_k`,
`auc_pr`, `cluster_bootstrap`), `baseline.py` (`evaluate`, `score_rows`), `tracking.py`.

New dependencies: `lightgbm`, `optuna` (pinned in `requirements.txt`).

## 4. Data (`experiment.load_table`)

`data/features/features.parquet` → drop rows with `timeline_may_be_truncated` → sort by
`created_at, pr_id` → reset index. `language_dominant` cast to pandas `category`.

Scenario A: `splits.scenario_a(table)` — train `created_at < 2026-01-01` capped at 5% per
repo (seeded), test `>= 2026-01-01` uncapped. Scenario B: `splits.scenario_b(table)` —
`GroupKFold(5)` on repo, training folds capped. Both functions already accept any frame
with `repo` and `created_at`; no new split code.

## 5. Feature sets (`featuresets.py`)

All derived from `features.COLUMN_SPEC`; none hand-listed.

| Name | Definition | Count |
|---|---|---|
| `FULL` | `features.feature_columns()` | 37 |
| `NO_SNAPSHOT` | FULL minus columns with `status == "snapshot"` | 29 |
| `NO_LABEL_REPLAY` | FULL minus `{trailing_90d_slow_rate, trailing_n, author_prior_slow_rate_here, author_prior_n}` | 33 |
| `PR_ONLY` | FULL restricted to `group in {"static", "at_open"}` | 17 |

`assert_hygiene(featureset)` raises if any column's status is `key`, `label`, or `flag`.
Called by every consumer and by a test over all four sets.

## 6. Model (`model.py`)

```python
FIXED = dict(objective="binary", seed=SEED, deterministic=True, force_row_wise=True,
             n_jobs=1, verbose=-1)
```

`fit(X, y, params) -> lgb.Booster`: `scale_pos_weight = n_neg / n_pos` computed from `y`;
categorical feature `language_dominant` passed by name when present; `params` are the
tuned keys (§7) merged over `FIXED`. `predict(booster, X) -> np.ndarray` of P(is_slow).
`save(booster, path)` / `load(path)` use LightGBM's native text format.

Reproducibility is a gate item (§10 #5): with `deterministic=True, n_jobs=1` and fixed
seeds, two fits on identical inputs must agree to 1e-6 in AUC-PR.

## 7. Tuning (`tune.py`)

- Rows: Scenario A **training** rows (capped), sorted by `created_at`.
- CV: `TimeSeriesSplit(n_splits=3)` — each fold trains on earlier rows, validates on later.
- Objective: mean validation AUC-PR (`metrics.auc_pr`) across the 3 folds. Maximise.
- Search space (Optuna TPE, `seed=SEED`, 100 trials):

| Param | Range |
|---|---|
| `num_leaves` | int [8, 128] |
| `learning_rate` | float [0.01, 0.2] log |
| `n_estimators` | int [100, 1000] |
| `min_child_samples` | int [10, 200] |
| `feature_fraction` | float [0.5, 1.0] |
| `bagging_fraction` | float [0.5, 1.0], with `bagging_freq = 1` |
| `lambda_l2` | float [1e-3, 10] log |

- Feature set: FULL only.
- Output `data/models/params.json`: `{"best_params", "cv_auc_pr", "n_trials", "seed",
  "trials": [...]}`. `experiment.py` refuses to run without it.

## 8. Experiment runner (`experiment.py`)

`run(scenario, fold, featureset_name, table, params) -> dict`:

1. Obtain `(train_idx, test_idx)` for the scenario/fold (Scenario A has one fold).
2. `X = table.loc[idx, FEATURE_SETS[name]]`, `y = table.loc[idx, "is_slow"]`.
3. Fit on train; `p_hat` on test.
4. Metrics on test: `precision_at_k(y, p_hat, repo, k=10, seed)` → per-repo + mean;
   `cluster_bootstrap(per_repo)` → CI; `auc_pr`; `base_rate`; `base_rate_p10`.
5. Baseline on the **same rows**: the baseline's score for a row is the table's own
   `trailing_90d_slow_rate` column — the identical value `baseline.py` computes (same
   `features_at`, same Scenario A priors), proven against the brute-force twin in Phase 3.
   `baseline_p10` and `baseline_auc_pr` come from it via the same metric functions, so
   model and baseline are always compared on exactly the same test rows.
6. Save `data/models/{scenario}_{name}_fold{k}.txt` and
   `data/predictions/{scenario}_{name}_fold{k}.parquet` with columns
   `pr_id, repo, created_at, is_slow, p_hat, baseline_score, is_first_pr_here,
   created_hour_utc, diff_is_exact`.
7. `tracking.log({... model="lgbm", features=name, params=<sha256 of params.json>[:12],
   n_train, n_test, n_test_repos, precision_at_10, p10_ci_lo, p10_ci_hi, base_rate_p10,
   auc_pr, base_rate, notes=f"baseline_p10={..:.3f} baseline_auc_pr={..:.3f}"})`.

`main()` runs all 24 (A: 4 sets × 1 fold; B: 4 sets × 5 folds) and writes
`data/phase4_runs.json` (the list of result dicts) for the report.

## 9. Report (`report4.py`)

Writes `docs/phase4_results.md` and `figures/phase4_p10.png`, `figures/phase4_pr_A.png`,
`figures/phase4_pr_B.png`.

1. **Setup** — tuned params, CV AUC-PR, trial count, row counts, the 18 dropped rows.
2. **Scenario A** — table: featureset × {P@10 [CI], baseline P@10, base_rate_p10, AUC-PR,
   baseline AUC-PR}.
3. **Scenario B** — same table with per-fold means; per-fold detail for FULL.
4. **Headline** — FULL vs baseline P@10 on A: the difference in points, whether the CI
   excludes the baseline value, and the same for base_rate_p10. Then the A→B gap for
   FULL. Written to be true either way.
5. **Ablations** — Δ P@10 and Δ AUC-PR vs FULL for each set, both scenarios. One sentence
   each on what NO_LABEL_REPLAY and NO_SNAPSHOT say.
6. **Bias slices** (FULL predictions, both scenarios) — by `is_first_pr_here` and by
   `created_hour_utc` bucket {00–06, 06–12, 12–18, 18–24}: n, actual slow rate, mean
   `p_hat`, AUC-PR within slice. A one-line reading of whether the model is more
   pessimistic about first-timers than their actual rate warrants (blueprint §4).
7. **Figures** — P@10 with CI bars per featureset per scenario; PR curves FULL vs
   baseline (A: the single fold; B: pooled over folds).
8. **Gate** table.

## 10. Phase 4 gate — pre-registered, validity only

| # | Check | Pass | On fail |
|---|---|---|---|
| 1 | Split disjointness, from the saved prediction files: A — every test `created_at` ≥ 2026-01-01 and the training max (recomputed) < it; B — for every fold, test repos ∩ train repos = ∅ | exact | leakage by construction — stop |
| 2 | Feature hygiene: no column with status `key`/`label`/`flag` in any of the four sets | exact | stop |
| 3 | NO_LABEL_REPLAY ran on both scenarios (present in `phase4_runs.json`) | exact | the blueprint's question is unanswered — rerun |
| 4 | Every run has a bootstrap CI; `experiments.csv` contains all 24 `lgbm` rows for this params sha | exact | rerun the missing ones |
| 5 | Reproducibility: refit A/FULL from `params.json` and compare AUC-PR to the saved run | \|Δ\| < 1e-6 | non-determinism — fix before any number is quoted |

**Not a gate:** whether FULL beats the baseline. That is §9 item 4, reported either way.

## 11. Testing

- `test_featuresets.py`: counts (37/29/33/17); hygiene assertion fires on a set that
  smuggles `is_slow`; every set is a subset of FULL.
- `test_model.py`: fit/predict on a 300-row toy with a categorical column; two fits agree
  exactly; `scale_pos_weight` computed from `y`.
- `test_experiment.py`: `run()` on a small synthetic table (2 repos, 400 rows, planted
  signal) returns all metric keys, writes model + predictions, and the predictions file
  has disjoint train/test by time; baseline columns present.
- `test_report4.py`: gate checks as a pure function on synthetic run dicts — each
  failure mode fails exactly its check; bias-slice table has the four hour buckets.

## 12. Out of scope

Survival model and calibration (Phase 5). SHAP and manual error analysis (Phase 6). The
Streamlit demo (Phase 7). Any re-tuning per ablation or per Scenario B fold.

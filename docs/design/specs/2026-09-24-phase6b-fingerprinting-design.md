# Phase 6b Design — The Repo-Fingerprinting Test

Written for: whoever implements Phase 6b and whoever writes Phase 8's report.

**Status:** approved in design review 2026-09-24; awaiting spec review.
**Depends on:** Phase 4 (the six `NO_LABEL_REPLAY` boosters, their predictions, the tuned
params) and Phase 6 (the attribution machinery). Both merged to `main`.
**Produces:** a pre-registered verdict on one candidate mechanism for Phase 4's cold-start
finding, and the evidence behind it.

---

## 1. Why this exists

Phase 4 found that the model does not transfer across projects. Remove the four
label-replay features and Scenario B (leave-repos-out) falls below the trailing-rate
baseline, AUC-PR 0.763 vs 0.821, while Scenario A (within-project) still beats it,
0.902 vs 0.887.

The blueprint's fallback framing, now the operative one, promises to report "cross-project
transfer is not viable, **and here's the SHAP evidence for why**." Phase 6 set out to
supply that and did not. Its test compared the two **FULL** models, which turned out
near-identical by attribution (label-replay share 60.9% A vs 63.1% B). But Phase 4's
finding is about the **NO_LABEL_REPLAY ablation**, a different pair of models, and Phase 6
did not examine those.

This phase tests one specific mechanism for that ablation's failure.

## 2. The hypothesis, and how it was formed

**Repo fingerprinting.** Stripped of the label-replay features, the model uses repo-level
features, which are constant within a repo, to *identify* repos and memorise their base
rates. With ~31 training repos per fold, a feature like `n_mentionable_users` takes roughly
one value per repo, so trees can separate repos by it. That is label replay by proxy. On a
repo the model has never seen, those values map to nothing it learned.

**Disclosure: this hypothesis is post-hoc.** It was formed after an exploratory peek at
LightGBM **gain** importance on the NLR boosters. The 9 empirically repo-constant features
carried 41.0% of gain on B vs 28.6% on A, and `n_mentionable_users` rose from 4.1% to
17.5%. Gain is a training-time measure and biased toward high-cardinality features, so it
is **motivation, not evidence**. Everything below is the confirmatory test. Its decision
rule is fixed in this document, committed before any of it runs.

(The peek's 9 empirically-constant features are this spec's 8 plus
`requested_team_at_open`, a PR-level feature that is constant only because it is
degenerate.)

## 3. Decisions locked in design review

| Decision | Choice | Why |
|---|---|---|
| Scope | **SHAP on the 6 existing NLR boosters + one new interventional ablation** | SHAP shows mechanism but is correlational; retraining without the features makes it interventional |
| Verdict rule | **Both tests must agree** for "supported" | Each covers the other's weakness; hardest to claim falsely |
| Repo-feature definition | **`group == 'repo'` AND `status == 'snapshot'` in `features.COLUMN_SPEC`: 8 features** | Fixed by code metadata, not data; exactly what Phase 4's `NO_SNAPSHOT` removed |
| SHAP coverage | **All test rows**, no sampling | Per-repo means need every repo covered |
| Output | **Its own script and fully generated document** | Re-running `report6.py` destroys Phase 6's hand-written §4 |

## 4. The eight repo-level features

`n_assignable_users`, `n_mentionable_users`, `owner_is_org`, `has_codeowners`,
`has_pr_template`, `has_contributing`, `n_ci_workflows`, `language_dominant`.

Selected in code as `group == 'repo' and status == 'snapshot'`. Each is a single 2026
snapshot value per repo, which is exactly why each is constant within every repo. This
set equals `FEATURE_SETS["FULL"]` minus `FEATURE_SETS["NO_SNAPSHOT"]`, and the gate
verifies that equality.

**Excluded, and why:** `repo_age_days_at_open` is also `group == 'repo'`, but it varies
within a repo (it is the repo's age at each PR's creation), so it is at most a partial
fingerprint. Excluding it makes the definition **conservative**: if it errs, it
under-counts fingerprinting rather than inflating it.

## 5. The three measurements

All three use the same 39 repos. Both scenarios' test sets cover all 39, and in B each
repo is held out in exactly one fold (verified 2026-09-24). That makes every A-vs-B
comparison **paired by repo**.

### 5.1 Reliance (descriptive, not part of the verdict)

The share of mean |SHAP| carried by the 8 features, on NLR-A and on NLR-B pooled over its
five folds, computed with `attribution.importance` on all test rows. Reported as
B − A in percentage points, and read with Phase 6's band: within ±5 points counts as
unchanged.

This is what the gain peek measured, now measured properly. It is descriptive because
reliance alone was exactly Phase 6's test, and that proved insufficient to establish a
mechanism.

### 5.2 Transfer (SHAP), test T

For each repo *r* in scenario *s*:
- **c_r**: the mean, over *r*'s test rows, of the summed SHAP values of the 8 features, in
  raw-margin (log-odds) units. In B, each repo's rows are explained by the fold model that
  held it out.
- **y_r**: *r*'s actual slow rate on those same test rows.

Then **ρ_s = Spearman(c_r, y_r)** over the 39 repos.

Fingerprinting predicts **ρ_A > 0**, because A's repos were seen in training, memorised
rates persist and the contribution tracks reality, and **ρ_B ≈ 0**, because on unseen repos
the contribution is arbitrary routing.

**Statistic:** ρ_A − ρ_B. If ρ_B were close to ρ_A, the repo features would be carrying
real signal that transfers, which is not fingerprinting.

### 5.3 Intervention (retrain), test I

A new ablation, **`NLR_NO_REPO`**: `FEATURE_SETS["NO_LABEL_REPLAY"]` minus the 8, giving
25 features. It is trained on **the identical folds** (`experiment.folds_for`) with **the
identical tuned params** (`data/models/params.json` `best_params`, semantically equal to
the committed `data/phase4_params.json`): 1 fit for A and 5 for B.

**Δ_s = AUC-PR(NLR_NO_REPO) − AUC-PR(NLR)** on scenario *s*, where the two models are scored
on identical test rows.

AUC-PR follows Phase 4's convention exactly: A is a single fold; **B is the mean of its
five per-fold AUC-PRs**, not a pooled figure. The NLR baselines are Phase 4's saved values:
A 0.901745; B folds 0.964792 / 0.834803 / 0.507069 / 0.643918 / 0.866883, mean 0.763493.

Fingerprinting predicts that removing the 8 is **relatively better for B than for A**,
because A's fingerprints are valid and B's are not.

**Statistic:** the interaction **Δ_B − Δ_A**. The strong form of the prediction is the
**crossover**, Δ_A < 0 ≤ Δ_B. It is reported as a descriptive flag, not a gate on the
verdict.

### 5.4 Intervals: one paired repo bootstrap

This uses **2,000 draws** of the 39 repos **with replacement**, `np.random.default_rng(SEED)`
and percentile 95% intervals, the same conventions as `metrics.cluster_bootstrap`. The
**same draw** is applied to both scenarios and to both statistics, which is what makes them
paired. For each draw:

- **T:** recompute ρ_A and ρ_B on the drawn repo multiset (duplicates kept), then take the
  difference.
- **I:** for A, recompute both models' AUC-PR on the drawn repos' rows (with multiplicity).
  For B, recompute each fold's AUC-PR on the drawn repos belonging to that fold, then
  average over folds. A fold is included only if it has at least one drawn repo and both
  classes present. Take Δ_B − Δ_A.

Point estimates are computed on the undrawn data with the same code path. The interval
exists because B's per-fold AUC-PR ranges from 0.507 to 0.965: with ~8 held-out repos per
fold, fold composition dominates, and five fold-level deltas cannot support an interval.

## 6. The pre-registered verdict

Each test resolves to exactly one of three outcomes from its 95% interval:
- **confirms**: the interval lies entirely above 0.
- **contradicts**: the interval lies entirely below 0.
- **inconclusive**: the interval contains 0.

| T | I | Verdict |
|---|---|---|
| confirms | confirms | **SUPPORTED**: fingerprinting explains the ablation's cold-start failure |
| confirms | inconclusive | **PARTIAL (SHAP only)**: the mechanism is visible, the intervention does not confirm it |
| inconclusive | confirms | **PARTIAL (intervention only)**: removing the features helps B relatively, but the SHAP transfer pattern is not visible |
| inconclusive | inconclusive | **NOT SUPPORTED** |
| confirms | contradicts | **CONFLICTING** |
| contradicts | confirms | **CONFLICTING** |
| inconclusive | contradicts | **CONTRADICTED** |
| contradicts | inconclusive | **CONTRADICTED** |
| contradicts | contradicts | **CONTRADICTED** |

The verdict is a pure function of the two outcomes and is tested on all nine cells. The
generated document states the verdict first, in plain language, whichever cell occurs.
Reliance (§5.1) and the crossover flag (§5.3) are reported alongside it and never change it.

**Not gated:** what the verdict says. All nine cells are valid outcomes of this phase.

## 7. Gate: pre-registered, validity only

| # | Check | Pass | On fail |
|---|---|---|---|
| 1 | **SHAP additivity** on all 6 NLR boosters, all test rows: summed SHAP values plus expected value equal `booster.predict(X, raw_score=True)` | max \|Δ\| < 1e-6 | **HARD STOP**: the attributions are not the model's |
| 2 | **Behaviour-preserving refit**: A/NLR re-fit through the new `cols=` code path, written to a scratch directory, reproduces Phase 4's saved AUC-PR 0.901745 | \|Δ\| < 1e-6 | **HARD STOP**: the intervention would not compare like with like |
| 3 | **Identical test rows**: for every (scenario, fold), the `NLR_NO_REPO` test `pr_id` set equals the NLR one | exact | the intervention compares different rows |
| 4 | **Feature-set integrity**: the 8 are constant within every repo; they equal FULL − NO_SNAPSHOT; `NLR_NO_REPO` = NLR − exactly the 8 (25 columns); hygiene passes | exact | the definition's premise is false |
| 5 | **Coverage and completeness**: both scenarios cover the same 39 repos; B holds out each exactly once; every interval is finite; every artifact is written and non-empty | exact | rerun |

## 8. Engineering

**`experiment.run` gains `cols: list[str] | None = None`.** When it is `None`, the columns
are `fs.FEATURE_SETS[name]` exactly as today. This change is backward-compatible, and
gate check 2 proves it is behaviour-preserving.

**`NLR_NO_REPO` is NOT added to `fs.FEATURE_SETS`.** `experiment.main()` iterates every entry
and writes all of them to `data/phase4_runs.json`, so registering a fifth set would
silently change what a Phase 4 re-run trains and break Phase 4's own gate check 4
(`len(runs) == 24`). The set is defined in `fingerprint.py`.

**Phase 4's gate check 4 gets a one-line fix.** It currently counts every `lgbm` row in
`data/experiments.csv` under the params sha and requires exactly 24. That assumes Phase 4
owns the shared log, and blueprint §5 says it does not ("a single `experiments.csv`
logging … every metric"). The six new runs are logged there as `model="lgbm"`,
`features="NLR_NO_REPO"`. The fix adds `experiments["features"].isin(fs.FEATURE_SETS)` to
the filter. Today's count stays 24, and a test pins both facts.

**Artifacts:**
- Boosters and predictions go to `data/models/` and `data/predictions/` under tags
  `{A,B}_NLR_NO_REPO_fold{k}`, the same layout as Phase 4 and distinct names, so nothing is
  overwritten. Check `.gitignore` for which are tracked.
- Run metadata goes to **`data/phase6b_runs.json`**, never `phase4_runs.json`.
- Results go to `data/phase6b_{reliance,transfer,intervention}.csv` and `data/phase6b_gate.json`.
- The document is `docs/phase6b_fingerprinting.md`, **fully generated**, with no hand-written
  section. Phase 6's §4 showed what a hand-maintained section inside a generated document
  costs.

**Module map:**
```
fingerprint.py    REPO_FEATURES; nlr_no_repo_cols(); per_repo_contribution(); transfer_rho();
                  paired_repo_bootstrap(); auc_delta(); outcome(); verdict()     (all pure)
report6b.py       trains NLR_NO_REPO via experiment.run(cols=...); explains the 6 NLR boosters
                  via attribution.explain; runs the bootstrap; writes the doc, CSVs and gate
experiment.py     + cols=None on run()
report4.py        check 4 filter: + features.isin(fs.FEATURE_SETS)
tests/test_fingerprint.py, tests/test_report6b.py, + tests for the two small changes above
```

Consumed unchanged: `attribution.explain`, `attribution.importance`, `model.load`,
`metrics.auc_pr`, `experiment.folds_for`, `experiment.load_table`, `splits.SEED`,
`features.COLUMN_SPEC`.

## 9. Testing

The project's standing defect class is a test whose assertions hold whether or not the
behaviour it names works. Six shipped during Phase 6. Every test below must fail when its
named behaviour is broken, and the plan will name the mutation each test must catch.

- **`REPO_FEATURES`**: equals exactly FULL − NO_SNAPSHOT; excludes `repo_age_days_at_open`.
- **`per_repo_contribution`**: on a planted fixture, returns per-repo means of the summed
  repo-feature SHAP columns only. It must fail if a non-repo column is included, or if it
  takes a sum rather than a mean over rows.
- **`transfer_rho` + bootstrap**: a synthetic where contributions track rates in "A" and are
  random in "B" gives a T interval above 0. A control where both track gives an interval
  containing 0. The bootstrap is seed-deterministic, and a repo missing from one scenario
  raises, because the pairing premise has broken.
- **`auc_delta`**: on undrawn data, B's value equals the direct mean-of-folds computation.
  A fold with a single class is skipped, not scored.
- **`verdict`**: all nine cells.
- **`experiment.run(cols=...)`**: on a tiny synthetic, `cols=None` and
  `cols=FEATURE_SETS[name]` give byte-identical predictions.
- **Phase 4 check 4**: extra `lgbm` rows with a non-Phase-4 `features` value under the same
  sha do not change the count.
- **`report6b.gate_checks`**: each failure mode fails exactly its own check, using the
  `_failed(c) == [N]` pattern from Phase 6.

## 10. Out of scope

Re-tuning hyperparameters. Changing any Phase 4 or Phase 6 artifact beyond the one-line
check-4 filter. The FULL models, whose attribution Phase 6 already reported. Any fix to
the mechanism if it is confirmed. Phase 8's prose: this phase produces its evidence, not
its narrative.

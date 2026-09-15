# Phase 2 Design — Labels, EDA, Baseline

Written for: whoever implements Phase 2 and whoever builds Phases 3–6 on top of it.

**Status:** approved in design review 2026-09-16; awaiting spec review.
**Depends on:** Phase 1 collection complete for the 45-repo cohort (`collect_cohort.py`,
launched 2026-09-16), `parse.py` run on every repo.
**Produces:** the target variable, the evaluation harness, the baseline every model must
beat, and the Phase 2 gate report.

---

## 1. Purpose

Phase 1 captured raw signal and deliberately did *not* decide what a "review" is. Phase 2
decides it, on the full cohort, with a pre-registered rule — and builds the three pieces
every later phase consumes: **labels**, **splits + metrics**, and the **point-in-time
replay engine**. The trailing-90-day baseline is the first client of all three.

Blueprint §3, Phase 2 `[R1]`: "Label construction incl. choosing the definition on cohort
data, EDA, Kaplan–Meier curve of unreviewed-fraction over time, baseline computed,
subset 45 → 30 repos." The subset rule was revised in design review: **drop degenerates
only, cap rows** — see §5.

## 2. Decisions locked in design review

| Decision | Choice | Why |
|---|---|---|
| Architecture | Importable modules, script-driven, no notebooks | Labels, splits, metrics, replay are consumed by Phases 3–6; build them once as modules. `gate_report.py` is refactored to import `labels.py` so the target has one implementation. |
| Label definition | **D5 primary, D3 as sensitivity**, pre-registered | A drive-by comment from someone with no standing in the repo is not a review. Decided on principle before seeing cohort numbers, so it cannot be the flattering choice. |
| Precision@10 unit | **Per repo, over the whole test period**, averaged over repos | Matches "which PRs in this repo should I have chased." The trailing-rate baseline is constant within a repo, so its top-10 is a random draw = base rate — the honest claim that without PR-level signal you cannot rank within a repo. |
| Scenario A cutoff | **2026-01-01** (train 24 months, test 6) | Every test PR has full 30-day follow-up; trailing features for early-test PRs come from training-period labels, which *is* the deployment scenario. |
| 45 → 30 | **Drop degenerates only; cap rows** in Scenario A training | More repos strictly helps the cold-start claim; concentration is handled explicitly rather than by exclusion. |
| Concentration cap | 5% of Scenario A training rows per repo, seeded subsample | Three repos hold ~10–11% each; the cap stops the within-repo model from mostly learning three review cultures. Test sets are never capped. |

## 3. Module map

```
load.py         unified frames across repos: prs_tier1, prs_tier2, reviews,
                thread_comments, issue_comments, timeline, commits, repo_meta
labels.py       definitions D1–D5; first_human_event(); label()
replay.py       History per repo; features_at(t) — leakage structural by construction
splits.py       scenario_a(), scenario_b(); bot-author exclusion; training cap
metrics.py      precision_at_k per repo; auc_pr; cluster_bootstrap over repos
baseline.py     trailing-90d slow-rate baseline on A and B; logs to experiments.csv
cohort_qc.py    pre-registered structural checks → data/cohort/kept.json
eda_report.py   docs/phase2_eda.md + figures/; Phase 2 gate verdict
tracking.py     append-only experiments.csv writer (blueprint §5)
```

Each module is importable with no side effects; each script has a `main()`.
Dependencies added to `requirements.txt`: `lifelines` (Kaplan–Meier with CI bands),
`scikit-learn` (GroupKFold, `average_precision_score`).

## 4. Labels (`labels.py`)

### 4.1 Definitions

A **definition** is a named config: which streams count, which commenter associations
are excluded, whether minimized comments are excluded.

| Name | Streams | Excluded associations | Excl. minimized | Role |
|---|---|---|---|---|
| D1 | reviews | — | no | sensitivity |
| D2 | reviews + thread_comments | — | no | sensitivity |
| D3 | reviews + thread_comments + issue_comments | — | no | **blueprint definition; reported alongside** |
| D4 | D3 | — | yes | sensitivity |
| D5 | D3 | `NONE` | no | **primary** |

All five are always computed. The sensitivity table is a by-product, never a re-run.

### 4.2 Event eligibility

An event counts toward "first human review" if **all** of:

1. Its **visible timestamp** is non-null:
   - `reviews`: `submitted_at`. This is the connection's sort key (verified on
     litestream). `PENDING` reviews have null `submitted_at` and never count.
   - `thread_comments`, `issue_comments`: `published_at`, falling back to `created_at`.
2. Its author is **not the PR author** (`author_login != pr.author_login`; a null
   commenter login never counts).
3. Its author is **not a bot**: `author_typename == "Bot"` OR `login.endswith("[bot]")`
   OR `login in KNOWN_BOTS` (`dependabot`, `renovate`, `github-actions`, `codecov`,
   `coveralls`, `sonarcloud`, `netlify`, `vercel`, `pre-commit-ci`, `allcontributors`,
   `stale`, `mergify`, `copilot`). The list is a module constant; extend it, don't hide it.
4. It passes the definition's association / minimized filters.

`first_human_event(prs, streams, definition) -> Series[pr_id → timestamp]` returns the
minimum eligible visible timestamp per PR.

### 4.3 Label

`label(prs, first_event, threshold_h=168, censor_h=720) -> DataFrame` with columns:

| Column | Meaning |
|---|---|
| `first_event_at` | as above, NaT if none |
| `wait_h` | `first_event_at − created_at` in hours; NaN if none |
| `never_reviewed_30d` | `wait_h` is NaN **or** `wait_h > censor_h` |
| `is_slow` | `never_reviewed_30d` **or** `wait_h > threshold_h` |
| `wait_h_censored` | `min(wait_h, censor_h)`; `censor_h` when never reviewed. For Phase 5. |
| `event_observed` | `not never_reviewed_30d`. For Phase 5. |

The 30-day window is why censoring is a non-issue for the classifier: any PR without a
qualifying event inside 720h is `is_slow = 1` with certainty, since 720 > 168.

### 4.4 Documented limitation

Commenter `author_association` is computed at read time (the same mutability proven for
PR authors in the Phase 0 gate). A `NONE` commenter in 2024 who later became a
`CONTRIBUTOR` reads as `CONTRIBUTOR` today, so **D5 slightly over-counts reviews**
relative to true at-the-time standing. The error is in the conservative direction for
the spam hazard (it counts *more* things as reviews, never fewer) and is stated in the
EDA report.

### 4.5 `gate_report.py` refactor

The label logic currently inlined in `gate_report.py` is deleted; the script imports
`labels.first_human_event` / `labels.label`. Its output for litestream and skills must be
identical before and after — that is the regression test for the refactor.

## 5. Cohort QC (`cohort_qc.py`)

Pre-registered, **structural-only** checks on in-window PRs. Writes
`data/cohort/kept.json`: every repo with `kept: bool` and `reasons: [...]`.

| Check | Drop if | Rationale |
|---|---|---|
| Bot dominance | bot-authored share of in-window PRs > 50% | Blueprint: bot PRs follow different timelines and are analysed separately; a repo that is mostly Dependabot has no human review process to model. |
| Too small | < 100 human-authored in-window PRs | Precision@10 and per-repo rates are meaningless below this. |
| Language mismatch | `repo_meta.language_dominant` not in the stratum's family — Python → {Python}; TypeScript → {TypeScript, JavaScript}; Go → {Go} | Catches a guidelines/docs repo drawn into a code cell (`zero-to-mastery/start-here-guidelines` in a trial draw). |

**Deliberately absent:** any check on the label distribution. A 98%-never-reviewed repo
is data. Dropping on the outcome is the one thing the blueprint forbids for selection,
and post-hoc QC is selection.

Also emits, without dropping, a **look-at-these** list: row share > 8% of the kept
cohort, or per-repo |D3 − D5| spread > 15pp. A human glances at it before Phase 3.

## 6. Replay (`replay.py`)

The blueprint's leakage defence: "implement a `features_at(pr, history_before_pr)`
function and generate the table by chronological replay — this makes leakage
structurally impossible rather than merely checked for."

### 6.1 `History`

One per repo. Built from **Tier 1** (all-time, so backlog and author history are exact)
joined with labels (available only for in-window Tier 2 rows). Sorted by `created_at`;
arrays for `created_at`, `closed_at`, `author_login`, `first_event_at`, `is_slow`.

`features_at(t)` locates the prefix `created_at < t` by bisect and computes every
feature from that prefix only. There is no code path by which a row at or after `t` is
visible. That is the structural guarantee; there is no separate "leakage check."

### 6.2 Phase 2 features (exactly two)

| Feature | Definition |
|---|---|
| `open_backlog_at_t` | count of prefix rows with `closed_at` null or `closed_at > t` |
| `trailing_90d_slow_rate` | over prefix rows with `created_at >= t − 90d` **and resolvable at t**: `created_at <= t − 168h` OR `first_event_at < t`. Rate = `(n_slow + α·g) / (n + α)` with `α = 5`, `g` = the D5 slow rate over the **training rows of the split being evaluated** (never the test rows; in Scenario B, the fold's training repos). Rows with no label (pre-window) are excluded from the rate but still count toward backlog. |
| `trailing_window_complete` | `t − 90d >= 2024-01-01` — flag, not a feature; marks the noisy first 90 days |

Labels feeding the rate are computed on human-authored, in-window PRs only
(`splits.modelling_prs`); bot PRs still count toward `open_backlog_at_t` because backlog
comes from Tier 1.

The resolvability predicate is the `[R1]` correction to blueprint §4: "strictly earlier
than the row" is insufficient because a PR opened 3 days ago has no knowable label yet.

### 6.3 Interface contract for Phase 3

`features_at(t: Timestamp, author: str | None) -> dict[str, float]`. Phase 3 adds
author-history and the at-open reconstructions by extending this function and the
arrays `History` carries. The bisect-prefix discipline is the invariant; it must not be
bypassed for convenience.

## 7. Splits (`splits.py`)

Both scenarios first **exclude bot-authored PRs** (`pr.author_is_bot`) and restrict to
kept repos and in-window rows.

**Scenario A — known-project (primary).** Train: `created_at < 2026-01-01`. Test:
`created_at >= 2026-01-01`. Training rows are **capped at 5% per repo** by seeded random
subsample (`seed = 20260912`); test rows are never capped. (For Phase 4: cross-validation
inside training uses `TimeSeriesSplit` per blueprint §4 — the baseline has no
hyperparameters, so Phase 2 does not need it.)

**Scenario B — cold-start.** `GroupKFold(n_splits=5)` on repo id over all kept repos.
Each fold's training rows get the same 5% cap. The trailing-rate baseline is still
computable on a held-out repo because it uses only that repo's *own* history — which is
also the deployment reality when onboarding a new project. The cold-start claim is about
the *model* not having seen the repo, not about having no history.

Returns for each scenario a list of `(train_idx, test_idx)` over a canonical row order,
plus the row-share table used by the EDA report.

## 8. Metrics (`metrics.py`)

| Function | Definition |
|---|---|
| `precision_at_k(y_true, score, repo, k=10, seed)` | Per repo: sort by `score` desc, ties broken by seeded random permutation, take top `k` (or all if fewer), precision = mean(`y_true`). Return the per-repo series **and** its mean. Random tie-breaking is what makes a constant score honestly random. |
| `auc_pr(y_true, score)` | `sklearn.metrics.average_precision_score`, global over the test rows. |
| `cluster_bootstrap(per_repo_values, n=2000, seed)` | Resample **repos** with replacement, recompute the mean, report the 2.5/97.5 percentiles. This is the `[R1]` correction: the effective sample size for a cross-repo claim is the number of repos. |

## 9. Baseline (`baseline.py`)

Score = `trailing_90d_slow_rate` at `created_at`. Threshold 0.5 for the binary
prediction. Evaluated on Scenario A (single split) and Scenario B (5 folds, pooled
per-repo values). Reports Precision@10 (mean + cluster-bootstrap CI), AUC-PR, and the
base rate for comparison.

Appends one row per (scenario, fold) to `experiments.csv` via `tracking.py`:
`date, scenario, fold, model, features, params, n_train, n_test, precision_at_10,
p10_ci_lo, p10_ci_hi, auc_pr, base_rate, notes`.

P@10 vs base rate is a reported finding, not a leakage test: the trailing rate varies
within a repo over time, so within-repo top-10 can beat the base rate via temporal
autocorrelation. Leakage is tested directly by a replay audit (§11 #5).

## 10. EDA report (`eda_report.py`)

Writes `docs/phase2_eda.md` and `figures/*.png`. Sections, in order:

1. **Cohort table** — every repo: cell, kept / dropped + reason, in-window PRs, human
   PRs, bot share, row share, `is_slow` (D5), never-reviewed (D5).
2. **Label sensitivity** — D1–D5 global rates; per-repo D3 − D5 spread distribution;
   the look-at-these list.
3. **Wait-time distribution** — histogram of `log10(wait_h)` for reviewed PRs, overall and
   by star tier.
4. **Kaplan–Meier** — unreviewed fraction vs. hours since open, censored at 720h, with CI
   bands; overall and by star tier. The blueprint's named EDA deliverable.
5. **Threshold sensitivity** — `is_slow` rate at 72 / 120 / 168 / 240h, overall and per
   tier. Closes the blueprint's open item on whether 168h is right; the recommendation
   is stated, the default stays 168h unless the curve argues otherwise.
6. **Baseline** — the table from §9, both scenarios, with CIs.
7. **Phase 2 gate** — the checks in §11 with pass/fail.

## 11. Phase 2 gate — pre-registered

| # | Check | Pass |
|---|---|---|
| 1 | Repos kept after QC | ≥ 30 |
| 2 | D5 global `is_slow` rate | in [10%, 70%] |
| 3 | Scenario A test coverage | ≥ 20 repos with ≥ 10 test PRs each (Precision@10 well-defined) |
| 4 | Scenario B fold size | every fold holds out ≥ 6 repos |
| 5 | Replay audit: brute-force recomputation of backlog and trailing rate matches `features_at` on ≥200 seeded test rows per scenario | max \|Δrate\| < 1e-9, Δbacklog = 0 (**Hard stop** — replay leaks) |
| 6 | `gate_report.py` refactor regression | litestream and skills outputs identical pre/post |

Failing 1–4 means the cohort or the split is wrong and Phase 3 does not start.
Failing 5 means leakage in the replay. Failing 6 means the label implementation changed.

## 12. Out of scope

Feature engineering beyond the two replay features (Phase 3). Any model other than the
baseline (Phase 4). Survival analysis (Phase 5) — `wait_h_censored` and `event_observed`
are emitted for it but not used. The bot-authored-PR separate analysis (blueprint §2) is
a Phase 6 write-up item; Phase 2 only excludes and counts them.

## 13. Verification

1. `python cohort_qc.py` → `kept.json`; spot-check that any docs/guidelines repo was
   caught by the language rule, and that no repo was dropped for its label rate.
2. `python gate_report.py --repo benbjohnson/litestream` before and after the refactor;
   diff the `gate.json` files — must be byte-identical on the label fields.
3. `python baseline.py` → Precision@10 ≈ base rate on both scenarios. If it exceeds the
   CI, stop and audit `replay.py` before anything else.
4. Replay unit test: construct a 5-PR toy repo by hand with known backlog and slow-rate
   at each `t`; assert `features_at` reproduces them. Include one PR created 2 days
   before `t` with a known label to prove the resolvability predicate excludes it.
5. `python eda_report.py` → `docs/phase2_eda.md` renders; every figure present; gate
   table at the bottom.
6. Manual: pick 3 PRs from the look-at-these repos and check their D3 vs D5 first event
   against the GitHub UI.

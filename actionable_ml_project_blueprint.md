# actionable_ml_project_blueprint.md
## PRFlowPredict — PR review-stall risk prediction

> **Revised 2026-09-16** after the Phase 0 gate. Changes are marked `[R1]` inline.
> Evidence for each is in `docs/phase0_gate_results.md` and `docs/data_dictionary.md`.
> Original text is preserved in git history.
>
> **Status updated 2026-10-03** after Phase 8. Changes are marked `[R3]` inline.

---

## 1. Validated Problem & Approach

**Problem statement.** When a developer opens a pull request, it enters a review queue that nobody actively monitors. Most PRs get a first review within a day; some sit for weeks and quietly go stale. PRFlowPredict predicts, at the moment a PR is opened, whether it is likely to wait more than 7 days for its first human review — and surfaces the highest-risk open PRs so a team can act before the PR is forgotten.

**Target variable.** Binary: `is_slow = 1` if time from `created_at` to first non-author, non-bot review/comment exceeds 168 hours (7 days), else `0`. A PR still unreviewed after 30 days is `is_slow = 1` with certainty — no ambiguity, no censoring problem, because 30 days > 7 days. This is why the classification framing was chosen over raw regression: it converts your hardest data problem (censored wait times) into a non-issue.

**Validation evidence.** `[R1]` The pilot ran on two repos (in-stratum `benbjohnson/litestream` driving go/no-go; `anthropics/skills` as an adversarial case study). Litestream: 47.6% never reviewed at 30 days, `is_slow` positive rate 55.4%, right-skewed wait times — modelable, but the original 10–35% censoring expectation was a guess and was exceeded. Skills: 85.8% never reviewed, and ~8% of its "first reviews" come from non-affiliated commenters (median wait 13.2h → 0.9h when excluded). Consequence: the label definition is **deferred to Phase 2** and must be chosen on the full cohort, not assumed. The 168h threshold stays until the cohort's wait-time distribution is in.

**Why ML, versus simpler alternatives.** A rule-based heuristic ("flag if open > 3 days") ignores that normal wait time varies hugely by repo, author history, and current backlog — a 3-day wait is normal for a quiet repo and alarming for a fast-moving one. The baseline this must beat: **predict `is_slow` using only the repo's trailing 90-day slow-rate** (no PR-specific features at all). If the trained model can't beat that, PR-level signal isn't adding value, and that is a legitimate, reportable finding — not a failure to hide.

**Who consumes the output, and what decision it informs.** A tech lead or maintainer opening a dashboard on a given day, seeing their team's open PRs ranked by stall risk, and deciding which 2–3 to personally chase down or reassign for review. The decision informed is "where do I spend my next 10 minutes of triage."

**Prior art and this project's angle.** PR/code-review latency has an existing software-engineering research literature (search terms for the write-up: "pull request latency prediction," "code review time prediction"). Most prior work reports a single averaged accuracy or R² over a random or time-based split. This project's contribution is narrower and more honest: (1) explicit handling of never-reviewed PRs as censored rather than discarded, (2) evaluation under both a within-repo split and a leave-repos-out (cold-start) split, reported separately, and (3) a baseline the model must beat, stated up front.

---

## 2. Data & Modeling Specification

**Data source.** GitHub GraphQL API v4, personal access token, public repos only, no special scopes required.

**Repo selection.** `[R1]` **45 repos collected, subset to ~30 in Phase 2.** Stratified: 3 languages (Python, TypeScript, Go) × 3 star tiers (200–800 / 800–3,000 / 3,000–15,000), 5 per cell, drawn by seeded random sample within each cell (`select_repos.py --seed 20260912`; pool and raw searches persisted in `data/cohort/`). The spares exist because re-scraping is forbidden by design: degenerate repos (bot-dominated, mirrors, monorepos holding >10% of rows) are dropped in Phase 2 without a second collection, and Scenario B can hold out ~10 repos instead of 5. Selection criteria are structural only (language, stars, active, not archived, not a fork) — never selected on review behavior, which would bias the sample toward the outcome being predicted.

**Time window.** PRs opened 1 Jan 2024 – 30 Jun 2026 are the **training rows**. `[R1]` But the window is a row filter, not a collection filter: **every PR in each repo's history is collected** (cheap "Tier 1"), because the repo-state features below are not computable from a windowed slice — "open backlog at instant *t*" needs a PR opened in 2019 and still open in 2024. Measured on litestream: 9 PRs were open on the window's first day but created before it, and 33% of early-window authors had pre-window PRs. Collection stops well before "today" so every PR has ≥60 days of observed follow-up.

**Expected volume.** `[R1]` Measured: **49,757 in-window PRs across 45 repos** (~0.3 GB raw). Concentration: three repos hold ~10–11% of rows each; the top five hold 45%. Phase 2's subset-to-30 must address this.

**Known limitations.**
- Public-repo-only sample; may not generalize to private/enterprise review culture.
- Bot-authored PRs (Dependabot, Renovate) follow different timelines and are flagged (`is_bot_author`) at collection time, analyzed separately, and excluded from the primary training set by default.
- Author identity can be `null` for deleted accounts — handle as a category, not a crash.
- `reviewRequests` reflects current state — excluded. `[R1]` But the **`ReviewRequestedEvent`** is timestamped and point-in-time safe: a CODEOWNERS auto-request fires within seconds of open and is available at prediction time. Use events with `created_at <= pr.created_at + 60s`.
- `[R1]` **`authorAssociation` is computed at read time, not frozen at creation.** Proven: 40% of litestream authors read `CONTRIBUTOR` on their *first* PR, which is impossible unless later merges rewrote it. Since "later merged" correlates with "got reviewed," this is the outcome leaking into a feature. **Never use it.** Reconstruct author standing from history (below).
- `[R1]` **GitHub has PRs it cannot serve.** Deleted PRs (e.g. spam removals) stay in `totalCount` but return `INTERNAL` at every page size; the collector steps over them via a synthesized cursor and records them by number.

**Feature plan** `[R1]` — the original list called all of these "computable at `created_at`." The gate showed several are current-state snapshots, one is leaky, and the safe set is smaller than written. Classification per `docs/data_dictionary.md`:

- *Truly static at open:* `created_hour_utc`, `created_dayofweek`, `is_weekend`, `is_bot_author`, `is_cross_repository`, `author_account_age_days` (from `User.createdAt` — immutable and transfers to unseen repos).
- *Snapshot fields — MUST be reconstructed to at-open values from timeline/commit events, never used raw:*
  - `additions`, `deletions`, `changed_files` → the PR object holds the **final** diff. 47.6% of litestream PRs are multi-commit; reconstruct from commits with `authored_date <= created_at` (not `committed_date`, which rebase rewrites). `diff_is_exact` marks single-commit rows.
  - `is_draft` → 10.6% of PRs had a `ReadyForReviewEvent` after open.
  - `label_count` → 23.1% had `LabeledEvent` after open.
  - `title_length`, `base_branch` → `RenamedTitleEvent`, `BaseRefChangedEvent` (~5% each).
  - `body_length`, `has_body` → editable; use `last_edited_at` to flag, accept as approximate.
- *~~`author_association`~~ → **LEAKY, removed.** Replaced by author history below.
- *Author history (chronological replay over full Tier 1 history, strictly `< created_at`):* `is_first_pr_here`, `n_prior_prs_here`, `n_prior_merged_here` (using `merged_at < created_at`, not `merged == True`), `prior_merge_rate_here` (with a prior), `days_since_first_pr_here`.
- *Repo state (as-of `created_at`, from full history):* open PR backlog at that instant, PRs opened in trailing 7 days, trailing-90-day slow-rate — **with the resolvability predicate**: only PRs whose label was *knowable* at *t*, i.e. `created_at <= t − 168h` OR `first_human_event_at < t`. "Strictly earlier than the row" is insufficient; a PR opened 3 days ago has no label yet.
- *New — repo-level, transferable (the cold-start features Scenario B needs):* `n_assignable_users` (maintainer capacity), `owner_type` (Organization vs User), `has_codeowners`, `has_pr_template`, `has_contributing`, `n_ci_workflows`, `language_dominant`. Snapshots of HEAD at collection — a stated limitation.
- *New — point-in-time review request:* `reviewer_requested_at_open` from `ReviewRequestedEvent` within 60s of open.
- Repo identity is represented through behavioral features (above), not one-hot encoded — a one-hot column is meaningless for a repo absent from training, which would silently break the cold-start evaluation.

**Chosen models, and reasoning.**
1. **Baseline** — repo's trailing 90-day slow-rate, thresholded at 0.5. No learning. Every subsequent model is measured against this.
2. **Primary model — LightGBM binary classifier** predicting `is_slow`. Gradient boosting is the standard strong choice for a ~20-column tabular problem at this row count; a neural net would be slower to train and no better on data this size and shape.
3. **Secondary/depth model — survival analysis** (XGBoost `survival:aft` or `scikit-survival` gradient-boosted survival) predicting the full time-to-review distribution, using proper censoring at 30 days. Positioned as the technical-depth section of the report, not the product dependency — it demonstrates the more rigorous approach without gating the deliverable on it.

**Metrics, and why.**
| Framing | Metric | Real-world justification |
|---|---|---|
| Primary (classification) | Precision@10, AUC-PR | A tech lead only looks at the top few flagged PRs; AUC-PR (not ROC) because slow PRs are the minority class and ROC over-flatters imbalanced problems. |
| Secondary (survival) | Harrell's C-index | Natively handles censored observations; answers "did you rank the slower PR as slower," which is what stakeholder-facing ranking needs. |
| Reported, not headline | MAE on log-hours, uncensored rows only | Useful context, explicitly labeled as computed on a biased subset (rows with an actual review). |

**Train/validation/test split strategy — two schemes, both reported:**
- **Scenario A (known-project, primary):** time-based split — train on PRs before a cutoff date, test on PRs after it, within the same 30 repos. Reflects the realistic deployment: a team's own history predicting their own future.
- **Scenario B (cold-start):** leave-repos-out — train on 25 repos, test on the other 5, entirely unseen. Reflects onboarding a new project. Expected, and worth stating in advance, to score meaningfully worse than Scenario A — reporting that gap honestly is a feature of the project, not a weakness.
- Random row-level splits are explicitly rejected: they leak both temporal order and repo identity across train/test.

---

## 3. Execution Roadmap

| Phase | Milestone | Must-have / nice-to-have |
|---|---|---|
| 0 | `[R1]` **Done 2026-09-13.** Two-repo gate (in-stratum + adversarial), 10 pre-registered checks, all hard stops pass. `docs/phase0_gate_results.md` | Must-have — done |
| 1 | `[R1]` **Pipeline done; 45-repo collection launched 2026-09-16.** Two-stage (raw bytes → Parquet), two-tier (all-history thin + in-window rich), resumable, dead-PR bypass. ~4–5h | Must-have |
| 2 | `[R2]` **Done 2026-09-17.** D5 primary (46.4% slow; D3 44.0%, per-repo spread median 0.3pp, max 51pp on `michaelfeil/infinity`); 39/45 repos kept (5 bot-dominated, 1 language mismatch); baseline logged; gate PASS 5/5. `docs/phase2_eda.md` | Must-have — done |
| 3 | `[R2]` **Done 2026-09-21.** 50-column feature table (`data/features/features.parquet`, 38,462 rows) built by chronological replay; every replay feature proven by independent brute-force recomputation (max |Δ| = 0.0 on 500 rows); 5 rows verified against live GitHub (40/40). `docs/feature_dictionary.md` is generated from code. Found and fixed a Phase 1 parse bug (zeros collapsed to NaN) via re-parse, no re-scrape | Must-have — done |
| 4 | `[R2]` **Done 2026-09-22.** LightGBM on both splits + 3 pre-registered ablations, Optuna-tuned on Scenario A train only (CV AUC-PR 0.907); validity gate 5/5, refit delta 0.0. **A: P@10 0.769 [0.669, 0.856] vs baseline 0.585 — bar MET.** **But NO_LABEL_REPLAY on B scores AUC-PR 0.763 vs baseline 0.821 — within a project PR-level features carry signal; across projects they do not.** `docs/phase4_results.md` | Must-have — done |
| 5 (week 5) | `[R3]` **Not built.** Survival model, C-index, calibration check. So §4's Scenario B bar (C-index > 0.65) was never measured. | Nice-to-have (depth section) — cut |
| 6 | `[R3]` **Done 2026-09-23.** SHAP attribution on both FULL models: label-replay share 60.9% (A) vs 63.1% (B), so differential feature reliance does not explain the cold-start gap. Manual error analysis of the worst 50 predictions; newcomer-fairness check. `docs/phase6_interpretation.md` | Must-have — done |
| 6b | `[R3]` **Done 2026-09-27.** Pre-registered repo-fingerprinting test, verdict `PARTIAL_SHAP_ONLY`: the SHAP transfer test's outcome is `confirms`, the intervention's is `inconclusive`. `docs/phase6b_fingerprinting.md` | Added — done |
| 7 (week 7) | `[R3]` **Done 2026-10-03.** Streamlit demo: pick a repo and a day in the 2026 test period, see the PRs awaiting a first review ranked by risk, under the within-project or the cold-start model. `demo/` | Nice-to-have — done |
| 8 | `[R3]` **Done 2026-10-03.** Write-up: `README.md` and `docs/REPORT.md`, every cited result checked by `tests/test_writeup_claims.py` | Must-have — done |

**Checkpoints:** after Phase 0, confirm the labeling logic isn't degenerate before scaling collection. After Phase 3, confirm zero leakage by spot-checking 5 rows manually against raw timestamps. `[R2]` Both done: Phase 3's check was automated (brute-force twin, gate #2) AND performed against live GitHub (gate #5, `data/phase3_gate5_live.json`).

`[R2]` **This is what happened, in the precise form the blueprint anticipated:** Scenario B's FULL model beats the baseline, but strip the label-replay features and it does not (AUC-PR 0.763 vs 0.821, at-or-below in 4/5 folds). The cold-start advantage rests on the repo's own trailing rate, not on transferable PR-level structure. Phase 6's SHAP work should explain why. `[R3]` Phase 6 ruled out differential feature reliance; Phase 6b found the fingerprinting attribution pattern but not that it causes the gap (`PARTIAL_SHAP_ONLY`). Note also that B's raw P@10 (0.796) EXCEEDS A's (0.769) only because B's per-repo candidate pool is ~7x larger (median 433 vs 64 test PRs); that is a pool-size artifact, not cold-start superiority.

**Fallback if the primary approach stalls:** if the leave-repos-out split (Scenario B) collapses to near-baseline performance, that is not a dead end — pivot the report's framing to "within-project prediction is viable; cross-project transfer is not, and here's the SHAP evidence for why" (likely: repo culture dominates, and it's not transferable through the features collected). This is a legitimate, gradeable finding, not a failure state.

---

## 4. Evaluation & Risk Matrix

**Generalization verification.** Both split scenarios, always reported side by side, never just the flattering one. Cross-validation inside Scenario A uses `TimeSeriesSplit` (never random `KFold`); inside Scenario B uses `GroupKFold` on repo identity.

**Leakage risks specific to this project:**
- *Target leakage:* review comments/approvals/reviewer assignments must never appear as features — they don't exist at prediction time for an unreviewed PR. Verified by design, not just by code review.
- *Temporal leakage:* author-history and repo-state features must use only PRs strictly earlier than the row being predicted. `[R1]` **For label-derived features (trailing slow-rate) that is not enough:** a PR created at *t − 3 days* has no knowable label at *t*. Use only PRs whose label was resolvable by *t*: `created_at <= t − 168h` OR `first_human_event_at < t`. Verification method: implement a `features_at(pr, history_before_pr)` function and generate the table by chronological replay — this makes leakage structurally impossible rather than merely "checked for."
- *Group leakage:* the same repo's PRs must never split across train and test within Scenario B.

**Imbalance risk.** `is_slow` positive rate will likely sit well under 50%. Use `scale_pos_weight` in LightGBM and AUC-PR (not accuracy) as the primary metric — accuracy on an imbalanced target is close to meaningless here.

**Small-sample risk.** `[R1]` Scenario B's test set is ~10 repos (of 45 collected). The effective sample size for a *cross-repo* claim is the number of repos, not of PRs — so ~~bootstrap over PRs, stratified by repo~~ is the wrong resampling scheme: it holds the repo set fixed and yields intervals far too narrow. Use a **cluster bootstrap resampling whole repos**, and say explicitly in the write-up that ~10 held-out repos is still a small sample for the cold-start claim.

**Bias/fairness checks worth doing.** Check whether `is_slow` predictions differ systematically for first-time contributors vs. repeat contributors, and for PRs authored outside the maintainers' typical active hours (timezone proxy). If the model is more pessimistic about first-time contributors than warranted by actual outcomes, that's worth a paragraph — a triage tool that systematically deprioritizes newcomers' PRs would compound an existing open-source problem rather than fix one.

**What "good enough" looks like, numerically.** Precision@10 on Scenario A beats the trailing-90-day baseline by a clear, stated margin (even 5–10 percentage points is a real, reportable result at this data size). C-index > 0.65 on Scenario B is a reasonable bar for "the model learned something that transfers across projects" — 0.5 is coin-flip.

---

## 5. Reproducibility & Documentation

**Experiment tracking.** A single `experiments.csv` (or MLflow if you want a tool for the CV line-item) logging: date, split scenario, model, features included, hyperparameters, and every metric — not just the best run's numbers. This is what lets you write an honest "here's what didn't work" section.

**What gets documented for future-you.** The raw→processed data lineage (`README.md`), the exact label definition — `[R1]` including *which review streams count* and *why* (the skills spam finding), with the sensitivity table across definitions — and the point-in-time replay function's assumptions. `docs/data_dictionary.md` already records which columns are safe, snapshot, or leaky; keep it current.

**Environment.** `requirements.txt` pinned versions for `lightgbm`, `xgboost`, `scikit-survival`, `pandas`, `pyarrow`; Python version noted. Raw JSON and processed Parquet kept separate on disk so a parsing bug never requires re-scraping GitHub.

---

## 6. Resources & Tools

- **Collection:** GitHub GraphQL API v4 + a Python `gql` client or raw `requests`.
- **Modeling:** `lightgbm`, `xgboost` (AFT objective), `scikit-survival`, `lifelines` (Kaplan–Meier plot for the EDA section).
- **Tuning:** `optuna` for hyperparameter search on the classifier.
- **Interpretation:** `shap` for feature attribution.
- **Demo:** `streamlit`, single page, repo selector + ranked table.
- **Where to get unstuck:** GitHub GraphQL API docs and explorer (developer.github.com/graphql), `scikit-survival` docs for censored-data handling, and — if the censoring/AFT framing gets confusing — searching "concordance index tutorial" tends to clarify faster than the sklearn-survival docs alone.

---

## Open items `[R1]`

Resolved by the gate: censoring rate (47.6% in-stratum), positive rate (55.4%), snapshot contamination rates, `authorAssociation` mutability, review-connection sort key (`submittedAt`).

**Still open, for Phase 2 on cohort data:**
1. **Label definition** — D3 (blueprint: any non-author non-bot review *or comment*) vs D5 (same, excluding `NONE`-association commenters). Stable on the in-stratum repo (3.7pp spread), decisive on the adversarial one. Decide on 45 repos; report the sensitivity table either way.
2. **168h threshold** — chosen as a default, not derived. Re-examine against the cohort's wait-time histogram; keep 168h unless the distribution argues otherwise.
3. **Subset 45 → 30** — drop degenerate repos and address row concentration (`Scottcjn/Rustchain` at 5,262 PRs for a 200–800-star repo is anomalous).

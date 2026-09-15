# actionable_ml_project_blueprint.md
## PRFlowPredict — PR review-stall risk prediction

---

## 1. Validated Problem & Approach

**Problem statement.** When a developer opens a pull request, it enters a review queue that nobody actively monitors. Most PRs get a first review within a day; some sit for weeks and quietly go stale. PRFlowPredict predicts, at the moment a PR is opened, whether it is likely to wait more than 7 days for its first human review — and surfaces the highest-risk open PRs so a team can act before the PR is forgotten.

**Target variable.** Binary: `is_slow = 1` if time from `created_at` to first non-author, non-bot review/comment exceeds 168 hours (7 days), else `0`. A PR still unreviewed after 30 days is `is_slow = 1` with certainty — no ambiguity, no censoring problem, because 30 days > 7 days. This is why the classification framing was chosen over raw regression: it converts your hardest data problem (censored wait times) into a non-issue.

**Validation evidence.** A 300-PR single-repo pilot (`pallets/flask` or similar) will confirm before full build-out: (a) a non-trivial censoring rate at 30 days (expect 10–35%), (b) a right-skewed wait-time distribution, and (c) enough class balance in `is_slow` to model without extreme reweighting. If the pilot shows near-0% or near-100% never-reviewed, the review definition or window needs adjusting before scaling to 30 repos.

**Why ML, versus simpler alternatives.** A rule-based heuristic ("flag if open > 3 days") ignores that normal wait time varies hugely by repo, author history, and current backlog — a 3-day wait is normal for a quiet repo and alarming for a fast-moving one. The baseline this must beat: **predict `is_slow` using only the repo's trailing 90-day slow-rate** (no PR-specific features at all). If the trained model can't beat that, PR-level signal isn't adding value, and that is a legitimate, reportable finding — not a failure to hide.

**Who consumes the output, and what decision it informs.** A tech lead or maintainer opening a dashboard on a given day, seeing their team's open PRs ranked by stall risk, and deciding which 2–3 to personally chase down or reassign for review. The decision informed is "where do I spend my next 10 minutes of triage."

**Prior art and this project's angle.** PR/code-review latency has an existing software-engineering research literature (search terms for the write-up: "pull request latency prediction," "code review time prediction"). Most prior work reports a single averaged accuracy or R² over a random or time-based split. This project's contribution is narrower and more honest: (1) explicit handling of never-reviewed PRs as censored rather than discarded, (2) evaluation under both a within-repo split and a leave-repos-out (cold-start) split, reported separately, and (3) a baseline the model must beat, stated up front.

---

## 2. Data & Modeling Specification

**Data source.** GitHub GraphQL API v4, personal access token, public repos only, no special scopes required.

**Repo selection.** ~30 repos, stratified: 3 languages (e.g. Python, TypeScript, Go) × 3 star tiers (200–800 / 800–3,000 / 3,000–15,000), ~3–4 repos per cell. Selection criteria are structural only (language, stars, active, not archived, not a fork) — never selected on review behavior, which would bias the sample toward the outcome being predicted.

**Time window.** PRs opened 1 Jan 2024 – 30 Jun 2026. Collection stops well before "today" so every PR has ≥60 days of observed follow-up — no PR is falsely censored just because data collection happened too early.

**Expected volume.** ~1,000–2,500 PRs/repo × 30 repos ≈ 35,000–60,000 rows. Comfortably laptop-scale as Parquet.

**Known limitations.**
- Public-repo-only sample; may not generalize to private/enterprise review culture.
- Bot-authored PRs (Dependabot, Renovate) follow different timelines and are flagged (`is_bot_author`) at collection time, analyzed separately, and excluded from the primary training set by default.
- Author identity can be `null` for deleted accounts — handle as a category, not a crash.
- `reviewRequests` reflects current state, not PR-creation-time state — excluded from features entirely.

**Feature plan** (all computable at `created_at`, i.e., point-in-time correct):
- *Static:* `additions`, `deletions`, `changed_files`, `title_length`, `body_length`, `has_body`, `is_draft`, `label_count`, `base_branch`, `created_hour_utc`, `created_dayofweek`, `is_weekend`, `author_association`, `is_bot_author`.
- *Author history (expanding window, `.shift(1)` before aggregating):* prior PR count in this repo, prior merge rate, days since first PR here, is-first-time-contributor flag.
- *Repo state (rolling window, as-of PR creation):* open PR backlog at that instant, PRs opened in trailing 7 days, trailing-90-day slow-rate (fraction of PRs that were `is_slow`).
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
| 0 (1 evening) | Single-repo pilot (300 PRs): full pipeline, histogram of wait times, censoring rate | Must-have — gate before scaling |
| 1 (week 1) | Full collection running across 30 repos; raw JSON persisted before parsing | Must-have |
| 2 (week 2) | Label construction, EDA, Kaplan–Meier curve of unreviewed-fraction over time, baseline computed | Must-have |
| 3 (week 3) | Point-in-time feature engineering, replay-validated (see Risk Matrix) | Must-have — budget extra time here |
| 4 (week 4) | LightGBM classifier trained and evaluated on both splits | Must-have |
| 5 (week 5) | Survival model, C-index, calibration check | Nice-to-have (depth section) — cut first if time runs short |
| 6 (week 6) | SHAP feature attribution, manual error analysis on worst 50 predictions | Must-have |
| 7 (week 7) | Streamlit demo: pick a repo, see open PRs ranked by risk | Nice-to-have but high payoff for the demo |
| 8 | Write-up | Must-have |

**Checkpoints:** after Phase 0, confirm the labeling logic isn't degenerate before scaling collection. After Phase 3, confirm zero leakage by spot-checking 5 rows manually against raw timestamps.

**Fallback if the primary approach stalls:** if the leave-repos-out split (Scenario B) collapses to near-baseline performance, that is not a dead end — pivot the report's framing to "within-project prediction is viable; cross-project transfer is not, and here's the SHAP evidence for why" (likely: repo culture dominates, and it's not transferable through the features collected). This is a legitimate, gradeable finding, not a failure state.

---

## 4. Evaluation & Risk Matrix

**Generalization verification.** Both split scenarios, always reported side by side, never just the flattering one. Cross-validation inside Scenario A uses `TimeSeriesSplit` (never random `KFold`); inside Scenario B uses `GroupKFold` on repo identity.

**Leakage risks specific to this project:**
- *Target leakage:* review comments/approvals/reviewer assignments must never appear as features — they don't exist at prediction time for an unreviewed PR. Verified by design, not just by code review.
- *Temporal leakage:* author-history and repo-state features must use only PRs strictly earlier than the row being predicted. Verification method: implement a `features_at(pr, history_before_pr)` function and generate the table by chronological replay — this makes leakage structurally impossible rather than merely "checked for."
- *Group leakage:* the same repo's PRs must never split across train and test within Scenario B.

**Imbalance risk.** `is_slow` positive rate will likely sit well under 50%. Use `scale_pos_weight` in LightGBM and AUC-PR (not accuracy) as the primary metric — accuracy on an imbalanced target is close to meaningless here.

**Small-sample risk.** Scenario B's test set is only 5 repos — report the C-index/precision with a confidence interval (bootstrap over PRs, stratified by repo) rather than a bare point estimate, and say explicitly in the write-up that 5 held-out repos is a small sample for the cold-start claim.

**Bias/fairness checks worth doing.** Check whether `is_slow` predictions differ systematically for first-time contributors vs. repeat contributors, and for PRs authored outside the maintainers' typical active hours (timezone proxy). If the model is more pessimistic about first-time contributors than warranted by actual outcomes, that's worth a paragraph — a triage tool that systematically deprioritizes newcomers' PRs would compound an existing open-source problem rather than fix one.

**What "good enough" looks like, numerically.** Precision@10 on Scenario A beats the trailing-90-day baseline by a clear, stated margin (even 5–10 percentage points is a real, reportable result at this data size). C-index > 0.65 on Scenario B is a reasonable bar for "the model learned something that transfers across projects" — 0.5 is coin-flip.

---

## 5. Reproducibility & Documentation

**Experiment tracking.** A single `experiments.csv` (or MLflow if you want a tool for the CV line-item) logging: date, split scenario, model, features included, hyperparameters, and every metric — not just the best run's numbers. This is what lets you write an honest "here's what didn't work" section.

**What gets documented for future-you.** The raw→processed data lineage (Phase 1 note above), the exact label definition (7-day threshold, 30-day window, human-non-bot review), and the point-in-time replay function's assumptions — these three are the pieces you will not remember in 8 weeks.

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

## Open item for next revision

Confirm after the Phase 0 pilot: actual censoring rate, actual `is_slow` positive rate, and whether 7 days is the right threshold for this data (it was chosen as a reasonable default, not derived from your specific repos). Bring the histogram back and we'll adjust the threshold and re-check the baseline before you commit to it in the proposal.

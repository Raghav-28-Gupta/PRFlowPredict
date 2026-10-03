# Phase 7 Design — The Demo

Written for: whoever implements Phase 7.

**Status:** approved in design review 2026-10-03; awaiting spec review.
**Depends on:** Phases 0–4, 6, 6b and 8, all merged to `main` (PRs #1–#7).
**Produces:** a Streamlit app that replays the evaluated 2026 test period. A visitor picks a
repo and a day and sees the PRs awaiting their first review, ranked by risk, under the
within-project model or the cold-start model. It also produces the committed data extract
the app runs on, and the script that builds that extract.

---

## 1. Purpose and audience

The audience is the same as Phase 8's: **hiring managers and technical interviewers**, who
reach the demo from the README. It is the blueprint's Phase 7 ("pick a repo, see open PRs
ranked by risk") and serves the decision named in blueprint §1: a maintainer opening a
dashboard on a given day and choosing which 2–3 PRs to chase.

**Success criterion:** within a minute, a visitor sees one repo's PRs awaiting review on a
chosen day, ranked by risk, next to what actually happened to each. They can switch to the
cold-start model and watch the ranking change. Nothing in the app lets them believe something
the report does not support.

## 2. Decisions locked in design review

| Decision | Choice |
|---|---|
| Data source | **Historical replay** of the evaluated test period (2026-01-01 to 2026-06-30) from a committed extract. No live GitHub mode |
| Models shown | A toggle: **Seen in training** (Scenario A model) and **Unseen repo** (the Scenario B model of the fold that held that repo out) |
| Publishing | Deployable on Streamlit Community Cloud from committed files only. **The user deploys** from their own account, and the README links the URL afterwards |
| The list | PRs **awaiting their first review** at the chosen moment, ranked by the score each got when it was opened |
| Outcomes | **Shown beside each PR**, plus a one-line tally for the top 3 |
| Explanations | **Top 3 drivers per PR per model**, from exact TreeSHAP precomputed offline, labelled as model attribution rather than cause |
| Runtime | **Committed extract only.** No model, sklearn, lightgbm or shap at deploy time |

## 3. Artifacts

| File | Change |
|---|---|
| `build_demo_data.py` | New. Runs locally only: reads the gitignored Phase 4 outputs and writes the extract (§4) |
| `demo/data/prs.parquet` | New, generated, **committed**. The extract (§4) |
| `demo/triage.py` | New. Pure functions: who is awaiting review at a moment, ranking, tally (§5). Imports pandas only |
| `demo/app.py` | New. A thin Streamlit layer over `triage` (§6) |
| `demo/requirements.txt` | New. Only what the deployed app needs: `streamlit`, `pandas`, `pyarrow`, pinned to the versions the root file uses |
| `requirements.txt` | Adds `streamlit` (a Phase 7 section), so the tests can run the app headless |
| `tests/test_build_demo_data.py`, `tests/test_demo_triage.py`, `tests/test_demo_extract.py`, `tests/test_demo_app.py` | New (§9) |
| `README.md`, `docs/REPORT.md`, `actionable_ml_project_blueprint.md` | Small updates (§8) |

## 4. The extract (`build_demo_data.py` → `demo/data/prs.parquet`)

**Rows.** One per Scenario A test PR: exactly the rows of `data/predictions/A_FULL_fold0.parquet`
(14,135 PRs, 39 repos, opened 2026-01-01 to 2026-06-30), **in that file's row order**, so
Phase 4's seeded P@10 tie-breaking can be reproduced from the extract (§9).

**Columns.**

| Column | Source |
|---|---|
| `repo`, `pr_id`, `created_at`, `is_slow` | the A predictions file |
| `number`, `url`, `title`, `closed_at` | `data/processed/<repo>/pr_tier2.parquet` (`title` is `title_current`, the PR's title at collection time) |
| `first_review_at` | `created_at + wait_h` hours from `data/features/features.parquet`. Null when `wait_h` is null, meaning never reviewed. This is the D5 first human event, the same one the label uses |
| `score_a` | `p_hat` from `A_FULL_fold0.parquet` |
| `fold_b`, `score_b` | the B run whose `test_repos` (in `data/phase4_runs.json`) contains the PR's repo, and `p_hat` from `B_FULL_fold{fold_b}.parquet` for the same `pr_id` |
| `baseline_score` | from the A predictions file. Kept only for the integrity test (§9); the app does not show it |
| `drivers_a`, `drivers_b` | top 3 drivers (below) |

**Author identities are not included**, neither logins nor IDs. The demo is about PRs, not
people.

**Drivers.** For each scenario, exact TreeSHAP via Phase 6's `attribution.prepare()` and
`attribution.explain()`. The A model explains every row; for B, the fold model `fold_b`
explains its own repos' rows, over the FULL feature set as Phase 4 trained it. The three
features with the largest |SHAP| become one string, in the form
`author's past slow rate here = 0.92 ↑ · open backlog = 41 ↑ · has a PR template = no ↓`:
- **Labels:** a friendly label for every FULL feature, from a `DRIVER_LABELS` map in the
  script.
- **Values:** rates and shares to 2 decimals, counts as integers, booleans as yes/no, and a
  missing value as "missing".
- **Arrows:** ↑ means the feature pushed the risk up (positive SHAP on the raw margin), ↓
  means it pushed the risk down.

**Build checks.** The script refuses to write the extract unless:
- every A test row joins to `pr_tier2`, to the feature table and to exactly one B fold
  prediction;
- SHAP additivity holds for both scenarios: `attribution.additivity_delta` is below 1e-6 on
  every row;
- every FULL feature has a `DRIVER_LABELS` entry.

**Size.** About 1–3 MB of parquet.

## 5. Triage logic (`demo/triage.py`)

- **The moment.** The chosen day at 00:00 UTC, call it `t`: a maintainer opening the
  dashboard at the start of the day.
- **Awaiting first review at `t`** means `created_at < t`, and `closed_at` is null or
  `closed_at > t`, and `first_review_at` is null or `first_review_at > t`. A PR reviewed at
  exactly `t`, or closed at exactly `t`, is no longer awaiting review.
- **Ranking.** Descending by the chosen scenario's score. Ties are broken by `created_at`
  ascending (the older PR first), then `number`, so the order is deterministic.
- **Days waited so far** is `(t − created_at)` in days.
- **Outcome.** "reviewed after N days" from `first_review_at`, or "never reviewed". Also
  "stalled" when `is_slow` is true, meaning no first review within 7 days.
- **Tally.** For the top `k = 3` rows, how many stalled, as an "N of 3" count. When fewer
  than 3 are awaiting review, `k` is the list length.
- **PRs opened before 2026-01-01 have no score.** They never appear, even if they were still
  open on the chosen day.

## 6. The app (`demo/app.py`)

**Controls (sidebar):**
- **Repo:** a select box of the 39 repos.
- **Day:** a date input limited to 2026-01-02 to 2026-06-30.
- **Model:** a radio with "Seen in training (within-project)" and "Unseen repo
  (cold-start)".

**Defaults:** day 2026-04-01, and the repo with the most PRs awaiting review at that moment,
with ties broken alphabetically. Both are computed from the extract by `triage`.

**Main area, in order:**
1. A short header: what the app is (a replay of the evaluated 2026 test period, not live
   data), and that each score is the one the model gave when the PR was opened, never
   updated afterwards.
2. One line describing the selected model. Unseen: "scores from the model trained without
   this repo, as if it were new". Then one sentence of context with no numbers, pointing to
   the README and report for results: within a project the model ranks well, while on unseen
   repos it leans on the repo's slow-rate history.
3. The tally line for the top 3.
4. The table, one row per PR awaiting review: rank, risk score (2 decimals), PR (number
   linked to its URL, then the title), days waited, top drivers, outcome.
5. A caption under the table: drivers are model attribution, not cause, and the title shown
   is the title at collection time.

**Empty list.** When no PR is awaiting review at the moment, a message says so. On early
January days it adds that only PRs opened from 2026-01-01 have scores.

**Performance.** The extract is loaded once with `st.cache_data`.

## 7. Honesty constraints

1. The app never presents the replay as live, and never implies that scores update as a PR
   waits.
2. **The app hard-codes no result number** (no AUC-PR, no P@10). Results live in the README
   and report, where `tests/test_writeup_claims.py` checks them. Everything the app displays
   is computed from the extract at runtime.
3. Drivers are always labelled as model attribution, not cause.
4. The cold-start view is shown alongside the within-project view, never hidden behind it.
5. Nothing in the app or in the docs changes made by this phase may use a phrasing in
   `writeup_claims.RETRACTED`.

## 8. Docs and status changes

- **`README.md`:**
  - A "Try the demo" section after the result: one sentence on what it shows, then
    `pip install -r demo/requirements.txt` and `streamlit run demo/app.py`. The public URL
    is added once the user has deployed.
  - The repo map gains a Demo row: `build_demo_data.py` · `demo/app.py` · `demo/triage.py`.
- **`docs/REPORT.md` §10:** `python build_demo_data.py` after `report6b.py` in the full
  pipeline, and `streamlit run demo/app.py` in the committed-state commands.
- **Blueprint, Phase 7 row:** status changes to done, with the completion date and a pointer
  to `demo/`, and a `[R3]` marker as on the other status rows.

## 9. Testing

- **`triage` units**, on small synthetic frames:
  - a PR created after `t`;
  - one closed before `t`, and one closed exactly at `t`;
  - one reviewed before `t`, and one reviewed exactly at `t`;
  - one never reviewed;
  - tied scores, which resolve by age and then number;
  - the tally when fewer than 3 PRs are listed;
  - an empty list;
  - the default-repo choice, including its alphabetical tie-break.
- **Extract integrity**, against the committed `demo/data/prs.parquet` and
  `data/phase4_runs.json`. These run from a plain clone:
  - recomputed with `metrics.auc_pr`, the A AUC-PR from `score_a` and the baseline AUC-PR
    from `baseline_score` equal the A FULL run's `auc_pr` and `baseline_auc_pr` to 1e-12;
  - recomputed with `metrics.precision_at_k` in the extract's row order, P@10 from `score_a`
    equals the run's `precision_at_10` to 1e-12;
  - for every B fold `k`, the set of repos with `fold_b == k` equals that run's
    `test_repos`;
  - 14,135 rows, unique `pr_id`, no null score, and three non-empty drivers per scenario;
  - no author-identity column is present.
- **Build-script units**, on synthetic data: driver formatting (rates, counts, booleans,
  missing values, arrows), and `DRIVER_LABELS` covering every FULL feature in
  `featuresets.FEATURE_SETS["FULL"]`.
- **App smoke test** with Streamlit's `streamlit.testing.v1.AppTest`:
  - the app runs headless without an exception and renders the table;
  - switching the model radio and changing the day re-render without error, and switching
    the model changes the risk scores shown;
  - a day with no awaiting PRs shows the empty-list message;
  - `demo/app.py` contains no hard-coded result number (no match for the regex
    `\b0\.\d{3}\b`) and no phrasing from `writeup_claims.RETRACTED` (§7).
- **The existing 392 tests** still pass. This includes the Phase 8 claims test over the
  edited README and REPORT.

## 10. Out of scope

- Live GitHub mode, and scoring PRs outside the test period.
- A baseline column or baseline ranking in the app. The baseline's scores barely vary within
  a repo, so any top-3 it produced would depend on how ties were broken.
- Per-repo metrics inside the app.
- Retraining, new models or new statistics.
- Deploying the app, which the user does from their own account.

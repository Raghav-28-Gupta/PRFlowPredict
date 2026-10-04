# Demo Story Design — an interactive, visual walkthrough of the demo

Written for: whoever implements the demo story.

**Status:** approved in design review 2026-10-04.
**Depends on:** Phase 7 (the demo, PRs #8, #10, #11), merged to `main`.
**Produces:** the deployed Streamlit demo, restructured into six chapters for a 3–5 minute live
presentation, with interactive Altair charts, a guess-the-stall game and hidden speaker notes.

---

## 1. Purpose and audience

The owner presents the demo **live** (screen share or in the room) to interviewers and hiring
managers, in a **3–5 minute** slot. The same app stays public behind the README link.

**Success criterion:** in 3–5 minutes, a viewer understands the problem, watches the model work
on real PRs, sees why it decides as it does, tries it against their own judgement, and leaves with
the honest headline: it works within a project, and on unseen repos its advantage depends on the
repo's slow-rate history. Nothing on screen may let a viewer believe something the report and the
committed artifacts do not support.

## 2. Decisions locked in design review

| Decision | Choice |
|---|---|
| Use | Presented **live** by the owner; also public |
| Length | **3–5 minutes** |
| Shape | A **guided story in six chapters**, each with one interactive visual, Back/Next between them |
| Structure | **Native multipage**: `st.navigation` with one `st.Page` per chapter, each with its own URL |
| Visuals | Hoverable results chart; per-PR "why" chart; repo timeline explorer; guess-the-stall game; wait-time histogram |
| Game | **Pick the one that stalled**: 4 PRs from the same repo and ISO week, exactly one stalled |
| Opening | **Wait-time distribution** |
| Cold-start | **Results chart plus the Seen/Unseen switch**; no 6b scatter |
| Presenter help | **Speaker notes** per chapter, behind a sidebar toggle that is off by default |
| Explanation data | **All values**: every PR's SHAP value for all 37 features under both models, plus its feature values (extract grows from 1.7 MB to about 8 MB) |
| Charts | **Altair** (already installed with Streamlit), now pinned explicitly |

## 3. Artifacts

| File | Change |
|---|---|
| `build_demo_data.py` | Also writes per-PR SHAP values, base values and feature values (§4), and `demo/data/features.json` |
| `demo/data/prs.parquet` | Regenerated, committed, with the new columns (§4) |
| `demo/data/features.json` | New, generated, committed: the 37 FULL features in model order, each with its label and dtype |
| `demo/triage.py` | Gains the logic for each chapter (§5). Still imports no Streamlit |
| `demo/charts.py` | New: pure functions that return Altair charts (§6) |
| `demo/chapters.py` | New: one function per chapter, plus the speaker notes (§7) |
| `demo/views/*.py` | New: one two-line page file per chapter, each calling its function in `chapters.py`. Streamlit's AppTest keeps the current chapter between runs only for file pages, which the tests need |
| `demo/app.py` | Becomes a thin shell: page config, sidebar, navigation (§7) |
| `demo/requirements.txt`, `requirements.txt` | Add `altair==6.3.0` |
| `tests/test_demo_*.py` | Extended and migrated to the chapter structure (§9) |
| `README.md`, `docs/design/specs/2026-10-03-phase7-demo-design.md` | Short updates (§8) |

## 4. The extract additions

**New columns in `demo/data/prs.parquet`.** The row set and row order are unchanged (§4 of the
Phase 7 spec), so every existing integrity test keeps passing.

| Columns | Content |
|---|---|
| `shap_a__<feature>`, `shap_b__<feature>` (37 each, float32) | Exact TreeSHAP value per feature on the raw margin: the A model for every row, and the B fold model `fold_b` for its own repos' rows |
| `base_a`, `base_b` (float64) | The explainer's expected value: one A value; for B, the value of the row's fold model |
| `x__<feature>` (37) | The model input the SHAP values explain, from `data/features/features.parquet` |

**`demo/data/features.json`.** A list, in `featuresets.FEATURE_SETS["FULL"]` order, of
`{"feature", "label", "dtype", "rate"}`. `label` comes from `build_demo_data.DRIVER_LABELS`,
`dtype` from `features.COLUMN_SPEC`, and `rate` from `build_demo_data.RATES`. This is the single source of feature labels for the app.

**One formatter.** `format_value` moves to `demo/triage.py`. It takes a feature's dtype and
whether it is a rate, and `build_demo_data.py` imports it, so the drivers text in the extract and
every value the app shows come from the same code.

**Build checks**, added to the existing ones:
- SHAP additivity holds per row, as before: below 1e-6 on the raw margin before the cast to
  float32;
- `base + sum(shap)` reproduces `logit(score)` for both models;
- `features.json` lists exactly the FULL features.

## 5. Logic (`demo/triage.py`)

Everything is pure pandas plus the standard library (`math`, `json`, `re`, `random`), unit-tested
on synthetic frames, and computed at runtime from committed files.

- **`wait_buckets(prs)`.** The share and count of PRs whose first review came within 1 hour,
  1 hour to 1 day, 1 to 7 days, or after more than 7 days, plus two buckets for no first review:
  "closed without a review" and "never reviewed, still open". All six buckets sum to every PR.
  The 7-day line falls between the third and fourth buckets.
- **`wait_summary(prs)`.** Chapter 1's headline shares: reviewed within a day; stalled; of the PRs
  closed without a review, how many closed within a day of opening; and how many were still open
  and unreviewed a week after opening.
- **`timeline(prs, repo, scenario, at)`.** One row per PR in the repo: `created_at`, `risk`
  (the chosen model's score), `stalled`, `waiting` (awaiting review at `at`, by the existing
  `awaiting_review` rule), `pr_id`, `number`, `title`, `outcome`.
- **`selected_pr_id(event)`.** The `pr_id` from an Altair point-selection event dict, or `None`
  when nothing is selected.
- **`why(pr, scenario, features, k=8)`.** The k features with the largest |SHAP| for that PR
  and model, each with its label, formatted value and SHAP value, plus an "all other features"
  row (the sum of the rest), plus the base value. Invariant: `base + sum(rows) = logit(score)`.
- **`importance(path)`.** Phase 6's committed `data/phase6_importance_{A,B}.csv`: feature, label,
  share.
- **The game.**
  - `eligible_weeks(prs)`: the (repo, ISO week of `created_at`, UTC) groups with at least one
    stalled PR and at least three that did not stall.
  - `deal(prs, rng)`: choose an eligible group uniformly, one stalled PR uniformly from it, three
    distinct non-stalled PRs uniformly from it, and shuffle the four.
  - `model_pick(round, scenario)`: the highest score among the four; ties go to the lower
    number.
  - `game_hit_rate(prs, scenario)` is the exact probability, under `deal`'s draw, that
    `model_pick` is the stalled PR. It is the mean over eligible groups of the mean over their
    stalled PRs `s` of `C(L_s, 3) / C(N, 3)`. Here `N` is the group's number of non-stalled PRs
    and `L_s` is how many of them `model_pick` would rank below `s`: a lower score, or the same
    score and a higher number. That makes the formula exact under the tie-break. The
    random-guess rate is 1/4. In the design-review check this came to 53.4% for the Seen model
    and 51.4% for the Unseen model, against 25% for random.
- **`results(runs_path)`.** From the committed `data/phase4_runs.json`:
  - per scenario: AUC-PR for FULL, NO_LABEL_REPLAY and the baseline, as the mean over folds plus
    each fold's value;
  - Scenario A's P@10 with its interval, the baseline's P@10, and the random-pick rate
    (`base_rate_p10`).

  Every number is formatted the way `writeup_claims.py` registers it.

## 6. Charts (`demo/charts.py`)

Pure functions that take frames from §5 and return an `altair.Chart`. They touch no
Streamlit, so tests inspect `chart.to_dict()`. Altair's 5,000-row limit is lifted, because the
largest repo has 5,178 replayed PRs.

- **`wait_histogram(buckets)`.** Bars in bucket order, a labelled 7-day rule, and a tooltip with
  count and share.
- **`timeline(frame, at)`.**
  - Each PR is a point: x = opened, y = risk, colour = stalled or not.
  - PRs waiting at `at` are drawn emphasised; the rest are faded.
  - A vertical rule marks `at`.
  - The tooltip shows number, title, risk and outcome.
  - A point selection on `pr_id`, named `pick`.
- **`why_bars(rows)`.** Horizontal bars sorted by |SHAP|: red for pushing the risk up, blue for
  pushing it down. The axis is labelled "push on the model's score (log-odds)". The tooltip
  shows the feature value.
- **`importance_bars(frame, top=10)`.** The share of mean |SHAP|, from Phase 6.
- **`results_chart(results)`.** Grouped bars of mean AUC-PR by scenario and feature set, with
  B's per-fold values as hoverable dots. Colours and labels match Figure 1.
- **`p10_chart(results)`.** Model vs baseline vs random pick for Scenario A, with the model's
  interval.

## 7. The app

**Shell (`demo/app.py`).**
- Page config, the cached data loads (extract, `features.json`, results, importance) and the
  sidebar.
- `st.navigation` over the six chapters' page files in `demo/views/` (§3), then `run()`.
- It reloads its own modules (`triage`, `charts`, `chapters`) on every run, extending the stale
  module fix from PR #11.

**Sidebar, on every chapter:** the chapter list; a **model** switch (Seen in training /
Unseen repo), used by chapters 2–5; a **repo** picker, used by chapters 2–3; a **Speaker notes**
toggle, off by default.

**Shared state in `st.session_state`:**
- `_ctx`: this session's context, which `app.py` sets on every run. Never a module global: viewers
  share one server process, and callbacks run before the script does.
- `selected_pr`: a `pr_id`. A timeline selection left over from another repo is ignored.
- `game`: the current round, whether it has been revealed, the model it was revealed with, and a
  tally per model. A second Reveal on the same round counts nothing.

**Chapters (`demo/chapters.py`).** Each chapter shows a title, one visual, at most two short
paragraphs, its speaker note when the toggle is on, and Back/Next buttons (`st.switch_page`).

| # | URL | Content |
|---|---|---|
| 1 | `problem` | One-sentence framing and `wait_histogram`. Its text quotes the shares computed by `wait_buckets` and calls them the replayed period's, not the whole cohort's |
| 2 | `watch-it-work` | A day slider (2026-01-02 to 2026-06-30, default 2026-04-01) and `timeline`; clicking a point sets `selected_pr`. Below it, the Phase 7 triage list for that day, with its tally line and caveats unchanged. The default repo is the one with the most PRs waiting on the default day |
| 3 | `why` | The lookup box and Random PR, from PR #10, which set `selected_pr`. The selected PR's card, `why_bars`, and `importance_bars` for the chosen model. With nothing selected, it uses the riskiest PR waiting on the default day in the chosen repo, or the repo's riskiest PR if none was waiting, and says how to pick another |
| 4 | `test-yourself` | The game. It draws from every eligible group in the replay and ignores the repo picker. Four cards show only what was known when each PR opened: title, lines added and deleted, commits, author's earlier PRs here, author's past slow rate here, repo's recent slow rate, description length, and whether it opened as a draft. The viewer picks one, then Reveal shows the truth, the model's pick and the four scores. The page also shows the session tally (you vs the model), `game_hit_rate` for the chosen model with one sentence on how rounds are drawn, the 25% random rate, and Deal again |
| 5 | `does-it-transfer` | `results_chart` and `p10_chart` with a two-sentence reading, then a prompt to flip the model switch and revisit chapters 2–4 |
| 6 | `honest-limits` | Four points: the C-index bar was never measured; 39 repos is a small sample; the repo attributes are 2026 snapshots; a list of PRs still waiting cannot measure ranking. Links to the README and report |

The Phase 7 header facts carry into chapter 2's text: the replay is not live data, scores are
fixed at open, and only PRs opened from 2026-01-01 are scored. The Phase 7 caption facts carry
into every PR view: risk is a ranking score, not a calibrated probability, and drivers are model
attribution, not cause.

## 8. Docs

- **`README.md`:** the "Try the demo" paragraph describes the six chapters in one sentence.
- **Phase 7 spec:** a note in §6 pointing to this spec for the restructured app.

## 9. Testing

- **Logic:**
  - `wait_buckets`: sums to all PRs, and the two no-review buckets split correctly.
  - `timeline`: its `waiting` flag matches `awaiting_review`.
  - `why`: rows plus base equal `logit(score)` on the committed extract for a sample of PRs
    under both models.
  - `selected_pr_id`: empty and non-empty events.
  - **Game:**
    - every dealt round has 4 distinct PRs from one repo and week with exactly one stalled;
    - `deal` is reproducible with a seeded `rng`;
    - the `model_pick` tie-break;
    - `game_hit_rate` equals brute-force enumeration of every round on small synthetic frames,
      including ties.
  - `results`: its formatted strings equal `writeup_claims`' registered strings for the claims
    they share.
- **Charts:** each builder returns a chart whose spec has the documented encodings, rule,
  selection and colour domain, and whose data equals the input frame.
- **Extract:**
  - the new columns are present;
  - per-row additivity holds for both models (to 1e-4, given float32);
  - each row's `drivers_a` and `drivers_b` text is recomputable from its stored SHAP and feature
    values with the shared formatter;
  - `features.json` matches `DRIVER_LABELS` and the FULL order;
  - no author-identity column.
- **App (AppTest):**
  - every chapter renders without an exception;
  - Next and Back move between chapters;
  - speaker notes are hidden by default and shown when toggled;
  - game: deal, pick, reveal shows the truth and the model's pick, and the tally updates;
  - chapter 5 shows the README's numbers;
  - chapter 3's lookup and Random PR work;
  - the stale-module reload covers every local module.
- **Migrated Phase 7 checks.** Each one is kept, rewritten to reach its chapter: the default repo
  and day; the model switch changing scores; the empty-day message; the tally beside the list's
  stall count; the Seen note; the 2026-only header fact; the ranking-score caption; the lookup's
  not-found message; the requirements pins; no hard-coded result number and no RETRACTED phrasing
  in any `demo/*.py`; imports limited to what `demo/requirements.txt` installs.
- **The existing suite** (462 tests) keeps passing, apart from the app tests migrated as above.

## 10. Out of scope

- Live scoring of new PRs.
- Per-repo precision tables.
- The Phase 6b fingerprint scatter.
- Any new model or retraining.
- Any number that is neither computed from committed data nor equal to the README's registered
  claims.

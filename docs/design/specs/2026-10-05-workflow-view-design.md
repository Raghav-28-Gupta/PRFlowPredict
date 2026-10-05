# Workflow View Design — "How it was built", a visual tour of the pipeline

Written for: whoever implements the workflow view.

**Status:** approved in design review 2026-10-05.
**Depends on:** the demo story (PR #12), merged to `main` at `fbf6931`.
**Produces:** a second navigation section in the deployed Streamlit demo, "How it was built", with
a clickable pipeline map and four deep-dive pages, plus a stage link on each story chapter and two
newly found limitations disclosed in the app and the report.

---

## 1. Purpose and audience

The demo story (six chapters) shows what the model does. Nothing on screen shows how the project
got there: where the data came from, how "slow" was defined, how leakage was prevented, how the
evaluation held data out, or which checks were written down before results were read.

Two audiences use the new section:

- **The owner, presenting live.** After or during the 3–5 minute story, they open the pipeline
  map for 60–90 seconds and drill into one or two stages when asked.
- **A viewer browsing alone** from the README link, who wants to see the method behind the
  numbers.

**Success criterion:** a viewer can follow the pipeline from 44,198 searched repos to the 14,135
test-period PRs in the demo, see why each step is trustworthy, and see what is still weak. Every
number on screen is computed at runtime from committed files, and the six-chapter story works
exactly as before.

## 2. Decisions locked in design review

| Decision | Choice |
|---|---|
| Placement | A separate **"How it was built"** sidebar section; the six story chapters are unchanged |
| Depth | A **pipeline-map hub plus four deep dives**: data funnel, known at time t, two test designs, validity checks |
| New data | **Small derived files** built offline from committed sources, each with a rebuild-and-compare test |
| Undocumented problems | **Disclose both now** (CI bot counted as a reviewer; star-sorted candidate pools); the bot fix is a later, separate phase |
| Build approach | **Second `st.navigation` section** with five file pages, three new modules, its own stage stepper, and one stage link per story chapter |
| Interaction | Presenter-driven clicks: Altair `selection_point` with `on_select`, segmented controls, a select slider. No auto-play; none of the light-only `figures/*.png` |
| Bot disclosure numbers | **Committed numbers only**: the two repos' D5 slow rates from `data/cohort/kept.json`. The 525-label measurement (from gitignored data) is not shown |

## 3. Artifacts

| Path | Change |
|---|---|
| `demo/workflow.py` | **new**: loads committed files, derives every table the section shows. pandas only, no Streamlit |
| `demo/workflow_charts.py` | **new**: Altair charts for the section, reusing `charts.PALETTE` and `charts.NEUTRAL` |
| `demo/stages.py` | **new**: the five page functions, the stage stepper and the section's speaker notes |
| `demo/views/{pipeline,funnel,known_at_t,test_designs,checks}.py` | **new**: two-line page files, as for the story |
| `demo/app.py` | sectioned navigation, cached workflow bundle, reload list, sidebar caption |
| `demo/chapters.py` | `Context` gains `wf` and `wf_pages`; one stage link per chapter; two bullets in *Honest limits* |
| `build_workflow_data.py` | **new**: writes the two derived files below from committed sources |
| `demo/data/phase0_gate.json` | **new**: the 10 Phase 0 pilot checks, transcribed from `docs/phase0_gate_results.md` |
| `demo/data/feature_groups.json` | **new**: the 37 model features with their timing group |
| `writeup_claims.py`, `docs/REPORT.md` | two limitation bullets in REPORT §8, their numbers registered as claims |
| `README.md` | the repo map's Demo row lists the demo's modules (it is missing `charts.py`, `chapters.py`, `views/`) |
| `tests/test_workflow.py`, `tests/test_workflow_charts.py`, `tests/test_build_workflow_data.py` | **new** |
| `tests/test_demo_app.py` | new page tests; guards extended to the new modules |

`demo/requirements.txt` does not change.

## 4. Data

### 4.1 Committed sources the section reads

| Page | Files |
|---|---|
| Pipeline map | `data/cohort/cohort.json`, `data/phase2_rows.json`, `data/phase{2,3,4,6,6b}_gate.json`, `data/phase4_runs.json`, `data/phase6_importance_A.csv`, `data/phase6_worst50.csv`, `data/cohort/kept.json`, `README.md` (6b verdict), `demo/data/phase0_gate.json`, `demo/data/feature_groups.json`, `demo/data/prs.parquet` |
| Data funnel | `data/cohort/raw_search/*.json`, `data/cohort/candidate_pool.json`, `data/cohort/cohort.json`, `data/cohort/kept.json`, `data/phase2_rows.json`, `data/phase4_runs.json` |
| Known at time t | `demo/data/prs.parquet`, `demo/data/feature_groups.json`, `data/phase3_gate.json` |
| Two test designs | `data/phase6b_transfer.csv` (per-repo row counts), `data/phase4_runs.json` |
| Validity checks | `data/phase{2,3,4,6,6b}_gate.json`, `data/phase3_gate5_live.json`, `demo/data/phase0_gate.json` |

Paths resolve from the repo root (`Path(__file__).parent.parent`), so they work locally and on
Streamlit Community Cloud. `docs/REPORT.md` is uppercase, which matters on Linux. Never display
`model_path` or `pred_path` from `phase4_runs.json`: they are absolute local Windows paths.

Counts are verified against these files in design review: repos 44,198 (sum of the nine cells'
`repositoryCount`) → 1,800 pooled → 45 selected → 39 kept; PRs 49,778 in window (45 repos) →
42,646 (39 kept) → 38,462 human-authored rows → 38,444 modelled (sum of Scenario B `n_test`) →
17,338 Scenario A training rows after the cap and 14,135 test PRs.

### 4.2 New derived files

Both are built by `build_workflow_data.py` from committed sources and written with `newline="\n"`.
A test rebuilds each and asserts byte equality with the committed file.

- **`demo/data/phase0_gate.json`**: a list of 10 objects, one per Phase 0 check id 1–10, each with
  `id`, `check` (the check's name as the doc titles it), `outcome` and `detail` (the doc's short
  result text for the in-stratum repo and the adversarial repo). `outcome` is what the doc
  states: `pass` (checks 1, 2, 8, 9, 10), `measured` (3, 5, 6), `resolved` (7, resolved as
  leaky) or `expectation missed, not a stop` (4: litestream's 30-day censoring was above the
  blueprint's guessed range). The board shows check 4 as missed, never as a pass. The file
  carries no "why" prose from the doc (which says "proven" in places) and must pass the
  `writeup_claims.RETRACTED` and `FORBIDDEN_PATTERNS` scans.
- **`demo/data/feature_groups.json`**: a list of 37 objects `{feature, group}` in model order,
  `group` being `features.COLUMN_SPEC[f]["status"]` for each `f` in
  `featuresets.FEATURE_SETS["FULL"]`: `static` (11), `reconstructed` (7, rebuilt to its value at
  open), `replay` (11, replayed from earlier PRs) or `snapshot` (8, 2026 values). The app maps
  these to display labels; the test checks the sizes and that every FULL feature appears once.

### 4.3 Constants mirrored from project code

The deployed app cannot import `splits`, `replay`, `features` or `fingerprint` (they need numpy,
sklearn, lightgbm or scipy), so `workflow.py` copies these values. A test compares each with its
source module:

| Constant | Value | Source |
|---|---|---|
| `CUTOFF_A` | 2026-01-01 UTC | `splits.CUTOFF_A` |
| `CAP_FRAC` | 5% of training rows per repo | `splits.CAP_FRAC` |
| `THRESHOLD_H` | 168 | `replay.History.threshold_h` |
| `TRAILING_DAYS` | 90 | `replay.History.trailing_days` |
| `ALPHA` | 5 | `replay.History.features_at` default `alpha` |
| `VERDICT_NAMES` | the nine 6b verdicts | `fingerprint.VERDICTS.values()` |

## 5. Logic (`demo/workflow.py`)

Pure functions; no Streamlit. The app caches one bundle of their results (`st.cache_data`).

- **`STAGES`**: six stages in order, keys `collect`, `label`, `features`, `evaluate`, `explain`,
  `ship`. Each has a title, a short number-free description, its artifacts as
  `(path, committed, what)`, its gate source, its doc link, its deep-dive page (if any) and the
  story chapter that uses it. `committed` is a static flag; a test checks it against
  `git ls-files`. Phase 5 (survival model) is recorded as not built.
- **`badges(bundle) -> dict[str, str]`**: one headline per stage, computed at runtime: repos
  selected from candidates pooled; labelled PRs; model features; runs; worst errors read; PRs in
  this demo.
- **`gates() -> pd.DataFrame`**: `phase, id, check, value, passed, status` for Phases 2, 3, 4, 6, 6b
  (25 rows) plus the 10 Phase 0 rows. `gate_value_text(value) -> str` formats scalars, lists and
  dicts readably.
- **`repo_funnel()`, `pr_funnel() -> pd.DataFrame`**: one row per step with count, source file and
  reason. `removed_at(step) -> pd.DataFrame` lists what a step removed, with reasons from
  `cohort.json` `rejected[]` and `kept.json` `repos[].reasons`.
- **`pool_bias() -> pd.DataFrame`**: per cell, the cell's size, the pool's size and star range,
  and the pool's share of the cell, from the raw search responses.
- **`label_rates() -> dict`**: global D5 slow rate (Phase 2 gate check 2), global D3 rate
  (row-weighted from `kept.json`), and the D5 rates of `kubernetes/autoscaler` and
  `kubernetes-sigs/gateway-api-inference-extension`.
- **Replay** (known at time t):
  - `prior(prs) -> float`: the shrinkage prior, read from rows with `x__trailing_n == 0`.
  - `replay_at(prs, repo, pr_id, rule) -> (frame, summary)`: the repo's PRs opened before the
    chosen PR, each tagged `counted`, `outcome not yet knowable` or `outside the 90-day window`;
    `rule` is `resolvable` (the project's rule: opened at least 168h before t, or reviewed before
    t) or `naive` (every earlier PR in the window). `summary` gives k, slow count, the shrunk rate
    `(slow + ALPHA·prior) / (k + ALPHA)` and, for `resolvable`, whether it equals the stored
    `x__trailing_90d_slow_rate` and `x__trailing_n`.
  - `exact_repos(prs) -> list[str]`: repos where the recomputation equals the stored feature on
    every PR. Vectorised with pandas (`searchsorted`, `cumsum`), no numpy import.
- **`split_rows() -> pd.DataFrame`**: per repo, pre-2026 rows (`n_B − n_A`), rows kept by the cap,
  2026 test rows, and the Scenario B fold that holds it out. `cap(rows_by_repo) -> pd.Series`
  applies `min(n, max(1, int(CAP_FRAC · total)))`; it reproduces all six stored `n_train` values.
- **`fold_results() -> pd.DataFrame`**: per Scenario B fold, repos held out, `n_train`, `n_test`,
  base rate, and AUC-PR for FULL, NO_LABEL_REPLAY and the baseline.
- **`verdict_6b() -> str`**: the backticked verdict in `README.md`, which must be exactly one
  distinct member of `VERDICT_NAMES` (the claims test already pins README's verdict to its
  recomputation).
- **`top_drivers(n=3) -> pd.DataFrame`**: from `phase6_importance_A.csv`.

## 6. Charts (`demo/workflow_charts.py`)

Pure functions from `workflow.py`'s frames to Altair charts, each taking `mode`:

- `pipeline_map(stages, selected, mode)`: six rect nodes left to right with title, badge and gate
  chip, arrows between them; `selection_point` named `stage` on field `key`.
- `funnel(steps, mode, log)`: horizontal step bars labelled with count and reason;
  `selection_point` named `step`.
- `replay_strip(frame, t, mode)`: one dot per earlier PR on a time axis, coloured by status, with
  rules at t−90d, t−168h and t. `trailing_line(prs_repo, t, prior, mode)`: the stored trailing
  rate as a step line, a cursor at t and a dashed prior line. Both share the time domain.
- `split_bars(rows, scenario, fold, mode, log)` and `fold_dots(folds, fold, mode)` (each fold's
  three AUC-PRs, the chosen fold emphasised, a base-rate floor tick per fold).
- `gate_tiles(gates, mode)`: one tile per check, rows by phase; `selection_point` named `check`.

Colours come from `charts.PALETTE` roles where one fits (blue for counted/pass, `NEUTRAL` for
measured and for text and rules). Any new colour gets light and dark steps and must reach 3:1
contrast on both `#fcfcfb` and `#1a1a19`, checked by a test. Orange keeps its story meaning
(stalled) and is not reused for anything else.

## 7. The app

### 7.1 Navigation and shell

- `st.navigation({"The story": [six story pages], "How it was built": [five workflow pages]})`.
  The story pages keep their titles, URLs and default page.
- Workflow pages: `Pipeline map` (`pipeline`), `Data funnel` (`data-funnel`), `Known at time t`
  (`known-at-time-t`), `Two test designs` (`test-designs`), `Validity checks` (`validity-checks`).
- `Context` gains `wf` (the cached workflow bundle) and `wf_pages`. `pages` stays the six story
  pages, so the story's Back/Next and its tests are unchanged.
- `app.py` reloads `workflow`, `workflow_charts` and `stages` every run, dependencies first.
- The sidebar gains one caption under Model and Repo: they drive chapters 2–4.
- Each workflow page: `st.title` heading, its visual, its speaker note when the toggle is on, and
  a stepper (`← Previous stage` / `Next stage →`, keys per page; the hub has no Previous, Validity
  checks has no Next).

### 7.2 Stage links from the story

Under each chapter's title, one caption-sized line naming its pipeline stage with a button to
the matching page:

| Chapter | Stage | Opens |
|---|---|---|
| 1 The problem | Label | Pipeline map, Label stage |
| 2 Watch it work | Features | Known at time t |
| 3 Why it decides | Explain | Pipeline map, Explain stage |
| 4 Test yourself | Evaluate | Two test designs |
| 5 Does it transfer? | Evaluate | Two test designs |
| 6 Honest limits | all | Validity checks |

The requested hub stage travels in a non-widget session key; the hub copies it into its stage
picker before drawing it. No module globals.

### 7.3 Pipeline map (hub)

The six-stage diagram, and under it a segmented stage picker (default `collect`). A click on a
node or a picker choice opens that stage's panel; a selection naming no known stage is ignored.
The panel shows the stage's description with runtime numbers, its artifacts with *committed* /
*not deployed* chips, its gate (`n/m checks pass`, "pilot gate, 10 checks", or "no gate"), and
links: the deep dive, the story chapter, the doc on GitHub.

Stages without a deep dive get fuller panels:

- **Label**: the D5 definition, the 168h and 30-day rules, global D3 vs D5 slow rates, and the CI
  bot disclosure (§8).
- **Explain**: the top three drivers by mean |SHAP| share, the 6b verdict with one sentence on
  what it means, and links to both interpretation docs.
- **Ship**: README and REPORT numbers are registered and recomputed by tests; the demo reads only
  committed files. No claim counts.

Phase 5 appears in the Evaluate panel as "not built: the survival model and its C-index bar".

### 7.4 Data funnel

Two step funnels: repos (log scale) and PRs (linear). Clicking a step lists what it removed and
why. Captions: training rows are capped at 5% of all training rows per repo while test rows are
not, so 17,338 + 14,135 ≠ 38,444; QC rules are structural (PR volume, bot share, language). The
pool-bias disclosure (§8) sits on this page with `pool_bias()`'s table in an expander.

### 7.5 Known at time t

1. **Feature timing**: the 37 features as chips in their four groups, and the two Phase 3 audits
   read from `phase3_gate.json` (the brute-force replay audit's row count and max difference;
   the live GitHub spot check's rows × features).
2. **The scrubber**: a repo picker limited to `exact_repos` (default `kdlbs/kandev` when present)
   and a select slider over its PRs. The strip and the trailing-rate line update together; a
   readout shows k, the slow count, the formula with its numbers, and "matches the stored
   feature". A toggle switches to the naive rule and shows how many more PRs it would count and
   the rate it would give, labelled as a counterfactual the demo computes, not a project result.
   The default PR is the first whose window holds at least one `outcome not yet knowable` PR.
3. Caption: only the trailing slow rate is re-derived here; backlog and author-history features
   cannot be rebuilt from the extract (bot PRs and author identities are not in it).

### 7.6 Two test designs

A segmented control: **Seen in training (time cut)** / **Unseen repo (repo folds)**, a log-scale
toggle, and KPI metrics (`n_train`, `n_test`, repos).

- **Seen**: one bar per repo split into *trained on*, *dropped by the cap* and *2026 test*.
  `kdlbs/kandev` has no pre-2026 rows.
- **Unseen**: one bar per repo coloured by its fold; a fold picker (1–5) emphasises one fold, and
  `fold_dots` shows that fold's AUC-PR for the model, the model without the four slow-rate
  history features, and the baseline, against the fold's base-rate floor.
- Notes: per-repo counts come from a Phase 6b output file; tuning used Scenario A's training
  rows, which include the held-out repos' pre-2026 rows; the story's Unseen switch shows only
  2026 rows while fold results cover all 30 months; compare scenarios with AUC-PR, not P@10.

### 7.7 Validity checks

Title framing: **validity, not success**. These checks were written before results were read
and test that the pipeline did what it says, not that the model is good.

The tile grid has one row per phase (Phase 0 pilot with 10 checks; Phases 2, 3, 4, 6, 6b with 5
each). Tiles show their status: pass, measured, resolved, or expectation missed (Phase 0 check 4).
Clicking a tile shows the check text, its formatted value and its status, with the raw JSON in an
expander. Two notes attach to specific tiles: Phase 2 check 4 passes exactly
at its threshold; Phase 3 check 5 had one live mismatch, resolved by the REST API (values read
from `phase3_gate5_live.json`). Captions: Phase 1 has no gate file; Phase 5 was not built.

### 7.8 Speaker notes

One hidden note per workflow page, shown with the existing sidebar toggle. The hub's note covers
the 60–90 second walk: data → label → features without peeking → two test designs → explain →
ship, then "ask me to open any stage".

## 8. Disclosures

Both problems were found while mapping the pipeline and confirmed with read-only checks.

1. **A CI bot counts as a reviewer.** Kubernetes' `k8s-ci-robot` is a GitHub `User`, not on
   `labels.KNOWN_BOTS`, so its automated comments count as first human reviews under D5. It is
   the first D5 event on most PRs of `kubernetes/autoscaler` and
   `kubernetes-sigs/gateway-api-inference-extension`, so their slow rates are understated. Shown
   in the hub's Label panel, in *Honest limits*, and in REPORT §8, citing the two repos' D5
   rates from `kept.json`. Fixing it means relabelling and re-running Phases 2–8: a later phase.
2. **Each candidate pool is the top-starred slice of its cell.** GitHub search returned results
   sorted by stars, so the seeded draw chose from each cell's 200 highest-starred repos, not
   across the whole tier. Shown on the funnel page, in *Honest limits*, and in REPORT §8, with
   the pool-share range from `pool_bias()`.

REPORT's numbers for both are registered in `writeup_claims.py` and recomputed from the committed
files above. The demo text names no spam account and no PR author.

## 9. Constraints (carried from the demo)

- Imports: deployed files may import only the existing allowlist plus `workflow`,
  `workflow_charts` and `stages`. No numpy, no project modules.
- No hard-coded result numbers: the `\b0\.\d{3}\b` scan covers every new demo file.
- No `writeup_claims.RETRACTED` phrasing in demo source; strings read from files at runtime are
  limited to the derived files above, the gate check texts and the README verdict.
- Per-viewer state lives in `st.session_state`; chart selections are guarded against stale or
  unknown ids.
- Phase docs, phase generators and gitignored artifacts are not modified; nothing is retrained.

## 10. Testing

- **`tests/test_workflow.py`**: funnel steps equal their sources and chain; `removed_at` reasons
  match the cohort files; `pool_bias` has nine cells with 200-repo pools; gates load 25 + 10
  rows and Phases 2–6b all pass; `gate_value_text` handles scalar, list and dict;
  `exact_repos` includes `kdlbs/kandev`; for every exact repo, `replay_at` matches the stored
  feature on every PR; the naive rule never counts fewer PRs; `cap` reproduces all six stored
  `n_train`; `fold_results` matches `phase4_runs.json`; `verdict_6b` is in `VERDICT_NAMES`;
  mirrored constants equal their sources; `STAGES` committed flags match `git ls-files` (skipped
  outside a git checkout).
- **`tests/test_workflow_charts.py`**: every chart builds; selection names are `stage`, `step`,
  `check`; new colours pass the contrast check in both modes.
- **`tests/test_build_workflow_data.py`**: both derived files rebuild byte-identical; Phase 0 rows
  match the doc, check 4 is `expectation missed, not a stop`, and the file passes the retracted
  and forbidden scans; feature groups cover FULL exactly once with sizes 11/7/11/8.
- **`tests/test_demo_app.py`**: each workflow page renders with its heading; the stepper moves
  and the hub has no Previous, Validity checks no Next; the stage picker and an injected chart
  selection both open a panel, and an unknown selection does not crash; each story chapter's
  stage link opens its target (hub on the right stage); the scrubber's slider and naive toggle
  update the readout; fold and scenario switches render; a tile click shows its check text;
  *Honest limits* shows both disclosures with runtime numbers; speaker notes are hidden by
  default; the six-chapter story tests pass unchanged; the import, reload, number and
  retracted-phrasing guards cover the new modules.
- **`tests/test_writeup_claims.py`**: the new REPORT claims recompute and appear (existing
  parametrised tests pick them up).

## 11. Out of scope

- The Phase 6b decision matrix and fingerprint scatter (still out of scope per the story spec).
- Label definitions D1–D5 as a page, the ablation grid, the worst-50 error map, tuning trials,
  a git timeline and a claims ledger (candidates for a later round).
- Fixing the CI-bot label and re-running Phases 2–8.
- A presenter-only notes view; changes to story chapters beyond the stage link and the two
  *Honest limits* bullets.

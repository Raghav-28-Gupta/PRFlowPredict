# Workflow View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "How it was built" section to the deployed Streamlit demo: a clickable pipeline map and four deep dives (data funnel, known at time t, two test designs, validity checks), a pipeline-stage link on each story chapter, and two newly found limitations disclosed in the app and the report.

**Architecture:** Logic in a new Streamlit-free `demo/workflow.py` (reads committed files, derives every table), Altair charts in `demo/workflow_charts.py`, page functions in `demo/stages.py`, two-line page files in `demo/views/`, a second `st.navigation` section in `demo/app.py`. Two small derived files are built offline by `build_workflow_data.py` from committed sources. The six-chapter story keeps its pages, Back/Next and tests.

**Tech Stack:** Python 3.13, Streamlit 1.65.0 (AppTest, `persist_state`), Altair 6.3.0, pandas 2.3.3, pytest.

**Spec:** `docs/design/specs/2026-10-05-workflow-view-design.md`

## Global Constraints

- Deployed demo files import only: `__future__`, `dataclasses`, `datetime`, `importlib`, `json`, `math`, `pathlib`, `random`, `re`, `sys`, `streamlit`, `pandas`, `altair`, `triage`, `charts`, `chapters`, `workflow`, `workflow_charts`, `stages`. No numpy, no project modules. `demo/requirements.txt` does not change.
- No hard-coded result numbers: no `\b0\.\d{3}\b` anywhere in `demo/*.py` or `demo/views/*.py` (format specs like `:.3f` are fine).
- No `writeup_claims.RETRACTED` phrasing in demo source; README and REPORT also avoid `writeup_claims.FORBIDDEN_PATTERNS`.
- Per-viewer state lives in `st.session_state`, never a module global; chart selections are checked against known ids before use.
- Every colour comes from `charts.PALETTE`; orange keeps its story meaning (stalled) and is not used in the new charts.
- `st.title` is reserved for a page's heading; story chapters gain no `st.caption` above their existing ones (tests read `caption[0]`).
- Page files never use `test_*.py` names (pytest would collect them).
- Never retrain or run experiment.py, tune.py, report6.py, report6b.py, features.py or collection scripts; phase docs, phase generators and gitignored artifacts are not modified.
- Write-up (README, REPORT) never mentions AI tooling; project voice, no first-person authorship.
- Commits end with the trailer `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- On Windows, write multi-line edits with the editor, not shell heredocs (they mangle backslashes). Run tests from the repo root with `python -m pytest`.

## Review Focus

- **The browser re-sends a stale or foreign chart selection** (an unknown stage key, a step from the other funnel, a tile for a check that does not exist): the page ignores it and does not crash. Tests: Task 5 `test_clicking_a_stage_box_opens_its_panel_and_an_unknown_one_is_ignored`, Task 6 `test_a_selection_naming_the_other_funnels_step_is_ignored` and `test_a_tile_selection_for_no_such_check_shows_nothing`.
- **A presenter changes the scrubber's repo after moving its slider:** the PR resets to that repo's default and the readout still matches the stored feature. Test: Task 7 `test_moving_the_scrubber_and_changing_repo_keep_the_readout_exact`.
- **A presenter leaves a workflow page and comes back:** its choices are still there. Test: Task 7 `test_workflow_choices_survive_a_page_switch`.
- **Streamlit Community Cloud keeps old copies of the new modules in memory:** the app reloads them. Test: Task 5 `test_the_workflow_modules_are_reloaded_too`.
- **A link to the story's first page:** Streamlit gives the default page an empty url_path, so chapters are linked by index; every stage panel renders and every deep-dive link resolves. Tests: Task 5 `test_every_stage_panel_renders_with_its_links`, Task 7 `test_each_stage_with_a_deep_dive_links_to_it`.

---

## How to read the steps

Each code step is one of: **Create** `path` (write the whole file), **Append to** `path` (add the block at the end of the file), or **In** `path`, **replace** (the first block must match exactly once; replace it with the second). Code blocks are complete; nothing is elided.

## File map

| File | Task | Responsibility |
|---|---|---|
| `build_workflow_data.py`, `demo/data/phase0_gate.json`, `demo/data/feature_groups.json` | 1 | the two derived files and their builder |
| `demo/workflow.py` | 2, 3 | all logic: stages, gates, funnel, pools, labels, test designs, the time-t replay, the bundle |
| `demo/workflow_charts.py` | 4 | the seven Altair charts |
| `demo/stages.py` | 5, 6, 7 | the five page functions, stepper, speaker notes |
| `demo/views/{pipeline,funnel,checks,known_at_t,designs}.py` | 5, 6, 7 | page files |
| `demo/app.py` | 5, 6, 7 | sectioned navigation, cached bundle, reloads |
| `demo/chapters.py` | 5, 8 | `Context` fields; stage links; *Honest limits* disclosures |
| `writeup_claims.py`, `docs/REPORT.md`, `README.md` | 8 | REPORT §8 limitations and their claims; README demo text and repo map |
| `tests/test_build_workflow_data.py`, `tests/test_workflow.py`, `tests/test_workflow_charts.py`, `tests/test_demo_app.py` | 1–8 | tests |

---

### Task 1: The derived data files

**Files:**
- Create: `build_workflow_data.py`, `tests/test_build_workflow_data.py`
- Create (generated): `demo/data/phase0_gate.json`, `demo/data/feature_groups.json`

**Interfaces:**
- Produces: `demo/data/phase0_gate.json`, a list of 10 `{"id", "check", "outcome", "detail"}` with `outcome` in `pass`, `measured`, `resolved`, `expectation missed, not a stop`; `demo/data/feature_groups.json`, a list of 37 `{"feature", "group"}` in model order, `group` in `static`, `reconstructed`, `replay`, `snapshot`.

- [ ] **Step 1: Write the failing tests**

**Create `tests/test_build_workflow_data.py`:**

```python
"""The workflow view's two derived files (workflow-view spec, section 4.2)."""
import json
import re

import pytest

import build_workflow_data as bwd
import featuresets
import writeup_claims as wc


def _doc() -> str:
    """The Phase 0 doc without Markdown's ` and * marks, whitespace collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"[`*]", "", bwd.PHASE0_DOC.read_text(encoding="utf-8")))


def test_both_files_rebuild_byte_identical():
    assert bwd.PHASE0_OUT.read_bytes() == bwd.dump(bwd.phase0_rows()).encode("utf-8")
    assert bwd.GROUPS_OUT.read_bytes() == bwd.dump(bwd.feature_groups()).encode("utf-8")


def test_phase0_lists_the_ten_checks_with_the_docs_outcomes():
    rows = bwd.phase0_rows()
    assert [r["id"] for r in rows] == list(range(1, 11))
    outcome = {r["id"]: r["outcome"] for r in rows}
    assert {i for i, o in outcome.items() if o == "pass"} == {1, 2, 8, 9, 10}
    assert {i for i, o in outcome.items() if o == "measured"} == {3, 5, 6}
    assert outcome[7] == "resolved"
    assert outcome[4] == "expectation missed, not a stop"


@pytest.mark.parametrize("row", bwd.PHASE0, ids=lambda r: f"check{r[0]}")
def test_each_phase0_check_is_pinned_to_the_doc(row):
    i, check, _, detail, evidence = row
    doc = _doc()
    assert check in doc, f"check {i}: name not in the doc"
    assert evidence in doc, f"check {i}: evidence not in the doc"
    for number in re.findall(r"\d+(?:\.\d+)?", detail):
        assert number in doc, f"check {i}: {number} is not in the Phase 0 doc"


def test_phase0_carries_no_retracted_or_forbidden_phrasing():
    text = bwd.PHASE0_OUT.read_text(encoding="utf-8")
    for pattern, _ in wc.RETRACTED:
        assert re.search(pattern, text, flags=re.IGNORECASE) is None, pattern
    for pattern in wc.FORBIDDEN_PATTERNS:
        assert re.search(pattern, text, flags=re.IGNORECASE) is None, pattern


def test_feature_groups_cover_the_full_set_once_by_when_each_is_known():
    rows = bwd.feature_groups()
    assert [r["feature"] for r in rows] == list(featuresets.FEATURE_SETS["FULL"])
    groups = [r["group"] for r in rows]
    assert {g: groups.count(g) for g in set(groups)} == {"static": 11, "reconstructed": 7,
                                                         "replay": 11, "snapshot": 8}


def test_feature_groups_follow_the_extracts_feature_order():
    extract = json.loads((bwd.ROOT / "demo" / "data" / "features.json").read_text(encoding="utf-8"))
    assert [r["feature"] for r in bwd.feature_groups()] == [f["feature"] for f in extract]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_build_workflow_data.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'build_workflow_data'`

- [ ] **Step 3: Write the builder**

**Create `build_workflow_data.py`:**

```python
"""Build the workflow view's two small derived files from committed sources.

    python build_workflow_data.py

demo/data/phase0_gate.json transcribes the Phase 0 pilot gate, which exists only as prose in
docs/phase0_gate_results.md; tests/test_build_workflow_data.py pins every number and one
verbatim excerpt per check to that document. demo/data/feature_groups.json records when each of
the 37 model features is known, read from features.COLUMN_SPEC. The deployed demo reads these
files because it can neither import the project modules nor parse the prose itself
(workflow-view spec, section 4.2)."""
from __future__ import annotations

import json
from pathlib import Path

import features
import featuresets

ROOT = Path(__file__).parent
PHASE0_DOC = ROOT / "docs" / "phase0_gate_results.md"
PHASE0_OUT = ROOT / "demo" / "data" / "phase0_gate.json"
GROUPS_OUT = ROOT / "demo" / "data" / "feature_groups.json"

MISSED = "expectation missed, not a stop"

# id, check, outcome, detail, evidence. `evidence` is a verbatim excerpt of the doc once its `
# and * marks are removed; `detail` may use only numbers the doc states.
PHASE0 = (
    (1, "Ordering monotonicity", "pass",
     "Review and comment timestamps come back in order on both pilot repos.",
     "| 1 | Ordering monotonicity | PASS | PASS |"),
    (2, "Truncation", "pass",
     "0.23% and 0.31% of PRs had a stream longer than the captured page, under the 1% threshold.",
     "### [2] Truncation — PASS on both"),
    (3, "Label-definition sensitivity", "measured",
     "Five label definitions moved the in-stratum repo's slow rate by 3.7pp and the adversarial "
     "repo's by 8.4pp.",
     "### [3] Label-definition sensitivity"),
    (4, "Censoring / degeneracy", MISSED,
     "47.6% of litestream PRs were never reviewed within 30 days, above the blueprint's guessed "
     "10–35%. Ruled not a stop: the positive rate, 55.4%, is near-balanced.",
     "Litestream fails the blueprint's 10–35% censoring expectation."),
    (5, "Snapshot contamination", "measured",
     "The final diff differs from the diff at open on 47.6% of litestream PRs and 29.4% of the "
     "adversarial repo's, so at-open values are rebuilt from the timeline and commits.",
     "At-open reconstruction from timeline.parquet and commits.parquet is mandatory, not optional."),
    (6, "Warm-up necessity", "measured",
     "33.3% of early-window authors had a PR before the window, so every PR since each repo's "
     "start was collected.",
     "### [6] Warm-up necessity — full-history Tier 1 was required"),
    (7, "authorAssociation mutability", "resolved",
     "35 of 88 litestream authors read CONTRIBUTOR on their first PR: the field is rewritten "
     "later, so author history is rebuilt by replay instead.",
     "### [7] authorAssociation mutability — RESOLVED: LEAKY"),
    (8, "Measured cost at the frozen query shape", "pass",
     "A projected ~5 hours to collect 45 repos, well under the 24-hour line.",
     "well under the 24h line"),
    (9, "Resume idempotency", "pass",
     "A run killed and resumed collected the same 400 PRs as an uninterrupted one: 0 missing, "
     "0 extra.",
     "PASS — 400 = 400, 0 missing, 0 extra"),
    (10, "Raw→parse conservation", "pass",
     "Every collected PR parsed, with 0 checksum failures on either repo.",
     "| 10 | Raw→parse conservation | PASS"),
)


def phase0_rows() -> list[dict]:
    return [{"id": i, "check": check, "outcome": outcome, "detail": detail}
            for i, check, outcome, detail, _ in PHASE0]


def feature_groups() -> list[dict]:
    """The FULL feature set in model order, each with its COLUMN_SPEC status: static,
    reconstructed (rebuilt to its value at open), replay (from earlier PRs) or snapshot."""
    return [{"feature": f, "group": features.COLUMN_SPEC[f]["status"]}
            for f in featuresets.FEATURE_SETS["FULL"]]


def dump(rows: list[dict]) -> str:
    return json.dumps(rows, indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    PHASE0_OUT.write_text(dump(phase0_rows()), encoding="utf-8", newline="\n")
    GROUPS_OUT.write_text(dump(feature_groups()), encoding="utf-8", newline="\n")
    print(f"wrote {PHASE0_OUT.relative_to(ROOT)} and {GROUPS_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Build the files and run the tests**

Run: `python build_workflow_data.py`
Expected: `wrote demo\data\phase0_gate.json and demo\data\feature_groups.json`

Run: `python -m pytest tests/test_build_workflow_data.py -q`
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add build_workflow_data.py tests/test_build_workflow_data.py demo/data/phase0_gate.json demo/data/feature_groups.json
git commit -F - <<'EOF'
feat(demo): derived files for the workflow view -- Phase 0 pilot gate, feature groups

The Phase 0 gate exists only as prose; each transcribed check is pinned to the doc by its
numbers and one verbatim excerpt. Feature groups come from features.COLUMN_SPEC.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

### Task 2: Workflow logic — stages, gates, funnel, labels, test designs

**Files:**
- Create: `demo/workflow.py`, `tests/test_workflow.py`

**Interfaces:**
- Consumes: Task 1's two JSON files.
- Produces (all in `demo/workflow.py`, imported as `workflow` by later tasks): constants `CUTOFF_A`, `CAP_FRAC`, `THRESHOLD_H`, `TRAILING_DAYS`, `ALPHA`, `VERDICT_NAMES`, `TOLERANCE`, `PHASE_NAMES`, `MISSED`, `GROUP_LABELS`, `BOT_REPOS`, `COUNTED`, `UNKNOWN`, `LEAK`, `OLD`, `STALLED_LANE`, `FINE_LANE`, `UNKNOWN_LANE`, `SEGMENTS`, `STAGES` (tuple of `Stage(key, title, phases, summary, artifacts, gates, docs, page, chapter)`), `STAGE_KEYS`; functions `picked(event, param, field=None)`, `stage(key) -> Stage`, `gates() -> DataFrame[phase, id, check, value, passed, status]`, `gate_value_text(value) -> str`, `gate_line(phases, table) -> str`, `live_mismatches() -> list[dict]`, `audits() -> dict`, `repo_funnel()`, `pr_funnel() -> DataFrame[step, count, source, why]`, `removed(step) -> (DataFrame, str)`, `pool_bias() -> DataFrame[group, matched, pooled, share, stars_min, stars_max]`, `label_rates() -> dict`, `bot_note(labels) -> str`, `pool_note(pool) -> str`, `verdict_6b() -> str`, `feature_groups() -> list[dict]`, `badges(prs, groups) -> dict`, `stage_table(badge, table) -> DataFrame`, `cap(rows) -> Series`, `fold_of() -> dict`, `split_rows() -> DataFrame[repo, test_2026, all_rows, pre_2026, trained, capped_out, fold]`, `design_rows(rows, fold=None) -> DataFrame[repo, segment, rows]`, `scenario_a() -> dict`, `fold_results() -> DataFrame[fold, repos, n_train, n_test, base_rate, model, without_history, baseline]`.

- [ ] **Step 1: Write the failing tests**

**Create `tests/test_workflow.py`:**

```python
"""The workflow view's logic (workflow-view spec, sections 4 and 5)."""
import inspect
import json
import subprocess

import pandas as pd
import pytest

import fingerprint
import replay
import splits
import writeup_claims as wc
from demo import triage
from demo import workflow as wf

CLAIMS = {c.name: c.expected for c in wc.CLAIMS}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _runs() -> pd.DataFrame:
    return pd.DataFrame(_json(wf.RUNS)["runs"])


@pytest.fixture(scope="module")
def prs():
    return triage.load()


# ---------------------------------------------------------------------------
# constants, stages, selections
# ---------------------------------------------------------------------------

def test_mirrored_constants_equal_their_sources():
    assert wf.CUTOFF_A == splits.CUTOFF_A
    assert wf.CAP_FRAC == splits.CAP_FRAC
    init = inspect.signature(replay.History.__init__).parameters
    assert wf.THRESHOLD_H == init["threshold_h"].default
    assert wf.TRAILING_DAYS == init["trailing_days"].default
    assert wf.ALPHA == inspect.signature(replay.History.features_at).parameters["alpha"].default
    assert set(wf.VERDICT_NAMES) == set(fingerprint.VERDICTS.values())


def test_stage_artifacts_say_truthfully_whether_they_are_committed():
    """The deployed app cannot run git, so the flags are static; check them against git here."""
    try:
        out = subprocess.run(["git", "ls-files"], cwd=wf.ROOT, capture_output=True, text=True)
    except FileNotFoundError:
        pytest.skip("git is not installed")
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    tracked = out.stdout.splitlines()
    for s in wf.STAGES:
        assert s.chapter is None or 0 <= s.chapter < 6, s.key
        for path, committed, _ in s.artifacts:
            hit = any(t.startswith(path) for t in tracked) if path.endswith("/") else path in tracked
            assert hit == committed, (s.key, path)
        for doc in s.docs:
            assert doc in tracked, (s.key, doc)


def test_stages_run_in_order_and_badges_cover_each(prs):
    assert wf.STAGE_KEYS == ("collect", "label", "features", "evaluate", "explain", "ship")
    badges = wf.badges(prs, wf.feature_groups())
    assert set(badges) == set(wf.STAGE_KEYS)
    assert badges["ship"] == f"{len(prs):,} demo PRs"
    assert badges["features"] == f"{len(wf.feature_groups())} features"
    table = wf.stage_table(badges, wf.gates())
    assert list(table["key"]) == list(wf.STAGE_KEYS) and table["gate"].str.len().gt(0).all()


def test_picked_reads_a_chart_selection_and_tolerates_empties():
    event = {"selection": {"stage": [{"key": "label"}]}}
    assert wf.picked(event, "stage", "key") == "label"
    assert wf.picked(event, "stage") == {"key": "label"}
    for empty in (None, {}, {"selection": {}}, {"selection": {"stage": []}}):
        assert wf.picked(empty, "stage", "key") is None


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def test_gates_hold_ten_pilot_checks_and_five_per_phase_all_passing():
    g = wf.gates()
    assert g.groupby("phase").size().to_dict() == {"0": 10, "2": 5, "3": 5, "4": 5, "6": 5, "6b": 5}
    assert g.loc[g["phase"] != "0", "passed"].all()
    pilot = g[g["phase"] == "0"].set_index("id")["status"]
    assert pilot[4] == wf.MISSED and pilot[7] == "resolved"


def test_gate_value_text_reads_scalars_lists_and_dicts():
    assert wf.gate_value_text(0.4639) == "0.4639"
    assert wf.gate_value_text([38462, 38462]) == "38462, 38462"
    assert wf.gate_value_text({"n": 500, "max_abs_diff": 0.0, "worst": None}) == "n: 500; max_abs_diff: 0; worst: None"
    assert wf.gate_value_text({}) == "none" and wf.gate_value_text(True) == "yes"


def test_gate_lines_count_each_stages_checks():
    table = wf.gates()
    assert wf.gate_line(("0",), table) == "pilot gate: 10 checks"
    assert wf.gate_line(("6", "6b"), table) == "10/10 checks pass"
    assert wf.gate_line((), table) == "claims tests"


def test_audits_and_the_live_mismatch_come_from_the_phase3_files():
    a = wf.audits()
    assert f"{a['audit_n']} rows, largest difference {a['audit_max_diff']}" == CLAIMS["audit_rows"]
    assert a["live_checks"] == a["live_matched"] == a["live_rows"] * 8
    assert [m["pr"] for m in wf.live_mismatches()] == ["radixark/miles#784"]


# ---------------------------------------------------------------------------
# the funnel, the pools, the labels
# ---------------------------------------------------------------------------

def test_the_repo_funnel_reads_each_count_from_its_source():
    f = wf.repo_funnel()["count"].tolist()
    cohort, kept = _json(wf.COHORT), _json(wf.KEPT)
    searched = sum(_json(p)["data"]["search"]["repositoryCount"] for p in wf.SEARCH.glob("search_*_p0.json"))
    assert f == [searched, cohort["n_candidates_pooled"], cohort["n_selected"], len(kept["kept"])]


def test_the_pr_funnel_runs_from_the_window_to_this_demos_prs(prs):
    f = wf.pr_funnel().set_index("step")["count"]
    a = _runs().query("scenario == 'A' and featureset == 'FULL'").iloc[0]
    assert f["human-authored"] == _json(wf.PHASE2_ROWS)["n_rows"]
    assert f["training rows, seen-repo model"] == a["n_train"]
    assert f["test PRs, this demo"] == a["n_test"] == len(prs)
    narrowing = f.iloc[:4].tolist()
    assert narrowing == sorted(narrowing, reverse=True)
    # the cap trims training rows only, so train + test falls short of the modelled rows
    assert f["training rows, seen-repo model"] + f["test PRs, this demo"] < f["modelled"]


def test_every_funnel_step_says_what_it_removed():
    cohort = _json(wf.COHORT)
    for step in list(wf.repo_funnel()["step"]) + list(wf.pr_funnel()["step"]):
        _, note = wf.removed(step)
        assert note, step
    rejected, _ = wf.removed("selected")
    assert len(rejected) == len(cohort["rejected"])
    dropped, _ = wf.removed("kept")
    assert len(dropped) == cohort["n_selected"] - len(_json(wf.KEPT)["kept"])
    assert dropped["reason"].str.len().gt(0).all()
    capped, _ = wf.removed("training rows, seen-repo model")
    assert len(capped) and (capped["capped_out"] > 0).all()
    with pytest.raises(KeyError):
        wf.removed("no such step")


def test_each_pool_is_a_slice_of_its_groups_search_results():
    p = wf.pool_bias()
    cohort = _json(wf.COHORT)
    assert len(p) == len(cohort["languages"]) * len(cohort["star_tiers"])
    assert (p["pooled"] == cohort["n_candidates_pooled"] // len(p)).all()
    assert (p["pooled"] < p["matched"]).all()


def test_each_groups_search_results_came_back_sorted_by_stars():
    """The finding the pool-bias disclosure rests on: the pool took each group's first results,
    and those are its most-starred repos."""
    for group in wf.pool_bias()["group"]:
        lang, tier = group.split(":")
        lo, hi = tier.split("-")
        pages = sorted(wf.SEARCH.glob(f"search_{lang}_{lo}_{hi}_p*.json"))
        stars = [n["stargazerCount"] for p in pages for n in _json(p)["data"]["search"]["nodes"] if n]
        assert stars == sorted(stars, reverse=True), group


def test_label_rates_match_the_registered_claims():
    r = wf.label_rates()
    assert f"{r['d5']:.1%} slow under D5" == CLAIMS["d5_rate"]
    assert f"{r['d3']:.1%} under D3" == CLAIMS["d3_rate"]
    assert set(r["bot_repos"]) == set(wf.BOT_REPOS)


def test_the_disclosures_cite_runtime_numbers():
    rates = wf.label_rates()["bot_repos"]
    a, b = (rates[r] for r in wf.BOT_REPOS)
    assert f"({a:.1%} and {b:.1%})" in wf.bot_note(wf.label_rates())
    pool = wf.pool_bias()
    note = wf.pool_note(pool)
    assert f"between {pool['share'].min():.1%} and {pool['share'].max():.1%} of the group" in note
    assert f"the first {pool['pooled'].iloc[0]} of each" in note


def test_the_6b_verdict_is_read_from_the_readme():
    assert wf.verdict_6b() == CLAIMS["verdict_6b"]


def test_a_readme_naming_two_verdicts_is_refused(tmp_path, monkeypatch):
    readme = tmp_path / "README.md"
    readme.write_text("`PARTIAL_SHAP_ONLY` and `SUPPORTED`", encoding="utf-8")
    monkeypatch.setattr(wf, "README", readme)
    with pytest.raises(LookupError):
        wf.verdict_6b()


# ---------------------------------------------------------------------------
# the two test designs
# ---------------------------------------------------------------------------

def test_the_cap_reproduces_every_stored_training_and_test_count():
    rows = wf.split_rows()
    full = _runs().query("featureset == 'FULL'")
    a = full.query("scenario == 'A'").iloc[0]
    seen = wf.design_rows(rows).groupby("segment")["rows"].sum()
    assert (seen["trained on"], seen["tested on"]) == (a["n_train"], a["n_test"])
    assert rows["trained"].sum() == a["n_train"]
    for _, run in full.query("scenario == 'B'").iterrows():
        fold = wf.design_rows(rows, run["fold"]).groupby("segment")["rows"].sum()
        assert (fold["trained on"], fold["tested on"]) == (run["n_train"], run["n_test"]), run["fold"]


def test_design_rows_split_each_repo_without_losing_a_row():
    rows = wf.split_rows()
    for fold in (None, 0):
        long = wf.design_rows(rows, fold).groupby("repo")["rows"].sum()
        expected = rows.set_index("repo")["all_rows"]
        assert long.reindex(expected.index, fill_value=0).equals(expected)


def test_split_rows_cover_every_kept_repo_once_with_its_fold(prs):
    rows = wf.split_rows()
    assert rows["repo"].is_unique and set(rows["repo"]) == set(_json(wf.KEPT)["kept"])
    assert rows["fold"].notna().all()
    assert rows["test_2026"].sum() == len(prs)
    assert rows.set_index("repo").loc["kdlbs/kandev", "pre_2026"] == 0


def test_fold_results_match_the_runs():
    f = wf.fold_results()
    b = _runs().query("scenario == 'B'")
    assert f["fold"].tolist() == sorted(b["fold"].unique().tolist())
    for _, r in f.iterrows():
        nlr = b.query("featureset == 'NO_LABEL_REPLAY' and fold == @r.fold").iloc[0]
        assert r["without_history"] == nlr["auc_pr"] and r["baseline"] == nlr["baseline_auc_pr"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_workflow.py -q`
Expected: collection error, `ImportError: cannot import name 'workflow' from 'demo'`

- [ ] **Step 3: Write the module**

**Create `demo/workflow.py`:**

```python
"""The workflow view's logic: how the project was built, read from committed files.

Kept free of Streamlit so it can be tested directly. Every number the "How it was built" pages
show is computed here at runtime from committed files (workflow-view spec, section 4): the
cohort files, the phase gates and results under data/, the Phase 0 and feature-group files under
demo/data/ (built by build_workflow_data.py), and the demo's extract. The deployed app cannot
import the project's own modules, so the few constants it needs are copied below, and a test
checks each against its source. Imports pandas and the standard library only."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
COHORT = ROOT / "data" / "cohort" / "cohort.json"
KEPT = ROOT / "data" / "cohort" / "kept.json"
POOL = ROOT / "data" / "cohort" / "candidate_pool.json"
SEARCH = ROOT / "data" / "cohort" / "raw_search"
PHASE2_ROWS = ROOT / "data" / "phase2_rows.json"
GATE = str(ROOT / "data" / "phase{}_gate.json")
LIVE = ROOT / "data" / "phase3_gate5_live.json"
RUNS = ROOT / "data" / "phase4_runs.json"
TRANSFER = ROOT / "data" / "phase6b_transfer.csv"
WORST50 = ROOT / "data" / "phase6_worst50.csv"
README = ROOT / "README.md"
PHASE0 = HERE / "data" / "phase0_gate.json"
GROUPS = HERE / "data" / "feature_groups.json"

# Copied from the project's modules, which the deployed app cannot import. A test compares each.
CUTOFF_A = pd.Timestamp("2026-01-01T00:00:00Z")        # splits.CUTOFF_A
CAP_FRAC = 0.05                                        # splits.CAP_FRAC
THRESHOLD_H = 168.0                                    # replay.History.threshold_h
TRAILING_DAYS = 90                                     # replay.History.trailing_days
ALPHA = 5.0                                            # replay.History.features_at's alpha
VERDICT_NAMES = ("SUPPORTED", "PARTIAL_SHAP_ONLY", "PARTIAL_INTERVENTION_ONLY", "NOT_SUPPORTED",
                 "CONFLICTING", "CONTRADICTED")       # fingerprint.VERDICTS' values
TOLERANCE = 1e-9

GATED = ("2", "3", "4", "6", "6b")
PHASE_NAMES = {"0": "Phase 0 (pilot)", "2": "Phase 2", "3": "Phase 3", "4": "Phase 4",
               "6": "Phase 6", "6b": "Phase 6b"}
MISSED = "expectation missed, not a stop"
GROUP_LABELS = {"static": "Static: fixed when the PR opens",
                "reconstructed": "Rebuilt to its value at open",
                "replay": "Replayed from earlier PRs",
                "snapshot": "2026 snapshot of the repo"}
BOT_REPOS = ("kubernetes/autoscaler", "kubernetes-sigs/gateway-api-inference-extension")
COUNTED, UNKNOWN, LEAK, OLD = ("counted", "outcome not yet knowable", "counted, though not yet knowable",
                               "outside the 90-day window")
STALLED_LANE, FINE_LANE, UNKNOWN_LANE = "stalled", "reviewed within 7 days", "not yet known at t"


def _json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def picked(event, param: str, field: str | None = None):
    """The first point a chart selection `param` holds (a dict), or its `field`; None if empty."""
    try:
        points = event["selection"][param]
    except (KeyError, TypeError):
        return None
    if not points:
        return None
    return points[0] if field is None else points[0].get(field)


# ---------------------------------------------------------------------------
# the six stages
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    phases: str
    summary: str                              # number-free; the page adds runtime numbers
    artifacts: tuple[tuple[str, bool, str], ...]   # (repo path, committed?, what it holds)
    gates: tuple[str, ...]                    # gate phases; "0" is the Phase 0 pilot
    docs: tuple[str, ...]                     # repo-relative documents
    page: str | None                          # url_path of the deep-dive page
    chapter: int | None                       # index of the story chapter that uses it (the first
                                              # page's url_path is empty, so not looked up by path)


STAGES: tuple[Stage, ...] = (
    Stage("collect", "Collect", "Phases 0–1",
          "Draw public GitHub repos in nine language × star-tier groups with a seeded rule that "
          "uses only structural criteria, then collect every PR's history. A two-repo pilot gate "
          "first decided which GitHub fields are safe at open time.",
          (("data/cohort/cohort.json", True, "the seeded draw: repos picked and rejected, with reasons"),
           ("data/cohort/candidate_pool.json", True, "every candidate the searches returned"),
           ("data/cohort/raw_search/", True, "the search responses, verbatim"),
           ("data/raw/", False, "the collected GitHub responses, gzipped"),
           ("data/processed/", False, "parsed PR, review, comment and timeline tables")),
          ("0",), ("docs/data_collection.md", "docs/phase0_gate_results.md"), "data-funnel", None),
    Stage("label", "Label", "Phase 2",
          "Define a PR's first review (D5): the first review or comment by a human other than the "
          "author, with a tie to the repo. A PR is slow without one within 7 days. Repos are "
          "dropped on structural rules only, and the trailing-rate baseline is fixed.",
          (("data/cohort/kept.json", True, "per-repo QC: kept or dropped, with D3 and D5 slow rates"),
           ("data/phase2_gate.json", True, "the Phase 2 checks"),
           ("docs/phase2_eda.md", True, "label-definition and threshold sensitivity")),
          ("2",), ("docs/phase2_eda.md",), None, 0),
    Stage("features", "Features", "Phase 3",
          "Turn each PR into model features using only what was knowable when it opened: fields "
          "that change later are rebuilt to their values at open, and history features replay "
          "the repo in time order.",
          (("docs/feature_dictionary.md", True, "every column: group, status and derivation"),
           ("data/phase3_gate.json", True, "the Phase 3 checks, including the replay audit"),
           ("demo/data/prs.parquet", True, "the test-period PRs with feature values, scores and SHAP"),
           ("data/features/", False, "the feature table, one row per PR")),
          ("3",), ("docs/feature_dictionary.md",), "known-at-time-t", 1),
    Stage("evaluate", "Train & test", "Phase 4",
          "Hold data out two ways: future PRs of repos seen in training, and whole repos never "
          "seen. One LightGBM configuration was tuned once and frozen, then trained per test "
          "design and feature set, and scored against the baseline.",
          (("data/phase4_runs.json", True, "every run's metrics, intervals and repo lists"),
           ("data/phase4_params.json", True, "the tuned parameters and every tuning trial"),
           ("data/experiments.csv", True, "the run ledger"),
           ("data/models/", False, "the trained boosters"),
           ("data/predictions/", False, "per-PR test predictions")),
          ("4",), ("docs/phase4_results.md",), "test-designs", 4),
    Stage("explain", "Explain", "Phases 6 and 6b",
          "Explain the frozen models with exact TreeSHAP, read the most confident errors, check "
          "newcomers are not treated worse, and test one explanation for the cold-start gap under "
          "a decision rule written down before any code ran.",
          (("data/phase6_importance_A.csv", True, "each feature's share of the attribution"),
           ("data/phase6_worst50.csv", True, "the most confident errors"),
           ("data/phase6_fairness.csv", True, "first-time vs repeat contributors"),
           ("data/phase6b_transfer.csv", True, "the pre-registered transfer test, per repo")),
          ("6", "6b"), ("docs/phase6_interpretation.md", "docs/phase6b_fingerprinting.md"), None, 2),
    Stage("ship", "Ship", "Phases 7–8",
          "Write it up and build this demo. The results the README and the report cite are "
          "registered and recomputed from committed files by a test, and the demo reads only "
          "committed files.",
          (("README.md", True, "the summary"),
           ("docs/REPORT.md", True, "the full report"),
           ("writeup_claims.py", True, "every cited number, recomputed by a test"),
           ("demo/app.py", True, "this demo")),
          (), ("docs/REPORT.md",), None, 5),
)
STAGE_KEYS = tuple(s.key for s in STAGES)


def stage(key: str) -> Stage:
    return next(s for s in STAGES if s.key == key)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def gates() -> pd.DataFrame:
    """Every recorded check: Phase 0's ten pilot checks, then five for each gated phase."""
    rows = [{"phase": "0", "id": r["id"], "check": r["check"], "value": r["detail"],
             "passed": r["outcome"] == "pass", "status": r["outcome"]} for r in _json(PHASE0)]
    for ph in GATED:
        rows += [{"phase": ph, "id": c["id"], "check": c["check"], "value": c["value"],
                  "passed": bool(c["pass"]), "status": "pass" if c["pass"] else "fail"}
                 for c in _json(Path(GATE.format(ph)))]
    return pd.DataFrame(rows)


def gate_value_text(value) -> str:
    """A recorded check value, readable: dicts as key: value, lists joined, empties as none."""
    if isinstance(value, dict):
        return "; ".join(f"{k}: {gate_value_text(v)}" for k, v in value.items()) or "none"
    if isinstance(value, list):
        return ", ".join(gate_value_text(v) for v in value) or "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def gate_line(phases: tuple[str, ...], table: pd.DataFrame) -> str:
    if not phases:
        return "claims tests"
    if phases == ("0",):
        return f"pilot gate: {int((table['phase'] == '0').sum())} checks"
    rows = table[table["phase"].isin(phases)]
    return f"{int(rows['passed'].sum())}/{len(rows)} checks pass"


def live_mismatches() -> list[dict]:
    """The live GitHub spot check's disagreements: repo#number, field, live and ours."""
    return [{"pr": f"{r['repo']}#{r['number']}", "field": f, "live": v["live"], "ours": v["ours"]}
            for r in _json(LIVE) if not r["match"]
            for f, v in r["checks"].items() if v["live"] != v["ours"]]


def audits() -> dict:
    """Phase 3's two audits: the brute-force replay audit and the live GitHub spot check."""
    g3 = {c["id"]: c["value"] for c in _json(Path(GATE.format("3")))}
    return {"audit_n": g3[2]["n"], "audit_max_diff": g3[2]["max_abs_diff"],
            "live_rows": len(g3[5]["rows"]), "live_checks": g3[5]["checks"],
            "live_matched": g3[5]["matched"]}


# ---------------------------------------------------------------------------
# the funnel and the cohort
# ---------------------------------------------------------------------------

def _cells() -> dict[str, int]:
    """Each search group (e.g. Go:200-800) and how many repos its search matched."""
    out = {}
    for f in sorted(SEARCH.glob("search_*_p0.json")):
        _, lang, lo, hi, _ = f.stem.split("_")
        out[f"{lang}:{lo}-{hi}"] = _json(f)["data"]["search"]["repositoryCount"]
    return out


def repo_funnel() -> pd.DataFrame:
    cohort, kept = _json(COHORT), _json(KEPT)
    return pd.DataFrame([
        {"step": "matched the searches", "count": sum(_cells().values()), "source": "data/cohort/raw_search/",
         "why": f"public, non-fork, non-archived repos pushed since {cohort['pushed_since']}, in "
                f"{len(cohort['languages'])} languages × {len(cohort['star_tiers'])} star tiers"},
        {"step": "pooled as candidates", "count": cohort["n_candidates_pooled"],
         "source": "data/cohort/candidate_pool.json",
         "why": "the first results of each group's search, which came back sorted by stars"},
        {"step": "selected", "count": cohort["n_selected"], "source": "data/cohort/cohort.json",
         "why": f"a seeded random draw of {cohort['per_cell']} per group, skipping repos with fewer "
                f"than {cohort['min_window_prs']} PRs"},
        {"step": "kept", "count": len(kept["kept"]), "source": "data/cohort/kept.json",
         "why": "structural QC only: bot share, human PR count and language"},
    ])


def pr_funnel() -> pd.DataFrame:
    reps = pd.DataFrame(_json(KEPT)["repos"])
    kept = reps[reps["kept"]]
    runs = pd.DataFrame(_json(RUNS)["runs"])
    full = runs[runs["featureset"] == "FULL"]
    a = full[full["scenario"] == "A"].iloc[0]
    start, end = _json(COHORT)["window"]
    return pd.DataFrame([
        {"step": "in the window, selected repos", "count": int(reps["n_in_window"].sum()),
         "source": "data/cohort/kept.json", "why": f"PRs opened from {start} to {end}"},
        {"step": "in the kept repos", "count": int(kept["n_in_window"].sum()),
         "source": "data/cohort/kept.json", "why": "PRs of the repos QC dropped removed"},
        {"step": "human-authored", "count": int(kept["n_human"].sum()), "source": "data/phase2_rows.json",
         "why": "bot-authored PRs removed; every remaining PR gets a label and features"},
        {"step": "modelled", "count": int(full.loc[full["scenario"] == "B", "n_test"].sum()),
         "source": "data/phase4_runs.json",
         "why": "PRs whose timeline may be truncated removed; each is tested once across the unseen-repo folds"},
        {"step": "training rows, seen-repo model", "count": int(a["n_train"]), "source": "data/phase4_runs.json",
         "why": f"PRs opened before {CUTOFF_A:%Y-%m-%d}, capped at {CAP_FRAC:.0%} of training rows per repo"},
        {"step": "test PRs, this demo", "count": int(a["n_test"]), "source": "data/phase4_runs.json",
         "why": f"PRs opened from {CUTOFF_A:%Y-%m-%d} to {end}, never capped"},
    ])


def removed(step: str) -> tuple[pd.DataFrame, str]:
    """What a funnel step removed, and why: a table where one exists, and a sentence."""
    cohort, reps = _json(COHORT), pd.DataFrame(_json(KEPT)["repos"])
    dropped = reps[~reps["kept"]]
    none = pd.DataFrame()
    if step == "matched the searches":
        return none, "The first step: what GitHub's nine searches matched."
    if step == "pooled as candidates":
        per = cohort["n_candidates_pooled"] // len(_cells())
        return none, (f"Only the first {per} results of each group's search were fetched. They came "
                      "back sorted by stars, so the pool is each group's most-starred repos.")
    if step == "selected":
        frame = pd.DataFrame(cohort["rejected"])[["repo", "cell", "reason"]]
        return frame, (f"The draw visited each group's pool in seeded random order and skipped "
                       f"{len(frame)} repos with too few PRs until it had {cohort['per_cell']}; "
                       "the rest of the pool was never visited.")
    if step == "kept":
        frame = dropped.assign(reason=dropped["reasons"].map("; ".join))[["repo", "cell", "reason"]]
        return frame, f"{len(frame)} repos failed the structural QC rules."
    if step == "in the window, selected repos":
        return none, "The first step: every PR the selected repos opened in the window."
    if step == "in the kept repos":
        return dropped[["repo", "n_in_window"]], "The window PRs of the repos QC dropped."
    if step == "human-authored":
        kept = reps[reps["kept"]]
        frame = kept.assign(bot_authored=kept["n_in_window"] - kept["n_human"])[["repo", "bot_authored"]]
        return frame[frame["bot_authored"] > 0], "Bot-authored PRs in the kept repos."
    if step == "modelled":
        return none, "Rows flagged as possibly truncated timelines were left out of every model."
    if step == "training rows, seen-repo model":
        rows = split_rows()
        frame = rows[rows["capped_out"] > 0][["repo", "pre_2026", "trained", "capped_out"]]
        return frame, (f"Each repo contributes at most {CAP_FRAC:.0%} of the training rows, so a few "
                       "very active repos cannot dominate the model.")
    if step == "test PRs, this demo":
        return none, "Every PR opened from January to June 2026 in the kept repos is a test PR."
    raise KeyError(step)


def pool_bias() -> pd.DataFrame:
    """Per search group: how many repos matched, how many were pooled, and their star range."""
    pool = pd.DataFrame(_json(POOL))
    out = (pool.groupby("_cell")["stargazerCount"].agg(pooled="size", stars_min="min", stars_max="max")
           .reset_index().rename(columns={"_cell": "group"}))
    out["matched"] = out["group"].map(_cells())
    out["share"] = out["pooled"] / out["matched"]
    return out[["group", "matched", "pooled", "share", "stars_min", "stars_max"]]


def label_rates() -> dict:
    """Global D5 and D3 slow rates over the kept repos, and the two CI-bot repos' D5 rates."""
    reps = pd.DataFrame(_json(KEPT)["repos"])
    kept = reps[reps["kept"]]
    d5 = next(c["value"] for c in _json(Path(GATE.format("2"))) if c["check"].startswith("D5 global"))
    d3 = float((kept["is_slow_d3"] * kept["n_human"]).sum() / kept["n_human"].sum())
    bots = reps.set_index("repo").loc[list(BOT_REPOS), "is_slow_d5"]
    return {"d5": float(d5), "d3": d3, "bot_repos": {r: float(v) for r, v in bots.items()}}


def bot_note(labels: dict) -> str:
    """The CI-bot disclosure (workflow-view spec, section 8), with the two repos' D5 rates."""
    a, b = (labels["bot_repos"][r] for r in BOT_REPOS)
    return ("**A CI bot counts as a reviewer.** Kubernetes' `k8s-ci-robot` account is registered on "
            "GitHub as an ordinary user, not a bot, so the label counts its automated comments as first "
            f"reviews. It supplies most first reviews in {BOT_REPOS[0]} and {BOT_REPOS[1]}, so their "
            f"slow rates under D5 ({a:.1%} and {b:.1%}) understate how long people took. Fixing it means "
            "relabelling and re-running every later phase.")


def pool_note(pool: pd.DataFrame) -> str:
    """The pool-bias disclosure (workflow-view spec, section 8), with the pools' share of their groups."""
    return ("**Each group's candidates are its most-starred repos.** The searches returned each group's "
            f"matches sorted by stars, and the pool took the first {int(pool['pooled'].iloc[0])} of each, "
            f"between {pool['share'].min():.1%} and {pool['share'].max():.1%} of the group, so the seeded "
            "draw chose among each group's most-starred repos, not across its whole star range.")


def verdict_6b() -> str:
    """The Phase 6b verdict as the README states it (the claims test pins it to its recomputation)."""
    found = set(re.findall(r"`([A-Z_]+)`", README.read_text(encoding="utf-8"))) & set(VERDICT_NAMES)
    if len(found) != 1:
        raise LookupError(f"README.md names {len(found)} Phase 6b verdicts")
    return found.pop()


def feature_groups() -> list[dict]:
    return _json(GROUPS)


def badges(prs: pd.DataFrame, groups: list[dict]) -> dict[str, str]:
    """One headline per stage, computed from committed files."""
    cohort = _json(COHORT)
    return {"collect": f"{cohort['n_selected']} of {cohort['n_candidates_pooled']:,} repos",
            "label": f"{_json(PHASE2_ROWS)['n_rows']:,} labelled PRs",
            "features": f"{len(groups)} features",
            "evaluate": f"{len(_json(RUNS)['runs'])} runs",
            "explain": f"{len(pd.read_csv(WORST50))} errors read",
            "ship": f"{len(prs):,} demo PRs"}


def stage_table(badge: dict[str, str], table: pd.DataFrame) -> pd.DataFrame:
    """The pipeline map's nodes, in order."""
    return pd.DataFrame([{"key": s.key, "order": i, "title": f"{i + 1}. {s.title}", "phases": s.phases,
                          "badge": badge[s.key], "gate": gate_line(s.gates, table)}
                         for i, s in enumerate(STAGES)])


# ---------------------------------------------------------------------------
# the two test designs
# ---------------------------------------------------------------------------

def _runs() -> pd.DataFrame:
    return pd.DataFrame(_json(RUNS)["runs"])


def cap(rows: pd.Series) -> pd.Series:
    """Training rows each repo keeps: at most CAP_FRAC of all training rows (splits.cap_per_repo)."""
    return rows.clip(upper=max(1, int(CAP_FRAC * rows.sum())))


def fold_of() -> dict[str, int]:
    """The Scenario B fold that holds out each repo."""
    b = _runs()
    b = b[(b["scenario"] == "B") & (b["featureset"] == "FULL")]
    return {repo: int(r["fold"]) for _, r in b.iterrows() for repo in r["test_repos"]}


def split_rows() -> pd.DataFrame:
    """Per repo: 2026 test PRs, pre-2026 PRs, how many of those the cap kept for training, and the
    fold that holds the repo out. Counts come from Phase 6b's per-repo table: n_A is a repo's
    Scenario A test rows (2026) and n_B all its rows (Scenario B tests the whole repo)."""
    t = pd.read_csv(TRANSFER)[["repo", "n_A", "n_B"]].rename(columns={"n_A": "test_2026", "n_B": "all_rows"})
    t["pre_2026"] = t["all_rows"] - t["test_2026"]
    t["trained"] = cap(t["pre_2026"])
    t["capped_out"] = t["pre_2026"] - t["trained"]
    t["fold"] = t["repo"].map(fold_of())
    return t.sort_values("all_rows", ascending=False).reset_index(drop=True)


SEGMENTS = ("trained on", "dropped by the cap", "tested on")


def design_rows(rows: pd.DataFrame, fold: int | None = None) -> pd.DataFrame:
    """Each repo's rows as trained on / dropped by the cap / tested on (long format). With no
    fold, Scenario A: pre-2026 rows train (capped) and 2026 rows test. With a fold, Scenario B:
    the fold's repos are tested whole and every row of the others trains, capped."""
    if fold is None:
        trained, pool, tested = rows["trained"], rows["pre_2026"], rows["test_2026"]
    else:
        held = rows["fold"] == fold
        pool = rows["all_rows"].where(~held, 0)
        trained = cap(rows.loc[~held, "all_rows"]).reindex(rows.index, fill_value=0)
        tested = rows["all_rows"].where(held, 0)
    parts = zip(SEGMENTS, (trained, pool - trained, tested))
    out = pd.concat([pd.DataFrame({"repo": rows["repo"], "segment": seg, "rows": vals.astype(int)})
                     for seg, vals in parts], ignore_index=True)
    return out[out["rows"] > 0].reset_index(drop=True)


def scenario_a() -> dict:
    a = _runs()
    a = a[(a["scenario"] == "A") & (a["featureset"] == "FULL")].iloc[0]
    return {"n_train": int(a["n_train"]), "n_test": int(a["n_test"]),
            "train_repos": len(a["train_repos"]), "test_repos": len(a["test_repos"])}


def fold_results() -> pd.DataFrame:
    """Per Scenario B fold: repos held out, rows, base rate and the three AUC-PRs."""
    b = _runs()
    b = b[b["scenario"] == "B"]
    full = b[b["featureset"] == "FULL"].set_index("fold").sort_index()
    nlr = b[b["featureset"] == "NO_LABEL_REPLAY"].set_index("fold").sort_index()
    return pd.DataFrame({"fold": list(full.index), "repos": list(full["n_test_repos"]),
                         "n_train": list(full["n_train"]), "n_test": list(full["n_test"]),
                         "base_rate": list(full["base_rate"]), "model": list(full["auc_pr"]),
                         "without_history": list(nlr["auc_pr"]),
                         "baseline": list(full["baseline_auc_pr"])})
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_workflow.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add demo/workflow.py tests/test_workflow.py
git commit -F - <<'EOF'
feat(demo): workflow logic -- stages, gates, funnel, labels, test designs

Every number is read from committed files at runtime; constants the app cannot import are
copied and tested against their sources. The cap reproduces all six stored training counts.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

### Task 3: Workflow logic — known at time t, and the bundle

**Files:**
- Modify: `demo/workflow.py` (append), `tests/test_workflow.py` (append)

**Interfaces:**
- Consumes: Task 2's module.
- Produces: `prior(prs) -> float`, `shrunk(slow, k, prior_rate) -> float`, `replay_at(prs, repo, pr_id, rule="resolvable", prior_rate=None) -> (DataFrame[pr_id, number, created_at, first_review_at, is_slow, status, lane], dict[t, number, k, slow, prior, rate, unknown, stored_k, stored_rate, matches])`, `trailing_check(prs, prior_rate=None) -> DataFrame[..., k, slow, unknown, rate, matches]`, `exact_repos(check) -> list[str]`, `default_pr(check, repo) -> str`, `bundle(prs) -> dict` with keys `stages, gates, groups, repo_funnel, pr_funnel, pool_bias, labels, audits, live, prior, check, exact, splits, folds, scenario_a, verdict`.

- [ ] **Step 1: Write the failing tests**

**Append to `tests/test_workflow.py`:**

```python


# ---------------------------------------------------------------------------
# known at time t
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def check(prs):
    return wf.trailing_check(prs)


def _toy() -> pd.DataFrame:
    """One repo; P0 opens at t. Hand-computed with the prior 0.5 and alpha 5:
    P1 opened 10 days before, reviewed       -> counted (old enough), not slow
    P2 opened 2 days before, reviewed 1 later -> counted (reviewed before t), not slow
    P3 opened 1 day before, no review yet     -> not yet knowable; it later stalls
    P4 opened 100 days before                 -> outside the 90-day window
    resolvable: k=2, slow=0 -> (0 + 2.5) / 7;  naive: k=3, slow=1 -> (1 + 2.5) / 8.
    The stored values of the others: P4 has an empty window (0.5); P1 counts P4, exactly 90 days
    earlier; P2 counts P1; P3 counts P1 only, as P2's review lands exactly at P3's opening."""
    t = pd.Timestamp("2026-05-01T00:00:00Z")
    d = pd.Timedelta(days=1)
    return pd.DataFrame({
        "repo": ["r"] * 5, "pr_id": ["P4", "P1", "P2", "P3", "P0"], "number": [4, 1, 2, 3, 0],
        "created_at": [t - 100 * d, t - 10 * d, t - 2 * d, t - d, t],
        "first_review_at": [t - 99 * d, t - 9 * d, t - d, pd.NaT, pd.NaT],
        "is_slow": [False, False, False, True, True],
        "x__trailing_n": [0, 1, 1, 1, 2],
        "x__trailing_90d_slow_rate": [0.5, 2.5 / 6, 2.5 / 6, 2.5 / 6, 2.5 / 7],
    })


def test_a_pr_counts_once_it_is_a_week_old_or_reviewed():
    frame, s = wf.replay_at(_toy(), "r", "P0", prior_rate=0.5)
    assert (s["k"], s["slow"], s["unknown"]) == (2, 0, 1)
    assert s["rate"] == pytest.approx(2.5 / 7) and s["matches"]
    status = frame.set_index("pr_id")["status"]
    assert status.to_dict() == {"P4": wf.OLD, "P1": wf.COUNTED, "P2": wf.COUNTED, "P3": wf.UNKNOWN}
    assert frame.set_index("pr_id").loc["P3", "lane"] == wf.UNKNOWN_LANE


def test_the_naive_rule_counts_what_was_not_yet_knowable():
    frame, s = wf.replay_at(_toy(), "r", "P0", rule="naive", prior_rate=0.5)
    assert (s["k"], s["slow"]) == (3, 1) and s["rate"] == pytest.approx(3.5 / 8) and not s["matches"]
    leaked = frame.set_index("pr_id").loc["P3"]
    assert leaked["status"] == wf.LEAK and leaked["lane"] == wf.STALLED_LANE


def test_the_vectorised_check_agrees_on_the_toy():
    out = wf.trailing_check(_toy(), prior_rate=0.5).set_index("pr_id")
    assert (out.loc["P0", "k"], out.loc["P0", "unknown"]) == (2, 1)
    assert out["matches"].all()


def test_the_prior_is_the_rate_of_an_empty_window(prs):
    g = wf.prior(prs)
    assert (prs.loc[prs["x__trailing_n"] == 0, "x__trailing_90d_slow_rate"] == g).all()


def test_exact_repos_match_on_every_pr_and_include_the_default(check):
    exact = wf.exact_repos(check)
    assert "kdlbs/kandev" in exact
    assert check[check["repo"].isin(exact)]["matches"].all()
    assert not check[~check["repo"].isin(exact)].groupby("repo")["matches"].all().any()


def test_replay_at_matches_the_stored_feature_on_every_pr_of_an_exact_repo(prs, check):
    exact = wf.exact_repos(check)
    repo = min(exact, key=lambda r: int((prs["repo"] == r).sum()))
    g = wf.prior(prs)
    for pid in prs.loc[prs["repo"] == repo, "pr_id"]:
        assert wf.replay_at(prs, repo, pid, prior_rate=g)[1]["matches"], pid


def test_replay_at_agrees_with_the_vectorised_check_across_repos(prs, check):
    g = wf.prior(prs)
    for _, r in check.sample(150, random_state=0).iterrows():
        _, s = wf.replay_at(prs, r["repo"], r["pr_id"], prior_rate=g)
        assert (s["k"], s["unknown"]) == (r["k"], r["unknown"]), r["pr_id"]
        assert s["rate"] == pytest.approx(r["rate"], abs=1e-12)


def test_the_default_pr_has_the_most_unknowable_neighbours(check):
    pid = wf.default_pr(check, "kdlbs/kandev")
    rows = check[check["repo"] == "kdlbs/kandev"].set_index("pr_id")
    assert rows.loc[pid, "unknown"] == rows["unknown"].max() > 0


def test_the_bundle_holds_everything_the_pages_read(prs):
    b = wf.bundle(prs)
    assert set(b) == {"stages", "gates", "groups", "repo_funnel", "pr_funnel", "pool_bias", "labels",
                      "audits", "live", "prior", "check", "exact", "splits", "folds", "scenario_a", "verdict"}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_workflow.py -q`
Expected: the new tests fail with `AttributeError: module 'demo.workflow' has no attribute 'replay_at'` (or `trailing_check`, `prior`, `bundle`)

- [ ] **Step 3: Implement**

**Append to `demo/workflow.py`:**

```python


# ---------------------------------------------------------------------------
# known at time t: the trailing slow rate, re-derived from the extract
# ---------------------------------------------------------------------------

def prior(prs: pd.DataFrame) -> float:
    """The shrinkage prior, read off the PRs whose 90-day window held no knowable outcome."""
    empty = prs.loc[prs["x__trailing_n"] == 0, "x__trailing_90d_slow_rate"]
    if empty.empty:
        raise LookupError("no PR with an empty trailing window to read the prior from")
    return float(empty.iloc[0])


def shrunk(slow: int, k: int, prior_rate: float) -> float:
    return (slow + ALPHA * prior_rate) / (k + ALPHA)


def replay_at(prs: pd.DataFrame, repo: str, pr_id: str, rule: str = "resolvable",
              prior_rate: float | None = None) -> tuple[pd.DataFrame, dict]:
    """The repo's PRs opened in the 120 days before `pr_id`, each with its status at that moment,
    and the trailing slow rate they give. `rule` is "resolvable" (the project's: a PR counts once
    it is 168h old or reviewed) or "naive" (every earlier PR in the 90-day window)."""
    rows = prs[prs["repo"] == repo].sort_values(["created_at", "pr_id"], kind="mergesort")
    me = rows[rows["pr_id"] == pr_id].iloc[0]
    t = me["created_at"]
    before = rows[(rows["created_at"] < t) & (rows["created_at"] >= t - pd.Timedelta(days=TRAILING_DAYS + 30))]
    window = before["created_at"] >= t - pd.Timedelta(days=TRAILING_DAYS)
    knowable = (before["created_at"] <= t - pd.Timedelta(hours=THRESHOLD_H)) | (before["first_review_at"] < t)
    counted = window & (knowable | (rule == "naive"))
    status = pd.Series(OLD, index=before.index)
    status[window & ~knowable] = LEAK if rule == "naive" else UNKNOWN
    status[window & knowable] = COUNTED
    lane = before["is_slow"].map({True: STALLED_LANE, False: FINE_LANE})
    lane[status == UNKNOWN] = UNKNOWN_LANE
    g = prior(prs) if prior_rate is None else prior_rate
    k, slow = int(counted.sum()), int(before.loc[counted, "is_slow"].sum())
    rate = shrunk(slow, k, g)
    summary = {"t": t, "number": int(me["number"]), "k": k, "slow": slow, "prior": g, "rate": rate,
               "unknown": int((window & ~knowable).sum()), "stored_k": int(me["x__trailing_n"]),
               "stored_rate": float(me["x__trailing_90d_slow_rate"])}
    summary["matches"] = k == summary["stored_k"] and abs(rate - summary["stored_rate"]) <= TOLERANCE
    frame = before[["pr_id", "number", "created_at", "first_review_at", "is_slow"]].assign(status=status, lane=lane)
    return frame.reset_index(drop=True), summary


def trailing_check(prs: pd.DataFrame, prior_rate: float | None = None) -> pd.DataFrame:
    """Every PR's trailing slow rate re-derived from the extract, beside the stored feature.
    Per repo: PRs at least 168h old come from sorted positions and running sums; only the last
    week's PRs are checked one by one for a review before t."""
    g = prior(prs) if prior_rate is None else prior_rate
    parts = []
    for _, rows in prs.groupby("repo", sort=True):
        rows = rows.sort_values(["created_at", "pr_id"], kind="mergesort")
        c = rows["created_at"].reset_index(drop=True)
        lo = c.searchsorted(c - pd.Timedelta(days=TRAILING_DAYS), side="left")
        thr = c.searchsorted(c - pd.Timedelta(hours=THRESHOLD_H), side="right")
        now = c.searchsorted(c, side="left")
        slow = rows["is_slow"].astype(int).tolist()
        cum = [0]
        for s in slow:
            cum.append(cum[-1] + s)
        reviewed, opened = rows["first_review_at"].tolist(), c.tolist()
        ks, slows, unknown = [], [], []
        for i, t in enumerate(opened):
            recent = [j for j in range(thr[i], now[i]) if reviewed[j] < t]   # NaT compares False
            ks.append(int(thr[i] - lo[i]) + len(recent))
            slows.append(cum[thr[i]] - cum[lo[i]] + sum(slow[j] for j in recent))
            unknown.append(int(now[i] - thr[i]) - len(recent))
        part = rows[["repo", "pr_id", "number", "created_at", "x__trailing_n", "x__trailing_90d_slow_rate"]].copy()
        part["k"], part["slow"], part["unknown"] = ks, slows, unknown
        part["rate"] = [shrunk(s, k, g) for s, k in zip(slows, ks)]
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["matches"] = ((out["k"] == out["x__trailing_n"])
                      & ((out["rate"] - out["x__trailing_90d_slow_rate"]).abs() <= TOLERANCE))
    return out


def exact_repos(check: pd.DataFrame) -> list[str]:
    """Repos where the re-derived rate equals the stored feature on every PR."""
    ok = check.groupby("repo")["matches"].all()
    return sorted(ok[ok].index)


def default_pr(check: pd.DataFrame, repo: str) -> str:
    """The repo's PR with the most earlier PRs whose outcome was not yet knowable (the earliest
    on a tie): the moment the 7-day rule matters most."""
    rows = check[check["repo"] == repo].sort_values(["unknown", "created_at"], ascending=[False, True],
                                                     kind="mergesort")
    return rows["pr_id"].iloc[0]


# ---------------------------------------------------------------------------
# everything the section shows, built once
# ---------------------------------------------------------------------------

def bundle(prs: pd.DataFrame) -> dict:
    groups, table = feature_groups(), gates()
    check = trailing_check(prs)
    return {"stages": stage_table(badges(prs, groups), table), "gates": table, "groups": groups,
            "repo_funnel": repo_funnel(), "pr_funnel": pr_funnel(), "pool_bias": pool_bias(),
            "labels": label_rates(), "audits": audits(), "live": live_mismatches(),
            "prior": prior(prs), "check": check, "exact": exact_repos(check),
            "splits": split_rows(), "folds": fold_results(), "scenario_a": scenario_a(),
            "verdict": verdict_6b()}
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_workflow.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add demo/workflow.py tests/test_workflow.py
git commit -F - <<'EOF'
feat(demo): re-derive the trailing slow rate at time t from the extract

A PR's outcome counts at t once it is 168h old or reviewed. The re-derivation matches the stored
feature exactly on every PR of five repos; only those are offered by the scrubber.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

### Task 4: The charts

**Files:**
- Create: `demo/workflow_charts.py`, `tests/test_workflow_charts.py`

**Interfaces:**
- Consumes: `workflow` (Tasks 2–3), `charts.PALETTE`, `triage.SERIES_LABELS`.
- Produces: `pipeline_map(stages, selected, mode)` (selection `stage` on `key`), `funnel(steps, unit, mode, log=False)` (selection `step`), `replay_strip(frame, t, mode, domain=None)`, `trailing_line(rows, t, prior_rate, mode, domain=None)`, `split_bars(long, order, mode)`, `fold_dots(folds, fold, mode)`, `gate_tiles(gates, mode)` (selection `check` on `phase`, `id`).

- [ ] **Step 1: Write the failing tests**

**Create `tests/test_workflow_charts.py`:**

```python
"""The workflow view's charts, read through their Vega-Lite specs (workflow-view spec, section 6).

workflow_charts imports its siblings by bare name, as the deployed app does, so demo/ goes on the
path first."""
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

DEMO = Path(__file__).parents[1] / "demo"
sys.path.insert(0, str(DEMO))
import charts  # noqa: E402
import triage  # noqa: E402
import workflow as wf  # noqa: E402
import workflow_charts as wc  # noqa: E402


@pytest.fixture(scope="module")
def prs():
    return triage.load()


@pytest.fixture(scope="module")
def bundle(prs):
    return wf.bundle(prs)


def _params(chart) -> dict:
    return {p["name"]: p for p in chart.to_dict().get("params", [])}


def _layers(chart) -> list[dict]:
    return chart.to_dict()["layer"]


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_every_chart_builds_a_valid_spec_in_both_themes(bundle, prs, mode):
    pid = wf.default_pr(bundle["check"], "kdlbs/kandev")
    frame, s = wf.replay_at(prs, "kdlbs/kandev", pid, prior_rate=bundle["prior"])
    rows = prs[prs["repo"] == "kdlbs/kandev"]
    order = list(bundle["splits"]["repo"])
    for chart in (wc.pipeline_map(bundle["stages"], "collect", mode),
                  wc.funnel(bundle["repo_funnel"], "repos", mode, log=True),
                  wc.funnel(bundle["pr_funnel"], "pull requests", mode),
                  wc.replay_strip(frame, s["t"], mode),
                  wc.trailing_line(rows, s["t"], s["prior"], mode),
                  wc.split_bars(wf.design_rows(bundle["splits"]), order, mode),
                  wc.fold_dots(bundle["folds"], 0, mode),
                  wc.gate_tiles(bundle["gates"], mode)):
        assert chart.to_dict()                      # validates against the Vega-Lite schema


def test_the_pipeline_map_selects_a_stage_by_key_and_marks_the_chosen_one(bundle):
    chart = wc.pipeline_map(bundle["stages"], "evaluate")
    assert _params(chart)["stage"]["select"]["fields"] == ["key"]
    boxes = next(layer for layer in _layers(chart) if layer["mark"]["type"] == "rect")
    data = chart.data if isinstance(chart.data, pd.DataFrame) else chart.layer[0].data
    assert data.loc[data["chosen"], "key"].tolist() == ["evaluate"]
    assert boxes["encoding"]["fill"]["condition"]["value"] == charts.PALETTE["light"]["full"]


def test_funnels_select_a_step_and_start_log_bars_at_one(bundle):
    log = wc.funnel(bundle["repo_funnel"], "repos", log=True)
    assert _params(log)["step"]["select"]["fields"] == ["step"]
    bars = next(layer for layer in _layers(log) if layer["mark"]["type"] == "bar")
    assert bars["encoding"]["x"]["scale"]["type"] == "log" and bars["encoding"]["x2"]["datum"] == 1
    linear = next(layer for layer in _layers(wc.funnel(bundle["pr_funnel"], "pull requests"))
                  if layer["mark"]["type"] == "bar")
    assert linear["encoding"]["x2"]["datum"] == 0


def test_the_strip_shows_the_leak_only_under_the_naive_rule(bundle, prs):
    pid = wf.default_pr(bundle["check"], "kdlbs/kandev")
    for rule, has_leak in (("resolvable", False), ("naive", True)):
        frame, s = wf.replay_at(prs, "kdlbs/kandev", pid, rule, bundle["prior"])
        dots = next(layer for layer in _layers(wc.replay_strip(frame, s["t"])) if layer["mark"]["type"] == "circle")
        domain = dots["encoding"]["color"]["scale"]["domain"]
        assert (wf.LEAK in domain) is has_leak and (wf.UNKNOWN in domain) is not has_leak
        assert dots["encoding"]["y"]["sort"] == list(wc.LANES)


def test_the_trailing_line_dashes_the_prior(prs, bundle):
    rows = prs[prs["repo"] == "kdlbs/kandev"]
    t = rows["created_at"].iloc[100]
    chart = wc.trailing_line(rows, t, bundle["prior"])
    rules = [layer for layer in chart.layer if layer.to_dict()["mark"].get("strokeDash")]
    assert any(isinstance(r.data, pd.DataFrame) and r.data["y"].tolist() == [bundle["prior"]] for r in rules)


def test_split_bars_colour_the_three_segments_in_order(bundle):
    chart = wc.split_bars(wf.design_rows(bundle["splits"]), list(bundle["splits"]["repo"]))
    colour = chart.to_dict()["encoding"]["color"]["scale"]
    pal = charts.PALETTE["light"]
    assert colour["domain"] == list(wf.SEGMENTS) and colour["range"] == [pal["full"], pal["random"], pal["nlr"]]


def test_fold_dots_use_the_storys_series_names_and_emphasise_one_fold(bundle):
    chart = wc.fold_dots(bundle["folds"], 2)
    dots = next(layer for layer in chart.layer if layer.to_dict()["mark"]["type"] == "circle")
    assert set(dots.data["series"]) == {triage.SERIES_LABELS[k] for k in ("FULL", "NO_LABEL_REPLAY", "baseline")}
    assert dots.data.loc[dots.data["chosen"], "name"].unique().tolist() == ["fold 3"]


def test_gate_tiles_select_a_check_and_mark_the_missed_expectation(bundle):
    chart = wc.gate_tiles(bundle["gates"])
    assert _params(chart)["check"]["select"]["fields"] == ["phase", "id"]
    tiles = next(layer for layer in _layers(chart) if layer["mark"]["type"] == "rect")
    scale = tiles["encoding"]["color"]["scale"]
    assert dict(zip(scale["domain"], scale["range"]))[wf.MISSED] == charts.PALETTE["light"]["up"]
    data = chart.data if isinstance(chart.data, pd.DataFrame) else chart.layer[0].data
    assert "value" not in data.columns and len(data) == len(bundle["gates"])


def test_the_charts_use_only_the_validated_palette():
    """Every colour comes from charts.PALETTE, whose pairs were validated in both themes; any
    other colour literal would have to clear 3:1 on both chart surfaces."""
    known = {c for theme in charts.PALETTE.values() for c in theme.values()}
    literals = set(re.findall(r"#[0-9a-fA-F]{6}", (DEMO / "workflow_charts.py").read_text(encoding="utf-8")))
    assert literals <= known, literals - known
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_workflow_charts.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'workflow_charts'`

- [ ] **Step 3: Write the module**

**Create `demo/workflow_charts.py`:**

```python
"""The workflow view's charts: pure functions from workflow.py's frames to Altair charts.

Nothing here touches Streamlit, so tests read each chart's spec with .to_dict(). Colours are the
story's palette roles (charts.PALETTE), so both themes stay validated: blue for what the model
counts, trains on or passes; light blue for what it is tested on; grey for context; red for the
leak the 7-day rule prevents and for the one pilot check that missed its expectation. Orange
keeps its story meaning (stalled) and is not used here."""
from __future__ import annotations

import altair as alt
import pandas as pd

import triage
import workflow
from charts import PALETTE

STATUS_ORDER = (workflow.COUNTED, workflow.UNKNOWN, workflow.LEAK, workflow.OLD)
LANES = (workflow.STALLED_LANE, workflow.FINE_LANE, workflow.UNKNOWN_LANE)
TILE_TEXT = {"pass": "pass", "measured": "measured", "resolved": "resolved", workflow.MISSED: "missed",
             "fail": "fail"}
SERIES = {"model": triage.SERIES_LABELS["FULL"], "without_history": triage.SERIES_LABELS["NO_LABEL_REPLAY"],
          "baseline": triage.SERIES_LABELS["baseline"]}


def pipeline_map(stages: pd.DataFrame, selected: str, mode: str = "light") -> alt.LayerChart:
    """The six stages left to right, the chosen one emphasised; clicking a box selects its key
    ('stage')."""
    pal = PALETTE[mode]
    n = len(stages)
    data = stages.assign(x0=stages["order"] + 0.05, x1=stages["order"] + 0.95, mid=stages["order"] + 0.5,
                         chosen=stages["key"] == selected)
    pick = alt.selection_point(name="stage", fields=["key"], on="click", empty=False)
    x = alt.X("x0:Q", scale=alt.Scale(domain=[0, n]), axis=None)
    y = alt.Y("y0:Q", scale=alt.Scale(domain=[0, 1]), axis=None)
    boxes = alt.Chart(data.assign(y0=0.04, y1=0.96)).mark_rect(cornerRadius=8, strokeWidth=2).encode(
        x=x, x2="x1:Q", y=y, y2="y1:Q",
        fill=alt.condition(alt.datum.chosen, alt.value(pal["full"]), alt.value(pal["muted"])),
        fillOpacity=alt.condition(alt.datum.chosen, alt.value(0.22), alt.value(0.06)),
        stroke=alt.condition(alt.datum.chosen, alt.value(pal["full"]), alt.value(pal["muted"])),
        tooltip=[alt.Tooltip("title:N", title="stage"), alt.Tooltip("phases:N"), alt.Tooltip("badge:N"),
                 alt.Tooltip("gate:N")],
    ).add_params(pick)

    def text(field: str, at: float, size: int, bold: bool = False) -> alt.Chart:
        return alt.Chart(data.assign(y0=at)).mark_text(
            fontSize=size, fontWeight="bold" if bold else "normal", color=pal["ink"], limit=150).encode(
            x=alt.X("mid:Q", scale=alt.Scale(domain=[0, n]), axis=None), y=y, text=f"{field}:N")

    gaps = pd.DataFrame({"x0": [i + 0.95 for i in range(n - 1)], "x1": [i + 1.05 for i in range(n - 1)],
                         "y0": [0.5] * (n - 1)})
    arrows = alt.Chart(gaps).mark_rule(color=pal["muted"], strokeWidth=2).encode(x=x, x2="x1:Q", y=y)
    heads = alt.Chart(gaps).mark_point(shape="triangle-right", filled=True, size=60, color=pal["muted"]).encode(
        x=alt.X("x1:Q", scale=alt.Scale(domain=[0, n]), axis=None), y=y)
    return (boxes + text("title", 0.76, 14, True) + text("badge", 0.5, 12) + text("gate", 0.24, 11)
            + arrows + heads).properties(height=130)


def funnel(steps: pd.DataFrame, unit: str, mode: str = "light", log: bool = False) -> alt.LayerChart:
    """One narrowing step per bar, labelled with its count; clicking a bar selects it ('step')."""
    pal = PALETTE[mode]
    order = list(steps["step"])
    pick = alt.selection_point(name="step", fields=["step"], on="click", empty=False)
    base = alt.Chart(steps.assign(label=[f"{c:,}" for c in steps["count"]])).encode(
        y=alt.Y("step:N", sort=order, title=None, axis=alt.Axis(labelLimit=240)),
        x=alt.X("count:Q", title=f"{unit} (log scale)" if log else unit,
                scale=alt.Scale(type="log") if log else alt.Scale(), axis=alt.Axis(format="~s")),
        tooltip=[alt.Tooltip("step:N"), alt.Tooltip("count:Q", format=","), alt.Tooltip("why:N"),
                 alt.Tooltip("source:N")])
    bars = base.mark_bar(cornerRadiusEnd=4, height=22, color=pal["full"]).encode(
        x2=alt.datum(1 if log else 0),          # a log scale has no zero: bars start at 1
        strokeWidth=alt.condition(pick, alt.value(3), alt.value(0)), stroke=alt.value(pal["ring"])).add_params(pick)
    labels = base.mark_text(align="left", dx=5, color=pal["ink"]).encode(text="label:N")
    return (bars + labels).properties(height=alt.Step(34))


def replay_strip(frame: pd.DataFrame, t: pd.Timestamp, mode: str = "light",
                 domain: tuple | None = None) -> alt.LayerChart:
    """The repo's earlier PRs on a time axis, one lane per outcome as known at t, coloured by
    whether the trailing rate counts them. Rules mark t - 90 days, t - 7 days and t."""
    pal = PALETTE[mode]
    colours = dict(zip(STATUS_ORDER, (pal["full"], pal["muted"], pal["up"], pal["random"])))
    present = [s for s in STATUS_ORDER if s in set(frame["status"])]
    scale = alt.Scale(domain=list(domain)) if domain else alt.Scale()
    dots = alt.Chart(frame).transform_calculate(jitter="random()").mark_circle(size=34, opacity=0.85).encode(
        x=alt.X("created_at:T", title="opened", scale=scale),
        y=alt.Y("lane:N", sort=list(LANES), title=None, axis=alt.Axis(labelLimit=200)),
        yOffset=alt.YOffset("jitter:Q", scale=alt.Scale(domain=[-0.5, 1.5])),
        color=alt.Color("status:N", scale=alt.Scale(domain=present, range=[colours[s] for s in present]),
                        legend=alt.Legend(title=None, orient="top", labelLimit=320)),
        tooltip=[alt.Tooltip("number:Q", title="PR #"), alt.Tooltip("created_at:T", title="opened", format="%d %b %Y %H:%M"),
                 alt.Tooltip("first_review_at:T", title="first review", format="%d %b %Y %H:%M"),
                 alt.Tooltip("status:N")])
    marks = pd.DataFrame({"at": [t - pd.Timedelta(days=workflow.TRAILING_DAYS),
                                 t - pd.Timedelta(hours=workflow.THRESHOLD_H), t],
                          "label": ["90 days before", "7 days before", "t: this PR opens"]})
    rules = alt.Chart(marks).mark_rule(color=pal["muted"], strokeDash=[4, 4]).encode(x=alt.X("at:T", scale=scale))

    def label(rows: pd.DataFrame, y: int) -> alt.Chart:
        return alt.Chart(rows).mark_text(color=pal["muted"], align="right", dx=-4, y=y).encode(
            x=alt.X("at:T", scale=scale), text="label:N")

    # t's label sits a line lower: "7 days before" ends only a week to its left
    return (dots + rules + label(marks.iloc[:2], 6) + label(marks.iloc[2:], 20)).properties(height=alt.Step(70))


def trailing_line(rows: pd.DataFrame, t: pd.Timestamp, prior_rate: float, mode: str = "light",
                  domain: tuple | None = None) -> alt.LayerChart:
    """The stored trailing slow rate of each of the repo's PRs at its opening, with t marked and
    the prior it shrinks toward dashed."""
    pal = PALETTE[mode]
    scale = alt.Scale(domain=list(domain)) if domain else alt.Scale()
    y = alt.Y("x__trailing_90d_slow_rate:Q", title="trailing 90-day slow rate", scale=alt.Scale(domain=[0, 1]))
    line = alt.Chart(rows).mark_line(interpolate="step-after", color=pal["full"]).encode(
        x=alt.X("created_at:T", title="opened", scale=scale), y=y)
    now = rows[rows["created_at"] == t]
    point = alt.Chart(now).mark_circle(size=120, color=pal["full"], stroke=pal["ring"], strokeWidth=1.5).encode(
        x=alt.X("created_at:T", scale=scale), y=y,
        tooltip=[alt.Tooltip("x__trailing_90d_slow_rate:Q", title="stored rate at t", format=".3f")])
    prior = pd.DataFrame({"y": [prior_rate], "label": ["prior"]})
    prior_rule = alt.Chart(prior).mark_rule(color=pal["muted"], strokeDash=[6, 4]).encode(y="y:Q")
    prior_label = alt.Chart(prior).mark_text(color=pal["muted"], align="left", x=4, dy=-6).encode(y="y:Q", text="label:N")
    cursor = alt.Chart(pd.DataFrame({"at": [t]})).mark_rule(color=pal["muted"]).encode(x=alt.X("at:T", scale=scale))
    return (line + prior_rule + prior_label + cursor + point).properties(height=200)


def split_bars(long: pd.DataFrame, order: list[str], mode: str = "light") -> alt.Chart:
    """Each repo's rows, stacked: trained on, dropped by the cap, tested on."""
    pal = PALETTE[mode]
    return alt.Chart(long).mark_bar(height=10).encode(
        y=alt.Y("repo:N", sort=order, title=None, axis=alt.Axis(labelLimit=260, labelFontSize=10)),
        x=alt.X("sum(rows):Q", title="pull requests", stack="zero"),
        color=alt.Color("segment:N", scale=alt.Scale(domain=list(workflow.SEGMENTS),
                                                     range=[pal["full"], pal["random"], pal["nlr"]]),
                        legend=alt.Legend(title=None, orient="top")),
        order=alt.Order("order:Q"),
        tooltip=[alt.Tooltip("repo:N"), alt.Tooltip("segment:N"), alt.Tooltip("rows:Q", format=",")],
    ).transform_calculate(order=f"indexof({list(workflow.SEGMENTS)}, datum.segment)").properties(height=alt.Step(13))


def fold_dots(folds: pd.DataFrame, fold: int, mode: str = "light") -> alt.LayerChart:
    """Each held-out fold's AUC-PR for the model, the model without the slow-rate history and the
    baseline, with the fold's base rate (the floor AUC-PR starts from) as a tick; the chosen fold
    stands out."""
    pal = PALETTE[mode]
    data = folds.assign(name=[f"fold {f + 1}" for f in folds["fold"]], chosen=folds["fold"] == fold)
    long = data.melt(id_vars=["name", "chosen", "base_rate"], value_vars=list(SERIES),
                     var_name="series", value_name="auc_pr")
    long["series"] = long["series"].map(SERIES)
    order = list(data["name"])
    y = alt.Y("name:N", sort=order, title=None)
    x = alt.X("auc_pr:Q", title="AUC-PR", scale=alt.Scale(domain=[0, 1]))
    floor = alt.Chart(data).mark_tick(color=pal["ink"], thickness=2, size=18).encode(
        x=alt.X("base_rate:Q", scale=alt.Scale(domain=[0, 1])), y=y,
        opacity=alt.condition(alt.datum.chosen, alt.value(1), alt.value(0.3)),
        tooltip=[alt.Tooltip("name:N", title="fold"), alt.Tooltip("base_rate:Q", title="base rate (the floor)", format=".3f")])
    dots = alt.Chart(long).mark_circle(size=110, stroke=pal["ring"], strokeWidth=1).encode(
        x=x, y=y,
        color=alt.Color("series:N", scale=alt.Scale(domain=list(SERIES.values()),
                                                    range=[pal["full"], pal["nlr"], pal["baseline"]]),
                        legend=alt.Legend(title=None, orient="top", labelLimit=320)),
        opacity=alt.condition(alt.datum.chosen, alt.value(1), alt.value(0.25)),
        tooltip=[alt.Tooltip("name:N", title="fold"), alt.Tooltip("series:N"), alt.Tooltip("auc_pr:Q", format=".3f")])
    return (floor + dots).properties(height=alt.Step(36))


def gate_tiles(gates: pd.DataFrame, mode: str = "light") -> alt.LayerChart:
    """One tile per recorded check, a row per phase; clicking a tile selects it ('check')."""
    pal = PALETTE[mode]
    colours = {"pass": pal["full"], "measured": pal["muted"], "resolved": pal["muted"],
               workflow.MISSED: pal["up"], "fail": pal["up"]}
    data = gates.drop(columns="value").assign(
        phase_name=gates["phase"].map(workflow.PHASE_NAMES), tile=gates["status"].map(TILE_TEXT),
        value_text=gates["value"].map(workflow.gate_value_text))
    present = [s for s in colours if s in set(data["status"])]
    pick = alt.selection_point(name="check", fields=["phase", "id"], on="click", empty=False)
    base = alt.Chart(data).encode(
        x=alt.X("id:O", title="check", axis=alt.Axis(labelAngle=0), scale=alt.Scale(paddingInner=0.08)),
        y=alt.Y("phase_name:N", sort=list(workflow.PHASE_NAMES.values()), title=None,
                scale=alt.Scale(paddingInner=0.12)))
    tiles = base.mark_rect(cornerRadius=4, stroke=pal["ring"]).encode(
        color=alt.Color("status:N", scale=alt.Scale(domain=present, range=[colours[s] for s in present]),
                        legend=alt.Legend(title=None, orient="top", labelLimit=320)),
        strokeWidth=alt.condition(pick, alt.value(3), alt.value(0)),
        tooltip=[alt.Tooltip("phase_name:N", title="phase"), alt.Tooltip("id:O", title="check"),
                 alt.Tooltip("check:N", title="what it checks"), alt.Tooltip("status:N"),
                 alt.Tooltip("value_text:N", title="recorded")],
    ).add_params(pick)
    text = base.mark_text(color=pal["dot"], fontSize=11).encode(text="tile:N")
    return (tiles + text).properties(height=alt.Step(40))
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_workflow_charts.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add demo/workflow_charts.py tests/test_workflow_charts.py
git commit -F - <<'EOF'
feat(demo): workflow charts -- pipeline map, funnels, time-t strip, splits, folds, checks

Pure Altair functions; every colour comes from the validated story palette.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

### Task 5: The section shell and the pipeline map

**Files:**
- Create: `demo/stages.py`, `demo/views/pipeline.py`
- Modify: `demo/app.py`, `demo/chapters.py`, `tests/test_demo_app.py`

**Interfaces:**
- Consumes: `workflow.bundle`, `workflow_charts.pipeline_map`, `chapters._c`, `chapters._chart`, `chapters._data`, `chapters.GITHUB`, `triage.importance`.
- Produces: `chapters.Context.wf` (the bundle) and `chapters.Context.wf_pages` (the section's `st.Page`s); `stages.hub()`; helpers `stages._note(key)`, `stages._page(url_path)` (None if the app does not register it), `stages._stepper(url_path)` (buttons `wf_back_{i}` / `wf_next_{i}`, `i` the page's position); session keys `wf_stage` (the stage picker, persisted) and `wf_focus` (a stage requested from elsewhere); chart key `pipeline_map`.

- [ ] **Step 1: Write the failing tests**

**In `tests/test_demo_app.py`, replace:**

```python
import writeup_claims as wc
from demo import triage
```

**with:**

```python
import writeup_claims as wc
from demo import triage
from demo import workflow
```

**In `tests/test_demo_app.py`, replace:**

```python
    allowed = {"__future__", "dataclasses", "datetime", "importlib", "json", "math", "pathlib",
               "random", "re", "sys", "streamlit", "pandas", "altair", "triage", "charts", "chapters"}
```

**with:**

```python
    allowed = {"__future__", "dataclasses", "datetime", "importlib", "json", "math", "pathlib",
               "random", "re", "sys", "streamlit", "pandas", "altair", "triage", "charts", "chapters",
               "workflow", "workflow_charts", "stages"}
```

**Append to `tests/test_demo_app.py`:**

```python


# ---------------------------------------------------------------------------
# "How it was built" (workflow-view spec, section 7)
# ---------------------------------------------------------------------------

WF_VIEWS = ["pipeline", "funnel", "known_at_t", "designs", "checks"]
WF_HEADINGS = ["How it was built", "Data funnel", "Known at time t", "Two test designs", "Validity checks"]


def _chart_id(at: AppTest, key: str) -> str:
    """The element id of the Altair chart drawn with `key`."""
    def walk(node):
        yield node
        children = getattr(node, "children", None)
        if isinstance(children, dict):
            for child in children.values():
                yield from walk(child)
    return next(n.proto.id for n in walk(at._tree)
                if type(getattr(n, "proto", None)).__name__ == "VegaLiteChart" and n.proto.id.endswith(f"-{key}"))


def _run_with_selection(at: AppTest, key: str, selection: dict) -> AppTest:
    """Re-run as if the browser sent a chart selection, through AppTest's private API (Streamlit
    1.65), as _run_with_timeline_click does."""
    chart = _chart_id(at, key)
    states = at._tree.get_widget_states()
    widget = states.widgets.add()
    widget.id = chart
    widget.string_value = json.dumps({"selection": selection})
    at._run(states)
    return at


def test_the_hub_renders_with_its_heading():
    at = _at("pipeline")
    assert not at.exception and at.title[0].value == WF_HEADINGS[0]


def test_the_stage_picker_opens_a_stages_panel():
    at = _at("pipeline")
    assert any(s.value.startswith("1. Collect") for s in at.subheader)
    at.segmented_control(key="wf_stage").set_value("evaluate").run()
    assert not at.exception
    assert any(s.value.startswith("4. Train & test") for s in at.subheader)
    assert "Not built: the survival model" in " ".join(c.value for c in at.caption)


@pytest.mark.parametrize("key", workflow.STAGE_KEYS)
def test_every_stage_panel_renders_with_its_links(key):
    at = _at("pipeline")
    at.segmented_control(key="wf_stage").set_value(key).run()
    assert not at.exception
    assert any(s.value.startswith(f"{workflow.STAGE_KEYS.index(key) + 1}.") for s in at.subheader)


def test_clicking_a_stage_box_opens_its_panel_and_an_unknown_one_is_ignored():
    at = _at("pipeline")
    _run_with_selection(at, "pipeline_map", {"stage": [{"key": "explain"}]})
    assert not at.exception and at.session_state["wf_stage"] == "explain"
    assert f"`{workflow.verdict_6b()}`" in _text(at)
    _run_with_selection(at, "pipeline_map", {"stage": [{"key": "no-such-stage"}]})
    assert not at.exception and at.session_state["wf_stage"] == "explain"


def test_the_label_panel_discloses_the_ci_bot():
    at = _at("pipeline")
    at.segmented_control(key="wf_stage").set_value("label").run()
    assert workflow.bot_note(workflow.label_rates()) in [i.value for i in at.info]


def test_the_workflow_modules_are_reloaded_too(monkeypatch):
    """The stale-module fix covers the new modules: stale copies must not reach the hub."""
    monkeypatch.syspath_prepend(str(DEMO))
    for name, attr in (("workflow", "STAGE_KEYS"), ("workflow_charts", "pipeline_map"), ("stages", "hub")):
        monkeypatch.delattr(importlib.import_module(name), attr)
    at = _at("pipeline")
    assert not at.exception and at.title[0].value == WF_HEADINGS[0]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_demo_app.py -q -k "hub or stage or label_panel or reloaded"`
Expected: the new tests fail: `views/pipeline.py` is not a page yet

- [ ] **Step 3: Implement**

**Create `demo/stages.py`:**

```python
"""The "How it was built" section: a pipeline map and four deep dives (workflow-view spec, section 7).

Each page is a function that st.navigation runs through a two-line file in views/, like the story
chapters. It reads this session's Context from st.session_state (set by app.py), shows its
visual, its speaker note when the sidebar toggle is on, and a Previous/Next stage stepper. Widget
choices persist for the session, so a presenter can leave a page and come back to it. Every number
is computed at runtime by workflow.py from committed files; none is typed in here."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import chapters
import triage
import workflow
import workflow_charts as wc
from chapters import _c, _chart, _data

VERDICT_MEANING = {
    "PARTIAL_SHAP_ONLY": "the attribution pattern it predicted is there, but the intervention test "
                         "could not tell an effect from none, so why the model does not transfer "
                         "remains open."}
NOTES = {
    "pipeline": "Walk the six boxes left to right in about a minute: where the repos came from, what "
                "counts as a first review, features that use only what was knowable when a PR opened, "
                "two ways of holding data out, explaining the model, and shipping with every result "
                "tested. Then offer to open any stage.",
    "funnel": "Every narrowing step has a reason; click one. Point at the cap: it is why training and "
              "test rows do not add up. Then the limitation: each group's pool is its most-starred repos.",
    "known": "This is the leak the project designed out. Flip the naive switch: PRs opened in the last "
             "week jump into the lanes of outcomes nobody could know at t.",
    "designs": "Seen in training tests the future of known repos; Unseen repo holds whole repos out. Pick "
               "a fold and compare each dot with that fold's floor, not with zero.",
    "checks": "Stress validity, not success: these were written down before results were read. Point at "
              "the red tile: the pilot missed a guessed expectation, and the doc says why it was not a stop.",
}


def _note(key: str) -> None:
    if _c().notes:
        st.info(f"**Speaker note:** {NOTES[key]}")


def _page(url_path: str):
    """A "How it was built" page by its url_path, or None if the app does not register it."""
    return next((p for p in _c().wf_pages if p.url_path == url_path), None)


def _stepper(url_path: str) -> None:
    """Previous/Next stage buttons, placed by this page's position in the section."""
    pages = _c().wf_pages
    i = [p.url_path for p in pages].index(url_path)
    back, nxt, _ = st.columns([1, 1, 4])
    # keys are per page: a key shared by every page confuses the widget state across a switch
    if i > 0 and back.button("← Previous stage", key=f"wf_back_{i}"):
        st.switch_page(pages[i - 1])
    if i < len(pages) - 1 and nxt.button("Next stage →", key=f"wf_next_{i}", type="primary"):
        st.switch_page(pages[i + 1])


# ---------------------------------------------------------------------------
# the pipeline map
# ---------------------------------------------------------------------------

def _on_stage_click() -> None:
    key = workflow.picked(st.session_state.get("pipeline_map"), "stage", "key")
    if key in workflow.STAGE_KEYS:      # the browser can re-send a stale selection: ignore one we don't know
        st.session_state["wf_stage"] = key


def _collect(c) -> None:
    f = c.wf["repo_funnel"].set_index("step")["count"]
    st.markdown(f"{f['matched the searches']:,} repos matched the nine searches; {f['pooled as candidates']:,} "
                f"were pooled, {f['selected']} drawn and {f['kept']} kept after QC. The pilot gate ran on two "
                "repos before any of it.")


def _label(c) -> None:
    lab, n = c.wf["labels"], c.wf["pr_funnel"].set_index("step").loc["human-authored", "count"]
    st.markdown(f"Under D5, **{lab['d5']:.1%}** of the {n:,} labelled PRs are slow; under D3, which also "
                f"counts commenters with no tie to the repo, {lab['d3']:.1%}. A PR with no first review "
                "within 30 days is slow too, so no label is left waiting.")
    st.info(workflow.bot_note(lab))


def _features(c) -> None:
    a, groups = c.wf["audits"], pd.Series([g["group"] for g in c.wf["groups"]])
    counts = ", ".join(f"{int((groups == g).sum())} {label.lower()}" for g, label in workflow.GROUP_LABELS.items())
    st.markdown(f"{len(groups)} features: {counts}. A brute-force replay of {a['audit_n']} rows matched the "
                f"stored values with a largest difference of {a['audit_max_diff']:g}, and {a['live_matched']} "
                f"of {a['live_checks']} values were confirmed against live GitHub.")


def _evaluate(c) -> None:
    a, folds = c.wf["scenario_a"], c.wf["folds"]
    st.markdown(f"Seen in training: {a['n_train']:,} capped training rows from {a['train_repos']} repos, tested "
                f"on {a['n_test']:,} PRs from 2026. Unseen repo: {len(folds)} folds, each holding out "
                f"{folds['repos'].min()} to {folds['repos'].max()} repos whole.")
    st.caption("Not built: the survival model (Phase 5) and its C-index bar for unseen repos.")


def _explain(c) -> None:
    top = triage.importance("A", c.features).head(3)
    st.markdown("Top drivers of the seen-in-training model, by share of its attribution: "
                + "; ".join(f"{r.label} ({r.share:.0%})" for r in top.itertuples()) + ".")
    verdict = c.wf["verdict"]
    st.markdown(f"The pre-registered Phase 6b test returned `{verdict}`: "
                + VERDICT_MEANING.get(verdict, "see the report."))


def _ship(c) -> None:
    st.markdown("The results the README and the report cite are registered in `writeup_claims.py`, and a "
                "test recomputes each from committed files; another bans phrasings an earlier draft got "
                "wrong. This demo reads only committed files.")


DETAILS = {"collect": _collect, "label": _label, "features": _features, "evaluate": _evaluate,
           "explain": _explain, "ship": _ship}


def _panel(stage: workflow.Stage) -> None:
    c = _c()
    row = c.wf["stages"].set_index("key").loc[stage.key]
    st.subheader(f"{row['title']} · {stage.phases}")
    st.markdown(stage.summary)
    DETAILS[stage.key](c)
    st.markdown("**What it produced**\n" + "\n".join(
        f"- `{path}`: {what} · {'committed' if committed else 'not deployed (gitignored)'}"
        for path, committed, what in stage.artifacts))
    st.markdown(f"**Gate:** {row['gate']}. **Docs:** " + " · ".join(
        f"[{d.split('/')[-1]}]({chapters.GITHUB}/blob/main/{d})" for d in stage.docs))
    links = st.columns(3)
    if stage.page and _page(stage.page):
        links[0].page_link(_page(stage.page), label="Open the deep dive →")
    if stage.chapter is not None:
        links[1].page_link(c.pages[stage.chapter], label="Where the story uses it →")


def hub() -> None:
    c = _c()
    st.title("How it was built")
    st.markdown("Six stages took PRFlowPredict from a search of GitHub to the model in this demo, and "
                "each recorded its checks before its results were read. Click a stage, or pick it below.")
    if "wf_focus" in st.session_state:          # a story chapter asked for this stage
        st.session_state["wf_stage"] = st.session_state.pop("wf_focus")
    st.session_state.setdefault("wf_stage", workflow.STAGE_KEYS[0])
    stages = c.wf["stages"]
    _chart(wc.pipeline_map(stages, st.session_state["wf_stage"], c.mode), on_select=_on_stage_click,
           key="pipeline_map")
    titles = dict(zip(stages["key"], stages["title"]))
    key = st.segmented_control("Stage", workflow.STAGE_KEYS, format_func=titles.get, key="wf_stage",
                               required=True, persist_state="session", label_visibility="collapsed")
    _panel(workflow.stage(key))
    _note("pipeline")
    _stepper("pipeline")
```

**Create `demo/views/pipeline.py`:**

```python
"""A "How it was built" page: the content lives in stages.py."""
import stages

stages.hub()
```

**In `demo/chapters.py`, replace:**

```python
    mode: str
    pages: list = field(default_factory=list)
```

**with:**

```python
    mode: str
    pages: list = field(default_factory=list)
    wf: dict = field(default_factory=dict)              # workflow.bundle(), for the stage links and limits
    wf_pages: list = field(default_factory=list)
```

**In `demo/app.py`, replace:**

```python
A thin shell: page config, cached data, the sidebar and the six chapters in chapters.py. It
hard-codes no result number: every number shown is computed from committed files, and the
results chapter's numbers are tested against the README's."""
```

**with:**

```python
A thin shell: page config, cached data, the sidebar, the six story chapters in chapters.py and
the "How it was built" pages in stages.py. It hard-codes no result number: every number shown is
computed from committed files, and the results chapter's numbers are tested against the README's."""
```

**In `demo/app.py`, replace:**

```python
    sys.path.insert(0, HERE)            # triage.py, charts.py and chapters.py sit next to this file
import triage  # noqa: E402
import charts  # noqa: E402
import chapters  # noqa: E402
```

**with:**

```python
    sys.path.insert(0, HERE)            # the helper modules sit next to this file
import triage  # noqa: E402
import charts  # noqa: E402
import workflow  # noqa: E402
import workflow_charts  # noqa: E402
import chapters  # noqa: E402
import stages  # noqa: E402
```

**In `demo/app.py`, replace:**

```python
triage = importlib.reload(triage)
charts = importlib.reload(charts)
chapters = importlib.reload(chapters)
```

**with:**

```python
triage = importlib.reload(triage)
charts = importlib.reload(charts)
workflow = importlib.reload(workflow)
workflow_charts = importlib.reload(workflow_charts)
chapters = importlib.reload(chapters)
stages = importlib.reload(stages)
```

**In `demo/app.py`, replace:**

```python
prs, features, results, hit_rates = _data()
```

**with:**

```python
@st.cache_data
def _workflow(_prs):
    return workflow.bundle(_prs)          # the leading underscore: Streamlit does not hash the frame


prs, features, results, hit_rates = _data()
wf = _workflow(prs)
```

**In `demo/app.py`, replace:**

```python
notes = st.sidebar.toggle("Speaker notes", value=False)
```

**with:**

```python
st.sidebar.caption("Model and Repo drive chapters 2–4 of the story.")
notes = st.sidebar.toggle("Speaker notes", value=False)
```

**In `demo/app.py`, replace:**

```python
st.session_state["_ctx"] = ctx = chapters.Context(prs, features, results, hit_rates, scenario, repo,
                                                   notes, mode)
```

**with:**

```python
st.session_state["_ctx"] = ctx = chapters.Context(prs, features, results, hit_rates, scenario, repo,
                                                   notes, mode, wf=wf)
```

**In `demo/app.py`, replace:**

```python
pages = [st.Page("views/problem.py", title="1. The problem", url_path="problem", default=True),
         st.Page("views/watch.py", title="2. Watch it work", url_path="watch-it-work"),
         st.Page("views/why.py", title="3. Why it decides", url_path="why"),
         st.Page("views/test_yourself.py", title="4. Test yourself", url_path="test-yourself"),
         st.Page("views/transfer.py", title="5. Does it transfer?", url_path="does-it-transfer"),
         st.Page("views/limits.py", title="6. Honest limits", url_path="honest-limits")]
ctx.pages = pages
st.navigation(pages).run()
```

**with:**

```python
story = [st.Page("views/problem.py", title="1. The problem", url_path="problem", default=True),
         st.Page("views/watch.py", title="2. Watch it work", url_path="watch-it-work"),
         st.Page("views/why.py", title="3. Why it decides", url_path="why"),
         st.Page("views/test_yourself.py", title="4. Test yourself", url_path="test-yourself"),
         st.Page("views/transfer.py", title="5. Does it transfer?", url_path="does-it-transfer"),
         st.Page("views/limits.py", title="6. Honest limits", url_path="honest-limits")]
built = [st.Page("views/pipeline.py", title="Pipeline map", url_path="pipeline")]
ctx.pages, ctx.wf_pages = story, built          # Back/Next walk the story; the stepper walks `built`
st.navigation({"The story": story, "How it was built": built}).run()
```

- [ ] **Step 4: Run the app tests**

Run: `python -m pytest tests/test_demo_app.py -q`
Expected: all pass (the story's tests unchanged)

- [ ] **Step 5: Commit**

```bash
git add demo/stages.py demo/views/pipeline.py demo/app.py demo/chapters.py tests/test_demo_app.py
git commit -F - <<'EOF'
feat(demo): a "How it was built" section with a clickable pipeline map

A second navigation section beside the six-chapter story. The map's six stages open a panel
with runtime counts, what each stage produced, its gate and its links. The new modules are
reloaded every run, like the story's.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

### Task 6: Data funnel and validity checks pages

**Files:**
- Create: `demo/views/funnel.py`, `demo/views/checks.py`
- Modify: `demo/stages.py` (append), `demo/app.py`, `tests/test_demo_app.py` (append)

**Interfaces:**
- Consumes: Task 5's helpers; `workflow.removed`, `workflow.pool_note`, `workflow.gate_value_text`, `workflow_charts.funnel`, `workflow_charts.gate_tiles`.
- Produces: `stages.funnel()`, `stages.checks()`; chart keys `repo_funnel`, `pr_funnel`, `gate_tiles`.

- [ ] **Step 1: Write the failing tests**

**Append to `tests/test_demo_app.py`:**

```python


def test_a_funnel_click_lists_what_the_step_removed():
    at = _at("funnel")
    dropped, note = workflow.removed("kept")
    assert note not in _text(at)
    _run_with_selection(at, "repo_funnel", {"step": [{"step": "kept"}]})
    assert not at.exception
    assert note in _text(at)
    assert any(d.value["repo"].tolist() == dropped["repo"].tolist() for d in at.dataframe if "repo" in d.value)


def test_a_selection_naming_the_other_funnels_step_is_ignored():
    at = _at("funnel")
    _run_with_selection(at, "repo_funnel", {"step": [{"step": "human-authored"}]})
    assert not at.exception and "**human-authored:**" not in _text(at)


def test_the_funnel_page_discloses_the_star_sorted_pools():
    assert workflow.pool_note(workflow.pool_bias()) in [i.value for i in _at("funnel").info]


@pytest.mark.parametrize("phase,check_id,phrase", [
    ("2", 4, "passes exactly at its threshold"),
    ("3", 5, "the REST API confirmed this project's value"),
    ("0", 4, "expectation missed, not a stop"),
])
def test_clicking_a_check_tile_shows_the_check(phase, check_id, phrase):
    at = _at("checks")
    _run_with_selection(at, "gate_tiles", {"check": [{"phase": phase, "id": check_id}]})
    assert not at.exception and phrase in _text(at)


def test_a_tile_selection_for_no_such_check_shows_nothing():
    at = _at("checks")
    _run_with_selection(at, "gate_tiles", {"check": [{"phase": "2", "id": 9}]})
    assert not at.exception and "check 9" not in _text(at)


def test_the_checks_page_frames_validity_not_success():
    text = _text(_at("checks"))
    assert "**Validity, not success.**" in text and "They do not say the model is good" in text
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_demo_app.py -q -k "funnel or tile or check"`
Expected: the new tests fail: the pages do not exist yet

- [ ] **Step 3: Implement**

**Append to `demo/stages.py`:**

```python


# ---------------------------------------------------------------------------
# data funnel
# ---------------------------------------------------------------------------

def _removed(event, steps: pd.DataFrame) -> None:
    step = workflow.picked(event, "step", "step")
    if step not in set(steps["step"]):
        st.caption("Click a step to see what it removed.")
        return
    frame, note = workflow.removed(step)
    st.markdown(f"**{step}:** {note}")
    if len(frame):
        st.dataframe(frame, hide_index=True, width="stretch")


def funnel() -> None:
    c, wf = _c(), _c().wf
    st.title("Data funnel")
    st.markdown("From every repo GitHub's searches matched to the pull requests this demo replays. "
                "Hover a step for its source file; click it to see what it removed and why.")
    left, right = st.columns(2)
    with left:
        st.subheader("Repos")
        event = _chart(wc.funnel(wf["repo_funnel"], "repos", c.mode, log=True), on_select="rerun",
                       key="repo_funnel")
        _removed(event, wf["repo_funnel"])
    with right:
        st.subheader("Pull requests")
        event = _chart(wc.funnel(wf["pr_funnel"], "pull requests", c.mode), on_select="rerun", key="pr_funnel")
        _removed(event, wf["pr_funnel"])
    st.caption(f"Training rows are capped at {workflow.CAP_FRAC:.0%} of all training rows per repo and test "
               "rows are not, so training rows and test PRs add up to fewer than the modelled PRs. QC rules "
               "are structural: bot share, human PR count and language.")
    st.info(workflow.pool_note(wf["pool_bias"]))
    with st.expander("Each group's search and pool"):
        st.dataframe(wf["pool_bias"], hide_index=True, width="stretch",
                     column_config={"share": st.column_config.NumberColumn("pool share of group", format="percent")})
    _data(pd.concat([wf["repo_funnel"].assign(unit="repos"), wf["pr_funnel"].assign(unit="pull requests")]),
          "Data behind the funnels")
    _note("funnel")
    _stepper("data-funnel")


# ---------------------------------------------------------------------------
# validity checks
# ---------------------------------------------------------------------------

def _tile_note(phase: str, check: int, wf: dict) -> str | None:
    if (phase, check) == ("2", 4):
        return "This check passes exactly at its threshold."
    if (phase, check) == ("3", 5) and wf["live"]:
        m = wf["live"][0]
        return (f"One live value differed at first: {m['pr']} {m['field']} read {m['live']} in GitHub's "
                f"search against {m['ours']} here, and the REST API confirmed this project's value.")
    return None


def _show_check(row: pd.Series, wf: dict) -> None:
    st.markdown(f"**{workflow.PHASE_NAMES[row['phase']]}, check {row['id']}:** {row['check']}")
    st.markdown(f"Status: **{row['status']}** · recorded: {workflow.gate_value_text(row['value'])}")
    note = _tile_note(row["phase"], int(row["id"]), wf)
    if note:
        st.markdown(note)
    with st.expander("Recorded value, raw"):
        if isinstance(row["value"], (dict, list)):
            st.json(row["value"])
        else:
            st.code(str(row["value"]), language=None)


def checks() -> None:
    c, wf = _c(), _c().wf
    st.title("Validity checks")
    st.markdown("**Validity, not success.** Each phase wrote down its checks before its results were "
                "read. They test that the pipeline did what it claims: rows conserved, no label in any "
                "feature, refits reproducing, attributions adding up. They do not say the model is good. "
                "Click a tile to read one.")
    g = wf["gates"]
    event = _chart(wc.gate_tiles(g, c.mode), on_select="rerun", key="gate_tiles")
    hit = workflow.picked(event, "check")
    row = g[(g["phase"] == str(hit.get("phase"))) & (g["id"] == hit.get("id"))] if hit else g.iloc[0:0]
    if len(row):                    # a stale selection from another page's data finds no row
        _show_check(row.iloc[0], wf)
    st.caption("Phase 1, the collection, has no gate file: the pilot gate tested its machinery first. "
               "Phase 5, a survival model, was not built.")
    _data(g.assign(value=g["value"].map(workflow.gate_value_text)), "Every check")
    _note("checks")
    _stepper("validity-checks")
```

**Create `demo/views/funnel.py`:**

```python
"""A "How it was built" page: the content lives in stages.py."""
import stages

stages.funnel()
```

**Create `demo/views/checks.py`:**

```python
"""A "How it was built" page: the content lives in stages.py."""
import stages

stages.checks()
```

**In `demo/app.py`, replace:**

```python
built = [st.Page("views/pipeline.py", title="Pipeline map", url_path="pipeline")]
```

**with:**

```python
built = [st.Page("views/pipeline.py", title="Pipeline map", url_path="pipeline"),
         st.Page("views/funnel.py", title="Data funnel", url_path="data-funnel"),
         st.Page("views/checks.py", title="Validity checks", url_path="validity-checks")]
```

- [ ] **Step 4: Run the app tests**

Run: `python -m pytest tests/test_demo_app.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add demo/stages.py demo/views/funnel.py demo/views/checks.py demo/app.py tests/test_demo_app.py
git commit -F - <<'EOF'
feat(demo): data funnel and validity checks pages

The funnels narrow 44,198 searched repos to the demo's PRs, each step clickable for what it
removed and why, with the star-sorted pool disclosed. The checks board shows all 35 recorded
checks as validity, not success.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

### Task 7: Known at time t and two test designs pages

**Files:**
- Create: `demo/views/known_at_t.py`, `demo/views/designs.py`
- Modify: `demo/stages.py` (append), `demo/app.py`, `tests/test_demo_app.py` (append)

**Interfaces:**
- Consumes: Task 5's helpers; `workflow.replay_at`, `workflow.default_pr`, `workflow.design_rows`, `workflow_charts.replay_strip`, `trailing_line`, `split_bars`, `fold_dots`.
- Produces: `stages.known_at_t()`, `stages.designs()`; persisted widget keys `kt_repo`, `kt_pr`, `kt_naive`, `td_design`, `td_fold`.

- [ ] **Step 1: Write the failing tests**

**Append to `tests/test_demo_app.py`:**

```python


@pytest.mark.parametrize("i", range(5))
def test_every_workflow_page_renders_with_its_heading(i):
    at = _at(WF_VIEWS[i])
    assert not at.exception and at.title[0].value == WF_HEADINGS[i]


@pytest.mark.parametrize("i", range(4))
def test_next_stage_opens_the_following_workflow_page(i):
    at = _at(WF_VIEWS[i])
    at.button(key=f"wf_next_{i}").click().run()
    assert at.title[0].value == WF_HEADINGS[i + 1]


@pytest.mark.parametrize("i", range(1, 5))
def test_previous_stage_opens_the_preceding_workflow_page(i):
    at = _at(WF_VIEWS[i])
    at.button(key=f"wf_back_{i}").click().run()
    assert at.title[0].value == WF_HEADINGS[i - 1]


def test_the_hub_has_no_previous_and_the_checks_page_no_next():
    assert "wf_back_0" not in [b.key for b in _at("pipeline").button]
    assert "wf_next_4" not in [b.key for b in _at("checks").button]


def test_the_scrubber_opens_on_the_default_repo_and_matches_the_stored_feature():
    at = _at("known_at_t")
    check = workflow.trailing_check(triage.load())
    assert at.selectbox(key="kt_repo").value == "kdlbs/kandev"
    assert at.select_slider(key="kt_pr").value == workflow.default_pr(check, "kdlbs/kandev")
    assert "✓ Matches the stored feature." in _text(at)


def test_the_naive_rule_is_labelled_a_counterfactual():
    at = _at("known_at_t")
    at.toggle(key="kt_naive").set_value(True).run()
    assert not at.exception
    assert "a counterfactual computed here, not a project result" in _text(at)


def test_moving_the_scrubber_and_changing_repo_keep_the_readout_exact():
    at = _at("known_at_t")
    prs = triage.load()
    check = workflow.trailing_check(prs)
    kandev = prs[prs["repo"] == "kdlbs/kandev"].sort_values(["created_at", "pr_id"])
    at.select_slider(key="kt_pr").set_value(kandev["pr_id"].iloc[50]).run()
    assert not at.exception and f"PR #{kandev['number'].iloc[50]} opens" in _text(at)
    other = next(r for r in workflow.exact_repos(check) if r != "kdlbs/kandev")
    at.selectbox(key="kt_repo").set_value(other).run()
    assert not at.exception
    assert at.select_slider(key="kt_pr").value == workflow.default_pr(check, other)
    assert "✓ Matches the stored feature." in _text(at)


def test_the_test_designs_page_shows_each_folds_counts():
    at = _at("designs")
    a = workflow.scenario_a()
    assert [m.value for m in at.metric][:2] == [f"{a['n_train']:,}", f"{a['n_test']:,}"]
    at.segmented_control(key="td_design").set_value("Unseen repo (repo folds)").run()
    at.segmented_control(key="td_fold").set_value(3).run()
    assert not at.exception
    r = workflow.fold_results().set_index("fold").loc[2]
    assert [m.value for m in at.metric] == [f"{int(r['n_train']):,}", f"{int(r['n_test']):,}", f"{int(r['repos'])}"]


def test_the_designs_page_states_the_tuning_overlap():
    assert "Its effect was not measured" in _text(_at("designs"))


def test_workflow_choices_survive_a_page_switch():
    at = _at("designs")
    at.segmented_control(key="td_design").set_value("Unseen repo (repo folds)").run()
    at.switch_page("views/pipeline.py").run()
    at.switch_page("views/designs.py").run()
    assert at.segmented_control(key="td_design").value == "Unseen repo (repo folds)"


def test_workflow_pages_hide_speaker_notes_until_toggled():
    for view in WF_VIEWS:
        at = _at(view)
        assert not [i for i in at.info if i.value.startswith("**Speaker note:**")], view
        at.sidebar.toggle[0].set_value(True).run()
        assert [i for i in at.info if i.value.startswith("**Speaker note:**")], view


@pytest.mark.parametrize("key", [s.key for s in workflow.STAGES if s.page])
def test_each_stage_with_a_deep_dive_links_to_it(key):
    at = _at("pipeline")
    at.segmented_control(key="wf_stage").set_value(key).run()
    assert workflow.stage(key).page in [link.proto.page for link in at.get("page_link")]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_demo_app.py -q -k "workflow or stage or scrubber or naive or designs"`
Expected: the new tests fail: the pages do not exist yet

- [ ] **Step 3: Implement**

**Append to `demo/stages.py`:**

```python


# ---------------------------------------------------------------------------
# known at time t
# ---------------------------------------------------------------------------

DEFAULT_REPO = "kdlbs/kandev"


def _feature_kinds(c) -> None:
    labels = {f["feature"]: f["label"] for f in c.features}
    groups = pd.DataFrame(c.wf["groups"])
    for col, (g, label) in zip(st.columns(len(workflow.GROUP_LABELS)), workflow.GROUP_LABELS.items()):
        names = groups.loc[groups["group"] == g, "feature"].map(labels)
        col.markdown(f"**{label}** ({len(names)})")
        col.caption(" · ".join(names))


def known_at_t() -> None:
    c, wf = _c(), _c().wf
    st.title("Known at time t")
    st.markdown("A score is only useful if it could be computed when the PR opened. Each of the model's "
                "features is one of four kinds, by when its value is known:")
    _feature_kinds(c)
    a = wf["audits"]
    snapshot = sum(g["group"] == "snapshot" for g in wf["groups"])
    st.markdown(f"The {snapshot} snapshot features are the stated exception: 2026 values applied to earlier "
                f"PRs. Two audits check the rest: a brute-force replay of {a['audit_n']} rows matched the "
                f"stored features with a largest difference of {a['audit_max_diff']:g}, and {a['live_matched']} "
                f"of {a['live_checks']} values on {a['live_rows']} PRs were confirmed against live GitHub.")

    st.subheader("Replay one repo's history")
    st.markdown("The trailing 90-day slow rate is both the baseline and one of the model's strongest "
                "features. When a PR opens at time t, an earlier PR's outcome counts only if it was "
                "knowable: the PR is at least 7 days old, or it already had its first review. Pick a "
                "moment and see which earlier PRs count.")
    exact = wf["exact"]
    repo = st.selectbox("Repo", exact, index=exact.index(DEFAULT_REPO) if DEFAULT_REPO in exact else 0,
                        key="kt_repo", persist_state="session")
    rows = c.prs[c.prs["repo"] == repo].sort_values(["created_at", "pr_id"], kind="mergesort")
    names = {p: f"#{n} · {d:%d %b %Y %H:%M}" for p, n, d in zip(rows["pr_id"], rows["number"], rows["created_at"])}
    if st.session_state.get("kt_pr") not in names:          # first visit, or a different repo
        st.session_state["kt_pr"] = workflow.default_pr(wf["check"], repo)
    pr_id = st.select_slider("Pull request, by when it opened", options=list(names), format_func=names.get,
                             key="kt_pr", persist_state="session")
    naive = st.toggle("Naive rule: count every earlier PR in the window, knowable or not", key="kt_naive",
                      persist_state="session")
    frame, s = workflow.replay_at(c.prs, repo, pr_id, "naive" if naive else "resolvable", wf["prior"])
    domain = (s["t"] - pd.Timedelta(days=workflow.TRAILING_DAYS + 30), s["t"] + pd.Timedelta(days=2))
    _chart(wc.replay_strip(frame, s["t"], c.mode, domain))
    near = rows[(rows["created_at"] >= domain[0]) & (rows["created_at"] <= domain[1])]
    _chart(wc.trailing_line(near, s["t"], s["prior"], c.mode, domain))
    if naive:
        st.markdown(f"The naive rule counts **{s['k']}** earlier PRs, **{s['unknown']}** of them with an "
                    f"outcome nobody could know at t, and gives **{s['rate']:.3f}**; the stored feature is "
                    f"{s['stored_rate']:.3f}. This is a counterfactual computed here, not a project result.")
    else:
        st.markdown(f"At t, PR #{s['number']} opens. **{s['k']}** earlier PRs in the 90-day window had a "
                    f"knowable outcome and **{s['slow']}** of them stalled; **{s['unknown']}** more opened "
                    "in the last 7 days without a review yet, so they are left out. Shrunk toward the prior: "
                    f"({s['slow']} + {workflow.ALPHA:g} × {s['prior']:.3f}) / ({s['k']} + "
                    f"{workflow.ALPHA:g}) = **{s['rate']:.3f}**.")
        st.markdown("✓ Matches the stored feature." if s["matches"] else "✗ Differs from the stored feature.")
    st.caption(f"The repo list holds the {len(exact)} of {len(triage.repos(c.prs))} repos where this page "
               "re-derives the stored feature exactly on every PR; for the others, the project's replay "
               "also saw PRs this extract does not hold. Only the trailing slow rate is re-derived here: "
               "backlog and author-history features need bot PRs and author identities the extract leaves out.")
    _data(frame, "Data behind the strip")
    _note("known")
    _stepper("known-at-time-t")


# ---------------------------------------------------------------------------
# two test designs
# ---------------------------------------------------------------------------

DESIGNS = {"Seen in training (time cut)": None, "Unseen repo (repo folds)": "B"}


def designs() -> None:
    c, wf = _c(), _c().wf
    st.title("Two test designs")
    st.markdown("A model is judged on data it was never shown. PRFlowPredict holds data out in two ways "
                "and reports both.")
    st.session_state.setdefault("td_design", next(iter(DESIGNS)))
    design = st.segmented_control("Test design", list(DESIGNS), key="td_design", required=True,
                                  persist_state="session")
    rows, folds = wf["splits"], wf["folds"]
    order = list(rows["repo"])
    m = st.columns(3)
    if DESIGNS[design] is None:
        a = wf["scenario_a"]
        m[0].metric("Training rows", f"{a['n_train']:,}")
        m[1].metric("Test PRs", f"{a['n_test']:,}")
        m[2].metric("Repos trained on / tested", f"{a['train_repos']} / {a['test_repos']}")
        st.markdown(f"Each repo's PRs opened before 2026 train the model, capped at {workflow.CAP_FRAC:.0%} "
                    "of the training rows per repo; its PRs from 2026 test it. The question: how well does it "
                    "predict the future of projects it knows?")
        long = workflow.design_rows(rows)
    else:
        st.session_state.setdefault("td_fold", 1)
        number = st.segmented_control("Held-out fold", list(range(1, len(folds) + 1)), key="td_fold",
                                      format_func=lambda f: f"fold {f}", required=True, persist_state="session")
        r = folds.set_index("fold").loc[number - 1]
        m[0].metric("Training rows", f"{int(r['n_train']):,}")
        m[1].metric("Test PRs", f"{int(r['n_test']):,}")
        m[2].metric("Repos held out", f"{int(r['repos'])}")
        st.markdown(f"The repos are split into {len(folds)} folds. Each model trains on every PR of the other "
                    "folds' repos, capped the same way, and is tested on every PR of its fold's repos, from "
                    "the whole window. The question: how well does it predict for a project it has never seen?")
        long = workflow.design_rows(rows, number - 1)
    _chart(wc.split_bars(long, order, c.mode))
    if DESIGNS[design] is not None:
        st.subheader("This fold's results")
        _chart(wc.fold_dots(folds, number - 1, c.mode))
        st.caption("AUC-PR starts from the fold's base rate (the tick), not from zero, and the folds' base "
                   "rates differ widely: compare each dot with its own fold's tick.")
    st.markdown(
        "- Per-repo counts come from a Phase 6b output file, `data/phase6b_transfer.csv`; they reproduce "
        "every stored training count exactly.\n"
        "- Tuning used the seen-in-training rows, which include the pre-2026 rows of the repos each "
        "unseen-repo fold later holds out. Its effect was not measured.\n"
        "- The story's Unseen repo switch shows only 2026 PRs; the fold results cover the whole window.\n"
        "- Compare the designs with AUC-PR, not precision on the top 10: the unseen-repo test ranks far "
        "larger pools of PRs per repo.")
    _data(folds, "Data behind the fold results")
    _note("designs")
    _stepper("test-designs")
```

**Create `demo/views/known_at_t.py`:**

```python
"""A "How it was built" page: the content lives in stages.py."""
import stages

stages.known_at_t()
```

**Create `demo/views/designs.py`:**

```python
"""A "How it was built" page: the content lives in stages.py."""
import stages

stages.designs()
```

**In `demo/app.py`, replace:**

```python
built = [st.Page("views/pipeline.py", title="Pipeline map", url_path="pipeline"),
         st.Page("views/funnel.py", title="Data funnel", url_path="data-funnel"),
         st.Page("views/checks.py", title="Validity checks", url_path="validity-checks")]
```

**with:**

```python
built = [st.Page("views/pipeline.py", title="Pipeline map", url_path="pipeline"),
         st.Page("views/funnel.py", title="Data funnel", url_path="data-funnel"),
         st.Page("views/known_at_t.py", title="Known at time t", url_path="known-at-time-t"),
         st.Page("views/designs.py", title="Two test designs", url_path="test-designs"),
         st.Page("views/checks.py", title="Validity checks", url_path="validity-checks")]
```

- [ ] **Step 4: Run the app tests**

Run: `python -m pytest tests/test_demo_app.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add demo/stages.py demo/views/known_at_t.py demo/views/designs.py demo/app.py tests/test_demo_app.py
git commit -F - <<'EOF'
feat(demo): known-at-time-t scrubber and two test designs pages

Scrub a repo's history and watch which earlier outcomes the trailing rate may count; a naive
switch shows the leak the 7-day rule prevents. The test-design page shows the time cut, the
repo folds and each fold's results against its own floor.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

### Task 8: Story links and the two disclosures in the write-up

**Files:**
- Modify: `demo/chapters.py`, `writeup_claims.py`, `docs/REPORT.md`, `README.md`, `tests/test_demo_app.py` (append)

**Interfaces:**
- Consumes: `workflow.bot_note`, `workflow.pool_note`, `Context.wf`, `Context.wf_pages`, session key `wf_focus`.
- Produces: `chapters.STAGE_LINKS`, `chapters._stage(key)` (button `stage_{key}`); claims `ci_bot_rates`, `pool_share`.

- [ ] **Step 1: Write the failing tests**

**Append to `tests/test_demo_app.py`:**

```python


@pytest.mark.parametrize("chapter,key,heading,stage", [
    ("problem", "problem", "How it was built", "label"),
    ("watch", "watch", "Known at time t", None),
    ("why", "why", "How it was built", "explain"),
    ("test_yourself", "test", "Two test designs", None),
    ("transfer", "transfer", "Two test designs", None),
    ("limits", "limits", "Validity checks", None),
])
def test_each_chapters_stage_link_opens_how_it_was_built(chapter, key, heading, stage):
    at = _at(chapter)
    at.button(key=f"stage_{key}").click().run()
    assert not at.exception and at.title[0].value == heading
    if stage:
        assert at.session_state["wf_stage"] == stage
        assert any(s.value.startswith(f"{workflow.STAGE_KEYS.index(stage) + 1}.") for s in at.subheader)


def test_honest_limits_discloses_the_ci_bot_and_the_star_sorted_pools():
    text = _text(_at("limits"))
    assert workflow.bot_note(workflow.label_rates()) in text
    assert workflow.pool_note(workflow.pool_bias()) in text
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_demo_app.py -q -k "stage_link or honest_limits"`
Expected: the new tests fail: no `stage_*` buttons, no disclosures

- [ ] **Step 3: Implement the story links and the disclosures**

**In `demo/chapters.py`, replace:**

```python
import charts
import triage
```

**with:**

```python
import charts
import triage
import workflow
```

**In `demo/chapters.py`, replace:**

```python
    "limits": "Close on what was not measured. This is what makes the rest credible.",
}
```

**with:**

```python
    "limits": "Close on what was not measured. This is what makes the rest credible.",
}
# chapter -> (its pipeline stage, the "How it was built" page that shows it, the hub stage to open)
STAGE_LINKS = {
    "problem": ("Label", "pipeline", "label"),
    "watch": ("Features", "known-at-time-t", None),
    "why": ("Explain", "pipeline", "explain"),
    "test": ("Train & test", "test-designs", None),
    "transfer": ("Train & test", "test-designs", None),
    "limits": ("all six stages", "validity-checks", None),
}
```

**In `demo/chapters.py`, replace:**

```python
def _nav(i: int) -> None:
```

**with:**

```python
def _stage(key: str) -> None:
    """One line under a chapter's title: its pipeline stage, and a jump to how it was built. The hub
    stage travels in a plain session key, which the hub copies into its stage picker."""
    name, url, stage = STAGE_LINKS[key]
    text, go = st.columns([4, 1], vertical_alignment="center")
    text.markdown(f"Pipeline stage: **{name}**")
    if go.button("How it was built →", key=f"stage_{key}"):
        if stage:
            st.session_state["wf_focus"] = stage
        st.switch_page(next(p for p in _c().wf_pages if p.url_path == url))


def _nav(i: int) -> None:
```

**In `demo/chapters.py`, replace:**

```python
    st.title("Some pull requests wait weeks for a first review")
```

**with:**

```python
    st.title("Some pull requests wait weeks for a first review")
    _stage("problem")
```

**In `demo/chapters.py`, replace:**

```python
    st.title("Watch it work")
```

**with:**

```python
    st.title("Watch it work")
    _stage("watch")
```

**In `demo/chapters.py`, replace:**

```python
    st.title("Why it decides")
```

**with:**

```python
    st.title("Why it decides")
    _stage("why")
```

**In `demo/chapters.py`, replace:**

```python
    st.title("Test yourself")
```

**with:**

```python
    st.title("Test yourself")
    _stage("test")
```

**In `demo/chapters.py`, replace:**

```python
    st.title("Does it transfer to a new project?")
```

**with:**

```python
    st.title("Does it transfer to a new project?")
    _stage("transfer")
```

**In `demo/chapters.py`, replace:**

```python
    st.title("Honest limits")
```

**with:**

```python
    st.title("Honest limits")
    _stage("limits")
```

**In `demo/chapters.py`, replace:**

```python
        "- **A list of PRs still waiting cannot measure ranking.** Most of them go on to stall; the "
        "ranking is measured when PRs open, over every test PR.")
```

**with:**

```python
        "- **A list of PRs still waiting cannot measure ranking.** Most of them go on to stall; the "
        "ranking is measured when PRs open, over every test PR.\n"
        f"- {workflow.bot_note(c.wf['labels'])}\n"
        f"- {workflow.pool_note(c.wf['pool_bias'])}")
```

- [ ] **Step 4: Register the report's new numbers, then write them**

**In `writeup_claims.py`, replace:**

```python
P6B_RELIANCE = "data/phase6b_reliance.csv"
EXPERIMENTS = "data/experiments.csv"
```

**with:**

```python
P6B_RELIANCE = "data/phase6b_reliance.csv"
EXPERIMENTS = "data/experiments.csv"
POOL = "data/cohort/candidate_pool.json"
SEARCH_P0 = tuple(f"data/cohort/raw_search/search_{lang}_{lo}_{hi}_p0.json"
                  for lang in ("Python", "TypeScript", "Go") for lo, hi in ((200, 800), (800, 3000), (3000, 15000)))
CI_BOT_REPOS = ("kubernetes/autoscaler", "kubernetes-sigs/gateway-api-inference-extension")
```

**In `writeup_claims.py`, replace:**

```python
def _nlr_b_fold_range() -> tuple[float, float]:
```

**with:**

```python
def _ci_bot_rates() -> str:
    """The two repos whose first reviews mostly come from k8s-ci-robot, at their D5 slow rates."""
    reps = {r["repo"]: r for r in json.loads(_text(KEPT))["repos"]}
    a, b = (reps[r]["is_slow_d5"] for r in CI_BOT_REPOS)
    return f"({a:.1%} and {b:.1%} under D5)"


def _pool_share() -> str:
    """Each search group's candidate pool against how many repos its search matched."""
    pool = pd.DataFrame(json.loads(_text(POOL))).groupby("_cell").size()
    matched = {}
    for rel in SEARCH_P0:
        _, lang, lo, hi, _ = Path(rel).stem.split("_")
        matched[f"{lang}:{lo}-{hi}"] = json.loads(_text(rel))["data"]["search"]["repositoryCount"]
    share = pool / pd.Series(matched)
    return f"each group's {pool.iloc[0]} most-starred matches, between {share.min():.1%} and {share.max():.1%} of the group"


def _nlr_b_fold_range() -> tuple[float, float]:
```

**In `writeup_claims.py`, replace:**

```python
    Claim("nlr_b_fold_range", lambda: (lambda lo, hi: f"from {_f3(lo)} to {_f3(hi)}")(*_nlr_b_fold_range()),
          "from 0.507 to 0.965", (REPORT,), (P4_RUNS,)),
```

**with:**

```python
    Claim("nlr_b_fold_range", lambda: (lambda lo, hi: f"from {_f3(lo)} to {_f3(hi)}")(*_nlr_b_fold_range()),
          "from 0.507 to 0.965", (REPORT,), (P4_RUNS,)),

    # --- limitations found while building the workflow view
    Claim("ci_bot_rates", _ci_bot_rates, "(3.1% and 0.7% under D5)", (REPORT,), (KEPT,)),
    Claim("pool_share", _pool_share,
          "each group's 200 most-starred matches, between 1.3% and 20.6% of the group", (REPORT,), (POOL, *SEARCH_P0)),
```

Run: `python -m pytest tests/test_writeup_claims.py -q`
Expected: fails: `'(3.1% and 0.7% under D5)' is missing from docs/REPORT.md` (and the pool claim)

**In `docs/REPORT.md`, replace:**

```markdown
- **Snapshot repo attributes.** Maintainer counts, CODEOWNERS and similar features are 2026
  snapshots applied to PRs from 2024 onward, and star tiers are defined by 2026 star counts.
```

**with:**

```markdown
- **Snapshot repo attributes.** Maintainer counts, CODEOWNERS and similar features are 2026
  snapshots applied to PRs from 2024 onward, and star tiers are defined by 2026 star counts.
- **A CI bot counts as a reviewer.** Kubernetes' `k8s-ci-robot` account is registered on GitHub as
  an ordinary user, not a bot, so the D5 label counts its automated comments as first reviews. It
  supplies most first reviews in kubernetes/autoscaler and
  kubernetes-sigs/gateway-api-inference-extension, so their slow rates
  (3.1% and 0.7% under D5) understate how long people took. Fixing it means relabelling and
  re-running every later phase; the results here use the label as built.
- **Candidate pools are each group's most-starred repos.** The searches returned each language ×
  star-tier group's matches sorted by stars, and the candidate pool took
  each group's 200 most-starred matches, between 1.3% and 20.6% of the group. The seeded random
  draw therefore chose among the most-starred repos of each group, not across its whole star range.
```

**In `README.md`, replace:**

```markdown
model that never saw the repo.
```

**with:**

```markdown
model that never saw the repo. A second section, *How it was built*, walks the pipeline: a
clickable map of the six stages, the data funnel, what the model could know when a PR opened, the
two test designs, and every validity check.
```

**In `README.md`, replace:**

```markdown
| Demo | `build_demo_data.py` · `demo/triage.py` · `demo/app.py` |
```

**with:**

```markdown
| Demo | `build_demo_data.py` · `build_workflow_data.py` · `demo/app.py` · `demo/chapters.py` · `demo/stages.py` · `demo/triage.py` · `demo/workflow.py` · `demo/charts.py` · `demo/workflow_charts.py` · `demo/views/` |
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_demo_app.py tests/test_writeup_claims.py -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add demo/chapters.py writeup_claims.py docs/REPORT.md README.md tests/test_demo_app.py
git commit -F - <<'EOF'
feat(demo): story chapters link to how they were built; disclose two limitations

Each chapter names its pipeline stage and opens the matching page. A CI bot registered as a
user counts as a reviewer in two Kubernetes repos, and each group's candidate pool is its
most-starred repos: both now in Honest limits and REPORT section 8, their numbers registered.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
```

### Task 9: Whole-suite check and a visual pass

**Files:** none new.

- [ ] **Step 1: Run the full suite**

Run: `python -m pytest tests -q`
Expected: all pass (540 before this plan; the new tests add about 110)

- [ ] **Step 2: Verify the gitignored Phase 4 artifacts are untouched**

Nothing in this plan writes under `data/models`, `data/predictions`, `data/features` or `data/processed`. Confirm no file there changed today:

Run: `find data/models data/predictions data/features data/processed -newermt "$(date +%Y-%m-%d)" -type f | head`
Expected: no output

- [ ] **Step 3: Look at it**

Run `streamlit run demo/app.py`, open *How it was built* in light and dark mode, click a stage box, a funnel step, a check tile; move the scrubber and flip the naive switch; switch test design and fold; use a story chapter's "How it was built →" button. Fix anything that renders badly, with a test where the fault is testable.

- [ ] **Step 4: Commit any fixes** (only if Step 3 changed something), then hand over with superpowers:finishing-a-development-branch.

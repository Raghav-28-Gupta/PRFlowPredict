# Demo Story — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the deployed Streamlit demo into a six-chapter, interactive walkthrough for a 3–5 minute live presentation:
1. the wait-time histogram;
2. a repo timeline you can click;
3. a per-PR "why" chart;
4. a guess-the-stall game;
5. hoverable results;
6. honest limits.

It also carries hidden speaker notes.

**Architecture:**
- `build_demo_data.py` adds every PR's per-feature SHAP values, base values and feature values to the committed extract, and writes `demo/data/features.json`.
- `demo/triage.py` holds all chapter logic: pandas plus the standard library.
- `demo/charts.py` builds Altair charts as pure functions.
- `demo/chapters.py` renders the six chapters.
- `demo/views/*.py` are two-line page files.
- `demo/app.py` is the shell: the sidebar plus `st.navigation`.

**Tech Stack:** Python 3.13, pandas, Streamlit 1.65.0 (`st.navigation`, `st.altair_chart(on_select=…)`, AppTest), Altair 6.3.0, pytest.

**Spec:** `docs/design/specs/2026-10-04-demo-story-design.md`

**This plan was dry-run before it was committed.** A replica of the repo was built from `git archive` of `demo-story`. Every file below was written and run there, and then **re-applied from this plan file alone** into a second fresh replica and run again, with the same results. The build ran against the real gitignored Phase 4 outputs, which it reads and never writes. The results:
- **The extract keeps its 15 original columns byte-identical, drivers text included,** so every existing test stays valid.
- **It adds 113 columns, 128 in total, 8.0 MB.** Base value plus SHAP values reproduces each score's logit to within 1.5e-7.
- **A rebuild is byte-identical:** `prs.parquet` sha256 `0c521f22d1b68021d043323dddb54927cb67dc490ddd07438b8a0c6d83efa10a`, `features.json` sha256 `f64f5a84d3bfc0e1a60ea286cd3cc09664d0dc30790c28b054adacbbb95872e5`.
- **The suite passes:** 528 passed, 2 skipped. The 2 skips need gitignored pilot data the replica lacks; expect **530 passed** in the real repo, up from 462.
- **Every mutation listed in Tasks 1–4 is caught** (19 in total), each restored with `git checkout`.
- **Every chart was rendered to PNG in light and dark mode and looked at.**
  - The first render showed four defects, all fixed: legends that labelled both colours alike, axis labels broken after every word, a results label colliding with fold dots, and a precision label that read like the whisker's value.
  - The colour pairs pass the dataviz palette validator in both modes.

Three findings from the dry run shaped this plan:
- **AppTest keeps the current page between runs only for file pages,** not function pages, hence `demo/views/`. The spec is amended to match.
- **The Back/Next keys are per chapter** (`back_i`, `next_i`).
- **One mutation survived the first sweep:** ranking the "why" bars by signed push instead of by size. A targeted test now catches it.

If something fails as written, suspect a transcription slip or an environment difference first, and say which.

## Global Constraints

- **Nothing is retrained. Never run** `experiment.py`, `tune.py`, `report6.py`, `report6b.py`, `features.py` or any collection script.
- **The gitignored Phase 4 outputs are read-only.** Those are `data/models/`, `data/predictions/`, `data/features/` and `data/processed/`. Verify by hash (Task 1), never by `git status`.
- **Do not modify any phase's generated document or its generator:** `docs/phase*.md`, `report4.py`, `report6.py`, `report6b.py`.
- **Design docs live under `docs/design/`.**
- **The app hard-codes no result number.** No `\b0\.\d{3}\b` may appear in any `demo/*.py` or `demo/views/*.py`. Every number shown is computed at runtime from committed files, and the results chapter's numbers are tested against `writeup_claims.py`.
- **No phrasing from `writeup_claims.RETRACTED`** may appear in `demo/**/*.py`, `README.md` or `docs/REPORT.md`.
- **No author identity** (login, database id, name, email) goes into the extract.
- **Deployed imports are limited to** `demo/requirements.txt` (streamlit, pandas, pyarrow, altair) plus the standard library and the demo's own modules.
- **Pins:** `altair==6.3.0` is added to both requirements files. Streamlit stays at 1.65.0.
- **Chart colours come from `charts.PALETTE`, validated in both modes.** Don't change a colour without re-running the dataviz palette validator.
- **Voice:** project voice in `README.md`, with no first-person authorship claims and no mention of AI tooling.
- **Mutation checks are restored with `git checkout -- <file>`, never by rewriting the text,** and `git status --porcelain` must be clean afterwards.
- **Tests:** run with `python -m pytest tests -q` from the repo root, and the output must be pristine. **462 pass today.**
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

These are failure modes the spec implies but no single happy-path test exercises. Each is pinned by a test in the task that owns it.

1. **The presenter flips the model switch between picking and revealing a game round.** The reveal must still work, and the hit-rate line must follow the new model. → Task 4: `test_switching_the_model_mid_round_still_reveals_cleanly`.
2. **A repo where nobody was waiting on the default day.** Chapter 3 must still explain a PR, falling back to the repo's riskiest. → Task 4: `test_why_falls_back_to_the_repos_riskiest_pr_when_nobody_was_waiting`.
3. **Clicking empty space on the timeline, or deselecting.** No selection must mean no crash, and the current PR stays put. → Task 2: `test_selected_pr_id_reads_a_point_selection_event`.
4. **A game week with tied scores.** The exact hit rate must still match the tie-break that `model_pick` uses. → Task 2: `test_game_hit_rate_equals_brute_force_over_every_round_including_ties`.
5. **The deployed app updated while running**, with stale `triage`, `charts` or `chapters` modules cached. → Task 4: `test_the_app_uses_the_current_helpers_even_when_old_copies_are_cached`.

---

## File structure

| File | Responsibility |
|---|---|
| `build_demo_data.py` (modify) | The extract plus per-PR SHAP, base and feature values, and `features.json`; `check_logit` and `feature_list` |
| `demo/data/prs.parquet` (regenerate, committed), `demo/data/features.json` (create, committed) | The data the app reads |
| `demo/triage.py` (modify) | The shared formatter, wait buckets, timeline, selection, "why", importance, the game, the results |
| `demo/charts.py` (create) | Altair chart builders and the validated palette |
| `demo/chapters.py` (create) | The six chapters, the speaker notes, the game's state handling |
| `demo/views/{problem,watch,why,test_yourself,transfer,limits}.py` (create) | Two-line page files |
| `demo/app.py` (rewrite) | The shell: module reloads, cached data, sidebar, navigation |
| `demo/requirements.txt`, `requirements.txt` (modify) | Pin Altair |
| `tests/test_build_demo_data.py`, `tests/test_demo_extract.py`, `tests/test_demo_triage.py` (modify), `tests/test_demo_charts.py` (create), `tests/test_demo_app.py` (rewrite) | Spec §9 |
| `README.md`, `docs/design/specs/2026-10-03-phase7-demo-design.md` (modify) | Spec §8 |

---

### Task 1: The extract additions

**Files:**
- Modify: `build_demo_data.py`, `demo/triage.py` (adds `format_value` only), `tests/test_build_demo_data.py`, `tests/test_demo_extract.py`
- Regenerate: `demo/data/prs.parquet`
- Create: `demo/data/features.json`

**Interfaces:**
- Consumes: Phase 6's `attribution.explain` and `attribution.additivity_delta`; the gitignored Phase 4 inputs, read only.
- Produces:
  - `triage.format_value(value, dtype, rate=False) -> str`
  - `build_demo_data.BASE_COLUMNS` (the original 15), `COLUMNS` (128), `FEATURES_JSON = "features.json"`
  - `feature_list() -> list[dict]`, `explain(booster, X) -> (sv, ev)`, `check_logit(sv, ev, score, tol=1e-6)`
  - extract columns `base_a`, `base_b`, `shap_{a,b}__<feature>` (float32), `x__<feature>`
  - `features.json` entries `{feature, label, dtype, rate}`

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_build_demo_data.py` with:

```python
"""build_demo_data.py's pure parts: labels, value formatting, driver text, validation."""
import numpy as np
import pandas as pd
import pytest

import build_demo_data as bdd
import featuresets as fs


def test_every_full_feature_has_a_label():
    assert set(bdd.DRIVER_LABELS) == set(fs.FEATURE_SETS["FULL"])


@pytest.mark.parametrize("feature,value,text", [
    ("trailing_90d_slow_rate", 0.456, "0.46"),
    ("open_backlog_at_t", 1234, "1,234"),
    ("additions_at_open", 6604.6, "6,605"),
    ("has_codeowners", True, "yes"),
    ("is_first_pr_here", False, "no"),
    ("language_dominant", "Go", "Go"),
    ("author_account_age_days", np.nan, "missing"),
])
def test_format_value(feature, value, text):
    assert bdd.format_value(feature, value) == text


def test_drivers_are_the_three_largest_by_size_with_the_direction_of_the_push():
    x = pd.Series({"trailing_90d_slow_rate": 0.25, "has_codeowners": True,
                   "open_backlog_at_t": 41, "is_weekend": False})
    text = bdd.format_drivers(np.array([0.1, -0.5, 0.3, 0.0]), x)
    assert text == ("has CODEOWNERS = yes ↓ · open PRs in the repo = 41 ↑ · "
                    "repo's recent slow rate = 0.25 ↑")


def _valid():
    return pd.DataFrame({
        "repo": ["o/a", "o/b"], "pr_id": ["P1", "P2"], "number": [1, 2], "url": ["u1", "u2"],
        "title": ["t1", "t2"], "score_a": [0.1, 0.2], "fold_b": [0, 1], "score_b": [0.3, 0.4],
        "drivers_a": ["d", "d"], "drivers_b": ["d", "d"], "base_a": [0.1, 0.1], "base_b": [0.2, 0.3],
    })


def test_validate_accepts_a_complete_extract():
    bdd.validate(_valid(), {"o/a": 0, "o/b": 1})


@pytest.mark.parametrize("breakage", ["missing B score", "wrong fold", "duplicate pr_id", "missing base"])
def test_validate_refuses_a_broken_extract(breakage):
    df = _valid()
    if breakage == "missing B score":
        df.loc[0, "score_b"] = np.nan
    elif breakage == "wrong fold":
        df.loc[1, "fold_b"] = 0
    elif breakage == "missing base":
        df.loc[0, "base_b"] = np.nan
    else:
        df.loc[1, "pr_id"] = "P1"
    with pytest.raises(ValueError):
        bdd.validate(df, {"o/a": 0, "o/b": 1})


def test_feature_list_is_the_full_features_in_model_order_with_labels_and_rates():
    listed = bdd.feature_list()
    assert [f["feature"] for f in listed] == list(fs.FEATURE_SETS["FULL"])
    assert all(f["label"] == bdd.DRIVER_LABELS[f["feature"]] for f in listed)
    assert {f["feature"] for f in listed if f["rate"]} == bdd.RATES


def test_check_logit_accepts_consistent_values_and_refuses_a_gap():
    sv = np.array([[0.5, -0.25], [1.0, 0.0]])
    base = 0.1
    score = 1 / (1 + np.exp(-(base + sv.sum(axis=1))))
    bdd.check_logit(sv, base, score)
    with pytest.raises(ValueError):
        bdd.check_logit(sv, base + 0.01, score)
```

Replace `tests/test_demo_extract.py` with:

```python
"""The committed demo extract reproduces Phase 4 and keeps its promises (Phase 7 spec, section 9).

Runs from a plain clone: it reads only demo/data/prs.parquet and data/phase4_runs.json."""
import json
import re

import numpy as np
import pandas as pd
import pytest

import build_demo_data as bdd
import metrics
import splits

RUNS = bdd.ROOT / "data" / "phase4_runs.json"


@pytest.fixture(scope="module")
def prs():
    return pd.read_parquet(bdd.OUT)


@pytest.fixture(scope="module")
def runs():
    return {(r["scenario"], r["featureset"], r["fold"]): r
            for r in json.loads(RUNS.read_text(encoding="utf-8"))["runs"]}


def test_scenario_a_auc_pr_and_baseline_match_phase4(prs, runs):
    a = runs[("A", "FULL", 0)]
    y = prs["is_slow"].astype(int)
    assert metrics.auc_pr(y, prs["score_a"]) == pytest.approx(a["auc_pr"], abs=1e-12)
    assert metrics.auc_pr(y, prs["baseline_score"]) == pytest.approx(a["baseline_auc_pr"], abs=1e-12)


def test_scenario_a_p10_matches_phase4_in_the_extracts_row_order(prs, runs):
    """P@10 breaks ties with a seeded permutation over the row order. The model's scores rarely
    tie, but the baseline's barely vary within a repo, so its P@10 only reproduces if the
    extract kept the predictions file's row order exactly."""
    def p10(col):
        return metrics.precision_at_k(prs["is_slow"].to_numpy(dtype=int), prs[col].to_numpy(),
                                      prs["repo"].to_numpy(), k=10, seed=splits.SEED)[1]
    a = runs[("A", "FULL", 0)]
    assert p10("score_a") == pytest.approx(a["precision_at_10"], abs=1e-12)
    assert p10("baseline_score") == pytest.approx(a["baseline_p10"], abs=1e-12)


def test_each_b_fold_holds_exactly_its_phase4_test_repos(prs, runs):
    for k in range(5):
        assert set(prs.loc[prs["fold_b"] == k, "repo"]) == set(runs[("B", "FULL", k)]["test_repos"])


def test_shape_columns_and_completeness(prs):
    assert list(prs.columns) == bdd.COLUMNS
    assert len(prs) == 14_135 and prs["pr_id"].is_unique and prs["repo"].nunique() == 39
    assert prs[["score_a", "score_b", "number", "url", "title"]].notna().all().all()
    assert prs["created_at"].min() >= pd.Timestamp("2026-01-01", tz="UTC")
    assert prs["created_at"].max() < pd.Timestamp("2026-07-01", tz="UTC")


def test_every_row_has_three_drivers_per_model(prs):
    for col in ("drivers_a", "drivers_b"):
        parts = prs[col].str.split(" · ")
        assert parts.map(len).eq(3).all() and parts.map(lambda p: all(x.strip() for x in p)).all()


def test_the_slow_flag_agrees_with_the_first_review_time(prs):
    wait_h = (prs["first_review_at"] - prs["created_at"]).dt.total_seconds() / 3600
    assert ((prs["first_review_at"].isna() | (wait_h > 168)) == prs["is_slow"]).all()


def test_no_author_identity_is_stored(prs):
    """Feature names such as author_prior_slow_rate_here are fine; logins and ids are not."""
    assert not [c for c in prs.columns if re.search(r"login|database_id|author_id|author_name|email", c)]


@pytest.mark.parametrize("s", ["a", "b"])
def test_each_rows_shap_values_add_up_to_its_score(prs, s):
    margin = prs[f"base_{s}"] + prs[[f"shap_{s}__{c}" for c in bdd.FULL]].astype(float).sum(axis=1)
    logit = np.log(prs[f"score_{s}"] / (1 - prs[f"score_{s}"]))
    assert (margin - logit).abs().max() < 1e-4


@pytest.mark.parametrize("s", ["a", "b"])
def test_each_rows_drivers_text_is_recomputable_from_its_stored_values(prs, s):
    for _, row in prs.sample(300, random_state=1).iterrows():
        shap_row = np.array([row[f"shap_{s}__{c}"] for c in bdd.FULL], dtype=float)
        x_row = pd.Series({c: row[f"x__{c}"] for c in bdd.FULL})
        assert bdd.format_drivers(shap_row, x_row) == row[f"drivers_{s}"]


def test_features_json_is_what_the_build_writes():
    assert json.loads((bdd.OUT.parent / bdd.FEATURES_JSON).read_text(encoding="utf-8")) == bdd.feature_list()
```

- [ ] **Step 2: Run them and confirm they fail**

Run `python -m pytest tests/test_build_demo_data.py tests/test_demo_extract.py -q`. Expected: failures.
- **Build tests:** `AttributeError` for `feature_list` and `check_logit`, and `KeyError: 'base_a'` in the validate tests.
- **Extract tests:** the column list, additivity, drivers and `features.json` tests, because the committed extract has only the 15 original columns.

- [ ] **Step 3: Add the shared formatter to `demo/triage.py`, then rewrite `build_demo_data.py`**

Insert this function into `demo/triage.py` directly above `def awaiting_review(` (Task 2 later replaces the whole file with a version containing the identical function):

```python
def format_value(value, dtype: str, rate: bool = False) -> str:
    """Rates and shares to 2 decimals, counts as integers, booleans as yes/no. The one
    formatter: build_demo_data.py writes the drivers text with it, the app shows values with it."""
    if pd.isna(value):
        return "missing"
    if dtype == "bool":
        return "yes" if bool(value) else "no"
    if dtype == "str":
        return str(value)
    if rate:
        return f"{float(value):.2f}"
    return f"{int(round(float(value))):,}"


```

Replace `build_demo_data.py` with:

```python
"""Phase 7: build the demo's committed extract, demo/data/prs.parquet, and its feature list,
demo/data/features.json.

Runs locally only. It reads Phase 4's gitignored outputs (predictions, boosters, the feature
table, parsed PRs) and writes small committed files, so the deployed app needs no model:

    python build_demo_data.py

One row per Scenario A test PR, in the A predictions file's row order. Phase 4's P@10 breaks
ties with a seeded permutation over that order, so keeping it is what lets
tests/test_demo_extract.py reproduce Phase 4's numbers from the extract. Each row also carries
every feature's exact TreeSHAP value under both models, the explainer's base value and the
feature values themselves, so the app's "why" chart adds up to the score it shows. Author
identities are deliberately left out: the demo is about PRs, not people."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import attribution
import experiment as ex
import features
import featuresets as fs
import model
from demo import triage

ROOT = Path(__file__).parent
OUT = ROOT / "demo" / "data" / "prs.parquet"
FEATURES_JSON = "features.json"                     # written next to the extract
FULL = fs.FEATURE_SETS["FULL"]
ADDITIVITY_TOL = 1e-6
RATES = {"trailing_90d_slow_rate", "prior_merge_rate_here", "author_prior_slow_rate_here"}
BASE_COLUMNS = ["repo", "pr_id", "number", "url", "title", "created_at", "closed_at",
                "first_review_at", "is_slow", "score_a", "fold_b", "score_b", "baseline_score",
                "drivers_a", "drivers_b"]
COLUMNS = (BASE_COLUMNS + ["base_a", "base_b"] + [f"shap_a__{c}" for c in FULL]
           + [f"shap_b__{c}" for c in FULL] + [f"x__{c}" for c in FULL])

# A reader-facing name for every FULL feature (tests/test_build_demo_data.py checks coverage).
DRIVER_LABELS = {
    "created_hour_utc": "hour opened (UTC)",
    "created_dayofweek": "weekday opened (Mon=0)",
    "is_weekend": "opened at the weekend",
    "is_cross_repository": "from a fork",
    "author_account_age_days": "author's GitHub account age (days)",
    "body_len": "description length",
    "has_body": "has a description",
    "is_draft_at_open": "opened as a draft",
    "n_labels_at_open": "labels at open",
    "title_len_at_open": "title length",
    "base_is_default": "targets the default branch",
    "reviewer_requested_at_open": "reviewer requested at open",
    "n_reviewers_requested_at_open": "reviewers requested at open",
    "requested_team_at_open": "team review requested at open",
    "additions_at_open": "lines added at open",
    "deletions_at_open": "lines deleted at open",
    "n_commits_at_open": "commits at open",
    "open_backlog_at_t": "open PRs in the repo",
    "prs_opened_trailing_7d": "PRs opened in the repo, last 7 days",
    "trailing_90d_slow_rate": "repo's recent slow rate",
    "trailing_n": "PRs behind the repo's slow rate",
    "is_first_pr_here": "author's first PR here",
    "n_prior_prs_here": "author's earlier PRs here",
    "n_prior_merged_here": "author's merged PRs here",
    "prior_merge_rate_here": "author's merge rate here",
    "days_since_first_pr_here": "days since author's first PR here",
    "author_prior_slow_rate_here": "author's past slow rate here",
    "author_prior_n": "PRs behind author's slow rate",
    "n_assignable_users": "maintainers (assignable users)",
    "n_mentionable_users": "community size (mentionable users)",
    "owner_is_org": "owned by an organisation",
    "has_codeowners": "has CODEOWNERS",
    "has_pr_template": "has a PR template",
    "has_contributing": "has CONTRIBUTING",
    "n_ci_workflows": "CI workflows",
    "language_dominant": "main language",
    "repo_age_days_at_open": "repo age (days)",
}


def format_value(feature: str, value) -> str:
    """Rates and shares to 2 decimals, counts as integers, booleans as yes/no (the shared
    formatter in demo/triage.py, so the app shows values exactly as the drivers text does)."""
    return triage.format_value(value, features.COLUMN_SPEC[feature]["dtype"], feature in RATES)


def format_drivers(shap_row: np.ndarray, x_row: pd.Series, k: int = 3) -> str:
    """The k features with the largest |SHAP|, largest first. The arrow is the direction the
    feature pushed the raw margin: up means towards 'slow'."""
    top = np.argsort(-np.abs(shap_row), kind="stable")[:k]
    return " · ".join(
        f"{DRIVER_LABELS[x_row.index[i]]} = {format_value(x_row.index[i], x_row.iloc[i])} "
        f"{'↑' if shap_row[i] > 0 else '↓'}" for i in top)


def feature_list() -> list[dict]:
    """What demo/data/features.json holds: the FULL features in model order."""
    return [{"feature": c, "label": DRIVER_LABELS[c], "dtype": features.COLUMN_SPEC[c]["dtype"],
             "rate": c in RATES} for c in FULL]


def explain(booster, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Exact TreeSHAP (Phase 6's attribution.explain), refusing any additivity failure."""
    sv, ev = attribution.explain(booster, X)
    delta = attribution.additivity_delta(booster, X, sv, ev)
    if delta > ADDITIVITY_TOL:
        raise RuntimeError(f"SHAP additivity failed: max delta {delta:.3g} > {ADDITIVITY_TOL}")
    return sv, ev


def check_logit(sv: np.ndarray, ev, score: np.ndarray, tol: float = ADDITIVITY_TOL) -> None:
    """base + sum(SHAP) must be the logit of the score the extract stores, row by row."""
    score = np.asarray(score, dtype=float)
    gap = np.max(np.abs(np.asarray(ev) + sv.sum(axis=1) - np.log(score / (1 - score))))
    if gap > tol:
        raise ValueError(f"base + SHAP differs from logit(score) by up to {gap:.3g} > {tol}")


def validate(out: pd.DataFrame, fold_of: dict[str, int]) -> None:
    """The build checks of spec section 4, apart from additivity (checked in explain and
    check_logit)."""
    missing = sorted(set(FULL) - set(DRIVER_LABELS))
    if missing:
        raise ValueError(f"no DRIVER_LABELS entry for {missing}")
    for col in ("number", "url", "title", "score_a", "fold_b", "score_b", "drivers_a", "drivers_b",
                "base_a", "base_b"):
        if out[col].isna().any():
            raise ValueError(f"{int(out[col].isna().sum())} rows have no {col}")
    wrong = out["fold_b"] != out["repo"].map(fold_of)
    if wrong.any():
        raise ValueError(f"{int(wrong.sum())} rows are not in the B fold that held their repo out")
    if not out["pr_id"].is_unique:
        raise ValueError("duplicate pr_id")


def _unique_index(df: pd.DataFrame, what: str) -> pd.DataFrame:
    if not df["pr_id"].is_unique:
        raise ValueError(f"duplicate pr_id in {what}")
    return df.set_index("pr_id")


def build(root: Path = ROOT) -> pd.DataFrame:
    a = pd.read_parquet(root / "data" / "predictions" / "A_FULL_fold0.parquet")
    runs = json.loads((root / "data" / "phase4_runs.json").read_text(encoding="utf-8"))["runs"]
    fold_of = {repo: r["fold"] for r in runs
               if r["scenario"] == "B" and r["featureset"] == "FULL" for repo in r["test_repos"]}
    table = ex.load_table(root / "data" / "features" / "features.parquet").set_index("pr_id")
    tier2 = _unique_index(pd.concat(
        pd.read_parquet(root / "data" / "processed" / repo.replace("/", "__") / "pr_tier2.parquet",
                        columns=["pr_id", "number", "url", "title_current", "closed_at"])
        for repo in sorted(a["repo"].unique())), "pr_tier2")
    b = _unique_index(pd.concat(
        pd.read_parquet(root / "data" / "predictions" / f"B_FULL_fold{k}.parquet",
                        columns=["pr_id", "p_hat"]).assign(fold_b=k)
        for k in sorted(set(fold_of.values()))), "B predictions")

    # .map keeps the A file's row order exactly (spec section 4).
    ids = a["pr_id"]
    out = pd.DataFrame({
        "repo": a["repo"], "pr_id": ids,
        "number": ids.map(tier2["number"]), "url": ids.map(tier2["url"]),
        "title": ids.map(tier2["title_current"]),
        "created_at": a["created_at"], "closed_at": ids.map(tier2["closed_at"]),
        "first_review_at": a["created_at"] + pd.to_timedelta(ids.map(table["wait_h"]), unit="h"),
        "is_slow": a["is_slow"], "score_a": a["p_hat"],
        "fold_b": ids.map(b["fold_b"]), "score_b": ids.map(b["p_hat"]),
        "baseline_score": a["baseline_score"],
    })
    if not ids.isin(table.index).all():
        raise ValueError("A test rows missing from the feature table")

    X = table.loc[ids, FULL]
    sva, eva = explain(model.load(root / "data" / "models" / "A_FULL_fold0.txt"), X)
    check_logit(sva, eva, out["score_a"].to_numpy())
    svb, evb = np.full(sva.shape, np.nan), np.full(len(out), np.nan)
    for k in sorted(set(fold_of.values())):
        rows = (out["fold_b"] == k).to_numpy()
        sv, ev = explain(model.load(root / "data" / "models" / f"B_FULL_fold{k}.txt"), X[rows])
        check_logit(sv, ev, out.loc[rows, "score_b"].to_numpy())
        svb[rows], evb[rows] = sv, ev
    out["drivers_a"] = [format_drivers(sva[i], X.iloc[i]) for i in range(len(X))]
    out["drivers_b"] = [format_drivers(svb[i], X.iloc[i]) for i in range(len(X))]
    out["base_a"], out["base_b"] = eva, evb

    values = X.reset_index(drop=True).add_prefix("x__")
    values["x__language_dominant"] = values["x__language_dominant"].astype(str)
    out = pd.concat([out,
                     pd.DataFrame(sva.astype("float32"), columns=[f"shap_a__{c}" for c in FULL]),
                     pd.DataFrame(svb.astype("float32"), columns=[f"shap_b__{c}" for c in FULL]),
                     values], axis=1)
    out["number"] = out["number"].astype("int64")
    out["fold_b"] = out["fold_b"].astype("int64")
    validate(out, fold_of)
    return out[COLUMNS]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=ROOT, help="repo holding the gitignored inputs")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    out = build(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    (args.out.parent / FEATURES_JSON).write_text(json.dumps(feature_list(), indent=2) + "\n",
                                                 encoding="utf-8", newline="\n")
    print(f"wrote {args.out}: {len(out):,} PRs, {out['repo'].nunique()} repos, "
          f"{args.out.stat().st_size / 1e6:.1f} MB, and {FEATURES_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Rebuild, verifying that the inputs are untouched and the output is the dry run's**

```bash
sha256sum data/predictions/A_FULL_fold0.parquet data/predictions/B_FULL_fold*.parquet data/models/A_FULL_fold0.txt data/models/B_FULL_fold*.txt data/features/features.parquet > "$TMP/story_inputs.sha256"
python build_demo_data.py
sha256sum -c "$TMP/story_inputs.sha256"
sha256sum demo/data/prs.parquet demo/data/features.json
```

Expected:
- the build prints `wrote ...prs.parquet: 14,135 PRs, 39 repos, 8.0 MB, and features.json` after about a minute;
- every input reports `OK`;
- `prs.parquet` sha256 is `0c521f22d1b68021d043323dddb54927cb67dc490ddd07438b8a0c6d83efa10a`;
- `features.json` sha256 is `f64f5a84d3bfc0e1a60ea286cd3cc09664d0dc30790c28b054adacbbb95872e5`.

A different hash means the build is not reproducing the dry run: stop and report it.

- [ ] **Step 5: Run the tests and the suite**

Run `python -m pytest tests/test_build_demo_data.py tests/test_demo_extract.py -q` and expect **28 passed** (16 + 12). Then run `python -m pytest tests -q` and expect **470 passed**.

- [ ] **Step 6: Commit**

```bash
git add build_demo_data.py demo/triage.py demo/data/prs.parquet demo/data/features.json tests/test_build_demo_data.py tests/test_demo_extract.py
git commit -m "feat(demo): extract carries per-PR SHAP, base and feature values, plus features.json

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 7: Mutation check (after the commit, restoring with git)**

For each mutation: apply it, run the named test and confirm it FAILS, then run `git checkout -- <file>` and confirm `git status --porcelain` is clean.
- `build_demo_data.py`: `    if gap > tol:` → `    if gap < tol:` → `tests/test_build_demo_data.py::test_check_logit_accepts_consistent_values_and_refuses_a_gap`
- `demo/triage.py`: `return f"{float(value):.2f}"` → `return f"{float(value):.3f}"` → `tests/test_demo_extract.py::test_each_rows_drivers_text_is_recomputable_from_its_stored_values`

---

### Task 2: The chapters' logic

**Files:**
- Modify: `demo/triage.py` (replaced whole), `tests/test_demo_triage.py` (replaced whole)

**Interfaces:**
- Consumes: Task 1's extract columns and `features.json`; the committed `data/phase4_runs.json` and `data/phase6_importance_{A,B}.csv`.
- Produces, in `demo/triage.py`:
  - **Constants:**
    - `FEATURES`, `RUNS`, `IMPORTANCE` (a template with `{}`)
    - `GAME_SIZE = 4`
    - `WAIT_BUCKETS` (6 names), `STALLED_BUCKETS`
    - `SCENARIO_LABELS = {"A": "repos seen in training", "B": "repos never seen"}`
    - `SERIES_LABELS`
  - **Loading and helpers:**
    - `load_features(path=None) -> list[dict]`
    - `format_value(value, dtype, rate=False) -> str`, `logit(p) -> float`
  - **Chapter logic:**
    - `wait_buckets(prs) -> DataFrame[bucket, count, share, stalled]`
    - `timeline(prs, repo, scenario, at) -> DataFrame[pr_id, number, title, created_at, risk, stalled, waiting, outcome]`
    - `selected_pr_id(event) -> str | None`
    - `why(pr, scenario, features, k=8) -> (DataFrame[label, value, push], base)`
    - `importance(scenario, features, template=None) -> DataFrame[feature, label, share]`
  - **The game:**
    - `week(created_at)`, `eligible_weeks(prs) -> list[(repo, week)]`
    - `deal(prs, rng: random.Random) -> DataFrame` (4 rows)
    - `model_pick(round_, scenario) -> str`, `game_hit_rate(prs, scenario) -> float`
  - **Results:** `results(path=None) -> {"folds", "means", "p10", "text"}`. The `text` keys are `a_p10`, `a_baseline_p10`, `a_random_p10`, `b_full_vs_baseline`, `b_nlr_vs_baseline`.
  - Every existing function and constant is unchanged.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_demo_triage.py` with:

```python
"""The demo's triage logic on small synthetic frames (Phase 7 spec, section 5)."""
from datetime import date

import pandas as pd
import pytest

from demo import triage

T = pd.Timestamp("2026-03-10", tz="UTC")
H = pd.Timedelta(hours=1)


def _prs(*rows, repo="o/r"):
    """Each row: (number, created, closed, first_review, score_a, is_slow); times are offsets
    from T, and None means never closed / never reviewed."""
    recs = []
    for number, created, closed, review, score, slow in rows:
        recs.append({
            "repo": repo, "pr_id": f"{repo}#{number}", "number": number,
            "url": f"https://github.com/{repo}/pull/{number}", "title": f"PR {number}",
            "created_at": T + created,
            "closed_at": T + closed if closed is not None else pd.NaT,
            "first_review_at": T + review if review is not None else pd.NaT,
            "is_slow": slow, "score_a": score, "score_b": 1 - score,
            "drivers_a": f"a{number}", "drivers_b": f"b{number}",
        })
    df = pd.DataFrame(recs)
    for col in ("created_at", "closed_at", "first_review_at"):    # tz-aware, as in the extract
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def _numbers(df, scenario="A"):
    return [int(u.rsplit("/", 1)[1]) for u in triage.ranked(df, "o/r", T, scenario)["url"]]


def test_the_moment_is_midnight_utc():
    assert triage.moment(date(2026, 3, 10)) == T


def test_a_pr_opened_after_or_exactly_at_the_moment_is_not_waiting_yet():
    df = _prs((1, H, None, None, 0.5, False), (2, 0 * H, None, None, 0.5, False))
    assert _numbers(df) == []


def test_a_pr_opened_a_minute_before_midnight_is_waiting():
    df = _prs((1, -pd.Timedelta(minutes=1), None, None, 0.5, False))
    assert _numbers(df) == [1]


def test_a_pr_closed_before_or_exactly_at_the_moment_drops_out():
    df = _prs((1, -9 * H, -H, None, 0.9, True), (2, -9 * H, 0 * H, None, 0.8, True),
              (3, -9 * H, H, None, 0.7, True))
    assert _numbers(df) == [3]


def test_a_pr_reviewed_before_or_exactly_at_the_moment_drops_out():
    df = _prs((1, -9 * H, None, -H, 0.9, False), (2, -9 * H, None, 0 * H, 0.8, False),
              (3, -9 * H, None, H, 0.7, False))
    assert _numbers(df) == [3]


def test_a_pr_never_reviewed_and_never_closed_stays_waiting():
    df = _prs((1, -900 * H, None, None, 0.5, True))
    assert _numbers(df) == [1]


def test_ranking_is_by_risk_then_age_then_number():
    df = _prs((5, -2 * H, None, None, 0.4, False), (4, -9 * H, None, None, 0.4, False),
              (3, -9 * H, None, None, 0.4, False), (9, -H, None, None, 0.8, False))
    assert _numbers(df) == [9, 3, 4, 5]


def test_the_unseen_model_ranks_by_its_own_score_and_drivers():
    df = _prs((1, -H, None, None, 0.9, False), (2, -H, None, None, 0.1, False))
    listing = triage.ranked(df, "o/r", T, "B")
    assert _numbers(df, "B") == [2, 1]
    assert listing["risk"].tolist() == pytest.approx([0.9, 0.1])
    assert listing["drivers"].tolist() == ["b2", "b1"]


def test_days_waited_and_rank():
    listing = triage.ranked(_prs((1, -36 * H, None, None, 0.5, False)), "o/r", T, "A")
    assert listing["days_waited"].tolist() == [1.5] and listing["rank"].tolist() == [1]


def test_outcome_says_when_the_review_came_and_whether_it_stalled():
    df = _prs((1, -H, None, 11 * H, 0.9, False), (2, -H, None, 239 * H, 0.8, True),
              (3, -H, None, None, 0.7, True))
    assert triage.ranked(df, "o/r", T, "A")["outcome"].tolist() == [
        "reviewed after 0.5 days", "reviewed after 10.0 days, stalled", "never reviewed, stalled"]


def test_outcome_says_when_a_pr_was_closed_without_a_review():
    """Most never-reviewed PRs were closed within days; 'never reviewed' alone would hide that."""
    df = _prs((1, -H, 64 * H, None, 0.9, True))
    assert triage.ranked(df, "o/r", T, "A")["outcome"].tolist() == [
        "closed after 2.7 days without a review, stalled"]


def test_tally_counts_the_stalled_among_the_top_three():
    df = _prs(*[(n, -H, None, None, 1 - n / 10, n % 2 == 1) for n in range(1, 6)])
    assert triage.tally(triage.ranked(df, "o/r", T, "A")) == (2, 3)


def test_tally_on_a_short_list_and_an_empty_one():
    one = triage.ranked(_prs((1, -H, None, None, 0.5, True)), "o/r", T, "A")
    none = triage.ranked(_prs((1, H, None, None, 0.5, True)), "o/r", T, "A")
    assert triage.tally(one) == (1, 1) and triage.tally(none) == (0, 0)


def test_default_repo_has_the_most_waiting_with_ties_alphabetical():
    waiting = (1, -H, None, None, 0.5, False)
    df = pd.concat([_prs(waiting, repo="z/z"), _prs(waiting, (2, -H, None, None, 0.5, False), repo="m/m"),
                    _prs(waiting, (2, -H, None, None, 0.5, False), repo="b/b")], ignore_index=True)
    assert triage.default_repo(df, T) == "b/b"


def test_default_repo_when_nobody_is_waiting_is_the_first_alphabetically():
    df = pd.concat([_prs((1, H, None, None, 0.5, False), repo="z/z"),
                    _prs((1, H, None, None, 0.5, False), repo="c/c")], ignore_index=True)
    assert triage.default_repo(df, T) == "c/c"


def test_load_works_from_any_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert len(triage.load()) > 0


# ---------------------------------------------------------------------------
# looking up one PR
# ---------------------------------------------------------------------------

def test_parse_accepts_a_pr_link_with_or_without_extra_parts():
    for text in ("https://github.com/o/r/pull/12", "https://github.com/o/r/pull/12/files",
                 "github.com/o/r/pull/12?x=1", "  https://github.com/o/r/pull/12  "):
        assert triage.parse_pr_ref(text, "d/d") == ("o/r", 12), text


def test_parse_accepts_owner_repo_hash_number():
    assert triage.parse_pr_ref("o/r#12", "d/d") == ("o/r", 12)


def test_parse_resolves_a_bare_number_in_the_default_repo():
    assert triage.parse_pr_ref("12", "d/d") == ("d/d", 12)
    assert triage.parse_pr_ref(" #12 ", "d/d") == ("d/d", 12)


def test_parse_rejects_anything_else():
    for text in ("", "   ", "hello", "o/r", "https://github.com/o/r/issues/12", "#"):
        assert triage.parse_pr_ref(text, "d/d") is None, text


def test_find_pr_matches_the_repo_case_insensitively_and_misses_cleanly():
    df = _prs((1, -H, None, None, 0.5, False))
    assert triage.find_pr(df, "O/R", 1)["number"].tolist() == [1]
    assert triage.find_pr(df, "o/r", 99).empty and triage.find_pr(df, "x/y", 1).empty


def test_rank_in_repo_is_the_share_of_the_repos_other_prs_scored_lower():
    df = pd.concat([_prs((1, -H, None, None, 0.9, False), (2, -H, None, None, 0.5, False),
                         (3, -H, None, None, 0.5, False), (4, -H, None, None, 0.1, False)),
                    _prs((1, -H, None, None, 0.0, False), repo="x/y")], ignore_index=True)
    rank = lambda n, sc="A": triage.rank_in_repo(df, triage.find_pr(df, "o/r", n), sc)
    assert rank(1) == (1.0, 3)                       # the riskiest: higher than all three others
    assert rank(4) == (0.0, 3)                       # the least risky
    assert rank(2) == (pytest.approx(1 / 3), 3)      # a tie does not count as lower
    assert rank(4, "B") == (1.0, 3)                  # the unseen model's own score (1 - score_a here)


def test_random_pr_is_one_replayed_row_and_reproducible_with_a_seed():
    df = _prs(*[(n, -H, None, None, n / 10, False) for n in range(1, 6)])
    a, b = triage.random_pr(df, seed=3), triage.random_pr(df, seed=3)
    assert len(a) == 1 and a["pr_id"].tolist() == b["pr_id"].tolist()


def test_md_escape_keeps_a_title_from_turning_into_markdown():
    assert triage.md_escape("fix *bold* [x](y) `code` #1") == r"fix \*bold\* \[x\]\(y\) \`code\` \#1"


# ---------------------------------------------------------------------------
# demo story: shared formatter, chapters' logic, the game, the results
# ---------------------------------------------------------------------------

import itertools
import math
import random

import writeup_claims as wc


def test_format_value_follows_dtype_and_rate():
    assert triage.format_value(0.456, "float", rate=True) == "0.46"
    assert triage.format_value(6604.6, "float") == "6,605"
    assert triage.format_value(True, "bool") == "yes" and triage.format_value(False, "bool") == "no"
    assert triage.format_value("Go", "str") == "Go"
    assert triage.format_value(float("nan"), "float") == "missing"


def test_wait_buckets_cover_every_pr_and_split_the_unreviewed():
    # each PR opens an hour before T, so a review at T + x waits x + 1 hours
    df = _prs((1, -H, None, -0.5 * H, 0.5, False), (2, -H, None, 4 * H, 0.5, False),
              (3, -H, None, 29 * H, 0.5, False), (4, -H, None, 199 * H, 0.5, True),
              (5, -H, 30 * H, None, 0.5, True), (6, -H, None, None, 0.5, True))
    b = triage.wait_buckets(df)
    assert b["bucket"].tolist() == list(triage.WAIT_BUCKETS)
    assert b["count"].tolist() == [1, 1, 1, 1, 1, 1] and b["share"].sum() == pytest.approx(1)


def test_the_stalled_wait_buckets_are_exactly_the_stalled_prs():
    prs = triage.load()
    b = triage.wait_buckets(prs)
    assert b["count"].sum() == len(prs)
    assert b.loc[b["stalled"], "count"].sum() == prs["is_slow"].sum()


def test_timeline_marks_waiting_with_the_triage_lists_rule():
    df = _prs((1, -9 * H, -H, None, 0.9, True), (2, -9 * H, None, None, 0.5, True), (3, H, None, None, 0.1, False))
    frame = triage.timeline(df, "o/r", "A", T)
    assert frame["waiting"].tolist() == [False, True, False]
    assert frame["risk"].tolist() == [0.9, 0.5, 0.1]


def test_selected_pr_id_reads_a_point_selection_event():
    assert triage.selected_pr_id({"selection": {"pick": [{"pr_id": "P7"}]}}) == "P7"
    for empty in ({"selection": {"pick": []}}, {"selection": {}}, {}, None):
        assert triage.selected_pr_id(empty) is None


@pytest.mark.parametrize("scenario", ["A", "B"])
def test_why_rows_and_base_add_up_to_the_models_logit(scenario):
    prs, features = triage.load(), triage.load_features()
    for _, row in prs.sample(200, random_state=0).iterrows():
        pr = prs[prs["pr_id"] == row["pr_id"]]
        rows, base = triage.why(pr, scenario, features)
        assert len(rows) == 9 and rows["label"].iloc[-1] == "all other 29 features"
        assert rows["push"].iloc[:-1].abs().is_monotonic_decreasing
        assert base + rows["push"].sum() == pytest.approx(triage.logit(row[triage.SCORES[scenario]]), abs=1e-4)


def test_importance_is_phase6s_shares_largest_first_with_labels():
    imp = triage.importance("A", triage.load_features())
    assert imp["share"].is_monotonic_decreasing and imp["share"].sum() == pytest.approx(1)
    assert imp["label"].notna().all() and imp["label"].iloc[0] == "author's past slow rate here"


def _weeks(*groups):
    """Synthetic game data: each group is (repo, week offset, [(number, score_a, is_slow), ...])."""
    rows = []
    for repo, wk, prs in groups:
        for number, score, slow in prs:
            rows.append((number, -H + pd.Timedelta(weeks=wk), None, None, score, slow, repo))
    frames = [_prs(*[r[:6] for r in rows if r[6] == repo], repo=repo) for repo in dict.fromkeys(r[6] for r in rows)]
    return pd.concat(frames, ignore_index=True)


def _brute_force_hit_rate(prs, scenario):
    rates = []
    for key in triage.eligible_weeks(prs):
        g = prs[(prs["repo"] == key[0]) & (triage.week(prs["created_at"]) == key[1])]
        per_stalled = []
        for _, s in g[g["is_slow"]].iterrows():
            combos = list(itertools.combinations(g[~g["is_slow"]].index, triage.GAME_SIZE - 1))
            wins = [triage.model_pick(pd.concat([g.loc[[s.name]], g.loc[list(c)]]), scenario) == s["pr_id"] for c in combos]
            per_stalled.append(sum(wins) / len(wins))
        rates.append(sum(per_stalled) / len(per_stalled))
    return sum(rates) / len(rates)


def test_game_hit_rate_equals_brute_force_over_every_round_including_ties():
    df = _weeks(("o/r", 0, [(1, 0.9, True), (2, 0.5, True), (3, 0.7, False), (4, 0.5, False), (5, 0.2, False), (6, 0.95, False)]),
                ("o/r", 1, [(7, 0.4, True), (8, 0.4, False), (9, 0.1, False), (10, 0.3, False)]),
                ("o/r", 2, [(11, 0.8, True), (12, 0.1, False)]))           # too few non-stalled: not eligible
    assert len(triage.eligible_weeks(df)) == 2
    assert triage.game_hit_rate(df, "A") == pytest.approx(_brute_force_hit_rate(df, "A"))


def test_model_pick_breaks_ties_by_the_lower_number():
    df = _prs((9, -H, None, None, 0.5, False), (3, -H, None, None, 0.5, True), (5, -H, None, None, 0.2, False))
    assert triage.model_pick(df, "A") == "o/r#3"


def test_every_dealt_round_is_four_prs_from_one_week_with_exactly_one_stalled():
    prs = triage.load()
    rng = random.Random(7)
    for _ in range(40):
        r = triage.deal(prs, rng)
        assert len(r) == 4 and r["pr_id"].is_unique and r["repo"].nunique() == 1
        assert triage.week(r["created_at"]).nunique() == 1 and r["is_slow"].sum() == 1


def test_deal_is_reproducible_with_a_seed():
    prs = triage.load()
    a, b = triage.deal(prs, random.Random(3)), triage.deal(prs, random.Random(3))
    assert a["pr_id"].tolist() == b["pr_id"].tolist()


def test_results_text_matches_the_readmes_registered_claims():
    text = triage.results()["text"]
    claims = {c.name: c.expected for c in wc.CLAIMS}
    assert text["a_p10"] == claims["a_p10"]
    assert claims["a_baseline_p10"] == f"baseline scores {text['a_baseline_p10']}"
    assert claims["a_base_rate"] == f"base rate of {text['a_random_p10']}"
    assert text["b_full_vs_baseline"] == claims["b_full_vs_baseline"]
    assert text["b_nlr_vs_baseline"] == claims["b_nlr_vs_baseline"]


def test_results_keep_each_fold_and_one_mean_per_bar():
    res = triage.results()
    assert len(res["means"]) == 6
    assert res["folds"].groupby("scenario")["fold"].nunique().to_dict() == {"repos never seen": 5, "repos seen in training": 1}
    assert res["p10"]["label"].tolist()[0] == res["text"]["a_p10"]


def test_why_ranks_by_size_so_a_large_push_down_comes_first():
    """Sorting by signed push would drop a big push towards 'fine' off the chart."""
    prs, features = triage.load(), triage.load_features()
    pr = prs.iloc[[0]].copy()
    pr[f"shap_a__{features[5]['feature']}"] = -5.0
    rows, _ = triage.why(pr, "A", features)
    assert rows["label"].iloc[0] == features[5]["label"] and rows["push"].iloc[0] == -5.0
```

- [ ] **Step 2: Run them and confirm they fail**

Run `python -m pytest tests/test_demo_triage.py -q`. Expected: the new tests fail with `AttributeError` (no `wait_buckets`, `timeline`, `why`, …). The 24 existing tests still pass.

- [ ] **Step 3: Implement**

Replace `demo/triage.py` with:

```python
"""The demo's logic, kept free of Streamlit so it can be tested directly.

Everything the app shows is computed here, at runtime, from committed files: the extract
demo/data/prs.parquet and its feature list demo/data/features.json (both built by
build_demo_data.py), and Phase 4's data/phase4_runs.json and Phase 6's
data/phase6_importance_{A,B}.csv. Imports pandas and the standard library only: the deployed
app installs demo/requirements.txt, not the project's full requirements."""
from __future__ import annotations

import json
import math
import random
import re
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data" / "prs.parquet"
FEATURES = HERE / "data" / "features.json"
RUNS = HERE.parent / "data" / "phase4_runs.json"
IMPORTANCE = str(HERE.parent / "data" / "phase6_importance_{}.csv")
SCORES = {"A": "score_a", "B": "score_b"}
DRIVERS = {"A": "drivers_a", "B": "drivers_b"}
TOP_K = 3
FIRST_DAY, LAST_DAY, DEFAULT_DAY = date(2026, 1, 2), date(2026, 6, 30), date(2026, 4, 1)
EARLY_JANUARY = date(2026, 1, 14)      # until here, few scored PRs can have been waiting yet
PR_LINK = re.compile(r"github\.com/([^/\s]+/[^/\s]+)/pull/(\d+)")
PR_REF = re.compile(r"^([^/\s#]+/[^/\s#]+)#(\d+)$")
GAME_SIZE = 4
WAIT_BUCKETS = ("within an hour", "1 hour to 1 day", "1 to 7 days", "after more than 7 days",
                "closed without a review", "never reviewed, still open")
STALLED_BUCKETS = WAIT_BUCKETS[3:]
SCENARIO_LABELS = {"A": "repos seen in training", "B": "repos never seen"}
SERIES_LABELS = {"FULL": "model, all features",
                 "NO_LABEL_REPLAY": "model, without the repo's slow-rate history",
                 "baseline": "trailing-rate baseline"}


def load(path: Path | None = None) -> pd.DataFrame:
    # resolved from this file, not the working directory, so `streamlit run demo/app.py`
    # works from anywhere
    return pd.read_parquet(path or DATA)


def load_features(path: Path | None = None) -> list[dict]:
    """The 37 model features in model order: feature, label, dtype, rate."""
    return json.loads(Path(path or FEATURES).read_text(encoding="utf-8"))


def repos(prs: pd.DataFrame) -> list[str]:
    return sorted(prs["repo"].unique())


def moment(day: date) -> pd.Timestamp:
    """The chosen day at 00:00 UTC: a maintainer opening the dashboard at the start of the day."""
    return pd.Timestamp(day, tz="UTC")


def format_value(value, dtype: str, rate: bool = False) -> str:
    """Rates and shares to 2 decimals, counts as integers, booleans as yes/no. The one
    formatter: build_demo_data.py writes the drivers text with it, the app shows values with it."""
    if pd.isna(value):
        return "missing"
    if dtype == "bool":
        return "yes" if bool(value) else "no"
    if dtype == "str":
        return str(value)
    if rate:
        return f"{float(value):.2f}"
    return f"{int(round(float(value))):,}"


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def awaiting_review(prs: pd.DataFrame, at: pd.Timestamp) -> pd.DataFrame:
    """PRs opened before `at`, still open, and not yet reviewed. A PR reviewed or closed at
    exactly `at` is no longer waiting."""
    still_open = prs["closed_at"].isna() | (prs["closed_at"] > at)
    unreviewed = prs["first_review_at"].isna() | (prs["first_review_at"] > at)
    return prs[(prs["created_at"] < at) & still_open & unreviewed]


def outcome(rows: pd.DataFrame) -> pd.Series:
    """What actually happened: when the first review came, or that the PR was closed without
    one (most never-reviewed PRs were), and whether the PR stalled."""
    def days(end: pd.Series) -> pd.Series:
        return (end - rows["created_at"]).dt.total_seconds() / 86400

    def text(reviewed: float, closed: float) -> str:
        if not pd.isna(reviewed):
            return f"reviewed after {reviewed:.1f} days"
        if not pd.isna(closed):
            return f"closed after {closed:.1f} days without a review"
        return "never reviewed"

    # built row by row: an empty selection must still give an (empty) column of strings
    return pd.Series([text(r, c) + (", stalled" if slow else "") for r, c, slow
                      in zip(days(rows["first_review_at"]), days(rows["closed_at"]), rows["is_slow"])],
                     index=rows.index, dtype=object)


def ranked(prs: pd.DataFrame, repo: str, at: pd.Timestamp, scenario: str) -> pd.DataFrame:
    """The triage list: one repo's PRs awaiting review at `at`, highest risk first. Ties go to
    the older PR, then the lower number, so the order is deterministic."""
    score = SCORES[scenario]
    rows = awaiting_review(prs[prs["repo"] == repo], at)
    rows = rows.sort_values([score, "created_at", "number"], ascending=[False, True, True],
                            kind="mergesort")
    return pd.DataFrame({
        "rank": range(1, len(rows) + 1),
        "risk": rows[score].to_numpy(),
        "url": rows["url"].to_numpy(),
        "title": rows["title"].to_numpy(),
        "days_waited": ((at - rows["created_at"]).dt.total_seconds() / 86400).to_numpy(),
        "drivers": rows[DRIVERS[scenario]].to_numpy(),
        "outcome": outcome(rows).to_numpy(),
        "stalled": rows["is_slow"].to_numpy(),
    })


def tally(listing: pd.DataFrame, k: int = TOP_K) -> tuple[int, int]:
    """(how many of the top k stalled, k), with k cut to the list's length."""
    top = listing.head(k)
    return int(top["stalled"].sum()), len(top)


def default_repo(prs: pd.DataFrame, at: pd.Timestamp) -> str:
    """The repo with the most PRs awaiting review at `at`; ties go alphabetically."""
    counts = awaiting_review(prs, at).groupby("repo").size()
    if counts.empty:
        return repos(prs)[0]
    return min(counts.index, key=lambda r: (-counts[r], r))


def parse_pr_ref(text: str, default_repo: str) -> tuple[str, int] | None:
    """A PR link, `owner/repo#N`, or a bare number (looked up in default_repo); else None."""
    text = text.strip()
    link = PR_LINK.search(text) or PR_REF.match(text)
    if link:
        return link.group(1), int(link.group(2))
    bare = text.lstrip("#")
    return (default_repo, int(bare)) if bare.isdigit() else None


def find_pr(prs: pd.DataFrame, repo: str, number: int) -> pd.DataFrame:
    """The replayed PR as a one-row frame (empty if it is not in the replay). GitHub treats
    owner/repo case-insensitively, so this does too."""
    return prs[(prs["repo"].str.lower() == repo.lower()) & (prs["number"] == number)]


def rank_in_repo(prs: pd.DataFrame, pr: pd.DataFrame, scenario: str) -> tuple[float, int]:
    """(share of the repo's other replayed PRs that scored strictly lower, how many others)."""
    score = SCORES[scenario]
    row = pr.iloc[0]
    others = prs.loc[(prs["repo"] == row["repo"]) & (prs["pr_id"] != row["pr_id"]), score]
    return (float((others < row[score]).mean()) if len(others) else 0.0), len(others)


def random_pr(prs: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    """One replayed PR at random, as a one-row frame."""
    return prs.sample(1, random_state=seed)


MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-!|>~<])")


def md_escape(text: str) -> str:
    """Backslash-escape Markdown's special characters, so a PR title renders exactly as typed."""
    return MD_SPECIAL.sub(r"\\\1", text)


# ---------------------------------------------------------------------------
# chapter 1: how long PRs waited
# ---------------------------------------------------------------------------

def wait_buckets(prs: pd.DataFrame) -> pd.DataFrame:
    """How long each PR waited for its first review. The last three buckets are exactly the
    PRs that stalled (no first review within 7 days), so their counts sum to the stalled count."""
    hours = (prs["first_review_at"] - prs["created_at"]).dt.total_seconds() / 3600
    unreviewed = prs["first_review_at"].isna()
    bucket = pd.Series(None, index=prs.index, dtype=object)
    bucket[hours < 1] = WAIT_BUCKETS[0]
    bucket[(hours >= 1) & (hours < 24)] = WAIT_BUCKETS[1]
    bucket[(hours >= 24) & (hours <= 168)] = WAIT_BUCKETS[2]
    bucket[hours > 168] = WAIT_BUCKETS[3]
    bucket[unreviewed & prs["closed_at"].notna()] = WAIT_BUCKETS[4]
    bucket[unreviewed & prs["closed_at"].isna()] = WAIT_BUCKETS[5]
    counts = bucket.value_counts().reindex(list(WAIT_BUCKETS), fill_value=0)
    return pd.DataFrame({"bucket": list(WAIT_BUCKETS), "count": counts.to_numpy(),
                         "share": counts.to_numpy() / max(len(prs), 1),
                         "stalled": [b in STALLED_BUCKETS for b in WAIT_BUCKETS]})


# ---------------------------------------------------------------------------
# chapter 2: one repo's PRs over time
# ---------------------------------------------------------------------------

def timeline(prs: pd.DataFrame, repo: str, scenario: str, at: pd.Timestamp) -> pd.DataFrame:
    """One row per replayed PR in the repo, with its risk under the chosen model and whether it
    was awaiting review at `at` (the same rule as the triage list)."""
    rows = prs[prs["repo"] == repo]
    return pd.DataFrame({
        "pr_id": rows["pr_id"].to_numpy(),
        "number": rows["number"].to_numpy(),
        "title": rows["title"].to_numpy(),
        "created_at": rows["created_at"].to_numpy(),
        "risk": rows[SCORES[scenario]].to_numpy(),
        "stalled": rows["is_slow"].to_numpy(),
        "waiting": rows.index.isin(awaiting_review(rows, at).index),
        "outcome": outcome(rows).to_numpy(),
    })


def selected_pr_id(event) -> str | None:
    """The pr_id clicked in the timeline chart, read from Streamlit's selection event (a point
    selection named 'pick' on pr_id), or None when nothing is selected."""
    try:
        points = event["selection"]["pick"]
    except (KeyError, TypeError):
        return None
    return points[0].get("pr_id") if points else None


# ---------------------------------------------------------------------------
# chapter 3: why one PR scored as it did, and what the model leans on overall
# ---------------------------------------------------------------------------

def why(pr: pd.DataFrame, scenario: str, features: list[dict], k: int = 8) -> tuple[pd.DataFrame, float]:
    """The k features that pushed this PR's score most (exact TreeSHAP, on the model's raw
    log-odds), largest first, then one row for all the others, and the base value. By SHAP's
    additivity, base + the pushes = the model's log-odds for this PR."""
    row, s = pr.iloc[0], scenario.lower()
    meta = {f["feature"]: f for f in features}
    push = pd.Series({f: float(row[f"shap_{s}__{f}"]) for f in meta})
    top = list(push.abs().sort_values(ascending=False, kind="mergesort").index[:k])
    rows = [{"label": meta[f]["label"],
             "value": format_value(row[f"x__{f}"], meta[f]["dtype"], meta[f]["rate"]),
             "push": push[f]} for f in top]
    rows.append({"label": f"all other {len(meta) - len(top)} features", "value": "",
                 "push": float(push.drop(top).sum())})
    return pd.DataFrame(rows), float(row[f"base_{s}"])


def importance(scenario: str, features: list[dict], template: str | None = None) -> pd.DataFrame:
    """Phase 6's share of mean |SHAP| per feature, for the chosen model, largest first."""
    imp = pd.read_csv((template or IMPORTANCE).format(scenario))
    labels = {f["feature"]: f["label"] for f in features}
    return (imp.assign(label=imp["feature"].map(labels))
            .sort_values("share", ascending=False, kind="mergesort")
            .reset_index(drop=True)[["feature", "label", "share"]])


# ---------------------------------------------------------------------------
# chapter 4: the game
# ---------------------------------------------------------------------------

def week(created_at: pd.Series) -> pd.Series:
    """ISO year and week of opening, in UTC."""
    return created_at.dt.strftime("%G-W%V")


def eligible_weeks(prs: pd.DataFrame) -> list[tuple[str, str]]:
    """The (repo, week) groups a round can come from: at least one PR that stalled and at least
    three that did not."""
    stats = prs.assign(week=week(prs["created_at"])).groupby(["repo", "week"])["is_slow"].agg(["sum", "size"])
    keep = stats[(stats["sum"] >= 1) & (stats["size"] - stats["sum"] >= GAME_SIZE - 1)]
    return list(keep.index)


def deal(prs: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    """One round: an eligible (repo, week) chosen uniformly, then one PR from it that stalled
    and three that did not, chosen uniformly, in shuffled order."""
    repo, wk = rng.choice(eligible_weeks(prs))
    group = prs[(prs["repo"] == repo) & (week(prs["created_at"]) == wk)]
    stalled = sorted(group.loc[group["is_slow"], "pr_id"])
    fine = sorted(group.loc[~group["is_slow"], "pr_id"])
    ids = [rng.choice(stalled)] + rng.sample(fine, GAME_SIZE - 1)
    rng.shuffle(ids)
    return prs.set_index("pr_id").loc[ids].reset_index()


def model_pick(round_: pd.DataFrame, scenario: str) -> str:
    """The PR the model ranks riskiest; ties go to the lower number."""
    order = round_.sort_values([SCORES[scenario], "number"], ascending=[False, True], kind="mergesort")
    return str(order["pr_id"].iloc[0])


def game_hit_rate(prs: pd.DataFrame, scenario: str) -> float:
    """The exact chance, under deal()'s draw, that model_pick is the PR that stalled. Per eligible
    week, each stalled PR s wins when all three PRs drawn with it rank below it: C(L_s, 3) / C(N, 3),
    where N is the week's PRs that did not stall and L_s how many of those model_pick ranks below
    s (a lower score, or the same score and a higher number). Averaged as deal() draws: over
    stalled PRs within a week, then over weeks."""
    score = SCORES[scenario]
    eligible = set(eligible_weeks(prs))
    rates = []
    for key, g in prs.assign(week=week(prs["created_at"])).groupby(["repo", "week"]):
        if key not in eligible:
            continue
        fine = g[~g["is_slow"]]
        n = len(fine)
        wins = [math.comb(int(((fine[score] < s[score]) |
                               ((fine[score] == s[score]) & (fine["number"] > s["number"]))).sum()),
                          GAME_SIZE - 1) / math.comb(n, GAME_SIZE - 1)
                for _, s in g[g["is_slow"]].iterrows()]
        rates.append(sum(wins) / len(wins))
    return sum(rates) / len(rates)


# ---------------------------------------------------------------------------
# chapter 5: the measured results, from Phase 4's committed runs
# ---------------------------------------------------------------------------

def results(path: Path | str | None = None) -> dict:
    """AUC-PR per scenario and feature set (mean over folds, plus each fold), Scenario A's P@10
    against the baseline and a random pick, and the headline strings, formatted the way
    writeup_claims.py registers them (Scenario B is the mean over its five folds)."""
    runs = json.loads(Path(path or RUNS).read_text(encoding="utf-8"))["runs"]

    def pick(scenario: str, featureset: str) -> list[dict]:
        return sorted((r for r in runs if r["scenario"] == scenario and r["featureset"] == featureset),
                      key=lambda r: r["fold"])

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    folds = []
    for sc in ("A", "B"):
        full, nlr = pick(sc, "FULL"), pick(sc, "NO_LABEL_REPLAY")
        for key, values in (("FULL", [r["auc_pr"] for r in full]),
                            ("NO_LABEL_REPLAY", [r["auc_pr"] for r in nlr]),
                            ("baseline", [r["baseline_auc_pr"] for r in full])):
            folds += [{"scenario": SCENARIO_LABELS[sc], "series": SERIES_LABELS[key], "fold": i,
                       "auc_pr": v, "folds": len(values)} for i, v in enumerate(values)]
    folds = pd.DataFrame(folds)
    means = (folds.groupby(["scenario", "series"], sort=False)
             .agg(auc_pr=("auc_pr", "mean"), folds=("folds", "first")).reset_index())
    a = pick("A", "FULL")[0]
    b_full, b_nlr = pick("B", "FULL"), pick("B", "NO_LABEL_REPLAY")
    b_base = mean([r["baseline_auc_pr"] for r in b_full])
    text = {
        "a_p10": f"{a['precision_at_10']:.3f} [{a['p10_ci_lo']:.3f}, {a['p10_ci_hi']:.3f}]",
        "a_baseline_p10": f"{a['baseline_p10']:.3f}",
        "a_random_p10": f"{a['base_rate_p10']:.3f}",
        "b_full_vs_baseline": f"{mean([r['auc_pr'] for r in b_full]):.3f} vs {b_base:.3f}",
        "b_nlr_vs_baseline": f"{mean([r['auc_pr'] for r in b_nlr]):.3f} vs {b_base:.3f}",
    }
    p10 = pd.DataFrame([
        {"series": "model", "p10": a["precision_at_10"], "lo": a["p10_ci_lo"], "hi": a["p10_ci_hi"],
         "label": text["a_p10"]},
        {"series": "trailing-rate baseline", "p10": a["baseline_p10"], "lo": None, "hi": None,
         "label": text["a_baseline_p10"]},
        {"series": "random pick", "p10": a["base_rate_p10"], "lo": None, "hi": None,
         "label": text["a_random_p10"]},
    ])
    return {"folds": folds, "means": means, "p10": p10, "text": text}
```

- [ ] **Step 4: Run the tests and the suite**

Run `python -m pytest tests/test_demo_triage.py -q` and expect **39 passed**. Then run `python -m pytest tests -q` and expect **485 passed**.

- [ ] **Step 5: Commit**

```bash
git add demo/triage.py tests/test_demo_triage.py
git commit -m "feat(demo): chapter logic -- wait buckets, timeline, why, the game, results

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Mutation check (after the commit, restoring with git)**

All mutations are in `demo/triage.py`, and all target tests are in `tests/test_demo_triage.py`. For each one: apply it, run the named test and confirm it FAILS, then run `git checkout -- demo/triage.py` and confirm the tree is clean.
- `bucket[unreviewed & prs["closed_at"].notna()] = WAIT_BUCKETS[4]` → `.isna()` → `test_wait_buckets_cover_every_pr_and_split_the_unreviewed`
- `"waiting": rows.index.isin(awaiting_review(rows, at).index),` → prefix `~` → `test_timeline_marks_waiting_with_the_triage_lists_rule`
- `return points[0].get("pr_id") if points else None` → `return None` → `test_selected_pr_id_reads_a_point_selection_event`
- `top = list(push.abs().sort_values(` → `top = list(push.sort_values(` → `test_why_ranks_by_size_so_a_large_push_down_comes_first`
- `"push": float(push.drop(top).sum())})` → `"push": 0.0})` → `test_why_rows_and_base_add_up_to_the_models_logit`
- `(fine["number"] > s["number"])` → `(fine["number"] < s["number"])` → `test_game_hit_rate_equals_brute_force_over_every_round_including_ties`
- In `model_pick`: `ascending=[False, True]` → `ascending=[False, False]` → `test_model_pick_breaks_ties_by_the_lower_number`
- `ids = [rng.choice(stalled)] + rng.sample(fine, GAME_SIZE - 1)` → `ids = rng.sample(stalled + fine, GAME_SIZE)` → `test_every_dealt_round_is_four_prs_from_one_week_with_exactly_one_stalled`
- `"a_random_p10": f"{a['base_rate_p10']:.3f}",` → use `a['baseline_p10']` → `test_results_text_matches_the_readmes_registered_claims`

---

### Task 3: The charts

**Files:**
- Create: `demo/charts.py`, `tests/test_demo_charts.py`

**Interfaces:**
- Consumes: the frames from Task 2.
- Produces, in `demo/charts.py`:
  - `PALETTE["light"|"dark"]` (validated colours)
  - `FINE`, `STALLED`, `BUCKET_STEP = 120`, `BUCKET_LINES`
  - `wait_histogram(buckets, mode)`, `timeline(frame, at, mode)` (the point selection is named `pick`, on `pr_id`, with `nearest=True`)
  - `why_bars(rows, mode)`, `importance_bars(frame, top, mode)`
  - `results_chart(res, mode)`, `p10_chart(res, mode)`
  - Each returns an `altair.LayerChart`.

- [ ] **Step 1: Write the failing tests**

`tests/test_demo_charts.py`:

```python
"""The demo's charts, read through their Vega-Lite specs (demo-story spec, section 6)."""
import pandas as pd
import pytest

from demo import charts, triage


@pytest.fixture(scope="module")
def prs():
    return triage.load()


def _layer(chart, mark):
    """The first layer drawing `mark` (as a dict), with its data frame. Altair moves data that
    layers share up to the layered chart, so fall back to the chart's own data."""
    for layer in chart.layer:
        spec = layer.to_dict()
        kind = spec["mark"]["type"] if isinstance(spec["mark"], dict) else spec["mark"]
        if kind == mark:
            data = layer.data if isinstance(layer.data, pd.DataFrame) else chart.data
            return spec, data
    raise LookupError(mark)


def _params(chart):
    """Selection parameters, which Altair lifts to the top of a layered chart."""
    return chart.to_dict().get("params", [])


def test_both_themes_define_every_colour_role():
    assert set(charts.PALETTE["light"]) == set(charts.PALETTE["dark"])


def test_wait_histogram_colours_by_stalled_and_puts_the_7_day_rule_between_bars(prs):
    buckets = triage.wait_buckets(prs)
    chart = charts.wait_histogram(buckets)
    bars, data = _layer(chart, "bar")
    assert bars["encoding"]["color"]["scale"]["domain"] == [charts.FINE, charts.STALLED]
    assert data["count"].tolist() == buckets["count"].tolist()
    rule, rule_data = _layer(chart, "rule")
    assert rule["mark"]["xOffset"] == -charts.BUCKET_STEP / 2
    assert rule_data["axis_label"].tolist() == [charts.BUCKET_LINES["after more than 7 days"]]


def test_timeline_plots_every_pr_with_a_nearest_click_selection_and_the_day_rule(prs):
    at = triage.moment(triage.DEFAULT_DAY)
    frame = triage.timeline(prs, "radixark/miles", "A", at)
    chart = charts.timeline(frame, at)
    points, data = _layer(chart, "circle")
    assert len(data) == len(frame) and data["risk"].tolist() == frame["risk"].tolist()
    pick = [p for p in _params(chart) if p["name"] == "pick"][0]
    assert pick["select"]["fields"] == ["pr_id"] and pick["select"]["nearest"] is True
    assert points["encoding"]["y"]["scale"]["domain"] == [0, 1]
    _, rule_data = _layer(chart, "rule")
    assert rule_data["at"].tolist() == [at]


def test_why_bars_keep_the_given_order_and_colour_by_direction(prs):
    rows, _ = triage.why(triage.find_pr(prs, "radixark/miles", 754), "A", triage.load_features())
    bars, data = _layer(charts.why_bars(rows), "bar")
    assert bars["encoding"]["y"]["sort"] is None
    assert bars["encoding"]["color"]["scale"]["domain"] == ["raises risk", "lowers risk"]
    assert data["push"].tolist() == rows["push"].tolist()
    assert data["direction"].tolist() == ["raises risk" if p > 0 else "lowers risk" for p in rows["push"]]


def test_importance_bars_show_the_top_features():
    imp = triage.importance("B", triage.load_features())
    _, data = _layer(charts.importance_bars(imp, top=7), "bar")
    assert data["feature"].tolist() == imp["feature"].head(7).tolist()


def test_results_chart_starts_at_zero_and_dots_only_the_five_fold_scenario():
    res = triage.results()
    chart = charts.results_chart(res)
    bars, data = _layer(chart, "bar")
    assert bars["encoding"]["y"]["scale"]["domain"] == [0, 1]
    assert len(data) == 6
    _, dots = _layer(chart, "circle")
    assert len(dots) == 15 and set(dots["scenario"]) == {triage.SCENARIO_LABELS["B"]}


def test_p10_chart_labels_each_bar_with_its_formatted_value():
    res = triage.results()
    chart = charts.p10_chart(res, mode="dark")
    _, labels = _layer(chart, "text")
    assert labels["label"].tolist() == [res["text"]["a_p10"], res["text"]["a_baseline_p10"], res["text"]["a_random_p10"]]
    _, whisker = _layer(chart, "rule")
    assert whisker["series"].tolist() == ["model"]
```

- [ ] **Step 2: Run them and confirm they fail**

Run `python -m pytest tests/test_demo_charts.py -q`. Expected: a collection error, `ImportError: cannot import name 'charts' from 'demo'`.

- [ ] **Step 3: Implement `demo/charts.py`**

```python
"""The demo's charts: pure functions from triage.py's frames to Altair charts.

Nothing here touches Streamlit, so tests read each chart's spec with .to_dict(). Colours come
from the validated reference palette, by job: stalled = orange and reviewed in time = blue
(categorical slots 2 and 1), pushes towards stalling = red and away = blue (the diverging
pair), and the results chart keeps Figure 1's roles (blue, light blue, gray baseline). Light
and dark steps are separate because Streamlit does not recolour explicit colours when a viewer
switches theme; every pair was checked with the palette validator in both modes."""
from __future__ import annotations

import altair as alt
import pandas as pd

alt.data_transformers.disable_max_rows()        # the largest repo has 5,178 replayed PRs

PALETTE = {
    "light": {"fine": "#2a78d6", "stalled": "#eb6834", "up": "#e34948", "down": "#2a78d6",
              "full": "#2a78d6", "nlr": "#86b6ef", "baseline": "#898781", "random": "#c3c2b7",
              "ink": "#0b0b0b", "muted": "#52514e"},
    "dark": {"fine": "#3987e5", "stalled": "#d95926", "up": "#e66767", "down": "#3987e5",
             "full": "#3987e5", "nlr": "#86b6ef", "baseline": "#898781", "random": "#52514e",
             "ink": "#ffffff", "muted": "#c3c2b7"},
}
FINE, STALLED = "first review within 7 days", "stalled: no first review within 7 days"
ON_BAR = {"model, all features": "#ffffff"}       # text inside bars: white on the dark blue, else near-black
BUCKET_STEP = 120                                 # px per histogram bar, so the 7-day rule sits between bars
BUCKET_LINES = {"within an hour": "within|an hour", "1 hour to 1 day": "1 hour|to 1 day",
                "1 to 7 days": "1 to|7 days", "after more than 7 days": "after more|than 7 days",
                "closed without a review": "closed without|a review",
                "never reviewed, still open": "never reviewed,|still open"}


def _status(stalled) -> list[str]:
    return [STALLED if s else FINE for s in stalled]


def _status_color(pal: dict) -> alt.Color:
    return alt.Color("status:N", scale=alt.Scale(domain=[FINE, STALLED], range=[pal["fine"], pal["stalled"]]),
                     legend=alt.Legend(title=None, orient="top", labelLimit=320))


def wait_histogram(buckets: pd.DataFrame, mode: str = "light") -> alt.LayerChart:
    """Share of PRs per wait bucket, blue before the 7-day line and orange after it."""
    pal = PALETTE[mode]
    data = buckets.assign(status=_status(buckets["stalled"]), axis_label=buckets["bucket"].map(BUCKET_LINES))
    order = list(data["axis_label"])
    base = alt.Chart(data).encode(
        x=alt.X("axis_label:N", sort=order, title=None,
                axis=alt.Axis(labelAngle=0, labelLimit=BUCKET_STEP, labelExpr="split(datum.label, '|')")),
        y=alt.Y("share:Q", title="share of replayed PRs", axis=alt.Axis(format="%", grid=True)),
        tooltip=[alt.Tooltip("bucket:N", title="first review"), alt.Tooltip("count:Q", format=","),
                 alt.Tooltip("share:Q", format=".1%")])
    bars = base.mark_bar(cornerRadiusEnd=4).encode(color=_status_color(pal))
    labels = base.mark_text(dy=-8, color=pal["ink"]).encode(text=alt.Text("share:Q", format=".0%"))
    rule_at = pd.DataFrame({"axis_label": [order[3]], "label": ["7 days"]})
    rule = alt.Chart(rule_at).mark_rule(color=pal["muted"], strokeDash=[4, 4], xOffset=-BUCKET_STEP / 2).encode(
        x=alt.X("axis_label:N", sort=order))
    rule_label = alt.Chart(rule_at).mark_text(color=pal["muted"], xOffset=-BUCKET_STEP / 2, dx=4, align="left",
                                              y=6).encode(x=alt.X("axis_label:N", sort=order), text="label:N")
    return (bars + labels + rule + rule_label).properties(width=alt.Step(BUCKET_STEP), height=300)


def timeline(frame: pd.DataFrame, at: pd.Timestamp, mode: str = "light") -> alt.LayerChart:
    """Each PR by opening date and risk at open, coloured by whether it stalled. PRs waiting at
    `at` are solid, the rest faded; a rule marks `at`. Clicking selects the nearest PR ('pick')."""
    pal = PALETTE[mode]
    pick = alt.selection_point(name="pick", fields=["pr_id"], on="click", nearest=True, empty=False)
    points = alt.Chart(frame.assign(status=_status(frame["stalled"]))).mark_circle(stroke=pal["ink"]).encode(
        x=alt.X("created_at:T", title="opened"),
        y=alt.Y("risk:Q", title="risk score at open", scale=alt.Scale(domain=[0, 1])),
        color=_status_color(pal),
        opacity=alt.condition(alt.datum.waiting, alt.value(0.95), alt.value(0.22)),
        size=alt.condition(pick, alt.value(220), alt.value(70)),
        strokeWidth=alt.condition(pick, alt.value(2), alt.value(0)),
        tooltip=[alt.Tooltip("number:Q", title="PR #"), alt.Tooltip("title:N"),
                 alt.Tooltip("risk:Q", format=".2f"), alt.Tooltip("outcome:N", title="what happened"),
                 alt.Tooltip("waiting:N", title="waiting on the chosen day")],
    ).add_params(pick)
    marker = pd.DataFrame({"at": [at], "label": ["chosen day"]})
    rule = alt.Chart(marker).mark_rule(color=pal["muted"], strokeWidth=1.5).encode(x="at:T")
    label = alt.Chart(marker).mark_text(color=pal["muted"], align="left", dx=4, y=8).encode(x="at:T", text="label:N")
    return (points + rule + label).properties(height=380)


def why_bars(rows: pd.DataFrame, mode: str = "light") -> alt.LayerChart:
    """One PR's biggest pushes on the model's log-odds: red raises the risk, blue lowers it."""
    pal = PALETTE[mode]
    data = rows.assign(direction=["raises risk" if p > 0 else "lowers risk" for p in rows["push"]],
                       shown=[f"{lab} = {val}" if val else lab for lab, val in zip(rows["label"], rows["value"])])
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=4, height=18).encode(
        y=alt.Y("shown:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
        x=alt.X("push:Q", title="push on the model's score (log-odds)"),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["raises risk", "lowers risk"],
                                                       range=[pal["up"], pal["down"]]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("label:N", title="feature"), alt.Tooltip("value:N"),
                 alt.Tooltip("push:Q", format="+.3f")])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=pal["muted"]).encode(x="x:Q")
    return (bars + zero).properties(height=alt.Step(26))


def importance_bars(frame: pd.DataFrame, top: int = 10, mode: str = "light") -> alt.LayerChart:
    """What the model leans on overall: Phase 6's share of mean |SHAP|, top features."""
    pal = PALETTE[mode]
    data = frame.head(top)
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
        x=alt.X("share:Q", title="share of the model's attribution", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("label:N", title="feature"), alt.Tooltip("share:Q", format=".1%")])
    bars = base.mark_bar(cornerRadiusEnd=4, height=16, color=pal["full"])
    labels = base.mark_text(align="left", dx=4, color=pal["ink"]).encode(text=alt.Text("share:Q", format=".0%"))
    return (bars + labels).properties(height=alt.Step(24))


def results_chart(res: dict, mode: str = "light") -> alt.LayerChart:
    """Mean AUC-PR by scenario and feature set, from zero; Scenario B's five folds as dots."""
    pal = PALETTE[mode]
    order = list(dict.fromkeys(res["means"]["series"]))
    color = alt.Color("series:N", sort=order, scale=alt.Scale(domain=order,
                                                              range=[pal["full"], pal["nlr"], pal["baseline"]]),
                      legend=alt.Legend(title=None, orient="top", labelLimit=320))
    x = alt.X("scenario:N", title=None, sort=list(dict.fromkeys(res["means"]["scenario"])), axis=alt.Axis(labelAngle=0))
    offset = alt.XOffset("series:N", sort=order)
    y = alt.Y("auc_pr:Q", title="AUC-PR", scale=alt.Scale(domain=[0, 1]))
    bars = alt.Chart(res["means"]).mark_bar(cornerRadiusEnd=4).encode(
        x=x, xOffset=offset, y=y, color=color,
        tooltip=[alt.Tooltip("scenario:N"), alt.Tooltip("series:N"), alt.Tooltip("auc_pr:Q", format=".3f", title="mean AUC-PR"),
                 alt.Tooltip("folds:Q")])
    # values at the bar's base, where no fold dot reaches, so a label never collides with a dot
    on_bar = res["means"].assign(text_color=[ON_BAR.get(s, "#0b0b0b") for s in res["means"]["series"]])
    labels = alt.Chart(on_bar).mark_text(baseline="bottom", dy=-6).encode(
        x=x, xOffset=offset, y=alt.datum(0), text=alt.Text("auc_pr:Q", format=".3f"),
        color=alt.Color("text_color:N", scale=None))
    folds = res["folds"][res["folds"]["folds"] > 1]
    dots = alt.Chart(folds).mark_circle(size=36, color=pal["ink"], opacity=0.75).encode(
        x=x, xOffset=offset, y=y,
        tooltip=[alt.Tooltip("series:N"), alt.Tooltip("fold:Q", title="held-out fold"),
                 alt.Tooltip("auc_pr:Q", format=".3f")])
    return (bars + labels + dots).properties(height=340)


def p10_chart(res: dict, mode: str = "light") -> alt.LayerChart:
    """Scenario A precision on each repo's top 10: model (with its interval), baseline, random pick."""
    pal = PALETTE[mode]
    data = res["p10"]
    order = list(data["series"])
    color = alt.Color("series:N", sort=order, scale=alt.Scale(domain=order,
                                                              range=[pal["full"], pal["baseline"], pal["random"]]),
                      legend=None)
    base = alt.Chart(data).encode(
        y=alt.Y("series:N", sort=order, title=None, axis=alt.Axis(labelLimit=200)),
        x=alt.X("p10:Q", title="precision on each repo's 10 riskiest PRs", scale=alt.Scale(domain=[0, 1])))
    bars = base.mark_bar(cornerRadiusEnd=4, height=22).encode(
        color=color, tooltip=[alt.Tooltip("series:N"), alt.Tooltip("p10:Q", format=".3f", title="precision"),
                              alt.Tooltip("lo:Q", format=".3f", title="interval low"),
                              alt.Tooltip("hi:Q", format=".3f", title="interval high")])
    whisker = alt.Chart(data.dropna(subset=["lo"])).mark_rule(color=pal["ink"], strokeWidth=2).encode(
        y=alt.Y("series:N", sort=order), x="lo:Q", x2="hi:Q")
    labels = base.mark_text(align="left", dx=6, color=pal["ink"]).encode(
        x=alt.X("plotted:Q"), text="label:N").transform_calculate(
        plotted="isValid(datum.hi) ? datum.hi : datum.p10")
    return (bars + whisker + labels).properties(height=alt.Step(40))
```

- [ ] **Step 4: Run the tests and the suite**

Run `python -m pytest tests/test_demo_charts.py -q` and expect **7 passed**. Then run `python -m pytest tests -q` and expect **492 passed**.

- [ ] **Step 5: Commit**

```bash
git add demo/charts.py tests/test_demo_charts.py
git commit -m "feat(demo): Altair charts from the validated palette, light and dark

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Mutation check (after the commit, restoring with git)**

All mutations are in `demo/charts.py`, and all target tests are in `tests/test_demo_charts.py`. For each one: apply it, run the named test and confirm it FAILS, then run `git checkout -- demo/charts.py` and confirm the tree is clean.
- `strokeDash=[4, 4], xOffset=-BUCKET_STEP / 2)` → `xOffset=BUCKET_STEP / 2)` (the first occurrence, on `mark_rule`) → `test_wait_histogram_colours_by_stalled_and_puts_the_7_day_rule_between_bars`
- `on="click", nearest=True, empty=False)` → `nearest=False` → `test_timeline_plots_every_pr_with_a_nearest_click_selection_and_the_day_rule`
- `folds = res["folds"][res["folds"]["folds"] > 1]` → `folds = res["folds"]` → `test_results_chart_starts_at_zero_and_dots_only_the_five_fold_scenario`

---

### Task 4: The chapters and the app shell

**Files:**
- Create: `demo/chapters.py`, `demo/views/problem.py`, `demo/views/watch.py`, `demo/views/why.py`, `demo/views/test_yourself.py`, `demo/views/transfer.py`, `demo/views/limits.py`
- Rewrite: `demo/app.py`, `tests/test_demo_app.py`
- Modify: `demo/requirements.txt`, `requirements.txt`

**Interfaces:**
- Consumes: `triage` (Task 2), `charts` (Task 3), `writeup_claims.CLAIMS` and `RETRACTED` (Phase 8).
- Produces:
  - **Chapters:** `chapters.Context` and `chapters.ctx`; `MODELS` (sidebar radio labels); `NOTES`; the chapter functions `problem`, `watch`, `why`, `test_yourself`, `transfer`, `limits`.
  - **Page files:** `views/<chapter>.py`.
  - **Widget keys the tests drive:** `next_i`, `back_i`, `day`, `pr_ref`, `random_pr`, `pick_0..3`, `reveal`, `deal`.
  - **Session state:** `selected_pr`, `game`.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_demo_app.py` with:

```python
"""The demo app, run headless with Streamlit's AppTest (Phase 7 and demo-story specs).

Chapters are reached with AppTest.switch_page on their page files (views/), which keeps the
current chapter between runs; the Back/Next buttons are tested separately."""
import ast
import importlib
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import writeup_claims as wc
from demo import triage

DEMO = Path(__file__).parents[1] / "demo"
APP = DEMO / "app.py"
VIEWS = ["problem", "watch", "why", "test_yourself", "transfer", "limits"]
HEADINGS = ["Some pull requests wait weeks for a first review", "Watch it work", "Why it decides",
            "Test yourself", "Does it transfer to a new project?", "Honest limits"]
UNSEEN = "Unseen repo (cold-start)"
SOURCES = sorted(DEMO.glob("*.py")) + sorted((DEMO / "views").glob("*.py"))


def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=90).run()


def _at(chapter: str, at: AppTest | None = None) -> AppTest:
    at = at or _app()
    return at.switch_page(f"views/{chapter}.py").run()


def _markdown(at, start):
    return next(m.value for m in at.markdown if m.value.startswith(start))


def _text(at) -> str:
    return " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# the story: six chapters, Back/Next, speaker notes
# ---------------------------------------------------------------------------

def test_the_app_opens_on_chapter_one():
    at = _app()
    assert not at.exception and at.title[0].value == HEADINGS[0]


@pytest.mark.parametrize("i", range(6))
def test_every_chapter_renders_without_an_exception(i):
    at = _at(VIEWS[i])
    assert not at.exception and at.title[0].value == HEADINGS[i]


@pytest.mark.parametrize("i", range(5))
def test_next_opens_the_following_chapter(i):
    at = _at(VIEWS[i])
    at.button(key=f"next_{i}").click().run()
    assert at.title[0].value == HEADINGS[i + 1]


@pytest.mark.parametrize("i", range(1, 6))
def test_back_opens_the_previous_chapter(i):
    at = _at(VIEWS[i])
    at.button(key=f"back_{i}").click().run()
    assert at.title[0].value == HEADINGS[i - 1]


def test_the_first_chapter_has_no_back_and_the_last_no_next():
    assert "back_0" not in [b.key for b in _at("problem").button]
    assert "next_5" not in [b.key for b in _at("limits").button]


def test_speaker_notes_are_hidden_by_default_and_shown_when_toggled():
    at = _at("problem")
    assert not [i for i in at.info if i.value.startswith("**Speaker note:**")]
    at.sidebar.toggle[0].set_value(True).run()
    assert [i for i in at.info if i.value.startswith("**Speaker note:**")]


# ---------------------------------------------------------------------------
# 1. the problem
# ---------------------------------------------------------------------------

def test_the_problem_chapter_quotes_the_computed_stalled_share():
    buckets = triage.wait_buckets(triage.load())
    stalled = buckets.loc[buckets["stalled"], "share"].sum()
    assert f"**{stalled:.0%} stalled**" in _text(_at("problem"))


# ---------------------------------------------------------------------------
# 2. watch it work (the Phase 7 triage checks, migrated)
# ---------------------------------------------------------------------------

def test_watch_opens_on_the_default_repo_and_day_with_a_list():
    at = _at("watch")
    prs = triage.load()
    assert at.sidebar.selectbox[0].value == triage.default_repo(prs, triage.moment(triage.DEFAULT_DAY))
    assert at.slider(key="day").value == triage.DEFAULT_DAY
    assert len(at.dataframe) >= 1 and len(at.dataframe[0].value) > 0


def test_switching_the_model_changes_the_scores_shown():
    at = _at("watch")
    seen = at.dataframe[0].value["risk"].tolist()
    at.sidebar.radio[0].set_value(UNSEEN).run()
    assert not at.exception
    assert at.dataframe[0].value["risk"].tolist() != seen


def test_changing_the_day_re_renders_the_list():
    at = _at("watch")
    before = at.dataframe[0].value["url"].tolist()
    at.slider(key="day").set_value(triage.DEFAULT_DAY.replace(month=2)).run()
    assert not at.exception
    assert at.dataframe[0].value["url"].tolist() != before


def test_a_day_with_nobody_waiting_says_so():
    prs, day = triage.load(), triage.FIRST_DAY
    empty = next(r for r in triage.repos(prs) if triage.ranked(prs, r, triage.moment(day), "A").empty)
    at = _at("watch")
    at.sidebar.selectbox[0].set_value(empty).run()
    at.slider(key="day").set_value(day).run()
    assert not at.exception
    assert "early January" in at.info[0].value


def test_the_tally_shows_the_whole_lists_stall_count_beside_the_top_three():
    """PRs still waiting at a moment are survivors and mostly stall, so a top-3 count alone
    reads as skill it does not measure. The line must show the list's own rate and say so."""
    at = _at("watch")
    prs = triage.load()
    at_ = triage.moment(triage.DEFAULT_DAY)
    listing = triage.ranked(prs, triage.default_repo(prs, at_), at_, "A")
    stalled, k = triage.tally(listing)
    line = _markdown(at, "**Of the top")
    assert f"Of the top {k} by risk, {stalled} stalled" in line
    assert f"of all {len(listing)} PRs waiting, {int(listing['stalled'].sum())} did" in line
    assert "cannot show how well the model ranks" in line


def test_the_seen_note_describes_one_model_for_all_repos():
    """kdlbs/kandev has no PRs before 2026, so 'trained on this repo's own earlier PRs' is false."""
    at = _at("watch")
    at.sidebar.selectbox[0].set_value("kdlbs/kandev").run()
    note = _markdown(at, "**Seen in training:**")
    assert f"all {len(triage.repos(triage.load()))} repos" in note and "if it had any" in note
    assert "this repo's own earlier PRs" not in note


def test_the_watch_chapter_says_only_prs_opened_in_2026_are_shown_and_none_is_live():
    text = _text(_at("watch"))
    assert "Only PRs opened from 1 January 2026 have scores" in text
    assert "not live data" in text and "never updated" in text


def test_the_caption_says_risk_is_a_ranking_score_not_a_probability():
    assert "not a calibrated probability" in _at("watch").caption[0].value


# ---------------------------------------------------------------------------
# 3. why it decides (the PR #10 lookup, migrated)
# ---------------------------------------------------------------------------

def _lookup(at: AppTest, text: str) -> AppTest:
    at.text_input(key="pr_ref").input(text).run()
    return at


def test_looking_up_a_pr_link_shows_both_models_scores_and_what_happened():
    prs = triage.load()
    pr = prs.iloc[[100]]
    row = pr.iloc[0]
    at = _lookup(_at("why"), row["url"])
    assert not at.exception
    assert [m.value for m in at.metric] == [f"{row['score_a']:.2f}", f"{row['score_b']:.2f}"]
    shown = _text(at)
    assert triage.outcome(pr).iloc[0] in shown
    share, n = triage.rank_in_repo(prs, pr, "A")
    assert f"Higher than {share:.0%} of this repo's other {n:,} replayed PRs" in shown


def test_an_unknown_pr_says_it_is_not_in_the_replay():
    at = _lookup(_at("why"), "nobody/nothing#1")
    assert not at.exception and len(at.metric) == 0
    assert "Not in the replay" in at.warning[0].value


def test_random_pr_fills_the_box_and_shows_its_card():
    at = _at("why")
    at.button(key="random_pr").click().run()
    assert not at.exception
    assert re.fullmatch(r"[^/\s]+/[^#\s]+#\d+", at.text_input(key="pr_ref").value)
    assert len(at.metric) == 2


def test_with_nothing_selected_why_explains_the_riskiest_pr_waiting_on_the_default_day():
    prs = triage.load()
    at_ = triage.moment(triage.DEFAULT_DAY)
    repo = triage.default_repo(prs, at_)
    waiting = triage.awaiting_review(prs[prs["repo"] == repo], at_)
    top = waiting.sort_values(["score_a", "number"], ascending=[False, True]).iloc[0]
    at = _at("why")
    assert [m.value for m in at.metric] == [f"{top['score_a']:.2f}", f"{top['score_b']:.2f}"]
    assert f"Why the Seen in training model scored it {top['score_a']:.2f}" in [s.value for s in at.subheader]


def test_the_why_caption_says_attribution_not_cause_and_that_bars_add_up():
    caption = _at("why").caption[0].value
    assert "Model attribution, not cause" in caption and "add up to the score shown" in caption


# ---------------------------------------------------------------------------
# 4. test yourself
# ---------------------------------------------------------------------------

def test_the_game_deals_four_cards_and_reveal_scores_the_round():
    at = _at("test_yourself")
    assert [f"pick_{i}" for i in range(4)] == [b.key for b in at.button if b.key.startswith("pick_")]
    at.button(key="pick_1").click().run()
    at.button(key="reveal").click().run()
    assert not at.exception
    text = _text(at)
    assert text.count("**Stalled**") == 1 and text.count("Reviewed within 7 days") == 3
    assert "You picked PR 2" in text and "This session: you" in text and "of 1" in text


def test_deal_again_deals_a_new_round():
    at = _at("test_yourself")
    first = list(at.session_state["game"]["ids"])
    at.button(key="pick_0").click().run()
    at.button(key="reveal").click().run()
    at.button(key="deal").click().run()
    game = at.session_state["game"]
    assert game["revealed"] is False and game["pick"] is None and game["ids"] != first


def test_the_game_shows_its_exact_hit_rate_against_a_random_guess():
    rate = triage.game_hit_rate(triage.load(), "A")
    line = _markdown(_at("test_yourself"), "Over every round this game can deal")
    assert f"**{rate:.0%}**" in line and "**25%**" in line


# ---------------------------------------------------------------------------
# 5. does it transfer? / 6. honest limits
# ---------------------------------------------------------------------------

def test_the_transfer_chapter_shows_the_readmes_checked_numbers():
    claims = {c.name: c.expected for c in wc.CLAIMS}
    text = _text(_at("transfer"))
    for name in ("a_p10", "b_full_vs_baseline", "b_nlr_vs_baseline"):
        assert claims[name] in text, name
    assert claims["a_baseline_p10"].removeprefix("baseline scores ") in text
    assert claims["a_base_rate"].removeprefix("base rate of ") in text


def test_the_limits_chapter_names_the_four_limits():
    text = _text(_at("limits"))
    for phrase in ("never measured", "small sample", "2026 snapshots", "cannot measure ranking"):
        assert phrase in text, phrase


# ---------------------------------------------------------------------------
# robustness and guards
# ---------------------------------------------------------------------------

def test_the_app_uses_the_current_helpers_even_when_old_copies_are_cached(monkeypatch):
    """Streamlit Community Cloud re-runs app.py when the repo updates but keeps imported modules
    in memory, so the deployed app once ran a new app.py against an old triage.py. Recreate stale
    copies of every helper module and check the app reloads them."""
    monkeypatch.syspath_prepend(str(DEMO))
    for name, attr in (("triage", "random_pr"), ("charts", "why_bars"), ("chapters", "why")):
        monkeypatch.delattr(importlib.import_module(name), attr)
    at = _at("why")
    at.button(key="random_pr").click().run()
    assert not at.exception


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_the_demo_hard_codes_no_result_number(path):
    assert re.search(r"\b0\.\d{3}\b", path.read_text(encoding="utf-8")) is None


@pytest.mark.parametrize("pattern,reason", wc.RETRACTED, ids=lambda v: str(v)[:30])
def test_the_demo_avoids_retracted_phrasing(pattern, reason):
    for path in SOURCES:
        hit = re.search(pattern, path.read_text(encoding="utf-8"), flags=re.IGNORECASE)
        assert hit is None, f"{path.name} says {hit.group(0)!r}, but {reason}"


def test_the_demo_imports_only_what_its_own_requirements_install():
    """Streamlit Cloud installs demo/requirements.txt, not the project's: no sklearn, lightgbm,
    shap, or project module may be imported by the deployed files."""
    allowed = {"__future__", "dataclasses", "datetime", "importlib", "json", "math", "pathlib",
               "random", "re", "sys", "streamlit", "pandas", "altair", "triage", "charts", "chapters"}
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert mods <= allowed, f"demo/{path.name} imports {sorted(mods - allowed)}"


def test_demo_requirements_pin_only_what_the_app_needs_at_the_root_versions():
    """Streamlit, pandas, pyarrow and altair only, at the root file's versions. Pinning numpy as
    well blocked Python 3.14, which numpy 2.2.6 has no wheel for."""
    def pins(path):
        lines = [ln.split("#")[0].strip() for ln in path.read_text(encoding="utf-8").splitlines()]
        return dict(ln.split("==") for ln in lines if "==" in ln)
    demo, root = pins(DEMO / "requirements.txt"), pins(DEMO.parent / "requirements.txt")
    assert set(demo) == {"streamlit", "pandas", "pyarrow", "altair"}
    assert all(demo[p] == root[p] for p in demo)


def test_switching_the_model_mid_round_still_reveals_cleanly():
    """A presenter may flip the sidebar switch between picking and revealing."""
    at = _at("test_yourself")
    at.button(key="pick_0").click().run()
    at.sidebar.radio[0].set_value(UNSEEN).run()
    at.button(key="reveal").click().run()
    assert not at.exception
    rate = triage.game_hit_rate(triage.load(), "B")
    assert f"**{rate:.0%}**" in _markdown(at, "Over every round this game can deal")


def test_why_falls_back_to_the_repos_riskiest_pr_when_nobody_was_waiting():
    prs = triage.load()
    at_ = triage.moment(triage.DEFAULT_DAY)
    quiet = next(r for r in triage.repos(prs) if triage.awaiting_review(prs[prs["repo"] == r], at_).empty)
    top = prs[prs["repo"] == quiet].sort_values(["score_a", "number"], ascending=[False, True]).iloc[0]
    at = _at("why")
    at.sidebar.selectbox[0].set_value(quiet).run()
    assert not at.exception
    assert [m.value for m in at.metric] == [f"{top['score_a']:.2f}", f"{top['score_b']:.2f}"]
```

- [ ] **Step 2: Run them and confirm they fail**

Run `python -m pytest tests/test_demo_app.py -q`. Expected: most tests fail. The headings don't match, the `views/` pages don't exist (`AppTest.switch_page` raises), and the requirements test fails because `altair` isn't pinned yet.

- [ ] **Step 3: Pin Altair**

Append to `demo/requirements.txt`:

```
altair==6.3.0
```

In `requirements.txt`, add this line directly after the `streamlit==1.65.0` line:

```
altair==6.3.0           # the demo's interactive charts (installed with streamlit; pinned because demo/charts.py imports it)
```

- [ ] **Step 4: Implement the chapters, the page files and the shell**

`demo/chapters.py`:

```python
"""The demo's six chapters, for a 3-5 minute live walkthrough (spec section 7).

Each chapter is a function that st.navigation runs as a page. The app shell (app.py) sets `ctx`
before navigation runs; a chapter reads its data and the sidebar's choices from it, shows one
visual, its speaker note when the sidebar toggle is on, and Back/Next buttons. Every number
shown is computed at runtime from committed files; none is typed in here."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

import charts
import triage

GITHUB = "https://github.com/Raghav-28-Gupta/PRFlowPredict"
README, REPORT = f"{GITHUB}#readme", f"{GITHUB}/blob/main/docs/REPORT.md"
MODELS = {"Seen in training (within-project)": "A", "Unseen repo (cold-start)": "B"}
MODEL_NAMES = {"A": "Seen in training", "B": "Unseen repo"}
MODEL_NOTES = {
    "A": "**Seen in training:** scores from one model trained on the PRs opened before 2026 in "
         "all {n} repos, so it has seen this repo's earlier PRs, if it had any.",
    "B": "**Unseen repo:** scores from the model trained without this repo, as if it were new "
         "to the model.",
}
GAME_FACTS = ["additions_at_open", "deletions_at_open", "n_commits_at_open", "n_prior_prs_here",
              "author_prior_slow_rate_here", "trailing_90d_slow_rate", "body_len", "is_draft_at_open"]
NOTES = {
    "problem": "Open with the pain: some PRs are reviewed within the hour, but about half never get "
               "a first review within a week, and most of those are closed without one. The goal is "
               "to flag the risky ones the moment they are opened.",
    "watch": "This is the 2026 test period, which the model never trained on. Each dot is a real PR "
             "scored when it opened. Drag the day: solid dots are the ones waiting then. Click a "
             "high-risk orange dot to explain it next.",
    "why": "Exact SHAP attribution. Point at the biggest bar: usually the author's track record in "
           "this repo. Then the overall chart: the model leans on history features most. Stress "
           "attribution, not cause.",
    "test": "Hand the choice to the interviewer. After the reveal, point at the hit-rate line: over "
            "every round the game can deal, the model beats a random guess clearly, but it is far "
            "from perfect.",
    "transfer": "The honest headline. Within a project the model beats the baseline. On repos it "
                "has never seen, it beats the baseline only with the repo's slow-rate history; "
                "without it, it falls below. Then flip the sidebar switch to Unseen repo.",
    "limits": "Close on what was not measured. This is what makes the rest credible.",
}


@dataclass
class Context:
    prs: pd.DataFrame
    features: list[dict]
    results: dict
    hit_rates: dict
    scenario: str
    repo: str
    notes: bool
    mode: str
    pages: list = field(default_factory=list)


ctx: Context | None = None


def _note(key: str) -> None:
    if ctx.notes:
        st.info(f"**Speaker note:** {NOTES[key]}")


def _nav(i: int) -> None:
    back, nxt, _ = st.columns([1, 1, 6])
    # keys are per chapter: a key shared by every page confuses the widget state across a switch
    if i > 0 and back.button("← Back", key=f"back_{i}"):
        st.switch_page(ctx.pages[i - 1])
    if i < len(ctx.pages) - 1 and nxt.button("Next →", key=f"next_{i}", type="primary"):
        st.switch_page(ctx.pages[i + 1])


def _data(frame: pd.DataFrame, label: str = "Data behind this chart") -> None:
    with st.expander(label):
        st.dataframe(frame, hide_index=True, width="stretch")


def _chart(chart, **kwargs):
    return st.altair_chart(chart, width="stretch", **kwargs)


# ---------------------------------------------------------------------------
# 1. the problem
# ---------------------------------------------------------------------------

def problem() -> None:
    st.title("Some pull requests wait weeks for a first review")
    buckets = triage.wait_buckets(ctx.prs)
    quick = buckets.loc[buckets["bucket"].isin(triage.WAIT_BUCKETS[:2]), "share"].sum()
    stalled = buckets.loc[buckets["stalled"], "share"].sum()
    st.markdown(
        f"Of the {len(ctx.prs):,} pull requests opened in {len(triage.repos(ctx.prs))} projects "
        f"from January to June 2026, {quick:.0%} got a first review within a day, but "
        f"**{stalled:.0%} stalled**: no first review within 7 days, and most of those were closed "
        "without ever getting one. PRFlowPredict scores every PR the moment it is opened, so a "
        "maintainer can see which ones are at risk.")
    st.altair_chart(charts.wait_histogram(buckets, ctx.mode), width="content")
    st.caption("A first review is the first review or comment by a human other than the author, "
               "with a tie to the repo: the project's D5 label.")
    _data(buckets)
    _note("problem")
    _nav(0)


# ---------------------------------------------------------------------------
# 2. watch it work
# ---------------------------------------------------------------------------

def _triage_list(day) -> None:
    """The Phase 7 triage list for the chosen day, with its tally and caveats unchanged."""
    listing = triage.ranked(ctx.prs, ctx.repo, triage.moment(day), ctx.scenario)
    if listing.empty:
        note = "No PR in this repo was waiting for a first review at the start of that day."
        if day <= triage.EARLY_JANUARY:
            note += " Only PRs opened from 2026-01-01 have scores, so early January looks sparse."
        st.info(note)
        return
    stalled, k = triage.tally(listing)
    st.markdown(
        f"**Of the top {k} by risk, {stalled} stalled; of all {len(listing)} PRs waiting, "
        f"{int(listing['stalled'].sum())} did** (stalled: no first review within 7 days of "
        "opening). Most PRs still waiting at a given moment go on to stall, and any already "
        "waiting 7 days has stalled by definition, so this list cannot show how well the model "
        f"ranks. That is measured when PRs open, over every test PR: see the [README]({README}).")
    st.dataframe(
        listing, hide_index=True, width="stretch",
        column_order=["rank", "risk", "url", "title", "days_waited", "drivers", "outcome"],
        column_config={
            "rank": st.column_config.NumberColumn("Rank"),
            "risk": st.column_config.NumberColumn("Risk", format="%.2f"),
            "url": st.column_config.LinkColumn("PR", display_text=r"/pull/(\d+)$"),
            "title": st.column_config.TextColumn("Title"),
            "days_waited": st.column_config.NumberColumn("Days waited", format="%.1f"),
            "drivers": st.column_config.TextColumn("Top drivers"),
            "outcome": st.column_config.TextColumn("What happened"),
        })
    st.caption(
        "Risk is a ranking score, not a calibrated probability. Top drivers are the three features "
        "that moved this PR's score most, by exact TreeSHAP: model attribution, not cause (↑ pushed "
        "the risk up, ↓ down). Titles are as of data collection.")


def watch() -> None:
    st.title("Watch it work")
    st.markdown(
        "A replay of the 2026 test period PRFlowPredict was evaluated on, not live data. Each dot "
        f"is one pull request in **{ctx.repo}**, placed by the day it opened and the risk score the "
        "model gave it **when it was opened**. Scores are never updated as a PR waits. Only PRs "
        "opened from 1 January 2026 have scores, so older PRs still waiting are not shown. Drag "
        "the day to see who was waiting for a first review; click a dot to see why it scored as "
        "it did.")
    st.markdown(MODEL_NOTES[ctx.scenario].format(n=len(triage.repos(ctx.prs))))
    day = st.slider("Day", min_value=triage.FIRST_DAY, max_value=triage.LAST_DAY,
                    value=triage.DEFAULT_DAY, key="day", format="D MMM YYYY")
    at = triage.moment(day)
    frame = triage.timeline(ctx.prs, ctx.repo, ctx.scenario, at)
    event = _chart(charts.timeline(frame, at, ctx.mode), on_select="rerun", key="timeline")
    picked = triage.selected_pr_id(event)
    if picked:
        st.session_state["selected_pr"], st.session_state["pr_ref"] = picked, ""
        row = frame[frame["pr_id"] == picked].iloc[0]
        st.success(f"Selected #{row['number']}: {triage.md_escape(row['title'])}. "
                   "The next chapter shows why it scored as it did.")
    _triage_list(day)
    _data(frame, "Data behind the timeline")
    _note("watch")
    _nav(1)


# ---------------------------------------------------------------------------
# 3. why it decides
# ---------------------------------------------------------------------------

def _lookup() -> None:
    parsed = triage.parse_pr_ref(st.session_state.get("pr_ref", ""), ctx.repo)
    hit = triage.find_pr(ctx.prs, *parsed) if parsed else ctx.prs.iloc[0:0]
    if len(hit):
        st.session_state["selected_pr"] = hit["pr_id"].iloc[0]


def _pick_random() -> None:
    pr = triage.random_pr(ctx.prs).iloc[0]
    st.session_state["pr_ref"] = f"{pr['repo']}#{int(pr['number'])}"
    st.session_state["selected_pr"] = pr["pr_id"]


def _current_pr() -> pd.DataFrame:
    """The selected PR; with none selected, the riskiest PR waiting on the default day in the
    chosen repo, or the repo's riskiest PR if none was waiting."""
    chosen = ctx.prs[ctx.prs["pr_id"] == st.session_state.get("selected_pr")]
    if len(chosen):
        return chosen
    score = triage.SCORES[ctx.scenario]
    repo = ctx.prs[ctx.prs["repo"] == ctx.repo]
    waiting = triage.awaiting_review(repo, triage.moment(triage.DEFAULT_DAY))
    pool = waiting if len(waiting) else repo
    return pool.sort_values([score, "number"], ascending=[False, True], kind="mergesort").head(1)


def _card(pr: pd.DataFrame) -> None:
    row = pr.iloc[0]
    st.markdown(f"#### {triage.md_escape(row['title'])}")
    st.markdown(
        f"[{row['repo']}#{row['number']}]({row['url']}) · opened "
        f"{row['created_at']:%Y-%m-%d %H:%M} UTC · what happened: **{triage.outcome(pr).iloc[0]}**")
    for col, (label, sc) in zip(st.columns(2), MODELS.items()):
        share, n = triage.rank_in_repo(ctx.prs, pr, sc)
        col.metric(label, f"{row[triage.SCORES[sc]]:.2f}")
        col.markdown(f"Higher than {share:.0%} of this repo's other {n:,} replayed PRs.")


def why() -> None:
    st.title("Why it decides")
    st.markdown(
        "Pick any replayed PR: paste its GitHub link, type `owner/repo#123` or a number in the repo "
        "chosen in the sidebar, click a dot in *Watch it work*, or let **Random PR** pick one.")
    box, button = st.columns([5, 1], vertical_alignment="bottom")
    ref = box.text_input("PR", key="pr_ref", on_change=_lookup,
                         placeholder=f"https://github.com/{ctx.repo}/pull/...")
    button.button("Random PR", key="random_pr", on_click=_pick_random)
    parsed = triage.parse_pr_ref(ref, ctx.repo) if ref.strip() else None
    if ref.strip() and (parsed is None or triage.find_pr(ctx.prs, *parsed).empty):
        st.warning(
            "Not in the replay. The lookup covers the PRs opened from 1 January to 30 June 2026 in "
            f"the {len(triage.repos(ctx.prs))} repos; try one from the triage list, or press Random PR.")
    else:
        if parsed:
            _lookup()
        pr = _current_pr()
        _card(pr)
        name = MODEL_NAMES[ctx.scenario]
        rows, base = triage.why(pr, ctx.scenario, ctx.features)
        st.subheader(f"Why the {name} model scored it {pr[triage.SCORES[ctx.scenario]].iloc[0]:.2f}")
        _chart(charts.why_bars(rows, ctx.mode))
        st.caption(
            "Exact TreeSHAP: each bar is how far that feature pushed this PR's score on the model's "
            f"log-odds scale, starting from the model's average of {base:+.2f}; together they add up "
            "to the score shown. Model attribution, not cause. Risk is a ranking score, not a "
            "calibrated probability.")
        _data(rows)
    st.subheader("What the model leans on overall")
    imp = triage.importance(ctx.scenario, ctx.features)
    _chart(charts.importance_bars(imp, mode=ctx.mode))
    st.caption("Each feature's share of the model's total attribution over the test PRs (Phase 6).")
    _note("why")
    _nav(2)


# ---------------------------------------------------------------------------
# 4. test yourself
# ---------------------------------------------------------------------------

def _new_round(game: dict) -> None:
    game.update(ids=list(triage.deal(ctx.prs, random.Random())["pr_id"]), pick=None, revealed=False)


def _choose(i: int) -> None:
    st.session_state["game"]["pick"] = i


def _reveal() -> None:
    game = st.session_state["game"]
    round_ = ctx.prs.set_index("pr_id").loc[game["ids"]].reset_index()
    game["model_pick"] = triage.model_pick(round_, ctx.scenario)
    game["revealed"] = True
    game["played"] += 1
    game["you"] += int(round_["is_slow"].iloc[game["pick"]])
    game["model"] += int(round_.set_index("pr_id").loc[game["model_pick"], "is_slow"])


def _deal_again() -> None:
    _new_round(st.session_state["game"])


def test_yourself() -> None:
    st.title("Test yourself")
    st.markdown(
        "Four pull requests opened in the same project in the same week. **Exactly one of them "
        "stalled**: no first review within 7 days. Each card shows only what was known when it "
        "opened. Which one stalled?")
    game = st.session_state.setdefault("game", {"ids": None, "pick": None, "revealed": False,
                                                "played": 0, "you": 0, "model": 0})
    if game["ids"] is None:
        _new_round(game)
    round_ = ctx.prs.set_index("pr_id").loc[game["ids"]].reset_index()
    meta = {f["feature"]: f for f in ctx.features}
    for i, (col, (_, pr)) in enumerate(zip(st.columns(4), round_.iterrows())):
        with col.container(border=True):
            st.markdown(f"**PR {i + 1}** · {pr['repo']}")
            st.markdown(triage.md_escape(pr["title"]))
            st.markdown("\n".join(
                f"- {meta[f]['label']}: **{triage.format_value(pr[f'x__{f}'], meta[f]['dtype'], meta[f]['rate'])}**"
                for f in GAME_FACTS))
            if not game["revealed"]:
                st.button("This one", key=f"pick_{i}", on_click=_choose, args=(i,),
                          type="primary" if game["pick"] == i else "secondary")
            else:
                marks = [m for m, on in (("your pick", game["pick"] == i),
                                         ("the model's pick", game.get("model_pick") == pr["pr_id"])) if on]
                verdict = "**Stalled**" if pr["is_slow"] else "Reviewed within 7 days"
                st.markdown(f"{verdict} · risk {pr[triage.SCORES[ctx.scenario]]:.2f} · "
                            f"[#{pr['number']}]({pr['url']})" + (f"  \n← {', '.join(marks)}" if marks else ""))
    if not game["revealed"]:
        if game["pick"] is not None:
            st.button("Reveal", key="reveal", type="primary", on_click=_reveal)
    else:
        you_right = bool(round_["is_slow"].iloc[game["pick"]])
        model_right = bool(round_.set_index("pr_id").loc[game["model_pick"], "is_slow"])
        st.markdown(f"You picked PR {game['pick'] + 1}: {'✓ right' if you_right else '✗ wrong'}. "
                    f"The model picked PR {list(round_['pr_id']).index(game['model_pick']) + 1}: "
                    f"{'✓ right' if model_right else '✗ wrong'}.")
        st.markdown(f"This session: you {game['you']} of {game['played']}, the model "
                    f"{game['model']} of {game['played']}.")
        st.button("Deal again", key="deal", type="primary", on_click=_deal_again)
    st.markdown(
        f"Over every round this game can deal, the {MODEL_NAMES[ctx.scenario]} model picks the PR "
        f"that stalled **{ctx.hit_rates[ctx.scenario]:.0%}** of the time; a random guess gets "
        f"**{1 / triage.GAME_SIZE:.0%}**. Rounds are drawn like this: a project and week at random, "
        "then one PR from it that stalled and three that did not.")
    _note("test")
    _nav(3)


# ---------------------------------------------------------------------------
# 5. does it transfer?
# ---------------------------------------------------------------------------

def transfer() -> None:
    st.title("Does it transfer to a new project?")
    t = ctx.results["text"]
    st.markdown(
        "**Within a project, it works.** On repos it was trained on, the model's precision on each "
        f"repo's 10 riskiest PRs is {t['a_p10']} (95% interval, resampling repos), against "
        f"{t['a_baseline_p10']} for the trailing-rate baseline and {t['a_random_p10']} for a random pick.")
    _chart(charts.p10_chart(ctx.results, ctx.mode))
    st.markdown(
        "**On repos it has never seen, it needs the repo's slow-rate history.** With it, the model "
        f"beats the baseline (AUC-PR {t['b_full_vs_baseline']}); without those four features it "
        f"falls below: {t['b_nlr_vs_baseline']}. Each dot is one held-out fold of repos.")
    _chart(charts.results_chart(ctx.results, ctx.mode))
    st.markdown("Try it: switch the sidebar model to **Unseen repo** and go back to *Watch it work* "
                f"or *Test yourself*. The full evaluation is in the [report]({REPORT}).")
    _data(ctx.results["folds"], "Data behind the AUC-PR chart")
    _note("transfer")
    _nav(4)


# ---------------------------------------------------------------------------
# 6. honest limits
# ---------------------------------------------------------------------------

def limits() -> None:
    st.title("Honest limits")
    st.markdown(
        "- **The cold-start bar was never measured.** The project plan set a C-index target for a "
        "survival model on unseen repos; that phase was not built.\n"
        f"- **{len(triage.repos(ctx.prs))} repos is a small sample** for any claim about projects in "
        "general, and results swing widely between held-out folds.\n"
        "- **Repo attributes are 2026 snapshots** (maintainer counts, CODEOWNERS and the like), "
        "applied to earlier PRs.\n"
        "- **A list of PRs still waiting cannot measure ranking.** Most of them go on to stall; the "
        "ranking is measured when PRs open, over every test PR.")
    st.markdown(f"Everything here is in the [README]({README}) and the [report]({REPORT}), where a "
                "test checks every number against the committed data.")
    _note("limits")
    _nav(5)
```

Create the six page files. Each holds a docstring line, `import chapters`, and a call to its chapter function:

```bash
mkdir -p demo/views
for p in problem watch why test_yourself transfer limits; do printf '"""A chapter page: the content lives in chapters.py."""\nimport chapters\n\nchapters.%s()\n' "$p" > demo/views/$p.py; done
cat demo/views/watch.py
```

Expected output of the `cat`:

```
"""A chapter page: the content lives in chapters.py."""
import chapters

chapters.watch()
```

Replace `demo/app.py` with:

```python
"""PRFlowPredict demo: an interactive walkthrough of the evaluated 2026 test period.

    pip install -r demo/requirements.txt
    streamlit run demo/app.py

A thin shell: page config, cached data, the sidebar and the six chapters in chapters.py. It
hard-codes no result number: every number shown is computed from committed files, and the
results chapter's numbers are tested against the README's."""
import importlib
import sys
from pathlib import Path

import streamlit as st

HERE = str(Path(__file__).parent)
if HERE not in sys.path:
    sys.path.insert(0, HERE)            # triage.py, charts.py and chapters.py sit next to this file
import triage  # noqa: E402
import charts  # noqa: E402
import chapters  # noqa: E402

# Streamlit Community Cloud re-runs this script when the repo updates but keeps imported modules
# in memory, so a new app.py could meet old helpers. Reload them every run, dependencies first.
triage = importlib.reload(triage)
charts = importlib.reload(charts)
chapters = importlib.reload(chapters)

st.set_page_config(page_title="PRFlowPredict demo", layout="wide")


@st.cache_data
def _data():
    prs = triage.load()
    return (prs, triage.load_features(), triage.results(),
            {sc: triage.game_hit_rate(prs, sc) for sc in triage.SCORES})


prs, features, results, hit_rates = _data()
repo_names = triage.repos(prs)
start = triage.default_repo(prs, triage.moment(triage.DEFAULT_DAY))

st.sidebar.markdown("### PRFlowPredict")
scenario = chapters.MODELS[st.sidebar.radio("Model", list(chapters.MODELS))]
repo = st.sidebar.selectbox("Repo", repo_names, index=repo_names.index(start))
notes = st.sidebar.toggle("Speaker notes", value=False)
theme = getattr(st.context, "theme", None)
mode = "dark" if getattr(theme, "type", None) == "dark" else "light"

chapters.ctx = chapters.Context(prs, features, results, hit_rates, scenario, repo, notes, mode)
# One small page file per chapter (views/), each calling its function in chapters.py: file pages
# keep their place between runs in Streamlit's AppTest, so the tests can drive every chapter.
pages = [st.Page("views/problem.py", title="1. The problem", url_path="problem", default=True),
         st.Page("views/watch.py", title="2. Watch it work", url_path="watch-it-work"),
         st.Page("views/why.py", title="3. Why it decides", url_path="why"),
         st.Page("views/test_yourself.py", title="4. Test yourself", url_path="test-yourself"),
         st.Page("views/transfer.py", title="5. Does it transfer?", url_path="does-it-transfer"),
         st.Page("views/limits.py", title="6. Honest limits", url_path="honest-limits")]
chapters.ctx.pages = pages
st.navigation(pages).run()
```

- [ ] **Step 5: Run the tests and the suite, then look at the app**

Run `python -m pytest tests/test_demo_app.py -q` and expect **64 passed**. Then run `python -m pytest tests -q` and expect **530 passed**.

Then serve the app: `python -m streamlit run demo/app.py --server.headless true --server.port 8599`. Check that `curl -s http://localhost:8599/_stcore/health` prints `ok`. **Open http://localhost:8599 and click through all six chapters,** in both light and dark theme (Settings in the ⋮ menu):
- in chapter 2, click a timeline dot, then press Next;
- in chapter 4, play one round of the game.

Stop the server afterwards.

- [ ] **Step 6: Commit**

```bash
git add demo/chapters.py demo/views demo/app.py demo/requirements.txt requirements.txt tests/test_demo_app.py
git commit -m "feat(demo): six-chapter walkthrough -- timeline, why chart, game, results, speaker notes

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 7: Mutation check (after the commit, restoring with git)**

For each mutation: apply it, run the named test and confirm it FAILS, then run `git checkout -- <file>` and confirm the tree is clean. All target tests are in `tests/test_demo_app.py`.
- `demo/chapters.py`: `st.switch_page(ctx.pages[i - 1])` → `st.switch_page(ctx.pages[max(i - 2, 0)])` → `test_back_opens_the_previous_chapter`
- `demo/chapters.py`: `f"that stalled **{ctx.hit_rates[ctx.scenario]:.0%}** of the time` → use `ctx.hit_rates['B']` → `test_the_game_shows_its_exact_hit_rate_against_a_random_guess`
- `demo/chapters.py`: `game["played"] += 1` → `game["played"] += 2` → `test_the_game_deals_four_cards_and_reveal_scores_the_round`
- `demo/app.py`: `notes = st.sidebar.toggle("Speaker notes", value=False)` → `value=True` → `test_speaker_notes_are_hidden_by_default_and_shown_when_toggled`
- `demo/app.py`: delete the line `charts = importlib.reload(charts)` → `test_the_app_uses_the_current_helpers_even_when_old_copies_are_cached`

---

### Task 5: Docs

**Files:**
- Modify: `README.md`, `docs/design/specs/2026-10-03-phase7-demo-design.md`

- [ ] **Step 1: Apply the edits with this script**

Save it to a scratch file outside the repo, for example `$TMP/story_docs.py`, and run `python "$TMP/story_docs.py"` from the repo root.

```python
"""Demo-story docs: the README's demo paragraph and a pointer in the Phase 7 spec."""
import io


def edit(path, old, new):
    t = io.open(path, encoding="utf-8", newline="").read()
    assert t.count(old) == 1, f"expected exactly one match in {path} for {old[:60]!r}, found {t.count(old)}"
    io.open(path, "w", encoding="utf-8", newline="").write(t.replace(old, new))


edit("README.md",
     """A small Streamlit app replays the 2026 test period: pick a repo and a day to see the PRs that
were waiting for a first review, ranked by the risk score each got when it was opened, next to
what actually happened. A toggle swaps in the model that never saw that repo. A **Look up a PR**
tab shows any replayed PR's scores from both models, its drivers and what happened, and
**Random PR** picks one for you.
""",
     """A six-chapter Streamlit walkthrough of the 2026 test period: how long PRs wait for a first
review; one repo's PRs on a timeline, scored when they opened (click any to see why it scored as
it did); a guess-the-stall game against the model; and the cold-start result, with a switch to the
model that never saw the repo.
""")

edit("docs/design/specs/2026-10-03-phase7-demo-design.md",
     "**Empty list.** When no PR is awaiting review at the moment, a message says so.",
     "**Restructured 2026-10-04:** the app is now a six-chapter walkthrough; see\n"
     "`docs/design/specs/2026-10-04-demo-story-design.md`.\n\n"
     "**Empty list.** When no PR is awaiting review at the moment, a message says so.")
print("docs updated")
```

- [ ] **Step 2: Verify and commit**

Run `git diff --stat`. Expect two files: `README.md` (a 5-line paragraph replaced by 4 lines) and the Phase 7 spec (3 lines added). Run `python -m pytest tests -q` and expect **530 passed**. The claims test re-checks the README.

```bash
git add README.md docs/design/specs/2026-10-03-phase7-demo-design.md
git commit -m "docs(demo): README and Phase 7 spec point to the six-chapter walkthrough

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

## Verification (spec §9)

1. `python -m pytest tests -q` gives **530 passed**, pristine. Every mutation in Tasks 1–4 was caught and restored with `git checkout`.
2. `python build_demo_data.py` reproduces both committed data files byte for byte, and the gitignored inputs' hashes are unchanged.
3. The served app answers `/_stcore/health` with `ok`. All six chapters were clicked through in light and dark theme, including a timeline click and a game round.
4. No phase document, phase generator or gitignored artifact was touched, and `data/experiments.csv` still has 36 rows.
5. After merge, Streamlit Cloud redeploys from `main`. The reload fix covers the new modules, so no reboot is needed, but a reboot is harmless if the page looks stale.

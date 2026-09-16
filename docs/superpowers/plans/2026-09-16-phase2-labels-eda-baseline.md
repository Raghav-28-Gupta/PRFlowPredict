# Phase 2 — Labels, EDA, Baseline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the target variable, the evaluation harness (splits + metrics), the point-in-time replay engine, and the trailing-90-day baseline — then produce the Phase 2 EDA report with a pre-registered gate.

**Architecture:** Importable pandas/numpy modules with no side effects, each with one responsibility, plus thin `main()` scripts. `labels.py` becomes the single implementation of the target (`gate_report.py` is refactored to import it). `replay.py` makes temporal leakage structural: `features_at(t)` can only see rows with `created_at < t` via bisect. Everything is tested on hand-built toy data with known answers before touching real data.

**Tech Stack:** Python 3.13, pandas 2.3, numpy, pyarrow, scikit-learn (GroupKFold, average_precision_score), lifelines (Kaplan–Meier), matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-phase2-labels-eda-baseline-design.md`

## Global Constraints

- All timestamps are tz-aware UTC in pandas; convert to naive `datetime64[ns]` only inside numpy hot paths, and only via the `_ns()` helper defined in Task 6.
- Modelling window: `2024-01-01T00:00:00Z` ≤ `created_at` ≤ `2026-06-30T23:59:59Z`. Scenario A cutoff: `2026-01-01T00:00:00Z`.
- Label: `threshold_h = 168.0`, `censor_h = 720.0`. Primary definition `D5`.
- Seed everywhere: `20260912`.
- Per-repo training cap: 5% of the uncapped training-row total, computed once, never iterated.
- Bot-authored PRs (`pr_tier2.author_is_bot`) are excluded from all splits.
- No check on the label distribution anywhere in cohort QC. Structural checks only.
- Repo directory naming: `owner/name` → `owner__name` under `data/processed/`.
- Tests live in `tests/`, run with `python -m pytest tests -q`. Every module gets tests on toy data before real data.
- **Commits are the user's call.** Steps say `git add`; do not commit unless asked.
- Data files (`data/raw`, `data/processed`, `experiments.csv` outputs under `data/`) are gitignored; `figures/` and `docs/` are tracked.

---

## File structure

| File | Responsibility |
|---|---|
| `requirements.txt` (modify) | add `lifelines`, `scikit-learn`, `pytest` |
| `tests/conftest.py` (create) | shared toy fixtures: 4-PR label toy, 5-PR replay toy |
| `tracking.py` (create) | append-only `experiments.csv` writer with a fixed column set |
| `load.py` (create) | load one repo's parquet tables; concat across repos |
| `labels.py` (create) | `Definition`, `DEFINITIONS`, `KNOWN_BOTS`, `first_human_event`, `label`, `label_all` |
| `gate_report.py` (modify) | delete inlined label logic; import `labels` |
| `cohort_qc.py` (create) | structural checks → `data/cohort/kept.json`; look-at-these list |
| `replay.py` (create) | `History`, `features_at` |
| `metrics.py` (create) | `precision_at_k`, `auc_pr`, `cluster_bootstrap` |
| `splits.py` (create) | `prepare_rows`, `cap_per_repo`, `scenario_a`, `scenario_b`, `row_share` |
| `baseline.py` (create) | trailing-rate baseline on A and B; logs to `experiments.csv` |
| `eda_report.py` (create) | `docs/phase2_eda.md` + `figures/`; Phase 2 gate |

---

### Task 1: Dependencies, test scaffold, and `tracking.py`

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`, `tests/conftest.py`, `tests/test_tracking.py`
- Create: `tracking.py`

**Interfaces:**
- Produces: `tracking.log(row: dict, path: Path = EXPERIMENTS) -> None`; `tracking.COLUMNS: list[str]`; fixtures `label_toy()` and `replay_toy()` used by Tasks 3 and 6.

- [ ] **Step 1: Install and pin dependencies**

Run:
```bash
cd C:/Users/ragha/project/PRFlowPredict && python -m pip install lifelines scikit-learn pytest && python -m pip list 2>/dev/null | grep -iE "^(lifelines|scikit-learn|pytest|scipy) "
```
Then append to `requirements.txt`, using the exact versions printed:
```
# Phase 2
scikit-learn==<printed>   # GroupKFold, average_precision_score
lifelines==<printed>      # Kaplan-Meier with CI bands (EDA)
pytest==<printed>         # tests/
```

- [ ] **Step 2: Write the conftest with both toy fixtures**

`tests/__init__.py` is empty. `tests/conftest.py`:

```python
"""Toy data with hand-computed answers. Every module is proven here before it
touches real data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

BASE = pd.Timestamp("2025-01-01T00:00:00Z")


def h(hours: float) -> pd.Timestamp:
    return BASE + pd.Timedelta(hours=hours)


EMPTY_STREAM_COLS = ["pr_id", "created_at", "published_at", "author_login",
                     "author_typename", "author_association", "is_minimized"]


@pytest.fixture
def label_toy():
    """Four PRs by alice, all opened at BASE.

    A: bob (MEMBER) reviews, submitted +2h            -> reviewed at 2h under every def
    B: 'spammer' (NONE) issue-comments at +1h,
       bob reviews submitted +200h                     -> D3: 1h (not slow); D5: 200h (slow)
    C: alice self-comments +5h, dependabot[bot] +6h    -> never reviewed under every def
    D: bob's review is PENDING (submitted_at null)     -> never reviewed under every def
    """
    prs = pd.DataFrame({
        "repo": ["r"] * 4,
        "pr_id": ["A", "B", "C", "D"],
        "created_at": [BASE] * 4,
        "author_login": ["alice"] * 4,
        "author_is_bot": [False] * 4,
    })
    reviews = pd.DataFrame({
        "pr_id": ["A", "B", "D"],
        "created_at": [h(1), h(199), h(3)],
        "submitted_at": [h(2), h(200), pd.NaT],
        "state": ["APPROVED", "APPROVED", "PENDING"],
        "author_login": ["bob"] * 3,
        "author_typename": ["User"] * 3,
        "author_association": ["MEMBER"] * 3,
    })
    issue_comments = pd.DataFrame({
        "pr_id": ["B", "C", "C"],
        "created_at": [h(1), h(5), h(6)],
        "published_at": [h(1), h(5), h(6)],
        "author_login": ["spammer", "alice", "dependabot[bot]"],
        "author_typename": ["User", "User", "Bot"],
        "author_association": ["NONE", "OWNER", "NONE"],
        "is_minimized": [False, False, False],
    })
    thread_comments = pd.DataFrame(columns=EMPTY_STREAM_COLS)
    return prs, {"reviews": reviews, "thread_comments": thread_comments,
                 "issue_comments": issue_comments}


@pytest.fixture
def replay_toy():
    """Five PRs in one repo, window_start 2024-01-01. Hand-computed at
    t = 2024-04-21T00:00Z (alpha=5, g=0.5):

      P0 created 2023-06-01, closed 2024-06-01, no label (pre-window) -> backlog only
      P1 created 2024-03-01, open,  first_event 2024-03-02, is_slow 0
      P2 created 2024-03-10, closed 2024-03-20, never reviewed, is_slow 1
      P3 created 2024-04-15, open,  first_event 2024-04-16, is_slow 0
      P4 created 2024-04-19, open,  first_event NaT, is_slow 1   <- 2 days old: NOT resolvable

      backlog @t     = P0, P1, P3, P4 open            = 4
      trailing 90d   = lo 2024-01-22 -> P1..P4 in window; labelled: all 4
      resolvable     = thr t-168h = 2024-04-14:
                       P1 (3-01<=thr) yes, P2 yes, P3 (4-15>thr but fe 4-16<t) yes, P4 no
                       -> k=3, n_slow=1 -> (1 + 5*0.5)/(3+5) = 0.4375
      complete       = lo >= window_start -> True
    """
    def ts(s): return pd.Timestamp(s, tz="UTC")
    tier1 = pd.DataFrame({
        "repo": ["r"] * 5,
        "pr_id": ["P0", "P1", "P2", "P3", "P4"],
        "created_at": [ts("2023-06-01"), ts("2024-03-01"), ts("2024-03-10"),
                       ts("2024-04-15"), ts("2024-04-19")],
        "closed_at": [ts("2024-06-01"), pd.NaT, ts("2024-03-20"), pd.NaT, pd.NaT],
        "author_login": ["u0", "u1", "u2", "u1", "u3"],
    })
    labels = pd.DataFrame({
        "pr_id": ["P1", "P2", "P3", "P4"],
        "first_event_at": [ts("2024-03-02"), pd.NaT, ts("2024-04-16"), pd.NaT],
        "is_slow": [False, True, False, True],
    })
    return tier1, labels
```

- [ ] **Step 3: Write the failing tracking test**

`tests/test_tracking.py`:
```python
import csv
import tracking


def test_log_creates_header_then_appends(tmp_path):
    p = tmp_path / "experiments.csv"
    tracking.log({"scenario": "A", "model": "baseline", "auc_pr": 0.5}, path=p)
    tracking.log({"scenario": "B", "model": "baseline", "auc_pr": 0.4}, path=p)
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    assert [r["scenario"] for r in rows] == ["A", "B"]
    assert list(rows[0].keys()) == tracking.COLUMNS
    assert rows[0]["date"]  # filled in automatically
    assert rows[0]["fold"] == ""  # unspecified columns are blank, never missing


def test_unknown_column_is_rejected(tmp_path):
    import pytest
    with pytest.raises(KeyError):
        tracking.log({"scenario": "A", "bogus": 1}, path=tmp_path / "e.csv")
```

- [ ] **Step 4: Run to verify it fails**

Run: `python -m pytest tests/test_tracking.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tracking'`

- [ ] **Step 5: Implement `tracking.py`**

```python
"""Append-only experiment log (blueprint §5): every run, not just the best one.

Fixed column set so the CSV stays machine-readable across phases. Add a column here
if a later phase needs one; never write ad-hoc keys."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENTS = Path(__file__).parent / "data" / "experiments.csv"

COLUMNS = [
    "date", "scenario", "fold", "model", "features", "params",
    "n_train", "n_test", "n_test_repos",
    "precision_at_10", "p10_ci_lo", "p10_ci_hi", "base_rate_p10",
    "auc_pr", "base_rate", "notes",
]


def log(row: dict, path: Path = EXPERIMENTS) -> None:
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise KeyError(f"unknown experiment columns: {sorted(unknown)}")
    full = {c: "" for c in COLUMNS}
    full["date"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    full.update({k: ("" if v is None else v) for k, v in row.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(full)
```

- [ ] **Step 6: Run to verify it passes**

Run: `python -m pytest tests/test_tracking.py -q`
Expected: `2 passed`

- [ ] **Step 7: Stage**

```bash
git add requirements.txt tests/ tracking.py
```

---

### Task 2: `load.py`

**Files:**
- Create: `load.py`, `tests/test_load.py`

**Interfaces:**
- Produces: `load.TABLES: tuple[str, ...]`; `load.repo_dir(repo: str) -> Path`; `load.load_repo(repo: str) -> dict[str, pd.DataFrame]` (missing tables → empty DataFrame with no columns); `load.load_all(repos: list[str]) -> dict[str, pd.DataFrame]` (concatenated, `repo` column present on every row).

- [ ] **Step 1: Write the failing test**

`tests/test_load.py`:
```python
import pandas as pd
import load


def test_repo_dir_naming():
    assert load.repo_dir("owner/name").name == "owner__name"


def test_load_all_concats_and_tolerates_missing_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(load, "PROCESSED", tmp_path)
    for repo in ("a/x", "b/y"):
        d = tmp_path / repo.replace("/", "__")
        d.mkdir()
        pd.DataFrame({"repo": [repo], "pr_id": [repo + "#1"]}).to_parquet(d / "pr_tier2.parquet")
    # only a/x has reviews
    pd.DataFrame({"repo": ["a/x"], "pr_id": ["a/x#1"]}).to_parquet(
        tmp_path / "a__x" / "reviews.parquet")

    frames = load.load_all(["a/x", "b/y"])
    assert sorted(frames["pr_tier2"]["repo"]) == ["a/x", "b/y"]
    assert list(frames["reviews"]["repo"]) == ["a/x"]
    assert frames["timeline"].empty                 # present, empty, not KeyError
    assert set(frames) == set(load.TABLES)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_load.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'load'`

- [ ] **Step 3: Implement `load.py`**

```python
"""Load parse.py output for one or many repos.

Missing tables come back as EMPTY DataFrames rather than KeyErrors: a repo with no
formal reviews has no reviews.parquet, and that is data, not an error."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROCESSED = Path(__file__).parent / "data" / "processed"

TABLES = ("pr_tier1", "pr_tier2", "reviews", "thread_comments", "issue_comments",
          "timeline", "commits", "repo_meta")


def repo_dir(repo: str) -> Path:
    return PROCESSED / repo.replace("/", "__")


def load_repo(repo: str) -> dict[str, pd.DataFrame]:
    d = repo_dir(repo)
    out = {}
    for t in TABLES:
        p = d / f"{t}.parquet"
        out[t] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    return out


def load_all(repos: list[str]) -> dict[str, pd.DataFrame]:
    parts: dict[str, list[pd.DataFrame]] = {t: [] for t in TABLES}
    for repo in repos:
        for t, df in load_repo(repo).items():
            if not df.empty:
                parts[t].append(df)
    return {t: (pd.concat(v, ignore_index=True) if v else pd.DataFrame())
            for t, v in parts.items()}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_load.py -q`
Expected: `2 passed`

- [ ] **Step 5: Stage**

```bash
git add load.py tests/test_load.py
```

---

### Task 3: `labels.py`

**Files:**
- Create: `labels.py`, `tests/test_labels.py`

**Interfaces:**
- Consumes: `label_toy` fixture (Task 1).
- Produces:
  - `labels.Definition(name, streams, exclude_associations=frozenset(), exclude_minimized=False)` (frozen dataclass)
  - `labels.DEFINITIONS: dict[str, Definition]` with keys `"D1".."D5"`; `labels.PRIMARY = "D5"`; `labels.KNOWN_BOTS: frozenset[str]`
  - `labels.is_bot(login: pd.Series, typename: pd.Series) -> pd.Series[bool]`
  - `labels.first_human_event(prs, streams: dict[str, pd.DataFrame], definition: Definition) -> pd.Series` indexed by `pr_id`, values tz-aware UTC timestamps
  - `labels.label(prs, first_event: pd.Series, threshold_h=168.0, censor_h=720.0) -> pd.DataFrame` with columns `repo, pr_id, created_at, first_event_at, wait_h, never_reviewed_30d, is_slow, wait_h_censored, event_observed`
  - `labels.label_all(prs, streams, **kw) -> dict[str, pd.DataFrame]` keyed by definition name

- [ ] **Step 1: Write the failing tests**

`tests/test_labels.py`:
```python
import pandas as pd
import pytest
import labels
from tests.conftest import h


def fe(prs, streams, name):
    return labels.first_human_event(prs, streams, labels.DEFINITIONS[name])


def test_is_bot_three_ways():
    login = pd.Series(["dependabot[bot]", "Dependabot", "alice", None, "renovate"])
    typename = pd.Series(["Bot", "User", "User", None, "User"])
    assert labels.is_bot(login, typename).tolist() == [True, True, False, False, True]


def test_d3_counts_spam_comment_d5_does_not(label_toy):
    prs, streams = label_toy
    d3, d5 = fe(prs, streams, "D3"), fe(prs, streams, "D5")
    assert d3["B"] == h(1)      # spammer's NONE comment counts under the blueprint rule
    assert d5["B"] == h(200)    # excluded under D5; bob's review is first


def test_pending_review_and_self_and_bot_never_count(label_toy):
    prs, streams = label_toy
    d3 = fe(prs, streams, "D3")
    assert "C" not in d3.index  # only alice (self) and dependabot[bot]
    assert "D" not in d3.index  # PENDING review has null submitted_at
    assert d3["A"] == h(2)      # submitted_at, not created_at (+1h)


def test_d1_reviews_only(label_toy):
    prs, streams = label_toy
    d1 = fe(prs, streams, "D1")
    assert set(d1.index) == {"A", "B"}
    assert d1["B"] == h(200)


def test_label_columns_and_rule(label_toy):
    prs, streams = label_toy
    out = labels.label(prs, fe(prs, streams, "D5")).set_index("pr_id")
    assert out.loc["A", "is_slow"] == False and out.loc["A", "wait_h"] == 2
    assert out.loc["B", "is_slow"] == True and out.loc["B", "wait_h"] == 200
    assert out.loc["C", "never_reviewed_30d"] == True and out.loc["C", "is_slow"] == True
    assert pd.isna(out.loc["C", "wait_h"])
    assert out.loc["C", "wait_h_censored"] == 720 and out.loc["C", "event_observed"] == False
    assert out.loc["A", "event_observed"] == True
    assert set(out.columns) >= {"repo", "created_at", "first_event_at", "wait_h",
                                "never_reviewed_30d", "is_slow", "wait_h_censored",
                                "event_observed"}


def test_wait_beyond_censor_is_never_reviewed(label_toy):
    prs, streams = label_toy
    fe_ = pd.Series({"A": h(800)})   # reviewed, but after the 30-day window
    out = labels.label(prs, fe_).set_index("pr_id")
    assert out.loc["A", "never_reviewed_30d"] == True
    assert out.loc["A", "wait_h_censored"] == 720


def test_label_all_has_all_definitions(label_toy):
    prs, streams = label_toy
    out = labels.label_all(prs, streams)
    assert set(out) == {"D1", "D2", "D3", "D4", "D5"}
    assert labels.PRIMARY in out
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_labels.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'labels'`

- [ ] **Step 3: Implement `labels.py`**

```python
"""The target variable. This is the ONLY implementation of it in the project.

Phase 1 captured three review streams separately and deferred the question of what
counts as a "review" because the blueprint's rule ("first non-author, non-bot review OR
comment") was shown to count spam comments on anthropics/skills. Phase 2 answers it
with a pre-registered choice: D5 is primary, D3 is reported alongside.

Every definition is always computed, so the sensitivity table is a by-product."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Belt and braces on top of __typename == "Bot" and the "[bot]" suffix: GitHub Actions
# tokens and self-hosted bots often appear as ordinary Users. Extend, don't hide.
KNOWN_BOTS = frozenset({
    "dependabot", "renovate", "github-actions", "codecov", "coveralls", "sonarcloud",
    "netlify", "vercel", "pre-commit-ci", "allcontributors", "stale", "mergify",
    "copilot",
})

ALL_STREAMS = ("reviews", "thread_comments", "issue_comments")


@dataclass(frozen=True)
class Definition:
    name: str
    streams: tuple[str, ...]
    exclude_associations: frozenset[str] = frozenset()
    exclude_minimized: bool = False


DEFINITIONS: dict[str, Definition] = {
    "D1": Definition("D1", ("reviews",)),
    "D2": Definition("D2", ("reviews", "thread_comments")),
    "D3": Definition("D3", ALL_STREAMS),                       # blueprint rule
    "D4": Definition("D4", ALL_STREAMS, exclude_minimized=True),
    "D5": Definition("D5", ALL_STREAMS, exclude_associations=frozenset({"NONE"})),
}
PRIMARY = "D5"


def is_bot(login: pd.Series, typename: pd.Series) -> pd.Series:
    low = login.fillna("").astype(str).str.lower()
    stem = low.str.replace(r"\[bot\]$", "", regex=True)
    return (typename == "Bot") | low.str.endswith("[bot]") | stem.isin(KNOWN_BOTS)


def visible_at(stream: str, df: pd.DataFrame) -> pd.Series:
    """When the PR author could SEE the event.

    reviews: submitted_at -- the connection's sort key (verified), null for PENDING.
    comments: published_at, falling back to created_at."""
    if stream == "reviews":
        return df["submitted_at"]
    return df["published_at"].fillna(df["created_at"])


def eligible_events(prs: pd.DataFrame, streams: dict[str, pd.DataFrame],
                    definition: Definition) -> pd.DataFrame:
    parts = []
    for name in definition.streams:
        df = streams.get(name)
        if df is None or df.empty:
            continue
        parts.append(pd.DataFrame({
            "pr_id": df["pr_id"].to_numpy(),
            "visible_at": visible_at(name, df).to_numpy(),
            "login": df["author_login"].to_numpy(),
            "typename": df["author_typename"].to_numpy(),
            "association": df["author_association"].to_numpy(),
            "minimized": (df["is_minimized"].to_numpy()
                          if "is_minimized" in df.columns else False),
        }))
    if not parts:
        return pd.DataFrame(columns=["pr_id", "visible_at"])
    ev = pd.concat(parts, ignore_index=True)
    ev["visible_at"] = pd.to_datetime(ev["visible_at"], utc=True)

    ev = ev[ev["visible_at"].notna() & ev["login"].notna()]
    ev = ev[~is_bot(ev["login"], ev["typename"])]
    # A deleted PR author (null login) cannot be matched, so any commenter counts.
    pr_author = prs.set_index("pr_id")["author_login"]
    ev = ev[ev["login"] != ev["pr_id"].map(pr_author)]
    if definition.exclude_associations:
        ev = ev[~ev["association"].isin(definition.exclude_associations)]
    if definition.exclude_minimized:
        ev = ev[~(ev["minimized"] == True)]  # noqa: E712 -- null-safe
    return ev


def first_human_event(prs: pd.DataFrame, streams: dict[str, pd.DataFrame],
                      definition: Definition) -> pd.Series:
    ev = eligible_events(prs, streams, definition)
    if ev.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return ev.groupby("pr_id")["visible_at"].min()


def label(prs: pd.DataFrame, first_event: pd.Series,
          threshold_h: float = 168.0, censor_h: float = 720.0) -> pd.DataFrame:
    """Blueprint §1. The 30-day window is why censoring is a non-issue for the
    classifier: no qualifying event within 720h => is_slow with certainty (720 > 168)."""
    out = prs[["repo", "pr_id", "created_at"]].copy()
    out["first_event_at"] = pd.to_datetime(out["pr_id"].map(first_event), utc=True)
    out["wait_h"] = (out["first_event_at"] - out["created_at"]).dt.total_seconds() / 3600.0
    out["never_reviewed_30d"] = out["wait_h"].isna() | (out["wait_h"] > censor_h)
    out["is_slow"] = out["never_reviewed_30d"] | (out["wait_h"] > threshold_h)
    out["wait_h_censored"] = out["wait_h"].clip(upper=censor_h).fillna(censor_h)
    out["event_observed"] = ~out["never_reviewed_30d"]
    return out


def label_all(prs: pd.DataFrame, streams: dict[str, pd.DataFrame],
              **kw) -> dict[str, pd.DataFrame]:
    return {name: label(prs, first_human_event(prs, streams, d), **kw)
            for name, d in DEFINITIONS.items()}
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_labels.py -q`
Expected: `7 passed`

- [ ] **Step 5: Stage**

```bash
git add labels.py tests/test_labels.py
```

---

### Task 4: Refactor `gate_report.py` to import `labels.py`

**Files:**
- Modify: `gate_report.py` — delete `first_human_event`, `label_under`, `build_definitions` (lines ~35–125); replace with a thin adapter.
- Create: `tests/test_gate_regression.py` (data-dependent; skipped if pilot data absent)

**Interfaces:**
- Consumes: `labels.label_all`, `labels.DEFINITIONS`, `load.load_repo`.
- Produces: `gate_report.build_definitions(repo_dir) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]` with the SAME signature and return shape as before, so `gate_for_repo` is unchanged.

- [ ] **Step 1: Capture the pre-refactor baseline**

Run:
```bash
cd C:/Users/ragha/project/PRFlowPredict && python gate_report.py --repo benbjohnson/litestream --json-out data/processed/benbjohnson__litestream/gate_pre.json >/dev/null && python gate_report.py --repo anthropics/skills --json-out data/processed/anthropics__skills/gate_pre.json >/dev/null && echo captured
```

- [ ] **Step 2: Write the regression test**

`tests/test_gate_regression.py`:
```python
"""Gate #6: the refactor must not change the target. Any difference must be explained
by exactly the two rule changes the spec makes: PENDING reviews no longer count, and
KNOWN_BOTS logins no longer count."""
import json
from pathlib import Path

import pytest

PRE = {
    "litestream": Path("data/processed/benbjohnson__litestream/gate_pre.json"),
    "skills": Path("data/processed/anthropics__skills/gate_pre.json"),
}
POST = {k: v.with_name("gate.json") for k, v in PRE.items()}
KEYS = ["label_rates", "never_reviewed_30d", "is_slow_rate", "label_spread"]


@pytest.mark.parametrize("name", list(PRE))
def test_label_fields_unchanged(name):
    if not (PRE[name].exists() and POST[name].exists()):
        pytest.skip("pilot gate data not present")
    pre = json.loads(PRE[name].read_text())[0]
    post = json.loads(POST[name].read_text())[0]
    for k in KEYS:
        if isinstance(pre[k], dict):
            assert list(pre[k].values()) == pytest.approx(list(post[k].values()), abs=1e-9), k
        else:
            assert pre[k] == pytest.approx(post[k], abs=1e-9), k
```

- [ ] **Step 3: Replace the inlined label logic in `gate_report.py`**

Delete the functions `first_human_event`, `label_under`, and `build_definitions` and their helper `cat`. Replace with:

```python
import labels
import load


# Display names keep the original wording so gate.json is comparable across runs.
DISPLAY = {
    "D1": "D1 reviews only",
    "D2": "D2 + inline review comments",
    "D3": "D3 + issue comments (blueprint)",
    "D4": "D4 D3 excl. minimized",
    "D5": "D5 D3 excl. NONE-association",
}


def build_definitions(repo_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Thin adapter over labels.py -- the target has exactly one implementation."""
    repo = repo_dir.name.replace("__", "/")
    frames = load.load_repo(repo)
    prs = frames["pr_tier2"]
    if prs.empty:
        return prs, {}
    streams = {k: frames[k] for k in labels.ALL_STREAMS}
    out = labels.label_all(prs, streams, threshold_h=SLOW_THRESHOLD_H, censor_h=CENSOR_WINDOW_H)
    return prs, {DISPLAY[k]: v for k, v in out.items()}
```

Also delete the now-unused `_load` calls inside the old `build_definitions` only; keep `_load` itself (items 5–8 still use it).

- [ ] **Step 4: Regenerate and compare**

Run:
```bash
python gate_report.py --repo benbjohnson/litestream --json-out data/processed/benbjohnson__litestream/gate.json >/dev/null && python gate_report.py --repo anthropics/skills --json-out data/processed/anthropics__skills/gate.json >/dev/null && python -m pytest tests/test_gate_regression.py -q
```
Expected: `2 passed`.

If a test FAILS: run this diagnostic and confirm every differing PR is explained by a PENDING review or a KNOWN_BOTS login; if so, the new numbers are correct and `gate_pre.json` is regenerated from the new code as the baseline. If any difference is NOT explained, stop — the refactor changed the label.
```bash
python - <<'EOF'
import pandas as pd, load, labels
f = load.load_repo("benbjohnson/litestream")
r = f["reviews"]; print("PENDING reviews:", (r["state"]=="PENDING").sum())
for k in ("reviews","thread_comments","issue_comments"):
    d = f[k]; stem = d["author_login"].fillna("").str.lower().str.replace(r"\[bot\]$","",regex=True)
    print(k, "KNOWN_BOTS non-Bot-typed:", ((stem.isin(labels.KNOWN_BOTS)) & (d["author_typename"]!="Bot")).sum())
EOF
```

- [ ] **Step 5: Stage**

```bash
git add gate_report.py tests/test_gate_regression.py
```

---

### Task 5: `cohort_qc.py`

**Files:**
- Create: `cohort_qc.py`, `tests/test_cohort_qc.py`

**Interfaces:**
- Consumes: `load.load_repo`, `labels.label_all`, `data/cohort/cohort.json` entries (`repo`, `cell`, `language_stratum`).
- Produces: `cohort_qc.FAMILY: dict[str, set[str]]`; `cohort_qc.check_repo(entry: dict, frames: dict, window) -> dict` with keys `repo, cell, kept, reasons, n_in_window, n_human, bot_share, language_dominant, is_slow_d5, is_slow_d3, spread_d3_d5`; `cohort_qc.run() -> list[dict]` writing `data/cohort/kept.json` (`{"kept": [...repos], "repos": [...dicts], "look_at_these": [...]}`); `cohort_qc.kept_repos() -> list[str]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_cohort_qc.py`:
```python
import pandas as pd
import cohort_qc
from tests.conftest import BASE

WINDOW = (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-06-30 23:59:59", tz="UTC"))


def frames(n_human, n_bot, language):
    n = n_human + n_bot
    prs = pd.DataFrame({
        "repo": ["o/r"] * n, "pr_id": [f"p{i}" for i in range(n)],
        "created_at": [BASE] * n, "author_login": ["u"] * n,
        "author_is_bot": [False] * n_human + [True] * n_bot,
    })
    meta = pd.DataFrame({"repo": ["o/r"], "language_dominant": [language]})
    empty = pd.DataFrame(columns=["pr_id", "created_at", "published_at", "submitted_at",
                                  "author_login", "author_typename",
                                  "author_association", "is_minimized"])
    return {"pr_tier2": prs, "repo_meta": meta, "reviews": empty,
            "thread_comments": empty, "issue_comments": empty}


def entry(stratum="Python"):
    return {"repo": "o/r", "cell": f"{stratum}:200-800", "language_stratum": stratum}


def test_clean_repo_is_kept():
    r = cohort_qc.check_repo(entry(), frames(150, 10, "Python"), WINDOW)
    assert r["kept"] and r["reasons"] == []
    assert r["n_human"] == 150 and abs(r["bot_share"] - 10 / 160) < 1e-9


def test_bot_dominated_is_dropped():
    r = cohort_qc.check_repo(entry(), frames(100, 120, "Python"), WINDOW)
    assert not r["kept"] and any("bot" in x for x in r["reasons"])


def test_too_small_is_dropped():
    r = cohort_qc.check_repo(entry(), frames(99, 0, "Python"), WINDOW)
    assert not r["kept"] and any("100" in x for x in r["reasons"])


def test_language_mismatch_is_dropped_and_js_is_fine_for_ts():
    bad = cohort_qc.check_repo(entry("Python"), frames(150, 0, "Shell"), WINDOW)
    assert not bad["kept"] and any("language" in x for x in bad["reasons"])
    ok = cohort_qc.check_repo(entry("TypeScript"), frames(150, 0, "JavaScript"), WINDOW)
    assert ok["kept"]


def test_label_rate_never_a_drop_reason():
    # 150 human PRs, none ever reviewed -> 100% never-reviewed. Must still be kept.
    r = cohort_qc.check_repo(entry(), frames(150, 0, "Python"), WINDOW)
    assert r["kept"] and r["is_slow_d5"] == 1.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_cohort_qc.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'cohort_qc'`

- [ ] **Step 3: Implement `cohort_qc.py`**

```python
"""Pre-registered, STRUCTURAL-ONLY cohort checks. Writes data/cohort/kept.json.

There is deliberately no check on the label distribution. A 98%-never-reviewed repo
is data. Dropping on the outcome is the one thing the blueprint forbids for selection,
and post-hoc QC is selection."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

import labels
import load

log = logging.getLogger("qc")

COHORT = Path(__file__).parent / "data" / "cohort" / "cohort.json"
KEPT = Path(__file__).parent / "data" / "cohort" / "kept.json"
WINDOW = (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-06-30 23:59:59", tz="UTC"))

FAMILY = {"Python": {"Python"}, "TypeScript": {"TypeScript", "JavaScript"}, "Go": {"Go"}}
MAX_BOT_SHARE = 0.50
MIN_HUMAN_PRS = 100
LOOK_ROW_SHARE = 0.08
LOOK_SPREAD_PP = 0.15


def check_repo(entry: dict, frames: dict[str, pd.DataFrame], window) -> dict:
    repo = entry["repo"]
    prs = frames["pr_tier2"]
    prs = prs[(prs["created_at"] >= window[0]) & (prs["created_at"] <= window[1])]
    n = len(prs)
    n_bot = int(prs["author_is_bot"].sum()) if n else 0
    n_human = n - n_bot
    bot_share = n_bot / n if n else 0.0
    meta = frames["repo_meta"]
    lang = str(meta["language_dominant"].iloc[0]) if not meta.empty else None

    reasons = []
    if bot_share > MAX_BOT_SHARE:
        reasons.append(f"bot-authored share {bot_share:.0%} > {MAX_BOT_SHARE:.0%}")
    if n_human < MIN_HUMAN_PRS:
        reasons.append(f"{n_human} human PRs < {MIN_HUMAN_PRS}")
    fam = FAMILY.get(entry["language_stratum"], set())
    if lang not in fam:
        reasons.append(f"language {lang!r} not in stratum family {sorted(fam)}")

    # Rates are REPORTED for the look-at-these list, never used as a drop reason.
    human = prs[~prs["author_is_bot"]]
    streams = {k: frames.get(k, pd.DataFrame()) for k in labels.ALL_STREAMS}
    d5 = d3 = float("nan")
    if len(human):
        lab = labels.label_all(human, streams)
        d5 = float(lab["D5"]["is_slow"].mean())
        d3 = float(lab["D3"]["is_slow"].mean())

    return {
        "repo": repo, "cell": entry["cell"], "kept": not reasons, "reasons": reasons,
        "n_in_window": n, "n_human": n_human, "bot_share": bot_share,
        "language_dominant": lang, "is_slow_d5": d5, "is_slow_d3": d3,
        "spread_d3_d5": abs(d5 - d3) if n_human else float("nan"),
    }


def run() -> list[dict]:
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))["selected"]
    results = [check_repo(e, load.load_repo(e["repo"]), WINDOW) for e in cohort]
    kept = [r for r in results if r["kept"]]
    total = sum(r["n_human"] for r in kept) or 1
    look = []
    for r in kept:
        share = r["n_human"] / total
        r["row_share"] = share
        why = []
        if share > LOOK_ROW_SHARE:
            why.append(f"row share {share:.0%}")
        if r["spread_d3_d5"] > LOOK_SPREAD_PP:
            why.append(f"D3-D5 spread {r['spread_d3_d5']:.0%}")
        if why:
            look.append({"repo": r["repo"], "why": why})
    KEPT.write_text(json.dumps({
        "kept": [r["repo"] for r in kept], "repos": results, "look_at_these": look,
        "rules": {"max_bot_share": MAX_BOT_SHARE, "min_human_prs": MIN_HUMAN_PRS,
                  "family": {k: sorted(v) for k, v in FAMILY.items()}},
    }, indent=2), encoding="utf-8")
    return results


def kept_repos() -> list[str]:
    return json.loads(KEPT.read_text(encoding="utf-8"))["kept"]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = run()
    kept = [r for r in res if r["kept"]]
    print(f"kept {len(kept)}/{len(res)}")
    for r in res:
        if not r["kept"]:
            print(f"  DROP {r['repo']:<45} {'; '.join(r['reasons'])}")
    look = json.loads(KEPT.read_text(encoding="utf-8"))["look_at_these"]
    if look:
        print("look at these (not dropped):")
        for l in look:
            print(f"  {l['repo']:<45} {'; '.join(l['why'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_cohort_qc.py -q`
Expected: `5 passed`

- [ ] **Step 5: Run on real data (requires collection + parse complete for the cohort)**

Run: `python parse.py >/dev/null 2>&1; python cohort_qc.py`
Expected: prints `kept N/45` with N ≥ 30 and a reason for every drop. Verify by eye that no drop reason mentions a rate.

- [ ] **Step 6: Stage**

```bash
git add cohort_qc.py tests/test_cohort_qc.py
```

---

### Task 6: `replay.py`

**Files:**
- Create: `replay.py`, `tests/test_replay.py`

**Interfaces:**
- Consumes: `replay_toy` fixture (Task 1); `pr_tier1` columns `repo, pr_id, created_at, closed_at, author_login`; label columns `pr_id, first_event_at, is_slow`.
- Produces:
  - `replay._ns(s: pd.Series | pd.Timestamp) -> np.ndarray | np.datetime64` (tz-aware UTC → naive ns)
  - `replay.History(repo, created_at, closed_at, first_event_at, is_slow, *, window_start, threshold_h=168.0, trailing_days=90)`
  - `replay.History.from_frames(repo: str, tier1: pd.DataFrame, labels: pd.DataFrame, window_start: pd.Timestamp) -> History`
  - `replay.History.features_at(t: pd.Timestamp, global_rate: float, alpha: float = 5.0) -> dict` with keys `open_backlog_at_t: int, trailing_90d_slow_rate: float, trailing_n: int, trailing_window_complete: bool`

- [ ] **Step 1: Write the failing tests**

`tests/test_replay.py`:
```python
import numpy as np
import pandas as pd
import pytest
import replay

WS = pd.Timestamp("2024-01-01", tz="UTC")


def test_hand_computed_features(replay_toy):
    tier1, labels = replay_toy
    hist = replay.History.from_frames("r", tier1, labels, WS)
    f = hist.features_at(pd.Timestamp("2024-04-21", tz="UTC"), global_rate=0.5, alpha=5.0)
    assert f["open_backlog_at_t"] == 4
    assert f["trailing_n"] == 3                      # P4 excluded: not resolvable
    assert f["trailing_90d_slow_rate"] == pytest.approx((1 + 5 * 0.5) / (3 + 5))
    assert f["trailing_window_complete"] is True


def test_before_window_is_prior_only_and_flagged(replay_toy):
    tier1, labels = replay_toy
    hist = replay.History.from_frames("r", tier1, labels, WS)
    f = hist.features_at(pd.Timestamp("2024-02-01", tz="UTC"), global_rate=0.5)
    assert f["open_backlog_at_t"] == 1               # only P0, still open
    assert f["trailing_n"] == 0
    assert f["trailing_90d_slow_rate"] == pytest.approx(0.5)   # pure prior
    assert f["trailing_window_complete"] is False


def test_row_at_exactly_t_is_not_visible(replay_toy):
    tier1, labels = replay_toy
    hist = replay.History.from_frames("r", tier1, labels, WS)
    # P3 created 2024-04-15T00:00. At t == that instant it must NOT be in the prefix.
    f_at = hist.features_at(pd.Timestamp("2024-04-15", tz="UTC"), global_rate=0.5)
    f_after = hist.features_at(pd.Timestamp("2024-04-15T00:00:01", tz="UTC"), global_rate=0.5)
    assert f_after["open_backlog_at_t"] == f_at["open_backlog_at_t"] + 1


def test_resolvability_flips_when_first_event_becomes_visible(replay_toy):
    tier1, labels = replay_toy
    hist = replay.History.from_frames("r", tier1, labels, WS)
    # P3: created 4-15, first_event 4-16. At t=4-15T12:00 it is <168h old and its
    # event is in the future -> not resolvable. At t=4-17 the event is visible.
    a = hist.features_at(pd.Timestamp("2024-04-15T12:00", tz="UTC"), global_rate=0.5)
    b = hist.features_at(pd.Timestamp("2024-04-17", tz="UTC"), global_rate=0.5)
    assert b["trailing_n"] == a["trailing_n"] + 1


def test_unsorted_input_is_sorted(replay_toy):
    tier1, labels = replay_toy
    shuffled = tier1.sample(frac=1, random_state=1)
    a = replay.History.from_frames("r", tier1, labels, WS)
    b = replay.History.from_frames("r", shuffled, labels, WS)
    t = pd.Timestamp("2024-04-21", tz="UTC")
    assert a.features_at(t, 0.5) == b.features_at(t, 0.5)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_replay.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'replay'`

- [ ] **Step 3: Implement `replay.py`**

```python
"""Point-in-time replay. Leakage is structural here, not checked for.

Blueprint §4: "implement a features_at(pr, history_before_pr) function and generate
the table by chronological replay -- this makes leakage structurally impossible."

`features_at(t)` locates the prefix `created_at < t` by bisect and computes every
feature from that prefix only. There is no code path by which a row at or after t is
visible. Phase 3 extends `features_at`; the bisect-prefix discipline is the invariant
and must not be bypassed for convenience.

Phase 2 implements exactly two features: open backlog and the trailing-90-day slow
rate under the RESOLVABILITY predicate -- the [R1] correction to blueprint §4:
"strictly earlier than the row" is insufficient, because a PR opened 3 days ago has no
knowable label yet. A prior row's label is usable at t only if
    created_at <= t - 168h   OR   first_event_at < t."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _ns(x):
    """tz-aware UTC -> naive datetime64[ns] for numpy. The ONLY sanctioned conversion."""
    if isinstance(x, pd.Timestamp):
        return np.datetime64(x.tz_convert("UTC").tz_localize(None), "ns")
    s = pd.to_datetime(x, utc=True)
    return s.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")


@dataclass
class History:
    repo: str
    created: np.ndarray        # datetime64[ns], sorted ascending
    closed: np.ndarray         # datetime64[ns], NaT if open
    first_event: np.ndarray    # datetime64[ns], NaT if none
    is_slow: np.ndarray        # float: 1.0 / 0.0 / nan (nan = no label, pre-window)
    window_start: np.datetime64
    threshold_h: float = 168.0
    trailing_days: int = 90

    def __init__(self, repo, created_at, closed_at, first_event_at, is_slow, *,
                 window_start, threshold_h=168.0, trailing_days=90):
        order = np.argsort(created_at, kind="stable")
        self.repo = repo
        self.created = np.asarray(created_at)[order]
        self.closed = np.asarray(closed_at)[order]
        self.first_event = np.asarray(first_event_at)[order]
        self.is_slow = np.asarray(is_slow, dtype=float)[order]
        self.window_start = window_start
        self.threshold_h = threshold_h
        self.trailing_days = trailing_days

    @classmethod
    def from_frames(cls, repo: str, tier1: pd.DataFrame, labels: pd.DataFrame,
                    window_start: pd.Timestamp) -> "History":
        t1 = tier1[tier1["repo"] == repo] if "repo" in tier1.columns else tier1
        lab = labels.drop_duplicates("pr_id").set_index("pr_id")
        fe = pd.to_datetime(t1["pr_id"].map(lab["first_event_at"]), utc=True)
        sl = t1["pr_id"].map(lab["is_slow"]).astype(float)   # NaN where unlabelled
        return cls(repo, _ns(t1["created_at"]), _ns(t1["closed_at"]), _ns(fe),
                   sl.to_numpy(), window_start=_ns(window_start))

    def features_at(self, t: pd.Timestamp, global_rate: float, alpha: float = 5.0) -> dict:
        t64 = _ns(t)
        n = int(np.searchsorted(self.created, t64, side="left"))   # created < t only
        created, closed = self.created[:n], self.closed[:n]
        fe, sl = self.first_event[:n], self.is_slow[:n]

        backlog = int(np.sum(np.isnat(closed) | (closed > t64)))

        lo = t64 - np.timedelta64(self.trailing_days, "D")
        thr = t64 - np.timedelta64(int(self.threshold_h * 3600), "s")
        in_window = created >= lo
        labelled = ~np.isnan(sl)
        resolvable = (created <= thr) | (~np.isnat(fe) & (fe < t64))
        m = in_window & labelled & resolvable
        k = int(m.sum())
        n_slow = float(np.nansum(sl[m]))
        rate = (n_slow + alpha * global_rate) / (k + alpha)

        return {
            "open_backlog_at_t": backlog,
            "trailing_90d_slow_rate": float(rate),
            "trailing_n": k,
            "trailing_window_complete": bool(lo >= self.window_start),
        }
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_replay.py -q`
Expected: `5 passed`

- [ ] **Step 5: Stage**

```bash
git add replay.py tests/test_replay.py
```

---

### Task 7: `metrics.py`

**Files:**
- Create: `metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `metrics.precision_at_k(y_true, score, repo, k=10, seed=20260912) -> tuple[pd.Series, float]` (per-repo Series indexed by repo, and its mean)
  - `metrics.auc_pr(y_true, score) -> float`
  - `metrics.cluster_bootstrap(per_repo, n=2000, seed=20260912, ci=0.95) -> tuple[float, float]`

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:
```python
import numpy as np
import pandas as pd
import pytest
import metrics


def test_perfect_ranking_gives_precision_one():
    y = [1] * 10 + [0] * 5
    s = [1.0] * 10 + [0.0] * 5
    per_repo, mean = metrics.precision_at_k(y, s, ["X"] * 15, k=10)
    assert per_repo["X"] == 1.0 and mean == 1.0


def test_fewer_than_k_uses_all_rows():
    per_repo, _ = metrics.precision_at_k([1, 0, 0, 1], [0.9, 0.8, 0.7, 0.6], ["Y"] * 4, k=10)
    assert per_repo["Y"] == 0.5


def test_constant_score_is_random_not_first_n():
    # 100 rows: first 50 positive, last 50 negative, constant score. If ties were
    # broken by row order, P@10 would be 1.0 every time. It must vary with the seed.
    y = [1] * 50 + [0] * 50
    s = [0.5] * 100
    vals = {metrics.precision_at_k(y, s, ["Z"] * 100, k=10, seed=i)[1] for i in range(20)}
    assert len(vals) > 1
    assert 0.0 <= min(vals) and max(vals) <= 1.0


def test_mean_is_over_repos_not_rows():
    y = [1] * 10 + [0] * 100
    s = [1.0] * 10 + [0.0] * 100
    repo = ["A"] * 10 + ["B"] * 100
    _, mean = metrics.precision_at_k(y, s, repo, k=10)
    assert mean == 0.5    # A=1.0, B=0.0, mean over the two repos


def test_auc_pr_perfect_and_random():
    assert metrics.auc_pr([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert 0 < metrics.auc_pr([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) <= 1.0


def test_cluster_bootstrap_constant_and_contains_mean():
    lo, hi = metrics.cluster_bootstrap(pd.Series([0.3, 0.3, 0.3]))
    assert lo == pytest.approx(0.3) and hi == pytest.approx(0.3)
    v = pd.Series(np.linspace(0, 1, 30))
    lo, hi = metrics.cluster_bootstrap(v)
    assert lo < v.mean() < hi
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'metrics'`

- [ ] **Step 3: Implement `metrics.py`**

```python
"""Evaluation. Precision@k is PER REPO (the product framing: rank this team's PRs), and
confidence intervals resample REPOS, because the effective sample size for a
cross-repo claim is the number of repos, not of PRs ([R1] correction to blueprint §4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

SEED = 20260912


def precision_at_k(y_true, score, repo, k: int = 10, seed: int = SEED) -> tuple[pd.Series, float]:
    """Per repo: top-k by score, ties broken by a seeded random permutation.

    Random tie-breaking matters: the trailing-rate baseline is CONSTANT within a repo,
    and breaking ties by row order would silently reward whatever order the rows
    happened to be in. With random ties its P@k is an honest random draw = base rate."""
    df = pd.DataFrame({
        "y": np.asarray(y_true, dtype=float),
        "s": np.asarray(score, dtype=float),
        "repo": np.asarray(repo),
    })
    df["tie"] = np.random.default_rng(seed).random(len(df))
    df = df.sort_values(["repo", "s", "tie"], ascending=[True, False, False])
    top = df.groupby("repo", sort=True).head(k)
    per_repo = top.groupby("repo")["y"].mean()
    return per_repo, float(per_repo.mean())


def auc_pr(y_true, score) -> float:
    return float(average_precision_score(np.asarray(y_true, dtype=int), np.asarray(score, dtype=float)))


def cluster_bootstrap(per_repo, n: int = 2000, seed: int = SEED, ci: float = 0.95) -> tuple[float, float]:
    v = np.asarray(per_repo, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n)])
    a = (1.0 - ci) / 2.0
    return float(np.quantile(means, a)), float(np.quantile(means, 1.0 - a))
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: `6 passed`

- [ ] **Step 5: Stage**

```bash
git add metrics.py tests/test_metrics.py
```

---

### Task 8: `splits.py`

**Files:**
- Create: `splits.py`, `tests/test_splits.py`

**Interfaces:**
- Consumes: `pr_tier2` columns `repo, pr_id, created_at, author_is_bot, author_login`; primary label columns `pr_id, is_slow, first_event_at, wait_h`.
- Produces:
  - `splits.CUTOFF_A`, `splits.WINDOW_START`, `splits.WINDOW_END`, `splits.CAP_FRAC = 0.05`, `splits.SEED`
  - `splits.prepare_rows(prs_tier2, label_primary, kept_repos) -> pd.DataFrame` (bot authors removed, in-window, kept repos, joined with `is_slow, first_event_at, wait_h`, sorted by `created_at, pr_id`, index reset)
  - `splits.cap_per_repo(rows, idx: np.ndarray, cap_frac, seed) -> np.ndarray`
  - `splits.scenario_a(rows, cap_frac=CAP_FRAC, seed=SEED) -> tuple[np.ndarray, np.ndarray]`
  - `splits.scenario_b(rows, n_splits=5, cap_frac=CAP_FRAC, seed=SEED) -> list[tuple[np.ndarray, np.ndarray]]`
  - `splits.row_share(rows) -> pd.Series` (fraction of rows per repo, descending)

- [ ] **Step 1: Write the failing tests**

`tests/test_splits.py`:
```python
import numpy as np
import pandas as pd
import splits


def rows_fixture():
    def ts(s): return pd.Timestamp(s, tz="UTC")
    n_big, n_small = 400, 40
    prs = pd.DataFrame({
        "repo": ["big"] * n_big + ["small"] * n_small + ["bot"] * 3 + ["dropped"] * 5,
        "pr_id": [f"p{i}" for i in range(n_big + n_small + 8)],
        "created_at": (list(pd.date_range("2024-02-01", "2026-06-01", periods=n_big, tz="UTC"))
                       + list(pd.date_range("2024-02-01", "2026-06-01", periods=n_small, tz="UTC"))
                       + [ts("2025-01-01")] * 3 + [ts("2025-01-01")] * 5),
        "author_is_bot": [False] * (n_big + n_small) + [True] * 3 + [False] * 5,
        "author_login": ["u"] * (n_big + n_small + 8),
    })
    lab = pd.DataFrame({"pr_id": prs["pr_id"], "is_slow": [False] * len(prs),
                        "first_event_at": [pd.NaT] * len(prs), "wait_h": [np.nan] * len(prs)})
    return splits.prepare_rows(prs, lab, kept_repos=["big", "small", "bot"])


def test_prepare_rows_drops_bots_and_unkept():
    rows = rows_fixture()
    assert set(rows["repo"]) == {"big", "small"}
    assert "is_slow" in rows.columns
    assert rows["created_at"].is_monotonic_increasing


def test_scenario_a_cutoff_and_cap():
    rows = rows_fixture()
    tr, te = splits.scenario_a(rows, cap_frac=0.05)
    assert (rows.loc[tr, "created_at"] < splits.CUTOFF_A).all()
    assert (rows.loc[te, "created_at"] >= splits.CUTOFF_A).all()
    pre = rows[rows["created_at"] < splits.CUTOFF_A]
    cap = max(1, int(0.05 * len(pre)))                         # 5% of the UNCAPPED total
    for repo in ("big", "small"):
        n_avail = int((pre["repo"] == repo).sum())
        assert (rows.loc[tr, "repo"] == repo).sum() == min(cap, n_avail)
    # test is never capped
    assert (rows.loc[te, "repo"] == "big").sum() == int(((rows["repo"] == "big") &
                                                         (rows["created_at"] >= splits.CUTOFF_A)).sum())


def test_scenario_a_is_deterministic():
    rows = rows_fixture()
    a1, _ = splits.scenario_a(rows)
    a2, _ = splits.scenario_a(rows)
    assert np.array_equal(a1, a2)


def test_scenario_b_holds_out_whole_repos():
    rows = rows_fixture()
    folds = splits.scenario_b(rows, n_splits=2)
    assert len(folds) == 2
    for tr, te in folds:
        assert not (set(rows.loc[tr, "repo"]) & set(rows.loc[te, "repo"]))
    covered = set().union(*[set(rows.loc[te, "repo"]) for _, te in folds])
    assert covered == {"big", "small"}


def test_row_share_sums_to_one():
    rows = rows_fixture()
    s = splits.row_share(rows)
    assert abs(s.sum() - 1.0) < 1e-9 and s.index[0] == "big"
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_splits.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'splits'`

- [ ] **Step 3: Implement `splits.py`**

```python
"""Train/test splits. Both scenarios are always reported (blueprint §2); random
row-level splits are rejected because they leak time and repo identity.

Scenario A (known-project, primary): time cutoff within the same repos.
Scenario B (cold-start): leave-repos-out via GroupKFold on repo.

Training rows are capped per repo so that three repos holding ~10% of rows each do
not become the model's whole notion of review culture. The cap is a fraction of the
UNCAPPED training total, computed once, never iterated. Test rows are never capped."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

SEED = 20260912
CAP_FRAC = 0.05
WINDOW_START = pd.Timestamp("2024-01-01T00:00:00Z")
WINDOW_END = pd.Timestamp("2026-06-30T23:59:59Z")
CUTOFF_A = pd.Timestamp("2026-01-01T00:00:00Z")


def prepare_rows(prs_tier2: pd.DataFrame, label_primary: pd.DataFrame,
                 kept_repos: list[str]) -> pd.DataFrame:
    rows = prs_tier2.merge(
        label_primary[["pr_id", "is_slow", "first_event_at", "wait_h"]], on="pr_id", how="inner")
    rows = rows[
        rows["repo"].isin(kept_repos)
        & ~rows["author_is_bot"].fillna(False).astype(bool)
        & (rows["created_at"] >= WINDOW_START) & (rows["created_at"] <= WINDOW_END)
    ]
    return rows.sort_values(["created_at", "pr_id"]).reset_index(drop=True)


def cap_per_repo(rows: pd.DataFrame, idx: np.ndarray, cap_frac: float, seed: int) -> np.ndarray:
    sub = rows.loc[idx]
    cap = max(1, int(cap_frac * len(sub)))
    keep = []
    for _, g in sub.groupby("repo", sort=True):
        keep.extend(g.index if len(g) <= cap else g.sample(n=cap, random_state=seed).index)
    return np.sort(np.asarray(keep))


def scenario_a(rows: pd.DataFrame, cap_frac: float = CAP_FRAC, seed: int = SEED):
    train = rows.index[rows["created_at"] < CUTOFF_A].to_numpy()
    test = rows.index[rows["created_at"] >= CUTOFF_A].to_numpy()
    return cap_per_repo(rows, train, cap_frac, seed), test


def scenario_b(rows: pd.DataFrame, n_splits: int = 5, cap_frac: float = CAP_FRAC,
               seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    gkf = GroupKFold(n_splits=n_splits)
    out = []
    for tr, te in gkf.split(rows, groups=rows["repo"]):
        tr_idx, te_idx = rows.index[tr].to_numpy(), rows.index[te].to_numpy()
        out.append((cap_per_repo(rows, tr_idx, cap_frac, seed), te_idx))
    return out


def row_share(rows: pd.DataFrame) -> pd.Series:
    return (rows["repo"].value_counts(normalize=True)).sort_values(ascending=False)
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_splits.py -q`
Expected: `5 passed`

- [ ] **Step 5: Stage**

```bash
git add splits.py tests/test_splits.py
```

---

### Task 9: `baseline.py`

**Files:**
- Create: `baseline.py`, `tests/test_baseline.py`

**Interfaces:**
- Consumes: everything from Tasks 2–8.
- Produces: `baseline.score_rows(rows, test_idx, train_idx, tier1, label_primary) -> np.ndarray` (trailing-rate score per test row); `baseline.evaluate(scenario: str, rows, folds, tier1, label_primary, seed) -> list[dict]` (one dict per fold with `precision_at_10, p10_ci_lo, p10_ci_hi, base_rate_p10, auc_pr, base_rate, n_train, n_test, n_test_repos`); `baseline.main()` runs both scenarios and logs via `tracking.log`.

- [ ] **Step 1: Write the failing test (toy, end to end)**

`tests/test_baseline.py`:
```python
import numpy as np
import pandas as pd
import baseline
import splits


def test_baseline_end_to_end_on_toy():
    def ts(s): return pd.Timestamp(s, tz="UTC")
    n = 300
    rng = np.random.default_rng(0)
    created = pd.date_range("2024-01-15", "2026-06-15", periods=n, tz="UTC")
    repo = np.where(np.arange(n) % 2 == 0, "r1", "r2")
    # r1 is slow 80% of the time, r2 20% -- the baseline should separate REPOS
    is_slow = rng.random(n) < np.where(repo == "r1", 0.8, 0.2)
    prs = pd.DataFrame({"repo": repo, "pr_id": [f"p{i}" for i in range(n)],
                        "created_at": created, "closed_at": [pd.NaT] * n,
                        "author_is_bot": False, "author_login": "u"})
    fe = pd.Series(created + pd.Timedelta(hours=2))
    fe[is_slow] = pd.NaT
    lab = pd.DataFrame({"pr_id": prs["pr_id"], "is_slow": is_slow,
                        "first_event_at": fe,
                        "wait_h": np.where(is_slow, np.nan, 2.0)})
    rows = splits.prepare_rows(prs, lab, ["r1", "r2"])
    tier1 = prs[["repo", "pr_id", "created_at", "closed_at", "author_login"]]

    tr, te = splits.scenario_a(rows, cap_frac=1.0)
    res = baseline.evaluate("A", rows, [(tr, te)], tier1, lab)[0]

    # within-repo the score is ~constant, so P@10 is a random draw ~ base rate
    assert res["p10_ci_lo"] - 0.15 <= res["base_rate_p10"] <= res["p10_ci_hi"] + 0.15
    # but ACROSS repos the score separates r1 from r2, so AUC-PR > base rate
    assert res["auc_pr"] > res["base_rate"]
    assert res["n_test_repos"] == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_baseline.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'baseline'`

- [ ] **Step 3: Implement `baseline.py`**

```python
"""The baseline every model must beat: the repo's trailing-90-day slow rate,
thresholded at 0.5 (blueprint §2, model 1). No learning.

EXPECTED RESULT, STATED IN ADVANCE: Precision@10 ~= base rate within CI on both
scenarios, because the score is constant within a repo. If the baseline BEATS the
within-repo base rate, something is leaking in replay.py -- stop and audit.

AUC-PR, being global, CAN exceed the base rate: the trailing rate ranks REPOS. That is
the honest statement of what a no-PR-signal model can do."""
from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

import cohort_qc
import labels
import load
import metrics
import replay
import splits
import tracking

log = logging.getLogger("baseline")


def score_rows(rows: pd.DataFrame, test_idx: np.ndarray, train_idx: np.ndarray,
               tier1: pd.DataFrame, label_primary: pd.DataFrame) -> np.ndarray:
    # g: the D5 slow rate over TRAINING rows of this split only (spec §6.2).
    g = float(rows.loc[train_idx, "is_slow"].mean())
    test = rows.loc[test_idx]
    hist = {r: replay.History.from_frames(r, tier1, label_primary, splits.WINDOW_START)
            for r in test["repo"].unique()}
    return np.array([
        hist[r].features_at(t, g)["trailing_90d_slow_rate"]
        for r, t in zip(test["repo"].to_numpy(), test["created_at"])
    ])


def evaluate(scenario: str, rows: pd.DataFrame, folds, tier1: pd.DataFrame,
             label_primary: pd.DataFrame, seed: int = splits.SEED) -> list[dict]:
    out = []
    for fold, (tr, te) in enumerate(folds):
        score = score_rows(rows, te, tr, tier1, label_primary)
        test = rows.loc[te]
        y = test["is_slow"].to_numpy(dtype=int)
        per_repo, p10 = metrics.precision_at_k(y, score, test["repo"].to_numpy(), k=10, seed=seed)
        lo, hi = metrics.cluster_bootstrap(per_repo, seed=seed)
        # the comparison point: each test repo's own base rate, averaged over repos
        base_p10 = float(test.groupby("repo")["is_slow"].mean().mean())
        res = {
            "scenario": scenario, "fold": fold, "model": "baseline_trailing90",
            "features": "trailing_90d_slow_rate", "params": "alpha=5;threshold=0.5",
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "n_test_repos": int(test["repo"].nunique()),
            "precision_at_10": p10, "p10_ci_lo": lo, "p10_ci_hi": hi,
            "base_rate_p10": base_p10,
            "auc_pr": metrics.auc_pr(y, score), "base_rate": float(y.mean()),
        }
        out.append(res)
        log.info("%s fold %d: P@10=%.3f [%.3f, %.3f] base=%.3f | AUC-PR=%.3f base=%.3f | %d repos",
                 scenario, fold, p10, lo, hi, base_p10, res["auc_pr"], res["base_rate"],
                 res["n_test_repos"])
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    kept = cohort_qc.kept_repos()
    frames = load.load_all(kept)
    streams = {k: frames[k] for k in labels.ALL_STREAMS}
    lab = labels.label(frames["pr_tier2"],
                       labels.first_human_event(frames["pr_tier2"], streams, labels.DEFINITIONS[labels.PRIMARY]))
    rows = splits.prepare_rows(frames["pr_tier2"], lab, kept)
    tier1 = frames["pr_tier1"]

    results = evaluate("A", rows, [splits.scenario_a(rows)], tier1, lab)
    results += evaluate("B", rows, splits.scenario_b(rows), tier1, lab)
    for r in results:
        tracking.log({**r, "notes": f"label={labels.PRIMARY}"})
    print(f"logged {len(results)} rows to {tracking.EXPERIMENTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_baseline.py -q`
Expected: `1 passed`

- [ ] **Step 5: Run on real data (requires Task 5 step 5 done)**

Run: `python baseline.py && tail -7 data/experiments.csv`
Expected: 6 log lines (A fold 0, B folds 0–4). For every row, `precision_at_10` lies inside `[p10_ci_lo − 0.05, p10_ci_hi + 0.05]` of `base_rate_p10`. **If P@10 clearly exceeds the base rate within-repo, STOP — audit `replay.py` before Phase 3.**

- [ ] **Step 6: Stage**

```bash
git add baseline.py tests/test_baseline.py
```

---

### Task 10: `eda_report.py` and the Phase 2 gate

**Files:**
- Create: `eda_report.py`, `tests/test_eda_gate.py`
- Output: `docs/phase2_eda.md`, `figures/*.png`, `data/phase2_gate.json`

**Interfaces:**
- Consumes: `cohort_qc.KEPT` json, `load.load_all`, `labels.label_all`, `splits.*`, `baseline.evaluate`, `metrics.*`, `lifelines.KaplanMeierFitter`.
- Produces: `eda_report.gate_checks(kept_n, d5_rate, a_cov, b_min_fold, baseline_rows) -> list[dict]` (pure; each `{"id", "check", "value", "pass"}`); `eda_report.main()`.

- [ ] **Step 1: Write the failing test for the gate logic (pure)**

`tests/test_eda_gate.py`:
```python
import eda_report


def base_rows(p10=0.50, lo=0.45, hi=0.55, base=0.50):
    return [{"scenario": "A", "precision_at_10": p10, "p10_ci_lo": lo, "p10_ci_hi": hi,
             "base_rate_p10": base},
            {"scenario": "B", "precision_at_10": p10, "p10_ci_lo": lo, "p10_ci_hi": hi,
             "base_rate_p10": base}]


def test_all_pass():
    checks = eda_report.gate_checks(kept_n=40, d5_rate=0.45, a_repos_with_10=25,
                                    b_min_fold_repos=8, baseline_rows=base_rows())
    assert all(c["pass"] for c in checks) and len(checks) == 5


def test_each_failure_is_named():
    fails = eda_report.gate_checks(kept_n=29, d5_rate=0.75, a_repos_with_10=19,
                                   b_min_fold_repos=5,
                                   baseline_rows=base_rows(p10=0.90, lo=0.85, hi=0.95, base=0.50))
    assert [c["pass"] for c in fails] == [False] * 5


def test_baseline_beating_base_rate_fails_sanity():
    checks = eda_report.gate_checks(kept_n=40, d5_rate=0.45, a_repos_with_10=25,
                                    b_min_fold_repos=8,
                                    baseline_rows=base_rows(p10=0.80, lo=0.75, hi=0.85, base=0.50))
    assert checks[4]["pass"] is False and "leak" in checks[4]["check"].lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_eda_gate.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'eda_report'`

- [ ] **Step 3: Implement `eda_report.py`**

```python
"""Phase 2 EDA report + pre-registered gate. Writes docs/phase2_eda.md and figures/."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

import baseline
import cohort_qc
import labels
import load
import splits

log = logging.getLogger("eda")
ROOT = Path(__file__).parent
DOC = ROOT / "docs" / "phase2_eda.md"
FIG = ROOT / "figures"
GATE_JSON = ROOT / "data" / "phase2_gate.json"
THRESHOLDS = [72, 120, 168, 240]


# ----------------------------------------------------------------------------
# Gate (pure, tested)
# ----------------------------------------------------------------------------

def gate_checks(kept_n: int, d5_rate: float, a_repos_with_10: int, b_min_fold_repos: int,
                baseline_rows: list[dict]) -> list[dict]:
    def sane(r):
        # P@10 must sit within the CI of the base rate (with a 5pp tolerance on the
        # bootstrap edges). Beating it within-repo means the replay leaks.
        return r["p10_ci_lo"] - 0.05 <= r["base_rate_p10"] <= r["p10_ci_hi"] + 0.05 \
            and r["precision_at_10"] <= r["base_rate_p10"] + 0.10
    return [
        {"id": 1, "check": "repos kept after QC >= 30", "value": kept_n, "pass": kept_n >= 30},
        {"id": 2, "check": "D5 global is_slow in [10%, 70%]", "value": round(d5_rate, 4),
         "pass": 0.10 <= d5_rate <= 0.70},
        {"id": 3, "check": "Scenario A: >= 20 test repos with >= 10 test PRs",
         "value": a_repos_with_10, "pass": a_repos_with_10 >= 20},
        {"id": 4, "check": "Scenario B: every fold holds out >= 6 repos",
         "value": b_min_fold_repos, "pass": b_min_fold_repos >= 6},
        {"id": 5, "check": "baseline sanity: P@10 within CI of base rate (else replay LEAKS)",
         "value": [round(r["precision_at_10"], 3) for r in baseline_rows],
         "pass": all(sane(r) for r in baseline_rows)},
    ]


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def fig_hist(lab: pd.DataFrame, tier_of: pd.Series) -> Path:
    FIG.mkdir(exist_ok=True)
    rev = lab[lab["event_observed"]].copy()
    rev["tier"] = rev["repo"].map(tier_of)
    fig, ax = plt.subplots(figsize=(8, 4))
    for tier, g in rev.groupby("tier"):
        ax.hist(np.log10(g["wait_h"].clip(lower=1e-2)), bins=40, alpha=0.5, label=tier, density=True)
    ax.axvline(np.log10(168), color="k", ls="--", label="168h")
    ax.set_xlabel("log10(hours to first human review)"); ax.set_ylabel("density"); ax.legend()
    p = FIG / "wait_hist.png"; fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_km(lab: pd.DataFrame, tier_of: pd.Series) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    kmf = KaplanMeierFitter()
    kmf.fit(lab["wait_h_censored"], event_observed=lab["event_observed"], label="all")
    kmf.plot_survival_function(ax=ax)
    for tier, g in lab.assign(tier=lab["repo"].map(tier_of)).groupby("tier"):
        KaplanMeierFitter().fit(g["wait_h_censored"], g["event_observed"], label=tier)\
            .plot_survival_function(ax=ax, ci_show=False)
    ax.axvline(168, color="k", ls="--"); ax.set_xlim(0, 720)
    ax.set_xlabel("hours since open"); ax.set_ylabel("fraction still unreviewed")
    p = FIG / "km_unreviewed.png"; fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    qc = json.loads(cohort_qc.KEPT.read_text(encoding="utf-8"))
    kept = qc["kept"]
    cohort = {e["repo"]: e for e in json.loads(cohort_qc.COHORT.read_text(encoding="utf-8"))["selected"]}
    tier_of = pd.Series({r: cohort[r]["star_tier"] for r in cohort})

    frames = load.load_all(kept)
    prs = frames["pr_tier2"]
    prs = prs[~prs["author_is_bot"].fillna(False).astype(bool)]
    prs = prs[(prs["created_at"] >= splits.WINDOW_START) & (prs["created_at"] <= splits.WINDOW_END)]
    streams = {k: frames[k] for k in labels.ALL_STREAMS}
    all_labels = labels.label_all(prs, streams)
    lab = all_labels[labels.PRIMARY]

    # 1. cohort table
    qc_df = pd.DataFrame(qc["repos"])[["repo", "cell", "kept", "reasons", "n_in_window", "n_human",
                                       "bot_share", "is_slow_d5", "is_slow_d3"]]
    qc_df["reasons"] = qc_df["reasons"].apply("; ".join)
    share = splits.row_share(prs)
    qc_df["row_share"] = qc_df["repo"].map(share).fillna(0.0)

    # 2. sensitivity
    sens = pd.DataFrame([{"definition": k, "is_slow": v["is_slow"].mean(),
                          "never_reviewed_30d": v["never_reviewed_30d"].mean(),
                          "median_wait_h": v.loc[v["event_observed"], "wait_h"].median()}
                         for k, v in all_labels.items()])
    per_repo_spread = (all_labels["D3"].groupby("repo")["is_slow"].mean()
                       - all_labels["D5"].groupby("repo")["is_slow"].mean()).abs()

    # 3-4. figures
    hist_p, km_p = fig_hist(lab, tier_of), fig_km(lab, tier_of)

    # 5. threshold sensitivity
    thr = pd.DataFrame([{"threshold_h": t,
                         "is_slow": float((lab["never_reviewed_30d"] | (lab["wait_h"] > t)).mean())}
                        for t in THRESHOLDS])

    # 6. baseline
    rows = splits.prepare_rows(frames["pr_tier2"], lab, kept)
    tr_a, te_a = splits.scenario_a(rows)
    folds_b = splits.scenario_b(rows)
    base_rows = baseline.evaluate("A", rows, [(tr_a, te_a)], frames["pr_tier1"], lab)
    base_rows += baseline.evaluate("B", rows, folds_b, frames["pr_tier1"], lab)
    base_df = pd.DataFrame(base_rows)[["scenario", "fold", "n_test_repos", "precision_at_10",
                                       "p10_ci_lo", "p10_ci_hi", "base_rate_p10", "auc_pr", "base_rate"]]

    # 7. gate
    a_cov = int((rows.loc[te_a].groupby("repo").size() >= 10).sum())
    b_min = min(rows.loc[te, "repo"].nunique() for _, te in folds_b)
    checks = gate_checks(len(kept), float(lab["is_slow"].mean()), a_cov, b_min, base_rows)
    GATE_JSON.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    verdict = "PASS" if all(c["pass"] for c in checks) else "FAIL"

    look = qc.get("look_at_these", [])
    doc = f"""# Phase 2 — EDA and Gate

Generated by `eda_report.py`. Primary label: **{labels.PRIMARY}**. Rows: human-authored,
in-window PRs from kept repos ({len(prs):,} PRs, {len(kept)} repos).

## Verdict: **{verdict}**

{md_table(pd.DataFrame(checks)[["id", "check", "value", "pass"]], "{}")}

## 1. Cohort

{md_table(qc_df)}

Look-at-these (not dropped): {"; ".join(f"{l['repo']} ({', '.join(l['why'])})" for l in look) or "none"}

## 2. Label sensitivity

{md_table(sens)}

Per-repo |D3 − D5| spread: median {per_repo_spread.median():.3f}, max {per_repo_spread.max():.3f}
({per_repo_spread.idxmax()}).

Limitation: commenter `author_association` is read-time-computed (proven for PR authors in
Phase 0), so D5 slightly over-counts reviews relative to true at-the-time standing.

## 3. Wait-time distribution

![wait](../figures/{hist_p.name})

## 4. Kaplan–Meier: fraction still unreviewed

![km](../figures/{km_p.name})

## 5. Threshold sensitivity

{md_table(thr)}

Recommendation: keep 168h unless the curve shows a natural break elsewhere.

## 6. Baseline (trailing-90d slow rate)

{md_table(base_df)}

Expected in advance: P@10 ≈ base rate (score is constant within a repo). AUC-PR may exceed
the base rate because the trailing rate ranks repos.
"""
    DOC.write_text(doc, encoding="utf-8")
    print(f"wrote {DOC}  verdict={verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify the gate test passes**

Run: `python -m pytest tests/test_eda_gate.py -q`
Expected: `3 passed`

- [ ] **Step 5: Run the report on real data**

Run: `python eda_report.py; echo "exit=$?"`
Expected: `docs/phase2_eda.md` written, `figures/wait_hist.png` and `figures/km_unreviewed.png` exist, verdict printed. Open the markdown and confirm every section has content and every figure renders.

- [ ] **Step 6: Full test suite**

Run: `python -m pytest tests -q`
Expected: all pass (`~34 passed`, 2 possibly skipped if pilot gate files are absent).

- [ ] **Step 7: Stage**

```bash
git add eda_report.py tests/test_eda_gate.py docs/phase2_eda.md figures/
```

---

## Verification (spec §13)

1. `python cohort_qc.py` — N ≥ 30 kept; every drop reason is structural; guidelines-type repos caught by the language rule.
2. `python -m pytest tests/test_gate_regression.py -q` — label fields identical pre/post refactor, or every difference explained by PENDING/KNOWN_BOTS.
3. `python baseline.py` — P@10 ≈ base rate on both scenarios. **If it exceeds the CI, stop and audit `replay.py`.**
4. `python -m pytest tests/test_replay.py -q` — the hand-computed toy passes, including the not-resolvable-yet PR.
5. `python eda_report.py` — renders with all figures; gate table at the top.
6. Manual: from `kept.json["look_at_these"]`, pick 3 PRs and check their D3 vs D5 first event against the GitHub UI.

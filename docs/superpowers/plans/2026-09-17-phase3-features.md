# Phase 3 — Point-in-Time Features — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `data/features/features.parquet` — one row per modelling PR with every blueprint-R1 feature computed strictly from information knowable at `created_at` — plus an automated leakage audit, a manual spot-check tool, and a feature dictionary generated from code so it cannot drift.

**Architecture:** Four independent, pure feature-group functions in `features.py` (`static_features`, `at_open_features`, `replay_features`, `repo_features`) composed by `build()`. The leakage-critical group extends Phase 2's `replay.History` and keeps its independent brute-force twin. A single `COLUMN_SPEC` registry is the contract: the dictionary is rendered from it and a test asserts the built table's columns equal it.

**Tech Stack:** Python 3.13, pandas 2.3, numpy, pyarrow, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-17-phase3-features-design.md`

**Deviation from spec, recorded here:** §5.2 says `global_merge_rate` is positional and existing callers pass it. This plan makes it `global_merge_rate: float | None = None` and raises `ValueError` if `author` is given without it. The author-history path is the only consumer, so Phase 2's `baseline.py`, `eda_report.py`, and their tests are untouched. Same guarantee, less churn.

## Global Constraints

- All pandas datetimes are tz-aware UTC; conversion to naive numpy goes ONLY through `replay._ns`.
- `features_at(t, ...)` sees only rows with `created_at < t` via `np.searchsorted(..., side="left")`. No other code path may read rows at or after `t`. Every new key in `features_at` obeys this.
- Resolvability predicate for any label-derived feature: `created_at <= t − 168h` OR (`first_event_at` not NaT AND `first_event_at < t`).
- `n_prior_merged_here` uses `merged_at < t`, never `merged == True`.
- Shrinkage: `(count + α·prior) / (n + α)`, α = 5.0. Priors `g` (D5 slow rate) and `g_merge` (merge rate) are computed over Scenario A training rows (`created_at < 2026-01-01T00:00:00Z`) once, when the table is built.
- Timeline events with `created_at > pr.created_at` are "post-open"; events at exactly `pr.created_at` are at-open. `reviewer_requested_at_open` uses `created_at <= pr.created_at + 60s`.
- Row set = `splits.modelling_prs(pr_tier2)` ∩ `cohort_qc.kept_repos()`, inner-joined with the D5 label — identical to Phase 2.
- Fidelity flags (`diff_is_exact`, `timeline_may_be_truncated`, `body_edited`, `trailing_window_complete`) have status `flag` and are never features.
- Seed everywhere: `20260912`. Tests in `tests/`, `python -m pytest tests -q`, output pristine.
- Commits on branch `phase3`; end commit messages with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Real-data steps run only if `data/cohort/kept.json` exists AND `data/phase2_gate.json` shows every check passing; otherwise the step is reported DEFERRED.

---

## File structure

| File | Responsibility |
|---|---|
| `replay.py` (modify) | `History` carries `author`, `merged`; `features_at` gains 8 keys; `brute_force_features` recomputes them |
| `tests/conftest.py` (modify) | `replay_toy` tier1 gains `merged_at`; docstring gains the author-history arithmetic |
| `tests/test_replay.py` (modify) | author-history tests; brute-force equality extended |
| `features.py` (create) | `COLUMN_SPEC`, four group functions, `build`, `audit`, `explain`, `render_dictionary`, CLI |
| `tests/test_features.py` (create) | toys for each group; `build` on a two-repo toy; audit on the toy |
| `docs/feature_dictionary.md` (generated) | rendered from `COLUMN_SPEC` by `python features.py --dictionary` |
| `data/features/features.parquet`, `data/phase3_gate.json` (outputs) | gitignored |

---

### Task 1: Extend `replay.py` — author history, trailing-7d, author slow-rate

**Files:**
- Modify: `replay.py` (class `History`, `features_at`, `brute_force_features`)
- Modify: `tests/conftest.py` (`replay_toy`)
- Modify: `tests/test_replay.py`

**Interfaces:**
- Consumes: existing `History`, `_ns`, `replay_toy` fixture (`tier1` cols `repo, pr_id, created_at, closed_at, author_login`; `labels` cols `pr_id, first_event_at, is_slow`).
- Produces:
  - `History.__init__(..., *, author=None, merged_at=None, window_start, threshold_h=168.0, trailing_days=90)`; `History.from_frames` reads `author_login` and (if present) `merged_at` from tier1.
  - `History.features_at(t, global_rate, global_merge_rate=None, alpha=5.0, *, author=None) -> dict` with the 4 existing keys plus `prs_opened_trailing_7d, is_first_pr_here, n_prior_prs_here, n_prior_merged_here, prior_merge_rate_here, days_since_first_pr_here, author_prior_slow_rate_here, author_prior_n`.
  - `brute_force_features(tier1, labels, repo, t, global_rate, alpha=5.0, threshold_h=168.0, trailing_days=90, global_merge_rate=None, author=None) -> dict` with the same keys (minus `trailing_window_complete`).
  - `replay.REPLAY_KEYS: tuple[str, ...]` — every key both functions return, for the audit.

- [ ] **Step 1: Extend the fixture (append to `replay_toy`, do not change existing values)**

In `tests/conftest.py`, add `merged_at` to the `tier1` frame and extend the docstring:

```python
    tier1 = pd.DataFrame({
        "repo": ["r"] * 5,
        "pr_id": ["P0", "P1", "P2", "P3", "P4"],
        "created_at": [ts("2023-06-01"), ts("2024-03-01"), ts("2024-03-10"),
                       ts("2024-04-15"), ts("2024-04-19")],
        "closed_at": [ts("2024-06-01"), pd.NaT, ts("2024-03-20"), pd.NaT, pd.NaT],
        "merged_at": [ts("2024-06-01"), pd.NaT, ts("2024-03-20"), pd.NaT, pd.NaT],
        "author_login": ["u0", "u1", "u2", "u1", "u3"],
    })
```

Append to the docstring (inside the triple quotes, after the `complete` line):

```
      merged_at: P0 2024-06-01, P2 2024-03-20, others NaT (coherent with closed_at).
      prs_opened_trailing_7d @t = P3 (4-15), P4 (4-19)            = 2

      Author history @t=2024-04-21, alpha=5, g=0.5, g_merge=0.4:
        u1: prior P1,P3 -> n=2; merged 0 -> rate (0+2.0)/7 = 0.285714
            days_since_first = 4-21 - 3-01 = 51
            slow: P1 (0, resolvable), P3 (0, fe 4-16<t) -> k=2, n_slow 0 -> (0+2.5)/7 = 0.357143
        u2: prior P2 -> n=1; merged 1 (3-20 < t) -> (1+2.0)/6 = 0.5; days 42
            slow: P2 (1, resolvable) -> k=1 -> (1+2.5)/6 = 0.583333
        u0: prior P0 -> n=1; merged 0 (6-01 > t: NOT prior knowledge) -> (0+2.0)/6 = 0.333333
            days = 325; slow: unlabelled -> k=0 -> 0.5 (prior)
        zz (unseen) and None: first-PR values, days NaN, rates = priors
```

- [ ] **Step 2: Write the failing tests (append to `tests/test_replay.py`)**

```python
T = pd.Timestamp("2024-04-21", tz="UTC")
G, GM = 0.5, 0.4


def _feat(replay_toy, author):
    tier1, labels = replay_toy
    hist = replay.History.from_frames("r", tier1, labels, WS)
    return hist.features_at(T, G, GM, author=author)


def test_trailing_7d_count(replay_toy):
    assert _feat(replay_toy, None)["prs_opened_trailing_7d"] == 2


def test_author_u1_history(replay_toy):
    f = _feat(replay_toy, "u1")
    assert f["is_first_pr_here"] is False
    assert f["n_prior_prs_here"] == 2 and f["n_prior_merged_here"] == 0
    assert f["prior_merge_rate_here"] == pytest.approx(2.0 / 7)
    assert f["days_since_first_pr_here"] == pytest.approx(51.0)
    assert f["author_prior_n"] == 2
    assert f["author_prior_slow_rate_here"] == pytest.approx(2.5 / 7)


def test_author_u2_merged_before_t_counts(replay_toy):
    f = _feat(replay_toy, "u2")
    assert f["n_prior_prs_here"] == 1 and f["n_prior_merged_here"] == 1
    assert f["prior_merge_rate_here"] == pytest.approx(3.0 / 6)
    assert f["author_prior_slow_rate_here"] == pytest.approx(3.5 / 6)


def test_author_u0_merged_after_t_does_not_count(replay_toy):
    f = _feat(replay_toy, "u0")
    assert f["n_prior_prs_here"] == 1 and f["n_prior_merged_here"] == 0
    assert f["prior_merge_rate_here"] == pytest.approx(2.0 / 6)
    assert f["author_prior_n"] == 0 and f["author_prior_slow_rate_here"] == pytest.approx(G)


def test_unseen_and_deleted_author_get_first_pr_values(replay_toy):
    for a in ("zz", None):
        f = _feat(replay_toy, a)
        assert f["is_first_pr_here"] is True
        assert f["n_prior_prs_here"] == 0 and f["n_prior_merged_here"] == 0
        assert np.isnan(f["days_since_first_pr_here"])
        assert f["prior_merge_rate_here"] == pytest.approx(GM)
        assert f["author_prior_slow_rate_here"] == pytest.approx(G)


def test_author_without_merge_rate_raises(replay_toy):
    tier1, labels = replay_toy
    hist = replay.History.from_frames("r", tier1, labels, WS)
    with pytest.raises(ValueError):
        hist.features_at(T, G, author="u1")


def test_phase2_keys_unchanged_by_author(replay_toy):
    a = _feat(replay_toy, None)
    b = _feat(replay_toy, "u1")
    for k in ("open_backlog_at_t", "trailing_90d_slow_rate", "trailing_n",
              "trailing_window_complete"):
        assert a[k] == b[k]


def test_brute_force_matches_all_keys(replay_toy):
    tier1, labels = replay_toy
    hist = replay.History.from_frames("r", tier1, labels, WS)
    for t in (T, pd.Timestamp("2024-02-01", tz="UTC"), pd.Timestamp("2024-07-01", tz="UTC")):
        for author in (None, "u0", "u1", "u2", "zz"):
            a = hist.features_at(t, G, GM, author=author)
            b = replay.brute_force_features(tier1, labels, "r", t, G,
                                            global_merge_rate=GM, author=author)
            for k in replay.REPLAY_KEYS:
                if isinstance(a[k], float) and np.isnan(a[k]):
                    assert np.isnan(b[k]), (t, author, k)
                else:
                    assert a[k] == pytest.approx(b[k]), (t, author, k)
```

Also update the EXISTING `test_brute_force_matches_features_at_on_toy` so it still passes: it compares the three Phase 2 keys — leave it as is; the new function returns a superset.

- [ ] **Step 3: Run to verify they fail**

Run: `python -m pytest tests/test_replay.py -q`
Expected: new tests FAIL (`TypeError: features_at() got an unexpected keyword argument 'author'` / `AttributeError: REPLAY_KEYS`); existing tests still pass.

- [ ] **Step 4: Implement in `replay.py`**

Replace `History.__init__`, `from_frames`, and `features_at`, and `brute_force_features`, with:

```python
REPLAY_KEYS = (
    "open_backlog_at_t", "trailing_90d_slow_rate", "trailing_n",
    "prs_opened_trailing_7d",
    "is_first_pr_here", "n_prior_prs_here", "n_prior_merged_here",
    "prior_merge_rate_here", "days_since_first_pr_here",
    "author_prior_slow_rate_here", "author_prior_n",
)


@dataclass(eq=False)
class History:
    repo: str
    created: np.ndarray        # datetime64[ns], sorted ascending
    closed: np.ndarray         # datetime64[ns], NaT if open
    merged: np.ndarray         # datetime64[ns], NaT if not merged
    first_event: np.ndarray    # datetime64[ns], NaT if none
    is_slow: np.ndarray        # float: 1.0 / 0.0 / nan (nan = no label, pre-window)
    author: np.ndarray         # object: login or None (deleted account)
    window_start: np.datetime64
    threshold_h: float = 168.0
    trailing_days: int = 90

    def __init__(self, repo, created_at, closed_at, first_event_at, is_slow, *,
                 author=None, merged_at=None, window_start, threshold_h=168.0,
                 trailing_days=90):
        created_at = np.asarray(created_at)
        order = np.argsort(created_at, kind="stable")
        n = len(created_at)
        self.repo = repo
        self.created = created_at[order]
        self.closed = np.asarray(closed_at)[order]
        self.first_event = np.asarray(first_event_at)[order]
        self.is_slow = np.asarray(is_slow, dtype=float)[order]
        self.merged = (np.asarray(merged_at)[order] if merged_at is not None
                       else np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]"))
        self.author = (np.asarray(author, dtype=object)[order] if author is not None
                       else np.full(n, None, dtype=object))
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
        merged = _ns(t1["merged_at"]) if "merged_at" in t1.columns else None
        author = (t1["author_login"].astype(object).where(t1["author_login"].notna(), None)
                  .to_numpy(dtype=object) if "author_login" in t1.columns else None)
        return cls(repo, _ns(t1["created_at"]), _ns(t1["closed_at"]), _ns(fe),
                   sl.to_numpy(), author=author, merged_at=merged,
                   window_start=_ns(window_start))

    def features_at(self, t: pd.Timestamp, global_rate: float,
                    global_merge_rate: float | None = None, alpha: float = 5.0, *,
                    author: str | None = None) -> dict:
        """Every value here is computed from rows with created_at < t. Nothing else.

        `author` enables the author-history keys; it requires `global_merge_rate` for
        the merge-rate prior. A None author (deleted account) yields first-PR values --
        a category, not a crash."""
        if author is not None and global_merge_rate is None:
            raise ValueError("author-history features need global_merge_rate")

        t64 = _ns(t)
        n = int(np.searchsorted(self.created, t64, side="left"))   # created < t only
        created, closed = self.created[:n], self.closed[:n]
        fe, sl = self.first_event[:n], self.is_slow[:n]
        merged, who = self.merged[:n], self.author[:n]

        backlog = int(np.sum(np.isnat(closed) | (closed > t64)))
        trailing_7d = int(np.sum(created >= t64 - np.timedelta64(7, "D")))

        lo = t64 - np.timedelta64(self.trailing_days, "D")
        thr = t64 - np.timedelta64(int(self.threshold_h * 3600), "s")
        in_window = created >= lo
        labelled = ~np.isnan(sl)
        resolvable = (created <= thr) | (~np.isnat(fe) & (fe < t64))
        m = in_window & labelled & resolvable
        k = int(m.sum())
        n_slow = float(np.nansum(sl[m]))
        rate = (n_slow + alpha * global_rate) / (k + alpha)

        # --- author history: the author's own prior PRs in this repo ---------------
        mine = (who == author) if author is not None else np.zeros(n, dtype=bool)
        n_prior = int(mine.sum())
        n_prior_merged = int(np.sum(mine & ~np.isnat(merged) & (merged < t64)))
        gm = global_merge_rate if global_merge_rate is not None else 0.0
        merge_rate = (n_prior_merged + alpha * gm) / (n_prior + alpha)
        days_since_first = (float((t64 - created[mine].min()) / np.timedelta64(1, "D"))
                            if n_prior else float("nan"))
        am = mine & labelled & resolvable            # whole history, no 90-day window
        ak = int(am.sum())
        a_slow = float(np.nansum(sl[am]))
        author_rate = (a_slow + alpha * global_rate) / (ak + alpha)

        return {
            "open_backlog_at_t": backlog,
            "trailing_90d_slow_rate": float(rate),
            "trailing_n": k,
            "trailing_window_complete": bool(lo >= self.window_start),
            "prs_opened_trailing_7d": trailing_7d,
            "is_first_pr_here": bool(n_prior == 0),
            "n_prior_prs_here": n_prior,
            "n_prior_merged_here": n_prior_merged,
            "prior_merge_rate_here": float(merge_rate),
            "days_since_first_pr_here": days_since_first,
            "author_prior_slow_rate_here": float(author_rate),
            "author_prior_n": ak,
        }


def brute_force_features(tier1: pd.DataFrame, labels: pd.DataFrame, repo: str,
                         t: pd.Timestamp, global_rate: float, alpha: float = 5.0,
                         threshold_h: float = 168.0, trailing_days: int = 90,
                         global_merge_rate: float | None = None,
                         author: str | None = None) -> dict:
    """Independent pandas re-derivation of features_at, for auditing.

    Deliberately does not touch History. If the two ever disagree, one is wrong."""
    h = tier1[tier1["repo"] == repo].merge(
        labels[["pr_id", "first_event_at", "is_slow"]], on="pr_id", how="left")
    h = h[h["created_at"] < t]
    backlog = int((h["closed_at"].isna() | (h["closed_at"] > t)).sum())
    trailing_7d = int((h["created_at"] >= t - pd.Timedelta(days=7)).sum())
    lo = t - pd.Timedelta(days=trailing_days)
    thr = t - pd.Timedelta(hours=threshold_h)
    resolvable = (h["created_at"] <= thr) | (h["first_event_at"].notna() & (h["first_event_at"] < t))
    w = h[(h["created_at"] >= lo) & h["is_slow"].notna() & resolvable]
    k = len(w)
    n_slow = float(w["is_slow"].astype(float).sum())

    mine = h[h["author_login"] == author] if author is not None else h.iloc[0:0]
    n_prior = len(mine)
    merged_col = mine["merged_at"] if "merged_at" in mine.columns else pd.Series(pd.NaT, index=mine.index)
    n_prior_merged = int((merged_col.notna() & (merged_col < t)).sum())
    gm = global_merge_rate if global_merge_rate is not None else 0.0
    days = (float((t - mine["created_at"].min()).total_seconds() / 86400.0)
            if n_prior else float("nan"))
    am = mine[mine["is_slow"].notna()
              & ((mine["created_at"] <= thr)
                 | (mine["first_event_at"].notna() & (mine["first_event_at"] < t)))]
    ak = len(am)
    a_slow = float(am["is_slow"].astype(float).sum())
    return {
        "open_backlog_at_t": backlog,
        "trailing_90d_slow_rate": (n_slow + alpha * global_rate) / (k + alpha),
        "trailing_n": k,
        "prs_opened_trailing_7d": trailing_7d,
        "is_first_pr_here": bool(n_prior == 0),
        "n_prior_prs_here": n_prior,
        "n_prior_merged_here": n_prior_merged,
        "prior_merge_rate_here": (n_prior_merged + alpha * gm) / (n_prior + alpha),
        "days_since_first_pr_here": days,
        "author_prior_slow_rate_here": (a_slow + alpha * global_rate) / (ak + alpha),
        "author_prior_n": ak,
    }
```

- [ ] **Step 5: Run to verify they pass**

Run: `python -m pytest tests/test_replay.py tests/test_baseline.py -q`
Expected: all pass (Phase 2 tests untouched and still green — `baseline.py` calls with the old positional signature, which is still valid).

- [ ] **Step 6: Full suite, stage, commit**

Run: `python -m pytest tests -q` → all pass, no warnings.
```bash
git add replay.py tests/conftest.py tests/test_replay.py
git commit -m "feat(replay): author history, trailing-7d, author slow-rate; brute-force twin extended"
```

---

### Task 2: `features.py` skeleton — `COLUMN_SPEC`, `static_features`, `repo_features`

**Files:**
- Create: `features.py`, `tests/test_features.py`

**Interfaces:**
- Consumes: `pr_tier2` columns `repo, pr_id, number, created_at, author_created_at, author_is_deleted, is_cross_repository, body_current, last_edited_at`; `repo_meta` columns `repo, created_at, n_assignable_users, n_mentionable_users, owner_type, has_codeowners, has_pr_template, has_contributing, n_ci_workflows, language_dominant, default_branch`.
- Produces:
  - `features.COLUMN_SPEC: dict[str, dict]` — every output column → `{"group", "status", "dtype", "nullable", "derivation"}`. Groups: `key, label, static, at_open, replay, repo, flag`. Status: `key, label, static, reconstructed, replay, snapshot, flag`.
  - `features.KEYS = ["repo", "pr_id", "number", "created_at"]`, `features.LABEL_COLS = ["is_slow", "wait_h", "wait_h_censored", "event_observed", "never_reviewed_30d"]`.
  - `features.static_features(prs) -> pd.DataFrame` indexed by `pr_id`.
  - `features.repo_features(repo_meta, prs) -> pd.DataFrame` indexed by `pr_id`.
  - `features.feature_columns() -> list[str]` — columns Phase 4 may train on (status in static/reconstructed/replay/snapshot).

- [ ] **Step 1: Write the failing tests**

`tests/test_features.py`:
```python
import numpy as np
import pandas as pd
import pytest
import features

UTC = "UTC"


def ts(s):
    return pd.Timestamp(s, tz=UTC)


@pytest.fixture
def prs_toy():
    return pd.DataFrame({
        "repo": ["o/r", "o/r", "o/s"],
        "pr_id": ["A", "B", "C"],
        "number": [1, 2, 7],
        "created_at": [ts("2025-03-03T10:00"), ts("2025-03-08T23:30"), ts("2025-06-01T00:00")],  # Mon, Sat, Sun
        "author_login": ["alice", None, "carol"],
        "author_created_at": [ts("2020-03-03T10:00"), pd.NaT, ts("2025-05-31T00:00")],
        "author_is_deleted": [False, True, False],
        "is_cross_repository": [True, False, False],
        "body_current": ["hello world", None, ""],
        "last_edited_at": [pd.NaT, ts("2025-03-09"), pd.NaT],
    })


@pytest.fixture
def repo_meta_toy():
    return pd.DataFrame({
        "repo": ["o/r", "o/s"],
        "created_at": [ts("2024-03-03T10:00"), ts("2025-05-01")],
        "n_assignable_users": [5, 1], "n_mentionable_users": [50, 3],
        "owner_type": ["Organization", "User"],
        "has_codeowners": [True, False], "has_pr_template": [False, False],
        "has_contributing": [True, False], "n_ci_workflows": [3, 0],
        "language_dominant": ["Go", "Python"], "default_branch": ["main", "master"],
    })


def test_static_features(prs_toy):
    s = features.static_features(prs_toy)
    assert list(s.index) == ["A", "B", "C"]
    assert s.loc["A", "created_hour_utc"] == 10 and s.loc["A", "created_dayofweek"] == 0
    assert s.loc["B", "is_weekend"] == True and s.loc["C", "is_weekend"] == True   # noqa: E712
    assert s.loc["A", "is_weekend"] == False                                        # noqa: E712
    assert s.loc["A", "author_account_age_days"] == pytest.approx(1826.0)        # 5y incl. leap day
    assert np.isnan(s.loc["B", "author_account_age_days"])                          # deleted
    assert s.loc["C", "author_account_age_days"] == pytest.approx(1.0)
    assert s.loc["A", "body_len"] == 11 and s.loc["A", "has_body"] == True         # noqa: E712
    assert s.loc["B", "body_len"] == 0 and s.loc["B", "has_body"] == False         # noqa: E712
    assert s.loc["B", "body_edited"] == True and s.loc["A", "body_edited"] == False  # noqa: E712
    assert s.loc["A", "is_cross_repository"] == True                                # noqa: E712


def test_repo_features(prs_toy, repo_meta_toy):
    r = features.repo_features(repo_meta_toy, prs_toy)
    assert list(r.index) == ["A", "B", "C"]
    assert r.loc["A", "owner_is_org"] == True and r.loc["C", "owner_is_org"] == False  # noqa: E712
    assert r.loc["A", "n_assignable_users"] == 5 and r.loc["C", "n_ci_workflows"] == 0
    assert r.loc["A", "language_dominant"] == "Go"
    assert r.loc["A", "repo_age_days_at_open"] == pytest.approx(365.0)
    assert r.loc["C", "repo_age_days_at_open"] == pytest.approx(31.0)


def test_column_spec_is_the_contract():
    spec = features.COLUMN_SPEC
    assert set(features.KEYS) <= set(spec) and set(features.LABEL_COLS) <= set(spec)
    for col, meta in spec.items():
        assert set(meta) == {"group", "status", "dtype", "nullable", "derivation"}, col
        assert meta["status"] in {"key", "label", "static", "reconstructed", "replay", "snapshot", "flag"}, col
    trainable = features.feature_columns()
    assert "is_slow" not in trainable and "pr_id" not in trainable
    assert "diff_is_exact" not in trainable and "body_edited" not in trainable
    assert "trailing_90d_slow_rate" in trainable and "author_account_age_days" in trainable
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_features.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'features'`

- [ ] **Step 3: Implement `features.py` (skeleton + two groups + registry)**

```python
"""Phase 3: the feature table. One row per modelling PR; every value knowable at
created_at.

Four independent groups composed by build():
  static_features   -- known at open from the PR object itself
  at_open_features  -- reverse-replayed from the current snapshot via timeline events
  replay_features   -- chronological replay over Tier 1 history (replay.History)
  repo_features     -- repo-level, transferable; snapshots of HEAD at collection

COLUMN_SPEC is the contract. docs/feature_dictionary.md is rendered from it, and a test
asserts the built table's columns equal it, so the two cannot drift. Phase 4 trains only
on feature_columns(); keys, labels and fidelity flags are never features."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import cohort_qc
import labels
import load
import replay
import splits

log = logging.getLogger("features")

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "features" / "features.parquet"
GATE_JSON = ROOT / "data" / "phase3_gate.json"
DICT_MD = ROOT / "docs" / "feature_dictionary.md"

SEED = 20260912
ALPHA = 5.0
REVIEW_REQUEST_GRACE = pd.Timedelta(seconds=60)
TIMELINE_CAP = 60            # queries.N_TIMELINE -- a PR with exactly this many rows may be truncated

KEYS = ["repo", "pr_id", "number", "created_at"]
LABEL_COLS = ["is_slow", "wait_h", "wait_h_censored", "event_observed", "never_reviewed_30d"]


def _c(group, status, dtype, derivation, nullable=False):
    return {"group": group, "status": status, "dtype": dtype, "nullable": nullable,
            "derivation": derivation}


COLUMN_SPEC: dict[str, dict] = {
    # --- keys ---------------------------------------------------------------
    "repo":       _c("key", "key", "str", "owner/name"),
    "pr_id":      _c("key", "key", "str", "GraphQL node id"),
    "number":     _c("key", "key", "int", "PR number"),
    "created_at": _c("key", "key", "datetime[UTC]", "PR open time; the prediction instant t"),
    # --- labels (Phase 2, D5) ----------------------------------------------
    "is_slow":            _c("label", "label", "bool", "never_reviewed_30d OR wait_h > 168"),
    "wait_h":             _c("label", "label", "float", "hours to first human review (D5)", nullable=True),
    "wait_h_censored":    _c("label", "label", "float", "min(wait_h, 720)"),
    "event_observed":     _c("label", "label", "bool", "reviewed within 720h"),
    "never_reviewed_30d": _c("label", "label", "bool", "no D5 event within 720h"),
    # --- static: known at open from the PR object ---------------------------
    "created_hour_utc":        _c("static", "static", "int", "created_at.hour"),
    "created_dayofweek":       _c("static", "static", "int", "created_at.dayofweek, Mon=0"),
    "is_weekend":              _c("static", "static", "bool", "dayofweek >= 5"),
    "is_cross_repository":     _c("static", "static", "bool", "PR from a fork"),
    "author_account_age_days": _c("static", "static", "float", "(created_at - author.createdAt).days; immutable", nullable=True),
    "body_len":                _c("static", "static", "int", "len(body); body is editable -> approximate"),
    "has_body":                _c("static", "static", "bool", "body_len > 0; approximate"),
    "body_edited":             _c("flag", "flag", "bool", "last_edited_at not null -- POST-OPEN info, never a feature"),
    # --- at-open reconstructions -------------------------------------------
    "is_draft_at_open":            _c("at_open", "reconstructed", "bool", "is_draft_current inverted once per post-open Ready/ConvertToDraft event (parity)"),
    "n_labels_at_open":            _c("at_open", "reconstructed", "int", "n_labels_current - post-open Labeled + post-open Unlabeled, floored at 0"),
    "title_len_at_open":           _c("at_open", "reconstructed", "int", "len(previous_title of earliest post-open RenamedTitleEvent, else title_current)"),
    "base_is_default":             _c("at_open", "reconstructed", "bool", "base_ref_at_open == repo default branch"),
    "reviewer_requested_at_open":  _c("at_open", "static", "bool", "any ReviewRequestedEvent within 60s of open"),
    "n_reviewers_requested_at_open": _c("at_open", "static", "int", "count of those"),
    "requested_team_at_open":      _c("at_open", "static", "bool", "any of those with requested_reviewer_type == Team"),
    "additions_at_open":           _c("at_open", "reconstructed", "float", "sum of commit additions with authored_date <= created_at (parse.py)", nullable=True),
    "deletions_at_open":           _c("at_open", "reconstructed", "float", "as above", nullable=True),
    "n_commits_at_open":           _c("at_open", "reconstructed", "float", "as above", nullable=True),
    "diff_is_exact":               _c("flag", "flag", "bool", "single-commit PR: final diff == at-open diff"),
    "timeline_may_be_truncated":   _c("flag", "flag", "bool", "PR has exactly 60 timeline rows (the first:60 cap)"),
    # --- replay: chronological, created_at < t, proven by brute-force audit --
    "open_backlog_at_t":           _c("replay", "replay", "int", "prior PRs still open at t"),
    "prs_opened_trailing_7d":      _c("replay", "replay", "int", "prior PRs with created_at >= t-7d"),
    "trailing_90d_slow_rate":      _c("replay", "replay", "float", "shrunk D5 slow rate over resolvable prior PRs in [t-90d, t)"),
    "trailing_n":                  _c("replay", "replay", "int", "rows behind trailing_90d_slow_rate"),
    "trailing_window_complete":    _c("flag", "flag", "bool", "t-90d >= window start"),
    "is_first_pr_here":            _c("replay", "replay", "bool", "author has no prior PR in this repo"),
    "n_prior_prs_here":            _c("replay", "replay", "int", "author's prior PRs here"),
    "n_prior_merged_here":         _c("replay", "replay", "int", "of those, merged_at < t"),
    "prior_merge_rate_here":       _c("replay", "replay", "float", "shrunk toward global merge rate"),
    "days_since_first_pr_here":    _c("replay", "replay", "float", "t - author's first PR here", nullable=True),
    "author_prior_slow_rate_here": _c("replay", "replay", "float", "shrunk D5 slow rate over author's resolvable prior PRs (whole history)"),
    "author_prior_n":              _c("replay", "replay", "int", "rows behind author_prior_slow_rate_here"),
    # --- repo-level, transferable (snapshots of HEAD at collection) ---------
    "n_assignable_users":   _c("repo", "snapshot", "int", "maintainer capacity proxy"),
    "n_mentionable_users":  _c("repo", "snapshot", "int", "community size proxy"),
    "owner_is_org":         _c("repo", "snapshot", "bool", "owner_type == Organization"),
    "has_codeowners":       _c("repo", "snapshot", "bool", "CODEOWNERS present at HEAD"),
    "has_pr_template":      _c("repo", "snapshot", "bool", "PR template present at HEAD"),
    "has_contributing":     _c("repo", "snapshot", "bool", "CONTRIBUTING present at HEAD"),
    "n_ci_workflows":       _c("repo", "snapshot", "int", "files under .github/workflows at HEAD"),
    "language_dominant":    _c("repo", "snapshot", "str", "largest language by bytes"),
    "repo_age_days_at_open": _c("repo", "static", "float", "(created_at - repo.createdAt).days; point-in-time safe"),
}

TRAINABLE_STATUS = {"static", "reconstructed", "replay", "snapshot"}


def feature_columns() -> list[str]:
    return [c for c, m in COLUMN_SPEC.items() if m["status"] in TRAINABLE_STATUS]


# ---------------------------------------------------------------------------
# Group 1: static
# ---------------------------------------------------------------------------

def static_features(prs: pd.DataFrame) -> pd.DataFrame:
    p = prs.set_index("pr_id")
    created = p["created_at"]
    age = (created - pd.to_datetime(p["author_created_at"], utc=True)).dt.total_seconds() / 86400.0
    body = p["body_current"].fillna("").astype(str)
    return pd.DataFrame({
        "created_hour_utc": created.dt.hour.astype(int),
        "created_dayofweek": created.dt.dayofweek.astype(int),
        "is_weekend": created.dt.dayofweek >= 5,
        "is_cross_repository": (p["is_cross_repository"] == True),          # noqa: E712
        "author_account_age_days": age.astype(float),                         # NaN if deleted
        "body_len": body.str.len().astype(int),
        "has_body": body.str.len() > 0,
        "body_edited": p["last_edited_at"].notna(),
    }, index=p.index)


# ---------------------------------------------------------------------------
# Group 4: repo-level
# ---------------------------------------------------------------------------

def repo_features(repo_meta: pd.DataFrame, prs: pd.DataFrame) -> pd.DataFrame:
    m = repo_meta.set_index("repo")
    p = prs.set_index("pr_id")
    joined = p[["repo", "created_at"]].join(m, on="repo", rsuffix="_repo")
    repo_created = pd.to_datetime(joined["created_at_repo"], utc=True)
    return pd.DataFrame({
        "n_assignable_users": joined["n_assignable_users"].astype(int),
        "n_mentionable_users": joined["n_mentionable_users"].astype(int),
        "owner_is_org": joined["owner_type"] == "Organization",
        "has_codeowners": joined["has_codeowners"].astype(bool),
        "has_pr_template": joined["has_pr_template"].astype(bool),
        "has_contributing": joined["has_contributing"].astype(bool),
        "n_ci_workflows": joined["n_ci_workflows"].astype(int),
        "language_dominant": joined["language_dominant"].astype(str),
        "repo_age_days_at_open": ((joined["created_at"] - repo_created).dt.total_seconds() / 86400.0).astype(float),
    }, index=p.index)
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_features.py -q`
Expected: `3 passed`

- [ ] **Step 5: Stage and commit**

```bash
git add features.py tests/test_features.py
git commit -m "feat(features): COLUMN_SPEC registry, static and repo feature groups"
```

---

### Task 3: `at_open_features` — reverse-replay from timeline events

**Files:**
- Modify: `features.py` (add the function)
- Modify: `tests/test_features.py` (add fixture + tests)

**Interfaces:**
- Consumes: `prs` columns `pr_id, repo, created_at, is_draft_current, n_labels_current, title_current, base_ref_current, additions_at_open, deletions_at_open, n_commits_at_open, diff_is_exact`; `timeline` columns `pr_id, event_type, created_at, previous_title, previous_ref, requested_reviewer_type`; `repo_meta` columns `repo, default_branch`.
- Produces: `features.at_open_features(prs, timeline, repo_meta) -> pd.DataFrame` indexed by `pr_id` with the 13 `at_open`/flag columns from `COLUMN_SPEC`.

- [ ] **Step 1: Write the failing tests (append to `tests/test_features.py`)**

```python
@pytest.fixture
def at_open_toy():
    """Four PRs, all opened 2025-01-01T00:00Z.

    A: currently draft=False, 2 labels, title 'new', base 'main'.
       post-open: ReadyForReview @+1h (so it WAS draft at open),
                  Labeled @+2h, Labeled @+3h, Unlabeled @+4h  -> at open: 2-2+1 = 1
                  RenamedTitle @+5h previous='old', RenamedTitle @+6h previous='new'-ish
                  -> earliest previous wins: 'old' (len 3)
                  BaseRefChanged @+7h previous='dev' -> base_at_open 'dev' != default
                  ReviewRequested @+1s (User), ReviewRequested @+30s (Team), ReviewRequested @+2h (User; too late)
                  -> requested=True, n=2, team=True
    B: draft=True currently, ConvertToDraft @+1h, ReadyForReview @+2h -> two flips -> draft at open = True
       no other events; base 'main' == default -> base_is_default True
    C: no timeline rows at all -> everything = current; requested False
    D: exactly 60 timeline rows (all LabeledEvent post-open) -> truncated flag; n_labels_at_open = max(0, 1-60) = 0
    """
    t0 = ts("2025-01-01T00:00")
    prs = pd.DataFrame({
        "repo": ["o/r"] * 4, "pr_id": list("ABCD"), "created_at": [t0] * 4,
        "is_draft_current": [False, True, False, False],
        "n_labels_current": [2, 0, 3, 1],
        "title_current": ["new", "b", "c-title", "d"],
        "base_ref_current": ["main", "main", "release", "main"],
        "additions_at_open": [10.0, np.nan, 5.0, 1.0],
        "deletions_at_open": [1.0, np.nan, 0.0, 0.0],
        "n_commits_at_open": [1.0, np.nan, 2.0, 1.0],
        "diff_is_exact": [True, False, False, True],
    })
    h = lambda x: t0 + pd.Timedelta(hours=x)
    ev = [
        ("A", "ReadyForReviewEvent", h(1), None, None, None),
        ("A", "LabeledEvent", h(2), None, None, None),
        ("A", "LabeledEvent", h(3), None, None, None),
        ("A", "UnlabeledEvent", h(4), None, None, None),
        ("A", "RenamedTitleEvent", h(5), "old", None, None),
        ("A", "RenamedTitleEvent", h(6), "newer", None, None),
        ("A", "BaseRefChangedEvent", h(7), None, "dev", None),
        ("A", "ReviewRequestedEvent", t0 + pd.Timedelta(seconds=1), None, None, "User"),
        ("A", "ReviewRequestedEvent", t0 + pd.Timedelta(seconds=30), None, None, "Team"),
        ("A", "ReviewRequestedEvent", h(2), None, None, "User"),
        ("B", "ConvertToDraftEvent", h(1), None, None, None),
        ("B", "ReadyForReviewEvent", h(2), None, None, None),
    ] + [("D", "LabeledEvent", h(1 + i), None, None, None) for i in range(60)]
    timeline = pd.DataFrame(ev, columns=["pr_id", "event_type", "created_at",
                                         "previous_title", "previous_ref", "requested_reviewer_type"])
    repo_meta = pd.DataFrame({"repo": ["o/r"], "default_branch": ["main"]})
    return prs, timeline, repo_meta


def test_at_open_draft_parity(at_open_toy):
    a = features.at_open_features(*at_open_toy)
    assert a.loc["A", "is_draft_at_open"] == True    # noqa: E712  one flip from False
    assert a.loc["B", "is_draft_at_open"] == True    # noqa: E712  two flips from True
    assert a.loc["C", "is_draft_at_open"] == False   # noqa: E712  no events


def test_at_open_labels_title_base(at_open_toy):
    a = features.at_open_features(*at_open_toy)
    assert a.loc["A", "n_labels_at_open"] == 1 and a.loc["C", "n_labels_at_open"] == 3
    assert a.loc["D", "n_labels_at_open"] == 0                       # floored
    assert a.loc["A", "title_len_at_open"] == 3                      # 'old'
    assert a.loc["C", "title_len_at_open"] == 7                      # 'c-title'
    assert a.loc["A", "base_is_default"] == False                    # noqa: E712  'dev'
    assert a.loc["B", "base_is_default"] == True                     # noqa: E712
    assert a.loc["C", "base_is_default"] == False                    # noqa: E712  'release'


def test_at_open_review_requests_and_flags(at_open_toy):
    a = features.at_open_features(*at_open_toy)
    assert a.loc["A", "reviewer_requested_at_open"] == True          # noqa: E712
    assert a.loc["A", "n_reviewers_requested_at_open"] == 2          # +2h one excluded
    assert a.loc["A", "requested_team_at_open"] == True              # noqa: E712
    assert a.loc["C", "reviewer_requested_at_open"] == False         # noqa: E712
    assert a.loc["C", "n_reviewers_requested_at_open"] == 0
    assert a.loc["D", "timeline_may_be_truncated"] == True           # noqa: E712
    assert a.loc["A", "timeline_may_be_truncated"] == False          # noqa: E712
    assert a.loc["A", "diff_is_exact"] == True and np.isnan(a.loc["B", "additions_at_open"])  # noqa: E712
    assert set(a.columns) == {c for c, m in features.COLUMN_SPEC.items() if m["group"] in ("at_open", "flag")} - {"body_edited", "trailing_window_complete"}
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_features.py -q -k at_open`
Expected: FAIL — `AttributeError: module 'features' has no attribute 'at_open_features'`

- [ ] **Step 3: Implement (add to `features.py` after `static_features`)**

```python
# ---------------------------------------------------------------------------
# Group 2: at-open reconstruction (reverse replay from the current snapshot)
#
# The PR object holds CURRENT state. Timeline events are timestamped, so the state at
# open is recoverable by undoing every post-open event. Draft state flips on each
# Ready/ConvertToDraft event, so parity of post-open flips decides it; label count
# subtracts post-open adds and re-adds post-open removals; title and base use the
# previous_* value of the EARLIEST post-open change.
# ---------------------------------------------------------------------------

def at_open_features(prs: pd.DataFrame, timeline: pd.DataFrame,
                     repo_meta: pd.DataFrame) -> pd.DataFrame:
    p = prs.set_index("pr_id")
    idx = p.index

    if timeline.empty:
        tl = pd.DataFrame(columns=["pr_id", "event_type", "created_at", "previous_title",
                                   "previous_ref", "requested_reviewer_type", "pr_created_at"])
    else:
        tl = timeline.merge(p[["created_at"]].rename(columns={"created_at": "pr_created_at"}),
                            left_on="pr_id", right_index=True, how="inner")
    post = tl[tl["created_at"] > tl["pr_created_at"]]

    def count(frame, types):
        return frame[frame["event_type"].isin(types)].groupby("pr_id").size().reindex(idx, fill_value=0)

    def earliest_prev(types, col):
        f = post[post["event_type"].isin(types)].sort_values("created_at")
        return f.groupby("pr_id")[col].first().reindex(idx)

    flips = count(post, ["ReadyForReviewEvent", "ConvertToDraftEvent"])
    is_draft_current = (p["is_draft_current"] == True)                                   # noqa: E712
    is_draft_at_open = is_draft_current ^ (flips % 2 == 1)

    n_labels_at_open = (p["n_labels_current"].fillna(0).astype(int)
                        - count(post, ["LabeledEvent"]) + count(post, ["UnlabeledEvent"])).clip(lower=0)

    title_at_open = earliest_prev(["RenamedTitleEvent"], "previous_title").fillna(p["title_current"]).fillna("")
    base_at_open = earliest_prev(["BaseRefChangedEvent"], "previous_ref").fillna(p["base_ref_current"])
    default_branch = p["repo"].map(repo_meta.set_index("repo")["default_branch"])

    grace = tl["created_at"] <= tl["pr_created_at"] + REVIEW_REQUEST_GRACE
    rr = tl[(tl["event_type"] == "ReviewRequestedEvent") & grace]
    n_rr = rr.groupby("pr_id").size().reindex(idx, fill_value=0)
    team = rr[rr["requested_reviewer_type"] == "Team"].groupby("pr_id").size().reindex(idx, fill_value=0)

    n_rows = tl.groupby("pr_id").size().reindex(idx, fill_value=0)

    return pd.DataFrame({
        "is_draft_at_open": is_draft_at_open.astype(bool),
        "n_labels_at_open": n_labels_at_open.astype(int),
        "title_len_at_open": title_at_open.astype(str).str.len().astype(int),
        "base_is_default": (base_at_open == default_branch).astype(bool),
        "reviewer_requested_at_open": (n_rr > 0).astype(bool),
        "n_reviewers_requested_at_open": n_rr.astype(int),
        "requested_team_at_open": (team > 0).astype(bool),
        "additions_at_open": p["additions_at_open"].astype(float),
        "deletions_at_open": p["deletions_at_open"].astype(float),
        "n_commits_at_open": p["n_commits_at_open"].astype(float),
        "diff_is_exact": (p["diff_is_exact"] == True).astype(bool),                      # noqa: E712
        "timeline_may_be_truncated": (n_rows >= TIMELINE_CAP).astype(bool),
    }, index=idx)
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_features.py -q`
Expected: `6 passed`

- [ ] **Step 5: Stage and commit**

```bash
git add features.py tests/test_features.py
git commit -m "feat(features): at-open reconstruction from timeline events"
```

---

### Task 4: `replay_features`, `build`, and the parquet writer

**Files:**
- Modify: `features.py` (add `replay_features`, `build`, `write_table`, `main` build path)
- Modify: `tests/test_features.py` (two-repo end-to-end toy)

**Interfaces:**
- Consumes: Task 1's `replay.History.from_frames`, `features_at(t, g, g_merge, author=...)`, `replay.REPLAY_KEYS`; `splits.modelling_prs`, `splits.prepare_rows`, `splits.CUTOFF_A`, `splits.WINDOW_START`; `labels.first_human_event/label/DEFINITIONS/PRIMARY/ALL_STREAMS`; `cohort_qc.kept_repos`; `load.load_all`.
- Produces:
  - `features.replay_features(rows, tier1, label_d5, g, g_merge) -> pd.DataFrame` indexed by `pr_id` with the 12 replay/flag keys.
  - `features.build(kept: list[str] | None = None, frames: dict | None = None) -> tuple[pd.DataFrame, dict]` — the table (columns exactly `list(COLUMN_SPEC)`) and a `context` dict `{rows, tier1, label_d5, g, g_merge}` for the audit.
  - `features.write_table(table, path=OUT)` — Parquet with metadata `built_at`, `git_sha`.

- [ ] **Step 1: Write the failing test (append)**

```python
@pytest.fixture
def frames_toy():
    """Two repos, 6 human PRs + 1 bot PR + 1 out-of-window PR, all needed tables."""
    t = lambda s: ts(s)
    tier1 = pd.DataFrame({
        "repo": ["o/r"] * 5 + ["o/s"] * 3,
        "pr_id": ["r1", "r2", "r3", "r4", "r5", "s1", "s2", "s3"],
        "created_at": [t("2023-01-01"), t("2024-02-01"), t("2024-06-01"), t("2025-01-15"),
                       t("2026-02-01"), t("2024-03-01"), t("2025-09-01"), t("2026-03-01")],
        "closed_at": [t("2023-02-01"), t("2024-02-10"), pd.NaT, t("2025-02-01"), pd.NaT,
                      t("2024-03-05"), pd.NaT, pd.NaT],
        "merged_at": [t("2023-02-01"), t("2024-02-10"), pd.NaT, pd.NaT, pd.NaT,
                      t("2024-03-05"), pd.NaT, pd.NaT],
        "author_login": ["a", "a", "b", "a", "b", "c", "c", None],
    })
    pr2 = tier1[tier1["pr_id"] != "r1"].copy()                    # r1 is pre-window (Tier 1 only)
    pr2["number"] = range(1, len(pr2) + 1)
    pr2["author_is_bot"] = [False, False, False, False, False, True, False]   # s1 is a bot
    pr2["author_created_at"] = t("2020-01-01")
    pr2["author_is_deleted"] = pr2["author_login"].isna()
    pr2["is_cross_repository"] = False
    pr2["body_current"] = "x"; pr2["last_edited_at"] = pd.NaT
    pr2["is_draft_current"] = False; pr2["n_labels_current"] = 0
    pr2["title_current"] = "t"; pr2["base_ref_current"] = "main"
    pr2["additions_at_open"] = 1.0; pr2["deletions_at_open"] = 0.0; pr2["n_commits_at_open"] = 1.0
    pr2["diff_is_exact"] = True
    reviews = pd.DataFrame({
        "pr_id": ["r2", "r4", "s2"], "created_at": [t("2024-02-02"), t("2025-01-16"), t("2025-09-02")],
        "submitted_at": [t("2024-02-02"), t("2025-01-16"), t("2025-09-02")],
        "state": ["APPROVED"] * 3, "author_login": ["m"] * 3, "author_typename": ["User"] * 3,
        "author_association": ["MEMBER"] * 3,
    })
    empty = pd.DataFrame(columns=["pr_id", "created_at", "published_at", "author_login",
                                  "author_typename", "author_association", "is_minimized"])
    timeline = pd.DataFrame(columns=["pr_id", "event_type", "created_at", "previous_title",
                                     "previous_ref", "requested_reviewer_type"])
    repo_meta = pd.DataFrame({
        "repo": ["o/r", "o/s"], "created_at": [t("2022-01-01"), t("2024-01-01")],
        "n_assignable_users": [3, 1], "n_mentionable_users": [9, 2],
        "owner_type": ["Organization", "User"], "has_codeowners": [True, False],
        "has_pr_template": [True, False], "has_contributing": [False, False],
        "n_ci_workflows": [2, 0], "language_dominant": ["Go", "Go"], "default_branch": ["main", "main"],
    })
    return {"pr_tier1": tier1, "pr_tier2": pr2, "reviews": reviews, "thread_comments": empty,
            "issue_comments": empty, "timeline": timeline, "commits": pd.DataFrame(),
            "repo_meta": repo_meta}


def test_build_end_to_end(frames_toy):
    table, ctx = features.build(kept=["o/r", "o/s"], frames=frames_toy)
    # r1 pre-window and s1 bot are excluded; 6 modelling rows remain
    assert sorted(table["pr_id"]) == ["r2", "r3", "r4", "r5", "s2", "s3"]
    assert list(table.columns) == list(features.COLUMN_SPEC)
    row = table.set_index("pr_id")
    # author 'a' at r4 (2025-01-15): prior r1 (2023, merged) and r2 (merged 2024-02-10) -> 2 prior, 2 merged
    assert row.loc["r4", "n_prior_prs_here"] == 2 and row.loc["r4", "n_prior_merged_here"] == 2
    assert row.loc["r4", "is_first_pr_here"] == False                 # noqa: E712
    assert row.loc["r2", "is_first_pr_here"] == False                 # noqa: E712  r1 is prior even though pre-window
    assert row.loc["r3", "is_first_pr_here"] == True                  # noqa: E712  b's first
    assert row.loc["s3", "is_first_pr_here"] == True                  # noqa: E712  deleted author
    assert row.loc["r4", "open_backlog_at_t"] == 1                    # r3 open at 2025-01-15
    assert row.loc["r4", "repo_age_days_at_open"] == pytest.approx((ts("2025-01-15") - ts("2022-01-01")).days)
    # only documented-nullable columns may be NaN
    nullable = {c for c, m in features.COLUMN_SPEC.items() if m["nullable"]}
    for c in table.columns:
        if c not in nullable:
            assert table[c].notna().all(), c
    assert 0.0 <= ctx["g"] <= 1.0 and 0.0 <= ctx["g_merge"] <= 1.0


def test_write_table_roundtrip(frames_toy, tmp_path):
    table, _ = features.build(kept=["o/r", "o/s"], frames=frames_toy)
    p = tmp_path / "f.parquet"
    features.write_table(table, p)
    back = pd.read_parquet(p)
    assert list(back.columns) == list(table.columns) and len(back) == len(table)
    import pyarrow.parquet as pq
    meta = pq.read_metadata(p).metadata
    assert b"built_at" in meta and b"git_sha" in meta
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_features.py -q -k "build or roundtrip"`
Expected: FAIL — `AttributeError: module 'features' has no attribute 'build'`

- [ ] **Step 3: Implement (add to `features.py`)**

```python
# ---------------------------------------------------------------------------
# Group 3: replay (chronological, created_at < t; proven by the brute-force audit)
# ---------------------------------------------------------------------------

def replay_features(rows: pd.DataFrame, tier1: pd.DataFrame, label_d5: pd.DataFrame,
                    g: float, g_merge: float) -> pd.DataFrame:
    hist = {r: replay.History.from_frames(r, tier1, label_d5, splits.WINDOW_START)
            for r in rows["repo"].unique()}
    recs = []
    for repo, t, author in zip(rows["repo"], rows["created_at"], rows["author_login"]):
        a = None if (author is None or (isinstance(author, float) and np.isnan(author))) else author
        recs.append(hist[repo].features_at(t, g, g_merge, ALPHA, author=a))
    out = pd.DataFrame(recs, index=rows["pr_id"].to_numpy())
    out.index.name = "pr_id"
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def priors(rows: pd.DataFrame) -> tuple[float, float]:
    """Shrinkage priors from Scenario A TRAINING rows only (spec §6.3)."""
    train = rows[rows["created_at"] < splits.CUTOFF_A]
    if train.empty:
        train = rows
    g = float(train["is_slow"].mean())
    g_merge = float((train["merged_at"].notna() & (train["merged_at"] < splits.CUTOFF_A)).mean())
    return g, g_merge


def build(kept: list[str] | None = None, frames: dict | None = None) -> tuple[pd.DataFrame, dict]:
    kept = kept if kept is not None else cohort_qc.kept_repos()
    frames = frames if frames is not None else load.load_all(kept)

    prs = splits.modelling_prs(frames["pr_tier2"])
    prs = prs[prs["repo"].isin(kept)].reset_index(drop=True)
    streams = {k: frames[k] for k in labels.ALL_STREAMS}
    lab = labels.label(prs, labels.first_human_event(prs, streams, labels.DEFINITIONS[labels.PRIMARY]))
    rows = splits.prepare_rows(prs, lab, kept)
    g, g_merge = priors(rows)
    log.info("rows=%d repos=%d g=%.3f g_merge=%.3f", len(rows), rows["repo"].nunique(), g, g_merge)

    static = static_features(rows)
    at_open = at_open_features(rows, frames["timeline"], frames["repo_meta"])
    rep = replay_features(rows, frames["pr_tier1"], lab, g, g_merge)
    repo = repo_features(frames["repo_meta"], rows)

    lab_idx = lab.set_index("pr_id")[[c for c in LABEL_COLS if c != "is_slow"]]
    table = (rows.set_index("pr_id")[[k for k in KEYS if k != "pr_id"] + ["is_slow"]]
             .join(lab_idx).join(static).join(at_open).join(rep).join(repo)
             .reset_index())
    missing = [c for c in COLUMN_SPEC if c not in table.columns]
    extra = [c for c in table.columns if c not in COLUMN_SPEC]
    if missing or extra:
        raise RuntimeError(f"COLUMN_SPEC drift: missing={missing} extra={extra}")
    table = table[list(COLUMN_SPEC)]
    return table, {"rows": rows, "tier1": frames["pr_tier1"], "label_d5": lab, "g": g, "g_merge": g_merge}


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=ROOT, timeout=10).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def write_table(table: pd.DataFrame, path: Path = OUT) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    t = pa.Table.from_pandas(table, preserve_index=False)
    meta = dict(t.schema.metadata or {})
    meta.update({b"built_at": datetime.now(timezone.utc).isoformat().encode(),
                 b"git_sha": _git_sha().encode()})
    pq.write_table(t.replace_schema_metadata(meta), path)
```

Also add, at the bottom of `features.py`, a first version of `main()` (the audit/explain/dictionary subcommands are added in Tasks 5 and 6):

```python
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    table, _ = build()
    write_table(table, Path(args.out))
    print(f"wrote {len(table):,} rows x {len(table.columns)} cols to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_features.py -q`
Expected: `8 passed`

- [ ] **Step 5: Full suite, stage, commit**

Run: `python -m pytest tests -q` → all pass, no warnings.
```bash
git add features.py tests/test_features.py
git commit -m "feat(features): replay group, build(), parquet writer with provenance"
```

---

### Task 5: `audit` (Phase 3 gate 1–4) and `explain` (gate 5)

**Files:**
- Modify: `features.py` (add `audit`, `explain`, CLI flags `--audit`, `--explain PR_ID`)
- Modify: `tests/test_features.py`

**Interfaces:**
- Consumes: Task 4's `build` context; `replay.brute_force_features`, `replay.REPLAY_KEYS`.
- Produces:
  - `features.audit(table, ctx, n=500, seed=SEED, expected_rows=None) -> list[dict]` — four checks `{"id","check","value","pass"}`; writes nothing.
  - `features.explain(pr_id, table, ctx, frames) -> str`.
  - CLI: `python features.py --audit` writes `data/phase3_gate.json` and exits 1 on any failure; `python features.py --explain PR_ID`.

- [ ] **Step 1: Write the failing tests (append)**

```python
def test_audit_passes_on_toy(frames_toy):
    table, ctx = features.build(kept=["o/r", "o/s"], frames=frames_toy)
    checks = features.audit(table, ctx, n=6, expected_rows=6)
    assert [c["id"] for c in checks] == [1, 2, 3, 4]
    assert all(c["pass"] for c in checks), checks
    assert checks[1]["value"]["n"] == 6 and checks[1]["value"]["max_abs_diff"] < 1e-9


def test_audit_detects_replay_drift(frames_toy):
    table, ctx = features.build(kept=["o/r", "o/s"], frames=frames_toy)
    table = table.copy()
    table.loc[table["pr_id"] == "r4", "n_prior_prs_here"] += 1      # simulate a leak/drift
    checks = features.audit(table, ctx, n=6, expected_rows=6)
    assert checks[1]["pass"] is False and "LEAK" in checks[1]["check"]


def test_audit_row_count_and_nan(frames_toy):
    table, ctx = features.build(kept=["o/r", "o/s"], frames=frames_toy)
    assert features.audit(table, ctx, n=6, expected_rows=7)[0]["pass"] is False
    bad = table.copy(); bad.loc[0, "open_backlog_at_t"] = np.nan
    assert features.audit(bad, ctx, n=6, expected_rows=6)[2]["pass"] is False


def test_explain_mentions_contributing_prs(frames_toy):
    table, ctx = features.build(kept=["o/r", "o/s"], frames=frames_toy)
    txt = features.explain("r4", table, ctx, frames_toy)
    assert "r4" in txt and "n_prior_prs_here" in txt
    assert "r1" in txt and "r2" in txt           # author a's prior PRs are listed
    assert "r3" in txt                            # the open-backlog contributor
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_features.py -q -k "audit or explain"`
Expected: FAIL — `AttributeError: module 'features' has no attribute 'audit'`

- [ ] **Step 3: Implement (add to `features.py`)**

```python
# ---------------------------------------------------------------------------
# Phase 3 gate
# ---------------------------------------------------------------------------

NULLABLE = {c for c, m in COLUMN_SPEC.items() if m["nullable"]}
AUDIT_KEYS = tuple(k for k in replay.REPLAY_KEYS)   # every replay-derived column


def audit(table: pd.DataFrame, ctx: dict, n: int = 500, seed: int = SEED,
          expected_rows: int | None = None) -> list[dict]:
    rows, tier1, lab, g, g_merge = ctx["rows"], ctx["tier1"], ctx["label_d5"], ctx["g"], ctx["g_merge"]

    # 1. row conservation
    exp = expected_rows if expected_rows is not None else len(rows)
    c1 = {"id": 1, "check": "row count == modelling rows for kept repos", "value": [len(table), exp],
          "pass": len(table) == exp}

    # 2. brute-force audit of every replay column -- the leakage hard stop
    sample = table.sample(n=min(n, len(table)), random_state=seed)
    by_id = rows.set_index("pr_id")
    max_diff, worst = 0.0, None
    for _, r in sample.iterrows():
        src = by_id.loc[r["pr_id"]]
        a = src["author_login"]
        a = None if (a is None or (isinstance(a, float) and np.isnan(a))) else a
        b = replay.brute_force_features(tier1, lab, r["repo"], r["created_at"], g, ALPHA,
                                        global_merge_rate=g_merge, author=a)
        for k in AUDIT_KEYS:
            x, y = r[k], b[k]
            if isinstance(x, (bool, np.bool_)) or isinstance(y, bool):
                d = 0.0 if bool(x) == bool(y) else 1.0
            elif (isinstance(x, float) and np.isnan(x)) or (isinstance(y, float) and np.isnan(y)):
                d = 0.0 if (isinstance(x, float) and np.isnan(x) and isinstance(y, float) and np.isnan(y)) else 1.0
            else:
                d = abs(float(x) - float(y))
            if d > max_diff:
                max_diff, worst = d, (r["pr_id"], k, x, y)
    c2 = {"id": 2, "check": "replay audit: brute-force recomputation of every replay column matches (else replay LEAKS)",
          "value": {"n": int(len(sample)), "max_abs_diff": max_diff, "worst": worst}, "pass": max_diff < 1e-9}

    # 3. NaN only where documented
    bad = {c: int(table[c].isna().sum()) for c in table.columns if c not in NULLABLE and table[c].isna().any()}
    c3 = {"id": 3, "check": "no NaN outside documented-nullable columns", "value": bad, "pass": not bad}

    # 4. timeline truncation rate
    rate = float(table["timeline_may_be_truncated"].mean()) if len(table) else 0.0
    c4 = {"id": 4, "check": "timeline_may_be_truncated rate < 2%", "value": round(rate, 4), "pass": rate < 0.02}
    return [c1, c2, c3, c4]


def explain(pr_id: str, table: pd.DataFrame, ctx: dict, frames: dict) -> str:
    """Gate #5 support: print every feature of one PR with the rows/events behind it,
    so a human can check them against the GitHub UI."""
    rows, tier1, lab = ctx["rows"], ctx["tier1"], ctx["label_d5"]
    r = table.set_index("pr_id").loc[pr_id]
    src = rows.set_index("pr_id").loc[pr_id]
    t, repo, author = src["created_at"], src["repo"], src["author_login"]
    lines = [f"PR {pr_id}  {repo}#{int(r['number'])}  opened {t}  author={author!r}  is_slow={r['is_slow']}", ""]
    lines.append("== features ==")
    for c in feature_columns():
        lines.append(f"  {c:<32} {r[c]}")
    h = tier1[(tier1["repo"] == repo) & (tier1["created_at"] < t)].merge(
        lab[["pr_id", "first_event_at", "is_slow"]], on="pr_id", how="left")
    lines += ["", f"== open backlog at t ({int(r['open_backlog_at_t'])}) == prior PRs open at {t}:"]
    for _, x in h[h["closed_at"].isna() | (h["closed_at"] > t)].iterrows():
        lines.append(f"  {x['pr_id']}  created {x['created_at']}  closed {x['closed_at']}")
    lines += ["", f"== author history ({author!r}) == prior PRs by this author:"]
    for _, x in h[h["author_login"] == author].iterrows():
        lines.append(f"  {x['pr_id']}  created {x['created_at']}  merged {x['merged_at']}  "
                     f"first_event {x['first_event_at']}  is_slow {x['is_slow']}")
    lo = t - pd.Timedelta(days=90)
    lines += ["", f"== trailing 90d window [{lo} .. {t}) == labelled prior PRs:"]
    for _, x in h[(h["created_at"] >= lo) & h["is_slow"].notna()].iterrows():
        lines.append(f"  {x['pr_id']}  created {x['created_at']}  first_event {x['first_event_at']}  is_slow {x['is_slow']}")
    tl = frames["timeline"]
    ev = tl[tl["pr_id"] == pr_id].sort_values("created_at") if not tl.empty else tl
    lines += ["", f"== timeline events ({len(ev)}) =="]
    for _, e in ev.iterrows():
        extra = e.get("previous_title") or e.get("previous_ref") or e.get("requested_reviewer_type") or ""
        lines.append(f"  {e['created_at']}  {e['event_type']}  {extra}")
    return "\n".join(lines)
```

Replace `main()` with:

```python
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--audit", action="store_true", help="run the Phase 3 gate (1-4) on the built table")
    ap.add_argument("--explain", metavar="PR_ID", help="print one PR's features with their evidence")
    ap.add_argument("--n", type=int, default=500, help="audit sample size")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    kept = cohort_qc.kept_repos()
    frames = load.load_all(kept)
    table, ctx = build(kept, frames)

    if args.explain:
        print(explain(args.explain, table, ctx, frames))
        return 0
    if args.audit:
        checks = audit(table, ctx, n=args.n)
        GATE_JSON.parent.mkdir(parents=True, exist_ok=True)
        GATE_JSON.write_text(json.dumps(checks, indent=2, default=str), encoding="utf-8")
        for c in checks:
            print(f"  [{c['id']}] {'PASS' if c['pass'] else 'FAIL'}  {c['check']}  -> {c['value']}")
        ok = all(c["pass"] for c in checks)
        print(f"gate 1-4: {'PASS' if ok else 'FAIL'}  (gate 5 is the manual --explain spot-check)")
        return 0 if ok else 1

    write_table(table, Path(args.out))
    print(f"wrote {len(table):,} rows x {len(table.columns)} cols to {args.out}")
    return 0
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_features.py -q`
Expected: `12 passed`

- [ ] **Step 5: Stage and commit**

```bash
git add features.py tests/test_features.py
git commit -m "feat(features): Phase 3 gate audit and --explain spot-check"
```

---

### Task 6: Feature dictionary rendered from `COLUMN_SPEC`

**Files:**
- Modify: `features.py` (add `render_dictionary`, `--dictionary` flag)
- Modify: `tests/test_features.py`
- Generate: `docs/feature_dictionary.md`

**Interfaces:**
- Produces: `features.render_dictionary() -> str`; CLI `python features.py --dictionary` writes `docs/feature_dictionary.md`.

- [ ] **Step 1: Write the failing test (append)**

```python
def test_dictionary_lists_every_column_once():
    md = features.render_dictionary()
    for c in features.COLUMN_SPEC:
        assert md.count(f"| `{c}` |") == 1, c
    assert "never a feature" in md.lower()
    assert "| Column | Group | Status | Type | Nullable | Derivation |" in md
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_features.py -q -k dictionary`
Expected: FAIL — `AttributeError: module 'features' has no attribute 'render_dictionary'`

- [ ] **Step 3: Implement (add to `features.py`)**

```python
def render_dictionary() -> str:
    lines = [
        "# Feature Dictionary",
        "",
        "Generated by `python features.py --dictionary` from `features.COLUMN_SPEC`. Do not edit by hand.",
        "",
        "Written for: whoever trains on `data/features/features.parquet`.",
        "",
        "**Status** says how each column relates to the prediction instant `t = created_at`:",
        "",
        "| Status | Meaning |",
        "|---|---|",
        "| `key` | identifies the row; never a feature |",
        "| `label` | the target and its survival companions; never a feature |",
        "| `static` | known at open from the PR object; safe |",
        "| `reconstructed` | at-open value recovered from current state + timestamped events; an approximation, paired with a flag |",
        "| `replay` | computed by chronological replay over rows with `created_at < t`; proven by the brute-force audit (gate #2) |",
        "| `snapshot` | repo-level value as of collection (HEAD), applied to all of that repo's rows; accepted with that limitation |",
        "| `flag` | fidelity indicator for ablation and error analysis; **never a feature** (some encode post-open information) |",
        "",
        "Phase 4 trains only on `static`, `reconstructed`, `replay`, `snapshot` — see `features.feature_columns()`.",
        "",
        "Priors for shrinkage (`g` = D5 slow rate, `g_merge` = merge rate) are computed over Scenario A training rows",
        "(`created_at < 2026-01-01`) once at build time; a per-fold recomputation for Scenario B shifts an α=5 shrunk rate negligibly.",
        "",
        "| Column | Group | Status | Type | Nullable | Derivation |",
        "|---|---|---|---|---|---|",
    ]
    for c, m in COLUMN_SPEC.items():
        lines.append(f"| `{c}` | {m['group']} | {m['status']} | {m['dtype']} | "
                     f"{'yes' if m['nullable'] else 'no'} | {m['derivation']} |")
    return "\n".join(lines) + "\n"
```

Add to `main()` before the `kept = ...` line:

```python
    if args.dictionary:
        DICT_MD.parent.mkdir(parents=True, exist_ok=True)
        DICT_MD.write_text(render_dictionary(), encoding="utf-8")
        print(f"wrote {DICT_MD}")
        return 0
```
and the flag: `ap.add_argument("--dictionary", action="store_true", help="render docs/feature_dictionary.md from COLUMN_SPEC")`.

- [ ] **Step 4: Run, generate, verify**

Run: `python -m pytest tests -q` → all pass, no warnings.
Run: `python features.py --dictionary` → `wrote .../docs/feature_dictionary.md`. Open it and confirm 50 rows.

- [ ] **Step 5: Stage and commit**

```bash
git add features.py tests/test_features.py docs/feature_dictionary.md
git commit -m "feat(features): feature dictionary rendered from COLUMN_SPEC"
```

---

### Task 7: Real-data run — build, audit, spot-check (CONDITIONAL)

**Files:** none created in git. Outputs: `data/features/features.parquet`, `data/phase3_gate.json`.

**Precondition (Global Constraints):** `data/cohort/kept.json` exists AND every check in `data/phase2_gate.json` passes. Otherwise report `DEFERRED` and stop.

- [ ] **Step 1: Build**

Run: `python features.py`
Expected: `wrote N rows x 50 cols`, where N equals the Phase 2 modelling row count (compare with `len(splits.prepare_rows(...))` from the Phase 2 run, or with the `rows` figure printed in `docs/phase2_eda.md`).

- [ ] **Step 2: Audit (gates 1–4)**

Run: `python features.py --audit --n 500`
Expected: all four PASS. If gate 2 FAILS: **stop** — `data/phase3_gate.json` names the worst `(pr_id, key, features_at, brute_force)`; run `--explain` on it and report. Do not proceed to Phase 4.

- [ ] **Step 3: Gate 5 — manual spot-check**

Run:
```bash
python - <<'EOF'
import pandas as pd
t = pd.read_parquet("data/features/features.parquet")
print(t.sample(n=5, random_state=20260912)[["repo", "number", "pr_id"]].to_string(index=False))
EOF
```
Then for each printed `pr_id`: `python features.py --explain <pr_id>`. Report the five outputs verbatim in the task report. The controller hands them to the user, who confirms against `https://github.com/<repo>/pull/<number>` and records the verdict in `data/phase3_gate.json` as `{"id": 5, "check": "manual spot-check", "value": ["<pr_id>", ...], "pass": true|false}`.

---

## Verification (spec §9, §10)

1. `python -m pytest tests -q` — every group proven on toys with hand-computed answers, including `merged_at < t`, deleted author, parity draft flip, earliest-previous title, 60-second review-request grace, and the 60-row truncation flag.
2. `test_brute_force_matches_all_keys` — two implementations agree on every replay key across 3 instants × 5 authors.
3. Real data: gate 1–4 via `--audit`; gate 5 via `--explain` on 5 seeded rows, user-confirmed.
4. `docs/feature_dictionary.md` lists exactly the 50 columns of the built table; `feature_columns()` excludes every key, label, and flag.

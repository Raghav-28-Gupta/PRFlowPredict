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
    if "merged_at" in mine.columns:
        merged_col = mine["merged_at"]
        n_prior_merged = int((merged_col.notna() & (merged_col < t)).sum())
    else:
        n_prior_merged = 0
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

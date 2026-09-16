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
    """
    def ts(s): return pd.Timestamp(s, tz="UTC")
    tier1 = pd.DataFrame({
        "repo": ["r"] * 5,
        "pr_id": ["P0", "P1", "P2", "P3", "P4"],
        "created_at": [ts("2023-06-01"), ts("2024-03-01"), ts("2024-03-10"),
                       ts("2024-04-15"), ts("2024-04-19")],
        "closed_at": [ts("2024-06-01"), pd.NaT, ts("2024-03-20"), pd.NaT, pd.NaT],
        "merged_at": [ts("2024-06-01"), pd.NaT, ts("2024-03-20"), pd.NaT, pd.NaT],
        "author_login": ["u0", "u1", "u2", "u1", "u3"],
    })
    labels = pd.DataFrame({
        "pr_id": ["P1", "P2", "P3", "P4"],
        "first_event_at": [ts("2024-03-02"), pd.NaT, ts("2024-04-16"), pd.NaT],
        "is_slow": [False, True, False, True],
    })
    return tier1, labels

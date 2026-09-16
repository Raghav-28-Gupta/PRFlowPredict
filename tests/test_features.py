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


def test_repo_features_names_missing_repos(prs_toy, repo_meta_toy):
    with pytest.raises(KeyError, match="o/s"):
        features.repo_features(repo_meta_toy[repo_meta_toy["repo"] == "o/r"], prs_toy)


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


def test_earliest_title_event_wins_even_when_its_previous_is_null():
    t0 = ts("2025-01-01T00:00")
    prs = pd.DataFrame({"repo": ["o/r"], "pr_id": ["X"], "created_at": [t0],
                        "is_draft_current": [False], "n_labels_current": [0],
                        "title_current": ["current-title"], "base_ref_current": ["main"],
                        "additions_at_open": [1.0], "deletions_at_open": [0.0],
                        "n_commits_at_open": [1.0], "diff_is_exact": [True]})
    timeline = pd.DataFrame([
        ("X", "RenamedTitleEvent", t0 + pd.Timedelta(hours=1), None, None, None),      # earliest: null previous
        ("X", "RenamedTitleEvent", t0 + pd.Timedelta(hours=2), "later-prev", None, None),
    ], columns=["pr_id", "event_type", "created_at", "previous_title", "previous_ref", "requested_reviewer_type"])
    repo_meta = pd.DataFrame({"repo": ["o/r"], "default_branch": ["main"]})
    a = features.at_open_features(prs, timeline, repo_meta)
    # the earliest event's previous_title is null -> fall back to title_current (13), NOT 'later-prev' (10)
    assert a.loc["X", "title_len_at_open"] == len("current-title")


def test_column_spec_is_the_contract():
    spec = features.COLUMN_SPEC
    assert len(spec) == 50
    assert set(features.KEYS) <= set(spec) and set(features.LABEL_COLS) <= set(spec)
    for col, meta in spec.items():
        assert set(meta) == {"group", "status", "dtype", "nullable", "derivation"}, col
        assert meta["status"] in {"key", "label", "static", "reconstructed", "replay", "snapshot", "flag"}, col
    trainable = features.feature_columns()
    assert "is_slow" not in trainable and "pr_id" not in trainable
    assert "diff_is_exact" not in trainable and "body_edited" not in trainable
    assert "trailing_90d_slow_rate" in trainable and "author_account_age_days" in trainable


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
    pr2["author_is_bot"] = [False, False, False, False, True, False, False]   # s1 is a bot
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

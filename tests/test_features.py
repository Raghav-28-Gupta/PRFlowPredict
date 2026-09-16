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

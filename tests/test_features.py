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

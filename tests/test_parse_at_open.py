"""Regression for the zero-collapsed-to-None bug in parse.tier2_rows."""
import parse


def _node(commits, created="2025-01-01T00:00:00Z"):
    return {
        "id": "PR_x", "createdAt": created, "author": {"login": "a", "__typename": "User"},
        "commits": {"totalCount": len(commits), "nodes": [{"commit": c} for c in commits]},
        "reviews": {"totalCount": 0, "nodes": []}, "comments": {"totalCount": 0, "nodes": []},
        "reviewThreads": {"totalCount": 0, "nodes": []}, "timelineItems": {"totalCount": 0, "nodes": []},
        "labels": {"totalCount": 0, "nodes": []},
    }


def test_pure_deletion_pr_has_zero_additions_not_none():
    stats = parse.Stats()
    built = parse.tier2_rows(_node([{"authoredDate": "2024-12-31T00:00:00Z", "additions": 0, "deletions": 7}]), "o/r", stats)
    pr = built["pr"]
    assert pr["additions_at_open"] == 0 and pr["deletions_at_open"] == 7 and pr["n_commits_at_open"] == 1
    assert pr["diff_is_exact"] is True


def test_no_commit_before_open_is_unknown_not_exact():
    stats = parse.Stats()
    built = parse.tier2_rows(_node([{"authoredDate": "2025-01-02T00:00:00Z", "additions": 5, "deletions": 0}]), "o/r", stats)
    pr = built["pr"]
    assert pr["additions_at_open"] is None and pr["n_commits_at_open"] is None
    assert pr["diff_is_exact"] is False

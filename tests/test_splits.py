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

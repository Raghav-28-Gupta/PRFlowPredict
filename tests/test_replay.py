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
    fa, fb = a.features_at(t, 0.5), b.features_at(t, 0.5)
    assert fa.keys() == fb.keys()
    for k in fa:
        va, vb = fa[k], fb[k]
        if isinstance(va, float) and np.isnan(va):
            assert isinstance(vb, float) and np.isnan(vb)
        else:
            assert va == vb


def test_brute_force_matches_features_at_on_toy(replay_toy):
    tier1, labels = replay_toy
    hist = replay.History.from_frames("r", tier1, labels, WS)
    for t in [pd.Timestamp("2024-04-21", tz="UTC"), pd.Timestamp("2024-02-01", tz="UTC")]:
        a = hist.features_at(t, global_rate=0.5)
        b = replay.brute_force_features(tier1, labels, "r", t, global_rate=0.5)
        assert a["open_backlog_at_t"] == b["open_backlog_at_t"]
        assert a["trailing_n"] == b["trailing_n"]
        assert a["trailing_90d_slow_rate"] == pytest.approx(b["trailing_90d_slow_rate"])


def test_trailing_window_excludes_old_labelled_rows():
    def ts(s): return pd.Timestamp(s, tz="UTC")
    tier1 = pd.DataFrame({"repo": ["r"] * 2, "pr_id": ["OLD", "NEW"],
                          "created_at": [ts("2024-01-01"), ts("2024-05-01")],
                          "closed_at": [pd.NaT, pd.NaT], "author_login": ["a", "b"]})
    labels = pd.DataFrame({"pr_id": ["OLD", "NEW"],
                           "first_event_at": [ts("2024-01-02"), ts("2024-05-02")],
                           "is_slow": [True, False]})
    hist = replay.History.from_frames("r", tier1, labels, WS)
    f = hist.features_at(ts("2024-06-01"), global_rate=0.5)  # OLD is 152 days old
    assert f["trailing_n"] == 1                               # outside the 90-day window
    assert f["open_backlog_at_t"] == 2                        # but still in the backlog


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


def test_author_rate_uses_whole_history_not_90d_window():
    def ts(s): return pd.Timestamp(s, tz="UTC")
    # OLD: 152 days before t, labelled slow, resolvable; NEW: 30 days before t, labelled fast.
    # GHOST: unlabelled, author_login None -- must not match author "a" and must not raise.
    # All-NaT columns must stay tz-aware, or brute_force_features' raw column comparison
    # against tz-aware `t` raises TypeError (pandas infers tz-naive for an all-NaT list).
    nat3 = pd.Series([pd.NaT] * 3, dtype="datetime64[ns, UTC]")
    tier1 = pd.DataFrame({"repo": ["r"] * 3, "pr_id": ["OLD", "NEW", "GHOST"],
                          "created_at": [ts("2024-01-01"), ts("2024-05-01"), ts("2024-03-01")],
                          "closed_at": nat3, "merged_at": nat3,
                          "author_login": ["a", "a", None]})
    labels = pd.DataFrame({"pr_id": ["OLD", "NEW"],
                           "first_event_at": [pd.NaT, ts("2024-05-02")],
                           "is_slow": [True, False]})
    hist = replay.History.from_frames("r", tier1, labels, WS)
    t = ts("2024-06-01")
    f = hist.features_at(t, 0.5, 0.4, author="a")
    assert f["trailing_n"] == 1                                   # repo rate: only NEW is in the window
    assert f["author_prior_n"] == 2                               # author rate: OLD counts too
    assert f["author_prior_slow_rate_here"] == pytest.approx((1 + 5 * 0.5) / (2 + 5))
    assert f["n_prior_prs_here"] == 2                             # GHOST (None) not matched, no crash
    assert f["open_backlog_at_t"] == 3
    b = replay.brute_force_features(tier1, labels, "r", t, 0.5, global_merge_rate=0.4, author="a")
    assert b["author_prior_n"] == 2 and b["author_prior_slow_rate_here"] == pytest.approx(f["author_prior_slow_rate_here"])

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

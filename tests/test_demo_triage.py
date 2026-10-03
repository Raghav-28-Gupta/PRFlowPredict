"""The demo's triage logic on small synthetic frames (Phase 7 spec, section 5)."""
from datetime import date

import pandas as pd
import pytest

from demo import triage

T = pd.Timestamp("2026-03-10", tz="UTC")
H = pd.Timedelta(hours=1)


def _prs(*rows, repo="o/r"):
    """Each row: (number, created, closed, first_review, score_a, is_slow); times are offsets
    from T, and None means never closed / never reviewed."""
    recs = []
    for number, created, closed, review, score, slow in rows:
        recs.append({
            "repo": repo, "pr_id": f"{repo}#{number}", "number": number,
            "url": f"https://github.com/{repo}/pull/{number}", "title": f"PR {number}",
            "created_at": T + created,
            "closed_at": T + closed if closed is not None else pd.NaT,
            "first_review_at": T + review if review is not None else pd.NaT,
            "is_slow": slow, "score_a": score, "score_b": 1 - score,
            "drivers_a": f"a{number}", "drivers_b": f"b{number}",
        })
    df = pd.DataFrame(recs)
    for col in ("created_at", "closed_at", "first_review_at"):    # tz-aware, as in the extract
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def _numbers(df, scenario="A"):
    return [int(u.rsplit("/", 1)[1]) for u in triage.ranked(df, "o/r", T, scenario)["url"]]


def test_the_moment_is_midnight_utc():
    assert triage.moment(date(2026, 3, 10)) == T


def test_a_pr_opened_after_or_exactly_at_the_moment_is_not_waiting_yet():
    df = _prs((1, H, None, None, 0.5, False), (2, 0 * H, None, None, 0.5, False))
    assert _numbers(df) == []


def test_a_pr_opened_a_minute_before_midnight_is_waiting():
    df = _prs((1, -pd.Timedelta(minutes=1), None, None, 0.5, False))
    assert _numbers(df) == [1]


def test_a_pr_closed_before_or_exactly_at_the_moment_drops_out():
    df = _prs((1, -9 * H, -H, None, 0.9, True), (2, -9 * H, 0 * H, None, 0.8, True),
              (3, -9 * H, H, None, 0.7, True))
    assert _numbers(df) == [3]


def test_a_pr_reviewed_before_or_exactly_at_the_moment_drops_out():
    df = _prs((1, -9 * H, None, -H, 0.9, False), (2, -9 * H, None, 0 * H, 0.8, False),
              (3, -9 * H, None, H, 0.7, False))
    assert _numbers(df) == [3]


def test_a_pr_never_reviewed_and_never_closed_stays_waiting():
    df = _prs((1, -900 * H, None, None, 0.5, True))
    assert _numbers(df) == [1]


def test_ranking_is_by_risk_then_age_then_number():
    df = _prs((5, -2 * H, None, None, 0.4, False), (4, -9 * H, None, None, 0.4, False),
              (3, -9 * H, None, None, 0.4, False), (9, -H, None, None, 0.8, False))
    assert _numbers(df) == [9, 3, 4, 5]


def test_the_unseen_model_ranks_by_its_own_score_and_drivers():
    df = _prs((1, -H, None, None, 0.9, False), (2, -H, None, None, 0.1, False))
    listing = triage.ranked(df, "o/r", T, "B")
    assert _numbers(df, "B") == [2, 1]
    assert listing["risk"].tolist() == pytest.approx([0.9, 0.1])
    assert listing["drivers"].tolist() == ["b2", "b1"]


def test_days_waited_and_rank():
    listing = triage.ranked(_prs((1, -36 * H, None, None, 0.5, False)), "o/r", T, "A")
    assert listing["days_waited"].tolist() == [1.5] and listing["rank"].tolist() == [1]


def test_outcome_says_when_the_review_came_and_whether_it_stalled():
    df = _prs((1, -H, None, 11 * H, 0.9, False), (2, -H, None, 239 * H, 0.8, True),
              (3, -H, None, None, 0.7, True))
    assert triage.ranked(df, "o/r", T, "A")["outcome"].tolist() == [
        "reviewed after 0.5 days", "reviewed after 10.0 days, stalled", "never reviewed, stalled"]


def test_outcome_says_when_a_pr_was_closed_without_a_review():
    """Most never-reviewed PRs were closed within days; 'never reviewed' alone would hide that."""
    df = _prs((1, -H, 64 * H, None, 0.9, True))
    assert triage.ranked(df, "o/r", T, "A")["outcome"].tolist() == [
        "closed after 2.7 days without a review, stalled"]


def test_tally_counts_the_stalled_among_the_top_three():
    df = _prs(*[(n, -H, None, None, 1 - n / 10, n % 2 == 1) for n in range(1, 6)])
    assert triage.tally(triage.ranked(df, "o/r", T, "A")) == (2, 3)


def test_tally_on_a_short_list_and_an_empty_one():
    one = triage.ranked(_prs((1, -H, None, None, 0.5, True)), "o/r", T, "A")
    none = triage.ranked(_prs((1, H, None, None, 0.5, True)), "o/r", T, "A")
    assert triage.tally(one) == (1, 1) and triage.tally(none) == (0, 0)


def test_default_repo_has_the_most_waiting_with_ties_alphabetical():
    waiting = (1, -H, None, None, 0.5, False)
    df = pd.concat([_prs(waiting, repo="z/z"), _prs(waiting, (2, -H, None, None, 0.5, False), repo="m/m"),
                    _prs(waiting, (2, -H, None, None, 0.5, False), repo="b/b")], ignore_index=True)
    assert triage.default_repo(df, T) == "b/b"


def test_default_repo_when_nobody_is_waiting_is_the_first_alphabetically():
    df = pd.concat([_prs((1, H, None, None, 0.5, False), repo="z/z"),
                    _prs((1, H, None, None, 0.5, False), repo="c/c")], ignore_index=True)
    assert triage.default_repo(df, T) == "c/c"


def test_load_works_from_any_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert len(triage.load()) > 0

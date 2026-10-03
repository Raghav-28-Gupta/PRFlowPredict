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


# ---------------------------------------------------------------------------
# looking up one PR
# ---------------------------------------------------------------------------

def test_parse_accepts_a_pr_link_with_or_without_extra_parts():
    for text in ("https://github.com/o/r/pull/12", "https://github.com/o/r/pull/12/files",
                 "github.com/o/r/pull/12?x=1", "  https://github.com/o/r/pull/12  "):
        assert triage.parse_pr_ref(text, "d/d") == ("o/r", 12), text


def test_parse_accepts_owner_repo_hash_number():
    assert triage.parse_pr_ref("o/r#12", "d/d") == ("o/r", 12)


def test_parse_resolves_a_bare_number_in_the_default_repo():
    assert triage.parse_pr_ref("12", "d/d") == ("d/d", 12)
    assert triage.parse_pr_ref(" #12 ", "d/d") == ("d/d", 12)


def test_parse_rejects_anything_else():
    for text in ("", "   ", "hello", "o/r", "https://github.com/o/r/issues/12", "#"):
        assert triage.parse_pr_ref(text, "d/d") is None, text


def test_find_pr_matches_the_repo_case_insensitively_and_misses_cleanly():
    df = _prs((1, -H, None, None, 0.5, False))
    assert triage.find_pr(df, "O/R", 1)["number"].tolist() == [1]
    assert triage.find_pr(df, "o/r", 99).empty and triage.find_pr(df, "x/y", 1).empty


def test_rank_in_repo_is_the_share_of_the_repos_other_prs_scored_lower():
    df = pd.concat([_prs((1, -H, None, None, 0.9, False), (2, -H, None, None, 0.5, False),
                         (3, -H, None, None, 0.5, False), (4, -H, None, None, 0.1, False)),
                    _prs((1, -H, None, None, 0.0, False), repo="x/y")], ignore_index=True)
    rank = lambda n, sc="A": triage.rank_in_repo(df, triage.find_pr(df, "o/r", n), sc)
    assert rank(1) == (1.0, 3)                       # the riskiest: higher than all three others
    assert rank(4) == (0.0, 3)                       # the least risky
    assert rank(2) == (pytest.approx(1 / 3), 3)      # a tie does not count as lower
    assert rank(4, "B") == (1.0, 3)                  # the unseen model's own score (1 - score_a here)


def test_random_pr_is_one_replayed_row_and_reproducible_with_a_seed():
    df = _prs(*[(n, -H, None, None, n / 10, False) for n in range(1, 6)])
    a, b = triage.random_pr(df, seed=3), triage.random_pr(df, seed=3)
    assert len(a) == 1 and a["pr_id"].tolist() == b["pr_id"].tolist()


def test_md_escape_keeps_a_title_from_turning_into_markdown():
    assert triage.md_escape("fix *bold* [x](y) `code` #1") == r"fix \*bold\* \[x\]\(y\) \`code\` \#1"

import pandas as pd
import labels
from tests.conftest import h


def fe(prs, streams, name):
    return labels.first_human_event(prs, streams, labels.DEFINITIONS[name])


def test_is_bot_three_ways():
    login = pd.Series(["dependabot[bot]", "Dependabot", "alice", None, "renovate"])
    typename = pd.Series(["Bot", "User", "User", None, "User"])
    assert labels.is_bot(login, typename).tolist() == [True, True, False, False, True]


def test_d3_counts_spam_comment_d5_does_not(label_toy):
    prs, streams = label_toy
    d3, d5 = fe(prs, streams, "D3"), fe(prs, streams, "D5")
    assert d3["B"] == h(1)      # spammer's NONE comment counts under the blueprint rule
    assert d5["B"] == h(200)    # excluded under D5; bob's review is first


def test_pending_review_and_self_and_bot_never_count(label_toy):
    prs, streams = label_toy
    d3 = fe(prs, streams, "D3")
    assert "C" not in d3.index  # only alice (self) and dependabot[bot]
    assert "D" not in d3.index  # PENDING review has null submitted_at
    assert d3["A"] == h(2)      # submitted_at, not created_at (+1h)


def test_d1_reviews_only(label_toy):
    prs, streams = label_toy
    d1 = fe(prs, streams, "D1")
    assert set(d1.index) == {"A", "B"}
    assert d1["B"] == h(200)


def test_label_columns_and_rule(label_toy):
    prs, streams = label_toy
    out = labels.label(prs, fe(prs, streams, "D5")).set_index("pr_id")
    assert out.loc["A", "is_slow"] == False and out.loc["A", "wait_h"] == 2
    assert out.loc["B", "is_slow"] == True and out.loc["B", "wait_h"] == 200
    assert out.loc["C", "never_reviewed_30d"] == True and out.loc["C", "is_slow"] == True
    assert pd.isna(out.loc["C", "wait_h"])
    assert out.loc["C", "wait_h_censored"] == 720 and out.loc["C", "event_observed"] == False
    assert out.loc["A", "event_observed"] == True
    assert set(out.columns) >= {"repo", "created_at", "first_event_at", "wait_h",
                                "never_reviewed_30d", "is_slow", "wait_h_censored",
                                "event_observed"}


def test_wait_beyond_censor_is_never_reviewed(label_toy):
    prs, streams = label_toy
    fe_ = pd.Series({"A": h(800)})   # reviewed, but after the 30-day window
    out = labels.label(prs, fe_).set_index("pr_id")
    assert out.loc["A", "never_reviewed_30d"] == True
    assert out.loc["A", "wait_h_censored"] == 720


def test_d4_excludes_minimized_but_d3_counts_it(label_toy):
    prs, streams = label_toy
    extra = pd.DataFrame({
        "pr_id": ["D"], "created_at": [h(3)], "published_at": [h(3)],
        "author_login": ["carol"], "author_typename": ["User"],
        "author_association": ["MEMBER"], "is_minimized": [True],
    })
    streams = {**streams,
               "issue_comments": pd.concat([streams["issue_comments"], extra], ignore_index=True)}
    d3 = fe(prs, streams, "D3")
    d4 = fe(prs, streams, "D4")
    assert d3["D"] == h(3)        # a minimized comment still counts under the blueprint rule
    assert "D" not in d4.index    # D4 drops it, and D has no other eligible event


def test_label_all_has_all_definitions(label_toy):
    prs, streams = label_toy
    out = labels.label_all(prs, streams)
    assert set(out) == {"D1", "D2", "D3", "D4", "D5"}
    assert labels.PRIMARY in out

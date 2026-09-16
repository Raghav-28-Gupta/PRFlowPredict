import pandas as pd
import cohort_qc
from tests.conftest import BASE

WINDOW = (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-06-30 23:59:59", tz="UTC"))


def frames(n_human, n_bot, language):
    n = n_human + n_bot
    prs = pd.DataFrame({
        "repo": ["o/r"] * n, "pr_id": [f"p{i}" for i in range(n)],
        "created_at": [BASE] * n, "author_login": ["u"] * n,
        "author_is_bot": [False] * n_human + [True] * n_bot,
    })
    meta = pd.DataFrame({"repo": ["o/r"], "language_dominant": [language]})
    empty = pd.DataFrame(columns=["pr_id", "created_at", "published_at", "submitted_at",
                                  "author_login", "author_typename",
                                  "author_association", "is_minimized"])
    return {"pr_tier2": prs, "repo_meta": meta, "reviews": empty,
            "thread_comments": empty, "issue_comments": empty}


def entry(stratum="Python"):
    return {"repo": "o/r", "cell": f"{stratum}:200-800", "language_stratum": stratum}


def test_clean_repo_is_kept():
    r = cohort_qc.check_repo(entry(), frames(150, 10, "Python"), WINDOW)
    assert r["kept"] and r["reasons"] == []
    assert r["n_human"] == 150 and abs(r["bot_share"] - 10 / 160) < 1e-9


def test_bot_dominated_is_dropped():
    r = cohort_qc.check_repo(entry(), frames(100, 120, "Python"), WINDOW)
    assert not r["kept"] and any("bot" in x for x in r["reasons"])


def test_too_small_is_dropped():
    r = cohort_qc.check_repo(entry(), frames(99, 0, "Python"), WINDOW)
    assert not r["kept"] and any("100" in x for x in r["reasons"])


def test_language_mismatch_is_dropped_and_js_is_fine_for_ts():
    bad = cohort_qc.check_repo(entry("Python"), frames(150, 0, "Shell"), WINDOW)
    assert not bad["kept"] and any("language" in x for x in bad["reasons"])
    ok = cohort_qc.check_repo(entry("TypeScript"), frames(150, 0, "JavaScript"), WINDOW)
    assert ok["kept"]


def test_label_rate_never_a_drop_reason():
    # 150 human PRs, none ever reviewed -> 100% never-reviewed. Must still be kept.
    r = cohort_qc.check_repo(entry(), frames(150, 0, "Python"), WINDOW)
    assert r["kept"] and r["is_slow_d5"] == 1.0


def test_null_author_is_bot_is_treated_as_human():
    f = frames(150, 10, "Python")
    prs = f["pr_tier2"]
    prs["author_is_bot"] = prs["author_is_bot"].astype(object)
    prs.loc[prs.index[0], "author_is_bot"] = None   # a human row, now null (object dtype)
    r = cohort_qc.check_repo(entry(), f, WINDOW)     # must not raise TypeError
    assert r["kept"]
    assert r["n_human"] == 150  # null counted as human (not True), same as before the edit


def test_no_data_collected_is_dropped_without_raising():
    f = frames(150, 10, "Python")
    f["pr_tier2"] = pd.DataFrame()
    r = cohort_qc.check_repo(entry(), f, WINDOW)
    assert r["kept"] is False
    assert r["reasons"] == ["no data collected"]

import numpy as np
import pandas as pd
import baseline
import splits


def test_baseline_end_to_end_on_toy():
    def ts(s): return pd.Timestamp(s, tz="UTC")
    n = 300
    rng = np.random.default_rng(0)
    created = pd.date_range("2024-01-15", "2026-06-15", periods=n, tz="UTC")
    repo = np.where(np.arange(n) % 2 == 0, "r1", "r2")
    # r1 is slow 80% of the time, r2 20% -- the baseline should separate REPOS
    is_slow = rng.random(n) < np.where(repo == "r1", 0.8, 0.2)
    prs = pd.DataFrame({"repo": repo, "pr_id": [f"p{i}" for i in range(n)],
                        "created_at": created, "closed_at": [pd.NaT] * n,
                        "author_is_bot": False, "author_login": "u"})
    fe = pd.Series(created + pd.Timedelta(hours=2))
    fe[is_slow] = pd.NaT
    lab = pd.DataFrame({"pr_id": prs["pr_id"], "is_slow": is_slow,
                        "first_event_at": fe,
                        "wait_h": np.where(is_slow, np.nan, 2.0)})
    rows = splits.prepare_rows(prs, lab, ["r1", "r2"])
    tier1 = prs[["repo", "pr_id", "created_at", "closed_at", "author_login"]]

    tr, te = splits.scenario_a(rows, cap_frac=1.0)
    res = baseline.evaluate("A", rows, [(tr, te)], tier1, lab)[0]

    # within-repo the score is ~constant, so P@10 is a random draw ~ base rate
    assert res["p10_ci_lo"] - 0.15 <= res["base_rate_p10"] <= res["p10_ci_hi"] + 0.15
    # but ACROSS repos the score separates r1 from r2, so AUC-PR > base rate
    assert res["auc_pr"] > res["base_rate"]
    assert res["n_test_repos"] == 2

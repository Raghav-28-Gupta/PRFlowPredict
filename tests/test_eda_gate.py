import eda_report


def base_rows(p10=0.50, lo=0.45, hi=0.55, base=0.50):
    return [{"scenario": "A", "precision_at_10": p10, "p10_ci_lo": lo, "p10_ci_hi": hi,
             "base_rate_p10": base},
            {"scenario": "B", "precision_at_10": p10, "p10_ci_lo": lo, "p10_ci_hi": hi,
             "base_rate_p10": base}]


def test_all_pass():
    checks = eda_report.gate_checks(kept_n=40, d5_rate=0.45, a_repos_with_10=25,
                                    b_min_fold_repos=8, baseline_rows=base_rows())
    assert all(c["pass"] for c in checks) and len(checks) == 5


def test_each_failure_is_named():
    fails = eda_report.gate_checks(kept_n=29, d5_rate=0.75, a_repos_with_10=19,
                                   b_min_fold_repos=5,
                                   baseline_rows=base_rows(p10=0.90, lo=0.85, hi=0.95, base=0.50))
    assert [c["pass"] for c in fails] == [False] * 5


def test_baseline_beating_base_rate_fails_sanity():
    checks = eda_report.gate_checks(kept_n=40, d5_rate=0.45, a_repos_with_10=25,
                                    b_min_fold_repos=8,
                                    baseline_rows=base_rows(p10=0.80, lo=0.75, hi=0.85, base=0.50))
    assert checks[4]["pass"] is False and "leak" in checks[4]["check"].lower()

import eda_report


def test_all_pass():
    audits = [{"pass": True, "n": 200, "max_abs_diff_rate": 0.0}] * 2
    checks = eda_report.gate_checks(kept_n=40, d5_rate=0.45, a_repos_with_10=25,
                                    b_min_fold_repos=8, audits=audits)
    assert all(c["pass"] for c in checks) and len(checks) == 5


def test_each_failure_is_named():
    audits = [{"pass": False, "n": 200, "max_abs_diff_rate": 0.3}]
    fails = eda_report.gate_checks(kept_n=29, d5_rate=0.75, a_repos_with_10=19,
                                   b_min_fold_repos=5, audits=audits)
    assert [c["pass"] for c in fails] == [False] * 5


def test_failed_audit_fails_check_5():
    audits = [{"pass": False, "n": 200, "max_abs_diff_rate": 0.3}]
    checks = eda_report.gate_checks(kept_n=40, d5_rate=0.45, a_repos_with_10=25,
                                    b_min_fold_repos=8, audits=audits)
    assert checks[4]["pass"] is False and "leak" in checks[4]["check"].lower()


def test_empty_cohort_fails_every_check():
    checks = eda_report.gate_checks(kept_n=0, d5_rate=float("nan"), a_repos_with_10=0,
                                    b_min_fold_repos=0, audits=[])
    assert [c["pass"] for c in checks] == [False] * 5

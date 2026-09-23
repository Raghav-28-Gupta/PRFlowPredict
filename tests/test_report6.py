import numpy as np
import pandas as pd

import report6


def _ok_inputs():
    additivity = {"A": [1e-9, 2e-9], "B": [3e-10]}
    stability = 0.97
    worst = pd.DataFrame({"kind": ["fp"] * 25 + ["fn"] * 25,
                          "pr_id": [f"p{i}" for i in range(50)],
                          "is_slow": [False] * 25 + [True] * 25})
    pred_a = pd.DataFrame({"pr_id": [f"p{i}" for i in range(100)],
                           "is_slow": [False] * 25 + [True] * 25 + [False] * 50})
    fair = pd.DataFrame({"gap_ci_lo": [0.1] * 6, "gap_ci_hi": [0.3] * 6, "n_repos": [8] * 6})
    artifacts = {"shift": 37, "importance_A": 37, "importance_B": 37, "worst50": 50, "fairness": 6}
    return additivity, stability, worst, pred_a, fair, artifacts


def _failed(checks):
    """ids of the checks that failed -- lets a test assert 'only check N failed'."""
    return [c["id"] for c in checks if not c["pass"]]


def test_gate_all_pass():
    checks = report6.gate_checks(*_ok_inputs())
    assert [c["id"] for c in checks] == [1, 2, 3, 4, 5]
    assert all(c["pass"] for c in checks), [c for c in checks if not c["pass"]]


def test_gate1_fails_on_broken_additivity():
    a, s, w, p, f, art = _ok_inputs()
    a["B"] = [1e-3]
    c = report6.gate_checks(a, s, w, p, f, art)
    assert c[0]["pass"] is False and "additivity" in c[0]["check"].lower()
    assert _failed(c) == [1]                            # only additivity broke


def test_gate2_fails_on_unstable_ranking():
    a, s, w, p, f, art = _ok_inputs()
    c = report6.gate_checks(a, 0.4, w, p, f, art)
    assert c[1]["pass"] is False
    assert _failed(c) == [2]                            # only stability broke


def test_gate3_fails_on_wrong_split_or_foreign_rows():
    a, s, w, p, f, art = _ok_inputs()
    lopsided = w.copy(); lopsided.loc[0, "kind"] = "fn"          # 24/26
    c = report6.gate_checks(a, s, lopsided, p, f, art)
    assert c[2]["pass"] is False
    assert _failed(c) == [3]
    foreign = w.copy(); foreign.loc[0, "pr_id"] = "NOT_IN_PRED"
    c = report6.gate_checks(a, s, foreign, p, f, art)
    assert c[2]["pass"] is False
    assert _failed(c) == [3]
    mismatched = w.copy(); mismatched.loc[0, "is_slow"] = True   # p0 is False in pred_a
    c = report6.gate_checks(a, s, mismatched, p, f, art)
    assert c[2]["pass"] is False
    assert _failed(c) == [3]


def test_gate4_fails_on_non_finite_ci_or_single_repo():
    a, s, w, p, f, art = _ok_inputs()
    bad_ci = f.copy(); bad_ci.loc[0, "gap_ci_hi"] = np.nan
    c = report6.gate_checks(a, s, w, p, bad_ci, art)
    assert c[3]["pass"] is False
    assert _failed(c) == [4]
    one_repo = f.copy(); one_repo.loc[0, "n_repos"] = 1
    c = report6.gate_checks(a, s, w, p, one_repo, art)
    assert c[3]["pass"] is False
    assert _failed(c) == [4]


def test_gate5_fails_on_empty_artifact():
    a, s, w, p, f, art = _ok_inputs()
    art["shift"] = 0
    c = report6.gate_checks(a, s, w, p, f, art)
    assert c[4]["pass"] is False
    assert _failed(c) == [5]                            # only the artifact check broke


def test_md_renders_a_table():
    out = report6.md(pd.DataFrame({"a": [1.0], "b": ["x"]}))
    assert out.splitlines()[0] == "| a | b |"
    assert "1.000" in out and "x" in out


def test_fmt_wait_h_renders_never_reviewed_for_nan():
    df = pd.DataFrame({"pr_id": ["p1", "p2"], "wait_h": [float("nan"), 12.5]})
    out = report6._fmt_wait_h(df)
    assert out.loc[0, "wait_h"] == "never reviewed"
    assert out.loc[1, "wait_h"] == 12.5
    assert pd.isna(df.loc[0, "wait_h"])                 # input frame is not mutated


def test_md_renders_never_reviewed_not_literal_nan():
    df = report6._fmt_wait_h(pd.DataFrame({"wait_h": [float("nan"), 5.0]}))
    out = report6.md(df)
    assert "never reviewed" in out
    assert "nan" not in out.lower()


def test_fairness_heading_names_the_scenario():
    a, b = report6._fairness_heading("A"), report6._fairness_heading("B")
    assert "Scenario A" in a and "within-project" in a
    assert "Scenario B" in b and "cold-start" in b
    assert a != b                                       # a reader must be able to tell them apart

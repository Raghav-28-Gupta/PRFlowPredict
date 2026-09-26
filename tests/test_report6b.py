import numpy as np
import pandas as pd
import pytest

import experiment as ex
import featuresets as fs
import fingerprint as fp
import model
import report6b as rb


@pytest.fixture(autouse=True)
def _phase4_directories_are_off_limits(monkeypatch, tmp_path):
    """No test in this file may write to Phase 4's real data/models or data/predictions.

    experiment.run's output directories are keyword-only DEFAULTS, so any code path that
    forgets to pass them -- including a deliberately mutated refit_delta during a mutation
    check -- would write straight into Phase 4's real artifacts. That happened once. This
    redirects the defaults to tmp_path for every test in the file; monkeypatch restores them."""
    monkeypatch.setattr(ex.run, "__kwdefaults__",
                        {**ex.run.__kwdefaults__,
                         "out_models": tmp_path / "phase4_models_redirected",
                         "out_preds": tmp_path / "phase4_preds_redirected"})


# ---------------------------------------------------------------------------
# score_rows: the join and the selection, on a real booster in tmp_path
# ---------------------------------------------------------------------------

def _nlr_fixture(synthetic_table, tmp_path, drop_one=False):
    """Two B folds (one repo each), plus a Scenario A decoy on both sides. The NLR_NO_REPO
    files list the same rows in REVERSED order with row-identifying scores, so a positional
    join would pair the wrong rows."""
    t = ex.load_table_frame(synthetic_table())
    cols = fs.FEATURE_SETS["NO_LABEL_REPLAY"]
    mp = tmp_path / "nlr.txt"
    model.save(model.fit(t[cols], t["is_slow"], model.DEFAULT_PARAMS), mp)
    nlr, new, truth = [], [], {}
    for k, repo in enumerate(["o/r", "o/s"]):
        te = t[t["repo"] == repo].head(60)
        old_p, new_p = tmp_path / f"nlr_{k}.parquet", tmp_path / f"new_{k}.parquet"
        pd.DataFrame({"pr_id": te["pr_id"], "repo": te["repo"], "is_slow": te["is_slow"],
                      "p_hat": np.linspace(0, 1, len(te))}).to_parquet(old_p, index=False)
        ids = te["pr_id"].to_numpy()[::-1]
        if drop_one and k == 0:
            ids = ids[1:]
        scores = 1000.0 * (k + 1) + np.arange(len(ids), dtype=float)
        pd.DataFrame({"pr_id": ids, "p_hat": scores}).to_parquet(new_p, index=False)
        truth.update(dict(zip(ids, scores)))
        nlr.append({"scenario": "B", "fold": k, "model_path": str(mp), "pred_path": str(old_p)})
        new.append({"scenario": "B", "fold": k, "pred_path": str(new_p)})
    nlr.append({"scenario": "A", "fold": 0, "model_path": str(mp), "pred_path": str(tmp_path / "nlr_0.parquet")})
    new.append({"scenario": "A", "fold": 0, "pred_path": str(tmp_path / "new_0.parquet")})
    return t, nlr, new, truth


def test_score_rows_joins_on_pr_id_and_filters_the_scenario(synthetic_table, tmp_path):
    t, nlr, new, truth = _nlr_fixture(synthetic_table, tmp_path)
    rows, sv, additivity, same = rb.score_rows("B", t, nlr, new)
    assert len(rows) == 120                                      # the A decoy added nothing
    assert set(fp.ROW_COLS) <= set(rows.columns)
    assert sorted(rows["fold"].unique()) == [0, 1]
    assert np.array_equal(rows["p_no_repo"].to_numpy(), rows["pr_id"].map(truth).to_numpy())
    assert same == {"B_fold0": True, "B_fold1": True}
    assert set(additivity) == {"B_fold0", "B_fold1"} and max(additivity.values()) < 1e-6
    assert sv.shape == (120, 33)
    assert np.allclose(rows["repo_contrib"], fp.repo_contribution(sv, fs.FEATURE_SETS["NO_LABEL_REPLAY"]))


def test_score_rows_reports_mismatched_test_rows_instead_of_crashing(synthetic_table, tmp_path):
    t, nlr, new, _ = _nlr_fixture(synthetic_table, tmp_path, drop_one=True)
    _, _, _, same = rb.score_rows("B", t, nlr, new)
    assert same == {"B_fold0": False, "B_fold1": True}


# ---------------------------------------------------------------------------
# reliance, coverage, fold_table
# ---------------------------------------------------------------------------

NLR = fs.FEATURE_SETS["NO_LABEL_REPLAY"]


def test_reliance_repo_shares():
    sv_a, sv_b = np.zeros((4, len(NLR))), np.zeros((4, len(NLR)))
    repo_i, other_i = NLR.index(fp.REPO_FEATURES[0]), NLR.index("body_len")
    sv_a[:, repo_i], sv_a[:, other_i] = 3.0, 1.0          # A: the repo feature carries 3 of 4
    sv_b[:, repo_i], sv_b[:, other_i] = 1.0, 3.0          # B: it carries 1 of 4
    rel = rb.reliance(sv_a, sv_b)
    assert list(rel.columns) == ["feature", "share_a", "share_b", "is_repo_feature"]
    assert rel.loc[rel.is_repo_feature, "share_a"].sum() == pytest.approx(0.75)
    assert rel.loc[rel.is_repo_feature, "share_b"].sum() == pytest.approx(0.25)


def test_reliance_is_sorted_by_b_share():
    """Three features whose B order (f3, f1, f2) differs from their positional order, their
    A-importance order and their lexicographic order -- so an unsorted result fails however
    it happens to be ordered."""
    f1, f2, f3 = "body_len", "open_backlog_at_t", "prior_merge_rate_here"
    sv_a, sv_b = np.zeros((2, len(NLR))), np.zeros((2, len(NLR)))
    for f, a, b in ((f1, 3.0, 2.0), (f2, 2.0, 1.0), (f3, 1.0, 3.0)):
        sv_a[:, NLR.index(f)], sv_b[:, NLR.index(f)] = a, b
    rel = rb.reliance(sv_a, sv_b)
    assert rel["feature"].head(3).tolist() == [f3, f1, f2]


def _cov(twice=False):
    runs = [{"scenario": "A", "test_repos": ["r0", "r1", "r2"]},
            {"scenario": "B", "test_repos": ["r0", "r1"]},
            {"scenario": "B", "test_repos": ["r2", "r0"] if twice else ["r2"]}]
    rows = pd.DataFrame({"repo": ["r0", "r1", "r2"]})
    return runs, rows, rows.copy()


def test_coverage_accepts_the_paired_design():
    assert rb.coverage(*_cov()) == {"n_repos": 3, "same_repo_set": True, "b_each_once": True}


def test_coverage_flags_a_repo_held_out_twice():
    assert rb.coverage(*_cov(twice=True))["b_each_once"] is False


def test_fold_table_scores_each_fold_on_its_own_rows():
    rows = pd.DataFrame({"fold": [0, 0, 1, 1], "is_slow": [0, 1, 0, 1],
                         "p_nlr": [0.2, 0.9, 0.9, 0.2], "p_no_repo": [0.9, 0.2, 0.2, 0.9]})
    ft = rb.fold_table({"B": rows}).set_index("fold")
    assert ft.loc[0, "auc_pr_nlr"] == pytest.approx(1.0)
    assert ft.loc[1, "auc_pr_nlr"] == pytest.approx(0.5)
    assert ft.loc[0, "delta"] == pytest.approx(ft.loc[0, "auc_pr_no_repo"] - ft.loc[0, "auc_pr_nlr"])
    assert ft["n_test"].tolist() == [2, 2]


# ---------------------------------------------------------------------------
# refit_delta: the behaviour-preserving refit (gate check 2)
# ---------------------------------------------------------------------------

def test_refit_delta_measures_against_the_saved_value(synthetic_table, tmp_path):
    t = ex.load_table_frame(synthetic_table())
    tr, te = ex.folds_for("A", t)[0]
    ref = ex.run("A", 0, "NO_LABEL_REPLAY", t, model.DEFAULT_PARAMS, tr, te,
                 out_models=tmp_path / "m", out_preds=tmp_path / "p")
    assert rb.refit_delta(t, model.DEFAULT_PARAMS, ref["auc_pr"]) == 0.0
    assert rb.refit_delta(t, model.DEFAULT_PARAMS, ref["auc_pr"] + 0.25) == pytest.approx(-0.25)


def test_refit_delta_never_writes_to_phase4s_directories(synthetic_table, monkeypatch):
    seen = {}
    real = ex.run

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(ex, "run", spy)
    rb.refit_delta(ex.load_table_frame(synthetic_table()), model.DEFAULT_PARAMS, 0.5)
    assert seen["out_models"] != ex.MODELS_DIR and seen["out_preds"] != ex.PRED_DIR
    assert seen["cols"] == fs.FEATURE_SETS["NO_LABEL_REPLAY"]       # through the NEW code path


# ---------------------------------------------------------------------------
# gate_checks
# ---------------------------------------------------------------------------

TAGS = ["A_fold0"] + [f"B_fold{k}" for k in range(5)]


def _ok():
    return dict(
        additivity={t: 1e-12 for t in TAGS},
        refit_delta=0.0,
        same_rows={t: True for t in TAGS},
        integrity={"non_constant": [], "equals_full_minus_no_snapshot": True,
                   "n_no_repo_cols": 25, "hygiene_ok": True},
        cov={"n_repos": 39, "same_repo_set": True, "b_each_once": True},
        intervals={"T_lo": 0.1, "T_hi": 0.5, "I_lo": -0.01, "I_hi": 0.04},
        artifacts={"phase6b_transfer": 39, "phase6b_intervention": 6,
                   "phase6b_reliance": 33, "phase6b_runs": 6},
    )


def _failed(checks):
    return [c["id"] for c in checks if not c["pass"]]


def test_gate_all_pass():
    checks = rb.gate_checks(**_ok())
    assert [c["id"] for c in checks] == [1, 2, 3, 4, 5] and _failed(checks) == []


def _m_additivity(d): d["additivity"]["B_fold3"] = 1e-3
def _m_booster_missing(d): del d["additivity"]["B_fold4"]
def _m_refit(d): d["refit_delta"] = 2e-4
def _m_no_refit(d): d["refit_delta"] = None
def _m_rows(d): d["same_rows"]["B_fold1"] = False
def _m_varies(d): d["integrity"]["non_constant"] = ["n_ci_workflows"]
def _m_nan(d): d["intervals"]["I_hi"] = float("nan")
def _m_twice(d): d["cov"]["b_each_once"] = False
def _m_empty(d): d["artifacts"]["phase6b_reliance"] = 0


@pytest.mark.parametrize("mutate,expected", [
    (_m_additivity, 1), (_m_booster_missing, 1), (_m_refit, 2), (_m_no_refit, 2), (_m_rows, 3),
    (_m_varies, 4), (_m_nan, 5), (_m_twice, 5), (_m_empty, 5),
])
def test_gate_each_failure_flips_only_its_own_check(mutate, expected):
    d = _ok()
    mutate(d)
    assert _failed(rb.gate_checks(**d)) == [expected]


def test_gate_without_intervals_fails_only_check_5():
    d = _ok(); d["intervals"] = None
    assert _failed(rb.gate_checks(**d)) == [5]


def test_hard_stopped_names_only_checks_1_and_2():
    checks = [{"id": i, "pass": False} for i in range(1, 6)]
    assert rb.hard_stopped(checks) == [1, 2]
    assert rb.hard_stopped([{"id": i, "pass": i != 4} for i in range(1, 6)]) == []


# ---------------------------------------------------------------------------
# verdict_text and render
# ---------------------------------------------------------------------------

def _stats(**kw):
    s = {"T": 0.8, "T_lo": 0.4, "T_hi": 1.2, "rho_a": 0.9, "rho_b": 0.1,
         "I": 0.05, "I_lo": 0.01, "I_hi": 0.09, "delta_a": -0.03, "delta_b": 0.02,
         "t_outcome": "confirms", "i_outcome": "confirms", "verdict": "SUPPORTED",
         "share_a": 0.286, "share_b": 0.410, "reliance": "higher",
         "n_draws": 2000, "n_repos": 39, "dropped_T": 0, "dropped_I": 0}
    return {**s, **kw}


# What each verdict's text must SAY, pinned here independently of report6b's own HEADLINE
# and READING dicts. Checking `rb.READING[code] in text` would compare the module against
# itself, and swapping two entries would still pass.
HEAD_SAYS = {
    "SUPPORTED": "fingerprinting is supported",
    "PARTIAL_SHAP_ONLY": "the SHAP transfer test confirms, the intervention does not",
    "PARTIAL_INTERVENTION_ONLY": "the intervention confirms, the SHAP transfer test does not",
    "NOT_SUPPORTED": "fingerprinting is not supported",
    "CONFLICTING": "The two tests conflict",
    "CONTRADICTED": "fingerprinting is contradicted",
}
READ_SAYS = {
    "SUPPORTED": "pre-registered signature of repo fingerprinting",
    "PARTIAL_SHAP_ONLY": "does not measurably favour cold start",
    "PARTIAL_INTERVENTION_ONLY": "not visible in the attributions",
    "NOT_SUPPORTED": "not explained by this mechanism",
    "CONFLICTING": "cannot adjudicate",
    "CONTRADICTED": "points away from fingerprinting",
}


@pytest.mark.parametrize("code", sorted(set(fp.VERDICTS.values())))
def test_verdict_text_says_the_right_thing_for_its_cell(code):
    text = rb.verdict_text(_stats(verdict=code))
    first = text.split("\n")[0]                                    # the headline comes first
    assert HEAD_SAYS[code] in first
    assert all(HEAD_SAYS[o] not in first for o in set(HEAD_SAYS) - {code})
    assert READ_SAYS[code] in text
    assert all(READ_SAYS[o] not in text for o in set(READ_SAYS) - {code})
    assert "+0.80" in text and "[+0.40, +1.20]" in text          # the transfer interval
    assert "+0.050" in text and "[+0.010, +0.090]" in text        # the intervention interval


def test_verdict_text_states_the_crossover_only_when_it_holds():
    yes = rb.verdict_text(_stats(delta_a=-0.03, delta_b=0.0))      # hurts A, B unharmed: Δ_B = 0 counts
    no = rb.verdict_text(_stats(delta_a=0.01, delta_b=0.02))       # helps both: not a crossover
    assert "The strong form holds" in yes and "does not hold" not in yes
    assert "does not hold" in no and "The strong form holds" not in no


def _checks(fail=()):
    return [{"id": i, "check": f"check {i}",
             "value": ({"non_constant": ["n_ci_workflows"] if 4 in fail else [],
                        "equals_full_minus_no_snapshot": 4 not in fail,
                        "n_no_repo_cols": 24 if 4 in fail else 25, "hygiene_ok": True}
                       if i == 4 else "ok"),
             "pass": i not in fail}
            for i in range(1, 6)]


def _frames():
    per_repo = pd.DataFrame({"repo": ["r0"], "c_A": [0.1], "y_A": [0.5], "n_A": [10],
                             "c_B": [0.0], "y_B": [0.4], "n_B": [20]})
    folds = pd.DataFrame({"scenario": ["A"], "fold": [0], "n_test": [10], "auc_pr_nlr": [0.9],
                          "auc_pr_no_repo": [0.88], "delta": [-0.02]})
    rel = pd.DataFrame({"feature": ["n_mentionable_users"], "share_a": [0.04], "share_b": [0.17],
                        "is_repo_feature": [True]})
    return per_repo, folds, rel


def test_render_puts_the_verdict_first():
    doc = rb.render(_stats(), _checks(), *_frames())
    assert doc.index("## Verdict") < doc.index("## Gate") < doc.index("## 2. Transfer test")
    assert rb.HEADLINE["SUPPORTED"] in doc and "No verdict" not in doc


@pytest.mark.parametrize("failed", [(1,), (2,), (1, 2)])
def test_render_suppresses_the_verdict_on_a_hard_stop(failed):
    doc = rb.render(_stats(), _checks(fail=failed), *_frames())
    assert "HARD STOP" in doc and "## Verdict" not in doc and "## 2. Transfer test" not in doc
    assert all(h not in doc for h in rb.HEADLINE.values())


def test_render_without_statistics_reports_no_verdict():
    doc = rb.render(None, _checks(fail=(5,)), None, None, None)
    assert "No verdict is reported" in doc and "## Verdict" not in doc


def test_render_with_a_soft_failure_still_reports_the_verdict():
    doc = rb.render(_stats(), _checks(fail=(4,)), *_frames())
    assert "## Verdict" in doc and "**FAIL**" in doc and "HARD STOP" not in doc


def test_render_states_the_feature_premise_only_when_check_4_passes():
    doc = rb.render(_stats(), _checks(), *_frames())
    assert "so each is constant within every repo" in doc
    doc = rb.render(_stats(), _checks(fail=(4,)), *_frames())
    assert "so each is constant within every repo" not in doc and "Gate check 4 failed" in doc


def test_render_reports_the_observed_no_repo_column_count():
    doc = rb.render(_stats(), _checks(), *_frames())
    assert "(25 features)" in doc
    doc = rb.render(_stats(), _checks(fail=(4,)), *_frames())
    assert "(24 features)" in doc and "(25 features)" not in doc


def test_render_flags_a_failed_gate_under_the_verdict():
    doc = rb.render(_stats(), _checks(fail=(3,)), *_frames())
    assert "## Verdict" in doc and "The validity gate failed" in doc
    assert doc.index("## Verdict") < doc.index("The validity gate failed") < doc.index(rb.HEADLINE["SUPPORTED"])
    assert "The validity gate failed" not in rb.render(_stats(), _checks(), *_frames())


def test_render_points_the_reader_the_right_way_to_the_gate_table():
    """Section 1 comes AFTER the gate table, the verdict BEFORE it. Each pointer must be
    true from where it sits -- a reader who follows it must find the table."""
    doc = rb.render(_stats(), _checks(fail=(4,)), *_frames())
    gate_at = doc.index("## Gate")
    verdict_part, sec1 = doc[:gate_at], doc[doc.index("## 1."):doc.index("## 2.")]
    assert "gate table below" in verdict_part and "gate table above" not in verdict_part
    assert "gate table above" in sec1 and "gate table below" not in sec1

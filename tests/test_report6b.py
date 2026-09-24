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

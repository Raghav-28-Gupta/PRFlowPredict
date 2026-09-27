import json
import numpy as np
import pandas as pd
import pytest
import experiment as ex
import featuresets as fs
import model
import splits


def test_load_table_drops_truncated_and_sorts(synthetic_table, tmp_path):
    t = synthetic_table()
    t.loc[5, "timeline_may_be_truncated"] = True
    t = t.sample(frac=1, random_state=1)                      # shuffle
    p = tmp_path / "f.parquet"; t.to_parquet(p, index=False)
    out = ex.load_table(p)
    assert len(out) == 1999 and out["created_at"].is_monotonic_increasing
    assert str(out["language_dominant"].dtype) == "category"
    assert list(out.index) == list(range(1999))


def test_folds_for_shapes(synthetic_table):
    t = ex.load_table_frame(synthetic_table())
    a = ex.folds_for("A", t); b = ex.folds_for("B", t, n_splits=2)
    assert len(a) == 1 and len(b) == 2
    tr, te = a[0]
    assert (t.loc[te, "created_at"] >= splits.CUTOFF_A).all()
    assert (t.loc[tr, "created_at"] < splits.CUTOFF_A).all()


def test_run_end_to_end(synthetic_table, tmp_path):
    t = ex.load_table_frame(synthetic_table())
    tr, te = ex.folds_for("A", t)[0]
    res = ex.run("A", 0, "FULL", t, model.DEFAULT_PARAMS, tr, te,
                 out_models=tmp_path / "m", out_preds=tmp_path / "p")
    for k in ("precision_at_10", "p10_ci_lo", "p10_ci_hi", "base_rate_p10", "auc_pr", "base_rate",
              "baseline_p10", "baseline_auc_pr", "n_test_repos", "train_max_created", "test_min_created"):
        assert k in res, k
    assert res["n_test_repos"] == 2 and res["p10_ci_lo"] <= res["precision_at_10"] <= res["p10_ci_hi"]
    assert pd.Timestamp(res["train_max_created"]) < pd.Timestamp(res["test_min_created"])
    assert res["auc_pr"] > res["base_rate"]                    # planted signal is learnable
    preds = pd.read_parquet(res["pred_path"])
    assert set(preds.columns) >= {"pr_id", "repo", "created_at", "is_slow", "p_hat", "baseline_score",
                                  "is_first_pr_here", "created_hour_utc", "diff_is_exact"}
    assert len(preds) == len(te)
    assert (tmp_path / "m" / "A_FULL_fold0.txt").exists()


def test_run_scenario_b_repos_disjoint(synthetic_table, tmp_path):
    t = ex.load_table_frame(synthetic_table())
    for k, (tr, te) in enumerate(ex.folds_for("B", t, n_splits=2)):
        res = ex.run("B", k, "PR_ONLY", t, model.DEFAULT_PARAMS, tr, te,
                     out_models=tmp_path / "m", out_preds=tmp_path / "p")
        assert not (set(res["train_repos"]) & set(res["test_repos"]))


def test_params_sha_is_stable(tmp_path):
    p = tmp_path / "params.json"; p.write_text('{"best_params": {"num_leaves": 8}}', encoding="utf-8")
    assert ex.params_sha(p) == ex.params_sha(p) and len(ex.params_sha(p)) == 12


def test_run_cols_none_is_unchanged_behaviour(synthetic_table, tmp_path):
    """Passing the registered set explicitly under a NEW name must reproduce the name lookup
    exactly. The new name matters: with name="FULL", a cols argument that was silently
    ignored would still pass."""
    t = ex.load_table_frame(synthetic_table())
    tr, te = ex.folds_for("A", t)[0]
    ref = ex.run("A", 0, "FULL", t, model.DEFAULT_PARAMS, tr, te,
                 out_models=tmp_path / "m1", out_preds=tmp_path / "p1")
    new = ex.run("A", 0, "FULL_EXPLICIT", t, model.DEFAULT_PARAMS, tr, te,
                 out_models=tmp_path / "m2", out_preds=tmp_path / "p2", cols=fs.FEATURE_SETS["FULL"])
    a = pd.read_parquet(ref["pred_path"])["p_hat"].to_numpy()
    b = pd.read_parquet(new["pred_path"])["p_hat"].to_numpy()
    assert np.array_equal(a, b)
    assert ref["auc_pr"] == new["auc_pr"]
    assert (tmp_path / "m2" / "A_FULL_EXPLICIT_fold0.txt").exists()


def test_run_cols_is_actually_used(synthetic_table, tmp_path):
    """Dropping the planted signal column must change the predictions. This proves the
    override reaches the model rather than being accepted and discarded."""
    t = ex.load_table_frame(synthetic_table())
    tr, te = ex.folds_for("A", t)[0]
    full = ex.run("A", 0, "FULL", t, model.DEFAULT_PARAMS, tr, te,
                  out_models=tmp_path / "m1", out_preds=tmp_path / "p1")
    fewer = [c for c in fs.FEATURE_SETS["FULL"] if c != "body_len"]
    cut = ex.run("A", 0, "FULL_MINUS_BODY", t, model.DEFAULT_PARAMS, tr, te,
                 out_models=tmp_path / "m2", out_preds=tmp_path / "p2", cols=fewer)
    a = pd.read_parquet(full["pred_path"])["p_hat"].to_numpy()
    b = pd.read_parquet(cut["pred_path"])["p_hat"].to_numpy()
    assert not np.array_equal(a, b)


def test_run_cols_still_enforces_hygiene(synthetic_table, tmp_path):
    t = ex.load_table_frame(synthetic_table())
    tr, te = ex.folds_for("A", t)[0]
    with pytest.raises(ValueError):
        ex.run("A", 0, "LEAKY", t, model.DEFAULT_PARAMS, tr, te,
               out_models=tmp_path / "m", out_preds=tmp_path / "p", cols=["body_len", "is_slow"])

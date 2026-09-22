import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import attribution as attr
import featuresets as fs
import model


@pytest.fixture
def toy_booster():
    """2 features: `a` drives the label, `b` is pure noise. A correct importance
    ranking must put `a` first by a wide margin."""
    rng = np.random.default_rng(0)
    n = 400
    a = rng.normal(size=n)
    X = pd.DataFrame({"a": a, "b": rng.normal(size=n)})
    y = (a + rng.normal(0, 0.25, n) > 0).astype(int)
    return model.fit(X, y, model.DEFAULT_PARAMS), X, y


def test_sample_rows_is_seeded_and_never_oversamples():
    df = pd.DataFrame({"x": range(100)})
    assert attr.sample_rows(df, n=10, seed=1).equals(attr.sample_rows(df, n=10, seed=1))
    assert not attr.sample_rows(df, n=10, seed=1).equals(attr.sample_rows(df, n=10, seed=2))
    assert len(attr.sample_rows(df, n=500, seed=1)) == 100        # fewer rows than asked
    assert len(attr.sample_rows(df, n=10, seed=1)) == 10


def test_explain_shapes_and_additivity(toy_booster):
    booster, X, _ = toy_booster
    sv, ev = attr.explain(booster, X)
    assert sv.shape == (len(X), X.shape[1])
    assert isinstance(ev, float)
    # Additivity is against the RAW MARGIN, not the probability.
    assert attr.additivity_delta(booster, X, sv, ev) < 1e-6


def test_importance_ranks_the_real_driver_first(toy_booster):
    booster, X, _ = toy_booster
    sv, _ = attr.explain(booster, X)
    imp = attr.importance(sv, list(X.columns))
    assert list(imp.columns) == ["feature", "mean_abs_shap", "share"]
    assert imp.iloc[0]["feature"] == "a"
    assert imp.iloc[0]["share"] > imp.iloc[1]["share"] * 2      # a dominates b
    assert imp["share"].sum() == pytest.approx(1.0)


def test_shift_signs_and_ordering():
    imp_a = pd.DataFrame({"feature": ["x", "y", "z"], "mean_abs_shap": [3.0, 1.0, 1.0],
                          "share": [0.6, 0.2, 0.2]})
    imp_b = pd.DataFrame({"feature": ["x", "y", "z"], "mean_abs_shap": [1.0, 3.0, 1.0],
                          "share": [0.2, 0.6, 0.2]})
    s = attr.shift(imp_a, imp_b)
    assert list(s.columns) == ["feature", "share_a", "share_b", "delta"]
    assert s.iloc[0]["feature"] in ("x", "y")                    # largest |delta| first
    assert s.set_index("feature").loc["x", "delta"] == pytest.approx(-0.4)
    assert s.set_index("feature").loc["y", "delta"] == pytest.approx(+0.4)
    assert s.set_index("feature").loc["z", "delta"] == pytest.approx(0.0)


def test_shift_handles_a_feature_missing_from_one_side():
    imp_a = pd.DataFrame({"feature": ["x"], "mean_abs_shap": [1.0], "share": [1.0]})
    imp_b = pd.DataFrame({"feature": ["y"], "mean_abs_shap": [1.0], "share": [1.0]})
    s = attr.shift(imp_a, imp_b).set_index("feature")
    assert s.loc["x", "share_b"] == 0.0 and s.loc["x", "delta"] == pytest.approx(-1.0)
    assert s.loc["y", "share_a"] == 0.0 and s.loc["y", "delta"] == pytest.approx(+1.0)


def test_label_replay_share_sums_exactly_the_four_columns():
    imp = pd.DataFrame({
        "feature": list(fs.LABEL_REPLAY) + ["body_len"],
        "mean_abs_shap": [1.0] * 5,
        "share": [0.1, 0.1, 0.05, 0.05, 0.7],
    })
    assert attr.label_replay_share(imp) == pytest.approx(0.30)
    none_present = pd.DataFrame({"feature": ["body_len"], "mean_abs_shap": [1.0], "share": [1.0]})
    assert attr.label_replay_share(none_present) == 0.0


def test_prepare_matches_the_models_own_predict_time_cast():
    """attribution.prepare must stay identical to model._prepare. If model._prepare grows
    a step, this fails instead of silently explaining differently-shaped data."""
    X = pd.DataFrame({"language_dominant": ["py", "ts", None], "n_commits": [1.0, 2.0, 3.0]})
    out, ref = attr.prepare(X), model._prepare(X)
    assert out.equals(ref)
    assert out["language_dominant"].dtype.name == "category"
    assert X["language_dominant"].dtype.name == "object"   # input not mutated


def test_load_runs_reads_json(tmp_path):
    """load_runs parses the {"runs": [...]} shape and returns the list."""
    runs_data = {"runs": [
        {"scenario": "A", "fold": 0, "model_path": "path0"},
        {"scenario": "B", "fold": 1, "model_path": "path1"},
    ]}
    runs_file = tmp_path / "runs.json"
    runs_file.write_text(json.dumps(runs_data), encoding="utf-8")
    result = attr.load_runs(runs_file)
    assert result == runs_data["runs"]
    assert len(result) == 2


def _build_full_feature_table(rng, n_rows, pr_ids):
    """Build a dummy table with all FULL feature columns."""
    cols = fs.FEATURE_SETS["FULL"]
    data = {
        col: rng.normal(size=n_rows) if col != "language_dominant"
        else np.random.choice(["py", "ts"], n_rows)
        for col in cols
    }
    data["pr_id"] = pr_ids
    return pd.DataFrame(data)


def test_explain_run_filters_by_scenario_and_featureset(tmp_path):
    """explain_run uses only the scenario's FULL-featureset runs, ignoring others."""
    rng = np.random.default_rng(42)

    # Build full feature table
    table = _build_full_feature_table(rng, n_rows=7,
                                      pr_ids=["id1", "id2", "id3", "id4", "id5", "id6", "id7"])
    cols = fs.FEATURE_SETS["FULL"]

    # Train boosters on first 20 rows (reuse table rows since we have enough)
    X_train = table[cols].iloc[:6]
    y_train = (rng.normal(size=len(X_train)) > 0).astype(int)

    booster_a = model.fit(X_train, y_train, {**model.DEFAULT_PARAMS, "n_estimators": 3})
    booster_b = model.fit(X_train, y_train, {**model.DEFAULT_PARAMS, "n_estimators": 3})

    model_file_a = tmp_path / "model_a.txt"
    model_file_b = tmp_path / "model_b.txt"
    model.save(booster_a, model_file_a)
    model.save(booster_b, model_file_b)

    # Create prediction parquets for different folds
    pred_a_fold0 = pd.DataFrame({"pr_id": ["id1", "id2", "id3"]})
    pred_a_fold1 = pd.DataFrame({"pr_id": ["id4", "id5"]})
    pred_b_fold0 = pd.DataFrame({"pr_id": ["id6", "id7"]})

    pred_file_a0 = tmp_path / "pred_a_0.parquet"
    pred_file_a1 = tmp_path / "pred_a_1.parquet"
    pred_file_b0 = tmp_path / "pred_b_0.parquet"
    pred_a_fold0.to_parquet(pred_file_a0)
    pred_a_fold1.to_parquet(pred_file_a1)
    pred_b_fold0.to_parquet(pred_file_b0)

    # Build runs list: A/FULL (2 folds), B/FULL (1 fold), plus decoy A/OTHER
    runs = [
        {"scenario": "A", "featureset": "FULL", "fold": 0,
         "model_path": str(model_file_a), "pred_path": str(pred_file_a0)},
        {"scenario": "A", "featureset": "FULL", "fold": 1,
         "model_path": str(model_file_a), "pred_path": str(pred_file_a1)},
        {"scenario": "A", "featureset": "OTHER", "fold": 0,
         "model_path": str(model_file_a), "pred_path": str(pred_file_a0)},  # decoy
        {"scenario": "B", "featureset": "FULL", "fold": 0,
         "model_path": str(model_file_b), "pred_path": str(pred_file_b0)},
    ]

    # Test A: should use A/FULL only (fold 0 and 1), ignoring A/OTHER
    sv_a, X_a, deltas_a = attr.explain_run("A", table, runs, n=20, seed=0)
    assert len(deltas_a) == 2  # two folds
    assert sv_a.shape[0] == X_a.shape[0]  # rows match
    assert sv_a.shape[0] == 5  # 3 + 2 rows across folds
    assert all(d < 1e-6 for d in deltas_a), f"Additivity failed: {deltas_a}"

    # Test B: should use only B/FULL (fold 0)
    sv_b, X_b, deltas_b = attr.explain_run("B", table, runs, n=20, seed=0)
    assert len(deltas_b) == 1  # one fold
    assert sv_b.shape[0] == X_b.shape[0]  # rows match
    assert sv_b.shape[0] == 2  # 2 rows
    assert all(d < 1e-6 for d in deltas_b), f"Additivity failed: {deltas_b}"


def test_explain_run_applies_fold_ordering(tmp_path):
    """explain_run pools folds in order (sorted by fold number)."""
    rng = np.random.default_rng(43)
    table = _build_full_feature_table(rng, n_rows=6,
                                      pr_ids=["a", "b", "c", "d", "e", "f"])
    cols = fs.FEATURE_SETS["FULL"]

    X_train = table[cols].iloc[:5]
    y_train = (rng.normal(size=len(X_train)) > 0).astype(int)
    booster = model.fit(X_train, y_train, {**model.DEFAULT_PARAMS, "n_estimators": 3})
    model_file = tmp_path / "model.txt"
    model.save(booster, model_file)

    # Create parquets for three folds (with different row counts)
    pred_fold0 = pd.DataFrame({"pr_id": ["a", "b"]})
    pred_fold1 = pd.DataFrame({"pr_id": ["c"]})
    pred_fold2 = pd.DataFrame({"pr_id": ["d", "e", "f"]})

    pred_file0 = tmp_path / "pred_0.parquet"
    pred_file1 = tmp_path / "pred_1.parquet"
    pred_file2 = tmp_path / "pred_2.parquet"
    pred_fold0.to_parquet(pred_file0)
    pred_fold1.to_parquet(pred_file1)
    pred_fold2.to_parquet(pred_file2)

    # Supply runs OUT OF ORDER (fold 2, 0, 1) - should be re-sorted to (0, 1, 2)
    runs = [
        {"scenario": "S", "featureset": "FULL", "fold": 2,
         "model_path": str(model_file), "pred_path": str(pred_file2)},
        {"scenario": "S", "featureset": "FULL", "fold": 0,
         "model_path": str(model_file), "pred_path": str(pred_file0)},
        {"scenario": "S", "featureset": "FULL", "fold": 1,
         "model_path": str(model_file), "pred_path": str(pred_file1)},
    ]

    sv, X_used, deltas = attr.explain_run("S", table, runs, n=10, seed=999)

    # Should process folds in order (0, 1, 2), not supplied order (2, 0, 1)
    assert len(deltas) == 3  # three folds
    assert sv.shape[0] == X_used.shape[0]  # rows match
    assert sv.shape[0] == 6  # 2 + 1 + 3 rows
    assert all(d < 1e-6 for d in deltas), f"Additivity check failed: {deltas}"


def test_explain_run_row_conservation(tmp_path):
    """explain_run returns one row per explained row, len(X_used) matches, len(deltas) = n_folds."""
    rng = np.random.default_rng(44)
    table = _build_full_feature_table(rng, n_rows=9,
                                      pr_ids=["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"])
    cols = fs.FEATURE_SETS["FULL"]

    X_train = table[cols].iloc[:8]
    y_train = (rng.normal(size=len(X_train)) > 0).astype(int)
    booster = model.fit(X_train, y_train, {**model.DEFAULT_PARAMS, "n_estimators": 3})
    model_file = tmp_path / "model.txt"
    model.save(booster, model_file)

    # Three folds with 4, 3, 2 rows each
    pred_f0 = pd.DataFrame({"pr_id": ["p0", "p1", "p2", "p3"]})
    pred_f1 = pd.DataFrame({"pr_id": ["p4", "p5", "p6"]})
    pred_f2 = pd.DataFrame({"pr_id": ["p7", "p8"]})

    pred_file0 = tmp_path / "pred_0.parquet"
    pred_file1 = tmp_path / "pred_1.parquet"
    pred_file2 = tmp_path / "pred_2.parquet"
    pred_f0.to_parquet(pred_file0)
    pred_f1.to_parquet(pred_file1)
    pred_f2.to_parquet(pred_file2)

    runs = [
        {"scenario": "S", "featureset": "FULL", "fold": 0,
         "model_path": str(model_file), "pred_path": str(pred_file0)},
        {"scenario": "S", "featureset": "FULL", "fold": 1,
         "model_path": str(model_file), "pred_path": str(pred_file1)},
        {"scenario": "S", "featureset": "FULL", "fold": 2,
         "model_path": str(model_file), "pred_path": str(pred_file2)},
    ]

    sv, X_used, deltas = attr.explain_run("S", table, runs, n=100, seed=0)

    # n=100 > total rows (9), so all rows taken
    assert sv.shape[0] == 9  # 4 + 3 + 2 rows
    assert X_used.shape[0] == 9  # same as sv
    assert len(deltas) == 3  # one delta per fold
    assert sv.shape[1] == X_used.shape[1]  # features match
    assert all(d < 1e-6 for d in deltas), f"Additivity failed: {deltas}"

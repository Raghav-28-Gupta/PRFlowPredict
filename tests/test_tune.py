import json
import experiment as ex
import model
import tune


def test_cv_score_is_time_ordered_and_sane(synthetic_table):
    t = ex.load_table_frame(synthetic_table())
    tr, _ = ex.folds_for("A", t)[0]
    rows = t.loc[tr].sort_values("created_at")
    s = tune.cv_score(rows[tune.COLS], rows["is_slow"].astype(int), model.DEFAULT_PARAMS, n_splits=3)
    assert 0.0 < s <= 1.0


def test_tune_returns_valid_params_and_is_reproducible(synthetic_table):
    t = ex.load_table_frame(synthetic_table())
    r1 = tune.tune(t, n_trials=4, seed=1, n_splits=2)
    r2 = tune.tune(t, n_trials=4, seed=1, n_splits=2)
    assert set(r1["best_params"]) == set(model.TUNED_KEYS)
    assert r1["n_trials"] == 4 and len(r1["trials"]) == 4
    assert r1["best_params"] == r2["best_params"] and r1["cv_auc_pr"] == r2["cv_auc_pr"]
    model.fit(t[tune.COLS].iloc[:50], t["is_slow"].iloc[:50], r1["best_params"])   # accepted by fit


def test_main_writes_params(synthetic_table, tmp_path, monkeypatch):
    t = synthetic_table(); p = tmp_path / "f.parquet"; t.to_parquet(p, index=False)
    monkeypatch.setattr(ex, "TABLE", p); monkeypatch.setattr(ex, "PARAMS", tmp_path / "params.json")
    assert tune.main(["--trials", "3", "--splits", "2"]) == 0
    out = json.loads((tmp_path / "params.json").read_text())
    assert "best_params" in out and out["n_trials"] == 3

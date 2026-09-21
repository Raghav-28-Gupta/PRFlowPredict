import numpy as np
import pandas as pd
import report4


def _run(scenario, fold, name, **kw):
    base = {"scenario": scenario, "fold": fold, "featureset": name, "n_train": 100, "n_test": 50,
            "n_test_repos": 6, "precision_at_10": 0.6, "p10_ci_lo": 0.5, "p10_ci_hi": 0.7,
            "base_rate_p10": 0.5, "auc_pr": 0.7, "base_rate": 0.45, "baseline_p10": 0.5,
            "baseline_auc_pr": 0.6, "train_max_created": "2025-12-31 23:00:00+00:00",
            "test_min_created": "2026-01-01 00:00:00+00:00", "train_repos": ["a", "b"],
            "test_repos": ["a", "b"] if scenario == "A" else ["c", "d"],
            "model_path": "m", "pred_path": "p"}
    return {**base, **kw}


def _all_runs():
    runs = [_run("A", 0, n) for n in ("FULL", "NO_SNAPSHOT", "NO_LABEL_REPLAY", "PR_ONLY")]
    runs += [_run("B", k, n) for n in ("FULL", "NO_SNAPSHOT", "NO_LABEL_REPLAY", "PR_ONLY") for k in range(5)]
    return runs


def _experiments(sha, n=24):
    return pd.DataFrame({"model": ["lgbm"] * n, "params": [sha] * n, "p10_ci_lo": [0.5] * n, "p10_ci_hi": [0.7] * n})


def test_gate_all_pass():
    checks = report4.gate_checks(_all_runs(), "abc123", _experiments("abc123"), repro_delta=0.0)
    assert [c["id"] for c in checks] == [1, 2, 3, 4, 5] and all(c["pass"] for c in checks)


def test_gate_catches_time_leak():
    runs = _all_runs(); runs[0]["train_max_created"] = "2026-02-01 00:00:00+00:00"
    assert report4.gate_checks(runs, "s", _experiments("s"), 0.0)[0]["pass"] is False


def test_gate_catches_repo_leak():
    runs = _all_runs(); b = next(r for r in runs if r["scenario"] == "B"); b["test_repos"] = ["a"]
    assert report4.gate_checks(runs, "s", _experiments("s"), 0.0)[0]["pass"] is False


def test_gate_requires_no_label_replay_on_both_and_all_24_logged():
    runs = [r for r in _all_runs() if not (r["featureset"] == "NO_LABEL_REPLAY" and r["scenario"] == "B")]
    c = report4.gate_checks(runs, "s", _experiments("s"), 0.0)
    assert c[2]["pass"] is False and c[3]["pass"] is False
    assert report4.gate_checks(_all_runs(), "s", _experiments("s", n=23), 0.0)[3]["pass"] is False


def test_gate_repro_threshold():
    assert report4.gate_checks(_all_runs(), "s", _experiments("s"), 1e-7)[4]["pass"] is True
    assert report4.gate_checks(_all_runs(), "s", _experiments("s"), 1e-3)[4]["pass"] is False
    assert report4.gate_checks(_all_runs(), "s", _experiments("s"), None)[4]["pass"] is False


def test_bias_slices_have_expected_rows():
    rng = np.random.default_rng(0); n = 300
    pred = pd.DataFrame({"is_slow": rng.random(n) < 0.4, "p_hat": rng.random(n),
                         "is_first_pr_here": rng.random(n) < 0.3, "created_hour_utc": rng.integers(0, 24, n)})
    s = report4.bias_slices(pred)
    assert set(s["slice"]) == {"is_first_pr_here", "hour_bucket"}
    assert set(s[s["slice"] == "hour_bucket"]["level"]) == {"00-06", "06-12", "12-18", "18-24"}
    assert (s["n"] > 0).all() and s["n"].sum() == 2 * n
    assert set(s.columns) >= {"slice", "level", "n", "actual_rate", "mean_p_hat", "auc_pr"}


def test_summarise_means_over_folds():
    df = report4.summarise(_all_runs())
    assert set(df["scenario"]) == {"A", "B"} and len(df) == 8
    assert df[(df.scenario == "B") & (df.featureset == "FULL")]["n_folds"].iloc[0] == 5

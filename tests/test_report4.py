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


def _pool(a_median=64.0, b_median=433.0):
    return pd.DataFrame([
        {"scenario": "A", "n_test_rows": 14135, "n_repos": 39, "pool_median": a_median,
         "pool_min": 5, "repos_under_10_test_prs": 5, "repos_with_10plus_slow": 32, "months_spanned": 6},
        {"scenario": "B", "n_test_rows": 38444, "n_repos": 39, "pool_median": b_median,
         "pool_min": 158, "repos_under_10_test_prs": 0, "repos_with_10plus_slow": 39, "months_spanned": 30},
    ])


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


def test_gate_check4_ignores_rows_from_other_models_or_params():
    # 24 rows present, but none match this params sha -> check 4 must fail.
    wrong_sha = _experiments("OTHER_SHA")
    assert report4.gate_checks(_all_runs(), "s", wrong_sha, 0.0)[3]["pass"] is False
    # 24 rows with the right sha but a different model -> also fails.
    wrong_model = _experiments("s"); wrong_model["model"] = "baseline_trailing90"
    assert report4.gate_checks(_all_runs(), "s", wrong_model, 0.0)[3]["pass"] is False
    # Mixed: 24 matching + 5 unrelated rows -> still passes (the filter counts, not the length).
    mixed = pd.concat([_experiments("s"), _experiments("OTHER_SHA", n=5)], ignore_index=True)
    assert report4.gate_checks(_all_runs(), "s", mixed, 0.0)[3]["pass"] is True


def test_gate_check4_fails_on_non_finite_ci():
    runs = _all_runs(); runs[0]["p10_ci_hi"] = float("nan")
    assert report4.gate_checks(runs, "s", _experiments("s"), 0.0)[3]["pass"] is False
    runs = _all_runs(); runs[3]["p10_ci_lo"] = float("inf")
    assert report4.gate_checks(runs, "s", _experiments("s"), 0.0)[3]["pass"] is False


def test_gate_check3_fails_when_scenario_a_ablation_missing():
    runs = [r for r in _all_runs() if not (r["featureset"] == "NO_LABEL_REPLAY" and r["scenario"] == "A")]
    c = report4.gate_checks(runs, "s", _experiments("s"), 0.0)
    assert c[2]["pass"] is False                       # the A half of the AND
    assert "NO_LABEL_REPLAY" in c[2]["check"]


def test_headline_is_true_in_both_directions():
    """The write-up's most important sentence must not overclaim in either outcome."""
    win = report4.summarise([_run("A", 0, "FULL", precision_at_10=0.65, p10_ci_lo=0.55, p10_ci_hi=0.75,
                                  baseline_p10=0.50, base_rate_p10=0.48),
                             _run("B", 0, "FULL", precision_at_10=0.55, baseline_p10=0.50)])
    s = report4.headline(win, _pool())
    assert "that bar is met" in s and "+0.150" in s and "CI excludes the baseline" in s

    lose = report4.summarise([_run("A", 0, "FULL", precision_at_10=0.47, p10_ci_lo=0.40, p10_ci_hi=0.55,
                                   baseline_p10=0.50, base_rate_p10=0.45),
                              _run("B", 0, "FULL", precision_at_10=0.46, baseline_p10=0.48)])
    s = report4.headline(lose, _pool())
    assert "bar is NOT met" in s and "reportable finding" in s
    assert "that bar is met" not in s.replace("bar is NOT met", "")


def test_pool_comparability_shape(tmp_path):
    rows = []
    for sc, n, span in (("A", 40, "2026-01-01"), ("B", 400, "2024-01-01")):
        d = pd.DataFrame({"repo": ["r1"] * n, "is_slow": [True] * 12 + [False] * (n - 12),
                          "p_hat": 0.5, "created_at": pd.date_range(span, periods=n, freq="D", tz="UTC"),
                          "is_first_pr_here": False, "created_hour_utc": 1, "baseline_score": 0.5})
        p = tmp_path / f"{sc}.parquet"; d.to_parquet(p, index=False)
        rows.append(_run(sc, 0, "FULL", pred_path=str(p)))
    out = report4.pool_comparability(rows)
    assert list(out["scenario"]) == ["A", "B"]
    assert set(out.columns) == {"scenario", "n_test_rows", "n_repos", "pool_median", "pool_min",
                                "repos_under_10_test_prs", "repos_with_10plus_slow", "months_spanned"}
    assert out.loc[out.scenario == "A", "pool_median"].iloc[0] == 40
    assert out.loc[out.scenario == "B", "n_test_rows"].iloc[0] == 400
    assert (out["repos_with_10plus_slow"] == 1).all()


def test_headline_caveats_the_a_to_b_comparison():
    summary = report4.summarise([_run("A", 0, "FULL", precision_at_10=0.769, p10_ci_lo=0.669, p10_ci_hi=0.856,
                                      baseline_p10=0.585, base_rate_p10=0.641),
                                 _run("B", 0, "FULL", precision_at_10=0.796, baseline_p10=0.60)])
    s = report4.headline(summary, _pool())
    assert "pool-size artifact" in s and "AUC-PR" in s
    assert "64" in s and "433" in s

import numpy as np
import pandas as pd
import pytest

import errors


@pytest.fixture
def aligned():
    """6 rows. The planted extremes: row 0 is the worst false positive (not slow, p_hat
    0.99), row 5 the worst false negative (slow, p_hat 0.01)."""
    pred = pd.DataFrame({
        "pr_id": [f"p{i}" for i in range(6)],
        "repo": ["o/r"] * 3 + ["o/s"] * 3,
        "number": [10, 11, 12, 13, 14, 15],
        "is_slow": [False, False, True, True, False, True],
        "p_hat": [0.99, 0.60, 0.55, 0.40, 0.20, 0.01],
        "wait_h": [2.0, 5.0, np.nan, np.nan, 9.0, np.nan],
    })
    cols = ["f0", "f1", "f2"]
    feature_frame = pd.DataFrame({"f0": [5.0, 1, 1, 1, 1, 1],
                                  "f1": [0.0, 0, 0, 0, 0, 0],
                                  "f2": [1.0, 1, 1, 1, 1, 1]})
    # row 0: f1 dominates (|2.0|), then f2 (|0.5|), then f0 (|0.1|)
    shap_values = np.tile([0.1, 0.5, 0.05], (6, 1))
    shap_values[0] = [0.1, -2.0, 0.5]        # order by |value|: f1, f2, f0
    return pred, shap_values, cols, feature_frame


def test_worst_rows_picks_the_planted_extremes(aligned):
    pred, sv, cols, ff = aligned
    w = errors.worst_rows(pred, sv, cols, ff, n=1)
    assert len(w) == 2
    fp = w[w["kind"] == "fp"].iloc[0]
    fn = w[w["kind"] == "fn"].iloc[0]
    assert fp["pr_id"] == "p0" and fp["is_slow"] == False    # noqa: E712
    assert fn["pr_id"] == "p5" and fn["is_slow"] == True     # noqa: E712
    assert fp["p_hat"] == 0.99 and fn["p_hat"] == 0.01


def test_worst_rows_builds_the_github_url(aligned):
    pred, sv, cols, ff = aligned
    w = errors.worst_rows(pred, sv, cols, ff, n=1).set_index("pr_id")
    assert w.loc["p0", "url"] == "https://github.com/o/r/pull/10"
    assert w.loc["p5", "url"] == "https://github.com/o/s/pull/15"


def test_worst_rows_orders_contributions_by_absolute_shap(aligned):
    pred, sv, cols, ff = aligned
    w = errors.worst_rows(pred, sv, cols, ff, n=1).set_index("pr_id")
    r = w.loc["p0"]
    assert r["top1_feature"] == "f1" and r["top1_shap"] == pytest.approx(-2.0)
    assert r["top2_feature"] == "f2" and r["top2_shap"] == pytest.approx(0.5)
    assert r["top3_feature"] == "f0" and r["top3_shap"] == pytest.approx(0.1)
    assert r["top1_value"] == 0.0 and r["top3_value"] == 5.0      # the row's own values


def test_worst_rows_caps_at_available_rows(aligned):
    pred, sv, cols, ff = aligned
    w = errors.worst_rows(pred, sv, cols, ff, n=25)     # only 3 fp and 3 fn exist
    assert (w["kind"] == "fp").sum() == 3 and (w["kind"] == "fn").sum() == 3


def test_error_patterns_flags_the_inflated_feature():
    """f0 is 0 everywhere except the two worst rows, where it is 50."""
    ff_all = pd.DataFrame({"f0": [0.0] * 98 + [50.0, 50.0], "f1": np.arange(100.0)},
                          index=[f"p{i}" for i in range(100)])
    worst = pd.DataFrame({"pr_id": ["p98", "p99"]})
    out = errors.error_patterns(worst, ff_all, ["f1", "f0"])
    assert list(out.columns) == ["feature", "mean_worst", "mean_all", "z"]
    assert out.iloc[0]["feature"] == "f0"          # largest |z| first
    assert out.iloc[0]["z"] > 3


def test_error_patterns_survives_a_non_numeric_column():
    """language_dominant is categorical; it must not raise, and must not rank."""
    ff_all = pd.DataFrame({"f0": [0.0] * 98 + [50.0, 50.0], "lang": ["py"] * 100},
                          index=[f"p{i}" for i in range(100)])
    out = errors.error_patterns(pd.DataFrame({"pr_id": ["p98", "p99"]}), ff_all, ["lang", "f0"])
    assert out.iloc[0]["feature"] == "f0"
    assert out.set_index("feature").loc["lang", "z"] == 0.0

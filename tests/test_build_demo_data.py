"""build_demo_data.py's pure parts: labels, value formatting, driver text, validation."""
import numpy as np
import pandas as pd
import pytest

import build_demo_data as bdd
import featuresets as fs


def test_every_full_feature_has_a_label():
    assert set(bdd.DRIVER_LABELS) == set(fs.FEATURE_SETS["FULL"])


@pytest.mark.parametrize("feature,value,text", [
    ("trailing_90d_slow_rate", 0.456, "0.46"),
    ("open_backlog_at_t", 1234, "1,234"),
    ("additions_at_open", 6604.6, "6,605"),
    ("has_codeowners", True, "yes"),
    ("is_first_pr_here", False, "no"),
    ("language_dominant", "Go", "Go"),
    ("author_account_age_days", np.nan, "missing"),
])
def test_format_value(feature, value, text):
    assert bdd.format_value(feature, value) == text


def test_drivers_are_the_three_largest_by_size_with_the_direction_of_the_push():
    x = pd.Series({"trailing_90d_slow_rate": 0.25, "has_codeowners": True,
                   "open_backlog_at_t": 41, "is_weekend": False})
    text = bdd.format_drivers(np.array([0.1, -0.5, 0.3, 0.0]), x)
    assert text == ("has CODEOWNERS = yes ↓ · open PRs in the repo = 41 ↑ · "
                    "repo's recent slow rate = 0.25 ↑")


def _valid():
    return pd.DataFrame({
        "repo": ["o/a", "o/b"], "pr_id": ["P1", "P2"], "number": [1, 2], "url": ["u1", "u2"],
        "title": ["t1", "t2"], "score_a": [0.1, 0.2], "fold_b": [0, 1], "score_b": [0.3, 0.4],
        "drivers_a": ["d", "d"], "drivers_b": ["d", "d"], "base_a": [0.1, 0.1], "base_b": [0.2, 0.3],
    })


def test_validate_accepts_a_complete_extract():
    bdd.validate(_valid(), {"o/a": 0, "o/b": 1})


@pytest.mark.parametrize("breakage", ["missing B score", "wrong fold", "duplicate pr_id", "missing base"])
def test_validate_refuses_a_broken_extract(breakage):
    df = _valid()
    if breakage == "missing B score":
        df.loc[0, "score_b"] = np.nan
    elif breakage == "wrong fold":
        df.loc[1, "fold_b"] = 0
    elif breakage == "missing base":
        df.loc[0, "base_b"] = np.nan
    else:
        df.loc[1, "pr_id"] = "P1"
    with pytest.raises(ValueError):
        bdd.validate(df, {"o/a": 0, "o/b": 1})


def test_feature_list_is_the_full_features_in_model_order_with_labels_and_rates():
    listed = bdd.feature_list()
    assert [f["feature"] for f in listed] == list(fs.FEATURE_SETS["FULL"])
    assert all(f["label"] == bdd.DRIVER_LABELS[f["feature"]] for f in listed)
    assert {f["feature"] for f in listed if f["rate"]} == bdd.RATES


def test_check_logit_accepts_consistent_values_and_refuses_a_gap():
    sv = np.array([[0.5, -0.25], [1.0, 0.0]])
    base = 0.1
    score = 1 / (1 + np.exp(-(base + sv.sum(axis=1))))
    bdd.check_logit(sv, base, score)
    with pytest.raises(ValueError):
        bdd.check_logit(sv, base + 0.01, score)

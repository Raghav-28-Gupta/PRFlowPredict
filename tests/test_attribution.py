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

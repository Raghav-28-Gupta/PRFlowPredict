import numpy as np
import pandas as pd
import pytest
import model
import featuresets as fs
import metrics


def _xy(synthetic_table, seed=0):
    t = synthetic_table(seed)
    return t[fs.FEATURE_SETS["FULL"]], t["is_slow"].astype(int)


def test_fit_predict_beats_chance(synthetic_table):
    X, y = _xy(synthetic_table)
    b = model.fit(X.iloc[:1500], y.iloc[:1500], model.DEFAULT_PARAMS)
    p = model.predict(b, X.iloc[1500:])
    assert p.shape == (500,) and (0 <= p).all() and (p <= 1).all()
    assert metrics.auc_pr(y.iloc[1500:], p) > y.iloc[1500:].mean() + 0.1   # planted signal


def test_two_fits_are_identical(synthetic_table):
    X, y = _xy(synthetic_table)
    p1 = model.predict(model.fit(X, y, model.DEFAULT_PARAMS), X)
    p2 = model.predict(model.fit(X, y, model.DEFAULT_PARAMS), X)
    assert np.array_equal(p1, p2)


def test_categorical_is_used_and_survives_roundtrip(synthetic_table, tmp_path):
    X, y = _xy(synthetic_table)
    b = model.fit(X, y, model.DEFAULT_PARAMS)
    assert "language_dominant" in b.feature_name()
    model.save(b, tmp_path / "m.txt")
    b2 = model.load(tmp_path / "m.txt")
    assert np.allclose(model.predict(b, X), model.predict(b2, X))


def test_scale_pos_weight_from_y(synthetic_table):
    X, y = _xy(synthetic_table)
    b = model.fit(X, y, model.DEFAULT_PARAMS)
    expected = (len(y) - y.sum()) / y.sum()
    assert float(b.params["scale_pos_weight"]) == pytest.approx(expected)


def test_spec_param_names_are_translated(synthetic_table):
    X, y = _xy(synthetic_table)
    b = model.fit(X, y, {**model.DEFAULT_PARAMS, "n_estimators": 7, "min_child_samples": 33})
    assert b.num_trees() == 7
    assert int(b.params["min_data_in_leaf"]) == 33


def test_unknown_param_is_rejected(synthetic_table):
    X, y = _xy(synthetic_table)
    with pytest.raises(KeyError):
        model.fit(X, y, {**model.DEFAULT_PARAMS, "bogus": 1})

"""LightGBM in one place: fixed seeds, the categorical, scale_pos_weight, save/load.

Native lgb.train rather than the sklearn wrapper: native save/load of the booster and
unambiguous parameter names. The SPEC's names (n_estimators, min_child_samples) stay the
public vocabulary -- tune.py and params.json use them -- and are translated here."""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

SEED = 20260912
CATEGORICAL = ["language_dominant"]

# Everything that must never vary between runs. deterministic + single thread + seeds is
# what makes gate #5 (refit reproduces AUC-PR to 1e-6) achievable.
FIXED = {
    "objective": "binary", "seed": SEED, "deterministic": True, "force_row_wise": True,
    "num_threads": 1, "verbose": -1, "bagging_seed": SEED, "feature_fraction_seed": SEED,
}

# The tunable vocabulary (spec §7). n_estimators and min_child_samples are sklearn-style
# names translated below; the rest are native LightGBM names.
TUNED_KEYS = ("num_leaves", "learning_rate", "n_estimators", "min_child_samples",
              "feature_fraction", "bagging_fraction", "bagging_freq", "lambda_l2")
_TRANSLATE = {"min_child_samples": "min_data_in_leaf"}

DEFAULT_PARAMS = {
    "num_leaves": 31, "learning_rate": 0.05, "n_estimators": 200, "min_child_samples": 10,
    "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1, "lambda_l2": 1.0,
}


def _prepare(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def fit(X: pd.DataFrame, y, params: dict) -> lgb.Booster:
    unknown = set(params) - set(TUNED_KEYS)
    if unknown:
        raise KeyError(f"unknown params: {sorted(unknown)}")
    y = np.asarray(y).astype(int)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    native = {**FIXED, "scale_pos_weight": neg / max(pos, 1)}
    for k, v in params.items():
        if k == "n_estimators":
            continue
        native[_TRANSLATE.get(k, k)] = v
    ds = lgb.Dataset(_prepare(X), label=y,
                     categorical_feature=[c for c in CATEGORICAL if c in X.columns],
                     free_raw_data=False)
    return lgb.train(native, ds, num_boost_round=int(params["n_estimators"]))


def predict(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(booster.predict(_prepare(X)))


def save(booster: lgb.Booster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))


def load(path: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))

"""Exact TreeSHAP attribution over the Phase 4 boosters. Nothing is retrained.

Phase 4 established THAT cold-start transfer fails: on Scenario B, removing the four
label-replay features drops the model below the trailing-rate baseline, while on A the
same ablation still wins. This module establishes WHY, by measuring where each model's
attribution mass sits.

Two design points worth stating, because both are easy to get wrong:

  * `share` normalises each scenario's mean-|SHAP| to sum to 1. Raw SHAP magnitudes are
    not comparable across models (different base rates, different margins); shares are.
  * Additivity is a property of the RAW MARGIN, not the probability. sum(shap) +
    expected_value reproduces `booster.predict(X, raw_score=True)`. Comparing against the
    sigmoid output would fail for correct attributions -- see additivity_delta."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap

import featuresets as fs
import model
import splits

SEED = splits.SEED
SAMPLE_N = 5000
ROOT = Path(__file__).parent
RUNS_JSON = ROOT / "data" / "phase4_runs.json"


def sample_rows(df: pd.DataFrame, n: int = SAMPLE_N, seed: int = SEED) -> pd.DataFrame:
    """Seeded sample; a frame smaller than n is returned whole."""
    return df if len(df) <= n else df.sample(n=n, random_state=seed)


def prepare(X: pd.DataFrame) -> pd.DataFrame:
    """Same categorical cast model._prepare applies at predict time."""
    X = X.copy()
    for c in model.CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def explain(booster, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Exact TreeSHAP. Returns (values [n_rows, n_features], expected_value).

    shap 0.52.0 warns on every call that the LightGBM-binary output form "has changed to a
    list of ndarray", then returns a plain (n, n_features) array -- verified. The warning is
    silenced by message (never a blanket ignore), and the shape is ASSERTED, because a
    version that actually returns the list form would otherwise sail through np.asarray as
    a (2, n, f) array and silently corrupt every number in this phase."""
    Xp = prepare(X)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*output has changed to a list of ndarray.*",
                                category=UserWarning)
        ex = shap.TreeExplainer(booster)
        raw = ex.shap_values(Xp)
    if isinstance(raw, list):                       # defensive: future shap, binary -> [neg, pos]
        raw = raw[1]
    sv = np.asarray(raw)
    if sv.shape != (len(Xp), Xp.shape[1]):
        raise RuntimeError(f"unexpected SHAP shape {sv.shape}, expected {(len(Xp), Xp.shape[1])}")
    return sv, float(np.asarray(ex.expected_value).ravel()[-1])


def additivity_delta(booster, X: pd.DataFrame, shap_values: np.ndarray,
                     expected_value: float) -> float:
    """max |sum(shap) + expected_value - raw_margin|. The gate's hard stop."""
    margin = np.asarray(booster.predict(prepare(X), raw_score=True))
    return float(np.max(np.abs(shap_values.sum(axis=1) + expected_value - margin)))


def importance(shap_values: np.ndarray, cols: list[str]) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    total = mean_abs.sum()
    return (pd.DataFrame({"feature": cols, "mean_abs_shap": mean_abs,
                          "share": mean_abs / total if total else 0.0})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def shift(imp_a: pd.DataFrame, imp_b: pd.DataFrame) -> pd.DataFrame:
    """share_b - share_a per feature. The phase's headline artifact."""
    a = imp_a.set_index("feature")["share"].rename("share_a")
    b = imp_b.set_index("feature")["share"].rename("share_b")
    out = pd.concat([a, b], axis=1).fillna(0.0).reset_index()
    out["delta"] = out["share_b"] - out["share_a"]
    return (out.reindex(out["delta"].abs().sort_values(ascending=False).index)
            .reset_index(drop=True))


def label_replay_share(imp: pd.DataFrame) -> float:
    """Attribution mass on the baseline's own signal: the four label-replay features."""
    return float(imp.loc[imp["feature"].isin(fs.LABEL_REPLAY), "share"].sum())


def explain_run(scenario: str, table: pd.DataFrame, runs: list[dict],
                n: int = SAMPLE_N, seed: int = SEED) -> tuple[np.ndarray, pd.DataFrame, list[float]]:
    """Pool a scenario's FULL folds: one SHAP matrix, one X, one delta per fold.

    Scenario B has five boosters over the same 37 columns. Pooling row-wise is what makes
    'what does the B model lean on' a single answer."""
    cols = fs.FEATURE_SETS["FULL"]
    by_id = table.set_index("pr_id")
    mats, frames, deltas = [], [], []
    for r in sorted((r for r in runs if r["scenario"] == scenario and r["featureset"] == "FULL"),
                    key=lambda r: r["fold"]):
        pred = pd.read_parquet(r["pred_path"])
        X = prepare(sample_rows(by_id.loc[pred["pr_id"].to_numpy(), cols], n=n, seed=seed))
        booster = model.load(Path(r["model_path"]))
        sv, ev = explain(booster, X)
        deltas.append(additivity_delta(booster, X, sv, ev))
        mats.append(sv)
        frames.append(X)
    # The returned frame is the PREPARED one -- exactly the rows that were explained, so the
    # beeswarms plot the same thing the SHAP values describe. Re-prepared after the concat:
    # concatenating per-fold categoricals with different category sets yields object dtype.
    return np.vstack(mats), prepare(pd.concat(frames, ignore_index=True)), deltas


def load_runs(path: Path = RUNS_JSON) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["runs"]

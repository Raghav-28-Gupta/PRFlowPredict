"""Optuna on Scenario A's TRAINING rows only, time-ordered CV, FULL feature set.

Scenario A's held-out test period never influences tuning. Scenario B's held-out repos
do: their pre-2026 rows are part of Scenario A's training rows, since every repo is held
out in some fold (an unmeasured, likely small effect -- see docs/REPORT.md). The
best params are frozen in data/models/params.json and reused for every scenario, fold
and ablation -- the fair comparison, stated in the report."""
from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import experiment as ex
import featuresets as fs
import metrics
import model

log = logging.getLogger("tune")
SEED = model.SEED
COLS = fs.FEATURE_SETS["FULL"]


def suggest(trial: optuna.Trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
    }


def cv_score(X: pd.DataFrame, y, params: dict, n_splits: int = 3) -> float:
    """Mean validation AUC-PR over a time-ordered split; X must be sorted by time."""
    y = pd.Series(np.asarray(y).astype(int), index=X.index)
    scores = []
    for tr, va in TimeSeriesSplit(n_splits=n_splits).split(X):
        b = model.fit(X.iloc[tr], y.iloc[tr], params)
        scores.append(metrics.auc_pr(y.iloc[va], model.predict(b, X.iloc[va])))
    return float(np.mean(scores))


def tune(table: pd.DataFrame, n_trials: int = 100, seed: int = SEED, n_splits: int = 3) -> dict:
    tr, _ = ex.folds_for("A", table)[0]
    rows = table.loc[tr].sort_values(["created_at", "pr_id"])
    X, y = rows[COLS], rows["is_slow"].astype(int)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda t: cv_score(X, y, suggest(t), n_splits), n_trials=n_trials)
    best = {**study.best_params, "bagging_freq": 1}
    return {
        "best_params": best, "cv_auc_pr": float(study.best_value), "n_trials": n_trials,
        "seed": seed, "n_splits": n_splits, "n_train_rows": int(len(rows)),
        "trials": [{"number": t.number, "value": t.value, "params": {**t.params, "bagging_freq": 1}}
                   for t in study.trials],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--splits", type=int, default=3)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    table = ex.load_table()
    out = tune(table, n_trials=args.trials, n_splits=args.splits)
    ex.PARAMS.parent.mkdir(parents=True, exist_ok=True)
    ex.PARAMS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"best cv AUC-PR={out['cv_auc_pr']:.4f} over {out['n_trials']} trials -> {ex.PARAMS}")
    print(json.dumps(out["best_params"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""One experiment = (scenario, fold, feature set). Fit, predict, score, save, log.

Model and baseline are always scored on the SAME test rows: the baseline's score for a
row is the table's own trailing_90d_slow_rate (proven identical to baseline.py's value by
the Phase 3 audit), so no second pipeline is needed and no rows can differ."""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import featuresets as fs
import metrics
import model
import splits
import tracking

log = logging.getLogger("experiment")

ROOT = Path(__file__).parent
TABLE = ROOT / "data" / "features" / "features.parquet"
PARAMS = ROOT / "data" / "models" / "params.json"
MODELS_DIR = ROOT / "data" / "models"
PRED_DIR = ROOT / "data" / "predictions"
RUNS_JSON = ROOT / "data" / "phase4_runs.json"
SEED = splits.SEED
SCENARIOS = ("A", "B")


def load_table_frame(t: pd.DataFrame) -> pd.DataFrame:
    t = t[~(t["timeline_may_be_truncated"] == True)]                    # noqa: E712
    t = t.sort_values(["created_at", "pr_id"]).reset_index(drop=True)
    t["language_dominant"] = t["language_dominant"].astype("category")
    return t


def load_table(path: Path | None = None) -> pd.DataFrame:
    # Default resolved at CALL time so tests can monkeypatch experiment.TABLE.
    return load_table_frame(pd.read_parquet(path or TABLE))


def folds_for(scenario: str, table: pd.DataFrame, n_splits: int = 5):
    if scenario == "A":
        return [splits.scenario_a(table)]
    if scenario == "B":
        return splits.scenario_b(table, n_splits=n_splits)
    raise ValueError(scenario)


def params_sha(path: Path | None = None) -> str:
    return hashlib.sha256((path or PARAMS).read_bytes()).hexdigest()[:12]


def run(scenario: str, fold: int, name: str, table: pd.DataFrame, params: dict,
        train_idx: np.ndarray, test_idx: np.ndarray, *, seed: int = SEED,
        out_models: Path = MODELS_DIR, out_preds: Path = PRED_DIR) -> dict:
    cols = fs.FEATURE_SETS[name]
    fs.assert_hygiene(cols)
    tr, te = table.loc[train_idx], table.loc[test_idx]

    booster = model.fit(tr[cols], tr["is_slow"], params)
    p_hat = model.predict(booster, te[cols])
    y = te["is_slow"].to_numpy(dtype=int)
    repo = te["repo"].to_numpy()

    per_repo, p10 = metrics.precision_at_k(y, p_hat, repo, k=10, seed=seed)
    lo, hi = metrics.cluster_bootstrap(per_repo, seed=seed)
    base_score = te["trailing_90d_slow_rate"].to_numpy(dtype=float)
    _, base_p10 = metrics.precision_at_k(y, base_score, repo, k=10, seed=seed)

    tag = f"{scenario}_{name}_fold{fold}"
    model_path = out_models / f"{tag}.txt"
    pred_path = out_preds / f"{tag}.parquet"
    model.save(booster, model_path)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "pr_id": te["pr_id"].to_numpy(), "repo": repo, "created_at": te["created_at"].to_numpy(),
        "is_slow": y.astype(bool), "p_hat": p_hat, "baseline_score": base_score,
        "is_first_pr_here": te["is_first_pr_here"].to_numpy(),
        "created_hour_utc": te["created_hour_utc"].to_numpy(),
        "diff_is_exact": te["diff_is_exact"].to_numpy(),
    }).to_parquet(pred_path, index=False)

    return {
        "scenario": scenario, "fold": fold, "featureset": name,
        "n_train": int(len(tr)), "n_test": int(len(te)), "n_test_repos": int(te["repo"].nunique()),
        "precision_at_10": float(p10), "p10_ci_lo": float(lo), "p10_ci_hi": float(hi),
        "base_rate_p10": float(te.groupby("repo")["is_slow"].mean().mean()),
        "auc_pr": metrics.auc_pr(y, p_hat), "base_rate": float(y.mean()),
        "baseline_p10": float(base_p10), "baseline_auc_pr": metrics.auc_pr(y, base_score),
        "train_max_created": str(tr["created_at"].max()), "test_min_created": str(te["created_at"].min()),
        "train_repos": sorted(tr["repo"].unique().tolist()), "test_repos": sorted(te["repo"].unique().tolist()),
        "model_path": str(model_path), "pred_path": str(pred_path),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if not PARAMS.exists():
        log.error("no %s -- run tune.py first", PARAMS)
        return 1
    params = json.loads(PARAMS.read_text(encoding="utf-8"))["best_params"]
    sha = params_sha()
    table = load_table()
    results = []
    for scenario in SCENARIOS:
        folds = folds_for(scenario, table)
        for name in fs.FEATURE_SETS:
            for k, (tr, te) in enumerate(folds):
                res = run(scenario, k, name, table, params, tr, te)
                results.append(res)
                tracking.log({
                    "scenario": scenario, "fold": k, "model": "lgbm", "features": name, "params": sha,
                    "n_train": res["n_train"], "n_test": res["n_test"], "n_test_repos": res["n_test_repos"],
                    "precision_at_10": res["precision_at_10"], "p10_ci_lo": res["p10_ci_lo"],
                    "p10_ci_hi": res["p10_ci_hi"], "base_rate_p10": res["base_rate_p10"],
                    "auc_pr": res["auc_pr"], "base_rate": res["base_rate"],
                    "notes": f"baseline_p10={res['baseline_p10']:.3f} baseline_auc_pr={res['baseline_auc_pr']:.3f}",
                })
                log.info("%s %-16s fold %d: P@10=%.3f [%.3f,%.3f] base=%.3f bl=%.3f | AUC-PR=%.3f bl=%.3f",
                         scenario, name, k, res["precision_at_10"], res["p10_ci_lo"], res["p10_ci_hi"],
                         res["base_rate_p10"], res["baseline_p10"], res["auc_pr"], res["baseline_auc_pr"])
    RUNS_JSON.write_text(json.dumps({"params_sha": sha, "runs": results}, indent=2), encoding="utf-8")
    print(f"{len(results)} runs -> {RUNS_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

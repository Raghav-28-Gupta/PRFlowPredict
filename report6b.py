"""Phase 6b report: the pre-registered repo-fingerprinting test.

Tests whether the NO_LABEL_REPLAY model's cold-start failure is repo fingerprinting -- the
model identifying repos by their static attributes and replaying their base rates.

Pre-registered in docs/superpowers/specs/2026-09-24-phase6b-fingerprinting-design.md,
committed before any of this ran. The document this writes is FULLY GENERATED: there is no
hand-written section, because Phase 6's hand-written section 4 showed what one costs inside
a generated document -- a re-run silently destroys it.

Re-running is safe but not free: it re-trains the six NLR_NO_REPO boosters (deterministic,
so identical) and appends their six rows to data/experiments.csv again. Phase 4's gate
check 4 counts only Phase 4's own feature sets, so that does not disturb it."""
from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import attribution as attr
import experiment as ex
import featuresets as fs
import fingerprint as fp
import metrics
import model
import tracking


def train_no_repo(table: pd.DataFrame, params: dict, sha: str, *,
                  out_models: Path = ex.MODELS_DIR, out_preds: Path = ex.PRED_DIR) -> list[dict]:
    """The intervention (spec section 5.3): NLR_NO_REPO on Phase 4's identical folds and
    tuned params, one fit for A and five for B, each logged to data/experiments.csv."""
    cols = fp.nlr_no_repo_cols()
    runs = []
    for scenario in ex.SCENARIOS:
        for k, (tr, te) in enumerate(ex.folds_for(scenario, table)):
            res = ex.run(scenario, k, fp.NO_REPO_NAME, table, params, tr, te,
                         out_models=out_models, out_preds=out_preds, cols=cols)
            runs.append(res)
            tracking.log({
                "scenario": scenario, "fold": k, "model": "lgbm", "features": fp.NO_REPO_NAME,
                "params": sha, "n_train": res["n_train"], "n_test": res["n_test"],
                "n_test_repos": res["n_test_repos"], "precision_at_10": res["precision_at_10"],
                "p10_ci_lo": res["p10_ci_lo"], "p10_ci_hi": res["p10_ci_hi"],
                "base_rate_p10": res["base_rate_p10"], "auc_pr": res["auc_pr"],
                "base_rate": res["base_rate"],
                "notes": "phase6b intervention: NO_LABEL_REPLAY minus the 8 repo-level snapshot features",
            })
    return runs


def refit_delta(table: pd.DataFrame, params: dict, saved_auc: float) -> float:
    """Gate check 2: re-fit A/NLR through the cols= code path and compare with Phase 4's
    saved AUC-PR. Everything goes to a scratch directory; the defaults are Phase 4's own
    directories and would overwrite A_NO_LABEL_REPLAY_fold0.txt."""
    tr, te = ex.folds_for("A", table)[0]
    with tempfile.TemporaryDirectory() as d:
        res = ex.run("A", 0, "NO_LABEL_REPLAY", table, params, tr, te,
                     out_models=Path(d) / "m", out_preds=Path(d) / "p",
                     cols=fs.FEATURE_SETS["NO_LABEL_REPLAY"])
    return float(res["auc_pr"] - saved_auc)


def score_rows(scenario: str, table: pd.DataFrame, nlr_runs: list[dict], no_repo_runs: list[dict]
               ) -> tuple[pd.DataFrame, np.ndarray, dict[str, float], dict[str, bool]]:
    """Every test row of one scenario: fold, label, both models' scores, and the NLR model's
    summed repo-feature SHAP. The two models' predictions are joined on pr_id, never by
    position.

    Returns the rows, the pooled SHAP matrix (for reliance), each booster's additivity delta
    (gate check 1), and whether each fold's two test sets are identical (gate check 3)."""
    cols = fs.FEATURE_SETS["NO_LABEL_REPLAY"]
    by_id = table.set_index("pr_id")
    new = {r["fold"]: r for r in no_repo_runs if r["scenario"] == scenario}
    parts, mats, additivity, same = [], [], {}, {}
    for r in sorted((r for r in nlr_runs if r["scenario"] == scenario), key=lambda r: r["fold"]):
        k = r["fold"]
        tag = f"{scenario}_fold{k}"
        old = pd.read_parquet(r["pred_path"])[["pr_id", "repo", "is_slow", "p_hat"]]
        nw = pd.read_parquet(new[k]["pred_path"])[["pr_id", "p_hat"]]
        same[tag] = set(old["pr_id"]) == set(nw["pr_id"])
        m = old.merge(nw, on="pr_id", how="inner", suffixes=("_nlr", "_no_repo"), validate="one_to_one")
        X = attr.prepare(by_id.loc[m["pr_id"].to_numpy(), cols])
        booster = model.load(Path(r["model_path"]))
        sv, ev = attr.explain(booster, X)
        additivity[tag] = attr.additivity_delta(booster, X, sv, ev)
        parts.append(pd.DataFrame({
            "pr_id": m["pr_id"].to_numpy(), "repo": m["repo"].to_numpy(), "fold": k,
            "is_slow": m["is_slow"].to_numpy(dtype=int),
            "p_nlr": m["p_hat_nlr"].to_numpy(dtype=float),
            "p_no_repo": m["p_hat_no_repo"].to_numpy(dtype=float),
            "repo_contrib": fp.repo_contribution(sv, cols),
        }))
        mats.append(sv)
    return pd.concat(parts, ignore_index=True), np.vstack(mats), additivity, same


def reliance(sv_a: np.ndarray, sv_b: np.ndarray) -> pd.DataFrame:
    """Share of mean |SHAP| per feature on the NLR models, A vs B (spec section 5.1,
    descriptive). Sorted by B's share, descending."""
    cols = fs.FEATURE_SETS["NO_LABEL_REPLAY"]
    a = attr.importance(sv_a, cols).set_index("feature")["share"].rename("share_a")
    b = attr.importance(sv_b, cols).set_index("feature")["share"].rename("share_b")
    out = pd.concat([a, b], axis=1).rename_axis("feature").reset_index()
    out["is_repo_feature"] = out["feature"].isin(fp.REPO_FEATURES)
    return out.sort_values("share_b", ascending=False, kind="mergesort").reset_index(drop=True)


def coverage(nlr_runs: list[dict], rows_a: pd.DataFrame, rows_b: pd.DataFrame) -> dict:
    """Gate check 5's premise: A and B cover the same repos, and B holds each out once."""
    held = Counter(repo for r in nlr_runs if r["scenario"] == "B" for repo in r["test_repos"])
    ra, rb = set(rows_a["repo"]), set(rows_b["repo"])
    return {"n_repos": len(ra), "same_repo_set": ra == rb,
            "b_each_once": bool(held) and set(held) == rb and all(v == 1 for v in held.values())}


def fold_table(rows: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per (scenario, fold): both models' AUC-PR on identical rows."""
    out = []
    for sc, r in rows.items():
        for k, g in r.groupby("fold"):
            y = g["is_slow"].to_numpy(dtype=int)
            old, new = metrics.auc_pr(y, g["p_nlr"]), metrics.auc_pr(y, g["p_no_repo"])
            out.append({"scenario": sc, "fold": int(k), "n_test": int(len(g)),
                        "auc_pr_nlr": old, "auc_pr_no_repo": new, "delta": new - old})
    return pd.DataFrame(out)

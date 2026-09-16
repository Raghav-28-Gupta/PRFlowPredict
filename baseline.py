"""The baseline every model must beat: the repo's trailing-90-day slow rate,
thresholded at 0.5 (blueprint §2, model 1). No learning.

P@10 vs base rate is a reported finding, not a leakage test: the trailing rate varies
within a repo over time, so within-repo top-10 can beat the base rate via temporal
autocorrelation. Leakage is tested directly by a replay audit (§11 #5).

AUC-PR, being global, CAN exceed the base rate: the trailing rate ranks REPOS. That is
the honest statement of what a no-PR-signal model can do."""
from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

import cohort_qc
import labels
import load
import metrics
import replay
import splits
import tracking

log = logging.getLogger("baseline")


def score_rows(rows: pd.DataFrame, test_idx: np.ndarray, train_idx: np.ndarray,
               tier1: pd.DataFrame, label_primary: pd.DataFrame) -> np.ndarray:
    # g: the D5 slow rate over TRAINING rows of this split only (spec §6.2).
    g = float(rows.loc[train_idx, "is_slow"].mean())
    test = rows.loc[test_idx]
    hist = {r: replay.History.from_frames(r, tier1, label_primary, splits.WINDOW_START)
            for r in test["repo"].unique()}
    return np.array([
        hist[r].features_at(t, g)["trailing_90d_slow_rate"]
        for r, t in zip(test["repo"].to_numpy(), test["created_at"])
    ])


def replay_audit(rows, test_idx, train_idx, tier1, label_primary, n: int = 200,
                 seed: int = splits.SEED) -> dict:
    """Gate #5: prove features_at is leak-free by independent recomputation."""
    g = float(rows.loc[train_idx, "is_slow"].mean())
    test = rows.loc[test_idx]
    sample = test.sample(n=min(n, len(test)), random_state=seed)
    hist = {r: replay.History.from_frames(r, tier1, label_primary, splits.WINDOW_START)
            for r in sample["repo"].unique()}
    d_rate, d_back = 0.0, 0
    for r, t in zip(sample["repo"].to_numpy(), sample["created_at"]):
        a = hist[r].features_at(t, g)
        b = replay.brute_force_features(tier1, label_primary, r, t, g)
        d_rate = max(d_rate, abs(a["trailing_90d_slow_rate"] - b["trailing_90d_slow_rate"]))
        d_back = max(d_back, abs(a["open_backlog_at_t"] - b["open_backlog_at_t"]))
    return {"n": int(len(sample)), "max_abs_diff_rate": d_rate,
            "max_abs_diff_backlog": int(d_back),
            "pass": bool(d_rate < 1e-9 and d_back == 0)}


def evaluate(scenario: str, rows: pd.DataFrame, folds, tier1: pd.DataFrame,
             label_primary: pd.DataFrame, seed: int = splits.SEED) -> list[dict]:
    out = []
    for fold, (tr, te) in enumerate(folds):
        score = score_rows(rows, te, tr, tier1, label_primary)
        test = rows.loc[te]
        y = test["is_slow"].to_numpy(dtype=int)
        per_repo, p10 = metrics.precision_at_k(y, score, test["repo"].to_numpy(), k=10, seed=seed)
        lo, hi = metrics.cluster_bootstrap(per_repo, seed=seed)
        # the comparison point: each test repo's own base rate, averaged over repos
        base_p10 = float(test.groupby("repo")["is_slow"].mean().mean())
        res = {
            "scenario": scenario, "fold": fold, "model": "baseline_trailing90",
            "features": "trailing_90d_slow_rate", "params": "alpha=5;threshold=0.5",
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "n_test_repos": int(test["repo"].nunique()),
            "precision_at_10": p10, "p10_ci_lo": lo, "p10_ci_hi": hi,
            "base_rate_p10": base_p10,
            "auc_pr": metrics.auc_pr(y, score), "base_rate": float(y.mean()),
        }
        if fold == 0:
            res["audit"] = replay_audit(rows, te, tr, tier1, label_primary, seed=seed)
        out.append(res)
        log.info("%s fold %d: P@10=%.3f [%.3f, %.3f] base=%.3f | AUC-PR=%.3f base=%.3f | %d repos",
                 scenario, fold, p10, lo, hi, base_p10, res["auc_pr"], res["base_rate"],
                 res["n_test_repos"])
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    kept = cohort_qc.kept_repos()
    frames = load.load_all(kept)
    streams = {k: frames[k] for k in labels.ALL_STREAMS}
    prs = splits.modelling_prs(frames["pr_tier2"])
    lab = labels.label(prs, labels.first_human_event(prs, streams, labels.DEFINITIONS[labels.PRIMARY]))
    rows = splits.prepare_rows(prs, lab, kept)
    tier1 = frames["pr_tier1"]

    results = evaluate("A", rows, [splits.scenario_a(rows)], tier1, lab)
    results += evaluate("B", rows, splits.scenario_b(rows), tier1, lab)
    for r in results:
        tracking.log({k: v for k, v in r.items() if k != "audit"} | {"notes": f"label={labels.PRIMARY}"})
    print(f"logged {len(results)} rows to {tracking.EXPERIMENTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

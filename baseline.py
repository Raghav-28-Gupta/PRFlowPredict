"""The baseline every model must beat: the repo's trailing-90-day slow rate,
thresholded at 0.5 (blueprint §2, model 1). No learning.

EXPECTED RESULT, STATED IN ADVANCE: Precision@10 ~= base rate within CI on both
scenarios, because the score is constant within a repo. If the baseline BEATS the
within-repo base rate, something is leaking in replay.py -- stop and audit.

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
    lab = labels.label(frames["pr_tier2"],
                       labels.first_human_event(frames["pr_tier2"], streams, labels.DEFINITIONS[labels.PRIMARY]))
    rows = splits.prepare_rows(frames["pr_tier2"], lab, kept)
    tier1 = frames["pr_tier1"]

    results = evaluate("A", rows, [splits.scenario_a(rows)], tier1, lab)
    results += evaluate("B", rows, splits.scenario_b(rows), tier1, lab)
    for r in results:
        tracking.log({**r, "notes": f"label={labels.PRIMARY}"})
    print(f"logged {len(results)} rows to {tracking.EXPERIMENTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

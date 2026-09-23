"""Blueprint §4's fairness check, finished.

Phase 4 measured that the model is more pessimistic about first-time contributors than
their outcomes warrant (+0.088 on A) but put no interval on it, so it could not be claimed
the gap is real. The quantity that actually bears on fairness is the DIFFERENCE between
the newcomer gap and the repeat-contributor gap: a model miscalibrated equally for
everyone is not a fairness problem, one that singles out newcomers is.

Intervals resample REPOS, matching every other interval in this project: the effective
sample size for a claim about review culture is the number of repos, not of PRs."""
from __future__ import annotations

import numpy as np
import pandas as pd

import metrics
import splits

SEED = splits.SEED
HOUR_BUCKETS = [(0, 6, "00-06"), (6, 12, "06-12"), (12, 18, "12-18"), (18, 24, "18-24")]


def _per_repo_gap(sub: pd.DataFrame) -> pd.Series:
    """mean(p_hat) - mean(is_slow) per repo. The bootstrap unit."""
    g = sub.groupby("repo")
    return g["p_hat"].mean() - g["is_slow"].mean().astype(float)


def _row(slice_name: str, level: str, sub: pd.DataFrame, seed: int) -> dict:
    per_repo = _per_repo_gap(sub)
    lo, hi = metrics.cluster_bootstrap(per_repo, seed=seed) if len(per_repo) else (np.nan, np.nan)
    return {
        "slice": slice_name, "level": level, "n": int(len(sub)),
        "n_repos": int(sub["repo"].nunique()),
        "actual_rate": float(sub["is_slow"].astype(float).mean()),
        "mean_p_hat": float(sub["p_hat"].mean()),
        "gap": float(per_repo.mean()), "gap_ci_lo": float(lo), "gap_ci_hi": float(hi),
    }


def slice_gaps(pred: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    first = pred["is_first_pr_here"].astype(bool)
    rows = [_row("is_first_pr_here", "first-time", pred[first], seed),
            _row("is_first_pr_here", "repeat", pred[~first], seed)]
    hour = pred["created_hour_utc"].astype(int)
    for lo, hi, name in HOUR_BUCKETS:
        rows.append(_row("hour_bucket", name, pred[(hour >= lo) & (hour < hi)], seed))
    return pd.DataFrame(rows)


def gap_difference(pred: pd.DataFrame, seed: int = SEED) -> dict:
    """(first-timer gap) - (repeat gap), with a cluster-bootstrap CI over repos.

    Paired by repo, so a repo that is simply hard to predict cancels out."""
    first = pred["is_first_pr_here"].astype(bool)
    a = _per_repo_gap(pred[first])
    b = _per_repo_gap(pred[~first])
    paired = (a - b).dropna()
    lo, hi = metrics.cluster_bootstrap(paired, seed=seed) if len(paired) else (np.nan, np.nan)
    return {
        "difference": float(paired.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
        "ci_excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
        "n_first_time": int(first.sum()), "n_repeat": int((~first).sum()),
    }

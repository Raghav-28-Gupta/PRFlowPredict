"""Evaluation. Precision@k is PER REPO (the product framing: rank this team's PRs), and
confidence intervals resample REPOS, because the effective sample size for a
cross-repo claim is the number of repos, not of PRs ([R1] correction to blueprint §4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

SEED = 20260912


def precision_at_k(y_true, score, repo, k: int = 10, seed: int = SEED) -> tuple[pd.Series, float]:
    """Per repo: top-k by score, ties broken by a seeded random permutation.

    Random tie-breaking matters: the trailing-rate baseline is CONSTANT within a repo,
    and breaking ties by row order would silently reward whatever order the rows
    happened to be in. With random ties its P@k is an honest random draw = base rate."""
    df = pd.DataFrame({
        "y": np.asarray(y_true, dtype=float),
        "s": np.asarray(score, dtype=float),
        "repo": np.asarray(repo),
    })
    df["tie"] = np.random.default_rng(seed).random(len(df))
    df = df.sort_values(["repo", "s", "tie"], ascending=[True, False, False])
    top = df.groupby("repo", sort=True).head(k)
    per_repo = top.groupby("repo")["y"].mean()
    return per_repo, float(per_repo.mean())


def auc_pr(y_true, score) -> float:
    return float(average_precision_score(np.asarray(y_true, dtype=int), np.asarray(score, dtype=float)))


def cluster_bootstrap(per_repo, n: int = 2000, seed: int = SEED, ci: float = 0.95) -> tuple[float, float]:
    v = np.asarray(per_repo, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n)])
    a = (1.0 - ci) / 2.0
    return float(np.quantile(means, a)), float(np.quantile(means, 1.0 - a))

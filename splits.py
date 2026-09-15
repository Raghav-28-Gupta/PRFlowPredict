"""Train/test splits. Both scenarios are always reported (blueprint §2); random
row-level splits are rejected because they leak time and repo identity.

Scenario A (known-project, primary): time cutoff within the same repos.
Scenario B (cold-start): leave-repos-out via GroupKFold on repo.

Training rows are capped per repo so that three repos holding ~10% of rows each do
not become the model's whole notion of review culture. The cap is a fraction of the
UNCAPPED training total, computed once, never iterated. Test rows are never capped."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

SEED = 20260912
CAP_FRAC = 0.05
WINDOW_START = pd.Timestamp("2024-01-01T00:00:00Z")
WINDOW_END = pd.Timestamp("2026-06-30T23:59:59Z")
CUTOFF_A = pd.Timestamp("2026-01-01T00:00:00Z")


def prepare_rows(prs_tier2: pd.DataFrame, label_primary: pd.DataFrame,
                 kept_repos: list[str]) -> pd.DataFrame:
    rows = prs_tier2.merge(
        label_primary[["pr_id", "is_slow", "first_event_at", "wait_h"]], on="pr_id", how="inner")
    rows = rows[
        rows["repo"].isin(kept_repos)
        & ~rows["author_is_bot"].fillna(False).astype(bool)
        & (rows["created_at"] >= WINDOW_START) & (rows["created_at"] <= WINDOW_END)
    ]
    return rows.sort_values(["created_at", "pr_id"]).reset_index(drop=True)


def cap_per_repo(rows: pd.DataFrame, idx: np.ndarray, cap_frac: float, seed: int) -> np.ndarray:
    sub = rows.loc[idx]
    cap = max(1, int(cap_frac * len(sub)))
    keep = []
    for _, g in sub.groupby("repo", sort=True):
        keep.extend(g.index if len(g) <= cap else g.sample(n=cap, random_state=seed).index)
    return np.sort(np.asarray(keep))


def scenario_a(rows: pd.DataFrame, cap_frac: float = CAP_FRAC, seed: int = SEED):
    train = rows.index[rows["created_at"] < CUTOFF_A].to_numpy()
    test = rows.index[rows["created_at"] >= CUTOFF_A].to_numpy()
    return cap_per_repo(rows, train, cap_frac, seed), test


def scenario_b(rows: pd.DataFrame, n_splits: int = 5, cap_frac: float = CAP_FRAC,
               seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    gkf = GroupKFold(n_splits=n_splits)
    out = []
    for tr, te in gkf.split(rows, groups=rows["repo"]):
        tr_idx, te_idx = rows.index[tr].to_numpy(), rows.index[te].to_numpy()
        out.append((cap_per_repo(rows, tr_idx, cap_frac, seed), te_idx))
    return out


def row_share(rows: pd.DataFrame) -> pd.Series:
    return (rows["repo"].value_counts(normalize=True)).sort_values(ascending=False)

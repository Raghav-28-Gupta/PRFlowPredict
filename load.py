"""Load parse.py output for one or many repos.

Missing tables come back as EMPTY DataFrames rather than KeyErrors: a repo with no
formal reviews has no reviews.parquet, and that is data, not an error."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROCESSED = Path(__file__).parent / "data" / "processed"

TABLES = ("pr_tier1", "pr_tier2", "reviews", "thread_comments", "issue_comments",
          "timeline", "commits", "repo_meta")


def repo_dir(repo: str) -> Path:
    return PROCESSED / repo.replace("/", "__")


def load_repo(repo: str) -> dict[str, pd.DataFrame]:
    d = repo_dir(repo)
    out = {}
    for t in TABLES:
        p = d / f"{t}.parquet"
        out[t] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    return out


def load_all(repos: list[str]) -> dict[str, pd.DataFrame]:
    parts: dict[str, list[pd.DataFrame]] = {t: [] for t in TABLES}
    for repo in repos:
        for t, df in load_repo(repo).items():
            if not df.empty:
                parts[t].append(df)
    return {t: (pd.concat(v, ignore_index=True) if v else pd.DataFrame())
            for t, v in parts.items()}

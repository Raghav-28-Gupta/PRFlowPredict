"""The demo's logic, kept free of Streamlit so it can be tested directly.

Everything the app shows is computed here, at runtime, from demo/data/prs.parquet (built by
build_demo_data.py). Imports pandas only: the deployed app installs demo/requirements.txt,
not the project's full requirements."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data" / "prs.parquet"
SCORES = {"A": "score_a", "B": "score_b"}
DRIVERS = {"A": "drivers_a", "B": "drivers_b"}
TOP_K = 3
FIRST_DAY, LAST_DAY, DEFAULT_DAY = date(2026, 1, 2), date(2026, 6, 30), date(2026, 4, 1)
EARLY_JANUARY = date(2026, 1, 14)      # until here, few scored PRs can have been waiting yet
PR_LINK = re.compile(r"github\.com/([^/\s]+/[^/\s]+)/pull/(\d+)")
PR_REF = re.compile(r"^([^/\s#]+/[^/\s#]+)#(\d+)$")


def load(path: Path | None = None) -> pd.DataFrame:
    # resolved from this file, not the working directory, so `streamlit run demo/app.py`
    # works from anywhere
    return pd.read_parquet(path or DATA)


def repos(prs: pd.DataFrame) -> list[str]:
    return sorted(prs["repo"].unique())


def moment(day: date) -> pd.Timestamp:
    """The chosen day at 00:00 UTC: a maintainer opening the dashboard at the start of the day."""
    return pd.Timestamp(day, tz="UTC")


def format_value(value, dtype: str, rate: bool = False) -> str:
    """Rates and shares to 2 decimals, counts as integers, booleans as yes/no. The one
    formatter: build_demo_data.py writes the drivers text with it, the app shows values with it."""
    if pd.isna(value):
        return "missing"
    if dtype == "bool":
        return "yes" if bool(value) else "no"
    if dtype == "str":
        return str(value)
    if rate:
        return f"{float(value):.2f}"
    return f"{int(round(float(value))):,}"


def awaiting_review(prs: pd.DataFrame, at: pd.Timestamp) -> pd.DataFrame:
    """PRs opened before `at`, still open, and not yet reviewed. A PR reviewed or closed at
    exactly `at` is no longer waiting."""
    still_open = prs["closed_at"].isna() | (prs["closed_at"] > at)
    unreviewed = prs["first_review_at"].isna() | (prs["first_review_at"] > at)
    return prs[(prs["created_at"] < at) & still_open & unreviewed]


def outcome(rows: pd.DataFrame) -> pd.Series:
    """What actually happened: when the first review came, or that the PR was closed without
    one (most never-reviewed PRs were), and whether the PR stalled."""
    def days(end: pd.Series) -> pd.Series:
        return (end - rows["created_at"]).dt.total_seconds() / 86400

    def text(reviewed: float, closed: float) -> str:
        if not pd.isna(reviewed):
            return f"reviewed after {reviewed:.1f} days"
        if not pd.isna(closed):
            return f"closed after {closed:.1f} days without a review"
        return "never reviewed"

    # built row by row: an empty selection must still give an (empty) column of strings
    return pd.Series([text(r, c) + (", stalled" if slow else "") for r, c, slow
                      in zip(days(rows["first_review_at"]), days(rows["closed_at"]), rows["is_slow"])],
                     index=rows.index, dtype=object)


def ranked(prs: pd.DataFrame, repo: str, at: pd.Timestamp, scenario: str) -> pd.DataFrame:
    """The triage list: one repo's PRs awaiting review at `at`, highest risk first. Ties go to
    the older PR, then the lower number, so the order is deterministic."""
    score = SCORES[scenario]
    rows = awaiting_review(prs[prs["repo"] == repo], at)
    rows = rows.sort_values([score, "created_at", "number"], ascending=[False, True, True],
                            kind="mergesort")
    return pd.DataFrame({
        "rank": range(1, len(rows) + 1),
        "risk": rows[score].to_numpy(),
        "url": rows["url"].to_numpy(),
        "title": rows["title"].to_numpy(),
        "days_waited": ((at - rows["created_at"]).dt.total_seconds() / 86400).to_numpy(),
        "drivers": rows[DRIVERS[scenario]].to_numpy(),
        "outcome": outcome(rows).to_numpy(),
        "stalled": rows["is_slow"].to_numpy(),
    })


def tally(listing: pd.DataFrame, k: int = TOP_K) -> tuple[int, int]:
    """(how many of the top k stalled, k), with k cut to the list's length."""
    top = listing.head(k)
    return int(top["stalled"].sum()), len(top)


def default_repo(prs: pd.DataFrame, at: pd.Timestamp) -> str:
    """The repo with the most PRs awaiting review at `at`; ties go alphabetically."""
    counts = awaiting_review(prs, at).groupby("repo").size()
    if counts.empty:
        return repos(prs)[0]
    return min(counts.index, key=lambda r: (-counts[r], r))


def parse_pr_ref(text: str, default_repo: str) -> tuple[str, int] | None:
    """A PR link, `owner/repo#N`, or a bare number (looked up in default_repo); else None."""
    text = text.strip()
    link = PR_LINK.search(text) or PR_REF.match(text)
    if link:
        return link.group(1), int(link.group(2))
    bare = text.lstrip("#")
    return (default_repo, int(bare)) if bare.isdigit() else None


def find_pr(prs: pd.DataFrame, repo: str, number: int) -> pd.DataFrame:
    """The replayed PR as a one-row frame (empty if it is not in the replay). GitHub treats
    owner/repo case-insensitively, so this does too."""
    return prs[(prs["repo"].str.lower() == repo.lower()) & (prs["number"] == number)]


def rank_in_repo(prs: pd.DataFrame, pr: pd.DataFrame, scenario: str) -> tuple[float, int]:
    """(share of the repo's other replayed PRs that scored strictly lower, how many others)."""
    score = SCORES[scenario]
    row = pr.iloc[0]
    others = prs.loc[(prs["repo"] == row["repo"]) & (prs["pr_id"] != row["pr_id"]), score]
    return (float((others < row[score]).mean()) if len(others) else 0.0), len(others)


def random_pr(prs: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    """One replayed PR at random, as a one-row frame."""
    return prs.sample(1, random_state=seed)


MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-!|>~<])")


def md_escape(text: str) -> str:
    """Backslash-escape Markdown's special characters, so a PR title renders exactly as typed."""
    return MD_SPECIAL.sub(r"\\\1", text)

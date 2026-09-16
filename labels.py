"""The target variable. This is the ONLY implementation of it in the project.

Phase 1 captured three review streams separately and deferred the question of what
counts as a "review" because the blueprint's rule ("first non-author, non-bot review OR
comment") was shown to count spam comments on anthropics/skills. Phase 2 answers it
with a pre-registered choice: D5 is primary, D3 is reported alongside.

Every definition is always computed, so the sensitivity table is a by-product."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Belt and braces on top of __typename == "Bot" and the "[bot]" suffix: GitHub Actions
# tokens and self-hosted bots often appear as ordinary Users. Extend, don't hide.
KNOWN_BOTS = frozenset({
    "dependabot", "renovate", "github-actions", "codecov", "coveralls", "sonarcloud",
    "netlify", "vercel", "pre-commit-ci", "allcontributors", "stale", "mergify",
    "copilot",
})

ALL_STREAMS = ("reviews", "thread_comments", "issue_comments")


@dataclass(frozen=True)
class Definition:
    name: str
    streams: tuple[str, ...]
    exclude_associations: frozenset[str] = frozenset()
    exclude_minimized: bool = False


DEFINITIONS: dict[str, Definition] = {
    "D1": Definition("D1", ("reviews",)),
    "D2": Definition("D2", ("reviews", "thread_comments")),
    "D3": Definition("D3", ALL_STREAMS),                       # blueprint rule
    "D4": Definition("D4", ALL_STREAMS, exclude_minimized=True),
    "D5": Definition("D5", ALL_STREAMS, exclude_associations=frozenset({"NONE"})),
}
PRIMARY = "D5"


def is_bot(login: pd.Series, typename: pd.Series) -> pd.Series:
    low = login.fillna("").astype(str).str.lower()
    stem = low.str.replace(r"\[bot\]$", "", regex=True)
    return (typename == "Bot") | low.str.endswith("[bot]") | stem.isin(KNOWN_BOTS)


def visible_at(stream: str, df: pd.DataFrame) -> pd.Series:
    """When the PR author could SEE the event.

    reviews: submitted_at -- the connection's sort key (verified), null for PENDING.
    comments: published_at, falling back to created_at."""
    if stream == "reviews":
        return df["submitted_at"]
    return df["published_at"].fillna(df["created_at"])


def eligible_events(prs: pd.DataFrame, streams: dict[str, pd.DataFrame],
                    definition: Definition) -> pd.DataFrame:
    parts = []
    for name in definition.streams:
        df = streams.get(name)
        if df is None or df.empty:
            continue
        parts.append(pd.DataFrame({
            "pr_id": df["pr_id"].to_numpy(),
            "visible_at": visible_at(name, df).to_numpy(),
            "login": df["author_login"].to_numpy(),
            "typename": df["author_typename"].to_numpy(),
            "association": df["author_association"].to_numpy(),
            "minimized": (df["is_minimized"].to_numpy()
                          if "is_minimized" in df.columns else False),
        }))
    if not parts:
        return pd.DataFrame(columns=["pr_id", "visible_at"])
    ev = pd.concat(parts, ignore_index=True)
    ev["visible_at"] = pd.to_datetime(ev["visible_at"], utc=True)

    ev = ev[ev["visible_at"].notna() & ev["login"].notna()]
    ev = ev[~is_bot(ev["login"], ev["typename"])]
    # A deleted PR author (null login) cannot be matched, so any commenter counts.
    pr_author = prs.set_index("pr_id")["author_login"]
    ev = ev[ev["login"] != ev["pr_id"].map(pr_author)]
    if definition.exclude_associations:
        ev = ev[~ev["association"].isin(definition.exclude_associations)]
    if definition.exclude_minimized:
        ev = ev[~(ev["minimized"] == True)]  # noqa: E712 -- null-safe
    return ev


def first_human_event(prs: pd.DataFrame, streams: dict[str, pd.DataFrame],
                      definition: Definition) -> pd.Series:
    ev = eligible_events(prs, streams, definition)
    if ev.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return ev.groupby("pr_id")["visible_at"].min()


def label(prs: pd.DataFrame, first_event: pd.Series,
          threshold_h: float = 168.0, censor_h: float = 720.0) -> pd.DataFrame:
    """Blueprint §1. The 30-day window is why censoring is a non-issue for the
    classifier: no qualifying event within 720h => is_slow with certainty (720 > 168)."""
    out = prs[["repo", "pr_id", "created_at"]].copy()
    out["first_event_at"] = pd.to_datetime(out["pr_id"].map(first_event), utc=True)
    out["wait_h"] = (out["first_event_at"] - out["created_at"]).dt.total_seconds() / 3600.0
    out["never_reviewed_30d"] = out["wait_h"].isna() | (out["wait_h"] > censor_h)
    out["is_slow"] = out["never_reviewed_30d"] | (out["wait_h"] > threshold_h)
    out["wait_h_censored"] = out["wait_h"].clip(upper=censor_h).fillna(censor_h)
    out["event_observed"] = ~out["never_reviewed_30d"]
    return out


def label_all(prs: pd.DataFrame, streams: dict[str, pd.DataFrame],
              **kw) -> dict[str, pd.DataFrame]:
    return {name: label(prs, first_human_event(prs, streams, d), **kw)
            for name, d in DEFINITIONS.items()}

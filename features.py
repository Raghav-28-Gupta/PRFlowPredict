"""Phase 3: the feature table. One row per modelling PR; every value knowable at
created_at.

Four independent groups composed by build():
  static_features   -- known at open from the PR object itself
  at_open_features  -- reverse-replayed from the current snapshot via timeline events
  replay_features   -- chronological replay over Tier 1 history (replay.History)
  repo_features     -- repo-level, transferable; snapshots of HEAD at collection

COLUMN_SPEC is the contract. docs/feature_dictionary.md is rendered from it, and a test
asserts the built table's columns equal it, so the two cannot drift. Phase 4 trains only
on feature_columns(); keys, labels and fidelity flags are never features."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import cohort_qc
import labels
import load
import replay
import splits

log = logging.getLogger("features")

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "features" / "features.parquet"
GATE_JSON = ROOT / "data" / "phase3_gate.json"
DICT_MD = ROOT / "docs" / "feature_dictionary.md"

SEED = 20260912
ALPHA = 5.0
REVIEW_REQUEST_GRACE = pd.Timedelta(seconds=60)
TIMELINE_CAP = 60            # queries.N_TIMELINE -- a PR with exactly this many rows may be truncated

KEYS = ["repo", "pr_id", "number", "created_at"]
LABEL_COLS = ["is_slow", "wait_h", "wait_h_censored", "event_observed", "never_reviewed_30d"]


def _c(group, status, dtype, derivation, nullable=False):
    return {"group": group, "status": status, "dtype": dtype, "nullable": nullable,
            "derivation": derivation}


COLUMN_SPEC: dict[str, dict] = {
    # --- keys ---------------------------------------------------------------
    "repo":       _c("key", "key", "str", "owner/name"),
    "pr_id":      _c("key", "key", "str", "GraphQL node id"),
    "number":     _c("key", "key", "int", "PR number"),
    "created_at": _c("key", "key", "datetime[UTC]", "PR open time; the prediction instant t"),
    # --- labels (Phase 2, D5) ----------------------------------------------
    "is_slow":            _c("label", "label", "bool", "never_reviewed_30d OR wait_h > 168"),
    "wait_h":             _c("label", "label", "float", "hours to first human review (D5)", nullable=True),
    "wait_h_censored":    _c("label", "label", "float", "min(wait_h, 720)"),
    "event_observed":     _c("label", "label", "bool", "reviewed within 720h"),
    "never_reviewed_30d": _c("label", "label", "bool", "no D5 event within 720h"),
    # --- static: known at open from the PR object ---------------------------
    "created_hour_utc":        _c("static", "static", "int", "created_at.hour"),
    "created_dayofweek":       _c("static", "static", "int", "created_at.dayofweek, Mon=0"),
    "is_weekend":              _c("static", "static", "bool", "dayofweek >= 5"),
    "is_cross_repository":     _c("static", "static", "bool", "PR from a fork"),
    "author_account_age_days": _c("static", "static", "float", "(created_at - author.createdAt).days; immutable", nullable=True),
    "body_len":                _c("static", "static", "int", "len(body); body is editable -> approximate"),
    "has_body":                _c("static", "static", "bool", "body_len > 0; approximate"),
    "body_edited":             _c("flag", "flag", "bool", "last_edited_at not null -- POST-OPEN info, never a feature"),
    # --- at-open reconstructions -------------------------------------------
    "is_draft_at_open":            _c("at_open", "reconstructed", "bool", "is_draft_current inverted once per post-open Ready/ConvertToDraft event (parity)"),
    "n_labels_at_open":            _c("at_open", "reconstructed", "int", "n_labels_current - post-open Labeled + post-open Unlabeled, floored at 0"),
    "title_len_at_open":           _c("at_open", "reconstructed", "int", "len(previous_title of earliest post-open RenamedTitleEvent, else title_current)"),
    "base_is_default":             _c("at_open", "reconstructed", "bool", "base_ref_at_open == repo default branch"),
    "reviewer_requested_at_open":  _c("at_open", "static", "bool", "any ReviewRequestedEvent within 60s of open"),
    "n_reviewers_requested_at_open": _c("at_open", "static", "int", "count of those"),
    "requested_team_at_open":      _c("at_open", "static", "bool", "any of those with requested_reviewer_type == Team"),
    "additions_at_open":           _c("at_open", "reconstructed", "float", "sum of commit additions with authored_date <= created_at (parse.py)", nullable=True),
    "deletions_at_open":           _c("at_open", "reconstructed", "float", "as above", nullable=True),
    "n_commits_at_open":           _c("at_open", "reconstructed", "float", "as above", nullable=True),
    "diff_is_exact":               _c("flag", "flag", "bool", "single-commit PR: final diff == at-open diff"),
    "timeline_may_be_truncated":   _c("flag", "flag", "bool", "PR has exactly 60 timeline rows (the first:60 cap)"),
    # --- replay: chronological, created_at < t, proven by brute-force audit --
    "open_backlog_at_t":           _c("replay", "replay", "int", "prior PRs still open at t"),
    "prs_opened_trailing_7d":      _c("replay", "replay", "int", "prior PRs with created_at >= t-7d"),
    "trailing_90d_slow_rate":      _c("replay", "replay", "float", "shrunk D5 slow rate over resolvable prior PRs in [t-90d, t)"),
    "trailing_n":                  _c("replay", "replay", "int", "rows behind trailing_90d_slow_rate"),
    "trailing_window_complete":    _c("flag", "flag", "bool", "t-90d >= window start"),
    "is_first_pr_here":            _c("replay", "replay", "bool", "author has no prior PR in this repo"),
    "n_prior_prs_here":            _c("replay", "replay", "int", "author's prior PRs here"),
    "n_prior_merged_here":         _c("replay", "replay", "int", "of those, merged_at < t"),
    "prior_merge_rate_here":       _c("replay", "replay", "float", "shrunk toward global merge rate"),
    "days_since_first_pr_here":    _c("replay", "replay", "float", "t - author's first PR here", nullable=True),
    "author_prior_slow_rate_here": _c("replay", "replay", "float", "shrunk D5 slow rate over author's resolvable prior PRs (whole history)"),
    "author_prior_n":              _c("replay", "replay", "int", "rows behind author_prior_slow_rate_here"),
    # --- repo-level, transferable (snapshots of HEAD at collection) ---------
    "n_assignable_users":   _c("repo", "snapshot", "int", "maintainer capacity proxy"),
    "n_mentionable_users":  _c("repo", "snapshot", "int", "community size proxy"),
    "owner_is_org":         _c("repo", "snapshot", "bool", "owner_type == Organization"),
    "has_codeowners":       _c("repo", "snapshot", "bool", "CODEOWNERS present at HEAD"),
    "has_pr_template":      _c("repo", "snapshot", "bool", "PR template present at HEAD"),
    "has_contributing":     _c("repo", "snapshot", "bool", "CONTRIBUTING present at HEAD"),
    "n_ci_workflows":       _c("repo", "snapshot", "int", "files under .github/workflows at HEAD"),
    "language_dominant":    _c("repo", "snapshot", "str", "largest language by bytes"),
    "repo_age_days_at_open": _c("repo", "static", "float", "(created_at - repo.createdAt).days; point-in-time safe"),
}

TRAINABLE_STATUS = {"static", "reconstructed", "replay", "snapshot"}


def feature_columns() -> list[str]:
    return [c for c, m in COLUMN_SPEC.items() if m["status"] in TRAINABLE_STATUS]


# ---------------------------------------------------------------------------
# Group 1: static
# ---------------------------------------------------------------------------

def static_features(prs: pd.DataFrame) -> pd.DataFrame:
    p = prs.set_index("pr_id")
    created = p["created_at"]
    age = (created - pd.to_datetime(p["author_created_at"], utc=True)).dt.total_seconds() / 86400.0
    body = p["body_current"].fillna("").astype(str)
    return pd.DataFrame({
        "created_hour_utc": created.dt.hour.astype(int),
        "created_dayofweek": created.dt.dayofweek.astype(int),
        "is_weekend": created.dt.dayofweek >= 5,
        "is_cross_repository": (p["is_cross_repository"] == True),          # noqa: E712
        "author_account_age_days": age.astype(float),                         # NaN if deleted
        "body_len": body.str.len().astype(int),
        "has_body": body.str.len() > 0,
        "body_edited": p["last_edited_at"].notna(),
    }, index=p.index)


# ---------------------------------------------------------------------------
# Group 4: repo-level
# ---------------------------------------------------------------------------

def repo_features(repo_meta: pd.DataFrame, prs: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(prs["repo"]) - set(repo_meta["repo"]))
    if missing:
        raise KeyError(f"repo_meta has no row for {len(missing)} repo(s): {missing[:5]}")
    m = repo_meta.set_index("repo")
    p = prs.set_index("pr_id")
    joined = p[["repo", "created_at"]].join(m, on="repo", rsuffix="_repo")
    repo_created = pd.to_datetime(joined["created_at_repo"], utc=True)
    return pd.DataFrame({
        "n_assignable_users": joined["n_assignable_users"].astype(int),
        "n_mentionable_users": joined["n_mentionable_users"].astype(int),
        "owner_is_org": joined["owner_type"] == "Organization",
        "has_codeowners": joined["has_codeowners"].astype(bool),
        "has_pr_template": joined["has_pr_template"].astype(bool),
        "has_contributing": joined["has_contributing"].astype(bool),
        "n_ci_workflows": joined["n_ci_workflows"].astype(int),
        "language_dominant": joined["language_dominant"].astype(str),
        "repo_age_days_at_open": ((joined["created_at"] - repo_created).dt.total_seconds() / 86400.0).astype(float),
    }, index=p.index)

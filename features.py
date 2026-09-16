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
PHASE2_ROWS = ROOT / "data" / "phase2_rows.json"
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
# Group 2: at-open reconstruction (reverse replay from the current snapshot)
#
# The PR object holds CURRENT state. Timeline events are timestamped, so the state at
# open is recoverable by undoing every post-open event. Draft state flips on each
# Ready/ConvertToDraft event, so parity of post-open flips decides it; label count
# subtracts post-open adds and re-adds post-open removals; title and base use the
# previous_* value of the EARLIEST post-open change.
# ---------------------------------------------------------------------------

def at_open_features(prs: pd.DataFrame, timeline: pd.DataFrame,
                     repo_meta: pd.DataFrame) -> pd.DataFrame:
    p = prs.set_index("pr_id")
    idx = p.index

    if timeline.empty:
        tl = pd.DataFrame(columns=["pr_id", "event_type", "created_at", "previous_title",
                                   "previous_ref", "requested_reviewer_type", "pr_created_at"])
    else:
        tl = timeline.merge(p[["created_at"]].rename(columns={"created_at": "pr_created_at"}),
                            left_on="pr_id", right_index=True, how="inner")
    post = tl[tl["created_at"] > tl["pr_created_at"]]

    def count(frame, types):
        return frame[frame["event_type"].isin(types)].groupby("pr_id").size().reindex(idx, fill_value=0)

    def earliest_prev(types, col):
        f = post[post["event_type"].isin(types)].sort_values("created_at", kind="stable")
        f = f.drop_duplicates("pr_id", keep="first")            # literal earliest row, null or not
        return f.set_index("pr_id")[col].reindex(idx)

    flips = count(post, ["ReadyForReviewEvent", "ConvertToDraftEvent"])
    is_draft_current = (p["is_draft_current"] == True)                                   # noqa: E712
    is_draft_at_open = is_draft_current ^ (flips % 2 == 1)

    n_labels_at_open = (p["n_labels_current"].fillna(0).astype(int)
                        - count(post, ["LabeledEvent"]) + count(post, ["UnlabeledEvent"])).clip(lower=0)

    title_at_open = earliest_prev(["RenamedTitleEvent"], "previous_title").fillna(p["title_current"]).fillna("")
    base_at_open = earliest_prev(["BaseRefChangedEvent"], "previous_ref").fillna(p["base_ref_current"])

    missing = sorted(set(prs["repo"]) - set(repo_meta["repo"]))
    if missing:
        raise KeyError(f"repo_meta has no row for {len(missing)} repo(s): {missing[:5]}")
    default_branch = p["repo"].map(repo_meta.set_index("repo")["default_branch"])

    grace = tl["created_at"] <= tl["pr_created_at"] + REVIEW_REQUEST_GRACE
    rr = tl[(tl["event_type"] == "ReviewRequestedEvent") & grace]
    n_rr = rr.groupby("pr_id").size().reindex(idx, fill_value=0)
    team = rr[rr["requested_reviewer_type"] == "Team"].groupby("pr_id").size().reindex(idx, fill_value=0)

    n_rows = tl.groupby("pr_id").size().reindex(idx, fill_value=0)

    return pd.DataFrame({
        "is_draft_at_open": is_draft_at_open.astype(bool),
        "n_labels_at_open": n_labels_at_open.astype(int),
        "title_len_at_open": title_at_open.astype(str).str.len().astype(int),
        "base_is_default": (base_at_open == default_branch).astype(bool),
        "reviewer_requested_at_open": (n_rr > 0).astype(bool),
        "n_reviewers_requested_at_open": n_rr.astype(int),
        "requested_team_at_open": (team > 0).astype(bool),
        "additions_at_open": p["additions_at_open"].astype(float),
        "deletions_at_open": p["deletions_at_open"].astype(float),
        "n_commits_at_open": p["n_commits_at_open"].astype(float),
        "diff_is_exact": (p["diff_is_exact"] == True).astype(bool),                      # noqa: E712
        "timeline_may_be_truncated": (n_rows >= TIMELINE_CAP).astype(bool),
    }, index=idx)


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


# ---------------------------------------------------------------------------
# Group 3: replay (chronological, created_at < t; proven by the brute-force audit)
# ---------------------------------------------------------------------------

def replay_features(rows: pd.DataFrame, tier1: pd.DataFrame, label_d5: pd.DataFrame,
                    g: float, g_merge: float) -> pd.DataFrame:
    hist = {r: replay.History.from_frames(r, tier1, label_d5, splits.WINDOW_START)
            for r in rows["repo"].unique()}
    recs = []
    for repo, t, author in zip(rows["repo"], rows["created_at"], rows["author_login"]):
        a = None if (author is None or (isinstance(author, float) and np.isnan(author))) else author
        recs.append(hist[repo].features_at(t, g, g_merge, ALPHA, author=a))
    out = pd.DataFrame(recs, index=rows["pr_id"].to_numpy())
    out.index.name = "pr_id"
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def priors(rows: pd.DataFrame) -> tuple[float, float]:
    """Shrinkage priors from Scenario A TRAINING rows only (spec §6.3)."""
    train = rows[rows["created_at"] < splits.CUTOFF_A]
    if train.empty:
        train = rows
    g = float(train["is_slow"].mean())
    g_merge = float((train["merged_at"].notna() & (train["merged_at"] < splits.CUTOFF_A)).mean())
    return g, g_merge


def build(kept: list[str] | None = None, frames: dict | None = None) -> tuple[pd.DataFrame, dict]:
    kept = kept if kept is not None else cohort_qc.kept_repos()
    frames = frames if frames is not None else load.load_all(kept)

    prs = splits.modelling_prs(frames["pr_tier2"])
    prs = prs[prs["repo"].isin(kept)].reset_index(drop=True)
    streams = {k: frames[k] for k in labels.ALL_STREAMS}
    lab = labels.label(prs, labels.first_human_event(prs, streams, labels.DEFINITIONS[labels.PRIMARY]))
    rows = splits.prepare_rows(prs, lab, kept)
    g, g_merge = priors(rows)
    log.info("rows=%d repos=%d g=%.3f g_merge=%.3f", len(rows), rows["repo"].nunique(), g, g_merge)

    static = static_features(rows)
    at_open = at_open_features(rows, frames["timeline"], frames["repo_meta"])
    rep = replay_features(rows, frames["pr_tier1"], lab, g, g_merge)
    repo = repo_features(frames["repo_meta"], rows)

    lab_idx = lab.set_index("pr_id")[[c for c in LABEL_COLS if c != "is_slow"]]
    table = (rows.set_index("pr_id")[[k for k in KEYS if k != "pr_id"] + ["is_slow"]]
             .join(lab_idx).join(static).join(at_open).join(rep).join(repo)
             .reset_index())
    missing = [c for c in COLUMN_SPEC if c not in table.columns]
    extra = [c for c in table.columns if c not in COLUMN_SPEC]
    if missing or extra:
        raise RuntimeError(f"COLUMN_SPEC drift: missing={missing} extra={extra}")
    table = table[list(COLUMN_SPEC)]
    return table, {"rows": rows, "tier1": frames["pr_tier1"], "label_d5": lab, "g": g, "g_merge": g_merge}


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=ROOT, timeout=10).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def write_table(table: pd.DataFrame, path: Path = OUT) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    t = pa.Table.from_pandas(table, preserve_index=False)
    meta = dict(t.schema.metadata or {})
    meta.update({b"built_at": datetime.now(timezone.utc).isoformat().encode(),
                 b"git_sha": _git_sha().encode()})
    pq.write_table(t.replace_schema_metadata(meta), path)


# ---------------------------------------------------------------------------
# Phase 3 gate
# ---------------------------------------------------------------------------

NULLABLE = {c for c, m in COLUMN_SPEC.items() if m["nullable"]}
AUDIT_KEYS = tuple(k for k in replay.REPLAY_KEYS)   # every replay-derived column


def audit(table: pd.DataFrame, ctx: dict, n: int = 500, seed: int = SEED,
          expected_rows: int | None = None) -> list[dict]:
    rows, tier1, lab, g, g_merge = ctx["rows"], ctx["tier1"], ctx["label_d5"], ctx["g"], ctx["g_merge"]

    # 1. row conservation -- cross-phase when expected_rows (Phase 2's independent count)
    # is supplied, internal-only (can only catch a join bug, not Phase 2 vs 3 drift) otherwise
    exp = expected_rows if expected_rows is not None else len(rows)
    c1_check = ("row count == Phase 2 modelling rows (data/phase2_rows.json)" if expected_rows is not None
               else "row count == build() rows (INTERNAL ONLY -- phase2_rows.json absent)")
    c1 = {"id": 1, "check": c1_check, "value": [len(table), exp], "pass": len(table) == exp}

    # 2. brute-force audit of every replay column -- the leakage hard stop
    sample = table.sample(n=min(n, len(table)), random_state=seed)
    by_id = rows.set_index("pr_id")
    max_diff, worst = 0.0, None
    for _, r in sample.iterrows():
        src = by_id.loc[r["pr_id"]]
        a = src["author_login"]
        a = None if (a is None or (isinstance(a, float) and np.isnan(a))) else a
        b = replay.brute_force_features(tier1, lab, r["repo"], r["created_at"], g, ALPHA,
                                        global_merge_rate=g_merge, author=a)
        for k in AUDIT_KEYS:
            x, y = r[k], b[k]
            if isinstance(x, (bool, np.bool_)) or isinstance(y, (bool, np.bool_)):
                d = 0.0 if bool(x) == bool(y) else 1.0
            elif (isinstance(x, float) and np.isnan(x)) or (isinstance(y, float) and np.isnan(y)):
                d = 0.0 if (isinstance(x, float) and np.isnan(x) and isinstance(y, float) and np.isnan(y)) else 1.0
            else:
                d = abs(float(x) - float(y))
            if d > max_diff:
                max_diff, worst = d, (r["pr_id"], k, x, y)
    c2 = {"id": 2, "check": "replay audit: brute-force recomputation of every replay column matches (else replay LEAKS)",
          "value": {"n": int(len(sample)), "max_abs_diff": max_diff, "worst": worst}, "pass": max_diff < 1e-9}

    # 3. NaN only where documented
    bad = {c: int(table[c].isna().sum()) for c in table.columns if c not in NULLABLE and table[c].isna().any()}
    c3 = {"id": 3, "check": "no NaN outside documented-nullable columns", "value": bad, "pass": not bad}

    # 4. timeline truncation rate
    rate = float(table["timeline_may_be_truncated"].mean()) if len(table) else 0.0
    c4 = {"id": 4, "check": "timeline_may_be_truncated rate < 2%", "value": round(rate, 4), "pass": rate < 0.02}
    return [c1, c2, c3, c4]


def render_dictionary() -> str:
    """Render the feature dictionary from COLUMN_SPEC."""
    lines = [
        "# Feature Dictionary",
        "",
        "Generated by `python features.py --dictionary` from `features.COLUMN_SPEC`. Do not edit by hand.",
        "",
        "Written for: whoever trains on `data/features/features.parquet`.",
        "",
        "**Status** says how each column relates to the prediction instant `t = created_at`:",
        "",
        "| Status | Meaning |",
        "|---|---|",
        "| `key` | identifies the row; never a feature |",
        "| `label` | the target and its survival companions; never a feature |",
        "| `static` | known at open from the PR object; safe |",
        "| `reconstructed` | at-open value recovered from current state + timestamped events; an approximation, paired with a flag |",
        "| `replay` | computed by chronological replay over rows with `created_at < t`; proven by the brute-force audit (gate #2) |",
        "| `snapshot` | repo-level value as of collection (HEAD), applied to all of that repo's rows; accepted with that limitation |",
        "| `flag` | fidelity indicator for ablation and error analysis; **never a feature** (some encode post-open information) |",
        "",
        "Phase 4 trains only on `static`, `reconstructed`, `replay`, `snapshot` — see `features.feature_columns()`.",
        "",
        "Priors for shrinkage (`g` = D5 slow rate, `g_merge` = merge rate) are computed over Scenario A training rows",
        "(`created_at < 2026-01-01`) once at build time; a per-fold recomputation for Scenario B shifts an α=5 shrunk rate negligibly.",
        "",
        "| Column | Group | Status | Type | Nullable | Derivation |",
        "|---|---|---|---|---|---|",
    ]
    for c, m in COLUMN_SPEC.items():
        lines.append(f"| `{c}` | {m['group']} | {m['status']} | {m['dtype']} | "
                     f"{'yes' if m['nullable'] else 'no'} | {m['derivation']} |")
    return "\n".join(lines) + "\n"


def explain(pr_id: str, table: pd.DataFrame, ctx: dict, frames: dict) -> str:
    """Gate #5 support: print every feature of one PR with the rows/events behind it,
    so a human can check them against the GitHub UI."""
    rows, tier1, lab = ctx["rows"], ctx["tier1"], ctx["label_d5"]
    r = table.set_index("pr_id").loc[pr_id]
    src = rows.set_index("pr_id").loc[pr_id]
    t, repo, author = src["created_at"], src["repo"], src["author_login"]
    lines = [f"PR {pr_id}  {repo}#{int(r['number'])}  opened {t}  author={author!r}  is_slow={r['is_slow']}", ""]
    lines.append("== features ==")
    for c in feature_columns():
        lines.append(f"  {c:<32} {r[c]}")
    h = tier1[(tier1["repo"] == repo) & (tier1["created_at"] < t)].merge(
        lab[["pr_id", "first_event_at", "is_slow"]], on="pr_id", how="left")
    thr = t - pd.Timedelta(hours=168)

    def resolvable(x):
        return bool(x["created_at"] <= thr or (pd.notna(x["first_event_at"]) and x["first_event_at"] < t))

    lines += ["", f"== open backlog at t ({int(r['open_backlog_at_t'])}) == prior PRs open at {t}:"]
    for _, x in h[h["closed_at"].isna() | (h["closed_at"] > t)].iterrows():
        lines.append(f"  {x['pr_id']}  created {x['created_at']}  closed {x['closed_at']}")
    lines += ["", f"== prs opened in trailing 7d ({int(r['prs_opened_trailing_7d'])}) == "
                 f"prior PRs with created_at >= {t - pd.Timedelta(days=7)}:"]
    for _, x in h[h["created_at"] >= t - pd.Timedelta(days=7)].iterrows():
        lines.append(f"  {x['pr_id']}  created {x['created_at']}")
    lines += ["", f"== author history ({author!r}) == prior PRs by this author "
                 f"(counted toward author rate = {int(r['author_prior_n'])}):"]
    for _, x in h[h["author_login"] == author].iterrows():
        tag = "[counted]" if (pd.notna(x["is_slow"]) and resolvable(x)) else "[not yet resolvable]"
        lines.append(f"  {x['pr_id']}  created {x['created_at']}  merged {x['merged_at']}  "
                     f"first_event {x['first_event_at']}  is_slow {x['is_slow']}  {tag}")
    lo = t - pd.Timedelta(days=90)
    lines += ["", f"== trailing 90d window [{lo} .. {t}) == labelled prior PRs (counted = {int(r['trailing_n'])}):"]
    for _, x in h[(h["created_at"] >= lo) & h["is_slow"].notna()].iterrows():
        tag = "[counted]" if resolvable(x) else "[not yet resolvable]"
        lines.append(f"  {x['pr_id']}  created {x['created_at']}  first_event {x['first_event_at']}  is_slow {x['is_slow']}  {tag}")
    tl = frames["timeline"]
    ev = tl[tl["pr_id"] == pr_id].sort_values("created_at") if not tl.empty else tl
    lines += ["", f"== timeline events ({len(ev)}) =="]
    for _, e in ev.iterrows():
        extra = e.get("previous_title") or e.get("previous_ref") or e.get("requested_reviewer_type") or ""
        lines.append(f"  {e['created_at']}  {e['event_type']}  {extra}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--audit", action="store_true", help="run the Phase 3 gate (1-4) on the built table")
    ap.add_argument("--explain", metavar="PR_ID", help="print one PR's features with their evidence")
    ap.add_argument("--dictionary", action="store_true", help="render docs/feature_dictionary.md from COLUMN_SPEC")
    ap.add_argument("--n", type=int, default=500, help="audit sample size")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.dictionary:
        DICT_MD.parent.mkdir(parents=True, exist_ok=True)
        DICT_MD.write_text(render_dictionary(), encoding="utf-8")
        print(f"wrote {DICT_MD}")
        return 0

    kept = cohort_qc.kept_repos()
    frames = load.load_all(kept)
    table, ctx = build(kept, frames)

    if args.explain:
        print(explain(args.explain, table, ctx, frames))
        return 0
    if args.audit:
        expected = None
        if PHASE2_ROWS.exists():
            expected = int(json.loads(PHASE2_ROWS.read_text(encoding="utf-8"))["n_rows"])
        else:
            log.warning("no %s -- check #1 is internal-only (run eda_report.py first for a cross-phase check)",
                       PHASE2_ROWS)
        checks = audit(table, ctx, n=args.n, expected_rows=expected)
        GATE_JSON.parent.mkdir(parents=True, exist_ok=True)
        GATE_JSON.write_text(json.dumps(checks, indent=2, default=str), encoding="utf-8")
        for c in checks:
            print(f"  [{c['id']}] {'PASS' if c['pass'] else 'FAIL'}  {c['check']}  -> {c['value']}")
        ok = all(c["pass"] for c in checks)
        print(f"gate 1-4: {'PASS' if ok else 'FAIL'}  (gate 5 is the manual --explain spot-check)")
        return 0 if ok else 1

    write_table(table, Path(args.out))
    print(f"wrote {len(table):,} rows x {len(table.columns)} cols to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Stage 2: raw GraphQL pages -> Parquet. Pure offline; never touches the network.

This is the half of the two-stage design that makes a parse bug cheap. If this script
is wrong, rerun it. If collect.py were wrong, we would be re-scraping GitHub.

WHAT THIS DOES NOT DO
---------------------
It does not compute `is_slow`, decide what counts as a "review", or judge whether a
commenter is a bot or a spammer. Those are Phase 2 decisions, deliberately deferred
after the anthropics/skills finding that a spam account comments on otherwise
unreviewed PRs. This script emits the three review streams SEPARATELY and with enough
metadata (authorAssociation, __typename, isMinimized/minimizedReason, bodyText, and
every available timestamp) for Phase 2 to evaluate competing definitions on real data.

ASSERTIONS ARE PERMANENT, NOT ONE-OFF CHECKS
--------------------------------------------
GitHub does not document the ordering of the reviews / reviewThreads / comments
connections, and our "first:N captures the earliest N" reasoning depends entirely on
them being ascending. So the monotonicity check runs on every parse, forever. If
GitHub ever changes it, this fails loudly instead of silently producing wrong labels.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

import queries

log = logging.getLogger("parse")

RAW_ROOT = Path(__file__).parent / "data" / "raw"
PROCESSED_ROOT = Path(__file__).parent / "data" / "processed"

# Connection sizes actually requested, used for the truncation guard. Read from
# queries.py so the two can never drift apart.
CAPTURED = {
    "reviews": queries.N_REVIEWS,
    "comments": queries.N_ISSUE_COMMENTS,
    "reviewThreads": queries.N_REVIEW_THREADS,
    "thread_comments": queries.N_THREAD_COMMENTS,
    "commits": queries.N_COMMITS,
    "labels": queries.N_LABELS,
}


class ParseError(RuntimeError):
    pass


@dataclass
class Stats:
    """Conservation bookkeeping. Every number here must reconcile or the parse fails."""

    pages_read: int = 0
    pages_sha_ok: int = 0
    pages_sha_bad: list[str] = field(default_factory=list)
    pages_with_errors: list[str] = field(default_factory=list)
    rows_seen: int = 0
    unique_prs: int = 0
    duplicate_rows: int = 0
    monotonicity_violations: list[dict] = field(default_factory=list)
    truncated_prs: int = 0
    query_versions: set = field(default_factory=set)
    # PRs GitHub itself cannot serve (deleted, but still counted in totalCount).
    # Recorded by collect.py's bypass; surfaced here so a totalCount reconciliation
    # gap has a named cause instead of looking like silent data loss.
    dead_prs: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Raw page access
# ---------------------------------------------------------------------------

def read_manifest(repo_dir: Path) -> list[dict[str, Any]]:
    mpath = repo_dir / "manifest.jsonl"
    if not mpath.exists():
        raise ParseError(f"no manifest at {mpath}; has collect.py been run?")
    out = []
    with open(mpath, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A hard kill can tear the final line. Tolerate exactly that case:
                # the page it described is re-fetched on resume anyway.
                log.warning("manifest line %d is torn; ignoring", lineno)
    return out


def load_page(path: Path, expected_sha: str | None, stats: Stats) -> dict[str, Any] | None:
    """Read one gzipped page, verifying it against the manifest's hash.

    On Windows a partially written .gz is corrupt rather than short, so a hash
    mismatch or a decompression failure both mean "re-collect this page", not
    "continue with what we have".
    """
    if not path.exists():
        log.error("manifest references missing page %s", path)
        stats.pages_sha_bad.append(str(path))
        return None
    try:
        raw = gzip.open(path, "rb").read()
    except (OSError, EOFError) as exc:
        log.error("page %s is unreadable (%s); re-collect it", path, exc)
        stats.pages_sha_bad.append(str(path))
        return None

    if expected_sha:
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected_sha:
            log.error("sha256 mismatch on %s (manifest %s, actual %s)",
                      path, expected_sha[:12], actual[:12])
            stats.pages_sha_bad.append(str(path))
            return None
    stats.pages_sha_ok += 1
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------

def author_fields(node: dict | None, prefix: str = "author") -> dict[str, Any]:
    """Flatten an author object.

    `author` is legitimately null for deleted accounts -- the blueprint calls this out
    as "handle as a category, not a crash". A null author yields nulls plus
    `{prefix}_is_deleted=True`, which is itself a usable signal.

    `is_bot` is deliberately belt-and-braces: __typename alone misses GitHub Actions
    tokens and self-hosted bots that appear as ordinary Users with a `[bot]` suffix.
    Note this is a COLLECTION-time convenience flag, not the Phase 2 bot decision --
    the underlying __typename and login are both preserved so Phase 2 can override it.
    """
    if not node:
        return {
            f"{prefix}_login": None,
            f"{prefix}_typename": None,
            f"{prefix}_database_id": None,
            f"{prefix}_created_at": None,
            f"{prefix}_is_deleted": True,
            f"{prefix}_is_bot": None,
        }
    login = node.get("login")
    typename = node.get("__typename")
    return {
        f"{prefix}_login": login,
        f"{prefix}_typename": typename,
        f"{prefix}_database_id": node.get("databaseId"),
        f"{prefix}_created_at": node.get("createdAt"),
        f"{prefix}_is_deleted": False,
        f"{prefix}_is_bot": bool(
            typename == "Bot" or (login or "").endswith("[bot]")
        ),
    }


def _conn(node: dict, key: str) -> tuple[list[dict], int]:
    """Return (non-null nodes, totalCount) for a connection, tolerating nulls."""
    conn = node.get(key) or {}
    nodes = [n for n in (conn.get("nodes") or []) if n is not None]
    return nodes, int(conn.get("totalCount") or 0)


def check_monotonic(
    pr_id: str, stream: str, timestamps: list[str | None], stats: Stats
) -> None:
    """Assert a connection came back oldest-first.

    Our entire truncation argument is "first:N returns the EARLIEST N, which is what
    'first review' needs". That is an empirical observation about GitHub's API, not a
    documented guarantee. If it ever stops holding, every label computed downstream is
    wrong in a way nothing else would catch -- so this runs on every parse.
    """
    seen = [t for t in timestamps if t]
    for i in range(len(seen) - 1):
        if seen[i] > seen[i + 1]:
            stats.monotonicity_violations.append(
                {"pr_id": pr_id, "stream": stream, "index": i,
                 "prev": seen[i], "next": seen[i + 1]}
            )
            return


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

def tier1_row(node: dict, repo: str) -> dict[str, Any]:
    """Thin all-history row. Feeds backlog reconstruction and author history."""
    reviews_n, reviews_total = _conn(node, "reviews")
    _, comments_total = _conn(node, "comments")
    _, threads_total = _conn(node, "reviewThreads")
    _, commits_total = _conn(node, "commits")
    row = {
        "repo": repo,
        "pr_id": node.get("id"),
        "pr_database_id": node.get("databaseId"),
        "number": node.get("number"),
        "created_at": node.get("createdAt"),
        "published_at": node.get("publishedAt"),
        "closed_at": node.get("closedAt"),
        "merged_at": node.get("mergedAt"),
        "merged": node.get("merged"),
        "closed": node.get("closed"),
        "state": node.get("state"),
        "is_draft_current": node.get("isDraft"),
        "is_cross_repository": node.get("isCrossRepository"),
        "author_association_current": node.get("authorAssociation"),
        "n_reviews_total": reviews_total,
        "n_issue_comments_total": comments_total,
        "n_review_threads_total": threads_total,
        "n_commits_total": commits_total,
    }
    row.update(author_fields(node.get("author")))
    return row


def tier2_rows(node: dict, repo: str, stats: Stats) -> dict[str, Any]:
    """Rich in-window row plus its long-table children.

    Returns {"pr": {...}, "reviews": [...], "thread_comments": [...],
             "issue_comments": [...], "timeline": [...], "commits": [...]}
    """
    pr_id = node.get("id")
    reviews, reviews_total = _conn(node, "reviews")
    issue_comments, comments_total = _conn(node, "comments")
    threads, threads_total = _conn(node, "reviewThreads")
    commits, commits_total = _conn(node, "commits")
    timeline, _timeline_total = _conn(node, "timelineItems")
    labels, labels_total = _conn(node, "labels")

    # --- ordering assertions (see check_monotonic docstring) ---
    #
    # `reviews` sorts by submittedAt, NOT createdAt. Resolved empirically on
    # benbjohnson/litestream: a CHANGES_REQUESTED review drafted at 20:54 and
    # submitted 18h later sits AFTER an APPROVED review created at 21:24 -- ordered by
    # submission. createdAt is therefore legitimately non-monotonic in this stream
    # and must not be asserted. submittedAt is the sort key, which is also the
    # timestamp the label should use (the moment the author could see it), so
    # "first:N by submittedAt" is exactly "the earliest N visible reviews".
    check_monotonic(pr_id, "reviews_submitted",
                    [r.get("submittedAt") for r in reviews], stats)
    check_monotonic(pr_id, "issue_comments",
                    [c.get("createdAt") for c in issue_comments], stats)
    # Timeline events are documented-ish as chronological and verified so in probes;
    # assert it because Phase 3 reconstructs at-open state by replaying them in order.
    check_monotonic(pr_id, "timeline", [t.get("createdAt") for t in timeline], stats)

    # --- DEFINITION-FREE truncation guard ---
    # Deliberately NOT "all captured entries are author-or-bot", which would require
    # knowing what "bot" and "author" mean -- the Phase 2 decision being deferred.
    # A guard computed under a definition Phase 2 later changes would be useless and
    # un-rerunnable without re-scraping. Pure totalCount comparisons have no such
    # dependency. timelineItems is excluded: its totalCount ignores the itemTypes
    # filter (verified), so it cannot support a truncation guard in either direction.
    thread_comment_truncated = any(
        int((t.get("comments") or {}).get("totalCount") or 0) > CAPTURED["thread_comments"]
        for t in threads
    )
    truncated = (
        reviews_total > CAPTURED["reviews"]
        or comments_total > CAPTURED["comments"]
        or threads_total > CAPTURED["reviewThreads"]
        or commits_total > CAPTURED["commits"]
        or labels_total > CAPTURED["labels"]
        or thread_comment_truncated
    )
    if truncated:
        stats.truncated_prs += 1

    created_at = node.get("createdAt")

    # --- at-open diff reconstruction ---
    # PR-level additions/deletions/changedFiles reflect the FINAL head after every
    # post-open push: outcome contamination, since slow review gives a PR time to grow.
    #
    # Reconstruct from commits authored at or before the PR opened, using authoredDate
    # NOT committedDate. Verified: rebase/squash rewrites committedDate, so filtering on
    # it returns 0 additions for most multi-commit PRs, while authoredDate survives.
    # There is no "when did this commit enter the PR" timestamp anywhere in the API
    # (PullRequestCommit exposes only commit/id/pullRequest/resourcePath/url), so this
    # is the best available approximation -- and it IS an approximation.
    #
    # For single-commit PRs (the majority: 35/40 in a sample) the final diff IS the
    # at-open diff exactly, so no approximation is involved. diff_is_exact records
    # which case each row is, so Phase 3 can weight or filter on it.
    at_open = [
        c["commit"] for c in commits
        if c.get("commit") and c["commit"].get("authoredDate")
        and created_at and c["commit"]["authoredDate"] <= created_at
    ]
    # exact only if that single commit is actually visible at open
    diff_is_exact = commits_total == 1 and not truncated and bool(at_open)

    pr = {
        "repo": repo,
        "pr_id": pr_id,
        "pr_database_id": node.get("databaseId"),
        "number": node.get("number"),
        "url": node.get("url"),
        "created_at": created_at,
        "published_at": node.get("publishedAt"),
        "updated_at": node.get("updatedAt"),
        "closed_at": node.get("closedAt"),
        "merged_at": node.get("mergedAt"),
        "merged": node.get("merged"),
        "closed": node.get("closed"),
        "state": node.get("state"),

        # --- SNAPSHOT fields: current state, NOT state at created_at. ---
        # Named with a _current suffix so nothing downstream mistakes them for
        # point-in-time values. Reconstruct the at-open values from timeline events.
        "is_draft_current": node.get("isDraft"),
        "title_current": node.get("title"),
        "body_current": node.get("body"),
        "base_ref_current": node.get("baseRefName"),
        "labels_current": [l.get("name") for l in labels],
        "n_labels_current": labels_total,
        "author_association_current": node.get("authorAssociation"),
        "additions_final": node.get("additions"),
        "deletions_final": node.get("deletions"),
        "changed_files_final": node.get("changedFiles"),

        # --- at-open reconstruction ---
        # None means UNKNOWN (no commit authored at/before open), never "zero". A
        # pure-deletion PR legitimately has 0 additions at open. `x or None` was
        # collapsing those zeros to None -- found by the Phase 3 spot-check.
        "additions_at_open": sum(c.get("additions") or 0 for c in at_open) if at_open else None,
        "deletions_at_open": sum(c.get("deletions") or 0 for c in at_open) if at_open else None,
        "n_commits_at_open": len(at_open) if at_open else None,
        "diff_is_exact": diff_is_exact,

        # --- genuinely point-in-time safe ---
        "is_cross_repository": node.get("isCrossRepository"),
        "head_ref": node.get("headRefName"),
        "head_repo_owner": (node.get("headRepositoryOwner") or {}).get("login"),
        "maintainer_can_modify": node.get("maintainerCanModify"),
        "locked": node.get("locked"),
        "active_lock_reason": node.get("activeLockReason"),
        "milestone": (node.get("milestone") or {}).get("title"),
        "mergeable_current": node.get("mergeable"),
        "last_edited_at": node.get("lastEditedAt"),
        "editor_login": (node.get("editor") or {}).get("login"),

        # --- counts + truncation bookkeeping ---
        "n_reviews_total": reviews_total,
        "n_reviews_captured": len(reviews),
        "n_issue_comments_total": comments_total,
        "n_issue_comments_captured": len(issue_comments),
        "n_review_threads_total": threads_total,
        "n_review_threads_captured": len(threads),
        "n_commits_total": commits_total,
        "n_commits_captured": len(commits),
        "n_participants": (node.get("participants") or {}).get("totalCount"),
        "n_files": (node.get("files") or {}).get("totalCount"),
        "n_assignees": (node.get("assignees") or {}).get("totalCount"),
        "n_closing_issues": (node.get("closingIssuesReferences") or {}).get("totalCount"),
        "is_truncated": truncated,
    }
    pr.update(author_fields(node.get("author")))

    # --- long tables -------------------------------------------------------
    review_rows = []
    for r in reviews:
        row = {
            "repo": repo, "pr_id": pr_id, "review_id": r.get("id"),
            "state": r.get("state"),
            "created_at": r.get("createdAt"),
            # submittedAt is the time the author could SEE it. createdAt can be the
            # drafting time (verified: createdAt 16:55:22 vs submittedAt 17:07:48).
            # Phase 2 picks; both are preserved.
            "submitted_at": r.get("submittedAt"),
            "published_at": r.get("publishedAt"),
            "last_edited_at": r.get("lastEditedAt"),
            "author_association": r.get("authorAssociation"),
            "body_text": r.get("bodyText"),
            "n_comments": (r.get("comments") or {}).get("totalCount"),
        }
        row.update(author_fields(r.get("author")))
        review_rows.append(row)

    thread_comment_rows = []
    for t in threads:
        tc, tc_total = _conn(t, "comments")
        check_monotonic(pr_id, "thread_comments", [c.get("createdAt") for c in tc], stats)
        for c in tc:
            row = {
                "repo": repo, "pr_id": pr_id,
                "thread_id": t.get("id"),
                "thread_is_resolved": t.get("isResolved"),
                "thread_is_outdated": t.get("isOutdated"),
                "thread_resolved_by": (t.get("resolvedBy") or {}).get("login"),
                "thread_n_comments_total": tc_total,
                "comment_id": c.get("id"),
                "created_at": c.get("createdAt"),
                "drafted_at": c.get("draftedAt"),
                "published_at": c.get("publishedAt"),
                "last_edited_at": c.get("lastEditedAt"),
                "state": c.get("state"),
                "author_association": c.get("authorAssociation"),
                # The spam-hazard signal. Maintainers hide junk; this is the closest
                # thing to ground truth available for the 98zc5g5jyw-arch class.
                "is_minimized": c.get("isMinimized"),
                "minimized_reason": c.get("minimizedReason"),
                "path": c.get("path"),
                "original_line": c.get("originalLine"),
                "reply_to_id": (c.get("replyTo") or {}).get("id"),
                "body_text": c.get("bodyText"),
            }
            row.update(author_fields(c.get("author")))
            thread_comment_rows.append(row)

    issue_comment_rows = []
    for c in issue_comments:
        row = {
            "repo": repo, "pr_id": pr_id, "comment_id": c.get("id"),
            "created_at": c.get("createdAt"),
            "published_at": c.get("publishedAt"),
            "last_edited_at": c.get("lastEditedAt"),
            "author_association": c.get("authorAssociation"),
            "is_minimized": c.get("isMinimized"),
            "minimized_reason": c.get("minimizedReason"),
            "body_text": c.get("bodyText"),
        }
        row.update(author_fields(c.get("author")))
        issue_comment_rows.append(row)

    timeline_rows = []
    for t in timeline:
        rr = t.get("requestedReviewer") or {}
        timeline_rows.append({
            "repo": repo, "pr_id": pr_id,
            "event_type": t.get("__typename"),
            "created_at": t.get("createdAt"),
            "actor_login": (t.get("actor") or {}).get("login"),
            "label_name": (t.get("label") or {}).get("name"),
            "previous_title": t.get("previousTitle"),
            "current_title": t.get("currentTitle"),
            "previous_ref": t.get("previousRefName"),
            "current_ref": t.get("currentRefName"),
            "requested_reviewer_type": rr.get("__typename"),
            "requested_reviewer": rr.get("login") or rr.get("name"),
        })

    commit_rows = []
    for c in commits:
        cm = c.get("commit") or {}
        commit_rows.append({
            "repo": repo, "pr_id": pr_id,
            "oid": cm.get("oid"),
            # committedDate is rewritten by rebase/squash -- do NOT use it for
            # point-in-time filtering. authoredDate survives. Both kept so Phase 3 can
            # quantify the divergence rather than trust either blindly.
            "committed_date": cm.get("committedDate"),
            "authored_date": cm.get("authoredDate"),
            "additions": cm.get("additions"),
            "deletions": cm.get("deletions"),
            "changed_files": cm.get("changedFilesIfAvailable"),
        })

    return {
        "pr": pr,
        "reviews": review_rows,
        "thread_comments": thread_comment_rows,
        "issue_comments": issue_comment_rows,
        "timeline": timeline_rows,
        "commits": commit_rows,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def superseded_giveups(manifest: list[dict[str, Any]]) -> set[int]:
    """Indices of fatal_transient records that a later record resolved.

    A give-up is a real problem only if collection never got past that cursor. If a
    LATER line for the same tier (a successful page, or a bypass) started from the
    same cursor_before, the wedge was cleared and the give-up is stale history, not
    missing data. Observed live: three give-ups at one cursor, then a bypass from it.
    """
    resolved: dict[tuple[int, str], bool] = {}
    for rec in manifest:
        if rec.get("fatal_transient"):
            continue
        cb = rec.get("cursor_before")
        if cb is None:
            continue
        resolved[(rec.get("tier"), cb)] = True
    return {
        i for i, rec in enumerate(manifest)
        if rec.get("fatal_transient")
        and (rec.get("tier"), rec.get("cursor_before")) in resolved
    }


def parse_repo(repo_dir: Path, out_root: Path) -> Stats:
    stats = Stats()
    manifest = read_manifest(repo_dir)
    repo_name = None
    stale = superseded_giveups(manifest)

    tier1: dict[str, dict] = {}
    tier2: dict[str, dict] = {}
    long_tables: dict[str, list] = defaultdict(list)
    seen_pr_ids: dict[int, set] = {1: set(), 2: set()}

    for i, rec in enumerate(manifest):
        tier = rec.get("tier")
        repo_name = rec.get("repo") or repo_name
        if rec.get("fatal_transient"):
            if i not in stale:
                stats.pages_with_errors.append(
                    f"tier{tier} page{rec.get('page_index')} (transient give-up, unresolved)")
            continue
        if rec.get("bypass"):
            # No page file behind this line -- it documents PRs that were stepped
            # over because GitHub cannot serve them. Not an error, but not nothing.
            stats.dead_prs.extend(rec.get("dead_numbers") or [])
            continue
        if tier == 0:
            continue  # repo metadata handled separately
        if rec.get("has_graphql_errors"):
            # collect.py refuses to advance past these, so their presence means the
            # corpus is incomplete. Surfaced loudly rather than parsed around.
            stats.pages_with_errors.append(
                f"tier{tier} page{rec.get('page_index')}")
            continue

        qv = rec.get("query_version")
        if qv:
            stats.query_versions.add(qv)

        path = repo_dir / f"tier{tier}" / f"page_{int(rec['page_index']):04d}.json.gz"
        body = load_page(path, rec.get("sha256"), stats)
        if body is None:
            continue
        stats.pages_read += 1

        nodes = [
            n for n in
            (((body.get("data") or {}).get("repository") or {})
             .get("pullRequests") or {}).get("nodes") or []
            if n is not None
        ]
        for node in nodes:
            pr_id = node.get("id")
            if not pr_id:
                continue
            stats.rows_seen += 1
            # Dedupe by PR id. Resume re-fetches a page when a crash lands between the
            # manifest append and the checkpoint write, so duplicates are EXPECTED,
            # not a bug. Verified live: a resumed run produced two identical pages.
            if pr_id in seen_pr_ids[tier]:
                stats.duplicate_rows += 1
                continue
            seen_pr_ids[tier].add(pr_id)

            if tier == 1:
                tier1[pr_id] = tier1_row(node, repo_name)
            else:
                built = tier2_rows(node, repo_name, stats)
                tier2[pr_id] = built["pr"]
                for key in ("reviews", "thread_comments", "issue_comments",
                            "timeline", "commits"):
                    long_tables[key].extend(built[key])

    stats.unique_prs = len(tier1) + len(tier2)

    # --- write -------------------------------------------------------------
    out_dir = out_root / repo_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    def dump(name: str, rows: Iterable[dict]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        df = pd.DataFrame(rows)
        for col in df.columns:
            if col.endswith(("_at", "_date")):
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        return len(df)

    written = {
        "pr_tier1": dump("pr_tier1", tier1.values()),
        "pr_tier2": dump("pr_tier2", tier2.values()),
        **{k: dump(k, v) for k, v in long_tables.items()},
    }

    # repo metadata, flattened
    meta_path = repo_dir / "repo_meta.json.gz"
    if meta_path.exists():
        meta = json.loads(gzip.open(meta_path, "rb").read())
        r = (meta.get("data") or {}).get("repository") or {}
        if r:
            langs = {e["node"]["name"]: e["size"]
                     for e in ((r.get("languages") or {}).get("edges") or [])}
            wf = r.get("workflows") or {}
            dump("repo_meta", [{
                "repo": r.get("nameWithOwner"),
                "repo_id": r.get("id"),
                "created_at": r.get("createdAt"),
                "pushed_at": r.get("pushedAt"),
                # SNAPSHOT as of collection. Never a feature -- see data dictionary.
                "stargazer_count_snapshot": r.get("stargazerCount"),
                "fork_count_snapshot": r.get("forkCount"),
                "is_fork": r.get("isFork"),
                "is_archived": r.get("isArchived"),
                "is_mirror": r.get("isMirror"),
                "disk_usage": r.get("diskUsage"),
                "owner_type": (r.get("owner") or {}).get("__typename"),
                "owner_login": (r.get("owner") or {}).get("login"),
                "primary_language": (r.get("primaryLanguage") or {}).get("name"),
                "languages": json.dumps(langs),
                "language_dominant": max(langs, key=langs.get) if langs else None,
                "license": (r.get("licenseInfo") or {}).get("key"),
                "default_branch": (r.get("defaultBranchRef") or {}).get("name"),
                # Maintainer-capacity proxy: one of the few features that genuinely
                # transfers to an unseen repo, which Scenario B needs.
                "n_assignable_users": (r.get("assignableUsers") or {}).get("totalCount"),
                "n_mentionable_users": (r.get("mentionableUsers") or {}).get("totalCount"),
                "n_pull_requests": (r.get("pullRequests") or {}).get("totalCount"),
                "n_issues": (r.get("issues") or {}).get("totalCount"),
                "has_codeowners": any(
                    r.get(k) for k in
                    ("codeowners_github", "codeowners_root", "codeowners_docs")),
                "has_pr_template": any(
                    r.get(k) for k in ("pr_template_lower", "pr_template_upper")),
                "has_contributing": any(
                    r.get(k) for k in ("contributing_md", "contributing_rst")),
                "n_ci_workflows": len(wf.get("entries") or []) if wf else 0,
            }])

    log.info("wrote %s", {k: v for k, v in written.items() if v})
    return stats


def report(stats: Stats, strict: bool) -> int:
    print("\n" + "=" * 66)
    print("PARSE CONSERVATION REPORT")
    print("=" * 66)
    print(f"  pages read / sha-verified : {stats.pages_read} / {stats.pages_sha_ok}")
    print(f"  rows seen                 : {stats.rows_seen}")
    print(f"  unique PRs written        : {stats.unique_prs}")
    print(f"  duplicate rows dropped    : {stats.duplicate_rows}  (expected after a resume)")
    print(f"  PRs with truncated stream : {stats.truncated_prs}")
    print(f"  dead PRs stepped over     : {len(stats.dead_prs)}  "
          f"{sorted(set(stats.dead_prs)) if stats.dead_prs else ''}")
    print(f"  query versions present    : {sorted(stats.query_versions) or '-'}")

    failures = []
    if stats.pages_sha_bad:
        failures.append(f"{len(stats.pages_sha_bad)} page(s) failed sha256/readability")
        for p in stats.pages_sha_bad[:5]:
            print(f"    BAD PAGE: {p}")
    if stats.pages_with_errors:
        failures.append(f"{len(stats.pages_with_errors)} page(s) carried GraphQL errors")
        for p in stats.pages_with_errors[:5]:
            print(f"    ERROR PAGE: {p}")
    if stats.monotonicity_violations:
        failures.append(
            f"{len(stats.monotonicity_violations)} ordering violation(s) -- "
            "the first:N truncation argument no longer holds")
        for v in stats.monotonicity_violations[:5]:
            print(f"    ORDER: {v}")
    if len(stats.query_versions) > 1:
        failures.append(
            f"mixed query versions {sorted(stats.query_versions)} -- "
            "pages were collected under different shapes")

    if failures:
        print("\n  STATUS: FAIL")
        for f in failures:
            print(f"    - {f}")
        print("=" * 66)
        return 1 if strict else 0
    print("\n  STATUS: PASS")
    print("=" * 66)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", help="owner/name; omit to parse every collected repo")
    ap.add_argument("--out", default=str(PROCESSED_ROOT))
    ap.add_argument("--no-strict", action="store_true",
                    help="report problems but exit 0 anyway")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    if args.repo:
        dirs = [RAW_ROOT / args.repo.replace("/", "__")]
    else:
        dirs = sorted(d for d in RAW_ROOT.glob("*") if d.is_dir())

    if not dirs:
        log.error("nothing to parse under %s", RAW_ROOT)
        return 1

    rc = 0
    for d in dirs:
        if not d.exists():
            log.error("no raw data at %s", d)
            rc = 1
            continue
        log.info("parsing %s", d.name)
        stats = parse_repo(d, Path(args.out))
        rc |= report(stats, strict=not args.no_strict)
    return rc


if __name__ == "__main__":
    sys.exit(main())

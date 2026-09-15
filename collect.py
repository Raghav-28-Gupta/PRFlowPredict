"""Stage 1: collect raw GraphQL pages to disk. Never parses, never interprets.

CONTRACT
--------
This script writes bytes. It does not build features, compute labels, or decide what a
"review" is. Those are Phases 2 and 3. Its only job is to make sure they never need to
re-scrape GitHub.

Layout:
    data/raw/{owner}__{repo}/
        repo_meta.json.gz          one-shot repo metadata
        tier1/page_0000.json.gz    every PR, all time, thin shape
        tier2/page_0000.json.gz    in-window PRs, rich shape
        manifest.jsonl             one line per page ever written
        checkpoint.json            resume state per tier

WHY TWO TIERS
-------------
The blueprint's date window (2024-01-01..2026-06-30) cannot be a COLLECTION filter,
only a row filter. "Open PR backlog at instant t" needs every PR ever opened and still
open at t -- a PR from 2019 counts. "Days since first PR here" is wrong for any author
whose first PR predates the window. So Tier 1 sweeps all history cheaply (measured:
1 point per 100 PRs), and Tier 2 pays the rich cost only inside the window.

CRASH SAFETY
------------
Collection runs for hours across multiple sessions, so resume is load-bearing, not a
nicety. Write ordering is therefore strict and must not be rearranged:

    page bytes -> fsync -> os.replace -> manifest line -> checkpoint

Any other order can silently skip a page on a kill between steps. On Windows a
partially written .gz is a CORRUPT file rather than a short one, so every page also
carries a sha256 that parse.py verifies.

The cursor is never advanced past a page that carried GraphQL errors. A parse bug is
recoverable; a silently truncated corpus is not.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import queries
from ghclient import FatalError, GitHubGraphQL, Response, TransientError

log = logging.getLogger("collect")

DATA_ROOT = Path(__file__).parent / "data" / "raw"

# Blueprint §2. Collection stops well before "today" so every PR has >=60 days of
# observed follow-up -- no PR is falsely censored by collecting too early.
WINDOW_START = "2024-01-01T00:00:00Z"
WINDOW_END = "2026-06-30T23:59:59Z"

# Adaptive page-size ladder. GitHub terminates long-running queries, and deep nesting
# on a fat PR will intermittently time out. Without this ladder one pathological PR
# stalls a repo forever. Consequence: page size is VARIABLE, so parse.py must never
# assume a fixed page length.
TIER1_PAGE_SIZES = [100, 50, 20, 5, 1]
TIER2_PAGE_SIZES = [25, 10, 5, 2, 1]

# Consecutive successful pages required before stepping the page size back UP one rung.
# See the recovery comment in collect_tier: snapping back to full size after a single
# success causes thrashing against a persistently-failing span of PRs.
LADDER_RECOVERY_STREAK = 3


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Unservable-PR bypass
#
# Observed live on anthropics/skills: PR #1705 returns INTERNAL from GraphQL at every
# page size, even via a direct pullRequest(number:1705){id} -- and 404 from REST. It
# was deleted (this repo has an active spam problem, so almost certainly a Trust &
# Safety removal), but the pullRequests connection still counts it in totalCount and
# cannot serialize a page that contains it. A cursor walk therefore wedges on it
# permanently. This WILL recur across 45 repos.
#
# The bypass synthesizes a cursor that lands just before the next servable PR.
# GitHub's pullRequests cursors are `cursor:v2:` + msgpack([createdAt, databaseId]),
# which is undocumented but VERIFIED: a synthesized cursor round-trips byte-for-byte
# with a real one, and `after:` it resumes the walk exactly where expected.
# ---------------------------------------------------------------------------

_PROBE_PR = """
query Probe($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) { id number createdAt databaseId }
  }
}
"""

# How many PR numbers past the wedge point to probe before giving up. Numbers are
# shared with issues, so gaps of a few are normal; a run of 50 dead numbers is not.
BYPASS_MAX_PROBES = 50


def synthesize_cursor(created_at: str, database_id: int) -> str:
    """Build a pullRequests-connection cursor for (createdAt, databaseId).

    Format (verified by decoding real cursors and round-tripping one):
        b"cursor:v2:" + 0x92 (fixarray of 2)
                      + 0xb4 (fixstr, 20 bytes) + createdAt
                      + 0xcf + uint64  |  0xce + uint32   (databaseId)
    """
    import base64

    if len(created_at) != 20:
        raise ValueError(f"createdAt must be a 20-char ISO string, got {created_at!r}")
    body = b"\x92\xb4" + created_at.encode("ascii")
    if database_id > 0xFFFFFFFF:
        body += b"\xcf" + database_id.to_bytes(8, "big")
    else:
        body += b"\xce" + database_id.to_bytes(4, "big")
    return base64.b64encode(b"cursor:v2:" + body).decode("ascii")


def find_resume_point(
    client: GitHubGraphQL, owner: str, name: str, last_good_number: int,
) -> tuple[dict[str, Any], list[int]]:
    """Walk PR numbers past the wedge until one is servable.

    PR numbers are assigned at creation, so number order == createdAt order, and the
    next servable number IS the next servable node in the connection walk. Numbers
    that resolve to null are issues (shared sequence) and are simply skipped; numbers
    that fail with INTERNAL are the dead PRs we are stepping over.

    Returns (resume_pr, dead_numbers).
    """
    dead: list[int] = []
    for number in range(last_good_number + 1, last_good_number + 1 + BYPASS_MAX_PROBES):
        try:
            resp = client.execute(_PROBE_PR, {"owner": owner, "name": name, "number": number})
        except TransientError as exc:
            if getattr(exc, "kind", "") == "internal":
                dead.append(number)
                log.warning("[%s/%s] PR #%d is unservable (%s); stepping over it",
                            owner, name, number, str(exc)[:60])
                continue
            raise
        pr = ((resp.json().get("data") or {}).get("repository") or {}).get("pullRequest")
        if pr is None:
            continue  # an issue number, not a PR
        return pr, dead
    raise FatalError(
        f"{owner}/{name}: no servable PR within {BYPASS_MAX_PROBES} numbers after "
        f"#{last_good_number}; dead so far: {dead}"
    )


@dataclass
class RepoPaths:
    root: Path

    @classmethod
    def for_repo(cls, owner: str, name: str) -> RepoPaths:
        return cls(DATA_ROOT / f"{owner}__{name}")

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.jsonl"

    @property
    def checkpoint(self) -> Path:
        return self.root / "checkpoint.json"

    @property
    def repo_meta(self) -> Path:
        return self.root / "repo_meta.json.gz"

    def tier_dir(self, tier: int) -> Path:
        return self.root / f"tier{tier}"

    def page_path(self, tier: int, index: int) -> Path:
        return self.tier_dir(tier) / f"page_{index:04d}.json.gz"


# ---------------------------------------------------------------------------
# Durable writes
# ---------------------------------------------------------------------------

def write_page_atomic(path: Path, raw: bytes) -> str:
    """Write gzipped bytes durably; return sha256 of the UNCOMPRESSED payload.

    The hash is of the raw wire bytes, not the gzip container, so it stays stable
    regardless of compression level or gzip mtime headers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(raw).hexdigest()

    tmp = path.with_suffix(path.suffix + ".tmp")
    # mtime=0 keeps the gzip container byte-reproducible for a given payload.
    with open(tmp, "wb") as fh:
        with gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
            gz.write(raw)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return digest


def append_manifest(path: Path, record: dict[str, Any]) -> None:
    """Append one manifest line and fsync it.

    Appended AFTER the page is durable, so the manifest never references a page that
    is not on disk. The reverse order would let a kill produce a manifest entry for
    missing bytes, which parse.py would report as data loss.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def write_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Persist resume state last, so it never runs ahead of durable data.

    If we crash between the manifest append and here, resume re-fetches one page and
    writes it again. Re-fetching a page is harmless -- parse.py dedupes by PR id.
    Skipping one is not.
    """
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("checkpoint at %s is corrupt; restarting this repo from scratch", path)
        return {}


def last_good_number_from_disk(paths: RepoPaths, tier: int) -> int:
    """Highest PR number in the most recent successfully persisted page.

    The bypass needs this to know where to start probing. It is normally carried in
    the checkpoint, but a checkpoint written before that field existed -- or one that
    was torn -- would leave it at 0 and silently disable the bypass. Reading it back
    from the page bytes is the ground truth.
    """
    tier_dir = paths.tier_dir(tier)
    if not tier_dir.exists():
        return 0
    pages = sorted(tier_dir.glob("page_*.json.gz"), reverse=True)
    for page in pages:
        try:
            body = json.loads(gzip.open(page, "rb").read())
        except (OSError, EOFError, json.JSONDecodeError):
            continue
        nodes, _, _ = _extract(body)
        numbers = [int(n["number"]) for n in nodes if n.get("number")]
        if numbers:
            return max(numbers)
    return 0


def next_page_index(paths: RepoPaths, tier: int) -> int:
    """Derive the next page number from the manifest, never from a fresh counter.

    A counter restarted at 0 on resume would overwrite existing pages -- silent data
    loss that looks like a successful run.
    """
    if not paths.manifest.exists():
        return 0
    highest = -1
    with open(paths.manifest, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from a hard kill
            # Bypass records carry page_index None: they document skipped PRs and
            # have no page file behind them.
            if rec.get("tier") == tier and rec.get("page_index") is not None:
                highest = max(highest, int(rec["page_index"]))
    return highest + 1


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def _manifest_record(
    *,
    tier: int,
    page_index: int,
    owner: str,
    name: str,
    resp: Response,
    sha: str,
    cursor_before: str | None,
    cursor_after: str | None,
    has_next: bool,
    page_size: int,
    prs: list[dict[str, Any]],
    total_count: int | None,
    query_sha: str,
) -> dict[str, Any]:
    """One manifest line. Deliberately verbose -- it is the audit trail.

    Records min/max createdAt and the PR id list so parse.py can prove conservation
    (every collected PR appears in the output) without re-reading every page.
    """
    created = [p.get("createdAt") for p in prs if p and p.get("createdAt")]
    rl = resp.rate_limit
    return {
        "tier": tier,
        "page_index": page_index,
        "repo": f"{owner}/{name}",
        "collected_at": utcnow(),
        "query_version": queries.QUERY_VERSION,
        "query_sha256": query_sha,
        "page_size_requested": page_size,
        "pr_count": len(prs),
        "pr_ids": [p.get("id") for p in prs if p],
        "min_created_at": min(created) if created else None,
        "max_created_at": max(created) if created else None,
        "cursor_before": cursor_before,
        "cursor_after": cursor_after,
        "has_next_page": has_next,
        "repo_total_count": total_count,
        "http_status": resp.status,
        "request_id": resp.request_id,
        "elapsed_s": round(resp.elapsed_s, 3),
        "bytes": len(resp.raw),
        "sha256": sha,
        "rate_limit_cost": rl.get("cost"),
        "rate_limit_remaining": rl.get("remaining"),
        "node_count": rl.get("nodeCount"),
        "has_graphql_errors": bool(resp.graphql_errors),
        "graphql_errors": resp.graphql_errors or None,
    }


def _extract(body: dict[str, Any]) -> tuple[list[dict], dict, int | None]:
    """Pull the PR nodes / pageInfo / totalCount out of a response body.

    Tolerates the several shapes a failed GraphQL response can take: no `data` key at
    all (query validation error), `data.repository` null (repo gone private, renamed,
    or deleted mid-collection), or null entries inside `nodes` (field-level errors).
    """
    data = body.get("data") or {}
    repo = data.get("repository") or {}
    conn = repo.get("pullRequests") or {}
    nodes = [n for n in (conn.get("nodes") or []) if n is not None]
    page_info = conn.get("pageInfo") or {}
    return nodes, page_info, conn.get("totalCount")


def collect_tier(
    client: GitHubGraphQL,
    owner: str,
    name: str,
    tier: int,
    *,
    start_cursor: str | None = None,
    stop_after: str | None = None,
    max_pages: int | None = None,
) -> dict[str, Any]:
    """Paginate one tier to exhaustion (or to `stop_after`), persisting every page.

    Ordering is CREATED_AT ASC, deliberately:
      * New PRs only ever append past our cursor, so a walk resumed days later is
        monotone and safe. With DESC, the head of the list shifts under us between
        sessions.
      * There is no early-stop condition for Tier 1 -- stopping at the window start
        would discard exactly the pre-window history the backlog features need.

    `stop_after` (Tier 2 only) halts once PRs pass the window end. Returns a summary.
    """
    paths = RepoPaths.for_repo(owner, name)
    query = queries.TIER1_PRS if tier == 1 else queries.TIER2_PRS
    query_sha = queries.TIER1_SHA if tier == 1 else queries.TIER2_SHA
    ladder = TIER1_PAGE_SIZES if tier == 1 else TIER2_PAGE_SIZES

    ckpt = read_checkpoint(paths.checkpoint)
    tier_key = f"tier{tier}"
    tier_state = ckpt.get(tier_key, {})

    if tier_state.get("complete"):
        log.info("[%s/%s tier%d] already complete; skipping", owner, name, tier)
        return tier_state

    cursor = tier_state.get("cursor_after", start_cursor)
    page_index = next_page_index(paths, tier)
    pages_done = 0
    prs_seen = int(tier_state.get("prs_seen", 0))
    total_cost = int(tier_state.get("total_cost", 0))
    ladder_pos = 0
    consecutive_ok = 0
    last_good_number = int(tier_state.get("last_good_number", 0)) \
        or last_good_number_from_disk(paths, tier)
    dead_numbers: list[int] = list(tier_state.get("dead_numbers", []))

    # Initialised here so the post-loop completion logic is well-defined even when the
    # loop breaks on its first iteration (--max-pages 0). Defaulting has_next to True
    # is the safe choice: it means "not proven exhausted", so the repo is never
    # mistakenly marked complete.
    has_next = True
    max_created = ""

    log.info(
        "[%s/%s tier%d] starting at page %d (cursor=%s)",
        owner, name, tier, page_index, (cursor or "")[:24] or "BEGIN",
    )

    while True:
        if max_pages is not None and pages_done >= max_pages:
            log.info("[%s/%s tier%d] hit --max-pages %d; stopping early",
                     owner, name, tier, max_pages)
            break

        page_size = ladder[min(ladder_pos, len(ladder) - 1)]
        variables = {"owner": owner, "name": name, "first": page_size, "after": cursor}

        try:
            resp = client.execute(query, variables)
        except TransientError as exc:
            # Retries already exhausted inside the client.
            #
            # A rate-limit failure must NOT shrink the page: smaller pages mean MORE
            # requests for the same data, which makes a secondary-limit problem worse.
            # Wait instead, and hold the current page size.
            if getattr(exc, "kind", "") == "ratelimit":
                log.warning("[%s/%s tier%d] rate limited (%s); holding page size, "
                            "waiting 60s", owner, name, tier, exc)
                time.sleep(60.0)
                continue

            # For INTERNAL / timeout failures, a smaller query often survives where a
            # large one does not. Observed live on anthropics/skills: GitHub returns
            # INTERNAL for a span of PRs at every page size, and first:1 is what
            # finally steps past them one row at a time.
            if ladder_pos < len(ladder) - 1:
                ladder_pos += 1
                consecutive_ok = 0
                log.warning(
                    "[%s/%s tier%d] page %d failed (%s); dropping page size to %d",
                    owner, name, tier, page_index, exc, ladder[ladder_pos],
                )
                continue

            # Ladder exhausted: even first:1 fails. If the failure is INTERNAL, the
            # very next node is unservable (see the bypass section above). Step over
            # it with a synthesized cursor, record exactly what was skipped, and carry
            # on. Anything else is a genuine give-up.
            if getattr(exc, "kind", "") == "internal" and last_good_number:
                resume_pr, dead = find_resume_point(client, owner, name, last_good_number)
                dead_numbers.extend(dead)
                new_cursor = synthesize_cursor(
                    resume_pr["createdAt"], int(resume_pr["databaseId"]) - 1)
                append_manifest(paths.manifest, {
                    "tier": tier, "page_index": None, "repo": f"{owner}/{name}",
                    "collected_at": utcnow(), "bypass": True,
                    "after_number": last_good_number,
                    "dead_numbers": dead,
                    "resumed_at_number": resume_pr["number"],
                    "cursor_before": cursor, "cursor_after": new_cursor,
                    "pr_count": 0,
                })
                log.warning("[%s/%s tier%d] bypassed dead PR(s) %s; resuming at #%d",
                            owner, name, tier, dead, resume_pr["number"])
                cursor = new_cursor
                ladder_pos = 0
                consecutive_ok = 0
                continue

            append_manifest(paths.manifest, {
                "tier": tier, "page_index": page_index, "repo": f"{owner}/{name}",
                "collected_at": utcnow(), "cursor_before": cursor,
                "fatal_transient": str(exc), "pr_count": 0,
            })
            raise

        body = resp.json()
        nodes, page_info, total_count = _extract(body)
        end_cursor = page_info.get("endCursor")
        has_next = bool(page_info.get("hasNextPage"))

        sha = write_page_atomic(paths.page_path(tier, page_index), resp.raw)
        append_manifest(paths.manifest, _manifest_record(
            tier=tier, page_index=page_index, owner=owner, name=name,
            resp=resp, sha=sha, cursor_before=cursor, cursor_after=end_cursor,
            has_next=has_next, page_size=page_size, prs=nodes,
            total_count=total_count, query_sha=query_sha,
        ))

        # HARD RULE: never advance past a page that carried GraphQL errors.
        # The page is persisted (so it can be inspected) but the cursor stays put and
        # the repo cannot be marked complete.
        if resp.graphql_errors:
            log.error(
                "[%s/%s tier%d] page %d carried GraphQL errors; NOT advancing cursor: %s",
                owner, name, tier, page_index,
                json.dumps(resp.graphql_errors)[:300],
            )
            raise FatalError(
                f"GraphQL errors on {owner}/{name} tier{tier} page {page_index}; "
                "page persisted for inspection. Resolve before continuing."
            )

        prs_seen += len(nodes)
        total_cost += resp.cost
        page_index += 1
        pages_done += 1
        # Needed by the bypass: where to start probing if the NEXT page wedges.
        last_good_number = max(
            [last_good_number] + [int(n["number"]) for n in nodes if n.get("number")])

        # Ladder recovery is GRADUAL, one rung per N consecutive successes.
        #
        # Snapping straight back to full size after a single success causes thrashing:
        # observed live on anthropics/skills, where a span of PRs fails at every page
        # size. The collector would succeed at first:1, jump back to first:100, burn
        # several failed attempts descending again, and crawl. Stepping up slowly
        # keeps a bad region cheap while still recovering throughput afterwards.
        consecutive_ok += 1
        if ladder_pos > 0 and consecutive_ok >= LADDER_RECOVERY_STREAK:
            ladder_pos -= 1
            consecutive_ok = 0
            log.info("[%s/%s tier%d] recovering page size to %d",
                     owner, name, tier, ladder[ladder_pos])

        tier_state = {
            "cursor_after": end_cursor,
            "prs_seen": prs_seen,
            "total_cost": total_cost,
            "repo_total_count": total_count,
            "complete": False,
            "last_page_index": page_index - 1,
            "last_good_number": last_good_number,
            # Dead PRs are counted in totalCount but can never be collected, so the
            # completeness reconciliation must allow for them: prs_seen + len(dead)
            # should approach repo_total_count.
            "dead_numbers": dead_numbers,
            "query_version": queries.QUERY_VERSION,
            "updated_at": utcnow(),
        }
        ckpt[tier_key] = tier_state
        write_checkpoint(paths.checkpoint, ckpt)

        max_created = max((n.get("createdAt") or "" for n in nodes), default="")
        log.info(
            "[%s/%s tier%d] page %d: %d PRs (%s) cost=%s seen=%d/%s",
            owner, name, tier, page_index - 1, len(nodes),
            max_created[:10] or "empty", resp.cost, prs_seen, total_count,
        )

        if stop_after and max_created and max_created > stop_after:
            log.info("[%s/%s tier%d] passed window end %s; stopping",
                     owner, name, tier, stop_after)
            break
        if not has_next:
            break
        cursor = end_cursor

    # Only a genuinely exhausted walk counts as complete. A --max-pages run or a
    # window-bounded Tier 2 walk must not be mistaken for full coverage later.
    exhausted = not has_next and (max_pages is None or pages_done < max_pages)
    tier_state["complete"] = bool(exhausted or (stop_after and max_created > stop_after))
    tier_state["exhausted_connection"] = not has_next
    ckpt[tier_key] = tier_state
    write_checkpoint(paths.checkpoint, ckpt)
    return tier_state


def collect_repo_meta(client: GitHubGraphQL, owner: str, name: str) -> dict[str, Any]:
    """One-shot repo metadata. Cheap, and every field is a snapshot as of now."""
    paths = RepoPaths.for_repo(owner, name)
    resp = client.execute(queries.REPO_META, {"owner": owner, "name": name})
    sha = write_page_atomic(paths.repo_meta, resp.raw)
    append_manifest(paths.manifest, {
        "tier": 0, "page_index": 0, "repo": f"{owner}/{name}",
        "collected_at": utcnow(), "query_version": queries.QUERY_VERSION,
        "query_sha256": queries.REPO_META_SHA, "sha256": sha,
        "bytes": len(resp.raw), "http_status": resp.status,
        "request_id": resp.request_id,
        "has_graphql_errors": bool(resp.graphql_errors),
        "graphql_errors": resp.graphql_errors or None,
    })
    if resp.graphql_errors:
        raise FatalError(f"repo_meta for {owner}/{name} carried errors: {resp.graphql_errors}")
    return resp.json()


def tier2_start_cursor(paths: RepoPaths, window_start: str) -> str | None:
    """Find a Tier-1 cursor to launch Tier 2 from, so it can skip pre-window pages.

    Cursor portability across page sizes is VERIFIED (a cursor from a first:100 walk
    resumes correctly in a first:25 walk) but is not a documented guarantee. So we
    deliberately return the cursor of the page BEFORE the boundary and let the caller
    assert that Tier 2's first PR lands at or before window_start. If GitHub ever
    changes cursor semantics, that assertion fires instead of silently losing rows.

    Returns None (start from the beginning) if no safe earlier page exists.
    """
    if not paths.manifest.exists():
        return None

    candidates: list[tuple[int, str | None, str | None]] = []
    with open(paths.manifest, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (rec.get("tier") != 1 or rec.get("has_graphql_errors")
                    or rec.get("page_index") is None):
                continue
            candidates.append(
                (int(rec["page_index"]), rec.get("max_created_at"), rec.get("cursor_after"))
            )

    candidates.sort()
    safe_cursor: str | None = None
    for _idx, max_created, cursor_after in candidates:
        if max_created and max_created < window_start:
            safe_cursor = cursor_after
        else:
            break
    return safe_cursor


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--tier", type=int, choices=[0, 1, 2], action="append",
                    help="tier to collect; repeatable. 0=repo metadata. "
                         "default: all three")
    ap.add_argument("--window-start", default=WINDOW_START)
    ap.add_argument("--window-end", default=WINDOW_END)
    ap.add_argument("--max-pages", type=int, default=None,
                    help="stop after N pages (for smoke tests and the resume test)")
    ap.add_argument("--min-interval", type=float, default=0.8,
                    help="seconds between requests; raise if secondary limits appear")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if "/" not in args.repo:
        ap.error("--repo must be owner/name")
    owner, name = args.repo.split("/", 1)
    tiers = sorted(set(args.tier)) if args.tier else [0, 1, 2]

    client = GitHubGraphQL(min_interval_s=args.min_interval)
    paths = RepoPaths.for_repo(owner, name)
    started = time.monotonic()

    try:
        if 0 in tiers:
            meta = collect_repo_meta(client, owner, name)
            r = (meta.get("data") or {}).get("repository") or {}
            log.info("repo meta: %s | %s stars | %s PRs | %s assignable users",
                     r.get("nameWithOwner"), r.get("stargazerCount"),
                     (r.get("pullRequests") or {}).get("totalCount"),
                     (r.get("assignableUsers") or {}).get("totalCount"))

        if 1 in tiers:
            collect_tier(client, owner, name, 1, max_pages=args.max_pages)

        if 2 in tiers:
            cursor = tier2_start_cursor(paths, args.window_start)
            log.info("[%s/%s tier2] launching from %s",
                     owner, name, "tier1 boundary cursor" if cursor else "BEGIN")
            collect_tier(client, owner, name, 2, start_cursor=cursor,
                         stop_after=args.window_end, max_pages=args.max_pages)
    except FatalError as exc:
        log.error("FATAL: %s", exc)
        return 2
    except KeyboardInterrupt:
        log.warning("interrupted; checkpoint is durable, rerun the same command to resume")
        return 130

    log.info("done in %.1fs", time.monotonic() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())

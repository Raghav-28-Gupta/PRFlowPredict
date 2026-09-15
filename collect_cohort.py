"""Run collect.py over every repo in the cohort, sequentially and resumably.

Sequential on purpose: concurrency is what trips GitHub's secondary rate limit, and
the primary limit is not the constraint (measured ~4 hours for the whole cohort).

Safe to rerun at any time. Repos whose checkpoint says complete are skipped; a repo
that failed mid-way resumes from its cursor. A per-repo failure is logged and the
loop moves on, so one bad repo cannot stall the other 44 -- rerun afterwards to
retry it.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import collect
from ghclient import FatalError, GitHubGraphQL, TransientError

log = logging.getLogger("cohort")
COHORT = Path(__file__).parent / "data" / "cohort" / "cohort.json"


def is_done(owner: str, name: str) -> bool:
    ck = collect.read_checkpoint(collect.RepoPaths.for_repo(owner, name).checkpoint)
    return bool(ck.get("tier1", {}).get("complete") and ck.get("tier2", {}).get("complete"))


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    repos = [r["repo"] for r in json.loads(COHORT.read_text(encoding="utf-8"))["selected"]]
    client = GitHubGraphQL()
    started = time.monotonic()
    failed: list[str] = []

    for i, repo in enumerate(repos, 1):
        owner, name = repo.split("/", 1)
        if is_done(owner, name):
            log.info("[%2d/%d] %s already complete; skipping", i, len(repos), repo)
            continue
        log.info("[%2d/%d] %s", i, len(repos), repo)
        try:
            collect.collect_repo_meta(client, owner, name)
            collect.collect_tier(client, owner, name, 1)
            paths = collect.RepoPaths.for_repo(owner, name)
            cursor = collect.tier2_start_cursor(paths, collect.WINDOW_START)
            collect.collect_tier(client, owner, name, 2, start_cursor=cursor,
                                 stop_after=collect.WINDOW_END)
        except (FatalError, TransientError) as exc:
            log.error("[%2d/%d] %s FAILED: %s -- continuing", i, len(repos), repo, exc)
            failed.append(repo)
        except KeyboardInterrupt:
            log.warning("interrupted at %s; rerun to resume", repo)
            return 130

    elapsed = (time.monotonic() - started) / 3600
    log.info("cohort pass finished in %.1fh; %d/%d complete, failed: %s",
             elapsed, len(repos) - len(failed), len(repos), failed or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

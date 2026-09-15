"""Gate item #9 (HARD STOP): prove that an interrupted collection resumes correctly.

WHY THIS IS A HARD STOP
-----------------------
Full collection is ~5 hours across multiple sessions. If resume silently skips a page,
the result is a corpus that looks complete, passes every other check, and is missing
rows -- and since re-scraping is the thing this whole design exists to avoid, the only
remedy is the expensive one. A broken resume is strictly worse than a crash.

WHAT IT PROVES
--------------
Collect N pages in one uninterrupted run. Separately, collect a prefix, simulate a
kill, resume, and compare. The resulting set of PR ids must be IDENTICAL.

Duplicate pages are expected and fine -- a crash between the manifest append and the
checkpoint write causes one page to be re-fetched, and parse.py dedupes by PR id. What
must never happen is a MISSING PR.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import collect
from ghclient import GitHubGraphQL

TEST_REPO = "pallets/flask"
PAGES = 4
TIER = 1


def pr_ids_from_manifest(repo_dir: Path, tier: int) -> set[str]:
    """Union of PR ids across every successfully persisted page."""
    ids: set[str] = set()
    mpath = repo_dir / "manifest.jsonl"
    if not mpath.exists():
        return ids
    for line in open(mpath, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("tier") == tier and not rec.get("has_graphql_errors"):
            ids.update(rec.get("pr_ids") or [])
    return ids


def run(label: str, owner: str, name: str, max_pages: int) -> set[str]:
    client = GitHubGraphQL()
    collect.collect_tier(client, owner, name, TIER, max_pages=max_pages)
    repo_dir = collect.RepoPaths.for_repo(owner, name).root
    ids = pr_ids_from_manifest(repo_dir, TIER)
    print(f"  {label:<34} pages<={max_pages}  -> {len(ids)} unique PR ids")
    return ids


def main() -> int:
    owner, name = TEST_REPO.split("/")
    repo_dir = collect.RepoPaths.for_repo(owner, name).root
    backup = repo_dir.with_name(repo_dir.name + "__resume_backup")

    # Preserve any real collected data so the test never destroys it.
    restored = False
    if repo_dir.exists():
        if backup.exists():
            shutil.rmtree(backup)
        shutil.move(str(repo_dir), str(backup))
        restored = True

    try:
        print("=" * 70)
        print("GATE ITEM #9 -- RESUME IDEMPOTENCY")
        print("=" * 70)

        print("\nA. uninterrupted run")
        clean_ids = run("uninterrupted", owner, name, PAGES)

        print("\nB. interrupted run, then resume")
        shutil.rmtree(repo_dir)
        # NOTE: --max-pages counts pages in THIS invocation, not cumulatively. So to
        # land on the same total as the uninterrupted run, the resume leg asks for the
        # REMAINDER, not the full count. Getting this wrong makes the test report
        # spurious "extra" PRs and obscures the only assertion that matters.
        first_leg = PAGES // 2
        partial_ids = run("  partial (simulated kill)", owner, name, first_leg)
        resumed_ids = run("  after resume", owner, name, PAGES - first_leg)

        print("\n" + "-" * 70)
        missing = clean_ids - resumed_ids
        extra = resumed_ids - clean_ids
        print(f"  uninterrupted unique PRs : {len(clean_ids)}")
        print(f"  resumed unique PRs       : {len(resumed_ids)}")
        print(f"  partial had              : {len(partial_ids)}")
        print(f"  MISSING after resume     : {len(missing)}   <- must be 0")
        print(f"  extra after resume       : {len(extra)}")

        ok = not missing
        print("\n  STATUS:", "PASS" if ok else "FAIL -- resume loses data, DO NOT SCALE")
        print("=" * 70)
        return 0 if ok else 1
    finally:
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        if restored and backup.exists():
            shutil.move(str(backup), str(repo_dir))
            print("  (restored pre-existing collected data)")


if __name__ == "__main__":
    sys.exit(main())

"""Select the stratified repo cohort. Structural criteria only, reproducibly.

THE ONE RULE THIS FILE EXISTS TO ENFORCE
----------------------------------------
Repos are NEVER selected on review behaviour. Selecting on the outcome being predicted
would bias the entire study, and no amount of careful modelling downstream recovers
from it. Every filter here is structural: language, stars, activity, fork/archive
status, and raw PR volume.

The PR-volume floor deserves a note because it is the closest call. Volume is a
capacity/size property, not a review-latency property -- a repo with 400 PRs might
review all of them in an hour or none of them ever, and we do not look. It is,
however, mildly correlated with throughput, so it is recorded as a stated limitation
rather than waved through.

REPRODUCIBILITY
---------------
GitHub search results are time-dependent: the same query run next month returns a
different pool. A seeded sample is therefore only reproducible if the POOL is stored
too. This script persists every raw search response and the complete candidate list
alongside the winners, so the selection can be audited and re-derived later.

WHY SEEDED RANDOM WITHIN A CELL, NOT TOP-N BY STARS
---------------------------------------------------
Probing showed top-N sorting clusters every pick at the tier ceiling -- a "800-3000"
tier whose members all sit at 2,990 stars is not a distinct stratum from the tier
above it. It also admits awesome-lists and docs repos, which have PRs but no real code
review. Random-within-cell spreads the draw across the tier, and the volume floor
removes most of the non-code repos.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

from ghclient import GitHubGraphQL

log = logging.getLogger("select")

OUT_DIR = Path(__file__).parent / "data" / "cohort"

# Blueprint §2: 3 languages x 3 star tiers, ~3-4 repos per cell.
# Raised to 5 per cell (45 total) because re-scraping is forbidden by design, so
# spares are the cheapest insurance available: repos that turn out to be 90%
# Dependabot, mirrors with no real review, or a monorepo contributing 40% of all rows
# can be dropped in Phase 2 without a second collection run. It also lets Scenario B
# hold out ~10 repos instead of 5 -- the blueprint itself flags 5 as too small a
# sample for the cold-start claim.
LANGUAGES = ["Python", "TypeScript", "Go"]
STAR_TIERS = [(200, 800), (800, 3000), (3000, 15000)]
PER_CELL = 5

# Structural activity filter. NOTE (stated limitation): requiring recent activity is
# survivorship bias toward repos that are still alive. Repos that died during the
# window -- plausibly those with the worst review latency -- are structurally excluded,
# which biases the is_slow base rate downward. Relaxed to "pushed in the last year"
# rather than "last 3 months" to soften it.
PUSHED_SINCE = "2025-09-01"

# Minimum PRs inside the modelling window for a repo to be worth collecting.
MIN_WINDOW_PRS = 200
WINDOW_START = "2024-01-01"
WINDOW_END = "2026-06-30"

# Deliberately THIN.
#
# A richer version of this query (adding assignableUsers, and 100 nodes per page)
# fails with RESOURCE_LIMITS_EXCEEDED partway through a page -- GitHub truncates
# somewhere around node 60-90 and returns a partial result. That failure mode is
# especially dangerous here: the nodes that survive are always the ones that sort
# first, so quietly accepting a partial page would bias the candidate pool toward
# high-star repos and undermine the whole point of sampling within a cell.
#
# So: keep the search cheap enough to complete, and fetch richer per-repo metadata
# later via queries.REPO_META, which collect.py already runs for each selected repo.
# `languages` is kept because stratum assignment needs it at selection time.
SEARCH_REPOS = """
query SearchRepos($q: String!, $first: Int!, $after: String) {
  rateLimit { cost remaining }
  search(query: $q, type: REPOSITORY, first: $first, after: $after) {
    repositoryCount
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on Repository {
        id
        nameWithOwner
        createdAt
        pushedAt
        stargazerCount
        forkCount
        isFork
        isArchived
        isMirror
        isPrivate
        primaryLanguage { name }
        languages(first: 5) { edges { size node { name } } }
        pullRequests { totalCount }
      }
    }
  }
}
"""

# Page size for the search connection. 100 exceeds GitHub's per-query resource limit
# for this shape (see comment above); 50 completes reliably.
SEARCH_PAGE_SIZE = 50

COUNT_WINDOW_PRS = """
query CountWindowPRs($q: String!) {
  rateLimit { cost remaining }
  search(query: $q, type: ISSUE, first: 1) { issueCount }
}
"""


def cell_query(language: str, lo: int, hi: int) -> str:
    """Structural filters only. Nothing here touches review behaviour."""
    return (
        f"language:{language} stars:{lo}..{hi} pushed:>{PUSHED_SINCE} "
        f"fork:false archived:false is:public"
    )


def harvest_cell(
    client: GitHubGraphQL, language: str, lo: int, hi: int,
    raw_dir: Path, max_pages: int = 4,
) -> list[dict[str, Any]]:
    """Page through one cell's candidates, persisting every raw search response."""
    q = cell_query(language, lo, hi)
    cursor, candidates, page = None, [], 0
    while page < max_pages:
        resp = client.execute(
            SEARCH_REPOS, {"q": q, "first": SEARCH_PAGE_SIZE, "after": cursor})
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"search_{language}_{lo}_{hi}_p{page}.json").write_bytes(resp.raw)

        body = resp.json()
        if resp.graphql_errors:
            log.error("search errors for %s: %s", q, resp.graphql_errors[:2])
            break
        search = (body.get("data") or {}).get("search") or {}
        nodes = [n for n in (search.get("nodes") or []) if n]
        candidates.extend(nodes)
        log.info("  %s %d..%d page %d: +%d candidates (pool total %d)",
                 language, lo, hi, page, len(nodes), search.get("repositoryCount"))
        if not (search.get("pageInfo") or {}).get("hasNextPage"):
            break
        cursor = search["pageInfo"]["endCursor"]
        page += 1
    return candidates


def window_pr_count(client: GitHubGraphQL, repo: str) -> int:
    q = f"repo:{repo} is:pr created:{WINDOW_START}..{WINDOW_END}"
    resp = client.execute(COUNT_WINDOW_PRS, {"q": q})
    if resp.graphql_errors:
        return -1
    return int(((resp.json().get("data") or {}).get("search") or {}).get("issueCount") or 0)


def dominant_language(node: dict) -> str | None:
    """Stratum language from byte sizes, not primaryLanguage.

    primaryLanguage is noisy -- TypeScript projects frequently report as JavaScript --
    and a mis-assigned stratum quietly corrupts the language dimension of the design.
    """
    edges = (node.get("languages") or {}).get("edges") or []
    if not edges:
        return (node.get("primaryLanguage") or {}).get("name")
    return max(edges, key=lambda e: e.get("size") or 0)["node"]["name"]


def select(client: GitHubGraphQL, seed: int, per_cell: int, out_dir: Path,
           check_floor: bool = True) -> dict[str, Any]:
    raw_dir = out_dir / "raw_search"
    all_candidates: list[dict] = []
    chosen: list[dict] = []
    rejected: list[dict] = []

    for language in LANGUAGES:
        for lo, hi in STAR_TIERS:
            cell = f"{language}:{lo}-{hi}"
            log.info("cell %s", cell)
            candidates = harvest_cell(client, language, lo, hi, raw_dir)
            for c in candidates:
                c["_cell"] = cell
            all_candidates.extend(candidates)

            # Deterministic shuffle, then walk in order accepting the first `per_cell`
            # that clear the volume floor. Reproducible AND cheap: we only spend a
            # query on the floor check for repos we might actually take, instead of
            # pricing the entire pool.
            pool = [c for c in candidates
                    if not c.get("isFork") and not c.get("isArchived")
                    and not c.get("isMirror") and not c.get("isPrivate")]
            rng = random.Random(f"{seed}:{cell}")
            rng.shuffle(pool)

            taken = 0
            for cand in pool:
                if taken >= per_cell:
                    break
                repo = cand["nameWithOwner"]
                n_all = (cand.get("pullRequests") or {}).get("totalCount") or 0
                if n_all < MIN_WINDOW_PRS:
                    rejected.append({"repo": repo, "cell": cell,
                                     "reason": f"all-time PRs {n_all} < {MIN_WINDOW_PRS}"})
                    continue
                n_win = window_pr_count(client, repo) if check_floor else n_all
                if n_win < MIN_WINDOW_PRS:
                    rejected.append({"repo": repo, "cell": cell,
                                     "reason": f"in-window PRs {n_win} < {MIN_WINDOW_PRS}"})
                    continue
                chosen.append({
                    "repo": repo,
                    "cell": cell,
                    "language_stratum": language,
                    "language_dominant": dominant_language(cand),
                    "star_tier": f"{lo}-{hi}",
                    "stargazer_count_at_selection": cand.get("stargazerCount"),
                    "n_prs_all_time": n_all,
                    "n_prs_in_window": n_win,
                    "n_assignable_users": (cand.get("assignableUsers") or {}).get("totalCount"),
                    "created_at": cand.get("createdAt"),
                    "pushed_at": cand.get("pushedAt"),
                })
                taken += 1
                log.info("    + %-45s %6d stars  %5d window PRs",
                         repo, cand.get("stargazerCount"), n_win)
            if taken < per_cell:
                log.warning("  cell %s only filled %d/%d", cell, taken, per_cell)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": seed,
        "per_cell": per_cell,
        "languages": LANGUAGES,
        "star_tiers": STAR_TIERS,
        "pushed_since": PUSHED_SINCE,
        "min_window_prs": MIN_WINDOW_PRS,
        "window": [WINDOW_START, WINDOW_END],
        "n_candidates_pooled": len(all_candidates),
        "n_selected": len(chosen),
        "selected": chosen,
        "rejected": rejected,
        "limitations": [
            "Star tiers are defined by stars AT SELECTION TIME but PRs span 2024-2026; "
            "a repo at 3,100 stars today may have had 700 in 2024. GitHub exposes no "
            "historical star API, so this is unfixable. stargazerCount must never be "
            "used as a model feature.",
            "Requiring recent activity and excluding archived repos is survivorship "
            "bias toward projects still alive; repos that died during the window -- "
            "plausibly the slowest reviewers -- are structurally excluded, biasing the "
            "is_slow base rate downward.",
            "The minimum-PR-volume floor is a size criterion, not a review-behaviour "
            "criterion, but it is mildly correlated with review throughput.",
        ],
    }
    (out_dir / "cohort.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Full pool, so the seeded draw stays auditable after search results drift.
    (out_dir / "candidate_pool.json").write_text(
        json.dumps(all_candidates, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--per-cell", type=int, default=PER_CELL)
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--dry-run", action="store_true",
                    help="skip the per-repo in-window PR count (fast, approximate)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    client = GitHubGraphQL()
    m = select(client, args.seed, args.per_cell, Path(args.out),
               check_floor=not args.dry_run)

    print(f"\nselected {m['n_selected']} repos from a pool of {m['n_candidates_pooled']}")
    print(f"written to {args.out}/cohort.json")
    for row in m["selected"]:
        print(f"  {row['cell']:<22} {row['repo']:<45} {row['n_prs_in_window']:>6} window PRs")
    return 0


if __name__ == "__main__":
    sys.exit(main())

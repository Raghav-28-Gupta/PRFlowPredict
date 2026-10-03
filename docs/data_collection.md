> **Moved here from the project README during Phase 8.** It describes Phase 1, the data
> collection, as it was built; relative links were re-based for this folder. The whole
> project is summarised in the [README](../README.md) and the [report](REPORT.md).

# PRFlowPredict — Phase 1: dataset collection

Predicts, at PR-open time, whether a GitHub pull request will wait more than 168h
(7 days) for its first non-author, non-bot human review.

Full project spec: [actionable_ml_project_blueprint.md](../actionable_ml_project_blueprint.md).
**Read [docs/data_dictionary.md](data_dictionary.md) before building any feature** —
it records which columns are point-in-time safe and which only look it.

This directory currently implements **Phase 1 (collection) with the Phase 0 gate folded
in**. Labels (Phase 2) and features (Phase 3) are not built yet, deliberately.

---

## The one idea that shapes everything here

Phase 1's job is to capture enough raw signal that **Phases 2–8 never need to re-scrape
GitHub**. Every design choice follows from that:

- **Two stages.** `collect.py` writes raw response bytes and never interprets them.
  `parse.py` reads only from disk and never touches the network. A parse bug costs a
  re-parse; a collection bug would cost a re-scrape.
- **Two tiers.** The blueprint's date window is a **row filter, not a collection
  filter**. "Open PR backlog at instant *t*" needs every PR ever opened and still open
  at *t* — a 2019 PR counts. So Tier 1 sweeps all history cheaply (~1 point per 100
  PRs) and Tier 2 pays rich cost only inside the window.
- **The label is not decided yet.** Three review streams are captured separately so
  Phase 2 can compare competing definitions on real data. See "Why" below.

---

## Setup

```bash
pip install -r requirements.txt
gh auth login          # the collector borrows gh's token; needs only public_repo
```

## Running it

```bash
# 1. pick the cohort (structural criteria only, seeded, reproducible)
python select_repos.py --seed 20260912

# 2. collect one repo: tier 0 = metadata, 1 = all history (thin), 2 = in-window (rich)
python collect.py --repo anthropics/skills

# 3. raw -> Parquet, with conservation and ordering assertions
python parse.py --repo anthropics/skills

# 4. the Phase 0 gate
python gate_report.py --repo anthropics/skills

# hard-stop check: prove an interrupted run resumes without losing rows
python test_resume.py
```

Collection is resumable: rerun the same `collect.py` command after any interruption.
Progress lives in `data/raw/{owner}__{repo}/checkpoint.json`.

## Layout

```
data/raw/{owner}__{repo}/     # verbatim GraphQL bytes, gzipped. Never edit.
    manifest.jsonl            #   one line per page: sha256, cost, cursors, errors
    checkpoint.json           #   resume state
data/processed/{owner}__{repo}/*.parquet
data/cohort/                  # cohort.json + the full candidate pool + raw searches
```

`data/raw` is the expensive artifact. `data/processed` is disposable — regenerate it
with `parse.py` any time.

---

## Measured facts (not estimates)

| | |
|---|---|
| Tier 1 cost | 1 point / 100 PRs |
| Tier 2 cost | 400 points / 1,000 PRs → ~12,500 PRs/hour |
| Raw size | ~3.1 KB/PR uncompressed → ~0.16 GB for 50k PRs |
| Full 45-repo run | ~5 hours wall clock |

Rate limit (5,000 pts/hr) is **not** the binding constraint. Wall clock across multiple
sessions is — which is why resume correctness is a hard-stop gate item, not a nicety.

---

## Why the label is deferred to Phase 2

The blueprint defines the target as "first non-author, non-bot review **or comment**."
In `anthropics/skills`, PRs from `authorAssociation: NONE` receive zero reviews but do
receive comments from `98zc5g5jyw-arch`, an apparent spam account. That rule would mark
those PRs *reviewed* and corrupt the target on exactly the population the project is
about.

So Phase 1 captures `reviews`, `thread_comments`, and `issue_comments` as three
separate timestamped streams, each with `author_association`, `__typename`,
`is_minimized` / `minimized_reason`, `body_text`, and every available timestamp.
`gate_report.py` then computes `is_slow` under five competing definitions and reports
the spread. If it exceeds 10 points, the label is definition-dominated and Phase 2 owes
an explicit decision plus a sensitivity table in the write-up.

Related: inline review comments can exist with **no parent review** (verified: a PR with
`reviews.totalCount == 0` and 10 real `reviewThreads`), so `reviews` alone undercounts.

---

## Traps encoded in the code — read before editing `queries.py`

1. **Never add `orderBy` to the `comments` connection.** It accepts one
   (`IssueCommentOrder`, `UPDATED_AT` only) while `reviews` and `reviewThreads` do not.
   `UPDATED_AT` reorders on edit and would silently destroy "first comment" semantics.
   It looks like a harmless addition.
2. **`timelineItems.totalCount` ignores the `itemTypes` filter.** Any "totalCount > N
   implies truncated" guard on it is wrong in both directions.
3. **Use `authored_date`, not `committed_date`,** to reconstruct the at-open diff.
   Rebase and squash rewrite `committed_date`, which makes most multi-commit PRs
   reconstruct to 0 additions.
4. **`PULL_REQUEST_COMMIT` is excluded from `timelineItems`** — it swamps the `first:N`
   budget. Commits come from the separate `commits` connection.
5. **The search query is deliberately thin.** A richer one fails partway through a page
   with `RESOURCE_LIMITS_EXCEEDED`, and the surviving nodes are always the
   highest-sorted ones — silently biasing the candidate pool.
6. **Editing any query string is a re-collection event.** Bump `QUERY_VERSION`;
   `parse.py` fails loudly on mixed versions rather than merging incompatible shapes.

## Assertions that run forever, not once

GitHub does not document the ordering of the `reviews` / `reviewThreads` / `comments`
connections, yet the entire "`first:N` captures the earliest N" argument depends on
them being ascending. `parse.py` therefore re-checks monotonicity on **every** run. If
GitHub ever changes it, the parse fails loudly instead of producing quietly wrong
labels.

Likewise every page is verified against the sha256 recorded in the manifest, and
`collect.py` **never advances its cursor past a page carrying GraphQL errors** —
GraphQL returns HTTP 200 for partial failures, so a status-code-only check would
happily persist nulls and mark a repo complete.

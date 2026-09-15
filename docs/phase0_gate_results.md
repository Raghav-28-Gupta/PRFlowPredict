# Phase 0 Gate — Results

Written for: the go/no-go decision on scaling collection to 45 repos.

Run 2026-09-13 on two repos, as pre-registered:

| Repo | Role | Why |
|---|---|---|
| `benbjohnson/litestream` | **Drives go/no-go** | In-stratum (Go, 3k–15k stars), created 2020, 2 maintainers, 761 PRs |
| `anthropics/skills` | Adversarial case study | 176k stars, 12 months old, 877 open / 54 merged, active spam vector |

Numbers below are from `gate_report.py` on rows inside the modelling window
(2024-01-01 → 2026-06-30) only. Machine-readable copies: `data/processed/*/gate.json`.

---

## Verdict

**Pipeline: GO.** Every hard stop passes on both repos.

**Labels: GO, with one Phase 2 decision owed.** The blueprint's label rule is stable
on the in-stratum repo (3.7pp spread across five definitions) and unstable on the
adversarial one in a specific, now-quantified way. Phase 2 must pick a definition
before Phase 3; the data to do that is collected.

**Blueprint §2 feature plan: needs revision before Phase 3.** Three of its "point-in-
time correct" features are proven leaky or contaminated at rates high enough to matter.
Details below; the fixes are all reconstructions from data already collected.

---

## Hard stops

| # | Check | litestream | skills |
|---|---|---|---|
| 1 | Ordering monotonicity | PASS | PASS |
| 9 | Resume idempotency (`test_resume.py`) | PASS — 400 = 400, 0 missing, 0 extra | — |
| 10 | Raw→parse conservation | PASS — 1,236 unique, 0 sha failures | PASS — 2,204 unique, 0 sha failures |

**Item 1 resolved an open question.** The first run *failed* this check on litestream,
which is what it is for. Investigation: the `reviews` connection sorts by
`submittedAt`, not `createdAt`. A `CHANGES_REQUESTED` review drafted at 20:54 and
submitted 18h later sits after an `APPROVED` review created at 21:24. `submittedAt`
is perfectly monotonic, and is also the timestamp the label should use — so `first:N`
returns exactly "the earliest N reviews the author could see." The assertion now checks
the right column and passes.

---

## Measurements

### [2] Truncation — PASS on both

0.23% (litestream) / 0.31% (skills) of PRs have any stream exceeding the captured
`first:N`. Under the 1% threshold. Current sizes are frozen.

### [3] Label-definition sensitivity

| Definition | litestream `is_slow` | skills `is_slow` | skills median wait |
|---|---|---|---|
| D1 reviews only | 59.1% | 89.5% | 11.9h |
| D2 + inline review comments | 59.1% | 89.5% | 11.9h |
| D3 + issue comments *(blueprint)* | 55.4% | 87.4% | 13.2h |
| D4 D3 excl. minimized | 55.4% | 87.4% | 13.2h |
| D5 D3 excl. `NONE`-association | 57.0% | **95.8%** | **0.9h** |
| **spread** | **3.7pp** | **8.4pp** | |

**In-stratum: stable.** The choice barely moves the label on a normal repo.

**Adversarial: the spam finding, quantified.** Excluding non-affiliated commenters
drops the median wait from 13.2h to **0.9h** — roughly 8% of skills PRs get their
"first review" from a `NONE` account, and when a real maintainer engages they do it in
under an hour. The repo is reviewed-instantly-or-never, and D3 hides that.

`is_minimized` (D4) changed nothing on either repo: maintainers are not hiding the spam.
It is not a usable spam signal in practice; `author_association` on the *commenter* is.

**Phase 2 decision owed:** D3 vs D5, or a hybrid. Recommend D5 as the primary
definition with D3 reported as sensitivity — but decide on the 45-repo distribution,
not two repos.

### [4] Censoring / degeneracy

| | litestream | skills | blueprint expectation |
|---|---|---|---|
| never reviewed @30d | 47.6% | 85.8% | 10–35% |
| `is_slow` positive rate | 55.4% | 87.4% | usable 5–60% |

**Litestream fails the blueprint's 10–35% censoring expectation.** That expectation
was a guess, not a measurement; 47.6% for a two-maintainer repo is plausible, and the
positive rate (55.4%) is near-balanced and comfortably modelable. **Not a stop** — but
the 168h threshold was "chosen as a reasonable default, not derived," per the
blueprint's own open item, and should be re-examined on the full cohort's wait-time
distribution before Phase 4. Recommend leaving it at 168h until then.

Skills fails by design and is why it does not drive go/no-go.

### [5] Snapshot contamination — the blueprint's feature list

| Field | litestream | skills | consequence |
|---|---|---|---|
| `is_draft` (`ReadyForReviewEvent` after open) | **10.6%** | 1.3% | wrong at open for 1 in 10 |
| `label_count` (`LabeledEvent` after open) | **23.1%** | 0.0% | wrong at open for 1 in 4 |
| `title_length` (`RenamedTitleEvent`) | 5.8% | 4.4% | |
| `base_branch` (`BaseRefChangedEvent`) | 4.6% | 0.2% | |
| `additions`/`deletions` (multi-commit) | **47.6%** | **29.4%** | final ≠ at-open for 1 in 2 |

Both repos exceed the 20% line on the diff fields. **At-open reconstruction from
`timeline.parquet` and `commits.parquet` is mandatory, not optional.** The
reconstruction uses `authored_date` (rebase-safe); `diff_is_exact` marks the ~50–70% of
rows where no approximation was needed.

### [6] Warm-up necessity — full-history Tier 1 was required

| | litestream |
|---|---|
| PRs open on 2024-01-01 that were created before it | **9** |
| early-window PRs whose author has a pre-window PR | **33.3%** |

Collecting only the 2024–2026 window would have undercounted the opening backlog and
mislabelled a third of early-window authors as first-timers. This was argued during
planning; it is now measured. (Skills shows 0 on both because the repo did not exist
before the window.)

### [7] `authorAssociation` mutability — RESOLVED: LEAKY

On litestream, **35 of 88 authors (40%) read `CONTRIBUTOR` on their very first PR** in
the repo. `CONTRIBUTOR` requires a prior merged contribution, which is impossible on a
first PR. The field is computed at read time and rewritten by later merges — and
"later merged" is correlated with "got reviewed," so this is the outcome leaking into
a feature the blueprint lists as static.

Replacement: reconstruct `is_first_pr_here`, `n_prior_prs_here`,
`n_prior_merged_here`, `days_since_first_pr_here` from `pr_tier1.parquet` by
chronological replay. Those are the blueprint's own "author history" features — right
idea, wrong column. See `data_dictionary.md` §5.

### [8] Measured cost at the frozen query shape

| | |
|---|---|
| Tier 1 | 1 point / 100 PRs |
| Tier 2 | **400 points / 1,000 PRs** → 12,500 PRs/hour |
| Raw size | 3.7–6.2 KB/PR → ~0.3 GB for 50k PRs |
| Projected 45-repo run | **~5 hours** wall clock, well under the 24h line |

Rate limit is not the constraint. Multi-session resume is — and it passes.

---

## One thing the plan did not anticipate

**GitHub has PRs it cannot serve.** `anthropics/skills` PR #1705 returns `INTERNAL`
from GraphQL at every page size and on a direct fetch by number, and 404 from REST —
deleted (almost certainly a spam removal), but still counted in `totalCount` and
wedging any cursor walk that reaches it.

The collector now steps over these by probing forward to the next servable PR and
synthesising a cursor for it (`cursor:v2:` + msgpack `[createdAt, databaseId]`, verified
to round-trip byte-for-byte with GitHub's own). Dead PRs are recorded by number in the
manifest and checkpoint, and reconciliation is `collected + dead == totalCount` —
exact on skills: 1,224 + 1 = 1,225.

This will recur across 45 repos. It is handled.

---

## Blueprint amendments owed (deferred until now, by decision)

1. **§2 feature plan.** Mark `additions`/`deletions`/`changed_files`, `is_draft`,
   `label_count`, `title_length`, `base_branch` as SNAPSHOT with reconstruction
   required; mark `author_association` LEAKY; add `REVIEW_REQUESTED_EVENT`,
   `author_created_at`, and the repo-level transferable features.
2. **§3 Phase 1 row.** Collection is all-history two-tier, 45 repos, not windowed 30.
3. **§4 bootstrap.** "Bootstrap over PRs, stratified by repo" holds the repo set fixed;
   the Scenario B claim needs a cluster bootstrap resampling whole repos.
4. **§4 leakage rule.** "Only PRs strictly earlier than the row" is insufficient for
   trailing-window label features; the predicate is "PRs whose label was resolvable
   by *t*."

---

## The cohort, ready to collect

`select_repos.py --seed 20260912` drew **45 repos, 5 per cell, from a pool of 1,800**
(85 candidates rejected for volume during the draw). Full list, the rejected set, the
raw search responses, and the complete candidate pool are in `data/cohort/`.

| | |
|---|---|
| In-window PRs | **49,757** — inside the blueprint's 35–60k target |
| Projected Tier 2 cost | ~19,900 points → **~4 hours** at the rate limit |
| Gate repo | `benbjohnson/litestream` was drawn *by the selector* (Go, 3k–15k) — in-stratum by its own criteria, not hand-picked |

**Concentration to watch in Phase 2's subset-to-30.** Three repos each hold ~10–11%
of rows (`Scottcjn/Rustchain`, `yunionio/cloudpods`, `rilldata/rill`); the top five hold
45%. `Rustchain` at 5,262 in-window PRs for a 200–800-star repo is anomalous and worth
a look before it is allowed to be a tenth of the training set. This is exactly the
situation the 45→30 spare capacity exists for.

---

## Decision requested

Proceed to the 45-repo collection?

If yes: `python collect.py --repo <each>` over `data/cohort/cohort.json`, ~4–5 hours,
resumable, and the dead-PR bypass is in place for the ones like #1705.

If the D3/D5 label question should be settled first: it cannot be settled on two
repos — it needs the distribution across the cohort.

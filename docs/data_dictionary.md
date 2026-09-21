# PRFlowPredict — Phase 1 Data Dictionary

Written for: whoever builds Phases 2–3, which in ~8 weeks will be you with no memory of
these decisions.

This file exists to stop one specific failure: reaching for a column that looks
point-in-time correct and is not. Every field below is classified into exactly one of:

| Class | Meaning |
|---|---|
| **SAFE** | Known at `created_at`. Usable as a feature. |
| **SNAPSHOT** | Current state as of `collected_at`, NOT state at `created_at`. Reconstruct the at-open value, or don't use it. |
| **LEAKY** | Encodes the outcome, or information from after prediction time. Never a feature. |
| **RAW** | Kept so Phase 2 can decide something; not itself a feature. |

---

## 1. The three corrections to the blueprint

The blueprint's §2 feature plan describes several fields as "computable at `created_at`,
i.e. point-in-time correct." Live API probes during Phase 1 planning showed that is not
true for all of them. Recorded here; the blueprint itself gets amended after the gate
measures the magnitude.

### 1.1 Current-state fields presented as at-open values

| Blueprint feature | Reality | Recover from |
|---|---|---|
| `additions`, `deletions`, `changed_files` | Reflect the **final head** after every post-open push | `commits` + `authored_date`; see §1.2 |
| `is_draft` | Current. A PR opened as draft and marked ready reads `false` | `ReadyForReviewEvent` / `ConvertToDraftEvent` |
| `label_count` | Current | `LabeledEvent` / `UnlabeledEvent` |
| `title_length` | Titles are editable | `RenamedTitleEvent.previousTitle` |
| `body_length`, `has_body` | Bodies are editable | `last_edited_at` (edit *content* is not collected) |
| `base_branch` | Current | `BaseRefChangedEvent` |

The diff one is the most consequential, because it is not merely stale — it is
**outcome-contaminated**. Large PRs attract slow review, *and* slow review gives a PR
time to grow. Using `additions_final` would let the model see the consequence of the
thing it is predicting.

Every such column in `pr_tier2.parquet` carries a `_current` or `_final` suffix. Treat
the suffix as a warning label.

### 1.2 At-open diff reconstruction: use `authored_date`, never `committed_date`

Verified on `pallets/flask`: filtering commits by `committed_date <= pr.created_at`
returns **0 additions for most multi-commit PRs**, because rebase and squash rewrite
`committed_date`. `authored_date` survives rebases and reconstructs sanely.

Also verified: `PullRequestCommit` exposes only `{commit, id, pullRequest,
resourcePath, url}`. There is **no** "when did this commit enter the PR" timestamp
anywhere in the API, so `authored_date` is the best available approximation — and it
*is* an approximation.

Mitigating: in a 40-PR sample, 35 were single-commit, where the final diff **is** the
at-open diff exactly. The `diff_is_exact` column records which case each row is. Prefer
filtering or weighting on it over silently mixing exact and approximated values.

**Fixed 2026-09-21 (Phase 3 spot-check):** `parse.py` used `sum(...) or None`, which
collapsed a legitimate 0 to None — every pure-deletion PR read `additions_at_open = NaN`
and every pure-addition PR read `deletions_at_open = NaN`. Now None means only "no commit
authored at or before open" (1,326 rows), and `diff_is_exact` additionally requires that
commit to be visible at open. Re-parsed from raw; no re-scrape.

### 1.3 A point-in-time bug the blueprint's own leakage rule does not catch

Blueprint §4 says repo-state features may use "only PRs strictly earlier than the row."
That is **insufficient** for the trailing-90-day slow-rate.

A PR created at `t − 3 days` has no *knowable* label at time `t` — its label is not
determined until 168h after its creation. Including it peeks 7 days into the future.

The correct filter is "PRs whose label was resolvable by `t`":

```
created_at <= t - 168h   OR   first_human_event_at < t
```

The second clause matters: fast PRs resolve early and are legitimately usable. This
needs no extra collection, only the right predicate — but Phase 3 will not rediscover
it.

---

## 2. Fields that must never be features

| Field | Why |
|---|---|
| `stargazer_count_snapshot`, `fork_count_snapshot` | Snapshots as of `collected_at` applied to 2024 rows. A repo at 3,100 stars today may have had 700 in 2024. GitHub exposes no historical star API — unfixable. |
| Any `User.followers` / `User.pullRequests` total | Same anachronism at author level. Not collected, deliberately. |
| Everything in `reviews`, `thread_comments`, `issue_comments` | These *are* the target. They do not exist at prediction time for an unreviewed PR. Used only to build the label. |
| `merged`, `closed_at`, `merged_at`, `state` | Outcomes. |
| `n_participants`, `mergeable_current` | Accumulate after open. |
| `author_association_current` | Computed at read time; rewritten by later merges. See §5 — proven, not suspected. |

**Consequence of the star snapshot, worth stating in the write-up:** the star strata
are defined by stars at *selection* time but the PRs span 2024–2026. Repos drift
between tiers. This is a stated limitation, not a fixable one.

---

## 3. Genuinely safe features

**Point-in-time safe by construction** (from `pr_tier2.parquet`):
`created_at` and everything derived from it (`created_hour_utc`, `created_dayofweek`,
`is_weekend`), `is_cross_repository`, `head_repo_owner`, `maintainer_can_modify`,
`author_is_bot`, `author_is_deleted`, `author_created_at` (account age at PR time —
immutable, and one of the few signals that transfers to an unseen repo).

**Reconstructed at-open** (from `timeline.parquet` + `commits.parquet`):
`additions_at_open`, `deletions_at_open`, `n_commits_at_open`, at-open draft state,
at-open label count, at-open title, at-open base branch.

**`REVIEW_REQUESTED_EVENT` — a feature the blueprint wrongly bans.**
Blueprint §2 excludes `reviewRequests` because it reflects current state. Correct. But
it then discards the *event*, which is timestamped and point-in-time safe. A CODEOWNERS
auto-request fires within seconds of open (verified: a probed PR opened at 13:41:54 with
a `ReviewRequestedEvent` at 13:41:55) and is available at prediction time. Filter to
`event.created_at <= pr.created_at + 60s` and use it.

**Repo-level, transferable** (from `repo_meta.parquet`) — these matter most for the
blueprint's Scenario B cold-start split, and its feature plan has nothing like them:
`n_assignable_users` (direct maintainer-capacity proxy), `owner_type`
(Organization vs User is a review-culture proxy), `has_codeowners`, `has_pr_template`,
`has_contributing`, `n_ci_workflows`, `language_dominant`.

*Caveat:* the process-maturity probes reflect `HEAD` at `collected_at`, not 2024. Treat
as time-invariant repo metadata with a stated limitation.

---

## 4. The label is not decided yet — deliberately

Phase 1 emits **three separate review streams** rather than a single "first review"
timestamp, because the blueprint's definition is demonstrably unsafe as written.

**The finding.** In `anthropics/skills`, PRs from `authorAssociation: NONE` receive zero
reviews but do receive comments from `98zc5g5jyw-arch`, an apparent spam account. The
blueprint's rule — "first non-author, non-bot review **or comment**" — would mark those
PRs *reviewed* and corrupt the target on exactly the population the project is about.

**Which timestamp means "reviewed".** Use the time the author could *see* it:

| Stream | Use | Not |
|---|---|---|
| `reviews` | `submitted_at` | `created_at` — can be the *drafting* time, invisible to the author (verified: `created_at` 16:55:22 vs `submitted_at` 17:07:48). `PENDING` reviews have a null `submitted_at` and were never visible. |
| `thread_comments` | `published_at`, falling back to `created_at` | `drafted_at` |
| `issue_comments` | `published_at`, falling back to `created_at` | — |

**The `reviews` connection sorts by `submitted_at`, not `created_at`.** Resolved
empirically during the gate on `benbjohnson/litestream`: a `CHANGES_REQUESTED` review
drafted at 20:54 and submitted 18 hours later sits *after* an `APPROVED` review created
at 21:24 — GitHub orders by submission. Consequences:

- `created_at` is legitimately non-monotonic in `reviews.parquet`. Do not sort by it.
- `first:N` returns the N *earliest-submitted* reviews, which is exactly the set the
  label needs. The truncation argument holds.
- That PR is a concrete instance of the drafting gap: using `created_at` would credit
  the maintainer's review 18h before the author could see it.

**Inline comments are not a subset of reviews.** Verified: a probed PR had
`reviews.totalCount == 0` alongside 10 real `reviewThreads`. All three streams must be
consulted; `reviews` alone undercounts.

**Spam signal.** `is_minimized` + `minimized_reason` (`SPAM`, `ABUSE`, `OFF_TOPIC`,
`OUTDATED`, `RESOLVED`) is the closest thing to ground truth available — maintainers
hide junk. Captured on every comment. `body_text` is captured too, because a spam
heuristic cannot be built from a length column and cannot be re-fetched.

`gate_report.py` computes `is_slow` under five competing definitions and reports the
spread. If it exceeds 10 percentage points, the label is definition-dominated and
Phase 2 owes an explicit decision plus a sensitivity table in the write-up.

---

## 5. `author_association` is LEAKY — resolved (gate item #7)

**`authorAssociation` is computed at read time from current state, not frozen at PR
creation.** Proof, from `benbjohnson/litestream`: 35 of 88 distinct authors (40%) read
`CONTRIBUTOR` on their **very first PR** in the repo. `CONTRIBUTOR` requires a
previously merged contribution, which is impossible on a first PR. The field was
rewritten by what happened later.

This is worse than stale. "Later got a PR merged" is correlated with "got reviewed" —
the outcome — so the blueprint's `author_association` feature would carry the target
into the model. It is classified **LEAKY**, and the column is named
`author_association_current` so nobody reaches for it by accident.

**Replacement.** Reconstruct a point-in-time association from Tier 1 by chronological
replay, using only PRs strictly before the row's `created_at`:

| Reconstructed feature | From `pr_tier1.parquet` |
|---|---|
| `is_first_pr_here` | no earlier PR by this `author_login` in this repo |
| `n_prior_prs_here` | count of earlier PRs by this author |
| `n_prior_merged_here` | count of earlier PRs with `merged_at < created_at` |
| `prior_merge_rate_here` | the ratio, with a prior for small counts |
| `days_since_first_pr_here` | `created_at − min(earlier created_at)` |

These are exactly the blueprint's "author history" features — the blueprint had the
right idea and the wrong column. Note `n_prior_merged_here` must use `merged_at <
created_at`, not `merged == True`: a PR that was open at the time and merged later is
not prior knowledge.

`OWNER` / `MEMBER` / `COLLABORATOR` are probably more stable than `CONTRIBUTOR` (they
reflect org membership rather than merge history) but have not been verified, and
membership also changes. Treat the whole column as LEAKY.

---

## 6. Cohort selection caveats

Recorded in `data/cohort/cohort.json` and repeated here:

1. **Star tiers drift** (see §2).
2. **Survivorship bias.** Requiring recent activity and excluding archived repos
   structurally excludes repos that died during the window — plausibly the slowest
   reviewers — biasing the `is_slow` base rate *downward*.
3. **The volume floor** is a size criterion, not a review-behaviour criterion, but is
   mildly correlated with review throughput.
4. **The volume floor does not exclude non-code repos.** A trial run selected
   `zero-to-mastery/start-here-guidelines` — a learning-guidelines repo with 2,226
   in-window PRs whose review dynamics have nothing to do with code review. This is
   precisely why the cohort is 45 repos and Phase 2 subsets to 30. Useful structural
   discriminators already collected: `n_ci_workflows == 0`, `has_codeowners == False`,
   `language_dominant` disagreeing with the assigned stratum.

**Statistical consequence for Phase 4, worth writing down now.** The effective sample
size for a cross-repo generalisation claim is the number of *repos* (~45), not the
number of PRs (~50k). The blueprint's §4 proposal — "bootstrap over PRs, stratified by
repo" — holds the repo set fixed and will produce confidence intervals that are far too
narrow for the Scenario B claim. A **cluster bootstrap resampling whole repos** is the
right scheme.

---

## 7. Table reference

| File | Grain | Notes |
|---|---|---|
| `pr_tier1.parquet` | one row per PR, **all of repo history** | Thin. Powers backlog reconstruction, author history, trailing-window warm-up. |
| `pr_tier2.parquet` | one row per PR, **in-window only** | Rich. `_current`/`_final` suffixes mark SNAPSHOT fields. |
| `reviews.parquet` | one row per formal review | Use `submitted_at`. |
| `thread_comments.parquet` | one row per inline review comment | Carries `is_minimized`. |
| `issue_comments.parquet` | one row per general comment | Carries `is_minimized`. |
| `timeline.parquet` | one row per state event | Replay in order to recover at-open state. |
| `commits.parquet` | one row per commit | Use `authored_date`, not `committed_date`. |
| `repo_meta.parquet` | one row per repo | All SNAPSHOT as of `collected_at`. |

**Why Tier 1 spans all history.** The blueprint's window (2024-01-01 → 2026-06-30) is a
**row filter, not a collection filter**. "Open PR backlog at instant `t`" needs every PR
ever opened and still open at `t` — a 2019 PR counts. Verified:
`pullRequests(states:OPEN){totalCount}` returns the *current* count and is useless as a
historical value. Truncating collection at the window start would undercount backlog
**worst at the window start and least at its end**, injecting a spurious time trend into
a feature — in a project whose primary split is time-based. Gate item #6 measures how
much this actually mattered.

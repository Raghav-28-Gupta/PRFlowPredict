"""Frozen, versioned GraphQL query shapes for PRFlowPredict Phase 1 collection.

WHY THIS FILE IS SEPARATE AND VERSIONED
---------------------------------------
Phase 1's single job is to capture enough raw signal that Phases 2-8 never need to
re-scrape GitHub. That makes the query shape the most consequential artifact in the
project: a field omitted here is a feature that cannot exist later without a full
re-collection.

So:
  * QUERY_VERSION is stamped into every raw page written to disk.
  * parse.py dispatches on it.
  * After the Phase 0 gate freezes the shape, ANY edit to a query string here is a
    re-collection event, not a patch. Bump QUERY_VERSION and say so out loud.

Nested connection sizes are hardcoded in the query strings (not passed as variables)
precisely so that sha256(query) captures them. The only variable that may change at
runtime is the top-level page size, because adaptive backoff shrinks it on timeout --
that changes batching, not data shape.
"""

from __future__ import annotations

import hashlib

# Bump on ANY change to a query string below. See module docstring.
QUERY_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Nested connection sizes.
#
# Phase 0 gate item #2 measures the truncation rate at these values and raises them
# until it is ~0. Rate limit is not the binding constraint (a thin Tier-1 page costs
# 1 point per 100 PRs, measured), so prefer raising these over paginating sub-streams.
# ---------------------------------------------------------------------------
N_REVIEWS = 50
N_ISSUE_COMMENTS = 50
N_REVIEW_THREADS = 30
N_THREAD_COMMENTS = 10
N_TIMELINE = 60
N_COMMITS = 30
N_LABELS = 20

# ---------------------------------------------------------------------------
# Shared fragments
# ---------------------------------------------------------------------------

# `Actor` is an interface and does NOT expose createdAt/databaseId, so the useful
# fields need inline fragments. author can legitimately be null (deleted accounts) --
# the blueprint calls this out; parse.py must treat it as a category, not a crash.
#
# User.createdAt (account age at PR time) is immutable and transfers across repos,
# making it one of the few genuinely useful cold-start signals. It cannot be derived
# later from anything else we store, so it must be collected now.
#
# databaseId survives login renames; login does not. Store both.
_AUTHOR = """
    author {
      __typename
      login
      ... on User { databaseId createdAt isSiteAdmin }
      ... on Bot  { databaseId }
    }
"""

# ---------------------------------------------------------------------------
# TIER 1 -- every PR, all of time, thin shape.
#
# Measured: 1 point / nodeCount 100 for a 100-PR page. Full history of a 2,908-PR
# repo costs ~30 points. Negligible.
#
# This tier exists because the blueprint's repo-state features are NOT reconstructable
# from a 2024-2026 slice:
#   * "open PR backlog at instant t" needs every PR ever opened and still open at t,
#     including one opened in 2019. Verified: pullRequests(states:OPEN){totalCount}
#     returns the CURRENT count and is useless as a historical value.
#   * "days since first PR here" / "is first-time contributor" are wrong for any
#     author whose first PR predates the window.
#
# Sub-connections request ONLY totalCount (no `first:`), which is legal and adds zero
# node cost -- verified. They give Phase 2 a cheap truncation oracle over all history.
# ---------------------------------------------------------------------------
TIER1_PRS = """
query Tier1($owner: String!, $name: String!, $first: Int!, $after: String) {
  rateLimit { cost remaining nodeCount resetAt }
  repository(owner: $owner, name: $name) {
    id
    nameWithOwner
    pullRequests(
      first: $first
      after: $after
      states: [OPEN, CLOSED, MERGED]
      orderBy: { field: CREATED_AT, direction: ASC }
    ) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes {
        id
        databaseId
        number
        createdAt
        publishedAt
        closedAt
        mergedAt
        merged
        closed
        state
        isDraft
        isCrossRepository
        authorAssociation
        __AUTHOR__
        reviews { totalCount }
        comments { totalCount }
        reviewThreads { totalCount }
        commits { totalCount }
      }
    }
  }
}
""".replace("__AUTHOR__", _AUTHOR.strip())

# ---------------------------------------------------------------------------
# TIER 2 -- in-window PRs only, rich shape. Supplies labels and features.
#
# Started from the cursor Tier 1 recorded at the window boundary. Cursor portability
# across page sizes is VERIFIED (a cursor from a first:100 walk resumes correctly in a
# first:25 walk), but it is not contractual, so collect.py starts one page EARLY and
# asserts the first PR returned is at or before the window start.
#
# TWO TRAPS ENCODED HERE -- read before editing:
#
#   1. NEVER add `orderBy` to `comments`. Verified via schema introspection:
#      PullRequest.comments accepts an orderBy argument (IssueCommentOrder, field
#      UPDATED_AT only) while reviews and reviewThreads do not. UPDATED_AT reorders on
#      edit, which would silently destroy "first comment" semantics -- the exact thing
#      the label depends on. The default (creation order, ascending) is what we want.
#      It looks like a harmless addition. It is not.
#
#   2. timelineItems is split into TWO selections, and PULL_REQUEST_COMMIT is excluded
#      from the state-event one. Verified: including commits in timelineItems swamps
#      the first:N budget (a probed PR returned 129 items of which the first 8 were all
#      commits from 2010). Commits come from the separate `commits` connection instead.
#      Also verified: timelineItems.totalCount IGNORES the itemTypes filter, so any
#      "totalCount > N implies truncated" guard applied to it is wrong in BOTH
#      directions. parse.py must not apply one.
# ---------------------------------------------------------------------------
TIER2_PRS = """
query Tier2($owner: String!, $name: String!, $first: Int!, $after: String) {
  rateLimit { cost remaining nodeCount resetAt }
  repository(owner: $owner, name: $name) {
    id
    nameWithOwner
    pullRequests(
      first: $first
      after: $after
      states: [OPEN, CLOSED, MERGED]
      orderBy: { field: CREATED_AT, direction: ASC }
    ) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes {
        id
        databaseId
        number
        url
        createdAt
        publishedAt
        updatedAt
        closedAt
        mergedAt
        merged
        closed
        state
        isDraft
        isCrossRepository
        locked
        activeLockReason
        title
        body
        lastEditedAt
        editor { login }
        additions
        deletions
        changedFiles
        baseRefName
        headRefName
        headRepositoryOwner { login }
        maintainerCanModify
        mergeable
        authorAssociation
        milestone { title }
        __AUTHOR__

        labels(first: __N_LABELS__) { totalCount nodes { name } }
        assignees(first: 10) { totalCount nodes { login } }
        participants { totalCount }
        files { totalCount }
        closingIssuesReferences(first: 5) { totalCount }

        # --- Review stream 1 of 3: formal reviews ---
        # submittedAt is the time the author could SEE it. createdAt can be the
        # drafting time (verified: a probed review had createdAt 16:55:22 but
        # submittedAt 17:07:48). Capture all timestamps; Phase 2 chooses.
        # PENDING reviews have a null submittedAt and are invisible to the author.
        reviews(first: __N_REVIEWS__) {
          totalCount
          nodes {
            id
            databaseId
            state
            createdAt
            submittedAt
            publishedAt
            lastEditedAt
            authorAssociation
            bodyText
            comments { totalCount }
            __AUTHOR__
          }
        }

        # --- Review stream 2 of 3: inline review threads ---
        # Kept separate because inline comments can exist with NO parent review --
        # verified: a probed PR had reviews.totalCount 0 alongside 10 real
        # reviewThreads. `reviews` does not subsume these.
        #
        # isMinimized/minimizedReason (SPAM, ABUSE, OFF_TOPIC, OUTDATED, RESOLVED) is
        # the best available ground truth for the spam-commenter hazard found in
        # anthropics/skills -- maintainers hide those. Capture it everywhere.
        reviewThreads(first: __N_REVIEW_THREADS__) {
          totalCount
          nodes {
            id
            isResolved
            isOutdated
            resolvedBy { login }
            comments(first: __N_THREAD_COMMENTS__) {
              totalCount
              nodes {
                id
                createdAt
                draftedAt
                publishedAt
                lastEditedAt
                state
                authorAssociation
                isMinimized
                minimizedReason
                path
                originalLine
                replyTo { id }
                bodyText
                __AUTHOR__
              }
            }
          }
        }

        # --- Review stream 3 of 3: general issue comments ---
        # SEE TRAP 1 ABOVE: no orderBy here, ever.
        # bodyText is captured because Phase 2 cannot build a spam heuristic from a
        # length column, and cannot re-fetch.
        comments(first: __N_ISSUE_COMMENTS__) {
          totalCount
          nodes {
            id
            createdAt
            publishedAt
            lastEditedAt
            authorAssociation
            isMinimized
            minimizedReason
            bodyText
            __AUTHOR__
          }
        }

        # --- State events: recover CREATION-TIME values ---
        # The blueprint lists is_draft / label_count / title_length / base_branch as
        # "computable at created_at". They are not: those fields are current-state
        # snapshots. These events are what make the at-open values recoverable.
        #
        # REVIEW_REQUESTED_EVENT is deliberately included. The blueprint excludes
        # `reviewRequests` for being current-state (correct) but then discards the
        # EVENT, which is timestamped and point-in-time safe. A CODEOWNERS auto-request
        # firing seconds after open is available at prediction time and is likely a
        # strong signal. Phase 3 filters to createdAt <= pr.createdAt + 60s.
        timelineItems(
          first: __N_TIMELINE__
          itemTypes: [
            READY_FOR_REVIEW_EVENT, CONVERT_TO_DRAFT_EVENT,
            LABELED_EVENT, UNLABELED_EVENT, RENAMED_TITLE_EVENT,
            BASE_REF_CHANGED_EVENT, HEAD_REF_FORCE_PUSHED_EVENT,
            REVIEW_REQUESTED_EVENT, REVIEW_REQUEST_REMOVED_EVENT,
            REVIEW_DISMISSED_EVENT, ASSIGNED_EVENT, UNASSIGNED_EVENT,
            CLOSED_EVENT, REOPENED_EVENT, MERGED_EVENT, AUTO_MERGE_ENABLED_EVENT
          ]
        ) {
          totalCount   # NOTE: ignores itemTypes. Do not use as a truncation guard.
          nodes {
            __typename
            ... on ReadyForReviewEvent  { createdAt actor { login } }
            ... on ConvertToDraftEvent  { createdAt actor { login } }
            ... on LabeledEvent         { createdAt actor { login } label { name } }
            ... on UnlabeledEvent       { createdAt actor { login } label { name } }
            ... on RenamedTitleEvent    { createdAt actor { login } previousTitle currentTitle }
            ... on BaseRefChangedEvent  { createdAt actor { login } previousRefName currentRefName }
            ... on HeadRefForcePushedEvent { createdAt actor { login } }
            ... on ReviewRequestedEvent {
              createdAt actor { login }
              requestedReviewer {
                __typename
                ... on User { login }
                ... on Team { name }
                ... on Bot  { login }
              }
            }
            ... on ReviewRequestRemovedEvent { createdAt actor { login } }
            ... on ReviewDismissedEvent { createdAt actor { login } }
            ... on AssignedEvent   { createdAt actor { login } }
            ... on UnassignedEvent { createdAt actor { login } }
            ... on ClosedEvent     { createdAt actor { login } }
            ... on ReopenedEvent   { createdAt actor { login } }
            ... on MergedEvent     { createdAt actor { login } }
            ... on AutoMergeEnabledEvent { createdAt actor { login } }
          }
        }

        # --- Commits: approximate the at-open diff ---
        # additions/deletions/changedFiles on the PR reflect the FINAL head after every
        # post-open push. That is outcome contamination: big PRs attract slow review,
        # and slow review gives time to grow.
        #
        # IMPORTANT: reconstruct using authoredDate, NOT committedDate. Verified:
        # rebase/squash rewrites committedDate, so filtering on
        # committedDate <= pr.createdAt returns 0 for most multi-commit PRs.
        # authoredDate survives rebases and reconstructs sanely.
        #
        # Also verified: PullRequestCommit exposes only {commit, id, pullRequest,
        # resourcePath, url} -- there is NO "when did this commit enter the PR"
        # timestamp available anywhere in the API. authoredDate is the best we get.
        #
        # Mitigating: most PRs are single-commit (35/40 in a sample), and for those the
        # final diff IS the at-open diff exactly. Gate item #5 measures the real rate.
        commits(first: __N_COMMITS__) {
          totalCount
          nodes {
            commit {
              oid
              committedDate
              authoredDate
              additions
              deletions
              changedFilesIfAvailable   # can be null -- the name is a warning
            }
          }
        }
      }
    }
  }
}
""".replace("__AUTHOR__", _AUTHOR.strip()) \
   .replace("__N_LABELS__", str(N_LABELS)) \
   .replace("__N_REVIEWS__", str(N_REVIEWS)) \
   .replace("__N_REVIEW_THREADS__", str(N_REVIEW_THREADS)) \
   .replace("__N_THREAD_COMMENTS__", str(N_THREAD_COMMENTS)) \
   .replace("__N_ISSUE_COMMENTS__", str(N_ISSUE_COMMENTS)) \
   .replace("__N_TIMELINE__", str(N_TIMELINE)) \
   .replace("__N_COMMITS__", str(N_COMMITS))

# ---------------------------------------------------------------------------
# Repo-level metadata. One query per repo, essentially free.
#
# assignableUsers.totalCount is a direct maintainer-capacity proxy and one of the very
# few features that genuinely transfers to an unseen repo -- exactly what the
# blueprint's Scenario B (cold-start) needs, and its feature plan has nothing like it.
#
# The object(expression:) probes detect review-process maturity (CODEOWNERS, PR
# template, CI). Verified: these return null cleanly when the path is absent.
#
# CAVEAT recorded in the data dictionary: every field here is a snapshot as of
# collected_at, describing HEAD today, not the repo as it was in 2024. stargazerCount
# in particular must NEVER be used as a feature.
# ---------------------------------------------------------------------------
REPO_META = """
query RepoMeta($owner: String!, $name: String!) {
  rateLimit { cost remaining nodeCount resetAt }
  repository(owner: $owner, name: $name) {
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
    diskUsage
    hasIssuesEnabled
    hasDiscussionsEnabled
    owner { __typename login }
    primaryLanguage { name }
    languages(first: 5) { edges { size node { name } } }
    licenseInfo { key }
    defaultBranchRef { name }
    assignableUsers { totalCount }
    mentionableUsers { totalCount }
    pullRequests { totalCount }
    issues { totalCount }

    codeowners_github:  object(expression: "HEAD:.github/CODEOWNERS")  { __typename }
    codeowners_root:    object(expression: "HEAD:CODEOWNERS")          { __typename }
    codeowners_docs:    object(expression: "HEAD:docs/CODEOWNERS")     { __typename }
    pr_template_lower:  object(expression: "HEAD:.github/pull_request_template.md") { __typename }
    pr_template_upper:  object(expression: "HEAD:.github/PULL_REQUEST_TEMPLATE.md") { __typename }
    contributing_md:    object(expression: "HEAD:CONTRIBUTING.md")     { __typename }
    contributing_rst:   object(expression: "HEAD:CONTRIBUTING.rst")    { __typename }
    workflows: object(expression: "HEAD:.github/workflows") {
      __typename
      ... on Tree { entries { name } }
    }
  }
}
"""


def query_sha256(query: str) -> str:
    """Stable fingerprint of a query string, stamped into every raw page.

    Lets parse.py detect that pages were collected under different shapes -- which
    WILL happen if nested connection sizes are retuned mid-collection after the gate.
    Without it that becomes a silent heterogeneity bug discovered months later.
    """
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


# Precomputed so collect.py stamps without recomputing per page.
TIER1_SHA = query_sha256(TIER1_PRS)
TIER2_SHA = query_sha256(TIER2_PRS)
REPO_META_SHA = query_sha256(REPO_META)

QUERY_SHAS = {
    "tier1": TIER1_SHA,
    "tier2": TIER2_SHA,
    "repo_meta": REPO_META_SHA,
}

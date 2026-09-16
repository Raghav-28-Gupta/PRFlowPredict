"""GitHub GraphQL transport for PRFlowPredict Stage 1.

WHY NOT `gh api graphql`
-----------------------
Two reasons, both load-bearing:

  1. The blueprint requires raw JSON persisted before parsing. `gh` reshapes and
     re-encodes output, which means anything it hands back has already been through a
     parser -- defeating the requirement. We need the exact bytes off the wire.
  2. Rate-limit handling needs response HEADERS (x-ratelimit-*, retry-after,
     x-github-request-id). `gh` does not surface them.

A third, lesser reason: collection is ~tens of thousands of requests, and that many
subprocess spawns is the wrong primitive on Windows.

We still borrow `gh`'s credential store rather than asking for a second token.

ERROR MODEL
-----------
GraphQL's failure modes do not map onto HTTP status codes, which is the single most
dangerous thing about collecting from it:

  * A partial failure returns HTTP 200 with {"data": {...nulls...}, "errors": [...]}.
    A collector that checks only `r.status_code` will happily persist a page full of
    nulls, advance its cursor, and mark the repo complete. A parse bug is recoverable;
    a silently truncated corpus is not.
  * A query-validation error returns errors with NO `data` key at all -- a different
    shape again.

So this module classifies every response and the caller must handle all three outcomes
explicitly. `execute()` never silently returns a degraded page.
"""

from __future__ import annotations

import json
import logging
import random
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import requests

log = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.github.com/graphql"

# GraphQL error `type` values we treat as worth retrying rather than recording as a
# permanent failure of the page.
TRANSIENT_ERROR_TYPES = {"RATE_LIMITED", "SERVICE_UNAVAILABLE", "INTERNAL"}

# Substrings in a GraphQL error message that indicate a server-side timeout. GitHub
# returns these with assorted/absent `type` values, so message matching is unavoidable.
TIMEOUT_HINTS = ("timeout", "timed out", "took too long", "try again")

# Substrings identifying the SECONDARY rate limit, which is a distinct mechanism from
# the 5,000-points/hour primary limit and is the one that actually bites during a long
# serial crawl.
SECONDARY_LIMIT_HINTS = ("secondary rate limit", "abuse detection", "exceeded a secondary")


class FatalError(RuntimeError):
    """Not retryable. Bad credentials, malformed query, repo genuinely gone."""


class TransientError(RuntimeError):
    """Retryable. Carries an optional server-instructed wait."""

    def __init__(self, message: str, retry_after: float | None = None,
                 kind: str = "transient"):
        super().__init__(message)
        self.retry_after = retry_after
        # "internal"  -> GitHub failed to execute the query itself
        # "timeout"   -> query too expensive; a smaller page may succeed
        # "ratelimit" -> back off in TIME; a smaller page makes it WORSE
        # The caller needs this distinction: shrinking the page is the right answer
        # for one of these and the wrong answer for another.
        self.kind = kind


@dataclass
class Response:
    """One GraphQL round trip, with the wire bytes preserved verbatim."""

    raw: bytes
    status: int
    headers: dict[str, str]
    elapsed_s: float
    request_id: str | None = None
    graphql_errors: list[dict[str, Any]] = field(default_factory=list)
    _parsed: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        """True only if HTTP succeeded AND the body carried no GraphQL errors."""
        return self.status == 200 and not self.graphql_errors

    def json(self) -> dict[str, Any]:
        """Parsed body, decoded once and cached.

        `raw` remains the authoritative artifact -- this is a convenience for the
        collector's bookkeeping only. What gets persisted to disk is always `raw`.
        """
        if self._parsed is None:
            self._parsed = json.loads(self.raw)
        return self._parsed

    @property
    def rate_limit(self) -> dict[str, Any]:
        """The `rateLimit` block, per-response rather than client-global.

        Read from this response's own body so a manifest line can never be attributed
        the cost of a different request.
        """
        return (self.json().get("data") or {}).get("rateLimit") or {}

    @property
    def cost(self) -> int:
        return int(self.rate_limit.get("cost") or 0)


def get_token() -> str:
    """Borrow the token from the `gh` credential store.

    Deliberately not read from GITHUB_TOKEN/GH_TOKEN: neither is set in this
    environment, and silently picking up an ambient token of unknown scope would make
    collection non-reproducible.
    """
    try:
        proc = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FatalError(f"could not invoke `gh auth token`: {exc}") from exc

    token = proc.stdout.strip()
    if proc.returncode != 0 or not token:
        raise FatalError(
            "`gh auth token` returned no token. Run `gh auth login` first. "
            f"stderr: {proc.stderr.strip()[:200]}"
        )
    return token


class GitHubGraphQL:
    """Serial GraphQL client with rate-limit awareness and classified retries.

    Serial by design. Concurrency is what triggers GitHub's secondary rate limit, and
    the primary limit is not our constraint anyway (a thin Tier-1 page costs 1 point
    per 100 PRs, measured), so there is nothing to gain and a multi-hour backoff to
    lose.
    """

    def __init__(
        self,
        token: str | None = None,
        min_interval_s: float = 0.8,
        max_retries: int = 6,
        # INTERNAL errors are deterministic for a given cursor+page size, so retrying
        # them on the full budget wastes minutes per page (6 retries x backoff up to
        # 60s). Fail fast instead and let the caller shrink the page.
        internal_max_retries: int = 2,
        reserve_points: int = 100,
        user_agent: str = "PRFlowPredict/1.0 (research; collection)",
    ):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"bearer {token or get_token()}",
                "Content-Type": "application/json",
                "User-Agent": user_agent,
            }
        )
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.internal_max_retries = internal_max_retries
        self.reserve_points = reserve_points
        self._last_request_at = 0.0

        # Populated from each response so the caller can log real cost per page
        # (gate item #8) rather than extrapolating from a probe query.
        self.last_cost: int | None = None
        self.last_remaining: int | None = None

    # -- pacing ---------------------------------------------------------------

    def _respect_min_interval(self) -> None:
        gap = time.monotonic() - self._last_request_at
        if gap < self.min_interval_s:
            time.sleep(self.min_interval_s - gap)

    def _respect_primary_limit(self, headers: dict[str, str]) -> None:
        """Sleep until reset if the points budget is nearly spent.

        Leaves `reserve_points` unspent so an in-flight retry cannot tip us over and
        turn a recoverable pause into a hard 403.
        """
        remaining = headers.get("x-ratelimit-remaining")
        reset = headers.get("x-ratelimit-reset")
        if remaining is None or reset is None:
            return
        try:
            remaining_i, reset_i = int(remaining), int(reset)
        except ValueError:
            return

        if remaining_i > self.reserve_points:
            return

        wait = max(0.0, reset_i - time.time()) + 5.0
        log.warning(
            "primary rate limit near exhaustion (%s remaining); sleeping %.0fs until reset",
            remaining_i,
            wait,
        )
        time.sleep(wait)

    # -- classification -------------------------------------------------------

    @staticmethod
    def _classify_http(resp: requests.Response) -> None:
        """Raise for HTTP-level failures. Returns quietly if the status is usable."""
        status = resp.status_code
        if status == 200:
            return

        body = resp.text[:500].lower()
        retry_after = resp.headers.get("retry-after")
        wait = float(retry_after) if retry_after and retry_after.isdigit() else None

        if status in (429,) or any(h in body for h in SECONDARY_LIMIT_HINTS):
            # Secondary limit. Retryable, and the server usually tells us how long.
            raise TransientError(f"secondary rate limit (HTTP {status})",
                                 retry_after=wait or 60.0, kind="ratelimit")
        if status in (500, 502, 503, 504):
            raise TransientError(f"server error HTTP {status}", retry_after=wait,
                                 kind="internal")
        if status in (401, 403):
            # 403 without a secondary-limit hint means scope/permission -- not retryable.
            raise FatalError(f"auth/permission failure HTTP {status}: {resp.text[:300]}")
        raise FatalError(f"unexpected HTTP {status}: {resp.text[:300]}")

    @staticmethod
    def _classify_graphql(errors: list[dict[str, Any]]) -> None:
        """Raise TransientError if the body's errors are worth retrying.

        Field-level errors that are NOT transient (e.g. NOT_FOUND on one field) are
        left for the caller to persist and handle -- they are data about the repo, not
        a transport failure.
        """
        for err in errors:
            etype = str(err.get("type", ""))
            msg = str(err.get("message", "")).lower()
            if etype == "RATE_LIMITED":
                raise TransientError(f"graphql {etype}: {msg[:200]}", kind="ratelimit")
            if etype in TRANSIENT_ERROR_TYPES:
                # Observed live on anthropics/skills: GitHub returns INTERNAL for a
                # specific span of PRs at EVERY page size. It is deterministic for a
                # given cursor, so retrying it many times just burns wall clock --
                # the caller's page-size ladder is the real remedy.
                raise TransientError(f"graphql {etype}: {msg[:200]}", kind="internal")
            if any(hint in msg for hint in TIMEOUT_HINTS):
                raise TransientError(f"graphql timeout: {msg[:200]}", kind="timeout")

    # -- main entry point -----------------------------------------------------

    def execute(
        self,
        query: str,
        variables: dict[str, Any],
        timeout: float = 90.0,
    ) -> Response:
        """Run one query, retrying transport failures with jittered backoff.

        Returns a Response whose `.raw` is the exact bytes received. A returned
        Response may still carry non-transient GraphQL errors in `.graphql_errors`;
        `.ok` is False in that case and the CALLER must decide what to do. It must not
        advance its cursor.
        """
        payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            self._respect_min_interval()
            started = time.monotonic()
            try:
                resp = self.session.post(
                    GRAPHQL_URL, data=payload, timeout=timeout
                )
                self._last_request_at = time.monotonic()
                elapsed = self._last_request_at - started

                self._classify_http(resp)

                raw = resp.content  # bytes, pre-decode. This is what gets persisted.
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise TransientError(f"undecodable response body: {exc}") from exc

                errors = body.get("errors") or []
                self._classify_graphql(errors)

                # Record real cost for the gate's cost measurement.
                rl = (body.get("data") or {}).get("rateLimit") or {}
                self.last_cost = rl.get("cost")
                self.last_remaining = rl.get("remaining")

                out = Response(
                    raw=raw,
                    status=resp.status_code,
                    headers=dict(resp.headers),
                    elapsed_s=elapsed,
                    request_id=resp.headers.get("x-github-request-id"),
                    graphql_errors=errors,
                )
                self._respect_primary_limit(out.headers)
                return out

            except FatalError:
                raise
            # requests.RequestException is the base: it covers Timeout and ConnectionError
            # AND the mid-body failures (ChunkedEncodingError, ContentDecodingError) that
            # are not ConnectionError subclasses. Observed live: a ChunkedEncodingError
            # escaped a narrower catch here and killed a 45-repo run at repo 16.
            except (TransientError, requests.RequestException) as exc:
                last_exc = exc
                self._last_request_at = time.monotonic()
                kind = getattr(exc, "kind", "transient")
                budget = (self.internal_max_retries if kind == "internal"
                          else self.max_retries)
                if attempt >= budget:
                    break
                wait = getattr(exc, "retry_after", None)
                if wait is None:
                    # Exponential backoff with full jitter, capped. Jitter matters:
                    # a deterministic retry train re-triggers the secondary limit.
                    wait = min(60.0, 2.0 ** attempt) * (0.5 + random.random() * 0.5)
                log.warning(
                    "attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt, self.max_retries, exc, wait,
                )
                time.sleep(wait)

        raise TransientError(
            f"exhausted {attempt} attempt(s); last error: {last_exc}",
            kind=getattr(last_exc, "kind", "transient"),
        )

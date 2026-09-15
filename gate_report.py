"""Phase 0 gate: decide whether the labelling logic is sound enough to scale.

A gate is theatre unless every number has a threshold written down BEFORE the run and
a stated consequence for failing. The thresholds below were pre-registered in the
Phase 1 plan; this script only measures.

It deliberately does NOT pick a label definition. It measures how much the choice
MATTERS (item 3), which is the actual open question after the anthropics/skills
finding that a spam account comments on otherwise-unreviewed PRs. If the spread across
definitions is wide, Phase 2 owes a decision and the write-up owes a sensitivity table.

Items 1, 9 and 10 are hard stops and are enforced elsewhere:
  1  ordering monotonicity  -> asserted on every run of parse.py
  9  resume idempotency     -> test_resume.py
 10  raw->parse conservation -> parse.py's conservation report
This script reports items 2-8 and restates the status of the hard stops.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger("gate")

PROCESSED = Path(__file__).parent / "data" / "processed"
RAW = Path(__file__).parent / "data" / "raw"

SLOW_THRESHOLD_H = 168.0   # blueprint §1: 7 days
CENSOR_WINDOW_H = 720.0    # blueprint §1: 30 days
WINDOW_START = pd.Timestamp("2024-01-01", tz="UTC")
WINDOW_END = pd.Timestamp("2026-06-30 23:59:59", tz="UTC")


def _load(repo_dir: Path, name: str) -> pd.DataFrame:
    p = repo_dir / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def first_human_event(
    prs: pd.DataFrame,
    events: pd.DataFrame,
    time_col: str,
    *,
    exclude_minimized: bool = False,
    exclude_association: set[str] | None = None,
) -> pd.Series:
    """Earliest event per PR that is not by the PR's own author and not by a bot.

    This is the mechanical part of the label, applied identically to every candidate
    definition so the comparison is apples to apples. What VARIES between definitions
    is which event streams are passed in and which exclusions are applied -- that is
    the Phase 2 decision this script is measuring the sensitivity of.
    """
    if events.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")

    ev = events.copy()
    if time_col not in ev.columns:
        return pd.Series(dtype="datetime64[ns, UTC]")

    ev = ev[ev[time_col].notna()]
    if exclude_minimized and "is_minimized" in ev.columns:
        ev = ev[~(ev["is_minimized"] == True)]  # noqa: E712 -- null-safe
    if exclude_association and "author_association" in ev.columns:
        ev = ev[~ev["author_association"].isin(exclude_association)]

    # non-bot
    if "author_is_bot" in ev.columns:
        ev = ev[~(ev["author_is_bot"] == True)]  # noqa: E712 -- null-safe

    # non-author: join the PR's own author login and drop self-events
    pr_author = prs.set_index("pr_id")["author_login"]
    ev = ev.join(pr_author.rename("_pr_author"), on="pr_id")
    ev = ev[ev["author_login"].notna() & (ev["author_login"] != ev["_pr_author"])]

    return ev.groupby("pr_id")[time_col].min()


def label_under(prs: pd.DataFrame, first_event: pd.Series) -> pd.DataFrame:
    """Apply the blueprint's label rule to one definition's first-event series.

    The 30-day observation window is what makes censoring a non-issue: any PR with no
    qualifying event within 720h is is_slow=1 with certainty, because 720h > 168h.
    """
    out = prs[["pr_id", "created_at"]].copy()
    out["first_event"] = out["pr_id"].map(first_event)
    out["wait_h"] = (out["first_event"] - out["created_at"]).dt.total_seconds() / 3600.0

    reviewed_in_window = out["wait_h"].notna() & (out["wait_h"] <= CENSOR_WINDOW_H)
    out["never_reviewed_30d"] = ~reviewed_in_window
    out["is_slow"] = (~reviewed_in_window) | (out["wait_h"] > SLOW_THRESHOLD_H)
    return out


def build_definitions(repo_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    prs = _load(repo_dir, "pr_tier2")
    if prs.empty:
        return prs, {}

    reviews = _load(repo_dir, "reviews")
    threads = _load(repo_dir, "thread_comments")
    issues = _load(repo_dir, "issue_comments")

    # Reviews use submittedAt: the time the author could SEE the review. createdAt can
    # be the drafting time, which is invisible to them (verified: a probed review had
    # createdAt 16:55:22 vs submittedAt 17:07:48). Falling back to createdAt only where
    # submittedAt is absent.
    if not reviews.empty:
        reviews = reviews.copy()
        reviews["visible_at"] = reviews["submitted_at"].fillna(reviews["created_at"])
    if not threads.empty:
        threads = threads.copy()
        threads["visible_at"] = threads["published_at"].fillna(threads["created_at"])
    if not issues.empty:
        issues = issues.copy()
        issues["visible_at"] = issues["published_at"].fillna(issues["created_at"])

    def cat(*frames):
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    defs = {
        "D1 reviews only":
            first_human_event(prs, reviews, "visible_at"),
        "D2 + inline review comments":
            first_human_event(prs, cat(reviews, threads), "visible_at"),
        "D3 + issue comments (blueprint)":
            first_human_event(prs, cat(reviews, threads, issues), "visible_at"),
        "D4 D3 excl. minimized":
            first_human_event(prs, cat(reviews, threads, issues), "visible_at",
                              exclude_minimized=True),
        "D5 D3 excl. NONE-association":
            first_human_event(prs, cat(reviews, threads, issues), "visible_at",
                              exclude_association={"NONE"}),
    }
    return prs, {k: label_under(prs, v) for k, v in defs.items()}


def gate_for_repo(repo_dir: Path) -> dict:
    name = repo_dir.name.replace("__", "/")
    prs, labelled = build_definitions(repo_dir)
    print("\n" + "=" * 74)
    print(f"PHASE 0 GATE -- {name}")
    print("=" * 74)

    if prs.empty:
        print("  no tier2 data; run collect.py --tier 2 first")
        return {"repo": name, "status": "NO DATA"}

    res: dict = {"repo": name, "n_prs": len(prs)}

    # State the actual coverage before any rate. A partial collection can easily hold
    # PRs from outside the modelling window (e.g. Tier 2 launched from an incomplete
    # Tier 1 boundary cursor), and every rate below would then describe the wrong
    # population. Two specific traps this guards against:
    #   * GitHub's review feature did not exist before 2016, so a "reviews only" label
    #     is degenerate on older PRs for reasons that say nothing about the cohort.
    #   * PRs newer than window_end have not had the full 30-day observation period.
    lo, hi = prs["created_at"].min(), prs["created_at"].max()
    in_window = (prs["created_at"] >= WINDOW_START) & (prs["created_at"] <= WINDOW_END)
    n_all, n_in = len(prs), int(in_window.sum())
    res.update({"created_range": [str(lo), str(hi)], "n_parsed": n_all, "n_in_window": n_in})
    print(f"\n  PRs parsed: {n_all}   covering {lo.date()} .. {hi.date()}")
    print(f"  inside the modelling window ({WINDOW_START.date()}..{WINDOW_END.date()}): "
          f"{n_in}  <- every rate below is computed on THESE rows only")

    # A handful of trailing rows past window_end is normal (the last Tier-2 page
    # overshoots before the stop check fires) and is simply filtered out. A LARGE
    # out-of-window share, or pre-2016 rows, means the parsed set is not the
    # modelling set at all and the run is a smoke test, not a gate.
    if n_in < n_all:
        frac_out = 1 - n_in / n_all
        if frac_out > 0.10 or lo < pd.Timestamp("2016-01-01", tz="UTC"):
            print(f"  *** WARNING: {frac_out:.0%} of parsed PRs fall outside the window.")
            if lo < pd.Timestamp("2016-01-01", tz="UTC"):
                print("      Pre-2016 PRs predate GitHub's review feature entirely, so a")
                print("      reviews-only label is degenerate on them by construction.")
            print("      Treat this run as a PIPELINE smoke test, NOT a go/no-go gate.")
    if n_in == 0:
        print("  no in-window rows; nothing to gate")
        return res
    prs = prs[in_window].reset_index(drop=True)
    labelled = {k: v[v["pr_id"].isin(prs["pr_id"])].reset_index(drop=True)
                for k, v in labelled.items()}

    # -- item 2: truncation rate -------------------------------------------
    trunc = float(prs["is_truncated"].mean())
    res["truncation_rate"] = trunc
    print(f"\n  [2] TRUNCATION RATE       {trunc:6.2%}   "
          f"{'PASS' if trunc < 0.01 else 'FAIL -> raise first:N and re-freeze query'}")

    # -- item 3: label-definition sensitivity ------------------------------
    print("\n  [3] LABEL SENSITIVITY  (the anthropics/skills spam finding)")
    print(f"      {'definition':<34} {'is_slow':>8} {'never@30d':>10} {'median wait h':>14}")
    rates = {}
    for label, df in labelled.items():
        rate = float(df["is_slow"].mean())
        never = float(df["never_reviewed_30d"].mean())
        med = df.loc[~df["never_reviewed_30d"], "wait_h"].median()
        rates[label] = rate
        print(f"      {label:<34} {rate:7.1%} {never:9.1%} "
              f"{(f'{med:.1f}' if pd.notna(med) else '-'):>14}")
    spread = (max(rates.values()) - min(rates.values())) if rates else 0.0
    res["label_rates"] = rates
    res["label_spread"] = spread
    print(f"\n      spread across definitions: {spread:.1%}  "
          f"{'(definition-dominated -- Phase 2 must resolve this)' if spread > 0.10 else '(stable)'}")

    # -- item 4: censoring / degeneracy ------------------------------------
    primary = labelled.get("D3 + issue comments (blueprint)")
    never = float(primary["never_reviewed_30d"].mean())
    pos = float(primary["is_slow"].mean())
    res["never_reviewed_30d"] = never
    res["is_slow_rate"] = pos
    ok4 = (0.10 <= never <= 0.35) and (0.05 <= pos <= 0.60)
    print(f"\n  [4] CENSORING / DEGENERACY")
    print(f"      never reviewed @30d : {never:6.1%}   (blueprint expects 10-35%)")
    print(f"      is_slow positive    : {pos:6.1%}   (usable range 5-60%)")
    print(f"      {'PASS' if ok4 else 'FAIL -> recalibrate the 168h threshold from the histogram'}")

    # -- item 5: snapshot contamination ------------------------------------
    tl = _load(repo_dir, "timeline")
    print("\n  [5] SNAPSHOT CONTAMINATION  (how wrong the blueprint's feature list is)")
    if not tl.empty:
        merged = tl.merge(prs[["pr_id", "created_at"]], on="pr_id",
                          suffixes=("", "_pr"))
        for ev, note in [("ReadyForReviewEvent", "is_draft wrong at open"),
                         ("LabeledEvent", "label_count wrong at open"),
                         ("RenamedTitleEvent", "title_length wrong at open"),
                         ("BaseRefChangedEvent", "base_branch wrong at open")]:
            sub = merged[(merged["event_type"] == ev) &
                         (merged["created_at"] > merged["created_at_pr"])]
            frac = sub["pr_id"].nunique() / len(prs)
            print(f"      {ev:<22} {frac:6.1%} of PRs   ({note})")
    multi = float((prs["n_commits_total"] > 1).mean())
    exact = float(prs["diff_is_exact"].mean())
    print(f"      {'multi-commit':<22} {multi:6.1%} of PRs   (final diff != at-open diff)")
    print(f"      {'diff_is_exact':<22} {exact:6.1%} of PRs   (no approximation needed)")
    res["multi_commit_rate"] = multi
    res["diff_exact_rate"] = exact
    if multi > 0.20:
        print("      -> >20%: timeline/commit reconstruction is MANDATORY, not optional")

    # -- item 6: warm-up necessity -----------------------------------------
    # Proves empirically whether full-history Tier 1 was needed, rather than arguing it.
    t1 = _load(repo_dir, "pr_tier1")
    print("\n  [6] WARM-UP NECESSITY  (was full-history Tier 1 actually required?)")
    if not t1.empty:
        open_at_start = t1[
            (t1["created_at"] < WINDOW_START)
            & (t1["closed_at"].isna() | (t1["closed_at"] > WINDOW_START))
        ]
        print(f"      PRs open at {WINDOW_START.date()} that were created BEFORE it: "
              f"{len(open_at_start)}")
        in_window = t1[t1["created_at"] >= WINDOW_START]
        early = in_window[in_window["created_at"] < WINDOW_START + pd.Timedelta(days=90)]
        if len(early):
            prior_authors = set(t1[t1["created_at"] < WINDOW_START]["author_login"].dropna())
            frac = early["author_login"].isin(prior_authors).mean()
            print(f"      early-window PRs whose author has a pre-window PR: {frac:6.1%}")
            res["early_authors_with_history"] = float(frac)
            if frac > 0.10:
                print("      -> >10%: truncating at the window start would have "
                      "mislabelled these authors as first-timers")
        res["open_at_window_start"] = len(open_at_start)
    else:
        print("      no tier1 data collected; run collect.py --tier 1")

    # -- item 7: authorAssociation mutability ------------------------------
    # CONTRIBUTOR requires a previously merged contribution, which is impossible on an
    # author's FIRST PR in the repo. Any CONTRIBUTOR on a first PR therefore proves the
    # field is computed at READ time from current state, not frozen at creation.
    print("\n  [7] authorAssociation MUTABILITY  (is the field frozen at creation?)")
    if not t1.empty and "author_association_current" in t1.columns:
        authored = t1[t1["author_login"].notna()].sort_values("created_at")
        first = authored.groupby("author_login").first()
        n_auth = len(first)
        n_contrib = int((first["author_association_current"] == "CONTRIBUTOR").sum())
        frac = n_contrib / n_auth if n_auth else 0.0
        res["first_pr_contributor_rate"] = frac
        print(f"      distinct authors: {n_auth}; CONTRIBUTOR on their FIRST PR: "
              f"{n_contrib} ({frac:.0%})")
        if n_contrib:
            print("      -> MUTABLE. author_association is computed at read time and is")
            print("         LEAKY (later merges rewrite it). Reconstruct point-in-time")
            print("         association from Tier 1 history instead.")
        else:
            print("      -> no evidence of mutability in this repo (inconclusive, not proof)")
    else:
        print("      no tier1 data; cannot test")

    # -- item 8: real cost --------------------------------------------------
    mpath = RAW / repo_dir.name / "manifest.jsonl"
    if mpath.exists():
        recs = [json.loads(l) for l in open(mpath, encoding="utf-8") if l.strip()]
        t2 = [r for r in recs if r.get("tier") == 2 and r.get("pr_count")]
        if t2:
            cost = sum(r.get("rate_limit_cost") or 0 for r in t2)
            n = sum(r["pr_count"] for r in t2)
            byts = sum(r.get("bytes") or 0 for r in t2)
            per_k = cost / n * 1000
            print("\n  [8] MEASURED COST  (at the frozen query shape, not a probe)")
            print(f"      tier2: {cost} points for {n} PRs = {per_k:.1f} points/1000 PRs")
            print(f"      at 5000 pts/hr -> {5000 / per_k * 1000:,.0f} PRs/hour")
            print(f"      raw bytes/PR: {byts / n:,.0f}  "
                  f"-> ~{byts / n * 50000 / 1e9:.2f} GB for 50k PRs (uncompressed JSON)")
            res["points_per_1k_prs"] = per_k

    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", help="owner/name; omit for all parsed repos")
    ap.add_argument("--json-out", help="write the machine-readable summary here")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dirs = ([PROCESSED / args.repo.replace("/", "__")] if args.repo
            else sorted(d for d in PROCESSED.glob("*") if d.is_dir()))
    if not dirs or not dirs[0].exists():
        log.error("no parsed data under %s -- run parse.py first", PROCESSED)
        return 1

    results = [gate_for_repo(d) for d in dirs if d.exists()]

    print("\n" + "=" * 74)
    print("HARD STOPS (enforced elsewhere)")
    print("=" * 74)
    print("  [1]  ordering monotonicity  -> asserted on every parse.py run")
    print("  [9]  resume idempotency     -> python test_resume.py")
    print("  [10] raw->parse conservation -> parse.py conservation report")
    print()

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2, default=str),
                                       encoding="utf-8")
        print(f"  summary written to {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

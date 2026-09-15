"""Pre-registered, STRUCTURAL-ONLY cohort checks. Writes data/cohort/kept.json.

There is deliberately no check on the label distribution. A 98%-never-reviewed repo
is data. Dropping on the outcome is the one thing the blueprint forbids for selection,
and post-hoc QC is selection."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

import labels
import load

log = logging.getLogger("qc")

COHORT = Path(__file__).parent / "data" / "cohort" / "cohort.json"
KEPT = Path(__file__).parent / "data" / "cohort" / "kept.json"
WINDOW = (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-06-30 23:59:59", tz="UTC"))

FAMILY = {"Python": {"Python"}, "TypeScript": {"TypeScript", "JavaScript"}, "Go": {"Go"}}
MAX_BOT_SHARE = 0.50
MIN_HUMAN_PRS = 100
LOOK_ROW_SHARE = 0.08
LOOK_SPREAD_PP = 0.15


def check_repo(entry: dict, frames: dict[str, pd.DataFrame], window) -> dict:
    repo = entry["repo"]
    prs = frames["pr_tier2"]
    prs = prs[(prs["created_at"] >= window[0]) & (prs["created_at"] <= window[1])]
    n = len(prs)
    n_bot = int(prs["author_is_bot"].sum()) if n else 0
    n_human = n - n_bot
    bot_share = n_bot / n if n else 0.0
    meta = frames["repo_meta"]
    lang = str(meta["language_dominant"].iloc[0]) if not meta.empty else None

    reasons = []
    if bot_share > MAX_BOT_SHARE:
        reasons.append(f"bot-authored share {bot_share:.0%} > {MAX_BOT_SHARE:.0%}")
    if n_human < MIN_HUMAN_PRS:
        reasons.append(f"{n_human} human PRs < {MIN_HUMAN_PRS}")
    fam = FAMILY.get(entry["language_stratum"], set())
    if lang not in fam:
        reasons.append(f"language {lang!r} not in stratum family {sorted(fam)}")

    # Rates are REPORTED for the look-at-these list, never used as a drop reason.
    human = prs[~prs["author_is_bot"]]
    streams = {k: frames.get(k, pd.DataFrame()) for k in labels.ALL_STREAMS}
    d5 = d3 = float("nan")
    if len(human):
        lab = labels.label_all(human, streams)
        d5 = float(lab["D5"]["is_slow"].mean())
        d3 = float(lab["D3"]["is_slow"].mean())

    return {
        "repo": repo, "cell": entry["cell"], "kept": not reasons, "reasons": reasons,
        "n_in_window": n, "n_human": n_human, "bot_share": bot_share,
        "language_dominant": lang, "is_slow_d5": d5, "is_slow_d3": d3,
        "spread_d3_d5": abs(d5 - d3) if n_human else float("nan"),
    }


def run() -> list[dict]:
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))["selected"]
    results = [check_repo(e, load.load_repo(e["repo"]), WINDOW) for e in cohort]
    kept = [r for r in results if r["kept"]]
    total = sum(r["n_human"] for r in kept) or 1
    look = []
    for r in kept:
        share = r["n_human"] / total
        r["row_share"] = share
        why = []
        if share > LOOK_ROW_SHARE:
            why.append(f"row share {share:.0%}")
        if r["spread_d3_d5"] > LOOK_SPREAD_PP:
            why.append(f"D3-D5 spread {r['spread_d3_d5']:.0%}")
        if why:
            look.append({"repo": r["repo"], "why": why})
    KEPT.write_text(json.dumps({
        "kept": [r["repo"] for r in kept], "repos": results, "look_at_these": look,
        "rules": {"max_bot_share": MAX_BOT_SHARE, "min_human_prs": MIN_HUMAN_PRS,
                  "family": {k: sorted(v) for k, v in FAMILY.items()}},
    }, indent=2), encoding="utf-8")
    return results


def kept_repos() -> list[str]:
    return json.loads(KEPT.read_text(encoding="utf-8"))["kept"]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = run()
    kept = [r for r in res if r["kept"]]
    print(f"kept {len(kept)}/{len(res)}")
    for r in res:
        if not r["kept"]:
            print(f"  DROP {r['repo']:<45} {'; '.join(r['reasons'])}")
    look = json.loads(KEPT.read_text(encoding="utf-8"))["look_at_these"]
    if look:
        print("look at these (not dropped):")
        for l in look:
            print(f"  {l['repo']:<45} {'; '.join(l['why'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

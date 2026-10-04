"""The demo's logic, kept free of Streamlit so it can be tested directly.

Everything the app shows is computed here, at runtime, from committed files: the extract
demo/data/prs.parquet and its feature list demo/data/features.json (both built by
build_demo_data.py), and Phase 4's data/phase4_runs.json and Phase 6's
data/phase6_importance_{A,B}.csv. Imports pandas and the standard library only: the deployed
app installs demo/requirements.txt, not the project's full requirements."""
from __future__ import annotations

import json
import math
import random
import re
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data" / "prs.parquet"
FEATURES = HERE / "data" / "features.json"
RUNS = HERE.parent / "data" / "phase4_runs.json"
IMPORTANCE = str(HERE.parent / "data" / "phase6_importance_{}.csv")
SCORES = {"A": "score_a", "B": "score_b"}
DRIVERS = {"A": "drivers_a", "B": "drivers_b"}
TOP_K = 3
FIRST_DAY, LAST_DAY, DEFAULT_DAY = date(2026, 1, 2), date(2026, 6, 30), date(2026, 4, 1)
EARLY_JANUARY = date(2026, 1, 14)      # until here, few scored PRs can have been waiting yet
PR_LINK = re.compile(r"github\.com/([^/\s]+/[^/\s]+)/pull/(\d+)")
PR_REF = re.compile(r"^([^/\s#]+/[^/\s#]+)#(\d+)$")
GAME_SIZE = 4
WAIT_BUCKETS = ("within an hour", "1 hour to 1 day", "1 to 7 days", "after more than 7 days",
                "closed without a review", "never reviewed, still open")
STALLED_BUCKETS = WAIT_BUCKETS[3:]
SCENARIO_LABELS = {"A": "repos seen in training", "B": "repos never seen"}
SERIES_LABELS = {"FULL": "model, all features",
                 "NO_LABEL_REPLAY": "model, without the repo's slow-rate history",
                 "baseline": "trailing-rate baseline"}


def load(path: Path | None = None) -> pd.DataFrame:
    # resolved from this file, not the working directory, so `streamlit run demo/app.py`
    # works from anywhere
    return pd.read_parquet(path or DATA)


def load_features(path: Path | None = None) -> list[dict]:
    """The 37 model features in model order: feature, label, dtype, rate."""
    return json.loads(Path(path or FEATURES).read_text(encoding="utf-8"))


def repos(prs: pd.DataFrame) -> list[str]:
    return sorted(prs["repo"].unique())


def moment(day: date) -> pd.Timestamp:
    """The chosen day at 00:00 UTC: a maintainer opening the dashboard at the start of the day."""
    return pd.Timestamp(day, tz="UTC")


def format_value(value, dtype: str, rate: bool = False) -> str:
    """Rates and shares to 2 decimals, counts as integers, booleans as yes/no. The one
    formatter: build_demo_data.py writes the drivers text with it, the app shows values with it."""
    if pd.isna(value):
        return "missing"
    if dtype == "bool":
        return "yes" if bool(value) else "no"
    if dtype == "str":
        return str(value)
    if rate:
        return f"{float(value):.2f}"
    return f"{int(round(float(value))):,}"


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def awaiting_review(prs: pd.DataFrame, at: pd.Timestamp) -> pd.DataFrame:
    """PRs opened before `at`, still open, and not yet reviewed. A PR reviewed or closed at
    exactly `at` is no longer waiting."""
    still_open = prs["closed_at"].isna() | (prs["closed_at"] > at)
    unreviewed = prs["first_review_at"].isna() | (prs["first_review_at"] > at)
    return prs[(prs["created_at"] < at) & still_open & unreviewed]


def outcome(rows: pd.DataFrame) -> pd.Series:
    """What actually happened: when the first review came, or that the PR was closed without
    one (most never-reviewed PRs were), and whether the PR stalled."""
    def days(end: pd.Series) -> pd.Series:
        return (end - rows["created_at"]).dt.total_seconds() / 86400

    def text(reviewed: float, closed: float) -> str:
        if not pd.isna(reviewed):
            return f"reviewed after {reviewed:.1f} days"
        if not pd.isna(closed):
            return f"closed after {closed:.1f} days without a review"
        return "never reviewed"

    # built row by row: an empty selection must still give an (empty) column of strings
    return pd.Series([text(r, c) + (", stalled" if slow else "") for r, c, slow
                      in zip(days(rows["first_review_at"]), days(rows["closed_at"]), rows["is_slow"])],
                     index=rows.index, dtype=object)


def ranked(prs: pd.DataFrame, repo: str, at: pd.Timestamp, scenario: str) -> pd.DataFrame:
    """The triage list: one repo's PRs awaiting review at `at`, highest risk first. Ties go to
    the older PR, then the lower number, so the order is deterministic."""
    score = SCORES[scenario]
    rows = awaiting_review(prs[prs["repo"] == repo], at)
    rows = rows.sort_values([score, "created_at", "number"], ascending=[False, True, True],
                            kind="mergesort")
    return pd.DataFrame({
        "rank": range(1, len(rows) + 1),
        "risk": rows[score].to_numpy(),
        "url": rows["url"].to_numpy(),
        "title": rows["title"].to_numpy(),
        "days_waited": ((at - rows["created_at"]).dt.total_seconds() / 86400).to_numpy(),
        "drivers": rows[DRIVERS[scenario]].to_numpy(),
        "outcome": outcome(rows).to_numpy(),
        "stalled": rows["is_slow"].to_numpy(),
    })


def tally(listing: pd.DataFrame, k: int = TOP_K) -> tuple[int, int]:
    """(how many of the top k stalled, k), with k cut to the list's length."""
    top = listing.head(k)
    return int(top["stalled"].sum()), len(top)


def default_repo(prs: pd.DataFrame, at: pd.Timestamp) -> str:
    """The repo with the most PRs awaiting review at `at`; ties go alphabetically."""
    counts = awaiting_review(prs, at).groupby("repo").size()
    if counts.empty:
        return repos(prs)[0]
    return min(counts.index, key=lambda r: (-counts[r], r))


def parse_pr_ref(text: str, default_repo: str) -> tuple[str, int] | None:
    """A PR link, `owner/repo#N`, or a bare number (looked up in default_repo); else None."""
    text = text.strip()
    link = PR_LINK.search(text) or PR_REF.match(text)
    if link:
        return link.group(1), int(link.group(2))
    bare = text.lstrip("#")
    return (default_repo, int(bare)) if bare.isdigit() else None


def find_pr(prs: pd.DataFrame, repo: str, number: int) -> pd.DataFrame:
    """The replayed PR as a one-row frame (empty if it is not in the replay). GitHub treats
    owner/repo case-insensitively, so this does too."""
    return prs[(prs["repo"].str.lower() == repo.lower()) & (prs["number"] == number)]


def rank_in_repo(prs: pd.DataFrame, pr: pd.DataFrame, scenario: str) -> tuple[float, int]:
    """(share of the repo's other replayed PRs that scored strictly lower, how many others)."""
    score = SCORES[scenario]
    row = pr.iloc[0]
    others = prs.loc[(prs["repo"] == row["repo"]) & (prs["pr_id"] != row["pr_id"]), score]
    return (float((others < row[score]).mean()) if len(others) else 0.0), len(others)


def random_pr(prs: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    """One replayed PR at random, as a one-row frame."""
    return prs.sample(1, random_state=seed)


MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-!|>~<])")


def md_escape(text: str) -> str:
    """Backslash-escape Markdown's special characters, so a PR title renders exactly as typed."""
    return MD_SPECIAL.sub(r"\\\1", text)


# ---------------------------------------------------------------------------
# chapter 1: how long PRs waited
# ---------------------------------------------------------------------------

def wait_buckets(prs: pd.DataFrame) -> pd.DataFrame:
    """How long each PR waited for its first review. The last three buckets are exactly the
    PRs that stalled (no first review within 7 days), so their counts sum to the stalled count."""
    hours = (prs["first_review_at"] - prs["created_at"]).dt.total_seconds() / 3600
    unreviewed = prs["first_review_at"].isna()
    bucket = pd.Series(None, index=prs.index, dtype=object)
    bucket[hours < 1] = WAIT_BUCKETS[0]
    bucket[(hours >= 1) & (hours < 24)] = WAIT_BUCKETS[1]
    bucket[(hours >= 24) & (hours <= 168)] = WAIT_BUCKETS[2]
    bucket[hours > 168] = WAIT_BUCKETS[3]
    bucket[unreviewed & prs["closed_at"].notna()] = WAIT_BUCKETS[4]
    bucket[unreviewed & prs["closed_at"].isna()] = WAIT_BUCKETS[5]
    counts = bucket.value_counts().reindex(list(WAIT_BUCKETS), fill_value=0)
    return pd.DataFrame({"bucket": list(WAIT_BUCKETS), "count": counts.to_numpy(),
                         "share": counts.to_numpy() / max(len(prs), 1),
                         "stalled": [b in STALLED_BUCKETS for b in WAIT_BUCKETS]})


# ---------------------------------------------------------------------------
# chapter 2: one repo's PRs over time
# ---------------------------------------------------------------------------

def timeline(prs: pd.DataFrame, repo: str, scenario: str, at: pd.Timestamp) -> pd.DataFrame:
    """One row per replayed PR in the repo, with its risk under the chosen model and whether it
    was awaiting review at `at` (the same rule as the triage list)."""
    rows = prs[prs["repo"] == repo]
    return pd.DataFrame({
        "pr_id": rows["pr_id"].to_numpy(),
        "number": rows["number"].to_numpy(),
        "title": rows["title"].to_numpy(),
        "created_at": rows["created_at"].to_numpy(),
        "risk": rows[SCORES[scenario]].to_numpy(),
        "stalled": rows["is_slow"].to_numpy(),
        "waiting": rows.index.isin(awaiting_review(rows, at).index),
        "outcome": outcome(rows).to_numpy(),
    })


def selected_pr_id(event) -> str | None:
    """The pr_id clicked in the timeline chart, read from Streamlit's selection event (a point
    selection named 'pick' on pr_id), or None when nothing is selected."""
    try:
        points = event["selection"]["pick"]
    except (KeyError, TypeError):
        return None
    return points[0].get("pr_id") if points else None


# ---------------------------------------------------------------------------
# chapter 3: why one PR scored as it did, and what the model leans on overall
# ---------------------------------------------------------------------------

def why(pr: pd.DataFrame, scenario: str, features: list[dict], k: int = 8) -> tuple[pd.DataFrame, float]:
    """The k features that pushed this PR's score most (exact TreeSHAP, on the model's raw
    log-odds), largest first, then one row for all the others, and the base value. By SHAP's
    additivity, base + the pushes = the model's log-odds for this PR."""
    row, s = pr.iloc[0], scenario.lower()
    meta = {f["feature"]: f for f in features}
    push = pd.Series({f: float(row[f"shap_{s}__{f}"]) for f in meta})
    top = list(push.abs().sort_values(ascending=False, kind="mergesort").index[:k])
    rows = [{"label": meta[f]["label"],
             "value": format_value(row[f"x__{f}"], meta[f]["dtype"], meta[f]["rate"]),
             "push": push[f]} for f in top]
    rows.append({"label": f"all other {len(meta) - len(top)} features", "value": "",
                 "push": float(push.drop(top).sum())})
    return pd.DataFrame(rows), float(row[f"base_{s}"])


def importance(scenario: str, features: list[dict], template: str | None = None) -> pd.DataFrame:
    """Phase 6's share of mean |SHAP| per feature, for the chosen model, largest first."""
    imp = pd.read_csv((template or IMPORTANCE).format(scenario))
    labels = {f["feature"]: f["label"] for f in features}
    return (imp.assign(label=imp["feature"].map(labels))
            .sort_values("share", ascending=False, kind="mergesort")
            .reset_index(drop=True)[["feature", "label", "share"]])


# ---------------------------------------------------------------------------
# chapter 4: the game
# ---------------------------------------------------------------------------

def week(created_at: pd.Series) -> pd.Series:
    """ISO year and week of opening, in UTC."""
    return created_at.dt.strftime("%G-W%V")


def eligible_weeks(prs: pd.DataFrame) -> list[tuple[str, str]]:
    """The (repo, week) groups a round can come from: at least one PR that stalled and at least
    three that did not."""
    stats = prs.assign(week=week(prs["created_at"])).groupby(["repo", "week"])["is_slow"].agg(["sum", "size"])
    keep = stats[(stats["sum"] >= 1) & (stats["size"] - stats["sum"] >= GAME_SIZE - 1)]
    return list(keep.index)


def deal(prs: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    """One round: an eligible (repo, week) chosen uniformly, then one PR from it that stalled
    and three that did not, chosen uniformly, in shuffled order."""
    repo, wk = rng.choice(eligible_weeks(prs))
    group = prs[(prs["repo"] == repo) & (week(prs["created_at"]) == wk)]
    stalled = sorted(group.loc[group["is_slow"], "pr_id"])
    fine = sorted(group.loc[~group["is_slow"], "pr_id"])
    ids = [rng.choice(stalled)] + rng.sample(fine, GAME_SIZE - 1)
    rng.shuffle(ids)
    return prs.set_index("pr_id").loc[ids].reset_index()


def model_pick(round_: pd.DataFrame, scenario: str) -> str:
    """The PR the model ranks riskiest; ties go to the lower number."""
    order = round_.sort_values([SCORES[scenario], "number"], ascending=[False, True], kind="mergesort")
    return str(order["pr_id"].iloc[0])


def game_hit_rate(prs: pd.DataFrame, scenario: str) -> float:
    """The exact chance, under deal()'s draw, that model_pick is the PR that stalled. Per eligible
    week, each stalled PR s wins when all three PRs drawn with it rank below it: C(L_s, 3) / C(N, 3),
    where N is the week's PRs that did not stall and L_s how many of those model_pick ranks below
    s (a lower score, or the same score and a higher number). Averaged as deal() draws: over
    stalled PRs within a week, then over weeks."""
    score = SCORES[scenario]
    eligible = set(eligible_weeks(prs))
    rates = []
    for key, g in prs.assign(week=week(prs["created_at"])).groupby(["repo", "week"]):
        if key not in eligible:
            continue
        fine = g[~g["is_slow"]]
        n = len(fine)
        wins = [math.comb(int(((fine[score] < s[score]) |
                               ((fine[score] == s[score]) & (fine["number"] > s["number"]))).sum()),
                          GAME_SIZE - 1) / math.comb(n, GAME_SIZE - 1)
                for _, s in g[g["is_slow"]].iterrows()]
        rates.append(sum(wins) / len(wins))
    return sum(rates) / len(rates)


# ---------------------------------------------------------------------------
# chapter 5: the measured results, from Phase 4's committed runs
# ---------------------------------------------------------------------------

def results(path: Path | str | None = None) -> dict:
    """AUC-PR per scenario and feature set (mean over folds, plus each fold), Scenario A's P@10
    against the baseline and a random pick, and the headline strings, formatted the way
    writeup_claims.py registers them (Scenario B is the mean over its five folds)."""
    runs = json.loads(Path(path or RUNS).read_text(encoding="utf-8"))["runs"]

    def pick(scenario: str, featureset: str) -> list[dict]:
        return sorted((r for r in runs if r["scenario"] == scenario and r["featureset"] == featureset),
                      key=lambda r: r["fold"])

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    folds = []
    for sc in ("A", "B"):
        full, nlr = pick(sc, "FULL"), pick(sc, "NO_LABEL_REPLAY")
        for key, values in (("FULL", [r["auc_pr"] for r in full]),
                            ("NO_LABEL_REPLAY", [r["auc_pr"] for r in nlr]),
                            ("baseline", [r["baseline_auc_pr"] for r in full])):
            folds += [{"scenario": SCENARIO_LABELS[sc], "series": SERIES_LABELS[key], "fold": i,
                       "auc_pr": v, "folds": len(values)} for i, v in enumerate(values)]
    folds = pd.DataFrame(folds)
    means = (folds.groupby(["scenario", "series"], sort=False)
             .agg(auc_pr=("auc_pr", "mean"), folds=("folds", "first")).reset_index())
    a = pick("A", "FULL")[0]
    b_full, b_nlr = pick("B", "FULL"), pick("B", "NO_LABEL_REPLAY")
    b_base = mean([r["baseline_auc_pr"] for r in b_full])
    text = {
        "a_p10": f"{a['precision_at_10']:.3f} [{a['p10_ci_lo']:.3f}, {a['p10_ci_hi']:.3f}]",
        "a_baseline_p10": f"{a['baseline_p10']:.3f}",
        "a_random_p10": f"{a['base_rate_p10']:.3f}",
        "b_full_vs_baseline": f"{mean([r['auc_pr'] for r in b_full]):.3f} vs {b_base:.3f}",
        "b_nlr_vs_baseline": f"{mean([r['auc_pr'] for r in b_nlr]):.3f} vs {b_base:.3f}",
    }
    p10 = pd.DataFrame([
        {"series": "model", "p10": a["precision_at_10"], "lo": a["p10_ci_lo"], "hi": a["p10_ci_hi"],
         "label": text["a_p10"]},
        {"series": "trailing-rate baseline", "p10": a["baseline_p10"], "lo": None, "hi": None,
         "label": text["a_baseline_p10"]},
        {"series": "random pick", "p10": a["base_rate_p10"], "lo": None, "hi": None,
         "label": text["a_random_p10"]},
    ])
    return {"folds": folds, "means": means, "p10": p10, "text": text}

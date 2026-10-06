"""The workflow view's logic: how the project was built, read from committed files.

Kept free of Streamlit so it can be tested directly. Every number the "How it was built" pages
show is computed here at runtime from committed files (workflow-view spec, section 4): the
cohort files, the phase gates and results under data/, the Phase 0 and feature-group files under
demo/data/ (built by build_workflow_data.py), and the demo's extract. The deployed app cannot
import the project's own modules, so the few constants it needs are copied below, and a test
checks each against its source. Imports pandas and the standard library only."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
COHORT = ROOT / "data" / "cohort" / "cohort.json"
KEPT = ROOT / "data" / "cohort" / "kept.json"
POOL = ROOT / "data" / "cohort" / "candidate_pool.json"
SEARCH = ROOT / "data" / "cohort" / "raw_search"
PHASE2_ROWS = ROOT / "data" / "phase2_rows.json"
GATE = str(ROOT / "data" / "phase{}_gate.json")
LIVE = ROOT / "data" / "phase3_gate5_live.json"
RUNS = ROOT / "data" / "phase4_runs.json"
TRANSFER = ROOT / "data" / "phase6b_transfer.csv"
WORST50 = ROOT / "data" / "phase6_worst50.csv"
README = ROOT / "README.md"
PHASE0 = HERE / "data" / "phase0_gate.json"
GROUPS = HERE / "data" / "feature_groups.json"

# Copied from the project's modules, which the deployed app cannot import. A test compares each.
CUTOFF_A = pd.Timestamp("2026-01-01T00:00:00Z")        # splits.CUTOFF_A
CAP_FRAC = 0.05                                        # splits.CAP_FRAC
THRESHOLD_H = 168.0                                    # replay.History.threshold_h
TRAILING_DAYS = 90                                     # replay.History.trailing_days
ALPHA = 5.0                                            # replay.History.features_at's alpha
VERDICT_NAMES = ("SUPPORTED", "PARTIAL_SHAP_ONLY", "PARTIAL_INTERVENTION_ONLY", "NOT_SUPPORTED",
                 "CONFLICTING", "CONTRADICTED")       # fingerprint.VERDICTS' values
TOLERANCE = 1e-9

GATED = ("2", "3", "4", "6", "6b")
PHASE_NAMES = {"0": "Phase 0 (pilot)", "2": "Phase 2", "3": "Phase 3", "4": "Phase 4",
               "6": "Phase 6", "6b": "Phase 6b"}
MISSED = "expectation missed, not a stop"
GROUP_LABELS = {"static": "Static: fixed when the PR opens",
                "reconstructed": "Rebuilt to its value at open",
                "replay": "Replayed from earlier PRs",
                "snapshot": "2026 snapshot of the repo"}
BOT_REPOS = ("kubernetes/autoscaler", "kubernetes-sigs/gateway-api-inference-extension")
COUNTED, UNKNOWN, LEAK, OLD = ("counted", "outcome not yet knowable", "counted, though not yet knowable",
                               "outside the 90-day window")
STALLED_LANE, FINE_LANE, UNKNOWN_LANE = "stalled", "reviewed within 7 days", "not yet known at t"


def _json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def picked(event, param: str, field: str | None = None):
    """The first point a chart selection `param` holds (a dict), or its `field`; None if empty."""
    try:
        points = event["selection"][param]
    except (KeyError, TypeError):
        return None
    if not points:
        return None
    return points[0] if field is None else points[0].get(field)


# ---------------------------------------------------------------------------
# the six stages
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    phases: str
    summary: str                              # number-free; the page adds runtime numbers
    artifacts: tuple[tuple[str, bool, str], ...]   # (repo path, committed?, what it holds)
    gates: tuple[str, ...]                    # gate phases; "0" is the Phase 0 pilot
    docs: tuple[str, ...]                     # repo-relative documents
    page: str | None                          # url_path of the deep-dive page
    chapter: int | None                       # index of the story chapter that uses it (the first
                                              # page's url_path is empty, so not looked up by path)


STAGES: tuple[Stage, ...] = (
    Stage("collect", "Collect", "Phases 0–1",
          "Draw public GitHub repos in nine language × star-tier groups with a seeded rule that "
          "uses only structural criteria, then collect every PR's history. A two-repo pilot gate "
          "first decided which GitHub fields are safe at open time.",
          (("data/cohort/cohort.json", True, "the seeded draw: repos picked and rejected, with reasons"),
           ("data/cohort/candidate_pool.json", True, "every candidate the searches returned"),
           ("data/cohort/raw_search/", True, "the search responses, verbatim"),
           ("data/raw/", False, "the collected GitHub responses, gzipped"),
           ("data/processed/", False, "parsed PR, review, comment and timeline tables")),
          ("0",), ("docs/data_collection.md", "docs/phase0_gate_results.md"), "data-funnel", None),
    Stage("label", "Label", "Phase 2",
          "Define a PR's first review (D5): the first review or comment by a human other than the "
          "author, with a tie to the repo. A PR is slow without one within 7 days. Repos are "
          "dropped on structural rules only, and the trailing-rate baseline is fixed.",
          (("data/cohort/kept.json", True, "per-repo QC: kept or dropped, with D3 and D5 slow rates"),
           ("data/phase2_gate.json", True, "the Phase 2 checks"),
           ("docs/phase2_eda.md", True, "label-definition and threshold sensitivity")),
          ("2",), ("docs/phase2_eda.md",), None, 0),
    Stage("features", "Features", "Phase 3",
          "Turn each PR into model features using only what was knowable when it opened: fields "
          "that change later are rebuilt to their values at open, and history features replay "
          "the repo in time order.",
          (("docs/feature_dictionary.md", True, "every column: group, status and derivation"),
           ("data/phase3_gate.json", True, "the Phase 3 checks, including the replay audit"),
           ("demo/data/prs.parquet", True, "the test-period PRs with feature values, scores and SHAP"),
           ("data/features/", False, "the feature table, one row per PR")),
          ("3",), ("docs/feature_dictionary.md",), "known-at-time-t", 1),
    Stage("evaluate", "Train & test", "Phase 4",
          "Hold data out two ways: future PRs of repos seen in training, and whole repos never "
          "seen. One LightGBM configuration was tuned once and frozen, then trained per test "
          "design and feature set, and scored against the baseline.",
          (("data/phase4_runs.json", True, "every run's metrics, intervals and repo lists"),
           ("data/phase4_params.json", True, "the tuned parameters and every tuning trial"),
           ("data/experiments.csv", True, "the run ledger"),
           ("data/models/", False, "the trained boosters"),
           ("data/predictions/", False, "per-PR test predictions")),
          ("4",), ("docs/phase4_results.md",), "test-designs", 4),
    Stage("explain", "Explain", "Phases 6 and 6b",
          "Explain the frozen models with exact TreeSHAP, read the most confident errors, check "
          "newcomers are not treated worse, and test one explanation for the cold-start gap under "
          "a decision rule written down before any code ran.",
          (("data/phase6_importance_A.csv", True, "each feature's share of the attribution"),
           ("data/phase6_worst50.csv", True, "the most confident errors"),
           ("data/phase6_fairness.csv", True, "first-time vs repeat contributors"),
           ("data/phase6b_transfer.csv", True, "the pre-registered transfer test, per repo")),
          ("6", "6b"), ("docs/phase6_interpretation.md", "docs/phase6b_fingerprinting.md"), None, 2),
    Stage("ship", "Ship", "Phases 7–8",
          "Write it up and build this demo. The results the README and the report cite are "
          "registered and recomputed from committed files by a test, and the demo reads only "
          "committed files.",
          (("README.md", True, "the summary"),
           ("docs/REPORT.md", True, "the full report"),
           ("writeup_claims.py", True, "every cited number, recomputed by a test"),
           ("demo/app.py", True, "this demo")),
          (), ("docs/REPORT.md",), None, 5),
)
STAGE_KEYS = tuple(s.key for s in STAGES)


def stage(key: str) -> Stage:
    return next(s for s in STAGES if s.key == key)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def gates() -> pd.DataFrame:
    """Every recorded check: Phase 0's ten pilot checks, then five for each gated phase."""
    rows = [{"phase": "0", "id": r["id"], "check": r["check"], "value": r["detail"],
             "passed": r["outcome"] == "pass", "status": r["outcome"]} for r in _json(PHASE0)]
    for ph in GATED:
        rows += [{"phase": ph, "id": c["id"], "check": c["check"], "value": c["value"],
                  "passed": bool(c["pass"]), "status": "pass" if c["pass"] else "fail"}
                 for c in _json(Path(GATE.format(ph)))]
    return pd.DataFrame(rows)


def gate_value_text(value) -> str:
    """A recorded check value, readable: dicts as key: value, lists joined, empties as none."""
    if isinstance(value, dict):
        return "; ".join(f"{k}: {gate_value_text(v)}" for k, v in value.items()) or "none"
    if isinstance(value, list):
        return ", ".join(gate_value_text(v) for v in value) or "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def gate_line(phases: tuple[str, ...], table: pd.DataFrame) -> str:
    if not phases:
        return "claims tests"
    if phases == ("0",):
        return f"pilot gate: {int((table['phase'] == '0').sum())} checks"
    rows = table[table["phase"].isin(phases)]
    return f"{int(rows['passed'].sum())}/{len(rows)} checks pass"


def live_mismatches() -> list[dict]:
    """The live GitHub spot check's disagreements: repo#number, field, live and ours."""
    return [{"pr": f"{r['repo']}#{r['number']}", "field": f, "live": v["live"], "ours": v["ours"]}
            for r in _json(LIVE) if not r["match"]
            for f, v in r["checks"].items() if v["live"] != v["ours"]]


def audits() -> dict:
    """Phase 3's two audits: the brute-force replay audit and the live GitHub spot check."""
    g3 = {c["id"]: c["value"] for c in _json(Path(GATE.format("3")))}
    return {"audit_n": g3[2]["n"], "audit_max_diff": g3[2]["max_abs_diff"],
            "live_rows": len(g3[5]["rows"]), "live_checks": g3[5]["checks"],
            "live_matched": g3[5]["matched"]}


# ---------------------------------------------------------------------------
# the funnel and the cohort
# ---------------------------------------------------------------------------

def _cells() -> dict[str, int]:
    """Each search group (e.g. Go:200-800) and how many repos its search matched."""
    out = {}
    for f in sorted(SEARCH.glob("search_*_p0.json")):
        _, lang, lo, hi, _ = f.stem.split("_")
        out[f"{lang}:{lo}-{hi}"] = _json(f)["data"]["search"]["repositoryCount"]
    return out


def repo_funnel() -> pd.DataFrame:
    cohort, kept = _json(COHORT), _json(KEPT)
    return pd.DataFrame([
        {"step": "matched the searches", "count": sum(_cells().values()), "source": "data/cohort/raw_search/",
         "why": f"public, non-fork, non-archived repos pushed since {cohort['pushed_since']}, in "
                f"{len(cohort['languages'])} languages × {len(cohort['star_tiers'])} star tiers"},
        {"step": "pooled as candidates", "count": cohort["n_candidates_pooled"],
         "source": "data/cohort/candidate_pool.json",
         "why": "the first results of each group's search, which came back sorted by stars"},
        {"step": "selected", "count": cohort["n_selected"], "source": "data/cohort/cohort.json",
         "why": f"a seeded random draw of {cohort['per_cell']} per group, skipping repos with fewer "
                f"than {cohort['min_window_prs']} PRs"},
        {"step": "kept", "count": len(kept["kept"]), "source": "data/cohort/kept.json",
         "why": "structural QC only: bot share, human PR count and language"},
    ])


def pr_funnel() -> pd.DataFrame:
    reps = pd.DataFrame(_json(KEPT)["repos"])
    kept = reps[reps["kept"]]
    runs = pd.DataFrame(_json(RUNS)["runs"])
    full = runs[runs["featureset"] == "FULL"]
    a = full[full["scenario"] == "A"].iloc[0]
    start, end = _json(COHORT)["window"]
    return pd.DataFrame([
        {"step": "in the window, selected repos", "count": int(reps["n_in_window"].sum()),
         "source": "data/cohort/kept.json", "why": f"PRs opened from {start} to {end}"},
        {"step": "in the kept repos", "count": int(kept["n_in_window"].sum()),
         "source": "data/cohort/kept.json", "why": "PRs of the repos QC dropped removed"},
        {"step": "human-authored", "count": int(kept["n_human"].sum()), "source": "data/phase2_rows.json",
         "why": "bot-authored PRs removed; every remaining PR gets a label and features"},
        {"step": "modelled", "count": int(full.loc[full["scenario"] == "B", "n_test"].sum()),
         "source": "data/phase4_runs.json",
         "why": "PRs whose timeline may be truncated removed; each is tested once across the unseen-repo folds"},
        {"step": "training rows, seen-repo model", "count": int(a["n_train"]), "source": "data/phase4_runs.json",
         "why": f"PRs opened before {CUTOFF_A:%Y-%m-%d}, capped at {CAP_FRAC:.0%} of training rows per repo"},
        {"step": "test PRs, this demo", "count": int(a["n_test"]), "source": "data/phase4_runs.json",
         "why": f"PRs opened from {CUTOFF_A:%Y-%m-%d} to {end}, never capped"},
    ])


def removed(step: str) -> tuple[pd.DataFrame, str]:
    """What a funnel step removed, and why: a table where one exists, and a sentence."""
    cohort, reps = _json(COHORT), pd.DataFrame(_json(KEPT)["repos"])
    dropped = reps[~reps["kept"]]
    none = pd.DataFrame()
    if step == "matched the searches":
        return none, "The first step: what GitHub's nine searches matched."
    if step == "pooled as candidates":
        per = cohort["n_candidates_pooled"] // len(_cells())
        return none, (f"Only the first {per} results of each group's search were fetched. They came "
                      "back sorted by stars, so the pool is each group's most-starred repos.")
    if step == "selected":
        frame = pd.DataFrame(cohort["rejected"])[["repo", "cell", "reason"]]
        return frame, (f"The draw visited each group's pool in seeded random order and skipped "
                       f"{len(frame)} repos with too few PRs until it had {cohort['per_cell']}; "
                       "the rest of the pool was never visited.")
    if step == "kept":
        frame = dropped.assign(reason=dropped["reasons"].map("; ".join))[["repo", "cell", "reason"]]
        return frame, f"{len(frame)} repos failed the structural QC rules."
    if step == "in the window, selected repos":
        return none, "The first step: every PR the selected repos opened in the window."
    if step == "in the kept repos":
        return dropped[["repo", "n_in_window"]], "The window PRs of the repos QC dropped."
    if step == "human-authored":
        kept = reps[reps["kept"]]
        frame = kept.assign(bot_authored=kept["n_in_window"] - kept["n_human"])[["repo", "bot_authored"]]
        return frame[frame["bot_authored"] > 0], "Bot-authored PRs in the kept repos."
    if step == "modelled":
        return none, "Rows flagged as possibly truncated timelines were left out of every model."
    if step == "training rows, seen-repo model":
        rows = split_rows()
        frame = rows[rows["capped_out"] > 0][["repo", "pre_2026", "trained", "capped_out"]]
        return frame, (f"Each repo contributes at most {CAP_FRAC:.0%} of the training rows, so a few "
                       "very active repos cannot dominate the model.")
    if step == "test PRs, this demo":
        return none, "Every PR opened from January to June 2026 in the kept repos is a test PR."
    raise KeyError(step)


def pool_bias() -> pd.DataFrame:
    """Per search group: how many repos matched, how many were pooled, and their star range."""
    pool = pd.DataFrame(_json(POOL))
    out = (pool.groupby("_cell")["stargazerCount"].agg(pooled="size", stars_min="min", stars_max="max")
           .reset_index().rename(columns={"_cell": "group"}))
    out["matched"] = out["group"].map(_cells())
    out["share"] = out["pooled"] / out["matched"]
    return out[["group", "matched", "pooled", "share", "stars_min", "stars_max"]]


def label_rates() -> dict:
    """Global D5 and D3 slow rates over the kept repos, and the two CI-bot repos' D5 rates."""
    reps = pd.DataFrame(_json(KEPT)["repos"])
    kept = reps[reps["kept"]]
    d5 = next(c["value"] for c in _json(Path(GATE.format("2"))) if c["check"].startswith("D5 global"))
    d3 = float((kept["is_slow_d3"] * kept["n_human"]).sum() / kept["n_human"].sum())
    bots = reps.set_index("repo").loc[list(BOT_REPOS), "is_slow_d5"]
    return {"d5": float(d5), "d3": d3, "bot_repos": {r: float(v) for r, v in bots.items()}}


def bot_note(labels: dict) -> str:
    """The CI-bot disclosure (workflow-view spec, section 8), with the two repos' D5 rates."""
    a, b = (labels["bot_repos"][r] for r in BOT_REPOS)
    return ("**A CI bot counts as a reviewer.** Kubernetes' `k8s-ci-robot` account is registered on "
            "GitHub as an ordinary user, not a bot, so the label counts its automated comments as first "
            f"reviews. It supplies most first reviews in {BOT_REPOS[0]} and {BOT_REPOS[1]}, so their "
            f"slow rates under D5 ({a:.1%} and {b:.1%}) understate how long people took. Fixing it means "
            "relabelling and re-running every later phase.")


def pool_note(pool: pd.DataFrame) -> str:
    """The pool-bias disclosure (workflow-view spec, section 8), with the pools' share of their groups."""
    return ("**Each group's candidates are its most-starred repos.** The searches returned each group's "
            f"matches sorted by stars, and the pool took the first {int(pool['pooled'].iloc[0])} of each, "
            f"between {pool['share'].min():.1%} and {pool['share'].max():.1%} of the group, so the seeded "
            "draw chose among each group's most-starred repos, not across its whole star range.")


def verdict_6b() -> str:
    """The Phase 6b verdict as the README states it (the claims test pins it to its recomputation)."""
    found = set(re.findall(r"`([A-Z_]+)`", README.read_text(encoding="utf-8"))) & set(VERDICT_NAMES)
    if len(found) != 1:
        raise LookupError(f"README.md names {len(found)} Phase 6b verdicts")
    return found.pop()


def feature_groups() -> list[dict]:
    return _json(GROUPS)


def badges(prs: pd.DataFrame, groups: list[dict]) -> dict[str, str]:
    """One headline per stage, computed from committed files."""
    cohort = _json(COHORT)
    return {"collect": f"{cohort['n_selected']} of {cohort['n_candidates_pooled']:,} repos",
            "label": f"{_json(PHASE2_ROWS)['n_rows']:,} labelled PRs",
            "features": f"{len(groups)} features",
            "evaluate": f"{len(_json(RUNS)['runs'])} runs",
            "explain": f"{len(pd.read_csv(WORST50))} errors read",
            "ship": f"{len(prs):,} demo PRs"}


def stage_table(badge: dict[str, str], table: pd.DataFrame) -> pd.DataFrame:
    """The pipeline map's nodes, in order."""
    return pd.DataFrame([{"key": s.key, "order": i, "title": f"{i + 1}. {s.title}", "phases": s.phases,
                          "badge": badge[s.key], "gate": gate_line(s.gates, table)}
                         for i, s in enumerate(STAGES)])


# ---------------------------------------------------------------------------
# the two test designs
# ---------------------------------------------------------------------------

def _runs() -> pd.DataFrame:
    return pd.DataFrame(_json(RUNS)["runs"])


def cap(rows: pd.Series) -> pd.Series:
    """Training rows each repo keeps: at most CAP_FRAC of all training rows (splits.cap_per_repo)."""
    return rows.clip(upper=max(1, int(CAP_FRAC * rows.sum())))


def fold_of() -> dict[str, int]:
    """The Scenario B fold that holds out each repo."""
    b = _runs()
    b = b[(b["scenario"] == "B") & (b["featureset"] == "FULL")]
    return {repo: int(r["fold"]) for _, r in b.iterrows() for repo in r["test_repos"]}


def split_rows() -> pd.DataFrame:
    """Per repo: 2026 test PRs, pre-2026 PRs, how many of those the cap kept for training, and the
    fold that holds the repo out. Counts come from Phase 6b's per-repo table: n_A is a repo's
    Scenario A test rows (2026) and n_B all its rows (Scenario B tests the whole repo)."""
    t = pd.read_csv(TRANSFER)[["repo", "n_A", "n_B"]].rename(columns={"n_A": "test_2026", "n_B": "all_rows"})
    t["pre_2026"] = t["all_rows"] - t["test_2026"]
    t["trained"] = cap(t["pre_2026"])
    t["capped_out"] = t["pre_2026"] - t["trained"]
    t["fold"] = t["repo"].map(fold_of())
    return t.sort_values("all_rows", ascending=False).reset_index(drop=True)


SEGMENTS = ("trained on", "dropped by the cap", "tested on")


def design_rows(rows: pd.DataFrame, fold: int | None = None) -> pd.DataFrame:
    """Each repo's rows as trained on / dropped by the cap / tested on (long format). With no
    fold, Scenario A: pre-2026 rows train (capped) and 2026 rows test. With a fold, Scenario B:
    the fold's repos are tested whole and every row of the others trains, capped."""
    if fold is None:
        trained, pool, tested = rows["trained"], rows["pre_2026"], rows["test_2026"]
    else:
        held = rows["fold"] == fold
        pool = rows["all_rows"].where(~held, 0)
        trained = cap(rows.loc[~held, "all_rows"]).reindex(rows.index, fill_value=0)
        tested = rows["all_rows"].where(held, 0)
    parts = zip(SEGMENTS, (trained, pool - trained, tested))
    out = pd.concat([pd.DataFrame({"repo": rows["repo"], "segment": seg, "rows": vals.astype(int)})
                     for seg, vals in parts], ignore_index=True)
    return out[out["rows"] > 0].reset_index(drop=True)


def scenario_a() -> dict:
    a = _runs()
    a = a[(a["scenario"] == "A") & (a["featureset"] == "FULL")].iloc[0]
    return {"n_train": int(a["n_train"]), "n_test": int(a["n_test"]),
            "train_repos": len(a["train_repos"]), "test_repos": len(a["test_repos"])}


def fold_results() -> pd.DataFrame:
    """Per Scenario B fold: repos held out, rows, base rate and the three AUC-PRs."""
    b = _runs()
    b = b[b["scenario"] == "B"]
    full = b[b["featureset"] == "FULL"].set_index("fold").sort_index()
    nlr = b[b["featureset"] == "NO_LABEL_REPLAY"].set_index("fold").sort_index()
    return pd.DataFrame({"fold": list(full.index), "repos": list(full["n_test_repos"]),
                         "n_train": list(full["n_train"]), "n_test": list(full["n_test"]),
                         "base_rate": list(full["base_rate"]), "model": list(full["auc_pr"]),
                         "without_history": list(nlr["auc_pr"]),
                         "baseline": list(full["baseline_auc_pr"])})


# ---------------------------------------------------------------------------
# known at time t: the trailing slow rate, re-derived from the extract
# ---------------------------------------------------------------------------

def prior(prs: pd.DataFrame) -> float:
    """The shrinkage prior, read off the PRs whose 90-day window held no knowable outcome."""
    empty = prs.loc[prs["x__trailing_n"] == 0, "x__trailing_90d_slow_rate"]
    if empty.empty:
        raise LookupError("no PR with an empty trailing window to read the prior from")
    return float(empty.iloc[0])


def shrunk(slow: int, k: int, prior_rate: float) -> float:
    return (slow + ALPHA * prior_rate) / (k + ALPHA)


def replay_at(prs: pd.DataFrame, repo: str, pr_id: str, rule: str = "resolvable",
              prior_rate: float | None = None) -> tuple[pd.DataFrame, dict]:
    """The repo's PRs opened in the 120 days before `pr_id`, each with its status at that moment,
    and the trailing slow rate they give. `rule` is "resolvable" (the project's: a PR counts once
    it is 168h old or reviewed) or "naive" (every earlier PR in the 90-day window)."""
    rows = prs[prs["repo"] == repo].sort_values(["created_at", "pr_id"], kind="mergesort")
    me = rows[rows["pr_id"] == pr_id].iloc[0]
    t = me["created_at"]
    before = rows[(rows["created_at"] < t) & (rows["created_at"] >= t - pd.Timedelta(days=TRAILING_DAYS + 30))]
    window = before["created_at"] >= t - pd.Timedelta(days=TRAILING_DAYS)
    knowable = (before["created_at"] <= t - pd.Timedelta(hours=THRESHOLD_H)) | (before["first_review_at"] < t)
    counted = window & (knowable | (rule == "naive"))
    status = pd.Series(OLD, index=before.index)
    status[window & ~knowable] = LEAK if rule == "naive" else UNKNOWN
    status[window & knowable] = COUNTED
    lane = before["is_slow"].map({True: STALLED_LANE, False: FINE_LANE})
    lane[status == UNKNOWN] = UNKNOWN_LANE
    g = prior(prs) if prior_rate is None else prior_rate
    k, slow = int(counted.sum()), int(before.loc[counted, "is_slow"].sum())
    rate = shrunk(slow, k, g)
    summary = {"t": t, "number": int(me["number"]), "k": k, "slow": slow, "prior": g, "rate": rate,
               "unknown": int((window & ~knowable).sum()), "stored_k": int(me["x__trailing_n"]),
               "stored_rate": float(me["x__trailing_90d_slow_rate"])}
    summary["matches"] = k == summary["stored_k"] and abs(rate - summary["stored_rate"]) <= TOLERANCE
    frame = before[["pr_id", "number", "created_at", "first_review_at", "is_slow"]].assign(status=status, lane=lane)
    return frame.reset_index(drop=True), summary


def trailing_check(prs: pd.DataFrame, prior_rate: float | None = None) -> pd.DataFrame:
    """Every PR's trailing slow rate re-derived from the extract, beside the stored feature.
    Per repo: PRs at least 168h old come from sorted positions and running sums; only the last
    week's PRs are checked one by one for a review before t."""
    g = prior(prs) if prior_rate is None else prior_rate
    parts = []
    for _, rows in prs.groupby("repo", sort=True):
        rows = rows.sort_values(["created_at", "pr_id"], kind="mergesort")
        c = rows["created_at"].reset_index(drop=True)
        lo = c.searchsorted(c - pd.Timedelta(days=TRAILING_DAYS), side="left")
        thr = c.searchsorted(c - pd.Timedelta(hours=THRESHOLD_H), side="right")
        now = c.searchsorted(c, side="left")
        slow = rows["is_slow"].astype(int).tolist()
        cum = [0]
        for s in slow:
            cum.append(cum[-1] + s)
        reviewed, opened = rows["first_review_at"].tolist(), c.tolist()
        ks, slows, unknown = [], [], []
        for i, t in enumerate(opened):
            recent = [j for j in range(thr[i], now[i]) if reviewed[j] < t]   # NaT compares False
            ks.append(int(thr[i] - lo[i]) + len(recent))
            slows.append(cum[thr[i]] - cum[lo[i]] + sum(slow[j] for j in recent))
            unknown.append(int(now[i] - thr[i]) - len(recent))
        part = rows[["repo", "pr_id", "number", "created_at", "x__trailing_n", "x__trailing_90d_slow_rate"]].copy()
        part["k"], part["slow"], part["unknown"] = ks, slows, unknown
        part["rate"] = [shrunk(s, k, g) for s, k in zip(slows, ks)]
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["matches"] = ((out["k"] == out["x__trailing_n"])
                      & ((out["rate"] - out["x__trailing_90d_slow_rate"]).abs() <= TOLERANCE))
    return out


def exact_repos(check: pd.DataFrame) -> list[str]:
    """Repos where the re-derived rate equals the stored feature on every PR."""
    ok = check.groupby("repo")["matches"].all()
    return sorted(ok[ok].index)


def default_pr(check: pd.DataFrame, repo: str) -> str:
    """The repo's PR with the most earlier PRs whose outcome was not yet knowable (the earliest
    on a tie): the moment the 7-day rule matters most."""
    rows = check[check["repo"] == repo].sort_values(["unknown", "created_at"], ascending=[False, True],
                                                     kind="mergesort")
    return rows["pr_id"].iloc[0]


# ---------------------------------------------------------------------------
# everything the section shows, built once
# ---------------------------------------------------------------------------

def bundle(prs: pd.DataFrame) -> dict:
    groups, table = feature_groups(), gates()
    check = trailing_check(prs)
    return {"stages": stage_table(badges(prs, groups), table), "gates": table, "groups": groups,
            "repo_funnel": repo_funnel(), "pr_funnel": pr_funnel(), "pool_bias": pool_bias(),
            "labels": label_rates(), "audits": audits(), "live": live_mismatches(),
            "prior": prior(prs), "check": check, "exact": exact_repos(check),
            "splits": split_rows(), "folds": fold_results(), "scenario_a": scenario_a(),
            "verdict": verdict_6b()}

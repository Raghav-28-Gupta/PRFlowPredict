"""Phase 7: build the demo's committed extract, demo/data/prs.parquet, and its feature list,
demo/data/features.json.

Runs locally only. It reads Phase 4's gitignored outputs (predictions, boosters, the feature
table, parsed PRs) and writes small committed files, so the deployed app needs no model:

    python build_demo_data.py

One row per Scenario A test PR, in the A predictions file's row order. Phase 4's P@10 breaks
ties with a seeded permutation over that order, so keeping it is what lets
tests/test_demo_extract.py reproduce Phase 4's numbers from the extract. Each row also carries
every feature's exact TreeSHAP value under both models, the explainer's base value and the
feature values themselves, so the app's "why" chart adds up to the score it shows. Author
identities are deliberately left out: the demo is about PRs, not people."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import attribution
import experiment as ex
import features
import featuresets as fs
import model
from demo import triage

ROOT = Path(__file__).parent
OUT = ROOT / "demo" / "data" / "prs.parquet"
FEATURES_JSON = "features.json"                     # written next to the extract
FULL = fs.FEATURE_SETS["FULL"]
ADDITIVITY_TOL = 1e-6
RATES = {"trailing_90d_slow_rate", "prior_merge_rate_here", "author_prior_slow_rate_here"}
BASE_COLUMNS = ["repo", "pr_id", "number", "url", "title", "created_at", "closed_at",
                "first_review_at", "is_slow", "score_a", "fold_b", "score_b", "baseline_score",
                "drivers_a", "drivers_b"]
COLUMNS = (BASE_COLUMNS + ["base_a", "base_b"] + [f"shap_a__{c}" for c in FULL]
           + [f"shap_b__{c}" for c in FULL] + [f"x__{c}" for c in FULL])

# A reader-facing name for every FULL feature (tests/test_build_demo_data.py checks coverage).
DRIVER_LABELS = {
    "created_hour_utc": "hour opened (UTC)",
    "created_dayofweek": "weekday opened (Mon=0)",
    "is_weekend": "opened at the weekend",
    "is_cross_repository": "from a fork",
    "author_account_age_days": "author's GitHub account age (days)",
    "body_len": "description length",
    "has_body": "has a description",
    "is_draft_at_open": "opened as a draft",
    "n_labels_at_open": "labels at open",
    "title_len_at_open": "title length",
    "base_is_default": "targets the default branch",
    "reviewer_requested_at_open": "reviewer requested at open",
    "n_reviewers_requested_at_open": "reviewers requested at open",
    "requested_team_at_open": "team review requested at open",
    "additions_at_open": "lines added at open",
    "deletions_at_open": "lines deleted at open",
    "n_commits_at_open": "commits at open",
    "open_backlog_at_t": "open PRs in the repo",
    "prs_opened_trailing_7d": "PRs opened in the repo, last 7 days",
    "trailing_90d_slow_rate": "repo's recent slow rate",
    "trailing_n": "PRs behind the repo's slow rate",
    "is_first_pr_here": "author's first PR here",
    "n_prior_prs_here": "author's earlier PRs here",
    "n_prior_merged_here": "author's merged PRs here",
    "prior_merge_rate_here": "author's merge rate here",
    "days_since_first_pr_here": "days since author's first PR here",
    "author_prior_slow_rate_here": "author's past slow rate here",
    "author_prior_n": "PRs behind author's slow rate",
    "n_assignable_users": "maintainers (assignable users)",
    "n_mentionable_users": "community size (mentionable users)",
    "owner_is_org": "owned by an organisation",
    "has_codeowners": "has CODEOWNERS",
    "has_pr_template": "has a PR template",
    "has_contributing": "has CONTRIBUTING",
    "n_ci_workflows": "CI workflows",
    "language_dominant": "main language",
    "repo_age_days_at_open": "repo age (days)",
}


def format_value(feature: str, value) -> str:
    """Rates and shares to 2 decimals, counts as integers, booleans as yes/no (the shared
    formatter in demo/triage.py, so the app shows values exactly as the drivers text does)."""
    return triage.format_value(value, features.COLUMN_SPEC[feature]["dtype"], feature in RATES)


def format_drivers(shap_row: np.ndarray, x_row: pd.Series, k: int = 3) -> str:
    """The k features with the largest |SHAP|, largest first. The arrow is the direction the
    feature pushed the raw margin: up means towards 'slow'."""
    top = np.argsort(-np.abs(shap_row), kind="stable")[:k]
    return " · ".join(
        f"{DRIVER_LABELS[x_row.index[i]]} = {format_value(x_row.index[i], x_row.iloc[i])} "
        f"{'↑' if shap_row[i] > 0 else '↓'}" for i in top)


def feature_list() -> list[dict]:
    """What demo/data/features.json holds: the FULL features in model order."""
    return [{"feature": c, "label": DRIVER_LABELS[c], "dtype": features.COLUMN_SPEC[c]["dtype"],
             "rate": c in RATES} for c in FULL]


def explain(booster, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Exact TreeSHAP (Phase 6's attribution.explain), refusing any additivity failure."""
    sv, ev = attribution.explain(booster, X)
    delta = attribution.additivity_delta(booster, X, sv, ev)
    if delta > ADDITIVITY_TOL:
        raise RuntimeError(f"SHAP additivity failed: max delta {delta:.3g} > {ADDITIVITY_TOL}")
    return sv, ev


def check_logit(sv: np.ndarray, ev, score: np.ndarray, tol: float = ADDITIVITY_TOL) -> None:
    """base + sum(SHAP) must be the logit of the score the extract stores, row by row."""
    score = np.asarray(score, dtype=float)
    gap = np.max(np.abs(np.asarray(ev) + sv.sum(axis=1) - np.log(score / (1 - score))))
    if gap > tol:
        raise ValueError(f"base + SHAP differs from logit(score) by up to {gap:.3g} > {tol}")


def validate(out: pd.DataFrame, fold_of: dict[str, int]) -> None:
    """The build checks of spec section 4, apart from additivity (checked in explain and
    check_logit)."""
    missing = sorted(set(FULL) - set(DRIVER_LABELS))
    if missing:
        raise ValueError(f"no DRIVER_LABELS entry for {missing}")
    for col in ("number", "url", "title", "score_a", "fold_b", "score_b", "drivers_a", "drivers_b",
                "base_a", "base_b"):
        if out[col].isna().any():
            raise ValueError(f"{int(out[col].isna().sum())} rows have no {col}")
    wrong = out["fold_b"] != out["repo"].map(fold_of)
    if wrong.any():
        raise ValueError(f"{int(wrong.sum())} rows are not in the B fold that held their repo out")
    if not out["pr_id"].is_unique:
        raise ValueError("duplicate pr_id")


def _unique_index(df: pd.DataFrame, what: str) -> pd.DataFrame:
    if not df["pr_id"].is_unique:
        raise ValueError(f"duplicate pr_id in {what}")
    return df.set_index("pr_id")


def build(root: Path = ROOT) -> pd.DataFrame:
    a = pd.read_parquet(root / "data" / "predictions" / "A_FULL_fold0.parquet")
    runs = json.loads((root / "data" / "phase4_runs.json").read_text(encoding="utf-8"))["runs"]
    fold_of = {repo: r["fold"] for r in runs
               if r["scenario"] == "B" and r["featureset"] == "FULL" for repo in r["test_repos"]}
    table = ex.load_table(root / "data" / "features" / "features.parquet").set_index("pr_id")
    tier2 = _unique_index(pd.concat(
        pd.read_parquet(root / "data" / "processed" / repo.replace("/", "__") / "pr_tier2.parquet",
                        columns=["pr_id", "number", "url", "title_current", "closed_at"])
        for repo in sorted(a["repo"].unique())), "pr_tier2")
    b = _unique_index(pd.concat(
        pd.read_parquet(root / "data" / "predictions" / f"B_FULL_fold{k}.parquet",
                        columns=["pr_id", "p_hat"]).assign(fold_b=k)
        for k in sorted(set(fold_of.values()))), "B predictions")

    # .map keeps the A file's row order exactly (spec section 4).
    ids = a["pr_id"]
    out = pd.DataFrame({
        "repo": a["repo"], "pr_id": ids,
        "number": ids.map(tier2["number"]), "url": ids.map(tier2["url"]),
        "title": ids.map(tier2["title_current"]),
        "created_at": a["created_at"], "closed_at": ids.map(tier2["closed_at"]),
        "first_review_at": a["created_at"] + pd.to_timedelta(ids.map(table["wait_h"]), unit="h"),
        "is_slow": a["is_slow"], "score_a": a["p_hat"],
        "fold_b": ids.map(b["fold_b"]), "score_b": ids.map(b["p_hat"]),
        "baseline_score": a["baseline_score"],
    })
    if not ids.isin(table.index).all():
        raise ValueError("A test rows missing from the feature table")

    X = table.loc[ids, FULL]
    sva, eva = explain(model.load(root / "data" / "models" / "A_FULL_fold0.txt"), X)
    check_logit(sva, eva, out["score_a"].to_numpy())
    svb, evb = np.full(sva.shape, np.nan), np.full(len(out), np.nan)
    for k in sorted(set(fold_of.values())):
        rows = (out["fold_b"] == k).to_numpy()
        sv, ev = explain(model.load(root / "data" / "models" / f"B_FULL_fold{k}.txt"), X[rows])
        check_logit(sv, ev, out.loc[rows, "score_b"].to_numpy())
        svb[rows], evb[rows] = sv, ev
    out["drivers_a"] = [format_drivers(sva[i], X.iloc[i]) for i in range(len(X))]
    out["drivers_b"] = [format_drivers(svb[i], X.iloc[i]) for i in range(len(X))]
    out["base_a"], out["base_b"] = eva, evb

    values = X.reset_index(drop=True).add_prefix("x__")
    values["x__language_dominant"] = values["x__language_dominant"].astype(str)
    out = pd.concat([out,
                     pd.DataFrame(sva.astype("float32"), columns=[f"shap_a__{c}" for c in FULL]),
                     pd.DataFrame(svb.astype("float32"), columns=[f"shap_b__{c}" for c in FULL]),
                     values], axis=1)
    out["number"] = out["number"].astype("int64")
    out["fold_b"] = out["fold_b"].astype("int64")
    validate(out, fold_of)
    return out[COLUMNS]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=ROOT, help="repo holding the gitignored inputs")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    out = build(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    (args.out.parent / FEATURES_JSON).write_text(json.dumps(feature_list(), indent=2) + "\n",
                                                 encoding="utf-8", newline="\n")
    print(f"wrote {args.out}: {len(out):,} PRs, {out['repo'].nunique()} repos, "
          f"{args.out.stat().st_size / 1e6:.1f} MB, and {FEATURES_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

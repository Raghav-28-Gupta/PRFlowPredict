"""Phase 4 results document + validity gate.

The gate tests VALIDITY (disjoint splits, feature hygiene, the blueprint's ablation ran,
everything logged, reproducible), never success: failing to beat the baseline is a
reportable finding, and the headline section is written to be true either way."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

import experiment as ex
import featuresets as fs
import metrics
import model
import splits
import tracking

log = logging.getLogger("report4")
ROOT = Path(__file__).parent
DOC = ROOT / "docs" / "phase4_results.md"
FIG = ROOT / "figures"
GATE_JSON = ROOT / "data" / "phase4_gate.json"
SETS = list(fs.FEATURE_SETS)
HOUR_BUCKETS = [(0, 6, "00-06"), (6, 12, "06-12"), (12, 18, "12-18"), (18, 24, "18-24")]
N_RUNS = 4 * (1 + 5)


# ----------------------------------------------------------------------------
# Gate (pure)
# ----------------------------------------------------------------------------

def gate_checks(runs: list[dict], params_sha: str, experiments: pd.DataFrame,
                repro_delta: float | None) -> list[dict]:
    # 1. disjointness
    leaks = []
    for r in runs:
        if r["scenario"] == "A":
            if not (pd.Timestamp(r["train_max_created"]) < pd.Timestamp(r["test_min_created"])
                    and pd.Timestamp(r["test_min_created"]) >= splits.CUTOFF_A):
                leaks.append(f"A/{r['featureset']}: time")
        else:
            if set(r["train_repos"]) & set(r["test_repos"]):
                leaks.append(f"B/{r['featureset']}/fold{r['fold']}: repo overlap")
    c1 = {"id": 1, "check": "splits disjoint: A by time (train < cutoff <= test), B by repo per fold",
          "value": leaks or "ok", "pass": not leaks}

    # 2. hygiene
    bad = []
    for name, cols in fs.FEATURE_SETS.items():
        try:
            fs.assert_hygiene(cols)
        except ValueError as exc:
            bad.append(f"{name}: {exc}")
    c2 = {"id": 2, "check": "no key/label/flag column in any feature set", "value": bad or "ok", "pass": not bad}

    # 3. the blueprint's ablation ran on both scenarios
    have = {(r["scenario"], r["featureset"]) for r in runs}
    ok3 = ("A", "NO_LABEL_REPLAY") in have and ("B", "NO_LABEL_REPLAY") in have
    c3 = {"id": 3, "check": "NO_LABEL_REPLAY ran on both scenarios", "value": sorted(map(str, have)), "pass": ok3}

    # 4. every run has a CI and all 24 are logged under this params sha
    ci_ok = all(np.isfinite(r["p10_ci_lo"]) and np.isfinite(r["p10_ci_hi"]) for r in runs)
    logged = int(((experiments["model"] == "lgbm") & (experiments["params"] == params_sha)).sum()) if len(experiments) else 0
    c4 = {"id": 4, "check": f"all {N_RUNS} runs have bootstrap CIs and are in experiments.csv under params sha",
          "value": {"runs": len(runs), "logged": logged, "ci_ok": ci_ok},
          "pass": ci_ok and len(runs) == N_RUNS and logged == N_RUNS}

    # 5. reproducibility
    c5 = {"id": 5, "check": "refit A/FULL from params.json reproduces AUC-PR (|delta| < 1e-6)",
          "value": repro_delta, "pass": repro_delta is not None and abs(repro_delta) < 1e-6}
    return [c1, c2, c3, c4, c5]


# ----------------------------------------------------------------------------
# Tables (pure)
# ----------------------------------------------------------------------------

def summarise(runs: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(runs)
    g = df.groupby(["scenario", "featureset"], sort=False)
    out = g.agg(n_folds=("fold", "count"), precision_at_10=("precision_at_10", "mean"),
                p10_ci_lo=("p10_ci_lo", "mean"), p10_ci_hi=("p10_ci_hi", "mean"),
                baseline_p10=("baseline_p10", "mean"), base_rate_p10=("base_rate_p10", "mean"),
                auc_pr=("auc_pr", "mean"), baseline_auc_pr=("baseline_auc_pr", "mean")).reset_index()
    order = {n: i for i, n in enumerate(SETS)}
    return out.sort_values(["scenario", "featureset"], key=lambda s: s.map(order) if s.name == "featureset" else s).reset_index(drop=True)


def bias_slices(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(slice_name, level, mask):
        sub = pred[mask]
        y = sub["is_slow"].to_numpy(dtype=int)
        auc = metrics.auc_pr(y, sub["p_hat"]) if 0 < y.sum() < len(y) else float("nan")
        rows.append({"slice": slice_name, "level": level, "n": int(len(sub)),
                     "actual_rate": float(y.mean()) if len(sub) else float("nan"),
                     "mean_p_hat": float(sub["p_hat"].mean()) if len(sub) else float("nan"), "auc_pr": auc})
    f = pred["is_first_pr_here"].astype(bool)
    add("is_first_pr_here", "first-time", f); add("is_first_pr_here", "repeat", ~f)
    h = pred["created_hour_utc"].astype(int)
    for lo, hi, name in HOUR_BUCKETS:
        add("hour_bucket", name, (h >= lo) & (h < hi))
    return pd.DataFrame(rows)


def pool_comparability(runs: list[dict]) -> pd.DataFrame:
    """Per scenario: how big and how time-spread is the per-repo Precision@10 candidate pool.

    metrics.precision_at_k ranks the top 10 PER REPO within a scenario's test rows, so a
    P@10 comparison across scenarios is only like-for-like if the two pools are comparable
    in size. This is what lets headline() caveat the A->B gap instead of overclaiming it."""
    rows = []
    for sc in ("A", "B"):
        paths = [r["pred_path"] for r in runs if r["scenario"] == sc and r["featureset"] == "FULL"]
        pred = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        per_repo_n = pred.groupby("repo").size()
        per_repo_slow = pred.groupby("repo")["is_slow"].sum()
        span_days = (pred["created_at"].max() - pred["created_at"].min()).days
        rows.append({
            "scenario": sc,
            "n_test_rows": int(len(pred)),
            "n_repos": int(pred["repo"].nunique()),
            "pool_median": float(per_repo_n.median()),
            "pool_min": int(per_repo_n.min()),
            "repos_under_10_test_prs": int((per_repo_n < 10).sum()),
            "repos_with_10plus_slow": int((per_repo_slow >= 10).sum()),
            "months_spanned": int(round(span_days / 30.4375)),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def fig_p10(summary: pd.DataFrame) -> Path:
    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, sc in zip(axes, ("A", "B")):
        s = summary[summary["scenario"] == sc]
        x = np.arange(len(s))
        ax.bar(x, s["precision_at_10"], yerr=[s["precision_at_10"] - s["p10_ci_lo"], s["p10_ci_hi"] - s["precision_at_10"]],
               capsize=4, label="LightGBM")
        ax.scatter(x, s["baseline_p10"], marker="_", s=400, color="k", label="baseline P@10")
        ax.scatter(x, s["base_rate_p10"], marker="x", color="gray", label="base rate")
        ax.set_xticks(x); ax.set_xticklabels(s["featureset"], rotation=20); ax.set_title(f"Scenario {sc}")
    axes[0].set_ylabel("Precision@10 (per repo, mean)"); axes[0].legend(fontsize=8)
    p = FIG / "phase4_p10.png"; fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_pr(scenario: str, pred_paths: list[Path]) -> Path:
    pred = pd.concat([pd.read_parquet(p) for p in pred_paths], ignore_index=True)
    y = pred["is_slow"].astype(int)
    fig, ax = plt.subplots(figsize=(5, 4))
    for col, label in (("p_hat", "LightGBM FULL"), ("baseline_score", "trailing-90d baseline")):
        pr, rc, _ = precision_recall_curve(y, pred[col]); ax.plot(rc, pr, label=label)
    ax.axhline(y.mean(), ls="--", color="gray", label=f"base rate {y.mean():.2f}")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title(f"Scenario {scenario}"); ax.legend(fontsize=8)
    p = FIG / f"phase4_pr_{scenario}.png"; fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


# ----------------------------------------------------------------------------
# Document
# ----------------------------------------------------------------------------

def md(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def headline(summary: pd.DataFrame, pool: pd.DataFrame) -> str:
    a = summary[(summary.scenario == "A") & (summary.featureset == "FULL")].iloc[0]
    b = summary[(summary.scenario == "B") & (summary.featureset == "FULL")].iloc[0]
    d_bl = a.precision_at_10 - a.baseline_p10
    d_br = a.precision_at_10 - a.base_rate_p10
    excl_bl = "excludes" if not (a.p10_ci_lo <= a.baseline_p10 <= a.p10_ci_hi) else "includes"
    excl_br = "excludes" if not (a.p10_ci_lo <= a.base_rate_p10 <= a.p10_ci_hi) else "includes"
    pa = pool[pool.scenario == "A"].iloc[0]
    pb = pool[pool.scenario == "B"].iloc[0]
    comparability = (
        f"That A→B comparison is **not** like-for-like: Precision@10 ranks the top 10 *per repo*, and "
        f"Scenario A draws them from a median of {pa.pool_median:.0f} test PRs over {pa.months_spanned} months "
        f"({pa.repos_under_10_test_prs} of {pa.n_repos} repos have fewer than 10 test PRs at all, and only "
        f"{pa.repos_with_10plus_slow}/{pa.n_repos} have 10 slow PRs available, so P@10 = 1.0 is unattainable for "
        f"{pa.n_repos - pa.repos_with_10plus_slow} of them), while Scenario B draws from a median of "
        f"{pb.pool_median:.0f} over {pb.months_spanned} months ({pb.repos_with_10plus_slow}/{pb.n_repos} attainable). "
        f"Ranking the 10 slowest out of a larger pool is easier, so B's higher P@10 is substantially a pool-size "
        f"artifact and must not be read as cold-start transfer being easy — for that comparison use AUC-PR, which "
        f"does not depend on pool size, and the per-fold detail below."
    )
    return (f"On Scenario A, the FULL model's within-repo Precision@10 is **{a.precision_at_10:.3f}** "
            f"[{a.p10_ci_lo:.3f}, {a.p10_ci_hi:.3f}] against the trailing-rate baseline's {a.baseline_p10:.3f} "
            f"(**{d_bl:+.3f}**; the CI {excl_bl} the baseline) and the base rate {a.base_rate_p10:.3f} "
            f"({d_br:+.3f}; the CI {excl_br} the base rate). AUC-PR {a.auc_pr:.3f} vs baseline {a.baseline_auc_pr:.3f}. "
            f"On Scenario B (cold-start), FULL averages P@10 {b.precision_at_10:.3f} vs baseline {b.baseline_p10:.3f}, "
            f"AUC-PR {b.auc_pr:.3f} vs {b.baseline_auc_pr:.3f} — an A→B P@10 gap of {a.precision_at_10 - b.precision_at_10:+.3f}. "
            f"{comparability} "
            f"The blueprint's bar was a clear margin over the baseline on A (5–10 points); "
            f"{'that bar is met' if d_bl >= 0.05 else 'that bar is NOT met — PR-level signal adds ' + ('little' if d_bl > 0 else 'nothing') + ' over the repo trailing rate within-repo, which is itself the reportable finding'}.")


def render(params: dict, params_sha: str, summary: pd.DataFrame, runs: list[dict], pool: pd.DataFrame,
           slices: dict[str, pd.DataFrame], checks: list[dict], figs: dict[str, Path], n_rows: int, n_dropped: int) -> str:
    abl = summary.copy()
    abl["d_p10_vs_FULL"] = abl.apply(lambda r: r.precision_at_10 - summary[(summary.scenario == r.scenario) & (summary.featureset == "FULL")].precision_at_10.iloc[0], axis=1)
    abl["d_auc_pr_vs_FULL"] = abl.apply(lambda r: r.auc_pr - summary[(summary.scenario == r.scenario) & (summary.featureset == "FULL")].auc_pr.iloc[0], axis=1)
    b_full = pd.DataFrame([r for r in runs if r["scenario"] == "B" and r["featureset"] == "FULL"])[
        ["fold", "n_test_repos", "precision_at_10", "p10_ci_lo", "p10_ci_hi", "baseline_p10", "base_rate_p10", "auc_pr", "baseline_auc_pr"]]
    verdict = "PASS" if all(c["pass"] for c in checks) else "FAIL"
    first = {sc: slices[sc][slices[sc]["slice"] == "is_first_pr_here"] for sc in slices}
    bias_line = []
    for sc, s in first.items():
        ft, rp = s[s.level == "first-time"].iloc[0], s[s.level == "repeat"].iloc[0]
        bias_line.append(f"Scenario {sc}: first-timers actual {ft.actual_rate:.3f} vs predicted {ft.mean_p_hat:.3f} "
                         f"({ft.mean_p_hat - ft.actual_rate:+.3f}); repeat {rp.actual_rate:.3f} vs {rp.mean_p_hat:.3f} ({rp.mean_p_hat - rp.actual_rate:+.3f}).")
    format_value = lambda v: f"{v:.2e}" if isinstance(v, float) else v
    checks_formatted = [{**c, "value": format_value(c["value"])} for c in checks]
    return f"""# Phase 4 — LightGBM Results

Generated by `report4.py`. Params sha `{params_sha}`. Rows: {n_rows:,} ({n_dropped} truncated-timeline rows dropped).

## Gate: **{verdict}** (validity, not success)

{md(pd.DataFrame(checks_formatted)[["id", "check", "value", "pass"]], "{}")}

## 1. Setup

Tuned on Scenario A training rows, `TimeSeriesSplit(3)`, {params['n_trials']} Optuna trials, CV AUC-PR **{params['cv_auc_pr']:.4f}**.
Params (frozen for every scenario, fold and ablation — ablations are therefore compared under identical settings, not re-tuned):

```
{json.dumps(params['best_params'], indent=2)}
```

## 2. Scenario A (known-project, time cutoff 2026-01-01)

{md(summary[summary.scenario == "A"].drop(columns=["scenario", "n_folds"]))}

## 3. Scenario B (cold-start, leave-repos-out, 5 folds; means over folds)

{md(summary[summary.scenario == "B"].drop(columns=["scenario"]))}

FULL per fold:

{md(b_full)}

## 4. Headline

{headline(summary, pool)}

## 4b. Are A and B comparable?

{md(pool)}

Precision@10 is a within-repo top-k metric, so it is sensitive to how many candidates each repo contributes. AUC-PR is not, which is why the headline points at it for the A→B comparison.

## 5. Ablations (Δ vs FULL, same scenario)

{md(abl[["scenario", "featureset", "precision_at_10", "d_p10_vs_FULL", "auc_pr", "d_auc_pr_vs_FULL"]])}

NO_LABEL_REPLAY answers the blueprint's question directly: it is the model without the baseline's own feature. NO_SNAPSHOT tests the feature dictionary's caveat that HEAD-at-collection repo facts may carry the cold-start result. PR_ONLY is what a PR looks like with no history at all.

## 6. Bias slices (FULL predictions)

{chr(10).join("### Scenario " + sc + chr(10) + chr(10) + md(s) for sc, s in slices.items())}

{" ".join(bias_line)}
A positive (predicted − actual) gap for first-timers means the model is more pessimistic about newcomers than their outcomes warrant — blueprint §4's fairness concern.

## 7. Figures

![p10](../figures/{figs['p10'].name})

![prA](../figures/{figs['pr_A'].name}) ![prB](../figures/{figs['pr_B'].name})
"""


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for required in (ex.RUNS_JSON, ex.PARAMS):
        if not required.exists():
            log.error("missing %s -- run tune.py then experiment.py first", required)
            return 1
    runs_doc = json.loads(ex.RUNS_JSON.read_text(encoding="utf-8"))
    runs, sha = runs_doc["runs"], runs_doc["params_sha"]
    params = json.loads(ex.PARAMS.read_text(encoding="utf-8"))
    experiments = pd.read_csv(tracking.EXPERIMENTS) if tracking.EXPERIMENTS.exists() else pd.DataFrame()

    # reproducibility: refit A/FULL and compare
    table = ex.load_table()
    raw = pd.read_parquet(ex.TABLE)
    tr, te = ex.folds_for("A", table)[0]
    b = model.fit(table.loc[tr, fs.FEATURE_SETS["FULL"]], table.loc[tr, "is_slow"], params["best_params"])
    auc_refit = metrics.auc_pr(table.loc[te, "is_slow"].astype(int), model.predict(b, table.loc[te, fs.FEATURE_SETS["FULL"]]))
    auc_saved = next(r["auc_pr"] for r in runs if r["scenario"] == "A" and r["featureset"] == "FULL")
    checks = gate_checks(runs, sha, experiments, auc_refit - auc_saved)
    GATE_JSON.write_text(json.dumps(checks, indent=2, default=str), encoding="utf-8")

    summary = summarise(runs)
    pool = pool_comparability(runs)
    slices = {}
    for sc in ("A", "B"):
        paths = [Path(r["pred_path"]) for r in runs if r["scenario"] == sc and r["featureset"] == "FULL"]
        slices[sc] = bias_slices(pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True))
    figs = {"p10": fig_p10(summary),
            "pr_A": fig_pr("A", [Path(r["pred_path"]) for r in runs if r["scenario"] == "A" and r["featureset"] == "FULL"]),
            "pr_B": fig_pr("B", [Path(r["pred_path"]) for r in runs if r["scenario"] == "B" and r["featureset"] == "FULL"])}
    DOC.write_text(render(params, sha, summary, runs, pool, slices, checks, figs, len(table), int(len(raw) - len(table))), encoding="utf-8")
    verdict = all(c["pass"] for c in checks)
    print(f"wrote {DOC}  gate={'PASS' if verdict else 'FAIL'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())

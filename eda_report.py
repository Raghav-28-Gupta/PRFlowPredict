"""Phase 2 EDA report + pre-registered gate. Writes docs/phase2_eda.md and figures/."""
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
from lifelines import KaplanMeierFitter

import baseline
import cohort_qc
import labels
import load
import splits

log = logging.getLogger("eda")
ROOT = Path(__file__).parent
DOC = ROOT / "docs" / "phase2_eda.md"
FIG = ROOT / "figures"
GATE_JSON = ROOT / "data" / "phase2_gate.json"
THRESHOLDS = [72, 120, 168, 240]


# ----------------------------------------------------------------------------
# Gate (pure, tested)
# ----------------------------------------------------------------------------

def gate_checks(kept_n: int, d5_rate: float, a_repos_with_10: int, b_min_fold_repos: int,
                baseline_rows: list[dict]) -> list[dict]:
    def sane(r):
        # P@10 must sit within the CI of the base rate (with a 5pp tolerance on the
        # bootstrap edges). Beating it within-repo means the replay leaks.
        return r["p10_ci_lo"] - 0.05 <= r["base_rate_p10"] <= r["p10_ci_hi"] + 0.05 \
            and r["precision_at_10"] <= r["base_rate_p10"] + 0.10
    return [
        {"id": 1, "check": "repos kept after QC >= 30", "value": kept_n, "pass": kept_n >= 30},
        {"id": 2, "check": "D5 global is_slow in [10%, 70%]", "value": round(d5_rate, 4),
         "pass": 0.10 <= d5_rate <= 0.70},
        {"id": 3, "check": "Scenario A: >= 20 test repos with >= 10 test PRs",
         "value": a_repos_with_10, "pass": a_repos_with_10 >= 20},
        {"id": 4, "check": "Scenario B: every fold holds out >= 6 repos",
         "value": b_min_fold_repos, "pass": b_min_fold_repos >= 6},
        {"id": 5, "check": "baseline sanity: P@10 within CI of base rate (else replay LEAKS)",
         "value": [round(r["precision_at_10"], 3) for r in baseline_rows],
         "pass": bool(baseline_rows) and all(sane(r) for r in baseline_rows)},
    ]


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def fig_hist(lab: pd.DataFrame, tier_of: pd.Series) -> Path:
    FIG.mkdir(exist_ok=True)
    rev = lab[lab["event_observed"]].copy()
    rev["tier"] = rev["repo"].map(tier_of)
    fig, ax = plt.subplots(figsize=(8, 4))
    for tier, g in rev.groupby("tier"):
        ax.hist(np.log10(g["wait_h"].clip(lower=1e-2)), bins=40, alpha=0.5, label=tier, density=True)
    ax.axvline(np.log10(168), color="k", ls="--", label="168h")
    ax.set_xlabel("log10(hours to first human review)"); ax.set_ylabel("density"); ax.legend()
    p = FIG / "wait_hist.png"; fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_km(lab: pd.DataFrame, tier_of: pd.Series) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    kmf = KaplanMeierFitter()
    kmf.fit(lab["wait_h_censored"], event_observed=lab["event_observed"], label="all")
    kmf.plot_survival_function(ax=ax)
    for tier, g in lab.assign(tier=lab["repo"].map(tier_of)).groupby("tier"):
        KaplanMeierFitter().fit(g["wait_h_censored"], g["event_observed"], label=tier)\
            .plot_survival_function(ax=ax, ci_show=False)
    ax.axvline(168, color="k", ls="--"); ax.set_xlim(0, 720)
    ax.set_xlabel("hours since open"); ax.set_ylabel("fraction still unreviewed")
    p = FIG / "km_unreviewed.png"; fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    qc = json.loads(cohort_qc.KEPT.read_text(encoding="utf-8"))
    kept = qc["kept"]
    if not kept:
        checks = gate_checks(0, float("nan"), 0, 0, [])
        GATE_JSON.parent.mkdir(parents=True, exist_ok=True)
        GATE_JSON.write_text(json.dumps(checks, indent=2), encoding="utf-8")
        print(f"no repos kept after QC; wrote FAIL gate to {GATE_JSON}")
        return 1
    cohort = {e["repo"]: e for e in json.loads(cohort_qc.COHORT.read_text(encoding="utf-8"))["selected"]}
    tier_of = pd.Series({r: cohort[r]["star_tier"] for r in cohort})

    frames = load.load_all(kept)
    prs = frames["pr_tier2"]
    prs = prs[~prs["author_is_bot"].fillna(False).astype(bool)]
    prs = prs[(prs["created_at"] >= splits.WINDOW_START) & (prs["created_at"] <= splits.WINDOW_END)]
    streams = {k: frames[k] for k in labels.ALL_STREAMS}
    all_labels = labels.label_all(prs, streams)
    lab = all_labels[labels.PRIMARY]

    # 1. cohort table
    qc_df = pd.DataFrame(qc["repos"])[["repo", "cell", "kept", "reasons", "n_in_window", "n_human",
                                       "bot_share", "is_slow_d5", "is_slow_d3"]]
    qc_df["reasons"] = qc_df["reasons"].apply("; ".join)
    share = splits.row_share(prs)
    qc_df["row_share"] = qc_df["repo"].map(share).fillna(0.0)

    # 2. sensitivity
    sens = pd.DataFrame([{"definition": k, "is_slow": v["is_slow"].mean(),
                          "never_reviewed_30d": v["never_reviewed_30d"].mean(),
                          "median_wait_h": v.loc[v["event_observed"], "wait_h"].median()}
                         for k, v in all_labels.items()])
    per_repo_spread = (all_labels["D3"].groupby("repo")["is_slow"].mean()
                       - all_labels["D5"].groupby("repo")["is_slow"].mean()).abs()

    # 3-4. figures
    hist_p, km_p = fig_hist(lab, tier_of), fig_km(lab, tier_of)

    # 5. threshold sensitivity
    thr = pd.DataFrame([{"threshold_h": t,
                         "is_slow": float((lab["never_reviewed_30d"] | (lab["wait_h"] > t)).mean())}
                        for t in THRESHOLDS])

    # 6. baseline
    rows = splits.prepare_rows(frames["pr_tier2"], lab, kept)
    tr_a, te_a = splits.scenario_a(rows)
    folds_b = splits.scenario_b(rows)
    base_rows = baseline.evaluate("A", rows, [(tr_a, te_a)], frames["pr_tier1"], lab)
    base_rows += baseline.evaluate("B", rows, folds_b, frames["pr_tier1"], lab)
    base_df = pd.DataFrame(base_rows)[["scenario", "fold", "n_test_repos", "precision_at_10",
                                       "p10_ci_lo", "p10_ci_hi", "base_rate_p10", "auc_pr", "base_rate"]]

    # 7. gate
    a_cov = int((rows.loc[te_a].groupby("repo").size() >= 10).sum())
    b_min = min(rows.loc[te, "repo"].nunique() for _, te in folds_b)
    checks = gate_checks(len(kept), float(lab["is_slow"].mean()), a_cov, b_min, base_rows)
    GATE_JSON.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    verdict = "PASS" if all(c["pass"] for c in checks) else "FAIL"

    look = qc.get("look_at_these", [])
    doc = f"""# Phase 2 — EDA and Gate

Generated by `eda_report.py`. Primary label: **{labels.PRIMARY}**. Rows: human-authored,
in-window PRs from kept repos ({len(prs):,} PRs, {len(kept)} repos).

## Verdict: **{verdict}**

{md_table(pd.DataFrame(checks)[["id", "check", "value", "pass"]], "{}")}

## 1. Cohort

{md_table(qc_df)}

Look-at-these (not dropped): {"; ".join(f"{l['repo']} ({', '.join(l['why'])})" for l in look) or "none"}

## 2. Label sensitivity

{md_table(sens)}

Per-repo |D3 − D5| spread: median {per_repo_spread.median():.3f}, max {per_repo_spread.max():.3f}
({per_repo_spread.idxmax()}).

Limitation: commenter `author_association` is read-time-computed (proven for PR authors in
Phase 0), so D5 slightly over-counts reviews relative to true at-the-time standing.

## 3. Wait-time distribution

![wait](../figures/{hist_p.name})

## 4. Kaplan–Meier: fraction still unreviewed

![km](../figures/{km_p.name})

## 5. Threshold sensitivity

{md_table(thr)}

Recommendation: keep 168h unless the curve shows a natural break elsewhere.

## 6. Baseline (trailing-90d slow rate)

{md_table(base_df)}

Expected in advance: P@10 ≈ base rate (score is constant within a repo). AUC-PR may exceed
the base rate because the trailing rate ranks repos.
"""
    DOC.write_text(doc, encoding="utf-8")
    print(f"wrote {DOC}  verdict={verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

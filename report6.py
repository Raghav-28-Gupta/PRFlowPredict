"""Phase 6 document + validity gate.

The gate tests VALIDITY, not what the attributions say. A shift table showing Scenario B
leaning on the label-replay features and one showing it not are both valid outcomes; §2
of the document reports whichever occurred. The hard stop is SHAP additivity: if
sum(shap) + expected_value does not reproduce the booster's raw margin, the attributions
are not the model's and nothing downstream means anything."""
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
import shap
from scipy.stats import spearmanr

import attribution as attr
import errors
import experiment as ex
import fairness
import featuresets as fs
import model

log = logging.getLogger("report6")
ROOT = Path(__file__).parent
DOC = ROOT / "docs" / "phase6_interpretation.md"
FIG = ROOT / "figures"
DATA = ROOT / "data"
GATE_JSON = DATA / "phase6_gate.json"
ADDITIVITY_TOL = 1e-6
STABILITY_MIN = 0.9


# ---------------------------------------------------------------------------
# Gate (pure)
# ---------------------------------------------------------------------------

def gate_checks(additivity: dict[str, list[float]], stability: float, worst: pd.DataFrame,
                pred_a: pd.DataFrame, fair: pd.DataFrame, artifacts: dict[str, int]) -> list[dict]:
    worst_delta = max((max(v) for v in additivity.values() if v), default=float("inf"))
    c1 = {"id": 1, "check": f"SHAP additivity vs raw margin (max |delta| < {ADDITIVITY_TOL})",
          "value": {k: max(v) if v else None for k, v in additivity.items()},
          "pass": worst_delta < ADDITIVITY_TOL}

    c2 = {"id": 2, "check": f"sampling stability: two seeds' top-10 rankings agree (Spearman >= {STABILITY_MIN})",
          "value": round(float(stability), 4), "pass": float(stability) >= STABILITY_MIN}

    truth = pred_a.set_index("pr_id")["is_slow"]
    unknown = [p for p in worst["pr_id"] if p not in truth.index]
    mismatched = [p for p in worst["pr_id"]
                  if p in truth.index and bool(truth.loc[p]) != bool(worst.set_index("pr_id").loc[p, "is_slow"])]
    n_fp = int((worst["kind"] == "fp").sum())
    n_fn = int((worst["kind"] == "fn").sum())
    c3 = {"id": 3, "check": "error rows come from Scenario A/FULL test set, 25 fp + 25 fn, labels match",
          "value": {"n_fp": n_fp, "n_fn": n_fn, "unknown": unknown[:5], "mismatched": mismatched[:5]},
          "pass": n_fp == 25 and n_fn == 25 and not unknown and not mismatched}

    ci_finite = bool(np.isfinite(fair["gap_ci_lo"]).all() and np.isfinite(fair["gap_ci_hi"]).all())
    repos_ok = bool((fair["n_repos"] >= 2).all())
    c4 = {"id": 4, "check": "every fairness CI is finite and computed from >= 2 repos",
          "value": {"ci_finite": ci_finite, "min_repos": int(fair["n_repos"].min())},
          "pass": ci_finite and repos_ok}

    empty = [k for k, v in artifacts.items() if not v]
    c5 = {"id": 5, "check": "all five data artifacts written and non-empty",
          "value": artifacts, "pass": not empty}
    return [c1, c2, c3, c4, c5]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_shift(shift_df: pd.DataFrame, top: int = 15) -> Path:
    FIG.mkdir(exist_ok=True)
    s = shift_df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["tab:red" if d > 0 else "tab:blue" for d in s["delta"]]
    ax.barh(s["feature"], s["delta"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("share of |SHAP| in B  minus  share in A")
    ax.set_title("Attribution shift, Scenario A -> B (FULL model)")
    p = FIG / "phase6_shift.png"; fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_beeswarm(shap_values: np.ndarray, X: pd.DataFrame, scenario: str) -> Path:
    FIG.mkdir(exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    p = FIG / f"phase6_beeswarm_{scenario}.png"
    plt.tight_layout(); plt.savefig(p, dpi=130); plt.close("all")
    return p


def fig_dependence(shap_values: np.ndarray, X: pd.DataFrame, feature: str, scenario: str) -> Path:
    FIG.mkdir(exist_ok=True)
    plt.figure()
    shap.dependence_plot(feature, shap_values, X, show=False, interaction_index=None)
    safe = feature.replace("/", "_")
    p = FIG / f"phase6_dep_{safe}_{scenario}.png"
    plt.tight_layout(); plt.savefig(p, dpi=130); plt.close("all")
    return p


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

def md(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, float) else str(v)
                                       for v in r) + " |")
    return "\n".join(lines)


def mechanism(share_a: float, share_b: float) -> str:
    """What the label-replay share says about the cold-start finding -- either way."""
    d = share_b - share_a
    if d > 0.05:
        verdict = (f"**B leans harder on the baseline's own signal.** The four label-replay features carry "
                   f"{share_b:.1%} of Scenario B's attribution mass against {share_a:.1%} in Scenario A "
                   f"({d:+.1%}). That is the mechanism behind Phase 4's ablation result: strip those "
                   f"features and the cold-start model has comparatively little left, which is exactly what "
                   f"NO_LABEL_REPLAY showed (B AUC-PR 0.763 vs baseline 0.821, while A still won at 0.902 "
                   f"vs 0.887). Cross-project, the model is largely re-deriving the repo's own trailing "
                   f"rate rather than learning transferable PR-level structure.")
    elif d < -0.05:
        verdict = (f"**B leans LESS on the label-replay features than A does** ({share_b:.1%} vs "
                   f"{share_a:.1%}, {d:+.1%}) — so the cold-start weakness Phase 4 measured is not "
                   f"explained by over-reliance on the baseline's signal, and the explanation lies "
                   f"elsewhere. Worth stating plainly: this contradicts the expected mechanism.")
    else:
        verdict = (f"**The label-replay share is essentially unchanged between scenarios** "
                   f"({share_a:.1%} A vs {share_b:.1%} B, {d:+.1%}). Attribution mass does not explain "
                   f"the cold-start gap; the difference must lie in how the same features behave on "
                   f"unseen repos rather than in which features are used.")
    return verdict


def fairness_verdict(diff: dict, scenario: str) -> str:
    if diff["ci_excludes_zero"] and diff["difference"] > 0:
        return (f"**Scenario {scenario}: the newcomer gap is real.** First-time contributors' predictions "
                f"run {diff['difference']:+.3f} more pessimistic than repeat contributors', "
                f"CI [{diff['ci_lo']:.3f}, {diff['ci_hi']:.3f}], which excludes zero. A triage tool built "
                f"on this model would systematically deprioritise newcomers' PRs beyond what their actual "
                f"outcomes warrant — the blueprint §4 concern, confirmed.")
    if diff["ci_excludes_zero"]:
        return (f"**Scenario {scenario}: the gap runs the other way.** Repeat contributors are treated more "
                f"pessimistically than newcomers by {-diff['difference']:.3f}, CI "
                f"[{diff['ci_lo']:.3f}, {diff['ci_hi']:.3f}].")
    return (f"**Scenario {scenario}: no distinguishable newcomer gap.** The difference is "
            f"{diff['difference']:+.3f} with CI [{diff['ci_lo']:.3f}, {diff['ci_hi']:.3f}], which contains "
            f"zero — the model is not measurably more pessimistic about first-time contributors than about "
            f"repeat ones, on {diff['n_first_time']:,} newcomer and {diff['n_repeat']:,} repeat rows.")


SCENARIO_WHAT = {"A": "within-project split, time cutoff", "B": "cold-start split, leave-repos-out"}


def _fmt_wait_h(df: pd.DataFrame) -> pd.DataFrame:
    """wait_h is NaN exactly when the PR was never reviewed at all -- the most important fact
    about a row the model called "fine". Rendering that as the literal string "nan" would read
    as missing data instead. worst_rows() keeps wait_h a float so data/phase6_worst50.csv stays
    numeric and machine-readable; this substitution belongs only in the rendered document."""
    out = df.copy()
    out["wait_h"] = out["wait_h"].apply(lambda v: "never reviewed" if pd.isna(v) else v)
    return out


def _fairness_heading(scenario: str) -> str:
    """Section 5 shows both scenarios' tables back to back; name which is which."""
    return f"### Scenario {scenario} ({SCENARIO_WHAT[scenario]})"


def render(shift_df, imp_a, imp_b, share_a, share_b, worst, patterns, fair, diffs,
           checks, figs, n_explained) -> str:
    verdict = "PASS" if all(c["pass"] for c in checks) else "FAIL"
    fp = _fmt_wait_h(worst[worst["kind"] == "fp"].head(25))
    fn = _fmt_wait_h(worst[worst["kind"] == "fn"].head(25))
    show = ["repo", "number", "url", "p_hat", "wait_h", "top1_feature", "top1_shap",
            "top2_feature", "top2_shap", "top3_feature", "top3_shap"]
    return f"""# Phase 6 — Interpretation, Error Analysis, Fairness

Generated by `report6.py`. Exact TreeSHAP over the Phase 4 boosters; nothing retrained.
{n_explained:,} explained rows per scenario (seeded sample).

## Gate: **{verdict}** (validity, not what the attributions say)

{md(pd.DataFrame(checks)[["id", "check", "value", "pass"]], "{}")}

## 1. Attribution shift, Scenario A -> B (FULL model)

`share` is each feature's mean-|SHAP| as a fraction of the scenario's total, so the two
scenarios are comparable despite different base rates and margins. `delta = share_b - share_a`.

{md(shift_df.head(20))}

![shift](../figures/{figs['shift'].name})

## 2. The cold-start mechanism

{mechanism(share_a, share_b)}

Label-replay share: **Scenario A {share_a:.1%}**, **Scenario B {share_b:.1%}**.

## 3. What each model leans on

Scenario A, top 15:

{md(imp_a.head(15))}

Scenario B, top 15:

{md(imp_b.head(15))}

![beeswarmA](../figures/{figs['beeswarm_A'].name})

![beeswarmB](../figures/{figs['beeswarm_B'].name})

## 4. The 50 worst predictions (Scenario A, FULL)

25 most-confident false positives — the model said "this will stall", it did not:

{md(fp[show])}

25 most-confident false negatives — the model said "this is fine", it stalled:

{md(fn[show])}

### What the failures have in common

Feature values among the 50 worst versus the whole test set, as z-scores:

{md(patterns.head(12))}

<!-- WRITTEN-ANALYSIS: replace this line with the hand-written reading of data/phase6_worst50.csv -->

## 5. Fairness (blueprint §4)

How `is_first_pr_here` moves the prediction, and how the top-shift feature behaves:

![dep_first](../figures/{figs['dep_first'].name})

![dep_top](../figures/{figs['dep_top'].name})

`actual_rate` and `mean_p_hat` below are PR-weighted means (averaged over individual PRs), while `gap`
and its confidence interval are repo-weighted — the per-repo gap is the bootstrap unit, matching every
other interval in this project. The two are on different weighting bases, so in this table
**`gap` does not equal `mean_p_hat - actual_rate`**; that is by construction, not an error, and the
columns are not meant to be subtracted against each other.

{chr(10).join(_fairness_heading(sc) + chr(10) + chr(10) + md(f) + chr(10) for sc, f in fair.items())}

{chr(10).join(fairness_verdict(d, sc) for sc, d in diffs.items())}
"""


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    runs = attr.load_runs()
    table = ex.load_table()
    cols = fs.FEATURE_SETS["FULL"]

    sv, X, add = {}, {}, {}
    for sc in ("A", "B"):
        sv[sc], X[sc], add[sc] = attr.explain_run(sc, table, runs)
        log.info("scenario %s: %d explained rows, max additivity delta %.2e",
                 sc, len(X[sc]), max(add[sc]))

    imp_a, imp_b = attr.importance(sv["A"], cols), attr.importance(sv["B"], cols)
    shift_df = attr.shift(imp_a, imp_b)
    share_a, share_b = attr.label_replay_share(imp_a), attr.label_replay_share(imp_b)

    # gate 2: same model, different sample seed -> compare top-10 rankings
    sv2, _, _ = attr.explain_run("A", table, runs, seed=attr.SEED + 1)
    imp_a2 = attr.importance(sv2, cols)
    top = imp_a.head(10)["feature"].tolist()
    r2 = imp_a2.set_index("feature")["mean_abs_shap"]
    stability = float(spearmanr(imp_a.set_index("feature").loc[top, "mean_abs_shap"],
                                r2.reindex(top)).statistic)

    # Errors: Scenario A / FULL, over the WHOLE test set rather than the 5k sample --
    # "the 50 worst predictions" must be the worst the model actually made.
    run_a = next(r for r in runs if r["scenario"] == "A" and r["featureset"] == "FULL")
    by_id = table.set_index("pr_id")
    pred_a = pd.read_parquet(run_a["pred_path"]).reset_index(drop=True)
    ids = pred_a["pr_id"].to_numpy()
    # The prediction parquet has neither `number` nor `wait_h`; both come from the table.
    pred_a["number"] = by_id.loc[ids, "number"].to_numpy()
    pred_a["wait_h"] = by_id.loc[ids, "wait_h"].to_numpy()
    feat_a = by_id.loc[ids, cols]                       # pr_id-indexed, row-aligned with pred_a
    booster_a = model.load(Path(run_a["model_path"]))
    sv_a_full, ev_a = attr.explain(booster_a, feat_a)
    worst = errors.worst_rows(pred_a, sv_a_full, cols, feat_a.reset_index(drop=True), n=25)
    patterns = errors.error_patterns(worst, feat_a, cols)

    # fairness on both scenarios
    fair, diffs = {}, {}
    for sc in ("A", "B"):
        paths = [r["pred_path"] for r in runs if r["scenario"] == sc and r["featureset"] == "FULL"]
        p = pd.concat([pd.read_parquet(q) for q in paths], ignore_index=True)
        fair[sc] = fairness.slice_gaps(p)
        diffs[sc] = fairness.gap_difference(p)

    DATA.mkdir(exist_ok=True)
    shift_df.to_csv(DATA / "phase6_shift.csv", index=False)
    imp_a.to_csv(DATA / "phase6_importance_A.csv", index=False)
    imp_b.to_csv(DATA / "phase6_importance_B.csv", index=False)
    worst.to_csv(DATA / "phase6_worst50.csv", index=False)
    fair_all = pd.concat([f.assign(scenario=sc) for sc, f in fair.items()], ignore_index=True)
    fair_all.to_csv(DATA / "phase6_fairness.csv", index=False)

    artifacts = {"shift": len(shift_df), "importance_A": len(imp_a), "importance_B": len(imp_b),
                 "worst50": len(worst), "fairness": len(fair_all)}
    checks = gate_checks({**add, "A_full": [attr.additivity_delta(booster_a, feat_a, sv_a_full, ev_a)]},
                         stability, worst, pred_a, fair_all, artifacts)
    GATE_JSON.write_text(json.dumps(checks, indent=2, default=str), encoding="utf-8")

    top_shift = shift_df.iloc[0]["feature"]             # spec 7.4: dependence for the top-shift feature
    figs = {"shift": fig_shift(shift_df),
            "beeswarm_A": fig_beeswarm(sv["A"], X["A"], "A"),
            "beeswarm_B": fig_beeswarm(sv["B"], X["B"], "B"),
            "dep_first": fig_dependence(sv["A"], X["A"], "is_first_pr_here", "A"),
            "dep_top": fig_dependence(sv["B"], X["B"], top_shift, "B")}
    DOC.write_text(render(shift_df, imp_a, imp_b, share_a, share_b, worst, patterns,
                          fair, diffs, checks, figs, len(X["A"])), encoding="utf-8")
    ok = all(c["pass"] for c in checks)
    print(f"wrote {DOC}  gate={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

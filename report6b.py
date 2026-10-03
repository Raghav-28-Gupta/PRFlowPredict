"""Phase 6b report: the pre-registered repo-fingerprinting test.

Tests whether the NO_LABEL_REPLAY model's cold-start failure is repo fingerprinting -- the
model identifying repos by their static attributes and replaying their base rates.

Pre-registered in docs/design/specs/2026-09-24-phase6b-fingerprinting-design.md,
committed before any of this ran. The document this writes is FULLY GENERATED: there is no
hand-written section, because Phase 6's hand-written section 4 showed what one costs inside
a generated document -- a re-run silently destroys it.

Re-running is safe but not free: it re-trains the six NLR_NO_REPO boosters (deterministic,
so identical) and appends their six rows to data/experiments.csv again. Phase 4's gate
check 4 counts only Phase 4's own feature sets, so that does not disturb it."""
from __future__ import annotations

import json
import logging
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import attribution as attr
import experiment as ex
import featuresets as fs
import fingerprint as fp
import metrics
import model
import tracking
from report6 import md

log = logging.getLogger("report6b")
ROOT = Path(__file__).parent
DOC = ROOT / "docs" / "phase6b_fingerprinting.md"
DATA = ROOT / "data"
RUNS_JSON = DATA / "phase6b_runs.json"
GATE_JSON = DATA / "phase6b_gate.json"
SPEC = "docs/design/specs/2026-09-24-phase6b-fingerprinting-design.md"

TOL = 1e-6
EXPECTED_BOOSTERS = 6          # 1 Scenario A fold + 5 Scenario B folds
EXPECTED_REPOS = 39            # the cohort after Phase 2's QC
EXPECTED_NO_REPO_COLS = 25     # NO_LABEL_REPLAY's 33 minus the 8 repo-level features
HARD_STOPS = (1, 2)


def train_no_repo(table: pd.DataFrame, params: dict, sha: str, *,
                  out_models: Path = ex.MODELS_DIR, out_preds: Path = ex.PRED_DIR) -> list[dict]:
    """The intervention (spec section 5.3): NLR_NO_REPO on Phase 4's identical folds and
    tuned params, one fit for A and five for B, each logged to data/experiments.csv."""
    cols = fp.nlr_no_repo_cols()
    runs = []
    for scenario in ex.SCENARIOS:
        for k, (tr, te) in enumerate(ex.folds_for(scenario, table)):
            res = ex.run(scenario, k, fp.NO_REPO_NAME, table, params, tr, te,
                         out_models=out_models, out_preds=out_preds, cols=cols)
            runs.append(res)
            tracking.log({
                "scenario": scenario, "fold": k, "model": "lgbm", "features": fp.NO_REPO_NAME,
                "params": sha, "n_train": res["n_train"], "n_test": res["n_test"],
                "n_test_repos": res["n_test_repos"], "precision_at_10": res["precision_at_10"],
                "p10_ci_lo": res["p10_ci_lo"], "p10_ci_hi": res["p10_ci_hi"],
                "base_rate_p10": res["base_rate_p10"], "auc_pr": res["auc_pr"],
                "base_rate": res["base_rate"],
                "notes": "phase6b intervention: NO_LABEL_REPLAY minus the 8 repo-level snapshot features",
            })
    return runs


def refit_delta(table: pd.DataFrame, params: dict, saved_auc: float) -> float:
    """Gate check 2: re-fit A/NLR through the cols= code path and compare with Phase 4's
    saved AUC-PR. Everything goes to a scratch directory; the defaults are Phase 4's own
    directories and would overwrite A_NO_LABEL_REPLAY_fold0.txt."""
    tr, te = ex.folds_for("A", table)[0]
    with tempfile.TemporaryDirectory() as d:
        res = ex.run("A", 0, "NO_LABEL_REPLAY", table, params, tr, te,
                     out_models=Path(d) / "m", out_preds=Path(d) / "p",
                     cols=fs.FEATURE_SETS["NO_LABEL_REPLAY"])
    return float(res["auc_pr"] - saved_auc)


def score_rows(scenario: str, table: pd.DataFrame, nlr_runs: list[dict], no_repo_runs: list[dict]
               ) -> tuple[pd.DataFrame, np.ndarray, dict[str, float], dict[str, bool]]:
    """Every test row of one scenario: fold, label, both models' scores, and the NLR model's
    summed repo-feature SHAP. The two models' predictions are joined on pr_id, never by
    position.

    Returns the rows, the pooled SHAP matrix (for reliance), each booster's additivity delta
    (gate check 1), and whether each fold's two test sets are identical (gate check 3)."""
    cols = fs.FEATURE_SETS["NO_LABEL_REPLAY"]
    by_id = table.set_index("pr_id")
    new = {r["fold"]: r for r in no_repo_runs if r["scenario"] == scenario}
    parts, mats, additivity, same = [], [], {}, {}
    for r in sorted((r for r in nlr_runs if r["scenario"] == scenario), key=lambda r: r["fold"]):
        k = r["fold"]
        tag = f"{scenario}_fold{k}"
        old = pd.read_parquet(r["pred_path"])[["pr_id", "repo", "is_slow", "p_hat"]]
        nw = pd.read_parquet(new[k]["pred_path"])[["pr_id", "p_hat"]]
        same[tag] = set(old["pr_id"]) == set(nw["pr_id"])
        m = old.merge(nw, on="pr_id", how="inner", suffixes=("_nlr", "_no_repo"), validate="one_to_one")
        X = attr.prepare(by_id.loc[m["pr_id"].to_numpy(), cols])
        booster = model.load(Path(r["model_path"]))
        sv, ev = attr.explain(booster, X)
        additivity[tag] = attr.additivity_delta(booster, X, sv, ev)
        parts.append(pd.DataFrame({
            "pr_id": m["pr_id"].to_numpy(), "repo": m["repo"].to_numpy(), "fold": k,
            "is_slow": m["is_slow"].to_numpy(dtype=int),
            "p_nlr": m["p_hat_nlr"].to_numpy(dtype=float),
            "p_no_repo": m["p_hat_no_repo"].to_numpy(dtype=float),
            "repo_contrib": fp.repo_contribution(sv, cols),
        }))
        mats.append(sv)
    return pd.concat(parts, ignore_index=True), np.vstack(mats), additivity, same


def reliance(sv_a: np.ndarray, sv_b: np.ndarray) -> pd.DataFrame:
    """Share of mean |SHAP| per feature on the NLR models, A vs B (spec section 5.1,
    descriptive). Sorted by B's share, descending."""
    cols = fs.FEATURE_SETS["NO_LABEL_REPLAY"]
    a = attr.importance(sv_a, cols).set_index("feature")["share"].rename("share_a")
    b = attr.importance(sv_b, cols).set_index("feature")["share"].rename("share_b")
    out = pd.concat([a, b], axis=1).rename_axis("feature").reset_index()
    out["is_repo_feature"] = out["feature"].isin(fp.REPO_FEATURES)
    return out.sort_values("share_b", ascending=False, kind="mergesort").reset_index(drop=True)


def coverage(nlr_runs: list[dict], rows_a: pd.DataFrame, rows_b: pd.DataFrame) -> dict:
    """Gate check 5's premise: A and B cover the same repos, and B holds each out once."""
    held = Counter(repo for r in nlr_runs if r["scenario"] == "B" for repo in r["test_repos"])
    ra, rb = set(rows_a["repo"]), set(rows_b["repo"])
    return {"n_repos": len(ra), "same_repo_set": ra == rb,
            "b_each_once": bool(held) and set(held) == rb and all(v == 1 for v in held.values())}


def fold_table(rows: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per (scenario, fold): both models' AUC-PR on identical rows."""
    out = []
    for sc, r in rows.items():
        for k, g in r.groupby("fold"):
            y = g["is_slow"].to_numpy(dtype=int)
            old, new = metrics.auc_pr(y, g["p_nlr"]), metrics.auc_pr(y, g["p_no_repo"])
            out.append({"scenario": sc, "fold": int(k), "n_test": int(len(g)),
                        "auc_pr_nlr": old, "auc_pr_no_repo": new, "delta": new - old})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Gate (pure) -- spec section 7
# ---------------------------------------------------------------------------

def gate_checks(additivity: dict[str, float], refit_delta: float | None, same_rows: dict[str, bool],
                integrity: dict, cov: dict, intervals: dict | None, artifacts: dict[str, int]) -> list[dict]:
    worst = max(additivity.values(), default=float("inf"))
    c1 = {"id": 1, "check": f"SHAP additivity vs raw margin on all {EXPECTED_BOOSTERS} NLR boosters "
                            f"(max |delta| < {TOL})",
          "value": {k: float(v) for k, v in additivity.items()},
          "pass": len(additivity) == EXPECTED_BOOSTERS and worst < TOL}
    c2 = {"id": 2, "check": f"A/NLR refit via the cols= code path reproduces Phase 4's saved AUC-PR "
                            f"(|delta| < {TOL})",
          "value": refit_delta, "pass": refit_delta is not None and abs(refit_delta) < TOL}
    c3 = {"id": 3, "check": "every NLR_NO_REPO fold has exactly the NLR fold's test rows",
          "value": same_rows, "pass": len(same_rows) == EXPECTED_BOOSTERS and all(same_rows.values())}
    ok4 = (not integrity["non_constant"] and integrity["equals_full_minus_no_snapshot"]
           and integrity["n_no_repo_cols"] == EXPECTED_NO_REPO_COLS and integrity["hygiene_ok"])
    c4 = {"id": 4, "check": "the 8 repo features are constant within every repo and equal FULL - NO_SNAPSHOT; "
                            f"NLR_NO_REPO is NLR minus exactly them ({EXPECTED_NO_REPO_COLS} columns)",
          "value": integrity, "pass": bool(ok4)}
    finite = intervals is not None and all(np.isfinite(intervals[k]) for k in ("T_lo", "T_hi", "I_lo", "I_hi"))
    written = bool(artifacts) and all(v > 0 for v in artifacts.values())
    ok5 = cov["n_repos"] == EXPECTED_REPOS and cov["same_repo_set"] and cov["b_each_once"] and finite and written
    c5 = {"id": 5, "check": f"A and B cover the same {EXPECTED_REPOS} repos, B holds each out once, "
                            "every interval is finite, every artifact is written",
          "value": {**cov, "intervals_finite": finite, "artifacts": artifacts}, "pass": bool(ok5)}
    return [c1, c2, c3, c4, c5]


def hard_stopped(checks: list[dict]) -> list[int]:
    return [c["id"] for c in checks if c["id"] in HARD_STOPS and not c["pass"]]


# ---------------------------------------------------------------------------
# Verdict prose (pure). Every sentence must hold for ANY numbers in its cell.
# ---------------------------------------------------------------------------

HEADLINE = {
    "SUPPORTED": "**Repo fingerprinting is supported.** Both pre-registered tests confirm it.",
    "PARTIAL_SHAP_ONLY": "**Partially supported: the SHAP transfer test confirms, the intervention does not.**",
    "PARTIAL_INTERVENTION_ONLY": "**Partially supported: the intervention confirms, the SHAP transfer test does not.**",
    "NOT_SUPPORTED": "**Repo fingerprinting is not supported.** Neither pre-registered test confirms it.",
    "CONFLICTING": "**The two tests conflict.** One confirms repo fingerprinting and the other contradicts it.",
    "CONTRADICTED": "**Repo fingerprinting is contradicted.** Neither test confirms it and at least one "
                    "points the other way.",
}

READING = {
    "SUPPORTED": (
        "On repos the model saw in training, the repo-level features' contribution tracks each repo's "
        "actual slow rate more closely than on held-out repos, and removing those features is relatively "
        "better for cold start than for within-project prediction. That is the pre-registered signature of "
        "repo fingerprinting: stripped of the label-replay features, the model identifies repos by their "
        "static attributes and replays their base rates, which cannot carry over to a repo it has never seen."),
    "PARTIAL_SHAP_ONLY": (
        "The repo-level features' contribution tracks actual slow rates more closely on seen repos than on "
        "held-out ones, which is the fingerprinting pattern. But removing those features does not measurably "
        "favour cold start over within-project prediction, so the pattern is not shown to be what costs "
        "cold-start accuracy."),
    "PARTIAL_INTERVENTION_ONLY": (
        "Removing the repo-level features is relatively better for cold start than for within-project "
        "prediction, which is what fingerprinting predicts. But their SHAP contribution does not track actual "
        "slow rates measurably better on seen repos than on held-out ones, so the mechanism fingerprinting "
        "names is not visible in the attributions."),
    "NOT_SUPPORTED": (
        "Neither the attributions nor the intervention separate cold start from within-project prediction in "
        "the way fingerprinting predicts. The cold-start failure is not explained by this mechanism at the "
        "resolution this cohort allows."),
    "CONFLICTING": (
        "The attributions and the intervention point in opposite directions, so this test cannot adjudicate "
        "fingerprinting. Neither result should be reported without the other."),
    "CONTRADICTED": (
        "The evidence points away from fingerprinting: at least one test shows the opposite of the "
        "pre-registered prediction, with an interval excluding zero, and neither confirms it."),
}


def verdict_text(s: dict) -> str:
    t = (f"ρ_A − ρ_B = {s['T']:+.2f}, 95% CI [{s['T_lo']:+.2f}, {s['T_hi']:+.2f}]; ρ_A = {s['rho_a']:+.2f} "
         f"on repos seen in training, ρ_B = {s['rho_b']:+.2f} on held-out repos")
    i = (f"Δ_B − Δ_A = {s['I']:+.3f}, 95% CI [{s['I_lo']:+.3f}, {s['I_hi']:+.3f}]; removing the 8 features "
         f"moved AUC-PR by {s['delta_a']:+.3f} on A and {s['delta_b']:+.3f} on B")
    strong = ("The strong form holds: removing them hurts A and does not hurt B."
              if s["delta_a"] < 0 <= s["delta_b"] else
              "The strong form, a crossover that hurts A while leaving B unharmed, does not hold.")
    band = {"higher": "higher on B", "lower": "lower on B",
            "unchanged": f"essentially unchanged, within ±{fp.RELIANCE_BAND * 100:.0f} points"}[s["reliance"]]
    pts = (s["share_b"] - s["share_a"]) * 100
    return (f"{HEADLINE[s['verdict']]}\n\n"
            f"- SHAP transfer test, **{s['t_outcome']}**: {t}.\n"
            f"- Intervention, **{s['i_outcome']}**: {i}. {strong}\n"
            f"- Reliance (descriptive, never part of the verdict): the 8 features carry {s['share_a']:.1%} of "
            f"mean |SHAP| on A and {s['share_b']:.1%} on B ({pts:+.1f} points), {band}.\n\n"
            f"{READING[s['verdict']]}")


# ---------------------------------------------------------------------------
# Document (pure)
# ---------------------------------------------------------------------------

def render(s: dict | None, checks: list[dict], per_repo_df: pd.DataFrame | None,
           fold_df: pd.DataFrame | None, rel_df: pd.DataFrame | None) -> str:
    gate = md(pd.DataFrame(checks)[["id", "check", "value", "pass"]], "{}")
    head = ("# Phase 6b — The repo-fingerprinting test\n\n"
            "Generated by `report6b.py`: every line below is produced from the data, and there is no "
            f"hand-written section. Pre-registered in `{SPEC}`, committed before any of this ran. The "
            "hypothesis itself is post-hoc (spec §2): it came from an exploratory gain-importance peek, which "
            "this test does not rely on.\n\n")
    stopped = hard_stopped(checks)
    if stopped or s is None:
        failed = [c["id"] for c in checks if not c["pass"]]
        why = (f"Gate check(s) {', '.join(map(str, stopped))} failed, a **HARD STOP**: the SHAP attributions "
               "or the intervention cannot be trusted, so no statistic or verdict is computed from them."
               if stopped else
               f"Gate check(s) {', '.join(map(str, failed))} failed, so the statistics a verdict needs are "
               "not available.")
        return head + f"## No verdict is reported\n\n{why}\n\n## Gate\n\n{gate}\n"
    status = "PASS" if all(c["pass"] for c in checks) else "FAIL"
    c4 = next(c for c in checks if c["id"] == 4)
    gate_note = ""
    if status == "FAIL":
        failed = [c["id"] for c in checks if not c["pass"]]
        gate_note = (f"**The validity gate failed** (check(s) {', '.join(map(str, failed))}): only checks 1 "
                     "and 2 are hard stops, so the verdict below is still reported, but it must be read "
                     "against the gate table below.\n\n")
    feats = ", ".join(f"`{c}`" for c in fp.REPO_FEATURES)
    premise = ("Each is one 2026 snapshot value per repo, so each is constant within every repo. They are the "
               "same eight Phase 4's `NO_SNAPSHOT` ablation removed. `repo_age_days_at_open` is excluded "
               "because it varies within a repo, so the definition can only under-count fingerprinting.\n\n"
               if c4["pass"] else
               "Gate check 4 failed on this data, so the premise that these eight features are constant "
               "within every repo and are exactly what Phase 4's `NO_SNAPSHOT` ablation removed is not "
               "confirmed here. See the gate table above for what this run actually observed.\n\n")
    return (
        head
        + f"## Verdict\n\n{gate_note}{verdict_text(s)}\n\n"
        + f"## Gate: **{status}** (validity, not what the verdict says)\n\n{gate}\n\n"
        + f"## 1. The eight repo-level features\n\n{feats}.\n\n"
        + premise
        + "## 2. Transfer test (SHAP)\n\n"
        + "For each repo, `c` is the mean summed SHAP contribution of the 8 features over its test rows "
          f"(log-odds) and `y` its actual slow rate. ρ_A = {s['rho_a']:+.3f}, ρ_B = {s['rho_b']:+.3f}; "
          f"ρ_A − ρ_B = {s['T']:+.3f}, 95% CI [{s['T_lo']:+.3f}, {s['T_hi']:+.3f}]: **{s['t_outcome']}**.\n\n"
        + f"{md(per_repo_df)}\n\n"
        + "## 3. Intervention\n\n"
        + f"`NLR_NO_REPO` is `NO_LABEL_REPLAY` minus the 8 ({c4['value']['n_no_repo_cols']} features), trained "
          "on the same folds with the same tuned params. AUC-PR follows Phase 4: Scenario B is the mean of its "
          "per-fold values.\n\n"
        + f"{md(fold_df)}\n\n"
        + f"Δ_A = {s['delta_a']:+.4f}, Δ_B = {s['delta_b']:+.4f}; Δ_B − Δ_A = {s['I']:+.4f}, "
          f"95% CI [{s['I_lo']:+.4f}, {s['I_hi']:+.4f}]: **{s['i_outcome']}**.\n\n"
        + "## 4. Reliance (descriptive)\n\n"
        + "Share of mean |SHAP| on the NLR models. The 8 repo-level features together carry "
          f"{s['share_a']:.1%} on A and {s['share_b']:.1%} on B.\n\n"
        + f"{md(rel_df.head(12))}\n\n"
        + "## Method notes\n\n"
        + f"Intervals come from {s['n_draws']:,} paired draws of the {s['n_repos']} repos with replacement; "
          "one draw is applied to both scenarios and both statistics (spec §5.4). Draws dropped as "
          f"non-finite: {s['dropped_T']} for the transfer test, {s['dropped_I']} for the intervention.\n"
    )


# ---------------------------------------------------------------------------
# main -- exercised only by Task 6's real run; the Task 5 reviewer traces it by hand
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    table = ex.load_table()
    params = json.loads(ex.PARAMS.read_text(encoding="utf-8"))["best_params"]
    sha = ex.params_sha()
    nlr_runs = [r for r in attr.load_runs() if r["featureset"] == "NO_LABEL_REPLAY"]
    saved_a = next(r["auc_pr"] for r in nlr_runs if r["scenario"] == "A")

    new_runs = train_no_repo(table, params, sha)
    DATA.mkdir(exist_ok=True)
    RUNS_JSON.write_text(json.dumps({"params_sha": sha, "runs": new_runs}, indent=2), encoding="utf-8")
    refit = refit_delta(table, params, saved_a)
    log.info("refit delta vs Phase 4's saved A/NLR AUC-PR: %.2e", refit)

    rows, svs, additivity, same = {}, {}, {}, {}
    for sc in ("A", "B"):
        rows[sc], svs[sc], add, sm = score_rows(sc, table, nlr_runs, new_runs)
        additivity.update(add)
        same.update(sm)
        log.info("scenario %s: %d rows explained, max additivity delta %.2e", sc, len(rows[sc]), max(add.values()))

    no_repo = fp.nlr_no_repo_cols()
    try:
        fs.assert_hygiene(no_repo)
        hygiene_ok = True
    except ValueError:
        hygiene_ok = False
    integrity = {
        "non_constant": fp.non_constant_within_repo(table, fp.REPO_FEATURES),
        "equals_full_minus_no_snapshot": (set(fp.REPO_FEATURES)
                                          == set(fs.FEATURE_SETS["FULL"]) - set(fs.FEATURE_SETS["NO_SNAPSHOT"])),
        "n_no_repo_cols": len(no_repo), "hygiene_ok": hygiene_ok,
    }
    cov = coverage(nlr_runs, rows["A"], rows["B"])

    pre = gate_checks(additivity, refit, same, integrity, cov, None, {"phase6b_runs": len(new_runs)})
    if hard_stopped(pre):
        GATE_JSON.write_text(json.dumps(pre, indent=2, default=str), encoding="utf-8")
        DOC.write_text(render(None, pre, None, None, None), encoding="utf-8")
        print(f"HARD STOP on gate check(s) {hard_stopped(pre)} -- wrote {DOC}")
        return 1

    boot = fp.paired_repo_bootstrap(rows["A"], rows["B"])
    rel = reliance(svs["A"], svs["B"])
    per_repo_df = (fp.per_repo(rows["A"]).add_suffix("_A")
                   .join(fp.per_repo(rows["B"]).add_suffix("_B")).reset_index())
    fold_df = fold_table(rows)
    per_repo_df.to_csv(DATA / "phase6b_transfer.csv", index=False)
    fold_df.to_csv(DATA / "phase6b_intervention.csv", index=False)
    rel.to_csv(DATA / "phase6b_reliance.csv", index=False)
    artifacts = {"phase6b_transfer": len(per_repo_df), "phase6b_intervention": len(fold_df),
                 "phase6b_reliance": len(rel), "phase6b_runs": len(new_runs)}
    checks = gate_checks(additivity, refit, same, integrity, cov, boot, artifacts)
    GATE_JSON.write_text(json.dumps(checks, indent=2, default=str), encoding="utf-8")

    finite = all(np.isfinite(boot[k]) for k in ("T_lo", "T_hi", "I_lo", "I_hi"))
    if not finite:
        DOC.write_text(render(None, checks, None, None, None), encoding="utf-8")
        print(f"no verdict: non-finite interval -- wrote {DOC}")
        return 1

    share_a = float(rel.loc[rel["is_repo_feature"], "share_a"].sum())
    share_b = float(rel.loc[rel["is_repo_feature"], "share_b"].sum())
    t_out, i_out = fp.outcome(boot["T_lo"], boot["T_hi"]), fp.outcome(boot["I_lo"], boot["I_hi"])
    s = {**boot, "t_outcome": t_out, "i_outcome": i_out, "verdict": fp.verdict(t_out, i_out),
         "share_a": share_a, "share_b": share_b, "reliance": fp.reliance_band(share_b - share_a)}
    DOC.write_text(render(s, checks, per_repo_df, fold_df, rel), encoding="utf-8")
    ok = all(c["pass"] for c in checks)
    print(f"wrote {DOC}  gate={'PASS' if ok else 'FAIL'}  verdict={s['verdict']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

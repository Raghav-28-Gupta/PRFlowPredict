"""The numbers the write-up cites, each recomputed from a committed artifact.

tests/test_writeup_claims.py checks every claim two ways: the recomputed value must render to
the registered string (so a stale artifact or a changed format fails), and that string must
appear in every document the claim lists (so a mistyped or missing number fails).

Sources follow the Phase 8 spec, section 7: a committed data file where one holds the value,
otherwise the committed generated phase document, parsed in its labelled context. Never a
gitignored artifact -- a reader who clones the repo cannot check those."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import featuresets as fs
import fingerprint as fp
import splits

ROOT = Path(__file__).parent
README = "README.md"
REPORT = "docs/REPORT.md"
BOTH = (README, REPORT)

P4_RUNS = "data/phase4_runs.json"
P4_DOC = "docs/phase4_results.md"
P4_PARAMS = "data/phase4_params.json"
P2_GATE = "data/phase2_gate.json"
P2_DOC = "docs/phase2_eda.md"
P3_GATE = "data/phase3_gate.json"
KEPT = "data/cohort/kept.json"
P6_IMP = "data/phase6_importance_{}.csv"
P6_FAIR = "data/phase6_fairness.csv"
P6_DOC = "docs/phase6_interpretation.md"
P6B_DOC = "docs/phase6b_fingerprinting.md"
P6B_TRANSFER = "data/phase6b_transfer.csv"
P6B_INTERVENTION = "data/phase6b_intervention.csv"
P6B_RELIANCE = "data/phase6b_reliance.csv"
EXPERIMENTS = "data/experiments.csv"


@dataclass(frozen=True)
class Claim:
    name: str
    render: Callable[[], str]      # recomputes the value from committed artifacts, then formats it
    expected: str                  # the exact string it must render as
    docs: tuple[str, ...]          # documents the string must appear in
    sources: tuple[str, ...]       # committed paths it reads; the test checks each is git-tracked


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def _text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


@cache
def _runs() -> tuple[dict, ...]:
    return tuple(json.loads(_text(P4_RUNS))["runs"])


def _p4(scenario: str, featureset: str, key: str) -> float:
    """Phase 4's reporting convention: Scenario B is the mean over its five folds."""
    vals = [r[key] for r in _runs() if r["scenario"] == scenario and r["featureset"] == featureset]
    if not vals:
        raise LookupError(f"no {scenario}/{featureset} runs in {P4_RUNS}")
    return float(np.mean(vals))


def _gate(rel: str, prefix: str):
    hits = [c["value"] for c in json.loads(_text(rel)) if c["check"].startswith(prefix)]
    if len(hits) != 1:
        raise LookupError(f"{rel}: {len(hits)} checks start with {prefix!r}")
    return hits[0]


def _parse(rel: str, pattern: str) -> tuple[float, ...]:
    """Parse numbers from a committed generated document, in their labelled context. Exactly
    one match is required, so a pattern that silently matched the wrong sentence fails."""
    hits = re.findall(pattern, _text(rel), flags=re.MULTILINE)
    if len(hits) != 1:
        raise LookupError(f"{rel}: pattern matched {len(hits)} times: {pattern!r}")
    hit = hits[0] if isinstance(hits[0], tuple) else (hits[0],)
    return tuple(float(x) for x in hit)


def _f3(x: float) -> str:
    return f"{x:.3f}"


def _s3(x: float) -> str:
    return f"{x:+.3f}"


def _ci(lo: float, hi: float, fmt=_f3) -> str:
    return f"[{fmt(lo)}, {fmt(hi)}]"


# ---------------------------------------------------------------------------
# Phase 6b values
# ---------------------------------------------------------------------------

@cache
def _transfer() -> pd.DataFrame:
    return pd.read_csv(ROOT / P6B_TRANSFER)


def _rho(s: str) -> float:
    t = _transfer()
    return float(spearmanr(t[f"c_{s}"], t[f"y_{s}"]).statistic)


@cache
def _intervention() -> pd.DataFrame:
    return pd.read_csv(ROOT / P6B_INTERVENTION)


def _delta(s: str) -> float:
    """Delta_s in Phase 4's convention: B is the mean over its folds."""
    iv = _intervention()
    return float(iv.loc[iv["scenario"] == s, "delta"].mean())


def _t_interval() -> tuple[float, float, float]:
    """T and its interval as the Phase 6b document states them, cross-checked against a
    recomputation of T from the committed per-repo table."""
    point, lo, hi = _parse(P6B_DOC, r"ρ_A − ρ_B = ([+-][\d.]+), 95% CI \[([+-][\d.]+), ([+-][\d.]+)\]: \*\*")
    if _s3(point) != _s3(_rho("A") - _rho("B")):
        raise ValueError(f"T in {P6B_DOC} ({point}) disagrees with {P6B_TRANSFER}")
    return point, lo, hi


def _i_interval() -> tuple[float, float, float]:
    point, lo, hi = _parse(P6B_DOC, r"Δ_B − Δ_A = ([+-][\d.]+), 95% CI \[([+-][\d.]+), ([+-][\d.]+)\]: \*\*")
    if _s3(point) != _s3(_delta("B") - _delta("A")):
        raise ValueError(f"I in {P6B_DOC} ({point}) disagrees with {P6B_INTERVENTION}")
    return point, lo, hi


def _verdict() -> str:
    _, tlo, thi = _t_interval()
    _, ilo, ihi = _i_interval()
    return fp.verdict(fp.outcome(tlo, thi), fp.outcome(ilo, ihi))


def _rho_b_descriptive_ci() -> tuple[float, float]:
    """DESCRIPTIVE only, outside the pre-registered verdict: a seeded bootstrap over the 39
    held-out repos' (c, y) pairs, the project's usual 2,000 draws and 95%."""
    t = _transfer()
    c, y, n = t["c_B"].to_numpy(), t["y_B"].to_numpy(), len(t)
    rng = np.random.default_rng(splits.SEED)
    draws = [spearmanr(c[d], y[d]).statistic for d in (rng.choice(n, size=n, replace=True) for _ in range(2000))]
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def _reliance(col: str) -> float:
    rel = pd.read_csv(ROOT / P6B_RELIANCE)
    return float(rel.loc[rel["is_repo_feature"], col].sum())


def _fair_slice(scenario: str) -> tuple[float, float, float]:
    f = pd.read_csv(ROOT / P6_FAIR)
    row = f[(f["scenario"] == scenario) & (f["slice"] == "is_first_pr_here") & (f["level"] == "first-time")]
    if len(row) != 1:
        raise LookupError(f"{P6_FAIR}: {len(row)} first-time rows for scenario {scenario}")
    r = row.iloc[0]
    return float(r["gap"]), float(r["gap_ci_lo"]), float(r["gap_ci_hi"])


def _share(scenario: str, features) -> float:
    imp = pd.read_csv(ROOT / P6_IMP.format(scenario))
    return float(imp.loc[imp["feature"].isin(features), "share"].sum())


# ---------------------------------------------------------------------------
# table rows: a whole row is the claim, so a number in the wrong row fails
# ---------------------------------------------------------------------------

def _main_row(label: str) -> str:
    if label == "P@10, model":
        cell = lambda s: f"{_f3(_p4(s, 'FULL', 'precision_at_10'))} {_ci(_p4(s, 'FULL', 'p10_ci_lo'), _p4(s, 'FULL', 'p10_ci_hi'))}"
    elif label == "P@10, trailing-rate baseline":
        cell = lambda s: _f3(_p4(s, "FULL", "baseline_p10"))
    elif label == "AUC-PR, model":
        cell = lambda s: _f3(_p4(s, "FULL", "auc_pr"))
    elif label == "AUC-PR, trailing-rate baseline":
        cell = lambda s: _f3(_p4(s, "FULL", "baseline_auc_pr"))
    else:
        raise KeyError(label)
    return f"| {label} | {cell('A')} | {cell('B')} |"


def _ablation_row(featureset: str) -> str:
    if featureset == "trailing-rate baseline":
        a, b = _p4("A", "FULL", "baseline_auc_pr"), _p4("B", "FULL", "baseline_auc_pr")
    else:
        a, b = _p4("A", featureset, "auc_pr"), _p4("B", featureset, "auc_pr")
    return f"| {featureset} | {_f3(a)} | {_f3(b)} |"


def _folds_below_baseline() -> int:
    nlr = sorted((r for r in _runs() if r["scenario"] == "B" and r["featureset"] == "NO_LABEL_REPLAY"),
                 key=lambda r: r["fold"])
    return sum(r["auc_pr"] <= r["baseline_auc_pr"] for r in nlr)


def _nlr_b_fold_range() -> tuple[float, float]:
    v = [r["auc_pr"] for r in _runs() if r["scenario"] == "B" and r["featureset"] == "NO_LABEL_REPLAY"]
    return min(v), max(v)


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

CLAIMS: tuple[Claim, ...] = (
    # --- headline, in both documents
    Claim("a_p10", lambda: f"{_f3(_p4('A', 'FULL', 'precision_at_10'))} {_ci(_p4('A', 'FULL', 'p10_ci_lo'), _p4('A', 'FULL', 'p10_ci_hi'))}",
          "0.769 [0.669, 0.856]", BOTH, (P4_RUNS,)),
    Claim("a_baseline_p10", lambda: f"baseline scores {_f3(_p4('A', 'FULL', 'baseline_p10'))}",
          "baseline scores 0.585", BOTH, (P4_RUNS,)),
    Claim("b_full_vs_baseline", lambda: f"{_f3(_p4('B', 'FULL', 'auc_pr'))} vs {_f3(_p4('B', 'FULL', 'baseline_auc_pr'))}",
          "0.859 vs 0.821", BOTH, (P4_RUNS,)),
    Claim("b_nlr_vs_baseline", lambda: f"{_f3(_p4('B', 'NO_LABEL_REPLAY', 'auc_pr'))} vs {_f3(_p4('B', 'FULL', 'baseline_auc_pr'))}",
          "0.763 vs 0.821", BOTH, (P4_RUNS,)),
    Claim("verdict_6b", _verdict, "PARTIAL_SHAP_ONLY", BOTH, (P6B_DOC, P6B_TRANSFER, P6B_INTERVENTION)),

    # --- data and design
    Claim("cohort_kept", lambda: (lambda k: f"{len(k['kept'])} of the {len(k['repos'])}")(json.loads(_text(KEPT))),
          "39 of the 45", (REPORT,), (KEPT,)),
    Claim("d5_rate", lambda: f"{_gate(P2_GATE, 'D5 global is_slow'):.1%} slow under D5",
          "46.4% slow under D5", (REPORT,), (P2_GATE,)),
    Claim("d3_rate", lambda: f"{_parse(P2_DOC, r'^\| D3 \| ([\d.]+) \|')[0]:.1%} under D3",
          "44.0% under D3", (REPORT,), (P2_DOC,)),
    Claim("modelling_rows", lambda: f"{_gate(P3_GATE, 'row count')[0]:,} modelling rows",
          "38,462 modelling rows", (REPORT,), (P3_GATE,)),
    Claim("audit_rows", lambda: (lambda v: f"{v['n']} rows, largest difference {v['max_abs_diff']}")(_gate(P3_GATE, "replay audit")),
          "500 rows, largest difference 0.0", (REPORT,), (P3_GATE,)),
    Claim("tune_cv", lambda: f"cross-validated AUC-PR {_f3(json.loads(_text(P4_PARAMS))['cv_auc_pr'])}",
          "cross-validated AUC-PR 0.907", (REPORT,), (P4_PARAMS,)),

    # --- results tables
    *(Claim(f"main_row::{lab}", (lambda lab=lab: _main_row(lab)), exp, (REPORT,), (P4_RUNS,)) for lab, exp in (
        ("P@10, model", "| P@10, model | 0.769 [0.669, 0.856] | 0.796 [0.614, 0.939] |"),
        ("P@10, trailing-rate baseline", "| P@10, trailing-rate baseline | 0.585 | 0.632 |"),
        ("AUC-PR, model", "| AUC-PR, model | 0.906 | 0.859 |"),
        ("AUC-PR, trailing-rate baseline", "| AUC-PR, trailing-rate baseline | 0.887 | 0.821 |"),
    )),
    *(Claim(f"ablation_row::{f}", (lambda f=f: _ablation_row(f)), exp, (REPORT,), (P4_RUNS,)) for f, exp in (
        ("FULL", "| FULL | 0.906 | 0.859 |"),
        ("NO_SNAPSHOT", "| NO_SNAPSHOT | 0.908 | 0.853 |"),
        ("NO_LABEL_REPLAY", "| NO_LABEL_REPLAY | 0.902 | 0.763 |"),
        ("PR_ONLY", "| PR_ONLY | 0.729 | 0.556 |"),
        ("trailing-rate baseline", "| trailing-rate baseline | 0.887 | 0.821 |"),
    )),

    # --- results prose
    Claim("a_advantage", lambda: f"{_s3(_p4('A', 'FULL', 'precision_at_10') - _p4('A', 'FULL', 'baseline_p10'))} over the baseline",
          "+0.185 over the baseline", (REPORT,), (P4_RUNS,)),
    Claim("a_base_rate", lambda: f"base rate of {_f3(_p4('A', 'FULL', 'base_rate_p10'))}",
          "base rate of 0.641", (REPORT,), (P4_RUNS,)),
    Claim("pool_a", lambda: f"a median of {_parse(P4_DOC, r'^\| A \| \d+ \| \d+ \| ([\d.]+) \|')[0]:.0f} test PRs",
          "a median of 64 test PRs", (REPORT,), (P4_DOC,)),
    Claim("pool_b", lambda: f"against {_parse(P4_DOC, r'^\| B \| \d+ \| \d+ \| ([\d.]+) \|')[0]:.0f} in Scenario B",
          "against 433 in Scenario B", (REPORT,), (P4_DOC,)),
    Claim("nlr_folds_below", lambda: f"{_folds_below_baseline()} of 5 folds",
          "4 of 5 folds", (REPORT,), (P4_RUNS,)),
    Claim("nlr_b_fold_range", lambda: (lambda lo, hi: f"from {_f3(lo)} to {_f3(hi)}")(*_nlr_b_fold_range()),
          "from 0.507 to 0.965", (REPORT,), (P4_RUNS,)),

    # --- Phase 6
    Claim("lr_share_a", lambda: f"{_share('A', fs.LABEL_REPLAY):.1%} in Scenario A",
          "60.9% in Scenario A", (REPORT,), (P6_IMP.format("A"),)),
    Claim("lr_share_b", lambda: f"{_share('B', fs.LABEL_REPLAY):.1%} in Scenario B",
          "63.1% in Scenario B", (REPORT,), (P6_IMP.format("B"),)),
    Claim("author_prior", lambda: f"{_share('A', ['author_prior_slow_rate_here']):.1%} and {_share('B', ['author_prior_slow_rate_here']):.1%}",
          "47.3% and 46.8%", (REPORT,), (P6_IMP.format("A"), P6_IMP.format("B"))),

    # --- Phase 6b
    Claim("rho_a", lambda: f"ρ_A = {_s3(_rho('A'))}", "ρ_A = +0.669", (REPORT,), (P6B_TRANSFER,)),
    Claim("rho_b", lambda: f"ρ_B = {_s3(_rho('B'))}", "ρ_B = +0.318", (REPORT,), (P6B_TRANSFER,)),
    Claim("t_6b", lambda: (lambda p, lo, hi: f"T = {_s3(p)}, 95% CI {_ci(lo, hi, _s3)}")(*_t_interval()),
          "T = +0.351, 95% CI [+0.092, +0.616]", (REPORT,), (P6B_DOC, P6B_TRANSFER)),
    Claim("i_6b", lambda: (lambda p, lo, hi: f"I = {_s3(p)}, 95% CI {_ci(lo, hi, _s3)}")(*_i_interval()),
          "I = +0.035, 95% CI [-0.043, +0.106]", (REPORT,), (P6B_DOC, P6B_INTERVENTION)),
    Claim("delta_a", lambda: f"Δ_A = {_s3(_delta('A'))}", "Δ_A = -0.040", (REPORT,), (P6B_INTERVENTION,)),
    Claim("delta_b", lambda: f"Δ_B = {_s3(_delta('B'))}", "Δ_B = -0.004", (REPORT,), (P6B_INTERVENTION,)),
    Claim("fold_delta_range", lambda: (lambda iv: f"from {_s3(iv.delta.max())} to {_s3(iv.delta.min())}")(_intervention()[_intervention().scenario == "B"]),
          "from +0.245 to -0.131", (REPORT,), (P6B_INTERVENTION,)),
    Claim("reliance", lambda: f"{_reliance('share_a'):.1%} of mean |SHAP| in Scenario A and {_reliance('share_b'):.1%} in Scenario B",
          "30.3% of mean |SHAP| in Scenario A and 40.7% in Scenario B", (REPORT,), (P6B_RELIANCE,)),
    Claim("rho_b_descriptive_ci", lambda: (lambda lo, hi: f"about [{lo:.2f}, {hi:.2f}]")(*_rho_b_descriptive_ci()),
          "about [0.00, 0.57]", (REPORT,), (P6B_TRANSFER,)),

    # --- fairness
    Claim("fair_a", lambda: (lambda p, lo, hi: f"{_s3(p)}, 95% CI {_ci(lo, hi)}")(*_parse(
        P6_DOC, r"\*\*Scenario A: no distinguishable newcomer gap\.\*\* The difference is ([+-]?[\d.]+) with CI \[([+-]?[\d.]+), ([+-]?[\d.]+)\]")),
          "-0.003, 95% CI [-0.091, 0.090]", (REPORT,), (P6_DOC,)),
    Claim("fair_b", lambda: (lambda p, lo, hi: f"{_s3(p)}, 95% CI {_ci(lo, hi)}")(*_parse(
        P6_DOC, r"paired difference \(([+-]?[\d.]+), CI \[([+-]?[\d.]+), ([+-]?[\d.]+)\]")),
          "+0.064, 95% CI [-0.010, 0.143]", (REPORT,), (P6_DOC,)),
    Claim("fair_b_slice", lambda: (lambda g, lo, hi: f"{_s3(g)}, 95% CI {_ci(lo, hi)}")(*_fair_slice("B")),
          "+0.067, 95% CI [0.002, 0.136]", (REPORT,), (P6_FAIR,)),

    # --- what was tried
    Claim("n_runs", lambda: f"{len(pd.read_csv(ROOT / EXPERIMENTS))} runs", "36 runs", (REPORT,), (EXPERIMENTS,)),
)

# Honesty constraints that carry no number (spec section 9): phrases that must appear, and
# phrasings that must not. The forbidden list guards section 9.7 -- never upgrade 6b.
REQUIRED_PHRASES: tuple[tuple[str, str], ...] = (
    (REPORT, "bar was never measured"),
    (REPORT, "C-index above 0.65"),
    (REPORT, "pool-size artifact"),
    (REPORT, "post-hoc"),
    (REPORT, "only marginally distinguishable"),
    (REPORT, "cannot distinguish a small intervention effect from none"),
    (REPORT, "marginal null"),
    (REPORT, "small sample"),
)
FORBIDDEN_PATTERNS: tuple[str, ...] = (
    r"fingerprinting (is|was) (supported|confirmed)",
    r"confirm(s|ed)? (repo )?fingerprinting",
    r"\bprov(e|es|ed|en)\b",
)

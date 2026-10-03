"""Phase 6b: is the NO_LABEL_REPLAY model's cold-start failure repo fingerprinting?

Hypothesis (post-hoc; see the spec's section 2): stripped of the label-replay features, the
model uses repo-level features, which are constant within a repo, to IDENTIFY repos and
memorise their base rates. That is label replay by proxy, and it cannot transfer to a repo
the model has never seen.

Pre-registered in docs/design/specs/2026-09-24-phase6b-fingerprinting-design.md,
committed before any of this ran. Every function here is pure; report6b.py does the I/O."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import features as F
import featuresets as fs
import metrics
import splits

SEED = splits.SEED
NO_REPO_NAME = "NLR_NO_REPO"
RELIANCE_BAND = 0.05          # Phase 6's band: within +/-5 points counts as unchanged

# group == 'repo' AND status == 'snapshot': one 2026 value per repo, hence constant within
# every repo -- pure repo identifiers. repo_age_days_at_open is also group 'repo' but varies
# within a repo, so it is excluded: the definition can only UNDER-count fingerprinting.
REPO_FEATURES: tuple[str, ...] = tuple(
    c for c, m in F.COLUMN_SPEC.items() if m.get("group") == "repo" and m.get("status") == "snapshot")


def nlr_no_repo_cols() -> list[str]:
    """The intervention's feature set: NO_LABEL_REPLAY minus REPO_FEATURES, in NLR's order.

    Deliberately NOT registered in fs.FEATURE_SETS: experiment.main() trains every registered
    set and writes them all to phase4_runs.json."""
    return [c for c in fs.FEATURE_SETS["NO_LABEL_REPLAY"] if c not in REPO_FEATURES]


def non_constant_within_repo(table: pd.DataFrame, feats, repo_col: str = "repo") -> list[str]:
    """Features taking more than one value inside at least one repo. An empty list means
    the fingerprinting premise -- one value per repo -- holds."""
    n = table.groupby(repo_col)[list(feats)].nunique(dropna=False)
    return [c for c in feats if (n[c] > 1).any()]


def repo_contribution(shap_values: np.ndarray, cols: list[str]) -> np.ndarray:
    """Per row: the summed SHAP values of REPO_FEATURES, in raw-margin (log-odds) units."""
    missing = [c for c in REPO_FEATURES if c not in cols]
    if missing:
        raise ValueError(f"SHAP columns lack repo-level features: {missing}")
    idx = [cols.index(c) for c in REPO_FEATURES]
    return np.asarray(shap_values, dtype=float)[:, idx].sum(axis=1)


def per_repo(rows: pd.DataFrame) -> pd.DataFrame:
    """Spec section 5.2's c_r and y_r: per repo, the mean repo contribution and the actual
    slow rate over its test rows, plus the row count. Keyed by repo label, never position."""
    g = rows.groupby("repo")
    return pd.DataFrame({"c": g["repo_contrib"].mean().astype(float),
                         "y": g["is_slow"].mean().astype(float),
                         "n": g.size()})


def spearman(x, y) -> float:
    return float(spearmanr(np.asarray(x, dtype=float), np.asarray(y, dtype=float)).statistic)


def outcome(lo: float, hi: float) -> str:
    """Spec section 6: a 95% interval resolves to exactly one of three outcomes. Touching
    zero is not excluding it. A non-finite bound is a broken computation, not a result."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"non-finite interval ({lo}, {hi})")
    if lo > 0:
        return "confirms"
    if hi < 0:
        return "contradicts"
    return "inconclusive"


# Spec section 6, verbatim. (T, I) -> verdict; T = the SHAP transfer test, I = the intervention.
VERDICTS: dict[tuple[str, str], str] = {
    ("confirms", "confirms"): "SUPPORTED",
    ("confirms", "inconclusive"): "PARTIAL_SHAP_ONLY",
    ("inconclusive", "confirms"): "PARTIAL_INTERVENTION_ONLY",
    ("inconclusive", "inconclusive"): "NOT_SUPPORTED",
    ("confirms", "contradicts"): "CONFLICTING",
    ("contradicts", "confirms"): "CONFLICTING",
    ("inconclusive", "contradicts"): "CONTRADICTED",
    ("contradicts", "inconclusive"): "CONTRADICTED",
    ("contradicts", "contradicts"): "CONTRADICTED",
}


def verdict(t: str, i: str) -> str:
    return VERDICTS[(t, i)]


def reliance_band(d: float) -> str:
    """Descriptive only (spec section 5.1): share_B - share_A on REPO_FEATURES."""
    if d > RELIANCE_BAND:
        return "higher"
    if d < -RELIANCE_BAND:
        return "lower"
    return "unchanged"


ROW_COLS = ("repo", "fold", "is_slow", "p_nlr", "p_no_repo", "repo_contrib")


def mean_fold_auc(y, p, fold) -> float:
    """Phase 4's AUC-PR convention: report4 summarises Scenario B by the MEAN of its
    per-fold AUC-PRs, not a pooled figure. A fold lacking both classes is skipped --
    average precision is undefined there."""
    y, p, fold = np.asarray(y, dtype=int), np.asarray(p, dtype=float), np.asarray(fold)
    vals = [metrics.auc_pr(y[m], p[m]) for m in (fold == k for k in np.unique(fold))
            if 0 < y[m].sum() < m.sum()]
    return float(np.mean(vals)) if vals else float("nan")


def auc_delta(y, p_new, p_old, fold) -> float:
    """AUC-PR(new) - AUC-PR(old) in the mean-of-folds convention, on identical rows."""
    return mean_fold_auc(y, p_new, fold) - mean_fold_auc(y, p_old, fold)


def _prepare(rows: pd.DataFrame, repos: list[str]):
    r = rows.reset_index(drop=True)
    pr = per_repo(r).reindex(repos)
    pos = r.groupby("repo").indices                       # repo -> row positions
    return (pr["c"].to_numpy(dtype=float), pr["y"].to_numpy(dtype=float),
            [np.asarray(pos[x]) for x in repos],
            r["is_slow"].to_numpy(dtype=int), r["p_no_repo"].to_numpy(dtype=float),
            r["p_nlr"].to_numpy(dtype=float), r["fold"].to_numpy())


def paired_repo_bootstrap(rows_a: pd.DataFrame, rows_b: pd.DataFrame, *, n: int = 2000,
                          seed: int = SEED, ci: float = 0.95) -> dict:
    """Spec section 5.4: ONE set of repo draws, applied to both scenarios and both statistics.

      T = rho_A - rho_B                          the SHAP transfer test
      I = Delta_B - Delta_A,  Delta_s = AUC-PR(no_repo) - AUC-PR(nlr), mean-of-folds

    c_r and y_r do not change under resampling, so T resamples per-repo PAIRS with
    multiplicity. AUC-PR is not decomposable per repo, so I resamples the drawn repos' ROWS
    with multiplicity. The point estimates go through the same code path, on the identity
    draw (every repo exactly once)."""
    for name, rows in (("A", rows_a), ("B", rows_b)):
        missing = sorted(set(ROW_COLS) - set(rows.columns))
        if missing:
            raise ValueError(f"rows_{name} lacks {missing}")
    repos = sorted(rows_a["repo"].unique())
    if sorted(rows_b["repo"].unique()) != repos:
        raise ValueError("A and B must cover the same repos -- the pairing premise (spec section 5)")
    ca, ya_r, ia, ya, na, oa, fa = _prepare(rows_a, repos)
    cb, yb_r, ib, yb, nb, ob, fb = _prepare(rows_b, repos)

    def components(d: np.ndarray) -> tuple[float, float, float, float]:
        xa, xb = np.concatenate([ia[j] for j in d]), np.concatenate([ib[j] for j in d])
        return (spearman(ca[d], ya_r[d]), spearman(cb[d], yb_r[d]),
                auc_delta(ya[xa], na[xa], oa[xa], fa[xa]), auc_delta(yb[xb], nb[xb], ob[xb], fb[xb]))

    rho_a, rho_b, delta_a, delta_b = components(np.arange(len(repos)))
    rng = np.random.default_rng(seed)
    ts, is_ = np.empty(n), np.empty(n)
    for k in range(n):
        ra, rb, da, db = components(rng.choice(len(repos), size=len(repos), replace=True))
        ts[k], is_[k] = ra - rb, db - da
    lo_q = (1.0 - ci) / 2.0

    def interval(v: np.ndarray) -> tuple[float, float]:
        v = v[np.isfinite(v)]
        if not len(v):
            return float("nan"), float("nan")
        return float(np.quantile(v, lo_q)), float(np.quantile(v, 1.0 - lo_q))

    t_lo, t_hi = interval(ts)
    i_lo, i_hi = interval(is_)
    return {"n_repos": len(repos), "rho_a": rho_a, "rho_b": rho_b, "T": rho_a - rho_b,
            "T_lo": t_lo, "T_hi": t_hi, "delta_a": delta_a, "delta_b": delta_b,
            "I": delta_b - delta_a, "I_lo": i_lo, "I_hi": i_hi, "n_draws": n,
            "dropped_T": int((~np.isfinite(ts)).sum()), "dropped_I": int((~np.isfinite(is_)).sum())}

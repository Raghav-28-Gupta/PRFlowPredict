"""Phase 6b: is the NO_LABEL_REPLAY model's cold-start failure repo fingerprinting?

Hypothesis (post-hoc; see the spec's section 2): stripped of the label-replay features, the
model uses repo-level features, which are constant within a repo, to IDENTIFY repos and
memorise their base rates. That is label replay by proxy, and it cannot transfer to a repo
the model has never seen.

Pre-registered in docs/superpowers/specs/2026-09-24-phase6b-fingerprinting-design.md,
committed before any of this ran. Every function here is pure; report6b.py does the I/O."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import features as F
import featuresets as fs
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

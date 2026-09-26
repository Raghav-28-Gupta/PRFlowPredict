import inspect
import re

import numpy as np
import pandas as pd
import pytest

import featuresets as fs
import fingerprint as fp
import report6b

EIGHT = ["n_assignable_users", "n_mentionable_users", "owner_is_org", "has_codeowners",
         "has_pr_template", "has_contributing", "n_ci_workflows", "language_dominant"]


# ---------------------------------------------------------------------------
# The definition
# ---------------------------------------------------------------------------

def test_repo_features_are_exactly_the_eight_snapshot_repo_columns():
    assert sorted(fp.REPO_FEATURES) == sorted(EIGHT)
    # the same eight Phase 4's NO_SNAPSHOT ablation removed: ties the definition to Phase 4
    removed = set(fs.FEATURE_SETS["FULL"]) - set(fs.FEATURE_SETS["NO_SNAPSHOT"])
    assert set(fp.REPO_FEATURES) == removed
    # group 'repo' but varies within a repo: deliberately excluded
    assert "repo_age_days_at_open" not in fp.REPO_FEATURES


def test_nlr_no_repo_is_nlr_minus_exactly_the_eight():
    nlr = fs.FEATURE_SETS["NO_LABEL_REPLAY"]
    cols = fp.nlr_no_repo_cols()
    assert len(nlr) == 33 and len(cols) == 25
    assert set(nlr) - set(cols) == set(EIGHT)
    assert [c for c in nlr if c in cols] == cols          # NLR's own order preserved
    fs.assert_hygiene(cols)


def test_non_constant_within_repo_flags_only_the_varying_feature():
    t = pd.DataFrame({"repo": ["a", "a", "b", "b"],
                      "k": [1, 1, 2, 2],                 # constant within each repo, not globally
                      "v": [1, 2, 3, 3]})                # varies inside repo a only
    assert fp.non_constant_within_repo(t, ["k", "v"]) == ["v"]
    assert fp.non_constant_within_repo(t, ["k"]) == []


# ---------------------------------------------------------------------------
# Per-row and per-repo quantities
# ---------------------------------------------------------------------------

def test_repo_contribution_sums_only_the_repo_feature_columns_per_row():
    cols = ["x"] + EIGHT
    sv = np.zeros((3, len(cols)))
    sv[:, 0] = 100.0                       # a big NON-repo contribution that must be ignored
    sv[0, 1], sv[0, 2] = 1.0, 0.5          # row 0: two repo features -> 1.5
    sv[1, 8] = -2.0                        # row 1: language_dominant, the LAST column -> -2.0
    assert fp.repo_contribution(sv, cols).tolist() == pytest.approx([1.5, -2.0, 0.0])


def test_repo_contribution_requires_every_repo_feature():
    cols = ["x"] + EIGHT[:-1]
    with pytest.raises(ValueError, match="language_dominant"):
        fp.repo_contribution(np.zeros((2, len(cols))), cols)


def test_per_repo_is_a_row_mean_keyed_by_repo_label():
    rows = pd.DataFrame({"repo": ["b", "a", "b", "a", "a"],        # interleaved on purpose
                         "repo_contrib": [2.0, 1.0, 6.0, 3.0, 5.0],
                         "is_slow": [1, 0, 0, 1, 1]})
    t = fp.per_repo(rows)
    assert list(t.columns) == ["c", "y", "n"]
    assert t.loc["a"].tolist() == pytest.approx([3.0, 2 / 3, 3])
    assert t.loc["b"].tolist() == pytest.approx([4.0, 0.5, 2])


def test_spearman_is_rank_based_not_linear():
    # monotone but nonlinear: Spearman is exactly 1, Pearson would be about 0.993
    assert fp.spearman([1, 2, 3, 4], [1, 4, 9, 100]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Outcomes and the pre-registered verdict (spec §6)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lo,hi,expected", [
    (0.01, 0.5, "confirms"),
    (-0.5, -0.01, "contradicts"),
    (-0.1, 0.2, "inconclusive"),
    (0.0, 0.3, "inconclusive"),          # touching zero is not excluding it
    (-0.3, 0.0, "inconclusive"),
])
def test_outcome(lo, hi, expected):
    assert fp.outcome(lo, hi) == expected


def test_outcome_rejects_a_non_finite_interval():
    with pytest.raises(ValueError):
        fp.outcome(float("nan"), 0.3)


# Copied from spec §6 on purpose: this test IS the pre-registration, checked against the code.
SPEC_TABLE = {
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


@pytest.mark.parametrize("cell", list(SPEC_TABLE))
def test_verdict_matches_the_spec_table_cell_by_cell(cell):
    assert fp.verdict(*cell) == SPEC_TABLE[cell]


def test_verdict_rejects_an_unknown_outcome():
    with pytest.raises(KeyError):
        fp.verdict("confirms", "maybe")


@pytest.mark.parametrize("d,expected", [
    (0.06, "higher"), (-0.06, "lower"), (0.05, "unchanged"), (-0.05, "unchanged"), (0.0, "unchanged"),
])
def test_reliance_band(d, expected):
    assert fp.reliance_band(d) == expected


# ---------------------------------------------------------------------------
# Phase 4's AUC-PR convention
# ---------------------------------------------------------------------------

import metrics


def test_mean_fold_auc_is_the_mean_of_per_fold_values_not_pooled():
    # Each fold is perfectly ranked internally (per-fold AP = 1.0), but fold 1's scores all
    # sit BELOW fold 0's, so pooling ranks fold 1's positive under fold 0's negative.
    y = np.array([0, 1, 0, 1])
    p = np.array([0.8, 0.9, 0.1, 0.2])
    fold = np.array([0, 0, 1, 1])
    assert fp.mean_fold_auc(y, p, fold) == pytest.approx(1.0)
    assert metrics.auc_pr(y, p) < 0.99          # the pooled figure differs, so the test can tell


def test_mean_fold_auc_skips_a_single_class_fold(recwarn):
    y = np.array([0, 1, 0, 1, 0, 0])
    p = np.array([0.8, 0.9, 0.1, 0.2, 0.5, 0.6])
    fold = np.array([0, 0, 1, 1, 2, 2])           # fold 2 has no positives
    assert fp.mean_fold_auc(y, p, fold) == pytest.approx(1.0)
    assert not [w for w in recwarn if "positive" in str(w.message).lower()]


def test_auc_delta_is_new_minus_old():
    y = np.array([0, 1, 0, 1]); fold = np.zeros(4, dtype=int)
    good = np.array([0.1, 0.9, 0.2, 0.8]); bad = np.array([0.9, 0.1, 0.8, 0.2])
    assert fp.auc_delta(y, good, bad, fold) > 0 > fp.auc_delta(y, bad, good, fold)


# ---------------------------------------------------------------------------
# The paired repo bootstrap -- deterministic fixtures
# ---------------------------------------------------------------------------

R, N_PER = 12, 40
PERM = [4, 9, 5, 1, 7, 8, 6, 3, 11, 2, 0, 10]          # Spearman(PERM, range(12)) == 0.0 exactly
POS = [round(r * N_PER) for r in np.linspace(0.15, 0.85, R)]   # 6, 9, 11, ... 34: strictly increasing
TRACK = [float(r) for r in range(R)]                   # contribution rank == slow-rate rank
SCRAMBLED = [float(p) for p in PERM]                   # rank-uncorrelated with the slow rate
NOISY = [0.0, 2.0, 1.0, 3.0, 5.0, 4.0, 6.0, 8.0, 7.0, 9.0, 11.0, 10.0]   # tracks, imperfectly
A_FOLDS, B_FOLDS = [0] * R, [r % 3 for r in range(R)]


def _rows(contrib, folds, *, sd_nlr=0.5, sd_new=None, seed=0):
    """Scored rows in the ROW_COLS schema. Repo r has exactly POS[r] slow PRs out of N_PER.
    Each model scores y + N(0, sd), so a larger sd is a worse model. sd_new=None scores the
    intervention IDENTICALLY to NLR: removing the features changes nothing."""
    rng = np.random.default_rng(seed)
    parts = []
    for r in range(R):
        y = np.r_[np.ones(POS[r]), np.zeros(N_PER - POS[r])].astype(int)
        p_nlr = y + rng.normal(0, sd_nlr, N_PER)
        p_new = p_nlr.copy() if sd_new is None else y + rng.normal(0, sd_new, N_PER)
        parts.append(pd.DataFrame({"repo": f"r{r:02d}", "fold": folds[r], "is_slow": y,
                                   "p_nlr": p_nlr, "p_no_repo": p_new,
                                   "repo_contrib": np.full(N_PER, contrib[r])}))
    return pd.concat(parts, ignore_index=True)


def test_bootstrap_confirms_a_planted_fingerprint():
    """A: the repo contribution tracks the slow rate (rho_A = 1). B: rank-uncorrelated
    (rho_B = 0). The intervention is scored identically, so I must stay exactly null --
    showing the two tests are independent."""
    b = fp.paired_repo_bootstrap(_rows(TRACK, A_FOLDS, seed=1), _rows(SCRAMBLED, B_FOLDS, seed=2), n=300)
    assert b["rho_a"] == pytest.approx(1.0) and b["rho_b"] == pytest.approx(0.0, abs=1e-12)
    assert fp.outcome(b["T_lo"], b["T_hi"]) == "confirms"
    assert b["I"] == 0.0 and fp.outcome(b["I_lo"], b["I_hi"]) == "inconclusive"


def test_bootstrap_draws_are_paired_across_scenarios():
    """Two IDENTICAL scenarios, with imperfect tracking (so Spearman varies from subset to
    subset) and a real intervention effect. Under ONE shared draw, both statistics are
    exactly zero in every draw; independent draws per scenario would make them vary. This
    test is what pins the pairing (spec section 5.4). Perfect tracking could not: Spearman
    is 1 on every subset whether or not the draws are shared."""
    a = _rows(NOISY, A_FOLDS, sd_nlr=0.3, sd_new=0.9, seed=11)
    b = fp.paired_repo_bootstrap(a, a.copy(), n=300)
    assert (b["T"], b["T_lo"], b["T_hi"]) == (0.0, 0.0, 0.0)
    assert (b["I"], b["I_lo"], b["I_hi"]) == (0.0, 0.0, 0.0)
    assert fp.outcome(b["T_lo"], b["T_hi"]) == fp.outcome(b["I_lo"], b["I_hi"]) == "inconclusive"


def test_bootstrap_confirms_a_planted_intervention_crossover():
    """Removing the features hurts A (sd 0.3 -> 1.2) and helps B (sd 1.2 -> 0.3). Both
    scenarios track, so the transfer test must stay exactly null."""
    b = fp.paired_repo_bootstrap(_rows(TRACK, A_FOLDS, sd_nlr=0.3, sd_new=1.2, seed=3),
                                 _rows(TRACK, B_FOLDS, sd_nlr=1.2, sd_new=0.3, seed=4), n=300)
    assert b["delta_a"] < 0 < b["delta_b"]
    assert fp.outcome(b["I_lo"], b["I_hi"]) == "confirms"
    assert b["T"] == 0.0


def test_bootstrap_point_estimates_match_a_direct_computation():
    a = _rows(TRACK, A_FOLDS, sd_nlr=0.3, sd_new=0.6, seed=7)
    b_ = _rows(SCRAMBLED, B_FOLDS, sd_nlr=0.6, sd_new=0.3, seed=8)
    out = fp.paired_repo_bootstrap(a, b_, n=20)
    pa, pb = fp.per_repo(a), fp.per_repo(b_)
    assert out["rho_a"] == pytest.approx(fp.spearman(pa["c"], pa["y"]))
    assert out["rho_b"] == pytest.approx(fp.spearman(pb["c"], pb["y"]))
    assert out["delta_a"] == pytest.approx(fp.auc_delta(a["is_slow"], a["p_no_repo"], a["p_nlr"], a["fold"]))
    assert out["delta_b"] == pytest.approx(fp.auc_delta(b_["is_slow"], b_["p_no_repo"], b_["p_nlr"], b_["fold"]))
    assert out["T"] == pytest.approx(out["rho_a"] - out["rho_b"])
    assert out["I"] == pytest.approx(out["delta_b"] - out["delta_a"])
    assert out["n_repos"] == R and out["n_draws"] == 20


def test_bootstrap_is_seed_deterministic():
    a = _rows(TRACK, A_FOLDS, sd_nlr=0.3, sd_new=0.6, seed=9)
    b_ = _rows(SCRAMBLED, B_FOLDS, sd_nlr=0.6, sd_new=0.3, seed=10)
    one = fp.paired_repo_bootstrap(a, b_, n=200, seed=7)
    assert one == fp.paired_repo_bootstrap(a, b_, n=200, seed=7)
    other = fp.paired_repo_bootstrap(a, b_, n=200, seed=8)
    assert (one["T_lo"], one["I_lo"]) != (other["T_lo"], other["I_lo"])


def test_bootstrap_refuses_unpaired_repo_sets():
    a, b_ = _rows(TRACK, A_FOLDS), _rows(SCRAMBLED, B_FOLDS)
    with pytest.raises(ValueError, match="pairing"):
        fp.paired_repo_bootstrap(a, b_[b_["repo"] != "r00"], n=10)


def test_bootstrap_refuses_rows_missing_a_column():
    a, b_ = _rows(TRACK, A_FOLDS), _rows(SCRAMBLED, B_FOLDS)
    with pytest.raises(ValueError, match="p_no_repo"):
        fp.paired_repo_bootstrap(a.drop(columns="p_no_repo"), b_, n=10)


# ---------------------------------------------------------------------------
# The pre-registered defaults (spec section 5.4) -- main() relies on these being unpinned
# by any call-site override.
# ---------------------------------------------------------------------------

def test_paired_repo_bootstrap_defaults_are_pinned():
    params = inspect.signature(fp.paired_repo_bootstrap).parameters
    assert fp.SEED == 20260912
    assert params["n"].default == 2000
    assert params["seed"].default == fp.SEED
    assert params["ci"].default == 0.95


def test_main_calls_bootstrap_without_overriding_the_pre_registered_defaults():
    src = inspect.getsource(report6b.main)
    match = re.search(r"fp\.paired_repo_bootstrap\([^)]*\)", src)
    assert match is not None
    assert match.group(0) == 'fp.paired_repo_bootstrap(rows["A"], rows["B"])'

import numpy as np
import pandas as pd
import pytest

import featuresets as fs
import fingerprint as fp

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

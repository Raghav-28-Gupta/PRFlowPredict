"""The committed demo extract reproduces Phase 4 and keeps its promises (Phase 7 spec, section 9).

Runs from a plain clone: it reads only demo/data/prs.parquet and data/phase4_runs.json."""
import json
import re

import numpy as np
import pandas as pd
import pytest

import build_demo_data as bdd
import metrics
import splits

RUNS = bdd.ROOT / "data" / "phase4_runs.json"


@pytest.fixture(scope="module")
def prs():
    return pd.read_parquet(bdd.OUT)


@pytest.fixture(scope="module")
def runs():
    return {(r["scenario"], r["featureset"], r["fold"]): r
            for r in json.loads(RUNS.read_text(encoding="utf-8"))["runs"]}


def test_scenario_a_auc_pr_and_baseline_match_phase4(prs, runs):
    a = runs[("A", "FULL", 0)]
    y = prs["is_slow"].astype(int)
    assert metrics.auc_pr(y, prs["score_a"]) == pytest.approx(a["auc_pr"], abs=1e-12)
    assert metrics.auc_pr(y, prs["baseline_score"]) == pytest.approx(a["baseline_auc_pr"], abs=1e-12)


def test_scenario_a_p10_matches_phase4_in_the_extracts_row_order(prs, runs):
    """P@10 breaks ties with a seeded permutation over the row order. The model's scores rarely
    tie, but the baseline's barely vary within a repo, so its P@10 only reproduces if the
    extract kept the predictions file's row order exactly."""
    def p10(col):
        return metrics.precision_at_k(prs["is_slow"].to_numpy(dtype=int), prs[col].to_numpy(),
                                      prs["repo"].to_numpy(), k=10, seed=splits.SEED)[1]
    a = runs[("A", "FULL", 0)]
    assert p10("score_a") == pytest.approx(a["precision_at_10"], abs=1e-12)
    assert p10("baseline_score") == pytest.approx(a["baseline_p10"], abs=1e-12)


def test_each_b_fold_holds_exactly_its_phase4_test_repos(prs, runs):
    for k in range(5):
        assert set(prs.loc[prs["fold_b"] == k, "repo"]) == set(runs[("B", "FULL", k)]["test_repos"])


def test_shape_columns_and_completeness(prs):
    assert list(prs.columns) == bdd.COLUMNS
    assert len(prs) == 14_135 and prs["pr_id"].is_unique and prs["repo"].nunique() == 39
    assert prs[["score_a", "score_b", "number", "url", "title"]].notna().all().all()
    assert prs["created_at"].min() >= pd.Timestamp("2026-01-01", tz="UTC")
    assert prs["created_at"].max() < pd.Timestamp("2026-07-01", tz="UTC")


def test_every_row_has_three_drivers_per_model(prs):
    for col in ("drivers_a", "drivers_b"):
        parts = prs[col].str.split(" · ")
        assert parts.map(len).eq(3).all() and parts.map(lambda p: all(x.strip() for x in p)).all()


def test_the_slow_flag_agrees_with_the_first_review_time(prs):
    wait_h = (prs["first_review_at"] - prs["created_at"]).dt.total_seconds() / 3600
    assert ((prs["first_review_at"].isna() | (wait_h > 168)) == prs["is_slow"]).all()


def test_no_author_identity_is_stored(prs):
    """Feature names such as author_prior_slow_rate_here are fine; logins and ids are not."""
    assert not [c for c in prs.columns if re.search(r"login|database_id|author_id|author_name|email", c)]


@pytest.mark.parametrize("s", ["a", "b"])
def test_each_rows_shap_values_add_up_to_its_score(prs, s):
    margin = prs[f"base_{s}"] + prs[[f"shap_{s}__{c}" for c in bdd.FULL]].astype(float).sum(axis=1)
    logit = np.log(prs[f"score_{s}"] / (1 - prs[f"score_{s}"]))
    assert (margin - logit).abs().max() < 1e-4


@pytest.mark.parametrize("s", ["a", "b"])
def test_each_rows_drivers_text_is_recomputable_from_its_stored_values(prs, s):
    for _, row in prs.sample(300, random_state=1).iterrows():
        shap_row = np.array([row[f"shap_{s}__{c}"] for c in bdd.FULL], dtype=float)
        x_row = pd.Series({c: row[f"x__{c}"] for c in bdd.FULL})
        assert bdd.format_drivers(shap_row, x_row) == row[f"drivers_{s}"]


def test_features_json_is_what_the_build_writes():
    assert json.loads((bdd.OUT.parent / bdd.FEATURES_JSON).read_text(encoding="utf-8")) == bdd.feature_list()

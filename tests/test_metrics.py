import numpy as np
import pandas as pd
import pytest
import metrics


def test_perfect_ranking_gives_precision_one():
    y = [1] * 10 + [0] * 5
    s = [1.0] * 10 + [0.0] * 5
    per_repo, mean = metrics.precision_at_k(y, s, ["X"] * 15, k=10)
    assert per_repo["X"] == 1.0 and mean == 1.0


def test_fewer_than_k_uses_all_rows():
    per_repo, _ = metrics.precision_at_k([1, 0, 0, 1], [0.9, 0.8, 0.7, 0.6], ["Y"] * 4, k=10)
    assert per_repo["Y"] == 0.5


def test_constant_score_is_random_not_first_n():
    # 100 rows: first 50 positive, last 50 negative, constant score. If ties were
    # broken by row order, P@10 would be 1.0 every time. It must vary with the seed.
    y = [1] * 50 + [0] * 50
    s = [0.5] * 100
    vals = {metrics.precision_at_k(y, s, ["Z"] * 100, k=10, seed=i)[1] for i in range(20)}
    assert len(vals) > 1
    assert 0.0 <= min(vals) and max(vals) <= 1.0


def test_mean_is_over_repos_not_rows():
    y = [1] * 10 + [0] * 100
    s = [1.0] * 10 + [0.0] * 100
    repo = ["A"] * 10 + ["B"] * 100
    _, mean = metrics.precision_at_k(y, s, repo, k=10)
    assert mean == 0.5    # A=1.0, B=0.0, mean over the two repos


def test_auc_pr_perfect_and_random():
    assert metrics.auc_pr([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert 0 < metrics.auc_pr([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) <= 1.0


def test_cluster_bootstrap_constant_and_contains_mean():
    lo, hi = metrics.cluster_bootstrap(pd.Series([0.3, 0.3, 0.3]))
    assert lo == pytest.approx(0.3) and hi == pytest.approx(0.3)
    v = pd.Series(np.linspace(0, 1, 30))
    lo, hi = metrics.cluster_bootstrap(v)
    assert lo < v.mean() < hi

import numpy as np
import pandas as pd
import pytest

import fairness


def _pred(first_gap=0.30, repeat_gap=0.0, n_repos=8, per_repo=60, seed=0):
    """p_hat = 0.5*actual + gap, so a group's mean gap is (0.5*rate + gap) - rate.

    Both groups therefore carry the same -0.5*rate term, and it cancels in the PAIRED
    difference gap_difference() computes -- leaving exactly (first_gap - repeat_gap).
    That is what the difference tests below assert."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_repos):
        for i in range(per_repo):
            first = i % 2 == 0
            actual = bool(rng.random() < 0.5)
            gap = first_gap if first else repeat_gap
            rows.append({"repo": f"r{r}", "is_slow": actual,
                         "p_hat": float(np.clip(0.5 * actual + gap, 0, 1)),
                         "is_first_pr_here": first,
                         "created_hour_utc": (i * 4) % 24})
    return pd.DataFrame(rows)


def test_slice_gaps_has_all_six_levels_with_intervals():
    pred = _pred()
    s = fairness.slice_gaps(pred, seed=1)
    assert set(s.columns) == {"slice", "level", "n", "n_repos", "actual_rate",
                              "mean_p_hat", "gap", "gap_ci_lo", "gap_ci_hi"}
    assert set(s[s["slice"] == "is_first_pr_here"]["level"]) == {"first-time", "repeat"}
    assert set(s[s["slice"] == "hour_bucket"]["level"]) == {"00-06", "06-12", "12-18", "18-24"}
    assert (s["gap_ci_lo"] <= s["gap"]).all() and (s["gap"] <= s["gap_ci_hi"]).all()
    assert (s["n_repos"] == 8).all()
    # each slicing partitions every row exactly once, so the two together double-count
    assert s["n"].sum() == 2 * len(pred)


def test_slice_gaps_recovers_a_planted_gap():
    s = fairness.slice_gaps(_pred(first_gap=0.30, repeat_gap=0.0), seed=1).set_index("level")
    assert s.loc["first-time", "gap"] > s.loc["repeat", "gap"] + 0.15


def test_gap_difference_detects_a_real_gap():
    d = fairness.gap_difference(_pred(first_gap=0.30, repeat_gap=0.0), seed=1)
    assert set(d) == {"difference", "ci_lo", "ci_hi", "ci_excludes_zero",
                      "n_first_time", "n_repeat"}
    assert d["difference"] > 0.15 and d["ci_excludes_zero"] is True
    assert d["ci_lo"] <= d["difference"] <= d["ci_hi"]


def test_gap_difference_is_honest_when_there_is_no_gap():
    """The negative control: no planted gap must not produce a 'real' one.

    With 8 repos the paired differences are centred on 0 with sd ~0.065, so a bad seed
    could land a CI just off zero. If this fails, raise n_repos/per_repo in the _pred()
    call -- MORE data, never a loosened assertion. `ci_excludes_zero is False` is the
    whole point of the test and must not be weakened."""
    d = fairness.gap_difference(_pred(first_gap=0.0, repeat_gap=0.0), seed=1)
    assert abs(d["difference"]) < 0.05 and d["ci_excludes_zero"] is False


def test_gap_difference_is_paired_by_repo():
    """Pins the PAIRED claim in gap_difference's docstring: "a repo that is simply hard
    to predict cancels out." A repo present in only one group must be excluded from the
    difference entirely, not folded into an unpaired group mean.

    r0-r2 carry both groups with a constant paired per-repo gap of 0.3 - 0.1 = 0.2.
    r_lone has ONLY first-time rows (no repeat-contributor rows at all, so it never
    appears in `b`) with a much larger first-time gap of 0.9. Paired arithmetic,
    (a - b).dropna(), drops r_lone (a[r_lone] - b[r_lone] is NaN) and the mean over
    r0-r2 alone is exactly 0.2.

    An unpaired computation (a.mean() - b.mean()) would instead fold r_lone's 0.9 into
    only the first-timer side: a.mean() = (0.3+0.3+0.3+0.9)/4 = 0.45, b.mean() = 0.1,
    giving 0.35 -- a visibly different answer from the paired 0.2, so this test tells
    the two implementations apart."""
    rows = []
    for r in ("r0", "r1", "r2"):
        rows += [
            {"repo": r, "is_first_pr_here": True,  "is_slow": True,  "p_hat": 0.8},
            {"repo": r, "is_first_pr_here": True,  "is_slow": False, "p_hat": 0.8},
            {"repo": r, "is_first_pr_here": False, "is_slow": True,  "p_hat": 0.6},
            {"repo": r, "is_first_pr_here": False, "is_slow": False, "p_hat": 0.6},
        ]
    rows += [
        {"repo": "r_lone", "is_first_pr_here": True, "is_slow": False, "p_hat": 0.9},
        {"repo": "r_lone", "is_first_pr_here": True, "is_slow": False, "p_hat": 0.9},
    ]
    pred = pd.DataFrame(rows)
    d = fairness.gap_difference(pred, seed=1)
    assert d["difference"] == pytest.approx(0.2)

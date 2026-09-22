# Phase 6 — SHAP Attribution, Error Analysis, Fairness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Explain Phase 4's cold-start finding by showing where each model's attribution mass actually sits, inspect 50 confident failures with GitHub links, and close the blueprint's §4 fairness check with confidence intervals — retraining nothing.

**Architecture:** Four small modules over artifacts that already exist on disk. `attribution.py` runs exact TreeSHAP on seeded samples and reduces two scenarios to one comparable shift table; `errors.py` joins predictions back to the feature table to build 50 inspectable failures; `fairness.py` puts cluster-bootstrap intervals on the slice gaps; `report6.py` assembles the document, figures, and a validity gate whose hard stop is SHAP additivity.

**Tech Stack:** Python 3.13, shap 0.52.0 (TreeExplainer), lightgbm 4.7.0, pandas, numpy, scipy (`spearmanr`), matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-22-phase6-interpretation-design.md`

**Three facts the spec did not pin down, established by inspecting the artifacts:**
1. Prediction parquets contain exactly `pr_id, repo, created_at, is_slow, p_hat, baseline_score, is_first_pr_here, created_hour_utc, diff_is_exact` — **no `number`, no `wait_h`, and no feature values.** The spec's §5 error table needs all three. `experiment.load_table()` (38,444 rows) has `number`, `wait_h` and every feature column; `report6.main` joins them on `pr_id` before calling `errors.worst_rows`.
2. SHAP additivity holds against the **raw margin**, not the probability. The gate must compare to `booster.predict(X, raw_score=True)`, never `model.predict`, which applies the sigmoid.
3. The two `feature_frame` arguments in `errors.py` are framed differently on purpose, and mixing them up is silent corruption: `worst_rows` indexes **positionally** (`.iloc[i]`, row-aligned with `pred` and `shap_values`), `error_patterns` indexes **by `pr_id`**. Each function's docstring says which, and `main()` passes `.reset_index(drop=True)` to the first and the `pr_id`-indexed frame to the second.

**One place the implementation is deliberately stronger than the spec:** spec §8 check 1 asks for additivity on "100 seeded rows per explained model". This checks it on every explained row (5,000 per model, plus the full Scenario A error frame) — a superset, same tolerance. Not a deviation to flag in review.

## Global Constraints

- Nothing is retrained. Boosters come from `data/models/{scenario}_{featureset}_fold{k}.txt` via `model.load`; predictions from `data/predictions/{same}.parquet`; both paths are recorded in `data/phase4_runs.json` under each run's `model_path` / `pred_path`.
- Explained models: `A_FULL_fold0` and the five `B_FULL_fold{0..4}`. Scenario B's five SHAP matrices are concatenated row-wise before `importance` is called.
- Feature columns for FULL are `featuresets.FEATURE_SETS["FULL"]` (37), in that order. `language_dominant` must be cast to `category` before `shap_values` — same as `model._prepare` does at predict time.
- Seed everywhere: `20260912` (`splits.SEED`). Sample size 5,000 rows per model; a frame with fewer rows is used whole.
- `share = mean_abs_shap / total_mean_abs_shap` per scenario, so shares sum to 1.0 and scenarios are comparable. `delta = share_b − share_a`.
- `LABEL_REPLAY = ("trailing_90d_slow_rate", "trailing_n", "author_prior_slow_rate_here", "author_prior_n")` — import from `featuresets`, never retype.
- Every confidence interval uses `metrics.cluster_bootstrap(per_repo, seed=SEED)`, resampling repos.
- Gate #1 (SHAP additivity, raw margin, |Δ| < 1e-6) is a HARD STOP.
- Tests in `tests/`, `python -m pytest tests -q`, output pristine. 115 tests pass today.
- Commits on branch `phase6`; every commit message ends with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Real-data steps (Task 5) run only after Tasks 1–4 are reviewed.

---

## File structure

| File | Responsibility |
|---|---|
| `requirements.txt` (modify) | pin `shap` |
| `attribution.py` (create) | `sample_rows`, `explain`, `importance`, `shift`, `label_replay_share`, `additivity_delta`, `explain_run` |
| `errors.py` (create) | `worst_rows`, `error_patterns` |
| `fairness.py` (create) | `slice_gaps`, `gap_difference` |
| `report6.py` (create) | `gate_checks`, figures, `render`, `main` |
| `tests/test_attribution.py`, `tests/test_errors.py`, `tests/test_fairness.py`, `tests/test_report6.py` (create) | |
| outputs: `data/phase6_{shift,importance_A,importance_B,worst50,fairness}.csv`, `data/phase6_gate.json`, `docs/phase6_interpretation.md`, `figures/phase6_*.png` | |

---

### Task 1: `attribution.py` — TreeSHAP, importance, the shift table

**Files:**
- Modify: `requirements.txt`
- Create: `attribution.py`, `tests/test_attribution.py`

**Interfaces:**
- Consumes: `featuresets.FEATURE_SETS`, `featuresets.LABEL_REPLAY`, `model.load`, `model.CATEGORICAL`, `splits.SEED`.
- Produces:
  - `attribution.SEED`, `attribution.SAMPLE_N = 5000`, `attribution.RUNS_JSON: Path`
  - `attribution.sample_rows(df: pd.DataFrame, n: int = SAMPLE_N, seed: int = SEED) -> pd.DataFrame`
  - `attribution.prepare(X: pd.DataFrame) -> pd.DataFrame` (categorical cast)
  - `attribution.explain(booster, X) -> tuple[np.ndarray, float]` — `(shap_values, expected_value)`
  - `attribution.additivity_delta(booster, X, shap_values, expected_value) -> float` — max \|Δ\| vs raw margin
  - `attribution.importance(shap_values: np.ndarray, cols: list[str]) -> pd.DataFrame` — `feature, mean_abs_shap, share`
  - `attribution.shift(imp_a, imp_b) -> pd.DataFrame` — `feature, share_a, share_b, delta`
  - `attribution.label_replay_share(imp: pd.DataFrame) -> float`
  - `attribution.explain_run(scenario: str, table: pd.DataFrame, runs: list[dict], n=SAMPLE_N, seed=SEED) -> tuple[np.ndarray, pd.DataFrame, list[float]]` — pooled `(shap_values, X_used, additivity_deltas)` over that scenario's FULL folds

- [ ] **Step 1: Pin the dependency**

Append to `requirements.txt`:
```
# Phase 6
shap==0.52.0            # exact TreeSHAP attribution for the LightGBM boosters
```

- [ ] **Step 2: Write the failing tests**

`tests/test_attribution.py`:
```python
import numpy as np
import pandas as pd
import pytest

import attribution as attr
import featuresets as fs
import model


@pytest.fixture
def toy_booster():
    """2 features: `a` drives the label, `b` is pure noise. A correct importance
    ranking must put `a` first by a wide margin."""
    rng = np.random.default_rng(0)
    n = 400
    a = rng.normal(size=n)
    X = pd.DataFrame({"a": a, "b": rng.normal(size=n)})
    y = (a + rng.normal(0, 0.25, n) > 0).astype(int)
    return model.fit(X, y, model.DEFAULT_PARAMS), X, y


def test_sample_rows_is_seeded_and_never_oversamples():
    df = pd.DataFrame({"x": range(100)})
    assert attr.sample_rows(df, n=10, seed=1).equals(attr.sample_rows(df, n=10, seed=1))
    assert not attr.sample_rows(df, n=10, seed=1).equals(attr.sample_rows(df, n=10, seed=2))
    assert len(attr.sample_rows(df, n=500, seed=1)) == 100        # fewer rows than asked
    assert len(attr.sample_rows(df, n=10, seed=1)) == 10


def test_explain_shapes_and_additivity(toy_booster):
    booster, X, _ = toy_booster
    sv, ev = attr.explain(booster, X)
    assert sv.shape == (len(X), X.shape[1])
    assert isinstance(ev, float)
    # Additivity is against the RAW MARGIN, not the probability.
    assert attr.additivity_delta(booster, X, sv, ev) < 1e-6


def test_importance_ranks_the_real_driver_first(toy_booster):
    booster, X, _ = toy_booster
    sv, _ = attr.explain(booster, X)
    imp = attr.importance(sv, list(X.columns))
    assert list(imp.columns) == ["feature", "mean_abs_shap", "share"]
    assert imp.iloc[0]["feature"] == "a"
    assert imp.iloc[0]["share"] > imp.iloc[1]["share"] * 2      # a dominates b
    assert imp["share"].sum() == pytest.approx(1.0)


def test_shift_signs_and_ordering():
    imp_a = pd.DataFrame({"feature": ["x", "y", "z"], "mean_abs_shap": [3.0, 1.0, 1.0],
                          "share": [0.6, 0.2, 0.2]})
    imp_b = pd.DataFrame({"feature": ["x", "y", "z"], "mean_abs_shap": [1.0, 3.0, 1.0],
                          "share": [0.2, 0.6, 0.2]})
    s = attr.shift(imp_a, imp_b)
    assert list(s.columns) == ["feature", "share_a", "share_b", "delta"]
    assert s.iloc[0]["feature"] in ("x", "y")                    # largest |delta| first
    assert s.set_index("feature").loc["x", "delta"] == pytest.approx(-0.4)
    assert s.set_index("feature").loc["y", "delta"] == pytest.approx(+0.4)
    assert s.set_index("feature").loc["z", "delta"] == pytest.approx(0.0)


def test_shift_handles_a_feature_missing_from_one_side():
    imp_a = pd.DataFrame({"feature": ["x"], "mean_abs_shap": [1.0], "share": [1.0]})
    imp_b = pd.DataFrame({"feature": ["y"], "mean_abs_shap": [1.0], "share": [1.0]})
    s = attr.shift(imp_a, imp_b).set_index("feature")
    assert s.loc["x", "share_b"] == 0.0 and s.loc["x", "delta"] == pytest.approx(-1.0)
    assert s.loc["y", "share_a"] == 0.0 and s.loc["y", "delta"] == pytest.approx(+1.0)


def test_label_replay_share_sums_exactly_the_four_columns():
    imp = pd.DataFrame({
        "feature": list(fs.LABEL_REPLAY) + ["body_len"],
        "mean_abs_shap": [1.0] * 5,
        "share": [0.1, 0.1, 0.05, 0.05, 0.7],
    })
    assert attr.label_replay_share(imp) == pytest.approx(0.30)
    none_present = pd.DataFrame({"feature": ["body_len"], "mean_abs_shap": [1.0], "share": [1.0]})
    assert attr.label_replay_share(none_present) == 0.0
```

- [ ] **Step 3: Run to verify they fail**

Run: `python -m pytest tests/test_attribution.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'attribution'`

- [ ] **Step 4: Implement `attribution.py`**

```python
"""Exact TreeSHAP attribution over the Phase 4 boosters. Nothing is retrained.

Phase 4 established THAT cold-start transfer fails: on Scenario B, removing the four
label-replay features drops the model below the trailing-rate baseline, while on A the
same ablation still wins. This module establishes WHY, by measuring where each model's
attribution mass sits.

Two design points worth stating, because both are easy to get wrong:

  * `share` normalises each scenario's mean-|SHAP| to sum to 1. Raw SHAP magnitudes are
    not comparable across models (different base rates, different margins); shares are.
  * Additivity is a property of the RAW MARGIN, not the probability. sum(shap) +
    expected_value reproduces `booster.predict(X, raw_score=True)`. Comparing against the
    sigmoid output would fail for correct attributions -- see additivity_delta."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap

import featuresets as fs
import model
import splits

SEED = splits.SEED
SAMPLE_N = 5000
ROOT = Path(__file__).parent
RUNS_JSON = ROOT / "data" / "phase4_runs.json"


def sample_rows(df: pd.DataFrame, n: int = SAMPLE_N, seed: int = SEED) -> pd.DataFrame:
    """Seeded sample; a frame smaller than n is returned whole."""
    return df if len(df) <= n else df.sample(n=n, random_state=seed)


def prepare(X: pd.DataFrame) -> pd.DataFrame:
    """Same categorical cast model._prepare applies at predict time."""
    X = X.copy()
    for c in model.CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def explain(booster, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Exact TreeSHAP. Returns (values [n_rows, n_features], expected_value).

    shap 0.52.0 warns on every call that the LightGBM-binary output form "has changed to a
    list of ndarray", then returns a plain (n, n_features) array -- verified. The warning is
    silenced by message (never a blanket ignore), and the shape is ASSERTED, because a
    version that actually returns the list form would otherwise sail through np.asarray as
    a (2, n, f) array and silently corrupt every number in this phase."""
    Xp = prepare(X)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*output has changed to a list of ndarray.*",
                                category=UserWarning)
        ex = shap.TreeExplainer(booster)
        raw = ex.shap_values(Xp)
    if isinstance(raw, list):                       # defensive: future shap, binary -> [neg, pos]
        raw = raw[1]
    sv = np.asarray(raw)
    if sv.shape != (len(Xp), Xp.shape[1]):
        raise RuntimeError(f"unexpected SHAP shape {sv.shape}, expected {(len(Xp), Xp.shape[1])}")
    return sv, float(np.asarray(ex.expected_value).ravel()[-1])


def additivity_delta(booster, X: pd.DataFrame, shap_values: np.ndarray,
                     expected_value: float) -> float:
    """max |sum(shap) + expected_value - raw_margin|. The gate's hard stop."""
    margin = np.asarray(booster.predict(prepare(X), raw_score=True))
    return float(np.max(np.abs(shap_values.sum(axis=1) + expected_value - margin)))


def importance(shap_values: np.ndarray, cols: list[str]) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    total = mean_abs.sum()
    return (pd.DataFrame({"feature": cols, "mean_abs_shap": mean_abs,
                          "share": mean_abs / total if total else 0.0})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def shift(imp_a: pd.DataFrame, imp_b: pd.DataFrame) -> pd.DataFrame:
    """share_b - share_a per feature. The phase's headline artifact."""
    a = imp_a.set_index("feature")["share"].rename("share_a")
    b = imp_b.set_index("feature")["share"].rename("share_b")
    out = pd.concat([a, b], axis=1).fillna(0.0).reset_index()
    out["delta"] = out["share_b"] - out["share_a"]
    return (out.reindex(out["delta"].abs().sort_values(ascending=False).index)
            .reset_index(drop=True))


def label_replay_share(imp: pd.DataFrame) -> float:
    """Attribution mass on the baseline's own signal: the four label-replay features."""
    return float(imp.loc[imp["feature"].isin(fs.LABEL_REPLAY), "share"].sum())


def explain_run(scenario: str, table: pd.DataFrame, runs: list[dict],
                n: int = SAMPLE_N, seed: int = SEED) -> tuple[np.ndarray, pd.DataFrame, list[float]]:
    """Pool a scenario's FULL folds: one SHAP matrix, one X, one delta per fold.

    Scenario B has five boosters over the same 37 columns. Pooling row-wise is what makes
    'what does the B model lean on' a single answer."""
    cols = fs.FEATURE_SETS["FULL"]
    by_id = table.set_index("pr_id")
    mats, frames, deltas = [], [], []
    for r in sorted((r for r in runs if r["scenario"] == scenario and r["featureset"] == "FULL"),
                    key=lambda r: r["fold"]):
        pred = pd.read_parquet(r["pred_path"])
        X = prepare(sample_rows(by_id.loc[pred["pr_id"].to_numpy(), cols], n=n, seed=seed))
        booster = model.load(Path(r["model_path"]))
        sv, ev = explain(booster, X)
        deltas.append(additivity_delta(booster, X, sv, ev))
        mats.append(sv)
        frames.append(X)
    # The returned frame is the PREPARED one -- exactly the rows that were explained, so the
    # beeswarms plot the same thing the SHAP values describe. Re-prepared after the concat:
    # concatenating per-fold categoricals with different category sets yields object dtype.
    return np.vstack(mats), prepare(pd.concat(frames, ignore_index=True)), deltas


def load_runs(path: Path = RUNS_JSON) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["runs"]
```

- [ ] **Step 5: Run to verify they pass; full suite; commit**

Run: `python -m pytest tests/test_attribution.py -q` → `6 passed`; `python -m pytest tests -q` → all pass.

Output must be pristine. The one warning this is known to produce is shap's `UserWarning: LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray`, already silenced by message inside `explain()`. If a *different* warning appears, silence it with a filter scoped to that exact message or category and name it in the report — never a blanket `ignore`, and never a filter that would also hide the one above.

Verified before this plan was written, so these are expectations rather than hopes: on the toy fixture the booster builds 200 trees, additivity comes out at 2.3e-14, and `a`'s share is 10.8× `b`'s.
```bash
git add requirements.txt attribution.py tests/test_attribution.py
git commit -m "feat(phase6): exact TreeSHAP attribution, importance shares, A-vs-B shift table"
```

---

### Task 2: `errors.py` — the 50 worst predictions

**Files:**
- Create: `errors.py`, `tests/test_errors.py`

**Interfaces:**
- Consumes: nothing from this project. `errors.py` imports only `numpy` and `pandas`; it receives the SHAP matrix and column list as arguments rather than importing `attribution`, which keeps it testable on hand-built fixtures with no booster.
- Produces:
  - `errors.GITHUB = "https://github.com/{repo}/pull/{number}"`
  - `errors.worst_rows(pred: pd.DataFrame, shap_values: np.ndarray, cols: list[str], feature_frame: pd.DataFrame, n: int = 25) -> pd.DataFrame`
  - `errors.error_patterns(worst: pd.DataFrame, feature_frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame` — `feature, mean_worst, mean_all, z`

Two caller contracts, both enforced by `report6.main`, neither checkable inside these functions:
- `pred`, `shap_values` and `feature_frame` are **row-aligned** for `worst_rows`: `shap_values[i]` explains `pred.iloc[i]`, whose feature values are `feature_frame.iloc[i]`.
- `pred` carries `number` and `wait_h`, joined from `experiment.load_table()` on `pr_id` — the prediction parquets have neither.
- `error_patterns`'s `feature_frame` is indexed **by `pr_id`**, not positionally.

- [ ] **Step 1: Write the failing tests**

`tests/test_errors.py`:
```python
import numpy as np
import pandas as pd
import pytest

import errors


@pytest.fixture
def aligned():
    """6 rows. The planted extremes: row 0 is the worst false positive (not slow, p_hat
    0.99), row 5 the worst false negative (slow, p_hat 0.01)."""
    pred = pd.DataFrame({
        "pr_id": [f"p{i}" for i in range(6)],
        "repo": ["o/r"] * 3 + ["o/s"] * 3,
        "number": [10, 11, 12, 13, 14, 15],
        "is_slow": [False, False, True, True, False, True],
        "p_hat": [0.99, 0.60, 0.55, 0.40, 0.20, 0.01],
        "wait_h": [2.0, 5.0, np.nan, np.nan, 9.0, np.nan],
    })
    cols = ["f0", "f1", "f2"]
    feature_frame = pd.DataFrame({"f0": [5.0, 1, 1, 1, 1, 1],
                                  "f1": [0.0, 0, 0, 0, 0, 0],
                                  "f2": [1.0, 1, 1, 1, 1, 1]})
    # row 0: f0 dominates (|2.0|), then f2 (|0.5|), then f1 (|0.1|)
    shap_values = np.tile([0.1, 0.5, 0.05], (6, 1))
    shap_values[0] = [0.1, -2.0, 0.5]        # order by |value|: f1, f2, f0
    return pred, shap_values, cols, feature_frame


def test_worst_rows_picks_the_planted_extremes(aligned):
    pred, sv, cols, ff = aligned
    w = errors.worst_rows(pred, sv, cols, ff, n=1)
    assert len(w) == 2
    fp = w[w["kind"] == "fp"].iloc[0]
    fn = w[w["kind"] == "fn"].iloc[0]
    assert fp["pr_id"] == "p0" and fp["is_slow"] == False    # noqa: E712
    assert fn["pr_id"] == "p5" and fn["is_slow"] == True     # noqa: E712
    assert fp["p_hat"] == 0.99 and fn["p_hat"] == 0.01


def test_worst_rows_builds_the_github_url(aligned):
    pred, sv, cols, ff = aligned
    w = errors.worst_rows(pred, sv, cols, ff, n=1).set_index("pr_id")
    assert w.loc["p0", "url"] == "https://github.com/o/r/pull/10"
    assert w.loc["p5", "url"] == "https://github.com/o/s/pull/15"


def test_worst_rows_orders_contributions_by_absolute_shap(aligned):
    pred, sv, cols, ff = aligned
    w = errors.worst_rows(pred, sv, cols, ff, n=1).set_index("pr_id")
    r = w.loc["p0"]
    assert r["top1_feature"] == "f1" and r["top1_shap"] == pytest.approx(-2.0)
    assert r["top2_feature"] == "f2" and r["top2_shap"] == pytest.approx(0.5)
    assert r["top3_feature"] == "f0" and r["top3_shap"] == pytest.approx(0.1)
    assert r["top1_value"] == 0.0 and r["top3_value"] == 5.0      # the row's own values


def test_worst_rows_caps_at_available_rows(aligned):
    pred, sv, cols, ff = aligned
    w = errors.worst_rows(pred, sv, cols, ff, n=25)     # only 3 fp and 3 fn exist
    assert (w["kind"] == "fp").sum() == 3 and (w["kind"] == "fn").sum() == 3


def test_error_patterns_flags_the_inflated_feature():
    """f0 is 0 everywhere except the two worst rows, where it is 50."""
    ff_all = pd.DataFrame({"f0": [0.0] * 98 + [50.0, 50.0], "f1": np.arange(100.0)},
                          index=[f"p{i}" for i in range(100)])
    worst = pd.DataFrame({"pr_id": ["p98", "p99"]})
    out = errors.error_patterns(worst, ff_all, ["f0", "f1"])
    assert list(out.columns) == ["feature", "mean_worst", "mean_all", "z"]
    assert out.iloc[0]["feature"] == "f0"          # largest |z| first
    assert out.iloc[0]["z"] > 3


def test_error_patterns_survives_a_non_numeric_column():
    """language_dominant is categorical; it must not raise, and must not rank."""
    ff_all = pd.DataFrame({"f0": [0.0] * 98 + [50.0, 50.0], "lang": ["py"] * 100},
                          index=[f"p{i}" for i in range(100)])
    out = errors.error_patterns(pd.DataFrame({"pr_id": ["p98", "p99"]}), ff_all, ["f0", "lang"])
    assert out.iloc[0]["feature"] == "f0"
    assert out.set_index("feature").loc["lang", "z"] == 0.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_errors.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'errors'`

- [ ] **Step 3: Implement `errors.py`**

```python
"""The 50 predictions worth reading by hand.

Blueprint §3, Phase 6: "manual error analysis on worst 50 predictions". Both tails matter
for a triage tool -- a confident false positive wastes a lead's attention, a confident
false negative is a PR that silently rots -- so this takes 25 of each rather than the 50
largest absolute errors, which would be dominated by whichever class is larger.

error_patterns() is a first pass, not the analysis: it says which features look unusual
among the failures. Reading the rows is still the job."""
from __future__ import annotations

import numpy as np
import pandas as pd

GITHUB = "https://github.com/{repo}/pull/{number}"
TOP_K = 5


def worst_rows(pred: pd.DataFrame, shap_values: np.ndarray, cols: list[str],
               feature_frame: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """25 most-confident false positives + 25 most-confident false negatives.

    pred, shap_values and feature_frame are POSITIONALLY aligned: shap_values[i] explains
    pred.iloc[i], whose feature values are feature_frame.iloc[i]. (error_patterns below
    takes a pr_id-INDEXED frame instead -- the two are not interchangeable.)

    pred must carry `number` and `wait_h`; the prediction parquets have neither, so the
    caller joins them from experiment.load_table()."""
    pos = np.arange(len(pred))
    is_slow = pred["is_slow"].to_numpy(dtype=bool)
    p_hat = pred["p_hat"].to_numpy(dtype=float)

    fp = pos[~is_slow][np.argsort(-p_hat[~is_slow])][:n]        # not slow, most confident
    fn = pos[is_slow][np.argsort(p_hat[is_slow])][:n]           # slow, least confident

    rows = []
    for kind, idx in (("fp", fp), ("fn", fn)):
        for i in idx:
            src = pred.iloc[i]
            sv = shap_values[i]
            order = np.argsort(-np.abs(sv))[:TOP_K]
            row = {
                "kind": kind, "pr_id": src["pr_id"], "repo": src["repo"],
                "number": int(src["number"]),
                "url": GITHUB.format(repo=src["repo"], number=int(src["number"])),
                "p_hat": float(src["p_hat"]), "is_slow": bool(src["is_slow"]),
                "wait_h": float(src["wait_h"]) if pd.notna(src["wait_h"]) else float("nan"),
            }
            vals = feature_frame.iloc[i]
            for rank, j in enumerate(order, start=1):
                row[f"top{rank}_feature"] = cols[j]
                row[f"top{rank}_shap"] = float(sv[j])
                row[f"top{rank}_value"] = vals[cols[j]]   # left raw: language_dominant is categorical
            rows.append(row)
    return pd.DataFrame(rows)


def error_patterns(worst: pd.DataFrame, feature_frame: pd.DataFrame,
                   cols: list[str]) -> pd.DataFrame:
    """How the 50 worst rows differ from the test set, per feature, as a z-score.

    feature_frame is indexed BY pr_id here (unlike worst_rows, which is positional).
    Non-numeric columns coerce to NaN and fall out of the ranking, which is intended --
    a z-score of `language_dominant` would mean nothing."""
    sub = feature_frame.loc[feature_frame.index.isin(worst["pr_id"])]
    out = []
    for c in cols:
        col = pd.to_numeric(feature_frame[c], errors="coerce")
        w = pd.to_numeric(sub[c], errors="coerce")
        sd = col.std()
        usable = np.isfinite(sd) and sd > 0          # note: `if sd` is True for NaN
        out.append({"feature": c, "mean_worst": float(w.mean()), "mean_all": float(col.mean()),
                    "z": float((w.mean() - col.mean()) / sd) if usable else 0.0})
    df = pd.DataFrame(out)
    return df.reindex(df["z"].abs().sort_values(ascending=False).index).reset_index(drop=True)
```

- [ ] **Step 4: Run to verify they pass; full suite; commit**

Run: `python -m pytest tests/test_errors.py -q` → `6 passed`; `python -m pytest tests -q` → all pass.
```bash
git add errors.py tests/test_errors.py
git commit -m "feat(phase6): 50 worst predictions with SHAP breakdowns and GitHub links"
```

---

### Task 3: `fairness.py` — slice gaps with intervals

**Files:**
- Create: `fairness.py`, `tests/test_fairness.py`

**Interfaces:**
- Consumes: `metrics.cluster_bootstrap`, `splits.SEED`.
- Produces:
  - `fairness.HOUR_BUCKETS = [(0, 6, "00-06"), (6, 12, "06-12"), (12, 18, "12-18"), (18, 24, "18-24")]`
  - `fairness.slice_gaps(pred: pd.DataFrame, seed: int = SEED) -> pd.DataFrame` — `slice, level, n, n_repos, actual_rate, mean_p_hat, gap, gap_ci_lo, gap_ci_hi`
  - `fairness.gap_difference(pred: pd.DataFrame, seed: int = SEED) -> dict` — `difference, ci_lo, ci_hi, ci_excludes_zero, n_first_time, n_repeat`

Per-repo gaps are the bootstrap unit: for each repo in a slice, `mean(p_hat) − mean(is_slow)`; `cluster_bootstrap` then resamples those repo-level values.

- [ ] **Step 1: Write the failing tests**

`tests/test_fairness.py`:
```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_fairness.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'fairness'`

- [ ] **Step 3: Implement `fairness.py`**

```python
"""Blueprint §4's fairness check, finished.

Phase 4 measured that the model is more pessimistic about first-time contributors than
their outcomes warrant (+0.088 on A) but put no interval on it, so it could not be claimed
the gap is real. The quantity that actually bears on fairness is the DIFFERENCE between
the newcomer gap and the repeat-contributor gap: a model miscalibrated equally for
everyone is not a fairness problem, one that singles out newcomers is.

Intervals resample REPOS, matching every other interval in this project: the effective
sample size for a claim about review culture is the number of repos, not of PRs."""
from __future__ import annotations

import numpy as np
import pandas as pd

import metrics
import splits

SEED = splits.SEED
HOUR_BUCKETS = [(0, 6, "00-06"), (6, 12, "06-12"), (12, 18, "12-18"), (18, 24, "18-24")]


def _per_repo_gap(sub: pd.DataFrame) -> pd.Series:
    """mean(p_hat) - mean(is_slow) per repo. The bootstrap unit."""
    g = sub.groupby("repo")
    return g["p_hat"].mean() - g["is_slow"].astype(float).mean()


def _row(slice_name: str, level: str, sub: pd.DataFrame, seed: int) -> dict:
    per_repo = _per_repo_gap(sub)
    lo, hi = metrics.cluster_bootstrap(per_repo, seed=seed) if len(per_repo) else (np.nan, np.nan)
    return {
        "slice": slice_name, "level": level, "n": int(len(sub)),
        "n_repos": int(sub["repo"].nunique()),
        "actual_rate": float(sub["is_slow"].astype(float).mean()),
        "mean_p_hat": float(sub["p_hat"].mean()),
        "gap": float(per_repo.mean()), "gap_ci_lo": float(lo), "gap_ci_hi": float(hi),
    }


def slice_gaps(pred: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    first = pred["is_first_pr_here"].astype(bool)
    rows = [_row("is_first_pr_here", "first-time", pred[first], seed),
            _row("is_first_pr_here", "repeat", pred[~first], seed)]
    hour = pred["created_hour_utc"].astype(int)
    for lo, hi, name in HOUR_BUCKETS:
        rows.append(_row("hour_bucket", name, pred[(hour >= lo) & (hour < hi)], seed))
    return pd.DataFrame(rows)


def gap_difference(pred: pd.DataFrame, seed: int = SEED) -> dict:
    """(first-timer gap) - (repeat gap), with a cluster-bootstrap CI over repos.

    Paired by repo, so a repo that is simply hard to predict cancels out."""
    first = pred["is_first_pr_here"].astype(bool)
    a = _per_repo_gap(pred[first])
    b = _per_repo_gap(pred[~first])
    paired = (a - b).dropna()
    lo, hi = metrics.cluster_bootstrap(paired, seed=seed) if len(paired) else (np.nan, np.nan)
    return {
        "difference": float(paired.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
        "ci_excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
        "n_first_time": int(first.sum()), "n_repeat": int((~first).sum()),
    }
```

- [ ] **Step 4: Run to verify they pass; full suite; commit**

Run: `python -m pytest tests/test_fairness.py -q` → `4 passed`; `python -m pytest tests -q` → all pass.
```bash
git add fairness.py tests/test_fairness.py
git commit -m "feat(phase6): fairness slice gaps and the newcomer gap difference, with intervals"
```

---

### Task 4: `report6.py` — gate, figures, document

**Files:**
- Create: `report6.py`, `tests/test_report6.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3, plus `experiment.load_table`, `featuresets.FEATURE_SETS`, `scipy.stats.spearmanr`.
- Produces:
  - `report6.gate_checks(additivity: dict[str, list[float]], stability: float, worst: pd.DataFrame, pred_a: pd.DataFrame, fair: pd.DataFrame, artifacts: dict[str, int]) -> list[dict]` — five `{id, check, value, pass}`
  - `report6.fig_shift`, `report6.fig_beeswarm`, `report6.fig_dependence`
  - `report6.md(df)`, `report6.render(...) -> str`, `report6.main()`

- [ ] **Step 1: Write the failing tests**

`tests/test_report6.py`:
```python
import numpy as np
import pandas as pd
import pytest

import report6


def _ok_inputs():
    additivity = {"A": [1e-9, 2e-9], "B": [3e-10]}
    stability = 0.97
    worst = pd.DataFrame({"kind": ["fp"] * 25 + ["fn"] * 25,
                          "pr_id": [f"p{i}" for i in range(50)],
                          "is_slow": [False] * 25 + [True] * 25})
    pred_a = pd.DataFrame({"pr_id": [f"p{i}" for i in range(100)],
                           "is_slow": [False] * 25 + [True] * 25 + [False] * 50})
    fair = pd.DataFrame({"gap_ci_lo": [0.1] * 6, "gap_ci_hi": [0.3] * 6, "n_repos": [8] * 6})
    artifacts = {"shift": 37, "importance_A": 37, "importance_B": 37, "worst50": 50, "fairness": 6}
    return additivity, stability, worst, pred_a, fair, artifacts


def test_gate_all_pass():
    checks = report6.gate_checks(*_ok_inputs())
    assert [c["id"] for c in checks] == [1, 2, 3, 4, 5]
    assert all(c["pass"] for c in checks), [c for c in checks if not c["pass"]]


def test_gate1_fails_on_broken_additivity():
    a, s, w, p, f, art = _ok_inputs()
    a["B"] = [1e-3]
    c = report6.gate_checks(a, s, w, p, f, art)
    assert c[0]["pass"] is False and "additivity" in c[0]["check"].lower()


def test_gate2_fails_on_unstable_ranking():
    a, s, w, p, f, art = _ok_inputs()
    c = report6.gate_checks(a, 0.4, w, p, f, art)
    assert c[1]["pass"] is False


def test_gate3_fails_on_wrong_split_or_foreign_rows():
    a, s, w, p, f, art = _ok_inputs()
    lopsided = w.copy(); lopsided.loc[0, "kind"] = "fn"          # 24/26
    assert report6.gate_checks(a, s, lopsided, p, f, art)[2]["pass"] is False
    foreign = w.copy(); foreign.loc[0, "pr_id"] = "NOT_IN_PRED"
    assert report6.gate_checks(a, s, foreign, p, f, art)[2]["pass"] is False
    mismatched = w.copy(); mismatched.loc[0, "is_slow"] = True   # p0 is False in pred_a
    assert report6.gate_checks(a, s, mismatched, p, f, art)[2]["pass"] is False


def test_gate4_fails_on_non_finite_ci_or_single_repo():
    a, s, w, p, f, art = _ok_inputs()
    bad_ci = f.copy(); bad_ci.loc[0, "gap_ci_hi"] = np.nan
    assert report6.gate_checks(a, s, w, p, bad_ci, art)[3]["pass"] is False
    one_repo = f.copy(); one_repo.loc[0, "n_repos"] = 1
    assert report6.gate_checks(a, s, w, p, one_repo, art)[3]["pass"] is False


def test_gate5_fails_on_empty_artifact():
    a, s, w, p, f, art = _ok_inputs()
    art["shift"] = 0
    assert report6.gate_checks(a, s, w, p, f, art)[4]["pass"] is False


def test_md_renders_a_table():
    out = report6.md(pd.DataFrame({"a": [1.0], "b": ["x"]}))
    assert out.splitlines()[0] == "| a | b |"
    assert "1.000" in out and "x" in out
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_report6.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'report6'`

- [ ] **Step 3: Implement `report6.py`**

```python
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
```

Then `render()` and `main()` in the same file:

```python
def render(shift_df, imp_a, imp_b, share_a, share_b, worst, patterns, fair, diffs,
           checks, figs, n_explained) -> str:
    verdict = "PASS" if all(c["pass"] for c in checks) else "FAIL"
    fp = worst[worst["kind"] == "fp"].head(25)
    fn = worst[worst["kind"] == "fn"].head(25)
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

{chr(10).join(md(f) + chr(10) for f in fair.values())}

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
    sv2, X2, _ = attr.explain_run("A", table, runs, seed=attr.SEED + 1)
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
```

- [ ] **Step 4: Run to verify they pass; full suite; commit**

Run: `python -m pytest tests/test_report6.py -q` → `7 passed`; `python -m pytest tests -q` → all pass, no warnings.
```bash
git add report6.py tests/test_report6.py
git commit -m "feat(phase6): interpretation report, figures, validity gate"
```

---

### Task 5: Real-data run (CONDITIONAL on Tasks 1–4 reviewed)

**Files:** none created in git beyond outputs. Produces `docs/phase6_interpretation.md`, `figures/phase6_*.png`, `data/phase6_*.csv`, `data/phase6_gate.json`.

- [ ] **Step 1: Run it**

Run: `time python report6.py; echo exit=$?`
Expected: exit 0, gate 5/5 PASS. Report wall-clock.

**If gate #1 (additivity) fails, STOP.** Report the per-scenario max delta and do not quote any attribution number — the explainer is wrong, not the model.

- [ ] **Step 2: Report the numbers verbatim**

In the task report, quote: (a) the full gate table from `data/phase6_gate.json`; (b) §1's shift table (top 20); (c) §2's mechanism paragraph and the two label-replay shares; (d) §3's two top-15 importance tables; (e) §4's `error_patterns` top 12; (f) §5's fairness tables and both verdict paragraphs.

- [ ] **Step 3: Read the 50 rows and write the analysis**

Open `data/phase6_worst50.csv`. Using the SHAP breakdowns and the `error_patterns` z-scores, write 3–6 sentences describing what the false positives share, what the false negatives share, and whether any repo or feature dominates either tail. Do not speculate beyond what the columns show; if a pattern is weak, say so.

`render()` emits this anchor line in §4:
```
<!-- WRITTEN-ANALYSIS: replace this line with the hand-written reading of data/phase6_worst50.csv -->
```
Replace that exact line in `docs/phase6_interpretation.md` with the paragraph. Edit the file directly — this prose is written by a reader, not generated, and re-running `report6.py` will restore the anchor and discard it. That is the intended trade: the generated document is reproducible, the analysis is appended once at the end.

- [ ] **Step 4: Commit**

```bash
git add docs/phase6_interpretation.md data/phase6_shift.csv data/phase6_importance_A.csv \
        data/phase6_importance_B.csv data/phase6_worst50.csv data/phase6_fairness.csv \
        data/phase6_gate.json figures/phase6_*.png
git commit -m "data: Phase 6 attribution, error analysis, and fairness results"
```

---

## Verification (spec §8–§9)

1. `python -m pytest tests -q` — toy booster additivity, importance ranking the planted driver, shift signs, planted fp/fn extremes, planted fairness gap recovered, each gate failure mode caught by exactly its own check.
2. Real data: `data/phase6_gate.json` 5/5 PASS, additivity delta < 1e-6 on every explained model.
3. `docs/phase6_interpretation.md` §2 states the mechanism in whichever direction the data went; §5 states the fairness verdict in whichever direction it went.
4. The 50-row analysis in §4 is written by a human reading `data/phase6_worst50.csv`, not generated.

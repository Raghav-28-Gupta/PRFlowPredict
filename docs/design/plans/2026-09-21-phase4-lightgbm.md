# Phase 4 — LightGBM Classifier — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate a LightGBM `is_slow` classifier on both split scenarios, with four pre-registered feature sets, against the trailing-90-day baseline on identical rows — producing saved models, per-row predictions, a results document that states the blueprint's headline comparison either way, and a validity gate.

**Architecture:** Five small modules around one `experiment.run(...)` core. `featuresets.py` derives the four column lists from `features.COLUMN_SPEC` (so nothing forbidden can leak in); `model.py` wraps LightGBM's native `lgb.train` with all seeds and the categorical in one place; `tune.py` is the Optuna loop on Scenario A's training rows; `experiment.py` runs one (scenario, fold, set), saves model + predictions, logs; `report4.py` aggregates into `docs/phase4_results.md` and applies the gate. Every module is proven on a synthetic table before real data.

**Tech Stack:** Python 3.13, pandas, numpy, lightgbm 4.7.0 (native API), optuna 5.0.0, scikit-learn (`TimeSeriesSplit`, `precision_recall_curve`), matplotlib, pytest.

**Spec:** `docs/design/specs/2026-09-21-phase4-lightgbm-design.md`

**Deviation from spec, recorded here:** §6 names `LGBMClassifier`; this plan uses LightGBM's native `lgb.train`/`lgb.Booster` — same algorithm, but native save/load and unambiguous parameter names. The spec's tuned-parameter names are kept as the public vocabulary (`n_estimators`, `min_child_samples`) and translated inside `model.fit` (`n_estimators` → `num_boost_round`, `min_child_samples` → `min_data_in_leaf`).

## Global Constraints

- Rows: `data/features/features.parquet` minus rows where `timeline_may_be_truncated` is True, sorted by `created_at, pr_id`, index reset. `language_dominant` as pandas `category`.
- Splits: `splits.scenario_a(table)` (cutoff `2026-01-01T00:00:00Z`, 5%/repo training cap, seed 20260912) and `splits.scenario_b(table)` (`GroupKFold(5)`), used unchanged.
- Feature sets are derived from `features.COLUMN_SPEC`, never hand-listed: FULL 37, NO_SNAPSHOT 29, NO_LABEL_REPLAY 33, PR_ONLY 17. No column with status `key`, `label`, or `flag` may appear in any set.
- Model fixed params: `objective=binary, seed=20260912, deterministic=True, force_row_wise=True, num_threads=1, verbose=-1, bagging_seed=20260912, feature_fraction_seed=20260912`; `scale_pos_weight = n_neg / n_pos` from the training `y`. No early stopping on test.
- Tuning: Optuna TPE `seed=20260912`, 100 trials, objective = mean AUC-PR over `TimeSeriesSplit(n_splits=3)` on Scenario A **training** rows sorted by time, FULL set only. Params frozen for every scenario, fold and ablation.
- Baseline score for any row = the table's `trailing_90d_slow_rate`.
- Every run logged via `tracking.log` with `model="lgbm"`, `features=<set name>`, `params=<sha256 of params.json>[:12]`.
- Seed everywhere: 20260912. Tests in `tests/`, `python -m pytest tests -q`, output pristine.
- Commits on branch `phase4`; commit messages end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Real-data steps (Task 6) run only after Tasks 1–5 are reviewed and only if `data/phase3_gate.json` shows all five checks passing.

---

## File structure

| File | Responsibility |
|---|---|
| `requirements.txt` (modify) | pin `lightgbm`, `optuna` |
| `featuresets.py` (create) | `FEATURE_SETS`, `assert_hygiene` |
| `model.py` (create) | `fit`, `predict`, `save`, `load`, `FIXED`, `TUNED_KEYS` |
| `experiment.py` (create) | `load_table`, `folds_for`, `run`, `params_sha`, `main` |
| `tune.py` (create) | `suggest`, `cv_score`, `tune`, `main` |
| `report4.py` (create) | `gate_checks`, `bias_slices`, `fig_p10`, `fig_pr`, `render`, `main` |
| `tests/conftest.py` (modify) | `synthetic_table` fixture: a 400-row, 2-repo table with every `COLUMN_SPEC` column and planted signal |
| `tests/test_featuresets.py`, `tests/test_model.py`, `tests/test_experiment.py`, `tests/test_tune.py`, `tests/test_report4.py` (create) | |
| outputs (gitignored except records): `data/models/`, `data/predictions/`, `data/phase4_runs.json`, `data/phase4_gate.json`, `docs/phase4_results.md`, `figures/phase4_*.png` | |

---

### Task 1: Dependencies, `featuresets.py`, and the synthetic-table fixture

**Files:**
- Modify: `requirements.txt`, `tests/conftest.py`, `.gitignore`
- Create: `featuresets.py`, `tests/test_featuresets.py`

**Interfaces:**
- Produces: `featuresets.FEATURE_SETS: dict[str, list[str]]` with keys `FULL, NO_SNAPSHOT, NO_LABEL_REPLAY, PR_ONLY`; `featuresets.LABEL_REPLAY: tuple[str, ...]`; `featuresets.assert_hygiene(cols: list[str]) -> None` (raises `ValueError`); fixture `synthetic_table(seed=0) -> pd.DataFrame` (2,000 rows, repos `o/r` and `o/s`, every `COLUMN_SPEC` column, `created_at` spanning 2024-06 → 2026-06, `is_slow` planted as `trailing_90d_slow_rate + 0.3·body_len_z + noise > 0.5`).

- [ ] **Step 1: Pin dependencies and ignore outputs**

Append to `requirements.txt`:
```
# Phase 4
lightgbm==4.7.0         # primary classifier (native API)
optuna==5.0.0           # hyperparameter search on Scenario A training rows
```
Append to `.gitignore`:
```
# Phase 4 model artefacts: regenerable via `python tune.py && python experiment.py`
data/models/
data/predictions/
```

- [ ] **Step 2: Add the synthetic-table fixture to `tests/conftest.py`**

Append:
```python
@pytest.fixture
def synthetic_table():
    """A features.parquet look-alike: every COLUMN_SPEC column, 2 repos, 2,000 rows spanning
    both sides of the Scenario A cutoff, with a PLANTED signal so a working model must
    beat chance. Values are random but typed like the real table."""
    import features as F

    def build(seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n = 2000    # after the 5%/repo training cap ~150 training rows survive; 400 would leave ~30
        created = pd.date_range("2024-06-01", "2026-06-15", periods=n, tz="UTC")
        repo = np.where(np.arange(n) % 2 == 0, "o/r", "o/s")
        df = pd.DataFrame({"repo": repo, "created_at": created,
                           "pr_id": [f"p{i}" for i in range(n)], "number": np.arange(n)})
        for col, meta in F.COLUMN_SPEC.items():
            if col in df.columns or meta["status"] in ("label",):
                continue
            dt = meta["dtype"]
            if col == "language_dominant":
                df[col] = np.where(repo == "o/r", "Go", "Python")
            elif dt == "bool":
                df[col] = rng.random(n) < 0.3
            elif dt == "int":
                df[col] = rng.integers(0, 50, n)
            elif dt == "float":
                df[col] = rng.random(n)
            else:
                df[col] = "x"
        df["timeline_may_be_truncated"] = False
        z = (df["body_len"] - df["body_len"].mean()) / (df["body_len"].std() + 1e-9)
        signal = df["trailing_90d_slow_rate"] + 0.5 * z + rng.normal(0, 0.10, n)
        df["is_slow"] = signal > 0.5
        df["wait_h"] = np.where(df["is_slow"], np.nan, rng.random(n) * 100)
        df["wait_h_censored"] = df["wait_h"].fillna(720.0)
        df["event_observed"] = ~df["is_slow"]
        df["never_reviewed_30d"] = df["is_slow"]
        return df[list(F.COLUMN_SPEC)].sort_values(["created_at", "pr_id"]).reset_index(drop=True)

    return build
```

- [ ] **Step 3: Write the failing tests**

`tests/test_featuresets.py`:
```python
import pytest
import features
import featuresets as fs


def test_counts_and_subsets():
    s = fs.FEATURE_SETS
    assert set(s) == {"FULL", "NO_SNAPSHOT", "NO_LABEL_REPLAY", "PR_ONLY"}
    assert len(s["FULL"]) == 37 and len(s["NO_SNAPSHOT"]) == 29
    assert len(s["NO_LABEL_REPLAY"]) == 33 and len(s["PR_ONLY"]) == 17
    for name, cols in s.items():
        assert set(cols) <= set(s["FULL"]), name
        assert len(cols) == len(set(cols)), name


def test_ablations_remove_what_they_claim():
    s = fs.FEATURE_SETS
    assert not any(features.COLUMN_SPEC[c]["status"] == "snapshot" for c in s["NO_SNAPSHOT"])
    assert not any(c in fs.LABEL_REPLAY for c in s["NO_LABEL_REPLAY"])
    assert all(features.COLUMN_SPEC[c]["group"] in ("static", "at_open") for c in s["PR_ONLY"])
    assert "trailing_90d_slow_rate" in s["FULL"] and "trailing_90d_slow_rate" not in s["NO_LABEL_REPLAY"]


@pytest.mark.parametrize("bad", ["is_slow", "pr_id", "diff_is_exact", "wait_h", "not_a_column"])
def test_hygiene_rejects_forbidden(bad):
    with pytest.raises(ValueError, match=bad):
        fs.assert_hygiene(fs.FEATURE_SETS["FULL"] + [bad])


def test_every_set_passes_hygiene():
    for cols in fs.FEATURE_SETS.values():
        fs.assert_hygiene(cols)


def test_synthetic_table_shape(synthetic_table):
    t = synthetic_table()
    assert list(t.columns) == list(features.COLUMN_SPEC) and len(t) == 2000
    assert t["is_slow"].mean() > 0.1 and t["is_slow"].mean() < 0.9
```

- [ ] **Step 4: Run to verify they fail**

Run: `python -m pytest tests/test_featuresets.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'featuresets'`

- [ ] **Step 5: Implement `featuresets.py`**

```python
"""The four pre-registered feature sets, derived from features.COLUMN_SPEC.

Nothing here is hand-listed: an ablation is a rule over the registry, so a column that
is a key, a label or a fidelity flag cannot end up in a model by accident. Ablations
exist to answer two questions the write-up must address -- does the model beat the
baseline WITHOUT the baseline's own feature (NO_LABEL_REPLAY), and are the repo-level
snapshots carrying the cold-start result (NO_SNAPSHOT)."""
from __future__ import annotations

from features import COLUMN_SPEC, feature_columns

# The baseline's feature and its author-level twin, with their support counts.
LABEL_REPLAY = ("trailing_90d_slow_rate", "trailing_n",
                "author_prior_slow_rate_here", "author_prior_n")

FORBIDDEN_STATUS = {"key", "label", "flag"}


def assert_hygiene(cols: list[str]) -> None:
    bad = [c for c in cols if c not in COLUMN_SPEC or COLUMN_SPEC[c]["status"] in FORBIDDEN_STATUS]
    if bad:
        raise ValueError(f"forbidden or unknown feature columns: {bad}")


def _full() -> list[str]:
    return list(feature_columns())


FEATURE_SETS: dict[str, list[str]] = {
    "FULL": _full(),
    "NO_SNAPSHOT": [c for c in _full() if COLUMN_SPEC[c]["status"] != "snapshot"],
    "NO_LABEL_REPLAY": [c for c in _full() if c not in LABEL_REPLAY],
    "PR_ONLY": [c for c in _full() if COLUMN_SPEC[c]["group"] in ("static", "at_open")],
}

for _name, _cols in FEATURE_SETS.items():
    assert_hygiene(_cols)
```

- [ ] **Step 6: Run to verify they pass; full suite; commit**

Run: `python -m pytest tests/test_featuresets.py -q` → `9 passed`. Then `python -m pytest tests -q` → all pass.
```bash
git add requirements.txt .gitignore featuresets.py tests/conftest.py tests/test_featuresets.py
git commit -m "feat(phase4): feature sets derived from COLUMN_SPEC; synthetic table fixture"
```

---

### Task 2: `model.py` — LightGBM wrapper

**Files:**
- Create: `model.py`, `tests/test_model.py`

**Interfaces:**
- Produces: `model.SEED`, `model.CATEGORICAL = ["language_dominant"]`, `model.FIXED: dict`, `model.TUNED_KEYS: tuple`, `model.fit(X: pd.DataFrame, y, params: dict) -> lgb.Booster`, `model.predict(booster, X) -> np.ndarray`, `model.save(booster, path: Path) -> None`, `model.load(path) -> lgb.Booster`, `model.DEFAULT_PARAMS: dict` (a sane untuned config for tests).

- [ ] **Step 1: Write the failing tests**

`tests/test_model.py`:
```python
import numpy as np
import pandas as pd
import pytest
import model
import featuresets as fs
import metrics


def _xy(synthetic_table, seed=0):
    t = synthetic_table(seed)
    return t[fs.FEATURE_SETS["FULL"]], t["is_slow"].astype(int)


def test_fit_predict_beats_chance(synthetic_table):
    X, y = _xy(synthetic_table)
    b = model.fit(X.iloc[:1500], y.iloc[:1500], model.DEFAULT_PARAMS)
    p = model.predict(b, X.iloc[1500:])
    assert p.shape == (500,) and (0 <= p).all() and (p <= 1).all()
    assert metrics.auc_pr(y.iloc[1500:], p) > y.iloc[1500:].mean() + 0.1   # planted signal


def test_two_fits_are_identical(synthetic_table):
    X, y = _xy(synthetic_table)
    p1 = model.predict(model.fit(X, y, model.DEFAULT_PARAMS), X)
    p2 = model.predict(model.fit(X, y, model.DEFAULT_PARAMS), X)
    assert np.array_equal(p1, p2)


def test_categorical_is_used_and_survives_roundtrip(synthetic_table, tmp_path):
    X, y = _xy(synthetic_table)
    b = model.fit(X, y, model.DEFAULT_PARAMS)
    assert "language_dominant" in b.feature_name()
    model.save(b, tmp_path / "m.txt")
    b2 = model.load(tmp_path / "m.txt")
    assert np.allclose(model.predict(b, X), model.predict(b2, X))


def test_scale_pos_weight_from_y(synthetic_table):
    X, y = _xy(synthetic_table)
    b = model.fit(X, y, model.DEFAULT_PARAMS)
    expected = (len(y) - y.sum()) / y.sum()
    assert float(b.params["scale_pos_weight"]) == pytest.approx(expected)


def test_spec_param_names_are_translated(synthetic_table):
    X, y = _xy(synthetic_table)
    b = model.fit(X, y, {**model.DEFAULT_PARAMS, "n_estimators": 7, "min_child_samples": 33})
    assert b.num_trees() == 7
    assert int(b.params["min_data_in_leaf"]) == 33


def test_unknown_param_is_rejected(synthetic_table):
    X, y = _xy(synthetic_table)
    with pytest.raises(KeyError):
        model.fit(X, y, {**model.DEFAULT_PARAMS, "bogus": 1})
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_model.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'model'`

- [ ] **Step 3: Implement `model.py`**

```python
"""LightGBM in one place: fixed seeds, the categorical, scale_pos_weight, save/load.

Native lgb.train rather than the sklearn wrapper: native save/load of the booster and
unambiguous parameter names. The SPEC's names (n_estimators, min_child_samples) stay the
public vocabulary -- tune.py and params.json use them -- and are translated here."""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

SEED = 20260912
CATEGORICAL = ["language_dominant"]

# Everything that must never vary between runs. deterministic + single thread + seeds is
# what makes gate #5 (refit reproduces AUC-PR to 1e-6) achievable.
FIXED = {
    "objective": "binary", "seed": SEED, "deterministic": True, "force_row_wise": True,
    "num_threads": 1, "verbose": -1, "bagging_seed": SEED, "feature_fraction_seed": SEED,
}

# The tunable vocabulary (spec §7). n_estimators and min_child_samples are sklearn-style
# names translated below; the rest are native LightGBM names.
TUNED_KEYS = ("num_leaves", "learning_rate", "n_estimators", "min_child_samples",
              "feature_fraction", "bagging_fraction", "bagging_freq", "lambda_l2")
_TRANSLATE = {"min_child_samples": "min_data_in_leaf"}

DEFAULT_PARAMS = {
    "num_leaves": 31, "learning_rate": 0.05, "n_estimators": 200, "min_child_samples": 10,
    "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1, "lambda_l2": 1.0,
}


def _prepare(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def fit(X: pd.DataFrame, y, params: dict) -> lgb.Booster:
    unknown = set(params) - set(TUNED_KEYS)
    if unknown:
        raise KeyError(f"unknown params: {sorted(unknown)}")
    y = np.asarray(y).astype(int)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    native = {**FIXED, "scale_pos_weight": neg / max(pos, 1)}
    for k, v in params.items():
        if k == "n_estimators":
            continue
        native[_TRANSLATE.get(k, k)] = v
    ds = lgb.Dataset(_prepare(X), label=y,
                     categorical_feature=[c for c in CATEGORICAL if c in X.columns],
                     free_raw_data=False)
    return lgb.train(native, ds, num_boost_round=int(params["n_estimators"]))


def predict(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(booster.predict(_prepare(X)))


def save(booster: lgb.Booster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))


def load(path: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))
```

- [ ] **Step 4: Run to verify they pass; full suite; commit**

Run: `python -m pytest tests/test_model.py -q` → `6 passed`; `python -m pytest tests -q` → all pass, no warnings. If LightGBM emits a categorical-related `UserWarning` at predict, it is a finding: fix by ensuring the categories at predict time match training (the `_prepare` cast) rather than suppressing.
```bash
git add model.py tests/test_model.py
git commit -m "feat(phase4): LightGBM wrapper with fixed seeds, categorical, spec param names"
```

---

### Task 3: `experiment.py` — one run, saved, logged

**Files:**
- Create: `experiment.py`, `tests/test_experiment.py`

**Interfaces:**
- Consumes: `featuresets.FEATURE_SETS`, `model.fit/predict/save`, `splits.scenario_a/scenario_b/CUTOFF_A/SEED`, `metrics.precision_at_k/auc_pr/cluster_bootstrap`, `tracking.log`.
- Produces: `experiment.TABLE, PARAMS, MODELS_DIR, PRED_DIR, RUNS_JSON: Path`; `experiment.load_table_frame(df) -> pd.DataFrame`; `experiment.load_table(path=None) -> pd.DataFrame`; `experiment.folds_for(scenario: str, table) -> list[tuple[np.ndarray, np.ndarray]]`; `experiment.params_sha(path=PARAMS) -> str`; `experiment.run(scenario, fold, name, table, params, train_idx, test_idx, *, seed=SEED, out_models=MODELS_DIR, out_preds=PRED_DIR) -> dict` with keys `scenario, fold, featureset, n_train, n_test, n_test_repos, precision_at_10, p10_ci_lo, p10_ci_hi, base_rate_p10, auc_pr, base_rate, baseline_p10, baseline_auc_pr, train_max_created, test_min_created, train_repos, test_repos, model_path, pred_path`; `experiment.main()`.

- [ ] **Step 1: Write the failing tests**

`tests/test_experiment.py`:
```python
import json
import numpy as np
import pandas as pd
import experiment as ex
import model
import splits


def test_load_table_drops_truncated_and_sorts(synthetic_table, tmp_path):
    t = synthetic_table()
    t.loc[5, "timeline_may_be_truncated"] = True
    t = t.sample(frac=1, random_state=1)                      # shuffle
    p = tmp_path / "f.parquet"; t.to_parquet(p, index=False)
    out = ex.load_table(p)
    assert len(out) == 1999 and out["created_at"].is_monotonic_increasing
    assert str(out["language_dominant"].dtype) == "category"
    assert list(out.index) == list(range(1999))


def test_folds_for_shapes(synthetic_table):
    t = ex.load_table_frame(synthetic_table())
    a = ex.folds_for("A", t); b = ex.folds_for("B", t, n_splits=2)
    assert len(a) == 1 and len(b) == 2
    tr, te = a[0]
    assert (t.loc[te, "created_at"] >= splits.CUTOFF_A).all()
    assert (t.loc[tr, "created_at"] < splits.CUTOFF_A).all()


def test_run_end_to_end(synthetic_table, tmp_path):
    t = ex.load_table_frame(synthetic_table())
    tr, te = ex.folds_for("A", t)[0]
    res = ex.run("A", 0, "FULL", t, model.DEFAULT_PARAMS, tr, te,
                 out_models=tmp_path / "m", out_preds=tmp_path / "p")
    for k in ("precision_at_10", "p10_ci_lo", "p10_ci_hi", "base_rate_p10", "auc_pr", "base_rate",
              "baseline_p10", "baseline_auc_pr", "n_test_repos", "train_max_created", "test_min_created"):
        assert k in res, k
    assert res["n_test_repos"] == 2 and res["p10_ci_lo"] <= res["precision_at_10"] <= res["p10_ci_hi"]
    assert pd.Timestamp(res["train_max_created"]) < pd.Timestamp(res["test_min_created"])
    assert res["auc_pr"] > res["base_rate"]                    # planted signal is learnable
    preds = pd.read_parquet(res["pred_path"])
    assert set(preds.columns) >= {"pr_id", "repo", "created_at", "is_slow", "p_hat", "baseline_score",
                                  "is_first_pr_here", "created_hour_utc", "diff_is_exact"}
    assert len(preds) == len(te)
    assert (tmp_path / "m" / "A_FULL_fold0.txt").exists()


def test_run_scenario_b_repos_disjoint(synthetic_table, tmp_path):
    t = ex.load_table_frame(synthetic_table())
    for k, (tr, te) in enumerate(ex.folds_for("B", t, n_splits=2)):
        res = ex.run("B", k, "PR_ONLY", t, model.DEFAULT_PARAMS, tr, te,
                     out_models=tmp_path / "m", out_preds=tmp_path / "p")
        assert not (set(res["train_repos"]) & set(res["test_repos"]))


def test_params_sha_is_stable(tmp_path):
    p = tmp_path / "params.json"; p.write_text('{"best_params": {"num_leaves": 8}}', encoding="utf-8")
    assert ex.params_sha(p) == ex.params_sha(p) and len(ex.params_sha(p)) == 12
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_experiment.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'experiment'`

- [ ] **Step 3: Implement `experiment.py`**

```python
"""One experiment = (scenario, fold, feature set). Fit, predict, score, save, log.

Model and baseline are always scored on the SAME test rows: the baseline's score for a
row is the table's own trailing_90d_slow_rate (proven identical to baseline.py's value by
the Phase 3 audit), so no second pipeline is needed and no rows can differ."""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import featuresets as fs
import metrics
import model
import splits
import tracking

log = logging.getLogger("experiment")

ROOT = Path(__file__).parent
TABLE = ROOT / "data" / "features" / "features.parquet"
PARAMS = ROOT / "data" / "models" / "params.json"
MODELS_DIR = ROOT / "data" / "models"
PRED_DIR = ROOT / "data" / "predictions"
RUNS_JSON = ROOT / "data" / "phase4_runs.json"
SEED = splits.SEED
SCENARIOS = ("A", "B")


def load_table_frame(t: pd.DataFrame) -> pd.DataFrame:
    t = t[~(t["timeline_may_be_truncated"] == True)]                    # noqa: E712
    t = t.sort_values(["created_at", "pr_id"]).reset_index(drop=True)
    t["language_dominant"] = t["language_dominant"].astype("category")
    return t


def load_table(path: Path | None = None) -> pd.DataFrame:
    # Default resolved at CALL time so tests can monkeypatch experiment.TABLE.
    return load_table_frame(pd.read_parquet(path or TABLE))


def folds_for(scenario: str, table: pd.DataFrame, n_splits: int = 5):
    if scenario == "A":
        return [splits.scenario_a(table)]
    if scenario == "B":
        return splits.scenario_b(table, n_splits=n_splits)
    raise ValueError(scenario)


def params_sha(path: Path | None = None) -> str:
    return hashlib.sha256((path or PARAMS).read_bytes()).hexdigest()[:12]


def run(scenario: str, fold: int, name: str, table: pd.DataFrame, params: dict,
        train_idx: np.ndarray, test_idx: np.ndarray, *, seed: int = SEED,
        out_models: Path = MODELS_DIR, out_preds: Path = PRED_DIR) -> dict:
    cols = fs.FEATURE_SETS[name]
    fs.assert_hygiene(cols)
    tr, te = table.loc[train_idx], table.loc[test_idx]

    booster = model.fit(tr[cols], tr["is_slow"], params)
    p_hat = model.predict(booster, te[cols])
    y = te["is_slow"].to_numpy(dtype=int)
    repo = te["repo"].to_numpy()

    per_repo, p10 = metrics.precision_at_k(y, p_hat, repo, k=10, seed=seed)
    lo, hi = metrics.cluster_bootstrap(per_repo, seed=seed)
    base_score = te["trailing_90d_slow_rate"].to_numpy(dtype=float)
    _, base_p10 = metrics.precision_at_k(y, base_score, repo, k=10, seed=seed)

    tag = f"{scenario}_{name}_fold{fold}"
    model_path = out_models / f"{tag}.txt"
    pred_path = out_preds / f"{tag}.parquet"
    model.save(booster, model_path)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "pr_id": te["pr_id"].to_numpy(), "repo": repo, "created_at": te["created_at"].to_numpy(),
        "is_slow": y.astype(bool), "p_hat": p_hat, "baseline_score": base_score,
        "is_first_pr_here": te["is_first_pr_here"].to_numpy(),
        "created_hour_utc": te["created_hour_utc"].to_numpy(),
        "diff_is_exact": te["diff_is_exact"].to_numpy(),
    }).to_parquet(pred_path, index=False)

    return {
        "scenario": scenario, "fold": fold, "featureset": name,
        "n_train": int(len(tr)), "n_test": int(len(te)), "n_test_repos": int(te["repo"].nunique()),
        "precision_at_10": float(p10), "p10_ci_lo": float(lo), "p10_ci_hi": float(hi),
        "base_rate_p10": float(te.groupby("repo")["is_slow"].mean().mean()),
        "auc_pr": metrics.auc_pr(y, p_hat), "base_rate": float(y.mean()),
        "baseline_p10": float(base_p10), "baseline_auc_pr": metrics.auc_pr(y, base_score),
        "train_max_created": str(tr["created_at"].max()), "test_min_created": str(te["created_at"].min()),
        "train_repos": sorted(tr["repo"].unique().tolist()), "test_repos": sorted(te["repo"].unique().tolist()),
        "model_path": str(model_path), "pred_path": str(pred_path),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if not PARAMS.exists():
        log.error("no %s -- run tune.py first", PARAMS)
        return 1
    params = json.loads(PARAMS.read_text(encoding="utf-8"))["best_params"]
    sha = params_sha()
    table = load_table()
    results = []
    for scenario in SCENARIOS:
        folds = folds_for(scenario, table)
        for name in fs.FEATURE_SETS:
            for k, (tr, te) in enumerate(folds):
                res = run(scenario, k, name, table, params, tr, te)
                results.append(res)
                tracking.log({
                    "scenario": scenario, "fold": k, "model": "lgbm", "features": name, "params": sha,
                    "n_train": res["n_train"], "n_test": res["n_test"], "n_test_repos": res["n_test_repos"],
                    "precision_at_10": res["precision_at_10"], "p10_ci_lo": res["p10_ci_lo"],
                    "p10_ci_hi": res["p10_ci_hi"], "base_rate_p10": res["base_rate_p10"],
                    "auc_pr": res["auc_pr"], "base_rate": res["base_rate"],
                    "notes": f"baseline_p10={res['baseline_p10']:.3f} baseline_auc_pr={res['baseline_auc_pr']:.3f}",
                })
                log.info("%s %-16s fold %d: P@10=%.3f [%.3f,%.3f] base=%.3f bl=%.3f | AUC-PR=%.3f bl=%.3f",
                         scenario, name, k, res["precision_at_10"], res["p10_ci_lo"], res["p10_ci_hi"],
                         res["base_rate_p10"], res["baseline_p10"], res["auc_pr"], res["baseline_auc_pr"])
    RUNS_JSON.write_text(json.dumps({"params_sha": sha, "runs": results}, indent=2), encoding="utf-8")
    print(f"{len(results)} runs -> {RUNS_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify they pass; full suite; commit**

Run: `python -m pytest tests/test_experiment.py -q` → `5 passed`; `python -m pytest tests -q` → all pass, no warnings.
```bash
git add experiment.py tests/test_experiment.py
git commit -m "feat(phase4): experiment runner -- fit, score vs baseline on same rows, save, log"
```

---

### Task 4: `tune.py` — Optuna on Scenario A training rows

**Files:**
- Create: `tune.py`, `tests/test_tune.py`

**Interfaces:**
- Consumes: `experiment.load_table`, `experiment.folds_for`, `featuresets.FEATURE_SETS["FULL"]`, `model.fit/predict`, `metrics.auc_pr`.
- Produces: `tune.suggest(trial) -> dict` (spec §7 space); `tune.cv_score(X, y, params, n_splits=3) -> float`; `tune.tune(table, n_trials=100, seed=SEED, n_splits=3) -> dict` with keys `best_params, cv_auc_pr, n_trials, seed, trials`; `tune.main()` writing `experiment.PARAMS`.

- [ ] **Step 1: Write the failing tests**

`tests/test_tune.py`:
```python
import json
import experiment as ex
import model
import tune


def test_cv_score_is_time_ordered_and_sane(synthetic_table):
    t = ex.load_table_frame(synthetic_table())
    tr, _ = ex.folds_for("A", t)[0]
    rows = t.loc[tr].sort_values("created_at")
    s = tune.cv_score(rows[tune.COLS], rows["is_slow"].astype(int), model.DEFAULT_PARAMS, n_splits=3)
    assert 0.0 < s <= 1.0


def test_tune_returns_valid_params_and_is_reproducible(synthetic_table):
    t = ex.load_table_frame(synthetic_table())
    r1 = tune.tune(t, n_trials=4, seed=1, n_splits=2)
    r2 = tune.tune(t, n_trials=4, seed=1, n_splits=2)
    assert set(r1["best_params"]) == set(model.TUNED_KEYS)
    assert r1["n_trials"] == 4 and len(r1["trials"]) == 4
    assert r1["best_params"] == r2["best_params"] and r1["cv_auc_pr"] == r2["cv_auc_pr"]
    model.fit(t[tune.COLS].iloc[:50], t["is_slow"].iloc[:50], r1["best_params"])   # accepted by fit


def test_main_writes_params(synthetic_table, tmp_path, monkeypatch):
    t = synthetic_table(); p = tmp_path / "f.parquet"; t.to_parquet(p, index=False)
    monkeypatch.setattr(ex, "TABLE", p); monkeypatch.setattr(ex, "PARAMS", tmp_path / "params.json")
    assert tune.main(["--trials", "3", "--splits", "2"]) == 0
    out = json.loads((tmp_path / "params.json").read_text())
    assert "best_params" in out and out["n_trials"] == 3
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_tune.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tune'`

- [ ] **Step 3: Implement `tune.py`**

```python
"""Optuna on Scenario A's TRAINING rows only, time-ordered CV, FULL feature set.

The held-out test period and Scenario B's held-out repos never influence tuning. The
best params are frozen in data/models/params.json and reused for every scenario, fold
and ablation -- the fair comparison, stated in the report."""
from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import experiment as ex
import featuresets as fs
import metrics
import model

log = logging.getLogger("tune")
SEED = model.SEED
COLS = fs.FEATURE_SETS["FULL"]


def suggest(trial: optuna.Trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
    }


def cv_score(X: pd.DataFrame, y, params: dict, n_splits: int = 3) -> float:
    """Mean validation AUC-PR over a time-ordered split; X must be sorted by time."""
    y = pd.Series(np.asarray(y).astype(int), index=X.index)
    scores = []
    for tr, va in TimeSeriesSplit(n_splits=n_splits).split(X):
        b = model.fit(X.iloc[tr], y.iloc[tr], params)
        scores.append(metrics.auc_pr(y.iloc[va], model.predict(b, X.iloc[va])))
    return float(np.mean(scores))


def tune(table: pd.DataFrame, n_trials: int = 100, seed: int = SEED, n_splits: int = 3) -> dict:
    tr, _ = ex.folds_for("A", table)[0]
    rows = table.loc[tr].sort_values(["created_at", "pr_id"])
    X, y = rows[COLS], rows["is_slow"].astype(int)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda t: cv_score(X, y, suggest(t), n_splits), n_trials=n_trials)
    best = {**study.best_params, "bagging_freq": 1}
    return {
        "best_params": best, "cv_auc_pr": float(study.best_value), "n_trials": n_trials,
        "seed": seed, "n_splits": n_splits, "n_train_rows": int(len(rows)),
        "trials": [{"number": t.number, "value": t.value, "params": {**t.params, "bagging_freq": 1}}
                   for t in study.trials],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--splits", type=int, default=3)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    table = ex.load_table()
    out = tune(table, n_trials=args.trials, n_splits=args.splits)
    ex.PARAMS.parent.mkdir(parents=True, exist_ok=True)
    ex.PARAMS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"best cv AUC-PR={out['cv_auc_pr']:.4f} over {out['n_trials']} trials -> {ex.PARAMS}")
    print(json.dumps(out["best_params"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify they pass; full suite; commit**

Run: `python -m pytest tests/test_tune.py -q` → `3 passed`; `python -m pytest tests -q` → all pass, no warnings (if Optuna emits an `ExperimentalWarning`, silence it at the source with `optuna.logging`/`warnings.filterwarnings` scoped to that category in `tune.py`, and say so in the report).
```bash
git add tune.py tests/test_tune.py
git commit -m "feat(phase4): Optuna tuning on Scenario A training rows, time-ordered CV"
```

---

### Task 5: `report4.py` — gate, bias slices, figures, results doc

**Files:**
- Create: `report4.py`, `tests/test_report4.py`

**Interfaces:**
- Consumes: `experiment.RUNS_JSON/PARAMS/PRED_DIR/load_table/folds_for/run`, `featuresets.FEATURE_SETS/assert_hygiene`, `tracking.EXPERIMENTS`, `splits.CUTOFF_A`, `model`.
- Produces: `report4.gate_checks(runs: list[dict], params_sha: str, experiments: pd.DataFrame, repro_delta: float | None) -> list[dict]` (five `{"id","check","value","pass"}`); `report4.bias_slices(pred: pd.DataFrame) -> pd.DataFrame` (columns `slice, level, n, actual_rate, mean_p_hat, auc_pr`); `report4.summarise(runs) -> pd.DataFrame`; `report4.fig_p10`, `report4.fig_pr`; `report4.render(...) -> str`; `report4.main()`.

- [ ] **Step 1: Write the failing tests**

`tests/test_report4.py`:
```python
import numpy as np
import pandas as pd
import report4


def _run(scenario, fold, name, **kw):
    base = {"scenario": scenario, "fold": fold, "featureset": name, "n_train": 100, "n_test": 50,
            "n_test_repos": 6, "precision_at_10": 0.6, "p10_ci_lo": 0.5, "p10_ci_hi": 0.7,
            "base_rate_p10": 0.5, "auc_pr": 0.7, "base_rate": 0.45, "baseline_p10": 0.5,
            "baseline_auc_pr": 0.6, "train_max_created": "2025-12-31 23:00:00+00:00",
            "test_min_created": "2026-01-01 00:00:00+00:00", "train_repos": ["a", "b"],
            "test_repos": ["a", "b"] if scenario == "A" else ["c", "d"],
            "model_path": "m", "pred_path": "p"}
    return {**base, **kw}


def _all_runs():
    runs = [_run("A", 0, n) for n in ("FULL", "NO_SNAPSHOT", "NO_LABEL_REPLAY", "PR_ONLY")]
    runs += [_run("B", k, n) for n in ("FULL", "NO_SNAPSHOT", "NO_LABEL_REPLAY", "PR_ONLY") for k in range(5)]
    return runs


def _experiments(sha, n=24):
    return pd.DataFrame({"model": ["lgbm"] * n, "params": [sha] * n, "p10_ci_lo": [0.5] * n, "p10_ci_hi": [0.7] * n})


def test_gate_all_pass():
    checks = report4.gate_checks(_all_runs(), "abc123", _experiments("abc123"), repro_delta=0.0)
    assert [c["id"] for c in checks] == [1, 2, 3, 4, 5] and all(c["pass"] for c in checks)


def test_gate_catches_time_leak():
    runs = _all_runs(); runs[0]["train_max_created"] = "2026-02-01 00:00:00+00:00"
    assert report4.gate_checks(runs, "s", _experiments("s"), 0.0)[0]["pass"] is False


def test_gate_catches_repo_leak():
    runs = _all_runs(); b = next(r for r in runs if r["scenario"] == "B"); b["test_repos"] = ["a"]
    assert report4.gate_checks(runs, "s", _experiments("s"), 0.0)[0]["pass"] is False


def test_gate_requires_no_label_replay_on_both_and_all_24_logged():
    runs = [r for r in _all_runs() if not (r["featureset"] == "NO_LABEL_REPLAY" and r["scenario"] == "B")]
    c = report4.gate_checks(runs, "s", _experiments("s"), 0.0)
    assert c[2]["pass"] is False and c[3]["pass"] is False
    assert report4.gate_checks(_all_runs(), "s", _experiments("s", n=23), 0.0)[3]["pass"] is False


def test_gate_repro_threshold():
    assert report4.gate_checks(_all_runs(), "s", _experiments("s"), 1e-7)[4]["pass"] is True
    assert report4.gate_checks(_all_runs(), "s", _experiments("s"), 1e-3)[4]["pass"] is False
    assert report4.gate_checks(_all_runs(), "s", _experiments("s"), None)[4]["pass"] is False


def test_bias_slices_have_expected_rows():
    rng = np.random.default_rng(0); n = 300
    pred = pd.DataFrame({"is_slow": rng.random(n) < 0.4, "p_hat": rng.random(n),
                         "is_first_pr_here": rng.random(n) < 0.3, "created_hour_utc": rng.integers(0, 24, n)})
    s = report4.bias_slices(pred)
    assert set(s["slice"]) == {"is_first_pr_here", "hour_bucket"}
    assert set(s[s["slice"] == "hour_bucket"]["level"]) == {"00-06", "06-12", "12-18", "18-24"}
    assert (s["n"] > 0).all() and s["n"].sum() == 2 * n
    assert set(s.columns) >= {"slice", "level", "n", "actual_rate", "mean_p_hat", "auc_pr"}


def test_summarise_means_over_folds():
    df = report4.summarise(_all_runs())
    assert set(df["scenario"]) == {"A", "B"} and len(df) == 8
    assert df[(df.scenario == "B") & (df.featureset == "FULL")]["n_folds"].iloc[0] == 5
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_report4.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'report4'`

- [ ] **Step 3: Implement `report4.py`**

```python
"""Phase 4 results document + validity gate.

The gate tests VALIDITY (disjoint splits, feature hygiene, the blueprint's ablation ran,
everything logged, reproducible), never success: failing to beat the baseline is a
reportable finding, and the headline section is written to be true either way."""
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
from sklearn.metrics import precision_recall_curve

import experiment as ex
import featuresets as fs
import metrics
import model
import splits
import tracking

log = logging.getLogger("report4")
ROOT = Path(__file__).parent
DOC = ROOT / "docs" / "phase4_results.md"
FIG = ROOT / "figures"
GATE_JSON = ROOT / "data" / "phase4_gate.json"
SETS = list(fs.FEATURE_SETS)
HOUR_BUCKETS = [(0, 6, "00-06"), (6, 12, "06-12"), (12, 18, "12-18"), (18, 24, "18-24")]
N_RUNS = 4 * (1 + 5)


# ----------------------------------------------------------------------------
# Gate (pure)
# ----------------------------------------------------------------------------

def gate_checks(runs: list[dict], params_sha: str, experiments: pd.DataFrame,
                repro_delta: float | None) -> list[dict]:
    # 1. disjointness
    leaks = []
    for r in runs:
        if r["scenario"] == "A":
            if not (pd.Timestamp(r["train_max_created"]) < pd.Timestamp(r["test_min_created"])
                    and pd.Timestamp(r["test_min_created"]) >= splits.CUTOFF_A):
                leaks.append(f"A/{r['featureset']}: time")
        else:
            if set(r["train_repos"]) & set(r["test_repos"]):
                leaks.append(f"B/{r['featureset']}/fold{r['fold']}: repo overlap")
    c1 = {"id": 1, "check": "splits disjoint: A by time (train < cutoff <= test), B by repo per fold",
          "value": leaks or "ok", "pass": not leaks}

    # 2. hygiene
    bad = []
    for name, cols in fs.FEATURE_SETS.items():
        try:
            fs.assert_hygiene(cols)
        except ValueError as exc:
            bad.append(f"{name}: {exc}")
    c2 = {"id": 2, "check": "no key/label/flag column in any feature set", "value": bad or "ok", "pass": not bad}

    # 3. the blueprint's ablation ran on both scenarios
    have = {(r["scenario"], r["featureset"]) for r in runs}
    ok3 = ("A", "NO_LABEL_REPLAY") in have and ("B", "NO_LABEL_REPLAY") in have
    c3 = {"id": 3, "check": "NO_LABEL_REPLAY ran on both scenarios", "value": sorted(map(str, have)), "pass": ok3}

    # 4. every run has a CI and all 24 are logged under this params sha
    ci_ok = all(np.isfinite(r["p10_ci_lo"]) and np.isfinite(r["p10_ci_hi"]) for r in runs)
    logged = int(((experiments["model"] == "lgbm") & (experiments["params"] == params_sha)).sum()) if len(experiments) else 0
    c4 = {"id": 4, "check": f"all {N_RUNS} runs have bootstrap CIs and are in experiments.csv under params sha",
          "value": {"runs": len(runs), "logged": logged, "ci_ok": ci_ok},
          "pass": ci_ok and len(runs) == N_RUNS and logged == N_RUNS}

    # 5. reproducibility
    c5 = {"id": 5, "check": "refit A/FULL from params.json reproduces AUC-PR (|delta| < 1e-6)",
          "value": repro_delta, "pass": repro_delta is not None and abs(repro_delta) < 1e-6}
    return [c1, c2, c3, c4, c5]


# ----------------------------------------------------------------------------
# Tables (pure)
# ----------------------------------------------------------------------------

def summarise(runs: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(runs)
    g = df.groupby(["scenario", "featureset"], sort=False)
    out = g.agg(n_folds=("fold", "count"), precision_at_10=("precision_at_10", "mean"),
                p10_ci_lo=("p10_ci_lo", "mean"), p10_ci_hi=("p10_ci_hi", "mean"),
                baseline_p10=("baseline_p10", "mean"), base_rate_p10=("base_rate_p10", "mean"),
                auc_pr=("auc_pr", "mean"), baseline_auc_pr=("baseline_auc_pr", "mean")).reset_index()
    order = {n: i for i, n in enumerate(SETS)}
    return out.sort_values(["scenario", "featureset"], key=lambda s: s.map(order) if s.name == "featureset" else s).reset_index(drop=True)


def bias_slices(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(slice_name, level, mask):
        sub = pred[mask]
        y = sub["is_slow"].to_numpy(dtype=int)
        auc = metrics.auc_pr(y, sub["p_hat"]) if 0 < y.sum() < len(y) else float("nan")
        rows.append({"slice": slice_name, "level": level, "n": int(len(sub)),
                     "actual_rate": float(y.mean()) if len(sub) else float("nan"),
                     "mean_p_hat": float(sub["p_hat"].mean()) if len(sub) else float("nan"), "auc_pr": auc})
    f = pred["is_first_pr_here"].astype(bool)
    add("is_first_pr_here", "first-time", f); add("is_first_pr_here", "repeat", ~f)
    h = pred["created_hour_utc"].astype(int)
    for lo, hi, name in HOUR_BUCKETS:
        add("hour_bucket", name, (h >= lo) & (h < hi))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def fig_p10(summary: pd.DataFrame) -> Path:
    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, sc in zip(axes, ("A", "B")):
        s = summary[summary["scenario"] == sc]
        x = np.arange(len(s))
        ax.bar(x, s["precision_at_10"], yerr=[s["precision_at_10"] - s["p10_ci_lo"], s["p10_ci_hi"] - s["precision_at_10"]],
               capsize=4, label="LightGBM")
        ax.scatter(x, s["baseline_p10"], marker="_", s=400, color="k", label="baseline P@10")
        ax.scatter(x, s["base_rate_p10"], marker="x", color="gray", label="base rate")
        ax.set_xticks(x); ax.set_xticklabels(s["featureset"], rotation=20); ax.set_title(f"Scenario {sc}")
    axes[0].set_ylabel("Precision@10 (per repo, mean)"); axes[0].legend(fontsize=8)
    p = FIG / "phase4_p10.png"; fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_pr(scenario: str, pred_paths: list[Path]) -> Path:
    pred = pd.concat([pd.read_parquet(p) for p in pred_paths], ignore_index=True)
    y = pred["is_slow"].astype(int)
    fig, ax = plt.subplots(figsize=(5, 4))
    for col, label in (("p_hat", "LightGBM FULL"), ("baseline_score", "trailing-90d baseline")):
        pr, rc, _ = precision_recall_curve(y, pred[col]); ax.plot(rc, pr, label=label)
    ax.axhline(y.mean(), ls="--", color="gray", label=f"base rate {y.mean():.2f}")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title(f"Scenario {scenario}"); ax.legend(fontsize=8)
    p = FIG / f"phase4_pr_{scenario}.png"; fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p


# ----------------------------------------------------------------------------
# Document
# ----------------------------------------------------------------------------

def md(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def headline(summary: pd.DataFrame) -> str:
    a = summary[(summary.scenario == "A") & (summary.featureset == "FULL")].iloc[0]
    b = summary[(summary.scenario == "B") & (summary.featureset == "FULL")].iloc[0]
    d_bl = a.precision_at_10 - a.baseline_p10
    d_br = a.precision_at_10 - a.base_rate_p10
    excl_bl = "excludes" if not (a.p10_ci_lo <= a.baseline_p10 <= a.p10_ci_hi) else "includes"
    excl_br = "excludes" if not (a.p10_ci_lo <= a.base_rate_p10 <= a.p10_ci_hi) else "includes"
    return (f"On Scenario A, the FULL model's within-repo Precision@10 is **{a.precision_at_10:.3f}** "
            f"[{a.p10_ci_lo:.3f}, {a.p10_ci_hi:.3f}] against the trailing-rate baseline's {a.baseline_p10:.3f} "
            f"(**{d_bl:+.3f}**; the CI {excl_bl} the baseline) and the base rate {a.base_rate_p10:.3f} "
            f"({d_br:+.3f}; the CI {excl_br} the base rate). AUC-PR {a.auc_pr:.3f} vs baseline {a.baseline_auc_pr:.3f}. "
            f"On Scenario B (cold-start), FULL averages P@10 {b.precision_at_10:.3f} vs baseline {b.baseline_p10:.3f}, "
            f"AUC-PR {b.auc_pr:.3f} vs {b.baseline_auc_pr:.3f} — an A→B P@10 gap of {a.precision_at_10 - b.precision_at_10:+.3f}. "
            f"The blueprint's bar was a clear margin over the baseline on A (5–10 points); "
            f"{'that bar is met' if d_bl >= 0.05 else 'that bar is NOT met — PR-level signal adds ' + ('little' if d_bl > 0 else 'nothing') + ' over the repo trailing rate within-repo, which is itself the reportable finding'}.")


def render(params: dict, params_sha: str, summary: pd.DataFrame, runs: list[dict], slices: dict[str, pd.DataFrame],
           checks: list[dict], figs: dict[str, Path], n_rows: int, n_dropped: int) -> str:
    full_a = summary[(summary.scenario == "A") & (summary.featureset == "FULL")].iloc[0]
    abl = summary.copy()
    abl["d_p10_vs_FULL"] = abl.apply(lambda r: r.precision_at_10 - summary[(summary.scenario == r.scenario) & (summary.featureset == "FULL")].precision_at_10.iloc[0], axis=1)
    abl["d_auc_pr_vs_FULL"] = abl.apply(lambda r: r.auc_pr - summary[(summary.scenario == r.scenario) & (summary.featureset == "FULL")].auc_pr.iloc[0], axis=1)
    b_full = pd.DataFrame([r for r in runs if r["scenario"] == "B" and r["featureset"] == "FULL"])[
        ["fold", "n_test_repos", "precision_at_10", "p10_ci_lo", "p10_ci_hi", "baseline_p10", "base_rate_p10", "auc_pr", "baseline_auc_pr"]]
    verdict = "PASS" if all(c["pass"] for c in checks) else "FAIL"
    first = {sc: slices[sc][slices[sc]["slice"] == "is_first_pr_here"] for sc in slices}
    bias_line = []
    for sc, s in first.items():
        ft, rp = s[s.level == "first-time"].iloc[0], s[s.level == "repeat"].iloc[0]
        bias_line.append(f"Scenario {sc}: first-timers actual {ft.actual_rate:.3f} vs predicted {ft.mean_p_hat:.3f} "
                         f"({ft.mean_p_hat - ft.actual_rate:+.3f}); repeat {rp.actual_rate:.3f} vs {rp.mean_p_hat:.3f} ({rp.mean_p_hat - rp.actual_rate:+.3f}).")
    return f"""# Phase 4 — LightGBM Results

Generated by `report4.py`. Params sha `{params_sha}`. Rows: {n_rows:,} ({n_dropped} truncated-timeline rows dropped).

## Gate: **{verdict}** (validity, not success)

{md(pd.DataFrame(checks)[["id", "check", "value", "pass"]], "{}")}

## 1. Setup

Tuned on Scenario A training rows, `TimeSeriesSplit(3)`, {params['n_trials']} Optuna trials, CV AUC-PR **{params['cv_auc_pr']:.4f}**.
Params (frozen for every scenario, fold and ablation — ablations are therefore compared under identical settings, not re-tuned):

```
{json.dumps(params['best_params'], indent=2)}
```

## 2. Scenario A (known-project, time cutoff 2026-01-01)

{md(summary[summary.scenario == "A"].drop(columns=["scenario", "n_folds"]))}

## 3. Scenario B (cold-start, leave-repos-out, 5 folds; means over folds)

{md(summary[summary.scenario == "B"].drop(columns=["scenario"]))}

FULL per fold:

{md(b_full)}

## 4. Headline

{headline(summary)}

## 5. Ablations (Δ vs FULL, same scenario)

{md(abl[["scenario", "featureset", "precision_at_10", "d_p10_vs_FULL", "auc_pr", "d_auc_pr_vs_FULL"]])}

NO_LABEL_REPLAY answers the blueprint's question directly: it is the model without the baseline's own feature. NO_SNAPSHOT tests the feature dictionary's caveat that HEAD-at-collection repo facts may carry the cold-start result. PR_ONLY is what a PR looks like with no history at all.

## 6. Bias slices (FULL predictions)

{chr(10).join("### Scenario " + sc + chr(10) + chr(10) + md(s) for sc, s in slices.items())}

{" ".join(bias_line)}
A positive (predicted − actual) gap for first-timers means the model is more pessimistic about newcomers than their outcomes warrant — blueprint §4's fairness concern.

## 7. Figures

![p10](../figures/{figs['p10'].name})

![prA](../figures/{figs['pr_A'].name}) ![prB](../figures/{figs['pr_B'].name})
"""


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    runs_doc = json.loads(ex.RUNS_JSON.read_text(encoding="utf-8"))
    runs, sha = runs_doc["runs"], runs_doc["params_sha"]
    params = json.loads(ex.PARAMS.read_text(encoding="utf-8"))
    experiments = pd.read_csv(tracking.EXPERIMENTS) if tracking.EXPERIMENTS.exists() else pd.DataFrame()

    # reproducibility: refit A/FULL and compare
    table = ex.load_table()
    raw = pd.read_parquet(ex.TABLE)
    tr, te = ex.folds_for("A", table)[0]
    b = model.fit(table.loc[tr, fs.FEATURE_SETS["FULL"]], table.loc[tr, "is_slow"], params["best_params"])
    auc_refit = metrics.auc_pr(table.loc[te, "is_slow"].astype(int), model.predict(b, table.loc[te, fs.FEATURE_SETS["FULL"]]))
    auc_saved = next(r["auc_pr"] for r in runs if r["scenario"] == "A" and r["featureset"] == "FULL")
    checks = gate_checks(runs, sha, experiments, auc_refit - auc_saved)
    GATE_JSON.write_text(json.dumps(checks, indent=2, default=str), encoding="utf-8")

    summary = summarise(runs)
    slices = {}
    for sc in ("A", "B"):
        paths = [Path(r["pred_path"]) for r in runs if r["scenario"] == sc and r["featureset"] == "FULL"]
        slices[sc] = bias_slices(pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True))
    figs = {"p10": fig_p10(summary),
            "pr_A": fig_pr("A", [Path(r["pred_path"]) for r in runs if r["scenario"] == "A" and r["featureset"] == "FULL"]),
            "pr_B": fig_pr("B", [Path(r["pred_path"]) for r in runs if r["scenario"] == "B" and r["featureset"] == "FULL"])}
    DOC.write_text(render(params, sha, summary, runs, slices, checks, figs, len(table), int(len(raw) - len(table))), encoding="utf-8")
    verdict = all(c["pass"] for c in checks)
    print(f"wrote {DOC}  gate={'PASS' if verdict else 'FAIL'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify they pass; full suite; commit**

Run: `python -m pytest tests/test_report4.py -q` → `7 passed`; `python -m pytest tests -q` → all pass, no warnings.
```bash
git add report4.py tests/test_report4.py
git commit -m "feat(phase4): results report, bias slices, figures, validity gate"
```

---

### Task 6: Real-data run — tune, 24 experiments, report, gate (CONDITIONAL)

**Precondition:** Tasks 1–5 reviewed; `data/phase3_gate.json` shows all five checks passing; `data/features/features.parquet` exists.

- [ ] **Step 1: Tune** — `time python tune.py --trials 100` (expect ~10–20 min; LightGBM on ~24k rows at 1 thread). Report `cv_auc_pr`, best params, wall-clock. Commit `data/models/params.json`? No — `data/models/` is gitignored; copy it to `data/phase4_params.json` and commit that (a 100-trial record is small).

- [ ] **Step 2: Experiments** — `time python experiment.py` → 24 runs, `data/phase4_runs.json`, 24 new `lgbm` rows in `data/experiments.csv`. Report the per-run log lines.

- [ ] **Step 3: Report + gate** — `python report4.py; echo exit=$?` → `docs/phase4_results.md`, three figures, `data/phase4_gate.json`. All five gate checks must PASS; if #5 (reproducibility) fails, stop and report the delta — nothing gets quoted until it's deterministic.

- [ ] **Step 4: Commit the records** — `git add data/phase4_params.json data/phase4_runs.json data/phase4_gate.json data/experiments.csv docs/phase4_results.md figures/phase4_*.png` and commit `"data: Phase 4 results and gate"`.

- [ ] **Step 5: Report the headline verbatim** from `docs/phase4_results.md §4`, the ablation table, and the first-timer bias line — whatever they say.

---

## Verification (spec §10–§11)

1. `python -m pytest tests -q` — feature-set hygiene, model determinism, runner end-to-end on the synthetic table with planted signal, tuning reproducibility, gate failure modes each caught by exactly their check.
2. Real data: gate 1–5 PASS in `data/phase4_gate.json`.
3. `docs/phase4_results.md §4` states the FULL-vs-baseline comparison with CI, either way; §5 has the NO_LABEL_REPLAY row on both scenarios.

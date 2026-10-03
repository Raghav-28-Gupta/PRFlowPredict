# Phase 6b — Repo-Fingerprinting Test — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the pre-registered test of whether the NO_LABEL_REPLAY model's cold-start failure is *repo fingerprinting*: the model identifying repos by their static attributes and replaying their base rates. The test combines a SHAP transfer test and one interventional ablation, and its verdict is fixed before any of it runs.

**Architecture:** `fingerprint.py` holds the pure statistics: the repo-feature definition, per-repo contributions, the paired repo bootstrap and the nine-cell verdict. `report6b.py` trains the one new ablation through Phase 4's own `experiment.run`, explains the six existing NLR boosters with Phase 6's `attribution.explain`, and writes a fully generated document plus a validity gate. It also makes two small, backward-compatible edits to Phase 4 code, so the new ablation can be trained without changing what a Phase 4 re-run does.

**Tech Stack:** Python 3.13, lightgbm 4.7.0, shap 0.52.0, scipy 1.17.0 (`spearmanr`), scikit-learn (`average_precision_score` via `metrics.auc_pr`), pandas, numpy, pytest.

**Spec:** `docs/design/specs/2026-09-24-phase6b-fingerprinting-design.md`. Committed at `4a5426f` before any code; that commit is the pre-registration.

**This plan's code was dry-run before it was committed.** Every code block below was extracted, assembled into the files exactly as the tasks describe, and run in a scratch copy against the real repo modules:
- **98 tests passed** across the four touched test files, with pristine output.
- **All 24 mandated mutation checks were caught.** Each target test passed unmutated (pytest exit 0) and failed once mutated (exit 1).

So if a test fails for you as written, suspect a transcription slip or an environment difference before suspecting the plan, and say which in your report. Two tests were rewritten *because* the dry run showed their mutation survived: `test_bootstrap_draws_are_paired_across_scenarios` and `test_verdict_text_says_the_right_thing_for_its_cell`. Don't simplify either back.

**What the dry run could not see, and what it cost.** The dry run imported a scratch copy of `experiment.py`, so `experiment.run`'s default output directories resolved to the scratch folder. In the real repo they are Phase 4's own `data/models/` and `data/predictions/`. Task 4's mutation #5 drops `refit_delta`'s explicit output directories, and its spy test delegates to the real `run()`. So when the mutation check ran for real, it **overwrote Phase 4's `A_NO_LABEL_REPLAY_fold0` booster and predictions with synthetic data**. Nothing reported it, because those directories are gitignored and invisible to `git status`. The artifact was restored and verified bit-exact against `data/phase4_runs.json`. Task 4's test file now carries an autouse fixture, `_phase4_directories_are_off_limits`, which redirects `run()`'s keyword defaults to `tmp_path` for every test in the file. That makes it structurally impossible for any test there, mutated or not, to reach Phase 4's directories. It was verified against the **real** `experiment` module, with both artifacts hashed and backed up first: under mutation #5, exactly the spy test fails and both files' SHA-256 are unchanged. **Do not remove that fixture.**

## Global Constraints

- Seed everywhere: `20260912` (`splits.SEED`). Bootstrap: **2,000** draws, `np.random.default_rng(SEED)`, percentile **95%** intervals.
- `REPO_FEATURES` = the 8 columns with `group == 'repo'` **and** `status == 'snapshot'` in `features.COLUMN_SPEC`: `n_assignable_users, n_mentionable_users, owner_is_org, has_codeowners, has_pr_template, has_contributing, n_ci_workflows, language_dominant`. `repo_age_days_at_open` is `group == 'repo'` but varies within a repo, and is **excluded**.
- The intervention's feature set is named `NLR_NO_REPO` = `FEATURE_SETS["NO_LABEL_REPLAY"]` (33) minus `REPO_FEATURES` = **25** columns, in NLR's order. It is **never** added to `fs.FEATURE_SETS`.
- Tuned params: `json.loads(experiment.PARAMS.read_text())["best_params"]` (`data/models/params.json`, semantically equal to the committed `data/phase4_params.json`).
- Phase 4's saved NLR AUC-PR: **A 0.901745**; **B folds 0.964792 / 0.834803 / 0.507069 / 0.643918 / 0.866883**, mean 0.763493.
- AUC-PR convention: Scenario A is one fold; **Scenario B is the mean of its per-fold AUC-PRs**. A fold lacking both classes is skipped.
- Additivity (raw margin, max \|Δ\| < **1e-6**) and the refit (\|Δ\| < **1e-6**) are **HARD STOPS**: no verdict is written if either fails.
- **Never** write to `data/phase4_runs.json`. **Never** overwrite a Phase 4 booster or prediction. The refit goes into a `tempfile.TemporaryDirectory`.
- **No test may be able to reach `data/models/` or `data/predictions/`, even while a mandated mutation is applied.** Those directories are gitignored, so `git status` cannot detect damage to them. After any mutation check, verify real artifacts by hash or mtime, never by `git diff`.
- `docs/phase6b_fingerprinting.md` is **fully generated**, with no hand-written section.
- Tests in `tests/`, run with `python -m pytest tests -q`, and **output pristine**. **147** pass today.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Every test must fail when the behaviour it names is broken.** Six tests in Phase 6 held whether or not their behaviour worked. Each task below names the mutation its key tests must catch; before committing, apply that mutation, confirm the test fails, restore the code, and confirm it passes.
- Real-data execution happens only in Task 6.

---

## File structure

| File | Responsibility |
|---|---|
| `experiment.py` (modify `run`) | accept `cols=None` so a set can be trained without registering it |
| `report4.py` (modify gate check 4) | count only Phase 4's own feature sets in the shared `experiments.csv` |
| `fingerprint.py` (create) | pure statistics: definition, per-repo contribution, bootstrap, verdict |
| `report6b.py` (create) | train the ablation, explain the NLR boosters, gate, document |
| `tests/test_experiment.py`, `tests/test_report4.py` (modify) | cover the two Phase 4 edits |
| `tests/test_fingerprint.py`, `tests/test_report6b.py` (create) | |
| outputs: `data/phase6b_{transfer,intervention,reliance}.csv`, `data/phase6b_runs.json`, `data/phase6b_gate.json`, `docs/phase6b_fingerprinting.md`; 6 rows appended to `data/experiments.csv` | |

---

### Task 1: Two backward-compatible Phase 4 edits

**Files:**
- Modify: `experiment.py` (the `run` signature and its first line)
- Modify: `report4.py:73` (the `logged = ...` line in `gate_checks`)
- Modify: `tests/test_experiment.py` (imports + 3 tests)
- Modify: `tests/test_report4.py` (the `_experiments` helper + 1 test)

**Interfaces:**
- Produces: `experiment.run(scenario, fold, name, table, params, train_idx, test_idx, *, seed=SEED, out_models=MODELS_DIR, out_preds=PRED_DIR, cols: list[str] | None = None) -> dict`. With `cols=None` the columns are `fs.FEATURE_SETS[name]`, exactly as today; otherwise they are `list(cols)`, and `name` is only a label for the output tag `{scenario}_{name}_fold{fold}`.

**Why these edits exist:**
1. `experiment.main()` loops `for name in fs.FEATURE_SETS` and writes every result to `data/phase4_runs.json`. Registering `NLR_NO_REPO` there would silently change what a Phase 4 re-run trains, and break Phase 4's gate check 4 (`len(runs) == 24`). A `cols=` parameter trains the set without registering it.
2. Phase 4's gate check 4 counts **every** `lgbm` row in `data/experiments.csv` under the params sha and requires exactly 24. But that log is shared across phases by design (blueprint §5: "a single `experiments.csv`"). Once this phase logs its 6 runs under the same sha, a Phase 4 re-run would count 30 and fail. The fix counts only rows whose `features` is a Phase 4 set.
3. **The test helper must change too.** `tests/test_report4.py::_experiments` builds its frame with **no `features` column**, so after edit 2 every check-4 test would raise `KeyError`. The real CSV always has that column; the helper must mirror it.

- [ ] **Step 1: Write the failing tests**

In `tests/test_experiment.py`, add `import pytest` and `import featuresets as fs` to the import block, then append:

```python
def test_run_cols_none_is_unchanged_behaviour(synthetic_table, tmp_path):
    """Passing the registered set explicitly under a NEW name must reproduce the name lookup
    exactly. The new name matters: with name="FULL", a cols argument that was silently
    ignored would still pass."""
    t = ex.load_table_frame(synthetic_table())
    tr, te = ex.folds_for("A", t)[0]
    ref = ex.run("A", 0, "FULL", t, model.DEFAULT_PARAMS, tr, te,
                 out_models=tmp_path / "m1", out_preds=tmp_path / "p1")
    new = ex.run("A", 0, "FULL_EXPLICIT", t, model.DEFAULT_PARAMS, tr, te,
                 out_models=tmp_path / "m2", out_preds=tmp_path / "p2", cols=fs.FEATURE_SETS["FULL"])
    a = pd.read_parquet(ref["pred_path"])["p_hat"].to_numpy()
    b = pd.read_parquet(new["pred_path"])["p_hat"].to_numpy()
    assert np.array_equal(a, b)
    assert ref["auc_pr"] == new["auc_pr"]
    assert (tmp_path / "m2" / "A_FULL_EXPLICIT_fold0.txt").exists()


def test_run_cols_is_actually_used(synthetic_table, tmp_path):
    """Dropping the planted signal column must change the predictions. This proves the
    override reaches the model rather than being accepted and discarded."""
    t = ex.load_table_frame(synthetic_table())
    tr, te = ex.folds_for("A", t)[0]
    full = ex.run("A", 0, "FULL", t, model.DEFAULT_PARAMS, tr, te,
                  out_models=tmp_path / "m1", out_preds=tmp_path / "p1")
    fewer = [c for c in fs.FEATURE_SETS["FULL"] if c != "body_len"]
    cut = ex.run("A", 0, "FULL_MINUS_BODY", t, model.DEFAULT_PARAMS, tr, te,
                 out_models=tmp_path / "m2", out_preds=tmp_path / "p2", cols=fewer)
    a = pd.read_parquet(full["pred_path"])["p_hat"].to_numpy()
    b = pd.read_parquet(cut["pred_path"])["p_hat"].to_numpy()
    assert not np.array_equal(a, b)


def test_run_cols_still_enforces_hygiene(synthetic_table, tmp_path):
    t = ex.load_table_frame(synthetic_table())
    tr, te = ex.folds_for("A", t)[0]
    with pytest.raises(ValueError):
        ex.run("A", 0, "LEAKY", t, model.DEFAULT_PARAMS, tr, te,
               out_models=tmp_path / "m", out_preds=tmp_path / "p", cols=["body_len", "is_slow"])
```

In `tests/test_report4.py`, replace the `_experiments` helper with:

```python
PHASE4_SETS = ("FULL", "NO_SNAPSHOT", "NO_LABEL_REPLAY", "PR_ONLY")


def _experiments(sha, n=24):
    # Mirrors the real data/experiments.csv schema, which always carries `features`.
    return pd.DataFrame({"model": ["lgbm"] * n, "params": [sha] * n,
                         "features": [PHASE4_SETS[i % 4] for i in range(n)],
                         "p10_ci_lo": [0.5] * n, "p10_ci_hi": [0.7] * n})
```

and append:

```python
def test_gate_check4_ignores_rows_a_later_phase_logs():
    """experiments.csv is shared across phases (blueprint §5). Six lgbm rows a later phase
    logs under the SAME params sha must not count toward Phase 4's 24."""
    later = _experiments("s", n=6)
    later["features"] = "NLR_NO_REPO"
    both = pd.concat([_experiments("s"), later], ignore_index=True)      # 30 lgbm rows, one sha
    assert report4.gate_checks(_all_runs(), "s", both, 0.0)[3]["pass"] is True
    # and a genuinely short Phase 4 log still fails, even with the later rows present
    short = pd.concat([_experiments("s", n=23), later], ignore_index=True)
    assert report4.gate_checks(_all_runs(), "s", short, 0.0)[3]["pass"] is False
```

- [ ] **Step 2: Run them and confirm the right failures**

Run: `python -m pytest tests/test_experiment.py tests/test_report4.py -q`
Expected: the three `test_run_cols_*` tests FAIL with `TypeError: run() got an unexpected keyword argument 'cols'`, and `test_gate_check4_ignores_rows_a_later_phase_logs` FAILS on its first assertion (30 ≠ 24). All pre-existing tests still PASS, because the helper change alone is harmless before the filter exists.

- [ ] **Step 3: Implement both edits**

In `experiment.py`, change the `run` signature and its first line:

```python
def run(scenario: str, fold: int, name: str, table: pd.DataFrame, params: dict,
        train_idx: np.ndarray, test_idx: np.ndarray, *, seed: int = SEED,
        out_models: Path = MODELS_DIR, out_preds: Path = PRED_DIR,
        cols: list[str] | None = None) -> dict:
    # `cols` lets a later phase train a feature set WITHOUT registering it in
    # fs.FEATURE_SETS: main() trains every registered set and writes them all to
    # phase4_runs.json, so registering one would silently change a Phase 4 re-run.
    cols = fs.FEATURE_SETS[name] if cols is None else list(cols)
    fs.assert_hygiene(cols)
```

Leave the rest of `run` untouched.

In `report4.py`, replace line 73 (`logged = int(((experiments["model"] == "lgbm") & (experiments["params"] == params_sha)).sum()) if len(experiments) else 0`) with:

```python
    # experiments.csv is shared across phases (blueprint §5), so count only Phase 4's own
    # feature sets: a later phase logging lgbm rows under the same params sha must not
    # break this check.
    logged = int(((experiments["model"] == "lgbm") & (experiments["params"] == params_sha)
                  & experiments["features"].isin(list(fs.FEATURE_SETS))).sum()) if len(experiments) else 0
```

Keep it a single conditional expression. The `if len(experiments)` guard must still short-circuit before any column is read.

- [ ] **Step 4: Run to verify, then run the mutation check**

Run: `python -m pytest tests/test_experiment.py tests/test_report4.py -q`. Expected: all pass.

Mutation check (the bar for this task):
- Revert `report4.py:73` to the old line → `test_gate_check4_ignores_rows_a_later_phase_logs` must FAIL. Restore it.
- In `experiment.run`, replace the new line with `cols = fs.FEATURE_SETS[name]` (ignoring the argument) → `test_run_cols_none_is_unchanged_behaviour` must FAIL with a `KeyError` on `"FULL_EXPLICIT"`. Restore it.

Then check the real log, with **no test file and no writes**:
`python -c "import pandas as pd, featuresets as fs, experiment as ex; e = pd.read_csv('data/experiments.csv'); print(int(((e.model=='lgbm') & (e.params==ex.params_sha()) & e.features.isin(list(fs.FEATURE_SETS))).sum()))"`
Expected: `24`.

Run the full suite: `python -m pytest tests -q` → **151 passed**, pristine.

- [ ] **Step 5: Commit**

```bash
git add experiment.py report4.py tests/test_experiment.py tests/test_report4.py
git commit -m "feat(phase6b): train a feature set without registering it; Phase 4 check 4 counts only its own sets

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `fingerprint.py` — the definition and the statistics

**Files:**
- Create: `fingerprint.py`, `tests/test_fingerprint.py`

**Interfaces:**
- Consumes: `features.COLUMN_SPEC` (dict of column → metadata with keys `group`, `status`, …), `featuresets.FEATURE_SETS`, `splits.SEED`.
- Produces:
  - `fingerprint.SEED`, `fingerprint.NO_REPO_NAME = "NLR_NO_REPO"`, `fingerprint.RELIANCE_BAND = 0.05`
  - `fingerprint.REPO_FEATURES: tuple[str, ...]`, the 8 columns in COLUMN_SPEC order
  - `fingerprint.nlr_no_repo_cols() -> list[str]`, 25 columns in NLR's order
  - `fingerprint.non_constant_within_repo(table: pd.DataFrame, feats, repo_col="repo") -> list[str]`
  - `fingerprint.repo_contribution(shap_values: np.ndarray, cols: list[str]) -> np.ndarray`, the per-row sum of REPO_FEATURES' SHAP values
  - `fingerprint.per_repo(rows: pd.DataFrame) -> pd.DataFrame`, indexed by `repo`, with columns exactly `["c", "y", "n"]` (mean `repo_contrib`, mean `is_slow`, row count)
  - `fingerprint.spearman(x, y) -> float`
  - `fingerprint.outcome(lo: float, hi: float) -> str`, one of `"confirms"`, `"contradicts"`, `"inconclusive"`; raises `ValueError` on a non-finite bound
  - `fingerprint.VERDICTS: dict[tuple[str, str], str]` and `fingerprint.verdict(t: str, i: str) -> str`
  - `fingerprint.reliance_band(d: float) -> str`, one of `"higher"`, `"lower"`, `"unchanged"`

**A deliberate departure from the spec's module map, not an omission.** The spec's §8 names `per_repo_contribution()` and `transfer_rho()`. This plan splits the first in two: `repo_contribution` works per row, which is what `score_rows` needs in Task 4, and `per_repo` works per repo, which is what the bootstrap needs. The bootstrap then computes ρ directly with `spearman` on `per_repo`'s columns. A separate `transfer_rho` would be a one-line alias with no caller, so it doesn't exist. Every behaviour the spec's §9 asks of those functions is tested below.

- [ ] **Step 1: Write the failing tests**

`tests/test_fingerprint.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_fingerprint.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'fingerprint'`.

- [ ] **Step 3: Implement `fingerprint.py`**

```python
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
    return pd.DataFrame({"c": g["repo_contrib"].mean(),
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
```

- [ ] **Step 4: Run to verify, then run the mutation check**

Run: `python -m pytest tests/test_fingerprint.py -q` → **28 passed**.

Mutation check. Each of these must make the named test FAIL; restore after each:
- `REPO_FEATURES` selects `group == 'repo'` only (drop the `status` clause) → `test_repo_features_are_exactly_the_eight_snapshot_repo_columns`.
- `per_repo` uses `.sum()` for `c` → `test_per_repo_is_a_row_mean_keyed_by_repo_label`.
- `repo_contribution` uses `idx[:-1]` → `test_repo_contribution_sums_only_the_repo_feature_columns_per_row`.
- `outcome` uses `lo >= 0` → `test_outcome[0.0-0.3-inconclusive]`.
- swap the two `PARTIAL_*` values in `VERDICTS` → `test_verdict_matches_the_spec_table_cell_by_cell`.

Then run `python -m pytest tests -q` → **179 passed**, pristine.

- [ ] **Step 5: Commit**

```bash
git add fingerprint.py tests/test_fingerprint.py
git commit -m "feat(phase6b): repo-feature definition, per-repo contribution, pre-registered verdict

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `fingerprint.py` — Phase 4's AUC-PR convention and the paired repo bootstrap

**Files:**
- Modify: `fingerprint.py` (add `import metrics` to the import block; append the functions below)
- Modify: `tests/test_fingerprint.py` (append)

**Interfaces:**
- Consumes (from Task 2): `fp.per_repo`, `fp.spearman`, `fp.SEED`. Also `metrics.auc_pr(y_true, score) -> float`.
- Produces:
  - `fingerprint.ROW_COLS = ("repo", "fold", "is_slow", "p_nlr", "p_no_repo", "repo_contrib")`, the scored-rows schema Task 4 builds
  - `fingerprint.mean_fold_auc(y, p, fold) -> float`
  - `fingerprint.auc_delta(y, p_new, p_old, fold) -> float` = `mean_fold_auc(p_new) − mean_fold_auc(p_old)`
  - `fingerprint.paired_repo_bootstrap(rows_a, rows_b, *, n=2000, seed=SEED, ci=0.95) -> dict` with keys exactly: `n_repos, rho_a, rho_b, T, T_lo, T_hi, delta_a, delta_b, I, I_lo, I_hi, n_draws, dropped_T, dropped_I`

**The design, which the code must follow (spec §5.4):** there is **one** set of repo draws, applied to both scenarios and both statistics. Per-repo quantities (`c_r`, `y_r`) do not change under resampling, so T resamples the per-repo **pairs** with multiplicity. AUC-PR is not decomposable per repo, so I resamples the drawn repos' **rows** with multiplicity. Point estimates go through the **same** code path, applied to the identity draw (every repo once).

The test fixtures are deterministic by construction. Each repo has an exact count of slow PRs, so its slow rate is fixed and strictly increasing. `PERM` is a permutation of 12 whose Spearman correlation with `range(12)` is **exactly 0.0** (verified). Planted effects and planted nulls therefore cannot pass or fail by a lucky draw.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fingerprint.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_fingerprint.py -q`
Expected: the 10 new tests FAIL with `AttributeError: module 'fingerprint' has no attribute 'mean_fold_auc'` (or `paired_repo_bootstrap`). The 28 from Task 2 still pass.

- [ ] **Step 3: Implement**

Add `import metrics` to `fingerprint.py`'s import block (alphabetically, after `import featuresets as fs`), then append:

```python
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
```

- [ ] **Step 4: Run to verify, then run the mutation check**

Run: `python -m pytest tests/test_fingerprint.py -q` → **38 passed**, in a few seconds.

Mutation check. Each must make the named test FAIL; restore after each:
- `mean_fold_auc` returns `metrics.auc_pr(y, p)` (pooled) → `test_mean_fold_auc_is_the_mean_of_per_fold_values_not_pooled`.
- drop the `0 < y[m].sum() < m.sum()` condition → `test_mean_fold_auc_skips_a_single_class_fold`.
- give A and B **independent** draws: in the loop, draw `d_a` and `d_b` separately and take A's terms from `components(d_a)` and B's from `components(d_b)` → `test_bootstrap_draws_are_paired_across_scenarios` (T and I stop being exactly 0).
- compute the point estimates on a bootstrap draw instead of the identity draw (replace `components(np.arange(len(repos)))` with `components(np.random.default_rng(seed).choice(len(repos), size=len(repos), replace=True))`) → `test_bootstrap_point_estimates_match_a_direct_computation`.
- swap the sign of T (`rb - ra`) → `test_bootstrap_confirms_a_planted_fingerprint`.

Then run `python -m pytest tests -q` → **189 passed**, pristine.

- [ ] **Step 5: Commit**

```bash
git add fingerprint.py tests/test_fingerprint.py
git commit -m "feat(phase6b): Phase 4's AUC-PR convention and the paired repo bootstrap

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `report6b.py` — the data layer

**Files:**
- Create: `report6b.py` (module docstring, imports, the six functions below)
- Create: `tests/test_report6b.py` (the data-layer tests)

**Interfaces:**
- Consumes: `experiment.run(..., cols=...)` (Task 1), `experiment.folds_for`, `experiment.SCENARIOS`, `experiment.MODELS_DIR`, `experiment.PRED_DIR`; `fingerprint.REPO_FEATURES`, `NO_REPO_NAME`, `nlr_no_repo_cols`, `repo_contribution` (Task 2); `attribution.prepare`, `attribution.explain(booster, X) -> (sv, ev)`, `attribution.additivity_delta(booster, X, sv, ev) -> float`, `attribution.importance(sv, cols) -> DataFrame[feature, mean_abs_shap, share]`; `model.load`, `metrics.auc_pr`, `tracking.log`.
- Produces:
  - `report6b.train_no_repo(table, params, sha, *, out_models=ex.MODELS_DIR, out_preds=ex.PRED_DIR) -> list[dict]`: 6 fits, each logged to `data/experiments.csv`
  - `report6b.refit_delta(table, params, saved_auc: float) -> float`
  - `report6b.score_rows(scenario, table, nlr_runs, no_repo_runs) -> tuple[pd.DataFrame, np.ndarray, dict[str, float], dict[str, bool]]`: rows in `fp.ROW_COLS` plus `pr_id`; the pooled SHAP matrix; additivity per tag; same-rows per tag. Tags are `"{scenario}_fold{k}"`.
  - `report6b.reliance(sv_a, sv_b) -> pd.DataFrame` with columns `feature, share_a, share_b, is_repo_feature`, sorted by `share_b` descending
  - `report6b.coverage(nlr_runs, rows_a, rows_b) -> dict` with keys exactly `n_repos, same_repo_set, b_each_once`
  - `report6b.fold_table(rows: dict[str, pd.DataFrame]) -> pd.DataFrame` with columns `scenario, fold, n_test, auc_pr_nlr, auc_pr_no_repo, delta`

**Two things that are easy to get wrong:**
- **`score_rows` must join the two models' predictions on `pr_id`, never by position.** A misaligned join silently pairs each NLR score with a different PR's intervention score and corrupts I. The fixture writes the intervention's file in **reversed** row order to catch this.
- **`refit_delta` must pass `out_models` and `out_preds` pointing into a `TemporaryDirectory`.** The defaults are Phase 4's directories, and `run` would overwrite `data/models/A_NO_LABEL_REPLAY_fold0.txt`. It must also pass `cols=`, because gate check 2 exists to prove that exact code path.

- [ ] **Step 1: Write the failing tests**

`tests/test_report6b.py`:

```python
import numpy as np
import pandas as pd
import pytest

import experiment as ex
import featuresets as fs
import fingerprint as fp
import model
import report6b as rb


@pytest.fixture(autouse=True)
def _phase4_directories_are_off_limits(monkeypatch, tmp_path):
    """No test in this file may write to Phase 4's real data/models or data/predictions.

    experiment.run's output directories are keyword-only DEFAULTS, so any code path that
    forgets to pass them -- including a deliberately mutated refit_delta during a mutation
    check -- would write straight into Phase 4's real artifacts. That happened once. This
    redirects the defaults to tmp_path for every test in the file; monkeypatch restores them."""
    monkeypatch.setattr(ex.run, "__kwdefaults__",
                        {**ex.run.__kwdefaults__,
                         "out_models": tmp_path / "phase4_models_redirected",
                         "out_preds": tmp_path / "phase4_preds_redirected"})


# ---------------------------------------------------------------------------
# score_rows: the join and the selection, on a real booster in tmp_path
# ---------------------------------------------------------------------------

def _nlr_fixture(synthetic_table, tmp_path, drop_one=False):
    """Two B folds (one repo each), plus a Scenario A decoy on both sides. The NLR_NO_REPO
    files list the same rows in REVERSED order with row-identifying scores, so a positional
    join would pair the wrong rows."""
    t = ex.load_table_frame(synthetic_table())
    cols = fs.FEATURE_SETS["NO_LABEL_REPLAY"]
    mp = tmp_path / "nlr.txt"
    model.save(model.fit(t[cols], t["is_slow"], model.DEFAULT_PARAMS), mp)
    nlr, new, truth = [], [], {}
    for k, repo in enumerate(["o/r", "o/s"]):
        te = t[t["repo"] == repo].head(60)
        old_p, new_p = tmp_path / f"nlr_{k}.parquet", tmp_path / f"new_{k}.parquet"
        pd.DataFrame({"pr_id": te["pr_id"], "repo": te["repo"], "is_slow": te["is_slow"],
                      "p_hat": np.linspace(0, 1, len(te))}).to_parquet(old_p, index=False)
        ids = te["pr_id"].to_numpy()[::-1]
        if drop_one and k == 0:
            ids = ids[1:]
        scores = 1000.0 * (k + 1) + np.arange(len(ids), dtype=float)
        pd.DataFrame({"pr_id": ids, "p_hat": scores}).to_parquet(new_p, index=False)
        truth.update(dict(zip(ids, scores)))
        nlr.append({"scenario": "B", "fold": k, "model_path": str(mp), "pred_path": str(old_p)})
        new.append({"scenario": "B", "fold": k, "pred_path": str(new_p)})
    nlr.append({"scenario": "A", "fold": 0, "model_path": str(mp), "pred_path": str(tmp_path / "nlr_0.parquet")})
    new.append({"scenario": "A", "fold": 0, "pred_path": str(tmp_path / "new_0.parquet")})
    return t, nlr, new, truth


def test_score_rows_joins_on_pr_id_and_filters_the_scenario(synthetic_table, tmp_path):
    t, nlr, new, truth = _nlr_fixture(synthetic_table, tmp_path)
    rows, sv, additivity, same = rb.score_rows("B", t, nlr, new)
    assert len(rows) == 120                                      # the A decoy added nothing
    assert set(fp.ROW_COLS) <= set(rows.columns)
    assert sorted(rows["fold"].unique()) == [0, 1]
    assert np.array_equal(rows["p_no_repo"].to_numpy(), rows["pr_id"].map(truth).to_numpy())
    assert same == {"B_fold0": True, "B_fold1": True}
    assert set(additivity) == {"B_fold0", "B_fold1"} and max(additivity.values()) < 1e-6
    assert sv.shape == (120, 33)
    assert np.allclose(rows["repo_contrib"], fp.repo_contribution(sv, fs.FEATURE_SETS["NO_LABEL_REPLAY"]))


def test_score_rows_reports_mismatched_test_rows_instead_of_crashing(synthetic_table, tmp_path):
    t, nlr, new, _ = _nlr_fixture(synthetic_table, tmp_path, drop_one=True)
    _, _, _, same = rb.score_rows("B", t, nlr, new)
    assert same == {"B_fold0": False, "B_fold1": True}


# ---------------------------------------------------------------------------
# reliance, coverage, fold_table
# ---------------------------------------------------------------------------

NLR = fs.FEATURE_SETS["NO_LABEL_REPLAY"]


def test_reliance_repo_shares():
    sv_a, sv_b = np.zeros((4, len(NLR))), np.zeros((4, len(NLR)))
    repo_i, other_i = NLR.index(fp.REPO_FEATURES[0]), NLR.index("body_len")
    sv_a[:, repo_i], sv_a[:, other_i] = 3.0, 1.0          # A: the repo feature carries 3 of 4
    sv_b[:, repo_i], sv_b[:, other_i] = 1.0, 3.0          # B: it carries 1 of 4
    rel = rb.reliance(sv_a, sv_b)
    assert list(rel.columns) == ["feature", "share_a", "share_b", "is_repo_feature"]
    assert rel.loc[rel.is_repo_feature, "share_a"].sum() == pytest.approx(0.75)
    assert rel.loc[rel.is_repo_feature, "share_b"].sum() == pytest.approx(0.25)


def test_reliance_is_sorted_by_b_share():
    """Three features whose B order (f3, f1, f2) differs from their positional order, their
    A-importance order and their lexicographic order -- so an unsorted result fails however
    it happens to be ordered."""
    f1, f2, f3 = "body_len", "open_backlog_at_t", "prior_merge_rate_here"
    sv_a, sv_b = np.zeros((2, len(NLR))), np.zeros((2, len(NLR)))
    for f, a, b in ((f1, 3.0, 2.0), (f2, 2.0, 1.0), (f3, 1.0, 3.0)):
        sv_a[:, NLR.index(f)], sv_b[:, NLR.index(f)] = a, b
    rel = rb.reliance(sv_a, sv_b)
    assert rel["feature"].head(3).tolist() == [f3, f1, f2]


def _cov(twice=False):
    runs = [{"scenario": "A", "test_repos": ["r0", "r1", "r2"]},
            {"scenario": "B", "test_repos": ["r0", "r1"]},
            {"scenario": "B", "test_repos": ["r2", "r0"] if twice else ["r2"]}]
    rows = pd.DataFrame({"repo": ["r0", "r1", "r2"]})
    return runs, rows, rows.copy()


def test_coverage_accepts_the_paired_design():
    assert rb.coverage(*_cov()) == {"n_repos": 3, "same_repo_set": True, "b_each_once": True}


def test_coverage_flags_a_repo_held_out_twice():
    assert rb.coverage(*_cov(twice=True))["b_each_once"] is False


def test_fold_table_scores_each_fold_on_its_own_rows():
    rows = pd.DataFrame({"fold": [0, 0, 1, 1], "is_slow": [0, 1, 0, 1],
                         "p_nlr": [0.2, 0.9, 0.9, 0.2], "p_no_repo": [0.9, 0.2, 0.2, 0.9]})
    ft = rb.fold_table({"B": rows}).set_index("fold")
    assert ft.loc[0, "auc_pr_nlr"] == pytest.approx(1.0)
    assert ft.loc[1, "auc_pr_nlr"] == pytest.approx(0.5)
    assert ft.loc[0, "delta"] == pytest.approx(ft.loc[0, "auc_pr_no_repo"] - ft.loc[0, "auc_pr_nlr"])
    assert ft["n_test"].tolist() == [2, 2]


# ---------------------------------------------------------------------------
# refit_delta: the behaviour-preserving refit (gate check 2)
# ---------------------------------------------------------------------------

def test_refit_delta_measures_against_the_saved_value(synthetic_table, tmp_path):
    t = ex.load_table_frame(synthetic_table())
    tr, te = ex.folds_for("A", t)[0]
    ref = ex.run("A", 0, "NO_LABEL_REPLAY", t, model.DEFAULT_PARAMS, tr, te,
                 out_models=tmp_path / "m", out_preds=tmp_path / "p")
    assert rb.refit_delta(t, model.DEFAULT_PARAMS, ref["auc_pr"]) == 0.0
    assert rb.refit_delta(t, model.DEFAULT_PARAMS, ref["auc_pr"] + 0.25) == pytest.approx(-0.25)


def test_refit_delta_never_writes_to_phase4s_directories(synthetic_table, monkeypatch):
    seen = {}
    real = ex.run

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(ex, "run", spy)
    rb.refit_delta(ex.load_table_frame(synthetic_table()), model.DEFAULT_PARAMS, 0.5)
    assert seen["out_models"] != ex.MODELS_DIR and seen["out_preds"] != ex.PRED_DIR
    assert seen["cols"] == fs.FEATURE_SETS["NO_LABEL_REPLAY"]       # through the NEW code path
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_report6b.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'report6b'`.

- [ ] **Step 3: Implement the data layer**

`report6b.py`:

```python
"""Phase 6b report: the pre-registered repo-fingerprinting test.

Tests whether the NO_LABEL_REPLAY model's cold-start failure is repo fingerprinting -- the
model identifying repos by their static attributes and replaying their base rates.

Pre-registered in docs/design/specs/2026-09-24-phase6b-fingerprinting-design.md,
committed before any of this ran. The document this writes is FULLY GENERATED: there is no
hand-written section, because Phase 6's hand-written section 4 showed what one costs inside
a generated document -- a re-run silently destroys it.

Re-running is safe but not free: it re-trains the six NLR_NO_REPO boosters (deterministic,
so identical) and appends their six rows to data/experiments.csv again. Phase 4's gate
check 4 counts only Phase 4's own feature sets, so that does not disturb it."""
from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import attribution as attr
import experiment as ex
import featuresets as fs
import fingerprint as fp
import metrics
import model
import tracking


def train_no_repo(table: pd.DataFrame, params: dict, sha: str, *,
                  out_models: Path = ex.MODELS_DIR, out_preds: Path = ex.PRED_DIR) -> list[dict]:
    """The intervention (spec section 5.3): NLR_NO_REPO on Phase 4's identical folds and
    tuned params, one fit for A and five for B, each logged to data/experiments.csv."""
    cols = fp.nlr_no_repo_cols()
    runs = []
    for scenario in ex.SCENARIOS:
        for k, (tr, te) in enumerate(ex.folds_for(scenario, table)):
            res = ex.run(scenario, k, fp.NO_REPO_NAME, table, params, tr, te,
                         out_models=out_models, out_preds=out_preds, cols=cols)
            runs.append(res)
            tracking.log({
                "scenario": scenario, "fold": k, "model": "lgbm", "features": fp.NO_REPO_NAME,
                "params": sha, "n_train": res["n_train"], "n_test": res["n_test"],
                "n_test_repos": res["n_test_repos"], "precision_at_10": res["precision_at_10"],
                "p10_ci_lo": res["p10_ci_lo"], "p10_ci_hi": res["p10_ci_hi"],
                "base_rate_p10": res["base_rate_p10"], "auc_pr": res["auc_pr"],
                "base_rate": res["base_rate"],
                "notes": "phase6b intervention: NO_LABEL_REPLAY minus the 8 repo-level snapshot features",
            })
    return runs


def refit_delta(table: pd.DataFrame, params: dict, saved_auc: float) -> float:
    """Gate check 2: re-fit A/NLR through the cols= code path and compare with Phase 4's
    saved AUC-PR. Everything goes to a scratch directory; the defaults are Phase 4's own
    directories and would overwrite A_NO_LABEL_REPLAY_fold0.txt."""
    tr, te = ex.folds_for("A", table)[0]
    with tempfile.TemporaryDirectory() as d:
        res = ex.run("A", 0, "NO_LABEL_REPLAY", table, params, tr, te,
                     out_models=Path(d) / "m", out_preds=Path(d) / "p",
                     cols=fs.FEATURE_SETS["NO_LABEL_REPLAY"])
    return float(res["auc_pr"] - saved_auc)


def score_rows(scenario: str, table: pd.DataFrame, nlr_runs: list[dict], no_repo_runs: list[dict]
               ) -> tuple[pd.DataFrame, np.ndarray, dict[str, float], dict[str, bool]]:
    """Every test row of one scenario: fold, label, both models' scores, and the NLR model's
    summed repo-feature SHAP. The two models' predictions are joined on pr_id, never by
    position.

    Returns the rows, the pooled SHAP matrix (for reliance), each booster's additivity delta
    (gate check 1), and whether each fold's two test sets are identical (gate check 3)."""
    cols = fs.FEATURE_SETS["NO_LABEL_REPLAY"]
    by_id = table.set_index("pr_id")
    new = {r["fold"]: r for r in no_repo_runs if r["scenario"] == scenario}
    parts, mats, additivity, same = [], [], {}, {}
    for r in sorted((r for r in nlr_runs if r["scenario"] == scenario), key=lambda r: r["fold"]):
        k = r["fold"]
        tag = f"{scenario}_fold{k}"
        old = pd.read_parquet(r["pred_path"])[["pr_id", "repo", "is_slow", "p_hat"]]
        nw = pd.read_parquet(new[k]["pred_path"])[["pr_id", "p_hat"]]
        same[tag] = set(old["pr_id"]) == set(nw["pr_id"])
        m = old.merge(nw, on="pr_id", how="inner", suffixes=("_nlr", "_no_repo"), validate="one_to_one")
        X = attr.prepare(by_id.loc[m["pr_id"].to_numpy(), cols])
        booster = model.load(Path(r["model_path"]))
        sv, ev = attr.explain(booster, X)
        additivity[tag] = attr.additivity_delta(booster, X, sv, ev)
        parts.append(pd.DataFrame({
            "pr_id": m["pr_id"].to_numpy(), "repo": m["repo"].to_numpy(), "fold": k,
            "is_slow": m["is_slow"].to_numpy(dtype=int),
            "p_nlr": m["p_hat_nlr"].to_numpy(dtype=float),
            "p_no_repo": m["p_hat_no_repo"].to_numpy(dtype=float),
            "repo_contrib": fp.repo_contribution(sv, cols),
        }))
        mats.append(sv)
    return pd.concat(parts, ignore_index=True), np.vstack(mats), additivity, same


def reliance(sv_a: np.ndarray, sv_b: np.ndarray) -> pd.DataFrame:
    """Share of mean |SHAP| per feature on the NLR models, A vs B (spec section 5.1,
    descriptive). Sorted by B's share, descending."""
    cols = fs.FEATURE_SETS["NO_LABEL_REPLAY"]
    a = attr.importance(sv_a, cols).set_index("feature")["share"].rename("share_a")
    b = attr.importance(sv_b, cols).set_index("feature")["share"].rename("share_b")
    out = pd.concat([a, b], axis=1).rename_axis("feature").reset_index()
    out["is_repo_feature"] = out["feature"].isin(fp.REPO_FEATURES)
    return out.sort_values("share_b", ascending=False, kind="mergesort").reset_index(drop=True)


def coverage(nlr_runs: list[dict], rows_a: pd.DataFrame, rows_b: pd.DataFrame) -> dict:
    """Gate check 5's premise: A and B cover the same repos, and B holds each out once."""
    held = Counter(repo for r in nlr_runs if r["scenario"] == "B" for repo in r["test_repos"])
    ra, rb = set(rows_a["repo"]), set(rows_b["repo"])
    return {"n_repos": len(ra), "same_repo_set": ra == rb,
            "b_each_once": bool(held) and set(held) == rb and all(v == 1 for v in held.values())}


def fold_table(rows: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per (scenario, fold): both models' AUC-PR on identical rows."""
    out = []
    for sc, r in rows.items():
        for k, g in r.groupby("fold"):
            y = g["is_slow"].to_numpy(dtype=int)
            old, new = metrics.auc_pr(y, g["p_nlr"]), metrics.auc_pr(y, g["p_no_repo"])
            out.append({"scenario": sc, "fold": int(k), "n_test": int(len(g)),
                        "auc_pr_nlr": old, "auc_pr_no_repo": new, "delta": new - old})
    return pd.DataFrame(out)
```

- [ ] **Step 4: Run to verify, then run the mutation check**

Run: `python -m pytest tests/test_report6b.py -q` → **9 passed**.

If `refit_delta`'s `TemporaryDirectory` cleanup raises a `PermissionError` on Windows, a file handle is still open inside it. Fix the handle, not the test, and say so in your report.

Mutation check. Each must make the named test FAIL; restore after each:
- in `score_rows`, replace the merge with a positional concat (`old.assign(p_hat_no_repo=nw["p_hat"].to_numpy())` and rename `p_hat` → `p_hat_nlr`) → `test_score_rows_joins_on_pr_id_and_filters_the_scenario`.
- drop the `r["scenario"] == scenario` filter → the same test (the decoy adds 60 rows).
- delete `reliance`'s `sort_values` → `test_reliance_is_sorted_by_b_share`.
- in `coverage`, compute `b_each_once` as `set(held) == rb` only → `test_coverage_flags_a_repo_held_out_twice`.
- in `refit_delta`, drop `out_models=`/`out_preds=` → `test_refit_delta_never_writes_to_phase4s_directories`.

Then run `python -m pytest tests -q` → **198 passed**, pristine.

- [ ] **Step 5: Commit**

```bash
git add report6b.py tests/test_report6b.py
git commit -m "feat(phase6b): train the intervention, score and explain the NLR boosters

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `report6b.py` — gate, verdict prose, document, `main`

**Files:**
- Modify: `report6b.py` (add imports and constants, then append the functions below)
- Modify: `tests/test_report6b.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 2–4, plus `report6.md(df, fmt="{:.3f}") -> str` (Phase 6's markdown-table helper, reused unchanged).
- Produces:
  - constants `TOL = 1e-6`, `EXPECTED_BOOSTERS = 6`, `EXPECTED_REPOS = 39`, `EXPECTED_NO_REPO_COLS = 25`, `HARD_STOPS = (1, 2)`
  - `report6b.gate_checks(additivity, refit_delta, same_rows, integrity, cov, intervals, artifacts) -> list[dict]`: five `{id, check, value, pass}`. `intervals` may be `None`, which fails check 5.
  - `report6b.hard_stopped(checks) -> list[int]`
  - `report6b.HEADLINE`, `report6b.READING`: dicts keyed by the 6 verdict codes
  - `report6b.verdict_text(s: dict) -> str`
  - `report6b.render(s: dict | None, checks, per_repo_df, fold_df, rel_df) -> str`
  - `report6b.main() -> int`

**Rules the prose must obey.** These come from the Phase 6 review, which caught two misleading sentences in generated text:
- Every sentence must be true for **any** numbers that land in its verdict cell, not just this run's. "Tracks more closely", never "tracks far more closely"; "is relatively better for cold start", never "costs within-project accuracy" (Δ_A could be positive).
- The verdict comes **first**, before the gate table.
- Only checks 1 and 2 suppress the verdict. A failing check 3, 4 or 5 still renders the full document, with the gate marked **FAIL**.

`main()` is exercised only by Task 6's real run. **The Task 5 reviewer must trace it by hand end to end**, as Phase 6's reviewer did for `report6.main`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_report6b.py`:

```python
# ---------------------------------------------------------------------------
# gate_checks
# ---------------------------------------------------------------------------

TAGS = ["A_fold0"] + [f"B_fold{k}" for k in range(5)]


def _ok():
    return dict(
        additivity={t: 1e-12 for t in TAGS},
        refit_delta=0.0,
        same_rows={t: True for t in TAGS},
        integrity={"non_constant": [], "equals_full_minus_no_snapshot": True,
                   "n_no_repo_cols": 25, "hygiene_ok": True},
        cov={"n_repos": 39, "same_repo_set": True, "b_each_once": True},
        intervals={"T_lo": 0.1, "T_hi": 0.5, "I_lo": -0.01, "I_hi": 0.04},
        artifacts={"phase6b_transfer": 39, "phase6b_intervention": 6,
                   "phase6b_reliance": 33, "phase6b_runs": 6},
    )


def _failed(checks):
    return [c["id"] for c in checks if not c["pass"]]


def test_gate_all_pass():
    checks = rb.gate_checks(**_ok())
    assert [c["id"] for c in checks] == [1, 2, 3, 4, 5] and _failed(checks) == []


def _m_additivity(d): d["additivity"]["B_fold3"] = 1e-3
def _m_booster_missing(d): del d["additivity"]["B_fold4"]
def _m_refit(d): d["refit_delta"] = 2e-4
def _m_no_refit(d): d["refit_delta"] = None
def _m_rows(d): d["same_rows"]["B_fold1"] = False
def _m_varies(d): d["integrity"]["non_constant"] = ["n_ci_workflows"]
def _m_nan(d): d["intervals"]["I_hi"] = float("nan")
def _m_twice(d): d["cov"]["b_each_once"] = False
def _m_empty(d): d["artifacts"]["phase6b_reliance"] = 0


@pytest.mark.parametrize("mutate,expected", [
    (_m_additivity, 1), (_m_booster_missing, 1), (_m_refit, 2), (_m_no_refit, 2), (_m_rows, 3),
    (_m_varies, 4), (_m_nan, 5), (_m_twice, 5), (_m_empty, 5),
])
def test_gate_each_failure_flips_only_its_own_check(mutate, expected):
    d = _ok()
    mutate(d)
    assert _failed(rb.gate_checks(**d)) == [expected]


def test_gate_without_intervals_fails_only_check_5():
    d = _ok(); d["intervals"] = None
    assert _failed(rb.gate_checks(**d)) == [5]


def test_hard_stopped_names_only_checks_1_and_2():
    checks = [{"id": i, "pass": False} for i in range(1, 6)]
    assert rb.hard_stopped(checks) == [1, 2]
    assert rb.hard_stopped([{"id": i, "pass": i != 4} for i in range(1, 6)]) == []


# ---------------------------------------------------------------------------
# verdict_text and render
# ---------------------------------------------------------------------------

def _stats(**kw):
    s = {"T": 0.8, "T_lo": 0.4, "T_hi": 1.2, "rho_a": 0.9, "rho_b": 0.1,
         "I": 0.05, "I_lo": 0.01, "I_hi": 0.09, "delta_a": -0.03, "delta_b": 0.02,
         "t_outcome": "confirms", "i_outcome": "confirms", "verdict": "SUPPORTED",
         "share_a": 0.286, "share_b": 0.410, "reliance": "higher",
         "n_draws": 2000, "n_repos": 39, "dropped_T": 0, "dropped_I": 0}
    return {**s, **kw}


# What each verdict's text must SAY, pinned here independently of report6b's own HEADLINE
# and READING dicts. Checking `rb.READING[code] in text` would compare the module against
# itself, and swapping two entries would still pass.
HEAD_SAYS = {
    "SUPPORTED": "fingerprinting is supported",
    "PARTIAL_SHAP_ONLY": "the SHAP transfer test confirms, the intervention does not",
    "PARTIAL_INTERVENTION_ONLY": "the intervention confirms, the SHAP transfer test does not",
    "NOT_SUPPORTED": "fingerprinting is not supported",
    "CONFLICTING": "The two tests conflict",
    "CONTRADICTED": "fingerprinting is contradicted",
}
READ_SAYS = {
    "SUPPORTED": "pre-registered signature of repo fingerprinting",
    "PARTIAL_SHAP_ONLY": "does not measurably favour cold start",
    "PARTIAL_INTERVENTION_ONLY": "not visible in the attributions",
    "NOT_SUPPORTED": "not explained by this mechanism",
    "CONFLICTING": "cannot adjudicate",
    "CONTRADICTED": "points away from fingerprinting",
}


@pytest.mark.parametrize("code", sorted(set(fp.VERDICTS.values())))
def test_verdict_text_says_the_right_thing_for_its_cell(code):
    text = rb.verdict_text(_stats(verdict=code))
    first = text.split("\n")[0]                                    # the headline comes first
    assert HEAD_SAYS[code] in first
    assert all(HEAD_SAYS[o] not in first for o in set(HEAD_SAYS) - {code})
    assert READ_SAYS[code] in text
    assert all(READ_SAYS[o] not in text for o in set(READ_SAYS) - {code})
    assert "+0.80" in text and "[+0.40, +1.20]" in text          # the transfer interval
    assert "+0.050" in text and "[+0.010, +0.090]" in text        # the intervention interval


def test_verdict_text_states_the_crossover_only_when_it_holds():
    yes = rb.verdict_text(_stats(delta_a=-0.03, delta_b=0.0))      # hurts A, B unharmed: Δ_B = 0 counts
    no = rb.verdict_text(_stats(delta_a=0.01, delta_b=0.02))       # helps both: not a crossover
    assert "The strong form holds" in yes and "does not hold" not in yes
    assert "does not hold" in no and "The strong form holds" not in no


def _checks(fail=()):
    return [{"id": i, "check": f"check {i}", "value": "ok", "pass": i not in fail} for i in range(1, 6)]


def _frames():
    per_repo = pd.DataFrame({"repo": ["r0"], "c_A": [0.1], "y_A": [0.5], "n_A": [10],
                             "c_B": [0.0], "y_B": [0.4], "n_B": [20]})
    folds = pd.DataFrame({"scenario": ["A"], "fold": [0], "n_test": [10], "auc_pr_nlr": [0.9],
                          "auc_pr_no_repo": [0.88], "delta": [-0.02]})
    rel = pd.DataFrame({"feature": ["n_mentionable_users"], "share_a": [0.04], "share_b": [0.17],
                        "is_repo_feature": [True]})
    return per_repo, folds, rel


def test_render_puts_the_verdict_first():
    doc = rb.render(_stats(), _checks(), *_frames())
    assert doc.index("## Verdict") < doc.index("## Gate") < doc.index("## 2. Transfer test")
    assert rb.HEADLINE["SUPPORTED"] in doc and "No verdict" not in doc


@pytest.mark.parametrize("failed", [(1,), (2,), (1, 2)])
def test_render_suppresses_the_verdict_on_a_hard_stop(failed):
    doc = rb.render(_stats(), _checks(fail=failed), *_frames())
    assert "HARD STOP" in doc and "## Verdict" not in doc and "## 2. Transfer test" not in doc
    assert all(h not in doc for h in rb.HEADLINE.values())


def test_render_without_statistics_reports_no_verdict():
    doc = rb.render(None, _checks(fail=(5,)), None, None, None)
    assert "No verdict is reported" in doc and "## Verdict" not in doc


def test_render_with_a_soft_failure_still_reports_the_verdict():
    doc = rb.render(_stats(), _checks(fail=(4,)), *_frames())
    assert "## Verdict" in doc and "**FAIL**" in doc and "HARD STOP" not in doc
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_report6b.py -q`
Expected: the new tests FAIL with `AttributeError: module 'report6b' has no attribute 'gate_checks'` (or `verdict_text` / `render` / `HEADLINE`). Task 4's 9 still pass.

- [ ] **Step 3: Implement**

Add to `report6b.py`'s import block: `import json`, `import logging`, `import sys` (with the other stdlib imports) and `from report6 import md` (after the project imports). Then, directly below the imports:

```python
log = logging.getLogger("report6b")
ROOT = Path(__file__).parent
DOC = ROOT / "docs" / "phase6b_fingerprinting.md"
DATA = ROOT / "data"
RUNS_JSON = DATA / "phase6b_runs.json"
GATE_JSON = DATA / "phase6b_gate.json"
SPEC = "docs/design/specs/2026-09-24-phase6b-fingerprinting-design.md"

TOL = 1e-6
EXPECTED_BOOSTERS = 6          # 1 Scenario A fold + 5 Scenario B folds
EXPECTED_REPOS = 39            # the cohort after Phase 2's QC
EXPECTED_NO_REPO_COLS = 25     # NO_LABEL_REPLAY's 33 minus the 8 repo-level features
HARD_STOPS = (1, 2)
```

Append:

```python
# ---------------------------------------------------------------------------
# Gate (pure) -- spec section 7
# ---------------------------------------------------------------------------

def gate_checks(additivity: dict[str, float], refit_delta: float | None, same_rows: dict[str, bool],
                integrity: dict, cov: dict, intervals: dict | None, artifacts: dict[str, int]) -> list[dict]:
    worst = max(additivity.values(), default=float("inf"))
    c1 = {"id": 1, "check": f"SHAP additivity vs raw margin on all {EXPECTED_BOOSTERS} NLR boosters "
                            f"(max |delta| < {TOL})",
          "value": {k: float(v) for k, v in additivity.items()},
          "pass": len(additivity) == EXPECTED_BOOSTERS and worst < TOL}
    c2 = {"id": 2, "check": f"A/NLR refit via the cols= code path reproduces Phase 4's saved AUC-PR "
                            f"(|delta| < {TOL})",
          "value": refit_delta, "pass": refit_delta is not None and abs(refit_delta) < TOL}
    c3 = {"id": 3, "check": "every NLR_NO_REPO fold has exactly the NLR fold's test rows",
          "value": same_rows, "pass": len(same_rows) == EXPECTED_BOOSTERS and all(same_rows.values())}
    ok4 = (not integrity["non_constant"] and integrity["equals_full_minus_no_snapshot"]
           and integrity["n_no_repo_cols"] == EXPECTED_NO_REPO_COLS and integrity["hygiene_ok"])
    c4 = {"id": 4, "check": "the 8 repo features are constant within every repo and equal FULL - NO_SNAPSHOT; "
                            f"NLR_NO_REPO is NLR minus exactly them ({EXPECTED_NO_REPO_COLS} columns)",
          "value": integrity, "pass": bool(ok4)}
    finite = intervals is not None and all(np.isfinite(intervals[k]) for k in ("T_lo", "T_hi", "I_lo", "I_hi"))
    written = bool(artifacts) and all(v > 0 for v in artifacts.values())
    ok5 = cov["n_repos"] == EXPECTED_REPOS and cov["same_repo_set"] and cov["b_each_once"] and finite and written
    c5 = {"id": 5, "check": f"A and B cover the same {EXPECTED_REPOS} repos, B holds each out once, "
                            "every interval is finite, every artifact is written",
          "value": {**cov, "intervals_finite": finite, "artifacts": artifacts}, "pass": bool(ok5)}
    return [c1, c2, c3, c4, c5]


def hard_stopped(checks: list[dict]) -> list[int]:
    return [c["id"] for c in checks if c["id"] in HARD_STOPS and not c["pass"]]


# ---------------------------------------------------------------------------
# Verdict prose (pure). Every sentence must hold for ANY numbers in its cell.
# ---------------------------------------------------------------------------

HEADLINE = {
    "SUPPORTED": "**Repo fingerprinting is supported.** Both pre-registered tests confirm it.",
    "PARTIAL_SHAP_ONLY": "**Partially supported: the SHAP transfer test confirms, the intervention does not.**",
    "PARTIAL_INTERVENTION_ONLY": "**Partially supported: the intervention confirms, the SHAP transfer test does not.**",
    "NOT_SUPPORTED": "**Repo fingerprinting is not supported.** Neither pre-registered test confirms it.",
    "CONFLICTING": "**The two tests conflict.** One confirms repo fingerprinting and the other contradicts it.",
    "CONTRADICTED": "**Repo fingerprinting is contradicted.** Neither test confirms it and at least one "
                    "points the other way.",
}

READING = {
    "SUPPORTED": (
        "On repos the model saw in training, the repo-level features' contribution tracks each repo's "
        "actual slow rate more closely than on held-out repos, and removing those features is relatively "
        "better for cold start than for within-project prediction. That is the pre-registered signature of "
        "repo fingerprinting: stripped of the label-replay features, the model identifies repos by their "
        "static attributes and replays their base rates, which cannot carry over to a repo it has never seen."),
    "PARTIAL_SHAP_ONLY": (
        "The repo-level features' contribution tracks actual slow rates more closely on seen repos than on "
        "held-out ones, which is the fingerprinting pattern. But removing those features does not measurably "
        "favour cold start over within-project prediction, so the pattern is not shown to be what costs "
        "cold-start accuracy."),
    "PARTIAL_INTERVENTION_ONLY": (
        "Removing the repo-level features is relatively better for cold start than for within-project "
        "prediction, which is what fingerprinting predicts. But their SHAP contribution does not track actual "
        "slow rates measurably better on seen repos than on held-out ones, so the mechanism fingerprinting "
        "names is not visible in the attributions."),
    "NOT_SUPPORTED": (
        "Neither the attributions nor the intervention separate cold start from within-project prediction in "
        "the way fingerprinting predicts. The cold-start failure is not explained by this mechanism at the "
        "resolution this cohort allows."),
    "CONFLICTING": (
        "The attributions and the intervention point in opposite directions, so this test cannot adjudicate "
        "fingerprinting. Neither result should be reported without the other."),
    "CONTRADICTED": (
        "The evidence points away from fingerprinting: at least one test shows the opposite of the "
        "pre-registered prediction, with an interval excluding zero, and neither confirms it."),
}


def verdict_text(s: dict) -> str:
    t = (f"ρ_A − ρ_B = {s['T']:+.2f}, 95% CI [{s['T_lo']:+.2f}, {s['T_hi']:+.2f}]; ρ_A = {s['rho_a']:+.2f} "
         f"on repos seen in training, ρ_B = {s['rho_b']:+.2f} on held-out repos")
    i = (f"Δ_B − Δ_A = {s['I']:+.3f}, 95% CI [{s['I_lo']:+.3f}, {s['I_hi']:+.3f}]; removing the 8 features "
         f"moved AUC-PR by {s['delta_a']:+.3f} on A and {s['delta_b']:+.3f} on B")
    strong = ("The strong form holds: removing them hurts A and does not hurt B."
              if s["delta_a"] < 0 <= s["delta_b"] else
              "The strong form, a crossover that hurts A while leaving B unharmed, does not hold.")
    band = {"higher": "higher on B", "lower": "lower on B",
            "unchanged": f"essentially unchanged, within ±{fp.RELIANCE_BAND * 100:.0f} points"}[s["reliance"]]
    pts = (s["share_b"] - s["share_a"]) * 100
    return (f"{HEADLINE[s['verdict']]}\n\n"
            f"- SHAP transfer test, **{s['t_outcome']}**: {t}.\n"
            f"- Intervention, **{s['i_outcome']}**: {i}. {strong}\n"
            f"- Reliance (descriptive, never part of the verdict): the 8 features carry {s['share_a']:.1%} of "
            f"mean |SHAP| on A and {s['share_b']:.1%} on B ({pts:+.1f} points), {band}.\n\n"
            f"{READING[s['verdict']]}")


# ---------------------------------------------------------------------------
# Document (pure)
# ---------------------------------------------------------------------------

def render(s: dict | None, checks: list[dict], per_repo_df: pd.DataFrame | None,
           fold_df: pd.DataFrame | None, rel_df: pd.DataFrame | None) -> str:
    gate = md(pd.DataFrame(checks)[["id", "check", "value", "pass"]], "{}")
    head = ("# Phase 6b — The repo-fingerprinting test\n\n"
            "Generated by `report6b.py`: every line below is produced from the data, and there is no "
            f"hand-written section. Pre-registered in `{SPEC}`, committed before any of this ran. The "
            "hypothesis itself is post-hoc (spec §2): it came from an exploratory gain-importance peek, which "
            "this test does not rely on.\n\n")
    stopped = hard_stopped(checks)
    if stopped or s is None:
        failed = [c["id"] for c in checks if not c["pass"]]
        why = (f"Gate check(s) {', '.join(map(str, stopped))} failed, a **HARD STOP**: the SHAP attributions "
               "or the intervention cannot be trusted, so no statistic or verdict is computed from them."
               if stopped else
               f"Gate check(s) {', '.join(map(str, failed))} failed, so the statistics a verdict needs are "
               "not available.")
        return head + f"## No verdict is reported\n\n{why}\n\n## Gate\n\n{gate}\n"
    status = "PASS" if all(c["pass"] for c in checks) else "FAIL"
    feats = ", ".join(f"`{c}`" for c in fp.REPO_FEATURES)
    return (
        head
        + f"## Verdict\n\n{verdict_text(s)}\n\n"
        + f"## Gate: **{status}** (validity, not what the verdict says)\n\n{gate}\n\n"
        + f"## 1. The eight repo-level features\n\n{feats}.\n\n"
        + "Each is one 2026 snapshot value per repo, so each is constant within every repo. They are the same "
          "eight Phase 4's `NO_SNAPSHOT` ablation removed. `repo_age_days_at_open` is excluded because it "
          "varies within a repo, so the definition can only under-count fingerprinting.\n\n"
        + "## 2. Transfer test (SHAP)\n\n"
        + "For each repo, `c` is the mean summed SHAP contribution of the 8 features over its test rows "
          f"(log-odds) and `y` its actual slow rate. ρ_A = {s['rho_a']:+.3f}, ρ_B = {s['rho_b']:+.3f}; "
          f"ρ_A − ρ_B = {s['T']:+.3f}, 95% CI [{s['T_lo']:+.3f}, {s['T_hi']:+.3f}]: **{s['t_outcome']}**.\n\n"
        + f"{md(per_repo_df)}\n\n"
        + "## 3. Intervention\n\n"
        + f"`NLR_NO_REPO` is `NO_LABEL_REPLAY` minus the 8 ({EXPECTED_NO_REPO_COLS} features), trained on the "
          "same folds with the same tuned params. AUC-PR follows Phase 4: Scenario B is the mean of its "
          "per-fold values.\n\n"
        + f"{md(fold_df)}\n\n"
        + f"Δ_A = {s['delta_a']:+.4f}, Δ_B = {s['delta_b']:+.4f}; Δ_B − Δ_A = {s['I']:+.4f}, "
          f"95% CI [{s['I_lo']:+.4f}, {s['I_hi']:+.4f}]: **{s['i_outcome']}**.\n\n"
        + "## 4. Reliance (descriptive)\n\n"
        + "Share of mean |SHAP| on the NLR models. The 8 repo-level features together carry "
          f"{s['share_a']:.1%} on A and {s['share_b']:.1%} on B.\n\n"
        + f"{md(rel_df.head(12))}\n\n"
        + "## Method notes\n\n"
        + f"Intervals come from {s['n_draws']:,} paired draws of the {s['n_repos']} repos with replacement; "
          "one draw is applied to both scenarios and both statistics (spec §5.4). Draws dropped as "
          f"non-finite: {s['dropped_T']} for the transfer test, {s['dropped_I']} for the intervention.\n"
    )


# ---------------------------------------------------------------------------
# main -- exercised only by Task 6's real run; the Task 5 reviewer traces it by hand
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    table = ex.load_table()
    params = json.loads(ex.PARAMS.read_text(encoding="utf-8"))["best_params"]
    sha = ex.params_sha()
    nlr_runs = [r for r in attr.load_runs() if r["featureset"] == "NO_LABEL_REPLAY"]
    saved_a = next(r["auc_pr"] for r in nlr_runs if r["scenario"] == "A")

    new_runs = train_no_repo(table, params, sha)
    DATA.mkdir(exist_ok=True)
    RUNS_JSON.write_text(json.dumps({"params_sha": sha, "runs": new_runs}, indent=2), encoding="utf-8")
    refit = refit_delta(table, params, saved_a)
    log.info("refit delta vs Phase 4's saved A/NLR AUC-PR: %.2e", refit)

    rows, svs, additivity, same = {}, {}, {}, {}
    for sc in ("A", "B"):
        rows[sc], svs[sc], add, sm = score_rows(sc, table, nlr_runs, new_runs)
        additivity.update(add)
        same.update(sm)
        log.info("scenario %s: %d rows explained, max additivity delta %.2e", sc, len(rows[sc]), max(add.values()))

    no_repo = fp.nlr_no_repo_cols()
    try:
        fs.assert_hygiene(no_repo)
        hygiene_ok = True
    except ValueError:
        hygiene_ok = False
    integrity = {
        "non_constant": fp.non_constant_within_repo(table, fp.REPO_FEATURES),
        "equals_full_minus_no_snapshot": (set(fp.REPO_FEATURES)
                                          == set(fs.FEATURE_SETS["FULL"]) - set(fs.FEATURE_SETS["NO_SNAPSHOT"])),
        "n_no_repo_cols": len(no_repo), "hygiene_ok": hygiene_ok,
    }
    cov = coverage(nlr_runs, rows["A"], rows["B"])

    pre = gate_checks(additivity, refit, same, integrity, cov, None, {"phase6b_runs": len(new_runs)})
    if hard_stopped(pre):
        GATE_JSON.write_text(json.dumps(pre, indent=2, default=str), encoding="utf-8")
        DOC.write_text(render(None, pre, None, None, None), encoding="utf-8")
        print(f"HARD STOP on gate check(s) {hard_stopped(pre)} -- wrote {DOC}")
        return 1

    boot = fp.paired_repo_bootstrap(rows["A"], rows["B"])
    rel = reliance(svs["A"], svs["B"])
    per_repo_df = (fp.per_repo(rows["A"]).add_suffix("_A")
                   .join(fp.per_repo(rows["B"]).add_suffix("_B")).reset_index())
    fold_df = fold_table(rows)
    per_repo_df.to_csv(DATA / "phase6b_transfer.csv", index=False)
    fold_df.to_csv(DATA / "phase6b_intervention.csv", index=False)
    rel.to_csv(DATA / "phase6b_reliance.csv", index=False)
    artifacts = {"phase6b_transfer": len(per_repo_df), "phase6b_intervention": len(fold_df),
                 "phase6b_reliance": len(rel), "phase6b_runs": len(new_runs)}
    checks = gate_checks(additivity, refit, same, integrity, cov, boot, artifacts)
    GATE_JSON.write_text(json.dumps(checks, indent=2, default=str), encoding="utf-8")

    finite = all(np.isfinite(boot[k]) for k in ("T_lo", "T_hi", "I_lo", "I_hi"))
    if not finite:
        DOC.write_text(render(None, checks, None, None, None), encoding="utf-8")
        print(f"no verdict: non-finite interval -- wrote {DOC}")
        return 1

    share_a = float(rel.loc[rel["is_repo_feature"], "share_a"].sum())
    share_b = float(rel.loc[rel["is_repo_feature"], "share_b"].sum())
    t_out, i_out = fp.outcome(boot["T_lo"], boot["T_hi"]), fp.outcome(boot["I_lo"], boot["I_hi"])
    s = {**boot, "t_outcome": t_out, "i_outcome": i_out, "verdict": fp.verdict(t_out, i_out),
         "share_a": share_a, "share_b": share_b, "reliance": fp.reliance_band(share_b - share_a)}
    DOC.write_text(render(s, checks, per_repo_df, fold_df, rel), encoding="utf-8")
    ok = all(c["pass"] for c in checks)
    print(f"wrote {DOC}  gate={'PASS' if ok else 'FAIL'}  verdict={s['verdict']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify, then run the mutation check**

Run: `python -m pytest tests/test_report6b.py -q` → **34 passed** (Task 4's 9 plus these 25).

Mutation check. Each must make the named test FAIL; restore after each:
- `gate_checks` returns `pass: False` for every check → `test_gate_all_pass` and every `test_gate_each_failure_flips_only_its_own_check` case.
- `HARD_STOPS = (1, 2, 4)` → `test_hard_stopped_names_only_checks_1_and_2` and `test_render_with_a_soft_failure_still_reports_the_verdict`.
- swap the `PARTIAL_SHAP_ONLY` and `PARTIAL_INTERVENTION_ONLY` entries in `READING` → `test_verdict_text_says_the_right_thing_for_its_cell`.
- swap the same two entries in `HEADLINE` → `test_verdict_text_says_the_right_thing_for_its_cell`.
- make the crossover condition `s["delta_a"] < 0 < s["delta_b"]` (strict) → `test_verdict_text_states_the_crossover_only_when_it_holds`.

Then run `python -m pytest tests -q` → **223 passed**, pristine.

- [ ] **Step 5: Commit**

```bash
git add report6b.py tests/test_report6b.py
git commit -m "feat(phase6b): validity gate, verdict prose, fully generated document

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Real-data run (CONDITIONAL on Tasks 1–5 reviewed)

**Files:** no code. Produces `data/phase6b_{transfer,intervention,reliance}.csv`, `data/phase6b_runs.json`, `data/phase6b_gate.json` and `docs/phase6b_fingerprinting.md`, and appends 6 rows to `data/experiments.csv`. The boosters and predictions land in the gitignored `data/models/` and `data/predictions/`.

- [ ] **Step 1: Run it**

First, record Phase 4's A/NLR booster timestamp. Step 2 needs it to prove the run overwrote nothing:
`python -c "import os; print(os.path.getmtime('data/models/A_NO_LABEL_REPLAY_fold0.txt'))"`

Then run: `python report6b.py; echo exit=$?`. Report the exit code and wall-clock time. Expect a few minutes: 7 LightGBM fits, exact TreeSHAP on ~52,600 rows and 2,000 bootstrap draws.

**If gate check 1 or 2 fails, STOP.** The document will say "No verdict is reported". Report the failing check's value and quote no statistic anywhere: the attributions or the intervention cannot be trusted.

- [ ] **Step 2: Verify three things the gate does not check**

1. Phase 4's own numbers reproduce. Every `auc_pr_nlr` in `data/phase6b_intervention.csv` must equal the matching `auc_pr` in `data/phase4_runs.json` to 1e-6 (A 0.901745; B 0.964792 / 0.834803 / 0.507069 / 0.643918 / 0.866883). A mismatch means `p_nlr` is not Phase 4's saved prediction.
2. Phase 4's gate still holds on the updated log. Re-run the one-liner from Task 1 Step 4; it must print `24` even though `data/experiments.csv` now has 6 more `lgbm` rows. Also confirm those 6 rows exist with `features == "NLR_NO_REPO"`.
3. Nothing of Phase 4's was overwritten. `git status --porcelain` must show no change to `data/phase4_runs.json`, and `data/models/A_NO_LABEL_REPLAY_fold0.txt` must have the same modification time as before the run.

- [ ] **Step 3: Report the numbers verbatim**

In the task report, quote: the full gate table from `data/phase6b_gate.json`; the verdict section of the document; ρ_A, ρ_B, T and its interval; Δ_A, Δ_B, I and its interval; the per-fold intervention table; the repo-feature reliance shares; and the dropped-draw counts.

**Report the verdict whichever cell it lands in.** All nine cells are valid outcomes. The phase's own discipline is that a pre-registered test is reported as it came out. Do not edit `report6b.py`'s prose, thresholds or rules after seeing the result.

- [ ] **Step 4: Commit**

Check `.gitignore` first; commit only tracked paths:

```bash
git add docs/phase6b_fingerprinting.md data/phase6b_transfer.csv data/phase6b_intervention.csv \
        data/phase6b_reliance.csv data/phase6b_runs.json data/phase6b_gate.json data/experiments.csv
git commit -m "data: Phase 6b repo-fingerprinting test results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Verification (spec §7 and §9)

1. `python -m pytest tests -q` → 223 passed, pristine. Every task's mutation check caught its mutation.
2. Real data: `data/phase6b_gate.json` passes 5/5, with additivity < 1e-6 on all 6 NLR boosters and the refit exact.
3. `docs/phase6b_fingerprinting.md` states its verdict first, in whichever of the nine cells it landed.
4. Phase 4 is undisturbed: its saved per-fold AUC-PRs are reproduced, its gate check 4 still counts 24, and `phase4_runs.json` is unchanged.

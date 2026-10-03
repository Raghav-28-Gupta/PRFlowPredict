# Phase 8 — The Write-up — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A portfolio-facing README and report for hiring managers and technical interviewers, two headline figures, blueprint status corrections, and a test that checks every cited result against its committed artifact and its document.

**Architecture:** `writeup_figures.py` renders two figures from committed artifacts only. The README and `docs/REPORT.md` are prose, given verbatim below. `writeup_claims.py` is a registry of 42 cited results, each recomputed from a committed data file (or, where no data file holds it, parsed from a committed generated phase document). `tests/test_writeup_claims.py` checks each claim both ways, plus honesty phrases and dead links.

**Tech Stack:** Python 3.13, pandas, numpy, scipy (`spearmanr`), matplotlib (Agg), pytest.

**Spec:** `docs/design/specs/2026-09-30-phase8-writeup-design.md` (commit `5b4db6c`).

**This plan was dry-run before it was committed.** A replica of the repo was built from `git archive HEAD`. Every file, script and document was then **extracted from this plan file itself**, every step was run as written, and the results were:
- **All 42 claims render exactly** from the committed artifacts, and both documents contain every claim string.
- **The suite passes:** 347 passed and 2 skipped. The 2 skips need gitignored pilot data the replica lacks; expect **349 passed** in the real repo, where the same suite passes **240** today.
- **Every mutation in Tasks 1 and 3 is caught**, all 12 of them. Each target test passed unmutated (pytest exit 0) and failed mutated (exit 1). The tree was clean after each `git checkout` restore, and the hashes of the mutated artifacts were unchanged afterwards.
- **Task 4's script applies all six replacements**, each matching exactly once.
- **Both figures render byte-identically** across runs, and were inspected visually.

So if something fails as written, suspect a transcription slip or an environment difference first, and say which.

## Global Constraints

- **Nothing is retrained, and no new code reads a gitignored path** (`data/models/`, `data/predictions/`, `data/features/`, `data/raw/`, `data/processed/`). Everything reads committed artifacts.
- **Never run** `experiment.py`, `report6.py`, `report6b.py` or any collection script. They retrain, append duplicate rows to `data/experiments.csv`, or overwrite hand-written text.
- **Do not modify any phase's generated document or its generator:** `docs/phase*.md`, `report4.py`, `report6.py`, `report6b.py`.
- **Voice:** project voice ("the project pre-registers…"), with no first-person authorship claims and **no mention of AI tooling** anywhere in `README.md` or `docs/REPORT.md`.
- **Numbers:** negative numbers in claim strings use the ASCII hyphen `-`, not the Unicode minus `−`.
- **Honesty (spec §9) binds all prose.** `PARTIAL_SHAP_ONLY` is never restated as "supported" or "confirmed". P(I ≤ 0) is never cited.
- **Mutation checks are restored with `git checkout -- <file>`, never by rewriting the text.** On Windows, a text-mode rewrite silently converts LF to CRLF and leaves the file modified. After every mutation check, `git status --porcelain` must show a clean tree.
- Tests live in `tests/` and run with `python -m pytest tests -q`, and output must be **pristine**. **240 pass today.**
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

These are failure modes the spec implies but no single happy-path test exercises. Each is pinned by a test in the task that owns it.

1. **A cited number in the wrong sentence or table row.** A reader would be told the wrong thing while the number "appears". → Claim strings carry their context, and table claims are whole rows. Pinned in Task 3: swapping two values within a table row is a mandated mutation.
2. **Markdown line-wrapping splits a number across two lines**, so a correct document fails. → Presence is checked on whitespace-normalised text. Pinned in Task 3: `test_wrapped_claim_still_counts`.
3. **A reader clicks a dead link, a missing image or a stale heading anchor.** → The link test resolves every relative link and anchor in the README, the report and `docs/data_collection.md`. Pinned in Task 3.
4. **A figure draws different numbers than the report states.** → Each plotting function returns what it drew, and tests compare that against the artifact. Pinned in Task 1.
5. **A later regeneration of a phase document rewords a sentence the claims parse**, and a claim silently matches the wrong sentence. → `_parse` demands exactly one match, and 6b's T and I are cross-checked against a recomputation. Pinned in Task 3: `test_parse_demands_exactly_one_match`.

---

## File structure

| File | Responsibility |
|---|---|
| `writeup_figures.py` (create) | Render Figures 1 and 2 from committed artifacts; each plotting function returns the values it drew |
| `tests/test_writeup_figures.py` (create) | The figures show the committed numbers and render deterministically |
| `figures/headline_transfer.png`, `figures/fingerprint_scatter.png` (create, generated) | The two headline figures |
| `docs/data_collection.md` (create) | The current README's Phase 1 notes, moved verbatim apart from two re-based links |
| `README.md` (rewrite) | The front door, about 2 minutes to read |
| `docs/REPORT.md` (create) | The full write-up, about 15 minutes to read |
| `writeup_claims.py` (create) | The claims registry, the required phrases and the forbidden phrasings |
| `tests/test_writeup_claims.py` (create) | Claims in both directions, honesty phrases, dead links |
| `actionable_ml_project_blueprint.md` (modify) | Status corrections only |

---

### Task 1: The headline figures

**Files:**
- Create: `writeup_figures.py`, `tests/test_writeup_figures.py`
- Create, generated: `figures/headline_transfer.png`, `figures/fingerprint_scatter.png`

**Interfaces:**
- Consumes: the committed `data/phase4_runs.json` and `data/phase6b_transfer.csv`.
- Produces:
  - `writeup_figures.load_runs(path: Path | None = None) -> list[dict]`
  - `transfer_data(runs) -> dict[str, dict[str, list[float]]]`
  - `headline_transfer(runs, out: Path) -> dict`
  - `scatter_data(transfer: pd.DataFrame) -> dict[str, dict]`
  - `fingerprint_scatter(transfer, out: Path) -> dict`
  - `main(out_dir: Path = FIG) -> int`
  - constants `ROOT`, `FIG`, `RUNS`, `TRANSFER`

  `load_runs` resolves its default path **at call time, not definition time**, so tests can point it elsewhere. A default argument bound at definition time already caused one incident in this project.

- [ ] **Step 1: Write the failing test**

`tests/test_writeup_figures.py`:

```python
"""The two headline figures show exactly the committed numbers, and render deterministically."""
import numpy as np
import pandas as pd

import writeup_figures as wf


def _runs(runs, scenario, featureset):
    return sorted((r for r in runs if r["scenario"] == scenario and r["featureset"] == featureset),
                  key=lambda r: r["fold"])


def test_transfer_data_is_phase4s_numbers():
    runs = wf.load_runs()
    d = wf.transfer_data(runs)
    assert [len(d["A"][k]) for k in ("FULL", "NO_LABEL_REPLAY", "baseline")] == [1, 1, 1]
    assert [len(d["B"][k]) for k in ("FULL", "NO_LABEL_REPLAY", "baseline")] == [5, 5, 5]
    # each value is taken verbatim from the runs file, fold by fold
    assert d["B"]["NO_LABEL_REPLAY"] == [r["auc_pr"] for r in _runs(runs, "B", "NO_LABEL_REPLAY")]
    assert d["B"]["FULL"] == [r["auc_pr"] for r in _runs(runs, "B", "FULL")]
    # and the bars are the published Phase 4 numbers
    assert round(float(np.mean(d["A"]["FULL"])), 3) == 0.906
    assert round(float(np.mean(d["B"]["NO_LABEL_REPLAY"])), 3) == 0.763
    assert round(float(np.mean(d["B"]["baseline"])), 3) == 0.821


def test_the_baseline_is_the_same_whichever_model_supplies_it():
    """Figure 1 takes the baseline from FULL's runs; that is only valid because the baseline
    is scored on the same rows for every feature set."""
    runs = wf.load_runs()
    for sc in ("A", "B"):
        full = [r["baseline_auc_pr"] for r in _runs(runs, sc, "FULL")]
        nlr = [r["baseline_auc_pr"] for r in _runs(runs, sc, "NO_LABEL_REPLAY")]
        assert full == nlr


def test_scatter_data_is_the_transfer_table():
    t = pd.read_csv(wf.TRANSFER)
    d = wf.scatter_data(t)
    for sc in ("A", "B"):
        assert np.array_equal(d[sc]["c"], t[f"c_{sc}"].to_numpy())
        assert np.array_equal(d[sc]["y"], t[f"y_{sc}"].to_numpy())
    assert round(d["A"]["rho"], 3) == 0.669 and round(d["B"]["rho"], 3) == 0.318


def test_figures_render_deterministically_into_the_given_directory(tmp_path):
    committed = {p.name: p.stat().st_mtime_ns for p in wf.FIG.glob("*.png")}
    for sub in ("one", "two"):
        assert wf.main(tmp_path / sub) == 0
    for name in ("headline_transfer.png", "fingerprint_scatter.png"):
        a, b = (tmp_path / "one" / name).read_bytes(), (tmp_path / "two" / name).read_bytes()
        assert len(a) > 10_000
        assert a == b, f"{name} is not byte-identical across two renders"
    # rendering into a given directory must never touch the committed figures
    assert {p.name: p.stat().st_mtime_ns for p in wf.FIG.glob("*.png")} == committed
```

- [ ] **Step 2: Run it and confirm it fails**

Run `python -m pytest tests/test_writeup_figures.py -q`. Expected: FAIL with `ModuleNotFoundError: No module named 'writeup_figures'`.

- [ ] **Step 3: Implement `writeup_figures.py`**

```python
"""Phase 8's two headline figures, rendered from COMMITTED artifacts only.

Nothing is retrained, and nothing under the gitignored data/models, data/predictions or
data/features is read, so anyone who clones the repo can regenerate both figures:

    python writeup_figures.py

Each plotting function returns the exact values it drew, so tests can check the figure
against its source artifact rather than trusting the picture."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent
FIG = ROOT / "figures"
RUNS = ROOT / "data" / "phase4_runs.json"
TRANSFER = ROOT / "data" / "phase6b_transfer.csv"
PNG_META = {"Software": None}          # no version string in the PNG, so output is byte-stable
SETS = ("FULL", "NO_LABEL_REPLAY")
LABELS = {"FULL": "model, all features", "NO_LABEL_REPLAY": "model, without the repo's own history",
          "baseline": "trailing-rate baseline"}
COLOURS = {"FULL": "#2b6cb0", "NO_LABEL_REPLAY": "#90cdf4", "baseline": "#a0aec0"}


def load_runs(path: Path | None = None) -> list[dict]:
    # resolved at call time, not definition time, so tests can point it elsewhere
    return json.loads((path or RUNS).read_text(encoding="utf-8"))["runs"]


def transfer_data(runs: list[dict]) -> dict[str, dict[str, list[float]]]:
    """What Figure 1 plots. Per scenario: AUC-PR for FULL, NO_LABEL_REPLAY and the baseline,
    each a list over folds (one for Scenario A, five for Scenario B). A bar is the mean.
    The baseline is scored on the same rows as the model, so FULL's runs supply it."""
    out = {}
    for sc in ("A", "B"):
        by = {f: sorted((r for r in runs if r["scenario"] == sc and r["featureset"] == f),
                        key=lambda r: r["fold"]) for f in SETS}
        if not all(by.values()):
            raise LookupError(f"missing Scenario {sc} runs for {[f for f, v in by.items() if not v]}")
        out[sc] = {"FULL": [r["auc_pr"] for r in by["FULL"]],
                   "NO_LABEL_REPLAY": [r["auc_pr"] for r in by["NO_LABEL_REPLAY"]],
                   "baseline": [r["baseline_auc_pr"] for r in by["FULL"]]}
    return out


def headline_transfer(runs: list[dict], out: Path) -> dict:
    """Figure 1. Phase 4 recorded no AUC-PR intervals, so none are drawn: Scenario A is one
    value per bar, and Scenario B's five per-fold values are overlaid as dots."""
    data = transfer_data(runs)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    keys, width = ("FULL", "NO_LABEL_REPLAY", "baseline"), 0.26
    for i, sc in enumerate(("A", "B")):
        for j, k in enumerate(keys):
            x = i + (j - 1) * width
            vals = data[sc][k]
            mean = float(np.mean(vals))
            ax.bar(x, mean, width * 0.92, color=COLOURS[k], label=LABELS[k] if i == 0 else None)
            # value at the bar's base: no fold dot reaches that low, so the label never collides
            ax.text(x, 0.415, f"{mean:.3f}", ha="center", va="bottom", fontsize=9,
                    color="white" if k == "FULL" else "#1a202c")
            if len(vals) > 1:
                ax.scatter([x] * len(vals), vals, s=14, color="#1a202c", zorder=3,
                           label="one held-out fold" if (i == 1 and j == 0) else None)
    ax.set_xticks([0, 1], ["Scenario A: repos seen in training", "Scenario B: repos never seen"])
    ax.set_ylabel("AUC-PR")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Within a project the model beats the baseline;\n"
                 "on unseen repos it does so only with the repo's own history", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, metadata=PNG_META)
    plt.close(fig)
    return data


def scatter_data(transfer: pd.DataFrame) -> dict[str, dict]:
    """What Figure 2 plots: per scenario, each repo's mean repo-feature SHAP contribution c
    against its actual slow rate y, with the Spearman correlation across repos."""
    out = {}
    for sc in ("A", "B"):
        c, y = transfer[f"c_{sc}"].to_numpy(), transfer[f"y_{sc}"].to_numpy()
        out[sc] = {"c": c, "y": y, "rho": float(spearmanr(c, y).statistic)}
    return out


def fingerprint_scatter(transfer: pd.DataFrame, out: Path) -> dict:
    """Figure 2: the Phase 6b transfer test, one point per repo."""
    data = scatter_data(transfer)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    titles = {"A": "Seen in training (Scenario A)", "B": "Held out (Scenario B)"}
    for ax, sc in zip(axes, ("A", "B")):
        d = data[sc]
        ax.scatter(d["c"], d["y"], s=18, color=COLOURS["FULL"], alpha=0.8)
        ax.set_title(f"{titles[sc]}: Spearman ρ = {d['rho']:+.2f}", fontsize=10)
        ax.set_xlabel("repo-level features' mean SHAP contribution (log-odds)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("repo's actual slow rate")
    fig.suptitle("The 8 repo-level features track each repo's slow rate more closely\n"
                 "when the repo was seen in training (one point per repo, 39 repos)", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, metadata=PNG_META)
    plt.close(fig)
    return data


def main(out_dir: Path = FIG) -> int:
    headline_transfer(load_runs(), out_dir / "headline_transfer.png")
    fingerprint_scatter(pd.read_csv(TRANSFER), out_dir / "fingerprint_scatter.png")
    print(f"wrote {out_dir / 'headline_transfer.png'} and {out_dir / 'fingerprint_scatter.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests, then render and look at the figures**

Run `python -m pytest tests/test_writeup_figures.py -q`. Expect **4 passed**.

Run `python writeup_figures.py`. It writes `figures/headline_transfer.png` and `figures/fingerprint_scatter.png`. **Open both and look at them.** Figure 1 must show six bars, with values printed at each bar's base, Scenario B's five fold values as dots, and the legend below the chart. Figure 2 must show two panels with ρ = +0.67 and ρ = +0.32 in their titles.

Run `python -m pytest tests -q`. Expect **244 passed**.

- [ ] **Step 5: Commit**

```bash
git add writeup_figures.py tests/test_writeup_figures.py figures/headline_transfer.png figures/fingerprint_scatter.png
git commit -m "feat(phase8): headline figures rendered from committed artifacts

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Mutation check (after the commit, restoring with git)**

For each mutation: apply it, run the named test and confirm it FAILS, then run `git checkout -- writeup_figures.py` and confirm `git status --porcelain` is clean.
- In `transfer_data`, change `"NO_LABEL_REPLAY": [r["auc_pr"] for r in by["NO_LABEL_REPLAY"]],` to use `by["FULL"]` → `test_transfer_data_is_phase4s_numbers`.
- In `scatter_data`, change `transfer[f"c_{sc}"]` to `transfer["c_A"]` → `test_scatter_data_is_the_transfer_table`.

---

### Task 2: The documents

**Files:**
- Create: `docs/data_collection.md`, `docs/REPORT.md`
- Rewrite: `README.md`

**Interfaces:**
- Consumes: the two figures from Task 1, which both documents link to.
- Produces: the documents Task 3's claims test checks. **Every claim string in Task 3's registry appears verbatim in them**, so transcribe exactly: a changed digit, spacing or dash breaks a claim.

- [ ] **Step 1: Move the current README into `docs/data_collection.md` — before overwriting it**

The current README describes Phase 1, the collection, and it must not be lost. Move it verbatim, re-basing only its two relative links, because the file moves from the repo root into `docs/`:

```bash
python - <<'PYEOF'
import io
old = io.open("README.md", encoding="utf-8").read()
assert old.startswith("# PRFlowPredict — Phase 1: dataset collection"), "README was already replaced"
old = (old.replace("](actionable_ml_project_blueprint.md)", "](../actionable_ml_project_blueprint.md)")
          .replace("](docs/data_dictionary.md)", "](data_dictionary.md)"))
header = ("> **Moved here from the project README during Phase 8.** It describes Phase 1, the data\n"
          "> collection, as it was built; relative links were re-based for this folder. The whole\n"
          "> project is summarised in the [README](../README.md) and the [report](REPORT.md).\n\n")
io.open("docs/data_collection.md", "w", encoding="utf-8", newline="\n").write(header + old)
print("docs/data_collection.md written")
PYEOF
```

- [ ] **Step 2: Rewrite `README.md` with exactly this content**

````markdown
# PRFlowPredict

Predicts, at the moment a GitHub pull request is opened, whether it will wait **more than
7 days** for its first human review, so a maintainer can see which open PRs are at risk of
stalling.

## The result

- **Within a project, it works.** Predicting PRs opened after a time cutoff, in repos it was
  trained on, the model's precision on each repo's 10 highest-risk PRs is
  0.769 [0.669, 0.856] (95% interval, resampling repos). For comparison, the trailing-rate
  baseline scores 0.585; it ranks PRs by the repo's own recent slow rate, and falls well
  below the model's interval.
- **On repos it has never seen, it only works through the repo's own history.** It beats the
  baseline there (AUC-PR 0.859 vs 0.821) while it can use features that replay the repo's own
  recent review record. Remove those four features and it falls below the baseline:
  0.763 vs 0.821. In this cohort, PR-level signal learned in some projects does not carry to
  others.
- **Why is only partly explained.** The models lean on the same features whether or not a
  repo was seen in training. A pre-registered test of one mechanism, the model recognising
  repos by their fixed attributes, came back partly supported (`PARTIAL_SHAP_ONLY`): the
  attribution pattern is there, but removing those attributes does not measurably help on
  unseen repos.

![AUC-PR on seen and unseen repos, with and without the repo's own history, against the trailing-rate baseline](figures/headline_transfer.png)

## Why the numbers can be trusted

- **No leakage by construction.** Every feature is computed by replaying each repo's history
  in time order, so a PR only ever sees what existed when it was opened. An independent
  brute-force recomputation of the replayed features matches them exactly.
- **Validity gates at every phase.** Each phase pre-registered checks that its results are
  valid (disjoint splits, no label in any feature, reproducible refits) and passed them before
  its results were read. The gates test validity, never success.
- **Negative results are the headline, not a footnote.** The cold-start failure, a null
  attribution result and a partly supported hypothesis are all reported as found.
- **Every result here is checked.** `tests/test_writeup_claims.py` recomputes each measured
  result in this README and in the report from committed artifacts, and fails if one is
  wrong or stale.

## Reproduce

From the committed state alone (no GitHub access needed):

```bash
pip install -r requirements.txt
python -m pytest tests -q        # includes the check of every number in this README
python writeup_figures.py        # regenerates both headline figures
```

The full pipeline needs a GitHub token and a multi-hour collection, because raw data is not
committed. In order:

```bash
gh auth login                    # the collector borrows gh's token; public_repo scope is enough
python collect_cohort.py         # ~4-5 h, resumable: raw GraphQL responses into data/raw/
python parse.py                  # offline: raw responses into data/processed/
python cohort_qc.py              # structural cohort checks
python eda_report.py             # labels, EDA and the Phase 2 gate
python features.py               # the leakage-safe feature table
python features.py --audit       # the Phase 3 gate
python tune.py                   # hyperparameters, on Scenario A training rows only
python experiment.py             # every model run, logged to data/experiments.csv
python report4.py                # Phase 4 results and gate
python report6.py                # Phase 6 SHAP attribution
python report6b.py               # Phase 6b pre-registered fingerprinting test
```

See [the report's reproducing section](docs/REPORT.md#10-reproducing-it) for what re-running
each step overwrites.

## Read more

- [The full write-up](docs/REPORT.md): data, leakage prevention, evaluation, results, the
  search for why transfer fails, fairness, limitations (about 15 minutes).
- Per-phase detail: [collection gate](docs/phase0_gate_results.md) ·
  [labels and EDA](docs/phase2_eda.md) · [features](docs/feature_dictionary.md) ·
  [modelling results](docs/phase4_results.md) · [SHAP interpretation](docs/phase6_interpretation.md) ·
  [the fingerprinting test](docs/phase6b_fingerprinting.md)
- [The original project plan](actionable_ml_project_blueprint.md) and
  [how the data was collected](docs/data_collection.md).

## Repo map

| Stage | Modules |
|---|---|
| Collection | `select_repos.py` · `gate_report.py` · `collect.py` · `collect_cohort.py` · `ghclient.py` · `queries.py` · `parse.py` · `test_resume.py` |
| Labels and cohort | `load.py` · `labels.py` · `cohort_qc.py` · `eda_report.py` |
| Features | `replay.py` · `features.py` · `featuresets.py` |
| Modelling | `splits.py` · `model.py` · `baseline.py` · `metrics.py` · `tune.py` · `experiment.py` · `tracking.py` · `report4.py` |
| Interpretation | `attribution.py` · `errors.py` · `fairness.py` · `report6.py` · `fingerprint.py` · `report6b.py` |
| Write-up | `writeup_figures.py` · `writeup_claims.py` |
````

- [ ] **Step 3: Create `docs/REPORT.md` with exactly this content**

````markdown
# PRFlowPredict: predicting which pull requests will stall

Every measured result in this report is recomputed from a committed artifact by
`tests/test_writeup_claims.py`, which fails if a number here is wrong or stale.

## 1. The problem, and why a rule is not enough

A pull request that waits a week for its first review is a small failure that compounds:
contributors lose context, rebase against a moving target, and some may not come back.
PRFlowPredict scores each PR **at the moment it is opened**, using only information that
existed at that moment, and ranks a repo's open PRs by their risk of waiting more than 168
hours (7 days) for a first human review.

A simple rule such as "flag anything open more than 3 days" ignores that normal wait time
varies enormously by repo, by author history and by current backlog. Three days is alarming
in one project and routine in another. A model can learn what "normal" means per context.

PR and code-review latency has a research literature (search terms: *pull request latency
prediction*, *code review time prediction*). This project's angle is deliberately cautious:
PRs that are never reviewed stay in the label rather than being dropped; within-project and
cold-start performance are reported separately, never just the flattering one; and a
baseline the model must beat is fixed in advance.

## 2. Data

**Cohort.** Public repos stratified by language (Python, TypeScript, Go) and star tier
(200-800, 800-3,000, 3,000-15,000), five per cell, drawn at random with a fixed seed from a
persisted candidate pool. Pre-registered structural checks, which never looked at the label,
kept 39 of the 45 repos collected: five were dominated by bot-authored PRs and one did not
match its language stratum. The modelling rows are human-authored PRs opened from 1 January
2024 to 30 June 2026: 38,462 modelling rows.

**Collection.** GitHub's GraphQL API, in two stages and two tiers.
- **Two stages.** Raw response bytes are written to disk verbatim, and parsed into Parquet
  by a separate offline step. A parse bug therefore costs a re-parse, never a re-scrape. That
  paid off once: a parse bug that turned legitimate zeros into missing values was found
  during feature engineering and fixed by re-parsing.
- **Two tiers.** Every PR in each repo's *entire* history is collected thinly, and PRs in the
  modelling window richly. The window is a row filter, not a collection filter, because
  backlog and author-history features need everything that happened before it.

**The label.** A PR is *slow* if no one other than its author, and no bot, reviews or
comments on it within 168 hours of creation. The primary definition (D5) also ignores
commenters whose relationship to the repo is `NONE`: a pilot on `anthropics/skills` showed
spam accounts commenting on outsiders' PRs, which would otherwise count as a review. The
label is 46.4% slow under D5 and 44.0% under D3, the variant that keeps those commenters; the
full five-definition table is in [the Phase 2 report](phase2_eda.md). More detail on the
collection design is in [how the data was collected](data_collection.md).

## 3. Leakage prevention

Leakage is the main way a project like this produces a result that looks good and means
nothing, so it is prevented structurally rather than checked for afterwards.

- **Chronological replay.** Features are computed by replaying each repo's events in time
  order. A PR's features are built from state that existed strictly before it was opened.
- **"Resolvable by t" for label-derived features.** A repo's trailing slow rate at time *t*
  may only use PRs whose own label was already knowable at *t*: created at least 168 hours
  earlier, or already reviewed. Using every earlier PR, the obvious rule, silently looks up to
  seven days into the future.
- **Snapshots versus point-in-time values.** Some API fields (final diff size, draft status,
  labels) describe the PR *now*, not when it was opened. Where possible the at-open value is
  reconstructed from the timeline; fidelity flags recording how well that worked are kept as
  columns and never used as features. A field shown to change after the fact (the author's
  association with the repo) is excluded.
- **An independent audit.** Every replayed feature was recomputed by a separate brute-force
  implementation on a seeded sample of 500 rows, largest difference 0.0, and five rows were
  checked by hand against live GitHub.

## 4. Evaluation design

- **Scenario A, within-project.** The same 39 repos, split in time: train on PRs opened
  before 1 January 2026, test on PRs opened after.
- **Scenario B, cold-start.** Leave-repos-out: five folds, each holding out about eight whole
  repos the model never sees in training. Every repo is held out exactly once.
- **Metrics.** Precision@10 per repo, the share of slow PRs among the 10 a maintainer would
  look at first, averaged over repos; and AUC-PR, which does not depend on how many
  candidates each repo has.
- **Baseline.** Rank each repo's PRs by the repo's own trailing 90-day slow rate, computed
  with the same resolvable-by-*t* rule. The model has to beat this, not just chance.
- **Intervals.** A cluster bootstrap that resamples whole repos (2,000 draws). The effective
  sample size for a claim about repos is the number of repos, not the number of PRs.
- **Model.** LightGBM, trained deterministically (fixed seeds, single thread). Hyperparameters
  were tuned with Optuna on Scenario A's training rows only, in time-ordered folds, reaching a
  cross-validated AUC-PR 0.907. Neither test set influenced tuning.

## 5. Results

| | Scenario A | Scenario B |
|---|---|---|
| P@10, model | 0.769 [0.669, 0.856] | 0.796 [0.614, 0.939] |
| P@10, trailing-rate baseline | 0.585 | 0.632 |
| AUC-PR, model | 0.906 | 0.859 |
| AUC-PR, trailing-rate baseline | 0.887 | 0.821 |

**Within a project, the model clearly works.** On Scenario A its precision@10 is
0.769 [0.669, 0.856], +0.185 over the baseline, and the baseline scores 0.585, outside the
interval. It also clears the base rate of 0.641, the precision a random pick would get.

**Do not read Scenario B's higher P@10 as cold-start being easy.** Scenario B's P@10 is higher
than Scenario A's, but that is a pool-size artifact. P@10 picks the top 10 *per repo*, and
Scenario A draws them from a median of 64 test PRs per repo, against 433 in Scenario B.
Picking 10 slow PRs out of a larger pool is easier. AUC-PR does not depend on pool size, so it
is the fair comparison across scenarios.

**Across projects, the advantage comes from the repo's own history.** The pre-registered
ablations remove one group of features at a time (AUC-PR):

| Features | Scenario A | Scenario B |
|---|---|---|
| FULL | 0.906 | 0.859 |
| NO_SNAPSHOT | 0.908 | 0.853 |
| NO_LABEL_REPLAY | 0.902 | 0.763 |
| PR_ONLY | 0.729 | 0.556 |
| trailing-rate baseline | 0.887 | 0.821 |

`NO_LABEL_REPLAY` removes the four features that replay the repo's own review record, such as
its trailing slow rate and the author's past slow rate in the repo. Within a project, the model
barely notices and still beats the baseline: PR-level features carry real signal. Across
projects, the same ablation drops it to 0.763 vs 0.821, below the baseline, in 4 of 5 folds.
With the history features the cold-start model beats the baseline (0.859 vs 0.821); without
them it does not. In this cohort, PR-level signal learned in some projects does not carry to
others.

## 6. Why doesn't it transfer?

![AUC-PR on seen and unseen repos, with and without the repo's own history](../figures/headline_transfer.png)

**Phase 6: it is not that the model relies on different features.** Exact TreeSHAP
attributions for both full models show the four history features carrying 60.9% in Scenario A
and 63.1% in Scenario B of the models' mean absolute attribution: essentially the same. A single feature,
the author's past slow rate in the repo, carries 47.3% and 46.8% respectively. The cold-start
model does not fail because it leans on different things. One caveat applies: this compares
the two full models, while the failure is specific to the ablation without history features.
See [the Phase 6 report](phase6_interpretation.md).

**Phase 6b: a pre-registered test of one mechanism, repo fingerprinting.** Without its history
features, might the model use eight repo-level attributes that are constant within a repo
(maintainer counts, whether it has a CODEOWNERS file, its CI workflow count and so on) to
*recognise* repos and replay their base rates? That would work on repos seen in training and
mean nothing on unseen ones. The hypothesis is post-hoc: it came from an exploratory look at
feature importance. It was then tested under a decision rule
[committed before any code](superpowers/specs/2026-09-24-phase6b-fingerprinting-design.md),
with two tests that both had to confirm it for "supported":

- **A SHAP transfer test.** For each repo, compare the attribution those eight features give it
  with its actual slow rate. On repos seen in training ρ_A = +0.669; on held-out repos
  ρ_B = +0.318. The difference is T = +0.351, 95% CI [+0.092, +0.616], which lies above zero:
  **confirms**.
- **An intervention.** Retrain without the eight features and measure the change in AUC-PR:
  Δ_A = -0.040 within a project, Δ_B = -0.004 on unseen repos. Fingerprinting predicts removal
  should hurt the seen repos more; the difference is I = +0.035, 95% CI [-0.043, +0.106], which
  contains zero: **inconclusive**.

The verdict is `PARTIAL_SHAP_ONLY`. The attribution pattern is there: the eight features carry
30.3% of mean |SHAP| in Scenario A and 40.7% in Scenario B, and their contribution tracks a
repo's real slow rate more closely when the repo was seen in training. But removing them does
not measurably help on unseen repos, so the pattern is not shown to be what costs cold-start
accuracy.

![Repo-level contribution against actual slow rate, seen and held-out repos](../figures/fingerprint_scatter.png)

Two caveats keep this from being read too strongly:
- **ρ_B was predicted to be about zero, and is not.** The observed ρ_B = +0.318 is not the ≈0
  the pre-registration predicted, but its descriptive bootstrap interval, about [0.00, 0.57],
  makes it only marginally distinguishable from it. That interval is descriptive, outside the
  pre-registered verdict.
- **The intervention's per-fold effects are unstable.** Per-fold changes on unseen repos range
  from +0.245 to -0.131, and I's interval contains zero: this data cannot distinguish a small
  intervention effect from none.

See [the Phase 6b report](phase6b_fingerprinting.md).

## 7. Fairness

A triage tool that systematically under-ranks newcomers' PRs would compound an existing
open-source problem. The question is whether the model is *differentially* pessimistic about
first-time contributors: the paired difference between the newcomer gap (predicted minus
actual slow rate) and the repeat-contributor gap, within the same repos.

- **Scenario A:** -0.003, 95% CI [-0.091, 0.090]. No distinguishable newcomer penalty.
- **Scenario B:** +0.064, 95% CI [-0.010, 0.143]. The interval contains zero, but its lower
  bound sits just below it: a marginal null, which neither confirms nor rules out a penalty.
  On its own, the newcomer slice in Scenario B *is* over-predicted as slow, at
  +0.067, 95% CI [0.002, 0.136].

## 8. What didn't work, and limitations

- **The cold-start bar was never measured.** The project plan set a Scenario B bar of a
  C-index above 0.65, to come from a survival model. That phase was not built, so the bar was
  never measured. The cold-start evidence here is the AUC-PR ablation above.
- **39 repos is a small sample** for any claim about projects in general. Cold-start results
  swing widely between folds: without the history features, per-fold AUC-PR runs
  from 0.507 to 0.965.
- **Snapshot repo attributes.** Maintainer counts, CODEOWNERS and similar features are 2026
  snapshots applied to PRs from 2024 onward, and star tiers are defined by 2026 star counts.
- **What was tried.** All 36 runs are logged in [data/experiments.csv](../data/experiments.csv),
  including the ablations that did not help: dropping snapshot features changed little
  anywhere, and PR-level features alone fall far below the baseline in both scenarios.
- **Lessons from the process.**
  - A test once overwrote a model file in a gitignored directory, where version control could
    not see it. It was restored and verified exactly, and those directories are now fenced off
    from tests. Verify gitignored artifacts by hash, never by version-control status.
  - Several tests were found whose assertions held whether or not the behaviour they named
    worked. Tests are now checked by deliberately breaking the code they cover and confirming
    they fail.

## 9. How it was built

- **Validity gates, pre-registered.** Every phase wrote its pass criteria down before its
  results were read: disjoint splits, no label or key column in any feature set, reproducible
  refits, exact attribution additivity. The gates test that a result is *valid*, never that it
  is good. A disappointing result that passes its gate is reported as found.
- **A pre-registered hypothesis test.** Phase 6b's decision rule, nine outcomes from two tests,
  was committed before any code existed, and the result was reported in whichever cell it
  landed.
- **Mutation-checked tests.** Tests are verified by breaking the code they cover and confirming
  they fail.
- **Independent review** of every change before it was merged.
- **Reproducible from committed artifacts.** Every figure and every number in this report can
  be regenerated from files in the repository.

## 10. Reproducing it

From the committed state, with no GitHub access:

```bash
pip install -r requirements.txt
python -m pytest tests -q        # includes the check of every number in this report
python writeup_figures.py        # regenerates both headline figures
```

The full pipeline needs a GitHub token and a multi-hour collection, because raw data is not
committed:

```bash
gh auth login
python collect_cohort.py         # ~4-5 h, resumable
python parse.py
python cohort_qc.py
python eda_report.py
python features.py
python features.py --audit
python tune.py
python experiment.py
python report4.py
python report6.py
python report6b.py
```

Re-running some steps overwrites committed results:
- `experiment.py` and `report6b.py` append their runs to `data/experiments.csv` again, creating
  duplicate rows.
- `report6.py` regenerates `docs/phase6_interpretation.md`, which replaces its hand-written
  error analysis.

The original plan is [the project blueprint](../actionable_ml_project_blueprint.md).
````

- [ ] **Step 4: Check before committing**

Run `python -m pytest tests -q`. Expect **244 passed**; nothing tests the documents until Task 3.

Then confirm that every relative link and image in the three documents resolves:

```bash
python - <<'PYEOF'
import re, pathlib
for doc in ("README.md", "docs/REPORT.md", "docs/data_collection.md"):
    p = pathlib.Path(doc)
    for t in re.findall(r"!?\[[^\]]*\]\(([^)\s#]+)", p.read_text(encoding="utf-8")):
        if not t.startswith("http") and not (p.parent / t).exists():
            print("BROKEN", doc, t)
print("link check done")
PYEOF
```

It must print no `BROKEN` lines.

Finally, confirm the voice constraint. This search must find nothing:
`grep -n -i -E "claude|\bAI\b|assistant|LLM|\bI (wrote|built|made)" README.md docs/REPORT.md`

- [ ] **Step 5: Commit**

```bash
git add README.md docs/REPORT.md docs/data_collection.md
git commit -m "docs(phase8): portfolio README and full report; Phase 1 notes moved to docs/

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: The claims test

**Files:**
- Create: `writeup_claims.py`, `tests/test_writeup_claims.py`

**Interfaces:**
- Consumes: the committed data files and phase documents (the source hierarchy in spec §7), `fingerprint.verdict` / `fingerprint.outcome`, `featuresets.LABEL_REPLAY`, `splits.SEED`, and Task 2's documents.
- Produces:
  - `writeup_claims.Claim(name, render, expected, docs, sources)`
  - the registry `CLAIMS` (42 claims)
  - `REQUIRED_PHRASES` and `FORBIDDEN_PATTERNS`
  - `_parse(rel, pattern) -> tuple[float, ...]`, which demands exactly one match
  - `ROOT`, which the tests read **at call time** so they can monkeypatch it

- [ ] **Step 1: Write the failing test**

`tests/test_writeup_claims.py`:

```python
"""Every result the write-up cites, checked against its committed artifact and its document.

See writeup_claims.py for the registry, and the Phase 8 spec (section 7) for the rules."""
import re
import subprocess

import pytest

import writeup_claims as wc

DOCS = (wc.README, wc.REPORT)
LINKED_DOCS = DOCS + ("docs/data_collection.md",)
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")


def _doc(rel: str) -> str:
    """Whitespace-normalised, so a claim that Markdown wraps across two lines still counts.
    Reads wc.ROOT at call time so tests can point it at a temporary directory."""
    return re.sub(r"\s+", " ", (wc.ROOT / rel).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# the claims, both directions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("claim", wc.CLAIMS, ids=lambda c: c.name)
def test_claim_recomputes_to_its_registered_string(claim):
    """Artifact direction: if a committed artifact changes, the registered string goes stale."""
    assert claim.render() == claim.expected


@pytest.mark.parametrize("claim", wc.CLAIMS, ids=lambda c: c.name)
def test_claim_appears_in_every_listed_document(claim):
    """Document direction: a mistyped, missing or stale number in the prose fails here."""
    for doc in claim.docs:
        assert claim.expected in _doc(doc), f"{claim.expected!r} is missing from {doc}"


def test_claim_names_are_unique():
    names = [c.name for c in wc.CLAIMS]
    assert len(names) == len(set(names))


def test_every_claim_source_is_committed():
    """A reader who clones the repo must be able to check every number (spec section 7.3)."""
    out = subprocess.run(["git", "ls-files"], cwd=wc.ROOT, capture_output=True, text=True, check=True)
    tracked = set(out.stdout.splitlines())
    untracked = sorted({s for c in wc.CLAIMS for s in c.sources} - tracked)
    assert not untracked, f"claims read files a clone would not have: {untracked}"


def test_parse_demands_exactly_one_match(tmp_path, monkeypatch):
    """If a regenerated phase document rewords a sentence, the claim must fail loudly rather
    than silently match a different sentence or nothing."""
    (tmp_path / "d.md").write_text("value is 1.5\nvalue is 2.5\n", encoding="utf-8")
    monkeypatch.setattr(wc, "ROOT", tmp_path)
    with pytest.raises(LookupError):
        wc._parse("d.md", r"value is ([\d.]+)")          # two matches
    with pytest.raises(LookupError):
        wc._parse("d.md", r"total is ([\d.]+)")          # no match
    assert wc._parse("d.md", r"value is (2\.[\d]+)") == (2.5,)


def test_wrapped_claim_still_counts(tmp_path, monkeypatch):
    (tmp_path / "x.md").write_text("a result of 0.769\n[0.669, 0.856] here\n", encoding="utf-8")
    monkeypatch.setattr(wc, "ROOT", tmp_path)
    assert "0.769 [0.669, 0.856]" in _doc("x.md")


# ---------------------------------------------------------------------------
# honesty constraints that carry no number (spec section 9)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("doc,phrase", wc.REQUIRED_PHRASES, ids=lambda v: str(v)[:30])
def test_required_honesty_phrase_present(doc, phrase):
    assert phrase in _doc(doc)


@pytest.mark.parametrize("pattern", wc.FORBIDDEN_PATTERNS)
@pytest.mark.parametrize("doc", DOCS)
def test_forbidden_phrasing_absent(doc, pattern):
    hit = re.search(pattern, _doc(doc), flags=re.IGNORECASE)
    assert hit is None, f"{doc} says {hit.group(0)!r}"


# ---------------------------------------------------------------------------
# a reader clicking around must not hit a dead link
# ---------------------------------------------------------------------------

def _slug(heading: str) -> str:
    """GitHub's heading anchor: lower-case, punctuation dropped, spaces to hyphens."""
    s = re.sub(r"[^\w\s-]", "", heading.strip().lower())
    return re.sub(r"\s", "-", s)


@pytest.mark.parametrize("doc", LINKED_DOCS)
def test_relative_links_resolve(doc):
    base = (wc.ROOT / doc).parent
    broken = []
    for target in LINK.findall((wc.ROOT / doc).read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        path, _, anchor = target.partition("#")
        full = (base / path).resolve() if path else (wc.ROOT / doc).resolve()
        if not full.exists():
            broken.append(target)
            continue
        if anchor:
            heads = [ln.lstrip("#").strip() for ln in full.read_text(encoding="utf-8").splitlines()
                     if ln.startswith("#")]
            if anchor not in {_slug(h) for h in heads}:
                broken.append(target)
    assert not broken, f"{doc}: broken links {broken}"
```

- [ ] **Step 2: Run it and confirm it fails**

Run `python -m pytest tests/test_writeup_claims.py -q`. Expected: FAIL with `ModuleNotFoundError: No module named 'writeup_claims'`.

- [ ] **Step 3: Implement `writeup_claims.py`**

```python
"""The numbers the write-up cites, each recomputed from a committed artifact.

tests/test_writeup_claims.py checks every claim two ways: the recomputed value must render to
the registered string (so a stale artifact or a changed format fails), and that string must
appear in every document the claim lists (so a mistyped or missing number fails).

Sources follow the Phase 8 spec, section 7: a committed data file where one holds the value,
otherwise the committed generated phase document, parsed in its labelled context. Never a
gitignored artifact -- a reader who clones the repo cannot check those."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import featuresets as fs
import fingerprint as fp
import splits

ROOT = Path(__file__).parent
README = "README.md"
REPORT = "docs/REPORT.md"
BOTH = (README, REPORT)

P4_RUNS = "data/phase4_runs.json"
P4_DOC = "docs/phase4_results.md"
P4_PARAMS = "data/phase4_params.json"
P2_GATE = "data/phase2_gate.json"
P2_DOC = "docs/phase2_eda.md"
P3_GATE = "data/phase3_gate.json"
KEPT = "data/cohort/kept.json"
P6_IMP = "data/phase6_importance_{}.csv"
P6_FAIR = "data/phase6_fairness.csv"
P6_DOC = "docs/phase6_interpretation.md"
P6B_DOC = "docs/phase6b_fingerprinting.md"
P6B_TRANSFER = "data/phase6b_transfer.csv"
P6B_INTERVENTION = "data/phase6b_intervention.csv"
P6B_RELIANCE = "data/phase6b_reliance.csv"
EXPERIMENTS = "data/experiments.csv"


@dataclass(frozen=True)
class Claim:
    name: str
    render: Callable[[], str]      # recomputes the value from committed artifacts, then formats it
    expected: str                  # the exact string it must render as
    docs: tuple[str, ...]          # documents the string must appear in
    sources: tuple[str, ...]       # committed paths it reads; the test checks each is git-tracked


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def _text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


@cache
def _runs() -> tuple[dict, ...]:
    return tuple(json.loads(_text(P4_RUNS))["runs"])


def _p4(scenario: str, featureset: str, key: str) -> float:
    """Phase 4's reporting convention: Scenario B is the mean over its five folds."""
    vals = [r[key] for r in _runs() if r["scenario"] == scenario and r["featureset"] == featureset]
    if not vals:
        raise LookupError(f"no {scenario}/{featureset} runs in {P4_RUNS}")
    return float(np.mean(vals))


def _gate(rel: str, prefix: str):
    hits = [c["value"] for c in json.loads(_text(rel)) if c["check"].startswith(prefix)]
    if len(hits) != 1:
        raise LookupError(f"{rel}: {len(hits)} checks start with {prefix!r}")
    return hits[0]


def _parse(rel: str, pattern: str) -> tuple[float, ...]:
    """Parse numbers from a committed generated document, in their labelled context. Exactly
    one match is required, so a pattern that silently matched the wrong sentence fails."""
    hits = re.findall(pattern, _text(rel), flags=re.MULTILINE)
    if len(hits) != 1:
        raise LookupError(f"{rel}: pattern matched {len(hits)} times: {pattern!r}")
    hit = hits[0] if isinstance(hits[0], tuple) else (hits[0],)
    return tuple(float(x) for x in hit)


def _f3(x: float) -> str:
    return f"{x:.3f}"


def _s3(x: float) -> str:
    return f"{x:+.3f}"


def _ci(lo: float, hi: float, fmt=_f3) -> str:
    return f"[{fmt(lo)}, {fmt(hi)}]"


# ---------------------------------------------------------------------------
# Phase 6b values
# ---------------------------------------------------------------------------

@cache
def _transfer() -> pd.DataFrame:
    return pd.read_csv(ROOT / P6B_TRANSFER)


def _rho(s: str) -> float:
    t = _transfer()
    return float(spearmanr(t[f"c_{s}"], t[f"y_{s}"]).statistic)


@cache
def _intervention() -> pd.DataFrame:
    return pd.read_csv(ROOT / P6B_INTERVENTION)


def _delta(s: str) -> float:
    """Delta_s in Phase 4's convention: B is the mean over its folds."""
    iv = _intervention()
    return float(iv.loc[iv["scenario"] == s, "delta"].mean())


def _t_interval() -> tuple[float, float, float]:
    """T and its interval as the Phase 6b document states them, cross-checked against a
    recomputation of T from the committed per-repo table."""
    point, lo, hi = _parse(P6B_DOC, r"ρ_A − ρ_B = ([+-][\d.]+), 95% CI \[([+-][\d.]+), ([+-][\d.]+)\]: \*\*")
    if _s3(point) != _s3(_rho("A") - _rho("B")):
        raise ValueError(f"T in {P6B_DOC} ({point}) disagrees with {P6B_TRANSFER}")
    return point, lo, hi


def _i_interval() -> tuple[float, float, float]:
    point, lo, hi = _parse(P6B_DOC, r"Δ_B − Δ_A = ([+-][\d.]+), 95% CI \[([+-][\d.]+), ([+-][\d.]+)\]: \*\*")
    if _s3(point) != _s3(_delta("B") - _delta("A")):
        raise ValueError(f"I in {P6B_DOC} ({point}) disagrees with {P6B_INTERVENTION}")
    return point, lo, hi


def _verdict() -> str:
    _, tlo, thi = _t_interval()
    _, ilo, ihi = _i_interval()
    return fp.verdict(fp.outcome(tlo, thi), fp.outcome(ilo, ihi))


def _rho_b_descriptive_ci() -> tuple[float, float]:
    """DESCRIPTIVE only, outside the pre-registered verdict: a seeded bootstrap over the 39
    held-out repos' (c, y) pairs, the project's usual 2,000 draws and 95%."""
    t = _transfer()
    c, y, n = t["c_B"].to_numpy(), t["y_B"].to_numpy(), len(t)
    rng = np.random.default_rng(splits.SEED)
    draws = [spearmanr(c[d], y[d]).statistic for d in (rng.choice(n, size=n, replace=True) for _ in range(2000))]
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def _reliance(col: str) -> float:
    rel = pd.read_csv(ROOT / P6B_RELIANCE)
    return float(rel.loc[rel["is_repo_feature"], col].sum())


def _fair_slice(scenario: str) -> tuple[float, float, float]:
    f = pd.read_csv(ROOT / P6_FAIR)
    row = f[(f["scenario"] == scenario) & (f["slice"] == "is_first_pr_here") & (f["level"] == "first-time")]
    if len(row) != 1:
        raise LookupError(f"{P6_FAIR}: {len(row)} first-time rows for scenario {scenario}")
    r = row.iloc[0]
    return float(r["gap"]), float(r["gap_ci_lo"]), float(r["gap_ci_hi"])


def _share(scenario: str, features) -> float:
    imp = pd.read_csv(ROOT / P6_IMP.format(scenario))
    return float(imp.loc[imp["feature"].isin(features), "share"].sum())


# ---------------------------------------------------------------------------
# table rows: a whole row is the claim, so a number in the wrong row fails
# ---------------------------------------------------------------------------

def _main_row(label: str) -> str:
    if label == "P@10, model":
        cell = lambda s: f"{_f3(_p4(s, 'FULL', 'precision_at_10'))} {_ci(_p4(s, 'FULL', 'p10_ci_lo'), _p4(s, 'FULL', 'p10_ci_hi'))}"
    elif label == "P@10, trailing-rate baseline":
        cell = lambda s: _f3(_p4(s, "FULL", "baseline_p10"))
    elif label == "AUC-PR, model":
        cell = lambda s: _f3(_p4(s, "FULL", "auc_pr"))
    elif label == "AUC-PR, trailing-rate baseline":
        cell = lambda s: _f3(_p4(s, "FULL", "baseline_auc_pr"))
    else:
        raise KeyError(label)
    return f"| {label} | {cell('A')} | {cell('B')} |"


def _ablation_row(featureset: str) -> str:
    if featureset == "trailing-rate baseline":
        a, b = _p4("A", "FULL", "baseline_auc_pr"), _p4("B", "FULL", "baseline_auc_pr")
    else:
        a, b = _p4("A", featureset, "auc_pr"), _p4("B", featureset, "auc_pr")
    return f"| {featureset} | {_f3(a)} | {_f3(b)} |"


def _folds_below_baseline() -> int:
    nlr = sorted((r for r in _runs() if r["scenario"] == "B" and r["featureset"] == "NO_LABEL_REPLAY"),
                 key=lambda r: r["fold"])
    return sum(r["auc_pr"] <= r["baseline_auc_pr"] for r in nlr)


def _nlr_b_fold_range() -> tuple[float, float]:
    v = [r["auc_pr"] for r in _runs() if r["scenario"] == "B" and r["featureset"] == "NO_LABEL_REPLAY"]
    return min(v), max(v)


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

CLAIMS: tuple[Claim, ...] = (
    # --- headline, in both documents
    Claim("a_p10", lambda: f"{_f3(_p4('A', 'FULL', 'precision_at_10'))} {_ci(_p4('A', 'FULL', 'p10_ci_lo'), _p4('A', 'FULL', 'p10_ci_hi'))}",
          "0.769 [0.669, 0.856]", BOTH, (P4_RUNS,)),
    Claim("a_baseline_p10", lambda: f"baseline scores {_f3(_p4('A', 'FULL', 'baseline_p10'))}",
          "baseline scores 0.585", BOTH, (P4_RUNS,)),
    Claim("b_full_vs_baseline", lambda: f"{_f3(_p4('B', 'FULL', 'auc_pr'))} vs {_f3(_p4('B', 'FULL', 'baseline_auc_pr'))}",
          "0.859 vs 0.821", BOTH, (P4_RUNS,)),
    Claim("b_nlr_vs_baseline", lambda: f"{_f3(_p4('B', 'NO_LABEL_REPLAY', 'auc_pr'))} vs {_f3(_p4('B', 'FULL', 'baseline_auc_pr'))}",
          "0.763 vs 0.821", BOTH, (P4_RUNS,)),
    Claim("verdict_6b", _verdict, "PARTIAL_SHAP_ONLY", BOTH, (P6B_DOC, P6B_TRANSFER, P6B_INTERVENTION)),

    # --- data and design
    Claim("cohort_kept", lambda: (lambda k: f"{len(k['kept'])} of the {len(k['repos'])}")(json.loads(_text(KEPT))),
          "39 of the 45", (REPORT,), (KEPT,)),
    Claim("d5_rate", lambda: f"{_gate(P2_GATE, 'D5 global is_slow'):.1%} slow under D5",
          "46.4% slow under D5", (REPORT,), (P2_GATE,)),
    Claim("d3_rate", lambda: f"{_parse(P2_DOC, r'^\| D3 \| ([\d.]+) \|')[0]:.1%} under D3",
          "44.0% under D3", (REPORT,), (P2_DOC,)),
    Claim("modelling_rows", lambda: f"{_gate(P3_GATE, 'row count')[0]:,} modelling rows",
          "38,462 modelling rows", (REPORT,), (P3_GATE,)),
    Claim("audit_rows", lambda: (lambda v: f"{v['n']} rows, largest difference {v['max_abs_diff']}")(_gate(P3_GATE, "replay audit")),
          "500 rows, largest difference 0.0", (REPORT,), (P3_GATE,)),
    Claim("tune_cv", lambda: f"cross-validated AUC-PR {_f3(json.loads(_text(P4_PARAMS))['cv_auc_pr'])}",
          "cross-validated AUC-PR 0.907", (REPORT,), (P4_PARAMS,)),

    # --- results tables
    *(Claim(f"main_row::{lab}", (lambda lab=lab: _main_row(lab)), exp, (REPORT,), (P4_RUNS,)) for lab, exp in (
        ("P@10, model", "| P@10, model | 0.769 [0.669, 0.856] | 0.796 [0.614, 0.939] |"),
        ("P@10, trailing-rate baseline", "| P@10, trailing-rate baseline | 0.585 | 0.632 |"),
        ("AUC-PR, model", "| AUC-PR, model | 0.906 | 0.859 |"),
        ("AUC-PR, trailing-rate baseline", "| AUC-PR, trailing-rate baseline | 0.887 | 0.821 |"),
    )),
    *(Claim(f"ablation_row::{f}", (lambda f=f: _ablation_row(f)), exp, (REPORT,), (P4_RUNS,)) for f, exp in (
        ("FULL", "| FULL | 0.906 | 0.859 |"),
        ("NO_SNAPSHOT", "| NO_SNAPSHOT | 0.908 | 0.853 |"),
        ("NO_LABEL_REPLAY", "| NO_LABEL_REPLAY | 0.902 | 0.763 |"),
        ("PR_ONLY", "| PR_ONLY | 0.729 | 0.556 |"),
        ("trailing-rate baseline", "| trailing-rate baseline | 0.887 | 0.821 |"),
    )),

    # --- results prose
    Claim("a_advantage", lambda: f"{_s3(_p4('A', 'FULL', 'precision_at_10') - _p4('A', 'FULL', 'baseline_p10'))} over the baseline",
          "+0.185 over the baseline", (REPORT,), (P4_RUNS,)),
    Claim("a_base_rate", lambda: f"base rate of {_f3(_p4('A', 'FULL', 'base_rate_p10'))}",
          "base rate of 0.641", (REPORT,), (P4_RUNS,)),
    Claim("pool_a", lambda: f"a median of {_parse(P4_DOC, r'^\| A \| \d+ \| \d+ \| ([\d.]+) \|')[0]:.0f} test PRs",
          "a median of 64 test PRs", (REPORT,), (P4_DOC,)),
    Claim("pool_b", lambda: f"against {_parse(P4_DOC, r'^\| B \| \d+ \| \d+ \| ([\d.]+) \|')[0]:.0f} in Scenario B",
          "against 433 in Scenario B", (REPORT,), (P4_DOC,)),
    Claim("nlr_folds_below", lambda: f"{_folds_below_baseline()} of 5 folds",
          "4 of 5 folds", (REPORT,), (P4_RUNS,)),
    Claim("nlr_b_fold_range", lambda: (lambda lo, hi: f"from {_f3(lo)} to {_f3(hi)}")(*_nlr_b_fold_range()),
          "from 0.507 to 0.965", (REPORT,), (P4_RUNS,)),

    # --- Phase 6
    Claim("lr_share_a", lambda: f"{_share('A', fs.LABEL_REPLAY):.1%} in Scenario A",
          "60.9% in Scenario A", (REPORT,), (P6_IMP.format("A"),)),
    Claim("lr_share_b", lambda: f"{_share('B', fs.LABEL_REPLAY):.1%} in Scenario B",
          "63.1% in Scenario B", (REPORT,), (P6_IMP.format("B"),)),
    Claim("author_prior", lambda: f"{_share('A', ['author_prior_slow_rate_here']):.1%} and {_share('B', ['author_prior_slow_rate_here']):.1%}",
          "47.3% and 46.8%", (REPORT,), (P6_IMP.format("A"), P6_IMP.format("B"))),

    # --- Phase 6b
    Claim("rho_a", lambda: f"ρ_A = {_s3(_rho('A'))}", "ρ_A = +0.669", (REPORT,), (P6B_TRANSFER,)),
    Claim("rho_b", lambda: f"ρ_B = {_s3(_rho('B'))}", "ρ_B = +0.318", (REPORT,), (P6B_TRANSFER,)),
    Claim("t_6b", lambda: (lambda p, lo, hi: f"T = {_s3(p)}, 95% CI {_ci(lo, hi, _s3)}")(*_t_interval()),
          "T = +0.351, 95% CI [+0.092, +0.616]", (REPORT,), (P6B_DOC, P6B_TRANSFER)),
    Claim("i_6b", lambda: (lambda p, lo, hi: f"I = {_s3(p)}, 95% CI {_ci(lo, hi, _s3)}")(*_i_interval()),
          "I = +0.035, 95% CI [-0.043, +0.106]", (REPORT,), (P6B_DOC, P6B_INTERVENTION)),
    Claim("delta_a", lambda: f"Δ_A = {_s3(_delta('A'))}", "Δ_A = -0.040", (REPORT,), (P6B_INTERVENTION,)),
    Claim("delta_b", lambda: f"Δ_B = {_s3(_delta('B'))}", "Δ_B = -0.004", (REPORT,), (P6B_INTERVENTION,)),
    Claim("fold_delta_range", lambda: (lambda iv: f"from {_s3(iv.delta.max())} to {_s3(iv.delta.min())}")(_intervention()[_intervention().scenario == "B"]),
          "from +0.245 to -0.131", (REPORT,), (P6B_INTERVENTION,)),
    Claim("reliance", lambda: f"{_reliance('share_a'):.1%} of mean |SHAP| in Scenario A and {_reliance('share_b'):.1%} in Scenario B",
          "30.3% of mean |SHAP| in Scenario A and 40.7% in Scenario B", (REPORT,), (P6B_RELIANCE,)),
    Claim("rho_b_descriptive_ci", lambda: (lambda lo, hi: f"about [{lo:.2f}, {hi:.2f}]")(*_rho_b_descriptive_ci()),
          "about [0.00, 0.57]", (REPORT,), (P6B_TRANSFER,)),

    # --- fairness
    Claim("fair_a", lambda: (lambda p, lo, hi: f"{_s3(p)}, 95% CI {_ci(lo, hi)}")(*_parse(
        P6_DOC, r"\*\*Scenario A: no distinguishable newcomer gap\.\*\* The difference is ([+-]?[\d.]+) with CI \[([+-]?[\d.]+), ([+-]?[\d.]+)\]")),
          "-0.003, 95% CI [-0.091, 0.090]", (REPORT,), (P6_DOC,)),
    Claim("fair_b", lambda: (lambda p, lo, hi: f"{_s3(p)}, 95% CI {_ci(lo, hi)}")(*_parse(
        P6_DOC, r"paired difference \(([+-]?[\d.]+), CI \[([+-]?[\d.]+), ([+-]?[\d.]+)\]")),
          "+0.064, 95% CI [-0.010, 0.143]", (REPORT,), (P6_DOC,)),
    Claim("fair_b_slice", lambda: (lambda g, lo, hi: f"{_s3(g)}, 95% CI {_ci(lo, hi)}")(*_fair_slice("B")),
          "+0.067, 95% CI [0.002, 0.136]", (REPORT,), (P6_FAIR,)),

    # --- what was tried
    Claim("n_runs", lambda: f"{len(pd.read_csv(ROOT / EXPERIMENTS))} runs", "36 runs", (REPORT,), (EXPERIMENTS,)),
)

# Honesty constraints that carry no number (spec section 9): phrases that must appear, and
# phrasings that must not. The forbidden list guards section 9.7 -- never upgrade 6b.
REQUIRED_PHRASES: tuple[tuple[str, str], ...] = (
    (REPORT, "bar was never measured"),
    (REPORT, "C-index above 0.65"),
    (REPORT, "pool-size artifact"),
    (REPORT, "post-hoc"),
    (REPORT, "only marginally distinguishable"),
    (REPORT, "cannot distinguish a small intervention effect from none"),
    (REPORT, "marginal null"),
    (REPORT, "small sample"),
)
FORBIDDEN_PATTERNS: tuple[str, ...] = (
    r"fingerprinting (is|was) (supported|confirmed)",
    r"confirm(s|ed)? (repo )?fingerprinting",
    r"\bprov(e|es|ed|en)\b",
)
```

- [ ] **Step 4: Run the tests**

Run `python -m pytest tests/test_writeup_claims.py -q`. Expect **105 passed**:
- 42 claims recomputed
- 42 claims found in their documents
- 1 for unique names, 1 for committed sources, 1 parse guard, 1 wrapped-line check
- 8 required phrases
- 6 forbidden-phrasing checks
- 3 link checks

Run `python -m pytest tests -q`. Expect **349 passed**, pristine.

If a claim fails, **fix the document, not the registry**, unless the registry's `expected` string disagrees with what `render()` computes from the committed artifact. That would mean the plan's text is wrong; report it rather than editing either side to match.

- [ ] **Step 5: Commit**

```bash
git add writeup_claims.py tests/test_writeup_claims.py
git commit -m "test(phase8): every cited result checked against its artifact and its document

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Mutation check (after the commit, restoring with git)**

First record `sha256sum data/phase4_runs.json docs/phase6b_fingerprinting.md`. For each mutation: apply it, run the named test and confirm it FAILS, then run `git checkout -- <file>` and confirm `git status --porcelain` is clean. At the end, re-run `sha256sum` and confirm both hashes are unchanged.

**Document direction:**
- `docs/REPORT.md`: `0.763 vs 0.821` → `0.736 vs 0.821` → `test_claim_appears_in_every_listed_document`
- `README.md`: `PARTIAL_SHAP_ONLY` → `SUPPORTED` → the same test
- `docs/REPORT.md`: `| NO_LABEL_REPLAY | 0.902 | 0.763 |` → `| NO_LABEL_REPLAY | 0.763 | 0.902 |`, swapping values within a row → the same test
- `docs/REPORT.md`: replace **every** `never measured` with `not measured` → `test_required_honesty_phrase_present`
- `docs/REPORT.md`: `The verdict is `PARTIAL_SHAP_ONLY`.` → `… : fingerprinting is supported.` → `test_forbidden_phrasing_absent`
- `README.md`: `(docs/REPORT.md):` → `(docs/REPORTS.md):` → `test_relative_links_resolve`
- `docs/REPORT.md`: `## 10. Reproducing it` → `## 10. Reproduce`, which breaks the README's anchor → `test_relative_links_resolve`

**Artifact direction:**
- `data/phase4_runs.json`: the first `"precision_at_10": 0.7692307692307693` → `0.7792307692307693` → `test_claim_recomputes_to_its_registered_string`
- `docs/phase6b_fingerprinting.md`: `ρ_A − ρ_B = +0.351` → `+0.451`. This trips the T cross-check against the recomputation → the same test.
- `writeup_claims.py`: point the `a_p10` claim's `sources` at `"data/predictions/A_FULL_fold0.parquet"` → `test_every_claim_source_is_committed`

---

### Task 4: Blueprint status corrections

**Files:**
- Modify: `actionable_ml_project_blueprint.md`, status only. None of the original plan text is rewritten.

- [ ] **Step 1: Apply the six replacements with this script**

The script asserts that each old string matches exactly once before it writes anything. The 6b row uses the project's own per-test outcome labels from `fingerprint.outcome` (`confirms`, `inconclusive`). That keeps spec §9.7: the verdict itself is never restated as "supported" or "confirmed".

```bash
python - <<'PYEOF'
import io
P = "actionable_ml_project_blueprint.md"
t = io.open(P, encoding="utf-8").read()
R = [
    # 1. explain the new marker, under the existing revision note
    ("> Original text is preserved in git history.\n",
     "> Original text is preserved in git history.\n>\n"
     "> **Status updated 2026-09-30** after Phase 8. Changes are marked `[R3]` inline.\n"),
    # 2. Phase 5: not built, so the C-index bar was never measured
    ("| 5 (week 5) | Survival model, C-index, calibration check | Nice-to-have (depth section) — cut first if time runs short |",
     "| 5 (week 5) | `[R3]` **Not built.** Survival model, C-index, calibration check. So §4's Scenario B bar "
     "(C-index > 0.65) was never measured. | Nice-to-have (depth section) — cut |"),
    # 3. Phase 6 done, plus a new 6b row directly under it
    ("| 6 (week 6) | SHAP feature attribution, manual error analysis on worst 50 predictions | Must-have |",
     "| 6 | `[R3]` **Done 2026-09-23.** SHAP attribution on both FULL models: label-replay share 60.9% (A) vs "
     "63.1% (B), so differential feature reliance does not explain the cold-start gap. Manual error analysis "
     "of the worst 50 predictions; newcomer-fairness check. `docs/phase6_interpretation.md` | Must-have — done |\n"
     "| 6b | `[R3]` **Done 2026-09-27.** Pre-registered repo-fingerprinting test, verdict `PARTIAL_SHAP_ONLY`: "
     "the SHAP transfer test's outcome is `confirms`, the intervention's is `inconclusive`. "
     "`docs/phase6b_fingerprinting.md` | Added — done |"),
    # 4. Phase 7: not built, still open
    ("| 7 (week 7) | Streamlit demo: pick a repo, see open PRs ranked by risk | Nice-to-have but high payoff for the demo |",
     "| 7 (week 7) | `[R3]` **Not built (open).** Streamlit demo: pick a repo, see open PRs ranked by risk "
     "| Nice-to-have but high payoff for the demo |"),
    # 5. Phase 8 done
    ("| 8 | Write-up | Must-have |",
     "| 8 | `[R3]` **Done 2026-09-30.** Write-up: `README.md` and `docs/REPORT.md`, every cited result checked "
     "by `tests/test_writeup_claims.py` | Must-have — done |"),
    # 6. what Phases 6 and 6b actually found
    ("Phase 6's SHAP work should explain why.",
     "Phase 6's SHAP work should explain why. `[R3]` It explained part of it: see the Phase 6 and 6b rows above."),
]
for old, new in R:
    assert t.count(old) == 1, f"expected exactly one match for {old[:60]!r}, found {t.count(old)}"
    t = t.replace(old, new)
io.open(P, "w", encoding="utf-8", newline="\n").write(t)
print("blueprint: 6 replacements applied")
PYEOF
```

- [ ] **Step 2: Verify only status changed**

Run `git diff --stat actionable_ml_project_blueprint.md`. Expect `1 file changed, 8 insertions(+), 5 deletions(-)`. Then read `git diff actionable_ml_project_blueprint.md`: only the six places above may change.

Run `python -m pytest tests -q`. Expect **349 passed**.

- [ ] **Step 3: Commit**

```bash
git add actionable_ml_project_blueprint.md
git commit -m "docs(phase8): blueprint status -- Phases 5-8 as built

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

## Verification (spec §10)

1. `python -m pytest tests -q` gives **349 passed**, pristine. Every mandated mutation in Tasks 1 and 3 was caught and restored with `git checkout`.
2. `README.md` and `docs/REPORT.md` contain every claim string and every required honesty phrase, and no forbidden phrasing, AI mention or dead link.
3. `python writeup_figures.py` regenerates both figures, byte-identical to the committed PNGs.
4. No phase document, phase generator or gitignored artifact was touched, and `data/experiments.csv` still has 36 rows.

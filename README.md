# PRFlowPredict

Predicts, at the moment a GitHub pull request is opened, whether it will wait **more than
7 days** for its first human review, so a maintainer can see which open PRs are at risk of
stalling.

## The result

- **Within a project, it works.** Predicting PRs opened after a time cutoff, in repos it was
  trained on, the model's precision on each repo's 10 highest-risk PRs is
  0.769 [0.669, 0.856] (95% interval, resampling repos), against a base rate of 0.641 for a
  random pick. The trailing-rate baseline scores 0.585: it ranks PRs by the repo's own recent
  slow rate, which barely varies within a repo, so there it ranks close to randomly.
- **On repos it has never seen, it only works through the repo's slow-rate history.** It beats
  the baseline there (AUC-PR 0.859 vs 0.821) while it can use four features that replay the
  repo's own recent review record, such as its trailing slow rate. Remove those four and its
  AUC-PR falls below the baseline: 0.763 vs 0.821. In this cohort, what the model learns from
  the remaining features in some projects does not carry to others.
- **Why remains open.** The models lean on the same features whether or not a repo was seen
  in training. A pre-registered test of one mechanism, the model recognising repos by their
  fixed attributes, came back partly supported (`PARTIAL_SHAP_ONLY`): the attribution pattern
  is there, but removing those attributes does not measurably hurt seen repos more than unseen
  ones, so it is not shown to cause the gap.

![AUC-PR on seen and unseen repos, with and without the repo's slow-rate history, against the trailing-rate baseline](figures/headline_transfer.png)

## Try the demo

A six-chapter Streamlit walkthrough of the 2026 test period: how long PRs wait for a first
review; one repo's PRs on a timeline, scored when they opened (click any to see why it scored as
it did); a guess-the-stall game against the model; and the cold-start result, with a switch to the
model that never saw the repo. A second section, *How it was built*, walks the pipeline: a
clickable map of the six stages, the data funnel, what the model could know when a PR opened, the
two test designs, and every validity check.

**[Open the live demo](https://prflowpredict-858tjrjzhhafzgblxb6bmo.streamlit.app/)**. It can take
a few seconds to wake up. To run it locally instead:

```bash
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

## Why the numbers can be trusted

- **Leakage prevented by construction.** History features are computed by replaying each
  repo's events in time order, so a PR only ever sees what existed when it was opened, and an
  independent brute-force recomputation matches them exactly. The stated exceptions: eight
  repo-level attributes are 2026 snapshots, and a few at-open PR fields are reconstructed
  approximations.
- **Validity gates at every phase.** Each phase pre-registered checks that its results are
  valid (disjoint splits, no label in any feature, reproducible refits) and passed them before
  its results were read. The gates test validity, never success.
- **Negative results are the headline, not a footnote.** The cold-start failure, a null
  attribution result and a partly supported hypothesis are all reported as found.
- **Every result here is checked.** `tests/test_writeup_claims.py` checks each measured result
  in this README and in the report against the committed data file or phase report it comes
  from, and fails if one is wrong or stale.

## Reproduce

From the committed state alone (Python 3.13; no GitHub access needed):

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
python baseline.py               # the trailing-rate baseline, logged to data/experiments.csv
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
| Demo | `build_demo_data.py` · `build_workflow_data.py` · `demo/app.py` · `demo/chapters.py` · `demo/stages.py` · `demo/triage.py` · `demo/workflow.py` · `demo/charts.py` · `demo/workflow_charts.py` · `demo/views/` |

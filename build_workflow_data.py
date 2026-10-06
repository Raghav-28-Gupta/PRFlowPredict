"""Build the workflow view's two small derived files from committed sources.

    python build_workflow_data.py

demo/data/phase0_gate.json transcribes the Phase 0 pilot gate, which exists only as prose in
docs/phase0_gate_results.md; tests/test_build_workflow_data.py pins every number and one
verbatim excerpt per check to that document. demo/data/feature_groups.json records when each of
the 37 model features is known, read from features.COLUMN_SPEC. The deployed demo reads these
files because it can neither import the project modules nor parse the prose itself
(workflow-view spec, section 4.2)."""
from __future__ import annotations

import json
from pathlib import Path

import features
import featuresets

ROOT = Path(__file__).parent
PHASE0_DOC = ROOT / "docs" / "phase0_gate_results.md"
PHASE0_OUT = ROOT / "demo" / "data" / "phase0_gate.json"
GROUPS_OUT = ROOT / "demo" / "data" / "feature_groups.json"

MISSED = "expectation missed, not a stop"

# id, check, outcome, detail, evidence. `evidence` is a verbatim excerpt of the doc once its `
# and * marks are removed; `detail` may use only numbers the doc states.
PHASE0 = (
    (1, "Ordering monotonicity", "pass",
     "Review and comment timestamps come back in order on both pilot repos.",
     "| 1 | Ordering monotonicity | PASS | PASS |"),
    (2, "Truncation", "pass",
     "0.23% and 0.31% of PRs had a stream longer than the captured page, under the 1% threshold.",
     "### [2] Truncation — PASS on both"),
    (3, "Label-definition sensitivity", "measured",
     "Five label definitions moved the in-stratum repo's slow rate by 3.7pp and the adversarial "
     "repo's by 8.4pp.",
     "### [3] Label-definition sensitivity"),
    (4, "Censoring / degeneracy", MISSED,
     "47.6% of litestream PRs were never reviewed within 30 days, above the blueprint's guessed "
     "10–35%. Ruled not a stop: the positive rate, 55.4%, is near-balanced.",
     "Litestream fails the blueprint's 10–35% censoring expectation."),
    (5, "Snapshot contamination", "measured",
     "The final diff differs from the diff at open on 47.6% of litestream PRs and 29.4% of the "
     "adversarial repo's, so at-open values are rebuilt from the timeline and commits.",
     "At-open reconstruction from timeline.parquet and commits.parquet is mandatory, not optional."),
    (6, "Warm-up necessity", "measured",
     "33.3% of early-window authors had a PR before the window, so every PR since each repo's "
     "start was collected.",
     "### [6] Warm-up necessity — full-history Tier 1 was required"),
    (7, "authorAssociation mutability", "resolved",
     "35 of 88 litestream authors read CONTRIBUTOR on their first PR: the field is rewritten "
     "later, so author history is rebuilt by replay instead.",
     "### [7] authorAssociation mutability — RESOLVED: LEAKY"),
    (8, "Measured cost at the frozen query shape", "pass",
     "A projected ~5 hours to collect 45 repos, well under the 24-hour line.",
     "well under the 24h line"),
    (9, "Resume idempotency", "pass",
     "A run killed and resumed collected the same 400 PRs as an uninterrupted one: 0 missing, "
     "0 extra.",
     "PASS — 400 = 400, 0 missing, 0 extra"),
    (10, "Raw→parse conservation", "pass",
     "Every collected PR parsed, with 0 checksum failures on either repo.",
     "| 10 | Raw→parse conservation | PASS"),
)


def phase0_rows() -> list[dict]:
    return [{"id": i, "check": check, "outcome": outcome, "detail": detail}
            for i, check, outcome, detail, _ in PHASE0]


def feature_groups() -> list[dict]:
    """The FULL feature set in model order, each with its COLUMN_SPEC status: static,
    reconstructed (rebuilt to its value at open), replay (from earlier PRs) or snapshot."""
    return [{"feature": f, "group": features.COLUMN_SPEC[f]["status"]}
            for f in featuresets.FEATURE_SETS["FULL"]]


def dump(rows: list[dict]) -> str:
    return json.dumps(rows, indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    PHASE0_OUT.write_text(dump(phase0_rows()), encoding="utf-8", newline="\n")
    GROUPS_OUT.write_text(dump(feature_groups()), encoding="utf-8", newline="\n")
    print(f"wrote {PHASE0_OUT.relative_to(ROOT)} and {GROUPS_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

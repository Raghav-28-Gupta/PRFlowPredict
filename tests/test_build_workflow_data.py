"""The workflow view's two derived files (workflow-view spec, section 4.2)."""
import json
import re

import pytest

import build_workflow_data as bwd
import featuresets
import writeup_claims as wc


def _doc() -> str:
    """The Phase 0 doc without Markdown's ` and * marks, whitespace collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"[`*]", "", bwd.PHASE0_DOC.read_text(encoding="utf-8")))


def test_both_files_rebuild_byte_identical():
    assert bwd.PHASE0_OUT.read_bytes() == bwd.dump(bwd.phase0_rows()).encode("utf-8")
    assert bwd.GROUPS_OUT.read_bytes() == bwd.dump(bwd.feature_groups()).encode("utf-8")


def test_phase0_lists_the_ten_checks_with_the_docs_outcomes():
    rows = bwd.phase0_rows()
    assert [r["id"] for r in rows] == list(range(1, 11))
    outcome = {r["id"]: r["outcome"] for r in rows}
    assert {i for i, o in outcome.items() if o == "pass"} == {1, 2, 8, 9, 10}
    assert {i for i, o in outcome.items() if o == "measured"} == {3, 5, 6}
    assert outcome[7] == "resolved"
    assert outcome[4] == "expectation missed, not a stop"


@pytest.mark.parametrize("row", bwd.PHASE0, ids=lambda r: f"check{r[0]}")
def test_each_phase0_check_is_pinned_to_the_doc(row):
    i, check, _, detail, evidence = row
    doc = _doc()
    assert check in doc, f"check {i}: name not in the doc"
    assert evidence in doc, f"check {i}: evidence not in the doc"
    for number in re.findall(r"\d+(?:\.\d+)?", detail):
        assert number in doc, f"check {i}: {number} is not in the Phase 0 doc"


def test_phase0_carries_no_retracted_or_forbidden_phrasing():
    text = bwd.PHASE0_OUT.read_text(encoding="utf-8")
    for pattern, _ in wc.RETRACTED:
        assert re.search(pattern, text, flags=re.IGNORECASE) is None, pattern
    for pattern in wc.FORBIDDEN_PATTERNS:
        assert re.search(pattern, text, flags=re.IGNORECASE) is None, pattern


def test_feature_groups_cover_the_full_set_once_by_when_each_is_known():
    rows = bwd.feature_groups()
    assert [r["feature"] for r in rows] == list(featuresets.FEATURE_SETS["FULL"])
    groups = [r["group"] for r in rows]
    assert {g: groups.count(g) for g in set(groups)} == {"static": 11, "reconstructed": 7,
                                                         "replay": 11, "snapshot": 8}


def test_feature_groups_follow_the_extracts_feature_order():
    extract = json.loads((bwd.ROOT / "demo" / "data" / "features.json").read_text(encoding="utf-8"))
    assert [r["feature"] for r in bwd.feature_groups()] == [f["feature"] for f in extract]

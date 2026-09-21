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

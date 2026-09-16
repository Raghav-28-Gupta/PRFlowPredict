import csv
import tracking


def test_log_creates_header_then_appends(tmp_path):
    p = tmp_path / "experiments.csv"
    tracking.log({"scenario": "A", "model": "baseline", "auc_pr": 0.5}, path=p)
    tracking.log({"scenario": "B", "model": "baseline", "auc_pr": 0.4}, path=p)
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    assert [r["scenario"] for r in rows] == ["A", "B"]
    assert list(rows[0].keys()) == tracking.COLUMNS
    assert rows[0]["date"]  # filled in automatically
    assert rows[0]["fold"] == ""  # unspecified columns are blank, never missing


def test_unknown_column_is_rejected(tmp_path):
    import pytest
    with pytest.raises(KeyError):
        tracking.log({"scenario": "A", "bogus": 1}, path=tmp_path / "e.csv")

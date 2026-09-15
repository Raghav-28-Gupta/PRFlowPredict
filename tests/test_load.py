import pandas as pd
import load


def test_repo_dir_naming():
    assert load.repo_dir("owner/name").name == "owner__name"


def test_load_all_concats_and_tolerates_missing_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(load, "PROCESSED", tmp_path)
    for repo in ("a/x", "b/y"):
        d = tmp_path / repo.replace("/", "__")
        d.mkdir()
        pd.DataFrame({"repo": [repo], "pr_id": [repo + "#1"]}).to_parquet(d / "pr_tier2.parquet")
    # only a/x has reviews
    pd.DataFrame({"repo": ["a/x"], "pr_id": ["a/x#1"]}).to_parquet(
        tmp_path / "a__x" / "reviews.parquet")

    frames = load.load_all(["a/x", "b/y"])
    assert sorted(frames["pr_tier2"]["repo"]) == ["a/x", "b/y"]
    assert list(frames["reviews"]["repo"]) == ["a/x"]
    assert frames["timeline"].empty                 # present, empty, not KeyError
    assert set(frames) == set(load.TABLES)

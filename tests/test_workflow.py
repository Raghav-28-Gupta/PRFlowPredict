"""The workflow view's logic (workflow-view spec, sections 4 and 5)."""
import inspect
import json
import subprocess

import pandas as pd
import pytest

import fingerprint
import replay
import splits
import writeup_claims as wc
from demo import triage
from demo import workflow as wf

CLAIMS = {c.name: c.expected for c in wc.CLAIMS}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _runs() -> pd.DataFrame:
    return pd.DataFrame(_json(wf.RUNS)["runs"])


@pytest.fixture(scope="module")
def prs():
    return triage.load()


# ---------------------------------------------------------------------------
# constants, stages, selections
# ---------------------------------------------------------------------------

def test_mirrored_constants_equal_their_sources():
    assert wf.CUTOFF_A == splits.CUTOFF_A
    assert wf.CAP_FRAC == splits.CAP_FRAC
    init = inspect.signature(replay.History.__init__).parameters
    assert wf.THRESHOLD_H == init["threshold_h"].default
    assert wf.TRAILING_DAYS == init["trailing_days"].default
    assert wf.ALPHA == inspect.signature(replay.History.features_at).parameters["alpha"].default
    assert set(wf.VERDICT_NAMES) == set(fingerprint.VERDICTS.values())


def test_stage_artifacts_say_truthfully_whether_they_are_committed():
    """The deployed app cannot run git, so the flags are static; check them against git here."""
    try:
        out = subprocess.run(["git", "ls-files"], cwd=wf.ROOT, capture_output=True, text=True)
    except FileNotFoundError:
        pytest.skip("git is not installed")
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    tracked = out.stdout.splitlines()
    for s in wf.STAGES:
        assert s.chapter is None or 0 <= s.chapter < 6, s.key
        for path, committed, _ in s.artifacts:
            hit = any(t.startswith(path) for t in tracked) if path.endswith("/") else path in tracked
            assert hit == committed, (s.key, path)
        for doc in s.docs:
            assert doc in tracked, (s.key, doc)


def test_stages_run_in_order_and_badges_cover_each(prs):
    assert wf.STAGE_KEYS == ("collect", "label", "features", "evaluate", "explain", "ship")
    badges = wf.badges(prs, wf.feature_groups())
    assert set(badges) == set(wf.STAGE_KEYS)
    assert badges["ship"] == f"{len(prs):,} demo PRs"
    assert badges["features"] == f"{len(wf.feature_groups())} features"
    table = wf.stage_table(badges, wf.gates())
    assert list(table["key"]) == list(wf.STAGE_KEYS) and table["gate"].str.len().gt(0).all()


def test_picked_reads_a_chart_selection_and_tolerates_empties():
    event = {"selection": {"stage": [{"key": "label"}]}}
    assert wf.picked(event, "stage", "key") == "label"
    assert wf.picked(event, "stage") == {"key": "label"}
    for empty in (None, {}, {"selection": {}}, {"selection": {"stage": []}}):
        assert wf.picked(empty, "stage", "key") is None


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def test_gates_hold_ten_pilot_checks_and_five_per_phase_all_passing():
    g = wf.gates()
    assert g.groupby("phase").size().to_dict() == {"0": 10, "2": 5, "3": 5, "4": 5, "6": 5, "6b": 5}
    assert g.loc[g["phase"] != "0", "passed"].all()
    pilot = g[g["phase"] == "0"].set_index("id")["status"]
    assert pilot[4] == wf.MISSED and pilot[7] == "resolved"


def test_gate_value_text_reads_scalars_lists_and_dicts():
    assert wf.gate_value_text(0.4639) == "0.4639"
    assert wf.gate_value_text([38462, 38462]) == "38462, 38462"
    assert wf.gate_value_text({"n": 500, "max_abs_diff": 0.0, "worst": None}) == "n: 500; max_abs_diff: 0; worst: None"
    assert wf.gate_value_text({}) == "none" and wf.gate_value_text(True) == "yes"


def test_gate_lines_count_each_stages_checks():
    table = wf.gates()
    assert wf.gate_line(("0",), table) == "pilot gate: 10 checks"
    assert wf.gate_line(("6", "6b"), table) == "10/10 checks pass"
    assert wf.gate_line((), table) == "no gate"


def test_audits_and_the_live_mismatch_come_from_the_phase3_files():
    a = wf.audits()
    assert f"{a['audit_n']} rows, largest difference {a['audit_max_diff']}" == CLAIMS["audit_rows"]
    assert a["live_checks"] == a["live_matched"] == a["live_rows"] * 8
    assert [m["pr"] for m in wf.live_mismatches()] == ["radixark/miles#784"]


# ---------------------------------------------------------------------------
# the funnel, the pools, the labels
# ---------------------------------------------------------------------------

def test_the_repo_funnel_reads_each_count_from_its_source():
    f = wf.repo_funnel()["count"].tolist()
    cohort, kept = _json(wf.COHORT), _json(wf.KEPT)
    searched = sum(_json(p)["data"]["search"]["repositoryCount"] for p in wf.SEARCH.glob("search_*_p0.json"))
    assert f == [searched, cohort["n_candidates_pooled"], cohort["n_selected"], len(kept["kept"])]


def test_the_pr_funnel_runs_from_the_window_to_this_demos_prs(prs):
    f = wf.pr_funnel().set_index("step")["count"]
    a = _runs().query("scenario == 'A' and featureset == 'FULL'").iloc[0]
    assert f["human-authored"] == _json(wf.PHASE2_ROWS)["n_rows"]
    assert f["training rows, seen-repo model"] == a["n_train"]
    assert f["test PRs, this demo"] == a["n_test"] == len(prs)
    narrowing = f.iloc[:4].tolist()
    assert narrowing == sorted(narrowing, reverse=True)
    # the cap trims training rows only, so train + test falls short of the modelled rows
    assert f["training rows, seen-repo model"] + f["test PRs, this demo"] < f["modelled"]


def test_every_funnel_step_says_what_it_removed():
    cohort = _json(wf.COHORT)
    for step in list(wf.repo_funnel()["step"]) + list(wf.pr_funnel()["step"]):
        _, note = wf.removed(step)
        assert note, step
    rejected, _ = wf.removed("selected")
    assert len(rejected) == len(cohort["rejected"])
    dropped, _ = wf.removed("kept")
    assert len(dropped) == cohort["n_selected"] - len(_json(wf.KEPT)["kept"])
    assert dropped["reason"].str.len().gt(0).all()
    capped, _ = wf.removed("training rows, seen-repo model")
    assert len(capped) and (capped["capped_out"] > 0).all()
    with pytest.raises(KeyError):
        wf.removed("no such step")


def test_each_pool_is_a_slice_of_its_groups_search_results():
    p = wf.pool_bias()
    cohort = _json(wf.COHORT)
    assert len(p) == len(cohort["languages"]) * len(cohort["star_tiers"])
    assert (p["pooled"] == cohort["n_candidates_pooled"] // len(p)).all()
    assert (p["pooled"] < p["matched"]).all()


def test_each_groups_search_results_came_back_sorted_by_stars():
    """The finding the pool-bias disclosure rests on: the pool took each group's first results,
    and those are its most-starred repos."""
    for group in wf.pool_bias()["group"]:
        lang, tier = group.split(":")
        lo, hi = tier.split("-")
        pages = sorted(wf.SEARCH.glob(f"search_{lang}_{lo}_{hi}_p*.json"))
        stars = [n["stargazerCount"] for p in pages for n in _json(p)["data"]["search"]["nodes"] if n]
        assert stars == sorted(stars, reverse=True), group


def test_label_rates_match_the_registered_claims():
    r = wf.label_rates()
    assert f"{r['d5']:.1%} slow under D5" == CLAIMS["d5_rate"]
    assert f"{r['d3']:.1%} under D3" == CLAIMS["d3_rate"]
    assert set(r["bot_repos"]) == set(wf.BOT_REPOS)


def test_the_disclosures_cite_runtime_numbers():
    rates = wf.label_rates()["bot_repos"]
    a, b = (rates[r] for r in wf.BOT_REPOS)
    assert f"({a:.1%} and {b:.1%})" in wf.bot_note(wf.label_rates())
    note = wf.bot_note(wf.label_rates())
    assert "not on the project's list of known bots" in note
    assert f"in {wf.BOT_REPOS[0]} and {wf.BOT_REPOS[1]}," in note and "{" not in note
    pool = wf.pool_bias()
    note = wf.pool_note(pool)
    assert f"between {pool['share'].min():.1%} and {pool['share'].max():.1%} of the group" in note
    assert f"the first {pool['pooled'].iloc[0]} of each" in note


def test_the_6b_verdict_is_read_from_the_readme():
    assert wf.verdict_6b() == CLAIMS["verdict_6b"]


def test_a_readme_naming_two_verdicts_is_refused(tmp_path, monkeypatch):
    readme = tmp_path / "README.md"
    readme.write_text("`PARTIAL_SHAP_ONLY` and `SUPPORTED`", encoding="utf-8")
    monkeypatch.setattr(wf, "README", readme)
    with pytest.raises(LookupError):
        wf.verdict_6b()


# ---------------------------------------------------------------------------
# the two test designs
# ---------------------------------------------------------------------------

def test_the_cap_reproduces_every_stored_training_and_test_count():
    rows = wf.split_rows()
    full = _runs().query("featureset == 'FULL'")
    a = full.query("scenario == 'A'").iloc[0]
    seen = wf.design_rows(rows).groupby("segment")["rows"].sum()
    assert (seen["trained on"], seen["tested on"]) == (a["n_train"], a["n_test"])
    assert rows["trained"].sum() == a["n_train"]
    for _, run in full.query("scenario == 'B'").iterrows():
        fold = wf.design_rows(rows, run["fold"]).groupby("segment")["rows"].sum()
        assert (fold["trained on"], fold["tested on"]) == (run["n_train"], run["n_test"]), run["fold"]


def test_design_rows_split_each_repo_without_losing_a_row():
    rows = wf.split_rows()
    for fold in (None, 0):
        long = wf.design_rows(rows, fold).groupby("repo")["rows"].sum()
        expected = rows.set_index("repo")["all_rows"]
        assert long.reindex(expected.index, fill_value=0).equals(expected)


def test_split_rows_cover_every_kept_repo_once_with_its_fold(prs):
    rows = wf.split_rows()
    assert rows["repo"].is_unique and set(rows["repo"]) == set(_json(wf.KEPT)["kept"])
    assert rows["fold"].notna().all()
    assert rows["test_2026"].sum() == len(prs)
    assert rows.set_index("repo").loc["kdlbs/kandev", "pre_2026"] == 0


def test_fold_results_match_the_runs():
    f = wf.fold_results()
    b = _runs().query("scenario == 'B'")
    assert f["fold"].tolist() == sorted(b["fold"].unique().tolist())
    for _, r in f.iterrows():
        nlr = b.query("featureset == 'NO_LABEL_REPLAY' and fold == @r.fold").iloc[0]
        assert r["without_history"] == nlr["auc_pr"] and r["baseline"] == nlr["baseline_auc_pr"]


# ---------------------------------------------------------------------------
# known at time t
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def check(prs):
    return wf.trailing_check(prs)


def _toy() -> pd.DataFrame:
    """One repo; P0 opens at t. Hand-computed with the prior 0.5 and alpha 5:
    P1 opened 10 days before, reviewed       -> counted (old enough), not slow
    P2 opened 2 days before, reviewed 1 later -> counted (reviewed before t), not slow
    P3 opened 1 day before, no review yet     -> not yet knowable; it later stalls
    P4 opened 100 days before                 -> outside the 90-day window
    resolvable: k=2, slow=0 -> (0 + 2.5) / 7;  naive: k=3, slow=1 -> (1 + 2.5) / 8.
    The stored values of the others: P4 has an empty window (0.5); P1 counts P4, exactly 90 days
    earlier; P2 counts P1; P3 counts P1 only, as P2's review lands exactly at P3's opening."""
    t = pd.Timestamp("2026-05-01T00:00:00Z")
    d = pd.Timedelta(days=1)
    return pd.DataFrame({
        "repo": ["r"] * 5, "pr_id": ["P4", "P1", "P2", "P3", "P0"], "number": [4, 1, 2, 3, 0],
        "created_at": [t - 100 * d, t - 10 * d, t - 2 * d, t - d, t],
        "first_review_at": [t - 99 * d, t - 9 * d, t - d, pd.NaT, pd.NaT],
        "is_slow": [False, False, False, True, True],
        "x__trailing_n": [0, 1, 1, 1, 2],
        "x__trailing_90d_slow_rate": [0.5, 2.5 / 6, 2.5 / 6, 2.5 / 6, 2.5 / 7],
    })


def test_a_pr_counts_once_it_is_a_week_old_or_reviewed():
    frame, s = wf.replay_at(_toy(), "r", "P0", prior_rate=0.5)
    assert (s["k"], s["slow"], s["unknown"]) == (2, 0, 1)
    assert s["rate"] == pytest.approx(2.5 / 7) and s["matches"]
    status = frame.set_index("pr_id")["status"]
    assert status.to_dict() == {"P4": wf.OLD, "P1": wf.COUNTED, "P2": wf.COUNTED, "P3": wf.UNKNOWN}
    assert frame.set_index("pr_id").loc["P3", "lane"] == wf.UNKNOWN_LANE


def test_the_naive_rule_counts_what_was_not_yet_knowable():
    frame, s = wf.replay_at(_toy(), "r", "P0", rule="naive", prior_rate=0.5)
    assert (s["k"], s["slow"]) == (3, 1) and s["rate"] == pytest.approx(3.5 / 8) and not s["matches"]
    leaked = frame.set_index("pr_id").loc["P3"]
    assert leaked["status"] == wf.LEAK and leaked["lane"] == wf.STALLED_LANE


def test_the_vectorised_check_agrees_on_the_toy():
    out = wf.trailing_check(_toy(), prior_rate=0.5).set_index("pr_id")
    assert (out.loc["P0", "k"], out.loc["P0", "unknown"]) == (2, 1)
    assert out["matches"].all()


def test_the_prior_is_the_rate_of_an_empty_window(prs):
    g = wf.prior(prs)
    assert (prs.loc[prs["x__trailing_n"] == 0, "x__trailing_90d_slow_rate"] == g).all()


def test_exact_repos_match_on_every_pr_and_include_the_default(check):
    exact = wf.exact_repos(check)
    assert "kdlbs/kandev" in exact
    assert check[check["repo"].isin(exact)]["matches"].all()
    assert not check[~check["repo"].isin(exact)].groupby("repo")["matches"].all().any()


def test_replay_at_matches_the_stored_feature_on_every_pr_of_an_exact_repo(prs, check):
    exact = wf.exact_repos(check)
    repo = min(exact, key=lambda r: int((prs["repo"] == r).sum()))
    g = wf.prior(prs)
    for pid in prs.loc[prs["repo"] == repo, "pr_id"]:
        assert wf.replay_at(prs, repo, pid, prior_rate=g)[1]["matches"], pid


def test_replay_at_agrees_with_the_vectorised_check_across_repos(prs, check):
    g = wf.prior(prs)
    for _, r in check.sample(150, random_state=0).iterrows():
        _, s = wf.replay_at(prs, r["repo"], r["pr_id"], prior_rate=g)
        assert (s["k"], s["unknown"]) == (r["k"], r["unknown"]), r["pr_id"]
        assert s["rate"] == pytest.approx(r["rate"], abs=1e-12)


def test_the_default_pr_has_the_most_unknowable_neighbours(check):
    pid = wf.default_pr(check, "kdlbs/kandev")
    rows = check[check["repo"] == "kdlbs/kandev"].set_index("pr_id")
    assert rows.loc[pid, "unknown"] == rows["unknown"].max() > 0


def test_the_bundle_holds_everything_the_pages_read(prs):
    b = wf.bundle(prs)
    assert set(b) == {"stages", "gates", "groups", "repo_funnel", "pr_funnel", "pool_bias", "labels",
                      "audits", "live", "prior", "check", "exact", "splits", "folds", "scenario_a", "verdict"}

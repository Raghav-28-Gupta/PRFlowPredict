"""The demo app, run headless with Streamlit's AppTest (Phase 7 spec, sections 6, 7 and 9)."""
import ast
import importlib
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import writeup_claims as wc
from demo import triage

APP = Path(__file__).parents[1] / "demo" / "app.py"
UNSEEN = "Unseen repo (cold-start)"


def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60).run()


def test_the_app_opens_on_the_default_repo_and_day_with_a_list():
    at = _app()
    assert not at.exception
    prs = triage.load()
    assert at.sidebar.selectbox[0].value == triage.default_repo(prs, triage.moment(triage.DEFAULT_DAY))
    assert at.sidebar.date_input[0].value == triage.DEFAULT_DAY
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) > 0


def test_switching_the_model_changes_the_scores_shown():
    at = _app()
    seen = at.dataframe[0].value["risk"].tolist()
    at.sidebar.radio[0].set_value(UNSEEN).run()
    assert not at.exception
    assert at.dataframe[0].value["risk"].tolist() != seen


def test_changing_the_day_re_renders_the_list():
    at = _app()
    before = at.dataframe[0].value["url"].tolist()
    at.sidebar.date_input[0].set_value(triage.DEFAULT_DAY.replace(month=2)).run()
    assert not at.exception
    assert len(at.dataframe) == 0 or at.dataframe[0].value["url"].tolist() != before


def test_a_day_with_nobody_waiting_says_so():
    prs, day = triage.load(), triage.FIRST_DAY
    empty = next(r for r in triage.repos(prs) if triage.ranked(prs, r, triage.moment(day), "A").empty)
    at = _app()
    at.sidebar.selectbox[0].set_value(empty).run()
    at.sidebar.date_input[0].set_value(day).run()
    assert not at.exception
    assert len(at.dataframe) == 0
    assert "early January" in at.info[0].value


def _markdown(at, start):
    return next(m.value for m in at.markdown if m.value.startswith(start))


def test_the_tally_shows_the_whole_lists_stall_count_beside_the_top_three():
    """PRs still waiting at a moment are survivors and mostly stall, so a top-3 count alone
    reads as skill it does not measure. The line must show the list's own rate and say so."""
    at = _app()
    prs = triage.load()
    at_ = triage.moment(triage.DEFAULT_DAY)
    listing = triage.ranked(prs, triage.default_repo(prs, at_), at_, "A")
    stalled, k = triage.tally(listing)
    line = _markdown(at, "**Of the top")
    assert f"Of the top {k} by risk, {stalled} stalled" in line
    assert f"of all {len(listing)} PRs waiting, {int(listing['stalled'].sum())} did" in line
    assert "cannot show how well the model ranks" in line


def test_the_seen_note_describes_one_model_for_all_repos():
    """kdlbs/kandev has no PRs before 2026, so 'trained on this repo's own earlier PRs' is false."""
    at = _app()
    at.sidebar.selectbox[0].set_value("kdlbs/kandev").run()
    note = _markdown(at, "**Seen in training:**")
    assert f"all {len(triage.repos(triage.load()))} repos" in note and "if it had any" in note
    assert "this repo's own earlier PRs" not in note


def test_the_header_says_only_prs_opened_in_2026_are_listed():
    at = _app()
    assert any("Only PRs opened from 1 January 2026 have scores" in m.value for m in at.markdown)


def test_the_caption_says_risk_is_a_ranking_score_not_a_probability():
    at = _app()
    assert "not a calibrated probability" in at.caption[0].value


def test_demo_requirements_pin_only_what_the_app_needs_at_the_root_versions():
    """Spec section 3: streamlit, pandas and pyarrow only. Pinning numpy as well blocked Python
    3.14, which numpy 2.2.6 has no wheel for."""
    def pins(path):
        lines = [ln.split("#")[0].strip() for ln in path.read_text(encoding="utf-8").splitlines()]
        return dict(ln.split("==") for ln in lines if "==" in ln)
    demo, root = pins(APP.parent / "requirements.txt"), pins(APP.parents[1] / "requirements.txt")
    assert set(demo) == {"streamlit", "pandas", "pyarrow"}
    assert all(demo[p] == root[p] for p in demo)


def _lookup(at: AppTest, text: str) -> AppTest:
    at.text_input(key="pr_ref").input(text).run()
    return at


def test_looking_up_a_pr_link_shows_both_models_scores_and_what_happened():
    prs = triage.load()
    pr = prs.iloc[[100]]
    row = pr.iloc[0]
    at = _lookup(_app(), row["url"])
    assert not at.exception
    assert [m.value for m in at.metric] == [f"{row['score_a']:.2f}", f"{row['score_b']:.2f}"]
    shown = " ".join(m.value for m in at.markdown)
    assert triage.outcome(pr).iloc[0] in shown
    share, n = triage.rank_in_repo(prs, pr, "A")
    assert f"Higher than {share:.0%} of this repo's other {n:,} replayed PRs" in shown


def test_an_unknown_pr_says_it_is_not_in_the_replay():
    at = _lookup(_app(), "nobody/nothing#1")
    assert not at.exception and len(at.metric) == 0
    assert "Not in the replay" in at.warning[0].value


def test_random_pr_fills_the_box_and_shows_its_card():
    at = _app()
    at.button(key="random_pr").click().run()
    assert not at.exception
    assert re.fullmatch(r"[^/\s]+/[^#\s]+#\d+", at.text_input(key="pr_ref").value)
    assert len(at.metric) == 2


def test_the_app_uses_the_current_triage_even_when_an_old_copy_is_cached(monkeypatch):
    """Streamlit Community Cloud re-runs app.py when the repo updates but keeps imported
    modules in memory, so the deployed app once ran the new app.py against the old triage.py:
    AttributeError: module 'triage' has no attribute 'random_pr'. Recreate that stale copy."""
    monkeypatch.syspath_prepend(str(APP.parent))
    cached = importlib.import_module("triage")
    monkeypatch.delattr(cached, "random_pr")          # what the old triage.py did not have
    at = _app()
    at.button(key="random_pr").click().run()
    assert not at.exception


def test_the_app_hard_codes_no_result_number():
    assert re.search(r"\b0\.\d{3}\b", APP.read_text(encoding="utf-8")) is None


@pytest.mark.parametrize("pattern,reason", wc.RETRACTED, ids=lambda v: str(v)[:30])
def test_the_app_avoids_retracted_phrasing(pattern, reason):
    hit = re.search(pattern, APP.read_text(encoding="utf-8"), flags=re.IGNORECASE)
    assert hit is None, f"demo/app.py says {hit.group(0)!r}, but {reason}"


def test_the_demo_imports_only_what_its_own_requirements_install():
    """Streamlit Cloud installs demo/requirements.txt, not the project's: no sklearn, lightgbm,
    shap, or project module may be imported by the deployed files."""
    allowed = {"__future__", "datetime", "importlib", "json", "math", "pathlib", "random", "re", "sys",
               "streamlit", "pandas", "triage"}
    for name in ("app.py", "triage.py"):
        tree = ast.parse((APP.parent / name).read_text(encoding="utf-8"))
        mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert mods <= allowed, f"demo/{name} imports {sorted(mods - allowed)}"

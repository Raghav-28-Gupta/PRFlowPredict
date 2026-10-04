"""The demo app, run headless with Streamlit's AppTest (Phase 7 and demo-story specs).

Chapters are reached with AppTest.switch_page on their page files (views/), which keeps the
current chapter between runs; the Back/Next buttons are tested separately."""
import ast
import importlib
import json
import random
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import writeup_claims as wc
from demo import triage

DEMO = Path(__file__).parents[1] / "demo"
APP = DEMO / "app.py"
VIEWS = ["problem", "watch", "why", "test_yourself", "transfer", "limits"]
HEADINGS = ["Some pull requests wait weeks for a first review", "Watch it work", "Why it decides",
            "Test yourself", "Does it transfer to a new project?", "Honest limits"]
UNSEEN = "Unseen repo (cold-start)"
SOURCES = sorted(DEMO.glob("*.py")) + sorted((DEMO / "views").glob("*.py"))


def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=90).run()


def _at(chapter: str, at: AppTest | None = None) -> AppTest:
    at = at or _app()
    return at.switch_page(f"views/{chapter}.py").run()


def _markdown(at, start):
    return next(m.value for m in at.markdown if m.value.startswith(start))


def _text(at) -> str:
    return " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# the story: six chapters, Back/Next, speaker notes
# ---------------------------------------------------------------------------

def test_the_app_opens_on_chapter_one():
    at = _app()
    assert not at.exception and at.title[0].value == HEADINGS[0]


@pytest.mark.parametrize("i", range(6))
def test_every_chapter_renders_without_an_exception(i):
    at = _at(VIEWS[i])
    assert not at.exception and at.title[0].value == HEADINGS[i]


@pytest.mark.parametrize("i", range(5))
def test_next_opens_the_following_chapter(i):
    at = _at(VIEWS[i])
    at.button(key=f"next_{i}").click().run()
    assert at.title[0].value == HEADINGS[i + 1]


@pytest.mark.parametrize("i", range(1, 6))
def test_back_opens_the_previous_chapter(i):
    at = _at(VIEWS[i])
    at.button(key=f"back_{i}").click().run()
    assert at.title[0].value == HEADINGS[i - 1]


def test_the_first_chapter_has_no_back_and_the_last_no_next():
    assert "back_0" not in [b.key for b in _at("problem").button]
    assert "next_5" not in [b.key for b in _at("limits").button]


def test_speaker_notes_are_hidden_by_default_and_shown_when_toggled():
    at = _at("problem")
    assert not [i for i in at.info if i.value.startswith("**Speaker note:**")]
    at.sidebar.toggle[0].set_value(True).run()
    assert [i for i in at.info if i.value.startswith("**Speaker note:**")]


# ---------------------------------------------------------------------------
# 1. the problem
# ---------------------------------------------------------------------------

def test_the_problem_chapter_quotes_the_computed_stalled_share():
    buckets = triage.wait_buckets(triage.load())
    stalled = buckets.loc[buckets["stalled"], "share"].sum()
    assert f"**{stalled:.0%} stalled**" in _text(_at("problem"))


# ---------------------------------------------------------------------------
# 2. watch it work (the Phase 7 triage checks, migrated)
# ---------------------------------------------------------------------------

def test_watch_opens_on_the_default_repo_and_day_with_a_list():
    at = _at("watch")
    prs = triage.load()
    assert at.sidebar.selectbox[0].value == triage.default_repo(prs, triage.moment(triage.DEFAULT_DAY))
    assert at.slider(key="day").value == triage.DEFAULT_DAY
    assert len(at.dataframe) >= 1 and len(at.dataframe[0].value) > 0


def test_switching_the_model_changes_the_scores_shown():
    at = _at("watch")
    seen = at.dataframe[0].value["risk"].tolist()
    at.sidebar.radio[0].set_value(UNSEEN).run()
    assert not at.exception
    assert at.dataframe[0].value["risk"].tolist() != seen


def test_changing_the_day_re_renders_the_list():
    at = _at("watch")
    before = at.dataframe[0].value["url"].tolist()
    at.slider(key="day").set_value(triage.DEFAULT_DAY.replace(month=2)).run()
    assert not at.exception
    assert at.dataframe[0].value["url"].tolist() != before


def test_a_day_with_nobody_waiting_says_so():
    prs, day = triage.load(), triage.FIRST_DAY
    empty = next(r for r in triage.repos(prs) if triage.ranked(prs, r, triage.moment(day), "A").empty)
    at = _at("watch")
    at.sidebar.selectbox[0].set_value(empty).run()
    at.slider(key="day").set_value(day).run()
    assert not at.exception
    assert "early January" in at.info[0].value


def test_the_tally_shows_the_whole_lists_stall_count_beside_the_top_three():
    """PRs still waiting at a moment are survivors and mostly stall, so a top-3 count alone
    reads as skill it does not measure. The line must show the list's own rate and say so."""
    at = _at("watch")
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
    at = _at("watch")
    at.sidebar.selectbox[0].set_value("kdlbs/kandev").run()
    note = _markdown(at, "**Seen in training:**")
    assert f"all {len(triage.repos(triage.load()))} repos" in note and "if it had any" in note
    assert "this repo's own earlier PRs" not in note


def test_the_watch_chapter_says_only_prs_opened_in_2026_are_shown_and_none_is_live():
    text = _text(_at("watch"))
    assert "Only PRs opened from 1 January 2026 have scores" in text
    assert "not live data" in text and "never updated" in text


def test_the_caption_says_risk_is_a_ranking_score_not_a_probability():
    assert "not a calibrated probability" in _at("watch").caption[0].value


# ---------------------------------------------------------------------------
# 3. why it decides (the PR #10 lookup, migrated)
# ---------------------------------------------------------------------------

def _lookup(at: AppTest, text: str) -> AppTest:
    at.text_input(key="pr_ref").input(text).run()
    return at


def test_looking_up_a_pr_link_shows_both_models_scores_and_what_happened():
    prs = triage.load()
    pr = prs.iloc[[100]]
    row = pr.iloc[0]
    at = _lookup(_at("why"), row["url"])
    assert not at.exception
    assert [m.value for m in at.metric] == [f"{row['score_a']:.2f}", f"{row['score_b']:.2f}"]
    shown = _text(at)
    assert triage.outcome(pr).iloc[0] in shown
    share, n = triage.rank_in_repo(prs, pr, "A")
    assert f"Higher than {share:.0%} of this repo's other {n:,} replayed PRs" in shown


def test_an_unknown_pr_says_it_is_not_in_the_replay():
    at = _lookup(_at("why"), "nobody/nothing#1")
    assert not at.exception and len(at.metric) == 0
    assert "Not in the replay" in at.warning[0].value


def test_random_pr_fills_the_box_and_shows_its_card():
    at = _at("why")
    at.button(key="random_pr").click().run()
    assert not at.exception
    assert re.fullmatch(r"[^/\s]+/[^#\s]+#\d+", at.text_input(key="pr_ref").value)
    assert len(at.metric) == 2


def test_with_nothing_selected_why_explains_the_riskiest_pr_waiting_on_the_default_day():
    prs = triage.load()
    at_ = triage.moment(triage.DEFAULT_DAY)
    repo = triage.default_repo(prs, at_)
    waiting = triage.awaiting_review(prs[prs["repo"] == repo], at_)
    top = waiting.sort_values(["score_a", "number"], ascending=[False, True]).iloc[0]
    at = _at("why")
    assert [m.value for m in at.metric] == [f"{top['score_a']:.2f}", f"{top['score_b']:.2f}"]
    assert f"Why the Seen in training model scored it {top['score_a']:.2f}" in [s.value for s in at.subheader]


def test_the_why_caption_says_the_bars_reach_the_scores_log_odds_not_the_score():
    prs, features = triage.load(), triage.load_features()
    at_ = triage.moment(triage.DEFAULT_DAY)
    repo = triage.default_repo(prs, at_)
    waiting = triage.awaiting_review(prs[prs["repo"] == repo], at_)
    pr = waiting.sort_values(["score_a", "number"], ascending=[False, True]).head(1)
    rows, base = triage.why(pr, "A", features)
    caption = _at("why").caption[0].value
    assert "Model attribution, not cause" in caption and "add up to the score shown" not in caption
    assert f"Together they reach {base + rows['push'].sum():+.2f}, the log-odds of the score shown" in caption
    assert f"{base + rows['push'].sum():+.2f}" == f"{triage.logit(pr['score_a'].iloc[0]):+.2f}"


# ---------------------------------------------------------------------------
# 4. test yourself
# ---------------------------------------------------------------------------

def test_the_game_deals_four_cards_and_reveal_scores_the_round():
    at = _at("test_yourself")
    assert [f"pick_{i}" for i in range(4)] == [b.key for b in at.button if b.key.startswith("pick_")]
    at.button(key="pick_1").click().run()
    at.button(key="reveal").click().run()
    assert not at.exception
    text = _text(at)
    assert text.count("**Stalled**") == 1 and text.count("Reviewed within 7 days") == 3
    assert "You picked PR 2" in text and "This session with the Seen in training model: you" in text
    assert "of 1" in text


def test_deal_again_deals_a_new_round():
    at = _at("test_yourself")
    first = list(at.session_state["game"]["ids"])
    at.button(key="pick_0").click().run()
    at.button(key="reveal").click().run()
    at.button(key="deal").click().run()
    game = at.session_state["game"]
    assert game["revealed"] is False and game["pick"] is None and game["ids"] != first


def test_the_game_shows_its_exact_hit_rate_against_a_random_guess():
    rate = triage.game_hit_rate(triage.load(), "A")
    line = _markdown(_at("test_yourself"), "Over every round this game can deal")
    assert f"**{rate:.0%}**" in line and "**25%**" in line


# ---------------------------------------------------------------------------
# 5. does it transfer? / 6. honest limits
# ---------------------------------------------------------------------------

def test_the_transfer_chapter_shows_the_readmes_checked_numbers():
    claims = {c.name: c.expected for c in wc.CLAIMS}
    text = _text(_at("transfer"))
    for name in ("a_p10", "b_full_vs_baseline", "b_nlr_vs_baseline"):
        assert claims[name] in text, name
    assert claims["a_baseline_p10"].removeprefix("baseline scores ") in text
    assert claims["a_base_rate"].removeprefix("base rate of ") in text


def test_the_limits_chapter_names_the_four_limits():
    text = _text(_at("limits"))
    for phrase in ("never measured", "small sample", "2026 snapshots", "cannot measure ranking"):
        assert phrase in text, phrase


# ---------------------------------------------------------------------------
# final-review fixes
# ---------------------------------------------------------------------------

def test_the_problem_chapter_says_most_stalled_prs_were_closed_and_few_were_left_waiting():
    prs = triage.load()
    unreviewed = prs[prs["first_review_at"].isna() & prs["closed_at"].notna()]
    fast = ((unreviewed["closed_at"] - unreviewed["created_at"]).dt.total_seconds() < 86400).mean()
    week = 7 * 86400
    no_review = prs["first_review_at"].isna() | ((prs["first_review_at"] - prs["created_at"]).dt.total_seconds() > week)
    still_open = prs["closed_at"].isna() | ((prs["closed_at"] - prs["created_at"]).dt.total_seconds() > week)
    limbo = (no_review & still_open).mean()
    text = _text(_at("problem"))
    assert "human-authored" in text
    assert f"{fast:.0%} of them within a day of opening" in text
    assert f"{limbo:.0%} were still open and unreviewed a week after opening" in text


def test_the_game_sentence_names_its_eligibility_filter():
    prs = triage.load()
    weeks = triage.eligible_weeks(prs)
    line = _markdown(_at("test_yourself"), "Over every round this game can deal")
    assert f"{len(weeks)} project-weeks, in {len({r for r, _ in weeks})} of the {len(triage.repos(prs))} projects" in line


def test_a_revealed_round_keeps_the_model_it_was_revealed_with():
    at = _at("test_yourself")
    ids = list(at.session_state["game"]["ids"])
    at.button(key="pick_0").click().run()
    at.button(key="reveal").click().run()
    at.sidebar.radio[0].set_value(UNSEEN).run()
    assert not at.exception
    text = _text(at)
    rnd = triage.load().set_index("pr_id").loc[ids]
    assert all(f"risk {s:.2f}" in text for s in rnd["score_a"])
    assert "The Seen in training model picked" in text and "with the Seen in training model" in text


def test_reveal_counts_a_round_only_once(monkeypatch):
    """A stale second click on Reveal must not add a phantom round to the tally."""
    monkeypatch.syspath_prepend(str(DEMO))
    chapters = importlib.import_module("chapters")
    prs = triage.load()
    rnd = triage.deal(prs, random.Random(1))
    ctx = chapters.Context(prs, triage.load_features(), {}, {"A": 0.5, "B": 0.5}, "A", "x/y", False, "light")
    state = {"_ctx": ctx, "game": {"ids": list(rnd["pr_id"]), "pick": 0, "revealed": False}}
    monkeypatch.setattr(chapters.st, "session_state", state)
    chapters._reveal()
    chapters._reveal()
    assert state["game"]["tally"]["A"]["played"] == 1


def test_a_callback_uses_its_own_sessions_model_not_the_last_viewers():
    """Viewers share one server process. A game callback must read its own session's choices, not
    whatever another viewer's run left behind."""
    prs = triage.load()
    rnd = next(r for r in (triage.deal(prs, random.Random(s)) for s in range(200))
               if triage.model_pick(r, "A") != triage.model_pick(r, "B"))
    presenter = _at("test_yourself")
    game = dict(presenter.session_state["game"])
    game.update(ids=list(rnd["pr_id"]), pick=None, revealed=False)
    presenter.session_state["game"] = game
    presenter.run()
    presenter.button(key="pick_0").click().run()
    viewer = _app()
    viewer.sidebar.radio[0].set_value(UNSEEN).run()          # another viewer's run, on the Unseen model
    presenter.button(key="reveal").click().run()
    assert not presenter.exception
    assert presenter.session_state["game"]["model_pick"] == triage.model_pick(rnd, "A")


def _timeline_chart_id(at: AppTest) -> str:
    def walk(node):
        yield node
        children = getattr(node, "children", None)
        if isinstance(children, dict):
            for child in children.values():
                yield from walk(child)
    return next(n.proto.id for n in walk(at._tree)
                if type(getattr(n, "proto", None)).__name__ == "VegaLiteChart" and n.proto.id.endswith("-timeline"))


def _run_with_timeline_click(at: AppTest, chart_id: str, pr_id: str) -> AppTest:
    """Re-run as if the browser sent a click on pr_id. Chart selections cannot be driven through
    AppTest's public API, so this uses its private one (Streamlit 1.65): the selection travels as a
    JSON string_value on the chart's widget id."""
    states = at._tree.get_widget_states()
    widget = states.widgets.add()
    widget.id = chart_id
    widget.string_value = json.dumps({"selection": {"pick": [{"pr_id": pr_id}]}})
    at._run(states)
    return at


def test_a_stale_timeline_click_after_a_repo_change_does_not_crash():
    at = _at("watch")
    prs = triage.load()
    repo = at.sidebar.selectbox[0].value
    pr_id = prs.loc[prs["repo"] == repo, "pr_id"].iloc[0]
    chart = _timeline_chart_id(at)
    _run_with_timeline_click(at, chart, pr_id)
    assert not at.exception and at.session_state["selected_pr"] == pr_id
    at.sidebar.selectbox[0].set_value(next(r for r in triage.repos(prs) if r != repo))
    _run_with_timeline_click(at, chart, pr_id)                # the browser re-sends the old selection
    assert not at.exception


def test_the_revealed_game_and_an_empty_watch_day_carry_the_risk_caption():
    at = _at("test_yourself")
    at.button(key="pick_0").click().run()
    at.button(key="reveal").click().run()
    assert any("not a calibrated probability" in c.value for c in at.caption)
    prs, day = triage.load(), triage.FIRST_DAY
    empty = next(r for r in triage.repos(prs) if triage.ranked(prs, r, triage.moment(day), "A").empty)
    at = _at("watch")
    at.sidebar.selectbox[0].set_value(empty).run()
    at.slider(key="day").set_value(day).run()
    assert any("not a calibrated probability" in c.value for c in at.caption)


def test_the_game_cards_say_titles_are_as_of_data_collection():
    text = _text(_at("test_yourself"))
    assert "title (as of data collection)" in text and "only what was known" not in text


def test_the_importance_caption_says_it_is_a_sample():
    assert any("a sample of the test PRs" in c.value for c in _at("why").caption)


# ---------------------------------------------------------------------------
# robustness and guards
# ---------------------------------------------------------------------------

def test_the_app_uses_the_current_helpers_even_when_old_copies_are_cached(monkeypatch):
    """Streamlit Community Cloud re-runs app.py when the repo updates but keeps imported modules
    in memory, so the deployed app once ran a new app.py against an old triage.py. Recreate stale
    copies of every helper module and check the app reloads them."""
    monkeypatch.syspath_prepend(str(DEMO))
    for name, attr in (("triage", "random_pr"), ("charts", "why_bars"), ("chapters", "why")):
        monkeypatch.delattr(importlib.import_module(name), attr)
    at = _at("why")
    at.button(key="random_pr").click().run()
    assert not at.exception


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_the_demo_hard_codes_no_result_number(path):
    assert re.search(r"\b0\.\d{3}\b", path.read_text(encoding="utf-8")) is None


@pytest.mark.parametrize("pattern,reason", wc.RETRACTED, ids=lambda v: str(v)[:30])
def test_the_demo_avoids_retracted_phrasing(pattern, reason):
    for path in SOURCES:
        hit = re.search(pattern, path.read_text(encoding="utf-8"), flags=re.IGNORECASE)
        assert hit is None, f"{path.name} says {hit.group(0)!r}, but {reason}"


def test_the_demo_imports_only_what_its_own_requirements_install():
    """Streamlit Cloud installs demo/requirements.txt, not the project's: no sklearn, lightgbm,
    shap, or project module may be imported by the deployed files."""
    allowed = {"__future__", "dataclasses", "datetime", "importlib", "json", "math", "pathlib",
               "random", "re", "sys", "streamlit", "pandas", "altair", "triage", "charts", "chapters"}
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert mods <= allowed, f"demo/{path.name} imports {sorted(mods - allowed)}"


def test_demo_requirements_pin_only_what_the_app_needs_at_the_root_versions():
    """Streamlit, pandas, pyarrow and altair only, at the root file's versions. Pinning numpy as
    well blocked Python 3.14, which numpy 2.2.6 has no wheel for."""
    def pins(path):
        lines = [ln.split("#")[0].strip() for ln in path.read_text(encoding="utf-8").splitlines()]
        return dict(ln.split("==") for ln in lines if "==" in ln)
    demo, root = pins(DEMO / "requirements.txt"), pins(DEMO.parent / "requirements.txt")
    assert set(demo) == {"streamlit", "pandas", "pyarrow", "altair"}
    assert all(demo[p] == root[p] for p in demo)


def test_switching_the_model_mid_round_still_reveals_cleanly():
    """A presenter may flip the sidebar switch between picking and revealing."""
    at = _at("test_yourself")
    at.button(key="pick_0").click().run()
    at.sidebar.radio[0].set_value(UNSEEN).run()
    at.button(key="reveal").click().run()
    assert not at.exception
    rate = triage.game_hit_rate(triage.load(), "B")
    assert f"**{rate:.0%}**" in _markdown(at, "Over every round this game can deal")


def test_why_falls_back_to_the_repos_riskiest_pr_when_nobody_was_waiting():
    prs = triage.load()
    at_ = triage.moment(triage.DEFAULT_DAY)
    quiet = next(r for r in triage.repos(prs) if triage.awaiting_review(prs[prs["repo"] == r], at_).empty)
    top = prs[prs["repo"] == quiet].sort_values(["score_a", "number"], ascending=[False, True]).iloc[0]
    at = _at("why")
    at.sidebar.selectbox[0].set_value(quiet).run()
    assert not at.exception
    assert [m.value for m in at.metric] == [f"{top['score_a']:.2f}", f"{top['score_b']:.2f}"]

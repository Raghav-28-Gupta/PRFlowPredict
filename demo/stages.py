"""The "How it was built" section: a pipeline map and four deep dives (workflow-view spec, section 7).

Each page is a function that st.navigation runs through a two-line file in views/, like the story
chapters. It reads this session's Context from st.session_state (set by app.py), shows its
visual, its speaker note when the sidebar toggle is on, and a Previous/Next stage stepper. Widget
choices persist for the session, so a presenter can leave a page and come back to it. Every number
is computed at runtime by workflow.py from committed files; none is typed in here."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import chapters
import triage
import workflow
import workflow_charts as wc
from chapters import _c, _chart, _data

VERDICT_MEANING = {
    "PARTIAL_SHAP_ONLY": "the attribution pattern it predicted is there, but the intervention test "
                         "could not tell an effect from none, so why the model does not transfer "
                         "remains open."}
NOTES = {
    "pipeline": "Walk the six boxes left to right in about a minute: where the repos came from, what "
                "counts as a first review, features that use only what was knowable when a PR opened, "
                "two ways of holding data out, explaining the model, and shipping with every result "
                "tested. Then offer to open any stage.",
    "funnel": "Every narrowing step has a reason; click one. Point at the cap: it is why training and "
              "test rows do not add up. Then the limitation: each group's pool is its most-starred repos.",
    "known": "This is the leak the project designed out. Flip the naive switch: PRs opened in the last "
             "week jump into the lanes of outcomes nobody could know at t.",
    "designs": "Seen in training tests the future of known repos; Unseen repo holds whole repos out. Pick "
               "a fold and compare each dot with that fold's floor, not with zero.",
    "checks": "Stress validity, not success: these were written down before results were read. Point at "
              "the red tile: the pilot missed a guessed expectation, and the doc says why it was not a stop.",
}


def _note(key: str) -> None:
    if _c().notes:
        st.info(f"**Speaker note:** {NOTES[key]}")


def _page(url_path: str):
    """A "How it was built" page by its url_path, or None if the app does not register it."""
    return next((p for p in _c().wf_pages if p.url_path == url_path), None)


def _stepper(url_path: str) -> None:
    """Previous/Next stage buttons, placed by this page's position in the section."""
    pages = _c().wf_pages
    i = [p.url_path for p in pages].index(url_path)
    back, nxt, _ = st.columns([1, 1, 4])
    # keys are per page: a key shared by every page confuses the widget state across a switch
    if i > 0 and back.button("← Previous stage", key=f"wf_back_{i}"):
        st.switch_page(pages[i - 1])
    if i < len(pages) - 1 and nxt.button("Next stage →", key=f"wf_next_{i}", type="primary"):
        st.switch_page(pages[i + 1])


# ---------------------------------------------------------------------------
# the pipeline map
# ---------------------------------------------------------------------------

def _on_stage_click() -> None:
    key = workflow.picked(st.session_state.get("pipeline_map"), "stage", "key")
    if key in workflow.STAGE_KEYS:      # the browser can re-send a stale selection: ignore one we don't know
        st.session_state["wf_stage"] = key


def _collect(c) -> None:
    f = c.wf["repo_funnel"].set_index("step")["count"]
    st.markdown(f"{f['matched the searches']:,} repos matched the nine searches; {f['pooled as candidates']:,} "
                f"were pooled, {f['selected']} drawn and {f['kept']} kept after QC. The pilot gate ran on two "
                "repos before any of it.")


def _label(c) -> None:
    lab, n = c.wf["labels"], c.wf["pr_funnel"].set_index("step").loc["human-authored", "count"]
    st.markdown(f"Under D5, **{lab['d5']:.1%}** of the {n:,} labelled PRs are slow; under D3, which also "
                f"counts commenters with no tie to the repo, {lab['d3']:.1%}. A PR with no first review "
                "within 30 days is slow too, so no label is left waiting.")
    st.info(workflow.bot_note(lab))


def _features(c) -> None:
    a, groups = c.wf["audits"], pd.Series([g["group"] for g in c.wf["groups"]])
    counts = ", ".join(f"{int((groups == g).sum())} {label.lower()}" for g, label in workflow.GROUP_LABELS.items())
    st.markdown(f"{len(groups)} features: {counts}. A brute-force replay of {a['audit_n']} rows matched the "
                f"stored values with a largest difference of {a['audit_max_diff']:g}, and {a['live_matched']} "
                f"of {a['live_checks']} values were confirmed against live GitHub.")


def _evaluate(c) -> None:
    a, folds = c.wf["scenario_a"], c.wf["folds"]
    st.markdown(f"Seen in training: {a['n_train']:,} capped training rows from {a['train_repos']} repos, tested "
                f"on {a['n_test']:,} PRs from 2026. Unseen repo: {len(folds)} folds, each holding out "
                f"{folds['repos'].min()} to {folds['repos'].max()} repos whole.")
    st.caption("Not built: the survival model (Phase 5) and its C-index bar for unseen repos.")


def _explain(c) -> None:
    top = triage.importance("A", c.features).head(3)
    st.markdown("Top drivers of the seen-in-training model, by share of its attribution: "
                + "; ".join(f"{r.label} ({r.share:.0%})" for r in top.itertuples()) + ".")
    verdict = c.wf["verdict"]
    st.markdown(f"The pre-registered Phase 6b test returned `{verdict}`: "
                + VERDICT_MEANING.get(verdict, "see the report."))


def _ship(c) -> None:
    st.markdown("The results the README and the report cite are registered in `writeup_claims.py`, and a "
                "test recomputes each from committed files; another bans phrasings an earlier draft got "
                "wrong. This demo reads only committed files.")


DETAILS = {"collect": _collect, "label": _label, "features": _features, "evaluate": _evaluate,
           "explain": _explain, "ship": _ship}


def _panel(stage: workflow.Stage) -> None:
    c = _c()
    row = c.wf["stages"].set_index("key").loc[stage.key]
    st.subheader(f"{row['title']} · {stage.phases}")
    st.markdown(stage.summary)
    DETAILS[stage.key](c)
    st.markdown("**What it produced**\n" + "\n".join(
        f"- `{path}`: {what} · {'committed' if committed else 'not deployed (gitignored)'}"
        for path, committed, what in stage.artifacts))
    st.markdown(f"**Gate:** {row['gate']}. **Docs:** " + " · ".join(
        f"[{d.split('/')[-1]}]({chapters.GITHUB}/blob/main/{d})" for d in stage.docs))
    links = st.columns(3)
    if stage.page and _page(stage.page):
        links[0].page_link(_page(stage.page), label="Open the deep dive →")
    if stage.chapter is not None:
        links[1].page_link(c.pages[stage.chapter], label="Where the story uses it →")


def hub() -> None:
    c = _c()
    st.title("How it was built")
    st.markdown("Six stages took PRFlowPredict from a search of GitHub to the model in this demo, and "
                "each recorded its checks before its results were read. Click a stage, or pick it below.")
    if "wf_focus" in st.session_state:          # a story chapter asked for this stage
        st.session_state["wf_stage"] = st.session_state.pop("wf_focus")
    st.session_state.setdefault("wf_stage", workflow.STAGE_KEYS[0])
    stages = c.wf["stages"]
    _chart(wc.pipeline_map(stages, st.session_state["wf_stage"], c.mode), on_select=_on_stage_click,
           key="pipeline_map")
    titles = dict(zip(stages["key"], stages["title"]))
    key = st.segmented_control("Stage", workflow.STAGE_KEYS, format_func=titles.get, key="wf_stage",
                               required=True, persist_state="session", label_visibility="collapsed")
    _panel(workflow.stage(key))
    _note("pipeline")
    _stepper("pipeline")

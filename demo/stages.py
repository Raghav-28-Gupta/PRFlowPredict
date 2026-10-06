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


# ---------------------------------------------------------------------------
# data funnel
# ---------------------------------------------------------------------------

def _removed(event, steps: pd.DataFrame) -> None:
    step = workflow.picked(event, "step", "step")
    if step not in set(steps["step"]):
        st.caption("Click a step to see what it removed.")
        return
    frame, note = workflow.removed(step)
    st.markdown(f"**{step}:** {note}")
    if len(frame):
        st.dataframe(frame, hide_index=True, width="stretch")


def funnel() -> None:
    c, wf = _c(), _c().wf
    st.title("Data funnel")
    st.markdown("From every repo GitHub's searches matched to the pull requests this demo replays. "
                "Hover a step for its source file; click it to see what it removed and why.")
    left, right = st.columns(2)
    with left:
        st.subheader("Repos")
        event = _chart(wc.funnel(wf["repo_funnel"], "repos", c.mode, log=True), on_select="rerun",
                       key="repo_funnel")
        _removed(event, wf["repo_funnel"])
    with right:
        st.subheader("Pull requests")
        event = _chart(wc.funnel(wf["pr_funnel"], "pull requests", c.mode), on_select="rerun", key="pr_funnel")
        _removed(event, wf["pr_funnel"])
    st.caption(f"Training rows are capped at {workflow.CAP_FRAC:.0%} of all training rows per repo and test "
               "rows are not, so training rows and test PRs add up to fewer than the modelled PRs. QC rules "
               "are structural: bot share, human PR count and language.")
    st.info(workflow.pool_note(wf["pool_bias"]))
    with st.expander("Each group's search and pool"):
        st.dataframe(wf["pool_bias"], hide_index=True, width="stretch",
                     column_config={"share": st.column_config.NumberColumn("pool share of group", format="percent")})
    _data(pd.concat([wf["repo_funnel"].assign(unit="repos"), wf["pr_funnel"].assign(unit="pull requests")]),
          "Data behind the funnels")
    _note("funnel")
    _stepper("data-funnel")


# ---------------------------------------------------------------------------
# validity checks
# ---------------------------------------------------------------------------

def _tile_note(phase: str, check: int, wf: dict) -> str | None:
    if (phase, check) == ("2", 4):
        return "This check passes exactly at its threshold."
    if (phase, check) == ("3", 5) and wf["live"]:
        m = wf["live"][0]
        return (f"One live value differed at first: {m['pr']} {m['field']} read {m['live']} in GitHub's "
                f"search against {m['ours']} here, and the REST API confirmed this project's value.")
    return None


def _show_check(row: pd.Series, wf: dict) -> None:
    st.markdown(f"**{workflow.PHASE_NAMES[row['phase']]}, check {row['id']}:** {row['check']}")
    st.markdown(f"Status: **{row['status']}** · recorded: {workflow.gate_value_text(row['value'])}")
    note = _tile_note(row["phase"], int(row["id"]), wf)
    if note:
        st.markdown(note)
    with st.expander("Recorded value, raw"):
        if isinstance(row["value"], (dict, list)):
            st.json(row["value"])
        else:
            st.code(str(row["value"]), language=None)


def checks() -> None:
    c, wf = _c(), _c().wf
    st.title("Validity checks")
    st.markdown("**Validity, not success.** Each phase wrote down its checks before its results were "
                "read. They test that the pipeline did what it claims: rows conserved, no label in any "
                "feature, refits reproducing, attributions adding up. They do not say the model is good. "
                "Click a tile to read one.")
    g = wf["gates"]
    event = _chart(wc.gate_tiles(g, c.mode), on_select="rerun", key="gate_tiles")
    hit = workflow.picked(event, "check")
    row = g[(g["phase"] == str(hit.get("phase"))) & (g["id"] == hit.get("id"))] if hit else g.iloc[0:0]
    if len(row):                    # a stale selection from another page's data finds no row
        _show_check(row.iloc[0], wf)
    st.caption("Phase 1, the collection, has no gate file: the pilot gate tested its machinery first. "
               "Phase 5, a survival model, was not built.")
    _data(g.assign(value=g["value"].map(workflow.gate_value_text)), "Every check")
    _note("checks")
    _stepper("validity-checks")


# ---------------------------------------------------------------------------
# known at time t
# ---------------------------------------------------------------------------

DEFAULT_REPO = "kdlbs/kandev"


def _feature_kinds(c) -> None:
    labels = {f["feature"]: f["label"] for f in c.features}
    groups = pd.DataFrame(c.wf["groups"])
    for col, (g, label) in zip(st.columns(len(workflow.GROUP_LABELS)), workflow.GROUP_LABELS.items()):
        names = groups.loc[groups["group"] == g, "feature"].map(labels)
        col.markdown(f"**{label}** ({len(names)})")
        col.caption(" · ".join(names))


def known_at_t() -> None:
    c, wf = _c(), _c().wf
    st.title("Known at time t")
    st.markdown("A score is only useful if it could be computed when the PR opened. Each of the model's "
                "features is one of four kinds, by when its value is known:")
    _feature_kinds(c)
    a = wf["audits"]
    snapshot = sum(g["group"] == "snapshot" for g in wf["groups"])
    st.markdown(f"The {snapshot} snapshot features are the stated exception: 2026 values applied to earlier "
                f"PRs. Two audits check the rest: a brute-force replay of {a['audit_n']} rows matched the "
                f"stored features with a largest difference of {a['audit_max_diff']:g}, and {a['live_matched']} "
                f"of {a['live_checks']} values on {a['live_rows']} PRs were confirmed against live GitHub.")

    st.subheader("Replay one repo's history")
    st.markdown("The trailing 90-day slow rate is both the baseline and one of the model's strongest "
                "features. When a PR opens at time t, an earlier PR's outcome counts only if it was "
                "knowable: the PR is at least 7 days old, or it already had its first review. Pick a "
                "moment and see which earlier PRs count.")
    exact = wf["exact"]
    repo = st.selectbox("Repo", exact, index=exact.index(DEFAULT_REPO) if DEFAULT_REPO in exact else 0,
                        key="kt_repo", persist_state="session")
    rows = c.prs[c.prs["repo"] == repo].sort_values(["created_at", "pr_id"], kind="mergesort")
    names = {p: f"#{n} · {d:%d %b %Y %H:%M}" for p, n, d in zip(rows["pr_id"], rows["number"], rows["created_at"])}
    if st.session_state.get("kt_pr") not in names:          # first visit, or a different repo
        st.session_state["kt_pr"] = workflow.default_pr(wf["check"], repo)
    pr_id = st.select_slider("Pull request, by when it opened", options=list(names), format_func=names.get,
                             key="kt_pr", persist_state="session")
    naive = st.toggle("Naive rule: count every earlier PR in the window, knowable or not", key="kt_naive",
                      persist_state="session")
    frame, s = workflow.replay_at(c.prs, repo, pr_id, "naive" if naive else "resolvable", wf["prior"])
    domain = (s["t"] - pd.Timedelta(days=workflow.TRAILING_DAYS + 30), s["t"] + pd.Timedelta(days=2))
    _chart(wc.replay_strip(frame, s["t"], c.mode, domain))
    near = rows[(rows["created_at"] >= domain[0]) & (rows["created_at"] <= domain[1])]
    _chart(wc.trailing_line(near, s["t"], s["prior"], c.mode, domain))
    if naive:
        st.markdown(f"The naive rule counts **{s['k']}** earlier PRs, **{s['unknown']}** of them with an "
                    f"outcome nobody could know at t, and gives **{s['rate']:.3f}**; the stored feature is "
                    f"{s['stored_rate']:.3f}. This is a counterfactual computed here, not a project result.")
    else:
        st.markdown(f"At t, PR #{s['number']} opens. **{s['k']}** earlier PRs in the 90-day window had a "
                    f"knowable outcome and **{s['slow']}** of them stalled; **{s['unknown']}** more opened "
                    "in the last 7 days without a review yet, so they are left out. Shrunk toward the prior: "
                    f"({s['slow']} + {workflow.ALPHA:g} × {s['prior']:.3f}) / ({s['k']} + "
                    f"{workflow.ALPHA:g}) = **{s['rate']:.3f}**.")
        st.markdown("✓ Matches the stored feature." if s["matches"] else "✗ Differs from the stored feature.")
    st.caption(f"The repo list holds the {len(exact)} of {len(triage.repos(c.prs))} repos where this page "
               "re-derives the stored feature exactly on every PR; for the others, the project's replay "
               "also saw PRs this extract does not hold. Only the trailing slow rate is re-derived here: "
               "backlog and author-history features need bot PRs and author identities the extract leaves out.")
    _data(frame, "Data behind the strip")
    _note("known")
    _stepper("known-at-time-t")


# ---------------------------------------------------------------------------
# two test designs
# ---------------------------------------------------------------------------

DESIGNS = {"Seen in training (time cut)": None, "Unseen repo (repo folds)": "B"}


def designs() -> None:
    c, wf = _c(), _c().wf
    st.title("Two test designs")
    st.markdown("A model is judged on data it was never shown. PRFlowPredict holds data out in two ways "
                "and reports both.")
    st.session_state.setdefault("td_design", next(iter(DESIGNS)))
    design = st.segmented_control("Test design", list(DESIGNS), key="td_design", required=True,
                                  persist_state="session")
    rows, folds = wf["splits"], wf["folds"]
    order = list(rows["repo"])
    m = st.columns(3)
    if DESIGNS[design] is None:
        a = wf["scenario_a"]
        m[0].metric("Training rows", f"{a['n_train']:,}")
        m[1].metric("Test PRs", f"{a['n_test']:,}")
        m[2].metric("Repos trained on / tested", f"{a['train_repos']} / {a['test_repos']}")
        st.markdown(f"Each repo's PRs opened before 2026 train the model, capped at {workflow.CAP_FRAC:.0%} "
                    "of the training rows per repo; its PRs from 2026 test it. The question: how well does it "
                    "predict the future of projects it knows?")
        long = workflow.design_rows(rows)
    else:
        st.session_state.setdefault("td_fold", 1)
        number = st.segmented_control("Held-out fold", list(range(1, len(folds) + 1)), key="td_fold",
                                      format_func=lambda f: f"fold {f}", required=True, persist_state="session")
        r = folds.set_index("fold").loc[number - 1]
        m[0].metric("Training rows", f"{int(r['n_train']):,}")
        m[1].metric("Test PRs", f"{int(r['n_test']):,}")
        m[2].metric("Repos held out", f"{int(r['repos'])}")
        st.markdown(f"The repos are split into {len(folds)} folds. Each model trains on every PR of the other "
                    "folds' repos, capped the same way, and is tested on every PR of its fold's repos, from "
                    "the whole window. The question: how well does it predict for a project it has never seen?")
        long = workflow.design_rows(rows, number - 1)
    _chart(wc.split_bars(long, order, c.mode))
    if DESIGNS[design] is not None:
        st.subheader("This fold's results")
        _chart(wc.fold_dots(folds, number - 1, c.mode))
        st.caption("AUC-PR starts from the fold's base rate (the tick), not from zero, and the folds' base "
                   "rates differ widely: compare each dot with its own fold's tick.")
    st.markdown(
        "- Per-repo counts come from a Phase 6b output file, `data/phase6b_transfer.csv`; they reproduce "
        "every stored training count exactly.\n"
        "- Tuning used the seen-in-training rows, which include the pre-2026 rows of the repos each "
        "unseen-repo fold later holds out. Its effect was not measured.\n"
        "- The story's Unseen repo switch shows only 2026 PRs; the fold results cover the whole window.\n"
        "- Compare the designs with AUC-PR, not precision on the top 10: the unseen-repo test ranks far "
        "larger pools of PRs per repo.")
    _data(folds, "Data behind the fold results")
    _note("designs")
    _stepper("test-designs")

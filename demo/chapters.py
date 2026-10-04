"""The demo's six chapters, for a 3-5 minute live walkthrough (spec section 7).

Each chapter is a function that st.navigation runs as a page. The app shell (app.py) sets `ctx`
before navigation runs; a chapter reads its data and the sidebar's choices from it, shows one
visual, its speaker note when the sidebar toggle is on, and Back/Next buttons. Every number
shown is computed at runtime from committed files; none is typed in here."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

import charts
import triage

GITHUB = "https://github.com/Raghav-28-Gupta/PRFlowPredict"
README, REPORT = f"{GITHUB}#readme", f"{GITHUB}/blob/main/docs/REPORT.md"
MODELS = {"Seen in training (within-project)": "A", "Unseen repo (cold-start)": "B"}
MODEL_NAMES = {"A": "Seen in training", "B": "Unseen repo"}
MODEL_NOTES = {
    "A": "**Seen in training:** scores from one model trained on the PRs opened before 2026 in "
         "all {n} repos, so it has seen this repo's earlier PRs, if it had any.",
    "B": "**Unseen repo:** scores from the model trained without this repo, as if it were new "
         "to the model.",
}
GAME_FACTS = ["additions_at_open", "deletions_at_open", "n_commits_at_open", "n_prior_prs_here",
              "author_prior_slow_rate_here", "trailing_90d_slow_rate", "body_len", "is_draft_at_open"]
NOTES = {
    "problem": "Open with the pain: some PRs are reviewed within the hour, but about half never get "
               "a first review within a week, and most of those are closed without one. The goal is "
               "to flag the risky ones the moment they are opened.",
    "watch": "This is the 2026 test period, which the model never trained on. Each dot is a real PR "
             "scored when it opened. Drag the day: solid dots are the ones waiting then. Click a "
             "high-risk orange dot to explain it next.",
    "why": "Exact SHAP attribution. Point at the biggest bar: usually the author's track record in "
           "this repo. Then the overall chart: the model leans on history features most. Stress "
           "attribution, not cause.",
    "test": "Hand the choice to the interviewer. After the reveal, point at the hit-rate line: over "
            "every round the game can deal, the model beats a random guess clearly, but it is far "
            "from perfect.",
    "transfer": "The honest headline. Within a project the model beats the baseline. On repos it "
                "has never seen, it beats the baseline only with the repo's slow-rate history; "
                "without it, it falls below. Then flip the sidebar switch to Unseen repo.",
    "limits": "Close on what was not measured. This is what makes the rest credible.",
}


@dataclass
class Context:
    prs: pd.DataFrame
    features: list[dict]
    results: dict
    hit_rates: dict
    scenario: str
    repo: str
    notes: bool
    mode: str
    pages: list = field(default_factory=list)


ctx: Context | None = None


def _note(key: str) -> None:
    if ctx.notes:
        st.info(f"**Speaker note:** {NOTES[key]}")


def _nav(i: int) -> None:
    back, nxt, _ = st.columns([1, 1, 6])
    # keys are per chapter: a key shared by every page confuses the widget state across a switch
    if i > 0 and back.button("← Back", key=f"back_{i}"):
        st.switch_page(ctx.pages[i - 1])
    if i < len(ctx.pages) - 1 and nxt.button("Next →", key=f"next_{i}", type="primary"):
        st.switch_page(ctx.pages[i + 1])


def _data(frame: pd.DataFrame, label: str = "Data behind this chart") -> None:
    with st.expander(label):
        st.dataframe(frame, hide_index=True, width="stretch")


def _chart(chart, **kwargs):
    return st.altair_chart(chart, width="stretch", **kwargs)


# ---------------------------------------------------------------------------
# 1. the problem
# ---------------------------------------------------------------------------

def problem() -> None:
    st.title("Some pull requests wait weeks for a first review")
    buckets = triage.wait_buckets(ctx.prs)
    quick = buckets.loc[buckets["bucket"].isin(triage.WAIT_BUCKETS[:2]), "share"].sum()
    stalled = buckets.loc[buckets["stalled"], "share"].sum()
    st.markdown(
        f"Of the {len(ctx.prs):,} pull requests opened in {len(triage.repos(ctx.prs))} projects "
        f"from January to June 2026, {quick:.0%} got a first review within a day, but "
        f"**{stalled:.0%} stalled**: no first review within 7 days, and most of those were closed "
        "without ever getting one. PRFlowPredict scores every PR the moment it is opened, so a "
        "maintainer can see which ones are at risk.")
    st.altair_chart(charts.wait_histogram(buckets, ctx.mode), width="content")
    st.caption("A first review is the first review or comment by a human other than the author, "
               "with a tie to the repo: the project's D5 label.")
    _data(buckets)
    _note("problem")
    _nav(0)


# ---------------------------------------------------------------------------
# 2. watch it work
# ---------------------------------------------------------------------------

def _triage_list(day) -> None:
    """The Phase 7 triage list for the chosen day, with its tally and caveats unchanged."""
    listing = triage.ranked(ctx.prs, ctx.repo, triage.moment(day), ctx.scenario)
    if listing.empty:
        note = "No PR in this repo was waiting for a first review at the start of that day."
        if day <= triage.EARLY_JANUARY:
            note += " Only PRs opened from 2026-01-01 have scores, so early January looks sparse."
        st.info(note)
        return
    stalled, k = triage.tally(listing)
    st.markdown(
        f"**Of the top {k} by risk, {stalled} stalled; of all {len(listing)} PRs waiting, "
        f"{int(listing['stalled'].sum())} did** (stalled: no first review within 7 days of "
        "opening). Most PRs still waiting at a given moment go on to stall, and any already "
        "waiting 7 days has stalled by definition, so this list cannot show how well the model "
        f"ranks. That is measured when PRs open, over every test PR: see the [README]({README}).")
    st.dataframe(
        listing, hide_index=True, width="stretch",
        column_order=["rank", "risk", "url", "title", "days_waited", "drivers", "outcome"],
        column_config={
            "rank": st.column_config.NumberColumn("Rank"),
            "risk": st.column_config.NumberColumn("Risk", format="%.2f"),
            "url": st.column_config.LinkColumn("PR", display_text=r"/pull/(\d+)$"),
            "title": st.column_config.TextColumn("Title"),
            "days_waited": st.column_config.NumberColumn("Days waited", format="%.1f"),
            "drivers": st.column_config.TextColumn("Top drivers"),
            "outcome": st.column_config.TextColumn("What happened"),
        })
    st.caption(
        "Risk is a ranking score, not a calibrated probability. Top drivers are the three features "
        "that moved this PR's score most, by exact TreeSHAP: model attribution, not cause (↑ pushed "
        "the risk up, ↓ down). Titles are as of data collection.")


def watch() -> None:
    st.title("Watch it work")
    st.markdown(
        "A replay of the 2026 test period PRFlowPredict was evaluated on, not live data. Each dot "
        f"is one pull request in **{ctx.repo}**, placed by the day it opened and the risk score the "
        "model gave it **when it was opened**. Scores are never updated as a PR waits. Only PRs "
        "opened from 1 January 2026 have scores, so older PRs still waiting are not shown. Drag "
        "the day to see who was waiting for a first review; click a dot to see why it scored as "
        "it did.")
    st.markdown(MODEL_NOTES[ctx.scenario].format(n=len(triage.repos(ctx.prs))))
    day = st.slider("Day", min_value=triage.FIRST_DAY, max_value=triage.LAST_DAY,
                    value=triage.DEFAULT_DAY, key="day", format="D MMM YYYY")
    at = triage.moment(day)
    frame = triage.timeline(ctx.prs, ctx.repo, ctx.scenario, at)
    event = _chart(charts.timeline(frame, at, ctx.mode), on_select="rerun", key="timeline")
    picked = triage.selected_pr_id(event)
    if picked:
        st.session_state["selected_pr"], st.session_state["pr_ref"] = picked, ""
        row = frame[frame["pr_id"] == picked].iloc[0]
        st.success(f"Selected #{row['number']}: {triage.md_escape(row['title'])}. "
                   "The next chapter shows why it scored as it did.")
    _triage_list(day)
    _data(frame, "Data behind the timeline")
    _note("watch")
    _nav(1)


# ---------------------------------------------------------------------------
# 3. why it decides
# ---------------------------------------------------------------------------

def _lookup() -> None:
    parsed = triage.parse_pr_ref(st.session_state.get("pr_ref", ""), ctx.repo)
    hit = triage.find_pr(ctx.prs, *parsed) if parsed else ctx.prs.iloc[0:0]
    if len(hit):
        st.session_state["selected_pr"] = hit["pr_id"].iloc[0]


def _pick_random() -> None:
    pr = triage.random_pr(ctx.prs).iloc[0]
    st.session_state["pr_ref"] = f"{pr['repo']}#{int(pr['number'])}"
    st.session_state["selected_pr"] = pr["pr_id"]


def _current_pr() -> pd.DataFrame:
    """The selected PR; with none selected, the riskiest PR waiting on the default day in the
    chosen repo, or the repo's riskiest PR if none was waiting."""
    chosen = ctx.prs[ctx.prs["pr_id"] == st.session_state.get("selected_pr")]
    if len(chosen):
        return chosen
    score = triage.SCORES[ctx.scenario]
    repo = ctx.prs[ctx.prs["repo"] == ctx.repo]
    waiting = triage.awaiting_review(repo, triage.moment(triage.DEFAULT_DAY))
    pool = waiting if len(waiting) else repo
    return pool.sort_values([score, "number"], ascending=[False, True], kind="mergesort").head(1)


def _card(pr: pd.DataFrame) -> None:
    row = pr.iloc[0]
    st.markdown(f"#### {triage.md_escape(row['title'])}")
    st.markdown(
        f"[{row['repo']}#{row['number']}]({row['url']}) · opened "
        f"{row['created_at']:%Y-%m-%d %H:%M} UTC · what happened: **{triage.outcome(pr).iloc[0]}**")
    for col, (label, sc) in zip(st.columns(2), MODELS.items()):
        share, n = triage.rank_in_repo(ctx.prs, pr, sc)
        col.metric(label, f"{row[triage.SCORES[sc]]:.2f}")
        col.markdown(f"Higher than {share:.0%} of this repo's other {n:,} replayed PRs.")


def why() -> None:
    st.title("Why it decides")
    st.markdown(
        "Pick any replayed PR: paste its GitHub link, type `owner/repo#123` or a number in the repo "
        "chosen in the sidebar, click a dot in *Watch it work*, or let **Random PR** pick one.")
    box, button = st.columns([5, 1], vertical_alignment="bottom")
    ref = box.text_input("PR", key="pr_ref", on_change=_lookup,
                         placeholder=f"https://github.com/{ctx.repo}/pull/...")
    button.button("Random PR", key="random_pr", on_click=_pick_random)
    parsed = triage.parse_pr_ref(ref, ctx.repo) if ref.strip() else None
    if ref.strip() and (parsed is None or triage.find_pr(ctx.prs, *parsed).empty):
        st.warning(
            "Not in the replay. The lookup covers the PRs opened from 1 January to 30 June 2026 in "
            f"the {len(triage.repos(ctx.prs))} repos; try one from the triage list, or press Random PR.")
    else:
        if parsed:
            _lookup()
        pr = _current_pr()
        _card(pr)
        name = MODEL_NAMES[ctx.scenario]
        rows, base = triage.why(pr, ctx.scenario, ctx.features)
        st.subheader(f"Why the {name} model scored it {pr[triage.SCORES[ctx.scenario]].iloc[0]:.2f}")
        _chart(charts.why_bars(rows, ctx.mode))
        st.caption(
            "Exact TreeSHAP: each bar is how far that feature pushed this PR's score on the model's "
            f"log-odds scale, starting from the model's average of {base:+.2f}; together they add up "
            "to the score shown. Model attribution, not cause. Risk is a ranking score, not a "
            "calibrated probability.")
        _data(rows)
    st.subheader("What the model leans on overall")
    imp = triage.importance(ctx.scenario, ctx.features)
    _chart(charts.importance_bars(imp, mode=ctx.mode))
    st.caption("Each feature's share of the model's total attribution over the test PRs (Phase 6).")
    _note("why")
    _nav(2)


# ---------------------------------------------------------------------------
# 4. test yourself
# ---------------------------------------------------------------------------

def _new_round(game: dict) -> None:
    game.update(ids=list(triage.deal(ctx.prs, random.Random())["pr_id"]), pick=None, revealed=False)


def _choose(i: int) -> None:
    st.session_state["game"]["pick"] = i


def _reveal() -> None:
    game = st.session_state["game"]
    round_ = ctx.prs.set_index("pr_id").loc[game["ids"]].reset_index()
    game["model_pick"] = triage.model_pick(round_, ctx.scenario)
    game["revealed"] = True
    game["played"] += 1
    game["you"] += int(round_["is_slow"].iloc[game["pick"]])
    game["model"] += int(round_.set_index("pr_id").loc[game["model_pick"], "is_slow"])


def _deal_again() -> None:
    _new_round(st.session_state["game"])


def test_yourself() -> None:
    st.title("Test yourself")
    st.markdown(
        "Four pull requests opened in the same project in the same week. **Exactly one of them "
        "stalled**: no first review within 7 days. Each card shows only what was known when it "
        "opened. Which one stalled?")
    game = st.session_state.setdefault("game", {"ids": None, "pick": None, "revealed": False,
                                                "played": 0, "you": 0, "model": 0})
    if game["ids"] is None:
        _new_round(game)
    round_ = ctx.prs.set_index("pr_id").loc[game["ids"]].reset_index()
    meta = {f["feature"]: f for f in ctx.features}
    for i, (col, (_, pr)) in enumerate(zip(st.columns(4), round_.iterrows())):
        with col.container(border=True):
            st.markdown(f"**PR {i + 1}** · {pr['repo']}")
            st.markdown(triage.md_escape(pr["title"]))
            st.markdown("\n".join(
                f"- {meta[f]['label']}: **{triage.format_value(pr[f'x__{f}'], meta[f]['dtype'], meta[f]['rate'])}**"
                for f in GAME_FACTS))
            if not game["revealed"]:
                st.button("This one", key=f"pick_{i}", on_click=_choose, args=(i,),
                          type="primary" if game["pick"] == i else "secondary")
            else:
                marks = [m for m, on in (("your pick", game["pick"] == i),
                                         ("the model's pick", game.get("model_pick") == pr["pr_id"])) if on]
                verdict = "**Stalled**" if pr["is_slow"] else "Reviewed within 7 days"
                st.markdown(f"{verdict} · risk {pr[triage.SCORES[ctx.scenario]]:.2f} · "
                            f"[#{pr['number']}]({pr['url']})" + (f"  \n← {', '.join(marks)}" if marks else ""))
    if not game["revealed"]:
        if game["pick"] is not None:
            st.button("Reveal", key="reveal", type="primary", on_click=_reveal)
    else:
        you_right = bool(round_["is_slow"].iloc[game["pick"]])
        model_right = bool(round_.set_index("pr_id").loc[game["model_pick"], "is_slow"])
        st.markdown(f"You picked PR {game['pick'] + 1}: {'✓ right' if you_right else '✗ wrong'}. "
                    f"The model picked PR {list(round_['pr_id']).index(game['model_pick']) + 1}: "
                    f"{'✓ right' if model_right else '✗ wrong'}.")
        st.markdown(f"This session: you {game['you']} of {game['played']}, the model "
                    f"{game['model']} of {game['played']}.")
        st.button("Deal again", key="deal", type="primary", on_click=_deal_again)
    st.markdown(
        f"Over every round this game can deal, the {MODEL_NAMES[ctx.scenario]} model picks the PR "
        f"that stalled **{ctx.hit_rates[ctx.scenario]:.0%}** of the time; a random guess gets "
        f"**{1 / triage.GAME_SIZE:.0%}**. Rounds are drawn like this: a project and week at random, "
        "then one PR from it that stalled and three that did not.")
    _note("test")
    _nav(3)


# ---------------------------------------------------------------------------
# 5. does it transfer?
# ---------------------------------------------------------------------------

def transfer() -> None:
    st.title("Does it transfer to a new project?")
    t = ctx.results["text"]
    st.markdown(
        "**Within a project, it works.** On repos it was trained on, the model's precision on each "
        f"repo's 10 riskiest PRs is {t['a_p10']} (95% interval, resampling repos), against "
        f"{t['a_baseline_p10']} for the trailing-rate baseline and {t['a_random_p10']} for a random pick.")
    _chart(charts.p10_chart(ctx.results, ctx.mode))
    st.markdown(
        "**On repos it has never seen, it needs the repo's slow-rate history.** With it, the model "
        f"beats the baseline (AUC-PR {t['b_full_vs_baseline']}); without those four features it "
        f"falls below: {t['b_nlr_vs_baseline']}. Each dot is one held-out fold of repos.")
    _chart(charts.results_chart(ctx.results, ctx.mode))
    st.markdown("Try it: switch the sidebar model to **Unseen repo** and go back to *Watch it work* "
                f"or *Test yourself*. The full evaluation is in the [report]({REPORT}).")
    _data(ctx.results["folds"], "Data behind the AUC-PR chart")
    _note("transfer")
    _nav(4)


# ---------------------------------------------------------------------------
# 6. honest limits
# ---------------------------------------------------------------------------

def limits() -> None:
    st.title("Honest limits")
    st.markdown(
        "- **The cold-start bar was never measured.** The project plan set a C-index target for a "
        "survival model on unseen repos; that phase was not built.\n"
        f"- **{len(triage.repos(ctx.prs))} repos is a small sample** for any claim about projects in "
        "general, and results swing widely between held-out folds.\n"
        "- **Repo attributes are 2026 snapshots** (maintainer counts, CODEOWNERS and the like), "
        "applied to earlier PRs.\n"
        "- **A list of PRs still waiting cannot measure ranking.** Most of them go on to stall; the "
        "ranking is measured when PRs open, over every test PR.")
    st.markdown(f"Everything here is in the [README]({README}) and the [report]({REPORT}), where a "
                "test checks every number against the committed data.")
    _note("limits")
    _nav(5)

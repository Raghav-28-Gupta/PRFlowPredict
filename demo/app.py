"""PRFlowPredict demo: a replay of the evaluated 2026 test period.

    pip install -r demo/requirements.txt
    streamlit run demo/app.py

A thin layer over triage.py. It hard-codes no result number: the measured results live in
the README and the report, where tests/test_writeup_claims.py checks them."""
import importlib
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))      # triage.py sits next to this file
import triage  # noqa: E402

# Streamlit Community Cloud re-runs this script when the repo updates but keeps imported
# modules in memory, so a new app.py could meet an old triage.py. Reload it every run.
triage = importlib.reload(triage)

GITHUB = "https://github.com/Raghav-28-Gupta/PRFlowPredict"
MODELS = {"Seen in training (within-project)": "A", "Unseen repo (cold-start)": "B"}
MODEL_NOTES = {
    "A": "**Seen in training:** scores from one model trained on the PRs opened before 2026 in "
         "all {n} repos, so it has seen this repo's earlier PRs, if it had any.",
    "B": "**Unseen repo:** scores from the model trained without this repo, as if it were new "
         "to the model.",
}

st.set_page_config(page_title="PRFlowPredict demo", layout="wide")


@st.cache_data
def _load():
    return triage.load()


prs = _load()
repo_names = triage.repos(prs)
start = triage.default_repo(prs, triage.moment(triage.DEFAULT_DAY))

repo = st.sidebar.selectbox("Repo", repo_names, index=repo_names.index(start))
day = st.sidebar.date_input("Day", value=triage.DEFAULT_DAY,
                            min_value=triage.FIRST_DAY, max_value=triage.LAST_DAY)
scenario = MODELS[st.sidebar.radio("Model", list(MODELS))]

st.title("Which pull requests will stall?")
st.markdown(
    "A replay of the 2026 test period PRFlowPredict was evaluated on, not live data. Pick a "
    "repo and a day: the list shows the PRs that were waiting for their first review at the "
    "start of that day (00:00 UTC), ranked by the risk score the model gave each one **when it "
    "was opened**. Scores are never updated as a PR waits. Only PRs opened from 1 January 2026 "
    "have scores, so older PRs still waiting are not listed.")
st.markdown(MODEL_NOTES[scenario].format(n=len(repo_names)))
st.markdown(
    "Within a project the model ranks well; on unseen repos its advantage depends on the "
    "repo's slow-rate history. The measured results, with their caveats, are in the "
    f"[README]({GITHUB}#readme) and the [report]({GITHUB}/blob/main/docs/REPORT.md).")


def _pick_random():
    pr = triage.random_pr(prs).iloc[0]
    st.session_state["pr_ref"] = f"{pr['repo']}#{int(pr['number'])}"


triage_tab, lookup_tab = st.tabs(["Triage list", "Look up a PR"])

with triage_tab:
    listing = triage.ranked(prs, repo, triage.moment(day), scenario)
    if listing.empty:
        note = "No PR in this repo was waiting for a first review at the start of that day."
        if day <= triage.EARLY_JANUARY:
            note += " Only PRs opened from 2026-01-01 have scores, so early January looks sparse."
        st.info(note)
    else:
        stalled, k = triage.tally(listing)
        st.markdown(
            f"**Of the top {k} by risk, {stalled} stalled; of all {len(listing)} PRs waiting, "
            f"{int(listing['stalled'].sum())} did** (stalled: no first review within 7 days of "
            "opening). Most PRs still waiting at a given moment go on to stall, and any already "
            "waiting 7 days has stalled by definition, so this list cannot show how well the model "
            f"ranks. That is measured when PRs open, over every test PR: see the [README]({GITHUB}#readme).")
        st.dataframe(
            listing,
            hide_index=True,
            width="stretch",
            column_order=["rank", "risk", "url", "title", "days_waited", "drivers", "outcome"],
            column_config={
                "rank": st.column_config.NumberColumn("Rank"),
                "risk": st.column_config.NumberColumn("Risk", format="%.2f"),
                "url": st.column_config.LinkColumn("PR", display_text=r"/pull/(\d+)$"),
                "title": st.column_config.TextColumn("Title"),
                "days_waited": st.column_config.NumberColumn("Days waited", format="%.1f"),
                "drivers": st.column_config.TextColumn("Top drivers"),
                "outcome": st.column_config.TextColumn("What happened"),
            },
        )
        st.caption(
            "Risk is a ranking score, not a calibrated probability. "
            "Top drivers are the three features that moved this PR's score most, by exact "
            "TreeSHAP: model attribution, not cause (↑ pushed the risk up, ↓ down). Titles are as "
            "of data collection. A first review is the first review or comment by a human other "
            "than the author, with a tie to the repo: the project's D5 label.")

with lookup_tab:
    st.markdown(
        "Look up any PR in the replay: paste its GitHub link, type `owner/repo#123`, or type a "
        "number in the repo chosen in the sidebar. Or let **Random PR** pick one.")
    box, button = st.columns([5, 1], vertical_alignment="bottom")
    ref = box.text_input("PR", key="pr_ref", placeholder=f"https://github.com/{repo}/pull/...")
    button.button("Random PR", key="random_pr", on_click=_pick_random)
    if ref.strip():
        parsed = triage.parse_pr_ref(ref, repo)
        pr = triage.find_pr(prs, *parsed) if parsed else prs.iloc[0:0]
        if pr.empty:
            st.warning(
                "Not in the replay. The lookup covers the PRs opened from 1 January to 30 June "
                f"2026 in the {len(repo_names)} repos; try one from the triage list, or press "
                "Random PR.")
        else:
            row = pr.iloc[0]
            st.markdown(f"#### {triage.md_escape(row['title'])}")
            st.markdown(
                f"[{row['repo']}#{row['number']}]({row['url']}) · opened "
                f"{row['created_at']:%Y-%m-%d %H:%M} UTC · what happened: "
                f"**{triage.outcome(pr).iloc[0]}**")
            for col, (label, sc) in zip(st.columns(2), MODELS.items()):
                share, n = triage.rank_in_repo(prs, pr, sc)
                col.metric(label, f"{row[triage.SCORES[sc]]:.2f}")
                col.markdown(f"Higher than {share:.0%} of this repo's other {n:,} replayed PRs.")
                col.markdown(f"**Top drivers:** {triage.md_escape(row[triage.DRIVERS[sc]])}")
            st.caption(
                "Both scores were given when the PR was opened and never updated afterwards. "
                "Risk is a ranking score, not a calibrated probability, and drivers are model "
                "attribution, not cause.")

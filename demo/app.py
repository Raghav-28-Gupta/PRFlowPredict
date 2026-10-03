"""PRFlowPredict demo: a replay of the evaluated 2026 test period.

    pip install -r demo/requirements.txt
    streamlit run demo/app.py

A thin layer over triage.py. It hard-codes no result number: the measured results live in
the README and the report, where tests/test_writeup_claims.py checks them."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))      # triage.py sits next to this file
import triage  # noqa: E402

GITHUB = "https://github.com/Raghav-28-Gupta/PRFlowPredict"
MODELS = {"Seen in training (within-project)": "A", "Unseen repo (cold-start)": "B"}
MODEL_NOTES = {
    "A": "**Seen in training:** scores from the model trained on this repo's own earlier PRs "
         "(those opened before 2026).",
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
    "was opened**. Scores are never updated as a PR waits.")
st.markdown(MODEL_NOTES[scenario])
st.markdown(
    "Within a project the model ranks well; on unseen repos its advantage depends on the "
    "repo's slow-rate history. The measured results, with their caveats, are in the "
    f"[README]({GITHUB}#readme) and the [report]({GITHUB}/blob/main/docs/REPORT.md).")

listing = triage.ranked(prs, repo, triage.moment(day), scenario)
if listing.empty:
    note = "No PR in this repo was waiting for a first review at the start of that day."
    if day <= triage.EARLY_JANUARY:
        note += " Only PRs opened from 2026-01-01 have scores, so early January looks sparse."
    st.info(note)
else:
    stalled, k = triage.tally(listing)
    st.markdown(f"**Of the top {k} by risk, {stalled} stalled** "
                "(no first review within 7 days of opening).")
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
        "Top drivers are the three features that moved this PR's score most, by exact "
        "TreeSHAP: model attribution, not cause (↑ pushed the risk up, ↓ down). Titles are as "
        "of data collection. A first review is the first review or comment by a human other "
        "than the author, with a tie to the repo: the project's D5 label.")

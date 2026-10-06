"""PRFlowPredict demo: an interactive walkthrough of the evaluated 2026 test period.

    pip install -r demo/requirements.txt
    streamlit run demo/app.py

A thin shell: page config, cached data, the sidebar, the six story chapters in chapters.py and
the "How it was built" pages in stages.py. It hard-codes no result number: every number shown is
computed from committed files, and the results chapter's numbers are tested against the README's."""
import importlib
import sys
from pathlib import Path

import streamlit as st

HERE = str(Path(__file__).parent)
if HERE not in sys.path:
    sys.path.insert(0, HERE)            # the helper modules sit next to this file
import triage  # noqa: E402
import charts  # noqa: E402
import workflow  # noqa: E402
import workflow_charts  # noqa: E402
import chapters  # noqa: E402
import stages  # noqa: E402

# Streamlit Community Cloud re-runs this script when the repo updates but keeps imported modules
# in memory, so a new app.py could meet old helpers. Reload them every run, dependencies first.
triage = importlib.reload(triage)
charts = importlib.reload(charts)
workflow = importlib.reload(workflow)
workflow_charts = importlib.reload(workflow_charts)
chapters = importlib.reload(chapters)
stages = importlib.reload(stages)

st.set_page_config(page_title="PRFlowPredict demo", layout="wide")


@st.cache_data
def _data():
    prs = triage.load()
    return (prs, triage.load_features(), triage.results(),
            {sc: triage.game_hit_rate(prs, sc) for sc in triage.SCORES})


@st.cache_data
def _workflow(_prs):
    return workflow.bundle(_prs)          # the leading underscore: Streamlit does not hash the frame


prs, features, results, hit_rates = _data()
wf = _workflow(prs)
repo_names = triage.repos(prs)
start = triage.default_repo(prs, triage.moment(triage.DEFAULT_DAY))

st.sidebar.markdown("### PRFlowPredict")
scenario = chapters.MODELS[st.sidebar.radio("Model", list(chapters.MODELS))]
repo = st.sidebar.selectbox("Repo", repo_names, index=repo_names.index(start))
st.sidebar.caption("Model and Repo drive chapters 2–4 of the story.")
notes = st.sidebar.toggle("Speaker notes", value=False)
theme = getattr(st.context, "theme", None)
mode = "dark" if getattr(theme, "type", None) == "dark" else "light"

# Per session, never a module global: viewers share one server process, and callbacks run
# before this script does.
st.session_state["_ctx"] = ctx = chapters.Context(prs, features, results, hit_rates, scenario, repo,
                                                   notes, mode, wf=wf)
# One small page file per chapter (views/), each calling its function in chapters.py: file pages
# keep their place between runs in Streamlit's AppTest, so the tests can drive every chapter.
story = [st.Page("views/problem.py", title="1. The problem", url_path="problem", default=True),
         st.Page("views/watch.py", title="2. Watch it work", url_path="watch-it-work"),
         st.Page("views/why.py", title="3. Why it decides", url_path="why"),
         st.Page("views/test_yourself.py", title="4. Test yourself", url_path="test-yourself"),
         st.Page("views/transfer.py", title="5. Does it transfer?", url_path="does-it-transfer"),
         st.Page("views/limits.py", title="6. Honest limits", url_path="honest-limits")]
built = [st.Page("views/pipeline.py", title="Pipeline map", url_path="pipeline"),
         st.Page("views/funnel.py", title="Data funnel", url_path="data-funnel"),
         st.Page("views/checks.py", title="Validity checks", url_path="validity-checks")]
ctx.pages, ctx.wf_pages = story, built          # Back/Next walk the story; the stepper walks `built`
st.navigation({"The story": story, "How it was built": built}).run()

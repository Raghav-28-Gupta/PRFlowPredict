"""PRFlowPredict demo: an interactive walkthrough of the evaluated 2026 test period.

    pip install -r demo/requirements.txt
    streamlit run demo/app.py

A thin shell: page config, cached data, the sidebar and the six chapters in chapters.py. It
hard-codes no result number: every number shown is computed from committed files, and the
results chapter's numbers are tested against the README's."""
import importlib
import sys
from pathlib import Path

import streamlit as st

HERE = str(Path(__file__).parent)
if HERE not in sys.path:
    sys.path.insert(0, HERE)            # triage.py, charts.py and chapters.py sit next to this file
import triage  # noqa: E402
import charts  # noqa: E402
import chapters  # noqa: E402

# Streamlit Community Cloud re-runs this script when the repo updates but keeps imported modules
# in memory, so a new app.py could meet old helpers. Reload them every run, dependencies first.
triage = importlib.reload(triage)
charts = importlib.reload(charts)
chapters = importlib.reload(chapters)

st.set_page_config(page_title="PRFlowPredict demo", layout="wide")


@st.cache_data
def _data():
    prs = triage.load()
    return (prs, triage.load_features(), triage.results(),
            {sc: triage.game_hit_rate(prs, sc) for sc in triage.SCORES})


prs, features, results, hit_rates = _data()
repo_names = triage.repos(prs)
start = triage.default_repo(prs, triage.moment(triage.DEFAULT_DAY))

st.sidebar.markdown("### PRFlowPredict")
scenario = chapters.MODELS[st.sidebar.radio("Model", list(chapters.MODELS))]
repo = st.sidebar.selectbox("Repo", repo_names, index=repo_names.index(start))
notes = st.sidebar.toggle("Speaker notes", value=False)
theme = getattr(st.context, "theme", None)
mode = "dark" if getattr(theme, "type", None) == "dark" else "light"

# Per session, never a module global: viewers share one server process, and callbacks run
# before this script does.
st.session_state["_ctx"] = ctx = chapters.Context(prs, features, results, hit_rates, scenario, repo,
                                                   notes, mode)
# One small page file per chapter (views/), each calling its function in chapters.py: file pages
# keep their place between runs in Streamlit's AppTest, so the tests can drive every chapter.
pages = [st.Page("views/problem.py", title="1. The problem", url_path="problem", default=True),
         st.Page("views/watch.py", title="2. Watch it work", url_path="watch-it-work"),
         st.Page("views/why.py", title="3. Why it decides", url_path="why"),
         st.Page("views/test_yourself.py", title="4. Test yourself", url_path="test-yourself"),
         st.Page("views/transfer.py", title="5. Does it transfer?", url_path="does-it-transfer"),
         st.Page("views/limits.py", title="6. Honest limits", url_path="honest-limits")]
ctx.pages = pages
st.navigation(pages).run()

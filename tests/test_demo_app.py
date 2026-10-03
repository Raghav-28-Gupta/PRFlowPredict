"""The demo app, run headless with Streamlit's AppTest (Phase 7 spec, sections 6, 7 and 9)."""
import ast
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


def test_the_app_hard_codes_no_result_number():
    assert re.search(r"\b0\.\d{3}\b", APP.read_text(encoding="utf-8")) is None


@pytest.mark.parametrize("pattern,reason", wc.RETRACTED, ids=lambda v: str(v)[:30])
def test_the_app_avoids_retracted_phrasing(pattern, reason):
    hit = re.search(pattern, APP.read_text(encoding="utf-8"), flags=re.IGNORECASE)
    assert hit is None, f"demo/app.py says {hit.group(0)!r}, but {reason}"


def test_the_demo_imports_only_what_its_own_requirements_install():
    """Streamlit Cloud installs demo/requirements.txt, not the project's: no sklearn, lightgbm,
    shap, or project module may be imported by the deployed files."""
    allowed = {"__future__", "datetime", "pathlib", "sys", "streamlit", "pandas", "triage"}
    for name in ("app.py", "triage.py"):
        tree = ast.parse((APP.parent / name).read_text(encoding="utf-8"))
        mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert mods <= allowed, f"demo/{name} imports {sorted(mods - allowed)}"

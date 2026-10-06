"""The workflow view's charts, read through their Vega-Lite specs (workflow-view spec, section 6).

workflow_charts imports its siblings by bare name, as the deployed app does, so demo/ goes on the
path first."""
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

DEMO = Path(__file__).parents[1] / "demo"
sys.path.insert(0, str(DEMO))
import charts  # noqa: E402
import triage  # noqa: E402
import workflow as wf  # noqa: E402
import workflow_charts as wc  # noqa: E402


@pytest.fixture(scope="module")
def prs():
    return triage.load()


@pytest.fixture(scope="module")
def bundle(prs):
    return wf.bundle(prs)


def _params(chart) -> dict:
    return {p["name"]: p for p in chart.to_dict().get("params", [])}


def _layers(chart) -> list[dict]:
    return chart.to_dict()["layer"]


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_every_chart_builds_a_valid_spec_in_both_themes(bundle, prs, mode):
    pid = wf.default_pr(bundle["check"], "kdlbs/kandev")
    frame, s = wf.replay_at(prs, "kdlbs/kandev", pid, prior_rate=bundle["prior"])
    rows = prs[prs["repo"] == "kdlbs/kandev"]
    order = list(bundle["splits"]["repo"])
    for chart in (wc.pipeline_map(bundle["stages"], "collect", mode),
                  wc.funnel(bundle["repo_funnel"], "repos", mode, log=True),
                  wc.funnel(bundle["pr_funnel"], "pull requests", mode),
                  wc.replay_strip(frame, s["t"], mode),
                  wc.trailing_line(rows, s["t"], s["prior"], mode),
                  wc.split_bars(wf.design_rows(bundle["splits"]), order, mode),
                  wc.fold_dots(bundle["folds"], 0, mode),
                  wc.gate_tiles(bundle["gates"], mode)):
        assert chart.to_dict()                      # validates against the Vega-Lite schema


def test_the_pipeline_map_selects_a_stage_by_key_and_marks_the_chosen_one(bundle):
    chart = wc.pipeline_map(bundle["stages"], "evaluate")
    assert _params(chart)["stage"]["select"]["fields"] == ["key"]
    boxes = next(layer for layer in _layers(chart) if layer["mark"]["type"] == "rect")
    data = chart.data if isinstance(chart.data, pd.DataFrame) else chart.layer[0].data
    assert data.loc[data["chosen"], "key"].tolist() == ["evaluate"]
    assert boxes["encoding"]["fill"]["condition"]["value"] == charts.PALETTE["light"]["full"]


def test_funnels_select_a_step_and_start_log_bars_at_one(bundle):
    log = wc.funnel(bundle["repo_funnel"], "repos", log=True)
    assert _params(log)["step"]["select"]["fields"] == ["step"]
    bars = next(layer for layer in _layers(log) if layer["mark"]["type"] == "bar")
    assert bars["encoding"]["x"]["scale"]["type"] == "log" and bars["encoding"]["x2"]["datum"] == 1
    linear = next(layer for layer in _layers(wc.funnel(bundle["pr_funnel"], "pull requests"))
                  if layer["mark"]["type"] == "bar")
    assert linear["encoding"]["x2"]["datum"] == 0


def test_the_strip_shows_the_leak_only_under_the_naive_rule(bundle, prs):
    pid = wf.default_pr(bundle["check"], "kdlbs/kandev")
    for rule, has_leak in (("resolvable", False), ("naive", True)):
        frame, s = wf.replay_at(prs, "kdlbs/kandev", pid, rule, bundle["prior"])
        dots = next(layer for layer in _layers(wc.replay_strip(frame, s["t"])) if layer["mark"]["type"] == "circle")
        domain = dots["encoding"]["color"]["scale"]["domain"]
        assert (wf.LEAK in domain) is has_leak and (wf.UNKNOWN in domain) is not has_leak
        assert dots["encoding"]["y"]["sort"] == list(wc.LANES)


def test_the_trailing_line_dashes_the_prior(prs, bundle):
    rows = prs[prs["repo"] == "kdlbs/kandev"]
    t = rows["created_at"].iloc[100]
    chart = wc.trailing_line(rows, t, bundle["prior"])
    rules = [layer for layer in chart.layer if layer.to_dict()["mark"].get("strokeDash")]
    assert any(isinstance(r.data, pd.DataFrame) and r.data["y"].tolist() == [bundle["prior"]] for r in rules)


def test_split_bars_colour_the_three_segments_in_order(bundle):
    chart = wc.split_bars(wf.design_rows(bundle["splits"]), list(bundle["splits"]["repo"]))
    colour = chart.to_dict()["encoding"]["color"]["scale"]
    pal = charts.PALETTE["light"]
    assert colour["domain"] == list(wf.SEGMENTS) and colour["range"] == [pal["full"], pal["random"], pal["nlr"]]


def test_fold_dots_use_the_storys_series_names_and_emphasise_one_fold(bundle):
    chart = wc.fold_dots(bundle["folds"], 2)
    dots = next(layer for layer in chart.layer if layer.to_dict()["mark"]["type"] == "circle")
    assert set(dots.data["series"]) == {triage.SERIES_LABELS[k] for k in ("FULL", "NO_LABEL_REPLAY", "baseline")}
    assert dots.data.loc[dots.data["chosen"], "name"].unique().tolist() == ["fold 3"]


def test_gate_tiles_select_a_check_and_mark_the_missed_expectation(bundle):
    chart = wc.gate_tiles(bundle["gates"])
    assert _params(chart)["check"]["select"]["fields"] == ["phase", "id"]
    tiles = next(layer for layer in _layers(chart) if layer["mark"]["type"] == "rect")
    scale = tiles["encoding"]["color"]["scale"]
    assert dict(zip(scale["domain"], scale["range"]))[wf.MISSED] == charts.PALETTE["light"]["up"]
    data = chart.data if isinstance(chart.data, pd.DataFrame) else chart.layer[0].data
    assert "value" not in data.columns and len(data) == len(bundle["gates"])


def test_the_charts_use_only_the_validated_palette():
    """Every colour comes from charts.PALETTE, whose pairs were validated in both themes; any
    other colour literal would have to clear 3:1 on both chart surfaces."""
    known = {c for theme in charts.PALETTE.values() for c in theme.values()}
    literals = set(re.findall(r"#[0-9a-fA-F]{6}", (DEMO / "workflow_charts.py").read_text(encoding="utf-8")))
    assert literals <= known, literals - known

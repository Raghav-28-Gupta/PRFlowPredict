"""The workflow view's charts: pure functions from workflow.py's frames to Altair charts.

Nothing here touches Streamlit, so tests read each chart's spec with .to_dict(). Colours are the
story's palette roles (charts.PALETTE), so both themes stay validated: blue for what the model
counts, trains on or passes; light blue for what it is tested on; grey for context; red for the
leak the 7-day rule prevents and for the one pilot check that missed its expectation. Orange
keeps its story meaning (stalled) and is not used here."""
from __future__ import annotations

import altair as alt
import pandas as pd

import triage
import workflow
from charts import PALETTE

STATUS_ORDER = (workflow.COUNTED, workflow.UNKNOWN, workflow.LEAK, workflow.OLD)
LANES = (workflow.STALLED_LANE, workflow.FINE_LANE, workflow.UNKNOWN_LANE)
TILE_TEXT = {"pass": "pass", "measured": "measured", "resolved": "resolved", workflow.MISSED: "missed",
             "fail": "fail"}
SERIES = {"model": triage.SERIES_LABELS["FULL"], "without_history": triage.SERIES_LABELS["NO_LABEL_REPLAY"],
          "baseline": triage.SERIES_LABELS["baseline"]}


def pipeline_map(stages: pd.DataFrame, selected: str, mode: str = "light") -> alt.LayerChart:
    """The six stages left to right, the chosen one emphasised; clicking a box selects its key
    ('stage')."""
    pal = PALETTE[mode]
    n = len(stages)
    data = stages.assign(x0=stages["order"] + 0.05, x1=stages["order"] + 0.95, mid=stages["order"] + 0.5,
                         chosen=stages["key"] == selected)
    pick = alt.selection_point(name="stage", fields=["key"], on="click", empty=False)
    x = alt.X("x0:Q", scale=alt.Scale(domain=[0, n]), axis=None)
    y = alt.Y("y0:Q", scale=alt.Scale(domain=[0, 1]), axis=None)
    boxes = alt.Chart(data.assign(y0=0.04, y1=0.96)).mark_rect(cornerRadius=8, strokeWidth=2).encode(
        x=x, x2="x1:Q", y=y, y2="y1:Q",
        fill=alt.condition(alt.datum.chosen, alt.value(pal["full"]), alt.value(pal["muted"])),
        fillOpacity=alt.condition(alt.datum.chosen, alt.value(0.22), alt.value(0.06)),
        stroke=alt.condition(alt.datum.chosen, alt.value(pal["full"]), alt.value(pal["muted"])),
        tooltip=[alt.Tooltip("title:N", title="stage"), alt.Tooltip("phases:N"), alt.Tooltip("badge:N"),
                 alt.Tooltip("gate:N")],
    ).add_params(pick)

    def text(field: str, at: float, size: int, bold: bool = False) -> alt.Chart:
        return alt.Chart(data.assign(y0=at)).mark_text(
            fontSize=size, fontWeight="bold" if bold else "normal", color=pal["ink"], limit=150).encode(
            x=alt.X("mid:Q", scale=alt.Scale(domain=[0, n]), axis=None), y=y, text=f"{field}:N")

    gaps = pd.DataFrame({"x0": [i + 0.95 for i in range(n - 1)], "x1": [i + 1.05 for i in range(n - 1)],
                         "y0": [0.5] * (n - 1)})
    arrows = alt.Chart(gaps).mark_rule(color=pal["muted"], strokeWidth=2).encode(x=x, x2="x1:Q", y=y)
    heads = alt.Chart(gaps).mark_point(shape="triangle-right", filled=True, size=60, color=pal["muted"]).encode(
        x=alt.X("x1:Q", scale=alt.Scale(domain=[0, n]), axis=None), y=y)
    return (boxes + text("title", 0.76, 14, True) + text("badge", 0.5, 12) + text("gate", 0.24, 11)
            + arrows + heads).properties(height=130)


def funnel(steps: pd.DataFrame, unit: str, mode: str = "light", log: bool = False) -> alt.LayerChart:
    """One narrowing step per bar, labelled with its count; clicking a bar selects it ('step')."""
    pal = PALETTE[mode]
    order = list(steps["step"])
    pick = alt.selection_point(name="step", fields=["step"], on="click", empty=False)
    base = alt.Chart(steps.assign(label=[f"{c:,}" for c in steps["count"]])).encode(
        y=alt.Y("step:N", sort=order, title=None, axis=alt.Axis(labelLimit=240)),
        x=alt.X("count:Q", title=f"{unit} (log scale)" if log else unit,
                scale=alt.Scale(type="log") if log else alt.Scale(), axis=alt.Axis(format="~s")),
        tooltip=[alt.Tooltip("step:N"), alt.Tooltip("count:Q", format=","), alt.Tooltip("why:N"),
                 alt.Tooltip("source:N")])
    bars = base.mark_bar(cornerRadiusEnd=4, height=22, color=pal["full"]).encode(
        x2=alt.datum(1 if log else 0),          # a log scale has no zero: bars start at 1
        strokeWidth=alt.condition(pick, alt.value(3), alt.value(0)), stroke=alt.value(pal["ring"])).add_params(pick)
    labels = base.mark_text(align="left", dx=5, color=pal["ink"]).encode(text="label:N")
    return (bars + labels).properties(height=alt.Step(34))


def replay_strip(frame: pd.DataFrame, t: pd.Timestamp, mode: str = "light",
                 domain: tuple | None = None) -> alt.LayerChart:
    """The repo's earlier PRs on a time axis, one lane per outcome as known at t, coloured by
    whether the trailing rate counts them. Rules mark t - 90 days, t - 7 days and t."""
    pal = PALETTE[mode]
    colours = dict(zip(STATUS_ORDER, (pal["full"], pal["muted"], pal["up"], pal["random"])))
    present = [s for s in STATUS_ORDER if s in set(frame["status"])]
    scale = alt.Scale(domain=list(domain)) if domain else alt.Scale()
    dots = alt.Chart(frame).transform_calculate(jitter="random()").mark_circle(size=34, opacity=0.85).encode(
        x=alt.X("created_at:T", title="opened", scale=scale),
        y=alt.Y("lane:N", sort=list(LANES), title=None, axis=alt.Axis(labelLimit=200)),
        yOffset=alt.YOffset("jitter:Q", scale=alt.Scale(domain=[-0.5, 1.5])),
        color=alt.Color("status:N", scale=alt.Scale(domain=present, range=[colours[s] for s in present]),
                        legend=alt.Legend(title=None, orient="top", labelLimit=320)),
        tooltip=[alt.Tooltip("number:Q", title="PR #"), alt.Tooltip("created_at:T", title="opened", format="%d %b %Y %H:%M"),
                 alt.Tooltip("first_review_at:T", title="first review", format="%d %b %Y %H:%M"),
                 alt.Tooltip("status:N")])
    marks = pd.DataFrame({"at": [t - pd.Timedelta(days=workflow.TRAILING_DAYS),
                                 t - pd.Timedelta(hours=workflow.THRESHOLD_H), t],
                          "label": ["90 days before", "7 days before", "t: this PR opens"]})
    rules = alt.Chart(marks).mark_rule(color=pal["muted"], strokeDash=[4, 4]).encode(x=alt.X("at:T", scale=scale))

    def label(rows: pd.DataFrame, y: int) -> alt.Chart:
        return alt.Chart(rows).mark_text(color=pal["muted"], align="right", dx=-4, y=y).encode(
            x=alt.X("at:T", scale=scale), text="label:N")

    # t's label sits a line lower: "7 days before" ends only a week to its left
    return (dots + rules + label(marks.iloc[:2], 6) + label(marks.iloc[2:], 20)).properties(height=alt.Step(70))


def trailing_line(rows: pd.DataFrame, t: pd.Timestamp, prior_rate: float, mode: str = "light",
                  domain: tuple | None = None) -> alt.LayerChart:
    """The stored trailing slow rate of each of the repo's PRs at its opening, with t marked and
    the prior it shrinks toward dashed."""
    pal = PALETTE[mode]
    scale = alt.Scale(domain=list(domain)) if domain else alt.Scale()
    y = alt.Y("x__trailing_90d_slow_rate:Q", title="trailing 90-day slow rate", scale=alt.Scale(domain=[0, 1]))
    line = alt.Chart(rows).mark_line(interpolate="step-after", color=pal["full"]).encode(
        x=alt.X("created_at:T", title="opened", scale=scale), y=y)
    now = rows[rows["created_at"] == t]
    point = alt.Chart(now).mark_circle(size=120, color=pal["full"], stroke=pal["ring"], strokeWidth=1.5).encode(
        x=alt.X("created_at:T", scale=scale), y=y,
        tooltip=[alt.Tooltip("x__trailing_90d_slow_rate:Q", title="stored rate at t", format=".3f")])
    prior = pd.DataFrame({"y": [prior_rate], "label": ["prior"]})
    prior_rule = alt.Chart(prior).mark_rule(color=pal["muted"], strokeDash=[6, 4]).encode(y="y:Q")
    prior_label = alt.Chart(prior).mark_text(color=pal["muted"], align="left", x=4, dy=-6).encode(y="y:Q", text="label:N")
    cursor = alt.Chart(pd.DataFrame({"at": [t]})).mark_rule(color=pal["muted"]).encode(x=alt.X("at:T", scale=scale))
    return (line + prior_rule + prior_label + cursor + point).properties(height=200)


def split_bars(long: pd.DataFrame, order: list[str], mode: str = "light") -> alt.Chart:
    """Each repo's rows, stacked: trained on, dropped by the cap, tested on."""
    pal = PALETTE[mode]
    return alt.Chart(long).mark_bar(height=10).encode(
        y=alt.Y("repo:N", sort=order, title=None, axis=alt.Axis(labelLimit=260, labelFontSize=10)),
        x=alt.X("sum(rows):Q", title="pull requests", stack="zero"),
        color=alt.Color("segment:N", scale=alt.Scale(domain=list(workflow.SEGMENTS),
                                                     range=[pal["full"], pal["random"], pal["nlr"]]),
                        legend=alt.Legend(title=None, orient="top")),
        order=alt.Order("order:Q"),
        tooltip=[alt.Tooltip("repo:N"), alt.Tooltip("segment:N"), alt.Tooltip("rows:Q", format=",")],
    ).transform_calculate(order=f"indexof({list(workflow.SEGMENTS)}, datum.segment)").properties(height=alt.Step(13))


def fold_dots(folds: pd.DataFrame, fold: int, mode: str = "light") -> alt.LayerChart:
    """Each held-out fold's AUC-PR for the model, the model without the slow-rate history and the
    baseline, with the fold's base rate (the floor AUC-PR starts from) as a tick; the chosen fold
    stands out."""
    pal = PALETTE[mode]
    data = folds.assign(name=[f"fold {f + 1}" for f in folds["fold"]], chosen=folds["fold"] == fold)
    long = data.melt(id_vars=["name", "chosen", "base_rate"], value_vars=list(SERIES),
                     var_name="series", value_name="auc_pr")
    long["series"] = long["series"].map(SERIES)
    order = list(data["name"])
    y = alt.Y("name:N", sort=order, title=None)
    x = alt.X("auc_pr:Q", title="AUC-PR", scale=alt.Scale(domain=[0, 1]))
    floor = alt.Chart(data).mark_tick(color=pal["ink"], thickness=2, size=18).encode(
        x=alt.X("base_rate:Q", scale=alt.Scale(domain=[0, 1])), y=y,
        opacity=alt.condition(alt.datum.chosen, alt.value(1), alt.value(0.3)),
        tooltip=[alt.Tooltip("name:N", title="fold"), alt.Tooltip("base_rate:Q", title="base rate (the floor)", format=".3f")])
    dots = alt.Chart(long).mark_circle(size=110, stroke=pal["ring"], strokeWidth=1).encode(
        x=x, y=y,
        color=alt.Color("series:N", scale=alt.Scale(domain=list(SERIES.values()),
                                                    range=[pal["full"], pal["nlr"], pal["baseline"]]),
                        legend=alt.Legend(title=None, orient="top", labelLimit=320)),
        opacity=alt.condition(alt.datum.chosen, alt.value(1), alt.value(0.25)),
        tooltip=[alt.Tooltip("name:N", title="fold"), alt.Tooltip("series:N"), alt.Tooltip("auc_pr:Q", format=".3f")])
    return (floor + dots).properties(height=alt.Step(36))


def gate_tiles(gates: pd.DataFrame, mode: str = "light") -> alt.LayerChart:
    """One tile per recorded check, a row per phase; clicking a tile selects it ('check')."""
    pal = PALETTE[mode]
    colours = {"pass": pal["full"], "measured": pal["muted"], "resolved": pal["muted"],
               workflow.MISSED: pal["up"], "fail": pal["up"]}
    data = gates.drop(columns="value").assign(
        phase_name=gates["phase"].map(workflow.PHASE_NAMES), tile=gates["status"].map(TILE_TEXT),
        value_text=gates["value"].map(workflow.gate_value_text))
    present = [s for s in colours if s in set(data["status"])]
    pick = alt.selection_point(name="check", fields=["phase", "id"], on="click", empty=False)
    base = alt.Chart(data).encode(
        x=alt.X("id:O", title="check", axis=alt.Axis(labelAngle=0), scale=alt.Scale(paddingInner=0.08)),
        y=alt.Y("phase_name:N", sort=list(workflow.PHASE_NAMES.values()), title=None,
                scale=alt.Scale(paddingInner=0.12)))
    tiles = base.mark_rect(cornerRadius=4, stroke=pal["ring"]).encode(
        color=alt.Color("status:N", scale=alt.Scale(domain=present, range=[colours[s] for s in present]),
                        legend=alt.Legend(title=None, orient="top", labelLimit=320)),
        strokeWidth=alt.condition(pick, alt.value(3), alt.value(0)),
        tooltip=[alt.Tooltip("phase_name:N", title="phase"), alt.Tooltip("id:O", title="check"),
                 alt.Tooltip("check:N", title="what it checks"), alt.Tooltip("status:N"),
                 alt.Tooltip("value_text:N", title="recorded")],
    ).add_params(pick)
    text = base.mark_text(color=pal["dot"], fontSize=11).encode(text="tile:N")
    return (tiles + text).properties(height=alt.Step(40))

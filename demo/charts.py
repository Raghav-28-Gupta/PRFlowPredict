"""The demo's charts: pure functions from triage.py's frames to Altair charts.

Nothing here touches Streamlit, so tests read each chart's spec with .to_dict(). Colours come
from the validated reference palette, by job: stalled = orange and reviewed in time = blue
(categorical slots 2 and 1), pushes towards stalling = red and away = blue (the diverging
pair), and the results chart keeps Figure 1's roles (blue, light blue, gray baseline). Light
and dark steps are separate because Streamlit does not recolour explicit colours when a viewer
switches theme; every pair was checked with the palette validator in both modes."""
from __future__ import annotations

import altair as alt
import pandas as pd

alt.data_transformers.disable_max_rows()        # the largest repo has 5,178 replayed PRs

PALETTE = {
    "light": {"fine": "#2a78d6", "stalled": "#eb6834", "up": "#e34948", "down": "#2a78d6",
              "full": "#2a78d6", "nlr": "#86b6ef", "baseline": "#898781", "random": "#c3c2b7",
              "ink": "#0b0b0b", "muted": "#52514e"},
    "dark": {"fine": "#3987e5", "stalled": "#d95926", "up": "#e66767", "down": "#3987e5",
             "full": "#3987e5", "nlr": "#86b6ef", "baseline": "#898781", "random": "#52514e",
             "ink": "#ffffff", "muted": "#c3c2b7"},
}
FINE, STALLED = "first review within 7 days", "stalled: no first review within 7 days"
ON_BAR = {"model, all features": "#ffffff"}       # text inside bars: white on the dark blue, else near-black
BUCKET_STEP = 120                                 # px per histogram bar, so the 7-day rule sits between bars
BUCKET_LINES = {"within an hour": "within|an hour", "1 hour to 1 day": "1 hour|to 1 day",
                "1 to 7 days": "1 to|7 days", "after more than 7 days": "after more|than 7 days",
                "closed without a review": "closed without|a review",
                "never reviewed, still open": "never reviewed,|still open"}


def _status(stalled) -> list[str]:
    return [STALLED if s else FINE for s in stalled]


def _status_color(pal: dict) -> alt.Color:
    return alt.Color("status:N", scale=alt.Scale(domain=[FINE, STALLED], range=[pal["fine"], pal["stalled"]]),
                     legend=alt.Legend(title=None, orient="top", labelLimit=320))


def wait_histogram(buckets: pd.DataFrame, mode: str = "light") -> alt.LayerChart:
    """Share of PRs per wait bucket, blue before the 7-day line and orange after it."""
    pal = PALETTE[mode]
    data = buckets.assign(status=_status(buckets["stalled"]), axis_label=buckets["bucket"].map(BUCKET_LINES))
    order = list(data["axis_label"])
    base = alt.Chart(data).encode(
        x=alt.X("axis_label:N", sort=order, title=None,
                axis=alt.Axis(labelAngle=0, labelLimit=BUCKET_STEP, labelExpr="split(datum.label, '|')")),
        y=alt.Y("share:Q", title="share of replayed PRs", axis=alt.Axis(format="%", grid=True)),
        tooltip=[alt.Tooltip("bucket:N", title="first review"), alt.Tooltip("count:Q", format=","),
                 alt.Tooltip("share:Q", format=".1%")])
    bars = base.mark_bar(cornerRadiusEnd=4).encode(color=_status_color(pal))
    labels = base.mark_text(dy=-8, color=pal["ink"]).encode(text=alt.Text("share:Q", format=".0%"))
    rule_at = pd.DataFrame({"axis_label": [order[3]], "label": ["7 days"]})
    rule = alt.Chart(rule_at).mark_rule(color=pal["muted"], strokeDash=[4, 4], xOffset=-BUCKET_STEP / 2).encode(
        x=alt.X("axis_label:N", sort=order))
    rule_label = alt.Chart(rule_at).mark_text(color=pal["muted"], xOffset=-BUCKET_STEP / 2, dx=4, align="left",
                                              y=6).encode(x=alt.X("axis_label:N", sort=order), text="label:N")
    return (bars + labels + rule + rule_label).properties(width=alt.Step(BUCKET_STEP), height=300)


def timeline(frame: pd.DataFrame, at: pd.Timestamp, mode: str = "light") -> alt.LayerChart:
    """Each PR by opening date and risk at open, coloured by whether it stalled. PRs waiting at
    `at` are solid, the rest faded; a rule marks `at`. Clicking selects the nearest PR ('pick')."""
    pal = PALETTE[mode]
    pick = alt.selection_point(name="pick", fields=["pr_id"], on="click", nearest=True, empty=False)
    points = alt.Chart(frame.assign(status=_status(frame["stalled"]))).mark_circle(stroke=pal["ink"]).encode(
        x=alt.X("created_at:T", title="opened"),
        y=alt.Y("risk:Q", title="risk score at open", scale=alt.Scale(domain=[0, 1])),
        color=_status_color(pal),
        opacity=alt.condition(alt.datum.waiting, alt.value(0.95), alt.value(0.22)),
        size=alt.condition(pick, alt.value(220), alt.value(70)),
        strokeWidth=alt.condition(pick, alt.value(2), alt.value(0)),
        tooltip=[alt.Tooltip("number:Q", title="PR #"), alt.Tooltip("title:N"),
                 alt.Tooltip("risk:Q", format=".2f"), alt.Tooltip("outcome:N", title="what happened"),
                 alt.Tooltip("waiting:N", title="waiting on the chosen day")],
    ).add_params(pick)
    marker = pd.DataFrame({"at": [at], "label": ["chosen day"]})
    rule = alt.Chart(marker).mark_rule(color=pal["muted"], strokeWidth=1.5).encode(x="at:T")
    label = alt.Chart(marker).mark_text(color=pal["muted"], align="left", dx=4, y=8).encode(x="at:T", text="label:N")
    return (points + rule + label).properties(height=380)


def why_bars(rows: pd.DataFrame, mode: str = "light") -> alt.LayerChart:
    """One PR's biggest pushes on the model's log-odds: red raises the risk, blue lowers it."""
    pal = PALETTE[mode]
    data = rows.assign(direction=["raises risk" if p > 0 else "lowers risk" for p in rows["push"]],
                       shown=[f"{lab} = {val}" if val else lab for lab, val in zip(rows["label"], rows["value"])])
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=4, height=18).encode(
        y=alt.Y("shown:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
        x=alt.X("push:Q", title="push on the model's score (log-odds)"),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["raises risk", "lowers risk"],
                                                       range=[pal["up"], pal["down"]]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("label:N", title="feature"), alt.Tooltip("value:N"),
                 alt.Tooltip("push:Q", format="+.3f")])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=pal["muted"]).encode(x="x:Q")
    return (bars + zero).properties(height=alt.Step(26))


def importance_bars(frame: pd.DataFrame, top: int = 10, mode: str = "light") -> alt.LayerChart:
    """What the model leans on overall: Phase 6's share of mean |SHAP|, top features."""
    pal = PALETTE[mode]
    data = frame.head(top)
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
        x=alt.X("share:Q", title="share of the model's attribution", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("label:N", title="feature"), alt.Tooltip("share:Q", format=".1%")])
    bars = base.mark_bar(cornerRadiusEnd=4, height=16, color=pal["full"])
    labels = base.mark_text(align="left", dx=4, color=pal["ink"]).encode(text=alt.Text("share:Q", format=".0%"))
    return (bars + labels).properties(height=alt.Step(24))


def results_chart(res: dict, mode: str = "light") -> alt.LayerChart:
    """Mean AUC-PR by scenario and feature set, from zero; Scenario B's five folds as dots."""
    pal = PALETTE[mode]
    order = list(dict.fromkeys(res["means"]["series"]))
    color = alt.Color("series:N", sort=order, scale=alt.Scale(domain=order,
                                                              range=[pal["full"], pal["nlr"], pal["baseline"]]),
                      legend=alt.Legend(title=None, orient="top", labelLimit=320))
    x = alt.X("scenario:N", title=None, sort=list(dict.fromkeys(res["means"]["scenario"])), axis=alt.Axis(labelAngle=0))
    offset = alt.XOffset("series:N", sort=order)
    y = alt.Y("auc_pr:Q", title="AUC-PR", scale=alt.Scale(domain=[0, 1]))
    bars = alt.Chart(res["means"]).mark_bar(cornerRadiusEnd=4).encode(
        x=x, xOffset=offset, y=y, color=color,
        tooltip=[alt.Tooltip("scenario:N"), alt.Tooltip("series:N"), alt.Tooltip("auc_pr:Q", format=".3f", title="mean AUC-PR"),
                 alt.Tooltip("folds:Q")])
    # values at the bar's base, where no fold dot reaches, so a label never collides with a dot
    on_bar = res["means"].assign(text_color=[ON_BAR.get(s, "#0b0b0b") for s in res["means"]["series"]])
    labels = alt.Chart(on_bar).mark_text(baseline="bottom", dy=-6).encode(
        x=x, xOffset=offset, y=alt.datum(0), text=alt.Text("auc_pr:Q", format=".3f"),
        color=alt.Color("text_color:N", scale=None))
    folds = res["folds"][res["folds"]["folds"] > 1]
    dots = alt.Chart(folds).mark_circle(size=36, color=pal["ink"], opacity=0.75).encode(
        x=x, xOffset=offset, y=y,
        tooltip=[alt.Tooltip("series:N"), alt.Tooltip("fold:Q", title="held-out fold"),
                 alt.Tooltip("auc_pr:Q", format=".3f")])
    return (bars + labels + dots).properties(height=340)


def p10_chart(res: dict, mode: str = "light") -> alt.LayerChart:
    """Scenario A precision on each repo's top 10: model (with its interval), baseline, random pick."""
    pal = PALETTE[mode]
    data = res["p10"]
    order = list(data["series"])
    color = alt.Color("series:N", sort=order, scale=alt.Scale(domain=order,
                                                              range=[pal["full"], pal["baseline"], pal["random"]]),
                      legend=None)
    base = alt.Chart(data).encode(
        y=alt.Y("series:N", sort=order, title=None, axis=alt.Axis(labelLimit=200)),
        x=alt.X("p10:Q", title="precision on each repo's 10 riskiest PRs", scale=alt.Scale(domain=[0, 1])))
    bars = base.mark_bar(cornerRadiusEnd=4, height=22).encode(
        color=color, tooltip=[alt.Tooltip("series:N"), alt.Tooltip("p10:Q", format=".3f", title="precision"),
                              alt.Tooltip("lo:Q", format=".3f", title="interval low"),
                              alt.Tooltip("hi:Q", format=".3f", title="interval high")])
    whisker = alt.Chart(data.dropna(subset=["lo"])).mark_rule(color=pal["ink"], strokeWidth=2).encode(
        y=alt.Y("series:N", sort=order), x="lo:Q", x2="hi:Q")
    labels = base.mark_text(align="left", dx=6, color=pal["ink"]).encode(
        x=alt.X("plotted:Q"), text="label:N").transform_calculate(
        plotted="isValid(datum.hi) ? datum.hi : datum.p10")
    return (bars + whisker + labels).properties(height=alt.Step(40))

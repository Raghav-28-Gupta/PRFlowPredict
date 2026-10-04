"""The demo's charts, read through their Vega-Lite specs (demo-story spec, section 6)."""
import pandas as pd
import pytest

from demo import charts, triage


@pytest.fixture(scope="module")
def prs():
    return triage.load()


def _layer(chart, mark):
    """The first layer drawing `mark` (as a dict), with its data frame. Altair moves data that
    layers share up to the layered chart, so fall back to the chart's own data."""
    for layer in chart.layer:
        spec = layer.to_dict()
        kind = spec["mark"]["type"] if isinstance(spec["mark"], dict) else spec["mark"]
        if kind == mark:
            data = layer.data if isinstance(layer.data, pd.DataFrame) else chart.data
            return spec, data
    raise LookupError(mark)


def _params(chart):
    """Selection parameters, which Altair lifts to the top of a layered chart."""
    return chart.to_dict().get("params", [])


def _luminance(colour: str) -> float:
    def linear(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b)


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_text_and_rule_colours_read_on_both_light_and_dark_surfaces():
    """st.context.theme can be stale on a first load, so a chart drawn for the other theme must
    still be legible: text, rules and rings use one colour in both modes, clearing 3:1 on both
    chart surfaces."""
    for role in ("ink", "muted"):
        assert charts.PALETTE["light"][role] == charts.PALETTE["dark"][role]
        for surface in ("#fcfcfb", "#1a1a19"):
            assert _contrast(charts.PALETTE["light"][role], surface) >= 3, (role, surface)


def test_both_themes_define_every_colour_role():
    assert set(charts.PALETTE["light"]) == set(charts.PALETTE["dark"])


def test_wait_histogram_colours_by_stalled_and_puts_the_7_day_rule_between_bars(prs):
    buckets = triage.wait_buckets(prs)
    chart = charts.wait_histogram(buckets)
    bars, data = _layer(chart, "bar")
    assert bars["encoding"]["color"]["scale"]["domain"] == [charts.FINE, charts.STALLED]
    assert data["count"].tolist() == buckets["count"].tolist()
    rule, rule_data = _layer(chart, "rule")
    assert rule["mark"]["xOffset"] == -charts.BUCKET_STEP / 2
    assert rule_data["axis_label"].tolist() == [charts.BUCKET_LINES["after more than 7 days"]]


def test_timeline_plots_every_pr_with_a_nearest_click_selection_and_the_day_rule(prs):
    at = triage.moment(triage.DEFAULT_DAY)
    frame = triage.timeline(prs, "radixark/miles", "A", at)
    chart = charts.timeline(frame, at)
    points, data = _layer(chart, "circle")
    assert len(data) == len(frame) and data["risk"].tolist() == frame["risk"].tolist()
    pick = [p for p in _params(chart) if p["name"] == "pick"][0]
    assert pick["select"]["fields"] == ["pr_id"] and pick["select"]["nearest"] is True
    assert points["encoding"]["y"]["scale"]["domain"] == [0, 1]
    _, rule_data = _layer(chart, "rule")
    assert rule_data["at"].tolist() == [at]


def test_why_bars_keep_the_given_order_and_colour_by_direction(prs):
    rows, _ = triage.why(triage.find_pr(prs, "radixark/miles", 754), "A", triage.load_features())
    bars, data = _layer(charts.why_bars(rows), "bar")
    assert bars["encoding"]["y"]["sort"] is None
    assert bars["encoding"]["color"]["scale"]["domain"] == ["raises risk", "lowers risk"]
    assert data["push"].tolist() == rows["push"].tolist()
    assert data["direction"].tolist() == ["raises risk" if p > 0 else "lowers risk" for p in rows["push"]]


def test_importance_bars_show_the_top_features():
    imp = triage.importance("B", triage.load_features())
    _, data = _layer(charts.importance_bars(imp, top=7), "bar")
    assert data["feature"].tolist() == imp["feature"].head(7).tolist()


def test_results_chart_starts_at_zero_and_dots_only_the_five_fold_scenario():
    res = triage.results()
    chart = charts.results_chart(res)
    bars, data = _layer(chart, "bar")
    assert bars["encoding"]["y"]["scale"]["domain"] == [0, 1]
    assert len(data) == 6
    _, dots = _layer(chart, "circle")
    assert len(dots) == 15 and set(dots["scenario"]) == {triage.SCENARIO_LABELS["B"]}


def test_p10_chart_labels_each_bar_with_its_formatted_value():
    res = triage.results()
    chart = charts.p10_chart(res, mode="dark")
    _, labels = _layer(chart, "text")
    assert labels["label"].tolist() == [res["text"]["a_p10"], res["text"]["a_baseline_p10"], res["text"]["a_random_p10"]]
    _, whisker = _layer(chart, "rule")
    assert whisker["series"].tolist() == ["model"]

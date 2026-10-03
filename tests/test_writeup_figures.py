"""The two headline figures show exactly the committed numbers, and render deterministically."""
import numpy as np
import pandas as pd
import pytest

import writeup_figures as wf


def _runs(runs, scenario, featureset):
    return sorted((r for r in runs if r["scenario"] == scenario and r["featureset"] == featureset),
                  key=lambda r: r["fold"])


def test_transfer_data_is_phase4s_numbers():
    runs = wf.load_runs()
    d = wf.transfer_data(runs)
    assert [len(d["A"][k]) for k in ("FULL", "NO_LABEL_REPLAY", "baseline")] == [1, 1, 1]
    assert [len(d["B"][k]) for k in ("FULL", "NO_LABEL_REPLAY", "baseline")] == [5, 5, 5]
    # each value is taken verbatim from the runs file, fold by fold
    assert d["B"]["NO_LABEL_REPLAY"] == [r["auc_pr"] for r in _runs(runs, "B", "NO_LABEL_REPLAY")]
    assert d["B"]["FULL"] == [r["auc_pr"] for r in _runs(runs, "B", "FULL")]
    # and the bars are the published Phase 4 numbers
    assert round(float(np.mean(d["A"]["FULL"])), 3) == 0.906
    assert round(float(np.mean(d["B"]["NO_LABEL_REPLAY"])), 3) == 0.763
    assert round(float(np.mean(d["B"]["baseline"])), 3) == 0.821


def test_the_baseline_is_the_same_whichever_model_supplies_it():
    """Figure 1 takes the baseline from FULL's runs; that is only valid because the baseline
    is scored on the same rows for every feature set."""
    runs = wf.load_runs()
    for sc in ("A", "B"):
        full = [r["baseline_auc_pr"] for r in _runs(runs, sc, "FULL")]
        nlr = [r["baseline_auc_pr"] for r in _runs(runs, sc, "NO_LABEL_REPLAY")]
        assert full == nlr


@pytest.fixture
def headline(monkeypatch, tmp_path):
    """Figure 1 as a matplotlib Figure: keep it open long enough to inspect, then close it."""
    captured = []
    monkeypatch.setattr(wf.plt, "close", captured.append)
    wf.headline_transfer(wf.load_runs(), tmp_path / "headline.png")
    monkeypatch.undo()
    yield captured[0]
    wf.plt.close(captured[0])


def test_headline_bars_start_at_zero(headline):
    """Bar length encodes the value, so a bar axis that starts above zero exaggerates the gaps."""
    assert headline.axes[0].get_ylim()[0] == 0


def test_headline_names_what_the_ablation_removes(headline):
    """NO_LABEL_REPLAY drops only the four slow-rate features, not 'the repo's own history'."""
    ax = headline.axes[0]
    texts = [ax.get_title()] + [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("slow-rate history" in t for t in texts)
    assert not any("own history" in t for t in texts)


def test_scatter_data_is_the_transfer_table():
    t = pd.read_csv(wf.TRANSFER)
    d = wf.scatter_data(t)
    for sc in ("A", "B"):
        assert np.array_equal(d[sc]["c"], t[f"c_{sc}"].to_numpy())
        assert np.array_equal(d[sc]["y"], t[f"y_{sc}"].to_numpy())
    assert round(d["A"]["rho"], 3) == 0.669 and round(d["B"]["rho"], 3) == 0.318


def test_figures_render_deterministically_into_the_given_directory(tmp_path):
    committed = {p.name: p.stat().st_mtime_ns for p in wf.FIG.glob("*.png")}
    for sub in ("one", "two"):
        assert wf.main(tmp_path / sub) == 0
    for name in ("headline_transfer.png", "fingerprint_scatter.png"):
        a, b = (tmp_path / "one" / name).read_bytes(), (tmp_path / "two" / name).read_bytes()
        assert len(a) > 10_000
        assert a == b, f"{name} is not byte-identical across two renders"
    # rendering into a given directory must never touch the committed figures
    assert {p.name: p.stat().st_mtime_ns for p in wf.FIG.glob("*.png")} == committed

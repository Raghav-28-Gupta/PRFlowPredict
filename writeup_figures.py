"""Phase 8's two headline figures, rendered from COMMITTED artifacts only.

Nothing is retrained, and nothing under the gitignored data/models, data/predictions or
data/features is read, so anyone who clones the repo can regenerate both figures:

    python writeup_figures.py

Each plotting function returns the exact values it drew, so tests can check the figure
against its source artifact rather than trusting the picture."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent
FIG = ROOT / "figures"
RUNS = ROOT / "data" / "phase4_runs.json"
TRANSFER = ROOT / "data" / "phase6b_transfer.csv"
PNG_META = {"Software": None}          # no version string in the PNG, so output is byte-stable
SETS = ("FULL", "NO_LABEL_REPLAY")
LABELS = {"FULL": "model, all features", "NO_LABEL_REPLAY": "model, without the repo's own history",
          "baseline": "trailing-rate baseline"}
COLOURS = {"FULL": "#2b6cb0", "NO_LABEL_REPLAY": "#90cdf4", "baseline": "#a0aec0"}


def load_runs(path: Path | None = None) -> list[dict]:
    # resolved at call time, not definition time, so tests can point it elsewhere
    return json.loads((path or RUNS).read_text(encoding="utf-8"))["runs"]


def transfer_data(runs: list[dict]) -> dict[str, dict[str, list[float]]]:
    """What Figure 1 plots. Per scenario: AUC-PR for FULL, NO_LABEL_REPLAY and the baseline,
    each a list over folds (one for Scenario A, five for Scenario B). A bar is the mean.
    The baseline is scored on the same rows as the model, so FULL's runs supply it."""
    out = {}
    for sc in ("A", "B"):
        by = {f: sorted((r for r in runs if r["scenario"] == sc and r["featureset"] == f),
                        key=lambda r: r["fold"]) for f in SETS}
        if not all(by.values()):
            raise LookupError(f"missing Scenario {sc} runs for {[f for f, v in by.items() if not v]}")
        out[sc] = {"FULL": [r["auc_pr"] for r in by["FULL"]],
                   "NO_LABEL_REPLAY": [r["auc_pr"] for r in by["NO_LABEL_REPLAY"]],
                   "baseline": [r["baseline_auc_pr"] for r in by["FULL"]]}
    return out


def headline_transfer(runs: list[dict], out: Path) -> dict:
    """Figure 1. Phase 4 recorded no AUC-PR intervals, so none are drawn: Scenario A is one
    value per bar, and Scenario B's five per-fold values are overlaid as dots."""
    data = transfer_data(runs)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    keys, width = ("FULL", "NO_LABEL_REPLAY", "baseline"), 0.26
    for i, sc in enumerate(("A", "B")):
        for j, k in enumerate(keys):
            x = i + (j - 1) * width
            vals = data[sc][k]
            mean = float(np.mean(vals))
            ax.bar(x, mean, width * 0.92, color=COLOURS[k], label=LABELS[k] if i == 0 else None)
            # value at the bar's base: no fold dot reaches that low, so the label never collides
            ax.text(x, 0.415, f"{mean:.3f}", ha="center", va="bottom", fontsize=9,
                    color="white" if k == "FULL" else "#1a202c")
            if len(vals) > 1:
                ax.scatter([x] * len(vals), vals, s=14, color="#1a202c", zorder=3,
                           label="one held-out fold" if (i == 1 and j == 0) else None)
    ax.set_xticks([0, 1], ["Scenario A: repos seen in training", "Scenario B: repos never seen"])
    ax.set_ylabel("AUC-PR")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Within a project the model beats the baseline;\n"
                 "on unseen repos it does so only with the repo's own history", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, metadata=PNG_META)
    plt.close(fig)
    return data


def scatter_data(transfer: pd.DataFrame) -> dict[str, dict]:
    """What Figure 2 plots: per scenario, each repo's mean repo-feature SHAP contribution c
    against its actual slow rate y, with the Spearman correlation across repos."""
    out = {}
    for sc in ("A", "B"):
        c, y = transfer[f"c_{sc}"].to_numpy(), transfer[f"y_{sc}"].to_numpy()
        out[sc] = {"c": c, "y": y, "rho": float(spearmanr(c, y).statistic)}
    return out


def fingerprint_scatter(transfer: pd.DataFrame, out: Path) -> dict:
    """Figure 2: the Phase 6b transfer test, one point per repo."""
    data = scatter_data(transfer)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    titles = {"A": "Seen in training (Scenario A)", "B": "Held out (Scenario B)"}
    for ax, sc in zip(axes, ("A", "B")):
        d = data[sc]
        ax.scatter(d["c"], d["y"], s=18, color=COLOURS["FULL"], alpha=0.8)
        ax.set_title(f"{titles[sc]}: Spearman ρ = {d['rho']:+.2f}", fontsize=10)
        ax.set_xlabel("repo-level features' mean SHAP contribution (log-odds)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("repo's actual slow rate")
    fig.suptitle("The 8 repo-level features track each repo's slow rate more closely\n"
                 "when the repo was seen in training (one point per repo, 39 repos)", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, metadata=PNG_META)
    plt.close(fig)
    return data


def main(out_dir: Path = FIG) -> int:
    headline_transfer(load_runs(), out_dir / "headline_transfer.png")
    fingerprint_scatter(pd.read_csv(TRANSFER), out_dir / "fingerprint_scatter.png")
    print(f"wrote {out_dir / 'headline_transfer.png'} and {out_dir / 'fingerprint_scatter.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

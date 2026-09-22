"""The 50 predictions worth reading by hand.

Blueprint §3, Phase 6: "manual error analysis on worst 50 predictions". Both tails matter
for a triage tool -- a confident false positive wastes a lead's attention, a confident
false negative is a PR that silently rots -- so this takes 25 of each rather than the 50
largest absolute errors, which would be dominated by whichever class is larger.

error_patterns() is a first pass, not the analysis: it says which features look unusual
among the failures. Reading the rows is still the job."""
from __future__ import annotations

import numpy as np
import pandas as pd

GITHUB = "https://github.com/{repo}/pull/{number}"
TOP_K = 5


def worst_rows(pred: pd.DataFrame, shap_values: np.ndarray, cols: list[str],
               feature_frame: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """25 most-confident false positives + 25 most-confident false negatives.

    pred, shap_values and feature_frame are POSITIONALLY aligned: shap_values[i] explains
    pred.iloc[i], whose feature values are feature_frame.iloc[i]. (error_patterns below
    takes a pr_id-INDEXED frame instead -- the two are not interchangeable.)

    pred must carry `number` and `wait_h`; the prediction parquets have neither, so the
    caller joins them from experiment.load_table()."""
    pos = np.arange(len(pred))
    is_slow = pred["is_slow"].to_numpy(dtype=bool)
    p_hat = pred["p_hat"].to_numpy(dtype=float)

    fp = pos[~is_slow][np.argsort(-p_hat[~is_slow])][:n]        # not slow, most confident
    fn = pos[is_slow][np.argsort(p_hat[is_slow])][:n]           # slow, least confident

    rows = []
    for kind, idx in (("fp", fp), ("fn", fn)):
        for i in idx:
            src = pred.iloc[i]
            sv = shap_values[i]
            order = np.argsort(-np.abs(sv))[:TOP_K]
            row = {
                "kind": kind, "pr_id": src["pr_id"], "repo": src["repo"],
                "number": int(src["number"]),
                "url": GITHUB.format(repo=src["repo"], number=int(src["number"])),
                "p_hat": float(src["p_hat"]), "is_slow": bool(src["is_slow"]),
                "wait_h": float(src["wait_h"]),
            }
            vals = feature_frame.iloc[i]
            for rank, j in enumerate(order, start=1):
                row[f"top{rank}_feature"] = cols[j]
                row[f"top{rank}_shap"] = float(sv[j])
                row[f"top{rank}_value"] = vals[cols[j]]   # left raw: language_dominant is categorical
            rows.append(row)
    return pd.DataFrame(rows)


def error_patterns(worst: pd.DataFrame, feature_frame: pd.DataFrame,
                   cols: list[str]) -> pd.DataFrame:
    """How the 50 worst rows differ from the test set, per feature, as a z-score.

    feature_frame is indexed BY pr_id here (unlike worst_rows, which is positional).
    Non-numeric columns coerce to NaN and fall out of the ranking, which is intended --
    a z-score of `language_dominant` would mean nothing."""
    sub = feature_frame.loc[feature_frame.index.isin(worst["pr_id"])]
    out = []
    for c in cols:
        col = pd.to_numeric(feature_frame[c], errors="coerce")
        w = pd.to_numeric(sub[c], errors="coerce")
        sd = col.std()
        usable = np.isfinite(sd) and sd > 0          # note: `if sd` is True for NaN
        out.append({"feature": c, "mean_worst": float(w.mean()), "mean_all": float(col.mean()),
                    "z": float((w.mean() - col.mean()) / sd) if usable else 0.0})
    df = pd.DataFrame(out)
    return df.reindex(df["z"].abs().sort_values(ascending=False).index).reset_index(drop=True)

"""Append-only experiment log (blueprint §5): every run, not just the best one.

Fixed column set so the CSV stays machine-readable across phases. Add a column here
if a later phase needs one; never write ad-hoc keys."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENTS = Path(__file__).parent / "data" / "experiments.csv"

COLUMNS = [
    "date", "scenario", "fold", "model", "features", "params",
    "n_train", "n_test", "n_test_repos",
    "precision_at_10", "p10_ci_lo", "p10_ci_hi", "base_rate_p10",
    "auc_pr", "base_rate", "notes",
]


def log(row: dict, path: Path = EXPERIMENTS) -> None:
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise KeyError(f"unknown experiment columns: {sorted(unknown)}")
    full = {c: "" for c in COLUMNS}
    full["date"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    full.update({k: ("" if v is None else v) for k, v in row.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(full)

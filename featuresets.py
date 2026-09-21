"""The four pre-registered feature sets, derived from features.COLUMN_SPEC.

Nothing here is hand-listed: an ablation is a rule over the registry, so a column that
is a key, a label or a fidelity flag cannot end up in a model by accident. Ablations
exist to answer two questions the write-up must address -- does the model beat the
baseline WITHOUT the baseline's own feature (NO_LABEL_REPLAY), and are the repo-level
snapshots carrying the cold-start result (NO_SNAPSHOT)."""
from __future__ import annotations

from features import COLUMN_SPEC, feature_columns

# The baseline's feature and its author-level twin, with their support counts.
LABEL_REPLAY = ("trailing_90d_slow_rate", "trailing_n",
                "author_prior_slow_rate_here", "author_prior_n")

FORBIDDEN_STATUS = {"key", "label", "flag"}


def assert_hygiene(cols: list[str]) -> None:
    bad = [c for c in cols if c not in COLUMN_SPEC or COLUMN_SPEC[c]["status"] in FORBIDDEN_STATUS]
    if bad:
        raise ValueError(f"forbidden or unknown feature columns: {bad}")


def _full() -> list[str]:
    return list(feature_columns())


FEATURE_SETS: dict[str, list[str]] = {
    "FULL": _full(),
    "NO_SNAPSHOT": [c for c in _full() if COLUMN_SPEC[c]["status"] != "snapshot"],
    "NO_LABEL_REPLAY": [c for c in _full() if c not in LABEL_REPLAY],
    "PR_ONLY": [c for c in _full() if COLUMN_SPEC[c]["group"] in ("static", "at_open")],
}

for _name, _cols in FEATURE_SETS.items():
    assert_hygiene(_cols)

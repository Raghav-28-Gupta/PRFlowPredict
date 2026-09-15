"""Gate #6: the refactor must not change the target. Any difference must be explained
by exactly the two rule changes the spec makes: PENDING reviews no longer count, and
KNOWN_BOTS logins no longer count."""
import json
from pathlib import Path

import pytest

PRE = {
    "litestream": Path("data/processed/benbjohnson__litestream/gate_pre.json"),
    "skills": Path("data/processed/anthropics__skills/gate_pre.json"),
}
POST = {k: v.with_name("gate.json") for k, v in PRE.items()}
KEYS = ["label_rates", "never_reviewed_30d", "is_slow_rate", "label_spread"]


@pytest.mark.parametrize("name", list(PRE))
def test_label_fields_unchanged(name):
    if not (PRE[name].exists() and POST[name].exists()):
        pytest.skip("pilot gate data not present")
    pre = json.loads(PRE[name].read_text())[0]
    post = json.loads(POST[name].read_text())[0]
    for k in KEYS:
        if isinstance(pre[k], dict):
            assert list(pre[k].values()) == pytest.approx(list(post[k].values()), abs=1e-9), k
        else:
            assert pre[k] == pytest.approx(post[k], abs=1e-9), k

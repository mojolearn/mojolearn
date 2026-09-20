import json

import pytest

from tools.byte_lm_pool_ab_compare import compare


def test_requires_complete_equal_gate_records(tmp_path):
    record = {"status": "PASS", "scope": "two GPUs",
              "checks": [{"logical_shards": 3, "state_sha256": "abc",
                          "native_faults": True}]}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(record))
    b.write_text(json.dumps(record))
    assert compare(a, b)["exact"] is True
    moved = dict(record, checks=[dict(record["checks"][0], state_sha256="moved")])
    b.write_text(json.dumps(moved))
    with pytest.raises(ValueError, match="witnesses differ"):
        compare(a, b)


def test_refuses_nonpassing_arm(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"status": "PASS", "scope": "s", "checks": []}))
    b.write_text(json.dumps({"status": "FAIL", "scope": "s", "checks": []}))
    with pytest.raises(ValueError, match="must pass"):
        compare(a, b)

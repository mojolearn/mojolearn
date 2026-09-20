import copy
import json

import pytest

from tools.lm_shards_ab_compare import compare


def _run(seconds, devices):
    arm = {
        "logical_shards": 2, "devices": devices, "refused": None,
        "steady_median_seconds": seconds,
        "final_sha256": {"parameters": "p", "m": "m", "v": "v",
                         "flags": "f", "gradients": "g"},
        "steps": [{"losses_sha256": "loss"}],
    }
    return {"schema": "s", "shape": {"d_model": 768}, "parameters": 10,
            "steps_requested": 1, "seed": 7, "arms": [arm]}


def _files(tmp_path):
    runs = {
        "baseline_one": _run(4.0, [0]), "baseline_two": _run(2.0, [0, 1]),
        "candidate_one": _run(3.0, [0]), "candidate_two": _run(1.0, [0, 1]),
    }
    paths = {}
    for name, run in runs.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(run))
        paths[name] = path
    return runs, paths


def test_reports_all_four_ratios_after_strict_identity(tmp_path):
    _, paths = _files(tmp_path)
    row = compare(paths)[0]
    assert row["exact"] is True
    assert row["baseline_two_device_speedup"] == 2.0
    assert row["candidate_two_device_speedup"] == 3.0
    assert row["candidate_vs_baseline_two"] == 2.0


@pytest.mark.parametrize("field", ["final_sha256", "steps"])
def test_refuses_any_state_or_loss_difference(tmp_path, field):
    runs, paths = _files(tmp_path)
    broken = copy.deepcopy(runs["candidate_two"])
    if field == "final_sha256":
        broken["arms"][0][field]["m"] = "different"
    else:
        broken["arms"][0][field][0]["losses_sha256"] = "different"
    paths["candidate_two"].write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="differ"):
        compare(paths)

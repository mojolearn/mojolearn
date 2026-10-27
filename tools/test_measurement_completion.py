"""Missing scores/times are recorded; bad quality never becomes an admission gate."""
import copy

import pytest

import bench_board as B


def result():
    cells = [dict(arm=arm, status="ok", mode=mode, rounds=1,
                  times_ms=[2.], median_ms=2., warmup_ms=3.,
                  quality=dict(accuracy=.5, logloss=10.))
             for arm, mode in (("ours", "identical"), ("torch-eager-fp32", "opponent"))]
    return dict(family="algos", lane="gru-clf", cells=cells)


def test_poor_but_recorded_quality_is_complete_and_values_are_preserved():
    rec = result()
    before = copy.deepcopy(rec)
    assert B.measurement_completion(rec)["status"] == "recorded"
    assert rec == before


@pytest.mark.parametrize("change,expected", [
    (lambda c: c["quality"].pop("logloss"), "quality metric: logloss"),
    (lambda c: c.update(quality={}), "quality scores"),
    (lambda c: c.update(times_ms=[]), "complete finite timing samples"),
    (lambda c: c.update(times_ms=[float('nan')]), "complete finite timing samples"),
    (lambda c: c.update(rounds=2), "complete finite timing samples"),
    (lambda c: c.pop("median_ms"), "median_ms"),
    (lambda c: c.pop("warmup_ms"), "warmup_ms"),
])
@pytest.mark.parametrize("phase", ["fit", "infer"])
def test_missing_values_identify_arm_and_phase(change, expected, phase):
    rec = result()
    if phase == "infer":
        rec["infer_cells"] = copy.deepcopy(rec["cells"])
    cell = rec["cells" if phase == "fit" else "infer_cells"][1]
    change(cell)
    report = B.measurement_completion(rec)
    assert report["status"] == "incomplete"
    assert any(i["arm"] == cell["arm"] and i["phase"] == phase and expected in i["missing"]
               for i in report["issues"])


def test_existing_scorer_metrics_detect_asymmetric_recording():
    rec = result()
    rec.update(family="neural", lane="mlp-fit")
    rec["cells"][0]["quality"]["auc"] = .7
    assert "quality metric: auc" in B.measurement_completion(rec)["issues"][0]["missing"]


def test_self_reference_and_failed_arms_are_not_invented_missing_scores():
    rec = result()
    rec.update(lane="gcn")
    rec["cells"][0]["quality"] = {"rel_fro_vs_torch_eager_fp32": .002}
    rec["cells"][1]["quality"] = {"reference": "self", "relative_error": 0.}
    rec["cells"].append(dict(arm="unavailable", status="REFUSED", quality={}))
    assert B.measurement_completion(rec)["status"] == "recorded"


def test_no_scored_cells_is_not_a_success_claim():
    assert B.measurement_completion(dict(cells=[]))["status"] == "not measured"

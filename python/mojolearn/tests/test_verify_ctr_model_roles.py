"""Saved-file CPU replay must not replace a GPU model-byte reference."""
import copy

from mojolearn import _verify_reference as ref
from mojolearn import _verify_all as verify
from mojolearn._verification_coverage import reference_support


LANE = "gbdt-categorical-ctr-tables"
HASH = "bf3bea6dea6870d5"
NA = "n/a:gpu-saved-file (a CPU column loads the GPU column's model and writes none)"


def table():
    return dict(cells={LANE + "/base": {"model": dict(ref=NA, cols={"apple": [0, HASH], "cpu": 1})}},
                records=[dict(dir="gpu", file="column.json", vendor="apple-m4", commit="a" * 40),
                         dict(dir="cpu", file="column.json", vendor="cpu", commit="b" * 40)])


def test_newer_cpu_na_preserves_gpu_reference_and_real_mismatch_detection():
    data = table()
    before = copy.deepcopy(data)
    row = dict(lane=LANE, fixture="base", part="model", value=HASH)
    good, = verify.judge_rows([row], data, device_class="apple")
    assert good["state"] == "IDENTICAL" and good["reference"] == HASH
    assert set(good["columns"]) == {"apple"}
    bad, = verify.judge_rows([dict(row, value="0" * 16)], data, device_class="apple")
    assert bad["state"] == "DIVERGENT"
    assert data == before, "a device projection must not rewrite the table"


def test_cpu_counts_no_model_bytes_and_reload_errors_still_fail():
    row = dict(lane=LANE, fixture="base", part="model", value=NA)
    good, = verify.judge_rows([row], table(), device_class="cpu")
    assert good["state"] == "N/A" and set(good["columns"]) == {"cpu"}
    bad, = verify.judge_rows([dict(row, value=None, error="second load changed predictions")],
                            table(), device_class="cpu")
    assert bad["state"] == "REFUSED"


def test_gpu_disagreement_is_never_resolved_by_picking_this_device():
    data = table()
    data["cells"][LANE + "/base"]["model"]["cols"]["nvidia"] = [2, "0" * 16]
    ent = ref.entry(data, LANE, "base", "model", device_class="apple")
    assert ent["conflict"] and ref.judge(HASH, ent)[0] == "OWED"


def test_unrecorded_cpu_exemption_and_other_lane_na_are_not_invented():
    data = table()
    data["cells"][LANE + "/base"]["model"]["cols"].pop("cpu")
    assert ref.entry(data, LANE, "base", "model", device_class="apple")["ref"] == NA
    data = table()
    data["cells"]["par-cd/base"] = data["cells"].pop(LANE + "/base")
    assert ref.entry(data, "par-cd", "base", "model", device_class="apple")["ref"] == NA


def test_coverage_reports_the_same_device_role_as_execution():
    gpu = reference_support(table(), LANE, ["base"], ["model"], vendor_class="apple")["model"]
    cpu = reference_support(table(), LANE, ["base"], ["model"], vendor_class="cpu")["model"]
    assert gpu["numerical_fixtures"] == gpu["agreeing_device_classes"]["apple"] == 1
    assert gpu["agreeing_device_classes"]["cpu"] == 0
    assert cpu["not_applicable_fixtures"] == 1 and cpu["numerical_fixtures"] == 0

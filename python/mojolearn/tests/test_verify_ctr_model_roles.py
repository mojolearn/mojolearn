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


def test_apple_disagreement_conflicts_with_other_gpu_model_bytes():
    data = table()
    data["cells"][LANE + "/base"]["model"]["cols"]["nvidia"] = [2, "0" * 16]
    ent = ref.entry(data, LANE, "base", "model", device_class="apple")
    assert ent["conflict"]
    assert ref.judge(HASH, ent)[0] == "OWED"
    assert ref.judge("0" * 16, ent)[0] == "OWED"


def test_nvidia_amd_disagreement_still_has_no_reference():
    data = table()
    data["cells"][LANE + "/base"]["model"]["cols"].update(
        nvidia=[2, HASH], amd=[3, "0" * 16])
    for device in ("apple", "nvidia", "amd"):
        ent = ref.entry(data, LANE, "base", "model", device_class=device)
        assert ent["conflict"] and ref.judge(HASH, ent)[0] == "OWED"


def test_apple_cannot_pass_against_its_own_differing_record():
    shared = "a" * 16
    apple = "b" * 16
    data = dict(cells={"x/base": {"train": dict(ref=shared, cols={
        "apple": [0, apple], "nvidia": 1, "amd": 2})}}, records=[
        dict(dir="run", file=name + ".json", vendor=name, commit="a" * 40)
        for name in ("apple", "nvidia", "amd")])
    for device in ("apple", "nvidia", "amd"):
        good, = verify.judge_rows([dict(lane="x", fixture="base", part="train", value=shared)],
                                 data, device_class=device)
        bad, = verify.judge_rows([dict(lane="x", fixture="base", part="train", value=apple)],
                                data, device_class=device)
        assert good["state"] == "IDENTICAL"
        assert bad["state"] == "DIVERGENT"
        assert bad["reference"] == shared and bad["reference_scope"] == "shared"


def test_apple_uses_shared_reference_without_an_apple_observation():
    ent = dict(ref=HASH, cols={"nvidia": 0, "amd": 1})
    data = dict(cells={"x/base": {"train": ent}})
    assert ref.entry(data, "x", "base", "train", device_class="apple") is ent
    assert ref.judge(HASH, ent)[0] == "IDENTICAL"


def test_unrecorded_cpu_exemption_and_other_lane_na_are_not_invented():
    data = table()
    data["cells"][LANE + "/base"]["model"]["cols"].pop("cpu")
    assert ref.entry(data, LANE, "base", "model", device_class="cpu")["ref"] == NA
    data = table()
    data["cells"]["par-cd/base"] = data["cells"].pop(LANE + "/base")
    assert ref.entry(data, "par-cd", "base", "model", device_class="cpu")["ref"] == NA


def test_coverage_reports_the_same_device_role_as_execution():
    gpu = reference_support(table(), LANE, ["base"], ["model"], vendor_class="apple")["model"]
    cpu = reference_support(table(), LANE, ["base"], ["model"], vendor_class="cpu")["model"]
    assert gpu["numerical_fixtures"] == gpu["agreeing_device_classes"]["apple"] == 1
    assert gpu["agreeing_device_classes"]["cpu"] == 0
    assert cpu["not_applicable_fixtures"] == 1 and cpu["numerical_fixtures"] == 0

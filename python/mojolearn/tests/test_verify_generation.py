# SPDX-License-Identifier: Apache-2.0
import copy
import pytest
from mojolearn._verify_generation import AUDITS, CASES, CHECKS, PARTS, PROFILE, compare


def report():
    return dict(profile=PROFILE, source_sha256="source", device="gpu", status="CAPTURED_UNQUALIFIED", cases=[
        dict(family=family, numeric_profile=profile, mode=1, vendor="cuda",
             checks=dict.fromkeys(CHECKS, True), parts=dict.fromkeys(PARTS, "a" * 64),
             checkpoints={"model": "b" * 64}, audits=[dict(operation=name,
                 schema="mojolearn.native-runtime-audit.v1", passed=True, wrapper_entries=1,
                 wrapper_work=[], python_frames=[], forbidden_native_calls=[],
                 unobserved_existing_threads=[], exception=None) for name in AUDITS])
        for family, profile in CASES])


def test_independent_capture_and_output_fault():
    left = report()
    right = copy.deepcopy(left)
    for cell in right["cases"]:
        cell["vendor"] = "hip"
    assert compare(left, right)
    right["cases"][0]["parts"]["sampled"] = "c" * 64
    assert not compare(left, right)


@pytest.mark.parametrize("fault", ["truncated", "empty", "duplicate", "mode", "vendor", "check", "audit", "audit_name", "audit_duplicate", "audit_frames", "source", "checkpoint", "part"])
def test_incomplete_or_misrouted_capture_never_passes(fault):
    record = report()
    if fault == "truncated": record["cases"].pop()
    if fault == "empty": record["cases"] = []
    if fault == "duplicate": record["cases"].append(copy.deepcopy(record["cases"][0]))
    if fault == "mode": record["cases"][0]["mode"] = 0
    if fault == "vendor": record["cases"][0]["vendor"] = "cpu"
    if fault == "check": record["cases"][0]["checks"]["stream"] = False
    if fault == "audit": record["cases"][0]["audits"].pop()
    if fault == "audit_name": record["cases"][0]["audits"][0]["operation"] = "unrelated"
    if fault == "audit_duplicate": record["cases"][0]["audits"][0] = record["cases"][0]["audits"][1]
    if fault == "audit_frames": record["cases"][0]["audits"][0]["python_frames"] = [{"function": "callback"}]
    if fault == "source": record["source_sha256"] = ""
    if fault == "checkpoint": record["cases"][0]["checkpoints"] = {}
    if fault == "part": record["cases"][0]["parts"].pop("resumed")
    with pytest.raises(ValueError):
        compare(record, record)

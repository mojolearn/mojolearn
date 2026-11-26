"""`verify --all`'s single-device commands (_verify_extended.py): the units, and the verdict of each profile.

Judging is a pure function of the captures, so these run without a device: every verdict can be reached and a
capture with nothing to compare against reads OWED, never VERIFIED.
"""
import pytest

from mojolearn import _verify_extended as ext
from mojolearn import _verify_distributed_program as dist


def ok(value):
    return dict(ok=True, value=value)


def program_report(digest="a" * 64, status="PASS"):
    return dict(format=dist.FORMAT, status=status, cases=[
        dict(name="analytic_sgd", status="PASS", digest=digest),
        dict(name="checkpoint_resume", status="PASS", digest="b" * 64),
        dict(name="uninterrupted_replay", status="PASS", digest="b" * 64)])


@pytest.mark.parametrize("vclass, sides", [("cpu", 1), ("nvidia", 2)])
def test_units_are_one_device_and_cover_every_profile(vclass, sides):
    units = ext.units(vclass)
    assert {p for p, _ in units} == set(ext.PROFILES)
    assert sum(p == "verify-causal-lm" for p, _ in units) == sides
    assert dict(units)["verify-model"][0] == "save"
    assert not any("devices" in str(steps) for _, steps in units)


def test_program_without_a_reference_is_owed_never_verified():
    block = ext.judge("verify-distributed-program", {"run": ok(program_report())})
    assert block["verdict"] == ext.OWED and block["counts"]["passed"] == 3
    assert "PENDING" not in str(block)


def test_program_against_a_reference_verifies_or_diverges():
    ref = program_report()
    assert ext.judge("verify-distributed-program", {"run": ok(program_report())}, ref)["verdict"] == ext.VERIFIED
    moved = ext.judge("verify-distributed-program", {"run": ok(program_report("c" * 64))}, ref)
    assert moved["verdict"] == ext.DIVERGENT and "analytic_sgd" in moved["detail"]
    other = dict(ref, format="something-else")
    assert ext.judge("verify-distributed-program", {"run": ok(program_report())}, other)["verdict"] == ext.OWED


def test_a_failed_case_or_a_dead_worker_fails():
    bad = program_report()
    bad["status"], bad["failure"] = "FAIL", dict(message="wrong analytic answer")
    assert ext.judge("verify-partitioned-program", {"run": ok(bad)})["verdict"] == ext.FAILED
    dead = ext.judge("verify-graph", {"run": dict(ok=False, error="worker exited -11")})
    assert dead["verdict"] == ext.FAILED and "-11" in dead["detail"]


def test_two_sided_captures_compare_cpu_with_gpu(monkeypatch):
    from mojolearn import _verify_causal_lm as lm
    cap = dict(status="CAPTURED_UNQUALIFIED", cases=[])
    monkeypatch.setattr(lm, "compare", lambda a, b, notes=None: True)
    assert ext.judge("verify-causal-lm", {"cpu": ok(cap), "gpu": ok(cap)})["verdict"] == ext.VERIFIED
    monkeypatch.setattr(lm, "compare", lambda a, b, notes=None: False)
    assert ext.judge("verify-causal-lm", {"cpu": ok(cap), "gpu": ok(cap)})["verdict"] == ext.DIVERGENT
    assert ext.judge("verify-causal-lm", {"cpu": ok(cap)})["verdict"] == ext.OWED
    failed = dict(status="PROPERTY_FAILURE", cases=[dict(architecture="llama", weight_format="int8",
                                                         checks=dict(reload=False, batch=True))])
    block = ext.judge("verify-causal-lm", {"cpu": ok(failed)})
    assert block["verdict"] == ext.FAILED and "reload" in block["detail"]


def check(device, verdict="MATCH"):
    return ok(dict(device=device, verdict=verdict, notes=[],
                   outputs=[dict(name="output0", verdict=verdict)]))


def test_saved_program_round_trip():
    saved = ok(dict(path="p", report=dict(install=dict(vendor="cuda"))))
    gpu = ext.judge("verify-model", {"save": saved, "check-cpu": check("cpu"), "check-gpu": check("gpu")})
    assert gpu["verdict"] == ext.VERIFIED
    on_cpu = ok(dict(path="p", report=dict(install=dict(vendor="cpu"))))
    assert ext.judge("verify-model", {"save": on_cpu, "check-cpu": check("cpu")})["verdict"] == ext.OWED
    moved = ext.judge("verify-model", {"save": saved, "check-cpu": check("cpu", "DIFFER"), "check-gpu": check("gpu")})
    assert moved["verdict"] == ext.DIVERGENT and "cpu: output0" in moved["detail"]


def test_gate_and_summary():
    blocks = {"a": dict(verdict=ext.OWED), "b": dict(verdict=ext.FAILED), "c": dict(verdict=ext.VERIFIED)}
    assert ext.gate(blocks) == ["b"]
    assert ext.summary(blocks) == {ext.VERIFIED: 1, ext.DIVERGENT: 0, ext.FAILED: 1, ext.OWED: 1}


def test_no_multi_gpu_command_is_named_anywhere_in_verify():
    import inspect
    from mojolearn import __main__ as cli
    text = inspect.getsource(ext) + "".join(ext.human_lines({}))
    assert "verify-distributed " not in text and "verify-cross-validation" not in text
    for command in ("verify-distributed", "verify-cross-validation"):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([command])

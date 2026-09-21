# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`python -m mojolearn verify --par`: the two-device column against the
one-device column, and the two things that make it worth running.

    cd python && python3 -m mojolearn.tests.test_verify_par      (or pytest)

THE TWO THINGS. First, the comparison must be able to report DIVERGENT; a
comparison never seen to fail is not a comparison, so the perturbed arm is
tested here and is also runnable end to end (`verify --par-self-test`).
Second, the two-device column must be WITNESSED; without that, a driver that
silently falls back to one device produces a clean AGREE in a second, which is
a failure this repository has already shipped once. Both are tested against
the real functions, not against a printed verdict.

No GPU is needed for any test in this file. The lane-selection test needs the
real harness, which needs NumPy, and states that the same way test_verify_all
does.
"""
import os

import pytest

from mojolearn import __main__ as cli
from mojolearn import _verify_all as va
from mojolearn import _verify_par as vpar

try:
    import numpy as _numpy  # noqa: F401
except ImportError as _numpy_exc:  # pragma: no cover - depends on the interpreter
    _NO_NUMPY = f"verification needs NumPy and this interpreter has none: {_numpy_exc}"
else:
    _NO_NUMPY = ""


def _need_numpy():
    if _NO_NUMPY:
        pytest.skip(_NO_NUMPY)


def _cell(**parts):
    """{part: (value, error)} in the shape `run_cell` returns."""
    out = {}
    for part in ("train", "infer", "model", "batch", "stepfull"):
        value = parts.get(part, "n/a:not-in-this-test")
        out[part] = value if isinstance(value, tuple) else (value, None)
    return out


def _recorded(monkeypatch, one):
    """The reference table's one-device values, taken from a `_cell`: the
    two-device column is the only thing that runs."""
    monkeypatch.setattr(vpar.vref, "load_table", lambda *a, **k: {})
    monkeypatch.setattr(vpar.vref, "entry", lambda table, lane, fx, part, cls=None:
                        dict(ref=one.get(part, (None, "missing"))[0]))


# ------------------------------------------------------- the comparison

def test_the_comparison_reports_divergent_and_names_both_hashes():
    """The whole point. Two columns that disagree must read DIVERGENT and BOTH
    hashes must survive to the report, or a reader cannot see which two values
    disagreed and is taking the verdict on trust."""
    same = vpar.compare_parts(_cell(train="aaaa000011112222"),
                              _cell(train="aaaa000011112222"))
    assert [r for r in same if r["part"] == "train"][0]["verdict"] == "IDENTICAL"

    moved = vpar.compare_parts(_cell(train="aaaa000011112222"),
                               _cell(train="ffff999988887777"))
    row = [r for r in moved if r["part"] == "train"][0]
    assert row["verdict"] == "DIVERGENT"
    assert "DIVERGENT" in vpar.GATING, "sharding that changed the bits must gate"

    r = dict(ran=True, vendor="hip", devices=[0, 1], repeats=1, lanes=["par-gram"],
             fixtures=["base"], compared=1, counts={"DIVERGENT": 1}, cells=[
                 dict(lane="par-gram", fixture="base", seconds=1.0, **row)],
             perturbed=False, witness=dict(pools=2, workers=4, devices=[0, 1],
                                           placement="physical", detail=[]),
             witness_refusal=None, passed=False)
    text = vpar.format_par_check(r)
    assert "DISAGREE" in text
    assert "aaaa000011112222" in text and "ffff999988887777" in text


def test_a_part_refused_on_exactly_one_column_is_a_finding_and_gates():
    """`par-queries-nn`'s batch shape. On 2026-09-19 a two-device MI300X run
    refused that lane's batch part on all nine fixtures while the one-device
    column produced a stable hash. That is a real shipped defect and the only
    run that can see it is a two-device one, so ONE-COLUMN gates."""
    rows = vpar.compare_parts(_cell(batch="da944e45126bd8e0"),
                              _cell(batch=(None, "batch: RuntimeError: GPU worker failed")))
    row = [r for r in rows if r["part"] == "batch"][0]
    assert row["verdict"] == "ONE-COLUMN"
    assert "ONE-COLUMN" in vpar.GATING
    assert row["one"] == "da944e45126bd8e0" and row["two"] is None


def test_the_cpu_routes_own_declared_limit_is_named_not_counted_as_a_defect():
    """MEASURED 2026-09-20 on a CPU-only install: par-mlp, par-samba and
    par-samba-clip refuse the two-device column with `_cpu_refusal`'s own
    sentence, because `mlp_update` and `samba_update` split their gradient
    columns inside the GPU binding and no host binding restates that. Reading
    those three as the same finding as `par-queries-nn` would bury the one real
    defect under three declared limits. It is a separate verdict, it prints,
    and it does not gate. On a GPU install `_cpu_refusal` never fires."""
    refusal = ("no CPU implementation of the cooperative multi-GPU driver mlp_update across "
               "2 devices yet")
    rows = vpar.compare_parts(_cell(train="6e5cde4953362927"),
                              _cell(train=(None, refusal)), cpu_route=True)
    assert [r for r in rows if r["part"] == "train"][0]["verdict"] == "CPU-ROUTE-LIMIT"
    assert "CPU-ROUTE-LIMIT" not in vpar.GATING

    # the same shape on a GPU install is the real finding and still gates
    gpu = vpar.compare_parts(_cell(train="6e5cde4953362927"),
                             _cell(train=(None, refusal)), cpu_route=False)
    assert [r for r in gpu if r["part"] == "train"][0]["verdict"] == "ONE-COLUMN"

    # and a CPU-only install still gates on a refusal that is NOT that sentence
    real = vpar.compare_parts(_cell(batch="da944e45126bd8e0"),
                              _cell(batch=(None, "batch: RuntimeError: GPU worker failed")),
                              cpu_route=True)
    assert [r for r in real if r["part"] == "batch"][0]["verdict"] == "ONE-COLUMN"


def test_both_columns_refusing_the_same_way_is_not_a_difference():
    """A lane with no route on this install refuses on both sides. That is not
    a disagreement about bits and must not be reported as one; the reason is
    printed instead."""
    rows = vpar.compare_parts(_cell(train=(None, "no CPU implementation of gmm_fit")),
                              _cell(train=(None, "no CPU implementation of gmm_fit")))
    row = [r for r in rows if r["part"] == "train"][0]
    assert row["verdict"] == "REFUSED"
    assert "REFUSED" not in vpar.GATING


def test_a_declared_n_a_part_is_respected_not_invented():
    rows = vpar.compare_parts(_cell(batch="n/a:transductive"), _cell(batch="n/a:transductive"))
    assert [r for r in rows if r["part"] == "batch"][0]["verdict"] == "N/A"


# ------------------------------------- the guard against a silent fallback

class _FakePool:
    def __init__(self, devices, cooperative=False):
        self.devices = tuple(devices)
        self.cooperative = cooperative


def _record(pid, uuid, bus):
    return dict(kind="visible-device-inventory", vendor="hip", pid=pid,
                devices=[dict(ordinal=0, uuid=uuid, pci_bus_id=bus, name="fake")])


def test_a_column_that_started_no_pool_is_refused_not_counted():
    """THE FAILURE THIS FEATURE EXISTS TO PREVENT. A driver that falls back to
    one device makes both columns agree instantly. Reported as AGREE that is a
    lie; the witness must refuse it by name."""
    w = vpar.PoolWitness("hip", (0, 1))
    assert w.refusal() is not None
    assert "NO device pool" in w.refusal()


def test_a_pool_on_the_wrong_devices_is_refused():
    w = vpar.PoolWitness("hip", (0, 1))
    with pytest.raises(RuntimeError, match=r"devices \[0\] while the column asked"):
        w._admit(_FakePool((0,)), [_record(11, "a" * 32, "0000:01:00.0")])


def test_two_workers_on_one_physical_gpu_are_refused():
    """The lifted rule. `require_distinct_workers` rejects a repeated UUID or
    PCI id, which is how two workers masked onto one card would otherwise
    produce a two-device column that never used two devices."""
    w = vpar.PoolWitness("hip", (0, 1))
    w._admit(_FakePool((0, 1)), [_record(11, "a" * 32, "0000:01:00.0"),
                                 _record(12, "b" * 32, "0000:02:00.0")])
    with pytest.raises(RuntimeError, match="repeated physical device or process"):
        w._admit(_FakePool((0, 1)), [_record(11, "a" * 32, "0000:01:00.0"),
                                     _record(12, "a" * 32, "0000:02:00.0")])


def test_a_cooperative_pool_seeing_one_device_is_refused():
    """A cooperative pool is ONE worker with the whole group visible to it, so
    `require_distinct_workers` is the wrong rule and is not used. What it must
    still refuse is a worker that can see fewer devices than the column names."""
    w = vpar.PoolWitness("hip", (0, 1))
    both = dict(kind="visible-device-inventory", vendor="hip", pid=11,
                devices=[dict(ordinal=0, uuid="a" * 32, pci_bus_id="0000:01:00.0", name="f"),
                         dict(ordinal=1, uuid="b" * 32, pci_bus_id="0000:02:00.0", name="f")])
    w._admit(_FakePool((0, 1), cooperative=True), [both])
    one = dict(both, devices=both["devices"][:1])
    with pytest.raises(RuntimeError, match="sees 1 device"):
        w._admit(_FakePool((0, 1), cooperative=True), [one])


# ------------------------------- drivers that shard inside the native binding

class _FakeTrainer:
    """The three attributes `_admit_session` reads from a byte-LM trainer."""

    def __init__(self, devices, rows):
        self.devices, self._rows = tuple(devices), rows

    def optimizer_ownership(self):
        return tuple(self._rows)


_TWO_GPUS = [dict(ordinal=0, uuid="a" * 32, pci_bus_id="0000:01:00.0", name="f"),
             dict(ordinal=1, uuid="b" * 32, pci_bus_id="0000:02:00.0", name="f")]


def _session_witness(monkeypatch, inventory=None, **kw):
    w = vpar.PoolWitness("hip", (0, 1), **kw)
    monkeypatch.setattr(w, "_physical_inventory", lambda: list(inventory or _TWO_GPUS))
    return w


def test_a_native_session_on_both_devices_is_a_witness(monkeypatch):
    """MEASURED 2026-09-20 on 2x RTX 4090: `par-byte-lm` shards through the
    native binding in this process and starts no `DevicePool`, so a witness
    that counts only pools refused a column that WAS sharding."""
    w = _session_witness(monkeypatch)
    rows = [dict(device=0, first=0, count=10, moment_bytes=80, rollback_bytes=0, reduction_bytes=8),
            dict(device=1, first=10, count=10, moment_bytes=80, rollback_bytes=0, reduction_bytes=8)]
    w.sessions.append(w._admit_session(_FakeTrainer((0, 1), rows)))
    assert w.refusal() is None
    s = w.summary()
    assert (s["pools"], s["native_sessions"]) == (0, 1)
    assert s["detail"][0]["workers"][0]["devices"] == ["a" * 32, "b" * 32]


def test_a_native_session_whose_second_device_owns_nothing_is_refused(monkeypatch):
    w = _session_witness(monkeypatch)
    rows = [dict(device=0, first=0, count=20, moment_bytes=160),
            dict(device=1, first=20, count=0, moment_bytes=0)]
    with pytest.raises(RuntimeError, match="own nothing, so this column did not shard"):
        w._admit_session(_FakeTrainer((0, 1), rows))


def test_a_native_session_on_one_physical_gpu_twice_is_refused(monkeypatch):
    same = [_TWO_GPUS[0], dict(_TWO_GPUS[1], uuid="a" * 32)]
    w = _session_witness(monkeypatch, inventory=same)
    rows = [dict(device=0, moment_bytes=8), dict(device=1, moment_bytes=8)]
    with pytest.raises(RuntimeError, match="one physical GPU twice"):
        w._admit_session(_FakeTrainer((0, 1), rows))


def test_the_one_device_reference_session_admits_nothing(monkeypatch):
    """Every byte-LM lane opens its in-cell replica reference on the first
    device. That session is recorded and must never stand in for the driver
    under test."""
    w = _session_witness(monkeypatch)
    w.sessions.append(w._admit_session(_FakeTrainer((0,), [dict(device=0, moment_bytes=8)])))
    assert w.sessions[0]["placed"] is False
    assert "NO device pool" in w.refusal()


def test_a_native_session_is_never_a_witness_on_the_cpu_route(monkeypatch):
    w = vpar.PoolWitness("cpu", (0, 1))
    w.sessions.append(w._admit_session(_FakeTrainer((0, 1), [])))
    assert "NO device pool" in w.refusal()


def test_one_device_by_design_is_checked_not_trusted(monkeypatch):
    driver = vpar.ONE_DEVICE_BY_DESIGN["par-byte-lm-offload"][0]
    w = _session_witness(monkeypatch, one_device_driver=driver)
    assert "never opened that driver" in w.refusal()
    w.sessions.append(dict(kind=vpar.NATIVE_SESSION, driver=driver, devices=[0],
                           placed=False, workers=[]))
    assert w.refusal() is None
    w.sessions.append(dict(kind=vpar.NATIVE_SESSION, driver=driver, devices=[0, 1],
                           placed=True, workers=[]))
    assert "declaration is stale" in w.refusal()


def test_a_one_device_lane_reads_na_and_never_identical(monkeypatch):
    """Both columns of `par-byte-lm-offload` run on the first device, so an
    agreement is not a two-device comparison. It must not be counted as one,
    and a part that DIFFERS must still gate."""
    same = _cell(train="aaaa000011112222")
    r = _par_check(monkeypatch, same, same, lanes=("par-byte-lm-offload",))
    assert r["counts"].get("IDENTICAL", 0) == 0
    assert (r["state"], r["passed"]) == ("NOTHING COMPARED", None)
    assert [c for c in r["cells"] if c["part"] == "train"][0]["note"].startswith("n/a:one-device")
    r = _par_check(monkeypatch, same, _cell(train="bbbb000011112222"),
                   lanes=("par-byte-lm-offload",))
    assert (r["state"], r["passed"]) == ("MISMATCH", False)


def test_the_witness_refusal_beats_a_clean_agreement_in_the_report():
    """Even when every part agrees, a column that was not shown to have
    sharded must not read as a pass."""
    r = dict(ran=True, vendor="hip", devices=[0, 1], repeats=1, lanes=["par-gram"],
             fixtures=["base"], compared=1, counts={"IDENTICAL": 1},
             cells=[dict(lane="par-gram", fixture="base", seconds=0.1, part="train",
                         verdict="IDENTICAL", one="aaaa", two="aaaa",
                         one_error=None, two_error=None)],
             perturbed=False,
             witness=dict(pools=0, workers=0, devices=[0, 1], placement="physical", detail=[]),
             witness_refusal="the two-device column started NO device pool at all",
             passed=True)
    text = vpar.format_par_check(r)
    assert "WITNESS REFUSED" in text
    assert "CANNOT RUN" in text
    assert "AGREE on" not in text


# --------------------------------------------------- what the box may run

def test_apple_is_refused_structurally_and_says_where_the_rule_lives():
    """Sending someone to find a Mac with two GPUs is sending them after a
    machine this code could never use. The refusal must name the line."""
    why = vpar.refuse_to_run("metal", (0, 1))
    assert why and "STRUCTURALLY" in why
    assert "DevicePool._start" in why and "group == (0,)" in why


def test_the_device_count_door_exists_and_refuses_a_box_without_the_devices():
    """`visible_gpu_inventory` is asked inside a worker whose mask is already
    one device, so it answers 1 by construction and cannot say whether this box
    has two. This is the door that can, and it must refuse rather than guess."""
    from mojolearn import _gpu_witness as gw
    assert callable(gw.require_device_count) and callable(gw.unmasked_device_count)
    with pytest.raises(RuntimeError, match="requires CUDA or HIP"):
        gw.unmasked_device_count("metal")
    original = gw.unmasked_device_count
    try:
        gw.unmasked_device_count = lambda vendor, timeout=None: 1
        with pytest.raises(RuntimeError, match="shows 1 hip device"):
            gw.require_device_count("hip", 2)
        assert gw.require_device_count("hip", 1) == 1
    finally:
        gw.unmasked_device_count = original


def test_par_devices_must_name_more_than_one_device():
    for bad in ("0", "0,0", "1", "-1,0", "a,b"):
        with pytest.raises(va.CannotRun):
            vpar._parse_devices(bad)
    assert vpar._parse_devices("") == (0, 1)
    assert vpar._parse_devices("2,5") == (2, 5)


# ----------------------------------------------------- the column flip

def test_the_env_var_is_set_per_column_and_restored():
    """The whole flip is this variable, which `identity_break._par_devices()`
    reads at FIT time and `DevicePool._start` copies into each worker at spawn
    time. Nothing else changes, and nothing is left behind."""
    before = os.environ.get(vpar.PAR_DEVICES_ENV)
    with vpar.par_devices((0,)):
        assert os.environ[vpar.PAR_DEVICES_ENV] == "0"
        with vpar.par_devices((0, 1)):
            assert os.environ[vpar.PAR_DEVICES_ENV] == "0,1"
        assert os.environ[vpar.PAR_DEVICES_ENV] == "0"
    assert os.environ.get(vpar.PAR_DEVICES_ENV) == before


def test_an_ambient_par_devices_is_still_refused_but_with_the_right_sentence():
    """The `_verify_all` refusal stays for every other command. For --par it is
    still raised, because silently overwriting what the user exported is worse
    than saying so, but 'the reference is the one-device record' is not the
    reason here and the message must not claim it is."""
    previous = os.environ.get("MOJOLEARN_PAR_DEVICES")
    os.environ["MOJOLEARN_PAR_DEVICES"] = "0,1"
    try:
        with pytest.raises(va.CannotRun, match="the reference is the one-device record"):
            va.load_harness()
        with pytest.raises(va.CannotRun, match="--par-devices"):
            va.load_harness(par_axis=True)
    finally:
        if previous is None:
            os.environ.pop("MOJOLEARN_PAR_DEVICES", None)
        else:
            os.environ["MOJOLEARN_PAR_DEVICES"] = previous


# ---------------------------------------------------------- the lane set

def test_every_par_lane_is_in_the_default_scope_and_none_is_scoped_out():
    """A verifier that reports one real failure is worth more than one that
    reports nothing because the failing case was scoped out. `par-queries-nn`
    in particular, whose batch part is the known two-device defect."""
    _need_numpy()
    harness = va.load_harness()
    chosen, every, per_family = vpar.par_lanes(harness, "default")
    assert chosen == every
    assert set(every) == {n for n in harness.LANES if n.startswith("par-")}
    assert "par-queries-nn" in chosen, "the one lane with a known two-device defect must not be cut"
    quick, _, _ = vpar.par_lanes(harness, "quick")
    assert set(quick) == set(per_family.values())
    assert set(quick) <= set(every)


def test_the_cli_routes_par_to_the_suite_and_not_to_the_kmeans_card():
    p = cli.build_parser()
    args = p.parse_args(["verify", "--par"])
    assert args.par == "default" and cli._wants_suite(args)
    args = p.parse_args(["verify", "--par", "quick", "--par-devices", "0,2"])
    assert args.par == "quick" and args.par_devices == "0,2"
    args = p.parse_args(["verify", "--par-self-test"])
    assert args.par_self_test and cli._wants_suite(args)
    plain = p.parse_args(["verify"])
    assert plain.par is None and not plain.par_self_test


# ------------------------------------------------- a run that ran nothing
# CLOSED 2026-09-20. `par_check` ended in
# `passed = None if not cells else (gated == 0)`, and `cells` is appended once
# per PART whatever the part read, so `None` could only mean the LANE LIST was
# empty. A run where every part REFUSED on both columns counted zero gating
# verdicts and printed `RESULT: THE TWO COLUMNS AGREE` at exit 0. The module's
# own docstring at `par_check` records that state as measured.


class _StubWitness:
    """Two pools, four workers, nothing to refuse: the witness is not what
    these tests are about and must not be what decides them."""

    def __init__(self, vendor, devices, **kw):
        self.vendor, self.devices = vendor, tuple(devices)

    def watching(self):
        import contextlib
        return contextlib.nullcontext()

    def summary(self):
        return dict(pools=2, workers=4, devices=list(self.devices), placement="physical", detail=[])

    def refusal(self):
        return None


class _StubHarness:
    FIXTURES = ("base",)

    def fixture(self, name):
        import numpy as np
        return np.zeros((4, 2)), np.zeros(4), np.zeros(4)

    def heldout(self, name):
        import numpy as np
        return np.zeros((2, 2))


def _par_check(monkeypatch, one, two, lanes=("par-gram",)):
    """`par_check` with the two columns handed to it, so the verdict ladder is
    exercised against the real function and not against a rebuilt dict."""
    import contextlib
    from mojolearn import _backend, _cpu_reference
    _need_numpy()
    monkeypatch.setattr(_backend, "vendor", lambda: "hip")
    monkeypatch.setattr(_cpu_reference, "reference_training",
                        lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(vpar, "PoolWitness", _StubWitness)
    monkeypatch.setattr(vpar, "par_devices", lambda devices: contextlib.nullcontext())
    _recorded(monkeypatch, one)
    monkeypatch.setattr(vpar, "run_cell", lambda *a, **k: two)
    return vpar.par_check(_StubHarness(), None, list(lanes), ["base"], devices=(0, 1))


_REFUSED = _cell(train=(None, "RuntimeError: GPU worker failed"),
                 infer=(None, "RuntimeError: GPU worker failed"),
                 model=(None, "RuntimeError: GPU worker failed"),
                 batch=(None, "RuntimeError: GPU worker failed"),
                 stepfull=(None, "RuntimeError: GPU worker failed"))


def test_every_part_refused_on_both_columns_is_not_verified(monkeypatch):
    """THE INPUT THAT USED TO PASS. Both columns refuse every part, the same
    way. `compare_parts` reads REFUSED, which is correctly not a DIVERGENT --
    and the run used to end VERIFIED at exit 0 on the strength of it."""
    r = _par_check(monkeypatch, _REFUSED, _REFUSED)
    assert r["counts"].get("REFUSED"), r["counts"]
    assert not r["counts"].get("IDENTICAL")
    assert r["state"] == "INCOMPLETE"
    assert r["passed"] is not True
    assert "INCOMPLETE" in vpar.format_par_check(r)
    assert "AGREE" not in vpar.format_par_check(r)


def test_one_refused_part_beside_real_agreement_is_still_incomplete(monkeypatch):
    """A part that did not run is not made up for by the parts that did.
    `_verify_all.verdict` has refused that trade since 2026-09-16."""
    one = _cell(train="aaaa000011112222", infer=(None, "RuntimeError: refused"))
    two = _cell(train="aaaa000011112222", infer=(None, "RuntimeError: refused"))
    r = _par_check(monkeypatch, one, two)
    assert r["counts"].get("IDENTICAL") and r["counts"].get("REFUSED")
    assert r["state"] == "INCOMPLETE" and r["passed"] is not True


def test_a_column_of_nothing_but_declared_limits_compares_nothing(monkeypatch):
    """CPU-ROUTE-LIMIT is named rather than hidden and does not gate, but it
    is not evidence either: a run with no IDENTICAL part compared nothing."""
    refusal = "no CPU implementation of the cooperative multi-GPU driver mlp_update across 2 devices yet"
    one = _cell(train="6e5cde4953362927")
    two = _cell(train=(None, refusal))
    import contextlib
    from mojolearn import _backend, _cpu_reference
    _need_numpy()
    monkeypatch.setattr(_backend, "vendor", lambda: "cpu")
    monkeypatch.setattr(_cpu_reference, "reference_training",
                        lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(vpar, "PoolWitness", _StubWitness)
    monkeypatch.setattr(vpar, "par_devices", lambda devices: contextlib.nullcontext())
    _recorded(monkeypatch, one)
    monkeypatch.setattr(vpar, "run_cell", lambda *a, **k: two)
    r = vpar.par_check(_StubHarness(), None, ["par-mlp"], ["base"], devices=(0, 1))
    assert r["counts"].get("CPU-ROUTE-LIMIT") and not r["counts"].get("IDENTICAL")
    assert r["state"] == "NOTHING COMPARED" and r["passed"] is None


def test_a_real_agreement_still_verifies(monkeypatch):
    """The control. Without it the three tests above prove only that the
    command now refuses everything."""
    same = _cell(train="aaaa000011112222", infer="bbbb000011112222")
    r = _par_check(monkeypatch, same, same)
    assert r["counts"].get("IDENTICAL") == 2
    assert r["state"] == "VERIFIED" and r["passed"] is True
    assert "AGREE" in vpar.format_par_check(r)


def test_a_divergence_still_outranks_a_refusal(monkeypatch):
    """A wrong answer outranks an absent one, so MISMATCH is read first."""
    one = _cell(train="aaaa000011112222", infer=(None, "RuntimeError: refused"))
    two = _cell(train="ffff999988887777", infer=(None, "RuntimeError: refused"))
    r = _par_check(monkeypatch, one, two)
    assert r["state"] == "MISMATCH" and r["passed"] is False


def test_the_state_ladder_reads_worst_first():
    """`par_state` on its own, so the order is stated and not inferred."""
    assert vpar.par_state({}, []) == ("NOTHING COMPARED", None)
    assert vpar.par_state({"DIVERGENT": 1, "REFUSED": 9, "IDENTICAL": 9}, [1])[0] == "MISMATCH"
    assert vpar.par_state({"REFUSED": 1, "IDENTICAL": 9}, [1])[0] == "INCOMPLETE"
    assert vpar.par_state({"N/A": 9}, [1])[0] == "NOTHING COMPARED"
    assert vpar.par_state({"IDENTICAL": 1}, [1]) == ("VERIFIED", True)
    assert [s for s in vpar.PAR_STATES] == ["MISMATCH", "INCOMPLETE", "NOTHING COMPARED", "VERIFIED"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))


@pytest.mark.parametrize('axis', ['lane', 'fixture'])
@pytest.mark.parametrize('missing_index', [0, 1])
def test_each_parallel_cell_needs_its_own_device_witness(monkeypatch, axis, missing_index):
    """A valid neighboring cell must not certify a silent single-device fallback."""
    import contextlib
    from mojolearn import _backend, _cpu_reference
    _need_numpy()
    active = []

    class Witness(vpar.PoolWitness):
        @contextlib.contextmanager
        def watching(self):
            active.append(self)
            try:
                yield
            finally:
                active.pop()

    monkeypatch.setattr(_backend, 'vendor', lambda: 'hip')
    monkeypatch.setattr(_cpu_reference, 'reference_training',
                        lambda: contextlib.nullcontext())
    monkeypatch.setattr(vpar, 'PoolWitness', Witness)
    monkeypatch.setattr(vpar, 'par_devices', lambda devices: contextlib.nullcontext())
    two_calls = 0

    def run_cell(*args, **kwargs):
        nonlocal two_calls
        if active:
            if two_calls != missing_index:
                active[-1].pools.append(dict(devices=[0, 1], workers=[
                    dict(pid=101, devices=['GPU-a']), dict(pid=102, devices=['GPU-b'])]))
            two_calls += 1
        return _cell(train='aaaa000011112222')

    monkeypatch.setattr(vpar, 'run_cell', run_cell)
    _recorded(monkeypatch, _cell(train='aaaa000011112222'))
    lanes = ['par-gram', 'par-covariance'] if axis == 'lane' else ['par-gram']
    fixtures = ['base'] if axis == 'lane' else ['base', 'wide']
    result = vpar.par_check(_StubHarness(), None, lanes, fixtures)
    assert result['counts']['IDENTICAL'] == 2
    assert result['witness']['pools'] == 1
    assert result['state'] == 'INCOMPLETE' and result['passed'] is None
    assert len(result['cell_witnesses']) == 2
    missing = result['cell_witnesses'][missing_index]
    assert missing['witness']['pools'] == 0
    assert 'NO device pool' in missing['witness_refusal']
    assert result['witness_refusal'].startswith(missing['lane'] + '/' + missing['fixture'])
    assert result['cell_witnesses'][1 - missing_index]['witness_refusal'] is None
    assert 'CANNOT RUN' in vpar.format_par_check(result)


@pytest.mark.parametrize('scope', ['quick', 'default', 'all'])
@pytest.mark.parametrize('override', [None, 'odd,wide'])
def test_par_cli_fixture_scope_and_explicit_override(monkeypatch, scope, override):
    """The all mode must exercise non-base fixtures without needing another flag."""
    from mojolearn import _backend
    _need_numpy()
    harness = va.load_harness()
    selected = []
    monkeypatch.setattr(_backend, 'vendor', lambda: 'cuda')
    monkeypatch.setattr(vpar, 'refuse_to_run', lambda *args: None)
    monkeypatch.setattr(vpar, 'load_harness', lambda **kwargs: harness)

    def capture(harness, ml, lanes, fixtures, **kwargs):
        selected.extend(fixtures)
        return dict(cells=[dict(lane=lanes[0], fixture=fixtures[0])],
                    witness_refusal=None, state='VERIFIED')

    monkeypatch.setattr(vpar, 'par_check', capture)
    monkeypatch.setattr(vpar, '_emit', lambda *args: None)
    argv = ['verify', '--par', scope, '--json']
    if override:
        argv += ['--fixtures', override]
    args = cli.build_parser().parse_args(argv)
    assert vpar.cmd_par_check(args, None) == vpar.EXIT_VERIFIED
    expected = override.split(',') if override else list(harness.FIXTURES) if scope == 'all' else ['base']
    assert selected == expected
    assert len(harness.FIXTURES) == 9

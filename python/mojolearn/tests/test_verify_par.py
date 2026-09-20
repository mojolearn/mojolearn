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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))

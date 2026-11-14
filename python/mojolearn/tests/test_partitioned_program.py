# SPDX-License-Identifier: Apache-2.0
"""Independent correctness and physical ownership evidence for native partitions.

These are test controllers, not product runtime. Multi-device qualification
requires MOJOLEARN_DISTRIBUTED_PROGRAM_DEVICES to name distinct physical GPUs.
The single-device column-partition SGD and residency, pipeline forward and
dense gradient, shared-parameter alias and Adam-moment migration checks run
natively in checks/native_tests/core_partitioned_program_*.mojo.
"""
import os
import numpy as np
import pytest
from mojolearn import _backend, ops
from mojolearn._verify_native_runtime import audit_native_call


@pytest.fixture(autouse=True)
def require_native_partition():
    b = _backend.binding("_mojolearn_x_sequence", "identical")
    names = ("open", "load", "fit", "run", "state", "residency", "save", "close", "profile")
    if not all(hasattr(b, "ops_partitioned_" + name) for name in names):
        if os.environ.get("MOJOLEARN_REQUIRE_PARTITIONED_PROGRAM") == "1":
            pytest.fail("installed sequence binding lacks native partitioned-program entries")
        pytest.skip("native partitioned-program binding has not been installed")


def _devices():
    return tuple(map(int, os.environ.get("MOJOLEARN_DISTRIBUTED_PROGRAM_DEVICES", "0").split(",")))


def _linear(s):
    x = s.tensor(np.zeros((3, 1), np.float32))
    w = s.tensor(np.zeros((1, 4), np.float32))
    return s.capture([x], [s.matmul(x, w)])


def _network(s, dropout=False):
    x = s.tensor(np.zeros((3, 2), np.float32))
    w = s.tensor(np.arange(8, dtype=np.float32).reshape(2, 4) / 32)
    b = s.tensor(np.full((1, 4), .125, np.float32))
    v = s.tensor(np.arange(8, dtype=np.float32).reshape(4, 2) / 64)
    hidden = s.tanh(s.linear(x, w, b))
    if dropout:
        hidden = s.dropout(hidden, p=.25, seed=17)
    return s.capture([x], [s.matmul(hidden, v)])


def _same(a, b):
    for key in ("epoch", "step", "seed", "logical_stages", "microbatches", "tensor_shards"):
        assert a[key] == b[key]
    for key in ("parameters", "first_moments", "second_moments"):
        for left, right in zip(a[key], b[key], strict=True):
            np.testing.assert_array_equal(left.view(np.uint32), right.view(np.uint32))


def test_complete_partition_checkpoint_and_single_device_resume(tmp_path):
    rng = np.random.default_rng(26)
    x = rng.normal(size=(11, 2)).astype(np.float32)
    y = rng.normal(size=(11, 2)).astype(np.float32)
    path = str(tmp_path / "partition.ckpt")
    with ops.Session(numeric_mode="identical") as s:
        p = _network(s, dropout=True)
        opt = s.optimizer("adamw", lr=.01, weight_decay=.02,
                          schedule="step", step_size=2, gamma=.5)
        with p.partition(_devices(), stages=max(3, len(_devices())), tensor_shards=2,
                         microbatches=3, optimizer=opt, seed=41) as d:
            d.fit(x, targets=y, epochs=2, batch_size=4)
            before = d.state()
            audit_native_call(d.save, path).assert_native()
            d.fit(x, targets=y, epochs=3, batch_size=4)
            after = d.state()
        with s.load_partitioned(path, devices=(0,)) as replay:
            _same(before, replay.state())
            assert replay.residency()["partitioned_products"] == 2
            replay.fit(x, targets=y, epochs=3, batch_size=4)
            _same(after, replay.state())


def test_physical_pipeline_events_and_unique_weight_owners():
    devices = _devices()
    if len(devices) < 2:
        pytest.skip("physical pipeline overlap requires at least two distinct GPUs")
    rng = np.random.default_rng(18)
    x = rng.normal(size=(48, 32)).astype(np.float32)
    y = rng.normal(size=(48, 32)).astype(np.float32)
    with ops.Session(numeric_mode="identical") as s:
        a = s.tensor(np.zeros((3, 32), np.float32))
        w = s.tensor(rng.normal(0, .02, size=(32, 64)).astype(np.float32))
        v = s.tensor(rng.normal(0, .02, size=(64, 32)).astype(np.float32))
        p = s.capture([a], [s.matmul(s.tanh(s.matmul(a, w)), v)])
        opt = s.optimizer("sgd", lr=.01)
        with p.partition(devices, stages=len(devices), tensor_shards=1,
                         microbatches=4, optimizer=opt) as d:
            d.fit(x, targets=y, epochs=1, batch_size=48)
            r = d.residency()
            assert sum(r["parameter_elements_per_rank"]) == 4096
            assert max(r["parameter_elements_per_rank"]) < 4096
            events = d.profile()
            assert set(e["device"] for e in events) == set(devices)
            assert all(e["end_ns"] >= e["start_ns"] for e in events)
            overlaps = [(a,b) for a in events for b in events
                        if a["device"] != b["device"] and a["microbatch"] != b["microbatch"]
                        and max(a["start_ns"], b["start_ns"]) < min(a["end_ns"], b["end_ns"])]
            assert overlaps, "no observed overlap of native operations on distinct physical devices"


def test_partition_constructor_admission_and_checkpoint_integrity(tmp_path):
    with ops.Session(numeric_mode="identical") as s:
        p = _linear(s)
        opt = s.optimizer("sgd", lr=.125)
        for kwargs in ({"stages": 0}, {"stages": 1, "tensor_shards": 2},
                       {"microbatches": 0}, {"seed": -1}, {"seed": 1 << 63},
                       {"parameters": (99999999,)}, {"devices": (0, 0)}):
            with pytest.raises(Exception):
                p.partition(optimizer=opt, **kwargs)
        path = str(tmp_path / "bad.ckpt")
        with p.partition((0,), stages=2, tensor_shards=2, optimizer=opt) as d:
            with pytest.raises(Exception, match="sample count|targets"):
                d.fit(np.ones((3, 1), np.float32), targets=np.ones((2, 4), np.float32))
            assert d.state()["step"] == 0
            d.save(path)
        from pathlib import Path
        data = bytearray(Path(path).read_bytes())
        data[len(data)//2] ^= 1
        Path(path).write_bytes(data)
        with pytest.raises(Exception):
            s.load_partitioned(path)

# SPDX-License-Identifier: Apache-2.0
"""Independent correctness and physical ownership evidence for native partitions.

These are test controllers, not product runtime. Multi-device qualification
requires MOJOLEARN_DISTRIBUTED_PROGRAM_DEVICES to name distinct physical GPUs.
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


def test_column_partition_independent_sgd_and_residency():
    # Four output columns: d MSE/dw_j = -4/4 = -1 at w=0, X=1,y=2.
    x = np.ones((5, 1), np.float32)
    y = np.full((5, 4), 2, np.float32)
    devices = _devices()
    with ops.Session(numeric_mode="identical") as s:
        p = _linear(s)
        opt = s.optimizer("sgd", lr=.125)
        with p.partition(devices, stages=max(2, len(devices)), tensor_shards=2,
                         microbatches=3, optimizer=opt) as d:
            loss = audit_native_call(d.fit, x, targets=y, epochs=1, batch_size=5,
                                     shuffle=False).assert_native()
            state = audit_native_call(d.state).assert_native()
            assert len(state["parameters"]) == 2
            for weight in state["parameters"]:
                np.testing.assert_allclose(weight, .125, rtol=0, atol=2e-8)
            np.testing.assert_array_equal(loss, [4])
            np.testing.assert_allclose(audit_native_call(d.run, x).assert_native(), .125,
                                       rtol=0, atol=2e-8)
            r = d.residency()
            assert r["partitioned_products"] == 1
            assert sum(r["parameter_elements_per_rank"]) == 4
            assert len(r["parameter_owner_devices"]) == 2
            if len(devices) > 1:
                assert len(set(r["parameter_owner_devices"])) == 2
                assert max(r["parameter_elements_per_rank"]) < 4


def test_pipeline_forward_independent_numpy_and_dense_parameter_gradient():
    x = np.array([[.5, -.25], [.125, .75], [-.5, 1]], np.float32)
    w = np.arange(8, dtype=np.float32).reshape(2, 4) / 32
    b = np.full((1, 4), .125, np.float32)
    v = np.arange(8, dtype=np.float32).reshape(4, 2) / 64
    y = np.full((3, 2), .25, np.float32)
    h = np.tanh(x @ w + b)
    pred = h @ v
    grad_out = np.float32(2 / pred.size) * (pred - y)
    gv = h.T @ grad_out
    gh = (grad_out @ v.T) * (1 - h*h)
    gw = x.T @ gh
    gb = gh.sum(axis=0, keepdims=True)
    with ops.Session(numeric_mode="identical") as s:
        p = _network(s)
        opt = s.optimizer("sgd", lr=.0625)
        with p.partition(_devices(), stages=max(2, len(_devices())), tensor_shards=1,
                         microbatches=1, optimizer=opt) as d:
            np.testing.assert_allclose(d.run(x), pred, rtol=2e-6, atol=2e-7)
            d.fit(x, targets=y, epochs=1, batch_size=3, shuffle=False)
            expected = [w-.0625*gw, b-.0625*gb, v-.0625*gv]
            for actual, want in zip(d.state()["parameters"], expected, strict=True):
                np.testing.assert_allclose(actual, want, rtol=2e-6, atol=2e-7)


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


def test_shared_parameter_alias_gradient_is_not_lost():
    # Shared weights stay one whole owner. y=2*x*w, MSE to2 at w=0:
    # gradient=-8; SGD(lr=1/16) updates w to .5.
    x = np.ones((4, 1), np.float32)
    y = np.full((4, 1), 2, np.float32)
    with ops.Session(numeric_mode="identical") as s:
        a = s.tensor(np.zeros((3, 1), np.float32))
        w = s.tensor(np.zeros((1, 1), np.float32))
        h = s.matmul(a, w)
        p = s.capture([a], [s.add(h, h)])
        opt = s.optimizer("sgd", lr=.0625)
        with p.partition(_devices(), stages=max(2, len(_devices())), tensor_shards=2,
                         microbatches=2, optimizer=opt) as d:
            d.fit(x, targets=y, epochs=1, batch_size=4, shuffle=False)
            np.testing.assert_array_equal(d.state()["parameters"][0], [[.5]])
            np.testing.assert_array_equal(d.run(x), np.ones((4, 1), np.float32))


def test_source_adam_moments_are_column_partitioned_before_resume():
    x = np.ones((8, 1), np.float32)
    y = np.full((8, 4), 2, np.float32)
    with ops.Session(numeric_mode="identical") as s:
        p = _linear(s)
        opt = s.optimizer("adam", lr=.01)
        p.fit(x, targets=y, optimizer=opt, epochs=3, batch_size=4, shuffle=False)
        with p.partition((0,), stages=2, tensor_shards=2, microbatches=1, optimizer=opt) as d:
            st = d.state()
            assert (st["epoch"], st["step"]) == (3, 6)
            assert any(np.any(a != 0) for a in st["first_moments"])
            d.fit(x, targets=y, epochs=1, batch_size=4, shuffle=False)
            actual = np.concatenate(d.state()["parameters"], axis=1)
        p.fit(x, targets=y, optimizer=opt, epochs=1, batch_size=4, shuffle=False)
        np.testing.assert_array_equal(actual.view(np.uint32), s.numpy(p.params[0]).view(np.uint32))


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

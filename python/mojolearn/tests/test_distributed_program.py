# SPDX-License-Identifier: Apache-2.0
"""Native user-composed graphs: independent gradient oracle, topology replay,
physical optimizer residency, complete checkpoint/restart, failure behavior.

Python here is an independent test/data controller. Product fit/run/save/load
are single binding calls. Set MOJOLEARN_REQUIRE_DISTRIBUTED_PROGRAM=1 to fail
instead of skipping an older installed wheel. Physical multi-GPU coverage
requires MOJOLEARN_DISTRIBUTED_PROGRAM_DEVICES=0,1 (or more real device IDs).
"""
import hashlib
import os

import numpy as np
import pytest

from mojolearn import _backend, ops


def _available():
    try:
        b = _backend.binding("_mojolearn_x_sequence", "identical")
        return all(hasattr(b, f"ops_distributed_{name}") for name in
                   ("open", "load", "fit", "run", "state", "residency", "save", "close"))
    except Exception:
        return False


@pytest.fixture(autouse=True)
def require_native_distributed():
    if not _available():
        if os.environ.get("MOJOLEARN_REQUIRE_DISTRIBUTED_PROGRAM") == "1":
            pytest.fail("installed sequence binding lacks native distributed-program entries")
        pytest.skip("native distributed-program binding has not been installed")


def _devices():
    return tuple(int(v) for v in os.environ.get("MOJOLEARN_DISTRIBUTED_PROGRAM_DEVICES", "0").split(","))


def _state_words(state):
    return [np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
            for name in ("parameters", "first_moments", "second_moments") for a in state[name]]


def _assert_same(a, b):
    assert (a["epoch"], a["step"], a["seed"], a["logical_shards"]) == (
        b["epoch"], b["step"], b["seed"], b["logical_shards"])
    for left, right in zip(_state_words(a), _state_words(b), strict=True):
        np.testing.assert_array_equal(left, right)


def _emit(name, state):
    h = hashlib.sha256()
    for words in _state_words(state):
        h.update(words.tobytes())
    h.update(str((state["epoch"], state["step"], state["seed"], state["logical_shards"])).encode())
    print(f"DISTRIBUTED_PROGRAM_DIGEST {name} {h.hexdigest()}")


def _linear(s, width=1, outputs=1):
    x = s.tensor(np.zeros((3, width), np.float32))
    w = s.tensor(np.zeros((width, outputs), np.float32))
    out = s.matmul(x, w)
    return s.capture([x], [out]), w


def _nonlinear(s, *, dropout=False):
    rng = np.random.default_rng(12)
    x = s.tensor(np.zeros((3, 3), np.float32))
    w = s.tensor(rng.normal(0, 0.1, size=(3, 5)).astype(np.float32))
    b = s.tensor(np.zeros((1, 5), np.float32))
    w2 = s.tensor(rng.normal(0, 0.1, size=(5, 2)).astype(np.float32))
    h = s.tanh(s.linear(x, w, b))
    if dropout:
        h = s.dropout(h, p=0.25, seed=17)
    return s.capture([x], [s.matmul(h, w2)])


@pytest.mark.parametrize("logical_shards", [1, 3])
def test_independent_sgd_gradient_and_replica_snapshot(logical_shards):
    # Analytic MSE: w=0, X=1, y=2 => gradient=-4; lr=1/8 => w=1/2.
    x = np.ones((5, 1), np.float32)
    y = np.full((5, 1), 2, np.float32)
    with ops.Session(numeric_mode="identical") as s:
        p, w = _linear(s)
        o = s.optimizer("sgd", lr=0.125)
        with p.distribute(_devices(), logical_shards=max(logical_shards, len(_devices())), optimizer=o) as d:
            losses = d.fit(x, targets=y, epochs=1, batch_size=5, shuffle=False)
            st = d.state()
            np.testing.assert_array_equal(st["parameters"][0], [[0.5]])
            np.testing.assert_array_equal(d.run(x), np.full((5, 1), 0.5, np.float32))
            np.testing.assert_array_equal(losses, np.array([4], np.float32))
            assert st["step"] == 1 and st["epoch"] == 1
            # The independently owned source is never mutated by distributed fit.
            np.testing.assert_array_equal(s.numpy(w), [[0]])
            _emit("sgd_analytic", st)


def test_adam_checkpoint_resume_includes_dropout_rng_schedule_and_uneven_batches(tmp_path):
    rng = np.random.default_rng(23)
    x = rng.normal(size=(17, 3)).astype(np.float32)
    y = rng.normal(size=(17, 2)).astype(np.float32)
    path = str(tmp_path / "distributed.ckpt")
    with ops.Session(numeric_mode="identical") as s:
        p = _nonlinear(s, dropout=True)
        o = s.optimizer("adam", lr=0.01, schedule="step", step_size=2, gamma=0.5)
        with p.distribute(_devices(), logical_shards=max(4, len(_devices())), optimizer=o, seed=91) as d:
            d.fit(x, targets=y, epochs=2, batch_size=6)
            d.save(path)
            first = d.state()
            # Nine more updates cross t=7; exponent/counter restoration matters.
            d.fit(x, targets=y, epochs=3, batch_size=6)
            uninterrupted = d.state()
        with s.load_distributed(path, devices=(0,)) as replay:
            _assert_same(first, replay.state())
            replay.fit(x, targets=y, epochs=3, batch_size=6)
            _assert_same(uninterrupted, replay.state())
            _emit("adam_resume", replay.state())


def test_physical_topology_replay_and_optimizer_range_witness(tmp_path):
    devices = _devices()
    if len(devices) < 2:
        pytest.skip("requires two distinct physical GPU IDs; one-device replay is not multi-GPU evidence")
    rng = np.random.default_rng(24)
    x = rng.normal(size=(19, 3)).astype(np.float32)
    y = rng.normal(size=(19, 2)).astype(np.float32)
    with ops.Session(numeric_mode="identical") as s:
        p = _nonlinear(s)
        o = s.optimizer("adamw", lr=0.01, weight_decay=0.02)
        logical = max(5, len(devices))
        with p.distribute(devices, logical_shards=logical, optimizer=o, seed=37) as multi:
            r = multi.residency()
            assert tuple(r["devices"]) == devices
            assert r["parameter_residency"] == "replicated"
            assert all(n > 0 for n in r["optimizer_elements_per_rank"])
            assert sum(r["optimizer_elements_per_rank"]) == 30  # 3*5 + 5 + 5*2
            for j, n in enumerate((15, 5, 10)):
                intervals = [row[j] for row in r["optimizer_ranges"]]
                assert intervals[0][0] == 0
                assert all(intervals[k][0] + intervals[k][1] == intervals[k + 1][0]
                           for k in range(len(intervals) - 1))
                assert intervals[-1][0] + intervals[-1][1] == n
            multi.fit(x, targets=y, epochs=3, batch_size=7)
            actual = multi.state()
            path = str(tmp_path / "multi.ckpt")
            multi.save(path)
        with p.distribute((0,), logical_shards=logical, optimizer=o, seed=37) as single:
            single.fit(x, targets=y, epochs=3, batch_size=7)
            _assert_same(actual, single.state())
        with s.load_distributed(path, devices=(0,)) as restored:
            _assert_same(actual, restored.state())
        _emit("physical_topology", actual)


def test_optimizer_state_migrates_from_source():
    x = np.ones((8, 1), np.float32)
    y = np.full((8, 1), 2, np.float32)
    with ops.Session(numeric_mode="identical") as s:
        p, w = _linear(s)
        o = s.optimizer("adam", lr=0.01)
        p.fit(x, targets=y, optimizer=o, epochs=3, batch_size=4, shuffle=False)
        before = s.numpy(w)
        with p.distribute((0,), logical_shards=1, optimizer=o) as d:
            st = d.state()
            assert st["step"] == 6 and st["epoch"] == 3
            np.testing.assert_array_equal(st["parameters"][0], before)
            d.fit(x, targets=y, epochs=1, batch_size=4, shuffle=False)
            actual = d.state()["parameters"][0]
        p.fit(x, targets=y, optimizer=o, epochs=1, batch_size=4, shuffle=False)
        np.testing.assert_array_equal(actual.view(np.uint32), s.numpy(w).view(np.uint32))


def test_distributed_refusals_and_damaged_checkpoint(tmp_path):
    with ops.Session(numeric_mode="identical") as s:
        p, _ = _linear(s)
        o = s.optimizer("sgd", lr=0.1)
        with pytest.raises(Exception, match="duplicate"):
            p.distribute((0, 0), optimizer=o)
        with pytest.raises(Exception, match="logical_shards"):
            p.distribute((0,), logical_shards=0, optimizer=o)
        with p.distribute((0,), optimizer=o) as d:
            with pytest.raises(Exception, match="sample count"):
                d.fit(np.ones((3, 1), np.float32), targets=np.ones((2, 1), np.float32))
            # Admission errors happen before mutation and do not poison state.
            assert d.state()["step"] == 0
            path = tmp_path / "damaged.ckpt"
            d.save(str(path))
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 1
        path.write_bytes(data)
        with pytest.raises(Exception):
            s.load_distributed(str(path))

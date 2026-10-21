# SPDX-License-Identifier: Apache-2.0
"""mojolearn.cross_vendor on the host: the fold, the framing and the
coordinator's refusals, with independent test-only socket peers (no GPU)."""
import array
import hashlib
import math
import json
import socket
import struct
import threading

import pytest

from mojolearn import cross_vendor as cv


def _f(values):
    return array.array("f", values).tobytes()


def test_fold_is_a_left_fold_with_one_rounding_per_add():
    # 1 + 2**-24 rounds back to 1 in float32; 2**-24 + 2**-24 does not.
    a, b, c = _f([1.0]), _f([2.0 ** -24]), _f([2.0 ** -24])
    assert array.array("f", cv.ordered_fold([a, b, c]))[0] == 1.0
    assert array.array("f", cv.ordered_fold([b, c, a]))[0] == 1.0 + 2.0 ** -23


def test_fold_flushes_subnormals_keeping_sign():
    sub = 1e-39
    # ftz(-sub) is -0, and -0 + -0 is -0; (-0) + (+0) would be +0, as on the device.
    out = array.array("f", cv.ordered_fold([_f([sub, -sub, 3.0]), _f([0.0, -0.0, -3.0])]))
    assert out[0] == 0.0 and math.copysign(1.0, out[0]) == 1.0
    assert out[1] == 0.0 and math.copysign(1.0, out[1]) == -1.0
    assert out[2] == 0.0


def test_fold_equals_an_independent_numpy_oracle_bit_for_bit():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(7)
    grads = [(rng.standard_normal(4096) * 10.0 ** rng.integers(-40, 3, 4096)).astype(np.float32)
             for _ in range(5)]

    def ftz(x):
        b = x.view(np.uint32)
        sub = ((b & 0x7F800000) == 0) & ((b & 0x007FFFFF) != 0)
        return np.where(sub, b & np.uint32(0x80000000), b).view(np.float32)

    want = grads[0].copy()
    for g in grads[1:]:
        want = ftz(ftz(want) + ftz(g))
    assert cv.ordered_fold([g.tobytes() for g in grads]) == want.tobytes()


def test_frame_round_trip_and_oversize_refusal():
    import socket
    a, b = socket.socketpair()
    try:
        cv._send(a, {"cmd": "apply", "step": 3}, b"\x00" * 16)
        head, payload = cv._recv(b, 16)
        assert head == {"cmd": "apply", "step": 3, "nbytes": 16} and payload == b"\x00" * 16
        cv._send(a, {"cmd": "apply"}, b"\x00" * 32)
        with pytest.raises(ConnectionError):
            cv._recv(b, 16)
    finally:
        a.close()
        b.close()


class _Fake:
    """A trainer whose 'gradient' of shard k is a fixed vector and whose
    update is param -= total; state is param only (m, v, flags empty)."""

    def __init__(self, n=8, drift_at=None):
        self.param = array.array("f", [0.5] * n)
        self.step_ = 0
        self.drift_at = drift_at

    def state_dict(self):
        return dict(parameters=self.param, m=array.array("f"), v=array.array("f"), flags=array.array("i"))

    def shard_gradient(self, ids):
        k = ids
        return float(k), array.array("f", [(k + 1) * 0.25 + i for i in range(len(self.param))])

    def apply_gradient(self, total):
        for i in range(len(self.param)):
            self.param[i] -= total[i]
        self.step_ += 1
        if self.drift_at == self.step_:
            self.param[0] += 1.0


def _wire(sock, header, payload=b""):
    raw = json.dumps(dict(header, nbytes=len(payload))).encode()
    sock.sendall(struct.pack(">I", len(raw)) + raw + payload)


def _read_wire(sock):
    def exact(count):
        data = b""
        while len(data) < count:
            part = sock.recv(count - len(data))
            if not part:
                raise ConnectionError("test peer connection closed")
            data += part
        return data
    header = json.loads(exact(struct.unpack(">I", exact(4))[0]))
    return header, exact(header["nbytes"])


def _oracle_peer(port, shards, name, chained, drift):
    # Python is deliberately an independent test peer, never passed to Worker.
    trainer = _Fake(drift_at=drift)
    def digest():
        state = trainer.state_dict()
        return hashlib.sha256(b"".join(memoryview(state[k]).cast("B")
                                      for k in ("parameters", "m", "v", "flags"))).hexdigest()
    with socket.create_connection(("127.0.0.1", port), timeout=30) as sock:
        _wire(sock, dict(protocol=cv.PROTOCOL, name=name, vendor="oracle",
                         shards=shards, completed=0, state=digest(),
                         n_total=len(trainer.param), chained=chained))
        held, total = [], None
        while True:
            head, payload = _read_wire(sock)
            if head["cmd"] == "step":
                outputs = [trainer.shard_gradient(k) for k in shards]
                held = [g.tobytes() for _, g in outputs]
                _wire(sock, dict(step=head["step"], shards=shards,
                                 losses=[loss for loss, _ in outputs]),
                      b"" if chained else b"".join(held))
            elif head["cmd"] == "fold":
                parts = ([payload] if payload else []) + held
                values = array.array("f", parts[0])
                for part in parts[1:]:
                    other = array.array("f", part)
                    for index in range(len(values)):
                        values[index] = values[index] + other[index]
                total = values.tobytes()
                _wire(sock, dict(step=head["step"]), total)
            elif head["cmd"] == "apply":
                total = payload or total
                assert hashlib.sha256(total).hexdigest() == head["total_sha256"]
                trainer.apply_gradient(array.array("f", total))
                _wire(sock, dict(completed=trainer.step_, state=digest()))
            elif head["cmd"] == "done":
                return
            elif head["cmd"] == "refuse":
                raise cv.CrossVendorMismatch(head["reason"])
            else:
                raise AssertionError(head)


def _group(splits, steps=3, drift=None, chained=False):
    coord = cv.Coordinator(host="127.0.0.1", port=0, workers=len(splits), logical_shards=4, steps=steps,
                           accept_timeout=30, timeout=30, chained=chained)
    out, errors = {}, []
    def collect(call):
        try:
            call()
        except BaseException as error:
            errors.append(error)
    def run_coord():
        out["rows"] = coord.run()
    thread = threading.Thread(target=lambda: collect(run_coord))
    thread.start()
    assert coord.ready.wait(10)
    workers = [threading.Thread(target=lambda index=i, ids=ids: collect(
        lambda: _oracle_peer(coord.bound_port, ids, f"w{index}", chained,
                             drift if index == 1 else None))) for i, ids in enumerate(splits)]
    for worker in workers:
        worker.start()
    for worker in workers + [thread]:
        worker.join(timeout=40)
        assert not worker.is_alive()
    return out.get("rows"), errors


def _safe(w, errors):
    try:
        w.run()
    except BaseException as e:  # noqa: BLE001
        errors.append(e)


@pytest.fixture(autouse=True)
def _vendor(monkeypatch):
    import mojolearn
    monkeypatch.setattr(mojolearn, "vendor", lambda: "fake", raising=False)


def test_group_agrees_and_matches_one_process():
    rows, errors = _group([[0, 2], [1, 3]])
    assert not errors and [r["step"] for r in rows] == [1, 2, 3]
    one = _Fake()
    for _ in range(3):
        grads = [one.shard_gradient(k)[1].tobytes() for k in range(4)]
        one.apply_gradient(array.array("f", cv.ordered_fold(grads)))
    h = hashlib.sha256(memoryview(one.param).cast("B")).hexdigest()
    assert rows[-1]["state"] == h


def test_refuses_a_replica_that_drifts():
    rows, errors = _group([[0, 1], [2, 3]], drift=2)
    assert rows is None
    assert any("disagree after step 2" in str(e) for e in errors)


@pytest.mark.parametrize("splits", [[[0, 1], [1, 2, 3]], [[0], [2, 3]]])
def test_refuses_shards_not_owned_exactly_once(splits):
    rows, errors = _group(splits)
    assert rows is None and any("exactly once" in str(e) for e in errors)


# ------------------------------------------------------------ the chained fold

def test_ordered_fold_with_a_prefix_equals_the_flat_fold():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(5)
    grads = [(rng.standard_normal(512) * 10.0 ** rng.integers(-40, 3, 512)).astype(np.float32).tobytes()
             for _ in range(6)]
    flat = cv.ordered_fold(grads)
    prefix = cv.ordered_fold(grads[:2])
    assert cv.ordered_fold(grads[2:], prefix=prefix) == flat
    # and a prefix of one shard is that shard
    assert cv.ordered_fold(grads[1:], prefix=grads[0]) == flat
    # the reverse order is a different fold (the check can fail)
    assert cv.ordered_fold(list(reversed(grads))) != flat


def _chained_group(splits, steps=3, drift=None):
    return _group(splits, steps, drift, chained=True)


def test_chained_group_equals_the_gathered_group_and_one_process():
    chained, errors = _chained_group([[0, 1], [2, 3]])
    assert not errors and [r["step"] for r in chained] == [1, 2, 3]
    gathered, errors = _group([[0, 1], [2, 3]])
    assert not errors
    assert [r["state"] for r in chained] == [r["state"] for r in gathered]
    assert [r["total_sha256"] for r in chained] == [r["total_sha256"] for r in gathered]
    assert chained[0]["protocol"] == "chained" and gathered[0]["protocol"] == "gathered"
    one = _Fake()
    for _ in range(3):
        grads = [one.shard_gradient(k)[1].tobytes() for k in range(4)]
        one.apply_gradient(array.array("f", cv.ordered_fold(grads)))
    assert chained[-1]["state"] == hashlib.sha256(memoryview(one.param).cast("B")).hexdigest()


def test_chained_refuses_a_non_contiguous_block():
    rows, errors = _chained_group([[0, 2], [1, 3]])
    assert rows is None and any("contiguous" in str(e) for e in errors)


def test_chained_refuses_a_replica_that_drifts():
    rows, errors = _chained_group([[0, 1], [2, 3]], drift=2)
    assert rows is None and any("disagree after step 2" in str(e) for e in errors)


def test_chained_three_workers_unequal_blocks():
    rows, errors = _chained_group([[0], [1, 2], [3]])
    assert not errors and len(rows) == 3
    gathered, _ = _group([[0, 1], [2, 3]])
    assert [r["state"] for r in rows] == [r["state"] for r in gathered]

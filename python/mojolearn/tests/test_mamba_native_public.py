# SPDX-License-Identifier: Apache-2.0
"""The Mamba public operations that became one native call each.

`Mamba3Block` forward (plain, carried state and ragged), step, allocate_state,
the owned weight snapshot and pickling, `Mamba3State.set_input_states`, and the
Mamba-1/2/3 decode sessions. Each call is traced with `audit_native_call` (no
Python frame beyond the one-call wrapper) and its bytes are held to the
retained independent driver (`_call`) or to the per-call `step`.
"""
import os
import pickle

import numpy as np
import pytest

os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")

import mojolearn as ml  # noqa: E402
from mojolearn import Array, _ragged  # noqa: E402
from mojolearn._verify_native_runtime import audit_native_call  # noqa: E402
from mojolearn.tests.test_causal_model_native_loading import _poison  # noqa: E402


def _weights(shapes, ones=()):
    out = {}
    for i, (name, shape) in enumerate(sorted(shapes.items())):
        n = int(np.prod(shape))
        a = np.ones(n, np.float32) if name in ones else ((np.arange(n, dtype=np.float32) + 7 * i) % 11 - 5) / 32.0
        out[name] = np.ascontiguousarray(a.astype(np.float32).reshape(shape))
    return out


def _block(kind):
    dm, di, nh = 32, 64, 1
    try:
        if kind == "mamba1":
            r = 2
            return ml.Mamba1Block(_weights({
                "norm.weight": (dm,), "in_proj.weight": (2 * di, dm), "conv1d.weight": (di, 1, 4),
                "conv1d.bias": (di,), "x_proj.weight": (r + 32, di), "dt_proj.weight": (di, r),
                "dt_proj.bias": (di,), "A_log": (di, 16), "D": (di,), "out_proj.weight": (dm, di)},
                ones=("norm.weight",))), dm
        if kind == "mamba2":
            cd, dip = di + 256, 2 * di + 256 + nh
            return ml.Mamba2Block(_weights({
                "block_norm.weight": (dm,), "in_proj.weight": (dip, dm), "conv1d.weight": (cd, 1, 4),
                "conv1d.bias": (cd,), "dt_bias": (nh,), "A_log": (nh,), "D": (nh,), "norm.weight": (di,),
                "out_proj.weight": (dm, di)}, ones=("block_norm.weight", "norm.weight"))), dm
        dip = 2 * di + 256 + 3 * nh + 32
        return ml.Mamba3Block(_weights({
            "block_norm.weight": (dm,), "in_proj.weight": (dip, dm), "dt_bias": (nh,),
            "B_norm.weight": (128,), "C_norm.weight": (128,), "B_bias": (nh, 128), "C_bias": (nh, 128),
            "D": (nh,), "out_proj.weight": (dm, di)}, ones=("block_norm.weight",))), dm
    except ImportError as exc:
        pytest.skip(f"no mamba binding on this install: {exc}")


def _tokens(rows, length, dm):
    a = ((np.arange(rows * length * dm, dtype=np.float32)) % 13 - 6) / 64.0
    return np.ascontiguousarray(a.reshape(rows, length, dm))


def _native(monkeypatch, operation, *args, **kwargs):
    with monkeypatch.context() as patch:
        _poison(patch)
        return audit_native_call(operation, *args, **kwargs).assert_native()


def test_mamba3_forward_step_ragged_are_native_and_match_the_retained_driver(monkeypatch):
    block, dm = _block("mamba3")
    x = _tokens(2, 5, dm)
    expected = np.asarray(block._call(x, None, step=False)).copy()
    wanted_ragged, _ = _ragged.ragged_forward(lambda xp: block._call(xp, None, step=False),
                                              Array.from_buffer(x), None, [3, 5], "<f4", "test")
    wanted_ragged = np.asarray(wanted_ragged).copy()
    actual = _native(monkeypatch, block.forward, x)
    assert np.asarray(actual).tobytes() == expected.tobytes()
    assert block.h_last_ is not None
    ragged = _native(monkeypatch, block.forward, x, lengths=[3, 5])
    assert np.asarray(ragged).tobytes() == wanted_ragged.tobytes()
    state = _native(monkeypatch, block.allocate_state, 2)
    assert type(state) is ml.Mamba3State and state.buffered_tokens == 0 and state.pending is False
    prefix = _native(monkeypatch, block.forward, x[:, :4], state)
    tail = _native(monkeypatch, block.step, x[:, 4], state)
    joined = np.concatenate((np.asarray(prefix), np.asarray(tail)), axis=1)
    assert joined.tobytes() == expected.tobytes()
    with pytest.raises(ValueError, match="cannot carry a state"):
        block.forward(x, block.allocate_state(2), lengths=[3, 5])
    with pytest.raises(ValueError, match="state is required"):
        block.step(x[:, :1], None)


def test_mamba3_set_input_states_is_native_and_reaches_the_core(monkeypatch):
    block, dm = _block("mamba3")
    x = _tokens(1, 3, dm)
    plain = block.allocate_state(1)
    y0 = np.asarray(block.forward(x, plain)).copy()
    st = block.allocate_state(1)
    shapes = [np.asarray(getattr(st, n)).shape for n in ("theta", "h", "pending_k", "pending_v")]
    pieces = [np.full(s, 0.125, np.float32) for s in shapes]
    _native(monkeypatch, st.set_input_states, *pieces)
    assert st.pending is True
    assert np.asarray(st.pending_k).tobytes() == pieces[2].tobytes()
    y1 = np.asarray(block.forward(x, st)).copy()
    assert st.pending is False and y1.tobytes() != y0.tobytes()
    with pytest.raises(TypeError, match="theta"):
        st.set_input_states(pieces[0].astype(np.float64), *pieces[1:])


def test_mamba3_pickle_round_trip_is_native(monkeypatch):
    block, dm = _block("mamba3")
    x = _tokens(2, 4, dm)
    want = np.asarray(block.forward(x)).copy()
    state = _native(monkeypatch, block.__getstate__)
    assert "_prefill_session" not in state and state["_owned_generation"] is None
    clone = pickle.loads(pickle.dumps(block))
    assert np.asarray(clone.forward(x)).tobytes() == want.tobytes()


def test_mamba3_owned_weights_are_native(monkeypatch):
    block, dm = _block("mamba3")
    x = _tokens(2, 4, dm)
    want = np.asarray(block.forward(x)).copy()
    try:
        generation = _native(monkeypatch, block.install_owned_weights)
    except AssertionError as exc:
        if "NN40" in str(exc):
            pytest.skip("this binding has no owned-weight arm")
        raise
    assert generation is not None
    assert np.asarray(block.forward(x)).tobytes() == want.tobytes()
    exported = _native(monkeypatch, block.export_owned_weights)
    for name, w in zip(block._W_NAMES, block._w):
        assert np.asarray(exported[name]).tobytes() == np.asarray(w).tobytes(), name
    info = _native(monkeypatch, block.session_info)
    assert set(info) == {"weight_uploads", "weight_recopies", "weight_reuses",
                         "backward_reuses", "backward_recomputes", "stages_held"}
    clone = pickle.loads(pickle.dumps(block))
    assert np.asarray(clone.forward(x)).tobytes() == want.tobytes()
    _native(monkeypatch, block.release_owned_weights)
    assert block._owned_generation is None
    assert np.asarray(block.forward(x)).tobytes() == want.tobytes()


_STATE = {
    "mamba1": ("conv_window", "h"),
    "mamba2": ("conv_window", "h", "buffer_xbc", "buffer_dtraw"),
    "mamba3": ("theta", "h", "buffer_qrot", "buffer_krot", "buffer_v",
               "buffer_dt", "buffer_sig", "buffer_adt", "pending_k", "pending_v"),
}


@pytest.mark.parametrize("kind", ["mamba1", "mamba2", "mamba3"])
def test_decode_session_methods_are_native_and_equal_the_per_call_step(monkeypatch, kind):
    block, dm = _block(kind)
    rows, length = 2, 6
    x = _tokens(rows, length, dm)
    owned = block.allocate_state(rows)
    door = block.decode_session if kind == "mamba1" else block._decode_session
    try:
        session = _native(monkeypatch, door, owned)
    except AssertionError as exc:
        if "NotImplementedError" in str(exc):
            pytest.skip(f"no decode session on this install: {exc}")
        raise
    assert session.is_open
    with pytest.raises(ValueError, match="resident"):
        door(owned)
    with pytest.raises(ValueError, match="rows"):
        session.step(np.zeros((rows + 1, 1, dm), np.float32))
    out = [np.asarray(_native(monkeypatch, session.step, np.ascontiguousarray(x[:, t:t + 1]))).copy()
           for t in range(length)]
    _native(monkeypatch, session.sync_state)
    _native(monkeypatch, session.load_state)
    _native(monkeypatch, session.close)
    assert not session.is_open and getattr(owned, "_resident_session", None) is None
    with pytest.raises(ValueError, match="session is closed"):
        session.step(np.zeros((rows, 1, dm), np.float32))
    plain = block.allocate_state(rows)
    want = [np.asarray(block.step(np.ascontiguousarray(x[:, t:t + 1]), plain)).copy() for t in range(length)]
    assert np.concatenate(out, axis=1).tobytes() == np.concatenate(want, axis=1).tobytes()
    for name in _STATE[kind]:
        assert np.asarray(getattr(owned, name)).tobytes() == np.asarray(getattr(plain, name)).tobytes(), name
    if kind != "mamba1":
        assert owned.buffered_tokens == plain.buffered_tokens
    with pytest.raises(ValueError, match="state is required"):
        door(None)

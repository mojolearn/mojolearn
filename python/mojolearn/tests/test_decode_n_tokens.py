# SPDX-License-Identifier: Apache-2.0
"""`decode(x)` on every resident decode session against `N` session steps,
byte for byte (decode plan D2): the Mamba-1/2/3, Transformer and Samba
sessions. One `decode` call over N tokens must return the bytes N `step`
calls return, token by token, and leave the state (synced back) and the
reports where the steps leave them; a second `decode` after it continues the
same state. Holds on whichever arm the box has (the device session where the
binding exports one, the host arm where it does not). Not a lane. The Mamba
relation is the native check checks/native_tests/mamba_decode_n_tokens.mojo;
here the Mamba doors keep their shape, sync and refusal tests.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")

import mojolearn as ml  # noqa: E402


# Test fixtures (weights, Mamba-2/3 blocks, tokens, state bytes). They lived in
# test_mamba23_decode_sessions.py until 3fe4f407d6 moved its session-step test
# to checks/native_tests/mamba_decode_sessions.mojo and deleted them there;
# test_neural_repeat_calls.py and test_neural_coverage.py import them from here.
def _weights(shapes, ones=()):
    out = {}
    for i, (name, shape) in enumerate(sorted(shapes.items())):
        n = int(np.prod(shape))
        if name in ones:
            a = np.ones(n, dtype=np.float32)
        else:
            a = ((np.arange(n, dtype=np.float32) + 7 * i) % 11 - 5) / 32.0
        out[name] = np.ascontiguousarray(a.astype(np.float32).reshape(shape))
    return out


def _mamba2_block():
    dm, di, nh = 32, 64, 1
    cd, dip = di + 256, 2 * di + 256 + nh
    return ml.Mamba2Block(_weights({
        "block_norm.weight": (dm,), "in_proj.weight": (dip, dm), "conv1d.weight": (cd, 1, 4),
        "conv1d.bias": (cd,), "dt_bias": (nh,), "A_log": (nh,), "D": (nh,), "norm.weight": (di,),
        "out_proj.weight": (dm, di)}, ones=("block_norm.weight", "norm.weight"))), dm


def _mamba3_block():
    dm, di, nh = 32, 64, 1
    dip = 2 * di + 256 + 3 * nh + 32
    return ml.Mamba3Block(_weights({
        "block_norm.weight": (dm,), "in_proj.weight": (dip, dm), "dt_bias": (nh,),
        "B_norm.weight": (128,), "C_norm.weight": (128,), "B_bias": (nh, 128), "C_bias": (nh, 128),
        "D": (nh,), "out_proj.weight": (dm, di)}, ones=("block_norm.weight",))), dm


_STATE = {
    "mamba2": ("conv_window", "h", "buffer_xbc", "buffer_dtraw"),
    "mamba3": ("theta", "h", "buffer_qrot", "buffer_krot", "buffer_v",
               "buffer_dt", "buffer_sig", "buffer_adt", "pending_k", "pending_v"),
}


def _block(which):
    try:
        return (_mamba2_block if which == "mamba2" else _mamba3_block)()
    except ImportError as exc:
        pytest.skip(f"no mamba binding on this install: {exc}")


def _tokens(rows, length, dm):
    a = ((np.arange(rows * length * dm, dtype=np.float32)) % 13 - 6) / 64.0
    return np.ascontiguousarray(a.reshape(rows, length, dm))


def _state_bytes(which, state):
    parts = [np.ascontiguousarray(np.asarray(getattr(state, n))).reshape(-1).tobytes()
             for n in _STATE[which]]
    return parts, (int(state.buffered_tokens), getattr(state, "pending", None))


def _mamba1_block():
    dm, di, r = 32, 64, 2
    try:
        return ml.Mamba1Block(_weights({
            "norm.weight": (dm,), "in_proj.weight": (2 * di, dm), "conv1d.weight": (di, 1, 4),
            "conv1d.bias": (di,), "x_proj.weight": (r + 32, di), "dt_proj.weight": (di, r),
            "dt_proj.bias": (di,), "A_log": (di, 16), "D": (di,), "out_proj.weight": (dm, di)},
            ones=("norm.weight",))), dm
    except ImportError as exc:
        pytest.skip(f"no mamba binding on this install: {exc}")


def _mamba_block(which):
    return _mamba1_block() if which == "mamba1" else _block(which)


def _m1_state_bytes(state):
    return [np.asarray(state.conv_window).tobytes(), np.asarray(state.h).tobytes()]


# The Mamba-1/2/3 relation (one decode over N tokens, in two calls, equals N
# session steps: y, the reports, the state and the carried integers) is the
# native check checks/native_tests/mamba_decode_n_tokens.mojo (host session
# `MambaDecodeSessionRun.decode`, device program `MambaBlockProgram.decode`).
# The public door's returned shape, dtype and sync stay here.
@pytest.mark.parametrize("which", ["mamba1", "mamba2", "mamba3"])
def test_mamba_decode_returns_every_token(which):
    blk, dm = _mamba_block(which)
    x = _tokens(2, 10, dm)
    decoded = blk.allocate_state(2)
    zero = _m1_state_bytes(decoded) if which == "mamba1" else _state_bytes(which, decoded)
    with blk.decode_session(decoded) as sess:
        first = sess.decode(np.ascontiguousarray(x[:, :6]))
        second = sess.decode(np.ascontiguousarray(x[:, 6:]))
        assert np.asarray(first).shape == (2, 6, dm) and np.asarray(second).shape == (2, 4, dm)
        assert np.asarray(first).dtype == np.float32
        assert np.isfinite(np.asarray(first)).all() and np.isfinite(np.asarray(second)).all()
        sess.sync_state()
    synced = _m1_state_bytes(decoded) if which == "mamba1" else _state_bytes(which, decoded)
    assert synced != zero


@pytest.mark.parametrize("which", ["mamba1", "mamba2", "mamba3"])
def test_mamba_decode_refusals(which):
    blk, dm = _mamba_block(which)
    st = blk.allocate_state(2)
    sess = blk.decode_session(st)
    with pytest.raises(ValueError, match="rows"):
        sess.decode(_tokens(3, 4, dm))
    sess.close()
    with pytest.raises(ValueError, match="closed"):
        sess.decode(_tokens(2, 4, dm))


def _transformer_block():
    dm, nh, nkv, hd, it = 32, 2, 1, 16, 64
    try:
        return ml.TransformerBlock(_weights({
            "input_layernorm.weight": (dm,), "post_attention_layernorm.weight": (dm,),
            "q_proj.weight": (nh * hd, dm), "k_proj.weight": (nkv * hd, dm), "v_proj.weight": (nkv * hd, dm),
            "o_proj.weight": (dm, nh * hd), "gate_proj.weight": (it, dm), "up_proj.weight": (it, dm),
            "down_proj.weight": (dm, it)},
            ones=("input_layernorm.weight", "post_attention_layernorm.weight")),
            n_heads=nh, n_kv_heads=nkv), dm
    except ImportError as exc:
        pytest.skip(f"no transformer binding on this install: {exc}")


def test_transformer_decode_equals_the_steps():
    blk, dm = _transformer_block()
    x = _tokens(2, 10, dm)
    stepped = blk.allocate_state(2, max_tokens=16)
    with blk.decode_session(stepped) as sess:
        want = np.concatenate([np.asarray(sess.step(np.ascontiguousarray(x[:, t:t + 1])))
                               for t in range(10)], axis=1)
        sess.sync_state()
    decoded = blk.allocate_state(2, max_tokens=16)
    with blk.decode_session(decoded) as sess:
        got = np.concatenate([np.asarray(sess.decode(np.ascontiguousarray(x[:, :6]))),
                              np.asarray(sess.decode(np.ascontiguousarray(x[:, 6:])))], axis=1)
        assert int(decoded.cached_tokens) == 10
        sess.sync_state()
    assert got.tobytes() == want.tobytes()
    assert np.asarray(decoded.k_cache).tobytes() == np.asarray(stepped.k_cache).tobytes()
    assert np.asarray(decoded.v_cache).tobytes() == np.asarray(stepped.v_cache).tobytes()


def test_samba_decode_equals_the_steps():
    from mojolearn.tests.test_samba_decode_session import _ids, _stack
    from mojolearn.tests.test_samba_decode_session import _state_bytes as samba_state_bytes
    m = _stack()
    ids = _ids(2, 10)
    stepped = m.allocate_state(2, 16)
    with m.decode_session(stepped) as sess:
        want = np.stack([np.asarray(sess.step(np.ascontiguousarray(ids[:, t]))) for t in range(10)], axis=1)
        sess.sync_state()
    decoded = m.allocate_state(2, 16)
    with m.decode_session(decoded) as sess:
        got = np.concatenate([np.asarray(sess.decode(np.ascontiguousarray(ids[:, :6]))),
                              np.asarray(sess.decode(np.ascontiguousarray(ids[:, 6:])))], axis=1)
        sess.sync_state()
    assert got.shape == want.shape
    assert got.tobytes() == want.tobytes()
    assert samba_state_bytes(decoded) == samba_state_bytes(stepped)


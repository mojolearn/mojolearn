# SPDX-License-Identifier: Apache-2.0
"""`decode(x)` on every resident decode session against `N` session steps,
byte for byte (decode plan D2): the Mamba-1/2/3, Transformer and Samba
sessions. One `decode` call over N tokens must return the bytes N `step`
calls return, token by token, and leave the state (synced back) and the
reports where the steps leave them; a second `decode` after it continues the
same state. Holds on whichever arm the box has (the device session where the
binding exports one, the host arm where it does not). Not a lane.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")

import mojolearn as ml  # noqa: E402
from mojolearn.tests.test_mamba23_decode_sessions import (  # noqa: E402
    _STATE, _block, _state_bytes, _tokens, _weights,
)


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


@pytest.mark.parametrize("which", ["mamba1", "mamba2", "mamba3"])
def test_mamba_decode_equals_the_steps(which):
    blk, dm = _mamba_block(which)
    x = _tokens(2, 10, dm)
    stepped = blk.allocate_state(2)
    with blk.decode_session(stepped) as sess:
        want = np.concatenate([np.asarray(sess.step(np.ascontiguousarray(x[:, t:t + 1])))
                               for t in range(10)], axis=1)
        want_h = None if which == "mamba1" else np.asarray(blk.h_last_).tobytes()
        sess.sync_state()
    decoded = blk.allocate_state(2)
    with blk.decode_session(decoded) as sess:
        # Two calls (6 then 4 tokens): the second continues the first's state.
        got = np.concatenate([np.asarray(sess.decode(np.ascontiguousarray(x[:, :6]))),
                              np.asarray(sess.decode(np.ascontiguousarray(x[:, 6:])))], axis=1)
        assert got.shape == (2, 10, dm)
        if which != "mamba1":
            assert np.asarray(blk.h_last_).tobytes() == want_h
        sess.sync_state()
    assert got.tobytes() == want.tobytes()
    if which == "mamba1":
        assert _m1_state_bytes(decoded) == _m1_state_bytes(stepped)
    else:
        assert _state_bytes(which, decoded) == _state_bytes(which, stepped)


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


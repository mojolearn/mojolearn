# SPDX-License-Identifier: Apache-2.0
"""The PRIVATE Mamba-2 and Mamba-3 resident decode sessions against the
per-call step, byte for byte.

They are private (`Mamba2Block._decode_session`, `Mamba3Block._decode_session`)
because no identity lane records them on any column yet; the reason and what
the lanes need are in `_RESIDENT_SESSION_NOTE` in python/mojolearn/_mamba_impl.py.
This file is what keeps the private code honest in the meantime: every output
and every carried state piece the session hands back must be the bytes the
per-call `block.step` produces on a second fresh state, on whichever arm this
box has (the device session where the binding exports one, the host arm where
it does not). It is not a lane and admits nothing.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")

import mojolearn as ml  # noqa: E402
from mojolearn import mamba  # noqa: E402


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


def test_the_sessions_are_not_public():
    """Until the lanes are recorded; see `_RESIDENT_SESSION_NOTE`."""
    for name in ("Mamba2DecodeSession", "Mamba3DecodeSession"):
        assert name not in mamba.__all__ and not hasattr(mamba, name), name
    assert not hasattr(ml.Mamba2Block, "decode_session")
    assert not hasattr(ml.Mamba3Block, "decode_session")


@pytest.mark.parametrize("which", ["mamba2", "mamba3"])
def test_session_step_is_byte_for_byte_the_per_call_step(which):
    block, dm = _block(which)
    rows, length = 2, 8
    x = _tokens(rows, length, dm)
    owned = block.allocate_state(rows)
    try:
        session = block._decode_session(owned)
    except (ImportError, NotImplementedError) as exc:
        pytest.skip(f"no decode session on this install: {exc}")
    route = "host" if session._native is None else "device"
    out = []
    with session as sess:
        for t in range(length):
            out.append(np.asarray(sess.step(np.ascontiguousarray(x[:, t:t + 1]))).copy())
        sess.sync_state()
    plain = block.allocate_state(rows)
    want = [np.asarray(block.step(np.ascontiguousarray(x[:, t:t + 1]), plain)).copy()
            for t in range(length)]
    got_y, want_y = np.concatenate(out, axis=1), np.concatenate(want, axis=1)
    assert np.isfinite(want_y).all()
    assert got_y.tobytes() == want_y.tobytes(), f"{which} {route} session step differs from per-call step"
    got_s, got_q = _state_bytes(which, owned)
    want_s, want_q = _state_bytes(which, plain)
    assert got_q == want_q, f"{which} {route}: (buffered_tokens, pending) {got_q} vs {want_q}"
    for name, g, w in zip(_STATE[which], got_s, want_s):
        assert g == w, f"{which} {route}: state piece {name} differs after sync_state"

# SPDX-License-Identifier: Apache-2.0
"""`SambaStack.decode_session` against the per-call `SambaStack.step`, byte
for byte: every logits row, the synced state and its cursors, after a
`load_state` rewind, and the refusals. On a binding with the resident decode
the session steps on the device; elsewhere it steps the caller's state through
the per-call step, and the same assertions hold.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")

import mojolearn as ml  # noqa: E402

_M3 = ("theta", "h", "buffer_qrot", "buffer_krot", "buffer_v",
       "buffer_dt", "buffer_sig", "buffer_adt", "pending_k", "pending_v")


def _stack():
    cfg = ml.SambaConfig(vocab=256, d_model=32, layers=("mamba3", "attention"), n_heads=2, intermediate=64)
    try:
        return ml.SambaStack(cfg, generator=ml.training.Generator(1), lr=1e-3)
    except ImportError as exc:
        pytest.skip(f"no Samba binding on this install: {exc}")


def _ids(rows, length):
    return ((np.arange(rows * length, dtype=np.int64) * 37 + 11) % 256).astype(np.int32).reshape(rows, length)


def _state_bytes(st):
    parts, cursors = [], []
    for layer in st.layers:
        if hasattr(layer, "k_cache"):
            parts += [np.asarray(layer.k_cache).tobytes(), np.asarray(layer.v_cache).tobytes()]
            cursors.append(int(layer.cached_tokens))
        else:
            parts += [np.asarray(getattr(layer, n)).tobytes() for n in _M3]
            cursors += [int(layer.buffered_tokens), bool(layer.pending)]
    return parts, cursors


def test_session_steps_and_state_equal_the_per_call_step():
    m = _stack()
    ids = _ids(2, 12)
    st = m.allocate_state(2, 16)
    with m.decode_session(st) as sess:
        assert type(sess).__name__ == "SambaDecodeSession" and sess.is_open and sess.state is st
        first = [np.asarray(sess.step(np.ascontiguousarray(ids[:, t]))).copy() for t in range(4)]
        sess.sync_state()
        synced = _state_bytes(st)
        for t in range(4, 8):
            sess.step(np.ascontiguousarray(ids[:, t:t + 1]))
        sess.load_state()
        after = [np.asarray(sess.step(np.ascontiguousarray(ids[:, t]))).copy() for t in range(8, 12)]
    assert not sess.is_open
    final = _state_bytes(st)
    plain = m.allocate_state(2, 16)
    want_first = [np.asarray(m.step(np.ascontiguousarray(ids[:, t]), plain)).copy() for t in range(4)]
    want_synced = _state_bytes(plain)
    want_after = [np.asarray(m.step(np.ascontiguousarray(ids[:, t]), plain)).copy() for t in range(8, 12)]
    want_final = _state_bytes(plain)
    for got, want in zip(first + after, want_first + want_after):
        assert got.shape == want.shape == (2, 256)
        assert got.tobytes() == want.tobytes()
    assert synced == want_synced
    assert final == want_final


def test_session_refusals():
    m = _stack()
    st = m.allocate_state(2, 8)
    sess = m.decode_session(st)
    try:
        if sess._native is not None:
            # the per-call step and a second session refuse a held state
            with pytest.raises(ValueError):
                m.step(np.zeros(2, dtype=np.int32), st)
            with pytest.raises(ValueError):
                m.decode_session(st)
        with pytest.raises(ValueError):
            sess.step(np.zeros(3, dtype=np.int32))
    finally:
        sess.close()
    sess.close()  # closing twice is a no-op
    with pytest.raises(ValueError):
        sess.step(np.zeros(2, dtype=np.int32))
    with pytest.raises(ValueError):
        m.decode_session(None)
    # after close the state is the caller's again
    m.step(np.zeros(2, dtype=np.int32), st)

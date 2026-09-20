# SPDX-License-Identifier: Apache-2.0
"""`Mamba1DecodeSession` and `TransformerDecodeSession` on the CPU route.

WHAT THIS FILE USED TO SAY, AND WHY IT NO LONGER DOES
(lane/cpu-routes-gpu-only-four, 2026-09-20). It opened by explaining why
neither class could have an identity lane: `mamba1_session_create` and
`transformer_decode_session_create` exist only in the GPU bindings, each
constructor refused by name when the loaded binding did not export its
entry, and a lane would have read REFUSED on every CPU column -- which in a
column total is indistinguishable from a pass.

The refusal was reading the wrong thing. A session's ARITHMETIC is not its
residency: `Mamba1DecodeSession.step` is `mamba_step`, the block at L = 1
with the state carried, and `TransformerDecodeSession.step`/`forward` are
`transformer_decode_step` and `transformer_forward` -- and all three of
those entries ARE exported by the host bindings, under the per-call names
`Mamba1Block.step` and `TransformerBlock.step`/`forward` already take. What
the host route lacks is a device to hold anything resident IN, which is a
statement about cost and not about bytes. So each class now carries a HOST
ARM: it owns its copies of the weights and the state and calls those
per-call entries, and `mamba1-decode-session` and
`transformer-decode-session` take a CPU column in `tools/identity_break.py`.

THE LANES HASH THE ARITHMETIC. This file pins what a hash cannot see, and
the assertions are the session docstrings' own words:

  1. The host arm's output is BYTE FOR BYTE the per-call `step` (and, for
     the transformer, `forward`) on a second fresh state. The lane asserts
     this too, in the cell; it is here as well because the lane can be run
     with `--lanes` and this runs in the ordinary CPU gate.
  2. The OWNERSHIP clause: the session copies at open, the caller's state
     buffers are STALE until `sync_state()`, the block refuses the owned
     state by name, and `load_state()` is the explicit refresh.
  3. A weight edited AFTER open is not observed, which is the other half of
     "the session COPIES the weights at open" and the one a shared buffer
     would silently break.

On a GPU box the same assertions hold on the device arm, so nothing here
skips by column; only the two tests that ask for a missing binding skip.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")

import mojolearn as ml  # noqa: E402


def _weights(shapes, ones=()):
    """Small deterministic weights, the identity harness's shape without its
    machinery: a fixed ramp per tensor so a failure is reproducible."""
    out = {}
    for i, (name, shape) in enumerate(sorted(shapes.items())):
        n = int(np.prod(shape))
        if name in ones:
            a = np.ones(n, dtype=np.float32)
        else:
            a = ((np.arange(n, dtype=np.float32) + 7 * i) % 11 - 5) / 32.0
        out[name] = np.ascontiguousarray(a.astype(np.float32).reshape(shape))
    return out


def _mamba1_block():
    dm, di, r = 32, 64, 2
    w = _weights({
        "norm.weight": (dm,), "in_proj.weight": (2 * di, dm), "conv1d.weight": (di, 1, 4),
        "conv1d.bias": (di,), "x_proj.weight": (r + 32, di), "dt_proj.weight": (di, r),
        "dt_proj.bias": (di,), "A_log": (di, 16), "D": (di,), "out_proj.weight": (dm, di)},
        ones=("norm.weight",))
    try:
        return ml.Mamba1Block(w), dm
    except ImportError as exc:
        pytest.skip(f"no mamba binding on this install: {exc}")


def _transformer_block():
    dm, nh, nkv, hd, it = 32, 2, 1, 16, 64
    w = _weights({
        "input_layernorm.weight": (dm,), "post_attention_layernorm.weight": (dm,),
        "q_proj.weight": (nh * hd, dm), "k_proj.weight": (nkv * hd, dm),
        "v_proj.weight": (nkv * hd, dm), "o_proj.weight": (dm, nh * hd),
        "gate_proj.weight": (it, dm), "up_proj.weight": (it, dm), "down_proj.weight": (dm, it)})
    try:
        return ml.TransformerBlock(w, n_heads=nh, n_kv_heads=nkv), dm
    except ImportError as exc:
        pytest.skip(f"no transformer binding on this install: {exc}")


def _block(which):
    if which == "mamba1":
        block, dm = _mamba1_block()
        return block, dm, lambda: block.allocate_state(2)
    block, dm = _transformer_block()
    return block, dm, lambda: block.allocate_state(2, max_tokens=32)


def _tokens(rows, length, dm, offset=0):
    n = rows * length * dm
    a = ((np.arange(n, dtype=np.float32) + offset) % 13 - 6) / 64.0
    return np.ascontiguousarray(a.reshape(rows, length, dm))


def _state_bytes(state):
    out = []
    for name in ("conv_window", "h", "k_cache", "v_cache"):
        buf = getattr(state, name, None)
        if buf is not None:
            out.append(np.ascontiguousarray(np.asarray(buf)).reshape(-1))
    return np.concatenate(out).tobytes()


@pytest.mark.parametrize("which", ["mamba1", "transformer"])
def test_session_step_is_byte_for_byte_the_per_call_step(which):
    """The session docstrings' own claim, on whichever arm this box has."""
    block, dm, fresh = _block(which)
    x = _tokens(2, 8, dm)
    session_state = fresh()
    session_out = []
    try:
        session = block.decode_session(session_state)
    except (ImportError, NotImplementedError) as exc:
        pytest.skip(f"no decode session on this install: {exc}")
    with session as sess:
        for t in range(8):
            session_out.append(np.asarray(sess.step(np.ascontiguousarray(x[:, t:t + 1]))).copy())
        sess.sync_state()
        resident = _state_bytes(session_state)
    plain_state = fresh()
    plain_out = [np.asarray(block.step(np.ascontiguousarray(x[:, t:t + 1]), plain_state)).copy()
                 for t in range(8)]
    assert np.concatenate(session_out, axis=1).tobytes() == \
        np.concatenate(plain_out, axis=1).tobytes()
    # A session whose outputs agree but whose state does not is a session
    # that cannot be handed back.
    assert resident == _state_bytes(plain_state)


def test_transformer_session_forward_is_byte_for_byte_the_per_call_forward():
    """The prefill half, which only the transformer session has."""
    block, dm = _transformer_block()
    chunk = _tokens(2, 8, dm, offset=101)
    state = block.allocate_state(2, max_tokens=32)
    try:
        session = block.decode_session(state)
    except (ImportError, NotImplementedError) as exc:
        pytest.skip(f"no decode session on this install: {exc}")
    with session as sess:
        got = np.asarray(sess.forward(chunk)).copy()
        assert int(state.cached_tokens) == 8
    plain = block.allocate_state(2, max_tokens=32)
    assert got.tobytes() == np.asarray(block.forward(chunk, plain)).tobytes()
    assert _state_bytes(state) == _state_bytes(plain)


@pytest.mark.parametrize("which", ["mamba1", "transformer"])
def test_the_open_session_owns_the_state_and_load_state_is_the_refresh(which):
    """OWNERSHIP, in the docstring's order: the caller's buffers go stale,
    the block refuses the owned state BY NAME, `sync_state` publishes and
    `load_state` re-reads."""
    block, dm, fresh = _block(which)
    state = fresh()
    x = _tokens(2, 1, dm)
    try:
        sess = block.decode_session(state)
    except (ImportError, NotImplementedError) as exc:
        pytest.skip(f"no decode session on this install: {exc}")
    try:
        zero = _state_bytes(state)
        sess.step(x)
        # STALE: the caller's buffers have not moved, though the session has.
        assert _state_bytes(state) == zero
        with pytest.raises(ValueError, match="resident"):
            block.step(x, state)
        with pytest.raises(ValueError, match="resident"):
            block.decode_session(state)
        sess.sync_state()
        assert _state_bytes(state) != zero
        # `load_state` re-reads the caller's bytes. Zero them, re-upload, and
        # the next step must be the step a FRESH state gives -- which is not
        # the step the session's own carried state would have given.
        carried = np.asarray(sess.step(np.ascontiguousarray(x * 2))).copy()
        for name in ("conv_window", "h", "k_cache", "v_cache"):
            buf = getattr(state, name, None)
            if buf is not None:
                np.asarray(buf)[...] = 0.0
        if which == "transformer":
            state.cached_tokens = 0
        sess.load_state()
        reloaded = np.asarray(sess.step(np.ascontiguousarray(x * 2))).copy()
        assert reloaded.tobytes() == np.asarray(
            block.step(np.ascontiguousarray(x * 2), fresh())).tobytes()
        assert reloaded.tobytes() != carried.tobytes(), \
            "a zeroed reload gave the carried state's answer, so load_state did nothing"
    finally:
        sess.close()
    assert not sess.is_open
    # Handed back: the block takes the state again, and the session refuses.
    block.step(x, state)
    with pytest.raises(ValueError, match="closed"):
        sess.step(x)


@pytest.mark.parametrize("which", ["mamba1", "transformer"])
def test_a_weight_edited_after_open_is_not_observed(which):
    """"Edits to the weights after open are NOT observed: close and open a
    new session." A host arm that aliased the block's buffers instead of
    copying them would pass every other test in this file and fail this
    one."""
    block, dm, fresh = _block(which)
    x = _tokens(2, 1, dm)

    def two_steps(session):
        return (np.asarray(session.step(x)).copy(),
                np.asarray(session.step(np.ascontiguousarray(x * 2))).copy())

    target = np.asarray(block._w[-1])
    keep = target.copy()
    try:
        sess = block.decode_session(fresh())
    except (ImportError, NotImplementedError) as exc:
        pytest.skip(f"no decode session on this install: {exc}")
    try:
        # UNEDITED throughout: the answer the copy must keep giving.
        clean = two_steps(block.decode_session(fresh()))
        first = np.asarray(sess.step(x)).copy()
        target[...] = keep + np.float32(0.5)      # edited AFTER `sess` opened
        second = np.asarray(sess.step(np.ascontiguousarray(x * 2))).copy()
        edited = two_steps(block.decode_session(fresh()))
    finally:
        target[...] = keep
        sess.close()
    # The edit is a real edit, or this test proves nothing.
    assert edited[0].tobytes() != clean[0].tobytes()
    # And `sess`, opened before it, never saw it -- on BOTH steps, so a copy
    # taken lazily on the first step would still be caught.
    assert first.tobytes() == clean[0].tobytes()
    assert second.tobytes() == clean[1].tobytes()

# SPDX-License-Identifier: Apache-2.0
"""Repeated public calls on one block keep their outputs apart (lane
fable-perf-neural-0855, 2026-11-13).

The owned executors (the Mamba block programs, the transformer step arms)
reuse their device buffers and pinned stages across calls, and the transformer
decode step copies from and to the caller's arrays directly. What the public
boundary must keep: every call returns its own array, a later call does not
overwrite an earlier call's output or the caller's input, and the bytes of a
repeated call are the bytes of the first. Not identity or speed evidence."""
import os

import numpy as np
import pytest

os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")

import mojolearn as ml  # noqa: E402
from mojolearn.tests.test_decode_n_tokens import _tokens, _transformer_block  # noqa: E402
from mojolearn.tests.test_mamba23_decode_sessions import _mamba2_block, _weights  # noqa: E402


def _mamba1_block():
    dm, di, dt_rank = 16, 32, 1
    try:
        return ml.Mamba1Block(_weights({
            "norm.weight": (dm,), "in_proj.weight": (2 * di, dm), "conv1d.weight": (di, 1, 4),
            "conv1d.bias": (di,), "x_proj.weight": (dt_rank + 32, di), "dt_proj.weight": (di, dt_rank),
            "dt_proj.bias": (di,), "A_log": (di, 16), "D": (di,), "out_proj.weight": (dm, di)},
            ones=("norm.weight",))), dm
    except ImportError as exc:
        pytest.skip(f"no mamba binding on this install: {exc}")


def _mamba2():
    try:
        return _mamba2_block()
    except ImportError as exc:
        pytest.skip(f"no mamba binding on this install: {exc}")


@pytest.mark.parametrize("make", [_mamba1_block, _mamba2], ids=["mamba1", "mamba2"])
def test_repeated_forward_keeps_each_output_and_the_input(make):
    block, dm = make()
    x = _tokens(2, 6, dm)
    x_bytes = x.tobytes()
    first = block.forward(x)
    first_bytes = np.asarray(first).tobytes()
    second = block.forward(x)
    third = block.forward(np.ascontiguousarray(x * 0.5))
    assert first is not second and second is not third
    assert np.asarray(first).tobytes() == first_bytes, "a later call overwrote an earlier output"
    assert np.asarray(second).tobytes() == first_bytes, "the same call gives the same bytes"
    assert np.asarray(third).tobytes() != first_bytes
    assert x.tobytes() == x_bytes, "the input is read, never written"


def test_transformer_step_forms_agree_and_keep_their_arrays():
    block, dm = _transformer_block()
    x = _tokens(2, 4, dm)
    x_bytes = x.tobytes()
    state = block.allocate_state(2, max_tokens=8)
    outs = []
    with block.decode_session(state) as session:
        for t in range(4):
            outs.append(session.step(np.ascontiguousarray(x[:, t:t + 1])))
        kept = [np.asarray(o).tobytes() for o in outs]
        session.sync_state()
    assert len({id(o) for o in outs}) == 4, "every step returns its own array"
    assert [np.asarray(o).tobytes() for o in outs] == kept, "a later step overwrote an earlier output"
    assert all(np.asarray(o).shape == (2, 1, dm) for o in outs)
    assert x.tobytes() == x_bytes, "the token is read, never written"
    # the (B, d_model) spelling of a token is the (B, 1, d_model) one, bit for bit
    again = block.allocate_state(2, max_tokens=8)
    with block.decode_session(again) as session:
        flat = [np.asarray(session.step(np.ascontiguousarray(x[:, t]))).tobytes() for t in range(4)]
        session.sync_state()
    assert flat == kept
    assert np.asarray(again.k_cache).tobytes() == np.asarray(state.k_cache).tobytes()
    assert np.asarray(again.v_cache).tobytes() == np.asarray(state.v_cache).tobytes()

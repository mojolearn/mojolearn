# SPDX-License-Identifier: Apache-2.0
"""`Mamba1DecodeSession` and `TransformerDecodeSession` on the CPU route.

WHY THIS IS A TEST AND NOT A LANE (lane/laneless-public-classes,
2026-09-19). `tools/verification_matrix.py` reports both classes as public
entries with no identity lane at all, and they must stay that way until
someone writes the lane ON A GPU COLUMN, because there is no CPU column they
could run on: `mamba1_session_create` and `transformer_decode_session_create`
exist only in `bindings/_mojolearn_mamba.mojo` and
`bindings/_mojolearn_transformer.mojo`, never in the host bindings, and each
constructor refuses by name when the loaded binding does not export its
entry. A lane would read REFUSED on every CPU column, and a REFUSED cell is
indistinguishable from a pass in a column total -- which is the failure mode
`tools/verify_lanes.py`'s property 0 exists to refuse.

WHAT THIS FILE DOES HOLD, which is the part a CPU box can hold:

  1. The refusal is BY NAME and names the class, so a build that silently
     lost the export cannot read as "the CPU path took over". A bare
     `AttributeError` or an `ImportError` out of `_backend`'s stand-in would
     pass a `try/except Exception` in a caller and hide the same defect.
  2. `decode_session` is reached through the BLOCK, so the refusal is the
     block's answer and not an import-time absence.
  3. The per-call `step`/`forward` path the refusal points the caller at
     actually works on this route, which is what makes the message true.

On a GPU box (Metal, CUDA or HIP) with the session entry present, the first
two assertions do not apply and the tests skip by name: the CONTRACT they
would then need -- that a session's output is byte for byte the per-call
step's -- belongs to an identity lane on that column, and
`test_transformer_options.py::test_decode_session_carries_the_record`
already holds the transformer half of it where the export exists.
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


def _exports(block, entry):
    """Whether the block's loaded binding exports `entry`. A CPU-only
    install's stand-in raises ImportError from __getattr__, which `hasattr`
    does not swallow, so this is spelled the way `_transformer_impl._exports`
    spells it."""
    try:
        return hasattr(block._extension(), entry)
    except ImportError:
        return False


def test_mamba1_decode_session_refuses_by_name_without_the_export():
    block, dm = _mamba1_block()
    state = block.allocate_state(2)
    if _exports(block, "mamba1_session_create"):
        pytest.skip("this binding exports the resident session; the refusal is not this column's")
    with pytest.raises(NotImplementedError) as excinfo:
        block.decode_session(state)
    message = str(excinfo.value)
    assert "Mamba1DecodeSession" in message, message
    assert "resident decode session" in message, message
    assert "step()" in message, message


def test_transformer_decode_session_refuses_by_name_without_the_export():
    block, dm = _transformer_block()
    state = block.allocate_state(2, max_tokens=32)
    if _exports(block, "transformer_decode_session_create"):
        pytest.skip("this binding exports the resident session; the refusal is not this column's")
    with pytest.raises(NotImplementedError) as excinfo:
        block.decode_session(state)
    message = str(excinfo.value)
    assert "TransformerDecodeSession" in message, message
    assert "resident decode session" in message, message
    assert "step()" in message, message


@pytest.mark.parametrize("which", ["mamba1", "transformer"])
def test_the_per_call_path_the_refusal_names_actually_runs(which):
    """The refusal says "the per-call step() is the path here". If that were
    not true the message would be worse than no message, so it is asserted:
    one token through `step` on the state the session refused to take, twice,
    for the same bytes."""
    if which == "mamba1":
        block, dm = _mamba1_block()
        state = block.allocate_state(2)
    else:
        block, dm = _transformer_block()
        state = block.allocate_state(2, max_tokens=32)
    x = np.ascontiguousarray(((np.arange(2 * dm, dtype=np.float32) % 13 - 6) / 64.0).reshape(2, 1, dm))
    try:
        first = np.asarray(block.step(x, state)).copy()
    except (ImportError, NotImplementedError) as exc:
        pytest.skip(f"no step entry on this install: {exc}")
    if which == "mamba1":
        state2 = block.allocate_state(2)
    else:
        state2 = block.allocate_state(2, max_tokens=32)
    second = np.asarray(block.step(x, state2))
    assert first.shape == (2, 1, dm), first.shape
    assert first.tobytes() == second.tobytes()

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The retained per-call driver of a `TransformerBlock`, kept as a test reference.

`TransformerBlock._call` (with `_call_impl`, `_call_fresh` and `_call_int15`) left the
product in lane x4 (a34f8af7c2): forward / step / backward run through the native
neural-block door. The tests that compared the door against that driver call this copy
instead. It drives the binding's per-call entries directly (`transformer_forward`,
`transformer_forward_fresh`, `transformer_decode_step`, `transformer_forward_int15`) with
the addresses and params the deleted driver built; it never runs in product execution.
The GPU binding's session arm of the old driver (`transformer_session_forward*`) is the
same entry arithmetic on a retained context and is not repeated here.
"""
import numpy as np

from mojolearn._buffer import addr, addr_ro, empty


def _batch_tokens(x, what, d_model, step):
    x = np.ascontiguousarray(x, dtype=np.float32)
    if step and x.ndim == 2:
        x = x.reshape((x.shape[0], 1, x.shape[1]))
    if x.ndim != 3 or x.shape[2] != d_model:
        raise ValueError(f"{what}: x must be (B, L, {d_model}), got {x.shape}")
    if step and x.shape[1] != 1:
        raise ValueError(f"{what}: a decode step takes exactly one token per batch row")
    return x


def _state_buf(a, name, n):
    """The caller's own buffer (never a copy: the kernel updates it in place)."""
    view = memoryview(a)
    if view.readonly or not view.c_contiguous or view.nbytes != 4 * n:
        raise ValueError(f"state buffer {name} must be a writable C-contiguous float32 buffer of {n} values")
    return a


def transformer_call(block, x, state, step):
    """`block`'s forward (`step=False`) or one decode step (`step=True`) through the
    per-call binding entries; `state` (a `TransformerState`, or None for a stateless
    prefill) is updated in place. Returns y (B, L, d_model) float32."""
    what = "TransformerBlock.step" if step else "TransformerBlock.forward"
    x = _batch_tokens(x, what, block.d_model, step)
    ext = block._extension()
    b, l = int(x.shape[0]), int(x.shape[1])
    w = block._w
    wopt = block._wopt  # noqa: F841  (keeps the optional arrays alive)
    planes = getattr(block, "_int15", None)
    if (planes is None and state is None and not step and b > 0 and l > 0 and block.window >= 0
            and hasattr(ext, "transformer_forward_fresh")):
        y = empty((b, l, block.d_model), "<f4")
        addrs, params = block._with_tails(
            [addr_ro(x)] + [0 if a is None else addr_ro(a) for a in w] + [addr(y)],
            [b, l, block.d_model, block.n_heads, block.n_kv_heads, block.head_dim,
             block.intermediate, block.window])
        ext.transformer_forward_fresh(addrs, params)
        return y
    if state is None:
        state = block.allocate_state(b, l)
    if state.batch_size != b:
        raise ValueError(f"{what}: the state was allocated for batch_size {state.batch_size} but x has B = {b}")
    smax = int(state.max_tokens)
    n = b * block.n_kv_heads * state.capacity * block.head_dim
    kc = _state_buf(state.k_cache, "k_cache", n)
    vc = _state_buf(state.v_cache, "v_cache", n)
    s0 = int(state.cached_tokens)
    y = empty((b, l, block.d_model), "<f4")
    head = [addr_ro(x)] + [0 if a is None else addr_ro(a) for a in w] + [addr(kc), addr(vc), addr(y)]
    if planes is not None:
        if not hasattr(ext, "transformer_forward_int15"):
            raise NotImplementedError(f"{what}: this binding has no transformer_forward_int15 entry")
        params = ([b, l, block.d_model, block.n_heads, block.n_kv_heads, block.head_dim,
                   block.intermediate, smax, s0, block.window] + list(block._opts_tail))
        new_len = ext.transformer_forward_int15(head + block._tail_addrs(), params)
    elif step:
        a2, p2 = block._with_tails(
            head, [b, block.d_model, block.n_heads, block.n_kv_heads, block.head_dim,
                   block.intermediate, smax, s0, block.window])
        new_len = ext.transformer_decode_step(a2, p2)
    else:
        a2, p2 = block._with_tails(
            head, [b, l, block.d_model, block.n_heads, block.n_kv_heads, block.head_dim,
                   block.intermediate, smax, s0, block.window])
        new_len = ext.transformer_forward(a2, p2)
    state.cached_tokens = int(new_len)
    return y

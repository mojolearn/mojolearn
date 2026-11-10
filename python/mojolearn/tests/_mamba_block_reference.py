# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The retained Python drivers of the Mamba-1/2/3 blocks, as test oracles.

Each `mamba{1,2,3}_call(block, x, state, step)` spells the block's kernel
arguments and layout independently of the native public forward
(bindings/mamba_public_native.mojo) and calls the binding's
`mamba*_forward` / `mamba*_decode_step` entry, so the native forward can be
held to it (test_mamba_native_public.py, test_neural_block_native.py,
test_parallel_causal_lm.py). Moved out of the product package with the
helpers only they used (`_state_buf`, `_batch_tokens`, `_f32_strict`,
`_m1_profile`, `_m1_affine_buffers`, `_refuse_resident`, Mamba3's
`_call_fresh`) by lane p5-cpu-route-relabel-and-dead-code (2026-11-09): no
product code called them. `block_call` picks the driver by class and falls
back to the block's own `_call` (the transformer block).
"""
import os

from mojolearn import _buffer as _buffers
from mojolearn._arrays import _addr, _addr_ro
from mojolearn._buffer import addr, addr_ro, as_f32_c, empty
from mojolearn._bufcheck import dtype_name, is_native_f32, probe
from mojolearn._mamba_impl import (
    Mamba1Block, Mamba2Block, Mamba3Block,
    _M1_D_CONV, _M1_D_STATE, _M2_CHUNK_SIZE, _M2_D_CONV, _M2_D_STATE, _M2_HEADDIM,
    _M3_CHUNK_SIZE, _M3_D_STATE, _M3_HEADDIM, _M3_NUM_ROPE_ANGLES,
)


def _f32_strict(a, what, name):
    """`a` as a C-contiguous float32 `Array`, REFUSING every other dtype
    BY NAME (module header: this surface certifies bits, so there is no
    convenience downcast). Layout may be fixed silently -- a copy to C
    order moves bytes untouched, so no output bit can move -- but the
    dtype may not. DEVIATION 2407: the dtype is the buffer's FORMAT,
    read through `_buffer.view`; `as_f32_c` is reached only by a float32
    buffer, where its one job is the layout (a zero-copy borrow when the
    buffer is already C-contiguous)."""
    try:
        pb = probe(a)
    except TypeError:
        raise TypeError(
            f"mojolearn {what}: {name} is a {type(a).__name__}, which does "
            "not support the buffer protocol; this surface is float32 ONLY "
            "and takes a float32 array (a numpy array, an array.array('f') "
            "or a mojolearn.Array). Convert yourself and pass float32."
        ) from None
    if not is_native_f32(pb.format):
        raise TypeError(
            f"mojolearn {what}: {name} has dtype {dtype_name(a, pb)}; this "
            "surface is float32 ONLY (the fp32 profiles, "
            "mamba/IDENTICAL_MAMBA_CONTRACT.md section 3 / "
            "IDENTICAL_MAMBA2_CONTRACT.md section 3). bfloat16 and "
            "float16 are refused BY NAME: the references' "
            "reduced-precision runs are a mixture of cast boundaries "
            "that is not this profile, and a certified bf16 profile is "
            "SHIP LATER under its own version name "
            "(archive/evidence/mamba/FEATURE_PARITY.md section 7). float64 is refused "
            "rather than downcast so the bits that ran are bits you "
            "made. Convert yourself and pass float32."
        )
    arr, _copied = as_f32_c(a, ndim=pb.ndim, name=name)
    return arr


def _state_buf(a, what, name, shape):
    """A state buffer: float32, C-contiguous, WRITABLE, exactly `shape`.
    No silent fixups at all -- the state is read AND written in place
    (DEVIATION 792), so a copy here would update the copy and the
    caller's next call would carry a stale state, which is a wrong
    answer with no diagnostic. DEVIATION 2408: "a buffer" is ANY object
    supporting the buffer protocol -- the `mojolearn.Array` that
    `allocate_state` hands out, a NumPy array, an `array.array('f')` --
    judged through `_buffer.view`; the object itself is returned and its
    address taken with `_buffer.addr` (writable required), so the
    caller's own memory is what the kernel updates."""
    try:
        pb = probe(a)
    except TypeError:
        raise TypeError(
            f"mojolearn {what}: state buffer {name} must be a numpy "
            "ndarray or a mojolearn Array -- any writable float32 buffer "
            f"-- (it is read and written in place), got {type(a)!r}"
        ) from None
    if not is_native_f32(pb.format):
        raise TypeError(
            f"mojolearn {what}: state buffer {name} has dtype "
            f"{dtype_name(a, pb)}, want float32 (the state is part of the "
            "fp32 profile and round-trips byte for byte)"
        )
    if pb.shape != shape:
        raise ValueError(
            f"mojolearn {what}: state buffer {name} has shape {pb.shape},"
            f" want {shape}"
        )
    if not pb.c_contiguous or pb.readonly:
        raise ValueError(
            f"mojolearn {what}: state buffer {name} must be C-contiguous"
            " and writable; it is updated IN PLACE so the caller's array"
            " always holds the post-call state (allocate_state() makes"
            " conforming buffers)"
        )
    return a


def _batch_tokens(x, what, d_model, step):
    """`x` as (B, L, d_model) float32. A step call also admits
    (B, d_model) -- their `hidden_states.squeeze(1)`, same bytes."""
    x = _f32_strict(x, what, "x")
    if step and x.ndim == 2:
        x = x.reshape((x.shape[0], 1, x.shape[1]))
    if x.ndim != 3:
        raise ValueError(
            f"mojolearn {what}: x must be (B, L, d_model)"
            + (" -- or (B, d_model) for a step --" if step else "")
            + f", got {x.ndim}-D shape {x.shape}"
        )
    if x.shape[2] != d_model:
        raise ValueError(
            f"mojolearn {what}: x has d_model {x.shape[2]}, the weights "
            f"have {d_model}"
        )
    if step and x.shape[1] != 1:
        raise ValueError(
            f"mojolearn {what}: a decode step takes exactly one token "
            f"per batch row (mamba_simple.py:210), got L = {x.shape[1]};"
            " use forward() for prefill"
        )
    return x


def _m1_profile(ext):
    return int(getattr(ext, "mamba1_profile", lambda: 1)())


def _m1_affine_buffers(state, b, di, profile, what):
    """API metadata/buffer validation only; all numerical work stays Mojo."""
    if int(getattr(state, "profile_id", 1)) != profile:
        raise ValueError(f"mojolearn {what}: checkpoint numerical profile differs from the binding")
    if profile != 2:
        return []
    if not getattr(state, "_nn34_usable", True):
        raise ValueError(f"mojolearn {what}: NN34 state was lost after a failed call; restore a checkpoint")
    position = int(state.absolute_position)
    if position < 0 or position > (1 << 62) - 32:
        raise ValueError(f"mojolearn {what}: invalid absolute checkpoint position")
    return [
        _state_buf(state.affine_boundary, what, "affine_boundary", (b, di, _M1_D_STATE)),
        _state_buf(state.affine_a, what, "affine_a", (b, di, _M1_D_STATE, 6)),
        _state_buf(state.affine_b, what, "affine_b", (b, di, _M1_D_STATE, 6)),
    ]


def _refuse_resident(state, what):
    owner = getattr(state, "_resident_session", None)
    if owner is not None:
        raise ValueError(
            f"mojolearn {what}: the state is owned by an open resident "
            "decode session (its pieces live in the session, on the "
            "device or on the host, and the caller's buffers are stale); "
            "call sync_state() or close() on the session first"
        )


def mamba1_call(self, x, state, step):
    what = "Mamba1Block.step" if step else "Mamba1Block.forward"
    x = _batch_tokens(x, what, self.d_model, step)
    b, l = int(x.shape[0]), int(x.shape[1])
    if state is None:
        state = self.allocate_state(b)
    _refuse_resident(state, what)
    win = _state_buf(state.conv_window, what, "conv_window",
                     (b, self.d_inner, _M1_D_CONV))
    h = _state_buf(state.h, what, "h",
                   (b, self.d_inner, _M1_D_STATE))
    y = empty((b, l, self.d_model), "<f4")
    # TWO LISTS, NOT FOURTEEN ARGUMENTS (DEVIATION 791). Every array
    # addressed below is bound in this frame -- x, the ten entries of
    # self._w (alive on self), win, h, y -- which is what keeps the
    # addresses alive (_buffer.py). `addr` (writable) for the state
    # pieces and the output, `addr_ro` for x and the weights.
    w = self._w
    ext = self._extension()
    addrs = (
        # ORDER MATCHES bindings/_mojolearn_mamba.mojo::
        # mamba1_forward_binding: x, norm.weight, in_proj.weight,
        # conv1d.weight, conv1d.bias, x_proj.weight, dt_proj.weight,
        # dt_proj.bias, A_log, D, out_proj.weight, conv_window, h,
        # y_out
        [addr_ro(x, name="x")]
        + [addr_ro(a, name="weight") for a in w]  # glue: one address per weight tensor
        + [addr(win, name="conv_window"), addr(h, name="h"),
           addr(y, name="y")]
    )
    profile = _m1_profile(ext)
    affine = _m1_affine_buffers(state, b, self.d_inner, profile, what)
    addrs += [addr(a, name="affine checkpoint") for a in affine]  # glue: fixed state-buffer list
    params = [b, self.d_model] if step else [b, l, self.d_model]
    if profile == 2:
        params.append(state.absolute_position)
    try:
        if step:
            ext.mamba1_decode_step(addrs, params)
        else:
            ext.mamba1_forward(addrs, params)
    except Exception:
        if profile == 2:
            state._nn34_usable = False
        raise
    if profile == 2:
        state.absolute_position += l
    return y


def mamba2_call(self, x, state, step):
    what = "Mamba2Block.step" if step else "Mamba2Block.forward"
    x = _batch_tokens(x, what, self.d_model, step)
    b, l = int(x.shape[0]), int(x.shape[1])
    if state is None:
        state = self.allocate_state(b)
    _refuse_resident(state, what)
    win = _state_buf(state.conv_window, what, "conv_window",
                     (b, self.conv_dim, _M2_D_CONV))
    h = _state_buf(state.h, what, "h",
                   (b, self.nheads, _M2_HEADDIM, _M2_D_STATE))
    bx = _state_buf(state.buffer_xbc, what, "buffer_xbc",
                    (b, _M2_CHUNK_SIZE, self.conv_dim))
    bd = _state_buf(state.buffer_dtraw, what, "buffer_dtraw",
                    (b, _M2_CHUNK_SIZE, self.nheads))
    q0 = int(state.buffered_tokens)
    # buffered_tokens outside [0, 256) is a boundary disagreement the
    # binding refuses by name; it goes down unjudged.
    y = empty((b, l, self.d_model), "<f4")
    h_last = empty((b, self.nheads, _M2_HEADDIM, _M2_D_STATE), "<f4")
    # TWO LISTS, NOT SIXTEEN ARGUMENTS (DEVIATION 791). Every array
    # addressed below is bound in this frame -- x, the nine entries
    # of self._w (alive on self), win, h, bx, bd, y, h_last.
    w = self._w
    ext = self._extension()
    addrs = (
        # ORDER MATCHES bindings/_mojolearn_mamba.mojo::
        # mamba2_forward_binding: x, block norm.weight,
        # in_proj.weight, conv1d.weight, conv1d.bias, dt_bias, A_log,
        # D, gated norm.weight, out_proj.weight, conv_window, h,
        # buffer_xbc, buffer_dtraw, y_out, h_last_out
        [addr_ro(x, name="x")]
        + [addr_ro(a, name="weight") for a in w]  # glue: one address per weight tensor
        + [addr(win, name="conv_window"), addr(h, name="h"),
           addr(bx, name="buffer_xbc"), addr(bd, name="buffer_dtraw"),
           addr(y, name="y"), addr(h_last, name="h_last")]
    )
    lo, hi = self.dt_limit
    if step:
        # B, d_model, buf_len, dt_lo, dt_hi.
        new_len = ext.mamba2_decode_step(
            addrs, [b, self.d_model, q0, lo, hi]
        )
    else:
        # B, L, d_model, buf_len, dt_lo, dt_hi.
        new_len = ext.mamba2_forward(
            addrs, [b, l, self.d_model, q0, lo, hi]
        )
    state.buffered_tokens = int(new_len)
    #: The REPORT state (S17 after the final PADDED chunk), NOT the
    #: resumption state -- contract section 5's distinction.
    self.h_last_ = h_last
    return y


def _mamba3_call_fresh(self, x, ext):
    """Discard-only prefill: return all reports without a host cache."""
    if getattr(self, "_owned_restore_pending", False):
        self.install_owned_weights()
    b, l = int(x.shape[0]), int(x.shape[1])
    nh = self.nheads
    y = _buffers.empty((b, l, self.d_model), '<f4')
    # lane/neural-net-experiment (2026-09-30): the session keeps the
    # weights on the device across calls (reused only when their bytes
    # are unchanged; the binding compares them). Same entry arithmetic.
    session_forward = None
    try:
        session_forward = getattr(ext, "mamba3_prefill_session_forward", None)
    except (ImportError, AttributeError):
        session_forward = None
    owned = getattr(self, "_owned_generation", None) is not None
    use_session = ((session_forward is not None or owned)
                   and os.environ.get("MOJOLEARN_MAMBA3_LEGACY_SETUP") != "1")
    if owned and getattr(self, "_prefill_binding", None) is not ext:
        raise RuntimeError("Mamba3 owned snapshot belongs to another binding; release or explicitly reinstall it")
    if getattr(self, "_owned_generation", None) is not None and not use_session:
        raise RuntimeError("Mamba3 owned weights require the installed prefill session; release it before changing binding/setup")
    # lane/fam2-lm (2026-10-04): a block owned by a stack (SambaStack
    # sets `_discard_reports`) reads no report, so where the binding
    # admits it the four report downloads are not requested (address 0)
    # and the `*_last_` attributes are None after the call.
    skip_reports = False
    if use_session and getattr(self, "_discard_reports", False):
        optional = getattr(ext, "mamba3_prefill_session_reports_optional", None)
        skip_reports = bool(optional()) if optional is not None else False
    if skip_reports:
        h_last = k_last = v_last = theta_last = None
        report_addrs = [0, 0, 0, 0]
    else:
        h_last = _buffers.empty((b, nh, _M3_HEADDIM, _M3_D_STATE), '<f4')
        k_last = _buffers.empty((b, nh, _M3_D_STATE), '<f4')
        v_last = _buffers.empty((b, nh, _M3_HEADDIM), '<f4')
        theta_last = _buffers.empty((b, nh, _M3_NUM_ROPE_ANGLES), '<f4')
        report_addrs = [_addr(h_last), _addr(k_last), _addr(v_last), _addr(theta_last)]
    addrs = ([_addr_ro(x)] + [_addr_ro(w) for w in self._w]  # glue: one address per weight tensor
             + [_addr(y)] + report_addrs)
    if use_session:
        if getattr(self, "_prefill_session", None) is None or getattr(self, "_prefill_binding", None) is not ext:
            self._prefill_session = ext.mamba3_prefill_session_create()
            self._prefill_binding = ext
        generation = getattr(self, "_owned_generation", None)
        if generation is not None:
            ext.mamba3_prefill_session_forward_owned(
                self._prefill_session, [_addr_ro(x), _addr(y)] + report_addrs,
                [b, l, self.d_model, generation])
        else:
            session_forward(self._prefill_session, addrs, [b, l, self.d_model])
    else:
        ext.mamba3_forward_fresh(addrs, [b, l, self.d_model])
    self.h_last_ = h_last
    self.k_last_ = k_last
    self.v_last_ = v_last
    self.theta_last_ = theta_last
    return y


def mamba3_call(self, x, state, step):
    what = "Mamba3Block.step" if step else "Mamba3Block.forward"
    x = _batch_tokens(x, what, self.d_model, step)
    b, l = int(x.shape[0]), int(x.shape[1])
    fresh_ext = None
    if state is None and not step:
        fresh_ext = self._extension()
        if hasattr(fresh_ext, "mamba3_forward_fresh"):
            return _mamba3_call_fresh(self, x, fresh_ext)
    if state is None:
        state = self.allocate_state(b)
    _refuse_resident(state, what)
    nh, q = self.nheads, _M3_CHUNK_SIZE
    theta = _state_buf(state.theta, what, "theta",
                       (b, nh, _M3_NUM_ROPE_ANGLES))
    h = _state_buf(state.h, what, "h",
                   (b, nh, _M3_HEADDIM, _M3_D_STATE))
    bq = _state_buf(state.buffer_qrot, what, "buffer_qrot",
                    (b, q, nh, _M3_D_STATE))
    bk = _state_buf(state.buffer_krot, what, "buffer_krot",
                    (b, q, nh, _M3_D_STATE))
    bv = _state_buf(state.buffer_v, what, "buffer_v",
                    (b, q, nh, _M3_HEADDIM))
    bd = _state_buf(state.buffer_dt, what, "buffer_dt", (b, q, nh))
    bs = _state_buf(state.buffer_sig, what, "buffer_sig", (b, q, nh))
    ba = _state_buf(state.buffer_adt, what, "buffer_adt", (b, q, nh))
    pk = _state_buf(state.pending_k, what, "pending_k",
                    (b, nh, _M3_D_STATE))
    pv = _state_buf(state.pending_v, what, "pending_v",
                    (b, nh, _M3_HEADDIM))
    q0 = int(state.buffered_tokens)
    pend = 1 if state.pending else 0
    # buffered_tokens outside [0, 64] is a boundary disagreement the
    # binding refuses by name (DEVIATION 794); it goes down unjudged.
    y = empty((b, l, self.d_model), "<f4")
    h_last = empty((b, nh, _M3_HEADDIM, _M3_D_STATE), "<f4")
    k_last = empty((b, nh, _M3_D_STATE), "<f4")
    v_last = empty((b, nh, _M3_HEADDIM), "<f4")
    theta_last = empty((b, nh, _M3_NUM_ROPE_ANGLES), "<f4")
    # TWO LISTS, NOT TWENTY-FIVE ARGUMENTS (DEVIATION 791). Every
    # array addressed below is bound in this frame -- x, the nine
    # entries of self._w (alive on self), the ten state pieces, y and
    # the four reports.
    w = self._w
    ext = fresh_ext if fresh_ext is not None else self._extension()
    addrs = (
        # ORDER MATCHES bindings/_mojolearn_mamba.mojo::
        # mamba3_forward_binding: x, block norm.weight,
        # in_proj.weight, dt_bias, B_norm.weight, C_norm.weight,
        # B_bias, C_bias, D, out_proj.weight, theta, h, buffer_qrot,
        # buffer_krot, buffer_v, buffer_dt, buffer_sig, buffer_adt,
        # pending_k, pending_v, y_out, h_last_out, k_last_out,
        # v_last_out, theta_last_out
        [addr_ro(x, name="x")]
        + [addr_ro(a, name="weight") for a in w]  # glue: one address per weight tensor
        + [addr(theta, name="theta"), addr(h, name="h"),
           addr(bq, name="buffer_qrot"), addr(bk, name="buffer_krot"),
           addr(bv, name="buffer_v"), addr(bd, name="buffer_dt"),
           addr(bs, name="buffer_sig"), addr(ba, name="buffer_adt"),
           addr(pk, name="pending_k"), addr(pv, name="pending_v"),
           addr(y, name="y"), addr(h_last, name="h_last"),
           addr(k_last, name="k_last"), addr(v_last, name="v_last"),
           addr(theta_last, name="theta_last")]
    )
    if step:
        # B, d_model, buf_len, pending.
        new_len = ext.mamba3_decode_step(
            addrs, [b, self.d_model, q0, pend]
        )
    else:
        # B, L, d_model, buf_len, pending.
        new_len = ext.mamba3_forward(
            addrs, [b, l, self.d_model, q0, pend]
        )
    state.buffered_tokens = int(new_len)
    # ALWAYS False after a call: the unarmed core consumes a pending
    # continuation and an armed build aborts at PyInit, so no return
    # slot exists for it (DEVIATION 794).
    state.pending = False
    #: The REPORT stages (h/k/v after the final working chunk, theta
    #: after the final token), NOT the resumption state -- DEVIATION
    #: 832's sealed-boundary distinction.
    self.h_last_ = h_last
    self.k_last_ = k_last
    self.v_last_ = v_last
    self.theta_last_ = theta_last
    return y


def block_call(block, x, state, step):
    """The retained driver of `block` (a Mamba block, or any block with its own `_call`)."""
    if isinstance(block, Mamba1Block):
        return mamba1_call(block, x, state, step)
    if isinstance(block, Mamba2Block):
        return mamba2_call(block, x, state, step)
    if isinstance(block, Mamba3Block):
        return mamba3_call(block, x, state, step)
    return block._call(x, state, step)

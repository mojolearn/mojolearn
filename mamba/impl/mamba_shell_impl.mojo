# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed Mamba-1, Mamba-2 and Mamba-3 block entries behind `_mojolearn_mamba`.

The sessions, the run functions and the host transfer helpers that the
Python boundary in `bindings/_mojolearn_mamba.mojo` calls. Nothing here
touches Python: every entry takes buffer addresses, Ints and typed sessions.
"""

from core.hostptr import f32_ptr, read_f32, copy_f32
from std.memory import bitcast, memcpy
from std.time import perf_counter_ns
from std.sys.compile import is_defined, get_defined_int
from mamba.impl.ops.mamba3_siso import m3_phase_tick
from mamba.impl.modules.mamba3_transfer import m3_upload, m3_download
from std.os import abort, getenv

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.neural_context import neural_ctx
from core.device_scan import device_first_nonfinite
# One process-lifetime DeviceContext per binding and tier (core/neural_context.mojo).
comptime _NEURAL_CTX = "MojoNeuralMambaContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoNeuralMambaContextFast"
from checks.vendor import COMPILED_VENDOR

from mamba.checks.mamba_fixture import (
    D_CONV,
    D_STATE,
    MambaDims,
    MambaWeights,
)
from mamba.impl.modeling.modeling_mamba import (
    BLOCK_ANY_SABOTAGE,
    MAMBA_GUARD,
    mamba_block_forward,
    mamba_copy_in,
    mamba_device_alloc,
    mamba_download,
    mamba_upload,
    mamba_zeros,
    MambaDeviceStages,
    MambaDeviceState,
    MambaDeviceWeights,
)
from mamba.impl.modules.mamba_simple import mamba_step
#: lane afn-mamba (2026-10-03): under FAST + Apple + `-D MOJOLEARN_AFN_MAMBA_ARENA`
#: every block call's device buffers are views of ONE arena and the caller's
#: arrays are copied in with no per-buffer wait (`_mamba{1,2,3}_run_arena`).
from mamba.impl.modules.afn_defines import AFN_MAMBA_ARENA, IDN_MAMBA_ALLOC_NOWAIT
from mamba.impl.modules.afn_defines import IDN_MAMBA3_REPORTS_ON_REQUEST
from mamba.impl.modules.afn_defines import IDN_M3_SESSION_STAGE_REUSE
from mamba.impl.modules.afn_arena import MambaArena
from mamba.impl.modeling.modeling_mamba import mamba1_arena_floats
from mamba.impl.modules.mamba2 import mamba2_arena_floats
from mamba.impl.modules.mamba3 import mamba3_arena_floats
from mamba.checks.mamba2_fixture import (
    M2_CHUNK_SIZE,
    M2_D_CONV,
    M2_D_STATE,
    M2_HEADDIM,
    Mamba2Dims,
    Mamba2Weights,
)
from mamba.impl.modules.mamba2 import (
    BLOCK2_ANY_SABOTAGE,
    Mamba2DeviceStages,
    Mamba2DeviceState,
    Mamba2DeviceWeights,
    mamba2_block_forward,
)
from mamba.checks.mamba3_fixture import (
    M3_CHUNK_SIZE,
    M3_D_STATE,
    M3_HEADDIM,
    M3_NUM_ROPE_ANGLES,
    Mamba3Dims,
    Mamba3Weights,
)
from mamba.impl.modules.mamba3 import (
    BLOCK3_ANY_SABOTAGE,
    Mamba3DeviceStages,
    Mamba3DeviceState,
    Mamba3DeviceWeights,
    mamba3_block_forward,
)


from autograd.kernels.mamba2_proj import ANY_BWD2_SABOTAGE
from autograd.kernels.mamba3_proj import ANY_BWD3_SABOTAGE
# The Mamba-1 / Mamba-2 backward entries assemble the VJP on the tape
# (autograd/rules_mamba.mojo), which launches the hand kernels in the hand order.
from autograd.rules_mamba import mamba1_tape_backward, mamba2_tape_backward
from autograd.rules_mamba3 import mamba3_tape_backward, mamba3_tape_backward_on
from autograd.kernels.mamba1_prefill import Mamba1PrefillGradients
from autograd.kernels.mamba2_prefill import Mamba2PrefillGradients
from autograd.kernels.mamba1_proj import ANY_BWD_SABOTAGE as M1_ANY_BWD_SABOTAGE
from autograd.kernels.mamba1 import ANY_BWD_BLOCK_SABOTAGE
from autograd.kernels.mamba1_scan import ANY_BWD_SCAN_SABOTAGE
from gemm.checks.gemm_backward import ANY_BWD_SABOTAGE as GEMM_ANY_BWD_SABOTAGE
from gemm.checks.gemm_identical import ANY_SABOTAGE as GEMM_ANY_SABOTAGE

def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _read_f32(addr: Int, n: Int) raises -> List[Float32]:
    return read_f32(addr, max(0, n))


def _write_f32(addr: Int, values: List[Float32]) raises:
    copy_f32(values.unsafe_ptr(), _f32_ptr(addr), len(values))


comptime M3_DIRECT_TRANSFER = not is_defined["MOJOLEARN_MAMBA3_LEGACY_DIRECT_TRANSFER"]()


def _m3_read_f32(addr: Int, n: Int) raises -> List[Float32]:
    return read_f32(addr, n)


def _m3_write_f32(addr: Int, values: List[Float32]) raises:
    var p = _f32_ptr(addr)
    if len(values) > 0:
        memcpy(dest=p, src=values.unsafe_ptr(), count=len(values))


def _m3_upload_addr[wait: Bool = True](ctx: DeviceContext, addr: Int, n: Int) raises -> DeviceBuffer[DType.float32]:
    """`wait=False` (lane/neural-apple, 2026-09-28) enqueues the copy and
    returns: the caller's array outlives the call, every reader is enqueued
    after it on the same in-order `ctx`, and the forward entry waits once
    at its end."""
    comptime if not M3_DIRECT_TRANSFER:
        return m3_upload(ctx, _m3_read_f32(addr, n))
    else:
        var src = _f32_ptr(addr)
        var count = max(n, 1)
        var dev = mamba_device_alloc(ctx, count)
        # DeviceContext accepts ordinary host pointers; the Python frame
        # retains each NumPy array until the synchronized copy completes.
        comptime if not is_defined["MOJOLEARN_MAMBA3_LEGACY_CALLER_TRANSFER"]():
            if n > 0:
                mamba_copy_in(ctx, dev, src, count)
                comptime if wait:
                    ctx.synchronize()
                return dev^
        var host = ctx.enqueue_create_host_buffer[DType.float32](count)
        ctx.synchronize()
        if n > 0:
            memcpy(dest=host.unsafe_ptr(), src=src, count=n)
        else:
            host.unsafe_ptr().unsafe_store(0, Float32(0.0))
        mamba_copy_in(ctx, dev, host.unsafe_ptr(), count)
        ctx.synchronize()
        _ = host^
        return dev^


def _m3_download_addr[wait: Bool = True](ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int, addr: Int) raises:
    """`wait=False` (lane/neural-apple, 2026-09-28) enqueues the copy into
    the caller's array and returns; the caller waits once after its last
    download, before any buffer is released or the host reads."""
    comptime if not M3_DIRECT_TRANSFER:
        _m3_write_f32(addr, m3_download(ctx, buf, n))
    else:
        var dst = _f32_ptr(addr)
        comptime if not is_defined["MOJOLEARN_MAMBA3_LEGACY_CALLER_TRANSFER"]():
            if n == len(buf):
                ctx.enqueue_copy(dst_ptr=dst, src_buf=buf)
            else:
                var direct_view = buf.create_sub_buffer[DType.float32](0, n)
                ctx.enqueue_copy(dst_ptr=dst, src_buf=direct_view)
            comptime if wait:
                ctx.synchronize()
            return
        var host = ctx.enqueue_create_host_buffer[DType.float32](n)
        ctx.synchronize()
        if n == len(buf):
            ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
        else:
            var view = buf.create_sub_buffer[DType.float32](0, n)
            ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
        ctx.synchronize()
        if n > 0:
            memcpy(dest=dst, src=host.unsafe_ptr(), count=n)
        _ = host^




def _mamba1_run(a: List[Int], b: Int, l: Int, dm: Int, decode: Bool = False, absolute_start: Int = 0) raises:
    """The GIL-free half of `mamba1_forward_binding`: everything after
    the `PythonObject`s have been read. Builds the host weights, uploads
    the caller's state, runs THE certified entry point once, and writes
    the output and the post-call state back into the caller's buffers.

    `[[mojo-buffer-freed-at-last-use]]`: every device buffer below is
    still alive when `mamba_block_forward` returns because that function
    synchronizes before it does, and the explicit transfers at the end
    hold them past the last download anyway."""
    comptime if AFN_MAMBA_ARENA:
        _mamba1_run_arena(a, b, l, dm, decode)
        return
    var dims = MambaDims.of(dm)
    var di = dims.d_inner
    var r = dims.dt_rank
    var xr = dims.x_proj_rows()

    # Host weights, THROUGH the lane's own struct (its field table is the
    # reference shape authority); values arrive as given bits, unjudged
    # (the initializer rows of FEATURE_PARITY.md are REFUSED as bitwise
    # claims; weights-in is what ships).
    var w = MambaWeights(dims)
    w.norm_w = _read_f32(a[1], dm)
    w.w_in = _read_f32(a[2], 2 * di * dm)
    w.conv_w = _read_f32(a[3], di * D_CONV)
    w.conv_b = _read_f32(a[4], di)
    w.w_x = _read_f32(a[5], xr * di)
    w.w_dt = _read_f32(a[6], di * r)
    w.b_dt = _read_f32(a[7], di)
    w.a_log = _read_f32(a[8], di * D_STATE)
    w.d_skip = _read_f32(a[9], di)
    w.w_out = _read_f32(a[10], dm * di)

    var ctx = neural_ctx[_NEURAL_CTX]()
    var dw = MambaDeviceWeights(ctx, w)
    # The caller's state, uploaded over the fresh zeros. Zeros in IS
    # allocate_inference_cache; anything else is a carried sequence.
    var dstate = MambaDeviceState(ctx, b, dims)
    dstate.conv_win = mamba_upload(ctx, _read_f32(a[11], b * di * D_CONV))
    dstate.h = mamba_upload(ctx, _read_f32(a[12], b * di * D_STATE))
    var dstages = MambaDeviceStages(ctx, b, l, dims)
    var dx = mamba_upload(ctx, _read_f32(a[0], b * l * dm))

    var trace = IdentityTrace.disabled()
    if decode:
        mamba_step(ctx, dstages, dstate, dw, dx, b, trace, String("py"))
    else:
        mamba_block_forward(
            ctx, dstages, dstate, dw, dx, b, l, trace, String("py")
        )

    # The block output is the residual stage (contract section 2's
    # `hidden = residual + mixer(norm(residual))`), and the state buffers
    # go back to their owners.
    # Straight into the caller's arrays, one wait (lane/neural-apple,
    # 2026-09-28; was a host List per piece, a wait and a second copy each).
    _m3_download_addr[False](ctx, dstages.residual_out, b * l * dm, a[13])
    _m3_download_addr[False](ctx, dstate.conv_win, b * di * D_CONV, a[11])
    _m3_download_addr[False](ctx, dstate.h, b * di * D_STATE, a[12])
    ctx.synchronize()
    _ = dw^
    _ = dstate^
    _ = dstages^
    _ = dx^
    _ = ctx^


def _mamba1_run_arena(a: List[Int], b: Int, l: Int, dm: Int, decode: Bool) raises:
    """`_mamba1_run` under MOJOLEARN_AFN_MAMBA_ARENA (lane afn-mamba): one
    arena allocation, every weight / state piece / stage / x a view of it,
    the caller's arrays copied in directly (they outlive the call), ONE
    wait at the end. The same certified entry point runs."""
    var dims = MambaDims.of(dm)
    var di = dims.d_inner
    var r = dims.dt_rank
    var xr = dims.x_proj_rows()
    var ctx = neural_ctx[_NEURAL_CTX]()
    var arena = MambaArena(ctx, mamba1_arena_floats(b, l, dims), MAMBA_GUARD)
    var dw = MambaDeviceWeights(ctx, dims, arena)
    mamba_copy_in(ctx, dw.norm_w, _f32_ptr(a[1]), dm)
    mamba_copy_in(ctx, dw.w_in, _f32_ptr(a[2]), 2 * di * dm)
    mamba_copy_in(ctx, dw.conv_w, _f32_ptr(a[3]), di * D_CONV)
    mamba_copy_in(ctx, dw.conv_b, _f32_ptr(a[4]), di)
    mamba_copy_in(ctx, dw.w_x, _f32_ptr(a[5]), xr * di)
    mamba_copy_in(ctx, dw.w_dt, _f32_ptr(a[6]), di * r)
    mamba_copy_in(ctx, dw.b_dt, _f32_ptr(a[7]), di)
    mamba_copy_in(ctx, dw.a_log, _f32_ptr(a[8]), di * D_STATE)
    mamba_copy_in(ctx, dw.d_skip, _f32_ptr(a[9]), di)
    mamba_copy_in(ctx, dw.w_out, _f32_ptr(a[10]), dm * di)
    var dstate = MambaDeviceState(ctx, b, dims, arena)
    mamba_copy_in(ctx, dstate.conv_win, _f32_ptr(a[11]), b * di * D_CONV)
    mamba_copy_in(ctx, dstate.h, _f32_ptr(a[12]), b * di * D_STATE)
    var dstages = MambaDeviceStages(ctx, b, l, dims, arena)
    var dx = arena.take(ctx, b * l * dm)
    mamba_copy_in(ctx, dx, _f32_ptr(a[0]), b * l * dm)

    var trace = IdentityTrace.disabled()
    if decode:
        mamba_step(ctx, dstages, dstate, dw, dx, b, trace, String("py"))
    else:
        mamba_block_forward(
            ctx, dstages, dstate, dw, dx, b, l, trace, String("py")
        )

    _m3_download_addr[False](ctx, dstages.residual_out, b * l * dm, a[13])
    _m3_download_addr[False](ctx, dstate.conv_win, b * di * D_CONV, a[11])
    _m3_download_addr[False](ctx, dstate.h, b * di * D_STATE, a[12])
    ctx.synchronize()
    _ = dw^
    _ = dstate^
    _ = dstages^
    _ = dx^
    _ = arena^
    _ = ctx^


def _mamba1_backward_run(a: List[Int], b: Int, l: Int, dm: Int) raises:
    var dims = MambaDims.of(dm)
    var di = dims.d_inner
    var r = dims.dt_rank
    var xr = dims.x_proj_rows()

    # Host weights, THROUGH the lane's own struct (its field table is the
    # reference shape authority); values arrive as given bits, unjudged
    # (the initializer rows of FEATURE_PARITY.md are REFUSED as bitwise
    # claims; weights-in is what ships).
    var w = MambaWeights(dims)
    w.norm_w = _read_f32(a[1], dm)
    w.w_in = _read_f32(a[2], 2 * di * dm)
    w.conv_w = _read_f32(a[3], di * D_CONV)
    w.conv_b = _read_f32(a[4], di)
    w.w_x = _read_f32(a[5], xr * di)
    w.w_dt = _read_f32(a[6], di * r)
    w.b_dt = _read_f32(a[7], di)
    w.a_log = _read_f32(a[8], di * D_STATE)
    w.d_skip = _read_f32(a[9], di)
    w.w_out = _read_f32(a[10], dm * di)

    # TOMBSTONE: MOJOLEARN_AUTOGRAD_MAMBA_OFF (hand Mamba-1 prefill backward arm) deleted 2026-10-19 by 47582a7fb; code recoverable at 47582a7fb^.
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_MAMBA_OFF.patch
    var gradients: Mamba1PrefillGradients = mamba1_tape_backward(
        w, _read_f32(a[0], b * l * dm), _read_f32(a[11], b * l * dm), b, l,
        neural_ctx[_NEURAL_CTX](),
    )
    _write_f32(a[12], gradients.x)
    _write_f32(a[13], gradients.norm_weight)
    _write_f32(a[14], gradients.in_proj_weight)
    _write_f32(a[15], gradients.conv1d_weight)
    _write_f32(a[16], gradients.conv1d_bias)
    _write_f32(a[17], gradients.x_proj_weight)
    _write_f32(a[18], gradients.dt_proj_weight)
    _write_f32(a[19], gradients.dt_proj_bias)
    _write_f32(a[20], gradients.A_log)
    _write_f32(a[21], gradients.D)
    _write_f32(a[22], gradients.out_proj_weight)


# ===========================================================================
# The Mamba-1 resident decode session (DEVIATION 2941, lane/infer-speed-
# neural, 2026-09-17): device-owned weights, state and L = 1 stages that
# live across decode steps. The transformer binding's DEVIATION 2940, the
# same shape and the same reason it cannot move a bit: every step is
# `mamba_step`, THE certified decode entry (the block at L = 1), on
# `MambaDeviceWeights`, `MambaDeviceState` and `MambaDeviceStages` holding
# the bytes the per-call entry uploads per token, built once instead. Stage
# reuse across calls is the Mamba-1 stage struct's own contract (every stage
# the block reads it writes first in the same call). Ownership: the session
# COPIES the weights and the state at open; the caller's state arrays are
# STALE until export_state or close; weights refresh is close and open.
# ===========================================================================


struct Mamba1DecodeSession(Movable, Writable):
    var ctx: Optional[DeviceContext]
    var w: Optional[MambaDeviceWeights]
    var state: Optional[MambaDeviceState]
    var stages: Optional[MambaDeviceStages]
    var dx: Optional[DeviceBuffer[DType.float32]]
    var b: Int
    var dm: Int
    var di: Int
    var busy: Bool
    var usable: Bool

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()
        self.w = Optional[MambaDeviceWeights]()
        self.state = Optional[MambaDeviceState]()
        self.stages = Optional[MambaDeviceStages]()
        self.dx = Optional[DeviceBuffer[DType.float32]]()
        self.b = 0
        self.dm = 0
        self.di = 0
        self.busy = False
        self.usable = True

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Mamba1DecodeSession")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("Mamba1DecodeSession")

    def is_open(self) -> Bool:
        return Bool(self.ctx) and Bool(self.state)

    def release(mut self):
        self.stages = None
        self.dx = None
        self.state = None
        self.w = None
        if self.ctx:
            try:
                self.ctx.value().synchronize()
            except:
                pass
        self.ctx = None

    def __deinit__(deinit self):
        self.release()

    def close(mut self) raises:
        if self.busy:
            raise Error("mamba1 session: busy")
        self.usable = False
        self.release()


def _m1_require_session_open(s: Mamba1DecodeSession) raises:
    if s.busy:
        raise Error("mamba1 session: busy")
    if not s.is_open():
        raise Error("mamba1 session: not open (mamba1_session_open first)")
    if not s.usable:
        raise Error("mamba1 session: lost after a failed call; close it and open a new one")


def _m1_session_open_run(mut s: Mamba1DecodeSession, a: List[Int], b: Int, dm: Int, position: Int = 0) raises:
    """The GIL-free half of `mamba1_session_open`: the host weight lists
    through the lane's own struct, one context, the device weights, the
    state over fresh zeros, the L = 1 stages and the resident x buffer."""
    var dims = MambaDims.of(dm)
    var di = dims.d_inner
    var r = dims.dt_rank
    var xr = dims.x_proj_rows()
    var w = MambaWeights(dims)
    w.norm_w = _read_f32(a[1], dm)
    w.w_in = _read_f32(a[2], 2 * di * dm)
    w.conv_w = _read_f32(a[3], di * D_CONV)
    w.conv_b = _read_f32(a[4], di)
    w.w_x = _read_f32(a[5], xr * di)
    w.w_dt = _read_f32(a[6], di * r)
    w.b_dt = _read_f32(a[7], di)
    w.a_log = _read_f32(a[8], di * D_STATE)
    w.d_skip = _read_f32(a[9], di)
    w.w_out = _read_f32(a[10], dm * di)
    s.ctx = neural_ctx[_NEURAL_CTX]()
    s.w = MambaDeviceWeights(s.ctx.value(), w)
    var dstate = MambaDeviceState(s.ctx.value(), b, dims)
    dstate.conv_win = mamba_upload(s.ctx.value(), _read_f32(a[11], b * di * D_CONV))
    dstate.h = mamba_upload(s.ctx.value(), _read_f32(a[12], b * di * D_STATE))
    s.state = dstate^
    s.stages = MambaDeviceStages(s.ctx.value(), b, 1, dims)
    s.dx = mamba_zeros(s.ctx.value(), b * dm)
    s.b = b
    s.dm = dm
    s.di = di


def _m1_session_step_run(mut s: Mamba1DecodeSession, px: Int, py: Int) raises:
    """One decode token on the resident structs, one completion wait."""
    ref ctx = s.ctx.value()
    var n = s.b * s.dm
    mamba_copy_in(ctx, s.dx.value(), f32_ptr(px), n)
    var trace = IdentityTrace.disabled()
    mamba_step(ctx, s.stages.value(), s.state.value(), s.w.value(), s.dx.value(), s.b, trace,
               String("py.session"))
    comptime if MAMBA_GUARD > 0:
        var body = s.stages.value().residual_out.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=f32_ptr(py), src_buf=body)
        ctx.synchronize()
        _ = body^
    else:
        if n != len(s.stages.value().residual_out):
            raise Error("mamba1 session: the L = 1 stages hold a different output length")
        ctx.enqueue_copy(dst_ptr=f32_ptr(py), src_buf=s.stages.value().residual_out)
        ctx.synchronize()


def _m1_session_export_run(mut s: Mamba1DecodeSession, pw: Int, ph: Int) raises:
    ref ctx = s.ctx.value()
    _write_f32(pw, mamba_download(ctx, s.state.value().conv_win, s.b * s.di * D_CONV))
    _write_f32(ph, mamba_download(ctx, s.state.value().h, s.b * s.di * D_STATE))


def _m1_session_load_run(mut s: Mamba1DecodeSession, pw: Int, ph: Int) raises:
    ref ctx = s.ctx.value()
    s.state.value().conv_win = mamba_upload(ctx, _read_f32(pw, s.b * s.di * D_CONV))
    s.state.value().h = mamba_upload(ctx, _read_f32(ph, s.b * s.di * D_STATE))


# ===========================================================================
# Mamba-2: profile mojolearn.identical.mamba2.fp32.v1
# ===========================================================================


def _mamba2_run(
    a: List[Int],
    b: Int,
    l: Int,
    dm: Int,
    q0: Int,
    dt_lo: Float32,
    dt_hi: Float32,
) raises -> Int:
    """The GIL-free half of the two Mamba-2 entry points. Returns the
    post-call `buf_len` (DEVIATION 792: the one integer piece of the
    three-piece state)."""
    comptime if AFN_MAMBA_ARENA:
        return _mamba2_run_arena(a, b, l, dm, q0, dt_lo, dt_hi)
    # Mamba2Dims.of REFUSES a d_model that is not a multiple of 32, by
    # name, with the profile rule -- reached BEFORE any buffer size below
    # is computed from it, which is why nothing here pre-judges dm.
    var dims = Mamba2Dims.of(dm)
    var di = dims.d_inner
    var cd = dims.conv_dim()
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    if q0 < 0 or q0 >= M2_CHUNK_SIZE:
        raise Error(
            String("mamba2: buf_len must be in [0, ")
            + String(M2_CHUNK_SIZE)
            + "), got "
            + String(q0)
            + "; the open-chunk buffer holds at most CHUNK_SIZE - 1 rows"
            " between calls (contract section 5), so the two sides of this"
            " boundary disagree about the state"
        )

    var w = Mamba2Weights(dims)
    w.norm_w = _read_f32(a[1], dm)
    w.w_in = _read_f32(a[2], dip * dm)
    w.conv_w = _read_f32(a[3], cd * M2_D_CONV)
    w.conv_b = _read_f32(a[4], cd)
    w.dt_bias = _read_f32(a[5], nh)
    w.a_log = _read_f32(a[6], nh)
    w.d_skip = _read_f32(a[7], nh)
    w.gnorm_w = _read_f32(a[8], di)
    w.w_out = _read_f32(a[9], dm * di)

    var h_n = b * nh * M2_HEADDIM * M2_D_STATE

    var ctx = neural_ctx[_NEURAL_CTX]()
    var dw = Mamba2DeviceWeights(ctx, w)
    # The caller's three-piece state over the fresh zeros. A nonzero h
    # with q0 == 0 IS the initial_states path (module header, DEVIATION
    # 792).
    var dstate = Mamba2DeviceState(ctx, b, dims)
    # Keep the constructor's state allocations. Replacing these fields
    # triggers an AMD HIP memory fault on the B2/L4/DM32 Python path;
    # copying the same incoming bytes into the retained buffers passes
    # the forward, decode and carried-state gates without changing math.
    var incoming_conv = mamba_upload(ctx, _read_f32(a[10], b * cd * M2_D_CONV))
    var incoming_h = mamba_upload(ctx, _read_f32(a[11], h_n))
    var incoming_xbc = mamba_upload(ctx, _read_f32(a[12], b * M2_CHUNK_SIZE * cd))
    var incoming_dt = mamba_upload(ctx, _read_f32(a[13], b * M2_CHUNK_SIZE * nh))
    ctx.enqueue_copy(dst_buf=dstate.conv_win, src_buf=incoming_conv)
    ctx.enqueue_copy(dst_buf=dstate.h, src_buf=incoming_h)
    ctx.enqueue_copy(dst_buf=dstate.buf_xbc, src_buf=incoming_xbc)
    ctx.enqueue_copy(dst_buf=dstate.buf_dtraw, src_buf=incoming_dt)
    ctx.synchronize()
    _ = incoming_conv^
    _ = incoming_h^
    _ = incoming_xbc^
    _ = incoming_dt^
    dstate.buf_len = q0
    var dstages = Mamba2DeviceStages(ctx, b, l, q0, dims)
    var dx = mamba_upload(ctx, _read_f32(a[0], b * l * dm))

    var trace = IdentityTrace.disabled()
    mamba2_block_forward(
        ctx, dstages, dstate, dw, dx, b, l, dt_lo, dt_hi, trace, String("py")
    )

    # Straight into the caller's arrays, one wait (lane/neural-apple,
    # 2026-09-28; was a host List per piece, a wait and a second copy each).
    _m3_download_addr[False](ctx, dstages.residual_out, b * l * dm, a[14])
    # h_last: the REPORT stage (S17 after the final PADDED chunk), NOT
    # the resumption state -- contract section 5's distinction, kept.
    _m3_download_addr[False](ctx, dstages.h_last, h_n, a[15])
    _m3_download_addr[False](ctx, dstate.conv_win, b * cd * M2_D_CONV, a[10])
    _m3_download_addr[False](ctx, dstate.h, h_n, a[11])
    _m3_download_addr[False](ctx, dstate.buf_xbc, b * M2_CHUNK_SIZE * cd, a[12])
    _m3_download_addr[False](ctx, dstate.buf_dtraw, b * M2_CHUNK_SIZE * nh, a[13])
    ctx.synchronize()
    var out_len = dstate.buf_len
    _ = dw^
    _ = dstate^
    _ = dstages^
    _ = dx^
    _ = ctx^
    return out_len


def _mamba2_run_arena(
    a: List[Int],
    b: Int,
    l: Int,
    dm: Int,
    q0: Int,
    dt_lo: Float32,
    dt_hi: Float32,
) raises -> Int:
    """`_mamba2_run` under MOJOLEARN_AFN_MAMBA_ARENA (lane afn-mamba): one
    arena, views, direct copies in, ONE wait at the end."""
    var dims = Mamba2Dims.of(dm)
    var di = dims.d_inner
    var cd = dims.conv_dim()
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    if q0 < 0 or q0 >= M2_CHUNK_SIZE:
        raise Error(
            String("mamba2: buf_len must be in [0, ")
            + String(M2_CHUNK_SIZE)
            + "), got "
            + String(q0)
            + "; the open-chunk buffer holds at most CHUNK_SIZE - 1 rows"
            " between calls (contract section 5), so the two sides of this"
            " boundary disagree about the state"
        )
    var h_n = b * nh * M2_HEADDIM * M2_D_STATE
    var ctx = neural_ctx[_NEURAL_CTX]()
    var arena = MambaArena(ctx, mamba2_arena_floats(b, l, q0, dims), MAMBA_GUARD)
    var dw = Mamba2DeviceWeights(ctx, dims, arena)
    mamba_copy_in(ctx, dw.norm_w, _f32_ptr(a[1]), dm)
    mamba_copy_in(ctx, dw.w_in, _f32_ptr(a[2]), dip * dm)
    mamba_copy_in(ctx, dw.conv_w, _f32_ptr(a[3]), cd * M2_D_CONV)
    mamba_copy_in(ctx, dw.conv_b, _f32_ptr(a[4]), cd)
    mamba_copy_in(ctx, dw.dt_bias, _f32_ptr(a[5]), nh)
    mamba_copy_in(ctx, dw.a_log, _f32_ptr(a[6]), nh)
    mamba_copy_in(ctx, dw.d_skip, _f32_ptr(a[7]), nh)
    mamba_copy_in(ctx, dw.gnorm_w, _f32_ptr(a[8]), di)
    mamba_copy_in(ctx, dw.w_out, _f32_ptr(a[9]), dm * di)
    var dstate = Mamba2DeviceState(ctx, b, dims, arena)
    mamba_copy_in(ctx, dstate.conv_win, _f32_ptr(a[10]), b * cd * M2_D_CONV)
    mamba_copy_in(ctx, dstate.h, _f32_ptr(a[11]), h_n)
    mamba_copy_in(ctx, dstate.buf_xbc, _f32_ptr(a[12]), b * M2_CHUNK_SIZE * cd)
    mamba_copy_in(ctx, dstate.buf_dtraw, _f32_ptr(a[13]), b * M2_CHUNK_SIZE * nh)
    dstate.buf_len = q0
    var dstages = Mamba2DeviceStages(ctx, b, l, q0, dims, arena)
    var dx = arena.take(ctx, b * l * dm)
    mamba_copy_in(ctx, dx, _f32_ptr(a[0]), b * l * dm)

    var trace = IdentityTrace.disabled()
    mamba2_block_forward(
        ctx, dstages, dstate, dw, dx, b, l, dt_lo, dt_hi, trace, String("py")
    )

    _m3_download_addr[False](ctx, dstages.residual_out, b * l * dm, a[14])
    _m3_download_addr[False](ctx, dstages.h_last, h_n, a[15])
    _m3_download_addr[False](ctx, dstate.conv_win, b * cd * M2_D_CONV, a[10])
    _m3_download_addr[False](ctx, dstate.h, h_n, a[11])
    _m3_download_addr[False](ctx, dstate.buf_xbc, b * M2_CHUNK_SIZE * cd, a[12])
    _m3_download_addr[False](ctx, dstate.buf_dtraw, b * M2_CHUNK_SIZE * nh, a[13])
    ctx.synchronize()
    var out_len = dstate.buf_len
    _ = dw^
    _ = dstate^
    _ = dstages^
    _ = dx^
    _ = arena^
    _ = ctx^
    return out_len


# Resident Mamba-2 decode.  The arithmetic entry remains
# `mamba2_block_forward(..., L=1)`.  Only the device context, immutable
# weights, recurrent state, and input allocation survive between calls.
# Stages are rebuilt because their working extents depend on buf_len.
struct Mamba2DecodeSession(Movable, Writable):
    var ctx: Optional[DeviceContext]
    var w: Optional[Mamba2DeviceWeights]
    var state: Optional[Mamba2DeviceState]
    var dx: Optional[DeviceBuffer[DType.float32]]
    var b: Int
    var dm: Int
    var dt_lo: Float32
    var dt_hi: Float32
    var busy: Bool
    var usable: Bool

    def __init__(out self):
        self.ctx = None
        self.w = None
        self.state = None
        self.dx = None
        self.b = 0
        self.dm = 0
        self.dt_lo = 0.0
        self.dt_hi = 0.0
        self.busy = False
        self.usable = True

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Mamba2DecodeSession")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("Mamba2DecodeSession")

    def is_open(self) -> Bool:
        return Bool(self.ctx) and Bool(self.state)

    def release(mut self):
        self.dx = None
        self.state = None
        self.w = None
        if self.ctx:
            try:
                self.ctx.value().synchronize()
            except:
                pass
        self.ctx = None

    def __deinit__(deinit self):
        self.release()

    def close(mut self) raises:
        if self.busy:
            raise Error("mamba2 session: busy")
        self.usable = False
        self.release()


def _m2_require_session_open(s: Mamba2DecodeSession) raises:
    if s.busy:
        raise Error("mamba2 session: busy")
    if not s.is_open():
        raise Error("mamba2 session: not open")
    if not s.usable:
        raise Error("mamba2 session: lost after a failed call")


def _m2_session_open_run(
    mut s: Mamba2DecodeSession, a: List[Int], b: Int, dm: Int, q0: Int,
    dt_lo: Float32, dt_hi: Float32,
) raises:
    var dims = Mamba2Dims.of(dm)
    var di = dims.d_inner
    var cd = dims.conv_dim()
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    if q0 < 0 or q0 >= M2_CHUNK_SIZE:
        raise Error("mamba2_session_open: buf_len out of range")
    var hw = Mamba2Weights(dims)
    hw.norm_w = _read_f32(a[0], dm)
    hw.w_in = _read_f32(a[1], dip * dm)
    hw.conv_w = _read_f32(a[2], cd * M2_D_CONV)
    hw.conv_b = _read_f32(a[3], cd)
    hw.dt_bias = _read_f32(a[4], nh)
    hw.a_log = _read_f32(a[5], nh)
    hw.d_skip = _read_f32(a[6], nh)
    hw.gnorm_w = _read_f32(a[7], di)
    hw.w_out = _read_f32(a[8], dm * di)
    s.ctx = neural_ctx[_NEURAL_CTX]()
    ref ctx = s.ctx.value()
    s.w = Mamba2DeviceWeights(ctx, hw)
    s.state = Mamba2DeviceState(ctx, b, dims)
    ref state = s.state.value()
    mamba_copy_in(ctx, state.conv_win, f32_ptr(a[9]), b * cd * M2_D_CONV)
    mamba_copy_in(ctx, state.h, f32_ptr(a[10]), b * nh * M2_HEADDIM * M2_D_STATE)
    mamba_copy_in(ctx, state.buf_xbc, f32_ptr(a[11]), b * M2_CHUNK_SIZE * cd)
    mamba_copy_in(ctx, state.buf_dtraw, f32_ptr(a[12]), b * M2_CHUNK_SIZE * nh)
    ctx.synchronize()
    state.buf_len = q0
    s.dx = mamba_zeros(ctx, b * dm)
    s.b = b
    s.dm = dm
    s.dt_lo = dt_lo
    s.dt_hi = dt_hi


def _m2_session_step_run(mut s: Mamba2DecodeSession, px: Int, py: Int, ph: Int) raises -> Int:
    ref ctx = s.ctx.value()
    ref state = s.state.value()
    var dims = Mamba2Dims.of(s.dm)
    var stages = Mamba2DeviceStages(ctx, s.b, 1, state.buf_len, dims)
    mamba_copy_in(ctx, s.dx.value(), f32_ptr(px), s.b * s.dm)
    var trace = IdentityTrace.disabled()
    mamba2_block_forward(ctx, stages, state, s.w.value(), s.dx.value(), s.b, 1,
                         s.dt_lo, s.dt_hi, trace, String("py.session"))
    _write_f32(py, mamba_download(ctx, stages.residual_out, s.b * s.dm))
    _write_f32(ph, mamba_download(ctx, stages.h_last,
        s.b * dims.nheads * M2_HEADDIM * M2_D_STATE))
    return state.buf_len


def _m2_session_export_run(mut s: Mamba2DecodeSession, a: List[Int]) raises -> Int:
    ref ctx = s.ctx.value()
    ref state = s.state.value()
    var dims = Mamba2Dims.of(s.dm)
    var cd = dims.conv_dim()
    var nh = dims.nheads
    _write_f32(a[0], mamba_download(ctx, state.conv_win, s.b * cd * M2_D_CONV))
    _write_f32(a[1], mamba_download(ctx, state.h, s.b * nh * M2_HEADDIM * M2_D_STATE))
    _write_f32(a[2], mamba_download(ctx, state.buf_xbc, s.b * M2_CHUNK_SIZE * cd))
    _write_f32(a[3], mamba_download(ctx, state.buf_dtraw, s.b * M2_CHUNK_SIZE * nh))
    return state.buf_len


def _m2_session_load_run(mut s: Mamba2DecodeSession, a: List[Int], q0: Int) raises:
    if q0 < 0 or q0 >= M2_CHUNK_SIZE:
        raise Error("mamba2_session_load_state: buf_len out of range")
    ref ctx = s.ctx.value()
    ref state = s.state.value()
    var dims = Mamba2Dims.of(s.dm)
    var cd = dims.conv_dim()
    var nh = dims.nheads
    # Retain constructor-owned buffers; replacing them faults on AMD.
    mamba_copy_in(ctx, state.conv_win, f32_ptr(a[0]), s.b * cd * M2_D_CONV)
    mamba_copy_in(ctx, state.h, f32_ptr(a[1]), s.b * nh * M2_HEADDIM * M2_D_STATE)
    mamba_copy_in(ctx, state.buf_xbc, f32_ptr(a[2]), s.b * M2_CHUNK_SIZE * cd)
    mamba_copy_in(ctx, state.buf_dtraw, f32_ptr(a[3]), s.b * M2_CHUNK_SIZE * nh)
    ctx.synchronize()
    state.buf_len = q0


# ===========================================================================
# Mamba-3: profile mojolearn.identical.mamba3.siso.fp32.v1
# ===========================================================================


def _m3_load_weights[wait: Bool = True](ctx: DeviceContext, a: List[Int], dims: Mamba3Dims) raises -> Mamba3DeviceWeights:
    var dm = dims.d_model
    var di = dims.d_inner
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    comptime if M3_DIRECT_TRANSFER:
        var norm_w = _m3_upload_addr[wait](ctx, a[1], dm)
        var w_in = _m3_upload_addr[wait](ctx, a[2], dip * dm)
        var dt_bias = _m3_upload_addr[wait](ctx, a[3], nh)
        var bnorm_w = _m3_upload_addr[wait](ctx, a[4], M3_D_STATE)
        var cnorm_w = _m3_upload_addr[wait](ctx, a[5], M3_D_STATE)
        var b_bias = _m3_upload_addr[wait](ctx, a[6], nh * M3_D_STATE)
        var c_bias = _m3_upload_addr[wait](ctx, a[7], nh * M3_D_STATE)
        var d_skip = _m3_upload_addr[wait](ctx, a[8], nh)
        var w_out = _m3_upload_addr[wait](ctx, a[9], dm * di)
        return Mamba3DeviceWeights(dims, norm_w^, w_in^, dt_bias^, bnorm_w^, cnorm_w^, b_bias^, c_bias^, d_skip^, w_out^)
    else:
        var w = Mamba3Weights(dims)
        w.norm_w = _m3_read_f32(a[1], dm)
        w.w_in = _m3_read_f32(a[2], dip * dm)
        w.dt_bias = _m3_read_f32(a[3], nh)
        w.bnorm_w = _m3_read_f32(a[4], M3_D_STATE)
        w.cnorm_w = _m3_read_f32(a[5], M3_D_STATE)
        w.b_bias = _m3_read_f32(a[6], nh * M3_D_STATE)
        w.c_bias = _m3_read_f32(a[7], nh * M3_D_STATE)
        w.d_skip = _m3_read_f32(a[8], nh)
        w.w_out = _m3_read_f32(a[9], dm * di)
        return Mamba3DeviceWeights(ctx, w)


def _mamba3_run[discard_state: Bool = False](
    a: List[Int], b: Int, l: Int, dm: Int, q0: Int, pend: Int
) raises -> Int:
    """The GIL-free half of the two Mamba-3 entry points. Returns the
    post-call `buf_len` (DEVIATION 794: `pending` needs no return slot
    because a shipped binding always consumes it)."""
    comptime if AFN_MAMBA_ARENA:
        return _mamba3_run_arena[discard_state](a, b, l, dm, q0, pend)
    # Mamba3Dims.of REFUSES a d_model that is not a multiple of 32, by
    # name -- reached BEFORE any buffer size below is computed from it.
    var dims = Mamba3Dims.of(dm)
    var di = dims.d_inner
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    if q0 < 0 or q0 > M3_CHUNK_SIZE:
        raise Error(
            String("mamba3: buf_len must be in [0, ")
            + String(M3_CHUNK_SIZE)
            + "] (INCLUSIVE -- the buffer never empties, DEVIATION"
            " 832(i): r in [1, Q] after every call, 0 only before the"
            " first token), got "
            + String(q0)
            + "; the two sides of this boundary disagree about the state"
        )
    if pend != 0 and pend != 1:
        raise Error(
            "mamba3: pending must be 0 or 1, got "
            + String(pend)
            + "; the two sides of this boundary disagree about the state"
        )

    var phase_tick = 0
    comptime if is_defined["MOJOLEARN_MAMBA3_PHASE_TIMERS"]():
        phase_tick = Int(perf_counter_ns())

    var theta_n = b * nh * M3_NUM_ROPE_ANGLES
    var h_n = b * nh * M3_HEADDIM * M3_D_STATE
    var qrow_n = b * M3_CHUNK_SIZE * nh
    var k_n = b * nh * M3_D_STATE
    var v_n = b * nh * M3_HEADDIM

    var ctx = neural_ctx[_NEURAL_CTX]()
    m3_phase_tick(ctx, phase_tick, String("surface.weight_lists_and_context"))
    var dw = _m3_load_weights[False](ctx, a, dims)
    m3_phase_tick(ctx, phase_tick, String("surface.weight_upload"))
    var out_len = _mamba3_run_on[discard_state](ctx, dw, a, b, l, dm, q0, pend, phase_tick)
    _ = dw^
    _ = ctx^
    comptime if is_defined["MOJOLEARN_MAMBA3_PHASE_TIMERS"]():
        print("M3_PHASE", "surface.context_destruction", Float64(Int(perf_counter_ns()) - phase_tick) / 1e6)
    return out_len


def _mamba3_check_state_words(q0: Int, pend: Int) raises:
    if q0 < 0 or q0 > M3_CHUNK_SIZE:
        raise Error(
            String("mamba3: buf_len must be in [0, ")
            + String(M3_CHUNK_SIZE)
            + "] (INCLUSIVE -- the buffer never empties, DEVIATION"
            " 832(i): r in [1, Q] after every call, 0 only before the"
            " first token), got "
            + String(q0)
            + "; the two sides of this boundary disagree about the state"
        )
    if pend != 0 and pend != 1:
        raise Error(
            "mamba3: pending must be 0 or 1, got "
            + String(pend)
            + "; the two sides of this boundary disagree about the state"
        )


def _mamba3_run_owned(mut s: Mamba3PrefillSession, a: List[Int], b: Int, l: Int, dm: Int, q0: Int, pend: Int) raises -> Int:
    """`_mamba3_run` (the carried-state forward or decode step) reading the
    session's installed owned weights where they live on the device
    (lane migrate-neural-2: the state and decode calls uploaded the host
    weights again although the snapshot was resident). `a` is the 25-slot
    list; its nine weight slots are not read. The same block on the same
    weight bytes, so the same bits."""
    var dims = Mamba3Dims.of(dm)
    _mamba3_check_state_words(q0, pend)
    comptime if not NN40_OWNED_WEIGHTS:
        raise Error("mamba3 owned weights: NN40 arm is disabled")
    if not s.owned_weights or not s.w or s.dm != dm:
        raise Error("mamba3 owned weights: install a matching snapshot first")
    if not s.ctx:
        s.ctx = neural_ctx[_NEURAL_CTX]()
    var phase_tick = 0
    comptime if is_defined["MOJOLEARN_MAMBA3_PHASE_TIMERS"]():
        phase_tick = Int(perf_counter_ns())
    ref ctx = s.ctx.value()
    return _mamba3_run_on[False](ctx, s.w.value(), a, b, l, dm, q0, pend, phase_tick)


def _mamba3_run_on[discard_state: Bool = False](
    ctx: DeviceContext, mut dw: Mamba3DeviceWeights, a: List[Int], b: Int, l: Int, dm: Int, q0: Int, pend: Int,
    var phase_tick: Int,
) raises -> Int:
    """`_mamba3_run` after its weights are on the device: the state upload,
    the block, the downloads and the one wait."""
    var dims = Mamba3Dims.of(dm)
    var nh = dims.nheads
    var theta_n = b * nh * M3_NUM_ROPE_ANGLES
    var h_n = b * nh * M3_HEADDIM * M3_D_STATE
    var qrow_n = b * M3_CHUNK_SIZE * nh
    var k_n = b * nh * M3_D_STATE
    var v_n = b * nh * M3_HEADDIM
    # The caller's ten-piece state over the fresh zeros (DEVIATION 794).
    var dstate = Mamba3DeviceState(ctx, b, dims)
    comptime if not discard_state:
        dstate.buf_qrot = _m3_upload_addr[False](ctx, a[12], qrow_n * M3_D_STATE)
        dstate.buf_krot = _m3_upload_addr[False](ctx, a[13], qrow_n * M3_D_STATE)
        dstate.buf_v = _m3_upload_addr[False](ctx, a[14], qrow_n * M3_HEADDIM)
        dstate.buf_dt = _m3_upload_addr[False](ctx, a[15], qrow_n)
        dstate.buf_sig = _m3_upload_addr[False](ctx, a[16], qrow_n)
        dstate.buf_adt = _m3_upload_addr[False](ctx, a[17], qrow_n)
        dstate.buf_len = q0
        if pend == 1:
            # THROUGH the lane's own helper, AFTER buf_len is set, so its
            # fresh-state refusal (buf_len must be 0) fires from Python by
            # the lane's own words (DEVIATION 794).
            dstate.set_input_states(
                ctx,
                _m3_read_f32(a[10], theta_n),
                _m3_read_f32(a[11], h_n),
                _m3_read_f32(a[18], k_n),
                _m3_read_f32(a[19], v_n),
            )
        else:
            dstate.theta = _m3_upload_addr[False](ctx, a[10], theta_n)
            dstate.h = _m3_upload_addr[False](ctx, a[11], h_n)
            # Idle outside a pending continuation, uploaded anyway so the
            # caller's bytes round-trip unchanged (DEVIATION 792's rule).
            dstate.pend_k = _m3_upload_addr[False](ctx, a[18], k_n)
            dstate.pend_v = _m3_upload_addr[False](ctx, a[19], v_n)
    m3_phase_tick(ctx, phase_tick, String("surface.state_upload"))
    var dstages = Mamba3DeviceStages(ctx, b, l, q0, dims)
    m3_phase_tick(ctx, phase_tick, String("surface.stage_allocations"))
    var dx = _m3_upload_addr[False](ctx, a[0], b * l * dm)

    m3_phase_tick(ctx, phase_tick, String("surface.x_upload"))
    var trace = IdentityTrace.disabled()
    mamba3_block_forward(
        ctx, dstages, dstate, dw, dx, b, l, trace, String("py")
    )

    m3_phase_tick(ctx, phase_tick, String("surface.block"))
    _m3_download_addr[False](ctx, dstages.residual_out, b * l * dm, a[20])
    # The FOUR reports (contract section 7's ssd.* stages), NOT the
    # resumption state -- DEVIATION 794's last clause.
    _m3_download_addr[False](ctx, dstages.h_last, h_n, a[21])
    _m3_download_addr[False](ctx, dstages.k_last, k_n, a[22])
    _m3_download_addr[False](ctx, dstages.v_last, v_n, a[23])
    _m3_download_addr[False](ctx, dstages.theta_last, theta_n, a[24])
    comptime if not discard_state:
        _m3_download_addr[False](ctx, dstate.theta, theta_n, a[10])
        _m3_download_addr[False](ctx, dstate.h, h_n, a[11])
        _m3_download_addr[False](ctx, dstate.buf_qrot, qrow_n * M3_D_STATE, a[12])
        _m3_download_addr[False](ctx, dstate.buf_krot, qrow_n * M3_D_STATE, a[13])
        _m3_download_addr[False](ctx, dstate.buf_v, qrow_n * M3_HEADDIM, a[14])
        _m3_download_addr[False](ctx, dstate.buf_dt, qrow_n, a[15])
        _m3_download_addr[False](ctx, dstate.buf_sig, qrow_n, a[16])
        _m3_download_addr[False](ctx, dstate.buf_adt, qrow_n, a[17])
        _m3_download_addr[False](ctx, dstate.pend_k, k_n, a[18])
        _m3_download_addr[False](ctx, dstate.pend_v, v_n, a[19])
    # One wait for every download above (and anything still in flight).
    ctx.synchronize()
    m3_phase_tick(ctx, phase_tick, String("surface.downloads"))
    var out_len = dstate.buf_len
    _ = dstate^
    _ = dstages^
    _ = dx^
    m3_phase_tick(ctx, phase_tick, String("surface.buffer_destruction"))
    return out_len


def _mamba3_run_arena[discard_state: Bool = False](
    a: List[Int], b: Int, l: Int, dm: Int, q0: Int, pend: Int
) raises -> Int:
    """`_mamba3_run` under MOJOLEARN_AFN_MAMBA_ARENA (lane afn-mamba): one
    arena, views, direct copies in, ONE wait at the end."""
    var dims = Mamba3Dims.of(dm)
    var di = dims.d_inner
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    if q0 < 0 or q0 > M3_CHUNK_SIZE:
        raise Error(
            String("mamba3: buf_len must be in [0, ")
            + String(M3_CHUNK_SIZE)
            + "] (INCLUSIVE -- the buffer never empties, DEVIATION"
            " 832(i): r in [1, Q] after every call, 0 only before the"
            " first token), got "
            + String(q0)
            + "; the two sides of this boundary disagree about the state"
        )
    if pend != 0 and pend != 1:
        raise Error(
            "mamba3: pending must be 0 or 1, got "
            + String(pend)
            + "; the two sides of this boundary disagree about the state"
        )
    var theta_n = b * nh * M3_NUM_ROPE_ANGLES
    var h_n = b * nh * M3_HEADDIM * M3_D_STATE
    var qrow_n = b * M3_CHUNK_SIZE * nh
    var k_n = b * nh * M3_D_STATE
    var v_n = b * nh * M3_HEADDIM

    var ctx = neural_ctx[_NEURAL_CTX]()
    var arena = MambaArena(ctx, mamba3_arena_floats(b, l, q0, dims), MAMBA_GUARD)
    var dw = Mamba3DeviceWeights(ctx, dims, arena)
    mamba_copy_in(ctx, dw.norm_w, _f32_ptr(a[1]), dm)
    mamba_copy_in(ctx, dw.w_in, _f32_ptr(a[2]), dip * dm)
    mamba_copy_in(ctx, dw.dt_bias, _f32_ptr(a[3]), nh)
    mamba_copy_in(ctx, dw.bnorm_w, _f32_ptr(a[4]), M3_D_STATE)
    mamba_copy_in(ctx, dw.cnorm_w, _f32_ptr(a[5]), M3_D_STATE)
    mamba_copy_in(ctx, dw.b_bias, _f32_ptr(a[6]), nh * M3_D_STATE)
    mamba_copy_in(ctx, dw.c_bias, _f32_ptr(a[7]), nh * M3_D_STATE)
    mamba_copy_in(ctx, dw.d_skip, _f32_ptr(a[8]), nh)
    mamba_copy_in(ctx, dw.w_out, _f32_ptr(a[9]), dm * di)
    var dstate = Mamba3DeviceState(ctx, b, dims, arena)
    comptime if not discard_state:
        mamba_copy_in(ctx, dstate.buf_qrot, _f32_ptr(a[12]), qrow_n * M3_D_STATE)
        mamba_copy_in(ctx, dstate.buf_krot, _f32_ptr(a[13]), qrow_n * M3_D_STATE)
        mamba_copy_in(ctx, dstate.buf_v, _f32_ptr(a[14]), qrow_n * M3_HEADDIM)
        mamba_copy_in(ctx, dstate.buf_dt, _f32_ptr(a[15]), qrow_n)
        mamba_copy_in(ctx, dstate.buf_sig, _f32_ptr(a[16]), qrow_n)
        mamba_copy_in(ctx, dstate.buf_adt, _f32_ptr(a[17]), qrow_n)
        dstate.buf_len = q0
        if pend == 1:
            # THROUGH the lane's own helper (its fresh-state refusal fires
            # by the lane's own words); it copies into the views.
            dstate.set_input_states(
                ctx,
                _m3_read_f32(a[10], theta_n),
                _m3_read_f32(a[11], h_n),
                _m3_read_f32(a[18], k_n),
                _m3_read_f32(a[19], v_n),
            )
        else:
            mamba_copy_in(ctx, dstate.theta, _f32_ptr(a[10]), theta_n)
            mamba_copy_in(ctx, dstate.h, _f32_ptr(a[11]), h_n)
            mamba_copy_in(ctx, dstate.pend_k, _f32_ptr(a[18]), k_n)
            mamba_copy_in(ctx, dstate.pend_v, _f32_ptr(a[19]), v_n)
    var dstages = Mamba3DeviceStages(ctx, b, l, q0, dims, arena)
    var dx = arena.take(ctx, b * l * dm)
    mamba_copy_in(ctx, dx, _f32_ptr(a[0]), b * l * dm)

    var trace = IdentityTrace.disabled()
    mamba3_block_forward(
        ctx, dstages, dstate, dw, dx, b, l, trace, String("py")
    )

    _m3_download_addr[False](ctx, dstages.residual_out, b * l * dm, a[20])
    _m3_download_addr[False](ctx, dstages.h_last, h_n, a[21])
    _m3_download_addr[False](ctx, dstages.k_last, k_n, a[22])
    _m3_download_addr[False](ctx, dstages.v_last, v_n, a[23])
    _m3_download_addr[False](ctx, dstages.theta_last, theta_n, a[24])
    comptime if not discard_state:
        _m3_download_addr[False](ctx, dstate.theta, theta_n, a[10])
        _m3_download_addr[False](ctx, dstate.h, h_n, a[11])
        _m3_download_addr[False](ctx, dstate.buf_qrot, qrow_n * M3_D_STATE, a[12])
        _m3_download_addr[False](ctx, dstate.buf_krot, qrow_n * M3_D_STATE, a[13])
        _m3_download_addr[False](ctx, dstate.buf_v, qrow_n * M3_HEADDIM, a[14])
        _m3_download_addr[False](ctx, dstate.buf_dt, qrow_n, a[15])
        _m3_download_addr[False](ctx, dstate.buf_sig, qrow_n, a[16])
        _m3_download_addr[False](ctx, dstate.buf_adt, qrow_n, a[17])
        _m3_download_addr[False](ctx, dstate.pend_k, k_n, a[18])
        _m3_download_addr[False](ctx, dstate.pend_v, v_n, a[19])
    ctx.synchronize()
    var out_len = dstate.buf_len
    _ = dw^
    _ = dstate^
    _ = dstages^
    _ = dx^
    _ = arena^
    _ = ctx^
    return out_len


# Resident Mamba-3 decode, parallel to Mamba-1/2.  Stages are rebuilt at
# each step because their working dimensions include the carried buf_len;
# context, weights, ten-piece state, and x allocation remain resident.
struct Mamba3DecodeSession(Movable, Writable):
    var ctx: Optional[DeviceContext]
    var w: Optional[Mamba3DeviceWeights]
    var state: Optional[Mamba3DeviceState]
    var dx: Optional[DeviceBuffer[DType.float32]]
    var decode_stages: Optional[Mamba3DeviceStages]
    var b: Int
    var dm: Int
    var busy: Bool
    var usable: Bool

    def __init__(out self):
        self.ctx = None
        self.w = None
        self.state = None
        self.dx = None
        self.decode_stages = None
        self.b = 0
        self.dm = 0
        self.busy = False
        self.usable = True

    def write_to(self, mut writer: Some[Writer]): writer.write("Mamba3DecodeSession")
    def write_repr_to(self, mut writer: Some[Writer]): writer.write("Mamba3DecodeSession")
    def is_open(self) -> Bool: return Bool(self.ctx) and Bool(self.state)

    def release(mut self):
        self.decode_stages = None
        self.dx = None
        self.state = None
        self.w = None
        if self.ctx:
            try: self.ctx.value().synchronize()
            except: pass
        self.ctx = None

    def __deinit__(deinit self): self.release()
    def close(mut self) raises:
        if self.busy: raise Error("mamba3 session: busy")
        self.usable = False
        self.release()


def _m3_require_session_open(s: Mamba3DecodeSession) raises:
    if s.busy: raise Error("mamba3 session: busy")
    if not s.is_open(): raise Error("mamba3 session: not open")
    if not s.usable: raise Error("mamba3 session: lost after a failed call")


def _m3_session_copy_in(mut s: Mamba3DecodeSession, a: List[Int], q0: Int, pend: Int) raises:
    if q0 < 0 or q0 > M3_CHUNK_SIZE:
        raise Error("mamba3 session: buf_len out of range")
    if pend != 0 and pend != 1:
        raise Error("mamba3 session: pending must be 0 or 1")
    if pend == 1 and q0 != 0:
        raise Error("mamba3 session: Input_States pending requires a fresh state")
    ref ctx = s.ctx.value()
    ref st = s.state.value()
    var dims = Mamba3Dims.of(s.dm)
    var nh = dims.nheads
    var theta_n = s.b * nh * M3_NUM_ROPE_ANGLES
    var h_n = s.b * nh * M3_HEADDIM * M3_D_STATE
    var qrow_n = s.b * M3_CHUNK_SIZE * nh
    var k_n = s.b * nh * M3_D_STATE
    var v_n = s.b * nh * M3_HEADDIM
    mamba_copy_in(ctx, st.theta, f32_ptr(a[0]), theta_n)
    mamba_copy_in(ctx, st.h, f32_ptr(a[1]), h_n)
    mamba_copy_in(ctx, st.buf_qrot, f32_ptr(a[2]), qrow_n * M3_D_STATE)
    mamba_copy_in(ctx, st.buf_krot, f32_ptr(a[3]), qrow_n * M3_D_STATE)
    mamba_copy_in(ctx, st.buf_v, f32_ptr(a[4]), qrow_n * M3_HEADDIM)
    mamba_copy_in(ctx, st.buf_dt, f32_ptr(a[5]), qrow_n)
    mamba_copy_in(ctx, st.buf_sig, f32_ptr(a[6]), qrow_n)
    mamba_copy_in(ctx, st.buf_adt, f32_ptr(a[7]), qrow_n)
    mamba_copy_in(ctx, st.pend_k, f32_ptr(a[8]), k_n)
    mamba_copy_in(ctx, st.pend_v, f32_ptr(a[9]), v_n)
    ctx.synchronize()
    st.buf_len = q0
    st.pending = pend == 1


def _m3_session_open_run(mut s: Mamba3DecodeSession, a: List[Int], b: Int,
                         dm: Int, q0: Int, pend: Int) raises:
    var dims = Mamba3Dims.of(dm)
    s.ctx = neural_ctx[_NEURAL_CTX]()
    var weight_addrs = List[Int]()
    weight_addrs.append(0)
    for i in range(9): weight_addrs.append(a[i])
    s.w = _m3_load_weights(s.ctx.value(), weight_addrs, dims)
    s.state = Mamba3DeviceState(s.ctx.value(), b, dims)
    s.b = b
    s.dm = dm
    var state_addrs = List[Int]()
    for i in range(10): state_addrs.append(a[9 + i])
    _m3_session_copy_in(s, state_addrs, q0, pend)
    s.dx = mamba_zeros(s.ctx.value(), b * dm)


# NI47 same-graph boundary scheduling: retained capacity for one physical
# open chunk plus a new token; extent comes from the model state contract,
# never a board row. Default OFF. Cold allocation, repeated decode and maximum
# zero-fill traffic must all be included in later A/B. Existing host state and
# arithmetic remain the S counterpart. All validation/quality/timing NOT RUN.
comptime IDN_M3_DECODE_WINDOW_REUSE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_IDN_M3_DECODE_WINDOW_REUSE"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    and MAMBA_GUARD == 0
)


def _m3_session_step_run(mut s: Mamba3DecodeSession, a: List[Int]) raises -> Int:
    ref ctx = s.ctx.value()
    ref st = s.state.value()
    var dims = Mamba3Dims.of(s.dm)
    var stages: Mamba3DeviceStages
    comptime if IDN_M3_DECODE_WINDOW_REUSE:
        if s.decode_stages:
            stages = s.decode_stages.take()
        else:
            stages = Mamba3DeviceStages(ctx, s.b, 1, M3_CHUNK_SIZE, dims)
        stages.reuse_decode_window(st.buf_len)
    else:
        stages = Mamba3DeviceStages(ctx, s.b, 1, st.buf_len, dims)
    mamba_copy_in(ctx, s.dx.value(), f32_ptr(a[0]), s.b * s.dm)
    var trace = IdentityTrace.disabled()
    mamba3_block_forward(ctx, stages, st, s.w.value(), s.dx.value(), s.b, 1,
                         trace, String("py.session"))
    var nh = dims.nheads
    _write_f32(a[1], mamba_download(ctx, stages.residual_out, s.b * s.dm))
    _write_f32(a[2], mamba_download(ctx, stages.h_last, s.b * nh * M3_HEADDIM * M3_D_STATE))
    _write_f32(a[3], mamba_download(ctx, stages.k_last, s.b * nh * M3_D_STATE))
    _write_f32(a[4], mamba_download(ctx, stages.v_last, s.b * nh * M3_HEADDIM))
    _write_f32(a[5], mamba_download(ctx, stages.theta_last, s.b * nh * M3_NUM_ROPE_ANGLES))
    comptime if IDN_M3_DECODE_WINDOW_REUSE:
        # Every report download above completes before capacity is retained.
        s.decode_stages = stages^
    return st.buf_len


def _m3_session_export_run(mut s: Mamba3DecodeSession, a: List[Int]) raises -> Int:
    ref ctx = s.ctx.value()
    ref st = s.state.value()
    var dims = Mamba3Dims.of(s.dm)
    var nh = dims.nheads
    var theta_n = s.b * nh * M3_NUM_ROPE_ANGLES
    var h_n = s.b * nh * M3_HEADDIM * M3_D_STATE
    var qrow_n = s.b * M3_CHUNK_SIZE * nh
    _write_f32(a[0], mamba_download(ctx, st.theta, theta_n))
    _write_f32(a[1], mamba_download(ctx, st.h, h_n))
    _write_f32(a[2], mamba_download(ctx, st.buf_qrot, qrow_n * M3_D_STATE))
    _write_f32(a[3], mamba_download(ctx, st.buf_krot, qrow_n * M3_D_STATE))
    _write_f32(a[4], mamba_download(ctx, st.buf_v, qrow_n * M3_HEADDIM))
    _write_f32(a[5], mamba_download(ctx, st.buf_dt, qrow_n))
    _write_f32(a[6], mamba_download(ctx, st.buf_sig, qrow_n))
    _write_f32(a[7], mamba_download(ctx, st.buf_adt, qrow_n))
    _write_f32(a[8], mamba_download(ctx, st.pend_k, s.b * nh * M3_D_STATE))
    _write_f32(a[9], mamba_download(ctx, st.pend_v, s.b * nh * M3_HEADDIM))
    return st.buf_len


# ===========================================================================
# lane/neural-net-experiment (2026-09-30): the Mamba-3 PREFILL session.
#
# `mamba3_forward_fresh` uploaded the nine weights (nine allocations and
# copies with a wait), built the zero state and the stages, uploaded x, ran
# the block, downloaded the five reports and freed everything, on EVERY
# call; Samba runs two such blocks per forward and the board's
# `mamba3-forward` cell is exactly this call. The prefill session keeps
# the WEIGHTS on the device across calls, reused when the caller's bytes
# are unchanged (an exact integer compare, as the transformer session
# does), recopied into the same buffers when they moved (the block's own
# non-finite refusal re-runs: `weights_checked` is cleared), and keeps the
# x buffer. The zero state and the stages are still built per call: the
# decode session builds its stages per step too, nothing resets them, and
# a fresh construction is the certified zero state. Same kernels, same
# launches, same order.
# ===========================================================================


# NN40 explicit owned snapshots, not pointer identity or caller promises.
# A snapshot is copied into session-owned device storage by install/update;
# owned forward/backward never read the original external weight addresses.
comptime NN40_OWNED_WEIGHTS = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_NN40_OWNED_WEIGHTS"]() and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()

struct Mamba3PrefillSession(Movable, Writable):
    var ctx: Optional[DeviceContext]
    var w: Optional[Mamba3DeviceWeights]
    var w_host: List[Float32]
    var dm: Int
    var dx: Optional[DeviceBuffer[DType.float32]]
    var busy: Bool
    var usable: Bool
    var weight_uploads: Int
    var weight_recopies: Int
    var weight_reuses: Int
    var owned_weights: Bool
    var weight_generation: Int
    # lane/neural-net-experiment (backward session): the last forward's
    # recorded stages and zero state, kept for the backward that follows it
    # in a training step. `stages_valid` is cleared whenever the retained
    # weights change (recopy or upload), since the stages were computed on
    # the old bytes. `x_stage` is the host staging list the backward
    # downloads `dx` into to compare against the caller's x.
    var stages: Optional[Mamba3DeviceStages]
    var state: Optional[Mamba3DeviceState]
    var stages_b: Int
    var stages_l: Int
    var stages_valid: Bool
    var x_stage: List[Float32]
    var backward_reuses: Int
    var backward_recomputes: Int

    def __init__(out self):
        self.ctx = None
        self.w = None
        self.w_host = List[Float32]()
        self.owned_weights = False
        self.dm = 0
        self.dx = None
        self.busy = False
        self.usable = True
        self.weight_uploads = 0
        self.weight_recopies = 0
        self.weight_reuses = 0
        self.owned_weights = False
        self.weight_generation = 0
        self.stages = None
        self.state = None
        self.stages_b = 0
        self.stages_l = 0
        self.stages_valid = False
        self.x_stage = List[Float32]()
        self.backward_reuses = 0
        self.backward_recomputes = 0

    def write_to(self, mut writer: Some[Writer]): writer.write("Mamba3PrefillSession")
    def write_repr_to(self, mut writer: Some[Writer]): writer.write("Mamba3PrefillSession")

    def release(mut self):
        self.drop_stages()
        self.x_stage = List[Float32]()
        self.dx = None
        self.w = None
        self.w_host = List[Float32]()
        self.owned_weights = False
        self.dm = 0
        if self.ctx:
            try: self.ctx.value().synchronize()
            except: pass
        self.ctx = None

    def drop_stages(mut self):
        self.stages_valid = False
        self.stages = None
        self.state = None
        self.stages_b = 0
        self.stages_l = 0

    def __deinit__(deinit self): self.release()


def _m3_retain_stages() -> Bool:
    """MOJOLEARN_MAMBA3_RETAIN_STAGES=0 drops the forward's stages at the
    end of the forward (the pre-session behaviour: every backward recomputes
    the forward). Default on: the stages stay on the device for the
    backward of the same training step."""
    return String(getenv("MOJOLEARN_MAMBA3_RETAIN_STAGES")) != "0"


def _m3_weight_lens(dims: Mamba3Dims) -> List[Int]:
    """Element counts of the nine weights in slot order 1..9."""
    var dm = dims.d_model
    var di = dims.d_inner
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    var lens: List[Int] = [dm, dip * dm, nh, M3_D_STATE, M3_D_STATE, nh * M3_D_STATE,
                           nh * M3_D_STATE, nh, dm * di]
    return lens^


def _m3_weights_same_bits(a: List[Int], host: List[Float32], lens: List[Int]) raises -> Bool:
    """Whether the caller's nine weight arrays hold exactly the bytes in
    `host` (their concatenation in slot order): an integer compare of the
    bit patterns, eight lanes at a time."""
    comptime W = 8
    var off = 0
    for s in range(9):
        var n = lens[s]
        var p = _f32_ptr(a[s + 1])
        var hp = host.unsafe_ptr()
        var i = 0
        var body = n - n % W
        while i < body:
            var x = bitcast[DType.uint32, W](p.unsafe_load[width=W](i))
            var y = bitcast[DType.uint32, W](hp.unsafe_load[width=W](off + i))
            if (x ^ y).reduce_or() != UInt32(0):
                return False
            i += W
        while i < n:
            if bitcast[DType.uint32](p.unsafe_load(i)) != bitcast[DType.uint32](hp.unsafe_load(off + i)):
                return False
            i += 1
        off += n
    return True


def _m3_weights_host_copy(a: List[Int], lens: List[Int]) raises -> List[Float32]:
    var total = 0
    for s in range(9):
        total += lens[s]
    var out = List[Float32](length=total, fill=Float32(0))
    var off = 0
    for s in range(9):
        var n = lens[s]
        if n > 0:
            memcpy(dest=out.unsafe_ptr() + off, src=_f32_ptr(a[s + 1]), count=n)
        off += n
    return out^


def _m3_weights_recopy(ctx: DeviceContext, mut w: Mamba3DeviceWeights, a: List[Int]) raises:
    """Changed caller bytes into the RETAINED device buffers, slot order,
    one wait; `weights_checked` cleared so the block's own named refusal of
    non-finite weights runs again on the next forward."""
    ctx.enqueue_copy(dst_buf=w.norm_w, src_ptr=_f32_ptr(a[1]))
    ctx.enqueue_copy(dst_buf=w.w_in, src_ptr=_f32_ptr(a[2]))
    ctx.enqueue_copy(dst_buf=w.dt_bias, src_ptr=_f32_ptr(a[3]))
    ctx.enqueue_copy(dst_buf=w.bnorm_w, src_ptr=_f32_ptr(a[4]))
    ctx.enqueue_copy(dst_buf=w.cnorm_w, src_ptr=_f32_ptr(a[5]))
    ctx.enqueue_copy(dst_buf=w.b_bias, src_ptr=_f32_ptr(a[6]))
    ctx.enqueue_copy(dst_buf=w.c_bias, src_ptr=_f32_ptr(a[7]))
    ctx.enqueue_copy(dst_buf=w.d_skip, src_ptr=_f32_ptr(a[8]))
    ctx.enqueue_copy(dst_buf=w.w_out, src_ptr=_f32_ptr(a[9]))
    ctx.synchronize()
    w.weights_checked = False


def _m3_prefill_weights(mut s: Mamba3PrefillSession, a: List[Int], dims: Mamba3Dims, use_owned: Bool = False) raises:
    """Leave `s.w` holding this call's weights: reused, recopied, or freshly
    uploaded (see the section header)."""
    if use_owned:
        comptime if NN40_OWNED_WEIGHTS:
            if not s.owned_weights or not s.w or s.dm != dims.d_model:
                raise Error("mamba3 owned weights: install a matching snapshot first")
            s.weight_reuses += 1
            return
        else:
            raise Error("mamba3 owned weights: NN40 arm is disabled")
    # Normal APIs continue inspecting the weights passed on THIS call.
    # They relinquish the explicit snapshot route instead of silently
    # ignoring changed external buffers supplied through the ordinary API.
    s.owned_weights = False
    ref ctx = s.ctx.value()
    var lens = _m3_weight_lens(dims)
    # MOJOLEARN_MAMBA3_RETAIN_WEIGHTS=0: the per-call upload (the A/B
    # against the byte compare).
    if String(getenv("MOJOLEARN_MAMBA3_RETAIN_WEIGHTS")) == "0":
        s.stages_valid = False
        s.w = None
        s.w_host = List[Float32]()
        s.dm = 0
        s.w = _m3_load_weights(ctx, a, dims)
        s.weight_uploads += 1
        return
    if s.w and s.dm == dims.d_model:
        if _m3_weights_same_bits(a, s.w_host, lens):
            s.weight_reuses += 1
            return
        s.stages_valid = False
        var fresh = _m3_weights_host_copy(a, lens)
        try:
            ref retained = s.w.value()
            _m3_weights_recopy(ctx, retained, a)
        except error:
            s.w = None
            s.w_host = List[Float32]()
            s.dm = 0
            raise error
        s.w_host = fresh^
        s.weight_recopies += 1
        return
    s.stages_valid = False
    s.w = None
    s.w_host = List[Float32]()
    s.dm = 0
    var host = _m3_weights_host_copy(a, lens)
    s.w = _m3_load_weights(ctx, a, dims)
    s.w_host = host^
    s.dm = dims.d_model
    s.weight_uploads += 1


def _m3_prefill_run(mut s: Mamba3PrefillSession, a: List[Int], b: Int, l: Int, dm: Int, use_owned: Bool = False, force_retain: Bool = False) raises -> Int:
    """`_mamba3_run[True]` on the session: `a` is the fresh entry's 25-slot
    list (state slots zero, never read)."""
    var dims = Mamba3Dims.of(dm)
    var nh = dims.nheads
    var theta_n = b * nh * M3_NUM_ROPE_ANGLES
    var h_n = b * nh * M3_HEADDIM * M3_D_STATE
    var k_n = b * nh * M3_D_STATE
    var v_n = b * nh * M3_HEADDIM
    if not s.ctx:
        s.ctx = neural_ctx[_NEURAL_CTX]()
    var phase_tick = 0
    comptime if is_defined["MOJOLEARN_MAMBA3_PHASE_TIMERS"]():
        phase_tick = Int(perf_counter_ns())
    _m3_prefill_weights(s, a, dims, use_owned)
    # lane/fam2-lm (IDN_M3_SESSION_STAGE_REUSE): the retained stages and
    # state serve this call when they have its shape; they are refilled with
    # zeros below. (`stages_valid` is not asked: stale VALUES do not matter,
    # every buffer is refilled.)
    var reuse = False
    comptime if IDN_M3_SESSION_STAGE_REUSE and MAMBA_GUARD == 0:
        if s.stages and s.state:
            reuse = (
                s.stages_b == b
                and s.stages_l == l
                and s.stages.value().dims.d_model == dm
            )
    # The previous forward's stages go before this call's are built.
    if not reuse:
        s.drop_stages()
    ref ctx = s.ctx.value()
    ref dw = s.w.value()
    m3_phase_tick(ctx, phase_tick, String("surface.weight_upload"))
    # The certified zero state and this call's stages, built per call (or
    # the retained ones, zeroed).
    var dstate: Mamba3DeviceState
    var dstages: Mamba3DeviceStages
    if reuse:
        dstate = s.state.take()
        dstages = s.stages.take()
        s.stages_valid = False
        dstate.rezero()
        dstages.rezero()
    else:
        dstate = Mamba3DeviceState(ctx, b, dims)
        dstages = Mamba3DeviceStages(ctx, b, l, 0, dims)
    m3_phase_tick(ctx, phase_tick, String("surface.stage_allocations"))
    var n_x = b * l * dm
    var have_dx = False
    if s.dx:
        have_dx = len(s.dx.value()) == n_x
    if not have_dx:
        s.dx = None
        s.dx = mamba_device_alloc(ctx, max(n_x, 1))
    ref dx = s.dx.value()
    ctx.enqueue_copy(dst_buf=dx, src_ptr=_f32_ptr(a[0]))
    # lane fam-lm (IDN_MAMBA_ALLOC_NOWAIT, IDENTICAL): the caller's x array
    # outlives this call and the copy is ahead of every reader on the
    # in-order context, so the upload's own wait ordered nothing.
    comptime if not IDN_MAMBA_ALLOC_NOWAIT:
        ctx.synchronize()
    m3_phase_tick(ctx, phase_tick, String("surface.x_upload"))
    var trace = IdentityTrace.disabled()
    mamba3_block_forward(
        ctx, dstages, dstate, dw, dx, b, l, trace, String("py")
    )
    m3_phase_tick(ctx, phase_tick, String("surface.block"))
    _m3_download_addr[False](ctx, dstages.residual_out, n_x, a[20])
    # lane/fam2-lm (IDN_MAMBA3_REPORTS_ON_REQUEST): a report comes down only
    # where the caller gave it an address; the session forward entry admits
    # a null report address only under that switch, so with it off every
    # address here is non-null and all four copies run as before.
    if a[21] != 0:
        _m3_download_addr[False](ctx, dstages.h_last, h_n, a[21])
    if a[22] != 0:
        _m3_download_addr[False](ctx, dstages.k_last, k_n, a[22])
    if a[23] != 0:
        _m3_download_addr[False](ctx, dstages.v_last, v_n, a[23])
    if a[24] != 0:
        _m3_download_addr[False](ctx, dstages.theta_last, theta_n, a[24])
    ctx.synchronize()
    m3_phase_tick(ctx, phase_tick, String("surface.downloads"))
    var out_len = dstate.buf_len
    ctx.synchronize()
    if force_retain or _m3_retain_stages():
        s.stages = dstages^
        s.state = dstate^
        s.stages_b = b
        s.stages_l = l
        s.stages_valid = True
    else:
        _ = dstate^
        _ = dstages^
    return out_len


def _m3_x_same_bits(mut s: Mamba3PrefillSession, addr: Int, n: Int) raises -> Bool:
    """Whether the caller's x (`n` floats at `addr`) holds exactly the
    bytes of the retained device `s.dx`: `dx` comes down into `s.x_stage`
    (one wait) and the two are compared as integers, eight lanes at a
    time. Paid only by a backward, never by a forward-only workload."""
    if len(s.x_stage) != n:
        s.x_stage = List[Float32](length=n, fill=Float32(0))
    if n == 0:
        return True
    ref ctx = s.ctx.value()
    ref dx = s.dx.value()
    if n == len(dx):
        ctx.enqueue_copy(dst_ptr=s.x_stage.unsafe_ptr(), src_buf=dx)
    else:
        var view = dx.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=s.x_stage.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    comptime W = 8
    var p = _f32_ptr(addr)
    var hp = s.x_stage.unsafe_ptr()
    var i = 0
    var body = n - n % W
    while i < body:
        var x = bitcast[DType.uint32, W](p.unsafe_load[width=W](i))
        var y = bitcast[DType.uint32, W](hp.unsafe_load[width=W](i))
        if (x ^ y).reduce_or() != UInt32(0):
            return False
        i += W
    while i < n:
        if bitcast[DType.uint32](p.unsafe_load(i)) != bitcast[DType.uint32](hp.unsafe_load(i)):
            return False
        i += 1
    return True


def _m3_prefill_backward_forward(mut s: Mamba3PrefillSession, a: List[Int], b: Int, l: Int,
                                 dims: Mamba3Dims) raises:
    """The forward `mamba3_tape_backward` runs before its VJP, on the
    session: x uploaded into the retained `dx`, a fresh zero state and
    fresh stages (the same constructions as the entry's), the block
    forward on the retained weights; the stages then held for the VJP."""
    s.drop_stages()
    var n_x = b * l * dims.d_model
    var have_dx = False
    if s.dx:
        have_dx = len(s.dx.value()) == n_x
    if not have_dx:
        s.dx = None
        s.dx = mamba_device_alloc(s.ctx.value(), max(n_x, 1))
    ref ctx = s.ctx.value()
    ref dw = s.w.value()
    ref dx = s.dx.value()
    ctx.enqueue_copy(dst_buf=dx, src_ptr=_f32_ptr(a[0]))
    ctx.synchronize()
    var dstate = Mamba3DeviceState(ctx, b, dims)
    var dstages = Mamba3DeviceStages(ctx, b, l, 0, dims)
    var trace = IdentityTrace.disabled()
    mamba3_block_forward(
        ctx, dstages, dstate, dw, dx, b, l, trace, String("m3.backward.tail")
    )
    ctx.synchronize()
    s.stages = dstages^
    s.state = dstate^
    s.stages_b = b
    s.stages_l = l
    s.stages_valid = True


comptime _M3_BWD_NONFINITE_DY = "mamba3 backward: non-finite grad_output at flat index "


def _m3_prefill_backward_run(mut s: Mamba3PrefillSession, a: List[Int], b: Int, l: Int, dm: Int, use_owned: Bool = False) raises:
    """`mamba3_backward` on the session: `a` is the entry's 21-slot list
    (x, nine weights, grad_output, grad_x, nine weight gradients).

    The weights are the session's (reused when their bytes are unchanged,
    as the forward does). The forward is recomputed only when it has to
    be: when the session holds no stages, when they were recorded on
    other weights, at another (B, L), or on an x whose bytes differ from
    this call's. Otherwise the VJP runs on the stages the last forward
    recorded -- the same buffers the entry's own recompute would fill
    with the same bits, since the same kernels run on the same inputs.
    The ten gradients come down straight into the caller's arrays."""
    var dims = Mamba3Dims.of(dm)
    var n_x = b * l * dm
    if not s.ctx:
        s.ctx = neural_ctx[_NEURAL_CTX]()
    # cpu2-l11-neural (2026-10-04): the entry's refusal of a non-finite
    # grad_output is a DEVICE scan of the uploaded buffer (one launch, one
    # partials copy; the same first flat index the host walk reported), run
    # before the weights or the stages are touched, so the refusal leaves
    # the session as it was (the entry keeps it usable).
    var d_output = _m3_upload_addr(s.ctx.value(), a[10], n_x)
    var bad_dy = device_first_nonfinite(s.ctx.value(), d_output, n_x)
    if bad_dy >= 0:
        raise Error(String(_M3_BWD_NONFINITE_DY) + String(bad_dy))
    _m3_prefill_weights(s, a, dims, use_owned)
    var reuse = False
    if s.stages_valid and s.stages and s.stages_b == b and s.stages_l == l and s.dx:
        if len(s.dx.value()) == n_x:
            reuse = _m3_x_same_bits(s, a[0], n_x)
    if reuse:
        s.backward_reuses += 1
    else:
        s.backward_recomputes += 1
        _m3_prefill_backward_forward(s, a, b, l, dims)
    ref ctx = s.ctx.value()
    ref dw = s.w.value()
    ref stages = s.stages.value()
    ref dx = s.dx.value()
    var ton = String(getenv("MOJOLEARN_MAMBA_TIMING")) != ""
    var g = mamba3_tape_backward_on(ctx, dw, stages, dx, d_output, b, l, dims, ton)
    _m3_download_addr[False](ctx, g.x, n_x, a[11])
    _m3_download_addr[False](ctx, g.block_norm_weight, dm, a[12])
    _m3_download_addr[False](ctx, g.in_proj_weight, dims.d_in_proj() * dm, a[13])
    _m3_download_addr[False](ctx, g.dt_bias, dims.nheads, a[14])
    _m3_download_addr[False](ctx, g.B_norm_weight, M3_D_STATE, a[15])
    _m3_download_addr[False](ctx, g.C_norm_weight, M3_D_STATE, a[16])
    _m3_download_addr[False](ctx, g.B_bias, dims.nheads * M3_D_STATE, a[17])
    _m3_download_addr[False](ctx, g.C_bias, dims.nheads * M3_D_STATE, a[18])
    _m3_download_addr[False](ctx, g.D, dims.nheads, a[19])
    _m3_download_addr[False](ctx, g.out_proj_weight, dm * dims.d_inner, a[20])
    ctx.synchronize()
    _ = g^
    _ = d_output^


def _m3_install_weights_run(mut s: Mamba3PrefillSession, a: List[Int], dims: Mamba3Dims, native_d_model: Int) raises:
    """Upload nine caller weights into a fresh owned generation on the
    session (the GIL-free half of `mamba3_prefill_session_install_weights`)."""
    if not s.ctx:
        s.ctx = neural_ctx[_NEURAL_CTX]()
    var fresh = _m3_load_weights(s.ctx.value(), a, dims)
    s.ctx.value().synchronize()
    # Copy-on-install is the ownership boundary. Drop stages before
    # exposing the new generation, never reusing old-weight activations.
    s.drop_stages()
    s.w = fresh^
    s.w_host = List[Float32]()
    s.dm = native_d_model
    s.owned_weights = True
    s.weight_generation += 1
    s.weight_uploads += 1


def _m3_export_weights_run(mut s: Mamba3PrefillSession, a: List[Int], dm: Int, dims: Mamba3Dims) raises:
    """Download the installed generation into nine caller buffers (the
    GIL-free half of `mamba3_prefill_session_export_weights`)."""
    ref ctx = s.ctx.value()
    ref w = s.w.value()
    _m3_download_addr[False](ctx, w.norm_w, dm, a[0])
    _m3_download_addr[False](ctx, w.w_in, dims.d_in_proj() * dm, a[1])
    _m3_download_addr[False](ctx, w.dt_bias, dims.nheads, a[2])
    _m3_download_addr[False](ctx, w.bnorm_w, M3_D_STATE, a[3])
    _m3_download_addr[False](ctx, w.cnorm_w, M3_D_STATE, a[4])
    _m3_download_addr[False](ctx, w.b_bias, dims.nheads * M3_D_STATE, a[5])
    _m3_download_addr[False](ctx, w.c_bias, dims.nheads * M3_D_STATE, a[6])
    _m3_download_addr[False](ctx, w.d_skip, dims.nheads, a[7])
    _m3_download_addr[False](ctx, w.w_out, dm * dims.d_inner, a[8])
    ctx.synchronize()

# NI48: default-OFF owned forward tape. A tape has exactly one fresh session;
# input, weights and stages are native owned copies from the actual forward.
# There is no mutable external generation to compare: this tape's generation
# is its immutable snapshot and it can be consumed once. Future forwards use
# other owners. Native close/failure invalidates this owner before release.
# This removes both forward recomputation and device-to-host input comparison.
# Arithmetic is unchanged; CPU retains its equivalent recompute path. No
# compilation, identity, quality or end-to-end measurements have been run.
comptime IDN_SAMBA_M3_FORWARD_TAPE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_IDN_ACT_RETAIN_OFF"]()
    and get_defined_int["MOJOLEARN_IDN_ACT_RETAIN", 2]() == 3  # NI48 arm (default 2 since 2026-10-08)
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)


struct Mamba3ForwardTape(Movable, Writable):
    var owner: Mamba3PrefillSession
    var consumed: Bool

    def __init__(out self, var owner: Mamba3PrefillSession):
        self.owner = owner^
        self.consumed = False

    def write_to(self, mut writer: Some[Writer]): writer.write("Mamba3ForwardTape")
    def write_repr_to(self, mut writer: Some[Writer]): writer.write("Mamba3ForwardTape")


def _m3_backward_tape_run(mut s: Mamba3PrefillSession, a: List[Int]) raises:
    var b = s.stages_b; var l = s.stages_l
    var dims = s.stages.value().dims.copy()
    var dm = dims.d_model
    var n_x = b * l * dm
    ref ctx = s.ctx.value()
    var d_output = _m3_upload_addr(ctx, a[0], n_x)
    var bad_dy = device_first_nonfinite(ctx, d_output, n_x)
    if bad_dy >= 0:
        raise Error(String(_M3_BWD_NONFINITE_DY) + String(bad_dy))
    ref dw = s.w.value()
    ref stages = s.stages.value()
    ref dx = s.dx.value()
    var ton = String(getenv("MOJOLEARN_MAMBA_TIMING")) != ""
    var g = mamba3_tape_backward_on(ctx, dw, stages, dx, d_output, b, l, dims, ton)
    _m3_download_addr[False](ctx, g.x, n_x, a[1])
    _m3_download_addr[False](ctx, g.block_norm_weight, dm, a[2])
    _m3_download_addr[False](ctx, g.in_proj_weight, dims.d_in_proj() * dm, a[3])
    _m3_download_addr[False](ctx, g.dt_bias, dims.nheads, a[4])
    _m3_download_addr[False](ctx, g.B_norm_weight, M3_D_STATE, a[5])
    _m3_download_addr[False](ctx, g.C_norm_weight, M3_D_STATE, a[6])
    _m3_download_addr[False](ctx, g.B_bias, dims.nheads * M3_D_STATE, a[7])
    _m3_download_addr[False](ctx, g.C_bias, dims.nheads * M3_D_STATE, a[8])
    _m3_download_addr[False](ctx, g.D, dims.nheads, a[9])
    _m3_download_addr[False](ctx, g.out_proj_weight, dm * dims.d_inner, a[10])
    ctx.synchronize()
    s.backward_reuses += 1
    _ = g^
    _ = d_output^


def _mamba2_backward_run(a: List[Int], b: Int, l: Int, dm: Int, dt_lo: Float32, dt_hi: Float32) raises:
    var dims = Mamba2Dims.of(dm)
    var di = dims.d_inner
    var cd = dims.conv_dim()
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    var w = Mamba2Weights(dims)
    w.norm_w = _read_f32(a[1], dm)
    w.w_in = _read_f32(a[2], dip * dm)
    w.conv_w = _read_f32(a[3], cd * M2_D_CONV)
    w.conv_b = _read_f32(a[4], cd)
    w.dt_bias = _read_f32(a[5], nh)
    w.a_log = _read_f32(a[6], nh)
    w.d_skip = _read_f32(a[7], nh)
    w.gnorm_w = _read_f32(a[8], di)
    w.w_out = _read_f32(a[9], dm * di)
    # TOMBSTONE: MOJOLEARN_AUTOGRAD_MAMBA_OFF (hand Mamba-2 prefill backward arm) deleted 2026-10-19 by 47582a7fb; code recoverable at 47582a7fb^.
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_MAMBA_OFF.patch
    var gradients: Mamba2PrefillGradients = mamba2_tape_backward(
        w, _read_f32(a[0], b*l*dm), _read_f32(a[10], b*l*dm), b, l, dt_lo, dt_hi,
        neural_ctx[_NEURAL_CTX](),
    )
    _write_f32(a[11], gradients.x)
    _write_f32(a[12], gradients.block_norm_weight)
    _write_f32(a[13], gradients.in_proj_weight)
    _write_f32(a[14], gradients.conv1d_weight)
    _write_f32(a[15], gradients.conv1d_bias)
    _write_f32(a[16], gradients.dt_bias)
    _write_f32(a[17], gradients.A_log)
    _write_f32(a[18], gradients.D)
    _write_f32(a[19], gradients.norm_weight)
    _write_f32(a[20], gradients.out_proj_weight)


def _mamba3_backward_run(a: List[Int], b: Int, l: Int, dm: Int) raises:
    var dims = Mamba3Dims.of(dm)
    var di = dims.d_inner
    var dip = dims.d_in_proj()
    var nh = dims.nheads
    var w = Mamba3Weights(dims)
    w.norm_w = _read_f32(a[1], dm)
    w.w_in = _read_f32(a[2], dip * dm)
    w.dt_bias = _read_f32(a[3], nh)
    w.bnorm_w = _read_f32(a[4], M3_D_STATE)
    w.cnorm_w = _read_f32(a[5], M3_D_STATE)
    w.b_bias = _read_f32(a[6], nh * M3_D_STATE)
    w.c_bias = _read_f32(a[7], nh * M3_D_STATE)
    w.d_skip = _read_f32(a[8], nh)
    w.w_out = _read_f32(a[9], dm * di)
    var gradients = mamba3_tape_backward(
        w, _read_f32(a[0], b*l*dm), _read_f32(a[10], b*l*dm), b, l,
        neural_ctx[_NEURAL_CTX](),
    )
    _write_f32(a[11], gradients.x)
    _write_f32(a[12], gradients.block_norm_weight)
    _write_f32(a[13], gradients.in_proj_weight)
    _write_f32(a[14], gradients.dt_bias)
    _write_f32(a[15], gradients.B_norm_weight)
    _write_f32(a[16], gradients.C_norm_weight)
    _write_f32(a[17], gradients.B_bias)
    _write_f32(a[18], gradients.C_bias)
    _write_f32(a[19], gradients.D)
    _write_f32(a[20], gradients.out_proj_weight)
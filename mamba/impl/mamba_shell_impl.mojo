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
from core.neural_context import neural_ctx, neural_device_plan
from core.device_plan import DevicePlan
from core.device_scan import device_first_nonfinite
# One process-lifetime DeviceContext per binding and tier (core/neural_context.mojo).
comptime _NEURAL_CTX = "MojolearnNeuralMambaContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnNeuralMambaContextFastV3"
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
    MAMBA_POISON,
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
from mamba.impl.mamba_block_program import MambaBlockProgram, mamba_block_call
from autograd.block_step_ops import OPS_M1_STEP, OPS_M2_STEP, OPS_M3_STEP
#: lane afn-mamba (2026-10-03): MOJOLEARN_AFN_MAMBA_ARENA / IDN_MAMBA_ARENA make
#: the block structs views of one arena; the per-call `_mamba{1,2,3}_run_arena`
#: bodies that used it here are gone with the per-call bodies (lane x5: the
#: block step op builds its structs over the program's tensors the same way).
from mamba.impl.modules.afn_defines import AFN_MAMBA_ARENA, IDN_MAMBA_ALLOC_NOWAIT
from mamba.impl.modules.afn_defines import IDN_MAMBA3_REPORTS_ON_REQUEST
from mamba.impl.modules.afn_defines import IDN_M3_SESSION_STAGE_REUSE
from mamba.impl.modules.afn_arena import MambaArena
from mamba.impl.modeling.modeling_mamba import mamba1_arena_floats
from mamba.impl.modules.mamba2 import mamba2_arena_floats, mamba2_arena_work_floats
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




# TOMBSTONE: _mamba1_run, _mamba1_run_arena (the per-call Mamba-1 body: ten weight and two state uploads per call) (moved onto the block step op, lane x5-mamba-on-executor) deleted 2026-11-09 by 844ea17a80; code recoverable at 25e5650c14.
# Restore: git apply experiments/removed/X5_MAMBA_PER_CALL_BODIES.patch
# The per-call Mamba-1 forward/step is `mamba_block_call(OPS_M1_STEP, ...)`
# (mamba/impl/mamba_block_program.mojo): the step op's body on a cached
# program whose weights stay on the device while their bytes are unchanged.



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
# neural, 2026-09-17; on the executor since lane x5-mamba-on-executor,
# 2026-11-09): the weights and the state live on the device across decode
# steps as one executor program, and every step is the block step op
# (`OPS_M1_STEP`, autograd/block_step_ops.mojo), whose L = 1 body is
# `mamba_step`, THE certified decode entry, on resident L = 1 stages (stage
# reuse across calls is the Mamba-1 stage struct's own contract). Ownership:
# the session COPIES the weights and the state at open; the caller's state
# arrays are STALE until export_state or close; weights refresh is close and
# open.
# ===========================================================================


struct Mamba1DecodeSession(Movable, Writable):
    """The resident Mamba-1 decode session: ONE executor program
    (`MambaBlockProgram`, the weights a parameter and the state a declared
    state tensor, both uploaded once at open) whose every token is the block
    step op. The caller's state arrays are stale until export_state or
    close; weights refresh is close and open."""
    var prog: Optional[MambaBlockProgram]
    var b: Int
    var dm: Int
    var busy: Bool
    var usable: Bool

    def __init__(out self):
        self.prog = None
        self.b = 0
        self.dm = 0
        self.busy = False
        self.usable = True

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Mamba1DecodeSession")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("Mamba1DecodeSession")

    def is_open(self) -> Bool:
        return Bool(self.prog)

    def release(mut self):
        if self.prog:
            try:
                self.prog.value().close()
            except:
                pass
        self.prog = None

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
    """`a` = [0, ten weights in `mamba1_forward`'s order, conv_window, h]:
    the program over the weights, the caller's state written in."""
    var weights = List[Int](capacity=10)
    for i in range(1, 11):  # small-loop(the ten Mamba-1 weight addresses)
        weights.append(a[i])
    var prog = MambaBlockProgram(OPS_M1_STEP, b, dm, List[Float32](), weights)
    prog.load_state([a[11], a[12]], List[Int]())
    s.prog = prog^
    s.b = b
    s.dm = dm


def _m1_session_step_run(mut s: Mamba1DecodeSession, px: Int, py: Int) raises:
    """One decode token: the step op, one completion wait."""
    _ = s.prog.value().step(px, py, List[Int]())


def _m1_session_decode_run(mut s: Mamba1DecodeSession, px: Int, py: Int, steps: Int) raises:
    """N decode tokens in one call (decode plan D2), the step op per token."""
    _ = s.prog.value().decode(px, py, steps, List[Int]())


def _m1_session_export_run(mut s: Mamba1DecodeSession, pw: Int, ph: Int) raises:
    _ = s.prog.value().export_state([pw, ph])


def _m1_session_load_run(mut s: Mamba1DecodeSession, pw: Int, ph: Int) raises:
    s.prog.value().load_state([pw, ph], List[Int]())


# ===========================================================================
# Mamba-2: profile mojolearn.identical.mamba2.fp32.v1
# ===========================================================================


# TOMBSTONE: _mamba2_run, _mamba2_run_arena, _mamba2_arena_execute, _mamba2_arena_execute_prepared, Mamba2PublicWorkspace, _mamba2_workspace_step (the per-call Mamba-2 bodies and the public-step workspace), the Mamba-2 decode session body and its IDN_DECODE_RESIDENT_IO_M2 site, _m2_session_download_direct (moved onto the block step op, lane x5-mamba-on-executor) deleted 2026-11-09 by 844ea17a80; code recoverable at 25e5650c14.
# Restore: git apply experiments/removed/X5_MAMBA_PER_CALL_BODIES.patch
# The per-call Mamba-2 forward/step is `mamba_block_call(OPS_M2_STEP, ...)`;
# the public step's workspace (storage kept across steps, every weight
# re-uploaded per call) is the cached program. The session below returns y
# and h_last straight into the caller's arrays behind one wait (the ON arm of
# IDN_DECODE_RESIDENT_IO for Mamba-2 is the only arm now; the define stays for
# the transformer session).


struct Mamba2DecodeSession(Movable, Writable):
    """The resident Mamba-2 decode session: one executor program; every
    token is the block step op at L = 1 (`mamba2_block_forward`, stages at
    the carried buf_len). The state fields are views of the program's state
    tensor and are never replaced (the AMD field-replacement fault)."""
    var prog: Optional[MambaBlockProgram]
    var b: Int
    var dm: Int
    var busy: Bool
    var usable: Bool

    def __init__(out self):
        self.prog = None
        self.b = 0
        self.dm = 0
        self.busy = False
        self.usable = True

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Mamba2DecodeSession")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("Mamba2DecodeSession")

    def is_open(self) -> Bool:
        return Bool(self.prog)

    def release(mut self):
        if self.prog:
            try:
                self.prog.value().close()
            except:
                pass
        self.prog = None

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
    """`a` = [nine weights in `mamba2_forward`'s order, conv_window, h,
    buffer_xbc, buffer_dtraw]."""
    var weights = List[Int](capacity=9)
    for i in range(9):  # small-loop(the nine Mamba-2 weight addresses)
        weights.append(a[i])
    var fs: List[Float32] = [dt_lo, dt_hi]
    var prog = MambaBlockProgram(OPS_M2_STEP, b, dm, fs, weights)
    prog.load_state([a[9], a[10], a[11], a[12]], [q0])
    s.prog = prog^
    s.b = b
    s.dm = dm


def _m2_session_step_run(mut s: Mamba2DecodeSession, px: Int, py: Int, ph: Int) raises -> Int:
    """One token: y and the h_last report into the caller's arrays, one
    wait; returns the carried buf_len."""
    return s.prog.value().step(px, py, [ph])[0]


def _m2_session_decode_run(mut s: Mamba2DecodeSession, px: Int, py: Int, ph: Int, steps: Int) raises -> Int:
    """N tokens in one call (decode plan D2); the last token's h_last."""
    return s.prog.value().decode(px, py, steps, [ph])[0]


def _m2_session_export_run(mut s: Mamba2DecodeSession, a: List[Int]) raises -> Int:
    return s.prog.value().export_state(a)[0]


def _m2_session_load_run(mut s: Mamba2DecodeSession, a: List[Int], q0: Int) raises:
    s.prog.value().load_state(a, [q0])


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


# TOMBSTONE: _mamba3_run, _mamba3_check_state_words, _mamba3_run_on, _mamba3_run_arena (the per-call Mamba-3 body: nine weight and ten state uploads, fifteen downloads per call) and the Mamba-3 decode session body (moved onto the block step op, lane x5-mamba-on-executor) deleted 2026-11-09 by 844ea17a80; code recoverable at 25e5650c14.
# Restore: git apply experiments/removed/X5_MAMBA_PER_CALL_BODIES.patch
# The per-call Mamba-3 forward/step is `mamba_block_call(OPS_M3_STEP, ...)`;
# the discard-state prefill (`mamba3_forward_fresh`'s fallback) is the same
# call with `fresh=True`.


struct Mamba3DecodeSession(Movable, Writable):
    """The resident Mamba-3 decode session: one executor program; every
    token is the block step op (stages at the carried buf_len, or the NI47
    window reuse arm). `pending` mirrors the carried flag for export."""
    var prog: Optional[MambaBlockProgram]
    var b: Int
    var dm: Int
    var pending: Bool
    var busy: Bool
    var usable: Bool

    def __init__(out self):
        self.prog = None
        self.b = 0
        self.dm = 0
        self.pending = False
        self.busy = False
        self.usable = True

    def write_to(self, mut writer: Some[Writer]): writer.write("Mamba3DecodeSession")
    def write_repr_to(self, mut writer: Some[Writer]): writer.write("Mamba3DecodeSession")
    def is_open(self) -> Bool: return Bool(self.prog)

    def release(mut self):
        if self.prog:
            try:
                self.prog.value().close()
            except:
                pass
        self.prog = None

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
    """The caller's ten state pieces and [buf_len, pending] into the
    program, in place (the step op's checks refuse a bad pair first)."""
    s.prog.value().load_state(a, [q0, pend])
    s.pending = pend == 1


def _m3_session_open_run(mut s: Mamba3DecodeSession, a: List[Int], b: Int,
                         dm: Int, q0: Int, pend: Int) raises:
    """`a` = [nine weights in `mamba3_forward`'s order, the ten state pieces]."""
    var weights = List[Int](capacity=9)
    for i in range(9):  # small-loop(the nine Mamba-3 weight addresses)
        weights.append(a[i])
    var pieces = List[Int](capacity=10)
    for i in range(10):  # small-loop(the ten Mamba-3 state addresses)
        pieces.append(a[9 + i])
    var prog = MambaBlockProgram(OPS_M3_STEP, b, dm, List[Float32](), weights)
    prog.load_state(pieces, [q0, pend])
    s.prog = prog^
    s.b = b
    s.dm = dm
    s.pending = pend == 1


def _m3_session_step_run(mut s: Mamba3DecodeSession, a: List[Int]) raises -> Int:
    """`a` = [x, y, h_last, k_last, v_last, theta_last]: one token, one
    wait; returns the carried buf_len."""
    var words = s.prog.value().step(a[0], a[1], [a[2], a[3], a[4], a[5]])
    s.pending = words[1] == 1
    return words[0]


def _m3_session_decode_run(mut s: Mamba3DecodeSession, a: List[Int], steps: Int) raises -> Int:
    """N tokens in one call (decode plan D2): `a` as the step's, x and y
    (B, N, d_model), the last token's four reports."""
    var words = s.prog.value().decode(a[0], a[1], steps, [a[2], a[3], a[4], a[5]])
    s.pending = words[1] == 1
    return words[0]


def _m3_session_export_run(mut s: Mamba3DecodeSession, a: List[Int]) raises -> Int:
    var words = s.prog.value().export_state(a)
    s.pending = words[1] == 1
    return words[0]


def _mamba3_run_owned(mut s: Mamba3PrefillSession, a: List[Int], b: Int, l: Int, dm: Int, q0: Int, pend: Int) raises -> Int:
    """The carried-state forward or decode step on the session's installed
    owned snapshot (NN40): `mamba_block_call` over the snapshot's host copy
    (stable while it is installed, so the cached program keeps it on the
    device and never compares a changed byte). `a` is the 25-slot list; its
    nine weight slots are not read."""
    comptime if not NN40_OWNED_WEIGHTS:
        raise Error("mamba3 owned weights: NN40 arm is disabled")
    if not s.owned_weights or not s.w or s.dm != dm:
        raise Error("mamba3 owned weights: install a matching snapshot first")
    var lens = _m3_weight_lens(Mamba3Dims.of(dm))
    var weights = List[Int](capacity=9)
    var off = 0
    for i in range(9):  # small-loop(the nine Mamba-3 weight pieces of the snapshot copy)
        weights.append(Int(s.w_host.unsafe_ptr()) + 4 * off)
        off += lens[i]
    var state = List[Int](capacity=10)
    for i in range(10):  # small-loop(the ten Mamba-3 state addresses)
        state.append(a[10 + i])
    var words = mamba_block_call(OPS_M3_STEP, weights, state, [q0, pend], a[0], a[20],
                                 [a[21], a[22], a[23], a[24]], b, l, dm, List[Float32](), False)
    return words[0]


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
    """The discard-state prefill on the session: `a` is the fresh entry's 25-slot
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
    # The snapshot's host copy (lane x5): the owned carried-state calls run
    # the cached block program over it (`_mamba3_run_owned`).
    s.w_host = _m3_weights_host_copy(a, _m3_weight_lens(dims))
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
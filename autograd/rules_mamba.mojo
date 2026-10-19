# SPDX-License-Identifier: Apache-2.0
# Mamba-1 / Mamba-2 block-stack backward on the tape (-D MOJOLEARN_AUTOGRAD_MAMBA,
# default off; the binding's hand path, modeling_mamba_prefill_backward.mojo::
# mamba1_prefill_backward and mamba2_prefill_backward.mojo::
# mamba2_prefill_backward, is untouched). docs/plans/AUTOGRAD_API.md (c) 3.
#
# ============================================================ LAUNCH INVENTORY
# Hand path, Mamba-1 (one block, zero state; forward = mamba_block_forward):
#   B1  proj_a PROJ_OUT  dg   = dres W_out            proj_b PROJ_OUT dW_out
#   B2  mamba_bwd_gate        dsk, dz, du_D, pD  from dg, skip_out, silu_out, z, D
#   B3  selective_scan_checkpoint_fn  h_ckpt (state_in.h, u, delta, A, B)
#   B4  selective_scan_bwd_scan       dh, du_s, ddelta, scan_w  (dy = dsk)
#   B5  mamba_bwd_dbc                 dCm, dBm
#   B6  mamba_bwd_da (batch partials + pinned fold), mamba_bwd_da_log -> dA_log
#   B7  mamba_bwd_ddtp        ddtp = ddelta * softplus'(dt_proj + b_dt)
#   B8  proj_a PROJ_DT ddtl   proj_b PROJ_DT dW_dt    reduce RED_DT_BIAS db_dt
#   B9  concat_xp             dXP = [ddtl | dBm | dCm]
#   B10 proj_a PROJ_X du_x    proj_b PROJ_X dW_x
#   B11 mamba_bwd_du_join     du = (du_D + du_s) + du_x, dconv = du * silu'(conv)
#   B12 mamba_bwd_dhin        dhin (conv input), reduce RED_CONV_BIAS
#   B13 4 x (conv_tap_product, reduce RED_CONV_W_TAPk), m1_conv_w_interleave
#   B14 concat_p              dP = [dhin | dz]
#   B15 proj_a PROJ_IN dnrm   proj_b PROJ_IN dW_in
#   B16 mamba_bwd_norm        dx = dres + norm'(dnrm)  (T8: ONE add, dres left),
#                             pW -> reduce RED_NORM_W
#   B17 reduce RED_D          dD from pD
# Hand path, Mamba-2 (one block; forward = mamba2_block_forward):
#   C1  tail_into     proj_a/proj_b PROJ2_OUT   d_gnorm, dW_out
#   C2  gnorm_into    gated RMSNorm backward -> d_gate, reduce RED2_GNORM_W
#   C3  silu_gate     d_skip, d_z
#   C4  d_skip_into   d_scan, d_x_from_d, reduce RED2_D
#   C5  SSD: s18_direct_dpass, reverse_chunk_state (zero final cotangent),
#       cstate_ddecay, reduce_scale_product, reverse_cumsum_and_da,
#       ydiag_xd_and_partial_dt, postconv_merge
#   C6  conv_backward_prefill    d_in_xbc, d_w, d_b
#   C7  input_projection: pack [d_z | d_xbc | d_dt], proj_a/proj_b PROJ2_IN
#   C8  block_norm: gnorm kernel at the block norm, reduce RED2_NORM_W,
#       residual join d_block_x = d_norm_x + d_residual
#
# ============================================================ THE TAPE GRAPH
# The forward is the hand path's (mamba_block_forward / mamba2_block_forward,
# stage buffers recorded per block); the tape records nodes over those stage
# buffers and never recomputes a stage. Per block k, with x_k the block input:
#   Mamba-1: NRM = M1_NORM(x, norm_w)          P  = M1_PROJ_IN(NRM, w_in)
#            U   = M1_CONV(P, conv_w, conv_b)  XP = M1_PROJ_X(U, w_x)
#            DEL = M1_DT(XP, w_dt, b_dt)       Y  = M1_SCAN(U, DEL, XP, A_log)
#            G   = M1_GATE(Y, U, P, D)         O  = M1_PROJ_OUT(G, w_out)
#            x_{k+1} = M1_RESID(x_k, O)        (the residual, recorded)
#   Mamba-2: NRM = M2_NORM(x, norm_w)          P  = M2_PROJ_IN(NRM, w_in)
#            SK  = M2_SSD(P, conv_w, conv_b, dt_bias, A_log, D)
#            G   = M2_GATE(SK, P, gnorm_w)     O  = M2_PROJ_OUT(G, w_out)
#            x_{k+1} = M2_RESID(x_k, O)
# Blocks chain through the residual tensor (the stack); one
# backward_with_rules from the last block's output with the caller's dres.
#
# Every custom rule launches the hand path's kernels with the hand path's
# arguments, on the hand path's buffers, and nothing else:
#   * Residual (B16, C8). M*_RESID records x + O (the forward's residual
#     stage) and launches nothing. O's gradient storage is the residual's
#     (custom_into on both), as B1 / C1 read dres itself; the last block's is
#     the caller's dres, the seed, used in place (no seed copy). x's gradient
#     is written whole by the block's norm (BackwardRules.writes_total), which
#     runs B16 / C8 with that residual gradient as its dres / d_residual.
#   * du's three-way join (B11, DEVIATION 1075). GATE leaves du_D, SCAN du_s
#     in B11's term buffers; X_PROJ, U's last consumer in the tape's reverse
#     order, runs B10 into du_x and then B11 itself into U's gradient (and
#     dconv for the conv rule): (du_D + du_s) + du_x, the hand path's call.
#   * The concatenations (B9 dXP, B14 dP, C7's pack) run once, by the input's
#     last consumer, from the other consumers' term buffers: DT packs
#     [ddtl | dBm | dCm], CONV [dhin | dz], SSD [d_z | d_xbc | d_dt].
#   * Mamba-2's leaves, norm, gate and skip gradients are recorded as the hand
#     path's output fields (variable_into / custom_into), so the rules' copies
#     between the tape and those fields find one buffer and do not run.
#   * zeroed_destination is False everywhere: every kernel stores its whole
#     destination (the hand path's buffers start zeroed, but none of its
#     kernels reads one back).
# Expected result: the same bits as the hand path on every gradient
# (mamba/checks/mamba_tape_backward_check.mojo compares them word for word),
# now including the subnormal cases the old spelling's ftz copies and -0.0
# operands could move. No new kernel and no new opcode.
#
# ============================================================ LAUNCH COUNT
# Per block, from the code; H1 / H2 = the hand path's launches (B1-B17 /
# C1-C8, each launcher's own count), one stack backward:
#   Mamba-1 tape, before: H1 + 31: 22 zero fills (one per custom input), 5
#     accumulates (U twice, XP, P, x), 2 add-VJP copies, 2 extra concats
#     (B9 and B14 ran once per consumer with -0.0 operands); plus 1 forward
#     add launch per block and 1 seed copy per stack.
#   Mamba-2 tape, before: H2 + 31: 15 zero fills, 12 rule copies (into and
#     out of the hand path's fields), 1 accumulate (P), 2 add-VJP copies, 1
#     extra C7 pack; plus 1 forward add per block and 1 seed copy per stack.
#   After, both: H1 / H2 exactly; no forward add, no seed copy.
#
# Mamba-3 is not on the tape: its backward (mamba3_prefill_backward.mojo)
# shares neither the M1 kernels nor the M2 tail; it stays on its hand path.
from std.sys.compile import is_defined

from max.gpu.host import DeviceBuffer, DeviceContext

from autograd.tape import Tape, Tensor, BackwardRules
from sequence.exec_trait import Exec
from sequence.exec_device import DeviceExec
from sequence.ops import FP, Args, OP_COPY

from core.identity_trace import IdentityTrace
from mamba.impl.modules.mamba3_refusal import mamba_refuse_grad_output

# ---- Mamba-1
from mamba.checks.mamba_fixture import D_CONV, D_STATE, MambaDims, MambaWeights
from mamba.checks.mamba_backward import (
    PROJ_OUT,
    PROJ_DT,
    PROJ_IN,
    PROJ_X,
    RED_CONV_BIAS,
    RED_CONV_W_TAP0,
    RED_CONV_W_TAP1,
    RED_CONV_W_TAP2,
    RED_CONV_W_TAP3,
    RED_D,
    RED_DT_BIAS,
    RED_NORM_W,
    mamba_backward_proj_a_into,
    mamba_backward_proj_b_into,
    mamba_backward_reduce_into,
    mamba_reduction_tap,
    mamba_backward_workspace_max_floats,
)
from mamba.impl.modeling.modeling_mamba import (
    MambaDeviceStages,
    MambaDeviceState,
    MambaDeviceWeights,
    mamba_block_forward,
    mamba_download,
    mamba_upload,
    mamba_zeros,
)
from mamba.impl.modeling.modeling_mamba_backward import (
    mamba_bwd_concat_xp_into,
    mamba_bwd_concat_p_into,
    mamba_bwd_conv_tap_product_into,
    mamba_bwd_ddtp_into,
    mamba_bwd_dhin_into,
    mamba_bwd_du_join_into,
    mamba_bwd_gate_into,
    mamba_bwd_norm_into,
)
from mamba.impl.ops.selective_scan_backward import (
    bwd_da_partial_floats,
    bwd_dh_floats,
    bwd_h_checkpoint_floats,
    mamba_bwd_da_into,
    mamba_bwd_da_log_into,
    mamba_bwd_dbc_into,
    selective_scan_bwd_scan_into,
    selective_scan_checkpoint_fn,
)
from mamba.impl.modeling.modeling_mamba_prefill_backward import (
    Mamba1PrefillGradients,
    m1_conv_w_interleave_kernel,
)

# ---- Mamba-2
from mamba.checks.mamba2_fixture import M2_D_STATE, M2_HEADDIM, M2_D_CONV, Mamba2Dims, Mamba2Weights
from mamba.checks.mamba2_backward import PROJ2_OUT, PROJ2_IN, mamba2_backward_proj_a_into, mamba2_backward_proj_b_into
from mamba.impl.modules.mamba2 import (
    Mamba2DeviceStages,
    Mamba2DeviceState,
    Mamba2DeviceWeights,
    allocate_inference_cache,
    mamba2_block_forward,
)
from mamba.impl.modules.mamba2_backward import (
    M2_BWD_TPB,
    Mamba2BackwardTail,
    mamba2_backward_d_skip_into,
    mamba2_backward_gnorm_into,
    mamba2_backward_silu_gate_into,
    mamba2_backward_block_norm_into,
    mamba2_pack_inproj_grad_kernel,
)
from mamba.impl.modules.ssd_minimal import m2_q_eff
from mamba.impl.ops.mamba2_ssd_backward import (
    Mamba2SSDBackwardState,
    Mamba2SSDDiscretizeBackward,
    Mamba2ConvBackward,
    Mamba2SSDScaleReduction,
    mamba2_reduce_scale_product_into,
    mamba2_reverse_cumsum_and_da_into,
    mamba2_ydiag_xd_and_partial_dt_into,
    mamba2_reverse_chunk_state_into,
    mamba2_cstate_ddecay_into,
    mamba2_postconv_merge_into,
    mamba2_conv_backward_prefill_into,
    mamba2_s18_direct_dpass_into,
)
from mamba.impl.modules.mamba2_prefill_backward import Mamba2PrefillGradients

comptime AUTOGRAD_MAMBA = is_defined["MOJOLEARN_AUTOGRAD_MAMBA"]()

# Tags in a range of their own, clear of LayerNorm (1) and recurrent (16-18).
comptime M1_NORM = 48
comptime M1_PROJ_IN = 49
comptime M1_CONV = 50
comptime M1_PROJ_X = 51
comptime M1_DT = 52
comptime M1_SCAN = 53
comptime M1_GATE = 54
comptime M1_PROJ_OUT = 55
comptime M2_NORM = 56
comptime M2_PROJ_IN = 57
comptime M2_SSD = 58
comptime M2_GATE = 59
comptime M2_PROJ_OUT = 60
# x_{k+1} = x_k + O recorded as a launch-free node (see the header)
comptime M1_RESID = 61
comptime M2_RESID = 62
comptime MAMBA_TAG_FIRST = M1_NORM
comptime MAMBA_TAG_LAST = 63

comptime _TPB = 128


# ------------------------------------------------------------------ helpers
def _fp(buf: DeviceBuffer[DType.float32]) -> FP:
    """A Mamba device buffer's address as the tape's pointer type."""
    return FP(unsafe_from_address=Int(buf.unsafe_ptr()))


@always_inline
def _kp(p: FP) -> MutPointer[Float32, MutAnyOrigin]:
    return p.unsafe_origin_cast[MutAnyOrigin]()


def _view(ctx: DeviceContext, p: FP, n: Int) raises -> DeviceBuffer[DType.float32]:
    """A non-owning device buffer over `n` floats of tape storage at `p`
    (sequence/moe_grouped.mojo::_view's construction)."""
    return DeviceBuffer[DType.float32](ctx, p.unsafe_origin_cast[MutAnyOrigin](), n if n > 0 else 1, owning=False)


# Lane ag-pass6-host (2026-10-19, PASS6_IDEAS idea K), default on: the
# tape's host work around the hand kernels. (1) Block k + 1's input (acts[k +
# 1]) is a view of block k's residual_out, the buffer the forward wrote,
# instead of its own buffer and a device copy per block (and the zero fill
# and wait of that buffer's allocation). (2) The export queues every
# gradient download (DeviceExec.download_async for tape storage, one
# enqueued copy for the stack's own buffers) and drains once: Mamba-1's 11 +
# 2 synchronous downloads a block and Mamba-2's 8 + 2 become one wait a
# call. Bits: unchanged (the same kernels read the same words; the copies
# are bit copies either way). -D MOJOLEARN_AUTOGRAD_MAMBA_HOST_OFF keeps the
# per-block copy and the synchronous downloads (arm A).
comptime AUTOGRAD_MAMBA_HOST = not is_defined["MOJOLEARN_AUTOGRAD_MAMBA_HOST_OFF"]()


def _act_after(ctx: DeviceContext, mut residual_out: DeviceBuffer[DType.float32], n: Int) raises -> DeviceBuffer[DType.float32]:
    """Block k + 1's input under AUTOGRAD_MAMBA_HOST: a view of the first n
    words of block k's residual_out (the stage outlives the stack's acts)."""
    return residual_out.create_sub_buffer[DType.float32](0, n)


def _grad_queued(tape: Tape, mut ex: DeviceExec, x: Tensor) raises -> List[Float32]:
    """`Tape.gradients` with the download queued (DeviceExec.download_async):
    the list holds the gradient after the executor's next sync."""
    var p = tape.gradient(x)
    var n = tape.rows(x) * tape.cols(x)
    var result = List[Float32](length=n, fill=0)
    ex.download_async(FP(unsafe_from_address=Int(result.unsafe_ptr())), p, n)
    return result^


def _download_queued(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int) raises -> List[Float32]:
    """`mamba_download` without its wait: the copy into the list is queued
    on ctx and lands at the caller's next synchronize (the list's storage
    does not move when the list is moved)."""
    var out = List[Float32](length=n, fill=Float32(0.0))
    if n == 0:
        return out^
    var view = buf.create_sub_buffer[DType.float32](0, n)
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=view)
    _ = view^
    return out^


def _copy[E: Exec](mut ex: E, dst: FP, src: FP, n: Int) raises:
    """dst[0:n] = src[0:n]: OP_COPY, a copy of bits, in stream order."""
    var a = Args()
    a.p0 = src
    a.p1 = dst
    ex.launch[OP_COPY](a, n)


def _d2d(ctx: DeviceContext, dst: DeviceBuffer[DType.float32], src: DeviceBuffer[DType.float32], n: Int) raises:
    """dst[0:n] = src[0:n], device to device (training/samba_resident.mojo)."""
    if n <= 0:
        return
    ctx.enqueue_copy(
        dst_buf=dst.create_sub_buffer[DType.float32](0, n),
        src_buf=src.create_sub_buffer[DType.float32](0, n),
    )


def _saved(k: Int, out_cols: Int) -> Args:
    var a = Args()
    a.i0 = k
    a.i1 = out_cols
    return a^


def _saved_norm(k: Int, out_cols: Int, residual_grad: FP) -> Args:
    """A norm node's slots: block k, its width, and the gradient arriving at
    the block's residual output (the norm adds it into x's gradient)."""
    var a = _saved(k, out_cols)
    a.p2 = residual_grad
    return a^


def _copy_unless_same[E: Exec](mut ex: E, dst: FP, src: FP, n: Int) raises:
    """dst = src (OP_COPY, bits) unless the driver made them one buffer."""
    if Int(dst) != Int(src):
        _copy(ex, dst, src, n)


# ================================================================== Mamba-1
struct Mamba1TapeScratch(Movable):
    """One block's hand-path intermediates (the buffers mamba1_prefill_backward
    allocates), zeroed as it zeroes them."""
    var drstd: DeviceBuffer[DType.float32]
    var norm_product: DeviceBuffer[DType.float32]
    var h_ckpt: DeviceBuffer[DType.float32]
    var dh: DeviceBuffer[DType.float32]
    var scan_w: DeviceBuffer[DType.float32]
    var dcm: DeviceBuffer[DType.float32]
    var dbm: DeviceBuffer[DType.float32]
    var da: DeviceBuffer[DType.float32]
    var da_partial: DeviceBuffer[DType.float32]
    var ddtp: DeviceBuffer[DType.float32]
    var ddtl: DeviceBuffer[DType.float32]
    var du: DeviceBuffer[DType.float32]
    var dconv: DeviceBuffer[DType.float32]
    var dhin: DeviceBuffer[DType.float32]
    var tp0: DeviceBuffer[DType.float32]
    var tp1: DeviceBuffer[DType.float32]
    var tp2: DeviceBuffer[DType.float32]
    var tp3: DeviceBuffer[DType.float32]
    var tg0: DeviceBuffer[DType.float32]
    var tg1: DeviceBuffer[DType.float32]
    var tg2: DeviceBuffer[DType.float32]
    var tg3: DeviceBuffer[DType.float32]
    var product_d: DeviceBuffer[DType.float32]
    var dz: DeviceBuffer[DType.float32]
    # B11's three du terms (the hand path's du_d, du_s, du_x)
    var du_d: DeviceBuffer[DType.float32]
    var du_s: DeviceBuffer[DType.float32]
    var du_x: DeviceBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext, b: Int, l: Int, dims: MambaDims) raises:
        var m = b * l
        var di = dims.d_inner
        self.drstd = mamba_zeros(ctx, m)
        self.norm_product = mamba_zeros(ctx, m * dims.d_model)
        self.h_ckpt = mamba_zeros(ctx, bwd_h_checkpoint_floats(b, l, di))
        self.dh = mamba_zeros(ctx, bwd_dh_floats(b, l, di))
        self.scan_w = mamba_zeros(ctx, m * di)
        self.dcm = mamba_zeros(ctx, m * D_STATE)
        self.dbm = mamba_zeros(ctx, m * D_STATE)
        self.da = mamba_zeros(ctx, di * D_STATE)
        self.da_partial = mamba_zeros(ctx, bwd_da_partial_floats(b, di))
        self.ddtp = mamba_zeros(ctx, m * di)
        self.ddtl = mamba_zeros(ctx, m * dims.dt_rank)
        self.du = mamba_zeros(ctx, m * di)
        self.dconv = mamba_zeros(ctx, m * di)
        self.dhin = mamba_zeros(ctx, m * di)
        self.tp0 = mamba_zeros(ctx, m * di)
        self.tp1 = mamba_zeros(ctx, m * di)
        self.tp2 = mamba_zeros(ctx, m * di)
        self.tp3 = mamba_zeros(ctx, m * di)
        self.tg0 = mamba_zeros(ctx, di)
        self.tg1 = mamba_zeros(ctx, di)
        self.tg2 = mamba_zeros(ctx, di)
        self.tg3 = mamba_zeros(ctx, di)
        self.product_d = mamba_zeros(ctx, m * di)
        self.dz = mamba_zeros(ctx, m * di)
        self.du_d = mamba_zeros(ctx, m * di)
        self.du_s = mamba_zeros(ctx, m * di)
        self.du_x = mamba_zeros(ctx, m * di)


struct Mamba1TapeStack(Movable):
    """Everything a Mamba-1 stack's forward recorded, per block, plus the
    shared backward workspaces. acts[0] is the stack input, acts[k + 1] a
    copy of block k's residual output (the next block's input)."""
    var ctx: DeviceContext
    var dims: MambaDims
    var b: Int
    var l: Int
    var w: List[MambaDeviceWeights]
    var state_in: List[MambaDeviceState]
    var state: List[MambaDeviceState]
    var stages: List[MambaDeviceStages]
    var scratch: List[Mamba1TapeScratch]
    var acts: List[DeviceBuffer[DType.float32]]
    var ws: DeviceBuffer[DType.float32]
    var ones: DeviceBuffer[DType.float32]

    def __init__(
        out self, ctx: DeviceContext, weights: List[MambaWeights], input: List[Float32], b: Int, l: Int,
    ) raises:
        if len(weights) < 1:
            raise Error("mamba1 tape backward: the stack needs at least one block")
        self.ctx = ctx.copy()
        self.dims = weights[0].dims.copy()
        self.b = b
        self.l = l
        var m = b * l
        var dm = self.dims.d_model
        var di = self.dims.d_inner
        self.w = List[MambaDeviceWeights]()
        self.state_in = List[MambaDeviceState]()
        self.state = List[MambaDeviceState]()
        self.stages = List[MambaDeviceStages]()
        self.scratch = List[Mamba1TapeScratch]()
        self.acts = List[DeviceBuffer[DType.float32]]()
        self.acts.append(mamba_upload(ctx, input))
        for k in range(len(weights)):  # driver-loop: over blocks
            if weights[k].dims.d_model != dm:
                raise Error("mamba1 tape backward: every block of a stack has the same d_model")
            self.w.append(MambaDeviceWeights(ctx, weights[k]))
            self.state_in.append(MambaDeviceState(ctx, b, self.dims))
            self.state.append(MambaDeviceState(ctx, b, self.dims))
            self.stages.append(MambaDeviceStages(ctx, b, l, self.dims))
            self.scratch.append(Mamba1TapeScratch(ctx, b, l, self.dims))
            comptime if AUTOGRAD_MAMBA_HOST:
                self.acts.append(_act_after(ctx, self.stages[k].residual_out, m * dm))
            else:
                self.acts.append(mamba_zeros(ctx, m * dm))
        self.ws = mamba_zeros(ctx, mamba_backward_workspace_max_floats(self.dims, m))
        # mamba1_prefill_backward's ones: mamba_zeros' guard layout, body set to 1
        self.ones = mamba_zeros(ctx, m)
        var ones_body = self.ones.create_sub_buffer[DType.float32](0, m)
        ones_body.enqueue_fill(Float32(1.0))
        _ = ones_body^
        ctx.synchronize()


def _m1_norm_bwd(mut S: Mamba1TapeStack, k: Int, saved: Args, up: FP, gradients: List[FP]) raises:
    # B16 with the gradient at the block's residual output (saved.p2) as its
    # dres, so x's whole gradient dres + norm'(dnrm) lands in gradients[0]
    # (writes_total), then RED_NORM_W.
    var ctx = S.ctx.copy()
    var dims = S.dims.copy()
    var m = S.b * S.l
    var dm = dims.d_model
    ref st = S.stages[k]
    ref sc = S.scratch[k]
    ref w = S.w[k]
    var dx = _view(ctx, gradients[0], m * dm)
    var dnrm = _view(ctx, up, m * dm)
    var dres = _view(ctx, saved.p2, m * dm)
    mamba_bwd_norm_into(
        ctx, dx, sc.drstd, sc.norm_product, dres, dnrm, S.acts[k],
        st.norm_out, w.norm_w, st.norm_sumsq, m, dm,
    )
    var dwn = _view(ctx, gradients[1], dm)
    mamba_backward_reduce_into(ctx, dwn, sc.norm_product, S.ones, S.ws, RED_NORM_W, dims, m)


def _m1_proj_bwd(mut S: Mamba1TapeStack, k: Int, which: Int, saved: Args, up: FP,
                 gradients: List[FP], rows: List[Int], cols: List[Int]) raises:
    # B1 / B10 / B15: the activation (proj_a) and weight (proj_b) routings.
    var ctx = S.ctx.copy()
    var dims = S.dims.copy()
    var m = S.b * S.l
    ref st = S.stages[k]
    ref w = S.w[k]
    var din = _view(ctx, gradients[0], rows[0] * cols[0])
    var dw = _view(ctx, gradients[1], rows[1] * cols[1])
    var dout = _view(ctx, up, m * saved.i1)
    if which == PROJ_IN:
        mamba_backward_proj_a_into(ctx, din, dout, w.w_in, S.ws, PROJ_IN, dims, m)
        mamba_backward_proj_b_into(ctx, dw, dout, st.norm_out, S.ws, PROJ_IN, dims, m)
    elif which == PROJ_X:
        # B10 into du_x, then B11: U's last consumer joins du = (du_D + du_s)
        # + du_x straight into U's gradient (writes_total) and forms dconv
        ref sc = S.scratch[k]
        var di = dims.d_inner
        mamba_backward_proj_a_into(ctx, sc.du_x, dout, w.w_x, S.ws, PROJ_X, dims, m)
        mamba_backward_proj_b_into(ctx, dw, dout, st.silu_out, S.ws, PROJ_X, dims, m)
        mamba_bwd_du_join_into(ctx, din, sc.dconv, sc.du_d, sc.du_s, sc.du_x, st.conv_out, m, di)
    else:
        mamba_backward_proj_a_into(ctx, din, dout, w.w_out, S.ws, PROJ_OUT, dims, m)
        mamba_backward_proj_b_into(ctx, dw, dout, st.gate_out, S.ws, PROJ_OUT, dims, m)


def _m1_conv_bwd(mut S: Mamba1TapeStack, k: Int, up: FP, gradients: List[FP]) raises:
    # B12, B13, B14 (dconv from the X_PROJ rule's B11; dP = [dhin | dz], P's
    # whole gradient, writes_total).
    var ctx = S.ctx.copy()
    var dims = S.dims.copy()
    var b = S.b
    var l = S.l
    var m = b * l
    var di = dims.d_inner
    ref st = S.stages[k]
    ref sc = S.scratch[k]
    ref w = S.w[k]
    mamba_bwd_dhin_into(ctx, sc.dhin, sc.dconv, w.conv_w, b, l, di)
    var dcb = _view(ctx, gradients[2], di)
    mamba_backward_reduce_into(ctx, dcb, sc.dconv, S.ones, S.ws, RED_CONV_BIAS, dims, m)
    mamba_bwd_conv_tap_product_into(ctx, sc.tp0, sc.dconv, st.in_proj, S.state_in[k].conv_win, b, l, di, 0)
    mamba_backward_reduce_into(ctx, sc.tg0, sc.tp0, S.ones, S.ws, RED_CONV_W_TAP0, dims, m)
    mamba_bwd_conv_tap_product_into(ctx, sc.tp1, sc.dconv, st.in_proj, S.state_in[k].conv_win, b, l, di, 1)
    mamba_backward_reduce_into(ctx, sc.tg1, sc.tp1, S.ones, S.ws, RED_CONV_W_TAP1, dims, m)
    mamba_bwd_conv_tap_product_into(ctx, sc.tp2, sc.dconv, st.in_proj, S.state_in[k].conv_win, b, l, di, 2)
    mamba_backward_reduce_into(ctx, sc.tg2, sc.tp2, S.ones, S.ws, RED_CONV_W_TAP2, dims, m)
    mamba_bwd_conv_tap_product_into(ctx, sc.tp3, sc.dconv, st.in_proj, S.state_in[k].conv_win, b, l, di, 3)
    mamba_backward_reduce_into(ctx, sc.tg3, sc.tp3, S.ones, S.ws, RED_CONV_W_TAP3, dims, m)
    # the [d_inner, D_CONV] layout copy straight into the tape's gradient
    ctx.enqueue_function[m1_conv_w_interleave_kernel](
        _kp(gradients[1]),
        _kp(_fp(sc.tg0)),
        _kp(_fp(sc.tg1)),
        _kp(_fp(sc.tg2)),
        _kp(_fp(sc.tg3)),
        Int32(mamba_reduction_tap(RED_CONV_W_TAP0)),
        Int32(mamba_reduction_tap(RED_CONV_W_TAP1)),
        Int32(mamba_reduction_tap(RED_CONV_W_TAP2)),
        Int32(mamba_reduction_tap(RED_CONV_W_TAP3)),
        Int32(di),
        grid_dim=((di + _TPB - 1) // _TPB, 1, 1),
        block_dim=(_TPB, 1, 1),
    )
    var dp = _view(ctx, gradients[0], m * 2 * di)
    mamba_bwd_concat_p_into(ctx, dp, sc.dhin, sc.dz, m, di)


def _m1_dt_bwd(mut S: Mamba1TapeStack, k: Int, up: FP, gradients: List[FP]) raises:
    # B7, B8, then B9: dXP = [ddtl | dBm | dCm], XP's whole gradient
    # (writes_total; the scan rule, XP's other consumer, left dBm and dCm).
    var ctx = S.ctx.copy()
    var dims = S.dims.copy()
    var m = S.b * S.l
    var di = dims.d_inner
    ref st = S.stages[k]
    ref sc = S.scratch[k]
    ref w = S.w[k]
    var ddelta = _view(ctx, up, m * di)
    mamba_bwd_ddtp_into(ctx, sc.ddtp, ddelta, st.dt_proj, w.b_dt, m, di)
    mamba_backward_proj_a_into(ctx, sc.ddtl, sc.ddtp, w.w_dt, S.ws, PROJ_DT, dims, m)
    var dwdt = _view(ctx, gradients[1], di * dims.dt_rank)
    mamba_backward_proj_b_into(ctx, dwdt, sc.ddtp, st.dt_low, S.ws, PROJ_DT, dims, m)
    var dbdt = _view(ctx, gradients[2], di)
    mamba_backward_reduce_into(ctx, dbdt, sc.ddtp, S.ones, S.ws, RED_DT_BIAS, dims, m)
    var dxp = _view(ctx, gradients[0], m * dims.x_proj_rows())
    mamba_bwd_concat_xp_into(ctx, dxp, sc.ddtl, sc.dbm, sc.dcm, m, dims)


def _m1_scan_bwd(mut S: Mamba1TapeStack, k: Int, up: FP, gradients: List[FP]) raises:
    # B3-B6. du_s goes to B11's term buffer and dBm / dCm stay for B9: U's
    # and XP's gradients are written whole by their last consumers.
    var ctx = S.ctx.copy()
    var dims = S.dims.copy()
    var b = S.b
    var l = S.l
    var m = b * l
    var di = dims.d_inner
    ref st = S.stages[k]
    ref sc = S.scratch[k]
    var dy = _view(ctx, up, m * di)
    selective_scan_checkpoint_fn(
        ctx, sc.h_ckpt, S.state_in[k].h, st.silu_out, st.softplus_out, st.a_out, st.b_mat, b, l, di,
    )
    var dd = _view(ctx, gradients[1], m * di)
    selective_scan_bwd_scan_into(
        ctx, sc.dh, sc.du_s, dd, sc.scan_w, dy, st.c_mat, st.b_mat,
        st.softplus_out, st.a_out, st.silu_out, sc.h_ckpt, b, l, di,
    )
    mamba_bwd_dbc_into(ctx, sc.dcm, sc.dbm, dy, sc.h_ckpt, sc.scan_w, sc.dh, b, l, di)
    mamba_bwd_da_into(
        ctx, sc.da, sc.da_partial, sc.dh, st.softplus_out, st.a_out,
        st.silu_out, st.b_mat, sc.h_ckpt, b, l, di,
    )
    var dal = _view(ctx, gradients[3], di * D_STATE)
    mamba_bwd_da_log_into(ctx, dal, sc.da, st.a_out, di)


def _m1_gate_bwd(mut S: Mamba1TapeStack, k: Int, up: FP, gradients: List[FP]) raises:
    # B2 (du_D to B11's term buffer, dz kept for B14), B17.
    var ctx = S.ctx.copy()
    var dims = S.dims.copy()
    var m = S.b * S.l
    var di = dims.d_inner
    ref st = S.stages[k]
    ref sc = S.scratch[k]
    ref w = S.w[k]
    var dg = _view(ctx, up, m * di)
    var dsk = _view(ctx, gradients[0], m * di)
    mamba_bwd_gate_into(
        ctx, dsk, sc.dz, sc.du_d, sc.product_d, dg, st.skip_out, st.silu_out,
        st.in_proj, w.d_skip, m, di,
    )
    var ddsk = _view(ctx, gradients[3], di)
    mamba_backward_reduce_into(ctx, ddsk, sc.product_d, S.ones, S.ws, RED_D, dims, m)


@fieldwise_init
struct Mamba1Rules(BackwardRules, Copyable, Movable):
    """Backward adapters for a Mamba-1 stack's custom nodes."""
    # The stack's address, untracked (a struct field may not carry
    # MutAnyOrigin); the driver keeps the stack alive through the backward.
    var s: MutPointer[Mamba1TapeStack, MutUntrackedOrigin]

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # Never: every kernel these rules launch stores every element of the
        # destination it is handed (the hand path's buffers start zeroed but
        # none of its kernels reads them back), and M1_RESID writes nothing.
        return False

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # The joins (header): x by its norm (B16), U by X_PROJ (B11), XP by
        # DT (B9), P by CONV (B14); the other consumers of those inputs only
        # leave their terms in the hand path's buffers.
        if tag == M1_NORM or tag == M1_RESID or tag == M1_PROJ_X or tag == M1_DT or tag == M1_CONV:
            return edge == 0
        if tag == M1_SCAN:
            return edge == 0 or edge == 2
        if tag == M1_GATE:
            return edge == 1 or edge == 2
        return False

    def supports(self, tag: Int) -> Bool:
        return (tag >= M1_NORM and tag <= M1_PROJ_OUT) or tag == M1_RESID

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        ref S = self.s[]
        var k = saved.i0
        if tag == M1_NORM:
            _m1_norm_bwd(S, k, saved, upstream, gradients)
        elif tag == M1_PROJ_IN:
            _m1_proj_bwd(S, k, PROJ_IN, saved, upstream, gradients, rows, cols)
        elif tag == M1_CONV:
            _m1_conv_bwd(S, k, upstream, gradients)
        elif tag == M1_PROJ_X:
            _m1_proj_bwd(S, k, PROJ_X, saved, upstream, gradients, rows, cols)
        elif tag == M1_DT:
            _m1_dt_bwd(S, k, upstream, gradients)
        elif tag == M1_SCAN:
            _m1_scan_bwd(S, k, upstream, gradients)
        elif tag == M1_GATE:
            _m1_gate_bwd(S, k, upstream, gradients)
        elif tag == M1_PROJ_OUT:
            _m1_proj_bwd(S, k, PROJ_OUT, saved, upstream, gradients, rows, cols)
        elif tag == M1_RESID:
            # no launch: O's gradient is this node's storage, x's is written
            # whole by its norm
            _copy_unless_same(ex, gradients[1], upstream, rows[1] * cols[1])
        else:
            raise Error("unknown mamba1 backward rule")


def mamba1_tape_backward_stack(
    weights: List[MambaWeights],
    input: List[Float32],
    grad_output: List[Float32],
    b: Int,
    l: Int,
    ctx: DeviceContext,
) raises -> List[Mamba1PrefillGradients]:
    """Zero-state VJP of a stack of Mamba-1 blocks, assembled by the tape.

    Block k's input is block k - 1's residual output; `grad_output` is the
    gradient at the last block's output. Entry k holds block k's parameter
    gradients and the gradient at its input (entry 0's `x` is the stack
    input's)."""
    if b <= 0 or l <= 0:
        raise Error("mamba1 backward: B and L must be positive")
    if len(weights) < 1:
        raise Error("mamba1 tape backward: the stack needs at least one block")
    var dm = weights[0].dims.d_model
    if len(input) != b * l * dm or len(grad_output) != len(input):
        raise Error("mamba1 backward: input and grad_output lengths must equal B*L*d_model")
    var m = b * l
    var nb = len(weights)
    var S = Mamba1TapeStack(ctx, weights, input, b, l)
    var dims = S.dims.copy()
    var di = dims.d_inner
    var xr = dims.x_proj_rows()
    var trace = IdentityTrace.disabled()
    for k in range(nb):  # driver-loop: over blocks
        mamba_block_forward(
            S.ctx, S.stages[k], S.state[k], S.w[k], S.acts[k], b, l, trace,
            String("m1.backward.tail"),
        )
        comptime if not AUTOGRAD_MAMBA_HOST:
            _d2d(S.ctx, S.acts[k + 1], S.stages[k].residual_out, m * dm)

    var dres = mamba_upload(ctx, grad_output)
    mamba_refuse_grad_output(ctx, String("mamba1"), dres, grad_output)

    var ex = DeviceExec(ctx)
    var tape = Tape()
    var xs = List[Tensor]()
    var params = List[Tensor]()
    # The gradient at block k's residual output: the caller's dres for the
    # last block (the seed, used in place), tape storage otherwise. O_k's
    # gradient shares it; block k's norm adds it into x_k's (B16).
    var dres_at = List[FP]()
    for k in range(nb):  # driver-loop: over blocks
        dres_at.append(_fp(dres) if k == nb - 1 else tape.reserve(ex, m * dm))
    var x_t = tape.variable(ex, _fp(S.acts[0]), m, dm)
    for k in range(nb):  # driver-loop: over blocks
        xs.append(x_t)
        ref st = S.stages[k]
        ref w = S.w[k]
        var norm_w = tape.variable(ex, _fp(w.norm_w), 1, dm)
        var w_in = tape.variable(ex, _fp(w.w_in), 2 * di, dm)
        var conv_w = tape.variable(ex, _fp(w.conv_w), di, D_CONV)
        var conv_b = tape.variable(ex, _fp(w.conv_b), 1, di)
        var w_x = tape.variable(ex, _fp(w.w_x), xr, di)
        var w_dt = tape.variable(ex, _fp(w.w_dt), di, dims.dt_rank)
        var b_dt = tape.variable(ex, _fp(w.b_dt), 1, di)
        var a_log = tape.variable(ex, _fp(w.a_log), di, D_STATE)
        var d_skip = tape.variable(ex, _fp(w.d_skip), 1, di)
        var w_out = tape.variable(ex, _fp(w.w_out), dm, di)
        for t in [norm_w, w_in, conv_w, conv_b, w_x, w_dt, b_dt, a_log, d_skip, w_out]:  # driver-loop: over parameters
            params.append(t)
        # Creation order fixes the tape's accumulation order (see the header):
        # U's consumers X_PROJ, SCAN, GATE; P's CONV, GATE; XP's DT, SCAN.
        var nrm = tape.custom(ex, [x_t, norm_w], _fp(st.norm_out), m, dm, M1_NORM, _saved_norm(k, dm, dres_at[k]))
        var p = tape.custom(ex, [nrm, w_in], _fp(st.in_proj), m, 2 * di, M1_PROJ_IN, _saved(k, 2 * di))
        var u = tape.custom(ex, [p, conv_w, conv_b], _fp(st.silu_out), m, di, M1_CONV, _saved(k, di))
        var xp = tape.custom(ex, [u, w_x], _fp(st.x_proj), m, xr, M1_PROJ_X, _saved(k, xr))
        var delta = tape.custom(ex, [xp, w_dt, b_dt], _fp(st.softplus_out), m, di, M1_DT, _saved(k, di))
        var y = tape.custom(ex, [u, delta, xp, a_log], _fp(st.scan_y), m, di, M1_SCAN, _saved(k, di))
        var g = tape.custom(ex, [y, u, p, d_skip], _fp(st.gate_out), m, di, M1_GATE, _saved(k, di))
        var o = tape.custom_into(ex, [g, w_out], _fp(st.out_proj), dres_at[k], m, dm, M1_PROJ_OUT, _saved(k, dm))
        x_t = tape.custom_into_shared(ex, [x_t, o], _fp(S.acts[k + 1]), o, m, dm, M1_RESID, _saved(k, dm))
    var seed = tape.variable(ex, _fp(dres), m, dm, False)
    var rules = Mamba1Rules(MutPointer[Mamba1TapeStack, MutUntrackedOrigin](unsafe_from_address=Int(MutPointer(to=S))))
    tape.backward_with_rules(ex, x_t, seed, rules)

    var result = List[Mamba1PrefillGradients]()
    comptime if AUTOGRAD_MAMBA_HOST:
        # idea K(2): every download queued, one drain
        for k in range(nb):  # driver-loop: over blocks
            var g = Mamba1PrefillGradients()
            var base = 10 * k
            g.norm_weight = _grad_queued(tape, ex, params[base + 0])
            g.in_proj_weight = _grad_queued(tape, ex, params[base + 1])
            g.conv1d_weight = _grad_queued(tape, ex, params[base + 2])
            g.conv1d_bias = _grad_queued(tape, ex, params[base + 3])
            g.x_proj_weight = _grad_queued(tape, ex, params[base + 4])
            g.dt_proj_weight = _grad_queued(tape, ex, params[base + 5])
            g.dt_proj_bias = _grad_queued(tape, ex, params[base + 6])
            g.A_log = _grad_queued(tape, ex, params[base + 7])
            g.D = _grad_queued(tape, ex, params[base + 8])
            g.out_proj_weight = _grad_queued(tape, ex, params[base + 9])
            g.x = _grad_queued(tape, ex, xs[k])
            g.stage_dB = _download_queued(ctx, S.scratch[k].dbm, m * D_STATE)
            g.stage_dC = _download_queued(ctx, S.scratch[k].dcm, m * D_STATE)
            result.append(g^)
        # the one wait: every queued copy lands (DeviceExec copies the
        # staged words into the lists here)
        ex.sync()
        tape.release(ex)
        ctx.synchronize()
        _ = rules^
        _ = dres^
        _ = S^
        return result^
    else:
        for k in range(nb):  # driver-loop: over blocks
            var g = Mamba1PrefillGradients()
            var base = 10 * k
            g.norm_weight = tape.gradients(ex, params[base + 0])
            g.in_proj_weight = tape.gradients(ex, params[base + 1])
            g.conv1d_weight = tape.gradients(ex, params[base + 2])
            g.conv1d_bias = tape.gradients(ex, params[base + 3])
            g.x_proj_weight = tape.gradients(ex, params[base + 4])
            g.dt_proj_weight = tape.gradients(ex, params[base + 5])
            g.dt_proj_bias = tape.gradients(ex, params[base + 6])
            g.A_log = tape.gradients(ex, params[base + 7])
            g.D = tape.gradients(ex, params[base + 8])
            g.out_proj_weight = tape.gradients(ex, params[base + 9])
            g.x = tape.gradients(ex, xs[k])
            g.stage_dB = mamba_download(ctx, S.scratch[k].dbm, m * D_STATE)
            g.stage_dC = mamba_download(ctx, S.scratch[k].dcm, m * D_STATE)
            result.append(g^)
        tape.release(ex)
        ctx.synchronize()
        _ = rules^
        _ = dres^
        _ = S^
        return result^


def mamba1_tape_backward(
    weights: MambaWeights,
    input: List[Float32],
    grad_output: List[Float32],
    b: Int,
    l: Int,
    ctx: DeviceContext,
) raises -> Mamba1PrefillGradients:
    """mamba1_prefill_backward's contract, assembled by the tape (one block)."""
    var stack = List[MambaWeights]()
    stack.append(weights.copy())
    var grads = mamba1_tape_backward_stack(stack, input, grad_output, b, l, ctx)
    return grads.pop(0)


# ================================================================== Mamba-2
struct Mamba2TapeStack(Movable):
    """A Mamba-2 stack's forward stages and the hand path's backward state,
    per block (mamba2_prefill_backward's allocations)."""
    var ctx: DeviceContext
    var dims: Mamba2Dims
    var b: Int
    var l: Int
    var dt_lo: Float32
    var dt_hi: Float32
    var w: List[Mamba2DeviceWeights]
    var state: List[Mamba2DeviceState]
    var stages: List[Mamba2DeviceStages]
    var acts: List[DeviceBuffer[DType.float32]]
    var tail: List[Mamba2BackwardTail]
    var ssd: List[Mamba2SSDBackwardState]
    var d_final: List[DeviceBuffer[DType.float32]]
    var scale: List[Mamba2SSDScaleReduction]
    var disc: List[Mamba2SSDDiscretizeBackward]
    var conv: List[Mamba2ConvBackward]

    def __init__(
        out self, ctx: DeviceContext, weights: List[Mamba2Weights], input: List[Float32],
        b: Int, l: Int, dt_lo: Float32, dt_hi: Float32,
    ) raises:
        if len(weights) < 1:
            raise Error("mamba2 tape backward: the stack needs at least one block")
        self.ctx = ctx.copy()
        self.dims = weights[0].dims.copy()
        self.b = b
        self.l = l
        self.dt_lo = dt_lo
        self.dt_hi = dt_hi
        var m = b * l
        var dm = self.dims.d_model
        self.w = List[Mamba2DeviceWeights]()
        self.state = List[Mamba2DeviceState]()
        self.stages = List[Mamba2DeviceStages]()
        self.acts = List[DeviceBuffer[DType.float32]]()
        self.tail = List[Mamba2BackwardTail]()
        self.ssd = List[Mamba2SSDBackwardState]()
        self.d_final = List[DeviceBuffer[DType.float32]]()
        self.scale = List[Mamba2SSDScaleReduction]()
        self.disc = List[Mamba2SSDDiscretizeBackward]()
        self.conv = List[Mamba2ConvBackward]()
        self.acts.append(mamba_upload(ctx, input))
        for k in range(len(weights)):  # driver-loop: over blocks
            if weights[k].dims.d_model != dm:
                raise Error("mamba2 tape backward: every block of a stack has the same d_model")
            self.w.append(Mamba2DeviceWeights(ctx, weights[k]))
            self.state.append(allocate_inference_cache(ctx, b, self.dims))
            self.stages.append(Mamba2DeviceStages(ctx, b, l, 0, self.dims))
            comptime if AUTOGRAD_MAMBA_HOST:
                self.acts.append(_act_after(ctx, self.stages[k].residual_out, m * dm))
            else:
                self.acts.append(mamba_zeros(ctx, m * dm))
            self.tail.append(Mamba2BackwardTail(ctx, self.dims, m))
        ctx.synchronize()


def _m2_pack(ctx: DeviceContext, dst: FP, z: FP, xbc: FP, dt: FP, m: Int, dims: Mamba2Dims) raises:
    """C7's pack kernel, [d_z | d_xbc | d_dt] into the tape's d(in_proj)."""
    var n = m * dims.d_in_proj()
    if n < 1:
        return
    ctx.enqueue_function[mamba2_pack_inproj_grad_kernel](
        _kp(dst), _kp(z), _kp(xbc), _kp(dt),
        Int32(m), Int32(dims.d_inner), Int32(dims.conv_dim()), Int32(dims.d_in_proj()), Int32(dims.nheads),
        grid_dim=((n + M2_BWD_TPB - 1) // M2_BWD_TPB, 1, 1),
        block_dim=(M2_BWD_TPB, 1, 1),
    )


def _m2_norm_bwd[E: Exec](
    mut ex: E, mut S: Mamba2TapeStack, k: Int, saved: Args, up: FP, gradients: List[FP],
) raises:
    # C8 with the gradient at the block's residual output (saved.p2) as its
    # d_residual: x's whole gradient d_norm_x + d_residual (writes_total).
    # The driver records the norm's gradient as T.d_norm, x's as
    # T.d_block_x and the weight's as T.d_block_w, so no copy runs.
    var ctx = S.ctx.copy()
    var dims = S.dims.copy()
    var m = S.b * S.l
    var dm = dims.d_model
    ref st = S.stages[k]
    ref T = S.tail[k]
    ref w = S.w[k]
    _copy_unless_same(ex, _fp(T.d_norm), up, m * dm)
    var residual = _view(ctx, saved.p2, m * dm)
    mamba2_backward_block_norm_into(ctx, T, residual, S.acts[k], st.norm_sumsq, w.norm_w, dims, m)
    _copy_unless_same(ex, gradients[0], _fp(T.d_block_x), m * dm)
    _copy_unless_same(ex, gradients[1], _fp(T.d_block_w), dm)


def _m2_proj_bwd(mut S: Mamba2TapeStack, k: Int, which: Int, saved: Args, up: FP,
                 gradients: List[FP], rows: List[Int], cols: List[Int]) raises:
    # C1 / C7's two GEMM routings (the pack is the tape's accumulation).
    var ctx = S.ctx.copy()
    var dims = S.dims.copy()
    var m = S.b * S.l
    ref st = S.stages[k]
    ref T = S.tail[k]
    ref w = S.w[k]
    var din = _view(ctx, gradients[0], rows[0] * cols[0])
    var dw = _view(ctx, gradients[1], rows[1] * cols[1])
    var dout = _view(ctx, up, m * saved.i1)
    if which == PROJ2_IN:
        mamba2_backward_proj_a_into(ctx, din, dout, w.w_in, T.workspace, PROJ2_IN, dims, m)
        mamba2_backward_proj_b_into(ctx, dw, dout, st.norm_out, T.workspace, PROJ2_IN, dims, m)
    else:
        mamba2_backward_proj_a_into(ctx, din, dout, w.w_out, T.workspace, PROJ2_OUT, dims, m)
        mamba2_backward_proj_b_into(ctx, dw, dout, st.gnorm_out, T.workspace, PROJ2_OUT, dims, m)


def _m2_gate_bwd[E: Exec](mut ex: E, mut S: Mamba2TapeStack, k: Int, up: FP, gradients: List[FP]) raises:
    # C2, C3; d_z stays in T.d_z for C7's pack, which the SSD rule (P's
    # other consumer) runs into P's gradient. The driver records G's
    # gradient as T.d_gnorm, SK's as T.d_skip and gnorm_w's as T.d_gnorm_w.
    var ctx = S.ctx.copy()
    var dims = S.dims.copy()
    var m = S.b * S.l
    var di = dims.d_inner
    ref st = S.stages[k]
    ref T = S.tail[k]
    ref w = S.w[k]
    _copy_unless_same(ex, _fp(T.d_gnorm), up, m * di)
    mamba2_backward_gnorm_into(ctx, T, st.gnorm_gate, st.gnorm_sumsq, w.gnorm_w, dims, m)
    _copy_unless_same(ex, gradients[2], _fp(T.d_gnorm_w), di)
    mamba2_backward_silu_gate_into(ctx, T, st.skip_out, st.in_proj, dims, m)
    _copy_unless_same(ex, gradients[0], _fp(T.d_skip), m * di)


def _m2_ssd_bwd[E: Exec](mut ex: E, mut S: Mamba2TapeStack, k: Int, up: FP, gradients: List[FP]) raises:
    # C4, C5, C6 in the hand path's order, then C7's pack [d_z | d_xbc |
    # d_dt] into P's whole gradient (writes_total). The driver records the
    # parameters' gradients as the hand path's output fields: no copy runs.
    var ctx = S.ctx.copy()
    var dims = S.dims.copy()
    var b = S.b
    var l = S.l
    var m = b * l
    var di = dims.d_inner
    var nh = dims.nheads
    var cd = dims.conv_dim()
    ref st = S.stages[k]
    ref T = S.tail[k]
    ref w = S.w[k]
    ref ssd = S.ssd[k]
    ref sr = S.scale[k]
    ref disc = S.disc[k]
    ref cv = S.conv[k]
    _copy_unless_same(ex, _fp(T.d_skip), up, m * di)
    mamba2_backward_d_skip_into(ctx, T, st.silu_out, w.d_skip, dims, m)
    mamba2_s18_direct_dpass_into(
        ctx, ssd, T.d_scan, st.xbc_work, st.pass_states, st.dacs,
        b, st.t_work, nh, di, cd, st.nc, m2_q_eff(),
    )
    mamba2_reverse_chunk_state_into(
        ctx, ssd, S.d_final[k], st.pass_states, st.dacs, b, nh, st.nc, m2_q_eff(),
    )
    mamba2_cstate_ddecay_into(
        ctx, ssd, st.xd_work, st.xbc_work, b, st.t_work, nh, di, cd, st.nc, m2_q_eff(),
    )
    mamba2_reduce_scale_product_into(ctx, sr, ssd, st.dacs, st.decay, b, st.nc, nh, m2_q_eff())
    mamba2_reverse_cumsum_and_da_into(
        ctx, disc, sr.d_dacs_total, st.dt_work, st.a_out, b, st.t_work, nh, st.nc, m2_q_eff(),
    )
    mamba2_ydiag_xd_and_partial_dt_into(
        ctx, disc, T.d_scan, st.cb_g, st.seg_l, st.xd_work,
        st.xbc_work, st.dt_work, st.a_out, st.dtraw_work, w.dt_bias,
        ssd.d_cstate, st.decay,
        b, st.t_work, nh, di, cd, st.nc, m2_q_eff(), S.dt_lo, S.dt_hi,
    )
    mamba2_postconv_merge_into(ctx, disc, ssd, T.d_x_from_d, b, st.t_work, nh)
    mamba2_conv_backward_prefill_into(
        ctx, cv, disc, st.conv_out, st.in_proj, w.conv_w, b, l, di, cd, dims.d_in_proj(), st.q0,
    )
    _copy_unless_same(ex, gradients[1], _fp(cv.d_w), cd * M2_D_CONV)
    _copy_unless_same(ex, gradients[2], _fp(cv.d_b), cd)
    _copy_unless_same(ex, gradients[3], _fp(disc.d_dt_bias), nh)
    _copy_unless_same(ex, gradients[4], _fp(disc.d_a_log), nh)
    _copy_unless_same(ex, gradients[5], _fp(T.d_d), nh)
    _m2_pack(ctx, gradients[0], _fp(T.d_z), _fp(cv.d_in_xbc), _fp(disc.d_dtraw), m, dims)


@fieldwise_init
struct Mamba2Rules(BackwardRules, Copyable, Movable):
    """Backward adapters for a Mamba-2 stack's custom nodes."""
    # The stack's address, untracked (a struct field may not carry
    # MutAnyOrigin); the driver keeps the stack alive through the backward.
    var s: MutPointer[Mamba2TapeStack, MutUntrackedOrigin]

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # Never: every destination is stored whole by a GEMM, the C7 pack
        # (every column of [M, d_in_proj]) or a bit copy, or is the hand
        # path's own output field; M2_RESID writes nothing.
        return False

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # The joins (header): x by its norm (C8), P by the SSD rule's C7
        # pack; the gate rule leaves its d_z in T.d_z.
        if tag == M2_NORM or tag == M2_RESID or tag == M2_SSD:
            return edge == 0
        if tag == M2_GATE:
            return edge == 1
        return False

    def supports(self, tag: Int) -> Bool:
        return (tag >= M2_NORM and tag <= M2_PROJ_OUT) or tag == M2_RESID

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        ref S = self.s[]
        var k = saved.i0
        if tag == M2_NORM:
            _m2_norm_bwd(ex, S, k, saved, upstream, gradients)
        elif tag == M2_PROJ_IN:
            _m2_proj_bwd(S, k, PROJ2_IN, saved, upstream, gradients, rows, cols)
        elif tag == M2_SSD:
            _m2_ssd_bwd(ex, S, k, upstream, gradients)
        elif tag == M2_GATE:
            _m2_gate_bwd(ex, S, k, upstream, gradients)
        elif tag == M2_PROJ_OUT:
            _m2_proj_bwd(S, k, PROJ2_OUT, saved, upstream, gradients, rows, cols)
        elif tag == M2_RESID:
            # no launch: O's gradient is this node's storage, x's is written
            # whole by its norm
            _copy_unless_same(ex, gradients[1], upstream, rows[1] * cols[1])
        else:
            raise Error("unknown mamba2 backward rule")


def mamba2_tape_backward_stack(
    weights: List[Mamba2Weights],
    input: List[Float32],
    grad_output: List[Float32],
    b: Int,
    l: Int,
    dt_lo: Float32,
    dt_hi: Float32,
    ctx: DeviceContext,
) raises -> List[Mamba2PrefillGradients]:
    """Zero-state VJP of a stack of Mamba-2 blocks, assembled by the tape.
    Entry k holds block k's parameter gradients and the gradient at its
    input; the final-state cotangent is zero, as in the one-block VJP."""
    if b <= 0 or l <= 0:
        raise Error("mamba2 backward: B and L must be positive")
    if len(weights) < 1:
        raise Error("mamba2 tape backward: the stack needs at least one block")
    var dm = weights[0].dims.d_model
    if len(input) != b * l * dm or len(grad_output) != len(input):
        raise Error("mamba2 backward: input and grad_output lengths must equal B*L*d_model")
    var m = b * l
    var nb = len(weights)
    var S = Mamba2TapeStack(ctx, weights, input, b, l, dt_lo, dt_hi)
    var dims = S.dims.copy()
    var di = dims.d_inner
    var dip = dims.d_in_proj()
    var cd = dims.conv_dim()
    var nh = dims.nheads
    var trace = IdentityTrace.disabled()
    for k in range(nb):  # driver-loop: over blocks
        mamba2_block_forward(
            S.ctx, S.stages[k], S.state[k], S.w[k], S.acts[k], b, l, dt_lo, dt_hi, trace,
            String("m2.backward.tail"),
        )
        comptime if not AUTOGRAD_MAMBA_HOST:
            _d2d(S.ctx, S.acts[k + 1], S.stages[k].residual_out, m * dm)
        # the hand path's SSD backward state, sized by this block's stages
        var nc = S.stages[k].nc
        var t_work = S.stages[k].t_work
        S.ssd.append(Mamba2SSDBackwardState(ctx, b, nc, nh))
        S.d_final.append(mamba_zeros(ctx, b * nh * M2_HEADDIM * M2_D_STATE))
        S.scale.append(Mamba2SSDScaleReduction(ctx, b, nc, nh, m2_q_eff()))
        S.disc.append(Mamba2SSDDiscretizeBackward(ctx, b, t_work, nh))
        S.conv.append(Mamba2ConvBackward(ctx, b, l, cd))

    var dres = mamba_upload(ctx, grad_output)
    mamba_refuse_grad_output(ctx, String("mamba2"), dres, grad_output)

    var ex = DeviceExec(ctx)
    var tape = Tape()
    var xs = List[Tensor]()
    var params = List[Tensor]()
    # The gradient at block k's residual output: the caller's dres for the
    # last block (the seed, used in place), block k + 1's T.d_block_x (its
    # input's gradient, written whole by its norm) otherwise. O_k's gradient
    # shares it; block k's norm adds it into x_k's (C8).
    var dres_at = List[FP]()
    for k in range(nb):  # driver-loop: over blocks
        dres_at.append(_fp(dres) if k == nb - 1 else _fp(S.tail[k + 1].d_block_x))
    var x_t = tape.variable_into(ex, _fp(S.acts[0]), _fp(S.tail[0].d_block_x), m, dm)
    for k in range(nb):  # driver-loop: over blocks
        xs.append(x_t)
        ref st = S.stages[k]
        ref w = S.w[k]
        ref T = S.tail[k]
        ref cv = S.conv[k]
        ref disc = S.disc[k]
        # gradients land in the hand path's output fields (no rule copies)
        var norm_w = tape.variable_into(ex, _fp(w.norm_w), _fp(T.d_block_w), 1, dm)
        var w_in = tape.variable(ex, _fp(w.w_in), dip, dm)
        var conv_w = tape.variable_into(ex, _fp(w.conv_w), _fp(cv.d_w), cd, M2_D_CONV)
        var conv_b = tape.variable_into(ex, _fp(w.conv_b), _fp(cv.d_b), 1, cd)
        var dt_bias = tape.variable_into(ex, _fp(w.dt_bias), _fp(disc.d_dt_bias), 1, nh)
        var a_log = tape.variable_into(ex, _fp(w.a_log), _fp(disc.d_a_log), 1, nh)
        var d_skip = tape.variable_into(ex, _fp(w.d_skip), _fp(T.d_d), 1, nh)
        var gnorm_w = tape.variable_into(ex, _fp(w.gnorm_w), _fp(T.d_gnorm_w), 1, di)
        var w_out = tape.variable(ex, _fp(w.w_out), dm, di)
        for t in [norm_w, w_in, conv_w, conv_b, dt_bias, a_log, d_skip, gnorm_w, w_out]:  # driver-loop: over parameters
            params.append(t)
        # P's consumers SSD then GATE: the gate rule leaves d_z, the SSD rule
        # packs P's whole gradient.
        var nrm = tape.custom_into(ex, [x_t, norm_w], _fp(st.norm_out), _fp(T.d_norm), m, dm, M2_NORM,
                                   _saved_norm(k, dm, dres_at[k]))
        var p = tape.custom(ex, [nrm, w_in], _fp(st.in_proj), m, dip, M2_PROJ_IN, _saved(k, dip))
        var sk = tape.custom_into(
            ex, [p, conv_w, conv_b, dt_bias, a_log, d_skip], _fp(st.skip_out), _fp(T.d_skip), m, di, M2_SSD,
            _saved(k, di),
        )
        var g = tape.custom_into(ex, [sk, p, gnorm_w], _fp(st.gnorm_out), _fp(T.d_gnorm), m, di, M2_GATE, _saved(k, di))
        var o = tape.custom_into(ex, [g, w_out], _fp(st.out_proj), dres_at[k], m, dm, M2_PROJ_OUT, _saved(k, dm))
        x_t = tape.custom_into_shared(ex, [x_t, o], _fp(S.acts[k + 1]), o, m, dm, M2_RESID, _saved(k, dm))
    var seed = tape.variable(ex, _fp(dres), m, dm, False)
    var rules = Mamba2Rules(MutPointer[Mamba2TapeStack, MutUntrackedOrigin](unsafe_from_address=Int(MutPointer(to=S))))
    tape.backward_with_rules(ex, x_t, seed, rules)

    var result = List[Mamba2PrefillGradients]()
    # The gradients recorded as the hand path's output fields live in the
    # stack's own device buffers, which DeviceExec did not allocate (its
    # download refuses them): they are read with the hand path's download,
    # after the tape's queue drains. w_in and w_out keep tape storage.
    comptime if AUTOGRAD_MAMBA_HOST:
        # idea K(2): the copies are queued behind the backward on the same
        # in-order queue (so they read the finished words), one drain
        for k in range(nb):  # driver-loop: over blocks
            var g = Mamba2PrefillGradients()
            var base = 9 * k
            ref T = S.tail[k]
            ref cv = S.conv[k]
            ref disc = S.disc[k]
            g.block_norm_weight = _download_queued(ctx, T.d_block_w, dm)
            g.in_proj_weight = _grad_queued(tape, ex, params[base + 1])
            g.conv1d_weight = _download_queued(ctx, cv.d_w, cd * M2_D_CONV)
            g.conv1d_bias = _download_queued(ctx, cv.d_b, cd)
            g.dt_bias = _download_queued(ctx, disc.d_dt_bias, nh)
            g.A_log = _download_queued(ctx, disc.d_a_log, nh)
            g.D = _download_queued(ctx, T.d_d, nh)
            g.norm_weight = _download_queued(ctx, T.d_gnorm_w, di)
            g.out_proj_weight = _grad_queued(tape, ex, params[base + 8])
            g.x = _download_queued(ctx, T.d_block_x, m * dm)
            result.append(g^)
        ex.sync()
        tape.release(ex)
        ctx.synchronize()
        _ = rules^
        _ = dres^
        _ = S^
        return result^
    else:
        ex.sync()
        for k in range(nb):  # driver-loop: over blocks
            var g = Mamba2PrefillGradients()
            var base = 9 * k
            ref T = S.tail[k]
            ref cv = S.conv[k]
            ref disc = S.disc[k]
            g.block_norm_weight = mamba_download(ctx, T.d_block_w, dm)
            g.in_proj_weight = tape.gradients(ex, params[base + 1])
            g.conv1d_weight = mamba_download(ctx, cv.d_w, cd * M2_D_CONV)
            g.conv1d_bias = mamba_download(ctx, cv.d_b, cd)
            g.dt_bias = mamba_download(ctx, disc.d_dt_bias, nh)
            g.A_log = mamba_download(ctx, disc.d_a_log, nh)
            g.D = mamba_download(ctx, T.d_d, nh)
            g.norm_weight = mamba_download(ctx, T.d_gnorm_w, di)
            g.out_proj_weight = tape.gradients(ex, params[base + 8])
            g.x = mamba_download(ctx, T.d_block_x, m * dm)
            result.append(g^)
        tape.release(ex)
        ctx.synchronize()
        _ = rules^
        _ = dres^
        _ = S^
        return result^


def mamba2_tape_backward(
    weights: Mamba2Weights,
    input: List[Float32],
    grad_output: List[Float32],
    b: Int,
    l: Int,
    dt_lo: Float32,
    dt_hi: Float32,
    ctx: DeviceContext,
) raises -> Mamba2PrefillGradients:
    """mamba2_prefill_backward's contract, assembled by the tape (one block)."""
    var stack = List[Mamba2Weights]()
    stack.append(weights.copy())
    var grads = mamba2_tape_backward_stack(stack, input, grad_output, b, l, dt_lo, dt_hi, ctx)
    return grads.pop(0)

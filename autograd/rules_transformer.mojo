# SPDX-License-Identifier: Apache-2.0
# Transformer decoder-layer backward on the tape (-D MOJOLEARN_AUTOGRAD_TRANSFORMER,
# default off). docs/plans/AUTOGRAD_API.md section (c) item 2.
#
# ---------------------------------------------------------------- launch inventory
# The hand path, transformer/checks/transformer_backward.mojo::
# llama_decoder_layer_backward_device, for ONE LlamaDecoderLayer.forward call
# (IDENTICAL, trace off, materialize off, fused attention, default options),
# in launch order. "fwd." is the saved LlamaDeviceStages, "w." the weights,
# everything else is a LlamaBackwardStages field.
#
#   refuse   device_first_nonfinite(d_out)                     d_out = d(residual2)
#   S0-1     bwd_copy_kernel d_out -> in_d_residual2           (skipped by
#            IDN_BWD_ENTRY_ALIAS: readers take d_out itself; d_down_proj_out
#            is a view of in_d_residual2)
#   S2-3     down:  _route_a d_mlp_gated <- din . w_down
#                   _route_b dw_down     <- din, fwd.gated
#   S4-6     SwiGLU: bwd_mul2_silu_backward_kernel (fused; reads d_mlp_gated,
#            fwd.up_proj, fwd.silu_out, fwd.gate_proj) -> d_silu_out,
#            d_up_proj_out, d_gate_proj_out; split: bwd_mul2_kernel then
#            bwd_silu_backward_kernel
#   S7-9     gate/up: _route_b dw_gate, dw_up <- d_*_proj_out, fwd.norm2_out
#                     _route_a tmp0 (gate), tmp1 (up) <- d_*_proj_out . w_*
#                     bwd_add2_kernel d_norm2_out = tmp0 + tmp1
#   S10-12   norm2: bwd_rms_norm_routed[2] (reads d_norm2_out, fwd.residual1,
#            w.norm2_w, fwd.norm2_sumsq): bwd_norm_dh(_dot) -> dh, norm2_dot,
#            rstd, dvcoef; bwd_norm_dx_kernel -> norm2_dx, dprod and (fused
#            residual, IDENTICAL on Apple/NVIDIA/AMD) d_residual1 =
#            norm2_dx + din; GEMM ones . dprod -> dw_norm2
#   S13      (split only) bwd_add2_kernel d_residual1 = norm2_dx + din
#   S14-16   o: d_o_proj_out is a view of d_residual1;
#               _route_a d_attn_ctx <- d_residual1 . w_o
#               _route_b dw_o       <- d_residual1, fwd.ctxv
#   S17-24   attention: bwd_attention_dispatch (fused_backward_launch, its
#            estash/recompute arms, or the eager gather/GEMM/scatter, softmax,
#            mask, scale and dq/dk/dv kernels); reads d_attn_ctx, fwd.q_rope,
#            fwd.k_cache, fwd.v_cache, fwd.amax, fwd.denom (eager: fwd.weights)
#            -> attn_zdot, d_q_rope, d_k_cache, d_v_cache
#   S25-26   bwd_kv_slice2_kernel -> d_k_rope, d_v_proj_out
#   S27-28   bwd_rope_kernel x2 (cos/sin tables) -> d_q_proj_out, d_k_proj_out
#   S29-32   q/k/v: _route_b dw_q, dw_k, dw_v <- d_*_proj_out, fwd.norm1_out
#                   _route_a tmp0 (q), tmp1 (k), tmp2 (v) <- d_*_proj_out . w_*
#                   bwd_add3_kernel d_norm1_out = (q + k) + v
#   S33-35   norm1: bwd_rms_norm_routed[1] (reads d_norm1_out, x, w.norm1_w,
#            fwd.norm1_sumsq) -> norm1_dot, norm1_dx, dw_norm1
#   S36      bwd_add2_kernel d_x = norm1_dx + d_residual1
#
# ---------------------------------------------------------------- the tape graph
# What moves to the tape: the order of the stages and the residual fan-outs.
# What stays: every launch above, with its arguments and its destination, as
# seven custom rules that call the hand path's launchers:
#
#   TF_RMS_NORM_INPUT / TF_RMS_NORM_POST  [x, weight] -> norm out
#                     bwd_rms_norm_routed[1|2] with the hand path's residual
#                     join: norm2 fused (BWD_FUSE_NORM2_RESIDUAL) or S13's
#                     add2, norm1 S36's add2 (or _AFN_NORM1_RESID's fusion)
#   TF_LINEAR         [X, W] -> X W^T (OP_NT): _route_b dW, _route_a dX; a
#                     fan-in member writes dX to the hand path's tmp0/1/2 and
#                     the last one runs S9's add2 or S32's add3
#   TF_ROPE           [proj] -> rope(proj): bwd_rope_kernel
#   TF_ATTENTION      [q_rope, k_rope, v_proj] -> ctxv: bwd_attention_dispatch
#                     then bwd_kv_slice2_kernel
#   TF_SWIGLU         [gate, up] -> silu(gate) * up: the fused (or split)
#                     S4-6 kernels
#   TF_RESIDUAL       [x, branch] -> x + branch (recorded, not relaunched):
#                     no launch. The branch's gradient storage IS the sum's
#                     (custom_into on both), as in the hand path, where
#                     d_down_proj_out aliases d(residual2) and d_o_proj_out
#                     aliases d_residual1; x's gradient is written whole by
#                     the norm that also reads x (writes_total).
#
# Fan-ins (BackwardRules.writes_total: the rule owns the input's whole
# gradient, the tape adds nothing): d(norm2.out) = tmp0 (gate) + tmp1 (up),
# S9; d(norm1.out) = (tmp0 (q) + tmp1 (k)) + tmp2 (v), S32, the pinned
# order; d_residual1 = norm2_dx + d(residual2), S10-13; d_x = norm1_dx +
# d_residual1, S36. Every one is the hand path's own kernel on the hand
# path's buffers, so every gradient carries the hand path's bits (the old
# tape spelling flushed the incoming gradient through the add's VJP before
# the down and o GEMMs; that copy is gone, so subnormals now match too).
# The seed is used in place: d(residual2)'s storage is d_out under
# IDN_BWD_ENTRY_ALIAS (the hand path's entry alias), else in_d_residual2,
# where the tape's one OP_COPY is the hand path's S0-1 bit copy.
# zeroed_destination is False for every rule: each kernel above overwrites
# every element of the destination it is given (the hand path reuses its
# stage buffers across calls without zeroing them).
#
# ---------------------------------------------------------------- launch count
# Per layer, from the code, A = the attention dispatch's launches, R = the
# refusal scan (both paths), and norm = bwd_rms_norm_routed's kernels + GEMM:
#   hand path:    down 2, SwiGLU 1, gate/up 4 + add2 1, norm2 3, o 2,
#                 kv slice 1, rope 2, qkv 6 + add3 1, norm1 3, add2 1
#                 = 27 + A (+ R; + 1 bit copy without the entry alias)
#   tape, before: 59 + A (+ R): 1 seed copy, 4 ADD VJP copies, 25 zero
#                 fills of custom destinations, 5 accumulates in place of
#                 the add2 / add3 / fused residual joins, and the 27 + A
#                 hand launches minus those joins (24 + A)
#   tape, after:  27 + A (+ R; + 1 seed copy without the entry alias): the
#                 hand path's launches, one for one, nothing else
#
# Scope: one decoder layer, the unit both bindings differentiate. The
# embedding gather and LM head belong to the language-model trainers
# (training/byte_lm.mojo, migration item 4) and get their rules there.
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from autograd.tape import Tape, Tensor, BackwardRules
from sequence.exec_trait import Exec
from sequence.exec_device import DeviceExec
from sequence.ops import FP, Args
from core.identity_trace import IdentityTrace
from core.device_scan import device_first_nonfinite
from core.step_phase import step_count_launch
from gemm.contract import OP_NT
from transformer.impl.llama.modeling_llama import (
    LlamaDeviceStages,
    LlamaDeviceWeights,
    LlamaDims,
    llama_attention_scale,
    llama_key_lo,
    llama_key_span,
)
from transformer.checks.transformer_backward import (
    BWD_FUSE_GATED_SILU,
    BWD_FUSE_NORM2_RESIDUAL,
    BWD_TPB,
    IDN_BWD_ENTRY_ALIAS,
    _AFN_NORM1_RESID,
    LlamaBackwardStages,
    _route_a,
    _route_b,
    bwd_add2_kernel,
    bwd_add3_kernel,
    bwd_attention_dispatch,
    bwd_kv_slice2_kernel,
    bwd_mul2_kernel,
    bwd_mul2_silu_backward_kernel,
    bwd_rms_norm_routed,
    bwd_rope_kernel,
    bwd_silu_backward_kernel,
)

comptime AUTOGRAD_TRANSFORMER = is_defined["MOJOLEARN_AUTOGRAD_TRANSFORMER"]()

# Tags in a range of their own: LayerNorm 1 (autograd/layers.mojo),
# recurrent 16-18 (autograd/rules_recurrent.mojo), transformer 32-47.
comptime TF_RMS_NORM_INPUT = 32
comptime TF_RMS_NORM_POST = 33
comptime TF_LINEAR = 34
comptime TF_ROPE = 35
comptime TF_ATTENTION = 36
comptime TF_SWIGLU = 37
comptime TF_RESIDUAL = 38

# TF_LINEAR's saved.i0: where its dX goes. Plain: the tape's destination.
# The fan-in members write the hand path's tmp buffers; the last member of
# each fan-in (in the tape's reverse order) then runs the hand path's join
# into the destination: S9 (up tmp1, then gate tmp0 + add2) and S32 (q tmp0,
# k tmp1, then v tmp2 + add3, (q + k) + v).
comptime TF_DX_PLAIN = 0
comptime TF_DX_TMP0 = 1
comptime TF_DX_TMP1 = 2
comptime TF_DX_TMP0_ADD2 = 3
comptime TF_DX_TMP2_ADD3 = 4


@always_inline
def _blocks(n: Int) -> Int:
    # transformer_backward._grid: the hand path's launch geometry
    var g = (n + BWD_TPB - 1) // BWD_TPB
    if g < 1:
        return 1
    return g


@always_inline
def _view(ctx: DeviceContext, p: FP, n: Int) -> DeviceBuffer[DType.float32]:
    """A non-owning view of n floats at p (tape or stage storage)."""
    return DeviceBuffer[DType.float32](ctx, p.unsafe_origin_cast[MutAnyOrigin](), n if n > 0 else 1, owning=False)


@always_inline
def _fp(mut buf: DeviceBuffer[DType.float32]) -> FP:
    return FP(unsafe_from_address=Int(buf.unsafe_ptr()))


struct TransformerRules(BackwardRules):
    """Backward adapters for one Llama decoder layer's fused native ops.

    Holds the layer's backward stages (scratch, GEMM workspace, attention
    outputs) and saved forward stages; both must outlive the backward."""

    var ctx: DeviceContext
    var bst: MutPointer[LlamaBackwardStages, MutUntrackedOrigin]
    var fwd: MutPointer[LlamaDeviceStages, MutUntrackedOrigin]
    var cos_tab: FP
    var sin_tab: FP
    var eps: Float32
    var b: Int
    var l: Int
    var pos0: Int

    def __init__(
        out self, ctx: DeviceContext,
        bst: MutPointer[LlamaBackwardStages, MutUntrackedOrigin],
        fwd: MutPointer[LlamaDeviceStages, MutUntrackedOrigin],
        cos_tab: FP, sin_tab: FP, eps: Float32, b: Int, l: Int, pos0: Int,
    ):
        self.ctx = ctx.copy()
        self.bst = bst
        self.fwd = fwd
        self.cos_tab = cos_tab
        self.sin_tab = sin_tab
        self.eps = eps
        self.b = b
        self.l = l
        self.pos0 = pos0

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # Never: every kernel these rules launch overwrites every element of
        # the destination it is handed (the hand path reuses the same stage
        # buffers across calls with no fill), and TF_RESIDUAL writes nothing.
        return False

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # The joins (header): the norms write their input's whole gradient
        # (norm term + residual branch); TF_RESIDUAL leaves x's to that norm;
        # a fan-in member owns its input's gradient (the last one writes it).
        if tag == TF_RMS_NORM_INPUT or tag == TF_RMS_NORM_POST or tag == TF_RESIDUAL:
            return edge == 0
        if tag == TF_LINEAR:
            return edge == 0 and saved.i0 != TF_DX_PLAIN
        return False

    def supports(self, tag: Int) -> Bool:
        return tag >= TF_RMS_NORM_INPUT and tag <= TF_RESIDUAL

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        # Every launch goes on self.ctx, the context `ex` wraps, so it runs
        # in order with the tape's own launches.
        if tag == TF_RMS_NORM_INPUT:
            self._rms_norm[1](saved, upstream, inputs, gradients, rows, cols)
        elif tag == TF_RMS_NORM_POST:
            self._rms_norm[2](saved, upstream, inputs, gradients, rows, cols)
        elif tag == TF_LINEAR:
            self._linear(saved, upstream, inputs, gradients, rows, cols)
        elif tag == TF_ROPE:
            self._rope(saved, upstream, gradients, rows, cols)
        elif tag == TF_ATTENTION:
            self._attention(upstream, gradients, rows, cols)
        elif tag == TF_SWIGLU:
            self._swiglu(saved, upstream, inputs, gradients, rows, cols)
        elif tag == TF_RESIDUAL:
            self._residual(upstream, gradients, rows, cols)
        else:
            raise Error("unknown transformer backward rule")

    def _add2(self, dst: FP, x: FP, y: FP, n: Int) raises:
        # bwd_add2_kernel, the hand path's two-term join, its geometry
        var d = _view(self.ctx, dst, n)
        var a = _view(self.ctx, x, n)
        var b = _view(self.ctx, y, n)
        step_count_launch()
        self.ctx.enqueue_function[bwd_add2_kernel](
            d.unsafe_ptr(), a.unsafe_ptr(), b.unsafe_ptr(), Int32(n),
            grid_dim=(_blocks(n), 1, 1),
            block_dim=(BWD_TPB, 1, 1),
        )

    def _rms_norm[which: Int](
        self, saved: Args, upstream: FP, inputs: List[FP], gradients: List[FP],
        rows: List[Int], cols: List[Int],
    ) raises:
        # S10-13 / S33-36 with the hand path's residual join; saved.p0 the
        # forward's row sums of squares, saved.p1 the recorded dot stage,
        # saved.p2 the residual branch's gradient (d(residual2) for norm2,
        # d_residual1 for norm1). gradients[0] receives x's whole gradient
        # (writes_total): norm dx + branch, the hand path's d_residual1 / d_x.
        var m = rows[0]
        var dm = cols[0]
        ref bst = self.bst[]
        var dot = _view(self.ctx, saved.p1, m)
        var dw = _view(self.ctx, gradients[1], dm)
        var dy = _view(self.ctx, upstream, m * dm)
        var x = _view(self.ctx, inputs[0], m * dm)
        var weight = _view(self.ctx, inputs[1], dm)
        var sumsq = _view(self.ctx, saved.p0, m)
        var total = gradients[0].unsafe_origin_cast[MutAnyOrigin]()
        var branch = saved.p2.unsafe_origin_cast[MutAnyOrigin]()
        comptime if which == 2:
            # S10-12, the residual fused (BWD_FUSE_NORM2_RESIDUAL) or S13
            bwd_rms_norm_routed[2](
                self.ctx, bst.gemm_workspace, dot, bst.norm2_dx, dw, bst.dh, bst.dprod, bst.rstd,
                bst.dvcoef, bst.ones, dy, x, weight, sumsq, total, branch,
                BWD_FUSE_NORM2_RESIDUAL, m, dm, self.eps,
            )
            comptime if not BWD_FUSE_NORM2_RESIDUAL:
                self._add2(gradients[0], _fp(bst.norm2_dx), saved.p2, m * dm)
        else:
            # S33-35, then S36 (or _AFN_NORM1_RESID's fused residual)
            comptime if _AFN_NORM1_RESID:
                bwd_rms_norm_routed[1](
                    self.ctx, bst.gemm_workspace, dot, bst.norm1_dx, dw, bst.dh, bst.dprod, bst.rstd,
                    bst.dvcoef, bst.ones, dy, x, weight, sumsq, total, branch, True, m, dm, self.eps,
                )
            else:
                var own = _fp(bst.norm1_dx).unsafe_origin_cast[MutAnyOrigin]()
                bwd_rms_norm_routed[1](
                    self.ctx, bst.gemm_workspace, dot, bst.norm1_dx, dw, bst.dh, bst.dprod, bst.rstd,
                    bst.dvcoef, bst.ones, dy, x, weight, sumsq, own, own, False, m, dm, self.eps,
                )
                self._add2(gradients[0], _fp(bst.norm1_dx), saved.p2, m * dm)

    def _linear(
        self, saved: Args, upstream: FP, inputs: List[FP], gradients: List[FP],
        rows: List[Int], cols: List[Int],
    ) raises:
        # forward C [m, n] = X [m, k] . W [n, k]^T (OP_NT), as every
        # projection of the block; saved.i0 where dX goes (TF_DX_*)
        var m = rows[0]
        var k = cols[0]
        var n = rows[1]
        if cols[1] != k:
            raise Error("transformer linear: weight must be (out, in)")
        ref bst = self.bst[]
        var dc = _view(self.ctx, upstream, m * n)
        var x = _view(self.ctx, inputs[0], m * k)
        var w = _view(self.ctx, inputs[1], n * k)
        var dw = _view(self.ctx, gradients[1], n * k)
        _route_b(self.ctx, bst.gemm_workspace, dw, dc, x, OP_NT, m, n, k)
        var mode = saved.i0
        if mode == TF_DX_PLAIN:
            var dx = _view(self.ctx, gradients[0], m * k)
            _route_a(self.ctx, bst.gemm_workspace, dx, dc, w, OP_NT, m, n, k)
        elif mode == TF_DX_TMP0:
            _route_a(self.ctx, bst.gemm_workspace, bst.tmp0, dc, w, OP_NT, m, n, k)
        elif mode == TF_DX_TMP1:
            _route_a(self.ctx, bst.gemm_workspace, bst.tmp1, dc, w, OP_NT, m, n, k)
        elif mode == TF_DX_TMP0_ADD2:
            # S9: d(norm2.out) = tmp0 (gate) + tmp1 (up)
            _route_a(self.ctx, bst.gemm_workspace, bst.tmp0, dc, w, OP_NT, m, n, k)
            self._add2(gradients[0], _fp(bst.tmp0), _fp(bst.tmp1), m * k)
        elif mode == TF_DX_TMP2_ADD3:
            # S32: d(norm1.out) = (tmp0 (q) + tmp1 (k)) + tmp2 (v)
            _route_a(self.ctx, bst.gemm_workspace, bst.tmp2, dc, w, OP_NT, m, n, k)
            var d = _view(self.ctx, gradients[0], m * k)
            step_count_launch()
            self.ctx.enqueue_function[bwd_add3_kernel](
                d.unsafe_ptr(), bst.tmp0.unsafe_ptr(), bst.tmp1.unsafe_ptr(), bst.tmp2.unsafe_ptr(),
                Int32(m * k),
                grid_dim=(_blocks(m * k), 1, 1),
                block_dim=(BWD_TPB, 1, 1),
            )
        else:
            raise Error("transformer linear: unknown dX destination")

    def _residual(self, upstream: FP, gradients: List[FP], rows: List[Int], cols: List[Int]) raises:
        # No launch: the branch's gradient storage is this node's (the
        # driver records both with the same custom_into storage) and x's
        # gradient is written whole by its norm (writes_total). A driver
        # that gave the branch storage of its own gets the hand path's
        # bit copy.
        if Int(gradients[1]) != Int(upstream):
            var n = rows[1] * cols[1]
            var src = _view(self.ctx, upstream, n)
            var dst = _view(self.ctx, gradients[1], n)
            self.ctx.enqueue_copy(dst_buf=dst, src_buf=src)

    def _rope(self, saved: Args, upstream: FP, gradients: List[FP], rows: List[Int], cols: List[Int]) raises:
        # S27 / S28; saved.i0 heads, saved.i1 head_dim
        var m = rows[0]
        var width = cols[0]
        var dst = _view(self.ctx, gradients[0], m * width)
        var src = _view(self.ctx, upstream, m * width)
        var cs = _view(self.ctx, self.cos_tab, 1)
        var sn = _view(self.ctx, self.sin_tab, 1)
        step_count_launch()
        self.ctx.enqueue_function[bwd_rope_kernel](
            dst.unsafe_ptr(),
            src.unsafe_ptr(),
            cs.unsafe_ptr(),
            sn.unsafe_ptr(),
            Int32(m),
            Int32(self.l),
            Int32(saved.i0),
            Int32(saved.i1),
            Int32(self.pos0),
            grid_dim=(_blocks(m * width), 1, 1),
            block_dim=(BWD_TPB, 1, 1),
        )

    def _attention(self, upstream: FP, gradients: List[FP], rows: List[Int], cols: List[Int]) raises:
        # S17-26: the hand path's dispatch reads bst.d_attn_ctx and writes
        # bst.d_q_rope / d_k_cache / d_v_cache; the KV slice then writes the
        # k_rope and v_proj gradients. In this graph the tape's buffers ARE
        # those fields (custom_into), so neither copy below runs.
        ref bst = self.bst[]
        ref fwd = self.fwd[]
        var dims = fwd.dims.copy()
        var m = self.b * self.l
        var qw = dims.q_width()
        var kw = dims.kv_width()
        var hd = dims.head_dim
        var nkv = dims.n_kv
        if rows[0] != m or cols[0] != qw or rows[1] != m or cols[1] != kw or rows[2] != m or cols[2] != kw:
            raise Error("transformer attention: inputs must be q_rope [M, qw], k_rope and v [M, kw]")
        var window = fwd.window
        var key_lo = llama_key_lo(self.pos0, window)
        var s = llama_key_span(self.pos0, self.l, window)
        var scale = llama_attention_scale(hd)
        if Int(upstream) != Int(bst.d_attn_ctx.unsafe_ptr()):
            var src = _view(self.ctx, upstream, m * qw)
            var dst = _view(self.ctx, _fp(bst.d_attn_ctx), m * qw)
            self.ctx.enqueue_copy(dst_buf=dst, src_buf=src)
        var off = IdentityTrace.disabled()
        bwd_attention_dispatch(
            self.ctx, bst, fwd, self.b, self.l, s, self.pos0, key_lo, window, dims, scale,
            False, off, String("autograd"),
        )
        if Int(gradients[0]) != Int(bst.d_q_rope.unsafe_ptr()):
            var src_q = _view(self.ctx, _fp(bst.d_q_rope), m * qw)
            var dst_q = _view(self.ctx, gradients[0], m * qw)
            self.ctx.enqueue_copy(dst_buf=dst_q, src_buf=src_q)
        var dk = _view(self.ctx, gradients[1], m * kw)
        var dv = _view(self.ctx, gradients[2], m * kw)
        step_count_launch()
        self.ctx.enqueue_function[bwd_kv_slice2_kernel](
            dk.unsafe_ptr(),
            dv.unsafe_ptr(),
            bst.d_k_cache.unsafe_ptr(),
            bst.d_v_cache.unsafe_ptr(),
            Int32(self.b),
            Int32(self.l),
            Int32(nkv),
            Int32(hd),
            Int32(s),
            Int32(self.pos0 - key_lo),
            grid_dim=(_blocks(m * kw), 1, 1),
            block_dim=(BWD_TPB, 1, 1),
        )

    def _swiglu(
        self, saved: Args, upstream: FP, inputs: List[FP], gradients: List[FP],
        rows: List[Int], cols: List[Int],
    ) raises:
        # S4-6; inputs gate, up; saved.p0 the forward's silu(gate)
        var n = rows[0] * cols[0]
        ref bst = self.bst[]
        var d_gated = _view(self.ctx, upstream, n)
        var gate = _view(self.ctx, inputs[0], n)
        var up = _view(self.ctx, inputs[1], n)
        var silu = _view(self.ctx, saved.p0, n)
        var dg = _view(self.ctx, gradients[0], n)
        var dup = _view(self.ctx, gradients[1], n)
        step_count_launch()
        comptime if BWD_FUSE_GATED_SILU:
            self.ctx.enqueue_function[bwd_mul2_silu_backward_kernel](
                bst.d_silu_out.unsafe_ptr(), dup.unsafe_ptr(), dg.unsafe_ptr(),
                d_gated.unsafe_ptr(), up.unsafe_ptr(), silu.unsafe_ptr(),
                gate.unsafe_ptr(), Int32(n),
                grid_dim=(_blocks(n), 1, 1),
                block_dim=(BWD_TPB, 1, 1),
            )
        else:
            self.ctx.enqueue_function[bwd_mul2_kernel](
                bst.d_silu_out.unsafe_ptr(),
                dup.unsafe_ptr(),
                d_gated.unsafe_ptr(),
                up.unsafe_ptr(),
                silu.unsafe_ptr(),
                Int32(n),
                grid_dim=(_blocks(n), 1, 1),
                block_dim=(BWD_TPB, 1, 1),
            )
            step_count_launch()
            self.ctx.enqueue_function[bwd_silu_backward_kernel](
                dg.unsafe_ptr(),
                bst.d_silu_out.unsafe_ptr(),
                gate.unsafe_ptr(),
                silu.unsafe_ptr(),
                Int32(n),
                grid_dim=(_blocks(n), 1, 1),
                block_dim=(BWD_TPB, 1, 1),
            )


def _dx_mode(mode: Int) -> Args:
    var a = Args()
    a.i0 = mode
    return a^


def llama_decoder_layer_backward_autograd(
    ctx: DeviceContext,
    mut bst: LlamaBackwardStages,
    mut fwd: LlamaDeviceStages,
    mut w: LlamaDeviceWeights,
    mut cos_tab: DeviceBuffer[DType.float32],
    mut sin_tab: DeviceBuffer[DType.float32],
    mut x_dev: DeviceBuffer[DType.float32],
    mut d_out: DeviceBuffer[DType.float32],
    b: Int,
    l: Int,
    pos0: Int,
    mut trace: IdentityTrace,
    prefix: String,
    materialize: Bool = False,
) raises:
    """`llama_decoder_layer_backward_device` with the stage order and the
    fan-in sums owned by the autograd tape (same arguments, same outputs in
    the same `bst` fields). Records no identity card: a traced or
    materialized call is refused (use the hand path)."""
    if trace.enabled or materialize:
        raise Error("llama backward (autograd): traced or materialized calls take the hand path")
    if not w.opts.is_default():
        raise Error("llama backward (autograd): only the default Llama block is recorded")
    var dims = fwd.dims.copy()
    dims.validate()
    var dm = dims.d_model
    var nh = dims.n_heads
    var nkv = dims.n_kv
    var hd = dims.head_dim
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var it = dims.intermediate
    var m = b * l
    # The hand path's refusals, before any launch.
    if l <= 0 or b <= 0 or pos0 < 0:
        raise Error("llama backward: B and L must be positive and pos0 non-negative")
    if b != bst.b or l != bst.l:
        raise Error("llama backward: the stage buffers do not match (B, L)")
    var s = llama_key_span(pos0, l, fwd.window)
    if s > bst.s_max:
        raise Error(String("llama backward: pos0 + L = ") + String(s) + " exceeds the allocated s_max " + String(bst.s_max))
    if len(d_out) != m * dm:
        raise Error(String("llama backward: d_out has ") + String(len(d_out)) + " elements and [B, L, d_model] wants " + String(m * dm))
    if len(x_dev) < m * dm:
        raise Error(String("llama backward: x_dev holds ") + String(len(x_dev)) + " floats and the block input [B, L, d_model] wants " + String(m * dm))
    var bad = device_first_nonfinite(ctx, d_out, len(d_out))
    if bad >= 0:
        raise Error(String("llama backward: NaN or infinity in d_residual2 at flat index ") + String(bad) + " REFUSED")

    var ex = DeviceExec(ctx)
    var tape = Tape()
    # ---- leaves: the block input and the nine weights, gradients into bst
    var x = tape.variable_into(ex, _fp(x_dev), _fp(bst.d_x), m, dm)
    var g_norm1 = tape.variable_into(ex, _fp(w.norm1_w), _fp(bst.dw_norm1), 1, dm)
    var g_q = tape.variable_into(ex, _fp(w.w_q), _fp(bst.dw_q), qw, dm)
    var g_k = tape.variable_into(ex, _fp(w.w_k), _fp(bst.dw_k), kw, dm)
    var g_v = tape.variable_into(ex, _fp(w.w_v), _fp(bst.dw_v), kw, dm)
    var g_o = tape.variable_into(ex, _fp(w.w_o), _fp(bst.dw_o), dm, qw)
    var g_norm2 = tape.variable_into(ex, _fp(w.norm2_w), _fp(bst.dw_norm2), 1, dm)
    var g_gate = tape.variable_into(ex, _fp(w.w_gate), _fp(bst.dw_gate), it, dm)
    var g_up = tape.variable_into(ex, _fp(w.w_up), _fp(bst.dw_up), it, dm)
    var g_down = tape.variable_into(ex, _fp(w.w_down), _fp(bst.dw_down), dm, it)

    # ---- the forward as it ran, recorded (nothing is relaunched)
    # d(residual2): the seed's own storage under the hand path's entry alias
    # (used in place), else in_d_residual2 (the tape's one bit copy, S0-1).
    # down's gradient shares it (d_down_proj_out is a view of d(residual2)),
    # o's shares d_residual1 (d_o_proj_out is a view of d_residual1).
    var din = _fp(d_out) if IDN_BWD_ENTRY_ALIAS else _fp(bst.in_d_residual2)
    var s_norm1 = Args()
    s_norm1.p0 = _fp(fwd.norm1_sumsq)
    s_norm1.p1 = _fp(bst.norm1_dot)
    s_norm1.p2 = _fp(bst.d_residual1)
    var norm1 = tape.custom_into(ex, [x, g_norm1], _fp(fwd.norm1_out), _fp(bst.d_norm1_out), m, dm, TF_RMS_NORM_INPUT, s_norm1)
    # v, k, q in the tape's reverse order q, k, v: dX into tmp0, tmp1, then
    # tmp2 and S32's add3 (q + k) + v into d(norm1.out)
    var v = tape.custom_into(ex, [norm1, g_v], _fp(fwd.v_proj), _fp(bst.d_v_proj_out), m, kw, TF_LINEAR, _dx_mode(TF_DX_TMP2_ADD3))
    var k = tape.custom_into(ex, [norm1, g_k], _fp(fwd.k_proj), _fp(bst.d_k_proj_out), m, kw, TF_LINEAR, _dx_mode(TF_DX_TMP1))
    var q = tape.custom_into(ex, [norm1, g_q], _fp(fwd.q_proj), _fp(bst.d_q_proj_out), m, qw, TF_LINEAR, _dx_mode(TF_DX_TMP0))
    var s_rope_k = Args()
    s_rope_k.i0 = nkv
    s_rope_k.i1 = hd
    var k_rope = tape.custom_into(ex, [k], _fp(fwd.k_rope), _fp(bst.d_k_rope), m, kw, TF_ROPE, s_rope_k)
    var s_rope_q = Args()
    s_rope_q.i0 = nh
    s_rope_q.i1 = hd
    var q_rope = tape.custom_into(ex, [q], _fp(fwd.q_rope), _fp(bst.d_q_rope), m, qw, TF_ROPE, s_rope_q)
    var attn = tape.custom_into(ex, [q_rope, k_rope, v], _fp(fwd.ctxv), _fp(bst.d_attn_ctx), m, qw, TF_ATTENTION, Args())
    var o = tape.custom_into(ex, [attn, g_o], _fp(fwd.o_proj), _fp(bst.d_residual1), m, dm, TF_LINEAR, _dx_mode(TF_DX_PLAIN))
    var residual1 = tape.custom_into(ex, [x, o], _fp(fwd.residual1), _fp(bst.d_residual1), m, dm, TF_RESIDUAL, Args())
    var s_norm2 = Args()
    s_norm2.p0 = _fp(fwd.norm2_sumsq)
    s_norm2.p1 = _fp(bst.norm2_dot)
    s_norm2.p2 = din
    var norm2 = tape.custom_into(ex, [residual1, g_norm2], _fp(fwd.norm2_out), _fp(bst.d_norm2_out), m, dm, TF_RMS_NORM_POST, s_norm2)
    # gate, up in the tape's reverse order up, gate: dX into tmp1, then tmp0
    # and S9's add2 into d(norm2.out)
    var gate = tape.custom_into(ex, [norm2, g_gate], _fp(fwd.gate_proj), _fp(bst.d_gate_proj_out), m, it, TF_LINEAR, _dx_mode(TF_DX_TMP0_ADD2))
    var up = tape.custom_into(ex, [norm2, g_up], _fp(fwd.up_proj), _fp(bst.d_up_proj_out), m, it, TF_LINEAR, _dx_mode(TF_DX_TMP1))
    var s_swiglu = Args()
    s_swiglu.p0 = _fp(fwd.silu_out)
    var gated = tape.custom_into(ex, [gate, up], _fp(fwd.gated), _fp(bst.d_mlp_gated), m, it, TF_SWIGLU, s_swiglu)
    var down = tape.custom_into(ex, [gated, g_down], _fp(fwd.down_proj), din, m, dm, TF_LINEAR, _dx_mode(TF_DX_PLAIN))
    var residual2 = tape.custom_into(ex, [residual1, down], _fp(fwd.residual2), din, m, dm, TF_RESIDUAL, Args())

    # ---- one backward from d(residual2)
    var seed = tape.variable(ex, _fp(d_out), m, dm, False)
    var rules = TransformerRules(
        ctx,
        MutPointer[LlamaBackwardStages, MutUntrackedOrigin](unsafe_from_address=Int(MutPointer(to=bst))),
        MutPointer[LlamaDeviceStages, MutUntrackedOrigin](unsafe_from_address=Int(MutPointer(to=fwd))),
        _fp(cos_tab), _fp(sin_tab), w.eps, b, l, pos0,
    )
    tape.backward_with_rules(ex, residual2, seed, rules)
    tape.release(ex)
    # Drain the queue (and the tape storage freed above) before the caller's
    # downloads; ex's destructor would drain it again.
    ex.sync()

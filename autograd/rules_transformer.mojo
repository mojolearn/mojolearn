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
# What moves to the tape: the order of the stages, the residual fan-outs and
# every fan-in sum (S9, S13, S32, S36). What stays: every launch above, as six
# custom rules that call the hand path's launchers with their arguments:
#
#   TF_RMS_NORM_INPUT / TF_RMS_NORM_POST  [x, weight] -> norm out
#                     bwd_rms_norm_routed[1|2], residual NOT fused (the tape
#                     adds the residual branch)
#   TF_LINEAR         [X, W] -> X W^T (OP_NT): _route_b dW, _route_a dX
#   TF_ROPE           [proj] -> rope(proj): bwd_rope_kernel
#   TF_ATTENTION      [q_rope, k_rope, v_proj] -> ctxv: bwd_attention_dispatch
#                     then bwd_kv_slice2_kernel
#   TF_SWIGLU         [gate, up] -> silu(gate) * up: the fused (or split)
#                     S4-6 kernels
#
# The residual adds are plain tape adds of already computed values
# (Tape.binary_recorded: no forward launch, the tape's ADD VJP). Nodes are
# recorded v, k, q (not q, k, v) so the tape's reverse traversal adds the
# three input-projection terms of d(norm1.out) as (q + k) + v, the hand
# path's pinned order. Every two-term fan-in is commutative (IEEE addition
# is), and OP_AG_ACCUMULATE is ftz(ftz(a) + ftz(b)), bwd_add2_kernel's
# arithmetic, so every gradient is expected to carry the hand path's bits.
# The one place a value can differ: the down GEMM and the o branch read the
# incoming gradient through the ADD VJP, which flushes subnormals
# (sequence/ops.mojo ld/st); the identical GEMM flushes its operands too.
#
# Gradients land in the LlamaBackwardStages fields the hand path writes
# (custom_into / variable_into), so the caller's downloads are unchanged.
# Costs over the hand path (per layer): one seed copy (the entry alias goes),
# four ADD VJP copies, one zero fill per custom input (25, the tape's
# contract for custom destinations) and the five accumulates in place of
# the four add2/add3 launches and the fused norm2 residual.
#
# Scope: one decoder layer, the unit both bindings differentiate. The
# embedding gather and LM head belong to the language-model trainers
# (training/byte_lm.mojo, migration item 4) and get their rules there.
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from autograd.tape import Tape, Tensor, BackwardRules
from autograd.ops import ADD
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
    BWD_TPB,
    LlamaBackwardStages,
    _route_a,
    _route_b,
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

    def supports(self, tag: Int) -> Bool:
        return tag >= TF_RMS_NORM_INPUT and tag <= TF_SWIGLU

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
            self._linear(upstream, inputs, gradients, rows, cols)
        elif tag == TF_ROPE:
            self._rope(saved, upstream, gradients, rows, cols)
        elif tag == TF_ATTENTION:
            self._attention(upstream, gradients, rows, cols)
        elif tag == TF_SWIGLU:
            self._swiglu(saved, upstream, inputs, gradients, rows, cols)
        else:
            raise Error("unknown transformer backward rule")

    def _rms_norm[which: Int](
        self, saved: Args, upstream: FP, inputs: List[FP], gradients: List[FP],
        rows: List[Int], cols: List[Int],
    ) raises:
        # S10-12 / S33-35 with the residual unfused; saved.p0 the forward's
        # row sums of squares, saved.p1 the recorded dot stage.
        var m = rows[0]
        var dm = cols[0]
        ref bst = self.bst[]
        var dot = _view(self.ctx, saved.p1, m)
        var dx = _view(self.ctx, gradients[0], m * dm)
        var dw = _view(self.ctx, gradients[1], dm)
        var dy = _view(self.ctx, upstream, m * dm)
        var x = _view(self.ctx, inputs[0], m * dm)
        var weight = _view(self.ctx, inputs[1], dm)
        var sumsq = _view(self.ctx, saved.p0, m)
        var dxp = gradients[0].unsafe_origin_cast[MutAnyOrigin]()
        bwd_rms_norm_routed[which](
            self.ctx, bst.gemm_workspace, dot, dx, dw, bst.dh, bst.dprod, bst.rstd,
            bst.dvcoef, bst.ones, dy, x, weight, sumsq, dxp, dxp, False, m, dm, self.eps,
        )

    def _linear(
        self, upstream: FP, inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        # forward C [m, n] = X [m, k] . W [n, k]^T (OP_NT), as every
        # projection of the block
        var m = rows[0]
        var k = cols[0]
        var n = rows[1]
        if cols[1] != k:
            raise Error("transformer linear: weight must be (out, in)")
        ref bst = self.bst[]
        var dc = _view(self.ctx, upstream, m * n)
        var x = _view(self.ctx, inputs[0], m * k)
        var w = _view(self.ctx, inputs[1], n * k)
        var dx = _view(self.ctx, gradients[0], m * k)
        var dw = _view(self.ctx, gradients[1], n * k)
        _route_b(self.ctx, bst.gemm_workspace, dw, dc, x, OP_NT, m, n, k)
        _route_a(self.ctx, bst.gemm_workspace, dx, dc, w, OP_NT, m, n, k)

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
    var s_norm1 = Args()
    s_norm1.p0 = _fp(fwd.norm1_sumsq)
    s_norm1.p1 = _fp(bst.norm1_dot)
    var norm1 = tape.custom_into(ex, [x, g_norm1], _fp(fwd.norm1_out), _fp(bst.d_norm1_out), m, dm, TF_RMS_NORM_INPUT, s_norm1)
    # v, k, q: the reverse traversal then sums d(norm1.out) as (q + k) + v
    var v = tape.custom_into(ex, [norm1, g_v], _fp(fwd.v_proj), _fp(bst.d_v_proj_out), m, kw, TF_LINEAR, Args())
    var k = tape.custom_into(ex, [norm1, g_k], _fp(fwd.k_proj), _fp(bst.d_k_proj_out), m, kw, TF_LINEAR, Args())
    var q = tape.custom_into(ex, [norm1, g_q], _fp(fwd.q_proj), _fp(bst.d_q_proj_out), m, qw, TF_LINEAR, Args())
    var s_rope_k = Args()
    s_rope_k.i0 = nkv
    s_rope_k.i1 = hd
    var k_rope = tape.custom_into(ex, [k], _fp(fwd.k_rope), _fp(bst.d_k_rope), m, kw, TF_ROPE, s_rope_k)
    var s_rope_q = Args()
    s_rope_q.i0 = nh
    s_rope_q.i1 = hd
    var q_rope = tape.custom_into(ex, [q], _fp(fwd.q_rope), _fp(bst.d_q_rope), m, qw, TF_ROPE, s_rope_q)
    var attn = tape.custom_into(ex, [q_rope, k_rope, v], _fp(fwd.ctxv), _fp(bst.d_attn_ctx), m, qw, TF_ATTENTION, Args())
    var o = tape.custom(ex, [attn, g_o], _fp(fwd.o_proj), m, dm, TF_LINEAR, Args())
    var residual1 = tape.binary_recorded(ex, x, o, ADD, _fp(fwd.residual1), _fp(bst.d_residual1), True)
    var s_norm2 = Args()
    s_norm2.p0 = _fp(fwd.norm2_sumsq)
    s_norm2.p1 = _fp(bst.norm2_dot)
    var norm2 = tape.custom_into(ex, [residual1, g_norm2], _fp(fwd.norm2_out), _fp(bst.d_norm2_out), m, dm, TF_RMS_NORM_POST, s_norm2)
    var gate = tape.custom_into(ex, [norm2, g_gate], _fp(fwd.gate_proj), _fp(bst.d_gate_proj_out), m, it, TF_LINEAR, Args())
    var up = tape.custom_into(ex, [norm2, g_up], _fp(fwd.up_proj), _fp(bst.d_up_proj_out), m, it, TF_LINEAR, Args())
    var s_swiglu = Args()
    s_swiglu.p0 = _fp(fwd.silu_out)
    var gated = tape.custom_into(ex, [gate, up], _fp(fwd.gated), _fp(bst.d_mlp_gated), m, it, TF_SWIGLU, s_swiglu)
    var down = tape.custom(ex, [gated, g_down], _fp(fwd.down_proj), m, dm, TF_LINEAR, Args())
    var residual2 = tape.binary_recorded(ex, residual1, down, ADD, _fp(fwd.residual2), _fp(bst.in_d_residual2), True)

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

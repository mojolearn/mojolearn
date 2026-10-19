# SPDX-License-Identifier: Apache-2.0
# Mamba-3 block backward on the tape as ONE opaque rule (TAPE_EVERYWHERE_PLAN.md
# path 1, step 1a: the Samba unblocker). The decomposition into kernel-group
# rules (step 1b, tags 73-86) is a later lane's; this file holds M3_BLOCK only.
#
# ---------------------------------------------------------------- the rule
#   M3_BLOCK (tag 72)  inputs [x, norm_w, w_in, dt_bias, bnorm_w, cnorm_w,
#                      b_bias, c_bias, D, w_out] -> residual_out
#                      apply: mamba3_prefill_backward_on verbatim, on the
#                      stages the block's forward recorded
#                      (mamba3_block_forward from the zero state), then one
#                      bit copy of each of the ten gradients into the
#                      destinations the tape hands over (the input's
#                      gradient storage and the nine weight leaves'
#                      variable_into storage, e.g. a trainer's flat
#                      gradient registry).
#
# Bits: the VJP is the hand path's function on the hand path's operands, in
# the same order; the copies are bit copies. Same bits as the hand path by
# construction.
#
# writes_total is True on every edge: the VJP writes each gradient whole and
# the rule copies it whole, so the tape neither fills nor accumulates.
# zeroed_destination is False for the same reason.
#
# Lifetime: mamba3_prefill_backward_on synchronizes before it returns and
# allocates its ten outputs itself. The rule enqueues the ten copies and
# hands the gradient struct to the driver's keep list (no wait here); the
# driver drops the list after its one closing wait. This is what removes the
# per-block synchronize the Samba hand path pays after its copies.
#
# Why copies remain (ten per block): mamba3_prefill_backward_on allocates
# its outputs; an `_into` variant that writes the caller's destinations is a
# change to mamba/impl/modules/mamba3_prefill_backward.mojo (not this lane's
# file) or falls out of the step 1b decomposition, where each kernel group's
# rule writes its tape destination directly.
#
# Import direction: this module imports the mamba modules; no mamba module
# imports autograd.
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext

from autograd.tape import BackwardRules
from sequence.exec_trait import Exec
from sequence.ops import FP, Args
from mamba.impl.modules.mamba3 import Mamba3DeviceStages, Mamba3DeviceWeights
from mamba.impl.modules.mamba3_prefill_backward import (
    Mamba3DeviceGradients,
    mamba3_prefill_backward_on,
)

# Tags: LayerNorm 1, recurrent 16-18, transformer 32-47, Mamba-1/2 48-63,
# LM 64-71, Mamba-3 72-87 (72 the opaque block; 73-86 reserved for the
# decomposition), Samba stack 88-95 (autograd/rules_samba.mojo).
comptime M3_BLOCK = 72
comptime M3_TAG_FIRST = 72
comptime M3_TAG_LAST = 87
#: x, then the nine weights in Mamba3DeviceWeights / registry order
comptime M3_BLOCK_INPUTS = 10

comptime M3WeightsRef = MutPointer[Mamba3DeviceWeights, MutUntrackedOrigin]
comptime M3StagesRef = MutPointer[Mamba3DeviceStages, MutUntrackedOrigin]
comptime M3KeepRef = MutPointer[List[Mamba3DeviceGradients], MutUntrackedOrigin]


@always_inline
def _m3_view(ctx: DeviceContext, p: FP, n: Int) -> DeviceBuffer[DType.float32]:
    """A non-owning view of n floats at p."""
    return DeviceBuffer[DType.float32](ctx, p.unsafe_origin_cast[MutAnyOrigin](), n if n > 0 else 1, owning=False)


def _m3_copy_into(ctx: DeviceContext, dst: FP, mut src: DeviceBuffer[DType.float32], n: Int, what: String) raises:
    """`dst[0:n] = src[0:n]`, device to device, in stream order (no wait).
    A copy of bits, not arithmetic; the hand path's `_d2d`."""
    if len(src) != n:
        raise Error(
            String("mamba3 backward (autograd): the ") + what + " gradient holds " + String(len(src))
            + " floats, the tape's destination " + String(n)
        )
    if n <= 0:
        return
    if Int(dst) == Int(src.unsafe_ptr()):
        return
    var d = _m3_view(ctx, dst, n)
    ctx.enqueue_copy(dst_buf=d, src_buf=src.create_sub_buffer[DType.float32](0, n))


struct Mamba3BlockRules(BackwardRules):
    """The backward of ONE recorded Mamba-3 block forward (M3_BLOCK).

    Holds pointers to the block's weights and recorded forward stages and to
    the driver's keep list; all must outlive the backward, and the driver
    must wait on the context before it drops the keep list."""

    var ctx: DeviceContext
    var w: M3WeightsRef
    var st: M3StagesRef
    var keep: M3KeepRef
    var b: Int
    var l: Int
    var ton: Bool

    def __init__(
        out self, ctx: DeviceContext, w: M3WeightsRef, st: M3StagesRef, keep: M3KeepRef,
        b: Int, l: Int, ton: Bool,
    ):
        self.ctx = ctx.copy()
        self.w = w
        self.st = st
        self.keep = keep
        self.b = b
        self.l = l
        self.ton = ton

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # Never: every destination is written whole by a copy.
        return False

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # The VJP owns each input's whole gradient (one consumer per input).
        return tag == M3_BLOCK

    def supports(self, tag: Int) -> Bool:
        return tag == M3_BLOCK

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        # Every launch goes on self.ctx, the context `ex` wraps.
        if tag != M3_BLOCK:
            raise Error("unknown Mamba-3 backward rule")
        if len(inputs) != M3_BLOCK_INPUTS or len(gradients) != M3_BLOCK_INPUTS:
            raise Error("mamba3 backward (autograd): a block node takes x and the nine weights")
        ref w = self.w[]
        ref st = self.st[]
        var dims = w.dims.copy()
        var m = self.b * self.l
        var dm = dims.d_model
        if rows[0] * cols[0] != m * dm:
            raise Error("mamba3 backward (autograd): the block input must hold [B, L, d_model]")
        var x = _m3_view(self.ctx, inputs[0], m * dm)
        var d_output = _m3_view(self.ctx, upstream, m * dm)
        var tk = Int(perf_counter_ns())
        var g = mamba3_prefill_backward_on(self.ctx, w, st, x, d_output, self.b, self.l, dims, self.ton, tk)
        _m3_copy_into(self.ctx, gradients[0], g.x, rows[0] * cols[0], "x")
        _m3_copy_into(self.ctx, gradients[1], g.block_norm_weight, rows[1] * cols[1], "block norm weight")
        _m3_copy_into(self.ctx, gradients[2], g.in_proj_weight, rows[2] * cols[2], "in_proj weight")
        _m3_copy_into(self.ctx, gradients[3], g.dt_bias, rows[3] * cols[3], "dt_bias")
        _m3_copy_into(self.ctx, gradients[4], g.B_norm_weight, rows[4] * cols[4], "B norm weight")
        _m3_copy_into(self.ctx, gradients[5], g.C_norm_weight, rows[5] * cols[5], "C norm weight")
        _m3_copy_into(self.ctx, gradients[6], g.B_bias, rows[6] * cols[6], "B bias")
        _m3_copy_into(self.ctx, gradients[7], g.C_bias, rows[7] * cols[7], "C bias")
        _m3_copy_into(self.ctx, gradients[8], g.D, rows[8] * cols[8], "D")
        _m3_copy_into(self.ctx, gradients[9], g.out_proj_weight, rows[9] * cols[9], "out_proj weight")
        # the copies read g's buffers: they live in the driver's keep list
        # until its closing wait
        self.keep[].append(g^)

# SPDX-License-Identifier: Apache-2.0
# CNNClassifier's trainer backward on the tape, the only backward (the hand walk
# was deleted 2026-10-19, see the tombstone below). TAPE_EVERYWHERE_PLAN path 3.
#
# ---------------------------------------------------------------- launch inventory
# The deleted hand path, bindings/_mojolearn_x_cnn.mojo fit_epoch_r_binding /
# fit_epoch_d_binding, per step after the forward (the pair gather, every
# block's conv_block_forward_into[True], the head's linear_forward_into[True]
# and softmax_xent_into[True] / softmax_xent_res_loss, which already writes
# the logits gradient `glog`):
#
#   head    linear_backward_into[True] (x_cnn/device.mojo linear_backward_m):
#           dW GEMM TN, bias_grad_gemm (leaf + fold, or the GEMM on ones),
#           dX GEMM NN -> the last block's gout (or `ghead` with no block);
#           one host wait
#   blocks  for j = nb-1 .. 0: conv_block_backward_into[True]
#           (x_cnn/device.mojo conv_block_backward_into): [the conv + ReLU
#           recompute when the forward saved no cols / conv output], the
#           pool + ReLU + row-layout kernel (pool_relu_rows_bwd_at or the
#           tiled rows kernel) or the ReLU rows kernel(s) without a pool,
#           [fill_one_at], the dW GEMM (device_gemm OP_TN, nn14_wgrad_at or
#           _implicit_wgrad), bias_grad_gemm, and for j > 0 the dX GEMM
#           (device_gemm OP_NN + _col2im, or nn14_input_grad_at) -> block
#           j-1's gout; one host wait per block
#   step    opt_many_resident[_h] (unchanged, after the backward)
#
# Glue (moved to the tape): which buffer is each block's input and output
# gradient (the chaining of the blocks and the head), the order of the
# stages, and whether a block owes dX (block 0's input is the data: no
# gradient). Fused kernels (kept, one rule each, called with the hand path's
# arguments on the hand path's buffers): the whole conv block backward
# (CNN_CONV_BLOCK), the head's backward (CNN_LINEAR: the CNN's pinned GEMM
# plan, not the sequence GEMM, so it is not the tape's linear node) and the
# softmax cross-entropy (CNN_SOFTMAX_XENT: its row kernel already wrote the
# logits gradient in the forward, so the rule launches nothing, as LM_CE
# under AFN_LM_HEAD_FUSE).
#
# ---------------------------------------------------------------- the tape graph
#   w_j, b_j (variable_into gw_j, gb_j)
#     -> CNN_CONV_BLOCK j   out_j = pool(relu(conv(in_j)))   grad storage gout_j
#        in_0 is the data (not recorded: no gradient, no scratch); in_j for
#        j > 0 is out_{j-1}
#   head w, b (variable_into head gw, gb)
#     -> CNN_LINEAR         logits = out_{nb-1} W^T + b      grad storage glog
#     -> CNN_SOFTMAX_XENT   the mean loss                    grad storage = the seed
#   seed: one resident word holding 1.0 (the driver's), recorded as the
#   loss's gradient storage, so the tape copies nothing into the root.
#
# ACCUMULATION ORDER. No tensor of this graph has two consumers: each block
# output feeds only the next block (or the head), the logits only the loss,
# and no weight is shared. So every gradient is ONE contribution, written in
# place by its rule (writes_total on every edge: no fill, no scratch, no
# OP_AG_ACCUMULATE). The tape adds in no order of its own: the same kernels
# on the same operands in the same order give the hand path's bits.
#
# ---------------------------------------------------------------- launch count
# Tape: the hand launches above, one for one, in the same order, with the
# same arguments, each entry still ending in its own wait (the `_into[True]`
# entries synchronize and keep their host parameter lists alive until then;
# removing those waits is a kernel-entry change, not this migration's). Plus
# one ex.sync() on an already drained queue at the end of the backward. The
# tape itself launches nothing and allocates nothing (no input without a
# gradient is recorded, zeroed_destination is False, writes_total is True).
#
# The rules launch on cnn_ctx(), the context every x_cnn entry uses, and keep
# the entries' slot workspaces; the DeviceExec wraps that same context, so its
# one sync drains the queue the kernels ran on.
from autograd.tape import Tape, Tensor, BackwardRules
from sequence.exec_trait import Exec
from sequence.exec_device import DeviceExec
from sequence.ops import FP, Args
from x_cnn.ops import FP as XFP, IP as XIP
from x_cnn.ops import CP_N, CP_C, CP_OC, CP_KH, CP_KW, CP_OH, CP_OW, PP_N, PP_C, PP_OH, PP_OW
from x_cnn.device import cnn_ctx, conv_block_backward_into, linear_backward_into

# TOMBSTONE: MOJOLEARN_AUTOGRAD_CNN_OFF (removed: tape is the only backward) deleted 2026-10-19 by lane/del-cnn; code recoverable at the deletion commit^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_CNN_OFF.patch (the AUTOGRAD_CNN switch and the hand walk;
# the tape matched it word for word on Metal, NVIDIA and AMD, x_cnn/checks/tape_check.mojo now checks stored digests).

# Tags 96-111 (LayerNorm 1, recurrent 16-18, transformer 32-47, Mamba 48-63,
# LM 64-71). 99-111 are reserved for the per-layer rules (bare conv, pools,
# BatchNorm, Dropout2d, pad, graph layers).
comptime CNN_CONV_BLOCK = 96
comptime CNN_LINEAR = 97
comptime CNN_SOFTMAX_XENT = 98

# A block handle list, as fit_epoch_r/d receive it.
comptime BH_W = 0
comptime BH_B = 1
comptime BH_GW = 2
comptime BH_GB = 3
comptime BH_OUT = 4
comptime BH_IDX = 5
comptime BH_GOUT = 6
comptime BH_COLS = 7
comptime BH_YCONV = 8


@always_inline
def _x(a: Int) -> XFP:
    return XFP(unsafe_from_address=a)


@always_inline
def _xi(a: Int) -> XIP:
    return XIP(unsafe_from_address=a)


@always_inline
def _t(a: Int) -> FP:
    return FP(unsafe_from_address=a)


def _same(p: FP, addr: Int, what: String) raises:
    # The rules launch on the trainer's own buffers; the tape must hand them
    # exactly those (the driver records them so). Anything else is a driver bug.
    if Int(p) != addr:
        raise Error(String("CNN backward (autograd): ") + what + " is not the trainer's buffer")


def conv_block_out_count(cprm: List[Int32], pprm: List[Int32], pool: Bool) -> Int:
    """Words of a conv block's output: the pool's, or the conv's without one."""
    if pool:
        return Int(pprm[PP_N]) * Int(pprm[PP_C]) * Int(pprm[PP_OH]) * Int(pprm[PP_OW])
    return Int(cprm[CP_N]) * Int(cprm[CP_OC]) * Int(cprm[CP_OH]) * Int(cprm[CP_OW])


struct CnnRules(BackwardRules):
    """Backward adapters for CNNClassifier's fused native ops: each conv
    block (by index), the head and the loss. Holds the trainer's resident
    addresses; all must outlive the backward."""

    var bh: List[List[Int]]
    var head: List[Int]
    var a: List[Int]
    var cs: List[List[Int32]]
    var ps: List[List[Int32]]
    var pools: List[Bool]
    var m: Int
    var flat: Int
    var k: Int
    var seed: Int

    def __init__(
        out self, var bh: List[List[Int]], var head: List[Int], var a: List[Int], var cs: List[List[Int32]],
        var ps: List[List[Int32]], var pools: List[Bool], m: Int, flat: Int, k: Int, seed: Int,
    ):
        self.bh = bh^
        self.head = head^
        self.a = a^
        self.cs = cs^
        self.ps = ps^
        self.pools = pools^
        self.m = m
        self.flat = flat
        self.k = k
        self.seed = seed

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # Never: every entry overwrites every word of the destinations it is
        # handed (the hand path reuses gw, gb, gout and glog across steps
        # with no fill).
        return False

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # Every input has one consumer (accumulation order, header): the
        # rule's write is the input's whole gradient, in place.
        return True

    def supports(self, tag: Int) -> Bool:
        return tag >= CNN_CONV_BLOCK and tag <= CNN_SOFTMAX_XENT

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        # Every launch goes on cnn_ctx(), the context `ex` wraps.
        if tag == CNN_CONV_BLOCK:
            self._block(saved.i0, upstream, gradients)
        elif tag == CNN_LINEAR:
            self._head(upstream, gradients)
        elif tag == CNN_SOFTMAX_XENT:
            self._xent(upstream, gradients)
        else:
            raise Error("unknown CNN backward rule")

    def _block(self, j: Int, upstream: FP, gradients: List[FP]) raises:
        # fit_epoch_r/d's block backward, verbatim
        var nb = len(self.bh)
        if j < 0 or j >= nb:
            raise Error("CNN backward (autograd): node carries no valid block index")
        ref h = self.bh[j]
        _same(upstream, h[BH_GOUT], "the block output gradient")
        var bsrc = self.bh[j - 1][BH_OUT] if j > 0 else self.a[0]
        var dx = self.bh[j - 1][BH_GOUT] if j > 0 else 0
        var want = dx != 0
        var e = 0
        if j > 0:
            if len(gradients) != 3:
                raise Error("CNN backward (autograd): a chained block has three inputs")
            _same(gradients[0], dx, "the block input gradient")
            e = 1
        elif len(gradients) != 2:
            raise Error("CNN backward (autograd): the first block has two inputs")
        _same(gradients[e], h[BH_GW], "the block weight gradient")
        _same(gradients[e + 1], h[BH_GB], "the block bias gradient")
        conv_block_backward_into[True](
            _x(bsrc), _x(h[BH_W]), _x(h[BH_B]), _x(h[BH_GOUT]), _xi(h[BH_IDX]), self.cs[j], self.ps[j],
            self.pools[j], want, _x(dx) if want else _x(h[BH_GB]), _x(h[BH_GW]), _x(h[BH_GB]), h[BH_COLS],
            h[BH_YCONV],
        )

    def _head(self, upstream: FP, gradients: List[FP]) raises:
        # fit_epoch_r/d's head backward, verbatim: dX into the last block's
        # gout, or `ghead` (a[5], never read) with no block
        var nb = len(self.bh)
        _same(upstream, self.a[3], "the logits gradient")
        var src = self.bh[nb - 1][BH_OUT] if nb > 0 else self.a[0]
        var glast = self.bh[nb - 1][BH_GOUT] if nb > 0 else self.a[5]
        var e = 0
        if nb > 0:
            if len(gradients) != 3:
                raise Error("CNN backward (autograd): the head has three inputs after a block")
            _same(gradients[0], glast, "the head input gradient")
            e = 1
        elif len(gradients) != 2:
            raise Error("CNN backward (autograd): the head has two inputs without a block")
        _same(gradients[e], self.head[2], "the head weight gradient")
        _same(gradients[e + 1], self.head[3], "the head bias gradient")
        linear_backward_into[True](
            _x(src), _x(self.head[0]), _x(self.a[3]), self.m, self.flat, self.k, _x(glast), _x(self.head[2]),
            _x(self.head[3]),
        )

    def _xent(self, upstream: FP, gradients: List[FP]) raises:
        # The mean loss's VJP at the unit seed: the forward's row kernel
        # (softmax_xent_row_at, the 1/m folded in) already wrote glog, so
        # the rule reads no upstream, launches nothing and refuses any
        # other seed.
        if Int(upstream) != self.seed:
            raise Error("CNN backward (autograd): the loss rule takes only the unit seed")
        _same(gradients[0], self.a[3], "the logits gradient")


def cnn_backward_autograd(
    bh: List[List[Int]], head: List[Int], a: List[Int], cs: List[List[Int32]], ps: List[List[Int32]],
    pools: List[Bool], m: Int, flat: Int, k: Int, seed_addr: Int, loss_addr: Int,
) raises:
    """CNNClassifier's step backward on one tape: every conv block, the
    head and the softmax cross-entropy, recorded over the forward that
    already ran (its outputs, saved cols and conv outputs, pool winners and
    the logits gradient), then one backward.

    Writes exactly what the hand walk writes: the head's gw and gb, every
    block's gout (but block 0's input gradient, which nobody reads) and
    every block's gw and gb, with the same kernels on the same buffers.

    bh: per block [w, b, gw, gb, out, idx, gout, cols, conv out] (resident
    addresses); head [w, b, gw, gb]; a [x, y, logits, glog, proba, ghead];
    cs / ps / pools: the step's plan per block; m rows, flat head inputs, k
    classes. seed_addr: one resident word holding 1.0; loss_addr: one
    resident word the tape records as the loss's value (never read)."""
    var nb = len(bh)
    if len(cs) != nb or len(ps) != nb or len(pools) != nb:
        raise Error("CNN backward (autograd): one plan per block")
    if len(head) != 4 or len(a) != 6:
        raise Error("CNN backward (autograd): head is [w, b, gw, gb], a is [x, y, logits, glog, proba, ghead]")
    if m <= 0 or flat <= 0 or k <= 0:
        raise Error("CNN backward (autograd): positive rows, inputs and classes required")
    if seed_addr == 0 or loss_addr == 0 or seed_addr == loss_addr:
        raise Error("CNN backward (autograd): the seed and the loss word must be two resident words")
    var ctx = cnn_ctx()
    var ex = DeviceExec(ctx)
    var tape = Tape()
    # ---- the blocks; block j's output gradient is its gout, the storage
    # the hand path hands block j as `g` and block j+1 (or the head) as dx
    var h = Tensor(0, 0, -1)
    for j in range(nb):  # driver-loop: over conv blocks
        ref b = bh[j]
        if len(b) != 9:
            raise Error("CNN backward (autograd): a block is [w, b, gw, gb, out, idx, gout, cols, conv out]")
        var oc = Int(cs[j][CP_OC])
        var ckk = Int(cs[j][CP_C]) * Int(cs[j][CP_KH]) * Int(cs[j][CP_KW])
        if j > 0 and Int(cs[j][CP_N]) != m:
            raise Error("CNN backward (autograd): every block's plan must be at the step's rows")
        var g_w = tape.variable_into(ex, _t(b[BH_W]), _t(b[BH_GW]), oc, ckk)
        var g_b = tape.variable_into(ex, _t(b[BH_B]), _t(b[BH_GB]), 1, oc)
        var saved = Args()
        saved.i0 = j
        var no = conv_block_out_count(cs[j], ps[j], pools[j])
        if j == 0:
            h = tape.custom_into(ex, [g_w, g_b], _t(b[BH_OUT]), _t(b[BH_GOUT]), 1, no, CNN_CONV_BLOCK, saved)
        else:
            h = tape.custom_into(ex, [h, g_w, g_b], _t(b[BH_OUT]), _t(b[BH_GOUT]), 1, no, CNN_CONV_BLOCK, saved)
    # ---- head and loss; the loss's gradient is the unit seed
    var g_hw = tape.variable_into(ex, _t(head[0]), _t(head[2]), k, flat)
    var g_hb = tape.variable_into(ex, _t(head[1]), _t(head[3]), 1, k)
    var lg: Tensor
    if nb > 0:
        lg = tape.custom_into(ex, [h, g_hw, g_hb], _t(a[2]), _t(a[3]), m, k, CNN_LINEAR, Args())
    else:
        lg = tape.custom_into(ex, [g_hw, g_hb], _t(a[2]), _t(a[3]), m, k, CNN_LINEAR, Args())
    var loss = tape.custom_into(ex, [lg], _t(loss_addr), _t(seed_addr), 1, 1, CNN_SOFTMAX_XENT, Args())
    var seed = tape.variable(ex, _t(seed_addr), 1, 1, False)

    var rules = CnnRules(bh.copy(), head.copy(), a.copy(), cs.copy(), ps.copy(), pools.copy(), m, flat, k, seed_addr)
    tape.backward_with_rules(ex, loss, seed, rules)
    tape.release(ex)
    # Drain the queue before the views go back to the caller (every entry
    # already waited; this wait finds the queue empty).
    ex.sync()

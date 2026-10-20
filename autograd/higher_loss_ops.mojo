# SPDX-License-Identifier: Apache-2.0
# THE ONE-HOT CONSTANT OF THE RECORDED LOSS VJPS (lane ag-ho-fused, H2 of
# docs/plans/AUTOGRAD_HIGHER_ORDER.md). backward_recorded lowers the
# cross-entropy VJPs (LOSS_CE, LOSS_SCE) to recorded tape operations,
# (softmax(logits) - onehot(codes)) * g, and the one-hot matrix of the class
# codes is the one value no tape operation computes. It is a constant (no
# node kind, no graph: the second derivative through a target is zero), so
# its only kernel is the forward body below; autograd/higher_fused.mojo
# records the result with Tape.variable(..., requires_grad=False), exactly
# as detach records a constant.
#
# Shared by HostExec and DeviceExec through sequence/dispatch.mojo, so the
# host column and every GPU run the same body. One thread per (row, class)
# element, never one thread per row looping over classes (GPU rule). The
# body is a compare and a store of 1.0 or 0.0: no arithmetic, so the words
# are the same on Apple, NVIDIA, AMD and the host. Typed FP slots only
# (p0, p1), no pointer select, @always_inline (Metal rules).
#
# Codes OP_HL_FIRST..OP_HL_LAST (3610..3619) are this family; 3611..3619
# are reserved for later recorded-loss constants. sequence/dispatch.mojo
# routes and fences them; autograd/higher_ops.mojo (lane H1) holds
# 3600..3609.
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, ld, st

comptime OP_HL_ONEHOT = 3610
comptime OP_HL_FIRST = OP_HL_ONEHOT
comptime OP_HL_LAST = 3619


# ---------------------------------------------------------------- bodies
@always_inline
def op_hl_onehot(t: Int, a: Args):
    """dst[t] = 1.0 when element t's column is its row's class code, else
    0.0. p0 the codes (rows, 1) as float values (exact below 2^24, the
    loss family's convention), p1 dst (rows, i1); i1 the class count.
    Thread t = element t of dst. A code outside [0, i1) matches no column
    (an all-zero row); the cross-entropy forward already gives NaN there."""
    var cols = a.i1
    var r = t // cols
    var c = t - r * cols
    var code = Int(ld(a.p0, r))
    if code == c:
        st(a.p1, t, Float32(1))
    else:
        st(a.p1, t, Float32(0))


@always_inline
def apply_higher_loss[OP: Int](t: Int, a: Args):
    comptime if OP == OP_HL_ONEHOT:
        op_hl_onehot(t, a)


# ---------------------------------------------------------------- launches
def onehot_into[E: Exec](mut ex: E, codes: FP, dst: FP, rows: Int, cols: Int) raises:
    """dst (rows, cols) = the one-hot matrix of codes (rows, 1, class codes
    as float values): one launch, one thread per element."""
    if rows <= 0 or cols <= 0:
        raise Error("onehot: rows and classes must be positive")
    var a = Args()
    a.p0 = codes
    a.p1 = dst
    a.i1 = cols
    ex.launch[OP_HL_ONEHOT](a, rows * cols)

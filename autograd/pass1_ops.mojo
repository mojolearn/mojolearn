# SPDX-License-Identifier: Apache-2.0
# Element bodies of the tape's fused nodes (lane ag-pass1-launches), shared by
# HostExec and DeviceExec. Codes OP_P1_FIRST..OP_P1_LAST (1300..1300) are this
# family; sequence/dispatch.mojo routes the range and asserts it collides with
# no other family (tape 1000, model 1040, data 1100, optim 1110, loss 1200).
#
# Each body reads and writes only the slots in its docstring, never reads back
# a slot it writes, and loads through branches, never a pointer select (Metal
# rule, autograd/ops.mojo _ld_right).
from sequence.ops import Args, ld, st, sub, mul
from checks.numerics import ftz, identical_div, identical_exp
from autograd.loss_ops import _row_max, _row_expsum
from autograd.row_ops import ROW_COOP, row_ce_grad

comptime OP_P1_SCE_VJP = 1300
comptime OP_P1_FIRST = OP_P1_SCE_VJP
comptime OP_P1_LAST = OP_P1_SCE_VJP


@always_inline
def op_p1_sce_vjp(t: Int, a: Args):
    """Row t of the mean softmax cross-entropy's backward, straight into the
    logits' gradient: p0 logits (rows x i1), p1 codes (rows, float values),
    p2 the loss's upstream (one word), p3 destination (rows x i1) =
    (softmax - onehot) * g with g = upstream * f0 (f0 = 1 / rows).

    g is the value the mean's VJP stores for every row (op_ag_reduce_vjp:
    mul(ld(upstream), factor)) and the row body is op_ls_ce_vjp's term for
    term, so the gradient equals cross_entropy_rows followed by mean bit for
    bit, in one launch instead of two. Under ROW_COOP (the IDENTICAL default)
    the row is row_ce_grad's (fold32 sum), op_ls_ce_vjp's under the same
    define, so the equality holds in both arms."""
    comptime if ROW_COOP:
        row_ce_grad(t, a, mul(ld(a.p2, 0), a.f0))
        return
    var cols = a.i1
    var base = t * cols
    var y = Int(ld(a.p1, t))
    var g = mul(ld(a.p2, 0), a.f0)
    var m = _row_max(a.p0, base, cols)
    var s = _row_expsum(a.p0, base, cols, m)
    for c in range(cols):
        var p = ftz(identical_div(ftz(identical_exp(sub(ld(a.p0, base + c), m))), s))
        if c == y:
            p = sub(p, Float32(1))
        st(a.p3, base + c, mul(p, g))


@always_inline
def apply_pass1[OP: Int](t: Int, a: Args):
    comptime if OP == OP_P1_SCE_VJP:
        op_p1_sce_vjp(t, a)

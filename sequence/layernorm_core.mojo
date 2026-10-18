# SPDX-License-Identifier: Apache-2.0
# Shared native LayerNorm launch scheduling. The binding, resident entry,
# and autograd adapter all call these same routines.
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, OP_LN_FWD, OP_LN_BWD_X, OP_LN_BWD_W, dummy_ptr
from sequence.layernorm import LN_FOLD_BLOCK, ln_fold_rows
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


def layer_norm_fwd_core[E: Exec](mut ex: E, X: FP, W: FP, Bb: FP, Y: FP, mean: FP, rstd: FP,
                                 M: Int, D: Int, hw: Bool, hb: Bool, eps: Float32) raises:
    """The forward launch of `layer_norm_py` on executor-side buffers, queued,
    no transfer (lane gap-neural-io: shared with the resident-tensor entry
    `sequence/layernorm_dev.mojo`, the same launch on the same values)."""
    var a = Args()
    a.p0 = X
    a.p1 = W
    a.p2 = Bb
    a.p3 = Y
    a.p4 = mean
    a.p5 = rstd
    a.i0 = D
    a.i1 = 1 if hw else 0
    a.i2 = 1 if hb else 0
    a.f0 = eps
    ex.launch[OP_LN_FWD](a, M)


def layer_norm_bwd_core[E: Exec](mut ex: E, X: FP, W: FP, DY: FP, DX: FP, DW: FP, DB: FP, mean: FP,
                                 rstd: FP, M: Int, D: Int, hw: Bool,
                                 parts_w: FP = dummy_ptr(), parts_b: FP = dummy_ptr()) raises:
    """The backward launches of `layer_norm_py` (dx, then the dweight / dbias
    column folds) after `layer_norm_fwd_core` wrote mean and rstd, queued, no
    transfer (lane gap-neural-io: shared with `sequence/layernorm_dev.mojo`)."""
    var b = Args()
    b.p0 = DY
    b.p1 = X
    b.p2 = W
    b.p3 = DX
    b.p4 = mean
    b.p5 = rstd
    b.i0 = D
    b.i1 = 1 if hw else 0
    ex.launch[OP_LN_BWD_X](b, M)
    var c = Args()
    c.p0 = DY
    c.p1 = X
    c.p2 = DW
    c.p3 = DB
    c.p4 = mean
    c.p5 = rstd
    c.i0 = D
    c.i1 = M
    # FAST: the column folds split over row blocks (about 8192 threads,
    # at least 1024 rows each), then one ordered sum of the S partials
    var S = min(max(8192 // D, 1), M // 1024) if (GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL) else 0
    var RS = (M + S - 1) // S if S > 1 else M
    # IDENTICAL (lane idn-loss-norm-folds, sequence/layernorm.mojo
    # LN_FOLD_BLOCK): fixed blocks of ln_fold_rows(M) rows
    comptime if LN_FOLD_BLOCK:
        RS = ln_fold_rows(M)
    S = (M + RS - 1) // RS
    if S > 1:
        var PW = parts_w
        var PB = parts_b
        if Int(PW) == Int(dummy_ptr()):
            PW = ex.alloc(S * D)
        if Int(PB) == Int(dummy_ptr()):
            PB = ex.alloc(S * D)
        c.p6 = PW
        c.p7 = PB
        c.i2 = S
        c.i3 = RS
        ex.launch[OP_LN_BWD_W](c, S * D)
        c.i4 = 1
        ex.launch[OP_LN_BWD_W](c, D)
    else:
        ex.launch[OP_LN_BWD_W](c, D)

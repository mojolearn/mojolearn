# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE KERNEL BODIES OF THE RECURRENT ESTIMATORS' FIT OPTIONS (lane rnn-parity, Andrew 2026-11-25).

`sequence/recurrent.mojo rnn_fit` with `RnnOptions`: the per-row loss weights (sample weight times class weight)
that the generic weighted loss (autograd/rules_recurrent.mojo rnn_weighted_loss, tape nodes) reads. The gradient
clip is the generic training loop's (autograd/optim.mojo). One body per
code, shared by HostExec and DeviceExec through sequence/dispatch.mojo, so the host column and every GPU run the
same words. Codes OP_RNO_FIRST..OP_RNO_LAST (6740..6749). Elementwise: no reduction here.
"""
from sequence.ops import Args, FP, ld, st, mul

comptime OP_RNO_ROW_WEIGHTS = 6740
comptime OP_RNO_FIRST = 6740
comptime OP_RNO_LAST = 6749


@always_inline
def op_rno_row_weights(t: Int, a: Args):
    """The loss weight of row t of the fit's N S rows (S = i0: T with output="sequence", row i S + s the step s of
    sequence i; else 1): p3[t] = (sample weight) * (class weight of its label), each factor 1.0 when absent.
    i1 = 1: p0 holds the sample weights, i2 = 1 one per step (index t), else one per sequence (index t / S);
    i3 = 1: p2 holds one weight per class code and p1 the rows' codes (float32 words, exact)."""
    var w = Float32(1.0)
    if a.i1 != 0:
        var at = t if a.i2 != 0 else t // a.i0
        w = ld(a.p0, at)
    if a.i3 != 0:
        w = mul(w, ld(a.p2, Int(ld(a.p1, t))))
    st(a.p3, t, w)


@always_inline
def apply_rnn_options[OP: Int](t: Int, a: Args):
    comptime if OP == OP_RNO_ROW_WEIGHTS:
        op_rno_row_weights(t, a)

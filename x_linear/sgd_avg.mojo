# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SGD_PERC_AVG on the device (lane apple-fast-q-clf, 2026-10-04, QUALITY-FIX):
the minibatch Perceptron's mean of its epoch-end iterates (x_linear/sgd.mojo
`SGD_PERC_AVG`, `sgd_perc_avg_on`; the host column's form is in `sgd_mb_one`).
`_sgd_mb_grid` launches `sgd_avg_acc_kernel` after each epoch from
`sgd_perc_avg_from(max_iter)` on and `sgd_avg_fin_kernel` once before the
result words. One thread a weight (and one for the intercept): a grid, no
host step. FAST only; nothing launches them under IDENTICAL or with
-D MOJOLEARN_SGD_PERC_QOLD."""
from std.gpu import block_idx, block_dim, thread_idx
from x_linear.ops import FP, ld, st, fa, fm


def sgd_avg_acc_kernel(w: FP, bias: FP, acc: FP, d: Int32):
    """acc[j] += w[j] (j < d), acc[d] += bias[0]."""
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var dd = Int(d)
    if j < dd:
        st(acc, j, fa(ld(acc, j), ld(w, j)))
    elif j == dd:
        st(acc, dd, fa(ld(acc, dd), ld(bias, 0)))


def sgd_avg_fin_kernel(w: FP, bias: FP, acc: FP, d: Int32, inv: Float32):
    """w[j] = acc[j] * inv (j < d), bias[0] = acc[d] * inv: the mean."""
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var dd = Int(d)
    if j < dd:
        st(w, j, fm(ld(acc, j), inv))
    elif j == dd:
        st(bias, 0, fm(ld(acc, dd), inv))

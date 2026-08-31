# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""`Softmax`: the multinomial objective, `logSoftmaxKernel` and the `C > 1`
linear kernels `linearFwd` / `linearBwd` need for it.

PORT OF `cuml/cpp/src/glm/qn/glm_softmax.cuh` at cuML `00094f7`
(`logSoftmaxKernel`, `launchLogsoftmax`, `Softmax::getLossAndDZ`,
`Softmax::gradNorm = nrmMax`) plus the `C > 1` arms of `glm_base.cuh`'s
`linearFwd` (`:39-57`) and `linearBwd` (`:63-94`), which `glm_base.mojo`
ported for `C = 1` only and now dispatches here for `C > 1`. Do not
improve.

THEIR LAYOUT, copied. `W` is `C x dims` COLUMN-MAJOR (`SimpleDenseMat<T>
W(wFlat.data, C, dims)`, `dense.hpp` default order): weight of class `c`,
feature `j` at `w[c + C*j]`; the bias column is `j = D`, so `w[C*D + c]`.
`Z` and `dZ` are `C x N` column-major: `z[c + C*i]` is the logit of class
`c` for row `i` (`logSoftmaxKernel`: `idx = x + y * C`, `x` the class,
`y` the row). `G` is `W`'s shape. For `C = 1` every one of these collapses
to the vectors `glm_base.mojo` already uses.

THEIR KERNEL, AND WHAT IS REPLACED (DEVIATION 705)
--------------------------------------------------
`logSoftmaxKernel<T, BX, BY>` gives a row (their "column" of the `C x N`
matrix) to `BX` threads and four phases:

    1. etaMax = max over classes, seeded -1e9, `cub::WarpReduce::Reduce(max)`
    2. lse    = sum over classes of exp(eta - etaMax), `WarpReduce::Sum`,
                then etaMax + log(sum)
    3. dZ     = exp(eta - lse) - (class == label)
    4. loss   = (lse - eta_y) / N if the label was seen, `BlockReduce::Sum`,
                then `raft::myAtomicAdd(out, blockSum)`

Three of those are the fold defects IDENTITY_PATHS enumerates: phase 2's
warp sum is a lane-width tree (row 20); phase 4 is a float atomic across
blocks (the `mapThenSumReduce` defect DEVIATION 547 replaced everywhere
else in this solver); and phase 1's `raft::max` is a hardware float max
that answers `max(+0.0, -0.0)` per vendor (row 39, ADDENDUM 11). Here ONE
THREAD OWNS ONE ROW and walks the `C` classes ASCENDING:

    1. the max is a SELECTION: strict `>` from the -1e9 seed, so on a tie
       the LOWER INDEX wins and a hardware max never sees both zeros
    2. the sum is serial ascending through `identical_exp` and `ftz`
    3. as theirs, through `identical_exp`
    4. the per-row loss is STORED to `loss_terms[i]` and summed by
       `glm_base.mojo::sum_terms_kernel` -- one pinned block -- exactly as
       the binary loss's terms are

so the loss, every `dZ` cell and the gradient are a pure function of the
inputs and of `STATS_TPB`. The arithmetic per element is theirs character
for character: `exp(eta - etaMax)`, `etaMax + log(sum)`, `exp(eta - lse) -
delta`, `(lse - eta_y) / N` -- a DIVISION by `N` (int promoted to `T`),
not the `* normalization` multiply the `C = 1` losses use. `C` is small
(their own comment: "performs best for small number of classes") and a
serial walk over it costs nothing that matters.

`-1e9` is their seed and is kept: a row whose logits are all below -1e9
would keep the seed as its max. `x == label` compares the class index,
promoted to `T`, against the float label; a label outside `0..C-1` finds
no class, contributes no loss and subtracts no delta (`delta = false`),
also theirs.

THE `C > 1` LINEAR ARMS (DEVIATION 706)
---------------------------------------
`linearFwd`: `Z <- b` then `Z <- W_weights X^T + Z` through cuBLAS. The
product is `core/gemm.mojo::gemm_nt` -- the vendor matmul under FAST, the
one-thread-per-cell pinned product under IDENTICAL (DEVIATION 526) --
which wants its second operand ROW-major `C x D`; `transpose_w_kernel`
materializes that `C*D` copy (the `C = 1` path already copies `W`'s head
for the same reason). `z[i*C + c]` of the row-major `N x C` result IS
`z[c + C*i]` of their column-major `C x N`, no second layout. The bias is
added as its own seam by `add_bias_multi_kernel`.

`linearBwd`: `G_weights = (1/N) dZ X (+ G)` and `G_bias = mean(dZ, rows)`.
`xtdz_multi_kernel` is `core/column_stats.mojo::xty_kernel` with a class
stride -- one block per `(c, j)` cell, `STATS_TPB` strided partials,
`identical_mul_add`, `pinned_block_sum` -- writing `c + C*j`; the cuBLAS
epilogue is `glm_base.mojo::gemm_epilogue_kernel` over the `C*D` cells;
`mean_rows_multi_kernel` is `raft::stats::mean<true>` per class, `sum *
(1/N)`, one block per class. For `C = 1` each of these is the `C = 1`
kernel it stands beside, and the `C = 1` path keeps calling those
(`glm_base.mojo` dispatches on `dims.C`), so the binary logistic path's
launches are untouched.
"""

from std.gpu import block_dim, block_idx, thread_idx

from core.column_stats import STATS_TPB
from core.pinned_reduce import pinned_block_sum
from mojo_only.numerics import (
    ftz,
    identical_exp,
    identical_log,
    identical_mul_add,
)


#: Their seed for the per-row max (`T etaMax = -1e9`), exactly representable.
comptime SOFTMAX_MAX_SEED = Float32(-1e9)


@always_inline
def softmax_row_max(
    z: MutPointer[Float32, MutAnyOrigin], i: Int, n_classes: Int
) -> Float32:
    """Phase 1's SELECTION over the `C` logits of row `i`: a serial
    ascending scan with a strict `>` from the seed, so the lower index wins
    a tie and `+0.0` / `-0.0` are never adjudicated by a hardware max
    (ADDENDUM 11). The VALUE is theirs (`raft::max` over the same
    operands); only the tie and signed-zero rule is pinned."""
    var eta_max = SOFTMAX_MAX_SEED
    for c in range(n_classes):
        var v = z.unsafe_load(c + n_classes * i)
        if v > eta_max:
            eta_max = v
    return eta_max


def softmax_loss_dz_kernel(
    loss_terms: MutPointer[Float32, MutAnyOrigin],
    z: MutPointer[Float32, MutAnyOrigin],
    labels: MutPointer[Float32, MutAnyOrigin],
    n_classes_in: Int32,
    n_in: Int32,
):
    """`logSoftmaxKernel<T>(out, dZ, in, labels, C, N, getDerivative=true)`
    with `dZ == in` (in place, as `Softmax::getLossAndDZ` calls it), one
    thread per row. `loss_terms[i]` receives the row's `(lse - eta_y) / N`
    (0 when the label matched no class); the caller folds it."""
    var n = Int(n_in)
    var C = Int(n_classes_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var label = labels.unsafe_load(i)
    # Phase 1: the max (selection) and the label's logit.
    var eta_max = softmax_row_max(z, i, C)
    var delta = False
    var eta_y = Float32(0.0)
    for c in range(C):
        if Float32(c) == label:
            delta = True
            eta_y = z.unsafe_load(c + C * i)
    # Phase 2: lse = etaMax + log(sum exp(eta - etaMax)), serial ascending.
    var s = Float32(0.0)
    for c in range(C):
        var e = ftz(identical_exp(ftz(z.unsafe_load(c + C * i) - eta_max)))
        s = ftz(s + e)
    var lse = ftz(eta_max + ftz(identical_log(s)))
    # Phase 3: dZ = exp(eta - lse) - (c == label).
    for c in range(C):
        var p = ftz(identical_exp(ftz(z.unsafe_load(c + C * i) - lse)))
        var d = Float32(1.0) if Float32(c) == label else Float32(0.0)
        z.unsafe_store(c + C * i, ftz(p - d))
    # Phase 4's map: (lse - eta_y) / N; the sum is `sum_terms_kernel`.
    var loss_val = Float32(0.0)
    if delta:
        loss_val = ftz(ftz(lse - eta_y) / Float32(n))
    loss_terms.unsafe_store(i, loss_val)


def transpose_w_kernel(
    w_rm: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    n_classes_in: Int32,
    d_in: Int32,
):
    """`col_slice(W, weights, 0, D)` as a ROW-major `C x D` copy for the
    vendor product: `w_rm[c*D + j] = w[c + C*j]`. A pure copy, no
    arithmetic."""
    var C = Int(n_classes_in)
    var D = Int(d_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell < C * D:
        var c = cell // D
        var j = cell % D
        w_rm.unsafe_store(cell, w.unsafe_load(c + C * j))


def add_bias_multi_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    n_classes_in: Int32,
    d_in: Int32,
    n_in: Int32,
):
    """`linearFwd`'s `+ b` for `C > 1`: `z[c + C*i] += w[C*D + c]`, the bias
    column of `W`, stored through `ftz` as the seam it is."""
    var C = Int(n_classes_in)
    var D = Int(d_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell < C * Int(n_in):
        var c = cell % C
        var b = w.unsafe_load(C * D + c)
        z.unsafe_store(cell, ftz(z.unsafe_load(cell) + b))


def xtdz_multi_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    dz: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
    n_classes_in: Int32,
):
    """`dZ (C x N) * X (N x D)` -> `C x D` column-major, one block per
    output cell `b = c + C*j`, `STATS_TPB` strided row partials and the
    pinned fold -- `core/column_stats.mojo::xty_kernel` with a class
    stride on `dz`. Launch `grid = C*D, block = STATS_TPB` and nothing
    else."""
    var n_rows = Int(n_rows_in)
    var D = Int(n_cols_in)
    var C = Int(n_classes_in)
    var b = Int(block_idx.x)
    var c = b % C
    var j = b // C
    var tid = Int(thread_idx.x)
    var acc = Float32(0.0)
    var r = tid
    while r < n_rows:
        acc = identical_mul_add(
            x.unsafe_load(r * D + j), dz.unsafe_load(c + C * r), acc
        )
        r += STATS_TPB
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        out_v.unsafe_store(b, s0)


def mean_rows_multi_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    dz: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_classes_in: Int32,
):
    """`raft::stats::mean<true>(Gbias, dZ, C, N, false)`: per class `c`,
    `sum_i dz[c + C*i] * (1/N)` -- a MULTIPLY by the ratio (`mean.cuh:36`),
    as `glm_base.mojo::mean_kernel` for `C = 1`. One block per class,
    `STATS_TPB` strided partials, pinned fold; writes `out_v[c]`."""
    var n = Int(n_rows_in)
    var C = Int(n_classes_in)
    var c = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var acc = Float32(0.0)
    var i = tid
    while i < n:
        acc = ftz(acc + dz.unsafe_load(c + C * i))
        i += STATS_TPB
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        var ratio = Float32(1.0) / Float32(n)
        out_v.unsafe_store(c, ftz(s0 * ratio))

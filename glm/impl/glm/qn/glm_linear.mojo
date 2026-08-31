# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`SquaredLoss` and `AbsLoss`: the two regression objectives, per row, and
`nrm1`, the gradient norm the absolute-value losses converge on.

PORT OF `cuml/cpp/src/glm/qn/glm_linear.cuh` at cuML `00094f7`. Whole
file: both `Lz`/`Dlz` pairs and both `gradNorm`s. Plus `nrm1` from
`simple_mat/dense.hpp:313-321`, which `simple_mat/dense.mojo` does not carry
and which three losses (`AbsLoss`, `SVCL1Loss`, `SVRL1Loss`) return from
`gradNorm`; it is placed HERE under the QN-losses lane's file scope and
belongs beside `nrm_max` in `dense.mojo` (HAND-OFF in `glm/README.md`). Do
not improve.

THEIR FOUR FUNCTORS, copied (`glm_linear.cuh:21-60`):

    Squared  lz(y, z)  = diff * diff * 0.5,  diff = z - y
             dlz(y, z) = z - y
             gradNorm  = squaredNorm(grad) * 0.5
    Abs      lz(y, z)  = |z - y|
             dlz(y, z) = z > y ? 1 : (z < y ? -1 : 0)
             gradNorm  = nrm1(grad)

`diff * diff * 0.5` is `T * T` then `* 0.5` -- a DOUBLE literal in C++, so
the float product is promoted, halved exactly and narrowed back; the result
is the float `(diff*diff) * 0.5` bit for bit on every normal value, and a
subnormal result is what `ftz` is for. Every intermediate is stored
through `ftz` (row 10). No multiply-add candidate exists in either loss.

THE LAUNCH SHAPE IS `glm_logistic.mojo`'s: one fused kernel per loss
writing `loss_terms[i] = lz * normalization` (the map half of
`mapThenSumReduce`, `glm_base.cuh:156-163`) and `z[i] = dlz` (`binaryOp`,
`:164`), one thread per row; the SUM is `glm_base.mojo::sum_terms_kernel`
(one pinned block, DEVIATION 547).

`nrm1` upstream is `raft::linalg::rowNorm<L1Norm, rowMajor=true>` over one
row of `len` entries -- `raft::linalg::reduce` with `abs_op`, `add_op`: a
CUB-shaped fold. Here it is the same ONE-BLOCK pinned shape as `dot` and
`nrmMax` (DEVIATION 547): `STATS_TPB` strided partials `acc = acc + |u_i|`
through `ftz`, `pinned_block_sum`, thread 0 writes. DEVIATION 707.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from core.column_stats import STATS_TPB
from core.pinned_reduce import pinned_block_sum
from glm.impl.glm.qn.simple_mat.dense import _read_scalar
from checks.numerics import ftz


@always_inline
def squared_lz(y: Float32, z: Float32) -> Float32:
    """`SquaredLoss::Lz`: `diff * diff * 0.5`."""
    var diff = ftz(z - y)
    return ftz(ftz(diff * diff) * Float32(0.5))


@always_inline
def squared_dlz(y: Float32, z: Float32) -> Float32:
    """`SquaredLoss::Dlz`: `z - y`."""
    return ftz(z - y)


@always_inline
def abs_lz(y: Float32, z: Float32) -> Float32:
    """`AbsLoss::Lz`: `|z - y|`."""
    return abs(ftz(z - y))


@always_inline
def abs_dlz(y: Float32, z: Float32) -> Float32:
    """`AbsLoss::Dlz`: `z > y ? 1 : (z < y ? -1 : 0)`."""
    if z > y:
        return Float32(1.0)
    if z < y:
        return Float32(-1.0)
    return Float32(0.0)


def squared_loss_dz_kernel(
    loss_terms: MutPointer[Float32, MutAnyOrigin],
    z: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    normalization: Float32,
):
    """Per row: `loss_terms[i] = lz * normalization`, `z[i] = dlz`."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var yi = y.unsafe_load(i)
        var zi = z.unsafe_load(i)
        loss_terms.unsafe_store(i, ftz(squared_lz(yi, zi) * normalization))
        z.unsafe_store(i, squared_dlz(yi, zi))


def abs_loss_dz_kernel(
    loss_terms: MutPointer[Float32, MutAnyOrigin],
    z: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    normalization: Float32,
):
    """Per row: `loss_terms[i] = lz * normalization`, `z[i] = dlz`."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var yi = y.unsafe_load(i)
        var zi = z.unsafe_load(i)
        loss_terms.unsafe_store(i, ftz(abs_lz(yi, zi) * normalization))
        z.unsafe_store(i, abs_dlz(yi, zi))


def nrm1_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    u: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`nrm1(u) = sum |u_i|`: ONE block, strided partials, pinned fold.
    Launch `grid = 1, block = STATS_TPB` and nothing else."""
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var acc = Float32(0.0)
    var i = tid
    while i < n:
        acc = ftz(acc + abs(u.unsafe_load(i)))
        i += STATS_TPB
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        out_v.unsafe_store(0, s0)


def nrm1(
    ctx: DeviceContext,
    mut u: DeviceBuffer[DType.float32],
    n: Int,
    mut scalar: DeviceBuffer[DType.float32],
) raises -> Float32:
    """`nrm1(u, tmp_dev, stream)`, `dense.hpp:313`."""
    ctx.enqueue_function[nrm1_kernel](
        scalar.unsafe_ptr(), u.unsafe_ptr(), Int32(n),
        grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
    )
    return _read_scalar(ctx, scalar)

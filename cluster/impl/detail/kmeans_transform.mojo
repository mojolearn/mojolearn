# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The `n x k` distance matrix of `KMeans.transform` (2026-09-15).

Reference: `cuvs::cluster::kmeans::detail::kmeans_transform`
(`cuvs/cpp/src/cluster/detail/kmeans.cuh:1178-1219`, v26.08.00), which tiles
the rows and calls `pairwise_distance_kmeans` (`kmeans_common.cuh:315-345`)
under the model's metric: `L2Expanded` gives SQUARED distances and
`L2SqrtExpanded` their roots; every other metric is refused, as the fit
refuses it.

THE CELL IS THE FUSED KERNEL'S EPILOG, statement for statement. Both of the
reference's paths, the pairwise `distance<L2Expanded>` of transform and the
fused argmin of predict, form a cell with the same `l2_exp_distance_op`
(`distance/detail/distance_ops/l2_exp.cuh:95-136`): `acc += x * y` over the
features, `xn + yn - 2 * acc`, the positivity clamp and the self-neighbor
guard `!((val * val < 1e-6) * (xn == yn))`, then `sqrt` per cell for
`L2SqrtExpanded`. This kernel writes that cell with the fused kernel's
pinned spelling (`simt_kernel.mojo`, IDENTITY_PATHS rows 9 and 10): the
feature sum in ascending feature order through `identical_mul_add`, every
partial through `ftz`, the combine as ONE `identical_mul_add(-2, acc,
xn + yn)`, and the root `identical_sqrt` (DEVIATION 2715). So
`transform(X)[i, predict(X)[i]]` is the row minimum bit for bit, and the
host restatement (`kmeans_oracle.mojo::host_kmeans_transform`) is the same
arithmetic.

DEVIATION 2790: the reference's pairwise kernel accumulates the dot product
in its contraction policy's tile order (or a GEMM), which is a vendor and
policy dependent summation order; here the order is ascending feature index
on every column, the order the fused kernel already pins. No reduction runs
across cells, so the launch geometry below is scheduling only and moves no
bit.
"""

from max.gpu import block_dim, block_idx, thread_idx

from checks.numerics import ftz, identical_mul_add, identical_sqrt


#: Threads per block. SCHEDULING ONLY: each thread writes one cell from its
#: own row and column, with no reduction across threads, so this width
#: cannot move a bit.
comptime TRANSFORM_TPB = 256

#: `get_clamp_precision<float, float>()`, the fused kernel's
#: `FUSED_CLAMP_PRECISION`.
comptime TRANSFORM_CLAMP_PRECISION = Float32(1.0e-6)


def kmeans_transform_kernel(
    dist_out: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    centroids: MutPointer[Float32, MutAnyOrigin],
    x_norm: MutPointer[Float32, MutAnyOrigin],
    centroid_norm: MutPointer[Float32, MutAnyOrigin],
    n_samples_in: Int32,
    n_clusters_in: Int32,
    n_features_in: Int32,
    is_sqrt_in: Int32,
):
    """One thread per cell `(row, col)` of the row-major `n x k` output."""
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var k = Int(n_clusters_in)
    var n = Int(n_samples_in)
    if cell >= n * k:
        return
    var d = Int(n_features_in)
    var row = cell // k
    var col = cell - row * k
    var acc = Float32(0.0)
    for p in range(d):
        acc = ftz(
            identical_mul_add(
                ftz(x.unsafe_load(row * d + p)),
                ftz(centroids.unsafe_load(col * d + p)),
                acc,
            )
        )
    var xn = x_norm.unsafe_load(row)
    var yn = centroid_norm.unsafe_load(col)
    var dist = ftz(
        identical_mul_add(Float32(-2.0), ftz(acc), ftz(ftz(xn) + ftz(yn)))
    )
    if dist <= Float32(0.0) or (
        dist * dist < TRANSFORM_CLAMP_PRECISION and xn == yn
    ):
        dist = Float32(0.0)
    if is_sqrt_in != 0:
        dist = identical_sqrt(dist)
    dist_out.unsafe_store(cell, dist)

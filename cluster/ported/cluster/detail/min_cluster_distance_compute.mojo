# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""Assign every sample to its nearest centroid, tiled to bound memory.

PORT OF `minClusterAndDistanceCompute`,
`cuvs/src/cluster/detail/kmeans_common.cuh:360-493`, at cuVS `94c2819`.
Partial. Do not improve. (There is no `minClusterDistanceCompute.cu` in
cuVS; this function lives in `kmeans_common.cuh`.)

This is the assignment half of Lloyd's algorithm and it is where essentially
all the time goes. Their dispatch has exactly TWO arms and the selector is a
metric test with nothing else in it (`is_fused`, `:378-379`):

    L2Expanded / L2SqrtExpanded -> fusedDistanceNNMinReduce  (`:430-449`)
    anything else               -> pairwise matrix + reduce  (`:450-491`)

k-means's default metric is L2Expanded, so THEIR DISPATCH TAKES THE FUSED
ARM, and so does this file. The fused kernel is ported at
`distance/fused_distance_nn/simt_kernel.mojo` and writes no distance tile at
all. The second arm is kept below only so the two can be diffed; it is not
the path.

Note what this means for the tiling: on the fused arm THEY force
`dataBatchSize = n_samples` (`:380`) and log that `batch_samples` is being
ignored (`detail/kmeans.cuh:837-846`). Tiling is the OTHER arm's property.

THE TILING IS THE POINT OF THE FILE
-----------------------------------
Their loop structure, copied:

    for dIdx over samples, step dataBatchSize
        for cIdx over centroids, step centroidsBatchSize
            reduce this tile into the running per-row minimum

The output is `n_samples` pairs, so it is small. What tiling bounds is the
INTERMEDIATE: the GEMM writes `dataBatchSize x centroidsBatchSize` floats
and nothing else in the algorithm is that large. At their default
`batch_samples = 1 << 15` and `batch_centroids = 0` (meaning all centroids)
the buffer is 32768 x k floats, which is 4 MB at k=32 and stays flat as the
dataset grows. **That is the entire reason k-means fits on a laptop GPU with
a dataset that does not**, and it is a property of their defaults, not of the
algorithm.

One departure of theirs is copied even though it looks like a bug guard:

    dataBatchSize = min(dataBatchSize, numeric_limits<IndexT>::max() / centroidsBatchSize)

The tile is indexed with `IndexT`, so the tile itself, not the dataset, is
what must fit the index type.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.gemm import gemm_nt
from core.row_norms import NORM_TPB, row_norm_kernel
from cluster.ported.cluster.detail.kmeans_common import (
    centroid_norms_take_sqrt,
    metric_is_sqrt,
)
from cluster.ported.cluster.kmeans_params import (
    get_centroids_batch_size,
    get_data_batch_size,
)
from cluster.ported.distance.fused_distance_nn.simt_kernel import (
    FUSED_NORMAL_KBLK,
    FUSED_NORMAL_TC,
    FUSED_NORMAL_TR,
    FUSED_SKINNY_KBLK,
    FUSED_SKINNY_TC,
    FUSED_SKINNY_TR,
    fused_distance_nn_kernel,
    fused_is_skinny,
    fused_veclen_for,
)
from neighbors.ported.distance.detail.pairwise_distance_base import (
    launch_config_generator,
)
from cluster.ported.distance.unfused_distance_nn import (
    REDUCE_MIN_TPB,
    reduce_min_kernel,
)


def compute_centroid_norms(
    ctx: DeviceContext,
    mut centroids: DeviceBuffer[DType.float32],
    mut centroid_norm: DeviceBuffer[DType.float32],
    n_clusters: Int,
    n_features: Int,
    metric: Int,
) raises:
    """`kmeans_common.cuh:383-389`, the fused arm's centroid `rowNorm`.

    Every assignment, not once per fit: the centroids moved.
    """
    ctx.enqueue_function[row_norm_kernel](
        centroid_norm.unsafe_ptr(),
        centroids.unsafe_ptr(),
        Int32(n_features),
        Int32(1 if centroid_norms_take_sqrt(metric) else 0),
        grid_dim=(n_clusters, 1, 1),
        block_dim=(NORM_TPB, 1, 1),
    )


def _launch_fused[
    veclen: Int, kblk: Int, tr: Int, tc: Int
](
    ctx: DeviceContext,
    mut out_key: DeviceBuffer[DType.uint32],
    mut out_value: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut centroids: DeviceBuffer[DType.float32],
    mut x_norm: DeviceBuffer[DType.float32],
    mut centroid_norm: DeviceBuffer[DType.float32],
    n_samples: Int,
    n_clusters: Int,
    n_features: Int,
    is_sqrt: Int32,
) raises:
    """One policy instantiation of the fused kernel, launched with THEIR grid
    computation: `launchConfigGenerator<P>(m, n, shmemSize, kernel)` at
    `fused_l2_nn.cuh:135-138`, ported (M4 inputs) in
    `neighbors/gbdt/distance/detail/pairwise_distance_base.mojo`.

    `grid.x` is PINNED to 1: the cross-block merge (`updateReducedVal`'s
    mutex) is the `replaced` row in `PORTED_MAP.tsv`, so a `grid.x > 1`
    launch would race the per-row writes. Their generator returns
    `grid.x == 1` anyway whenever the row tiles alone fill the device, which
    is every k-means shape this repo ships; the pin only bites at small `m`,
    where their kernel would split the column axis and ours runs fewer
    blocks. `grid.y` is theirs, and the kernel grid-strides the row axis
    exactly as `PairwiseDistances::run()` does, so `grid.y` is occupancy,
    not coverage.

    The smem argument is the kernel's ACTUAL single-buffered footprint
    (`fused_smem_bytes`' terms, inlined comptime here), which is what their
    call feeds the occupancy query -- theirs passes the double-buffered
    `P::SmemSize` because that is what their kernel allocates.
    """
    comptime nthreads = tr * tc
    comptime mblk = 4 * tr
    comptime nblk = 4 * tc
    comptime smem_stride = kblk + veclen
    comptime smem_bytes = (mblk + nblk) * smem_stride * 4 + (mblk + nblk) * 4

    var cfg = launch_config_generator(
        n_samples, n_clusters, mblk, nblk, nthreads, smem_bytes
    )
    comptime kern = fused_distance_nn_kernel[veclen, kblk, tr, tc]
    ctx.enqueue_function[kern](
        out_key.unsafe_ptr(),
        out_value.unsafe_ptr(),
        x.unsafe_ptr(),
        centroids.unsafe_ptr(),
        x_norm.unsafe_ptr(),
        centroid_norm.unsafe_ptr(),
        Int32(n_samples),
        Int32(n_clusters),
        Int32(n_features),
        is_sqrt,
        grid_dim=(1, cfg[1], 1),
        block_dim=(nthreads, 1, 1),
    )


def min_cluster_and_distance_compute(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut x_norm: DeviceBuffer[DType.float32],
    mut centroids: DeviceBuffer[DType.float32],
    mut centroid_norm: DeviceBuffer[DType.float32],
    mut dist_buf: DeviceBuffer[DType.float32],
    mut out_key: DeviceBuffer[DType.uint32],
    mut out_value: DeviceBuffer[DType.float32],
    n_samples: Int,
    n_features: Int,
    n_clusters: Int,
    metric: Int,
    batch_samples: Int,
    batch_centroids: Int,
) raises:
    """`minClusterAndDistanceCompute`, unfused arm.

    `dist_buf` must hold `data_batch x centroid_batch` floats. The caller
    allocates it once for the whole fit, which is theirs: the buffer is
    `L2NormBuf_OR_DistBuf`, a single resizable allocation reused for two
    unrelated purposes, and their own comment says the reuse exists because
    RMM would otherwise reallocate every call.

    The `initOutBuffer` flag becomes "is this the first centroid tile". They
    pre-fill the output with `(0, FLT_MAX)` in a separate `matrix::fill` and
    then always merge; folding the first tile into an initializing write
    removes a full-length pass and cannot change the result, because merging
    against `FLT_MAX` is the identity.
    """
    var data_batch = get_data_batch_size(batch_samples, n_samples)
    var centroid_batch = get_centroids_batch_size(batch_centroids, n_clusters)

    var is_sqrt = Int32(1 if metric_is_sqrt(metric) else 0)

    # THE FUSED PATH, which is the one their dispatch takes for this metric
    # (`is_fused`, `kmeans_common.cuh:378-379` and `:430-449`). The CUTLASS
    # specialization of it is unportable but the SIMT one is not, and it is
    # the one that never writes the distance tile. See
    # `distance/fused_distance_nn/simt_kernel.mojo`.
    #
    # The INSTANTIATION is their selection computation, not a constant:
    # `fused_veclen_for` (the 16/8/1-byte alignment ladder of
    # `fused_distance_nn-inl.cuh:107-110/158/210`, fed the real base
    # addresses) times `fused_is_skinny` (`:105`, `k < 32` takes
    # Policy4x4Skinny). `check_fused_policy_dispatch` pins both.
    #
    # `dist_buf` is now unused on this path and is kept in the signature so
    # the unfused arm stays reachable for differential testing.
    var vl = fused_veclen_for(
        n_features, Int(x.unsafe_ptr()), Int(centroids.unsafe_ptr())
    )
    if fused_is_skinny(n_features):
        if vl == 4:
            _launch_fused[
                4, FUSED_SKINNY_KBLK, FUSED_SKINNY_TR, FUSED_SKINNY_TC
            ](
                ctx, out_key, out_value, x, centroids, x_norm, centroid_norm,
                n_samples, n_clusters, n_features, is_sqrt,
            )
        elif vl == 2:
            _launch_fused[
                2, FUSED_SKINNY_KBLK, FUSED_SKINNY_TR, FUSED_SKINNY_TC
            ](
                ctx, out_key, out_value, x, centroids, x_norm, centroid_norm,
                n_samples, n_clusters, n_features, is_sqrt,
            )
        else:
            _launch_fused[
                1, FUSED_SKINNY_KBLK, FUSED_SKINNY_TR, FUSED_SKINNY_TC
            ](
                ctx, out_key, out_value, x, centroids, x_norm, centroid_norm,
                n_samples, n_clusters, n_features, is_sqrt,
            )
    else:
        if vl == 4:
            _launch_fused[
                4, FUSED_NORMAL_KBLK, FUSED_NORMAL_TR, FUSED_NORMAL_TC
            ](
                ctx, out_key, out_value, x, centroids, x_norm, centroid_norm,
                n_samples, n_clusters, n_features, is_sqrt,
            )
        elif vl == 2:
            _launch_fused[
                2, FUSED_NORMAL_KBLK, FUSED_NORMAL_TR, FUSED_NORMAL_TC
            ](
                ctx, out_key, out_value, x, centroids, x_norm, centroid_norm,
                n_samples, n_clusters, n_features, is_sqrt,
            )
        else:
            _launch_fused[
                1, FUSED_NORMAL_KBLK, FUSED_NORMAL_TR, FUSED_NORMAL_TC
            ](
                ctx, out_key, out_value, x, centroids, x_norm, centroid_norm,
                n_samples, n_clusters, n_features, is_sqrt,
            )


def min_cluster_and_distance_compute_unfused(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut x_norm: DeviceBuffer[DType.float32],
    mut centroids: DeviceBuffer[DType.float32],
    mut centroid_norm: DeviceBuffer[DType.float32],
    mut dist_buf: DeviceBuffer[DType.float32],
    mut out_key: DeviceBuffer[DType.uint32],
    mut out_value: DeviceBuffer[DType.float32],
    n_samples: Int,
    n_features: Int,
    n_clusters: Int,
    metric: Int,
    batch_samples: Int,
    batch_centroids: Int,
) raises:
    """The unfused arm, kept reachable so the two can be diffed.

    Their selector picks between these; keeping both means a disagreement
    between them is findable, which is worth more than the file it costs.
    """
    var data_batch = get_data_batch_size(batch_samples, n_samples)
    var centroid_batch = get_centroids_batch_size(batch_centroids, n_clusters)
    var is_sqrt = Int32(1 if metric_is_sqrt(metric) else 0)

    var d_idx = 0
    while d_idx < n_samples:
        var ns = min(data_batch, n_samples - d_idx)

        var c_idx = 0
        while c_idx < n_clusters:
            var nc = min(centroid_batch, n_clusters - c_idx)

            # z[ns x nc] = X[ns x d] . C[nc x d]^T
            #
            # `create_sub_buffer` windows rather than pointer offsets: MAX's
            # matmul takes a TileTensor over a DeviceBuffer and there is no
            # offset form of that. Same bytes, no copy. Same reason the k-NN
            # driver windows its query tile.
            var x_tile = x.create_sub_buffer[DType.float32](
                d_idx * n_features, ns * n_features
            )
            var c_tile = centroids.create_sub_buffer[DType.float32](
                c_idx * n_features, nc * n_features
            )
            gemm_nt(ctx, dist_buf, x_tile, c_tile, ns, nc, n_features)

            # One block per row of the tile; the block strides the centroids.
            ctx.enqueue_function[reduce_min_kernel](
                out_key.unsafe_ptr().unsafe_offset(d_idx),
                out_value.unsafe_ptr().unsafe_offset(d_idx),
                dist_buf.unsafe_ptr(),
                x_norm.unsafe_ptr().unsafe_offset(d_idx),
                centroid_norm.unsafe_ptr().unsafe_offset(c_idx),
                Int32(nc),
                Int32(metric),
                is_sqrt,
                Int32(1 if c_idx == 0 else 0),
                Int32(c_idx),
                grid_dim=(ns, 1, 1),
                block_dim=(REDUCE_MIN_TPB, 1, 1),
            )

            c_idx += nc
        d_idx += ns

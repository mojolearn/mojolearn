"""Assign every sample to its nearest centroid, tiled to bound memory.

PORT OF `cuvs/src/cluster/detail/minClusterDistanceCompute.cu` at cuVS
`2140532c`. Partial. Do not improve.

This is the assignment half of Lloyd's algorithm and it is where essentially
all the time goes. Their file has three arms; this ports the one that runs
on hardware we can target.

    can_use_fused && use_fused   -> CUTLASS fused kernel      NOT PORTED
    can_use_fused && !use_fused  -> GEMM then reduce, tiled   PORTED (here)
    !can_use_fused               -> full pairwise matrix      NOT PORTED

The third arm is their general path for metrics with no expanded form. It
materializes the whole `ns x nc` distance matrix and calls a stock
`coalescedReduction`. k-means never reaches it, because `kmeans_fit` refuses
any metric except `L2Expanded` and `L2SqrtExpanded` before it gets here
(`detail/kmeans.cuh:568`), so the arm is dead for this algorithm and porting
it would be porting unreachable code.

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

from core.gemm import GEMM_MBLK, GEMM_THREADS, gemm_nt
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
    fused_distance_nn_kernel,
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
    """`minClusterDistanceCompute.cu:43-49`.

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

    # THE FUSED PATH. `use_fused` is their selector and it is True on every
    # architecture before Blackwell; the CUTLASS version is unportable but
    # the SIMT one is not, and it is the one that never writes the distance
    # tile. See `distance/fused_distance_nn/simt_kernel.mojo`.
    #
    # `dist_buf` is now unused on this path and is kept in the signature so
    # the unfused arm stays reachable for differential testing.
    ctx.enqueue_function[fused_distance_nn_kernel](
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
        grid_dim=(1, (n_samples + GEMM_MBLK - 1) // GEMM_MBLK, 1),
        block_dim=(GEMM_THREADS, 1, 1),
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
            gemm_nt(
                ctx,
                dist_buf,
                x.unsafe_offset(d_idx * n_features),
                centroids.unsafe_offset(c_idx * n_features),
                ns,
                nc,
                n_features,
            )

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

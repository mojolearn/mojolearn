"""RAFT `cpp/include/raft/stats/detail/silhouette_score.cuh` (ebf9268): the
pieces the BATCHED path reaches -- `SilOp` (:156-169) and `countLabels`
(:100-136). The unbatched `silhouette_score` driver (:187-308:
`pairwise_distance` of the whole n x n matrix, `reduce_cols_by_key`,
`populateAKernel`, `matrixVectorOp(DivOp)`, `reduce(min_op)`) is NOT
PORTED: cuML's Python surface (`silhouette_score.pyx:_silhouette_coeff`)
always calls the batched entry with `chunksize = 40000` by default, so
their dispatch never takes it (PORTING_RULES 0b-i). UNPORTED.tsv.

SilOp, THEIRS (:156-169), per sample with `a` its mean intra-cluster
distance and `b` the least mean distance to another cluster:

    if (a == 0 && b == 0 || a == b) return 0;
    else if (a == -1)               return 0;     (the unbatched path's singleton mark)
    else if (a > b)                 return (b - a) / a;
    else                            return (b - a) / b;

sklearn `silhouette_samples`: `(b - a) / max(a, b)` then `nan_to_num`.
THE +0.0 / -0.0 HAZARD ON `max(a, b)`: sklearn's spelling divides `b - a`
by `max(a, b)`; on an exact tie `b - a` is `+0.0` (x - x is +0.0 in
round-to-nearest) and the quotient is +0.0, and on `a == b == 0` it is
0/0 = NaN, mapped to 0 by `nan_to_num`. RAFT never calls `max`: its first
branch returns `+0.0` for both cases, so no vendor's `max(+0, -0)`
convention (IDENTITY_PATHS row 13) can reach the score. `-0.0` cannot
arise as `a` or `b` here at all -- both are sums of nonnegative terms
seeded `+0.0` (or `numeric_limits::max()`), and a `-0.0` minus a `+0.0`
never happens. Ours is RAFT's spelling, branch for branch.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from metrics.ported.stats.detail.histogram import histogram
from mojo_only.numerics import ftz


@always_inline
def sil_op(a: Float32, b: Float32) -> Float32:
    """`SilOp<DataT>::operator()(a, b)` (:156-169)."""
    if (a == Float32(0.0) and b == Float32(0.0)) or a == b:
        return Float32(0.0)
    elif a == Float32(-1.0):
        return Float32(0.0)
    elif a > b:
        return ftz(ftz(b - a) / a)
    else:
        return ftz(ftz(b - a) / b)


def count_labels(
    ctx: DeviceContext,
    mut labels: DeviceBuffer[DType.int32],
    mut bin_count_array: DeviceBuffer[DType.int32],
    n_rows: Int,
    n_unique_labels: Int,
) raises:
    """`countLabels(labels, binCountArray, nRows, nUniqueLabels, ...)`
    (:100-136): `cub::DeviceHistogram::HistogramEven` with unit bins over
    `[0, nUniqueLabels)`, the same integers `histogram.mojo` produces. The
    batched path's `get_cluster_counts` (batched/silhouette_score.cuh:
    109-124) calls exactly this."""
    histogram(ctx, bin_count_array, n_unique_labels, labels, n_rows, Int32(0))

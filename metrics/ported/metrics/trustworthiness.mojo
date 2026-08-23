"""cuML `cpp/src/metrics/trustworthiness.cu` (265b9da): the one explicit
instantiation `trustworthiness_score<float, DistanceType::L2SqrtUnexpanded>
(h, X, X_embedded, n, m, d, n_neighbors, batchSize)` forwarding to
`cuvs::stats::trustworthiness_score` (= RAFT's detail). `X` and
`X_embedded` are host row-major Float32 here because the embedded k-NN
goes through `neighbors/estimator.mojo::knn_search`, whose boundary is host
pointers (DEVIATION 655)."""

from max.gpu.host import DeviceContext

from metrics.ported.stats.detail.trustworthiness_score import (
    trustworthiness_score as _raft_trust,
)


def trustworthiness_score(
    ctx: DeviceContext,
    mut x: List[Float32],
    mut x_embedded: List[Float32],
    n: Int,
    m: Int,
    d: Int,
    n_neighbors: Int,
    batch_size: Int = 512,
) raises -> Float64:
    """`double trustworthiness_score<float, L2SqrtUnexpanded>(h, X,
    X_embedded, n, m, d, n_neighbors, batchSize)`."""
    return _raft_trust(ctx, x, x_embedded, n, m, d, n_neighbors, batch_size)

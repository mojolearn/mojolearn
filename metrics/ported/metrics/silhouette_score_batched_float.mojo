# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""cuML `cpp/src/metrics/silhouette_score_batched_float.cu` (265b9da):
`ML::Metrics::Batched::silhouette_score(handle, X, n_rows, n_cols, y,
n_labels, scores, chunk, metric)` forwards to
`cuvs::stats::silhouette_score_batched` (= RAFT's batched detail). The
`float` instantiation; `silhouette_score_batched_double.cu` and the
unbatched `silhouette_score.cu` are NOT ported (UNPORTED.tsv: no Float64
on device; the unbatched path is never dispatched by cuML's Python).
`scores` may be a buffer of >= n_rows floats and receives the per-sample
coefficients (cuML's `silhouette_samples`); RAFT's `nullptr` arm allocates
one internally -- here the caller always passes one."""

from max.gpu.host import DeviceBuffer, DeviceContext

from metrics.ported.stats.detail.batched.silhouette_score import (
    DISTANCE_L2_SQRT_UNEXPANDED,
    silhouette_score as _raft_batched,
)


def silhouette_score(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    mut y: DeviceBuffer[DType.int32],
    n_labels: Int,
    mut scores: DeviceBuffer[DType.float32],
    chunk: Int = 40000,
    metric: Int = DISTANCE_L2_SQRT_UNEXPANDED,
) raises -> Float32:
    """`float Batched::silhouette_score(handle, float* X, int n_rows, int
    n_cols, int* y, int n_labels, float* scores, int chunk, DistanceType
    metric)`. Defaults are cuML's Python defaults (`chunksize = 40000`,
    `metric = 'euclidean'`)."""
    return _raft_batched(
        ctx, x, n_rows, n_cols, y, n_labels, scores, chunk, metric
    )

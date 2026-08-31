# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""cuML `cpp/src/metrics/adjusted_rand_index.cu` (265b9da): `ML::Metrics::adjusted_rand_index` forwards to `raft::stats::adjusted_rand_index<int, unsigned long long>`. The `int` overloads (cuML's Python passes int32 labels)."""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from metrics.impl.stats.detail.adjusted_rand_index import (
    compute_adjusted_rand_index,
    compute_adjusted_rand_index_traced,
)


def adjusted_rand_index(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.int32],
    mut y_hat: DeviceBuffer[DType.int32],
    n: Int,
) raises -> Float64:
    """`double adjusted_rand_index(handle, const int* y, const int* y_hat, const int n)`."""
    return compute_adjusted_rand_index(ctx, y, y_hat, n)


def adjusted_rand_index_traced(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut y: DeviceBuffer[DType.int32],
    mut y_hat: DeviceBuffer[DType.int32],
    n: Int,
) raises -> Float64:
    """The same call carrying a card (`metrics.ari.uniq`,
    `metrics.ari.contingency`, `metrics.ari.pair_sums`)."""
    return compute_adjusted_rand_index_traced(ctx, trace, y, y_hat, n)

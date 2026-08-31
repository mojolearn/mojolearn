# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""cuML `cpp/src/metrics/rand_index.cu` (265b9da): `ML::Metrics::rand_index` forwards to `raft::stats::rand_index`. The `int` overloads (cuML's Python passes int32 labels). cuML's signature takes `const double* y`; labels here are Int32 (the Python side casts), which is what `compute_rand_index<T>` compares with `==` either way."""

from max.gpu.host import DeviceBuffer, DeviceContext

from metrics.derived.stats.detail.rand_index import compute_rand_index


def rand_index(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.int32],
    mut y_hat: DeviceBuffer[DType.int32],
    n: Int,
) raises -> Float64:
    """`double rand_index(handle, y, y_hat, n)`."""
    return compute_rand_index(ctx, y, y_hat, n)

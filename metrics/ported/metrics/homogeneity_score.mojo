# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""cuML `cpp/src/metrics/homogeneity_score.cu` (265b9da): `ML::Metrics::homogeneity_score` forwards to `raft::stats::homogeneity_score`. The `int` overloads (cuML's Python passes int32 labels)."""

from max.gpu.host import DeviceBuffer, DeviceContext

from metrics.ported.stats.detail.homogeneity_score import (
    homogeneity_score as _raft_h,
)


def homogeneity_score(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.int32],
    mut y_hat: DeviceBuffer[DType.int32],
    n: Int,
    lower_class_range: Int32,
    upper_class_range: Int32,
) raises -> Float64:
    """`double homogeneity_score(handle, y, y_hat, n, lower, upper)`."""
    return _raft_h(ctx, y, y_hat, n, lower_class_range, upper_class_range)

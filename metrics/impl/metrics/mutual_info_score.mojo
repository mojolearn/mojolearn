# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""cuML `cpp/src/metrics/mutual_info_score.cu` (265b9da): `ML::Metrics::mutual_info_score` forwards to `raft::stats::mutual_info_score`. The `int` overloads (cuML's Python passes int32 labels)."""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from metrics.impl.stats.detail.mutual_info_score import (
    mutual_info_score as _raft_mi,
    mutual_info_score_traced as _raft_mi_traced,
)


def mutual_info_score(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.int32],
    mut y_hat: DeviceBuffer[DType.int32],
    n: Int,
    lower_class_range: Int32,
    upper_class_range: Int32,
) raises -> Float64:
    """`double mutual_info_score(handle, y, y_hat, n, lower, upper)`."""
    return _raft_mi(ctx, y, y_hat, n, lower_class_range, upper_class_range)


def mutual_info_score_traced(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut y: DeviceBuffer[DType.int32],
    mut y_hat: DeviceBuffer[DType.int32],
    n: Int,
    lower_class_range: Int32,
    upper_class_range: Int32,
    tag_prefix: String,
) raises -> Float64:
    """The same call carrying a card (`<tag_prefix>.contingency`,
    `.row_sums`, `.col_sums`, `.terms`, `.acc`). The prefix is the
    caller's: one card computes MI over BOTH argument orders."""
    return _raft_mi_traced(
        ctx,
        trace,
        y,
        y_hat,
        n,
        lower_class_range,
        upper_class_range,
        tag_prefix,
    )

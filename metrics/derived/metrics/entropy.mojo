# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""cuML `cpp/src/metrics/entropy.cu` (265b9da): `ML::Metrics::entropy` forwards to `raft::stats::entropy`. The `int` overloads (cuML's Python passes int32 labels)."""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from metrics.derived.stats.detail.entropy import (
    entropy as _raft_entropy,
    entropy_traced as _raft_entropy_traced,
)


def entropy(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.int32],
    n: Int,
    lower_class_range: Int32,
    upper_class_range: Int32,
) raises -> Float64:
    """`double entropy(handle, const int* y, const int n, lower, upper)`."""
    return _raft_entropy(ctx, y, n, lower_class_range, upper_class_range)


def entropy_traced(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut y: DeviceBuffer[DType.int32],
    n: Int,
    lower_class_range: Int32,
    upper_class_range: Int32,
    tag_prefix: String,
) raises -> Float64:
    """The same call carrying a card (`<tag_prefix>.counts`, the device
    histogram, and `<tag_prefix>.acc`, the running host fold). The prefix
    is the caller's: one card computes the entropy of TWO label arrays."""
    return _raft_entropy_traced(
        ctx, trace, y, n, lower_class_range, upper_class_range, tag_prefix
    )

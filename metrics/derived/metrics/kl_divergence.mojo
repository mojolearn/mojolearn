# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""cuML `cpp/src/metrics/kl_divergence.cu` (265b9da): `ML::Metrics::kl_divergence` forwards to `raft::stats::kl_divergence`. The `float` overload; the `double` one cannot run on Apple (no Float64 on device) and is refused by the Python surface (NOT_IMPLEMENTED.tsv)."""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from metrics.derived.stats.detail.kl_divergence import (
    kl_divergence as _raft_kl,
    kl_divergence_traced as _raft_kl_traced,
)


def kl_divergence(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.float32],
    mut y_hat: DeviceBuffer[DType.float32],
    n: Int,
) raises -> Float32:
    """`float kl_divergence(handle, const float* y, const float* y_hat, int n)`."""
    return _raft_kl(ctx, y, y_hat, n)


def kl_divergence_traced(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut y: DeviceBuffer[DType.float32],
    mut y_hat: DeviceBuffer[DType.float32],
    n: Int,
) raises -> Float32:
    """The same call carrying a card (`metrics.kl.partials`,
    `metrics.kl.sum_raw`), the `trustworthiness_score_traced` pattern.
    Returns exactly what `kl_divergence` returns, from one call."""
    return _raft_kl_traced(ctx, trace, y, y_hat, n)

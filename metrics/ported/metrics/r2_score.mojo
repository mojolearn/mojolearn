# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""cuML `cpp/src/metrics/r2_score.cu` (265b9da): `ML::Metrics::r2_score_py` forwards to `raft::stats::r2_score`. The `int` overloads (cuML's Python passes int32 labels). The `float` overload; the `double` one cannot run on Apple (no Float64 on device) and is refused by the Python surface (UNPORTED.tsv)."""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from metrics.mojo_only.pinned_sum import PINNED_SUM_TPB
from metrics.ported.stats.detail.scores import (
    r2_score as _raft_r2,
    r2_score_parts as _raft_r2_parts,
    r2_score_parts_traced as _raft_r2_parts_traced,
)


def r2_score_py(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.float32],
    mut y_hat: DeviceBuffer[DType.float32],
    n: Int,
) raises -> Float32:
    """`float r2_score_py(handle, float* y, float* y_hat, int n)`."""
    return _raft_r2(ctx, y, y_hat, n)


def r2_score_py_parts(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.float32],
    mut y_hat: DeviceBuffer[DType.float32],
    n: Int,
) raises -> Tuple[Float32, Float32, Float32, Float32]:
    """`(y_bar, sse, ssto, r2)` from ONE call at cuML's default launch.

    NOT A SECOND SURFACE: `r2_score_py` is `r2_score_launch[PINNED_SUM_TPB]
    (..., 0)` is `r2_score_parts[PINNED_SUM_TPB](..., 0)[3]`
    (`scores.mojo:258,274`), so element 3 of this tuple IS `r2_score_py`'s
    return, bit for bit, from the same launch -- the caller that wants the
    three sums on its card does not run the metric twice to get them. It
    exists because `r2_score_launch` DISCARDS the three sums and the card
    could then record only the ratio, which `r2_score_parts`'s own docstring
    measured to ABSORB a last-bit move in either sum whenever `sse << ssto`.
    """
    return _raft_r2_parts[PINNED_SUM_TPB](ctx, y, y_hat, n, 0)


def r2_score_py_parts_traced(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut y: DeviceBuffer[DType.float32],
    mut y_hat: DeviceBuffer[DType.float32],
    n: Int,
) raises -> Tuple[Float32, Float32, Float32, Float32]:
    """`r2_score_py_parts` carrying a card: the three chunk-partial buffers
    (`metrics.r2.*_partials`) as well as the four returned values. One
    launch, one set of numbers."""
    return _raft_r2_parts_traced[PINNED_SUM_TPB](ctx, trace, y, y_hat, n, 0)

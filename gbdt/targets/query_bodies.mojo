# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The QueryRMSE target's per-row math, written ONCE (AGENTS.md "One body per
operation", lane gbdt-cpu-rest 2026-11-23).

`query_mean_step` is one lane step of `ComputeGroupMeansImpl`
(`query_helper.cu:132-176`): `sumTarget += t * w` in the default build's
fused form (lane/pinned-mul-contract-free), `w` the row weight (1.0 without
weights, where the fused step is the plain add bit for bit).
`query_rmse_der` and `query_rmse_score` are `query_rmse.cu:12-51`'s two
per-row values, `der = weight * (val - queryMean)` flushed at derivation
(IDENTITY_PATHS row 10) and `-weight * (val - queryMean) * (val -
queryMean)`. Every column calls these bodies: `compute_group_means_kernel`
(`gbdt/gpu_data/kernel/query_helper.mojo`) and `query_rmse_kernel`
(`gbdt/targets/kernel/query_rmse.mojo`) on the device, and
`gbdt/host/gbdt_oracle_query.mojo` on the CPU column.

HOST SAFE: no `max.gpu` / `std.gpu` import and no kernel.
"""

from std.math import fma

from checks.numerics import ftz


@always_inline
def query_mean_step(t: Float32, w: Float32, sum_target: Float32) -> Float32:
    """One lane step of the query mean's target sum: `fma(t, w, sum)`."""
    return fma(t, w, sum_target)


@always_inline
def query_rmse_der(val: Float32, query_mean: Float32, weight: Float32) -> Float32:
    """`der = weight * (val - queryMean)`, flushed at derivation."""
    var direction = val - query_mean
    return ftz(weight * direction)


@always_inline
def query_rmse_score(val: Float32, query_mean: Float32, weight: Float32) -> Float32:
    """`-weight * (val - queryMean) * (val - queryMean)`."""
    return -weight * (val - query_mean) * (val - query_mean)

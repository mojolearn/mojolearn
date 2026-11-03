# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Multi-device resampling: the owner loops and their dispatchers.

`resample/estimator.mojo` keeps only the single-device paths
(`_bootstrap_theta`, `_launch_perm_stat`, `_mc_partials`). Each public entry
there takes a `DevicePlan` value from its binding entry (which reads the
thread-scoped plan once) and calls one dispatcher here per stage. A plan
whose count resolves to one device runs the single-device path unchanged:
the same condition, the same call, the same arguments in the same order as
before the move (docs/plans/MULTI_GPU_ENCAPSULATION_20261103.md).

Owner layout (unchanged by the move): contiguous global replicate ranges
(bootstrap, permutation) or whole `PINNED_SUM_W` chunks (Monte Carlo) run on
owner `rank`'s device, and their bytes are copied into place on the root.
No floating-point value from one owner is ever combined with another's
here; the root folds the gathered result exactly as the single-device path
does.
"""
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.device_plan import DevicePlan
from metrics.checks.pinned_sum import PINNED_SUM_W
from resample.estimator import (
    _bootstrap_theta,
    _launch_perm_stat,
    _mc_partials,
    _upload,
)


def resample_device_count(plan: DevicePlan) raises -> Int:
    """MOJOLEARN_RESAMPLE_DEVICE_COUNT, as the plan carries it: the number of
    owners that replicate or sample ranges run on.

    The same parse and the same refusals the estimator applied to
    `device_plan_setting("MOJOLEARN_RESAMPLE_DEVICE_COUNT")`; "" means one
    device. The caller reads it at the point the estimator always did, so a
    refused setting raises before the same launches as before."""
    if plan.single():
        # "" or "1": one device, the parse below returned 1 for both.
        return 1
    var value = String(plan.setting)
    var count = Int(value)
    if count < 1 or count > 64:
        raise Error("resample device count must be in [1, 64]")
    if count > 1 and GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("multi-GPU resampling requires IDENTICAL numeric mode")
    return count


def _owner_offset(rank: Int) -> Int:
    """Check-only arm: later owners compute their range one position late."""
    comptime if is_defined["MOJOLEARN_RESAMPLE_PARALLEL_SABOTAGE"]():
        if rank > 0:
            return 1
    return 0


def _bootstrap_theta_owners(
    ctx: DeviceContext,
    mut theta: DeviceBuffer[DType.float32],
    x: List[Float32],
    key: UInt64,
    r_first: Int,
    n_resamples: Int,
    n: Int,
    n_features: Int,
    statistic: Int,
    q_or_prop: Float32,
    tpb: Int,
    map_tpb: Int,
    plan: DevicePlan,
    count: Int,
) raises:
    """Contiguous global replicate ranges on owners, bytes copied into place."""
    var active = min(count, n_resamples)
    ctx.synchronize()
    for rank in range(active):
        var begin = n_resamples * rank // active
        var rows = n_resamples * (rank + 1) // active - begin
        var owner = DeviceContext(device_id=plan.device(rank))
        var ox = _upload(owner, x)
        var ot = owner.enqueue_create_buffer[DType.float32](rows)
        owner.synchronize()
        _bootstrap_theta(
            owner, ot, ox, key, r_first + begin + _owner_offset(rank), rows,
            n, n_features, statistic, q_or_prop, tpb, map_tpb,
        )
        owner.synchronize()
        var view = theta.create_sub_buffer[DType.float32](begin, rows)
        ot.enqueue_copy_to(view)
        owner.synchronize()
        ctx.synchronize()
        _ = view^
        _ = ox^
        _ = ot^
        _ = owner^
    ctx.synchronize()


def _perm_null_owners(
    ctx: DeviceContext,
    mut null_buf: DeviceBuffer[DType.float32],
    pooled: List[Float32],
    key: UInt64,
    r_first: Int,
    n_resamples: Int,
    n_pooled: Int,
    n_x: Int,
    statistic: Int,
    tpb: Int,
    plan: DevicePlan,
    count: Int,
) raises:
    """Contiguous global permutation ranges on owners, bytes copied into place."""
    var active = min(count, n_resamples)
    ctx.synchronize()
    for rank in range(active):
        var begin = n_resamples * rank // active
        var rows = n_resamples * (rank + 1) // active - begin
        var owner = DeviceContext(device_id=plan.device(rank))
        var op = _upload(owner, pooled)
        var on = owner.enqueue_create_buffer[DType.float32](rows)
        owner.synchronize()
        _launch_perm_stat(
            owner, on, op, key, r_first + begin + _owner_offset(rank), rows,
            n_pooled, n_x, statistic, tpb,
        )
        owner.synchronize()
        var view = null_buf.create_sub_buffer[DType.float32](begin, rows)
        on.enqueue_copy_to(view)
        owner.synchronize()
        ctx.synchronize()
        _ = view^
        _ = op^
        _ = on^
        _ = owner^
    ctx.synchronize()


def _mc_partials_owners[
    f_id: Int
](
    ctx: DeviceContext,
    mut partials: DeviceBuffer[DType.float32],
    lower: List[Float32],
    span: List[Float32],
    key: UInt64,
    i_first: Int,
    n_samples: Int,
    n_chunks: Int,
    tpb: Int,
    plan: DevicePlan,
    count: Int,
) raises:
    """Whole PINNED_SUM_W chunks on owners. An owner starting at global chunk
    c0 draws from `i_first + c0 * PINNED_SUM_W`, so its chunk j is global
    chunk c0 + j with the same values and the same pinned fold; the root folds
    all partials in chunk order."""
    var active = min(count, n_chunks)
    ctx.synchronize()
    for rank in range(active):
        var c0 = n_chunks * rank // active
        var c1 = n_chunks * (rank + 1) // active
        var first_sample = c0 * PINNED_SUM_W
        var last_sample = c1 * PINNED_SUM_W
        if last_sample > n_samples:
            last_sample = n_samples
        var chunks = c1 - c0
        var owner = DeviceContext(device_id=plan.device(rank))
        var ol = _upload(owner, lower)
        var ospan = _upload(owner, span)
        var opart = owner.enqueue_create_buffer[DType.float32](chunks)
        owner.synchronize()
        _mc_partials[f_id](
            owner, opart, ol, ospan, key,
            i_first + first_sample + _owner_offset(rank),
            last_sample - first_sample, chunks, chunks, tpb,
        )
        owner.synchronize()
        var view = partials.create_sub_buffer[DType.float32](c0, chunks)
        opart.enqueue_copy_to(view)
        owner.synchronize()
        ctx.synchronize()
        _ = view^
        _ = ol^
        _ = ospan^
        _ = opart^
        _ = owner^
    ctx.synchronize()


# ===========================================================================
# DISPATCHERS: one per multi-device stage. `owners` is
# `resample_device_count(plan)`, read by the estimator where it always was.
# One owner (or one replicate / one chunk) runs the single-device path with
# the same arguments it was always called with.
# ===========================================================================


def bootstrap_theta_dispatch(
    plan: DevicePlan,
    owners: Int,
    ctx: DeviceContext,
    mut theta: DeviceBuffer[DType.float32],
    mut dx: DeviceBuffer[DType.float32],
    x: List[Float32],
    key: UInt64,
    r_first: Int,
    n_resamples: Int,
    n: Int,
    n_features: Int,
    statistic: Int,
    q_or_prop: Float32,
    tpb: Int,
    map_tpb: Int,
) raises:
    """The bootstrap replicates into `theta`: `_bootstrap_theta` on `ctx`,
    or contiguous replicate ranges on the plan's owners."""
    if owners > 1 and n_resamples > 1:
        _bootstrap_theta_owners(
            ctx, theta, x, key, r_first, n_resamples, n, n_features,
            statistic, q_or_prop, tpb, map_tpb, plan, owners,
        )
    else:
        _bootstrap_theta(
            ctx, theta, dx, key, r_first, n_resamples, n, n_features,
            statistic, q_or_prop, tpb, map_tpb,
        )


def perm_null_dispatch(
    plan: DevicePlan,
    owners: Int,
    ctx: DeviceContext,
    mut null_buf: DeviceBuffer[DType.float32],
    mut dpool: DeviceBuffer[DType.float32],
    pooled: List[Float32],
    key: UInt64,
    r_first: Int,
    n_resamples: Int,
    n_pooled: Int,
    n_x: Int,
    statistic: Int,
    tpb: Int,
) raises:
    """The permutation null into `null_buf`: `_launch_perm_stat` on `ctx`,
    or contiguous permutation ranges on the plan's owners. (The FAST select
    runs before this in the estimator and, when it takes the null, this is
    not called; owners > 1 is IDENTICAL only.)"""
    if owners > 1 and n_resamples > 1:
        _perm_null_owners(
            ctx, null_buf, pooled, key, r_first, n_resamples, n_pooled, n_x,
            statistic, tpb, plan, owners,
        )
    else:
        _launch_perm_stat(
            ctx,
            null_buf,
            dpool,
            key,
            r_first,
            n_resamples,
            n_pooled,
            n_x,
            statistic,
            tpb,
        )


def mc_partials_dispatch[
    f_id: Int
](
    plan: DevicePlan,
    owners: Int,
    ctx: DeviceContext,
    mut partials: DeviceBuffer[DType.float32],
    mut dlower: DeviceBuffer[DType.float32],
    mut dspan: DeviceBuffer[DType.float32],
    lower: List[Float32],
    span: List[Float32],
    key: UInt64,
    i_first: Int,
    n_samples: Int,
    n_chunks: Int,
    blocks: Int,
    tpb: Int,
) raises:
    """The Monte Carlo chunk partials: `_mc_partials` on `ctx`, or whole
    `PINNED_SUM_W` chunks on the plan's owners."""
    if owners > 1 and n_chunks > 1:
        _mc_partials_owners[f_id](
            ctx, partials, lower, span, key, i_first, n_samples, n_chunks, tpb,
            plan, owners,
        )
    else:
        _mc_partials[f_id](
            ctx, partials, dlower, dspan, key, i_first, n_samples, n_chunks,
            blocks, tpb,
        )

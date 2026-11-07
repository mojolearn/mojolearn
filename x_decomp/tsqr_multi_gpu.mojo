# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The blocked TSQR of `x_decomp/tsqr_device.mojo` over a `DevicePlan`
(lane ne2-i-j, 2026-11-07; docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md
section 2J, docs/plans/MULTI_GPU_ENCAPSULATION_20261103.md).

WHY. `LinearRegression` in IDENTICAL fits by the centered TSQR
(`DevExec.ols_tsqr_factor`), which read no plan: under
`fit_gram_estimator(..., devices=...)` every row block ran on one device.

WHAT. The factor is three phases (`ts_factor_begin`, `ts_leaf_phase`,
`ts_factor_finish`). The leaf phase factors each row block of `TS_ROWS` rows
(the last block takes the remainder) on its own: a block's launches read and
write only its rows of the matrix and its own T-factor slots, and every
kernel takes the global block count and the global block index. Here whole
blocks go to the plan's devices (rank r: blocks `[B r / a, B (r + 1) / a)` of
`B`, `a = min(count, B)` owners). Each owner holds a full copy of the packed
matrix (so the kernels' row and slot addresses are the one-device ones), runs
`ts_leaf_phase` over its blocks, and its blocks' rows and T slots come back to
the root, which runs `ts_factor_finish` (the R tiles and the tree combine) as
one device does. Every block's words are the one-device words: NO BIT MOVES at
any device count. One device (`plan.single()`) calls `ts_factor_device`
unchanged.

THE NEGATIVE CONTROL. `-D MOJOLEARN_QR_PARALLEL_SABOTAGE=1` (the QR family's
define, `core/qr_multi_gpu.mojo`) makes every owner above rank 0 copy its rows
back one row late; offsets and lengths otherwise keep the true block range. A
`comptime if`, inert at one device.
"""
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.device_plan import DevicePlan, device_plan_current
from core.multi_gpu import peer_clone
from x_decomp.cells import F32Ptr
from x_decomp.tsqr_core import TS_NB, TS_T2, TS_WY_PAIR, ts_block_hi, ts_block_lo, ts_blocks, ts_pairs, ts_panels
from x_decomp.tsqr_device import (
    TsFactorWork,
    ts_factor_begin,
    ts_factor_device,
    ts_factor_finish,
    ts_factor_work,
    ts_leaf_phase,
)


def tsqr_plan_current() raises -> DevicePlan:
    """The QR family's plan (`MOJOLEARN_QR_DEVICE_COUNT`), read once at the
    x_decomp device entry that factors (`DevExec.ols_tsqr_factor`)."""
    return device_plan_current("MOJOLEARN_QR_DEVICE_COUNT")


struct _TsShard(Movable):
    var ctx: DeviceContext
    var a: DeviceBuffer[DType.float32]
    var work: TsFactorWork
    var blo: Int
    var bhi: Int

    def __init__(out self, var ctx: DeviceContext, var a: DeviceBuffer[DType.float32], var work: TsFactorWork,
                 blo: Int, bhi: Int):
        self.ctx = ctx^
        self.a = a^
        self.work = work^
        self.blo = blo
        self.bhi = bhi

    def __deinit__(deinit self):
        _ = self.a^
        _ = self.work^
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^


def ts_factor_dispatch(
    plan: DevicePlan, ctx: DeviceContext, mut da: DeviceBuffer[DType.float32], m: Int, n: Int, r: F32Ptr, keep: Bool,
) raises:
    """`ts_factor_device(ctx, da, m, n, r, keep)` over `plan` (module
    docstring): the same R in `r`, the same factored `da` and, with `keep`,
    the same kept state."""
    var nb = ts_blocks(m)
    if plan.single() or nb < 2:
        # one device, or one block: the one-device factor, unchanged
        ts_factor_device(ctx, da, m, n, r, keep)
        return
    var count = Int(plan.setting)
    if count < 1 or count > 64 or GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("parallel TSQR requires IDENTICAL and 1..64 devices")
    var active = min(count, nb)
    var npan = ts_panels(n)
    var npair = ts_pairs(n)
    var t_words = npan * TS_NB * TS_NB
    var t2_words = npair * TS_T2
    var work = ts_factor_begin(ctx, m, n)
    ctx.synchronize()
    var shards = List[_TsShard]()
    for rank in range(active):  # driver-loop: over the plan's device shards
        var blo = nb * rank // active
        var bhi = nb * (rank + 1) // active
        var device = DeviceContext(device_id=plan.device(rank))
        var local_a = peer_clone(ctx, device, da)
        var local_work = ts_factor_work(device, m, n)
        device.synchronize()
        shards.append(_TsShard(device^, local_a^, local_work^, blo, bhi))
    for rank in range(active):  # driver-loop: over the plan's device shards
        ref shard = shards[rank]
        ts_leaf_phase(shard.ctx, shard.a, shard.work, m, n, shard.blo, shard.bhi)
    for rank in range(active):  # driver-loop: over the plan's device shards
        ref shard = shards[rank]
        var lo = ts_block_lo(shard.blo)
        var hi = ts_block_hi(shard.bhi - 1, nb, m)
        var source = lo
        comptime if is_defined["MOJOLEARN_QR_PARALLEL_SABOTAGE"]():
            # Check-only arm: later owners hand back their rows one row late.
            if rank > 0:
                source = lo + 1
        var rows = min(hi, m - (source - lo)) - lo
        var src_a = shard.a.create_sub_buffer[DType.float32](source * n, rows * n)
        var dst_a = da.create_sub_buffer[DType.float32](lo * n, rows * n)
        src_a.enqueue_copy_to(dst_a)
        var src_t = shard.work.dt.create_sub_buffer[DType.float32](shard.blo * t_words, (shard.bhi - shard.blo) * t_words)
        var dst_t = work.dt.create_sub_buffer[DType.float32](shard.blo * t_words, (shard.bhi - shard.blo) * t_words)
        src_t.enqueue_copy_to(dst_t)
        comptime if TS_WY_PAIR:
            var src_t2 = shard.work.dt2.create_sub_buffer[DType.float32](
                shard.blo * t2_words, (shard.bhi - shard.blo) * t2_words)
            var dst_t2 = work.dt2.create_sub_buffer[DType.float32](
                shard.blo * t2_words, (shard.bhi - shard.blo) * t2_words)
            src_t2.enqueue_copy_to(dst_t2)
        shard.ctx.synchronize()
    _ = shards^
    ctx.synchronize()
    ts_factor_finish(ctx, da, work, m, n, r, keep)
    _ = work^

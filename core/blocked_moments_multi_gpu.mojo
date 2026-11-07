# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The centered Gram and cross moments of `core/blocked_moments.mojo` over a
`DevicePlan` (lane ne2-i-j, 2026-11-07; docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md
section 2J, docs/plans/MULTI_GPU_ENCAPSULATION_20261103.md).

WHY. `LinearRegression` and `Ridge` in IDENTICAL fit by the resident centered
Gram (`glm/impl/gram_solve.mojo` `linear_gram_fit_host`) before any other
route, and that fit read no plan: `fit_gram_estimator(..., devices=...)`
bound the Gram family's plan and then ran every row on one device.

WHAT. `bm_centered_gram_panels` and `bm_centered_cross_panels` cut the rows
into leaves of `leaf` rows, fill one partial per (leaf, cell) with the leaf
kernels, then fold the partials over the leaves in the binary-counter tree
(`bm_sum_fold`). A leaf's partial reads only that leaf's rows. Here whole
leaves go to the plan's devices (rank r: leaves `[L r / a, L (r + 1) / a)` of
`L`, `a = min(count, L)` owners), each device runs the SAME leaf kernels over
its rows (a shard's first row is a leaf boundary, so its local leaf `k` is
global leaf `first + k` with the same rows), the partials come back to the
root at their global leaf offsets, and the root runs the same `bm_sum_fold`.
Every partial and the fold are the one-device words: NO BIT MOVES at any
device count. One device (`plan.single()`) calls the two one-device launchers
unchanged.

THE NEGATIVE CONTROL. `-D MOJOLEARN_GRAM_PARALLEL_SABOTAGE=1` (the Gram
family's define, `core/gram_multi_gpu.mojo`) makes every owner above rank 0
read its rows one row early; lengths and offsets keep the true first leaf. A
`comptime if`, inert at one device.
"""
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.device_plan import DevicePlan
from core.multi_gpu import peer_clone
from core.blocked_moments import (
    BmPtr,
    bm_centered_cross_panels,
    bm_centered_gram_panels,
    bm_leaf_cross_kernel,
    bm_leaf_gram_kernel,
    bm_sum_fold,
)
from core.blocked_moments_ops import BM_TILE, BM_TPB, bm_leaf_count


@always_inline
def _bp(buf: DeviceBuffer[DType.float32]) -> BmPtr:
    # the address form `core/blocked_moments.mojo` passes to its kernels
    return BmPtr(unsafe_from_address=Int(buf.unsafe_ptr()))


struct _MomentShard(Movable):
    var ctx: DeviceContext
    var x: DeviceBuffer[DType.float32]
    var y: DeviceBuffer[DType.float32]
    var cx: DeviceBuffer[DType.float32]
    var cy: DeviceBuffer[DType.float32]
    var gram: DeviceBuffer[DType.float32]
    var cross: DeviceBuffer[DType.float32]
    var first: Int
    var leaves: Int
    var rows: Int

    def __init__(
        out self, var ctx: DeviceContext, var x: DeviceBuffer[DType.float32], var y: DeviceBuffer[DType.float32],
        var cx: DeviceBuffer[DType.float32], var cy: DeviceBuffer[DType.float32],
        var gram: DeviceBuffer[DType.float32], var cross: DeviceBuffer[DType.float32],
        first: Int, leaves: Int, rows: Int,
    ):
        self.ctx = ctx^
        self.x = x^
        self.y = y^
        self.cx = cx^
        self.cy = cy^
        self.gram = gram^
        self.cross = cross^
        self.first = first
        self.leaves = leaves
        self.rows = rows

    def __deinit__(deinit self):
        _ = self.x^
        _ = self.y^
        _ = self.cx^
        _ = self.cy^
        _ = self.gram^
        _ = self.cross^
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^


def bm_centered_gram_cross_dispatch(
    plan: DevicePlan,
    ctx: DeviceContext,
    mut g: DeviceBuffer[DType.float32],
    mut c: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut cx: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    mut cy: DeviceBuffer[DType.float32],
    n: Int,
    d: Int,
    leaf: Int,
) raises:
    """`g` (d x d) = `bm_centered_gram_panels(x, cx)` and `c` (d) =
    `bm_centered_cross_panels(x, cx, y, cy)` with leaves of `leaf` rows, over
    `plan` (module docstring). `x` is row-major n x d, `cx` d floats, `y` n
    floats, `cy` one float, all on `ctx`. Waits for the device."""
    var leaves = bm_leaf_count(n, leaf)
    if plan.single() or d < 1 or leaves < 2:
        # one device, or nothing to split: the one-device launchers, unchanged
        bm_centered_gram_panels(ctx, _bp(g), _bp(x), _bp(cx), n, d, leaf)
        bm_centered_cross_panels(ctx, _bp(c), _bp(x), _bp(cx), _bp(y), _bp(cy), n, d, leaf)
        return
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        # FAST keeps the one-device launchers it ran before the plan reached it
        bm_centered_gram_panels(ctx, _bp(g), _bp(x), _bp(cx), n, d, leaf)
        bm_centered_cross_panels(ctx, _bp(c), _bp(x), _bp(cx), _bp(y), _bp(cy), n, d, leaf)
        return
    var count = Int(plan.setting)
    if count < 1 or count > 64:
        raise Error("parallel Gram requires 1..64 devices")
    var active = min(count, leaves)
    var cells = d * d
    var tiles = (d + BM_TILE - 1) // BM_TILE
    ctx.synchronize()
    var shards = List[_MomentShard]()
    for rank in range(active):  # driver-loop: over the plan's device shards
        var first = leaves * rank // active
        var end = leaves * (rank + 1) // active
        var begin_row = first * leaf
        var rows = min(n, end * leaf) - begin_row
        var source_row = begin_row
        comptime if is_defined["MOJOLEARN_GRAM_PARALLEL_SABOTAGE"]():
            # Check-only arm: later owners read their rows one row early.
            if rank > 0:
                source_row = begin_row - 1
        var device = DeviceContext(device_id=plan.device(rank))
        var x_view = x.create_sub_buffer[DType.float32](source_row * d, rows * d)
        var y_view = y.create_sub_buffer[DType.float32](source_row, rows)
        var local_x = peer_clone(ctx, device, x_view)
        var local_y = peer_clone(ctx, device, y_view)
        var local_cx = peer_clone(ctx, device, cx)
        var local_cy = peer_clone(ctx, device, cy)
        var part_g = device.enqueue_create_buffer[DType.float32]((end - first) * cells)
        var part_c = device.enqueue_create_buffer[DType.float32]((end - first) * d)
        device.synchronize()
        shards.append(_MomentShard(device^, local_x^, local_y^, local_cx^, local_cy^, part_g^, part_c^,
                                   first, end - first, rows))
    for rank in range(active):  # driver-loop: over the plan's device shards
        ref shard = shards[rank]
        # the one-device launches of `bm_centered_gram_panels` and
        # `bm_centered_cross_panels` over this shard's rows
        shard.ctx.enqueue_function[bm_leaf_gram_kernel](
            _bp(shard.gram), _bp(shard.gram), _bp(shard.x), _bp(shard.cx),
            Int32(shard.rows), Int32(d), Int32(leaf), Int32(tiles), Int32(0),
            grid_dim=(shard.leaves, tiles * (tiles + 1) // 2, 1), block_dim=BM_TPB,
        )
        shard.ctx.enqueue_function[bm_leaf_cross_kernel](
            _bp(shard.cross), _bp(shard.x), _bp(shard.cx), _bp(shard.y), _bp(shard.cy),
            Int32(shard.rows), Int32(d), Int32(leaf), grid_dim=shard.leaves, block_dim=BM_TPB,
        )
    var part_g = ctx.enqueue_create_buffer[DType.float32](leaves * cells)
    var part_c = ctx.enqueue_create_buffer[DType.float32](leaves * d)
    ctx.synchronize()
    for rank in range(active):  # driver-loop: over the plan's device shards
        ref shard = shards[rank]
        var dst_g = part_g.create_sub_buffer[DType.float32](shard.first * cells, shard.leaves * cells)
        var dst_c = part_c.create_sub_buffer[DType.float32](shard.first * d, shard.leaves * d)
        shard.gram.enqueue_copy_to(dst_g)
        shard.cross.enqueue_copy_to(dst_c)
        shard.ctx.synchronize()
    _ = shards^
    # the one-device folds, in the one-device order (Gram, then cross)
    bm_sum_fold(ctx, part_g, leaves, cells, _bp(g), 0)
    bm_sum_fold(ctx, part_c, leaves, d, _bp(c), 0)
    _ = part_g^
    _ = part_c^

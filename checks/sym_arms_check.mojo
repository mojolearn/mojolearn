# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATIONS 2580 and 2581, BOTH SIDES OF EACH SWITCH (ENGINEERING_RULES 8).

`launch_histograms_for_blocks` is instantiated here with the switches set
EXPLICITLY, so this one binary exercises every side whatever the build
defines say:

    check_2580_off   level_quant False: one-byte cells against a host tally
    check_2580_on    level_quant True: bit-equal to the off side, and the
                     float plane POISONED after the level quantize still
                     gives the same bits (the kernels read the Int32 plane;
                     the same poison moves the off side, so it has teeth)
    check_2581_off   group_width False: bit-equal to a second off run
    check_2581_on    group_width True: bit-equal to the off side, and a
                     SWAPPED column map moves the bits (the map is read)
    check_2580_2581_on   both True: bit-equal to the off side

at depth 0 (direct loads, one leaf) and depth 1 (gather through a permuted
row index, three non-contiguous leaves of four). The fixture's one-byte
groups hit all four widths, two 5-bit groups are not adjacent, and the last
group is short. Bins are SCATTERED (hashed per row and feature), and the
stats are non-integer, so the dither decides cells.

Only one-byte cells are compared: on a FAST float column the binary and
half-byte cells come from float atomics and are not run-to-run exact.

    pixi run check-sym-arms                       (FAST column)
    pixi run check-sym-arms-identical             (IDENTICAL)
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from checks.fixed_point import choose_scale
from checks.numerics import numeric_mode_name
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.feature_blocks import blocks_for
from gbdt.gpu_data.grid_policy import POLICY_ONE_BYTE
from gbdt.gpu_data.kernel.binarize import (
    WRITE_BLOCK_SIZE,
    write_compressed_index_kernel,
)
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import (
    DeviceBlock,
    OneByteWidthPlan,
    enqueue_level_quantize,
    launch_histograms_for_blocks,
    sym_arms_path_line,
    upload_blocks,
    upload_scale,
    upload_width_plans,
)
from gbdt.methods.greedy_subsets_searcher.kernel.hist_2_one_byte_base import (
    HIST2_SMEM_MODE,
)

comptime N_ROWS = 70001
comptime STAT_COUNT = 2
comptime MAX_LEAVES = 4
comptime SM_COUNT = 32


def mix(a: Int, b: Int) -> UInt32:
    var x = UInt32(a * 2654435761 + b * 40503 + 0x2545F491)
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    return x


def append4(mut f: List[Int], a: Int, b: Int, c: Int, d: Int):
    f.append(a)
    f.append(b)
    f.append(c)
    f.append(d)


def fixture_folds() -> List[Int]:
    var f = List[Int]()
    # binary and half-byte first, so the one-byte block's column base is not 0
    f.append(1)
    f.append(1)
    f.append(1)
    f.append(8)
    f.append(12)
    append4(f, 20, 30, 25, 31)      # 5 bits
    append4(f, 40, 64, 50, 33)      # 6 bits
    append4(f, 100, 128, 70, 90)    # 7 bits
    append4(f, 254, 200, 20, 129)   # 8 bits
    append4(f, 16, 17, 32, 18)      # 5 bits again, not adjacent
    f.append(60)                    # a short last group, 7 bits
    f.append(120)
    return f^


def run_arm[level_quant: Bool, group_width: Bool](
    ctx: DeviceContext,
    mut dblocks: List[DeviceBlock],
    plans: List[OneByteWidthPlan],
    depth: Int,
    n_live: Int,
    quantize: Bool,
    mut cindex: DeviceBuffer[DType.uint32],
    mut row_index: DeviceBuffer[DType.uint32],
    mut stats: DeviceBuffer[DType.float32],
    mut qstats: DeviceBuffer[DType.int32],
    mut p_off: DeviceBuffer[DType.uint32],
    mut p_sz: DeviceBuffer[DType.uint32],
    mut ids: DeviceBuffer[DType.uint32],
    mut dense_ids: DeviceBuffer[DType.uint32],
    mut hist: DeviceBuffer[DType.float32],
    mut acc: DeviceBuffer[DType.int32],
    mut block_hist: DeviceBuffer[DType.float32],
    hist_cells_per_leaf: Int,
    scale: Float32,
) raises -> HostBuffer[DType.float32]:
    """One launch with the switches as given. `quantize` False skips the
    level quantize so a poisoned float plane cannot reach the Int32 one."""
    var total = MAX_LEAVES * STAT_COUNT * hist_cells_per_leaf
    var zf = ctx.enqueue_create_host_buffer[DType.float32](total)
    var zi = ctx.enqueue_create_host_buffer[DType.int32](total)
    for i in range(total):
        zf.unsafe_ptr().unsafe_store(i, Float32(0.0))
        zi.unsafe_ptr().unsafe_store(i, Int32(0))
    ctx.enqueue_copy(dst_buf=hist, src_ptr=zf.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=acc, src_ptr=zi.unsafe_ptr())
    ctx.synchronize()
    var scale_keep = upload_scale(ctx, scale)
    var scale_ptr = rebind[MutPointer[Float32, MutAnyOrigin]](
        scale_keep.unsafe_ptr()
    )
    if quantize:
        enqueue_level_quantize(
            ctx, n_live, N_ROWS, STAT_COUNT, SM_COUNT, stats, qstats,
            p_off, p_sz, ids, scale_ptr,
        )
    launch_histograms_for_blocks[
        HIST2_SMEM_MODE, False, level_quant, group_width
    ](
        ctx, dblocks, depth, n_live, N_ROWS, STAT_COUNT, MAX_LEAVES,
        SM_COUNT, scale_ptr, cindex, row_index, stats, p_off, p_sz, ids,
        dense_ids, hist, acc, block_hist, hist_cells_per_leaf,
        qstats=Optional(qstats.copy()), width_plans=plans,
    )
    ctx.synchronize()
    var out = ctx.enqueue_create_host_buffer[DType.float32](total)
    ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=hist)
    ctx.synchronize()
    _ = scale_keep^
    _ = zf^
    _ = zi^
    return out^


def count_diff(
    a: HostBuffer[DType.float32],
    b: HostBuffer[DType.float32],
    want: List[Int],
    ob_first: Int,
    ob_len: Int,
    hist_cells_per_leaf: Int,
) -> Int:
    var diff = 0
    for k in range(len(want)):
        var leaf = want[k]
        for s in range(STAT_COUNT):
            var base = (leaf * STAT_COUNT + s) * hist_cells_per_leaf + ob_first
            for c in range(ob_len):
                if a.unsafe_ptr().unsafe_load(base + c) != b.unsafe_ptr().unsafe_load(
                    base + c
                ):
                    diff += 1
    return diff


def upload_plane(
    ctx: DeviceContext, mut dst: DeviceBuffer[DType.float32], src: List[Float32]
) raises:
    var h = ctx.enqueue_create_host_buffer[DType.float32](len(src))
    for i in range(len(src)):
        h.unsafe_ptr().unsafe_store(i, src[i])
    ctx.enqueue_copy(dst_buf=dst, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h^


def check_depth(depth: Int) raises -> Int:
    """Every side at one depth. Returns the number of failed sides."""
    var ctx = DeviceContext()
    var folds = fixture_folds()
    var n_features = len(folds)
    var lay = build_layout(folds)
    var blocks = blocks_for(lay, N_ROWS)
    var dblocks = upload_blocks(ctx, blocks)
    var plans = upload_width_plans(ctx, blocks)
    print("  depth", depth, "mode", numeric_mode_name())
    print("  ", sym_arms_path_line(blocks))

    # the one-byte block's flat cells, contiguous by `blocks_for`'s order
    var ob_first = lay.hist_cells
    var ob_len = 0
    for f in range(n_features):
        if folds[f] > 15:
            if Int(lay.features[f].first_fold_index) < ob_first:
                ob_first = Int(lay.features[f].first_fold_index)
            ob_len += folds[f]

    var cindex = ctx.enqueue_create_buffer[DType.uint32](N_ROWS * lay.columns)
    var z = ctx.enqueue_create_host_buffer[DType.uint32](N_ROWS * lay.columns)
    for i in range(N_ROWS * lay.columns):
        z.unsafe_ptr().unsafe_store(i, UInt32(0))
    ctx.enqueue_copy(dst_buf=cindex, src_ptr=z.unsafe_ptr())
    ctx.synchronize()
    var host_bin = List[List[Int]]()
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](N_ROWS)
    var bins = ctx.enqueue_create_buffer[DType.uint8](N_ROWS)
    for f in range(n_features):
        var col = List[Int]()
        ref cf = lay.features[f]
        for r in range(N_ROWS):
            var v = Int(mix(r, f) % UInt32(folds[f] + 1))
            col.append(v)
            hb.unsafe_ptr().unsafe_store(r, UInt8(v))
        host_bin.append(col^)
        ctx.enqueue_copy(dst_buf=bins, src_ptr=hb.unsafe_ptr())
        ctx.enqueue_function[write_compressed_index_kernel](
            Int32(Int(cf.offset) * N_ROWS), cf.mask, cf.shift,
            bins.unsafe_ptr(), Int32(N_ROWS), cindex.unsafe_ptr(),
            grid_dim=(N_ROWS + WRITE_BLOCK_SIZE - 1) // WRITE_BLOCK_SIZE,
            block_dim=WRITE_BLOCK_SIZE,
        )
        ctx.synchronize()

    # non-integer stats; magnitudes bound the scale as `fit` bounds it
    var plane = List[Float32]()
    var mag_w = Float64(0.0)
    var mag_g = Float64(0.0)
    for r in range(N_ROWS):
        var w = Float32(0.5) + Float32(Int(mix(r, 977) % UInt32(1000))) / Float32(1000.0)
        plane.append(w)
        mag_w += Float64(w)
    for r in range(N_ROWS):
        var g = Float32(Int(mix(r, 4099) % UInt32(2001)) - 1000) / Float32(997.0)
        plane.append(g)
        mag_g += Float64(abs(g))
    var mag = mag_w if mag_w > mag_g else mag_g
    var scale = Float32(choose_scale(mag, N_ROWS))
    var stats = ctx.enqueue_create_buffer[DType.float32](STAT_COUNT * N_ROWS)
    upload_plane(ctx, stats, plane)
    var qstats = ctx.enqueue_create_buffer[DType.int32](STAT_COUNT * N_ROWS)

    # depth 0: the identity index and one leaf; depth 1: a permutation
    var rows = List[Int]()
    for r in range(N_ROWS):
        rows.append(r)
    if depth > 0:
        for r in range(N_ROWS - 1, 0, -1):
            var j = Int(mix(r, 31337) % UInt32(r + 1))
            var t = rows[r]
            rows[r] = rows[j]
            rows[j] = t
    var row_index = ctx.enqueue_create_buffer[DType.uint32](N_ROWS)
    var hi = ctx.enqueue_create_host_buffer[DType.uint32](N_ROWS)
    for r in range(N_ROWS):
        hi.unsafe_ptr().unsafe_store(r, UInt32(rows[r]))
    ctx.enqueue_copy(dst_buf=row_index, src_ptr=hi.unsafe_ptr())

    var off = List[Int]()
    var siz = List[Int]()
    var want = List[Int]()
    if depth == 0:
        off.append(0)
        siz.append(N_ROWS)
        for _ in range(MAX_LEAVES - 1):
            off.append(0)
            siz.append(0)
        want.append(0)
    else:
        off.append(0)
        siz.append(9001)
        off.append(9001)
        siz.append(21000)
        off.append(30001)
        siz.append(10999)
        off.append(41000)
        siz.append(29001)
        want.append(2)
        want.append(0)
        want.append(3)
    var n_live = len(want)
    var p_off = ctx.enqueue_create_buffer[DType.uint32](MAX_LEAVES)
    var p_sz = ctx.enqueue_create_buffer[DType.uint32](MAX_LEAVES)
    var ids = ctx.enqueue_create_buffer[DType.uint32](MAX_LEAVES)
    var dense_ids = ctx.enqueue_create_buffer[DType.uint32](MAX_LEAVES)
    var ho = ctx.enqueue_create_host_buffer[DType.uint32](MAX_LEAVES)
    var hs = ctx.enqueue_create_host_buffer[DType.uint32](MAX_LEAVES)
    var hid = ctx.enqueue_create_host_buffer[DType.uint32](MAX_LEAVES)
    var hd = ctx.enqueue_create_host_buffer[DType.uint32](MAX_LEAVES)
    for i in range(MAX_LEAVES):
        ho.unsafe_ptr().unsafe_store(i, UInt32(off[i]))
        hs.unsafe_ptr().unsafe_store(i, UInt32(siz[i]))
        hid.unsafe_ptr().unsafe_store(i, UInt32(0))
        hd.unsafe_ptr().unsafe_store(i, UInt32(i))
    for i in range(n_live):
        hid.unsafe_ptr().unsafe_store(i, UInt32(want[i]))
    ctx.enqueue_copy(dst_buf=p_off, src_ptr=ho.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=p_sz, src_ptr=hs.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=ids, src_ptr=hid.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dense_ids, src_ptr=hd.unsafe_ptr())

    var hcl = lay.hist_cells
    var total = MAX_LEAVES * STAT_COUNT * hcl
    var hist = ctx.enqueue_create_buffer[DType.float32](total)
    var acc = ctx.enqueue_create_buffer[DType.int32](total)
    var widest = 1
    for b in range(len(blocks)):
        var tf = 0
        for k in range(blocks[b].count()):
            tf += Int(blocks[b].folds[k])
        if tf > widest:
            widest = tf
    var block_hist = ctx.enqueue_create_buffer[DType.float32](
        MAX_LEAVES * STAT_COUNT * widest
    )
    ctx.synchronize()

    var failed = 0

    # ---- check_2580_off: the off side against a host tally -------------
    var off_side = run_arm[False, False](
        ctx, dblocks, plans, depth, n_live, False, cindex, row_index, stats,
        qstats, p_off, p_sz, ids, dense_ids, hist, acc, block_hist, hcl,
        scale,
    )
    var wrong = 0
    var nonzero = 0
    for k in range(n_live):
        var leaf = want[k]
        var expect = List[Float64]()
        var count = List[Int]()
        for _ in range(STAT_COUNT * hcl):
            expect.append(0.0)
            count.append(0)
        for pos in range(off[leaf], off[leaf] + siz[leaf]):
            var row = rows[pos]
            for f in range(n_features):
                if folds[f] <= 15:
                    continue
                var bin = host_bin[f][row]
                if bin < folds[f]:
                    var cell = Int(lay.features[f].first_fold_index) + bin
                    for s in range(STAT_COUNT):
                        expect[s * hcl + cell] += Float64(plane[s * N_ROWS + pos])
                        count[s * hcl + cell] += 1
        for s in range(STAT_COUNT):
            for c in range(ob_first, ob_first + ob_len):
                var got = Float64(
                    off_side.unsafe_ptr().unsafe_load(
                        (leaf * STAT_COUNT + s) * hcl + c
                    )
                )
                var e = expect[s * hcl + c]
                var tol = Float64(count[s * hcl + c] + 1) / Float64(scale) + 1e-4 * abs(e)
                if abs(got - e) > tol:
                    wrong += 1
                if count[s * hcl + c] > 0:
                    nonzero += 1
    print("    check_2580_off: one-byte cells off the host tally:", wrong,
          "of", n_live * STAT_COUNT * ob_len, "(", nonzero, "populated )")
    if wrong != 0 or nonzero < n_live * ob_len:
        print("    check_2580_off FAILED")
        failed += 1

    # ---- check_2581_off: the off side is exact run to run --------------
    var off_again = run_arm[False, False](
        ctx, dblocks, plans, depth, n_live, False, cindex, row_index, stats,
        qstats, p_off, p_sz, ids, dense_ids, hist, acc, block_hist, hcl,
        scale,
    )
    var d_off = count_diff(off_side, off_again, want, ob_first, ob_len, hcl)
    print("    check_2581_off: second off run differs in", d_off, "cells")
    if d_off != 0:
        print("    check_2581_off FAILED")
        failed += 1

    # ---- check_2580_on -------------------------------------------------
    var a_side = run_arm[True, False](
        ctx, dblocks, plans, depth, n_live, True, cindex, row_index, stats,
        qstats, p_off, p_sz, ids, dense_ids, hist, acc, block_hist, hcl,
        scale,
    )
    var d_a = count_diff(off_side, a_side, want, ob_first, ob_len, hcl)
    var poison = List[Float32]()
    for _ in range(STAT_COUNT * N_ROWS):
        poison.append(Float32(1234.5))
    upload_plane(ctx, stats, poison)
    var a_poison = run_arm[True, False](
        ctx, dblocks, plans, depth, n_live, False, cindex, row_index, stats,
        qstats, p_off, p_sz, ids, dense_ids, hist, acc, block_hist, hcl,
        scale,
    )
    var off_poison = run_arm[False, False](
        ctx, dblocks, plans, depth, n_live, False, cindex, row_index, stats,
        qstats, p_off, p_sz, ids, dense_ids, hist, acc, block_hist, hcl,
        scale,
    )
    upload_plane(ctx, stats, plane)
    var d_ap = count_diff(off_side, a_poison, want, ob_first, ob_len, hcl)
    var d_op = count_diff(off_side, off_poison, want, ob_first, ob_len, hcl)
    print("    check_2580_on: on vs off", d_a, "cells; on with the float plane"
          " poisoned after the quantize", d_ap, "; off with it poisoned", d_op)
    if d_a != 0 or d_ap != 0 or d_op == 0:
        print("    check_2580_on FAILED")
        failed += 1

    # ---- check_2581_on -------------------------------------------------
    var b_side = run_arm[False, True](
        ctx, dblocks, plans, depth, n_live, False, cindex, row_index, stats,
        qstats, p_off, p_sz, ids, dense_ids, hist, acc, block_hist, hcl,
        scale,
    )
    var d_b = count_diff(off_side, b_side, want, ob_first, ob_len, hcl)
    # swap the two 5-bit groups' columns in the map, then restore it
    var d_swap = -1
    for pi in range(len(plans)):
        if plans[pi].feat_count[0] >= 8:
            var length = plans[pi].feat_start[3] + plans[pi].feat_count[3]
            length += plans[pi].feat_count[3] // 4
            var host = ctx.enqueue_create_host_buffer[DType.uint32](length)
            ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=plans[pi].folds)
            ctx.synchronize()
            var at = plans[pi].feat_start[0] + plans[pi].feat_count[0]
            var c0 = host.unsafe_ptr().unsafe_load(at)
            var c1 = host.unsafe_ptr().unsafe_load(at + 1)
            host.unsafe_ptr().unsafe_store(at, c1)
            host.unsafe_ptr().unsafe_store(at + 1, c0)
            ctx.enqueue_copy(dst_buf=plans[pi].folds, src_ptr=host.unsafe_ptr())
            ctx.synchronize()
            var b_swap = run_arm[False, True](
                ctx, dblocks, plans, depth, n_live, False, cindex, row_index,
                stats, qstats, p_off, p_sz, ids, dense_ids, hist, acc,
                block_hist, hcl, scale,
            )
            d_swap = count_diff(off_side, b_swap, want, ob_first, ob_len, hcl)
            host.unsafe_ptr().unsafe_store(at, c0)
            host.unsafe_ptr().unsafe_store(at + 1, c1)
            ctx.enqueue_copy(dst_buf=plans[pi].folds, src_ptr=host.unsafe_ptr())
            ctx.synchronize()
            _ = host^
    print("    check_2581_on: on vs off", d_b, "cells; with two 5-bit groups'"
          " columns swapped", d_swap)
    if d_b != 0 or d_swap <= 0:
        print("    check_2581_on FAILED")
        failed += 1

    # ---- check_2580_2581_on --------------------------------------------
    var ab_side = run_arm[True, True](
        ctx, dblocks, plans, depth, n_live, True, cindex, row_index, stats,
        qstats, p_off, p_sz, ids, dense_ids, hist, acc, block_hist, hcl,
        scale,
    )
    var d_ab = count_diff(off_side, ab_side, want, ob_first, ob_len, hcl)
    print("    check_2580_2581_on: on vs off", d_ab, "cells")
    if d_ab != 0:
        print("    check_2580_2581_on FAILED")
        failed += 1

    _ = z^
    _ = hb^
    _ = hi^
    _ = ho^
    _ = hs^
    _ = hid^
    _ = hd^
    return failed


def main() raises:
    print("sym_arms_check: DEVIATIONS 2580 and 2581, both sides")
    var failed = check_depth(0) + check_depth(1)
    if failed != 0:
        raise Error("sym_arms_check: " + String(failed) + " sides FAILED")
    print("sym_arms_check: every side PASSED")

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Wide fixed-point 8-bit histograms: NS stat planes x FG compressed-index
columns per block, in ONE walk of the partition (lane trees-hist-ideas,
2026-10-07). Default off; reached only through two IDENTICAL switches in
`gbdt/trees_hist_switches.mojo`:

  MOJOLEARN_TREES_HIST_MULTISTAT=4|8 (idea 3). MultiClass carries
      `stat_count` > 2 planes, and its >128-bin one-byte blocks take the PASS
      route `launch_one_byte[8]` with grid z = stat_count: the compressed
      index, the row index and the dither are read once PER PLANE. Here a
      block covers NS planes (grid z = ceil(stat_count / NS)), so they are
      read once per NS planes. FG = 1.

  MOJOLEARN_TREES_HIST_SYM_FEATURE_PARALLEL (idea 4). Two-stat SymmetricTree
      blocks. The incumbent fused kernel gives each block ONE cindex column
      (4 features) and re-reads the row index and both stat planes for every
      column: per gathered row, 4 B of bins against 4 + 8 B of index and stats.
      With FG columns per block the index/stat bytes are paid once per FG
      columns. NS = 2. The launcher picks FG per level by cost
      (`wide_columns_for`), and FG = 1 keeps the incumbent kernel.

BITS. Every addend is the incumbent's: `hist2_quantize(plane[s][p or row],
fixed_scale, hist2_dither(position))` with the position the storage
position in both the direct and the gather arm (and the stat read through
the row id under `ridx_stats`, DEVIATION 1902), exactly as
`hist_one_byte.one_byte_hist_kernel` (PASS) and `hist2_8bit_kernel` (fused)
quantize. Sums are Int32 and integer addition is associative, so the
partition of rows into blocks, slices and stat groups cannot move a bit
(`replication_for`'s note: "any partition of rows into blocks gives the same
bits" for the hist_2/one-byte families). The flush writes the same
`acc_i32` cells with the same dual-branch rule (store when one block covers
the part, atomic otherwise).

Shared memory: one slice is FG * 256 bins * 4 features * NS stats Int32
cells; the block keeps the fused kernel's 8192-cell (32 KB) budget, so the
legal products are NS * FG <= 8 and there are 8 / (NS * FG) slices (1 slice
at the cap: every thread of the block shares it). The atomic count per
(row, stat) is unchanged; contention per slice rises as slices fall, which
is the cost the A/B weighs against the saved walks.

NOT COMPILED -- NOT TESTED -- IDENTITY NOT VERIFIED -- NOT MEASURED.
"""

from std.atomic import Atomic, Ordering
from std.gpu import block_idx, grid_dim, thread_idx
from std.gpu.intrinsics import ldg
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from gbdt.methods.greedy_subsets_searcher.kernel.histogram_utils import (
    hist2_dither,
    hist2_quantize,
)
from gbdt.methods.greedy_subsets_searcher.kernel.hist_2_one_byte_8bit import (
    H8_BLOCK,
    H8_MIN_DOCS,
)

#: the fused kernel's shared budget: 8192 Int32 cells, 32 KB.
comptime HW_SMEM = 8192
#: one column's cells per stat: 256 bins x 4 features.
comptime HW_COLUMN_CELLS = 1024


@always_inline
def hw_slice_cells[ns: Int, fg: Int]() -> Int:
    return HW_COLUMN_CELLS * ns * fg


@always_inline
def hw_slices[ns: Int, fg: Int]() -> Int:
    return HW_SMEM // hw_slice_cells[ns, fg]()


def hist2_8bit_wide_kernel[
    ns: Int, fg: Int, gather: Bool, ridx_stats: Bool = False
](
    feature_folds: MutPointer[UInt32, MutAnyOrigin],
    feature_fold_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_group_offset: MutPointer[UInt32, MutAnyOrigin],
    feature_group_size: MutPointer[UInt32, MutAnyOrigin],
    f_count_in32: Int32,
    cindex: MutPointer[UInt32, MutAnyOrigin],
    bins_line_size_in: Int32,
    cindex_base_in: Int32,
    indices: MutPointer[UInt32, MutAnyOrigin],
    stats: MutPointer[Float32, MutAnyOrigin],
    stat_line_size_in: Int32,
    part_offset: MutPointer[UInt32, MutAnyOrigin],
    part_size: MutPointer[UInt32, MutAnyOrigin],
    part_ids: MutPointer[UInt32, MutAnyOrigin],
    acc_i32: MutPointer[Int32, MutAnyOrigin],
    fixed_scale_ptr: MutPointer[Float32, MutAnyOrigin],
    leaf_count_in: Int32,
    stat_count_in: Int32,
):
    """Grid: x = ceil(columns / fg) * replicas, y = live parts, z =
    ceil(stat_count / ns); block H8_BLOCK. `gather` False is the depth-0
    direct arm (position == row); True reads bins through `indices`."""
    comptime assert ns >= 1 and fg >= 1, "ns and fg are counts"
    comptime assert ns * fg <= 8, "NS * FG must fit the 8192-cell budget"
    comptime assert not (ridx_stats and not gather), (
        "ridx_stats is a gather-arm schedule"
    )
    comptime SLICE = hw_slice_cells[ns, fg]()
    comptime SLICES = hw_slices[ns, fg]()
    comptime PER_SLICE = H8_BLOCK // SLICES
    comptime assert PER_SLICE >= 1, "more slices than threads"
    comptime assert H8_BLOCK >= 256, "the flush covers 256 folds per pass"

    var fixed_scale = fixed_scale_ptr.unsafe_load(0)
    var f_count_in = Int(f_count_in32)
    var bins_line_size = Int(bins_line_size_in)
    var stat_line_size = Int(stat_line_size_in)
    var leaf_count = Int(leaf_count_in)
    var stat_count = Int(stat_count_in)
    var tid = Int(thread_idx.x)

    var part_id = Int(part_ids.unsafe_load(Int(block_idx.y)))
    var p_offset = Int(part_offset.unsafe_load(part_id))
    var p_size = Int(part_size.unsafe_load(part_id))

    var columns = (f_count_in + 3) // 4
    var col_groups = (columns + fg - 1) // fg
    var max_blocks_per_part = Int(grid_dim.x) // col_groups
    var gb = Int(block_idx.x) // max_blocks_per_part
    var local_block_idx = Int(block_idx.x) % max_blocks_per_part
    # the fused kernel's per-block row floor: a block exists only for a full
    # `H8_MIN_DOCS` share of the part, so the fixed zero/fold/flush cost of a
    # block is always amortized over at least that many rows
    var active_block_count = min(
        (p_size + H8_MIN_DOCS - 1) // H8_MIN_DOCS, max_blocks_per_part
    )
    if local_block_idx >= active_block_count:
        return
    var stat0 = Int(block_idx.z) * ns
    var cbase = cindex + Int(cindex_base_in)

    var smem = stack_allocation[
        HW_SMEM,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var z = tid
    while z < HW_SMEM:
        smem[z] = Int32(0)
        z += H8_BLOCK
    barrier()
    var slice_base = SLICE * (tid // PER_SLICE)

    # Row-parallel walk: consecutive threads take consecutive positions, the
    # active blocks of this part interleave by H8_BLOCK. Order is free
    # (Int32 sums); each position is visited exactly once.
    var i = local_block_idx * H8_BLOCK + tid
    var stride = active_block_count * H8_BLOCK
    while i < p_size:
        var pos = p_offset + i
        var row = pos
        comptime if gather:
            row = Int(ldg(indices + pos))
        var u = hist2_dither(pos)
        var q = SIMD[DType.int32, ns](0)
        comptime for s in range(ns):
            var st = stat0 + s
            if st < stat_count:
                var sp = pos
                comptime if ridx_stats:
                    sp = row
                q[s] = hist2_quantize(
                    ldg(stats + (st * stat_line_size + sp)), fixed_scale, u
                )
        comptime for c in range(fg):
            var col = gb * fg + c
            if col < columns:
                var ci = ldg(cbase + (bins_line_size * col + row))
                comptime for k in range(4):
                    # rotate the feature by lane, as the fused kernel does,
                    # to spread a warp's atomics over banks
                    var f = (tid + k) & 3
                    var bin = Int((ci >> UInt32(24 - 8 * f)) & UInt32(255))
                    var cell = slice_base + (
                        ((c * 256 + bin) * 4 + f) * ns
                    )
                    comptime for s in range(ns):
                        if q[s] != Int32(0):
                            _ = Atomic.fetch_add[ordering = Ordering.RELAXED](
                                smem.unsafe_offset(cell + s), q[s]
                            )
        i += stride

    # Fold the slices into slice 0: each residue of SLICE is owned by one
    # thread, which reads every copy before writing (the fused kernel's
    # stage-1 argument). Int32 sums: the fold order cannot move a bit.
    barrier()
    comptime if SLICES > 1:
        var start = tid
        while start < SLICE:
            var acc = smem[start]
            comptime for sl in range(1, SLICES):
                acc += smem[start + sl * SLICE]
            smem[start] = acc
            start += H8_BLOCK
        barrier()

    # Flush, one thread per fold, the fused kernel's dual-branch rule.
    comptime for c in range(fg):
        var col = gb * fg + c
        if col < columns:
            var feature_offset = col * 4
            var f_count = min(f_count_in - feature_offset, 4)
            var fold = tid
            while fold < 256:
                for fid in range(f_count):
                    var folds = Int(
                        feature_folds.unsafe_load(feature_offset + fid)
                    )
                    if fold < folds:
                        var group_offset = Int(
                            feature_group_offset.unsafe_load(
                                feature_offset + fid
                            )
                        )
                        var group_size = Int(
                            feature_group_size.unsafe_load(feature_offset + fid)
                        )
                        var fold_off = Int(
                            feature_fold_offset.unsafe_load(
                                feature_offset + fid
                            )
                        )
                        var device_offset = (
                            group_offset * stat_count * leaf_count
                        )
                        var entries_per_leaf = stat_count * group_size
                        comptime for s in range(ns):
                            var st = stat0 + s
                            if st < stat_count:
                                var qv = smem[
                                    ((c * 256 + fold) * 4 + fid) * ns + s
                                ]
                                if qv != Int32(0):
                                    var dst = (
                                        device_offset
                                        + Int(block_idx.y) * entries_per_leaf
                                        + st * group_size
                                        + fold_off
                                        + fold
                                    )
                                    if active_block_count > 1:
                                        _ = Atomic.fetch_add[
                                            ordering = Ordering.RELAXED
                                        ](acc_i32.unsafe_offset(dst), qv)
                                    else:
                                        acc_i32.unsafe_store(dst, qv)
                fold += H8_BLOCK


def wide_columns_for(columns: Int, n_live: Int, sm_count: Int) -> Int:
    """Idea 4's cost rule: how many cindex columns one two-stat block takes.

    Per gathered row a block reads 4 B of row index and 8 B of stats once,
    plus 4 B of bins per column it covers; the incumbent (FG = 1) pays the
    12 B again for every column, so FG columns cut that traffic by a factor
    FG (the direct depth-0 arm reads no row index: 8 B saved per extra
    column instead of 12). The price is parallelism: the grid has
    ceil(columns / FG) * n_live feature-blocks before any row replication.
    Widening is therefore taken only while the widened grid still holds at
    least the device's concurrent block capacity (`2 * sm_count`, CatBoost's
    `blocksPerSm = 2` over the device's REAL multiprocessor count -- a
    hardware quantity), i.e. when the feature axis alone fills the machine
    and row replication would add nothing. Wide data and deep levels reach
    it; narrow data never does and keeps the row-parallel incumbent.
    Legal results {1, 2, 4} (NS = 2, so NS * FG <= 8). No dataset dimension
    enters: only the live-part count, the column count and the SM count."""
    var capacity = 2 * sm_count
    if capacity < 1:
        capacity = 1
    if columns >= 4 and ((columns + 3) // 4) * n_live >= capacity:
        return 4
    if columns >= 2 and ((columns + 1) // 2) * n_live >= capacity:
        return 2
    return 1

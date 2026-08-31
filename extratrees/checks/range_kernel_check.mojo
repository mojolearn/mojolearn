# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Does the per-(node, feature) RANGE kernel agree with the host, bit for bit?

    pixi run mojo run -I . extratrees/checks/range_kernel_check.mojo

NO CUML COUNTERPART -- this is a check. It covers
`extratrees/impl/decisiontree/batched_levelalgo/kernels/
builder_kernels_impl.mojo::node_feature_range_kernel`, which is step 1 of
DEVIATION 137 and carries DEVIATION BLOCKS 160-163.

WHY THE COMPARISON CAN BE EXACT, WITH NO TOLERANCE AT ALL
---------------------------------------------------------
`min` and `max` on IEEE-754 floats SELECT an input and return it unchanged.
Nothing rounds, so they are associative and commutative EXACTLY, and no
regrouping -- not the device's strided per-thread accumulation, not the block
collective, not the cross-block mutex merge -- can change which input
survives. The NaN count is an exact integer sum with the same property. So the
device answer and `node_feature_min_max`'s sequential
`for p in range(begin, end)` loop are OBLIGED to produce identical bits, and a
tolerance here would hide a defect rather than absorb float noise. Every
assertion below is on `.to_bits()`.

(This is the property a HISTOGRAM kernel does not have, which is why
`gbdt/` compares its own histograms against a recorded oracle and this file
does not have to.)

WHAT IS DELIBERATELY ADVERSARIAL ABOUT THE FIXTURE
--------------------------------------------------
1. **Column-major storage.** `fixtures.Dataset` is ROW-major
   (`x[row * n_cols + col]`); cuML's `Dataset` is COLUMN-major
   (`dataset.h:24`). The buffer handed to the device is transposed here, so a
   kernel that indexes `row * N + col` reads a different cell -- not a
   different layout of the same cell.
2. **Shuffled `row_ids`, and nodes that do not start at slot 0.** A slot index
   and a row id are different numbers for essentially every slot, so dropping
   the indirection is visible.
3. **`fslot` is not `col`.** Each node's sampled-column list is a different
   permutation of all 12 columns, so a kernel that used `blockIdx.y` as the
   column id would read the wrong feature for six of the seven nodes.
4. **A RAGGED batch, with both paths present and NAMED.** Node row counts are
   0, 1, 3, 127, 128, 129 and 1000 against a 128-thread block, which is one
   block for the first five and two and eight blocks for the last two. Rule 8
   says a parameter that selects a kernel path is a parameter the checks
   enumerate, and a check that cannot name the path it ran can pass about a
   different one -- so the kernel REPORTS how many blocks published into each
   cell (`out_n_merges`, DEVIATION 162) and this file asserts that count
   against `ceildiv`, per cell. The device names the path, not the host.
5. **The adversarial column shapes**, straight from `fixtures.mojo`: exactly
   constant, near-constant one ulp below / at / above FEATURE_THRESHOLD,
   two-valued with nothing between, one far outlier, all-negative,
   spanning zero (where float ordering is NOT integer ordering), all-equal
   but one row, and hashed noise.
6. **Two NaN columns, built HERE and not in `fixtures.mojo`.** DEVIATION 150
   records that no fixture contains a missing value and DEVIATION 136 refuses
   NaN at the estimator boundary -- but `node_feature_min_max` HAS NaN
   branches (`_partitioner.pyx:146-163`), the kernel transcribes them, and an
   unchecked branch is an unreached branch. Column 10 is NaN on a hash bit,
   scattered; column 11 is entirely NaN, which is the only column that
   reaches the all-missing sentinel of DEVIATION 163. `fixtures.mojo` is not
   modified: these two columns live in this file, and no other check sees
   them.

THE ARMS
--------
  A. PER-CELL, on float bit patterns, all 84 (node, feature) cells against
     `node_feature_min_max` -- min, max and n_missing, three comparisons a
     cell.
  B. THE PATH, from the device's own report: `out_n_merges` per cell against
     `max(ceildiv(count, TPB), 1)`, so the single-block and multi-block arms
     are each named and counted rather than assumed.
  C. SABOTAGE, one per MECHANISM, each a kernel ARGUMENT so that every arm
     runs the SAME binary that ships. Five mechanisms: the column-major
     indexing, the `row_ids` indirection, the multi-block combine, the NaN
     test, and the empty sentinel. Each states what it predicts BEFORE the
     run, and a sabotage that does not turn arm A red is a defect in this
     FIXTURE, not a pass.
"""

from std.math import ceildiv
from std.sys import has_accelerator
from std.sys.info import (
    has_amd_gpu_accelerator,
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    size_of,
)
from max.gpu.host import DeviceContext

from extratrees.checks.fixtures import (
    all_shapes,
    cell_hash,
    float_from_bits,
    shape_name,
    shaped_dataset,
    SALT_X,
)
from extratrees.impl.decisiontree.batched_levelalgo.dataset import Dataset
from extratrees.impl.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    NodeWorkItem,
    WorkloadInfo,
)
from extratrees.impl.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import (
    build_workload_info,
    FeatureRange,
    node_feature_min_max,
    node_feature_range_decode_kernel,
    node_feature_range_init_kernel,
    node_feature_range_kernel,
    RANGE_SAB_BLOCK0_ONLY,
    RANGE_SAB_NAN_AS_VALUE,
    RANGE_SAB_NO_ROW_IDS,
    RANGE_SAB_NO_SENTINEL,
    RANGE_SAB_NONE,
    RANGE_SAB_EMPTY_NOT_IDENTITY,
    RANGE_SAB_SIGN_UNFLIPPED,
    RANGE_SAB_ROW_MAJOR,
    TPB_DEFAULT,
)


comptime TPB = TPB_DEFAULT
comptime N_ROWS = 2048
comptime SEED: UInt64 = 0xC0FFEE_123
comptime QNAN_BITS: UInt32 = 0x7FC00000
"""A quiet NaN as a BIT PATTERN. `0.0 / 0.0` is a constant the compiler is
free to fold differently at different optimisation levels; a bit pattern is
the same NaN on the host and on the device."""

comptime COL_SOME_NAN = 10
comptime COL_ALL_NAN = 11


@fieldwise_init
struct RangeRun(Copyable, Movable):
    """One launch's four output arrays, copied back to the host."""

    var min_v: List[Float32]
    var max_v: List[Float32]
    var n_missing: List[Int32]
    var n_merges: List[Int32]


def _node_ranges() -> List[InstanceRange]:
    """The RAGGED batch. Against TPB = 128 these are 1, 1, 1, 1, 1, 2 and 8
    blocks, so both the single-block and the multi-block arm run, and the
    boundary is straddled on both sides (127 / 128 / 129).

    None of them begins at slot 0, and they do not tile `row_ids` -- there are
    gaps between them, so a kernel that walked from the previous node's end
    would read rows that belong to no node.
    """
    var out = List[InstanceRange]()
    out.append(InstanceRange(Int32(7), Int32(1)))
    out.append(InstanceRange(Int32(11), Int32(3)))
    out.append(InstanceRange(Int32(20), Int32(127)))
    out.append(InstanceRange(Int32(200), Int32(128)))
    out.append(InstanceRange(Int32(400), Int32(129)))
    out.append(InstanceRange(Int32(600), Int32(1000)))
    out.append(InstanceRange(Int32(1900), Int32(0)))
    return out^


def _shuffled_row_ids(n_rows: Int) -> List[Int32]:
    """A Fisher-Yates shuffle over a deterministic hash, so a slot index and
    the row id it holds are different numbers."""
    var ids = List[Int32]()
    for r in range(n_rows):
        ids.append(Int32(r))
    for i in range(n_rows - 1, 0, -1):
        var h = UInt32(i) * 2654435761
        h ^= h >> 15
        var j = Int(h % UInt32(i + 1))
        var t = ids[i]
        ids[i] = ids[j]
        ids[j] = t
    return ids^


def main() raises:
    comptime assert has_accelerator(), "this check must run on a GPU"

    # ---------------------------------------------------------------------
    # The data. Ten shaped columns from `fixtures.mojo`, plus two NaN columns
    # built here (see the module docstring, point 6).
    # ---------------------------------------------------------------------
    var shapes = all_shapes()
    var fixture = shaped_dataset(SEED, N_ROWS, shapes)
    var n_shaped = fixture.n_cols
    var n_cols = n_shaped + 2
    if n_shaped != COL_SOME_NAN:
        raise Error(
            "fixtures.all_shapes() no longer returns "
            + String(COL_SOME_NAN)
            + " shapes; the NaN column indices in this file are stale"
        )

    var qnan = float_from_bits(QNAN_BITS)

    # COLUMN MAJOR, `dataset.h:24`: `data[col * M + row]`. The fixture is
    # row major, so this transpose is where the two layouts meet.
    var flat = List[Float32](length=n_cols * N_ROWS, fill=Float32(0.0))
    for c in range(n_shaped):
        for r in range(N_ROWS):
            flat[c * N_ROWS + r] = fixture.value(r, c)
    var planted_some_nan = 0
    for r in range(N_ROWS):
        # Scattered on a hash bit, not blocked: placement has to matter.
        var h = cell_hash(SEED, r, COL_SOME_NAN, SALT_X)
        var v: Float32
        if (h >> 33) % 5 == 0:
            v = qnan
            planted_some_nan += 1
        else:
            v = Float32(Float64((h >> 40)) / 16777216.0) * 100.0 - 50.0
        flat[COL_SOME_NAN * N_ROWS + r] = v
    for r in range(N_ROWS):
        flat[COL_ALL_NAN * N_ROWS + r] = qnan

    var labels = List[Float32](length=N_ROWS, fill=Float32(0.0))
    var row_ids = _shuffled_row_ids(N_ROWS)

    # Every node samples ALL the columns, but in its OWN permuted order, so
    # `fslot` and `col` are different numbers for six of the seven nodes.
    # `5` and `12` are coprime, so each row of this table is a permutation.
    var ranges = _node_ranges()
    var n_nodes = len(ranges)
    var n_sampled_cols = n_cols
    var colids = List[Int32]()
    for nid in range(n_nodes):
        for fslot in range(n_sampled_cols):
            colids.append(Int32((fslot * 5 + nid * 7) % n_cols))

    var work_items = List[NodeWorkItem]()
    for nid in range(n_nodes):
        # `idx` is the node's id IN THE TREE and deliberately does not start
        # at 0 either; the kernel never reads it, and that is worth stating.
        work_items.append(NodeWorkItem(Int32(100 + nid), Int32(3), ranges[nid]))

    var plan = build_workload_info(work_items, TPB)
    var n_cells = n_nodes * n_sampled_cols

    print("[fixture]", n_nodes, "nodes x", n_sampled_cols, "sampled columns =",
          n_cells, "cells;", N_ROWS, "rows,", n_cols, "columns")
    print("          workload_info flattens the batch into", plan.n_blocks_dimx,
          "blocks,", plan.n_large_nodes, "of the nodes are LARGE (> 1 block)")
    print("          column", COL_SOME_NAN, "carries", planted_some_nan,
          "planted NaNs; column", COL_ALL_NAN, "is entirely NaN")

    # ---------------------------------------------------------------------
    # The host oracle, per cell. `node_feature_min_max` is
    # `_partitioner.pyx:129-165` transcribed, and it is the authority here.
    # ---------------------------------------------------------------------
    var dataset = Dataset(
        rebind[MutPointer[Float32, MutUntrackedOrigin]](flat.unsafe_ptr()),
        rebind[MutPointer[Float32, MutUntrackedOrigin]](labels.unsafe_ptr()),
        Int32(N_ROWS),
        Int32(n_cols),
        Int32(N_ROWS),
        Int32(n_sampled_cols),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](row_ids.unsafe_ptr()),
        Int32(1),
    )

    var want_min = List[Float32]()
    var want_max = List[Float32]()
    var want_missing = List[Int32]()
    for nid in range(n_nodes):
        for fslot in range(n_sampled_cols):
            var col = colids[nid * n_sampled_cols + fslot]
            var e = node_feature_min_max(dataset, work_items[nid], col)
            want_min.append(e.min_value)
            want_max.append(e.max_value)
            want_missing.append(e.n_missing)

    # ---------------------------------------------------------------------
    # The device. One allocation set, six launches -- the shipping arm and
    # five sabotages, all of the SAME binary, selected by a kernel argument.
    # ---------------------------------------------------------------------
    var ctx = DeviceContext()
    # REACH, stated rather than assumed. `out_n_merges` is written ONLY by
    # the device kernel -- no host code touches it -- so arm B's "8 blocks
    # published into every cell of node 5" is a report from eight distinct
    # threadblocks that ran. `enqueue_create_host_buffer` is never passed to
    # a kernel here: a kernel writing one writes nothing, silently.
    print("[device] accelerator present;", _vendor(), "-- kernels are",
          "enqueued on it, and every arm below runs the SAME binary")

    var d_min = ctx.enqueue_create_buffer[DType.float32](n_cells)
    var d_max = ctx.enqueue_create_buffer[DType.float32](n_cells)
    var d_missing = ctx.enqueue_create_buffer[DType.int32](n_cells)
    var d_merges = ctx.enqueue_create_buffer[DType.int32](n_cells)
    var d_minkey = ctx.enqueue_create_buffer[DType.uint32](n_cells)
    var d_maxkey = ctx.enqueue_create_buffer[DType.uint32](n_cells)
    var d_data = ctx.enqueue_create_buffer[DType.float32](n_cols * N_ROWS)
    var d_row_ids = ctx.enqueue_create_buffer[DType.int32](N_ROWS)
    var d_colids = ctx.enqueue_create_buffer[DType.int32](len(colids))
    var d_items = ctx.enqueue_create_buffer[DType.uint8](
        n_nodes * size_of[NodeWorkItem]()
    )
    var d_wl = ctx.enqueue_create_buffer[DType.uint8](
        plan.n_blocks_dimx * size_of[WorkloadInfo]()
    )

    # One host staging buffer per copy: they are async, and a shared staging
    # buffer would be rewritten under an in-flight copy.
    var h_data = ctx.enqueue_create_host_buffer[DType.float32](n_cols * N_ROWS)
    var h_row_ids = ctx.enqueue_create_host_buffer[DType.int32](N_ROWS)
    var h_colids = ctx.enqueue_create_host_buffer[DType.int32](len(colids))
    var h_items = ctx.enqueue_create_host_buffer[DType.uint8](
        n_nodes * size_of[NodeWorkItem]()
    )
    var h_wl = ctx.enqueue_create_host_buffer[DType.uint8](
        plan.n_blocks_dimx * size_of[WorkloadInfo]()
    )
    ctx.synchronize()

    for i in range(n_cols * N_ROWS):
        h_data.unsafe_ptr().unsafe_store(i, flat[i])
    for i in range(N_ROWS):
        h_row_ids.unsafe_ptr().unsafe_store(i, row_ids[i])
    for i in range(len(colids)):
        h_colids.unsafe_ptr().unsafe_store(i, colids[i])
    var items_ptr = h_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem]()
    for i in range(n_nodes):
        items_ptr[unsafe_offset=i] = work_items[i]
    var wl_ptr = h_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo]()
    for i in range(plan.n_blocks_dimx):
        wl_ptr[unsafe_offset=i] = plan.info[i]

    ctx.enqueue_copy(dst_buf=d_data, src_ptr=h_data.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_row_ids, src_ptr=h_row_ids.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_colids, src_ptr=h_colids.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_items, src_ptr=h_items.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=d_wl, src_ptr=h_wl.unsafe_ptr())
    ctx.synchronize()

    var arms = [
        RANGE_SAB_NONE,
        RANGE_SAB_ROW_MAJOR,
        RANGE_SAB_NO_ROW_IDS,
        RANGE_SAB_BLOCK0_ONLY,
        RANGE_SAB_NAN_AS_VALUE,
        RANGE_SAB_NO_SENTINEL,
        RANGE_SAB_SIGN_UNFLIPPED,
        RANGE_SAB_EMPTY_NOT_IDENTITY,
    ]
    var runs = List[RangeRun]()

    for a in range(len(arms)):
        var sab = arms[a]
        ctx.enqueue_function[node_feature_range_init_kernel](
            d_min.unsafe_ptr(),
            d_max.unsafe_ptr(),
            d_minkey.unsafe_ptr(),
            d_maxkey.unsafe_ptr(),
            d_missing.unsafe_ptr(),
            d_merges.unsafe_ptr(),
            Int32(n_cells),
            grid_dim=ceildiv(n_cells, 64),
            block_dim=64,
        )
        ctx.enqueue_function[node_feature_range_kernel[TPB]](
            d_minkey.unsafe_ptr(),
            d_maxkey.unsafe_ptr(),
            d_missing.unsafe_ptr(),
            d_merges.unsafe_ptr(),
            d_data.unsafe_ptr(),
            d_row_ids.unsafe_ptr(),
            d_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem](),
            d_wl.unsafe_ptr().unsafe_bitcast[WorkloadInfo](),
            d_colids.unsafe_ptr(),
            Int32(N_ROWS),
            Int32(n_cols),
            Int32(n_sampled_cols),
            Int32(sab),
            grid_dim=(plan.n_blocks_dimx, n_sampled_cols, 1),
            block_dim=(TPB, 1, 1),
        )
        # DEVIATION 204: the merge writes order-preserving KEYS; this turns
        # them back into the `(min, max)` cells every assertion below reads.
        # The sabotage argument goes to BOTH, because the empty-cell sentinel
        # now lives in the decode and the key map lives in the merge.
        ctx.enqueue_function[node_feature_range_decode_kernel](
            d_min.unsafe_ptr(),
            d_max.unsafe_ptr(),
            d_minkey.unsafe_ptr(),
            d_maxkey.unsafe_ptr(),
            Int32(n_cells),
            Int32(sab),
            grid_dim=ceildiv(n_cells, 64),
            block_dim=64,
        )
        var o_min = ctx.enqueue_create_host_buffer[DType.float32](n_cells)
        var o_max = ctx.enqueue_create_host_buffer[DType.float32](n_cells)
        var o_missing = ctx.enqueue_create_host_buffer[DType.int32](n_cells)
        var o_merges = ctx.enqueue_create_host_buffer[DType.int32](n_cells)
        ctx.enqueue_copy(dst_buf=o_min, src_buf=d_min)
        ctx.enqueue_copy(dst_buf=o_max, src_buf=d_max)
        ctx.enqueue_copy(dst_buf=o_missing, src_buf=d_missing)
        ctx.enqueue_copy(dst_buf=o_merges, src_buf=d_merges)
        ctx.synchronize()

        var mn = List[Float32]()
        var mx = List[Float32]()
        var nm = List[Int32]()
        var mg = List[Int32]()
        for i in range(n_cells):
            mn.append(o_min.unsafe_ptr().unsafe_load(i))
            mx.append(o_max.unsafe_ptr().unsafe_load(i))
            nm.append(o_missing.unsafe_ptr().unsafe_load(i))
            mg.append(o_merges.unsafe_ptr().unsafe_load(i))
        runs.append(RangeRun(mn^, mx^, nm^, mg^))

    var base = runs[0].copy()
    var failures = 0

    # ---------------------------------------------------------------------
    # ARM A -- per cell, on bit patterns, no tolerance.
    # ---------------------------------------------------------------------
    print("")
    print("[arm A] per (node, feature) cell against node_feature_min_max,"
          " on float BIT PATTERNS")
    var cells_ok = 0
    var cells_bad = 0
    var reported = 0
    for nid in range(n_nodes):
        for fslot in range(n_sampled_cols):
            var s = nid * n_sampled_cols + fslot
            var col = Int(colids[s])
            var ok = (
                base.min_v[s].to_bits() == want_min[s].to_bits()
                and base.max_v[s].to_bits() == want_max[s].to_bits()
                and base.n_missing[s] == want_missing[s]
            )
            if ok:
                cells_ok += 1
            else:
                cells_bad += 1
                if reported < 6:
                    reported += 1
                    print("  MISMATCH node", nid, "fslot", fslot, "col", col,
                          "(", _col_name(col, shapes), ") got [",
                          base.min_v[s], ",", base.max_v[s], "] nan",
                          base.n_missing[s], " want [", want_min[s], ",",
                          want_max[s], "] nan", want_missing[s])
    if cells_bad == 0:
        print("  arm A OK:", cells_ok, "of", n_cells,
              "cells bit-identical to the host transcription")
    else:
        failures += 1
        print("  arm A FAILED:", cells_bad, "of", n_cells, "cells wrong")

    # ---------------------------------------------------------------------
    # ARM B -- which path each node took, from the DEVICE's own report.
    # ---------------------------------------------------------------------
    print("")
    print("[arm B] the path each node took, as reported by the kernel"
          " (out_n_merges), not inferred")
    var single = 0
    var multi = 0
    var path_bad = 0
    for nid in range(n_nodes):
        var count = Int(ranges[nid].count)
        var want_blocks = ceildiv(count, TPB)
        if want_blocks < 1:
            want_blocks = 1
        var got_blocks = Int(base.n_merges[nid * n_sampled_cols])
        var consistent = True
        for fslot in range(n_sampled_cols):
            if Int(base.n_merges[nid * n_sampled_cols + fslot]) != want_blocks:
                consistent = False
        var tag = "SINGLE-BLOCK" if want_blocks == 1 else "MULTI-BLOCK"
        if want_blocks == 1:
            single += 1
        else:
            multi += 1
        if consistent and got_blocks == want_blocks:
            print("  node", nid, "rows", count, "->", want_blocks, "block(s),",
                  tag, "- every one of its", n_sampled_cols,
                  "cells was published by exactly", want_blocks, "block(s)")
        else:
            path_bad += 1
            print("  node", nid, "rows", count, "PATH WRONG: expected",
                  want_blocks, "publishing block(s), cell 0 saw", got_blocks,
                  ", cells consistent:", consistent)
    if path_bad == 0 and single > 0 and multi > 0:
        print("  arm B OK:", single, "nodes took the single-block path and",
              multi, "took the multi-block path; BOTH ran")
    else:
        failures += 1
        print("  arm B FAILED: path_bad", path_bad, "single", single,
              "multi", multi, "-- a batch missing one of the two arms cannot"
              " speak for the arm it did not run")

    # ---------------------------------------------------------------------
    # ARM C -- one sabotage per mechanism, each with its prediction stated
    # before the count.
    # ---------------------------------------------------------------------
    print("")
    print("[arm C] sabotage, one per mechanism, same binary, selected by a"
          " kernel argument")

    failures += _sabotage(
        "column-major indexing (`col * M + row` -> `row * N + col`,"
        " dataset.h:24)",
        "moves nearly every cell: the transposed read lands on a different"
        " column's values entirely",
        runs[1], want_min, want_max, want_missing, n_cells,
    )
    failures += _sabotage(
        "the row_ids indirection (`row_ids[i]` -> `i`)",
        "moves nearly every cell: row_ids is a shuffle, so slot i and row"
        " row_ids[i] are different rows",
        runs[2], want_min, want_max, want_missing, n_cells,
    )
    failures += _sabotage(
        "the multi-block combine (publish only from offset_blockid == 0)",
        "moves exactly those cells where block 0's slots carry a DIFFERENT"
        " range from all the slots -- never a single-block node, and not"
        " every large-node cell either (see _block0_moved_cells)",
        runs[3], want_min, want_max, want_missing, n_cells,
    )
    failures += _sabotage(
        "the NaN test (`v != v` dropped, so NaN is neither counted nor"
        " diverted)",
        "moves n_missing and NOTHING else, in exactly the cells that"
        " actually contain a NaN (see _nan_moved_cells)",
        runs[4], want_min, want_max, want_missing, n_cells,
    )
    failures += _sabotage(
        "the empty-cell sentinel in the DECODE (publish (+inf, -inf)"
        " instead of (1.0, -1.0)) -- deviation 204 moved it there",
        "moves ONLY the cells with no non-missing value: the all-NaN column"
        " in every node, plus EVERY column of the 0-row node",
        runs[5], want_min, want_max, want_missing, n_cells,
    )

    failures += _sabotage(
        "the order-preserving key's NEGATIVE branch (invert-the-bits ->"
        " set-the-sign-bit), DEVIATION 204",
        "moves exactly the cells whose column carries a NEGATIVE value on"
        " this node's rows, where raw-bit order is REVERSED, and leaves every"
        " all-non-negative cell alone -- the exact set is computed by"
        " _negative_cells and checked below",
        runs[6], want_min, want_max, want_missing, n_cells,
    )

    # A sabotage that moves everything is not evidence either, so the three
    # NARROW predictions above are checked for their SHAPE, not just for
    # movement.
    print("")
    print("[arm C'] the narrow sabotages moved the cells they predicted and"
          " no others")
    failures += _moved_exactly(
        "multi-block combine",
        runs[3], base, n_nodes, n_sampled_cols,
        _block0_moved_cells(
            flat, row_ids, ranges, colids, n_nodes, n_sampled_cols,
            want_min, want_max, want_missing,
        ),
    )
    failures += _moved_exactly(
        "NaN test",
        runs[4], base, n_nodes, n_sampled_cols,
        _nan_moved_cells(want_missing),
    )
    failures += _moved_exactly(
        "empty sentinel",
        runs[5], base, n_nodes, n_sampled_cols,
        _empty_cells(ranges, colids, n_nodes, n_sampled_cols),
    )
    var neg = _negative_cells(
        flat, row_ids, ranges, colids, N_ROWS, n_nodes, n_sampled_cols
    )
    var n_neg = 0
    for i in range(len(neg)):
        if neg[i]:
            n_neg += 1
    if n_neg == 0:
        print("  order-preserving key FAILED: this fixture has no negative"
              " value anywhere, so RANGE_SAB_SIGN_UNFLIPPED cannot move a single"
              " cell and the arm proves nothing")
        failures += 1
    else:
        print("     (", n_neg, "of", n_cells,
              "cells carry a negative value; on the other",
              n_cells - n_neg,
              "the two spellings are the same expression, so this arm"
              " cannot move them and must not)")
    failures += _moved_exactly(
        "order-preserving key",
        runs[6], base, n_nodes, n_sampled_cols, neg,
    )
    failures += _sabotage(
        "the +/-inf IDENTITY an empty block relies on (publish"
        " range_key(0.0) instead), DEVIATION 204",
        "moves exactly the cells with an empty contribution -- the all-NaN"
        " column in every node and every column of the 0-row node -- which"
        " is what makes deleting 163's sentinel gate from the merge safe",
        runs[7], want_min, want_max, want_missing, n_cells,
    )
    failures += _moved_exactly(
        "empty-block identity",
        runs[7], base, n_nodes, n_sampled_cols,
        _empty_cells(ranges, colids, n_nodes, n_sampled_cols),
    )

    print("")
    if failures == 0:
        print("range_kernel_check: PASS --", n_cells,
              "cells bit-identical, both block paths named by the device,"
              " seven mechanisms sabotaged and each moved what it predicted")
    else:
        print("range_kernel_check: FAIL --", failures, "arm(s) red")
        raise Error("range_kernel_check failed")


def _vendor() -> String:
    """Which GPU family this run used. Host-side check (`has_*`), not a
    target check (`is_*`): the kernel source is one GPU-agnostic source and
    nothing in it branches on the vendor."""
    comptime if has_apple_gpu_accelerator():
        return "Apple"
    elif has_nvidia_gpu_accelerator():
        return "NVIDIA"
    elif has_amd_gpu_accelerator():
        return "AMD"
    else:
        return "unknown vendor"


def _col_name(col: Int, shapes: List[Int]) -> String:
    if col == COL_SOME_NAN:
        return "scattered NaN"
    if col == COL_ALL_NAN:
        return "all NaN"
    return shape_name(shapes[col])


def _sabotage(
    name: String,
    prediction: String,
    run: RangeRun,
    want_min: List[Float32],
    want_max: List[Float32],
    want_missing: List[Int32],
    n_cells: Int,
) -> Int:
    """A sabotage must turn arm A RED. If it does not, the FIXTURE is the
    defect -- it means arm A cannot see the mechanism at all."""
    var wrong = 0
    for s in range(n_cells):
        if (
            run.min_v[s].to_bits() != want_min[s].to_bits()
            or run.max_v[s].to_bits() != want_max[s].to_bits()
            or run.n_missing[s] != want_missing[s]
        ):
            wrong += 1
    print("  -", name)
    print("      predicted:", prediction)
    if wrong == 0:
        print("      *** DEFECT IN THE CHECK ***: 0 of", n_cells,
              "cells moved. Arm A cannot see this mechanism; the fixture is"
              " what needs fixing, not the kernel.")
        return 1
    print("      RED as required:", wrong, "of", n_cells, "cells wrong")
    return 0


def _moved_exactly(
    name: String,
    run: RangeRun,
    base: RangeRun,
    n_nodes: Int,
    n_sampled_cols: Int,
    expected_moved: List[Bool],
) -> Int:
    """A sabotage whose prediction is 'most of them' cannot fail. These three
    predict an exact SET of cells, computed rather than guessed."""
    var wrong_shape = 0
    var moved = 0
    for s in range(n_nodes * n_sampled_cols):
        var did_move = (
            run.min_v[s].to_bits() != base.min_v[s].to_bits()
            or run.max_v[s].to_bits() != base.max_v[s].to_bits()
            or run.n_missing[s] != base.n_missing[s]
        )
        if did_move:
            moved += 1
        if did_move != expected_moved[s]:
            wrong_shape += 1
    var want_n = 0
    for s in range(len(expected_moved)):
        if expected_moved[s]:
            want_n += 1
    if wrong_shape == 0:
        print("  -", name, "moved EXACTLY the", moved,
              "cells predicted, and no others")
        return 0
    print("  -", name, "SHAPE WRONG:", moved, "cells moved,", want_n,
          "predicted,", wrong_shape, "cells disagree with the prediction")
    return 1


def _host_range_subset(
    flat: List[Float32],
    row_ids: List[Int32],
    n_rows: Int,
    begin: Int,
    count: Int,
    col: Int,
    period: Int,
    width: Int,
) -> FeatureRange:
    """`node_feature_min_max` restricted to a STRIDED SUBSET of the node's
    slots: offset `o` is visited iff `o % period < width`.

    `period = 1, width = 1` is the whole node. `period = TPB * num_blocks,
    width = TPB` is exactly the set of slots block 0 visits, because the
    kernel's loop is `i = range_start + threadIdx.x + offset_blockid * TPB`
    stepping by `TPB * num_blocks` (`builder_kernels_impl.cuh:264-265, 281`).

    Same NaN convention as the host oracle, so the two are comparable.
    """
    var min_value = Float32(0.0)
    var max_value = Float32(0.0)
    var n_missing = 0
    var seen = False
    for o in range(count):
        if o % period >= width:
            continue
        var row = Int(row_ids[begin + o])
        var v = flat[col * n_rows + row]
        if v != v:
            n_missing += 1
        elif not seen:
            min_value = v
            max_value = v
            seen = True
        elif v < min_value:
            min_value = v
        elif v > max_value:
            max_value = v
    if not seen:
        return FeatureRange(1.0, -1.0, Int32(n_missing))
    return FeatureRange(min_value, max_value, Int32(n_missing))


def _block0_moved_cells(
    flat: List[Float32],
    row_ids: List[Int32],
    ranges: List[InstanceRange],
    colids: List[Int32],
    n_nodes: Int,
    n_sampled_cols: Int,
    want_min: List[Float32],
    want_max: List[Float32],
    want_missing: List[Int32],
) -> List[Bool]:
    """Which cells the BLOCK0_ONLY sabotage moves -- COMPUTED, not guessed.

    THE FIRST VERSION OF THIS PREDICTION WAS WRONG and it is worth recording
    why, because the wrong version is the obvious one. It said "every cell of
    the two large nodes", 24 cells; the run said 7. The run was right. A
    partial slice of a node can carry the SAME range as the whole node --
    trivially so for the exactly-constant column, and often for the
    two-valued and near-constant columns, whose only two distinct values are
    both present in the first 128 rows. So dropping the other blocks is
    INVISIBLE in those cells, and a prediction that demanded they move was a
    prediction about the fixture's data rather than about the mechanism.

    What the mechanism actually guarantees is narrower and is what is
    computed here: a cell moves iff the range over block 0's slots differs
    from the range over all of them. Cells of single-block nodes are included
    in the computation and come out False by construction, which is the other
    half of the prediction.
    """
    var out = List[Bool]()
    for nid in range(n_nodes):
        var begin = Int(ranges[nid].begin)
        var count = Int(ranges[nid].count)
        var num_blocks = ceildiv(count, TPB)
        if num_blocks < 1:
            num_blocks = 1
        for fslot in range(n_sampled_cols):
            var s = nid * n_sampled_cols + fslot
            var col = Int(colids[s])
            var partial = _host_range_subset(
                flat, row_ids, N_ROWS, begin, count, col,
                TPB * num_blocks, TPB,
            )
            out.append(
                partial.min_value.to_bits() != want_min[s].to_bits()
                or partial.max_value.to_bits() != want_max[s].to_bits()
                or partial.n_missing != want_missing[s]
            )
    return out^


def _nan_moved_cells(want_missing: List[Int32]) -> List[Bool]:
    """Which cells the NAN_AS_VALUE sabotage moves -- derived, not guessed.

    THE FIRST VERSION SAID "both NaN columns, 14 cells"; the run said 11, and
    the run was right. Dropping the `v != v` test cannot change min or max at
    all: a NaN fails BOTH `v < min` and `v > max`, so it is excluded from the
    reduction either way. The only thing that changes is the COUNT. So a cell
    moves iff it actually had a NaN in it -- which excludes the 0-row node's
    two NaN cells and the 1-row node's scattered-NaN cell, whose single row
    happens not to be a NaN.

    That this sabotage moves ONLY n_missing is itself worth having stated:
    the NaN branch's effect on the RANGE is carried entirely by the empty
    sentinel, which is the next sabotage.
    """
    var out = List[Bool]()
    for s in range(len(want_missing)):
        out.append(want_missing[s] != Int32(0))
    return out^


def _negative_cells(
    flat: List[Float32],
    row_ids: List[Int32],
    ranges: List[InstanceRange],
    colids: List[Int32],
    n_rows: Int,
    n_nodes: Int,
    n_sampled_cols: Int,
) -> List[Bool]:
    """Cells whose column carries a NEGATIVE non-missing value on this node's
    rows.

    `range_key` and the raw bit pattern AGREE on non-negative floats and
    disagree on negative ones, so this is exactly the set
    `RANGE_SAB_SIGN_UNFLIPPED` can move. Computing it rather than asserting "most
    of them" is what makes the arm evidence: if this fixture ever lost its
    negative values the predicted set would go empty, and an empty predicted
    set with a green sabotage is a CHECK DEFECT rather than a passing check.
    """
    var out = List[Bool]()
    for nid in range(n_nodes):
        var begin = Int(ranges[nid].begin)
        var count = Int(ranges[nid].count)
        for fslot in range(n_sampled_cols):
            var col = Int(colids[nid * n_sampled_cols + fslot])
            var has_neg = False
            for i in range(begin, begin + count):
                var r = Int(row_ids[i])
                var v = flat[col * n_rows + r]
                if v == v and v < 0.0:
                    has_neg = True
            out.append(has_neg)
    return out^


def _empty_cells(
    ranges: List[InstanceRange],
    colids: List[Int32],
    n_nodes: Int,
    n_sampled_cols: Int,
) -> List[Bool]:
    """Cells with no non-missing value at all: the all-NaN column everywhere,
    and every column of a node with no rows."""
    var out = List[Bool]()
    for nid in range(n_nodes):
        var empty_node = Int(ranges[nid].count) == 0
        for fslot in range(n_sampled_cols):
            var col = Int(colids[nid * n_sampled_cols + fslot])
            out.append(empty_node or col == COL_ALL_NAN)
    return out^

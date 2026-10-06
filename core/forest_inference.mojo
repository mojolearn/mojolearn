# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Experimental shared GPU flat RF/ET inference for the opt-in parallel_groves engine.

Provenance: nvForest v26.08.00 cef3a50da0f74b0015876b9d6d424c86141898dc,
cpp/include/nvforest/detail/infer_kernel/gpu.cuh:100-204: row/tree tasks,
per-grove accumulation, vector leaves and shuffle-down grove reduction.
FOREST-INFER-1: existing RF/ET flat arrays replace nvForest packed node types;
children are local IDs, right=left+1, leaf left=-1. Inputs are row-major.
FOREST-INFER-2: GROVE=False preserves increasing-tree association. GROVE=True
uses EXACTLY32 groves: lane g adds trees g,g+32,..., then halves32->1. The
shared-memory topology (16,8,4,2,1) never consults hardware warp width, unlike
nvForest's device/launch-derived grove count. It can differ from ordered bits.
FOREST-INFER-3: IDENTICAL adds flush both operands/result; final division uses
identical_div. This explicitly defined arithmetic may differ from legacy
unflushed host inference for cancellation/subnormals. No legacy equivalence
or speed claim is implied. FAST/DETERMINISTIC use their existing scalar seams.
FOREST-INFER-4: finite Float32 comparison uses integer keys, preserving ET
subnormals and equating signed zeros. RF flushes ONLY the input feature in
IDENTICAL, matching ensemble/decisiontree/decisiontree.mojo:651; thresholds
remain unflushed. NaN/infinity inputs/model values are refused for this slice.
Host work validates/stages only. All prediction arithmetic/traversal is GPU.
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.sys.compile import is_defined
from std.atomic import Atomic
from core.device_fold import device_exclusive_scan_total
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import ftz, identical_div, GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from std.sys.info import has_apple_gpu_accelerator
from checks.kernel_matrix import TARGET_COLUMN, forest_row_threads_for
from core.apple_fast_tree_experiments import (
    AFT_P02, AFT_P03, AFT_GROVE_BLOCK, AFT_GROVES_PER_BLOCK,
    AFT_ROW_STAGE_BYTES, AFT_ORDERED_ITEMS, AFT_ARGMAX_ROWS,
)
from core.forest_experiments import (T31_PACKED_A, T31_PACKED_B, T32_SHARED_ROWS, T33_COST_SCHEDULE, T34_CHUNK_FOLD, T35_LEAF_REUSE, T36_FINITE_STAGE, T38_FUSED_LABELS, FOREST_CHUNK, forest_chunk_sum, forest_chunk_finish)

# AFCL-T08: two Apple SIMD groups per RF traversal workgroup instead of
# four. Logical 32-tree groves, tree order and all outputs stay unchanged.
# Lower per-group register/shared-memory use applies across neighboring
# row counts; no dataset, feature count or forest shape selects this arm.
# NEVER RUN — PENDING MEASUREMENT; uncompiled/unverified; default OFF.
comptime AFCL_T08 = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_AFCL_T08"]()
)


@always_inline
def _afcl_rf_block[RF_INPUT: Bool]() -> Int:
    comptime if AFCL_T08 and RF_INPUT:
        return 64
    return AFT_GROVE_BLOCK


@always_inline
def forest_add(a: Float32, b: Float32) -> Float32:
    return ftz(ftz(a) + ftz(b))


@always_inline
def finite_key(value: Float32) -> UInt32:
    var bits = bitcast[DType.uint32](value)
    if (bits & UInt32(0x7fffffff)) == 0:
        bits = 0
    return ~bits if (bits & UInt32(0x80000000)) != 0 else bits ^ UInt32(0x80000000)


#: DEVIATION 2963 (lane/forest-groves-cpu-and-speed, 2026-09-17): two
#: kernel candidates that keep the 32-group topology, the 16/8/4/2/1 fold and
#: every FTZ rule and change only how the same bits are fetched.
#: `MOJOLEARN_FOREST_PACKED_NODES` (the existing packed layout) now reads a
#: node's four Int32 words with ONE 16-byte load instead of three field
#: loads; `MOJOLEARN_FOREST_SHARED_ROWS=1` stages a block's four input rows
#: in shared memory once (when `features <= FOREST_SHARED_ROW_CAPACITY`) and
#: every lane's feature reads come from that tile. The values compared are
#: the same words in either case. Both are default off until the pod A/B.
# F13 M3 2026-10-06 MEASURED shared rows, depths4/11/9skew:
# cold B/A0.9331/0.9938/0.9510, repeat1.1091/0.9332/0.9786; quality equal.
# One warmup+score, mixed; FAST opt-in remains OFF. ab-20261006/repairs-54c1f35a5/F13.
comptime FOREST_SHARED_ROWS = T32_SHARED_ROWS or AFT_P03 or is_defined["MOJOLEARN_FOREST_SHARED_ROWS"]() or (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and is_defined["MOJOLEARN_FOREST_FAST_SHARED_ROWS"]())
comptime FOREST_SHARED_ROW_CAPACITY = AFT_ROW_STAGE_BYTES // (4 * AFT_GROVES_PER_BLOCK) if AFT_P03 else 256

#: DEVIATION 2964 (lane/forest-groves-row-schedule, 2026-09-17): the groves
#: engine's SCHEDULE, not its graph. The 32-lane kernels give one row 32
#: threads, so the 32 threads of a thread group walk 32 DIFFERENT trees at
#: once and meet at seven barriers over a shared-memory fold. With
#: `-D MOJOLEARN_FOREST_ROW_THREADS=1` one thread owns an item: lane by lane
#: it sums trees `lane, lane + 32, ...` in that order from +0.0, as the
#: 32-thread kernel's lane does, stores the 32 sums privately, and
#: folds its 32 private sums 16/8/4/2/1 with the same `forest_add` on the
#: same operand pairs. Every addition has the operands and the order it had;
#: only the thread that performs it changed, which is the schedule
#: `core/forest_host_groves.mojo` already runs on the CPU to the GPU's bits.
#: Neighboring threads now walk the SAME tree on adjacent rows. The default
#: is the kernel matrix's row `forest_row_threads_for` (NVIDIA, rows of at
#: most FOREST_ROW_THREADS_MAX_FEATURES floats); `-D MOJOLEARN_FOREST_ROW_THREADS=1`
#: forces it on every column for an A/B, `..._OFF=1` forces it off.
comptime FOREST_ROW_THREADS = forest_row_threads_for[TARGET_COLUMN]()
#: A row of more floats than this keeps the 32-thread kernels: with one row
#: per thread, adjacent threads read 32 different rows, and past two cache
#: lines a row the feature reads stop coalescing (RTX 4090, 2026-09-18:
#: 16 and 28 columns win 1.2x to 1.5x; 54, 90 and 220 columns lose, down to
#: 0.44x at 220). The cut sits between the measured 28 and 54.
comptime FOREST_ROW_THREADS_MAX_FEATURES = 32
#: The row schedule's negative control, default off, never shipped: the 32
#: private sums fold in lane order (a left fold) instead of 16/8/4/2/1.
comptime FOREST_ROW_THREADS_SABOTAGE = is_defined["MOJOLEARN_FOREST_ROW_THREADS_SABOTAGE"]()

#: The resident node layout. Packed is the default since
#: lane/forest-groves-cpu-and-speed (2026-09-17, an L40S A/B);
#: `-D MOJOLEARN_FOREST_SEPARATE_NODES=1` selects the separate-arrays layout,
#: the comparison arm. The old opt-in `MOJOLEARN_FOREST_PACKED_NODES` is
#: accepted and changes nothing. The layout is a device-side cache of the
#: same nodes; the archive arrays, the comparison and the fold are the same.
# F13 packed-layout M3 2026-10-06 MEASURED on same cases: cold B/A
# 1.2171/1.1355/1.2153, repeat0.9942/0.9843/0.9526, refit1.0205/1.0163/1.0359.
# Quality/output survival equal; one warmup+score. Mixed/cold regression:
# no new promotion; existing cross-mode layout default retained, OFF escape above.
comptime FOREST_PACKED_NODES = T31_PACKED_A or (not T31_PACKED_B and not is_defined["MOJOLEARN_FOREST_SEPARATE_NODES"]())


@always_inline
def reached_leaf[xorigin: MutOrigin, xspace: AddressSpace, //, RF_INPUT: Bool, PACKED: Bool = False](
    offsets: MutPointer[Int32, MutAnyOrigin], columns: MutPointer[Int32, MutAnyOrigin],
    thresholds: MutPointer[Float32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    x: MutPointer[Float32, xorigin, address_space=xspace], tree: Int, row: Int, features: Int,
) -> Int:
    comptime if PACKED:
        # nvForest cef3a50d detail/node.hpp:81-175 and evaluate_tree.hpp:44-65.
        # Four Int32 words: threshold-bits OR compact leaf ID, local left,
        # feature, padding, read as one 16-byte vector (DEVIATION 2963; the
        # buffer base is 16-byte aligned and node * 4 words keeps it so).
        # Existing sibling layout and inclusive finite-key comparison remain.
        var base = Int(offsets.unsafe_load(tree))
        var node = base
        var words = columns.unsafe_load[width=4](node * 4)
        var child = Int(words[1])
        while child != -1:
            var value = x.unsafe_load(row * features + Int(words[2]))
            comptime if RF_INPUT:
                value = ftz(value)
            var threshold = bitcast[DType.float32](words[0])
            var go_left = finite_key(value) <= finite_key(threshold)
            node = base + child + (0 if go_left else 1)
            words = columns.unsafe_load[width=4](node * 4)
            child = Int(words[1])
        return Int(words[0])
    var base = Int(offsets.unsafe_load(tree))
    var node = base
    var child = Int(left.unsafe_load(node))
    while child != -1:
        var value = x.unsafe_load(row*features + Int(columns.unsafe_load(node)))
        comptime if RF_INPUT:
            value = ftz(value)
        var go_left = finite_key(value) <= finite_key(thresholds.unsafe_load(node))
        node = base + child + (0 if go_left else 1)
        child = Int(left.unsafe_load(node))
    return node


def forest_ordered_kernel[RF_INPUT: Bool, PACKED: Bool = False](
    offsets: MutPointer[Int32, MutAnyOrigin], columns: MutPointer[Int32, MutAnyOrigin],
    thresholds: MutPointer[Float32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    leaves: MutPointer[Float32, MutAnyOrigin], x: MutPointer[Float32, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin], rows_in: Int32, features_in: Int32, outputs_in: Int32, trees_in: Int32,
):
    var rows = Int(rows_in)
    var features = Int(features_in)
    var outputs = Int(outputs_in)
    var trees = Int(trees_in)
    var worker = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    # P02: neighboring lanes keep contiguous items in each tile plane.
    var item = Int(block_idx.x)*Int(block_dim.x)*AFT_ORDERED_ITEMS+Int(thread_idx.x)
    comptime if AFT_P02:
        var item1 = item + Int(block_dim.x)
        var total0 = Float32(0)
        var total1 = Float32(0)
        if item < rows*outputs:
            for tree in range(trees):
                var node0 = reached_leaf[RF_INPUT, PACKED](offsets,columns,thresholds,left,x,tree,item//outputs,features)
                total0 = forest_add(total0,leaves.unsafe_load(node0*outputs+item%outputs))
                if item1 < rows*outputs:
                    var node1 = reached_leaf[RF_INPUT, PACKED](offsets,columns,thresholds,left,x,tree,item1//outputs,features)
                    total1 = forest_add(total1,leaves.unsafe_load(node1*outputs+item1%outputs))
            output.unsafe_store(item,ftz(identical_div(ftz(total0),Float32(trees))))
            if item1 < rows*outputs:
                output.unsafe_store(item1,ftz(identical_div(ftz(total1),Float32(trees))))
    else:
        if worker < rows*outputs:
            var total = Float32(0)
            for tree in range(trees):
                var node = reached_leaf[RF_INPUT, PACKED](offsets,columns,thresholds,left,x,tree,worker//outputs,features)
                total = forest_add(total,leaves.unsafe_load(node*outputs+worker%outputs))
            output.unsafe_store(worker,ftz(identical_div(ftz(total),Float32(trees))))


@always_inline
def forest_argmax_row(
    scores: MutPointer[Float32, MutAnyOrigin], row: Int, outputs: Int,
) -> Int32:
    """First-index maximum, with the original nonfinite sentinel."""
    var base = row * outputs
    var best = 0
    var best_value = scores.unsafe_load(base)
    if (bitcast[DType.uint32](best_value) & UInt32(0x7f800000)) == UInt32(0x7f800000):
        return Int32(-1)
    for c in range(1, outputs):
        var value = scores.unsafe_load(base + c)
        if (bitcast[DType.uint32](value) & UInt32(0x7f800000)) == UInt32(0x7f800000):
            return Int32(-1)
        if value > best_value:
            best = c
            best_value = value
    return Int32(best)


def forest_argmax_kernel(
    scores: MutPointer[Float32, MutAnyOrigin],
    codes: MutPointer[Int32, MutAnyOrigin], rows_in: Int32, outputs_in: Int32,
):
    # P04: two row planes share a worker; every row still scans all classes.
    var first = Int(block_idx.x) * Int(block_dim.x) * AFT_ARGMAX_ROWS + Int(thread_idx.x)
    @parameter
    for tile in range(AFT_ARGMAX_ROWS):
        var row = first + tile * Int(block_dim.x)
        if row < Int(rows_in):
            codes.unsafe_store(row, forest_argmax_row(scores, row, Int(outputs_in)))


def launch_forest_argmax(ctx: DeviceContext,
    mut scores: DeviceBuffer[DType.float32], mut codes: DeviceBuffer[DType.int32],
    rows: Int, outputs: Int) raises:
    if rows > 0:
        ctx.enqueue_function[forest_argmax_kernel](
            scores.unsafe_ptr(), codes.unsafe_ptr(), Int32(rows), Int32(outputs),
            grid_dim=(rows + 128*AFT_ARGMAX_ROWS - 1) // (128*AFT_ARGMAX_ROWS), block_dim=128,
        )


def forest_grove32_kernel[RF_INPUT: Bool, PACKED: Bool = False](
    offsets: MutPointer[Int32, MutAnyOrigin], columns: MutPointer[Int32, MutAnyOrigin],
    thresholds: MutPointer[Float32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    leaves: MutPointer[Float32, MutAnyOrigin], x: MutPointer[Float32, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin], rows_in: Int32, features_in: Int32, outputs_in: Int32, trees_in: Int32,
):
    comptime traversal_tpb = _afcl_rf_block[RF_INPUT]()
    comptime traversal_rows = traversal_tpb // 32
    var rows = Int(rows_in)
    var features = Int(features_in)
    var outputs = Int(outputs_in)
    var trees = Int(trees_in)
    # P01: a fixed number of logical 32-lane groves; all threads hit barriers.
    var tid = Int(thread_idx.x)
    var lane = tid%32
    var item = Int(block_idx.x)*traversal_rows+tid//32
    var total = Float32(0)
    var tiled = False
    comptime if FOREST_SHARED_ROWS:
        # DEVIATION 2963: with one output an item is a row, so the block's
        # four rows tile into shared memory; every thread hits the barrier.
        var xs = stack_allocation[traversal_rows*FOREST_SHARED_ROW_CAPACITY,Float32,address_space=AddressSpace.SHARED]()
        if outputs == 1 and features <= FOREST_SHARED_ROW_CAPACITY:
            tiled = True
            var row0 = Int(block_idx.x)*traversal_rows
            var count = traversal_rows*features
            var i = tid
            while i < count:
                var r = row0+i//features
                xs[unsafe_offset=i] = x.unsafe_load(r*features+i%features) if r < rows else Float32(0)
                i += traversal_tpb
            barrier()
            if item < rows:
                var tree = lane
                while tree < trees:
                    var node = reached_leaf[RF_INPUT, PACKED](offsets,columns,thresholds,left,xs,tree,tid//32,features)
                    total = forest_add(total,leaves.unsafe_load(node))
                    tree += 32
    if not tiled and item < rows*outputs:
        var tree = lane
        while tree < trees:
            var node = reached_leaf[RF_INPUT, PACKED](offsets,columns,thresholds,left,x,tree,item//outputs,features)
            total = forest_add(total,leaves.unsafe_load(node*outputs+item%outputs))
            tree += 32
    var sums = stack_allocation[traversal_tpb,Float32,address_space=AddressSpace.SHARED]()
    sums[unsafe_offset=tid] = total
    barrier()
    var step = 16
    while step > 0:
        if lane < step:
            sums[unsafe_offset=tid] = forest_add(sums[unsafe_offset=tid],sums[unsafe_offset=tid+step])
        barrier()
        step //= 2
    if lane == 0 and item < rows*outputs:
        output.unsafe_store(item,ftz(identical_div(ftz(sums[unsafe_offset=tid]),Float32(trees))))



def vector_groves_for(outputs: Int) -> Bool:
    """Default vector traversal for 2–8 outputs; force-scalar is diagnostic.

    Same fixed32 output graph, qualified against scalar on Metal FAST and
    CUDA IDENTICAL large resident fixtures. The public engine stays opt-in.
    MOJOLEARN_FOREST_VECTOR_GROVES is no longer required; old builds passing
    it retain the same route. Outputs1/>8 retain the bounded scalar fallback.
    """
    return (not is_defined["MOJOLEARN_FOREST_SCALAR_GROVES"]()
            and outputs >= 2 and outputs <= 8)


def forest_vector_grove32_kernel[RF_INPUT: Bool, OUTPUT_CAPACITY: Int, PACKED: Bool = False](
    offsets: MutPointer[Int32, MutAnyOrigin], columns: MutPointer[Int32, MutAnyOrigin],
    thresholds: MutPointer[Float32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    leaves: MutPointer[Float32, MutAnyOrigin], x: MutPointer[Float32, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin], rows_in: Int32, features_in: Int32, outputs_in: Int32, trees_in: Int32,
):
    """Reuse nvForest's vector-leaf task: evaluate once, then all outputs.

    Source pin/path in module header, gpu.cuh:139-158: evaluate_tree produces
    one leaf ID, then the vector-output loop updates each class for that
    task/grove. Keep our existing fixed32 logical groves and addition order
    per output; only independent output work shares traversal. Capacity2/4/8
    bounds register use and shared scratch (at most4KiB). More outputs keep
    the existing scalar-output kernel. No row*tree global scratch is added.
    """
    comptime traversal_tpb = _afcl_rf_block[RF_INPUT]()
    comptime traversal_rows = traversal_tpb // 32
    var rows = Int(rows_in)
    var features = Int(features_in)
    var outputs = Int(outputs_in)
    var trees = Int(trees_in)
    var tid = Int(thread_idx.x)
    var lane = tid%32
    var row = Int(block_idx.x)*traversal_rows+tid//32
    var totals = stack_allocation[OUTPUT_CAPACITY,Float32]()
    @parameter
    for c in range(OUTPUT_CAPACITY):
        totals[unsafe_offset=c] = Float32(0)
    var tiled = False
    comptime if FOREST_SHARED_ROWS:
        # DEVIATION 2963: the block's four rows tiled into shared memory.
        var xs = stack_allocation[traversal_rows*FOREST_SHARED_ROW_CAPACITY,Float32,address_space=AddressSpace.SHARED]()
        if features <= FOREST_SHARED_ROW_CAPACITY:
            tiled = True
            var row0 = Int(block_idx.x)*traversal_rows
            var count = traversal_rows*features
            var i = tid
            while i < count:
                var r = row0+i//features
                xs[unsafe_offset=i] = x.unsafe_load(r*features+i%features) if r < rows else Float32(0)
                i += traversal_tpb
            barrier()
            if row < rows:
                var tree = lane
                while tree < trees:
                    var node = reached_leaf[RF_INPUT, PACKED](offsets,columns,thresholds,left,xs,tree,tid//32,features)
                    @parameter
                    for c in range(OUTPUT_CAPACITY):
                        if c < outputs:
                            totals[unsafe_offset=c] = forest_add(totals[unsafe_offset=c],leaves.unsafe_load(node*outputs+c))
                    tree += 32
    if not tiled and row < rows:
        var tree = lane
        while tree < trees:
            var node = reached_leaf[RF_INPUT, PACKED](offsets,columns,thresholds,left,x,tree,row,features)
            @parameter
            for c in range(OUTPUT_CAPACITY):
                if c < outputs:
                    totals[unsafe_offset=c] = forest_add(totals[unsafe_offset=c],leaves.unsafe_load(node*outputs+c))
            tree += 32
    var sums = stack_allocation[traversal_tpb*OUTPUT_CAPACITY,Float32,address_space=AddressSpace.SHARED]()
    @parameter
    for c in range(OUTPUT_CAPACITY):
        sums[unsafe_offset=c*traversal_tpb+tid] = totals[unsafe_offset=c]
    barrier()
    var step = 16
    while step > 0:
        if lane < step:
            @parameter
            for c in range(OUTPUT_CAPACITY):
                sums[unsafe_offset=c*traversal_tpb+tid] = forest_add(sums[unsafe_offset=c*traversal_tpb+tid],sums[unsafe_offset=c*traversal_tpb+tid+step])
        barrier()
        step //= 2
    if lane == 0 and row < rows:
        @parameter
        for c in range(OUTPUT_CAPACITY):
            if c < outputs:
                output.unsafe_store(row*outputs+c,ftz(identical_div(ftz(sums[unsafe_offset=c*traversal_tpb+tid]),Float32(trees))))


def forest_grove32_row_kernel[RF_INPUT: Bool, PACKED: Bool = False](
    offsets: MutPointer[Int32, MutAnyOrigin], columns: MutPointer[Int32, MutAnyOrigin],
    thresholds: MutPointer[Float32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    leaves: MutPointer[Float32, MutAnyOrigin], x: MutPointer[Float32, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin], rows_in: Int32, features_in: Int32, outputs_in: Int32, trees_in: Int32,
):
    """DEVIATION 2964: `forest_grove32_kernel`'s graph, one thread per item.

    Lane `l` sums trees `l, l + 32, ...` ascending from +0.0 and the 32 sums
    fold 16/8/4/2/1, as there; the sums are thread-private, so there is no
    shared memory and no barrier.
    """
    var rows = Int(rows_in)
    var features = Int(features_in)
    var outputs = Int(outputs_in)
    var trees = Int(trees_in)
    var item = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if item < rows*outputs:
        var sums = stack_allocation[32,Float32]()
        for lane in range(32):
            sums[unsafe_offset=lane] = Float32(0)
        var row = item//outputs
        var out = item%outputs
        # Lane by lane, so one running total lives in a register and each
        # private sum is stored once (a per-tree read-modify-write of the
        # private array measured slower at eight outputs on the RTX 4090).
        for lane in range(32):
            var total = Float32(0)
            var tree = lane
            while tree < trees:
                var node = reached_leaf[RF_INPUT, PACKED](offsets,columns,thresholds,left,x,tree,row,features)
                total = forest_add(total,leaves.unsafe_load(node*outputs+out))
                tree += 32
            sums[unsafe_offset=lane] = total
        comptime if FOREST_ROW_THREADS_SABOTAGE:
            for lane in range(1, 32):
                sums[unsafe_offset=0] = forest_add(sums[unsafe_offset=0],sums[unsafe_offset=lane])
        else:
            var step = 16
            while step > 0:
                for lane in range(step):
                    sums[unsafe_offset=lane] = forest_add(sums[unsafe_offset=lane],sums[unsafe_offset=lane+step])
                step //= 2
        output.unsafe_store(item,ftz(identical_div(ftz(sums[unsafe_offset=0]),Float32(trees))))


def forest_vector_grove32_row_kernel[RF_INPUT: Bool, OUTPUT_CAPACITY: Int, PACKED: Bool = False](
    offsets: MutPointer[Int32, MutAnyOrigin], columns: MutPointer[Int32, MutAnyOrigin],
    thresholds: MutPointer[Float32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    leaves: MutPointer[Float32, MutAnyOrigin], x: MutPointer[Float32, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin], rows_in: Int32, features_in: Int32, outputs_in: Int32, trees_in: Int32,
):
    """DEVIATION 2964: `forest_vector_grove32_kernel`'s graph, one thread per
    row; one traversal per tree serves every output, as there."""
    var rows = Int(rows_in)
    var features = Int(features_in)
    var outputs = Int(outputs_in)
    var trees = Int(trees_in)
    var row = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if row < rows:
        var sums = stack_allocation[32*OUTPUT_CAPACITY,Float32]()
        for i in range(32*OUTPUT_CAPACITY):
            sums[unsafe_offset=i] = Float32(0)
        # Lane by lane: OUTPUT_CAPACITY running totals in registers, each
        # private sum stored once (see forest_grove32_row_kernel).
        for lane in range(32):
            var totals = SIMD[DType.float32, OUTPUT_CAPACITY](0)
            var tree = lane
            while tree < trees:
                var node = reached_leaf[RF_INPUT, PACKED](offsets,columns,thresholds,left,x,tree,row,features)
                @parameter
                for c in range(OUTPUT_CAPACITY):
                    if c < outputs:
                        totals[c] = forest_add(totals[c],leaves.unsafe_load(node*outputs+c))
                tree += 32
            @parameter
            for c in range(OUTPUT_CAPACITY):
                sums[unsafe_offset=c*32+lane] = totals[c]
        comptime if FOREST_ROW_THREADS_SABOTAGE:
            for lane in range(1, 32):
                @parameter
                for c in range(OUTPUT_CAPACITY):
                    sums[unsafe_offset=c*32] = forest_add(sums[unsafe_offset=c*32],sums[unsafe_offset=c*32+lane])
        else:
            var step = 16
            while step > 0:
                for lane in range(step):
                    @parameter
                    for c in range(OUTPUT_CAPACITY):
                        sums[unsafe_offset=c*32+lane] = forest_add(sums[unsafe_offset=c*32+lane],sums[unsafe_offset=c*32+lane+step])
                step //= 2
        @parameter
        for c in range(OUTPUT_CAPACITY):
            if c < outputs:
                output.unsafe_store(row*outputs+c,ftz(identical_div(ftz(sums[unsafe_offset=c*32]),Float32(trees))))


def launch_forest_inference[RF_INPUT: Bool, GROVE: Bool, PACKED: Bool = False](
    ctx: DeviceContext,
    mut doff: DeviceBuffer[DType.int32], mut dcol: DeviceBuffer[DType.int32],
    mut dthr: DeviceBuffer[DType.float32], mut dleft: DeviceBuffer[DType.int32],
    mut dleaf: DeviceBuffer[DType.float32], mut dx: DeviceBuffer[DType.float32],
    mut dout: DeviceBuffer[DType.float32], n_rows: Int, n_features: Int,
    n_outputs: Int, trees: Int,
) raises:
    """One shared enqueue dispatcher for transient and resident model owners.

    Caller validates all shapes/indices/finite values, owns buffer lifetimes
    through completion and performs required readback/synchronization. No
    allocations or device drains occur here. Empty row batches enqueue nothing.
    """
    comptime traversal_tpb = _afcl_rf_block[RF_INPUT]()
    if n_rows == 0:
        return
    comptime if T35_LEAF_REUSE or T36_FINITE_STAGE:
        _launch_forest_leaf_reuse[RF_INPUT, GROVE, PACKED](ctx, doff, dcol, dthr, dleft, dleaf, dx, dout, n_rows, n_features, n_outputs, trees)
        return
    comptime if T34_CHUNK_FOLD:
        ctx.enqueue_function[forest_chunk_kernel[RF_INPUT, PACKED]](
            doff.unsafe_ptr(), dcol.unsafe_ptr(), dthr.unsafe_ptr(), dleft.unsafe_ptr(),
            dleaf.unsafe_ptr(), dx.unsafe_ptr(), dout.unsafe_ptr(), Int32(n_rows), Int32(n_features), Int32(n_outputs), Int32(trees),
            grid_dim=(n_rows*n_outputs+127)//128, block_dim=128)
        return
    comptime if GROVE:
        comptime if T33_COST_SCHEDULE:
            # Compare feature cache lines with live logical tree lanes. All
            # neighboring shapes use this work/cache estimate, never a board row.
            var feature_lines = (n_features * 4 + 63) // 64
            if feature_lines <= max(1, min(trees, 32) // 8):
                _launch_grove_rows[RF_INPUT, PACKED](ctx, doff, dcol, dthr, dleft, dleaf, dx, dout, n_rows, n_features, n_outputs, trees)
            else:
                _launch_grove_lanes[RF_INPUT, PACKED](ctx, doff, dcol, dthr, dleft, dleaf, dx, dout, n_rows, n_features, n_outputs, trees)
            return
        comptime if FOREST_ROW_THREADS:
            if n_features <= FOREST_ROW_THREADS_MAX_FEATURES:
                _launch_grove_rows[RF_INPUT, PACKED](ctx, doff, dcol, dthr, dleft, dleaf, dx, dout, n_rows, n_features, n_outputs, trees)
                return
        _launch_grove_lanes[RF_INPUT, PACKED](ctx, doff, dcol, dthr, dleft, dleaf, dx, dout, n_rows, n_features, n_outputs, trees)
    else:
        ctx.enqueue_function[forest_ordered_kernel[RF_INPUT,PACKED]](
            doff.unsafe_ptr(),dcol.unsafe_ptr(),dthr.unsafe_ptr(),dleft.unsafe_ptr(),
            dleaf.unsafe_ptr(),dx.unsafe_ptr(),dout.unsafe_ptr(),Int32(n_rows),Int32(n_features),Int32(n_outputs),Int32(trees),
            grid_dim=(n_rows*n_outputs+128*AFT_ORDERED_ITEMS-1)//(128*AFT_ORDERED_ITEMS),block_dim=128,
        )


def _launch_grove_lanes[RF_INPUT: Bool, PACKED: Bool](
    ctx: DeviceContext,
    mut doff: DeviceBuffer[DType.int32], mut dcol: DeviceBuffer[DType.int32],
    mut dthr: DeviceBuffer[DType.float32], mut dleft: DeviceBuffer[DType.int32],
    mut dleaf: DeviceBuffer[DType.float32], mut dx: DeviceBuffer[DType.float32],
    mut dout: DeviceBuffer[DType.float32], n_rows: Int, n_features: Int,
    n_outputs: Int, trees: Int,
) raises:
    """The 32-thread kernels: a row's lanes across a thread group, the fold in shared memory."""
    comptime traversal_tpb = _afcl_rf_block[RF_INPUT]()
    comptime traversal_rows = traversal_tpb // 32
    if vector_groves_for(n_outputs):
        if n_outputs <= 2:
            ctx.enqueue_function[forest_vector_grove32_kernel[RF_INPUT,2,PACKED]](
                doff.unsafe_ptr(),dcol.unsafe_ptr(),dthr.unsafe_ptr(),dleft.unsafe_ptr(),
                dleaf.unsafe_ptr(),dx.unsafe_ptr(),dout.unsafe_ptr(),Int32(n_rows),Int32(n_features),Int32(n_outputs),Int32(trees),
                grid_dim=(n_rows+traversal_rows-1)//traversal_rows,block_dim=traversal_tpb,
            )
        elif n_outputs <= 4:
            ctx.enqueue_function[forest_vector_grove32_kernel[RF_INPUT,4,PACKED]](
                doff.unsafe_ptr(),dcol.unsafe_ptr(),dthr.unsafe_ptr(),dleft.unsafe_ptr(),
                dleaf.unsafe_ptr(),dx.unsafe_ptr(),dout.unsafe_ptr(),Int32(n_rows),Int32(n_features),Int32(n_outputs),Int32(trees),
                grid_dim=(n_rows+traversal_rows-1)//traversal_rows,block_dim=traversal_tpb,
            )
        else:
            ctx.enqueue_function[forest_vector_grove32_kernel[RF_INPUT,8,PACKED]](
                doff.unsafe_ptr(),dcol.unsafe_ptr(),dthr.unsafe_ptr(),dleft.unsafe_ptr(),
                dleaf.unsafe_ptr(),dx.unsafe_ptr(),dout.unsafe_ptr(),Int32(n_rows),Int32(n_features),Int32(n_outputs),Int32(trees),
                grid_dim=(n_rows+traversal_rows-1)//traversal_rows,block_dim=traversal_tpb,
            )
    else:
        ctx.enqueue_function[forest_grove32_kernel[RF_INPUT,PACKED]](
            doff.unsafe_ptr(),dcol.unsafe_ptr(),dthr.unsafe_ptr(),dleft.unsafe_ptr(),
            dleaf.unsafe_ptr(),dx.unsafe_ptr(),dout.unsafe_ptr(),Int32(n_rows),Int32(n_features),Int32(n_outputs),Int32(trees),
            grid_dim=(n_rows*n_outputs+traversal_rows-1)//traversal_rows,block_dim=traversal_tpb,
        )


def _launch_grove_rows[RF_INPUT: Bool, PACKED: Bool](
    ctx: DeviceContext,
    mut doff: DeviceBuffer[DType.int32], mut dcol: DeviceBuffer[DType.int32],
    mut dthr: DeviceBuffer[DType.float32], mut dleft: DeviceBuffer[DType.int32],
    mut dleaf: DeviceBuffer[DType.float32], mut dx: DeviceBuffer[DType.float32],
    mut dout: DeviceBuffer[DType.float32], n_rows: Int, n_features: Int,
    n_outputs: Int, trees: Int,
) raises:
    """DEVIATION 2964: one thread per row (vector leaves) or per item."""
    comptime traversal_tpb = _afcl_rf_block[RF_INPUT]()
    comptime traversal_rows = traversal_tpb // 32
    if vector_groves_for(n_outputs):
        if n_outputs <= 2:
            ctx.enqueue_function[forest_vector_grove32_row_kernel[RF_INPUT,2,PACKED]](
                doff.unsafe_ptr(),dcol.unsafe_ptr(),dthr.unsafe_ptr(),dleft.unsafe_ptr(),
                dleaf.unsafe_ptr(),dx.unsafe_ptr(),dout.unsafe_ptr(),Int32(n_rows),Int32(n_features),Int32(n_outputs),Int32(trees),
                grid_dim=(n_rows+traversal_tpb-1)//traversal_tpb,block_dim=traversal_tpb,
            )
        elif n_outputs <= 4:
            ctx.enqueue_function[forest_vector_grove32_row_kernel[RF_INPUT,4,PACKED]](
                doff.unsafe_ptr(),dcol.unsafe_ptr(),dthr.unsafe_ptr(),dleft.unsafe_ptr(),
                dleaf.unsafe_ptr(),dx.unsafe_ptr(),dout.unsafe_ptr(),Int32(n_rows),Int32(n_features),Int32(n_outputs),Int32(trees),
                grid_dim=(n_rows+traversal_tpb-1)//traversal_tpb,block_dim=traversal_tpb,
            )
        else:
            ctx.enqueue_function[forest_vector_grove32_row_kernel[RF_INPUT,8,PACKED]](
                doff.unsafe_ptr(),dcol.unsafe_ptr(),dthr.unsafe_ptr(),dleft.unsafe_ptr(),
                dleaf.unsafe_ptr(),dx.unsafe_ptr(),dout.unsafe_ptr(),Int32(n_rows),Int32(n_features),Int32(n_outputs),Int32(trees),
                grid_dim=(n_rows+traversal_tpb-1)//traversal_tpb,block_dim=traversal_tpb,
            )
    else:
        ctx.enqueue_function[forest_grove32_row_kernel[RF_INPUT,PACKED]](
            doff.unsafe_ptr(),dcol.unsafe_ptr(),dthr.unsafe_ptr(),dleft.unsafe_ptr(),
            dleaf.unsafe_ptr(),dx.unsafe_ptr(),dout.unsafe_ptr(),Int32(n_rows),Int32(n_features),Int32(n_outputs),Int32(trees),
            grid_dim=(n_rows*n_outputs+traversal_tpb-1)//traversal_tpb,block_dim=traversal_tpb,
        )


def require_finite(values: List[Float32]) raises:
    for value in values:
        if (bitcast[DType.uint32](value) & UInt32(0x7f800000)) == UInt32(0x7f800000):
            raise Error("forest inference prototype requires finite Float32 values")


comptime FOREST_FINITE_TPB = 256


def forest_nonfinite_kernel(v: MutPointer[Float32, MutAnyOrigin], n: Int32, flag: MutPointer[Int32, MutAnyOrigin]):
    """The finiteness predicate on the device, one thread a value: any
    exponent field of all ones (inf or NaN) sets flag[0] (every writer
    stores the same 1, so the flag has no order)."""
    var i = Int(block_idx.x) * FOREST_FINITE_TPB + Int(thread_idx.x)
    if i < Int(n):
        if (bitcast[DType.uint32](v.unsafe_load(i)) & UInt32(0x7f800000)) == UInt32(0x7f800000):
            flag.unsafe_store(0, Int32(1))


def device_all_finite(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int) raises -> Bool:
    """True when the first n values of `buf` are finite: the scan runs on
    the device and one int comes back (cpu-gpu-cleanup: the forest predict
    paths scanned their inputs and outputs on host threads)."""
    return device_ptr_all_finite(ctx, buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), n)


def device_ptr_all_finite(ctx: DeviceContext, v: MutPointer[Float32, MutAnyOrigin], n: Int) raises -> Bool:
    """`device_all_finite` on n values at device address `v` (a slice of a
    larger device buffer)."""
    if n <= 0:
        return True
    var flag = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(flag, Int32(0))
    ctx.enqueue_function[forest_nonfinite_kernel](
        v, Int32(n),
        flag.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        grid_dim=(n + FOREST_FINITE_TPB - 1) // FOREST_FINITE_TPB, block_dim=FOREST_FINITE_TPB,
    )
    var h = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=flag)
    ctx.synchronize()
    var ok = h.unsafe_ptr().unsafe_load(0) == Int32(0)
    _ = h^
    _ = flag^
    return ok


def _forest_shape_checks(
    offsets: List[Int32], columns: List[Int32], thresholds: List[Float32],
    left: List[Int32], leaves: List[Float32], x: List[Float32],
    rows: Int, features: Int, outputs: Int,
) raises:
    """The constant-time shape checks (lengths, ranges, first and last
    offset); the per-node checks are `forest_validate_device` on a GPU route
    and `validate_flat_forest_host` on the host."""
    if rows < 0 or features < 1 or outputs < 1 or len(offsets) < 2:
        raise Error("forest inference requires rows>=0, features/outputs/trees>=1")
    if features > 2147483647 or outputs > 2147483647 or len(offsets)-1 > 2147483647:
        raise Error("forest inference dimensions exceed Int32 range")
    if rows > 2147483647//features or rows > 2147483647//outputs:
        raise Error("forest inference input/output element count exceeds Int32 range")
    var nodes = len(columns)
    if nodes < 1 or nodes > 2147483647//outputs:
        raise Error("forest inference leaf element count exceeds Int32 range")
    if len(thresholds) != nodes or len(left) != nodes or len(leaves) != nodes*outputs or len(x) != rows*features:
        raise Error("forest inference flat array shape mismatch")
    if offsets[0] != 0 or Int(offsets[len(offsets)-1]) != nodes:
        raise Error("forest inference offsets must cover all nodes")


def validate_flat_forest_host(
    offsets: List[Int32], columns: List[Int32], thresholds: List[Float32],
    left: List[Int32], leaves: List[Float32], x: List[Float32],
    rows: Int, features: Int, outputs: Int,
) raises:
    """The host validator (CPU-only callers and the bench harness); the GPU
    predict validates with `forest_validate_device` instead."""
    _forest_shape_checks(offsets,columns,thresholds,left,leaves,x,rows,features,outputs)
    var nodes = len(columns)
    require_finite(thresholds)
    require_finite(leaves)
    # the input rows are scanned on the device where they land
    # (forest_predict_gpu), not here on host threads
    _ = len(x)
    for t in range(len(offsets)-1):
        var base = Int(offsets[t])
        var end = Int(offsets[t+1])
        if base < 0 or end <= base or end > nodes:
            raise Error("forest inference offsets must be strictly increasing")
        var seen = List[UInt8](length=end-base,fill=UInt8(0))
        var pending: List[Int] = [0]
        var visited = 0
        while len(pending) > 0:
            var local = pending.pop()
            if seen[local] != 0:
                raise Error("forest inference requires acyclic trees without shared children")
            seen[local] = 1
            visited += 1
            var node = base+local
            var child = Int(left[node])
            if child != -1:
                if child < 0 or child+1 >= end-base or columns[node] < 0 or Int(columns[node]) >= features:
                    raise Error("forest inference feature/child index out of bounds")
                pending.append(child)
                pending.append(child+1)
        if visited != end-base:
            raise Error("forest inference tree contains unreachable nodes")


# ---- model validation on the device (lane cpu3-core) ----------------------
# The GPU predict used to walk every tree on the host (a stack DFS over all
# nodes) before each launch. The same predicate, in parallel: every node's
# child pair is in bounds and in its own tree; every non-root node has
# exactly one parent and the root none; every node reaches its root by
# parent pointers (pointer jumping, log2(nodes) + 1 rounds). Flags:
# 0 offsets, 1 bounds, 2 cycle or shared child, 3 unreachable, 4 non-finite.
comptime FOREST_VALIDATE_FLAGS = 5


@always_inline
def _forest_tree_of(off: MutPointer[Int32, MutAnyOrigin], trees: Int, i: Int) -> Int:
    """The tree t with off[t] <= i < off[t + 1] (binary search over sorted
    offsets; the callers re-check the bounds, so bad offsets stay safe)."""
    var lo = 0
    var hi = trees
    while hi - lo > 1:
        var mid = (lo + hi) // 2
        if Int(off.unsafe_load(mid)) <= i:
            lo = mid
        else:
            hi = mid
    return lo


def forest_offsets_check_kernel(
    off: MutPointer[Int32, MutAnyOrigin], trees_in: Int32, nodes_in: Int32,
    flag: MutPointer[Int32, MutAnyOrigin],
):
    """One thread a tree: offsets strictly increasing inside [0, nodes]."""
    var t = Int(block_idx.x) * FOREST_FINITE_TPB + Int(thread_idx.x)
    if t < Int(trees_in):
        var base = Int(off.unsafe_load(t))
        var end = Int(off.unsafe_load(t + 1))
        if base < 0 or end <= base or end > Int(nodes_in):
            flag.unsafe_store(0, Int32(1))


def forest_edges_kernel(
    off: MutPointer[Int32, MutAnyOrigin], trees_in: Int32, nodes_in: Int32,
    col: MutPointer[Int32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    features_in: Int32, indeg: MutPointer[Int32, MutAnyOrigin],
    parent: MutPointer[Int32, MutAnyOrigin], flag: MutPointer[Int32, MutAnyOrigin],
):
    """One thread a node: its child pair in bounds (flag 1), each child's
    in-degree counted and its parent recorded (exact integer atomics; a
    parent slot written twice is a shared child, which flag 2 reports)."""
    var i = Int(block_idx.x) * FOREST_FINITE_TPB + Int(thread_idx.x)
    if i >= Int(nodes_in):
        return
    var t = _forest_tree_of(off, Int(trees_in), i)
    var base = Int(off.unsafe_load(t))
    var end = Int(off.unsafe_load(t + 1))
    if i < base or i >= end:
        return
    var child = Int(left.unsafe_load(i))
    if child == -1:
        return
    var c = Int(col.unsafe_load(i))
    if child < 0 or child + 1 >= end - base or c < 0 or c >= Int(features_in):
        flag.unsafe_store(1, Int32(1))
        return
    _ = Atomic.fetch_add(indeg.unsafe_offset(base + child), Int32(1))
    _ = Atomic.fetch_add(indeg.unsafe_offset(base + child + 1), Int32(1))
    parent.unsafe_store(base + child, Int32(i))
    parent.unsafe_store(base + child + 1, Int32(i))


def forest_parent_init_kernel(
    off: MutPointer[Int32, MutAnyOrigin], trees_in: Int32, nodes_in: Int32,
    indeg: MutPointer[Int32, MutAnyOrigin], parent: MutPointer[Int32, MutAnyOrigin],
    flag: MutPointer[Int32, MutAnyOrigin],
):
    """One thread a node: the in-degree rule (root 0, others exactly 1:
    flag 2 for a second parent, flag 3 for none); a root or a rejected node
    points at itself, so pointer jumping never leaves its tree."""
    var i = Int(block_idx.x) * FOREST_FINITE_TPB + Int(thread_idx.x)
    if i >= Int(nodes_in):
        return
    var t = _forest_tree_of(off, Int(trees_in), i)
    var base = Int(off.unsafe_load(t))
    var d = indeg.unsafe_load(i)
    if i == base:
        if d != Int32(0):
            flag.unsafe_store(2, Int32(1))
        parent.unsafe_store(i, Int32(i))
    elif d != Int32(1):
        if d == Int32(0):
            flag.unsafe_store(3, Int32(1))
        else:
            flag.unsafe_store(2, Int32(1))
        parent.unsafe_store(i, Int32(i))


def forest_parent_jump_kernel(nodes_in: Int32, parent: MutPointer[Int32, MutAnyOrigin]):
    """One round of pointer jumping, in place: parent[i] = parent[parent[i]].
    Every value stays an ancestor of i (or i's own cycle), so the in-place
    races only speed it up; the final predicate does not depend on them."""
    var i = Int(block_idx.x) * FOREST_FINITE_TPB + Int(thread_idx.x)
    if i < Int(nodes_in):
        var p = Int(parent.unsafe_load(i))
        parent.unsafe_store(i, parent.unsafe_load(p))


def forest_reach_check_kernel(
    off: MutPointer[Int32, MutAnyOrigin], trees_in: Int32, nodes_in: Int32,
    parent: MutPointer[Int32, MutAnyOrigin], flag: MutPointer[Int32, MutAnyOrigin],
):
    """One thread a node: after the jumps every node points at its root, or
    it sits on a cycle cut off from the root (flag 3, as the host walk's
    unreachable-node error)."""
    var i = Int(block_idx.x) * FOREST_FINITE_TPB + Int(thread_idx.x)
    if i >= Int(nodes_in):
        return
    var t = _forest_tree_of(off, Int(trees_in), i)
    var base = Int(off.unsafe_load(t))
    if Int(parent.unsafe_load(i)) != base:
        flag.unsafe_store(3, Int32(1))


def forest_validate_device(
    ctx: DeviceContext, mut doff: DeviceBuffer[DType.int32], mut dcol: DeviceBuffer[DType.int32],
    mut dthr: DeviceBuffer[DType.float32], mut dleft: DeviceBuffer[DType.int32],
    mut dleaf: DeviceBuffer[DType.float32], trees: Int, nodes: Int, features: Int, outputs: Int,
) raises:
    """`validate_flat_forest_host`'s per-node checks on the uploaded model:
    every step a parallel kernel, one small flag vector read back."""
    var flag = ctx.enqueue_create_buffer[DType.int32](FOREST_VALIDATE_FLAGS)
    ctx.enqueue_memset(flag, Int32(0))
    var indeg = ctx.enqueue_create_buffer[DType.int32](nodes)
    ctx.enqueue_memset(indeg, Int32(0))
    var parent = ctx.enqueue_create_buffer[DType.int32](nodes)
    var fp = flag.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var op = doff.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var ip = indeg.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var pp = parent.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var node_grid = (nodes + FOREST_FINITE_TPB - 1) // FOREST_FINITE_TPB
    ctx.enqueue_function[forest_nonfinite_kernel](
        dthr.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(nodes), fp + 4,
        grid_dim=node_grid, block_dim=FOREST_FINITE_TPB,
    )
    ctx.enqueue_function[forest_nonfinite_kernel](
        dleaf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(nodes * outputs), fp + 4,
        grid_dim=(nodes * outputs + FOREST_FINITE_TPB - 1) // FOREST_FINITE_TPB, block_dim=FOREST_FINITE_TPB,
    )
    ctx.enqueue_function[forest_offsets_check_kernel](
        op, Int32(trees), Int32(nodes), fp,
        grid_dim=(trees + FOREST_FINITE_TPB - 1) // FOREST_FINITE_TPB, block_dim=FOREST_FINITE_TPB,
    )
    ctx.enqueue_function[forest_edges_kernel](
        op, Int32(trees), Int32(nodes),
        dcol.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        dleft.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        Int32(features), ip, pp, fp,
        grid_dim=node_grid, block_dim=FOREST_FINITE_TPB,
    )
    ctx.enqueue_function[forest_parent_init_kernel](
        op, Int32(trees), Int32(nodes), ip, pp, fp,
        grid_dim=node_grid, block_dim=FOREST_FINITE_TPB,
    )
    # ceil(log2(nodes)) + 1 rounds take every node of a valid tree to its root
    var span = 1
    var rounds = 1
    while span < nodes:
        span *= 2
        rounds += 1
    var r = 0
    while r < rounds:
        ctx.enqueue_function[forest_parent_jump_kernel](
            Int32(nodes), pp, grid_dim=node_grid, block_dim=FOREST_FINITE_TPB,
        )
        r += 1
    ctx.enqueue_function[forest_reach_check_kernel](
        op, Int32(trees), Int32(nodes), pp, fp,
        grid_dim=node_grid, block_dim=FOREST_FINITE_TPB,
    )
    var h = ctx.enqueue_create_host_buffer[DType.int32](FOREST_VALIDATE_FLAGS)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=flag)
    ctx.synchronize()
    var f_off = h.unsafe_ptr().unsafe_load(0)
    var f_bounds = h.unsafe_ptr().unsafe_load(1)
    var f_cycle = h.unsafe_ptr().unsafe_load(2)
    var f_reach = h.unsafe_ptr().unsafe_load(3)
    var f_fin = h.unsafe_ptr().unsafe_load(4)
    _ = h^
    _ = flag^
    _ = indeg^
    _ = parent^
    if f_fin != Int32(0):
        raise Error("forest inference prototype requires finite Float32 values")
    if f_off != Int32(0):
        raise Error("forest inference offsets must be strictly increasing")
    if f_bounds != Int32(0):
        raise Error("forest inference feature/child index out of bounds")
    if f_cycle != Int32(0):
        raise Error("forest inference requires acyclic trees without shared children")
    if f_reach != Int32(0):
        raise Error("forest inference tree contains unreachable nodes")


def forest_validate_lists_device(
    ctx: DeviceContext, offsets: List[Int32], columns: List[Int32],
    thresholds: List[Float32], left: List[Int32], leaves: List[Float32],
    features: Int, outputs: Int,
) raises:
    """Upload a flat forest to `ctx` and run `forest_validate_device` on it
    (for routes that partition the model before their own uploads)."""
    var nodes = len(columns)
    var doff = ctx.enqueue_create_buffer[DType.int32](len(offsets))
    var dcol = ctx.enqueue_create_buffer[DType.int32](nodes)
    var dthr = ctx.enqueue_create_buffer[DType.float32](nodes)
    var dleft = ctx.enqueue_create_buffer[DType.int32](nodes)
    var dleaf = ctx.enqueue_create_buffer[DType.float32](nodes * outputs)
    ctx.enqueue_copy(dst_buf=doff, src_ptr=offsets.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dcol, src_ptr=columns.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dthr, src_ptr=thresholds.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dleft, src_ptr=left.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dleaf, src_ptr=leaves.unsafe_ptr())
    forest_validate_device(ctx, doff, dcol, dthr, dleft, dleaf, len(offsets) - 1, nodes, features, outputs)
    _ = len(offsets)
    _ = len(columns)
    _ = len(thresholds)
    _ = len(left)
    _ = len(leaves)
    _ = doff^
    _ = dcol^
    _ = dthr^
    _ = dleft^
    _ = dleaf^


def forest_leaf_flag_kernel(
    left: MutPointer[Int32, MutAnyOrigin], nodes_in: Int32, flag: MutPointer[Int32, MutAnyOrigin],
):
    """One thread a node: 1 for a leaf (`left == -1`), else 0."""
    var i = Int(block_idx.x) * FOREST_FINITE_TPB + Int(thread_idx.x)
    if i < Int(nodes_in):
        flag.unsafe_store(i, Int32(1) if left.unsafe_load(i) == Int32(-1) else Int32(0))


def forest_pack_nodes_kernel(
    thr: MutPointer[Float32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    col: MutPointer[Int32, MutAnyOrigin], leaves: MutPointer[Float32, MutAnyOrigin],
    rank: MutPointer[Int32, MutAnyOrigin], nodes_in: Int32, outputs_in: Int32,
    packed: MutPointer[Int32, MutAnyOrigin], compact: MutPointer[Float32, MutAnyOrigin],
):
    """One thread a node: the FOREST-PACKED-1 node word (payload, left,
    column, 0), where a leaf's payload is its rank among the leaves (the
    exclusive scan of the leaf flags, the host loop's running leaf count)
    and its leaf vector moves to that rank's slot. Integer moves only."""
    var i = Int(block_idx.x) * FOREST_FINITE_TPB + Int(thread_idx.x)
    if i >= Int(nodes_in):
        return
    var outputs = Int(outputs_in)
    var l = left.unsafe_load(i)
    var payload = bitcast[DType.int32](thr.unsafe_load(i))
    if l == Int32(-1):
        var r = Int(rank.unsafe_load(i))
        payload = Int32(r)
        for c in range(outputs):
            compact.unsafe_store(r * outputs + c, leaves.unsafe_load(i * outputs + c))
    packed.unsafe_store(4 * i, payload)
    packed.unsafe_store(4 * i + 1, l)
    packed.unsafe_store(4 * i + 2, col.unsafe_load(i))
    packed.unsafe_store(4 * i + 3, Int32(0))


def forest_pack_device(
    ctx: DeviceContext, offsets: List[Int32], columns: List[Int32],
    thresholds: List[Float32], left: List[Int32], leaves: List[Float32],
    features: Int, outputs: Int,
    mut packed_out: Optional[DeviceBuffer[DType.int32]],
    mut leaves_out: Optional[DeviceBuffer[DType.float32]],
) raises:
    """Upload the flat forest once, validate it on the device
    (`forest_validate_device`) and pack it there (FOREST-PACKED-1 layout):
    the host packing loop over every node is gone (lane cpu3-core). One
    word, the leaf count, comes back to size the compact leaves."""
    var nodes = len(columns)
    var trees = len(offsets) - 1
    var doff = ctx.enqueue_create_buffer[DType.int32](len(offsets))
    var dcol = ctx.enqueue_create_buffer[DType.int32](nodes)
    var dthr = ctx.enqueue_create_buffer[DType.float32](nodes)
    var dleft = ctx.enqueue_create_buffer[DType.int32](nodes)
    var dleaf = ctx.enqueue_create_buffer[DType.float32](nodes * outputs)
    ctx.enqueue_copy(dst_buf=doff, src_ptr=offsets.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dcol, src_ptr=columns.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dthr, src_ptr=thresholds.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dleft, src_ptr=left.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dleaf, src_ptr=leaves.unsafe_ptr())
    # validates and drains: the host lists are read by the time it returns
    forest_pack_resident(
        ctx, doff, dcol, dthr, dleft, dleaf, trees, nodes, features, outputs,
        packed_out, leaves_out,
    )
    _ = len(offsets)
    _ = len(columns)
    _ = len(thresholds)
    _ = len(left)
    _ = len(leaves)
    _ = doff^
    _ = dcol^
    _ = dthr^
    _ = dleft^
    _ = dleaf^


def forest_pack_resident(
    ctx: DeviceContext, mut doff: DeviceBuffer[DType.int32], mut dcol: DeviceBuffer[DType.int32],
    mut dthr: DeviceBuffer[DType.float32], mut dleft: DeviceBuffer[DType.int32],
    mut dleaf: DeviceBuffer[DType.float32], trees: Int, nodes: Int, features: Int, outputs: Int,
    mut packed_out: Optional[DeviceBuffer[DType.int32]],
    mut leaves_out: Optional[DeviceBuffer[DType.float32]],
) raises:
    """`forest_pack_device` on a flat forest already resident on `ctx`
    (lane cpu4-misc: the multi-GPU grove owners gather their groves on the
    device and pack them here). Validates, then packs; the input buffers
    stay the caller's. One word, the leaf count, comes back."""
    forest_validate_device(ctx, doff, dcol, dthr, dleft, dleaf, trees, nodes, features, outputs)
    var rank = ctx.enqueue_create_buffer[DType.int32](nodes + 1)
    var grid = (nodes + FOREST_FINITE_TPB - 1) // FOREST_FINITE_TPB
    ctx.enqueue_function[forest_leaf_flag_kernel](
        dleft.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(nodes),
        rank.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        grid_dim=grid, block_dim=FOREST_FINITE_TPB,
    )
    device_exclusive_scan_total(ctx, rank, nodes)
    var h = ctx.enqueue_create_host_buffer[DType.int32](1)
    var tail = rank.create_sub_buffer[DType.int32](nodes, 1)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=tail)
    ctx.synchronize()
    var n_leaf = Int(h.unsafe_ptr().unsafe_load(0))
    _ = tail^
    _ = h^
    var packed = ctx.enqueue_create_buffer[DType.int32](4 * nodes)
    var compact = ctx.enqueue_create_buffer[DType.float32](max(n_leaf * outputs, 1))
    ctx.enqueue_function[forest_pack_nodes_kernel](
        dthr.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        dleft.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        dcol.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        dleaf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        rank.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        Int32(nodes), Int32(outputs),
        packed.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        compact.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        grid_dim=grid, block_dim=FOREST_FINITE_TPB,
    )
    ctx.synchronize()
    _ = rank^
    packed_out = packed^
    leaves_out = compact^


def forest_predict_gpu[RF_INPUT: Bool, GROVE: Bool](
    ctx: DeviceContext, offsets: List[Int32], columns: List[Int32],
    thresholds: List[Float32], left: List[Int32], leaves: List[Float32],
    x: List[Float32], n_rows: Int, n_features: Int, n_outputs: Int,
) raises -> List[Float32]:
    """Validated synchronous prototype; upload and readback included by caller."""
    _forest_shape_checks(offsets,columns,thresholds,left,leaves,x,n_rows,n_features,n_outputs)
    var trees = len(offsets)-1
    var doff = ctx.enqueue_create_buffer[DType.int32](len(offsets))
    var dcol = ctx.enqueue_create_buffer[DType.int32](len(columns))
    var dthr = ctx.enqueue_create_buffer[DType.float32](len(thresholds))
    var dleft = ctx.enqueue_create_buffer[DType.int32](len(left))
    var dleaf = ctx.enqueue_create_buffer[DType.float32](len(leaves))
    ctx.enqueue_copy(dst_buf=doff,src_ptr=offsets.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dcol,src_ptr=columns.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dthr,src_ptr=thresholds.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dleft,src_ptr=left.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dleaf,src_ptr=leaves.unsafe_ptr())
    # the model's per-node checks run on the device where it landed
    forest_validate_device(
        ctx, doff, dcol, dthr, dleft, dleaf, trees, len(columns), n_features, n_outputs,
    )
    if n_rows == 0:
        _ = len(offsets)
        _ = len(columns)
        _ = len(thresholds)
        _ = len(left)
        _ = len(leaves)
        return List[Float32]()
    var dx = ctx.enqueue_create_buffer[DType.float32](len(x))
    var dout = ctx.enqueue_create_buffer[DType.float32](n_rows*n_outputs)
    ctx.enqueue_copy(dst_buf=dx,src_ptr=x.unsafe_ptr())
    if not device_all_finite(ctx,dx,len(x)):
        raise Error("forest inference prototype requires finite Float32 values")
    launch_forest_inference[RF_INPUT,GROVE](
        ctx,doff,dcol,dthr,dleft,dleaf,dx,dout,n_rows,n_features,n_outputs,trees,
    )
    var out_ok = device_all_finite(ctx,dout,n_rows*n_outputs)
    # the answer lands straight in the result list (no host copy loop)
    var result = List[Float32](length=n_rows*n_outputs, fill=Float32(0.0))
    ctx.enqueue_copy(dst_ptr=result.unsafe_ptr(),src_buf=dout)
    ctx.synchronize()
    # Keep borrowed host inputs and device operands live through the drain.
    _ = len(offsets)
    _ = len(columns)
    _ = len(thresholds)
    _ = len(left)
    _ = len(leaves)
    _ = len(x)
    _ = doff^
    _ = dcol^
    _ = dthr^
    _ = dleft^
    _ = dleaf^
    _ = dx^
    _ = dout^
    if not out_ok:
        raise Error("forest inference prototype requires finite Float32 values")
    return result^

# T34/T35/T38, C50. Default OFF through core.forest_experiments.
# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
@always_inline
def forest_prediction_value[RF_INPUT: Bool, GROVE: Bool, PACKED: Bool](
    offsets: MutPointer[Int32, MutAnyOrigin], columns: MutPointer[Int32, MutAnyOrigin],
    thresholds: MutPointer[Float32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    leaves: MutPointer[Float32, MutAnyOrigin], x: MutPointer[Float32, MutAnyOrigin],
    row: Int, features: Int, outputs: Int, trees: Int, channel: Int,
) -> Float32:
    var total = Float32(0)
    comptime if T34_CHUNK_FOLD:
        var chunk = InlineArray[Float32, FOREST_CHUNK](fill=Float32(0))
        var first = 0
        while first < trees:
            for i in range(FOREST_CHUNK):
                chunk[i] = 0
                if first + i < trees:
                    var node = reached_leaf[RF_INPUT, PACKED](offsets, columns, thresholds, left, x, first+i, row, features)
                    chunk[i] = leaves.unsafe_load(node*outputs+channel)
            total = forest_add(total, forest_chunk_sum(chunk))
            first += FOREST_CHUNK
    elif GROVE:
        var sums = InlineArray[Float32, 32](fill=Float32(0))
        for lane in range(32):
            var tree = lane
            while tree < trees:
                var node = reached_leaf[RF_INPUT, PACKED](offsets, columns, thresholds, left, x, tree, row, features)
                sums[lane] = forest_add(sums[lane], leaves.unsafe_load(node*outputs+channel))
                tree += 32
        var step = 16
        while step > 0:
            for i in range(step):
                sums[i] = forest_add(sums[i], sums[i+step])
            step //= 2
        total = sums[0]
    else:
        for tree in range(trees):
            var node = reached_leaf[RF_INPUT, PACKED](offsets, columns, thresholds, left, x, tree, row, features)
            total = forest_add(total, leaves.unsafe_load(node*outputs+channel))
    return forest_chunk_finish(total, trees)


def forest_chunk_kernel[RF_INPUT: Bool, PACKED: Bool](
    offsets: MutPointer[Int32, MutAnyOrigin], columns: MutPointer[Int32, MutAnyOrigin],
    thresholds: MutPointer[Float32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    leaves: MutPointer[Float32, MutAnyOrigin], x: MutPointer[Float32, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin], rows: Int32, features: Int32, outputs: Int32, trees: Int32,
):
    var u = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if u < Int(rows)*Int(outputs):
        output.unsafe_store(u, forest_prediction_value[RF_INPUT, False, PACKED](offsets, columns, thresholds, left, leaves, x,
            u//Int(outputs), Int(features), Int(outputs), Int(trees), u%Int(outputs)))


def forest_leaf_ids_kernel[RF_INPUT: Bool, PACKED: Bool](
    offsets: MutPointer[Int32, MutAnyOrigin], columns: MutPointer[Int32, MutAnyOrigin],
    thresholds: MutPointer[Float32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin], ids: MutPointer[Int32, MutAnyOrigin],
    row_start: Int32, rows: Int32, features: Int32, trees: Int32,
    bad: MutPointer[Int32, MutAnyOrigin],
):
    var u = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if u < Int(rows)*Int(trees):
        comptime if T36_FINITE_STAGE:
            # The mandatory leaf-ID preparation owns full rows, including every
            # unused feature. Prediction cannot return until this flag is read.
            if u%Int(trees) == 0:
                for c in range(Int(features)):
                    var value = x.unsafe_load((Int(row_start)+u//Int(trees))*Int(features)+c)
                    if (bitcast[DType.uint32](value)&UInt32(0x7f800000)) == UInt32(0x7f800000):
                        _ = Atomic.max(bad, Int32(1))
        ids.unsafe_store(u, Int32(reached_leaf[RF_INPUT, PACKED](offsets, columns, thresholds, left, x,
            u%Int(trees), Int(row_start)+u//Int(trees), Int(features))))


@always_inline
def _forest_leaf_value[GROVE: Bool](ids: MutPointer[Int32, MutAnyOrigin], leaves: MutPointer[Float32, MutAnyOrigin],
                                   row: Int, k: Int, nt: Int, channel: Int) -> Float32:
    var total = Float32(0)
    var sums = InlineArray[Float32, 32](fill=Float32(0))
    comptime if T34_CHUNK_FOLD:
        var first = 0
        while first < nt:
            for i in range(32):
                sums[i] = 0
                if first+i < nt:
                    sums[i] = leaves.unsafe_load(Int(ids.unsafe_load(row*nt+first+i))*k+channel)
            total = forest_add(total, forest_chunk_sum(sums))
            first += 32
    elif GROVE:
        for lane in range(32):
            var t = lane
            while t < nt:
                sums[lane] = forest_add(sums[lane], leaves.unsafe_load(Int(ids.unsafe_load(row*nt+t))*k+channel))
                t += 32
        var step = 16
        while step > 0:
            for i in range(step):
                sums[i] = forest_add(sums[i], sums[i+step])
            step //= 2
        total = sums[0]
    else:
        for t in range(nt):
            total = forest_add(total, leaves.unsafe_load(Int(ids.unsafe_load(row*nt+t))*k+channel))
    return forest_chunk_finish(total, nt)


def forest_leaf_reduce_kernel[GROVE: Bool](
    ids: MutPointer[Int32, MutAnyOrigin], leaves: MutPointer[Float32, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin], row_start: Int32, rows: Int32, outputs: Int32, trees: Int32,
):
    var u = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    var k = Int(outputs)
    var nt = Int(trees)
    if u >= Int(rows)*k:
        return
    var row = u//k
    var channel = u%k
    output.unsafe_store((Int(row_start)+row)*k+channel, _forest_leaf_value[GROVE](ids, leaves, row, k, nt, channel))


def _launch_forest_leaf_reuse[RF_INPUT: Bool, GROVE: Bool, PACKED: Bool](
    ctx: DeviceContext, mut offsets: DeviceBuffer[DType.int32], mut columns: DeviceBuffer[DType.int32],
    mut thresholds: DeviceBuffer[DType.float32], mut left: DeviceBuffer[DType.int32],
    mut leaves: DeviceBuffer[DType.float32], mut x: DeviceBuffer[DType.float32], mut output: DeviceBuffer[DType.float32],
    rows: Int, features: Int, outputs: Int, trees: Int,
) raises:
    # 8 MiB bounded leaf-ID scratch; one complete row is the minimum atomic
    # unit. No feature or dataset boundary. Every output shares the walk.
    var capacity = max(1, min(rows, (8*1024*1024)//max(4*trees, 1)))
    var ids = ctx.enqueue_create_buffer[DType.int32](capacity*trees)
    var bad = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(bad, Int32(0))
    var first = 0
    while first < rows:
        var count = min(capacity, rows-first)
        _launch_leaf_ids[RF_INPUT, PACKED](ctx, offsets, columns, thresholds, left, x, ids, bad, first, count, features, trees)
        ctx.enqueue_function[forest_leaf_reduce_kernel[GROVE]](
            ids.unsafe_ptr(), leaves.unsafe_ptr(), output.unsafe_ptr(), Int32(first), Int32(count), Int32(outputs), Int32(trees),
            grid_dim=(count*outputs+127)//128, block_dim=128)
        first += count
    # Scratch must remain alive through the last dependent fold.
    var host_bad = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=host_bad.unsafe_ptr(), src_buf=bad)
    ctx.synchronize()
    if host_bad.unsafe_ptr().unsafe_load(0) != 0:
        raise Error("resident forest requires finite Float32 values")
    _ = host_bad^
    _ = bad^
    _ = ids^


def forest_labels_kernel[RF_INPUT: Bool, GROVE: Bool, PACKED: Bool](
    offsets: MutPointer[Int32, MutAnyOrigin], columns: MutPointer[Int32, MutAnyOrigin],
    thresholds: MutPointer[Float32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    leaves: MutPointer[Float32, MutAnyOrigin], x: MutPointer[Float32, MutAnyOrigin], codes: MutPointer[Int32, MutAnyOrigin],
    bad: MutPointer[Int32, MutAnyOrigin], rows: Int32, features: Int32, outputs: Int32, trees: Int32,
):
    var row = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if row >= Int(rows):
        return
    var best = 0
    var best_value = Float32(0)
    for c in range(Int(outputs)):
        var value = forest_prediction_value[RF_INPUT, GROVE, PACKED](offsets, columns, thresholds, left, leaves, x,
            row, Int(features), Int(outputs), Int(trees), c)
        if (bitcast[DType.uint32](value)&UInt32(0x7f800000)) == UInt32(0x7f800000):
            _ = Atomic.max(bad, Int32(1))
            codes.unsafe_store(row, Int32(-1))
            return
        if c == 0 or value > best_value:
            best = c
            best_value = value
    codes.unsafe_store(row, Int32(best))

# T32×T33×T35×T36: physical schedule of the bounded leaf preparation.
# Arithmetic is solely in the following output fold. Shared tiles carry all
# feature bits and every required row is checked, including unused columns.
def forest_leaf_ids_grouped_kernel[RF_INPUT: Bool, PACKED: Bool](
    offsets: MutPointer[Int32, MutAnyOrigin], columns: MutPointer[Int32, MutAnyOrigin],
    thresholds: MutPointer[Float32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin], ids: MutPointer[Int32, MutAnyOrigin],
    row_start: Int32, rows: Int32, features: Int32, trees: Int32, bad: MutPointer[Int32, MutAnyOrigin],
):
    var tid = Int(thread_idx.x)
    var group = tid//32
    var lane = tid%32
    var first = Int(block_idx.x)*4
    var row = first+group
    var tiled = False
    var shared = stack_allocation[4*FOREST_SHARED_ROW_CAPACITY, Float32, address_space=AddressSpace.SHARED]()
    comptime if FOREST_SHARED_ROWS:
        if Int(features) <= FOREST_SHARED_ROW_CAPACITY:
            tiled = True
            var u = tid
            while u < 4*Int(features):
                var r = first+u//Int(features)
                if r < Int(rows):
                    shared.unsafe_store(u, x.unsafe_load((Int(row_start)+r)*Int(features)+u%Int(features)))
                u += 128
    barrier()
    if row < Int(rows):
        comptime if T36_FINITE_STAGE:
            var c = lane
            while c < Int(features):
                var value = shared.unsafe_load(group*Int(features)+c) if tiled else x.unsafe_load((Int(row_start)+row)*Int(features)+c)
                if (bitcast[DType.uint32](value)&UInt32(0x7f800000)) == UInt32(0x7f800000):
                    _ = Atomic.max(bad, Int32(1))
                c += 32
        var tree = lane
        while tree < Int(trees):
            var node = 0
            if tiled:
                node = reached_leaf[RF_INPUT, PACKED](offsets, columns, thresholds, left, shared, tree, group, Int(features))
            else:
                node = reached_leaf[RF_INPUT, PACKED](offsets, columns, thresholds, left, x, tree, Int(row_start)+row, Int(features))
            ids.unsafe_store(row*Int(trees)+tree, Int32(node))
            tree += 32


def forest_leaf_ids_row_kernel[RF_INPUT: Bool, PACKED: Bool](
    offsets: MutPointer[Int32, MutAnyOrigin], columns: MutPointer[Int32, MutAnyOrigin],
    thresholds: MutPointer[Float32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin], ids: MutPointer[Int32, MutAnyOrigin],
    row_start: Int32, rows: Int32, features: Int32, trees: Int32, bad: MutPointer[Int32, MutAnyOrigin],
):
    var row = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if row < Int(rows):
        comptime if T36_FINITE_STAGE:
            for c in range(Int(features)):
                var value = x.unsafe_load((Int(row_start)+row)*Int(features)+c)
                if (bitcast[DType.uint32](value)&UInt32(0x7f800000)) == UInt32(0x7f800000):
                    _ = Atomic.max(bad, Int32(1))
        for tree in range(Int(trees)):
            ids.unsafe_store(row*Int(trees)+tree, Int32(reached_leaf[RF_INPUT, PACKED](offsets, columns, thresholds, left,
                x, tree, Int(row_start)+row, Int(features))))


def _launch_leaf_ids[RF_INPUT: Bool, PACKED: Bool](ctx: DeviceContext,
    mut offsets: DeviceBuffer[DType.int32], mut columns: DeviceBuffer[DType.int32],
    mut thresholds: DeviceBuffer[DType.float32], mut left: DeviceBuffer[DType.int32],
    mut x: DeviceBuffer[DType.float32], mut ids: DeviceBuffer[DType.int32], mut bad: DeviceBuffer[DType.int32],
    first: Int, rows: Int, features: Int, trees: Int) raises:
    comptime if T33_COST_SCHEDULE:
        if (features*4+63)//64 <= max(1, min(trees, 32)//8):
            ctx.enqueue_function[forest_leaf_ids_row_kernel[RF_INPUT, PACKED]](
                offsets.unsafe_ptr(), columns.unsafe_ptr(), thresholds.unsafe_ptr(), left.unsafe_ptr(), x.unsafe_ptr(), ids.unsafe_ptr(),
                Int32(first), Int32(rows), Int32(features), Int32(trees), bad.unsafe_ptr(), grid_dim=(rows+127)//128, block_dim=128)
            return
    ctx.enqueue_function[forest_leaf_ids_grouped_kernel[RF_INPUT, PACKED]](
        offsets.unsafe_ptr(), columns.unsafe_ptr(), thresholds.unsafe_ptr(), left.unsafe_ptr(), x.unsafe_ptr(), ids.unsafe_ptr(),
        Int32(first), Int32(rows), Int32(features), Int32(trees), bad.unsafe_ptr(), grid_dim=(rows+3)//4, block_dim=128)


def forest_leaf_labels_kernel[GROVE: Bool](ids: MutPointer[Int32, MutAnyOrigin], leaves: MutPointer[Float32, MutAnyOrigin],
    codes: MutPointer[Int32, MutAnyOrigin], bad: MutPointer[Int32, MutAnyOrigin], first: Int32, rows: Int32, outputs: Int32, trees: Int32):
    var row = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if row < Int(rows):
        var best = 0
        var best_value = Float32(0)
        for c in range(Int(outputs)):
            var value = _forest_leaf_value[GROVE](ids, leaves, row, Int(outputs), Int(trees), c)
            if (bitcast[DType.uint32](value)&UInt32(0x7f800000)) == UInt32(0x7f800000):
                _ = Atomic.max(bad, Int32(1))
                return
            if c == 0 or value > best_value:
                best = c
                best_value = value
        codes.unsafe_store(Int(first)+row, Int32(best))


def launch_forest_leaf_labels[RF_INPUT: Bool, GROVE: Bool, PACKED: Bool](ctx: DeviceContext,
    mut offsets: DeviceBuffer[DType.int32], mut columns: DeviceBuffer[DType.int32], mut thresholds: DeviceBuffer[DType.float32],
    mut left: DeviceBuffer[DType.int32], mut leaves: DeviceBuffer[DType.float32], mut x: DeviceBuffer[DType.float32],
    mut codes: DeviceBuffer[DType.int32], rows: Int, features: Int, outputs: Int, trees: Int) raises:
    var capacity = max(1, min(rows, (8*1024*1024)//max(4*trees, 1)))
    var ids = ctx.enqueue_create_buffer[DType.int32](capacity*trees)
    var bad = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(bad, Int32(0))
    var first = 0
    while first < rows:
        var count = min(capacity, rows-first)
        _launch_leaf_ids[RF_INPUT, PACKED](ctx, offsets, columns, thresholds, left, x, ids, bad, first, count, features, trees)
        ctx.enqueue_function[forest_leaf_labels_kernel[GROVE]](ids.unsafe_ptr(), leaves.unsafe_ptr(), codes.unsafe_ptr(), bad.unsafe_ptr(),
            Int32(first), Int32(count), Int32(outputs), Int32(trees), grid_dim=(count+127)//128, block_dim=128)
        first += count
    var host_bad = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=host_bad.unsafe_ptr(), src_buf=bad)
    ctx.synchronize()
    if host_bad.unsafe_ptr().unsafe_load(0) != 0:
        raise Error("resident forest requires finite Float32 values")
    _ = host_bad^
    _ = bad^
    _ = ids^

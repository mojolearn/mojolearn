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
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import ftz, identical_div
from checks.kernel_matrix import TARGET_COLUMN, forest_row_threads_for


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
comptime FOREST_SHARED_ROWS = is_defined["MOJOLEARN_FOREST_SHARED_ROWS"]()
comptime FOREST_SHARED_ROW_CAPACITY = 256

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
comptime FOREST_PACKED_NODES = not is_defined["MOJOLEARN_FOREST_SEPARATE_NODES"]()


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
    var item = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if item < rows*outputs:
        var total = Float32(0)
        for tree in range(trees):
            var node = reached_leaf[RF_INPUT, PACKED](offsets,columns,thresholds,left,x,tree,item//outputs,features)
            total = forest_add(total,leaves.unsafe_load(node*outputs+item%outputs))
        output.unsafe_store(item,ftz(identical_div(ftz(total),Float32(trees))))


def forest_argmax_kernel(
    scores: MutPointer[Float32, MutAnyOrigin],
    codes: MutPointer[Int32, MutAnyOrigin], rows_in: Int32, outputs_in: Int32,
):
    """Row-wise first-max argmax; `-1` reports a non-finite score."""
    var rows = Int(rows_in)
    var outputs = Int(outputs_in)
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= rows:
        return
    var base = row * outputs
    var best = 0
    var best_value = scores.unsafe_load(base)
    if (bitcast[DType.uint32](best_value) & UInt32(0x7f800000)) == UInt32(0x7f800000):
        codes.unsafe_store(row, Int32(-1))
        return
    for c in range(1, outputs):
        var value = scores.unsafe_load(base + c)
        if (bitcast[DType.uint32](value) & UInt32(0x7f800000)) == UInt32(0x7f800000):
            codes.unsafe_store(row, Int32(-1))
            return
        if value > best_value:
            best = c
            best_value = value
    codes.unsafe_store(row, Int32(best))


def launch_forest_argmax(ctx: DeviceContext,
    mut scores: DeviceBuffer[DType.float32], mut codes: DeviceBuffer[DType.int32],
    rows: Int, outputs: Int) raises:
    if rows > 0:
        ctx.enqueue_function[forest_argmax_kernel](
            scores.unsafe_ptr(), codes.unsafe_ptr(), Int32(rows), Int32(outputs),
            grid_dim=(rows + 127) // 128, block_dim=128,
        )


def forest_grove32_kernel[RF_INPUT: Bool, PACKED: Bool = False](
    offsets: MutPointer[Int32, MutAnyOrigin], columns: MutPointer[Int32, MutAnyOrigin],
    thresholds: MutPointer[Float32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    leaves: MutPointer[Float32, MutAnyOrigin], x: MutPointer[Float32, MutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin], rows_in: Int32, features_in: Int32, outputs_in: Int32, trees_in: Int32,
):
    var rows = Int(rows_in)
    var features = Int(features_in)
    var outputs = Int(outputs_in)
    var trees = Int(trees_in)
    # Four scalar row/output tasks per128-thread block; all threads hit barriers.
    var tid = Int(thread_idx.x)
    var lane = tid%32
    var item = Int(block_idx.x)*4+tid//32
    var total = Float32(0)
    var tiled = False
    comptime if FOREST_SHARED_ROWS:
        # DEVIATION 2963: with one output an item is a row, so the block's
        # four rows tile into shared memory; every thread hits the barrier.
        var xs = stack_allocation[4*FOREST_SHARED_ROW_CAPACITY,Float32,address_space=AddressSpace.SHARED]()
        if outputs == 1 and features <= FOREST_SHARED_ROW_CAPACITY:
            tiled = True
            var row0 = Int(block_idx.x)*4
            var count = 4*features
            var i = tid
            while i < count:
                var r = row0+i//features
                xs[unsafe_offset=i] = x.unsafe_load(r*features+i%features) if r < rows else Float32(0)
                i += 128
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
    var sums = stack_allocation[128,Float32,address_space=AddressSpace.SHARED]()
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
    var rows = Int(rows_in)
    var features = Int(features_in)
    var outputs = Int(outputs_in)
    var trees = Int(trees_in)
    var tid = Int(thread_idx.x)
    var lane = tid%32
    var row = Int(block_idx.x)*4+tid//32
    var totals = stack_allocation[OUTPUT_CAPACITY,Float32]()
    @parameter
    for c in range(OUTPUT_CAPACITY):
        totals[unsafe_offset=c] = Float32(0)
    var tiled = False
    comptime if FOREST_SHARED_ROWS:
        # DEVIATION 2963: the block's four rows tiled into shared memory.
        var xs = stack_allocation[4*FOREST_SHARED_ROW_CAPACITY,Float32,address_space=AddressSpace.SHARED]()
        if features <= FOREST_SHARED_ROW_CAPACITY:
            tiled = True
            var row0 = Int(block_idx.x)*4
            var count = 4*features
            var i = tid
            while i < count:
                var r = row0+i//features
                xs[unsafe_offset=i] = x.unsafe_load(r*features+i%features) if r < rows else Float32(0)
                i += 128
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
    var sums = stack_allocation[128*OUTPUT_CAPACITY,Float32,address_space=AddressSpace.SHARED]()
    @parameter
    for c in range(OUTPUT_CAPACITY):
        sums[unsafe_offset=c*128+tid] = totals[unsafe_offset=c]
    barrier()
    var step = 16
    while step > 0:
        if lane < step:
            @parameter
            for c in range(OUTPUT_CAPACITY):
                sums[unsafe_offset=c*128+tid] = forest_add(sums[unsafe_offset=c*128+tid],sums[unsafe_offset=c*128+tid+step])
        barrier()
        step //= 2
    if lane == 0 and row < rows:
        @parameter
        for c in range(OUTPUT_CAPACITY):
            if c < outputs:
                output.unsafe_store(row*outputs+c,ftz(identical_div(ftz(sums[unsafe_offset=c*128+tid]),Float32(trees))))


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
    if n_rows == 0:
        return
    comptime if GROVE:
        comptime if FOREST_ROW_THREADS:
            if n_features <= FOREST_ROW_THREADS_MAX_FEATURES:
                _launch_grove_rows[RF_INPUT, PACKED](ctx, doff, dcol, dthr, dleft, dleaf, dx, dout, n_rows, n_features, n_outputs, trees)
                return
        _launch_grove_lanes[RF_INPUT, PACKED](ctx, doff, dcol, dthr, dleft, dleaf, dx, dout, n_rows, n_features, n_outputs, trees)
    else:
        ctx.enqueue_function[forest_ordered_kernel[RF_INPUT,PACKED]](
            doff.unsafe_ptr(),dcol.unsafe_ptr(),dthr.unsafe_ptr(),dleft.unsafe_ptr(),
            dleaf.unsafe_ptr(),dx.unsafe_ptr(),dout.unsafe_ptr(),Int32(n_rows),Int32(n_features),Int32(n_outputs),Int32(trees),
            grid_dim=(n_rows*n_outputs+127)//128,block_dim=128,
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
    if vector_groves_for(n_outputs):
        if n_outputs <= 2:
            ctx.enqueue_function[forest_vector_grove32_kernel[RF_INPUT,2,PACKED]](
                doff.unsafe_ptr(),dcol.unsafe_ptr(),dthr.unsafe_ptr(),dleft.unsafe_ptr(),
                dleaf.unsafe_ptr(),dx.unsafe_ptr(),dout.unsafe_ptr(),Int32(n_rows),Int32(n_features),Int32(n_outputs),Int32(trees),
                grid_dim=(n_rows+3)//4,block_dim=128,
            )
        elif n_outputs <= 4:
            ctx.enqueue_function[forest_vector_grove32_kernel[RF_INPUT,4,PACKED]](
                doff.unsafe_ptr(),dcol.unsafe_ptr(),dthr.unsafe_ptr(),dleft.unsafe_ptr(),
                dleaf.unsafe_ptr(),dx.unsafe_ptr(),dout.unsafe_ptr(),Int32(n_rows),Int32(n_features),Int32(n_outputs),Int32(trees),
                grid_dim=(n_rows+3)//4,block_dim=128,
            )
        else:
            ctx.enqueue_function[forest_vector_grove32_kernel[RF_INPUT,8,PACKED]](
                doff.unsafe_ptr(),dcol.unsafe_ptr(),dthr.unsafe_ptr(),dleft.unsafe_ptr(),
                dleaf.unsafe_ptr(),dx.unsafe_ptr(),dout.unsafe_ptr(),Int32(n_rows),Int32(n_features),Int32(n_outputs),Int32(trees),
                grid_dim=(n_rows+3)//4,block_dim=128,
            )
    else:
        ctx.enqueue_function[forest_grove32_kernel[RF_INPUT,PACKED]](
            doff.unsafe_ptr(),dcol.unsafe_ptr(),dthr.unsafe_ptr(),dleft.unsafe_ptr(),
            dleaf.unsafe_ptr(),dx.unsafe_ptr(),dout.unsafe_ptr(),Int32(n_rows),Int32(n_features),Int32(n_outputs),Int32(trees),
            grid_dim=(n_rows*n_outputs+3)//4,block_dim=128,
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
    if vector_groves_for(n_outputs):
        if n_outputs <= 2:
            ctx.enqueue_function[forest_vector_grove32_row_kernel[RF_INPUT,2,PACKED]](
                doff.unsafe_ptr(),dcol.unsafe_ptr(),dthr.unsafe_ptr(),dleft.unsafe_ptr(),
                dleaf.unsafe_ptr(),dx.unsafe_ptr(),dout.unsafe_ptr(),Int32(n_rows),Int32(n_features),Int32(n_outputs),Int32(trees),
                grid_dim=(n_rows+127)//128,block_dim=128,
            )
        elif n_outputs <= 4:
            ctx.enqueue_function[forest_vector_grove32_row_kernel[RF_INPUT,4,PACKED]](
                doff.unsafe_ptr(),dcol.unsafe_ptr(),dthr.unsafe_ptr(),dleft.unsafe_ptr(),
                dleaf.unsafe_ptr(),dx.unsafe_ptr(),dout.unsafe_ptr(),Int32(n_rows),Int32(n_features),Int32(n_outputs),Int32(trees),
                grid_dim=(n_rows+127)//128,block_dim=128,
            )
        else:
            ctx.enqueue_function[forest_vector_grove32_row_kernel[RF_INPUT,8,PACKED]](
                doff.unsafe_ptr(),dcol.unsafe_ptr(),dthr.unsafe_ptr(),dleft.unsafe_ptr(),
                dleaf.unsafe_ptr(),dx.unsafe_ptr(),dout.unsafe_ptr(),Int32(n_rows),Int32(n_features),Int32(n_outputs),Int32(trees),
                grid_dim=(n_rows+127)//128,block_dim=128,
            )
    else:
        ctx.enqueue_function[forest_grove32_row_kernel[RF_INPUT,PACKED]](
            doff.unsafe_ptr(),dcol.unsafe_ptr(),dthr.unsafe_ptr(),dleft.unsafe_ptr(),
            dleaf.unsafe_ptr(),dx.unsafe_ptr(),dout.unsafe_ptr(),Int32(n_rows),Int32(n_features),Int32(n_outputs),Int32(trees),
            grid_dim=(n_rows*n_outputs+127)//128,block_dim=128,
        )


def require_finite(values: List[Float32]) raises:
    for value in values:
        if (bitcast[DType.uint32](value) & UInt32(0x7f800000)) == UInt32(0x7f800000):
            raise Error("forest inference prototype requires finite Float32 values")


def validate_flat_forest(
    offsets: List[Int32], columns: List[Int32], thresholds: List[Float32],
    left: List[Int32], leaves: List[Float32], x: List[Float32],
    rows: Int, features: Int, outputs: Int,
) raises:
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
    require_finite(thresholds)
    require_finite(leaves)
    require_finite(x)
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


def forest_predict_gpu[RF_INPUT: Bool, GROVE: Bool](
    ctx: DeviceContext, offsets: List[Int32], columns: List[Int32],
    thresholds: List[Float32], left: List[Int32], leaves: List[Float32],
    x: List[Float32], n_rows: Int, n_features: Int, n_outputs: Int,
) raises -> List[Float32]:
    """Validated synchronous prototype; upload and readback included by caller."""
    validate_flat_forest(offsets,columns,thresholds,left,leaves,x,n_rows,n_features,n_outputs)
    if n_rows == 0:
        return List[Float32]()
    var trees = len(offsets)-1
    var doff = ctx.enqueue_create_buffer[DType.int32](len(offsets))
    var dcol = ctx.enqueue_create_buffer[DType.int32](len(columns))
    var dthr = ctx.enqueue_create_buffer[DType.float32](len(thresholds))
    var dleft = ctx.enqueue_create_buffer[DType.int32](len(left))
    var dleaf = ctx.enqueue_create_buffer[DType.float32](len(leaves))
    var dx = ctx.enqueue_create_buffer[DType.float32](len(x))
    var dout = ctx.enqueue_create_buffer[DType.float32](n_rows*n_outputs)
    var hout = ctx.enqueue_create_host_buffer[DType.float32](n_rows*n_outputs)
    ctx.enqueue_copy(dst_buf=doff,src_ptr=offsets.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dcol,src_ptr=columns.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dthr,src_ptr=thresholds.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dleft,src_ptr=left.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dleaf,src_ptr=leaves.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dx,src_ptr=x.unsafe_ptr())
    launch_forest_inference[RF_INPUT,GROVE](
        ctx,doff,dcol,dthr,dleft,dleaf,dx,dout,n_rows,n_features,n_outputs,trees,
    )
    ctx.enqueue_copy(dst_ptr=hout.unsafe_ptr(),src_buf=dout)
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
    var result = List[Float32]()
    for i in range(n_rows*n_outputs):
        result.append(hout[i])
    _ = hout^
    require_finite(result)
    return result^

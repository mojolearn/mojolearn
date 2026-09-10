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


@always_inline
def forest_add(a: Float32, b: Float32) -> Float32:
    return ftz(ftz(a) + ftz(b))


@always_inline
def finite_key(value: Float32) -> UInt32:
    var bits = bitcast[DType.uint32](value)
    if (bits & UInt32(0x7fffffff)) == 0:
        bits = 0
    return ~bits if (bits & UInt32(0x80000000)) != 0 else bits ^ UInt32(0x80000000)


@always_inline
def reached_leaf[RF_INPUT: Bool](
    offsets: MutPointer[Int32, MutAnyOrigin], columns: MutPointer[Int32, MutAnyOrigin],
    thresholds: MutPointer[Float32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin], tree: Int, row: Int, features: Int,
) -> Int:
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


def forest_ordered_kernel[RF_INPUT: Bool](
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
            var node = reached_leaf[RF_INPUT](offsets,columns,thresholds,left,x,tree,item//outputs,features)
            total = forest_add(total,leaves.unsafe_load(node*outputs+item%outputs))
        output.unsafe_store(item,ftz(identical_div(ftz(total),Float32(trees))))


def forest_grove32_kernel[RF_INPUT: Bool](
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
    if item < rows*outputs:
        var tree = lane
        while tree < trees:
            var node = reached_leaf[RF_INPUT](offsets,columns,thresholds,left,x,tree,item//outputs,features)
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
    """Compile-time experiment selector; force-off retains the scalar reference."""
    return (is_defined["MOJOLEARN_FOREST_VECTOR_GROVES"]()
            and not is_defined["MOJOLEARN_FOREST_SCALAR_GROVES"]()
            and outputs >= 2 and outputs <= 8)


def forest_vector_grove32_kernel[RF_INPUT: Bool, OUTPUT_CAPACITY: Int](
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
    if row < rows:
        var tree = lane
        while tree < trees:
            var node = reached_leaf[RF_INPUT](offsets,columns,thresholds,left,x,tree,row,features)
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


def launch_forest_inference[RF_INPUT: Bool, GROVE: Bool](
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
        if vector_groves_for(n_outputs):
            if n_outputs <= 2:
                ctx.enqueue_function[forest_vector_grove32_kernel[RF_INPUT,2]](
                    doff.unsafe_ptr(),dcol.unsafe_ptr(),dthr.unsafe_ptr(),dleft.unsafe_ptr(),
                    dleaf.unsafe_ptr(),dx.unsafe_ptr(),dout.unsafe_ptr(),Int32(n_rows),Int32(n_features),Int32(n_outputs),Int32(trees),
                    grid_dim=(n_rows+3)//4,block_dim=128,
                )
            elif n_outputs <= 4:
                ctx.enqueue_function[forest_vector_grove32_kernel[RF_INPUT,4]](
                    doff.unsafe_ptr(),dcol.unsafe_ptr(),dthr.unsafe_ptr(),dleft.unsafe_ptr(),
                    dleaf.unsafe_ptr(),dx.unsafe_ptr(),dout.unsafe_ptr(),Int32(n_rows),Int32(n_features),Int32(n_outputs),Int32(trees),
                    grid_dim=(n_rows+3)//4,block_dim=128,
                )
            else:
                ctx.enqueue_function[forest_vector_grove32_kernel[RF_INPUT,8]](
                    doff.unsafe_ptr(),dcol.unsafe_ptr(),dthr.unsafe_ptr(),dleft.unsafe_ptr(),
                    dleaf.unsafe_ptr(),dx.unsafe_ptr(),dout.unsafe_ptr(),Int32(n_rows),Int32(n_features),Int32(n_outputs),Int32(trees),
                    grid_dim=(n_rows+3)//4,block_dim=128,
                )
        else:
            ctx.enqueue_function[forest_grove32_kernel[RF_INPUT]](
                doff.unsafe_ptr(),dcol.unsafe_ptr(),dthr.unsafe_ptr(),dleft.unsafe_ptr(),
                dleaf.unsafe_ptr(),dx.unsafe_ptr(),dout.unsafe_ptr(),Int32(n_rows),Int32(n_features),Int32(n_outputs),Int32(trees),
                grid_dim=(n_rows*n_outputs+3)//4,block_dim=128,
            )
    else:
        ctx.enqueue_function[forest_ordered_kernel[RF_INPUT]](
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

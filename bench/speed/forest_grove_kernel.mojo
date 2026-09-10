# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Resident GPU scalar/vector-grove A/B with complete synthetic forests.

 tools/with_build_lock.sh pixi run mojo run -I . bench/speed/forest_grove_kernel.mojo
 Optional args: rows trees depth outputs repetitions (defaults1M,100,16,2,6).
 IDENTICAL adds -D MOJOLEARN_NUMERIC_IDENTICAL=1; no vector flag is required:
 this benchmark explicitly launches BOTH implementations in the same process.

Each tree owns distinct full binary nodes (100 depth16 trees =13,107,100
nodes), varied feature IDs/thresholds/vector leaves; X is varied1M×28. This
is a synthetic complete-tree stress case, not representative-fit evidence.
Context wall timings include enqueue+drain, exclude upload/validation/readback.
Every paired run compares ALL output bits and reports a checksum; alternating
pair order AB/BA yields ABBA. Two paired warmups are outside retained samples.
Host+device array-size estimate is capped at2GiB; no training or rentals.
"""
from std.memory import bitcast
from std.sys import argv
from std.time import perf_counter_ns
from std.testing import assert_equal
from max.gpu.host import DeviceContext, DeviceBuffer
from core.forest_inference import forest_grove32_kernel, forest_vector_grove32_kernel, validate_flat_forest
from checks.numerics import numeric_mode_name
from checks.vendor import COMPILED_VENDOR


def mix32(v: UInt32) -> UInt32:
    # Existing integer avalanche spelling used by synthetic fixture generators.
    var x = v
    x ^= x >> 16
    x *= UInt32(0x7feb352d)
    x ^= x >> 15
    x *= UInt32(0x846ca68b)
    return x ^ (x >> 16)


def enqueue[CAP: Int](
    ctx: DeviceContext, vector: Bool,
    mut offsets: DeviceBuffer[DType.int32], mut cols: DeviceBuffer[DType.int32],
    mut thresholds: DeviceBuffer[DType.float32], mut left: DeviceBuffer[DType.int32],
    mut leaves: DeviceBuffer[DType.float32], mut x: DeviceBuffer[DType.float32],
    mut output: DeviceBuffer[DType.float32], rows: Int, trees: Int, outputs: Int,
) raises:
    if vector:
        ctx.enqueue_function[forest_vector_grove32_kernel[False,CAP]](
            offsets.unsafe_ptr(),cols.unsafe_ptr(),thresholds.unsafe_ptr(),left.unsafe_ptr(),
            leaves.unsafe_ptr(),x.unsafe_ptr(),output.unsafe_ptr(),Int32(rows),Int32(28),Int32(outputs),Int32(trees),
            grid_dim=(rows+3)//4,block_dim=128,
        )
    else:
        ctx.enqueue_function[forest_grove32_kernel[False]](
            offsets.unsafe_ptr(),cols.unsafe_ptr(),thresholds.unsafe_ptr(),left.unsafe_ptr(),
            leaves.unsafe_ptr(),x.unsafe_ptr(),output.unsafe_ptr(),Int32(rows),Int32(28),Int32(outputs),Int32(trees),
            grid_dim=(rows*outputs+3)//4,block_dim=128,
        )


def run[CAP: Int](rows: Int, trees: Int, depth: Int, outputs: Int, repetitions: Int) raises:
    var nodes_per_tree = (1 << (depth+1))-1
    var first_leaf = (1 << depth)-1
    var nodes = trees*nodes_per_tree
    var estimated_bytes = 2*(nodes*(12+4*outputs)+4*(trees+1)+rows*28*4)+rows*outputs*4*4
    if estimated_bytes > 2*1024*1024*1024 or nodes*outputs > 2147483647 or rows*28 > 2147483647:
        raise Error("fixture exceeds2GiB host+device array estimate or Int32 element bounds")
    print("numeric_mode",numeric_mode_name())
    print("vendor",COMPILED_VENDOR)
    print("dimensions",rows,28,trees,depth,outputs,"nodes",nodes,"array_bytes_estimate",estimated_bytes)
    var setup_start = perf_counter_ns()
    var offsets = List[Int32](length=trees+1,fill=Int32(0))
    var cols = List[Int32](length=nodes,fill=Int32(0))
    var thresholds = List[Float32](length=nodes,fill=Float32(0))
    var left = List[Int32](length=nodes,fill=Int32(-1))
    var leaves = List[Float32](length=nodes*outputs,fill=Float32(0))
    var x = List[Float32](length=rows*28,fill=Float32(0))
    for t in range(trees):
        offsets[t] = Int32(t*nodes_per_tree)
        for node in range(nodes_per_tree):
            var global_node = t*nodes_per_tree+node
            var h = mix32(UInt32(global_node)+UInt32(0x18407))
            cols[global_node] = Int32(h % 28)
            thresholds[global_node] = Float32(Int((h >> 8)&1023)-512)/Float32(1024)
            if node < first_leaf:
                left[global_node] = Int32(2*node+1)
            else:
                for c in range(outputs):
                    var value = mix32(h+UInt32(c)*UInt32(97))
                    leaves[global_node*outputs+c] = Float32(Int(value&1023)-512)/Float32(1024)
    offsets[trees] = Int32(nodes)
    for i in range(rows*28):
        var h = mix32(UInt32(i)+UInt32(0x97531))
        x[i] = Float32(Int(h&65535)-32768)/Float32(32768)
    validate_flat_forest(offsets,cols,thresholds,left,leaves,x,rows,28,outputs)
    var ctx = DeviceContext()
    var doff = ctx.enqueue_create_buffer[DType.int32](trees+1)
    var dcol = ctx.enqueue_create_buffer[DType.int32](nodes)
    var dthr = ctx.enqueue_create_buffer[DType.float32](nodes)
    var dleft = ctx.enqueue_create_buffer[DType.int32](nodes)
    var dleaf = ctx.enqueue_create_buffer[DType.float32](nodes*outputs)
    var dx = ctx.enqueue_create_buffer[DType.float32](rows*28)
    var scalar = ctx.enqueue_create_buffer[DType.float32](rows*outputs)
    var vector = ctx.enqueue_create_buffer[DType.float32](rows*outputs)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](rows*outputs)
    var hv = ctx.enqueue_create_host_buffer[DType.float32](rows*outputs)
    ctx.enqueue_copy(dst_buf=doff,src_ptr=offsets.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dcol,src_ptr=cols.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dthr,src_ptr=thresholds.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dleft,src_ptr=left.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dleaf,src_ptr=leaves.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dx,src_ptr=x.unsafe_ptr())
    ctx.synchronize()
    print("fixture_validation_upload_ms",Float64(perf_counter_ns()-setup_start)/1e6)
    var reference_hash = UInt64(0)
    for rep in range(repetitions+2):
        for turn in range(2):
            var candidate = (turn == 1) if rep%2 == 0 else (turn == 0)
            var start = perf_counter_ns()
            if candidate:
                enqueue[CAP](ctx,True,doff,dcol,dthr,dleft,dleaf,dx,vector,rows,trees,outputs)
            else:
                enqueue[CAP](ctx,False,doff,dcol,dthr,dleft,dleaf,dx,scalar,rows,trees,outputs)
            ctx.synchronize()
            var elapsed = Float64(perf_counter_ns()-start)/1e6
            print("warmup_ms" if rep < 2 else "kernel_ms",rep, "vector" if candidate else "scalar",elapsed)
        ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(),src_buf=scalar)
        ctx.enqueue_copy(dst_ptr=hv.unsafe_ptr(),src_buf=vector)
        ctx.synchronize()
        var fingerprint = UInt64(1469598103934665603)
        for i in range(rows*outputs):
            var a = bitcast[DType.uint32](hs[i])
            var b = bitcast[DType.uint32](hv[i])
            assert_equal(a,b)
            fingerprint = (fingerprint ^ UInt64(a))*UInt64(1099511628211)
        if rep == 0:
            reference_hash = fingerprint
        else:
            assert_equal(fingerprint,reference_hash)
        print("fingerprint",rep,fingerprint)
    # Explicit operand/staging lifetime holds through final completion.
    _ = len(offsets)
    _ = len(cols)
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
    _ = scalar^
    _ = vector^
    _ = hs^
    _ = hv^
    print("PASS resident scalar/vector exact output bits")


def main() raises:
    var args = argv()
    var rows = 1000000
    var trees = 100
    var depth = 16
    var outputs = 2
    var repetitions = 6
    if len(args) != 1 and len(args) != 6:
        raise Error("arguments: rows trees depth outputs repetitions")
    if len(args) == 6:
        rows = Int(String(args[1]))
        trees = Int(String(args[2]))
        depth = Int(String(args[3]))
        outputs = Int(String(args[4]))
        repetitions = Int(String(args[5]))
    if rows < 1 or rows > 10000000 or trees < 1 or trees > 1000 or depth < 1 or depth > 16 or outputs < 2 or outputs > 8 or repetitions < 1:
        raise Error("bounds: rows1..10M trees1..1000 depth1..16 outputs2..8 repetitions>=1")
    if outputs <= 2:
        run[2](rows,trees,depth,outputs,repetitions)
    elif outputs <= 4:
        run[4](rows,trees,depth,outputs,repetitions)
    else:
        run[8](rows,trees,depth,outputs,repetitions)

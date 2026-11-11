# SPDX-License-Identifier: Apache-2.0
"""RunPod-only RF/ET fixed32 pool route, all-grove bits and failure gate."""
from std.os import getenv, setenv
from std.sys.compile import is_defined
from std.memory import bitcast
from max.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext
from checks.numerics import ftz
from core.forest_inference import forest_add, reached_leaf
from core.forest_inference_model import ResidentForest
from metrics.checks.device_io import upload_i32, upload_f32, download_f32


def original_totals[RF_INPUT: Bool](
    offsets: MutPointer[Int32, MutAnyOrigin], columns: MutPointer[Int32, MutAnyOrigin],
    thresholds: MutPointer[Float32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    leaves: MutPointer[Float32, MutAnyOrigin], x: MutPointer[Float32, MutAnyOrigin],
    destination: MutPointer[Float32, MutAnyOrigin], rows: Int32, outputs: Int32, trees: Int32,
):
    # Independent original global-index traversal; does not read the pool's
    # owner assignments, local offsets, per-grove starts or local counts.
    var index = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if index >= Int(rows) * Int(outputs) * 32:
        return
    var grove = index % 32
    var item = index // 32
    var total = Float32(0.0)
    for tree in range(grove, Int(trees), 32):
        var node = reached_leaf[RF_INPUT](offsets, columns, thresholds, left,
            x, tree, item // Int(outputs), 2)
        total = forest_add(total, leaves[node * Int(outputs) + item % Int(outputs)])
    destination[index] = total


def run_case[RF_INPUT: Bool](ctx: DeviceContext, trees: Int, outputs: Int,
    witness: Bool = False) raises:
    var offsets = List[Int32]()
    var columns = List[Int32]()
    var thresholds = List[Float32]()
    var left = List[Int32]()
    var leaves = List[Float32]()
    offsets.append(0)
    for tree in range(trees):
        columns.append(Int32(tree % 2))
        columns.append(0)
        columns.append(0)
        thresholds.append(bitcast[DType.float32](UInt32(1)))
        thresholds.append(0.0)
        thresholds.append(-0.0)
        left.append(1)
        left.append(-1)
        left.append(-1)
        offsets.append(Int32((tree + 1) * 3))
        for node in range(3):
            for c in range(outputs):
                var value = Float32((tree * 19 + c * 7 + node * 3) % 29 - 14) / Float32(8)
                if tree % 4 == 0:
                    value = Float32(67108864.0)
                elif tree % 4 == 2:
                    value = Float32(-67108864.0)
                elif c % 3 == 0:
                    value = bitcast[DType.float32](UInt32(1 if node == 1 else 0x80000001))
                if witness and tree % 32 == 0:
                    # All four visits belong to the SAME logical grove.
                    # Serial gives 1, balanced/reversed folding gives 0.
                    value = Float32(1.0)
                    if tree == 0:
                        value = Float32(67108864.0)
                    elif tree == 64:
                        value = Float32(-67108864.0)
                leaves.append(value)
    var x = List[Float32]()
    for row in range(129):
        x.append(bitcast[DType.float32](UInt32(row % 4)))
        x.append(Float32(row % 3 - 1))
    if not setenv("MOJOLEARN_FOREST_DEVICE_COUNT", "1", True):
        raise Error("setenv")
    var one = ResidentForest(offsets, columns, thresholds, left, leaves, 2, outputs)
    if not setenv("MOJOLEARN_FOREST_DEVICE_COUNT", "2", True):
        raise Error("setenv")
    var many = ResidentForest(offsets, columns, thresholds, left, leaves, 2, outputs)
    if not many.pool or one.pool or many.ctx or many.offsets or many.columns or many.leaves:
        raise Error("pool route retained full root model")
    if len(many.pool.value().owners) != min(2, trees):
        raise Error("owner count mismatch")
    var seen = List[Int](length=32, fill=0)
    var total_nodes = 0
    var total_trees = 0
    for rank in range(len(many.pool.value().owners)):
        ref owner = many.pool.value().owners[rank]
        total_nodes += owner.nodes
        total_trees += owner.trees
        for i in range(len(owner.groves)):
            seen[owner.groves[i]] += 1
    if total_nodes != 3 * trees or total_trees != trees:
        raise Error("resident tree coverage mismatch")
    for g in range(32):
        if seen[g] != (1 if g < min(32, trees) else 0):
            raise Error("duplicate or missing grove")
    var a = one.predict[RF_INPUT](x, 129, 2, outputs)
    var b = many.predict[RF_INPUT](x, 129, 2, outputs)
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            raise Error("prediction bits differ")
    var reused = List[Float32](length=129 * outputs, fill=Float32(-999))
    many.predict_into[RF_INPUT](x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        reused.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), 129, 2, outputs, True)
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](reused[i]):
            raise Error("predict_into bits differ")
    var doff = upload_i32(ctx, offsets)
    var dcol = upload_i32(ctx, columns)
    var dthr = upload_f32(ctx, thresholds)
    var dleft = upload_i32(ctx, left)
    var dleaf = upload_f32(ctx, leaves)
    var dx = upload_f32(ctx, x)
    var dt = ctx.enqueue_create_buffer[DType.float32](5 * outputs * 32)
    ctx.enqueue_function[original_totals[RF_INPUT]](doff.unsafe_ptr(),
        dcol.unsafe_ptr(), dthr.unsafe_ptr(), dleft.unsafe_ptr(), dleaf.unsafe_ptr(),
        dx.unsafe_ptr(), dt.unsafe_ptr(), Int32(5), Int32(outputs), Int32(trees),
        grid_dim=(5 * outputs * 32 + 127) // 128, block_dim=128)
    ctx.synchronize()
    var expected = download_f32(ctx, dt, 5 * outputs * 32)
    var actual = many.pool.value().collect[RF_INPUT](x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), 5)
    if witness:
        for item in range(5 * outputs):
            if bitcast[DType.uint32](actual[item * 32]) != bitcast[DType.uint32](Float32(1.0)):
                raise Error("grove zero did not preserve serial cancellation witness")
    for i in range(len(expected)):
        if bitcast[DType.uint32](expected[i]) != bitcast[DType.uint32](actual[i]):
            raise Error("individual grove bits differ " + String(i))
    _ = doff^
    _ = dcol^
    _ = dthr^
    _ = dleft^
    _ = dleaf^
    _ = dx^
    _ = dt^
    comptime if is_defined["MOJOLEARN_FOREST_POOL_FAULT"]():
        for rank in range(len(many.pool.value().owners)):
            if not setenv("MOJOLEARN_FOREST_FAIL_OWNER", String(rank), True):
                raise Error("setenv")
            var canary = List[Float32](length=129 * outputs, fill=Float32(-999))
            var refused = False
            try:
                many.predict_into[RF_INPUT](x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                    canary.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), 129, 2, outputs, True)
            except:
                refused = True
            if not refused:
                raise Error("injected owner failure accepted")
            for i in range(len(canary)):
                if canary[i] != Float32(-999):
                    raise Error("failed prediction published output")
        if not setenv("MOJOLEARN_FOREST_FAIL_OWNER", "-1", True):
            raise Error("setenv")
        var recovered = many.predict[RF_INPUT](x, 129, 2, outputs)
        for i in range(len(a)):
            if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](recovered[i]):
                raise Error("failed owner recovery differs")
    one.close()
    many.close()
    if witness:
        print("PASS forest pool serial-order witness", RF_INPUT, trees, outputs)
    else:
        print("PASS forest pool", RF_INPUT, trees, outputs)


def main() raises:
    if String(getenv("RUNPOD_POD_ID")) == "":
        raise Error("RunPod required; no local execution")
    var ctx = DeviceContext()
    if String(getenv("MOJOLEARN_FOREST_ORDER_ONLY", "0")) == "1":
        run_case[True](ctx, 97, 3, True)
        run_case[False](ctx, 97, 3, True)
        ctx.synchronize()
        return
    var counts: List[Int] = [1, 3, 31, 32, 33, 65, 97]
    var outputs: List[Int] = [1, 2, 3, 8, 9]
    for n in counts:
        for c in outputs:
            run_case[True](ctx, n, c)
            run_case[False](ctx, n, c)
    # Non-divisor cardinalities exercise item tiles starting within a row.
    run_case[True](ctx, 3, 129)
    run_case[False](ctx, 3, 129)
    run_case[True](ctx, 3, 4097)
    run_case[False](ctx, 3, 4097)
    run_case[True](ctx, 97, 3, True)
    run_case[False](ctx, 97, 3, True)
    ctx.synchronize()

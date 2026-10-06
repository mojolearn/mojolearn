# SPDX-License-Identifier: Apache-2.0
"""NN63 canonical neural leaf merge after supported shard transport.

This implements device reassembly/folding only, not a new transport backend.
Every shard must compute the same logical leaves as the single-device profile;
one permutation maps arriving leaf storage to its canonical global position.
The native owner must validate permutation completeness/uniqueness and admit
all shard completion before this component. No unordered floating collective.
Source-only, default OFF; public distributed training integration remains owed.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz

comptime NN63_CANONICAL_SHARD_MERGE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NN63_CANONICAL_SHARD_MERGE"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime _FP = MutPointer[Float32, MutAnyOrigin]
comptime _IP = MutPointer[Int32, MutAnyOrigin]


@always_inline
def nn_shard_pair(a: Float32, b: Float32) -> Float32:
    return ftz(ftz(a) + ftz(b))


def nn_shard_reassemble(dst: _FP, arrived: _FP, logical_leaf: _IP, leaves_in: Int32, width_in: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var width = Int(width_in)
    if i < Int(leaves_in) * width:
        var leaf = i // width
        dst[Int(logical_leaf[leaf]) * width + i - leaf * width] = arrived[i]


def nn_shard_fold_level(dst: _FP, src: _FP, leaves_in: Int32, width_in: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var width = Int(width_in)
    var leaves = Int(leaves_in)
    if i < ((leaves + 1) // 2) * width:
        var pair = i // width
        var feature = i - pair * width
        var value = src[(2 * pair) * width + feature]
        if 2 * pair + 1 < leaves:
            value = nn_shard_pair(value, src[(2 * pair + 1) * width + feature])
        dst[i] = value


def nn_shard_copy_result(dst: _FP, src: _FP, width_in: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(width_in):
        dst[i] = src[i]


def nn_shard_merge_into(ctx: DeviceContext, mut result: DeviceBuffer[DType.float32], mut arrived: DeviceBuffer[DType.float32], mut logical_leaf: DeviceBuffer[DType.int32], mut first: DeviceBuffer[DType.float32], mut second: DeviceBuffer[DType.float32], leaves: Int, width: Int) raises:
    comptime if not NN63_CANONICAL_SHARD_MERGE:
        raise Error("NN63 canonical shard merge is not enabled")
    if leaves < 1 or width < 1 or width > 2147483647 or leaves > 2147483647 // width:
        raise Error("NN63 shape exceeds native component bounds")
    # Each independent scalar/node is scheduled across 128 threads. The
    # arithmetic tree is leaves-in-global-order, never device/topology order.
    ctx.enqueue_function[nn_shard_reassemble](first.unsafe_ptr(), arrived.unsafe_ptr(), logical_leaf.unsafe_ptr(), Int32(leaves), Int32(width), grid_dim=((leaves * width + 127) // 128, 1, 1), block_dim=(128, 1, 1))
    var count = leaves
    var in_first = True
    while count > 1:
        var next_count = (count + 1) // 2
        if in_first:
            ctx.enqueue_function[nn_shard_fold_level](second.unsafe_ptr(), first.unsafe_ptr(), Int32(count), Int32(width), grid_dim=((next_count * width + 127) // 128, 1, 1), block_dim=(128, 1, 1))
        else:
            ctx.enqueue_function[nn_shard_fold_level](first.unsafe_ptr(), second.unsafe_ptr(), Int32(count), Int32(width), grid_dim=((next_count * width + 127) // 128, 1, 1), block_dim=(128, 1, 1))
        in_first = not in_first
        count = next_count
    if in_first:
        ctx.enqueue_function[nn_shard_copy_result](result.unsafe_ptr(), first.unsafe_ptr(), Int32(width), grid_dim=((width + 127) // 128, 1, 1), block_dim=(128, 1, 1))
    else:
        ctx.enqueue_function[nn_shard_copy_result](result.unsafe_ptr(), second.unsafe_ptr(), Int32(width), grid_dim=((width + 127) // 128, 1, 1), block_dim=(128, 1, 1))


def nn_shard_merge_host(arrived: _FP, logical_leaf: _IP, leaves: Int, width: Int) raises -> List[Float32]:
    """Host counterpart with the same per-cell pair function, not a test."""
    comptime if not NN63_CANONICAL_SHARD_MERGE:
        raise Error("NN63 canonical shard merge is not enabled")
    if leaves < 1 or width < 1 or width > 2147483647 or leaves > 2147483647 // width:
        raise Error("NN63 shape exceeds native component bounds")
    var values = List[Float32](length=leaves * width, fill=Float32(0.0))
    for leaf in range(leaves):
        for feature in range(width):
            values[Int(logical_leaf[leaf]) * width + feature] = arrived[leaf * width + feature]
    var count = leaves
    while count > 1:
        var next_count = (count + 1) // 2
        for pair in range(next_count):
            for feature in range(width):
                var value = values[(2 * pair) * width + feature]
                if 2 * pair + 1 < count:
                    value = nn_shard_pair(value, values[(2 * pair + 1) * width + feature])
                values[pair * width + feature] = value
        count = next_count
    var result = List[Float32]()
    for feature in range(width):
        result.append(values[feature])
    return result^

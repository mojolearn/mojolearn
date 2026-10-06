# SPDX-License-Identifier: Apache-2.0
"""NN54/NN57 source-only neural reduction profiles, default OFF.

Profile nn-reduce-v2(L,S): ascending leaves of L values starting at +0,
ftz operands then (optionally) an explicitly rounded square then an FTZ add;
adjacent leaf pairs combine with FTZ operands/add, odd leaves carry unchanged.
L and S are arithmetic parameters shared by host and every GPU. Physical block
size never enters the profile. Old-version equality is deliberately not a gate.

These are explicit component APIs. Public loss/clip wiring, all-vendor identity,
task quality and full-model timing remain pending. Nothing has been compiled or
verified. Caller owns both scratch planes and every input until its own drain.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul

comptime NN54_LOSS_PROFILE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NN54_LOSS_PROFILE"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime NN57_NORM_PROFILE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NN57_NORM_PROFILE"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime _FP = MutPointer[Float32, MutAnyOrigin]


@always_inline
def nn_reduce_leaf[LEAF: Int, SQUARE: Bool](x: _FP, n: Int, leaf: Int) -> Float32:
    comptime assert LEAF > 0
    var value = Float32(0.0)
    var first = leaf * LEAF
    for i in range(first, min(n, first + LEAF)):
        var term = ftz(x[i])
        comptime if SQUARE:
            term = ftz(identical_mul(term, term))
        value = ftz(ftz(value) + term)
    return value


@always_inline
def nn_reduce_pair(left: Float32, right: Float32) -> Float32:
    return ftz(ftz(left) + ftz(right))


def nn_reduce_leaves_kernel[LEAF: Int, SQUARE: Bool](dst: _FP, x: _FP, n_in: Int32):
    var leaf = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_in)
    if leaf < (n + LEAF - 1) // LEAF:
        dst[leaf] = nn_reduce_leaf[LEAF, SQUARE](x, n, leaf)


def nn_reduce_level_kernel(dst: _FP, src: _FP, width_in: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var width = Int(width_in)
    if 2 * i < width:
        var v = src[2 * i]
        if 2 * i + 1 < width:
            v = nn_reduce_pair(v, src[2 * i + 1])
        dst[i] = v


def nn_reduce_result_kernel(dst: _FP, src: _FP, empty: Int32):
    dst[0] = Float32(0.0) if empty != 0 else src[0]


def nn_reduce_scratch_floats[LEAF: Int](n: Int) -> Int:
    return max(1, (n + LEAF - 1) // LEAF)


def nn_reduce_into[LEAF: Int, SQUARE: Bool](
    ctx: DeviceContext, mut dst: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32], mut left: DeviceBuffer[DType.float32],
    mut right: DeviceBuffer[DType.float32], n: Int,
) raises:
    """Enqueue the profile sum; normalization/clip finishing is caller-owned.

    Inputs must already satisfy the caller's finite/shape/refusal policy; this
    is not a replacement for those public checks. Both scratch buffers have at
    least nn_reduce_scratch_floats(n) elements and may not alias input/output.
    """
    comptime if SQUARE:
        comptime if not NN57_NORM_PROFILE:
            raise Error("NN57 norm profile is not enabled")
    else:
        comptime if not NN54_LOSS_PROFILE:
            raise Error("NN54 loss profile is not enabled")
    if n < 0 or n > 2147483647:
        raise Error("neural reduction count exceeds native index range")
    var width = (n + LEAF - 1) // LEAF
    # 128 threads only schedules independent logical leaves/nodes, keeping
    # shared-memory demand zero and valid on the supported vendor block limits.
    if width > 0:
        ctx.enqueue_function[nn_reduce_leaves_kernel[LEAF, SQUARE]](
            left.unsafe_ptr(), x.unsafe_ptr(), Int32(n),
            grid_dim=((width + 127) // 128, 1, 1), block_dim=(128, 1, 1),
        )
    var in_left = True
    while width > 1:
        var count = (width + 1) // 2
        if in_left:
            ctx.enqueue_function[nn_reduce_level_kernel](right.unsafe_ptr(), left.unsafe_ptr(), Int32(width), grid_dim=((count + 127) // 128, 1, 1), block_dim=(128, 1, 1))
        else:
            ctx.enqueue_function[nn_reduce_level_kernel](left.unsafe_ptr(), right.unsafe_ptr(), Int32(width), grid_dim=((count + 127) // 128, 1, 1), block_dim=(128, 1, 1))
        in_left = not in_left
        width = count
    if in_left:
        ctx.enqueue_function[nn_reduce_result_kernel](dst.unsafe_ptr(), left.unsafe_ptr(), Int32(n == 0), grid_dim=(1, 1, 1), block_dim=(1, 1, 1))
    else:
        ctx.enqueue_function[nn_reduce_result_kernel](dst.unsafe_ptr(), right.unsafe_ptr(), Int32(n == 0), grid_dim=(1, 1, 1), block_dim=(1, 1, 1))


def nn_reduce_host[LEAF: Int, SQUARE: Bool](x: _FP, n: Int) raises -> Float32:
    """The same leaf/pair functions on the host; a version oracle, not a test."""
    comptime if SQUARE:
        comptime if not NN57_NORM_PROFILE:
            raise Error("NN57 norm profile is not enabled")
    else:
        comptime if not NN54_LOSS_PROFILE:
            raise Error("NN54 loss profile is not enabled")
    if n < 0 or n > 2147483647:
        raise Error("neural reduction count exceeds native index range")
    if n == 0:
        return Float32(0.0)
    var width = (n + LEAF - 1) // LEAF
    var values = List[Float32](length=width, fill=Float32(0.0))
    for i in range(width):
        values[i] = nn_reduce_leaf[LEAF, SQUARE](x, n, i)
    while width > 1:
        var count = (width + 1) // 2
        for i in range(count):
            var v = values[2 * i]
            if 2 * i + 1 < width:
                v = nn_reduce_pair(v, values[2 * i + 1])
            values[i] = v
        width = count
    return values[0]

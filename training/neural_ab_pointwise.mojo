# SPDX-License-Identifier: Apache-2.0
"""NN58/NN59 explicit neural pointwise experiments. Uncompiled and unverified.

No public caller is redirected. New entrypoints require individual opt-in
defines. Scalars use existing portable math and Philox mapping, on host/device.
The owner must preserve gradient contribution order and validate buffers before
calling. No floating atomic update appears here.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul
from core.philox_neural import neural_unit_at

comptime NN58_ACCUMULATE_STATUS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NN58_ACCUMULATE_STATUS"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime NN59_DROPOUT_RESIDUAL = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NN59_DROPOUT_RESIDUAL"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime _FP = MutPointer[Float32, MutAnyOrigin]
comptime _IP = MutPointer[Int32, MutAnyOrigin]


@always_inline
def nn_gradient_add(left: Float32, right: Float32) -> Float32:
    return ftz(ftz(left) + ftz(right))


def nn_accumulate_status_kernel(dst: _FP, left: _FP, right: _FP, status: _IP, n_in: Int32):
    """One declared tree edge, not a reassociation of a microbatch tree.

    One status cell per independent output; n means finite. Owner combines
    integer minima later and decides whether tentative state can be committed.
    This source experiment trades rereading floats for writing integer status;
    only full-step A/B can decide whether that trade wins.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var v = nn_gradient_add(left[i], right[i])
        dst[i] = v
        var invalid = (bitcast[DType.uint32](v) & UInt32(0x7F800000)) == UInt32(0x7F800000)
        status[i] = Int32(i) if invalid else n_in


def nn_accumulate_status_into(ctx: DeviceContext, mut dst: DeviceBuffer[DType.float32], mut left: DeviceBuffer[DType.float32], mut right: DeviceBuffer[DType.float32], mut status: DeviceBuffer[DType.int32], n: Int) raises:
    comptime if not NN58_ACCUMULATE_STATUS:
        raise Error("NN58 is not enabled")
    if n < 0 or n > 2147483647:
        raise Error("NN58 count exceeds native index range")
    if n == 0:
        return
    ctx.enqueue_function[nn_accumulate_status_kernel](dst.unsafe_ptr(), left.unsafe_ptr(), right.unsafe_ptr(), status.unsafe_ptr(), Int32(n), grid_dim=((n + 127) // 128, 1, 1), block_dim=(128, 1, 1))


@always_inline
def nn_dropout_cell(value: Float32, p: Float32, scale: Float32, seed_lo: UInt32, seed_hi: UInt32, stream: UInt32, index: Int) -> Float32:
    if neural_unit_at(seed_lo, seed_hi, stream, index) >= p:
        return ftz(identical_mul(ftz(value), ftz(scale)))
    return Float32(0.0)


@always_inline
def nn_dropout_residual_cell(value: Float32, residual: Float32, p: Float32, scale: Float32, seed_lo: UInt32, seed_hi: UInt32, stream: UInt32, index: Int) -> Float32:
    var dropped = nn_dropout_cell(value, p, scale, seed_lo, seed_hi, stream, index)
    # Two operations, including the exact dropout materialization seam.
    return ftz(ftz(residual) + ftz(dropped))


def nn_dropout_residual_kernel(dst: _FP, x: _FP, residual: _FP, n_in: Int32, offset: Int64, p: Float32, scale: Float32, seed_lo: UInt32, seed_hi: UInt32, stream: UInt32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        dst[i] = nn_dropout_residual_cell(x[i], residual[i], p, scale, seed_lo, seed_hi, stream, Int(offset) + i)


def nn_dropout_residual_backward_kernel(dx: _FP, dresidual: _FP, dy: _FP, n_in: Int32, offset: Int64, p: Float32, scale: Float32, seed_lo: UInt32, seed_hi: UInt32, stream: UInt32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var g = dy[i]
        dx[i] = nn_dropout_cell(g, p, scale, seed_lo, seed_hi, stream, Int(offset) + i)
        dresidual[i] = g


def nn_dropout_residual_into[BACKWARD: Bool](ctx: DeviceContext, mut first: DeviceBuffer[DType.float32], mut second: DeviceBuffer[DType.float32], mut x: DeviceBuffer[DType.float32], mut residual: DeviceBuffer[DType.float32], n: Int, offset: Int, p: Float32, scale: Float32, seed_lo: UInt32, seed_hi: UInt32, stream: UInt32) raises:
    """Forward writes first; backward writes first=dx, second=dResidual.

    Caller validates finite inputs/scalars and owns buffers until completion.
    Backward receives x=dOutput and ignores residual. Distinct backward outputs
    are required; forward may use same-cell in-place storage after ownership
    admission. Physical geometry is independent of the RNG coordinate.
    """
    comptime if not NN59_DROPOUT_RESIDUAL:
        raise Error("NN59 is not enabled")
    if n < 0 or n > 2147483647 or offset < 0 or offset > 9223372036854775807 - n:
        raise Error("NN59 index range exceeds native bounds")
    if not (p >= Float32(0.0) and p < Float32(1.0) and scale > Float32(0.0)):
        raise Error("NN59 invalid dropout configuration")
    if n == 0:
        return
    comptime if BACKWARD:
        ctx.enqueue_function[nn_dropout_residual_backward_kernel](first.unsafe_ptr(), second.unsafe_ptr(), x.unsafe_ptr(), Int32(n), Int64(offset), p, scale, seed_lo, seed_hi, stream, grid_dim=((n + 127) // 128, 1, 1), block_dim=(128, 1, 1))
    else:
        ctx.enqueue_function[nn_dropout_residual_kernel](first.unsafe_ptr(), x.unsafe_ptr(), residual.unsafe_ptr(), Int32(n), Int64(offset), p, scale, seed_lo, seed_hi, stream, grid_dim=((n + 127) // 128, 1, 1), block_dim=(128, 1, 1))

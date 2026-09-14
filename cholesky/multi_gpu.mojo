# SPDX-License-Identifier: Apache-2.0
"""Operation-level multi-GPU Cholesky: whole trailing-update rows and whole right-hand sides.

Factor. Each panel of the blocked right-looking factorization computes
`G = L21 . L21^T` with `identical_gemm_into(OP_NT)` and subtracts its lower
triangle from `A22`. A GEMM output cell `(i, j)` contracts row `i` of the left
operand with row `j` of the right operand over the panel width, in the gemm
fp32.v1 order, which depends on `k` (the panel width) and not on how many
output rows share a launch. So contiguous output-row ranges of `G` run on
owners that hold their rows of `L21` and all of `L21`, and the rows are copied
back as bytes. The panel factorization, its `info` read-back, the panel solve,
the subtraction and the pivot decision stay on the root unchanged: the order
in which panels are applied (the outer running subtraction of DEVIATION 1630)
is sequential by construction and is not partitioned.

Solve. `trsm_lower_kernel` and `trsm_upper_kernel` run one thread per
right-hand-side column, and a column reads only the factor and values it wrote
itself. A single column's substitution is sequential in its rows and is not
partitioned; whole columns are (see `cholesky/checks/trsm.mojo::cho_solve`).

Enabled by MOJOLEARN_CHOLESKY_DEVICE_COUNT > 1 in IDENTICAL mode. No float from
one owner is ever combined with a float from another.
"""
from std.os import getenv
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.multi_gpu import peer_clone, copy_columns_kernel
from core.step_phase import STEP_PHASE_TIMERS
from gemm.checks.gemm_identical import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.checks.gemm_oracle import OP_NT


def chol_device_count() raises -> Int:
    var value = String(getenv("MOJOLEARN_CHOLESKY_DEVICE_COUNT"))
    if value == "":
        return 1
    var count = Int(value)
    if count < 1 or count > 64:
        raise Error("Cholesky device count must be in [1, 64]")
    if count > 1:
        if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
            raise Error("multi-GPU Cholesky requires IDENTICAL numeric mode")
        comptime if STEP_PHASE_TIMERS:
            raise Error("multi-GPU Cholesky cannot use process-global GEMM phase counters")
    return count


@fieldwise_init
struct CholTrailingShard(Movable):
    var ctx: DeviceContext
    var left: DeviceBuffer[DType.float32]
    var right: DeviceBuffer[DType.float32]
    var output: DeviceBuffer[DType.float32]
    var workspace: DeviceBuffer[DType.float32]
    var first: Int
    var width: Int

    def __deinit__(deinit self):
        _ = self.workspace^
        _ = self.output^
        _ = self.right^
        _ = self.left^
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^


def chol_trailing_rows(
    ctx: DeviceContext,
    mut g: DeviceBuffer[DType.float32],
    mut packed: DeviceBuffer[DType.float32],
    n_trail: Int,
    w: Int,
    count: Int,
) raises:
    """`g[n_trail x n_trail] = packed . packed^T` by whole output rows across owners."""
    if n_trail > 2147483647 // n_trail:
        raise Error("multi-GPU Cholesky trailing update exceeds signed 32-bit indexing")
    var active = min(count, n_trail)
    # The packed operand must be complete before any owner reads its copy.
    ctx.synchronize()
    var shards = List[CholTrailingShard]()
    for rank in range(active):
        var first = n_trail * rank // active
        var width = n_trail * (rank + 1) // active - first
        var source = first
        comptime if is_defined["MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE"]():
            # Check-only arm: later owners read their left rows one row early.
            if rank > 0:
                source = first - 1
        var device = DeviceContext(device_id=rank)
        var left = device.enqueue_create_buffer[DType.float32](width * w)
        var output = device.enqueue_create_buffer[DType.float32](width * n_trail)
        var need = identical_gemm_workspace_max_floats(width, n_trail, w)
        var workspace = device.enqueue_create_buffer[DType.float32](need if need > 0 else 1)
        device.synchronize()
        var left_view = packed.create_sub_buffer[DType.float32](source * w, width * w)
        left_view.enqueue_copy_to(left)
        ctx.synchronize()
        _ = left_view^
        var right = peer_clone(ctx, device, packed)
        device.synchronize()
        ctx.synchronize()
        shards.append(CholTrailingShard(device^, left^, right^, output^, workspace^, first, width))
    for rank in range(active):
        ref s = shards[rank]
        identical_gemm_into(s.ctx, s.output, s.left, s.right, s.workspace,
            s.width, n_trail, w, OP_NT)
    for rank in range(active):
        ref s = shards[rank]
        s.ctx.synchronize()
        var target = g.create_sub_buffer[DType.float32](s.first * n_trail, s.width * n_trail)
        s.output.enqueue_copy_to(target)
        s.ctx.synchronize()
        ctx.synchronize()
        _ = target^
    _ = shards^
    ctx.synchronize()


@fieldwise_init
struct CholSolveShard(Movable):
    var ctx: DeviceContext
    var l: DeviceBuffer[DType.float32]
    var b: DeviceBuffer[DType.float32]
    var first: Int
    var width: Int

    def __deinit__(deinit self):
        _ = self.b^
        _ = self.l^
        try:
            self.ctx.synchronize()
        except:
            pass
        _ = self.ctx^


def chol_gather_columns(
    ctx: DeviceContext,
    mut full: DeviceBuffer[DType.float32],
    rows: Int,
    columns: Int,
    first: Int,
    width: Int,
) raises -> DeviceBuffer[DType.float32]:
    """Columns [first, first+width) of a rows x columns row-major matrix, as bytes."""
    var packed = ctx.enqueue_create_buffer[DType.float32](rows * width)
    var cells = rows * width
    ctx.enqueue_function[copy_columns_kernel[False]](
        full.unsafe_ptr(), packed.unsafe_ptr(), Int32(columns), Int32(first),
        Int32(width), Int32(cells), grid_dim=((cells + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    ctx.synchronize()
    return packed^


def chol_scatter_columns(
    ctx: DeviceContext,
    mut full: DeviceBuffer[DType.float32],
    mut packed: DeviceBuffer[DType.float32],
    rows: Int,
    columns: Int,
    first: Int,
    width: Int,
) raises:
    var cells = rows * width
    ctx.enqueue_function[copy_columns_kernel[True]](
        full.unsafe_ptr(), packed.unsafe_ptr(), Int32(columns), Int32(first),
        Int32(width), Int32(cells), grid_dim=((cells + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    ctx.synchronize()

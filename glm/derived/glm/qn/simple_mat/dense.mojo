# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`SimpleVec` / `SimpleDenseMat`: the vector operations the QN solver is written in.

PORT OF `cuml/cpp/src/glm/qn/simple_mat/dense.hpp` at cuML `00094f7`:
`ax`, `axpy`, `dot`, `squaredNorm`, `nrmMax`, `nrm2`, `copy_async`, `fill`.
Partial (`assign_gemm` is `core/gemm.mojo`'s business and is called from
`glm_base.mojo` directly; the sparse twin `sparse.hpp` is not ported). Do
not improve.

THEIR REDUCTIONS ARE FLOAT ATOMICS, AND THAT IS THE WHOLE IDENTITY STORY
------------------------------------------------------------------------
`dot` is `raft::linalg::mapThenSumReduce` (`dense.hpp:288-296`), whose
kernel folds each 256-thread block with `cub::BlockReduce` and then
**`raft::myAtomicAdd`s the block partials into `out`**
(`raft/linalg/detail/map_then_reduce.cuh:33-38`). The order in which
blocks arrive at that atomic is the scheduler's, so `dot(grad, drt)` -- the
number the line search compares against zero and against the Armijo bound
-- is a different float run to run ON ONE GPU, and every branch in
`qn_linesearch.cuh` and `qn_util.cuh` downstream of it is a data-dependent
branch on a non-reproducible bit. cuML accepts that (one backend, no
identity claim). We do not: IDENTITY_PATHS' rule has three moves and a
float atomic is the textbook case of REPLACE.

DEVIATION 547: every reduction here is ONE BLOCK of `STATS_TPB` threads
striding the vector, each partial through `identical_mul_add`, folded by
`core/pinned_reduce.pinned_block_sum` -- `block.sum` under FAST, a
lane-width-independent halving tree under IDENTICAL -- and written once by
thread 0. No atomic, no second block, so the sum is a pure function of the
vector's bits and of `STATS_TPB`, which `lib_block_bounds_a_float_fold`
pins to one value on every column. The vectors these fold are
`n_param = D + fit_intercept` long, so one block is also the right size.
`nrmMax` is a selection and needs no fold pin (`pinned_block_max`, row 30's
reasoning); it is here for the same reason as the others, to be one block
and to read back through one path.

`axpy`'s `a * x + y` is row 9's contraction exactly (`dense.hpp:179`, a
device lambda nvcc contracts by default): `identical_mul_add`. `ax`'s
`a * x` is one rounding. The stores are seams the next kernel reads: `ftz`.

IN-PLACE VARIANTS EXIST BECAUSE MOJO REFUSES ALIASED LAUNCH ARGUMENTS.
`drt.ax(ys / yy, drt)` and `drt.axpy(-alpha, yj, drt)` pass one buffer as
both operand and result; `enqueue_function` rejects the same origin twice,
so `ax_inplace_kernel` / `axpy_inplace_kernel` are the same arithmetic with
one pointer. Same for `squaredNorm = dot(u, u)`.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

from core.column_stats import STATS_TPB
from core.pinned_reduce import pinned_block_max, pinned_block_sum
from original.numerics import ftz, identical_mul_add


#: Elementwise launch width for `ax`/`axpy`. SCHEDULING: one thread, one cell.
comptime VEC_ELEM_TPB = 256


def _grid(n: Int) -> Int:
    return (n + VEC_ELEM_TPB - 1) // VEC_ELEM_TPB


def ax_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    a: Float32,
):
    """`this = a * x` (`dense.hpp:162-168`)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        out_v.unsafe_store(i, ftz(a * x.unsafe_load(i)))


def ax_inplace_kernel(
    x: MutPointer[Float32, MutAnyOrigin], n_in: Int32, a: Float32
):
    """`x = a * x`."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        x.unsafe_store(i, ftz(a * x.unsafe_load(i)))


def axpy_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    a: Float32,
):
    """`this = a * x + y` (`dense.hpp:171-181`), one rounding under IDENTICAL."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        out_v.unsafe_store(
            i, ftz(identical_mul_add(a, x.unsafe_load(i), y.unsafe_load(i)))
        )


def axpy_inplace_kernel(
    y: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    a: Float32,
):
    """`y = a * x + y`."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        y.unsafe_store(
            i, ftz(identical_mul_add(a, x.unsafe_load(i), y.unsafe_load(i)))
        )


def dot_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    u: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`dot(u, v)`: ONE block, strided partials, pinned fold. See the module
    docstring for what this replaces. Launch with `grid = 1, block =
    STATS_TPB` and nothing else: the fold's contract is that every thread
    of the block arrives."""
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var acc = Float32(0.0)
    var i = tid
    while i < n:
        acc = identical_mul_add(u.unsafe_load(i), v.unsafe_load(i), acc)
        i += STATS_TPB
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        out_v.unsafe_store(0, s0)


def dot_self_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    u: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`squaredNorm(u) = dot(u, u)`, character for character the kernel
    above with one pointer."""
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var acc = Float32(0.0)
    var i = tid
    while i < n:
        var x = u.unsafe_load(i)
        acc = identical_mul_add(x, x, acc)
        i += STATS_TPB
    var s0 = ftz(pinned_block_sum[STATS_TPB](acc))
    if tid == 0:
        out_v.unsafe_store(0, s0)


def nrm_max_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    u: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`nrmMax(u)`: `max(|u_i|)` seeded at 0 (`dense.hpp:306-316`)."""
    var n = Int(n_in)
    var tid = Int(thread_idx.x)
    var acc = Float32(0.0)
    var i = tid
    while i < n:
        var x = abs(u.unsafe_load(i))
        if x > acc:
            acc = x
        i += STATS_TPB
    var m = pinned_block_max[STATS_TPB](acc)
    if tid == 0:
        out_v.unsafe_store(0, m)


# ---------------------------------------------------------------------------
# host wrappers: launch, read the one scalar back, return it
# ---------------------------------------------------------------------------


def _read_scalar(
    ctx: DeviceContext, mut scalar: DeviceBuffer[DType.float32]
) raises -> Float32:
    var h = ctx.enqueue_create_host_buffer[DType.float32](1)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=scalar)
    ctx.synchronize()
    var v = h.unsafe_ptr().unsafe_load(0)
    _ = h^
    return v


def dot(
    ctx: DeviceContext,
    mut u: DeviceBuffer[DType.float32],
    mut v: DeviceBuffer[DType.float32],
    n: Int,
    mut scalar: DeviceBuffer[DType.float32],
) raises -> Float32:
    """`dot(u, v, tmp_dev, stream)`, `dense.hpp:288`."""
    ctx.enqueue_function[dot_kernel](
        scalar.unsafe_ptr(), u.unsafe_ptr(), v.unsafe_ptr(), Int32(n),
        grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
    )
    return _read_scalar(ctx, scalar)


def squared_norm(
    ctx: DeviceContext,
    mut u: DeviceBuffer[DType.float32],
    n: Int,
    mut scalar: DeviceBuffer[DType.float32],
) raises -> Float32:
    """`squaredNorm(u) = dot(u, u)`, `dense.hpp:300`."""
    ctx.enqueue_function[dot_self_kernel](
        scalar.unsafe_ptr(), u.unsafe_ptr(), Int32(n),
        grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
    )
    return _read_scalar(ctx, scalar)


def nrm_max(
    ctx: DeviceContext,
    mut u: DeviceBuffer[DType.float32],
    n: Int,
    mut scalar: DeviceBuffer[DType.float32],
) raises -> Float32:
    """`nrmMax(u)`, `dense.hpp:306`."""
    ctx.enqueue_function[nrm_max_kernel](
        scalar.unsafe_ptr(), u.unsafe_ptr(), Int32(n),
        grid_dim=(1, 1, 1), block_dim=(STATS_TPB, 1, 1),
    )
    return _read_scalar(ctx, scalar)


def nrm2(
    ctx: DeviceContext,
    mut u: DeviceBuffer[DType.float32],
    n: Int,
    mut scalar: DeviceBuffer[DType.float32],
) raises -> Float32:
    """`nrm2(u) = raft::mySqrt(squaredNorm(u))`, `dense.hpp:318`. The sqrt
    is on the HOST in theirs and here: a Float32 IEEE sqrt, correctly
    rounded on every host (`sqrtss` / `fsqrt`), no libm."""
    from std.math import sqrt

    return sqrt(squared_norm(ctx, u, n, scalar))


def ax(
    ctx: DeviceContext,
    mut out_v: DeviceBuffer[DType.float32],
    a: Float32,
    mut x: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    ctx.enqueue_function[ax_kernel](
        out_v.unsafe_ptr(), x.unsafe_ptr(), Int32(n), a,
        grid_dim=(_grid(n), 1, 1), block_dim=(VEC_ELEM_TPB, 1, 1),
    )


def ax_inplace(
    ctx: DeviceContext, mut x: DeviceBuffer[DType.float32], a: Float32, n: Int
) raises:
    ctx.enqueue_function[ax_inplace_kernel](
        x.unsafe_ptr(), Int32(n), a,
        grid_dim=(_grid(n), 1, 1), block_dim=(VEC_ELEM_TPB, 1, 1),
    )


def axpy(
    ctx: DeviceContext,
    mut out_v: DeviceBuffer[DType.float32],
    a: Float32,
    mut x: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """`out = a * x + y`."""
    ctx.enqueue_function[axpy_kernel](
        out_v.unsafe_ptr(), x.unsafe_ptr(), y.unsafe_ptr(), Int32(n), a,
        grid_dim=(_grid(n), 1, 1), block_dim=(VEC_ELEM_TPB, 1, 1),
    )


def axpy_inplace(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.float32],
    a: Float32,
    mut x: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """`y = a * x + y`."""
    ctx.enqueue_function[axpy_inplace_kernel](
        y.unsafe_ptr(), x.unsafe_ptr(), Int32(n), a,
        grid_dim=(_grid(n), 1, 1), block_dim=(VEC_ELEM_TPB, 1, 1),
    )


def copy_vec(
    ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.float32],
    mut src: DeviceBuffer[DType.float32],
) raises:
    """`copy_async`. Buffer to buffer, whole length."""
    ctx.enqueue_copy(dst_buf=dst, src_buf=src)

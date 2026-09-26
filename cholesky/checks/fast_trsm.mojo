"""FAST-only blocked triangular solves for `cho_solve` (Apple).

The pinned `trsm_lower_kernel` / `trsm_upper_kernel` give each right-hand
side ONE thread that walks all n rows, so a KernelRidge solve at n = 10,000
is ~10^8 dependent loads on a single GPU thread (11 of 13 s of the fit).
Here the rows go in blocks of FTS_BLOCK:

  * `fts_lower_diag_kernel` / `fts_upper_diag_kernel`: one threadgroup per
    right-hand side solves the block's diagonal triangle, the block's
    unknowns in threadgroup memory, each row's dot product split across the
    threads and folded with warp sums;
  * `fts_lower_update_kernel` / `fts_upper_update_kernel`: every row still
    to be solved subtracts the block's contribution, one thread per
    (row, right-hand side).

Kernel boundaries carry every device-memory hand-off, so nothing relies on
in-kernel ordering of device writes. Same `L X = B` / `L^T X = B` in place
over `B` (`n x nrhs` row-major), FAST arithmetic.
"""

from std.gpu import WARP_SIZE, block_dim, block_idx, thread_idx
from std.gpu.primitives.warp import shuffle_idx, sum as warp_sum
from std.gpu import lane_id
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from std.utils.index import IndexList

comptime FTS_BLOCK = 512
comptime FTS_TPB = 256
comptime FTS_WARPS = FTS_TPB // 32
comptime FTS_UPD_TPB = 256


@always_inline
def _block_sum(
    v: Float32,
    red: MutPointer[Float32, MutAnyOrigin, address_space = AddressSpace.SHARED],
) -> Float32:
    """Sum over the threadgroup, the answer on thread 0 only; two barriers
    so `red` can be reused by the next call."""
    var t = Int(thread_idx.x)
    var s = warp_sum(v)
    if t % 32 == 0:
        red[t // 32] = s
    barrier()
    var tot = Float32(0)
    if t == 0:
        for w in range(FTS_WARPS):
            tot += red[w]
    barrier()
    return tot


def fts_lower_diag_kernel(
    l: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    nrhs_in: Int32,
    r0_in: Int32,
    bs_in: Int32,
):
    var n = Int(n_in)
    var nrhs = Int(nrhs_in)
    var r0 = Int(r0_in)
    var bs = Int(bs_in)
    var j = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var xs = stack_allocation[
        FTS_BLOCK, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var red = stack_allocation[
        FTS_WARPS, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var cur = stack_allocation[
        1, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    for i in range(bs):
        var row = r0 + i
        var p = Float32(0)
        var k = t
        while k < i:
            p += l[row * n + r0 + k] * xs[k]
            k += FTS_TPB
        var s = _block_sum(
            p, red.unsafe_origin_cast[MutAnyOrigin]()
        )
        if t == 0:
            cur[0] = (b[row * nrhs + j] - s) / l[row * n + row]
            xs[i] = cur[0]
        barrier()
    var e = t
    while e < bs:
        b[(r0 + e) * nrhs + j] = xs[e]
        e += FTS_TPB


def fts_lower_update_kernel(
    l: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    nrhs_in: Int32,
    r0_in: Int32,
    bs_in: Int32,
):
    """Rows below the block: b[row] -= L[row, r0:r0+bs] . x[r0:r0+bs]."""
    var n = Int(n_in)
    var nrhs = Int(nrhs_in)
    var r0 = Int(r0_in)
    var bs = Int(bs_in)
    var j = Int(block_idx.y)
    var row = r0 + bs + Int(block_idx.x) * FTS_UPD_TPB + Int(thread_idx.x)
    if row >= n:
        return
    var acc = Float32(0)
    for k in range(bs):
        acc += l[row * n + r0 + k] * b[(r0 + k) * nrhs + j]
    b[row * nrhs + j] = b[row * nrhs + j] - acc


def fts_upper_diag_kernel(
    l: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    nrhs_in: Int32,
    r0_in: Int32,
    bs_in: Int32,
):
    """`L^T x = b` on rows [r0, r0 + bs), descending; L^T[i, k] = L[k, i]."""
    var n = Int(n_in)
    var nrhs = Int(nrhs_in)
    var r0 = Int(r0_in)
    var bs = Int(bs_in)
    var j = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var xs = stack_allocation[
        FTS_BLOCK, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var red = stack_allocation[
        FTS_WARPS, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var cur = stack_allocation[
        1, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    for ii in range(bs):
        var i = bs - 1 - ii
        var row = r0 + i
        var p = Float32(0)
        var k = i + 1 + t
        while k < bs:
            p += l[(r0 + k) * n + row] * xs[k]
            k += FTS_TPB
        var s = _block_sum(
            p, red.unsafe_origin_cast[MutAnyOrigin]()
        )
        if t == 0:
            cur[0] = (b[row * nrhs + j] - s) / l[row * n + row]
            xs[i] = cur[0]
        barrier()
    var e = t
    while e < bs:
        b[(r0 + e) * nrhs + j] = xs[e]
        e += FTS_TPB


def fts_upper_update_kernel(
    l: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    nrhs_in: Int32,
    r0_in: Int32,
    bs_in: Int32,
):
    """Rows above the block: b[i] -= sum_k L[k, i] x[k], k in the block
    (consecutive threads read consecutive i: coalesced)."""
    var n = Int(n_in)
    var nrhs = Int(nrhs_in)
    var r0 = Int(r0_in)
    var bs = Int(bs_in)
    var j = Int(block_idx.y)
    var i = Int(block_idx.x) * FTS_UPD_TPB + Int(thread_idx.x)
    if i >= r0:
        return
    var acc = Float32(0)
    for k in range(bs):
        acc += l[(r0 + k) * n + i] * b[(r0 + k) * nrhs + j]
    b[i * nrhs + j] = b[i * nrhs + j] - acc


def fast_cho_solve(
    ctx: DeviceContext,
    mut l: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    n: Int,
    nrhs: Int,
) raises:
    """`L L^T X = B` in place over `B`, blocked."""
    var r0 = 0
    while r0 < n:
        var bs = min(FTS_BLOCK, n - r0)
        ctx.enqueue_function[fts_lower_diag_kernel](
            l.unsafe_ptr(), b.unsafe_ptr(), Int32(n), Int32(nrhs),
            Int32(r0), Int32(bs), grid_dim=(nrhs, 1, 1), block_dim=(FTS_TPB, 1, 1),
        )
        var below = n - r0 - bs
        if below > 0:
            ctx.enqueue_function[fts_lower_update_kernel](
                l.unsafe_ptr(), b.unsafe_ptr(), Int32(n), Int32(nrhs),
                Int32(r0), Int32(bs),
                grid_dim=((below + FTS_UPD_TPB - 1) // FTS_UPD_TPB, nrhs, 1),
                block_dim=(FTS_UPD_TPB, 1, 1),
            )
        r0 += bs
    var hi = n
    while hi > 0:
        var bs = min(FTS_BLOCK, hi)
        var lo = hi - bs
        ctx.enqueue_function[fts_upper_diag_kernel](
            l.unsafe_ptr(), b.unsafe_ptr(), Int32(n), Int32(nrhs),
            Int32(lo), Int32(bs), grid_dim=(nrhs, 1, 1), block_dim=(FTS_TPB, 1, 1),
        )
        if lo > 0:
            ctx.enqueue_function[fts_upper_update_kernel](
                l.unsafe_ptr(), b.unsafe_ptr(), Int32(n), Int32(nrhs),
                Int32(lo), Int32(bs),
                grid_dim=((lo + FTS_UPD_TPB - 1) // FTS_UPD_TPB, nrhs, 1),
                block_dim=(FTS_UPD_TPB, 1, 1),
            )
        hi = lo


comptime FTP_REGS = 8
comptime FTP_MAX_NB = 32 * FTP_REGS


def fast_trsm_panel_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    j0_in: Int32,
    nb_in: Int32,
    n_trail_in: Int32,
):
    """`trsm_panel_kernel` (L21 = A21 L11^-T, row by row) with a SIMDGROUP
    per row: lane l keeps the row's solved values y[l + 32 m] in registers,
    every column's dot product over the earlier columns is split across the
    lanes (L11's row read coalesced) and folded with a warp sum, and lane 0
    finishes y_c and broadcasts it. nb <= FTP_MAX_NB."""
    var n = Int(n_in)
    var j0 = Int(j0_in)
    var nb = Int(nb_in)
    var gw = (Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)) // 32
    if gw >= Int(n_trail_in):
        return
    var lane = Int(lane_id())
    var r = j0 + nb + gw
    var y = InlineArray[Float32, FTP_REGS](fill=0.0)
    for c in range(nb):
        var jc = j0 + c
        var p = Float32(0)
        comptime for m in range(FTP_REGS):
            var k = lane + 32 * m
            if k < c:
                p += a[jc * n + j0 + k] * y[m]
        var s = warp_sum(p)
        var yc = Float32(0)
        if lane == 0:
            yc = (a[r * n + jc] - s) / a[jc * n + jc]
        yc = shuffle_idx(yc, UInt32(0))
        comptime for m in range(FTP_REGS):
            if c == lane + 32 * m:
                y[m] = yc
    comptime for m in range(FTP_REGS):
        var k = lane + 32 * m
        if k < nb:
            a[r * n + j0 + k] = y[m]


def fast_gemm_nt_sub_lower(
    ctx: DeviceContext,
    a: MutPointer[Float32, MutAnyOrigin],
    n: Int,
    base: Int,
    off: Int,
    mut shape: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    m: Int,
    cols: Int,
    k: Int,
) raises:
    """`A[base + off + r, base + off + c] -= (x y^T)[r, c]` for every `c <= r`
    (the lower triangle), `x` `m x k`, `y` `cols x k`, row-major. The
    subtraction rides the vendor GEMM's epilogue, so the product is never
    stored and read back. `shape` only gives the output tensor its extent
    (`m * cols` floats; nothing is written to it)."""

    @parameter
    @always_inline
    def epi[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]) capturing -> None:
        var i = off + idx[0]
        comptime for w in range(Int(width)):
            var j = off + idx[1] + w
            if j <= i:
                var p = (base + i) * n + base + j
                a[p] = a[p] - rebind[Float32](val[w].cast[DType.float32]())

    var tz = TileTensor(shape, row_major(m, cols))
    var tx = TileTensor(x, row_major(m, k))
    var ty = TileTensor(y, row_major(cols, k))
    matmul[
        transpose_b=True, target="gpu", elementwise_lambda_fn=epi
    ](tz, tx, ty, ctx)


def fts_identity_kernel(b: MutPointer[Float32, MutAnyOrigin], w_in: Int32):
    var w = Int(w_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= w * w:
        return
    b[i] = Float32(1.0) if i // w == i % w else Float32(0.0)


def fts_unpack_panel_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    packed: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    j0_in: Int32,
    w_in: Int32,
    n_trail_in: Int32,
):
    """`a[j0 + w + r, j0 + c] = packed[r * w + c]` (the inverse of
    `pack_panel_kernel`)."""
    var w = Int(w_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_trail_in) * w:
        return
    var r = i // w
    var c = i % w
    var n = Int(n_in)
    var j0 = Int(j0_in)
    a[(j0 + w + r) * n + j0 + c] = packed[i]


def fast_panel_solve_inv(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    n: Int,
    j0: Int,
    w: Int,
    n_trail: Int,
    mut linv: DeviceBuffer[DType.float32],
    mut praw: DeviceBuffer[DType.float32],
    mut packed: DeviceBuffer[DType.float32],
) raises:
    """`L21 = A21 L11^{-T}` as `L11^{-1}` (the blocked forward solve against
    the identity, `w <= FTS_BLOCK`) and one vendor GEMM. `praw` holds the
    packed `A21` on entry (`n_trail x w`); `packed` receives `L21` packed,
    and `a` gets it back in place."""
    var ww = w * w
    ctx.enqueue_function[fts_identity_kernel](
        linv.unsafe_ptr(), Int32(w),
        grid_dim=((ww + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    var l11 = a.unsafe_ptr() + j0 * n + j0
    ctx.enqueue_function[fts_lower_diag_kernel](
        l11, linv.unsafe_ptr(), Int32(n), Int32(w), Int32(0), Int32(w),
        grid_dim=(w, 1, 1), block_dim=(FTS_TPB, 1, 1),
    )
    var lv = linv.create_sub_buffer[DType.float32](0, ww)
    var pr = praw.create_sub_buffer[DType.float32](0, n_trail * w)
    var pk = packed.create_sub_buffer[DType.float32](0, n_trail * w)
    var tz = TileTensor(pk, row_major(n_trail, w))
    var tx = TileTensor(pr, row_major(n_trail, w))
    var ty = TileTensor(lv, row_major(w, w))
    matmul[transpose_b=True, target="gpu"](tz, tx, ty, ctx)
    ctx.enqueue_function[fts_unpack_panel_kernel](
        a.unsafe_ptr(), pk.unsafe_ptr(), Int32(n), Int32(j0), Int32(w),
        Int32(n_trail),
        grid_dim=((n_trail * w + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    _ = lv^
    _ = pr^
    _ = pk^

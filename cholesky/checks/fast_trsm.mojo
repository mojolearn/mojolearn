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

from max.gpu import WARP_SIZE, block_dim, block_idx, thread_idx
from max.gpu.primitives.warp import sum as warp_sum
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

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

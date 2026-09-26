"""FAST-only SMO gradient update without the kernel tile (Apple).

After each block solve the reference computes the kernel rows of the
working-set points whose alpha moved against EVERY training row
(`KernelCache::getNextBatchKernel`, an `nnz x batch` tile per batch) and
then folds them into `f` (`UpdateF`). The tile is written once and read
once: at 50k rows and a 2048-row working set that is ~400 MB of traffic
per outer iteration for a few GFLOP of arithmetic.

`fused_update_f_kernel` computes each kernel value where it is used: one
thread per training row keeps its row in registers, the moved points (and
their norms and alpha changes) stream through threadgroup memory, and the
thread accumulates `sum_j K(x_j, x_i) * da_j` into `f_i` (and, for SVR,
into the second half at `i + n_rows`, which shares the kernel row). FAST
arithmetic: the natural j order, a plain fused multiply-add.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import exp
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

comptime FUF_TPB = 256
comptime FUF_TILE_FLOATS = 3072
comptime FUF_MAX_K = 64


def fused_update_f_kernel[KPAD: Int, RBF: Bool](
    f: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    norm_x: MutPointer[Float32, MutAnyOrigin],
    xw: MutPointer[Float32, MutAnyOrigin],
    norm_w: MutPointer[Float32, MutAnyOrigin],
    da: MutPointer[Float32, MutAnyOrigin],
    nnz_in: Int32,
    n_rows_in: Int32,
    k_in: Int32,
    gain: Float32,
    svr: Int32,
):
    comptime ROWS = FUF_TILE_FLOATS // KPAD
    var nnz = Int(nnz_in)
    var n = Int(n_rows_in)
    var k = Int(k_in)
    var tid = Int(thread_idx.x)
    var i = Int(block_idx.x) * FUF_TPB + tid
    var active = i < n
    var tile = stack_allocation[
        FUF_TILE_FLOATS, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var tnorm = stack_allocation[
        ROWS, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var tda = stack_allocation[
        ROWS, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var xr = InlineArray[Float32, KPAD](fill=0.0)
    var ni = Float32(0.0)
    if active:
        comptime for c in range(KPAD):
            if c < k:
                xr[c] = x[i * k + c]
        comptime if RBF:
            ni = norm_x[i]
    var acc = Float32(0.0)
    var j0 = 0
    while j0 < nnz:
        var rows_here = min(ROWS, nnz - j0)
        barrier()
        var idx = tid
        while idx < rows_here * KPAD:
            var r = idx // KPAD
            var c = idx - r * KPAD
            tile[idx] = xw[(j0 + r) * k + c] if c < k else Float32(0.0)
            idx += FUF_TPB
        idx = tid
        while idx < rows_here:
            tda[idx] = da[j0 + idx]
            comptime if RBF:
                tnorm[idx] = norm_w[j0 + idx]
            idx += FUF_TPB
        barrier()
        if active:
            for r in range(rows_here):
                var dot = Float32(0.0)
                comptime for c in range(KPAD):
                    dot += xr[c] * tile[r * KPAD + c]
                var kv: Float32
                comptime if RBF:
                    kv = exp(-gain * (tnorm[r] + ni - dot * Float32(2.0)))
                else:
                    kv = dot
                acc += kv * tda[r]
        j0 += ROWS
    if active:
        f[i] = f[i] + acc
        if svr != 0:
            f[i + n] = f[i + n] + acc


def fast_update_f(
    ctx: DeviceContext,
    mut f: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut norm_x: DeviceBuffer[DType.float32],
    mut xw: DeviceBuffer[DType.float32],
    mut norm_w: DeviceBuffer[DType.float32],
    mut da: DeviceBuffer[DType.float32],
    nnz: Int,
    n_rows: Int,
    k: Int,
    gain: Float32,
    rbf: Bool,
    svr: Bool,
) raises -> Bool:
    """`f[i] += sum_j K(xw_j, x_i) * da_j` over all `n_rows` rows (and the
    SVR second half). False when `k` is too wide for this kernel."""
    if k > FUF_MAX_K or nnz <= 0:
        return False
    var kpad = ((k + 3) // 4) * 4
    var grid = (n_rows + FUF_TPB - 1) // FUF_TPB
    var s = Int32(1) if svr else Int32(0)
    comptime for KP in [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]:
        if kpad == KP:
            if rbf:
                ctx.enqueue_function[fused_update_f_kernel[KP, True]](
                    f.unsafe_ptr(), x.unsafe_ptr(), norm_x.unsafe_ptr(),
                    xw.unsafe_ptr(), norm_w.unsafe_ptr(), da.unsafe_ptr(),
                    Int32(nnz), Int32(n_rows), Int32(k), gain, s,
                    grid_dim=grid, block_dim=FUF_TPB,
                )
            else:
                ctx.enqueue_function[fused_update_f_kernel[KP, False]](
                    f.unsafe_ptr(), x.unsafe_ptr(), norm_x.unsafe_ptr(),
                    xw.unsafe_ptr(), norm_w.unsafe_ptr(), da.unsafe_ptr(),
                    Int32(nnz), Int32(n_rows), Int32(k), gain, s,
                    grid_dim=grid, block_dim=FUF_TPB,
                )
    return True

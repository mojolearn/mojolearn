"""FAST-only `X^T dZ` for the QN gradient (Apple).

`xtdz_multi_kernel` / `xty_kernel` give each output cell `(c, j)` a block
that walks every row reading `x[r * D + j]`: consecutive threads read
addresses `D` floats apart, so each load is its own cache line and the whole
of X is re-read once per cell (covtype 500k x 54, 7 classes: ~12 GB per
gradient). Here the rows are split across blocks; a block stages a tile of
X rows and of dZ rows in threadgroup memory (coalesced) and every thread
accumulates its own cells over the tile, then a second pass folds the
per-block partials. X is read once per gradient. Same `out[c + C * j]`
layout; FAST arithmetic (a different summation order).
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

comptime FX_TPB = 256
comptime FX_ROWS = 16
comptime FX_MAX_D = 256
comptime FX_CELLS_PER_THREAD = 4
comptime FX_MAX_CELLS = FX_TPB * FX_CELLS_PER_THREAD
comptime FX_MAX_C = 64


def fast_xtdz_applies(d: Int, c: Int) -> Bool:
    return d <= FX_MAX_D and c <= FX_MAX_C and d * c <= FX_MAX_CELLS


def fx_partial_kernel(
    partial: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    dz: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    d_in: Int32,
    c_in: Int32,
    chunk_in: Int32,
):
    var n = Int(n_rows_in)
    var D = Int(d_in)
    var C = Int(c_in)
    var cells = D * C
    var t = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var r_begin = b * Int(chunk_in)
    var r_end = min(n, r_begin + Int(chunk_in))
    var xs = stack_allocation[
        FX_ROWS * FX_MAX_D, Float32, address_space=AddressSpace.SHARED
    ]()
    var zs = stack_allocation[
        FX_ROWS * FX_MAX_C, Float32, address_space=AddressSpace.SHARED
    ]()
    var acc = InlineArray[Float32, FX_CELLS_PER_THREAD](fill=0.0)
    var jj = InlineArray[Int, FX_CELLS_PER_THREAD](fill=0)
    var cc = InlineArray[Int, FX_CELLS_PER_THREAD](fill=0)
    comptime for k in range(FX_CELLS_PER_THREAD):
        var cell = t + k * FX_TPB
        jj[k] = cell // C
        cc[k] = cell % C
    var r0 = r_begin
    while r0 < r_end:
        var rows = min(FX_ROWS, r_end - r0)
        var e = t
        while e < rows * D:
            xs[e] = x[r0 * D + e]
            e += FX_TPB
        e = t
        while e < rows * C:
            zs[e] = dz[r0 * C + e]
            e += FX_TPB
        barrier()
        comptime for k in range(FX_CELLS_PER_THREAD):
            if t + k * FX_TPB < cells:
                var a = acc[k]
                for r in range(rows):
                    a += xs[r * D + jj[k]] * zs[r * C + cc[k]]
                acc[k] = a
        barrier()
        r0 += rows
    comptime for k in range(FX_CELLS_PER_THREAD):
        var cell = t + k * FX_TPB
        if cell < cells:
            partial[b * cells + cell] = acc[k]


def fx_fold_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    partial: MutPointer[Float32, MutAnyOrigin],
    n_blocks_in: Int32,
    cells_in: Int32,
):
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var cells = Int(cells_in)
    if cell >= cells:
        return
    var s = Float32(0.0)
    for b in range(Int(n_blocks_in)):
        s += partial[b * cells + cell]
    # partial cells are `j * C + c`, which IS `c + C * j`.
    out_v[cell] = s


def fast_xtdz(
    ctx: DeviceContext,
    mut out_v: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut dz: DeviceBuffer[DType.float32],
    n_rows: Int,
    d: Int,
    c: Int,
) raises:
    """`out_v[c + C * j] = sum_r x[r, j] * dz[r, c]` (`x` `N x D`, `dz`
    `N x C`, both row-major). Requires `fast_xtdz_applies(d, c)`."""
    var cells = d * c
    var n_blocks = min(256, max(1, (n_rows + 2047) // 2048))
    var chunk = (n_rows + n_blocks - 1) // n_blocks
    var partial = ctx.enqueue_create_buffer[DType.float32](n_blocks * cells)
    ctx.enqueue_function[fx_partial_kernel](
        partial.unsafe_ptr(), x.unsafe_ptr(), dz.unsafe_ptr(),
        Int32(n_rows), Int32(d), Int32(c), Int32(chunk),
        grid_dim=(n_blocks, 1, 1), block_dim=(FX_TPB, 1, 1),
    )
    ctx.enqueue_function[fx_fold_kernel](
        out_v.unsafe_ptr(), partial.unsafe_ptr(), Int32(n_blocks),
        Int32(cells),
        grid_dim=((cells + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    _ = partial^

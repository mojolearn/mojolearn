# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane sk-weq-trees (2026-10-15): drop the zero-weight rows of a weighted
tree fit, on the device.

scikit-learn's tree splitter keeps only the rows whose sample_weight is not
zero (`sklearn/tree/_splitter.pyx`, `Splitter.init`: "Only work with
positively weighted samples"), so a zero-weight row shapes nothing: not the
split candidates (our quantile edges), not a node's row count
(`min_samples_leaf`, `min_samples_split`), not ExtraTrees' per-node min/max.
A fit with weight w == the fit on that row repeated w times, and weight 0 ==
the row removed (scikit-learn's `check_sample_weight_equivalence`).

The weighted tree entries call this only when the weights are not all 1.0
and the fit has no bootstrap (with a bootstrap the weights drive the draw
and a zero-weight row is never drawn). When no weight is zero nothing is
gathered and the caller keeps its buffers, so a fit without a zero weight
keeps every bit.

Steps, all on the device: a flag per row (`w != 0`), an exclusive scan
(`frs_exclusive_scan`), an emit of the kept row ids in ascending order (the
host loop's order), and gathers of X (column-major), labels and weights at
those ids. One Int (the kept count) is read back. Exact: no arithmetic on
any value, only copies.
"""
from std.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from core.device_scan import SCAN_TPB
from core.device_fold import device_count_nonzero_i32
from core.fast_radix_sort import frs_exclusive_scan, frs_scan_blocks

comptime _I32P = MutPointer[Int32, MutAnyOrigin]
comptime _F32P = MutPointer[Float32, MutAnyOrigin]


def _wc_grid(n: Int) -> Int:
    return max(1, (n + SCAN_TPB - 1) // SCAN_TPB)


def _wc_flag_kernel(w: _F32P, flag: _I32P, scan: _I32P, n_in: Int32):
    """flag[i] = scan[i] = 1 when w[i] != 0 (NaN never reaches here: the
    weight scan refused it)."""
    var i = Int(block_idx.x) * SCAN_TPB + Int(thread_idx.x)
    if i < Int(n_in):
        var f = Int32(1) if w.unsafe_load(i) != Float32(0) else Int32(0)
        flag.unsafe_store(i, f)
        scan.unsafe_store(i, f)


def _wc_emit_kernel(flag: _I32P, scan: _I32P, n_in: Int32, rows: _I32P):
    var i = Int(block_idx.x) * SCAN_TPB + Int(thread_idx.x)
    if i < Int(n_in) and flag.unsafe_load(i) != Int32(0):
        rows.unsafe_store(Int(scan.unsafe_load(i)), Int32(i))


def _wc_gather_kernel[
    dtype: DType
](
    dst: MutPointer[Scalar[dtype], MutAnyOrigin],
    src: MutPointer[Scalar[dtype], MutAnyOrigin],
    rows: _I32P,
    m_in: Int32,
):
    """dst[k] = src[rows[k]], k < m."""
    var k = Int(block_idx.x) * SCAN_TPB + Int(thread_idx.x)
    if k < Int(m_in):
        dst.unsafe_store(k, src.unsafe_load(Int(rows.unsafe_load(k))))


def _wc_gather_colmajor_kernel(
    dst: _F32P, src: _F32P, rows: _I32P, n_src: Int32, m_in: Int32, n_cols: Int32
):
    """dst[c * m + k] = src[c * n_src + rows[k]]: column-major X at the kept
    rows, one thread per output element."""
    var idx = Int(block_idx.x) * SCAN_TPB + Int(thread_idx.x)
    var m = Int(m_in)
    if idx < m * Int(n_cols):
        var c = idx // m
        var k = idx - c * m
        dst.unsafe_store(
            idx, src.unsafe_load(c * Int(n_src) + Int(rows.unsafe_load(k)))
        )


def device_nonzero_weight_rows(
    ctx: DeviceContext,
    mut w: DeviceBuffer[DType.float32],
    n: Int,
    mut rows: DeviceBuffer[DType.int32],
) raises -> Int:
    """Writes, ascending, every row `i < n` with `w[i] != 0` into `rows`
    (at least n slots) and returns how many."""
    if n <= 0:
        return 0
    var flag = ctx.enqueue_create_buffer[DType.int32](n)
    var scan = ctx.enqueue_create_buffer[DType.int32](n)
    var bsum = ctx.enqueue_create_buffer[DType.int32](max(1, frs_scan_blocks(n)))
    ctx.enqueue_function[_wc_flag_kernel](
        w.unsafe_ptr(), flag.unsafe_ptr(), scan.unsafe_ptr(), Int32(n),
        grid_dim=(_wc_grid(n), 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    frs_exclusive_scan(ctx, scan, n, bsum)
    ctx.enqueue_function[_wc_emit_kernel](
        flag.unsafe_ptr(), scan.unsafe_ptr(), Int32(n), rows.unsafe_ptr(),
        grid_dim=(_wc_grid(n), 1, 1), block_dim=(SCAN_TPB, 1, 1),
    )
    var kept = device_count_nonzero_i32(ctx, flag, n)
    _ = bsum^
    _ = scan^
    _ = flag^
    return kept


def device_gather_rows[
    dtype: DType
](
    ctx: DeviceContext,
    mut src: DeviceBuffer[dtype],
    mut rows: DeviceBuffer[DType.int32],
    m: Int,
) raises -> DeviceBuffer[dtype]:
    """A new buffer of `m` elements: src at the kept rows."""
    var dst = ctx.enqueue_create_buffer[dtype](max(1, m))
    if m > 0:
        ctx.enqueue_function[_wc_gather_kernel[dtype]](
            dst.unsafe_ptr(), src.unsafe_ptr(), rows.unsafe_ptr(), Int32(m),
            grid_dim=(_wc_grid(m), 1, 1), block_dim=(SCAN_TPB, 1, 1),
        )
    return dst^


def device_gather_rows_colmajor(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    n: Int,
    n_cols: Int,
    mut rows: DeviceBuffer[DType.int32],
    m: Int,
) raises -> DeviceBuffer[DType.float32]:
    """A new column-major `m x n_cols` X: the kept rows of the column-major
    `n x n_cols` x."""
    var dst = ctx.enqueue_create_buffer[DType.float32](max(1, m * n_cols))
    if m > 0 and n_cols > 0:
        if m * n_cols > 2147483647:
            raise Error("device_gather_rows_colmajor: m * n_cols exceeds Int32")
        ctx.enqueue_function[_wc_gather_colmajor_kernel](
            dst.unsafe_ptr(), x.unsafe_ptr(), rows.unsafe_ptr(),
            Int32(n), Int32(m), Int32(n_cols),
            grid_dim=(_wc_grid(m * n_cols), 1, 1), block_dim=(SCAN_TPB, 1, 1),
        )
    return dst^



def device_count_nonzero_weights(
    ctx: DeviceContext, weights_addr: Int, n: Int
) raises -> Int:
    """How many of the caller's `n` Float32 weights (borrowed through this
    call) are not zero: one upload, the flag pass, one Int back. ExtraTrees
    resolves its plan (the unlimited-depth cap) from the rows the fit will
    keep, as the host column does after its own drop."""
    if weights_addr == 0 or n <= 0:
        return n
    var dw = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(
        dst_buf=dw,
        src_ptr=MutPointer[Float32, MutUntrackedOrigin](
            unsafe_from_address=weights_addr
        ),
    )
    var rows = ctx.enqueue_create_buffer[DType.int32](n)
    var kept = device_nonzero_weight_rows(ctx, dw, n, rows)
    _ = rows^
    _ = dw^
    return kept

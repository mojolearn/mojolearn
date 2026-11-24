# SPDX-License-Identifier: Apache-2.0
"""The resident geqrf, orgqr and triu of the linalg QR routes (lane vdf-linalg2, 2026-11-24).

`mojolearn.linalg.qr` (modes 'complete', 'reduced' off the TSQR route, 'raw' and the wide 'r') ran geqrf through the
host-address operation XD_GEQRF: upload A, factor on the device, download h and tau; orgqr then uploaded h and tau
AGAIN to form Q; and R was cut from the downloaded h on the host (`_lin_triu`, one zero fill per row, tagged
cpu-route). Here the three run on pool ids: A is uploaded once, h, tau, Q and R stay on the device and each result is
downloaded once. The kernels are the same (`qs_geqrf_device`, `qs_orgqr_device`, x_decomp/qr_sliced_device.mojo) on
the same words, so Q and h are the same bits; triu is data movement (a copy and zeros).
"""
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer
from x_decomp.cells import F32Ptr
from x_decomp.device import xd_ctx
from x_decomp.resident import pool_buffer
from x_decomp.qr_sliced_device import qs_geqrf_device, qs_orgqr_device

comptime RQ_TPB = 256


def dev_geqrf_run(ia: Int, ih: Int, it: Int, m: Int, n: Int) raises -> Int:
    """geqrf of the device matrix a (m x n, left as it is) into the device
    matrix h (m x n, LAPACK's factored form) and tau (min(m, n)). Returns
    min(m, n). The checks of `geqrf_run` (x_decomp/api.mojo)."""
    if m <= 0 or n <= 0:
        raise Error("x_decomp: geqrf needs m, n >= 1")
    var cells = m * n
    if cells > 2147483647:
        raise Error("x_decomp: geqrf exceeds the Int32 index bound")
    var kk = m if m < n else n
    var source = pool_buffer(ia, cells)
    var factored = pool_buffer(ih, cells)
    var taus = pool_buffer(it, kk)
    var ctx = xd_ctx()
    var dh = factored.create_sub_buffer[DType.float32](0, cells)
    var dt = taus.create_sub_buffer[DType.float32](0, kk)
    ctx.enqueue_copy(dst_buf=dh, src_buf=source.create_sub_buffer[DType.float32](0, cells))
    qs_geqrf_device(ctx, dh, dt, m, n)
    ctx.synchronize()
    _ = dh^
    _ = dt^
    _ = source^
    _ = factored^
    _ = taus^
    return kk


def dev_orgqr_run(ih: Int, it: Int, iq: Int, m: Int, n: Int, kk: Int, qc: Int) raises -> Int:
    """orgqr of the device factored form h (m x n) and tau (kk) into the
    device matrix q (m x qc). The checks of `orgqr_run`."""
    if m <= 0 or n <= 0 or kk > m or kk > n or qc > m or qc <= 0:
        raise Error("x_decomp: orgqr needs 1 <= qc <= m and kk <= min(m, n)")
    var factored = pool_buffer(ih, m * n)
    var taus = pool_buffer(it, kk if kk > 0 else 1)
    var target = pool_buffer(iq, m * qc)
    var ctx = xd_ctx()
    var dh = factored.create_sub_buffer[DType.float32](0, m * n)
    var dt = taus.create_sub_buffer[DType.float32](0, kk if kk > 0 else 1)
    var dq = target.create_sub_buffer[DType.float32](0, m * qc)
    qs_orgqr_device(ctx, dh, dt, dq, m, n, kk, qc)
    ctx.synchronize()
    _ = dh^
    _ = dt^
    _ = dq^
    _ = factored^
    _ = taus^
    _ = target^
    return qc


def _triu_rows_kernel(src: F32Ptr, dst: F32Ptr, r_in: Int32, c_in: Int32):
    """dst (r x c) = the first r rows of src (c columns), every word below
    the diagonal zero: one word per thread."""
    var r = Int(r_in)
    var c = Int(c_in)
    var t = Int(block_idx.x) * RQ_TPB + Int(thread_idx.x)
    if t >= r * c:
        return
    var row = t // c
    var col = t - row * c
    if col < row:
        dst.unsafe_store(t, Float32(0.0))
    else:
        dst.unsafe_store(t, src.unsafe_load(t))


def dev_triu_run(ih: Int, io: Int, r: Int, c: Int) raises -> Int:
    """R = triu of the first r rows of the device matrix h (c columns) into
    the device matrix o (r x c). Was the door's host `_lin_triu`."""
    if r < 0 or c < 0 or r * c > 2147483647:
        raise Error("x_decomp: triu shape out of range")
    var cells = r * c
    if cells == 0:
        return 0
    var source = pool_buffer(ih, cells)
    var target = pool_buffer(io, cells)
    var ctx = xd_ctx()
    var ps = F32Ptr(unsafe_from_address=Int(source.unsafe_ptr()))
    var po = F32Ptr(unsafe_from_address=Int(target.unsafe_ptr()))
    ctx.enqueue_function[_triu_rows_kernel](
        ps, po, Int32(r), Int32(c), grid_dim=(cells + RQ_TPB - 1) // RQ_TPB, block_dim=RQ_TPB
    )
    ctx.synchronize()
    _ = source^
    _ = target^
    return cells

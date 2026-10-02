# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SLICED geqrf / orgqr ON THE DEVICE (lane hr-qr, 2026-10-02): the kernels
of x_decomp/qr_sliced.mojo's order. x_decomp/qr_sliced_host.mojo is their
host replay.

THE LAUNCHES, per step k (no host step, no host read-back between steps):
  geqrf: `qs_norm_part_kernel` (one thread per slice: its (s, q) pair),
         `qs_head_kernel` (one thread: the pair tree, dlarfg),
         `qs_scale_kernel` (one thread per row: the multipliers),
         `qs_dot_part_kernel` (one thread per (column, slice)),
         `qs_dot_fold_kernel` (one thread per column: the tree),
         `qs_update_kernel` (one thread per cell);
  orgqr: the last three over Q's columns, k descending.
Thread (column, slice) of `qs_dot_part_kernel`: a block is 32 consecutive
columns x 8 slices, so a warp reads one row segment of 32 consecutive words
per step of its chains and the reflector's word is a broadcast. No
threadgroup memory and no barrier anywhere.

Every launch is sliced to about QS_LAUNCH_CELLS multiply-adds (slices or
rows per launch; the slicing never changes a bit, the cells of one launch
are independent), and on the Apple column the host waits whenever the
enqueued work since the last wait passes QS_LAUNCH_CELLS: macOS cuts a long
command buffer silently.
"""
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN
from checks.numerics import ftz, identical_mul_add
from x_decomp.cells import F32Ptr, geqrf_scale_elem, geqrf_update_elem, orgqr_init_elem, orgqr_update_elem
from x_decomp.qr_sliced import (
    QS_LAUNCH_CELLS,
    QS_ROWS,
    qs_dot_finish,
    qs_head,
    qs_slice_hi,
    qs_slice_lo,
    qs_slice_ssq,
    qs_slices,
)

comptime QS_TPB = 256
#: columns x slices of one dot block
comptime QS_COLS = 32
comptime QS_SL = QS_TPB // QS_COLS


@always_inline
def _grid(count: Int) -> Int:
    return (count + QS_TPB - 1) // QS_TPB if count > 0 else 1


def qs_norm_part_kernel(a: F32Ptr, ps: F32Ptr, pq: F32Ptr, k: Int32, m: Int32, n: Int32, ns: Int32):
    """Slice c's (s, q) of column k below the diagonal."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c >= Int(ns):
        return
    var K = Int(k)
    var r = qs_slice_ssq(a, Int(n), K, qs_slice_lo(K, c), qs_slice_hi(K, c, Int(m)))
    ps.unsafe_store(c, r[0])
    pq.unsafe_store(c, r[1])


def qs_head_kernel(a: F32Ptr, tau: F32Ptr, scal: F32Ptr, ps: F32Ptr, pq: F32Ptr, k: Int32, n: Int32, ns: Int32):
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        qs_head(a, tau, scal, ps, pq, Int(k), Int(n), Int(ns))


def qs_scale_kernel(a: F32Ptr, scal: F32Ptr, k: Int32, m: Int32, n: Int32):
    var i = Int(k) + 1 + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(m):
        geqrf_scale_elem(a, scal, Int(k), i, Int(n))


def qs_dot_part_kernel(
    x: F32Ptr, xs: Int32, y: F32Ptr, ys: Int32, gate: F32Ptr, gi: Int32, dp: F32Ptr, nsmax: Int32,
    k: Int32, m: Int32, j0: Int32, ncols: Int32, c0: Int32, cnt: Int32,
):
    """Partial p[j, c] = the chain of ftz(x[i, k]) ftz(y[i, j]) over slice c's
    rows ascending, from 0, for columns j0 .. j0 + ncols - 1 and slices
    c0 .. c0 + cnt - 1; nothing when ftz(gate[gi]) is 0 (the step does not act)."""
    if ftz(gate.unsafe_load(Int(gi))) == Float32(0):
        return
    var tid = Int(thread_idx.x)
    var ncg = (Int(ncols) + QS_COLS - 1) // QS_COLS
    var b = Int(block_idx.x)
    var jo = (b % ncg) * QS_COLS + tid % QS_COLS
    var co = (b // ncg) * QS_SL + tid // QS_COLS
    if jo >= Int(ncols) or co >= Int(cnt):
        return
    var j = Int(j0) + jo
    var c = Int(c0) + co
    var K = Int(k)
    var X = Int(xs)
    var Y = Int(ys)
    var lo = qs_slice_lo(K, c)
    var hi = qs_slice_hi(K, c, Int(m))
    var acc = Float32(0)
    var i = lo
    var body = hi - ((hi - lo) % 4)
    while i < body:
        var x0 = ftz(x.unsafe_load(i * X + K))
        var x1 = ftz(x.unsafe_load((i + 1) * X + K))
        var x2 = ftz(x.unsafe_load((i + 2) * X + K))
        var x3 = ftz(x.unsafe_load((i + 3) * X + K))
        var y0 = ftz(y.unsafe_load(i * Y + j))
        var y1 = ftz(y.unsafe_load((i + 1) * Y + j))
        var y2 = ftz(y.unsafe_load((i + 2) * Y + j))
        var y3 = ftz(y.unsafe_load((i + 3) * Y + j))
        acc = ftz(identical_mul_add(x0, y0, acc))
        acc = ftz(identical_mul_add(x1, y1, acc))
        acc = ftz(identical_mul_add(x2, y2, acc))
        acc = ftz(identical_mul_add(x3, y3, acc))
        i += 4
    while i < hi:
        acc = ftz(identical_mul_add(ftz(x.unsafe_load(i * X + K)), ftz(y.unsafe_load(i * Y + j)), acc))
        i += 1
    dp.unsafe_store(j * Int(nsmax) + c, acc)


def qs_dot_fold_kernel(
    y: F32Ptr, ys: Int32, gate: F32Ptr, gi: Int32, dp: F32Ptr, nsmax: Int32, w: F32Ptr, k: Int32, j0: Int32, ncols: Int32, ns: Int32
):
    """w[j] = `qs_dot_finish`(y[k, j], the slice partials of column j); 0
    when the step does not act."""
    var jo = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if jo >= Int(ncols):
        return
    var j = Int(j0) + jo
    if ftz(gate.unsafe_load(Int(gi))) == Float32(0):
        w.unsafe_store(j, Float32(0))
        return
    w.unsafe_store(j, qs_dot_finish(y.unsafe_load(Int(k) * Int(ys) + j), dp + j * Int(nsmax), Int(ns)))


def qs_geqrf_update_kernel(a: F32Ptr, tau: F32Ptr, scal: F32Ptr, w: F32Ptr, k: Int32, n: Int32, r0: Int32, rows: Int32):
    """`geqrf_update_elem` for rows r0 .. r0 + rows - 1 (>= k), columns > k."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var K = Int(k)
    var N = Int(n)
    var cols = N - K - 1
    if cols <= 0 or t >= Int(rows) * cols:
        return
    var i = Int(r0) + t // cols
    var j = K + 1 + t % cols
    geqrf_update_elem(a, tau, scal, K, i, j, N, w.unsafe_load(j))


def qs_orgqr_init_kernel(q: F32Ptr, m: Int32, qc: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(m) * Int(qc):
        orgqr_init_elem(q, t // Int(qc), t % Int(qc), Int(qc))


def qs_orgqr_update_kernel(h: F32Ptr, tau: F32Ptr, q: F32Ptr, w: F32Ptr, k: Int32, n: Int32, qc: Int32, r0: Int32, rows: Int32):
    """`orgqr_update_elem` for rows r0 .. r0 + rows - 1 (>= k), every column."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var QC = Int(qc)
    if t >= Int(rows) * QC:
        return
    var i = Int(r0) + t // QC
    var j = t % QC
    orgqr_update_elem(h, tau, q, Int(k), i, j, Int(n), QC, w.unsafe_load(j))


@always_inline
def _ptr(buf: DeviceBuffer[DType.float32]) -> F32Ptr:
    return F32Ptr(unsafe_from_address=Int(buf.unsafe_ptr()))


struct _Pace(Movable):
    """The Apple column's wait: the host synchronizes once the work enqueued
    since the last wait passes QS_LAUNCH_CELLS (other columns never wait)."""
    var cells: Int

    def __init__(out self):
        self.cells = 0

    def add(mut self, ctx: DeviceContext, cells: Int) raises:
        comptime if TARGET_COLUMN == COLUMN_APPLE:
            self.cells += cells
            if self.cells >= QS_LAUNCH_CELLS:
                ctx.synchronize()
                self.cells = 0


def _dots(
    ctx: DeviceContext, mut pace: _Pace, x: F32Ptr, xs: Int, y: F32Ptr, ys: Int, gate: F32Ptr, gi: Int,
    dp: F32Ptr, nsmax: Int, w: F32Ptr, k: Int, m: Int, j0: Int, ncols: Int,
) raises:
    """w[j0 .. j0 + ncols) for step k: the partial launches (sliced to
    QS_LAUNCH_CELLS), then the fold."""
    var ns = qs_slices(m - k - 1)
    if ncols <= 0:
        return
    if ns > 0:
        var per = QS_LAUNCH_CELLS // max(ncols * QS_ROWS, 1)
        per = max(QS_SL, (per // QS_SL) * QS_SL)
        var ncg = (ncols + QS_COLS - 1) // QS_COLS
        var c0 = 0
        while c0 < ns:
            var cnt = min(per, ns - c0)
            var nsg = (cnt + QS_SL - 1) // QS_SL
            ctx.enqueue_function[qs_dot_part_kernel](
                x, Int32(xs), y, Int32(ys), gate, Int32(gi), dp, Int32(nsmax),
                Int32(k), Int32(m), Int32(j0), Int32(ncols), Int32(c0), Int32(cnt),
                grid_dim=ncg * nsg, block_dim=QS_TPB,
            )
            pace.add(ctx, ncols * cnt * QS_ROWS)
            c0 += cnt
    ctx.enqueue_function[qs_dot_fold_kernel](
        y, Int32(ys), gate, Int32(gi), dp, Int32(nsmax), w, Int32(k), Int32(j0), Int32(ncols), Int32(ns),
        grid_dim=_grid(ncols), block_dim=QS_TPB,
    )
    pace.add(ctx, ncols * ns)


def qs_geqrf_device(ctx: DeviceContext, da: DeviceBuffer[DType.float32], dt: DeviceBuffer[DType.float32], m: Int, n: Int) raises:
    """geqrf in place on the device-resident row-major m x n `da`, tau into
    `dt` (min(m, n) floats); the caller downloads and synchronizes."""
    var kk = m if m < n else n
    var nsmax = max(qs_slices(m - 1), 1)
    var ds = ctx.enqueue_create_buffer[DType.float32](2)
    var dw = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    var dp = ctx.enqueue_create_buffer[DType.float32](max(n, 1) * nsmax)
    var dps = ctx.enqueue_create_buffer[DType.float32](nsmax)
    var a = _ptr(da)
    var t = _ptr(dt)
    var s = _ptr(ds)
    var w = _ptr(dw)
    var p = _ptr(dp)
    var pms = _ptr(dps)
    var pace = _Pace()
    for k in range(kk):
        var ns = qs_slices(m - k - 1)
        # the norm's pairs land in column k's partial row, free at step k
        var pq = p + k * nsmax
        if ns > 0:
            ctx.enqueue_function[qs_norm_part_kernel](
                a, pms, pq, Int32(k), Int32(m), Int32(n), Int32(ns), grid_dim=_grid(ns), block_dim=QS_TPB
            )
        ctx.enqueue_function[qs_head_kernel](a, t, s, pms, pq, Int32(k), Int32(n), Int32(ns), grid_dim=1, block_dim=1)
        pace.add(ctx, 2 * (m - k))
        if ns > 0:
            ctx.enqueue_function[qs_scale_kernel](a, s, Int32(k), Int32(m), Int32(n), grid_dim=_grid(m - k - 1), block_dim=QS_TPB)
        var cols = n - k - 1
        if cols <= 0:
            continue
        _dots(ctx, pace, a, n, a, n, s, 1, p, nsmax, w, k, m, k + 1, cols)
        var per = max(1, QS_LAUNCH_CELLS // cols)
        var r0 = k
        while r0 < m:
            var rows = min(per, m - r0)
            ctx.enqueue_function[qs_geqrf_update_kernel](
                a, t, s, w, Int32(k), Int32(n), Int32(r0), Int32(rows), grid_dim=_grid(rows * cols), block_dim=QS_TPB
            )
            pace.add(ctx, rows * cols)
            r0 += rows
    ctx.synchronize()
    _ = ds^
    _ = dw^
    _ = dp^
    _ = dps^


def qs_orgqr_device(
    ctx: DeviceContext, dh: DeviceBuffer[DType.float32], dt: DeviceBuffer[DType.float32], dq: DeviceBuffer[DType.float32],
    m: Int, n: Int, kk: Int, qc: Int,
) raises:
    """The first qc columns of Q = H_0 ... H_{kk-1} into the device-resident
    m x qc `dq` from geqrf's (dh, dt); the caller downloads and synchronizes."""
    if qc <= 0:
        return
    var nsmax = max(qs_slices(m - 1), 1)
    var dw = ctx.enqueue_create_buffer[DType.float32](qc)
    var dp = ctx.enqueue_create_buffer[DType.float32](qc * nsmax)
    var h = _ptr(dh)
    var t = _ptr(dt)
    var q = _ptr(dq)
    var w = _ptr(dw)
    var p = _ptr(dp)
    var pace = _Pace()
    ctx.enqueue_function[qs_orgqr_init_kernel](q, Int32(m), Int32(qc), grid_dim=_grid(m * qc), block_dim=QS_TPB)
    for r in range(kk):
        var k = kk - 1 - r
        _dots(ctx, pace, h, n, q, qc, t, k, p, nsmax, w, k, m, 0, qc)
        var per = max(1, QS_LAUNCH_CELLS // qc)
        var r0 = k
        while r0 < m:
            var rows = min(per, m - r0)
            ctx.enqueue_function[qs_orgqr_update_kernel](
                h, t, q, w, Int32(k), Int32(n), Int32(qc), Int32(r0), Int32(rows), grid_dim=_grid(rows * qc), block_dim=QS_TPB
            )
            pace.add(ctx, rows * qc)
            r0 += rows
    ctx.synchronize()
    _ = dw^
    _ = dp^

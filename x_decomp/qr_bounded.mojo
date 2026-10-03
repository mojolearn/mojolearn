# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`core/householder_qr.mojo::qr_factor`, BOUNDED IN WORK PER LAUNCH
(lane/lle-timeout, 2026-09-29).

`qr_panel_kernel` factors a whole slice in ONE launch: a square n x n
matrix is one threadgroup doing O(n^3) work behind n^2 / 2 barriers (0.75 s
at 500 columns on the M2 Pro, tens of seconds at 2,000). macOS aborts a long
Metal launch SILENTLY (a ~4 s launch on the M2 Pro was cut, its output partly
stale, synchronize silent), and a stale R became a stale embedding with no
error. `qr_factor_bounded` runs the passes `qr_factor` runs (the same slice
count, the same TSQR stack) as the split launches, one column step per
launch, waits for the device every QRB_SPLIT_SYNC columns, and POISONS r_out
(and the TSQR tiles) with NaN first: every cell of them is written by a
finished pass, so a cut launch leaves a NaN the caller's read-back refuses
(`DevExec.svd`).
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import block_idx, thread_idx

from checks.numerics import ftz, identical_div, identical_mul_add, identical_sqrt
from core.device_zero import enqueue_fill
from core.householder_qr import (
    QR_TPB,
    fold_and_broadcast,
    qr_reflector_r,
    qr_reflector_tau,
    qr_reflector_u1,
    qr_slice_count,
)
from core.householder_qr import QR_APPLY_TPB, QR_APPLY_WARPS, qr_apply_kernel, qr_r_copy_kernel, qr_reflector_kernel

#: The callers' per-launch cell budget (kept in the signature; the split
#: launch is one column step).
comptime QRB_CELLS = 1 << 24
comptime QRB_SPLIT_SYNC = 32


def _qrb_pass(
    ctx: DeviceContext,
    a: MutPointer[Float32, MutAnyOrigin],
    r_out: MutPointer[Float32, MutAnyOrigin],
    m: Int,
    n: Int,
    ns: Int,
    cells: Int,
) raises:
    """One `qr_factor` pass (grid `ns`, one block per slice) as the split
    launches (core/householder_qr.mojo qr_reflector_kernel + qr_apply_kernel
    per column, then the R copy): a launch is one column step, so a wait
    every QRB_SPLIT_SYNC columns keeps every launch short and the cut-launch
    poison check as it was. `cells` is kept for the callers; the split launch
    is already bounded (lane cgr5-owed2 deleted the MOJOLEARN_QR_SPLIT=0 arm,
    `qr_panel_cols_kernel`)."""
    _ = cells
    var tau = ctx.enqueue_create_buffer[DType.float32](ns if ns > 0 else 1)
    for j in range(n):
        ctx.enqueue_function[qr_reflector_kernel](
            a, r_out, tau.unsafe_ptr(), Int32(m), Int32(n), Int32(n), Int32(ns), Int32(j),
            grid_dim=(ns, 1, 1), block_dim=(QR_TPB, 1, 1),
        )
        var cols = n - j - 1
        if cols > 0:
            ctx.enqueue_function[qr_apply_kernel](
                a, tau.unsafe_ptr(), Int32(m), Int32(n), Int32(n), Int32(ns), Int32(j),
                grid_dim=((cols + QR_APPLY_WARPS - 1) // QR_APPLY_WARPS, ns, 1),
                block_dim=(QR_APPLY_TPB, 1, 1),
            )
        if (j + 1) % QRB_SPLIT_SYNC == 0:
            ctx.synchronize()
    ctx.enqueue_function[qr_r_copy_kernel](
        a, r_out, Int32(m), Int32(n), Int32(n), Int32(ns), grid_dim=(ns, 1, 1), block_dim=(QR_TPB, 1, 1),
    )
    ctx.synchronize()
    _ = tau^


def qr_factor_bounded(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut r_scratch: DeviceBuffer[DType.float32],
    mut r_out: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    cells: Int = QRB_CELLS,
) raises -> Int:
    """`qr_factor(ctx, a, r_scratch, r_out, n_rows, n_cols)` (the same slices,
    the same passes, the same bits; DESTROYS `a`), each pass cut into bounded
    launches of about `cells` cells (the bits do not depend on it), r_out
    (and the TSQR tiles) poisoned with NaN first."""
    if n_rows < n_cols:
        raise Error("qr_factor_bounded needs at least as many rows as columns, got "
                    + String(n_rows) + " x " + String(n_cols))
    var ns = qr_slice_count(n_rows, n_cols)
    var pa = MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(a.unsafe_ptr()))
    var ps = MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(r_scratch.unsafe_ptr()))
    var pr = MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(r_out.unsafe_ptr()))
    enqueue_fill(ctx, r_out, Float32(0.0) / Float32(0.0))
    if ns == 1:
        _qrb_pass(ctx, pa, pr, n_rows, n_cols, 1, cells)
        return 1
    enqueue_fill(ctx, r_scratch, Float32(0.0) / Float32(0.0))
    _qrb_pass(ctx, pa, ps, n_rows, n_cols, ns, cells)
    _qrb_pass(ctx, ps, pr, ns * n_cols, n_cols, 1, cells)
    return ns

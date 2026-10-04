# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The blocked right-looking Cholesky of lane/apple-fast-decomp-linalg
(2026-10-02, pass 2; recovered onto main 2026-10-04), FAST on Apple only,
behind -D MOJOLEARN_CHOL_FAST_BLOCKED (`CHOL_FAST_BLOCKED` below).
Two callers launch it, each under its own FAST + Apple guard and the
define: `cholesky/checks/potrf.mojo potrf_lower` (the board's cholesky lane,
the gp binding) and `x_decomp/device.mojo` (the kit's `chol` and the Gram
Cholesky inside the CholeskyQR2 pass of `orth_on_device_diag`). IDENTICAL
and every other column never reach these kernels.

Per panel of CH_NB columns, three launches: the diagonal block factored in
threadgroup memory by one block of fixed size (`chol_panel_kernel`), the
rows below it solved against that block one thread per row with the
mirror cells zeroed (`chol_trsm_kernel`), and the trailing symmetric
update as a register-blocked tile kernel (`chol_trail_rb_kernel`: 64 x 64
cells a block, 4 x 4 a thread, the operands staged through threadgroup
memory, tiles above the diagonal skipped). 3 n / CH_NB launches. Lower
factor, upper triangle +0.0, info[0] = k + 1 at the first non-positive
pivot (the pivot then taken as 1 and the factorization continued, as
`chol_serial` does)."""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.kernel_matrix import TARGET_COLUMN, lib_smem_page_fits_for
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_mul_add
from x_decomp.cells import F32Ptr, div0, sqrt0

comptime CH_NB = 32
comptime CH_TPB = 256
comptime CH_TILE = 16
comptime CH_RB = 4
comptime CH_RB_TILE = CH_TILE * CH_RB
comptime CH_TILE_TPB = CH_TILE * CH_TILE
comptime CH_SMEM_BYTES = (2 * CH_RB_TILE * CH_NB) * 4
comptime CH_FITS = lib_smem_page_fits_for[TARGET_COLUMN, CH_SMEM_BYTES]()
"""Whether the trail kernel's two slabs (16 KB) fit one shared page on this
column; the define below includes it."""

#: Recovered candidate (lane/apple-fast-rec-decomp, 2026-10-04), default OFF,
#: FAST + Apple only. Source lane/apple-fast-decomp-linalg@74d52352b (pass 2,
#: 792161a0a). What it does: the whole Cholesky as this file's blocked
#: right-looking route (32-column panels; 3 n / 32 launches, no vendor GEMM,
#: no workspace) from `cholesky/checks/potrf.mojo potrf_lower` (the board's
#: cholesky lane through the gp binding, ahead of the CHOL_FAST_NB route with
#: the left-looking MMA update) and from x_decomp/device.mojo (the kit's
#: `chol` and the Gram Cholesky of SVD_FAST_CHOLQR). Known: prior M3 B arm
#: (dlin-chol-blocked-synthetic, lane/apple-fast-batch prebuilt arms, old
#: head) cholesky synthetic 476 ms against the board's FAST 261 ms (main's
#: CHOL_APPLE_LEFT / NOSYNC route), so the lead is negative unless main's
#: route regressed; never had a same-build A arm and no failure was recorded.
#: Fixed in the port: the define moved here (one FAST + Apple guard for both
#: callers instead of two copies), and the SVD_FAST_CHOLQR caller no longer
#: scans the factor on the host. The trail tile (64 x 64, 16 KB of slabs) is
#: the kernel's shape, gated by the shared-page fit (CH_FITS); no dimension
#: window. A non-positive pivot sets info = k + 1 and continues with the
#: pivot taken as 1 (x_decomp `chol_serial`'s rule), so a failed factor's
#: tail differs from potrf's stopped one.
comptime CHOL_FAST_BLOCKED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_CHOL_FAST_BLOCKED"]()
    and CH_FITS
)


def ch_info_init_kernel(info: F32Ptr):
    if block_idx.x == 0 and thread_idx.x == 0:
        info.unsafe_store(0, Float32(0))


def _blocks(count: Int) -> Int:
    return (count + CH_TPB - 1) // CH_TPB if count > 0 else 1


def chol_panel_kernel(a: F32Ptr, info: F32Ptr, k0: Int32, ld: Int32, w: Int32):
    """The w x w (w <= CH_NB, a fixed panel width) diagonal block at (k0, k0)
    of the ld-strided matrix, already trailing-updated by the panels before
    it, factored in threadgroup memory: column j's diagonal on thread 0
    (its j-long chain inside the block, `chol_diag`'s statements and info
    rule), then its rows below one thread each (`chol_col_elem`'s chain
    inside the block). Written back with the block's upper cells zero. One
    block over a fixed size: nothing here grows with the matrix."""
    var kk0 = Int(k0)
    var nn = Int(ld)
    var ww = Int(w)
    var tid = Int(thread_idx.x)
    var sb = stack_allocation[CH_NB * CH_NB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var q = tid
    while q < ww * ww:
        var r = q // ww
        var c = q - r * ww
        sb[r * CH_NB + c] = a.unsafe_load((kk0 + r) * nn + kk0 + c)
        q += CH_TPB
    barrier()
    for j in range(ww):
        if tid == 0:
            var acc = ftz(sb[j * CH_NB + j])
            for p in range(j):
                var l = ftz(sb[j * CH_NB + p])
                acc = ftz(identical_mul_add(-l, l, acc))
            if not (acc > Float32(0)):
                if info.unsafe_load(0) == Float32(0):
                    info.unsafe_store(0, Float32(kk0 + j + 1))
                acc = Float32(1)
            sb[j * CH_NB + j] = sqrt0(acc)
        barrier()
        var i = j + 1 + tid
        while i < ww:
            var d = sb[j * CH_NB + j]
            var acc = ftz(sb[i * CH_NB + j])
            for p in range(j):
                acc = ftz(identical_mul_add(-ftz(sb[i * CH_NB + p]), ftz(sb[j * CH_NB + p]), acc))
            sb[i * CH_NB + j] = div0(acc, d)
            i += CH_TPB
        barrier()
    q = tid
    while q < ww * ww:
        var r = q // ww
        var c = q - r * ww
        a.unsafe_store((kk0 + r) * nn + kk0 + c, sb[r * CH_NB + c] if c <= r else Float32(0))
        q += CH_TPB


def chol_trsm_kernel(a: F32Ptr, k0: Int32, k1: Int32, n: Int32):
    """Row i >= k1 of the panel's columns [k0, k1): a[i, j] = (a[i, j] -
    sum_{k0 <= p < j} a[i, p] L[j, p]) / L[j, j] over j ascending, L the
    factored diagonal block staged in threadgroup memory; the mirror cells
    a[j, i] zeroed. One thread per row, reading only its own row and the
    block."""
    var kk0 = Int(k0)
    var kk1 = Int(k1)
    var nn = Int(n)
    var ww = kk1 - kk0
    var tid = Int(thread_idx.x)
    var lb = stack_allocation[CH_NB * CH_NB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var q = tid
    while q < ww * ww:
        var r = q // ww
        var c = q - r * ww
        lb[r * CH_NB + c] = a.unsafe_load((kk0 + r) * nn + kk0 + c)
        q += Int(block_dim.x)
    barrier()
    var i = kk1 + Int(block_idx.x) * Int(block_dim.x) + tid
    if i < nn:
        for jj in range(ww):
            var acc = ftz(a.unsafe_load(i * nn + kk0 + jj))
            for p in range(jj):
                acc = ftz(identical_mul_add(-ftz(a.unsafe_load(i * nn + kk0 + p)), ftz(lb[jj * CH_NB + p]), acc))
            a.unsafe_store(i * nn + kk0 + jj, div0(acc, lb[jj * CH_NB + jj]))
            a.unsafe_store((kk0 + jj) * nn + i, Float32(0))


def chol_trail_rb_kernel(a: F32Ptr, k0: Int32, k1: Int32, n: Int32):
    """The trailing cells (i, j), i >= j >= k1: a[i, j] -= sum_{k0 <= p < k1}
    a[i, p] a[j, p] (the panel's columns, final), x_decomp/device.mojo `lu_trail_rb_kernel`'s
    shape with the U slab the panel rows' transpose: each thread CH_RB x
    CH_RB cells of a 64 x 64 tile, the slabs through threadgroup memory. A
    tile wholly above the diagonal returns at once; the cells above the
    diagonal of a diagonal tile are computed and later zeroed by the panel
    and trsm kernels that own them."""
    var nn = Int(n)
    var kk0 = Int(k0)
    var kk1 = Int(k1)
    var width = kk1 - kk0
    var i0 = kk1 + Int(block_idx.y) * CH_RB_TILE
    var j0 = kk1 + Int(block_idx.x) * CH_RB_TILE
    if j0 > i0:
        return
    var tid = Int(thread_idx.x)
    var ty = tid // CH_TILE
    var tx = tid - ty * CH_TILE
    var ls = stack_allocation[CH_NB * CH_RB_TILE, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var us = stack_allocation[CH_NB * CH_RB_TILE, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var q = tid
    while q < CH_RB_TILE * width:
        var rr = q // width
        var cc = q - rr * width
        var v = Float32(0)
        if i0 + rr < nn:
            v = ftz(a.unsafe_load((i0 + rr) * nn + kk0 + cc))
        ls[cc * CH_RB_TILE + rr] = v
        q += CH_TILE_TPB
    q = tid
    while q < CH_RB_TILE * width:
        var rr = q // width
        var cc = q - rr * width
        var v = Float32(0)
        if j0 + rr < nn:
            v = ftz(a.unsafe_load((j0 + rr) * nn + kk0 + cc))
        us[cc * CH_RB_TILE + rr] = v
        q += CH_TILE_TPB
    barrier()
    var acc = InlineArray[Float32, CH_RB * CH_RB](fill=Float32(0))
    comptime for r in range(CH_RB):
        comptime for c in range(CH_RB):
            var i = i0 + ty + CH_TILE * r
            var j = j0 + tx + CH_TILE * c
            if i < nn and j < nn:
                acc[r * CH_RB + c] = ftz(a.unsafe_load(i * nn + j))
    for kp in range(width):
        var lv = InlineArray[Float32, CH_RB](fill=Float32(0))
        var uv = InlineArray[Float32, CH_RB](fill=Float32(0))
        comptime for r in range(CH_RB):
            lv[r] = ls[kp * CH_RB_TILE + ty + CH_TILE * r]
        comptime for c in range(CH_RB):
            uv[c] = us[kp * CH_RB_TILE + tx + CH_TILE * c]
        comptime for r in range(CH_RB):
            comptime for c in range(CH_RB):
                acc[r * CH_RB + c] = ftz(identical_mul_add(-lv[r], uv[c], ftz(acc[r * CH_RB + c])))
    comptime for r in range(CH_RB):
        comptime for c in range(CH_RB):
            var i = i0 + ty + CH_TILE * r
            var j = j0 + tx + CH_TILE * c
            if i < nn and j < nn:
                a.unsafe_store(i * nn + j, acc[r * CH_RB + c])


def launch_chol_blocked(ctx: DeviceContext, a: F32Ptr, info: F32Ptr, n: Int) raises:
    """The blocked Cholesky's launches on device pointers, enqueued (no
    sync): info cleared, then per panel of CH_NB columns the panel block,
    the rows below it and the trailing update (see the kernels above)."""
    ctx.enqueue_function[ch_info_init_kernel](info, grid_dim=1, block_dim=1)
    # the panel block's launch carries the row stride and the panel width
    # (at most CH_NB): its one block covers a fixed-size diagonal block,
    # not the matrix
    var ld = n
    var k0 = 0
    while k0 < n:
        var k1 = min(k0 + CH_NB, n)
        ctx.enqueue_function[chol_panel_kernel](
            a, info, Int32(k0), Int32(ld), Int32(k1 - k0), grid_dim=1, block_dim=CH_TPB
        )
        if k1 < n:
            ctx.enqueue_function[chol_trsm_kernel](
                a, Int32(k0), Int32(k1), Int32(n), grid_dim=_blocks(n - k1), block_dim=CH_TPB
            )
            var tiles = (n - k1 + CH_RB_TILE - 1) // CH_RB_TILE
            ctx.enqueue_function[chol_trail_rb_kernel](
                a, Int32(k0), Int32(k1), Int32(n), grid_dim=(tiles, tiles, 1), block_dim=(CH_TILE_TPB, 1, 1),
            )
        k0 = k1



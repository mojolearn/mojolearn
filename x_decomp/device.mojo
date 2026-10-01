# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DevExec: the decomp lane's cells on the GPU. One thread per output, the
cell of `x_decomp/cells.mojo` verbatim; the serial routines run on ONE
device thread. Host in, host dst: upload, launch, download."""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from std.time import perf_counter_ns
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.ffi import _Global
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.memory import memcpy
from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count

from checks.vendor import COMPILED_VENDOR
from core.householder_qr import qr_factor, qr_slice_count
from decomposition.impl.linalg.detail.svd_full import svd_of_r
from decomposition.linalg_public_device import device_eigh, device_qr_r
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div, identical_mul, identical_mul_add
from x_decomp.cells import (
    lu_solve_col,
    div0,
    sqrt0,
    sub,
    F32Ptr,
    absmax_fold_cell,
    absmax_part_cell,
    absmax_sign_cell,
    FOLD_BLOCK,
    colsum_part_cell,
    rowsum_part_cell,
    fold_cell,
    gemm_part_cell,
    X_DECOMP_SVD_SWEEPS,
    X_DECOMP_SVD_TOL,
    I32Ptr,
    bidx,
    cd_row,
    chol_serial,
    chol_diag,
    chol_col_elem,
    colsum_cell,
    ew_cell,
    gemm_cell,
    als_row,
    als_row_solve,
    add,
    mul,
    als_cg_row,
    geqrf_dot,
    geqrf_head,
    geqrf_scale_elem,
    geqrf_serial,
    geqrf_update_elem,
    orgqr_col,
    orgqr_dot,
    orgqr_init_elem,
    orgqr_update_elem,
    barycenter_row,
    dijkstra_arc_count,
    dijkstra_arcs,
    dijkstra_row,
    gamma_cell,
    lasso_row,
    lda_doc_row,
    lu_diag,
    lu_l_elem,
    lu_pivot,
    lu_serial,
    lu_swap_elem,
    lu_update_elem,
    omp_row,
    lu_solve_serial,
    orth_rank_guard,
    trsm_row,
    rand_cell,
    rowsum_cell,
    pdist_cell,
    sqdist_cell,
)
from x_decomp.qr_host import geqrf_host_rows, orgqr_host_rows, xd_qr_on_host
from x_decomp.exec_trait import Exec
from x_decomp.host import HostExec
from x_decomp.jacobi2 import (
    J2_TPB,
    jacobi_eigh2_kernel,
    one_sided_svd2_chunk_kernel,
    one_sided_svd2_finish_kernel,
)
from x_decomp.qr_bounded import QRB_CELLS, qr_factor_bounded
from x_decomp.jacobi_par import (
    PJ_TPB,
    eigh_par_cs_kernel,
    eigh_par_off_kernel,
    eigh_par_update_kernel,
    pj_identity_kernel,
    pj_transpose_kernel,
    svd_par_norm_kernel,
    svd_par_round_kernel,
)
from std.os import getenv
from core.device_zero import enqueue_fill
from decomposition.checks.jacobi_eigh_device import JACOBI_INFO_UNWRITTEN, JACOBI_SWEEPS, JACOBI_TOL
from decomposition.host.linalg_public import eigh_ascending
from decomposition.impl.linalg.detail.pca import SIGNFLIP_TPB, sign_flip_kernel


def lu_serial_max() -> Int:
    """Largest n whose LU runs as ONE `lu_kernel` launch (MOJOLEARN_XD_LU_SERIAL,
    default 16; timing only, the cells and their order are the same)."""
    var v = String(getenv("MOJOLEARN_XD_LU_SERIAL", "16"))
    try:
        return Int(v)
    except:
        return 16


#: Sweep budgets of the round-robin solvers (FAST on Metal). A solve that
#: does not converge inside its budget is handed to the cyclic solver.
comptime PJ_EIGH_SWEEPS = 30
comptime PJ_SVD_SWEEPS = X_DECOMP_SVD_SWEEPS
#: rounds enqueued between two synchronize() calls
comptime PJ_SYNC_ROUNDS = 512


def pj_eigh_min() -> Int:
    """Smallest n whose eigh takes the round-robin solver of
    x_decomp/jacobi_par.mojo (FAST builds for Metal only; 0 = never).
    MOJOLEARN_XD_PJ_EIGH_MIN overrides it."""
    var v = String(getenv("MOJOLEARN_XD_PJ_EIGH_MIN", "0"))
    try:
        return Int(v)
    except:
        return 0


def host_eigh_max() -> Int:
    """Largest n whose eigh runs on the host executor inside the GPU binding
    (FAST builds for Metal only; 0 = never): the same cyclic Jacobi, without
    the upload, launch, readback and sync a device solve of a few hundred
    values is made of (`Kit[E, S]`'s rule for the native drivers).
    MOJOLEARN_XD_HOST_EIGH_MAX overrides it."""
    var v = String(getenv("MOJOLEARN_XD_HOST_EIGH_MAX", "0"))
    try:
        return Int(v)
    except:
        return 0


def pj_svd_min() -> Int:
    """`pj_eigh_min` for the one-sided SVD (MOJOLEARN_XD_PJ_SVD_MIN)."""
    var v = String(getenv("MOJOLEARN_XD_PJ_SVD_MIN", "0"))
    try:
        return Int(v)
    except:
        return 0


def jacobi2_eigh_on() -> Bool:
    """Whether the kit's eigh runs `jacobi_eigh2_kernel` (x_decomp/jacobi2.mojo)
    in place of `device_eigh`. MOJOLEARN_XD_JACOBI_EIGH=1 / =2 names the
    kernel outright (the eigh only; timing A/B).

    Metal, IDENTICAL: `device_eigh`, main's choice after the 2026-09-28 M4
    consolidated check crashed MTLCompilerService in five eigh callers
    (METAL SIGABRT, "cannot select: 113 7, 1" in agc.main). That build held
    the FENCED jacobi2 (5c144678d: an atomic fence, then barrier()); the
    kernel in this tree orders device memory with `llvm.air.wg.barrier(3, 1)`
    (bd6af0c4b) and builds and runs on M4 (m4-a 1790626766529, every digest
    equal to `device_eigh`'s). MOJOLEARN_XD_JACOBI=2 opts in; the default
    stays until the consolidated check qualifies it.

    Metal, FAST: jacobi2 (lane/decomp-apple3, m4-a 1790626766529: eigh 800
    8.88 -> 5.36 s, Isomap 1000 rows 20.2 -> 10.5 s, ClassicalMDS 11.1 ->
    5.66 s, the same output bytes as `device_eigh`).

    CUDA/HIP keep their default. The SVD default is `jacobi2_on`'s."""
    var e = String(getenv("MOJOLEARN_XD_JACOBI_EIGH", "0"))
    if e == "1":
        return False
    if e == "2":
        return True
    comptime if COMPILED_VENDOR == "metal":
        comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
            return jacobi2_on()
        return String(getenv("MOJOLEARN_XD_JACOBI", "1")) != "1"
    else:
        return jacobi2_on()


def jacobi2_on() -> Bool:
    """MOJOLEARN_XD_JACOBI=1 selects the shipped-before Jacobi kernels
    (timing A/B only: `x_decomp/jacobi2.mojo` stores the same bits)."""
    return String(getenv("MOJOLEARN_XD_JACOBI", "2")) != "1"


struct _XdContext(Defaultable, Movable):
    """ONE process-lifetime DeviceContext for every x_decomp entry (the x_cnn
    `_Global` pattern, CURRENT DIRECTIVES 2026-09-27): a context per call
    exhausts Metal's per-process command queues within one fit, and a
    fit here is hundreds of calls. Every entry still frees and drains its
    own buffers before returning; the context outlives them all."""
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()


comptime X_DECOMP_CONTEXT = _Global[StorageType=_XdContext, name="MojoXDecompContextIdentical", init_fn=_XdContext.__init__]


def xd_ctx() raises -> DeviceContext:
    """The shared context, created on first use."""
    var slot = X_DECOMP_CONTEXT.get_or_create_ptr()
    if not slot[].ctx:
        slot[].ctx = DeviceContext()
    return slot[].ctx.value().copy()

comptime TPB = 128
#: rows of shortest paths per launch (the heap scratch is DIJKSTRA_ROWS x n ints)
comptime DIJKSTRA_ROWS = 4096


def gemm_kernel(a: F32Ptr, b: F32Ptr, c: F32Ptr, m: Int32, k: Int32, n: Int32, ta: Int32, tb: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(m) * Int(n):
        var i = t // Int(n)
        var j = t % Int(n)
        c.unsafe_store(t, gemm_cell(a, b, i, j, Int(m), Int(k), Int(n), ta != 0, tb != 0))


def gemm_part_kernel(
    a: F32Ptr, b: F32Ptr, p: F32Ptr, m: Int32, k: Int32, n: Int32, ta: Int32, tb: Int32, nb: Int32
):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var mn = Int(m) * Int(n)
    if t < mn * Int(nb):
        var bl = t // mn
        var c = t % mn
        p.unsafe_store(t, gemm_part_cell(
            a, b, c // Int(n), c % Int(n), Int(m), Int(k), Int(n), ta != 0, tb != 0,
            bl * FOLD_BLOCK, min(Int(k), (bl + 1) * FOLD_BLOCK)))


def fold_kernel(p: F32Ptr, dst: F32Ptr, count: Int32, nb: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(count):
        dst.unsafe_store(t, fold_cell(p, t, Int(nb), Int(count)))


def colsum_part_kernel(a: F32Ptr, p: F32Ptr, n: Int32, d: Int32, nb: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(d) * Int(nb):
        var bl = t // Int(d)
        var j = t % Int(d)
        p.unsafe_store(t, colsum_part_cell(a, j, Int(n), Int(d), bl * FOLD_BLOCK, min(Int(n), (bl + 1) * FOLD_BLOCK)))


def rowsum_part_kernel(a: F32Ptr, p: F32Ptr, n: Int32, d: Int32, nb: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(n) * Int(nb):
        var bl = t // Int(n)
        var i = t % Int(n)
        p.unsafe_store(t, rowsum_part_cell(a, i, Int(d), bl * FOLD_BLOCK, min(Int(d), (bl + 1) * FOLD_BLOCK)))


def absmax_part_kernel(a: F32Ptr, p: F32Ptr, n: Int32, d: Int32, by_col: Int32, nb: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var cnt = Int(d) if by_col != 0 else Int(n)
    var length = Int(n) if by_col != 0 else Int(d)
    if t < cnt * Int(nb):
        var bl = t // cnt
        var v = t % cnt
        var r = absmax_part_cell(
            a, v, Int(n), Int(d), by_col != 0, bl * FOLD_BLOCK, min(length, (bl + 1) * FOLD_BLOCK)
        )
        p.unsafe_store(2 * t, r[0])
        p.unsafe_store(2 * t + 1, r[1])


def absmax_fold_kernel(p: F32Ptr, dst: F32Ptr, cnt: Int32, nb: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(cnt):
        dst.unsafe_store(t, absmax_fold_cell(p, t, Int(nb), Int(cnt)))


def absmax_kernel(a: F32Ptr, dst: F32Ptr, n: Int32, d: Int32, by_col: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var cnt = Int(d) if by_col != 0 else Int(n)
    if t < cnt:
        dst.unsafe_store(t, absmax_sign_cell(a, t, Int(n), Int(d), by_col != 0))


def ew_kernel(
    op: Int32, a: F32Ptr, b: F32Ptr, bm: Int32, c: F32Ptr, cm: Int32, dst: F32Ptr,
    count: Int32, d: Int32, s: Float32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(count):
        dst.unsafe_store(
            i,
            ew_cell(
                Int(op), a.unsafe_load(i), b.unsafe_load(bidx(Int(bm), i, Int(d))),
                c.unsafe_load(bidx(Int(cm), i, Int(d))), s,
            ),
        )


def colsum_kernel(a: F32Ptr, dst: F32Ptr, n: Int32, d: Int32):
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j < Int(d):
        dst.unsafe_store(j, colsum_cell(a, j, Int(n), Int(d)))


def rowsum_kernel(a: F32Ptr, dst: F32Ptr, n: Int32, d: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        dst.unsafe_store(i, rowsum_cell(a, i, Int(d)))


def sqdist_kernel(a: F32Ptr, b: F32Ptr, dst: F32Ptr, na: Int32, nb: Int32, d: Int32, kind: Int32, pw: Float32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(na) * Int(nb):
        if kind == 0:
            dst.unsafe_store(t, sqdist_cell(a, b, t // Int(nb), t % Int(nb), Int(d)))
        else:
            dst.unsafe_store(t, pdist_cell(a, b, t // Int(nb), t % Int(nb), Int(d), Int(kind), pw))


def rand_kernel(dst: F32Ptr, count: Int32, seed: UInt32, stream: UInt32, kind: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(count):
        dst.unsafe_store(i, rand_cell(i, seed, stream, Int(kind)))


def lu_kernel(a: F32Ptr, piv: I32Ptr, info: F32Ptr, n: Int32):
    if block_idx.x == 0 and thread_idx.x == 0:
        lu_serial(a, piv, Int(n), info)


def lu_info_init_kernel(info: F32Ptr):
    if block_idx.x == 0 and thread_idx.x == 0:
        info.unsafe_store(0, Float32(0))


def lu_pivot_kernel(a: F32Ptr, piv: I32Ptr, k: Int32, n: Int32):
    if block_idx.x == 0 and thread_idx.x == 0:
        lu_pivot(a, piv, Int(k), Int(n))


comptime LU_PIVOT_TPB = 256


def lu_pivot_block_kernel(a: F32Ptr, piv: I32Ptr, k: Int32, n: Int32):
    """`lu_pivot` on ONE BLOCK of LU_PIVOT_TPB threads (lane/neural-net-
    experiment, 2026-09-30, the classical pass): the largest |a[i, k]| for
    i >= k, ties to the LOWEST row. Comparisons only, no arithmetic, so the
    result is the serial scan's by construction: every thread starts from
    (|a[k, k]|, k) and takes a later row only on a STRICT greater value,
    exactly as the serial scan does, and the tree combine prefers the
    greater value and, on equal values, the lower row. A NaN never wins a
    strict compare, so it is skipped as the serial scan skips it; a NaN at
    row k makes every compare false and keeps row k, as the serial scan
    does. The serial scan was one thread walking a column of n strided
    loads per step: at n = 8192 on an MI325X that was most of a 600 s
    factorization (bench_board 0.8.25, `lu-factor`)."""
    var kk = Int(k)
    var nn = Int(n)
    var rv = stack_allocation[
        LU_PIVOT_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var ri = stack_allocation[
        LU_PIVOT_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    var best = abs(ftz(a.unsafe_load(kk * nn + kk)))
    var p = kk
    var i = kk + 1 + tid
    while i < nn:
        var v = abs(ftz(a.unsafe_load(i * nn + kk)))
        if v > best:
            best = v
            p = i
        i += LU_PIVOT_TPB
    rv.unsafe_store(tid, best)
    ri.unsafe_store(tid, Int32(p))
    barrier()
    var active = LU_PIVOT_TPB // 2
    while active > 0:
        if tid < active:
            var ov = rv.unsafe_load(tid + active)
            var oi = ri.unsafe_load(tid + active)
            var cv = rv.unsafe_load(tid)
            var ci = ri.unsafe_load(tid)
            if ov > cv or (ov == cv and oi < ci):
                rv.unsafe_store(tid, ov)
                ri.unsafe_store(tid, oi)
        barrier()
        active = active // 2
    if tid == 0:
        piv.unsafe_store(kk, ri.unsafe_load(0))


def lu_pivot_parallel() -> Bool:
    """`MOJOLEARN_XD_LU_PIVOT_SERIAL=1` restores the one-thread pivot scan
    (the A/B arm); default the block kernel."""
    return String(getenv("MOJOLEARN_XD_LU_PIVOT_SERIAL")) != "1"


def lu_solve_cols_kernel(lu: F32Ptr, piv: I32Ptr, b: F32Ptr, n: Int32, nrhs: Int32, trans: Int32):
    """One thread per right-hand side column (`lu_solve_col`): the columns
    are independent in every statement of the serial solve, so the bits are
    the serial solve's. The serial solve was one thread for n^2 * nrhs
    dependent multiply-adds: 616 s at 8192 x 64 on an MI325X."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c < Int(nrhs):
        lu_solve_col(lu, piv, b, Int(n), Int(nrhs), Int(trans), c)


def lu_swap_kernel(a: F32Ptr, piv: I32Ptr, k: Int32, n: Int32):
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j < Int(n):
        lu_swap_elem(a, piv, Int(k), j, Int(n))


def lu_diag_kernel(a: F32Ptr, info: F32Ptr, scal: F32Ptr, k: Int32, n: Int32):
    if block_idx.x == 0 and thread_idx.x == 0:
        lu_diag(a, info, scal, Int(k), Int(n))


def lu_l_kernel(a: F32Ptr, scal: F32Ptr, k: Int32, n: Int32):
    var i = Int(k) + 1 + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        lu_l_elem(a, scal, Int(k), i, Int(n))


def lu_update_kernel(a: F32Ptr, scal: F32Ptr, k: Int32, n: Int32):
    var w = Int(n) - Int(k) - 1
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if w > 0 and t < w * w:
        lu_update_elem(a, scal, Int(k), Int(k) + 1 + t // w, Int(k) + 1 + t % w, Int(n))


# ---- the blocked LU (lane neural-pass32, 2026-10-01) ------------------------------------
#: The panel width (MOJOLEARN_XD_LU_PANEL, default LU_PANEL_NB; 0 keeps the
#: per-step route) and the trailing tile: LU_TILE x LU_TILE cells a block,
#: one thread per cell, the L slab (LU_TILE rows x NB) and the U slab (NB x
#: LU_TILE columns) in threadgroup memory.
comptime LU_PANEL_NB = 32
comptime LU_TILE = 16
comptime LU_TILE_TPB = LU_TILE * LU_TILE


def lu_panel_width() -> Int:
    """The blocked LU's panel width: MOJOLEARN_XD_LU_PANEL (0 = the
    per-step route, the A/B arm), default LU_PANEL_NB."""
    var v = String(getenv("MOJOLEARN_XD_LU_PANEL", String(LU_PANEL_NB)))
    try:
        return max(0, Int(v))
    except:
        return LU_PANEL_NB


def lu_swap_cols_kernel(a: F32Ptr, piv: I32Ptr, k: Int32, n: Int32, col_lo: Int32, col_hi: Int32):
    """`lu_swap_elem` over the columns [col_lo, col_hi) of rows k and
    piv[k]: the panel's steps swap the panel's and the left columns at once,
    the trailing columns later, in the same order (`lu_apply_swaps_kernel`)."""
    var j = Int(col_lo) + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j < Int(col_hi):
        lu_swap_elem(a, piv, Int(k), j, Int(n))


def lu_update_panel_kernel(a: F32Ptr, scal: F32Ptr, k: Int32, n: Int32, col_hi: Int32):
    """`lu_update_elem` for the cells (i, j), k < i < n, k < j < col_hi: step
    k's update restricted to the panel's columns."""
    var kk = Int(k)
    var w = Int(col_hi) - kk - 1
    var h = Int(n) - kk - 1
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if w > 0 and h > 0 and t < h * w:
        lu_update_elem(a, scal, kk, kk + 1 + t // w, kk + 1 + t % w, Int(n))


def lu_act_kernel(scal: F32Ptr, act: F32Ptr, k: Int32):
    """act[k] = scal[1]: whether step k eliminated (a zero pivot skips its
    step, and the trailing kernels skip it the same way)."""
    if block_idx.x == 0 and thread_idx.x == 0:
        act.unsafe_store(Int(k), scal.unsafe_load(1))


def lu_apply_swaps_kernel(a: F32Ptr, piv: I32Ptr, k0: Int32, k1: Int32, n: Int32):
    """One thread per trailing column j >= k1: the panel's swaps k0 .. k1 - 1
    applied to that column in order (`lu_swap_elem`, which the serial loop
    applied to every column at each step; a column's swaps are its own)."""
    var j = Int(k1) + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j < Int(n):
        for k in range(Int(k0), Int(k1)):
            lu_swap_elem(a, piv, k, j, Int(n))


def lu_trsm_kernel(a: F32Ptr, act: F32Ptr, k0: Int32, k1: Int32, n: Int32):
    """One thread per trailing column j >= k1: the panel's U rows brought up
    to date. Row k (k0 < k < k1) receives the panel's earlier steps k' = k0
    .. k - 1 in order, each `lu_update_elem`'s statement with l = a[k, k']
    (the panel's multiplier) and a[k', j] already final (row k' done
    first), a step with act[k'] = 0 skipped: the serial loop's chain for
    that cell."""
    var nn = Int(n)
    var j = Int(k1) + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j < nn:
        for k in range(Int(k0) + 1, Int(k1)):
            var acc = ftz(a.unsafe_load(k * nn + j))
            for kp in range(Int(k0), k):
                if act.unsafe_load(kp) != Float32(0):
                    var l = a.unsafe_load(k * nn + kp)
                    acc = ftz(identical_mul_add(-l, ftz(a.unsafe_load(kp * nn + j)), ftz(acc)))
            a.unsafe_store(k * nn + j, acc)


def lu_trail_tiled_kernel(a: F32Ptr, act: F32Ptr, k0: Int32, k1: Int32, n: Int32, nb: Int32):
    """The trailing cells (i, j), i >= k1, j >= k1: the panel's steps k' =
    k0 .. k1 - 1 applied in order, each `lu_update_elem`'s statement with
    l = a[i, k'] and u = a[k', j] (both final), a step with act[k'] = 0
    skipped: the serial loop's chain for that cell, the operands staged
    through threadgroup memory a tile at a time instead of read from device
    memory once per step."""
    var nn = Int(n)
    var kk0 = Int(k0)
    var kk1 = Int(k1)
    var width = Int(nb)
    var tid = Int(thread_idx.x)
    var r = tid // LU_TILE
    var c = tid - r * LU_TILE
    var i0 = kk1 + Int(block_idx.y) * LU_TILE
    var j0 = kk1 + Int(block_idx.x) * LU_TILE
    var ls = stack_allocation[LU_TILE * LU_PANEL_NB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var us = stack_allocation[LU_PANEL_NB * LU_TILE, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var acts = stack_allocation[LU_PANEL_NB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    # the L slab: LU_TILE rows x width (unflushed, as the item reads l)
    var q = tid
    while q < LU_TILE * width:
        var rr = q // width
        var cc = q - rr * width
        var v = Float32(0)
        if i0 + rr < nn:
            v = a.unsafe_load((i0 + rr) * nn + kk0 + cc)
        ls[q] = v
        q += LU_TILE_TPB
    # the U slab: width rows x LU_TILE columns (flushed, as the item reads u)
    q = tid
    while q < width * LU_TILE:
        var rr = q // LU_TILE
        var cc = q - rr * LU_TILE
        var v = Float32(0)
        if j0 + cc < nn:
            v = ftz(a.unsafe_load((kk0 + rr) * nn + j0 + cc))
        us[q] = v
        q += LU_TILE_TPB
    if tid < width:
        acts[tid] = act.unsafe_load(kk0 + tid)
    barrier()
    var i = i0 + r
    var j = j0 + c
    if i < nn and j < nn:
        var acc = ftz(a.unsafe_load(i * nn + j))
        for kp in range(width):
            if acts[kp] != Float32(0):
                acc = ftz(identical_mul_add(-ls[r * width + kp], us[kp * LU_TILE + c], ftz(acc)))
        a.unsafe_store(i * nn + j, acc)


def lu_solve_kernel(lu: F32Ptr, piv: I32Ptr, b: F32Ptr, n: Int32, nrhs: Int32, trans: Int32):
    if block_idx.x == 0 and thread_idx.x == 0:
        lu_solve_serial(lu, piv, b, Int(n), Int(nrhs), Int(trans))


def chol_kernel(a: F32Ptr, info: F32Ptr, n: Int32):
    if block_idx.x == 0 and thread_idx.x == 0:
        chol_serial(a, Int(n), info)


def chol_diag_kernel(a: F32Ptr, info: F32Ptr, j: Int32, n: Int32):
    if block_idx.x == 0 and thread_idx.x == 0:
        chol_diag(a, info, Int(j), Int(n))


def chol_col_kernel(a: F32Ptr, j: Int32, n: Int32):
    var jj = Int(j)
    var i = jj + 1 + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        chol_col_elem(a, jj, i, Int(n))


def chol_serial_max() -> Int:
    """Largest n whose Cholesky runs as ONE `chol_kernel` launch on one
    thread (MOJOLEARN_XD_CHOL_SERIAL, default 16; timing only: the column
    driver below stores the same cells in the same order). A value at or
    above every n restores the one-thread kernel for the A/B."""
    var v = String(getenv("MOJOLEARN_XD_CHOL_SERIAL", "16"))
    try:
        return Int(v)
    except:
        return 16


def cd_rows_kernel(w: F32Ptr, hht: F32Ptr, xht: F32Ptr, perm: I32Ptr, viol: F32Ptr, n: Int32, k: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        viol.unsafe_store(i, cd_row(w, hht, xht, perm, i, Int(k)))


def orth_guard_kernel(r: F32Ptr, l: Int32):
    if block_idx.x == 0 and thread_idx.x == 0:
        orth_rank_guard(r, Int(l))


def trsm_kernel(a: F32Ptr, r: F32Ptr, q: F32Ptr, m: Int32, l: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(m):
        trsm_row(a, r, q, i, Int(l))


def lasso_rows_kernel(
    g: F32Ptr, q: F32Ptr, w: F32Ptr, h: F32Ptr, its: F32Ptr, n: Int32, k: Int32, alpha: Float32,
    max_iter: Int32, tol: Float32, positive: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        its.unsafe_store(i, lasso_row(g, q, w, h, i, Int(k), alpha, Int(max_iter), tol, positive != 0))


def omp_rows_kernel(g: F32Ptr, q: F32Ptr, w: F32Ptr, s: F32Ptr, na: F32Ptr, n: Int32, k: Int32, nnz: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        na.unsafe_store(i, omp_row(g, q, w, s, i, Int(k), Int(nnz)))


def gamma_kernel(dst: F32Ptr, count: Int32, seed: UInt32, stream: UInt32, shape: Float32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(count):
        dst.unsafe_store(i, gamma_cell(i, seed, stream, shape))


def lda_rows_kernel(
    x: F32Ptr, ew: F32Ptr, d: F32Ptr, e: F32Ptr, s: F32Ptr, its: F32Ptr, n: Int32, k: Int32, v: Int32,
    prior: Float32, max_iter: Int32, tol: Float32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        its.unsafe_store(i, lda_doc_row(x, ew, d, e, s, i, Int(k), Int(v), prior, Int(max_iter), tol))


def dijkstra_kernel(
    rp: I32Ptr, adj: I32Ptr, wa: F32Ptr, wb: F32Ptr, dist: F32Ptr, heap: I32Ptr, pos: I32Ptr, reached: F32Ptr,
    n: Int32, row0: Int32, rows: Int32,
):
    """Rows row0 .. row0 + rows - 1; pos and dist are the full n x n, heap
    is `rows` x n (slot t = i - row0)."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(rows):
        var i = Int(row0) + t
        reached.unsafe_store(i, dijkstra_row(rp, adj, wa, wb, dist, heap, pos, i, Int(n), t * Int(n)))


def barycenter_kernel(
    x: F32Ptr, y: F32Ptr, nbr: F32Ptr, wt: F32Ptr, s: F32Ptr, flags: F32Ptr, n: Int32, d: Int32, k: Int32, reg: Float32
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        flags.unsafe_store(i, barycenter_row(x, y, nbr, wt, s, i, Int(d), Int(k), reg))


def als_kernel(
    c: F32Ptr, y: F32Ptr, yty: F32Ptr, x: F32Ptr, s: F32Ptr, flags: F32Ptr, n: Int32, m: Int32, f: Int32, reg: Float32
):
    var u = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if u < Int(n):
        flags.unsafe_store(u, als_row(c, y, yty, x, s, u, Int(m), Int(f), reg))


comptime ALS_TEAM_TPB = 256
#: Largest f*f + f the team kernel keeps in threadgroup memory (f <= 63).
comptime ALS_TEAM_CELLS = 4096


def als_team_kernel(
    c: F32Ptr, y: F32Ptr, yty: F32Ptr, x: F32Ptr, s: F32Ptr, flags: F32Ptr, n: Int32, m: Int32, f: Int32, reg: Float32
):
    """`als_row` for user `block_idx.x`, by a block (lane/decomp-apple2):
    thread t owns cells t, t + 256, ... of the user's f*f + f accumulators
    (A row-major, then b) and runs, for each of them, exactly `als_row`'s
    sequence (the YtY value, + reg on the diagonal, then one fused
    multiply-add per item with c_ui != 0, items ascending), in threadgroup
    memory. Thread 0 then writes them to the row's scratch and runs
    `als_row_solve`, `als_row`'s own tail. No device word crosses threads."""
    var u = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var ff = Int(f)
    var mm = Int(m)
    var cells = ff * ff + ff
    var sh = stack_allocation[ALS_TEAM_CELLS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var cc = tid
    while cc < cells:
        if cc < ff * ff:
            var v = yty.unsafe_load(cc)
            if cc // ff == cc % ff:
                v = add(v, reg)
            sh[cc] = v
        else:
            sh[cc] = Float32(0)
        cc += ALS_TEAM_TPB
    for i in range(mm):
        var conf = ftz(c.unsafe_load(u * mm + i))
        if conf == Float32(0):
            continue
        var pos = conf > Float32(0)
        var ca = conf if pos else -conf
        var cm1 = sub(ca, Float32(1))
        cc = tid
        while cc < cells:
            if cc < ff * ff:
                var j = cc // ff
                var l = cc - j * ff
                var t = mul(cm1, y.unsafe_load(i * ff + j))
                sh[cc] = ftz(identical_mul_add(t, ftz(y.unsafe_load(i * ff + l)), ftz(sh[cc])))
            elif pos:
                var j = cc - ff * ff
                sh[cc] = ftz(identical_mul_add(conf, ftz(y.unsafe_load(i * ff + j)), ftz(sh[cc])))
            cc += ALS_TEAM_TPB
    barrier()
    if tid == 0:
        var ab = u * cells
        for q in range(cells):
            s.unsafe_store(ab + q, sh[q])
        flags.unsafe_store(u, als_row_solve(x, s, u, ff))


def als_cg_kernel(
    c: F32Ptr, y: F32Ptr, yty: F32Ptr, x: F32Ptr, s: F32Ptr, steps: F32Ptr, n: Int32, m: Int32, f: Int32, reg: Float32,
    cg: Int32,
):
    var u = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if u < Int(n):
        steps.unsafe_store(u, als_cg_row(c, y, yty, x, s, u, Int(m), Int(f), reg, Int(cg)))


def geqrf_kernel(a: F32Ptr, tau: F32Ptr, m: Int32, n: Int32):
    if block_idx.x == 0 and thread_idx.x == 0:
        geqrf_serial(a, tau, Int(m), Int(n))


def geqrf_head_kernel(a: F32Ptr, tau: F32Ptr, scal: F32Ptr, k: Int32, m: Int32, n: Int32):
    if block_idx.x == 0 and thread_idx.x == 0:
        geqrf_head(a, tau, scal, Int(k), Int(m), Int(n))


def geqrf_scale_kernel(a: F32Ptr, scal: F32Ptr, k: Int32, m: Int32, n: Int32):
    var i = Int(k) + 1 + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(m):
        geqrf_scale_elem(a, scal, Int(k), i, Int(n))


def geqrf_dot_kernel(a: F32Ptr, scal: F32Ptr, w: F32Ptr, k: Int32, m: Int32, n: Int32):
    var j = Int(k) + 1 + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j < Int(n):
        w.unsafe_store(j, geqrf_dot(a, scal, Int(k), j, Int(m), Int(n)))


def geqrf_update_kernel(a: F32Ptr, tau: F32Ptr, scal: F32Ptr, w: F32Ptr, k: Int32, m: Int32, n: Int32):
    var cols = Int(n) - Int(k) - 1
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cols > 0 and t < (Int(m) - Int(k)) * cols:
        var i = Int(k) + t // cols
        var j = Int(k) + 1 + t % cols
        geqrf_update_elem(a, tau, scal, Int(k), i, j, Int(n), w.unsafe_load(j))


def orgqr_init_kernel(q: F32Ptr, m: Int32, qc: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(m) * Int(qc):
        orgqr_init_elem(q, t // Int(qc), t % Int(qc), Int(qc))


def orgqr_dot_kernel(h: F32Ptr, tau: F32Ptr, q: F32Ptr, w: F32Ptr, k: Int32, m: Int32, n: Int32, qc: Int32):
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j < Int(qc):
        w.unsafe_store(j, orgqr_dot(h, tau, q, Int(k), j, Int(m), Int(n), Int(qc)))


def orgqr_update_kernel(h: F32Ptr, tau: F32Ptr, q: F32Ptr, w: F32Ptr, k: Int32, m: Int32, n: Int32, qc: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < (Int(m) - Int(k)) * Int(qc):
        var i = Int(k) + t // Int(qc)
        var j = t % Int(qc)
        orgqr_update_elem(h, tau, q, Int(k), i, j, Int(n), Int(qc), w.unsafe_load(j))


def orgqr_kernel(h: F32Ptr, tau: F32Ptr, q: F32Ptr, m: Int32, n: Int32, kk: Int32, qc: Int32):
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j < Int(qc):
        orgqr_col(h, tau, q, j, Int(m), Int(n), Int(kk), Int(qc))


# ---- STAGED SERIAL FOLDS (lane decomp-apple, 2026-09-28). geqrf and orgqr
# fold a column of an m x n row-major matrix in ONE thread, rows ascending
# (their bits are that order). A lone GPU thread walking a strided column
# pays a cache line per element: 7 s for linalg.qr at 200k x 28 on the M4
# Pro. Here the whole block LOADS the column in chunks of STAGE rows into
# threadgroup memory, and thread 0 folds each chunk from there: the same
# cells' arithmetic on the same values in the same order, only the loads are
# shared. Block STAGE_TPB threads; 2 x STAGE floats = 16 KB of threadgroup
# memory.
comptime STAGE = 2048
comptime STAGE_TPB = 256


@always_inline
def _chain_fma(
    sh: UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED], lo: Int, cnt: Int, off: Int, acc0: Float32
) -> Float32:
    """acc = ftz(fma(x_u, y_u, acc)) for u in lo .. lo + cnt - 1, x from `sh[u]`
    and y from `sh[off + u]` (both already flushed by the staging threads),
    the one serial chain of `geqrf_dot` / `orgqr_dot` (lane neural-pass18):
    the same operations in the same order, written eight steps at a time so
    the sixteen shared loads of a group issue before the eight dependent
    multiply-adds (the chain's latency is the fma's, not the load's); the
    tail runs one step at a time."""
    var acc = acc0
    var u = lo
    var end = lo + cnt
    var body = end - (cnt % 8)
    while u < body:
        var x0 = sh[u]
        var x1 = sh[u + 1]
        var x2 = sh[u + 2]
        var x3 = sh[u + 3]
        var x4 = sh[u + 4]
        var x5 = sh[u + 5]
        var x6 = sh[u + 6]
        var x7 = sh[u + 7]
        var y0 = sh[off + u]
        var y1 = sh[off + u + 1]
        var y2 = sh[off + u + 2]
        var y3 = sh[off + u + 3]
        var y4 = sh[off + u + 4]
        var y5 = sh[off + u + 5]
        var y6 = sh[off + u + 6]
        var y7 = sh[off + u + 7]
        acc = ftz(identical_mul_add(x0, y0, acc))
        acc = ftz(identical_mul_add(x1, y1, acc))
        acc = ftz(identical_mul_add(x2, y2, acc))
        acc = ftz(identical_mul_add(x3, y3, acc))
        acc = ftz(identical_mul_add(x4, y4, acc))
        acc = ftz(identical_mul_add(x5, y5, acc))
        acc = ftz(identical_mul_add(x6, y6, acc))
        acc = ftz(identical_mul_add(x7, y7, acc))
        u += 8
    while u < end:
        acc = ftz(identical_mul_add(sh[u], sh[off + u], acc))
        u += 1
    return acc


def geqrf_head_staged_kernel(a: F32Ptr, tau: F32Ptr, scal: F32Ptr, k: Int32, m: Int32, n: Int32):
    """`geqrf_head` (with its `reflector_norm`), the folds staged. Lane
    neural-pass18: the xmax scan is a maximum, exact in any order, so every
    thread scans its own rows (seeded 0, `if v > local`, so a NaN never
    enters, as in the serial scan) and the block folds the locals with the
    same comparison; the norm chain's per-row division (independent of the
    accumulator) is done by the staging threads into shared memory, and the
    chain runs through `_chain_fma`."""
    var sh = stack_allocation[STAGE, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var red = stack_allocation[STAGE_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var bc = stack_allocation[2, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var K = Int(k)
    var M = Int(m)
    var N = Int(n)
    var alpha = ftz(a.unsafe_load(K * N + K))
    # pass 1: xmax over rows k+1.. (geqrf_head's scan): the maximum of the
    # |values|, which every order gives alike; each thread scans a strided
    # share of the rows, the block folds the shares
    var local = Float32(0)
    var i = K + 1 + tid
    while i < M:
        var v = abs(ftz(a.unsafe_load(i * N + K)))
        if v > local:
            local = v
        i += STAGE_TPB
    red[tid] = local
    barrier()
    var half = STAGE_TPB // 2
    while half > 0:
        if tid < half:
            var other = red[tid + half]
            if other > red[tid]:
                red[tid] = other
        barrier()
        half //= 2
    var xmax = red[0]
    barrier()
    if xmax == Float32(0):
        if tid == 0:
            tau.unsafe_store(K, Float32(0))
            scal.unsafe_store(0, Float32(1))
            scal.unsafe_store(1, Float32(0))
        return
    # reflector_norm(a, k, k, m, n): its scan of rows k.. is |alpha| then
    # the rows above in order, the same maximum
    var mx = Float32(0)
    if abs(alpha) > mx:
        mx = abs(alpha)
    if xmax > mx:
        mx = xmax
    var acc = Float32(0)
    var c0 = K
    while c0 < M:
        var cnt = min(STAGE, M - c0)
        var t = tid
        while t < cnt:
            # v = ftz(x / mx), the chain's operand, flushed here
            sh[t] = ftz(identical_div(ftz(a.unsafe_load((c0 + t) * N + K)), mx))
            t += STAGE_TPB
        barrier()
        if tid == 0:
            acc = _chain_fma(sh, 0, cnt, 0, acc)
        barrier()
        c0 += STAGE
    if tid == 0:
        var nrm = ftz(identical_mul(sqrt0(acc), mx))
        var beta = -nrm if alpha >= Float32(0) else nrm
        tau.unsafe_store(K, div0(sub(beta, alpha), beta))
        scal.unsafe_store(0, sub(alpha, beta))
        scal.unsafe_store(1, Float32(1))
        a.unsafe_store(K * N + K, beta)


def geqrf_dot_staged_kernel(a: F32Ptr, scal: F32Ptr, w: F32Ptr, k: Int32, m: Int32, n: Int32):
    """`geqrf_dot` for column j = k + 1 + block, the fold staged."""
    var sh = stack_allocation[2 * STAGE, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var K = Int(k)
    var M = Int(m)
    var N = Int(n)
    var j = K + 1 + Int(block_idx.x)
    if j >= N:
        return
    if scal.unsafe_load(1) == Float32(0):
        if tid == 0:
            w.unsafe_store(j, Float32(0))
        return
    var acc = ftz(a.unsafe_load(K * N + j))
    var c0 = K + 1
    while c0 < M:
        var cnt = min(STAGE, M - c0)
        var t = tid
        while t < cnt:
            sh[t] = ftz(a.unsafe_load((c0 + t) * N + K))
            sh[STAGE + t] = ftz(a.unsafe_load((c0 + t) * N + j))
            t += STAGE_TPB
        barrier()
        if tid == 0:
            acc = _chain_fma(sh, 0, cnt, STAGE, acc)
        barrier()
        c0 += STAGE
    if tid == 0:
        w.unsafe_store(j, acc)


# ---- the dot tiles (lane neural-pass41, 2026-10-01) -----------------------------------------
#: geqrf_dot_staged_kernel / orgqr_dot_staged_kernel give each trailing
#: column its own block, whose threads stage that column's rows one word
#: each at a stride of n (a scattered load per row per column: at 200,000
#: x 220 the two kernels are 3.0 + 7.5 s of the device QR on an M4). Here a
#: block takes DOT_COLS consecutive columns: its threads stage a slab of
#: DOT_ROWS rows of those columns as contiguous row segments (one coalesced
#: load per row), plus column k's slab, into threadgroup memory, and
#: DOT_COLS chain threads each advance their column's chain over the slab.
#: Every chain is the cell's: `w = ftz(a[k, j])`, then for i ascending
#: `w = ftz(fma(ftz(a[i, k]), ftz(a[i, j]), w))`, the same operands in the
#: same order. MOJOLEARN_XD_QR_DOT_TILE=0 keeps the one-column kernels.
comptime DOT_COLS = 16
comptime DOT_ROWS = 256
comptime DOT_TPB = 256
comptime DOT_LD = DOT_COLS + 1
comptime DOT_BYTES = (DOT_ROWS * DOT_LD + DOT_ROWS) * 4


def xd_qr_dot_tile() -> Bool:
    return String(getenv("MOJOLEARN_XD_QR_DOT_TILE")) != "0"


@always_inline
def _dot_tile_chain(
    colk: UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    tile: UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    c: Int, cnt: Int, acc0: Float32,
) -> Float32:
    """acc = ftz(fma(colk[r], tile[r, c], acc)) for r ascending over the slab:
    the same chain, software-pipelined one group of eight rows ahead (the
    next group's sixteen operands are loaded before this group's eight
    dependent multiply-adds issue, so the shared loads' latency overlaps
    the chain instead of adding to every step)."""
    var acc = acc0
    var r = 0
    if cnt >= 8:
        var k0 = colk[0]
        var k1 = colk[1]
        var k2 = colk[2]
        var k3 = colk[3]
        var k4 = colk[4]
        var k5 = colk[5]
        var k6 = colk[6]
        var k7 = colk[7]
        var x0 = tile[c]
        var x1 = tile[DOT_LD + c]
        var x2 = tile[2 * DOT_LD + c]
        var x3 = tile[3 * DOT_LD + c]
        var x4 = tile[4 * DOT_LD + c]
        var x5 = tile[5 * DOT_LD + c]
        var x6 = tile[6 * DOT_LD + c]
        var x7 = tile[7 * DOT_LD + c]
        while r + 8 <= cnt:
            var nr = r + 8
            var more = nr + 8 <= cnt
            var nk0 = k0
            var nk1 = k1
            var nk2 = k2
            var nk3 = k3
            var nk4 = k4
            var nk5 = k5
            var nk6 = k6
            var nk7 = k7
            var nx0 = x0
            var nx1 = x1
            var nx2 = x2
            var nx3 = x3
            var nx4 = x4
            var nx5 = x5
            var nx6 = x6
            var nx7 = x7
            if more:
                nk0 = colk[nr]
                nk1 = colk[nr + 1]
                nk2 = colk[nr + 2]
                nk3 = colk[nr + 3]
                nk4 = colk[nr + 4]
                nk5 = colk[nr + 5]
                nk6 = colk[nr + 6]
                nk7 = colk[nr + 7]
                nx0 = tile[nr * DOT_LD + c]
                nx1 = tile[(nr + 1) * DOT_LD + c]
                nx2 = tile[(nr + 2) * DOT_LD + c]
                nx3 = tile[(nr + 3) * DOT_LD + c]
                nx4 = tile[(nr + 4) * DOT_LD + c]
                nx5 = tile[(nr + 5) * DOT_LD + c]
                nx6 = tile[(nr + 6) * DOT_LD + c]
                nx7 = tile[(nr + 7) * DOT_LD + c]
            acc = ftz(identical_mul_add(k0, x0, acc))
            acc = ftz(identical_mul_add(k1, x1, acc))
            acc = ftz(identical_mul_add(k2, x2, acc))
            acc = ftz(identical_mul_add(k3, x3, acc))
            acc = ftz(identical_mul_add(k4, x4, acc))
            acc = ftz(identical_mul_add(k5, x5, acc))
            acc = ftz(identical_mul_add(k6, x6, acc))
            acc = ftz(identical_mul_add(k7, x7, acc))
            k0 = nk0
            k1 = nk1
            k2 = nk2
            k3 = nk3
            k4 = nk4
            k5 = nk5
            k6 = nk6
            k7 = nk7
            x0 = nx0
            x1 = nx1
            x2 = nx2
            x3 = nx3
            x4 = nx4
            x5 = nx5
            x6 = nx6
            x7 = nx7
            r = nr
    while r < cnt:
        acc = ftz(identical_mul_add(colk[r], tile[r * DOT_LD + c], acc))
        r += 1
    return acc


def geqrf_dot_tile_kernel(a: F32Ptr, scal: F32Ptr, w: F32Ptr, k: Int32, m: Int32, n: Int32):
    """`geqrf_dot` for the columns j0 .. j0 + DOT_COLS - 1 (j0 = k + 1 +
    block * DOT_COLS), the slabs coalesced through threadgroup memory."""
    var tile = stack_allocation[DOT_ROWS * DOT_LD, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var colk = stack_allocation[DOT_ROWS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var K = Int(k)
    var M = Int(m)
    var N = Int(n)
    var j0 = K + 1 + Int(block_idx.x) * DOT_COLS
    if j0 >= N:
        return
    var j = j0 + tid
    var mine = tid < DOT_COLS and j < N
    if scal.unsafe_load(1) == Float32(0):
        if mine:
            w.unsafe_store(j, Float32(0))
        return
    var acc = Float32(0)
    if mine:
        acc = ftz(a.unsafe_load(K * N + j))
    var c0 = K + 1
    while c0 < M:
        var cnt = min(DOT_ROWS, M - c0)
        var idx = tid
        while idx < cnt * DOT_COLS:
            var r = idx // DOT_COLS
            var c = idx - r * DOT_COLS
            if j0 + c < N:
                tile[r * DOT_LD + c] = ftz(a.unsafe_load((c0 + r) * N + j0 + c))
            idx += DOT_TPB
        var t = tid
        while t < cnt:
            colk[t] = ftz(a.unsafe_load((c0 + t) * N + K))
            t += DOT_TPB
        barrier()
        if mine:
            acc = _dot_tile_chain(colk, tile, tid, cnt, acc)
        barrier()
        c0 += DOT_ROWS
    if mine:
        w.unsafe_store(j, acc)


def orgqr_dot_tile_kernel(
    h: F32Ptr, tau: F32Ptr, q: F32Ptr, w: F32Ptr, k: Int32, m: Int32, n: Int32, qc: Int32,
):
    """`orgqr_dot` for the columns j0 .. j0 + DOT_COLS - 1 of Q (j0 = block *
    DOT_COLS), h's column k and Q's slabs through threadgroup memory."""
    var tile = stack_allocation[DOT_ROWS * DOT_LD, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var colk = stack_allocation[DOT_ROWS, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var K = Int(k)
    var M = Int(m)
    var N = Int(n)
    var QC = Int(qc)
    var j0 = Int(block_idx.x) * DOT_COLS
    if j0 >= QC:
        return
    var j = j0 + tid
    var mine = tid < DOT_COLS and j < QC
    if ftz(tau.unsafe_load(K)) == Float32(0):
        if mine:
            w.unsafe_store(j, Float32(0))
        return
    var acc = Float32(0)
    if mine:
        acc = ftz(q.unsafe_load(K * QC + j))
    var c0 = K + 1
    while c0 < M:
        var cnt = min(DOT_ROWS, M - c0)
        var idx = tid
        while idx < cnt * DOT_COLS:
            var r = idx // DOT_COLS
            var c = idx - r * DOT_COLS
            if j0 + c < QC:
                tile[r * DOT_LD + c] = ftz(q.unsafe_load((c0 + r) * QC + j0 + c))
            idx += DOT_TPB
        var t = tid
        while t < cnt:
            colk[t] = ftz(h.unsafe_load((c0 + t) * N + K))
            t += DOT_TPB
        barrier()
        if mine:
            acc = _dot_tile_chain(colk, tile, tid, cnt, acc)
        barrier()
        c0 += DOT_ROWS
    if mine:
        w.unsafe_store(j, acc)


def orgqr_dot_staged_kernel(
    h: F32Ptr, tau: F32Ptr, q: F32Ptr, w: F32Ptr, k: Int32, m: Int32, n: Int32, qc: Int32
):
    """`orgqr_dot` for column j = block, the fold staged."""
    var sh = stack_allocation[2 * STAGE, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var K = Int(k)
    var M = Int(m)
    var N = Int(n)
    var QC = Int(qc)
    var j = Int(block_idx.x)
    if j >= QC:
        return
    if ftz(tau.unsafe_load(K)) == Float32(0):
        if tid == 0:
            w.unsafe_store(j, Float32(0))
        return
    var acc = ftz(q.unsafe_load(K * QC + j))
    var c0 = K + 1
    while c0 < M:
        var cnt = min(STAGE, M - c0)
        var t = tid
        while t < cnt:
            sh[t] = ftz(h.unsafe_load((c0 + t) * N + K))
            sh[STAGE + t] = ftz(q.unsafe_load((c0 + t) * QC + j))
            t += STAGE_TPB
        barrier()
        if tid == 0:
            acc = _chain_fma(sh, 0, cnt, STAGE, acc)
        barrier()
        c0 += STAGE
    if tid == 0:
        w.unsafe_store(j, acc)



def _qr_tick(ctx: DeviceContext, mut qrt: List[Int], mut qrk: List[Int], slot: Int) raises:
    """MOJOLEARN_XD_QR_TIMING=1: a sync before and after each launch kind of
    the device QR, the wall accumulated per kind (slot -1 starts a span).
    Off by default: no sync, no bit."""
    if qrk[0] != 1:
        return
    ctx.synchronize()
    var now = Int(perf_counter_ns())
    if slot >= 0:
        qrt[slot] += now - qrk[1]
    qrk[1] = now


def _blocks(count: Int) -> Int:
    return (count + TPB - 1) // TPB if count > 0 else 1


# ---- host <-> device transfers through a pinned stage (lane neural-pass27, 2026-10-01)
#
# Measured on the Apple M4 through Mojo's Metal context, per 64 MB: a raw
# host-pointer upload costs ~20 ms the first time a host region is used (every
# fresh numpy array) and 1.5 ms warm; a device-to-host-pointer download ~21 ms
# every time; a pinned host buffer DMAs both ways in 2-3 ms, the CPU writes it
# at full speed, and the CPU reads it at ~3 GB/s on one thread (it is
# write-combined) but at 10 GB/s and more on four or more threads; a raw
# upload of a fresh array, on the other hand, is 1.6-2.4 ms, faster than a
# staged one. So on the Apple column every DOWNLOAD of at least XD_STAGE_MIN
# floats goes in chunks through ONE pinned stage of XD_STAGE_FLOATS: DMA out,
# then a read over host tasks; uploads stay raw. Copies only: no bit moves.
# MOJOLEARN_XD_STAGE=0 keeps the raw download on Apple; MOJOLEARN_XD_STAGE=1
# turns the stage on elsewhere (NVIDIA and AMD copy pageable memory at tens
# of GB/s and are not measured).
comptime XD_STAGE_FLOATS = 1 << 24
comptime XD_STAGE_MIN = 1 << 18


struct _XdStage(Defaultable, Movable):
    var buf: Optional[HostBuffer[DType.float32]]

    def __init__(out self):
        self.buf = Optional[HostBuffer[DType.float32]]()


comptime X_DECOMP_STAGE = _Global[StorageType=_XdStage, name="MojoXDecompStageIdentical", init_fn=_XdStage.__init__]


def _xd_staged(n: Int) -> Bool:
    if n < XD_STAGE_MIN:
        return False
    var v = String(getenv("MOJOLEARN_XD_STAGE"))
    comptime if TARGET_COLUMN == COLUMN_APPLE:
        return v != "0"
    return v == "1"


def _xd_stage_ptr(ctx: DeviceContext) raises -> F32Ptr:
    """The process's pinned stage (XD_STAGE_FLOATS floats), created on first use."""
    var slot = X_DECOMP_STAGE.get_or_create_ptr()
    if not slot[].buf:
        slot[].buf = ctx.enqueue_create_host_buffer[DType.float32](XD_STAGE_FLOATS)
        ctx.synchronize()
    return F32Ptr(unsafe_from_address=Int(slot[].buf.value().unsafe_ptr()))


def _xd_read_out(dst: F32Ptr, src: F32Ptr, n: Int):
    """`memcpy(dst, src, n)` over host tasks: the one read of pinned memory."""
    var tasks = host_predict_task_count(1 << 30)
    if tasks > 16:
        tasks = 16
    if n < XD_STAGE_MIN or tasks <= 1:
        memcpy(dest=dst, src=src, count=n)
        return
    var chunk = (n + tasks - 1) // tasks

    def _piece(t: Int) {imm dst, imm src, imm n, imm chunk}:
        var lo = t * chunk
        var hi = min(lo + chunk, n)
        if hi > lo:
            memcpy(dest=dst + lo, src=src + lo, count=hi - lo)

    host_parallelize(_piece, tasks)


def _up_into(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], p: F32Ptr, n: Int) raises:
    """Host floats into `buf[0, n)`: the raw host-pointer copy. Measured on
    the M4 a fresh 64 MB array uploads this way in 1.6-2.4 ms (only the
    process's first upload pays ~9-20 ms); a memcpy-plus-DMA stage took
    4.3 ms, so uploads are not staged."""
    if n > 0:
        ctx.enqueue_copy(dst_buf=buf.create_sub_buffer[DType.float32](0, n), src_ptr=p)


def _up(ctx: DeviceContext, p: F32Ptr, n: Int) raises -> DeviceBuffer[DType.float32]:
    var buf = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
    _up_into(ctx, buf, p, n)
    return buf^


def _up_i(ctx: DeviceContext, p: I32Ptr, n: Int) raises -> DeviceBuffer[DType.int32]:
    var buf = ctx.enqueue_create_buffer[DType.int32](n if n > 0 else 1)
    if n > 0:
        ctx.enqueue_copy(dst_buf=buf.create_sub_buffer[DType.int32](0, n), src_ptr=p)
    return buf^


def _down(ctx: DeviceContext, buf: DeviceBuffer[DType.float32], p: F32Ptr, n: Int) raises:
    if n <= 0:
        return
    if not _xd_staged(n):
        ctx.enqueue_copy(dst_ptr=p, src_buf=buf.create_sub_buffer[DType.float32](0, n))
        return
    var stage = _xd_stage_ptr(ctx)
    var off = 0
    while off < n:
        var cnt = min(XD_STAGE_FLOATS, n - off)
        ctx.enqueue_copy(dst_ptr=stage, src_buf=buf.create_sub_buffer[DType.float32](off, cnt))
        ctx.synchronize()
        _xd_read_out(p + off, stage, cnt)
        off += cnt


def _down_i(ctx: DeviceContext, buf: DeviceBuffer[DType.int32], p: I32Ptr, n: Int) raises:
    if n > 0:
        ctx.enqueue_copy(dst_ptr=p, src_buf=buf.create_sub_buffer[DType.int32](0, n))


def _p(buf: DeviceBuffer[DType.float32]) -> F32Ptr:
    """A device buffer's address as the cells' pointer type."""
    return F32Ptr(unsafe_from_address=Int(buf.unsafe_ptr()))


#: Pair-column cells (pairs x n) per chunk launch of the bounded one-sided
#: Jacobi SVD: about 0.2 s on the M2 Pro at n = 1,000 and 2,000.
comptime J2_CHUNK_CELLS = 1 << 22
#: `svd_of_r`'s single launch is left to shapes under this many columns (a
#: few milliseconds); from here on every solve is the bounded chunk route.
comptime J2_BOUNDED_MIN_N = 64


def _svd2_of_r(
    ctx: DeviceContext,
    mut r: DeviceBuffer[DType.float32],
    mut v: DeviceBuffer[DType.float32],
    mut s: DeviceBuffer[DType.float32],
    n: Int,
    cells: Int = J2_CHUNK_CELLS,
) raises:
    """`svd_of_r` (x_decomp's sweeps and tolerance) as `one_sided_svd2_kernel`
    stores it, BOUNDED IN WORK PER LAUNCH (x_decomp/jacobi2.mojo, lane/
    lle-timeout): rt = R^T and vt = I, then each sweep's cyclic pairs in
    chunks of about J2_CHUNK_CELLS pair-column cells (`one_sided_svd2_chunk_
    kernel`, the device waited for after each), each chunk's rotation count
    in its own slot, poisoned with -1 before the sweep: a slot still -1 after
    it is a launch that did not finish, refused. A sweep without a rotation
    ends the solve (the one launch's test); none in X_DECOMP_SVD_SWEEPS is
    the same refusal. Then the tail (`one_sided_svd2_finish_kernel`). The
    bits do not depend on `cells` (dense_check cuts a sweep at 7 pairs)."""
    var rt = ctx.enqueue_create_buffer[DType.float32](n * n)
    var vt = ctx.enqueue_create_buffer[DType.float32](n * n)
    ctx.enqueue_function[pj_transpose_kernel](
        r.unsafe_ptr(), rt.unsafe_ptr(), Int32(n), grid_dim=_pj_blocks(n * n), block_dim=PJ_TPB
    )
    ctx.enqueue_function[pj_identity_kernel](vt.unsafe_ptr(), Int32(n), grid_dim=_pj_blocks(n * n), block_dim=PJ_TPB)
    ctx.synchronize()
    var per = cells // n if n > 0 else 1
    if per < 1:
        per = 1
    var pairs = n * (n - 1) // 2
    var slots = (pairs + per - 1) // per if pairs > 0 else 1
    var rots = ctx.enqueue_create_buffer[DType.float32](slots)
    var hrots = ctx.enqueue_create_host_buffer[DType.float32](slots)
    var converged = pairs == 0
    var last = 0
    var sweep = 0
    while not converged and sweep < X_DECOMP_SVD_SWEEPS:
        enqueue_fill(ctx, rots, Float32(-1.0))
        var p = 0
        var q = 1
        var done = 0
        var k = 0
        while done < pairs:
            var cnt = per if pairs - done > per else pairs - done
            ctx.enqueue_function[one_sided_svd2_chunk_kernel](
                rt.unsafe_ptr(), vt.unsafe_ptr(), _p(rots) + k, Int32(n), Int32(p), Int32(q), Int32(cnt),
                X_DECOMP_SVD_TOL, grid_dim=(1, 1, 1), block_dim=(J2_TPB, 1, 1),
            )
            ctx.synchronize()
            # (p, q) advanced by cnt pairs in the cyclic order
            var left = cnt
            while left > 0:
                var row = n - 1 - q
                if left <= row:
                    q += left
                    left = 0
                else:
                    left -= row + 1
                    p += 1
                    q = p + 1
            done += cnt
            k += 1
        ctx.enqueue_copy(dst_ptr=hrots.unsafe_ptr(), src_buf=rots)
        ctx.synchronize()
        var total = 0
        for t in range(k):
            var c = hrots.unsafe_ptr().unsafe_load(t)
            if not (c >= Float32(0.0)):
                raise Error(
                    "the one-sided Jacobi SVD at n_cols = " + String(n) + ": chunk " + String(t) + " of sweep "
                    + String(sweep) + " did not finish (its rotation slot kept its poison); a launch the device"
                    " cut short is refused, never read as a converged answer"
                )
            total += Int(c)
        last = total
        sweep += 1
        if total == 0:
            converged = True
    if not converged:
        raise Error(
            "the one-sided Jacobi SVD did not converge in "
            + String(X_DECOMP_SVD_SWEEPS)
            + " sweeps at n_cols = "
            + String(n)
            + ": the last sweep still performed "
            + String(last)
            + " rotations against a tolerance of "
            + String(X_DECOMP_SVD_TOL)
            + ". The remedy is more sweeps, the same one cuSOLVER's syevj"
            " has. An unconverged decomposition is not returned as if it"
            " were one; see DEVIATION 590."
        )
    ctx.enqueue_function[one_sided_svd2_finish_kernel](
        r.unsafe_ptr(), v.unsafe_ptr(), s.unsafe_ptr(), rt.unsafe_ptr(), vt.unsafe_ptr(), Int32(n),
        grid_dim=(1, 1, 1), block_dim=(J2_TPB, 1, 1),
    )
    ctx.synchronize()
    _ = rots^
    _ = hrots^
    _ = rt^
    _ = vt^


def _pj_blocks(count: Int) -> Int:
    return (count + PJ_TPB - 1) // PJ_TPB if count > 0 else 1


def _svd_par_of_r(
    ctx: DeviceContext,
    mut r: DeviceBuffer[DType.float32],
    mut v: DeviceBuffer[DType.float32],
    mut s: DeviceBuffer[DType.float32],
    n: Int,
) raises -> Bool:
    """The one-sided Jacobi SVD of R in the round-robin ordering
    (x_decomp/jacobi_par.mojo; FAST on Metal): `v` and `s` as `svd_of_r`
    leaves them. `r` is NOT written, so on False (no convergence in
    PJ_SVD_SWEEPS sweeps) the caller runs the cyclic solver on it."""
    var m = n + (n % 2)
    var h = m // 2
    var rt = ctx.enqueue_create_buffer[DType.float32](n * n)
    var vt = ctx.enqueue_create_buffer[DType.float32](n * n)
    var flags = ctx.enqueue_create_buffer[DType.float32](2 * h)
    var hflags = ctx.enqueue_create_host_buffer[DType.float32](2 * h)
    var hs = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_function[pj_transpose_kernel](
        r.unsafe_ptr(), rt.unsafe_ptr(), Int32(n), grid_dim=_pj_blocks(n * n), block_dim=PJ_TPB
    )
    ctx.enqueue_function[pj_identity_kernel](vt.unsafe_ptr(), Int32(n), grid_dim=_pj_blocks(n * n), block_dim=PJ_TPB)
    # ||R||_F^2 before the sweeps (the rotations keep it): the column norms
    ctx.enqueue_function[svd_par_norm_kernel](
        rt.unsafe_ptr(), s.unsafe_ptr(), Int32(n), grid_dim=(n, 1, 1), block_dim=(PJ_TPB, 1, 1)
    )
    ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=s)
    ctx.synchronize()
    var fro_in = Float64(0.0)
    for i in range(n):
        var x = Float64(hs.unsafe_ptr().unsafe_load(i))
        fro_in += x * x
    var converged = False
    var ran = True
    for _sweep in range(PJ_SVD_SWEEPS):
        enqueue_fill(ctx, flags, Float32(0.0))
        for rd in range(m - 1):
            ctx.enqueue_function[svd_par_round_kernel](
                rt.unsafe_ptr(), vt.unsafe_ptr(), flags.unsafe_ptr(), Int32(n), Int32(m), Int32(rd), X_DECOMP_SVD_TOL,
                grid_dim=(h, 1, 1), block_dim=(PJ_TPB, 1, 1),
            )
            if rd % PJ_SYNC_ROUNDS == PJ_SYNC_ROUNDS - 1:
                ctx.synchronize()
        ctx.enqueue_copy(dst_ptr=hflags.unsafe_ptr(), src_buf=flags)
        ctx.synchronize()
        var rots = 0
        for i in range(h):
            if hflags.unsafe_ptr().unsafe_load(i) != Float32(0.0):
                rots += 1
            # every block of the sweep's last round wrote its round number
            if hflags.unsafe_ptr().unsafe_load(h + i) != Float32(m - 1):
                ran = False
        if not ran:
            break
        if rots == 0:
            converged = True
            break
    if converged:
        ctx.enqueue_function[svd_par_norm_kernel](
            rt.unsafe_ptr(), s.unsafe_ptr(), Int32(n), grid_dim=(n, 1, 1), block_dim=(PJ_TPB, 1, 1)
        )
        ctx.enqueue_function[pj_transpose_kernel](
            vt.unsafe_ptr(), v.unsafe_ptr(), Int32(n), grid_dim=_pj_blocks(n * n), block_dim=PJ_TPB
        )
        ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=s)
        ctx.synchronize()
        # the rotations are orthogonal: sum of s^2 is ||R||_F^2, or the
        # solve is not an answer (a dropped dispatch, a wrong launch)
        var fro_out = Float64(0.0)
        for i in range(n):
            var x = Float64(hs.unsafe_ptr().unsafe_load(i))
            fro_out += x * x
        if not (abs(fro_out - fro_in) <= 1.0e-3 * fro_in):
            converged = False
    ctx.synchronize()
    _ = rt^
    _ = vt^
    _ = flags^
    _ = hflags^
    _ = hs^
    return converged


def gemm_scratch(m: Int, k: Int, n: Int) -> Int:
    """Floats of partial-sum scratch `launch_gemm` needs (0: none)."""
    var nb = (k + FOLD_BLOCK - 1) // FOLD_BLOCK
    return nb * m * n if nb > 1 else 0


def launch_gemm(
    ctx: DeviceContext, a: F32Ptr, b: F32Ptr, c: F32Ptr, p: F32Ptr, m: Int, k: Int, n: Int, ta: Bool, tb: Bool
) raises:
    """C = op(A) op(B) on device pointers, enqueued (no sync): FOLD_BLOCK
    partial sums then the fold past one block (DEVIATIONS 5300/5301)."""
    var nb = (k + FOLD_BLOCK - 1) // FOLD_BLOCK
    if nb > 1:
        ctx.enqueue_function[gemm_part_kernel](
            a, b, p, Int32(m), Int32(k), Int32(n),
            Int32(1 if ta else 0), Int32(1 if tb else 0), Int32(nb), grid_dim=_blocks(nb * m * n), block_dim=TPB,
        )
        ctx.enqueue_function[fold_kernel](p, c, Int32(m * n), Int32(nb), grid_dim=_blocks(m * n), block_dim=TPB)
    else:
        ctx.enqueue_function[gemm_kernel](
            a, b, c, Int32(m), Int32(k), Int32(n),
            Int32(1 if ta else 0), Int32(1 if tb else 0), grid_dim=_blocks(m * n), block_dim=TPB,
        )


def launch_ew(
    ctx: DeviceContext, op: Int, a: F32Ptr, b: F32Ptr, bm: Int, c: F32Ptr, cm: Int, dst: F32Ptr,
    count: Int, d: Int, s: Float32,
) raises:
    ctx.enqueue_function[ew_kernel](
        Int32(op), a, b, Int32(bm), c, Int32(cm), dst, Int32(count), Int32(d), s,
        grid_dim=_blocks(count), block_dim=TPB,
    )


def colsum_scratch(n: Int, d: Int) -> Int:
    var nb = (n + FOLD_BLOCK - 1) // FOLD_BLOCK
    return nb * d if nb > 1 else 0


def launch_colsum(ctx: DeviceContext, a: F32Ptr, dst: F32Ptr, p: F32Ptr, n: Int, d: Int) raises:
    var nb = (n + FOLD_BLOCK - 1) // FOLD_BLOCK
    if nb > 1:
        ctx.enqueue_function[colsum_part_kernel](
            a, p, Int32(n), Int32(d), Int32(nb), grid_dim=_blocks(nb * d), block_dim=TPB
        )
        ctx.enqueue_function[fold_kernel](p, dst, Int32(d), Int32(nb), grid_dim=_blocks(d), block_dim=TPB)
    else:
        ctx.enqueue_function[colsum_kernel](a, dst, Int32(n), Int32(d), grid_dim=_blocks(d), block_dim=TPB)


def rowsum_scratch(n: Int, d: Int) -> Int:
    var nb = (d + FOLD_BLOCK - 1) // FOLD_BLOCK
    return nb * n if nb > 1 else 0


def launch_rowsum(ctx: DeviceContext, a: F32Ptr, dst: F32Ptr, p: F32Ptr, n: Int, d: Int) raises:
    var nb = (d + FOLD_BLOCK - 1) // FOLD_BLOCK
    if nb > 1:
        ctx.enqueue_function[rowsum_part_kernel](
            a, p, Int32(n), Int32(d), Int32(nb), grid_dim=_blocks(nb * n), block_dim=TPB
        )
        ctx.enqueue_function[fold_kernel](p, dst, Int32(n), Int32(nb), grid_dim=_blocks(n), block_dim=TPB)
    else:
        ctx.enqueue_function[rowsum_kernel](a, dst, Int32(n), Int32(d), grid_dim=_blocks(n), block_dim=TPB)


def launch_sqdist(
    ctx: DeviceContext, a: F32Ptr, b: F32Ptr, dst: F32Ptr, na: Int, nb: Int, d: Int, kind: Int, pw: Float32
) raises:
    ctx.enqueue_function[sqdist_kernel](
        a, b, dst, Int32(na), Int32(nb), Int32(d), Int32(kind), pw, grid_dim=_blocks(na * nb), block_dim=TPB,
    )


def absmax_scratch(n: Int, d: Int, by_col: Bool) -> Int:
    var cnt = d if by_col else n
    var length = n if by_col else d
    var nb = (length + FOLD_BLOCK - 1) // FOLD_BLOCK
    return 2 * nb * cnt if nb > 1 and cnt > 0 else 0


def launch_absmax(ctx: DeviceContext, a: F32Ptr, dst: F32Ptr, p: F32Ptr, n: Int, d: Int, by_col: Bool) raises:
    """absmax_sign_cell per column (by_col) or row, in FOLD_BLOCK slices past
    one block (DEVIATION 5317), enqueued."""
    var cnt = d if by_col else n
    var length = n if by_col else d
    var nb = (length + FOLD_BLOCK - 1) // FOLD_BLOCK
    if nb > 1:
        ctx.enqueue_function[absmax_part_kernel](
            a, p, Int32(n), Int32(d), Int32(1 if by_col else 0), Int32(nb),
            grid_dim=_blocks(nb * cnt), block_dim=TPB,
        )
        ctx.enqueue_function[absmax_fold_kernel](p, dst, Int32(cnt), Int32(nb), grid_dim=_blocks(cnt), block_dim=TPB)
    else:
        ctx.enqueue_function[absmax_kernel](
            a, dst, Int32(n), Int32(d), Int32(1 if by_col else 0), grid_dim=_blocks(cnt), block_dim=TPB,
        )


def orth_on_device(ctx: DeviceContext, da: DeviceBuffer[DType.float32], m: Int, l: Int) raises:
    """DevExec.orth's two passes on a device matrix, in place (its values
    replaced by the orthonormalized columns): R of the matrix (`qr_factor`
    on a device copy, as `device_qr_r`), the rank guard on the host
    (DEVIATION 5318), then A R^-1 by rows (`trsm_kernel`, DEVIATION 5309).
    Waits for the device (the guard reads R on the host)."""
    var cells = m * l if m * l > 0 else 1
    var dq = ctx.enqueue_create_buffer[DType.float32](cells)
    var dw = ctx.enqueue_create_buffer[DType.float32](cells)
    var scratch = ctx.enqueue_create_buffer[DType.float32](qr_slice_count(m, l) * l * l if l > 0 else 1)
    var r_buf = ctx.enqueue_create_buffer[DType.float32](l * l if l > 0 else 1)
    var r = List[Float32](length=l * l if l > 0 else 1, fill=Float32(0))
    var dev_guard = String(getenv("MOJOLEARN_XD_ORTH_DEV", "1")) != "0"
    for p in range(2):
        var src = da if p == 0 else dq
        var dst = dq if p == 0 else da
        ctx.enqueue_copy(dst_buf=dw, src_buf=src)
        if dev_guard:
            # lane/decomp-apple2: the guard cell on one device thread, in
            # stream order after the R it reads (the same cell the host ran;
            # IDENTICAL cells are bit-equal on both), so R never leaves the
            # device and the pass waits once, not three times.
            _ = qr_factor(ctx, dw, scratch, r_buf, m, l)
            ctx.enqueue_function[orth_guard_kernel](r_buf.unsafe_ptr(), Int32(l), grid_dim=1, block_dim=1)
        else:
            ctx.synchronize()
            _ = qr_factor(ctx, dw, scratch, r_buf, m, l)
            _down(ctx, r_buf, F32Ptr(unsafe_from_address=Int(r.unsafe_ptr())), l * l)
            ctx.synchronize()
            orth_rank_guard(F32Ptr(unsafe_from_address=Int(r.unsafe_ptr())), l)
            ctx.enqueue_copy(dst_buf=r_buf.create_sub_buffer[DType.float32](0, l * l), src_ptr=F32Ptr(unsafe_from_address=Int(r.unsafe_ptr())))
        ctx.enqueue_function[trsm_kernel](
            src.unsafe_ptr(), r_buf.unsafe_ptr(), dst.unsafe_ptr(), Int32(m), Int32(l), grid_dim=_blocks(m), block_dim=TPB
        )
        ctx.synchronize()
        _ = src^
        _ = dst^
    _ = dq^
    _ = dw^
    _ = scratch^
    _ = r_buf^
    ctx.synchronize()


@fieldwise_init
struct DevExec(Exec):
    @staticmethod
    def gemm(a: F32Ptr, b: F32Ptr, c: F32Ptr, m: Int, k: Int, n: Int, ta: Bool, tb: Bool) raises:
        var ctx = xd_ctx()
        var da = _up(ctx, a, m * k)
        var db = _up(ctx, b, k * n)
        var dc = ctx.enqueue_create_buffer[DType.float32](m * n if m * n > 0 else 1)
        var ns = gemm_scratch(m, k, n)
        var dp = ctx.enqueue_create_buffer[DType.float32](ns if ns > 0 else 1)
        launch_gemm(ctx, _p(da), _p(db), _p(dc), _p(dp), m, k, n, ta, tb)
        _down(ctx, dc, c, m * n)
        ctx.synchronize()
        _ = da^
        _ = db^
        _ = dc^
        _ = dp^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def ew(
        op: Int, a: F32Ptr, b: F32Ptr, lb: Int, bm: Int, c: F32Ptr, lc: Int, cm: Int,
        dst: F32Ptr, count: Int, d: Int, s: Float32,
    ) raises:
        var ctx = xd_ctx()
        var da = _up(ctx, a, count)
        var db = _up(ctx, b, lb)
        var dc = _up(ctx, c, lc)
        var dout = ctx.enqueue_create_buffer[DType.float32](count if count > 0 else 1)
        launch_ew(ctx, op, _p(da), _p(db), bm, _p(dc), cm, _p(dout), count, d, s)
        _down(ctx, dout, dst, count)
        ctx.synchronize()
        _ = da^
        _ = db^
        _ = dc^
        _ = dout^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def colsum(a: F32Ptr, dst: F32Ptr, n: Int, d: Int) raises:
        var ctx = xd_ctx()
        var da = _up(ctx, a, n * d)
        var dout = ctx.enqueue_create_buffer[DType.float32](d if d > 0 else 1)
        var ns = colsum_scratch(n, d)
        var dp = ctx.enqueue_create_buffer[DType.float32](ns if ns > 0 else 1)
        launch_colsum(ctx, _p(da), _p(dout), _p(dp), n, d)
        _down(ctx, dout, dst, d)
        ctx.synchronize()
        _ = da^
        _ = dout^
        _ = dp^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def rowsum(a: F32Ptr, dst: F32Ptr, n: Int, d: Int) raises:
        var ctx = xd_ctx()
        var da = _up(ctx, a, n * d)
        var dout = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        var ns = rowsum_scratch(n, d)
        var dp = ctx.enqueue_create_buffer[DType.float32](ns if ns > 0 else 1)
        launch_rowsum(ctx, _p(da), _p(dout), _p(dp), n, d)
        _down(ctx, dout, dst, n)
        ctx.synchronize()
        _ = da^
        _ = dout^
        _ = dp^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def sqdist(a: F32Ptr, b: F32Ptr, dst: F32Ptr, na: Int, nb: Int, d: Int, kind: Int = 0, pw: Float32 = Float32(2)) raises:
        var ctx = xd_ctx()
        var da = _up(ctx, a, na * d)
        var db = _up(ctx, b, nb * d)
        var dout = ctx.enqueue_create_buffer[DType.float32](na * nb if na * nb > 0 else 1)
        launch_sqdist(ctx, _p(da), _p(db), _p(dout), na, nb, d, kind, pw)
        _down(ctx, dout, dst, na * nb)
        ctx.synchronize()
        _ = da^
        _ = db^
        _ = dout^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def rand(dst: F32Ptr, count: Int, seed: UInt32, stream: UInt32, kind: Int) raises:
        var ctx = xd_ctx()
        var dout = ctx.enqueue_create_buffer[DType.float32](count if count > 0 else 1)
        ctx.enqueue_function[rand_kernel](
            dout.unsafe_ptr(), Int32(count), seed, stream, Int32(kind), grid_dim=_blocks(count), block_dim=TPB
        )
        _down(ctx, dout, dst, count)
        ctx.synchronize()
        _ = dout^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def lu(a: F32Ptr, piv: I32Ptr, info: F32Ptr, n: Int) raises:
        var ctx = xd_ctx()
        var da = _up(ctx, a, n * n)
        var dp = ctx.enqueue_create_buffer[DType.int32](n if n > 0 else 1)
        var di = ctx.enqueue_create_buffer[DType.float32](1)
        var ds = ctx.enqueue_create_buffer[DType.float32](2)
        # lu_serial's cells, step by step: the pivot search one thread, the
        # swap, the multipliers and the trailing update one thread per cell
        if n <= lu_serial_max():
            # A small matrix: lu_serial itself on one device thread, the
            # same cells in the same order as the step-by-step launches
            # below (which exist for large n), in ONE launch instead of 5n.
            ctx.enqueue_function[lu_kernel](da.unsafe_ptr(), dp.unsafe_ptr(), di.unsafe_ptr(), Int32(n), grid_dim=1, block_dim=1)
        else:
            ctx.enqueue_function[lu_info_init_kernel](di.unsafe_ptr(), grid_dim=1, block_dim=1)
        var pivot_block = lu_pivot_parallel()
        var nb = min(lu_panel_width(), LU_PANEL_NB)
        var dact = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        if n > lu_serial_max() and nb > 0:
            # The blocked route (lane neural-pass32): the panel's steps run
            # the per-step kernels over the panel's columns; the trailing
            # columns then get the panel's swaps in order, the U rows
            # brought up to date, and every trailing cell the panel's
            # steps in order through tiles. The same cells in the same
            # order as the per-step route below.
            var k0 = 0
            while k0 < n:
                var k1 = min(k0 + nb, n)
                for k in range(k0, k1):
                    if pivot_block:
                        ctx.enqueue_function[lu_pivot_block_kernel](
                            da.unsafe_ptr(), dp.unsafe_ptr(), Int32(k), Int32(n), grid_dim=1, block_dim=LU_PIVOT_TPB
                        )
                    else:
                        ctx.enqueue_function[lu_pivot_kernel](da.unsafe_ptr(), dp.unsafe_ptr(), Int32(k), Int32(n), grid_dim=1, block_dim=1)
                    ctx.enqueue_function[lu_swap_cols_kernel](
                        da.unsafe_ptr(), dp.unsafe_ptr(), Int32(k), Int32(n), Int32(0), Int32(k1),
                        grid_dim=_blocks(k1), block_dim=TPB,
                    )
                    ctx.enqueue_function[lu_diag_kernel](
                        da.unsafe_ptr(), di.unsafe_ptr(), ds.unsafe_ptr(), Int32(k), Int32(n), grid_dim=1, block_dim=1
                    )
                    ctx.enqueue_function[lu_act_kernel](ds.unsafe_ptr(), dact.unsafe_ptr(), Int32(k), grid_dim=1, block_dim=1)
                    if n - k - 1 > 0:
                        ctx.enqueue_function[lu_l_kernel](
                            da.unsafe_ptr(), ds.unsafe_ptr(), Int32(k), Int32(n), grid_dim=_blocks(n - k - 1), block_dim=TPB
                        )
                    if k1 - k - 1 > 0 and n - k - 1 > 0:
                        ctx.enqueue_function[lu_update_panel_kernel](
                            da.unsafe_ptr(), ds.unsafe_ptr(), Int32(k), Int32(n), Int32(k1),
                            grid_dim=_blocks((n - k - 1) * (k1 - k - 1)), block_dim=TPB,
                        )
                if k1 < n:
                    ctx.enqueue_function[lu_apply_swaps_kernel](
                        da.unsafe_ptr(), dp.unsafe_ptr(), Int32(k0), Int32(k1), Int32(n),
                        grid_dim=_blocks(n - k1), block_dim=TPB,
                    )
                    if k1 - k0 > 1:
                        ctx.enqueue_function[lu_trsm_kernel](
                            da.unsafe_ptr(), dact.unsafe_ptr(), Int32(k0), Int32(k1), Int32(n),
                            grid_dim=_blocks(n - k1), block_dim=TPB,
                        )
                    var tiles = (n - k1 + LU_TILE - 1) // LU_TILE
                    ctx.enqueue_function[lu_trail_tiled_kernel](
                        da.unsafe_ptr(), dact.unsafe_ptr(), Int32(k0), Int32(k1), Int32(n), Int32(k1 - k0),
                        grid_dim=(tiles, tiles, 1), block_dim=(LU_TILE_TPB, 1, 1),
                    )
                k0 = k1
        for k in range(n if n > lu_serial_max() and nb == 0 else 0):
            if pivot_block:
                ctx.enqueue_function[lu_pivot_block_kernel](
                    da.unsafe_ptr(), dp.unsafe_ptr(), Int32(k), Int32(n), grid_dim=1, block_dim=LU_PIVOT_TPB
                )
            else:
                ctx.enqueue_function[lu_pivot_kernel](da.unsafe_ptr(), dp.unsafe_ptr(), Int32(k), Int32(n), grid_dim=1, block_dim=1)
            ctx.enqueue_function[lu_swap_kernel](
                da.unsafe_ptr(), dp.unsafe_ptr(), Int32(k), Int32(n), grid_dim=_blocks(n), block_dim=TPB
            )
            ctx.enqueue_function[lu_diag_kernel](
                da.unsafe_ptr(), di.unsafe_ptr(), ds.unsafe_ptr(), Int32(k), Int32(n), grid_dim=1, block_dim=1
            )
            if n - k - 1 > 0:
                ctx.enqueue_function[lu_l_kernel](
                    da.unsafe_ptr(), ds.unsafe_ptr(), Int32(k), Int32(n), grid_dim=_blocks(n - k - 1), block_dim=TPB
                )
                ctx.enqueue_function[lu_update_kernel](
                    da.unsafe_ptr(), ds.unsafe_ptr(), Int32(k), Int32(n),
                    grid_dim=_blocks((n - k - 1) * (n - k - 1)), block_dim=TPB,
                )
        _down(ctx, da, a, n * n)
        _down_i(ctx, dp, piv, n)
        _down(ctx, di, info, 1)
        ctx.synchronize()
        _ = da^
        _ = dp^
        _ = di^
        _ = ds^
        _ = dact^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def lu_solve(lu: F32Ptr, piv: I32Ptr, b: F32Ptr, n: Int, nrhs: Int, trans: Int = 0) raises:
        var ctx = xd_ctx()
        var dl = _up(ctx, lu, n * n)
        var dp = _up_i(ctx, piv, n)
        var db = _up(ctx, b, n * nrhs)
        if String(getenv("MOJOLEARN_XD_LU_SOLVE_SERIAL")) == "1":
            ctx.enqueue_function[lu_solve_kernel](
                dl.unsafe_ptr(), dp.unsafe_ptr(), db.unsafe_ptr(), Int32(n), Int32(nrhs), Int32(trans), grid_dim=1, block_dim=1
            )
        else:
            # One thread per right-hand side (lane/neural-net-experiment).
            ctx.enqueue_function[lu_solve_cols_kernel](
                dl.unsafe_ptr(), dp.unsafe_ptr(), db.unsafe_ptr(), Int32(n), Int32(nrhs), Int32(trans),
                grid_dim=_blocks(nrhs), block_dim=TPB,
            )
        _down(ctx, db, b, n * nrhs)
        ctx.synchronize()
        _ = dl^
        _ = dp^
        _ = db^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def chol(a: F32Ptr, info: F32Ptr, n: Int) raises:
        var ctx = xd_ctx()
        var da = _up(ctx, a, n * n)
        var di = ctx.enqueue_create_buffer[DType.float32](1)
        # lane/neural-net-experiment (2026-09-30, the classical pass): the
        # unblocked left-looking Cholesky ran as ONE thread for every n
        # (n^3 / 6 fmas in a row). `chol_serial`'s cells, step by step:
        # column j's diagonal on one thread (its j-long chain), then every
        # row below it one thread per row (each its own j-long chain, the
        # same chain `chol_serial` walks for that cell, reading only cells
        # final before the step), and the mirror cell zeroed there. Same
        # cells, same order per cell, same bits; 2n launches.
        if n <= chol_serial_max():
            ctx.enqueue_function[chol_kernel](da.unsafe_ptr(), di.unsafe_ptr(), Int32(n), grid_dim=1, block_dim=1)
        else:
            ctx.enqueue_function[lu_info_init_kernel](di.unsafe_ptr(), grid_dim=1, block_dim=1)
            for j in range(n):
                ctx.enqueue_function[chol_diag_kernel](
                    da.unsafe_ptr(), di.unsafe_ptr(), Int32(j), Int32(n), grid_dim=1, block_dim=1
                )
                if n - j - 1 > 0:
                    ctx.enqueue_function[chol_col_kernel](
                        da.unsafe_ptr(), Int32(j), Int32(n), grid_dim=_blocks(n - j - 1), block_dim=TPB
                    )
        _down(ctx, da, a, n * n)
        _down(ctx, di, info, 1)
        ctx.synchronize()
        _ = da^
        _ = di^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def eigh(a: F32Ptr, w: F32Ptr, v: F32Ptr, n: Int) raises:
        # lane/neural-net-experiment (2026-09-30): the two routes below were
        # compiled for Metal FAST only. Both envs are honoured on every
        # vendor and tier now, with their defaults unchanged (0: never),
        # so a box can A/B them:
        #   MOJOLEARN_XD_HOST_EIGH_MAX=n  the host executor's cyclic Jacobi
        #     for n at or under it: the host column's own arithmetic, held
        #     bit for bit to the device kernels by the identity gates.
        #   MOJOLEARN_XD_PJ_EIGH_MIN=n  the round-robin ordering
        #     (x_decomp/jacobi_par.mojo) from n up: NOT the pinned cyclic
        #     order, so NOT the identical tier's bits -- an experiment the
        #     digest check must report as MOVED. It is the only route here
        #     whose rotations run across the GPU: the cyclic kernels are one
        #     block of 256 threads for the whole solve (eigh at n = 4096
        #     timed out on the AMD board).
        if n <= host_eigh_max():
            HostExec.eigh(a, w, v, n)
            return
        var lo = pj_eigh_min()
        if lo > 0 and n >= lo:
            if DevExec._eigh_par(a, w, v, n):
                return
        if jacobi2_eigh_on():
            DevExec._eigh2(a, w, v, n)
            return
        var m = List[Float32](capacity=n * n)
        for i in range(n * n):
            m.append(a.unsafe_load(i))
        var got = device_eigh(xd_ctx(), m, n)
        for i in range(n):
            w.unsafe_store(i, got.w[i])
        for i in range(n * n):
            v.unsafe_store(i, got.v[i])

    @staticmethod
    def _eigh_par(a: F32Ptr, w: F32Ptr, v: F32Ptr, n: Int) raises -> Bool:
        """The two-sided Jacobi in the round-robin ordering
        (x_decomp/jacobi_par.mojo; FAST on Metal), then `device_eigh`'s own
        tail (`sign_flip_kernel`, the ascending permutation). The cyclic
        kernel's convergence test, taken on the host before every sweep.
        `a` is not written; False = not converged in PJ_EIGH_SWEEPS sweeps
        (nothing stored), and the caller runs the cyclic solver."""
        var ctx = xd_ctx()
        var m = n + (n % 2)
        var h = m // 2
        var da = _up(ctx, a, n * n)
        var dv = ctx.enqueue_create_buffer[DType.float32](n * n)
        var dcs = ctx.enqueue_create_buffer[DType.float32](2 * h)
        var doff = ctx.enqueue_create_buffer[DType.float32](3 * n)
        var hoff = ctx.enqueue_create_host_buffer[DType.float32](3 * n)
        ctx.enqueue_function[pj_identity_kernel](dv.unsafe_ptr(), Int32(n), grid_dim=_pj_blocks(n * n), block_dim=PJ_TPB)
        var tol2 = Float64(JACOBI_TOL) * Float64(JACOBI_TOL)
        var converged = False
        var executed = 0
        var fro_in = Float64(-1.0)
        var fro_now = Float64(0.0)
        for sweep in range(PJ_EIGH_SWEEPS + 1):
            # a sum of squares is never negative: -1 left in the readback is
            # a dispatch that did not run
            enqueue_fill(ctx, doff, Float32(-1.0))
            ctx.enqueue_function[eigh_par_off_kernel](
                da.unsafe_ptr(), doff.unsafe_ptr(), Int32(n), grid_dim=_pj_blocks(n), block_dim=PJ_TPB
            )
            ctx.enqueue_copy(dst_ptr=hoff.unsafe_ptr(), src_buf=doff)
            ctx.synchronize()
            var off = Float64(0.0)
            var dg = Float64(0.0)
            var ran = True
            for i in range(n):
                var o = Float64(hoff.unsafe_ptr().unsafe_load(i))
                var d2 = Float64(hoff.unsafe_ptr().unsafe_load(n + i))
                if o < 0.0 or d2 < 0.0:
                    ran = False
                off += o
                dg += d2
            if not ran:
                break
            fro_now = off + dg
            if fro_in < 0.0:
                fro_in = fro_now
            if off <= tol2 * fro_now:
                converged = True
                break
            if sweep == PJ_EIGH_SWEEPS:
                break
            executed += 1
            for rd in range(m - 1):
                ctx.enqueue_function[eigh_par_cs_kernel](
                    da.unsafe_ptr(), dcs.unsafe_ptr(), Int32(n), Int32(m), Int32(rd),
                    grid_dim=_pj_blocks(h), block_dim=PJ_TPB,
                )
                ctx.enqueue_function[eigh_par_update_kernel](
                    da.unsafe_ptr(), dv.unsafe_ptr(), dcs.unsafe_ptr(), Int32(n), Int32(m), Int32(rd),
                    grid_dim=_pj_blocks(h * h + n * h), block_dim=PJ_TPB,
                )
                if rd % PJ_SYNC_ROUNDS == PJ_SYNC_ROUNDS - 1:
                    ctx.synchronize()
        # J^T A J keeps ||A||_F: a solve that moved it is not an answer
        if converged and not (abs(fro_now - fro_in) <= 1.0e-3 * fro_in):
            converged = False
        if converged:
            ctx.enqueue_function[sign_flip_kernel](
                dv.unsafe_ptr(), Int32(n), grid_dim=(n, 1, 1), block_dim=(SIGNFLIP_TPB, 1, 1)
            )
            var hv = ctx.enqueue_create_host_buffer[DType.float32](n * n)
            ctx.enqueue_copy(dst_ptr=hv.unsafe_ptr(), src_buf=dv)
            ctx.synchronize()
            # the last test's readback holds the diagonal of the converged A
            var diag = List[Float32](capacity=n)
            for i in range(n):
                diag.append(hoff.unsafe_ptr().unsafe_load(2 * n + i))
            var vecs = List[Float32](capacity=n * n)
            for i in range(n * n):
                vecs.append(hv.unsafe_ptr().unsafe_load(i))
            var got = eigh_ascending(diag, vecs, n, True, executed)
            for i in range(n):
                w.unsafe_store(i, got.w[i])
            for i in range(n * n):
                v.unsafe_store(i, got.v[i])
            _ = hv^
        _ = da^
        _ = dv^
        _ = dcs^
        _ = doff^
        _ = hoff^
        ctx.synchronize()
        _ = ctx^
        return converged

    @staticmethod
    def _eigh2(a: F32Ptr, w: F32Ptr, v: F32Ptr, n: Int) raises:
        """`device_eigh` with `jacobi_eigh2_kernel` in place of
        `jacobi_eigh_kernel`: same launch pair (the sweep, then
        `sign_flip_kernel`), same refusals, same ascending permutation."""
        var ctx = xd_ctx()
        var da = _up(ctx, a, n * n)
        var dv = ctx.enqueue_create_buffer[DType.float32](n * n)
        var dinfo = ctx.enqueue_create_buffer[DType.float32](3)
        enqueue_fill(ctx, dinfo, JACOBI_INFO_UNWRITTEN)
        var dvt = ctx.enqueue_create_buffer[DType.float32](n * n)
        # unroll 1 is the default: m4pro-b 1790619265077, eigh 1500 46.3 s
        # at unroll 1 against 82.7 s at 4 and 98.6 s for the old kernel,
        # every digest equal (MOJOLEARN_XD_J2_U=4 keeps the other one)
        if String(getenv("MOJOLEARN_XD_J2_U", "1")) != "4":
            ctx.enqueue_function[jacobi_eigh2_kernel[1]](
                da.unsafe_ptr(), dv.unsafe_ptr(), dinfo.unsafe_ptr(), dvt.unsafe_ptr(), Int32(n), Int32(JACOBI_SWEEPS), Float32(JACOBI_TOL),
                grid_dim=(1, 1, 1), block_dim=(J2_TPB, 1, 1),
            )
        else:
            ctx.enqueue_function[jacobi_eigh2_kernel[4]](
                da.unsafe_ptr(), dv.unsafe_ptr(), dinfo.unsafe_ptr(), dvt.unsafe_ptr(), Int32(n), Int32(JACOBI_SWEEPS), Float32(JACOBI_TOL),
                grid_dim=(1, 1, 1), block_dim=(J2_TPB, 1, 1),
            )
        ctx.enqueue_function[sign_flip_kernel](dv.unsafe_ptr(), Int32(n), grid_dim=(n, 1, 1), block_dim=(SIGNFLIP_TPB, 1, 1))
        var hinfo = ctx.enqueue_create_host_buffer[DType.float32](3)
        var hwork = ctx.enqueue_create_host_buffer[DType.float32](n * n)
        ctx.enqueue_copy(dst_ptr=hinfo.unsafe_ptr(), src_buf=dinfo)
        ctx.enqueue_copy(dst_ptr=hwork.unsafe_ptr(), src_buf=da.create_sub_buffer[DType.float32](0, n * n))
        _down(ctx, dv, v, n * n)
        ctx.synchronize()
        var i0 = hinfo.unsafe_ptr().unsafe_load(0)
        var i1 = hinfo.unsafe_ptr().unsafe_load(1)
        var i2 = hinfo.unsafe_ptr().unsafe_load(2)
        if i0 == JACOBI_INFO_UNWRITTEN:
            raise Error(
                "eigh: the device Jacobi eigensolver DID NOT WRITE its info"
                " buffer, so it never ran or its launch failed. This is NOT a"
                " convergence failure and must not be reported as one: -1.0 is a"
                " value the kernel never stores. Check that the binding is built"
                " for this device."
            )
        if i0 == Float32(0.0):
            raise Error(
                "eigh: the Jacobi eigensolver did not converge in "
                + String(JACOBI_SWEEPS)
                + " sweeps at n = "
                + String(n)
                + ": ||offdiag(A)||_F / ||A||_F is still "
                + String(i1)
                + ". An unconverged decomposition is not returned as if it were"
                " one; see DEVIATION 590. The remedy is more sweeps, the same one"
                " cuSOLVER's syevj has"
            )
        var diag = List[Float32](capacity=n)
        for i in range(n):
            diag.append(hwork.unsafe_ptr().unsafe_load(i * n + i))
        var vecs = List[Float32](capacity=n * n)
        for i in range(n * n):
            vecs.append(v.unsafe_load(i))
        var got = eigh_ascending(diag, vecs, n, True, Int(i2))
        for i in range(n):
            w.unsafe_store(i, got.w[i])
        for i in range(n * n):
            v.unsafe_store(i, got.v[i])
        _ = da^
        _ = dv^
        _ = dinfo^
        _ = dvt^
        _ = hinfo^
        _ = hwork^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def cd_rows(w: F32Ptr, hht: F32Ptr, xht: F32Ptr, perm: I32Ptr, viol: F32Ptr, n: Int, k: Int) raises:
        var ctx = xd_ctx()
        var dw = _up(ctx, w, n * k)
        var dh = _up(ctx, hht, k * k)
        var dx = _up(ctx, xht, n * k)
        var dp = _up_i(ctx, perm, k)
        var dv = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        ctx.enqueue_function[cd_rows_kernel](
            dw.unsafe_ptr(), dh.unsafe_ptr(), dx.unsafe_ptr(), dp.unsafe_ptr(), dv.unsafe_ptr(), Int32(n), Int32(k),
            grid_dim=_blocks(n), block_dim=TPB,
        )
        _down(ctx, dw, w, n * k)
        _down(ctx, dv, viol, n)
        ctx.synchronize()
        _ = dw^
        _ = dh^
        _ = dx^
        _ = dp^
        _ = dv^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def orth(a: F32Ptr, m: Int, l: Int) raises:
        """Two passes of: R of the matrix (`qr_factor` on a device copy, as
        `device_qr_r`), the rank guard on the host, then A R^-1 by rows
        (`trsm_kernel`); `orth_on_device`. One upload, one download."""
        var ctx = xd_ctx()
        var da = _up(ctx, a, m * l)
        orth_on_device(ctx, da, m, l)
        _down(ctx, da, a, m * l)
        ctx.synchronize()
        _ = da^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def svd(a: F32Ptr, m: Int, n: Int, s: F32Ptr, v: F32Ptr) raises:
        """`device_svdvals`'s route (qr_factor, then svd_of_r) keeping V."""
        DevExec.svd_cells(a, m, n, s, v, QRB_CELLS, J2_CHUNK_CELLS)

    @staticmethod
    def svd_cells(a: F32Ptr, m: Int, n: Int, s: F32Ptr, v: F32Ptr, qr_cells: Int, j2_cells: Int) raises:
        """`svd` with the work per launch named (QR cells, Jacobi pair-column
        cells): dense_check proves the bits do not depend on it."""
        var ctx = xd_ctx()
        var da = _up(ctx, a, m * n)
        var scratch = ctx.enqueue_create_buffer[DType.float32](qr_slice_count(m, n) * n * n)
        var r_buf = ctx.enqueue_create_buffer[DType.float32](n * n)
        var v_buf = ctx.enqueue_create_buffer[DType.float32](n * n)
        var s_buf = ctx.enqueue_create_buffer[DType.float32](n)
        ctx.synchronize()
        # bounded in work per launch and poisoned (x_decomp/qr_bounded.mojo):
        # a launch macOS cut short leaves NaN in R, hence in s, refused below
        _ = qr_factor_bounded(ctx, da, scratch, r_buf, m, n, qr_cells)
        var nan = Float32(0.0) / Float32(0.0)
        enqueue_fill(ctx, s_buf, nan)
        enqueue_fill(ctx, v_buf, nan)
        var done = False
        # MOJOLEARN_XD_PJ_SVD_MIN=n: the round-robin one-sided Jacobi from n
        # up, on every vendor and tier (lane/neural-net-experiment; it was
        # Metal FAST only). Default 0: never. NOT the pinned cyclic order's
        # bits under IDENTICAL: an experiment the digest check reports.
        var lo_svd = pj_svd_min()
        if lo_svd > 0 and n >= lo_svd:
            done = _svd_par_of_r(ctx, r_buf, v_buf, s_buf, n)
        if done:
            pass
        elif jacobi2_on() and n >= J2_BOUNDED_MIN_N:
            # measured (m4pro-b 1790606245923): 0.45x at n = 28, 1.07x at
            # 256, 1.21x at 800 (one launch); from 64 columns the bounded
            # chunk route (lane/lle-timeout), the old one launch below
            _svd2_of_r(ctx, r_buf, v_buf, s_buf, n, j2_cells)
        else:
            svd_of_r(ctx, r_buf, v_buf, s_buf, n, X_DECOMP_SVD_SWEEPS, X_DECOMP_SVD_TOL)
        _down(ctx, s_buf, s, n)
        _down(ctx, v_buf, v, n * n)
        ctx.synchronize()
        # READ BACK WHOLE: every value was poisoned before the solve and is
        # written by a finished one; a NaN left is a launch cut short (or a
        # NaN in the input), refused rather than returned
        for t in range(n):
            if s.unsafe_load(t) != s.unsafe_load(t):
                raise Error("x_decomp svd: singular value " + String(t) + " of " + String(n)
                            + " is NaN after the solve (a device launch cut short, or a NaN input): refused")
        for t in range(n * n):
            if v.unsafe_load(t) != v.unsafe_load(t):
                raise Error("x_decomp svd: V entry " + String(t) + " of " + String(n * n)
                            + " is NaN after the solve (a device launch cut short, or a NaN input): refused")
        _ = da^
        _ = scratch^
        _ = r_buf^
        _ = v_buf^
        _ = s_buf^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def lasso_rows(
        g: F32Ptr, q: F32Ptr, w: F32Ptr, h: F32Ptr, its: F32Ptr, n: Int, k: Int, alpha: Float32,
        max_iter: Int, tol: Float32, positive: Bool,
    ) raises:
        var ctx = xd_ctx()
        var dg = _up(ctx, g, k * k)
        var dq = _up(ctx, q, n * k)
        var dw = _up(ctx, w, n * k)
        var dh = ctx.enqueue_create_buffer[DType.float32](n * k if n * k > 0 else 1)
        var di = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        ctx.enqueue_function[lasso_rows_kernel](
            dg.unsafe_ptr(), dq.unsafe_ptr(), dw.unsafe_ptr(), dh.unsafe_ptr(), di.unsafe_ptr(), Int32(n), Int32(k),
            alpha, Int32(max_iter), tol, Int32(1 if positive else 0), grid_dim=_blocks(n), block_dim=TPB,
        )
        _down(ctx, dw, w, n * k)
        _down(ctx, di, its, n)
        ctx.synchronize()
        _ = dg^
        _ = dq^
        _ = dw^
        _ = dh^
        _ = di^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def omp_rows(g: F32Ptr, q: F32Ptr, w: F32Ptr, s: F32Ptr, na: F32Ptr, n: Int, k: Int, nnz: Int) raises:
        var ctx = xd_ctx()
        var per = k * k + 3 * k
        var dg = _up(ctx, g, k * k)
        var dq = _up(ctx, q, n * k)
        var dw = ctx.enqueue_create_buffer[DType.float32](n * k if n * k > 0 else 1)
        var ds = ctx.enqueue_create_buffer[DType.float32](n * per if n * per > 0 else 1)
        var dn = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        ctx.enqueue_function[omp_rows_kernel](
            dg.unsafe_ptr(), dq.unsafe_ptr(), dw.unsafe_ptr(), ds.unsafe_ptr(), dn.unsafe_ptr(), Int32(n), Int32(k),
            Int32(nnz), grid_dim=_blocks(n), block_dim=TPB,
        )
        _down(ctx, dw, w, n * k)
        _down(ctx, dn, na, n)
        ctx.synchronize()
        _ = dg^
        _ = dq^
        _ = dw^
        _ = ds^
        _ = dn^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def rand_gamma(dst: F32Ptr, count: Int, seed: UInt32, stream: UInt32, shape: Float32) raises:
        var ctx = xd_ctx()
        var dout = ctx.enqueue_create_buffer[DType.float32](count if count > 0 else 1)
        ctx.enqueue_function[gamma_kernel](
            dout.unsafe_ptr(), Int32(count), seed, stream, shape, grid_dim=_blocks(count), block_dim=TPB
        )
        _down(ctx, dout, dst, count)
        ctx.synchronize()
        _ = dout^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def lda_rows(
        x: F32Ptr, ew: F32Ptr, d: F32Ptr, e: F32Ptr, s: F32Ptr, its: F32Ptr, n: Int, k: Int, v: Int,
        prior: Float32, max_iter: Int, tol: Float32,
    ) raises:
        var ctx = xd_ctx()
        var dx = _up(ctx, x, n * v)
        var dw = _up(ctx, ew, k * v)
        var dd = _up(ctx, d, n * k)
        var de = _up(ctx, e, n * k)
        var ds = ctx.enqueue_create_buffer[DType.float32](n * (v + k) if n > 0 else 1)
        var di = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        ctx.enqueue_function[lda_rows_kernel](
            dx.unsafe_ptr(), dw.unsafe_ptr(), dd.unsafe_ptr(), de.unsafe_ptr(), ds.unsafe_ptr(), di.unsafe_ptr(),
            Int32(n), Int32(k), Int32(v), prior, Int32(max_iter), tol, grid_dim=_blocks(n), block_dim=TPB,
        )
        _down(ctx, dd, d, n * k)
        _down(ctx, de, e, n * k)
        _down(ctx, di, its, n)
        ctx.synchronize()
        _ = dx^
        _ = dw^
        _ = dd^
        _ = de^
        _ = ds^
        _ = di^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def dijkstra_rows(w: F32Ptr, dist: F32Ptr, reached: F32Ptr, n: Int) raises:
        # the compressed arcs are built on the host (data movement only),
        # then every row runs on the device, DIJKSTRA_ROWS rows a launch so
        # the heap scratch stays small
        var ne = dijkstra_arc_count(w, n)
        var rp = List[Int32](length=n + 1, fill=Int32(0))
        var adj = List[Int32](length=ne if ne > 0 else 1, fill=Int32(0))
        var wa = List[Float32](length=ne if ne > 0 else 1, fill=Float32(0))
        var wb = List[Float32](length=ne if ne > 0 else 1, fill=Float32(0))
        dijkstra_arcs(
            w, n, I32Ptr(unsafe_from_address=Int(rp.unsafe_ptr())), I32Ptr(unsafe_from_address=Int(adj.unsafe_ptr())),
            F32Ptr(unsafe_from_address=Int(wa.unsafe_ptr())), F32Ptr(unsafe_from_address=Int(wb.unsafe_ptr())),
        )
        var ctx = xd_ctx()
        var drp = _up_i(ctx, I32Ptr(unsafe_from_address=Int(rp.unsafe_ptr())), n + 1)
        var dadj = _up_i(ctx, I32Ptr(unsafe_from_address=Int(adj.unsafe_ptr())), ne)
        var dwa = _up(ctx, F32Ptr(unsafe_from_address=Int(wa.unsafe_ptr())), ne)
        var dwb = _up(ctx, F32Ptr(unsafe_from_address=Int(wb.unsafe_ptr())), ne)
        var dd = ctx.enqueue_create_buffer[DType.float32](n * n if n > 0 else 1)
        var dpos = ctx.enqueue_create_buffer[DType.int32](n * n if n > 0 else 1)
        var chunk = max(1, min(n, DIJKSTRA_ROWS))
        var dh = ctx.enqueue_create_buffer[DType.int32](chunk * n if n > 0 else 1)
        var dr = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        var r0 = 0
        while r0 < n:
            var rows = min(chunk, n - r0)
            ctx.enqueue_function[dijkstra_kernel](
                drp.unsafe_ptr(), dadj.unsafe_ptr(), dwa.unsafe_ptr(), dwb.unsafe_ptr(), dd.unsafe_ptr(),
                dh.unsafe_ptr(), dpos.unsafe_ptr(), dr.unsafe_ptr(), Int32(n), Int32(r0), Int32(rows),
                grid_dim=_blocks(rows), block_dim=TPB,
            )
            r0 += rows
        _down(ctx, dd, dist, n * n)
        _down(ctx, dr, reached, n)
        ctx.synchronize()
        _ = drp^
        _ = dadj^
        _ = dwa^
        _ = dwb^
        _ = dd^
        _ = dpos^
        _ = dh^
        _ = dr^
        ctx.synchronize()
        _ = ctx^
        _ = rp^
        _ = adj^
        _ = wa^
        _ = wb^

    @staticmethod
    def barycenter_rows(
        x: F32Ptr, y: F32Ptr, nbr: F32Ptr, wt: F32Ptr, flags: F32Ptr, n: Int, ny: Int, d: Int, k: Int, reg: Float32
    ) raises:
        var ctx = xd_ctx()
        var dx = _up(ctx, x, n * d)
        var dy = _up(ctx, y, ny * d)
        var dnb = _up(ctx, nbr, n * k)
        var dwt = ctx.enqueue_create_buffer[DType.float32](n * k if n * k > 0 else 1)
        var ds = ctx.enqueue_create_buffer[DType.float32](n * (k * k + k * d) if n > 0 else 1)
        var df = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        ctx.enqueue_function[barycenter_kernel](
            dx.unsafe_ptr(), dy.unsafe_ptr(), dnb.unsafe_ptr(), dwt.unsafe_ptr(), ds.unsafe_ptr(), df.unsafe_ptr(),
            Int32(n), Int32(d), Int32(k), reg, grid_dim=_blocks(n), block_dim=TPB,
        )
        _down(ctx, dwt, wt, n * k)
        _down(ctx, df, flags, n)
        ctx.synchronize()
        _ = dx^
        _ = dy^
        _ = dnb^
        _ = dwt^
        _ = ds^
        _ = df^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def als_rows(c: F32Ptr, y: F32Ptr, yty: F32Ptr, x: F32Ptr, flags: F32Ptr, n: Int, m: Int, f: Int, reg: Float32) raises:
        var ctx = xd_ctx()
        var dc = _up(ctx, c, n * m)
        var dy = _up(ctx, y, m * f)
        var dg = _up(ctx, yty, f * f)
        var dx = ctx.enqueue_create_buffer[DType.float32](n * f if n * f > 0 else 1)
        var ds = ctx.enqueue_create_buffer[DType.float32](n * (f * f + f) if n > 0 else 1)
        var df = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        if n > 0 and f * f + f <= ALS_TEAM_CELLS and String(getenv("MOJOLEARN_XD_ALS_TEAM", "1")) != "0":
            ctx.enqueue_function[als_team_kernel](
                dc.unsafe_ptr(), dy.unsafe_ptr(), dg.unsafe_ptr(), dx.unsafe_ptr(), ds.unsafe_ptr(), df.unsafe_ptr(),
                Int32(n), Int32(m), Int32(f), reg, grid_dim=n, block_dim=ALS_TEAM_TPB,
            )
        else:
            ctx.enqueue_function[als_kernel](
                dc.unsafe_ptr(), dy.unsafe_ptr(), dg.unsafe_ptr(), dx.unsafe_ptr(), ds.unsafe_ptr(), df.unsafe_ptr(),
                Int32(n), Int32(m), Int32(f), reg, grid_dim=_blocks(n), block_dim=TPB,
            )
        _down(ctx, dx, x, n * f)
        _down(ctx, df, flags, n)
        ctx.synchronize()
        _ = dc^
        _ = dy^
        _ = dg^
        _ = dx^
        _ = ds^
        _ = df^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def absmax_sign(a: F32Ptr, dst: F32Ptr, n: Int, d: Int, by_col: Bool) raises:
        var ctx = xd_ctx()
        var da = _up(ctx, a, n * d)
        var cnt = d if by_col else n
        var dout = ctx.enqueue_create_buffer[DType.float32](cnt if cnt > 0 else 1)
        var ns = absmax_scratch(n, d, by_col)
        var dp = ctx.enqueue_create_buffer[DType.float32](ns if ns > 0 else 1)
        launch_absmax(ctx, _p(da), _p(dout), _p(dp), n, d, by_col)
        _down(ctx, dout, dst, cnt)
        ctx.synchronize()
        _ = da^
        _ = dout^
        _ = dp^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def geqrf(a: F32Ptr, tau: F32Ptr, m: Int, n: Int) raises:
        var qrt = List[Int](length=5, fill=0)
        var qrk = List[Int](length=2, fill=0)
        qrk[0] = 1 if String(getenv("MOJOLEARN_XD_QR_TIMING")) == "1" else 0
        if xd_qr_on_host(m):
            # lane neural-pass37: the row-streaming host walk of the same cells
            geqrf_host_rows(a, tau, m, n)
            return
        var ctx = xd_ctx()
        var kk = m if m < n else n
        var da = _up(ctx, a, m * n)
        var dt = ctx.enqueue_create_buffer[DType.float32](kk if kk > 0 else 1)
        var ds = ctx.enqueue_create_buffer[DType.float32](2)
        var dw = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        # step k: the reflector (one thread: its folds ascending), then v scaled
        # (rows), w = v^T A[k:, j] (one thread per column, rows ascending), then
        # A[k:, k+1:] updated (one thread per cell): geqrf_serial's cells, in
        # its order per column, with no host round trip between the steps
        for k in range(kk):
            _qr_tick(ctx, qrt, qrk, -1)
            ctx.enqueue_function[geqrf_head_staged_kernel](
                da.unsafe_ptr(), dt.unsafe_ptr(), ds.unsafe_ptr(), Int32(k), Int32(m), Int32(n), grid_dim=1, block_dim=STAGE_TPB
            )
            _qr_tick(ctx, qrt, qrk, 0)
            if m - k - 1 > 0:
                _qr_tick(ctx, qrt, qrk, -1)
                ctx.enqueue_function[geqrf_scale_kernel](
                    da.unsafe_ptr(), ds.unsafe_ptr(), Int32(k), Int32(m), Int32(n), grid_dim=_blocks(m - k - 1), block_dim=TPB
                )
                _qr_tick(ctx, qrt, qrk, 1)
            if n - k - 1 > 0:
                if xd_qr_dot_tile():
                    _qr_tick(ctx, qrt, qrk, -1)
                    ctx.enqueue_function[geqrf_dot_tile_kernel](
                        da.unsafe_ptr(), ds.unsafe_ptr(), dw.unsafe_ptr(), Int32(k), Int32(m), Int32(n),
                        grid_dim=(n - k - 1 + DOT_COLS - 1) // DOT_COLS, block_dim=DOT_TPB,
                    )
                    _qr_tick(ctx, qrt, qrk, 3)
                else:
                    _qr_tick(ctx, qrt, qrk, -1)
                    ctx.enqueue_function[geqrf_dot_staged_kernel](
                        da.unsafe_ptr(), ds.unsafe_ptr(), dw.unsafe_ptr(), Int32(k), Int32(m), Int32(n),
                        grid_dim=n - k - 1, block_dim=STAGE_TPB,
                    )
                    _qr_tick(ctx, qrt, qrk, 2)
                _qr_tick(ctx, qrt, qrk, -1)
                ctx.enqueue_function[geqrf_update_kernel](
                    da.unsafe_ptr(), dt.unsafe_ptr(), ds.unsafe_ptr(), dw.unsafe_ptr(), Int32(k), Int32(m), Int32(n),
                    grid_dim=_blocks((m - k) * (n - k - 1)), block_dim=TPB,
                )
                _qr_tick(ctx, qrt, qrk, 4)
        _down(ctx, da, a, m * n)
        _down(ctx, dt, tau, kk)
        ctx.synchronize()
        _ = da^
        _ = dt^
        _ = ds^
        _ = dw^
        if qrk[0] == 1:
            print("geqrf geqrf_head_staged_kernel", qrt[0] // 1000000, "ms")
            print("geqrf geqrf_scale_kernel", qrt[1] // 1000000, "ms")
            print("geqrf geqrf_dot_staged_kernel", qrt[2] // 1000000, "ms")
            print("geqrf geqrf_dot_tile_kernel", qrt[3] // 1000000, "ms")
            print("geqrf geqrf_update_kernel", qrt[4] // 1000000, "ms")
        _ = qrt^
        _ = qrk^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def orgqr(h: F32Ptr, tau: F32Ptr, q: F32Ptr, m: Int, n: Int, kk: Int, qc: Int) raises:
        var qrt = List[Int](length=4, fill=0)
        var qrk = List[Int](length=2, fill=0)
        qrk[0] = 1 if String(getenv("MOJOLEARN_XD_QR_TIMING")) == "1" else 0
        if xd_qr_on_host(m):
            orgqr_host_rows(h, tau, q, m, n, kk, qc)
            return
        var ctx = xd_ctx()
        var dh = _up(ctx, h, m * n)
        var dt = _up(ctx, tau, kk if kk > 0 else 1)
        var dq = ctx.enqueue_create_buffer[DType.float32](m * qc if m * qc > 0 else 1)
        var dw = ctx.enqueue_create_buffer[DType.float32](qc if qc > 0 else 1)
        # orgqr_col's cells: e_j, then H_k for k descending (w per column, rows
        # ascending; then every cell of the rows k.. updated)
        _qr_tick(ctx, qrt, qrk, -1)
        ctx.enqueue_function[orgqr_init_kernel](dq.unsafe_ptr(), Int32(m), Int32(qc), grid_dim=_blocks(m * qc), block_dim=TPB)
        _qr_tick(ctx, qrt, qrk, 0)
        for r in range(kk):
            var k = kk - 1 - r
            if qc > 0:
                if xd_qr_dot_tile():
                    _qr_tick(ctx, qrt, qrk, -1)
                    ctx.enqueue_function[orgqr_dot_tile_kernel](
                        dh.unsafe_ptr(), dt.unsafe_ptr(), dq.unsafe_ptr(), dw.unsafe_ptr(), Int32(k), Int32(m), Int32(n), Int32(qc),
                        grid_dim=(qc + DOT_COLS - 1) // DOT_COLS, block_dim=DOT_TPB,
                    )
                    _qr_tick(ctx, qrt, qrk, 2)
                else:
                    _qr_tick(ctx, qrt, qrk, -1)
                    ctx.enqueue_function[orgqr_dot_staged_kernel](
                        dh.unsafe_ptr(), dt.unsafe_ptr(), dq.unsafe_ptr(), dw.unsafe_ptr(), Int32(k), Int32(m), Int32(n), Int32(qc),
                        grid_dim=qc, block_dim=STAGE_TPB,
                    )
                    _qr_tick(ctx, qrt, qrk, 1)
            _qr_tick(ctx, qrt, qrk, -1)
            ctx.enqueue_function[orgqr_update_kernel](
                dh.unsafe_ptr(), dt.unsafe_ptr(), dq.unsafe_ptr(), dw.unsafe_ptr(), Int32(k), Int32(m), Int32(n), Int32(qc),
                grid_dim=_blocks((m - k) * qc), block_dim=TPB,
            )
            _qr_tick(ctx, qrt, qrk, 3)
        _down(ctx, dq, q, m * qc)
        ctx.synchronize()
        _ = dh^
        _ = dt^
        _ = dq^
        _ = dw^
        if qrk[0] == 1:
            print("orgqr orgqr_init_kernel", qrt[0] // 1000000, "ms")
            print("orgqr orgqr_dot_staged_kernel", qrt[1] // 1000000, "ms")
            print("orgqr orgqr_dot_tile_kernel", qrt[2] // 1000000, "ms")
            print("orgqr orgqr_update_kernel", qrt[3] // 1000000, "ms")
        _ = qrt^
        _ = qrk^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def als_cg_rows(c: F32Ptr, y: F32Ptr, yty: F32Ptr, x: F32Ptr, steps: F32Ptr, n: Int, m: Int, f: Int, reg: Float32, cg: Int) raises:
        var ctx = xd_ctx()
        var dc = _up(ctx, c, n * m)
        var dy = _up(ctx, y, m * f)
        var dg = _up(ctx, yty, f * f)
        var dx = _up(ctx, x, n * f)
        var ds = ctx.enqueue_create_buffer[DType.float32](n * 3 * f if n * f > 0 else 1)
        var dstp = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        ctx.enqueue_function[als_cg_kernel](
            dc.unsafe_ptr(), dy.unsafe_ptr(), dg.unsafe_ptr(), dx.unsafe_ptr(), ds.unsafe_ptr(), dstp.unsafe_ptr(),
            Int32(n), Int32(m), Int32(f), reg, Int32(cg), grid_dim=_blocks(n), block_dim=TPB,
        )
        _down(ctx, dx, x, n * f)
        _down(ctx, dstp, steps, n)
        ctx.synchronize()
        _ = dc^
        _ = dy^
        _ = dg^
        _ = dx^
        _ = ds^
        _ = dstp^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def qr_r(a: F32Ptr, m: Int, n: Int, r: F32Ptr) raises:
        var w = List[Float32](capacity=m * n)
        for t in range(m * n):
            w.append(a.unsafe_load(t))
        var got = device_qr_r(xd_ctx(), w, m, n)
        for t in range(n * n):
            r.unsafe_store(t, got[t])

    @staticmethod
    def vendor() -> String:
        return String(COMPILED_VENDOR)

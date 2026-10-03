# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DevExec: the decomp lane's cells on the GPU. One thread per output, the
cell of `x_decomp/cells.mojo` verbatim; the serial routines run on ONE
device thread. Host in, host dst: upload, launch, download."""
from std.sys.compile import is_defined
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.ffi import _Global
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.memory import memcpy
from checks.kernel_matrix import COLUMN_AMD, COLUMN_APPLE, TARGET_COLUMN, lib_smem_page_fits_for

from checks.vendor import COMPILED_VENDOR
from core.householder_qr import qr_factor, qr_slice_count
from decomposition.linalg_public_device import device_qr_r
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL, ftz, identical_div, identical_mul, identical_mul_add
from std.sys.info import has_apple_gpu_accelerator

# lane/apple-fast-decomp-linalg and -sparse (2026-10-02): the FAST + Apple
# guard of every `-D MOJOLEARN_..._FAST_...` switch in this file. Compiled
# only there; IDENTICAL and every other vendor compile main's code unchanged.
comptime XD_FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
from x_decomp.cells import (
    lu_perm_src,
    lu_aux_clamp,
    lu_aux_join,
    lu_aux_val,
    trs_block_col,
    trs_coef,
    trs_divides,
    trs_feed_cell,
    trs_gather,
    knn_select_row,
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
    chol_diag,
    colsum_cell,
    ew_cell,
    gemm_cell,
    als_row,
    als_row_solve,
    add,
    mul,
    als_cg_row,
    barycenter_row,
    dijkstra_arc_count,
    dijkstra_arcs,
    dijkstra_row,
    gamma_cell,
    lasso_row,
    lda_doc_row,
    digamma,
    exp_c,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_EXP,
    OP_LOGS,
    OP_MAX,
    lu_diag,
    lu_l_elem,
    lu_swap_elem,
    lu_update_elem,
    omp_row,
    lars_row,
    LARS_ROW_EXTRA,
    orth_diag_cell,
    orth_rank_guard,
    trsm_row,
    rand_cell,
    rowsum_cell,
    pdist_cell,
    sqdist_cell,
)
from x_decomp.exec_trait import Exec
from x_decomp.jacobi2 import dev_barrier
from x_decomp.qr_bounded import QRB_CELLS, qr_factor_bounded
from x_decomp.fast_chol import CH_FITS, CH_NB, launch_chol_blocked
from x_decomp.fast_gemm import FG_TPB, fast_gemm_on, fg_gemm_tiled_kernel, fg_tiles
from x_decomp.fast_qr import (
    FQ_TPB,
    fast_qr_on,
    fq_dot_blocks,
    fq_geqrf_dot_kernel,
    fq_head_blocks,
    fq_head_finish_kernel,
    fq_head_part_kernel,
    fq_orgqr_dot_kernel,
    geqrf_scale_kernel,
    geqrf_update_kernel,
    orgqr_init_kernel,
    orgqr_update_kernel,
)
from x_decomp.rr import RR_EIGH_SWEEPS, RR_OFF_TPB, rr_converged, rr_fro_kept
from x_decomp.rr_batch import rr_batch_kernel, rrb_cs_len, rrb_part_len
from x_decomp.rr_svd import RS_TPB
from x_decomp.rr_svd_device import rs_norm_kernel, rs_round_kernel
from x_decomp.lle_local import hessian_ncy
from x_decomp.lle_device import (
    lle_apply_kernel,
    hessian_comp_kernel,
    hessian_kernel,
    hessian_q_kernel,
    lle_gram_kernel,
    lle_mean_kernel,
    ltsa_kernel,
    mlle_key_kernel,
    mlle_rows_kernel,
    mlle_unkey_kernel,
    mlle_weights_kernel,
)
from core.fast_radix_sort import fast_radix_sort_pairs_u32, frs_counts_len
from x_decomp.qr_sliced_device import qs_geqrf_device, qs_orgqr_device
from x_decomp.tsqr_device import ts_apply_device, ts_factor_device, ts_free_device, ts_pack_device
from x_decomp.jacobi_par import (
    PJ_TPB,
    eigh_par_cs_kernel,
    eigh_par_off_fold_kernel,
    eigh_par_off_part_kernel,
    eigh_par_update_kernel,
    pj_identity_kernel,
    pj_transpose_kernel,
    sym_from_triangle_kernel,
)
from core.device_zero import enqueue_fill
from decomposition.checks.jacobi_eigh_device import JACOBI_TOL
from decomposition.spectrum_order_device import enqueue_eigh_ascending
from decomposition.impl.linalg.detail.pca import SIGNFLIP_TPB, sign_flip_kernel


#: rounds enqueued between two synchronize() calls
comptime PJ_SYNC_ROUNDS = 512


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


# ---- the random projection's transform (lane gap-nb-maxabs-grp, 2026-10-02)
#: C = A B^T for a tall A (m x k) and a short B (n x k), a block of PJ_THREADS
#: threads per PJ_TM x PJ_TN tile of C: the A and B tiles PJ_KT terms at a time
#: through shared memory (each row of A read once per column tile, coalesced
#: along k, where gemm_kernel's thread per output read its row k words apart),
#: each thread PJ_TM / (PJ_THREADS / PJ_TN) rows of one column.
comptime PJ_TM = 64
comptime PJ_TN = 16
comptime PJ_KT = 32
comptime PJ_LD = PJ_KT + 1
comptime PJ_THREADS = 256
comptime PJ_ROWS = PJ_TM // (PJ_THREADS // PJ_TN)
comptime PJ_BYTES = (PJ_TM + PJ_TN) * PJ_LD * 4
#: the tile's shared page fits the target column (it does on every column at
#: 10,560 bytes; a column where it did not would keep launch_gemm)
comptime PROJECT_TILED = lib_smem_page_fits_for[TARGET_COLUMN, PJ_BYTES]()


@always_inline
def _nonfinite(v: Float32) -> Bool:
    return (bitcast[DType.uint32](v) & UInt32(0x7F800000)) == UInt32(0x7F800000)


def project_kernel(a: F32Ptr, b: F32Ptr, c: F32Ptr, flag: F32Ptr, m: Int32, k: Int32, n: Int32):
    """launch_gemm(ta=False, tb=True)'s WORDS by tiles: each output's chain
    is gemm_part_cell's (p ascending from zero, `ftz(identical_mul_add(ftz(x),
    ftz(y), acc))`), restarted every FOLD_BLOCK terms, and past one block the
    partials folded ascending from zero by `add` (fold_cell); PJ_KT divides
    FOLD_BLOCK, so the restarts fall on tile edges. Any entry of A that is
    not finite (every entry is staged once) sets flag[0] to one: the
    caller's finiteness refusal, on the device."""
    var As = stack_allocation[PJ_TM * PJ_LD, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var Bs = stack_allocation[PJ_TN * PJ_LD, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var M = Int(m)
    var K = Int(k)
    var N = Int(n)
    var ncb = (N + PJ_TN - 1) // PJ_TN
    var row0 = (Int(block_idx.x) // ncb) * PJ_TM
    var col0 = (Int(block_idx.x) % ncb) * PJ_TN
    var tid = Int(thread_idx.x)
    var tx = tid % PJ_TN
    var ty = tid // PJ_TN
    var nb = (K + FOLD_BLOCK - 1) // FOLD_BLOCK
    var acc = SIMD[DType.float32, PJ_ROWS](0)
    var tot = SIMD[DType.float32, PJ_ROWS](0)
    var bad = False
    var k0 = 0
    while k0 < K:
        var kn = min(PJ_KT, K - k0)
        comptime for s in range(PJ_TM * PJ_KT // PJ_THREADS):
            var e = tid + s * PJ_THREADS
            var r = e // PJ_KT
            var kk = e % PJ_KT
            var v = Float32(0)
            if row0 + r < M and kk < kn:
                v = a.unsafe_load((row0 + r) * K + k0 + kk)
                if _nonfinite(v):
                    bad = True
            As[r * PJ_LD + kk] = v
        comptime for s in range(PJ_TN * PJ_KT // PJ_THREADS):
            var e = tid + s * PJ_THREADS
            var r = e // PJ_KT
            var kk = e % PJ_KT
            var v = Float32(0)
            if col0 + r < N and kk < kn:
                v = b.unsafe_load((col0 + r) * K + k0 + kk)
            Bs[r * PJ_LD + kk] = v
        barrier()
        for kk in range(kn):
            var y = ftz(Bs[tx * PJ_LD + kk])
            comptime for r in range(PJ_ROWS):
                acc[r] = ftz(identical_mul_add(ftz(As[(ty + r * (PJ_THREADS // PJ_TN)) * PJ_LD + kk]), y, acc[r]))
        barrier()
        k0 += kn
        if nb > 1 and (k0 % FOLD_BLOCK == 0 or k0 == K):
            comptime for r in range(PJ_ROWS):
                tot[r] = add(tot[r], acc[r])
                acc[r] = Float32(0)
    if bad:
        flag.unsafe_store(0, Float32(1))
    var col = col0 + tx
    if col < N:
        comptime for r in range(PJ_ROWS):
            var row = row0 + ty + r * (PJ_THREADS // PJ_TN)
            if row < M:
                c.unsafe_store(row * N + col, tot[r] if nb > 1 else acc[r])


def launch_project(ctx: DeviceContext, a: F32Ptr, b: F32Ptr, c: F32Ptr, flag: F32Ptr, m: Int, k: Int, n: Int) raises:
    """project_kernel over every tile of C, enqueued (no sync); flag[0] must
    be zero before it runs."""
    comptime if PROJECT_TILED:
        var tiles = ((m + PJ_TM - 1) // PJ_TM) * ((n + PJ_TN - 1) // PJ_TN)
        ctx.enqueue_function[project_kernel](
            a, b, c, flag, Int32(m), Int32(k), Int32(n), grid_dim=tiles, block_dim=PJ_THREADS,
        )
    else:
        raise Error("x_decomp: the projection tile does not fit this column's shared memory")


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


def lu_info_init_kernel(info: F32Ptr):
    if block_idx.x == 0 and thread_idx.x == 0:
        info.unsafe_store(0, Float32(0))


comptime LU_PIVOT_TPB = 256
#: most blocks of the pivot search, and rows a thread scans before another
#: block is added
comptime LU_PIV_MAXB = 64
comptime LU_PIV_ROWS = 8
#: floats of the LU step scratch `scal`: [d, acts] then the pivot search's
#: partial values and rows (as floats: exact below 2^24 rows)
comptime LU_SCAL_LEN = 2 + 2 * LU_PIV_MAXB


def lu_pivot_blocks(k: Int, n: Int) -> Int:
    """Blocks of step k's pivot search: one per LU_PIVOT_TPB * LU_PIV_ROWS
    rows below the diagonal, at least one, at most LU_PIV_MAXB."""
    var rows = n - k - 1
    var g = (rows + LU_PIVOT_TPB * LU_PIV_ROWS - 1) // (LU_PIVOT_TPB * LU_PIV_ROWS)
    return max(1, min(g, LU_PIV_MAXB))


def lu_pivot_part_kernel(a: F32Ptr, scal: F32Ptr, k: Int32, n: Int32, g: Int32):
    """`lu_pivot` over every block (cpu-gpu-cleanup c-decomp, 2026-10-02;
    was ONE block): the largest |a[i, k]| for i >= k, ties to the LOWEST
    row. Comparisons only, so the result is the serial scan's by
    construction: every thread starts from (|a[k, k]|, k) and takes a later
    row only on a STRICT greater value, exactly as the serial scan does, and
    every combine (here and in `lu_pivot_fin_kernel`) prefers the greater
    value and, on equal values, the lower row. A NaN never wins a strict
    compare, so it is skipped as the serial scan skips it; a NaN at row k
    makes every compare false and keeps row k, as the serial scan does.
    Block b's best goes to scal[2 + b] (value) and scal[2 + LU_PIV_MAXB + b]
    (row)."""
    var kk = Int(k)
    var nn = Int(n)
    var gg = Int(g)
    var rv = stack_allocation[
        LU_PIVOT_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var ri = stack_allocation[
        LU_PIVOT_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var best = abs(ftz(a.unsafe_load(kk * nn + kk)))
    var p = kk
    var i = kk + 1 + b * LU_PIVOT_TPB + tid
    while i < nn:
        var v = abs(ftz(a.unsafe_load(i * nn + kk)))
        if v > best:
            best = v
            p = i
        i += gg * LU_PIVOT_TPB
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
        scal.unsafe_store(2 + b, rv.unsafe_load(0))
        scal.unsafe_store(2 + LU_PIV_MAXB + b, Float32(Int(ri.unsafe_load(0))))


def lu_pivot_fin_kernel(scal: F32Ptr, piv: I32Ptr, k: Int32, g: Int32):
    """The g block partials of `lu_pivot_part_kernel` combined by the same
    rule (greater value, then lower row) into piv[k]."""
    var rv = stack_allocation[
        LU_PIVOT_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var ri = stack_allocation[
        LU_PIVOT_TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    var gg = Int(g)
    var cv = Float32(0)
    var ci = Int32(-1)
    var b = tid
    while b < gg:
        var ov = scal.unsafe_load(2 + b)
        var oi = Int32(Int(scal.unsafe_load(2 + LU_PIV_MAXB + b)))
        if ci < 0 or ov > cv or (ov == cv and oi < ci):
            cv = ov
            ci = oi
        b += LU_PIVOT_TPB
    rv.unsafe_store(tid, cv)
    ri.unsafe_store(tid, ci)
    barrier()
    var active = LU_PIVOT_TPB // 2
    while active > 0:
        if tid < active:
            var ov = rv.unsafe_load(tid + active)
            var oi = ri.unsafe_load(tid + active)
            var c2 = rv.unsafe_load(tid)
            var i2 = ri.unsafe_load(tid)
            if oi >= 0 and (i2 < 0 or ov > c2 or (ov == c2 and oi < i2)):
                rv.unsafe_store(tid, ov)
                ri.unsafe_store(tid, oi)
        barrier()
        active = active // 2
    if tid == 0:
        piv.unsafe_store(Int(k), ri.unsafe_load(0))


# lane/apple-fast-decomp-linalg (2026-10-02, pass 3): -D MOJOLEARN_LU_FAST_PIVOT_GRID,
# FAST on Apple only. main's pivot search is already two grid launches
# (`lu_pivot_part_kernel` over up to LU_PIV_MAXB blocks, `lu_pivot_fin_kernel`
# over their partials) and its swap a third (`lu_swap_cols_kernel`, with
# step k's diag and act folded in). Here the finish and the swap are ONE
# grid launch: every block of the swap grid re-derives the winner from the
# <= LU_PIV_MAXB partials in threadgroup memory (the same compare and tie
# rule as `lu_pivot_fin_kernel`: the largest |value|, the lowest row among
# equals, so the same pivot), block 0's thread 0 records it in piv[k], and
# each thread swaps its column of rows k and the winner; column k's thread
# then runs `lu_diag` and act[k]. Two launches a column instead of three
# (8192: 8,192 fewer), no one-block launch over a runtime size, no host
# step. The trailing columns' swaps read piv[k] later, as before.
comptime LU_FAST_PIVOT_GRID = XD_FAST_APPLE and is_defined["MOJOLEARN_LU_FAST_PIVOT_GRID"]()


def lu_pivot_swap_kernel(
    a: F32Ptr, piv: I32Ptr, info: F32Ptr, scal: F32Ptr, act: F32Ptr, k: Int32, n: Int32, col_lo: Int32, col_hi: Int32, g: Int32
):
    """`lu_pivot_fin_kernel` in every block, then `lu_swap_cols_kernel`'s
    work with the winner from threadgroup memory (see above). TPB >=
    LU_PIV_MAXB, so each thread folds at most one partial."""
    var rv = stack_allocation[TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var ri = stack_allocation[TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var gg = Int(g)
    var cv = Float32(0)
    var ci = Int32(-1)
    var b = tid
    while b < gg:
        var ov = scal.unsafe_load(2 + b)
        var oi = Int32(Int(scal.unsafe_load(2 + LU_PIV_MAXB + b)))
        if ci < 0 or ov > cv or (ov == cv and oi < ci):
            cv = ov
            ci = oi
        b += TPB
    rv.unsafe_store(tid, cv)
    ri.unsafe_store(tid, ci)
    barrier()
    var active = TPB // 2
    while active > 0:
        if tid < active:
            var ov = rv.unsafe_load(tid + active)
            var oi = ri.unsafe_load(tid + active)
            var c2 = rv.unsafe_load(tid)
            var i2 = ri.unsafe_load(tid)
            if oi >= 0 and (i2 < 0 or ov > c2 or (ov == c2 and oi < i2)):
                rv.unsafe_store(tid, ov)
                ri.unsafe_store(tid, oi)
        barrier()
        active = active // 2
    var kk = Int(k)
    var nn = Int(n)
    var p = Int(ri.unsafe_load(0))
    if block_idx.x == 0 and tid == 0:
        piv.unsafe_store(kk, Int32(p))
    var j = Int(col_lo) + Int(block_idx.x) * Int(block_dim.x) + tid
    if j < Int(col_hi):
        if p != kk:
            var t = a.unsafe_load(kk * nn + j)
            a.unsafe_store(kk * nn + j, a.unsafe_load(p * nn + j))
            a.unsafe_store(p * nn + j, t)
        if j == kk:
            lu_diag(a, info, scal, kk, nn)
            act.unsafe_store(kk, scal.unsafe_load(1))


def enqueue_lu_pivot(ctx: DeviceContext, a: F32Ptr, piv: I32Ptr, scal: F32Ptr, k: Int, n: Int) raises:
    """Step k's pivot row into piv[k]: the block partials, then their combine."""
    var g = lu_pivot_blocks(k, n)
    ctx.enqueue_function[lu_pivot_part_kernel](
        a, scal, Int32(k), Int32(n), Int32(g), grid_dim=g, block_dim=LU_PIVOT_TPB
    )
    ctx.enqueue_function[lu_pivot_fin_kernel](scal, piv, Int32(k), Int32(g), grid_dim=1, block_dim=LU_PIVOT_TPB)


def lu_swap_kernel(a: F32Ptr, piv: I32Ptr, info: F32Ptr, scal: F32Ptr, act: F32Ptr, k: Int32, n: Int32):
    """`lu_swap_elem` for every column; column k's thread then runs step
    k's `lu_diag` (and act[k] = scal[1]) on the swapped pivot: the cell no
    other thread of the launch touches (cpu-gpu-cleanup c-decomp: was its
    own one-thread launch)."""
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j < Int(n):
        lu_swap_elem(a, piv, Int(k), j, Int(n))
        if j == Int(k):
            lu_diag(a, info, scal, Int(k), Int(n))
            act.unsafe_store(Int(k), scal.unsafe_load(1))


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
    """The blocked LU's panel width, LU_PANEL_NB (cgr-decomp: the
    MOJOLEARN_XD_LU_PANEL A/B switch is deleted)."""
    return LU_PANEL_NB


def lu_swap_cols_kernel(
    a: F32Ptr, piv: I32Ptr, info: F32Ptr, scal: F32Ptr, act: F32Ptr, k: Int32, n: Int32, col_lo: Int32, col_hi: Int32
):
    """`lu_swap_elem` over the columns [col_lo, col_hi) of rows k and
    piv[k]: the panel's steps swap the panel's and the left columns at once,
    the trailing columns later, in the same order (`lu_apply_swaps_kernel`).
    Column k's thread then runs step k's `lu_diag` and act[k] = scal[1]
    (cpu-gpu-cleanup c-decomp: were two one-thread launches)."""
    var j = Int(col_lo) + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j < Int(col_hi):
        lu_swap_elem(a, piv, Int(k), j, Int(n))
        if j == Int(k):
            lu_diag(a, info, scal, Int(k), Int(n))
            act.unsafe_store(Int(k), scal.unsafe_load(1))


def lu_update_panel_kernel(a: F32Ptr, scal: F32Ptr, k: Int32, n: Int32, col_hi: Int32):
    """`lu_update_elem` for the cells (i, j), k < i < n, k < j < col_hi: step
    k's update restricted to the panel's columns."""
    var kk = Int(k)
    var w = Int(col_hi) - kk - 1
    var h = Int(n) - kk - 1
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if w > 0 and h > 0 and t < h * w:
        lu_update_elem(a, scal, kk, kk + 1 + t // w, kk + 1 + t % w, Int(n))


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



# lane/neural-pass114 (2026-10-02): the trailing update with 4 x 4 cells a
# thread. `lu_trail_tiled_kernel` gives each thread ONE cell of a 16 x 16
# tile, so every staged operand feeds one multiply-add per thread (M4,
# lu_factor at 8192: 7.0 s, ~50 GFLOP/s). Here a block owns 64 x 64 cells
# and each thread 4 x 4 of them (rows r + 16a, columns c + 16b): every
# cell's chain is unchanged (the panel's steps in order, the same
# statement, a step with act = 0 skipped), so every word is the same.
# A 16 KB page (fits gate; the one-cell kernel otherwise);
# `MOJOLEARN_XD_LU_TRAIL_R4=0` restores it.
comptime LUR_R = 4
comptime LUR_T = LU_TILE * LUR_R
comptime LUR_BYTES = (2 * LUR_T * LU_PANEL_NB + LU_PANEL_NB) * 4
comptime LU_TRAIL_R4 = lib_smem_page_fits_for[TARGET_COLUMN, LUR_BYTES]()


def lu_trail_r4_kernel(a: F32Ptr, act: F32Ptr, k0: Int32, k1: Int32, n: Int32, nb: Int32):
    var nn = Int(n)
    var kk0 = Int(k0)
    var kk1 = Int(k1)
    var width = Int(nb)
    var tid = Int(thread_idx.x)
    var r = tid // LU_TILE
    var c = tid - r * LU_TILE
    var i0 = kk1 + Int(block_idx.y) * LUR_T
    var j0 = kk1 + Int(block_idx.x) * LUR_T
    var ls = stack_allocation[LUR_T * LU_PANEL_NB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var us = stack_allocation[LU_PANEL_NB * LUR_T, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var acts = stack_allocation[LU_PANEL_NB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    # the L slab: LUR_T rows x width (unflushed, as the item reads l)
    var q = tid
    while q < LUR_T * width:
        var rr = q // width
        var cc = q - rr * width
        var v = Float32(0)
        if i0 + rr < nn:
            v = a.unsafe_load((i0 + rr) * nn + kk0 + cc)
        ls[q] = v
        q += LU_TILE_TPB
    # the U slab: width rows x LUR_T columns (flushed, as the item reads u)
    q = tid
    while q < width * LUR_T:
        var rr = q // LUR_T
        var cc = q - rr * LUR_T
        var v = Float32(0)
        if j0 + cc < nn:
            v = ftz(a.unsafe_load((kk0 + rr) * nn + j0 + cc))
        us[q] = v
        q += LU_TILE_TPB
    if tid < width:
        acts[tid] = act.unsafe_load(kk0 + tid)
    barrier()
    var acc = SIMD[DType.float32, LUR_R * LUR_R](0)
    comptime for qa in range(LUR_R):
        comptime for qb in range(LUR_R):
            var i = i0 + r + qa * LU_TILE
            var j = j0 + c + qb * LU_TILE
            if i < nn and j < nn:
                acc[qa * LUR_R + qb] = ftz(a.unsafe_load(i * nn + j))
    for kp in range(width):
        if acts[kp] != Float32(0):
            var lv = SIMD[DType.float32, LUR_R]()
            var uv = SIMD[DType.float32, LUR_R]()
            comptime for qa in range(LUR_R):
                lv[qa] = ls[(r + qa * LU_TILE) * width + kp]
            comptime for qb in range(LUR_R):
                uv[qb] = us[kp * LUR_T + c + qb * LU_TILE]
            comptime for qa in range(LUR_R):
                comptime for qb in range(LUR_R):
                    acc[qa * LUR_R + qb] = ftz(identical_mul_add(-lv[qa], uv[qb], ftz(acc[qa * LUR_R + qb])))
    comptime for qa in range(LUR_R):
        comptime for qb in range(LUR_R):
            var i = i0 + r + qa * LU_TILE
            var j = j0 + c + qb * LU_TILE
            if i < nn and j < nn:
                a.unsafe_store(i * nn + j, acc[qa * LUR_R + qb])


# ---- the panel LU trailing kernels (lane neural-pass135, 2026-10-02) -----------------------------
#: The register-blocked trailing kernel: LU_RB x LU_RB cells a thread at
#: stride LU_TILE, so a block of LU_TILE x LU_TILE threads owns an
#: LU_RB_TILE x LU_RB_TILE tile; the L slab (LU_RB_TILE rows x NB, stored
#: k-major) and the U slab (NB x LU_RB_TILE columns) in threadgroup memory.
comptime LU_RB = 4
comptime LU_RB_TILE = LU_TILE * LU_RB
comptime LU_RB_SMEM_BYTES = (2 * LU_RB_TILE * LU_PANEL_NB + LU_PANEL_NB) * 4
comptime LU_RB_FITS = lib_smem_page_fits_for[TARGET_COLUMN, LU_RB_SMEM_BYTES]()


def lu_swaps_trsm_on() -> Bool:
    """Whether the trailing columns' swaps and U rows run as ONE
    `lu_swaps_trsm_kernel` launch (lane neural-pass135): MOJOLEARN_XD_LU_PANEL1=0
    restored `lu_apply_swaps_kernel` + `lu_trsm_kernel` (the A/B arm,
    deleted by cgr-decomp). (np135's one-block `lu_panel1_kernel` is not on
    main: a new one-block launch, refused by tools/hooks/no_host_routes.py.)"""
    return True


def lu_trail_rb_on() -> Bool:
    """Whether the trailing update runs `lu_trail_rb_kernel` (MOJOLEARN_XD_LU_
    TRAIL_RB=0, or MOJOLEARN_XD_LU_PANEL1=0, keeps `lu_trail_tiled_kernel`;
    a column whose shared limit cannot hold its page keeps it too)."""
    comptime if not LU_RB_FITS:
        return False
    return True


def lu_swaps_trsm_kernel(a: F32Ptr, piv: I32Ptr, act: F32Ptr, k0: Int32, k1: Int32, n: Int32):
    """One thread per trailing column j >= k1: `lu_apply_swaps_kernel`'s
    swaps, then `lu_trsm_kernel`'s U rows, in one launch. Both touch column
    j only (the multipliers a[k, k'] they read are the panel's, final), so
    a column's statements and their order are the two kernels' exactly."""
    var nn = Int(n)
    var j = Int(k1) + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j < nn:
        for k in range(Int(k0), Int(k1)):
            lu_swap_elem(a, piv, k, j, nn)
        for k in range(Int(k0) + 1, Int(k1)):
            var acc = ftz(a.unsafe_load(k * nn + j))
            for kp in range(Int(k0), k):
                if act.unsafe_load(kp) != Float32(0):
                    var l = a.unsafe_load(k * nn + kp)
                    acc = ftz(identical_mul_add(-l, ftz(a.unsafe_load(kp * nn + j)), ftz(acc)))
            a.unsafe_store(k * nn + j, acc)


def lu_trail_rb_kernel(a: F32Ptr, act: F32Ptr, k0: Int32, k1: Int32, n: Int32, nb: Int32):
    """`lu_trail_tiled_kernel` register-blocked: each thread owns LU_RB x
    LU_RB cells (rows i0 + ty + LU_TILE r, columns j0 + tx + LU_TILE c),
    every cell the panel's steps k' = k0 .. k1 - 1 in order, each
    `lu_update_elem`'s statement with l = a[i, k'] (unflushed) and
    u = ftz(a[k', j]), a step with act[k'] = 0 skipped: the same chain per
    cell as the tiled kernel and the serial loop, with LU_RB^2 fused
    multiply-adds per 2 LU_RB threadgroup loads instead of 1 per 2."""
    var nn = Int(n)
    var kk0 = Int(k0)
    var kk1 = Int(k1)
    var width = Int(nb)
    var tid = Int(thread_idx.x)
    var ty = tid // LU_TILE
    var tx = tid - ty * LU_TILE
    var i0 = kk1 + Int(block_idx.y) * LU_RB_TILE
    var j0 = kk1 + Int(block_idx.x) * LU_RB_TILE
    var ls = stack_allocation[LU_PANEL_NB * LU_RB_TILE, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var us = stack_allocation[LU_PANEL_NB * LU_RB_TILE, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var acts = stack_allocation[LU_PANEL_NB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    # the L slab, k-major: ls[k' * LU_RB_TILE + row] (unflushed, as the item reads l)
    var q = tid
    while q < LU_RB_TILE * width:
        var rr = q // width
        var cc = q - rr * width
        var v = Float32(0)
        if i0 + rr < nn:
            v = a.unsafe_load((i0 + rr) * nn + kk0 + cc)
        ls[cc * LU_RB_TILE + rr] = v
        q += LU_TILE_TPB
    # the U slab: us[k' * LU_RB_TILE + col] (flushed, as the item reads u)
    q = tid
    while q < width * LU_RB_TILE:
        var rr = q // LU_RB_TILE
        var cc = q - rr * LU_RB_TILE
        var v = Float32(0)
        if j0 + cc < nn:
            v = ftz(a.unsafe_load((kk0 + rr) * nn + j0 + cc))
        us[q] = v
        q += LU_TILE_TPB
    if tid < width:
        acts[tid] = act.unsafe_load(kk0 + tid)
    barrier()
    var acc = InlineArray[Float32, LU_RB * LU_RB](fill=Float32(0))
    comptime for r in range(LU_RB):
        comptime for c in range(LU_RB):
            var i = i0 + ty + LU_TILE * r
            var j = j0 + tx + LU_TILE * c
            if i < nn and j < nn:
                acc[r * LU_RB + c] = ftz(a.unsafe_load(i * nn + j))
    for kp in range(width):
        if acts[kp] != Float32(0):
            var lv = InlineArray[Float32, LU_RB](fill=Float32(0))
            var uv = InlineArray[Float32, LU_RB](fill=Float32(0))
            comptime for r in range(LU_RB):
                lv[r] = ls[kp * LU_RB_TILE + ty + LU_TILE * r]
            comptime for c in range(LU_RB):
                uv[c] = us[kp * LU_RB_TILE + tx + LU_TILE * c]
            comptime for r in range(LU_RB):
                comptime for c in range(LU_RB):
                    acc[r * LU_RB + c] = ftz(identical_mul_add(-lv[r], uv[c], ftz(acc[r * LU_RB + c])))
    comptime for r in range(LU_RB):
        comptime for c in range(LU_RB):
            var i = i0 + ty + LU_TILE * r
            var j = j0 + tx + LU_TILE * c
            if i < nn and j < nn:
                a.unsafe_store(i * nn + j, acc[r * LU_RB + c])


# lane/apple-fast-decomp-linalg (2026-10-02, pass 2): the blocked
# right-looking Cholesky of x_decomp/fast_chol.mojo, FAST on Apple only,
# behind -D MOJOLEARN_CHOL_FAST_BLOCKED (IDENTICAL and every other column
# compile the column driver unchanged). Cause, for the kit's `chol`: the
# column driver is 2n launches, column j's cells each a j-long serial chain
# per thread. The board's cholesky lane runs cholesky/checks/potrf.mojo
# (the gp binding), which launches the same module under its own guard.
comptime CHOL_FAST_BLOCKED = XD_FAST_APPLE and is_defined["MOJOLEARN_CHOL_FAST_BLOCKED"]() and CH_FITS


def chol_step_kernel(a: F32Ptr, info: F32Ptr, j: Int32, n: Int32):
    """Column step j of `chol_serial`, one thread per row i >= j. Row j's
    thread is `chol_diag` (the info store, a[j, j] = sqrt(acc)); every row
    i > j forms the same pivot chain itself (the same fma chain, so the same
    d that `chol_diag` stores) and runs `chol_col_elem`'s statements with
    it. Reads only columns 0..j-1, final since their own steps; a[j, j] is
    written by row j's thread and read by none."""
    var jj = Int(j)
    var nn = Int(n)
    var i = jj + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= nn:
        return
    if i == jj:
        chol_diag(a, info, jj, nn)
        return
    var acc = ftz(a.unsafe_load(jj * nn + jj))
    for p in range(jj):
        var l = ftz(a.unsafe_load(jj * nn + p))
        acc = ftz(identical_mul_add(-l, l, acc))
    if not (acc > Float32(0)):
        acc = Float32(1)
    var d = sqrt0(acc)
    var s = ftz(a.unsafe_load(i * nn + jj))
    for p in range(jj):
        s = ftz(identical_mul_add(-ftz(a.unsafe_load(i * nn + p)), ftz(a.unsafe_load(jj * nn + p)), s))
    a.unsafe_store(i * nn + jj, div0(s, d))
    a.unsafe_store(jj * nn + i, Float32(0))


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


def orth_diag_kernel(r: F32Ptr, diag: F32Ptr, l: Int32):
    """diag[j] *= R[j, j] (after the rank guard, so a dependent column leaves
    0): one thread per column, the cell `orth_diag_cell`."""
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j < Int(l):
        orth_diag_cell(r, diag, j, Int(l))


def lasso_rows_kernel(
    g: F32Ptr, q: F32Ptr, w: F32Ptr, h: F32Ptr, its: F32Ptr, n: Int32, k: Int32, alpha: Float32,
    max_iter: Int32, tol: Float32, positive: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        its.unsafe_store(i, lasso_row(g, q, w, h, i, Int(k), alpha, Int(max_iter), tol, positive != 0))


def lars_rows_kernel(g: F32Ptr, q: F32Ptr, w: F32Ptr, s: F32Ptr, na: F32Ptr, n: Int32, k: Int32, m: Int32, nnz: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        na.unsafe_store(i, lars_row(g, q, w, s, i, Int(k), Int(m), Int(nnz)))


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


def als_cg_kernel(
    c: F32Ptr, y: F32Ptr, yty: F32Ptr, x: F32Ptr, s: F32Ptr, steps: F32Ptr, n: Int32, m: Int32, f: Int32, reg: Float32,
    cg: Int32,
):
    var u = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if u < Int(n):
        steps.unsafe_store(u, als_cg_row(c, y, yty, x, s, u, Int(m), Int(f), reg, Int(cg)))


# ---- ALS and LDA, one block per row (lane gap-lda-als, 2026-10-02)
#
# The board's als (factors=64) ran `als_kernel`, ONE THREAD per user/item:
# f*f + f = 4160 cells is past the team kernel's 4096-cell cap, so every row
# fell back to the serial accumulation, and the item half-sweep (4096 text
# columns over 86,626 users) timed out on every vendor. `als_block_kernel`
# is one block per row for any f: the cells accumulate in threadgroup memory
# when they fit (ALS_SH_CELLS, f <= 64: 16,640 B), else in the row's device
# scratch, each cell's sequence exactly `als_row`'s; the Cholesky runs column
# by column with the rows below the diagonal in parallel (each element's sum
# exactly `als_row_solve`'s, p ascending), the two triangular solves on one
# thread. C is read through strides (su, si), so the item half-sweep reads
# the user x item matrix in place (no transpose). Same bits as `als_row`.
comptime ALS_BLK_TPB = 256
#: Largest f*f + f kept in threadgroup memory (f <= 64): 16,640 B, the fits
#: gate of the threadgroup page (Apple allows 32 KB).
comptime ALS_SH_CELLS = 4160

comptime ShF32 = UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]


@always_inline
def _ald[SH: Bool](sh: ShF32, g: F32Ptr, o: Int, q: Int) -> Float32:
    comptime if SH:
        return sh[q]
    else:
        return g.unsafe_load(o + q)


@always_inline
def _ast[SH: Bool](sh: ShF32, g: F32Ptr, o: Int, q: Int, v: Float32):
    comptime if SH:
        sh[q] = v
    else:
        g.unsafe_store(o + q, v)


def als_block_kernel[SH: Bool](
    c: F32Ptr, y: F32Ptr, yty: F32Ptr, x: F32Ptr, s: F32Ptr, flags: F32Ptr, n: Int32, m: Int32, f: Int32,
    su: Int32, si: Int32, reg: Float32,
):
    """`als_row` for row u = block_idx.x of C (element (u, i) at c[u * su +
    i * si]), by a block of ALS_BLK_TPB threads; SH: the f*f + f cells in
    threadgroup memory, else at s[u * (f*f + f)]."""
    var u = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var ff = Int(f)
    var mm = Int(m)
    var cells = ff * ff + ff
    var o = u * cells
    var sh = stack_allocation[ALS_SH_CELLS if SH else 1, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var bad = stack_allocation[1, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var cc = tid
    while cc < cells:
        if cc < ff * ff:
            var v = yty.unsafe_load(cc)
            if cc // ff == cc % ff:
                v = add(v, reg)
            _ast[SH](sh, s, o, cc, v)
        else:
            _ast[SH](sh, s, o, cc, Float32(0))
        cc += ALS_BLK_TPB
    var cb = u * Int(su)
    var cs = Int(si)
    for i in range(mm):
        var conf = ftz(c.unsafe_load(cb + i * cs))
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
                _ast[SH](sh, s, o, cc, ftz(identical_mul_add(t, ftz(y.unsafe_load(i * ff + l)), ftz(_ald[SH](sh, s, o, cc)))))
            elif pos:
                var j = cc - ff * ff
                _ast[SH](sh, s, o, cc, ftz(identical_mul_add(conf, ftz(y.unsafe_load(i * ff + j)), ftz(_ald[SH](sh, s, o, cc)))))
            cc += ALS_BLK_TPB
    if tid == 0:
        bad[0] = Float32(0)
    dev_barrier()
    # Cholesky (lower, left-looking): the diagonal on thread 0, then the rows
    # below it in parallel; every element's sum p ascending, as als_row_solve
    for j in range(ff):
        if tid == 0:
            var acc = ftz(_ald[SH](sh, s, o, j * ff + j))
            for p in range(j):
                var l = ftz(_ald[SH](sh, s, o, j * ff + p))
                acc = ftz(identical_mul_add(-l, l, acc))
            if not (acc > Float32(0)):
                bad[0] = Float32(1)
            else:
                _ast[SH](sh, s, o, j * ff + j, sqrt0(acc))
        dev_barrier()
        if bad[0] != Float32(0):
            break
        var dj = _ald[SH](sh, s, o, j * ff + j)
        var r = j + 1 + tid
        while r < ff:
            var acc2 = ftz(_ald[SH](sh, s, o, r * ff + j))
            for p in range(j):
                acc2 = ftz(identical_mul_add(-ftz(_ald[SH](sh, s, o, r * ff + p)), ftz(_ald[SH](sh, s, o, j * ff + p)), acc2))
            _ast[SH](sh, s, o, r * ff + j, div0(acc2, dj))
            r += ALS_BLK_TPB
        dev_barrier()
    if tid != 0:
        return
    if bad[0] != Float32(0):
        for q in range(ff):
            x.unsafe_store(u * ff + q, Float32(0))
        flags.unsafe_store(u, Float32(1))
        return
    var bb = ff * ff
    for a in range(ff):
        var acc = ftz(_ald[SH](sh, s, o, bb + a))
        for p in range(a):
            acc = ftz(identical_mul_add(-ftz(_ald[SH](sh, s, o, a * ff + p)), ftz(_ald[SH](sh, s, o, bb + p)), acc))
        _ast[SH](sh, s, o, bb + a, div0(acc, _ald[SH](sh, s, o, a * ff + a)))
    for aa in range(ff):
        var a = ff - 1 - aa
        var acc = ftz(_ald[SH](sh, s, o, bb + a))
        for p in range(a + 1, ff):
            acc = ftz(identical_mul_add(-ftz(_ald[SH](sh, s, o, p * ff + a)), ftz(_ald[SH](sh, s, o, bb + p)), acc))
        _ast[SH](sh, s, o, bb + a, div0(acc, _ald[SH](sh, s, o, a * ff + a)))
    for q in range(ff):
        x.unsafe_store(u * ff + q, _ald[SH](sh, s, o, bb + q))
    flags.unsafe_store(u, Float32(0))


def als_scratch(n: Int, f: Int) -> Int:
    """Device scratch floats `launch_als_rows` needs: none when the cells fit
    threadgroup memory, else f*f + f per row."""
    var cells = f * f + f
    return 0 if cells <= ALS_SH_CELLS else n * cells


def launch_als_rows(
    ctx: DeviceContext, c: F32Ptr, y: F32Ptr, yty: F32Ptr, x: F32Ptr, s: F32Ptr, flags: F32Ptr, n: Int, m: Int,
    f: Int, su: Int, si: Int, reg: Float32,
) raises:
    """Every row's `als_row`, one block per row (`als_block_kernel`).
    (cgr-decomp: the MOJOLEARN_XD_ALS_BLOCK=0 A/B arm is deleted.)"""
    if n <= 0:
        return
    if f * f + f <= ALS_SH_CELLS:
        comptime k_sh = als_block_kernel[True]
        ctx.enqueue_function[k_sh](
            c, y, yty, x, s, flags, Int32(n), Int32(m), Int32(f), Int32(su), Int32(si), reg,
            grid_dim=n, block_dim=ALS_BLK_TPB,
        )
    else:
        comptime k_g = als_block_kernel[False]
        ctx.enqueue_function[k_g](
            c, y, yty, x, s, flags, Int32(n), Int32(m), Int32(f), Int32(su), Int32(si), reg,
            grid_dim=n, block_dim=ALS_BLK_TPB,
        )


# The board's lda/text (86,626 documents x 4,096 bins) ran `lda_rows_kernel`,
# ONE THREAD per document scanning all v words twice per inner iteration
# (550 s on MI325X, the race ceiling on Apple). `lda_block_kernel` is one
# block per document: the document's nonzero words (ftz(x) != 0, ascending)
# compacted once into threadgroup memory, norm_phi per word in parallel (its
# topic fold ascending, as `lda_doc_row`), each topic's word fold on its own
# thread over the compacted words ascending (the cell's skip of zero counts,
# so the same sequence), the total, digamma and change on thread 0 in topic
# order. Same bits as `lda_doc_row`. A document past LDA_NNZ_CAP nonzeros
# keeps the per-word scan through its device scratch row (same sequence).
comptime LDA_BLK_TPB = 128
#: Nonzero words kept in threadgroup memory (index + norm_phi ratio: 16 KB).
comptime LDA_NNZ_CAP = 2048
#: Largest n_components the block kernel carries (3 x 1 KB of topic state).
comptime LDA_K_CAP = 256


def lda_block_kernel(
    x: F32Ptr, ew: F32Ptr, d: F32Ptr, e: F32Ptr, s: F32Ptr, its: F32Ptr, n: Int32, k: Int32, v: Int32,
    prior: Float32, max_iter: Int32, tol: Float32,
):
    var i = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var kk = Int(k)
    var vv = Int(v)
    var base = i * kk
    var sb = i * (vv + kk)
    var xb = i * vv
    var eps = Float32(2.220446049250313e-16)
    var idx = stack_allocation[LDA_NNZ_CAP, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var sw = stack_allocation[LDA_NNZ_CAP, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var ds = stack_allocation[LDA_K_CAP, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var es = stack_allocation[LDA_K_CAP, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var dif = stack_allocation[LDA_K_CAP, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var cnt = stack_allocation[LDA_BLK_TPB + 1, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var misc = stack_allocation[2, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var t0 = tid
    while t0 < kk:
        ds[t0] = d.unsafe_load(base + t0)
        es[t0] = e.unsafe_load(base + t0)
        t0 += LDA_BLK_TPB
    # the nonzero words, ascending: contiguous segments, counts, offsets
    var seg = (vv + LDA_BLK_TPB - 1) // LDA_BLK_TPB
    var lo = tid * seg
    var hi = lo + seg
    if hi > vv:
        hi = vv
    var mine = 0
    for w in range(lo, hi):
        if ftz(x.unsafe_load(xb + w)) != Float32(0):
            mine += 1
    cnt[tid] = Int32(mine)
    barrier()
    if tid == 0:
        var run = 0
        for t in range(LDA_BLK_TPB):
            var cn = Int(cnt[t])
            cnt[t] = Int32(run)
            run += cn
        cnt[LDA_BLK_TPB] = Int32(run)
    barrier()
    var nnz = Int(cnt[LDA_BLK_TPB])
    var compact = nnz <= LDA_NNZ_CAP
    if compact:
        var at = Int(cnt[tid])
        for w in range(lo, hi):
            if ftz(x.unsafe_load(xb + w)) != Float32(0):
                idx[at] = Int32(w)
                at += 1
    barrier()
    var it = 0
    for _ in range(Int(max_iter)):
        it += 1
        # norm_phi per word (topics ascending), x_w / (norm_phi_w + eps)
        if compact:
            var j = tid
            while j < nnz:
                var w = Int(idx[j])
                var xw = ftz(x.unsafe_load(xb + w))
                var acc = Float32(0)
                for t in range(kk):
                    acc = ftz(identical_mul_add(ftz(es[t]), ftz(ew.unsafe_load(t * vv + w)), acc))
                sw[j] = div0(xw, add(acc, eps))
                j += LDA_BLK_TPB
        else:
            var w = tid
            while w < vv:
                var xw = ftz(x.unsafe_load(xb + w))
                if xw != Float32(0):
                    var acc = Float32(0)
                    for t in range(kk):
                        acc = ftz(identical_mul_add(ftz(es[t]), ftz(ew.unsafe_load(t * vv + w)), acc))
                    s.unsafe_store(sb + w, div0(xw, add(acc, eps)))
                w += LDA_BLK_TPB
        dev_barrier()
        # each topic's fold over the words ascending
        var t = tid
        while t < kk:
            var acc = Float32(0)
            var eb = t * vv
            if compact:
                for j in range(nnz):
                    acc = ftz(identical_mul_add(ftz(sw[j]), ftz(ew.unsafe_load(eb + Int(idx[j]))), acc))
            else:
                for w in range(vv):
                    if ftz(x.unsafe_load(xb + w)) == Float32(0):
                        continue
                    acc = ftz(identical_mul_add(ftz(s.unsafe_load(sb + w)), ftz(ew.unsafe_load(eb + w)), acc))
            var dt = add(mul(es[t], acc), prior)
            var old = ds[t]
            ds[t] = dt
            dif[t] = abs(sub(old, dt))
            t += LDA_BLK_TPB
        barrier()
        if tid == 0:
            var total = Float32(0)
            for q in range(kk):
                total = add(total, ds[q])
            misc[0] = digamma(total)
            var change = Float32(0)
            for q in range(kk):
                change = add(change, dif[q])
            misc[1] = Float32(1) if div0(change, Float32(kk)) < tol else Float32(0)
        barrier()
        var psi_total = misc[0]
        t = tid
        while t < kk:
            es[t] = exp_c(sub(digamma(ds[t]), psi_total))
            t += LDA_BLK_TPB
        var stop = misc[1] != Float32(0)
        barrier()
        if stop:
            break
    t0 = tid
    while t0 < kk:
        d.unsafe_store(base + t0, ds[t0])
        e.unsafe_store(base + t0, es[t0])
        t0 += LDA_BLK_TPB
    if tid == 0:
        its.unsafe_store(i, Float32(it))


def launch_lda_rows(
    ctx: DeviceContext, x: F32Ptr, ew: F32Ptr, d: F32Ptr, e: F32Ptr, s: F32Ptr, its: F32Ptr, n: Int, k: Int, v: Int,
    prior: Float32, max_iter: Int, tol: Float32,
) raises:
    """Every document's `lda_doc_row`: one block per document
    (`lda_block_kernel`) for k <= LDA_K_CAP, else one thread per document.
    s: n * (v + k) floats of scratch."""
    if n <= 0:
        return
    if k <= LDA_K_CAP:
        ctx.enqueue_function[lda_block_kernel](
            x, ew, d, e, s, its, Int32(n), Int32(k), Int32(v), prior, Int32(max_iter), tol,
            grid_dim=n, block_dim=LDA_BLK_TPB,
        )
    else:
        ctx.enqueue_function[lda_rows_kernel](
            x, ew, d, e, s, its, Int32(n), Int32(k), Int32(v), prior, Int32(max_iter), tol,
            grid_dim=_blocks(n), block_dim=TPB,
        )


# LatentDirichletAllocation._approx_bound's word term built 16 n x v term
# matrices plus their max, exp sums and logs (about 22 n x v buffers live:
# 31 GB at the text block, the L40S CUDA_ERROR_OUT_OF_MEMORY). This cell is
# the same chain of `ew_cell` calls per (i, w) with nothing stored but the
# product: terms_t = (0 + ddt[i, t]) + dcomp[t, w], their running max over t
# ascending, acc = sum_t exp(term_t - max) (t ascending, from 0), then
# x * (log_floor(acc, floor) + max). The caller folds it with the kit's own
# total (rowsum, then colsum), so the bound is the same bits.
def lda_bound_kernel(
    x: F32Ptr, ddt: F32Ptr, dcomp: F32Ptr, dst: F32Ptr, n: Int32, k: Int32, v: Int32, floor: Float32
):
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var vv = Int(v)
    var kk = Int(k)
    if c >= Int(n) * vv:
        return
    var i = c // vv
    var w = c - i * vv
    var z = Float32(0)
    var mx = Float32(0)
    for t in range(kk):
        var term = ew_cell(OP_ADD, ew_cell(OP_ADD, z, ddt.unsafe_load(i * kk + t), z, z), dcomp.unsafe_load(t * vv + w), z, z)
        mx = term if t == 0 else ew_cell(OP_MAX, mx, term, z, z)
    var acc = Float32(0)
    for t in range(kk):
        var term = ew_cell(OP_ADD, ew_cell(OP_ADD, z, ddt.unsafe_load(i * kk + t), z, z), dcomp.unsafe_load(t * vv + w), z, z)
        acc = ew_cell(OP_ADD, acc, ew_cell(OP_EXP, ew_cell(OP_SUB, term, mx, z, z), z, z, z), z, z)
    var lse = ew_cell(OP_ADD, ew_cell(OP_LOGS, acc, z, z, floor), mx, z, z)
    dst.unsafe_store(c, ew_cell(OP_MUL, x.unsafe_load(c), lse, z, z))


def launch_lda_bound(
    ctx: DeviceContext, x: F32Ptr, ddt: F32Ptr, dcomp: F32Ptr, dst: F32Ptr, n: Int, k: Int, v: Int, floor: Float32
) raises:
    if n * v <= 0:
        return
    ctx.enqueue_function[lda_bound_kernel](
        x, ddt, dcomp, dst, Int32(n), Int32(k), Int32(v), floor, grid_dim=_blocks(n * v), block_dim=TPB,
    )


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
    comptime if TARGET_COLUMN == COLUMN_APPLE:
        return True
    return False


def _xd_stage_ptr(ctx: DeviceContext) raises -> F32Ptr:
    """The process's pinned stage (XD_STAGE_FLOATS floats), created on first use."""
    var slot = X_DECOMP_STAGE.get_or_create_ptr()
    if not slot[].buf:
        slot[].buf = ctx.enqueue_create_host_buffer[DType.float32](XD_STAGE_FLOATS)
        ctx.synchronize()
    return F32Ptr(unsafe_from_address=Int(slot[].buf.value().unsafe_ptr()))


def _xd_read_out(dst: F32Ptr, src: F32Ptr, n: Int):
    """`memcpy(dst, src, n)`: the one read of pinned memory (one thread;
    cpu-gpu-cleanup c-decomp removed the host-task split)."""
    memcpy(dest=dst, src=src, count=n)


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


def _pj_blocks(count: Int) -> Int:
    return (count + PJ_TPB - 1) // PJ_TPB if count > 0 else 1


def _pj_off_blocks(n: Int) -> Int:
    return max((n + RR_OFF_TPB - 1) // RR_OFF_TPB, 1)


def _eigh_par_test(
    ctx: DeviceContext,
    mut da: DeviceBuffer[DType.float32],
    mut doff: DeviceBuffer[DType.float32],
    mut dpart: DeviceBuffer[DType.float32],
    mut dfold: DeviceBuffer[DType.float32],
    mut hfold: HostBuffer[DType.float32],
    n: Int,
) raises -> SIMD[DType.float32, 4]:
    """The round-robin eigh's convergence test on the device: the block
    trees of the rows' off-diagonal squares and a_kk^2 (a_kk left in
    doff[2 n, 3 n)), then the tree past the blocks (x_decomp/rr.mojo
    `rr_off_fold`, the host column's order). Returns (off, diag, ran, 0);
    ran < 0 is a dispatch that did not run (the buffers are filled -1)."""
    var nb = _pj_off_blocks(n)
    enqueue_fill(ctx, dpart, Float32(-1.0))
    enqueue_fill(ctx, dfold, Float32(-1.0))
    ctx.enqueue_function[eigh_par_off_part_kernel](
        da.unsafe_ptr(), doff.unsafe_ptr(), dpart.unsafe_ptr(), Int32(n), grid_dim=nb, block_dim=RR_OFF_TPB
    )
    ctx.enqueue_function[eigh_par_off_fold_kernel](
        dpart.unsafe_ptr(), dfold.unsafe_ptr(), Int32(nb), grid_dim=1, block_dim=RR_OFF_TPB
    )
    ctx.enqueue_copy(dst_ptr=hfold.unsafe_ptr(), src_buf=dfold)
    ctx.synchronize()
    var p = hfold.unsafe_ptr()
    return SIMD[DType.float32, 4](p.unsafe_load(0), p.unsafe_load(1), p.unsafe_load(2), Float32(0.0))


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
    comptime if XD_FAST_APPLE:
        # -D MOJOLEARN_DECOMP_FAST_GEMM_TILED (lane/apple-fast-decomp-linalg,
        # 2026-10-02, FAST on Apple only): the threadgroup-tiled kernel of
        # x_decomp/fast_gemm.mojo, 32 x 32 output tiles with the k axis
        # staged 16 deep, the same FOLD_BLOCK partials (grid z) and fold.
        # Cause: `gemm_kernel` / `gemm_part_kernel` are one thread per
        # output cell reading the whole k axis from device memory (every
        # operand word re-read once per output row or column it feeds).
        if fast_gemm_on() and m > 0 and n > 0 and k > 0:
            ctx.enqueue_function[fg_gemm_tiled_kernel](
                a, b, p if nb > 1 else c, Int32(m), Int32(k), Int32(n),
                Int32(1 if ta else 0), Int32(1 if tb else 0), Int32(nb),
                grid_dim=(fg_tiles(n), fg_tiles(m), nb), block_dim=(FG_TPB, 1, 1),
            )
            if nb > 1:
                ctx.enqueue_function[fold_kernel](p, c, Int32(m * n), Int32(nb), grid_dim=_blocks(m * n), block_dim=TPB)
            return
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


#: The device triangular solve's diagonal block: its rows are finished on
#: one thread per (row, column) cell before the rows it feeds take its steps.
#: The bits do not depend on it (every cell's chain is all of its j in one
#: order, a block at a time); a wider block halves the launches (lane hr-lu,
#: 2026-10-02: 64 -> 128).
comptime TRS_BLOCK = 128
#: Columns of B per thread block of `trs_diag_kernel` and its shared page.
comptime TRS_DIAG_COLS = 2
comptime TRS_DIAG_TPB = TRS_BLOCK * TRS_DIAG_COLS
comptime TRS_DIAG_PAGE_BYTES = TRS_DIAG_TPB * 4
comptime TRS_DIAG_FITS = lib_smem_page_fits_for[TARGET_COLUMN, TRS_DIAG_PAGE_BYTES]()


def trs_gather_kernel(src: F32Ptr, idx: F32Ptr, dst: F32Ptr, n: Int32, nrhs: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(n) * Int(nrhs):
        trs_gather(src, idx, dst, Int(nrhs), t // Int(nrhs), t % Int(nrhs))


def trs_copy_kernel(src: F32Ptr, dst: F32Ptr, count: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(count):
        dst.unsafe_store(t, src.unsafe_load(t))


def trs_block_kernel(lu: F32Ptr, b: F32Ptr, n: Int32, nrhs: Int32, tri: Int32, lo: Int32, hi: Int32):
    """`trs_block_col`, one thread per column."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c < Int(nrhs):
        trs_block_col(lu, b, Int(n), Int(nrhs), Int(tri), Int(lo), Int(hi), c)


def trs_diag_kernel(lu: F32Ptr, b: F32Ptr, n: Int32, nrhs: Int32, tri: Int32, lo: Int32, hi: Int32):
    """`trs_block_col` with a thread per (row, column) cell of the diagonal
    block [lo, hi) (lane hr-lu, 2026-10-02): the block's rows sit in one
    shared page, and at step j (forward ascending, backward descending)
    every row that j feeds takes `trs_step`'s fused multiply-add from the
    finished row j, then the next row to finish divides (`trs_div`), one
    barrier a step. Each cell's chain is `trs_block_col`'s, the same operands
    in the same order; the serial kernel walked h^2 / 2 dependent global
    loads and stores on ONE thread per column."""
    var s = stack_allocation[TRS_DIAG_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var r = tid // TRS_DIAG_COLS
    var cl = tid % TRS_DIAG_COLS
    var c = Int(block_idx.x) * TRS_DIAG_COLS + cl
    var nn = Int(n)
    var w = Int(nrhs)
    var t = Int(tri)
    var l0 = Int(lo)
    var h = Int(hi) - l0
    var i = l0 + r
    var live = r < h and c < w
    var divs = trs_divides(t)
    var d = Float32(0)
    if live:
        s.unsafe_store(tid, b.unsafe_load(i * w + c))
        if divs:
            d = lu.unsafe_load(i * nn + i)
    if t == 0 or t == 2:
        if live and divs and r == 0:
            s.unsafe_store(tid, div0(ftz(s.unsafe_load(tid)), d))
        barrier()
        for jr in range(h - 1):
            if live and r > jr:
                var acc = ftz(identical_mul_add(
                    -ftz(trs_coef(lu, nn, t, i, l0 + jr)), ftz(s.unsafe_load(jr * TRS_DIAG_COLS + cl)),
                    ftz(s.unsafe_load(tid))))
                if divs and r == jr + 1:
                    acc = div0(ftz(acc), d)
                s.unsafe_store(tid, acc)
            barrier()
    else:
        if live and divs and r == h - 1:
            s.unsafe_store(tid, div0(ftz(s.unsafe_load(tid)), d))
        barrier()
        for q in range(h - 1):
            var jr = h - 1 - q
            if live and r < jr:
                var acc = ftz(identical_mul_add(
                    -ftz(trs_coef(lu, nn, t, i, l0 + jr)), ftz(s.unsafe_load(jr * TRS_DIAG_COLS + cl)),
                    ftz(s.unsafe_load(tid))))
                if divs and r == jr - 1:
                    acc = div0(ftz(acc), d)
                s.unsafe_store(tid, acc)
            barrier()
    if live:
        b.unsafe_store(i * w + c, s.unsafe_load(tid))


def trs_feed_kernel(lu: F32Ptr, b: F32Ptr, n: Int32, nrhs: Int32, tri: Int32, lo: Int32, hi: Int32):
    """`trs_feed_cell` for every cell the block feeds: rows [hi, n) going
    forward, rows [0, lo) going backward."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var w = Int(nrhs)
    var fwd = Int(tri) == 0 or Int(tri) == 2
    var rows = Int(n) - Int(hi) if fwd else Int(lo)
    if t < rows * w:
        var i = t // w + (Int(hi) if fwd else 0)
        trs_feed_cell(lu, b, Int(n), w, Int(tri), Int(lo), Int(hi), i, t % w)


def launch_trs_tri(ctx: DeviceContext, lu: F32Ptr, b: F32Ptr, n: Int, nrhs: Int, tri: Int) raises:
    """One triangle by TRS_BLOCK-row blocks in the order they finish: the
    same chains as `trs_tri_serial`."""
    var nblk = (n + TRS_BLOCK - 1) // TRS_BLOCK
    for q in range(nblk):
        var bq = q if (tri == 0 or tri == 2) else nblk - 1 - q
        var lo = bq * TRS_BLOCK
        var hi = min(n, lo + TRS_BLOCK)
        comptime if TRS_DIAG_FITS:
            ctx.enqueue_function[trs_diag_kernel](
                lu, b, Int32(n), Int32(nrhs), Int32(tri), Int32(lo), Int32(hi),
                grid_dim=(nrhs + TRS_DIAG_COLS - 1) // TRS_DIAG_COLS, block_dim=TRS_DIAG_TPB,
            )
        else:
            ctx.enqueue_function[trs_block_kernel](
                lu, b, Int32(n), Int32(nrhs), Int32(tri), Int32(lo), Int32(hi), grid_dim=_blocks(nrhs), block_dim=TPB
            )
        var rows = n - hi if (tri == 0 or tri == 2) else lo
        if rows > 0:
            ctx.enqueue_function[trs_feed_kernel](
                lu, b, Int32(n), Int32(nrhs), Int32(tri), Int32(lo), Int32(hi), grid_dim=_blocks(rows * nrhs), block_dim=TPB
            )


def launch_trisolve(
    ctx: DeviceContext, lu: F32Ptr, idx: F32Ptr, src: F32Ptr, dst: F32Ptr, tmp: F32Ptr, n: Int, nrhs: Int, trans: Int
) raises:
    """`trisolve_serial` on device pointers, enqueued (no sync)."""
    var cells = n * nrhs
    if cells <= 0:
        return
    if trans == 0:
        ctx.enqueue_function[trs_gather_kernel](src, idx, dst, Int32(n), Int32(nrhs), grid_dim=_blocks(cells), block_dim=TPB)
        launch_trs_tri(ctx, lu, dst, n, nrhs, 0)
        launch_trs_tri(ctx, lu, dst, n, nrhs, 1)
        return
    ctx.enqueue_function[trs_copy_kernel](src, tmp, Int32(cells), grid_dim=_blocks(cells), block_dim=TPB)
    launch_trs_tri(ctx, lu, tmp, n, nrhs, 2)
    launch_trs_tri(ctx, lu, tmp, n, nrhs, 3)
    ctx.enqueue_function[trs_gather_kernel](tmp, idx, dst, Int32(n), Int32(nrhs), grid_dim=_blocks(cells), block_dim=TPB)


def lu_perm_kernel(piv: I32Ptr, idx: F32Ptr, n: Int32, trans: Int32):
    """The swaps' row order as `trisolve`'s idx, a thread per row: trans 0
    idx[i] = `lu_perm_src`(i) (the gather getrs's swaps make), trans 1 its
    inverse (the scatter of 'T''s swaps last to first). Integer compares."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        var p = lu_perm_src(piv, i)
        if Int(trans) == 0:
            idx.unsafe_store(i, Float32(p))
        else:
            idx.unsafe_store(p, Float32(i))


def lu_aux_part_kernel(lu: F32Ptr, piv: I32Ptr, part: F32Ptr, n_in: Int32):
    """Block b's join of rows b RR_OFF_TPB .. (`lu_aux_val`, `lu_aux_join`,
    a pairwise tree) to part[4 b ..]."""
    var n = Int(n_in)
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[4 * RR_OFF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var i = b * RR_OFF_TPB + tid
    var v = SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)
    if i < n:
        v = lu_aux_val(lu, piv, i, n)
    for c in range(4):
        sh[4 * tid + c] = v[c]
    barrier()
    var w = RR_OFF_TPB // 2
    while w > 0:
        if tid < w:
            var x = lu_aux_join(
                SIMD[DType.float32, 4](sh[4 * tid], sh[4 * tid + 1], sh[4 * tid + 2], sh[4 * tid + 3]),
                SIMD[DType.float32, 4](sh[4 * (tid + w)], sh[4 * (tid + w) + 1], sh[4 * (tid + w) + 2], sh[4 * (tid + w) + 3]),
            )
            for c in range(4):
                sh[4 * tid + c] = x[c]
        barrier()
        w = w // 2
    if tid == 0:
        for c in range(4):
            part.unsafe_store(4 * b + c, sh[c])


def lu_aux_fold_kernel(part: F32Ptr, stats: F32Ptr, nb_in: Int32):
    """ONE block over the nb block shares (nb = ceil(n / RR_OFF_TPB), a
    k-sized fold past the parallel part)."""
    var nb = Int(nb_in)
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[4 * RR_OFF_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var v = SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)
    var b = tid
    while b < nb:
        v = lu_aux_join(v, SIMD[DType.float32, 4](part.unsafe_load(4 * b), part.unsafe_load(4 * b + 1), part.unsafe_load(4 * b + 2), part.unsafe_load(4 * b + 3)))
        b += RR_OFF_TPB
    for c in range(4):
        sh[4 * tid + c] = v[c]
    barrier()
    var w = RR_OFF_TPB // 2
    while w > 0:
        if tid < w:
            var x = lu_aux_join(
                SIMD[DType.float32, 4](sh[4 * tid], sh[4 * tid + 1], sh[4 * tid + 2], sh[4 * tid + 3]),
                SIMD[DType.float32, 4](sh[4 * (tid + w)], sh[4 * (tid + w) + 1], sh[4 * (tid + w) + 2], sh[4 * (tid + w) + 3]),
            )
            for c in range(4):
                sh[4 * tid + c] = x[c]
        barrier()
        w = w // 2
    if tid == 0:
        for c in range(4):
            stats.unsafe_store(c, sh[c])


def lu_aux_clamp_kernel(lu: F32Ptr, diag: F32Ptr, stats: F32Ptr, n: Int32, clamp: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        lu_aux_clamp(lu, diag, stats.unsafe_load(0), i, Int(n), Int(clamp) != 0)


def launch_lu_solve(
    ctx: DeviceContext, lu: F32Ptr, piv: I32Ptr, b: F32Ptr, idx: F32Ptr, dst: F32Ptr, n: Int, nrhs: Int, trans: Int
) raises:
    """lu_solve in the right-looking order (DEVIATION 5308), enqueued (no
    sync): the swaps' row order, then `launch_trisolve` from b into dst
    (trans 1 solves in b itself before its gather)."""
    if n <= 0 or nrhs <= 0:
        return
    ctx.enqueue_function[lu_perm_kernel](piv, idx, Int32(n), Int32(trans), grid_dim=_blocks(n), block_dim=TPB)
    launch_trisolve(ctx, lu, idx, b, dst, b, n, nrhs, trans)


def knn_select_kernel(dmat: F32Ptr, dist: F32Ptr, idx: F32Ptr, n: Int32, m: Int32, k: Int32, exclude_self: Int32):
    """`knn_select_row`, one thread per row."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(n):
        knn_select_row(dmat, dist, idx, t, Int(m), Int(k), Int(exclude_self))


def launch_knn_select(
    ctx: DeviceContext, dmat: F32Ptr, dist: F32Ptr, idx: F32Ptr, n: Int, m: Int, k: Int, exclude_self: Int
) raises:
    if n > 0 and k > 0:
        ctx.enqueue_function[knn_select_kernel](
            dmat, dist, idx, Int32(n), Int32(m), Int32(k), Int32(exclude_self), grid_dim=_blocks(n), block_dim=TPB
        )


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
    var none = ctx.enqueue_create_buffer[DType.float32](1)
    orth_on_device_diag(ctx, da, m, l, none, False)
    _ = none^


def orth_on_device_diag(
    ctx: DeviceContext, da: DeviceBuffer[DType.float32], m: Int, l: Int, mut ddiag: DeviceBuffer[DType.float32],
    with_diag: Bool,
) raises:
    """`orth_on_device`, and with `with_diag` the product of the two passes'
    guarded R diagonals into `ddiag` (l floats the caller filled with 1.0):
    `ddiag[j] = R2[j, j] * R1[j, j]`, whose sign orients Q's column j along
    the input's (Q_j . A_j = (R2 R1)[j, j]), and a dependent column's is 0
    (lane neural-pass17, `linalg.svd`'s U)."""
    var cells = m * l if m * l > 0 else 1
    var dq = ctx.enqueue_create_buffer[DType.float32](cells)
    var dw = ctx.enqueue_create_buffer[DType.float32](cells)
    var scratch = ctx.enqueue_create_buffer[DType.float32](qr_slice_count(m, l) * l * l if l > 0 else 1)
    var r_buf = ctx.enqueue_create_buffer[DType.float32](l * l if l > 0 else 1)
    # -D MOJOLEARN_SVD_FAST_CHOLQR (lane/apple-fast-decomp-linalg, 2026-10-02,
    # FAST on Apple only, the `with_diag` caller = linalg.svd's U): each pass
    # as CholeskyQR, G = A^T A on the kit's split-K gemm (a grid), L = chol(G)
    # by the per-column kernels, R = L^T, then the same row-parallel A R^-1;
    # two passes = CholeskyQR2. Cause: a pass here is a sliced Householder
    # TSQR over the 1,000,000 x d matrix (64 slices x 32 threads, every load
    # a column at stride d), and svd runs two of them on A V / s, whose
    # columns are already near-orthonormal, so the Gram is well conditioned.
    # Guard (quality never lowered): the pass falls back to the TSQR when the
    # factorization fails (info != 0) or the factor's diagonal spans more
    # than 2^8 (cond(A) past CholeskyQR2's float32 bound); read once per
    # pass on the host, where the pass waits anyway.
    var cholqr = False
    comptime if XD_FAST_APPLE:
        cholqr = with_diag and l > 0 and m >= l and is_defined["MOJOLEARN_SVD_FAST_CHOLQR"]()
    var gcells = l * l if l > 0 else 1
    var gscr_n = gemm_scratch(l, m, l) if cholqr else 0
    var dg = ctx.enqueue_create_buffer[DType.float32](gcells)
    var dinfo = ctx.enqueue_create_buffer[DType.float32](1)
    var gscr = ctx.enqueue_create_buffer[DType.float32](gscr_n if gscr_n > 0 else 1)
    var hg = ctx.enqueue_create_host_buffer[DType.float32](gcells)
    var hinfo = ctx.enqueue_create_host_buffer[DType.float32](1)
    for p in range(2):
        var src = da if p == 0 else dq
        var dst = dq if p == 0 else da
        var done = False
        if cholqr:
            launch_gemm(
                ctx, src.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), src.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                dg.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), gscr.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                l, m, l, True, False,
            )
            var gblocked = False
            comptime if CHOL_FAST_BLOCKED:
                # pass 2: the l x l Cholesky of the Gram as the blocked
                # launch (3 l / CH_NB launches) instead of 2 l one-thread
                # and column launches (l = 220: 440 a pass, 880 an svd).
                if l > CH_NB:
                    launch_chol_blocked(ctx, _p(dg), _p(dinfo), l)
                    gblocked = True
            if not gblocked:
                # main's column driver (c-decomp): one `chol_step_kernel`
                # launch a column of the l x l Gram
                ctx.enqueue_function[lu_info_init_kernel](dinfo.unsafe_ptr(), grid_dim=1, block_dim=1)
                for j in range(l):
                    ctx.enqueue_function[chol_step_kernel](
                        dg.unsafe_ptr(), dinfo.unsafe_ptr(), Int32(j), Int32(l), grid_dim=_blocks(l - j), block_dim=TPB
                    )
            ctx.enqueue_copy(dst_ptr=hinfo.unsafe_ptr(), src_buf=dinfo)
            ctx.enqueue_copy(dst_ptr=hg.unsafe_ptr(), src_buf=dg)
            ctx.synchronize()
            var ok = hinfo.unsafe_ptr().unsafe_load(0) == Float32(0)
            var dmax = Float32(0)
            var dmin = Float32(0)
            for j in range(l):
                var d = hg.unsafe_ptr().unsafe_load(j * l + j)
                if not (d > Float32(0)):
                    ok = False
                if j == 0 or d > dmax:
                    dmax = d
                if j == 0 or d < dmin:
                    dmin = d
            if ok and not (dmin * Float32(256.0) >= dmax):
                ok = False
            if ok:
                ctx.enqueue_function[pj_transpose_kernel](
                    dg.unsafe_ptr(), r_buf.unsafe_ptr(), Int32(l), grid_dim=_pj_blocks(l * l), block_dim=PJ_TPB
                )
                done = True
        if done:
            pass
        else:
            ctx.enqueue_copy(dst_buf=dw, src_buf=src)
            # lane/decomp-apple2: the guard cell (k-sized: the l x l R) on one
            # device thread, in stream order after the R it reads, so R never
            # leaves the device (cgr-decomp: the MOJOLEARN_XD_ORTH_DEV=0 host
            # round trip is deleted)
            _ = qr_factor(ctx, dw, scratch, r_buf, m, l)
            ctx.enqueue_function[orth_guard_kernel](r_buf.unsafe_ptr(), Int32(l), grid_dim=1, block_dim=1)
        if with_diag and l > 0:
            ctx.enqueue_function[orth_diag_kernel](
                r_buf.unsafe_ptr(), ddiag.unsafe_ptr(), Int32(l), grid_dim=_blocks(l), block_dim=TPB
            )
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
    _ = dg^
    _ = dinfo^
    _ = gscr^
    _ = hg^
    _ = hinfo^
    ctx.synchronize()


def launch_lu(
    ctx: DeviceContext, a: F32Ptr, piv: I32Ptr, info: F32Ptr, scal: F32Ptr, act: F32Ptr, n: Int
) raises:
    """DevExec.lu's launches on device pointers, enqueued (no sync): `a`
    (n x n) factored in place, `piv` (n), `info` (1), `scal` (LU_SCAL_LEN) and `act`
    (n) scratch. DevExec.lu and the resident kit (x_decomp/kit_device.mojo)
    both call it, so the two run one launch sequence."""
    # lu_serial's cells, step by step at every n: the pivot search over
    # every block (a parallel reduction, `lu_pivot`'s choice), the swap (with
    # the diagonal step on column k's thread), the
    # multipliers and the trailing update one thread per cell. (The
    # one-thread `lu_kernel` for n <= MOJOLEARN_XD_LU_SERIAL and the
    # one-thread pivot scan are deleted: the same cells in the same order.)
    ctx.enqueue_function[lu_info_init_kernel](info, grid_dim=1, block_dim=1)
    # lane neural-pass115: the r4 trail defaults off on AMD (the peer's
    # MI325X: lu-factor 868 -> 930 ms with it; the L40S and the M4 gain):
    # =1 opts in. The same words either way. (np115's fused one-block step,
    # MOJOLEARN_XD_LU_STEP_FUSED, is not on main: a new one-block launch,
    # refused by tools/hooks/no_host_routes.py.)
    var trail_r4 = True
    comptime if TARGET_COLUMN == COLUMN_AMD:
        trail_r4 = False
    var swaps_trsm = lu_swaps_trsm_on()
    var trail_rb = lu_trail_rb_on()
    var nb = min(lu_panel_width(), LU_PANEL_NB)
    if nb > 0:
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
                comptime if LU_FAST_PIVOT_GRID:
                    var g = lu_pivot_blocks(k, n)
                    ctx.enqueue_function[lu_pivot_part_kernel](
                        a, scal, Int32(k), Int32(n), Int32(g), grid_dim=g, block_dim=LU_PIVOT_TPB
                    )
                    ctx.enqueue_function[lu_pivot_swap_kernel](
                        a, piv, info, scal, act, Int32(k), Int32(n), Int32(0), Int32(k1), Int32(g),
                        grid_dim=_blocks(k1), block_dim=TPB,
                    )
                else:
                    enqueue_lu_pivot(ctx, a, piv, scal, k, n)
                    ctx.enqueue_function[lu_swap_cols_kernel](
                        a, piv, info, scal, act, Int32(k), Int32(n), Int32(0), Int32(k1),
                        grid_dim=_blocks(k1), block_dim=TPB,
                    )
                if n - k - 1 > 0:
                    ctx.enqueue_function[lu_l_kernel](
                        a, scal, Int32(k), Int32(n), grid_dim=_blocks(n - k - 1), block_dim=TPB
                    )
                if k1 - k - 1 > 0 and n - k - 1 > 0:
                    ctx.enqueue_function[lu_update_panel_kernel](
                        a, scal, Int32(k), Int32(n), Int32(k1),
                        grid_dim=_blocks((n - k - 1) * (k1 - k - 1)), block_dim=TPB,
                    )
            if k1 < n:
                if swaps_trsm:
                    ctx.enqueue_function[lu_swaps_trsm_kernel](
                        a, piv, act, Int32(k0), Int32(k1), Int32(n),
                        grid_dim=_blocks(n - k1), block_dim=TPB,
                    )
                else:
                    ctx.enqueue_function[lu_apply_swaps_kernel](
                        a, piv, Int32(k0), Int32(k1), Int32(n),
                        grid_dim=_blocks(n - k1), block_dim=TPB,
                    )
                    if k1 - k0 > 1:
                        ctx.enqueue_function[lu_trsm_kernel](
                            a, act, Int32(k0), Int32(k1), Int32(n),
                            grid_dim=_blocks(n - k1), block_dim=TPB,
                        )
                var r4 = False
                comptime if LU_TRAIL_R4:
                    r4 = trail_r4
                if trail_rb:
                    # lane neural-pass135's register-blocked trail (default);
                    # MOJOLEARN_XD_LU_TRAIL_RB=0 falls back to np115's r4 arm
                    var rtiles = (n - k1 + LU_RB_TILE - 1) // LU_RB_TILE
                    ctx.enqueue_function[lu_trail_rb_kernel](
                        a, act, Int32(k0), Int32(k1), Int32(n), Int32(k1 - k0),
                        grid_dim=(rtiles, rtiles, 1), block_dim=(LU_TILE_TPB, 1, 1),
                    )
                elif r4:
                    var t4 = (n - k1 + LUR_T - 1) // LUR_T
                    ctx.enqueue_function[lu_trail_r4_kernel](
                        a, act, Int32(k0), Int32(k1), Int32(n), Int32(k1 - k0),
                        grid_dim=(t4, t4, 1), block_dim=(LU_TILE_TPB, 1, 1),
                    )
                else:
                    var tiles = (n - k1 + LU_TILE - 1) // LU_TILE
                    ctx.enqueue_function[lu_trail_tiled_kernel](
                        a, act, Int32(k0), Int32(k1), Int32(n), Int32(k1 - k0),
                        grid_dim=(tiles, tiles, 1), block_dim=(LU_TILE_TPB, 1, 1),
                    )
            k0 = k1
    for k in range(n if nb == 0 else 0):
        enqueue_lu_pivot(ctx, a, piv, scal, k, n)
        ctx.enqueue_function[lu_swap_kernel](
            a, piv, info, scal, act, Int32(k), Int32(n), grid_dim=_blocks(n), block_dim=TPB
        )
        if n - k - 1 > 0:
            ctx.enqueue_function[lu_l_kernel](
                a, scal, Int32(k), Int32(n), grid_dim=_blocks(n - k - 1), block_dim=TPB
            )
            ctx.enqueue_function[lu_update_kernel](
                a, scal, Int32(k), Int32(n),
                grid_dim=_blocks((n - k - 1) * (n - k - 1)), block_dim=TPB,
            )


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
        var ds = ctx.enqueue_create_buffer[DType.float32](LU_SCAL_LEN)
        var dact = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        launch_lu(
            ctx, _p(da), I32Ptr(unsafe_from_address=Int(dp.unsafe_ptr())), _p(di), _p(ds), _p(dact), n
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
    def knn_select(dmat: F32Ptr, dist: F32Ptr, idx: F32Ptr, n: Int, m: Int, k: Int, exclude_self: Int) raises:
        var ctx = xd_ctx()
        var dd = _up(ctx, dmat, n * m)
        var ds = ctx.enqueue_create_buffer[DType.float32](max(n * k, 1))
        var di = ctx.enqueue_create_buffer[DType.float32](max(n * k, 1))
        launch_knn_select(ctx, _p(dd), _p(ds), _p(di), n, m, k, exclude_self)
        _down(ctx, ds, dist, n * k)
        _down(ctx, di, idx, n * k)
        ctx.synchronize()
        _ = dd^
        _ = ds^
        _ = di^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def trisolve(lu: F32Ptr, idx: F32Ptr, src: F32Ptr, dst: F32Ptr, n: Int, nrhs: Int, trans: Int) raises:
        var ctx = xd_ctx()
        var dl = _up(ctx, lu, n * n)
        var di = _up(ctx, idx, n)
        var ds = _up(ctx, src, n * nrhs)
        var dd = ctx.enqueue_create_buffer[DType.float32](max(n * nrhs, 1))
        var dt = ctx.enqueue_create_buffer[DType.float32](max(n * nrhs, 1))
        launch_trisolve(ctx, _p(dl), _p(di), _p(ds), _p(dd), _p(dt), n, nrhs, trans)
        _down(ctx, dd, dst, n * nrhs)
        ctx.synchronize()
        _ = dl^
        _ = di^
        _ = ds^
        _ = dd^
        _ = dt^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def lu_solve(lu: F32Ptr, piv: I32Ptr, b: F32Ptr, n: Int, nrhs: Int, trans: Int = 0) raises:
        if n <= 0 or nrhs <= 0:
            return
        var cx = xd_ctx()
        var ul = _up(cx, lu, n * n)
        var up = _up_i(cx, piv, n)
        var ub = _up(cx, b, n * nrhs)
        var ui = cx.enqueue_create_buffer[DType.float32](n)
        var ud = cx.enqueue_create_buffer[DType.float32](n * nrhs)
        launch_lu_solve(cx, _p(ul), I32Ptr(unsafe_from_address=Int(up.unsafe_ptr())), _p(ub), _p(ui), _p(ud), n, nrhs, trans)
        _down(cx, ud, b, n * nrhs)
        cx.synchronize()
        _ = ul^
        _ = up^
        _ = ub^
        _ = ui^
        _ = ud^
        cx.synchronize()
        _ = cx^

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
        var blocked = False
        comptime if CHOL_FAST_BLOCKED:
            # -D MOJOLEARN_CHOL_FAST_BLOCKED (lane/apple-fast-decomp-linalg,
            # pass 2): `launch_chol_blocked` (see the kernels), FAST on
            # Apple only; a matrix within one panel keeps the driver below.
            if n > CH_NB:
                launch_chol_blocked(ctx, _p(da), _p(di), n)
                blocked = True
        if not blocked:
            ctx.enqueue_function[lu_info_init_kernel](di.unsafe_ptr(), grid_dim=1, block_dim=1)
            for j in range(n):
                ctx.enqueue_function[chol_step_kernel](
                    da.unsafe_ptr(), di.unsafe_ptr(), Int32(j), Int32(n), grid_dim=_blocks(n - j), block_dim=TPB
                )
        _down(ctx, da, a, n * n)
        _down(ctx, di, info, 1)
        ctx.synchronize()
        _ = da^
        _ = di^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def eigh(a: F32Ptr, w: F32Ptr, v: F32Ptr, n: Int, uplo: Int) raises:
        """THE eigh at every size: the round-robin Jacobi (x_decomp/rr.mojo,
        x_decomp/jacobi_par.mojo; the host runs the same rounds). cgr-decomp
        (2026-10-03): the one-block cyclic solvers it fell back to
        (`device_eigh`, `jacobi_eigh2_kernel`) and the cyclic define are
        deleted; a solve that does not converge in RR_EIGH_SWEEPS raises.
        uplo 1 / 2 reads the lower / upper triangle (numpy's UPLO), mirrored
        on the device; 0 the whole matrix."""
        var ctx = xd_ctx()
        var da = _up(ctx, a, n * n)
        if uplo != 0:
            ctx.enqueue_function[sym_from_triangle_kernel](
                da.unsafe_ptr(), Int32(n), Int32(uplo), grid_dim=_pj_blocks(n * n), block_dim=PJ_TPB
            )
        _ = DevExec._eigh_par_on(ctx, da, w, v, n)
        _ = da^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def _eigh_par_on(ctx: DeviceContext, mut da: DeviceBuffer[DType.float32], w: F32Ptr, v: F32Ptr, n: Int) raises -> Int:
        """The two-sided Jacobi in the round-robin ordering on the device
        copy in `da` (consumed): `eigh_par_cs_kernel` / `eigh_par_update_kernel`
        per round, the convergence test before every sweep folded on the
        device (`_eigh_par_test`, three scalars read), then
        `sign_flip_kernel` and the ascending permutation; w (n) and v (n x n)
        out to host memory. The resident kit (x_decomp/kit_device.mojo)
        hands its own copy here, the Lanczos projected solve too. Returns the
        sweeps run."""
        var m = n + (n % 2)
        var h = m // 2
        var dv = ctx.enqueue_create_buffer[DType.float32](n * n)
        var dcs = ctx.enqueue_create_buffer[DType.float32](2 * h)
        var doff = ctx.enqueue_create_buffer[DType.float32](3 * n)
        var dpart = ctx.enqueue_create_buffer[DType.float32](3 * _pj_off_blocks(n))
        var dfold = ctx.enqueue_create_buffer[DType.float32](3)
        var hfold = ctx.enqueue_create_host_buffer[DType.float32](3)
        ctx.enqueue_function[pj_identity_kernel](dv.unsafe_ptr(), Int32(n), grid_dim=_pj_blocks(n * n), block_dim=PJ_TPB)
        var converged = False
        var executed = 0
        var fro_in = Float32(-1.0)
        var fro_now = Float32(0.0)
        var off_last = Float32(0.0)
        for sweep in range(RR_EIGH_SWEEPS + 1):
            var tst = _eigh_par_test(ctx, da, doff, dpart, dfold, hfold, n)
            if not (tst[2] >= Float32(0.0)):
                raise Error(
                    "eigh: a block of the round-robin Jacobi's convergence test did not run (its mark is"
                    " still -1): a launch failure, not a convergence failure. Check that the binding is"
                    " built for this device."
                )
            off_last = tst[0]
            fro_now = ftz(tst[0] + tst[1])
            if fro_in < Float32(0.0):
                fro_in = fro_now
            if rr_converged(tst[0], tst[1], Float32(JACOBI_TOL)):
                converged = True
                break
            if sweep == RR_EIGH_SWEEPS:
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
        if converged and not rr_fro_kept(fro_in, fro_now):
            converged = False
        if not converged:
            raise Error(
                "eigh: the round-robin Jacobi did not converge in " + String(RR_EIGH_SWEEPS)
                + " sweeps at n = " + String(n) + " (off-diagonal mass " + String(off_last)
                + " of " + String(fro_now) + "). An unconverged decomposition is not returned as if"
                " it were one (DEVIATION 590)."
            )
        ctx.enqueue_function[sign_flip_kernel](
            dv.unsafe_ptr(), Int32(n), grid_dim=(n, 1, 1), block_dim=(SIGNFLIP_TPB, 1, 1)
        )
        # the last test's kernel left the diagonal of the converged A in
        # doff[2n, 3n); the ascending order on the device
        # (decomposition/spectrum_order_device.mojo), then w and v out
        var dw = ctx.enqueue_create_buffer[DType.float32](n)
        var dvo = ctx.enqueue_create_buffer[DType.float32](n * n)
        var dpos = ctx.enqueue_create_buffer[DType.int32](n)
        enqueue_eigh_ascending(ctx, _p(doff) + 2 * n, 1, _p(dv), n, dpos, _p(dw), _p(dvo))
        _down(ctx, dw, w, n)
        _down(ctx, dvo, v, n * n)
        ctx.synchronize()
        _ = dw^
        _ = dvo^
        _ = dpos^
        _ = dv^
        _ = dcs^
        _ = doff^
        _ = dpart^
        _ = dfold^
        _ = hfold^
        return executed

    @staticmethod
    def _rr_batch_on(
        ctx: DeviceContext, mut da: DeviceBuffer[DType.float32], batch: Int, n: Int,
        mut dw: DeviceBuffer[DType.float32], mut dvo: DeviceBuffer[DType.float32],
    ) raises:
        """x_decomp/rr_batch.mojo on `batch` n x n problems already on the
        device in `da` (consumed): dw (batch x n ascending) and dvo (batch x
        n x n, vectors in columns). One sync to read the convergence marks;
        an unconverged problem raises."""
        var dv = ctx.enqueue_create_buffer[DType.float32](batch * n * n)
        var dcs = ctx.enqueue_create_buffer[DType.float32](batch * rrb_cs_len(n))
        var dpart = ctx.enqueue_create_buffer[DType.float32](batch * rrb_part_len(n))
        var ddg = ctx.enqueue_create_buffer[DType.float32](batch * n)
        var dinfo = ctx.enqueue_create_buffer[DType.float32](2 * batch)
        enqueue_fill(ctx, dinfo, Float32(-1.0))
        ctx.enqueue_function[rr_batch_kernel](
            _p(da), _p(dv), _p(dcs), _p(dpart), _p(ddg), _p(dinfo), _p(dw), _p(dvo), Int32(n),
            Int32(RR_EIGH_SWEEPS), Float32(JACOBI_TOL), grid_dim=batch, block_dim=RR_OFF_TPB,
        )
        var hinfo = List[Float32](length=2 * batch, fill=Float32(0.0))
        _down(ctx, dinfo, F32Ptr(unsafe_from_address=Int(hinfo.unsafe_ptr())), 2 * batch)
        ctx.synchronize()
        _ = dv^
        _ = dcs^
        _ = dpart^
        _ = ddg^
        _ = dinfo^
        for b in range(batch):
            if hinfo[2 * b] < Float32(0.0):
                raise Error("eigh_batch: a block of the batched Jacobi did not run (a launch failure)")
            if hinfo[2 * b] == Float32(0.0):
                raise Error(
                    "eigh_batch: problem " + String(b) + " of " + String(batch) + " (n = " + String(n)
                    + ") did not converge in " + String(RR_EIGH_SWEEPS) + " sweeps. An unconverged"
                    " decomposition is not returned as if it were one (DEVIATION 590)."
                )
        _ = hinfo^

    @staticmethod
    def eigh_batch(a: F32Ptr, w: F32Ptr, v: F32Ptr, batch: Int, n: Int) raises:
        """`eigh` of `batch` n x n problems stacked in `a`, one block each
        (x_decomp/rr_batch.mojo): w batch x n ascending, v batch x n x n
        (vectors in columns); the same words `eigh` gives each one."""
        if batch <= 0:
            return
        var ctx = xd_ctx()
        var da = _up(ctx, a, batch * n * n)
        var dw = ctx.enqueue_create_buffer[DType.float32](batch * n)
        var dvo = ctx.enqueue_create_buffer[DType.float32](batch * n * n)
        DevExec._rr_batch_on(ctx, da, batch, n, dw, dvo)
        _down(ctx, dw, w, batch * n)
        _down(ctx, dvo, v, batch * n * n)
        ctx.synchronize()
        _ = da^
        _ = dw^
        _ = dvo^
        _ = ctx^

    @staticmethod
    def lle_apply(wb: F32Ptr, idx: F32Ptr, emb: F32Ptr, dst: F32Ptr, nq: Int, nf: Int, nn: Int, nc: Int) raises:
        """LLE transform's out = W E[idx] (`lle_apply_cell`), one thread a cell."""
        var ctx = xd_ctx()
        var dwb = _up(ctx, wb, nq * nn)
        var di = _up(ctx, idx, nq * nn)
        var de = _up(ctx, emb, nf * nc)
        var dout = ctx.enqueue_create_buffer[DType.float32](max(nq * nc, 1))
        ctx.enqueue_function[lle_apply_kernel](
            _p(dwb), _p(di), _p(de), _p(dout), Int32(nq), Int32(nn), Int32(nc), grid_dim=_blocks(nq * nc), block_dim=TPB
        )
        _down(ctx, dout, dst, nq * nc)
        ctx.synchronize()
        _ = dwb^
        _ = di^
        _ = de^
        _ = dout^
        _ = ctx^

    @staticmethod
    def lle_local(
        x: F32Ptr, idx: F32Ptr, bmat: F32Ptr, method: Int, n: Int, d: Int, nn: Int, nc: Int, tol: Float32
    ) raises:
        """LocallyLinearEmbedding's stacked factor B for method 0 LTSA (n nn x
        n), 1 Hessian (n (nn - 1 - nc) x n), 2 modified (n nn x n), every
        per-sample step a cell of x_decomp/lle_local.mojo on the device and
        the local eigensolves batched (x_decomp/rr_batch.mojo). B is written
        whole (zeros included); one download."""
        var ctx = xd_ctx()
        var nn2 = n * nn * nn
        var rows = n * (nn - 1 - nc) if method == 1 else n * nn
        var dx = _up(ctx, x, n * d)
        var di = _up(ctx, idx, n * nn)
        var db = ctx.enqueue_create_buffer[DType.float32](max(rows * n, 1))
        enqueue_fill(ctx, db, Float32(0.0))
        var dg = ctx.enqueue_create_buffer[DType.float32](max(nn2, 1))
        var dmu = ctx.enqueue_create_buffer[DType.float32](max(n * d, 1))
        if method == 2:
            ctx.enqueue_function[lle_gram_kernel](
                _p(dx), _p(di), _p(dx), _p(dg), Int32(n), Int32(d), Int32(nn), grid_dim=_blocks(nn2), block_dim=TPB
            )
        else:
            ctx.enqueue_function[lle_mean_kernel](
                _p(dx), _p(di), _p(dmu), Int32(n), Int32(d), Int32(nn), grid_dim=_blocks(n * d), block_dim=TPB
            )
            ctx.enqueue_function[lle_gram_kernel](
                _p(dx), _p(di), _p(dmu), _p(dg), Int32(n), Int32(d), Int32(nn), grid_dim=_blocks(nn2), block_dim=TPB
            )
        var dw = ctx.enqueue_create_buffer[DType.float32](max(n * nn, 1))
        var dv = ctx.enqueue_create_buffer[DType.float32](max(nn2, 1))
        DevExec._rr_batch_on(ctx, dg, n, nn, dw, dv)
        if method == 0:
            ctx.enqueue_function[ltsa_kernel](
                _p(dv), _p(di), _p(db), Int32(n), Int32(nn), Int32(nc), grid_dim=_blocks(nn2), block_dim=TPB
            )
        elif method == 1:
            var ncy = hessian_ncy(nc)
            var ncol = nn - 1 - nc
            var extra = ncol - nc * (nc + 1) // 2
            var dq = ctx.enqueue_create_buffer[DType.float32](max(n * nn * ncy, 1))
            ctx.enqueue_function[hessian_q_kernel](
                _p(dv), _p(dq), Int32(n), Int32(nn), Int32(nc), grid_dim=_blocks(n), block_dim=TPB
            )
            var dvc = ctx.enqueue_create_buffer[DType.float32](max(nn2, 1))
            if extra > 0:
                var dc = ctx.enqueue_create_buffer[DType.float32](max(nn2, 1))
                ctx.enqueue_function[hessian_comp_kernel](
                    _p(dq), _p(dc), Int32(n), Int32(nn), Int32(nc), grid_dim=_blocks(nn2), block_dim=TPB
                )
                var dw2 = ctx.enqueue_create_buffer[DType.float32](max(n * nn, 1))
                DevExec._rr_batch_on(ctx, dc, n, nn, dw2, dvc)
                _ = dc^
                _ = dw2^
            ctx.enqueue_function[hessian_kernel](
                _p(dq), _p(dvc), _p(di), _p(db), Int32(n), Int32(nn), Int32(nc), tol,
                grid_dim=_blocks(n * ncol), block_dim=TPB,
            )
            ctx.synchronize()
            _ = dq^
            _ = dvc^
        else:
            var nev = min(d, nn)
            var dwr = ctx.enqueue_create_buffer[DType.float32](max(n * nn, 1))
            var drho = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
            var dscr = ctx.enqueue_create_buffer[DType.float32](max(3 * n * nn, 1))
            ctx.enqueue_function[mlle_weights_kernel](
                _p(dw), _p(dv), _p(dwr), _p(drho), _p(dscr), Int32(n), Int32(nn), Int32(nev), Int32(nc),
                grid_dim=_blocks(n), block_dim=TPB,
            )
            # eta = the median of rho: an exact radix sort of its keys
            var keys = ctx.enqueue_create_buffer[DType.uint32](max(n, 1))
            var vals = ctx.enqueue_create_buffer[DType.uint32](max(n, 1))
            var tk = ctx.enqueue_create_buffer[DType.uint32](max(n, 1))
            var tv = ctx.enqueue_create_buffer[DType.uint32](max(n, 1))
            var cnt = ctx.enqueue_create_buffer[DType.int32](max(frs_counts_len(n), 1))
            ctx.enqueue_function[mlle_key_kernel](
                _p(drho), keys.unsafe_ptr(), vals.unsafe_ptr(), Int32(n), grid_dim=_blocks(n), block_dim=TPB
            )
            fast_radix_sort_pairs_u32(ctx, n, keys, vals, tk, tv, cnt)
            var dsrt = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
            ctx.enqueue_function[mlle_unkey_kernel](keys.unsafe_ptr(), _p(dsrt), Int32(n), grid_dim=_blocks(n), block_dim=TPB)
            ctx.enqueue_function[mlle_rows_kernel](
                _p(dw), _p(dv), _p(dwr), _p(di), _p(db), _p(dscr), _p(dsrt), Int32(n), Int32(nn), Int32(nev), tol,
                grid_dim=_blocks(n), block_dim=TPB,
            )
            ctx.synchronize()
            _ = dwr^
            _ = drho^
            _ = dscr^
            _ = keys^
            _ = vals^
            _ = tk^
            _ = tv^
            _ = cnt^
            _ = dsrt^
        _down(ctx, db, bmat, rows * n)
        ctx.synchronize()
        _ = dx^
        _ = di^
        _ = db^
        _ = dg^
        _ = dmu^
        _ = dw^
        _ = dv^
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
    def orth_diag(a: F32Ptr, m: Int, l: Int, diag: F32Ptr) raises:
        """`orth`, and `diag` (l floats) the product of the two passes' R
        diagonals (`orth_on_device_diag`). One upload, two downloads."""
        var ctx = xd_ctx()
        var da = _up(ctx, a, m * l)
        var dd = ctx.enqueue_create_buffer[DType.float32](l if l > 0 else 1)
        enqueue_fill(ctx, dd, Float32(1.0))
        orth_on_device_diag(ctx, da, m, l, dd, True)
        _down(ctx, da, a, m * l)
        _down(ctx, dd, diag, l)
        ctx.synchronize()
        _ = da^
        _ = dd^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def svd(a: F32Ptr, m: Int, n: Int, s: F32Ptr, v: F32Ptr) raises:
        """The tall route: the bounded Householder QR, then the one-sided
        Jacobi SVD of R in the round-robin order (x_decomp/rr_svd.mojo, one
        block a pair, every pair of a round at once) keeping V."""
        DevExec.svd_cells(a, m, n, s, v, QRB_CELLS)

    @staticmethod
    def svd_cells(a: F32Ptr, m: Int, n: Int, s: F32Ptr, v: F32Ptr, qr_cells: Int) raises:
        """`svd` with the QR's work per launch named: dense_check proves the
        bits do not depend on it. cgr-decomp (2026-10-03): the one-block
        cyclic solvers (`svd_of_r`, `one_sided_svd2_chunk_kernel`) and their
        MOJOLEARN_XD_JACOBI switch are replaced by the round-robin rounds."""
        var ctx = xd_ctx()
        var da = _up(ctx, a, m * n)
        var scratch = ctx.enqueue_create_buffer[DType.float32](qr_slice_count(m, n) * n * n)
        var r_buf = ctx.enqueue_create_buffer[DType.float32](n * n)
        var rt = ctx.enqueue_create_buffer[DType.float32](n * n)
        var vt = ctx.enqueue_create_buffer[DType.float32](n * n)
        var v_buf = ctx.enqueue_create_buffer[DType.float32](n * n)
        var s_buf = ctx.enqueue_create_buffer[DType.float32](n)
        var mm = n + (n % 2)
        var h = mm // 2
        var flags = ctx.enqueue_create_buffer[DType.float32](max(h, 1))
        var hflags = List[Float32](length=max(h, 1), fill=Float32(0.0))
        ctx.synchronize()
        # bounded in work per launch and poisoned (x_decomp/qr_bounded.mojo):
        # a launch macOS cut short leaves NaN in R, hence in s, refused below
        _ = qr_factor_bounded(ctx, da, scratch, r_buf, m, n, qr_cells)
        ctx.enqueue_function[pj_transpose_kernel](_p(r_buf), _p(rt), Int32(n), grid_dim=_pj_blocks(n * n), block_dim=PJ_TPB)
        ctx.enqueue_function[pj_identity_kernel](_p(vt), Int32(n), grid_dim=_pj_blocks(n * n), block_dim=PJ_TPB)
        var converged = n < 2
        var executed = 0
        while not converged and executed < X_DECOMP_SVD_SWEEPS:
            executed += 1
            enqueue_fill(ctx, flags, Float32(0.0))
            for rd in range(mm - 1):
                ctx.enqueue_function[rs_round_kernel](
                    _p(rt), _p(vt), _p(flags), Int32(n), Int32(mm), Int32(rd), X_DECOMP_SVD_TOL,
                    grid_dim=h, block_dim=RS_TPB,
                )
                if rd % PJ_SYNC_ROUNDS == PJ_SYNC_ROUNDS - 1:
                    ctx.synchronize()
            _down(ctx, flags, F32Ptr(unsafe_from_address=Int(hflags.unsafe_ptr())), h)
            ctx.synchronize()
            var any = False
            for b in range(h):
                if hflags[b] != Float32(0.0):
                    any = True
            if not any:
                converged = True
        if not converged:
            raise Error(
                "x_decomp svd: the round-robin one-sided Jacobi did not converge in " + String(X_DECOMP_SVD_SWEEPS)
                + " sweeps at n_cols = " + String(n) + ". An unconverged decomposition is not returned as if it"
                " were one; see DEVIATION 590."
            )
        ctx.enqueue_function[rs_norm_kernel](_p(rt), _p(s_buf), Int32(n), grid_dim=n, block_dim=RS_TPB)
        ctx.enqueue_function[pj_transpose_kernel](_p(vt), _p(v_buf), Int32(n), grid_dim=_pj_blocks(n * n), block_dim=PJ_TPB)
        _down(ctx, s_buf, s, n)
        _down(ctx, v_buf, v, n * n)
        ctx.synchronize()
        # a NaN left is a launch cut short (or a NaN input): refused
        for t in range(n):
            if s.unsafe_load(t) != s.unsafe_load(t):
                raise Error("x_decomp svd: singular value " + String(t) + " of " + String(n)
                            + " is NaN after the solve (a device launch cut short, or a NaN input): refused")
        _ = da^
        _ = scratch^
        _ = r_buf^
        _ = rt^
        _ = vt^
        _ = v_buf^
        _ = s_buf^
        _ = flags^
        _ = hflags^
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
    def lu_aux(
        lu: F32Ptr, piv: I32Ptr, pm: F32Ptr, im: F32Ptr, diag: F32Ptr, stats: F32Ptr, n: Int, clamp: Int
    ) raises:
        """An LU factor's companions on the device: pm / im the swaps' row
        order and its inverse (`lu_perm_kernel`), stats = (max |u_ii|, zero
        pivots, negative pivots, swaps), diag = u_ii, and with `clamp` the
        pivots under eps max |u_jj| floored (lu written back)."""
        var ctx = xd_ctx()
        var nb = _pj_off_blocks(n)
        var dl = _up(ctx, lu, n * n)
        var dp = _up_i(ctx, piv, n)
        var dpm = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
        var dim = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
        var dd = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
        var dpart = ctx.enqueue_create_buffer[DType.float32](4 * nb)
        var dst = ctx.enqueue_create_buffer[DType.float32](4)
        var pp = I32Ptr(unsafe_from_address=Int(dp.unsafe_ptr()))
        ctx.enqueue_function[lu_perm_kernel](pp, _p(dpm), Int32(n), Int32(0), grid_dim=_blocks(n), block_dim=TPB)
        ctx.enqueue_function[lu_perm_kernel](pp, _p(dim), Int32(n), Int32(1), grid_dim=_blocks(n), block_dim=TPB)
        ctx.enqueue_function[lu_aux_part_kernel](_p(dl), pp, _p(dpart), Int32(n), grid_dim=nb, block_dim=RR_OFF_TPB)
        ctx.enqueue_function[lu_aux_fold_kernel](_p(dpart), _p(dst), Int32(nb), grid_dim=1, block_dim=RR_OFF_TPB)
        ctx.enqueue_function[lu_aux_clamp_kernel](
            _p(dl), _p(dd), _p(dst), Int32(n), Int32(clamp), grid_dim=_blocks(n), block_dim=TPB
        )
        _down(ctx, dpm, pm, n)
        _down(ctx, dim, im, n)
        _down(ctx, dd, diag, n)
        _down(ctx, dst, stats, 4)
        if clamp != 0:
            _down(ctx, dl, lu, n * n)
        ctx.synchronize()
        _ = dl^
        _ = dp^
        _ = dpm^
        _ = dim^
        _ = dd^
        _ = dpart^
        _ = dst^
        _ = ctx^

    @staticmethod
    def lars_rows(g: F32Ptr, q: F32Ptr, w: F32Ptr, na: F32Ptr, n: Int, k: Int, m: Int, nnz: Int) raises:
        """sparse_encode 'lars': one thread a row (`lars_row`)."""
        var ctx = xd_ctx()
        var per = k * k + LARS_ROW_EXTRA * k
        var dg = _up(ctx, g, k * k)
        var dq = _up(ctx, q, n * k)
        var dw = ctx.enqueue_create_buffer[DType.float32](n * k if n * k > 0 else 1)
        var ds = ctx.enqueue_create_buffer[DType.float32](n * per if n * per > 0 else 1)
        var dn = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        ctx.enqueue_function[lars_rows_kernel](
            dg.unsafe_ptr(), dq.unsafe_ptr(), dw.unsafe_ptr(), ds.unsafe_ptr(), dn.unsafe_ptr(), Int32(n), Int32(k),
            Int32(m), Int32(nnz), grid_dim=_blocks(n), block_dim=TPB,
        )
        _down(ctx, dw, w, n * k)
        _down(ctx, dn, na, n)
        ctx.synchronize()
        _ = dg^
        _ = dq^
        _ = dw^
        _ = ds^
        _ = dn^
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
        launch_lda_rows(ctx, _p(dx), _p(dw), _p(dd), _p(de), _p(ds), _p(di), n, k, v, prior, max_iter, tol)
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
        var ns = als_scratch(n, f)
        var ds = ctx.enqueue_create_buffer[DType.float32](ns if ns > 0 else 1)
        var df = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        launch_als_rows(ctx, _p(dc), _p(dy), _p(dg), _p(dx), _p(ds), _p(df), n, m, f, m, 1, reg)
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
        """Every fold over slices of rows, a fixed tree (x_decomp/
        qr_sliced.mojo; lane hr-qr), on every column at every size."""
        comptime if XD_FAST_APPLE:
            # -D MOJOLEARN_QR_FAST_DEV (lane/apple-fast-decomp-linalg, 2026-10-02):
            # the grid-fold route of x_decomp/fast_qr.mojo, on the device,
            # ahead of the sliced route below. FAST on Apple only.
            if fast_qr_on():
                DevExec._geqrf_fast(a, tau, m, n)
                return
        var ctx = xd_ctx()
        var kk = m if m < n else n
        var da = _up(ctx, a, m * n)
        var dt = ctx.enqueue_create_buffer[DType.float32](kk if kk > 0 else 1)
        qs_geqrf_device(ctx, da, dt, m, n)
        _down(ctx, da, a, m * n)
        _down(ctx, dt, tau, kk)
        ctx.synchronize()
        _ = da^
        _ = dt^
        _ = ctx^

    @staticmethod
    def orgqr(h: F32Ptr, tau: F32Ptr, q: F32Ptr, m: Int, n: Int, kk: Int, qc: Int) raises:
        comptime if XD_FAST_APPLE:
            # -D MOJOLEARN_QR_FAST_DEV (lane/apple-fast-decomp-linalg): see geqrf.
            if fast_qr_on():
                DevExec._orgqr_fast(h, tau, q, m, n, kk, qc)
                return
        var ctx = xd_ctx()
        var dh = _up(ctx, h, m * n)
        var dt = _up(ctx, tau, kk if kk > 0 else 1)
        var dq = ctx.enqueue_create_buffer[DType.float32](m * qc if m * qc > 0 else 1)
        qs_orgqr_device(ctx, dh, dt, dq, m, n, kk, qc)
        _down(ctx, dq, q, m * qc)
        ctx.synchronize()
        _ = dh^
        _ = dt^
        _ = dq^
        _ = ctx^

    @staticmethod
    def _geqrf_fast(a: F32Ptr, tau: F32Ptr, m: Int, n: Int) raises:
        """`geqrf` with every fold a grid reduction (x_decomp/fast_qr.mojo;
        -D MOJOLEARN_QR_FAST_DEV, FAST on Apple, lane/apple-fast-decomp-linalg
        2026-10-02). Written against the host walk the Apple column once
        took (7.7 s on the M3 Ultra for the board's qr lane at 1,000,000
        x d; gone from main with lane hr-qr): the A/B arm now races main's
        sliced device route below and, from the linalg door, the blocked
        TSQR (x_decomp/tsqr_device.mojo). Step k: the
        norm's (scale, ssq) pairs over `fq_head_blocks` blocks and a
        one-block fold of the pairs (dlarfg's tau and beta on its thread 0), the scale kernel, the
        reflector products as row-chunk partials folded by `fold_kernel`,
        then the elementwise update: 5 launches a column, all grid."""
        var ctx = xd_ctx()
        var kk = m if m < n else n
        var da = _up(ctx, a, m * n)
        var dt = ctx.enqueue_create_buffer[DType.float32](kk if kk > 0 else 1)
        var ds = ctx.enqueue_create_buffer[DType.float32](2)
        var dw = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
        var nbh = fq_head_blocks(m)
        var dph = ctx.enqueue_create_buffer[DType.float32](2 * nbh)
        var nbd = fq_dot_blocks(m, 0)
        var dpd = ctx.enqueue_create_buffer[DType.float32](nbd * n if n > 0 else 1)
        for k in range(kk):
            ctx.enqueue_function[fq_head_part_kernel](
                da.unsafe_ptr(), dph.unsafe_ptr(), Int32(k), Int32(m), Int32(n), Int32(nbh),
                grid_dim=nbh, block_dim=FQ_TPB,
            )
            ctx.enqueue_function[fq_head_finish_kernel](
                da.unsafe_ptr(), dt.unsafe_ptr(), ds.unsafe_ptr(), dph.unsafe_ptr(), Int32(k), Int32(k * n + k), Int32(nbh),
                grid_dim=1, block_dim=FQ_TPB,
            )
            if m - k - 1 > 0:
                ctx.enqueue_function[geqrf_scale_kernel](
                    da.unsafe_ptr(), ds.unsafe_ptr(), Int32(k), Int32(m), Int32(n), grid_dim=_blocks(m - k - 1), block_dim=TPB
                )
            if n - k - 1 > 0:
                var nb = fq_dot_blocks(m, k)
                ctx.enqueue_function[fq_geqrf_dot_kernel](
                    da.unsafe_ptr(), ds.unsafe_ptr(), dpd.unsafe_ptr(), Int32(k), Int32(m), Int32(n),
                    grid_dim=nb, block_dim=FQ_TPB,
                )
                ctx.enqueue_function[fold_kernel](
                    dpd.unsafe_ptr(), dw.unsafe_ptr(), Int32(n), Int32(nb), grid_dim=_blocks(n), block_dim=TPB
                )
                ctx.enqueue_function[geqrf_update_kernel](
                    da.unsafe_ptr(), dt.unsafe_ptr(), ds.unsafe_ptr(), dw.unsafe_ptr(), Int32(k), Int32(m), Int32(n),
                    grid_dim=_blocks((m - k) * (n - k - 1)), block_dim=TPB,
                )
        _down(ctx, da, a, m * n)
        _down(ctx, dt, tau, kk)
        ctx.synchronize()
        _ = da^
        _ = dt^
        _ = ds^
        _ = dw^
        _ = dph^
        _ = dpd^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def _orgqr_fast(h: F32Ptr, tau: F32Ptr, q: F32Ptr, m: Int, n: Int, kk: Int, qc: Int) raises:
        """`orgqr` with the reflector products as row-chunk grid partials
        (`fq_orgqr_dot_kernel` + `fold_kernel`); the init and update kernels
        are the shipped ones. -D MOJOLEARN_QR_FAST_DEV, FAST on Apple
        (lane/apple-fast-decomp-linalg, 2026-10-02); cause as `_geqrf_fast`."""
        var ctx = xd_ctx()
        var dh = _up(ctx, h, m * n)
        var dt = _up(ctx, tau, kk if kk > 0 else 1)
        var dq = ctx.enqueue_create_buffer[DType.float32](m * qc if m * qc > 0 else 1)
        var dw = ctx.enqueue_create_buffer[DType.float32](qc if qc > 0 else 1)
        var nbd = fq_dot_blocks(m, 0)
        var dpd = ctx.enqueue_create_buffer[DType.float32](nbd * qc if qc > 0 else 1)
        ctx.enqueue_function[orgqr_init_kernel](dq.unsafe_ptr(), Int32(m), Int32(qc), grid_dim=_blocks(m * qc), block_dim=TPB)
        for r in range(kk):
            var k = kk - 1 - r
            if qc > 0:
                var nb = fq_dot_blocks(m, k)
                ctx.enqueue_function[fq_orgqr_dot_kernel](
                    dh.unsafe_ptr(), dq.unsafe_ptr(), dpd.unsafe_ptr(), Int32(k), Int32(m), Int32(n), Int32(qc),
                    grid_dim=nb, block_dim=FQ_TPB,
                )
                ctx.enqueue_function[fold_kernel](
                    dpd.unsafe_ptr(), dw.unsafe_ptr(), Int32(qc), Int32(nb), grid_dim=_blocks(qc), block_dim=TPB
                )
            ctx.enqueue_function[orgqr_update_kernel](
                dh.unsafe_ptr(), dt.unsafe_ptr(), dq.unsafe_ptr(), dw.unsafe_ptr(), Int32(k), Int32(m), Int32(n), Int32(qc),
                grid_dim=_blocks((m - k) * qc), block_dim=TPB,
            )
        _down(ctx, dq, q, m * qc)
        ctx.synchronize()
        _ = dh^
        _ = dt^
        _ = dq^
        _ = dw^
        _ = dpd^
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
        comptime if XD_FAST_APPLE:
            # -D MOJOLEARN_FA_FAST_QRR (lane/apple-fast-decomp-linalg, 2026-10-02,
            # FAST on Apple only): the matrix uploaded straight from the
            # caller's floats and `qr_factor` run on it, R downloaded once.
            # Cause: the route below copies all m x n values into a host
            # List one `append` at a time (FactorAnalysis's `k.qr_r(Xc)` at
            # 1,000,000 x 220: 220 million host appends), and
            # `device_qr_r` then uploads that copy. Same kernels, same
            # slice count; only the host copy goes.
            if is_defined["MOJOLEARN_FA_FAST_QRR"]() and m >= n and n > 0:
                var ctx = xd_ctx()
                var da = _up(ctx, a, m * n)
                var scratch = ctx.enqueue_create_buffer[DType.float32](qr_slice_count(m, n) * n * n)
                var r_buf = ctx.enqueue_create_buffer[DType.float32](n * n)
                ctx.synchronize()
                _ = qr_factor(ctx, da, scratch, r_buf, m, n)
                _down(ctx, r_buf, r, n * n)
                ctx.synchronize()
                _ = da^
                _ = scratch^
                _ = r_buf^
                ctx.synchronize()
                _ = ctx^
                return
        var w = List[Float32](capacity=m * n)
        for t in range(m * n):
            w.append(a.unsafe_load(t))
        var got = device_qr_r(xd_ctx(), w, m, n)
        for t in range(n * n):
            r.unsafe_store(t, got[t])

    @staticmethod
    def tsqr_factor(a: F32Ptr, b: F32Ptr, r: F32Ptr, m: Int, d: Int, nrhs: Int, keep: Bool) raises:
        """The blocked TSQR on the device (x_decomp/tsqr_device.mojo)."""
        var ctx = xd_ctx()
        var n = d + nrhs
        if nrhs == 0:
            var da = _up(ctx, a, m * n)
            ts_factor_device(ctx, da, m, n, r, keep)
            _ = da^
        else:
            var ta = _up(ctx, a, m * d)
            var tb = _up(ctx, b, m * nrhs)
            var da = ts_pack_device(ctx, ta, tb, m, d, nrhs)
            ctx.synchronize()
            _ = ta^
            _ = tb^
            ts_factor_device(ctx, da, m, n, r, keep)
            _ = da^
        ctx.synchronize()
        _ = ctx^

    @staticmethod
    def tsqr_apply(c: F32Ptr, q: F32Ptr, m: Int, n: Int, k: Int) raises:
        if k == 0:
            ts_free_device()
            return
        var ctx = xd_ctx()
        var dq = ts_apply_device(ctx, c, m, n, k)
        _down(ctx, dq, q, m * k)
        ctx.synchronize()
        _ = dq^
        _ = ctx^

    @staticmethod
    def vendor() -> String:
        return String(COMPILED_VENDOR)

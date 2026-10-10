# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-q-linalg (2026-10-04): FAST quality repairs of the dense
linear algebra rows the board audit flagged
(~/mojolearn-evidence/board-quality-audit-2026-10-04.md). FAST only: none of
these is compiled into an IDENTICAL binding, and no IDENTICAL kernel changes.
Python reads which are on through `x_decomp_qfix_flags()`:

  bit 1  SVD_QFIX   (REVERTED 2026-10-04, DELETED 2026-10-09 with its old
         name SVD_QOLD; the bit is always 0; record kept below)
         `svd(full_matrices=False)` of a tall matrix (the TSQR route): U_R
         keeps every direction with s_j > 2^-40 s_0 (was 2^-20) and is
         orthonormalized by Householder QR. The 2^-20 cut replaced genuine
         small singular directions with arbitrary complement columns, an
         error of up to 2 s_j each in U S V^T. Audit: svd istella
         relative_reconstruction_error_100k_rows 3.84e-05 (numpy 4.10e-08),
         taxi 1.83e-06 (numpy 4.31e-08). Mechanism reproduced in float32
         numpy (~/mojolearn-evidence/q-linalg/sim_svd2.py: 2^-20 2.1e-06,
         2^-30 and below 3.2e-07 at 100k x 220).
         OUTCOME (M3 afc_ab_def, full board size, 1 run per arm, 2026-10-04,
         tag rab5-svd): no quality gain (istella and taxi
         max_rel_singular_value_error and reconstruction unchanged), taxi
         48.54 -> 54.96 ms (+13.2%), istella +0.3%. REVERT: the old route is
         the default again.
  bit 2  TSVD_QFIX  (`-D MOJOLEARN_TSVD_QOLD`)
         TruncatedSVD 'covariance_eigh' / 'jacobi' on a tall matrix (n <= 512
         columns): the components are the top right singular vectors of the
         TSQR R (the one-sided Jacobi of R), not the eigenvectors of the
         float32 Gram X^T X, whose rounding (about eps lambda_0 per entry)
         swamps every direction under ~1e-5 lambda_0. Audit: tsvd istella
         relative_reconstruction_error 2.55e-03 (sklearn 1.22e-04).
         OUTCOME (M3 afc_ab_def, full board size, 1 run per arm, 2026-10-04,
         tag rab8-tsvd): istella relative_reconstruction_error 2.554e-3 ->
         1.219e-4 (sklearn 1.22e-4), time 362.9 -> 864.1 ms; taxi
         reconstruction unchanged 3.257e-3, 34.5 -> 42.2 ms. KEEP as the FAST
         default for quality (the old route was worse than the opponent);
         speed follow-up owed in lane apple-fast-s-linalg.
  bit 4  LU_QFIX    (`-D MOJOLEARN_LU_QOLD`)
         `solve` and `lu_solve(lu_factor(A), B)`: one step of iterative
         refinement, x += LU^-1 (B - A x), the residual folded in float-float
         (`lu_resid_ff_kernel`, below) so it is accurate past float32.
         Audit: lu-solve / lu-factor synthetic relative_residual 3.26e-06
         (numpy 3.26e-08, torch-gpu 8.23e-07). The float32 trailing updates
         of an 8192 x 8192 LU round each entry hundreds of times; a float32
         residual would carry the same ~3e-6 error, hence float-float.
         OUTCOME (M3 afc_ab_def, full board size, 1 run per arm, 2026-10-04,
         tag rab5-lu): relative_residual 2.59e-6 -> 3.26e-8 (numpy 3.26e-8)
         on lu-factor and lu-solve synthetic, time 787.6 -> 906.6 ms (+15.1%)
         and 788.6 -> 896.3 ms (+13.7%). KEEP as the FAST default for
         quality.
"""
from std.gpu import block_idx, thread_idx
from std.math import fma
from std.memory import stack_allocation
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.sys.compile import is_defined
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, pinned_mul_f32
from x_decomp.cells import F32Ptr
from x_decomp.device import _down, _p, _up, xd_ctx

# TOMBSTONE: MOJOLEARN_SVD_QFIX (DROPPED-slower) deleted 2026-10-09 by lane/owed-deletions-D3; code recoverable at b639a2bd2.
# Tried: svd's TSQR route keeping every direction above 2^-40 s_0 (was 2^-20) with a Householder QR of U_R; svd taxi 48.54 -> 54.96 ms (+13.2%), istella +0.3%, no quality gain (rab5-svd).
# Restore: git apply experiments/removed/MOJOLEARN_SVD_QFIX.patch; record in docs/TOMBSTONES.md.
comptime TSVD_QFIX = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and not is_defined["MOJOLEARN_TSVD_QOLD"]()
comptime LU_QFIX = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and not is_defined["MOJOLEARN_LU_QOLD"]()

#: threads per residual block (one block per row and QF_RT right-hand sides)
comptime QF_TPB = 128
#: right-hand sides per block (each thread holds QF_RT float-float sums)
comptime QF_RT = 8
#: multiply-adds per launch (macOS silently cuts a long Metal command buffer)
comptime QF_LAUNCH_CELLS = 1 << 30


def qfix_flags_py() raises -> PythonObject:
    """Bit 2 TSVD_QFIX, bit 4 LU_QFIX (0 in IDENTICAL); bit 1 (SVD_QFIX) was
    deleted 2026-10-09 and is always 0."""
    var f = 0
    comptime if TSVD_QFIX:
        f |= 2
    comptime if LU_QFIX:
        f |= 4
    return PythonObject(f)


@always_inline
def _two_sum(a: Float32, b: Float32) -> SIMD[DType.float32, 2]:
    """Knuth's two-sum: (s, e) with s + e == a + b exactly."""
    var s = a + b
    var bb = s - a
    var e = (a - (s - bb)) + (b - bb)
    return SIMD[DType.float32, 2](s, e)


def lu_resid_ff_kernel(a: F32Ptr, x: F32Ptr, b: F32Ptr, r: F32Ptr, n_in: Int32, nrhs_in: Int32, row0_in: Int32):
    """r[i, c] = b[i, c] - sum_j a[i, j] x[j, c] for row i = row0 + block_idx.x
    and right-hand sides c0 .. c0 + QF_RT - 1, c0 = block_idx.y QF_RT; a is
    n x n, x, b and r n x nrhs, all row major. Each lane folds its j's (a
    stride of QF_TPB) as Ogita-Rump-Oishi Dot2: the product's exact error by
    fma, the sum's by two-sum, both carried in the low word; the lanes'
    (hi, lo) pairs are combined by a shared-memory tree, then b - (hi + lo)
    is rounded once."""
    var n = Int(n_in)
    var nrhs = Int(nrhs_in)
    var i = Int(row0_in) + Int(block_idx.x)
    var c0 = Int(block_idx.y) * QF_RT
    var t = Int(thread_idx.x)
    var sh = stack_allocation[QF_TPB * QF_RT, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sl = stack_allocation[QF_TPB * QF_RT, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var hi = SIMD[DType.float32, QF_RT](0.0)
    var lo = SIMD[DType.float32, QF_RT](0.0)
    if i < n:
        var j = t
        while j < n:
            var aij = a.unsafe_load(i * n + j)
            comptime for q in range(QF_RT):
                var c = c0 + q
                if c < nrhs:
                    var xv = x.unsafe_load(j * nrhs + c)
                    var p = pinned_mul_f32(aij, xv)
                    var pe = fma(aij, xv, -p)
                    var s = _two_sum(hi[q], p)
                    hi[q] = s[0]
                    lo[q] = lo[q] + (s[1] + pe)
            j += QF_TPB
    comptime for q in range(QF_RT):
        sh[t * QF_RT + q] = hi[q]
        sl[t * QF_RT + q] = lo[q]
    barrier()
    var w = QF_TPB // 2
    while w > 0:
        if t < w:
            comptime for q in range(QF_RT):
                var s = _two_sum(sh[t * QF_RT + q], sh[(t + w) * QF_RT + q])
                sh[t * QF_RT + q] = s[0]
                sl[t * QF_RT + q] = sl[t * QF_RT + q] + sl[(t + w) * QF_RT + q] + s[1]
        barrier()
        w = w // 2
    if t == 0 and i < n:
        comptime for q in range(QF_RT):
            var c = c0 + q
            if c < nrhs:
                var bv = b.unsafe_load(i * nrhs + c)
                var d = _two_sum(bv, -sh[q])
                r.unsafe_store(i * nrhs + c, d[0] + (d[1] - sl[q]))


def lu_resid_py(a: PythonObject, x: PythonObject, b: PythonObject, r: PythonObject, p: PythonObject) raises -> PythonObject:
    """p = [n, nrhs]: host r = b - a x (float-float residual, rounded once);
    a n x n, x / b / r n x nrhs, row major host float32 buffers."""
    var n = Int(py=p[0])
    var nrhs = Int(py=p[1])
    if n <= 0 or nrhs <= 0 or n * n > 2147483647 or n * nrhs > 2147483647:
        raise Error("x_decomp: lu_resid needs 0 < n, nrhs and n * n inside Int32")
    var pa = F32Ptr(unsafe_from_address=Int(py=a))
    var px = F32Ptr(unsafe_from_address=Int(py=x))
    var pb = F32Ptr(unsafe_from_address=Int(py=b))
    var pr = F32Ptr(unsafe_from_address=Int(py=r))
    with GILReleased(Python()):
        var ctx = xd_ctx()
        var da = _up(ctx, pa, n * n)
        var dx = _up(ctx, px, n * nrhs)
        var db = _up(ctx, pb, n * nrhs)
        var dr = ctx.enqueue_create_buffer[DType.float32](n * nrhs)
        var cols = (nrhs + QF_RT - 1) // QF_RT
        var per = max(1, QF_LAUNCH_CELLS // max(1, n * nrhs))
        var row0 = 0
        while row0 < n:
            var rows = min(per, n - row0)
            ctx.enqueue_function[lu_resid_ff_kernel](
                _p(da), _p(dx), _p(db), _p(dr), Int32(n), Int32(nrhs), Int32(row0),
                grid_dim=(rows, cols, 1), block_dim=(QF_TPB, 1, 1),
            )
            ctx.synchronize()
            row0 += rows
        _down(ctx, dr, pr, n * nrhs)
        ctx.synchronize()
        _ = da^
        _ = dx^
        _ = db^
        _ = dr^
    return PythonObject(n)

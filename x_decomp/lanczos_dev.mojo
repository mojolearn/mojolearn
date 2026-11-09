# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-gap-linalg2-kpca (2026-10-03): KPCA_FAST_LANCZOS_DEV,
the Lanczos steps of `_expansion_decomp._lanczos_top` on the device with no
host read inside a batch (every mode and vendor since lane cpu2-l8-decomp;
the host column: x_decomp/lanczos_host.mojo).

The Python loop paid, per Lanczos step, a download of q (QT grows on the
host), an upload of the whole basis Q_j twice (a fresh host `_M` each
product), and three scalar reads (alpha's two halves and beta's dot), each a
device drain: KernelPCA at 10,000 rows spent most of its fit there. Here the
basis lives in one pooled device matrix (row j = q_j), and steps j0 .. j1-1
run as enqueued launches, the same products in the same order as the
Python step (`launch_gemm`, the kit's `mm`):

    w = A q_j;  c = Q_j w;  alpha = c[j];  w -= Q_j^T c;
    c = Q_j w;  alpha += c[j];  w -= Q_j^T c;  beta = sqrt(max(w . w, 0));
    q_{j+1} = w / beta   (0 when beta <= 1e-30 max(1, |alpha|): a breakdown,
                          which the host finds in the betas and stops at)

alpha and beta land in a device array (alphas at [0, cap), betas at
[cap, 2 cap)) the host reads ONCE per batch. Differences from the old Python
step: alpha's two halves add in float32 (Python added them in float64) and
1 / beta is a float32 reciprocal; both are below the route's 1e-7 Ritz
residual test, which still decides convergence on the host.
"""

from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from std.gpu import block_dim, block_idx, thread_idx
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from x_decomp.cells import F32Ptr
from x_decomp.device import TPB, _blocks, gemm_scratch, launch_gemm, xd_ctx
from x_decomp.lanczos_step import lz_alpha, lz_beta, lz_inv, lz_q, lz_w
from x_decomp.resident import X_DECOMP_POOL, _ptr, pool_alloc, pool_free
from x_decomp.eigh_ops import eigh_op_tridiag
from core.step_gate import step_gate_open_word
from max.gpu.host import DeviceBuffer

#: The FAST + Apple default since 2026-10-03 (M3 A/B, one run per arm:
#: kernel-pca taxi -9%, istella -12.5%, tags gl2k-kpca-lzdev-taxi/-istella;
#: the board reports no kernel-pca quality, so the M2 quality-only check
#: gl2k-kpca-quality-r3: eigenvalues max relative difference 1.795e-07,
#: transform subspace angle 1.879e-06 rad against arm A).
#: Lane cpu2-l8-decomp (2026-10-04, re-audit L8 `_lanczos_top`): THE route,
#: in every mode on every vendor. The host-driven Python loop (the basis
#: downloaded and re-uploaded every step) is deleted; the host column runs
#: x_decomp/lanczos_host.mojo, the same steps through HostExec's gemm and the
#: same scalar functions (x_decomp/lanczos_step.mojo), so the words agree.
#: The scalar steps moved to the cells' flush-to-zero arithmetic (alpha's
#: halves add in float32, beta through the correctly rounded sqrt, 1 / beta
#: through the correctly rounded division): FAST Apple's words change.
comptime KPCA_FAST_LANCZOS_DEV = True


def lz_sub_kernel(w: F32Ptr, t: F32Ptr, ab: F32Ptr, c: F32Ptr, j: Int32, add: Int32, n: Int32):
    """w -= t; thread 0 also folds c[j] into alpha[j] (stored, or added on
    the second pass)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        w.unsafe_store(i, lz_w(w.unsafe_load(i), t.unsafe_load(i)))
    if i == 0:
        ab.unsafe_store(Int(j), lz_alpha(ab.unsafe_load(Int(j)), c.unsafe_load(Int(j)), Int(add) != 0))


def lz_scale_kernel(w: F32Ptr, dot: F32Ptr, ab: F32Ptr, q: F32Ptr, j: Int32, cap: Int32, n: Int32):
    """beta[j] and q_{j+1} = w / beta (every thread forms the same beta;
    thread 0 stores it)."""
    var b = lz_beta(dot.unsafe_load(0))
    var s = lz_inv(b, ab.unsafe_load(Int(j)))
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i == 0:
        ab.unsafe_store(Int(cap) + Int(j), b)
    if i < Int(n):
        q.unsafe_store(i, lz_q(w.unsafe_load(i), s))


def dev_lanczos_run(ia: Int, iq: Int, iab: Int, n: Int, j0: Int, j1: Int, cap: Int) raises -> Int:
    """`dev_lanczos_py`'s typed body: Lanczos steps j0 .. j1-1 (see the module
    docstring), enqueued, on pool buffers ia (n x n), iq (the basis, cap + 1
    rows) and iab (2 cap floats); returns the step count."""
    if j1 > cap or j0 > j1 or n < 1:
        raise Error("x_decomp: dev_lanczos steps out of range")
    var pa = _ptr(ia, n * n)
    var pq = _ptr(iq, (cap + 1) * n)
    var pab = _ptr(iab, 2 * cap)
    var ns = max(gemm_scratch(n, n, 1), gemm_scratch(cap, n, 1), gemm_scratch(n, cap, 1), gemm_scratch(1, n, 1), 1)
    var wid = pool_alloc(n)
    var tid = pool_alloc(n)
    var cid = pool_alloc(max(cap, 1))
    var did = pool_alloc(1)
    var sid = pool_alloc(ns)
    var pw = _ptr(wid, n)
    var pt = _ptr(tid, n)
    var pc = _ptr(cid, cap)
    var pd = _ptr(did, 1)
    var ps = _ptr(sid, ns)
    var ctx = xd_ctx()
    for j in range(j0, j1):
        var qj = pq + j * n
        launch_gemm(ctx, pa, qj, pw, ps, n, n, 1, False, False)
        for h in range(2):
            launch_gemm(ctx, pq, pw, pc, ps, j + 1, n, 1, False, False)
            launch_gemm(ctx, pq, pc, pt, ps, n, j + 1, 1, True, False)
            ctx.enqueue_function[lz_sub_kernel](pw, pt, pab, pc, Int32(j), Int32(h), Int32(n), grid_dim=_blocks(n), block_dim=TPB)
        launch_gemm(ctx, pw, pw, pd, ps, 1, n, 1, True, False)
        ctx.enqueue_function[lz_scale_kernel](
            pw, pd, pab, pq + (j + 1) * n, Int32(j), Int32(cap), Int32(n), grid_dim=_blocks(n), block_dim=TPB
        )
    pool_free(sid)
    pool_free(did)
    pool_free(cid)
    pool_free(tid)
    pool_free(wid)
    return j1 - j0


#: lane/apple-fast-gap-linalg2-kpca: IPCA_FAST_DEV, the FAST + Apple default
#: (M3 A/B, one run per arm: incremental-pca taxi 159 -> 136 ms, output digest
#: bit-identical; tag gl2k-ipca-dev-taxi). -D MOJOLEARN_IPCA_FAST_DEV_OFF
#: restores the host stack (the A/B arm).
#: No kernel of its own: IncrementalPCA.fit (python/mojolearn/_expansion_decomp.py)
#: reads this constant and stacks each batch on the device (`_Kit.vstack_dev`,
#: the PLACE_COLS move) instead of downloading the centered batch to stack it
#: on the host and uploading it again, and reads its public arrays once after
#: the last batch instead of after every batch.
comptime IPCA_FAST_DEV = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_IPCA_FAST_DEV_OFF"]()
)


def dev_eigh_tridiag_run(iab: Int, iw: Int, iz: Int, j: Int, cap: Int) raises -> Int:
    """The Lanczos projected solve on the device (lane n9-decomp-eigh-families,
    2026-11-09): T of the batch's own alphas (ab[0, j)) and betas (ab[cap,
    cap + j - 1)), read where `dev_lanczos_run` wrote them, through the
    executor op x_decomp/eigh_ops.mojo `eigh_op_tridiag` (bisection and
    twisted vectors, the Jacobi of T on a refusal) into the pool matrices w
    (1 x j, ascending) and z (j x j, eigenvector c in column c,
    sign-flipped). The host column runs `eigh_op_tridiag_host`
    (x_decomp/lanczos_host.mojo `eigh_tridiag_host_run`). One sync, for the
    status word: a solve that holds on neither route raises.

    Before, the driver (bindings/mani_decomp_native.mojo) assembled T as a
    j x j host matrix and ran the dense eigh on it (an upload of j^2 floats
    and the dense route per basis round)."""
    if j < 1 or j > cap:
        raise Error("x_decomp: the Lanczos projected solve needs 1 <= j <= cap, got j=" + String(j) + " cap=" + String(cap))
    _ = _ptr(iab, 2 * cap)
    _ = _ptr(iw, j)
    _ = _ptr(iz, j * j)
    var pool = X_DECOMP_POOL.get_or_create_ptr()
    var ctx = xd_ctx()
    var d = pool[].bufs[iab].create_sub_buffer[DType.float32](0, j)
    var e = pool[].bufs[iab].create_sub_buffer[DType.float32](cap, max(j - 1, 1))
    var w = pool[].bufs[iw].create_sub_buffer[DType.float32](0, j)
    var z = pool[].bufs[iz].create_sub_buffer[DType.float32](0, j * j)
    var gate = step_gate_open_word(ctx)
    var status = ctx.enqueue_create_buffer[DType.int32](1)
    eigh_op_tridiag(ctx, gate, d, e, w, z, status, j)
    var hs = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=status)
    ctx.synchronize()
    var bad = Int(hs.unsafe_ptr().unsafe_load(0))
    _ = d^
    _ = e^
    _ = w^
    _ = z^
    _ = gate^
    _ = status^
    _ = hs^
    if bad != 0:
        raise Error(
            "x_decomp: the Lanczos projected tridiagonal solve (j = " + String(j) + ") did not hold on the"
            " bisection route nor on its Jacobi fallback. An unconverged decomposition is not returned as if"
            " it were one (DEVIATION 590)."
        )
    return j

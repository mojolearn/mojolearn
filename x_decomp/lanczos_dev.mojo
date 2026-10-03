# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-gap-linalg2-kpca (2026-10-03): -D MOJOLEARN_KPCA_FAST_LANCZOS_DEV,
the Lanczos steps of `_expansion_decomp._lanczos_top` on the device with no
host read inside a batch (FAST on Apple only).

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
[cap, 2 cap)) the host reads ONCE per batch. Differences from the Python
step: alpha's two halves add in float32 (Python added them in float64) and
1 / beta is a float32 reciprocal; both are below the route's 1e-7 Ritz
residual test, which still decides convergence on the host.
"""

from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from std.math import sqrt
from std.gpu import block_dim, block_idx, thread_idx
from std.python import PythonObject
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from x_decomp.cells import F32Ptr
from x_decomp.device import TPB, _blocks, gemm_scratch, launch_gemm, xd_ctx
from x_decomp.resident import _id, _n, _ptr, pool_alloc, pool_free

comptime KPCA_FAST_LANCZOS_DEV = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_KPCA_FAST_LANCZOS_DEV"]()
)


def lz_sub_kernel(w: F32Ptr, t: F32Ptr, ab: F32Ptr, c: F32Ptr, j: Int32, add: Int32, n: Int32):
    """w -= t; thread 0 also folds c[j] into alpha[j] (stored, or added on
    the second pass)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        w.unsafe_store(i, w.unsafe_load(i) - t.unsafe_load(i))
    if i == 0:
        var v = c.unsafe_load(Int(j))
        if Int(add) != 0:
            v = ab.unsafe_load(Int(j)) + v
        ab.unsafe_store(Int(j), v)


def lz_scale_kernel(w: F32Ptr, dot: F32Ptr, ab: F32Ptr, q: F32Ptr, j: Int32, cap: Int32, n: Int32):
    """beta[j] and q_{j+1} = w / beta (every thread forms the same beta;
    thread 0 stores it)."""
    var dd = dot.unsafe_load(0)
    if not (dd > Float32(0)):
        dd = Float32(0)
    var b = sqrt(dd)
    var a = abs(ab.unsafe_load(Int(j)))
    var s = Float32(0)
    if b > Float32(1e-30) * max(Float32(1), a):
        s = Float32(1) / b
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i == 0:
        ab.unsafe_store(Int(cap) + Int(j), b)
    if i < Int(n):
        q.unsafe_store(i, w.unsafe_load(i) * s)


def kpca_lanczos_dev_on_py() raises -> PythonObject:
    """1 when this binding carries the device Lanczos route."""
    comptime if KPCA_FAST_LANCZOS_DEV:
        return PythonObject(1)
    return PythonObject(0)


def dev_lanczos_py(a: PythonObject, q: PythonObject, ab: PythonObject, p: PythonObject) raises -> PythonObject:
    """Lanczos steps j0 .. j1-1 (see the module docstring), enqueued.
    a: n x n; q: the basis, rows 0 .. j0 hold q_0 .. q_j0 on entry, room
    for cap + 1 rows; ab: 2 cap floats. p = [n, j0, j1, cap]."""
    var n = _n(p, 0)
    var j0 = _n(p, 1)
    var j1 = _n(p, 2)
    var cap = _n(p, 3)
    if j1 > cap or j0 > j1 or n < 1:
        raise Error("x_decomp: dev_lanczos steps out of range")
    var pa = _ptr(_id(a), n * n)
    var pq = _ptr(_id(q), (cap + 1) * n)
    var pab = _ptr(_id(ab), 2 * cap)
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
    return PythonObject(j1 - j0)


#: lane/apple-fast-gap-linalg2-kpca: -D MOJOLEARN_IPCA_FAST_DEV (FAST + Apple).
#: No kernel of its own: IncrementalPCA.fit (python/mojolearn/_expansion_decomp.py)
#: reads this constant and stacks each batch on the device (`_Kit.vstack_dev`,
#: the PLACE_COLS move) instead of downloading the centered batch to stack it
#: on the host and uploading it again, and reads its public arrays once after
#: the last batch instead of after every batch.
comptime IPCA_FAST_DEV = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_IPCA_FAST_DEV"]()
)


def ipca_dev_on_py() raises -> PythonObject:
    comptime if IPCA_FAST_DEV:
        return PythonObject(1)
    return PythonObject(0)

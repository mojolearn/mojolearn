# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-gap-linalg2 (2026-10-03): -D MOJOLEARN_SVD_FAST_AW, the thin
SVD's U without applying the TSQR's reflectors to every column (FAST on
Apple only).

`svd(full_matrices=False)` of a tall A runs the blocked TSQR twice over the
rows: R (the factor), then U = Q U_R (`ts_apply_device`, its reflectors
applied to an n x n C in every 4096-row block). At 1,000,000 x 220 the apply
is the larger pass (4 m n k multiply-adds against the factor's ~2 m n^2) and
both are latency-bound threadgroup chains.

Here, with S and V from the SVD of R as before:
- the r leading directions with s_j > SVD_AW_TOL s_0 take U_j = A v_j / s_j,
  one GEMM against an unfactored device copy of A stashed by the factor;
- the trailing (numerically small) directions keep Q U_R[:, r:] from the
  reflectors, an apply over only those n - r columns;
- the n columns [A W | Q U_R[:, r:]] are re-orthonormalized by one
  Cholesky QR (G = U^T U, G = L L^T, U <- U L^-T). With s_j > 2^-10 s_0 the
  A W columns are orthonormal to ~2^10 eps, so G is within ~1e-4 of I and
  the Cholesky QR is exact to float32 rounding; U S Vh still reproduces A to
  ~eps ||A|| (the triangular correction couples column j only into later,
  smaller directions).
S and Vh are the TSQR route's, unchanged. A Cholesky that fails (never on
the guard's conditioning) returns -1 with the factorization still kept, and
the caller takes the reflector route.
"""

from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from std.gpu import block_dim, block_idx, thread_idx
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from x_decomp.cells import F32Ptr
from x_decomp.device import (
    TPB, _blocks, _down, _p, _up, chol_step_kernel, gemm_scratch, launch_gemm, lu_info_init_kernel,
    trsm_kernel, xd_ctx,
)
from x_decomp.tsqr_device import TS_DEV_STATE, ts_apply_device, ts_free_device

comptime SVD_FAST_AW = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_SVD_FAST_AW"]()
)
#: directions with s_j > SVD_AW_TOL s_0 take A v_j / s_j (2^-10)
comptime SVD_AW_TOL = Float32(0.0009765625)


def aw_w_kernel(vt: F32Ptr, s: F32Ptr, w: F32Ptr, n: Int32, r: Int32):
    """w (n x r) = V[:, :r] S^-1: w[i, j] = vt[j, i] / s[j]."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var rr = Int(r)
    if t < Int(n) * rr:
        var i = t // rr
        var j = t - i * rr
        w.unsafe_store(t, vt.unsafe_load(j * Int(n) + i) / s.unsafe_load(j))


def aw_place_kernel(g: F32Ptr, qb: F32Ptr, u: F32Ptr, m: Int32, n: Int32, r: Int32):
    """u (m x n) = [g (m x r) | qb (m x (n - r))] (data movement)."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nn = Int(n)
    var rr = Int(r)
    if t < Int(m) * nn:
        var i = t // nn
        var j = t - i * nn
        u.unsafe_store(t, g.unsafe_load(i * rr + j) if j < rr else qb.unsafe_load(i * (nn - rr) + j - rr))


def aw_lt_kernel(g: F32Ptr, rt: F32Ptr, n: Int32):
    """rt (n x n, upper) = L^T of the lower Cholesky factor in g."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nn = Int(n)
    if t < nn * nn:
        var i = t // nn
        var j = t - i * nn
        rt.unsafe_store(t, g.unsafe_load(j * nn + i) if j >= i else Float32(0))


def ts_svd_aw_device(ur: F32Ptr, s: F32Ptr, vt: F32Ptr, out: F32Ptr, m: Int, n: Int) raises -> Int:
    """U (m x n) into the host `out` (see the module docstring); 0 done (the
    kept factorization released), -1 not taken (the factorization kept)."""
    var st = TS_DEV_STATE.get_or_create_ptr()
    if len(st[].acopy) != 1 or len(st[].bufs) != 4 or st[].m != m or st[].n != n:
        return -1
    var s0 = s.unsafe_load(0)
    var r = 0
    while r < n and s.unsafe_load(r) > SVD_AW_TOL * s0:
        r += 1
    if r == 0:
        return -1
    var ctx = xd_ctx()
    var da = st[].acopy[0]
    var dqb = ctx.enqueue_create_buffer[DType.float32](1)
    if r < n:
        dqb = ts_apply_device(ctx, ur, m, n, n - r, n, r, False)
    var dvt = _up(ctx, vt, n * n)
    var ds = _up(ctx, s, n)
    var dw = ctx.enqueue_create_buffer[DType.float32](n * r)
    ctx.enqueue_function[aw_w_kernel](_p(dvt), _p(ds), _p(dw), Int32(n), Int32(r), grid_dim=_blocks(n * r), block_dim=TPB)
    var dg = ctx.enqueue_create_buffer[DType.float32](m * r)
    var ns = gemm_scratch(m, n, r)
    var dsc = ctx.enqueue_create_buffer[DType.float32](ns if ns > 0 else 1)
    launch_gemm(ctx, _p(da), _p(dw), _p(dg), _p(dsc), m, n, r, False, False)
    var du = ctx.enqueue_create_buffer[DType.float32](m * n)
    ctx.enqueue_function[aw_place_kernel](
        _p(dg), _p(dqb), _p(du), Int32(m), Int32(n), Int32(r), grid_dim=_blocks(m * n), block_dim=TPB
    )
    var dgm = ctx.enqueue_create_buffer[DType.float32](n * n)
    var ns2 = gemm_scratch(n, m, n)
    var dsc2 = ctx.enqueue_create_buffer[DType.float32](ns2 if ns2 > 0 else 1)
    launch_gemm(ctx, _p(du), _p(du), _p(dgm), _p(dsc2), n, m, n, True, False)
    var dinfo = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.enqueue_function[lu_info_init_kernel](_p(dinfo), grid_dim=1, block_dim=1)
    for j in range(n):
        ctx.enqueue_function[chol_step_kernel](_p(dgm), _p(dinfo), Int32(j), Int32(n), grid_dim=_blocks(n - j), block_dim=TPB)
    var drt = ctx.enqueue_create_buffer[DType.float32](n * n)
    ctx.enqueue_function[aw_lt_kernel](_p(dgm), _p(drt), Int32(n), grid_dim=_blocks(n * n), block_dim=TPB)
    var dout = ctx.enqueue_create_buffer[DType.float32](m * n)
    ctx.enqueue_function[trsm_kernel](_p(du), _p(drt), _p(dout), Int32(m), Int32(n), grid_dim=_blocks(m), block_dim=TPB)
    var hinfo = ctx.enqueue_create_host_buffer[DType.float32](1)
    ctx.enqueue_copy(dst_buf=hinfo, src_buf=dinfo)
    ctx.synchronize()
    var info = hinfo.unsafe_ptr()[0]
    var rc = -1
    if info == Float32(0):
        _down(ctx, dout, out, m * n)
        ctx.synchronize()
        rc = 0
    _ = da^
    _ = dqb^
    _ = dvt^
    _ = ds^
    _ = dw^
    _ = dg^
    _ = dsc^
    _ = du^
    _ = dgm^
    _ = dsc2^
    _ = dinfo^
    _ = drt^
    _ = dout^
    _ = hinfo^
    ctx.synchronize()
    _ = ctx^
    if rc == 0:
        ts_free_device()
    return rc


def svd_aw_on_py() raises -> PythonObject:
    """1 when this binding was built with -D MOJOLEARN_SVD_FAST_AW (FAST +
    Apple), else 0: `_svd_tsqr` picks its route from it (no env read)."""
    comptime if SVD_FAST_AW:
        return PythonObject(1)
    return PythonObject(0)


def svd_aw_arm_py() raises -> PythonObject:
    """The next kept TSQR factorization stashes an unfactored copy of A."""
    TS_DEV_STATE.get_or_create_ptr()[].arm = True
    return PythonObject(1)


def svd_aw_py(ur: PythonObject, s: PythonObject, vt: PythonObject, out: PythonObject, p: PythonObject) raises -> PythonObject:
    """`ts_svd_aw_device`: ur (n x n, U_R), s (n, descending), vt (n x n),
    out (m x n), p = [m, n]. Returns 0 (out written) or -1 (not taken)."""
    var m = Int(py=p[0])
    var n = Int(py=p[1])
    var pu = F32Ptr(unsafe_from_address=Int(py=ur))
    var ps = F32Ptr(unsafe_from_address=Int(py=s))
    var pv = F32Ptr(unsafe_from_address=Int(py=vt))
    var po = F32Ptr(unsafe_from_address=Int(py=out))
    var rc = -1
    with GILReleased(Python()):
        rc = ts_svd_aw_device(pu, ps, pv, po, m, n)
    return PythonObject(rc)

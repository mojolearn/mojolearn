# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-s-linalg (2026-10-04): FAST + Apple speed candidates for
the kit's device routes, each default OFF behind its own define. Python
reads which are compiled through `x_decomp_s_flags()`:

  bit 1  RSVD_FAST_DEVSCAN   (`-D MOJOLEARN_RSVD_FAST_DEVSCAN`)
  bit 2  DECOMP_FAST_ORTH_WS (`-D MOJOLEARN_DECOMP_FAST_ORTH_WS`)
  bit 4  LU_FAST_RESIDENT    (`-D MOJOLEARN_LU_FAST_RESIDENT`)

IDENTICAL and every other vendor compile none of this.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from core.device_pool import pool_give, pool_take
from core.device_scan import device_first_nonfinite
from core.householder_qr import qr_factor, qr_slice_count
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.device import (
    LU_SCAL_LEN, TPB, XD_FAST_APPLE, _blocks, _down, _p, _up, _up_into, launch_lu, launch_lu_solve,
    orth_guard_kernel, trsm_kernel, xd_ctx,
)
from x_decomp.qfix import LU_QFIX, QF_LAUNCH_CELLS, QF_RT, QF_TPB, lu_resid_ff_kernel
from x_decomp.resident import X_DECOMP_POOL, _id, _n, _ptr

#: What: `randomized_svd`'s direct input (RSVD_FAST_DIRECT_IN, the FAST +
#: Apple default) refuses NaN/inf with the base binding's `all_finite_f32`, a
#: single-thread host loop with an early-exit branch over every value
#: (bindings/_mojolearn.mojo `all_finite_f32_binding`): 220 million floats
#: (880 MB) at the board's istella 1M x 220, on the host, inside the fit.
#: This uploads M into its pooled device matrix first and scans THAT buffer
#: on the device (`core/device_scan.mojo device_first_nonfinite`, one read,
#: one int per block back), raising the same ValueError (Python) when a
#: value is not finite. Same words reach the device; no arithmetic changes.
#: Expect: randomized-svd istella 501 -> ~430-460 ms (the host scan is
#: ~1 cycle a float), taxi a few ms.
comptime RSVD_FAST_DEVSCAN = XD_FAST_APPLE and is_defined["MOJOLEARN_RSVD_FAST_DEVSCAN"]()

#: What: `Kit.orth` on a device matrix (`x_decomp_dev_orth`, resident.mojo
#: `dev_orth_py` -> device.mojo `orth_on_device_diag`) copies A into the
#: output, then allocates three FRESH device buffers per call (dq and dw of
#: m x l, the TSQR scratch) and frees them; randomized_svd calls it 9 times
#: per fit (n_iter 4: 2 per power iteration + 1) on 1M x 18 (72 MB each),
#: so ~1.3 GB of fresh Metal buffers whose pages are faulted in by the GPU
#: on first touch every fit (the cost PCA_FAST_POOL removed for PCA's input:
#: pca istella 471 -> 218 ms). This runs the same two passes (copy, the same
#: `qr_factor`, the rank guard, `trsm_kernel`) with dq / dw / scratch / R
#: taken from a named pool (`core/device_pool.mojo`, returned after the
#: final wait), pass 0 reads A in place (no A -> output copy), pass 1
#: writes the output: the same kernels on the same words, so the same bits.
#: Every device `Kit.orth` caller takes it (randomized_svd, randomized PCA /
#: TruncatedSVD, FastICA's whitening, ...).
comptime DECOMP_FAST_ORTH_WS = XD_FAST_APPLE and is_defined["MOJOLEARN_DECOMP_FAST_ORTH_WS"]()


#: What: `lu_factor` + `lu_solve` (the lu-factor / lu-solve rows, 8192 x 8192,
#: LU_QFIX's refinement on) move the 268 MB matrix through the host again
#: and again (python/mojolearn/_expansion_decomp.py): `_M.from_input(a)`
#: (a host copy + the host finiteness loop), `Kit.lu`'s `A.copy()`, an upload
#: into a fresh device buffer, the factor's download, `lu.out()`'s copies;
#: then `lu_solve`'s `_M.from_input(lu)` (a copy + the host loop), an upload
#: of L, LU_QFIX's residual uploading A again and the correction solve
#: uploading L once more. Here `lu_factor` uploads A once from the caller's
#: buffer into a pooled device matrix, scans it there for NaN/inf, factors a
#: device copy (`launch_lu`, DevExec.lu's launches) and downloads LU and piv
#: once into the returned arrays, keeping A, LU and piv on the device in the
#: returned pair; `lu_solve` on that same pair (the same lu / piv objects)
#: uploads only B and runs the solve, LU_QFIX's float-float residual
#: (`lu_resid_ff_kernel`), the correction solve and the add on the device.
#: The same launches on the same words as the host-staged route.
#: Expect: lu-factor 654 -> ~450-550 ms, lu-solve 734 -> ~480-600 ms.
comptime LU_FAST_RESIDENT = XD_FAST_APPLE and LU_QFIX and is_defined["MOJOLEARN_LU_FAST_RESIDENT"]()


def s_flags_py() raises -> PythonObject:
    """Bit 1 RSVD_FAST_DEVSCAN, bit 2 DECOMP_FAST_ORTH_WS, bit 4
    LU_FAST_RESIDENT (0 when none)."""
    var f = 0
    comptime if RSVD_FAST_DEVSCAN:
        f |= 1
    comptime if DECOMP_FAST_ORTH_WS:
        f |= 2
    comptime if LU_FAST_RESIDENT:
        f |= 4
    return PythonObject(f)


def lur_add_kernel(x: F32Ptr, d: F32Ptr, count: Int32):
    """x[i] = x[i] + d[i] (the refinement's update), a thread per word."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(count):
        x.unsafe_store(i, x.unsafe_load(i) + d.unsafe_load(i))


def dev_lu_factor_py(
    a: PythonObject, lu: PythonObject, piv: PythonObject, info: PythonObject, ia: PythonObject,
    ilu: PythonObject, ipv: PythonObject, p: PythonObject,
) raises -> PythonObject:
    """p = [n]. Host a (n x n) into device matrix ia; -1 when a holds a NaN
    or infinity (nothing else done). Else device matrix ilu = the LU factor
    of a copy of it (`launch_lu`), device matrix ipv (n words) its int32
    pivots, and host lu (n x n float32), piv (n int32) and info (1 float32)
    their downloads; returns 0. Waits."""
    var n = _n(p, 0)
    if n <= 0 or n * n > 2147483647:
        raise Error("x_decomp: dev_lu_factor needs 0 < n and n * n inside Int32")
    var src = F32Ptr(unsafe_from_address=Int(py=a))
    var plu = F32Ptr(unsafe_from_address=Int(py=lu))
    var ppv = F32Ptr(unsafe_from_address=Int(py=piv))     # int32 words, moved as 4-byte floats
    var pinf = F32Ptr(unsafe_from_address=Int(py=info))
    var ida = _id(ia)
    var idl = _id(ilu)
    var idp = _id(ipv)
    _ = _ptr(ida, n * n)
    var dl = _ptr(idl, n * n)
    var dp = I32Ptr(unsafe_from_address=Int(_ptr(idp, n)))
    var pool = X_DECOMP_POOL.get_or_create_ptr()
    var rc = 0
    with GILReleased(Python()):
        var ctx = xd_ctx()
        _up_into(ctx, pool[].bufs[ida], src, n * n)
        if device_first_nonfinite(ctx, pool[].bufs[ida], n * n) >= 0:
            rc = -1
        else:
            ctx.enqueue_copy(
                dst_buf=pool[].bufs[idl].create_sub_buffer[DType.float32](0, n * n),
                src_buf=pool[].bufs[ida].create_sub_buffer[DType.float32](0, n * n),
            )
            var di = ctx.enqueue_create_buffer[DType.float32](1)
            var ds = ctx.enqueue_create_buffer[DType.float32](LU_SCAL_LEN)
            var dact = ctx.enqueue_create_buffer[DType.float32](n)
            launch_lu(ctx, dl, dp, _p(di), _p(ds), _p(dact), n)
            _down(ctx, pool[].bufs[idl], plu, n * n)
            _down(ctx, pool[].bufs[idp], ppv, n)
            _down(ctx, di, pinf, 1)
            ctx.synchronize()
            _ = di^
            _ = ds^
            _ = dact^
        ctx.synchronize()
    return PythonObject(rc)


def dev_lu_solve_py(
    ilu: PythonObject, ipv: PythonObject, ia: PythonObject, b: PythonObject, x: PythonObject, p: PythonObject
) raises -> PythonObject:
    """p = [n, nrhs, refine]. Host x (n x nrhs) = the solve of A x = b (host
    b, n x nrhs) from the device LU ilu / pivots ipv (`dev_lu_factor_py`'s),
    `launch_lu_solve` as DevExec.lu_solve runs it; with refine, one LU_QFIX
    step x += LU^-1 (b - A x), A the device matrix ia, the residual in
    float-float (`lu_resid_ff_kernel`, sliced as `lu_resid_py` slices it).
    Waits."""
    var n = _n(p, 0)
    var nrhs = _n(p, 1)
    var refine = _n(p, 2) != 0
    if n <= 0 or nrhs <= 0 or n * n > 2147483647 or n * nrhs > 2147483647:
        raise Error("x_decomp: dev_lu_solve needs 0 < n, nrhs and n * n inside Int32")
    var pb = F32Ptr(unsafe_from_address=Int(py=b))
    var px = F32Ptr(unsafe_from_address=Int(py=x))
    var dl = _ptr(_id(ilu), n * n)
    var dp = I32Ptr(unsafe_from_address=Int(_ptr(_id(ipv), n)))
    var da = _ptr(_id(ia), n * n)
    var cells = n * nrhs
    with GILReleased(Python()):
        var ctx = xd_ctx()
        var db = _up(ctx, pb, cells)
        var dbs = ctx.enqueue_create_buffer[DType.float32](cells)
        ctx.enqueue_copy(dst_buf=dbs, src_buf=db)
        var didx = ctx.enqueue_create_buffer[DType.float32](n)
        var dx = ctx.enqueue_create_buffer[DType.float32](cells)
        # launch_lu_solve uses its b as scratch: the copy keeps b for the residual
        launch_lu_solve(ctx, dl, dp, _p(dbs), _p(didx), _p(dx), n, nrhs, 0)
        if refine:
            var dr = ctx.enqueue_create_buffer[DType.float32](cells)
            var cols = (nrhs + QF_RT - 1) // QF_RT
            var per = max(1, QF_LAUNCH_CELLS // max(1, cells))
            var row0 = 0
            while row0 < n:
                var rows = min(per, n - row0)
                ctx.enqueue_function[lu_resid_ff_kernel](
                    da, _p(dx), _p(db), _p(dr), Int32(n), Int32(nrhs), Int32(row0),
                    grid_dim=(rows, cols, 1), block_dim=(QF_TPB, 1, 1),
                )
                ctx.synchronize()
                row0 += rows
            var dd = ctx.enqueue_create_buffer[DType.float32](cells)
            launch_lu_solve(ctx, dl, dp, _p(dr), _p(didx), _p(dd), n, nrhs, 0)
            ctx.enqueue_function[lur_add_kernel](_p(dx), _p(dd), Int32(cells), grid_dim=_blocks(cells), block_dim=TPB)
            ctx.synchronize()
            _ = dr^
            _ = dd^
        _down(ctx, dx, px, cells)
        ctx.synchronize()
        _ = db^
        _ = dbs^
        _ = didx^
        _ = dx^
    return PythonObject(cells)


def dev_upload_scan_py(id: PythonObject, addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """`dev_upload_py` (host floats into the device matrix `id`), then the
    first flat index of a NaN or infinity among those n device floats, or -1
    (one device read; waits)."""
    var cnt = Int(py=n)
    var src = F32Ptr(unsafe_from_address=Int(py=addr))
    var p = X_DECOMP_POOL.get_or_create_ptr()
    var i = _id(id)
    _ = _ptr(i, cnt)
    var r = -1
    if cnt <= 0:
        return PythonObject(r)
    with GILReleased(Python()):
        var ctx = xd_ctx()
        _up_into(ctx, p[].bufs[i], src, cnt)
        r = device_first_nonfinite(ctx, p[].bufs[i], cnt)
        ctx.synchronize()
    return PythonObject(r)


def orth_ws_on_device(
    ctx: DeviceContext, a: DeviceBuffer[DType.float32], dst: DeviceBuffer[DType.float32], m: Int, l: Int
) raises:
    """dst = the orthonormalized columns of a (both m x l, a unchanged):
    `orth_on_device`'s two passes with pooled work buffers; waits."""
    comptime NAME = "MojoXDecompOrthWS"
    var cells = m * l
    var ns = qr_slice_count(m, l)
    var dw = pool_take[NAME](ctx, cells)
    var dq = pool_take[NAME](ctx, cells)
    var scratch = pool_take[NAME](ctx, ns * l * l if ns > 1 else 1)
    var r_buf = pool_take[NAME](ctx, l * l)
    for ps in range(2):
        var src = a if ps == 0 else dq
        var out = dq if ps == 0 else dst
        ctx.enqueue_copy(dst_buf=dw, src_buf=src)
        _ = qr_factor(ctx, dw, scratch, r_buf, m, l)
        ctx.enqueue_function[orth_guard_kernel](  # small-launch(l: R columns): the l x l R factor, l = the matrix's column count
            r_buf.unsafe_ptr(), Int32(l), grid_dim=1, block_dim=1
        )
        ctx.enqueue_function[trsm_kernel](
            src.unsafe_ptr(), r_buf.unsafe_ptr(), out.unsafe_ptr(), Int32(m), Int32(l), grid_dim=_blocks(m), block_dim=TPB
        )
        _ = src^
        _ = out^
    # every launch that reads the pooled buffers is done before they go back
    ctx.synchronize()
    pool_give[NAME](dw^)
    pool_give[NAME](dq^)
    pool_give[NAME](scratch^)
    pool_give[NAME](r_buf^)


def dev_orth_ws_py(a: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """`dev_orth_py` (dst = the orthonormalized columns of a; a unchanged)
    with `orth_ws_on_device`."""
    var m = _n(p, 0)
    var l = _n(p, 1)
    var cells = m * l
    var ia = _id(a)
    var io = _id(dst)
    _ = _ptr(ia, cells)
    _ = _ptr(io, cells)
    if ia == io:
        raise Error("x_decomp: dev_orth_ws needs distinct input and output matrices")
    var pool = X_DECOMP_POOL.get_or_create_ptr()
    if cells > 0:
        var src = pool[].bufs[ia].create_sub_buffer[DType.float32](0, cells)
        var out = pool[].bufs[io].create_sub_buffer[DType.float32](0, cells)
        with GILReleased(Python()):
            var ctx = xd_ctx()
            orth_ws_on_device(ctx, src, out, m, l)
    return PythonObject(cells)

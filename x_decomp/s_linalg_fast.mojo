# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-s-linalg (2026-10-04): FAST + Apple speed candidates for
the kit's device routes, each default OFF behind its own define. Python
reads which are compiled through `x_decomp_s_flags()`:

  bit 1  RSVD_FAST_DEVSCAN   (`-D MOJOLEARN_RSVD_FAST_DEVSCAN`)
  bit 2  DECOMP_FAST_ORTH_WS (`-D MOJOLEARN_DECOMP_FAST_ORTH_WS`)

IDENTICAL and every other vendor compile none of this.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from core.device_pool import pool_give, pool_take
from core.device_scan import device_first_nonfinite
from core.householder_qr import qr_factor, qr_slice_count
from x_decomp.cells import F32Ptr
from x_decomp.device import TPB, XD_FAST_APPLE, _blocks, _up_into, orth_guard_kernel, trsm_kernel, xd_ctx
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


def s_flags_py() raises -> PythonObject:
    """Bit 1 RSVD_FAST_DEVSCAN, bit 2 DECOMP_FAST_ORTH_WS (0 when none)."""
    var f = 0
    comptime if RSVD_FAST_DEVSCAN:
        f |= 1
    comptime if DECOMP_FAST_ORTH_WS:
        f |= 2
    return PythonObject(f)


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

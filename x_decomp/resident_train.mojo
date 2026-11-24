# SPDX-License-Identifier: Apache-2.0
"""Training device arrays in and out of the x_decomp pool, and the TSQR on pool ids (lane vdf-linalg6, 2026-11-24).

A `TrainDeviceArray` (core/train_dev_pool.mojo, the one device-array convention of the training ops) is read by the
decompositions that run on the x_decomp kit (`mojolearn.linalg.eigh`, `qr` 'reduced' / 'complete' / 'raw' and a wide
'r', `svd` with compute_uv; bindings/linalg_ops_kit_run.mojo `LinKit.adopt`) without a host trip:

- XD_DEV_FROM_TRAIN copies a training handle's first n words device to device into an x_decomp pool id, and
  XD_DEV_TO_TRAIN copies a pool id back into a training handle (the PACK of a device result). Both run on the
  x_decomp context and synchronize before they return; the training entries synchronize their own context before
  they return (core/train_dev_pool.mojo), so a handle is complete when either side reads it.
- XD_DEV_TSQR_R / XD_DEV_TSQR_Q are XD_TSQR_R / XD_TSQR_Q (x_decomp/device.mojo `DevExec.tsqr_factor`,
  `tsqr_apply`) with X, C and Q on pool ids: X is copied device to device into the factor's own buffer (the host
  route uploads it there), the factor and the apply are the SAME calls (`ts_factor_dispatch`, `ts_apply_device*`),
  so a device input takes the host input's route and gives its bits. R still lands in host memory (n x n, at most
  TS_MAX_N squared): the factor's NaN refusal reads it, and qr's R_SIGNS signs it on the host, as for a host input.

Identical images only (the device arrays' set; gemm/linalg_dev_arrays.mojo `lin_dev_ready`): a fast image refuses
in words, so it can never read the other mode's pool. The host image refuses every resident operation.
"""
from max.gpu.host import DeviceBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.train_dev_pool import train_dev_view
from x_decomp.cells import F32Ptr
from x_decomp.device import xd_ctx
from x_decomp.resident import pool_buffer
from x_decomp.tsqr_core import TS_MAX_N
from x_decomp.tsqr_device import ts_apply_device, ts_apply_device_owned, ts_dev_owner, ts_dev_owner_new, ts_dev_release
from x_decomp.tsqr_multi_gpu import ts_factor_dispatch, tsqr_plan_current


def _train_ready(who: String) raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("RuntimeError: x_decomp " + who + ": training device arrays are read by the identical image only")


def dev_from_train_run(handle: Int, dst: Int, n: Int) raises -> Int:
    """Training handle `handle`'s first `n` words into pool id `dst`, device to device. Waits. Returns n."""
    _train_ready("from_train")
    if n < 1:
        raise Error("x_decomp: from_train needs n >= 1")
    var ctx = xd_ctx()
    var source = train_dev_view(ctx, handle, n, "the x_decomp copy")
    var target = pool_buffer(dst, n)
    ctx.enqueue_copy(dst_buf=target.create_sub_buffer[DType.float32](0, n), src_buf=source)
    ctx.synchronize()
    _ = source^
    _ = target^
    return n


def dev_to_train_run(src: Int, handle: Int, n: Int) raises -> Int:
    """Pool id `src`'s first `n` words into training handle `handle`, device to device. Waits. Returns n."""
    _train_ready("to_train")
    if n < 1:
        raise Error("x_decomp: to_train needs n >= 1")
    var ctx = xd_ctx()
    var source = pool_buffer(src, n)
    var target = train_dev_view(ctx, handle, n, "the x_decomp result")
    ctx.enqueue_copy(dst_buf=target, src_buf=source.create_sub_buffer[DType.float32](0, n))
    ctx.synchronize()
    _ = source^
    _ = target^
    return n


def dev_tsqr_r_run(ix: Int, pr: F32Ptr, m: Int, d: Int, keep: Bool) raises -> Int:
    """`tsqr_r_run` (x_decomp/api.mojo, nrhs 0) of the pool matrix ix (m x d, left as it is): R (d x d) into host
    memory at pr; with `keep` the factorization's owner handle (XD_DEV_TSQR_Q or XD_TSQR_Q frees it), else d.
    `DevExec.tsqr_factor`'s nrhs == 0 branch with the upload replaced by one device copy: the same words."""
    _train_ready("tsqr_r")
    if d < 1 or d > TS_MAX_N or m < d:
        raise Error("x_decomp: tsqr_r needs 1 <= d, d + nrhs <= " + String(TS_MAX_N) + " and m >= d + nrhs")
    if m * d > 2147483647:
        raise Error("x_decomp: tsqr_r exceeds the Int32 index bound")
    var plan = tsqr_plan_current()
    var ctx = xd_ctx()
    var cells = m * d
    var source = pool_buffer(ix, cells)
    var da = ctx.enqueue_create_buffer[DType.float32](cells)
    ctx.enqueue_copy(dst_buf=da, src_buf=source.create_sub_buffer[DType.float32](0, cells))
    var held = ts_factor_dispatch(plan, ctx, da, m, d, pr, keep)
    _ = da^
    ctx.synchronize()
    _ = source^
    if keep:
        return ts_dev_owner_new(held^, m, d)
    return d


def dev_tsqr_q_run(handle: Int, ic: Int, pc: F32Ptr, iq: Int, m: Int, n: Int, k: Int) raises -> Int:
    """`tsqr_q_run` into pool id iq (m x k): Q C over the factorization `handle` owns, C (n x k) the pool matrix ic,
    or host memory at pc when ic < 0. The owner is freed on every path (k == 0 only frees it). Returns k."""
    var owner = ts_dev_owner(handle)
    try:
        if k == 0:
            return 0
        _train_ready("tsqr_q")
        if m * k > 2147483647:
            raise Error("x_decomp: tsqr_q exceeds the Int32 index bound")
        if owner[].m != m or owner[].n != n:
            raise Error("x_decomp tsqr: no kept factorization of this shape (tsqr_r with keep first)")
        var ctx = xd_ctx()
        var target = pool_buffer(iq, m * k)
        if ic >= 0:
            var c = pool_buffer(ic, n * k)
            var dq = ts_apply_device_owned(ctx, c, owner[].bufs, m, n, k)
            ctx.enqueue_copy(dst_buf=target.create_sub_buffer[DType.float32](0, m * k),
                             src_buf=dq.create_sub_buffer[DType.float32](0, m * k))
            ctx.synchronize()
            _ = dq^
            _ = c^
        else:
            var dq = ts_apply_device(ctx, pc, owner[].bufs, m, n, k)
            ctx.enqueue_copy(dst_buf=target.create_sub_buffer[DType.float32](0, m * k),
                             src_buf=dq.create_sub_buffer[DType.float32](0, m * k))
            ctx.synchronize()
            _ = dq^
        _ = target^
        return k
    finally:
        ts_dev_release(handle)

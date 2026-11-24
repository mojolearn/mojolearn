# SPDX-License-Identifier: Apache-2.0
"""Training device arrays in and out of the x_decomp pool, and the TSQR on pool ids (lane vdf-linalg6, 2026-11-24;
lane vdf-linalg7, 2026-11-24).

A `TrainDeviceArray` (core/train_dev_pool.mojo, the one device-array convention of the training ops) is read by the
decompositions that run on the x_decomp kit (`mojolearn.linalg.eigh`, `qr`, `svd`; `solve`, `lstsq`,
`randomized_svd`, `sparse_encode`; bindings/linalg_ops_kit_run.mojo `LinKit.adopt`, bindings/decomp_kit_run.mojo
`DKit.adopt`) without a host trip:

- XD_DEV_FROM_TRAIN copies a training handle's first n words device to device into an x_decomp pool id, and
  XD_DEV_TO_TRAIN copies a pool id back into a training handle (the PACK of a device result). Both run on the
  x_decomp context and synchronize before they return; the training entries synchronize their own context before
  they return (core/train_dev_pool.mojo), so a handle is complete when either side reads it.
- XD_DEV_FINITE is the admission's finite test of a pool matrix, on the device (one status word comes back): a device
  operand is refused in the words a host operand's admission uses, without its words coming down.
- XD_DEV_TSQR_R / XD_DEV_TSQR_Q are XD_TSQR_R / XD_TSQR_Q (x_decomp/device.mojo `DevExec.tsqr_factor`,
  `tsqr_apply`) with X, [X | B], C and Q on pool ids: X (and B) are read in place where the host route uploads them,
  the factor and the apply are the SAME calls (`ts_pack_device`, `ts_factor_dispatch`, `ts_apply_device*`), so a
  device input takes the host input's route and gives its bits. R goes to host memory as before, or (lane
  vdf-linalg7) into a pool id: the factor's NaN refusal is already a device scan (one status word comes down), so R
  stays on the device for the kit steps that read it next (qr's signs, svd's small SVD).
- XD_DEV_R_SIGNS is `r_signs_run` (x_decomp/api.mojo) on pool ids: the same exact sign moves, a thread per word.
- XD_DEV_LU_GESV is `DevExec.lu_gesv` with A and B read from pool ids and X written into one: the same launches on
  copies of the same words (the factor and the solve write their operands), info (one word) to the host.

Both GPU images (identical and fast) carry these operations: the pool is one per process (core/train_dev_pool.mojo),
so the fast kit reads the arrays the identical training binding made. Lane vdf-linalg6 refused them in the fast image
because the pool name was keyed on the tier. The host image refuses every resident operation.
"""
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer
from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN
from core.admitted_finite_status import enqueue_finite_word
from core.train_dev_pool import train_dev_view
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.device import xd_ctx, LU_SCAL_LEN, TPB, _blocks, _down, _p, launch_lu, launch_lu_solve
from x_decomp.resident import pool_buffer
from x_decomp.tsqr_core import TS_MAX_N
from x_decomp.tsqr_device import (
    ts_apply_device, ts_apply_device_owned, ts_dev_owner, ts_dev_owner_new, ts_dev_release, ts_pack_device,
)
from x_decomp.tsqr_multi_gpu import ts_factor_dispatch, tsqr_plan_current


def dev_from_train_run(handle: Int, dst: Int, n: Int) raises -> Int:
    """Training handle `handle`'s first `n` words into pool id `dst`, device to device. Waits. Returns n."""
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


def dev_finite_run(id: Int, n: Int) raises -> Int:
    """The finite status of pool id `id`'s first `n` words, tested on the device (`enqueue_finite_word`, the test
    XD_DEV_UPLOAD_FINITE runs on its uploaded words): 0 when every word is finite, 1 otherwise. One word comes down."""
    if n < 1:
        return 0
    var buffer = pool_buffer(id, n)
    var dp = F32Ptr(unsafe_from_address=Int(buffer.unsafe_ptr()))
    var ctx = xd_ctx()
    var status = ctx.enqueue_create_buffer[DType.int32](1)
    enqueue_finite_word(ctx, dp, n, status, 0)
    var word = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=word.unsafe_ptr(), src_buf=status)
    ctx.synchronize()
    var flagged = Int(word.unsafe_ptr()[0])
    _ = word^
    _ = status^
    _ = buffer^
    return 1 if flagged != 0 else 0


def dev_tsqr_r_run(ix: Int, pr: F32Ptr, m: Int, d: Int, keep: Bool, ir: Int, ib: Int, nrhs: Int) raises -> Int:
    """`tsqr_r_run` (x_decomp/api.mojo) of the pool matrix ix (m x d, left as it is), with `nrhs` > 0 the right-hand
    sides pool matrix ib (m x nrhs, left as it is) packed beside it: R (n x n, n = d + nrhs) into host memory at pr
    (ir < 0), or into pool id ir (ir >= 0: pr unread, R never leaves the device); with `keep` the factorization's
    owner handle (XD_DEV_TSQR_Q or XD_TSQR_Q frees it), else n. `DevExec.tsqr_factor` with the uploads replaced by
    one device copy (nrhs 0) or by `ts_pack_device` reading the pool buffers in place (nrhs > 0): the same words."""
    var n = d + nrhs
    if d < 1 or nrhs < 0 or n > TS_MAX_N or m < n:
        raise Error("x_decomp: tsqr_r needs 1 <= d, d + nrhs <= " + String(TS_MAX_N) + " and m >= d + nrhs")
    if m * n > 2147483647:
        raise Error("x_decomp: tsqr_r exceeds the Int32 index bound")
    var plan = tsqr_plan_current()
    var ctx = xd_ctx()
    var source = pool_buffer(ix, m * d)
    var da: DeviceBuffer[DType.float32]
    if nrhs == 0:
        da = ctx.enqueue_create_buffer[DType.float32](m * d)
        ctx.enqueue_copy(dst_buf=da, src_buf=source.create_sub_buffer[DType.float32](0, m * d))
    else:
        var tb = pool_buffer(ib, m * nrhs)
        var ta = source.create_sub_buffer[DType.float32](0, m * d)
        var tbv = tb.create_sub_buffer[DType.float32](0, m * nrhs)
        da = ts_pack_device(ctx, ta, tbv, m, d, nrhs)
        ctx.synchronize()
        _ = ta^
        _ = tbv^
        _ = tb^
    # R to the device: the factor keeps its R tiles (held[2], R is its first n * n words) and writes no host R
    # (a null `r`, x_decomp/tsqr_device.mojo `ts_factor_finish`); the copy below takes R before the owner is freed.
    var on_device = ir >= 0
    var held = ts_factor_dispatch(plan, ctx, da, m, n, F32Ptr(unsafe_from_address=Int(0)) if on_device else pr,
                                  keep or on_device)
    _ = da^
    if on_device:
        var target = pool_buffer(ir, n * n)
        ctx.enqueue_copy(dst_buf=target.create_sub_buffer[DType.float32](0, n * n),
                         src_buf=held[2].create_sub_buffer[DType.float32](0, n * n))
        ctx.synchronize()
        _ = target^
    ctx.synchronize()
    _ = source^
    if keep:
        return ts_dev_owner_new(held^, m, n)
    _ = held^
    return n


def dev_tsqr_q_run(handle: Int, ic: Int, pc: F32Ptr, iq: Int, m: Int, n: Int, k: Int) raises -> Int:
    """`tsqr_q_run` into pool id iq (m x k): Q C over the factorization `handle` owns, C (n x k) the pool matrix ic,
    or host memory at pc when ic < 0. The owner is freed on every path (k == 0 only frees it). Returns k."""
    var owner = ts_dev_owner(handle)
    try:
        if k == 0:
            return 0
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


def r_signs_kernel(r: F32Ptr, c: F32Ptr, n: Int32):
    """`r_signs_run` word for word, a thread per word of the n x n R and C, except R's diagonal: the off-diagonal
    words of row j of R negated when R[j, j] < 0 (an exact sign flip; -0.0 and NaN count as non-negative, as the
    host `<`); C the diagonal of the signs, zeros elsewhere. No thread of this launch writes a diagonal word of R,
    so every thread reads its row's original diagonal; `r_diag_kernel` signs the diagonal after."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nn = Int(n) * Int(n)
    if i >= nn:
        return
    var j = i // Int(n)
    var t = i - j * Int(n)
    var neg = r.unsafe_load(j * Int(n) + j) < Float32(0.0)
    if t != j and neg:
        r.unsafe_store(i, -r.unsafe_load(i))
    c.unsafe_store(i, (Float32(-1.0) if neg else Float32(1.0)) if t == j else Float32(0.0))


def r_diag_kernel(r: F32Ptr, n: Int32):
    """The diagonal words of R negated where negative (the second launch of XD_DEV_R_SIGNS: every off-diagonal word
    of a row was signed by `r_signs_kernel` from the original diagonal, so the diagonal changes last)."""
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j >= Int(n):
        return
    var w = r.unsafe_load(j * Int(n) + j)
    if w < Float32(0.0):
        r.unsafe_store(j * Int(n) + j, -w)


def dev_r_signs_run(ir: Int, ic: Int, n: Int) raises -> Int:
    """XD_R_SIGNS on pool ids: R (n x n, pool id ir) signed in place so its diagonal is non-negative, C (n x n, pool
    id ic) the diagonal of the signs. Two launches: the off-diagonal words and C from the original diagonal, then the
    diagonal itself. The same exact sign moves as the host body, so the same words. Returns n."""
    if n < 1 or n > TS_MAX_N:
        raise Error("x_decomp: r_signs needs 1 <= n <= " + String(TS_MAX_N))
    var ctx = xd_ctx()
    var rb = pool_buffer(ir, n * n)
    var cb = pool_buffer(ic, n * n)
    var pr = F32Ptr(unsafe_from_address=Int(rb.unsafe_ptr()))
    var pc = F32Ptr(unsafe_from_address=Int(cb.unsafe_ptr()))
    ctx.enqueue_function[r_signs_kernel](pr, pc, Int32(n), grid_dim=_blocks(n * n), block_dim=TPB)
    ctx.enqueue_function[r_diag_kernel](pr, Int32(n), grid_dim=_blocks(n), block_dim=TPB)
    ctx.synchronize()
    _ = rb^
    _ = cb^
    return n


def dev_lu_gesv_run(ia: Int, ib: Int, ix: Int, pinf: F32Ptr, n: Int, nrhs: Int) raises -> Int:
    """`DevExec.lu_gesv` (x_decomp/device.mojo) on pool ids: A (n x n, pool id ia) and B (n x nrhs, pool id ib) are
    left as they are, X (n x nrhs) is written into pool id ix, info (one word) into host memory at pinf. The same
    launches (`launch_lu`, then `launch_lu_solve` on the same device factor and pivots) on device copies of the same
    words: the host route uploads A and B into fresh buffers, which the factor and the solve then write; here the
    copies are made on the device. `lu_gesv_run`'s exact-integer check first. Returns n."""
    if n >= 1 << 24:
        raise Error("x_decomp: lu_gesv row numbers exceed float32's exact integers")
    if n <= 0 or nrhs <= 0:
        return n
    if n * n > 2147483647 or n * nrhs > 2147483647:
        raise Error("x_decomp: lu_gesv exceeds the Int32 index bound")
    var ctx = xd_ctx()
    var sa = pool_buffer(ia, n * n)
    var sb = pool_buffer(ib, n * nrhs)
    var sx = pool_buffer(ix, n * nrhs)
    var da = ctx.enqueue_create_buffer[DType.float32](n * n)
    ctx.enqueue_copy(dst_buf=da, src_buf=sa.create_sub_buffer[DType.float32](0, n * n))
    var dp = ctx.enqueue_create_buffer[DType.int32](n)
    var di = ctx.enqueue_create_buffer[DType.float32](1)
    var ds = ctx.enqueue_create_buffer[DType.float32](LU_SCAL_LEN)
    var dact = ctx.enqueue_create_buffer[DType.float32](n)
    var pp = I32Ptr(unsafe_from_address=Int(dp.unsafe_ptr()))
    launch_lu(ctx, _p(da), pp, _p(di), _p(ds), _p(dact), n)
    comptime if TARGET_COLUMN == COLUMN_APPLE:
        ctx.synchronize()
    var ub = ctx.enqueue_create_buffer[DType.float32](n * nrhs)
    ctx.enqueue_copy(dst_buf=ub, src_buf=sb.create_sub_buffer[DType.float32](0, n * nrhs))
    var ui = ctx.enqueue_create_buffer[DType.float32](n)
    var px = F32Ptr(unsafe_from_address=Int(sx.unsafe_ptr()))
    launch_lu_solve(ctx, _p(da), pp, _p(ub), _p(ui), px, n, nrhs, 0)
    _down(ctx, di, pinf, 1)
    ctx.synchronize()
    _ = da^
    _ = dp^
    _ = di^
    _ = ds^
    _ = dact^
    _ = ub^
    _ = ui^
    _ = sa^
    _ = sb^
    _ = sx^
    return n

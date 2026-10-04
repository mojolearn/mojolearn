# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device forms of x_decomp/select_ops.mojo (lane cpu2-l8-decomp,
2026-10-04): on resident matrices, enqueued with no sync, the per-element
functions shared with the host entries, so the words are the host's.

  * `dev_reduce_py`: max |x|, max or min of a device matrix into a 1 x 1
    device matrix; slices of SEL_SLICE values per thread, then the slice
    results the same way until one value is left (ceil(log_256(count))
    launches).
  * `dev_order_small_py`: the stable ascending order of a short device
    vector, one thread per element counting the elements before it (an
    exact rank; every rank is distinct) and writing its index there."""
from std.gpu import block_dim, block_idx, thread_idx
from std.python import PythonObject

from x_decomp.cells import F32Ptr
from x_decomp.device import TPB, _blocks, xd_ctx
from x_decomp.resident import _id, _n, _ptr, pool_alloc, pool_free
from x_decomp.select_ops import SEL_LAST, SEL_ORDER_MAX, SEL_SLICE, order_rank, sel_fold
from x_decomp.pca_mle import (
    MLE_P, mle_base_kernel, mle_cand_kernel, mle_ll_kernel, mle_part_kernel, mle_pick_kernel, mle_prep_kernel,
    mle_scratch_words,
)


def sel_slice_kernel(src: F32Ptr, dst: F32Ptr, count: Int32, op: Int32):
    """dst[t] = the op over the slice [t * SEL_SLICE, ...) of the first
    `count` values."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var cnt = Int(count)
    var nb = (cnt + SEL_SLICE - 1) // SEL_SLICE
    if t < nb:
        var q0 = t * SEL_SLICE
        var q1 = q0 + SEL_SLICE
        if q1 > cnt:
            q1 = cnt
        dst.unsafe_store(t, sel_fold(Int(op), src, q0, q1))


def order_small_kernel(src: F32Ptr, dst: F32Ptr, n: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nn = Int(n)
    if i < nn:
        dst.unsafe_store(order_rank(src, nn, i), Float32(i))


def dev_reduce_py(a: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """dst (a device matrix of >= 1 value) = the op (p = [count, op]) over
    the first count values of the device matrix a. A max |x| slice result is
    already >= 0 (or NaN), so the later passes fold it with the same op."""
    var count = _n(p, 0)
    var op = Int(py=p[1])
    if count < 1:
        raise Error("x_decomp: reduce needs at least one value")
    if op < 0 or op > SEL_LAST:
        raise Error("x_decomp: unknown reduce op")
    var cur = _ptr(_id(a), count)
    var pd = _ptr(_id(dst), 1)
    var ctx = xd_ctx()
    var cur_id = -1
    var cnt = count
    while cnt > SEL_SLICE:
        var nb = (cnt + SEL_SLICE - 1) // SEL_SLICE
        var nid = pool_alloc(nb)
        var pn = _ptr(nid, nb)
        ctx.enqueue_function[sel_slice_kernel](cur, pn, Int32(cnt), Int32(op), grid_dim=_blocks(nb), block_dim=TPB)
        if cur_id >= 0:
            pool_free(cur_id)      # the context runs in order: a reuse comes after this read
        cur_id = nid
        cur = pn
        cnt = nb
    ctx.enqueue_function[sel_slice_kernel](cur, pd, Int32(cnt), Int32(op), grid_dim=1, block_dim=TPB)
    if cur_id >= 0:
        pool_free(cur_id)
    return PythonObject(1)


def dev_order_small_py(a: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """dst (n exact floats, device) = the stable ascending order of the
    first n values of the device matrix a (p = [n])."""
    var n = _n(p, 0)
    if n > SEL_ORDER_MAX:
        raise Error("x_decomp: order_small exceeds its bound")
    if n == 0:
        return PythonObject(0)
    var ps = _ptr(_id(a), n)
    var pd = _ptr(_id(dst), n)
    xd_ctx().enqueue_function[order_small_kernel](ps, pd, Int32(n), grid_dim=_blocks(n), block_dim=TPB)
    return PythonObject(n)


def dev_pca_mle_rank_py(sp: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """dst (one exact float, device) = Minka's MLE rank of the spectrum in
    the device matrix sp (d binary64 values, 2 d float32 words), p = [d,
    n_samples]: x_decomp/pca_mle.mojo's six phases, enqueued with no sync
    (lane cpu3-python). The host column is `x_decomp_pca_mle_rank`."""
    var d = _n(p, 0)
    var n = _n(p, 1)
    if d < 2:
        raise Error("x_decomp: pca_mle needs at least two spectrum values")
    var psp = _ptr(_id(sp), 2 * d)
    var pd = _ptr(_id(dst), 1)
    var words = mle_scratch_words(d)
    var sid = pool_alloc(words)
    var ps = _ptr(sid, words)
    var ctx = xd_ctx()
    var d32 = Int32(d)
    var n32 = Int32(n)
    ctx.enqueue_function[mle_prep_kernel](psp, d32, ps, grid_dim=_blocks(d), block_dim=TPB)
    ctx.enqueue_function[mle_base_kernel](psp, d32, n32, ps, grid_dim=_blocks(d - 1), block_dim=TPB)
    ctx.enqueue_function[mle_part_kernel](psp, d32, n32, ps, grid_dim=_blocks((d - 1) * MLE_P), block_dim=TPB)
    ctx.enqueue_function[mle_ll_kernel](psp, d32, n32, ps, grid_dim=_blocks(d), block_dim=TPB)
    ctx.enqueue_function[mle_cand_kernel](d32, ps, grid_dim=_blocks(MLE_P), block_dim=TPB)
    ctx.enqueue_function[mle_pick_kernel](d32, ps, pd, grid_dim=1, block_dim=1)
    pool_free(sid)      # the context runs in order: a reuse comes after these reads
    return PythonObject(1)

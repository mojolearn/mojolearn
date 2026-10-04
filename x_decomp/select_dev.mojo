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
from max.gpu.host import DeviceBuffer, DeviceContext

from x_decomp.cells import F32Ptr
from x_decomp.device import TPB, _blocks, xd_ctx
from x_decomp.resident import _id, _n, _ptr, pool_alloc, pool_free
from x_decomp.select_ops import SEL_LAST, SEL_MIN, SEL_ORDER_MAX, SEL_SLICE, order_rank, sel_fold
from x_decomp.moves import DSUM_CHUNK, U32Ptr, dsum_fold_chunk, dsum_sq_chunk, dsum_st, order_key
from core.stable_radix_sort import (
    key_bits_for, stable_radix_bsum_len, stable_radix_counts_len, stable_radix_sort_pairs_u32,
)
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
    enqueue_sel_reduce(_ptr(_id(a), count), count, op, _ptr(_id(dst), 1))
    return PythonObject(1)


def enqueue_sel_reduce(src: F32Ptr, count: Int, op: Int, pd: F32Ptr) raises:
    """pd[0] = the op over src[0:count] (count >= 1): slices of SEL_SLICE
    values per thread, then the slice results the same way until one value
    is left. Enqueued only."""
    var cur = src
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


# ---- lane cpu3-python: the decomp kit's orders, selections and the
# binary64 sum of squares on the device (the host column: x_decomp/moves.mojo
# order_f / select_smallest / argmin_all / topn_desc / dsum_sq_host, the
# same order and the same words)

#: the key of an excluded position (skipped, or not equal to the minimum):
#: above every `order_key` (+inf's key is 0xFF800000; NaN is refused first)
comptime ORD_SENT = UInt32(0xFFFFFFFF)


def ord_key_kernel(
    src: F32Ptr, skip: F32Ptr, has_skip: Int32, neg: Int32, keys: U32Ptr, vals: U32Ptr, n: Int32
):
    """keys[t] = order_key(src[t]) (of -src[t] when neg), ORD_SENT where the
    skip row is nonzero (Python's `!= 0`: NaN skips); vals[t] = t."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(n):
        var k = order_key(src.unsafe_load(t), neg != 0)
        if has_skip != 0 and skip.unsafe_load(t) != Float32(0):
            k = ORD_SENT
        keys.unsafe_store(t, k)
        vals.unsafe_store(t, UInt32(t))


def eq_key_kernel(src: F32Ptr, mn: F32Ptr, keys: U32Ptr, vals: U32Ptr, n: Int32):
    """keys[t] = 0 where src[t] == mn[0] (Python's `==`: -0.0 is +0.0, NaN
    equals nothing), else 1; vals[t] = t."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(n):
        keys.unsafe_store(t, UInt32(0) if src.unsafe_load(t) == mn.unsafe_load(0) else UInt32(1))
        vals.unsafe_store(t, UInt32(t))


def ord_out_kernel(keys: U32Ptr, vals: U32Ptr, dst: F32Ptr, cnt: F32Ptr, n: Int32, sent: UInt32):
    """dst[t] = vals[t] as an exact float; cnt[0] = the number of keys
    below `sent` (the sorted keys' boundary: one thread writes it)."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nn = Int(n)
    if t < nn:
        dst.unsafe_store(t, Float32(Int(vals.unsafe_load(t))))
        var kt = keys.unsafe_load(t)
        if kt != sent and (t == nn - 1 or keys.unsafe_load(t + 1) == sent):
            cnt.unsafe_store(0, Float32(t + 1))
        if t == 0 and kt == sent:
            cnt.unsafe_store(0, Float32(0))


def mask_kernel(vals: U32Ptr, mask: F32Ptr, k2: U32Ptr, v2: U32Ptr, n: Int32, h: Int32):
    """mask[vals[t]] = 1 for the first h sorted positions, else 0 (vals is a
    permutation: one writer per slot); k2[t] = vals[t] for t < h."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(n):
        var v = vals.unsafe_load(t)
        mask.unsafe_store(Int(v), Float32(1) if t < Int(h) else Float32(0))
        if t < Int(h):
            k2.unsafe_store(t, v)
            v2.unsafe_store(t, UInt32(t))


def keys_out_kernel(k2: U32Ptr, dst: F32Ptr, h: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(h):
        dst.unsafe_store(t, Float32(Int(k2.unsafe_load(t))))


struct _SortBufs(Movable):
    """One call's radix-sort storage: keys, values, their temporaries and a
    second (keys, values) pair, plus the digit counts and scan sums."""
    var pairs: DeviceBuffer[DType.uint32]
    var ints: DeviceBuffer[DType.int32]
    var keys: DeviceBuffer[DType.uint32]
    var vals: DeviceBuffer[DType.uint32]
    var tkeys: DeviceBuffer[DType.uint32]
    var tvals: DeviceBuffer[DType.uint32]
    var k2: DeviceBuffer[DType.uint32]
    var v2: DeviceBuffer[DType.uint32]
    var cnt: DeviceBuffer[DType.int32]
    var bsum: DeviceBuffer[DType.int32]

    def __init__(out self, ctx: DeviceContext, n: Int) raises:
        var cl = stable_radix_counts_len(n)
        var bl = stable_radix_bsum_len(n)
        var pairs = ctx.enqueue_create_buffer[DType.uint32](6 * n)
        var ints = ctx.enqueue_create_buffer[DType.int32](cl + bl)
        self.keys = pairs.create_sub_buffer[DType.uint32](0, n)
        self.vals = pairs.create_sub_buffer[DType.uint32](n, n)
        self.tkeys = pairs.create_sub_buffer[DType.uint32](2 * n, n)
        self.tvals = pairs.create_sub_buffer[DType.uint32](3 * n, n)
        self.k2 = pairs.create_sub_buffer[DType.uint32](4 * n, n)
        self.v2 = pairs.create_sub_buffer[DType.uint32](5 * n, n)
        self.cnt = ints.create_sub_buffer[DType.int32](0, cl)
        self.bsum = ints.create_sub_buffer[DType.int32](cl, bl)
        self.pairs = pairs^
        self.ints = ints^


def _sort_out(ctx: DeviceContext, mut sb: _SortBufs, n: Int, key_bits: Int, dst: F32Ptr, pc: F32Ptr, sent: UInt32) raises:
    stable_radix_sort_pairs_u32(ctx, n, key_bits, sb.keys, sb.vals, sb.tkeys, sb.tvals, sb.cnt, sb.bsum)
    ctx.enqueue_function[ord_out_kernel](
        sb.keys.unsafe_ptr(), sb.vals.unsafe_ptr(), dst, pc, Int32(n), sent, grid_dim=_blocks(n), block_dim=TPB
    )


def dev_order_f_py(a: PythonObject, skip: PythonObject, dst: PythonObject, cnt: PythonObject, p: PythonObject) raises -> PythonObject:
    """dst (n exact floats, device) = the stable ascending order of the first
    n values of the device matrix a by (value, index) (p = [n, neg,
    has_skip]: neg orders -a, Python's `key=lambda i: (-x[i], i)`), the
    positions where the device row `skip` is nonzero last; cnt (one word,
    device) = how many are not skipped. The caller refuses NaN (one word,
    the max) first: `order_f` / `topn_desc`'s order, the same positions.
    Waits once before its sort storage is released."""
    var n = _n(p, 0)
    var neg = Int(py=p[1])
    var has = Int(py=p[2])
    if n < 1 or n >= 16777216:
        raise Error("x_decomp: order_f needs 1 <= n < 2^24")
    var ps = _ptr(_id(a), n)
    var pk = _ptr(_id(skip), n) if has != 0 else ps
    var pd = _ptr(_id(dst), n)
    var pc = _ptr(_id(cnt), 1)
    var ctx = xd_ctx()
    var sb = _SortBufs(ctx, n)
    ctx.enqueue_function[ord_key_kernel](
        ps, pk, Int32(has), Int32(neg), sb.keys.unsafe_ptr(), sb.vals.unsafe_ptr(), Int32(n),
        grid_dim=_blocks(n), block_dim=TPB,
    )
    _sort_out(ctx, sb, n, 32, pd, pc, ORD_SENT)
    ctx.synchronize()
    _ = sb^
    return PythonObject(n)


def dev_argmin_all_py(a: PythonObject, dst: PythonObject, cnt: PythonObject, p: PythonObject) raises -> PythonObject:
    """dst = every position (exact floats, ascending) of the device matrix
    a's first n values equal to their minimum, then the rest; cnt (one
    word, device) = how many are equal (0 when a NaN is present, as the
    host column). Waits once."""
    var n = _n(p, 0)
    if n < 1 or n >= 16777216:
        raise Error("x_decomp: argmin_all needs 1 <= n < 2^24")
    var ps = _ptr(_id(a), n)
    var pd = _ptr(_id(dst), n)
    var pc = _ptr(_id(cnt), 1)
    var ctx = xd_ctx()
    var mid = pool_alloc(1)
    var pm = _ptr(mid, 1)
    enqueue_sel_reduce(ps, n, SEL_MIN, pm)
    var sb = _SortBufs(ctx, n)
    ctx.enqueue_function[eq_key_kernel](
        ps, pm, sb.keys.unsafe_ptr(), sb.vals.unsafe_ptr(), Int32(n), grid_dim=_blocks(n), block_dim=TPB
    )
    _sort_out(ctx, sb, n, 1, pd, pc, UInt32(1))
    ctx.synchronize()
    pool_free(mid)
    _ = sb^
    return PythonObject(n)


def dev_select_smallest_py(a: PythonObject, sel: PythonObject, mask: PythonObject, p: PythonObject) raises -> PythonObject:
    """p = [n, h]: sel (h exact floats, device) = the positions of the h
    smallest of a's first n values by (value, index), ascending by
    position; mask (n floats, device) = 1 at those positions, else 0. NaN
    refused by the caller first. Waits once."""
    var n = _n(p, 0)
    var h = _n(p, 1)
    if n < 1 or n >= 16777216 or h > n:
        raise Error("x_decomp: select_smallest out of range")
    var ps = _ptr(_id(a), n)
    var pm = _ptr(_id(mask), n)
    var ctx = xd_ctx()
    var sb = _SortBufs(ctx, n)
    ctx.enqueue_function[ord_key_kernel](
        ps, ps, Int32(0), Int32(0), sb.keys.unsafe_ptr(), sb.vals.unsafe_ptr(), Int32(n),
        grid_dim=_blocks(n), block_dim=TPB,
    )
    stable_radix_sort_pairs_u32(ctx, n, 32, sb.keys, sb.vals, sb.tkeys, sb.tvals, sb.cnt, sb.bsum)
    ctx.enqueue_function[mask_kernel](
        sb.vals.unsafe_ptr(), pm, sb.k2.unsafe_ptr(), sb.v2.unsafe_ptr(), Int32(n), Int32(h),
        grid_dim=_blocks(n), block_dim=TPB,
    )
    if h > 0:
        var pd = _ptr(_id(sel), h)
        # the h positions ascending: a second stable sort keyed by position
        stable_radix_sort_pairs_u32(ctx, h, key_bits_for(n - 1), sb.k2, sb.v2, sb.tkeys, sb.tvals, sb.cnt, sb.bsum)
        ctx.enqueue_function[keys_out_kernel](sb.k2.unsafe_ptr(), pd, Int32(h), grid_dim=_blocks(h), block_dim=TPB)
    ctx.synchronize()
    _ = sb^
    return PythonObject(h)


def dsum_part_kernel(src: F32Ptr, dst: F32Ptr, n: Int32):
    """dst[b] (binary64, two words) = dsum_sq_chunk over chunk b of src."""
    var b = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nn = Int(n)
    var nb = (nn + DSUM_CHUNK - 1) // DSUM_CHUNK
    if b < nb:
        dsum_st(dst, 2 * b, dsum_sq_chunk(src, b * DSUM_CHUNK, min((b + 1) * DSUM_CHUNK, nn)))


def dsum_fold_kernel(src: F32Ptr, dst: F32Ptr, n: Int32):
    """dst[b] = dsum_fold_chunk over chunk b of the n binary64 words of src."""
    var b = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nn = Int(n)
    var nb = (nn + DSUM_CHUNK - 1) // DSUM_CHUNK
    if b < nb:
        dsum_st(dst, 2 * b, dsum_fold_chunk(src, b * DSUM_CHUNK, min((b + 1) * DSUM_CHUNK, nn)))


def dev_dsum_sq_py(a: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """dst (two float32 words, device: one binary64, low first) = the sum of
    squares of a's first n values (p = [n >= 1]) in `dsum_sq_host`'s order:
    chunks of DSUM_CHUNK, then the partials DSUM_CHUNK at a time. Enqueued
    only (the caller's download waits)."""
    var n = _n(p, 0)
    if n < 1:
        raise Error("x_decomp: dsum_sq needs at least one value")
    var ps = _ptr(_id(a), n)
    var pd = _ptr(_id(dst), 2)
    var ctx = xd_ctx()
    var nb = (n + DSUM_CHUNK - 1) // DSUM_CHUNK
    var cur_id = pool_alloc(2 * nb)
    var cur = _ptr(cur_id, 2 * nb)
    ctx.enqueue_function[dsum_part_kernel](ps, cur, Int32(n), grid_dim=_blocks(nb), block_dim=TPB)
    while nb > 1:
        var nn = (nb + DSUM_CHUNK - 1) // DSUM_CHUNK
        var nid = pool_alloc(2 * nn)
        var pn = _ptr(nid, 2 * nn)
        ctx.enqueue_function[dsum_fold_kernel](cur, pn, Int32(nb), grid_dim=_blocks(nn), block_dim=TPB)
        pool_free(cur_id)      # the context runs in order: a reuse comes after this read
        cur_id = nid
        cur = pn
        nb = nn
    # the one partial left is the sum: a fold of one word copies it
    ctx.enqueue_function[dsum_fold_kernel](cur, pd, Int32(1), grid_dim=1, block_dim=TPB)
    pool_free(cur_id)
    return PythonObject(1)

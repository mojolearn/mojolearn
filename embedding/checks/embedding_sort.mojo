# SPDX-License-Identifier: Apache-2.0
"""PLAN_SORT: device total-key sort; no floating arithmetic or atomics.

Each compare/exchange pass has disjoint pairs. Stream-ordered launches are
its global barriers. Padding and power-of-two slack sort to the sentinel tail.
"""
from std.sys.compile import is_defined
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.stable_radix_sort import (
    key_bits_for,
    stable_radix_bsum_len,
    stable_radix_counts_len,
    stable_radix_sort_pairs_u32,
)

comptime _REVERSE_TIES = is_defined["MOJOLEARN_EMB_SORT_NEGATIVE_CONTROL"]()
# Contract 11.1's EMB_SORT_KEY_ID_ONLY_UNSTABLE (built 2026-09-15): the
# compare/exchange network compares the id half of the key ONLY. The key still
# carries the position in its low half, so `_perm` decodes a real position,
# but ties inside a run are left wherever the bitonic network moves them, and
# a bitonic network is not stable. `_runs` stays correct: `key < v << 32` is
# still `id < v`, so the run boundaries do not move and only `emb.perm` (and
# through it `emb.dw`) can.
comptime _ID_ONLY_UNSTABLE = is_defined["MOJOLEARN_EMB_SABOTAGE_SORT_KEY_ID_ONLY_UNSTABLE"]()

# lane nr-small D5 (2026-10-04): PLAN_SORT's bitonic network (log2(S) *
# (log2(S) + 1) / 2 exchange launches over the power-of-two padded size S,
# about 120 at T = 32,768) becomes a STABLE LSD radix sort of (id, t) pairs
# over the T positions (`core/stable_radix_sort`: two 8-bit passes of five
# launches while vocab < 65,536, four above). Keys are the ids (padding and
# nothing else mapped to `vocab`, above every real id), values the
# positions, which enter ascending, so the stable order by id IS the total
# (id, t) order the bitonic network produced: `counts`, `run_begin` and
# `perm` are the same words, and dW does not move. Integer work only, so
# every vendor and the host column agree. The two tie controls above need
# the bitonic network (an unstable compare, a reversed position code), so
# either one keeps it. IDENTICAL only; -D MOJOLEARN_IDN_EMB_RADIX_SORT_OFF
# (or MOJOLEARN_IDN_ALL_OFF) restores the bitonic network.
comptime EMB_RADIX_SORT = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    # I11 current experiment: NEVER RUN — PENDING MEASUREMENT; existing defaults preserved.
    and not (is_defined["MOJOLEARN_IDN_EMB_RADIX_SORT_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
    and not _REVERSE_TIES
    and not _ID_ONLY_UNSTABLE
)

comptime PLAN_SCAN = 0
comptime PLAN_SORT = 1
comptime _SENTINEL = UInt64(0xFFFFFFFFFFFFFFFF)


def _pack(keys: MutPointer[UInt64, MutAnyOrigin], ids: MutPointer[Int32, MutAnyOrigin], n: Int32, size: Int32, padding: Int32):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= Int(size):
        return
    var key = _SENTINEL
    if i < Int(n):
        var v = Int(ids.unsafe_load(i))
        if v != Int(padding):
            var position = i
            comptime if _REVERSE_TIES:
                position = Int(n) - 1 - i
            key = ((UInt64(v) & UInt64(0xFFFFFFFF)) << 32) | (UInt64(position) & UInt64(0xFFFFFFFF))
    keys.unsafe_store(i, key)


def _exchange(keys: MutPointer[UInt64, MutAnyOrigin], size: Int32, span: Int32, stride: Int32):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    var j = i ^ Int(stride)
    if i >= Int(size) or j <= i:
        return
    var a = keys.unsafe_load(i)
    var b = keys.unsafe_load(j)
    var a_cmp = a
    var b_cmp = b
    comptime if _ID_ONLY_UNSTABLE:
        a_cmp = a >> 32
        b_cmp = b >> 32
    if (a_cmp > b_cmp) == ((i & Int(span)) == 0):
        keys.unsafe_store(i, b)
        keys.unsafe_store(j, a)


def _lower(keys: MutPointer[UInt64, MutAnyOrigin], size: Int, needle: UInt64) -> Int:
    var lo = 0
    var hi = size
    while lo < hi:
        var mid = (lo + hi) // 2
        if keys.unsafe_load(mid) < needle:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _runs(keys: MutPointer[UInt64, MutAnyOrigin], counts: MutPointer[Int32, MutAnyOrigin], begin: MutPointer[Int32, MutAnyOrigin], size: Int32, vocab: Int32):
    var v = Int(block_idx.x * block_dim.x + thread_idx.x)
    if v > Int(vocab):
        return
    var lo = _lower(keys, Int(size), UInt64(v) << 32)
    begin.unsafe_store(v, Int32(lo))
    if v < Int(vocab):
        var hi = _lower(keys, Int(size), UInt64(v + 1) << 32)
        counts.unsafe_store(v, Int32(hi - lo))


def _perm(keys: MutPointer[UInt64, MutAnyOrigin], perm: MutPointer[Int32, MutAnyOrigin], size: Int32, n: Int32):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i < Int(size):
        var key = keys.unsafe_load(i)
        if key != _SENTINEL:
            var position = Int(key & UInt64(0xFFFFFFFF))
            comptime if _REVERSE_TIES:
                position = Int(n) - 1 - position
            perm.unsafe_store(i, Int32(position))


def _pack_radix(keys: MutPointer[UInt32, MutAnyOrigin], vals: MutPointer[UInt32, MutAnyOrigin], ids: MutPointer[Int32, MutAnyOrigin], n: Int32, vocab: Int32, padding: Int32):
    """key = the id (`vocab` for a padding position), value = the position."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= Int(n):
        return
    var v = Int(ids.unsafe_load(i))
    var key = UInt32(vocab)
    if v != Int(padding):
        key = UInt32(v)
    keys.unsafe_store(i, key)
    vals.unsafe_store(i, UInt32(i))


@always_inline
def _lower_radix(keys: MutPointer[UInt32, MutAnyOrigin], size: Int, needle: UInt32) -> Int:
    var lo = 0
    var hi = size
    while lo < hi:
        var mid = (lo + hi) // 2
        if keys.unsafe_load(mid) < needle:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _runs_radix(keys: MutPointer[UInt32, MutAnyOrigin], counts: MutPointer[Int32, MutAnyOrigin], begin: MutPointer[Int32, MutAnyOrigin], n: Int32, vocab: Int32, padding: Int32):
    """`_runs` over the sorted ids. No real id equals `padding` (mapped to
    `vocab`), so its count is 0 and its begin is the same lower bound;
    `begin[vocab]` is the non-padding count, as before."""
    var v = Int(block_idx.x * block_dim.x + thread_idx.x)
    if v > Int(vocab):
        return
    var lo = _lower_radix(keys, Int(n), UInt32(v))
    begin.unsafe_store(v, Int32(lo))
    if v < Int(vocab):
        var c = 0
        if v != Int(padding):
            c = _lower_radix(keys, Int(n), UInt32(v + 1)) - lo
        counts.unsafe_store(v, Int32(c))


def _perm_radix(keys: MutPointer[UInt32, MutAnyOrigin], vals: MutPointer[UInt32, MutAnyOrigin], perm: MutPointer[Int32, MutAnyOrigin], n: Int32, vocab: Int32):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i < Int(n):
        if keys.unsafe_load(i) != UInt32(vocab):
            perm.unsafe_store(i, Int32(vals.unsafe_load(i)))


def _embedding_sort_runs_radix(ctx: DeviceContext, mut ids: DeviceBuffer[DType.int32], mut counts: DeviceBuffer[DType.int32], mut begin: DeviceBuffer[DType.int32], mut perm: DeviceBuffer[DType.int32], n: Int, vocab: Int, padding: Int, threads: Int) raises:
    """`embedding_sort_runs` by the stable radix sort (EMB_RADIX_SORT).
    Scratch: one u32 buffer (keys, values and their ping-pong halves, 4n)
    and one int32 buffer (digit counts and scan block sums). Waits once
    before releasing them, as the bitonic form did."""
    var cl = stable_radix_counts_len(n)
    var bl = stable_radix_bsum_len(n)
    var pairs = ctx.enqueue_create_buffer[DType.uint32](4 * n)
    var ints = ctx.enqueue_create_buffer[DType.int32](cl + bl)
    var keys = pairs.create_sub_buffer[DType.uint32](0, n)
    var vals = pairs.create_sub_buffer[DType.uint32](n, n)
    var tkeys = pairs.create_sub_buffer[DType.uint32](2 * n, n)
    var tvals = pairs.create_sub_buffer[DType.uint32](3 * n, n)
    var cnt = ints.create_sub_buffer[DType.int32](0, cl)
    var bsum = ints.create_sub_buffer[DType.int32](cl, bl)
    var grid = (n + threads - 1) // threads
    ctx.enqueue_function[_pack_radix](keys.unsafe_ptr(), vals.unsafe_ptr(), ids.unsafe_ptr(), Int32(n), Int32(vocab), Int32(padding), grid_dim=(grid, 1, 1), block_dim=(threads, 1, 1))
    stable_radix_sort_pairs_u32(ctx, n, key_bits_for(vocab), keys, vals, tkeys, tvals, cnt, bsum)
    ctx.enqueue_function[_runs_radix](keys.unsafe_ptr(), counts.unsafe_ptr(), begin.unsafe_ptr(), Int32(n), Int32(vocab), Int32(padding), grid_dim=((vocab + 1 + threads - 1) // threads, 1, 1), block_dim=(threads, 1, 1))
    ctx.enqueue_function[_perm_radix](keys.unsafe_ptr(), vals.unsafe_ptr(), perm.unsafe_ptr(), Int32(n), Int32(vocab), grid_dim=(grid, 1, 1), block_dim=(threads, 1, 1))
    ctx.synchronize()
    _ = keys^
    _ = vals^
    _ = tkeys^
    _ = tvals^
    _ = cnt^
    _ = bsum^
    _ = pairs^
    _ = ints^


def embedding_sort_runs(ctx: DeviceContext, mut ids: DeviceBuffer[DType.int32], mut counts: DeviceBuffer[DType.int32], mut begin: DeviceBuffer[DType.int32], mut perm: DeviceBuffer[DType.int32], n: Int, vocab: Int, padding: Int, threads: Int) raises:
    """Build identical run arrays. Synchronizes before releasing local key scratch.

    IDs must already have passed the production refusal check. The caller owns
    output buffers, including untouched perm entries after the nonpadding count.
    """
    comptime if EMB_RADIX_SORT:
        if n > 0 and vocab > 0:
            _embedding_sort_runs_radix(ctx, ids, counts, begin, perm, n, vocab, padding, threads)
            return
    var size = 1
    while size < n:
        size *= 2
    var keys = ctx.enqueue_create_buffer[DType.uint64](size)
    var grid = (size + threads - 1) // threads
    ctx.enqueue_function[_pack](keys.unsafe_ptr(), ids.unsafe_ptr(), Int32(n), Int32(size), Int32(padding), grid_dim=(grid, 1, 1), block_dim=(threads, 1, 1))
    var span = 2
    while span <= size:
        var stride = span // 2
        while stride > 0:
            ctx.enqueue_function[_exchange](keys.unsafe_ptr(), Int32(size), Int32(span), Int32(stride), grid_dim=(grid, 1, 1), block_dim=(threads, 1, 1))
            stride //= 2
        span *= 2
    ctx.enqueue_function[_runs](keys.unsafe_ptr(), counts.unsafe_ptr(), begin.unsafe_ptr(), Int32(size), Int32(vocab), grid_dim=((vocab + 1 + threads - 1) // threads, 1, 1), block_dim=(threads, 1, 1))
    ctx.enqueue_function[_perm](keys.unsafe_ptr(), perm.unsafe_ptr(), Int32(size), Int32(n), grid_dim=(grid, 1, 1), block_dim=(threads, 1, 1))
    ctx.synchronize()
    _ = keys^

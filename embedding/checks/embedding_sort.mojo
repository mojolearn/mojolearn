# SPDX-License-Identifier: Apache-2.0
"""PLAN_SORT: device total-key sort; no floating arithmetic or atomics.

Each compare/exchange pass has disjoint pairs. Stream-ordered launches are
its global barriers. Padding and power-of-two slack sort to the sentinel tail.
"""
from std.sys.compile import is_defined
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

comptime _REVERSE_TIES = is_defined["MOJOLEARN_EMB_SORT_NEGATIVE_CONTROL"]()

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
    if (a > b) == ((i & Int(span)) == 0):
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


def embedding_sort_runs(ctx: DeviceContext, mut ids: DeviceBuffer[DType.int32], mut counts: DeviceBuffer[DType.int32], mut begin: DeviceBuffer[DType.int32], mut perm: DeviceBuffer[DType.int32], n: Int, vocab: Int, padding: Int, threads: Int) raises:
    """Build identical run arrays. Synchronizes before releasing local key scratch.

    IDs must already have passed the production refusal check. The caller owns
    output buffers, including untouched perm entries after the nonpadding count.
    """
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

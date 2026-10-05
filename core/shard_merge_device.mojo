# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Multi-GPU neighbor shard merges and the IVF shard plan on the device
(lane cpu4-python).

The GPU bindings merged every device's top-k candidates on the host
(`ivf_merge_shards`, `shard_topk_merge_f32`), rooted the merged IVF
distances on the host (`ivf_finalize_distances`) and planned the IVF shard
split on the host (`ivf_shard_plan`). Each is a kernel here; the host loops
stay the host column (the `_host` bindings).

- The merges: one thread per query row walks the shards' candidates in the
  host loop's order and keeps the k best in its output row by the host's
  order ((float32 distance, original id) for IVF; the composite key
  (twiddled distance bits, global index) then the float insertion sort for
  the reference shards). The k best of a total order are the k first of the
  host's full sort: the same words. A refusal records its row's status and
  the lowest refusing row wins (`Atomic.min`), the row the host stopped at.
- The root: `ftz(identical_sqrt(d))` per word, the host's statement.
- The plan: a permutation check by atomic counts, each shard's ascending ids
  by `device_compact_equal_i32` over the owner array (an exclusive scan: the
  host's ascending walk), ranks by a scatter, and the clipped offsets.
"""
from std.atomic import Atomic
from std.memory import bitcast
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import ftz, identical_sqrt
from core.device_fold import device_compact_equal_i32

comptime _TPB = 256
comptime _F32 = MutPointer[Float32, MutAnyOrigin]
comptime _I32 = MutPointer[Int32, MutAnyOrigin]
comptime _U32 = MutPointer[UInt32, MutAnyOrigin]
comptime _I64 = MutPointer[Int64, MutAnyOrigin]
comptime _U64 = MutPointer[UInt64, MutAnyOrigin]
comptime _HF32 = MutPointer[Float32, MutUntrackedOrigin]
comptime _HI32 = MutPointer[Int32, MutUntrackedOrigin]
comptime _HU32 = MutPointer[UInt32, MutUntrackedOrigin]
comptime _HI64 = MutPointer[Int64, MutUntrackedOrigin]
#: "no refusing row" for the Atomic.min of the refusing rows
comptime _NO_ROW = Int32(2147483647)


@always_inline
def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _blocks(count: Int) -> Int:
    return (count + _TPB - 1) // _TPB if count > 0 else 1


def _read2(ctx: DeviceContext, mut buf: DeviceBuffer[DType.int32]) raises -> SIMD[DType.int32, 2]:
    var h = ctx.enqueue_create_host_buffer[DType.int32](2)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var out = SIMD[DType.int32, 2](h.unsafe_ptr().unsafe_load(0), h.unsafe_ptr().unsafe_load(1))
    _ = h^
    return out


# ---- the IVF merge -----------------------------------------------------------


def _ivf_merge_kernel(
    dist: _F32, ids: _I32, cnt: _I32, maps: _I32, meta: _I32,
    P_: Int32, m_: Int32, k_: Int32,
    od: _F32, oi: _I32, oc: _I32, st: _I32, bad: _I32,
):
    """Row `row`: shard s's first min(k, count) candidates (dist/ids at
    ((s m + row) k + j), count at s m + row), each local id mapped through
    shard s's id map (maps[meta[P + s] + local], meta[s] its length), kept
    k best by (distance, id) in od/oi; oc[row] the candidate total. Status
    1 a count outside [0, len], 2 a local id outside the map, 4 a NaN
    distance, 3 fewer than k candidates."""
    var row = _tid()
    var m = Int(m_)
    if row >= m:
        return
    var P = Int(P_)
    var k = Int(k_)
    var base = row * k
    var filled = 0
    var total = 0
    var status = 0
    for s in range(P):
        var n = Int(cnt.unsafe_load(s * m + row))
        var sn = Int(meta.unsafe_load(s))
        var moff = Int(meta.unsafe_load(P + s))
        if n < 0 or n > sn:
            status = 1
            break
        total += n
        var take = n if n < k else k
        for j in range(take):
            var at_src = (s * m + row) * k + j
            var local = Int(ids.unsafe_load(at_src))
            if local < 0 or local >= sn:
                status = 2
                break
            var d = dist.unsafe_load(at_src)
            if d != d:
                status = 4
                break
            var id = maps.unsafe_load(moff + local)
            var at: Int
            if filled < k:
                at = filled
                filled += 1
            else:
                var ld = od.unsafe_load(base + k - 1)
                var li = oi.unsafe_load(base + k - 1)
                if not (ld > d or (ld == d and li > id)):
                    continue
                at = k - 1
            while at > 0:
                var pd = od.unsafe_load(base + at - 1)
                var pi = oi.unsafe_load(base + at - 1)
                if pd > d or (pd == d and pi > id):
                    od.unsafe_store(base + at, pd)
                    oi.unsafe_store(base + at, pi)
                    at -= 1
                else:
                    break
            od.unsafe_store(base + at, d)
            oi.unsafe_store(base + at, id)
        if status != 0:
            break
    if status == 0 and total < k:
        status = 3
    if status != 0:
        st.unsafe_store(row, Int32(status))
        _ = Atomic[DType.int32].min(bad, Int32(row))
        return
    oc.unsafe_store(row, Int32(total))


def device_ivf_merge_shards(
    ctx: DeviceContext, m: Int, k: Int,
    d_addrs: List[Int], i_addrs: List[Int], c_addrs: List[Int], map_addrs: List[Int], sizes: List[Int],
    od_addr: Int, oi_addr: Int, oc_addr: Int,
) raises -> SIMD[DType.int32, 2]:
    """`ivf_merge_shards` over P = len(sizes) shards on the device:
    (status, refusing row). On status 0 the outputs are written."""
    var P = len(sizes)
    var total_map = 0
    for s in range(P):  # small-loop(P: one shard per device): map offsets, no row data
        total_map += sizes[s]
    var dist = ctx.enqueue_create_buffer[DType.float32](max(P * m * k, 1))
    var ids = ctx.enqueue_create_buffer[DType.int32](max(P * m * k, 1))
    var cnt = ctx.enqueue_create_buffer[DType.int32](max(P * m, 1))
    var maps = ctx.enqueue_create_buffer[DType.int32](max(total_map, 1))
    var hmeta = ctx.enqueue_create_host_buffer[DType.int32](2 * P)
    ctx.synchronize()
    var moff = 0
    for s in range(P):  # small-loop(P: one shard per device): one copy per shard array, launch arguments only
        hmeta.unsafe_ptr().unsafe_store(s, Int32(sizes[s]))
        hmeta.unsafe_ptr().unsafe_store(P + s, Int32(moff))
        if m > 0:
            var dsub = dist.create_sub_buffer[DType.float32](s * m * k, m * k)
            ctx.enqueue_copy(dst_buf=dsub, src_ptr=_HF32(unsafe_from_address=d_addrs[s]))
            var isub = ids.create_sub_buffer[DType.int32](s * m * k, m * k)
            ctx.enqueue_copy(dst_buf=isub, src_ptr=_HI32(unsafe_from_address=i_addrs[s]))
            var csub = cnt.create_sub_buffer[DType.int32](s * m, m)
            ctx.enqueue_copy(dst_buf=csub, src_ptr=_HI32(unsafe_from_address=c_addrs[s]))
            _ = dsub^
            _ = isub^
            _ = csub^
        if sizes[s] > 0:
            var msub = maps.create_sub_buffer[DType.int32](moff, sizes[s])
            ctx.enqueue_copy(dst_buf=msub, src_ptr=_HI32(unsafe_from_address=map_addrs[s]))
            _ = msub^
        moff += sizes[s]
    var meta = ctx.enqueue_create_buffer[DType.int32](2 * P)
    ctx.enqueue_copy(dst_buf=meta, src_ptr=hmeta.unsafe_ptr())
    var od = ctx.enqueue_create_buffer[DType.float32](max(m * k, 1))
    var oi = ctx.enqueue_create_buffer[DType.int32](max(m * k, 1))
    var oc = ctx.enqueue_create_buffer[DType.int32](max(m, 1))
    var st = ctx.enqueue_create_buffer[DType.int32](max(m, 1))
    var bad = ctx.enqueue_create_buffer[DType.int32](2)
    ctx.enqueue_memset(st, Int32(0))
    ctx.enqueue_memset(bad, _NO_ROW)
    if m > 0:
        ctx.enqueue_function[_ivf_merge_kernel](
            dist.unsafe_ptr(), ids.unsafe_ptr(), cnt.unsafe_ptr(), maps.unsafe_ptr(), meta.unsafe_ptr(),
            Int32(P), Int32(m), Int32(k),
            od.unsafe_ptr(), oi.unsafe_ptr(), oc.unsafe_ptr(), st.unsafe_ptr(), bad.unsafe_ptr(),
            grid_dim=_blocks(m), block_dim=_TPB,
        )
    var b = _read2(ctx, bad)
    var out = SIMD[DType.int32, 2](0, 0)
    if b[0] != _NO_ROW:
        var row = Int(b[0])
        var one = st.create_sub_buffer[DType.int32](row, 1)
        var h = ctx.enqueue_create_host_buffer[DType.int32](1)
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=one)
        ctx.synchronize()
        out = SIMD[DType.int32, 2](h.unsafe_ptr().unsafe_load(0), Int32(row))
        _ = h^
        _ = one^
    elif m > 0:
        ctx.enqueue_copy(dst_ptr=_HF32(unsafe_from_address=od_addr), src_buf=od)
        ctx.enqueue_copy(dst_ptr=_HI32(unsafe_from_address=oi_addr), src_buf=oi)
        ctx.enqueue_copy(dst_ptr=_HI32(unsafe_from_address=oc_addr), src_buf=oc)
        ctx.synchronize()
    _ = hmeta^
    _ = dist^
    _ = ids^
    _ = cnt^
    _ = maps^
    _ = meta^
    _ = od^
    _ = oi^
    _ = oc^
    _ = st^
    _ = bad^
    return out


# ---- the IVF root --------------------------------------------------------------


def _root_kernel(d: _F32, n_: Int32):
    var i = _tid()
    if i < Int(n_):
        d.unsafe_store(i, ftz(identical_sqrt(d.unsafe_load(i))))


def device_root_f32(ctx: DeviceContext, addr: Int, n: Int) raises:
    """`postprocess_distances`' L2SqrtExpanded arm over n host words in place."""
    if n <= 0:
        return
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=buf, src_ptr=_HF32(unsafe_from_address=addr))
    ctx.enqueue_function[_root_kernel](buf.unsafe_ptr(), Int32(n), grid_dim=_blocks(n), block_dim=_TPB)
    ctx.enqueue_copy(dst_ptr=_HF32(unsafe_from_address=addr), src_buf=buf)
    ctx.synchronize()
    _ = buf^


# ---- the reference shard merge -----------------------------------------------


@always_inline
def _twiddle(bits: UInt32) -> UInt32:
    return bits ^ (UInt32(0xFFFFFFFF) if (bits & UInt32(0x80000000)) != 0 else UInt32(0x80000000))


@always_inline
def _untwiddle(tw: UInt32) -> UInt32:
    return tw ^ (UInt32(0x80000000) if (tw & UInt32(0x80000000)) != 0 else UInt32(0xFFFFFFFF))


def _topk_merge_kernel(
    dist: _U32, ids: _I64, meta: _I64, S_: Int32, nq_: Int32, k_: Int32,
    keys: _U64, od: _U32, oi: _U32, st: _I32, bad: _I32,
):
    """Row `row`: every shard's candidates as composite keys (twiddled
    distance bits << 32 | global index), the k smallest kept ascending in
    keys[row k ..], then the float insertion sort (ties by index) into
    od/oi. meta per shard s: [offset of its block, first, end, width].
    Status 1 a local id outside its shard, 2 fewer than k candidates."""
    var row = _tid()
    if row >= Int(nq_):
        return
    var S = Int(S_)
    var k = Int(k_)
    var base = row * k
    var filled = 0
    var status = 0
    for r in range(S):
        var off = Int(meta.unsafe_load(4 * r))
        var first = Int(meta.unsafe_load(4 * r + 1))
        var end = Int(meta.unsafe_load(4 * r + 2))
        var width = Int(meta.unsafe_load(4 * r + 3))
        for j in range(width):
            var at_src = off + row * width + j
            var local = Int(ids.unsafe_load(at_src))
            if local < 0 or local >= end - first:
                status = 1
                break
            var key = (UInt64(_twiddle(dist.unsafe_load(at_src))) << 32) | UInt64(first + local)
            var at: Int
            if filled < k:
                at = filled
                filled += 1
            else:
                if not (keys.unsafe_load(base + k - 1) > key):
                    continue
                at = k - 1
            while at > 0 and keys.unsafe_load(base + at - 1) > key:
                keys.unsafe_store(base + at, keys.unsafe_load(base + at - 1))
                at -= 1
            keys.unsafe_store(base + at, key)
        if status != 0:
            break
    if status == 0 and filled < k:
        status = 2
    if status != 0:
        st.unsafe_store(row, Int32(status))
        _ = Atomic[DType.int32].min(bad, Int32(row))
        return
    for a in range(k):
        var kv = keys.unsafe_load(base + a)
        od.unsafe_store(base + a, _untwiddle(UInt32(kv >> 32)))
        oi.unsafe_store(base + a, UInt32(kv & UInt64(0xFFFFFFFF)))
    # estimator.mojo's host insertion sort on the float distance
    for a in range(1, k):
        var tb = od.unsafe_load(base + a)
        var tix = oi.unsafe_load(base + a)
        var dv = bitcast[DType.float32](tb)
        var b = a - 1
        while b >= 0:
            var dprev = bitcast[DType.float32](od.unsafe_load(base + b))
            if dprev < dv or (dprev == dv and oi.unsafe_load(base + b) <= tix):
                break
            od.unsafe_store(base + b + 1, od.unsafe_load(base + b))
            oi.unsafe_store(base + b + 1, oi.unsafe_load(base + b))
            b -= 1
        od.unsafe_store(base + b + 1, tb)
        oi.unsafe_store(base + b + 1, tix)


def device_shard_topk_merge_f32(
    ctx: DeviceContext, table: _HI64, S: Int, nq: Int, k: Int, od_addr: Int, oi_addr: Int,
) raises -> Int:
    """`shard_topk_merge_f32` on the device: 0, or the refusing row's status
    (the lowest refusing row's, as the host loop stops there)."""
    var total = 0
    for r in range(S):  # small-loop(S: one reference shard per device): block offsets, no row data
        total += nq * Int(table[5 * r + 4])
    var dist = ctx.enqueue_create_buffer[DType.uint32](max(total, 1))
    var ids = ctx.enqueue_create_buffer[DType.int64](max(total, 1))
    var hmeta = ctx.enqueue_create_host_buffer[DType.int64](4 * S)
    ctx.synchronize()
    var off = 0
    for r in range(S):  # small-loop(S: one reference shard per device): one copy per shard array, launch arguments only
        var width = Int(table[5 * r + 4])
        hmeta.unsafe_ptr().unsafe_store(4 * r, Int64(off))
        hmeta.unsafe_ptr().unsafe_store(4 * r + 1, table[5 * r + 2])
        hmeta.unsafe_ptr().unsafe_store(4 * r + 2, table[5 * r + 3])
        hmeta.unsafe_ptr().unsafe_store(4 * r + 3, Int64(width))
        if nq * width > 0:
            var dsub = dist.create_sub_buffer[DType.uint32](off, nq * width)
            ctx.enqueue_copy(dst_buf=dsub, src_ptr=_HU32(unsafe_from_address=Int(table[5 * r])))
            var isub = ids.create_sub_buffer[DType.int64](off, nq * width)
            ctx.enqueue_copy(dst_buf=isub, src_ptr=_HI64(unsafe_from_address=Int(table[5 * r + 1])))
            _ = dsub^
            _ = isub^
        off += nq * width
    var meta = ctx.enqueue_create_buffer[DType.int64](4 * S)
    ctx.enqueue_copy(dst_buf=meta, src_ptr=hmeta.unsafe_ptr())
    var keys = ctx.enqueue_create_buffer[DType.uint64](nq * k)
    var od = ctx.enqueue_create_buffer[DType.uint32](nq * k)
    var oi = ctx.enqueue_create_buffer[DType.uint32](nq * k)
    var st = ctx.enqueue_create_buffer[DType.int32](nq)
    var bad = ctx.enqueue_create_buffer[DType.int32](2)
    ctx.enqueue_memset(st, Int32(0))
    ctx.enqueue_memset(bad, _NO_ROW)
    ctx.enqueue_function[_topk_merge_kernel](
        dist.unsafe_ptr(), ids.unsafe_ptr(), meta.unsafe_ptr(), Int32(S), Int32(nq), Int32(k),
        keys.unsafe_ptr(), od.unsafe_ptr(), oi.unsafe_ptr(), st.unsafe_ptr(), bad.unsafe_ptr(),
        grid_dim=_blocks(nq), block_dim=_TPB,
    )
    var b = _read2(ctx, bad)
    var rc = 0
    if b[0] != _NO_ROW:
        var one = st.create_sub_buffer[DType.int32](Int(b[0]), 1)
        var h = ctx.enqueue_create_host_buffer[DType.int32](1)
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=one)
        ctx.synchronize()
        rc = Int(h.unsafe_ptr().unsafe_load(0))
        _ = h^
        _ = one^
    else:
        ctx.enqueue_copy(dst_ptr=_HU32(unsafe_from_address=od_addr), src_buf=od)
        ctx.enqueue_copy(dst_ptr=_HU32(unsafe_from_address=oi_addr), src_buf=oi)
        ctx.synchronize()
    _ = hmeta^
    _ = dist^
    _ = ids^
    _ = meta^
    _ = keys^
    _ = od^
    _ = oi^
    _ = st^
    _ = bad^
    return rc


# ---- the IVF shard plan ----------------------------------------------------------


@always_inline
def _shard_of(i: Int, n: Int, P: Int) -> Int:
    """The shard p holding stored row i: p n // P <= i < (p + 1) n // P."""
    var p = (i * P) // n
    while p + 1 < P and ((p + 1) * n) // P <= i:
        p += 1
    while p > 0 and (p * n) // P > i:
        p -= 1
    return p


def _plan_off_kernel(off: _I32, L_: Int32, n_: Int32, flag: _I32):
    """flag[0] = 1 when the offsets are not 0 .. n nondecreasing."""
    var j = _tid()
    var L = Int(L_)
    if j > L:
        return
    if j == 0 and Int(off.unsafe_load(0)) != 0:
        flag.unsafe_store(0, Int32(1))
    if j == L:
        if Int(off.unsafe_load(L)) != Int(n_):
            flag.unsafe_store(0, Int32(1))
        return
    if off.unsafe_load(j) > off.unsafe_load(j + 1):
        flag.unsafe_store(0, Int32(1))


def _plan_owner_kernel(ind: _I32, n_: Int32, P_: Int32, cnt: _I32, owner: _I32, flag: _I32):
    """Stored row i's original id v: cnt[v] += 1, owner[v] = i's shard;
    an id outside [0, n) sets flag[1]."""
    var i = _tid()
    var n = Int(n_)
    if i >= n:
        return
    var v = Int(ind.unsafe_load(i))
    if v < 0 or v >= n:
        flag.unsafe_store(1, Int32(1))
        return
    _ = Atomic[DType.int32].fetch_add(cnt + v, Int32(1))
    owner.unsafe_store(v, Int32(_shard_of(i, n, Int(P_))))


def _plan_dup_kernel(cnt: _I32, n_: Int32, flag: _I32):
    """flag[1] = 1 when an id is held by other than exactly one row."""
    var v = _tid()
    if v < Int(n_) and cnt.unsafe_load(v) != Int32(1):
        flag.unsafe_store(1, Int32(1))


def _plan_rank_kernel(rows: _I32, c_: Int32, rank: _I32):
    """rank[rows[j]] = j: each id's place among its shard's ascending ids."""
    var j = _tid()
    if j < Int(c_):
        rank.unsafe_store(Int(rows.unsafe_load(j)), Int32(j))


def _plan_local_kernel(ind: _I32, n_: Int32, rank: _I32, loc: _I32):
    var i = _tid()
    if i < Int(n_):
        loc.unsafe_store(i, rank.unsafe_load(Int(ind.unsafe_load(i))))


def _plan_soff_kernel(off: _I32, L1_: Int32, n_: Int32, P_: Int32, soff: _I32):
    """soff[p, j] = max(0, min(off[j], hi_p) - lo_p)."""
    var t = _tid()
    var L1 = Int(L1_)
    var P = Int(P_)
    if t >= P * L1:
        return
    var n = Int(n_)
    var p = t // L1
    var j = t - p * L1
    var lo = p * n // P
    var hi = (p + 1) * n // P
    var o = Int(off.unsafe_load(j))
    var v = (o if o < hi else hi) - lo
    soff.unsafe_store(t, Int32(v if v > 0 else 0))


def device_ivf_shard_plan(
    ctx: DeviceContext, ind_addr: Int, off_addr: Int, soff_addr: Int, loc_addr: Int, map_addr: Int,
    n: Int, n_lists: Int, P: Int,
) raises -> Int:
    """`ivf_shard_plan` on the device: 0 planned (the three outputs
    written), 2 the offsets are not 0 .. n nondecreasing, 1 the ids are not
    a permutation of 0 .. n - 1 (the host's checks, in its order)."""
    var L1 = n_lists + 1
    var off = ctx.enqueue_create_buffer[DType.int32](L1)
    ctx.enqueue_copy(dst_buf=off, src_ptr=_HI32(unsafe_from_address=off_addr))
    var flag = ctx.enqueue_create_buffer[DType.int32](2)
    ctx.enqueue_memset(flag, Int32(0))
    ctx.enqueue_function[_plan_off_kernel](
        off.unsafe_ptr(), Int32(n_lists), Int32(n), flag.unsafe_ptr(),
        grid_dim=_blocks(L1), block_dim=_TPB,
    )
    var f = _read2(ctx, flag)
    if f[0] != 0:
        _ = off^
        _ = flag^
        return 2
    var soff = ctx.enqueue_create_buffer[DType.int32](P * L1)
    ctx.enqueue_function[_plan_soff_kernel](
        off.unsafe_ptr(), Int32(L1), Int32(n), Int32(P), soff.unsafe_ptr(),
        grid_dim=_blocks(P * L1), block_dim=_TPB,
    )
    if n == 0:
        ctx.enqueue_copy(dst_ptr=_HI32(unsafe_from_address=soff_addr), src_buf=soff)
        ctx.synchronize()
        _ = soff^
        _ = off^
        _ = flag^
        return 0
    var ind = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=ind, src_ptr=_HI32(unsafe_from_address=ind_addr))
    var cnt = ctx.enqueue_create_buffer[DType.int32](n)
    var owner = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_memset(cnt, Int32(0))
    ctx.enqueue_memset(owner, Int32(-1))
    ctx.enqueue_function[_plan_owner_kernel](
        ind.unsafe_ptr(), Int32(n), Int32(P), cnt.unsafe_ptr(), owner.unsafe_ptr(), flag.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=_TPB,
    )
    ctx.enqueue_function[_plan_dup_kernel](
        cnt.unsafe_ptr(), Int32(n), flag.unsafe_ptr(), grid_dim=_blocks(n), block_dim=_TPB,
    )
    f = _read2(ctx, flag)
    if f[1] != 0:
        _ = ind^
        _ = cnt^
        _ = owner^
        _ = soff^
        _ = off^
        _ = flag^
        return 1
    var mp = ctx.enqueue_create_buffer[DType.int32](n)
    var rank = ctx.enqueue_create_buffer[DType.int32](n)
    var rows = ctx.enqueue_create_buffer[DType.int32](n)
    for p in range(P):  # small-loop(P: one shard per device): one compaction launch per shard, launch arguments only
        var lo = p * n // P
        var c = device_compact_equal_i32(ctx, owner, n, Int32(p), rows)
        if c > 0:
            var src = rows.create_sub_buffer[DType.int32](0, c)
            var dst = mp.create_sub_buffer[DType.int32](lo, c)
            ctx.enqueue_copy(dst_buf=dst, src_buf=src)
            ctx.enqueue_function[_plan_rank_kernel](
                rows.unsafe_ptr(), Int32(c), rank.unsafe_ptr(), grid_dim=_blocks(c), block_dim=_TPB,
            )
            ctx.synchronize()
            _ = src^
            _ = dst^
    var loc = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_function[_plan_local_kernel](
        ind.unsafe_ptr(), Int32(n), rank.unsafe_ptr(), loc.unsafe_ptr(), grid_dim=_blocks(n), block_dim=_TPB,
    )
    ctx.enqueue_copy(dst_ptr=_HI32(unsafe_from_address=soff_addr), src_buf=soff)
    ctx.enqueue_copy(dst_ptr=_HI32(unsafe_from_address=loc_addr), src_buf=loc)
    ctx.enqueue_copy(dst_ptr=_HI32(unsafe_from_address=map_addr), src_buf=mp)
    ctx.synchronize()
    _ = loc^
    _ = rows^
    _ = rank^
    _ = mp^
    _ = ind^
    _ = cnt^
    _ = owner^
    _ = soff^
    _ = off^
    _ = flag^
    return 0

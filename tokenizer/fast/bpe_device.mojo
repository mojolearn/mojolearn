# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""BPE TRAINING AND ENCODE ON THE APPLE GPU, FAST TIER ONLY.

Ported from lane/apple-fast-bpe (3355d37b3, 2026-10-03) onto the thin-shell layout (lane
port-apple-fast-bpe, 2026-10-21). Kernels and device orchestration only: no Python, no host loop over
data. The typed entries the shell calls, and the host input/output steps around the device work, are
`tokenizer/fast/bpe_fast_impl.mojo`; the Python door is `bindings/_mojolearn_tokenizer_fast.mojo`,
built only by `bindings/build_tokenizer_fast.sh` (FAST only, Apple target). Every launch below sits
under `comptime if` on a switch that is False unless `GLOBAL_NUMERIC_MODE == NUMERIC_FAST and
has_apple_gpu_accelerator()` AND its own `-D MOJOLEARN_BPE_<NAME>` define, so IDENTICAL, the host
binding and a FAST build with no define never compile a kernel of this file.

THE RESULT IS THE HOST TRAINER'S, MERGE FOR MERGE (`tokenizer/train/bpe_train.mojo`): the same
pre-token groups (built by the same host code), the same total order on the selection (highest
count, then the smallest key `left * V + right`), the same left-to-right non-overlapping rewrite,
the same `n_ties_broken`. docs/apple-fast/notes/bpe.md holds the exactness arguments for the
atomic count deltas and for MERGE_BATCH. The encoder gives the host encoder's ids, id for id
(`tokenizer/impl/bpe.mojo::bpe_append` per pre-token), packed back to back per document as the
host binding's `bpe_encode_batch` writes them.

THE SWITCHES (default OFF; docs/apple-fast/ab/bpe.md):
    MOJOLEARN_BPE_TRAIN_DEVICE   the merge loop on the device: one merge per pass, 3 launches a pass,
                                 one 32-byte state readback per BPE_CHUNK passes
    MOJOLEARN_BPE_MERGE_BATCH    (implies TRAIN_DEVICE) up to BPE_K - 1 exact merges per pass from a
                                 top-BPE_K selection list
    MOJOLEARN_BPE_GROUP_FILTER   (implies TRAIN_DEVICE) a 64-bit token-presence mask per group, so the
                                 apply pass skips groups that cannot hold a selected pair
    MOJOLEARN_BPE_ENCODE_DEVICE  encode_batch's per-pre-token min-rank merge on the device, one thread
                                 per pre-token, then a three-kernel scan packs the ids per document
    MOJOLEARN_BPE_LIVEBUF        (implies TRAIN_DEVICE and ENCODE_DEVICE) grow-only pooled buffers kept
                                 across calls (3 live buffers instead of ~12 per call) and the rank
                                 table kept on the device for the tokenizer's life
    MOJOLEARN_BPE_ALL            every switch above
"""
from std.atomic import Atomic, Ordering
from std.ffi import _Global
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from core.neural_context import process_ctx
from tokenizer.impl.ranks import RankTable
from tokenizer.train.bpe_train import PieceGroups

comptime _FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
comptime _ALL = is_defined["MOJOLEARN_BPE_ALL"]()
comptime BPE_LIVEBUF = _FAST_APPLE and (_ALL or is_defined["MOJOLEARN_BPE_LIVEBUF"]())
comptime BPE_MERGE_BATCH = _FAST_APPLE and (_ALL or is_defined["MOJOLEARN_BPE_MERGE_BATCH"]())
comptime BPE_GROUP_FILTER = _FAST_APPLE and (_ALL or is_defined["MOJOLEARN_BPE_GROUP_FILTER"]())
comptime BPE_TRAIN_DEVICE = _FAST_APPLE and (
    is_defined["MOJOLEARN_BPE_TRAIN_DEVICE"]() or BPE_MERGE_BATCH or BPE_GROUP_FILTER or BPE_LIVEBUF or _ALL
)
comptime BPE_ENCODE_DEVICE = _FAST_APPLE and (
    is_defined["MOJOLEARN_BPE_ENCODE_DEVICE"]() or BPE_LIVEBUF or _ALL
)

comptime I32P = MutPointer[Int32, MutAnyOrigin]
comptime I64P = MutPointer[Int64, MutAnyOrigin]
comptime U8P = MutPointer[UInt8, MutAnyOrigin]

#: threads per block of the per-group, per-pre-token and per-document kernels
comptime BPE_TPB = 128
#: the selection list's length: the winner and its runner-up (the tie test), or the top 8
comptime BPE_K = 8 if BPE_MERGE_BATCH else 2
#: blocks (and threads per block) of the top-K partial pass; the selection kernel is ONE
#: threadgroup of BPE_RB threads over BPE_RB constant-size partial lists
comptime BPE_RB = 256
comptime BPE_RT = 256
#: passes enqueued between two reads of the state words
comptime BPE_CHUNK = 64
#: the largest vocabulary whose pair key `left * V + right` (+1) fits an int32
comptime BPE_MAX_VOCAB = 46340
comptime KEY_NONE = Int32(2147483647)

# Fits gate: the two shared K-lists of one threadgroup (counts and keys, int32) in Apple's 32 KiB,
# asserted at the top of bpe_topk_part_kernel and bpe_select_kernel (comptime assert needs a function).

# state words
comptime S_DONE = 0
comptime S_NM = 1
comptime S_TIES = 2
comptime S_NSEL = 3
comptime S_NOCC = 4
comptime S_FAIL = 5
comptime S_BASE = 6
comptime S_TOTAL = 7
comptime S_LEN = 8

comptime FNV_OFFSET64: UInt64 = 14695981039346656037
comptime FNV_PRIME64: UInt64 = 1099511628211


# ------------------------------------------------------------------ device helpers


@always_inline
def _gtid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _blocks(count: Int) -> Int:
    return (count + BPE_TPB - 1) // BPE_TPB if count > 0 else 1


@always_inline
def _better(c1: Int32, k1: Int32, c2: Int32, k2: Int32) -> Bool:
    """(c1, k1) comes first in the selection order: higher count, then smaller key."""
    return c1 > c2 or (c1 == c2 and k1 < k2)


@always_inline
def _pslot(key: Int32, mask: Int) -> Int:
    # Fibonacci hashing, as PairTable._slot: the hash reaches only WHERE a pair sits.
    var h = UInt64(UInt32(key)) * UInt64(11400714819323198485)
    return Int((h >> 32) ^ h) & mask


@always_inline
def _pair_add(
    keys: I32P, vals: I32P, occ: I32P, st: I32P, key: Int32, delta: Int32, mask: Int, occ_cap: Int
):
    """Add `delta` to the pair `key`'s count; a missing key is inserted (an empty slot claimed by
    compare-exchange, its index appended to `occ`). A subtraction always finds its key: it was
    inserted by an earlier pass and nothing is ever deleted. Integer adds commute, so the table
    after a pass does not depend on the order threads ran in."""
    var want = key + Int32(1)
    var s = _pslot(key, mask)
    var probes = 0
    while probes <= mask:
        var cur = Atomic.load[ordering=Ordering.RELAXED](keys.unsafe_offset(s))
        if cur == want:
            _ = Atomic.fetch_add(vals.unsafe_offset(s), delta)
            return
        if cur == Int32(0):
            if delta < Int32(0):
                st.unsafe_store(S_FAIL, Int32(1))
                return
            var expected = Int32(0)
            if Atomic.compare_exchange[
                success_ordering=Ordering.RELAXED,
                failure_ordering=Ordering.RELAXED,
                weak=True,
            ](keys.unsafe_offset(s), expected, want):
                var at = Int(Atomic.fetch_add(st.unsafe_offset(S_NOCC), Int32(1)))
                if at < occ_cap:
                    occ.unsafe_store(at, Int32(s))
                else:
                    st.unsafe_store(S_FAIL, Int32(2))
                _ = Atomic.fetch_add(vals.unsafe_offset(s), delta)
                return
            # lost the slot to another thread (or a spurious weak failure): look at it again
            continue
        s = (s + 1) & mask
        probes += 1
    st.unsafe_store(S_FAIL, Int32(3))


@always_inline
def _bit_has(lo: UInt32, hi: UInt32, t: Int32) -> Bool:
    var b = Int(t & Int32(63))
    if b < 32:
        return ((lo >> UInt32(b)) & UInt32(1)) != UInt32(0)
    return ((hi >> UInt32(b - 32)) & UInt32(1)) != UInt32(0)


@always_inline
def _match(
    sa: InlineArray[Int32, BPE_K], sb: InlineArray[Int32, BPE_K], nsel: Int, x: Int32, y: Int32
) -> Int:
    """Which selected pair `(x, y)` is, or -1. Selected pairs share no token, so at most one."""
    for j in range(BPE_K):
        if j < nsel and sa[j] == x and sb[j] == y:
            return j
    return -1


# ------------------------------------------------------------------ training kernels


def bpe_fill_kernel(p: I32P, words: Int64, v: Int32):
    var t = _gtid()
    if t < Int(words):
        p.unsafe_store(t, v)


def bpe_group_init_kernel(
    arena: U8P, goff: I64P, glen0: I64P, gcnt: I64P, seq: I32P, glen: I32P, gmask: I32P,
    keys: I32P, vals: I32P, occ: I32P, st: I32P, ngroups: Int64, vocab: Int64, mask: Int64, occ_cap: Int64,
):
    """One thread per group: its bytes become ids, its pairs are counted (x the group's count),
    and its token-presence mask is set."""
    var g = _gtid()
    if g >= Int(ngroups):
        return
    var gs = Int(goff[g])
    var glen_g = Int(glen0[g])
    var c = Int32(gcnt[g])
    var v32 = Int32(vocab)
    var lo = UInt32(0)
    var hi = UInt32(0)
    for i in range(glen_g):
        var x = Int32(arena[gs + i])
        seq.unsafe_store(gs + i, x)
        var b = Int(x & Int32(63))
        if b < 32:
            lo |= UInt32(1) << UInt32(b)
        else:
            hi |= UInt32(1) << UInt32(b - 32)
    glen.unsafe_store(g, Int32(glen_g))
    gmask.unsafe_store(2 * g, bitcast[DType.int32](lo))
    gmask.unsafe_store(2 * g + 1, bitcast[DType.int32](hi))
    for i in range(glen_g - 1):
        _pair_add(keys, vals, occ, st, seq[gs + i] * v32 + seq[gs + i + 1], c, Int(mask), Int(occ_cap))


def bpe_topk_part_kernel(keys: I32P, vals: I32P, occ: I32P, st: I32P, pc: I32P, pk: I32P, occ_cap: Int64):
    """Block b's top BPE_K live pairs, in selection order, into pc/pk[b * BPE_K ...]: each thread
    keeps a sorted list over its strided entries, then the block merges the lists pairwise."""
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    comptime assert BPE_RT * BPE_K * 8 <= 32768, "bpe_device: the top-K partial lists exceed 32 KiB of threadgroup memory"
    var sc = stack_allocation[BPE_RT * BPE_K, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var sk = stack_allocation[BPE_RT * BPE_K, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var lc = InlineArray[Int32, BPE_K](fill=Int32(0))
    var lk = InlineArray[Int32, BPE_K](fill=KEY_NONE)
    var live = min(Int(st[S_NOCC]), Int(occ_cap))
    if st[S_DONE] != Int32(0):
        live = 0
    var i = b * BPE_RT + tid
    while i < live:
        var s = Int(occ[i])
        var c = vals[s]
        var k = keys[s] - Int32(1)
        if c > Int32(0) and _better(c, k, lc[BPE_K - 1], lk[BPE_K - 1]):
            var at = BPE_K - 1
            while at > 0 and _better(c, k, lc[at - 1], lk[at - 1]):
                lc[at] = lc[at - 1]
                lk[at] = lk[at - 1]
                at -= 1
            lc[at] = c
            lk[at] = k
        i += BPE_RB * BPE_RT
    for j in range(BPE_K):
        sc[tid * BPE_K + j] = lc[j]
        sk[tid * BPE_K + j] = lk[j]
    barrier()
    var half = BPE_RT // 2
    while half > 0:
        if tid < half:
            var oc = InlineArray[Int32, BPE_K](fill=Int32(0))
            var ok = InlineArray[Int32, BPE_K](fill=KEY_NONE)
            var x = 0
            var y = 0
            var da = tid * BPE_K
            var db = (tid + half) * BPE_K
            for o in range(BPE_K):
                if _better(sc[da + x], sk[da + x], sc[db + y], sk[db + y]):
                    oc[o] = sc[da + x]
                    ok[o] = sk[da + x]
                    x += 1
                else:
                    oc[o] = sc[db + y]
                    ok[o] = sk[db + y]
                    y += 1
            for o in range(BPE_K):
                sc[da + o] = oc[o]
                sk[da + o] = ok[o]
        barrier()
        half //= 2
    if tid < BPE_K:
        pc.unsafe_store(b * BPE_K + tid, sc[tid])
        pk.unsafe_store(b * BPE_K + tid, sk[tid])


def bpe_select_kernel(pc: I32P, pk: I32P, st: I32P, sel: I32P, mrg: I32P, vocab: Int64, vocab_size: Int64, minf: Int64):
    """ONE threadgroup of BPE_RB threads over the BPE_RB partial lists (a constant size): the global
    top BPE_K, then thread 0 picks this pass's merges (the host's winner, plus under MERGE_BATCH the
    exact batch of docs/apple-fast/notes/bpe.md) and records them."""
    var tid = Int(thread_idx.x)
    comptime assert BPE_RB * BPE_K * 8 <= 32768, "bpe_device: the selection lists exceed 32 KiB of threadgroup memory"
    var sc = stack_allocation[BPE_RB * BPE_K, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var sk = stack_allocation[BPE_RB * BPE_K, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    for j in range(BPE_K):
        sc[tid * BPE_K + j] = pc[tid * BPE_K + j]
        sk[tid * BPE_K + j] = pk[tid * BPE_K + j]
    barrier()
    var half = BPE_RB // 2
    while half > 0:
        if tid < half:
            var oc = InlineArray[Int32, BPE_K](fill=Int32(0))
            var ok = InlineArray[Int32, BPE_K](fill=KEY_NONE)
            var x = 0
            var y = 0
            var da = tid * BPE_K
            var db = (tid + half) * BPE_K
            for o in range(BPE_K):
                if _better(sc[da + x], sk[da + x], sc[db + y], sk[db + y]):
                    oc[o] = sc[da + x]
                    ok[o] = sk[da + x]
                    x += 1
                else:
                    oc[o] = sc[db + y]
                    ok[o] = sk[db + y]
                    y += 1
            for o in range(BPE_K):
                sc[da + o] = oc[o]
                sk[da + o] = ok[o]
        barrier()
        half //= 2
    if tid != 0:
        return
    if st[S_DONE] != Int32(0):
        st.unsafe_store(S_NSEL, Int32(0))
        return
    var v32 = Int32(vocab)
    var nm = Int(st[S_NM])
    var room = Int(vocab_size) - 256 - nm
    var c0 = sc[0]
    if room <= 0 or c0 <= Int32(0) or Int(c0) < Int(minf):
        st.unsafe_store(S_DONE, Int32(1))
        st.unsafe_store(S_NSEL, Int32(0))
        return
    var a0 = sk[0] // v32
    var b0 = sk[0] % v32
    sel.unsafe_store(0, a0)
    sel.unsafe_store(1, b0)
    mrg.unsafe_store(2 * nm, a0)
    mrg.unsafe_store(2 * nm + 1, b0)
    if sc[1] == c0:
        st.unsafe_store(S_TIES, st[S_TIES] + Int32(1))
    var ns = 1
    comptime if BPE_MERGE_BATCH:
        var open = a0 != b0
        for j in range(1, BPE_K - 1):
            if not open or ns >= room:
                break
            var c = sc[j]
            if c <= Int32(0) or Int(c) < Int(minf) or not (c > sc[j + 1]):
                break
            var a = sk[j] // v32
            var bb = sk[j] % v32
            var touch = False
            for q in range(BPE_K):
                if q < ns:
                    var qa = sel[2 * q]
                    var qb = sel[2 * q + 1]
                    if a == qa or a == qb or bb == qa or bb == qb:
                        touch = True
            if touch:
                break
            sel.unsafe_store(2 * ns, a)
            sel.unsafe_store(2 * ns + 1, bb)
            mrg.unsafe_store(2 * (nm + ns), a)
            mrg.unsafe_store(2 * (nm + ns) + 1, bb)
            ns += 1
            if a == bb:
                open = False
    st.unsafe_store(S_BASE, Int32(256 + nm))
    st.unsafe_store(S_NSEL, Int32(ns))
    st.unsafe_store(S_NM, Int32(nm + ns))


def bpe_apply_kernel(
    seq: I32P, glen: I32P, goff: I64P, gcnt: I64P, gmask: I32P, keys: I32P, vals: I32P, occ: I32P,
    st: I32P, sel: I32P, ngroups: Int64, vocab: Int64, mask: Int64, occ_cap: Int64,
):
    """One thread per group: the host's left-to-right non-overlapping rewrite of this pass's merges,
    in place, with the exact count deltas (notes: each changed old pair subtracted once, each new
    pair added once)."""
    var g = _gtid()
    if g >= Int(ngroups):
        return
    var nsel = Int(st[S_NSEL])
    if nsel == 0:
        return
    var glen_g = Int(glen[g])
    if glen_g < 2:
        return
    var sa = InlineArray[Int32, BPE_K](fill=Int32(-1))
    var sb = InlineArray[Int32, BPE_K](fill=Int32(-1))
    for j in range(BPE_K):
        if j < nsel:
            sa[j] = sel[2 * j]
            sb[j] = sel[2 * j + 1]
    var lo = bitcast[DType.uint32](gmask[2 * g])
    var hi = bitcast[DType.uint32](gmask[2 * g + 1])
    comptime if BPE_GROUP_FILTER:
        var maybe = False
        for j in range(BPE_K):
            if j < nsel and _bit_has(lo, hi, sa[j]) and _bit_has(lo, hi, sb[j]):
                maybe = True
        if not maybe:
            return
    var gs = Int(goff[g])
    var k0 = -1
    for k in range(glen_g - 1):
        if _match(sa, sb, nsel, seq[gs + k], seq[gs + k + 1]) >= 0:
            k0 = k
            break
    if k0 < 0:
        return
    var c = Int32(gcnt[g])
    var v32 = Int32(vocab)
    var base = st[S_BASE]
    var m = Int(mask)
    var cap = Int(occ_cap)
    var prev_old = Int32(-1)
    if k0 > 0:
        prev_old = seq[gs + k0 - 1]
    var prev_new = prev_old
    var r = k0
    var w = k0
    while r < glen_g:
        var x = seq[gs + r]
        var j = -1
        if r + 1 < glen_g:
            j = _match(sa, sb, nsel, x, seq[gs + r + 1])
        if j >= 0:
            var y = seq[gs + r + 1]
            var nt = base + Int32(j)
            if r > 0:
                _pair_add(keys, vals, occ, st, prev_old * v32 + x, -c, m, cap)
            _pair_add(keys, vals, occ, st, x * v32 + y, -c, m, cap)
            if r + 2 < glen_g:
                var z = seq[gs + r + 2]
                var next_merges = False
                if r + 3 < glen_g and _match(sa, sb, nsel, z, seq[gs + r + 3]) >= 0:
                    next_merges = True
                if not next_merges:
                    _pair_add(keys, vals, occ, st, y * v32 + z, -c, m, cap)
                    _pair_add(keys, vals, occ, st, nt * v32 + z, c, m, cap)
            if w > 0:
                _pair_add(keys, vals, occ, st, prev_new * v32 + nt, c, m, cap)
            prev_old = y
            seq.unsafe_store(gs + w, nt)
            prev_new = nt
            var bt = Int(nt & Int32(63))
            if bt < 32:
                lo |= UInt32(1) << UInt32(bt)
            else:
                hi |= UInt32(1) << UInt32(bt - 32)
            w += 1
            r += 2
        else:
            prev_old = x
            seq.unsafe_store(gs + w, x)
            prev_new = x
            w += 1
            r += 1
    glen.unsafe_store(g, Int32(w))
    gmask.unsafe_store(2 * g, bitcast[DType.int32](lo))
    gmask.unsafe_store(2 * g + 1, bitcast[DType.int32](hi))


# ------------------------------------------------------------------ encode kernels


@always_inline
def _rank(text: U8P, at: Int, cnt: Int, arena: U8P, toff: I64P, tlen: I64P, buckets: I64P, tmask: Int) -> Int:
    """RankTable.rank on the device: the same FNV-1a hash, the same linear probe."""
    if cnt <= 0:
        return -1
    var h = FNV_OFFSET64
    for i in range(cnt):
        h = (h ^ UInt64(text[at + i])) * FNV_PRIME64
    var s = Int(h & UInt64(tmask))
    var probes = 0
    while probes <= tmask:
        var bk = Int(buckets[s])
        if bk == 0:
            return -1
        var id = bk - 1
        if Int(tlen[id]) == cnt:
            var o = Int(toff[id])
            var eq = True
            for i in range(cnt):
                if arena[o + i] != text[at + i]:
                    eq = False
                    break
            if eq:
                return id
        s = (s + 1) & tmask
        probes += 1
    return -1


def bpe_encode_pre_kernel(
    text: U8P, pb: I32P, arena: U8P, toff: I64P, tlen: I64P, buckets: I64P, bnd: I32P, outi: I32P,
    cnt: I32P, st: I32P, npre: Int64, tmask: Int64,
):
    """One thread per pre-token: `tokenizer/impl/bpe.mojo::bpe_append`, the same loop (merge the
    lowest-rank adjacent pair until none has a rank). Ids go to outi[start ...]; the boundary list
    lives in bnd[start + p ...] (start + p is unique per pre-token and pre-token lengths sum to n)."""
    var p = _gtid()
    if p >= Int(npre):
        return
    var s = Int(pb[p])
    var e = Int(pb[p + 1])
    var plen = e - s
    var tm = Int(tmask)
    var whole = _rank(text, s, plen, arena, toff, tlen, buckets, tm)
    if whole >= 0:
        outi.unsafe_store(s, Int32(whole))
        cnt.unsafe_store(p, Int32(1))
        return
    var bo = s + p
    for i in range(plen + 1):
        bnd.unsafe_store(bo + i, Int32(s + i))
    var nb = plen + 1
    while nb > 2:
        var best_rank = -1
        var best_at = -1
        for k in range(nb - 2):
            var lo = Int(bnd[bo + k])
            var r = _rank(text, lo, Int(bnd[bo + k + 2]) - lo, arena, toff, tlen, buckets, tm)
            if r >= 0 and (best_at < 0 or r < best_rank):
                best_rank = r
                best_at = k
        if best_at < 0:
            break
        for k in range(best_at + 1, nb - 1):
            bnd.unsafe_store(bo + k, bnd[bo + k + 1])
        nb -= 1
    for k in range(nb - 1):
        var lo = Int(bnd[bo + k])
        var id = _rank(text, lo, Int(bnd[bo + k + 1]) - lo, arena, toff, tlen, buckets, tm)
        if id < 0:
            st.unsafe_store(S_FAIL, Int32(1))
            id = 0
        outi.unsafe_store(s + k, Int32(id))
    cnt.unsafe_store(p, Int32(nb - 1))


def bpe_scan_block_kernel(cnt: I32P, pfx: I32P, bsum: I32P, npre: Int64):
    """Block b's exclusive prefix of the per-pre-token id counts into pfx, and its total into
    bsum[b]: a Hillis-Steele scan of BPE_TPB entries in threadgroup memory. Integer adds, so the
    result does not depend on the order the adds ran in."""
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var i = b * BPE_TPB + tid
    var sh = stack_allocation[BPE_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var own = Int32(0)
    if i < Int(npre):
        own = cnt[i]
    sh[tid] = own
    barrier()
    var off = 1
    while off < BPE_TPB:
        var add = Int32(0)
        if tid >= off:
            add = sh[tid - off]
        barrier()
        sh[tid] = sh[tid] + add
        barrier()
        off *= 2
    if i < Int(npre):
        pfx.unsafe_store(i, sh[tid] - own)
    if tid == BPE_TPB - 1:
        bsum.unsafe_store(b, sh[tid])


def bpe_scan_sums_kernel(bsum: I32P, st: I32P, nblk: Int64):
    """ONE threadgroup of BPE_RT threads over the nblk block totals (pre-tokens / BPE_TPB of them):
    each thread sums a contiguous chunk, the chunk sums are scanned in threadgroup memory, and each
    thread rewrites its chunk as the exclusive prefix. The grand total goes to st[S_TOTAL]."""
    var tid = Int(thread_idx.x)
    var nb = Int(nblk)
    var per = (nb + BPE_RT - 1) // BPE_RT
    var lo = min(tid * per, nb)
    var hi = min(lo + per, nb)
    var sh = stack_allocation[BPE_RT, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var own = Int32(0)
    for j in range(lo, hi):
        own += bsum[j]
    sh[tid] = own
    barrier()
    var off = 1
    while off < BPE_RT:
        var add = Int32(0)
        if tid >= off:
            add = sh[tid - off]
        barrier()
        sh[tid] = sh[tid] + add
        barrier()
        off *= 2
    var run = sh[tid] - own
    for j in range(lo, hi):
        var v = bsum[j]
        bsum.unsafe_store(j, run)
        run += v
    if tid == BPE_RT - 1:
        st.unsafe_store(S_TOTAL, sh[tid])


def bpe_scan_add_kernel(pfx: I32P, bsum: I32P, st: I32P, npre: Int64):
    """pfx becomes the global exclusive prefix (npre + 1 entries, the last one the total)."""
    var i = _gtid()
    if i < Int(npre):
        pfx.unsafe_store(i, pfx[i] + bsum[i // BPE_TPB])
    elif i == Int(npre):
        pfx.unsafe_store(i, st[S_TOTAL])


def bpe_encode_pack_kernel(pb: I32P, cnt: I32P, pfx: I32P, outi: I32P, packed: I32P, npre: Int64):
    """One thread per pre-token: its ids (at its byte offset in outi) to its place in the packed
    stream. Documents are consecutive runs of pre-tokens, so the packed stream is every document's
    ids back to back, in document order."""
    var p = _gtid()
    if p >= Int(npre):
        return
    var s = Int(pb[p])
    var c = Int(cnt[p])
    var at = Int(pfx[p])
    for j in range(c):
        packed.unsafe_store(at + j, outi[s + j])


def bpe_encode_counts_kernel(dp: I32P, pfx: I32P, counts: I64P, ndocs: Int64):
    """One thread per document: its id count, the prefix difference over its pre-tokens."""
    var d = _gtid()
    if d >= Int(ndocs):
        return
    counts.unsafe_store(d, Int64(pfx[Int(dp[d + 1])] - pfx[Int(dp[d])]))


# ------------------------------------------------------------------ host side: context and buffers


def bpe_ctx() raises -> DeviceContext:
    """ONE process-lifetime DeviceContext for this binding (buffers must not outlive their context,
    and a context per call exhausts Metal's queues)."""
    return process_ctx["MojoTokenizerFastContext"]()


@fieldwise_init
struct _Slot(Copyable, Movable):
    var idx: Int
    var off: Int
    var count: Int


struct BpeMem(Defaultable, Movable):
    """Device memory for one call. Unpooled (the default): one buffer per array, dropped after the
    call. Pooled (LIVEBUF): three grow-only buffers (u8, i32, i64) kept across calls, each array a
    range inside one of them, so a call allocates nothing once warm and holds 3 live buffers."""

    var b8: List[DeviceBuffer[DType.uint8]]
    var b32: List[DeviceBuffer[DType.int32]]
    var b64: List[DeviceBuffer[DType.int64]]
    var cap8: Int
    var cap32: Int
    var cap64: Int
    var o8: Int
    var o32: Int
    var o64: Int
    var pooled: Bool

    def __init__(out self):
        self.b8 = List[DeviceBuffer[DType.uint8]]()
        self.b32 = List[DeviceBuffer[DType.int32]]()
        self.b64 = List[DeviceBuffer[DType.int64]]()
        self.cap8 = 0
        self.cap32 = 0
        self.cap64 = 0
        self.o8 = 0
        self.o32 = 0
        self.o64 = 0
        self.pooled = False

    def begin(mut self, ctx: DeviceContext, n8: Int, n32: Int, n64: Int, pooled: Bool) raises:
        """Start a call; pooled, grow each pool to hold its total (word counts include the +1
        each empty array is given)."""
        self.pooled = pooled
        self.o8 = 0
        self.o32 = 0
        self.o64 = 0
        if not pooled:
            return
        if self.cap8 < n8:
            self.b8.clear()
            self.b8.append(ctx.enqueue_create_buffer[DType.uint8](n8))
            self.cap8 = n8
        if self.cap32 < n32:
            self.b32.clear()
            self.b32.append(ctx.enqueue_create_buffer[DType.int32](n32))
            self.cap32 = n32
        if self.cap64 < n64:
            self.b64.clear()
            self.b64.append(ctx.enqueue_create_buffer[DType.int64](n64))
            self.cap64 = n64

    def a8(mut self, ctx: DeviceContext, count: Int) raises -> _Slot:
        var m = max(count, 1)
        if self.pooled:
            if self.o8 + m > self.cap8:
                raise Error("bpe_device: u8 pool overrun")
            var s = _Slot(0, self.o8, m)
            self.o8 += m
            return s^
        self.b8.append(ctx.enqueue_create_buffer[DType.uint8](m))
        return _Slot(len(self.b8) - 1, 0, m)

    def a32(mut self, ctx: DeviceContext, count: Int) raises -> _Slot:
        var m = max(count, 1)
        if self.pooled:
            if self.o32 + m > self.cap32:
                raise Error("bpe_device: i32 pool overrun")
            var s = _Slot(0, self.o32, m)
            self.o32 += m
            return s^
        self.b32.append(ctx.enqueue_create_buffer[DType.int32](m))
        return _Slot(len(self.b32) - 1, 0, m)

    def a64(mut self, ctx: DeviceContext, count: Int) raises -> _Slot:
        var m = max(count, 1)
        if self.pooled:
            if self.o64 + m > self.cap64:
                raise Error("bpe_device: i64 pool overrun")
            var s = _Slot(0, self.o64, m)
            self.o64 += m
            return s^
        self.b64.append(ctx.enqueue_create_buffer[DType.int64](m))
        return _Slot(len(self.b64) - 1, 0, m)

    def p8(mut self, s: _Slot) -> U8P:
        return self.b8[s.idx].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]() + s.off

    def p32(mut self, s: _Slot) -> I32P:
        return self.b32[s.idx].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]() + s.off

    def p64(mut self, s: _Slot) -> I64P:
        return self.b64[s.idx].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]() + s.off

    def up8(self, ctx: DeviceContext, s: _Slot, src_addr: Int, count: Int) raises:
        if count > 0:
            ctx.enqueue_copy(
                dst_buf=self.b8[s.idx].create_sub_buffer[DType.uint8](s.off, count),
                src_ptr=U8P(unsafe_from_address=src_addr),
            )

    def up32(self, ctx: DeviceContext, s: _Slot, src_addr: Int, count: Int) raises:
        if count > 0:
            ctx.enqueue_copy(
                dst_buf=self.b32[s.idx].create_sub_buffer[DType.int32](s.off, count),
                src_ptr=I32P(unsafe_from_address=src_addr),
            )

    def up64(self, ctx: DeviceContext, s: _Slot, src_addr: Int, count: Int) raises:
        if count > 0:
            ctx.enqueue_copy(
                dst_buf=self.b64[s.idx].create_sub_buffer[DType.int64](s.off, count),
                src_ptr=I64P(unsafe_from_address=src_addr),
            )

    def down32(self, ctx: DeviceContext, s: _Slot, dst_addr: Int, count: Int) raises:
        if count > 0:
            ctx.enqueue_copy(
                dst_ptr=I32P(unsafe_from_address=dst_addr),
                src_buf=self.b32[s.idx].create_sub_buffer[DType.int32](s.off, count),
            )

    def down64(self, ctx: DeviceContext, s: _Slot, dst_addr: Int, count: Int) raises:
        if count > 0:
            ctx.enqueue_copy(
                dst_ptr=I64P(unsafe_from_address=dst_addr),
                src_buf=self.b64[s.idx].create_sub_buffer[DType.int64](s.off, count),
            )


comptime BPE_TRAIN_POOL = _Global[StorageType=BpeMem, name="MojoTokenizerFastTrainPool", init_fn=BpeMem.__init__]
comptime BPE_ENCODE_POOL = _Global[StorageType=BpeMem, name="MojoTokenizerFastEncodePool", init_fn=BpeMem.__init__]


def _fill(ctx: DeviceContext, p: I32P, words: Int, v: Int32) raises:
    ctx.enqueue_function[bpe_fill_kernel](p, Int64(words), v, grid_dim=_blocks(words), block_dim=BPE_TPB)


# ------------------------------------------------------------------ host side: training


struct BpeMerges(Movable):
    """What the device trainer hands back: the merges as (left, right) id pairs back to back
    (2 * n_merges int32), and the host trainer's tie count."""

    var pairs: List[Int32]
    var n_merges: Int
    var n_ties: Int

    def __init__(out self, n_words: Int):
        self.pairs = List[Int32](length=max(n_words, 1), fill=Int32(0))
        self.n_merges = 0
        self.n_ties = 0


def _train_passes(
    mut mem: BpeMem, ctx: DeviceContext, groups: PieceGroups, vocab_size: Int, min_frequency: Int, ts: Int,
    occ_cap: Int, mut h_st: List[Int32], mut h_mrg: List[Int32],
) raises:
    """Upload the groups, count every pair, run passes until the selection says done, read the
    state words into h_st and the merges into h_mrg."""
    var ng = groups.n()
    var syms = len(groups.arena)
    var mcap = 2 * (vocab_size - 256)
    var s_arena = mem.a8(ctx, syms)
    var s_goff = mem.a64(ctx, ng)
    var s_glen0 = mem.a64(ctx, ng)
    var s_gcnt = mem.a64(ctx, ng)
    var s_seq = mem.a32(ctx, syms)
    var s_glen = mem.a32(ctx, ng)
    var s_gmask = mem.a32(ctx, 2 * ng)
    var s_keys = mem.a32(ctx, ts)
    var s_vals = mem.a32(ctx, ts)
    var s_occ = mem.a32(ctx, occ_cap)
    var s_st = mem.a32(ctx, S_LEN)
    var s_sel = mem.a32(ctx, 2 * BPE_K)
    var s_mrg = mem.a32(ctx, mcap)
    var s_pc = mem.a32(ctx, BPE_RB * BPE_K)
    var s_pk = mem.a32(ctx, BPE_RB * BPE_K)
    mem.up8(ctx, s_arena, Int(groups.arena.unsafe_ptr()), syms)
    mem.up64(ctx, s_goff, Int(groups.offset.unsafe_ptr()), ng)
    mem.up64(ctx, s_glen0, Int(groups.length.unsafe_ptr()), ng)
    mem.up64(ctx, s_gcnt, Int(groups.count.unsafe_ptr()), ng)
    var keys = mem.p32(s_keys)
    var vals = mem.p32(s_vals)
    var occ = mem.p32(s_occ)
    var st = mem.p32(s_st)
    _fill(ctx, keys, ts, Int32(0))
    _fill(ctx, vals, ts, Int32(0))
    _fill(ctx, st, S_LEN, Int32(0))
    var gb = _blocks(ng)
    var mask = ts - 1
    ctx.enqueue_function[bpe_group_init_kernel](
        mem.p8(s_arena), mem.p64(s_goff), mem.p64(s_glen0), mem.p64(s_gcnt), mem.p32(s_seq), mem.p32(s_glen),
        mem.p32(s_gmask), keys, vals, occ, st, Int64(ng), Int64(vocab_size), Int64(mask), Int64(occ_cap),
        grid_dim=gb, block_dim=BPE_TPB,
    )
    # One state readback per BPE_CHUNK passes; a pass after the selection said done is a no-op.
    while True:
        for _c in range(BPE_CHUNK):
            ctx.enqueue_function[bpe_topk_part_kernel](
                keys, vals, occ, st, mem.p32(s_pc), mem.p32(s_pk), Int64(occ_cap),
                grid_dim=BPE_RB, block_dim=BPE_RT,
            )
            # One threadgroup over the BPE_RB constant-size partial lists (BPE_RB * BPE_K entries).
            ctx.enqueue_function[bpe_select_kernel](
                mem.p32(s_pc), mem.p32(s_pk), st, mem.p32(s_sel), mem.p32(s_mrg), Int64(vocab_size),
                Int64(vocab_size), Int64(min_frequency),
                grid_dim=1, block_dim=BPE_RB,
            )
            ctx.enqueue_function[bpe_apply_kernel](
                mem.p32(s_seq), mem.p32(s_glen), mem.p64(s_goff), mem.p64(s_gcnt), mem.p32(s_gmask), keys, vals,
                occ, st, mem.p32(s_sel), Int64(ng), Int64(vocab_size), Int64(mask), Int64(occ_cap),
                grid_dim=gb, block_dim=BPE_TPB,
            )
        mem.down32(ctx, s_st, Int(h_st.unsafe_ptr()), S_LEN)
        ctx.synchronize()
        if h_st[S_FAIL] != Int32(0):
            raise Error("bpe_train_device: the pair table failed (code " + String(Int(h_st[S_FAIL])) + ")")
        if h_st[S_DONE] != Int32(0):
            break
    var nm = Int(h_st[S_NM])
    mem.down32(ctx, s_mrg, Int(h_mrg.unsafe_ptr()), 2 * nm)
    ctx.synchronize()


def bpe_train_merges_device(groups: PieceGroups, vocab_size: Int, min_frequency: Int) raises -> BpeMerges:
    """The host trainer's merge loop on the GPU over `groups` (train_bpe's step 1, built by the
    caller): passes of 3 launches each, one state readback per BPE_CHUNK passes, one download of
    the merges at the end."""
    if vocab_size < 256 or vocab_size > BPE_MAX_VOCAB:
        raise Error("bpe_train_device: vocab_size " + String(vocab_size) + " outside [256, 46340]")
    if min_frequency < 1:
        raise Error("bpe_train_device: min_frequency must be at least 1")
    var ng = groups.n()
    var syms = len(groups.arena)
    var mcap = 2 * (vocab_size - 256)
    var out = BpeMerges(mcap)
    if ng == 0 or vocab_size == 256 or syms - ng < 1:
        return out^
    # Distinct pairs ever seen <= 3 * symbols (initial pairs, plus at most 2 new per merge
    # occurrence, each of which removes a symbol), so a table of 2x that keeps the load <= 1/2.
    var occ_cap = 3 * syms + 16
    var ts = 1024
    while ts < 2 * occ_cap:
        ts *= 2
    var ctx = bpe_ctx()
    var n8 = syms
    var n32 = syms + ng + 2 * ng + ts + ts + occ_cap + S_LEN + 2 * BPE_K + mcap + 2 * BPE_RB * BPE_K + 16
    var n64 = 3 * ng + 4
    var h_st = List[Int32](length=S_LEN, fill=Int32(0))
    comptime if BPE_LIVEBUF:
        var pool = BPE_TRAIN_POOL.get_or_create_ptr()
        pool[].begin(ctx, n8, n32, n64, True)
        _train_passes(pool[], ctx, groups, vocab_size, min_frequency, ts, occ_cap, h_st, out.pairs)
    else:
        var local = BpeMem()
        local.begin(ctx, n8, n32, n64, False)
        _train_passes(local, ctx, groups, vocab_size, min_frequency, ts, occ_cap, h_st, out.pairs)
    out.n_merges = Int(h_st[S_NM])
    out.n_ties = Int(h_st[S_TIES])
    return out^


# ------------------------------------------------------------------ host side: encode


struct BpeDeviceTable(Defaultable, Movable):
    """The rank table's device copy, kept for the tokenizer's life under LIVEBUF (uploaded by the
    first encode); unused otherwise."""

    var arena: List[DeviceBuffer[DType.uint8]]
    var idx: List[DeviceBuffer[DType.int64]]

    def __init__(out self):
        self.arena = List[DeviceBuffer[DType.uint8]]()
        self.idx = List[DeviceBuffer[DType.int64]]()

    def ready(self) -> Bool:
        return len(self.arena) == 1 and len(self.idx) == 3


def _encode_launch(
    mut mem: BpeMem, ctx: DeviceContext, ranks: RankTable, mut table: BpeDeviceTable, pb: List[Int32],
    dp: List[Int32], text_addr: Int, ndocs: Int, nbytes: Int, ids_addr: Int, counts_addr: Int,
    mut h_st: List[Int32],
) raises:
    """Upload, launch the merge, scan and pack kernels, download ids, counts and the state words,
    wait once."""
    var npre = len(pb) - 1
    var nblk = _blocks(npre)
    var ntok = len(ranks.offset)
    var nbk = len(ranks.buckets)
    var narena = len(ranks.arena)
    var s_text = mem.a8(ctx, nbytes)
    var s_pb = mem.a32(ctx, npre + 1)
    var s_dp = mem.a32(ctx, ndocs + 1)
    var s_bnd = mem.a32(ctx, nbytes + npre + 1)
    var s_out = mem.a32(ctx, nbytes)
    var s_cnt = mem.a32(ctx, npre)
    var s_pfx = mem.a32(ctx, npre + 1)
    var s_bsum = mem.a32(ctx, nblk)
    var s_packed = mem.a32(ctx, nbytes)
    var s_counts = mem.a64(ctx, ndocs)
    var s_st = mem.a32(ctx, S_LEN)
    mem.up8(ctx, s_text, text_addr, nbytes)
    mem.up32(ctx, s_pb, Int(pb.unsafe_ptr()), npre + 1)
    mem.up32(ctx, s_dp, Int(dp.unsafe_ptr()), ndocs + 1)
    _fill(ctx, mem.p32(s_st), S_LEN, Int32(0))
    var t_arena: U8P
    var t_off: I64P
    var t_len: I64P
    var t_bk: I64P
    comptime if BPE_LIVEBUF:
        if not table.ready():
            table.arena.clear()
            table.idx.clear()
            table.arena.append(ctx.enqueue_create_buffer[DType.uint8](max(narena, 1)))
            table.idx.append(ctx.enqueue_create_buffer[DType.int64](max(ntok, 1)))
            table.idx.append(ctx.enqueue_create_buffer[DType.int64](max(ntok, 1)))
            table.idx.append(ctx.enqueue_create_buffer[DType.int64](max(nbk, 1)))
            if narena > 0:
                ctx.enqueue_copy(dst_buf=table.arena[0], src_ptr=U8P(unsafe_from_address=Int(ranks.arena.unsafe_ptr())))
            if ntok > 0:
                ctx.enqueue_copy(dst_buf=table.idx[0], src_ptr=I64P(unsafe_from_address=Int(ranks.offset.unsafe_ptr())))
                ctx.enqueue_copy(dst_buf=table.idx[1], src_ptr=I64P(unsafe_from_address=Int(ranks.length.unsafe_ptr())))
            if nbk > 0:
                ctx.enqueue_copy(dst_buf=table.idx[2], src_ptr=I64P(unsafe_from_address=Int(ranks.buckets.unsafe_ptr())))
        t_arena = table.arena[0].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        t_off = table.idx[0].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        t_len = table.idx[1].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        t_bk = table.idx[2].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    else:
        var s_ta = mem.a8(ctx, narena)
        var s_to = mem.a64(ctx, ntok)
        var s_tl = mem.a64(ctx, ntok)
        var s_tb = mem.a64(ctx, nbk)
        mem.up8(ctx, s_ta, Int(ranks.arena.unsafe_ptr()), narena)
        mem.up64(ctx, s_to, Int(ranks.offset.unsafe_ptr()), ntok)
        mem.up64(ctx, s_tl, Int(ranks.length.unsafe_ptr()), ntok)
        mem.up64(ctx, s_tb, Int(ranks.buckets.unsafe_ptr()), nbk)
        t_arena = mem.p8(s_ta)
        t_off = mem.p64(s_to)
        t_len = mem.p64(s_tl)
        t_bk = mem.p64(s_tb)
    if npre > 0:
        ctx.enqueue_function[bpe_encode_pre_kernel](
            mem.p8(s_text), mem.p32(s_pb), t_arena, t_off, t_len, t_bk, mem.p32(s_bnd), mem.p32(s_out),
            mem.p32(s_cnt), mem.p32(s_st), Int64(npre), Int64(ranks.mask),
            grid_dim=_blocks(npre), block_dim=BPE_TPB,
        )
    ctx.enqueue_function[bpe_scan_block_kernel](
        mem.p32(s_cnt), mem.p32(s_pfx), mem.p32(s_bsum), Int64(npre), grid_dim=nblk, block_dim=BPE_TPB,
    )
    ctx.enqueue_function[bpe_scan_sums_kernel](  # small-launch(nblk: per-block totals, pre-tokens / BPE_TPB): one threadgroup scans them in strided chunks
        mem.p32(s_bsum), mem.p32(s_st), Int64(nblk), grid_dim=1, block_dim=BPE_RT,
    )
    ctx.enqueue_function[bpe_scan_add_kernel](
        mem.p32(s_pfx), mem.p32(s_bsum), mem.p32(s_st), Int64(npre), grid_dim=_blocks(npre + 1), block_dim=BPE_TPB,
    )
    if npre > 0:
        ctx.enqueue_function[bpe_encode_pack_kernel](
            mem.p32(s_pb), mem.p32(s_cnt), mem.p32(s_pfx), mem.p32(s_out), mem.p32(s_packed), Int64(npre),
            grid_dim=_blocks(npre), block_dim=BPE_TPB,
        )
    ctx.enqueue_function[bpe_encode_counts_kernel](
        mem.p32(s_dp), mem.p32(s_pfx), mem.p64(s_counts), Int64(ndocs), grid_dim=_blocks(ndocs), block_dim=BPE_TPB,
    )
    # The packed ids are at most one per byte; the caller's output holds n_bytes ids.
    mem.down32(ctx, s_packed, ids_addr, nbytes)
    mem.down64(ctx, s_counts, counts_addr, ndocs)
    mem.down32(ctx, s_st, Int(h_st.unsafe_ptr()), S_LEN)
    ctx.synchronize()
    _ = len(pb) + len(dp) + len(ranks.arena)
    if h_st[S_FAIL] != Int32(0):
        raise Error("bpe_encode_batch_device: a piece has no rank; the table is missing a single-byte token")


def bpe_encode_batch_device(
    ranks: RankTable, mut table: BpeDeviceTable, pb: List[Int32], dp: List[Int32], text_addr: Int,
    ndocs: Int, nbytes: Int, ids_addr: Int, counts_addr: Int,
) raises -> Int:
    """encode_batch without `<|endoftext|>` recognition, every pre-token merged on the device.
    `pb` holds the npre + 1 pre-token byte bounds (absolute, document after document) and `dp`
    each document's first pre-token (ndocs + 1 entries). Writes every document's ids back to back
    as int32 at `ids_addr` (room for `nbytes`) and each document's count as int64 at
    `counts_addr`; returns the total, computed on the device."""
    var npre = len(pb) - 1
    var nblk = _blocks(npre)
    var ctx = bpe_ctx()
    var ntok = len(ranks.offset)
    var nbk = len(ranks.buckets)
    var narena = len(ranks.arena)
    var n8 = nbytes + narena + 2
    var n32 = 4 * (npre + 1) + (ndocs + 1) + 3 * (nbytes + 1) + nblk + S_LEN + 16
    var n64 = (ndocs + 1) + 2 * ntok + nbk + 8
    var h_st = List[Int32](length=S_LEN, fill=Int32(0))
    comptime if BPE_LIVEBUF:
        var pool = BPE_ENCODE_POOL.get_or_create_ptr()
        pool[].begin(ctx, n8, n32, n64, True)
        _encode_launch(pool[], ctx, ranks, table, pb, dp, text_addr, ndocs, nbytes, ids_addr, counts_addr, h_st)
    else:
        var local = BpeMem()
        local.begin(ctx, n8, n32, n64, False)
        _encode_launch(local, ctx, ranks, table, pb, dp, text_addr, ndocs, nbytes, ids_addr, counts_addr, h_st)
    return Int(h_st[S_TOTAL])

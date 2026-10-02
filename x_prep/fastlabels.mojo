# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The labels' run scan by flag-and-scan threadgroups (lane/apple-fast-prep3,
2026-10-02). FAST on Apple with -D MOJOLEARN_PREP3_LABELS only: x_prep/device.mojo
gates every launch on PREP3_LABELS, so no other build compiles one.

LabelEncoder, LabelBinarizer and MultiLabelBinarizer take their classes from
`_label_unique` (python/mojolearn/_expansion_prep.py): the sorted labels, then
`uniq_count` (one thread per chunk of ~sqrt(n) rows walking them), `uniq_scan`
(ONE thread over the chunk counts) and `uniq_write` (one thread per chunk
again). Here every chunk is a threadgroup: each thread flags the rows it
holds (a run opens where the `key` differs from the row before), a shared
memory tree gives the chunk's count, a shared memory prefix scan gives each
run start its slot, and the chunk counts are scanned by one threadgroup in
rounds of TGL instead of one thread. Counts are integers and the slots are
ranks in ascending row order, so the words are the serial units' word for
word: CNT, OFF, TOT and U unchanged. `chunk_neg` (LabelEncoder.transform's
unseen-label count) takes the same tree.
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from x_prep.common import FP, IP, p, ld, raw, st, ldi, sti, key

#: threads per chunk threadgroup
comptime TGL = 256


@always_inline
def _starts_run(f: FP, S: Int, i: Int) -> Bool:
    """Row i of the sorted column S opens a run (x_prep/labels.mojo's test)."""
    return i == 0 or key(raw(f, S + i)) != key(raw(f, S + i - 1))


def uniq_count_fast_kernel(f: FP, q: IP):
    """`uniq_count_unit` for chunk block_idx.x: q = [S, n, CH, CNT]. Each
    thread counts the run starts among rows lo + tid, lo + tid + TGL, ...
    of the chunk, then the tree."""
    var S = p(q, 0)
    var n = p(q, 1)
    var ch = p(q, 2)
    var t = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[TGL, Int32, address_space = AddressSpace.SHARED]()
    var lo = t * ch
    var hi = min(lo + ch, n)
    var k = 0
    var i = lo + tid
    while i < hi:
        if _starts_run(f, S, i):
            k += 1
        i += TGL
    sh[tid] = Int32(k)
    barrier()
    var w = TGL // 2
    while w >= 1:
        if tid < w:
            sh[tid] = sh[tid] + sh[tid + w]
        barrier()
        w //= 2
    if tid == 0:
        sti(f, p(q, 3) + t, Int(sh[0]))


def uniq_scan_fast_kernel(f: FP, q: IP):
    """`uniq_scan_unit` on one threadgroup: q = [CNT, nch, OFF, TOT]. The
    chunk counts in rounds of TGL, each round an exclusive prefix scan in
    shared memory carried from the rounds before; OFF int32 bits, TOT the
    total as a float (`unique_cols`' count)."""
    var C = p(q, 0)
    var nch = p(q, 1)
    var O = p(q, 2)
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[TGL, Int32, address_space = AddressSpace.SHARED]()
    var carry = 0
    var base = 0
    while base < nch:
        var idx = base + tid
        var v = 0
        if idx < nch:
            v = ldi(f, C + idx)
        sh[tid] = Int32(v)
        barrier()
        var off = 1
        while off < TGL:
            var add = Int32(0)
            if tid >= off:
                add = sh[tid - off]
            barrier()
            sh[tid] = sh[tid] + add
            barrier()
            off *= 2
        var incl = Int(sh[tid])
        var total = Int(sh[TGL - 1])
        if idx < nch:
            sti(f, O + idx, carry + incl - v)
        carry += total
        barrier()
        base += TGL
    if tid == 0:
        st(f, p(q, 3), Float32(carry))


def uniq_write_fast_kernel(f: FP, q: IP):
    """`uniq_write_unit` for chunk block_idx.x: q = [S, n, CH, OFF, U]. The
    chunk's rows in rounds of TGL, ascending: a prefix scan of the run-start
    flags gives each run start its slot after OFF[t] and the rounds before,
    and its word goes there bit for bit (the serial unit's order)."""
    var S = p(q, 0)
    var n = p(q, 1)
    var ch = p(q, 2)
    var U = p(q, 4)
    var t = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[TGL, Int32, address_space = AddressSpace.SHARED]()
    var lo = t * ch
    var hi = min(lo + ch, n)
    var k = ldi(f, p(q, 3) + t)
    var i0 = lo
    while i0 < hi:
        var i = i0 + tid
        var flag = 0
        if i < hi and _starts_run(f, S, i):
            flag = 1
        sh[tid] = Int32(flag)
        barrier()
        var off = 1
        while off < TGL:
            var add = Int32(0)
            if tid >= off:
                add = sh[tid - off]
            barrier()
            sh[tid] = sh[tid] + add
            barrier()
            off *= 2
        var incl = Int(sh[tid])
        var total = Int(sh[TGL - 1])
        if flag != 0:
            f.unsafe_store(U + k + incl - 1, raw(f, S + i))
        k += total
        barrier()
        i0 += TGL


def chunk_neg_fast_kernel(f: FP, q: IP):
    """`chunk_neg_unit` for chunk block_idx.x: q = [CODES, n, CH, OUT]: how
    many float codes of the chunk are negative, int32 bits, by the tree."""
    var C = p(q, 0)
    var n = p(q, 1)
    var ch = p(q, 2)
    var t = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var sh = stack_allocation[TGL, Int32, address_space = AddressSpace.SHARED]()
    var lo = t * ch
    var hi = min(lo + ch, n)
    var k = 0
    var i = lo + tid
    while i < hi:
        if ld(f, C + i) < Float32(0):
            k += 1
        i += TGL
    sh[tid] = Int32(k)
    barrier()
    var w = TGL // 2
    while w >= 1:
        if tid < w:
            sh[tid] = sh[tid] + sh[tid + w]
        barrier()
        w //= 2
    if tid == 0:
        sti(f, p(q, 3) + t, Int(sh[0]))

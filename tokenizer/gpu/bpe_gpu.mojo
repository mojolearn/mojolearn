# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""BPE ENCODE AND DECODE ON THE GPU, IDENTICAL MODE, EVERY VENDOR (lane vdf-parallel, 2026-11-24;
addenda 2 and 3 of the vd-functions briefs).

The `device="gpu"` side of `mojolearn.tokenizer.BpeTokenizer`: the same token ids, id for id, as
the host column (`tokenizer/encoding.mojo` `BpeTokenizer.encode_bytes` / `decode_bytes`), on CUDA,
HIP and Metal. Everything here is integer work (byte compares, an FNV-1a hash, rank compares,
integer prefix sums), so the ids do not depend on the order threads run in: there is no fold whose
order could move a bit, and the host column is unchanged.

THE PRE-TOKENIZER IS PARALLEL PER BYTE. The host scans the GPT-2 pattern left to right
(`tokenizer/impl/pretokenize.mojo` `pretoken_end`, leftmost-first over seven alternatives). With
possessive runs that scan has a LOCAL characterization, which three per-byte kernels compute:

  units  (`gpu_unit_kernel`): every byte that is not a UTF-8 continuation byte starts a unit; a
         continuation byte belongs to the unit of the nearest earlier non-continuation byte (at most
         3 back) when that byte decodes (the host `decode_codepoint`, bounded by the segment end)
         to a sequence that reaches it, else it is a one-byte invalid unit. Each unit gets its class:
         letter L, number N, other O (`[^\\s\\p{L}\\p{N}]`), space S, invalid X.
  starts (`gpu_start_kernel`): a unit p (previous unit q in the same segment) starts a pre-token iff
     * p is the first unit of its segment;
     * p is X; or p is L/N/O and q is X, or q is S other than U+0020 (a U+0020 just before an
       L/N/O unit is the optional leading space of its run: ` ?\\p{L}++` and the rest), or q has
       another class (except a letter right after an apostrophe that starts a contraction
       `'(?:[sdmt]|ll|ve|re)` covering it), or q is a letter that ends a contraction;
     * p is S and q is not S (a whitespace run starts), or p is the last unit of a run of two or
       more that does not reach the end of the segment (`\\s+(?!\\S)` leaves the last codepoint
       for the next pre-token: alternative 7, or the leading space of the next run).
     An apostrophe starts a contraction only where a pre-token starts at it: its previous unit is
     not O and not U+0020, or it is the segment's first unit.
  pieces (`gpu_merge_kernel`): a start's pre-token runs to the next start or the document end; one
         thread per pre-token runs the host `bpe_append` loop (merge the lowest-rank adjacent pair
         until none has a rank) over the device rank table (the host `RankTable` hash and probe).

SEGMENTS. A document is encoded alone (`encode_batch`); with `allow_endoftext=True` every
`<|endoftext|>` inside a document is its own piece (the eot id) and ends the text before it, as the
host splits and encodes the segments independently (`\\s++$` means end of SEGMENT). The literal
cannot overlap itself, so every byte offset where it matches is an occurrence (the host's leftmost
search finds the same set); a match never crosses a document end.

PACKING. Per-byte id counts, an integer exclusive scan (the scan kernels of
`tokenizer/fast/bpe_device.mojo`, shared, not copied), then one thread per pre-token copies its ids
to its place: every document's ids back to back, each document's count, the total. The host knows
only the byte and document counts, so nothing is read back before the result.

DECODE is a gather: per id its byte length (the eot literal for the eot id; an id outside
[0, n_vocab) is recorded, the FIRST such position by an atomic min, and refused after the call in
the host's words), a scan, a per-id copy. The ids may already be on the device (consumed in place).

TRANSFERS (addendum 3): the rank table, the Unicode class tables and the eot literal are uploaded
once per tokenizer (`BpeGpuTable`); host text is uploaded once per call by view; ids that are already
device-resident are read in place; the result (ids or bytes, counts, the state words) is read back
once, after the last kernel. No host step inside the call touches the data.
"""
from std.atomic import Atomic
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from core.neural_context import process_ctx
from tokenizer.encoding import BpeTokenizer
from tokenizer.impl.ranks import RankTable
from tokenizer.impl.unicode_class import UnicodeClasses
from tokenizer.fast.bpe_device import (
    BPE_TPB, BPE_RT, S_FAIL, S_TOTAL, S_LEN, FNV_OFFSET64, FNV_PRIME64,
    bpe_scan_block_kernel, bpe_scan_sums_kernel, bpe_scan_add_kernel,
)

comptime I32P = MutPointer[Int32, MutAnyOrigin]
comptime I64P = MutPointer[Int64, MutAnyOrigin]
comptime U8P = MutPointer[UInt8, MutAnyOrigin]

# Unit classes (the pre-tokenizer's), stored one byte per input byte.
comptime CL_L = 0
comptime CL_N = 1
comptime CL_O = 2
comptime CL_S = 3
comptime CL_X = 4
comptime CL_SPAN = 5  # a byte of an `<|endoftext|>` occurrence (allow_endoftext=True)

#: decode's state word: the first id position outside [0, n_vocab) (the scan kernels own S_TOTAL,
#: the merge kernel S_FAIL; slot 6 is free in this path's state words)
comptime G_BAD = 6
comptime G_NONE = Int32(2147483647)


# ------------------------------------------------------------------ device helpers


@always_inline
def _gtid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _blocks(count: Int) -> Int:
    return (count + BPE_TPB - 1) // BPE_TPB if count > 0 else 1


@always_inline
def _doc_of(offs: I64P, ndocs: Int, b: Int) -> Int:
    """The document holding byte `b`: the largest d < ndocs with offs[d] <= b (empty documents
    share their offset with the next one and are skipped by `<=`)."""
    var lo = 0
    var hi = ndocs - 1
    while lo < hi:
        var mid = (lo + hi + 1) // 2
        if Int(offs[mid]) <= b:
            lo = mid
        else:
            hi = mid - 1
    return lo


@always_inline
def _in_span(mflag: U8P, allow: Bool, lo: Int, b: Int, elen: Int) -> Bool:
    """Byte `b` lies inside an `<|endoftext|>` occurrence of its document."""
    if not allow:
        return False
    var j = max(lo, b - elen + 1)
    while j <= b:
        if mflag[j] != 0:
            return True
        j += 1
    return False


@always_inline
def _limit(mflag: U8P, allow: Bool, hi: Int, b: Int) -> Int:
    """The end of byte `b`'s segment as seen by a decode or a contraction at `b` (which reads at
    most 3 bytes on): the document end, or an occurrence starting within 3 bytes after `b`."""
    if allow:
        for j in range(1, 4):
            if b + j < hi and mflag[b + j] != 0:
                return b + j
    return hi


@always_inline
def _cont(c: Int) -> Bool:
    return c >= 0x80 and c <= 0xBF


@always_inline
def _decode_cp(text: U8P, i: Int, n: Int) -> Int:
    """`decode_codepoint`'s codepoint at `i` in a segment ending at `n` (-1 for invalid)."""
    var c = Int(text[i])
    if c < 0x80:
        return c
    if c >= 0xC2 and c <= 0xDF and i + 1 < n:
        var b1 = Int(text[i + 1])
        if b1 >= 0x80 and b1 <= 0xBF:
            return ((c & 0x1F) << 6) | (b1 & 0x3F)
        return -1
    if c >= 0xE0 and c <= 0xEF and i + 2 < n:
        var b1 = Int(text[i + 1])
        var b2 = Int(text[i + 2])
        var lo = 0xA0 if c == 0xE0 else 0x80
        var hi = 0x9F if c == 0xED else 0xBF
        if b1 >= lo and b1 <= hi and b2 >= 0x80 and b2 <= 0xBF:
            return ((c & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F)
        return -1
    if c >= 0xF0 and c <= 0xF4 and i + 3 < n:
        var b1 = Int(text[i + 1])
        var b2 = Int(text[i + 2])
        var b3 = Int(text[i + 3])
        var lo = 0x90 if c == 0xF0 else 0x80
        var hi = 0x8F if c == 0xF4 else 0xBF
        if b1 >= lo and b1 <= hi and b2 >= 0x80 and b2 <= 0xBF and b3 >= 0x80 and b3 <= 0xBF:
            return ((c & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F)
        return -1
    return -1


@always_inline
def _width(cp: Int) -> Int:
    """The byte width of a valid codepoint's UTF-8 form (1 for invalid: a one-byte unit)."""
    if cp < 0 or cp < 0x80:
        return 1
    if cp < 0x800:
        return 2
    if cp < 0x10000:
        return 3
    return 4


@always_inline
def _in_ranges(tab: I64P, nranges: Int, cp: Int) -> Bool:
    """`UnicodeClasses._in`: binary search over `[lo0, hi0, lo1, hi1, ...]`."""
    var lo = 0
    var hi = nranges - 1
    while lo <= hi:
        var mid = (lo + hi) // 2
        if cp < Int(tab[2 * mid]):
            hi = mid - 1
        elif cp > Int(tab[2 * mid + 1]):
            lo = mid + 1
        else:
            return True
    return False


@always_inline
def _class(cp: Int, letters: I64P, nl: Int, numbers: I64P, nn: Int, spaces: I64P, ns: Int) -> Int:
    """The pre-tokenizer's class of a codepoint: `_in_class`'s ASCII map, then the pinned tables,
    tried in `pretoken_end`'s order (letter, number, space, else other)."""
    if cp < 0:
        return CL_X
    if cp < 0x80:
        if (cp >= 0x41 and cp <= 0x5A) or (cp >= 0x61 and cp <= 0x7A):
            return CL_L
        if cp >= 0x30 and cp <= 0x39:
            return CL_N
        if cp == 0x20 or (cp >= 0x09 and cp <= 0x0D):
            return CL_S
        return CL_O
    if _in_ranges(letters, nl, cp):
        return CL_L
    if _in_ranges(numbers, nn, cp):
        return CL_N
    if _in_ranges(spaces, ns, cp):
        return CL_S
    return CL_O


@always_inline
def _prev_unit(ustart: U8P, ucls: U8P, lo: Int, b: Int) -> Int:
    """The start of the unit before byte `b` in its segment, or -1 at the segment's start (the
    document start, or the byte after an `<|endoftext|>` occurrence)."""
    if b <= lo:
        return -1
    if ucls[b - 1] == UInt8(CL_SPAN):
        return -1
    var j = b - 1
    while j >= max(lo, b - 4):
        if ustart[j] != 0:
            return j
        j -= 1
    return -1


@always_inline
def _apostrophe_starts(text: U8P, ustart: U8P, ucls: U8P, lo: Int, a: Int) -> Bool:
    """A pre-token starts AT the apostrophe unit `a` (so its contraction alternative is tried):
    its previous unit is not O (an O run would hold it) and not U+0020 (the space would lead an O
    run holding it)."""
    var q = _prev_unit(ustart, ucls, lo, a)
    if q < 0:
        return True
    var kq = Int(ucls[q])
    if kq == CL_O:
        return False
    if kq == CL_S:
        return Int(text[q]) != 0x20
    return True


@always_inline
def _contraction_end(text: U8P, ustart: U8P, ucls: U8P, mflag: U8P, allow: Bool, lo: Int, hi: Int, a: Int) -> Int:
    """`_contraction_end` at the apostrophe `a` where a pre-token starts at it, else -1."""
    if a < lo or Int(text[a]) != 0x27 or ucls[a] == UInt8(CL_SPAN):
        return -1
    if not _apostrophe_starts(text, ustart, ucls, lo, a):
        return -1
    var n = _limit(mflag, allow, hi, a)
    if a + 1 >= n:
        return -1
    var c1 = Int(text[a + 1])
    if c1 == 0x73 or c1 == 0x64 or c1 == 0x6D or c1 == 0x74:  # s d m t
        return a + 2
    if a + 2 >= n:
        return -1
    var c2 = Int(text[a + 2])
    if (c1 == 0x6C and c2 == 0x6C) or (c1 == 0x76 and c2 == 0x65) or (c1 == 0x72 and c2 == 0x65):  # ll ve re
        return a + 3
    return -1


@always_inline
def _rank(text: U8P, at: Int, cnt: Int, arena: U8P, toff: I64P, tlen: I64P, buckets: I64P, tmask: Int) -> Int:
    """`RankTable.rank` on the device: the same FNV-1a hash, the same linear probe."""
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


# ------------------------------------------------------------------ encode kernels


def gpu_eot_match_kernel(text: U8P, offs: I64P, ndocs: Int64, n: Int64, eot: U8P, elen: Int64, mflag: U8P):
    """One thread per byte: an `<|endoftext|>` occurrence starts here, inside one document."""
    var b = _gtid()
    if b >= Int(n):
        return
    var d = _doc_of(offs, Int(ndocs), b)
    var hi = Int(offs[d + 1])
    var m = Int(elen)
    var hit = m > 0 and b + m <= hi
    if hit:
        for k in range(m):
            if text[b + k] != eot[k]:
                hit = False
                break
    mflag.unsafe_store(b, UInt8(1) if hit else UInt8(0))


def gpu_unit_kernel(
    text: U8P, offs: I64P, ndocs: Int64, n: Int64, mflag: U8P, elen: Int64, allow: Int64,
    letters: I64P, nl: Int64, numbers: I64P, nn: Int64, spaces: I64P, ns: Int64,
    ustart: U8P, ulen: U8P, ucls: U8P,
):
    """One thread per byte: does a unit (a codepoint, or one invalid byte) start here, its byte
    width and its class (see the module docstring)."""
    var b = _gtid()
    if b >= Int(n):
        return
    var on = allow != 0
    var d = _doc_of(offs, Int(ndocs), b)
    var lo = Int(offs[d])
    var hi = Int(offs[d + 1])
    if _in_span(mflag, on, lo, b, Int(elen)):
        ustart.unsafe_store(b, UInt8(0))
        ulen.unsafe_store(b, UInt8(1))
        ucls.unsafe_store(b, UInt8(CL_SPAN))
        return
    var c = Int(text[b])
    if _cont(c):
        # The nearest earlier non-continuation byte of this segment, at most 3 back.
        var j = b - 1
        var lead = -1
        while j >= max(lo, b - 3):
            if _in_span(mflag, on, lo, j, Int(elen)):
                break
            if not _cont(Int(text[j])):
                lead = j
                break
            j -= 1
        if lead >= 0:
            var cp = _decode_cp(text, lead, _limit(mflag, on, hi, lead))
            if cp >= 0 and lead + _width(cp) > b:
                ustart.unsafe_store(b, UInt8(0))
                ulen.unsafe_store(b, UInt8(1))
                ucls.unsafe_store(b, UInt8(CL_X))
                return
        ustart.unsafe_store(b, UInt8(1))
        ulen.unsafe_store(b, UInt8(1))
        ucls.unsafe_store(b, UInt8(CL_X))
        return
    var cp = _decode_cp(text, b, _limit(mflag, on, hi, b))
    ustart.unsafe_store(b, UInt8(1))
    ulen.unsafe_store(b, UInt8(_width(cp)))
    ucls.unsafe_store(b, UInt8(_class(cp, letters, Int(nl), numbers, Int(nn), spaces, Int(ns))))


def gpu_start_kernel(
    text: U8P, offs: I64P, ndocs: Int64, n: Int64, mflag: U8P, allow: Int64,
    ustart: U8P, ulen: U8P, ucls: U8P, tflag: U8P,
):
    """One thread per byte: 1 where a pre-token starts, 2 where an `<|endoftext|>` occurrence
    starts (its piece is the eot id), 0 elsewhere (see the module docstring for the rule)."""
    var b = _gtid()
    if b >= Int(n):
        return
    var on = allow != 0
    var k = Int(ucls[b])
    if k == CL_SPAN:
        tflag.unsafe_store(b, UInt8(2) if mflag[b] != 0 else UInt8(0))
        return
    if ustart[b] == 0:
        tflag.unsafe_store(b, UInt8(0))
        return
    var d = _doc_of(offs, Int(ndocs), b)
    var lo = Int(offs[d])
    var hi = Int(offs[d + 1])
    var q = _prev_unit(ustart, ucls, lo, b)
    var start = False
    if k == CL_X:
        start = True
    elif k == CL_S:
        if q < 0 or Int(ucls[q]) != CL_S:
            start = True
        else:
            var nb = b + Int(ulen[b])
            var has_next = nb < hi and not (on and mflag[nb] != 0)
            start = has_next and Int(ucls[nb]) != CL_S
    elif q < 0:
        start = True
    else:
        var kq = Int(ucls[q])
        if kq == CL_X:
            start = True
        elif kq == CL_S:
            start = Int(text[q]) != 0x20
        elif kq != k:
            # A letter right after an apostrophe whose contraction covers it continues that
            # contraction; any other change of class starts a pre-token.
            start = not (k == CL_L and kq == CL_O and q == b - 1
                and _contraction_end(text, ustart, ucls, mflag, on, lo, hi, q) > b)
        elif k == CL_L:
            # Same class: a run continues, unless the previous letter ends a contraction.
            start = (_contraction_end(text, ustart, ucls, mflag, on, lo, hi, q - 1) == b
                or _contraction_end(text, ustart, ucls, mflag, on, lo, hi, q - 2) == b)
    tflag.unsafe_store(b, UInt8(1) if start else UInt8(0))


def gpu_merge_kernel(
    text: U8P, offs: I64P, ndocs: Int64, n: Int64, tflag: U8P, arena: U8P, toff: I64P, tlen: I64P,
    buckets: I64P, tmask: Int64, eot_id: Int64, bnd: I32P, outi: I32P, cnt: I32P, st: I32P,
):
    """One thread per byte; a thread whose byte starts a piece runs `bpe_append` over it (the
    lowest-rank adjacent pair merged until none has a rank) and writes its ids at outi[b ...] and
    their count at cnt[b]; every other byte writes count 0. An `<|endoftext|>` piece is the eot id.
    The piece's boundary list lives at bnd[2b ...] (pieces are disjoint and at least one byte
    long, so the ranges [2b, 2b + len + 1) are disjoint)."""
    var b = _gtid()
    if b >= Int(n):
        return
    var t = Int(tflag[b])
    if t == 0:
        cnt.unsafe_store(b, Int32(0))
        return
    if t == 2:
        outi.unsafe_store(b, Int32(eot_id))
        cnt.unsafe_store(b, Int32(1))
        return
    var d = _doc_of(offs, Int(ndocs), b)
    var hi = Int(offs[d + 1])
    var e = b + 1
    while e < hi and tflag[e] == 0:
        e += 1
    var plen = e - b
    var tm = Int(tmask)
    var whole = _rank(text, b, plen, arena, toff, tlen, buckets, tm)
    if whole >= 0:
        outi.unsafe_store(b, Int32(whole))
        cnt.unsafe_store(b, Int32(1))
        return
    var bo = 2 * b
    for i in range(plen + 1):
        bnd.unsafe_store(bo + i, Int32(b + i))
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
        outi.unsafe_store(b + k, Int32(id))
    cnt.unsafe_store(b, Int32(nb - 1))


def gpu_pack_kernel(cnt: I32P, pfx: I32P, outi: I32P, packed: I32P, n: Int64):
    """One thread per byte: a piece's ids to their place in the packed stream (documents in order,
    pieces in byte order)."""
    var b = _gtid()
    if b >= Int(n):
        return
    var c = Int(cnt[b])
    var at = Int(pfx[b])
    for j in range(c):
        packed.unsafe_store(at + j, outi[b + j])


def gpu_doc_counts_kernel(offs: I64P, pfx: I32P, counts: I64P, ndocs: Int64):
    """One thread per document: its id count, the scan's difference over its bytes."""
    var d = _gtid()
    if d >= Int(ndocs):
        return
    counts.unsafe_store(d, Int64(pfx[Int(offs[d + 1])] - pfx[Int(offs[d])]))


# ------------------------------------------------------------------ decode kernels


def gpu_decode_len_kernel(
    ids: I64P, nids: Int64, tlen: I64P, ntok: Int64, eot_id: Int64, elen: Int64, lens: I32P, st: I32P,
):
    """One thread per id: its byte length; an id outside [0, n_vocab) records its position (the
    first one wins, an atomic min) and contributes no bytes."""
    var i = _gtid()
    if i >= Int(nids):
        return
    var id = Int(ids[i])
    var m = 0
    if id == Int(eot_id):
        m = Int(elen)
    elif id >= 0 and id < Int(ntok):
        m = Int(tlen[id])
    else:
        _ = Atomic.min(st + G_BAD, Int32(i))
    lens.unsafe_store(i, Int32(m))


def gpu_decode_copy_kernel(
    ids: I64P, nids: Int64, arena: U8P, toff: I64P, tlen: I64P, ntok: Int64, eot_id: Int64, eot: U8P,
    elen: Int64, pfx: I32P, out_bytes: U8P,
):
    """One thread per id: its bytes (the eot literal for the eot id) at its scanned offset."""
    var i = _gtid()
    if i >= Int(nids):
        return
    var id = Int(ids[i])
    var at = Int(pfx[i])
    if id == Int(eot_id):
        for k in range(Int(elen)):
            out_bytes.unsafe_store(at + k, eot[k])
    elif id >= 0 and id < Int(ntok):
        var o = Int(toff[id])
        for k in range(Int(tlen[id])):
            out_bytes.unsafe_store(at + k, arena[o + k])


def gpu_state_init_kernel(st: I32P):
    """decode's first-bad-position word starts at G_NONE."""
    if _gtid() == 0:
        st.unsafe_store(G_BAD, G_NONE)


def gpu_seq_lens_kernel(soffs: I64P, pfx: I32P, lens: I64P, nseq: Int64):
    """One thread per sequence: its byte count, the scan's difference over its ids."""
    var s = _gtid()
    if s >= Int(nseq):
        return
    lens.unsafe_store(s, Int64(pfx[Int(soffs[s + 1])] - pfx[Int(soffs[s])]))


# ------------------------------------------------------------------ host side


def bpe_gpu_ctx() raises -> DeviceContext:
    """ONE process-lifetime DeviceContext for the identical GPU tokenizer."""
    return process_ctx["MojolearnTokenizerGpuContextV1"]()


struct BpeGpuTable(Movable):
    """A tokenizer's device-resident tables, uploaded ONCE (at construction with device='gpu'):
    the rank table (arena, offsets, lengths, hash buckets), the three Unicode class range tables
    and the `<|endoftext|>` literal."""

    var arena: DeviceBuffer[DType.uint8]
    var toff: DeviceBuffer[DType.int64]
    var tlen: DeviceBuffer[DType.int64]
    var buckets: DeviceBuffer[DType.int64]
    var letters: DeviceBuffer[DType.int64]
    var numbers: DeviceBuffer[DType.int64]
    var spaces: DeviceBuffer[DType.int64]
    var eot: DeviceBuffer[DType.uint8]
    var mask: Int
    var ntok: Int
    var nl: Int
    var nn: Int
    var ns: Int
    var elen: Int

    def __init__(out self, ctx: DeviceContext, tok: BpeTokenizer) raises:
        ref ranks = tok.ranks
        ref classes = tok.classes
        self.mask = ranks.mask
        self.ntok = len(ranks.offset)
        self.nl = len(classes.letters) // 2
        self.nn = len(classes.numbers) // 2
        self.ns = len(classes.spaces) // 2
        self.elen = len(tok.eot)
        self.arena = ctx.enqueue_create_buffer[DType.uint8](max(len(ranks.arena), 1))
        self.toff = ctx.enqueue_create_buffer[DType.int64](max(self.ntok, 1))
        self.tlen = ctx.enqueue_create_buffer[DType.int64](max(self.ntok, 1))
        self.buckets = ctx.enqueue_create_buffer[DType.int64](max(len(ranks.buckets), 1))
        self.letters = ctx.enqueue_create_buffer[DType.int64](max(len(classes.letters), 1))
        self.numbers = ctx.enqueue_create_buffer[DType.int64](max(len(classes.numbers), 1))
        self.spaces = ctx.enqueue_create_buffer[DType.int64](max(len(classes.spaces), 1))
        self.eot = ctx.enqueue_create_buffer[DType.uint8](max(self.elen, 1))
        _up8(ctx, self.arena, Int(ranks.arena.unsafe_ptr()), len(ranks.arena))
        _up64(ctx, self.toff, Int(ranks.offset.unsafe_ptr()), self.ntok)
        _up64(ctx, self.tlen, Int(ranks.length.unsafe_ptr()), self.ntok)
        _up64(ctx, self.buckets, Int(ranks.buckets.unsafe_ptr()), len(ranks.buckets))
        _up64(ctx, self.letters, Int(classes.letters.unsafe_ptr()), len(classes.letters))
        _up64(ctx, self.numbers, Int(classes.numbers.unsafe_ptr()), len(classes.numbers))
        _up64(ctx, self.spaces, Int(classes.spaces.unsafe_ptr()), len(classes.spaces))
        _up8(ctx, self.eot, Int(tok.eot.unsafe_ptr()), self.elen)
        # The host lists are read by the copies above until the context runs them.
        ctx.synchronize()


def _up8(ctx: DeviceContext, buf: DeviceBuffer[DType.uint8], src_addr: Int, count: Int) raises:
    if count > 0:
        ctx.enqueue_copy(dst_buf=buf.create_sub_buffer[DType.uint8](0, count), src_ptr=U8P(unsafe_from_address=src_addr))


def _up64(ctx: DeviceContext, buf: DeviceBuffer[DType.int64], src_addr: Int, count: Int) raises:
    if count > 0:
        ctx.enqueue_copy(dst_buf=buf.create_sub_buffer[DType.int64](0, count), src_ptr=I64P(unsafe_from_address=src_addr))


@always_inline
def _p8(buf: DeviceBuffer[DType.uint8]) -> U8P:
    return buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def _p32(buf: DeviceBuffer[DType.int32]) -> I32P:
    return buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def _p64(buf: DeviceBuffer[DType.int64]) -> I64P:
    return buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _scan(ctx: DeviceContext, cnt: I32P, pfx: I32P, st: I32P, count: Int) raises:
    """pfx[0 .. count] = the exclusive prefix of cnt[0 .. count) and its total (integer adds:
    the shared scan kernels of tokenizer/fast/bpe_device.mojo)."""
    var nblk = _blocks(count)
    var bsum = ctx.enqueue_create_buffer[DType.int32](nblk)
    ctx.enqueue_function[bpe_scan_block_kernel](cnt, pfx, _p32(bsum), Int64(count), grid_dim=nblk, block_dim=BPE_TPB)
    ctx.enqueue_function[bpe_scan_sums_kernel](  # small-launch(nblk: per-block totals, count / BPE_TPB): one threadgroup scans them in strided chunks
        _p32(bsum), st, Int64(nblk), grid_dim=1, block_dim=BPE_RT,
    )
    ctx.enqueue_function[bpe_scan_add_kernel](pfx, _p32(bsum), st, Int64(count), grid_dim=_blocks(count + 1), block_dim=BPE_TPB)
    # Named again after the last launch that reads it (Mojo frees a local at its last named use).
    ctx.synchronize()
    _ = bsum^


def bpe_gpu_encode_device(
    ctx: DeviceContext, table: BpeGpuTable, text: U8P, offs: I64P, ndocs: Int, n: Int, allow: Bool,
    eot_id: Int, packed: I32P, counts: I64P, st: I32P,
) raises:
    """encode_batch on device data: `text` (n bytes) and `offs` (ndocs + 1 document offsets) are
    device pointers; every document's ids are written back to back at `packed` (room for n ids),
    each document's count at `counts`, the total at st[S_TOTAL] and a missing-rank failure at
    st[S_FAIL] (st holds S_LEN words, zeroed by the caller). Every step runs on the device."""
    if n == 0:
        return
    var nblk = _blocks(n)
    var mflag = ctx.enqueue_create_buffer[DType.uint8](n)
    var ustart = ctx.enqueue_create_buffer[DType.uint8](n)
    var ulen = ctx.enqueue_create_buffer[DType.uint8](n)
    var ucls = ctx.enqueue_create_buffer[DType.uint8](n)
    var tflag = ctx.enqueue_create_buffer[DType.uint8](n)
    var bnd = ctx.enqueue_create_buffer[DType.int32](2 * n + 2)
    var outi = ctx.enqueue_create_buffer[DType.int32](n)
    var cnt = ctx.enqueue_create_buffer[DType.int32](n)
    var pfx = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var allow_word = Int64(1) if allow else Int64(0)
    if allow:
        ctx.enqueue_function[gpu_eot_match_kernel](
            text, offs, Int64(ndocs), Int64(n), _p8(table.eot), Int64(table.elen), _p8(mflag),
            grid_dim=nblk, block_dim=BPE_TPB,
        )
    else:
        mflag.enqueue_fill(UInt8(0))
    ctx.enqueue_function[gpu_unit_kernel](
        text, offs, Int64(ndocs), Int64(n), _p8(mflag), Int64(table.elen), allow_word,
        _p64(table.letters), Int64(table.nl), _p64(table.numbers), Int64(table.nn), _p64(table.spaces), Int64(table.ns),
        _p8(ustart), _p8(ulen), _p8(ucls), grid_dim=nblk, block_dim=BPE_TPB,
    )
    ctx.enqueue_function[gpu_start_kernel](
        text, offs, Int64(ndocs), Int64(n), _p8(mflag), allow_word, _p8(ustart), _p8(ulen), _p8(ucls), _p8(tflag),
        grid_dim=nblk, block_dim=BPE_TPB,
    )
    ctx.enqueue_function[gpu_merge_kernel](
        text, offs, Int64(ndocs), Int64(n), _p8(tflag), _p8(table.arena), _p64(table.toff), _p64(table.tlen),
        _p64(table.buckets), Int64(table.mask), Int64(eot_id), _p32(bnd), _p32(outi), _p32(cnt), st,
        grid_dim=nblk, block_dim=BPE_TPB,
    )
    _scan(ctx, _p32(cnt), _p32(pfx), st, n)
    ctx.enqueue_function[gpu_pack_kernel](_p32(cnt), _p32(pfx), _p32(outi), packed, Int64(n), grid_dim=nblk, block_dim=BPE_TPB)
    ctx.enqueue_function[gpu_doc_counts_kernel](offs, _p32(pfx), counts, Int64(ndocs), grid_dim=_blocks(ndocs), block_dim=BPE_TPB)
    ctx.synchronize()
    _ = mflag^
    _ = ustart^
    _ = ulen^
    _ = ucls^
    _ = tflag^
    _ = bnd^
    _ = outi^
    _ = cnt^
    _ = pfx^


def bpe_gpu_encode_batch_into(
    table: BpeGpuTable, text_addr: Int, n: Int, offs_host: List[Int64], allow: Bool, eot_id: Int,
    ids_addr: Int, counts_addr: Int,
) raises -> Int:
    """encode_batch on host input: the text and the document offsets are uploaded once, the work is
    `bpe_gpu_encode_device`, and the ids (int32, room for n at `ids_addr`), each document's count
    (int64 at `counts_addr`) and the state words come back once, after the last kernel. Returns the
    total; a missing rank is refused in the host's words."""
    var ctx = bpe_gpu_ctx()
    var ndocs = len(offs_host) - 1
    var d_text = ctx.enqueue_create_buffer[DType.uint8](max(n, 1))
    var d_offs = ctx.enqueue_create_buffer[DType.int64](ndocs + 1)
    var d_packed = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var d_counts = ctx.enqueue_create_buffer[DType.int64](max(ndocs, 1))
    var d_st = ctx.enqueue_create_buffer[DType.int32](S_LEN)
    _up8(ctx, d_text, text_addr, n)
    _up64(ctx, d_offs, Int(offs_host.unsafe_ptr()), ndocs + 1)
    d_st.enqueue_fill(Int32(0))
    d_counts.enqueue_fill(Int64(0))
    bpe_gpu_encode_device(ctx, table, _p8(d_text), _p64(d_offs), ndocs, n, allow, eot_id, _p32(d_packed),
                          _p64(d_counts), _p32(d_st))
    var h_st = List[Int32](length=S_LEN, fill=Int32(0))
    ctx.enqueue_copy(dst_ptr=I32P(unsafe_from_address=Int(h_st.unsafe_ptr())), src_buf=d_st)
    if n > 0:
        ctx.enqueue_copy(dst_ptr=I32P(unsafe_from_address=ids_addr), src_buf=d_packed.create_sub_buffer[DType.int32](0, n))
    if ndocs > 0:
        ctx.enqueue_copy(dst_ptr=I64P(unsafe_from_address=counts_addr), src_buf=d_counts.create_sub_buffer[DType.int64](0, ndocs))
    ctx.synchronize()
    _ = d_text^
    _ = d_offs^
    _ = d_packed^
    _ = d_counts^
    _ = d_st^
    _ = len(offs_host)
    if h_st[S_FAIL] != Int32(0):
        raise Error("bpe: a piece has no rank; the table is missing a single-byte token")
    return 0 if n == 0 else Int(h_st[S_TOTAL])


def bpe_gpu_decode_device(
    ctx: DeviceContext, table: BpeGpuTable, ids: I64P, nids: Int, soffs: I64P, nseq: Int, eot_id: Int,
    out_bytes: DeviceBuffer[DType.uint8], seq_lens: I64P, st: I32P,
) raises:
    """decode on device ids (int64, `nids` of them, read in place): every sequence's bytes back to
    back in `out_bytes` (room for nids * max token length), each sequence's byte count at
    `seq_lens` (`soffs` holds the nseq + 1 sequence offsets into ids), the total at st[S_TOTAL] and
    the first id outside [0, n_vocab) at st[G_BAD] (G_NONE when none; st is set by the caller)."""
    if nids == 0:
        return
    var nblk = _blocks(nids)
    var lens = ctx.enqueue_create_buffer[DType.int32](nids)
    var pfx = ctx.enqueue_create_buffer[DType.int32](nids + 1)
    ctx.enqueue_function[gpu_decode_len_kernel](
        ids, Int64(nids), _p64(table.tlen), Int64(table.ntok), Int64(eot_id), Int64(table.elen), _p32(lens), st,
        grid_dim=nblk, block_dim=BPE_TPB,
    )
    _scan(ctx, _p32(lens), _p32(pfx), st, nids)
    ctx.enqueue_function[gpu_decode_copy_kernel](
        ids, Int64(nids), _p8(table.arena), _p64(table.toff), _p64(table.tlen), Int64(table.ntok), Int64(eot_id),
        _p8(table.eot), Int64(table.elen), _p32(pfx), _p8(out_bytes), grid_dim=nblk, block_dim=BPE_TPB,
    )
    ctx.enqueue_function[gpu_seq_lens_kernel](soffs, _p32(pfx), seq_lens, Int64(nseq), grid_dim=_blocks(nseq), block_dim=BPE_TPB)
    ctx.synchronize()
    _ = lens^
    _ = pfx^


def bpe_gpu_decode_batch_into(
    table: BpeGpuTable, ids_device: Optional[I64P], ids_host_addr: Int, soffs_host: List[Int64],
    max_token_bytes: Int, eot_id: Int, out_addr: Int, cap: Int, lens_addr: Int,
) raises -> Int:
    """decode_batch: the ids (int64) are either already on the device (`ids_device` given: read in
    place, never downloaded) or uploaded once from the host address `ids_host_addr`; the sequence offsets
    (`soffs_host`, nseq + 1 of them) are uploaded once. After the last kernel the state words come
    back, then the bytes (into `out_addr`, room for `cap`) and each sequence's byte count (int64 at
    `lens_addr`). Returns the byte total, or -1 - p for the first id outside [0, n_vocab) at global
    position p (nothing is written then; the caller raises its words)."""
    var ctx = bpe_gpu_ctx()
    var nseq = len(soffs_host) - 1
    var nids = Int(soffs_host[nseq])
    var width = max(max(max_token_bytes, table.elen), 1)
    var d_ids = ctx.enqueue_create_buffer[DType.int64](max(nids, 1))
    var ids = _p64(d_ids)
    if ids_device:
        ids = ids_device.value()
    elif nids > 0:
        ctx.enqueue_copy(dst_buf=d_ids.create_sub_buffer[DType.int64](0, nids),
                         src_ptr=I64P(unsafe_from_address=ids_host_addr))
    var d_soffs = ctx.enqueue_create_buffer[DType.int64](nseq + 1)
    _up64(ctx, d_soffs, Int(soffs_host.unsafe_ptr()), nseq + 1)
    var d_out = ctx.enqueue_create_buffer[DType.uint8](max(nids * width, 1))
    var d_lens = ctx.enqueue_create_buffer[DType.int64](max(nseq, 1))
    var d_st = ctx.enqueue_create_buffer[DType.int32](S_LEN)
    d_st.enqueue_fill(Int32(0))
    d_lens.enqueue_fill(Int64(0))
    ctx.enqueue_function[gpu_state_init_kernel](_p32(d_st), grid_dim=1, block_dim=1)  # small-launch(one state word)
    bpe_gpu_decode_device(ctx, table, ids, nids, _p64(d_soffs), nseq, eot_id, d_out, _p64(d_lens), _p32(d_st))
    var h_st = List[Int32](length=S_LEN, fill=Int32(0))
    ctx.enqueue_copy(dst_ptr=I32P(unsafe_from_address=Int(h_st.unsafe_ptr())), src_buf=d_st)
    ctx.synchronize()
    var result: Int
    if nids > 0 and h_st[G_BAD] != G_NONE:
        result = -1 - Int(h_st[G_BAD])
    else:
        var total = 0 if nids == 0 else Int(h_st[S_TOTAL])
        if total > cap:
            raise Error("bpe_decode_batch: " + String(total) + " bytes do not fit " + String(cap))
        if total > 0:
            ctx.enqueue_copy(dst_ptr=U8P(unsafe_from_address=out_addr), src_buf=d_out.create_sub_buffer[DType.uint8](0, total))
        if nseq > 0:
            ctx.enqueue_copy(dst_ptr=I64P(unsafe_from_address=lens_addr), src_buf=d_lens.create_sub_buffer[DType.int64](0, nseq))
        ctx.synchronize()
        result = total
    _ = d_ids^
    _ = d_soffs^
    _ = d_out^
    _ = d_lens^
    _ = d_st^
    _ = len(soffs_host)
    return result

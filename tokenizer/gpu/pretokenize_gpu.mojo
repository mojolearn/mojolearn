# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE PRE-TOKENIZER CUT ON THE GPU, EVERY VENDOR (lane vdf2-public, 2026-11-24): the
`device="gpu"` side of `mojolearn.models.tokenizer.pretokenize(data, pattern)`, the same `m + 1`
boundaries as the host column (`tokenizer/impl/pretokenize.mojo` `pretokenize_pattern`), offset
for offset, for the three patterns.

Integer work only (byte compares, class tables, integer scans, index gathers): no fold whose
order could move a bit, so the host column is unchanged and every vendor gives the same offsets.

GPT-2 is `tokenizer/gpu/bpe_gpu.mojo`'s per-byte pre-tokenizer, reused, not copied
(`gpu_unit_kernel`, `gpu_start_kernel`, one document, no special token): a boundary is every
pre-token start and the end of the data.

LLAMA 3 AND QWEN 2 (`pretoken_end_llama`: contractions without case, an optional non-letter before
a letter run, digits one to three at a time (Qwen 2: one), punctuation runs with trailing CR/LF,
whitespace through its last CR/LF, `\\s+(?!\\S)`, `\\s+`) have no local start rule like GPT-2's
possessive one, so the scan's own step is computed for EVERY unit and the scan's chain is found by
pointer jumping:

  units  `gpu_unit_kernel` (bpe_gpu.mojo): unit starts, widths and classes; the host scan only
         ever stands on unit starts (it advances by a codepoint's width, by one byte at an invalid
         byte, or to a run end), so these are the only positions it can visit;
  runs   a run is a maximal span of units of one class (an invalid unit is a run of its own); per
         run its end, its last unit and its last CR/LF unit (`gpu_cut_run_*`), and per byte the end
         of the CR/LF byte run starting there, from two integer scans;
  steps  `gpu_cut_llama_next_kernel`: nxt[p] = `pretoken_end_llama` at every unit start p, in O(1)
         from those tables (nxt[n] = n);
  chain  the boundaries are 0, nxt[0], nxt[nxt[0]], ... up to n. With J_k = nxt^(2^k) and the
         marks started at {0}, the levels k = K .. 0 each mark J_k[i] for every marked i: after
         level k every chain position whose index is a multiple of 2^k is marked, so after level 0
         every one is (2^(K+1) > n >= the chain length). J of a chain position is a chain
         position, so a mark set while a level runs is a true one too: the race is benign. J_k is
         rebuilt from nxt by k squarings (O(n) memory, K(K+1)/2 gather passes, every one parallel);
  pack   the marked positions, in order, by an integer scan.

TRANSFERS: the data is uploaded once and the class tables once per call (`BpeGpuClasses`); the
boundaries and the state words come back once, after the last kernel. No host step touches the
data.
"""
from std.atomic import Atomic
from max.gpu.host import DeviceBuffer, DeviceContext

from tokenizer.impl.unicode_class import builtin_unicode_classes
from tokenizer.impl.pretokenize import PATTERN_GPT2, PATTERN_LLAMA3, PATTERN_QWEN2
from tokenizer.fast.bpe_device import BPE_TPB, S_LEN, S_TOTAL
from tokenizer.gpu.bpe_gpu import (
    CL_L, CL_N, CL_O, CL_S, CL_X, I32P, I64P, U8P, BpeGpuClasses, bpe_gpu_ctx, gpu_unit_kernel,
    gpu_start_kernel, _gtid, _blocks, _doc_of, _decode_cp, _width, _prev_unit, _up8, _up64, _p8, _p32, _p64, _scan,
)


# ------------------------------------------------------------------ kernels


def gpu_cut_runflag_kernel(ustart: U8P, ucls: U8P, offs: I64P, ndocs: Int64, n: Int64, flag: I32P):
    """One thread per byte: 1 where a run starts (a unit whose previous unit has another class, or
    an invalid unit, or the first unit of its segment), 0 elsewhere (inside a unit or a run). The
    segments are the `ndocs` pieces at `offs` (one for the plain cut; the text between added tokens
    for `Tokenizer.encode(allow_special=True)`): a run never crosses a segment's end."""
    var b = _gtid()
    if b >= Int(n):
        return
    if ustart[b] == 0:
        flag.unsafe_store(b, Int32(0))
        return
    var q = _prev_unit(ustart, ucls, Int(offs[_doc_of(offs, Int(ndocs), b)]), b)
    var k = Int(ucls[b])
    var start = q < 0 or k == CL_X or Int(ucls[q]) != k
    flag.unsafe_store(b, Int32(1) if start else Int32(0))


def gpu_cut_runpos_kernel(flag: I32P, pfx: I32P, n: Int64, runpos: I32P):
    """One thread per byte: run r's first byte at runpos[r]."""
    var b = _gtid()
    if b >= Int(n):
        return
    if flag[b] != 0:
        runpos.unsafe_store(Int(pfx[b]), Int32(b))


def gpu_cut_runinfo_kernel(
    text: U8P, ustart: U8P, ulen: U8P, flag: I32P, pfx: I32P, runpos: I32P, n: Int64,
    runend: I32P, lastu: I32P, lastcr: I32P,
):
    """One thread per byte: at a unit start u of run r, its run's end (runend[u]); the run's last
    unit (lastu[r], written by that unit); the run's last CR/LF unit (lastcr[r], an atomic max over
    its CR/LF units; -1 when it has none, set by the caller)."""
    var u = _gtid()
    var m = Int(n)
    if u >= m or ustart[u] == 0:
        return
    var nruns = Int(pfx[m])
    var r = Int(pfx[u + 1]) - 1
    runend.unsafe_store(u, runpos[r + 1] if r + 1 < nruns else Int32(m))
    var nxt = u + Int(ulen[u])
    if nxt >= m or flag[nxt] != 0:
        lastu.unsafe_store(r, Int32(u))
    var c = Int(text[u])
    if c == 0x0A or c == 0x0D:
        _ = Atomic.max(lastcr + r, Int32(u))


def gpu_cut_notcrlf_kernel(text: U8P, n: Int64, flag: I32P):
    """One thread per byte: 1 where the byte is neither CR nor LF."""
    var b = _gtid()
    if b >= Int(n):
        return
    var c = Int(text[b])
    flag.unsafe_store(b, Int32(0) if (c == 0x0A or c == 0x0D) else Int32(1))


@always_inline
def _fold_letter(cp: Int) -> Int:
    """`pretokenize.mojo` `_fold_letter`: the ASCII lowercase letter a codepoint case-folds to,
    when that fold is a contraction letter, else -1 (U+017F folds to s)."""
    if cp == 0x17F:
        return 0x73
    if cp >= 0x41 and cp <= 0x5A:
        return cp + 0x20
    if cp >= 0x61 and cp <= 0x7A:
        return cp
    return -1


@always_inline
def _contraction_end_ci(text: U8P, n: Int, i: Int) -> Int:
    """`(?i:'s|'t|'re|'ve|'m|'ll|'d)` at `i`, or -1 (`_contraction_end_ci`)."""
    if Int(text[i]) != 0x27 or i + 1 >= n:
        return -1
    var cp1 = _decode_cp(text, i + 1, n)
    var f1 = _fold_letter(cp1)
    if f1 < 0:
        return -1
    var w1 = _width(cp1)
    if f1 == 0x73 or f1 == 0x74 or f1 == 0x6D or f1 == 0x64:  # s t m d
        return i + 1 + w1
    if (f1 == 0x72 or f1 == 0x76 or f1 == 0x6C) and i + 1 + w1 < n:
        var cp2 = _decode_cp(text, i + 1 + w1, n)
        var f2 = _fold_letter(cp2)
        if (f1 == 0x72 and f2 == 0x65) or (f1 == 0x76 and f2 == 0x65) or (f1 == 0x6C and f2 == 0x6C):
            return i + 1 + w1 + _width(cp2)
    return -1


@always_inline
def _crlf_end(pfx2: I32P, ncpos: I32P, n: Int, hi: Int, j: Int) -> Int:
    """The first byte at or after `j` that is neither CR nor LF, within the segment ending at `hi`
    (hi when none; `n` is the whole text's length, the scan's total)."""
    if j >= hi:
        return hi
    var before = Int(pfx2[j])
    if before >= Int(pfx2[n]):
        return hi
    return min(Int(ncpos[before]), hi)


@always_inline
def _llama_end(
    text: U8P, n: Int, hi: Int, i: Int, digits: Int, ulen: U8P, ucls: U8P, runend: I32P, pfx: I32P,
    lastu: I32P, lastcr: I32P, pfx2: I32P, ncpos: I32P,
) -> Int:
    """`pretoken_end_llama` at the unit start `i` of the segment ending at `hi` (the host scans each
    segment alone), alternative for alternative, from the run tables. Always in (i, hi]."""
    var e = _contraction_end_ci(text, hi, i)
    if e > 0:
        return e
    var k0 = Int(ucls[i])
    var w0 = Int(ulen[i])
    var cp0 = Int(text[i])
    # 2.  [^\r\n\p{L}\p{N}]?\p{L}+
    if k0 == CL_L:
        return Int(runend[i])
    if k0 != CL_X and cp0 != 0x0A and cp0 != 0x0D and k0 != CL_N and i + w0 < hi:
        if Int(ucls[i + w0]) == CL_L:
            return Int(runend[i + w0])
    # 3.  \p{N}{1,digits}
    if k0 == CL_N:
        var j = i
        var count = 0
        while j < hi and count < digits:
            if Int(ucls[j]) != CL_N:
                break
            j += Int(ulen[j])
            count += 1
        return j
    # 4.   ?[^\s\p{L}\p{N}]+[\r\n]*
    var start = i
    if cp0 == 0x20 and i + 1 < hi and Int(ucls[i + 1]) == CL_O:
        start = i + 1
    if start > i or k0 == CL_O:
        return _crlf_end(pfx2, ncpos, n, hi, Int(runend[start]))
    # 5.  \s*[\r\n]+, 6.  \s+(?!\S), 7.  \s+
    if k0 == CL_S:
        var r = Int(pfx[i + 1]) - 1
        var cr = Int(lastcr[r])
        if cr >= i:
            return cr + 1
        var ws_end = Int(runend[i])
        if ws_end == hi:
            return ws_end
        var last = Int(lastu[r])
        if last > i:
            return last
        return ws_end
    # an invalid UTF-8 byte is its own one-byte pre-token
    return i + 1


def gpu_cut_llama_next_kernel(
    text: U8P, offs: I64P, ndocs: Int64, n: Int64, digits: Int64, ustart: U8P, ulen: U8P, ucls: U8P,
    runend: I32P, pfx: I32P, lastu: I32P, lastcr: I32P, pfx2: I32P, ncpos: I32P, spk: I32P, splen: I64P,
    has_sp: Int64, nxt: I32P,
):
    """One thread per position p in [0, n]: the scan's step from p (a unit start), p itself
    elsewhere (never reached from 0), n at n (the chain's end holds). With `has_sp`, an added
    token chosen at p (spk[p] = its index, -1 elsewhere) is one step of its own length: the chain
    walks segment, token, segment, ... as the host's split does."""
    var p = _gtid()
    var m = Int(n)
    if p > m:
        return
    if p == m or ustart[p] == 0:
        nxt.unsafe_store(p, Int32(p))
        return
    if has_sp != 0:
        var r = Int(spk[p])
        if r >= 0:
            nxt.unsafe_store(p, Int32(p + Int(splen[r])))
            return
    var hi = Int(offs[_doc_of(offs, Int(ndocs), p) + 1])
    var end = _llama_end(text, m, hi, p, Int(digits), ulen, ucls, runend, pfx, lastu, lastcr, pfx2, ncpos)
    nxt.unsafe_store(p, Int32(end))


def gpu_cut_copy_kernel(src: I32P, dst: I32P, count: Int64):
    var i = _gtid()
    if i < Int(count):
        dst.unsafe_store(i, src[i])


def gpu_cut_square_kernel(src: I32P, dst: I32P, count: Int64):
    """dst = src o src (one pointer-jumping step)."""
    var i = _gtid()
    if i < Int(count):
        dst.unsafe_store(i, src[Int(src[i])])


def gpu_cut_mark_kernel(on: I32P, jump: I32P, count: Int64):
    """Every marked position marks its jump (only chain positions are ever marked)."""
    var i = _gtid()
    if i < Int(count) and on[i] != 0:
        on.unsafe_store(Int(jump[i]), Int32(1))


def gpu_cut_gpt2_flag_kernel(tflag: U8P, n: Int64, on: I32P):
    """One thread per position p in [0, n]: 1 at a pre-token start and at the end."""
    var p = _gtid()
    var m = Int(n)
    if p > m:
        return
    on.unsafe_store(p, Int32(1) if (p == m or tflag[p] == 1) else Int32(0))


def gpu_cut_pack_kernel(on: I32P, pfx: I32P, count: Int64, bounds: I64P):
    var i = _gtid()
    if i < Int(count) and on[i] != 0:
        bounds.unsafe_store(Int(pfx[i]), Int64(i))


# ------------------------------------------------------------------ orchestration


def _bit_length(value: Int) -> Int:
    var bits = 0
    var v = value
    while v > 0:
        bits += 1
        v >>= 1
    return bits


def cut_chain_marks(ctx: DeviceContext, nxt: I32P, on: DeviceBuffer[DType.int32], count: Int) raises:
    """Mark the chain 0, nxt[0], nxt[nxt[0]], ... among `count` positions (nxt[i] > i for every
    i < count - 1 and nxt[count - 1] = count - 1, the end): `on` (count words) is zeroed, position 0
    marked, then the pointer-jumping levels of the header mark the rest. Shared by the pre-tokenizer
    cut and the corpus document cutter (tokenizer/gpu/corpus_gpu.mojo); returns synchronized."""
    var nblk1 = _blocks(count)
    var jump_a = ctx.enqueue_create_buffer[DType.int32](count)
    var jump_b = ctx.enqueue_create_buffer[DType.int32](count)
    on.enqueue_fill(Int32(0))
    var start_mark: List[Int32] = [Int32(1)]
    ctx.enqueue_copy(dst_buf=on.create_sub_buffer[DType.int32](0, 1),
                     src_ptr=I32P(unsafe_from_address=Int(start_mark.unsafe_ptr())))
    # driver-loop: over the pointer-jumping levels (log2 count of them) and the squarings that
    # rebuild each level; every pass is one parallel kernel over the `count` positions.
    var top = max(_bit_length(count - 1) - 1, 0)
    var level = top
    while level >= 0:
        var src = _p32(jump_a)
        var dst = _p32(jump_b)
        ctx.enqueue_function[gpu_cut_copy_kernel](nxt, src, Int64(count), grid_dim=nblk1, block_dim=BPE_TPB)
        for _ in range(level):  # driver-loop: over the squarings of one level
            ctx.enqueue_function[gpu_cut_square_kernel](src, dst, Int64(count), grid_dim=nblk1, block_dim=BPE_TPB)
            var held = src
            src = dst
            dst = held
        ctx.enqueue_function[gpu_cut_mark_kernel](_p32(on), src, Int64(count), grid_dim=nblk1, block_dim=BPE_TPB)
        level -= 1
    ctx.synchronize()
    _ = len(start_mark)
    _ = jump_a^
    _ = jump_b^


def cut_llama_marks(
    ctx: DeviceContext, text: U8P, offs: I64P, ndocs: Int, n: Int, digits: Int, ustart: U8P, ulen: U8P,
    ucls: U8P, spk: I32P, splen: I64P, has_sp: Bool, on: DeviceBuffer[DType.int32], st: I32P,
) raises:
    """The Llama 3 / Qwen 2 piece starts of the `n` device bytes at `text`, whose units are already
    computed over the `ndocs` segments at `offs`: `on` (n + 1 words) marks every piece start and n.
    The runs, the CR/LF tables, the per-unit step (segment-aware: a step never passes its segment's
    end) and the pointer-jumping chain (`cut_chain_marks`); with `has_sp` the added tokens chosen
    at `spk` (lengths `splen`) are pieces of their own. Shared by the plain cut and the GPU
    `Tokenizer.encode` (tokenizer/gpu/vocab_gpu.mojo); returns synchronized."""
    var nblk = _blocks(n)
    var nblk1 = _blocks(n + 1)
    var flag = ctx.enqueue_create_buffer[DType.int32](n)
    var pfx = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var runpos = ctx.enqueue_create_buffer[DType.int32](n)
    var runend = ctx.enqueue_create_buffer[DType.int32](n)
    var lastu = ctx.enqueue_create_buffer[DType.int32](n)
    var lastcr = ctx.enqueue_create_buffer[DType.int32](n)
    var flag2 = ctx.enqueue_create_buffer[DType.int32](n)
    var pfx2 = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var ncpos = ctx.enqueue_create_buffer[DType.int32](n)
    var nxt = ctx.enqueue_create_buffer[DType.int32](n + 1)
    lastcr.enqueue_fill(Int32(-1))
    ctx.enqueue_function[gpu_cut_runflag_kernel](ustart, ucls, offs, Int64(ndocs), Int64(n), _p32(flag),
                                                 grid_dim=nblk, block_dim=BPE_TPB)
    _scan(ctx, _p32(flag), _p32(pfx), st, n)
    ctx.enqueue_function[gpu_cut_runpos_kernel](_p32(flag), _p32(pfx), Int64(n), _p32(runpos),
                                                grid_dim=nblk, block_dim=BPE_TPB)
    ctx.enqueue_function[gpu_cut_runinfo_kernel](
        text, ustart, ulen, _p32(flag), _p32(pfx), _p32(runpos), Int64(n), _p32(runend),
        _p32(lastu), _p32(lastcr), grid_dim=nblk, block_dim=BPE_TPB,
    )
    ctx.enqueue_function[gpu_cut_notcrlf_kernel](text, Int64(n), _p32(flag2), grid_dim=nblk, block_dim=BPE_TPB)
    _scan(ctx, _p32(flag2), _p32(pfx2), st, n)
    ctx.enqueue_function[gpu_cut_runpos_kernel](_p32(flag2), _p32(pfx2), Int64(n), _p32(ncpos),
                                                grid_dim=nblk, block_dim=BPE_TPB)
    ctx.enqueue_function[gpu_cut_llama_next_kernel](
        text, offs, Int64(ndocs), Int64(n), Int64(digits), ustart, ulen, ucls, _p32(runend), _p32(pfx),
        _p32(lastu), _p32(lastcr), _p32(pfx2), _p32(ncpos), spk, splen, Int64(1) if has_sp else Int64(0),
        _p32(nxt), grid_dim=nblk1, block_dim=BPE_TPB,
    )
    cut_chain_marks(ctx, _p32(nxt), on, n + 1)
    _ = flag^
    _ = pfx^
    _ = runpos^
    _ = runend^
    _ = lastu^
    _ = lastcr^
    _ = flag2^
    _ = pfx2^
    _ = ncpos^
    _ = nxt^


def pretokenize_gpu_device(
    ctx: DeviceContext, classes: BpeGpuClasses, text: U8P, n: Int, pattern: Int, bounds: I64P, st: I32P,
) raises:
    """The boundaries of the `n` device bytes at `text` under `pattern`, written in order at
    `bounds` (room for n + 1), their count at st[S_TOTAL] (st holds S_LEN words, zeroed by the
    caller). Every step runs on the device."""
    if pattern != PATTERN_GPT2 and pattern != PATTERN_LLAMA3 and pattern != PATTERN_QWEN2:
        raise Error("pretokenize: unknown pattern code " + String(pattern))
    var nblk = _blocks(n)
    var nblk1 = _blocks(n + 1)
    var offs_host: List[Int64] = [Int64(0), Int64(n)]
    var d_offs = ctx.enqueue_create_buffer[DType.int64](2)
    _up64(ctx, d_offs, Int(offs_host.unsafe_ptr()), 2)
    var mflag = ctx.enqueue_create_buffer[DType.uint8](n)
    var ustart = ctx.enqueue_create_buffer[DType.uint8](n)
    var ulen = ctx.enqueue_create_buffer[DType.uint8](n)
    var ucls = ctx.enqueue_create_buffer[DType.uint8](n)
    var on = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var pfx_on = ctx.enqueue_create_buffer[DType.int32](n + 2)
    mflag.enqueue_fill(UInt8(0))
    ctx.enqueue_function[gpu_unit_kernel](
        text, _p64(d_offs), Int64(1), Int64(n), _p8(mflag), Int64(0), Int64(0),
        _p64(classes.letters), Int64(classes.nl), _p64(classes.numbers), Int64(classes.nn),
        _p64(classes.spaces), Int64(classes.ns), _p8(ustart), _p8(ulen), _p8(ucls),
        grid_dim=nblk, block_dim=BPE_TPB,
    )
    if pattern == PATTERN_GPT2:
        var tflag = ctx.enqueue_create_buffer[DType.uint8](n)
        ctx.enqueue_function[gpu_start_kernel](
            text, _p64(d_offs), Int64(1), Int64(n), _p8(mflag), Int64(0), _p8(ustart), _p8(ulen), _p8(ucls),
            _p8(tflag), grid_dim=nblk, block_dim=BPE_TPB,
        )
        ctx.enqueue_function[gpu_cut_gpt2_flag_kernel](_p8(tflag), Int64(n), _p32(on), grid_dim=nblk1, block_dim=BPE_TPB)
        ctx.synchronize()
        _ = tflag^
    else:
        # One segment, no added token: the special arguments are never read (has_sp 0).
        cut_llama_marks(ctx, text, _p64(d_offs), 1, n, 3 if pattern == PATTERN_LLAMA3 else 1, _p8(ustart),
                        _p8(ulen), _p8(ucls), _p32(on), _p64(d_offs), False, on, st)
    _scan(ctx, _p32(on), _p32(pfx_on), st, n + 1)
    ctx.enqueue_function[gpu_cut_pack_kernel](_p32(on), _p32(pfx_on), Int64(n + 1), bounds,
                                              grid_dim=nblk1, block_dim=BPE_TPB)
    ctx.synchronize()
    _ = len(offs_host)
    _ = d_offs^
    _ = mflag^
    _ = ustart^
    _ = ulen^
    _ = ucls^
    _ = on^
    _ = pfx_on^


def pretokenize_gpu_into(text_addr: Int, n: Int, pattern: Int, bounds_addr: Int) raises -> Int:
    """`pretokenize_pattern` on host input: the data uploaded once, every step on the device
    (`pretokenize_gpu_device`), the boundaries (int64, room for n + 1 at `bounds_addr`) and the state
    words read back once, after the last kernel. Returns the number of boundaries (n + 1 at most,
    1 for empty data)."""
    var bounds = I64P(unsafe_from_address=bounds_addr)
    if n == 0:
        bounds.unsafe_store(0, Int64(0))
        return 1
    var ctx = bpe_gpu_ctx()
    var classes = BpeGpuClasses(ctx, builtin_unicode_classes())
    var d_text = ctx.enqueue_create_buffer[DType.uint8](n)
    var d_bounds = ctx.enqueue_create_buffer[DType.int64](n + 1)
    var d_st = ctx.enqueue_create_buffer[DType.int32](S_LEN)
    _up8(ctx, d_text, text_addr, n)
    d_st.enqueue_fill(Int32(0))
    pretokenize_gpu_device(ctx, classes, _p8(d_text), n, pattern, _p64(d_bounds), _p32(d_st))
    # One read-back after the last kernel: the state words and the whole boundary buffer (the
    # caller's room for n + 1; only the first `count` are boundaries).
    var h_st = List[Int32](length=S_LEN, fill=Int32(0))
    ctx.enqueue_copy(dst_ptr=I32P(unsafe_from_address=Int(h_st.unsafe_ptr())), src_buf=d_st)
    ctx.enqueue_copy(dst_ptr=bounds, src_buf=d_bounds)
    ctx.synchronize()
    var count = Int(h_st[S_TOTAL])
    if count < 2 or count > n + 1:
        raise Error("pretokenize (gpu): the device cut returned " + String(count) + " boundaries for "
            + String(n) + " bytes")
    _ = d_text^
    _ = d_bounds^
    _ = d_st^
    _ = classes^
    return count

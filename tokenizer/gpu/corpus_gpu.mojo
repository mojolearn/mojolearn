# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CORPUS PREPARATION ON THE GPU, EVERY VENDOR (lane vdf2-tok, 2026-11-24): the `device="gpu"` side
of `mojolearn.lm_corpus.prepare`, the same vocabulary and the same token ids, id for id, as the host
column (`bindings/lm_corpus_prepare_run.mojo` over `tokenizer/impl/tokenizer_host_shell_impl.mojo`
`bpe_train_corpus_fill` / `bpe_encode_corpus_into`).

Integer work only (byte compares, integer scans, index gathers, integer atomics whose result does
not depend on their order: a max and a count): no fold whose order could move a bit, so the host
column is unchanged and every vendor gives the same tokens.

THE DOCUMENT CUTTER (`corpus_cut_device`). The host `_cut_documents` walks each piece between split
points: a document runs from `at` to min(at + document_bytes, piece end), pulled back to just after
the last 0x0A inside it when it stops before the piece end. That walk is a chain, so it is computed
as `tokenizer/gpu/pretokenize_gpu.mojo` computes the pre-tokenizer's:

  newlines  one flag per byte, an integer scan, and the scatter of each newline's position
            (`gpu_cut_runpos_kernel`, reused): the last 0x0A at or before byte j is
            nlpos[pfx[j + 1] - 1];
  steps     `gpu_corpus_next_kernel`: nxt[i] = the end of the document the host walk would start
            at i, for EVERY position i (the split points are a handful of words read per thread);
  chain     `cut_chain_marks` (pretokenize_gpu.mojo, shared): pointer jumping marks 0, nxt[0], ...;
  pack      the marked positions in order (`gpu_cut_pack_kernel`, reused): the document offsets,
            left on the device for the trainer or the encoder to read in place.

TRAINING reads the cut documents in place (`tokenizer/gpu/bpe_gpu.mojo` `bpe_gpu_train_device`); the
merges come back once. ENCODING is `bpe_gpu_encode_device` on the device offsets, then on the device
the token offset of every split point (a scan of the per-document counts and one binary search per
point) and the two id statistics; the state words, the offsets and the statistics come back once,
then the ids.

TRANSFERS: the corpus bytes are uploaded once per operation, the class tables and the rank table once
per operation; nothing is read back between kernels except the one sizing word of the cut (the
document count, which sizes the launches after it).

THE ENTRY (`corpus_gpu_entry_dispatch`, the body of `_mojolearn_x_tokenizer`'s `abi("C")` entry,
ABI in core/tokenizer_gpu_entry_table.mojo): called by the host RUN with the GIL released; it never
touches the GIL.
"""
from std.atomic import Atomic
from max.gpu.host import DeviceBuffer, DeviceContext

from core.entry_table import entry_op_plain
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.tokenizer_gpu_entry_table import (
    TOK_GPU_OP_CORPUS_ENCODE, TOK_GPU_OP_CORPUS_TRAIN, TOK_GPU_OP_VOCAB_DECODE, TOK_GPU_OP_VOCAB_ENCODE,
    TOK_GPU_OP_VOCAB_FREE, TOK_GPU_OP_VOCAB_LOAD,
)
from tokenizer.gpu.vocab_gpu import vocab_gpu_decode, vocab_gpu_encode, vocab_gpu_free, vocab_gpu_load
from tokenizer.encoding import BpeTokenizer
from tokenizer.impl.ranks import parse_rank_table
from tokenizer.impl.unicode_class import builtin_unicode_classes
from tokenizer.fast.bpe_device import BPE_TPB, S_FAIL, S_LEN, S_TOTAL, BpeMerges
from tokenizer.gpu.bpe_gpu import (
    I32P, I64P, U8P, BpeGpuClasses, BpeGpuTable, bpe_gpu_ctx, bpe_gpu_encode_device, bpe_gpu_train_device,
    _gtid, _blocks, _up8, _up64, _p8, _p32, _p64, _scan,
)
from tokenizer.gpu.pretokenize_gpu import cut_chain_marks, gpu_cut_runpos_kernel, gpu_cut_pack_kernel

# The device tokenizer's positions are int32 (`bpe_gpu.mojo`'s scans and piece tables).
comptime CORPUS_GPU_MAX_BYTES = 2147483646


# ------------------------------------------------------------------ kernels


def gpu_corpus_nl_flag_kernel(text: U8P, m: Int64, flag: I32P):
    """One thread per byte: 1 at a 0x0A, else 0."""
    var b = _gtid()
    if b < Int(m):
        flag.unsafe_store(b, Int32(1) if text[b] == UInt8(0x0A) else Int32(0))


def gpu_corpus_next_kernel(cuts: I64P, ncuts: Int64, m: Int64, width: Int64, pfx: I32P, nlpos: I32P, nxt: I32P):
    """One thread per position i in [0, m]: nxt[i] = where the host `_cut_documents` ends the
    document it starts at i (`cuts` sorted, 0 and m included; pfx[j] = the 0x0A count in [0, j);
    nlpos[r] = the r-th 0x0A); nxt[m] = m."""
    var i = _gtid()
    var mm = Int(m)
    if i > mm:
        return
    if i == mm:
        nxt.unsafe_store(i, Int32(mm))
        return
    var b = mm
    for k in range(Int(ncuts)):  # small-loop(at most ten sorted split points)
        var c = Int(cuts[k])
        if c > i:
            b = c
            break
    var stop = min(i + Int(width), b)
    if stop < b:
        var c = Int(pfx[stop])
        if c > 0:
            var cut = Int(nlpos[c - 1])
            if cut >= i:
                stop = cut + 1
    nxt.unsafe_store(i, Int32(stop))


def gpu_corpus_count32_kernel(counts: I64P, c32: I32P, ndocs: Int64):
    var d = _gtid()
    if d < Int(ndocs):
        c32.unsafe_store(d, Int32(Int(counts[d])))


def gpu_corpus_bounds_kernel(points: I64P, npoints: Int64, offs: I64P, ndocs: Int64, cpfx: I32P, bounds: I64P):
    """One thread per split point: the token offset of the document starting at it (cpfx is the
    exclusive scan of the per-document counts, cpfx[ndocs] the total), 0 where none starts (the
    host's fill)."""
    var j = _gtid()
    if j >= Int(npoints):
        return
    var p = points[j]
    var lo = 0
    var hi = Int(ndocs)
    while lo < hi:
        var mid = (lo + hi) // 2
        if offs[mid] < p:
            lo = mid + 1
        else:
            hi = mid
    bounds.unsafe_store(j, Int64(Int(cpfx[lo])) if offs[lo] == p else Int64(0))


def gpu_corpus_stats_kernel(packed: I32P, st: I32P, cap: Int64, ist: I32P):
    """One thread per id slot: the largest id (atomic max) and the ids above 255 (atomic count)
    over the st[S_TOTAL] ids written; integer atomics, so the order threads run in moves nothing."""
    var i = _gtid()
    if i >= Int(cap) or i >= Int(st[S_TOTAL]):
        return
    var v = packed[i]
    _ = Atomic.max(ist, v)
    if v > Int32(255):
        _ = Atomic.fetch_add(ist + 1, Int32(1))


# ------------------------------------------------------------------ orchestration


def corpus_cut_device(
    ctx: DeviceContext, text: U8P, m: Int, cuts_host: List[Int64], width: Int, offs: DeviceBuffer[DType.int64],
) raises -> Int:
    """The host `_cut_documents` of the m device bytes at `text` (cut at `cuts_host`, sorted, 0 and m
    included; documents of at most `width` >= 1 bytes): the document offsets (ndocs + 1 of them, the
    last m) written at `offs` (room m + 1). Returns ndocs (0 for m == 0)."""
    if m == 0:
        return 0
    var nblk = _blocks(m)
    var nblk1 = _blocks(m + 1)
    var ncuts = len(cuts_host)
    var d_cuts = ctx.enqueue_create_buffer[DType.int64](max(ncuts, 1))
    _up64(ctx, d_cuts, Int(cuts_host.unsafe_ptr()), ncuts)
    var flag = ctx.enqueue_create_buffer[DType.int32](m)
    var pfx = ctx.enqueue_create_buffer[DType.int32](m + 1)
    var nlpos = ctx.enqueue_create_buffer[DType.int32](m)
    var nxt = ctx.enqueue_create_buffer[DType.int32](m + 1)
    var on = ctx.enqueue_create_buffer[DType.int32](m + 1)
    var pfx_on = ctx.enqueue_create_buffer[DType.int32](m + 2)
    var st = ctx.enqueue_create_buffer[DType.int32](S_LEN)
    st.enqueue_fill(Int32(0))
    ctx.enqueue_function[gpu_corpus_nl_flag_kernel](text, Int64(m), _p32(flag), grid_dim=nblk, block_dim=BPE_TPB)
    _scan(ctx, _p32(flag), _p32(pfx), _p32(st), m)
    ctx.enqueue_function[gpu_cut_runpos_kernel](_p32(flag), _p32(pfx), Int64(m), _p32(nlpos),
                                                grid_dim=nblk, block_dim=BPE_TPB)
    ctx.enqueue_function[gpu_corpus_next_kernel](
        _p64(d_cuts), Int64(ncuts), Int64(m), Int64(width), _p32(pfx), _p32(nlpos), _p32(nxt),
        grid_dim=nblk1, block_dim=BPE_TPB,
    )
    cut_chain_marks(ctx, _p32(nxt), on, m + 1)
    _scan(ctx, _p32(on), _p32(pfx_on), _p32(st), m + 1)
    ctx.enqueue_function[gpu_cut_pack_kernel](_p32(on), _p32(pfx_on), Int64(m + 1), _p64(offs),
                                              grid_dim=nblk1, block_dim=BPE_TPB)
    # The one sizing word: the offset count (it sizes every launch after the cut).
    var h_st = List[Int32](length=S_LEN, fill=Int32(0))
    ctx.enqueue_copy(dst_ptr=I32P(unsafe_from_address=Int(h_st.unsafe_ptr())), src_buf=st)
    ctx.synchronize()
    var count = Int(h_st[S_TOTAL])
    if count < 2 or count > m + 1:
        raise Error("RuntimeError: lm_corpus.prepare (gpu): the document cutter returned " + String(count)
                    + " offsets for " + String(m) + " bytes")
    _ = len(cuts_host)
    _ = d_cuts^
    _ = flag^
    _ = pfx^
    _ = nlpos^
    _ = nxt^
    _ = on^
    _ = pfx_on^
    _ = st^
    return count - 1


def _check_span(m: Int, width: Int) raises:
    if m < 0:
        raise Error("ValueError: lm_corpus.prepare (gpu): negative byte count")
    if m > CORPUS_GPU_MAX_BYTES:
        raise Error("ValueError: mojolearn: lm_corpus.prepare(device='gpu') takes a corpus of at most "
                    + String(CORPUS_GPU_MAX_BYTES) + " bytes; this one has " + String(m)
                    + "; use device='cpu' for it")
    if width < 1:
        raise Error("ValueError: document_bytes must be >= 1")


def corpus_gpu_train(
    span_addr: Int, m: Int, cuts: List[Int64], width: Int, vocab_size: Int, min_frequency: Int, pairs_addr: Int,
    pairs_cap: Int, words_addr: Int,
) raises -> Int:
    """TOK_GPU_OP_CORPUS_TRAIN (core/tokenizer_gpu_entry_table.mojo): the documents of the span cut
    on the device and trained on the device in place. Returns the document count."""
    _check_span(m, width)
    if m == 0:
        raise Error("ValueError: train needs at least one document")
    if vocab_size < 256:
        raise Error("ValueError: vocab_size is below the 256 single-byte tokens")
    if min_frequency < 1:
        raise Error("ValueError: min_frequency must be at least 1")
    var merges = BpeMerges(max(2 * (vocab_size - 256), 2))
    if pairs_cap < len(merges.pairs):
        raise Error("RuntimeError: lm_corpus.prepare (gpu): the merge buffer holds " + String(pairs_cap)
                    + " words, " + String(len(merges.pairs)) + " needed")
    var ctx = bpe_gpu_ctx()
    var classes = BpeGpuClasses(ctx, builtin_unicode_classes())
    var d_text = ctx.enqueue_create_buffer[DType.uint8](m)
    var d_offs = ctx.enqueue_create_buffer[DType.int64](m + 1)
    _up8(ctx, d_text, span_addr, m)
    var ndocs = corpus_cut_device(ctx, _p8(d_text), m, cuts, width, d_offs)
    var ng = 0
    if vocab_size > 256:
        ng = bpe_gpu_train_device(ctx, classes, _p8(d_text), _p64(d_offs), ndocs, m, vocab_size, min_frequency,
                                  merges)
    var pairs = I32P(unsafe_from_address=pairs_addr)
    for k in range(2 * merges.n_merges):  # driver-loop: over the merges read back once (2 * (vocab_size - 256) words at most)
        pairs.unsafe_store(k, merges.pairs[k])
    var words = I64P(unsafe_from_address=words_addr)
    words.unsafe_store(0, Int64(merges.n_merges))
    words.unsafe_store(1, Int64(merges.n_ties))
    words.unsafe_store(2, Int64(ng))
    _ = d_text^
    _ = d_offs^
    _ = classes^
    return ndocs


def corpus_gpu_encode(
    data_addr: Int, n: Int, points: List[Int64], width: Int, ranks_addr: Int, ranks_bytes: Int, ids_addr: Int,
    bounds_addr: Int, stats_addr: Int,
) raises -> Int:
    """TOK_GPU_OP_CORPUS_ENCODE (core/tokenizer_gpu_entry_table.mojo): the host
    `bpe_encode_corpus_into` on the device. Returns the id count."""
    _check_span(n, width)
    var npoints = len(points)
    var bounds = I64P(unsafe_from_address=bounds_addr)
    var stats = I64P(unsafe_from_address=stats_addr)
    if n == 0:
        for j in range(npoints):  # small-loop(at most ten split points)
            bounds.unsafe_store(j, Int64(0))
        stats.unsafe_store(0, Int64(0))
        stats.unsafe_store(1, Int64(-1))
        stats.unsafe_store(2, Int64(0))
        return 0
    var raw = List[UInt8](unsafe_uninit_length=max(ranks_bytes, 0))
    var src = U8P(unsafe_from_address=ranks_addr)
    for k in range(ranks_bytes):  # driver-loop: over the rank text bytes, once per call (admission)
        raw[k] = src[k]
    var text = String(StringSlice(unsafe_from_utf8=Span(raw)))
    var tok = BpeTokenizer(parse_rank_table(text, "<rank text>"), builtin_unicode_classes())
    var ctx = bpe_gpu_ctx()
    var table = BpeGpuTable(ctx, tok)
    var d_text = ctx.enqueue_create_buffer[DType.uint8](n)
    var d_offs = ctx.enqueue_create_buffer[DType.int64](n + 1)
    _up8(ctx, d_text, data_addr, n)
    var ndocs = corpus_cut_device(ctx, _p8(d_text), n, points, width, d_offs)
    var d_packed = ctx.enqueue_create_buffer[DType.int32](n)
    var d_counts = ctx.enqueue_create_buffer[DType.int64](ndocs)
    var d_st = ctx.enqueue_create_buffer[DType.int32](S_LEN)
    d_st.enqueue_fill(Int32(0))
    d_counts.enqueue_fill(Int64(0))
    bpe_gpu_encode_device(ctx, table, _p8(d_text), _p64(d_offs), ndocs, n, False, tok.eot_id(), _p32(d_packed),
                          _p64(d_counts), _p32(d_st))
    # The split points' token offsets and the two id statistics, on the device.
    var d_c32 = ctx.enqueue_create_buffer[DType.int32](ndocs)
    var d_cpfx = ctx.enqueue_create_buffer[DType.int32](ndocs + 1)
    var d_st2 = ctx.enqueue_create_buffer[DType.int32](S_LEN)
    var d_points = ctx.enqueue_create_buffer[DType.int64](max(npoints, 1))
    var d_bounds = ctx.enqueue_create_buffer[DType.int64](max(npoints, 1))
    var d_ist = ctx.enqueue_create_buffer[DType.int32](2)
    var ist_init: List[Int32] = [Int32(-1), Int32(0)]
    d_st2.enqueue_fill(Int32(0))
    _up64(ctx, d_points, Int(points.unsafe_ptr()), npoints)
    ctx.enqueue_copy(dst_buf=d_ist, src_ptr=I32P(unsafe_from_address=Int(ist_init.unsafe_ptr())))
    ctx.enqueue_function[gpu_corpus_count32_kernel](_p64(d_counts), _p32(d_c32), Int64(ndocs),
                                                    grid_dim=_blocks(ndocs), block_dim=BPE_TPB)
    _scan(ctx, _p32(d_c32), _p32(d_cpfx), _p32(d_st2), ndocs)
    ctx.enqueue_function[gpu_corpus_bounds_kernel](  # small-launch(one thread per split point, at most ten)
        _p64(d_points), Int64(npoints), _p64(d_offs), Int64(ndocs), _p32(d_cpfx), _p64(d_bounds),
        grid_dim=1, block_dim=BPE_TPB,
    )
    ctx.enqueue_function[gpu_corpus_stats_kernel](_p32(d_packed), _p32(d_st), Int64(n), _p32(d_ist),
                                                  grid_dim=_blocks(n), block_dim=BPE_TPB)
    # One read-back of the small results, then the ids it sizes.
    var h_st = List[Int32](length=S_LEN, fill=Int32(0))
    var h_ist = List[Int32](length=2, fill=Int32(0))
    ctx.enqueue_copy(dst_ptr=I32P(unsafe_from_address=Int(h_st.unsafe_ptr())), src_buf=d_st)
    ctx.enqueue_copy(dst_ptr=I32P(unsafe_from_address=Int(h_ist.unsafe_ptr())), src_buf=d_ist)
    if npoints > 0:
        ctx.enqueue_copy(dst_ptr=bounds, src_buf=d_bounds.create_sub_buffer[DType.int64](0, npoints))
    ctx.synchronize()
    if h_st[S_FAIL] != Int32(0):
        raise Error("bpe_encode_corpus: internal document encode failed")
    var total = Int(h_st[S_TOTAL])
    if total < 0 or total > n:
        raise Error("RuntimeError: corpus encoder returned an invalid token count")
    if total > 0:
        ctx.enqueue_copy(dst_ptr=I32P(unsafe_from_address=ids_addr),
                         src_buf=d_packed.create_sub_buffer[DType.int32](0, total))
        ctx.synchronize()
    stats.unsafe_store(0, Int64(ndocs))
    stats.unsafe_store(1, Int64(Int(h_ist[0])))
    stats.unsafe_store(2, Int64(Int(h_ist[1])))
    _ = len(points)
    _ = len(ist_init)
    _ = d_text^
    _ = d_offs^
    _ = d_packed^
    _ = d_counts^
    _ = d_st^
    _ = d_c32^
    _ = d_cpfx^
    _ = d_st2^
    _ = d_points^
    _ = d_bounds^
    _ = d_ist^
    _ = table^
    return total


def _words(address: Int, count: Int) -> List[Int64]:
    var source = I64P(unsafe_from_address=address)
    var words = List[Int64](capacity=count)
    for k in range(count):  # small-loop(at most ten split points)
        words.append(source[k])
    return words^


def corpus_gpu_entry_dispatch(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    """The body of `_mojolearn_x_tokenizer`'s `abi("C")` entry: span admission, the operation,
    its result word, a refusal to the sink. No GIL operation (the caller released it)."""
    try:
        var op = entry_op_plain(operation)
        if not result or na < 0 or np < 0 or (na > 0 and not aa) or (np > 0 and not pp):
            raise Error("tokenizer gpu entry: invalid request spans")
        var a = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=aa)
        var p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pp)
        var value = 0
        if op == TOK_GPU_OP_CORPUS_TRAIN:
            if na != 4 or np != 6 or p[1] < 0:
                raise Error("tokenizer gpu entry: corpus train takes 4 addresses and 6 words")
            value = corpus_gpu_train(a[0], p[0], _words(a[1], p[1]), p[2], p[3], p[4], a[2], p[5], a[3])
        elif op == TOK_GPU_OP_CORPUS_ENCODE:
            if na != 6 or np != 4 or p[1] < 0:
                raise Error("tokenizer gpu entry: corpus encode takes 6 addresses and 4 words")
            value = corpus_gpu_encode(a[0], p[0], _words(a[1], p[1]), p[2], a[2], p[3], a[3], a[4], a[5])
        elif op == TOK_GPU_OP_VOCAB_LOAD:
            if na != 9 or np != 8:
                raise Error("tokenizer gpu entry: vocab load takes 9 addresses and 8 words")
            value = vocab_gpu_load(a, p)
        elif op == TOK_GPU_OP_VOCAB_ENCODE:
            if na != 3 or np != 2 or p[0] < 0:
                raise Error("tokenizer gpu entry: vocab encode takes 3 addresses and 2 words")
            value = vocab_gpu_encode(a[0], a[1], p[0], p[1] != 0, a[2])
        elif op == TOK_GPU_OP_VOCAB_DECODE:
            if na != 3 or np != 2 or p[0] < 0 or p[1] < 0:
                raise Error("tokenizer gpu entry: vocab decode takes 3 addresses and 2 words")
            value = vocab_gpu_decode(a[0], a[1], p[0], a[2], p[1])
        elif op == TOK_GPU_OP_VOCAB_FREE:
            if na != 1:
                raise Error("tokenizer gpu entry: vocab free takes 1 address")
            vocab_gpu_free(a[0])
        else:
            raise Error("tokenizer gpu entry: unknown operation " + String(op))
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mojolearn.models.Tokenizer` ENCODE AND DECODE ON THE GPU, EVERY VENDOR (lane vdf2-tokmodel,
2026-11-24): the `device="gpu"` side of the pretrained-vocabulary tokenizer, the same ids, id for
id, and the same bytes as the host column (`tokenizer/vocab.mojo` `VocabTokenizer.encode` /
`append_decoded`), for the GPT-2, Llama 3 and Qwen 2 patterns and the file's added tokens.

Integer work only (byte compares, the rank table's FNV-1a hash and probe, integer scans, index
gathers): no fold whose order could move a bit, so the host column is unchanged and every vendor
gives the same ids. NFC normalization stays in the door's UNPACK on both sides.

THE TABLE (`VocabGpuTable`) is uploaded ONCE, when a Tokenizer is built with device='gpu': the rank
table (arena, offsets, lengths, hash buckets), the pinned Unicode class tables, the added tokens
(content arena, offsets, lengths, ids, in the host's order: longest first, then by bytes) and the
id -> added-token map. It lives in this image (`alloc`), addressed by the handle word the door
keeps on its `VocabHandle`, and is freed through the entry when that handle goes.

ENCODE (`vocab_gpu_encode_device`):

  added tokens (allow_special and the file has some). The host splits leftmost first, the longer
     content winning a tie in position: from i it takes the first position p >= i where any added
     token matches, the first token (in its order) matching there, encodes [i, p) alone and goes
     on at p + its length. Per byte the first matching token (`vgpu_special_match_kernel`); the
     host's walk is then the chain 0, s(0), s(s(0)), ... with s(p) = p + length where a token
     matches at p and p + 1 elsewhere, which `cut_chain_marks` (tokenizer/gpu/pretokenize_gpu.mojo)
     finds by pointer jumping. A token is CHOSEN where the chain stands on a match. The chosen
     tokens' starts and ends cut the text into segments (an integer scan packs them as segment
     offsets; the tail is filled with n so the segment count stays a bound known on the host);
     a token is a segment of its own.
  pieces. The units of every segment (`gpu_unit_kernel`, bounded by its segment), then for GPT-2
     the local start rule (`gpu_start_kernel`) and for Llama 3 / Qwen 2 the segment-aware step
     chain (`cut_llama_marks`); a chosen token's segment is one piece of flag 2.
  merge. `gpu_merge_kernel` per piece (the host `bpe_append`), a chosen token's piece written as its
     id (`vgpu_special_id_kernel`), then the ids packed by an integer scan.

DECODE is a gather: per id its byte length (an added token's content wins over a rank with the
same id; an id that is neither records its position, the first by an atomic min, refused after
the call in the host's words), a scan, a per-id copy.

TRANSFERS: the text (or the ids) is uploaded once per call; the result and the state words come
back once, after the last kernel. No host step inside the call touches the data.
"""
from std.atomic import Atomic
from std.memory import alloc
from max.gpu.host import DeviceBuffer, DeviceContext

from tokenizer.impl.unicode_class import builtin_unicode_classes
from tokenizer.impl.pretokenize import PATTERN_GPT2, PATTERN_LLAMA3, PATTERN_QWEN2
from tokenizer.fast.bpe_device import BPE_TPB, S_FAIL, S_LEN, S_TOTAL
from tokenizer.gpu.bpe_gpu import (
    G_BAD, G_NONE, I32P, I64P, U8P, BpeGpuClasses, bpe_gpu_ctx, gpu_unit_kernel, gpu_start_kernel,
    gpu_merge_kernel, gpu_pack_kernel, gpu_state_init_kernel, _gtid, _blocks, _up8, _up64, _p8, _p32,
    _p64, _scan,
)
from tokenizer.gpu.pretokenize_gpu import cut_chain_marks, cut_llama_marks, gpu_cut_pack_kernel


# ------------------------------------------------------------------ the device table


struct VocabGpuTable(Movable):
    """A Tokenizer's device-resident tables, uploaded once at construction with device='gpu'."""

    var arena: DeviceBuffer[DType.uint8]
    var toff: DeviceBuffer[DType.int64]
    var tlen: DeviceBuffer[DType.int64]
    var buckets: DeviceBuffer[DType.int64]
    var classes: BpeGpuClasses
    var sp_arena: DeviceBuffer[DType.uint8]
    var sp_off: DeviceBuffer[DType.int64]
    var sp_len: DeviceBuffer[DType.int64]
    var sp_id: DeviceBuffer[DType.int64]
    var added_of: DeviceBuffer[DType.int64]
    var mask: Int
    var ntok: Int
    var nsp: Int
    var nadded: Int
    var pattern: Int

    def __init__(
        out self, ctx: DeviceContext, arena_addr: Int, arena_bytes: Int, off_addr: Int, len_addr: Int,
        ntok: Int, buckets_addr: Int, nbuckets: Int, mask: Int, sp_arena_addr: Int, sp_arena_bytes: Int,
        sp_off_addr: Int, sp_len_addr: Int, sp_id_addr: Int, nsp: Int, added_of_addr: Int, nadded: Int,
        pattern: Int,
    ) raises:
        if pattern != PATTERN_GPT2 and pattern != PATTERN_LLAMA3 and pattern != PATTERN_QWEN2:
            raise Error("tokenizer (gpu): unknown pattern code " + String(pattern))
        if ntok < 1 or nsp < 0 or nadded < 0 or nbuckets < 1 or mask < 0:
            raise Error("tokenizer (gpu): invalid vocabulary table sizes")
        self.mask = mask
        self.ntok = ntok
        self.nsp = nsp
        self.nadded = nadded
        self.pattern = pattern
        self.classes = BpeGpuClasses(ctx, builtin_unicode_classes())
        self.arena = ctx.enqueue_create_buffer[DType.uint8](max(arena_bytes, 1))
        self.toff = ctx.enqueue_create_buffer[DType.int64](ntok)
        self.tlen = ctx.enqueue_create_buffer[DType.int64](ntok)
        self.buckets = ctx.enqueue_create_buffer[DType.int64](nbuckets)
        self.sp_arena = ctx.enqueue_create_buffer[DType.uint8](max(sp_arena_bytes, 1))
        self.sp_off = ctx.enqueue_create_buffer[DType.int64](max(nsp, 1))
        self.sp_len = ctx.enqueue_create_buffer[DType.int64](max(nsp, 1))
        self.sp_id = ctx.enqueue_create_buffer[DType.int64](max(nsp, 1))
        self.added_of = ctx.enqueue_create_buffer[DType.int64](max(nadded, 1))
        _up8(ctx, self.arena, arena_addr, arena_bytes)
        _up64(ctx, self.toff, off_addr, ntok)
        _up64(ctx, self.tlen, len_addr, ntok)
        _up64(ctx, self.buckets, buckets_addr, nbuckets)
        _up8(ctx, self.sp_arena, sp_arena_addr, sp_arena_bytes)
        _up64(ctx, self.sp_off, sp_off_addr, nsp)
        _up64(ctx, self.sp_len, sp_len_addr, nsp)
        _up64(ctx, self.sp_id, sp_id_addr, nsp)
        _up64(ctx, self.added_of, added_of_addr, nadded)
        # The caller's host spans are read by the copies above until the context runs them.
        ctx.synchronize()


# ------------------------------------------------------------------ encode kernels


def vgpu_special_match_kernel(text: U8P, n: Int64, sp_arena: U8P, sp_off: I64P, sp_len: I64P, nsp: Int64,
                              sm: I32P):
    """One thread per position p in [0, n]: the first added token (in the host's order) whose
    content matches at p and ends inside the text, or -1 (always -1 at n)."""
    var b = _gtid()
    var m = Int(n)
    if b > m:
        return
    var found = -1
    if b < m:
        for r in range(Int(nsp)):
            var k = Int(sp_len[r])
            if b + k > m:
                continue
            var o = Int(sp_off[r])
            var hit = True
            for j in range(k):
                if text[b + j] != sp_arena[o + j]:
                    hit = False
                    break
            if hit:
                found = r
                break
    sm.unsafe_store(b, Int32(found))


def vgpu_special_step_kernel(sm: I32P, sp_len: I64P, n: Int64, step: I32P):
    """One thread per position p in [0, n]: the host split's step from p (past the matching token,
    or one byte on); n at n."""
    var p = _gtid()
    var m = Int(n)
    if p > m:
        return
    if p == m:
        step.unsafe_store(p, Int32(m))
        return
    var r = Int(sm[p])
    step.unsafe_store(p, Int32(p + Int(sp_len[r])) if r >= 0 else Int32(p + 1))


def vgpu_special_chosen_kernel(sm: I32P, on: I32P, sp_len: I64P, n: Int64, spk: I32P, bflag: I32P):
    """One thread per position p in [0, n]: spk[p] = the token chosen at p (the chain stands on a
    match there) or -1; the segment boundaries (0, n, every chosen token's start and end) are
    marked 1 in `bflag` (zeroed by the caller; only 1s are stored, so overlapping marks agree)."""
    var b = _gtid()
    var m = Int(n)
    if b > m:
        return
    if b == 0 or b == m:
        bflag.unsafe_store(b, Int32(1))
    if b == m:
        spk.unsafe_store(b, Int32(-1))
        return
    var r = Int(sm[b])
    if on[b] != 0 and r >= 0:
        spk.unsafe_store(b, Int32(r))
        bflag.unsafe_store(b, Int32(1))
        bflag.unsafe_store(b + Int(sp_len[r]), Int32(1))
    else:
        spk.unsafe_store(b, Int32(-1))


def vgpu_special_tflag_kernel(spk: I32P, sp_len: I64P, n: Int64, tflag: U8P):
    """One thread per byte (GPT-2): a chosen token's segment is one piece of flag 2."""
    var b = _gtid()
    if b >= Int(n):
        return
    var r = Int(spk[b])
    if r < 0:
        return
    tflag.unsafe_store(b, UInt8(2))
    for j in range(1, Int(sp_len[r])):
        tflag.unsafe_store(b + j, UInt8(0))


def vgpu_llama_tflag_kernel(on: I32P, spk: I32P, n: Int64, tflag: U8P):
    """One thread per byte (Llama 3 / Qwen 2): 2 at a chosen token, 1 at every other piece start of
    the chain, 0 elsewhere."""
    var b = _gtid()
    if b >= Int(n):
        return
    if spk[b] >= 0:
        tflag.unsafe_store(b, UInt8(2))
    else:
        tflag.unsafe_store(b, UInt8(1) if on[b] != 0 else UInt8(0))


def vgpu_special_id_kernel(spk: I32P, sp_id: I64P, n: Int64, outi: I32P):
    """One thread per byte: a chosen token's piece (one id, written by the merge kernel as a
    placeholder) becomes its id."""
    var b = _gtid()
    if b >= Int(n):
        return
    var r = Int(spk[b])
    if r >= 0:
        outi.unsafe_store(b, Int32(sp_id[r]))


# ------------------------------------------------------------------ decode kernels


@always_inline
def _added(added_of: I64P, nadded: Int, id: Int) -> Int:
    if id >= 0 and id < nadded:
        return Int(added_of[id])
    return -1


def vgpu_decode_len_kernel(
    ids: I64P, nids: Int64, tlen: I64P, ntok: Int64, added_of: I64P, nadded: Int64, sp_len: I64P,
    lens: I32P, st: I32P,
):
    """One thread per id: its byte length (an added token's content first, then a rank's bytes);
    an id that is neither records its position (the first wins, an atomic min)."""
    var i = _gtid()
    if i >= Int(nids):
        return
    var id = Int(ids[i])
    var r = _added(added_of, Int(nadded), id)
    var m = 0
    if r >= 0:
        m = Int(sp_len[r])
    elif id >= 0 and id < Int(ntok):
        m = Int(tlen[id])
    else:
        _ = Atomic.min(st + G_BAD, Int32(i))
    lens.unsafe_store(i, Int32(m))


def vgpu_decode_copy_kernel(
    ids: I64P, nids: Int64, arena: U8P, toff: I64P, tlen: I64P, ntok: Int64, added_of: I64P, nadded: Int64,
    sp_arena: U8P, sp_off: I64P, sp_len: I64P, pfx: I32P, out_bytes: U8P,
):
    """One thread per id: its bytes at its scanned offset."""
    var i = _gtid()
    if i >= Int(nids):
        return
    var id = Int(ids[i])
    var at = Int(pfx[i])
    var r = _added(added_of, Int(nadded), id)
    if r >= 0:
        var o = Int(sp_off[r])
        for k in range(Int(sp_len[r])):
            out_bytes.unsafe_store(at + k, sp_arena[o + k])
    elif id >= 0 and id < Int(ntok):
        var o = Int(toff[id])
        for k in range(Int(tlen[id])):
            out_bytes.unsafe_store(at + k, arena[o + k])


# ------------------------------------------------------------------ orchestration


def vocab_gpu_encode_device(
    ctx: DeviceContext, t: VocabGpuTable, text: U8P, n: Int, allow: Bool, packed: I32P, st: I32P,
) raises:
    """`VocabTokenizer.encode` on the `n` device bytes at `text`: the ids back to back at `packed`
    (room for n), their count at st[S_TOTAL], a missing rank at st[S_FAIL] (st holds S_LEN words,
    zeroed by the caller). Every step runs on the device."""
    if n == 0:
        return
    var nblk = _blocks(n)
    var nblk1 = _blocks(n + 1)
    var has_sp = allow and t.nsp > 0
    # Segment offsets: [0, n] without added tokens; with them, every boundary packed in order and
    # the tail filled with n (n + 1 segments at most, the trailing ones empty at n).
    var d_offs = ctx.enqueue_create_buffer[DType.int64](n + 2)
    var spk = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var ndocs = 1
    var offs_host: List[Int64] = [Int64(0), Int64(n)]
    if has_sp:
        var sm = ctx.enqueue_create_buffer[DType.int32](n + 1)
        var step = ctx.enqueue_create_buffer[DType.int32](n + 1)
        var on0 = ctx.enqueue_create_buffer[DType.int32](n + 1)
        var bflag = ctx.enqueue_create_buffer[DType.int32](n + 1)
        var bpfx = ctx.enqueue_create_buffer[DType.int32](n + 2)
        ctx.enqueue_function[vgpu_special_match_kernel](
            text, Int64(n), _p8(t.sp_arena), _p64(t.sp_off), _p64(t.sp_len), Int64(t.nsp), _p32(sm),
            grid_dim=nblk1, block_dim=BPE_TPB,
        )
        ctx.enqueue_function[vgpu_special_step_kernel](_p32(sm), _p64(t.sp_len), Int64(n), _p32(step),
                                                       grid_dim=nblk1, block_dim=BPE_TPB)
        cut_chain_marks(ctx, _p32(step), on0, n + 1)
        bflag.enqueue_fill(Int32(0))
        ctx.enqueue_function[vgpu_special_chosen_kernel](
            _p32(sm), _p32(on0), _p64(t.sp_len), Int64(n), _p32(spk), _p32(bflag), grid_dim=nblk1, block_dim=BPE_TPB,
        )
        _scan(ctx, _p32(bflag), _p32(bpfx), st, n + 1)
        d_offs.enqueue_fill(Int64(n))
        ctx.enqueue_function[gpu_cut_pack_kernel](_p32(bflag), _p32(bpfx), Int64(n + 1), _p64(d_offs),
                                                  grid_dim=nblk1, block_dim=BPE_TPB)
        ndocs = n + 1
        ctx.synchronize()
        _ = sm^
        _ = step^
        _ = on0^
        _ = bflag^
        _ = bpfx^
    else:
        spk.enqueue_fill(Int32(-1))
        _up64(ctx, d_offs, Int(offs_host.unsafe_ptr()), 2)
    var mflag = ctx.enqueue_create_buffer[DType.uint8](n)
    var ustart = ctx.enqueue_create_buffer[DType.uint8](n)
    var ulen = ctx.enqueue_create_buffer[DType.uint8](n)
    var ucls = ctx.enqueue_create_buffer[DType.uint8](n)
    var tflag = ctx.enqueue_create_buffer[DType.uint8](n)
    var on = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var bnd = ctx.enqueue_create_buffer[DType.int32](2 * n + 2)
    var outi = ctx.enqueue_create_buffer[DType.int32](n)
    var cnt = ctx.enqueue_create_buffer[DType.int32](n)
    var pfx = ctx.enqueue_create_buffer[DType.int32](n + 1)
    mflag.enqueue_fill(UInt8(0))
    # The units of every segment, each bounded by its own end (no `<|endoftext|>` literal: allow 0).
    ctx.enqueue_function[gpu_unit_kernel](
        text, _p64(d_offs), Int64(ndocs), Int64(n), _p8(mflag), Int64(0), Int64(0),
        _p64(t.classes.letters), Int64(t.classes.nl), _p64(t.classes.numbers), Int64(t.classes.nn),
        _p64(t.classes.spaces), Int64(t.classes.ns), _p8(ustart), _p8(ulen), _p8(ucls),
        grid_dim=nblk, block_dim=BPE_TPB,
    )
    if t.pattern == PATTERN_GPT2:
        ctx.enqueue_function[gpu_start_kernel](
            text, _p64(d_offs), Int64(ndocs), Int64(n), _p8(mflag), Int64(0), _p8(ustart), _p8(ulen), _p8(ucls),
            _p8(tflag), grid_dim=nblk, block_dim=BPE_TPB,
        )
        if has_sp:
            ctx.enqueue_function[vgpu_special_tflag_kernel](_p32(spk), _p64(t.sp_len), Int64(n), _p8(tflag),
                                                            grid_dim=nblk, block_dim=BPE_TPB)
    else:
        cut_llama_marks(ctx, text, _p64(d_offs), ndocs, n, 3 if t.pattern == PATTERN_LLAMA3 else 1, _p8(ustart),
                        _p8(ulen), _p8(ucls), _p32(spk), _p64(t.sp_len), has_sp, on, st)
        ctx.enqueue_function[vgpu_llama_tflag_kernel](_p32(on), _p32(spk), Int64(n), _p8(tflag),
                                                      grid_dim=nblk, block_dim=BPE_TPB)
    ctx.enqueue_function[gpu_merge_kernel](
        text, _p64(d_offs), Int64(ndocs), Int64(n), _p8(tflag), _p8(t.arena), _p64(t.toff), _p64(t.tlen),
        _p64(t.buckets), Int64(t.mask), Int64(-1), _p32(bnd), _p32(outi), _p32(cnt), st,
        grid_dim=nblk, block_dim=BPE_TPB,
    )
    if has_sp:
        ctx.enqueue_function[vgpu_special_id_kernel](_p32(spk), _p64(t.sp_id), Int64(n), _p32(outi),
                                                     grid_dim=nblk, block_dim=BPE_TPB)
    _scan(ctx, _p32(cnt), _p32(pfx), st, n)
    ctx.enqueue_function[gpu_pack_kernel](_p32(cnt), _p32(pfx), _p32(outi), packed, Int64(n),
                                          grid_dim=nblk, block_dim=BPE_TPB)
    ctx.synchronize()
    _ = len(offs_host)
    _ = d_offs^
    _ = spk^
    _ = mflag^
    _ = ustart^
    _ = ulen^
    _ = ucls^
    _ = tflag^
    _ = on^
    _ = bnd^
    _ = outi^
    _ = cnt^
    _ = pfx^


comptime VocabGpuPtr = MutPointer[VocabGpuTable, MutUntrackedOrigin]


def _table(handle: Int) raises -> VocabGpuPtr:
    if handle == 0:
        raise Error("tokenizer (gpu): the vocabulary is not loaded on the device")
    return VocabGpuPtr(unsafe_from_address=handle)


def vocab_gpu_load(a: MutPointer[Int, MutUntrackedOrigin], p: MutPointer[Int, MutUntrackedOrigin]) raises -> Int:
    """TOK_GPU_OP_VOCAB_LOAD: the tables uploaded once; the handle word (this image's heap address
    of the `VocabGpuTable`) is the result."""
    var ctx = bpe_gpu_ctx()
    var table = VocabGpuTable(ctx, a[0], p[0], a[1], a[2], p[1], a[3], p[2], p[3], a[4], p[4], a[5], a[6],
                              a[7], p[5], a[8], p[6], p[7])
    var slot = alloc[VocabGpuTable](1)
    slot.unsafe_write(table^)
    return Int(slot)


def vocab_gpu_free(handle: Int) raises:
    """TOK_GPU_OP_VOCAB_FREE: the device buffers and the table released."""
    var table = _table(handle)
    table.unsafe_deinit_pointee()
    table.free()


def vocab_gpu_encode(handle: Int, text_addr: Int, n: Int, allow: Bool, ids_addr: Int) raises -> Int:
    """TOK_GPU_OP_VOCAB_ENCODE on host text: uploaded once, `vocab_gpu_encode_device`, the ids
    (int32, room for max(n, 1) at `ids_addr`) and the state words read back once. Returns the id
    count; a missing rank is refused in the host's words."""
    var table = _table(handle)
    if n == 0:
        return 0
    var ctx = bpe_gpu_ctx()
    var d_text = ctx.enqueue_create_buffer[DType.uint8](n)
    var d_packed = ctx.enqueue_create_buffer[DType.int32](n)
    var d_st = ctx.enqueue_create_buffer[DType.int32](S_LEN)
    _up8(ctx, d_text, text_addr, n)
    d_st.enqueue_fill(Int32(0))
    vocab_gpu_encode_device(ctx, table[], _p8(d_text), n, allow, _p32(d_packed), _p32(d_st))
    var h_st = List[Int32](length=S_LEN, fill=Int32(0))
    ctx.enqueue_copy(dst_ptr=I32P(unsafe_from_address=Int(h_st.unsafe_ptr())), src_buf=d_st)
    ctx.enqueue_copy(dst_ptr=I32P(unsafe_from_address=ids_addr), src_buf=d_packed)
    ctx.synchronize()
    _ = d_text^
    _ = d_packed^
    _ = d_st^
    if h_st[S_FAIL] != Int32(0):
        raise Error("bpe: a piece has no rank; the table is missing a single-byte token")
    var total = Int(h_st[S_TOTAL])
    if total < 0 or total > n:
        raise Error("tokenizer (gpu): the device encode returned " + String(total) + " ids for " + String(n) + " bytes")
    return total


def vocab_gpu_decode(handle: Int, ids_addr: Int, nids: Int, out_addr: Int, cap: Int) raises -> Int:
    """TOK_GPU_OP_VOCAB_DECODE: the ids (int64) uploaded once, the gather on the device, the bytes
    (room `cap` at `out_addr`) read back once. Returns the byte count, or -1 - p for the first id at
    position p that is neither an added token nor a rank (nothing is written then)."""
    var table = _table(handle)
    if nids == 0:
        return 0
    var ctx = bpe_gpu_ctx()
    var nblk = _blocks(nids)
    var d_ids = ctx.enqueue_create_buffer[DType.int64](nids)
    var lens = ctx.enqueue_create_buffer[DType.int32](nids)
    var pfx = ctx.enqueue_create_buffer[DType.int32](nids + 1)
    var d_out = ctx.enqueue_create_buffer[DType.uint8](max(cap, 1))
    var d_st = ctx.enqueue_create_buffer[DType.int32](S_LEN)
    ctx.enqueue_copy(dst_buf=d_ids, src_ptr=I64P(unsafe_from_address=ids_addr))
    d_st.enqueue_fill(Int32(0))
    ctx.enqueue_function[gpu_state_init_kernel](_p32(d_st), grid_dim=1, block_dim=1)  # small-launch(one state word)
    ref t = table[]
    ctx.enqueue_function[vgpu_decode_len_kernel](
        _p64(d_ids), Int64(nids), _p64(t.tlen), Int64(t.ntok), _p64(t.added_of), Int64(t.nadded), _p64(t.sp_len),
        _p32(lens), _p32(d_st), grid_dim=nblk, block_dim=BPE_TPB,
    )
    _scan(ctx, _p32(lens), _p32(pfx), _p32(d_st), nids)
    ctx.enqueue_function[vgpu_decode_copy_kernel](
        _p64(d_ids), Int64(nids), _p8(t.arena), _p64(t.toff), _p64(t.tlen), Int64(t.ntok), _p64(t.added_of),
        Int64(t.nadded), _p8(t.sp_arena), _p64(t.sp_off), _p64(t.sp_len), _p32(pfx), _p8(d_out),
        grid_dim=nblk, block_dim=BPE_TPB,
    )
    var h_st = List[Int32](length=S_LEN, fill=Int32(0))
    ctx.enqueue_copy(dst_ptr=I32P(unsafe_from_address=Int(h_st.unsafe_ptr())), src_buf=d_st)
    ctx.synchronize()
    var result: Int
    if h_st[G_BAD] != G_NONE:
        result = -1 - Int(h_st[G_BAD])
    else:
        var total = Int(h_st[S_TOTAL])
        if total < 0 or total > cap:
            raise Error("tokenizer (gpu): " + String(total) + " decoded bytes do not fit " + String(cap))
        if total > 0:
            ctx.enqueue_copy(dst_ptr=U8P(unsafe_from_address=out_addr), src_buf=d_out.create_sub_buffer[DType.uint8](0, total))
            ctx.synchronize()
        result = total
    _ = d_ids^
    _ = lens^
    _ = pfx^
    _ = d_out^
    _ = d_st^
    return result

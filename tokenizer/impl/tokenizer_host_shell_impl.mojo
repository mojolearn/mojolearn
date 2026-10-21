# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed tokenizer work behind the tokenizer host shell
(bindings/_mojolearn_tokenizer_host.mojo).

Every loop over documents, ids, bytes or tokens, every host_parallelize
region and every check over the caller's data lives here, over native
pointers, lists and handles. The shell only unpacks Python objects, calls
these functions and converts the typed results back. Moved word for word
from the shell (thin-neural, 2026-10-21): same operations, same order.
"""
from std.memory import memcpy
from std.sys.compile import is_defined
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_chunk, host_predict_task_count

from core.hostptr import i32_ptr
from autograd.model_verify import sha256_hex
from tokenizer.encoding import (
    GPT2_ENDOFTEXT,
    BpeTokenizer,
    bytes_string,
    load_bpe_tokenizer_from,
    string_bytes,
)
from tokenizer.impl.ranks import RankTable
from tokenizer.impl.unicode_class import builtin_unicode_classes
from tokenizer.impl.vocab_build import (
    MergeCheck,
    check_loose,
    check_merge,
    split_merge_line,
    table_from_tokens,
    unspell,
)
from tokenizer.vocab import VocabTokenizer
from tokenizer.train.bpe_train import TrainedVocabulary, train_bpe

comptime TOKENIZER_HOST_SABOTAGE = is_defined["MOJOLEARN_TOKENIZER_HOST_SABOTAGE"]()
comptime TOKENIZER_BATCH_SABOTAGE = is_defined["MOJOLEARN_TOKENIZER_BATCH_SABOTAGE"]()


struct BpeHandle(Movable, Writable):
    """Python-owned tokenizer lifetime: the parsed rank table and Unicode
    classes, loaded once by `bpe_load` and kept for the handle's life. No
    global slot and no retained host pointer."""

    var tok: Optional[BpeTokenizer]
    var max_token_bytes: Int

    def __init__(out self):
        self.tok = Optional[BpeTokenizer]()
        self.max_token_bytes = 0

    # Both spelled out: `add_type` derives whichever is missing by
    # reflection over the fields, and `Optional[BpeTokenizer]` is not
    # Writable (the byte LM session does the same).
    def write_to(self, mut writer: Some[Writer]):
        writer.write("_BpeHandle(loaded=", Bool(self.tok), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("_BpeHandle(loaded=", Bool(self.tok), ")")


struct BpeTrainedHandle(Movable, Writable):
    """Python-owned result of one `bpe_train` call: the trained vocabulary,
    held until `bpe_trained_copy` has copied it into the caller's buffers."""

    var vocab: Optional[TrainedVocabulary]

    def __init__(out self):
        self.vocab = Optional[TrainedVocabulary]()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("_BpeTrainedHandle(trained=", Bool(self.vocab), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("_BpeTrainedHandle(trained=", Bool(self.vocab), ")")


struct VocabHandle(Movable, Writable):
    """Python-owned `mojolearn.models.Tokenizer` lifetime: the rank table,
    the pattern and the added tokens (`tokenizer/vocab.mojo`)."""

    var tok: Optional[VocabTokenizer]

    def __init__(out self):
        self.tok = Optional[VocabTokenizer]()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("_VocabHandle(loaded=", Bool(self.tok), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("_VocabHandle(loaded=", Bool(self.tok), ")")


def _u8_ptr(addr: Int) raises -> MutPointer[UInt8, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null uint8 buffer address")
    return MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr)


def _i64_ptr(addr: Int) raises -> MutPointer[Int64, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int64 buffer address")
    return MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=addr)


def tokenizer_store_i64(dst: MutPointer[Int64, MutUntrackedOrigin], values: List[Int]):
    """`values` as int64 at `dst`, in order."""
    for k in range(len(values)):
        dst[k] = Int64(values[k])


def tokenizer_store_i32(dst: MutPointer[Int32, MutUntrackedOrigin], values: List[Int]):
    """`values` as int32 at `dst`, in order."""
    for k in range(len(values)):
        dst[k] = Int32(values[k])


def tokenizer_store_u8(dst: MutPointer[UInt8, MutUntrackedOrigin], values: List[UInt8]):
    """`values` at `dst` (one memcpy)."""
    if len(values) > 0:
        memcpy(dest=dst, src=values.unsafe_ptr(), count=len(values))


def tokenizer_append_bytes(mut arena: List[UInt8], bytes: List[UInt8]):
    """`bytes` appended to `arena`, byte by byte."""
    for j in range(len(bytes)):
        arena.append(bytes[j])


def _i64_list(addr: Int, n: Int) raises -> List[Int]:
    var out = List[Int](capacity=n)
    if n > 0:
        var p = _i64_ptr(addr)
        for k in range(n):
            out.append(Int(p[k]))
    return out^


def _u8_list(addr: Int, n: Int) raises -> List[UInt8]:
    var out = List[UInt8](length=n, fill=UInt8(0))
    if n > 0:
        memcpy(dest=out.unsafe_ptr(), src=_u8_ptr(addr), count=n)
    return out^


def _bytes_total(lengths: List[Int]) raises -> Int:
    var total = 0
    for k in range(len(lengths)):
        if lengths[k] < 0:
            raise Error("ValueError: length " + String(k) + " is negative")
        total += lengths[k]
    return total


def _handle_from_table(var table: RankTable) raises -> BpeHandle:
    var handle = BpeHandle()
    var longest = len(String(GPT2_ENDOFTEXT).as_bytes())
    for i in range(table.n_tokens()):
        if table.length[i] > longest:
            longest = table.length[i]
    handle.max_token_bytes = longest
    handle.tok = BpeTokenizer(table^, builtin_unicode_classes())
    return handle^


def _quoted(line: List[UInt8]) -> String:
    var s = String("'")
    s += bytes_string(line)
    s += "'"
    return s^


# ------------------------------------------------------------------------
# The BPE entries (bpe_load .. bpe_trained_copy).


def bpe_handle_load(mut handle: BpeHandle, rp: String) raises:
    """`bpe_load`'s work: parse the rank file, record the longest token."""
    var tok = load_bpe_tokenizer_from(rp)
    var longest = len(String(GPT2_ENDOFTEXT).as_bytes())
    for i in range(tok.ranks.n_tokens()):
        if tok.ranks.length[i] > longest:
            longest = tok.ranks.length[i]
    handle.max_token_bytes = longest
    handle.tok = tok^


def bpe_encode_into(
    handle: BpeHandle, text_address: Int, n: Int, out_address: Int, cap: Int, allow: Bool
) raises -> Int:
    """`bpe_encode`'s work: encode `n` bytes, write the ids as int32."""
    var src = _u8_ptr(text_address)
    var text = List[UInt8](length=n, fill=UInt8(0))
    memcpy(dest=text.unsafe_ptr(), src=src, count=n)
    # THE ONE CALL THAT COMPUTES ANYTHING.
    var ids = handle.tok.value().encode_bytes(text, allow)
    var count = len(ids)
    if count > cap:
        raise Error(
            "bpe_encode: "
            + String(count)
            + " ids do not fit an output of "
            + String(cap)
            + "; nothing written"
        )
    var dst = i32_ptr(out_address)
    for k in range(count):
        comptime if TOKENIZER_HOST_SABOTAGE:
            dst[k] = Int32(ids[count - 1 - k])
        else:
            dst[k] = Int32(ids[k])
    return count


def bpe_encode_batch_into(
    handle: BpeHandle,
    text_address: Int,
    offsets_address: Int,
    out_address: Int,
    counts_address: Int,
    n_docs: Int,
    n: Int,
    cap: Int,
    allow: Bool,
) raises -> Int:
    """`bpe_encode_batch`'s work: each document encoded ALONE, the ids back
    to back as int32, each document's count as int64. Returns the total."""
    var owner = UnsafePointer(to=handle)
    # `offsets_addr` holds the n_docs document LENGTHS (lane/pyglue-
    # text-io): the running offsets are taken here, not in Python.
    var lens = _i64_ptr(offsets_address)
    var offsets_list = List[Int64](length=n_docs + 1, fill=Int64(0))
    for k in range(n_docs):
        if Int(lens[k]) < 0:
            raise Error("bpe_encode_batch: document " + String(k) + " has a negative length")
        offsets_list[k + 1] = offsets_list[k] + lens[k]
    var offs = offsets_list.unsafe_ptr()
    if Int(offs[0]) != 0 or Int(offs[n_docs]) != n:
        raise Error(
            "bpe_encode_batch: document lengths must sum to n_bytes "
            + String(n)
            + ", got "
            + String(Int(offs[0]))
            + " and "
            + String(Int(offs[n_docs]))
        )
    for k in range(n_docs):
        if Int(offs[k + 1]) < Int(offs[k]):
            raise Error(
                "bpe_encode_batch: offsets decrease at document "
                + String(k)
            )
    var all_ids = List[Int](capacity=n)
    var counts = List[Int](length=n_docs, fill=0)
    # A document produces at most one id per input byte.  Give each one
    # its byte-offset-sized temporary range, so independent encodes can
    # run concurrently without a shared append or a prefix scan first.
    var staged = List[Int32](length=n, fill=Int32(0))
    var failed = List[Int](length=n_docs, fill=0)
    var countp = counts.unsafe_ptr()
    var stagep = staged.unsafe_ptr()
    var failp = failed.unsafe_ptr()
    var tasks = host_predict_task_count(n_docs) if n >= 16384 else 1
    var chunk = host_predict_chunk(n_docs, tasks)
    def _documents(task: Int) {imm owner, imm offs, imm text_address, imm allow, imm countp, imm stagep, imm failp, imm chunk, imm n_docs}:
        var lo = task * chunk
        var hi = min(lo + chunk, n_docs)
        for k in range(lo, hi):
            var a = Int(offs[k])
            var m = Int(offs[k + 1]) - a
            var doc = List[UInt8](length=m, fill=UInt8(0))
            try:
                if m > 0:
                    var src = _u8_ptr(text_address + a)
                    memcpy(dest=doc.unsafe_ptr(), src=src, count=m)
                var ids = owner[].tok.value().encode_bytes(doc, allow)
                var c = len(ids)
                countp.unsafe_store(k, c)
                for j in range(c):
                    comptime if TOKENIZER_HOST_SABOTAGE:
                        stagep.unsafe_store(a + j, Int32(ids[c - 1 - j]))
                    else:
                        stagep.unsafe_store(a + j, Int32(ids[j]))
            except:
                failp.unsafe_store(k, 1)
    if tasks == 1:
        _documents(0)
    else:
        host_parallelize(_documents, tasks)
    for k in range(n_docs):
        if failed[k] != 0:
            raise Error("bpe_encode_batch: internal document encode failed")
        var a = Int(offs[k])
        for j in range(counts[k]):
            all_ids.append(Int(staged[a + j]))
    comptime if TOKENIZER_BATCH_SABOTAGE:
        # The batch part's negative control: swap the last id of each
        # document with the first id of the next non-empty one, which
        # a batch of one document can never reach.
        var start = 0
        for k in range(n_docs - 1):
            var end = start + counts[k]
            if counts[k] > 0 and counts[k + 1] > 0:
                var t = all_ids[end - 1]
                all_ids[end - 1] = all_ids[end]
                all_ids[end] = t
            start = end
    var total = len(all_ids)
    if total > cap:
        raise Error(
            "bpe_encode_batch: "
            + String(total)
            + " ids do not fit an output of "
            + String(cap)
            + "; nothing written"
        )
    var cnt = _i64_ptr(counts_address)
    for k in range(n_docs):
        cnt[k] = Int64(counts[k])
    if total > 0:
        var dst = i32_ptr(out_address)
        for j in range(total):
            dst[j] = Int32(all_ids[j])
    return total


def bpe_decode_into(
    handle: BpeHandle, ids_address: Int, n: Int, out_address: Int, cap: Int
) raises -> Int:
    """`bpe_decode`'s work: read `n` int32 ids, write the bytes."""
    var src = i32_ptr(ids_address)
    var ids = List[Int](capacity=n)
    for k in range(n):
        ids.append(Int(src[k]))
    # THE ONE CALL THAT COMPUTES ANYTHING; an id outside [0, n_vocab) is
    # refused here by name and position, nothing written.
    var out = handle.tok.value().decode_bytes(ids)
    var count = len(out)
    if count > cap:
        raise Error(
            "bpe_decode: "
            + String(count)
            + " bytes do not fit an output of "
            + String(cap)
            + "; nothing written"
        )
    var dst = _u8_ptr(out_address)
    if count > 0:
        memcpy(dest=dst, src=out.unsafe_ptr(), count=count)
    return count


def bpe_train_fill(
    mut handle: BpeTrainedHandle,
    text_address: Int,
    offsets_address: Int,
    n_docs: Int,
    n: Int,
    vocab_size: Int,
    min_frequency: Int,
    reverse: Bool,
) raises:
    """`bpe_train`'s work: cut the documents by their lengths, train."""
    var lens = _i64_ptr(offsets_address)
    var offsets_list = List[Int64](length=n_docs + 1, fill=Int64(0))
    for k in range(n_docs):
        if Int(lens[k]) < 0:
            raise Error("bpe_train: document " + String(k) + " has a negative length")
        offsets_list[k + 1] = offsets_list[k] + lens[k]
    var offs = offsets_list.unsafe_ptr()
    if Int(offs[0]) != 0 or Int(offs[n_docs]) != n:
        raise Error(
            "bpe_train: document lengths must sum to n_bytes "
            + String(n)
            + ", got "
            + String(Int(offs[0]))
            + " and "
            + String(Int(offs[n_docs]))
        )
    for k in range(n_docs):
        if Int(offs[k + 1]) < Int(offs[k]):
            raise Error(
                "bpe_train: offsets decrease at document " + String(k)
            )
    var documents = List[List[UInt8]](capacity=n_docs)
    for k in range(n_docs):
        var a = Int(offs[k])
        var m = Int(offs[k + 1]) - a
        var doc = List[UInt8](length=m, fill=UInt8(0))
        if m > 0:
            memcpy(
                dest=doc.unsafe_ptr(), src=_u8_ptr(text_address + a), count=m
            )
        documents.append(doc^)
    var classes = builtin_unicode_classes()
    # THE ONE CALL THAT COMPUTES ANYTHING.
    handle.vocab = train_bpe(
        documents, classes, vocab_size, min_frequency, reverse
    )


def bpe_trained_copy_into(
    v: TrainedVocabulary,
    arena_address: Int,
    lengths_address: Int,
    nm: Int,
    left_address: Int,
    right_address: Int,
) raises:
    """`bpe_trained_copy`'s work: token bytes in rank order, lengths, merges."""
    var arena = _u8_ptr(arena_address)
    var lengths = _i64_ptr(lengths_address)
    # The arena is written in RANK order from each token's own offset, so
    # the caller's slicing by cumulative length is correct whatever order
    # the trainer's arena was filled in.
    var at = 0
    for id in range(v.n_tokens()):
        var m = v.length[id]
        var src = v.offset[id]
        for i in range(m):
            arena[at + i] = v.arena[src + i]
        at += m
        lengths[id] = Int64(m)
    if nm > 0:
        var left = _i64_ptr(left_address)
        var right = _i64_ptr(right_address)
        for k in range(nm):
            left[k] = Int64(v.merge_left[k])
            right[k] = Int64(v.merge_right[k])


# ------------------------------------------------------------------------
# The spelled vocabulary (`_table_from_spelled`). The shell converts each
# Python item into the typed record below; the checks run here in the
# shell's former order and report the first refusal by item and code, so
# the shell can name it with the item's Python repr.

comptime SPELLED_OK = 0
comptime SPELLED_BAD_ID = 1        # id not a unique int in [0, n)
comptime SPELLED_NOT_STR = 2       # the spelling (or merge part) is not a str
comptime SPELLED_UNSPELL = 3       # `unspell` refused codepoint `bad`
comptime SPELLED_RAISED = 4        # the Python conversion raised; message kept
comptime SPELLED_NOT_PAIR = 5      # a merge item that is not a pair
comptime SPELLED_NOT_LEVEL = 6     # a "a b" merge string not byte-level


@fieldwise_init
struct SpelledFailure(Copyable, Movable):
    """The first refusal: item `k`, `code`, merge `part` (0 or 1) and the
    `bad` codepoint position for SPELLED_UNSPELL."""

    var k: Int
    var code: Int
    var part: Int
    var bad: Int


struct SpelledVocab(Movable):
    """`{spelling: id}` items as the shell read them."""

    var id_is_int: List[Bool]
    var ids: List[Int]
    var key_kind: List[Int]          # 0 str, 1 not str, 2 conversion raised
    var keys: List[List[UInt8]]

    def __init__(out self, n: Int):
        self.id_is_int = List[Bool](capacity=n)
        self.ids = List[Int](capacity=n)
        self.key_kind = List[Int](capacity=n)
        self.keys = List[List[UInt8]](capacity=n)


struct SpelledMerges(Movable):
    """A merge list as the shell read it: per item its `kind` (0 str, 1 a
    list/tuple of two, 2 other), the str's bytes (`kind` 0, `str_kind` 0
    or 2 when the conversion raised) and each part's kind and bytes."""

    var kind: List[Int]
    var str_kind: List[Int]
    var text: List[List[UInt8]]
    var part_kind: List[Int]         # 2 per item: 0 str, 1 not str, 2 raised
    var parts: List[List[UInt8]]     # 2 per item

    def __init__(out self, m: Int):
        self.kind = List[Int](capacity=m)
        self.str_kind = List[Int](capacity=m)
        self.text = List[List[UInt8]](capacity=m)
        self.part_kind = List[Int](capacity=2 * m)
        self.parts = List[List[UInt8]](capacity=2 * m)


def spelled_vocab_tokens(
    v: SpelledVocab, inv: List[Int], mut arena: List[UInt8], mut lengths: List[Int]
) raises -> SpelledFailure:
    """Place every spelling at its id and lay the tokens out in rank
    order. Returns the first refusal (code SPELLED_OK when none)."""
    var n = len(v.ids)
    var toks = List[List[UInt8]](capacity=n)
    var placed = List[Bool](length=n, fill=False)
    for _ in range(n):
        toks.append(List[UInt8]())
    for k in range(n):
        var i = -1
        if v.id_is_int[k]:
            i = v.ids[k]
        if not v.id_is_int[k] or i < 0 or i >= n or placed[i]:
            return SpelledFailure(k, SPELLED_BAD_ID, 0, 0)
        if v.key_kind[k] == 1:
            return SpelledFailure(k, SPELLED_NOT_STR, 0, 0)
        if v.key_kind[k] == 2:
            return SpelledFailure(k, SPELLED_RAISED, 0, 0)
        var out = List[UInt8]()
        var bad = unspell(v.keys[k], inv, out)
        if bad != -1:
            return SpelledFailure(k, SPELLED_UNSPELL, 0, bad)
        toks[i] = out^
        placed[i] = True
    for i in range(n):
        lengths.append(len(toks[i]))
        for j in range(len(toks[i])):
            arena.append(toks[i][j])
    return SpelledFailure(-1, SPELLED_OK, 0, 0)


def spelled_merges_text(
    table: RankTable, text: List[UInt8], inv: List[Int], version_line: Bool, mut chk: MergeCheck
) raises:
    """The text of a `vocab.bpe` file: one `a b` per line, an optional
    leading `#version` line, blank lines skipped; every merge checked."""
    var lineno = 0
    var at = 0
    var total = len(text)
    while at <= total:
        var stop = at
        while stop < total and Int(text[stop]) != 0x0A:
            stop += 1
        lineno += 1
        var line = List[UInt8](capacity=stop - at)
        var blank = True
        for i in range(at, stop):
            line.append(text[i])
            var c = Int(text[i])
            if not (c == 0x20 or (c >= 0x09 and c <= 0x0D)):
                blank = False
        at = stop + 1
        if blank:
            continue
        if version_line and lineno == 1 and len(line) >= 8:
            var v = String("#version").as_bytes()
            var is_version = True
            for i in range(8):
                if line[i] != v[i]:
                    is_version = False
            if is_version:
                continue
        var label = "line " + String(lineno)
        var sa = List[UInt8]()
        var sb = List[UInt8]()
        var a = List[UInt8]()
        var b = List[UInt8]()
        if (
            not split_merge_line(line, sa, sb)
            or unspell(sa, inv, a) != -1
            or unspell(sb, inv, b) != -1
        ):
            raise Error(
                "ValueError: "
                + label
                + ": "
                + _quoted(line)
                + " is not a merge of two tokens"
            )
        check_merge(table, a, b, label, chk)


def _spelled_part(
    m: SpelledMerges, k: Int, p: Int, inv: List[Int], mut out: List[UInt8]
) raises -> SpelledFailure:
    var pk = m.part_kind[2 * k + p]
    if pk == 1:
        return SpelledFailure(k, SPELLED_NOT_STR, p, 0)
    if pk == 2:
        return SpelledFailure(k, SPELLED_RAISED, p, 0)
    var bad = unspell(m.parts[2 * k + p], inv, out)
    if bad != -1:
        return SpelledFailure(k, SPELLED_UNSPELL, p, bad)
    return SpelledFailure(-1, SPELLED_OK, 0, 0)


def spelled_merges_list(
    table: RankTable, m: SpelledMerges, inv: List[Int], mut chk: MergeCheck
) raises -> SpelledFailure:
    """A `tokenizer.json`'s `model.merges`: `"a b"` strings or `[a, b]`
    pairs, each checked in order. Returns the first refusal the shell must
    name by repr (code SPELLED_OK when none); `check_merge` raises its own."""
    for k in range(len(m.kind)):
        var label = "merge " + String(k)
        var a = List[UInt8]()
        var b = List[UInt8]()
        if m.kind[k] == 0:
            if m.str_kind[k] == 2:
                return SpelledFailure(k, SPELLED_RAISED, 0, 0)
            var sa = List[UInt8]()
            var sb = List[UInt8]()
            if (
                not split_merge_line(m.text[k], sa, sb)
            ):
                return SpelledFailure(k, SPELLED_NOT_PAIR, 0, 0)
            if unspell(sa, inv, a) != -1 or unspell(sb, inv, b) != -1:
                return SpelledFailure(k, SPELLED_NOT_LEVEL, 0, 0)
        elif m.kind[k] == 1:
            var f0 = _spelled_part(m, k, 0, inv, a)
            if f0.code != SPELLED_OK:
                return f0^
            var f1 = _spelled_part(m, k, 1, inv, b)
            if f1.code != SPELLED_OK:
                return f1^
        else:
            return SpelledFailure(k, SPELLED_NOT_PAIR, 0, 0)
        check_merge(table, a, b, label, chk)
    return SpelledFailure(-1, SPELLED_OK, 0, 0)


# ------------------------------------------------------------------------
# lane/pyglue-text-io entries.


def bpe_check_ids(ids: List[Int], nv: Int) raises:
    """Every id in [0, nv), refused by value and position."""
    for k in range(len(ids)):
        if ids[k] < 0 or ids[k] >= nv:
            raise Error(
                "ValueError: id " + String(ids[k]) + " at position " + String(k)
                + " is outside [0, " + String(nv) + ")"
            )


def bpe_decode_seq_bytes(handle: BpeHandle, ids: List[Int], cap: Int) raises -> List[UInt8]:
    """`bpe_decode_seq`'s decode and fit check."""
    var out = handle.tok.value().decode_bytes(ids)
    if len(out) > cap:
        raise Error("bpe_decode_seq: " + String(len(out)) + " bytes do not fit " + String(cap))
    return out^


def bpe_decode_batch_append(
    handle: BpeHandle, ids: List[Int], nv: Int, mut joined: List[UInt8], mut lens: List[Int]
) raises:
    """One sequence of `bpe_decode_batch`: checked, decoded, appended."""
    for k in range(len(ids)):
        if ids[k] < 0 or ids[k] >= nv:
            raise Error(
                "ValueError: id " + String(ids[k]) + " at position " + String(k)
                + " is outside [0, " + String(nv) + ")"
            )
    var out = handle.tok.value().decode_bytes(ids)
    lens.append(len(out))
    joined += out^


def bpe_decode_batch_fit(joined: List[UInt8], cap: Int) raises:
    if len(joined) > cap:
        raise Error("bpe_decode_batch: " + String(len(joined)) + " bytes do not fit " + String(cap))


def trained_vocab_settle(mut v: TrainedVocabulary, n: Int, nm: Int) raises:
    """`bpe_render`'s offsets from the lengths, then every merge id checked."""
    var at = 0
    for k in range(n):
        v.offset.append(at)
        at += v.length[k]
    for k in range(nm):
        if v.merge_left[k] < 0 or v.merge_left[k] >= n or v.merge_right[k] < 0 or v.merge_right[k] >= n:
            raise Error("ValueError: merge " + String(k) + " names an id outside [0, " + String(n) + ")")


def _cut_documents(
    data: MutPointer[UInt8, MutUntrackedOrigin], points: List[Int], lo: Int, hi: Int,
    document_bytes: Int, mut starts: List[Int], mut stops: List[Int]
):
    """`lm_corpus.documents`: `[lo, hi)` cut at every split point strictly
    inside it, each piece cut into documents of at most `document_bytes`
    bytes ending at the last 0x0A at or before the limit (or at the limit
    when the window holds none)."""
    var cuts = List[Int]()
    cuts.append(lo)
    for k in range(len(points)):
        if points[k] > lo and points[k] < hi:
            cuts.append(points[k])
    cuts.append(hi)
    for s in range(len(cuts) - 1):
        var a = cuts[s]
        var b = cuts[s + 1]
        var at = a
        while at < b:
            var stop = min(at + document_bytes, b)
            if stop < b:
                var cut = stop - 1
                while cut >= at and Int(data[cut]) != 0x0A:
                    cut -= 1
                if cut >= at and cut + 1 > at:
                    stop = cut + 1
            starts.append(at)
            stops.append(stop)
            at = stop


def bpe_encode_corpus_into(
    handle: BpeHandle,
    data_address: Int,
    points: List[Int],
    n: Int,
    n_points: Int,
    document_bytes: Int,
    cap: Int,
    ids_address: Int,
    bounds_address: Int,
    stats_address: Int,
) raises -> Int:
    """`bpe_encode_corpus`'s work: cut, encode each document ALONE, write
    ids, point bounds and stats. Returns the id count."""
    var owner = UnsafePointer(to=handle)
    var total = 0
    var starts = List[Int]()
    var stops = List[Int]()
    if n > 0:
        _cut_documents(_u8_ptr(data_address), points, 0, n, document_bytes, starts, stops)
    var n_docs = len(starts)
    var counts = List[Int](length=n_docs, fill=0)
    var staged = List[Int32](length=max(n, 1), fill=Int32(0))
    var failed = List[Int](length=n_docs, fill=0)
    var countp = counts.unsafe_ptr()
    var stagep = staged.unsafe_ptr()
    var failp = failed.unsafe_ptr()
    var startp = starts.unsafe_ptr()
    var stopp = stops.unsafe_ptr()
    var tasks = host_predict_task_count(n_docs) if n >= 16384 else 1
    var chunk = host_predict_chunk(n_docs, tasks) if n_docs > 0 else 1
    def _documents(task: Int) {imm owner, imm startp, imm stopp, imm data_address, imm countp, imm stagep, imm failp, imm chunk, imm n_docs}:
        var lo = task * chunk
        var hi = min(lo + chunk, n_docs)
        for k in range(lo, hi):
            var a = startp[k]
            var m = stopp[k] - a
            var doc = List[UInt8](length=m, fill=UInt8(0))
            try:
                if m > 0:
                    memcpy(dest=doc.unsafe_ptr(), src=_u8_ptr(data_address + a), count=m)
                var ids = owner[].tok.value().encode_bytes(doc, False)
                var c = len(ids)
                countp.unsafe_store(k, c)
                for j in range(c):
                    stagep.unsafe_store(a + j, Int32(ids[j]))
            except:
                failp.unsafe_store(k, 1)
    if n_docs > 0:
        if tasks == 1:
            _documents(0)
        else:
            host_parallelize(_documents, tasks)
    for k in range(n_docs):
        if failed[k] != 0:
            raise Error("bpe_encode_corpus: internal document encode failed")
        total += counts[k]
    if total > cap:
        raise Error(
            "bpe_encode_corpus: " + String(total) + " ids do not fit an output of "
            + String(cap) + "; nothing written"
        )
    var bounds = _i64_ptr(bounds_address)
    var at = 0
    var next_point = 0
    for k in range(n_docs):
        while next_point < n_points and points[next_point] < starts[k]:
            next_point += 1
        if next_point < n_points and points[next_point] == starts[k]:
            bounds[next_point] = Int64(at)
            next_point += 1
        at += counts[k]
    var max_id = -1
    var above = 0
    if total > 0:
        var dst = i32_ptr(ids_address)
        at = 0
        for k in range(n_docs):
            var a = starts[k]
            for j in range(counts[k]):
                var v = Int(staged[a + j])
                dst[at + j] = Int32(v)
                if v > max_id:
                    max_id = v
                if v > 255:
                    above += 1
            at += counts[k]
    for j in range(n_points):
        if points[j] >= n:
            bounds[j] = Int64(total)
    var stats = _i64_ptr(stats_address)
    stats[0] = Int64(n_docs)
    stats[1] = Int64(max_id)
    stats[2] = Int64(above)
    return total


def bpe_train_corpus_fill(
    mut handle: BpeTrainedHandle,
    data_address: Int,
    points: List[Int],
    lo: Int,
    hi: Int,
    document_bytes: Int,
    vocab_size: Int,
    min_frequency: Int,
    reverse: Bool,
) raises:
    """`bpe_train_corpus`'s work: the documents of `[lo, hi)`, trained."""
    var starts = List[Int]()
    var stops = List[Int]()
    if hi > lo:
        _cut_documents(_u8_ptr(data_address), points, lo, hi, document_bytes, starts, stops)
    if len(starts) == 0:
        raise Error("ValueError: train needs at least one document")
    var documents = List[List[UInt8]](capacity=len(starts))
    for k in range(len(starts)):
        documents.append(_u8_list(data_address + starts[k], stops[k] - starts[k]))
    var classes = builtin_unicode_classes()
    handle.vocab = train_bpe(documents, classes, vocab_size, min_frequency, reverse)


def tokens_gather_rows_width(
    n_src: Int, batch: Int, length: Int, lo: Int, modulus: Int, base: Int
) raises -> Int:
    """`tokens_gather_rows`'s dims checked; returns the row width."""
    if batch < 0 or length < 1 or modulus < 1 or lo < 0 or base < 0:
        raise Error("ValueError: tokens_gather_rows: bad dims")
    var width = length + 1
    if lo + modulus - 1 + width > n_src:
        raise Error("ValueError: tokens_gather_rows: a row runs past the id array")
    return width


def tokens_gather_rows_into(
    src: MutPointer[Int32, MutUntrackedOrigin],
    dst: MutPointer[Int32, MutUntrackedOrigin],
    batch: Int, length: Int, width: Int, lo: Int, modulus: Int, base: Int,
) -> Int:
    """Row b reads `[lo + (base + b*length) % modulus : + width]`."""
    for b in range(batch):
        var start = lo + (base + b * length) % modulus
        memcpy(dest=dst + b * width, src=src + start, count=width)
    return batch * width


def vocab_check_merges(
    table: RankTable, marena: List[UInt8], mlen: List[Int], n: Int, nm: Int, ignore: Bool
) raises:
    """`vocab_load_bytes`'s merges: two byte strings per merge, checked."""
    var chk = MergeCheck(n)
    var at = 0
    for k in range(nm):
        var a = List[UInt8]()
        var b = List[UInt8]()
        for i in range(mlen[2 * k]):
            a.append(marena[at + i])
        at += mlen[2 * k]
        for i in range(mlen[2 * k + 1]):
            b.append(marena[at + i])
        at += mlen[2 * k + 1]
        check_merge(table, a, b, "merge " + String(k), chk)
    check_loose(table, chk, ignore)


def vocab_decode_ids(tok: VocabTokenizer, ids: List[Int]) raises -> List[UInt8]:
    """Every id decoded in order (an added id renders its content)."""
    var out = List[UInt8]()
    for k in range(len(ids)):
        tok.append_decoded(out, ids[k], k)
    return out^


def vocab_encode_ids(tok: VocabTokenizer, bytes: List[UInt8], allow: Bool) raises -> List[Int]:
    """`models.Tokenizer.encode`'s ids, reversed in the sabotage build."""
    var ids = tok.encode(bytes, allow)
    comptime if TOKENIZER_HOST_SABOTAGE:
        var rev = List[Int](capacity=len(ids))
        for i in range(len(ids)):
            rev.append(ids[len(ids) - i - 1])
        return rev^
    else:
        return ids^


def vocab_check_pairs(
    table: RankTable, lefts: List[List[UInt8]], rights: List[List[UInt8]], mut chk: MergeCheck
) raises:
    """`vocab_init_public`'s merges, checked in order."""
    for i in range(len(lefts)):
        check_merge(table, lefts[i], rights[i], "merge " + String(i), chk)


def vocab_fingerprint_hex(tok: VocabTokenizer, tail: List[UInt8]) -> String:
    """SHA-256 hex of the vocabulary's identity stream: per rank
    `<len>:` + bytes, per added token `<id>:<len>:` + bytes, then `tail`
    (the shell's settings words). The stream is the concatenation the
    former incremental `hashlib.sha256` updates hashed, so the digest is
    the same."""
    var stream = List[UInt8]()
    for i in range(tok.ranks.n_tokens()):
        tokenizer_append_bytes(stream, string_bytes(String(tok.ranks.length[i]) + ":"))
        for j in range(tok.ranks.length[i]):
            stream.append(tok.ranks.arena[tok.ranks.offset[i] + j])
    for i in range(len(tok.sp_id)):
        tokenizer_append_bytes(
            stream, string_bytes(String(tok.sp_id[i]) + ":" + String(tok.sp_length[i]) + ":")
        )
        for j in range(tok.sp_length[i]):
            stream.append(tok.sp_arena[tok.sp_offset[i] + j])
    tokenizer_append_bytes(stream, tail)
    return sha256_hex(stream)

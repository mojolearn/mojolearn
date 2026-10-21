# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed work behind the FAST tokenizer shell (bindings/_mojolearn_tokenizer_fast.mojo).

The shell unpacks Python objects, calls ONE function below per entry and converts the result back.
Everything else lives here, over native pointers, lists and handles: the checks over the caller's
document lengths, the host input step, the device calls (tokenizer/fast/bpe_device.mojo) and the
copy of the result into the caller's buffers. No Python import.

HOST INPUT STEP, STILL OWED TO THE DEVICE (stated so the device path is not read as host-free):
pre-tokenization (the GPT-2 pattern, `tokenizer/impl/pretokenize.mojo`) and the training corpus's
pre-token dedup (`PieceGroups`, train_bpe's step 1) run on the host before the first upload, as the
host trainer and encoder run them. A device pre-tokenizer is the next lane: per-byte class codes,
then the boundary rules (contractions, ` ?\\p{L}+` and the rest, `\\s+(?!\\S)`), a segmented scan
for run ends, then a device hash dedup. The vocabulary emission after the merges come back (each
new token's bytes, the join of its two parts, in merge order) is the trained model's output, sized
by vocab_size, as the host trainer writes it.
"""
from std.memory import memcpy

from tokenizer.encoding import BpeTokenizer, load_bpe_tokenizer_from
from tokenizer.impl.pretokenize import pretokenize
from tokenizer.impl.unicode_class import UnicodeClasses, builtin_unicode_classes
from tokenizer.train.bpe_train import PieceGroups, TrainedVocabulary
from tokenizer.fast.bpe_device import (
    BPE_ENCODE_DEVICE,
    BPE_GROUP_FILTER,
    BPE_LIVEBUF,
    BPE_MAX_VOCAB,
    BPE_MERGE_BATCH,
    BPE_TRAIN_DEVICE,
    BpeDeviceTable,
    BpeMerges,
    bpe_encode_batch_device,
    bpe_train_merges_device,
)


struct FastBpeHandle(Movable, Writable):
    """Python-owned: the parsed rank table (the host loader's parse) and, under LIVEBUF, its
    device copy, kept for the handle's life."""

    var tok: Optional[BpeTokenizer]
    var table: BpeDeviceTable

    def __init__(out self):
        self.tok = Optional[BpeTokenizer]()
        self.table = BpeDeviceTable()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("_FastBpeHandle(loaded=", Bool(self.tok), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("_FastBpeHandle(loaded=", Bool(self.tok), ")")


struct FastTrainedHandle(Movable, Writable):
    """Python-owned result of one `bpe_train` call, held until `bpe_trained_copy` has copied it
    into the caller's buffers."""

    var vocab: Optional[TrainedVocabulary]

    def __init__(out self):
        self.vocab = Optional[TrainedVocabulary]()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("_FastTrainedHandle(trained=", Bool(self.vocab), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("_FastTrainedHandle(trained=", Bool(self.vocab), ")")


def bpe_fast_flag_list() -> List[Bool]:
    """[train_device, encode_device, merge_batch, group_filter, livebuf]: what this build compiled."""
    var out = List[Bool]()
    out.append(BPE_TRAIN_DEVICE)
    out.append(BPE_ENCODE_DEVICE)
    out.append(BPE_MERGE_BATCH)
    out.append(BPE_GROUP_FILTER)
    out.append(BPE_LIVEBUF)
    return out^


def _u8_ptr(addr: Int) raises -> MutPointer[UInt8, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null uint8 buffer address")
    return MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr)


def _i64_ptr(addr: Int) raises -> MutPointer[Int64, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int64 buffer address")
    return MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=addr)


def fast_doc_offsets(lens_address: Int, n_docs: Int, n: Int, who: String) raises -> List[Int]:
    """The n_docs + 1 running byte offsets of the documents whose n_docs int64 LENGTHS are at
    `lens_address`, with the host binding's checks and sentences (tokenizer_host_shell_impl's
    bpe_train_fill and bpe_encode_batch_into)."""
    var lens = _i64_ptr(lens_address)
    var offs = List[Int](length=n_docs + 1, fill=0)
    for k in range(n_docs):
        if Int(lens[k]) < 0:
            raise Error(who + ": document " + String(k) + " has a negative length")
        offs[k + 1] = offs[k] + Int(lens[k])
    if offs[n_docs] != n:
        raise Error(
            who + ": document lengths must sum to n_bytes " + String(n) + ", got 0 and " + String(offs[n_docs])
        )
    return offs^


def _document(text_address: Int, offs: List[Int], d: Int) raises -> List[UInt8]:
    var a = offs[d]
    var m = offs[d + 1] - a
    var doc = List[UInt8](length=m, fill=UInt8(0))
    if m > 0:
        memcpy(dest=doc.unsafe_ptr(), src=_u8_ptr(text_address + a), count=m)
    return doc^


def fast_piece_groups(text_address: Int, offs: List[Int], classes: UnicodeClasses) raises -> PieceGroups:
    """HOST INPUT STEP: train_bpe's step 1, word for word (each document pre-tokenized alone, the
    pre-tokens deduplicated into groups with their counts)."""
    var groups = PieceGroups()
    for d in range(len(offs) - 1):
        var doc = _document(text_address, offs, d)
        var bounds = pretokenize(doc, classes)
        for k in range(len(bounds) - 1):
            groups.add(doc, bounds[k], bounds[k + 1])
    return groups^


def fast_pretoken_bounds(
    text_address: Int, offs: List[Int], classes: UnicodeClasses, mut pb: List[Int32], mut dp: List[Int32]
) raises:
    """HOST INPUT STEP: each document pre-tokenized alone (the host encoder's pretokenize). `pb`
    gets the pre-token byte bounds, absolute, document after document (npre + 1 entries), `dp`
    each document's first pre-token (n_docs + 1 entries)."""
    var n_docs = len(offs) - 1
    pb.clear()
    pb.append(Int32(0))
    dp = List[Int32](length=n_docs + 1, fill=Int32(0))
    for d in range(n_docs):
        var a = offs[d]
        if offs[d + 1] > a:
            var bounds = pretokenize(_document(text_address, offs, d), classes)
            for k in range(1, len(bounds)):
                pb.append(Int32(a + bounds[k]))
        dp[d + 1] = Int32(len(pb) - 1)


def fast_vocab_from_merges(merges: BpeMerges, n_groups: Int) raises -> TrainedVocabulary:
    """The trained vocabulary as train_bpe writes it: the 256 single bytes, then each merge's token
    (the bytes of its left part, then its right part) in merge order, the merges and the tie count."""
    var vocab = TrainedVocabulary()
    vocab.n_groups = n_groups
    vocab.n_ties_broken = merges.n_ties
    for b in range(256):
        vocab.offset.append(len(vocab.arena))
        vocab.length.append(1)
        vocab.arena.append(UInt8(b))
    for k in range(merges.n_merges):
        var a = Int(merges.pairs[2 * k])
        var b = Int(merges.pairs[2 * k + 1])
        var joined = List[UInt8]()
        for i in range(vocab.length[a]):
            joined.append(vocab.arena[vocab.offset[a] + i])
        for i in range(vocab.length[b]):
            joined.append(vocab.arena[vocab.offset[b] + i])
        vocab.offset.append(len(vocab.arena))
        vocab.length.append(len(joined))
        for i in range(len(joined)):
            vocab.arena.append(joined[i])
        vocab.merge_left.append(a)
        vocab.merge_right.append(b)
    return vocab^


# ------------------------------------------------------------------------
# The entries, one per shell function.


def fast_bpe_load(mut handle: FastBpeHandle, rp: String) raises:
    """`bpe_fast_load`'s work: the host loader's parse of the caller's rank file."""
    handle.tok = load_bpe_tokenizer_from(rp)


def fast_bpe_train_fill(
    mut handle: FastTrainedHandle,
    text_address: Int,
    lens_address: Int,
    n_docs: Int,
    n: Int,
    vocab_size: Int,
    min_frequency: Int,
) raises:
    """`bpe_train`'s work: cut the documents by their lengths, build the pre-token groups, run the
    merge loop on the GPU, write the vocabulary."""
    comptime if BPE_TRAIN_DEVICE:
        if vocab_size > BPE_MAX_VOCAB:
            raise Error("bpe_train (fast): vocab_size above " + String(BPE_MAX_VOCAB))
        var offs = fast_doc_offsets(lens_address, n_docs, n, "bpe_train")
        var groups = fast_piece_groups(text_address, offs, builtin_unicode_classes())
        var merges = bpe_train_merges_device(groups, vocab_size, min_frequency)
        handle.vocab = fast_vocab_from_merges(merges, groups.n())
    else:
        raise Error("bpe_train (fast): this build has no device trainer (-D MOJOLEARN_BPE_TRAIN_DEVICE, FAST, Apple)")


def fast_bpe_encode_batch_into(
    mut handle: FastBpeHandle,
    text_address: Int,
    lens_address: Int,
    out_address: Int,
    counts_address: Int,
    n_docs: Int,
    n: Int,
    cap: Int,
) raises -> Int:
    """`bpe_encode_batch`'s work on the GPU, without `<|endoftext|>` recognition: each document
    encoded alone, the ids back to back as int32 at `out_address`, each document's count as int64
    at `counts_address`. Returns the total. The ids are the host encoder's, id for id."""
    comptime if BPE_ENCODE_DEVICE:
        if cap < n:
            raise Error("bpe_encode_batch (fast): out_cap " + String(cap) + " is below n_bytes " + String(n))
        if n > 0 and out_address == 0:
            raise Error("mojolearn: null int32 buffer address")
        if counts_address == 0:
            raise Error("mojolearn: null int64 buffer address")
        var offs = fast_doc_offsets(lens_address, n_docs, n, "bpe_encode_batch")
        # The rank table and its device copy are two fields of one handle: reach them through
        # one pointer, as the host shell does.
        var owner = UnsafePointer(to=handle)
        var pb = List[Int32]()
        var dp = List[Int32]()
        fast_pretoken_bounds(text_address, offs, owner[].tok.value().classes, pb, dp)
        return bpe_encode_batch_device(
            owner[].tok.value().ranks, owner[].table, pb, dp, text_address, n_docs, n, out_address,
            counts_address,
        )
    else:
        raise Error("bpe_encode_batch (fast): this build has no device encoder (-D MOJOLEARN_BPE_ENCODE_DEVICE, FAST, Apple)")


def fast_bpe_trained_copy_into(
    v: TrainedVocabulary,
    arena_address: Int,
    lengths_address: Int,
    nm: Int,
    left_address: Int,
    right_address: Int,
) raises:
    """`bpe_trained_copy`'s work, the host binding's word for word: token bytes in rank order,
    lengths, merges."""
    var arena = _u8_ptr(arena_address)
    var lengths = _i64_ptr(lengths_address)
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

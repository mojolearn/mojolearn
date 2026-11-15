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
from std.memory import unsafe_memcpy

from tokenizer.encoding import BpeTokenizer
from tokenizer.impl.ranks import parse_rank_table
from tokenizer.impl.unicode_class import builtin_unicode_classes
from tokenizer.train.bpe_train import TrainedVocabulary
from tokenizer.fast.bpe_device import (
    BPE_ENCODE_DEVICE,
    BPE_GROUP_FILTER,
    BPE_LIVEBUF,
    BPE_MERGE_BATCH,
    BPE_TRAIN_DEVICE,
    BpeDeviceTable,
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


# ------------------------------------------------------------------------
# The entries, one per shell function.


def fast_bpe_load_text(mut handle: FastBpeHandle, text: String) raises:
    """`bpe_fast_load`'s work: the host loader's parse (`parse_rank_table`) of the canonical rank
    text the host binding renders for the same tokenizer, with the compiled-in Unicode classes."""
    handle.tok = BpeTokenizer(parse_rank_table(text, "<rank text>"), builtin_unicode_classes())


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
    raise Error("bpe_encode_batch (fast): this build has no device encoder (-D MOJOLEARN_BPE_ENCODE_DEVICE, FAST, Apple)")


def fast_bpe_trained_copy_into(
    v: TrainedVocabulary,
    arena_address: Int,
    lengths_address: Int,
    nm: Int,
    left_address: Int,
    right_address: Int,
) raises:
    """`bpe_trained_copy`'s work: token bytes in rank order, lengths, merges, as the host binding
    writes them. `fast_vocab_from_merges` fills the arena in rank order (token k's bytes start
    where token k - 1's end), so each array is one copy."""
    var arena = _u8_ptr(arena_address)
    var lengths = _i64_ptr(lengths_address)
    unsafe_memcpy(dest=arena, src=v.arena.unsafe_ptr(), count=len(v.arena))
    unsafe_memcpy(dest=lengths, src=v.length.unsafe_ptr().bitcast[Int64](), count=v.n_tokens())
    if nm > 0:
        unsafe_memcpy(dest=_i64_ptr(left_address), src=v.merge_left.unsafe_ptr().bitcast[Int64](), count=nm)
        unsafe_memcpy(dest=_i64_ptr(right_address), src=v.merge_right.unsafe_ptr().bitcast[Int64](), count=nm)

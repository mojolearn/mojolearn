# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST-tier tokenizer binding: BPE training and encode_batch on the Apple GPU
(lane/apple-fast-bpe, 2026-10-03). Built ONLY by `bindings/build_tokenizer_fast.sh`, which refuses
every tier but fast; the IDENTICAL door stays `_mojolearn_tokenizer_host` (unchanged).

`python/mojolearn/tokenizer.py` loads this file by path only under MOJOLEARN_NUMERIC_MODE=fast and
routes to it only for what `bpe_fast_flags()` says was compiled in (tokenizer/fast/bpe_device.mojo:
TRAIN_DEVICE, ENCODE_DEVICE; both need FAST + an Apple GPU target + their `-D MOJOLEARN_BPE_*`
define). A build with no define reports nothing compiled and every call runs main's host binding.

    tokenizer_fast_numeric_mode() -> 0 (FAST) or this build is refused by the loader
    bpe_fast_flags() -> [train_device, encode_device, merge_batch, group_filter, livebuf]
    bpe_train(text_addr, offsets_addr, [n_docs, n_bytes, vocab_size, min_frequency, break_ties_high])
        the host binding's ABI; the merge loop on the GPU. break_ties_high (the sabotage arm) is refused
        here: the Python door sends it to the host binding.
    bpe_trained_sizes(handle), bpe_trained_copy(handle, arena, lengths, left, right): the host ABI
    bpe_fast_load(ranks_path) -> handle
    bpe_encode_batch(handle, text_addr, offsets_addr, ids_addr, counts_addr, [n_docs, n_bytes])
        no <|endoftext|> recognition; document k's ids (int32) at ids[offsets[k] ...], its count (int64)
        at counts[k]; returns the total.
"""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from std.memory import memcpy

from checks.numerics import GLOBAL_NUMERIC_MODE
from tokenizer.encoding import BpeTokenizer, load_bpe_tokenizer_from
from tokenizer.impl.unicode_class import builtin_unicode_classes
from tokenizer.train.bpe_train import TrainedVocabulary
from tokenizer.fast.bpe_device import (
    BPE_ENCODE_DEVICE,
    BPE_GROUP_FILTER,
    BPE_LIVEBUF,
    BPE_MAX_VOCAB,
    BPE_MERGE_BATCH,
    BPE_TRAIN_DEVICE,
    BpeDeviceTable,
    bpe_encode_batch_device,
    bpe_train_device,
)


struct FastBpeHandle(Movable, Writable):
    """Python-owned: the parsed rank table (the host's loader) and, under LIVEBUF, its device copy."""

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
    var vocab: Optional[TrainedVocabulary]

    def __init__(out self):
        self.vocab = Optional[TrainedVocabulary]()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("_FastTrainedHandle(trained=", Bool(self.vocab), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("_FastTrainedHandle(trained=", Bool(self.vocab), ")")


def _index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("tokenizer fast: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    return Int(py=operator_module.index(value))


def _flag(value: PythonObject, name: String) raises -> Bool:
    var type_name = String(py=value.__class__.__name__)
    if type_name != "bool":
        raise Error("tokenizer fast: " + name + " must be a bool, got " + type_name)
    return Bool(py=value)


def _u8_ptr(addr: Int) raises -> MutPointer[UInt8, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null uint8 buffer address")
    return MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr)


def _i64_ptr(addr: Int) raises -> MutPointer[Int64, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int64 buffer address")
    return MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=addr)


def _check_offsets(offs: MutPointer[Int64, MutUntrackedOrigin], n_docs: Int, nbytes: Int, who: String) raises:
    if Int(offs[0]) != 0 or Int(offs[n_docs]) != nbytes:
        raise Error(who + ": offsets must start at 0 and end at n_bytes " + String(nbytes))
    for k in range(n_docs):
        if Int(offs[k + 1]) < Int(offs[k]):
            raise Error(who + ": offsets decrease at document " + String(k))


def tokenizer_fast_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def bpe_fast_flags_binding() raises -> PythonObject:
    var out = Python.list()
    out.append(PythonObject(BPE_TRAIN_DEVICE))
    out.append(PythonObject(BPE_ENCODE_DEVICE))
    out.append(PythonObject(BPE_MERGE_BATCH))
    out.append(PythonObject(BPE_GROUP_FILTER))
    out.append(PythonObject(BPE_LIVEBUF))
    return out


def bpe_train_binding(text_addr: PythonObject, offsets_addr: PythonObject, dims: PythonObject) raises -> PythonObject:
    """The host `bpe_train` ABI with the merge loop on the GPU (TRAIN_DEVICE builds only)."""
    if Int(py=len(dims)) != 5:
        raise Error("bpe_train: dims must be [n_docs, n_bytes, vocab_size, min_frequency, break_ties_high]")
    var n_docs = _index(dims[0])
    var nbytes = _index(dims[1])
    var vocab_size = _index(dims[2])
    var min_frequency = _index(dims[3])
    var reverse = _flag(dims[4], "break_ties_high")
    if reverse:
        raise Error("bpe_train (fast): break_ties_high is the host binding's negative control")
    if n_docs < 1:
        raise Error("bpe_train: n_docs must be >= 1, got " + String(n_docs))
    if nbytes < 0:
        raise Error("bpe_train: n_bytes must be >= 0, got " + String(nbytes))
    if vocab_size > BPE_MAX_VOCAB:
        raise Error("bpe_train (fast): vocab_size above " + String(BPE_MAX_VOCAB))
    var text_address = _index(text_addr) if nbytes > 0 else 0
    var offsets_address = _index(offsets_addr)
    var handle = FastTrainedHandle()
    comptime if BPE_TRAIN_DEVICE:
        with GILReleased(Python()):
            var offs = _i64_ptr(offsets_address)
            _check_offsets(offs, n_docs, nbytes, "bpe_train")
            var documents = List[List[UInt8]](capacity=n_docs)
            for k in range(n_docs):
                var a = Int(offs[k])
                var m = Int(offs[k + 1]) - a
                var doc = List[UInt8](length=m, fill=UInt8(0))
                if m > 0:
                    memcpy(dest=doc.unsafe_ptr(), src=_u8_ptr(text_address + a), count=m)
                documents.append(doc^)
            var classes = builtin_unicode_classes()
            handle.vocab = bpe_train_device(documents, classes, vocab_size, min_frequency)
    else:
        raise Error("bpe_train (fast): this build has no device trainer (-D MOJOLEARN_BPE_TRAIN_DEVICE, FAST, Apple)")
    return PythonObject(alloc=handle^)


def bpe_trained_sizes_binding(handle: PythonObject) raises -> PythonObject:
    var owner = handle.downcast_value_ptr[FastTrainedHandle]()
    if not owner[].vocab:
        raise Error("tokenizer fast: the handle holds no trained vocabulary")
    ref v = owner[].vocab.value()
    var out = Python.list()
    out.append(PythonObject(v.n_tokens()))
    out.append(PythonObject(len(v.arena)))
    out.append(PythonObject(v.n_merges()))
    out.append(PythonObject(v.n_ties_broken))
    out.append(PythonObject(v.n_groups))
    return out


def bpe_trained_copy_binding(
    handle: PythonObject,
    arena_addr: PythonObject,
    lengths_addr: PythonObject,
    left_addr: PythonObject,
    right_addr: PythonObject,
) raises -> PythonObject:
    """The host binding's copy-out, word for word: token bytes in rank order, lengths, merges."""
    var owner = handle.downcast_value_ptr[FastTrainedHandle]()
    if not owner[].vocab:
        raise Error("tokenizer fast: the handle holds no trained vocabulary")
    var arena_address = _index(arena_addr)
    var lengths_address = _index(lengths_addr)
    var nm = owner[].vocab.value().n_merges()
    var left_address = _index(left_addr) if nm > 0 else 0
    var right_address = _index(right_addr) if nm > 0 else 0
    ref v = owner[].vocab.value()
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
    return PythonObject(v.n_tokens())


def bpe_fast_load_binding(ranks_path: PythonObject) raises -> PythonObject:
    """The host loader's parse of the caller's rank file, into this binding's own handle."""
    var rp = String(py=ranks_path)
    var handle = FastBpeHandle()
    with GILReleased(Python()):
        handle.tok = load_bpe_tokenizer_from(rp)
    return PythonObject(alloc=handle^)


def bpe_encode_batch_binding(
    handle: PythonObject,
    text_addr: PythonObject,
    offsets_addr: PythonObject,
    ids_addr: PythonObject,
    counts_addr: PythonObject,
    dims: PythonObject,
) raises -> PythonObject:
    """encode_batch on the GPU (ENCODE_DEVICE builds only), allow_endoftext=False."""
    var owner = handle.downcast_value_ptr[FastBpeHandle]()
    if not owner[].tok:
        raise Error("tokenizer fast: the handle holds no tokenizer; bpe_fast_load it")
    if Int(py=len(dims)) != 2:
        raise Error("bpe_encode_batch (fast): dims must be [n_docs, n_bytes]")
    var n_docs = _index(dims[0])
    var nbytes = _index(dims[1])
    if n_docs <= 0 or nbytes < 0:
        raise Error("bpe_encode_batch (fast): n_docs must be >= 1 and n_bytes >= 0")
    var text_address = _index(text_addr)
    var offsets_address = _index(offsets_addr)
    var ids_address = _index(ids_addr)
    var counts_address = _index(counts_addr)
    var total = 0
    comptime if BPE_ENCODE_DEVICE:
        with GILReleased(Python()):
            _check_offsets(_i64_ptr(offsets_address), n_docs, nbytes, "bpe_encode_batch (fast)")
            total = bpe_encode_batch_device(
                owner[].tok.value().ranks, owner[].tok.value().classes, owner[].table, text_address,
                offsets_address, n_docs, nbytes, ids_address, counts_address,
            )
    else:
        raise Error("bpe_encode_batch (fast): this build has no device encoder (-D MOJOLEARN_BPE_ENCODE_DEVICE, FAST, Apple)")
    return PythonObject(total)


@export
def PyInit__mojolearn_tokenizer_fast() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_tokenizer_fast")
        module.def_function[tokenizer_fast_numeric_mode_binding]("tokenizer_fast_numeric_mode")
        module.def_function[bpe_fast_flags_binding]("bpe_fast_flags")
        _ = module.add_type[FastTrainedHandle]("_FastTrainedHandle")
        module.def_function[bpe_train_binding]("bpe_train")
        module.def_function[bpe_trained_sizes_binding]("bpe_trained_sizes")
        module.def_function[bpe_trained_copy_binding]("bpe_trained_copy")
        _ = module.add_type[FastBpeHandle]("_FastBpeHandle")
        module.def_function[bpe_fast_load_binding]("bpe_fast_load")
        module.def_function[bpe_encode_batch_binding]("bpe_encode_batch")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_tokenizer_fast: ", error))

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the byte-level BPE tokenizer in the GPT-2 format
(`tokenizer/`), the expose-tokenizer lane, 2026-09-14.

HOST ONLY, AND THE ONLY BINDING THIS FAMILY HAS. `tokenizer/` is integer
and table work (a byte buffer, three Unicode range tables, a hash probe and
an argmin over small integers): no float arithmetic, no device kernel, no
GPU binding to route from. So unlike the other host families this one is
not a CPU twin of a GPU entry; it is the door itself, loaded by path through
`_backend.load_host_module` from `python/mojolearn/tokenizer.py`, and the
same binary serves a GPU box and a CPU-only install alike.

WHAT IT COMPUTES. `tokenizer/encoding.mojo::BpeTokenizer.encode_bytes` and
`decode_bytes` over the rank file the caller loads (mojolearn ships no
vocabulary, 2026-09-15): the ids of a byte string and the bytes back.
`tokenizer/checks/tokenizer_check.mojo` (`pixi run check-tokenizer`) is where
the algorithm is asserted; this file adds no arithmetic and no policy.
`<|endoftext|>` (the id after the last rank) is a token only when the caller
passes `allow_endoftext=True`, otherwise its thirteen characters are ordinary
text.

THE ADDRESS CONTRACT, mirrored word for word in `python/mojolearn/
tokenizer.py`:

    bpe_load(ranks_path) -> handle
        parses the caller's rank file once, with the Unicode classes
        compiled into this build, and returns an opaque `_BpeHandle` every
        other entry takes first. One handle per BpeTokenizer.
    bpe_n_vocab(handle) -> the ranks plus `<|endoftext|>`
    bpe_max_token_bytes(handle) -> the longest token's byte length, so the
        caller can size a decode output in one call
    bpe_encode(handle, text_addr, n_bytes, out_addr, out_cap, allow_endoftext)
        reads `n_bytes` uint8 at `text_addr`, writes the ids as int32 at
        `out_addr` and returns how many. An encoding never has more ids than
        bytes (a merge only shortens, and the 13-byte special token is one
        id), so `out_cap = n_bytes` always suffices; a count above `out_cap`
        is refused before anything is written. `n_bytes = 0` reads and
        writes nothing and returns 0.
    bpe_decode(handle, ids_addr, n_ids, out_addr, out_cap)
        reads `n_ids` int32 at `ids_addr`, writes the bytes at `out_addr`
        and returns how many. An id outside [0, n_vocab) is refused BY NAME
        AND POSITION (`encoding.mojo`'s own sentence) with nothing written;
        `out_cap = n_ids * bpe_max_token_bytes` always suffices.

THE SABOTAGE. `-D MOJOLEARN_TOKENIZER_HOST_SABOTAGE=1` makes `bpe_encode`
write the ids in REVERSE order. It is the Python gate's negative control: a
build with it defined must fail `python/mojolearn/tests/
test_tokenizer_surface.py`'s exact-id cases, or that test is not reading
this binary's output. `tokenizer_host_sabotage()` reads it back and
`_backend.load_host_module` refuses such a build outside the gate; the
batch entry honors it too (each document's ids reversed).

THE BATCH ENTRY (lane/inference-tokenizer-neural, 2026-09-15).
`bpe_encode_batch` encodes many documents in one call, each ALONE, so
its ids per document are `bpe_encode`'s. It exists because the crossing
is most of the cost for short documents: on the M4, one core, 20,000
documents of about 18 bytes took 0.23 s through 20,000 `bpe_encode`
calls and 0.033 s as one call on the same bytes concatenated.
`-D MOJOLEARN_TOKENIZER_BATCH_SABOTAGE=1` swaps ids across each document
boundary inside a batch (a batch of one is untouched), the batch part's
negative control; `tokenizer_host_sabotage()` reads True for either define.

THE TRAINER ENTRIES (lane/bpe-builder-native, 2026-09-18). `bpe_train`
runs `tokenizer/train/bpe_train.mojo::train_bpe` on documents passed like
`bpe_encode_batch`'s (bytes back to back, int64 offsets) and returns an
opaque `_BpeTrainedHandle`; `bpe_trained_sizes` and `bpe_trained_copy` read
the vocabulary out. `BpeVocabularyTrainer` (backend "auto" or "mojo") calls
them; `pixi run check-bpe-trainer` holds the trainer to the Python reference
file byte for file byte. `-D MOJOLEARN_BPE_TRAINER_SABOTAGE=1` reverses the
tie-break in this build too, and `tokenizer_host_sabotage()` reads True for
it; the `break_ties_high` flag in `dims` is the Python door's environment
spelling of the same arm.
"""
from std.memory import memcpy
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from std.sys.compile import is_defined
from core.host_parallel import host_parallelize

from bindings.hostptr import i32_ptr
from core.host_predict_threads import host_predict_chunk, host_predict_task_count
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from tokenizer.encoding import (
    GPT2_ENDOFTEXT,
    GPT2_PAT_STR,
    BpeTokenizer,
    bytes_string,
    load_bpe_tokenizer_from,
    string_bytes,
)
from tokenizer.impl.byte_unicode import codepoint_to_byte
from tokenizer.impl.pretokenize import pretokenize_pattern
from tokenizer.impl.ranks import RankTable
from tokenizer.impl.unicode_class import builtin_unicode_classes
from tokenizer.impl.vocab_build import (
    MergeCheck,
    check_loose,
    check_merge,
    ranks_text,
    split_merge_line,
    table_from_tokens,
    unspell,
    unspell_error,
)
from tokenizer.train.emit import render_ranks, render_tokenizer_json
from tokenizer.vocab import VocabTokenizer
from tokenizer.train.bpe_train import (
    BPE_TRAINER_SABOTAGE,
    TrainedVocabulary,
    train_bpe,
)

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


def _index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("tokenizer host: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    return Int(py=operator_module.index(value))


def _flag(value: PythonObject, name: String) raises -> Bool:
    var type_name = String(py=value.__class__.__name__)
    if type_name != "bool":
        raise Error(
            "tokenizer host: " + name + " must be a bool, got " + type_name
        )
    return Bool(py=value)


def _u8_ptr(addr: Int) raises -> MutPointer[UInt8, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null uint8 buffer address")
    return MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr)


def _require_loaded(loaded: Bool) raises:
    if not loaded:
        raise Error("tokenizer host: the handle holds no tokenizer; bpe_load it")


def tokenizer_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def tokenizer_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def tokenizer_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives
    in a function body PyInit registers, so it is compiled in every build.
    `bindings/build_host_family.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "tokenizer host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_tokenizer_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


# There is no `tokenizer_host_detected_column` read-back, for the reason
# 8d16ce2f removed it from the forest and byte LM host bindings: the detected
# column folds to the GPU of the machine that ran the build. The comptime
# assert above is the check.


def tokenizer_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary is a negative-control build: encode's ids in
    reverse order (-D MOJOLEARN_TOKENIZER_HOST_SABOTAGE=1), ids swapped
    across batch boundaries (-D MOJOLEARN_TOKENIZER_BATCH_SABOTAGE=1) or
    the trainer's tie-break reversed (-D MOJOLEARN_BPE_TRAINER_SABOTAGE=1)."""
    return PythonObject(
        TOKENIZER_HOST_SABOTAGE
        or TOKENIZER_BATCH_SABOTAGE
        or BPE_TRAINER_SABOTAGE
    )


def bpe_load_binding(ranks_path: PythonObject) raises -> PythonObject:
    """Parse the caller's rank file once into a handle. Every refusal in
    `tokenizer/impl/ranks.mojo::load_rank_table` (a rank out of order, an
    odd hex field, a duplicate token) and
    `unicode_class.mojo::builtin_unicode_classes` (a generated table that is
    not the pin) raises here with its own sentence."""
    var rp = String(py=ranks_path)
    var handle = BpeHandle()
    with GILReleased(Python()):
        var tok = load_bpe_tokenizer_from(rp)
        var longest = len(String(GPT2_ENDOFTEXT).as_bytes())
        for i in range(tok.ranks.n_tokens()):
            if tok.ranks.length[i] > longest:
                longest = tok.ranks.length[i]
        handle.max_token_bytes = longest
        handle.tok = tok^
    return PythonObject(alloc=handle^)


def bpe_n_vocab_binding(handle: PythonObject) raises -> PythonObject:
    var owner = handle.downcast_value_ptr[BpeHandle]()
    _require_loaded(Bool(owner[].tok))
    return PythonObject(owner[].tok.value().n_vocab())


def bpe_max_token_bytes_binding(handle: PythonObject) raises -> PythonObject:
    var owner = handle.downcast_value_ptr[BpeHandle]()
    _require_loaded(Bool(owner[].tok))
    return PythonObject(owner[].max_token_bytes)


def bpe_encode_binding(
    handle: PythonObject,
    text_addr: PythonObject,
    n_bytes: PythonObject,
    out_addr: PythonObject,
    out_cap: PythonObject,
    allow_endoftext: PythonObject,
) raises -> PythonObject:
    """`BpeTokenizer.encode_bytes` on the host. Returns the id count."""
    var owner = handle.downcast_value_ptr[BpeHandle]()
    _require_loaded(Bool(owner[].tok))
    var n = _index(n_bytes)
    var cap = _index(out_cap)
    var allow = _flag(allow_endoftext, "allow_endoftext")
    if n < 0:
        raise Error("bpe_encode: n_bytes must be >= 0, got " + String(n))
    if cap < 0:
        raise Error("bpe_encode: out_cap must be >= 0, got " + String(cap))
    if n == 0:
        return PythonObject(0)
    var text_address = _index(text_addr)
    var out_address = _index(out_addr)
    var count = 0
    with GILReleased(Python()):
        var src = _u8_ptr(text_address)
        var text = List[UInt8](length=n, fill=UInt8(0))
        memcpy(dest=text.unsafe_ptr(), src=src, count=n)
        # THE ONE CALL THAT COMPUTES ANYTHING.
        var ids = owner[].tok.value().encode_bytes(text, allow)
        count = len(ids)
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
    return PythonObject(count)


def _i64_ptr(addr: Int) raises -> MutPointer[Int64, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int64 buffer address")
    return MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=addr)


def bpe_encode_batch_binding(
    handle: PythonObject,
    text_addr: PythonObject,
    offsets_addr: PythonObject,
    out_addr: PythonObject,
    counts_addr: PythonObject,
    dims: PythonObject,
) raises -> PythonObject:
    """`BpeTokenizer.encode_batch` on the host: ONE crossing for many
    documents. `dims` is `[n_docs, n_bytes, out_cap, allow_endoftext]`.
    Reads the concatenated `n_bytes` uint8 at `text_addr` and `n_docs + 1`
    int64 offsets at `offsets_addr` (0 first, nondecreasing, `n_bytes`
    last); document k is bytes [offsets[k], offsets[k + 1]). Each document
    is encoded ALONE, by the same `encode_bytes` call `bpe_encode` makes on
    its own buffer, so its ids are those of `bpe_encode` on that document
    byte for byte. Writes every document's ids back to back as int32 at
    `out_addr` and each document's id count as int64 at `counts_addr`;
    returns the total. `out_cap = n_bytes` always suffices. Every offset is
    checked before any document is encoded; a total above `out_cap` is
    refused with nothing written."""
    var owner = handle.downcast_value_ptr[BpeHandle]()
    _require_loaded(Bool(owner[].tok))
    if Int(py=len(dims)) != 4:
        raise Error("bpe_encode_batch: dims must be [n_docs, n_bytes, out_cap, allow_endoftext]")
    var n_docs = _index(dims[0])
    var n = _index(dims[1])
    var cap = _index(dims[2])
    var allow = _flag(dims[3], "allow_endoftext")
    if n_docs < 0:
        raise Error("bpe_encode_batch: n_docs must be >= 0, got " + String(n_docs))
    if n < 0:
        raise Error("bpe_encode_batch: n_bytes must be >= 0, got " + String(n))
    if cap < 0:
        raise Error("bpe_encode_batch: out_cap must be >= 0, got " + String(cap))
    if n_docs == 0:
        return PythonObject(0)
    var text_address = _index(text_addr) if n > 0 else 0
    var offsets_address = _index(offsets_addr)
    var out_address = _index(out_addr) if cap > 0 else 0
    var counts_address = _index(counts_addr)
    var total = 0
    with GILReleased(Python()):
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
        total = len(all_ids)
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
    return PythonObject(total)


def bpe_decode_binding(
    handle: PythonObject,
    ids_addr: PythonObject,
    n_ids: PythonObject,
    out_addr: PythonObject,
    out_cap: PythonObject,
) raises -> PythonObject:
    """`BpeTokenizer.decode_bytes` on the host. Returns the byte count."""
    var owner = handle.downcast_value_ptr[BpeHandle]()
    _require_loaded(Bool(owner[].tok))
    var n = _index(n_ids)
    var cap = _index(out_cap)
    if n < 0:
        raise Error("bpe_decode: n_ids must be >= 0, got " + String(n))
    if cap < 0:
        raise Error("bpe_decode: out_cap must be >= 0, got " + String(cap))
    if n == 0:
        return PythonObject(0)
    var ids_address = _index(ids_addr)
    var out_address = _index(out_addr)
    var count = 0
    with GILReleased(Python()):
        var src = i32_ptr(ids_address)
        var ids = List[Int](capacity=n)
        for k in range(n):
            ids.append(Int(src[k]))
        # THE ONE CALL THAT COMPUTES ANYTHING; an id outside [0, n_vocab) is
        # refused here by name and position, nothing written.
        var out = owner[].tok.value().decode_bytes(ids)
        count = len(out)
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
    return PythonObject(count)


def bpe_train_binding(
    text_addr: PythonObject,
    offsets_addr: PythonObject,
    dims: PythonObject,
) raises -> PythonObject:
    """`tokenizer/train/bpe_train.mojo::train_bpe` on the host
    (lane/bpe-builder-native, 2026-09-18): TRAIN a vocabulary.

    `dims` is `[n_docs, n_bytes, vocab_size, min_frequency,
    break_ties_high]`. Reads the concatenated `n_bytes` uint8 at `text_addr`
    and `n_docs + 1` int64 offsets at `offsets_addr` (0 first,
    nondecreasing, `n_bytes` last); document k is bytes
    [offsets[k], offsets[k + 1]), and each is pre-tokenized ALONE, exactly
    as `BpeVocabularyTrainer` hands them over. `break_ties_high` is the
    Python door's MOJOLEARN_BPE_TRAINER_SABOTAGE environment arm and must be
    False outside a negative control. Returns an opaque handle;
    `bpe_trained_sizes` then `bpe_trained_copy` read the result out. This
    file adds no arithmetic: the one call below is the trainer
    `pixi run check-bpe-trainer` holds to the Python reference."""
    if Int(py=len(dims)) != 5:
        raise Error(
            "bpe_train: dims must be [n_docs, n_bytes, vocab_size,"
            " min_frequency, break_ties_high]"
        )
    var n_docs = _index(dims[0])
    var n = _index(dims[1])
    var vocab_size = _index(dims[2])
    var min_frequency = _index(dims[3])
    var reverse = _flag(dims[4], "break_ties_high")
    if n_docs < 1:
        raise Error("bpe_train: n_docs must be >= 1, got " + String(n_docs))
    if n < 0:
        raise Error("bpe_train: n_bytes must be >= 0, got " + String(n))
    var text_address = _index(text_addr) if n > 0 else 0
    var offsets_address = _index(offsets_addr)
    var handle = BpeTrainedHandle()
    with GILReleased(Python()):
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
    return PythonObject(alloc=handle^)


def bpe_trained_sizes_binding(handle: PythonObject) raises -> PythonObject:
    """`[n_tokens, arena_bytes, n_merges, n_ties_broken, n_groups]`, so the
    caller can size `bpe_trained_copy`'s buffers."""
    var owner = handle.downcast_value_ptr[BpeTrainedHandle]()
    if not owner[].vocab:
        raise Error("tokenizer host: the handle holds no trained vocabulary")
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
    """Copy the trained vocabulary out: the token bytes back to back in rank
    order (uint8, `arena_bytes`), each token's length (int64, `n_tokens`),
    and the merges' left and right ids (int64, `n_merges` each). Buffers are
    sized from `bpe_trained_sizes`. Returns `n_tokens`."""
    var owner = handle.downcast_value_ptr[BpeTrainedHandle]()
    if not owner[].vocab:
        raise Error("tokenizer host: the handle holds no trained vocabulary")
    var arena_address = _index(arena_addr)
    var lengths_address = _index(lengths_addr)
    var nm = owner[].vocab.value().n_merges()
    var left_address = _index(left_addr) if nm > 0 else 0
    var right_address = _index(right_addr) if nm > 0 else 0
    ref v = owner[].vocab.value()
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
    return PythonObject(v.n_tokens())


# ------------------------------------------------------------------------
# lane/pyglue-text-io (2026-10-03): the work the Python doors did over the
# caller's data, moved here. `tokenizer.py` and `models/tokenizer.py` now
# only validate arguments, pass the caller's objects or buffers and return
# results. Refusals raised here start with the Python exception name the
# door re-raises them as (`ValueError: `, `TypeError: `).


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


def _py_repr(obj: PythonObject) raises -> String:
    var bi = Python.import_module("builtins")
    return String(py=bi.repr(obj))


def _unspell_py(text: PythonObject, inv: List[Int], label: String, mut out: List[UInt8]) raises:
    var bi = Python.import_module("builtins")
    if String(py=text.__class__.__name__) != "str":
        raise Error("ValueError: " + label + " " + _py_repr(text) + " is not a string")
    var b = string_bytes(String(py=text))
    var bad = unspell(b, inv, out)
    if bad != -1:
        raise Error(unspell_error(label + " " + _py_repr(text), bad))
    _ = bi


def _table_from_spelled(
    vocab: PythonObject,
    merges: PythonObject,
    version_line: Bool,
    ignore_merges: Bool,
) raises -> RankTable:
    """A rank table from `{byte-level spelling: id}` (ids 0..n-1, each
    once) and a merge list: either the text of a `vocab.bpe` file (one
    `a b` per line, an optional leading `#version` line, blank lines
    skipped) or a list whose items are `"a b"` strings or `[a, b]` pairs
    (a `tokenizer.json`'s `model.merges`). Every check in
    `impl/vocab_build.mojo` runs here."""
    var bi = Python.import_module("builtins")
    var items = bi.list(vocab.items())
    var n = Int(py=bi.len(items))
    var inv = codepoint_to_byte()
    var toks = List[List[UInt8]](capacity=n)
    var placed = List[Bool](length=n, fill=False)
    for _ in range(n):
        toks.append(List[UInt8]())
    for k in range(n):
        var key = items[k][0]
        var val = items[k][1]
        var vt = String(py=val.__class__.__name__)
        var i = -1
        if vt == "int":
            i = Int(py=val)
        if vt != "int" or i < 0 or i >= n or placed[i]:
            raise Error(
                "ValueError: id "
                + _py_repr(val)
                + " of "
                + _py_repr(key)
                + " is not a unique id in [0, "
                + String(n)
                + ")"
            )
        var out = List[UInt8]()
        _unspell_py(key, inv, String("spelling"), out)
        toks[i] = out^
        placed[i] = True
    var arena = List[UInt8]()
    var lengths = List[Int](capacity=n)
    for i in range(n):
        lengths.append(len(toks[i]))
        for j in range(len(toks[i])):
            arena.append(toks[i][j])
    var table = table_from_tokens(arena, lengths)
    var chk = MergeCheck(n)
    if String(py=merges.__class__.__name__) == "str":
        var text = string_bytes(String(py=merges))
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
    else:
        var m = Int(py=bi.len(merges))
        for k in range(m):
            var item = merges[k]
            var label = "merge " + String(k)
            var it = String(py=item.__class__.__name__)
            var a = List[UInt8]()
            var b = List[UInt8]()
            if it == "str":
                var sa = List[UInt8]()
                var sb = List[UInt8]()
                if (
                    not split_merge_line(string_bytes(String(py=item)), sa, sb)
                ):
                    raise Error("ValueError: " + label + " " + _py_repr(item) + " is not a pair")
                if unspell(sa, inv, a) != -1 or unspell(sb, inv, b) != -1:
                    raise Error(
                        "ValueError: " + label + " names " + _py_repr(item) + ", not a byte-level spelling"
                    )
            elif (it == "list" or it == "tuple") and Int(py=bi.len(item)) == 2:
                _unspell_py(item[0], inv, label + " names", a)
                _unspell_py(item[1], inv, label + " names", b)
            else:
                raise Error("ValueError: " + label + " " + _py_repr(item) + " is not a pair")
            check_merge(table, a, b, label, chk)
    check_loose(table, chk, ignore_merges)
    return table^


def _quoted(line: List[UInt8]) -> String:
    var s = String("'")
    s += bytes_string(line)
    s += "'"
    return s^


def bpe_load_tokens_binding(addresses: PythonObject, dims: PythonObject) raises -> PythonObject:
    """`BpeTokenizer.from_token_bytes`: `addresses` `[token_bytes(u8),
    lengths(i64)]`, `dims` `[n_tokens]`. Refuses an empty or repeated token
    and a missing single byte by name (`vocab_build.mojo`)."""
    var n = _index(dims[0])
    var lengths = _i64_list(_index(addresses[1]) if n > 0 else 0, n)
    var total = _bytes_total(lengths)
    var arena = _u8_list(_index(addresses[0]) if total > 0 else 0, total)
    var table = table_from_tokens(arena, lengths)
    return PythonObject(alloc=_handle_from_table(table^))


def bpe_load_spelled_binding(vocab: PythonObject, merges: PythonObject) raises -> PythonObject:
    """`BpeTokenizer.from_files`: `vocab` the `encoder.json` object without
    `<|endoftext|>`, `merges` the `vocab.bpe` text."""
    var table = _table_from_spelled(vocab, merges, True, False)
    return PythonObject(alloc=_handle_from_table(table^))


def bpe_ranks_text_binding(handle: PythonObject) raises -> PythonObject:
    """The canonical rank file text of the loaded table (its identity is the
    SHA-256 of these bytes)."""
    var owner = handle.downcast_value_ptr[BpeHandle]()
    _require_loaded(Bool(owner[].tok))
    return PythonObject(ranks_text(owner[].tok.value().ranks))


def _ids_from_seq(seq: PythonObject) raises -> List[Int]:
    """The ids of a Python sequence, refused by type and position."""
    var bi = Python.import_module("builtins")
    var operator_module = Python.import_module("operator")
    var n = Int(py=bi.len(seq))
    var out = List[Int](capacity=n)
    for k in range(n):
        var v = seq[k]
        var tn = String(py=v.__class__.__name__)
        if tn == "bool":
            raise Error("TypeError: ids must be int, not bool, at position " + String(k))
        var iv = 0
        try:
            iv = Int(py=operator_module.index(v))
        except:
            raise Error("TypeError: ids must be int, got " + tn + " at position " + String(k))
        out.append(iv)
    return out^


def bpe_decode_seq_binding(
    handle: PythonObject, seq: PythonObject, out_addr: PythonObject, out_cap: PythonObject
) raises -> PythonObject:
    """`BpeTokenizer.decode_bytes` over a Python sequence of ids: type and
    range checked by position, decoded, written at `out_addr`."""
    var owner = handle.downcast_value_ptr[BpeHandle]()
    _require_loaded(Bool(owner[].tok))
    var ids = _ids_from_seq(seq)
    var nv = owner[].tok.value().n_vocab()
    for k in range(len(ids)):
        if ids[k] < 0 or ids[k] >= nv:
            raise Error(
                "ValueError: id " + String(ids[k]) + " at position " + String(k)
                + " is outside [0, " + String(nv) + ")"
            )
    var cap = _index(out_cap)
    if len(ids) == 0:
        return PythonObject(0)
    var out = owner[].tok.value().decode_bytes(ids)
    if len(out) > cap:
        raise Error("bpe_decode_seq: " + String(len(out)) + " bytes do not fit " + String(cap))
    memcpy(dest=_u8_ptr(_index(out_addr)), src=out.unsafe_ptr(), count=len(out))
    return PythonObject(len(out))


def bpe_decode_batch_binding(
    handle: PythonObject, batch: PythonObject, out_addr: PythonObject,
    out_cap: PythonObject, lengths_addr: PythonObject
) raises -> PythonObject:
    """`decode_bytes` of each id sequence in `batch`, back to back at
    `out_addr`, each one's byte count (int64) at `lengths_addr`. An id is
    refused by its position inside its own sequence."""
    var owner = handle.downcast_value_ptr[BpeHandle]()
    _require_loaded(Bool(owner[].tok))
    var bi = Python.import_module("builtins")
    var n = Int(py=bi.len(batch))
    var cap = _index(out_cap)
    var nv = owner[].tok.value().n_vocab()
    var joined = List[UInt8]()
    var lens = List[Int](capacity=n)
    for s in range(n):
        var ids = _ids_from_seq(batch[s])
        for k in range(len(ids)):
            if ids[k] < 0 or ids[k] >= nv:
                raise Error(
                    "ValueError: id " + String(ids[k]) + " at position " + String(k)
                    + " is outside [0, " + String(nv) + ")"
                )
        var out = owner[].tok.value().decode_bytes(ids)
        lens.append(len(out))
        joined += out
    if len(joined) > cap:
        raise Error("bpe_decode_batch: " + String(len(joined)) + " bytes do not fit " + String(cap))
    if n > 0:
        var lp = _i64_ptr(_index(lengths_addr))
        for s in range(n):
            lp[s] = Int64(lens[s])
    if len(joined) > 0:
        memcpy(dest=_u8_ptr(_index(out_addr)), src=joined.unsafe_ptr(), count=len(joined))
    return PythonObject(len(joined))


def bpe_render_binding(addresses: PythonObject, dims: PythonObject) raises -> PythonObject:
    """A trained vocabulary rendered by `tokenizer/train/emit.mojo`:
    `addresses` `[token_bytes(u8), lengths(i64), merge_left(i64),
    merge_right(i64)]`, `dims` `[n_tokens, n_merges, which]`, `which` 0 for
    the rank file, 1 for `tokenizer.json`."""
    var n = _index(dims[0])
    var nm = _index(dims[1])
    var which = _index(dims[2])
    var v = TrainedVocabulary()
    v.length = _i64_list(_index(addresses[1]) if n > 0 else 0, n)
    var total = _bytes_total(v.length)
    v.arena = _u8_list(_index(addresses[0]) if total > 0 else 0, total)
    var at = 0
    for k in range(n):
        v.offset.append(at)
        at += v.length[k]
    v.merge_left = _i64_list(_index(addresses[2]) if nm > 0 else 0, nm)
    v.merge_right = _i64_list(_index(addresses[3]) if nm > 0 else 0, nm)
    for k in range(nm):
        if v.merge_left[k] < 0 or v.merge_left[k] >= n or v.merge_right[k] < 0 or v.merge_right[k] >= n:
            raise Error("ValueError: merge " + String(k) + " names an id outside [0, " + String(n) + ")")
    if which == 0:
        return PythonObject(render_ranks(v))
    return PythonObject(render_tokenizer_json(v, String(GPT2_PAT_STR), String(GPT2_ENDOFTEXT)))


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


def bpe_encode_corpus_binding(
    handle: PythonObject, addresses: PythonObject, dims: PythonObject
) raises -> PythonObject:
    """`lm_corpus._tokenize`'s work in one call: cut the corpus into
    documents (`_cut_documents`), encode each ALONE (`allow_endoftext`
    False), ids back to back. `addresses` `[data(u8), points(i64),
    ids_out(i32), bounds_out(i64), stats_out(i64)]`, `dims` `[n_bytes,
    n_points, document_bytes, ids_cap]`. `points` are the sorted split
    points (0 and n_bytes included); `bounds_out[j]` is the id offset where
    point j starts. `stats_out` gets `[n_documents, max_id, ids_above_255]`.
    Returns the id count."""
    var owner = handle.downcast_value_ptr[BpeHandle]()
    _require_loaded(Bool(owner[].tok))
    var n = _index(dims[0])
    var n_points = _index(dims[1])
    var document_bytes = _index(dims[2])
    var cap = _index(dims[3])
    if document_bytes < 1:
        raise Error("ValueError: document_bytes must be >= 1")
    var data_address = _index(addresses[0]) if n > 0 else 0
    var points = _i64_list(_index(addresses[1]), n_points)
    var ids_address = _index(addresses[2]) if cap > 0 else 0
    var bounds_address = _index(addresses[3])
    var stats_address = _index(addresses[4])
    var total = 0
    with GILReleased(Python()):
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
    return PythonObject(total)


def bpe_train_corpus_binding(addresses: PythonObject, dims: PythonObject) raises -> PythonObject:
    """`lm_corpus`'s vocabulary training in one call: the documents of
    `[lo, hi)` cut by `_cut_documents`, trained by `train_bpe`.
    `addresses` `[data(u8), points(i64)]`, `dims` `[n_bytes, n_points, lo,
    hi, document_bytes, vocab_size, min_frequency, break_ties_high]`."""
    var n = _index(dims[0])
    var n_points = _index(dims[1])
    var lo = _index(dims[2])
    var hi = _index(dims[3])
    var document_bytes = _index(dims[4])
    var vocab_size = _index(dims[5])
    var min_frequency = _index(dims[6])
    var reverse = _flag(dims[7], "break_ties_high")
    if lo < 0 or hi > n or lo > hi:
        raise Error("ValueError: bpe_train_corpus: [lo, hi) must lie inside the corpus")
    if document_bytes < 1:
        raise Error("ValueError: document_bytes must be >= 1")
    var data_address = _index(addresses[0]) if n > 0 else 0
    var points = _i64_list(_index(addresses[1]), n_points)
    var handle = BpeTrainedHandle()
    with GILReleased(Python()):
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
    return PythonObject(alloc=handle^)


def tokens_gather_rows_binding(addresses: PythonObject, dims: PythonObject) raises -> PythonObject:
    """`lm_corpus.TokenBatches.ids`: row b of step k reads int32 ids
    `[lo + (base + b*length) % modulus : + length + 1]`, `base` =
    `k*batch*length`. `addresses` `[src(i32), out(i32)]`, `dims` `[n_src,
    batch, length, lo, modulus, base]`."""
    var n_src = _index(dims[0])
    var batch = _index(dims[1])
    var length = _index(dims[2])
    var lo = _index(dims[3])
    var modulus = _index(dims[4])
    var base = _index(dims[5])
    if batch < 0 or length < 1 or modulus < 1 or lo < 0 or base < 0:
        raise Error("ValueError: tokens_gather_rows: bad dims")
    var width = length + 1
    if lo + modulus - 1 + width > n_src:
        raise Error("ValueError: tokens_gather_rows: a row runs past the id array")
    if batch == 0:
        return PythonObject(0)
    var src = i32_ptr(_index(addresses[0]))
    var dst = i32_ptr(_index(addresses[1]))
    for b in range(batch):
        var start = lo + (base + b * length) % modulus
        memcpy(dest=dst + b * width, src=src + start, count=width)
    return PythonObject(batch * width)


def tokenizer_pretokenize_binding(
    text_addr: PythonObject, n_bytes: PythonObject, pattern: PythonObject, out_addr: PythonObject
) raises -> PythonObject:
    """Pre-token boundaries of `n_bytes` uint8 under `pattern` (0 GPT-2,
    1 Llama 3, 2 Qwen 2) as int64 at `out_addr` (`n_bytes + 1` suffices).
    Returns how many."""
    var n = _index(n_bytes)
    var text = _u8_list(_index(text_addr) if n > 0 else 0, n)
    var bounds = pretokenize_pattern(text, builtin_unicode_classes(), _index(pattern))
    var dst = _i64_ptr(_index(out_addr))
    for k in range(len(bounds)):
        dst[k] = Int64(bounds[k])
    return PythonObject(len(bounds))


def _specials(addresses: PythonObject, first: Int, n_specials: Int,
              mut arena: List[UInt8], mut lengths: List[Int], mut ids: List[Int]) raises:
    lengths = _i64_list(_index(addresses[first + 1]) if n_specials > 0 else 0, n_specials)
    var total = _bytes_total(lengths)
    arena = _u8_list(_index(addresses[first]) if total > 0 else 0, total)
    ids = _i64_list(_index(addresses[first + 2]) if n_specials > 0 else 0, n_specials)


def vocab_load_bytes_binding(addresses: PythonObject, dims: PythonObject) raises -> PythonObject:
    """`models.Tokenizer(tokens, merges, ...)`: `addresses` `[token_bytes,
    token_lengths, merge_bytes, merge_lengths (two per merge), special_bytes,
    special_lengths, special_ids]`, `dims` `[n_tokens, n_merges,
    n_specials, pattern, ignore_merges]`."""
    var n = _index(dims[0])
    var nm = _index(dims[1])
    var ns = _index(dims[2])
    var pattern = _index(dims[3])
    var ignore = _index(dims[4]) != 0
    var lengths = _i64_list(_index(addresses[1]) if n > 0 else 0, n)
    var total = _bytes_total(lengths)
    var arena = _u8_list(_index(addresses[0]) if total > 0 else 0, total)
    var table = table_from_tokens(arena, lengths)
    var mlen = _i64_list(_index(addresses[3]) if nm > 0 else 0, 2 * nm)
    var mtotal = _bytes_total(mlen)
    var marena = _u8_list(_index(addresses[2]) if mtotal > 0 else 0, mtotal)
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
    var sp_arena = List[UInt8]()
    var sp_lengths = List[Int]()
    var sp_ids = List[Int]()
    _specials(addresses, 4, ns, sp_arena, sp_lengths, sp_ids)
    var tok = VocabTokenizer(table^, builtin_unicode_classes(), pattern)
    tok.set_specials(sp_arena, sp_lengths, sp_ids)
    var handle = VocabHandle()
    handle.tok = tok^
    return PythonObject(alloc=handle^)


def vocab_load_spelled_binding(
    vocab: PythonObject, merges: PythonObject, addresses: PythonObject, dims: PythonObject
) raises -> PythonObject:
    """`models.Tokenizer._from_json`: `vocab` the `model.vocab` object with
    the added tokens taken out, `merges` the `model.merges` list,
    `addresses` `[special_bytes, special_lengths, special_ids]`, `dims`
    `[n_specials, pattern, ignore_merges]`."""
    var ns = _index(dims[0])
    var pattern = _index(dims[1])
    var ignore = _index(dims[2]) != 0
    var table = _table_from_spelled(vocab, merges, False, ignore)
    var sp_arena = List[UInt8]()
    var sp_lengths = List[Int]()
    var sp_ids = List[Int]()
    _specials(addresses, 0, ns, sp_arena, sp_lengths, sp_ids)
    var tok = VocabTokenizer(table^, builtin_unicode_classes(), pattern)
    tok.set_specials(sp_arena, sp_lengths, sp_ids)
    var handle = VocabHandle()
    handle.tok = tok^
    return PythonObject(alloc=handle^)


def vocab_info_binding(handle: PythonObject) raises -> PythonObject:
    """`[n_ranks, n_vocab, max_token_bytes]`."""
    var owner = handle.downcast_value_ptr[VocabHandle]()
    if not owner[].tok:
        raise Error("tokenizer host: the handle holds no vocabulary")
    var out = Python.list()
    out.append(PythonObject(owner[].tok.value().n_ranks()))
    out.append(PythonObject(owner[].tok.value().n_vocab()))
    out.append(PythonObject(owner[].tok.value().max_token_bytes()))
    return out


def vocab_encode_binding(
    handle: PythonObject, text_addr: PythonObject, n_bytes: PythonObject,
    out_addr: PythonObject, out_cap: PythonObject, allow_special: PythonObject
) raises -> PythonObject:
    """`models.Tokenizer.encode_bytes`: ids as int32 at `out_addr`
    (`out_cap = n_bytes` suffices). Returns the count."""
    var owner = handle.downcast_value_ptr[VocabHandle]()
    if not owner[].tok:
        raise Error("tokenizer host: the handle holds no vocabulary")
    var n = _index(n_bytes)
    var cap = _index(out_cap)
    var allow = _flag(allow_special, "allow_special")
    if n <= 0:
        return PythonObject(0)
    var text = _u8_list(_index(text_addr), n)
    var ids = owner[].tok.value().encode(text, allow)
    if len(ids) > cap:
        raise Error("vocab_encode: " + String(len(ids)) + " ids do not fit " + String(cap))
    var dst = i32_ptr(_index(out_addr))
    for k in range(len(ids)):
        dst[k] = Int32(ids[k])
    return PythonObject(len(ids))


def vocab_pretokenize_binding(
    handle: PythonObject, text_addr: PythonObject, n_bytes: PythonObject, out_addr: PythonObject
) raises -> PythonObject:
    var owner = handle.downcast_value_ptr[VocabHandle]()
    if not owner[].tok:
        raise Error("tokenizer host: the handle holds no vocabulary")
    var n = _index(n_bytes)
    var text = _u8_list(_index(text_addr) if n > 0 else 0, n)
    var bounds = owner[].tok.value().pretokenize(text)
    var dst = _i64_ptr(_index(out_addr))
    for k in range(len(bounds)):
        dst[k] = Int64(bounds[k])
    return PythonObject(len(bounds))


def vocab_decode_binding(
    handle: PythonObject, seq: PythonObject, out_addr: PythonObject, out_cap: PythonObject
) raises -> PythonObject:
    """`models.Tokenizer.decode_bytes` over a Python sequence of ids; an
    added id renders its content, an id that is neither a rank nor an added
    token is refused by value and position."""
    var owner = handle.downcast_value_ptr[VocabHandle]()
    if not owner[].tok:
        raise Error("tokenizer host: the handle holds no vocabulary")
    var ids = _ids_from_seq(seq)
    var out = List[UInt8]()
    for k in range(len(ids)):
        owner[].tok.value().append_decoded(out, ids[k], k)
    var cap = _index(out_cap)
    if len(out) > cap:
        raise Error("vocab_decode: " + String(len(out)) + " bytes do not fit " + String(cap))
    if len(out) > 0:
        memcpy(dest=_u8_ptr(_index(out_addr)), src=out.unsafe_ptr(), count=len(out))
    return PythonObject(len(out))


@export
def PyInit__mojolearn_tokenizer_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_tokenizer_host")
        module.def_function[tokenizer_host_numeric_mode_binding]("tokenizer_host_numeric_mode")
        module.def_function[tokenizer_host_vendor_binding]("tokenizer_host_vendor")
        module.def_function[tokenizer_host_column_binding]("tokenizer_host_column")
        module.def_function[tokenizer_host_sabotage_binding]("tokenizer_host_sabotage")
        _ = module.add_type[BpeHandle]("_BpeHandle")
        module.def_function[bpe_load_binding]("bpe_load")
        module.def_function[bpe_n_vocab_binding]("bpe_n_vocab")
        module.def_function[bpe_max_token_bytes_binding]("bpe_max_token_bytes")
        module.def_function[bpe_encode_binding]("bpe_encode")
        module.def_function[bpe_encode_batch_binding]("bpe_encode_batch")
        module.def_function[bpe_decode_binding]("bpe_decode")
        _ = module.add_type[BpeTrainedHandle]("_BpeTrainedHandle")
        module.def_function[bpe_train_binding]("bpe_train")
        module.def_function[bpe_trained_sizes_binding]("bpe_trained_sizes")
        module.def_function[bpe_trained_copy_binding]("bpe_trained_copy")
        module.def_function[bpe_load_tokens_binding]("bpe_load_tokens")
        module.def_function[bpe_load_spelled_binding]("bpe_load_spelled")
        module.def_function[bpe_ranks_text_binding]("bpe_ranks_text")
        module.def_function[bpe_decode_seq_binding]("bpe_decode_seq")
        module.def_function[bpe_decode_batch_binding]("bpe_decode_batch")
        module.def_function[bpe_render_binding]("bpe_render")
        module.def_function[bpe_encode_corpus_binding]("bpe_encode_corpus")
        module.def_function[bpe_train_corpus_binding]("bpe_train_corpus")
        module.def_function[tokens_gather_rows_binding]("tokens_gather_rows")
        module.def_function[tokenizer_pretokenize_binding]("tokenizer_pretokenize")
        _ = module.add_type[VocabHandle]("_VocabHandle")
        module.def_function[vocab_load_bytes_binding]("vocab_load_bytes")
        module.def_function[vocab_load_spelled_binding]("vocab_load_spelled")
        module.def_function[vocab_info_binding]("vocab_info")
        module.def_function[vocab_encode_binding]("vocab_encode")
        module.def_function[vocab_pretokenize_binding]("vocab_pretokenize")
        module.def_function[vocab_decode_binding]("vocab_decode")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_tokenizer_host: ", error))

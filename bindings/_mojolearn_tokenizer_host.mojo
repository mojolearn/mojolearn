# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the byte-level BPE tokenizer in the GPT-2 format
(`tokenizer/`), the expose-tokenizer lane, 2026-09-14
(docs/lanes/BRIEF_expose_tokenizer_2026-09-14.md).

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
from max.algorithm import sync_parallelize

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
    BpeTokenizer,
    load_bpe_tokenizer_from,
)
from tokenizer.impl.unicode_class import builtin_unicode_classes
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
        var offs = _i64_ptr(offsets_address)
        if Int(offs[0]) != 0 or Int(offs[n_docs]) != n:
            raise Error(
                "bpe_encode_batch: offsets must start at 0 and end at n_bytes "
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
            sync_parallelize(_documents, tasks)
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
        var offs = _i64_ptr(offsets_address)
        if Int(offs[0]) != 0 or Int(offs[n_docs]) != n:
            raise Error(
                "bpe_train: offsets must start at 0 and end at n_bytes "
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
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_tokenizer_host: ", error))

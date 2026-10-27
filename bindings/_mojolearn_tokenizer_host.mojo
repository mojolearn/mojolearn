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
`bpe_encode_batch`'s (bytes back to back, int64 lengths) and returns an
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
from bindings.py_balanced import py_call as _balanced_call
from bindings.sha256_door import sha256_hexdigest
from bindings.py_balanced import py_call, py_contains, py_get, py_setattr, py_setitem, py_type
from std.python._cpython import GILReleased, PyObjectPtr
from std.python.bindings import PythonModuleBuilder, ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from std.sys.compile import is_defined

from core.hostptr import i32_ptr
from bindings.native_json import native_json_parse, native_json_read, native_path, native_path_exists
from bindings.causal_model_config import model_config_c
from bindings.causal_checkpoint import checkpoint_c
from bindings.causal_lm_construct import causal_model_c
from bindings.neural_block_public import neural_block_c
from bindings.mamba_public_native import mamba_native_c, MambaDecodeHandle
from bindings.samba_config_native import samba_config_c
from bindings.samba_public_native import samba_public_c
from bindings.native_transport import transport_c, transport_initialize
from bindings.cross_vendor_native import cross_vendor_c
from bindings.parallel_byte_native import parallel_byte_c
from bindings.native_codec import codec_c
from bindings.native_process_pool import process_c
from bindings.parallel_mlp_native import parallel_mlp_c
from bindings.small_mlp_native import small_mlp_c
from bindings.parallel_neural_native import parallel_neural_c
from bindings.parallel_forecast_native import parallel_forecast_c
from bindings.parallel_shard_native import parallel_shard_c
from bindings.neighbor_query_native import neighbor_query_c, neighbor_query_attribute_c
from bindings.parallel_causal_native import parallel_causal_c, RemoteStateLifetime
from bindings.optimizer_lifecycle import optimizer_finalize_descriptor_binding
from bindings.optimizer_native_router import optimizer_native_c
from bindings.embedding_layer_native import embedding_layer_lifetime_c
from bindings.optimizer_device_native import device_finalize_c
from bindings.runtime_sequential_forward import runtime_sequential_forward_c
from bindings.runtime_classical_unavailable import runtime_classical_unavailable_c
from bindings.xseq_native import xseq_c
from bindings.byte_lm_host_trainer_native import byte_lm_host_trainer_c
from bindings.byte_lm_trainer_native import byte_lm_trainer_c, byte_lm_pooled_finalize_c
from bindings.xcnn_native import xcnn_c
from bindings.tsa_public_native import tsa_c
from bindings.classical_attributes import classical_attribute_c, classical_attribute_descriptor_c
from bindings.mode_metadata_native import mode_metadata_c
from bindings.params_metadata_native import params_metadata_c
from bindings.public_attributes_native import public_attribute_c
from bindings.linear_constructor_native import linear_constructor_c
from bindings.full_reset_params_native import full_reset_params_c
from bindings.full_reset_constructor_native import full_reset_constructor_c
from bindings.array_native import array_native_c
from bindings.params_setter_native import params_setter_c
from bindings.declared_params_native import declared_params_c
from bindings.arena_rows_native import arena_rows_c
from bindings.arena_prog_native import arena_prog_c
from bindings.generator_draw_native import generator_draw_c
# the user-object reader (lane purge-user-objects): label lists, index pairs, group ids
from bindings.user_objects_native import user_objects_c
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
from tokenizer.impl.unicode_class import builtin_unicode_classes, UNICODE_VERSION_PINNED
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
)
from tokenizer.impl.tokenizer_host_shell_impl import (
    TOKENIZER_HOST_SABOTAGE,
    TOKENIZER_BATCH_SABOTAGE,
    BpeHandle,
    BpeTrainedHandle,
    VocabHandle,
    SPELLED_OK,
    SPELLED_BAD_ID,
    SPELLED_NOT_STR,
    SPELLED_UNSPELL,
    SPELLED_RAISED,
    SPELLED_NOT_PAIR,
    SPELLED_NOT_LEVEL,
    SpelledFailure,
    SpelledMerges,
    SpelledVocab,
    _bytes_total,
    _handle_from_table,
    _i64_list,
    _i64_ptr,
    _u8_list,
    _u8_ptr,
    bpe_check_ids,
    bpe_decode_batch_append,
    bpe_decode_batch_fit,
    bpe_decode_seq_bytes,
    bpe_encode_batch_into,
    bpe_encode_corpus_into,
    bpe_encode_into,
    bpe_handle_load,
    bpe_train_corpus_fill,
    bpe_train_fill,
    bpe_trained_copy_into,
    spelled_merges_list,
    spelled_merges_text,
    spelled_vocab_tokens,
    tokenizer_append_bytes,
    tokenizer_store_i64,
    tokens_gather_rows_into,
    tokens_gather_rows_width,
    trained_vocab_settle,
    vocab_check_pairs,
    vocab_decode_ids,
    vocab_encode_ids,
    vocab_fingerprint_hex,
)
from bindings.generation_buffers import generation_initialize_binding, GENERATION_BOUNDARY
from providers.python.generation_session_pyapi import GenerationNativeProvider, GenerationNativeSession
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, NativeDescriptorResource, neural_descriptor_info
from providers.python.operation_lock_pyapi import ProviderOperationLock, operation_lock_connect, operation_lock_c
from bindings.causal_lm_public import (
    GenerationHandle, causal_public_c, causal_lm_allocate_binding,
    causal_lm_reset_binding, causal_lm_forward_binding, causal_lm_stream_binding,
    causal_lm_read_binding, causal_lm_info_binding, causal_lm_generate_binding,
    causal_lm_text_stream_binding, causal_lm_generate_text_binding, causal_lm_resident_binding,
    causal_lm_snapshot_binding, causal_lm_resume_binding,
    causal_lm_parameters_binding, causal_lm_blocks_binding,
)

def _index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("tokenizer host: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    # operator.index through the balanced call: PythonObject.__call__ keeps one
    # reference on each argument (bindings/py_balanced.mojo), which leaked every
    # buffer address passed to a host entry.
    return Int(py=_balanced_call(operator_module.index, value))


def _flag(value: PythonObject, name: String) raises -> Bool:
    var type_name = String(py=value.__class__.__name__)
    if type_name != "bool":
        raise Error(
            "tokenizer host: " + name + " must be a bool, got " + type_name
        )
    return Bool(py=value)


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


def tokenizer_host_runtime_classical_abi_binding() raises -> PythonObject:
    # Import-time ownership witness: this module no longer registers
    # TreesResident, so its canonical native owner may safely initialize.
    return PythonObject(1)


def bpe_load_binding(ranks_path: PythonObject) raises -> PythonObject:
    """Parse the caller's rank file once into a handle. Every refusal in
    `tokenizer/impl/ranks.mojo::load_rank_table` (a rank out of order, an
    odd hex field, a duplicate token) and
    `unicode_class.mojo::builtin_unicode_classes` (a generated table that is
    not the pin) raises here with its own sentence."""
    var rp = String(py=ranks_path)
    var handle = BpeHandle()
    with GILReleased(Python()):
        bpe_handle_load(handle, rp)
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
        count = bpe_encode_into(owner[], text_address, n, out_address, cap, allow)
    return PythonObject(count)


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
    Reads the concatenated `n_bytes` uint8 at `text_addr` and the `n_docs`
    int64 document LENGTHS at `offsets_addr` (summing to `n_bytes`; the
    running offsets are taken here, lane/pyglue-text-io). Each document
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
        total = bpe_encode_batch_into(
            owner[], text_address, offsets_address, out_address, counts_address,
            n_docs, n, cap, allow,
        )
    return PythonObject(total)


# TOMBSTONE: bpe_decode, causal_lm_admit, neural_prefill_call, optimizer_finalize exports (dead native entries: no live caller of the export) deleted 2026-10-26 by 5206fc31b; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_tokenizer_host.patch


def bpe_train_binding(
    text_addr: PythonObject,
    offsets_addr: PythonObject,
    dims: PythonObject,
) raises -> PythonObject:
    """`tokenizer/train/bpe_train.mojo::train_bpe` on the host
    (lane/bpe-builder-native, 2026-09-18): TRAIN a vocabulary.

    `dims` is `[n_docs, n_bytes, vocab_size, min_frequency,
    break_ties_high]`. Reads the concatenated `n_bytes` uint8 at `text_addr`
    and the `n_docs` int64 document LENGTHS at `offsets_addr` (summing to
    `n_bytes`), and each document is pre-tokenized ALONE, exactly
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
        bpe_train_fill(
            handle, text_address, offsets_address, n_docs, n, vocab_size,
            min_frequency, reverse,
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
    _ = py_call(out.append, PythonObject(v.n_tokens()))
    _ = py_call(out.append, PythonObject(len(v.arena)))
    _ = py_call(out.append, PythonObject(v.n_merges()))
    _ = py_call(out.append, PythonObject(v.n_ties_broken))
    _ = py_call(out.append, PythonObject(v.n_groups))
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
    bpe_trained_copy_into(v, arena_address, lengths_address, nm, left_address, right_address)
    return PythonObject(v.n_tokens())


# ------------------------------------------------------------------------
# lane/pyglue-text-io (2026-10-03): the work the Python doors did over the
# caller's data, moved here. `tokenizer.py` and `models/tokenizer.py` now
# only validate arguments, pass the caller's objects or buffers and return
# results. Refusals raised here start with the Python exception name the
# door re-raises them as (`ValueError: `, `TypeError: `).


def _py_repr(obj: PythonObject) raises -> String:
    var bi = Python.import_module("builtins")
    return String(py=py_call(bi.repr, obj))


def _py_str_bytes(text: PythonObject, mut out: List[UInt8], mut err: String) raises -> Int:
    """A Python object's spelling bytes: 0 a str (bytes in `out`), 1 not a
    str, 2 the conversion raised (its message in `err`)."""
    if String(py=text.__class__.__name__) != "str":
        return 1
    try:
        out = string_bytes(String(py=text))
    except error:
        err = String(error)
        return 2
    return 0


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
    `impl/vocab_build.mojo` runs in `tokenizer_host_shell_impl.mojo`; this
    door reads the Python items into typed records and names a refusal by
    the item's Python repr."""
    var bi = Python.import_module("builtins")
    var items = py_call(bi.list, vocab.items())
    var n = Int(py=py_call(bi.len, items))
    var inv = codepoint_to_byte()
    var sv = SpelledVocab(n)
    var key_err = List[String](capacity=n)
    for k in range(n):  # cpu-route: Python objects marshaled one element at a time before the runtime
        var key = items[k][0]
        var val = items[k][1]
        var is_int = String(py=val.__class__.__name__) == "int"
        sv.id_is_int.append(is_int)
        sv.ids.append(Int(py=val) if is_int else -1)
        var kb = List[UInt8]()
        var ke = String("")
        sv.key_kind.append(_py_str_bytes(key, kb, ke))
        sv.keys.append(kb^)
        key_err.append(ke^)
    var arena = List[UInt8]()
    var lengths = List[Int](capacity=n)
    var vf = spelled_vocab_tokens(sv, inv, arena, lengths)
    if vf.code != SPELLED_OK:
        var key = items[vf.k][0]
        if vf.code == SPELLED_BAD_ID:
            raise Error(
                "ValueError: id "
                + _py_repr(items[vf.k][1])
                + " of "
                + _py_repr(key)
                + " is not a unique id in [0, "
                + String(n)
                + ")"
            )
        if vf.code == SPELLED_NOT_STR:
            raise Error("ValueError: spelling " + _py_repr(key) + " is not a string")
        if vf.code == SPELLED_RAISED:
            raise Error(key_err[vf.k])
        raise Error(unspell_error("spelling " + _py_repr(key), vf.bad))
    var table = table_from_tokens(arena, lengths)
    var chk = MergeCheck(n)
    if String(py=merges.__class__.__name__) == "str":
        spelled_merges_text(table, string_bytes(String(py=merges)), inv, version_line, chk)
    else:
        var m = Int(py=py_call(bi.len, merges))
        var sm = SpelledMerges(m)
        var errs = List[String](capacity=3 * m)
        for k in range(m):  # cpu-route: Python objects marshaled one element at a time before the runtime
            var item = merges[k]
            var it = String(py=item.__class__.__name__)
            var kind = 2
            if it == "str":
                kind = 0
            elif (it == "list" or it == "tuple") and Int(py=py_call(bi.len, item)) == 2:
                kind = 1
            sm.kind.append(kind)
            var tb = List[UInt8]()
            var te = String("")
            var tk = 0
            if kind == 0:
                tk = _py_str_bytes(item, tb, te)
            sm.str_kind.append(tk)
            sm.text.append(tb^)
            errs.append(te^)
            comptime for p in range(2):
                var pb = List[UInt8]()
                var pe = String("")
                var pk = 0
                if kind == 1:
                    pk = _py_str_bytes(item[p], pb, pe)
                sm.part_kind.append(pk)
                sm.parts.append(pb^)
                errs.append(pe^)
        var mf = spelled_merges_list(table, sm, inv, chk)
        if mf.code != SPELLED_OK:
            var item = merges[mf.k]
            var label = "merge " + String(mf.k)
            if mf.code == SPELLED_NOT_PAIR:
                raise Error("ValueError: " + label + " " + _py_repr(item) + " is not a pair")
            if mf.code == SPELLED_NOT_LEVEL:
                raise Error(
                    "ValueError: " + label + " names " + _py_repr(item) + ", not a byte-level spelling"
                )
            if sm.kind[mf.k] == 0:
                raise Error(errs[3 * mf.k])
            var part = item[mf.part]
            if mf.code == SPELLED_NOT_STR:
                raise Error("ValueError: " + label + " names " + _py_repr(part) + " is not a string")
            if mf.code == SPELLED_RAISED:
                raise Error(errs[3 * mf.k + 1 + mf.part])
            raise Error(unspell_error(label + " names " + _py_repr(part), mf.bad))
    check_loose(table, chk, ignore_merges)
    return table^


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
    var n = Int(py=py_call(bi.len, seq))
    var out = List[Int](capacity=n)
    for k in range(n):  # cpu-route: Python objects marshaled one element at a time before the runtime
        var v = seq[k]
        var tn = String(py=v.__class__.__name__)
        if tn == "bool":
            raise Error("TypeError: ids must be int, not bool, at position " + String(k))
        var iv = 0
        try:
            iv = Int(py=py_call(operator_module.index, v))
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
    bpe_check_ids(ids, nv)
    var cap = _index(out_cap)
    if len(ids) == 0:
        return PythonObject(0)
    var out = bpe_decode_seq_bytes(owner[], ids, cap)
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
    var n = Int(py=py_call(bi.len, batch))
    var cap = _index(out_cap)
    var nv = owner[].tok.value().n_vocab()
    var joined = List[UInt8]()
    var lens = List[Int](capacity=n)
    for s in range(n):  # cpu-route: Python objects marshaled one element at a time before the runtime
        bpe_decode_batch_append(owner[], _ids_from_seq(batch[s]), nv, joined, lens)
    bpe_decode_batch_fit(joined, cap)
    if n > 0:
        tokenizer_store_i64(_i64_ptr(_index(lengths_addr)), lens)
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
    v.merge_left = _i64_list(_index(addresses[2]) if nm > 0 else 0, nm)
    v.merge_right = _i64_list(_index(addresses[3]) if nm > 0 else 0, nm)
    trained_vocab_settle(v, n, nm)
    if which == 0:
        return PythonObject(render_ranks(v))
    return PythonObject(render_tokenizer_json(v, String(GPT2_PAT_STR), String(GPT2_ENDOFTEXT)))


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
        total = bpe_encode_corpus_into(
            owner[], data_address, points, n, n_points, document_bytes, cap,
            ids_address, bounds_address, stats_address,
        )
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
        bpe_train_corpus_fill(
            handle, data_address, points, lo, hi, document_bytes, vocab_size,
            min_frequency, reverse,
        )
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
    var width = tokens_gather_rows_width(n_src, batch, length, lo, modulus, base)
    if batch == 0:
        return PythonObject(0)
    var src = i32_ptr(_index(addresses[0]))
    var dst = i32_ptr(_index(addresses[1]))
    return PythonObject(tokens_gather_rows_into(src, dst, batch, length, width, lo, modulus, base))


def _py_bytes_list(raw: PythonObject) raises -> List[UInt8]:
    """The bytes of a Python bytes-like object, one memcpy through an
    `array('B')` buffer (no Python loop over the bytes)."""
    var arr = py_call(Python.import_module("array").array, "B", raw)
    var n = len(arr)
    var out = List[UInt8](length=n, fill=UInt8(0))
    if n > 0:
        memcpy(dest=out.unsafe_ptr(), src=_u8_ptr(Int(py=arr.buffer_info()[0])), count=n)
        _ = arr  # keep the Python object alive through the native call above
    # Keepalive: Mojo drops `arr` at its last named use (`arr.buffer_info()`),
    # which freed the array before the memcpy read its buffer; small tokens
    # then read the allocator's free-list bytes (ValueError "token 1 repeats
    # token 0 (80)" from `models.Tokenizer`). Hold it until the copy is done.
    _ = arr^
    return out^


def _py_bytes(values: List[UInt8]) raises -> PythonObject:
    """`values` as a Python bytes object, one memcpy into an `array('B')`."""
    var bi = Python.import_module("builtins")
    var arr = py_call(Python.import_module("array").array, "B", py_call(bi.bytes, len(values)))
    if len(values) > 0:
        memcpy(dest=_u8_ptr(Int(py=arr.buffer_info()[0])), src=values.unsafe_ptr(), count=len(values))
        _ = arr  # keep the Python object alive through the native call above
    return arr.tobytes()


def _py_int_list(values: List[Int]) raises -> PythonObject:
    """`values` as a Python list of ints, through an `array('q')` filled
    by `tokenizer_store_i64` (no Python loop over the values)."""
    var bi = Python.import_module("builtins")
    var arr = py_call(Python.import_module("array").array, "q", py_call(bi.bytes, 8 * len(values)))
    if len(values) > 0:
        tokenizer_store_i64(_i64_ptr(Int(py=arr.buffer_info()[0])), values)
        _ = arr  # keep the Python object alive through the native call above
    return arr.tolist()


def vocab_encode_public_binding(
    tokenizer: PythonObject, input_text: PythonObject, add_special: PythonObject,
    allow_special: PythonObject, bytes_only: PythonObject,
) raises -> PythonObject:
    var text = input_text
    var bi = Python.import_module("builtins")
    var owner = tokenizer._handle.downcast_value_ptr[VocabHandle]()
    if not owner[].tok:
        raise Error("tokenizer: vocabulary is not loaded")
    var is_text = Bool(py=py_call(bi.isinstance, text, bi.str))
    if is_text and Bool(py=bytes_only):
        raise Error("tokenizer: encode_bytes takes bytes; use encode for str")
    if is_text:
        if tokenizer.normalizer is not None and String(tokenizer.normalizer) == "NFC":
            # CPython's Unicode normalization is a compiled native extension.
            # Keep its existing Unicode version and semantics; no Python
            # tokenizer method or normalization loop is invoked.
            var normalize = Python.import_module("unicodedata").normalize
            if py_type(normalize) is not py_type(bi.len):
                raise Error("tokenizer: NFC normalization requires the native Unicode entry")
            text = py_call(normalize, "NFC", text)
        text = py_call(text.encode, "utf-8")
    else:
        text = py_call(bi.memoryview, text).tobytes()
    var bytes = _py_bytes_list(text)
    var ids = vocab_encode_ids(owner[].tok.value(), bytes, Bool(py=allow_special))
    var out = bi.list()
    if Bool(py=add_special):
        _ = py_call(out.extend, tokenizer.bos_ids)
    _ = py_call(out.extend, _py_int_list(ids))
    if Bool(py=add_special):
        _ = py_call(out.extend, tokenizer.eos_ids)
    return out


def vocab_decode_public_binding(tokenizer: PythonObject, seq: PythonObject, bytes_only: PythonObject, errors: PythonObject) raises -> PythonObject:
    var owner = tokenizer._handle.downcast_value_ptr[VocabHandle]()
    if not owner[].tok:
        raise Error("tokenizer: vocabulary is not loaded")
    var bi = Python.import_module("builtins")
    if Bool(py=py_call(bi.isinstance, seq, bi.str)) or Bool(py=py_call(bi.isinstance, seq, bi.bytes)) or Bool(py=py_call(bi.isinstance, seq, bi.bytearray)):
        raise Error("tokenizer: ids must be an integer sequence")
    var ids = _ids_from_seq(seq)
    var out = vocab_decode_ids(owner[].tok.value(), ids)
    var raw = _py_bytes(out)
    if Bool(py=bytes_only):
        return py_call(bi.bytes, raw)
    return py_call(raw.decode, "utf-8", errors)


def sha256_hex_binding(value: PythonObject) raises -> PythonObject:
    """The SHA-256 of a bytes-like value, 64 lowercase hex digits (core/sha256.mojo)."""
    return sha256_hexdigest(value)


def vocab_fingerprint_binding(tokenizer: PythonObject) raises -> PythonObject:
    var owner = tokenizer._handle.downcast_value_ptr[VocabHandle]()
    if not owner[].tok:
        raise Error("tokenizer: vocabulary is not loaded")
    ref tok = owner[].tok.value()
    var bi = Python.import_module("builtins")
    var tail = List[UInt8]()
    # Pre-tokenization uses our compiled Unicode table, independent of the host
    # Python. Hashing unicodedata.unidata_version here made identical tokens
    # and generation checkpoints disagree across Python 3.11/3.12/3.13.
    # NFC additionally uses the host's native normalizer, so retain its version
    # only when that operation is enabled. See test_tokenizer_fingerprint.py.
    var unicode_version = PythonObject(UNICODE_VERSION_PINNED)
    if tokenizer.normalizer is not None:
        unicode_version = PythonObject(String(UNICODE_VERSION_PINNED) + "/normalizer:" + String(tokenizer.unicode_version))
    var settings = [tokenizer.pattern, tokenizer.normalizer, unicode_version, tokenizer.bos_ids, tokenizer.eos_ids]
    for i in range(5):  # small-loop(the five settings words of the fingerprint)
        tokenizer_append_bytes(tail, _py_bytes_list(py_call(py_call(bi.str, settings[i]).encode, "utf-8")))
        tokenizer_append_bytes(tail, _py_bytes_list(py_call(PythonObject("\n").encode, "ascii")))
    return PythonObject(vocab_fingerprint_hex(tok, tail))


def _vocab_pattern(pattern: PythonObject) raises -> Int:
    var name = String(pattern)
    if name == "gpt2":
        return 0
    if name == "llama3":
        return 1
    if name == "qwen2":
        return 2
    raise Error("tokenizer: unknown pattern " + name)


def _vocab_settle(tokenizer: PythonObject, options: PythonObject, module: PythonObject) raises:
    var bi = Python.import_module("builtins")
    _ = _vocab_pattern(options[0])
    if options[2] is not None and String(options[2]) != "NFC":
        raise Error("tokenizer: normalizer must be null or NFC")
    py_setattr(tokenizer, "pattern", options[0])
    py_setattr(tokenizer, "normalizer", options[2])
    py_setattr(tokenizer, "source", options[6])
    py_setattr(tokenizer, "added", bi.dict() if options[1] is None else py_call(bi.dict, options[1]))
    py_setattr(tokenizer, "_added_by_id", bi.dict())
    var keys = py_call(bi.list, tokenizer.added)
    for i in range(len(keys)):  # small-loop(added tokens: configuration entries, not input data)
        var content = keys[i]
        var id = tokenizer.added[content]
        if py_type(content) is not bi.str or len(content) == 0 or py_type(id) is not bi.int or Int(py=id) < 0:
            raise Error("tokenizer: added token must map nonempty str to nonnegative int")
        if py_contains(tokenizer._added_by_id, id):
            raise Error("tokenizer: added tokens share an id")
        py_setitem(tokenizer._added_by_id, id, content)
    var bos = bi.list()
    var eos = bi.list()
    var input = py_call(bi.list, options[3])
    for i in range(len(input)):  # small-loop(bos ids: configuration words)
        _ = py_call(bos.append, py_call(bi.int, input[i]))
    input = py_call(bi.list, options[4])
    for i in range(len(input)):  # small-loop(eos ids: configuration words)
        _ = py_call(eos.append, py_call(bi.int, input[i]))
    py_setattr(tokenizer, "bos_ids", py_call(bi.tuple, bos))
    py_setattr(tokenizer, "eos_ids", py_call(bi.tuple, eos))
    py_setattr(tokenizer, "_m", module)


def _vocab_finish(tokenizer: PythonObject, var tok: VocabTokenizer) raises:
    var bi = Python.import_module("builtins")
    var sp_arena = List[UInt8]()
    var sp_lengths = List[Int]()
    var sp_ids = List[Int]()
    var keys = py_call(bi.list, tokenizer.added)
    for i in range(len(keys)):  # small-loop(added tokens: configuration entries, not input data)
        var text = py_call(keys[i].encode, "utf-8")
        sp_lengths.append(len(text))
        sp_ids.append(Int(py=tokenizer.added[keys[i]]))
        tokenizer_append_bytes(sp_arena, _py_bytes_list(text))
    tok.set_specials(sp_arena, sp_lengths, sp_ids)
    py_setattr(tokenizer, "_n_ranks", PythonObject(tok.n_ranks()))
    py_setattr(tokenizer, "_n_vocab", PythonObject(tok.n_vocab()))
    py_setattr(tokenizer, "_max_token_bytes", PythonObject(tok.max_token_bytes()))
    var handle = VocabHandle()
    handle.tok = tok^
    py_setattr(tokenizer, "_handle", PythonObject(alloc=handle^))


def _token_bytes(value: PythonObject) raises -> List[UInt8]:
    var raw = PythonObject(None)
    try:
        raw = py_call(Python.import_module("builtins").memoryview, value).tobytes()
    except error:
        raise Error("TypeError: token/merge part must be bytes-like: " + String(error))
    return _py_bytes_list(raw)


def vocab_init_public_binding(tokenizer: PythonObject, tokens: PythonObject, merges: PythonObject,
                             options: PythonObject, module: PythonObject) raises -> PythonObject:
    _vocab_settle(tokenizer, options, module)
    var bi = Python.import_module("builtins")
    var entries = py_call(bi.list, tokens)
    var arena = List[UInt8]()
    var lengths = List[Int]()
    for i in range(len(entries)):  # cpu-route: Python objects marshaled one element at a time before the runtime
        var bytes = _token_bytes(entries[i])
        lengths.append(len(bytes))
        tokenizer_append_bytes(arena, bytes)
    var table = table_from_tokens(arena, lengths)
    var checks = MergeCheck(len(entries))
    var pairs = py_call(bi.list, merges)
    # Read the pairs up to the first one the conversion refuses; the merges
    # before it are checked first (`vocab_check_pairs`), then the refusal is
    # raised, the order the per-pair loop had.
    var lefts = List[List[UInt8]](capacity=len(pairs))
    var rights = List[List[UInt8]](capacity=len(pairs))
    var refused = String("")
    var failed = False
    for i in range(len(pairs)):  # cpu-route: Python objects marshaled one element at a time before the runtime
        try:
            if len(pairs[i]) != 2:
                raise Error("tokenizer: every merge must be a pair of bytes")
            var left = _token_bytes(pairs[i][0])
            var right = _token_bytes(pairs[i][1])
            lefts.append(left^)
            rights.append(right^)
        except error:
            refused = String(error)
            failed = True
            break
    vocab_check_pairs(table, lefts, rights, checks)
    if failed:
        raise Error(refused)
    check_loose(table, checks, Bool(py=options[5]))
    var tok = VocabTokenizer(table^, builtin_unicode_classes(), _vocab_pattern(options[0]))
    _vocab_finish(tokenizer, tok^)
    return PythonObject(None)


def vocab_spelled_public_binding(cls: PythonObject, vocab: PythonObject, merges: PythonObject,
                                options: PythonObject, module: PythonObject) raises -> PythonObject:
    var tokenizer = py_call(Python.import_module("builtins").object.__new__, cls)
    _vocab_settle(tokenizer, options, module)
    var table = _table_from_spelled(vocab, merges, False, Bool(py=options[5]))
    var tok = VocabTokenizer(table^, builtin_unicode_classes(), _vocab_pattern(options[0]))
    _vocab_finish(tokenizer, tok^)
    return tokenizer


def _vocab_config_pattern(pre: PythonObject, patterns: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    if pre is None:
        raise Error("tokenizer: pre_tokenizer is null; byte-level BPE requires a cut")
    var kind = String(py_get(pre, "type"))
    if kind == "ByteLevel":
        if Bool(py=py_get(pre, "use_regex", PythonObject(True))):
            return PythonObject("gpt2")
        raise Error("tokenizer: ByteLevel use_regex false is not a known cut")
    if kind == "Metaspace":
        raise Error("tokenizer: pre_tokenizer Metaspace is a SentencePiece family")
    if kind != "Sequence":
        raise Error("tokenizer: pre_tokenizer type " + kind + " is not a known cut")
    var found = PythonObject(None)
    var steps = py_get(pre, "pretokenizers", bi.list())
    for i in range(len(steps)):  # small-loop(pre_tokenizer steps: configuration structure)
        var step = steps[i]
        var t = String(py_get(step, "type"))
        if t == "Split":
            var pat = py_get(step, "pattern", bi.dict())
            if py_type(pat) is not bi.dict or py_get(pat, "Regex") is None:
                raise Error("tokenizer: Split pattern requires a Regex")
            var behavior = py_get(step, "behavior")
            if (behavior is not None and String(behavior) != "Isolated") or Bool(py=py_get(step, "invert", PythonObject(False))):
                raise Error("tokenizer: Split requires Isolated behavior without invert")
            var name = py_get(patterns, pat["Regex"])
            if name is None:
                raise Error("tokenizer: pre_tokenizer regex " + String(pat["Regex"]) + " is not one of the known patterns; refused by name")
            if found is not None and not Bool(py=found == name):
                raise Error("tokenizer: two Split steps with different patterns")
            found = name
        elif t == "ByteLevel":
            if Bool(py=py_get(step, "use_regex", PythonObject(False))):
                if found is not None and String(found) != "gpt2":
                    raise Error("tokenizer: Split followed by ByteLevel use_regex true cuts twice")
                found = PythonObject("gpt2")
        elif t == "Metaspace":
            raise Error("tokenizer: pre_tokenizer Metaspace is a SentencePiece family")
        else:
            raise Error("tokenizer: pre_tokenizer step " + t + " is not a known cut")
    if found is None:
        raise Error("tokenizer: pre_tokenizer Sequence names no cut")
    return found


def vocab_json_public_binding(cls: PythonObject, data: PythonObject, path: PythonObject,
                             cfg: PythonObject, module: PythonObject, patterns: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    if py_type(data) is not bi.dict:
        raise Error("tokenizer: not a JSON object")
    var model = py_get(data, "model")
    if model is None:
        model = bi.dict()
    if String(py_get(model, "type")) != "BPE":
        raise Error("tokenizer: model.type must be BPE; SentencePiece Unigram is unsupported")
    if Bool(py=py_get(model, "byte_fallback", PythonObject(False))):
        raise Error("tokenizer: model.byte_fallback identifies SentencePiece-style BPE")
    if Bool(py=py_get(model, "continuing_subword_prefix")) or Bool(py=py_get(model, "end_of_word_suffix")):
        raise Error("tokenizer: continuing_subword_prefix / end_of_word_suffix is not byte-level BPE")
    var dec = py_get(data, "decoder")
    if dec is not None and String(py_get(dec, "type")) != "ByteLevel":
        raise Error("tokenizer: decoder must be ByteLevel")
    var norm = py_get(data, "normalizer")
    var normalizer = PythonObject(None)
    if norm is not None:
        var kind = String(py_get(norm, "type"))
        if kind == "NFC":
            normalizer = PythonObject("NFC")
        elif kind == "Sequence":
            var steps = py_get(norm, "normalizers", bi.list())
            if len(steps) == 0:
                raise Error("tokenizer: empty normalizer Sequence")
            for i in range(len(steps)):  # small-loop(normalizer steps: configuration structure)
                if String(steps[i].get("type")) != "NFC":
                    raise Error("tokenizer: normalizer must be null or NFC")
            normalizer = PythonObject("NFC")
        else:
            raise Error("tokenizer: normalizer '" + kind + "' must be null or NFC")
    var pattern = _vocab_config_pattern(py_get(data, "pre_tokenizer"), patterns)
    var added = bi.dict()
    var entries = py_get(data, "added_tokens")
    if entries is not None:
        for i in range(len(entries)):  # small-loop(added_tokens: configuration entries, not input data)
            var content = py_get(entries[i], "content")
            var id = py_get(entries[i], "id")
            if py_type(content) is not bi.str or py_type(id) is not bi.int:
                raise Error("tokenizer: added token lacks string content or int id")
            py_setitem(added, content, id)
    var vocab = py_get(model, "vocab")
    if py_type(vocab) is not bi.dict:
        raise Error("tokenizer: model.vocab is not an object")
    var base = py_call(bi.dict, vocab)
    var keys = py_call(bi.list, added)
    for i in range(len(keys)):  # small-loop(added tokens: configuration entries)
        _ = py_call(base.pop, keys[i], PythonObject(None))
    var merges = py_get(model, "merges")
    if merges is None:
        merges = bi.list()
    if py_type(merges) is not bi.list:
        raise Error("tokenizer: model.merges is not a list")
    var bos = bi.list()
    var eos = bi.list()
    var post = py_get(data, "post_processor")
    if post is not None:
        var kind = String(py_get(post, "type"))
        if kind == "TemplateProcessing":
            var single = py_get(post, "single", bi.list())
            var specials = py_get(post, "special_tokens", bi.dict())
            var seen = False
            for i in range(len(single)):  # small-loop(post_processor template items: configuration structure)
                var item = single[i]
                if py_contains(item, "Sequence"):
                    seen = True
                elif py_contains(item, "SpecialToken"):
                    var name = py_get(item["SpecialToken"], "id")
                    var ids = py_get(py_get(specials, name, bi.dict()), "ids")
                    if ids is None or len(ids) == 0:
                        raise Error("tokenizer: post_processor special token has no ids")
                    for j in range(len(ids)):  # small-loop(special token ids: configuration words)
                        if seen:
                            _ = py_call(eos.append, py_call(bi.int, ids[j]))
                        else:
                            _ = py_call(bos.append, py_call(bi.int, ids[j]))
                else:
                    raise Error("tokenizer: post_processor template must contain Sequence or SpecialToken")
        elif kind != "ByteLevel":
            raise Error("tokenizer: post_processor type " + kind + " is not honored")
    if cfg is not None and py_type(cfg) is bi.dict:
        var flag = py_get(cfg, "add_bos_token")
        if py_type(flag) is bi.bool and not Bool(py=flag):
            bos.clear()
        flag = py_get(cfg, "add_eos_token")
        if py_type(flag) is bi.bool and not Bool(py=flag):
            eos.clear()
    var options = bi.list()
    _ = py_call(options.append, pattern)
    _ = py_call(options.append, added)
    _ = py_call(options.append, normalizer)
    _ = py_call(options.append, bos)
    _ = py_call(options.append, eos)
    _ = py_call(options.append, py_get(model, "ignore_merges", PythonObject(False)))
    var source = native_path(path)
    if not Bool(py=py_call(source.startswith, "/")):
        source = Python.import_module("os").getcwd() + PythonObject("/") + source
    _ = py_call(options.append, source)
    var tokenizer = vocab_spelled_public_binding(cls, base, merges, options, module)
    py_setattr(tokenizer, "bos_token_id", bos[0] if len(bos) != 0 else PythonObject(None))
    py_setattr(tokenizer, "eos_token_id", eos[len(eos) - 1] if len(eos) != 0 else PythonObject(None))
    if cfg is not None and py_type(cfg) is bi.dict:
        for key in [String("bos_token"), String("eos_token")]:  # small-loop(the two special-token names)
            var value = py_get(cfg, key)
            if py_type(value) is bi.dict:
                value = py_get(value, "content")
            if py_type(value) is bi.str and py_contains(added, value):
                _ = py_call(bi.setattr, tokenizer, key + "_id", added[value])
    return tokenizer


def vocab_pretrained_public_binding(cls: PythonObject, path: PythonObject, module: PythonObject,
                                   patterns: PythonObject) raises -> PythonObject:
    var text = native_path(path)
    var filename = text
    var directory = PythonObject(".")
    if native_path_exists(text, True):
        directory = py_call(text.rstrip, "/")
        filename = directory + PythonObject("/tokenizer.json")
        if not native_path_exists(filename):
            if native_path_exists(directory + PythonObject("/tokenizer.model")):
                raise Error("tokenizer: tokenizer.model without tokenizer.json is a SentencePiece family")
            raise Error("FileNotFoundError: directory holds no tokenizer.json")
    else:
        var pieces = py_call(text.rsplit, "/", 1)
        if len(pieces) == 2:
            directory = pieces[0] if len(pieces[0]) != 0 else PythonObject("/")
    if not native_path_exists(filename):
        raise Error("FileNotFoundError: tokenizer.json does not exist")
    var data = native_json_read(filename)
    var cfg = PythonObject(None)
    var config_path = directory + PythonObject("/tokenizer_config.json")
    if native_path_exists(config_path):
        cfg = native_json_read(config_path)
    if not Bool(py=py_call(filename.startswith, "/")):
        filename = Python.import_module("os").getcwd() + PythonObject("/") + filename
    return vocab_json_public_binding(cls, data, filename, cfg, module, patterns)


def vocab_pattern_name_binding(regex: PythonObject, patterns: PythonObject) raises -> PythonObject:
    return py_get(patterns, regex)


def vocab_pretokenize_pattern_binding(data: PythonObject, pattern: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var raw = py_call(data.encode, "utf-8") if py_type(data) is bi.str else py_call(bi.memoryview, data).tobytes()
    var bytes = _token_bytes(raw)
    var offsets = pretokenize_pattern(bytes, builtin_unicode_classes(), _vocab_pattern(pattern))
    _ = bi
    return _py_int_list(offsets)


def vocab_pretokenize_public_binding(tokenizer: PythonObject, data: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var raw = py_call(data.encode, "utf-8") if Bool(py=py_call(bi.isinstance, data, bi.str)) else py_call(bi.memoryview, data).tobytes()
    var bytes = _py_bytes_list(raw)
    var owner = tokenizer._handle.downcast_value_ptr[VocabHandle]()
    if not owner[].tok:
        raise Error("tokenizer: vocabulary is not loaded")
    var offsets = owner[].tok.value().pretokenize(bytes)
    # The pieces raw[offsets[i]:offsets[i + 1]], cut by CPython's own map
    # and slice (no Python-level loop over the pieces).
    var offs = _py_int_list(offsets)
    var starts = offs[py_call(bi.slice, 0, -1)]
    var stops = offs[py_call(bi.slice, 1, PythonObject(None))]
    return py_call(bi.list, py_call(bi.map, py_call(bi.getattr, raw, PythonObject("__getitem__")), py_call(bi.map, bi.slice, starts, stops)))


def _vocab_public_c[operation: Int](py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    # Supported stdlib C binding API keeps public exception classes native.
    # See std.python.bindings.raise_python_exception (Mojo 1.0).
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        comptime count = 5 if operation == 0 or operation == 1 or operation == 4 else (6 if operation == 2 else (4 if operation == 3 or operation == 5 else (1 if operation == 9 else 2)))
        if len(args) != count:
            raise Error("TypeError: tokenizer binding received incorrect argument count")
        var result = PythonObject(None)
        comptime if operation == 0:
            result = vocab_init_public_binding(args[0], args[1], args[2], args[3], args[4])
        elif operation == 1:
            result = vocab_spelled_public_binding(args[0], args[1], args[2], args[3], args[4])
        elif operation == 2:
            result = vocab_json_public_binding(args[0], args[1], args[2], args[3], args[4], args[5])
        elif operation == 3:
            result = vocab_pretrained_public_binding(args[0], args[1], args[2], args[3])
        elif operation == 4:
            result = vocab_encode_public_binding(args[0], args[1], args[2], args[3], args[4])
        elif operation == 5:
            result = vocab_decode_public_binding(args[0], args[1], args[2], args[3])
        elif operation == 6:
            result = vocab_pretokenize_public_binding(args[0], args[1])
        elif operation == 7:
            result = vocab_pretokenize_pattern_binding(args[0], args[1])
        elif operation == 8:
            result = vocab_pattern_name_binding(args[0], args[1])
        elif operation == 9:
            result = native_json_parse(args[0])
        return result.steal_data()
    except error:
        var message = String(error)
        if message.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if message.startswith("FileNotFoundError:"):
            return raise_python_exception(error_without_class(error, "FileNotFoundError"), ExceptionType("PyExc_FileNotFoundError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_tokenizer_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_tokenizer_host() -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_tokenizer_host")
        transport_initialize()
        module.def_py_c_function(transport_c, "native_transport_call")
        module.def_py_c_function(cross_vendor_c, "cross_vendor_call")
        module.def_py_c_function(parallel_byte_c, "parallel_byte_call")
        module.def_py_c_function(codec_c, "native_codec_call")
        module.def_py_c_function(process_c, "native_process_call")
        module.def_py_c_function(parallel_mlp_c, "parallel_mlp_call")
        module.def_py_c_function(small_mlp_c, "small_mlp_call")
        module.def_py_c_function(parallel_neural_c, "parallel_neural_call")
        module.def_py_c_function(parallel_forecast_c, "parallel_forecast_call")
        module.def_py_c_function(parallel_shard_c, "parallel_shard_call")
        module.def_py_c_function(neighbor_query_c, "neighbor_query_call")
        module.def_py_c_function(neighbor_query_attribute_c, "neighbor_query_attribute_call")
        module.def_py_c_function(parallel_causal_c, "parallel_causal_call")
        # TOMBSTONE: bpe_decode, causal_lm_admit, neural_prefill_call, optimizer_finalize exports (dead native entries: no live caller of the export) deleted 2026-10-26 by 5206fc31b; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_tokenizer_host.patch
        module.def_function[optimizer_finalize_descriptor_binding]("optimizer_finalize_descriptor")
        module.def_py_c_function(optimizer_native_c, "optimizer_native")
        module.def_py_c_function(embedding_layer_lifetime_c, "embedding_layer_lifetime")
        module.def_py_c_function(device_finalize_c, "optimizer_device_finalize")
        module.def_py_c_function(runtime_sequential_forward_c, "sequential_native_call")
        module.def_py_c_function(xseq_c, "xseq_native_call")
        module.def_py_c_function(byte_lm_host_trainer_c, "byte_lm_host_trainer_call")
        module.def_py_c_function(byte_lm_trainer_c, "byte_lm_trainer_call")
        module.def_py_c_function(byte_lm_pooled_finalize_c, "byte_lm_pooled_finalize_call")
        module.def_py_c_function(xcnn_c, "xcnn_native_call")
        module.def_py_c_function(tsa_c, "tsa_native_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "classical_native_call")
        module.def_py_c_function(classical_attribute_c, "classical_attribute_call")
        module.def_py_c_function(classical_attribute_descriptor_c, "classical_attribute_descriptor")
        module.def_py_c_function(mode_metadata_c, "mode_metadata_native_call")
        module.def_py_c_function(params_metadata_c, "params_metadata_native_call")
        module.def_py_c_function(public_attribute_c, "public_attribute_native_call")
        module.def_py_c_function(linear_constructor_c, "linear_constructor_native_call")
        module.def_py_c_function(full_reset_params_c, "full_reset_params_native_call")
        module.def_py_c_function(full_reset_constructor_c, "full_reset_constructor_native_call")
        module.def_py_c_function(array_native_c, "array_native_call")
        module.def_py_c_function(params_setter_c, "params_setter_native_call")
        module.def_py_c_function(declared_params_c, "declared_params_native_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "host_archive_finalize_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "host_svr_attribute_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "host_svc_attribute_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "pca_native_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "standard_scaler_fit_native")
        module.def_py_c_function(runtime_classical_unavailable_c, "standard_scaler_native_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "minmax_scaler_native_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "maxabs_scaler_native_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "prep_stateless_native_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "qn_native_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "logistic_native_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "decomp_native_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "linalg_decomp_native_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "als_native_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "mcd_native_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "pls_native_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "dict_learning_native_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "jl_native_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "ipca_native_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "tsvd_native_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "prep_classify_native_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "prep_encode_native_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "x_linear_native_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "str_labels_native_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "robust_scaler_native_call")
        module.def_py_c_function(arena_rows_c, "arena_rows_native_call")
        module.def_py_c_function(arena_prog_c, "arena_prog_native_call")
        module.def_py_c_function(runtime_classical_unavailable_c, "standard_scaler_attribute_call")
        module.def_py_c_function(generator_draw_c, "generator_draw")
        module.def_py_c_function(user_objects_c, "user_objects_native_call")
        module.def_py_c_function(samba_config_c, "samba_config_call")
        module.def_py_c_function(samba_public_c, "samba_native_call")
        module.def_function[tokenizer_host_numeric_mode_binding]("tokenizer_host_numeric_mode")
        module.def_function[tokenizer_host_vendor_binding]("tokenizer_host_vendor")
        module.def_function[tokenizer_host_column_binding]("tokenizer_host_column")
        module.def_function[tokenizer_host_sabotage_binding]("tokenizer_host_sabotage")
        module.def_function[tokenizer_host_runtime_classical_abi_binding]("tokenizer_host_runtime_classical_abi")
        _ = module.add_type[BpeHandle]("_BpeHandle")
        module.def_function[bpe_load_binding]("bpe_load")
        module.def_function[bpe_n_vocab_binding]("bpe_n_vocab")
        module.def_function[bpe_max_token_bytes_binding]("bpe_max_token_bytes")
        module.def_function[bpe_encode_binding]("bpe_encode")
        module.def_function[bpe_encode_batch_binding]("bpe_encode_batch")
        # TOMBSTONE: bpe_decode, causal_lm_admit, neural_prefill_call, optimizer_finalize exports (dead native entries: no live caller of the export) deleted 2026-10-26 by 5206fc31b; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_tokenizer_host.patch
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
        _ = module.add_type[VocabHandle]("_VocabHandle")
        module.def_py_c_function(_vocab_public_c[0], "vocab_init_public")
        module.def_py_c_function(_vocab_public_c[1], "vocab_spelled_public")
        module.def_py_c_function(_vocab_public_c[2], "vocab_json_public")
        module.def_py_c_function(_vocab_public_c[3], "vocab_pretrained_public")
        module.def_py_c_function(_vocab_public_c[4], "vocab_encode_public")
        module.def_py_c_function(_vocab_public_c[5], "vocab_decode_public")
        module.def_py_c_function(_vocab_public_c[6], "vocab_pretokenize_public")
        module.def_py_c_function(_vocab_public_c[7], "vocab_pretokenize_pattern")
        module.def_py_c_function(_vocab_public_c[8], "vocab_pattern_name")
        module.def_py_c_function(_vocab_public_c[9], "native_json_parse")
        module.def_py_c_function(model_config_c[0], "model_config_init")
        module.def_py_c_function(model_config_c[1], "model_config_load")
        module.def_py_c_function(model_config_c[2], "model_config_access")
        module.def_py_c_function(model_config_c[3], "model_plan_init")
        module.def_py_c_function(model_config_c[4], "model_layer_weights")
        module.def_py_c_function(model_config_c[5], "model_checkpoint_names")
        module.def_py_c_function(model_config_c[6], "model_plan_for")
        module.def_py_c_function(checkpoint_c[0], "tensor_info_init")
        module.def_py_c_function(checkpoint_c[1], "tensor_info_size")
        module.def_py_c_function(checkpoint_c[2], "safetensors_init")
        module.def_py_c_function(checkpoint_c[3], "safetensors_info")
        module.def_py_c_function(checkpoint_c[4], "safetensors_map")
        module.def_py_c_function(checkpoint_c[5], "safetensors_read")
        module.def_py_c_function(checkpoint_c[6], "safetensors_close")
        module.def_py_c_function(checkpoint_c[7], "checkpoint_init")
        module.def_py_c_function(checkpoint_c[8], "checkpoint_open")
        module.def_py_c_function(checkpoint_c[9], "checkpoint_file")
        module.def_py_c_function(checkpoint_c[10], "checkpoint_index")
        module.def_py_c_function(checkpoint_c[11], "checkpoint_access")
        module.def_py_c_function(checkpoint_c[12], "checkpoint_read")
        module.def_py_c_function(checkpoint_c[13], "checkpoint_close")
        module.def_py_c_function(checkpoint_c[14], "checkpoint_enter")
        module.def_py_c_function(checkpoint_c[15], "safetensors_names")
        module.def_py_c_function(checkpoint_c[16], "safetensors_contains")
        module.def_function[vocab_fingerprint_binding]("vocab_fingerprint")
        module.def_function[sha256_hex_binding]("sha256_hex")
        module.def_py_c_function(neural_block_c[0], "neural_block_init")
        module.def_py_c_function(neural_block_c[4], "neural_block_set_params")
        module.def_py_c_function(neural_block_c[5], "neural_block_getstate")
        module.def_py_c_function(neural_block_c[6], "neural_block_setstate")
        module.def_py_c_function(neural_block_c[1], "neural_block_allocate")
        module.def_py_c_function(neural_block_c[2], "neural_block_forward")
        module.def_py_c_function(neural_block_c[3], "neural_block_backward")
        # TOMBSTONE: bpe_decode, causal_lm_admit, neural_prefill_call, optimizer_finalize exports (dead native entries: no live caller of the export) deleted 2026-10-26 by 5206fc31b; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_tokenizer_host.patch
        module.def_py_c_function(mamba_native_c, "mamba_native_call")
        _ = module.add_type[MambaDecodeHandle]("_MambaDecodeHandle")
        module.def_py_c_function(causal_model_c[0], "causal_model_init")
        module.def_py_c_function(causal_model_c[1], "causal_model_load")
        module.def_py_c_function(causal_model_c[2], "causal_model_read_weights")
        _ = module.add_type[GenerationHandle]("_GenerationHandle")
        _ = module.add_type[GenerationNativeProvider]("_GenerationNativeProvider")
        _ = module.add_type[GenerationNativeSession]("_GenerationNativeSession")
        _ = module.add_type[NeuralDescriptorProvider]("_NeuralDescriptorProvider")
        _ = module.add_type[NativeDescriptorResource]("_NativeDescriptorResource")
        _ = module.add_type[ProviderOperationLock]("_NativeOperationLockOwner")
        module.def_function[operation_lock_connect]("operation_lock_connect")
        module.def_py_c_function(operation_lock_c[0], "operation_lock_init")
        module.def_py_c_function(operation_lock_c[1], "operation_lock_acquire")
        module.def_py_c_function(operation_lock_c[2], "operation_lock_release")
        module.def_py_c_function(operation_lock_c[3], "operation_lock_enter")
        module.def_py_c_function(operation_lock_c[4], "operation_lock_exit")
        module.def_py_c_function(operation_lock_c[5], "operation_lock_owned")
        module.def_function[neural_descriptor_info]("neural_native_dispatch_info")
        _ = module.add_type[RemoteStateLifetime]("_RemoteStateLifetime")
        # TOMBSTONE: bpe_decode, causal_lm_admit, neural_prefill_call, optimizer_finalize exports (dead native entries: no live caller of the export) deleted 2026-10-26 by 5206fc31b; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_tokenizer_host.patch
        module.def_py_c_function(causal_public_c[1], "causal_lm_allocate")
        module.def_py_c_function(causal_public_c[2], "causal_lm_reset")
        module.def_py_c_function(causal_public_c[3], "causal_lm_forward")
        module.def_py_c_function(causal_public_c[4], "causal_lm_stream")
        module.def_py_c_function(causal_public_c[5], "causal_lm_read")
        module.def_py_c_function(causal_public_c[6], "causal_lm_info")
        module.def_py_c_function(causal_public_c[7], "causal_lm_generate")
        module.def_py_c_function(causal_public_c[8], "causal_lm_text_stream")
        module.def_py_c_function(causal_public_c[9], "causal_lm_generate_text")
        module.def_py_c_function(causal_public_c[10], "causal_lm_resident")
        module.def_py_c_function(causal_public_c[11], "causal_lm_snapshot")
        module.def_py_c_function(causal_public_c[12], "causal_lm_resume")
        module.def_py_c_function(causal_public_c[13], "causal_lm_parameters")
        module.def_py_c_function(causal_public_c[14], "causal_lm_blocks")
        module.def_py_c_function(causal_public_c[15], "causal_state_init")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_tokenizer_host: ", error))

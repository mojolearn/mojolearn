# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST-tier tokenizer binding: BPE training and encode_batch on the Apple GPU.

Ported from lane/apple-fast-bpe (2026-10-03) as a thin shell (lane port-apple-fast-bpe,
2026-10-21). Built ONLY by `bindings/build_tokenizer_fast.sh`, which refuses every tier but fast;
the IDENTICAL door stays `_mojolearn_tokenizer_host` (unchanged). Each entry unpacks its Python
arguments, calls ONE typed function of `tokenizer/fast/bpe_fast_impl.mojo` and converts the result
back; the kernels are `tokenizer/fast/bpe_device.mojo`.

`python/mojolearn/tokenizer.py` loads this file from the fast tier's directory only under
numeric mode fast, and routes to it only for what `bpe_fast_flags()` says was compiled in (each
device path needs FAST, an Apple GPU target and its `-D MOJOLEARN_BPE_*` define). A build with no
define reports nothing compiled and every call runs the host binding.

THE ADDRESS CONTRACT is the host binding's, so the Python door passes the same buffers:

    tokenizer_fast_numeric_mode() -> 0 (FAST); the loader refuses anything else
    bpe_fast_flags() -> [train_device, encode_device, merge_batch, group_filter, livebuf]
    bpe_train(text_addr, lengths_addr, [n_docs, n_bytes, vocab_size, min_frequency, break_ties_high])
        the merge loop on the GPU. break_ties_high (the host trainer's negative control) is
        refused here: the Python door sends a sabotage run to the host binding.
    bpe_trained_sizes(handle), bpe_trained_copy(handle, arena, lengths, left, right)
    bpe_fast_load(ranks_path) -> handle
    bpe_encode_batch(handle, text_addr, lengths_addr, ids_addr, counts_addr,
                     [n_docs, n_bytes, out_cap, allow_endoftext])
        allow_endoftext must be False (the door sends True to the host binding) and out_cap
        at least n_bytes; ids back to back as int32, counts as int64; returns the total.
"""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from checks.numerics import GLOBAL_NUMERIC_MODE
from tokenizer.fast.bpe_fast_impl import (
    FastBpeHandle,
    FastTrainedHandle,
    bpe_fast_flag_list,
    fast_bpe_encode_batch_into,
    fast_bpe_load,
    fast_bpe_train_fill,
    fast_bpe_trained_copy_into,
)


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


def tokenizer_fast_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def bpe_fast_flags_binding() raises -> PythonObject:
    var flags = bpe_fast_flag_list()
    var out = Python.list()
    out.append(PythonObject(flags[0]))
    out.append(PythonObject(flags[1]))
    out.append(PythonObject(flags[2]))
    out.append(PythonObject(flags[3]))
    out.append(PythonObject(flags[4]))
    return out


def bpe_train_binding(text_addr: PythonObject, lengths_addr: PythonObject, dims: PythonObject) raises -> PythonObject:
    """The host `bpe_train` ABI with the merge loop on the GPU (TRAIN_DEVICE builds only)."""
    if Int(py=len(dims)) != 5:
        raise Error("bpe_train: dims must be [n_docs, n_bytes, vocab_size, min_frequency, break_ties_high]")
    var n_docs = _index(dims[0])
    var n = _index(dims[1])
    var vocab_size = _index(dims[2])
    var min_frequency = _index(dims[3])
    var reverse = _flag(dims[4], "break_ties_high")
    if reverse:
        raise Error("bpe_train (fast): break_ties_high is the host binding's negative control")
    if n_docs < 1:
        raise Error("bpe_train: n_docs must be >= 1, got " + String(n_docs))
    if n < 0:
        raise Error("bpe_train: n_bytes must be >= 0, got " + String(n))
    var text_address = _index(text_addr) if n > 0 else 0
    var lengths_address = _index(lengths_addr)
    var handle = FastTrainedHandle()
    with GILReleased(Python()):
        fast_bpe_train_fill(handle, text_address, lengths_address, n_docs, n, vocab_size, min_frequency)
    return PythonObject(alloc=handle^)


def bpe_trained_sizes_binding(handle: PythonObject) raises -> PythonObject:
    """`[n_tokens, arena_bytes, n_merges, n_ties_broken, n_groups]`, as the host binding."""
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
    """The host binding's copy-out: token bytes in rank order, lengths, merges. Returns n_tokens."""
    var owner = handle.downcast_value_ptr[FastTrainedHandle]()
    if not owner[].vocab:
        raise Error("tokenizer fast: the handle holds no trained vocabulary")
    var arena_address = _index(arena_addr)
    var lengths_address = _index(lengths_addr)
    var nm = owner[].vocab.value().n_merges()
    var left_address = _index(left_addr) if nm > 0 else 0
    var right_address = _index(right_addr) if nm > 0 else 0
    ref v = owner[].vocab.value()
    fast_bpe_trained_copy_into(v, arena_address, lengths_address, nm, left_address, right_address)
    return PythonObject(v.n_tokens())


def bpe_fast_load_binding(ranks_path: PythonObject) raises -> PythonObject:
    """The host loader's parse of the caller's rank file, into this binding's own handle."""
    var rp = String(py=ranks_path)
    var handle = FastBpeHandle()
    with GILReleased(Python()):
        fast_bpe_load(handle, rp)
    return PythonObject(alloc=handle^)


def bpe_encode_batch_binding(
    handle: PythonObject,
    text_addr: PythonObject,
    lengths_addr: PythonObject,
    ids_addr: PythonObject,
    counts_addr: PythonObject,
    dims: PythonObject,
) raises -> PythonObject:
    """encode_batch on the GPU (ENCODE_DEVICE builds only), the host binding's ABI."""
    var owner = handle.downcast_value_ptr[FastBpeHandle]()
    if not owner[].tok:
        raise Error("tokenizer fast: the handle holds no tokenizer; bpe_fast_load it")
    if Int(py=len(dims)) != 4:
        raise Error("bpe_encode_batch (fast): dims must be [n_docs, n_bytes, out_cap, allow_endoftext]")
    var n_docs = _index(dims[0])
    var n = _index(dims[1])
    var cap = _index(dims[2])
    var allow = _flag(dims[3], "allow_endoftext")
    if allow:
        raise Error("bpe_encode_batch (fast): allow_endoftext is served by the host binding")
    if n_docs < 0:
        raise Error("bpe_encode_batch: n_docs must be >= 0, got " + String(n_docs))
    if n < 0:
        raise Error("bpe_encode_batch: n_bytes must be >= 0, got " + String(n))
    if n_docs == 0:
        return PythonObject(0)
    var text_address = _index(text_addr) if n > 0 else 0
    var lengths_address = _index(lengths_addr)
    var ids_address = _index(ids_addr) if n > 0 else 0
    var counts_address = _index(counts_addr)
    var total = 0
    with GILReleased(Python()):
        total = fast_bpe_encode_batch_into(
            owner[], text_address, lengths_address, ids_address, counts_address, n_docs, n, cap
        )
    return PythonObject(total)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_tokenizer_fast.mojo
# (the per-binding build, tools/merged_bindings.py entries).
def build__mojolearn_tokenizer_fast() -> PythonObject:
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

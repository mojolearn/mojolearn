# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IDENTICAL GPU tokenizer binding: BPE encode and decode on CUDA, HIP and Metal (lane
vdf-parallel, 2026-11-24). The `device="gpu"` side of `mojolearn.tokenizer.BpeTokenizer`.

A thin shell: each entry unpacks its arguments, calls ONE typed function of
`tokenizer/gpu/bpe_gpu.mojo` (the kernels and the device orchestration) and returns a count. The
public door stays `bindings/tokenizer_public_native_door.mojo` in the host binding; it connects this
module at import (`python/mojolearn/tokenizer.py` `_gpu_binding`) and routes a tokenizer built with
`device="gpu"` here. Same ids, id for id, as the host column (integer work only).

THE ADDRESS CONTRACT (the host binding's buffers):

    tokenizer_gpu_numeric_mode() -> 1 (IDENTICAL); the loader refuses anything else
    bpe_gpu_load(ranks_text) -> handle: the host binding's canonical rank text of the same
        tokenizer, parsed, and its tables uploaded to the device ONCE for the handle's life
    bpe_gpu_encode_batch(handle, text_addr, lengths_addr, ids_addr, counts_addr,
                         [n_docs, n_bytes, out_cap, allow_endoftext]) -> total
        the text uploaded once, every step on the device, ids (int32, back to back) and counts
        (int64) read back once after the last kernel
    bpe_gpu_decode_batch(handle, ids_addr, offsets_addr, out_addr, lens_addr,
                         [n_seq, n_ids, out_cap]) -> total bytes, or -1 - position of the first
        id outside [0, n_vocab) (the door raises the host's words for it)
"""
from core.experiment_guards_neural import NEURAL_GUARDS_OK
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from bindings.native_loaded_module_door import native_loaded_module
from bindings.p3_lm_tokenizer_cpython_unpack import lm_try_index
from bindings.py_balanced import py_call as _balanced_call

from checks.numerics import GLOBAL_NUMERIC_MODE
from tokenizer.encoding import BpeTokenizer
from tokenizer.impl.ranks import parse_rank_table
from tokenizer.impl.unicode_class import builtin_unicode_classes
from tokenizer.gpu.bpe_gpu import BpeGpuTable, bpe_gpu_ctx, bpe_gpu_encode_batch_into, bpe_gpu_decode_batch_into


struct GpuBpeHandle(Movable, Writable):
    """Python-owned: the parsed tokenizer (its eot id, its maximum token length) and its device
    tables, uploaded once."""

    var tok: Optional[BpeTokenizer]
    var table: Optional[BpeGpuTable]
    var max_token_bytes: Int

    def __init__(out self):
        self.tok = Optional[BpeTokenizer]()
        self.table = Optional[BpeGpuTable]()
        self.max_token_bytes = 0

    def write_to(self, mut writer: Some[Writer]):
        writer.write("_GpuBpeHandle(loaded=", Bool(self.tok), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("_GpuBpeHandle(loaded=", Bool(self.tok), ")")


def _index(value: PythonObject) raises -> Int:
    var word = 0
    if lm_try_index(value, word):
        return word
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("tokenizer gpu: integers expected, not booleans")
    return Int(py=_balanced_call(native_loaded_module("operator").index, value))


def _flag(value: PythonObject, name: String) raises -> Bool:
    if value is PythonObject(True):
        return True
    if value is PythonObject(False):
        return False
    raise Error("tokenizer gpu: " + name + " must be a bool, got " + String(py=value.__class__.__name__))


def tokenizer_gpu_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def bpe_gpu_load_binding(ranks_text: PythonObject) raises -> PythonObject:
    """Parse the canonical rank text and upload the tables once."""
    var text = String(py=ranks_text)
    var handle = GpuBpeHandle()
    with GILReleased(Python()):
        var tok = BpeTokenizer(parse_rank_table(text, "<rank text>"), builtin_unicode_classes())
        var longest = 0
        for k in range(len(tok.ranks.length)):  # driver-loop: over vocabulary entries, once per tokenizer
            longest = max(longest, tok.ranks.length[k])
        handle.max_token_bytes = longest
        handle.table = BpeGpuTable(bpe_gpu_ctx(), tok)
        handle.tok = tok^
    return PythonObject(alloc=handle^)


def bpe_gpu_encode_batch_binding(
    handle: PythonObject, text_addr: PythonObject, lengths_addr: PythonObject, ids_addr: PythonObject,
    counts_addr: PythonObject, dims: PythonObject,
) raises -> PythonObject:
    var owner = handle.downcast_value_ptr[GpuBpeHandle]()
    if not owner[].tok or not owner[].table:
        raise Error("tokenizer gpu: the handle holds no tokenizer; bpe_gpu_load it")
    if Int(py=len(dims)) != 4:
        raise Error("bpe_encode_batch (gpu): dims must be [n_docs, n_bytes, out_cap, allow_endoftext]")
    var n_docs = _index(dims[0])
    var n = _index(dims[1])
    var cap = _index(dims[2])
    var allow = _flag(dims[3], "allow_endoftext")
    if n_docs < 0 or n < 0:
        raise Error("bpe_encode_batch (gpu): n_docs and n_bytes must be >= 0")
    if cap < n:
        raise Error("bpe_encode_batch (gpu): out_cap must hold one id per byte")
    if n_docs == 0:
        return PythonObject(0)
    var text_address = _index(text_addr) if n > 0 else 0
    var lengths = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=_index(lengths_addr))
    var ids_address = _index(ids_addr) if n > 0 else 0
    var counts_address = _index(counts_addr)
    # The document offsets from the caller's lengths (the host binding's admission, the same words).
    var offs = List[Int64](length=n_docs + 1, fill=Int64(0))
    for k in range(n_docs):  # driver-loop: over document lengths (admission, before the upload)
        if Int(lengths[k]) < 0:
            raise Error("bpe_encode_batch: document " + String(k) + " has a negative length")
        offs[k + 1] = offs[k] + lengths[k]
    if Int(offs[n_docs]) != n:
        raise Error("bpe_encode_batch: document lengths must sum to n_bytes " + String(n) + ", got 0 and "
            + String(Int(offs[n_docs])))
    var total = 0
    var eot_id = owner[].tok.value().eot_id()
    with GILReleased(Python()):
        total = bpe_gpu_encode_batch_into(owner[].table.value(), text_address, n, offs, allow, eot_id,
                                          ids_address, counts_address)
    return PythonObject(total)


def bpe_gpu_decode_batch_binding(
    handle: PythonObject, ids_addr: PythonObject, offsets_addr: PythonObject, out_addr: PythonObject,
    lens_addr: PythonObject, dims: PythonObject,
) raises -> PythonObject:
    var owner = handle.downcast_value_ptr[GpuBpeHandle]()
    if not owner[].tok or not owner[].table:
        raise Error("tokenizer gpu: the handle holds no tokenizer; bpe_gpu_load it")
    if Int(py=len(dims)) != 3:
        raise Error("bpe_decode_batch (gpu): dims must be [n_seq, n_ids, out_cap]")
    var nseq = _index(dims[0])
    var nids = _index(dims[1])
    var cap = _index(dims[2])
    if nseq < 0 or nids < 0:
        raise Error("bpe_decode_batch (gpu): n_seq and n_ids must be >= 0")
    var offs_p = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=_index(offsets_addr))
    var offs = List[Int64](length=nseq + 1, fill=Int64(0))
    for k in range(nseq + 1):  # driver-loop: over sequence offsets (admission)
        offs[k] = offs_p[k]
        if k > 0 and Int(offs[k]) < Int(offs[k - 1]):
            raise Error("bpe_decode_batch (gpu): sequence offsets decrease at sequence " + String(k - 1))
    if Int(offs[0]) != 0 or Int(offs[nseq]) != nids:
        raise Error("bpe_decode_batch (gpu): sequence offsets must run from 0 to n_ids")
    var ids_address = _index(ids_addr) if nids > 0 else 0
    var out_address = _index(out_addr) if cap > 0 else 0
    var lens_address = _index(lens_addr) if nseq > 0 else 0
    var result = 0
    with GILReleased(Python()):
        result = bpe_gpu_decode_batch_into(owner[].table.value(), Optional[MutPointer[Int64, MutAnyOrigin]](None),
            ids_address, offs, owner[].max_token_bytes, owner[].tok.value().eot_id(), out_address, cap,
            lens_address)
    return PythonObject(result)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_tokenizer_gpu.mojo.
def build__mojolearn_tokenizer_gpu() -> PythonObject:
    comptime assert NEURAL_GUARDS_OK, "invalid integrated experiment configuration (neural guards)"
    try:
        var module = PythonModuleBuilder("_mojolearn_tokenizer_gpu")
        module.def_function[tokenizer_gpu_numeric_mode_binding]("tokenizer_gpu_numeric_mode")
        _ = module.add_type[GpuBpeHandle]("_GpuBpeHandle")
        module.def_function[bpe_gpu_load_binding]("bpe_gpu_load")
        module.def_function[bpe_gpu_encode_batch_binding]("bpe_gpu_encode_batch")
        module.def_function[bpe_gpu_decode_batch_binding]("bpe_gpu_decode_batch")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_tokenizer_gpu: ", error))

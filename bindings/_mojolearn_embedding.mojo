# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the Embedding lane, profile `mojolearn.identical.embedding.fp32.v1`.

The door of `embedding/checks/embedding_identical.mojo`: the forward gather
(`identical_embedding_forward_into`, seams G1 and G2) and the backward fold
(`identical_embedding_backward_into`, seams E0 to E4, PLAN_SCAN or PLAN_SORT,
contract 6.1 and 6.2, which clause (d) holds bit-identical) with the
two knobs the training binding's `embedding_forward` and
`embedding_backward` (`training/samba_ops.mojo`) do not carry:
`padding_idx` (contract section 8, DEVIATION 1311) and `accumulate`, the
microbatch CARRY (contract 7.4, DEVIATION 1309). Nothing is re-decided.

REFUSED ON THIS HOST, BY NAME, BEFORE ANY UPLOAD (the contract's refusals):
    V <= 0, d < 0, T < 0, T > EMB_MAX_POSITIONS   emb_refuse_shape, contract 3 and 8
    an id below 0 or at or past V                  emb_refuse_ids, contract 8 (never clamped)
    padding_idx outside [0, V) other than -1        here
    a NaN or an infinity in W, dY or a carried dW   refuse_nonfinite, contract 9.1

The nonfinite refusal is the host half of DEVIATION 1506: the device entry
points still do not refuse NaN, so this boundary does it on the host copy.

THE ABI IS THE GP'S: two length-checked lists, orders written out below and
mirrored in `python/mojolearn/embedding.py`. THE GIL is released around the
device call, and nothing inside the `GILReleased` block touches a
`PythonObject`. The typed bodies (pools, transfers, runs, the work done
with the GIL released) are RUN code in `embedding/embedding_words_run.mojo`.
"""

from core.experiment_guards_neural import NEURAL_GUARDS_OK
from std.os import abort
from core.hostptr import f32_ptr, i32_ptr
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from embedding.checks.embedding_identical import PLAN_AUTO, emb_resolve_plan
from autograd.kernels.embedding_sort import PLAN_SCAN, PLAN_SORT
from embedding.checks.embedding_oracle import EMB_NO_PADDING_IDX, emb_refuse_shape
# RUN (lane rs-embedding-bytelm): typed Mojo only, no Python import.
from embedding.embedding_words_run import (
    _config,
    _emb_tail,
    embedding_backward_dev_released,
    embedding_backward_released,
    embedding_forward_dev_released,
    embedding_forward_released,
    embedding_table_release_words,
    seq_tensor_alloc_words,
    seq_tensor_copy_words,
    seq_tensor_download_words,
    seq_tensor_free_words,
    seq_tensor_upload_words,
)
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.neural_entry_table import (
    NeuralEntryKey, neural_entry_vendor, neural_table_publish, neural_entry_spans, neural_entry_result,
    NEURAL_TABLE_EMBEDDING, NEURAL_EMB_OPERATIONS, EMB_OP_FORWARD, EMB_OP_BACKWARD, EMB_OP_TABLE_RELEASE,
    EMB_OP_FORWARD_DEV, EMB_OP_BACKWARD_DEV, EMB_OP_TENSOR_ALLOC, EMB_OP_TENSOR_FREE, EMB_OP_TENSOR_UPLOAD,
)

# TOMBSTONE: MOJOLEARN_IDN_EMB_SCAN_DEFERRED (DROPPED-slower) deleted 2026-10-10 by lane/postmerge-act-6; code recoverable at 9f83ea479.
# Tried: the default-on deferred dY non-finite scan of the resident embedding backward (EMB_SCAN_DEFERRED, lane neural-io-2: scan and partials queued ahead of the fold, read after the one final wait). Its _OFF arm, the eager scans kept here, WON: embedding NV 2.70 -> 2.28 ms (0.84x) / AMD 2.09 -> 1.94 ms (0.92x) with _OFF (nv2 v1056 -> v1059, amd a1156 -> a1159), digest 2753e61a unchanged. The eager scans are the only path; both defines are refused.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_EMB_SCAN_DEFERRED.patch; record in docs/TOMBSTONES.md.


def embedding_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER as the `NUMERIC_*` code: 0 FAST, 1 IDENTICAL."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def embedding_vendor_binding() raises -> PythonObject:
    """'metal', 'cuda', 'hip' or 'none', from `checks/vendor.mojo`."""
    return PythonObject(String(COMPILED_VENDOR))


def embedding_table_release_binding(token_obj: PythonObject) raises -> PythonObject:
    return PythonObject(embedding_table_release_words(Int(py=token_obj)))


# TOMBSTONE: embedding_resident (dead native entry: a compiled-constant probe nothing reads; EMB_RESIDENT stays) deleted 2026-10-26 by 9aba8da70; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_embedding.patch


def embedding_forward_words(a: List[Int], p: List[Int]) raises -> Int:
    """`Y[t, j] = W[ids[t], j]` through seams G1 and G2. Returns `T * d`.

    `addrs`, in this exact order:

        0  weight     V * d float32, row-major, read
        1  ids        T int32, read
        2  y_out      T * d float32, WRITTEN

    `params`, in this exact order:

        0  V
        1  d
        2  T
        3  token      optional; a positive table token keeps the table's
                      device copy resident for the next forward with the
                      same token (EMB_RESIDENT); 0 or absent uploads it
    """
    if len(a) != 3:
        raise Error(
            "embedding_forward: addrs must contain 3 addresses (weight, ids,"
            " y_out), got " + String(len(a))
        )
    if len(p) != 3 and len(p) != 4:
        raise Error(
            "embedding_forward: params must contain 3 or 4 values (V, d, T[,"
            " token]), got " + String(len(p))
        )
    var vocab = p[0]
    var width = p[1]
    var n_positions = p[2]
    var token = 0
    if len(p) == 4:
        token = p[3]
    var cfg = _config(vocab, width, EMB_NO_PADDING_IDX, False)
    emb_refuse_shape(cfg, n_positions)
    var w_addr = a[0]
    var wp = f32_ptr(w_addr)
    # Lane ne3-c-blocks-embedding (row N158): the ids are read through the
    # caller's buffer and refused with the GIL released (no host list copy,
    # no walk under the GIL); the refusal order (ids, then the y address,
    # then W) and the words are unchanged.
    var idsp = i32_ptr(a[1])
    var y_addr = a[2]
    # the W scan runs on the device copy unless there is nothing to gather
    # (then nothing is uploaded)
    var no_run = n_positions * width <= 0
    with GILReleased(Python()):
        embedding_forward_released(wp, idsp, n_positions, cfg, y_addr, no_run, token, w_addr)
    return n_positions * width


def embedding_forward_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    return PythonObject(embedding_forward_words(_emb_words(addrs), _emb_words(params)))


def embedding_backward_words(a: List[Int], p: List[Int]) raises -> Int:
    """`dW` through seams E0 to E4 (contract 5.1's serial ascending fold). Returns `V * d`.

    `addrs`, in this exact order:

        0  dy         T * d float32, row-major, read
        1  ids        T int32, read
        2  dw         V * d float32, WRITTEN; READ FIRST when accumulate is 1
                      (the carried accumulator, contract 7.4)

    `params`, in this exact order:

        0  V
        1  d
        2  T
        3  padding_idx   -1 for none; its row is +0.0 STORED (contract 8)
        4  accumulate    0 fresh (+0.0 fill), 1 carry the dw buffer's bits
        5  plan          optional; 0 PLAN_SCAN (the default), 1 PLAN_SORT,
                         2 PLAN_AUTO (PLAN_SORT at large V * T, else
                         PLAN_SCAN; embedding_identical.mojo). The run
                         structure's execution plan (contract 6); both
                         give the same counts, perm and dW bits
    """
    if len(a) != 3:
        raise Error(
            "embedding_backward: addrs must contain 3 addresses (dy, ids, dw),"
            " got " + String(len(a))
        )
    if len(p) != 5 and len(p) != 6:
        raise Error(
            "embedding_backward: params must contain 5 or 6 values (V, d, T,"
            " padding_idx, accumulate[, plan]), got " + String(len(p))
        )
    var vocab = p[0]
    var width = p[1]
    var n_positions = p[2]
    var padding_idx = p[3]
    var acc_code = p[4]
    var plan = PLAN_SCAN
    if len(p) == 6:
        var plan_code = p[5]
        if plan_code != PLAN_SCAN and plan_code != PLAN_SORT and plan_code != PLAN_AUTO:
            raise Error(
                String("embedding_backward: plan must be 0 (PLAN_SCAN), 1")
                + " (PLAN_SORT) or 2 (PLAN_AUTO), got "
                + String(plan_code)
            )
        plan = emb_resolve_plan(plan_code, vocab, n_positions)
    if acc_code != 0 and acc_code != 1:
        raise Error(
            String("embedding_backward: accumulate must be 0 or 1, got ")
            + String(acc_code)
        )
    var cfg = _config(vocab, width, padding_idx, acc_code == 1)
    emb_refuse_shape(cfg, n_positions)
    var dyp = f32_ptr(a[0])
    # Lane ne3-c-blocks-embedding (row N158): ids through the caller's
    # buffer, refused with the GIL released; order and words unchanged.
    var idsp = i32_ptr(a[1])
    var dw_addr = a[2]
    var no_run = vocab * width <= 0
    with GILReleased(Python()):
        embedding_backward_released(dyp, idsp, n_positions, cfg, dw_addr, no_run, plan)
    return vocab * width


def embedding_backward_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    return PythonObject(embedding_backward_words(_emb_words(addrs), _emb_words(params)))


def seq_tensor_alloc_binding(n_obj: PythonObject, zero_obj: PythonObject) raises -> PythonObject:
    return PythonObject(seq_tensor_alloc_words(Int(py=n_obj), Int(py=zero_obj)))


def seq_tensor_free_binding(handle: PythonObject) raises -> PythonObject:
    return PythonObject(seq_tensor_free_words(Int(py=handle)))


def seq_tensor_upload_binding(handle: PythonObject, addr: PythonObject, n_obj: PythonObject) raises -> PythonObject:
    return PythonObject(seq_tensor_upload_words(Int(py=handle), Int(py=addr), Int(py=n_obj)))


def seq_tensor_download_binding(handle: PythonObject, addr: PythonObject, n_obj: PythonObject) raises -> PythonObject:
    return PythonObject(seq_tensor_download_words(Int(py=handle), Int(py=addr), Int(py=n_obj)))


def seq_tensor_copy_binding(dst: PythonObject, src: PythonObject, n_obj: PythonObject) raises -> PythonObject:
    return PythonObject(seq_tensor_copy_words(Int(py=dst), Int(py=src), Int(py=n_obj)))


# TOMBSTONE: MOJOLEARN_IDN_EMB_SCAN_DEFERRED (DROPPED-slower) deleted 2026-10-10 by lane/postmerge-act-6; code recoverable at 9f83ea479.
# Tried: _scan_enqueue, _refuse_scanned and _backward_dev_deferred (the deferred dY scan).
# Restore: git apply experiments/removed/MOJOLEARN_IDN_EMB_SCAN_DEFERRED.patch; record in docs/TOMBSTONES.md.


def embedding_forward_dev_words(a: List[Int], h: List[Int], p: List[Int]) raises -> Int:
    """`embedding_forward` into a resident y. addrs = [weight (V * d, host),
    ids (T int32, host)]; handles = [y (T * d resident tensor)]; params =
    [V, d, T, token]. Returns `T * d`."""
    if len(a) != 2 or len(h) != 1 or len(p) != 4:
        raise Error("embedding_forward_dev: requires 2 addresses (weight, ids), 1 handle (y) and [V, d, T, token]")
    var vocab = p[0]
    var width = p[1]
    var n_positions = p[2]
    var token = p[3]
    var cfg = _config(vocab, width, EMB_NO_PADDING_IDX, False)
    emb_refuse_shape(cfg, n_positions)
    var w_addr = a[0]
    var wp = f32_ptr(w_addr)
    var idsp = i32_ptr(a[1])
    var y_h = h[0]
    var no_run = n_positions * width <= 0
    with GILReleased(Python()):
        embedding_forward_dev_released(wp, idsp, n_positions, cfg, y_h, no_run, token, w_addr)
    return n_positions * width


def embedding_forward_dev_binding(
    addrs: PythonObject, handles: PythonObject, params: PythonObject
) raises -> PythonObject:
    return PythonObject(embedding_forward_dev_words(_emb_words(addrs), _emb_words(handles), _emb_words(params)))


def embedding_backward_dev_words(a: List[Int], h: List[Int], p: List[Int]) raises -> Int:
    """`embedding_backward` on resident tensors. addrs = [ids (T int32,
    host)]; handles = [dy (T * d), dw (V * d; its bits are the carry when
    accumulate is 1)]; params = [V, d, T, padding_idx, accumulate, plan]
    as `embedding_backward`'s. Returns `V * d`."""
    if len(a) != 1 or len(h) != 2 or len(p) != 6:
        raise Error("embedding_backward_dev: requires 1 address (ids), 2 handles (dy, dw) and 6 params")
    var vocab = p[0]
    var width = p[1]
    var n_positions = p[2]
    var padding_idx = p[3]
    var acc_code = p[4]
    var plan_code = p[5]
    if plan_code != PLAN_SCAN and plan_code != PLAN_SORT and plan_code != PLAN_AUTO:
        raise Error(
            String("embedding_backward_dev: plan must be 0 (PLAN_SCAN), 1 (PLAN_SORT) or 2 (PLAN_AUTO), got ")
            + String(plan_code)
        )
    var plan = emb_resolve_plan(plan_code, vocab, n_positions)
    if acc_code != 0 and acc_code != 1:
        raise Error(String("embedding_backward_dev: accumulate must be 0 or 1, got ") + String(acc_code))
    var cfg = _config(vocab, width, padding_idx, acc_code == 1)
    emb_refuse_shape(cfg, n_positions)
    var idsp = i32_ptr(a[0])
    var dy_h = h[0]
    var dw_h = h[1]
    with GILReleased(Python()):
        embedding_backward_dev_released(idsp, n_positions, cfg, dy_h, dw_h, plan)
    return vocab * width


def embedding_backward_dev_binding(
    addrs: PythonObject, handles: PythonObject, params: PythonObject
) raises -> PythonObject:
    return PythonObject(embedding_backward_dev_words(_emb_words(addrs), _emb_words(handles), _emb_words(params)))


def _emb_words(value: PythonObject) raises -> List[Int]:
    # UNPACK of a Python entry's argument words (the Python entries remain
    # for their other callers and the boundary tests).
    var out = List[Int]()
    for i in range(len(value)):  # small-loop(a Python entry's argument words)
        out.append(Int(py=value[i]))
    return out^


def embedding_neural_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                        result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """This image's Embedding descriptor entry (core/neural_entry_table.mojo
    family NEURAL_TABLE_EMBEDDING; the operation layouts are documented
    there): the same word functions the Python entries run."""
    try:
        if op < 0 or op >= NEURAL_EMB_OPERATIONS:
            raise Error("embedding entry: unknown operation")
        var spans = neural_entry_spans(aa, na, pp, np, ff, nf, result, 0)
        # perf-audit-3 (pattern 3): the two spans are read in place (no copy per call).
        ref a = spans[0]
        ref p = spans[1]
        var value = 0
        if op == EMB_OP_FORWARD:
            value = embedding_forward_words(a, p)
        elif op == EMB_OP_BACKWARD:
            value = embedding_backward_words(a, p)
        elif op == EMB_OP_TABLE_RELEASE:
            if len(a) != 0 or len(p) != 1:
                raise Error("embedding_table_release: expects one token")
            value = embedding_table_release_words(p[0])
        elif op == EMB_OP_FORWARD_DEV:
            if len(p) < 1:
                raise Error("embedding_forward_dev: requires 2 addresses (weight, ids), 1 handle (y) and [V, d, T, token]")
            value = embedding_forward_dev_words(a, [p[0]], _emb_tail(p, 1))
        elif op == EMB_OP_BACKWARD_DEV:
            if len(p) < 2:
                raise Error("embedding_backward_dev: requires 1 address (ids), 2 handles (dy, dw) and 6 params")
            value = embedding_backward_dev_words(a, [p[0], p[1]], _emb_tail(p, 2))
        elif op == EMB_OP_TENSOR_ALLOC:
            if len(a) != 0 or len(p) != 2:
                raise Error("seq_tensor_alloc: expects n and zero")
            value = seq_tensor_alloc_words(p[0], p[1])
        elif op == EMB_OP_TENSOR_FREE:
            if len(a) != 0 or len(p) != 1:
                raise Error("seq_tensor_free: expects one handle")
            value = seq_tensor_free_words(p[0])
        elif op == EMB_OP_TENSOR_UPLOAD:
            if len(a) != 1 or len(p) != 2:
                raise Error("seq_tensor_upload: expects one source address, a handle and n")
            value = seq_tensor_upload_words(p[0], a[0], p[1])
        else:
            if len(a) != 0 or len(p) != 3:
                raise Error("seq_tensor_copy: expects dst, src and n")
            value = seq_tensor_copy_words(p[0], p[1], p[2])
        neural_entry_result(result, value)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct EmbeddingNeuralAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(embedding_neural_v1)


comptime NATIVE_EMBEDDING_NEURAL_API = SharedSlot[EmbeddingNeuralAPI, ("MojolearnEmbeddingNeuralAPIIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnEmbeddingNeuralAPIFastV1")]


def publish_neural_embedding_device() raises:
    """Module init: publish under (tier, COMPILED_VENDOR) in the neural
    entry table. A build with no GPU vendor publishes nothing."""
    var vendor = neural_entry_vendor(String(COMPILED_VENDOR))
    if vendor <= 0:
        return
    var api = NATIVE_EMBEDDING_NEURAL_API.get_or_create_ptr()
    neural_table_publish[NEURAL_TABLE_EMBEDDING](NeuralEntryKey(Int(GLOBAL_NUMERIC_MODE), vendor), Int(api), api[].entries[0])


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_embedding.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_embedding() -> PythonObject:
    comptime assert NEURAL_GUARDS_OK, "invalid integrated experiment configuration (neural guards)"
    try:
        var m = PythonModuleBuilder("_mojolearn_embedding")
        # Neural-blocks entry table (core/neural_entry_table.mojo); the Python
        # entries remain for their other callers and the boundary tests.
        publish_neural_embedding_device()
        m.def_function[embedding_vendor_binding]("embedding_vendor")
        m.def_function[embedding_numeric_mode_binding]("embedding_numeric_mode")
        m.def_function[embedding_forward_binding]("embedding_forward")
        m.def_function[embedding_backward_binding]("embedding_backward")
        m.def_function[embedding_table_release_binding]("embedding_table_release")
        # TOMBSTONE: embedding_resident (dead native entry: a compiled-constant probe nothing reads; EMB_RESIDENT stays) deleted 2026-10-26 by 9aba8da70; code recoverable at a88745022.
        # Restore: git apply experiments/removed/dead_entries_embedding.patch
        # lane gap-neural-io: resident tensors and the device-I/O entries
        m.def_function[seq_tensor_alloc_binding]("seq_tensor_alloc")
        m.def_function[seq_tensor_free_binding]("seq_tensor_free")
        m.def_function[seq_tensor_upload_binding]("seq_tensor_upload")
        m.def_function[seq_tensor_download_binding]("seq_tensor_download")
        m.def_function[seq_tensor_copy_binding]("seq_tensor_copy")
        m.def_function[embedding_forward_dev_binding]("embedding_forward_dev")
        m.def_function[embedding_backward_dev_binding]("embedding_backward_dev")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_embedding: ", e))

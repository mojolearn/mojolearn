# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_embedding` family: the Embedding layer's
gather and fold (lane/cpu-training-embedding-ivf, 2026-09-15; the embedding
and embedding-sort lanes of tools/identity_break.py).

HOST ONLY. No DeviceContext, no kernel launch, no GPU. The arithmetic is
`embedding/host/embedding_host.mojo`, the device launch of
`embedding/checks/embedding_identical.mojo` (and PLAN_SORT's
`autograd/kernels/embedding_sort.mojo`) restated on the host; that file's
header names every kernel it restates. The refusals are the GPU binding's
(`bindings/_mojolearn_embedding.mojo`), in its words and its order, through
the same `embedding/checks/embedding_oracle.mojo` functions, so a bad call
raises the same error and nothing is written on a refusal.

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES: `embedding_forward` and
`embedding_backward` with the SAME address and `params` lists, plus the
read-backs `embedding_numeric_mode` (1) and `embedding_vendor` ("cpu"), so
`python/mojolearn/embedding.py` runs unchanged on a CPU-only install through
`_backend._HOST_MODULES` (`"_mojolearn_embedding":
"_mojolearn_embedding_host"`). The GPU binding exports nothing else.
"""
from core.experiment_guards_neural import NEURAL_GUARDS_OK
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from core.hostptr import f32_ptr, read_i32
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from embedding.checks.embedding_oracle import (
    EMB_NO_PADDING_IDX,
    emb_refuse_ids,
    emb_refuse_shape,
    refuse_nonfinite_at,
)
from bindings.embedding_host_forward_door import embedding_host_words
from bindings.embedding_host_forward_door import embedding_forward_binding, embedding_host_forward_words
# RUN (lane rs-embedding-bytelm): typed Mojo only, no Python import.
from embedding.host.embedding_host_words_run import embedding_host_backward_released, embedding_host_config
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.neural_entry_table import (
    NeuralEntryKey, neural_table_publish, neural_entry_spans, neural_entry_result,
    NEURAL_TABLE_EMBEDDING, EMB_OP_FORWARD, EMB_OP_BACKWARD,
)
from embedding.host.embedding_host import (
    EMBEDDING_HOST_SABOTAGE,
    HOST_PLAN_SCAN,
    HOST_PLAN_SORT,
)


def embedding_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "embedding host: this binding restates the IDENTICAL arms only;"
        " bindings/build_host_family.sh passes -D MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def embedding_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def embedding_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives
    in a function body PyInit registers, so it is compiled in every build.
    `bindings/build_host_family.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "embedding host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_embedding_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def embedding_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary stores `-0.0` over the padding row on purpose
    (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative control)."""
    return PythonObject(EMBEDDING_HOST_SABOTAGE)


# The GPU binding's names, same contract.


def embedding_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER as the `NUMERIC_*` code: always 1 here."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def embedding_vendor_binding() raises -> PythonObject:
    """"cpu", the answer `_backend.read_vendor` expects from a host binding."""
    return PythonObject(String("cpu"))


def embedding_host_backward_words(a: List[Int], p: List[Int]) raises -> Int:
    """`dW`, the device fold restated. Returns `V * d`.

    `addrs` = [dy (T * d f32, read), ids (T i32, read), dw (V * d f32,
    WRITTEN; READ FIRST when accumulate is 1)]; `params` = [V, d, T,
    padding_idx (-1 for none), accumulate (0 or 1)[, plan (0 PLAN_SCAN, the
    default, or 1 PLAN_SORT)]]. The GPU binding's order."""
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
    var plan = HOST_PLAN_SCAN
    if len(p) == 6:
        var plan_code = p[5]
        if plan_code != HOST_PLAN_SCAN and plan_code != HOST_PLAN_SORT:
            raise Error(
                String("embedding_backward: plan must be 0 (PLAN_SCAN) or 1")
                + " (PLAN_SORT), got "
                + String(plan_code)
            )
        plan = plan_code
    if acc_code != 0 and acc_code != 1:
        raise Error(
            String("embedding_backward: accumulate must be 0 or 1, got ")
            + String(acc_code)
        )
    var cfg = embedding_host_config(vocab, width, padding_idx, acc_code == 1)
    emb_refuse_shape(cfg, n_positions)
    # Lane ne3-c-blocks-embedding (row N164): dY and the carried dW are read
    # and folded through the caller's buffers (no list copies, no store-back
    # walk); the refusals keep their words, their order and their first index
    # (`refuse_nonfinite_at`), and nothing is written before they pass.
    var dyp = f32_ptr(a[0])
    var ids = read_i32(a[1], n_positions)
    emb_refuse_ids(ids, cfg)
    refuse_nonfinite_at(String("dY"), dyp, n_positions * width)
    var dwp = f32_ptr(a[2])
    if cfg.accumulate:
        refuse_nonfinite_at(String("the carried dW"), dwp, vocab * width)
    var cells = vocab * width
    with GILReleased(Python()):
        embedding_host_backward_released(dwp, dyp, ids, n_positions, cfg, plan, cells)
    return vocab * width


def embedding_backward_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    return PythonObject(embedding_host_backward_words(embedding_host_words(addrs), embedding_host_words(params)))



def embedding_host_neural_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
        result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """This image's Embedding descriptor entry (core/neural_entry_table.mojo
    family NEURAL_TABLE_EMBEDDING): FORWARD and BACKWARD, the same word functions the Python
    entries run. Every other operation refuses."""
    try:
        var spans = neural_entry_spans(aa, na, pp, np, ff, nf, result, 0)
        # perf-audit-3 (pattern 3): the two spans are read in place (no copy per call).
        ref a = spans[0]
        ref p = spans[1]
        var value: Int
        if op == EMB_OP_FORWARD:
            value = embedding_host_forward_words(a, p)
        elif op == EMB_OP_BACKWARD:
            value = embedding_host_backward_words(a, p)
        else:
            raise Error("embedding host entry: unavailable operation")
        neural_entry_result(result, value)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct EmbeddingHostNeuralAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(embedding_host_neural_v1)


comptime NATIVE_EMBEDDINGHOSTNEURAL_API = SharedSlot[EmbeddingHostNeuralAPI, ("MojolearnEmbeddingHostNeuralAPIIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnEmbeddingHostNeuralAPIFastV1")]


def publish_neural_embedding_host() raises:
    """Module init of `_mojolearn_embedding_host`: publish under (tier, cpu)."""
    var api = NATIVE_EMBEDDINGHOSTNEURAL_API.get_or_create_ptr()
    neural_table_publish[NEURAL_TABLE_EMBEDDING](NeuralEntryKey(Int(GLOBAL_NUMERIC_MODE), 0), Int(api), api[].entries[0])


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_embedding_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_embedding_host() -> PythonObject:
    comptime assert NEURAL_GUARDS_OK, "invalid integrated experiment configuration (neural guards)"
    try:
        var m = PythonModuleBuilder("_mojolearn_embedding_host")
        # Neural-blocks entry table (core/neural_entry_table.mojo); the Python
        # entries remain for their other callers and the boundary tests.
        publish_neural_embedding_host()
        m.def_function[embedding_host_numeric_mode_binding]("embedding_host_numeric_mode")
        m.def_function[embedding_host_vendor_binding]("embedding_host_vendor")
        m.def_function[embedding_host_column_binding]("embedding_host_column")
        m.def_function[embedding_host_sabotage_binding]("embedding_host_sabotage")
        m.def_function[embedding_vendor_binding]("embedding_vendor")
        m.def_function[embedding_numeric_mode_binding]("embedding_numeric_mode")
        m.def_function[embedding_forward_binding]("embedding_forward")
        m.def_function[embedding_backward_binding]("embedding_backward")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_embedding_host: ", e))

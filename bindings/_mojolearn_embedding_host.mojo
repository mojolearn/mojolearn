# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_embedding` family: the Embedding layer's
gather and fold (lane/cpu-training-embedding-ivf, 2026-09-15; the embedding
and embedding-sort lanes of tools/identity_break.py).

HOST ONLY. No DeviceContext, no kernel launch, no GPU. The arithmetic is
`embedding/host/embedding_host.mojo`, the device launch of
`embedding/checks/embedding_identical.mojo` (and PLAN_SORT's
`embedding/checks/embedding_sort.mojo`) restated on the host; that file's
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
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.hostptr import f32_ptr, read_f32, read_i32
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from embedding.checks.embedding_oracle import (
    EMB_NO_PADDING_IDX,
    EmbConfig,
    emb_refuse_ids,
    emb_refuse_shape,
    refuse_nonfinite,
)
from embedding.host.embedding_host import (
    EMBEDDING_HOST_SABOTAGE,
    HOST_PLAN_SCAN,
    HOST_PLAN_SORT,
    host_embedding_backward,
    host_embedding_forward,
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


def _config(vocab: Int, width: Int, padding_idx: Int, accumulate: Bool) raises -> EmbConfig:
    """The GPU binding's `_config`, in its words."""
    if padding_idx != EMB_NO_PADDING_IDX and (padding_idx < 0 or padding_idx >= vocab):
        raise Error(
            String("embedding: padding_idx = ")
            + String(padding_idx)
            + " is outside [0, "
            + String(vocab)
            + ") REFUSED (contract 8; -1 means no padding row)"
        )
    return EmbConfig(vocab, width, padding_idx, accumulate)


def embedding_forward_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`Y[t, j] = W[ids[t], j]`, the device gather restated. Returns `T * d`.

    `addrs` = [weight (V * d f32, read), ids (T i32, read), y_out (T * d
    f32, WRITTEN)]; `params` = [V, d, T]. The GPU binding's order."""
    if len(addrs) != 3:
        raise Error(
            "embedding_forward: addrs must contain 3 addresses (weight, ids,"
            " y_out), got " + String(len(addrs))
        )
    if len(params) != 3:
        raise Error(
            "embedding_forward: params must contain 3 values (V, d, T), got "
            + String(len(params))
        )
    var vocab = Int(py=params[0])
    var width = Int(py=params[1])
    var n_positions = Int(py=params[2])
    var cfg = _config(vocab, width, EMB_NO_PADDING_IDX, False)
    emb_refuse_shape(cfg, n_positions)
    var weight = read_f32(Int(py=addrs[0]), vocab * width)
    var ids = read_i32(Int(py=addrs[1]), n_positions)
    emb_refuse_ids(ids, cfg)
    refuse_nonfinite(String("W"), weight)
    var yp = f32_ptr(Int(py=addrs[2]))
    with GILReleased(Python()):
        host_embedding_forward(weight, ids, n_positions, cfg, yp)
    return PythonObject(n_positions * width)


def embedding_backward_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`dW`, the device fold restated. Returns `V * d`.

    `addrs` = [dy (T * d f32, read), ids (T i32, read), dw (V * d f32,
    WRITTEN; READ FIRST when accumulate is 1)]; `params` = [V, d, T,
    padding_idx (-1 for none), accumulate (0 or 1)[, plan (0 PLAN_SCAN, the
    default, or 1 PLAN_SORT)]]. The GPU binding's order."""
    if len(addrs) != 3:
        raise Error(
            "embedding_backward: addrs must contain 3 addresses (dy, ids, dw),"
            " got " + String(len(addrs))
        )
    if len(params) != 5 and len(params) != 6:
        raise Error(
            "embedding_backward: params must contain 5 or 6 values (V, d, T,"
            " padding_idx, accumulate[, plan]), got " + String(len(params))
        )
    var vocab = Int(py=params[0])
    var width = Int(py=params[1])
    var n_positions = Int(py=params[2])
    var padding_idx = Int(py=params[3])
    var acc_code = Int(py=params[4])
    var plan = HOST_PLAN_SCAN
    if len(params) == 6:
        var plan_code = Int(py=params[5])
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
    var cfg = _config(vocab, width, padding_idx, acc_code == 1)
    emb_refuse_shape(cfg, n_positions)
    var dy = read_f32(Int(py=addrs[0]), n_positions * width)
    var ids = read_i32(Int(py=addrs[1]), n_positions)
    emb_refuse_ids(ids, cfg)
    refuse_nonfinite(String("dY"), dy)
    var dwp = f32_ptr(Int(py=addrs[2]))
    var dw: List[Float32]
    if cfg.accumulate:
        dw = read_f32(Int(py=addrs[2]), vocab * width)
        refuse_nonfinite(String("the carried dW"), dw)
    else:
        dw = List[Float32](length=vocab * width, fill=Float32(0.0))
    var cells = vocab * width
    with GILReleased(Python()):
        if cells > 0:
            host_embedding_backward(dw, dy, ids, n_positions, cfg, plan)
            for i in range(cells):
                dwp.unsafe_store(i, dw[i])
    return PythonObject(vocab * width)


@export
def PyInit__mojolearn_embedding_host() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_embedding_host")
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

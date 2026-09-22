# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU INFERENCE binding for the Embedding layer
(lane/inference-embedding-ivf-cholesky, 2026-09-15): lookup in a saved
table, with no backward in the binary.

HOST ONLY, IDENTICAL ONLY. It registers the GPU binding's
`embedding_forward`, `embedding_vendor` and `embedding_numeric_mode` names,
the forward from `bindings/embedding_host_forward.mojo`, the same source the
internal reference binding `bindings/_mojolearn_embedding_host.mojo`
registers it from. `embedding_backward` is absent, so a CPU-only install
holding only this binding refuses a gradient by name.

Why a binding of its own: the reference binding carries the backward fold
(both execution plans), and training-only code does not ship in the
inference wheels. The
manifest declares this family with `routes=None` and
`serves=("_mojolearn_embedding",)`.

The sabotage arm (`embedding_infer_host_sabotage`) is
`embedding/host/embedding_host.mojo::EMBEDDING_HOST_SABOTAGE`: under
`-D MOJOLEARN_HOST_SABOTAGE=1` every lookup gathers the next row.
"""
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from bindings.embedding_host_forward import embedding_forward_binding
from embedding.host.embedding_host import EMBEDDING_HOST_SABOTAGE


def embedding_infer_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "embedding infer host: this binding restates the IDENTICAL arms only;"
        " bindings/build_host_family.sh passes -D MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def embedding_infer_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def embedding_infer_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu"."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "embedding infer host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_embedding_infer_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def embedding_infer_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary gathers the next row on purpose
    (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative control)."""
    return PythonObject(EMBEDDING_HOST_SABOTAGE)


def embedding_vendor_binding() raises -> PythonObject:
    """"cpu", as every host binding answers."""
    return PythonObject(String("cpu"))


def embedding_numeric_mode_binding() raises -> PythonObject:
    """The build's tier as the `NUMERIC_*` code: always 1 here."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


@export
def PyInit__mojolearn_embedding_infer_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_embedding_infer_host")
        module.def_function[embedding_infer_host_numeric_mode_binding]("embedding_infer_host_numeric_mode")
        module.def_function[embedding_infer_host_vendor_binding]("embedding_infer_host_vendor")
        module.def_function[embedding_infer_host_column_binding]("embedding_infer_host_column")
        module.def_function[embedding_infer_host_sabotage_binding]("embedding_infer_host_sabotage")
        module.def_function[embedding_vendor_binding]("embedding_vendor")
        module.def_function[embedding_numeric_mode_binding]("embedding_numeric_mode")
        module.def_function[embedding_forward_binding]("embedding_forward")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_embedding_infer_host: ", error))

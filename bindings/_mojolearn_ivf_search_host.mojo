# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU INFERENCE binding for IVFIndex (lane/inference-embedding-ivf-cholesky,
2026-09-15): search over a saved, GPU-built index, with no build in the
binary.

HOST ONLY, IDENTICAL ONLY. It registers the GPU binding's `ivf_flat_search`,
`ivf_vendor` and `ivf_numeric_mode` names, the search from
`bindings/ivf_host_search.mojo`, the same source the internal reference
binding `bindings/_mojolearn_ivf_host.mojo` registers it from.
`ivf_flat_build` and `ivf_flat_build_and_search` are absent, so a CPU-only
install holding only this binding refuses a build by name.

Why a binding of its own: the reference binding carries the k-means
quantizer fit, and training-only code does not ship in the inference wheels.
The manifest declares this
family with `routes=None` and `serves=("_mojolearn_ivf",)`.

The sabotage arm (`ivf_search_host_sabotage`) is
`ivf/host/ivf_host.mojo::IVF_HOST_SABOTAGE`: under
`-D MOJOLEARN_HOST_SABOTAGE=1` every candidate distance walks its feature
axis descending, and every returned distance has its bits moved
(`ivf_sabotage_value_flip`), so the integer `ties` fixture moves too.
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
from bindings.ivf_host_search import (
    ivf_finalize_distances_binding,
    ivf_flat_extend_binding,
    ivf_flat_partial_search_binding,
    ivf_flat_search_binding,
)
from ivf.host.ivf_host import IVF_HOST_SABOTAGE


def ivf_search_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "ivf search host: this binding restates the IDENTICAL arms only;"
        " bindings/build_host_family.sh passes -D MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def ivf_search_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def ivf_search_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu"."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "ivf search host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_ivf_search_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def ivf_search_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary walks every candidate distance's feature axis
    descending on purpose (-D MOJOLEARN_HOST_SABOTAGE=1)."""
    return PythonObject(IVF_HOST_SABOTAGE)


def ivf_vendor_binding() raises -> PythonObject:
    """"cpu", as every host binding answers."""
    return PythonObject(String("cpu"))


def ivf_numeric_mode_binding() raises -> PythonObject:
    """The build's tier as the `NUMERIC_*` code: always 1 here."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


@export
def PyInit__mojolearn_ivf_search_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_ivf_search_host")
        module.def_function[ivf_search_host_numeric_mode_binding]("ivf_search_host_numeric_mode")
        module.def_function[ivf_search_host_vendor_binding]("ivf_search_host_vendor")
        module.def_function[ivf_search_host_column_binding]("ivf_search_host_column")
        module.def_function[ivf_search_host_sabotage_binding]("ivf_search_host_sabotage")
        module.def_function[ivf_vendor_binding]("ivf_vendor")
        module.def_function[ivf_numeric_mode_binding]("ivf_numeric_mode")
        module.def_function[ivf_flat_search_binding]("ivf_flat_search")
        module.def_function[ivf_flat_extend_binding]("ivf_flat_extend")
        module.def_function[ivf_flat_partial_search_binding]("ivf_flat_partial_search")
        module.def_function[ivf_finalize_distances_binding]("ivf_finalize_distances")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_ivf_search_host: ", error))

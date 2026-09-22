# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""INFERENCE-ONLY CPU binding for the `_mojolearn_hdbscan` family:
`mojolearn.hdbscan.approximate_predict`, `membership_vector` and
`all_points_membership_vectors` of a saved HDBSCAN on a CPU-only
install (the neighbors and density inference lane, 2026-09-15).

This is the hdbscan binary a wheel ships. It registers
`bindings/hdbscan_host_predict.mojo`'s `hdbscan_approximate_predict`, the
same function the reference binding `bindings/_mojolearn_hdbscan_host.mojo`
registers, with its two soft clustering entries, and nothing that fits: `hdbh_fit`, the Boruvka MST, the condensed
tree and `generate_prediction_data` are not imported, so they are not
compiled into this file. `mojolearn.host_model` loads a saved HDBSCAN into a
host class that binds this file (`python/mojolearn/_classical_host.py`).

The sabotage arm (`hdbscan_infer_host_sabotage`) is
`hdbscan/host/hdbscan_host_oracle.mojo::HDBH_HOST_SABOTAGE`
(`-D MOJOLEARN_HOST_SABOTAGE=1`), which reads every core distance of the
held-out point's neighbor search one neighbor early.
"""
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from bindings.hdbscan_host_predict import (
    hdbscan_all_points_membership_vectors_binding,
    hdbscan_approximate_predict_binding,
    hdbscan_membership_vector_binding,
)
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from hdbscan.host.hdbscan_host_oracle import HDBH_HOST_SABOTAGE


def hdbscan_infer_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def hdbscan_infer_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def hdbscan_infer_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu"."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "hdbscan inference host: this binding compiles the CPU column only;"
        " pass -D MOJOLEARN_COLUMN_CPU (bindings/build_hdbscan_infer_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def hdbscan_infer_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary was built with -D MOJOLEARN_HOST_SABOTAGE=1, the
    gate's negative control."""
    return PythonObject(HDBH_HOST_SABOTAGE)


def hdbscan_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def hdbscan_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


@export
def PyInit__mojolearn_hdbscan_infer_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_hdbscan_infer_host")
        module.def_function[hdbscan_infer_host_numeric_mode_binding]("hdbscan_infer_host_numeric_mode")
        module.def_function[hdbscan_infer_host_vendor_binding]("hdbscan_infer_host_vendor")
        module.def_function[hdbscan_infer_host_column_binding]("hdbscan_infer_host_column")
        module.def_function[hdbscan_infer_host_sabotage_binding]("hdbscan_infer_host_sabotage")
        module.def_function[hdbscan_vendor_binding]("hdbscan_vendor")
        module.def_function[hdbscan_numeric_mode_binding]("hdbscan_numeric_mode")
        module.def_function[hdbscan_approximate_predict_binding]("hdbscan_approximate_predict")
        module.def_function[hdbscan_membership_vector_binding]("hdbscan_membership_vector")
        module.def_function[hdbscan_all_points_membership_vectors_binding]("hdbscan_all_points_membership_vectors")
        return module.finalize()
    except e:
        abort(String("failed to create _mojolearn_hdbscan_infer_host: ", e))

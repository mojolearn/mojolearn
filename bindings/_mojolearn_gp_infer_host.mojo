# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""INFERENCE-ONLY CPU binding for the `_mojolearn_gp` family: a saved
GaussianProcessRegressor's predictive mean and std and a saved
GaussianProcessClassifier's latent mean, variance and class probability on a
CPU-only install (the
neighbors and density inference lane, 2026-09-15).

This is the gp binary a wheel ships. It registers
`bindings/gp_host_predict.mojo`'s `gpr_predict` and `gpc_predict`, the
functions the reference binding `bindings/_mojolearn_gp_host.mojo` registers,
and no fit: `gpr_host_fit`, `gpc_host_fit` (the Laplace Newton loop), the log
marginal likelihood and the Cholesky door are not imported, so they are not
compiled into this file. The classifier's label decoding is host Python. normalize_y's scale-back is host Python in
`GaussianProcessRegressor.predict`, shared with the GPU class. `mojolearn.host_model`
loads a saved model into a host class that binds this file
(`python/mojolearn/_classical_host.py`).

The sabotage arm (`gp_infer_host_sabotage`) is the reference binding's:
`gaussian_process/host/gpr_oracle.mojo::GPR_ORACLE_HOST_SABOTAGE`
(`-D MOJOLEARN_HOST_SABOTAGE=1`).
"""
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from bindings.gp_host_predict import gpc_predict_binding, gpr_predict_binding
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from gaussian_process.host.gpr_oracle import GPR_ORACLE_HOST_SABOTAGE


def gp_infer_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def gp_infer_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def gp_infer_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu"."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "gp inference host: this binding compiles the CPU column only;"
        " pass -D MOJOLEARN_COLUMN_CPU (bindings/build_gp_infer_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def gp_infer_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary was built with -D MOJOLEARN_HOST_SABOTAGE=1, the
    gate's negative control."""
    return PythonObject(GPR_ORACLE_HOST_SABOTAGE)


def gp_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def gp_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


@export
def PyInit__mojolearn_gp_infer_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_gp_infer_host")
        module.def_function[gp_infer_host_numeric_mode_binding]("gp_infer_host_numeric_mode")
        module.def_function[gp_infer_host_vendor_binding]("gp_infer_host_vendor")
        module.def_function[gp_infer_host_column_binding]("gp_infer_host_column")
        module.def_function[gp_infer_host_sabotage_binding]("gp_infer_host_sabotage")
        module.def_function[gp_vendor_binding]("gp_vendor")
        module.def_function[gp_numeric_mode_binding]("gp_numeric_mode")
        module.def_function[gpr_predict_binding]("gpr_predict")
        module.def_function[gpc_predict_binding]("gpc_predict")
        return module.finalize()
    except e:
        abort(String("failed to create _mojolearn_gp_infer_host: ", e))

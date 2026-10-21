# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_preprocessing` family: StandardScaler and
MinMaxScaler (workstream E batch 2, lane/cpu-training-e2, 2026-09-14; the
standard-scaler and minmax-scaler lanes of tools/identity_break.py).

HOST ONLY. No DeviceContext, no kernel launch, no GPU. The arithmetic is
`preprocessing/host/scaler_oracle.mojo`, the second spelling of the four
kernels of `preprocessing/standard.mojo` and `preprocessing/minmax.mojo`;
that file's header names every original by file and line. The validation
is `preprocessing/estimator.mojo`'s and the GPU binding's, in their words
and order, so a bad call raises the same error and nothing is written on a
refusal.

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES
(`bindings/_mojolearn_preprocessing.mojo`): `standard_fit`,
`standard_transform`, `minmax_fit`, `minmax_transform`, each with the SAME
address contract and `params` list, plus the read-backs
`preprocessing_numeric_mode` (1) and `preprocessing_vendor` ("cpu"), so
`python/mojolearn/preprocessing.py` runs unchanged on a CPU-only install
through `_backend._HOST_MODULES` (`"_mojolearn_preprocessing":
"_mojolearn_preprocessing_host"`). The GPU binding exports nothing else.
"""
from std.math import isfinite
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from core.hostptr import copy_f32, f32_ptr, read_f32
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE
from preprocessing.host.scaler_oracle import (
    SCALER_ORACLE_HOST_SABOTAGE,
    host_minmax_fit,
    host_minmax_transform,
    host_standard_fit,
    host_standard_transform,
)
from core.mojolearn_preprocessing_host_impl import (
    ptr, load, validate_dimensions, finite_values, validate_standard, standard_finite,
    minmax_fit_host_into, minmax_transform_host_into, standard_fit_host_into, standard_transform_host_into,
)


def _index(value: PythonObject) raises -> Int:
    var type_name = String(py=value.__class__.__name__)
    if type_name == "bool" or type_name == "bool_":
        raise Error("preprocessing host: integers expected, not booleans")
    var operator_module = Python.import_module("operator")
    return Int(py=operator_module.index(value))


def preprocessing_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def preprocessing_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def preprocessing_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu".
    THE COLUMN IS THE CPU COLUMN, OR THIS DOES NOT BUILD; the assert lives
    in a function body PyInit registers, so it is compiled in every build.
    `bindings/build_preprocessing_host.sh` passes -D MOJOLEARN_COLUMN_CPU."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "preprocessing host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_preprocessing_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


# There is no `preprocessing_host_detected_column` read-back, for the reason
# 8d16ce2f removed it from the forest and byte LM host bindings: the
# detected column folds to the GPU of the machine that ran the build, so
# its name would land in the vendor-neutral binary. The comptime assert
# above is the check.


def preprocessing_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary was built with -D MOJOLEARN_HOST_SABOTAGE=1 (the
    gate's negative control): the slab tree's chunk boundaries shifted by
    one value and the min-max offset's subtraction turned into an
    addition; refused outside the gate as one set."""
    return PythonObject(SCALER_ORACLE_HOST_SABOTAGE)


# The GPU binding's names, same contract.


def preprocessing_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def preprocessing_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def minmax_fit_binding(x_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    """`minmax_fit` on the host. params=[n,d,feature_min,feature_max];
    output rows=min,max,range,scale,min_."""
    if len(params) != 4:
        raise Error("minmax_fit: requires 4 parameters")
    var n = _index(params[0])
    var d = _index(params[1])
    var lower = Float32(Float64(py=params[2]))
    var upper = Float32(Float64(py=params[3]))
    validate_dimensions(n, d, lower, upper)
    var x = load(_index(x_addr), n * d)
    var output = ptr(_index(out_addr))
    with GILReleased(Python()):
        minmax_fit_host_into(x, output, n, d, lower, upper)
    return PythonObject(5 * d)


def minmax_transform_binding(
    x_addr: PythonObject, scale_addr: PythonObject, min_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`minmax_transform` on the host.
    params=[n,d,inverse,clip,feature_min,feature_max]; output=n*d row-major."""
    if len(params) != 6:
        raise Error("minmax_transform: requires 6 parameters")
    var n = _index(params[0])
    var d = _index(params[1])
    var inverse = _index(params[2])
    var clip = _index(params[3])
    var lower = Float32(Float64(py=params[4]))
    var upper = Float32(Float64(py=params[5]))
    validate_dimensions(n, d, lower, upper)
    var x = load(_index(x_addr), n * d)
    var scale = load(_index(scale_addr), d)
    var offset = load(_index(min_addr), d)
    var output = ptr(_index(out_addr))
    with GILReleased(Python()):
        minmax_transform_host_into(x, scale, offset, output, n, d, inverse, clip, lower, upper)
    return PythonObject(n * d)


def standard_fit_binding(x_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    """`standard_fit` on the host. params=[n,d,with_mean,with_std]; output
    rows=mean,var,scale."""
    if len(params) != 4:
        raise Error("standard_fit: requires 4 parameters")
    var n = _index(params[0])
    var d = _index(params[1])
    var with_mean = _index(params[2])
    var with_std = _index(params[3])
    validate_standard(n, d, with_mean, with_std)
    var x = load(_index(x_addr), n * d)
    var output = ptr(_index(out_addr))
    with GILReleased(Python()):
        standard_fit_host_into(x, output, n, d, with_mean, with_std)
    return PythonObject(3 * d)


def standard_transform_binding(
    x_addr: PythonObject, mean_addr: PythonObject, scale_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`standard_transform` on the host.
    params=[n,d,inverse,with_mean,with_std]; output n*d row-major."""
    if len(params) != 5:
        raise Error("standard_transform: requires 5 parameters")
    var n = _index(params[0])
    var d = _index(params[1])
    var inverse = _index(params[2])
    var with_mean = _index(params[3])
    var with_std = _index(params[4])
    validate_standard(n, d, with_mean, with_std)
    var x = load(_index(x_addr), n * d)
    var mean = load(_index(mean_addr), d)
    var scale = load(_index(scale_addr), d)
    var output = ptr(_index(out_addr))
    with GILReleased(Python()):
        standard_transform_host_into(x, mean, scale, output, n, d, inverse, with_mean, with_std)
    return PythonObject(n * d)


# The module body. Its @export PyInit lives in bindings/entry/entry__mojolearn_preprocessing_host.mojo
# (the per-binding build) and in the merged roots tools/merged_bindings.py
# writes from bindings/merged_groups.json (one shared file per directory set).
def build__mojolearn_preprocessing_host() -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_preprocessing_host")
        m.def_function[preprocessing_host_numeric_mode_binding]("preprocessing_host_numeric_mode")
        m.def_function[preprocessing_host_vendor_binding]("preprocessing_host_vendor")
        m.def_function[preprocessing_host_column_binding]("preprocessing_host_column")
        m.def_function[preprocessing_host_sabotage_binding]("preprocessing_host_sabotage")
        m.def_function[standard_fit_binding]("standard_fit")
        m.def_function[standard_transform_binding]("standard_transform")
        m.def_function[minmax_fit_binding]("minmax_fit")
        m.def_function[minmax_transform_binding]("minmax_transform")
        m.def_function[preprocessing_numeric_mode_binding]("preprocessing_numeric_mode")
        m.def_function[preprocessing_vendor_binding]("preprocessing_vendor")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_preprocessing_host: ", e))

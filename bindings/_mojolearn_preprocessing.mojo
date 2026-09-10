# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Borrowed Float32 arrays; GPU arithmetic; no context or pointer retained."""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE
from preprocessing.estimator import validate_dimensions, minmax_fit_host, minmax_transform_host, validate_standard, standard_fit_host, standard_transform_host


def ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("preprocessing: null Float32 pointer")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def load(addr: Int, n: Int) raises -> List[Float32]:
    var p = ptr(addr)
    var result = List[Float32]()
    for i in range(n):
        result.append(p.unsafe_load(i))
    return result^


def fit_binding(x_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    # params=[n,d,feature_min,feature_max]; output rows=min,max,range,scale,min_.
    if len(params) != 4:
        raise Error("minmax_fit: requires 4 parameters")
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    var lower = Float32(Float64(py=params[2]))
    var upper = Float32(Float64(py=params[3]))
    validate_dimensions(n,d,lower,upper)
    var x = load(Int(py=x_addr),n*d)
    var output = ptr(Int(py=out_addr))
    with GILReleased(Python()):
        var result = minmax_fit_host(x,n,d,lower,upper)
        for i in range(5*d):
            output.unsafe_store(i,result[i])
    return PythonObject(5*d)


def transform_binding(
    x_addr: PythonObject, scale_addr: PythonObject, min_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    # params=[n,d,inverse,clip,feature_min,feature_max]; output=n*d row-major.
    if len(params) != 6:
        raise Error("minmax_transform: requires 6 parameters")
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    var inverse = Int(py=params[2])
    var clip = Int(py=params[3])
    var lower = Float32(Float64(py=params[4]))
    var upper = Float32(Float64(py=params[5]))
    validate_dimensions(n,d,lower,upper)
    var x = load(Int(py=x_addr),n*d)
    var scale = load(Int(py=scale_addr),d)
    var offset = load(Int(py=min_addr),d)
    var output = ptr(Int(py=out_addr))
    with GILReleased(Python()):
        var result = minmax_transform_host(x,scale,offset,n,d,inverse,clip,lower,upper)
        for i in range(n*d):
            output.unsafe_store(i,result[i])
    return PythonObject(n*d)


def standard_fit_binding(x_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    # params=[n,d,with_mean,with_std]; output rows=mean,var,scale.
    if len(params) != 4:
        raise Error("standard_fit: requires 4 parameters")
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    var with_mean = Int(py=params[2])
    var with_std = Int(py=params[3])
    validate_standard(n,d,with_mean,with_std)
    var x = load(Int(py=x_addr),n*d)
    var output = ptr(Int(py=out_addr))
    with GILReleased(Python()):
        var result = standard_fit_host(x,n,d,with_mean,with_std)
        for i in range(3*d):
            output.unsafe_store(i,result[i])
    return PythonObject(3*d)


def standard_transform_binding(
    x_addr: PythonObject, mean_addr: PythonObject, scale_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    # params=[n,d,inverse,with_mean,with_std]; output n*d row-major.
    if len(params) != 5:
        raise Error("standard_transform: requires 5 parameters")
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    var inverse = Int(py=params[2])
    var with_mean = Int(py=params[3])
    var with_std = Int(py=params[4])
    validate_standard(n,d,with_mean,with_std)
    var x = load(Int(py=x_addr),n*d)
    var mean = load(Int(py=mean_addr),d)
    var scale = load(Int(py=scale_addr),d)
    var output = ptr(Int(py=out_addr))
    with GILReleased(Python()):
        var result = standard_transform_host(x,mean,scale,n,d,inverse,with_mean,with_std)
        for i in range(n*d):
            output.unsafe_store(i,result[i])
    return PythonObject(n*d)


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_binding() raises -> PythonObject:
    return PythonObject(String(COMPILED_VENDOR))


@export
def PyInit__mojolearn_preprocessing() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_preprocessing")
        m.def_function[standard_fit_binding]("standard_fit")
        m.def_function[standard_transform_binding]("standard_transform")
        m.def_function[fit_binding]("minmax_fit")
        m.def_function[transform_binding]("minmax_transform")
        m.def_function[numeric_mode_binding]("preprocessing_numeric_mode")
        m.def_function[vendor_binding]("preprocessing_vendor")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_preprocessing: ",e))

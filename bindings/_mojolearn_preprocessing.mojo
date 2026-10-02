# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Borrowed Float32 arrays; GPU arithmetic; no context or pointer retained."""
# DEVIATION 2486: shared byte-preserving host copies.
from bindings.hostptr import f32_ptr, read_f32, copy_f32
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE
from preprocessing.estimator import validate_dimensions, minmax_fit_host, minmax_transform_host_into, validate_standard, standard_fit_host, standard_transform_host_into, minmax_fit_direct, standard_fit_direct, minmax_transform_direct
from preprocessing.minmax import PREP_FAST_MINMAX


def ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("preprocessing: null Float32 pointer")
    return f32_ptr(addr)


def load(addr: Int, n: Int) raises -> List[Float32]:
    _ = ptr(addr)  # Preserve this surface's null-pointer refusal.
    return read_f32(addr, max(0, n))


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
        copy_f32(result.unsafe_ptr(), output, 5*d)
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
        minmax_transform_host_into(x,scale,offset,output,n,d,inverse,clip,lower,upper)
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
        copy_f32(result.unsafe_ptr(), output, 3*d)
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
        standard_transform_host_into(x,mean,scale,output,n,d,inverse,with_mean,with_std)
    return PythonObject(n*d)


def fit_direct_binding(x_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    """minmax_fit from the caller's own buffer (lane gap-prep2): 1 when the
    five rows were written, 0 when X holds a NaN or an infinity (nothing
    written; the caller takes its NaN route). The same kernels and words."""
    if len(params) != 4:
        raise Error("minmax_fit_direct: requires 4 parameters")
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    var lower = Float32(Float64(py=params[2]))
    var upper = Float32(Float64(py=params[3]))
    validate_dimensions(n,d,lower,upper)
    var x = ptr(Int(py=x_addr))
    var output = ptr(Int(py=out_addr))
    var ok = 0
    with GILReleased(Python()):
        ok = minmax_fit_direct(x,n,d,lower,upper,output)
    return PythonObject(ok)


def standard_fit_direct_binding(x_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    """standard_fit from the caller's own buffer (lane gap-prep2): 1 when the
    three rows were written, 0 when X holds a NaN or an infinity."""
    if len(params) != 4:
        raise Error("standard_fit_direct: requires 4 parameters")
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    var with_mean = Int(py=params[2])
    var with_std = Int(py=params[3])
    validate_standard(n,d,with_mean,with_std)
    var x = ptr(Int(py=x_addr))
    var output = ptr(Int(py=out_addr))
    var ok = 0
    with GILReleased(Python()):
        ok = standard_fit_direct(x,n,d,with_mean,with_std,output)
    return PythonObject(ok)


def transform_direct_binding(
    x_addr: PythonObject, scale_addr: PythonObject, min_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """minmax_transform from the caller's own buffers (lane apple-fast-prep,
    registered only under PREP_FAST_MINMAX): 1 when the n*d words were
    written, 0 when the output holds a nonfinite word (nothing written; the
    caller tells an overflow from its NaN route). The same kernel and words."""
    if len(params) != 6:
        raise Error("minmax_transform_direct: requires 6 parameters")
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    var inverse = Int(py=params[2])
    var clip = Int(py=params[3])
    var lower = Float32(Float64(py=params[4]))
    var upper = Float32(Float64(py=params[5]))
    validate_dimensions(n,d,lower,upper)
    var x = ptr(Int(py=x_addr))
    var scale = ptr(Int(py=scale_addr))
    var offset = ptr(Int(py=min_addr))
    var output = ptr(Int(py=out_addr))
    var ok = 0
    with GILReleased(Python()):
        ok = minmax_transform_direct(x,scale,offset,output,n,d,inverse,clip,lower,upper)
    return PythonObject(ok)


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
        m.def_function[fit_direct_binding]("minmax_fit_direct")
        m.def_function[standard_fit_direct_binding]("standard_fit_direct")
        comptime if PREP_FAST_MINMAX:
            m.def_function[transform_direct_binding]("minmax_transform_direct")
        m.def_function[numeric_mode_binding]("preprocessing_numeric_mode")
        m.def_function[vendor_binding]("preprocessing_vendor")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_preprocessing: ",e))

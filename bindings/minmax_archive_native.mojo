# SPDX-License-Identifier: Apache-2.0
# Existing MinMaxScaler NPZ schema, decoded and published through native slots.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_fields, py_get, py_setitem, py_contains
from bindings.classical_arrays import classical_address, classical_typed, classical_i64, classical_f64, classical_is_array, classical_c_order
from bindings.classical_pandas import classical_admit_value
from bindings.classical_archive import classical_archive_write
from bindings.classical_archive_read import classical_archive_read, _archive_string, _archive_exact
from bindings.minmax_metadata import _flag, minmax_range_pair

alias F32P = MutPointer[Float32, MutUntrackedOrigin]
alias F64P = MutPointer[Float64, MutUntrackedOrigin]
alias I64P = MutPointer[Int64, MutUntrackedOrigin]


def minmax_archive_save(model: PythonObject, path: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if py_get(fields,"numeric_mode_") is None or py_get(fields,"scale_") is None:
        raise Error("RuntimeError: this estimator is not fitted yet")
    var b = Python.import_module("builtins")
    if py_type(fields["n_features_in_"]) is not b.int or py_type(fields["n_samples_seen_"]) is not b.int:
        raise Error("TypeError: MinMaxScaler dimensions and counts must remain native integers")
    var d = Int(py=fields["n_features_in_"])
    var meta = classical_typed(Python.tuple(3),"<i8",context["output_context"])
    var mp = I64P(unsafe_from_address=Int(py=classical_address(meta)))
    mp[0] = Int64(d)
    mp[1] = Int64(py=fields["n_samples_seen_"])
    mp[2] = Int64(_flag(fields["clip_"],"clip_",context))
    var endpoints = minmax_range_pair(fields["feature_range_"],context)
    var bounds = classical_typed(Python.tuple(2),"<f8",context["output_context"])
    var bp = F64P(unsafe_from_address=Int(py=classical_address(bounds)))
    bp[0] = Float64(py=endpoints[0])
    bp[1] = Float64(py=endpoints[1])
    var arrays = Python.dict(format="mojolearn-scaler-1",estimator=py_type(model).__name__,numeric_mode=fields["numeric_mode_"],meta=meta,feature_range=bounds)
    var names = Python.tuple("data_min","data_max","data_range","scale","min")
    for k in range(5):  # small-loop(stats: the five saved extrema statistic fields)
        var value = fields[names[k]+"_"]
        classical_admit_value(value,context)
        if not classical_is_array(value,context) or String(value.dtype) != "<f4" or Int(py=value.ndim) != 1 or Int(py=value.size) != d or not classical_c_order(value,context):
            raise Error("MinMaxScaler saved statistics must remain contiguous Float32 feature vectors")
        py_setitem(arrays,names[k],value)
    return classical_archive_write(path,arrays,context)


def minmax_archive_build(cls: PythonObject, arrays: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    if String(_archive_string(arrays,"format",context)) != "mojolearn-scaler-1":
        raise Error("model archive format does not match mojolearn-scaler-1")
    if py_type(cls) is not b.type:
        raise Error("TypeError: native scaler archives require an ordinary native class layout")
    var saved = _archive_string(arrays,"estimator",context)
    var mro = py_call(b.type.__dict__["__mro__"].__get__,cls)
    var known = False
    var dictionary = False
    for k in range(len(mro)):  # driver-loop: native saved estimator ancestry names
        var class_fields = py_call(b.type.__dict__["__dict__"].__get__,mro[k])
        if py_contains(class_fields,"__del__"):
            raise Error("TypeError: Python finalizer hooks cannot execute while loading a native scaler")
        if not dictionary and py_contains(class_fields,"__dict__"):
            if py_type(class_fields["__dict__"]) is not py_type(b.memoryview.obj):
                raise Error("TypeError: native scaler archives require a native instance dictionary")
            dictionary = True
        known = known or Bool(py=mro[k].__name__ == saved)
    if not dictionary:
        raise Error("TypeError: native scaler archives require an instance dictionary")
    if not known:
        raise Error("model estimator does not match loading class")
    var meta = _archive_exact(arrays,"meta","<i8",context)
    if Int(py=meta.size) != 3 or Int(py=meta.ndim) != 1:
        raise Error("MinMaxScaler meta must hold three fields")
    var bounds = _archive_exact(arrays,"feature_range","<f8",context)
    if Int(py=bounds.size) != 2 or Int(py=bounds.ndim) != 1:
        raise Error("MinMaxScaler feature_range must hold two values")
    var mode = _archive_string(arrays,"numeric_mode",context)
    if String(mode) != "identical" and String(mode) != "fast":
        raise Error("invalid saved numeric_mode")
    var output_context = py_get(context,"output_context",context)
    var d = classical_i64(meta,0)
    var lower = classical_f64(bounds,0)
    var upper = classical_f64(bounds,1)
    var clip = classical_i64(meta,2) != 0
    var model = py_call(b.object.__new__,cls)
    var fields = py_fields(model)
    py_setitem(fields,"copy",True)
    py_setitem(fields,"numeric_mode",PythonObject(None))
    py_setitem(fields,"clip",clip)
    py_setitem(fields,"feature_range",Python.tuple(lower,upper))
    var names = Python.tuple("data_min","data_max","data_range","scale","min")
    for k in range(5):  # small-loop(stats: exact saved feature vectors)
        var stored = _archive_exact(arrays,String(names[k]),"<f4",context)
        if Int(py=stored.ndim) != 1 or Int(py=stored.size) != d:
            raise Error("MinMaxScaler saved statistic does not match n_features_in_")
        var value = classical_typed(Python.tuple(d),"<f4",output_context)
        memcpy(dest=F32P(unsafe_from_address=Int(py=classical_address(value))),src=F32P(unsafe_from_address=Int(py=classical_address(stored))),count=d)
        py_setitem(fields,names[k]+"_",value)
        _ = stored^
    py_setitem(fields,"n_features_in_",d)
    py_setitem(fields,"n_samples_seen_",classical_i64(meta,1))
    py_setitem(fields,"numeric_mode_",mode)
    py_setitem(fields,"feature_range_",Python.tuple(lower,upper))
    py_setitem(fields,"clip_",clip)
    _ = meta^
    _ = bounds^
    return model


def minmax_archive_load(cls: PythonObject, path: PythonObject, context: PythonObject) raises -> PythonObject:
    return minmax_archive_build(cls,classical_archive_read(path,context),context)

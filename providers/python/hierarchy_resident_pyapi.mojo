# SPDX-License-Identifier: Apache-2.0
# Hierarchy-only reference ownership. Exact resident records and extension
# entries are admitted before access; no module getattr/property is invoked.
# Numerical calls below still use the existing CPython extension ABI. A native
# descriptor for estimators' labeled-reference entries is required to close that
# remaining dispatch boundary; these calls are not claimed Python-free.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.native_loaded_module import native_loaded_module
from bindings.mode_metadata_native import _fields
from bindings.family_common import family_array
from bindings.classical_arrays import classical_address
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry


def _holder(fields: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var holder = py_get(fields, "_resident_holder")
    if holder is None:
        holder = Python.dict()
        py_setitem(fields, "_resident_holder", holder)
    if py_type(holder) is not b.dict:
        raise Error("TypeError: hierarchy resident holder must be an exact native dictionary")
    var names = py_call(b.list, holder.keys())
    for i in range(len(names)):  # driver-loop: exact resident dictionary keys before any lookup
        if py_type(names[i]) is not b.str:
            raise Error("TypeError: hierarchy resident keys must be exact native strings")
    var held = py_get(holder, "index")
    if held is None:
        return holder
    if py_type(held) is not b.tuple or len(held) != 3 or py_type(held[1]) is not b.int or py_type(held[2]) is not py_type(b):
        raise Error("TypeError: hierarchy resident state requires its native key, handle and module")
    var key = held[0]
    if py_type(key) is not b.tuple or len(key) != 6:
        raise Error("TypeError: hierarchy resident cache key has the wrong schema")
    for i in range(3, 6):  # small-loop(reference count, feature count and metric)
        if py_type(key[i]) is not b.int:
            raise Error("TypeError: hierarchy resident dimensions must be native integers")
    return holder


def hierarchy_release(fields: PythonObject) raises:
    var holder = _holder(fields)
    var held = py_get(holder, "index")
    if held is None:
        return
    py_setitem(holder, "index", PythonObject(None))
    var release = neural_descriptor_entry(held[2], "labeled_reference_release", False)
    if release is not None:
        try:
            _ = py_call(release, held[1])
        except:
            pass  # existing release contract: handles already released are harmless


def hierarchy_fresh_holder(fields: PythonObject) raises:
    hierarchy_release(fields)
    py_setitem(fields, "_resident_holder", Python.dict())


def _handle(fields: PythonObject, binding: PythonObject, refs: PythonObject,
            labels: PythonObject, refs_address: Int, labels_address: Int, n: Int, d: Int) raises -> Int:
    var holder = _holder(fields)
    var prepare = neural_descriptor_entry(binding, "labeled_reference_prepare", False)
    var predict = neural_descriptor_entry(binding, "labeled_reference_predict_resident", False)
    if prepare is None or predict is None:
        return -1
    var held = py_get(holder, "index")
    if held is not None and held[2] is binding:
        var key = held[0]
        if key[0] is refs and key[1] is labels and key[2] is labels and Int(py=key[3]) == n and Int(py=key[4]) == d and Int(py=key[5]) == 0:
            return Int(py=held[1])
    hierarchy_release(fields)
    var handle = py_call(prepare, Python.list(refs_address, labels_address, labels_address), Python.list(n, d, 0))
    if py_type(handle) is not py_type(PythonObject(0)):
        raise Error("TypeError: native hierarchy resident handle must be an integer")
    var key = Python.tuple(refs, labels, labels, n, d, 0)
    py_setitem(holder, "index", Python.tuple(key, handle, binding))
    return Int(py=handle)


def hierarchy_mint(fields: PythonObject, binding: PythonObject, refs: PythonObject,
                   labels: PythonObject, refs_address: Int, labels_address: Int, n: Int, d: Int) raises:
    _ = _holder(fields)
    try:
        _ = _handle(fields, binding, refs, labels, refs_address, labels_address, n, d)
    except:
        hierarchy_release(fields)


def hierarchy_reference_labels(fields: PythonObject, binding: PythonObject, refs: PythonObject,
                               labels: PythonObject, refs_address: Int, labels_address: Int,
                               query_address: Int, n: Int, nq: Int, d: Int,
                               context: PythonObject) raises -> PythonObject:
    var handle = _handle(fields, binding, refs, labels, refs_address, labels_address, n, d)
    var result = family_array(Python.tuple(nq), "<i4", context)
    var output = classical_address(result)
    if handle < 0:
        var chosen = family_array(Python.tuple(nq), "<i4", context)
        _ = py_call(neural_descriptor_entry(binding, "labeled_reference_predict"),
                    Python.list(refs_address, labels_address, labels_address, query_address, output, classical_address(chosen)),
                    Python.list(n, nq, d, 0, 0.0, 0))
        _ = chosen^
    else:
        _ = py_call(neural_descriptor_entry(binding, "labeled_reference_predict_resident"), handle,
                    Python.list(query_address, output), Python.list(nq, d, 0, 0.0, 0))
    return result

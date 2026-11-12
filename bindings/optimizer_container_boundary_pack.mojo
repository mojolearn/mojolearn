# SPDX-License-Identifier: Apache-2.0
"""Optimizer container UNPACK/PACK; native vectors pin admitted Python owners."""
# The PACK half of the former `bindings/optimizer_container_boundary.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/optimizer_container_boundary_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.py_cpython_table import cpython_table, _PyBuffer_Release
from bindings.optimizer_admission_door import opt_value
from bindings.py_balanced import py_type



def optimizer_sequence_items(value: PythonObject, context: PythonObject) raises -> List[PythonObject]:
    var items = List[PythonObject]()
    var cls = py_type(value)
    var table = cpython_table()
    ref cpy = Python().cpython()
    if cls is context["container_types"][0] or cls is context["container_types"][1]:
        # Exact native sequences: no iterator object, Python list or hook.
        var is_list = cls is context["container_types"][0]
        for i in range(len(value)):  # driver-loop: over external parameter/microbatch owners
            var pointer = table[].PyList_GetItem(value._obj_ptr, i) if is_list else table[].PyTuple_GetItem(value._obj_ptr, i)
            if not pointer:
                raise cpy.unsafe_get_error()
            items.append(PythonObject(from_borrowed=pointer))
        return items^
    # Preserve other admitted builtin iterable behavior (e.g. dictionary
    # keys and strings) without ever admitting a user __iter__ callback.
    opt_value(value, context)
    var pointer = table[].PyObject_GetIter(value._obj_ptr)
    if not pointer:
        raise cpy.unsafe_get_error()
    var iterator = PythonObject(from_owned=pointer)
    while True:  # driver-loop: over an exact admitted external builtin container
        var item = table[].PyIter_Next(iterator._obj_ptr)
        if not item:
            if table[].PyErr_Occurred():
                raise cpy.unsafe_get_error()
            break
        items.append(PythonObject(from_owned=item))
    return items^


def optimizer_pack_list(items: List[PythonObject]) raises -> PythonObject:
    ref cpy = Python().cpython()
    var table = cpython_table()
    var pointer = table[].PyList_New(len(items))
    if not pointer:
        raise cpy.unsafe_get_error()
    var result = PythonObject(from_owned=pointer)
    for i in range(len(items)):  # driver-loop: publish external result/parameter owners
        if table[].PyList_SetItem(result._obj_ptr, i, cpy.Py_NewRef(items[i]._obj_ptr)) != 0:
            raise cpy.unsafe_get_error()
    return result

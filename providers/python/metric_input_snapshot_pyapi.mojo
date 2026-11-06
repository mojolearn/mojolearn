# SPDX-License-Identifier: Apache-2.0
# Private operation-owned metric inputs. Admission happens at the public door;
# this snapshot preserves original container/array dtype inference and aliases.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr
from bindings.native_buffer_owner import native_storage_allocate
from bindings.py_attr import py_attr
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_arrays import classical_is_array
from bindings.linalg_ops_buffers import LinAdmission, lin_admitted_c
from bindings.native_owner_lifetime import native_release_admit
from providers.python.resample_rows_pyapi import _row_dtype_admit


def _metric_flat_leaves(value: PythonObject, context: PythonObject) raises -> Bool:
    """True when no element of the list/tuple `value` is a container the
    snapshot copies (a mojolearn Array, array.array, ndarray, list, tuple or
    bytearray): the exact types are read once with `set(map(type, value))`
    and each distinct type is tested, not each element."""
    var b = native_loaded_module("builtins")
    var types = py_call(b.list, py_call(b.set, py_call(b.map, b.type, value)))
    var ndarray = py_get(context, "ndarray")
    var array_class = context["array_class"]
    var native_array = context["array_type"]
    for i in range(len(types)):  # driver-loop: over the distinct exact element types (a handful)
        var cls = types[i]
        if (cls is array_class or cls is native_array or cls is ndarray or cls is b.list
                or cls is b.tuple or cls is b.bytearray):
            return False
    return True


def _metric_array_alias(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """A private Array sharing `value`'s storage: every slot (storage owners,
    address and metadata) is read once through the import-pinned native slot
    descriptors and written to a fresh Array of the connected class. The GIL
    is held throughout, so the slots form one consistent capture."""
    var cls = context["array_class"]
    var fields = context["array_fields"]
    var array_alias = native_storage_allocate(cls)
    var names = py_call(native_loaded_module("builtins").list, fields)
    for i in range(len(names)):  # driver-loop: over the fixed Array slot names
        var name = names[i]
        py_setattr(array_alias, String(py=name), py_call(fields[name], value, cls))
    return array_alias


def metric_input_snapshot(value: PythonObject, memo: PythonObject,
                          context: PythonObject, depth: Int = 0) raises -> PythonObject:
    if depth > 64:
        raise Error("TypeError: metric input nesting exceeds the supported depth")
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    var array = classical_is_array(value, context)
    var ndarray = py_get(context, "ndarray")
    var native_array = cls is context["array_type"]
    if not array and not native_array and cls is not ndarray and cls is not b.list and cls is not b.tuple and cls is not b.bytearray:
        # Exact scalar leaves retain their dtype inside containers. Pandas and
        # Torch retain their existing native materialization semantics.
        return value
    var key = py_call(b.id, value)
    var known = py_get(memo, key)
    if known is not None:
        return known
    var result = PythonObject(None)
    if array:
        var input = LinAdmission(value, context)
        if input.view.c_contig:
            # Metrics only read their inputs. A private Array that shares the
            # admitted storage owners freezes the metadata (a caller replacing
            # its slots cannot change what later readers see, and the owners
            # keep the address alive); the data itself is not copied.
            _ = input^
            result = _metric_array_alias(value, context)
        else:
            var dtype = input.view.dtype
            # Copy the same dtype, with the original pin/slot owners retained until
            # copying finishes. Later readers see this private Array's metadata.
            var output = lin_admitted_c(input^, dtype, "metric input", context, True)
            result = output.keep
            _ = output^
    elif native_array:
        # Exact array.array construction preserves the native typecode and
        # copies its C storage without caller iteration/conversion hooks.
        result = py_call(native_loaded_module("array").array, value.typecode, value)
    elif cls is b.bytearray:
        result = py_call(b.bytearray, value)
    elif cls is ndarray:
        # Exact native ndarray.view captures the caller's metadata once. C
        # ndarray.copy only copies storage/references in the same dtype; no
        # NumPy computation, cast, __array__, or user __copy__ hook runs.
        var view = py_call(py_attr(ndarray, "view"), value)
        var dtype = view.dtype
        _row_dtype_admit(dtype, context)
        var flags = view.flags
        if (not Bool(py=dtype.hasobject)
                and (Bool(py=flags.c_contiguous) or Bool(py=flags.f_contiguous))):
            # Metrics only read their inputs (scikit-learn takes no snapshot).
            # The private view fixes shape/strides/dtype once and holds the
            # base storage alive; a concurrent writer can change values but
            # never the memory the kernel reads, so no data copy is needed.
            result = view
        else:
            result = py_call(py_attr(ndarray, "copy"), view, order="K")
            if Bool(py=dtype.hasobject):
                # A caller may replace its object cells while a kernel runs. The
                # private snapshot must never become the final callback owner.
                native_release_admit(result, context)
        _ = view^
    elif _metric_flat_leaves(value, context):
        # Every element is a leaf, which the walk below returns unchanged and
        # never memoizes: one C-level copy of the container is the same
        # snapshot, without a recursive call per element.
        result = py_call(b.tuple, value) if cls is b.tuple else py_call(b.list, value)
    else:
        var items = Python.list()
        for i in range(len(value)):  # driver-loop: stable nested native metric containers, preserving scalar leaf dtypes
            py_call(items.append, metric_input_snapshot(value[i], memo, context, depth + 1))
        result = py_call(b.tuple, items) if cls is b.tuple else items
    py_setitem(memo, key, result)
    return result

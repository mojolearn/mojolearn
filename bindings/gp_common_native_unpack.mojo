# SPDX-License-Identifier: Apache-2.0
# The shared pieces of the Gaussian process owners (bindings/gp_regressor_native.mojo,
# bindings/gp_classifier_native.mojo): the binding's compile-time mode cross-check
# (`_extension`), its typed descriptor capability word, `numeric_mode_used`, and the
# kernel objects read natively from their instance fields: the postfix spec
# `_kernel_arrays` packed (bindings/archive_schema_native.mojo reads the same node
# layouts for `save`), the free hyperparameters `_hyper` listed, and the copy
# `_replace` / `_with_free_values` built. No method of a kernel class runs.
from std.python import Python, PythonObject
from core.gp_native_storage import GPFreeParameters, GPKernelStorage
from bindings.gp_kernel_storage_boundary import GPPackedKernel
from bindings.admitted_family import admit_family, admit_family_owned
from bindings.native_loaded_module import native_loaded_module
from bindings.mode_metadata_native import _plan
from bindings.gp_native_boundary import GPConnection
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from std.memory import unsafe_memcpy
from bindings.family_common import family_repr, family_is_str, family_call, family_array, FAMILY_F32P
from bindings.classical_arrays_door import classical_address
from bindings.archive_schema_native_door import _schema_kernel_layouts
from bindings.gp_kernel_public_native import gp_kernel_native_nodes




def gp_caps(binding: PythonObject) raises -> Int:
    var connection = GPConnection(binding)
    return connection.capabilities


def gp_kernel_arrays(kernel: PythonObject, context: PythonObject) raises -> PythonObject:
    """UNPACK the admitted postfix graph into one native storage owner."""
    var table = context["family_gp_kernel_nodes"]
    _schema_kernel_layouts(table)
    var nodes = gp_kernel_native_nodes(kernel, table)
    var storage = GPKernelStorage()
    # Preserve the former separate Array conversions' refusal order: every
    # integer kind first, then parameters, lengths, and length-scale words.
    try:
        for i in range(len(nodes)):  # driver-loop: one admitted integer kind per postfix node
            var kind = Int(py=nodes[i].kind)
            if kind < -2147483648:
                raise Error("signed integer is less than minimum")
            if kind > 2147483647:
                raise Error("signed integer is greater than maximum")
            storage.kinds.append(Int32(kind))
    except error:
        raise Error("TypeError: mojolearn: cannot build a <i4 Array from these values: " + String(error))
    for i in range(len(nodes)):  # driver-loop: the admitted native parameter of each node
        storage.parameters.append(Float32(nodes[i].parameter))
    for i in range(len(nodes)):  # driver-loop: length metadata before the scale table
        var count = len(nodes[i].scales)
        if count > 2147483647:
            raise Error("TypeError: mojolearn: cannot build a <i4 Array from these values: signed integer is greater than maximum")
        storage.lengths.append(Int32(count))
    for i in range(len(nodes)):  # driver-loop: scales of each postfix node, in the original order
        for j in range(len(nodes[i].scales)):  # driver-loop: one admitted length-scale parameter
            storage.scales.append(Float32(nodes[i].scales[j]))
    storage.finish()
    return PythonObject(alloc=GPPackedKernel(storage^))


def gp_kernel_arrays_transport(kernel: PythonObject, context: PythonObject) raises -> PythonObject:
    """PACK real owning Arrays only for cross-interpreter worker transport."""
    var packed = gp_kernel_arrays(kernel, context)
    var owner = packed.downcast_value_ptr[GPPackedKernel]()
    var view = owner[].storage.view()
    var kinds = family_array(Python.tuple(view.nodes), "<i4", context)
    var parameters = family_array(Python.tuple(view.nodes), "<f4", context)
    var lengths = family_array(Python.tuple(view.nodes), "<i4", context)
    var scales = family_array(Python.tuple(max(1, view.scale_count)), "<f4", context)
    unsafe_memcpy(dest=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(kinds))),
           src=Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=view.kinds), count=view.nodes)
    unsafe_memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(parameters))),
           src=Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=view.parameters), count=view.nodes)
    unsafe_memcpy(dest=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(lengths))),
           src=Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=view.lengths), count=view.nodes)
    unsafe_memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(scales))),
           src=Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=view.scales), count=max(1, view.scale_count))
    _ = packed
    return Python.tuple(kinds, parameters, lengths, scales, PythonObject(view.scale_count))


def _gp_kernel_class(kernel: PythonObject, table: PythonObject) raises -> PythonObject:
    """The class of `kernel`'s ancestry that `table` connects (the walk
    archive_schema_native's layout lookup makes)."""
    var b = native_loaded_module("builtins")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(kernel))
    for i in range(len(mro)):  # driver-loop: over the kernel class ancestry metadata
        if py_contains(table, mro[i]):
            return mro[i]
        if py_contains(py_call(b.type.__dict__["__dict__"].__get__, mro[i]), "_nodes"):
            break
    raise Error("TypeError: mojolearn: a " + String(py_type(kernel).__name__) + " kernel has no saved node layout")


def _gp_fixed(bounds: PythonObject) raises -> Bool:
    """`bounds == "fixed"` for the stored bounds (the str or a list of pairs)."""
    return family_is_str(bounds) and String(bounds) == "fixed"


def gp_free(kernel: PythonObject, mut result: GPFreeParameters, context: PythonObject) raises:
    """Unpack admitted kernel hyperparameters in their original postfix order."""
    var nodes_table = context["family_gp_kernel_nodes"]
    var cls = _gp_kernel_class(kernel, nodes_table)
    var layout = nodes_table[cls]
    var fields = py_fields(kernel)
    var how = String(layout[0])
    if how == "pair":
        gp_free(fields[layout[1]], result, context)
        gp_free(fields[layout[2]], result, context)
        result.flags.append(Int32(0))
    elif how == "nodes":
        var saved = fields[layout[1]]
        for _ in range(len(saved)):  # driver-loop: saved kernel nodes are fixed
            result.flags.append(Int32(0))
    elif how == "leaf":
        var pairs = fields[context["family_gp_kernel_bounds"][cls]]
        if _gp_fixed(pairs):
            result.flags.append(Int32(0))
            return
        result.flags.append(Int32(1))
        if layout[3] is not None:
            var scales = fields[layout[3]]
            for i in range(len(scales)):  # driver-loop: one free length scale per kernel parameter
                result.values.append(Float64(py=scales[i]))
        else:
            result.values.append(Float64(py=fields[layout[2]]))
        for i in range(len(pairs)):  # driver-loop: bound pair for each free kernel parameter
            result.bounds.append(Float32(Float64(py=pairs[i][0])))
            result.bounds.append(Float32(Float64(py=pairs[i][1])))
    else:
        raise Error("RuntimeError: unknown kernel node layout " + how)


def gp_n_dims(kernel: PythonObject, context: PythonObject) raises -> Int:
    # Count metadata without narrowing a bound/value that the caller does not
    # otherwise consume yet. This preserves the former count-only refusal order.
    var table = context["family_gp_kernel_nodes"]
    var cls = _gp_kernel_class(kernel, table)
    var layout = table[cls]
    var fields = py_fields(kernel)
    var how = String(layout[0])
    if how == "pair":
        return gp_n_dims(fields[layout[1]], context) + gp_n_dims(fields[layout[2]], context)
    if how == "nodes":
        return 0
    if how == "leaf":
        if _gp_fixed(fields[context["family_gp_kernel_bounds"][cls]]):
            return 0
        return len(fields[layout[3]]) if layout[3] is not None else 1
    raise Error("RuntimeError: unknown kernel node layout " + how)


def gp_fit_owned(source: PythonObject, caller: PythonObject, context: PythonObject) raises -> PythonObject:
    """`admit_family_owned(source)`, except that the caller's own Array is never
    kept as `X_train_`: it is copied once (lane ne1-a-gp-svm). The resident
    predict freezes `X_train_` read-only (bindings/fitted_resident_door.mojo),
    which must never reach the caller's array; `copy_X_train=False` is
    refused, so `X_train_` is a copy in every admitted fit."""
    if source is caller:
        var b = native_loaded_module("builtins")
        var result = family_array(py_call(b.tuple, source.shape), "<f4", context)
        unsafe_memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(result))),
               src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(source))), count=Int(py=source.size))
        _ = result  # keep the Python object alive through the native call above
        return result
    return admit_family_owned(source, context)

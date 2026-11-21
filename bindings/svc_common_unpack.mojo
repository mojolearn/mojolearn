# SPDX-License-Identifier: Apache-2.0
# UNPACK pieces shared by the SVM family doors (SVC, SVR, LinearSVC,
# LinearSVR, OneClassSVM): the instance-field and scalar reads, the base
# binding's host-helper lookup, and the label shape / flatten / class-order
# reads the Python methods of python/mojolearn/_svm_impl.py, svm.py and
# _expansion_neighbors.py used, with their refusals in the same wording and
# order. The label ORDER RULE (`_labels.sorted_classes`) is the builtins'
# `sorted` over the first-seen labels, as the Python label input step ran
# it: the classes and codes are the ones `encode_labels` returned for a flat
# label list. PACK pieces: bindings/svc_common_pack.mojo; the two `score`
# metrics: bindings/svc_common_door.mojo.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_item
from bindings.classical_arrays_door import classical_address
from bindings.classical_pandas_door import _pd_class
from bindings.classical_labels_door import _pandas_values, labels_type_name
from bindings.classical_labels_door import _flatten_labels, _sorted_classes, labels_shape_of, labels_label_kind
from bindings.family_common_unpack import (
    FAMILY_F32P,
    FAMILY_F64P,
)
from bindings.family_common_pack import (
    family_call,
)
from bindings.svm_keys_unpack import svm_keys


# ---------------------------------------------------------------- small reads


def svc_builtins() raises -> PythonObject:
    return native_loaded_module("builtins")


def svc_field(fields: PythonObject, key: PythonObject, name: String) raises -> PythonObject:
    """`self.<name>` of an instance attribute (the instance dictionary, so no
    attribute hook runs); refused as the lookup refused it. `key` is the held
    interned name (bindings/svm_keys_unpack.mojo; lane bw2-svm: it built the
    str from `name` up to twice per call); `name` is the refusal text only."""
    var value = py_get(fields, key, PythonObject(None))
    if value is None and not py_contains(fields, key):
        raise Error("AttributeError: object has no attribute '" + name + "'")
    return value


def svc_attr(model: PythonObject, key: PythonObject) raises -> PythonObject:
    """`getattr(model, key)` for a class-level constant or a parameter a
    subclass may define on the class; `key` is a held interned name."""
    return family_call(svc_builtins().getattr, model, key)


def svc_has(fields: PythonObject, key: PythonObject) raises -> Bool:
    """`hasattr(self, key)` for an attribute fit sets on the instance; `key`
    is a held interned name."""
    return py_contains(fields, key)


def svc_name(model: PythonObject) raises -> String:
    """`type(self).__name__`, read through type's own descriptor."""
    return labels_type_name(py_type(model))


def svc_float(value: PythonObject) raises -> Float64:
    """`float(value)` with Python's conversion (and its refusals)."""
    return Float64(py=family_call(svc_builtins().float, value))


def svc_int(value: PythonObject) raises -> Int:
    """`int(value)` with Python's conversion (and its refusals)."""
    return Int(py=family_call(svc_builtins().int, value))


def svc_truthy(value: PythonObject) raises -> Bool:
    return Bool(py=family_call(svc_builtins().bool, value))


def svc_base(name: String, context: PythonObject) raises -> PythonObject:
    """The base binding's host helper `name` (`_buffer._native(name)`).

    Read from `_buffer._NATIVE`, the resolver's own once-per-process cache,
    with one dict lookup; only the first use of a name in a process calls the
    Python resolver (which fills that cache). Before (census py-census-30,
    2026-11-15) every svc_base call entered `_buffer._native`, a Python
    frame per decode / threshold / argmax / class-ratio step of every SVC
    fit and predict."""
    var key = PythonObject(name)
    var nk = svm_keys()
    var cached = py_get(py_item(context, nk[].k_family_svc_native_cache), key)
    if cached is not None:
        return cached
    return family_call(py_item(context, nk[].k_family_svc_native), key)


def svc_mode(fields: PythonObject) raises -> PythonObject:
    """`getattr(self, "numeric_mode", None)`."""
    var nk = svm_keys()
    return py_get(fields, nk[].k_numeric_mode, PythonObject(None))


# TOMBSTONE: `svc_binding` (`_backend.binding(name, mode)` called from the door: a Python call that selected the
# PROCESS set; no caller was left) deleted 2026-11-21 by lane dw-classical. Bindings come from the estimator's
# binding route (providers/python/estimator_binding_pyapi_door.mojo). Code recoverable at ea4e2e9a45.


def svc_zero_division(float_division: Bool) raises:
    """Raise Python's ZeroDivisionError with the message the Python division
    raised (`division by zero` for ints, `float division by zero`)."""
    var operator = Python.import_module("operator")
    if float_division:
        _ = family_call(operator.truediv, PythonObject(1.0), PythonObject(0.0))
    _ = family_call(operator.truediv, PythonObject(1), PythonObject(0))
    raise Error("RuntimeError: division by zero")


def svc_read_f64(array: PythonObject, index: Int) raises -> Float64:
    return FAMILY_F64P(unsafe_from_address=Int(py=classical_address(array)))[index]


def svc_read_f32(array: PythonObject, index: Int) raises -> Float64:
    return Float64(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(array)))[index])



# TOMBSTONE: svc_f32, svc_snapshot (the SVM family's copy of the family admission) deleted 2026-11-08 by lane a2-admit-family; code recoverable at f7d10d2b09.
# Restore: experiments/removed/A2_FAMILY_ADMISSION_HELPERS.patch. Doors call
# admit_family / admit_family_meta (bindings/admitted_family.mojo); the
# `input_copied_` flag is admit_family_copied.


# ---------------------------------------------------------------- shapes and labels


def svc_shape_of(obj: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_buffer.shape_of` from admitted native storage (an Array, a C buffer
    exporter, an exact pandas object or nested exact lists/tuples): never an
    arbitrary `.shape` property or `__len__` hook."""
    if py_type(obj) is _pd_class("pandas.core.series", "Series"):
        # A label Series may hold strings: read its block storage's shape
        # (the f32 frame reader would refuse non-numeric labels).
        var values = _pandas_values(obj)
        var nk = svm_keys()
        var ndarray = py_get(context, nk[].k_ndarray)
        if ndarray is not None and py_type(values) is ndarray:
            return py_call(svc_builtins().tuple, values.shape)
    var shape = labels_shape_of(obj, context)
    var dimensions = Python.list()
    for dimension in shape:  # driver-loop: publish captured native shape words
        _ = py_call(dimensions.append, dimension)
    return py_call(svc_builtins().tuple, dimensions)


def svc_flatten_labels(y: PythonObject, context: PythonObject) raises -> PythonObject:
    return _flatten_labels(y, context)


def svc_labels_1d(y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`linear_model._labels_1d`: (labels list or None, shape)."""
    var shape = svc_shape_of(y, context)
    if len(shape) != 1:
        return Python.tuple(PythonObject(None), shape)
    return Python.tuple(svc_flatten_labels(y, context), shape)


# TOMBSTONE: svc_encode_1d (the SVM's second label encoder) deleted 2026-11-08 by lane a2-admit-family; code recoverable at f7d10d2b09.
# Restore: experiments/removed/A2_FAMILY_ADMISSION_HELPERS.patch. Its callers
# test the 1-D shape (svc_shape_of) and call classical_encode_labels, which
# now carries the no-label-helpers branch it had.


def svc_sorted_classes(labels: PythonObject, context: PythonObject) raises -> PythonObject:
    # Preserve the shared first-seen representative and sorted-class rule.
    var encoded = _sorted_classes(labels, context)
    return Python.tuple(encoded.classes, encoded.codes)


def svc_label_kind(classes: PythonObject, context: PythonObject) raises -> Int:
    var kind = labels_label_kind(classes, context)
    return 0 if kind == "int" else 1 if kind == "float" else 2

# SPDX-License-Identifier: Apache-2.0
# Classifier target admission shared by the public GP classifier and the public boosting classifiers:
# the label walk and the finished target a classifier's encoder reads. Moved unchanged out of
# providers/python/gp_public_lifecycle_pyapi_door.mojo (lane door-cut-1, 2026-11-28) so that a caller in
# another family does not import the GP lifecycle door, and with it the GP state, for these two functions.
# Nothing here reads a GP model: `gp_label_list` and `gp_keys` are the label snapshot and the key table
# the functions always used.
from bindings.admitted_finite_wording import admitted_finite_wording
from std.math import isfinite, floor
from std.python import PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.classical_pandas_door import classical_admit_value
from bindings.classical_labels_door import _flatten_labels, labels_shape_of
from bindings.label_integer_storage_unpack import labels_integer_storage
from bindings.label_input_sources_unpack import labels_storage_source, labels_list_integer_storage
from bindings.gp_label_snapshot_native_unpack import gp_label_list
from bindings.classical_warnings_unpack import classical_warn_column
from bindings.host_inference_scalars_door import host_inference_primitive_number
from bindings.sequential_admission_door import _ordinary_torch
from bindings.native_torch_admission_door import native_torch_numpy
from bindings.gp_keys_unpack import gp_keys


def _gp_label_admit(value: PythonObject, name: String, context: PythonObject, depth: Int = 0) raises:
    if depth > 64:
        raise Error("TypeError: GP target nesting exceeds the supported depth")
    var b = native_loaded_module("builtins")
    var kind = py_type(value)
    if kind is b.complex:
        raise Error("Complex data not supported: " + name + " got a complex y.")
    if kind is b.list or kind is b.tuple:
        for i in range(len(value)):  # driver-loop: passive label admission before encoder operations
            _gp_label_admit(value[i], name, context, depth + 1)
    else:
        classical_admit_value(value, context)


def _gp_classifier_target(source: PythonObject, rows: Int, name: String, context: PythonObject) raises -> PythonObject:
    if source is None:
        raise Error(name + " requires y to be passed, but the target y is None.")
    var b = native_loaded_module("builtins")
    var kind = py_type(source)
    # An exact flat list / tuple of exact ints first (lane p3-svm-gp-kernel,
    # row 80): every item is an exact int, which the per-label admission walk
    # below passes unchanged, so the walk is skipped (bools keep it).
    if (kind is b.list or kind is b.tuple) and labels_list_integer_storage(source, False):
        if len(source) != rows:
            raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(len(source)) + "]")
        return py_call(b.list, source)
    _gp_label_admit(source, name, context)
    if kind is b.str or kind is b.bytes:
        raise Error("mojolearn: y must be a sequence of labels")
    # A rank-1 integer ndarray or Array goes to the encoder unchanged:
    # gp_encode_labels reads it with one native prepare_targets call. The
    # snapshot below boxes, admits and converts every label as a Python object
    # (about 2.3 s per 20000-label fit); it cannot refuse an integer label and
    # yields the same Python-int classes and codes, so no bits change. Bool,
    # float, 2-D, list, torch and object targets keep the snapshot.
    # perf3-label-doors: a numpy-backed pandas Series takes the same door
    # through its stored exact ndarray (the labels its snapshot lists, in the
    # same order), which the encoder then reads natively.
    var stored = labels_storage_source(source, context)
    if stored is not None and labels_integer_storage(stored, context, False):
        var target_shape = labels_shape_of(stored, context)
        if len(target_shape) == 1:
            if target_shape[0] != rows:
                raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(target_shape[0]) + "]")
            return stored
    # An exact flat list / tuple of exact ints (bools keep the snapshot): the
    # snapshot admits every label and lists the same ints; one native list
    # copy (no per-label call) stands in for its flattened list.
    if labels_list_integer_storage(source, False):
        if len(source) != rows:
            raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(len(source)) + "]")
        return py_call(b.list, source)
    # A single native container snapshot retains label identities and integer
    # precision. All following shape/finite/integral checks use that snapshot;
    # the numerical owner sees the finished native list, never the user input.
    var values = source
    if _ordinary_torch(kind, context):
        values = gp_label_list(native_torch_numpy(source), context)
    elif kind is not b.list and kind is not b.tuple:
        values = gp_label_list(source, context)
    # A list / tuple source was admitted label by label above; only a new
    # snapshot is admitted here (one walk per label, not two).
    if not (values is source):
        _gp_label_admit(values, name, context)
    if py_type(values) is not b.list and py_type(values) is not b.tuple:
        raise Error("mojolearn: y must be a sequence of labels")
    if len(values) != rows:
        raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(len(values)) + "]")
    if len(values) > 0 and kind is not b.list and kind is not b.tuple and kind is not b.memoryview:
        var first = values[0]
        if (py_type(first) is b.list or py_type(first) is b.tuple) and len(first) == 1:
            classical_warn_column(context)
    var flat = _flatten_labels(values, context)
    var nk = gp_keys()
    for i in range(len(flat)):  # driver-loop: admitted scalar labels before native encoder/comparison
        var item = host_inference_primitive_number(flat[i], context)
        if py_type(item) is b.float:
            var number = Float64(py=item)
            if not isfinite(number):
                raise Error(admitted_finite_wording(name, "y"))
            if floor(number) != number:
                raise Error("Unknown label type: continuous. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values.")
        # The primitive number is for the checks above only: the encoder keeps
        # the first ORIGINAL label object of each class (gp_mixed_classes,
        # 48664c4c59); writing the converted number back replaced NumPy scalar
        # labels with new Python numbers (1ec92cef40; smoke 2026-11-15). A 0-d
        # ndarray label is not a scalar the encoder admits: it keeps the number.
        if py_type(flat[i]) is py_get(context, nk[].k_ndarray):
            py_setitem(flat, i, item)
    return flat

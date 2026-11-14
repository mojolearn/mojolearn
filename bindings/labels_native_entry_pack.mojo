# SPDX-License-Identifier: Apache-2.0
# The label entry the Python label and metric shells call (lane ne1-h-labels,
# plan NATIVE_ENTRY_POINTS_AFTER_0866 section H): one compiled entry,
# `labels_native_call(op, args, context)`, with an integer operation code, so
# a shell hands its label objects over once and never walks, converts or
# re-encodes them itself.
#
#   LABELS_OP_ENCODE           (y, kind)                  -> (classes, int32 codes)
#   LABELS_OP_UNION_ENCODE     (y_true, y_pred, caller)   -> (true classes, true codes,
#                                                             pred classes, pred codes,
#                                                             kind, sorted union)
#   LABELS_OP_SCORE_FROM_CODES (y, classes, codes, w, mode) -> accuracy (float)
#   LABELS_OP_DECODE           (classes, codes)           -> labels (codes: any integer form)
#   LABELS_OP_CLASS_WEIGHT_ROWS (class_weight, classes, int32 codes, x_trees entries)
#                                                         -> float32 row weights or None
#   LABELS_OP_FLATTEN          (y,)                       -> flat list of the label objects
#   LABELS_OP_SORTED           (labels,)                  -> (classes, int32 codes)
#   LABELS_OP_PREPARE_TARGETS  (labels, weights, n, flags) -> (classes, int32 codes) Arrays
#                                                             or (None, None)
#   LABELS_OP_ENCODE_NUMERIC   (y,)                       -> (classes, int32 codes) or None
#   LABELS_OP_UNIQUE_INVERSE   (y,)                       -> (classes, int32 codes)
#   LABELS_OP_ARGMAX_ROWS      (scores,)                  -> int64 codes
#   LABELS_OP_CLASS_INDEX      (classes, key)             -> labels
#   ADMISSION_OP_CHECK_X       (X, est, shape, complex, scan) -> None (or the refusal)
#   ADMISSION_OP_CHECK_Y       (y, est, flags, x_rows, y_rows) -> None (or the refusal)
#   (operations 7-13: bindings/labels_admission_unpack.mojo, lane py-admission:
#   the refusals and scans _labels and _sk_validate made in Python)
#
# `kind` of ENCODE is LABELS_KIND_ORDER_RULE (0): the classes under the ORDER
# RULE of python/mojolearn/_labels.py (Python label objects, sorted) and one
# dense int32 code per label; other kinds are refused. The sources read are
# the ones bindings/label_input_sources.mojo admits (exact ndarray, Array, flat
# list or tuple, a numpy-backed pandas Series, '<U' ndarrays) plus the ORDER
# RULE walk for everything else (bindings/classical_labels_door.mojo
# `classical_encode_labels`).
#
# The context is the classical shared admission context merged with the label
# context (python/mojolearn/_native_labels_bridge.py `labels_entry`).
# Numerics: none touched; the encoders, gathers and the COUNT program are the
# ones the classical owners already call.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_type
from bindings.classical_error_boundary_pack import classical_python_error
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_labels_door import (
    classical_encode_labels, classical_decode_labels, classical_union_encode, classical_score_from_codes,
    labels_class_weight_rows, ClassicalLabels, _flatten_labels, _sorted_classes)
from bindings.labels_admission_unpack import (
    LABELS_BARE_EXCEPTION, LABELS_KEY_ERROR, LABELS_BARE_EXCEPTION_KIND, LABELS_KEY_ERROR_KIND,
    labels_prepare_targets_op, labels_encode_numeric_op, labels_unique_inverse_op, labels_argmax_op,
    labels_class_index_op, labels_decode_any_op, admission_check_x_op, admission_check_y_op)

comptime LABELS_OP_ENCODE = 0
comptime LABELS_OP_UNION_ENCODE = 1
comptime LABELS_OP_SCORE_FROM_CODES = 2
comptime LABELS_OP_DECODE = 3
comptime LABELS_OP_CLASS_WEIGHT_ROWS = 4
comptime LABELS_OP_FLATTEN = 5
comptime LABELS_OP_SORTED = 6
comptime LABELS_OP_PREPARE_TARGETS = 7
comptime LABELS_OP_ENCODE_NUMERIC = 8
comptime LABELS_OP_UNIQUE_INVERSE = 9
comptime LABELS_OP_ARGMAX_ROWS = 10
comptime LABELS_OP_CLASS_INDEX = 11
comptime ADMISSION_OP_CHECK_X = 12
comptime ADMISSION_OP_CHECK_Y = 13

comptime LABELS_KIND_ORDER_RULE = 0


def _labels_operation(op: Int, args: PythonObject, context: PythonObject) raises -> PythonObject:
    if op == LABELS_OP_ENCODE:
        if Int(py=args[1]) != LABELS_KIND_ORDER_RULE:
            raise Error("TypeError: mojolearn: unknown label encoding kind")
        var encoded = classical_encode_labels(args[0], context)
        return Python.tuple(encoded.classes, encoded.codes)
    if op == LABELS_OP_UNION_ENCODE:
        return classical_union_encode(args[0], args[1], String(args[2]), context)
    if op == LABELS_OP_SCORE_FROM_CODES:
        return PythonObject(classical_score_from_codes(args[0], args[1], args[2], args[3], args[4], context))
    if op == LABELS_OP_DECODE:
        # any integer codes (lane py-admission: the shell's normalization)
        return labels_decode_any_op(args[0], args[1], context)
    if op == LABELS_OP_CLASS_WEIGHT_ROWS:
        # `_class_weights._class_weight_rows` on the x_trees binding: the
        # counts, the k weight checks and the per-row gather in one call
        # (the twin LogisticRegression's owner already calls).
        var codes = args[2]
        return labels_class_weight_rows(args[0], ClassicalLabels(args[1], codes, Int(py=codes.size)), args[3], context)
    if op == LABELS_OP_FLATTEN:
        # `_labels.flatten_labels`: the label input step's flatten (exact
        # containers read by pointer, depth first; lane p2-labels-lists-arrays).
        return _flatten_labels(args[0], context)
    if op == LABELS_OP_SORTED:
        # `_labels.sorted_classes`: the ORDER RULE over a list of label objects.
        var ordered = _sorted_classes(args[0], context)
        return Python.tuple(ordered.classes, ordered.codes)
    if op == LABELS_OP_PREPARE_TARGETS:
        return labels_prepare_targets_op(args[0], args[1], Int(py=args[2]), Int(py=args[3]), context)
    if op == LABELS_OP_ENCODE_NUMERIC:
        return labels_encode_numeric_op(args[0], context)
    if op == LABELS_OP_UNIQUE_INVERSE:
        return labels_unique_inverse_op(args[0], context)
    if op == LABELS_OP_ARGMAX_ROWS:
        return labels_argmax_op(args[0], context)
    if op == LABELS_OP_CLASS_INDEX:
        return labels_class_index_op(args[0], args[1], context)
    if op == ADMISSION_OP_CHECK_X:
        admission_check_x_op(args[0], String(args[1]), args[2], Bool(py=args[3]), Bool(py=args[4]), context)
        return PythonObject(None)
    if op == ADMISSION_OP_CHECK_Y:
        admission_check_y_op(args[0], String(args[1]), Int(py=args[2]), Int(py=args[3]), Int(py=args[4]), context)
        return PythonObject(None)
    raise Error("TypeError: mojolearn: unknown label operation")


def labels_native_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        if len(call) != 3:
            raise Error("TypeError: labels_native_call requires an operation code, arguments and context")
        var b = native_loaded_module("builtins")
        if py_type(call[0]) is not b.int or py_type(call[1]) is not b.tuple or py_type(call[2]) is not b.dict:
            raise Error("TypeError: labels_native_call requires an exact int code, argument tuple and context dictionary")
        var context = call[2]
        classical_connect_numpy(context)
        var result = _labels_operation(Int(py=call[0]), call[1], context)
        return result.steal_data()
    except error:
        # The exception types the Python label shells raised before lane
        # py-admission: a base-binding helper's own refusal crossed as a bare
        # Exception, an unknown dtype code as a KeyError. Local prefixes, so no
        # other door's wording changes type.
        if String(error).startswith(LABELS_BARE_EXCEPTION):
            return raise_python_exception(error_without_class(error, LABELS_BARE_EXCEPTION_KIND),
                                          ExceptionType("PyExc_Exception"))
        if String(error).startswith(LABELS_KEY_ERROR):
            return raise_python_exception(error_without_class(error, LABELS_KEY_ERROR_KIND),
                                          ExceptionType("PyExc_KeyError"))
        return classical_python_error(error^, args_ptr)

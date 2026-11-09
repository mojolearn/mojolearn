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
#   LABELS_OP_DECODE           (classes, codes)           -> labels
#   LABELS_OP_CLASS_WEIGHT_ROWS (class_weight, classes, int32 codes, x_trees entries)
#                                                         -> float32 row weights or None
#
# `kind` of ENCODE is LABELS_KIND_ORDER_RULE (0): the classes under the ORDER
# RULE of python/mojolearn/_labels.py (Python label objects, sorted) and one
# dense int32 code per label; other kinds are refused. The sources read are
# the ones bindings/label_input_sources.mojo admits (exact ndarray, Array, flat
# list or tuple, a numpy-backed pandas Series, '<U' ndarrays) plus the ORDER
# RULE walk for everything else (bindings/classical_labels.mojo
# `classical_encode_labels`).
#
# The context is the classical shared admission context merged with the label
# context (python/mojolearn/_native_labels_bridge.py `labels_entry`).
# Numerics: none touched; the encoders, gathers and the COUNT program are the
# ones the classical owners already call.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_type
from bindings.classical_error_boundary import classical_python_error
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_labels import (
    classical_encode_labels, classical_decode_labels, classical_union_encode, classical_score_from_codes,
    labels_class_weight_rows, ClassicalLabels)

comptime LABELS_OP_ENCODE = 0
comptime LABELS_OP_UNION_ENCODE = 1
comptime LABELS_OP_SCORE_FROM_CODES = 2
comptime LABELS_OP_DECODE = 3
comptime LABELS_OP_CLASS_WEIGHT_ROWS = 4

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
        return classical_decode_labels(args[0], args[1], context)
    if op == LABELS_OP_CLASS_WEIGHT_ROWS:
        # `_class_weights._class_weight_rows` on the x_trees binding: the
        # counts, the k weight checks and the per-row gather in one call
        # (the twin LogisticRegression's owner already calls).
        var codes = args[2]
        return labels_class_weight_rows(args[0], ClassicalLabels(args[1], codes, Int(py=codes.size)), args[3], context)
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
        return classical_python_error(error^, args_ptr)

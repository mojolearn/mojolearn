# SPDX-License-Identifier: Apache-2.0
# The label/tree operation context with concrete native scalar type tuples,
# built once per connected class context instead of on every public call.
#
# Several doors (LDA, the classify door, LogisticRegression, the fixed CV
# owner, the GBDT adapters) used to copy the whole connected context and
# build the real/integral type tuples on EVERY public call (fit, predict,
# transform, score, ...): a dict copy plus four Python calls per call, and
# per-call state written to the copy (connected NumPy types, warning
# registry, NotFittedError bridge) was thrown away and rebuilt next call.
# The derived dict is now cached on the connected context. It is rebuilt
# when the source context gains or loses keys (connect-time additions), so
# a stale view never survives a structural change; replaced values of
# existing keys are connection metadata that stays fixed for a connected
# class. Keys an operation memoizes on the derived dict (connected NumPy
# types, warning registry, tag plan, NotFittedError bridge) persist across
# calls, exactly as they do on doors that pass their connected context
# straight through.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem

comptime _KEY_LABELS = "_mojolearn_typed_context_labels"
comptime _KEY_TREES = "_mojolearn_typed_context_trees"
comptime _KEY_BOTH = "_mojolearn_typed_context_both"


def native_typed_context(context: PythonObject, labels: Bool, trees: Bool) raises -> PythonObject:
    """`context` plus `labels_real_type`/`labels_integral_type` (when
    `labels`) and `trees_real_type`/`trees_integral_type` (when `trees`):
    tuples of the concrete native int/float/bool and connected NumPy scalar
    classes, so isinstance tests never run numbers.Real/Integral ABC hooks."""
    var b = native_loaded_module("builtins")
    var key = PythonObject(_KEY_BOTH if (labels and trees) else (_KEY_LABELS if labels else _KEY_TREES))
    if py_type(context) is b.dict:
        var cached = py_get(context, key)
        if cached is not None and py_type(cached) is b.tuple and len(cached) == 3:
            if Int(py=cached[1]) == len(context):
                return cached[0]
    var result = py_call(b.dict, context)
    _ = py_call(result.pop, PythonObject(_KEY_LABELS), PythonObject(None))
    _ = py_call(result.pop, PythonObject(_KEY_TREES), PythonObject(None))
    _ = py_call(result.pop, PythonObject(_KEY_BOTH), PythonObject(None))
    var types = Python.list()
    _ = py_call(types.extend, Python.tuple(b.int, b.float, b.bool))
    _ = py_call(types.extend, context["scalar_types"])
    var real = py_call(b.tuple, types)
    var integral = context["integer_types"]
    if labels:
        py_setitem(result, "labels_real_type", real)
        py_setitem(result, "labels_integral_type", integral)
    if trees:
        py_setitem(result, "trees_real_type", real)
        py_setitem(result, "trees_integral_type", integral)
    if py_type(context) is b.dict:
        # Insert first so the recorded source size counts the cache key.
        py_setitem(context, key, Python.tuple(result, PythonObject(0), PythonObject(0)))
        py_setitem(context, key, Python.tuple(result, PythonObject(len(context)), PythonObject(len(result))))
    return result

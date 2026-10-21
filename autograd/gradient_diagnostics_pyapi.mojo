# SPDX-License-Identifier: Apache-2.0
# Python door for the gradient diagnostics (gradient_diagnostics.mojo): it
# converts the native report rows to the established dict rows, nothing else.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_setitem
from sequence.exec_trait import Exec
from autograd.gradient_diagnostics import ValueSummary, gradient_report


def _summary_dict(summary: ValueSummary) raises -> PythonObject:
    var out = Python.import_module("builtins").dict()
    py_setitem(out, "has_nan", summary.has_nan)
    py_setitem(out, "has_infinity", summary.has_infinity)
    py_setitem(out, "has_zero", summary.has_zero)
    py_setitem(out, "all_zero", summary.all_zero)
    py_setitem(out, "squared_l2_float32", summary.squared_l2_float32)
    return out


def gradient_report_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, handles: List[Int], values: Bool,
) raises -> PythonObject:
    var report = gradient_report[E, KIND](sid, handles, values)
    var b = Python.import_module("builtins")
    var rows = Python.list()
    for j in range(len(report)):  # driver-loop: over requested gradient diagnostics
        ref r = report[j]
        var row = b.dict()
        py_setitem(row, "handle", r.handle)
        py_setitem(row, "node", r.node)
        py_setitem(row, "operation", r.operation)
        py_setitem(row, "requires_grad", r.requires_grad)
        py_setitem(row, "inference_value", r.inference_value)
        py_setitem(row, "stale", r.stale)
        var dims = Python.list()
        for k in range(r.shape.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata
            _ = py_call(dims.append, r.shape.d[k])
        py_setitem(row, "shape", py_call(b.tuple, dims))
        var parents = Python.list()
        for k in range(len(r.parents)):  # driver-loop: over original tape parent references
            _ = py_call(parents.append, r.parents[k])
        py_setitem(row, "parents", py_call(b.tuple, parents))
        # Key order matches the established rows: value, gradient, reason, available.
        py_setitem(row, "value", _summary_dict(r.value) if r.has_value else PythonObject(None))
        py_setitem(row, "gradient", _summary_dict(r.gradient) if r.available else PythonObject(None))
        py_setitem(row, "reason", PythonObject(None) if r.available else PythonObject(r.reason))
        py_setitem(row, "available", r.available)
        _ = py_call(rows.append, row)
    return rows

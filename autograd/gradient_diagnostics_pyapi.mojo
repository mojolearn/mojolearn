# SPDX-License-Identifier: Apache-2.0
# Python door for the gradient diagnostics (gradient_diagnostics.mojo): it
# converts the native report rows to the established dict rows, nothing else.
from std.python import Python, PythonObject
from sequence.exec_trait import Exec
from autograd.gradient_diagnostics import ValueSummary, gradient_report


def _summary_dict(summary: ValueSummary) raises -> PythonObject:
    var out = Python.import_module("builtins").dict()
    out["has_nan"] = summary.has_nan
    out["has_infinity"] = summary.has_infinity
    out["has_zero"] = summary.has_zero
    out["all_zero"] = summary.all_zero
    out["squared_l2_float32"] = summary.squared_l2_float32
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
        row["handle"] = r.handle
        row["node"] = r.node
        row["operation"] = r.operation
        row["requires_grad"] = r.requires_grad
        row["inference_value"] = r.inference_value
        row["stale"] = r.stale
        var dims = Python.list()
        for k in range(r.shape.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata
            dims.append(r.shape.d[k])
        row["shape"] = b.tuple(dims)
        var parents = Python.list()
        for k in range(len(r.parents)):  # driver-loop: over original tape parent references
            parents.append(r.parents[k])
        row["parents"] = b.tuple(parents)
        # Key order matches the established rows: value, gradient, reason, available.
        row["value"] = _summary_dict(r.value) if r.has_value else PythonObject(None)
        row["gradient"] = _summary_dict(r.gradient) if r.available else PythonObject(None)
        row["reason"] = PythonObject(None) if r.available else PythonObject(r.reason)
        row["available"] = r.available
        rows.append(row)
    return rows

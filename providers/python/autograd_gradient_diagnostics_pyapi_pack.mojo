# SPDX-License-Identifier: Apache-2.0
# Python door for the gradient diagnostics (gradient_diagnostics.mojo): it
# converts the native report rows to the established dict rows, nothing else.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_setitem
from sequence.exec_trait import Exec
from autograd.gradient_diagnostics import ValueSummary, gradient_report
from bindings.autograd_keys_unpack import autograd_keys


def _summary_dict(summary: ValueSummary) raises -> PythonObject:
    var out = native_loaded_module("builtins").dict()
    var nk = autograd_keys()
    py_setitem(out, nk[].k_has_nan, summary.has_nan)
    py_setitem(out, nk[].k_has_infinity, summary.has_infinity)
    py_setitem(out, nk[].k_has_zero, summary.has_zero)
    py_setitem(out, nk[].k_all_zero, summary.all_zero)
    py_setitem(out, nk[].k_squared_l2_float32, summary.squared_l2_float32)
    return out


def gradient_report_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, handles: List[Int], values: Bool,
) raises -> PythonObject:
    var report = gradient_report[E, KIND](sid, handles, values)
    var b = native_loaded_module("builtins")
    var rows = Python.list()
    var nk = autograd_keys()
    for j in range(len(report)):  # driver-loop: over requested gradient diagnostics
        ref r = report[j]
        var row = b.dict()
        py_setitem(row, nk[].k_handle, r.handle)
        py_setitem(row, nk[].k_node, r.node)
        py_setitem(row, nk[].k_operation, r.operation)
        py_setitem(row, nk[].k_requires_grad, r.requires_grad)
        py_setitem(row, nk[].k_inference_value, r.inference_value)
        py_setitem(row, nk[].k_stale, r.stale)
        var dims = Python.list()
        for k in range(r.shape.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata
            _ = py_call(dims.append, r.shape.d[k])
        py_setitem(row, nk[].k_shape, py_call(b.tuple, dims))
        var parents = Python.list()
        for k in range(len(r.parents)):  # driver-loop: over original tape parent references
            _ = py_call(parents.append, r.parents[k])
        py_setitem(row, nk[].k_parents, py_call(b.tuple, parents))
        # Key order matches the established rows: value, gradient, reason, available.
        py_setitem(row, nk[].k_value, _summary_dict(r.value) if r.has_value else PythonObject(None))
        py_setitem(row, nk[].k_gradient, _summary_dict(r.gradient) if r.available else PythonObject(None))
        py_setitem(row, nk[].k_reason, PythonObject(None) if r.available else PythonObject(r.reason))
        py_setitem(row, nk[].k_available, r.available)
        _ = py_call(rows.append, row)
    return rows

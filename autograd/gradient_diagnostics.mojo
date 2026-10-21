# SPDX-License-Identifier: Apache-2.0
# Opt-in diagnostics over the original tape. Numeric summaries run through
# ordinary native executor operations on an independent no-grad tape; only
# scalar summaries cross the device boundary. No gradient or optimizer state
# is modified, and diagnostic storage is released on success and failure.
from std.python import Python, PythonObject
from std.memory import bitcast
from sequence.exec_trait import Exec
from sequence.ops import FP, dummy_ptr
from autograd.ops_api import ops_session
from autograd.tape import Tape
from autograd.import_api import imp_unary, imp_binary
from autograd.import_ops import IMP_ABS, IMP_EQ, IMP_NE
from autograd.profile import ag_kind_name


def _summary[E: Exec](mut ex: E, source: FP, count: Int) raises -> PythonObject:
    var scratch = Tape()
    scratch.no_grad()
    var out = Python.import_module("builtins").dict()
    try:
        var x = scratch.variable(ex, source, count, 1, False)
        var zero = scratch.scalar(ex, Float32(0))
        var infinity = scratch.scalar(ex, bitcast[DType.float32](UInt32(0x7f800000)))
        var absolute = imp_unary(scratch, ex, x, IMP_ABS)
        var nan_mask = imp_binary(scratch, ex, x, x, IMP_NE)
        var inf_mask = imp_binary(scratch, ex, absolute, infinity, IMP_EQ)
        var zero_mask = imp_binary(scratch, ex, x, zero, IMP_EQ)
        var nonzero_mask = imp_binary(scratch, ex, x, zero, IMP_NE)
        var nan_sum = scratch.sum(ex, nan_mask)
        var inf_sum = scratch.sum(ex, inf_mask)
        var zero_sum = scratch.sum(ex, zero_mask)
        var nonzero_sum = scratch.sum(ex, nonzero_mask)
        var square = scratch.multiply(ex, x, x)
        var norm = scratch.sum(ex, square)
        # Presence predicates stay exact even above 2**24 elements; do not
        # turn FP32 mask sums into purported exact integer element counts.
        out["has_nan"] = scratch.values(ex, nan_sum)[0] != 0
        out["has_infinity"] = scratch.values(ex, inf_sum)[0] != 0
        out["has_zero"] = scratch.values(ex, zero_sum)[0] != 0
        out["all_zero"] = scratch.values(ex, nonzero_sum)[0] == 0
        out["squared_l2_float32"] = scratch.values(ex, norm)[0]
    except error:
        scratch.release(ex)
        ex.sync()
        raise error
    scratch.release(ex)
    ex.sync()
    return out


def gradient_report_py[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, handles: List[Int], values: Bool,
) raises -> PythonObject:
    var s = ops_session[E, KIND](sid)
    # Refuse a bad handle before running any diagnostic kernels.
    for j in range(len(handles)):  # driver-loop: over requested tensor handles
        _ = s[]._slot(handles[j])
    var rows = Python.list()
    for j in range(len(handles)):  # driver-loop: over requested gradient diagnostics
        var h = handles[j]
        var slot = s[]._slot(h)
        var t = s[].tensor(h)
        var node_id = s[].tape._check(t)
        var node = s[].tape._nodes[node_id]
        var shape = s[].shapes[slot]
        var row = Python.import_module("builtins").dict()
        row["handle"] = h
        row["node"] = node_id
        row["operation"] = ag_kind_name(node.op)
        row["requires_grad"] = node.requires_grad
        row["inference_value"] = node.inferred
        row["stale"] = len(s[].tape._stale) > node_id and s[].tape._stale[node_id]
        var dims = Python.list()
        for k in range(shape.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata
            dims.append(shape.d[k])
        row["shape"] = Python.import_module("builtins").tuple(dims)
        var parents = Python.list()
        for k in range(node.count):  # driver-loop: over original tape parent references
            parents.append(s[].tape._parents[node.start + k])
        row["parents"] = Python.import_module("builtins").tuple(parents)
        row["value"] = PythonObject(None)
        row["gradient"] = PythonObject(None)
        if values and not s[].int_words[slot]:
            var source = s[].tape.value_of(s[].ex, t)
            row["value"] = _summary(s[].ex, source, shape.count())
        var gradient = dummy_ptr()
        var available = False
        try:
            gradient = s[].tape.gradient(t)
            available = True
            row["reason"] = PythonObject(None)
        except error:
            row["reason"] = String(error)
        row["available"] = available
        if available:
            row["gradient"] = _summary(s[].ex, gradient, shape.count())
        rows.append(row)
    return rows

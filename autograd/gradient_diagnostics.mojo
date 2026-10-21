# SPDX-License-Identifier: Apache-2.0
# Opt-in diagnostics over the original tape. Numeric summaries run through
# ordinary native executor operations on an independent no-grad tape; only
# scalar summaries cross the device boundary. No gradient or optimizer state
# is modified, and diagnostic storage is released on success and failure.
# The Python result rows are built in gradient_diagnostics_pyapi.mojo.
from std.memory import bitcast
from sequence.exec_trait import Exec
from sequence.ops import FP, dummy_ptr
from autograd.ops_api import ops_session
from autograd.tape import Tape
from autograd.import_api import imp_unary, imp_binary
from autograd.import_ops import IMP_ABS, IMP_EQ, IMP_NE
from autograd.profile import ag_kind_name
from autograd.rankn import ShapeN


@fieldwise_init
struct ValueSummary(ImplicitlyCopyable, Movable):
    var has_nan: Bool
    var has_infinity: Bool
    var has_zero: Bool
    var all_zero: Bool
    var squared_l2_float32: Float32


@fieldwise_init
struct GradientRow(Copyable, Movable):
    var handle: Int
    var node: Int
    var operation: String
    var requires_grad: Bool
    var inference_value: Bool
    var stale: Bool
    var shape: ShapeN
    var parents: List[Int]
    var has_value: Bool
    var value: ValueSummary
    var reason: String
    var available: Bool
    var gradient: ValueSummary


def _empty_summary() -> ValueSummary:
    return ValueSummary(False, False, False, False, Float32(0))


def _summary[E: Exec](mut ex: E, source: FP, count: Int) raises -> ValueSummary:
    var scratch = Tape()
    scratch.no_grad()
    var out = _empty_summary()
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
        out.has_nan = scratch.values(ex, nan_sum)[0] != 0
        out.has_infinity = scratch.values(ex, inf_sum)[0] != 0
        out.has_zero = scratch.values(ex, zero_sum)[0] != 0
        out.all_zero = scratch.values(ex, nonzero_sum)[0] == 0
        out.squared_l2_float32 = scratch.values(ex, norm)[0]
    except error:
        scratch.release(ex)
        ex.sync()
        raise error
    scratch.release(ex)
    ex.sync()
    return out


def gradient_report[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, handles: List[Int], values: Bool,
) raises -> List[GradientRow]:
    var s = ops_session[E, KIND](sid)
    # Refuse a bad handle before running any diagnostic kernels.
    for j in range(len(handles)):  # driver-loop: over requested tensor handles
        _ = s[]._slot(handles[j])
    var rows = List[GradientRow]()
    for j in range(len(handles)):  # driver-loop: over requested gradient diagnostics
        var h = handles[j]
        var slot = s[]._slot(h)
        var t = s[].tensor(h)
        var node_id = s[].tape._check(t)
        var node = s[].tape._nodes[node_id]
        var shape = s[].shapes[slot]
        var parents = List[Int]()
        for k in range(node.count):  # driver-loop: over original tape parent references
            parents.append(s[].tape._parents[node.start + k])
        var row = GradientRow(
            h, node_id, ag_kind_name(node.op), node.requires_grad, node.inferred,
            len(s[].tape._stale) > node_id and s[].tape._stale[node_id], shape, parents^,
            False, _empty_summary(), String(""), False, _empty_summary(),
        )
        if values and not s[].int_words[slot]:
            var source = s[].tape.value_of(s[].ex, t)
            row.value = _summary(s[].ex, source, shape.count())
            row.has_value = True
        var gradient = dummy_ptr()
        try:
            gradient = s[].tape.gradient(t)
            row.available = True
        except error:
            row.reason = String(error)
        if row.available:
            row.gradient = _summary(s[].ex, gradient, shape.count())
        rows.append(row^)
    return rows^

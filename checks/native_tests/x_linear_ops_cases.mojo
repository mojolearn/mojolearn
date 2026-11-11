# SPDX-License-Identifier: Apache-2.0
"""The x_linear operations table entry (core/linear_entry_table_ext.mojo),
one case per operation the linear owner (bindings/x_linear_native.mojo)
calls, with expectations known by construction (dyadic data, exact small
systems), not by another call into the implementation. No Python objects.
"""
from std.memory import bitcast
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from core.linear_entry_table_ext import (
    XL_OP_FLAGS, XL_OP_FIT, XL_OP_FIT_FOLD, XL_OP_DECISION, XL_OP_DECISION_CODES,
    XL_OP_DECISION_ROWS, XL_OP_HUBER_OUT, XL_OP_CLASS_PREP, XL_OP_SPEARMAN, XL_OP_BEST_ALPHA,
    XL_HAS_FIT_FOLD,
)

comptime XL_OPS_CASES = 12
comptime _LINK_IDENTITY = 0
comptime _LINK_SIGMOID = 2
comptime _ALGO_RIDGE = 8
comptime _ROWS_SGD_PROBA = 2


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def close(actual: Float64, expected: Float64, atol: Float64, label: String) raises:
    require(abs(actual - expected) <= atol, label + ": " + String(actual) + " != " + String(expected))


def _nan() -> Float32:
    return bitcast[DType.float32](UInt32(0x7FC00000))


def _addr_f32(values: List[Float32]) -> Int:
    return Int(values.unsafe_ptr())


def case_flags(entry: NativeDescriptorCall, fold: Bool) raises:
    var flags = native_descriptor_call(entry, XL_OP_FLAGS, List[Int](), List[Int](), List[Float64]())
    require(((flags & XL_HAS_FIT_FOLD) != 0) == fold, "flags: fold bit " + String(flags))


def case_fit_ridge(entry: NativeDescriptorCall) raises:
    # X = [[1,0],[0,1],[1,0],[0,1]], y = [2,4,2,4], no intercept, alpha 2:
    # (X'X + 2I) w = X'y  ->  4 w = [4, 8]  ->  w = [1, 2].
    var n = 4
    var d = 2
    var x: List[Float32] = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    var y: List[Float32] = [2.0, 4.0, 2.0, 4.0]
    var a = 1
    var n_out = d + 1 + 2 + a + 1
    var out = List[Float32](length=n_out, fill=-7.0)
    var ip: List[Int] = [1, 0, a, 0]
    var ints: List[Int] = [_ALGO_RIDGE, n, d, n * d, n, n_out, 3 * d * d + 3 * d + 1 + d + 2 * n, 1, len(ip), 1]
    for i in range(len(ip)):  # small-loop(the route's four integer hyperparameters)
        ints.append(ip[i])
    var got = native_descriptor_call(entry, XL_OP_FIT, [_addr_f32(x), _addr_f32(y), _addr_f32(out)], ints, [2.0])
    require(got == n_out, "fit: n_out " + String(got))
    close(Float64(out[0]), 1.0, 1e-5, "fit: coef[0]")
    close(Float64(out[1]), 2.0, 1e-5, "fit: coef[1]")
    close(Float64(out[d]), 0.0, 1e-6, "fit: intercept")
    require(out[d + 1 + 2 + a] != 2.0, "fit: singular flag")
    _ = x
    _ = y
    _ = out


def case_decision(entry: NativeDescriptorCall) raises:
    # one class: x . [2, -1] + 0.5 on dyadic rows.
    var x: List[Float32] = [1.0, 2.0, 3.0, 0.5]
    var wb: List[Float32] = [2.0, -1.0, 0.5]
    var out = List[Float32](length=2, fill=-7.0)
    var cells = native_descriptor_call(entry, XL_OP_DECISION, [_addr_f32(x), _addr_f32(wb), _addr_f32(out)],
                                       [2, 2, 1, _LINK_IDENTITY, 0], List[Float64]())
    require(cells == 2, "decision: cells " + String(cells))
    require(out[0] == 0.5 and out[1] == 6.0, "decision: " + String(out[0]) + ", " + String(out[1]))
    _ = x
    _ = wb
    _ = out


def case_decision_status(entry: NativeDescriptorCall) raises:
    # The deferred finite status word: a NaN in X sets it nonzero.
    var x: List[Float32] = [1.0, _nan(), 3.0, 0.5]
    var wb: List[Float32] = [2.0, -1.0, 0.5]
    var out = List[Float32](length=2, fill=0.0)
    var status = List[Int32](length=1, fill=0)
    _ = native_descriptor_call(entry, XL_OP_DECISION, [_addr_f32(x), _addr_f32(wb), _addr_f32(out)],
                               [2, 2, 1, _LINK_IDENTITY, Int(status.unsafe_ptr())], List[Float64]())
    require(status[0] != 0, "decision: the non-finite X was not reported")
    _ = x
    _ = wb
    _ = out
    _ = status


def case_codes(entry: NativeDescriptorCall) raises:
    # k == 1 threshold at 0: scores -1 and 2 -> codes below, above.
    var x: List[Float32] = [-1.0, 2.0]
    var wb: List[Float32] = [1.0, 0.0]
    var out = List[Int32](length=2, fill=-7)
    var rows = native_descriptor_call(entry, XL_OP_DECISION_CODES,
                                      [_addr_f32(x), _addr_f32(wb), Int(out.unsafe_ptr())],
                                      [2, 1, 1, _LINK_IDENTITY, 1, 0, 1, 0], List[Float64]())
    require(rows == 2, "codes: rows " + String(rows))
    require(out[0] == 0 and out[1] == 1, "codes: " + String(out[0]) + ", " + String(out[1]))
    _ = x
    _ = wb
    _ = out


def case_codes_argmax(entry: NativeDescriptorCall) raises:
    # k == 2 argmax: row 0 favors class 1, row 1 class 0.
    var x: List[Float32] = [1.0, -1.0]
    var wb: List[Float32] = [-1.0, 0.0, 1.0, 0.0]
    var out = List[Int32](length=2, fill=-7)
    _ = native_descriptor_call(entry, XL_OP_DECISION_CODES,
                               [_addr_f32(x), _addr_f32(wb), Int(out.unsafe_ptr())],
                               [2, 1, 2, _LINK_IDENTITY, 1, 0, 1, 0], List[Float64]())
    require(out[0] == 1 and out[1] == 0, "codes argmax: " + String(out[0]) + ", " + String(out[1]))
    _ = x
    _ = wb
    _ = out


def case_decision_rows(entry: NativeDescriptorCall) raises:
    # zero weights: sigmoid(0) = 1/2, so each row is [1/2, 1/2].
    var x: List[Float32] = [3.0, -5.0, 0.25, 8.0]
    var wb: List[Float32] = [0.0, 0.0, 0.0]
    var out = List[Float32](length=4, fill=-7.0)
    var rows = native_descriptor_call(entry, XL_OP_DECISION_ROWS, [_addr_f32(x), _addr_f32(wb), _addr_f32(out)],
                                      [2, 2, 1, _LINK_SIGMOID, _ROWS_SGD_PROBA, 0], List[Float64]())
    require(rows == 2, "decision rows: rows " + String(rows))
    for i in range(4):  # small-loop(two rows of two probabilities)
        require(out[i] == 0.5, "decision rows: word " + String(i) + " = " + String(out[i]))
    _ = x
    _ = wb
    _ = out


def case_huber_out(entry: NativeDescriptorCall) raises:
    # pred = x; |y - pred| = [0.25, 4] against thr 1 -> [0, 1].
    var x: List[Float32] = [1.0, 2.0]
    var wb: List[Float32] = [1.0, 0.0]
    var y: List[Float32] = [1.25, 6.0]
    var flags = List[UInt8](length=2, fill=7)
    var rows = native_descriptor_call(entry, XL_OP_HUBER_OUT,
                                      [_addr_f32(x), _addr_f32(wb), _addr_f32(y), Int(flags.unsafe_ptr())],
                                      [2, 1, _LINK_IDENTITY], [1.0])
    require(rows == 2, "huber: rows " + String(rows))
    require(flags[0] == 0 and flags[1] == 1, "huber: " + String(flags[0]) + ", " + String(flags[1]))
    _ = x
    _ = wb
    _ = y
    _ = flags


def case_class_prep(entry: NativeDescriptorCall) raises:
    # codes [0, 0, 1, 0], 'balanced': n / (k * count) = [4/6, 2], largest 3.
    var codes: List[Int32] = [0, 0, 1, 0]
    var cw = List[Float32](length=2, fill=0.0)
    var largest = native_descriptor_call(entry, XL_OP_CLASS_PREP,
                                         [Int(codes.unsafe_ptr()), 0, _addr_f32(cw), 0], [4, 2, 1, 0],
                                         List[Float64]())
    require(largest == 3, "class prep: largest " + String(largest))
    close(Float64(cw[0]), 4.0 / 6.0, 1e-6, "class prep: balanced weight 0")
    close(Float64(cw[1]), 2.0, 1e-6, "class prep: balanced weight 1")
    _ = codes
    _ = cw


def case_spearman(entry: NativeDescriptorCall) raises:
    var x: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var down: List[Float32] = [8.0, 4.0, 2.0, 1.0]
    var up: List[Float32] = [0.5, 1.0, 1.5, 9.0]
    var s_down = native_descriptor_call(entry, XL_OP_SPEARMAN, [_addr_f32(x), _addr_f32(down)], [4], List[Float64]())
    var s_up = native_descriptor_call(entry, XL_OP_SPEARMAN, [_addr_f32(x), _addr_f32(up)], [4], List[Float64]())
    require(s_down == -1 and s_up == 1, "spearman: " + String(s_down) + ", " + String(s_up))
    _ = x
    _ = down
    _ = up


def case_best_alpha(entry: NativeDescriptorCall) raises:
    var nan = _nan()
    var scores: List[Float32] = [0.25, nan, 0.75, 0.75, 0.5]
    var none: List[Float32] = [nan, nan]
    var best = native_descriptor_call(entry, XL_OP_BEST_ALPHA, [_addr_f32(scores)], [5], List[Float64]())
    var missing = native_descriptor_call(entry, XL_OP_BEST_ALPHA, [_addr_f32(none)], [2], List[Float64]())
    require(best == 2 and missing == -1, "best alpha: " + String(best) + ", " + String(missing))
    _ = scores
    _ = none


def case_refusals(entry: NativeDescriptorCall, fold: Bool) raises:
    # Mismatched spans and an unknown operation refuse before any word is read.
    var refused = False
    try:
        _ = native_descriptor_call(entry, XL_OP_DECISION, [0], [1], List[Float64]())
    except:
        refused = True
    require(refused, "a short decision request was accepted")
    refused = False
    try:
        _ = native_descriptor_call(entry, 99, List[Int](), List[Int](), List[Float64]())
    except:
        refused = True
    require(refused, "an unknown operation was accepted")
    if not fold:
        refused = False
        try:
            _ = native_descriptor_call(entry, XL_OP_FIT_FOLD, [0, 0, 0],
                                       [_ALGO_RIDGE, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0], List[Float64]())
        except:
            refused = True
        require(refused, "a build without the fold store accepted a fold fit")


def run_x_linear_ops_case(entry: NativeDescriptorCall, fold: Bool, case_id: Int) raises:
    if case_id == 0:
        case_flags(entry, fold)
    elif case_id == 1:
        case_fit_ridge(entry)
    elif case_id == 2:
        case_decision(entry)
    elif case_id == 3:
        case_decision_status(entry)
    elif case_id == 4:
        case_codes(entry)
    elif case_id == 5:
        case_codes_argmax(entry)
    elif case_id == 6:
        case_decision_rows(entry)
    elif case_id == 7:
        case_huber_out(entry)
    elif case_id == 8:
        case_class_prep(entry)
    elif case_id == 9:
        case_spearman(entry)
    elif case_id == 10:
        case_best_alpha(entry)
    else:
        case_refusals(entry, fold)

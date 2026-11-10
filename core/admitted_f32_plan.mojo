# SPDX-License-Identifier: Apache-2.0
"""Native float32 admission plans, validation and preparation execution.

Python boundary code admits exact exporters and retains their storage. This
module receives only the resulting native dimensions, layout and addresses;
the existing parallel cast/finite/weight routines remain the operations.
"""
from core.admitted_input import admitted_cast_f32, admitted_check_f32, admitted_check_weights_f32


struct AdmittedInputShape(Copyable, Movable):
    var rank: Int
    var rows: Int
    var cols: Int

    def __init__(out self, rank: Int, rows: Int, cols: Int):
        self.rank = rank
        self.rows = rows
        self.cols = cols


def admitted_numeric_kind(kind: String) -> Bool:
    return kind == "b" or kind == "i" or kind == "u" or kind == "f"


struct AdmittedF32Plan(Movable):
    var address: Int
    var rows: Int
    var cols: Int
    var rank: Int
    var item: Int
    var kind: Int
    var s0: Int
    var s1: Int
    var borrowed: Bool
    var want_f: Bool
    var name: String

    def __init__(out self, address: Int, rows: Int, cols: Int, rank: Int,
        item: Int, kind: Int, s0: Int, s1: Int, borrowed: Bool, want_f: Bool, name: String):
        self.address = address
        self.rows = rows
        self.cols = cols
        self.rank = rank
        self.item = item
        self.kind = kind
        self.s0 = s0
        self.s1 = s1
        self.borrowed = borrowed
        self.want_f = want_f
        self.name = String(name)


def admitted_f32_column(expected_rank: Int, rank: Int, cols: Int,
    column_target: Bool, column_source: Bool) -> Bool:
    return column_target and expected_rank == 1 and rank == 2 and cols == 1 and column_source


def admitted_f32_shape_check(rank: Int, expected_rank: Int, column: Bool,
    rows: Int, cols: Int, name: String) raises -> Int:
    if expected_rank >= 0 and rank != expected_rank and not column:
        raise Error(name + " must be " + String(expected_rank) + "-D")
    if rank > 2:
        raise Error(name + " must be at most 2-D")
    if rows < 1 or cols < 1:
        raise Error(name + " is empty")
    return rank if expected_rank < 0 else expected_rank


def admitted_f32_byte_order(native_order: Bool, name: String) raises:
    if not native_order:
        raise Error("TypeError: " + name + " must be in native byte order")


def admitted_f32_order(rank: Int, column: Bool, order: String) raises -> Bool:
    if order != "C" and order != "F":
        raise Error("ValueError: order must be 'C' or 'F', not " + order)
    return order == "F" and rank == 2 and not column


def admitted_f32_plan(address: Int, rows: Int, cols: Int, rank: Int,
    item: Int, kind: Int, s0: Int, s1: Int, is_f32: Bool, is_array: Bool, c_order: Bool,
    column: Bool, want_f: Bool, as_array: Bool, name: String) -> AdmittedF32Plan:
    var in_order = c_order
    if want_f:
        in_order = (s0 == 4 or rows == 1) and (s1 == rows * 4 or cols == 1)
    var borrowed = not column and is_f32 and in_order and (not as_array or is_array)
    return AdmittedF32Plan(address, rows, cols, rank, item, kind, s0, s1, borrowed, want_f, name)


def admitted_f32_run(plan: AdmittedF32Plan, destination: Int, finite: Bool, defer_finite: Bool) raises:
    var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=destination)
    var s0 = plan.s1 if plan.want_f else plan.s0
    var s1 = plan.s0 if plan.want_f else plan.s1
    var rows = plan.cols if plan.want_f else plan.rows
    var cols = plan.rows if plan.want_f else plan.cols
    if not finite:
        if not plan.borrowed:
            _ = admitted_cast_f32(plan.address, s0, s1, rows, cols, plan.item, plan.kind, dst)
    elif not (plan.borrowed and defer_finite):
        admitted_check_f32(plan.address, s0, s1, rows, cols, plan.item, plan.kind, plan.name, plan.borrowed, dst)


def admitted_x_shape_error(rank: Int, rows: Int, cols: Int, numeric: Bool,
    complex: Bool, classical: Bool) -> Int:
    if classical and complex:
        return 1
    if classical and numeric:
        if rank < 2:
            return 2
        if rank == 2 and rows == 0:
            return 3
        if rank == 2 and cols == 0:
            return 4
    if rank != 2:
        return 5
    if not numeric:
        return 6
    if rows < 1 or cols < 1:
        return 7
    return 0


def admitted_x_shape_refuse(status: Int, rank: Int, shape: String, format: String,
    estimator: String) raises:
    if status == 1:
        raise Error("Complex data not supported: " + estimator + " got a complex X.")
    if status == 2:
        raise Error("Expected 2D array, got " + ("scalar" if rank == 0 else "1D") + " array instead (shape=" + shape
            + "). Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.")
    if status == 3:
        raise Error("Found array with 0 sample(s) (shape=" + shape + ") while a minimum of 1 is required by " + estimator + ".")
    if status == 4:
        raise Error("Found array with 0 feature(s) (shape=" + shape + ") while a minimum of 1 is required by " + estimator + ".")
    if status == 5:
        raise Error("mojolearn: X must be 2-D, got " + String(rank) + "-D shape " + shape)
    if status == 6:
        raise Error("TypeError: mojolearn: X has buffer format " + format + ", which is not a numeric array")
    if status == 7:
        raise Error("X is empty")


def admitted_target_present(present: Bool, estimator: String) raises:
    if not present:
        raise Error(estimator + " requires y to be passed, but the target y is None.")


def admitted_target_plan(rank: Int, cols: Int, column_source: Bool, complex: Bool,
    estimator: String) raises -> Int:
    if complex:
        raise Error("Complex data not supported: " + estimator + " got a complex y.")
    if rank == 1:
        return 1
    if rank == 2 and cols == 1 and column_source:
        return 2
    return 0


def admitted_target_shape(rows: Int, length: Int, valid: Bool, estimator: String) raises:
    if length != rows:
        raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(length) + "]")
    if not valid:
        raise Error("mojolearn " + estimator + " currently requires one target")


def admitted_weight_shape(rank: Int, length: Int, rows: Int) raises -> Bool:
    if rank > 1:
        raise Error("Sample weights must be 1D array or scalar")
    var scalar = rank == 0
    if not scalar and length != rows:
        raise Error("sample_weight.shape == (" + String(length) + ",), expected (" + String(rows) + ",)!")
    return scalar


def admitted_weights_run(input: Int, destination: Int, rows: Int, scalar: Bool) raises:
    if not admitted_check_weights_f32(
            MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=input),
            MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=destination), rows, scalar):
        raise Error("Sample weights must contain at least one non-zero number.")

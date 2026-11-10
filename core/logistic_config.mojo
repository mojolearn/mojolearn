# SPDX-License-Identifier: Apache-2.0
"""Typed scalar configuration for the existing QN operations.

These are bounded configuration decisions, not data kernels. NumPy's scalar
precision is retained: weak Python operands use the NumPy floating width;
legacy promotion uses binary64. No Python numeric slot participates.
"""
from std.memory import bitcast
from std.math import isfinite, isinf, isnan
from core.sequence_array_cast import SEQ_CAST_OVERFLOW, SEQ_CAST_UNDERFLOW, SEQ_CAST_INVALID

comptime LOGISTIC_NO_PENALTY = 0
comptime LOGISTIC_L1 = 1
comptime LOGISTIC_L2 = 2
comptime LOGISTIC_ELASTIC = 3


struct LogisticScalar(Copyable, Movable):
    var value: Float64
    # 0: builtin real, 1: NumPy floating, 2: NumPy integer, 3: NumPy bool,
    # 4: builtin bool (a strong boolean even under legacy scalar promotion).
    var kind: Int
    var width: Int
    var signaling: Bool

    def __init__(out self, value: Float64, kind: Int, width: Int, signaling: Bool = False):
        self.value = value
        self.kind = kind
        self.width = width
        self.signaling = signaling

    def numpy(self) -> Bool:
        return 1 <= self.kind <= 3


@fieldwise_init
struct LogisticArithmetic(Copyable, Movable):
    var scalar: LogisticScalar
    var status: Int
    var cast_status: Int
    var operation: Int
    var scalar_message: Bool


@fieldwise_init
struct LogisticStrengthPlan(Copyable, Movable):
    var l1: Float64
    var l2: Float64
    var division: LogisticArithmetic
    var first: LogisticArithmetic
    var complement: LogisticArithmetic
    var second: LogisticArithmetic

    def event(self, index: Int) -> LogisticArithmetic:
        if index == 0:
            return self.division.copy()
        if index == 1:
            return self.first.copy()
        if index == 2:
            return self.complement.copy()
        return self.second.copy()


struct LogisticPromotion(Copyable, Movable):
    var captured: Bool
    var weak: Bool

    def __init__(out self):
        self.captured = False
        self.weak = False


def logistic_penalty(name: String, missing: Bool) -> Int:
    if missing:
        return LOGISTIC_NO_PENALTY
    if name == "l1":
        return LOGISTIC_L1
    if name == "l2":
        return LOGISTIC_L2
    if name == "elasticnet":
        return LOGISTIC_ELASTIC
    return -1


def logistic_positive(value: Float64) -> Bool:
    # The original <= comparison accepts NaN; finite/data policy belongs to
    # the unchanged QN operation, not an extra constructor restriction.
    return not (value <= 0.0)


def logistic_ratio_valid(value: Float64) -> Bool:
    return not (value < 0.0 or value > 1.0)


def _logistic_float_width(value: LogisticScalar) -> Int:
    if value.kind == 1:
        return value.width
    if value.kind == 2:
        # NumPy's real promotion lattice: binary16 can represent int8;
        # binary32 can represent int16; wider integers promote to binary64.
        return 2 if value.width == 1 else (4 if value.width == 2 else 8)
    return 0


def _logistic_result_width(left: LogisticScalar, right: LogisticScalar, weak: Bool) -> Int:
    if not left.numpy() and not right.numpy():
        return 8
    if not left.numpy() or not right.numpy():
        var native = right.copy() if not left.numpy() else left.copy()
        var builtin_kind = left.kind if not left.numpy() else right.kind
        return native.width if (weak or builtin_kind == 4) and native.kind == 1 else 8
    return max(2, max(_logistic_float_width(left), _logistic_float_width(right)))


def _logistic_low_bit(value: Float64) -> Int:
    # Exponent of the least significant nonzero bit of a finite real. This
    # determines exact subnormal multiplication without host fenv state or
    # a second floating-point calculation that can itself set flags.
    var bits = bitcast[DType.uint64](value) & UInt64(0x7FFFFFFFFFFFFFFF)
    var exponent = Int(bits >> 52)
    var mantissa = bits & UInt64(0x000FFFFFFFFFFFFF)
    var scale = -1074
    if exponent != 0:
        mantissa |= UInt64(1) << 52
        scale = exponent - 1023 - 52
    for i in range(53):  # driver-loop: one scalar significand, never data traversal
        if (mantissa & UInt64(1)) != 0:
            break
        mantissa >>= 1
        scale += 1
    return scale


def _logistic_power_of_two(value: Float64) -> Bool:
    var bits = bitcast[DType.uint64](value) & UInt64(0x7FFFFFFFFFFFFFFF)
    var fraction = bits & UInt64(0x000FFFFFFFFFFFFF)
    if (bits >> 52) != 0:
        return fraction == 0
    return fraction != 0 and (fraction & (fraction - UInt64(1))) == 0


def logistic_scalar_operation(left: LogisticScalar, right: LogisticScalar,
                              operation: Int, weak: Bool) -> LogisticArithmetic:
    # operation: 0 multiply, 1 subtract, 2 divide. Each source expression
    # rounds separately, including float16's Float32 intermediate.
    var width = _logistic_result_width(left, right, weak)
    var result: Float64
    var a64 = left.value
    var b64 = right.value
    if width == 2:
        var a = Float32(Float16(left.value))
        var b = Float32(Float16(right.value))
        a64 = Float64(a)
        b64 = Float64(b)
        var value = a * b if operation == 0 else (a - b if operation == 1 else a / b)
        result = Float64(Float16(value))
    elif width == 4:
        var a = Float32(left.value)
        var b = Float32(right.value)
        a64 = Float64(a)
        b64 = Float64(b)
        var value = a * b if operation == 0 else (a - b if operation == 1 else a / b)
        result = Float64(value)
    else:
        result = (left.value * right.value if operation == 0 else
                  (left.value - right.value if operation == 1 else left.value / right.value))
    var numpy = left.numpy() or right.numpy()
    var status = 0
    var cast_status = 0
    if numpy:
        # NumPy scalar setitem reports overflow on weak Python real casts;
        # its underflow/inexact cast flags are cleared before the operation.
        if (not left.numpy() and isfinite(left.value) and isinf(a64)) or (not right.numpy() and isfinite(right.value) and isinf(b64)):
            cast_status = SEQ_CAST_OVERFLOW
        if (left.signaling and (left.numpy() or width == 8)) or (right.signaling and (right.numpy() or width == 8)):
            status |= SEQ_CAST_INVALID
        if not isnan(a64) and not isnan(b64):
            if (operation == 0 and ((a64 == 0.0 and isinf(b64)) or (b64 == 0.0 and isinf(a64)))) or (operation == 1 and isinf(a64) and a64 == b64):
                status |= SEQ_CAST_INVALID
            if isfinite(a64) and isfinite(b64):
                if isinf(result):
                    status |= SEQ_CAST_OVERFLOW
                else:
                    var minimum = bitcast[DType.float64](UInt64(1009 if width == 2 else (897 if width == 4 else 1)) << 52)
                    var last_bit = -24 if width == 2 else (-149 if width == 4 else -1074)
                    if abs(result) < minimum and a64 != 0.0 and b64 != 0.0:
                        if operation == 0 and _logistic_low_bit(a64) + _logistic_low_bit(b64) < last_bit:
                            status |= SEQ_CAST_UNDERFLOW
                        elif operation == 2 and not _logistic_power_of_two(b64):
                            # Division is used only for 1/C. A finite binary
                            # reciprocal is exact iff C is a power of two.
                            status |= SEQ_CAST_UNDERFLOW
    var scalar_message = False
    if numpy:
        # When neither original NumPy dtype is the promoted floating dtype,
        # NumPy dispatches a ufunc and omits "scalar " in its diagnostic.
        scalar_message = ((left.kind == 1 and left.width == width) or
                          (right.kind == 1 and right.width == width))
    return LogisticArithmetic(LogisticScalar(result, 1 if numpy else 0, width),
        status, cast_status, operation, scalar_message)


def logistic_strength_plan(penalty: Int, c: LogisticScalar, ratio: LogisticScalar,
                           weak: Bool) -> LogisticStrengthPlan:
    var none = LogisticArithmetic(LogisticScalar(0.0, 0, 8), 0, 0, 0, False)
    if penalty == LOGISTIC_NO_PENALTY:
        return LogisticStrengthPlan(0.0, 0.0, none.copy(), none.copy(), none.copy(), none^)
    var one = LogisticScalar(1.0, 0, 8)
    var strength = logistic_scalar_operation(one, c, 2, weak)
    if penalty == LOGISTIC_L1:
        return LogisticStrengthPlan(strength.scalar.value, 0.0, strength^, none.copy(), none.copy(), none^)
    if penalty == LOGISTIC_L2:
        return LogisticStrengthPlan(0.0, strength.scalar.value, strength^, none.copy(), none.copy(), none^)
    var left = logistic_scalar_operation(ratio, strength.scalar, 0, weak)
    var remaining = logistic_scalar_operation(one, ratio, 1, weak)
    var right = logistic_scalar_operation(remaining.scalar, strength.scalar, 0, weak)
    return LogisticStrengthPlan(left.scalar.value, right.scalar.value, strength^, left^, remaining^, right^)


def logistic_strengths(penalty: Int, c: LogisticScalar, ratio: LogisticScalar,
                       weak: Bool) -> Tuple[Float64, Float64]:
    var plan = logistic_strength_plan(penalty, c, ratio, weak)
    return (plan.l1, plan.l2)


def logistic_change_tolerance(tolerance: LogisticScalar, weak: Bool) -> Float64:
    return logistic_scalar_operation(tolerance, LogisticScalar(0.01, 0, 8), 0, weak).scalar.value

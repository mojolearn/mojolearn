# SPDX-License-Identifier: Apache-2.0
"""Typed comparison of admitted public Logistic feature metadata."""
from std.memory import bitcast
from core.native_scalar_decode import NativeRealBits, _scalar_ieee


def logistic_width_integer(value: Int) -> NativeRealBits:
    var magnitude = UInt64(-(value + 1)) + UInt64(1) if value < 0 else UInt64(value)
    return NativeRealBits(magnitude, negative=value < 0)


def _logistic_real_equal(left: NativeRealBits, right: NativeRealBits) -> Bool:
    if left.special == 2 or right.special == 2:
        return False
    if left.special == 1 or right.special == 1:
        return left.special == right.special and left.negative == right.negative
    if not left.truth() or not right.truth():
        return not left.truth() and not right.truth()
    if left.negative != right.negative:
        return False
    var top = left.highest() + left.scale
    if top != right.highest() + right.scale:
        return False
    for i in range(128):  # driver-loop: compare at most two limbs of scalar metadata
        if left.bit(top - i - left.scale) != right.bit(top - i - right.scale):
            return False
    return True


struct LogisticQueryWidth(ImplicitlyCopyable, Movable):
    var real: NativeRealBits
    var imag: NativeRealBits
    # kind: 0 decoded numeric, 1 unequal nonnumeric/oversized integer,
    # 2 legacy empty ndarray truth, 3 ordinary native integer. The common
    # fitted integer uses one integer comparison, never scalar bit traversal.
    var kind: Int
    # 0 retains exact integer comparison; 16/32/64 preserves the admitted
    # NumPy/Torch weak scalar conversion before comparison. Extended real
    # formats represent every native feature count exactly.
    var precision: Int
    var integer_bits: Int
    var integer_signed: Bool
    var numpy_cast: Bool
    var integer_value: Int

    def __init__(out self, real: NativeRealBits = NativeRealBits(),
                 imag: NativeRealBits = NativeRealBits(), kind: Int = 0,
                 precision: Int = 0, integer_bits: Int = 0,
                 integer_signed: Bool = False, numpy_cast: Bool = False,
                 integer_value: Int = 0):
        self.real = real
        self.imag = imag
        self.kind = kind
        self.precision = precision
        self.integer_bits = integer_bits
        self.integer_signed = integer_signed
        self.numpy_cast = numpy_cast
        self.integer_value = integer_value

    def cast_overflow(self, features: Int) -> Bool:
        # Only binary16 can overflow when converting a native feature count.
        return self.numpy_cast and self.precision == 16 and (features >= 65520 or features <= -65520)

    def matches(self, features: Int) -> Bool:
        if self.kind == 3:
            return self.integer_value == features
        if self.kind != 0:
            return self.kind == 2
        if self.imag.truth():
            return False
        if self.integer_bits != 0:
            # Torch's wrapped integral scalar follows the tensor dtype.
            # No tensor element operation is needed for one metadata value.
            var bits = UInt64(features)
            if self.integer_bits < 64:
                bits &= (UInt64(1) << UInt64(self.integer_bits)) - UInt64(1)
            var negative = self.integer_signed and ((bits >> UInt64(self.integer_bits - 1)) & UInt64(1)) != 0
            if negative:
                bits = UInt64(0) - bits
                if self.integer_bits < 64:
                    bits &= (UInt64(1) << UInt64(self.integer_bits)) - UInt64(1)
            return _logistic_real_equal(self.real, NativeRealBits(bits, negative=negative))
        if self.precision == 0:
            return self.real.equals_integer(features)
        var number = Float64(features)
        if self.precision == 16:
            number = Float64(Float16(number))
        elif self.precision == 8:
            number = Float64(Scalar[DType.bfloat16](Float32(number)))
        elif self.precision == 32:
            number = Float64(Float32(number))
        var other = _scalar_ieee(bitcast[DType.uint64](number), 0, 52, 11)
        return _logistic_real_equal(self.real, other)

    def validate(self, features: Int, host: Bool = False) raises:
        if not self.matches(features):
            raise Error((String("") if host else String("mojolearn ")) +
                        "LogisticRegression feature count differs from fit")


def logistic_query_integer_width(features: Int, fitted: Int, host: Bool = False) raises:
    # Binary prediction and explicit host inference historically converted
    # their admitted field to Int before comparing it; retain that separate
    # contract instead of silently applying the equality policy above.
    if features != fitted:
        raise Error((String("") if host else String("mojolearn ")) +
                    "LogisticRegression feature count differs from fit")

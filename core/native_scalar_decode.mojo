# SPDX-License-Identifier: Apache-2.0
# Primitive scalar storage decoded in Mojo. The public PEP3118 buffer, not
# a NumPy number slot, supplies these bytes. No precision is lost merely to
# compare metadata or decide truth; conversion rounds only at its boundary.
# Format: https://peps.python.org/pep-3118/
# Arm ABI: https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst
# Apple: https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms
from std.sys.info import CompilationTarget


struct NativeRealBits(ImplicitlyCopyable, Movable):
    # A finite magnitude is (high * 2**64 + low) * 2**scale.
    # special: 0 finite, 1 infinity, 2 NaN. NaNs retain their payload bits.
    var low: UInt64
    var high: UInt64
    var scale: Int
    var precision: Int
    var negative: Bool
    var special: Int

    def __init__(out self, low: UInt64 = 0, high: UInt64 = 0,
                 scale: Int = 0, precision: Int = 64,
                 negative: Bool = False, special: Int = 0):
        self.low = low
        self.high = high
        self.scale = scale
        self.precision = precision
        self.negative = negative
        self.special = special

    def bit(self, position: Int) -> UInt64:
        if position < 0 or position >= 128:
            return 0
        if position < 64:
            return (self.low >> UInt64(position)) & UInt64(1)
        return (self.high >> UInt64(position - 64)) & UInt64(1)

    def highest(self) -> Int:
        for i in range(128):  # driver-loop: at most two limbs of an admitted scalar
            var position = 127 - i
            if self.bit(position) != 0:
                return position
        return -1

    def truth(self) -> Bool:
        return self.special != 0 or self.low != 0 or self.high != 0

    def equals_integer(self, target: Int) -> Bool:
        if self.special != 0:
            return False
        if not self.truth():
            return target == 0
        if self.negative != (target < 0):
            return False
        var magnitude = UInt64(-(target + 1)) + UInt64(1) if target < 0 else UInt64(target)
        var top = self.highest() + self.scale
        if top < 0 or top >= 64:
            return False
        for i in range(128):  # driver-loop: exact integral equality, including extended precision residuals
            if self.bit(i) != 0 and i + self.scale < 0:
                return False
        for i in range(64):  # driver-loop: compare the machine integer without rounding the source
            if self.bit(i - self.scale) != ((magnitude >> UInt64(i)) & UInt64(1)):
                return False
        return True

    def _rounded(self, shift: Int) -> UInt64:
        # The destination has at most 53 bits. Extract those bits directly,
        # then apply round-to-nearest, ties-to-even using all discarded bits.
        var result = UInt64(0)
        for i in range(54):  # driver-loop: binary64 result significand and possible carry
            result |= self.bit(i + shift) << UInt64(i)
        if shift > 0 and self.bit(shift - 1) != 0:
            var sticky = False
            for i in range(128):  # driver-loop: discarded source significand bits
                sticky = sticky or (i < shift - 1 and self.bit(i) != 0)
            if sticky or (result & UInt64(1)) != 0:
                result += 1
        return result

    def float64_bits(self) -> UInt64:
        var sign = UInt64(self.negative) << 63
        if self.special == 1:
            return sign | UInt64(0x7FF0000000000000)
        if self.special == 2:
            var payload = UInt64(0)
            var shift = self.precision - 1 - 52
            for i in range(52):  # driver-loop: native conversion preserves the representable NaN payload
                payload |= self.bit(i + shift) << UInt64(i)
            # A binary64 scalar copies its existing bits. Widening/narrowing
            # another floating format quiets a signaling NaN, as native C
            # conversions do; no floating arithmetic or warning slot runs.
            if self.precision != 53:
                payload |= UInt64(0x0008000000000000)
            return sign | UInt64(0x7FF0000000000000) | payload
        var highest = self.highest()
        if highest < 0:
            return sign
        var exponent = highest + self.scale
        if exponent > 1023:
            return sign | UInt64(0x7FF0000000000000)
        if exponent < -1075:
            return sign
        if exponent < -1022:
            # A rounded subnormal may carry into the smallest normal.
            return sign | self._rounded(-1074 - self.scale)
        var significand = self._rounded(highest - 52)
        if significand == (UInt64(1) << 53):
            significand >>= 1
            exponent += 1
            if exponent > 1023:
                return sign | UInt64(0x7FF0000000000000)
        return sign | (UInt64(exponent + 1023) << 52) | (significand & UInt64(0x000FFFFFFFFFFFFF))

    def integer_text(self) -> String:
        # Truncate directly to an arbitrary precision builtin integer. Its
        # publication parser receives already-computed binary digits; no
        # NumPy scalar or user conversion protocol participates.
        var highest = self.highest() + self.scale
        if self.highest() < 0 or highest < 0:
            return "0"
        var result = String("-") if self.negative else String("")
        for i in range(highest + 1):  # driver-loop: bounded by the admitted long-double exponent range
            result += "1" if self.bit(highest - i - self.scale) != 0 else "0"
        return result


def _scalar_word(data: MutPointer[UInt8, MutUntrackedOrigin], offset: Int,
                 width: Int, little: Bool) -> UInt64:
    var result = UInt64(0)
    for i in range(width):  # driver-loop: unaligned scalar bytes, never padding or an adjacent item
        var index = i if little else width - 1 - i
        result |= UInt64(data[offset + index]) << UInt64(8 * i)
    return result


def _scalar_ieee(low: UInt64, high: UInt64, fraction: Int,
                 exponent_bits: Int) -> NativeRealBits:
    var negative = False
    var exponent = 0
    var fraction_low = low
    var fraction_high = UInt64(0)
    if fraction < 64:
        negative = ((low >> UInt64(fraction + exponent_bits)) & UInt64(1)) != 0
        exponent = Int((low >> UInt64(fraction)) & ((UInt64(1) << UInt64(exponent_bits)) - UInt64(1)))
        fraction_low &= (UInt64(1) << UInt64(fraction)) - UInt64(1)
    else:
        negative = (high >> UInt64(fraction - 64 + exponent_bits)) != 0
        exponent = Int((high >> UInt64(fraction - 64)) & ((UInt64(1) << UInt64(exponent_bits)) - UInt64(1)))
        fraction_high = high & ((UInt64(1) << UInt64(fraction - 64)) - UInt64(1))
    var maximum = (1 << exponent_bits) - 1
    if exponent == maximum:
        return NativeRealBits(fraction_low, fraction_high, 0, fraction + 1,
                              negative, 1 if fraction_low == 0 and fraction_high == 0 else 2)
    var bias = (1 << (exponent_bits - 1)) - 1
    var scale = (exponent if exponent != 0 else 1) - bias - fraction
    if exponent != 0:
        if fraction < 64:
            fraction_low |= UInt64(1) << UInt64(fraction)
        else:
            fraction_high |= UInt64(1) << UInt64(fraction - 64)
    return NativeRealBits(fraction_low, fraction_high, scale, fraction + 1, negative)


def native_scalar_real(data: MutPointer[UInt8, MutUntrackedOrigin], offset: Int,
                       code: String, width: Int, little: Bool) raises -> NativeRealBits:
    if code == "?" and width == 1:
        return NativeRealBits(UInt64(data[offset] != 0))
    if code == "b" or code == "B" or code == "h" or code == "H" or code == "i" or code == "I" or code == "l" or code == "L" or code == "q" or code == "Q" or code == "n" or code == "N":
        if width != 1 and width != 2 and width != 4 and width != 8:
            raise Error("TypeError: numeric scalar integer storage has an unsupported width")
        var bits = _scalar_word(data, offset, width, little)
        var signed = code == "b" or code == "h" or code == "i" or code == "l" or code == "q" or code == "n"
        var negative = signed and (bits >> UInt64(width * 8 - 1)) != 0
        if negative:
            bits = ~bits + UInt64(1)
            if width < 8:
                bits &= (UInt64(1) << UInt64(width * 8)) - UInt64(1)
        return NativeRealBits(bits, negative=negative)
    if code == "e" and width == 2:
        return _scalar_ieee(_scalar_word(data, offset, 2, little), 0, 10, 5)
    if code == "f" and width == 4:
        return _scalar_ieee(_scalar_word(data, offset, 4, little), 0, 23, 8)
    if (code == "d" or code == "g") and width == 8:
        # Apple arm64's documented long double ABI is IEEE binary64.
        return _scalar_ieee(_scalar_word(data, offset, 8, little), 0, 52, 11)
    if code == "g" and width == 16:
        # PEP3118 g denotes the platform C long double. SysV x86-64 stores
        # x87 binary80 in the first ten bytes plus six padding bytes;
        # AAPCS64 Linux uses IEEE binary128. These are public platform ABIs,
        # not NumPy object layouts. Never read or compare x87 padding.
        comptime if CompilationTarget.is_x86():
            var significand = _scalar_word(data, offset if little else offset + 8, 8, little)
            var head = _scalar_word(data, offset + 8 if little else offset + 6, 2, little)
            var exponent = Int(head & UInt64(0x7FFF))
            var negative = (head & UInt64(0x8000)) != 0
            if exponent == 32767:
                var payload = significand & UInt64(0x7FFFFFFFFFFFFFFF)
                return NativeRealBits(payload, 0, 0, 64, negative, 1 if payload == 0 else 2)
            return NativeRealBits(significand, 0, (exponent if exponent != 0 else 1) - 16383 - 63, 64, negative)
        elif CompilationTarget.is_arm() and CompilationTarget.is_linux():
            var low = _scalar_word(data, offset if little else offset + 8, 8, little)
            var high = _scalar_word(data, offset + 8 if little else offset, 8, little)
            return _scalar_ieee(low, high, 112, 15)
        else:
            raise Error("TypeError: numeric scalar long-double storage requires a supported platform ABI")
    raise Error("TypeError: numeric scalar requires a primitive numeric buffer format")

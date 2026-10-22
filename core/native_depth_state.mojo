# SPDX-License-Identifier: Apache-2.0
# Exact +/-1 for native floating fit-depth metadata. Each operation rounds
# separately, as the original NumPy scalar/ufunc operations did. This uses
# supported Mojo integer arithmetic over decoded public binary formats; it
# does not depend on a compiler long-double type or private NumPy layouts.
from core.native_scalar_decode import NativeRealBits
from std.sys.info import CompilationTarget, is_triple


struct NativeDepthFormat(ImplicitlyCopyable, Movable):
    var width: Int
    var precision: Int
    var exponent_bits: Int
    var bias: Int
    var minimum_power: Int
    var explicit_integer: Bool

    def __init__(out self, code: String, width: Int) raises:
        self.width = width
        self.explicit_integer = False
        if code == "e" and width == 2:
            self.precision = 11
            self.exponent_bits = 5
        elif code == "f" and width == 4:
            self.precision = 24
            self.exponent_bits = 8
        elif (code == "d" or code == "g") and width == 8:
            self.precision = 53
            self.exponent_bits = 11
        elif code == "g" and width == 16:
            comptime if CompilationTarget.is_x86():
                self.precision = 64
                self.exponent_bits = 15
                self.explicit_integer = True
            elif is_triple["aarch64-unknown-linux-gnu"]():
                self.precision = 113
                self.exponent_bits = 15
            else:
                raise Error("TypeError: fit depth requires a supported native long-double storage ABI")
        else:
            raise Error("TypeError: fit depth requires a native primitive floating format")
        self.bias = (1 << (self.exponent_bits-1))-1
        self.minimum_power = 1-self.bias-self.precision+1

    def write(self, data: MutPointer[UInt8, MutUntrackedOrigin],
              offset: Int, value: NativeRealBits):
        # Native output is little-endian on every currently supported host.
        # x87's explicit integer bit and padding are public ABI fields;
        # padding is initialized and never participates in the arithmetic.
        var low = UInt64(0)
        var high = UInt64(0)
        var fraction = self.precision-1
        var exponent = 0
        var normal = False
        var top = value.highest()
        if value.special != 0:
            exponent = (1 << self.exponent_bits)-1
            normal = True
            if value.special == 2:
                for i in range(fraction):  # driver-loop: retain the native arithmetic NaN payload
                    _depth_bit_write(low,high,i,value.bit(i))
        elif top >= 0:
            var power = top+value.scale
            normal = power >= 1-self.bias
            exponent = power+self.bias if normal else 0
            var source = top-fraction if normal else self.minimum_power-value.scale
            for i in range(fraction):  # driver-loop: encode rounded fraction or subnormal without another conversion
                _depth_bit_write(low,high,i,value.bit(i+source))
        if self.explicit_integer:
            _depth_bit_write(low,high,fraction,UInt64(normal))
            fraction += 1
        for i in range(self.exponent_bits):  # driver-loop: fixed native exponent field
            _depth_bit_write(low,high,fraction+i,UInt64((exponent >> i) & 1))
        _depth_bit_write(low,high,fraction+self.exponent_bits,UInt64(value.negative))
        for i in range(self.width):  # driver-loop: at most sixteen bytes of one primitive component
            var word = low if i < 8 else high
            data[offset+i] = UInt8((word >> UInt64(8*(i%8))) & 255)


def _depth_bit_write(mut low: UInt64, mut high: UInt64, at: Int, value: UInt64):
    if at < 64:
        low |= value << UInt64(at)
    else:
        high |= value << UInt64(at-64)


def _depth_round(bits: List[UInt8], anchor: Int, negative: Bool,
                 precision: Int, minimum_power: Int) -> NativeRealBits:
    # bits[i] is the coefficient of 2**(anchor+i). The normal significand
    # has precision bits; minimum_power is the smallest subnormal power.
    # Round once, nearest/ties-even, retaining all guard/sticky information.
    var highest = -1
    for i in range(len(bits)-1,-1,-1):  # driver-loop: bounded by the admitted binary floating exponent range
        if bits[i] != 0:
            highest = i
            break
    if highest < 0:
        return NativeRealBits(precision=precision,negative=negative)
    var shift = max(highest-precision+1,minimum_power-anchor)
    var low = UInt64(0)
    var high = UInt64(0)
    for i in range(precision):  # driver-loop: at most binary128's 113 significant bits
        var index = i+shift
        if index >= 0 and index < len(bits):
            _depth_bit_write(low,high,i,UInt64(bits[index]))
    var round_up = False
    if shift > 0 and shift-1 < len(bits) and bits[shift-1] != 0:
        var sticky = False
        for i in range(min(shift-1,len(bits))):  # driver-loop: every discarded low bit participates in ties-even
            sticky = sticky or bits[i] != 0
        round_up = sticky or (low & 1) != 0
    if round_up:
        low += 1
        if low == 0:
            high += 1
    # Rounding may add a leading bit. Renormalize while retaining the
    # same exact rounded value, including a subnormal-to-normal carry.
    var result = NativeRealBits(low,high,anchor+shift,precision,negative)
    if result.highest() >= precision:
        result.low = (low >> 1) | (high << 63)
        result.high = high >> 1
        result.scale += 1
    return result


def native_depth_float_cast(value: NativeRealBits, precision: Int,
                            minimum_power: Int) -> NativeRealBits:
    if value.special == 1:
        return NativeRealBits(precision=precision,negative=value.negative,special=1)
    if value.special == 2:
        var low = UInt64(0)
        var high = UInt64(0)
        for i in range(precision-1):  # driver-loop: retain the representable NaN payload on promotion
            _depth_bit_write(low,high,i,value.bit(i+value.precision-precision))
        if precision != value.precision:
            # The format conversion quiets an sNaN before arithmetic. The
            # caller reports the cast's invalid status at its own boundary.
            _depth_bit_write(low,high,precision-2,1)
        return NativeRealBits(low,high,0,precision,value.negative,2)
    var bits = List[UInt8](length=max(value.highest()+1,1),fill=UInt8(0))
    for i in range(len(bits)):  # driver-loop: exact source significand, no float conversion through NumPy
        bits[i] = UInt8(value.bit(i))
    return _depth_round(bits,value.scale,value.negative,precision,minimum_power)


def native_depth_add_one(value: NativeRealBits, subtract: Bool,
                         precision: Int, minimum_power: Int) -> NativeRealBits:
    if value.special != 0:
        var result = value
        if value.special == 2:
            # Arithmetic quiets an sNaN; the caller reports invalid at this
            # exact operation boundary before publishing the result.
            _depth_bit_write(result.low,result.high,precision-2,1)
        return result
    if not value.truth():
        return NativeRealBits(1,0,0,precision,subtract)
    var same_sign = value.negative == subtract
    var absolute = value
    absolute.negative = False
    var equal = absolute.equals_integer(1)
    if not same_sign and equal:
        # Exact cancellation under round-to-nearest produces positive zero.
        return NativeRealBits(precision=precision)
    var source_larger = value.highest()+value.scale >= 0
    var negative = value.negative if same_sign or source_larger else subtract
    var anchor = min(value.scale,0)
    var length = max(value.highest()+value.scale,0)-anchor+2
    var bits = List[UInt8](length=length,fill=UInt8(0))
    var carry = Int(0)
    for i in range(length):  # driver-loop: exact aligned addition/subtraction across the native format's exponent range
        var source = Int(value.bit(i+anchor-value.scale))
        var one = 1 if i+anchor == 0 else 0
        if same_sign:
            var total = source+one+carry
            bits[i] = UInt8(total & 1)
            carry = total >> 1
        else:
            var left = source if source_larger else one
            var right = one if source_larger else source
            var difference = left-right-carry
            bits[i] = UInt8(difference+2 if difference < 0 else difference)
            carry = 1 if difference < 0 else 0
    return _depth_round(bits,anchor,negative,precision,minimum_power)

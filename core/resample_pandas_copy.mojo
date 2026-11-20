# SPDX-License-Identifier: Apache-2.0
# The typed cell conversions of the pandas resample storage
# (providers/python/resample_pandas_pyapi_door.mojo): one cell of a source
# layout converted into a target layout, by native ABI word operations. No
# Python import.
#
# Lane door-run-split (2026-11-14): `_rp_copy` and `_rp_sparse_fill_copy`
# took the door's Python-holding `_RPArray` while reading only its typed
# `layout`; they take the `RPLayout` views here (with the conversions they
# use: `rp_number`, `rp_widen`, `rp_wide_word`, `rp_time_unit`,
# `rp_float_integer_fill`, moved unchanged), and the door keeps only the
# object-target branch (CPython boxing) before calling them. Same word
# operations in the same order: the bits do not move.
from std.memory import bitcast, unsafe_memcpy
from std.sys.info import CompilationTarget
from core.classical_pandas_impl import pd_raw_at, _pd_double
from core.resample_pandas_layout import RPLayout
from core.cast_in import cast_w64_bits, _top_bit
from core.sequence_array_cast import sequence_store_word


def rp_number(source: RPLayout, address: Int, imaginary: Bool = False) raises -> Float64:
    var kind = source.kind
    var size = source.item
    var offset = 0
    if kind == "c":
        kind = "f"
        size //= 2
        if imaginary:
            offset = size
    elif imaginary:
        return Float64(0)
    if size > 8:
        raise Error("TypeError: mixed pandas numeric conversion requires native widths up to float64")
    var tag = 2 if kind == "f" else (1 if kind == "i" else 0)
    return _pd_double(pd_raw_at(address, offset, size, source.native), tag, size)


def rp_wide_word(destination: Int, low: UInt64, high: UInt64, native: Bool):
    var out = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=destination)
    for byte in range(16):  # driver-loop: store the native extended floating representation without a foreign cast
        var word = low if byte < 8 else high
        out[byte if native else 15 - byte] = UInt8(word >> UInt64((byte % 8) * 8))


def rp_widen(source: RPLayout, address: Int, destination: Int, imaginary: Bool, native: Bool) raises:
    if imaginary and source.kind != "c":
        rp_wide_word(destination, 0, 0, native)
        return
    var component = source.item // 2 if source.kind == "c" else source.item
    var src = address + (component if imaginary else 0)
    if component == 16 and (source.kind == "f" or source.kind == "c"):
        var input = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=src)
        var output = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=destination)
        for byte in range(16):  # driver-loop: same-format extended precision is a raw native storage copy
            output[byte] = input[byte if source.native == native else 15 - byte]
        return
    var negative = False
    var exponent = 0
    var significant = UInt64(0)
    var special = 0
    if source.kind == "i" or source.kind == "u" or source.kind == "b":
        var word = cast_w64_bits(pd_raw_at(src, 0, component, source.native), 1 if source.kind == "i" else 0, component)
        negative = source.kind == "i" and (word >> 63) != 0
        var magnitude = (~word + UInt64(1)) if negative else word
        if magnitude != 0:
            exponent = _top_bit(magnitude)
            significant = magnitude << UInt64(63 - exponent)
    else:
        var word = bitcast[DType.uint64](rp_number(source, address, imaginary))
        negative = (word >> 63) != 0
        var encoded = Int((word >> 52) & UInt64(0x7ff))
        var fraction = word & UInt64(0xfffffffffffff)
        if encoded == 2047:
            special = 1 if fraction == 0 else 2
            significant = UInt64(0x8000000000000000) | (fraction << 11)
            if special == 2:
                significant |= UInt64(0x4000000000000000)
        elif encoded == 0:
            if fraction != 0:
                var top = _top_bit(fraction)
                exponent = top - 1074
                significant = fraction << UInt64(63 - top)
        else:
            exponent = encoded - 1023
            significant = (fraction | UInt64(0x10000000000000)) << 11
    var biased = 0 if significant == 0 else (32767 if special != 0 else exponent + 16383)
    var sign = UInt64(1) if negative else UInt64(0)
    comptime if CompilationTarget.is_x86():
        # SysV x86 long double: explicit integer bit, 15-bit exponent, padding.
        rp_wide_word(destination, significant, (sign << 15) | UInt64(biased), native)
    else:
        # AArch64 Linux long double: IEEE binary128. Apple's long double is
        # binary64 and never takes this width-16 branch.
        var low = significant << 49
        var high = (sign << 63) | (UInt64(biased) << 48) | ((significant >> 15) & UInt64(0xffffffffffff))
        rp_wide_word(destination, low, high, native)


def rp_time_unit(code: String) raises -> Int:
    if code.endswith("[s]"):
        return 7
    if code.endswith("[ms]"):
        return 8
    if code.endswith("[us]"):
        return 9
    if code.endswith("[ns]"):
        return 10
    raise Error("TypeError: pandas temporal storage requires seconds, milliseconds, microseconds or nanoseconds")


def rp_float_integer_fill(source: RPLayout, destination: Int, target: RPLayout) raises:
    var negative = False
    var exponent = 0
    var high = UInt64(0)
    var low = UInt64(0)
    var precision = 53
    var nonfinite = False
    if source.item == 16:
        low = pd_raw_at(source.address, 0 if source.native else 8, 8, source.native)
        high = pd_raw_at(source.address, 8 if source.native else 0, 8, source.native)
        comptime if CompilationTarget.is_x86():
            negative = (high & UInt64(0x8000)) != 0
            var encoded = Int(high & UInt64(0x7fff))
            exponent = encoded - 16383
            nonfinite = encoded == 32767
            precision = 64
            high = low
            low = 0
        else:
            negative = (high >> 63) != 0
            var encoded = Int((high >> 48) & UInt64(0x7fff))
            exponent = encoded - 16383
            nonfinite = encoded == 32767
            precision = 113
            high = (high & UInt64(0xffffffffffff)) | UInt64(0x1000000000000)
    else:
        var word = bitcast[DType.uint64](rp_number(source, source.address))
        negative = (word >> 63) != 0
        var encoded = Int((word >> 52) & UInt64(0x7ff))
        exponent = encoded - 1023
        nonfinite = encoded == 2047
        high = (word & UInt64(0xfffffffffffff)) | UInt64(0x10000000000000)
    if target.kind == "b":
        # Floating zero alone is false; NaN and infinity are nonzero.
        var zero = True
        var bytes = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=source.address)
        var meaningful = source.item
        comptime if CompilationTarget.is_x86():
            if meaningful == 16:
                meaningful = 10
        for byte in range(meaningful):  # driver-loop: native floating-to-bool scalar storage conversion
            var value = bytes[byte if source.native else source.item - 1 - byte]
            if byte == meaningful - 1:
                value &= UInt8(0x7f)
            zero = zero and value == 0
        sequence_store_word(destination, UInt64(0) if zero else UInt64(1), 1, target.native)
        return
    if nonfinite:
        raise Error("ValueError: a nonfinite sparse fill cannot be represented in the selected integer dtype")
    if exponent > 63:
        raise Error("OverflowError: sparse floating fill exceeds the selected integer dtype")
    var magnitude = UInt64(0)
    if exponent >= 0:
        if precision == 113:
            magnitude = (high >> UInt64(48 - exponent)) if exponent <= 48 else ((high << UInt64(exponent - 48)) | (low >> UInt64(112 - exponent)))
        elif exponent >= precision - 1:
            magnitude = high << UInt64(exponent - precision + 1)
        else:
            magnitude = high >> UInt64(precision - 1 - exponent)
    var limit = UInt64.MAX if target.item == 8 else (UInt64(1) << UInt64(target.item * 8)) - 1
    if target.kind == "i":
        limit = UInt64(1) << UInt64(target.item * 8 - 1)
        if not negative:
            limit -= 1
    if (negative and magnitude != 0 and target.kind == "u") or magnitude > limit:
        raise Error("OverflowError: sparse floating fill exceeds the selected integer dtype")
    var word = (~magnitude + UInt64(1)) if negative else magnitude
    sequence_store_word(destination, word, target.item, target.native)


def rp_copy_cell(source: RPLayout, address: Int, destination: Int, target: RPLayout) raises:
    """`_rp_copy` of one non-object cell: the word at `address` (source
    layout) converted to the target layout at `destination`. The door
    boxes an object (`O`) target before calling this."""
    if source.code == target.code:
        unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=destination),
               src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address), count=source.item)
        return
    if (target.kind == "M" or target.kind == "m") and source.kind == target.kind:
        var delta = rp_time_unit(target.code) - rp_time_unit(source.code)
        if delta < 0:
            raise Error("ValueError: pandas common temporal dtype must retain the finest resolution")
        var word = pd_raw_at(address, 0, 8, source.native)
        if word != UInt64(1) << 63:
            for i in range(delta):  # driver-loop: storage unit scaling to the common finest temporal dtype
                word *= UInt64(1000)
        sequence_store_word(destination, word, 8, target.native)
        return
    if target.kind == "f" or target.kind == "c":
        var components = 2 if target.kind == "c" else 1
        var size = target.item // components
        for component in range(components):  # driver-loop: native ABI scalar conversion, no NumPy conversion calls
            if size == 16:
                rp_widen(source, address, destination + component * size, component == 1, target.native)
                continue
            var value = rp_number(source, address, component == 1)
            var word = UInt64(0)
            if size == 8:
                word = bitcast[DType.uint64](value)
            elif size == 4:
                word = UInt64(bitcast[DType.uint32](Float32(value)))
            elif size == 2:
                word = UInt64(bitcast[DType.uint16](Float16(value)))
            else:
                raise Error("TypeError: mixed pandas floating widths above float64 need native conversion support")
            sequence_store_word(destination + component * size, word, size, target.native)
        return
    if (target.kind == "i" or target.kind == "u" or target.kind == "b") and (source.kind == "i" or source.kind == "u" or source.kind == "b"):
        var word = cast_w64_bits(pd_raw_at(address, 0, source.item, source.native), 1 if source.kind == "i" else 0, source.item)
        if target.kind == "b":
            word = UInt64(1) if word != 0 else UInt64(0)
        sequence_store_word(destination, word, target.item, target.native)
        return
    raise Error("TypeError: pandas blocks require unsupported native dtype promotion")


def rp_sparse_fill_copy(source: RPLayout, destination: Int, target: RPLayout) raises:
    """`_rp_sparse_fill_copy` into a non-object target (the door boxes an
    object target before calling this)."""
    if source.kind == "f" and (target.kind == "i" or target.kind == "u" or target.kind == "b"):
        rp_float_integer_fill(source, destination, target)
        return
    if (source.kind == "i" or source.kind == "u" or source.kind == "b") and (target.kind == "i" or target.kind == "u"):
        var word = cast_w64_bits(pd_raw_at(source.address, 0, source.item, source.native), 1 if source.kind == "i" else 0, source.item)
        var negative = source.kind == "i" and (word >> 63) != 0
        var magnitude = (~word + UInt64(1)) if negative else word
        var limit = UInt64.MAX if target.item == 8 else (UInt64(1) << UInt64(target.item * 8)) - 1
        if target.kind == "i":
            limit = UInt64(1) << UInt64(target.item * 8 - 1)
            if not negative:
                limit -= 1
        if (negative and target.kind == "u") or magnitude > limit:
            raise Error("OverflowError: pandas sparse fill is outside the selected integer dtype")
    # A scalar's actual unit can be finer than the sparse dtype even though
    # its TYPE in result_type is a generic datetime64/timedelta64 descriptor.
    if (target.kind == "M" or target.kind == "m") and source.kind == target.kind:
        var delta = rp_time_unit(target.code) - rp_time_unit(source.code)
        if delta < 0:
            var value = bitcast[DType.int64](pd_raw_at(source.address, 0, 8, source.native))
            if value != Int64.MIN:
                for i in range(-delta):  # driver-loop: native scalar assignment uses floor when reducing temporal resolution
                    value = value // 1000
            sequence_store_word(destination, bitcast[DType.uint64](value), 8, target.native)
            return
    rp_copy_cell(source, source.address, destination, target)

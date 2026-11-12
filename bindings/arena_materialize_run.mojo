# SPDX-License-Identifier: Apache-2.0
"""RUN of the arena row materializer (bindings/arena_materialize_native.mojo):
dtype tables, the ordered int-conversion refusal and the typed width
conversions of `_numpy_copy`. No Python import (tools/python_free_dirs.sh).
Lane rs-shared-infra-b (2026-11-12): moved word for word; the element loops
are the existing serial conversions of one fold buffer, unchanged (their bits
and refusal order are the contract)."""
from std.math import isinf, isnan
from gemm.f16_widen import f16_bits_to_f32


def _itemsize(dtype: String) raises -> Int:
    if dtype == "<f8" or dtype == "<i8":
        return 8
    if dtype == "<f4" or dtype == "<i4" or dtype == "<u4":
        return 4
    if dtype == "<f2" or dtype == "<u2":
        return 2
    if dtype == "<i1" or dtype == "<u1":
        return 1
    raise Error("TypeError: native row materialization requires a supported numeric Array dtype")


def _cast_code(dtype: String) raises -> Int:
    if dtype == "<f4":
        return 0
    if dtype == "<f8":
        return 1
    if dtype == "<i4":
        return 2
    if dtype == "<i8":
        return 3
    if dtype == "<u4":
        return 4
    if dtype == "<u1":
        return 5
    if dtype == "<u2":
        return 6
    if dtype == "<i1":
        return 7
    if dtype == "<f2":
        return 8
    raise Error("TypeError: native row materialization requires a supported numeric Array dtype")


def _cast_failure(address: Int, source: String, count: Int) raises:
    if source == "<f4" or source == "<f8" or source == "<f2":
        for i in range(count):  # driver-loop: preserve the old ordered int-conversion refusal without a Python reference path
            var number = Float64(0)
            if source == "<f8":
                number = MutPointer[Float64,MutUntrackedOrigin](unsafe_from_address=address)[i]
            elif source == "<f4":
                number = Float64(MutPointer[Float32,MutUntrackedOrigin](unsafe_from_address=address)[i])
            else:
                number = Float64(f16_bits_to_f32(MutPointer[UInt16,MutUntrackedOrigin](unsafe_from_address=address)[i]))
            if isnan(number):
                raise Error("ValueError: cannot convert float NaN to integer")
            if isinf(number):
                raise Error("OverflowError: cannot convert float infinity to integer")
            if number < -9223372036854775808.0 or number >= 9223372036854775808.0:
                raise Error("TypeError: numeric row index exceeds int64 storage")
    raise Error("OverflowError: row index exceeds the target integer range")


def _widen_elements(kind: String, item: Int, address: Int, output: Int, count: Int) raises:
    """`_numpy_copy`'s native-width conversion of `count` elements at `address`
    into `output`: bool to uint8, int16 to int32, uint16 to uint32, uint64 to
    int64 (refusing a word above the int64 range)."""
    for i in range(count):  # driver-loop: existing native-width buffer conversion without NumPy arithmetic
        if kind == "b":
            MutPointer[UInt8,MutUntrackedOrigin](unsafe_from_address=output)[i] = UInt8(MutPointer[UInt8,MutUntrackedOrigin](unsafe_from_address=address)[i] != 0)
        elif kind == "i":
            MutPointer[Int32,MutUntrackedOrigin](unsafe_from_address=output)[i] = Int32(MutPointer[Int16,MutUntrackedOrigin](unsafe_from_address=address)[i])
        elif item == 2:
            MutPointer[UInt32,MutUntrackedOrigin](unsafe_from_address=output)[i] = UInt32(MutPointer[UInt16,MutUntrackedOrigin](unsafe_from_address=address)[i])
        else:
            var word = MutPointer[UInt64,MutUntrackedOrigin](unsafe_from_address=address)[i]
            if word > UInt64(9223372036854775807):
                raise Error("OverflowError: unsigned fold data exceeds int64 storage")
            MutPointer[Int64,MutUntrackedOrigin](unsafe_from_address=output)[i] = Int64(word)


def _canonical_half_nans(address: Int, count: Int):
    """`_numpy_copy`'s strided half path: canonicalize the NaN payloads of the
    `count` float16 words at `address` as the struct codec did."""
    var words = MutPointer[UInt16,MutUntrackedOrigin](unsafe_from_address=address)
    for i in range(count):  # driver-loop: preserve the prior half storage conversion's NaN bits
        if (words[i] & 0x7FFF) > 0x7C00:
            words[i] = (words[i] & 0x8000) | 0x7E00

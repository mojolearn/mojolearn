# SPDX-License-Identifier: Apache-2.0
"""Typed parts of the NumPy string storage boundary (RUN): the validated (capsule, table) record,
the dladdr owner lookup of a storage entry, and the NaN classification of raw float words.
The capsule admission, allocator and pack calls stay in `bindings/sequence_string_storage_door.mojo`."""
from std.ffi import c_int, external_call
from std.sys.info import CompilationTarget
from core.shared_slot import SharedSlot

comptime StringStoragePointer = MutPointer[UInt8, MutUntrackedOrigin]


struct SequenceStringTableRecord(Defaultable, Movable):
    # p3-array-arena: the (capsule, table) pair whose dladdr owner checks and
    # API version check passed. Both are process constants of one loaded
    # NumPy extension; the capsule is retained once so its address cannot
    # recycle. A different capsule or table re-runs every check.
    var capsule: Int
    var table: Int

    def __init__(out self):
        self.capsule = 0
        self.table = 0


def sequence_storage_image(address: Int) raises -> Int:
    # dladdr is the supported POSIX owner lookup. Check the capsule table and
    # each function before dereferencing/calling it; a foreign capsule or a
    # Python callback trampoline is not a native NumPy storage entry.
    if address == 0:
        raise Error("TypeError: NumPy storage entry must not be null")
    var info = Array[Int, 4](uninitialized=True)
    var found = external_call["dladdr", c_int](StringStoragePointer(unsafe_from_address=address), info.unsafe_ptr())
    if found == 0:
        raise Error("TypeError: NumPy storage pointers require their native extension owner")
    return info[1]


def sequence_string_nan_word(word: UInt64, exponent: UInt64, fraction: UInt64, quiet: UInt64) -> Int:
    if (word & exponent) == exponent and (word & fraction) != 0:
        return 1 | (4 if (word & quiet) == 0 else 0)
    return 0


def sequence_string_nan_raw(address: Int, width: Int) -> Int:
    if width == 2:
        var word = UInt64(MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=address)[0])
        return sequence_string_nan_word(word, 0x7C00, 0x03FF, 0x0200)
    if width == 4:
        var word = UInt64(MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=address)[0])
        return sequence_string_nan_word(word, 0x7F800000, 0x007FFFFF, 0x00400000)
    var low = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=address)[0]
    if width == 8:
        return sequence_string_nan_word(low, 0x7FF0000000000000, 0x000FFFFFFFFFFFFF, 0x0008000000000000)
    comptime if CompilationTarget.is_x86():
        var exponent = MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=address + 8)[0] & UInt16(0x7FFF)
        if exponent == UInt16(0x7FFF) and (low & UInt64(0x7FFFFFFFFFFFFFFF)) != 0:
            return 1 | (4 if (low & UInt64(0x4000000000000000)) == 0 else 0)
    else:
        var high = MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=address + 8)[0]
        if (high & UInt64(0x7FFF000000000000)) == UInt64(0x7FFF000000000000) and ((high & UInt64(0x0000FFFFFFFFFFFF)) != 0 or low != 0):
            return 1 | (4 if (high & UInt64(0x0000800000000000)) == 0 else 0)
    return 0


comptime SEQUENCE_STRING_TABLE = SharedSlot[SequenceStringTableRecord, "MojolearnP3SequenceStringTableV2"]

# SPDX-License-Identifier: Apache-2.0
"""Native label-code ownership, independent of public Python class objects.

Independent owners execute across host cores. These are allocation/lifetime
contracts; class ordering and decoding numerical tests remain in their existing
native suites and public Python interoperability tests.
"""
from core.host_parallel import host_parallelize
from std.memory import bitcast
from core.native_label_codes import NativeLabelCodes
from core.gp_native_label_table import GPLabelTable, GP_LABEL_OBJECT, GP_LABEL_I64, GP_LABEL_F64
from core.labels_native_api import LabelsNativeDriver, LabelsGatherRequest
from core.labels_native_host_provider import labels_host_native_v2
from checks.native_tests.gp_state_suite import require

comptime GP_LABEL_OWNER_FIXTURES = 28


def returned_codes() -> NativeLabelCodes:
    var words: List[Int32] = [-2147483648, -1, 0, 1, 2147483647]
    return NativeLabelCodes(words^)


def returned_copy() -> NativeLabelCodes:
    var original = returned_codes()
    var copied = original.copy()
    original.values[0] = 91
    _ = original^
    return copied^


def exact_codes(owner: NativeLabelCodes) raises:
    require(owner.count() == 5 and owner.address() == Int(owner.values.unsafe_ptr())
        and owner.values[0] == Int32(-2147483648) and owner.values[1] == Int32(-1)
        and owner.values[2] == Int32(0) and owner.values[3] == Int32(1)
        and owner.values[4] == Int32(2147483647), "GP native code owner retains full signed Int32 words")
    _ = owner


def fixture(unit: Int) raises:
    if unit < 4:
        var count = 0 if unit == 0 else (1 if unit == 1 else (17 if unit == 2 else 257))
        var owner = NativeLabelCodes(count)
        require(owner.count() == count and len(owner.values) == count, "GP native code logical/storage extent")
        if count != 0:
            require(owner.values[0] == Int32(0) and owner.values[count - 1] == Int32(0),
                "GP native code allocation initializes both endpoints")
        _ = owner^
    elif unit == 4:
        var words: List[Int32] = [-2147483648, -1, 0, 1, 2147483647]
        var address = Int(words.unsafe_ptr())
        var owner = NativeLabelCodes(words^)
        require(owner.address() == address, "GP native code adoption does not copy allocation")
        exact_codes(owner)
        _ = owner^
    elif unit == 5:
        var owner = returned_codes()
        var copied = owner.copy()
        require(copied.address() != owner.address(), "GP native code copy owns independent storage")
        owner.values[0] = 71
        owner.values[4] = 73
        exact_codes(copied)
        require(owner.values[0] == Int32(71) and owner.values[4] == Int32(73), "GP source copy stays writable")
        _ = owner^
        _ = copied^
    elif unit == 6:
        var owner = returned_codes()
        var address = owner.address()
        var moved = owner^
        require(moved.address() == address, "GP moved native codes preserve allocation")
        exact_codes(moved)
        _ = moved^
    elif unit == 7:
        # The callee's original owner has been consumed and destroyed. The
        # returned deep copy must keep independent storage and original words.
        var owner = returned_copy()
        exact_codes(owner)
        _ = owner^
    elif unit == 8 or unit == 9:
        var count = -1 if unit == 8 else -9223372036854775807
        var refused = False
        try:
            _ = NativeLabelCodes(count)
        except error:
            require(String(error) == "native label codes require a nonnegative count", "GP native code negative count refusal")
            refused = True
        require(refused, "GP negative code count refuses before allocation")
    else:
        var empty = List[Int32]()
        var owner = NativeLabelCodes(empty^)
        if unit == 10:
            require(owner.count() == 0 and len(owner.values) == 0, "GP empty code list adoption")
        else:
            var copied = owner.copy()
            var moved = copied^
            require(moved.count() == 0 and owner.count() == 0, "GP empty code copy and move")
            _ = moved^
        _ = owner^


def populate(mut table: GPLabelTable):
    if table.kind == GP_LABEL_I64:
        table.integers[0] = -9223372036854775808
        table.integers[1] = 0
        table.integers[2] = 9007199254740993
        table.integers[3] = -1
        table.integers[4] = 9223372036854775807
    else:
        table.reals[0] = bitcast[DType.float64](UInt64(0x8000000000000000))
        table.reals[1] = 0.0
        table.reals[2] = 1.0000000000000002
        table.reals[3] = bitcast[DType.float64](UInt64(0x7ff8000000000042))
        table.reals[4] = bitcast[DType.float64](UInt64(0x7ff0000000000000))


def table_words(table: GPLabelTable) raises:
    if table.kind == GP_LABEL_I64:
        require(table.integers[0] == Int64(-9223372036854775808) and table.integers[1] == Int64(0)
            and table.integers[2] == Int64(9007199254740993) and table.integers[3] == Int64(-1)
            and table.integers[4] == Int64(9223372036854775807), "GP native class table exact Int64 words")
    else:
        require(bitcast[DType.uint64](table.reals[0]) == UInt64(0x8000000000000000)
            and bitcast[DType.uint64](table.reals[1]) == UInt64(0)
            and bitcast[DType.uint64](table.reals[2]) == UInt64(0x3ff0000000000001)
            and bitcast[DType.uint64](table.reals[3]) == UInt64(0x7ff8000000000042)
            and bitcast[DType.uint64](table.reals[4]) == UInt64(0x7ff0000000000000),
            "GP native class table retains binary64 words without coercion")
    _ = table


def gather_table(kind: Int) raises:
    var table = GPLabelTable(kind, 5)
    populate(table)
    var address = table.address()
    var moved = table^
    var codes: List[Int64] = [4, 0, 4, 2, 1, 3]
    var destination = List[UInt64](length=8, fill=0xdeadbeef)
    var expected: List[UInt64] = [0x7fffffffffffffff, 0x8000000000000000, 0x7fffffffffffffff,
        0x0020000000000001, 0, 0xffffffffffffffff]
    if kind == GP_LABEL_F64:
        expected = [0x7ff0000000000000, 0x8000000000000000, 0x7ff0000000000000,
            0x3ff0000000000001, 0, 0x7ff8000000000042]
    try:
        require(moved.address() == address, "GP gather borrows moved native table allocation")
        LabelsNativeDriver(labels_host_native_v2).gather(kind,
            LabelsGatherRequest(moved.address(), Int(codes.unsafe_ptr()), Int(destination.unsafe_ptr() + 1), 5, 6))
        var failures = List[Int](length=6, fill=0)
        var fp = failures.unsafe_ptr()
        var dp = destination.unsafe_ptr() + 1
        var ep = expected.unsafe_ptr()
        def cell(index: Int) {imm fp, imm dp, imm ep}:
            fp.unsafe_store(index, Int(dp[index] != ep[index]))
        host_parallelize(cell, 6)
        var failed = 0
        for value in failures:
            failed += value
        require(failed == 0, "GP moved table actual native gather preserves fixed order, duplicates and exact words")
        require(destination[0] == UInt64(0xdeadbeef) and destination[7] == UInt64(0xdeadbeef), "GP native table gather guards")
        table_words(moved)
    finally:
        _ = moved^
        _ = codes^
        _ = destination^
        _ = expected^


def table_case(unit: Int) raises:
    if unit < 6:
        var kind = unit % 3
        var count = 0 if unit < 3 else 5
        var table = GPLabelTable(kind, count)
        require(table.kind == kind and table.available == (kind != GP_LABEL_OBJECT and count != 0)
            and len(table.integers) == (count if kind == GP_LABEL_I64 else 0)
            and len(table.reals) == (count if kind == GP_LABEL_F64 else 0), "GP table kind/extent and object fallback")
        if kind == GP_LABEL_OBJECT or count == 0:
            require(table.address() == 0, "GP unavailable table publishes no numeric address")
        elif kind == GP_LABEL_I64:
            require(table.address() == Int(table.integers.unsafe_ptr()) and table.integers[0] == Int64(0)
                and table.integers[4] == Int64(0), "GP initialized Int64 table span")
        else:
            require(table.address() == Int(table.reals.unsafe_ptr()) and table.reals[0] == 0
                and table.reals[4] == 0, "GP initialized Float64 table span")
        _ = table^
    elif unit < 10:
        var kind = GP_LABEL_I64 if unit % 2 == 0 else GP_LABEL_F64
        var table = GPLabelTable(kind, 5)
        populate(table)
        var address = table.address()
        if unit >= 8:
            table.refuse_numeric()
            require(not table.available and table.address() == 0, "GP arbitrary-width refusal disables numeric address")
            table.refuse_numeric()
            table_words(table)
        var moved = table^
        require(moved.address() == (address if unit < 8 else 0), "GP moved table preserves enabled or disabled state")
        require((Int(moved.integers.unsafe_ptr()) if kind == GP_LABEL_I64 else Int(moved.reals.unsafe_ptr())) == address,
            "GP refusal/move retain allocation for safe native owner cleanup")
        table_words(moved)
        _ = moved^
    elif unit < 14:
        var kind = -1 if unit == 10 else (3 if unit == 11 else (GP_LABEL_OBJECT if unit == 12 else GP_LABEL_I64))
        var count = 5 if unit < 12 else -1
        var refused = False
        try:
            _ = GPLabelTable(kind, count)
        except error:
            require(String(error) == "GP label table: invalid count or kind", "GP native table invalid kind/count error")
            refused = True
        require(refused, "GP invalid native table refused before allocation")
    else:
        gather_table(GP_LABEL_I64 if unit == 14 else GP_LABEL_F64)


def run_checks() raises:
    var failures = List[Int](length=GP_LABEL_OWNER_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            if unit < 12:
                fixture(unit)
            else:
                table_case(unit - 12)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL gp_label_owner fixture", unit, String(error))
    host_parallelize(task, GP_LABEL_OWNER_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "gp_label_owner failed " + String(failed) + " fixtures")
    print("PASS gp_label_owner:", GP_LABEL_OWNER_FIXTURES, "native code ownership fixtures")

# SPDX-License-Identifier: Apache-2.0
"""Native label-code ownership, independent of public Python class objects.

Independent owners execute across host cores. These are allocation/lifetime
contracts; class ordering and decoding numerical tests remain in their existing
native suites and public Python interoperability tests.
"""
from core.host_parallel import host_parallelize
from core.native_label_codes import NativeLabelCodes
from checks.native_tests.gp_state_suite import require

comptime GP_LABEL_OWNER_FIXTURES = 12


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


def run_checks() raises:
    var failures = List[Int](length=GP_LABEL_OWNER_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL gp_label_owner fixture", unit, String(error))
    host_parallelize(task, GP_LABEL_OWNER_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "gp_label_owner failed " + String(failed) + " fixtures")
    print("PASS gp_label_owner:", GP_LABEL_OWNER_FIXTURES, "native code ownership fixtures")

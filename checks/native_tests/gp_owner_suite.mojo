# SPDX-License-Identifier: Apache-2.0
"""Independent lease identities and query scratch owners across host cores.

No fake lease reaches a provider. Scratch fields are never captured by a
parallel closure; each task owns its complete state until its final check.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.gp_native_state import GPLeaseState
from core.gp_native_storage import GPCQueryScratch
from checks.native_tests.gp_state_suite import require

comptime GP_OWNER_FIXTURES = 32


def lease_case(unit: Int) raises:
    var owners: List[Int] = [9007199254740993, 17, 9007199254741001]
    var expected = owners.copy()
    var address = Int(owners.unsafe_ptr())
    var state = GPLeaseState(9007199254741011, 23, 31, owners^)
    require(Int(state.owners.unsafe_ptr()) == address, "GP lease adopts native identity storage")
    if unit == 0:
        require(state.matches(23, 31, expected), "GP lease exact identities admitted")
    elif unit == 1:
        require(not state.matches(24, 31, expected), "GP lease provider replacement invalidates")
    elif unit == 2:
        require(not state.matches(23, 32, expected), "GP lease generation replacement invalidates")
    elif unit == 3:
        expected.append(29)
        require(not state.matches(23, 31, expected), "GP lease owner count mismatch")
    elif unit == 4:
        expected[0] = 9007199254740992
        require(not state.matches(23, 31, expected), "GP owner identity retains low bit above binary64 integer precision")
    elif unit == 5:
        expected[0] = 17
        expected[1] = 9007199254740993
        require(not state.matches(23, 31, expected), "GP lease owner ordering is significant")
    elif unit == 6:
        require(not state.matches_owner(-1, 17) and not state.matches_owner(3, 17)
            and state.matches_owner(2, 9007199254741001), "GP lease bounds and final owner")
    elif unit == 7:
        require(state.take() == 9007199254741011 and state.take() == 0
            and not state.matches_header(23, 31, 3), "GP lease exact handle consumed once and invalidates header")
    elif unit == 8:
        var moved = state^
        require(Int(moved.owners.unsafe_ptr()) == address and moved.matches(23, 31, expected)
            and moved.take() == 9007199254741011 and moved.take() == 0,
            "GP moved lease retains allocation and take-once handle")
        _ = moved^
        return
    elif unit == 9 or unit == 10:
        state.handle = 0 if unit == 9 else -1
        require(not state.matches_header(23, 31, 3) and not state.matches(23, 31, expected),
            "GP nonpositive lease handle never matches")
    else:
        var empty = List[Int]()
        var empty_expected = List[Int]()
        var minimal = GPLeaseState(1, 0, 0, empty^)
        require(minimal.matches(0, 0, empty_expected) and minimal.take() == 1
            and not minimal.matches(0, 0, empty_expected), "GP empty identity set and smallest positive handle")
        _ = minimal^
    _ = state^
    _ = expected^


def scratch_case(unit: Int) raises:
    var mask = unit % 8
    var rows = 0 if unit < 8 else 17
    var scratch = GPCQueryScratch(rows, mask)
    var mean_count = 0 if (mask & 1) != 0 else rows
    var variance_count = 0 if (mask & 2) != 0 else rows
    var probability_count = 0 if (mask & 4) != 0 else rows
    require(len(scratch.mean) == mean_count and len(scratch.variance) == variance_count
        and len(scratch.probability) == probability_count, "GP external masks omit exactly their owned spans")
    var mean_address = Int(scratch.mean.unsafe_ptr())
    var variance_address = Int(scratch.variance.unsafe_ptr())
    var probability_address = Int(scratch.probability.unsafe_ptr())
    if mean_count != 0:
        require(scratch.mean[0] == 0 and scratch.mean[16] == 0, "GP mean scratch initialized")
        scratch.mean[16] = -3.5
    if variance_count != 0:
        require(scratch.variance[0] == 0 and scratch.variance[16] == 0, "GP variance scratch initialized")
        scratch.variance[16] = 7.25
    if probability_count != 0:
        require(scratch.probability[0] == 0 and scratch.probability[16] == 0, "GP probability scratch initialized")
        scratch.probability[16] = 1.0000000000000002
    var moved = scratch^
    require(Int(moved.mean.unsafe_ptr()) == mean_address and Int(moved.variance.unsafe_ptr()) == variance_address
        and Int(moved.probability.unsafe_ptr()) == probability_address, "GP scratch move retains storage")
    if mean_count != 0:
        require(moved.mean[16] == Float32(-3.5), "GP mean span remains independent after move")
    if variance_count != 0:
        require(moved.variance[16] == Float32(7.25), "GP variance span remains independent after move")
    if probability_count != 0:
        require(bitcast[DType.uint64](moved.probability[16]) == UInt64(0x3ff0000000000001),
            "GP probability retains binary64 low bit after move")
    _ = moved^


def refusal_case(unit: Int) raises:
    var rows = -1 if unit == 0 else (17 if unit == 3 else 0)
    var mask = -1 if unit == 1 else (8 if unit == 2 else (255 if unit == 3 else 0))
    var refused = False
    try:
        _ = GPCQueryScratch(rows, mask)
    except error:
        require(String(error) == "GP query scratch: invalid row count or external span mask", "GP scratch refusal")
        refused = True
    require(refused, "GP invalid scratch shape/mask refused")


def run_checks() raises:
    var failures = List[Int](length=GP_OWNER_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            if unit < 12:
                lease_case(unit)
            elif unit < 28:
                scratch_case(unit - 12)
            else:
                refusal_case(unit - 28)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL gp_owner fixture", unit, String(error))
    host_parallelize(task, GP_OWNER_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "gp_owner failed " + String(failed) + " fixtures")
    print("PASS gp_owner:", GP_OWNER_FIXTURES, "native lease/query ownership fixtures")

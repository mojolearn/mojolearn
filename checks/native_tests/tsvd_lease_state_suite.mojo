# SPDX-License-Identifier: Apache-2.0
"""Native TSVD cache matching, move ownership and single-release token state."""
from core.host_parallel import host_parallelize
from core.tsvd_native_api import TSVDNativeDriver
from core.tsvd_owner_state import TSVDLeaseState
from checks.native_tests.tsvd_contract_suite import tsvd_wire_entry

comptime TSVD_LEASE_STATE_FIXTURES = 12


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def owner(lease: Int = 7331) -> TSVDLeaseState:
    return TSVDLeaseState(101, 211, 307, lease, TSVDNativeDriver(tsvd_wire_entry))


def transfer(var state: TSVDLeaseState) -> TSVDLeaseState:
    return state^


def fixture(unit: Int) raises:
    var state = owner(0 if unit == 8 else 7331)
    if unit < 8:
        var connection = 103 if (unit & 1) != 0 else 101
        var generation = 223 if (unit & 2) != 0 else 211
        var components = 311 if (unit & 4) != 0 else 307
        require(state.matches(connection, generation, components) == (unit == 0),
                "TSVD live owner requires exact connection/generation/component identity")
        require(state.lease == 7331, "TSVD identity lookup must not consume the lease")
    elif unit == 8:
        require(not state.matches(101, 211, 307), "TSVD zero lease cannot admit cached state")
        require(state.take_lease() == 0 and state.take_lease() == 0,
                "TSVD empty owner stays empty across cleanup attempts")
    elif unit == 9:
        var taken = state.take_lease()
        require(taken == 7331 and state.lease == 0, "TSVD cleanup receives exact token and clears owner")
        require(not state.matches(101, 211, 307) and state.take_lease() == 0,
                "TSVD consumed owner cannot match or release a second token")
        state.driver.close(taken)
    elif unit == 10:
        var moved = transfer(state^)
        require(moved.matches(101, 211, 307) and moved.lease == 7331,
                "TSVD moved owner retains its admitted identities")
        require(moved.driver.metadata(1) == 4 and moved.driver.metadata(2) == 63,
                "TSVD moved owner retains its native descriptor driver")
        var taken = moved.take_lease()
        require(taken == 7331 and moved.take_lease() == 0, "TSVD moved owner releases once")
        moved.driver.close(taken)
    else:
        _ = state.take_lease()
        require(state.connection == 101 and state.generation == 211 and state.components == 307,
                "TSVD token transfer changes no admission identity field")
        require(state.driver.metadata(3) == 2, "TSVD token transfer preserves the owning driver")
        var replacement = TSVDLeaseState(101, 223, 311, 8123, TSVDNativeDriver(tsvd_wire_entry))
        require(replacement.matches(101, 223, 311) and not replacement.matches(101, 211, 307),
                "TSVD replacement generation admits only its current components")
        require(state.take_lease() == 0 and replacement.take_lease() == 8123,
                "TSVD previous cleanup cannot consume a replacement owner's token")


def run_checks() raises:
    var failures = List[Int](length=TSVD_LEASE_STATE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL tsvd_lease_state fixture", unit, String(error))
    host_parallelize(task, TSVD_LEASE_STATE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "tsvd_lease_state failed " + String(failed) + " fixtures")
    print("PASS tsvd_lease_state:", TSVD_LEASE_STATE_FIXTURES, "native matching/move/single-release fixtures")

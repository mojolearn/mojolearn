# SPDX-License-Identifier: Apache-2.0
"""Independent host row-gather fixtures run across host cores."""
from core.host_parallel import host_parallelize
from checks.native_tests.core_helpers_cases import (
    CORE_HELPERS_FIXTURES, GatherFixture, require, gather_fixture, check_gather, OUT_ROWS, SOURCE_ROWS, WIDTH,
)
from core.host_helpers_impl import gather_rows_bytes


def gather_host(mut f: GatherFixture) -> Bool:
    """The host walk; True when it reports an index out of bounds."""
    return gather_rows_bytes(
        MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(f.source.unsafe_ptr())),
        MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(f.dst.unsafe_ptr())),
        MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(f.idx.unsafe_ptr())),
        SOURCE_ROWS, OUT_ROWS, WIDTH,
    )


def run_checks() raises:
    var failures = List[Int](length=CORE_HELPERS_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            var f = gather_fixture(unit)
            var bad = gather_host(f)
            check_gather(unit, f, bad)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL core_helpers_cpu fixture", unit, String(error))
    host_parallelize(task, CORE_HELPERS_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "core_helpers_cpu failed " + String(failed) + " fixtures")
    print("PASS core_helpers_cpu:", CORE_HELPERS_FIXTURES, "row-gather fixtures, parallel")

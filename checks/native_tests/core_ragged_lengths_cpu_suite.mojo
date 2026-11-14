# SPDX-License-Identifier: Apache-2.0
"""The host ragged passes; the four independent fixtures run across host
cores, each with its own buffers and failure slot."""
from checks.native_tests.core_ragged_lengths_cases import (
    RAGGED_FIXTURES, require, ragged_fixture, check_ragged,
)
from core.array_helpers_impl import ragged_rows_bytes
from core.host_parallel import host_parallelize


def check_unit(unit: Int) raises:
    var f = ragged_fixture(unit)
    var dst = Int(f.dst.unsafe_ptr())
    var src = Int(f.src.unsafe_ptr()) if f.mode != 1 else dst
    ragged_rows_bytes(
        MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dst),
        MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=src),
        MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(f.lens.unsafe_ptr())),
        f.n, f.row, f.pos, f.mode,
    )
    check_ragged(unit, f)


def run_checks() raises:
    var failures = List[Int](length=RAGGED_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            check_unit(unit)
        except error:
            print("FAIL core_ragged_lengths_cpu fixture", unit, String(error))
            fp.unsafe_store(unit, 1)
    host_parallelize(task, RAGGED_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "core_ragged_lengths_cpu failed " + String(failed) + " of " + String(RAGGED_FIXTURES))
    print("PASS core_ragged_lengths_cpu:", RAGGED_FIXTURES, "ragged byte-pass fixtures, parallel")

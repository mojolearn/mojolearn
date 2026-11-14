# SPDX-License-Identifier: Apache-2.0
"""The cross-vendor ordered fold on the host column's parallel base entry
(the `_mojolearn_core_host` publication), its independent fixtures across
host cores, each with a private failure slot."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.parallel_entry_table import PARALLEL_TABLE_BASE, ParallelEntryKey, parallel_ref
from bindings.parallel_base_entry_host import publish_parallel_base_host
from checks.native_tests.resample_cross_vendor_fold_cases import (
    CROSS_VENDOR_FOLD_FIXTURES, require, run_fold_fixture,
)


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    publish_parallel_base_host()
    var entry = parallel_ref(PARALLEL_TABLE_BASE, ParallelEntryKey(Int(GLOBAL_NUMERIC_MODE), 0))
    var failures = List[Int](length=CROSS_VENDOR_FOLD_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp, imm entry}:
        try:
            run_fold_fixture(entry, unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL resample_cross_vendor_fold_cpu fixture", unit, String(error))
    host_parallelize(task, CROSS_VENDOR_FOLD_FIXTURES)
    var failed = 0
    for value in failures:  # collect independent fixture status words
        failed += value
    _ = failures
    require(failed == 0, "resample_cross_vendor_fold_cpu failed " + String(failed) + " fixtures")
    print("PASS resample_cross_vendor_fold_cpu:", CROSS_VENDOR_FOLD_FIXTURES, "ordered fold contracts")

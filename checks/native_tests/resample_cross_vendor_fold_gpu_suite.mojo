# SPDX-License-Identifier: Apache-2.0
"""The cross-vendor ordered fold on the GPU image's parallel base entry
(`publish_parallel_base_device`, the compiled vendor's key). Fixtures run in
order: every fold is one launch on the one shared process device context,
and the kernel keeps its own parallelism."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.parallel_entry_table import PARALLEL_TABLE_BASE, ParallelEntryKey, parallel_entry_vendor, parallel_ref
from bindings.parallel_base_entry_device import publish_parallel_base_device
from checks.native_tests.resample_cross_vendor_fold_cases import (
    CROSS_VENDOR_FOLD_FIXTURES, require, run_fold_fixture,
)


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var vendor = parallel_entry_vendor(COMPILED_VENDOR)
    require(vendor >= 1, "resample_cross_vendor_fold_gpu: this build has no GPU vendor")
    publish_parallel_base_device()
    var entry = parallel_ref(PARALLEL_TABLE_BASE, ParallelEntryKey(Int(GLOBAL_NUMERIC_MODE), vendor))
    var failed = 0
    for unit in range(CROSS_VENDOR_FOLD_FIXTURES):  # ordered launches on the one shared device context
        try:
            run_fold_fixture(entry, unit)
        except error:
            failed += 1
            print("FAIL resample_cross_vendor_fold_gpu fixture", unit, String(error))
    require(failed == 0, "resample_cross_vendor_fold_gpu failed " + String(failed) + " fixtures")
    print("PASS resample_cross_vendor_fold_gpu:", CROSS_VENDOR_FOLD_FIXTURES, "ordered fold contracts")

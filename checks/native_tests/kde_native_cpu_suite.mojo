# SPDX-License-Identifier: Apache-2.0
"""Independent typed host KDE profiles run across host cores."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.kde_native_api import KDENativeDriver, kde_native_v1
from core.kde_native_host import KDEHostKernels
from checks.native_tests.kde_native_cases import KDE_NATIVE_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=KDE_NATIVE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(KDENativeDriver(kde_native_v1[KDEHostKernels]), unit, True)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL kde_native_cpu fixture", unit, String(error))
    host_parallelize(task, KDE_NATIVE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "kde_native_cpu failed " + String(failed) + " fixtures")
    print("PASS kde_native_cpu:", KDE_NATIVE_FIXTURES, "native density/total fixtures")

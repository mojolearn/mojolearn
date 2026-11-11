# SPDX-License-Identifier: Apache-2.0
"""Typed GPU TSVD queries and actual native resident lease ownership.

Fixture graphs submit in order because production contexts/store are shared;
existing GPU kernels and independent oracle cells retain parallel execution.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.tsvd_native_api import TSVDNativeDriver
from core.tsvd_native_device import tsvd_device_native_v1
from checks.native_tests.tsvd_native_cases import TSVD_NATIVE_FIXTURES, fixture, negative_lease, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    var failed = 0
    for unit in range(TSVD_NATIVE_FIXTURES + 1):
        try:
            var driver = TSVDNativeDriver(tsvd_device_native_v1)
            if unit < TSVD_NATIVE_FIXTURES:
                fixture(driver, unit, False)
            else:
                negative_lease(driver)
        except error:
            failed += 1
            print("FAIL tsvd_native_gpu fixture", unit, String(error))
    require(failed == 0, "tsvd_native_gpu failed " + String(failed) + " fixtures")
    print("PASS tsvd_native_gpu:", TSVD_NATIVE_FIXTURES + 1, "native query/state/lease/refusal fixtures")

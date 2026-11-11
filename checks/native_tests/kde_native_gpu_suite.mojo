# SPDX-License-Identifier: Apache-2.0
"""Typed KDE graphs run one at a time on one physical GPU/shared contexts.

Existing kernels and independent oracle cells stay parallel. This ordered
numerical suite does not clear recorded shared-context concurrency failures.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.kde_native_api import KDENativeDriver, kde_native_v1
from core.kde_native_device import KDEDeviceKernels
from checks.native_tests.kde_native_cases import KDE_NATIVE_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    var failed = 0
    for unit in range(KDE_NATIVE_FIXTURES):
        try:
            fixture(KDENativeDriver(kde_native_v1[KDEDeviceKernels]), unit, False)
        except error:
            failed += 1
            print("FAIL kde_native_gpu fixture", unit, String(error))
    require(failed == 0, "kde_native_gpu failed " + String(failed) + " fixtures")
    print("PASS kde_native_gpu:", KDE_NATIVE_FIXTURES, "native direct/resident/reuse/total fixtures")

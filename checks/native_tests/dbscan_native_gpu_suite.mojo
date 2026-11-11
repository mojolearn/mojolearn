# SPDX-License-Identifier: Apache-2.0
"""Typed DBSCAN graphs run one at a time on one physical GPU/shared contexts.

Existing kernels and independent oracle cells stay parallel. This ordered
numerical suite does not clear recorded shared-context concurrency failures.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.dbscan_native_api import DBSCANNativeDriver, dbscan_native_v1
from core.dbscan_native_device import DBSCANDeviceKernels
from checks.native_tests.dbscan_native_cases import DBSCAN_NATIVE_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    var failed = 0
    for unit in range(DBSCAN_NATIVE_FIXTURES):
        try:
            fixture(DBSCANNativeDriver(dbscan_native_v1[DBSCANDeviceKernels]), unit, False)
        except error:
            failed += 1
            print("FAIL dbscan_native_gpu fixture", unit, String(error))
    require(failed == 0, "dbscan_native_gpu failed " + String(failed) + " fixtures")
    print("PASS dbscan_native_gpu:", DBSCAN_NATIVE_FIXTURES, "native fit/core/direct/resident query fixtures")

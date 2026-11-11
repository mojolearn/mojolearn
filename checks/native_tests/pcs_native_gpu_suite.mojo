# SPDX-License-Identifier: Apache-2.0
"""PCS graphs run one at a time on one physical GPU with shared contexts.

Kernels and mathematical oracle cells remain parallel. Ordered numerical
coverage does not clear recorded independent shared-context concurrency failures.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.neighbors_native_driver import XnDriver
from core.pcs_native_api import PCSNativeDriver
from x_neighbors.native_descriptor_device import xn_native_v1
from checks.native_tests.pcs_native_cases import PCS_NATIVE_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    var failed = 0
    for unit in range(PCS_NATIVE_FIXTURES):
        try:
            fixture(PCSNativeDriver(XnDriver(xn_native_v1)), unit)
        except error:
            failed += 1
            print("FAIL pcs_native_gpu fixture", unit, String(error))
    require(failed == 0, "pcs_native_gpu failed " + String(failed) + " fixtures")
    print("PASS pcs_native_gpu:", PCS_NATIVE_FIXTURES, "native draw/sketch numerical fixtures")

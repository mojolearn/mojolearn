# SPDX-License-Identifier: Apache-2.0
"""Same LOF numerical graphs, ordered on the one physical GPU.

Only one GPU job runs at a time. Each graph submits through shared native
DeviceContexts, so graphs execute in order; existing kernels and independent
oracle cells remain parallel. This obtains numerical/refusal evidence only.
The separately recorded concurrent shared-context failure remains unresolved.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.neighbors_native_driver import XnDriver
from x_neighbors.native_descriptor_device import xn_native_v1
from checks.native_tests.lof_native_cases import LOF_NATIVE_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    var failed = 0
    for unit in range(LOF_NATIVE_FIXTURES):
        try:
            fixture(XnDriver(xn_native_v1), unit)
        except error:
            failed += 1
            print("FAIL lof_native_gpu fixture", unit, String(error))
    require(failed == 0, "lof_native_gpu failed " + String(failed) + " fixtures")
    print("PASS lof_native_gpu:", LOF_NATIVE_FIXTURES, "ordered native numerical/refusal fixtures; concurrency not tested")

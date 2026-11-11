# SPDX-License-Identifier: Apache-2.0
"""Existing NC numerical graphs, ordered on the one physical GPU.

Only one GPU job runs at a time. Each graph uses the shared DeviceContext, so
submissions are ordered here as in the PCA/TSVD numerical suites; kernels
retain their parallel execution. The separate parallel run_checks and device
concurrency fixtures remain unchanged, with their Metal failures still open.
This wrapper obtains numerical/refusal evidence, never concurrency clearance.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from checks.native_tests.nearest_centroid_descriptor_suite import fixture, empty_case
from checks.native_tests.nearest_centroid_plan_suite import require
from x_neighbors.native_descriptor_device import xn_native_v1


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    var failed = 0
    for unit in range(11):
        try:
            if unit < 8:
                fixture(xn_native_v1, unit)
            else:
                empty_case(xn_native_v1, unit - 8)
        except error:
            failed += 1
            print("FAIL nearest_centroid_numeric_gpu fixture", unit, String(error))
    require(failed == 0, "nearest_centroid_numeric_gpu failed " + String(failed) + " fixtures")
    print("PASS nearest_centroid_numeric_gpu: 11 ordered native numerical/refusal fixtures; concurrency not tested")

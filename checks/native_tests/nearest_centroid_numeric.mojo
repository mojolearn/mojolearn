# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""Existing NC numerical graphs, ordered on the one physical GPU.

Only one GPU job runs at a time. Each graph uses the shared DeviceContext, so
submissions are ordered here as in the PCA/TSVD numerical suites; kernels
retain their parallel execution. The separate parallel run_checks and device
concurrency fixtures remain unchanged, with their Metal failures still open.
This wrapper obtains numerical/refusal evidence, never concurrency clearance.
"""
from std.testing import TestSuite
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from checks.native_tests.nearest_centroid_descriptor_cases import fixture, empty_case
from x_neighbors.native_descriptor_device import xn_native_v1
from checks.native_tests.testsuite_cases import require


# ---- The device column (was nearest_centroid_numeric_gpu_suite.mojo) ----


def test_nearest_centroid_numeric() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    var failures = String("")
    for unit in range(11):  # driver-loop: over fixtures, in order
        try:
            if unit < 8:
                fixture(xn_native_v1, unit)
            else:
                empty_case(xn_native_v1, unit - 8)
        except error:
            failures += "\n      row " + String(unit) + ": " + String(error)
    if failures.byte_length() > 0:
        raise Error("nearest_centroid_numeric: device fixtures failed:" + failures)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

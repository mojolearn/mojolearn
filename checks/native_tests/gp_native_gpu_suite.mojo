# SPDX-License-Identifier: Apache-2.0
"""One actual GP graph at a time on the physical GPU; kernels stay parallel.

The separately reproduced concurrent shared-DeviceContext submission failure
remains open. This ordered numerical suite makes no concurrency claim.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.gp_native_api import GPDriver
from core.gp_native_provider import gp_native_v1
from core.gp_native_device import GPDeviceKernels
from checks.native_tests.gp_native_cases import GP_NATIVE_GPU_FIXTURES, fixture
from checks.native_tests.gp_native_learning_cases import GP_LEARNING_FIXTURES, learning_fixture
from checks.native_tests.gp_native_classifier_cases import GP_CLASSIFIER_GPU_FIXTURES, classifier_fixture
from checks.native_tests.gp_state_suite import require

comptime GP_NATIVE_GPU_TOTAL = GP_NATIVE_GPU_FIXTURES + GP_LEARNING_FIXTURES + GP_CLASSIFIER_GPU_FIXTURES


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failed = 0
    for unit in range(GP_NATIVE_GPU_TOTAL):
        try:
            if unit < GP_NATIVE_GPU_FIXTURES:
                fixture(GPDriver(gp_native_v1[GPDeviceKernels]), GPDriver(gp_native_v1[GPDeviceKernels]), unit, False)
            elif unit < GP_NATIVE_GPU_FIXTURES + GP_LEARNING_FIXTURES:
                learning_fixture(GPDriver(gp_native_v1[GPDeviceKernels]), unit - GP_NATIVE_GPU_FIXTURES, False)
            else:
                classifier_fixture(GPDriver(gp_native_v1[GPDeviceKernels]), GPDriver(gp_native_v1[GPDeviceKernels]),
                    unit - GP_NATIVE_GPU_FIXTURES - GP_LEARNING_FIXTURES, False)
        except error:
            failed += 1
            print("FAIL gp_native_gpu fixture", unit, String(error))
    require(failed == 0, "gp_native_gpu failed " + String(failed) + " fixtures")
    print("PASS gp_native_gpu:", GP_NATIVE_GPU_TOTAL, "actual ordered list/pointer/resident GP graphs")

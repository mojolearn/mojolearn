# SPDX-License-Identifier: Apache-2.0
"""Ordered GPR score numerical graphs on one physical GPU.

Shared DeviceContext concurrent submission remains separately unresolved.
Each graph preserves the parallel fit/query/metric kernels and range route.
"""
from core.gp_native_api import GPDriver
from core.gp_native_provider import gp_native_v1
from core.gp_native_device import GPDeviceKernels
from x_metrics.descriptor_device import metrics_program_device_v1
from checks.native_tests.gp_state_suite import require
from checks.native_tests.gpr_score_cases import fixture, GPR_SCORE_NATIVE_CASES


def run_checks() raises:
    var failed = 0
    for unit in range(GPR_SCORE_NATIVE_CASES):  # ordered graphs on the one shared physical GPU
        try:
            fixture(GPDriver(gp_native_v1[GPDeviceKernels]), metrics_program_device_v1, unit)
        except problem:
            failed += 1
            print("FAIL gpr_score_gpu fixture", unit, String(problem))
    require(failed == 0, "GPR score GPU failures: " + String(failed))
    print("PASS gpr_score_gpu:", GPR_SCORE_NATIVE_CASES, "fixtures")

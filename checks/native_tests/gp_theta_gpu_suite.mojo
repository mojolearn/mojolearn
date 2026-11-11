# SPDX-License-Identifier: Apache-2.0
"""Theta graphs run one at a time on the physical GPU; kernels stay parallel.

Shared-DeviceContext concurrent submission remains a separate recorded failure.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.gp_native_api import GPDriver
from core.gp_native_provider import gp_native_v1
from core.gp_native_device import GPDeviceKernels
from checks.native_tests.gp_theta_cases import GP_THETA_FIXTURES, theta_fixture
from checks.native_tests.gp_state_suite import require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failed = 0
    for unit in range(GP_THETA_FIXTURES):
        try:
            theta_fixture(GPDriver(gp_native_v1[GPDeviceKernels]), unit)
        except error:
            failed += 1
            print("FAIL gp_theta_gpu fixture", unit, String(error))
    require(failed == 0, "gp_theta_gpu failed " + String(failed) + " fixtures")
    print("PASS gp_theta_gpu:", GP_THETA_FIXTURES, "original precision/FTZ policy fixtures")

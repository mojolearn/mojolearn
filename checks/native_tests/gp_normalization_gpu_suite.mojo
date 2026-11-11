# SPDX-License-Identifier: Apache-2.0
"""One normalized GP graph at a time on the physical GPU; kernels stay parallel.

Shared-DeviceContext concurrent submission remains a separately recorded failure.
The two direct settings exercise existing scaler routes, not two algorithms."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.gp_native_api import GPDriver
from core.gp_native_provider import gp_native_v1
from core.gp_native_device import GPDeviceKernels
from core.scaler_native_api import ScalerDriver, PrepProgramDriver, scaler_native_v1, prep_program_v1
from core.scaler_native_device import ScalerDeviceKernels
from core.prep_program_native_device import PrepProgramDeviceKernels
from checks.native_tests.gp_normalization_cases import GP_NORMALIZATION_FIXTURES, normalization_fixture
from checks.native_tests.gp_state_suite import require

comptime GP_NORMALIZATION_GPU_FIXTURES = GP_NORMALIZATION_FIXTURES * 2


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failed = 0
    for unit in range(GP_NORMALIZATION_GPU_FIXTURES):
        try:
            normalization_fixture(GPDriver(gp_native_v1[GPDeviceKernels]),
                ScalerDriver(scaler_native_v1[ScalerDeviceKernels]),
                PrepProgramDriver(prep_program_v1[PrepProgramDeviceKernels]),
                unit % GP_NORMALIZATION_FIXTURES, unit >= GP_NORMALIZATION_FIXTURES)
        except error:
            failed += 1
            print("FAIL gp_normalization_gpu fixture", unit, String(error))
    require(failed == 0, "gp_normalization_gpu failed " + String(failed) + " fixtures")
    print("PASS gp_normalization_gpu:", GP_NORMALIZATION_GPU_FIXTURES, "actual normalization/GP graphs")

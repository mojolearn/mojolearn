# SPDX-License-Identifier: Apache-2.0
"""One actual GP graph at a time on the physical GPU; kernels stay parallel.

The separately reproduced concurrent shared-DeviceContext submission failure
remains open. This ordered numerical suite makes no concurrency claim.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.native_descriptor_abi import NativeDescriptorError
from core.gp_native_api import GPDriver, gp_native_dispatch
from core.gp_native_device import GPDeviceKernels
from checks.native_tests.gp_native_cases import GP_NATIVE_GPU_FIXTURES, fixture
from checks.native_tests.gp_state_suite import require


def device_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                 result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return gp_native_dispatch[GPDeviceKernels](operation, aa, na, pp, np, ff, nf, result, context, sink)


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failed = 0
    for unit in range(GP_NATIVE_GPU_FIXTURES):
        try:
            fixture(GPDriver(device_entry), GPDriver(device_entry), unit, False)
        except error:
            failed += 1
            print("FAIL gp_native_gpu fixture", unit, String(error))
    require(failed == 0, "gp_native_gpu failed " + String(failed) + " fixtures")
    print("PASS gp_native_gpu:", GP_NATIVE_GPU_FIXTURES, "actual ordered list/pointer/resident GP graphs")

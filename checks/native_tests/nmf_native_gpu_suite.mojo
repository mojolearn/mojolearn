# SPDX-License-Identifier: Apache-2.0
"""One physical GPU executes NMF graphs in order over its shared contexts.

GPU jobs cannot overlap these shared DeviceContext submissions; kernels and
independent oracle cells remain parallel. This suite does not test concurrent
GPU submissions and makes no claim that the recorded context race is fixed."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.nmf_native_api import NMFNativeDriver
from core.nmf_native_provider import nmf_native_v1
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from x_decomp.device import DevExec
from x_decomp.pca_randomized_dev import DevRDev
from x_decomp.decomp_owner_dev import DevDecomp
from checks.native_tests.nmf_native_cases import NMF_NATIVE_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    var failed = 0
    for unit in range(NMF_NATIVE_FIXTURES):
        try:
            fixture(NMFNativeDriver(nmf_native_v1[DevExec, DevRDev, True, DevDecomp]), unit, False)
        except error:
            failed += 1
            print("FAIL nmf_native_gpu fixture", unit, String(error))
    require(failed == 0, "nmf_native_gpu failed " + String(failed) + " fixtures")
    print("PASS nmf_native_gpu:", NMF_NATIVE_FIXTURES, "native fit/inverse/refusal fixtures")

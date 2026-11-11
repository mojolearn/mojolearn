# SPDX-License-Identifier: Apache-2.0
"""Actual KernelPCA graphs, one job at a time on the one physical GPU.

Shared DeviceContexts require ordered graph submission for this numerical
suite; kernels and oracle cells remain parallel. Separate recorded shared-
context concurrency failures are unchanged and are not cleared by this suite.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.neighbors_native_driver import XnDriver
from core.kpca_native_api import KPCANativeDriver, kpca_native_v1
from core.kpca_native_provider import KPCAEigenKernels
from core.kpca_native_plan import KPCANeighborDriver
from x_neighbors.native_descriptor_device import xn_native_v1
from x_decomp.descriptor_device import x_decomp_device_v1
from x_decomp.device import DevExec
from x_decomp.pca_randomized_dev import DevRDev
from x_decomp.decomp_owner_dev import DevDecomp
from checks.native_tests.kpca_native_cases import KPCA_NATIVE_FIXTURES, KPCA_RESIDENT_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    var failed = 0
    for unit in range(KPCA_NATIVE_FIXTURES + KPCA_RESIDENT_FIXTURES):
        try:
            fixture(KPCANativeDriver(kpca_native_v1[KPCAEigenKernels[DevExec, DevRDev, True, DevDecomp]]),
                KPCANeighborDriver(XnDriver(xn_native_v1)), x_decomp_device_v1, unit, False)
        except error:
            failed += 1
            print("FAIL kpca_native_gpu fixture", unit, String(error))
    require(failed == 0, "kpca_native_gpu failed " + String(failed) + " fixtures")
    print("PASS kpca_native_gpu:", KPCA_NATIVE_FIXTURES + KPCA_RESIDENT_FIXTURES, "native centering/eigen/query/refusal fixtures")

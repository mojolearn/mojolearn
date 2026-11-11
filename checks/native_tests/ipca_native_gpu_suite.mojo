# SPDX-License-Identifier: Apache-2.0
"""Same IncrementalPCA graphs plus real resident ownership on one physical GPU.

Graph submissions are ordered over shared DeviceContexts, one GPU job at a
time; existing kernels and oracle cells remain parallel. This is numerical
and ownership coverage, not clearance of the separate concurrency failure."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.ipca_native_api import IPCANativeDriver
from core.ipca_native_provider import ipca_native_v1
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from x_decomp.device import DevExec
from x_decomp.pca_randomized_dev import DevRDev
from x_decomp.ipca_owner_dev import DevIpcaDev
from checks.native_tests.ipca_native_cases import IPCA_NATIVE_FIXTURES, IPCA_RESIDENT_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    var failed = 0
    for unit in range(IPCA_NATIVE_FIXTURES + IPCA_RESIDENT_FIXTURES):
        try:
            fixture(IPCANativeDriver(ipca_native_v1[DevExec, DevRDev, True, DevIpcaDev]), unit, False)
        except error:
            failed += 1
            print("FAIL ipca_native_gpu fixture", unit, String(error))
    require(failed == 0, "ipca_native_gpu failed " + String(failed) + " fixtures")
    print("PASS ipca_native_gpu:", IPCA_NATIVE_FIXTURES + IPCA_RESIDENT_FIXTURES, "native fit/partial/query/refusal fixtures")

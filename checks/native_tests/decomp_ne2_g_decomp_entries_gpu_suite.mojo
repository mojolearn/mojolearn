# SPDX-License-Identifier: Apache-2.0
"""GaussianMixture fit_predict equals fit then predict, on the GPU provider.

Only the GPU provider owns the fused fit_predict (the host provider's
capability word is 0, checked in mixture_native_cpu), so there is no CPU
column. The production device context is shared, so fixture graphs submit
in order on the one device; the kernels keep their own parallelism and the
host-side row comparison uses the host pool.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.mixture_native_api import GmmDriver
from mixture.gmm_native_device import gmm_native_device_v1
from checks.native_tests.decomp_ne2_g_decomp_entries_cases import NE2G_FIXTURES, fixture, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    var failed = 0
    for unit in range(NE2G_FIXTURES):
        try:
            fixture(GmmDriver(gmm_native_device_v1), unit)
        except error:
            failed += 1
            print("FAIL decomp_ne2_g_decomp_entries_gpu fixture", unit, String(error))
    require(failed == 0, "decomp_ne2_g_decomp_entries_gpu failed " + String(failed) + " fixtures")
    print("PASS decomp_ne2_g_decomp_entries_gpu:", NE2G_FIXTURES, "fit_predict fixtures")

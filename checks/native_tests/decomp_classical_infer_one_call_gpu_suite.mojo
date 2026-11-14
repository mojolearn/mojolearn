# SPDX-License-Identifier: Apache-2.0
"""PCA transform writes every output word, on the GPU estimator provider.

The production device context is shared, so fixture graphs submit in order
on the one device; the device kernels keep their own parallelism and the
host-side cell comparisons use the host pool.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.pca_native_api import PCANativeDriver
from core.pca_native_device import pca_device_native_v1
from checks.native_tests.pca_native_cases import require
from checks.native_tests.decomp_classical_infer_one_call_cases import INFER_ONE_CALL_FIXTURES, fixture


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    var failed = 0
    for unit in range(INFER_ONE_CALL_FIXTURES):
        try:
            fixture(PCANativeDriver(pca_device_native_v1), unit)
        except error:
            failed += 1
            print("FAIL decomp_classical_infer_one_call_gpu fixture", unit, String(error))
    require(failed == 0, "decomp_classical_infer_one_call_gpu failed " + String(failed) + " fixtures")
    print("PASS decomp_classical_infer_one_call_gpu:", INFER_ONE_CALL_FIXTURES, "fully written transform fixtures")

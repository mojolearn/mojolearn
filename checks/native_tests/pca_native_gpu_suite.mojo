# SPDX-License-Identifier: Apache-2.0
"""Actual GPU PCA estimator and resident-fold providers.

Production contexts/stores are shared, so fixture graphs submit in order.
Device kernels retain their parallel algorithms; independent input decoding
and oracle cells use the host pool. No timing or identity comparison.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.pca_native_api import PCANativeDriver
from core.pca_native_device import pca_device_native_v1
from checks.native_tests.pca_native_cases import PCA_NATIVE_FIXTURES, fixture, require
from checks.native_tests.pca_native_fold_cases import PCA_NATIVE_FOLD_FIXTURES, fold_fixture


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    var failed = 0
    for unit in range(PCA_NATIVE_FIXTURES + PCA_NATIVE_FOLD_FIXTURES):
        try:
            if unit < PCA_NATIVE_FIXTURES:
                fixture(PCANativeDriver(pca_device_native_v1), unit, False)
            else:
                fold_fixture(unit - PCA_NATIVE_FIXTURES)
        except error:
            failed += 1
            print("FAIL pca_native_gpu fixture", unit, String(error))
    require(failed == 0, "pca_native_gpu failed " + String(failed) + " fixtures")
    print("PASS pca_native_gpu:", PCA_NATIVE_FIXTURES + PCA_NATIVE_FOLD_FIXTURES, "native fit/query/state/fold fixtures")

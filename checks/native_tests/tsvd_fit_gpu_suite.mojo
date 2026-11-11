# SPDX-License-Identifier: Apache-2.0
"""Actual estimator, randomized/TSQR and resident-fold TSVD fit providers.

Fixture graphs submit in order over shared device contexts/pools. Existing
GPU algorithms and independent oracle cells retain parallel execution.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.tsvd_native_api import TSVDNativeDriver
from core.tsvd_native_device import tsvd_device_native_v1
from core.tsvd_native_decomp import tsvd_decomp_native_v1
from x_decomp.device import DevExec
from x_decomp.pca_randomized_dev import DevRDev
from checks.native_tests.tsvd_fit_cases import TSVD_FIT_FIXTURES, fixture, require
from checks.native_tests.tsvd_fit_fold_cases import TSVD_FOLD_FIXTURES, fold_fixture


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    var failed = 0
    for unit in range(TSVD_FIT_FIXTURES + TSVD_FOLD_FIXTURES):
        try:
            if unit < TSVD_FIT_FIXTURES:
                fixture(TSVDNativeDriver(tsvd_device_native_v1),
                    TSVDNativeDriver(tsvd_decomp_native_v1[DevExec, DevRDev, True]), unit, False)
            else:
                fold_fixture(unit - TSVD_FIT_FIXTURES)
        except error:
            failed += 1
            print("FAIL tsvd_fit_gpu fixture", unit, String(error))
    require(failed == 0, "tsvd_fit_gpu failed " + String(failed) + " fixtures")
    print("PASS tsvd_fit_gpu:", TSVD_FIT_FIXTURES + TSVD_FOLD_FIXTURES, "native fit/reference/fold fixtures")

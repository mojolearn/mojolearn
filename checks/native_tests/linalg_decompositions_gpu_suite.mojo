# SPDX-License-Identifier: Apache-2.0
"""Device and host linalg entries agree word for word (from test_linalg_decompositions.py).

The five comparisons run in order: each one submits graphs to the shared
production DeviceContext, and concurrent submission from several host threads
to that one context is a separately recorded open failure. The device kernels
keep their own parallelism and the word comparisons run across host cores.
A CPU-only column has one route and nothing to compare, so this suite is GPU only.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from checks.native_tests.linalg_decompositions_cases import LINALG_DECOMPOSITIONS_FIXTURES, fixture
from checks.native_tests.linalg_numerical_cases import require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    var failed = 0
    for index in range(LINALG_DECOMPOSITIONS_FIXTURES):
        try:
            fixture(index)
        except error:
            failed += 1
            print("FAIL linalg_decompositions_gpu fixture", index, String(error))
    require(failed == 0, "linalg_decompositions_gpu failed " + String(failed) + " fixtures")
    print("PASS linalg_decompositions_gpu:", LINALG_DECOMPOSITIONS_FIXTURES, "device/host word comparisons")

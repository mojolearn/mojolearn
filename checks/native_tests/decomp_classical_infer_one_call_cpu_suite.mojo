# SPDX-License-Identifier: Apache-2.0
"""PCA transform writes every output word, on the CPU provider.

Independent fixtures run across host cores; the one fixture's cell
comparisons and data generation use the host pool as well.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.pca_native_api import PCANativeDriver
from core.pca_native_host import pca_host_native_v1
from checks.native_tests.pca_native_cases import require
from checks.native_tests.decomp_classical_infer_one_call_cases import INFER_ONE_CALL_FIXTURES, fixture


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=INFER_ONE_CALL_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(PCANativeDriver(pca_host_native_v1), unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL decomp_classical_infer_one_call_cpu fixture", unit, String(error))
    host_parallelize(task, INFER_ONE_CALL_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "decomp_classical_infer_one_call_cpu failed " + String(failed) + " fixtures")
    print("PASS decomp_classical_infer_one_call_cpu:", INFER_ONE_CALL_FIXTURES, "fully written transform fixtures")

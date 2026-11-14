# SPDX-License-Identifier: Apache-2.0
"""sparse_encode on the host Kit; the four independent fixtures run across
host cores, each with its own kit and failure slot."""
from checks.native_tests.core_verifier_gap_methods_cases import (
    VERIFIER_GAP_FIXTURES, require, encode_algo, encode_nnz, rows_mat, eye_mat, check_code,
)
from core.host_parallel import host_parallelize
from x_decomp.dictl import sparse_encode
from x_decomp.host import HostExec
from x_decomp.kit import Kit, Mat


def check_unit(unit: Int) raises:
    var k = Kit[HostExec]()
    var code = sparse_encode(k, rows_mat(), eye_mat(), encode_algo(unit), 0.125, encode_nnz(unit), Mat(0, 0),
                             False, 100, False)
    check_code(unit, code)


def run_checks() raises:
    var failures = List[Int](length=VERIFIER_GAP_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            check_unit(unit)
        except error:
            print("FAIL core_verifier_gap_methods_cpu fixture", unit, String(error))
            fp.unsafe_store(unit, 1)
    host_parallelize(task, VERIFIER_GAP_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "core_verifier_gap_methods_cpu failed " + String(failed) + " of " + String(VERIFIER_GAP_FIXTURES))
    print("PASS core_verifier_gap_methods_cpu:", VERIFIER_GAP_FIXTURES, "sparse_encode fixtures, parallel")

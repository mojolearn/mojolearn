# SPDX-License-Identifier: Apache-2.0
"""Five original numerical fixtures over the actual CPU linalg entry bodies."""
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr
from decomposition.linalg_host_impl import qr_r_host_run, svdvals_host_run
from x_decomp.cells import F32Ptr
from x_decomp.host import HostExec
from checks.native_tests.linalg_numerical_cases import LINALG_NUMERICAL_FIXTURES, LinalgNumericalCalls, fixture, require


struct HostLinalgNumerical(LinalgNumericalCalls):
    @staticmethod
    def qr(source: List[Float32], rows: Int, cols: Int, dst: Int) raises:
        qr_r_host_run(source, rows, cols, f32_ptr(dst))

    @staticmethod
    def singular(source: List[Float32], rows: Int, cols: Int, dst: Int) raises:
        svdvals_host_run(source, rows, cols, f32_ptr(dst))

    @staticmethod
    def eigen(source: Int, values: Int, vectors: Int, n: Int) raises:
        HostExec.eigh(F32Ptr(unsafe_from_address=source), F32Ptr(unsafe_from_address=values),
                      F32Ptr(unsafe_from_address=vectors), n, 1)


def run_checks() raises:
    var failures = List[Int](length=LINALG_NUMERICAL_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(index: Int) {imm fp}:
        try:
            fixture[HostLinalgNumerical](index)
        except error:
            fp.unsafe_store(index, 1)
            print("FAIL linalg_numerical_cpu fixture", index, String(error))
    host_parallelize(task, LINALG_NUMERICAL_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "linalg_numerical_cpu failed " + String(failed) + " fixtures")
    print("PASS linalg_numerical_cpu:", LINALG_NUMERICAL_FIXTURES, "native numerical fixtures")

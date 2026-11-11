# SPDX-License-Identifier: Apache-2.0
"""Independent dense preparation fixtures execute across host cores."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.dense_coo import F32P, I32P
from core.dense_coo_host import knn_affinity_f32
from checks.native_tests.spectral_dense_cases import SPECTRAL_COO_FIXTURES, SPECTRAL_AFFINITY_FIXTURES, SpectralDenseCalls, coo_case, affinity_case, require

comptime SPECTRAL_DENSE_CPU_FIXTURES = SPECTRAL_COO_FIXTURES + SPECTRAL_AFFINITY_FIXTURES


struct HostSpectralDense(SpectralDenseCalls):
    @staticmethod
    def affinity(source: Int, rows: Int, neighbors: Int, dst: Int, status: Int) raises:
        knn_affinity_f32(F32P(unsafe_from_address=source), I32P(unsafe_from_address=0),
            I32P(unsafe_from_address=0), F32P(unsafe_from_address=0), 0, False,
            rows, neighbors, F32P(unsafe_from_address=dst), I32P(unsafe_from_address=status))


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=SPECTRAL_DENSE_CPU_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            if unit < SPECTRAL_COO_FIXTURES:
                coo_case(unit)
            else:
                affinity_case[HostSpectralDense](unit - SPECTRAL_COO_FIXTURES)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL spectral_dense_cpu fixture", unit, String(error))
    host_parallelize(task, SPECTRAL_DENSE_CPU_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "spectral_dense_cpu failed " + String(failed) + " fixtures")
    print("PASS spectral_dense_cpu:", SPECTRAL_DENSE_CPU_FIXTURES, "original dense preparation fixtures")

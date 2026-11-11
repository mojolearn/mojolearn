# SPDX-License-Identifier: Apache-2.0
"""Independent dense preparation fixtures execute across host cores."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.spectral_preparation_api import SpectralPreparationDriver, SpectralKNNRequest, spectral_preparation_native_v1
from core.spectral_preparation_host import SpectralPreparationHost
from checks.native_tests.spectral_dense_cases import SPECTRAL_COO_FIXTURES, SPECTRAL_AFFINITY_FIXTURES, SpectralDenseCalls, coo_case, affinity_case, require

comptime SPECTRAL_DENSE_CPU_FIXTURES = SPECTRAL_COO_FIXTURES + SPECTRAL_AFFINITY_FIXTURES


struct HostSpectralDense(SpectralDenseCalls):
    @staticmethod
    def affinity(source: Int, rows: Int, neighbors: Int, dst: Int, status: Int) raises:
        var driver = SpectralPreparationDriver(spectral_preparation_native_v1[SpectralPreparationHost])
        driver.knn(SpectralKNNRequest(source, 0, 0, 0, dst, status, rows, neighbors, 0, False))


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=SPECTRAL_DENSE_CPU_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            if unit < SPECTRAL_COO_FIXTURES:
                var driver = SpectralPreparationDriver(spectral_preparation_native_v1[SpectralPreparationHost])
                coo_case(driver, unit)
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

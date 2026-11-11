# SPDX-License-Identifier: Apache-2.0
"""Independent RBF owner and actual host numerical graphs use all host cores."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from x_decomp.descriptor_host import x_decomp_host_v1
from checks.native_tests.spectral_rbf_cases import SPECTRAL_RBF_OWNERSHIP_FIXTURES, SPECTRAL_RBF_NUMERICAL_FIXTURES, ownership, numerical, require

comptime SPECTRAL_RBF_CPU_FIXTURES = SPECTRAL_RBF_OWNERSHIP_FIXTURES + SPECTRAL_RBF_NUMERICAL_FIXTURES


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=SPECTRAL_RBF_CPU_FIXTURES,fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            if unit < SPECTRAL_RBF_OWNERSHIP_FIXTURES:
                ownership(unit)
            else:
                numerical(x_decomp_host_v1,unit-SPECTRAL_RBF_OWNERSHIP_FIXTURES,False)
        except error:
            fp.unsafe_store(unit,1)
            print("FAIL spectral_rbf_cpu fixture",unit,String(error))
    host_parallelize(task,SPECTRAL_RBF_CPU_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0,"spectral_rbf_cpu failed " + String(failed) + " fixtures")
    print("PASS spectral_rbf_cpu:",SPECTRAL_RBF_CPU_FIXTURES,"supplementary owner and RBF formula fixtures")

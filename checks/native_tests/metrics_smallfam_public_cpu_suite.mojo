# SPDX-License-Identifier: Apache-2.0
"""lu_factor / lu_solve on the x_decomp host executor.

The seven units (the factor, then six trans x right-hand-side solves, each
on its own factor) are independent and run across host cores. The device
factor and solve are gated by the decomposition kit's own device checks;
the original ran wherever the public call ran, and its host column is
this executor.
"""
from core.host_parallel import host_parallelize
from checks.native_tests.metrics_smallfam_public_cases import LU_UNITS, require, lu_unit


def run_checks() raises:
    var failures = List[Int](length=LU_UNITS, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            lu_unit(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL metrics_smallfam_public_cpu unit", unit, String(error))
    host_parallelize(task, LU_UNITS)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "metrics_smallfam_public_cpu failed " + String(failed) + " units")
    print("PASS metrics_smallfam_public_cpu:", LU_UNITS, "parallel lu_factor / lu_solve units")

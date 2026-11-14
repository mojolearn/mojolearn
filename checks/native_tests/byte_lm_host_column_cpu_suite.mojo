# SPDX-License-Identifier: Apache-2.0
"""The byte LM host column units, independent, run across host cores."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from checks.native_tests.byte_lm_host_column_cases import BYTE_LM_HOST_COLUMN_CASES, case, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var failures = List[Int](length=BYTE_LM_HOST_COLUMN_CASES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            case(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL byte_lm_host_column_cpu unit", unit, String(error))
    host_parallelize(task, BYTE_LM_HOST_COLUMN_CASES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "byte_lm_host_column_cpu failed " + String(failed) + " units")
    print("PASS byte_lm_host_column_cpu:", BYTE_LM_HOST_COLUMN_CASES, "host inference and training units, parallel")

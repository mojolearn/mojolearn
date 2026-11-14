# SPDX-License-Identifier: Apache-2.0
"""Holt-Winters estimated initialization on the host fit body.

The ten fits are independent units and run across host cores, each into
its own record; the cross-unit comparisons follow. The fit is the host
reference body only (the device estimate is checked against it in
hw_check), so there is no GPU column, as in the original, which fitted
under the reference training context.
"""
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr
from checks.native_tests.tsa_holtwinters_initialization_cases import (
    HW_INIT_UNITS, HW_RECORD, require, fit_unit, check_units,
)


def run_checks() raises:
    var records = List[Float32](length=HW_INIT_UNITS * HW_RECORD, fill=Float32(0))
    var planes = List[Int](length=HW_INIT_UNITS, fill=0)
    var failures = List[Int](length=HW_INIT_UNITS, fill=0)
    var fp = failures.unsafe_ptr()
    var pp = planes.unsafe_ptr()
    var base = Int(records.unsafe_ptr())
    def task(unit: Int) {imm fp, imm pp, imm base}:
        try:
            pp.unsafe_store(unit, fit_unit(unit, f32_ptr(base + 4 * unit * HW_RECORD)))
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL tsa_holtwinters_initialization_cpu unit", unit, String(error))
    host_parallelize(task, HW_INIT_UNITS)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "tsa_holtwinters_initialization_cpu failed " + String(failed) + " fits")
    check_units(records, planes)
    print("PASS tsa_holtwinters_initialization_cpu:", HW_INIT_UNITS, "parallel Holt-Winters fits")

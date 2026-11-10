# SPDX-License-Identifier: Apache-2.0
"""Per-call and resident GPU Adafactor, matrix/vector, independent oracle.

Submission is ordered for the existing shared sequence_ctx/resident pool.
Kernels and native reference rows/columns/cells remain parallel; this is not
concurrent-session or multi-GPU evidence. No Python callbacks or objects.
"""
from sequence.exec_device import DeviceExec
from sequence.opt_resident import adafactor_resident_open, adafactor_resident_step, opt_resident_move, opt_resident_close
from checks.native_tests.adafactor_cases import (
    ADAFACTOR_FIXTURES, AF_STEPS, require, af_shape, af_config, af_data,
    af_expected, af_check, run_adafactor_fixture,
)


def resident_fixture(fixture_id: Int) raises:
    var shape = af_shape(fixture_id)
    var rows = shape[0]
    var cols = shape[1]
    var n = rows * cols if cols else rows
    var data = af_data(fixture_id)
    var expected = af_expected(fixture_id, data)
    var parameters = data[0].copy()
    var first = List[Float32](length=rows, fill=0)
    var second = List[Float32](length=max(cols, 1), fill=0)
    var opened = adafactor_resident_open([rows, cols])
    var handle = opened[0]
    try:
        for step in range(1, AF_STEPS + 1):  # dependent optimizer recurrence
            var addresses: List[Int] = [Int(parameters.unsafe_ptr()), Int(data[1].unsafe_ptr()) + 4 * (step - 1) * n]
            var ints: List[Int] = [rows, cols, step]
            require(adafactor_resident_step(handle, addresses, ints, af_config()) == n, "Adafactor resident extent")
        _ = opt_resident_move(handle, 0, Int(first.unsafe_ptr()), False)
        if cols:
            _ = opt_resident_move(handle, 1, Int(second.unsafe_ptr()), False)
    except error:
        _ = opt_resident_close(handle)
        raise error
    _ = opt_resident_close(handle)
    af_check(parameters, expected[0], "resident parameters")
    af_check(first, expected[1], "resident row/variance state")
    af_check(second, expected[2], "resident column state")
    _ = data
    _ = parameters
    _ = first
    _ = second


def main() raises:
    var failed = 0
    for fixture_id in range(2 * ADAFACTOR_FIXTURES):
        try:
            if fixture_id < ADAFACTOR_FIXTURES:
                var ex = DeviceExec()
                run_adafactor_fixture(ex, fixture_id)
            else:
                resident_fixture(fixture_id - ADAFACTOR_FIXTURES)
        except error:
            print("FAIL adafactor_gpu fixture", fixture_id, String(error))
            failed += 1
    require(failed == 0, "adafactor_gpu failed " + String(failed) + " fixtures")
    print("PASS adafactor_gpu:", 2 * ADAFACTOR_FIXTURES, "per-call/resident matrix/vector fixtures; shared-context ordered")

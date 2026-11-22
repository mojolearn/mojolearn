# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""log_loss against the clipped negative-log definition (lane py2mojo-D, batch 5, 2026-11-22; ported from
checks/log_loss_smoke.py).

The program is the one the log_loss door builds without sample weights (bindings/mcls_classify_door.mojo
`_log_loss`: the probabilities as the first input, `mc_proba_stages`, the encoded labels as the second input, then
`mc_row_mean` with the LOGLOSS row metric, all typed in bindings/mcls_classify_run.mojo), with the probabilities and
the codes as the caller's buffers, run on the metrics program entry of the column (x_metrics/descriptor_host.mojo on
the host column, x_metrics/descriptor_device.mojo on the device column). The composition repeats the door's stage
order line for line.

The original's five cases, with the labels already encoded as the door encodes them (the given `labels` order, else
the sorted labels seen; a one-column or 1-D probability is the positive class and is expanded to [1 - p, p]):

    case 0  y [0, 1], p [0.25, 0.75] (1-D):               selected [0.75, 0.75]
    case 1  y [0, 1], p [1, 0] (1-D):                     selected [eps, eps] after the clip
    case 2  y ["b", "a"], labels ["b", "a"] (codes 0, 1),
            p [[0.75, 0.25], [0.5, 0.5]]:                  selected [0.75, 0.5]
    case 3  y [2, 0, 1], p 3 x 3 rows of 0.25 / 0.5:       selected [0.5] * 3
    case 4  y [1] * 257, labels [0, 1], p (257, 1) 0.75:   selected [0.75] * 257

unit 2c + normalize (normalize False, True): the value equals `sum -log(min(1 - eps, max(eps, p)))` in binary64
(eps = float32 eps), divided by n when normalized, to `math.isclose(rel_tol=3e-5, abs_tol=2e-6)` as the original.

The string labels' encoding, the `fast` mode arm and the public float stay in Python (public boundary); the `fast`
column is Apple only.
"""
from std.math import log
from bindings.arena_prog_run import MetricProgramInput, metrics_plan_execute
from bindings.mcls_classify_run import mc_proba_code, mc_proba_stages, mc_row_mean
from core.metrics_program_plan import MetricProgramPlan
from core.native_descriptor_abi import NativeDescriptorCall
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_metrics.descriptor_host import metrics_program_host_v1
from x_metrics.descriptor_device import metrics_program_device_v1
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require, CPU_COLUMN_BUILD, check_cases_serial

comptime LOG_LOSS_UNITS = 10
comptime _ROW_LOGLOSS = 2  # bindings/mcls_classify_run.mojo
comptime EPS32 = 1.1920928955078125e-07  # np.finfo(np.float32).eps


def log_loss_value(entry: NativeDescriptorCall, proba: List[Float32], codes: List[Int32], n: Int, k: Int,
                   binary: Bool, normalize: Bool) raises -> Float64:
    """The `_log_loss` door program over the caller's probabilities and codes."""
    var plan = MetricProgramPlan()
    # bindings/mcls_classify_door.mojo `_log_loss`, in its order
    var a_off = plan.input(len(proba))
    var put = mc_proba_stages(plan, a_off, n, k, binary)
    var y = plan.input(n)
    var res = mc_row_mean(plan, put[0], k, y, n, _ROW_LOGLOSS, False, normalize, False)
    var records = List[MetricProgramInput]()
    var pin = 0
    for r in range(len(plan.input_offsets)):  # driver-loop: over the program's input records
        if plan.input_consts[r] >= 0:
            records.append(MetricProgramInput(plan.input_offsets[r], plan.const_address(r),
                                              len(plan.consts[plan.input_consts[r]])))
        else:
            if pin == 0:
                records.append(MetricProgramInput(plan.input_offsets[r], Int(proba.unsafe_ptr()), len(proba)))
            else:
                records.append(MetricProgramInput(plan.input_offsets[r], Int(codes.unsafe_ptr()), n))
            pin += 1
    require(pin == 2, "log_loss program: expected the two caller buffers, found " + String(pin))
    var arena = metrics_plan_execute(entry, plan, records, False, False)
    var code = mc_proba_code(arena, put[1])
    require(code == 0, "log_loss program: the probability check failed with code " + String(code))
    var value = arena.f64(res)
    _ = records^
    _ = arena^
    return value


def log_loss_unit(entry: NativeDescriptorCall, unit: Int) raises:
    var which = unit // 2
    var normalize = unit % 2 == 1
    var proba = List[Float32]()
    var codes = List[Int32]()
    var selected = List[Float64]()
    var k = 2
    var binary = True
    if which == 0:
        proba = [0.25, 0.75]
        codes = [0, 1]
        selected = [0.75, 0.75]
    elif which == 1:
        proba = [1.0, 0.0]
        codes = [0, 1]
        selected = [EPS32, EPS32]
    elif which == 2:
        proba = [0.75, 0.25, 0.5, 0.5]
        codes = [0, 1]
        selected = [0.75, 0.5]
        binary = False
    elif which == 3:
        proba = [0.25, 0.25, 0.5, 0.5, 0.25, 0.25, 0.25, 0.5, 0.25]
        codes = [2, 0, 1]
        selected = [0.5, 0.5, 0.5]
        k = 3
        binary = False
    else:
        proba = List[Float32](length=257, fill=Float32(0.75))
        codes = List[Int32](length=257, fill=Int32(1))
        selected = List[Float64](length=257, fill=0.75)
    var n = len(codes)
    var total = Float64(0)
    for i in range(len(selected)):  # small-loop(the binary64 definition over at most 257 values)
        total += -log(min(1.0 - EPS32, max(EPS32, selected[i])))
    var want = total / Float64(n) if normalize else total
    var got = log_loss_value(entry, proba, codes, n, k, binary, normalize)
    require(abs(got - want) <= max(3e-5 * max(abs(got), abs(want)), 2e-6),
        "log_loss case " + String(which) + " normalize " + String(normalize) + ": " + String(got) + " expected "
        + String(want))


def _host_unit(unit: Int) raises:
    log_loss_unit(metrics_program_host_v1, unit)


def _device_unit(unit: Int) raises:
    log_loss_unit(metrics_program_device_v1, unit)


def test_log_loss() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    comptime if CPU_COLUMN_BUILD:
        check_cases("log_loss", LOG_LOSS_UNITS, _host_unit)
    else:
        # ordered programs on the one shared device context
        check_cases_serial("log_loss", LOG_LOSS_UNITS, _device_unit)


def main() raises:
    # The test by name, not discover_tests: discovery instantiates every function of the module, and the device
    # column's functions do not compile in the host column build (as metrics_r2_epilogue.mojo).
    var suite = TestSuite()
    suite.test[test_log_loss]()
    suite^.run()

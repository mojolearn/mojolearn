# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The r2_score epilogue's exceptional arithmetic: per-output `1 - num / den` and the multioutput average with the NaN
rule (lane py2mojo-pilot, 2026-11-22; ported from python/mojolearn/tests/test_metrics_undefined_score.py
`test_zero_weight_infinite_score_is_explicit_positive_nan` and
`test_finite_scores_and_nonzero_weight_infinity_keep_semantics`).

The program is the one `r2_score_options` builds (bindings/metrics_reg_door.mojo `_r2_options`: the typed `_Reg`
builder of bindings/metrics_reg_run.mojo, the y mean as the Float32 column, SS_res, SS_tot, reg_epi ASM then AVG),
with y and p as the caller's buffers, run on the metrics program entry of the column (x_metrics/descriptor_host.mojo
on the host column, x_metrics/descriptor_device.mojo on the device column). The composition below repeats the door's
stage order line for line; the arithmetic under test is the reg_epi kernel (x_metrics/reg_epi.mojo).

Inputs are the original's 2 x 2 literals:
    INF   y [[0, 3], [2, 3]], p [[0, 3], [1, 4]]: num [1, 1], den [2, 0], scores [0.5, -inf]
    FIN   y [[0, 0], [2, 4]], p [[0, 1], [1, 3]]: num [1, 2], den [2, 8], scores [0.5, 0.75]
    ZERO  y = p = [[1, 1], [1, 1]]: num [0, 0], den [0, 0], scores [nan, nan]

    unit 0  INF variance_weighted, force_finite False: the positive quiet NaN 0x7ff8000000000000
    unit 1  INF custom weights [1, 0], force_finite False: the positive quiet NaN
    unit 2  INF variance_weighted, force_finite True: 0.5
    unit 3  FIN variance_weighted: 0.7 to rel 1e-15
    unit 4  INF uniform_average: -inf
    unit 5  INF raw_values: [0.5, -inf]
    unit 6  ZERO variance_weighted: NaN

The public call's input admission and the float or array it returns stay in the Python boundary tests.
"""
from std.math import isnan, isinf
from std.memory import bitcast
from bindings.arena_prog_run import MetricArenaOwner, MetricProgramInput, metrics_plan_execute
from bindings.metrics_reg_run import _Reg, reg_raw_values
from core.metrics_program_plan import metric_term
from core.native_descriptor_abi import NativeDescriptorCall
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_metrics.descriptor_host import metrics_program_host_v1
from x_metrics.descriptor_device import metrics_program_device_v1
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require, CPU_COLUMN_BUILD, check_cases_serial

comptime R2_UNITS = 7
comptime _RE_DIV = 1  # bindings/metrics_reg_run.mojo


def r2_values(entry: NativeDescriptorCall, y: List[Float32], p: List[Float32], mo: String, var custom: List[Float64],
              force_finite: Bool) raises -> List[Float64]:
    """The r2_score_options program over row-major (2, 2) y and p: one value, or the per-output raw values."""
    var n = 2
    var d = 2
    var r = _Reg(n, d, False, mo, custom^)
    var sq = metric_term("sq")
    r.program(0.0)
    # bindings/metrics_reg_door.mojo `_r2_options`, in its order
    var y_off = r.y_off
    var p_off = r.p_off
    var y_sums = r.sums(y_off)
    var y_mean = r.fin32(y_sums, _RE_DIV)
    var res_terms = r.term(sq, y_off, p_off, False)
    var res_sums = r.sums(res_terms)
    var num = r.fin(res_sums, 0)
    var tot_terms = r.term(sq, y_off, y_mean, True)
    var tot_sums = r.sums(tot_terms)
    var den = r.fin(tot_sums, 0)
    var out = r.assemble(num, den, force_finite, True)
    # the input records: the door's pins (y then p) and the program's constant words
    var records = List[MetricProgramInput]()
    var pin = 0
    for k in range(len(r.plan.input_offsets)):  # driver-loop: over the program's input records
        if r.plan.input_consts[k] >= 0:
            records.append(MetricProgramInput(r.plan.input_offsets[k], r.plan.const_address(k),
                                              len(r.plan.consts[r.plan.input_consts[k]])))
        else:
            records.append(MetricProgramInput(r.plan.input_offsets[k],
                                              Int(y.unsafe_ptr()) if pin == 0 else Int(p.unsafe_ptr()), n * d))
            pin += 1
    require(pin == 2, "r2 program: expected the two caller buffers y and p, found " + String(pin))
    var arena = metrics_plan_execute(entry, r.plan, records, False, False)
    var values: List[Float64]
    if mo == "raw_values":
        values = reg_raw_values(r, out, arena)
    else:
        values = [arena.f64(out)]
    _ = records^
    _ = arena^
    return values^


def r2_unit(entry: NativeDescriptorCall, unit: Int) raises:
    var y_inf: List[Float32] = [0.0, 3.0, 2.0, 3.0]
    var p_inf: List[Float32] = [0.0, 3.0, 1.0, 4.0]
    var y_fin: List[Float32] = [0.0, 0.0, 2.0, 4.0]
    var p_fin: List[Float32] = [0.0, 1.0, 1.0, 3.0]
    var ones: List[Float32] = [1.0, 1.0, 1.0, 1.0]
    var none = List[Float64]()
    if unit == 0 or unit == 1:
        var mo = String("variance_weighted") if unit == 0 else String("custom")
        var custom = List[Float64]()
        if unit == 1:
            custom = [1.0, 0.0]
        var got = r2_values(entry, y_inf, p_inf, mo, custom^, False)[0]
        require(bitcast[DType.uint64](got) == UInt64(0x7FF8000000000000),
            mo + ": a zero-weight infinite score must average to the positive quiet NaN, got " + String(got))
    elif unit == 2:
        var got = r2_values(entry, y_inf, p_inf, "variance_weighted", none.copy(), True)[0]
        require(got == 0.5, "force_finite variance_weighted: " + String(got) + " expected 0.5")
    elif unit == 3:
        var got = r2_values(entry, y_fin, p_fin, "variance_weighted", none.copy(), False)[0]
        require(abs(got - 0.7) <= 1e-15 * 0.7, "variance_weighted: " + String(got) + " expected 0.7")
    elif unit == 4:
        var got = r2_values(entry, y_inf, p_inf, "uniform_average", none.copy(), False)[0]
        require(isinf(got) and got < 0, "uniform_average: " + String(got) + " expected -inf")
    elif unit == 5:
        var got = r2_values(entry, y_inf, p_inf, "raw_values", none.copy(), False)
        require(len(got) == 2 and got[0] == 0.5 and isinf(got[1]) and got[1] < 0,
            "raw_values: expected [0.5, -inf], got " + String(got[0]) + ", " + String(got[len(got) - 1]))
    else:
        var got = r2_values(entry, ones, ones, "variance_weighted", none.copy(), False)[0]
        require(isnan(got), "zero num and den: " + String(got) + " expected NaN")


def _host_unit(unit: Int) raises:
    r2_unit(metrics_program_host_v1, unit)


def _device_unit(unit: Int) raises:
    r2_unit(metrics_program_device_v1, unit)


def test_r2_epilogue() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    comptime if CPU_COLUMN_BUILD:
        check_cases("r2_epilogue", R2_UNITS, _host_unit)
    else:
        # ordered programs on the one shared device context
        check_cases_serial("r2_epilogue", R2_UNITS, _device_unit)


def main() raises:
    # The test by name, not discover_tests: discovery instantiates every function of the module, and the device
    # column's functions do not compile in the host column build (as resample_cv_row_order.mojo).
    var suite = TestSuite()
    suite.test[test_r2_epilogue]()
    suite^.run()

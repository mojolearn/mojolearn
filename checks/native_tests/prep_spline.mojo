# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""SplineTransformer against scikit-learn: uniform, quantile and given knots, every extrapolation, sample_weight,
handle_missing='zeros' (lane py2mojo-A, 2026-11-22; ported from python/mojolearn/tests/test_x_prep_spline.py
`test_spline` and `test_spline_options`, their numeric part).

The SplineTransformer member (core/prep_member.mojo kind 9: the encode door's typed stages of
bindings/prep_encode_spline_run.mojo) runs over the host column's x_prep transport
(checks/native_tests/py2mojo_host_members.mojo `HostMember`) with the words `_spline_admit` writes ([9, degree, knots
0 given / 1 quantile / 2 uniform, n_knots, extrapolation 0 constant / 1 continue / 2 linear / 3 periodic, 0,
handle_missing is 'error', include_bias, 0 (the fast fused route), x_prep_idn_fam blocked bit, R3 work 1, order 'F',
r, c, the r x c given knots as float32 values]). Inputs: the original's 300 x 3 training and 100 x 3 held-out draws
(seed 0 for `test_spline`, seed 1 for the options: x1.5 and x1.7 held out, the integer weights in [0, 4), the 4 x 3
knots, NaN every 7th training row of feature 1 and every 5th held-out row of feature 0), with scikit-learn 1.9.1's
transforms, in `fixtures/prep_members/` (tools/py2mojo_native_fixtures_b1.py).

    units 0-3    test_spline: {}, n_knots=6 degree=2 quantile, continue, include_bias=False degree=1 (rtol 1e-3,
                 atol 2e-4)
    units 4-14   test_spline_options' eleven cases (given knots, linear, periodic, weights, quantile, order='F')
    units 15-16  one feature, linear extrapolation at degree 1 and 0
    units 17-19  handle_missing='zeros' (uniform, quantile, periodic)
                 (units 4-19: rtol 1e-3, atol 5e-4; the output width is checked first)

The NaN refusal of handle_missing='error' and order='F' memory layout stay in the Python file. Every unit is
independent and the case table runs across host cores.
"""
from core.member_abi import MEMBER_OP_TRANSFORM
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from checks.native_tests.py2mojo_host_members import (
    HostMember, MEMBER_PREP, PREP_MEMBERS_DIR, host_idn_fam, rows_close, require,
)
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, fixture_f32

comptime N = 300
comptime H = 100
comptime CASES = 20


def column0(x: List[Float32], n: Int) -> List[Float32]:
    var part = List[Float32](length=n, fill=0)
    for i in range(n):  # small-loop(fixture: X[:, :1])
        part[i] = x[3 * i]
    return part^


def spline_case(unit: Int) raises:
    # the original's keyword sets, in order (tools/py2mojo_native_fixtures_b1.py SPLINE_CASES)
    var degree: List[Int] = [3, 2, 3, 1, 3, 2, 3, 3, 2, 3, 3, 3, 3, 3, 2, 1, 0, 3, 3, 3]
    var knots: List[Int] = [2, 1, 2, 2, 0, 0, 2, 2, 2, 0, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2]
    var n_knots: List[Int] = [5, 6, 5, 5, 0, 0, 5, 5, 4, 0, 4, 5, 6, 4, 5, 5, 5, 5, 5, 5]
    var extrapolation: List[Int] = [0, 0, 1, 0, 0, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 2, 2, 0, 0, 3]
    var bias: List[Int] = [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    var label = String("SplineTransformer case ") + String(unit)
    var weighted = unit == 11 or unit == 12
    var missing_zeros = unit >= 17
    var cols = 1 if unit == 15 or unit == 16 else 3
    var params: List[Float64] = [9.0, Float64(degree[unit]), Float64(knots[unit]), Float64(n_knots[unit]),
        Float64(extrapolation[unit]), 0.0, 0.0 if missing_zeros else 1.0, Float64(bias[unit]), 0.0,
        1.0 if (host_idn_fam() & 1) != 0 else 0.0, 1.0, 1.0 if unit == 14 else 0.0, 0.0, 0.0]
    if knots[unit] == 0:
        var kn = fixture_f32(PREP_MEMBERS_DIR + "spl-kn.f32", 12)
        params[12] = 4.0
        params[13] = 3.0
        for i in range(12):  # small-loop(fixture: the given knot rows)
            params.append(Float64(kn[i]))
    var x: List[Float32]
    var xh: List[Float32]
    if unit < 4:
        x = fixture_f32(PREP_MEMBERS_DIR + "spl-x0.f32", N * 3)
        xh = fixture_f32(PREP_MEMBERS_DIR + "spl-xh0.f32", H * 3)
    elif missing_zeros:
        x = fixture_f32(PREP_MEMBERS_DIR + "spl-xn.f32", N * 3)
        xh = fixture_f32(PREP_MEMBERS_DIR + "spl-xhn.f32", H * 3)
    else:
        x = fixture_f32(PREP_MEMBERS_DIR + "spl-x1.f32", N * 3)
        xh = fixture_f32(PREP_MEMBERS_DIR + "spl-xh1.f32", H * 3)
    if cols == 1:
        x = column0(x, N)
        xh = column0(xh, H)
    var weights = fixture_f32(PREP_MEMBERS_DIR + "spl-sw.f32", N) if weighted else List[Float32]()
    var key = PREP_MEMBERS_DIR + "spl-" + String(unit)
    var width = Int(fixture_f64(key + "-width.f64", 1)[0])
    var member = HostMember(MEMBER_PREP, params^)
    try:
        member.fit_with(x, N, cols, weights)
        var got = member.width(MEMBER_OP_TRANSFORM)
        require(got == width, label + ": n_features_out_ " + String(got) + " expected " + String(width))
        rows_close(member.transform(xh, H), fixture_f64(key + "-transform.f64", H * width), 1e-3,
            2e-4 if unit < 4 else 5e-4, label + ": transform")
    finally:
        member.free()


def basic_case(unit: Int) raises:
    spline_case(unit)


def options_case(unit: Int) raises:
    spline_case(4 + unit)


def test_spline() raises:
    check_cases("spline", 4, basic_case)


def test_spline_options() raises:
    check_cases("spline_options", CASES - 4, options_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

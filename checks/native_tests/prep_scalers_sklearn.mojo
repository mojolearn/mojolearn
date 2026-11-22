# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""RobustScaler, MaxAbsScaler, Normalizer and Binarizer programs against scikit-learn's values (lane py2mojo-pilot,
2026-11-22; ported from python/mojolearn/tests/test_x_prep_scalers.py `test_robust_scaler`, `test_maxabs_scaler`,
test_x_prep_normalizer.py `test_normalizer` and test_x_prep_binarizer.py `test_binarizer`).

The inputs are the original tests' NumPy draws and the references the scikit-learn calls they made, stored as raw
words in `fixtures/prep_sanity/` by `tools/py2mojo_native_fixtures.py` (the command and versions are in its
docstring). The programs are the ones the Prep member runs on the host column in identical mode (core/prep_member.mojo
`_fit_robust`, `_fit_maxabs`, the binarizer and normalizer transforms): the same arena layout, input words, stages and
fused fit_transform, run through the x_prep host program runner.

    robust      301 x 6 float32 (ties, a constant column, a zero column, NaN every seventh row of feature 4), the five
                keyword sets of the original ({}, quantile_range (10, 90), with_centering False, unit_variance,
                unit_variance with quantile_range (5, 80)) x 2048-row statistics blocks off and on: center_ and scale_
                to rtol = atol = 1e-5, fit_transform to rtol = atol = 1e-4, NaN where scikit-learn has NaN
    maxabs      the same shape (seed 1), blocks off and on: scale_ to rtol 1e-6, fit_transform to rtol 1e-6,
                atol 1e-7
    normalizer  50 x 7 with a zero row, norm l1, l2, max: rtol 1e-6, atol 1e-7
    binarizer   60 x 5 rounded to one decimal, threshold 0.0, 0.3, -1.0: exact

The radix quantile selection and the device scratch region are the fast and device routes, not the host column. The
public dtype, shape and refusals stay in the Python files (public boundary). Every case is an independent unit and the
case tables run across host cores.
"""
from std.math import isnan
from std.memory import bitcast
from core.preprocessing_program import prep_program_stage
from x_prep.host.program import run_program_host
from x_prep.robust_native_run import _robust_fit_program
from x_prep.maxabs_native_run import _maxabs_fit_program
from checks.native_tests.py2mojo_fixture_words import PREP_SANITY_DIR, fixture_f64
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require, fixture_f32

comptime SCALER_N = 301
comptime SCALER_D = 6
comptime OP_BINARIZE = 12  # x_prep/prep_stateless_native_run.mojo
comptime OP_NORMALIZE = 47


def all_close(actual: Float32, expected: Float64, rtol: Float64, atol: Float64, label: String) raises:
    """numpy.testing.assert_allclose for one value: NaN only against NaN."""
    var a = Float64(actual)
    if isnan(expected) or isnan(a):
        require(isnan(expected) and isnan(a), label + ": " + String(actual) + " expected " + String(expected))
        return
    require(abs(a - expected) <= atol + rtol * abs(expected),
        label + ": " + String(actual) + " expected " + String(expected))


def run(mut arena: List[Float32], program: List[Int32]) raises:
    run_program_host(Int(arena.unsafe_ptr()), len(arena), Int(program.unsafe_ptr()), len(program) // 16)


# ---- RobustScaler (core/prep_member.mojo `_fit_robust`, the door's robust_fit layout, no work region) ----
def robust_case(unit: Int) raises:
    var k = unit // 2
    var blocked = unit % 2 == 1
    var lows: List[Float64] = [25.0, 10.0, 25.0, 25.0, 5.0]
    var highs: List[Float64] = [75.0, 90.0, 75.0, 75.0, 80.0]
    var centering = k != 2
    var unit_variance = k >= 3
    var label = String("robust kwargs ") + String(k) + " blocked " + String(blocked)
    var n = SCALER_N
    var d = SCALER_D
    var x = fixture_f32(PREP_SANITY_DIR + "scalers-x0.f32", n * d)
    var blocks = (n + 2047) // 2048
    var part_words = 5 * blocks * d if blocked else 0
    var sorted = n * d
    var stats = n * d + n * d
    var fractions = stats + 6 * d
    var q = fractions + 3
    var center = q + 3 * d
    var scale = center + d
    var host = scale + d + part_words
    var dest = host
    host += n * d
    var part = scale + d
    var arena = List[Float32](length=host, fill=0)
    for i in range(n * d):  # small-loop(fixture: the input rows into the arena)
        arena[i] = x[i]
    arena[fractions] = Float32(lows[k] / 100.0)
    arena[fractions + 1] = Float32(0.5)
    arena[fractions + 2] = Float32(highs[k] / 100.0)
    var program = List[Int32]()
    _robust_fit_program(program, False, blocked, unit_variance, True, centering, True, n, d, blocks,
        sorted, part, stats, fractions, q, center, scale, dest)
    run(arena, program)
    var want_center = fixture_f64(PREP_SANITY_DIR + "robust-" + String(k) + "-center.f64", d)
    var want_scale = fixture_f64(PREP_SANITY_DIR + "robust-" + String(k) + "-scale.f64", d)
    var want = fixture_f64(PREP_SANITY_DIR + "robust-" + String(k) + "-transform.f64", n * d)
    for c in range(d):  # small-loop(oracle: six features)
        if centering:
            all_close(arena[center + c], want_center[c], 1e-5, 1e-5, label + ": center_ feature " + String(c))
        all_close(arena[scale + c], want_scale[c], 1e-5, 1e-5, label + ": scale_ feature " + String(c))
    for i in range(n * d):  # small-loop(oracle: the scaled rows)
        all_close(arena[dest + i], want[i], 1e-4, 1e-4, label + ": transform word " + String(i))
    _ = arena^
    _ = program^


def test_robust_scaler() raises:
    check_cases("robust_scaler", 10, robust_case)


# ---- MaxAbsScaler (core/prep_member.mojo `_fit_maxabs`, no work region) ----
def maxabs_case(unit: Int) raises:
    var blocked = unit == 1
    var label = String("maxabs blocked ") + String(blocked)
    var n = SCALER_N
    var d = SCALER_D
    var x = fixture_f32(PREP_SANITY_DIR + "scalers-x1.f32", n * d)
    var blocks = (n + 2047) // 2048
    var host = n * d + ((2 * d + 5 * blocks * d) if blocked else 7 * d)
    var dest = host
    host += n * d
    var sc = n * d
    var arena = List[Float32](length=host, fill=0)
    for i in range(n * d):  # small-loop(fixture: the input rows into the arena)
        arena[i] = x[i]
    var program = List[Int32]()
    _ = _maxabs_fit_program(program, blocked, False, True, n, d, blocks, host, sc, dest)
    run(arena, program)
    var want_scale = fixture_f64(PREP_SANITY_DIR + "maxabs-scale.f64", d)
    var want = fixture_f64(PREP_SANITY_DIR + "maxabs-transform.f64", n * d)
    for c in range(d):  # small-loop(oracle: six features)
        all_close(arena[sc + c], want_scale[c], 1e-6, 0.0, label + ": scale_ feature " + String(c))
    for i in range(n * d):  # small-loop(oracle: the scaled rows)
        all_close(arena[dest + i], want[i], 1e-6, 1e-7, label + ": transform word " + String(i))
    _ = arena^
    _ = program^


def test_maxabs_scaler() raises:
    check_cases("maxabs_scaler", 2, maxabs_case)


# ---- Normalizer (the member's one-stage transform, host words, no output region) ----
def normalizer_case(unit: Int) raises:
    var names: List[String] = ["l1", "l2", "max"]
    var n = 50
    var d = 7
    var words = n * d
    var x = fixture_f32(PREP_SANITY_DIR + "normalizer-x.f32", words)
    var arena = List[Float32](length=2 * words, fill=0)
    for i in range(words):  # small-loop(fixture: the input rows into the arena)
        arena[i] = x[i]
    var program = List[Int32]()
    # the norm code of bindings/prep_stateless_native_door.mojo `_norm_code`: l1 0, l2 1, max 2
    prep_program_stage(program, OP_NORMALIZE, n, 0, n, d, unit, words)
    run(arena, program)
    var want = fixture_f64(PREP_SANITY_DIR + "normalizer-" + names[unit] + ".f64", words)
    for i in range(words):  # small-loop(oracle: the normalized rows)
        all_close(arena[words + i], want[i], 1e-6, 1e-7, "normalizer " + names[unit] + ": word " + String(i))
    _ = arena^
    _ = program^


def test_normalizer() raises:
    check_cases("normalizer", 3, normalizer_case)


# ---- Binarizer (the member's one-stage transform: X, the threshold word, the output) ----
def binarizer_case(unit: Int) raises:
    var thresholds: List[Float64] = [0.0, 0.3, -1.0]
    var n = 60
    var d = 5
    var words = n * d
    var x = fixture_f32(PREP_SANITY_DIR + "binarizer-x.f32", words)
    var out = words + 1
    var arena = List[Float32](length=out + words, fill=0)
    for i in range(words):  # small-loop(fixture: the input rows into the arena)
        arena[i] = x[i]
    arena[words] = Float32(thresholds[unit])
    var program = List[Int32]()
    prep_program_stage(program, OP_BINARIZE, words, 0, words, words, out)
    run(arena, program)
    var want = fixture_f32(PREP_SANITY_DIR + "binarizer-" + String(unit) + ".f32", words)
    for i in range(words):  # small-loop(oracle: exact words)
        require(arena[out + i] == want[i], "binarizer threshold " + String(thresholds[unit]) + ": word " + String(i)
            + " is " + String(arena[out + i]) + ", scikit-learn " + String(want[i]))
    _ = arena^
    _ = program^


def test_binarizer() raises:
    check_cases("binarizer", 3, binarizer_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

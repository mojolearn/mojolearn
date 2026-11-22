# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""MinMaxScaler fit, transform and inverse_transform against the host oracle, and the host-column StandardScaler and
MinMaxScaler statistics (lane py2mojo-A, 2026-11-22; ported from checks/minmax_scaler_smoke.py, the numeric half of
its `main`, and python/mojolearn/tests/test_cpu_training_e2.py `test_scalers_run_on_the_host_when_built`).

The scaler member (core/scaler_member.mojo) runs on the host column's scaler kernels and x_prep transport
(checks/native_tests/py2mojo_host_members.mojo `HostMember`) with the words the public UNPACK writes
([kind, with_mean or clip, with_std, lower, upper, identical]). Inputs and references are in `fixtures/prep_scalers/`
(tools/py2mojo_native_fixtures_b1.py): the original's five MinMax cases (3 x 3 and 1 x 3 literals, 513 x 7 and
259 x 7 uniform(-8, 8) of seed 20260910, the latter the original's transposed draw made contiguous, the 2 x 2 tiny
range) with its float32 `reference` (data_min_, data_max_, data_range_, scale_, min_ and x * scale_ + min_).

    units 0-9   case (5) x feature_range (0, 1), (-2, 3): the five rows to rtol 3e-6, atol 1e-7; the transform to
                rtol 3e-6, atol 2e-6; inverse_transform back to X to rtol 3e-6, atol 4e-6
    unit 10     clip=True on [[0, 4], [2, 4]]: [[-2, 3], [4, 6]] -> [[0, 0], [1, 1]] exactly
    unit 11     the signed zeros and the smallest subnormals: data_min_ and data_max_ keep their bits
    unit 12     test_cpu_training_e2 StandardScaler, 700 x 5 seed 3 with a 2.5 column: var_ 0, scale_ 1 and mean_ 2.5
                there, mean_ within 1e-4 of NumPy's, inverse_transform(transform(X[:64])) within 1e-4 of X
    unit 13     test_cpu_training_e2 MinMaxScaler: data_min_ / data_max_ the column extrema exactly, the varying
                columns transform into [0, 1], two fits transform to the same bytes

The fast-mode pass, dtype, no-alias, n_features_in_, pickling and the host-binding selection stay in the Python
files (public boundary). Every unit is independent and the case table runs across host cores.
"""
from std.memory import bitcast
from core.member_abi import MEMBER_OP_INVERSE_TRANSFORM
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from checks.native_tests.py2mojo_host_members import (
    HostMember, MEMBER_SCALER, PREP_SCALERS_DIR, all_close, rows_close, same_bits, take, take32,
)
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, fixture_f32, require


def minmax_params(lower: Float64, upper: Float64, clip: Bool) -> List[Float64]:
    var words: List[Float64] = [1.0, 1.0 if clip else 0.0, 0.0, lower, upper, 1.0]
    return words^


def minmax_case(unit: Int) raises:
    var k = unit // 2
    var b = unit % 2
    var rows: List[Int] = [3, 1, 513, 259, 2]
    var cols: List[Int] = [3, 3, 7, 7, 2]
    var n = rows[k]
    var d = cols[k]
    var lower = 0.0 if b == 0 else -2.0
    var upper = 1.0 if b == 0 else 3.0
    var label = String("minmax case ") + String(k) + " range " + String(lower) + ".." + String(upper)
    var x = fixture_f32(PREP_SCALERS_DIR + "mm-x" + String(k) + ".f32", n * d)
    var want = fixture_f64(PREP_SCALERS_DIR + "mm-want" + String(k) + "-" + String(b) + ".f64", 5 * d + n * d)
    var member = HostMember(MEMBER_SCALER, minmax_params(lower, upper, False))
    try:
        member.fit(x, n, d)
        rows_close(member.block(1), take(want, 0, 5 * d), 3e-6, 1e-7,
            label + ": data_min_, data_max_, data_range_, scale_, min_")
        var result = member.transform(x, n)
        rows_close(result, take(want, 5 * d, n * d), 3e-6, 2e-6, label + ": transform")
        var original = List[Float64](length=n * d, fill=0)
        for i in range(n * d):  # small-loop(oracle: X itself)
            original[i] = Float64(x[i])
        rows_close(member.apply(MEMBER_OP_INVERSE_TRANSFORM, result, n), original, 3e-6, 4e-6,
            label + ": inverse_transform")
    finally:
        member.free()


def clip_case() raises:
    var x: List[Float32] = [0, 4, 2, 4]
    var held: List[Float32] = [-2, 3, 4, 6]
    var want: List[Float32] = [0, 0, 1, 1]
    var member = HostMember(MEMBER_SCALER, minmax_params(0.0, 1.0, True))
    try:
        member.fit(x, 2, 2)
        same_bits(member.transform(held, 2), want, "clip=True transform")
    finally:
        member.free()


def extrema_case() raises:
    var tiny = bitcast[DType.float32](UInt32(1))
    var x: List[Float32] = [-0.0, -tiny, 0.0, tiny]
    var member = HostMember(MEMBER_SCALER, minmax_params(0.0, 1.0, False))
    try:
        member.fit(x, 2, 2)
        var stats = member.block(1)
        var low: List[Float32] = [-0.0, -tiny]
        var high: List[Float32] = [0.0, tiny]
        same_bits(take32(stats, 0, 2), low, "signed-zero extrema: data_min_")
        same_bits(take32(stats, 2, 2), high, "signed-zero extrema: data_max_")
    finally:
        member.free()


def e2_standard() raises:
    var n = 700
    var d = 5
    var x = fixture_f32(PREP_SCALERS_DIR + "e2-x.f32", n * d)
    var mean = fixture_f64(PREP_SCALERS_DIR + "e2-mean.f64", 4)
    var params: List[Float64] = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    var member = HostMember(MEMBER_SCALER, params^)
    try:
        member.fit(x, n, d)
        var stats = member.block(1)
        require(stats[d + 4] == 0.0 and stats[2 * d + 4] == 1.0 and stats[4] == Float32(2.5),
            "e2 StandardScaler constant column: mean_ " + String(stats[4]) + " var_ " + String(stats[d + 4])
            + " scale_ " + String(stats[2 * d + 4]))
        for c in range(4):  # small-loop(oracle: the varying features)
            require(abs(Float64(stats[c]) - mean[c]) < 1e-4, "e2 StandardScaler mean_ feature " + String(c) + ": "
                + String(stats[c]) + " NumPy " + String(mean[c]))
        var head = take32(x, 0, 64 * d)
        var back = member.apply(MEMBER_OP_INVERSE_TRANSFORM, member.transform(head, 64), 64)
        for i in range(64 * d):  # small-loop(oracle: the round trip)
            require(abs(back[i] - head[i]) < 1e-4, "e2 StandardScaler round trip word " + String(i) + ": "
                + String(back[i]) + " X " + String(head[i]))
    finally:
        member.free()


def e2_minmax() raises:
    var n = 700
    var d = 5
    var x = fixture_f32(PREP_SCALERS_DIR + "e2-x.f32", n * d)
    var head = take32(x, 0, 64 * d)
    var first = HostMember(MEMBER_SCALER, minmax_params(0.0, 1.0, False))
    var second = HostMember(MEMBER_SCALER, minmax_params(0.0, 1.0, False))
    try:
        first.fit(x, n, d)
        var stats = first.block(1)
        for c in range(d):  # small-loop(oracle: the column extrema)
            var low = x[c]
            var high = x[c]
            for i in range(n):  # small-loop(oracle: one column)
                low = min(low, x[i * d + c])
                high = max(high, x[i * d + c])
            require(stats[c] == low and stats[d + c] == high, "e2 MinMaxScaler feature " + String(c) + ": data_min_ "
                + String(stats[c]) + " / " + String(low) + ", data_max_ " + String(stats[d + c]) + " / " + String(high))
        var u = first.transform(head, 64)
        for i in range(64):  # small-loop(oracle: the varying columns)
            for c in range(4):  # small-loop(oracle: the varying columns)
                var value = u[i * d + c]
                require(value >= 0.0 and value <= 1.0, "e2 MinMaxScaler word " + String(i * d + c) + " = "
                    + String(value) + " outside [0, 1]")
        second.fit(x, n, d)
        same_bits(second.transform(head, 64), u, "e2 MinMaxScaler two host fits")
    finally:
        first.free()
        second.free()


def scalers_case(unit: Int) raises:
    if unit < 10:
        minmax_case(unit)
    elif unit == 10:
        clip_case()
    elif unit == 11:
        extrema_case()
    elif unit == 12:
        e2_standard()
    else:
        e2_minmax()


def test_minmax_scaler_smoke() raises:
    check_cases("minmax_scaler_smoke", 12, scalers_case)


def e2_case(unit: Int) raises:
    scalers_case(12 + unit)


def test_scalers_host_e2() raises:
    check_cases("scalers_host_e2", 2, e2_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""StandardScaler fit, transform and inverse_transform against NumPy's statistics (lane py2mojo-A, 2026-11-22;
ported from checks/standard_scaler_smoke.py, the numeric half of its `main`).

The StandardScaler member (core/scaler_member.mojo) runs on the host column's scaler kernels and x_prep transport
(checks/native_tests/py2mojo_host_members.mojo `HostMember`), with the parameter words the public UNPACK writes
([kind 0, with_mean, with_std, 0, 0, identical]). The inputs are the original's four cases, stored in
`fixtures/prep_scalers/` with NumPy's float64 mean and variance (tools/py2mojo_native_fixtures_b1.py):

    std-x0  3 x 3 literal (a constant column)      std-x1  1 x 3 literal
    std-x2  513 x 7 uniform(-8, 8), seed 20260910   std-x3  513 x 2: the constant 0.1 and linspace(-2, 2) columns of
                                                            the original's strided view, made contiguous

    units 0-15  case (4) x with_mean (2) x with_std (2): mean_ (when either flag), var_ and scale_ (with_std; scale 1
                where var is 0) to rtol 3e-5, atol 1e-6; the transform to rtol 3e-5, atol 2e-6; inverse_transform back
                to X to rtol 3e-5, atol 4e-6
    unit 16     with_mean = with_std = False: fit_transform returns the input words bit for bit (0, -0, the smallest
                subnormal, the largest float32)

The original's fast-mode pass, the None attributes, dtype refusal, pickling, no-alias, the process default mode and
the StandardScaler -> RandomForest pipeline stay in the Python file (public boundary and compositions). Every unit is
independent and the case table runs across host cores.
"""
from std.math import sqrt
from std.memory import bitcast
from core.member_abi import MEMBER_OP_TRANSFORM, MEMBER_OP_INVERSE_TRANSFORM
from checks.native_tests.tsa_metrics_fixture_words import fixture_f32
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from checks.native_tests.py2mojo_host_members import (
    HostMember, MEMBER_SCALER, PREP_SCALERS_DIR, all_close, rows_close, same_bits, require,
)
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases


def standard_params(with_mean: Bool, with_std: Bool) -> List[Float64]:
    var words: List[Float64] = [0.0, 1.0 if with_mean else 0.0, 1.0 if with_std else 0.0, 0.0, 0.0, 1.0]
    return words^


def smoke_case(unit: Int) raises:
    if unit == 16:
        bits_case()
        return
    var k = unit // 4
    var with_mean = (unit // 2) % 2 == 1
    var with_std = unit % 2 == 1
    var rows: List[Int] = [3, 1, 513, 513]
    var cols: List[Int] = [3, 3, 7, 2]
    var n = rows[k]
    var d = cols[k]
    var label = String("case ") + String(k) + " with_mean " + String(with_mean) + " with_std " + String(with_std)
    var x = fixture_f32(PREP_SCALERS_DIR + "std-x" + String(k) + ".f32", n * d)
    var mu = fixture_f64(PREP_SCALERS_DIR + "std-mu" + String(k) + ".f64", d)
    var var_ = fixture_f64(PREP_SCALERS_DIR + "std-var" + String(k) + ".f64", d)
    var member = HostMember(MEMBER_SCALER, standard_params(with_mean, with_std))
    try:
        member.fit(x, n, d)
        var stats = member.block(1)  # mean_, var_, scale_ rows
        for c in range(d):  # small-loop(oracle: the features)
            var scale = 1.0 if var_[c] == 0.0 else sqrt(var_[c])
            if with_mean or with_std:
                all_close(stats[c], mu[c], 3e-5, 1e-6, label + ": mean_ feature " + String(c))
            if with_std:
                all_close(stats[d + c], var_[c], 3e-5, 1e-6, label + ": var_ feature " + String(c))
                all_close(stats[2 * d + c], scale, 3e-5, 1e-6, label + ": scale_ feature " + String(c))
        var result = member.transform(x, n)
        var expected = List[Float64](length=n * d, fill=0)
        for i in range(n * d):  # small-loop(oracle: the original's float64 expectation)
            var c = i % d
            var value = Float64(x[i]) - mu[c] if with_mean else Float64(x[i])
            if with_std:
                value /= 1.0 if var_[c] == 0.0 else sqrt(var_[c])
            expected[i] = value
        rows_close(result, expected, 3e-5, 2e-6, label + ": transform")
        var back = member.apply(MEMBER_OP_INVERSE_TRANSFORM, result, n)
        var original = List[Float64](length=n * d, fill=0)
        for i in range(n * d):  # small-loop(oracle: X itself)
            original[i] = Float64(x[i])
        rows_close(back, original, 3e-5, 4e-6, label + ": inverse_transform")
    finally:
        member.free()


def bits_case() raises:
    var words: List[UInt32] = [0, 0x80000000, 1, 0x7F7FFFFF]
    var x = List[Float32](length=4, fill=0)
    for i in range(4):  # small-loop(fixture: four literal words)
        x[i] = bitcast[DType.float32](words[i])
    var member = HostMember(MEMBER_SCALER, standard_params(False, False))
    try:
        member.fit(x, 1, 4)
        same_bits(member.transform(x, 1), x, "with_mean=False with_std=False: transform")
    finally:
        member.free()


def test_standard_scaler_smoke() raises:
    check_cases("standard_scaler_smoke", 17, smoke_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

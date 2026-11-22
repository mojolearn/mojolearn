# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""PowerTransformer lambdas, transform and inverse_transform against scikit-learn (lane py2mojo-A, 2026-11-22;
ported from python/mojolearn/tests/test_x_prep_power.py `test_power`).

The PowerTransformer member (core/prep_member.mojo kind 6: the stateless door's `_power_program`) runs over the host
column's x_prep transport (checks/native_tests/py2mojo_host_members.mojo `HostMember`) with the words `_power_admit`
writes on the host identical image ([6, method, standardize, x_prep_idn_fam, 0, host column 1, identical 1, R3 work
1]). Inputs: the original's 600 x 4 columns (normal, exponential(2), gamma(2, 1.5), lognormal(0, 0.7)) for seeds 0
(fit) and 1 (held out), |X| + 0.1 for Box-Cox, with scikit-learn 1.9.1's lambdas_ and transform, in
`fixtures/prep_members/` (tools/py2mojo_native_fixtures_b1.py).

    units 0-2   {}, standardize=False, method='box-cox': lambdas_ to rtol = atol = 2e-3 (a search tolerance), the
                held-out transform to 5e-3; inverse_transform of that output with one NaN against the reference
                inverse at OUR lambdas_ and standardizer (the original sets them into scikit-learn; here the same
                formulas, sklearn/preprocessing/_data.py `_yeo_johnson_inverse_transform` and scipy `inv_boxcox`,
                in float64): the NaN cell kept, rtol = atol = 2e-4, and the finite cells back to X to 2e-3

The public refusals stay in the Python file. Every unit is independent and the case table runs across host cores.
"""
from std.math import exp, isnan
from std.memory import bitcast
from core.member_abi import MEMBER_OP_INVERSE_TRANSFORM
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from checks.native_tests.py2mojo_host_members import (
    HostMember, MEMBER_PREP, PREP_MEMBERS_DIR, host_idn_fam, all_close, rows_close,
)
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, fixture_f32, require

comptime N = 600
comptime D = 4
comptime SPACING_ONE = 2.220446049250313e-16


def yeo_johnson_inverse(x: Float64, lmbda: Float64) -> Float64:
    # NumPy propagates NaN through np.power; Mojo's Float64 `**` returned inf for a NaN base on cpc1 (2026-11-22),
    # so the oracle returns NaN first.
    if isnan(x):
        return x
    if x >= 0.0:
        if abs(lmbda) < SPACING_ONE:
            return exp(x) - 1.0
        return (x * lmbda + 1.0) ** (1.0 / lmbda) - 1.0
    if abs(lmbda - 2.0) > SPACING_ONE:
        return 1.0 - (-(2.0 - lmbda) * x + 1.0) ** (1.0 / (2.0 - lmbda))
    return 1.0 - exp(-x)


def box_cox_inverse(x: Float64, lmbda: Float64) -> Float64:
    if isnan(x):
        return x
    if lmbda == 0.0:
        return exp(x)
    return (x * lmbda + 1.0) ** (1.0 / lmbda)


def power_case(unit: Int) raises:
    var box_cox = unit == 2
    var standardize = unit != 1
    var params: List[Float64] = [6.0, 1.0 if box_cox else 0.0, 1.0 if standardize else 0.0, Float64(host_idn_fam()),
                                 0.0, 1.0, 1.0, 1.0]
    var label = String("PowerTransformer case ") + String(unit)
    var key = PREP_MEMBERS_DIR + "pw-" + String(unit)
    var x = fixture_f32(key + "-x.f32", N * D)
    var xh = fixture_f32(key + "-xh.f32", N * D)
    var member = HostMember(MEMBER_PREP, params^)
    try:
        member.fit(x, N, D)
        var lambdas = member.block(3)
        rows_close(lambdas, fixture_f64(key + "-lambdas.f64", D), 2e-3, 2e-3, label + ": lambdas_")
        var z = member.transform(xh, N)
        rows_close(z, fixture_f64(key + "-transform.f64", N * D), 5e-3, 5e-3, label + ": transform")
        var mean = member.block(1)
        var scale = member.block(2)
        z[3 * D] = bitcast[DType.float32](UInt32(0x7FC00000))
        var a = member.apply(MEMBER_OP_INVERSE_TRANSFORM, z, N)
        for i in range(N * D):  # small-loop(oracle: the reference inverse at our fitted words)
            var c = i % D
            var value = Float64(z[i])
            if standardize:
                value = value * Float64(scale[c]) + Float64(mean[c])
            var lmbda = Float64(lambdas[c])
            var b = box_cox_inverse(value, lmbda) if box_cox else yeo_johnson_inverse(value, lmbda)
            require(isnan(a[i]) == isnan(b), label + ": inverse NaN pattern word " + String(i) + " " + String(a[i])
                + " reference " + String(b))
            if not isnan(b):
                all_close(a[i], b, 2e-4, 2e-4, label + ": inverse_transform word " + String(i))
            if not isnan(z[i]):
                all_close(a[i], Float64(xh[i]), 2e-3, 2e-3, label + ": inverse_transform back to X word " + String(i))
    finally:
        member.free()


def test_power() raises:
    check_cases("power", 3, power_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

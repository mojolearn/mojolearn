# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""QuantileTransformer quantiles, transform and inverse_transform against scikit-learn (lane py2mojo-A, 2026-11-22;
ported from python/mojolearn/tests/test_x_prep_quantile.py `test_quantile`).

The QuantileTransformer member (core/prep_member.mojo kind 5: the stateless door's `_quantile_program`) runs over the
host column's x_prep transport (checks/native_tests/py2mojo_host_members.mojo `HostMember`) with the words
`_quantile_admit` writes for the defaults (subsample 10000 above the 700 rows: no subsample; random_state None: seed
0; x_prep_idn_fam of the host identical image; R3 work 1). Inputs: the original's `_data` (700 x 4: normal, rounded
2x, NaN every 11th row of feature 2, exp) for seeds 0 (fit) and 1 (held out), the inverse input Z it built from the
reference's forward output (the bounds exactly, beyond them, a fixed 0.37, NaN), with scikit-learn 1.9.1's
quantiles_, transform and inverse_transform, in `fixtures/prep_members/` (tools/py2mojo_native_fixtures_b1.py).

    units 0-1   n_quantiles=100 uniform, n_quantiles=50 normal: quantiles_ to rtol = atol = 1e-5, the held-out
                transform and the inverse_transform of Z to rtol 1e-3, atol 2e-4, NaN where scikit-learn has NaN

Every unit is independent and the case table runs across host cores.
"""
from core.member_abi import MEMBER_OP_INVERSE_TRANSFORM
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from checks.native_tests.py2mojo_host_members import HostMember, MEMBER_PREP, PREP_MEMBERS_DIR, host_idn_fam, rows_close
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, fixture_f32

comptime N = 700
comptime D = 4


def quantile_case(unit: Int) raises:
    var nq = 100 if unit == 0 else 50
    var params: List[Float64] = [5.0, Float64(unit), Float64(nq), 10000.0, 0.0, 0.0, Float64(host_idn_fam()), 1.0]
    var label = String("QuantileTransformer case ") + String(unit)
    var key = PREP_MEMBERS_DIR + "qt-" + String(unit)
    var x = fixture_f32(PREP_MEMBERS_DIR + "qt-x.f32", N * D)
    var xh = fixture_f32(PREP_MEMBERS_DIR + "qt-xh.f32", N * D)
    var member = HostMember(MEMBER_PREP, params^)
    try:
        member.fit(x, N, D)
        rows_close(member.block(3), fixture_f64(key + "-quantiles.f64", nq * D), 1e-5, 1e-5, label + ": quantiles_")
        rows_close(member.transform(xh, N), fixture_f64(key + "-transform.f64", N * D), 1e-3, 2e-4,
            label + ": transform")
        var z = fixture_f32(key + "-z.f32", N * D)
        rows_close(member.apply(MEMBER_OP_INVERSE_TRANSFORM, z, N), fixture_f64(key + "-inverse.f64", N * D), 1e-3,
            2e-4, label + ": inverse_transform")
    finally:
        member.free()


def test_quantile() raises:
    check_cases("quantile", 2, quantile_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

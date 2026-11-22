# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""PolynomialFeatures term table and transform against scikit-learn (lane py2mojo-A, 2026-11-22; ported from
python/mojolearn/tests/test_x_prep_polynomial.py `test_poly`, its numeric half).

The PolynomialFeatures member (core/prep_member.mojo kind 4: the stateless door's term table and OP_POLY program)
runs over the host column's x_prep transport (checks/native_tests/py2mojo_host_members.mojo `HostMember`) with the
words `_poly_admit` writes ([4, min_degree, max_degree, interaction_only, include_bias]). Input: the original's
40 x 4 standard normal draw of seed 0, with scikit-learn 1.9.1's powers_ and transform, in `fixtures/prep_members/`
(tools/py2mojo_native_fixtures_b1.py).

    units 0-2   {}, degree=3, degree=(2, 3) interaction_only include_bias=False: powers_ (rebuilt from the fitted
                term index and start blocks, as PACK publishes it) exactly, the transform to rtol 1e-5, atol 1e-6

order='F' (the public array's memory order only; the member writes row-major) stays in the Python file. Every unit is
independent and the case table runs across host cores.
"""
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from checks.native_tests.py2mojo_host_members import HostMember, MEMBER_PREP, PREP_MEMBERS_DIR, rows_close
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, fixture_f32, require

comptime N = 40
comptime D = 4


def poly_case(unit: Int) raises:
    var lows: List[Float64] = [0.0, 0.0, 2.0]
    var highs: List[Float64] = [2.0, 3.0, 3.0]
    var params: List[Float64] = [4.0, lows[unit], highs[unit], 1.0 if unit == 2 else 0.0, 0.0 if unit == 2 else 1.0]
    var label = String("PolynomialFeatures case ") + String(unit)
    var terms = Int(fixture_f64(PREP_MEMBERS_DIR + "poly-" + String(unit) + "-width.f64", 1)[0])
    var want = fixture_f64(PREP_MEMBERS_DIR + "poly-" + String(unit) + "-powers.f64", terms * D)
    var x = fixture_f32(PREP_MEMBERS_DIR + "poly-x.f32", N * D)
    var member = HostMember(MEMBER_PREP, params^)
    try:
        member.fit(x, N, D)
        var scalars = member.scalars()
        require(Int(scalars[1]) == terms, label + ": n_output_features_ " + String(scalars[1]) + " expected "
            + String(terms))
        var index = member.block(1)
        var start = member.block(2)
        for t in range(terms):  # small-loop(oracle: the term table)
            var powers = List[Int](length=D, fill=0)
            for i in range(Int(start[t]), Int(start[t + 1])):  # small-loop(oracle: one term's factors)
                powers[Int(index[i])] += 1
            for c in range(D):  # small-loop(oracle: one powers_ row)
                require(Float64(powers[c]) == want[t * D + c], label + ": powers_[" + String(t) + ", " + String(c)
                    + "] " + String(powers[c]) + " expected " + String(want[t * D + c]))
        rows_close(member.transform(x, N), fixture_f64(PREP_MEMBERS_DIR + "poly-" + String(unit) + "-transform.f64",
            N * terms), 1e-5, 1e-6, label + ": transform")
    finally:
        member.free()


def test_poly() raises:
    check_cases("poly", 3, poly_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""SimpleImputer statistics and transform against scikit-learn (lane py2mojo-A, 2026-11-22; ported from
python/mojolearn/tests/test_x_prep_imputer.py `test_strategies` and `test_numeric_missing`).

The SimpleImputer member (core/prep_member.mojo kind 7, the encode door's typed stages of
bindings/prep_encode_imputer_run.mojo) runs over the host column's x_prep transport
(checks/native_tests/py2mojo_host_members.mojo `HostMember`) with the parameter words the public UNPACK writes
(providers/python/member_prep_boundary_door.mojo `_imputer_admit`). Inputs: the original's `_data` (203 x 5, values
rounded to integers so the median and most_frequent tie, 20% NaN, column 3 all NaN) for seeds 0 (fit), 1 (held out)
and 2 (NaN replaced by the numeric missing value -1), with scikit-learn 1.9.1's statistics_ and transforms, in
`fixtures/prep_members/` (tools/py2mojo_native_fixtures_b1.py). Tolerance rtol 1e-5, atol 1e-6 (NaN only against
NaN: 1.9 leaves an all-missing column's statistic NaN, constant strategy included).

    units 0-5   strategy mean, median, most_frequent, constant (fill_value -3), median with keep_empty_features, mean
                with add_indicator: statistics_ and the transform of the held-out rows (its width included)
    unit 6      missing_values = -1.0, strategy mean: the transform

The public refusals and output types stay in the Python file. Every unit is independent and the case table runs
across host cores.
"""
from core.member_abi import MEMBER_OP_TRANSFORM
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from checks.native_tests.py2mojo_host_members import (
    HostMember, MEMBER_PREP, PREP_MEMBERS_DIR, host_idn_fam, rows_close, require,
)
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, fixture_f32

comptime N = 203
comptime D = 5


def imputer_params(strategy: Int, mark: Bool, value: Float64, fill: Float64, keep_empty: Bool,
                   indicator: Bool) -> List[Float64]:
    """`_imputer_admit`'s words on the host identical image (no radix median: that is the fast Metal route)."""
    var blocked = 1.0 if (host_idn_fam() & 1) != 0 else 0.0
    var words: List[Float64] = [7.0, Float64(strategy), 1.0 if mark else 0.0, value, 0.0, blocked, fill,
                                1.0 if keep_empty else 0.0, 1.0 if indicator else 0.0, 1.0]
    return words^


def imputer_case(unit: Int) raises:
    var label = String("SimpleImputer case ") + String(unit)
    var name = String("imp-num") if unit == 6 else String("imp-") + String(unit)
    var width = Int(fixture_f64(PREP_MEMBERS_DIR + name + "-width.f64", 1)[0])
    # (strategy, fill_value, keep_empty_features, add_indicator) of the original's keyword sets
    var strategies: List[Int] = [0, 1, 2, 3, 1, 0, 0]
    var fills: List[Float64] = [0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0]
    var member = HostMember(MEMBER_PREP, imputer_params(strategies[unit], unit == 6, -1.0 if unit == 6 else 0.0,
        fills[unit], unit == 4, unit == 5))
    try:
        if unit == 6:
            var x = fixture_f32(PREP_MEMBERS_DIR + "imp-num-x.f32", N * D)
            member.fit(x, N, D)
            require(member.width(MEMBER_OP_TRANSFORM) == width, label + ": width " + String(member.width(MEMBER_OP_TRANSFORM)))
            rows_close(member.transform(x, N), fixture_f64(PREP_MEMBERS_DIR + "imp-num-transform.f64", N * width),
                1e-5, 1e-6, label + ": transform")
            return
        var x = fixture_f32(PREP_MEMBERS_DIR + "imp-x.f32", N * D)
        var xh = fixture_f32(PREP_MEMBERS_DIR + "imp-xh.f32", N * D)
        member.fit(x, N, D)
        rows_close(member.block(1), fixture_f64(PREP_MEMBERS_DIR + name + "-stats.f64", D), 1e-5, 1e-6,
            label + ": statistics_")
        require(member.width(MEMBER_OP_TRANSFORM) == width, label + ": width " + String(member.width(MEMBER_OP_TRANSFORM))
            + " expected " + String(width))
        rows_close(member.transform(xh, N), fixture_f64(PREP_MEMBERS_DIR + name + "-transform.f64", N * width), 1e-5,
            1e-6, label + ": transform")
    finally:
        member.free()


def numeric_case(unit: Int) raises:
    imputer_case(6 + unit)


def test_simple_imputer_strategies() raises:
    check_cases("simple_imputer_strategies", 6, imputer_case)


def test_simple_imputer_numeric_missing() raises:
    check_cases("simple_imputer_numeric_missing", 1, numeric_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

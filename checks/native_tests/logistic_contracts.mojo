# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Native Logistic ABI admission; each fixture owns its parallel worker data.

These check metadata/refusal contracts, not kernel arithmetic or speed.
Payload span sizes remain the admitted caller's responsibility.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require as _require
from core.logistic_native_api import qn_fold_f32_ptr, qn_native_spans
from checks.native_tests.logistic_descriptor_fixtures import (
    logistic_driver_fixture, logistic_fold_fixture, logistic_fixed_fixture, logistic_concurrent_fixture,
    logistic_route_fixture, LOGISTIC_ROUTE_FIXTURES,
)


def _fold_pointer(unit: Int) raises:
    if unit == 0:
        var owner: List[Float32] = [1.25, -2.5]
        var pointer = qn_fold_f32_ptr(Int(owner.unsafe_ptr()))
        _require(pointer[0] == 1.25 and pointer[1] == -2.5, "accepted pointer retains its addressed words")
        _ = owner^
        return
    var invalid: List[Int] = [0, -4, -1, 1, 2, 3, 6]
    var message = String()
    try:
        _ = qn_fold_f32_ptr(invalid[unit - 1])
    except problem:
        message = String(problem)
    _require(message == "pca_transform_fold: pointers must be positive and float32-aligned",
        "resident fold must preserve its existing null/negative/misaligned refusal")


def _valid_span(operation: Int) raises:
    # Published ABI v1 spans: fit; four queries; two links; log; three metadata
    # operations. Counts are independent protocol fixtures, not product output.
    var address_counts: List[Int] = [6, 4, 4, 4, 4, 2, 2, 2, 0, 0, 0]
    var integer_counts: List[Int] = [9, 5, 5, 5, 5, 2, 2, 1, 0, 0, 0]
    var float_counts: List[Int] = [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    var addresses = List[Int](length=address_counts[operation], fill=0)
    var integers = List[Int](length=integer_counts[operation], fill=0)
    var floats = List[Float64](length=float_counts[operation], fill=0)
    var result: List[Int] = [37]
    qn_native_spans(operation,
        Int(addresses.unsafe_ptr()) if len(addresses) else 0, len(addresses),
        Int(integers.unsafe_ptr()) if len(integers) else 0, len(integers),
        Int(floats.unsafe_ptr()) if len(floats) else 0, len(floats), Int(result.unsafe_ptr()))
    _require(result[0] == 37, "span admission must not write the result")
    _ = addresses^
    _ = integers^
    _ = floats^


def _invalid_span(unit: Int) raises:
    # Every refusal occurs before metadata dereference: unmapped aligned
    # sentinels are intentional and may never become payload addresses.
    var operation = 0
    var aa = 8
    var na = 6
    var pp = 8
    var np = 9
    var ff = 8
    var nf = 5
    var result = 8
    var expected = String("logistic native ABI: invalid request spans")
    if unit == 0 or unit == 1:
        operation = -1 if unit == 0 else 11
        expected = "logistic native ABI: unknown operation"
    elif unit == 2:
        na = -1
    elif unit == 3:
        na = 5
    elif unit == 4:
        np = 8
    elif unit == 5:
        nf = 4
    elif unit == 6:
        result = 0
    elif unit == 7:
        result = -8
    elif unit == 8:
        result = 9
    else:
        expected = "logistic native ABI: invalid metadata address"
        if unit == 9:
            aa = 0
        elif unit == 10:
            aa = -8
        elif unit == 11:
            aa = 9
        elif unit == 12:
            pp = 0
        elif unit == 13:
            pp = 9
        elif unit == 14:
            ff = 0
        else:
            ff = 9
    var message = String()
    try:
        qn_native_spans(operation, aa, na, pp, np, ff, nf, result)
    except problem:
        message = String(problem)
    _require(message == expected, "native request must refuse malformed spans before dereference")


def test_fold_pointer() raises:
    check_cases("fold_pointer", 8, _fold_pointer)


def test_valid_span() raises:
    check_cases("valid_span", 11, _valid_span)


def test_invalid_span() raises:
    check_cases("invalid_span", 16, _invalid_span)


def _driver_fixture(operation: Int) raises:
    logistic_driver_fixture(operation)


def test_logistic_driver_fixture() raises:
    check_cases("logistic_driver_fixture", 8, _driver_fixture)


def test_logistic_fold_fixture() raises:
    check_cases("logistic_fold_fixture", 5, logistic_fold_fixture)


def test_logistic_fixed_fixture() raises:
    check_cases("logistic_fixed_fixture", 6, logistic_fixed_fixture)


def test_logistic_concurrent_fixture() raises:
    check_cases("logistic_concurrent_fixture", 64, logistic_concurrent_fixture)


def test_logistic_route_fixture() raises:
    check_cases("logistic_route_fixture", LOGISTIC_ROUTE_FIXTURES, logistic_route_fixture)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

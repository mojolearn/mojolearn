# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Prescribed public option table and native empty-core label fill.

The table is the original public metric/method contract, including admitted
RBC/precomputed before fit selects brute. Independent fixtures and output
word checks run in parallel. No fit kernel or Python object is involved.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases
from core.host_parallel import host_parallelize
from core.dbscan_native_api import dbscan_empty_labels
from core.dbscan_native_state import dbscan_metric_code, dbscan_method_code, dbscan_method_accepts

comptime DBSCAN_OPTIONS_FIXTURES = 41


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def fixture(unit: Int) raises:
    if unit < 9:
        var names: List[String] = ["euclidean", "l2", "manhattan", "l1", "cityblock", "cosine", "precomputed", "", "Euclidean"]
        var codes: List[Int] = [0, 0, 1, 1, 1, 2, 3, -1, -1]
        require(dbscan_metric_code(names[unit]) == codes[unit], "DBSCAN original metric aliases and unknown names")
    elif unit < 12:
        var names: List[String] = ["brute", "rbc", "RBC"]
        var codes: List[Int] = [0, 1, -1]
        require(dbscan_method_code(names[unit - 9]) == codes[unit - 9], "DBSCAN original method names")
    elif unit < 36:
        var row = (unit - 12) // 6
        var column = (unit - 12) % 6
        # Methods [-1, brute, RBC, 2], metrics [-1, L2, L1, cosine, precomputed, 4].
        # Prescribed public admission table; precomputed fit later selects brute.
        var admitted_masks: List[Int] = [0, 0b011110, 0b010010, 0]
        var expected = ((admitted_masks[row] >> column) & 1) != 0
        require(dbscan_method_accepts(row - 1, column - 1) == expected,
            "DBSCAN complete original method/metric admission table")
    elif unit < 40:
        var rows = 0 if unit == 36 else (1 if unit == 37 else (17 if unit == 38 else 1025))
        var values = List[Int32](length=rows + 2, fill=987654)
        var failures = List[Int](length=rows + 2, fill=0)
        var vp = values.unsafe_ptr()
        var fp = failures.unsafe_ptr()
        dbscan_empty_labels(Int(vp) + 4, rows)
        def cell(index: Int) {imm vp, imm fp, imm rows}:
            var expected = Int32(987654 if index == 0 or index == rows + 1 else -1)
            fp.unsafe_store(index, Int(vp[index] != expected))
        host_parallelize(cell, rows + 2)
        for failure in failures:
            require(failure == 0, "DBSCAN empty-core prediction fills every noise label and retains guards")
        _ = values^
    else:
        dbscan_empty_labels(0, 0)


def test_dbscan_options() raises:
    check_cases("dbscan_options", DBSCAN_OPTIONS_FIXTURES, fixture)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

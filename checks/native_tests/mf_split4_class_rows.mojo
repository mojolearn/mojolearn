# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The class-weight tail of a META member's tree base (lane mf-split4, core/ensemble_member.mojo
`_tree_class_rows`): the float32 row weights the Bagging kind's batched forest and the AdaBoost kind's rounds
receive, against an independent oracle of the forest door's rows (bindings/trees_public_inputs_door.mojo
`trees_class_rows_only`: 'balanced' = n / (k * count_c) per class rounded to float32, a dict's weights rounded to
float32, gathered by the row's code; every weight 1 = no rows).

Fixture: 9 rows, codes [0, 2, 1, 0, 2, 2, 1, 0, 2] (counts 3, 2, 4), k = 3; no generator, no pack.

Fail-proof edits (each makes the named case fail): in `_tree_class_rows` gather `plan.words` by `r % k` instead of
the code (cases 1, 2); return the rows when `plan.all_one` (case 3: answers non-zero); drop the K < 0 refusal
(case 4: no raise); read the dict weights from `at + META_CW_HEAD + 1` (case 2).

"""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from core.ensemble_member import (
    _tree_class_rows, META_CW_NONE, META_CW_BALANCED, META_CW_DICT,
)


def _codes() -> List[Int32]:
    var codes: List[Int32] = [0, 2, 1, 0, 2, 2, 1, 0, 2]
    return codes^


def _tail(mode: Int, k: Int, all_one: Bool, weights: List[Float64]) -> List[Float64]:
    var words: List[Float64] = [7.0, 7.0]  # two settings words before the tail (at = 2)
    words.append(Float64(mode))
    words.append(Float64(k))
    words.append(1.0 if all_one else 0.0)
    for i in range(len(weights)):
        words.append(weights[i])
    return words^


def test_balanced_rows() raises:
    """Case 1: 'balanced' = 9 / (3 * count) per class, float32, gathered by code."""
    var codes = _codes()
    var rows = List[Float32]()
    var at = _tree_class_rows(_tail(META_CW_BALANCED, 0, False, List[Float64]()), 2, Int(codes.unsafe_ptr()), 9, 3,
                              rows)
    assert_true(at != 0)
    var counts: List[Int] = [3, 2, 4]
    for r in range(9):
        var c = Int(codes[r])
        assert_equal(rows[r], Float32(Float64(9) / (Float64(3) * Float64(counts[c]))))
    _ = len(codes)


def test_dict_rows() raises:
    """Case 2: a dict's weights (codes 0 and 2 named, 1 not), float32, gathered by code."""
    var codes = _codes()
    var rows = List[Float32]()
    var weights: List[Float64] = [0.5, 1.0, 1.8]
    var at = _tree_class_rows(_tail(META_CW_DICT, 3, False, weights), 2, Int(codes.unsafe_ptr()), 9, 3, rows)
    assert_true(at != 0)
    for r in range(9):
        assert_equal(rows[r], Float32(weights[Int(codes[r])]))
    _ = len(codes)


def test_all_one_and_none() raises:
    """Case 3: every weight 1 hands no rows (address 0), as the door hands the fit no weights; no tail too."""
    var codes = _codes()
    var rows = List[Float32]()
    var ones: List[Float64] = [1.0, 1.0, 1.0]
    assert_equal(_tree_class_rows(_tail(META_CW_DICT, 3, True, ones), 2, Int(codes.unsafe_ptr()), 9, 3, rows), 0)
    assert_equal(_tree_class_rows(_tail(META_CW_NONE, 0, False, List[Float64]()), 2, Int(codes.unsafe_ptr()), 9, 3,
                                  rows), 0)
    var short: List[Float64] = [7.0, 7.0]
    assert_equal(_tree_class_rows(short, 2, Int(codes.unsafe_ptr()), 9, 3, rows), 0)
    _ = len(codes)


def test_pending_and_size_refusals() raises:
    """Case 4: a dict whose labels were never read (K = -1) and a table of the wrong size are refused."""
    var codes = _codes()
    var rows = List[Float32]()
    with assert_raises(contains="hands over its labels"):
        _ = _tree_class_rows(_tail(META_CW_DICT, -1, False, List[Float64]()), 2, Int(codes.unsafe_ptr()), 9, 3, rows)
    var two: List[Float64] = [0.5, 1.8]
    with assert_raises(contains="class_weight table has 2 entries for 3 classes"):
        _ = _tree_class_rows(_tail(META_CW_DICT, 2, False, two), 2, Int(codes.unsafe_ptr()), 9, 3, rows)
    _ = len(codes)


# TOMBSTONE: cases 5-7 (`_bag_class_rows`, the per-draw class rows of a Bagging unit) deleted 2026-11-24 by
# lane/w3f3-h with the function: a unit is now fitted on the whole y (scikit-learn 1.9.1 _bagging.py:164-172), its
# class rows `_tree_class_rows` over y (cases 1-4) times its draw's counts (checks/native_tests/w3f3_bag_draws.mojo);
# recoverable at a5b8bf2008.


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

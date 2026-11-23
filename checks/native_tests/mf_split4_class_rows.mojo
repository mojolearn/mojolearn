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

Lane mf-split4b, the per-member route (`_bag_class_rows`: each unit's rows over its own draw, as the tree's fit
door makes them for the rows the door's loop gives it): cases 5-7. Oracle: 'balanced' over the draw = m / (k' *
count_c) with m the drawn rows and k' the classes the draw holds; a dict's float32 weight of the row's class;
times the caller's float32 weight. Fail-proof edits: count over all n rows instead of the draw in `_count` (case 5);
drop the named-class refusal (case 6: no raise); multiply by `given[i]` instead of `given[r]` (case 7).
"""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from core.ensemble_member import (
    _tree_class_rows, _bag_class_rows, _held_and_codes, META_CW_NONE, META_CW_BALANCED, META_CW_DICT,
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


def _units(codes: List[Int32], draws: List[List[Int32]], mut held: List[List[Int32]],
           mut recoded: List[List[Int32]], mut at: List[Int], mut count: List[Int]):
    for u in range(len(draws)):
        at.append(Int(draws[u].unsafe_ptr()))
        count.append(len(draws[u]))
    _held_and_codes(Int(codes.unsafe_ptr()), 3, 9, at, count, held, recoded)


def test_per_member_balanced() raises:
    """Case 5: unit 0 draws rows 0-3 (codes 0, 2, 1, 0: counts 2, 1, 1 of m = 4): 4 / (3 * count) at each drawn row;
    unit 1 draws rows 1, 4, 5, 8 (only class 2): one class, ratio 1, so it reads the caller's weights (none: 0)."""
    var codes = _codes()
    var d0: List[Int32] = [0, 1, 2, 3]
    var d1: List[Int32] = [1, 4, 5, 8]
    var draws = List[List[Int32]]()
    draws.append(d0^)
    draws.append(d1^)
    var held = List[List[Int32]]()
    var recoded = List[List[Int32]]()
    var at = List[Int]()
    var count = List[Int]()
    _units(codes, draws, held, recoded, at, count)
    var unit_w = List[List[Float32]]()
    var unit_at = List[Int](length=2, fill=0)
    _bag_class_rows(_tail(META_CW_BALANCED, 0, False, List[Float64]()), 2, Int(codes.unsafe_ptr()), 3, 9, 0, at,
                    count, held, recoded, unit_w, unit_at)
    assert_true(unit_at[0] != 0)
    assert_equal(unit_at[1], 0)
    var counts: List[Int] = [2, 1, 1]
    for i in range(4):
        var r = Int(draws[0][i])
        assert_equal(unit_w[0][r], Float32(Float64(4) / (Float64(3) * Float64(counts[Int(codes[r])]))))
    _ = len(codes)
    _ = len(draws)


def _named_tail(weights: List[Float64], named: List[Float64]) -> List[Float64]:
    var words = _tail(META_CW_DICT, len(weights), False, weights)
    for c in range(len(named)):
        words.append(named[c])
    return words^


def test_per_member_dict_absent_label() raises:
    """Case 6: a dict naming class 1 and a draw without class 1 (rows 1, 4, 5, 8: only class 2): the tree door's
    refusal."""
    var codes = _codes()
    var d0: List[Int32] = [1, 4, 5, 8]
    var draws = List[List[Int32]]()
    draws.append(d0^)
    var held = List[List[Int32]]()
    var recoded = List[List[Int32]]()
    var at = List[Int]()
    var count = List[Int]()
    _units(codes, draws, held, recoded, at, count)
    var unit_w = List[List[Float32]]()
    var unit_at = List[Int](length=1, fill=0)
    var weights: List[Float64] = [1.0, 3.0, 1.0]
    var named: List[Float64] = [0.0, 1.0, 0.0]
    with assert_raises(contains="class_weight contains labels absent from y"):
        _bag_class_rows(_named_tail(weights, named), 2, Int(codes.unsafe_ptr()), 3, 9, 0, at, count, held, recoded,
                        unit_w, unit_at)
    _ = len(codes)
    _ = len(draws)


def test_per_member_dict_weighted() raises:
    """Case 7: a dict over every class (row weights 0.5, 1.0, 1.8 by class) times the caller's weights r + 1, at the
    drawn rows 8, 0, 6, 6 (codes 2, 0, 1: every class held)."""
    var codes = _codes()
    var d0: List[Int32] = [8, 0, 6, 6]
    var draws = List[List[Int32]]()
    draws.append(d0^)
    var held = List[List[Int32]]()
    var recoded = List[List[Int32]]()
    var at = List[Int]()
    var count = List[Int]()
    _units(codes, draws, held, recoded, at, count)
    var given = List[Float32](length=9, fill=0.0)
    for r in range(9):
        given[r] = Float32(r + 1)
    var unit_w = List[List[Float32]]()
    var unit_at = List[Int](length=1, fill=Int(given.unsafe_ptr()))
    var weights: List[Float64] = [0.5, 1.0, 1.8]
    var named: List[Float64] = [1.0, 0.0, 1.0]
    _bag_class_rows(_named_tail(weights, named), 2, Int(codes.unsafe_ptr()), 3, 9, Int(given.unsafe_ptr()), at, count,
                    held, recoded, unit_w, unit_at)
    assert_true(unit_at[0] != Int(given.unsafe_ptr()))
    for i in range(4):
        var r = Int(draws[0][i])
        assert_equal(unit_w[0][r], Float32(weights[Int(codes[r])]) * given[r])
    _ = len(codes)
    _ = len(draws)
    _ = len(given)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

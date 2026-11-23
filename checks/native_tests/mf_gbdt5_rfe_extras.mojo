# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""A member fit's extra inputs (lane mf-gbdt5, 2026-11-23), in a process with no Python: what RFE's member fits
read when RFE.fit(X, y, **fit_params) is given to a Gbdt member (core/member_set_run.mojo `MemberFitExtras`,
`member_eval_columns`, `member_fit_extras_apply`; bindings/prep_select_run.mojo `rfe_member_source`).

Cases (one row each, run across the host cores by `check_cases`):
  0 the held-out columns of a round (`member_eval_columns`): a 7 x 5 eval block restricted to int32 ids [3, 0, 4]
    equals the gather done here element by element (row-major, the ids' order), as scikit-learn's RFE fits the
    round on `X[:, features]`; int64 ids the same; ids [0 .. 4] in order copy nothing (False, `dst` untouched);
    an id outside the columns raises the member ABI's words. Fail-proof: read `src[i * cols + j]` in
    `_eval_columns_span` (the gathered words differ), or drop the `c != j` test (the identity is reported for
    [3, 0, 4]: no copy).
  1 the same gather split across the host cores (rows x ids above `_SET_SPLIT_WORDS`: 300000 x 4 words) equals
    the element-by-element gather. Fail-proof: start each part at `part * per + 1`.
  2 the views a member FIT call gets (`member_fit_extras_apply`): the weights (rows x 1 float32), the group ids
    (rows x 1 int32), the pairs (pairs x 2 int32) and their weights, the held-out X restricted to the round's ids
    (its address is the gathered block, `count` wide) and its targets in their dtype; with every column in
    order the held-out X is the caller's own block; with nothing given no view is set; `member_fit_extras_clear`
    empties them all. Fail-proof: drop `width = count` (the held-out X stays 5 wide), or set `call.pairs` with
    `extras.rows` rows.
  3 the no-importance refusal of a Gbdt kind (`rfe_member_source`, MEMBER_FAMILY_GBDT, a mask with neither
    MEMBER_CAN_COEF nor MEMBER_CAN_IMPORTANCES): 'auto' gives RFE_NO_IMPORTANCES, 'coef_' and
    'feature_importances_' scikit-learn's attrgetter words with the class name; a kind of another family with
    neither block keeps the estimator's own doors (block 0, no words). Fail-proof: drop MEMBER_FAMILY_GBDT from
    the family test in `rfe_member_source`.
The end-to-end fits (RFE's final fit over a Gbdt member with eval_set, weights, group_id and pairs against the
estimator's own fit on the same columns) need the gbdt image, so they are Python tests
(python/mojolearn/tests/test_rfe_gbdt_fit_params.py).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from core.member_abi import MemberCall, MemberMatrix, MEMBER_F32, MEMBER_I32, MEMBER_CAN_FIT, MEMBER_CAN_EXPORT
from core.device_route import DeviceRoute, DEVICE_CPU
from core.member_families import MEMBER_FAMILY_GBDT, MEMBER_FAMILY_LINEAR
from core.member_set_run import MemberFitExtras, member_eval_columns, member_fit_extras_apply, member_fit_extras_clear
from bindings.prep_select_run import rfe_member_source, RFE_NO_IMPORTANCES

comptime CASES = 4


def _block(rows: Int, cols: Int) -> List[Float32]:
    var x = List[Float32](length=rows * cols, fill=0.0)
    for i in range(rows):  # small-loop(fixture)
        for j in range(cols):  # small-loop(fixture)
            x[i * cols + j] = Float32(i * 100 + j) + Float32(0.25)
    return x^


def _gathered(x: List[Float32], rows: Int, cols: Int, ids: List[Int]) -> List[Float32]:
    """The oracle: element by element, row-major in the ids' order."""
    var count = len(ids)
    var got = List[Float32](length=rows * count, fill=0.0)
    for i in range(rows):  # small-loop(oracle)
        for j in range(count):  # small-loop(oracle)
            got[i * count + j] = x[i * cols + ids[j]]
    return got^


def _same(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):  # small-loop(compare)
        if a[i] != b[i]:
            return False
    return True


def check_eval_columns() raises:
    var rows = 7
    var cols = 5
    var x = _block(rows, cols)
    var picked: List[Int] = [3, 0, 4]
    var ids32: List[Int32] = [3, 0, 4]
    var ids64: List[Int64] = [3, 0, 4]
    var dst = List[Float32]()
    var copied = member_eval_columns(Int(x.unsafe_ptr()), rows, cols, Int(ids32.unsafe_ptr()), 3, False, dst)
    require(copied, "ids [3, 0, 4] are a selection: gathered")
    require(_same(dst, _gathered(x, rows, cols, picked)), "int32 ids: the round's columns in the ids' order")
    var dst64 = List[Float32]()
    _ = member_eval_columns(Int(x.unsafe_ptr()), rows, cols, Int(ids64.unsafe_ptr()), 3, True, dst64)
    require(_same(dst64, _gathered(x, rows, cols, picked)), "int64 ids: the same columns")
    var every: List[Int32] = [0, 1, 2, 3, 4]
    var untouched = List[Float32]()
    require(not member_eval_columns(Int(x.unsafe_ptr()), rows, cols, Int(every.unsafe_ptr()), 5, False, untouched),
            "every column in order: no copy")
    require(len(untouched) == 0, "every column in order: dst untouched")
    var outside: List[Int32] = [1, 5]
    var words = String("")
    try:
        _ = member_eval_columns(Int(x.unsafe_ptr()), rows, cols, Int(outside.unsafe_ptr()), 2, False, untouched)
    except error:
        words = String(error)
    require("outside the 5 columns" in words, "an id outside the columns is refused: " + words)
    _ = x^
    _ = ids32^
    _ = ids64^
    _ = every^
    _ = outside^


def check_eval_columns_split() raises:
    var rows = 300000
    var cols = 5
    var x = _block(rows, cols)
    var picked: List[Int] = [4, 1, 2, 0]
    var ids: List[Int32] = [4, 1, 2, 0]
    var dst = List[Float32]()
    require(member_eval_columns(Int(x.unsafe_ptr()), rows, cols, Int(ids.unsafe_ptr()), 4, False, dst),
            "split gather copies")
    require(_same(dst, _gathered(x, rows, cols, picked)), "the split gather equals the element-by-element gather")
    _ = x^
    _ = ids^


def check_extras_views() raises:
    var rows = 6
    var eval_rows = 4
    var cols = 5
    var weights = List[Float32](length=rows, fill=0.5)
    var groups: List[Int32] = [0, 0, 1, 1, 1, 2]
    var pairs: List[Int32] = [0, 1, 3, 2]
    var pair_weights: List[Float32] = [1.0, 2.0]
    var eval_x = _block(eval_rows, cols)
    var eval_y: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var extras = MemberFitExtras()
    extras.rows = rows
    extras.weights = Int(weights.unsafe_ptr())
    extras.groups = Int(groups.unsafe_ptr())
    extras.pairs = Int(pairs.unsafe_ptr())
    extras.pair_count = 2
    extras.pairs_weight = Int(pair_weights.unsafe_ptr())
    extras.eval_x = Int(eval_x.unsafe_ptr())
    extras.eval_rows = eval_rows
    extras.eval_cols = cols
    extras.eval_y = Int(eval_y.unsafe_ptr())
    extras.eval_y_dtype = MEMBER_F32
    require(extras.given(), "extras given")
    var ids: List[Int32] = [2, 4]
    var held = List[Float32]()
    var call = MemberCall(DeviceRoute(DEVICE_CPU))
    member_fit_extras_apply(extras, call, Int(ids.unsafe_ptr()), 2, False, held)
    require(call.weights.rows == rows and call.weights.cols == 1 and call.weights.data == extras.weights
            and call.weights.dtype == MEMBER_F32, "weights: rows x 1 float32")
    require(call.groups.rows == rows and call.groups.cols == 1 and call.groups.dtype == MEMBER_I32
            and call.groups.data == extras.groups, "groups: rows x 1 int32")
    require(call.pairs.rows == 2 and call.pairs.cols == 2 and call.pairs.dtype == MEMBER_I32, "pairs: 2 x 2 int32")
    require(call.pairs_weight.rows == 2 and call.pairs_weight.cols == 1, "pair weights: 2 x 1")
    require(call.eval_x.rows == eval_rows and call.eval_x.cols == 2 and call.eval_x.data == Int(held.unsafe_ptr()),
            "held-out X: the gathered block, the round's 2 columns")
    var picked: List[Int] = [2, 4]
    require(_same(held, _gathered(eval_x, eval_rows, cols, picked)), "held-out X: the round's columns")
    require(call.eval_y.rows == eval_rows and call.eval_y.cols == 1 and call.eval_y.dtype == MEMBER_F32
            and call.eval_y.data == extras.eval_y, "held-out y: its own words")
    member_fit_extras_clear(call)
    require(not call.weights.given() and not call.groups.given() and not call.pairs.given()
            and not call.pairs_weight.given() and not call.eval_x.given() and not call.eval_y.given(),
            "cleared after the fit")
    # Every column in order: the caller's own block.
    var every: List[Int32] = [0, 1, 2, 3, 4]
    var none_held = List[Float32]()
    var whole = MemberCall(DeviceRoute(DEVICE_CPU))
    member_fit_extras_apply(extras, whole, Int(every.unsafe_ptr()), 5, False, none_held)
    require(whole.eval_x.data == extras.eval_x and whole.eval_x.cols == cols and len(none_held) == 0,
            "every column in order: the caller's block, no copy")
    # Nothing given: no view.
    var empty = MemberCall(DeviceRoute(DEVICE_CPU))
    var nothing = List[Float32]()
    member_fit_extras_apply(MemberFitExtras(), empty, 0, 3, False, nothing)
    require(not empty.weights.given() and not empty.groups.given() and not empty.pairs.given()
            and not empty.eval_x.given(), "nothing given: no view set")
    require(not MemberFitExtras().given(), "empty extras are not given")
    _ = weights^
    _ = groups^
    _ = pairs^
    _ = pair_weights^
    _ = eval_x^
    _ = eval_y^
    _ = ids^
    _ = held^
    _ = every^


def check_gbdt_refusal() raises:
    var mask = MEMBER_CAN_FIT | MEMBER_CAN_EXPORT
    var auto = rfe_member_source(MEMBER_FAMILY_GBDT, mask, "auto", "GradientBoostingRegressor")
    require(auto.block == 0 and auto.refusal == String(RFE_NO_IMPORTANCES), "auto: " + auto.refusal)
    var coef = rfe_member_source(MEMBER_FAMILY_GBDT, mask, "coef_", "GradientBoosting")
    require(coef.refusal == "AttributeError: 'GradientBoosting' object has no attribute 'coef_'", "coef_: " + coef.refusal)
    var named = rfe_member_source(MEMBER_FAMILY_GBDT, mask, "feature_importances_", "GradientBoostingClassifier")
    require(named.refusal == "AttributeError: 'GradientBoostingClassifier' object has no attribute 'feature_importances_'",
            "feature_importances_: " + named.refusal)
    var other = rfe_member_source(MEMBER_FAMILY_LINEAR, mask, "auto", "")
    require(other.block == 0 and other.refusal.byte_length() == 0, "another family keeps its doors")


def test_mf_gbdt5_rfe_extras() raises:
    def run_case(which: Int) raises:
        if which == 0:
            check_eval_columns()
        elif which == 1:
            check_eval_columns_split()
        elif which == 2:
            check_extras_views()
        else:
            check_gbdt_refusal()
    check_cases("mf_gbdt5_rfe_extras", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

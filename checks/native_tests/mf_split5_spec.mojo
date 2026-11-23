# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""THE SPLITTER SPEC (lane mf-split5, bindings/msel_split_spec_run.mojo) and its callers, in a process with no
Python. Oracles are scikit-learn's own definitions of each splitter (sklearn/model_selection/_split.py), written
out here as literal rows, or the same RUN by the public door's path (the one-body checks).

 1. LeaveOneOut(n=4): split s tests row s, trains the others; n = 1 refused in the door's words.
 2. LeavePOut(p=2), n = 4: the six combinations in itertools order, train the complement; p = 0 one split with
    every row in train; p >= n refused in the door's words.
 3. `_validate_shuffle_split` (`msel_shuffle_sizes_run`): n=10 test 0.25 -> (7, 3); test 3 -> (7, 3); train 0.5
    -> (5, 5); a float32 test 0.1 -> NumPy's float32 product 1.0 -> ceil 1 (float64 would give 2); test 10 and an
    empty train refused in the door's words.
 4. Group codes: ids [30, 10, 30, 20] -> sorted-order codes [2, 0, 2, 1], text "[10, 20, 30]".
 5. LeaveOneGroupOut over groups [5, 5, 7, 9, 7]: tests {0,1}, {2,4}, {3}; one group refused.
 6. LeavePGroupsOut(n_groups=2) over the same: tests {0,1,2,4}, {0,1,3}, {2,3,4}; n_groups 3 refused.
 7. PredefinedSplit(test_fold=[0, 1, -1, 1, 0]) (codes [1, 2, 0, 2, 1] over classes [-1, 0, 1], wants 1 and 2):
    tests {0,4}, {1,3}; row 2 always trains.
 8. A group splitter with no groups: "The 'groups' parameter should not be None." (scikit-learn's words).
 9. KFold(3) unshuffled over 7 rows: tests {0,1,2}, {3,4}, {5,6} (the first n % k folds one row longer).
10. RepeatedKFold(n_splits=3, n_repeats=2, random_state=11) over 9 rows == the public door's repeat body
    (`msel_plan_kfold` seeded `msel_mix_seed(11, r)`), word for word; each repeat a partition.
11. An index list (LISTED) and the int32 narrowing: the pairs as given.
12. StratifiedKFold(2) over codes [0, 1, 1, 1, 1]: one "least populated class" warning operand, 1.
13. ShuffleSplit(n_splits=2, test_size=0.3, random_state=5) over 10 rows == the door's body (sizes 7 / 3, then
    `msel_plan_shuffle`), word for word.
14. StackingClassifier (META) with a KFold(3) spec == the same stack over the listed contiguous folds (sklearn's
    KFold rows): predict_proba word for word.
15. StackingClassifier with ShuffleSplit refused ("cross_val_predict only works for partitions: ..."), with
    GroupKFold refused in scikit-learn's groups words.
16. CalibratedClassifierCV (META, sigmoid, ensemble=True) with PredefinedSplit(test_fold = i % 3) == the same
    calibration over the listed folds: predict_proba word for word.
17. LassoCV (x_linear member) with a KFold(3) spec == the integer cv=3 member: coef, scalars and mse_path_ word for
    word (the spec's fold ids come from `xl_cv_fold_ids_run`, the int path's from `_kfold_ids`).

FAIL-PROOF EDITS (each must make the named case fail): in `msel_plan_leave_one_out` add `plan.add(n, 0)` for the
first split (case 1); start `combo[j] = combo[j - 1] + 2` in `msel_plan_leave_p_out` (case 2); compute the float32
product in float64 in `_size_product` (case 3, the float32 row); `dst[i] = Int32(lo + 1)` in `msel_group_codes_run`
(cases 4-6); `wants.append(g + 1)` in the LeaveOneGroupOut branch (case 5); drop the `groups.given` refusal (case
8); seed repeat r with `msel_mix_seed(base.value, r + 1)` in `msel_plan_repeat` (case 10); return `sides.least`
empty in `_take` (case 12); swap `sizes[0]` and `sizes[1]` in the SHUFFLE branch (case 13); in core/ensemble_member.mojo
`_split_folds` assign `slot[].fold_train = te32^` (case 14); in core/xlinear_member.mojo `_cv_fold_words` write the
codes of `ids` shifted by one fold (case 17).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.device_route import DeviceRoute, DEVICE_CPU
from core.entry_table import EntryRef, entry_ref_unresolved, entry_ref_word
from core.msel_entry_table import MselEntryKey
from core.msel_helpers_host import publish_msel_helpers_host
from core.msel_helpers_calls import msel_helpers_ref
from core.base_entry_table import BaseEntryKey
from core.base_helpers_host import publish_base_helpers_host
from core.base_helpers_calls import base_helpers_ref
from x_metrics.descriptor_host import metrics_program_host_v1
from bindings.msel_split_plan_run import (
    MselSeed, MselSplitRoute, MselSplitPlan, MselSizeWord, msel_size_none, msel_plan_leave_one_out,
    msel_plan_leave_p_out, msel_shuffle_sizes_run, msel_plan_kfold, msel_plan_shuffle, msel_mix_seed, msel_split_fill,
    MSEL_SIZE_INT, MSEL_SIZE_REAL,
)
from bindings.msel_split_spec_run import (
    MselSplitSpec, MselSides, MselGroupCodes, msel_spec_sides, msel_sides_i32, msel_group_codes_run, MSEL_SPEC_KFOLD,
    MSEL_SPEC_STRATIFIED_KFOLD, MSEL_SPEC_GROUP_KFOLD, MSEL_SPEC_LOGO, MSEL_SPEC_LPGO, MSEL_SPEC_PREDEFINED,
    MSEL_SPEC_SHUFFLE, MSEL_SPEC_REPEATED_KFOLD, MSEL_SPEC_LISTED, msel_spec_mint, msel_spec_release,
)
from core.ensemble_member import (
    ensemble_member_v2, meta_plan_mint_with, meta_plan_release, meta_plan_split, META_STACK_CLASSIFIER,
    META_CALIBRATED,
)
from core.logistic_member import logistic_member_v2
from core.logistic_native_api import QnNativeDriver
from core.logistic_route import LogisticRoute, logistic_route_pack
from core.member_abi import (
    MemberCall, MemberMatrix, MemberPlan, MemberRef, member_call, member_call_of, member_new, member_free,
    member_kind_capabilities, MEMBER_OP_CAPABILITIES, MEMBER_OP_FIT, MEMBER_OP_PREDICT_PROBA, MEMBER_OP_EXPORT,
    MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_BLOCK_COEF, MEMBER_BLOCK_SCALARS,
)
from core.member_families import MEMBER_FAMILY_ENSEMBLE, MEMBER_FAMILY_XLINEAR
from glm.host.logistic_native_host import qn_host_native_v1
from xtrees.glue_runs import publish_x_trees_glue
from xtrees.trees_meta_vote_run import STACK_PROBA
from core.xlinear_member import xlinear_member_v2, XL_MEMBER_BLOCK_MSE_PATH
from x_linear.ops_entry_host import x_linear_ops_host_v1

comptime CASES = 17
comptime N = 60
comptime D = 3
comptime K = 3


def _route() raises -> MselSplitRoute:
    var none = entry_ref_unresolved()
    var mode = Int(GLOBAL_NUMERIC_MODE)
    return MselSplitRoute(msel_helpers_ref(MselEntryKey(mode, 0)), base_helpers_ref(BaseEntryKey(mode, 0)),
                          metrics_program_host_v1, False, False)


def _spec(kind: Int) raises -> MselSplitSpec:
    var spec = MselSplitSpec()
    spec.kind = kind
    spec.route = _route()
    spec.rows_helpers = base_helpers_ref(BaseEntryKey(Int(GLOBAL_NUMERIC_MODE), 0))
    return spec^


def _rows(side: List[Int64], want: List[Int], what: String) raises:
    require(len(side) == len(want), what + ": " + String(len(side)) + " rows, expected " + String(len(want)))
    for i in range(len(want)):  # small-loop(oracle: a few literal rows)
        require(Int(side[i]) == want[i], what + ": row " + String(i) + " is " + String(Int(side[i])))


def _complement(n: Int, test: List[Int]) -> List[Int]:
    var out = List[Int]()
    for i in range(n):  # small-loop(oracle: the train side of a split, every row not tested)
        var tested = False
        for j in range(len(test)):  # small-loop(oracle)
            tested = tested or test[j] == i
        if not tested:
            out.append(i)
    return out^


def _check_split(sides: MselSides, s: Int, n: Int, test: List[Int], what: String) raises:
    _rows(sides.test[s], test, what + " test")
    _rows(sides.train[s], _complement(n, test), what + " train")


def _fill(route: MselSplitRoute, var plan: MselSplitPlan) raises -> MselSides:
    """The door's PACK + fill of one plan, as native runs."""
    var sides = MselSides()
    for s in range(plan.splits()):  # small-loop(oracle: one pair per split)
        sides.train.append(List[Int64](length=max(plan.train_sizes[s], 1), fill=Int64(-7)))
        sides.test.append(List[Int64](length=max(plan.test_sizes[s], 1), fill=Int64(-7)))
    for s in range(plan.splits()):  # small-loop(oracle: their addresses)
        plan.train_dst.append(Int(sides.train[s].unsafe_ptr()))
        plan.test_dst.append(Int(sides.test[s].unsafe_ptr()))
    msel_split_fill(route, plan)
    for s in range(plan.splits()):  # small-loop(oracle: exact lengths)
        sides.train[s].resize(plan.train_sizes[s], Int64(0))
        sides.test[s].resize(plan.test_sizes[s], Int64(0))
    return sides^


def _same_sides(a: MselSides, b: MselSides, what: String) raises:
    require(a.splits() == b.splits(), what + ": split counts differ")
    for s in range(a.splits()):  # small-loop(oracle: every split)
        require(len(a.train[s]) == len(b.train[s]) and len(a.test[s]) == len(b.test[s]), what + ": sizes differ")
        for i in range(len(a.train[s])):  # small-loop(oracle: every row)
            require(a.train[s][i] == b.train[s][i], what + ": split " + String(s) + " train row " + String(i))
        for i in range(len(a.test[s])):  # small-loop(oracle: every row)
            require(a.test[s][i] == b.test[s][i], what + ": split " + String(s) + " test row " + String(i))


def _refused(problem: String, words: String, what: String) raises:
    require(problem == words, what + ": got '" + problem + "'")


def _groups(ids: List[Int64]) raises -> MselGroupCodes:
    return msel_group_codes_run(Int(ids.unsafe_ptr()), True, len(ids))


# ------------------------------------------------------------ the spec's RUN


def check_leave_one_out() raises:
    var route = _route()
    var sides = _fill(route, msel_plan_leave_one_out(4))
    require(sides.splits() == 4, "LeaveOneOut: 4 splits")
    for s in range(4):  # small-loop(oracle)
        _check_split(sides, s, 4, [s], "LeaveOneOut split " + String(s))
    var words = String("")
    try:
        _ = msel_plan_leave_one_out(1)
    except e:
        words = String(e)
    _refused(words, "Cannot perform LeaveOneOut with n_samples=1.", "LeaveOneOut n=1")


def check_leave_p_out() raises:
    var route = _route()
    var sides = _fill(route, msel_plan_leave_p_out(4, 2))
    require(sides.splits() == 6, "LeavePOut: C(4, 2) = 6 splits")
    var combos: List[List[Int]] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    for s in range(6):  # small-loop(oracle: itertools.combinations order)
        _check_split(sides, s, 4, combos[s], "LeavePOut split " + String(s))
    var none = _fill(route, msel_plan_leave_p_out(4, 0))
    require(none.splits() == 1, "LeavePOut p=0: one split")
    _rows(none.train[0], [0, 1, 2, 3], "LeavePOut p=0 train")
    _rows(none.test[0], List[Int](), "LeavePOut p=0 test")
    var words = String("")
    try:
        _ = msel_plan_leave_p_out(4, 4)
    except e:
        words = String(e)
    _refused(words, "p=4 must be strictly less than the number of samples=4", "LeavePOut p=n")


def _int_size(v: Int) -> MselSizeWord:
    return MselSizeWord(MSEL_SIZE_INT, v, 0.0, 64, String(v))


def _real_size(v: Float64, bits: Int, text: String) -> MselSizeWord:
    return MselSizeWord(MSEL_SIZE_REAL, 0, v, bits, text)


def check_shuffle_sizes() raises:
    var a = msel_shuffle_sizes_run(10, _real_size(0.25, 64, "0.25"), msel_size_none(), False, String(""))
    require(a[0] == 7 and a[1] == 3, "test_size=0.25: ceil(2.5) = 3 test, 7 train")
    var b = msel_shuffle_sizes_run(10, _int_size(3), msel_size_none(), False, String(""))
    require(b[0] == 7 and b[1] == 3, "test_size=3: 7 train")
    var c = msel_shuffle_sizes_run(10, msel_size_none(), _real_size(0.5, 64, "0.5"), False, String(""))
    require(c[0] == 5 and c[1] == 5, "train_size=0.5: floor(5) = 5 train, 5 test")
    var f32 = Float64(Float32(0.1))
    var d = msel_shuffle_sizes_run(10, _real_size(f32, 32, "0.1"), msel_size_none(), False, String(""))
    require(d[1] == 1, "a float32 test_size 0.1: NumPy's float32 product rounds to 1.0, ceil 1 (got "
            + String(d[1]) + ")")
    var e64 = msel_shuffle_sizes_run(10, _real_size(f32, 64, "0.10000000149011612"), msel_size_none(), False,
                                     String(""))
    require(e64[1] == 2, "the same value as a float64 size: ceil 2")
    var words = String("")
    try:
        _ = msel_shuffle_sizes_run(10, _int_size(10), msel_size_none(), False, String(""))
    except e:
        words = String(e)
    _refused(words, "test_size=10 should be either positive and smaller than the number of samples 10 or a float "
             + "in the (0, 1) range", "test_size=n")
    words = String("")
    try:
        _ = msel_shuffle_sizes_run(10, _real_size(0.95, 64, "0.95"), msel_size_none(), False, String(""))
    except e:
        words = String(e)
    _refused(words, "With n_samples=10, test_size=0.95 and train_size=None, the resulting train set will be empty. "
             + "Adjust any of the aforementioned parameters.", "an empty train side")


def check_group_codes() raises:
    var ids: List[Int64] = [30, 10, 30, 20]
    var g = _groups(ids)
    require(g.m() == 3 and g.text() == "[10, 20, 30]", "group classes: " + g.text())
    var want: List[Int32] = [2, 0, 2, 1]
    for i in range(4):  # small-loop(oracle)
        require(g.codes[i] == want[i], "group code " + String(i))
    _ = len(ids)


def check_logo() raises:
    var ids: List[Int64] = [5, 5, 7, 9, 7]
    var g = _groups(ids)
    var sides = msel_spec_sides(_spec(MSEL_SPEC_LOGO), 5, 0, 0, g)
    require(sides.splits() == 3, "LeaveOneGroupOut: one split per group")
    _check_split(sides, 0, 5, [0, 1], "LeaveOneGroupOut group 5")
    _check_split(sides, 1, 5, [2, 4], "LeaveOneGroupOut group 7")
    _check_split(sides, 2, 5, [3], "LeaveOneGroupOut group 9")
    var one: List[Int64] = [4, 4, 4]
    var words = String("")
    try:
        _ = msel_spec_sides(_spec(MSEL_SPEC_LOGO), 3, 0, 0, _groups(one))
    except e:
        words = String(e)
    _refused(words, "The groups parameter contains fewer than 2 unique groups ([4]). LeaveOneGroupOut expects at "
             + "least 2.", "LeaveOneGroupOut, one group")
    _ = len(ids)
    _ = len(one)


def check_lpgo() raises:
    var ids: List[Int64] = [5, 5, 7, 9, 7]
    var spec = _spec(MSEL_SPEC_LPGO)
    spec.k = 2
    var sides = msel_spec_sides(spec, 5, 0, 0, _groups(ids))
    require(sides.splits() == 3, "LeavePGroupsOut(2): C(3, 2) splits")
    _check_split(sides, 0, 5, [0, 1, 2, 4], "LeavePGroupsOut groups 5, 7")
    _check_split(sides, 1, 5, [0, 1, 3], "LeavePGroupsOut groups 5, 9")
    _check_split(sides, 2, 5, [2, 3, 4], "LeavePGroupsOut groups 7, 9")
    spec.k = 3
    var words = String("")
    try:
        _ = msel_spec_sides(spec, 5, 0, 0, _groups(ids))
    except e:
        words = String(e)
    _refused(words, "The groups parameter contains fewer than (or equal to) n_groups (3) numbers of unique groups "
             + "([5, 7, 9]).", "LeavePGroupsOut n_groups = groups")
    _ = len(ids)


def check_predefined() raises:
    var spec = _spec(MSEL_SPEC_PREDEFINED)
    spec.folds = [1, 2, 0, 2, 1]
    spec.fold_classes = 3
    spec.wants = [1, 2]
    var sides = msel_spec_sides(spec, 5, 0, 0, MselGroupCodes())
    require(sides.splits() == 2, "PredefinedSplit: two folds")
    _check_split(sides, 0, 5, [0, 4], "PredefinedSplit fold 0")
    _check_split(sides, 1, 5, [1, 3], "PredefinedSplit fold 1")


def check_groups_none() raises:
    var spec = _spec(MSEL_SPEC_GROUP_KFOLD)
    spec.k = 2
    var words = String("")
    try:
        _ = msel_spec_sides(spec, 6, 0, 0, MselGroupCodes())
    except e:
        words = String(e)
    _refused(words, "The 'groups' parameter should not be None.", "GroupKFold without groups")


def check_kfold() raises:
    var spec = _spec(MSEL_SPEC_KFOLD)
    spec.k = 3
    var sides = msel_spec_sides(spec, 7, 0, 0, MselGroupCodes())
    require(sides.splits() == 3, "KFold(3): 3 splits")
    _check_split(sides, 0, 7, [0, 1, 2], "KFold fold 0")
    _check_split(sides, 1, 7, [3, 4], "KFold fold 1")
    _check_split(sides, 2, 7, [5, 6], "KFold fold 2")


def check_repeated() raises:
    var route = _route()
    var spec = _spec(MSEL_SPEC_REPEATED_KFOLD)
    spec.k = 3
    spec.repeats = 2
    spec.seed = MselSeed(11, False)
    var got = msel_spec_sides(spec, 9, 0, 0, MselGroupCodes())
    var want = MselSides()
    for r in range(2):  # small-loop(oracle: the door's repeats, one plan each)
        var one = _fill(route, msel_plan_kfold(route, 9, 3, True, MselSeed(UInt64(msel_mix_seed(11, r)), False),
                                               False))
        for s in range(one.splits()):  # small-loop(oracle)
            want.train.append(one.train[s].copy())
            want.test.append(one.test[s].copy())
    _same_sides(got, want, "RepeatedKFold vs the door's repeat body")
    for r in range(2):  # small-loop(oracle: each repeat's tests partition the rows)
        var seen = List[Int](length=9, fill=0)
        for s in range(3):  # small-loop(oracle)
            for i in range(len(got.test[3 * r + s])):  # small-loop(oracle)
                seen[Int(got.test[3 * r + s][i])] += 1
        for i in range(9):  # small-loop(oracle)
            require(seen[i] == 1, "RepeatedKFold repeat " + String(r) + ": row " + String(i) + " tested once")


def check_listed() raises:
    var spec = _spec(MSEL_SPEC_LISTED)
    var tr: List[Int64] = [0, 1, 2]
    var te: List[Int64] = [3, 4]
    spec.listed_train.append(tr^)
    spec.listed_test.append(te^)
    var sides = msel_spec_sides(spec, 5, 0, 0, MselGroupCodes())
    _rows(sides.train[0], [0, 1, 2], "listed train")
    _rows(sides.test[0], [3, 4], "listed test")
    var t32 = List[List[Int32]]()
    var e32 = List[List[Int32]]()
    msel_sides_i32(sides, t32, e32)
    require(len(t32) == 1 and len(t32[0]) == 3 and t32[0][2] == 2 and len(e32[0]) == 2 and e32[0][1] == 4,
            "the int32 narrowing")


def check_stratified_warning() raises:
    var spec = _spec(MSEL_SPEC_STRATIFIED_KFOLD)
    spec.k = 2
    var codes: List[Int32] = [0, 1, 1, 1, 1]
    var sides = msel_spec_sides(spec, 5, Int(codes.unsafe_ptr()), 2, MselGroupCodes())
    _ = len(codes)
    require(len(sides.least) == 1 and sides.least[0] == 1, "one least-populated warning, operand 1")
    var seen = List[Int](length=5, fill=0)
    for s in range(sides.splits()):  # small-loop(oracle: the tests partition the rows)
        for i in range(len(sides.test[s])):  # small-loop(oracle)
            seen[Int(sides.test[s][i])] += 1
    for i in range(5):  # small-loop(oracle)
        require(seen[i] == 1, "StratifiedKFold: row " + String(i) + " tested once")


def check_shuffle_split() raises:
    var route = _route()
    var spec = _spec(MSEL_SPEC_SHUFFLE)
    spec.k = 2
    spec.test = _real_size(0.3, 64, "0.3")
    spec.seed = MselSeed(5, False)
    var got = msel_spec_sides(spec, 10, 0, 0, MselGroupCodes())
    var want = _fill(route, msel_plan_shuffle(route, 10, 2, 7, 3, MselSeed(5, False)))
    _same_sides(got, want, "ShuffleSplit vs the door's body")
    require(len(got.test[0]) == 3 and len(got.train[0]) == 7, "ShuffleSplit sizes 7 / 3")


# ------------------------------------------------------------ the callers


def _x() -> List[Float32]:
    var x = List[Float32](length=N * D, fill=0.0)
    for i in range(N):  # small-loop(fixture)
        for j in range(D):  # small-loop(fixture)
            x[i * D + j] = Float32((i * 7 + j * 5) % 11) * 0.25 - 1.0 + Float32(i % 3) * 0.5 * Float32(j + 1)
    return x^


def _codes() -> List[Int32]:
    var y = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # small-loop(fixture: every class has N / 3 rows)
        y[i] = Int32((i + i // 5) % K)
    return y^


def _logistic(l2: Float64) raises -> MemberPlan:
    var plan = MemberPlan()
    plan.route = logistic_route_pack(LogisticRoute(DEVICE_CPU, Int(GLOBAL_NUMERIC_MODE), 0, False, False,
                                                   QnNativeDriver(qn_host_native_v1)))
    plan.member = MemberRef(EntryRef(logistic_member_v2, 0), 0)
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    plan.params = [0.0, 1.0, 0.0, l2, 100.0, 20.0, 10.0, 0.0, 1e-4, 1e-6]
    plan.classes = K
    return plan^


struct _Meta(Movable):
    var plan: MemberPlan
    var handle: Int

    def __init__(out self, var plan: MemberPlan, handle: Int):
        self.plan = plan^
        self.handle = handle

    def close(mut self):
        meta_plan_release(self.handle)
        self.handle = 0


def _meta_with(kind: Int, folds: Int, var nested: List[MemberPlan], var words: List[Float64],
               var fold_train: List[List[Int32]], var fold_test: List[List[Int32]], var stack_kinds: List[Int],
               spec: MselSplitSpec) raises -> _Meta:
    """A META plan as the boundary's `_finish` and `_finish_split` mint it."""
    var masks = List[Int]()
    for u in range(len(nested)):  # small-loop(fixture: each nested kind's mask)
        masks.append(member_kind_capabilities(nested[u]))
    var handle = meta_plan_mint_with(nested^, List[Float64](), stack_kinds^, masks^, words^, fold_train^, fold_test^)
    meta_plan_split(handle, spec)
    var plan = MemberPlan()
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = Int(GLOBAL_NUMERIC_MODE)
    route.vendor = 0
    plan.route = route
    plan.member = MemberRef(EntryRef(ensemble_member_v2, 0), 0)
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    plan.family = MEMBER_FAMILY_ENSEMBLE
    plan.params = [Float64(kind), Float64(handle), 0.0, Float64(folds), 0.0, 0.0]
    plan.classes = K
    return _Meta(plan^, handle)


def _fit(plan: MemberPlan, x: List[Float32], y: List[Int32]) raises -> Int:
    var call = member_new(plan)
    call.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    call.y = MemberMatrix(Int(y.unsafe_ptr()), N, 1, MEMBER_I32)
    call.classes = K
    try:
        _ = member_call(plan.member, MEMBER_OP_FIT, call)
    except e:
        member_free(plan, call)
        raise e^
    return call.state


def _proba(meta: _Meta, state: Int, x: List[Float32]) raises -> List[Float64]:
    var got = List[Float64](length=N * K, fill=-7.0)
    var call = member_call_of(meta.plan, state)
    call.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    call.dst = MemberMatrix(Int(got.unsafe_ptr()), N, K, MEMBER_F64)
    _ = member_call(meta.plan.member, MEMBER_OP_PREDICT_PROBA, call)
    return got^


def _free(plan: MemberPlan, state: Int):
    var call = MemberCall(plan.route, state)
    member_free(plan, call)


def _same_words(a: List[Float64], b: List[Float64], what: String) raises:
    require(len(a) == len(b), what + ": lengths differ")
    for i in range(len(a)):  # small-loop(oracle: every word)
        require(a[i] == b[i], what + " word " + String(i) + ": " + String(a[i]) + " != " + String(b[i]))


def _contiguous(mut tr: List[List[Int32]], mut te: List[List[Int32]]):
    """scikit-learn's KFold(3) rows over N = 60: fold f tests rows 20 f .. 20 f + 19."""
    for f in range(3):  # small-loop(oracle)
        var a = List[Int32]()
        var b = List[Int32]()
        for i in range(N):  # small-loop(oracle)
            if i // 20 == f:
                b.append(Int32(i))
            else:
                a.append(Int32(i))
        tr.append(a^)
        te.append(b^)


def _mod3(mut tr: List[List[Int32]], mut te: List[List[Int32]]):
    """PredefinedSplit(test_fold = i % 3)'s rows: fold f tests rows i % 3 == f."""
    for f in range(3):  # small-loop(oracle)
        var a = List[Int32]()
        var b = List[Int32]()
        for i in range(N):  # small-loop(oracle)
            if i % 3 == f:
                b.append(Int32(i))
            else:
                a.append(Int32(i))
        tr.append(a^)
        te.append(b^)


def _stack_nested() raises -> List[MemberPlan]:
    var nested: List[MemberPlan] = [_logistic(1.0), _logistic(0.25), _logistic(1.0)]
    return nested^


def check_stack_spec() raises:
    var x = _x()
    var y = _codes()
    var spec = _spec(MSEL_SPEC_KFOLD)
    spec.k = 3
    var kinds: List[Int] = [STACK_PROBA, STACK_PROBA]
    var meta = _meta_with(META_STACK_CLASSIFIER, 0, _stack_nested(), List[Float64](), List[List[Int32]](),
                          List[List[Int32]](), kinds^, spec)
    var state = _fit(meta.plan, x, y)
    var got = _proba(meta, state, x)
    _free(meta.plan, state)
    meta.close()
    var tr = List[List[Int32]]()
    var te = List[List[Int32]]()
    _contiguous(tr, te)
    var kinds2: List[Int] = [STACK_PROBA, STACK_PROBA]
    var listed = _meta_with(META_STACK_CLASSIFIER, 3, _stack_nested(), List[Float64](), tr^, te^, kinds2^,
                            MselSplitSpec())
    var s2 = _fit(listed.plan, x, y)
    var want = _proba(listed, s2, x)
    _free(listed.plan, s2)
    listed.close()
    _same_words(got, want, "Stacking over a KFold(3) spec vs its listed folds")
    _ = x^
    _ = y^


def _stack_refusal(spec: MselSplitSpec) raises -> String:
    var x = _x()
    var y = _codes()
    var kinds: List[Int] = [STACK_PROBA, STACK_PROBA]
    var meta = _meta_with(META_STACK_CLASSIFIER, 0, _stack_nested(), List[Float64](), List[List[Int32]](),
                          List[List[Int32]](), kinds^, spec)
    var words = String("")
    try:
        var state = _fit(meta.plan, x, y)
        _free(meta.plan, state)
    except e:
        words = String(e)
    meta.close()
    _ = x^
    _ = y^
    return words^


def check_stack_refusals() raises:
    var shuffle = _spec(MSEL_SPEC_SHUFFLE)
    shuffle.k = 2
    shuffle.test = _real_size(0.25, 64, "0.25")
    shuffle.seed = MselSeed(3, False)
    _refused(_stack_refusal(shuffle),
             "cross_val_predict only works for partitions: every row must be in exactly one test set",
             "Stacking over ShuffleSplit")
    var grouped = _spec(MSEL_SPEC_GROUP_KFOLD)
    grouped.k = 3
    _refused(_stack_refusal(grouped), "The 'groups' parameter should not be None.", "Stacking over GroupKFold")


def check_calibrated_predefined() raises:
    var x = _x()
    var y = _codes()
    var spec = _spec(MSEL_SPEC_PREDEFINED)
    spec.folds = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # small-loop(fixture: test_fold = i % 3, classes [0, 1, 2])
        spec.folds[i] = Int32(i % 3)
    spec.fold_classes = 3
    spec.wants = [0, 1, 2]
    var nested: List[MemberPlan] = [_logistic(1.0)]
    var words: List[Float64] = [1.0, 1.0, 1.0]  # sigmoid, ensemble, decision
    var meta = _meta_with(META_CALIBRATED, 0, nested^, words^, List[List[Int32]](), List[List[Int32]](), List[Int](),
                          spec)
    var state = _fit(meta.plan, x, y)
    var got = _proba(meta, state, x)
    _free(meta.plan, state)
    meta.close()
    var tr = List[List[Int32]]()
    var te = List[List[Int32]]()
    _mod3(tr, te)
    var nested2: List[MemberPlan] = [_logistic(1.0)]
    var words2: List[Float64] = [1.0, 1.0, 1.0]
    var listed = _meta_with(META_CALIBRATED, 3, nested2^, words2^, tr^, te^, List[Int](), MselSplitSpec())
    var s2 = _fit(listed.plan, x, y)
    var want = _proba(listed, s2, x)
    _free(listed.plan, s2)
    listed.close()
    _same_words(got, want, "CalibratedClassifierCV over PredefinedSplit vs its listed folds")
    _ = x^
    _ = y^


def _lasso_plan(folds_word: Float64) raises -> MemberPlan:
    """LassoCV(cv=..., n_alphas=5) as the x_linear provider admits it: [11, fit_intercept, max_iter, eps, tol,
    positive, folds, G, explicit, L, l1_ratio 1.0]."""
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var plan = MemberPlan()
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = mode
    route.vendor = 0
    route.entry = EntryRef(x_linear_ops_host_v1, entry_ref_word(mode, 0, 0))
    plan.family = MEMBER_FAMILY_XLINEAR
    plan.member = MemberRef(EntryRef(xlinear_member_v2, 0), 0)
    plan.kind = 11
    plan.params = [11.0, 1.0, 1000.0, 1e-3, 1e-4, 0.0, folds_word, 5.0, 0.0, 1.0, 1.0]
    plan.route = route
    var probe = MemberCall(route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    return plan^


def _lasso_words(plan: MemberPlan, x: List[Float32], t: List[Float32]) raises -> List[Float64]:
    """One fit; its COEF, SCALARS and MSE_PATH words in order."""
    var call = member_new(plan)
    call.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    call.y = MemberMatrix(Int(t.unsafe_ptr()), N, 1, MEMBER_F32)
    var out = List[Float64]()
    try:
        _ = member_call(plan.member, MEMBER_OP_FIT, call)
        var blocks: List[Int] = [MEMBER_BLOCK_COEF, MEMBER_BLOCK_SCALARS, XL_MEMBER_BLOCK_MSE_PATH]
        for b in range(3):  # small-loop(oracle: three blocks)
            var e = member_call_of(plan, call.state)
            e.block = blocks[b]
            e.flags = 0
            _ = member_call(plan.member, MEMBER_OP_EXPORT, e)
            var count = e.dst.rows * e.dst.cols
            out.append(Float64(e.dst.rows))
            out.append(Float64(e.dst.cols))
            for i in range(count):  # small-loop(oracle: one block's words)
                if e.dst.dtype == MEMBER_F64:
                    out.append(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=e.dst.data)[i])
                else:
                    out.append(Float64(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=e.dst.data)[i]))
    finally:
        member_free(plan, call)
    return out^


def check_lasso_spec() raises:
    var x = _x()
    var t = List[Float32](length=N, fill=0.0)
    for i in range(N):  # small-loop(fixture: a linear target of two columns)
        t[i] = x[i * D] * 1.5 - x[i * D + 2] * 0.5 + 0.25
    var spec = _spec(MSEL_SPEC_KFOLD)
    spec.k = 3
    var handle = msel_spec_mint(spec)
    var got = List[Float64]()
    try:
        got = _lasso_words(_lasso_plan(-Float64(handle)), x, t)
    finally:
        msel_spec_release(handle)
    var want = _lasso_words(_lasso_plan(3.0), x, t)
    _same_words(got, want, "LassoCV over a KFold(3) spec vs cv=3")
    _ = x^
    _ = t^


def test_mf_split5_spec() raises:
    # The publications the host images make at module init (a repeat publishes the same pointer).
    publish_msel_helpers_host()
    publish_base_helpers_host()
    publish_x_trees_glue("cpu")

    def run_case(which: Int) raises:
        if which == 0:
            check_leave_one_out()
        elif which == 1:
            check_leave_p_out()
        elif which == 2:
            check_shuffle_sizes()
        elif which == 3:
            check_group_codes()
        elif which == 4:
            check_logo()
        elif which == 5:
            check_lpgo()
        elif which == 6:
            check_predefined()
        elif which == 7:
            check_groups_none()
        elif which == 8:
            check_kfold()
        elif which == 9:
            check_repeated()
        elif which == 10:
            check_listed()
        elif which == 11:
            check_stratified_warning()
        elif which == 12:
            check_shuffle_split()
        elif which == 13:
            check_stack_spec()
        elif which == 14:
            check_stack_refusals()
        elif which == 15:
            check_calibrated_predefined()
        else:
            check_lasso_spec()

    check_cases("mf_split5_spec", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

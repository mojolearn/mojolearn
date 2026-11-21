# SPDX-License-Identifier: Apache-2.0
"""The typed RUN of the splitters (bindings/msel_split_plan_run.mojo), run
natively (device wave, lane dw-msel-parallel, 2026-11-21).

Each case builds a plan from typed values, allocates one native int64 run
per split and side (what the door's PACK allocates as Arrays), fills it and
compares the rows with literal expected values. The cases are the forms
that need only the model-selection helper table (published here by the host
binding's own publication): default folds, the time-series ranges, an
indexed split, the selections by code, a group set, GroupKFold without
shuffle, the first-seen renumbering and the seed refusal. The forms that
run an x_metrics program are covered by `resample_cv_row_order`, which
builds the same programs.

The cases are independent units and run across the native host worker
pool. No Python participates.
"""
from core.host_parallel import host_parallelize
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.entry_table import entry_ref_unresolved
from core.msel_entry_table import MselEntryKey
from core.msel_helpers_host import publish_msel_helpers_host
from core.msel_helpers_calls import msel_helpers_ref
from bindings.msel_split_plan_run import (
    MSEL_LABELS_ALIGNED, MselCodes, MselSeed, MselSplitPlan, MselSplitRoute, msel_first_seen_enc,
    msel_plan_default_folds, msel_plan_group_kfold, msel_plan_group_set, msel_plan_indexed, msel_plan_select,
    msel_plan_time_series, msel_seed_refusal, msel_split_fill,
)

comptime MSEL_SPLIT_PLAN_CASES = 8


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _route() raises -> MselSplitRoute:
    var none = entry_ref_unresolved()
    return MselSplitRoute(msel_helpers_ref(MselEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)), none, none.call, False, False)


struct Filled(Movable):
    """The rows of a filled plan: split s's train rows then its test rows."""

    var trains: List[List[Int64]]
    var tests: List[List[Int64]]

    def __init__(out self):
        self.trains = List[List[Int64]]()
        self.tests = List[List[Int64]]()


def _fill(route: MselSplitRoute, mut plan: MselSplitPlan) raises -> Filled:
    var rows = Filled()
    for s in range(plan.splits()):  # driver-loop: over splits (destination allocation only)
        rows.trains.append(List[Int64](length=max(plan.train_sizes[s], 1), fill=-7))
        rows.tests.append(List[Int64](length=max(plan.test_sizes[s], 1), fill=-7))
    for s in range(plan.splits()):  # driver-loop: over splits (addresses of the allocated runs)
        plan.train_dst.append(Int(rows.trains[s].unsafe_ptr()))
        plan.test_dst.append(Int(rows.tests[s].unsafe_ptr()))
    msel_split_fill(route, plan)
    return rows^


def _same(got: List[Int64], count: Int, want: List[Int], what: String) raises:
    require(count == len(want), what + ": " + String(count) + " rows, expected " + String(len(want)))
    for i in range(count):  # driver-loop: over a literal expectation of a few rows
        require(Int(got[i]) == want[i], what + ": row " + String(i) + " is " + String(Int(got[i])))


def case_default_folds() raises:
    # 10 rows in 3 folds without labels: KFold's contiguous folds of 4, 3, 3.
    var route = _route()
    var plan = msel_plan_default_folds(route, 10, 3, MselCodes(0, 0, 0), False, False)
    require(plan.splits() == 3, "default folds: split count")
    var rows = _fill(route, plan)
    _same(rows.tests[0], plan.test_sizes[0], [0, 1, 2, 3], "default folds test 0")
    _same(rows.tests[1], plan.test_sizes[1], [4, 5, 6], "default folds test 1")
    _same(rows.tests[2], plan.test_sizes[2], [7, 8, 9], "default folds test 2")
    _same(rows.trains[1], plan.train_sizes[1], [0, 1, 2, 3, 7, 8, 9], "default folds train 1")


def case_time_series() raises:
    # n=10, 3 splits, default test size 2, no gap: tests [4,5] [6,7] [8,9].
    var route = _route()
    var plan = msel_plan_time_series(10, 3, 0, False, 0, 0)
    require(plan.splits() == 3, "time series: split count")
    var rows = _fill(route, plan)
    _same(rows.trains[0], plan.train_sizes[0], [0, 1, 2, 3], "time series train 0")
    _same(rows.tests[0], plan.test_sizes[0], [4, 5], "time series test 0")
    _same(rows.tests[2], plan.test_sizes[2], [8, 9], "time series test 2")
    # a gap of 1 and at most 3 train rows: split 2 trains on rows 4..6
    var limited = msel_plan_time_series(10, 3, 0, False, 1, 3)
    var limited_rows = _fill(route, limited)
    _same(limited_rows.trains[2], limited.train_sizes[2], [4, 5, 6], "time series limited train 2")
    var refused = False
    try:
        _ = msel_plan_time_series(3, 5, 0, False, 0, 0)
    except problem:
        refused = "Cannot have number of folds=6" in String(problem)
    require(refused, "time series: more folds than samples is refused")


def case_indexed() raises:
    var route = _route()
    var picked: List[Int64] = [4, 1]
    var plan = msel_plan_indexed(6, Int(picked.unsafe_ptr()), 2)
    var rows = _fill(route, plan)
    _ = picked  # the plan reads the rows through their address
    _same(rows.tests[0], plan.test_sizes[0], [1, 4], "indexed test")
    _same(rows.trains[0], plan.train_sizes[0], [0, 2, 3, 5], "indexed train")


def case_select_counted() raises:
    var route = _route()
    var codes: List[Int32] = [0, 1, 0, 2, 1, 0]
    var view = MselCodes(Int(codes.unsafe_ptr()), 6, 3)
    var wants: List[Int] = [0, 2]
    var none = List[Int]()
    var plan = msel_plan_select(route, view, wants, True, none)
    var rows = _fill(route, plan)
    _ = codes  # the plan reads the codes through their address
    _same(rows.tests[0], plan.test_sizes[0], [0, 2, 5], "select code 0 test")
    _same(rows.trains[0], plan.train_sizes[0], [1, 3, 4], "select code 0 train")
    _same(rows.tests[1], plan.test_sizes[1], [3], "select code 2 test")


def case_select_sized() raises:
    var route = _route()
    var codes: List[Int32] = [0, 1, 0, 2, 1, 0]
    var view = MselCodes(Int(codes.unsafe_ptr()), 6, 3)
    var wants: List[Int] = [1]
    var sizes: List[Int] = [2]
    var plan = msel_plan_select(route, view, wants, False, sizes)
    var rows = _fill(route, plan)
    _same(rows.tests[0], plan.test_sizes[0], [1, 4], "select sized test")
    # a size that disagrees with the codes is refused by the fill
    var wrong: List[Int] = [3]
    var bad = msel_plan_select(route, view, wants, False, wrong)
    var refused = False
    try:
        _ = _fill(route, bad)
    except problem:
        refused = "disagrees with the group counts" in String(problem)
    _ = codes  # the plans read the codes through their address
    require(refused, "select: a wrong size is refused")


def case_group_set() raises:
    var route = _route()
    var codes: List[Int32] = [0, 1, 0, 2, 1, 0]
    var view = MselCodes(Int(codes.unsafe_ptr()), 6, 3)
    var chosen: List[Int] = [1, 2]
    var plan = msel_plan_group_set(route, view, chosen, 3)
    var rows = _fill(route, plan)
    _ = codes  # the plan reads the codes through their address
    _same(rows.tests[0], plan.test_sizes[0], [1, 3, 4], "group set test")
    _same(rows.trains[0], plan.train_sizes[0], [0, 2, 5], "group set train")


def case_group_kfold() raises:
    # Three groups of 3, 2 and 1 rows in 3 folds: every fold is one whole
    # group, every row is a test row exactly once.
    var route = _route()
    var codes: List[Int32] = [0, 1, 0, 2, 1, 0]
    var view = MselCodes(Int(codes.unsafe_ptr()), 6, 3)
    var plan = msel_plan_group_kfold(route, view, 3, False, MselSeed(0, False))
    require(plan.splits() == 3, "group k-fold: split count")
    var rows = _fill(route, plan)
    var seen = List[Int](length=6, fill=0)
    for f in range(3):  # driver-loop: over folds
        require(plan.test_sizes[f] + plan.train_sizes[f] == 6, "group k-fold: fold sizes")
        var group = Int(codes[Int(rows.tests[f][0])])
        for i in range(plan.test_sizes[f]):  # driver-loop: over one fold's few test rows
            var r = Int(rows.tests[f][i])
            require(Int(codes[r]) == group, "group k-fold: a fold holds one group")
            seen[r] += 1
    for r in range(6):  # driver-loop: over the six rows
        require(seen[r] == 1, "group k-fold: every row is tested once")
    _ = codes  # the plan reads the codes through their address
    var refused = False
    try:
        _ = msel_plan_group_kfold(route, view, 4, False, MselSeed(0, False))
    except problem:
        refused = "greater than the number of groups: 3" in String(problem)
    require(refused, "group k-fold: more folds than groups is refused")


def case_first_seen_and_seed() raises:
    var route = _route()
    # sorted-order codes [2, 0, 2, 1]: class 2 is seen first, then 0, then 1
    var codes: List[Int32] = [2, 0, 2, 1]
    var enc = msel_first_seen_enc(route, MselCodes(Int(codes.unsafe_ptr()), 4, 3), MSEL_LABELS_ALIGNED)
    _ = codes  # read through its address above
    require(enc.n_classes == 3 and len(enc.counts) == 3, "first seen: class count")
    require(enc.counts[0] == 2 and enc.counts[1] == 1 and enc.counts[2] == 1, "first seen: class sizes")
    require(Int(enc.words[0]) == 0 and Int(enc.words[1]) == 1 and Int(enc.words[2]) == 0 and Int(enc.words[3]) == 2,
            "first seen: codes")
    var refused = False
    try:
        _ = MselSeed(0, True).rng()
    except problem:
        refused = String(problem) == msel_seed_refusal()
    require(refused, "a bad seed is refused where the splitter draws")
    var rng = MselSeed(7, False).rng()
    require(rng.seed == 7 and rng.draws == 0, "a seed starts at draw 0")


def run_case(unit: Int) raises:
    if unit == 0:
        case_default_folds()
    elif unit == 1:
        case_time_series()
    elif unit == 2:
        case_indexed()
    elif unit == 3:
        case_select_counted()
    elif unit == 4:
        case_select_sized()
    elif unit == 5:
        case_group_set()
    elif unit == 6:
        case_group_kfold()
    else:
        case_first_seen_and_seed()


def run_checks() raises:
    # The publication _mojolearn_core_host makes at module init.
    publish_msel_helpers_host()
    var failures = List[Int](length=MSEL_SPLIT_PLAN_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            run_case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL model-selection split plan case", unit, String(problem))
    host_parallelize(task, MSEL_SPLIT_PLAN_CASES)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "model-selection split plan cases failed: " + String(failed))
    print("PASS model-selection split plan", MSEL_SPLIT_PLAN_CASES)

# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The stratified splitters over stored label codes (lane vdf-scope-split,
2026-11-24): device == host == CPU column.

The stratified splitters put their n int32 label codes once per split call
into the model-selection helper table's store and run the layout V3 helpers
on the stored block (core/msel_entry_table.mojo MH_FIRST_SEEN_RES,
MH_STRAT_FOLD_ASSIGN_RES, MH_SELECT_FOLDS_RES, MH_FOLD_IDS_RES;
bindings/msel_split_plan_run.mojo). Each case builds one splitter's plan from
a small fixture, fills it and returns every split's (train, test) rows:

  0  StratifiedKFold(2), unshuffled (the fold program: the codes renumbered
     inside it by `gather_i32`)                               literal rows
  1  the stratified default folds (MH_FOLD_IDS_RES, then the rows out of the
     stored fold block)                                       literal rows
  2  MH_STRAT_FOLD_ASSIGN_RES called directly (the path StratifiedKFold
     takes past its program bounds), rows by MH_SELECT_FOLDS_RES  literal rows
  3  MH_FIRST_SEEN_RES: class sizes, rank table, classes present  literal words
  4  StratifiedKFold(2), shuffled                             partition
  5  RepeatedStratifiedKFold(2), repeats 0 and 1              partition
  6  StratifiedShuffleSplit(2 splits, 6 train, 4 test)        disjoint, sized
  7  StratifiedGroupKFold(2), unshuffled and shuffled         partition, groups whole

The literal rows are scikit-learn's `StratifiedKFold(2).split` of the fixture
(classes in first-seen order: code 1 (rows 0 2 3 7 9), code 0 (rows 1 4 6),
code 2 (rows 5 8); the per-fold allocation [3 1 1] / [2 2 1]): fold 0 tests
rows 0 1 2 3 5, fold 1 rows 4 6 7 8 9. The default folds' closed form deals
the same folds.

The host column runs every case through the host helper table (the CPU
column) and checks the literal words and the properties. The device column
runs every case through the device table AND the host table published in the
same binary, requires the two answers word for word equal, and checks the
same literals and properties, so device == host == CPU column. The program
runner is the host x_metrics entry in both routes (its words are the same on
every column; this check is about the helper table). Cases run in order: the
device ones share the one store context.
"""
from std.testing import TestSuite
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.entry_table import EntryRef
from core.msel_entry_table import MselEntryKey, msel_entry_vendor
from core.msel_helpers_host import publish_msel_helpers_host
from core.msel_helpers_device import publish_msel_helpers_device
from core.msel_helpers_calls import (
    msel_helpers_ref, msel_store_put, msel_store_free, msel_first_seen_res, msel_strat_fold_assign_res,
    msel_select_folds_res,
)
from core.base_entry_table import BaseEntryKey, base_entry_vendor
from core.base_helpers_host import publish_base_helpers_host
from core.base_helpers_calls import base_helpers_ref
from x_metrics.descriptor_host import metrics_program_host_v1
from bindings.msel_split_plan_run import (
    MSEL_LABELS_SORTED, MselCodes, MselSeed, MselSplitPlan, MselSplitRoute, msel_plan_stratified,
    msel_plan_default_folds, msel_plan_repeat, msel_plan_strat_shuffle, msel_plan_strat_group, msel_split_fill,
)
from checks.native_tests.testsuite_cases import require, CPU_COLUMN_BUILD, check_cases_serial

comptime STRAT_RES_CASES = 8
comptime _N = 10


def _codes() -> List[Int32]:
    var codes: List[Int32] = [1, 0, 1, 1, 0, 2, 0, 1, 2, 1]
    return codes^


def _groups() -> List[Int32]:
    var groups: List[Int32] = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    return groups^


def _route(helpers: EntryRef) raises -> MselSplitRoute:
    var base = base_helpers_ref(BaseEntryKey(Int(GLOBAL_NUMERIC_MODE), base_entry_vendor("cpu")))
    return MselSplitRoute(helpers, base, metrics_program_host_v1, False, False)


def _fill_words(route: MselSplitRoute, mut plan: MselSplitPlan, mut out: List[Int64]) raises:
    """Fill the plan and append, per split, the train size, the test size,
    the train rows and the test rows."""
    var trains = List[List[Int64]]()
    var tests = List[List[Int64]]()
    for s in range(plan.splits()):  # driver-loop: over splits (destination allocation only)
        trains.append(List[Int64](length=max(plan.train_sizes[s], 1), fill=-7))
        tests.append(List[Int64](length=max(plan.test_sizes[s], 1), fill=-7))
    for s in range(plan.splits()):  # driver-loop: over splits (addresses of the allocated runs)
        plan.train_dst.append(Int(trains[s].unsafe_ptr()))
        plan.test_dst.append(Int(tests[s].unsafe_ptr()))
    msel_split_fill(route, plan)
    for s in range(plan.splits()):  # driver-loop: over splits (a few rows each)
        out.append(Int64(plan.train_sizes[s]))
        out.append(Int64(plan.test_sizes[s]))
        for i in range(plan.train_sizes[s]):  # driver-loop: over a fixture split's rows
            out.append(trains[s][i])
        for i in range(plan.test_sizes[s]):  # driver-loop: over a fixture split's rows
            out.append(tests[s][i])
    _ = trains^  # read through the addresses above
    _ = tests^


def _two_folds(route: MselSplitRoute, folds_id: Int, sizes: List[Int64], mut out: List[Int64]) raises:
    """Both folds' rows out of a stored fold block (MH_SELECT_FOLDS_RES)."""
    var t0 = List[Int64](length=_N, fill=-7)
    var r0 = List[Int64](length=_N, fill=-7)
    var t1 = List[Int64](length=_N, fill=-7)
    var r1 = List[Int64](length=_N, fill=-7)
    var tests: List[Int64] = [Int64(Int(t0.unsafe_ptr())), Int64(Int(t1.unsafe_ptr()))]
    var trains: List[Int64] = [Int64(Int(r0.unsafe_ptr())), Int64(Int(r1.unsafe_ptr()))]
    var bad = msel_select_folds_res(route.helpers, folds_id, _N, 2, Int(sizes.unsafe_ptr()), Int(tests.unsafe_ptr()),
                                    Int(trains.unsafe_ptr()))
    require(bad == -1, "stored folds: a fold count disagrees with its size (fold " + String(bad) + ")")
    var s0 = Int(sizes[0])
    var s1 = Int(sizes[1])
    out.append(Int64(_N - s0))
    out.append(Int64(s0))
    for i in range(_N - s0):  # driver-loop: over a fixture split's rows
        out.append(r0[i])
    for i in range(s0):  # driver-loop: over a fixture split's rows
        out.append(t0[i])
    out.append(Int64(_N - s1))
    out.append(Int64(s1))
    for i in range(_N - s1):  # driver-loop: over a fixture split's rows
        out.append(r1[i])
    for i in range(s1):  # driver-loop: over a fixture split's rows
        out.append(t1[i])
    _ = t0^  # read through the addresses above
    _ = r0^
    _ = t1^
    _ = r1^
    _ = tests^
    _ = trains^


def _case(route: MselSplitRoute, unit: Int) raises -> List[Int64]:
    var out = List[Int64]()
    var codes = _codes()
    var groups = _groups()
    var labels = MselCodes(Int(codes.unsafe_ptr()), _N, 3)
    var group_codes = MselCodes(Int(groups.unsafe_ptr()), _N, 5)
    if unit == 0:
        var plan = msel_plan_stratified(route, labels, MSEL_LABELS_SORTED, 2, False, MselSeed(0, False))
        _fill_words(route, plan, out)
    elif unit == 1:
        var plan = msel_plan_default_folds(route, _N, 2, labels, True, False)
        require(plan.folds_id >= 0, "default folds: the fold ids did not stay in the store")
        _fill_words(route, plan, out)
        require(plan.folds_id < 0, "default folds: the fill did not free the stored fold block")
    elif unit == 2:
        var held = msel_store_put(route.helpers, Int(codes.unsafe_ptr()), 4 * _N)
        var rank: List[Int32] = [1, 0, 2]
        var alloc: List[Int64] = [3, 1, 1, 2, 2, 1]
        var counts: List[Int64] = [5, 3, 2]
        var sizes: List[Int64] = [0, 0]
        var folds = msel_strat_fold_assign_res(route.helpers, held, _N, 3, 2, Int(rank.unsafe_ptr()), 3,
                                               Int(alloc.unsafe_ptr()), 0, Int(counts.unsafe_ptr()),
                                               Int(sizes.unsafe_ptr()))
        _ = msel_store_free(route.helpers, held)
        require(sizes[0] == 5 and sizes[1] == 5, "strat_fold_assign_res: fold sizes " + String(sizes[0]) + ", "
                + String(sizes[1]))
        _two_folds(route, folds, sizes, out)
        _ = msel_store_free(route.helpers, folds)
        _ = rank^  # read through the addresses above
        _ = alloc^
        _ = counts^
        _ = sizes^
    elif unit == 3:
        var held = msel_store_put(route.helpers, Int(codes.unsafe_ptr()), 4 * _N)
        var counts: List[Int64] = [9, 9, 9]
        var rank: List[Int32] = [9, 9, 9]
        var present = msel_first_seen_res(route.helpers, held, _N, 3, Int(counts.unsafe_ptr()),
                                          Int(rank.unsafe_ptr()))
        var bad: List[Int32] = [0, 3]
        var bad_id = msel_store_put(route.helpers, Int(bad.unsafe_ptr()), 8)
        var refused = False
        try:
            _ = msel_first_seen_res(route.helpers, bad_id, 2, 3, Int(counts.unsafe_ptr()), 0)
        except problem:
            refused = "code out of range" in String(problem)
        _ = msel_store_free(route.helpers, bad_id)
        _ = msel_store_free(route.helpers, held)
        require(refused, "first_seen_res: a code outside [0, k) was not refused in the host run's words")
        # the refused call may have written counts: the words compared are the first call's, re-read
        held = msel_store_put(route.helpers, Int(codes.unsafe_ptr()), 4 * _N)
        present = msel_first_seen_res(route.helpers, held, _N, 3, Int(counts.unsafe_ptr()), Int(rank.unsafe_ptr()))
        _ = msel_store_free(route.helpers, held)
        out.append(Int64(present))
        for c in range(3):  # driver-loop: over the fixture's classes
            out.append(counts[c])
        for c in range(3):  # driver-loop: over the fixture's classes
            out.append(Int64(rank[c]))
        _ = bad^  # read through the address above
        _ = counts^
        _ = rank^
    elif unit == 4:
        var plan = msel_plan_stratified(route, labels, MSEL_LABELS_SORTED, 2, True, MselSeed(5, False))
        _fill_words(route, plan, out)
    elif unit == 5:
        for r in range(2):  # driver-loop: over repeats
            var plan = msel_plan_repeat(route, _N, labels, MSEL_LABELS_SORTED, 2, True, MselSeed(3, False), r)
            _fill_words(route, plan, out)
    elif unit == 6:
        var plan = msel_plan_strat_shuffle(route, labels, 2, 6, 4, MselSeed(11, False))
        _fill_words(route, plan, out)
    elif unit == 7:
        var plain = msel_plan_strat_group(route, _N, labels, MSEL_LABELS_SORTED, group_codes, 2, False,
                                          MselSeed(0, False))
        _fill_words(route, plain, out)
        var shuffled = msel_plan_strat_group(route, _N, labels, MSEL_LABELS_SORTED, group_codes, 2, True,
                                             MselSeed(13, False))
        _fill_words(route, shuffled, out)
    else:
        raise Error("no stratified stored-code case " + String(unit))
    _ = codes^  # read through the addresses above
    _ = groups^
    return out^


def _same_words(got: List[Int64], want: List[Int], what: String) raises:
    require(len(got) == len(want), what + ": " + String(len(got)) + " words, expected " + String(len(want)))
    for i in range(len(want)):  # driver-loop: over a literal expectation of a few words
        require(Int(got[i]) == want[i], what + ": word " + String(i) + " is " + String(Int(got[i])) + ", expected "
                + String(want[i]))


def _partition(words: List[Int64], splits: Int, sized_train: Int, sized_test: Int, whole: Bool, what: String) raises:
    """Every split's sides are disjoint rows of the fixture (all of them when
    `whole`), with the given sizes when they are not -1."""
    var at = 0
    for s in range(splits):  # driver-loop: over splits
        require(at + 2 <= len(words), what + ": split " + String(s) + " is missing")
        var n_train = Int(words[at])
        var n_test = Int(words[at + 1])
        at += 2
        require(sized_train < 0 or n_train == sized_train, what + ": train size " + String(n_train))
        require(sized_test < 0 or n_test == sized_test, what + ": test size " + String(n_test))
        var seen = List[Int](length=_N, fill=0)
        for i in range(n_train + n_test):  # driver-loop: over a fixture split's rows
            var r = Int(words[at + i])
            require(r >= 0 and r < _N, what + ": row " + String(r) + " out of range")
            seen[r] += 1
        at += n_train + n_test
        for r in range(_N):  # driver-loop: over the fixture's rows
            require(seen[r] <= 1, what + ": row " + String(r) + " is on both sides")
            require(not whole or seen[r] == 1, what + ": row " + String(r) + " is on neither side")
    require(at == len(words), what + ": words past the last split")


def _groups_whole(words: List[Int64], splits: Int, what: String) raises:
    """No group of the fixture has rows on both sides of a split."""
    var groups = _groups()
    var at = 0
    for s in range(splits):  # driver-loop: over splits
        var n_train = Int(words[at])
        var n_test = Int(words[at + 1])
        at += 2
        var side = List[Int](length=5, fill=-1)
        for i in range(n_train + n_test):  # driver-loop: over a fixture split's rows
            var g = Int(groups[Int(words[at + i])])
            var here = 0 if i < n_train else 1
            require(side[g] < 0 or side[g] == here, what + ": group " + String(g) + " is split, split " + String(s))
            side[g] = here
        at += n_train + n_test


def _check(unit: Int, words: List[Int64]) raises:
    """The literal words and the properties of one case (every column)."""
    var folds: List[Int] = [5, 5, 4, 6, 7, 8, 9, 0, 1, 2, 3, 5, 5, 5, 0, 1, 2, 3, 5, 4, 6, 7, 8, 9]
    if unit == 0:
        _same_words(words, folds, "StratifiedKFold(2)")
    elif unit == 1:
        _same_words(words, folds, "default stratified folds")
    elif unit == 2:
        _same_words(words, folds, "strat_fold_assign_res")
    elif unit == 3:
        var want: List[Int] = [3, 5, 3, 2, 1, 0, 2]
        _same_words(words, want, "first_seen_res")
    elif unit == 4:
        _partition(words, 2, 5, 5, True, "StratifiedKFold(2, shuffle)")
    elif unit == 5:
        _partition(words, 4, 5, 5, True, "RepeatedStratifiedKFold(2)")
    elif unit == 6:
        _partition(words, 2, 6, 4, False, "StratifiedShuffleSplit")
    elif unit == 7:
        _partition(words, 4, -1, -1, True, "StratifiedGroupKFold(2)")
        _groups_whole(words, 4, "StratifiedGroupKFold(2)")


def _run_cpu() raises:
    publish_msel_helpers_host()
    publish_base_helpers_host()
    var host = _route(msel_helpers_ref(MselEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)))

    def task(unit: Int) raises {imm host}:
        _check(unit, _case(host, unit))

    check_cases_serial("stratified_res", STRAT_RES_CASES, task)


def _run_gpu() raises:
    var vendor = msel_entry_vendor(COMPILED_VENDOR)
    require(vendor >= 1, "msel_split_stratified_res: this build has no GPU vendor")
    publish_msel_helpers_host()
    publish_msel_helpers_device()
    publish_base_helpers_host()
    var host = _route(msel_helpers_ref(MselEntryKey(Int(GLOBAL_NUMERIC_MODE), 0)))
    var device = _route(msel_helpers_ref(MselEntryKey(Int(GLOBAL_NUMERIC_MODE), vendor)))

    # ordered: the device cases share the one store context
    def task(unit: Int) raises {imm host, imm device}:
        var on_host = _case(host, unit)
        var on_device = _case(device, unit)
        require(len(on_host) == len(on_device), "case " + String(unit) + ": device and host words differ in length")
        for i in range(len(on_host)):  # driver-loop: over a fixture case's words
            require(on_host[i] == on_device[i], "case " + String(unit) + ": device word " + String(i) + " is "
                    + String(on_device[i]) + ", host " + String(on_host[i]))
        _check(unit, on_device)

    check_cases_serial("stratified_res", STRAT_RES_CASES, task)


def test_stratified_res() raises:
    comptime if CPU_COLUMN_BUILD:
        _run_cpu()
    else:
        _run_gpu()


def main() raises:
    # The test by name, not discover_tests: the device column's functions do not compile in the host column
    # build.
    var suite = TestSuite()
    suite.test[test_stratified_res]()
    suite^.run()

# SPDX-License-Identifier: Apache-2.0
"""THE SPLITTER SPEC (lane mf-split5, 2026-11-23): one of OUR splitters as
typed words, and the folds a member fit computes from them over its own rows.

A meta-estimator (Stacking, CalibratedClassifierCV, the search kind) or an
x_linear CV member whose `cv` is one of our splitters reads the splitter ONCE
at UNPACK into a `MselSplitSpec` (bindings/msel_split_spec_unpack.mojo
`msel_split_spec_of`: the same stored fields, the same seed read and the same
entries the public split door reads for `split`). Each FIT then calls
`msel_spec_sides` over its own n rows, its int32 class codes and its group
codes, which makes the splits with THE SAME plan and fill the public door
uses (bindings/msel_split_plan_run.mojo: `msel_plan_<splitter>` and
`msel_split_fill`; a repeat is `msel_plan_repeat`, the sizes of the shuffle
family `msel_shuffle_sizes_run`): one body, two callers. The refusals are
the door's, in its words and order; a splitter warning (StratifiedKFold's
"least populated class") is returned as its operands for the caller's fit
notice.

Groups: the public door takes them from `split(X, y, groups)`. A member fit
takes them from its call (`MemberCall.groups`, int32 or int64 ids, encoded in
sorted order here as the label encoder sorts them); Stacking and
CalibratedClassifierCV hand their splitter none (scikit-learn's
`cross_val_predict(..., cv=cv)` and `cv.split(X, y)` pass no groups), so a
group splitter there is refused in scikit-learn's own words, "The 'groups'
parameter should not be None.", as the door refuses it.

No Python object is read, built or called here.
"""
from std.sys.info import size_of
from std.builtin.sort import sort
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from core.entry_table import EntryRef, entry_ref_unresolved
from core.label_encode import host_unique_inverse, LABEL_KIND_F64
from bindings.msel_split_plan_run import (
    MselSeed, MselSplitRoute, MselSplitPlan, MselCodes, MselSizeWord, msel_size_none, msel_split_route_none,
    msel_split_fill, msel_code_counts, MSEL_LABELS_SORTED, msel_plan_kfold, msel_plan_stratified,
    msel_plan_group_kfold, msel_plan_strat_group, msel_plan_time_series, msel_plan_shuffle,
    msel_plan_group_shuffle, msel_plan_strat_shuffle, msel_plan_select, msel_plan_group_set,
    msel_plan_leave_one_out, msel_plan_leave_p_out, msel_repeated_check, msel_plan_repeat,
    msel_shuffle_sizes_run,
)

comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]

#: The splitter of a spec: the public door's SPLITTER_* codes
#: (bindings/msel_split_door.mojo), the same numbers.
comptime MSEL_SPEC_NONE = 0
comptime MSEL_SPEC_KFOLD = 1
comptime MSEL_SPEC_STRATIFIED_KFOLD = 2
comptime MSEL_SPEC_GROUP_KFOLD = 3
comptime MSEL_SPEC_STRATIFIED_GROUP_KFOLD = 4
comptime MSEL_SPEC_TIME_SERIES = 5
comptime MSEL_SPEC_LOO = 6
comptime MSEL_SPEC_LPO = 7
comptime MSEL_SPEC_LOGO = 8
comptime MSEL_SPEC_LPGO = 9
comptime MSEL_SPEC_PREDEFINED = 10
comptime MSEL_SPEC_SHUFFLE = 11
comptime MSEL_SPEC_GROUP_SHUFFLE = 12
comptime MSEL_SPEC_STRATIFIED_SHUFFLE = 13
comptime MSEL_SPEC_REPEATED_KFOLD = 14
comptime MSEL_SPEC_REPEATED_STRATIFIED_KFOLD = 15
#: A user's (train, test) index pairs (a list, or an `_IterableCV`'s stored
#: pairs; the door's `_IterableCV` code), read once at UNPACK.
comptime MSEL_SPEC_LISTED = 16

comptime _CHUNK = 64
comptime _INDEX_SPAN = 16777216


def msel_spec_stratified(kind: Int) -> Bool:
    """The splitters that read the class labels (on a regressor, its
    targets' values: `msel_spec_sides_targets`, lane mf-del5)."""
    return (kind == MSEL_SPEC_STRATIFIED_KFOLD or kind == MSEL_SPEC_STRATIFIED_GROUP_KFOLD
            or kind == MSEL_SPEC_STRATIFIED_SHUFFLE or kind == MSEL_SPEC_REPEATED_STRATIFIED_KFOLD)


def msel_spec_has_n_splits(kind: Int) -> Bool:
    """The splitters whose scikit-learn class stores an `n_splits` attribute
    (`_BaseKFold`: the KFold family and TimeSeriesSplit; `BaseShuffleSplit`:
    the shuffle family). LeaveOneOut, LeavePOut, the group-out splitters,
    PredefinedSplit, the repeated splitters and an iterable store none
    (scikit-learn's `hasattr(cv, "n_splits")` is False for them)."""
    return (kind == MSEL_SPEC_KFOLD or kind == MSEL_SPEC_STRATIFIED_KFOLD or kind == MSEL_SPEC_GROUP_KFOLD
            or kind == MSEL_SPEC_STRATIFIED_GROUP_KFOLD or kind == MSEL_SPEC_TIME_SERIES
            or kind == MSEL_SPEC_SHUFFLE or kind == MSEL_SPEC_GROUP_SHUFFLE or kind == MSEL_SPEC_STRATIFIED_SHUFFLE)


def msel_spec_grouped(kind: Int) -> Bool:
    """The splitters that read `groups`."""
    return (kind == MSEL_SPEC_GROUP_KFOLD or kind == MSEL_SPEC_STRATIFIED_GROUP_KFOLD or kind == MSEL_SPEC_LOGO
            or kind == MSEL_SPEC_LPGO or kind == MSEL_SPEC_GROUP_SHUFFLE)


struct MselSplitSpec(Copyable, Movable, Defaultable):
    """One of our splitters as UNPACK read it (`kind` MSEL_SPEC_*; NONE: no
    splitter). `k`: n_splits (the KFold family, TimeSeriesSplit, the
    shuffle family's split count, a repeated splitter's per-repeat folds),
    p (LeavePOut), n_groups (LeavePGroupsOut). `shuffle` / `seed`: the
    KFold family's; the shuffle family and a repeated splitter draw from
    `seed` (a repeated splitter's base seed). `repeats`: n_repeats. The
    shuffle family's sizes after the default substitution (`test`,
    `train`, `sum_over`, `sum_text`). TimeSeriesSplit's test_size (`ts`,
    `has_ts`), gap and max_train_size (0: none). PredefinedSplit's test_fold
    as sorted-order codes (`folds`, `fold_classes` of them) and the codes it
    splits on (`wants`: every fold value but -1). `route`: the entries the
    splitters call (pinned by the admission); `sabotage`: the door's
    dormant negative control word. LISTED: the user's pairs (`listed_train`,
    `listed_test`, int64 rows as given). `int_cv`: the spec stands for an
    integer cv (an unshuffled KFold of `k` folds; its caller applies its own
    integer rules first). `rows_helpers`: the caller's base helper table
    (an x_linear CV member's, for its fold-id and role passes)."""

    var kind: Int
    var k: Int
    var shuffle: Bool
    var seed: MselSeed
    var repeats: Int
    var test: MselSizeWord
    var train: MselSizeWord
    var sum_over: Bool
    var sum_text: String
    var ts: Int
    var has_ts: Bool
    var gap: Int
    var max_train: Int
    var folds: List[Int32]
    var fold_classes: Int
    var wants: List[Int]
    var route: MselSplitRoute
    var sabotage: Bool
    var listed_train: List[List[Int64]]
    var listed_test: List[List[Int64]]
    var int_cv: Bool
    var rows_helpers: EntryRef
    #: TimeSeriesSplit (lane mf-lin6, a strict reader): a max_train_size the
    #: door could not read, and its words; raised at fit after the plan's own
    #: refusals and only when the plan makes a split, as the door does.
    var max_train_unread: Bool
    var max_train_refusal: String

    def __init__(out self):
        self.kind = MSEL_SPEC_NONE
        self.k = 0
        self.shuffle = False
        self.seed = MselSeed(0, False)
        self.repeats = 0
        self.test = msel_size_none()
        self.train = msel_size_none()
        self.sum_over = False
        self.sum_text = String("")
        self.ts = 0
        self.has_ts = False
        self.gap = 0
        self.max_train = 0
        self.folds = List[Int32]()
        self.fold_classes = 0
        self.wants = List[Int]()
        self.route = msel_split_route_none()
        self.sabotage = False
        self.listed_train = List[List[Int64]]()
        self.listed_test = List[List[Int64]]()
        self.int_cv = False
        self.rows_helpers = entry_ref_unresolved()
        self.max_train_unread = False
        self.max_train_refusal = String("")


struct MselGroupCodes(Movable):
    """A fit's group ids as sorted-order int32 codes (the label encoder's
    order): `codes` one per row, `values` the distinct ids in sorted order.
    `given` False: the fit has no groups."""

    var codes: List[Int32]
    var values: List[Int]
    var n: Int
    var given: Bool

    def __init__(out self):
        self.codes = List[Int32]()
        self.values = List[Int]()
        self.n = 0
        self.given = False

    def m(self) -> Int:
        return len(self.values)

    def view(self) -> MselCodes:
        return MselCodes(Int(self.codes.unsafe_ptr()), self.n, len(self.values))

    def text(self) -> String:
        """The encoder's classes as the door's message prints them (a list
        of Python ints)."""
        var out = String("[")
        for g in range(len(self.values)):  # small-loop(the distinct group ids, for a refusal's words)
            if g > 0:
                out += ", "
            out += String(self.values[g])
        out += "]"
        return out^


def msel_group_codes_run(address: Int, wide: Bool, n: Int) raises -> MselGroupCodes:
    """The n group ids at `address` (int64 when `wide`, else int32) as
    sorted-order codes: the distinct ids sorted, then each row's rank by a
    binary search, rows split across the host's workers."""
    var out = MselGroupCodes()
    out.given = True
    out.n = n
    if n == 0:
        return out^
    var ids = List[Int](length=n, fill=0)
    if wide:
        var src = _I64P(unsafe_from_address=address)
        for i in range(n):  # small-loop(one read per row of the ids, once per fit)
            ids[i] = Int(src[i])
    else:
        var src32 = _I32P(unsafe_from_address=address)
        for i in range(n):  # small-loop(one read per row of the ids, once per fit)
            ids[i] = Int(src32[i])
    var sorted_ids = ids.copy()
    sort(sorted_ids)
    for i in range(n):  # small-loop(the distinct ids of the sorted copy)
        if i == 0 or sorted_ids[i] != sorted_ids[i - 1]:
            out.values.append(sorted_ids[i])
    out.codes = List[Int32](length=n, fill=Int32(0))
    var m = len(out.values)
    var values_at = Int(out.values.unsafe_ptr())
    var ids_at = Int(ids.unsafe_ptr())
    var codes_at = Int(out.codes.unsafe_ptr())
    var tasks = host_predict_task_count_work(n, n)
    var chunk = host_predict_chunk(n, tasks)

    def rank(worker: Int) {imm values_at, imm ids_at, imm codes_at, imm chunk, imm n, imm m}:
        var values = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=values_at)
        var rows = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=ids_at)
        var dst = _I32P(unsafe_from_address=codes_at)
        var begin = worker * chunk
        var end = min(begin + chunk, n)
        for i in range(begin, end):
            var lo = 0
            var hi = m - 1
            var v = rows[i]
            while lo < hi:  # small-loop(a binary search over the distinct ids)
                var mid = (lo + hi) // 2
                if values[mid] < v:
                    lo = mid + 1
                else:
                    hi = mid
            dst[i] = Int32(lo)

    host_parallelize(rank, tasks)
    _ = len(ids)  # read through its address by the workers above
    return out^


struct MselSides(Movable):
    """Every split of one fit, in split order: the int64 train and test rows
    of each, and one entry per splitter warning (the least populated class's
    row count, for StratifiedKFold's "least populated class" notice)."""

    var train: List[List[Int64]]
    var test: List[List[Int64]]
    var least: List[Int]

    def __init__(out self):
        self.train = List[List[Int64]]()
        self.test = List[List[Int64]]()
        self.least = List[Int]()

    def splits(self) -> Int:
        return len(self.test)


def _least_held(codes: Int, n: Int, classes: Int) -> Int:
    """The row count of the least populated class the n int32 codes hold
    (-1: none), the operand of StratifiedKFold's warning."""
    var counts = List[Int](length=max(classes, 1), fill=0)
    var src = _I32P(unsafe_from_address=codes)
    for i in range(n):  # small-loop(one count per row, only when the splitter warned)
        var c = Int(src[i])
        if c >= 0 and c < classes:
            counts[c] += 1
    var least = -1
    for c in range(classes):  # small-loop(the class counts: the least one held)
        if counts[c] > 0 and (least < 0 or counts[c] < least):
            least = counts[c]
    return least


def _take(route: MselSplitRoute, var plan: MselSplitPlan, mut sides: MselSides, codes: Int, n: Int,
          classes: Int) raises:
    """One plan's splits into `sides`: its warning's operand, one int64
    (train, test) pair per split sized by the plan, the plan's fill."""
    if plan.warned:
        sides.least.append(_least_held(codes, n, classes))
    var first = sides.splits()
    for s in range(plan.splits()):  # driver-loop: over the plan's splits (one pair of sides each)
        sides.train.append(List[Int64](length=max(plan.train_sizes[s], 1), fill=Int64(0)))
        sides.test.append(List[Int64](length=max(plan.test_sizes[s], 1), fill=Int64(0)))
    for s in range(plan.splits()):  # driver-loop: over the plan's splits (their final addresses)
        plan.train_dst.append(Int(sides.train[first + s].unsafe_ptr()))
        plan.test_dst.append(Int(sides.test[first + s].unsafe_ptr()))
    msel_split_fill(route, plan)
    # The sides hold max(size, 1) words: trim each to the plan's size.
    for s in range(plan.splits()):  # driver-loop: over the plan's splits (their exact lengths)
        sides.train[first + s].resize(plan.train_sizes[s], Int64(0))
        sides.test[first + s].resize(plan.test_sizes[s], Int64(0))


def _present_codes(codes: Int, n: Int, classes: Int, mut dst: List[Int32]) -> Int:
    """The n codes renumbered over the classes they hold, in sorted order
    (the label encoder's classes of this fit's labels: a class a caller's
    code space has but these rows lack is not one). Returns the count."""
    var src = _I32P(unsafe_from_address=codes)
    var seen = List[Int32](length=max(classes, 1), fill=Int32(-1))
    for i in range(n):  # small-loop(one mark per row)
        var c = Int(src[i])
        if c >= 0 and c < classes:
            seen[c] = 0
    var m = 0
    for c in range(classes):  # small-loop(the held classes, in order)
        if seen[c] == 0:
            seen[c] = Int32(m)
            m += 1
    dst = List[Int32](length=max(n, 1), fill=Int32(0))
    for i in range(n):  # small-loop(one renumbering per row)
        dst[i] = seen[Int(src[i])]
    return m


def _combinations(m: Int, r: Int) raises -> List[List[Int]]:
    """itertools.combinations(range(m), r), in its lexicographic order."""
    if r < 0:
        raise Error("r must be non-negative")
    var out = List[List[Int]]()
    if r > m:
        return out^
    var combo = List[Int](capacity=max(r, 1))
    for i in range(r):  # small-loop(the first combination)
        combo.append(i)
    while True:  # driver-loop: over the combinations
        out.append(combo.copy())
        var at = r - 1
        while at >= 0 and combo[at] == m - r + at:  # small-loop(the last position that can still move)
            at -= 1
        if at < 0:
            break
        combo[at] += 1
        for j in range(at + 1, r):  # small-loop(the positions after it)
            combo[j] = combo[j - 1] + 1
    return out^


def msel_spec_sides(spec: MselSplitSpec, n: Int, codes: Int, classes: Int,
                    groups: MselGroupCodes) raises -> MselSides:
    """RUN: the splits of `spec` over THIS fit's n rows (int32 class codes at
    `codes` over `classes`, 0 when the caller has none; `groups` the fit's
    group codes), by the public door's plans and fill, with its refusals in
    its words."""
    var route = spec.route
    var kind = spec.kind
    var sides = MselSides()
    if msel_spec_grouped(kind):
        if not groups.given:
            raise Error("The 'groups' parameter should not be None.")
        if groups.n != n:
            raise Error("groups has " + String(groups.n) + " labels for " + String(n) + " samples")
    if msel_spec_stratified(kind) and codes == 0:
        raise Error("RuntimeError: mojolearn: a stratified splitter was given no class codes")
    var labels = MselCodes(codes, n, classes)
    if kind == MSEL_SPEC_KFOLD:
        _take(route, msel_plan_kfold(route, n, spec.k, spec.shuffle, spec.seed, spec.sabotage), sides, codes, n,
              classes)
    elif kind == MSEL_SPEC_STRATIFIED_KFOLD:
        if spec.shuffle and spec.seed.bad:
            # The door's order: the class-size refusal, then the seed's.
            _ = msel_plan_stratified(route, labels, MSEL_LABELS_SORTED, spec.k, False, spec.seed)
            _ = spec.seed.rng()
        _take(route, msel_plan_stratified(route, labels, MSEL_LABELS_SORTED, spec.k, spec.shuffle, spec.seed),
              sides, codes, n, classes)
    elif kind == MSEL_SPEC_GROUP_KFOLD:
        _take(route, msel_plan_group_kfold(route, groups.view(), spec.k, spec.shuffle, spec.seed), sides, codes, n,
              classes)
    elif kind == MSEL_SPEC_STRATIFIED_GROUP_KFOLD:
        _take(route, msel_plan_strat_group(route, n, labels, MSEL_LABELS_SORTED, groups.view(), spec.k, spec.shuffle,
                                           spec.seed), sides, codes, n, classes)
    elif kind == MSEL_SPEC_TIME_SERIES:
        var plan = msel_plan_time_series(n, spec.k, spec.ts, spec.has_ts, spec.gap, spec.max_train)
        if spec.max_train_unread and plan.splits() > 0:
            # The door's order (`msel_time_series_split`): the plan's refusals
            # first, then an unreadable max_train_size, only for a split.
            raise Error(spec.max_train_refusal)
        _take(route, plan^, sides, codes, n, classes)
    elif kind == MSEL_SPEC_LOO:
        _take(route, msel_plan_leave_one_out(n), sides, codes, n, classes)
    elif kind == MSEL_SPEC_LPO:
        _take(route, msel_plan_leave_p_out(n, spec.k), sides, codes, n, classes)
    elif kind == MSEL_SPEC_LOGO:
        var m = groups.m()
        if m <= 1:
            raise Error("The groups parameter contains fewer than 2 unique groups (" + groups.text() + "). "
                        + "LeaveOneGroupOut expects at least 2.")
        var wants = List[Int](capacity=m)
        for g in range(m):  # small-loop(one split per group, in sorted group order)
            wants.append(g)
        _take(route, msel_plan_select(route, groups.view(), wants, True, List[Int]()), sides, codes, n, classes)
    elif kind == MSEL_SPEC_LPGO:
        var m = groups.m()
        if spec.k >= m:
            raise Error("The groups parameter contains fewer than (or equal to) n_groups (" + String(spec.k)
                        + ") numbers of unique groups (" + groups.text() + ").")
        var counts = msel_code_counts(route, groups.view())
        var combos = _combinations(m, spec.k)
        for c in range(len(combos)):  # driver-loop: over the combinations of groups (one split each)
            var size = 0
            for i in range(len(combos[c])):  # small-loop(the chosen groups of one combination)
                size += Int(counts[combos[c][i]])
            _take(route, msel_plan_group_set(route, groups.view(), combos[c], size), sides, codes, n, classes)
    elif kind == MSEL_SPEC_PREDEFINED:
        var view = MselCodes(Int(spec.folds.unsafe_ptr()), len(spec.folds), spec.fold_classes)
        _take(route, msel_plan_select(route, view, spec.wants, True, List[Int]()), sides, codes, n, classes)
        _ = len(spec.folds)  # the select read the codes through their address
    elif kind == MSEL_SPEC_SHUFFLE:
        var sizes = msel_shuffle_sizes_run(n, spec.test, spec.train, spec.sum_over, spec.sum_text)
        _take(route, msel_plan_shuffle(route, n, spec.k, sizes[0], sizes[1], spec.seed), sides, codes, n, classes)
    elif kind == MSEL_SPEC_GROUP_SHUFFLE:
        var sizes = msel_shuffle_sizes_run(groups.m(), spec.test, spec.train, spec.sum_over, spec.sum_text)
        _take(route, msel_plan_group_shuffle(route, groups.view(), spec.k, sizes[0], sizes[1], spec.seed), sides,
              codes, n, classes)
    elif kind == MSEL_SPEC_STRATIFIED_SHUFFLE:
        var held = List[Int32]()
        var m = _present_codes(codes, n, classes, held)
        var sizes = msel_shuffle_sizes_run(n, spec.test, spec.train, spec.sum_over, spec.sum_text)
        _take(route, msel_plan_strat_shuffle(route, MselCodes(Int(held.unsafe_ptr()), n, m), spec.k, sizes[0],
                                             sizes[1], spec.seed), sides, codes, n, classes)
        _ = len(held)  # the plan and its fill read the codes through their address
    elif kind == MSEL_SPEC_REPEATED_KFOLD or kind == MSEL_SPEC_REPEATED_STRATIFIED_KFOLD:
        msel_repeated_check(spec.seed)
        var stratified = kind == MSEL_SPEC_REPEATED_STRATIFIED_KFOLD
        for r in range(spec.repeats):  # driver-loop: over repeats (one plan and one fill each, as the door)
            _take(route, msel_plan_repeat(route, n, labels, MSEL_LABELS_SORTED, spec.k, stratified, spec.seed, r),
                  sides, codes, n, classes)
    elif kind == MSEL_SPEC_LISTED:
        for s in range(len(spec.listed_test)):  # driver-loop: over the user's pairs (copies; the caller checks them)
            sides.train.append(spec.listed_train[s].copy())
            sides.test.append(spec.listed_test[s].copy())
    else:
        raise Error("RuntimeError: mojolearn: a splitter spec of unknown kind " + String(kind))
    _ = len(groups.codes)  # the plans and fills read the group codes through their address
    return sides^


struct MselTargetCodes(Movable):
    """A regressor's targets as the splitter door's label encoder codes them
    (`msel_target_codes_run`): one int32 code per row, `m` distinct values."""

    var codes: List[Int32]
    var m: Int

    def __init__(out self):
        self.codes = List[Int32]()
        self.m = 0


#: THE GIVEN y's element types (`msel_target_codes_run`'s `given_dtype`):
#: the values of core/member_abi.mojo's MEMBER_F64 / MEMBER_I32 / MEMBER_I64
#: (this module does not import the member ABI; checks/native_tests/
#: msel_search_run.mojo case 20 asserts the values are equal). Any other:
#: not given.
comptime MSEL_GIVEN_F64 = 1
comptime MSEL_GIVEN_I32 = 2
comptime MSEL_GIVEN_I64 = 3


def msel_target_codes_run(y: Int, n: Int, given: Int, given_dtype: Int) raises -> MselTargetCodes:
    """RUN (lanes mf-del5, mf-tail5): a regressor's n targets as sorted-order
    int32 codes, the labels the splitter door makes of a regressor's y for a
    stratified splitter (bindings/msel_split_door.mojo `_strat_labels`:
    `msel_label_codes`, MSEL_LABELS_SORTED: np.unique's inverse over the y
    it was GIVEN). `given` (0: none) is that y as the caller's UNPACK was
    given it (`given_dtype`: MSEL_GIVEN_F64 / _I32 / _I64 words; core/
    member_abi.mojo `MemberCall.y_given`); without it the float32 targets at
    `y` are the given y. Either is widened to float64 across the host cores
    (exact for float32, int32 and |int64| <= 2^53, the door's float64 order),
    then coded by core/label_encode.mojo `host_unique_inverse` (parallel;
    -0.0 and +0.0 are one value): two float64 targets equal in float32 stay
    two classes, as at the door. A NaN target is refused in check_array's
    words."""
    var out = MselTargetCodes()
    out.codes = List[Int32](length=max(n, 1), fill=Int32(0))
    if n <= 0:
        return out^
    var wide = List[Float64](length=n, fill=0.0)
    var classes = List[UInt64](length=n, fill=UInt64(0))
    var wide_at = Int(wide.unsafe_ptr())
    var tasks = host_predict_task_count_work(n, n)
    var chunk = host_predict_chunk(n, tasks)
    var kind = given_dtype if given != 0 else 0

    def widen(t: Int) {imm y, imm given, imm kind, imm wide_at, imm chunk, imm n}:
        var dst = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=wide_at)
        var begin = t * chunk
        var end = min(begin + chunk, n)
        if kind == MSEL_GIVEN_F64:
            var src64 = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=given)
            for i in range(begin, end):
                dst[i] = src64[i]
        elif kind == MSEL_GIVEN_I64:
            var srci64 = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=given)
            for i in range(begin, end):
                dst[i] = Float64(srci64[i])
        elif kind == MSEL_GIVEN_I32:
            var srci32 = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=given)
            for i in range(begin, end):
                dst[i] = Float64(srci32[i])
        else:
            var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=y)
            for i in range(begin, end):
                dst[i] = Float64(src[i])

    host_parallelize(widen, max(tasks, 1))
    var m = host_unique_inverse(MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=wide_at), n,
                                LABEL_KIND_F64,
                                MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=Int(classes.unsafe_ptr())),
                                MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(out.codes.unsafe_ptr())))
    _ = len(wide)  # read through its address by the workers and the encoder above
    _ = len(classes)
    if m < 0:
        raise Error("Input y contains NaN.")
    out.m = m
    return out^


def msel_spec_sides_targets(spec: MselSplitSpec, n: Int, y: Int, given: Int, given_dtype: Int,
                            groups: MselGroupCodes) raises -> MselSides:
    """RUN (lanes mf-del5, mf-tail5): `msel_spec_sides` for a REGRESSOR's
    fit (no class codes; its n float32 targets at `y`, and the y it was
    given at `given`, 0: none, `msel_target_codes_run`). A stratified
    splitter stratifies on the targets' values, as the splitter door does
    with a regressor's y: continuous targets meet its refusal ("n_splits=
    ... cannot be greater than the number of members in each class.") or
    its least-populated warning in its words, repeated values its folds.
    Any other splitter reads no labels."""
    if not msel_spec_stratified(spec.kind):
        return msel_spec_sides(spec, n, 0, 0, groups)
    var coded = msel_target_codes_run(y, n, given, given_dtype)
    var sides = msel_spec_sides(spec, n, Int(coded.codes.unsafe_ptr()), coded.m, groups)
    _ = len(coded.codes)  # the plans and fills read the codes through their address
    return sides^


def msel_sides_i32(sides: MselSides, mut train: List[List[Int32]], mut test: List[List[Int32]]):
    """The int64 sides as int32 rows (each row is below n, an int32 row
    count: one exact narrowing each), every split on the host's workers."""
    var splits = sides.splits()
    train = List[List[Int32]](capacity=splits)
    test = List[List[Int32]](capacity=splits)
    for s in range(splits):  # driver-loop: over the splits (their int32 sides)
        train.append(List[Int32](length=len(sides.train[s]), fill=Int32(0)))
        test.append(List[Int32](length=len(sides.test[s]), fill=Int32(0)))
    var src_tr = List[Int](capacity=max(splits, 1))
    var src_te = List[Int](capacity=max(splits, 1))
    var dst_tr = List[Int](capacity=max(splits, 1))
    var dst_te = List[Int](capacity=max(splits, 1))
    var n_tr = List[Int](capacity=max(splits, 1))
    var n_te = List[Int](capacity=max(splits, 1))
    for s in range(splits):  # driver-loop: over the splits (the narrowing's addresses)
        src_tr.append(Int(sides.train[s].unsafe_ptr()))
        src_te.append(Int(sides.test[s].unsafe_ptr()))
        dst_tr.append(Int(train[s].unsafe_ptr()))
        dst_te.append(Int(test[s].unsafe_ptr()))
        n_tr.append(len(sides.train[s]))
        n_te.append(len(sides.test[s]))
    var src_tr_at = Int(src_tr.unsafe_ptr())
    var src_te_at = Int(src_te.unsafe_ptr())
    var dst_tr_at = Int(dst_tr.unsafe_ptr())
    var dst_te_at = Int(dst_te.unsafe_ptr())
    var n_tr_at = Int(n_tr.unsafe_ptr())
    var n_te_at = Int(n_te.unsafe_ptr())

    def narrow(split: Int) {imm src_tr_at, imm src_te_at, imm dst_tr_at, imm dst_te_at, imm n_tr_at, imm n_te_at}:
        var st = _I64P(unsafe_from_address=MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=src_tr_at)[split])
        var se = _I64P(unsafe_from_address=MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=src_te_at)[split])
        var dt = _I32P(unsafe_from_address=MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=dst_tr_at)[split])
        var de = _I32P(unsafe_from_address=MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=dst_te_at)[split])
        var count_tr = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=n_tr_at)[split]
        var count_te = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=n_te_at)[split]
        for i in range(count_tr):  # small-loop(one split's train rows)
            dt[i] = Int32(st[i])
        for i in range(count_te):  # small-loop(one split's test rows)
            de[i] = Int32(se[i])

    host_parallelize(narrow, splits)
    _ = len(src_tr)  # read through their addresses by the workers above
    _ = len(src_te)
    _ = len(dst_tr)
    _ = len(dst_te)
    _ = len(n_tr)
    _ = len(n_te)


# ------------------------------------------------------------ the spec store
#
# A member whose parameter words are numbers (the x_linear CV members) holds
# its spec by handle: the admission mints it, its capsule
# (bindings/msel_split_spec_pack.mojo) releases it, each FIT copies it under
# the store's lock. Serial by nature: the store's bookkeeping.


struct _SpecSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var spec: MselSplitSpec

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.spec = MselSplitSpec()


struct _SpecStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[_SpecSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[_SpecSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _SPECS = SharedSlot[_SpecStore, "MojolearnMselSplitSpecStoreV1", 1]
comptime _SpecSlotPtr = MutPointer[_SpecSlot, MutUntrackedOrigin]


def _spec_lock() raises -> MutPointer[Int64, MutUntrackedOrigin]:
    var store = _SPECS.get_or_create_ptr()
    return rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())


def _spec_address(index: Int) raises -> Int:
    var store = _SPECS.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[_SpecSlot]()


def msel_spec_mint(spec: MselSplitSpec) raises -> Int:
    """UNPACK: keep a copy of `spec`; the handle (>= 1) goes into the
    member's words. Released by `msel_spec_release`."""
    var store = _SPECS.get_or_create_ptr()
    var lock = _spec_lock()
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live splitter specs")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[_SpecSlot](length=_CHUNK, fill=_SpecSlot()))
            index = store[].count
            store[].count += 1
        var slot = _SpecSlotPtr(unsafe_from_address=_spec_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].spec = spec.copy()
        return slot[].generation * _INDEX_SPAN + index + 1
    finally:
        shared_slot_unlock(lock)


def _spec_slot(handle: Int) raises -> Int:
    var store = _SPECS.get_or_create_ptr()
    var index = handle % _INDEX_SPAN - 1
    var generation = handle // _INDEX_SPAN
    if handle <= 0 or index < 0 or index >= store[].count:
        raise Error("RuntimeError: mojolearn: a splitter spec handle is not live")
    var slot = _SpecSlotPtr(unsafe_from_address=_spec_address(index))
    if not slot[].live or slot[].generation != generation:
        raise Error("RuntimeError: mojolearn: a splitter spec handle is not live")
    return Int(slot)


def msel_spec_copy(handle: Int) raises -> MselSplitSpec:
    """FIT: the spec of `handle`, copied under the store's lock."""
    var lock = _spec_lock()
    shared_slot_lock(lock)
    try:
        var slot = _SpecSlotPtr(unsafe_from_address=_spec_slot(handle))
        return slot[].spec.copy()
    finally:
        shared_slot_unlock(lock)


def msel_spec_release(handle: Int):
    """The admission's capsule destructor. Never raises."""
    try:
        var store = _SPECS.get_or_create_ptr()
        var lock = _spec_lock()
        shared_slot_lock(lock)
        try:
            var slot = _SpecSlotPtr(unsafe_from_address=_spec_slot(handle))
            slot[].live = False
            slot[].spec = MselSplitSpec()
            store[].free.append(handle % _INDEX_SPAN - 1)
        finally:
            shared_slot_unlock(lock)
    except:
        pass

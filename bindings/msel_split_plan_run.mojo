# SPDX-License-Identifier: Apache-2.0
"""RUN of the splitters (device wave, lane dw-msel-parallel, 2026-11-21):
every fold computation of bindings/msel_split_door.mojo on typed values.

One split operation is two typed calls with the door's allocation between
them, so the rows are written straight into the arrays the caller receives
(no copy of a finished fold):

  plan = msel_plan_<splitter>(route, <typed request>)   sizes of every split
  (the door allocates one (train, test) pair per split and records the
   addresses in plan.train_dst / plan.test_dst)
  msel_split_fill(route, plan)                          the rows

The door releases the GIL once around each call (before: once per helper
call). The helper calls, their arguments and their order are the ones the
door made; the x_metrics programs are built by bindings/msel_split_run.mojo
as before and run through `metrics_plan_execute`. Same draws in the same
order, same rows.

Scratch (fold words, counts, tables, masks, permutations) is native storage
owned by the plan or by the call; before, each was a zero-filled Python
array. Label codes are read through `MselCodes` (an address the door pins).

No Python object is read, built or called here.
"""
from std.memory import unsafe_memcpy
from std.builtin.sort import sort
from core.entry_table import EntryRef, entry_ref_unresolved
from core.native_descriptor_abi import NativeDescriptorCall
from core.msel_helpers_calls import (
    msel_arange_i64, msel_count_mask_u8, msel_select_mask_u8_i64, msel_mask_from_indices_u8, msel_leave_range_i64,
    msel_fold_ids, msel_select_fold_i64, msel_select_folds_i64, msel_gather_i32, msel_first_seen_i32,
    msel_strat_fold_assign_i32, msel_split_table_i32, msel_group_fold_perm_i32,
)
from core.base_helpers_calls import base_bincount_i64, base_bincount2_i32
# The host-only passes linked into the splitters' unit (lane calltable-msel).
from core.hotpath_impl_rows import group_fold_assign_i32, strat_alloc_i64, strat_group_plan_i32
from x_metrics.epilogue import first_rows_i32
from core.metrics_program_plan import MetricProgramPlan
from bindings.arena_prog_run import MetricArenaOwner, MetricProgramInput, metrics_plan_execute
from bindings.msel_split_run import (
    MSEL_SPLIT_GOLDEN, MSEL_SPLIT_NONE, MselCounterRng, MselFoldRows, msel_fold_rows_fits, msel_split_mix64,
    msel_split_signed32, msel_shuffled_fold_rows_plan, msel_stratified_fold_rows_plan,
)

comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]

#: How `msel_split_fill` writes the rows of a plan.
comptime MSEL_FILL_FOLDS = 1    # one `select_folds_i64` over the fold word of every row
comptime MSEL_FILL_COPY = 2     # each side is a finished int64 run (a program's arena)
comptime MSEL_FILL_INDEXED = 3  # each split's test rows are an int64 run: mask, then both sides
comptime MSEL_FILL_RANGES = 4   # each side is a row range
comptime MSEL_FILL_TABLES = 5   # each split has a per-group table: gather, then both sides
comptime MSEL_FILL_SELECT = 6   # each split is the rows whose word is one value
comptime MSEL_FILL_STRAT_SHUFFLE = 7  # each split is one x_metrics program

#: What the label codes handed to a stratified splitter are.
comptime MSEL_LABELS_ALIGNED = 0     # sorted-order codes, one per row: first-seen order by first rows
comptime MSEL_LABELS_SORTED = 1      # sorted-order codes: first-seen order by `first_seen_i32`
comptime MSEL_LABELS_FIRST_SEEN = 2  # codes already in first-seen order


def msel_seed_refusal() -> String:
    """The words for a random_state that is not None or a non-negative int."""
    return ("mojolearn model_selection: random_state must be None or a non-negative int; "
            + "numpy RandomState / Generator objects are NOT IMPLEMENTED (their stream "
            + "is numpy's, x_metrics/split.mojo DEVIATION 6108)")


def msel_mix_seed(seed: UInt64, r: Int) -> Int:
    """`_mix_seed(seed, r)`: repeat r's seed."""
    return Int(msel_split_mix64(seed * MSEL_SPLIT_GOLDEN + UInt64(0x632BE59BD9B4E019) * UInt64(r + 1)) >> 1)


@fieldwise_init
struct MselSeed(ImplicitlyCopyable, Movable):
    """A splitter's seed as UNPACK read it. `bad` is refused where the
    splitter first draws (after its size refusals, as before), not at
    UNPACK."""

    var value: UInt64
    var bad: Bool

    def rng(self) raises -> MselCounterRng:
        if self.bad:
            raise Error(msel_seed_refusal())
        return MselCounterRng(self.value, 0)


@fieldwise_init
struct MselSplitRoute(ImplicitlyCopyable, Movable):
    """The compiled entries one split operation calls, resolved once by
    UNPACK: the model-selection helper table, the base helper table and the
    x_metrics program runner with its two range switches. An entry the
    splitter never calls is `entry_ref_unresolved()`."""

    var helpers: EntryRef
    var base: EntryRef
    var program: NativeDescriptorCall
    var ranges: Bool
    var output_ranges: Bool


def msel_split_route_none() -> MselSplitRoute:
    var none = entry_ref_unresolved()
    return MselSplitRoute(none, none, none.call, False, False)


@fieldwise_init
struct MselCodes(ImplicitlyCopyable, Movable):
    """`n` int32 label codes at `address` over `m` classes. The door pins
    the storage for the operation."""

    var address: Int
    var n: Int
    var m: Int


struct MselProg(Movable):
    """A typed x_metrics program and the addresses of its buffer inputs (the
    typed twin of the door's `MetricDoorProg`, whose pins are Python
    owners)."""

    var plan: MetricProgramPlan
    var addresses: List[Int]
    var sizes: List[Int]

    def __init__(out self):
        self.plan = MetricProgramPlan()
        self.addresses = List[Int]()
        self.sizes = List[Int]()

    def alloc(mut self, n: Int) raises -> Int:
        return self.plan.alloc(n)

    def put_i32(mut self, address: Int, n: Int) raises -> Int:
        """`n` int32 words at `address`, copied in as their bits."""
        var off = self.plan.input(n)
        self.addresses.append(address)
        self.sizes.append(n)
        return off

    def want(mut self, off: Int, n: Int) raises -> Int:
        return self.plan.want(off, n, -1, 1)

    def stage(mut self, op: String, total: Int, params: List[Int]) raises:
        self.plan.stage_op(op, total, params)


def msel_prog_run(route: MselSplitRoute, prog: MselProg) raises -> MetricArenaOwner:
    """Run a typed program: its input records (program-owned constants and
    the buffer inputs, in input order; an empty buffer has no record), then
    `metrics_plan_execute`. The records are the ones the shared door runner
    builds (bindings/arena_prog_native_door.mojo `_metrics_run_plan`)."""
    if len(prog.addresses) != prog.plan.buffer_inputs():
        raise Error("ValueError: x_metrics: the program's inputs do not match its layout")
    var records = List[MetricProgramInput](capacity=len(prog.plan.input_offsets))
    var next_buffer = 0
    for k in range(len(prog.plan.input_offsets)):  # driver-loop: over the program's input records
        if prog.plan.input_consts[k] >= 0:
            records.append(MetricProgramInput(prog.plan.input_offsets[k], prog.plan.const_address(k),
                                              len(prog.plan.consts[prog.plan.input_consts[k]])))
            continue
        var at = next_buffer
        next_buffer += 1
        if prog.sizes[at] == 0:
            continue
        records.append(MetricProgramInput(prog.plan.input_offsets[k], prog.addresses[at], prog.sizes[at]))
    return metrics_plan_execute(route.program, prog.plan, records, route.ranges, route.output_ranges)


def msel_draw_rows(route: MselSplitRoute, mut rng: MselCounterRng, sizes: List[Int],
                   mut arena: MetricArenaOwner) raises -> List[Int]:
    """`permutation_rows`: one draw per size, in order, in ONE program. The
    int64 rows of draw i are at `arena.base() + 4 * offsets[i]`; `arena` is
    replaced by the run's arena (untouched when there is no draw)."""
    var offsets = List[Int]()
    if len(sizes) == 0:
        return offsets^
    var prog = MselProg()
    for i in range(len(sizes)):  # driver-loop: over the requested draws (one stage each)
        var o = rng.permute_stage(prog.plan, sizes[i])
        var slot = prog.alloc(2 * sizes[i])
        var w = prog.want(slot, 2 * sizes[i])
        if sizes[i] > 0:
            prog.stage("rows64", sizes[i], [o, w])
        offsets.append(w)
    arena = msel_prog_run(route, prog)
    return offsets^


def msel_code_counts(route: MselSplitRoute, codes: MselCodes) raises -> List[Int64]:
    """Each class's row count (`_GroupCodes.counts`), by the call the door
    made: `fold_ids` over the codes with two folds, its class counts read."""
    var counts = List[Int64](length=max(codes.m, 1), fill=0)
    if codes.m > 0:
        var fold_store = List[Int32](length=max(codes.n, 1), fill=0)
        var fold_counts = List[Int64](length=2, fill=0)
        _ = msel_fold_ids(route.helpers, codes.address, codes.n, codes.m, 2, Int(counts.unsafe_ptr()),
                          Int(fold_store.unsafe_ptr()), Int(fold_counts.unsafe_ptr()))
        _ = fold_store  # the addresses above are not owners
        _ = fold_counts
    return counts^


struct MselSplitPlan(Movable):
    """Every split of one operation before its rows are written: the sizes
    of both sides and what `msel_split_fill` needs to write them."""

    var form: Int
    var n: Int
    var train_sizes: List[Int]
    var test_sizes: List[Int]
    #: where the door allocated each side (filled between the two RUN calls)
    var train_dst: List[Int]
    var test_dst: List[Int]
    #: COPY / INDEXED: source addresses; RANGES: first rows
    var train_src: List[Int]
    var test_src: List[Int]
    #: FOLDS: each row's fold and each fold's size; SELECT: each row's word
    var words: List[Int32]
    var words_address: Int
    var sizes64: List[Int64]
    var counted_by: String
    var sabotage: Bool
    #: SELECT: the word each split selects
    var wants: List[Int]
    #: the arena the source addresses point into
    var arena: MetricArenaOwner
    #: TABLES: split s's per-group table at tables[s * table_width]
    var tables: List[Int32]
    var table_width: Int
    var codes: MselCodes
    #: STRAT_SHUFFLE: class sizes, the rows class by class, the class offsets
    var counts: List[Int]
    var counts32: List[Int32]
    var order32: List[Int32]
    var offsets32: List[Int32]
    var rng: MselCounterRng
    #: a UserWarning the door issues after the plan (empty: none)
    var warned: Bool
    var warning: String

    def __init__(out self):
        self.form = 0
        self.n = 0
        self.train_sizes = List[Int]()
        self.test_sizes = List[Int]()
        self.train_dst = List[Int]()
        self.test_dst = List[Int]()
        self.train_src = List[Int]()
        self.test_src = List[Int]()
        self.words = List[Int32]()
        self.words_address = 0
        self.sizes64 = List[Int64]()
        self.counted_by = String("")
        self.sabotage = False
        self.wants = List[Int]()
        self.arena = MetricArenaOwner(0)
        self.tables = List[Int32]()
        self.table_width = 0
        self.codes = MselCodes(0, 0, 0)
        self.counts = List[Int]()
        self.counts32 = List[Int32]()
        self.order32 = List[Int32]()
        self.offsets32 = List[Int32]()
        self.rng = MselCounterRng(0, 0)
        self.warned = False
        self.warning = String("")

    def splits(self) -> Int:
        return len(self.test_sizes)

    def add(mut self, train: Int, test: Int):
        """One more split with these row counts."""
        self.train_sizes.append(train)
        self.test_sizes.append(test)

    def warn(mut self, message: String):
        self.warned = True
        self.warning = message


# ---------------------------------------------------------------- shared steps


def _plan_fold_sizes(mut plan: MselSplitPlan, k: Int):
    """FOLDS: split f is fold f (test) against every other row (train)."""
    for f in range(k):  # driver-loop: over folds
        var size = Int(plan.sizes64[f])
        plan.add(plan.n - size, size)


def _plan_fold_rows(mut plan: MselSplitPlan, rows: MselFoldRows):
    """COPY from a `fold_rows` table in `plan.arena`."""
    plan.form = MSEL_FILL_COPY
    var base = plan.arena.base()
    for f in range(rows.k):  # driver-loop: over folds
        var c = rows.size(base, f)
        plan.test_src.append(rows.test_rows(base, f))
        plan.train_src.append(rows.train_rows(base, f))
        plan.add(rows.n - c, c)


def _plan_by_fold(route: MselSplitRoute, mut plan: MselSplitPlan, groups: MselCodes, to_fold: List[Int32],
                  k: Int) raises:
    """`_GroupCodes.split_by_fold`: each row's fold is its group's fold (one
    gather), then FOLDS. `plan.sizes64` holds each fold's row count."""
    plan.form = MSEL_FILL_FOLDS
    plan.n = groups.n
    plan.counted_by = String("the group counts")
    plan.words = List[Int32](length=max(groups.n, 1), fill=0)
    _ = msel_gather_i32(route.helpers, Int(to_fold.unsafe_ptr()), groups.m, groups.address, groups.n,
                        Int(plan.words.unsafe_ptr()))
    _ = to_fold  # the address above is not an owner
    _plan_fold_sizes(plan, k)


struct MselEnc(Movable):
    """Class codes in first-seen order: the n int32 words (native, or the
    door's pinned buffer when they arrived in that order), the class count
    and the class sizes in that order (empty for a pinned buffer)."""

    var words: List[Int32]
    var external: Int
    var owned: Bool
    var n: Int
    var n_classes: Int
    var counts: List[Int]

    def __init__(out self):
        self.words = List[Int32]()
        self.external = 0
        self.owned = False
        self.n = 0
        self.n_classes = 0
        self.counts = List[Int]()

    def address(self) -> Int:
        return Int(self.words.unsafe_ptr()) if self.owned else self.external


def msel_first_seen_enc(route: MselSplitRoute, labels: MselCodes, form: Int) raises -> MselEnc:
    """`_first_seen_native` / `first_seen_i32`: the labels' codes renumbered
    in first-seen order.

    ALIGNED: the first row of each sorted-order class (x_metrics' first-row
    pass, a host pass), the classes ranked by it (every first row is
    distinct, so the sort has no ties), one gather. SORTED: the helper
    kernel `first_seen_i32`."""
    var enc = MselEnc()
    enc.n = labels.n
    if form == MSEL_LABELS_FIRST_SEEN:
        enc.external = labels.address
        enc.n_classes = labels.m
        return enc^
    enc.owned = True
    enc.words = List[Int32](length=max(labels.n, 1), fill=0)
    if form == MSEL_LABELS_ALIGNED:
        var m = labels.m
        var sorted_counts = msel_code_counts(route, labels)
        var first = List[Int64](length=max(m, 1), fill=0)
        first_rows_i32(labels.address, labels.n, m, Int(first.unsafe_ptr()))
        var rows_sorted = List[Int](capacity=m)
        var class_of = Dict[Int, Int]()
        for c in range(m):  # driver-loop: over classes (one first-row word each)
            var r = Int(first[c])
            rows_sorted.append(r)
            class_of[r] = c
        sort(rows_sorted)
        var table = List[Int32](length=max(m, 1), fill=0)
        for rank in range(m):  # driver-loop: over classes (first-seen order)
            var c = class_of[rows_sorted[rank]]
            table[c] = Int32(rank)
            enc.counts.append(Int(sorted_counts[c]))
        _ = msel_gather_i32(route.helpers, Int(table.unsafe_ptr()), m, labels.address, labels.n,
                            Int(enc.words.unsafe_ptr()))
        _ = table  # the address above is not an owner
        enc.n_classes = m
        return enc^
    var width = max(labels.m, 1)
    var cnt = List[Int64](length=width, fill=0)
    enc.n_classes = msel_first_seen_i32(route.helpers, labels.address, labels.n, width, Int(enc.words.unsafe_ptr()),
                                        Int(cnt.unsafe_ptr()))
    for c in range(enc.n_classes):  # driver-loop: over classes
        enc.counts.append(Int(cnt[c]))
    return enc^


# ---------------------------------------------------------------- the plans


def msel_plan_default_folds(route: MselSplitRoute, n: Int, k: Int, codes: MselCodes, coded: Bool,
                            sabotage: Bool) raises -> MselSplitPlan:
    """`_native_default_folds`: `fold_ids` (stratified by `codes` when
    `coded`), its class count refusals, then FOLDS."""
    var plan = MselSplitPlan()
    plan.form = MSEL_FILL_FOLDS
    plan.n = n
    plan.counted_by = String("fold_ids")
    plan.sabotage = sabotage
    var n_classes = codes.m if coded else 0
    plan.words = List[Int32](length=max(n, 1), fill=0)
    plan.sizes64 = List[Int64](length=max(k, 1), fill=0)
    var class_counts = List[Int64](length=max(n_classes, 1), fill=0)
    _ = msel_fold_ids(route.helpers, codes.address if coded else 0, n, n_classes, k,
                      Int(class_counts.unsafe_ptr()), Int(plan.words.unsafe_ptr()), Int(plan.sizes64.unsafe_ptr()))
    if coded:
        if n_classes == 0:
            raise Error("max() iterable argument is empty")
        var most = Int(class_counts[0])
        var least = Int(class_counts[0])
        for c in range(n_classes):  # driver-loop: over classes (the fold count checks)
            most = max(most, Int(class_counts[c]))
            least = min(least, Int(class_counts[c]))
        if most < k:
            raise Error("cv cannot exceed the number of members in every class")
        if least < k:
            plan.warn("The least populated class has fewer members than cv folds")
    _ = class_counts  # the address above is not an owner
    _plan_fold_sizes(plan, k)
    return plan^


def msel_plan_kfold(route: MselSplitRoute, n: Int, k: Int, shuffle: Bool, seed: MselSeed,
                    sabotage: Bool) raises -> MselSplitPlan:
    """`KFold.split`. Shuffled: the K-fold row table of one program
    (COPY), or past the table bound one permutation whose consecutive runs
    are the folds' test rows (INDEXED)."""
    if k > n:
        raise Error("Cannot have number of splits n_splits=" + String(k) + " greater than the "
                    + "number of samples: n_samples=" + String(n) + ".")
    if not shuffle:
        if k < 2:
            raise Error("cv must specify at least two folds")
        return msel_plan_default_folds(route, n, k, MselCodes(0, 0, 0), False, sabotage)
    var rng = seed.rng()
    var plan = MselSplitPlan()
    plan.n = n
    if msel_fold_rows_fits(n, k):
        var prog = MselProg()
        var rows = msel_shuffled_fold_rows_plan(prog.plan, n, k, rng)
        plan.arena = msel_prog_run(route, prog)
        _plan_fold_rows(plan, rows)
        return plan^
    var offsets = msel_draw_rows(route, rng, [n], plan.arena)
    plan.form = MSEL_FILL_INDEXED
    var whole = plan.arena.base() + 4 * offsets[0]
    var start = 0
    for fold in range(k):  # driver-loop: over folds
        var size = n // k + (1 if fold < n % k else 0)
        plan.test_src.append(whole + 8 * start)
        plan.add(n - size, size)
        start += size
    return plan^


def msel_plan_stratified(route: MselSplitRoute, labels: MselCodes, form: Int, k: Int, shuffle: Bool,
                         seed: MselSeed) raises -> MselSplitPlan:
    """`StratifiedKFold.split`: the fold plan (first-seen codes, class
    sizes, the allocation table), then the folds in one program (COPY), or
    past its bounds `strat_fold_assign_i32` and FOLDS."""
    var enc = msel_first_seen_enc(route, labels, form)
    var n = enc.n
    var n_classes = enc.n_classes
    var plan = MselSplitPlan()
    plan.n = n
    var most = 0
    var least = 0
    for c in range(len(enc.counts)):  # driver-loop: over classes (the fold count checks)
        most = enc.counts[c] if c == 0 else max(most, enc.counts[c])
        least = enc.counts[c] if c == 0 else min(least, enc.counts[c])
    if len(enc.counts) == 0 or most < k:
        raise Error("n_splits=" + String(k) + " cannot be greater than the number of members in each class.")
    if least < k:
        plan.warn("The least populated class in y has only " + String(least) + " members, which is less "
                  + "than n_splits=" + String(k) + ".")
    var cnt = List[Int64](length=max(n_classes, 1), fill=0)
    for c in range(len(enc.counts)):  # driver-loop: over classes
        cnt[c] = Int64(enc.counts[c])
    var alloc = List[Int64](length=max(k * n_classes, 1), fill=0)
    # The allocation table's host pass (both base bindings exported the same host function).
    _ = strat_alloc_i64(Int(cnt.unsafe_ptr()), n_classes, k, Int(alloc.unsafe_ptr()))
    var rng = MselCounterRng(0, 0)
    if shuffle:
        rng = seed.rng()
    if k <= 256 and msel_fold_rows_fits(n, k):
        var prog = MselProg()
        var enc_slot = prog.put_i32(enc.address(), n)
        var rows = msel_stratified_fold_rows_plan(prog.plan, enc_slot, n, enc.counts, Int(alloc.unsafe_ptr()), k,
                                                  rng, shuffle)
        plan.arena = msel_prog_run(route, prog)
        _ = enc  # the program's input and the table address are not owners
        _ = alloc
        _ = cnt
        _plan_fold_rows(plan, rows)
        return plan^
    # the same assignment by `strat_fold_assign_i32`, then each fold's rows
    if shuffle:
        rng = seed.rng()
    var perms = List[Int64](length=max(n, 1), fill=0)
    if shuffle:
        var drawn = MetricArenaOwner(0)
        var offsets = msel_draw_rows(route, rng, enc.counts, drawn)
        var at = 0
        for c in range(len(enc.counts)):  # driver-loop: over classes (one draw each)
            if enc.counts[c] > 0:
                unsafe_memcpy(dest=_I64P(unsafe_from_address=Int(perms.unsafe_ptr()) + 8 * at),
                              src=_I64P(unsafe_from_address=drawn.base() + 4 * offsets[c]), count=enc.counts[c])
            at += enc.counts[c]
        _ = drawn  # the copies above read its words
    plan.form = MSEL_FILL_FOLDS
    plan.counted_by = String("fold_ids")
    plan.words = List[Int32](length=max(n, 1), fill=0)
    _ = msel_strat_fold_assign_i32(route.helpers, enc.address(), n, n_classes, k, Int(alloc.unsafe_ptr()),
                                   Int(perms.unsafe_ptr()) if shuffle and n > 0 else 0, Int(cnt.unsafe_ptr()),
                                   Int(plan.words.unsafe_ptr()))
    _ = enc  # the addresses above are not owners
    _ = alloc
    _ = perms
    _ = cnt
    plan.sizes64 = List[Int64](length=max(k, 1), fill=0)
    # The sizes are the folds' own bincount (int32 words: code 2), so the
    # selection's count check always holds.
    _ = base_bincount_i64(route.base, Int(plan.words.unsafe_ptr()), 2, n, k, Int(plan.sizes64.unsafe_ptr()), False)
    _plan_fold_sizes(plan, k)
    return plan^


def msel_plan_group_kfold(route: MselSplitRoute, groups: MselCodes, k: Int, shuffle: Bool,
                          seed: MselSeed) raises -> MselSplitPlan:
    """`GroupKFold.split`."""
    var m = groups.m
    var counts = msel_code_counts(route, groups)
    if k > m:
        raise Error("Cannot have number of splits n_splits=" + String(k) + " greater than the "
                    + "number of groups: " + String(m) + ".")
    var plan = MselSplitPlan()
    var to_fold = List[Int32](length=max(m, 1), fill=0)
    plan.sizes64 = List[Int64](length=max(k, 1), fill=0)
    if shuffle:
        var rng = seed.rng()
        var drawn = MetricArenaOwner(0)
        var offsets = msel_draw_rows(route, rng, [m], drawn)
        _ = msel_group_fold_perm_i32(route.helpers, Int(counts.unsafe_ptr()), m, k, drawn.base() + 4 * offsets[0],
                                     Int(to_fold.unsafe_ptr()), Int(plan.sizes64.unsafe_ptr()))
        _ = drawn  # the permutation address above is not an owner
    else:
        # The greedy plan's host pass (both base bindings exported the same host function).
        _ = group_fold_assign_i32(Int(counts.unsafe_ptr()), m, k, 0, Int(to_fold.unsafe_ptr()),  # cpu-route: sequential greedy fold plan over groups, not rows
                                  Int(plan.sizes64.unsafe_ptr()))
    _ = counts  # the address above is not an owner
    _plan_by_fold(route, plan, groups, to_fold, k)
    return plan^


def msel_plan_strat_group(route: MselSplitRoute, n: Int, labels: MselCodes, form: Int, groups: MselCodes, k: Int,
                          shuffle: Bool, seed: MselSeed) raises -> MselSplitPlan:
    """`StratifiedGroupKFold.split`: the group x class table, the
    group-level plan (a host pass over groups), FOLDS."""
    var enc = msel_first_seen_enc(route, labels, form)
    if enc.n != n:
        raise Error("y has " + String(enc.n) + " labels for " + String(n) + " samples")
    var n_classes = enc.n_classes
    var m = groups.m
    var plan = MselSplitPlan()
    var drawn = MetricArenaOwner(0)
    var perm = 0
    if shuffle:
        var rng = seed.rng()
        var offsets = msel_draw_rows(route, rng, [m], drawn)
        perm = drawn.base() + 4 * offsets[0]
    var to_fold = List[Int32](length=max(m, 1), fill=0)
    plan.sizes64 = List[Int64](length=max(k, 1), fill=0)
    var dist = List[Int64](length=max(m * n_classes, 1), fill=0)
    _ = base_bincount2_i32(route.base, enc.address(), groups.address, n, n_classes, m, Int(dist.unsafe_ptr()))
    _ = enc  # the address above is not an owner
    var bad = strat_group_plan_i32(Int(dist.unsafe_ptr()), perm, Int(to_fold.unsafe_ptr()),  # cpu-route: sequential greedy fold plan over groups, not rows
                                   Int(plan.sizes64.unsafe_ptr()), n_classes, m, k)
    _ = dist  # the addresses above are not owners
    _ = drawn
    if bad != 0:
        raise Error("n_splits=" + String(k) + " cannot be greater than the number of members in "
                    + "each class.")
    _plan_by_fold(route, plan, groups, to_fold, k)
    return plan^


def msel_plan_time_series(n: Int, k: Int, test_size: Int, has_test_size: Bool, gap: Int,
                          max_train: Int) raises -> MselSplitPlan:
    """`TimeSeriesSplit.split`: both sides of every split are row ranges.
    `max_train` 0 is no limit."""
    var folds = k + 1
    var ts = test_size if has_test_size else n // folds
    if folds > n:
        raise Error("Cannot have number of folds=" + String(folds) + " greater than the number of samples="
                    + String(n) + ".")
    if n - gap - ts * k <= 0:
        raise Error("Too many splits=" + String(k) + " for number of samples=" + String(n) + " with "
                    + "test_size=" + String(ts) + " and gap=" + String(gap) + ".")
    if ts == 0:
        raise Error("range() arg 3 must not be zero")
    var plan = MselSplitPlan()
    plan.form = MSEL_FILL_RANGES
    plan.n = n
    var start = n - k * ts
    while start < n:  # driver-loop: over splits
        var stop = start - gap
        var lo = 0
        if max_train != 0 and max_train < stop:
            lo = stop - max_train
        plan.train_src.append(lo)
        plan.test_src.append(start)
        plan.add(max(stop - lo, 0), max(ts, 0))
        start += ts
    return plan^


def msel_plan_shuffle(route: MselSplitRoute, n: Int, n_splits: Int, n_train: Int, n_test: Int,
                      seed: MselSeed) raises -> MselSplitPlan:
    """`ShuffleSplit.split`: one permutation per split, all in one program;
    its first n_test rows are the test side, the next n_train the train
    side (COPY)."""
    var rng = seed.rng()
    var plan = MselSplitPlan()
    plan.form = MSEL_FILL_COPY
    plan.n = n
    var want = List[Int](length=max(n_splits, 0), fill=n)
    var offsets = msel_draw_rows(route, rng, want, plan.arena)
    for s in range(len(want)):  # driver-loop: over splits
        var perm = plan.arena.base() + 4 * offsets[s]
        plan.train_src.append(perm + 8 * n_test)
        plan.test_src.append(perm)
        plan.add(n_train, n_test)
    return plan^


def msel_plan_group_shuffle(route: MselSplitRoute, groups: MselCodes, n_splits: Int, n_train: Int, n_test: Int,
                            seed: MselSeed) raises -> MselSplitPlan:
    """`GroupShuffleSplit.split`: one permutation of the groups per split
    (one program), each split's per-group table (0 train, 1 test, 2
    neither) and its two row counts (TABLES)."""
    var m = groups.m
    var counts = msel_code_counts(route, groups)
    var rng = seed.rng()
    var plan = MselSplitPlan()
    plan.form = MSEL_FILL_TABLES
    plan.n = groups.n
    plan.codes = groups
    plan.table_width = m
    var want = List[Int](length=max(n_splits, 0), fill=m)
    var drawn = MetricArenaOwner(0)
    var offsets = msel_draw_rows(route, rng, want, drawn)
    plan.tables = List[Int32](length=max(len(want) * m, 1), fill=0)
    var sums = List[Int64](length=2, fill=0)
    for s in range(len(want)):  # driver-loop: over splits (a pass over groups each)
        _ = msel_split_table_i32(route.helpers, drawn.base() + 4 * offsets[s], m, n_test, n_train,
                                 Int(counts.unsafe_ptr()), Int(plan.tables.unsafe_ptr()) + 4 * s * m,
                                 Int(sums.unsafe_ptr()))
        plan.add(Int(sums[0]), Int(sums[1]))
    _ = drawn  # the addresses above are not owners
    _ = counts
    return plan^


def msel_plan_strat_shuffle(route: MselSplitRoute, labels: MselCodes, n_splits: Int, n_train: Int, n_test: Int,
                            seed: MselSeed) raises -> MselSplitPlan:
    """`StratifiedShuffleSplit._native_split`: the class refusals and the
    rows class by class (a stable `group_sort` of the class codes, one
    program, read back once: N086). Each split is then one program
    (`msel_split_fill`). approx_mode hands out exactly n_train (then n_test)
    draws, so both lengths are known before any split's program runs."""
    var n = labels.n
    var k = labels.m
    var counts = msel_code_counts(route, labels)
    if k == 0:
        raise Error("min() iterable argument is empty")
    var least = Int(counts[0])
    for c in range(k):  # driver-loop: over classes
        least = min(least, Int(counts[c]))
    if least < 2:
        raise Error("The least populated classes in y have only 1 member, which is too few. The "
                    + "minimum number of groups for any class cannot be less than 2.")
    if n_train < k:
        raise Error("The train_size = " + String(n_train) + " should be greater or equal to the number of classes = " + String(k))
    if n_test < k:
        raise Error("The test_size = " + String(n_test) + " should be greater or equal to the number of classes = " + String(k))
    var plan = MselSplitPlan()
    plan.form = MSEL_FILL_STRAT_SHUFFLE
    plan.n = n
    var prog = MselProg()
    var key = prog.put_i32(labels.address, n)
    var off_slot = prog.alloc(k + 1)
    var off = prog.want(off_slot, k + 1)
    var ord_slot = prog.alloc(n)
    var ordr = prog.want(ord_slot, n)
    prog.stage("group_sort", 1, [key, n, k, off_slot, ord_slot])
    var arena = msel_prog_run(route, prog)
    plan.order32 = arena.i32_words(ordr, n)
    plan.offsets32 = arena.i32_words(off, k + 1)
    for c in range(k):  # driver-loop: over classes
        plan.counts.append(Int(counts[c]))
        plan.counts32.append(Int32(Int(counts[c])))
    plan.rng = seed.rng()
    for _ in range(max(n_splits, 0)):  # driver-loop: over splits
        plan.add(n_train, n_test)
    return plan^


def msel_plan_select(route: MselSplitRoute, codes: MselCodes, wants: List[Int], counted: Bool,
                     sizes: List[Int]) raises -> MselSplitPlan:
    """One split per wanted code: its rows (test) against every other row
    (train). `counted`: the test sizes are the codes' own counts
    (PredefinedSplit); else `sizes` gives them (a lazy group split, whose
    state carries the counts)."""
    var plan = MselSplitPlan()
    plan.form = MSEL_FILL_SELECT
    plan.n = codes.n
    plan.words_address = codes.address
    if counted:
        var counts = msel_code_counts(route, codes)
        for i in range(len(wants)):  # driver-loop: over the selected codes
            plan.wants.append(wants[i])
            plan.add(codes.n - Int(counts[wants[i]]), Int(counts[wants[i]]))
        return plan^
    for i in range(len(wants)):  # driver-loop: over the selected codes
        plan.wants.append(wants[i])
        plan.add(codes.n - sizes[i], sizes[i])
    return plan^


def msel_plan_group_set(route: MselSplitRoute, codes: MselCodes, chosen: List[Int],
                        size: Int) raises -> MselSplitPlan:
    """One LeavePGroupsOut split: the rows of the chosen groups (`size` of
    them) as the test side. A per-group 0/1 table, one gather, SELECT of
    word 1."""
    var plan = MselSplitPlan()
    plan.form = MSEL_FILL_SELECT
    plan.n = codes.n
    var table = List[Int32](length=max(codes.m, 1), fill=0)
    for i in range(len(chosen)):  # driver-loop: over the chosen groups of one combination
        table[chosen[i]] = 1
    plan.words = List[Int32](length=max(codes.n, 1), fill=0)
    _ = msel_gather_i32(route.helpers, Int(table.unsafe_ptr()), codes.m, codes.address, codes.n,
                        Int(plan.words.unsafe_ptr()))
    _ = table  # the address above is not an owner
    plan.wants.append(1)
    plan.add(codes.n - size, size)
    return plan^


def msel_plan_indexed(n: Int, rows: Int, count: Int) -> MselSplitPlan:
    """`_split_indices`: one split whose test rows are the `count` distinct
    int64 rows at `rows` (the door pins them)."""
    var plan = MselSplitPlan()
    plan.form = MSEL_FILL_INDEXED
    plan.n = n
    plan.test_src.append(rows)
    plan.add(n - count, count)
    return plan^


def msel_leave_one(route: MselSplitRoute, n: Int, row: Int, train: Int, test: Int) raises:
    """Split `row` of LeaveOneOut over n rows, written at the two addresses."""
    _ = msel_leave_range_i64(route.helpers, n, row, row + 1, train, test)


# ---------------------------------------------------------------- the rows


def _fill_folds(route: MselSplitRoute, plan: MselSplitPlan) raises:
    """Every fold's (train, test) in ONE `select_folds_i64` call (N051). An
    empty side shares one private scratch run, as before."""
    var k = plan.splits()
    var n = plan.n
    var tests = List[Int64](length=max(k, 1), fill=0)
    var trains = List[Int64](length=max(k, 1), fill=0)
    var scratch = List[Int64]()
    for fold in range(k):  # driver-loop: over folds (addresses only; the selection is one call)
        var size = plan.test_sizes[fold]
        if (size == 0 or n - size == 0) and len(scratch) == 0:
            scratch = List[Int64](length=max(n, 1), fill=0)
        tests[fold] = Int64(plan.test_dst[fold] if size else Int(scratch.unsafe_ptr()))
        trains[fold] = Int64(plan.train_dst[fold] if n - size else Int(scratch.unsafe_ptr()))
    var bad = msel_select_folds_i64(route.helpers, Int(plan.words.unsafe_ptr()), n, k, Int(plan.sizes64.unsafe_ptr()),
                                    Int(tests.unsafe_ptr()), Int(trains.unsafe_ptr()))
    _ = scratch  # the address words above are not owners
    _ = tests
    _ = trains
    if bad >= 0:
        raise Error("RuntimeError: mojolearn: select_fold_i64 disagrees with " + plan.counted_by)


def _fill_strat_shuffle(route: MselSplitRoute, mut plan: MselSplitPlan) raises:
    """N086 (lane ne2-b-search-workers): ONE program per split. Its two
    approx_mode stages, the class draws and the two final draws as
    `permute_drawn` stages (their draw numbers follow the approx_mode draws
    the same program consumed), the per-class selection (`strat_take`) and
    the final orders (`take_rows64`); the train and test rows and the
    consumed draw count are read back. The same draws in the same order."""
    var n = plan.n
    var k = len(plan.counts)
    var offsets = List[Int]()
    var at = 0
    for c in range(k):  # driver-loop: over classes (offsets of the class-ordered rows)
        offsets.append(at)
        at += plan.counts[c]
    for s in range(plan.splits()):  # driver-loop: over splits (one program each)
        var n_train = plan.train_sizes[s]
        var n_test = plan.test_sizes[s]
        var prog = MselProg()
        var cnt = prog.put_i32(Int(plan.counts32.unsafe_ptr()), k)
        var ordr = prog.put_i32(Int(plan.order32.unsafe_ptr()), n)
        var off = prog.put_i32(Int(plan.offsets32.unsafe_ptr()), k + 1)
        var modes = prog.alloc(2 * k)
        var used_slot = prog.alloc(2)
        var used = prog.want(used_slot, 2)
        var w0 = msel_split_signed32(plan.rng.seed)
        var w1 = msel_split_signed32(plan.rng.seed >> 32)
        var w2 = msel_split_signed32(plan.rng.draws)
        var w3 = msel_split_signed32(plan.rng.draws >> 32)
        prog.stage("approx_mode", k, [k, cnt, MSEL_SPLIT_NONE, n_train, modes, MSEL_SPLIT_NONE, used, w0, w1, w2, w3])
        prog.stage("approx_mode", k, [k, cnt, modes, n_test, modes + k, used, used + 1, w0, w1, w2, w3])
        var draws = prog.alloc(n)
        for c in range(k):  # driver-loop: over classes (one draw stage each, all in this program)
            var d = plan.rng.draws + UInt64(c)
            prog.stage("permute_drawn", 1, [plan.counts[c], draws + offsets[c], w0, w1, msel_split_signed32(d),
                                            msel_split_signed32(d >> 32), used, 2])
        var train = prog.alloc(n_train)
        var test = prog.alloc(n_test)
        prog.stage("strat_take", n, [k, ordr, off, draws, modes, train, test])
        var last_tr = prog.alloc(n_train)
        var last_te = prog.alloc(n_test)
        var d_tr = plan.rng.draws + UInt64(k)
        var d_te = plan.rng.draws + UInt64(k + 1)
        prog.stage("permute_drawn", 1, [n_train, last_tr, w0, w1, msel_split_signed32(d_tr),
                                        msel_split_signed32(d_tr >> 32), used, 2])
        prog.stage("permute_drawn", 1, [n_test, last_te, w0, w1, msel_split_signed32(d_te),
                                        msel_split_signed32(d_te >> 32), used, 2])
        var tr_slot = prog.alloc(2 * n_train)
        var tr_words = prog.want(tr_slot, 2 * n_train)
        var te_slot = prog.alloc(2 * n_test)
        var te_words = prog.want(te_slot, 2 * n_test)
        prog.stage("take_rows64", n_train, [train, last_tr, tr_words])
        prog.stage("take_rows64", n_test, [test, last_te, te_words])
        var arena = msel_prog_run(route, prog)
        var base = arena.base()
        var words = _I32P(unsafe_from_address=base)
        plan.rng.draws += UInt64(Int(words[used]) + Int(words[used + 1]) + k + 2)
        if n_train > 0:
            unsafe_memcpy(dest=_I64P(unsafe_from_address=plan.train_dst[s]),
                          src=_I64P(unsafe_from_address=base + 4 * tr_words), count=n_train)
        if n_test > 0:
            unsafe_memcpy(dest=_I64P(unsafe_from_address=plan.test_dst[s]),
                          src=_I64P(unsafe_from_address=base + 4 * te_words), count=n_test)
        # The copied addresses point into this arena: keep it through the copies.
        _ = arena^
        _ = prog^


def msel_split_fill(route: MselSplitRoute, mut plan: MselSplitPlan) raises:
    """Write every split's rows at `plan.train_dst` / `plan.test_dst` (one
    address per split and side, each sized by the plan)."""
    var n = plan.n
    var splits = plan.splits()
    if len(plan.train_dst) != splits or len(plan.test_dst) != splits:
        raise Error("RuntimeError: mojolearn: a split plan was not given its destinations")
    if plan.form == MSEL_FILL_FOLDS:
        _fill_folds(route, plan)
        if plan.sabotage:
            # THE DORMANT NEGATIVE CONTROL (a verification switch, never set in
            # production): each nonempty fold's smallest test row moves to the
            # previous nonempty fold (the last fold takes fold 0's). The fold
            # sizes do not change, so the rows are selected again in place.
            var nonempty = List[Int]()
            var firsts = List[Int]()
            for f in range(splits):  # driver-loop: over folds
                if plan.test_sizes[f] > 0:
                    nonempty.append(f)
                    firsts.append(Int(_I64P(unsafe_from_address=plan.test_dst[f])[0]))
            for at in range(len(nonempty)):  # driver-loop: over nonempty folds (one row id moved each)
                plan.words[firsts[(at + 1) % len(nonempty)]] = Int32(nonempty[at])
            _fill_folds(route, plan)
    elif plan.form == MSEL_FILL_COPY:
        for s in range(splits):  # driver-loop: over splits (two block copies each)
            if plan.test_sizes[s] > 0:
                unsafe_memcpy(dest=_I64P(unsafe_from_address=plan.test_dst[s]),
                              src=_I64P(unsafe_from_address=plan.test_src[s]), count=plan.test_sizes[s])
            if plan.train_sizes[s] > 0:
                unsafe_memcpy(dest=_I64P(unsafe_from_address=plan.train_dst[s]),
                              src=_I64P(unsafe_from_address=plan.train_src[s]), count=plan.train_sizes[s])
    elif plan.form == MSEL_FILL_INDEXED:
        # `_split_indices` / `_split_mask`: the mask kernel zeroes its n bytes
        # first, so one mask serves every split.
        var mask = List[UInt8](length=max(n, 1), fill=0)
        for s in range(splits):  # driver-loop: over splits (three helper calls each)
            var size = plan.test_sizes[s]
            _ = msel_mask_from_indices_u8(route.helpers, plan.test_src[s] if size else 0, size, n,
                                          Int(mask.unsafe_ptr()))
            var n_test = msel_count_mask_u8(route.helpers, Int(mask.unsafe_ptr()), n)
            if n_test != size:
                raise Error("RuntimeError: mojolearn: a split's test rows are not distinct")
            _ = msel_select_mask_u8_i64(route.helpers, Int(mask.unsafe_ptr()), n, plan.test_dst[s] if n_test else 0,
                                        plan.train_dst[s] if n - n_test else 0)
        _ = mask  # the address above is not an owner
    elif plan.form == MSEL_FILL_RANGES:
        for s in range(splits):  # driver-loop: over splits (two ranges each)
            if plan.train_sizes[s] > 0:
                _ = msel_arange_i64(route.helpers, plan.train_dst[s], plan.train_src[s], plan.train_sizes[s])
            if plan.test_sizes[s] > 0:
                _ = msel_arange_i64(route.helpers, plan.test_dst[s], plan.test_src[s], plan.test_sizes[s])
    elif plan.form == MSEL_FILL_TABLES:
        var m = plan.table_width
        var words = List[Int32](length=max(n, 1), fill=0)
        var scratch = List[Int64](length=max(n, 1), fill=0)
        for s in range(splits):  # driver-loop: over splits (one gather and two selections each)
            _ = msel_gather_i32(route.helpers, Int(plan.tables.unsafe_ptr()) + 4 * s * m, m, plan.codes.address, n,
                                Int(words.unsafe_ptr()))
            var got = msel_select_fold_i64(route.helpers, Int(words.unsafe_ptr()), n, 0, plan.train_dst[s],
                                           Int(scratch.unsafe_ptr()))
            if got != plan.train_sizes[s]:
                raise Error("RuntimeError: mojolearn: select_fold_i64 disagrees with the group counts")
            got = msel_select_fold_i64(route.helpers, Int(words.unsafe_ptr()), n, 1, plan.test_dst[s],
                                       Int(scratch.unsafe_ptr()))
            if got != plan.test_sizes[s]:
                raise Error("RuntimeError: mojolearn: select_fold_i64 disagrees with the group counts")
        _ = words  # the addresses above are not owners
        _ = scratch
    elif plan.form == MSEL_FILL_SELECT:
        var address = plan.words_address if plan.words_address != 0 else Int(plan.words.unsafe_ptr())
        for s in range(splits):  # driver-loop: over splits (one selection each)
            var got = msel_select_fold_i64(route.helpers, address, n, plan.wants[s], plan.test_dst[s],
                                           plan.train_dst[s])
            if got != plan.test_sizes[s]:
                raise Error("RuntimeError: mojolearn: select_fold_i64 disagrees with the group counts")
    elif plan.form == MSEL_FILL_STRAT_SHUFFLE:
        _fill_strat_shuffle(route, plan)
    elif splits != 0:
        raise Error("RuntimeError: mojolearn: a split plan has no fill form")

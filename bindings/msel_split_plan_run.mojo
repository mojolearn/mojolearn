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

THE STRATIFIED SPLITTERS' LABEL CODES (lane vdf-scope-split, 2026-11-24).
StratifiedKFold, RepeatedStratifiedKFold (each repeat is a StratifiedKFold
plan), StratifiedGroupKFold and the stratified default folds put the n int32
codes once per split call into the helper table's store (`MselHeld`; on the
device column a block of the store's context) and run the layout V3 helpers
on that block (core/msel_entry_table.mojo): the class sizes and the k-word
rank table come back, the first-seen codes and the fold ids never do, and
each fold's rows come down once, into the arrays the caller receives. The
fold program (k <= 256) takes the caller's codes as its one n-word input and
renumbers them itself (`gather_i32`, rank_of[code]). StratifiedShuffleSplit
reads its class sizes from its own `group_sort` offsets: its codes go up only
as that program's input. Before, each helper call uploaded (and most
downloaded) one n-word vector: first_seen_i32 (codes up, first-seen codes
down), the program (first-seen codes up), strat_fold_assign_i32 (up, down),
the fold sizes' bincount (up) and select_folds_i64 (up). Same integers.

On a program image with a resident store (`MselSplitRoute.store`; lane
vdf-scope-split2, 2026-11-24) StratifiedKFold's program route puts the codes
ONCE, into that store (`MselProgSlot`), and reads the slot twice: a small
first-seen program (`msel_first_seen_prog`: sizes and first rows, 2k + 1
words down) and the fold program (the slot is its input, copied device to
device). Before, they went up twice: into the helper table's store for
MH_FIRST_SEEN_RES, then as the fold program's host input. StratifiedGroupKFold's
group x class table is counted on the stored codes (MH_GROUP_CLASS_COUNTS_RES,
layout V4), its classes ranked where they are counted.

No Python object is read, built or called here.
"""
from std.math import ceil, floor, isinf, isnan
from std.memory import unsafe_memcpy
from std.builtin.sort import sort
from core.entry_table import EntryRef, entry_ref_unresolved
from core.native_descriptor_abi import NativeDescriptorCall
from core.msel_helpers_calls import (
    msel_arange_i64, msel_count_mask_u8, msel_select_mask_u8_i64, msel_mask_from_indices_u8, msel_leave_range_i64,
    msel_fold_ids, msel_select_fold_i64, msel_select_folds_i64, msel_gather_i32, msel_split_table_i32,
    msel_group_fold_perm_i32, msel_store_put, msel_store_free, msel_first_seen_res, msel_strat_fold_assign_res,
    msel_select_folds_res, msel_fold_ids_res, msel_group_class_counts_res,
)
# The host-only passes linked into the splitters' unit (lane calltable-msel).
from core.hotpath_impl_rows import group_fold_assign_i32, strat_alloc_i64, strat_group_plan_i32
from core.metrics_program_plan import MetricProgramPlan
from bindings.arena_prog_run import MetricArenaOwner, MetricRoute, metric_inputs_run, metric_slots_run
from core.native_descriptor_call import native_descriptor_call
from x_metrics.descriptor_schema import MP_MODE, MP_VENDOR
from x_prep.transport_abi import XP_DEV_PUT, XP_DEV_FREE
from core.prep_entry_table import PrepEntryKey, xp_transport_ref, xp_transport_call
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
comptime MSEL_FILL_LEAVE_ONE = 8  # split s is row s (test) against every other row (`leave_range_i64`)

#: What a `_validate_shuffle_split` size word is (`MselSizeWord.kind`).
comptime MSEL_SIZE_NONE = 0
comptime MSEL_SIZE_INT = 1
comptime MSEL_SIZE_REAL = 2
comptime MSEL_SIZE_BAD = 3  # a value the door refuses as no native numeric size

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


struct MselSplitRoute(ImplicitlyCopyable, Movable):
    """The compiled entries one split operation calls, resolved once by
    UNPACK: the model-selection helper table and the x_metrics program
    runner with its two range switches. An entry the splitter never calls
    is `entry_ref_unresolved()`.

    `store` (lane vdf-scope-split2, 2026-11-24): the program image has a
    resident input store on its range route (bindings/arena_prog_run.mojo
    `MetricRoute.store`, read once by UNPACK with the range switch), so a
    program input may be a slot of that store (`MselProgSlot`). False for
    the host column, a CPU-only install, `MOJOLEARN_ARENA_RANGES=0` and the
    four-word form.

    REMOVED (lane vdf-scope-split2, 2026-11-24): the base helper table entry
    `base`. Its last splitter calls were StratifiedKFold's past-bounds
    assignment and fold bincount (moved to the helper table's stored-code
    forms by lane vdf-scope-split) and StratifiedGroupKFold's
    `bincount2_i32` (now MH_GROUP_CLASS_COUNTS_RES); UNPACK no longer
    resolves its key."""

    var helpers: EntryRef
    var program: NativeDescriptorCall
    var ranges: Bool
    var output_ranges: Bool
    var store: Bool

    def __init__(out self, helpers: EntryRef, program: NativeDescriptorCall, ranges: Bool, output_ranges: Bool):
        self.helpers = helpers
        self.program = program
        self.ranges = ranges
        self.output_ranges = output_ranges
        self.store = False

    def __init__(out self, helpers: EntryRef, program: NativeDescriptorCall, ranges: Bool, output_ranges: Bool,
                 store: Bool):
        self.helpers = helpers
        self.program = program
        self.ranges = ranges
        self.output_ranges = output_ranges
        self.store = store and ranges


def msel_split_route_none() -> MselSplitRoute:
    var none = entry_ref_unresolved()
    return MselSplitRoute(none, none.call, False, False)


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
    """Run a typed program through the one typed program runner
    (bindings/arena_prog_run.mojo `metric_inputs_run`; lane bm-meta-3,
    2026-11-23: its record loop was a copy of the door runner's)."""
    return metric_inputs_run(MetricRoute(route.program, route.ranges, route.output_ranges), prog.plan,
                             prog.addresses, prog.sizes)


def msel_prog_run_slots(route: MselSplitRoute, prog: MselProg, slots: List[Int]) raises -> MetricArenaOwner:
    """`msel_prog_run` where buffer input i is the program store's slot
    `slots[i]` when it is >= 0 (bindings/arena_prog_run.mojo
    `metric_slots_run`: copied device to device into the arena, its host
    address not read). `slots` empty: every input is a host buffer."""
    return metric_slots_run(MetricRoute(route.program, route.ranges, route.output_ranges, route.store), prog.plan,
                            prog.addresses, prog.sizes, slots)


struct MselProgSlot(Movable):
    """A split call's n int32 label codes put once into the program image's
    resident store (lane vdf-scope-split2, 2026-11-24): the slot (-1:
    nothing put) and the store entry that frees it, on every path
    (`__del__`). Only on a route with `store`; every program of the call
    reads the slot. The store is the x_prep image's (lane vdf-xm-wire,
    x_metrics/descriptor_schema.mojo: the metrics entry's own MP_DEV_PUT /
    MP_DEV_FREE were deleted), reached through its transport entry
    (`XP_DEV_PUT` / `XP_DEV_FREE`) of the program image's tier and vendor,
    resolved once per put (two entry calls and the table read,
    bindings/x_metrics_resident_door.mojo `metric_store_ref`)."""

    var slot: Int
    var store: EntryRef

    def __init__(out self, route: MselSplitRoute, labels: MselCodes) raises:
        self.store = entry_ref_unresolved()
        self.slot = -1
        if labels.n > 0:
            var mode = native_descriptor_call(route.program, MP_MODE, [], [], [])
            var vendor = native_descriptor_call(route.program, MP_VENDOR, [], [], [])
            self.store = xp_transport_ref(PrepEntryKey(mode, vendor))
            var src: List[Int] = [labels.address]
            var count: List[Int] = [labels.n]
            self.slot = xp_transport_call(self.store, XP_DEV_PUT, src, count)

    def __init__(out self):
        self.store = entry_ref_unresolved()
        self.slot = -1

    def __del__(deinit self):
        if self.slot >= 0:
            try:
                var ids: List[Int] = [self.slot]
                _ = xp_transport_call(self.store, XP_DEV_FREE, List[Int](), ids)
            except error:
                pass


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
    #: LeavePOut: every combination's int64 test rows, p words each
    var rows64: List[Int64]
    #: FOLDS out of a stored fold block (layout V3, lane vdf-scope-split): its
    #: id (-1: none; the fold words are `words`) and the helper entry that
    #: frees it. `msel_split_fill` frees it after the rows are written; a plan
    #: dropped before (made only for its refusals, or an error between the two
    #: RUN calls) frees it in `__del__`.
    var folds_id: Int
    var folds_entry: EntryRef

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
        self.rows64 = List[Int64]()
        self.folds_id = -1
        self.folds_entry = entry_ref_unresolved()

    def __del__(deinit self):
        if self.folds_id >= 0:
            try:
                _ = msel_store_free(self.folds_entry, self.folds_id)
            except error:
                pass

    def release_folds(mut self) raises:
        """Free the stored fold block now (after the fill read it)."""
        if self.folds_id >= 0:
            var id = self.folds_id
            self.folds_id = -1
            _ = msel_store_free(self.folds_entry, id)

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


struct MselHeld(Movable):
    """A split call's label codes put once into the helper table's store
    (MH_STORE_PUT; lane vdf-scope-split): the block id (-1: nothing put) and
    the entry that frees it. Freed at the end of the plan, on every path
    (`__del__`): the plan keeps only the fold block it made."""

    var id: Int
    var entry: EntryRef

    def __init__(out self, route: MselSplitRoute, labels: MselCodes) raises:
        self.entry = route.helpers
        self.id = -1
        if labels.n > 0:
            self.id = msel_store_put(route.helpers, labels.address, 4 * labels.n)

    def __init__(out self):
        self.entry = entry_ref_unresolved()
        self.id = -1

    def __del__(deinit self):
        if self.id >= 0:
            try:
                _ = msel_store_free(self.entry, self.id)
            except error:
                pass


struct MselEnc(Movable):
    """Class codes in first-seen order, as the stored-code helpers give them
    (lane vdf-scope-split): the class count, the class sizes in that order
    (empty when the caller's codes already were in that order: nothing was
    counted) and the rank table rank_of[code] over the codes' range (`width`
    words; empty: the codes are already first-seen). The n first-seen codes
    are never formed on the host: the fold program and the stored-code
    assignment renumber the codes where they are."""

    var n: Int
    var n_classes: Int
    var counts: List[Int]
    var rank: List[Int32]
    var width: Int

    def __init__(out self):
        self.n = 0
        self.n_classes = 0
        self.counts = List[Int]()
        self.rank = List[Int32]()
        self.width = 0

    def rank_address(self) -> Int:
        return Int(self.rank.unsafe_ptr()) if len(self.rank) > 0 else 0


def msel_first_seen_enc(route: MselSplitRoute, labels: MselCodes, form: Int, held: MselHeld) raises -> MselEnc:
    """`_first_seen_native` / `first_seen_i32`: the labels' class sizes in
    first-seen order and the rank table that renumbers them, from the codes
    `held` put (MH_FIRST_SEEN_RES: 2k words come back, nothing n-long).

    ALIGNED (sorted-order codes of a unique pass: every class has a row) and
    SORTED are the same first-seen ranking (the first row of each class,
    ranked); ALIGNED keeps its m classes. FIRST_SEEN: the codes are used as
    they are (no count, no table)."""
    var enc = MselEnc()
    enc.n = labels.n
    if form == MSEL_LABELS_FIRST_SEEN:
        enc.n_classes = labels.m
        return enc^
    if labels.n == 0 or held.id < 0:
        return enc^
    var width = max(labels.m, 1)
    var cnt = List[Int64](length=width, fill=0)
    enc.rank = List[Int32](length=width, fill=0)
    enc.width = width
    var present = msel_first_seen_res(route.helpers, held.id, labels.n, width, Int(cnt.unsafe_ptr()),
                                      Int(enc.rank.unsafe_ptr()))
    enc.n_classes = labels.m if form == MSEL_LABELS_ALIGNED else present
    for c in range(enc.n_classes):  # driver-loop: over classes
        enc.counts.append(Int(cnt[c]))
    return enc^


def msel_first_seen_prog(route: MselSplitRoute, labels: MselCodes, form: Int, slot: Int) raises -> MselEnc:
    """`msel_first_seen_enc` over the codes in the program store's `slot`
    (lane vdf-scope-split2, 2026-11-24): the same class sizes in first-seen
    order and the same rank table, from ONE small x_metrics program that
    reads the slot where it is: a stable `group_sort` of the codes (each
    code's row count is its OFF step, its first row the first word of its
    run) and one `gather_i32` of ORD at OFF (each code's first row), so
    the 2k + 1 words come down and nothing n-long does. The codes are
    ranked by their first rows on the host (a driver pass over classes;
    rows are distinct, so `row * width + code` sorts by row alone), absent
    codes after the present ones in code order, as MH_FIRST_SEEN_RES does.
    The fold program then reads the same slot, so the codes cross the bus
    once. A code outside [0, k) is dropped by the sort and refused here
    with MH_FIRST_SEEN_RES's words."""
    var enc = MselEnc()
    enc.n = labels.n
    if labels.n == 0 or slot < 0:
        return enc^
    var n = labels.n
    var width = max(labels.m, 1)
    var prog = MselProg()
    var code_slot = prog.put_i32(labels.address, n)
    var off = prog.alloc(width + 1)
    # One word past the rows: a trailing absent code's OFF is n, and its
    # gathered word is read but never used.
    var ordr = prog.plan.scratch(n + 1)
    prog.stage("group_sort", 1, [code_slot, n, width, off, ordr])
    var first = prog.alloc(width)
    prog.stage("gather_i32", width, [ordr, off, first])
    _ = prog.want(off, width + 1)
    _ = prog.want(first, width)
    var arena = msel_prog_run_slots(route, prog, [slot])
    if arena.i32(off + width) != n:
        raise Error("first_seen_i32: code out of range")
    var keys = List[Int](capacity=width)
    for c in range(width):  # driver-loop: over classes (the present codes by first row)
        if arena.i32(off + c + 1) > arena.i32(off + c):
            keys.append(arena.i32(first + c) * width + c)
    sort(keys)
    var present = len(keys)
    enc.rank = List[Int32](length=width, fill=-1)
    enc.width = width
    for r in range(present):  # driver-loop: over classes (first-seen order)
        var c = keys[r] % width
        enc.rank[c] = Int32(r)
    var after = present
    for c in range(width):  # driver-loop: over classes (the absent codes' positions)
        if enc.rank[c] < 0:
            enc.rank[c] = Int32(after)
            after += 1
    enc.n_classes = labels.m if form == MSEL_LABELS_ALIGNED else present
    for r in range(enc.n_classes):  # driver-loop: over classes
        var c = keys[r] % width if r < present else -1
        enc.counts.append(arena.i32(off + c + 1) - arena.i32(off + c) if c >= 0 else 0)
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
    plan.sizes64 = List[Int64](length=max(k, 1), fill=0)
    var class_counts = List[Int64](length=max(n_classes, 1), fill=0)
    if coded and not sabotage and n >= 1 and n_classes >= 1 and k >= 2 and codes.n == n:
        # Stratified (lane vdf-scope-split): the codes are put once and the
        # fold ids stay in the store until `msel_split_fill` selects the rows.
        # The negative control (`sabotage`) moves fold words on the host, so
        # it keeps the host words.
        var held = MselHeld(route, codes)
        plan.folds_entry = route.helpers
        plan.folds_id = msel_fold_ids_res(route.helpers, held.id, n, n_classes, k, Int(class_counts.unsafe_ptr()),
                                          Int(plan.sizes64.unsafe_ptr()))
        _ = held^  # the stored codes are released here; the fold block stays with the plan
    else:
        plan.words = List[Int32](length=max(n, 1), fill=0)
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
    """`StratifiedKFold.split`: the fold plan (class sizes in first-seen
    order, the allocation table), then the folds in one program (COPY), or
    past its bounds the stored-code assignment and FOLDS out of the stored
    fold block. The codes are put once (`MselHeld`) and freed when the plan
    returns or refuses."""
    # The program route with the image's resident store (lane
    # vdf-scope-split2): the codes go up once, into the program store, and
    # both programs (first-seen, folds) read that slot. Otherwise the
    # helper table's store holds them (`MselHeld`) and the fold program
    # takes the caller's codes as its host input.
    var on_slot = (route.store and form != MSEL_LABELS_FIRST_SEEN and labels.n > 0 and k <= 256
                   and msel_fold_rows_fits(labels.n, k))
    var held = MselHeld()
    var xm = MselProgSlot()
    var enc = MselEnc()
    if on_slot:
        xm = MselProgSlot(route, labels)
        enc = msel_first_seen_prog(route, labels, form, xm.slot)
    else:
        if form != MSEL_LABELS_FIRST_SEEN:
            held = MselHeld(route, labels)
        enc = msel_first_seen_enc(route, labels, form, held)
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
        # The caller's codes are the program's one n-word input (the store
        # slot when `on_slot`: no upload); the program renumbers them by the
        # rank table (`gather_i32`), so the first-seen codes are formed where
        # they are used.
        var prog = MselProg()
        var enc_slot = prog.put_i32(labels.address, n)
        if len(enc.rank) > 0:
            var code_slot = enc_slot
            var rank_slot = prog.plan.input_words(enc.rank.copy())
            enc_slot = prog.plan.scratch(n)
            prog.stage("gather_i32", n, [rank_slot, code_slot, enc_slot])
        var rows = msel_stratified_fold_rows_plan(prog.plan, enc_slot, n, enc.counts, Int(alloc.unsafe_ptr()), k,
                                                  rng, shuffle)
        if on_slot:
            plan.arena = msel_prog_run_slots(route, prog, [xm.slot])
        else:
            plan.arena = msel_prog_run(route, prog)
        _ = alloc  # the table address above is not an owner
        _ = cnt
        _plan_fold_rows(plan, rows)
        _ = held^  # the stored codes are released here (the program read the caller's)
        _ = xm^  # the program store's slot is released here (both programs read it)
        return plan^
    # the same assignment by the stored-code `strat_fold_assign_i32`, then
    # each fold's rows out of the stored fold block
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
    # The sizes are the folds' own counts (the assignment counts each fold
    # as it deals), so the selection's count check always holds.
    plan.sizes64 = List[Int64](length=max(k, 1), fill=0)
    plan.folds_entry = route.helpers
    plan.folds_id = msel_strat_fold_assign_res(route.helpers, held.id, n, n_classes, k, enc.rank_address(), enc.width,
                                               Int(alloc.unsafe_ptr()),
                                               Int(perms.unsafe_ptr()) if shuffle and n > 0 else 0,
                                               Int(cnt.unsafe_ptr()), Int(plan.sizes64.unsafe_ptr()))
    _ = enc  # the addresses above are not owners
    _ = alloc
    _ = perms
    _ = cnt
    _ = held^  # the stored codes are released here; the fold block stays with the plan
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
    group-level plan (a host pass over groups), FOLDS. The codes are put
    once (`MselHeld`, every form) and both helper calls read them where
    they are: the first-seen sizes and rank table (2k words down), then
    the group x class table with its class columns already in first-seen
    order (MH_GROUP_CLASS_COUNTS_RES, layout V4; lane vdf-scope-split2,
    2026-11-24: the groups go up, m x classes words come down). Before, the
    base table's `bincount2_i32` uploaded the codes and the groups again
    and the class columns were permuted on the host."""
    var held = MselHeld(route, labels)
    var enc = msel_first_seen_enc(route, labels, form, held)
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
    if held.id >= 0 and m >= 1 and n_classes >= 1:
        if groups.n != n:
            raise Error("bincount2_i32: the label and group codes differ in length")
        # FIRST_SEEN: the codes are their own ranks over n_classes codes.
        var width = enc.width if len(enc.rank) > 0 else n_classes
        _ = msel_group_class_counts_res(route.helpers, held.id, n, width, groups.address, m, enc.rank_address(),
                                        n_classes, Int(dist.unsafe_ptr()))
    _ = held^  # the stored codes are released here
    _ = enc  # the rank address above is not an owner
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
    """`StratifiedShuffleSplit._native_split`: the rows class by class (a
    stable `group_sort` of the class codes, one program, read back once:
    N086), the class sizes from its offsets and the class refusals. Each
    split is then one program (`msel_split_fill`). approx_mode hands out
    exactly n_train (then n_test) draws, so both lengths are known before
    any split's program runs. The codes go up once, as that program's input
    (lane vdf-scope-split: the class sizes were a `fold_ids` call of their
    own, which uploaded the codes and downloaded n fold words)."""
    var n = labels.n
    var k = labels.m
    if k == 0:
        raise Error("min() iterable argument is empty")
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
    # Class c's rows are ORD[OFF[c] .. OFF[c + 1]): its size.
    for c in range(k):  # driver-loop: over classes
        var size = Int(plan.offsets32[c + 1]) - Int(plan.offsets32[c])
        plan.counts.append(size)
        plan.counts32.append(Int32(size))
    var least = plan.counts[0]
    for c in range(k):  # driver-loop: over classes
        least = min(least, plan.counts[c])
    if least < 2:
        raise Error("The least populated classes in y have only 1 member, which is too few. The "
                    + "minimum number of groups for any class cannot be less than 2.")
    if n_train < k:
        raise Error("The train_size = " + String(n_train) + " should be greater or equal to the number of classes = " + String(k))
    if n_test < k:
        raise Error("The test_size = " + String(n_test) + " should be greater or equal to the number of classes = " + String(k))
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


def msel_plan_leave_one_out(n: Int) raises -> MselSplitPlan:
    """`LeaveOneOut.split` as one plan of n splits (lane mf-split5: a member
    fit takes every split at once; the public door makes them one at a time
    through `msel_leave_one`, the same helper `msel_split_fill` calls for
    each split of this plan). The door's refusal, in its words."""
    if n <= 1:
        raise Error("Cannot perform LeaveOneOut with n_samples=" + String(n) + ".")
    var plan = MselSplitPlan()
    plan.form = MSEL_FILL_LEAVE_ONE
    plan.n = n
    for _ in range(n):  # driver-loop: over splits (one row each)
        plan.add(n - 1, 1)
    return plan^


def msel_plan_leave_p_out(n: Int, p: Int) raises -> MselSplitPlan:
    """`LeavePOut.split` as one plan (lane mf-split5): the combinations of
    range(n) in itertools' lexicographic order, each combination's p rows
    the test side of one split (INDEXED: the fill `msel_plan_indexed` gives
    the public door's lazy splits). p = 0 is one split with every row in
    train. The door's refusals, in its words and order."""
    if n <= p:
        raise Error("p=" + String(p) + " must be strictly less than the number of samples=" + String(n))
    if p < 0:
        raise Error("r must be non-negative")
    var plan = MselSplitPlan()
    plan.form = MSEL_FILL_INDEXED
    plan.n = n
    if p < 1:
        plan.test_src.append(0)
        plan.add(n, 0)
        return plan^
    var combo = List[Int](capacity=p)
    for i in range(p):  # small-loop(the first combination's p rows)
        combo.append(i)
    var count = 0
    while True:  # driver-loop: over the combinations (lexicographic, as itertools.combinations)
        for i in range(p):  # small-loop(one combination's p rows)
            plan.rows64.append(Int64(combo[i]))
        count += 1
        var at = p - 1
        while at >= 0 and combo[at] == n - p + at:  # small-loop(the last position that can still move)
            at -= 1
        if at < 0:
            break
        combo[at] += 1
        for j in range(at + 1, p):  # small-loop(the positions after it, in order)
            combo[j] = combo[j - 1] + 1
    # Every combination is written: the addresses are taken only now (the
    # appends above may have moved the words).
    var base = Int(plan.rows64.unsafe_ptr())
    for s in range(count):  # driver-loop: over splits (their test rows' addresses)
        plan.test_src.append(base + 8 * p * s)
        plan.add(n - p, p)
    return plan^


def msel_repeated_check(base: MselSeed) raises:
    """`_Repeated.split`'s first refusal: a base random_state that is not
    None or a non-negative int."""
    if base.bad:
        raise Error(msel_seed_refusal())


def msel_plan_repeat(route: MselSplitRoute, n: Int, labels: MselCodes, form: Int, k: Int, stratified: Bool,
                     base: MselSeed, r: Int) raises -> MselSplitPlan:
    """Repeat r of `_Repeated.split` (lane mf-split5: ONE body, called by the
    public door's `msel_repeated_split` and by a member fit): a shuffled
    StratifiedKFold over `labels` or KFold over n rows, seeded with
    `_mix_seed(base, r)`."""
    var seed = MselSeed(UInt64(msel_mix_seed(base.value, r)), False)
    if stratified:
        return msel_plan_stratified(route, labels, form, k, True, seed)
    return msel_plan_kfold(route, n, k, True, seed, False)


@fieldwise_init
struct MselSizeWord(Copyable, Movable):
    """One `_validate_shuffle_split` size as UNPACK read it (lane mf-split5):
    MSEL_SIZE_NONE, an integer (`whole`), a real (`real`, read at `bits` of
    precision: 64 for a Python float or a float64 scalar, 32 or 16 for a
    NumPy scalar of that width, whose product with the row count NumPy
    rounds at that width) or a refused value; `text` is the value's `str`
    for the refusal words."""

    var kind: Int
    var whole: Int
    var real: Float64
    var bits: Int
    var text: String


def msel_size_none() -> MselSizeWord:
    return MselSizeWord(MSEL_SIZE_NONE, 0, 0.0, 64, String("None"))


def _size_product(w: MselSizeWord, n: Int, up: Bool) raises -> Int:
    """`math.ceil(w * n)` (up) or `math.floor(w * n)` at the size's precision."""
    var v = 0.0
    # NumPy 2 (NEP 50) converts the Python int n to the scalar's width and
    # multiplies there; n is exact in float64, so one rounding to the width.
    if w.bits == 32:
        var p32 = w.real.cast[DType.float32]() * Float64(n).cast[DType.float32]()
        v = (ceil(p32) if up else floor(p32)).cast[DType.float64]()
    elif w.bits == 16:
        var p16 = w.real.cast[DType.float16]() * Float64(n).cast[DType.float16]()
        v = (ceil(p16) if up else floor(p16)).cast[DType.float64]()
    else:
        var p64 = w.real * Float64(n)
        v = ceil(p64) if up else floor(p64)
    if isinf(v):
        raise Error("OverflowError: cannot convert float infinity to integer")
    if isnan(v):
        raise Error("ValueError: cannot convert float NaN to integer")
    return Int(v)


def msel_shuffle_sizes_run(n: Int, test: MselSizeWord, train: MselSizeWord, sum_over: Bool,
                           sum_text: String) raises -> Tuple[Int, Int]:
    """`_validate_shuffle_split` (lane mf-split5: ONE body, the public door's
    and a member fit's): (n_train, n_test) of n rows with its refusals in
    the door's words and order. `test` is the size after the default
    substitution UNPACK made (both None: the splitter's default test size);
    `sum_over` / `sum_text`: whether two real sizes add up past 1 and their
    sum's `str` (both independent of n, read by UNPACK)."""
    if ((test.kind == MSEL_SIZE_INT and (test.whole >= n or test.whole <= 0))
            or (test.kind == MSEL_SIZE_REAL and not (0.0 < test.real and test.real < 1.0))):
        raise Error("test_size=" + test.text + " should be either positive and smaller than the number of "
                    + "samples " + String(n) + " or a float in the (0, 1) range")
    if ((train.kind == MSEL_SIZE_INT and (train.whole >= n or train.whole <= 0))
            or (train.kind == MSEL_SIZE_REAL and not (0.0 < train.real and train.real < 1.0))):
        raise Error("train_size=" + train.text + " should be either positive and smaller than the number "
                    + "of samples " + String(n) + " or a float in the (0, 1) range")
    if train.kind == MSEL_SIZE_BAD:
        raise Error("Invalid value for train_size: expected a native numeric size")
    if test.kind == MSEL_SIZE_BAD:
        raise Error("Invalid value for test_size: expected a native numeric size")
    if train.kind == MSEL_SIZE_REAL and test.kind == MSEL_SIZE_REAL and sum_over:
        raise Error("The sum of test_size and train_size = " + sum_text
                    + ", should be in the (0, 1) range. Reduce test_size and/or train_size.")
    var n_test = 0
    var n_train = 0
    if test.kind == MSEL_SIZE_REAL:
        n_test = _size_product(test, n, True)
    else:
        n_test = test.whole
    if train.kind == MSEL_SIZE_REAL:
        n_train = _size_product(train, n, False)
    else:
        n_train = train.whole
    if train.kind == MSEL_SIZE_NONE:
        n_train = n - n_test
    elif test.kind == MSEL_SIZE_NONE:
        n_test = n - n_train
    var total = n_train + n_test
    if total > n:
        raise Error("The sum of train_size and test_size = " + String(total) + ", should be smaller than "
                    + "the number of samples " + String(n) + ". Reduce test_size and/or train_size.")
    if n_train == 0:
        raise Error("With n_samples=" + String(n) + ", test_size=" + test.text + " and train_size="
                    + train.text + ", the resulting train set will be empty. Adjust any of the "
                    + "aforementioned parameters.")
    return (n_train, n_test)


# ---------------------------------------------------------------- the rows


def _fill_folds(route: MselSplitRoute, mut plan: MselSplitPlan) raises:
    """Every fold's (train, test) in ONE `select_folds_i64` call (N051), out
    of the stored fold block when the plan holds one (layout V3; the block
    is freed after), else over `plan.words`. An empty side shares one
    private scratch run, as before."""
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
    var bad = -1
    if plan.folds_id >= 0:
        bad = msel_select_folds_res(route.helpers, plan.folds_id, n, k, Int(plan.sizes64.unsafe_ptr()),
                                    Int(tests.unsafe_ptr()), Int(trains.unsafe_ptr()))
        plan.release_folds()
    else:
        bad = msel_select_folds_i64(route.helpers, Int(plan.words.unsafe_ptr()), n, k,
                                    Int(plan.sizes64.unsafe_ptr()), Int(tests.unsafe_ptr()), Int(trains.unsafe_ptr()))
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
    elif plan.form == MSEL_FILL_LEAVE_ONE:
        for s in range(splits):  # driver-loop: over splits (one helper call each, the door's per-split call)
            msel_leave_one(route, n, s, plan.train_dst[s], plan.test_dst[s])
    elif splits != 0:
        raise Error("RuntimeError: mojolearn: a split plan has no fill form")

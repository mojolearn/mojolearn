# SPDX-License-Identifier: Apache-2.0
"""RUN of a SET OF NATIVE MEMBERS fitted together (lane bm-meta-2,
2026-11-22; docs/plans/PYTHON_AT_THE_BOUNDARY.md): the typed core of
bindings/member_set_door.mojo, moved here so that every meta-estimator's fit
RUN (Voting, MultiOutput, OneVsRest, Bagging, Stacking, ...) and a Mojo test
can fit a member set without CPython. No Python: no Python object parameter,
field, local or import.

The door keeps what touches CPython: UNPACK (`member_set_admit`: the
member's estimator, its family's admission, its plan words), THE PIN RULE
(a reference to the Python object that owns X), THE GIL RULE's lock
(releasing the GIL around `member_set_fit_free` and around a
`MEMBER_GIL_DOOR` member's turn) and PACK. Everything a fit reads is in
`MemberSetRun`, built by UNPACK; the module note of
bindings/member_set_door.mojo describes the two modes, the class-cover pass,
the gather and the lifetime rule, unchanged by the move: the same functions
run in the same order, so no bits change.

The RUN, in order (`member_set_fit` is all of it for a caller that holds no
GIL, a Mojo test or another RUN):
  1. `member_set_prepare`: the worker count, the workers' units (members
     whose kind has `MEMBER_GIL_FREE`) and the class-cover units.
  2. `member_set_fit_free` (no GIL may be needed): the class-cover pass on
     the host cores, then, unless a member's rows lack a class, the
     GIL-free members on `workers` host tasks.
  3. `member_set_turn` per other member, one after the other on the calling
     thread (the door holds or releases the GIL around each turn).
  4. `member_set_first_failure`: the first failing member in member order
     (its words in `errors`); the caller releases every state
     (`member_set_run_release`) and raises them.
"""
from std.memory import unsafe_memcpy
from std.time import perf_counter_ns
from bindings.native_unit_os_threads_run import native_unit_cores
from core.device_route import DEVICE_CPU
from core.host_parallel import host_parallelize
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free, member_pin_settle,
    MEMBER_OP_FIT, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_I64, MEMBER_CAN_ROW_INDEX, MEMBER_CAN_COLUMN_INDEX, MEMBER_SCOPE_NONE,
)

#: Words below which a gather is not split (a task costs more than the
#: copy); a cost rule over any shape.
comptime _SET_SPLIT_WORDS = 1048576

#: `MemberSetRun.gil_free` words (THE GIL RULE): a worker member, a member
#: fitted on the calling thread with the GIL released around its turn
#: (`MEMBER_GIL_DOOR`), a member fitted on the calling thread with the GIL
#: held.
comptime MEMBER_SET_WORKER = UInt8(1)
comptime MEMBER_SET_TURN_RELEASED = UInt8(2)
comptime MEMBER_SET_TURN_HELD = UInt8(0)


struct MemberSetRun(Movable):
    """Everything typed the member fits read. Built by UNPACK; workers read
    it through one address and write only their own member's slots."""
    var plans: List[MemberPlan]
    # X: host float32, row-major, `rows` x `cols`.
    var x: Int
    var rows: Int
    var cols: Int
    # Per member: the address of its target column (`rows` values of
    # `y_dtype`: MEMBER_F32, or MEMBER_I32 class codes) and of its float32
    # sample weights (0: none).
    var targets: List[Int]
    var weights: List[Int]
    var y_dtype: Int
    # THE 2-D y (core/member_abi.mojo layout 4): the targets' columns, one per
    # output (1: every single-output caller). Each member's target address is
    # then `rows` x `y_cols` row-major words; its plan carries the per-output
    # class counts (`MemberPlan.class_counts`).
    var y_cols: Int
    # Per member, four words: row draw address and count (count 0: every
    # row), column draw address and count (count 0: every column). The
    # draws are `index_dtype` numbers (MEMBER_I32 or MEMBER_I64), in the
    # order the member sees them, repeats allowed.
    var picks: List[Int]
    var index_dtype: Int
    # Per member: the address and row count of an X block of its own (a cv
    # fold's rows the caller already gathered, shared by that fold's
    # members), 0: the set's X. A member with its own block has no draws.
    var own_x: List[Int]
    var own_rows: List[Int]
    # The caller's worker bound (an estimator's `n_jobs`), 0: none.
    var cap: Int
    var workers: Int
    var states: List[Int]
    var failed: List[UInt8]
    var errors: List[String]
    # The class-cover pass (bindings/member_set_door.mojo module note): the
    # units whose codes are read (one per distinct target and draw) and, per
    # member, 1 when its codes lack a class or hold a code out of range.
    var cover_units: List[Int]
    var uncovered: List[UInt8]
    # Per member: 1 when its KIND has `MEMBER_CAN_TEST_ROWS` (its entry
    # refuses non-finite X, y and weights itself: the caller scans nothing
    # for it). Read once per member class at admission.
    var tests: List[UInt8]
    # THE RESIDENT SCOPE: the caller's scope handle word (MEMBER_SCOPE_NONE:
    # none) and, per member, the pin token its fit offers (0: none; the
    # worker writes 0 when the member's cache took it).
    var scope: Int
    var pin_tokens: List[Int]
    # THE LIFETIME RULE (MEMBER-ABI-V2.md item 34): every buffer handed to a
    # member's FIT stays valid until its state is freed or has been exported.
    # Every state of a set waits for PACK, so the drawn block a worker gathers
    # for a member whose KIND borrows its fit buffers (`borrows[u]`,
    # MEMBER_BORROWS_FIT) is the RUN's: `part_x[u]`, `part_y[u]`, `part_w[u]`,
    # emptied once that state is freed or owns its rows. A kind that owns its
    # state gathers into the worker's own buffers, freed when its fit
    # returns.
    var borrows: List[UInt8]
    # THE GIL RULE: per member one of the MEMBER_SET_* words; `parallel`:
    # the worker members' numbers, the units of the workers' lanes.
    var gil_free: List[UInt8]
    var parallel: List[Int]
    var part_x: List[List[Float32]]
    var part_y: List[List[Int32]]
    var part_w: List[List[Float32]]
    # Per member, the fit notices PACK took from its state (empty: none).
    var notices: List[List[Float64]]
    # THE PER-UNIT COVER RULE (lane mf-cal, 2026-11-23; mf-cal2: the caller
    # reads the classes; default False): a caller whose units are
    # independent cv folds (CalibratedClassifierCV) has read every unit's
    # classes itself (xtrees/trees_calibrated_fit_run.mojo
    # `cal_fold_classes_run`: a fold that lacks a class, one class included,
    # is re-coded onto the classes it holds); the set then makes NO
    # class-cover pass and fits every unit (lane mf-cal3: no unit is left to
    # a door, so nothing is marked `uncovered` and the workers and turns no
    # longer skip one). False: the set's own cover pass, and one lacking unit
    # fails the whole set before any fit (every other caller).
    var skip_lacking: Bool
    # Per member, its fit's wall-clock seconds (lane mf-msel5: the cv tools'
    # `fit_time`, scikit-learn's `_fit_and_score` clock around one fit), 0.0
    # until fitted. One clock read before and one after each fit.
    var seconds: List[Float64]
    # THE GIVEN y (lane mf-tail5; core/member_abi.mojo `MemberCall.y_given`):
    # per member, the address of its targets as the caller was GIVEN them
    # (`given_dtype[u]`: MEMBER_F64, MEMBER_I64 or MEMBER_I32 words, rows x
    # `y_cols`, addressed exactly as `targets[u]`), 0: none. Set by
    # `given_target` only for a member whose kind has MEMBER_READS_Y_GIVEN;
    # a drawn member's are gathered with its rows (as float64 words) into
    # the worker's own buffer, which dies when the fit returns (the given y
    # is read during FIT only).
    var given: List[Int]
    var given_dtype: List[Int]

    def __init__(out self, x: Int, rows: Int, cols: Int, y_dtype: Int, index_dtype: Int):
        self.plans = List[MemberPlan]()
        self.x = x
        self.rows = rows
        self.cols = cols
        self.targets = List[Int]()
        self.weights = List[Int]()
        self.y_dtype = y_dtype
        self.y_cols = 1
        self.picks = List[Int]()
        self.index_dtype = index_dtype
        self.own_x = List[Int]()
        self.own_rows = List[Int]()
        self.cap = 0
        self.workers = 1
        self.states = List[Int]()
        self.failed = List[UInt8]()
        self.errors = List[String]()
        self.cover_units = List[Int]()
        self.uncovered = List[UInt8]()
        self.tests = List[UInt8]()
        self.scope = MEMBER_SCOPE_NONE
        self.pin_tokens = List[Int]()
        self.notices = List[List[Float64]]()
        self.borrows = List[UInt8]()
        self.gil_free = List[UInt8]()
        self.parallel = List[Int]()
        self.part_x = List[List[Float32]]()
        self.part_y = List[List[Int32]]()
        self.part_w = List[List[Float32]]()
        self.skip_lacking = False
        self.seconds = List[Float64]()
        self.given = List[Int]()
        self.given_dtype = List[Int]()

    def count(self) -> Int:
        return len(self.plans)

    def add(mut self, var plan: MemberPlan, target: Int, weights: Int, row_draw: Int, row_count: Int,
            column_draw: Int, column_count: Int, tests: UInt8, borrows: UInt8, gil_free: UInt8):
        """One member of the set: its plan, its target and weights
        addresses, its draws (counts 0: every row / column) and its kind's
        three answers. The member reads the set's X; `own_block` gives it
        one of its own."""
        self.tests.append(tests)
        self.plans.append(plan^)
        self.targets.append(target)
        self.weights.append(weights)
        self.picks.append(row_draw)
        self.picks.append(row_count)
        self.picks.append(column_draw)
        self.picks.append(column_count)
        self.own_x.append(0)
        self.own_rows.append(0)
        self.states.append(0)
        self.failed.append(UInt8(0))
        self.errors.append(String(""))
        self.uncovered.append(UInt8(0))
        self.pin_tokens.append(0)
        self.notices.append(List[Float64]())
        self.borrows.append(borrows)
        self.gil_free.append(gil_free)
        self.part_x.append(List[Float32]())
        self.part_y.append(List[Int32]())
        self.part_w.append(List[Float32]())
        self.seconds.append(0.0)
        self.given.append(0)
        self.given_dtype.append(MEMBER_F64)

    def given_target(mut self, address: Int, dtype: Int):
        """THE GIVEN y of the member added last (its targets' words as the
        caller was given them, addressed as its target is). The caller sets
        it only for a kind with MEMBER_READS_Y_GIVEN."""
        var last = len(self.given) - 1
        if last < 0:
            return
        self.given[last] = address
        self.given_dtype[last] = dtype

    def own_block(mut self, x: Int, rows: Int):
        """The member added last reads its own float32 C-order X block
        (`rows` x the set's columns at `x`) and not the set's X. THE
        LIFETIME RULE: the caller keeps it alive until every state of the
        set is packed or released."""
        var last = len(self.own_x) - 1
        if last < 0:
            return
        self.own_x[last] = x
        self.own_rows[last] = rows

    def part_free(mut self, u: Int):
        """Member `u`'s held gathered block is no longer read (its state was
        freed, or owns its rows)."""
        if len(self.part_x[u]) > 0 or len(self.part_y[u]) > 0:
            self.part_x[u] = List[Float32]()
            self.part_y[u] = List[Int32]()
            self.part_w[u] = List[Float32]()


def member_set_run_tests(run: MemberSetRun, u: Int) -> Bool:
    """Member `u`'s kind refuses non-finite X, y and weights itself."""
    return u >= 0 and u < len(run.tests) and run.tests[u] != UInt8(0)


def member_set_run_host(run: MemberSetRun) -> Bool:
    """Every member added so far runs on the CPU route."""
    for u in range(len(run.plans)):  # driver-loop: over the members (their resolved routes)
        if run.plans[u].route.device != DEVICE_CPU:
            return False
    return True


def member_set_run_cores(run: MemberSetRun) -> Int:
    """The host cores, bounded by the caller's worker bound when it gave one."""
    var cores = native_unit_cores()
    if run.cap > 0:
        cores = min(cores, run.cap)
    return max(cores, 1)


def member_set_run_workers(run: MemberSetRun) -> Int:
    """The host cores (bounded by `cap`) bounded by the count of worker
    members when every member's route is the CPU; one when any member runs
    on a GPU (the members share one device and its memory) or no member's
    kind runs on a worker."""
    if not member_set_run_host(run):
        return 1
    var free = 0
    for u in range(len(run.gil_free)):  # driver-loop: over the members (one flag each)
        if run.gil_free[u] == MEMBER_SET_WORKER:
            free += 1
    return max(min(member_set_run_cores(run), free), 1)


def member_set_run_width(run: MemberSetRun) -> Int:
    """How many members a caller that fits in waves puts in one set: the
    host cores when every member added so far runs on the CPU, else one."""
    if not member_set_run_host(run):
        return 1
    return member_set_run_cores(run)


@always_inline
def _set_pick(address: Int, i: Int, wide: Bool) -> Int:
    if wide:
        return Int(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=address)[i])
    return Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=address)[i])


def _set_gather_span(run: MutPointer[MemberSetRun, MutUntrackedOrigin], u: Int, first: Int, count: Int,
    dst_x: Int, dst_y: Int, dst_w: Int):
    """Drawn rows `first .. first + count` of member `u`: its X block (drawn
    columns), its targets and its weights into the worker's buffers. Rows
    are independent: a caller may split the range."""
    var wide = run[].index_dtype == MEMBER_I64
    var row_draw = run[].picks[4 * u]
    var row_count = run[].picks[4 * u + 1]
    var column_draw = run[].picks[4 * u + 2]
    var column_count = run[].picks[4 * u + 3]
    var cols = run[].cols
    var width = column_count if column_count > 0 else cols
    var source = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=run[].x)
    var target = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dst_x)
    var y_source = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=run[].targets[u])
    var y_target = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=dst_y)
    var has_w = run[].weights[u] != 0 and dst_w != 0
    var yc = run[].y_cols
    for i in range(first, first + count):
        var r = _set_pick(row_draw, i, wide) if row_count > 0 else i
        if column_count > 0:
            for j in range(width):
                target[i * width + j] = source[r * cols + _set_pick(column_draw, j, wide)]
        else:
            unsafe_memcpy(dest=target + i * width, src=source + r * cols, count=cols)
        # Targets are 4-byte words of either element type: copied as bits
        # (THE 2-D y: one word per output). A member fitted without a
        # target (address 0, lane mf-msel5b: an unsupervised base) has none.
        if run[].targets[u] != 0:
            for c in range(yc):
                y_target[i * yc + c] = y_source[r * yc + c]
        if has_w:
            MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dst_w)[i] = MutPointer[
                Float32, MutUntrackedOrigin](unsafe_from_address=run[].weights[u])[r]


def _set_gather_given_span(address: Int, u: Int, first: Int, count: Int, dst: Int):
    """Drawn rows `first .. first + count` of member `u`'s GIVEN y (lane
    mf-tail5) as float64 words at `dst` (exact for MEMBER_F64, MEMBER_I32
    and |MEMBER_I64| <= 2^53). Rows are independent: a caller may split the
    range."""
    var run = MutPointer[MemberSetRun, MutUntrackedOrigin](unsafe_from_address=address)
    var wide = run[].index_dtype == MEMBER_I64
    var row_draw = run[].picks[4 * u]
    var row_count = run[].picks[4 * u + 1]
    var yc = run[].y_cols
    var kind = run[].given_dtype[u]
    var src = run[].given[u]
    var target = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=dst)
    for i in range(first, first + count):  # native data loop (the drawn rows' given targets)
        var r = _set_pick(row_draw, i, wide) if row_count > 0 else i
        for c in range(yc):
            var at = r * yc + c
            if kind == MEMBER_I64:
                target[i * yc + c] = Float64(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=src)[at])
            elif kind == MEMBER_I32:
                target[i * yc + c] = Float64(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=src)[at])
            else:
                target[i * yc + c] = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=src)[at]


def _set_gather_given(address: Int, u: Int, count: Int, dst: Int, split: Bool):
    """THE GIVEN y of drawn member `u` (`_set_gather_given_span`); `split`:
    this member is the only one running, so the copy is split across the
    host cores, as `_set_gather` splits the block."""
    var cores = native_unit_cores()
    if not split or cores <= 1 or count < _SET_SPLIT_WORDS:
        _set_gather_given_span(address, u, 0, count, dst)
        return
    var chunks = min(cores, count)
    var per = (count + chunks - 1) // chunks

    def _chunk(part: Int) {imm address, imm u, imm count, imm dst, imm per}:
        var first = part * per
        if first >= count:
            return
        _set_gather_given_span(address, u, first, min(per, count - first), dst)

    host_parallelize(_chunk, chunks)


def _set_gather(address: Int, u: Int, count: Int, width: Int, dst_x: Int, dst_y: Int, dst_w: Int, split: Bool):
    """One member's gathered block. `split`: this member is the only one
    running, so the copy itself is split across the host cores."""
    var run = MutPointer[MemberSetRun, MutUntrackedOrigin](unsafe_from_address=address)
    var cores = native_unit_cores()
    if not split or cores <= 1 or count * width < _SET_SPLIT_WORDS:
        _set_gather_span(run, u, 0, count, dst_x, dst_y, dst_w)
        return
    var chunks = min(cores, count)
    var per = (count + chunks - 1) // chunks

    def _chunk(part: Int) {imm address, imm u, imm count, imm dst_x, imm dst_y, imm dst_w, imm per}:
        var first = part * per
        if first >= count:
            return
        _set_gather_span(MutPointer[MemberSetRun, MutUntrackedOrigin](unsafe_from_address=address), u, first,
            min(per, count - first), dst_x, dst_y, dst_w)

    host_parallelize(_chunk, chunks)


def _set_cover_span(targets: Int, draw: Int, wide: Bool, first: Int, count: Int, classes: Int, seen: Int) -> Bool:
    """Mark in the `classes` bytes at `seen` every code of the int32 codes at
    `targets` over rows `first .. first + count` (the rows the draw at `draw`
    names, or the rows themselves when `draw` is 0). False when a code is
    outside 0 .. classes-1. Rows are independent: a caller may split the
    range, one `seen` block per part."""
    var y = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=targets)
    var marks = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=seen)
    var inside = True
    for i in range(first, first + count):
        var r = _set_pick(draw, i, wide) if draw != 0 else i
        var code = Int(y[r])
        if code < 0 or code >= classes:
            inside = False
        else:
            marks[code] = UInt8(1)
    return inside


def _set_cover_unit(address: Int, u: Int, split: Bool) -> Bool:
    """True when classifier member `u`'s own rows hold every code
    0 .. classes-1 and no other. `split`: this is the only unit being read,
    so its rows are split across the host cores."""
    var run = MutPointer[MemberSetRun, MutUntrackedOrigin](unsafe_from_address=address)
    var classes = run[].plans[u].classes
    if classes <= 0 or run[].y_dtype != MEMBER_I32:
        return True
    var own = run[].own_x[u] != 0
    var row_count = run[].picks[4 * u + 1]
    var count = run[].own_rows[u] if own else (row_count if row_count > 0 else run[].rows)
    var draw = 0 if (own or row_count == 0) else run[].picks[4 * u]
    var wide = run[].index_dtype == MEMBER_I64
    var targets = run[].targets[u]
    var cores = native_unit_cores()
    var chunks = 1
    if split and cores > 1 and count >= _SET_SPLIT_WORDS:
        chunks = cores
    var seen = List[UInt8](length=chunks * classes, fill=UInt8(0))
    var outside = List[UInt8](length=chunks, fill=UInt8(0))
    var seen_address = Int(seen.unsafe_ptr())
    var outside_address = Int(outside.unsafe_ptr())
    var per = (count + chunks - 1) // chunks

    def _part(part: Int) {imm targets, imm draw, imm wide, imm count, imm classes, imm seen_address,
                          imm outside_address, imm per}:
        var first = part * per
        if first >= count:
            return
        if not _set_cover_span(targets, draw, wide, first, min(per, count - first), classes,
                               seen_address + part * classes):
            MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=outside_address)[part] = UInt8(1)

    if chunks == 1:
        _part(0)
    else:
        host_parallelize(_part, chunks)
    var covered = True
    for part in range(chunks):  # driver-loop: over the parts of this unit's rows
        if outside[part] != UInt8(0):
            covered = False
    if covered:
        for code in range(classes):  # driver-loop: over the classes (each part's mark)
            var marked = UInt8(0)
            for part in range(chunks):  # driver-loop: over the parts of this unit's rows
                marked = marked | seen[part * classes + code]
            if marked == UInt8(0):
                covered = False
                break
    # Integer addresses carry no lifetime: the mark blocks live to here.
    _ = len(seen)
    _ = len(outside)
    return covered


def _member_set_cover_lane(address: Int, lane: Int, lanes: Int):
    """One worker of the class-cover pass: units lane, lane + lanes, ... of
    `cover_units`; each writes only its own `uncovered` slot."""
    var run = MutPointer[MemberSetRun, MutUntrackedOrigin](unsafe_from_address=address)
    var total = len(run[].cover_units)
    var i = lane
    while i < total:  # driver-loop: over this worker's units
        var u = run[].cover_units[i]
        if not _set_cover_unit(address, u, lanes <= 1):
            run[].uncovered[u] = UInt8(1)
        i += lanes


def _set_same_codes(run: MemberSetRun, u: Int, v: Int) -> Bool:
    """Members `u` and `v` read the same codes over the same rows (one
    target shared by a set's members, no draw): one cover pass serves both."""
    return (run.targets[u] == run.targets[v] and run.own_rows[u] == run.own_rows[v]
        and (run.own_x[u] != 0) == (run.own_x[v] != 0) and run.plans[u].classes == run.plans[v].classes
        and run.picks[4 * u + 1] == 0 and run.picks[4 * v + 1] == 0)


def _member_set_unit(address: Int, u: Int, alone: Bool) raises:
    """Fit member `u` in a new state of its own; the state stays in
    `states[u]` for PACK. A failure releases it. `alone`: no other member
    is being fitted (one worker, or a turn on the calling thread), so a
    gather is split across the host cores. Touches no CPython."""
    var run = MutPointer[MemberSetRun, MutUntrackedOrigin](unsafe_from_address=address)
    var own = run[].own_x[u] != 0
    var x_address = run[].own_x[u] if own else run[].x
    var rows = run[].own_rows[u] if own else run[].rows
    var cols = run[].cols
    var row_draw = run[].picks[4 * u]
    var row_count = run[].picks[4 * u + 1]
    var column_draw = run[].picks[4 * u + 2]
    var column_count = run[].picks[4 * u + 3]
    var drawn = not own and (row_count > 0 or column_count > 0)
    var member = run[].plans[u].member
    var by_index = (drawn and (row_count == 0 or member.can(MEMBER_CAN_ROW_INDEX))
        and (column_count == 0 or member.can(MEMBER_CAN_COLUMN_INDEX)))
    var x_part = List[Float32]()
    var y_part = List[Int32]()
    var w_part = List[Float32]()
    var given_part = List[Float64]()  # THE GIVEN y's gathered rows (FIT only)
    # THE LIFETIME RULE: a borrowing kind's gathered block is the run's (its
    # state reads it again at PACK); this unit's own slots, no other worker's.
    var hold = run[].borrows[u] != UInt8(0)
    var started = perf_counter_ns()
    var call = member_new(run[].plans[u])
    call.scope = run[].scope
    call.pin = 0
    # THE PIN RULE: the member's X object is offered unless the worker
    # gathers the drawn block into its own buffer below.
    var offer = run[].pin_tokens[u]
    if drawn and not by_index:
        offer = 0
    var done = False
    try:
        call.x = MemberMatrix(x_address, rows, cols)
        var y_free = run[].targets[u] == 0  # a fit without a target (lane mf-msel5b)
        if not y_free:
            call.y = MemberMatrix(run[].targets[u], rows, run[].y_cols, run[].y_dtype)
        var given = run[].given[u] if not y_free else 0
        if given != 0:
            # THE GIVEN y (lane mf-tail5): a view addressed as y (a row index
            # selects both); a gathered member's is gathered below.
            call.y_given = MemberMatrix(given, rows, run[].y_cols, run[].given_dtype[u])
        if run[].weights[u] != 0:
            call.weights = MemberMatrix(run[].weights[u], rows, 1)
        if by_index:
            if row_count > 0:
                call.row_index = MemberMatrix(row_draw, row_count, 1, run[].index_dtype)
            if column_count > 0:
                call.column_index = MemberMatrix(column_draw, column_count, 1, run[].index_dtype)
        elif drawn:
            var n = row_count if row_count > 0 else rows
            var width = column_count if column_count > 0 else cols
            var x_at = 0
            var y_at = 0
            var w_address = 0
            if hold:
                run[].part_x[u] = List[Float32](length=n * width, fill=0.0)
                run[].part_y[u] = List[Int32](length=n * run[].y_cols, fill=Int32(0))
                x_at = Int(run[].part_x[u].unsafe_ptr())
                y_at = Int(run[].part_y[u].unsafe_ptr())
                if run[].weights[u] != 0:
                    run[].part_w[u] = List[Float32](length=n, fill=0.0)
                    w_address = Int(run[].part_w[u].unsafe_ptr())
            else:
                x_part = List[Float32](length=n * width, fill=0.0)
                y_part = List[Int32](length=n * run[].y_cols, fill=Int32(0))
                x_at = Int(x_part.unsafe_ptr())
                y_at = Int(y_part.unsafe_ptr())
                if run[].weights[u] != 0:
                    w_part = List[Float32](length=n, fill=0.0)
                    w_address = Int(w_part.unsafe_ptr())
            _set_gather(address, u, n, width, x_at, y_at, w_address, alone)
            call.x = MemberMatrix(x_at, n, width)
            if not y_free:
                call.y = MemberMatrix(y_at, n, run[].y_cols, run[].y_dtype)
            if given != 0:
                given_part = List[Float64](length=n * run[].y_cols, fill=0.0)
                _set_gather_given(address, u, n, Int(given_part.unsafe_ptr()), alone)
                call.y_given = MemberMatrix(Int(given_part.unsafe_ptr()), n, run[].y_cols, MEMBER_F64)
            if w_address != 0:
                call.weights = MemberMatrix(w_address, n, 1)
        call.pin = offer
        if offer == 0:
            _ = member_call(member, MEMBER_OP_FIT, call)
        else:
            try:
                _ = member_call(member, MEMBER_OP_FIT, call)
            finally:
                if member_pin_settle(call) != 0:
                    run[].pin_tokens[u] = 0  # the scope's cache holds this reference now
        run[].states[u] = call.state
        run[].seconds[u] = Float64(perf_counter_ns() - started) / 1e9
        done = True
    finally:
        if not done:
            run[].seconds[u] = Float64(perf_counter_ns() - started) / 1e9
            member_free(run[].plans[u], call)
            run[].part_free(u)
        # Integer addresses carry no lifetime: the gathered block lives to here.
        _ = len(x_part)
        _ = len(y_part)
        _ = len(w_part)
        _ = len(given_part)


def _member_set_lane(address: Int, lane: Int):
    """One worker (no GIL): entries lane, lane + workers, ... of `parallel`.
    A failing member records its words in its own slot; the others still
    run (each is independent)."""
    var run = MutPointer[MemberSetRun, MutUntrackedOrigin](unsafe_from_address=address)
    var count = len(run[].parallel)
    var i = lane
    while i < count:  # driver-loop: over this worker's members
        var u = run[].parallel[i]
        try:
            _member_set_unit(address, u, run[].workers <= 1)
        except error:
            run[].errors[u] = String(error)
            run[].failed[u] = UInt8(1)
        i += run[].workers


def member_set_prepare(mut run: MemberSetRun) -> Bool:
    """RUN step 1: the worker count, the workers' units and the class-cover
    units (one per distinct target of a classifier set). True when some
    member is not a worker member (the caller then takes its turns)."""
    var count = len(run.plans)
    run.workers = member_set_run_workers(run)
    run.parallel.clear()
    for u in range(count):  # driver-loop: over the members (the workers' units)
        if run.gil_free[u] == MEMBER_SET_WORKER:
            run.parallel.append(u)
    run.cover_units.clear()
    # skip_lacking: the caller read every unit's classes (no second pass).
    # A 2-D y has one class set per output: the members' own fits decide.
    if run.y_dtype == MEMBER_I32 and not run.skip_lacking and run.y_cols == 1:
        for u in range(count):  # driver-loop: over the members (one cover pass per distinct target)
            if run.plans[u].classes <= 0:
                continue
            if u > 0 and _set_same_codes(run, u, u - 1):
                continue
            run.cover_units.append(u)
    return len(run.parallel) < count


def member_set_fit_free(mut run: MemberSetRun) -> Bool:
    """RUN step 2, needing no GIL: the class-cover pass across the host
    cores, then (unless a member's rows lack a class) every worker member on
    `workers` host tasks. False when a member's rows lack a class: nothing
    was fitted. Never raises: a failing member records its words."""
    var address = Int(UnsafePointer(to=run))
    var cover_lanes = max(min(native_unit_cores(), len(run.cover_units)), 1)
    var lanes = run.workers

    def _cover(lane: Int) {imm address, imm cover_lanes}:
        _member_set_cover_lane(address, lane, cover_lanes)

    def _lane(lane: Int) {imm address}:
        _member_set_lane(address, lane)

    if len(run.cover_units) > 0:
        host_parallelize(_cover, cover_lanes)
        for u in range(len(run.plans)):  # driver-loop: over the members (the cover verdicts)
            if run.uncovered[u] != UInt8(0):
                return False
    if len(run.parallel) > 0:
        host_parallelize(_lane, lanes)
    return True


def member_set_turn(mut run: MemberSetRun, u: Int):
    """RUN step 3: member `u` (not a worker member) fitted on the calling
    thread. SERIAL BY RULE, not by nature (THE GIL RULE): a kind becomes a
    worker member when its trace earns `MEMBER_GIL_FREE`. The caller holds
    or releases the GIL around the call as `gil_free[u]` says. Never raises:
    a failure records its words."""
    var address = Int(UnsafePointer(to=run))
    try:
        _member_set_unit(address, u, True)
    except error:
        run.errors[u] = String(error)
        run.failed[u] = UInt8(1)


def member_set_first_failure(run: MemberSetRun) -> Int:
    """RUN step 4: the first failing member in member order, -1: none."""
    for u in range(len(run.failed)):  # driver-loop: over the members, for the first failure in member order
        if run.failed[u] != UInt8(0):
            return u
    return -1


def member_set_run_release(mut run: MemberSetRun):
    """Release every state the run still holds. Never raises."""
    for u in range(len(run.plans)):  # driver-loop: over the members (their states)
        if run.states[u] != 0:
            var call = MemberCall(run.plans[u].route, run.states[u])
            member_free(run.plans[u], call)
            run.states[u] = 0
        # After the state: nothing reads the held block any more.
        run.part_free(u)


comptime MEMBER_SET_LACKING = "ValueError: a member's rows do not hold every class of the ensemble: its own fit decides"


# ---------------------------------------------------------------- a member fit's extra inputs (lane mf-gbdt5)
struct MemberFitExtras(Copyable, Movable):
    """A member FIT's typed inputs beyond X and y (lane mf-gbdt5, 2026-11-23):
    the caller's fit parameters as its UNPACK admitted them in the
    estimator's own fit door's checks and words (for the Gbdt member:
    providers/python/member_gbdt_boundary_door.mojo
    `gbdt_member_fit_params`). Plain addresses of host words the caller's
    UNPACK holds alive until the fit returns (THE LIFETIME RULE); 0 / a count
    0: not given. No Python.

    `rows`: the fit's rows the weights and group ids cover. `eval_x`: the
    held-out set's float32 C-order rows over the caller's FULL columns
    (`eval_cols` of them); a fit on a column selection (RFE's rounds) reads
    the same columns of it (`member_fit_extras_apply`). `eval_y`: its targets
    (`eval_y_dtype`: MEMBER_F32 values, or MEMBER_I32 class codes of a
    classifier kind). `groups`: one int32 group id per fit row. `pairs`:
    `pair_count` int32 (winner, loser) rows, `pairs_weight` float32 (0: ones).
    `eval_cat_keys` (lane mf-meta5b): the held-out set's categorical keys as
    the estimator's fit door reads a text held-out X (int64 words, one row
    per categorical column, `eval_cat_count` rows, one column per held-out
    row; 0: the categorical columns' Float32 words are the keys); they cover
    the full columns only, so a fit on a column selection refuses them.

    THE EVAL SLOT. This is the per-member eval slot the member-set callers
    would carry; today only RFE hands a member fit parameters (Bagging,
    Voting, Stacking and the searches refuse every key but sample_weight,
    IterativeImputer and Calibrated take none: lane mf-split4b item 5), and
    RFE's rounds are serial and fit one member each on a block its own
    program gathered, so the slot rides RFE's member calls
    (bindings/prep_select_run.mojo `rfe_member_fit`), not `MemberSetRun`."""
    var rows: Int
    var weights: Int
    var eval_x: Int
    var eval_rows: Int
    var eval_cols: Int
    var eval_y: Int
    var eval_y_dtype: Int
    var groups: Int
    var pairs: Int
    var pair_count: Int
    var pairs_weight: Int
    var eval_cat_keys: Int
    var eval_cat_count: Int
    # THE GIVEN y (lane mf-tail5; core/member_abi.mojo `MemberCall.y_given`):
    # the fit's targets as the caller was given them (`y_given_dtype`:
    # MEMBER_F64 / _I64 / _I32 words, one per fit row and output), 0: none.
    # Handed only when the member's kind reads it (MEMBER_READS_Y_GIVEN): the
    # caller sets it after asking the kind's mask.
    var y_given: Int
    var y_given_dtype: Int

    def __init__(out self):
        self.rows = 0
        self.weights = 0
        self.eval_x = 0
        self.eval_rows = 0
        self.eval_cols = 0
        self.eval_y = 0
        self.eval_y_dtype = MEMBER_F32
        self.groups = 0
        self.pairs = 0
        self.pair_count = 0
        self.pairs_weight = 0
        self.eval_cat_keys = 0
        self.eval_cat_count = 0
        self.y_given = 0
        self.y_given_dtype = MEMBER_F64

    def given(self) -> Bool:
        return (self.weights != 0 or self.eval_rows > 0 or self.groups != 0 or self.pair_count > 0)


def _eval_columns_span(source: Int, cols: Int, columns: Int, count: Int, wide: Bool, first: Int, rows: Int,
                       dst: Int):
    """Rows `first .. first + rows` of the held-out set's columns `columns`
    (`count` ids) into `dst` (row-major, `count` wide). Rows are independent:
    a caller may split the range."""
    var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=source)
    var target = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dst)
    for i in range(first, first + rows):  # native data loop (one task's held-out rows)
        for j in range(count):
            target[i * count + j] = src[i * cols + _set_pick(columns, j, wide)]


def member_eval_columns(source: Int, rows: Int, cols: Int, columns: Int, count: Int, wide: Bool,
                        mut dst: List[Float32]) raises -> Bool:
    """The held-out set's columns for a fit on a column selection: the
    `rows` x `cols` float32 C-order block at `source` restricted to the
    `count` column ids at `columns` (int32, or int64 when `wide`), in their
    order, the way a member's drawn columns are gathered (`_set_gather_span`).
    False (nothing copied, `dst` untouched) when the selection is every
    column in order: the caller reads `source` itself. The rows are split
    across the host cores above `_SET_SPLIT_WORDS` words."""
    if columns == 0 or count <= 0:
        return False
    var identity = count == cols
    for j in range(count):  # driver-loop: over the selected column ids (bounds, and the in-order test)
        var c = _set_pick(columns, j, wide)
        if c < 0 or c >= cols:
            raise Error("member ABI: a held-out column id " + String(c) + " is outside the "
                        + String(cols) + " columns of the eval set")
        if c != j:
            identity = False
    if identity:
        return False
    dst = List[Float32](length=rows * count, fill=0.0)
    var dst_address = Int(dst.unsafe_ptr())
    var cores = native_unit_cores()
    if cores <= 1 or rows * count < _SET_SPLIT_WORDS:
        _eval_columns_span(source, cols, columns, count, wide, 0, rows, dst_address)
        return True
    var chunks = min(cores, rows)
    var per = (rows + chunks - 1) // chunks

    def _chunk(part: Int) {imm source, imm cols, imm columns, imm count, imm wide, imm rows, imm per,
                           imm dst_address}:
        var first = part * per
        if first >= rows:
            return
        _eval_columns_span(source, cols, columns, count, wide, first, min(per, rows - first), dst_address)

    host_parallelize(_chunk, chunks)
    _ = len(dst)
    return True


def member_fit_extras_apply(extras: MemberFitExtras, mut call: MemberCall, columns: Int, count: Int, wide: Bool,
                            mut held: List[Float32]) raises:
    """Fill a member FIT call's optional views from `extras`: the weights,
    the group ids, the pairs and the held-out set, whose X is the eval set's
    columns `columns` (`count` ids, the columns of the fit's X; 0: every
    column) gathered into `held` (`member_eval_columns`). THE LIFETIME RULE:
    the caller keeps `held` alive until the fit returns, and clears
    `call.eval_x` after it (`member_fit_extras_clear`)."""
    if extras.weights != 0:
        call.weights = MemberMatrix(extras.weights, extras.rows, 1)
    if extras.y_given != 0 and call.y.rows > 0:
        # THE GIVEN y, beside the call's y (the same rows: RFE's rounds select
        # columns only).
        call.y_given = MemberMatrix(extras.y_given, call.y.rows, call.y.cols, extras.y_given_dtype)
    if extras.groups != 0:
        call.groups = MemberMatrix(extras.groups, extras.rows, 1, MEMBER_I32)
    if extras.pair_count > 0:
        call.pairs = MemberMatrix(extras.pairs, extras.pair_count, 2, MEMBER_I32)
        if extras.pairs_weight != 0:
            call.pairs_weight = MemberMatrix(extras.pairs_weight, extras.pair_count, 1)
    if extras.eval_rows > 0:
        var width = extras.eval_cols
        var address = extras.eval_x
        if member_eval_columns(extras.eval_x, extras.eval_rows, extras.eval_cols, columns, count, wide, held):
            if extras.eval_cat_keys != 0:
                raise Error("member ABI: a held-out set's categorical keys cover every column of the fit's X; "
                            + "a fit on " + String(count) + " of its " + String(extras.eval_cols)
                            + " columns cannot read them")
            width = count
            address = Int(held.unsafe_ptr())
        call.eval_x = MemberMatrix(address, extras.eval_rows, width)
        call.eval_y = MemberMatrix(extras.eval_y, extras.eval_rows, 1, extras.eval_y_dtype)
        if extras.eval_cat_keys != 0:
            call.eval_cat_keys = MemberMatrix(extras.eval_cat_keys, extras.eval_cat_count, extras.eval_rows,
                                              MEMBER_I64)


def member_fit_extras_clear(mut call: MemberCall):
    """After the FIT: the call no longer names the caller's extra buffers
    (the gathered held-out block dies with the caller's frame)."""
    call.weights = MemberMatrix()
    call.groups = MemberMatrix()
    call.pairs = MemberMatrix()
    call.pairs_weight = MemberMatrix()
    call.eval_x = MemberMatrix()
    call.eval_y = MemberMatrix()
    call.eval_cat_keys = MemberMatrix()
    call.y_given = MemberMatrix()


def member_set_fit(mut run: MemberSetRun) raises:
    """The whole RUN for a caller that holds no GIL (a Mojo test, another
    RUN): steps 1 to 4, the turns one after the other on this thread. On a
    failure every state is released and the first failing member's words
    (in member order) are raised; a member whose rows lack a class raises
    `MEMBER_SET_LACKING` before any fit."""
    if len(run.plans) == 0:
        return
    var turns = member_set_prepare(run)
    if not member_set_fit_free(run):
        raise Error(MEMBER_SET_LACKING)
    if turns:
        for u in range(len(run.plans)):  # driver-loop: over the members that are not worker members
            if run.gil_free[u] != MEMBER_SET_WORKER:
                member_set_turn(run, u)
    var first = member_set_first_failure(run)
    if first >= 0:
        var words = run.errors[first].copy()
        member_set_run_release(run)
        raise Error(words)

# SPDX-License-Identifier: Apache-2.0
"""A SET OF NATIVE MEMBERS fitted together (core/member_abi.mojo): the member
fit loop of an ensemble or meta-estimator (bagging, voting, the multi-output
and one-vs-rest wrappers) without a call of a member's own fit door.

WHAT THIS REPLACES for a member whose family has published its member entry.
The callers' loops fitted one member after the other with the GIL held: each
member's fit door re-admitted X and y, read its parameters and published its
state through Python objects. Here:
  * UNPACK (GIL held, once per member): `member_set_admit` admits the
    member's estimator (`member_admit`: route, parameter words, resolved
    entry). X, the targets, the weights and the row / column draws are read
    once as addresses.
  * RUN (`member_set_run`, GIL released): the members are fitted on
    `workers` host tasks (`host_parallelize` over lanes; a lane takes
    members lane, lane + workers, ...). Each member has its own
    `MemberCall` and state. A member that draws rows or columns hands the
    draw to the family as an index view when the family reads one
    (`MEMBER_CAN_ROW_INDEX` / `MEMBER_CAN_COLUMN_INDEX`); otherwise the
    worker gathers the drawn block into its own buffers. A member over every
    row and column reads the caller's X by view: no copy.
  * PACK (`member_set_pack`, GIL held): the fitted state is written into the
    member's Python estimator (`member_pack`: the attributes its own fit
    writes, so the caller's predict paths and `estimators_` read it as
    before) and the state is released.

ALL OR NOTHING: a caller admits every member first; when one is not a native
member (`member_set_admit` answers False) nothing has run and the caller
keeps its own loop for the whole set.

WORKERS (`member_set_workers`): the host cores for members whose route is the
CPU, bounded by the member count; one when a member's route is a GPU (the
members share one device and its memory). Memory per worker: at most one
gathered block (drawn rows x drawn columns of float32, the targets and the
weights of those rows), which the serial loops held one at a time; nothing
for a member that reads the whole X or an index view. With one worker the
gather itself is split across the host cores.

A MEMBER ROUTE NEVER TURNS A REFUSAL INTO A RESULT. A classifier member is
fitted with the ensemble's class count, so a member whose own rows (a cv
fold, a bootstrap draw) lack a class would fit where its own fit door, which
encodes the classes it is given, fits another model or refuses (too few
classes), and where the caller then reads a narrower output (stacking's
"output width changed between folds"). `member_set_run` therefore reads every
classifier unit's codes first (`_set_cover_unit`: one pass over its int32
codes, the units across the host cores, a lone unit's rows split across
them) and raises before any fit when a unit lacks a class or holds a code
outside 0 .. classes-1: nothing is published and the caller's own loop runs,
whose doors decide as they always did.

WHO REFUSES A NON-FINITE ROW. A member kind with `MEMBER_CAN_TEST_ROWS`
refuses non-finite X, y and weights itself, from the pass its kernel already
makes: the caller makes NO host scan for it (`member_set_tests`,
`MemberHeldCall.tests`); its refusal fails the set or the call, the caller's
door runs and raises the estimator's own words. A kind without the bit reads
admitted words: the caller proves the rows finite first (one host scan per
distinct block) and takes its doors when they are not.

THE RESIDENT SCOPE (`member_set_resident`; THE PIN RULE at core/member_abi.mojo
`MemberCall.pin`). A caller that fits its set inside `resident()` names, per
member, the Python object that owns the member's X words. For a member whose
family reads a scope (`MEMBER_CAN_SCOPE`) the thread's scope is read once per
set; `member_set_run` then takes one reference per such member with the
interpreter held, the member's fit carries the scope word and offers the
reference (never for a block the worker gathered itself), the member's
cache keeps X on the device for the set's other members and for the doors
that follow, and the references no cache took are dropped after the workers
joined. Outside `resident()`, or for a member without the bit, nothing is
read, pinned or offered: the member puts X per fit, as its own door does
there.

THE PREDICT SIDE (`member_held_open` / `member_held_run` /
`member_held_close`): one member call on the
fitted state the estimator holds (bindings/member_held_door.mojo), written
into a new Array of the element type the member's OUTPUTS answers
(`call.out_dtype`); not served = the caller's door.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.sys.compile import is_defined
from std.python._cpython import GILReleased
from bindings.member_boundary_door import member_admit_serving, member_pack, member_pack_labeled
from bindings.member_held_door import MemberHeld, member_held, member_held_release, member_held_adopt
from bindings.admitted_input_unpack import admitted_unfilled
from bindings.native_unit_os_threads_run import native_unit_cores
from bindings.prep_native_estimator_door import prep_estimator_plan
from bindings.prep_native_estimator_door import prep_native_context
from bindings.member_scope_door import member_scope_word, member_pin_take, member_pin_return
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from core.device_route import DEVICE_CPU
from core.host_parallel import host_parallelize
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_call_of, member_new, member_free, member_outputs_of,
    member_kind_capabilities,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_FIT, MEMBER_OP_PREDICT, MEMBER_OP_PREDICT_PROBA, MEMBER_OP_DECISION,
    MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_I64, MEMBER_DTYPE_UNSAID,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_DECISION,
    MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_CLASSIFIER,
    MEMBER_CAN_ROW_INDEX, MEMBER_CAN_COLUMN_INDEX, MEMBER_CAN_TEST_ROWS,
    MEMBER_CAN_SCOPE, MEMBER_SCOPE_NONE, member_pin_settle,
)
from core.member_families import MEMBER_FAMILY_TREE, MEMBER_FAMILY_FOREST, MEMBER_FAMILY_EXTRA_TREES

#: Default ON. `-D MOJOLEARN_MEMBER_SET_OFF` makes `member_set_admit` answer
#: False for every member, so every caller keeps its own member loop: the B
#: arm of the owed A/B (docs/apple-fast/EXPERIMENTS.md). Not measured yet;
#: the callers' loops stay until it is, and for unpublished families.
comptime MEMBER_SET_NATIVE = not is_defined["MOJOLEARN_MEMBER_SET_OFF"]()

#: Words below which a gather is not split (a task costs more than the
#: copy); a cost rule over any shape.
comptime _SET_SPLIT_WORDS = 1048576


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
    # The class-cover pass of `member_set_run` (module note): the units whose
    # codes are read (one per distinct target and draw) and, per member, 1
    # when its codes lack a class or hold a code out of range.
    var cover_units: List[Int]
    var uncovered: List[UInt8]
    # Per member: 1 when its KIND has `MEMBER_CAN_TEST_ROWS` (its entry
    # refuses non-finite X, y and weights itself: the caller scans nothing
    # for it). Read once per member class at admission.
    var tests: List[UInt8]
    # THE RESIDENT SCOPE (module note): the caller's scope handle word
    # (MEMBER_SCOPE_NONE: none) and, per member, the pin token its fit offers
    # (0: none; the worker writes 0 when the member's cache took it).
    var scope: Int
    var pin_tokens: List[Int]
    # THE LIFETIME RULE (MEMBER-ABI-V2.md item 34): every buffer handed to a
    # member's FIT stays valid until its state is freed or has been exported.
    # Every state of a set waits for PACK, so the drawn block a worker gathers
    # for a member whose KIND borrows its fit buffers (`borrows[u]`,
    # MEMBER_BORROWS_FIT) is the RUN's: `part_x[u]`, `part_y[u]`, `part_w[u]`,
    # emptied by `member_set_pack` once that state is freed or owns its rows
    # (and by `member_set_release`). A kind that owns its state gathers into
    # the worker's own buffers, freed when its fit returns, as before.
    var borrows: List[UInt8]
    var part_x: List[List[Float32]]
    var part_y: List[List[Int32]]
    var part_w: List[List[Float32]]

    def __init__(out self, x: Int, rows: Int, cols: Int, y_dtype: Int, index_dtype: Int):
        self.plans = List[MemberPlan]()
        self.x = x
        self.rows = rows
        self.cols = cols
        self.targets = List[Int]()
        self.weights = List[Int]()
        self.y_dtype = y_dtype
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
        self.borrows = List[UInt8]()
        self.part_x = List[List[Float32]]()
        self.part_y = List[List[Int32]]()
        self.part_w = List[List[Float32]]()

    def part_free(mut self, u: Int):
        """Member `u`'s held gathered block is no longer read (its state was
        freed, or owns its rows)."""
        if len(self.part_x[u]) > 0 or len(self.part_y[u]) > 0:
            self.part_x[u] = List[Float32]()
            self.part_y[u] = List[Int32]()
            self.part_w[u] = List[Float32]()


struct MemberSet(Movable):
    """A caller's hold on its member set: the typed run plus, for UNPACK and
    PACK only, each member's Python estimator, its family context and the
    pins that keep its images loaded. Keep it alive past `member_set_pack`."""
    var run: MemberSetRun
    var estimators: PythonObject
    var contexts: PythonObject
    var fields: PythonObject
    var pins: PythonObject
    # Each member's import-time class plan (the held state's key at PACK).
    var class_plans: PythonObject
    # PACK leaves each fitted state with its estimator as its held member
    # state. Set by a caller whose predict side reads held states; else the
    # states are freed at PACK (a second copy of the model nobody reads).
    var adopt: Bool
    # With `adopt`: also the states of tree, forest and extra-trees members
    # whose route is the CPU. Set by a caller that predicts those through
    # their held states when no device copy of X is staged for them (bagging).
    var adopt_trees: Bool
    # After PACK: every packed member's state stayed with its estimator (a
    # later `member_held_open` of it is a hit: no admission, no IMPORT).
    var kept_all: Bool
    # THE RESIDENT SCOPE: per member, the object that owns its X words (None:
    # nothing is offered), and whether the thread's scope was read.
    var sources: PythonObject
    var scope_read: Bool

    def __init__(out self, x: Int, rows: Int, cols: Int, y_dtype: Int, index_dtype: Int) raises:
        self.run = MemberSetRun(x, rows, cols, y_dtype, index_dtype)
        self.estimators = Python.list()
        self.contexts = Python.list()
        self.fields = Python.list()
        self.pins = Python.list()
        self.class_plans = Python.list()
        self.adopt = False
        self.adopt_trees = False
        self.kept_all = True
        self.sources = Python.list()
        self.scope_read = False

    def count(self) -> Int:
        return len(self.run.plans)


def member_set_admit(mut set: MemberSet, estimator: PythonObject, target: Int, weights: Int, row_draw: Int,
    row_count: Int, column_draw: Int, column_count: Int, classes: Int, context: PythonObject,
    b: PythonObject) raises -> Bool:
    """UNPACK: add `estimator` (an unfitted clone the caller made; PACK writes
    its fitted state into it) to the set, or answer False having added
    nothing when it is not a native member this set can fit: its class has no
    plan or no member mark, its family's admit keeps the estimator's own
    doors or refuses (`member_admit_serving`: never raises, the fallback
    rule), it has no fit, it is given weights it does not read, or the
    targets are class codes and its family has no classifier.
    `context` is the caller's context (it holds the model-selection plans),
    `b` the builtins module. `classes`: the class count of code targets, 0
    for a regressor."""
    comptime if not MEMBER_SET_NATIVE:
        return False
    var plan = prep_estimator_plan(estimator, context)
    if plan is None:
        return False
    if py_get(plan, "member") is None:
        return False
    var member_context = plan["context"]
    var fields = py_call(plan["fields_get"], estimator, py_type(estimator))
    # The family's union mask: a family of regressors and classifiers answers
    # the kind-exact refusal itself at NEW / FIT, and the caller then runs
    # its own loop (a failing `member_set_run` publishes nothing).
    var needs = MEMBER_CAN_FIT
    if weights != 0:
        needs = needs | MEMBER_CAN_SAMPLE_WEIGHT
    if classes > 0:
        needs = needs | MEMBER_CAN_CLASSIFIER
    var owner = member_admit_serving(estimator, fields, plan, member_context, b, needs)
    if not owner.published:
        return False
    owner.plan.classes = classes
    # Whether this member's KIND tests the rows it reads: the answer of an
    # earlier member of the same family, kind and route (clones of one
    # estimator: the member before it), else asked once. A family whose union
    # mask lacks the bit is never asked (no state is minted for the answer).
    var tests = UInt8(0)
    if owner.plan.member.can(MEMBER_CAN_TEST_ROWS):
        var asked = False
        var v = len(set.run.plans) - 1
        while v >= 0:  # driver-loop: over the members admitted before (one compare when they are clones)
            if (set.run.plans[v].family == owner.plan.family and set.run.plans[v].kind == owner.plan.kind
                    and set.run.plans[v].route.device == owner.plan.route.device
                    and set.run.plans[v].route.vendor == owner.plan.route.vendor
                    and set.run.plans[v].route.mode == owner.plan.route.mode):
                tests = set.run.tests[v]
                asked = True
                break
            v -= 1
        if not asked:
            try:
                if (member_kind_capabilities(owner.plan) & MEMBER_CAN_TEST_ROWS) != 0:
                    tests = UInt8(1)
            except:
                tests = UInt8(0)
    set.run.tests.append(tests)
    set.run.plans.append(owner.plan.copy())
    set.run.targets.append(target)
    set.run.weights.append(weights)
    set.run.picks.append(row_draw)
    set.run.picks.append(row_count)
    set.run.picks.append(column_draw)
    set.run.picks.append(column_count)
    set.run.own_x.append(0)
    set.run.own_rows.append(0)
    set.run.states.append(0)
    set.run.failed.append(UInt8(0))
    set.run.errors.append(String(""))
    set.run.uncovered.append(UInt8(0))
    set.run.pin_tokens.append(0)
    set.run.borrows.append(UInt8(1))
    set.run.part_x.append(List[Float32]())
    set.run.part_y.append(List[Int32]())
    set.run.part_w.append(List[Float32]())
    _ = py_call(set.sources.append, PythonObject(None))
    _ = py_call(set.estimators.append, estimator)
    _ = py_call(set.contexts.append, member_context)
    _ = py_call(set.fields.append, fields)
    _ = py_call(set.pins.append, owner.pins)
    _ = py_call(set.class_plans.append, plan)
    _ = owner.published
    return True


def member_set_own_block(mut set: MemberSet, x: Int, rows: Int):
    """UNPACK: the member admitted last reads its own float32 C-order X
    block (`rows` x the set's columns, at address `x`; its target and
    weights addresses then hold `rows` values) and not the set's X: a cv
    fold's gathered rows, shared read-only by that fold's members. THE
    LIFETIME RULE: the caller keeps the block (and every target, weight and
    draw address it gave `member_set_admit`) alive until the LAST
    `member_set_pack` / `member_set_release` of the set has returned: a
    member that borrows its fit buffers reads them again at its export."""
    var last = len(set.run.own_x) - 1
    if last < 0:
        return
    set.run.own_x[last] = x
    set.run.own_rows[last] = rows


def member_set_resident(mut set: MemberSet, source: PythonObject, context: PythonObject) raises:
    """UNPACK, after the admit (and `member_set_own_block`) of the member
    admitted last: `source` is the Python object whose buffer holds that
    member's X words (the set's X, or its own block). Inside the calling
    thread's `resident()` scope a member whose family reads a scope then
    keeps that X on the device across the set's fits (module note, THE
    RESIDENT SCOPE). Nothing is read or recorded for a member without
    `MEMBER_CAN_SCOPE`; the scope is read once per set, at the first member
    that has it. `context` is the caller's context, as `member_set_admit`
    takes it. Raises what the scope's own doors raise for a thread-local
    that holds something else: the caller then keeps its own loop, whose
    doors refuse in their own words."""
    var last = len(set.run.plans) - 1
    if last < 0 or not set.run.plans[last].member.can(MEMBER_CAN_SCOPE):
        return
    if not set.scope_read:
        var rows_context = prep_native_context(context)["family_msel_rows_context"]
        set.run.scope = member_scope_word(rows_context)
        set.scope_read = True
    if set.run.scope < 0:
        return
    py_setitem(set.sources, PythonObject(last), source)


def member_set_tests(set: MemberSet, u: Int) -> Bool:
    """Member `u`'s kind refuses non-finite X, y and weights itself
    (`MEMBER_CAN_TEST_ROWS`): the caller makes no finite scan of what that
    member reads. False: the member reads admitted words and the caller
    proves them finite before `member_set_run`."""
    return u >= 0 and u < len(set.run.tests) and set.run.tests[u] != UInt8(0)


def member_set_host(set: MemberSet) -> Bool:
    """Every member admitted so far runs on the CPU route."""
    for u in range(len(set.run.plans)):  # driver-loop: over the members (their resolved routes)
        if set.run.plans[u].route.device != DEVICE_CPU:
            return False
    return True


def _set_cores(set: MemberSet) -> Int:
    """The host cores, bounded by the caller's worker bound when it gave one."""
    var cores = native_unit_cores()
    if set.run.cap > 0:
        cores = min(cores, set.run.cap)
    return max(cores, 1)


def member_set_workers(set: MemberSet) -> Int:
    """The host cores (bounded by the caller's `cap`) bounded by the member
    count when every member's route is the CPU; one when any member runs on
    a GPU (module note)."""
    if not member_set_host(set):
        return 1
    return max(min(_set_cores(set), len(set.run.plans)), 1)


def member_set_width(set: MemberSet) -> Int:
    """How many members a caller that fits in waves puts in one set: the
    host cores when every member admitted so far runs on the CPU, else one.
    A wave bounds what is held at once (the draws and the gathered blocks of
    one wave) to the worker count."""
    if not member_set_host(set):
        return 1
    return _set_cores(set)


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
    for i in range(first, first + count):
        var r = _set_pick(row_draw, i, wide) if row_count > 0 else i
        if column_count > 0:
            for j in range(width):
                target[i * width + j] = source[r * cols + _set_pick(column_draw, j, wide)]
        else:
            unsafe_memcpy(dest=target + i * width, src=source + r * cols, count=cols)
        # Targets are 4-byte words of either element type: copied as bits.
        y_target[i] = y_source[r]
        if has_w:
            MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dst_w)[i] = MutPointer[
                Float32, MutUntrackedOrigin](unsafe_from_address=run[].weights[u])[r]


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
    0 .. classes-1 and no other (module note). `split`: this is the only
    unit being read, so its rows are split across the host cores."""
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


def _member_set_unit(address: Int, u: Int) raises:
    """Fit member `u` in a new state of its own; the state stays in
    `states[u]` for PACK. A failure releases it."""
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
    # THE LIFETIME RULE: a borrowing kind's gathered block is the run's (its
    # state reads it again at PACK); this unit's own slots, no other worker's.
    var hold = run[].borrows[u] != UInt8(0)
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
        call.y = MemberMatrix(run[].targets[u], rows, 1, run[].y_dtype)
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
                run[].part_y[u] = List[Int32](length=n, fill=Int32(0))
                x_at = Int(run[].part_x[u].unsafe_ptr())
                y_at = Int(run[].part_y[u].unsafe_ptr())
                if run[].weights[u] != 0:
                    run[].part_w[u] = List[Float32](length=n, fill=0.0)
                    w_address = Int(run[].part_w[u].unsafe_ptr())
            else:
                x_part = List[Float32](length=n * width, fill=0.0)
                y_part = List[Int32](length=n, fill=Int32(0))
                x_at = Int(x_part.unsafe_ptr())
                y_at = Int(y_part.unsafe_ptr())
                if run[].weights[u] != 0:
                    w_part = List[Float32](length=n, fill=0.0)
                    w_address = Int(w_part.unsafe_ptr())
            _set_gather(address, u, n, width, x_at, y_at, w_address, run[].workers <= 1)
            call.x = MemberMatrix(x_at, n, width)
            call.y = MemberMatrix(y_at, n, 1, run[].y_dtype)
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
        done = True
    finally:
        if not done:
            member_free(run[].plans[u], call)
            run[].part_free(u)
        # Integer addresses carry no lifetime: the gathered block lives to here.
        _ = len(x_part)
        _ = len(y_part)
        _ = len(w_part)


def _member_set_lane(address: Int, lane: Int):
    """One worker: members lane, lane + workers, ... A failing member
    records its words in its own slot; the others still run (each is
    independent) and the caller raises the first in member order."""
    var run = MutPointer[MemberSetRun, MutUntrackedOrigin](unsafe_from_address=address)
    var count = len(run[].plans)
    var u = lane
    while u < count:  # driver-loop: over this worker's members
        try:
            _member_set_unit(address, u)
        except error:
            run[].errors[u] = String(error)
            run[].failed[u] = UInt8(1)
        u += run[].workers


def member_set_release(mut set: MemberSet):
    """Release every state the set still holds. Never raises."""
    for u in range(len(set.run.plans)):  # driver-loop: over the members (their states)
        if set.run.states[u] != 0:
            var call = MemberCall(set.run.plans[u].route, set.run.states[u])
            member_free(set.run.plans[u], call)
            set.run.states[u] = 0
        # After the state: nothing reads the held block any more.
        set.run.part_free(u)


def member_set_run(mut set: MemberSet) raises:
    """RUN: fit every member of the set, the GIL released, on
    `member_set_workers` host tasks. After it every member holds a fitted
    state for `member_set_pack`; on a failure every state is released and
    the first failing member's words (in member order) are raised.

    Classifier sets are read first (module note: the class-cover pass): when
    a member's own rows lack a class, or hold a code out of range, nothing
    is fitted and the set raises, so the caller's own loop and its doors
    decide. One pass over each distinct target's int32 codes, against the
    label encode every member's fit door ran on its rows."""
    var count = len(set.run.plans)
    if count == 0:
        return
    set.run.workers = member_set_workers(set)
    set.run.cover_units.clear()
    if set.run.y_dtype == MEMBER_I32:
        for u in range(count):  # driver-loop: over the members (one cover pass per distinct target)
            if set.run.plans[u].classes <= 0:
                continue
            if u > 0 and _set_same_codes(set.run, u, u - 1):
                continue
            set.run.cover_units.append(u)
    var address = Int(UnsafePointer(to=set.run))
    var lanes = set.run.workers
    # The cover pass is host work whatever the members' route.
    var cover_lanes = max(min(native_unit_cores(), len(set.run.cover_units)), 1)

    def _cover(lane: Int) {imm address, imm cover_lanes}:
        _member_set_cover_lane(address, lane, cover_lanes)

    def _lane(lane: Int) {imm address}:
        _member_set_lane(address, lane)

    var lacking = False
    var py = Python()
    # THE PIN RULE: one reference per member that offers its X object, taken
    # with the interpreter held; what no cache took is dropped below.
    for u in range(count):  # driver-loop: over the members (one reference each)
        set.run.pin_tokens[u] = 0
        var source = set.sources[u]
        if source is not None:
            set.run.pin_tokens[u] = member_pin_take(source, 1)
    try:
        with GILReleased(py):
            if len(set.run.cover_units) > 0:
                host_parallelize(_cover, cover_lanes)
                for u in range(count):  # driver-loop: over the members (the cover verdicts)
                    if set.run.uncovered[u] != UInt8(0):
                        lacking = True
            if not lacking:
                host_parallelize(_lane, lanes)
    finally:
        for u in range(count):  # driver-loop: over the members (the references no cache took)
            if set.run.pin_tokens[u] != 0:
                member_pin_return(set.run.pin_tokens[u], 1)
                set.run.pin_tokens[u] = 0
        _ = set.pins
    if lacking:
        raise Error("ValueError: a member's rows do not hold every class of the ensemble: its own fit decides")
    for u in range(count):  # driver-loop: over the members, for the first failure in member order
        if set.run.failed[u] != UInt8(0):
            var words = set.run.errors[u].copy()
            member_set_release(set)
            raise Error(words)


def member_set_pack(mut set: MemberSet, u: Int, classes: PythonObject, b: PythonObject) raises -> PythonObject:
    """PACK: member `u`'s fitted estimator (the clone given at admission,
    now holding the attributes its own fit writes); its state is released.
    `classes`: the labels the member's codes stand for (code c is
    `classes[c]`; a caller whose members see the ensemble's codes passes the
    codes themselves), or None for a regressor. Labels never cross the ABI:
    the family's pack takes them (`member_pack_labeled`) and writes its own
    class attributes, as its fit door does."""
    var estimator = set.estimators[u]
    var member_context = set.contexts[u]
    var fields = set.fields[u]
    try:
        if classes is None:
            member_pack(estimator, fields, member_context, b, set.run.plans[u], set.run.states[u])
        else:
            member_pack_labeled(estimator, fields, member_context, b, set.run.plans[u], set.run.states[u], classes)
    except error:
        var failed = MemberCall(set.run.plans[u].route, set.run.states[u])
        member_free(set.run.plans[u], failed)
        set.run.states[u] = 0
        set.run.part_free(u)
        raise error^
    # `adopt`: the fitted state stays with the estimator as its held member
    # state (bindings/member_held_door.mojo), so the caller's predict side
    # reads it without an import. Not kept: freed here, as before.
    var kept = False
    var family = set.run.plans[u].family
    # A tree, forest or extra-trees member is predicted by its own door (from
    # the one device copy of X the caller stages): its state would be a
    # second copy of the model nobody reads, except where the caller says it
    # reads them (`adopt_trees`, CPU routes: nothing is staged there).
    var read = (not (family == MEMBER_FAMILY_TREE or family == MEMBER_FAMILY_FOREST
        or family == MEMBER_FAMILY_EXTRA_TREES)
        or (set.adopt_trees and set.run.plans[u].route.device == DEVICE_CPU))
    if set.adopt and read:
        try:
            kept = member_held_adopt(estimator, set.class_plans[u], member_context, set.pins[u], set.run.plans[u],
                                     set.run.states[u])
        except:
            kept = False
    if not kept:
        set.kept_all = False
        var call = MemberCall(set.run.plans[u].route, set.run.states[u])
        member_free(set.run.plans[u], call)
    set.run.states[u] = 0
    # THE LIFETIME RULE: the state is freed, or adopted (`member_held_adopt`
    # made a borrowing kind own its rows first): the held block goes now.
    set.run.part_free(u)
    return estimator


def member_marked(estimator: PythonObject, context: PythonObject) -> Bool:
    """True when `estimator`'s class carries a member mark (its family may
    serve it; `member_held_open` still answers per call). One class-plan
    read: a caller asks once per public operation for members of one class,
    before work only the native route needs (its finite proof of X). Never
    raises."""
    comptime if not MEMBER_SET_NATIVE:
        return False
    try:
        var plan = prep_estimator_plan(estimator, context)
        if plan is None:
            return False
        return py_get(plan, "member") is not None
    except:
        return False


struct MemberHeldCall(Movable):
    """One public operation's opened call on a fitted member's held state
    (`member_held_open`). `served` False: nothing is held, the caller's door
    runs. `tests`: the member's kind refuses non-finite rows itself
    (`MEMBER_CAN_TEST_ROWS`): the caller scans nothing; False: the caller
    proves the rows finite before `member_held_run`, or closes and takes its
    door. `cols` / `dtype`: the columns per row and the element type
    (MEMBER_F32 / F64 / I32 / I64) the member writes for `operation`.
    `host`: the member's route is the CPU."""
    var held: MemberHeld
    var served: Bool
    var tests: Bool
    var host: Bool
    var operation: Int
    var cols: Int
    var dtype: Int

    def __init__(out self):
        self.held = MemberHeld()
        self.served = False
        self.tests = False
        self.host = False
        self.operation = MEMBER_OP_PREDICT
        self.cols = 0
        self.dtype = MEMBER_F32


def _out_typestr(dtype: Int) -> String:
    if dtype == MEMBER_F64:
        return "<f8"
    if dtype == MEMBER_I32:
        return "<i4"
    if dtype == MEMBER_I64:
        return "<i8"
    return "<f4"


def _member_held_open(mut opened: MemberHeldCall, member: PythonObject, operation: Int, classifier: Bool,
    width: Int, skip_trees: Bool, context: PythonObject, b: PythonObject) raises:
    var plan = prep_estimator_plan(member, context)
    if plan is None:
        return
    var mark = py_get(plan, "member")
    if mark is None or Bool(py=plan["classifier"]) != classifier:
        return
    if skip_trees:
        var family = Int(py=mark[0])
        if family == MEMBER_FAMILY_TREE or family == MEMBER_FAMILY_FOREST or family == MEMBER_FAMILY_EXTRA_TREES:
            return
    var needs = MEMBER_CAN_PREDICT
    if operation == MEMBER_OP_PREDICT_PROBA:
        needs = MEMBER_CAN_PREDICT_PROBA
    elif operation == MEMBER_OP_DECISION:
        needs = MEMBER_CAN_DECISION
    var member_context = plan["context"]
    var fields = py_call(plan["fields_get"], member, py_type(member))
    opened.held = member_held(member, fields, plan, member_context, b, needs)
    if not opened.held.published:
        return
    opened.operation = operation
    var call = member_call_of(opened.held.plan, opened.held.state)
    var cols = member_outputs_of(opened.held.plan, call, operation)
    if cols < 1 or (width != 0 and cols != width):
        return
    # The element type is the member's own answer (`call.out_dtype`, written
    # by the OUTPUTS call above): codes int32 or int64, scores and
    # probabilities float32 or float64. A member that does not say it
    # (published before the word existed): a regressor writes float32 (every
    # published regressor documents it); a classifier is not served and its
    # door runs. Nothing is learned from a refused call.
    var dtype = call.out_dtype
    var codes = classifier and operation == MEMBER_OP_PREDICT
    if dtype == MEMBER_DTYPE_UNSAID:
        if classifier:
            return
        dtype = MEMBER_F32
    if codes:
        if dtype != MEMBER_I32 and dtype != MEMBER_I64:
            return
    elif dtype != MEMBER_F32 and dtype != MEMBER_F64:
        return
    # Kind-exact, asked of the held state (no state is minted); a family
    # whose union mask lacks the bit is not asked.
    if opened.held.plan.member.can(MEMBER_CAN_TEST_ROWS):
        opened.tests = (member_call(opened.held.plan.member, MEMBER_OP_CAPABILITIES, call) & MEMBER_CAN_TEST_ROWS) != 0
    opened.host = opened.held.plan.route.device == DEVICE_CPU
    opened.cols = cols
    opened.dtype = dtype
    opened.served = True


def member_held_open(member: PythonObject, operation: Int, classifier: Bool, width: Int, skip_trees: Bool,
    context: PythonObject, b: PythonObject) -> MemberHeldCall:
    """PREDICT SIDE, UNPACK: open the fitted estimator `member`'s `operation`
    (MEMBER_OP_PREDICT, _PREDICT_PROBA or _DECISION) on the member state it
    holds (bindings/member_held_door.mojo: a hit after the first use, and the
    state a native fit left with it). Once per member per public operation.
    The answer says whether the member is served, how wide and of which
    element type its output is, and whether its kind tests the rows itself
    (`tests`): the caller makes its finite scan of X only when it does not.
    Close it with `member_held_close` on every path.

    `served` False, with nothing changed, when: the class has no member
    mark, it is not of the asked kind (`classifier`), its output has not
    `width` columns (0: any), its family lacks the operation, admits or
    imports nothing, or a classifier's member does not say its element type.
    The caller then runs the member's own door, which raises the estimator's
    own words (the fallback rule). Never raises. `skip_trees`: a tree,
    forest or extra-trees member is not served (the caller's door predicts
    those from the one device copy of X it staged for all of them)."""
    var opened = MemberHeldCall()
    comptime if not MEMBER_SET_NATIVE:
        return opened^
    try:
        _member_held_open(opened, member, operation, classifier, width, skip_trees, context, b)
    except:
        opened.served = False
    return opened^


def member_held_run(mut opened: MemberHeldCall, x: Int, n: Int, d: Int, context: PythonObject) -> PythonObject:
    """PREDICT SIDE, RUN: the opened member's operation over the `n` x `d`
    float32 C-order host rows at `x`: ONE member call, the GIL released,
    written into a new Array of the member's element type, (n,) for one
    column per row, else (n, columns). A classifier's predict writes its
    class CODES (code c is the member's c-th class); the caller maps them.
    None when the call refuses (a member that tests its rows refuses a
    non-finite one here): the caller's door runs and raises the estimator's
    own words. Never raises. THE CALLER keeps the rows alive through the
    call and, for a member that does not test them (`tests` False), has
    proven them finite. Serial by nature: one state, one call at a time (the
    member splits its own rows)."""
    if not opened.served:
        return PythonObject(None)
    try:
        var call = member_call_of(opened.held.plan, opened.held.state)
        var shape = List[Int](length=1, fill=n)
        if opened.cols > 1:
            shape.append(opened.cols)
        # Unfilled: the member writes every cell before anything reads it.
        var dst = admitted_unfilled(shape, _out_typestr(opened.dtype), context)
        call.x = MemberMatrix(x, n, d)
        call.dst = MemberMatrix(dst.addr, n, opened.cols, opened.dtype)
        var plan_member = opened.held.plan.member
        var operation = opened.operation
        with GILReleased(Python()):
            _ = member_call(plan_member, operation, call)
        _ = opened.served   # the hold (its plan words, its owner pin) outlives the call
        return dst.keep
    except:
        return PythonObject(None)


def member_held_close(mut opened: MemberHeldCall):
    """End of the opened call, every path, after `member_held_run` returned:
    a kept state stays with the estimator. Never raises."""
    member_held_release(opened.held)
    opened.served = False

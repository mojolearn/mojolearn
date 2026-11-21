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
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.member_boundary_door import member_admit, member_pack
from bindings.native_unit_os_threads_run import native_unit_cores
from bindings.prep_native_estimator_door import prep_estimator_plan
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from core.device_route import DEVICE_CPU
from core.host_parallel import host_parallelize
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_FIT, MEMBER_F32, MEMBER_I32, MEMBER_I64,
    MEMBER_CAN_FIT, MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_CLASSIFIER,
    MEMBER_CAN_ROW_INDEX, MEMBER_CAN_COLUMN_INDEX,
)

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
    var workers: Int
    var states: List[Int]
    var failed: List[UInt8]
    var errors: List[String]

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
        self.workers = 1
        self.states = List[Int]()
        self.failed = List[UInt8]()
        self.errors = List[String]()


struct MemberSet(Movable):
    """A caller's hold on its member set: the typed run plus, for UNPACK and
    PACK only, each member's Python estimator, its family context and the
    pins that keep its images loaded. Keep it alive past `member_set_pack`."""
    var run: MemberSetRun
    var estimators: PythonObject
    var contexts: PythonObject
    var fields: PythonObject
    var pins: PythonObject

    def __init__(out self, x: Int, rows: Int, cols: Int, y_dtype: Int, index_dtype: Int) raises:
        self.run = MemberSetRun(x, rows, cols, y_dtype, index_dtype)
        self.estimators = Python.list()
        self.contexts = Python.list()
        self.fields = Python.list()
        self.pins = Python.list()

    def count(self) -> Int:
        return len(self.run.plans)


def member_set_admit(mut set: MemberSet, estimator: PythonObject, target: Int, weights: Int, row_draw: Int,
    row_count: Int, column_draw: Int, column_count: Int, classes: Int, context: PythonObject,
    b: PythonObject) raises -> Bool:
    """UNPACK: add `estimator` (an unfitted clone the caller made; PACK writes
    its fitted state into it) to the set, or answer False having added
    nothing when it is not a native member this set can fit: its class has no
    plan or no member mark, its family's admit keeps the estimator's own
    doors, it has no fit, it is given weights it does not read, or the
    targets are class codes and it is not a classifier (and the reverse).
    `context` is the caller's context (it holds the model-selection plans),
    `b` the builtins module. `classes`: the class count of code targets, 0
    for a regressor."""
    var plan = prep_estimator_plan(estimator, context)
    if plan is None:
        return False
    if py_get(plan, "member") is None:
        return False
    var member_context = plan["context"]
    var fields = py_call(plan["fields_get"], estimator, py_type(estimator))
    var owner = member_admit(estimator, fields, plan, member_context, b)
    if not owner.published:
        return False
    if not owner.plan.member.can(MEMBER_CAN_FIT):
        return False
    if weights != 0 and not owner.plan.member.can(MEMBER_CAN_SAMPLE_WEIGHT):
        return False
    # The family's union mask: a family of regressors and classifiers answers
    # the kind-exact refusal itself at NEW / FIT.
    if classes > 0 and not owner.plan.member.can(MEMBER_CAN_CLASSIFIER):
        return False
    owner.plan.classes = classes
    set.run.plans.append(owner.plan.copy())
    set.run.targets.append(target)
    set.run.weights.append(weights)
    set.run.picks.append(row_draw)
    set.run.picks.append(row_count)
    set.run.picks.append(column_draw)
    set.run.picks.append(column_count)
    set.run.states.append(0)
    set.run.failed.append(UInt8(0))
    set.run.errors.append(String(""))
    _ = py_call(set.estimators.append, estimator)
    _ = py_call(set.contexts.append, member_context)
    _ = py_call(set.fields.append, fields)
    _ = py_call(set.pins.append, owner.pins)
    _ = owner.published
    return True


def member_set_workers(set: MemberSet) -> Int:
    """The host cores bounded by the member count when every member's route
    is the CPU; one when any member runs on a GPU (module note)."""
    var count = len(set.run.plans)
    for u in range(count):  # driver-loop: over the members (their resolved routes)
        if set.run.plans[u].route.device != DEVICE_CPU:
            return 1
    return max(min(native_unit_cores(), count), 1)


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


def _member_set_unit(address: Int, u: Int) raises:
    """Fit member `u` in a new state of its own; the state stays in
    `states[u]` for PACK. A failure releases it."""
    var run = MutPointer[MemberSetRun, MutUntrackedOrigin](unsafe_from_address=address)
    var rows = run[].rows
    var cols = run[].cols
    var row_draw = run[].picks[4 * u]
    var row_count = run[].picks[4 * u + 1]
    var column_draw = run[].picks[4 * u + 2]
    var column_count = run[].picks[4 * u + 3]
    var drawn = row_count > 0 or column_count > 0
    var member = run[].plans[u].member
    var by_index = (drawn and (row_count == 0 or member.can(MEMBER_CAN_ROW_INDEX))
        and (column_count == 0 or member.can(MEMBER_CAN_COLUMN_INDEX)))
    var x_part = List[Float32]()
    var y_part = List[Int32]()
    var w_part = List[Float32]()
    var call = member_new(run[].plans[u])
    var done = False
    try:
        call.x = MemberMatrix(run[].x, rows, cols)
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
            x_part = List[Float32](length=n * width, fill=0.0)
            y_part = List[Int32](length=n, fill=Int32(0))
            var w_address = 0
            if run[].weights[u] != 0:
                w_part = List[Float32](length=n, fill=0.0)
                w_address = Int(w_part.unsafe_ptr())
            _set_gather(address, u, n, width, Int(x_part.unsafe_ptr()), Int(y_part.unsafe_ptr()), w_address,
                run[].workers <= 1)
            call.x = MemberMatrix(Int(x_part.unsafe_ptr()), n, width)
            call.y = MemberMatrix(Int(y_part.unsafe_ptr()), n, 1, run[].y_dtype)
            if w_address != 0:
                call.weights = MemberMatrix(w_address, n, 1)
        _ = member_call(member, MEMBER_OP_FIT, call)
        run[].states[u] = call.state
        done = True
    finally:
        if not done:
            member_free(run[].plans[u], call)
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


def member_set_run(mut set: MemberSet) raises:
    """RUN: fit every member of the set, the GIL released, on
    `member_set_workers` host tasks. After it every member holds a fitted
    state for `member_set_pack`; on a failure every state is released and
    the first failing member's words (in member order) are raised."""
    var count = len(set.run.plans)
    if count == 0:
        return
    set.run.workers = member_set_workers(set)
    var address = Int(UnsafePointer(to=set.run))
    var lanes = set.run.workers

    def _lane(lane: Int) {imm address}:
        _member_set_lane(address, lane)

    var py = Python()
    try:
        with GILReleased(py):
            host_parallelize(_lane, lanes)
    finally:
        _ = set.pins
    for u in range(count):  # driver-loop: over the members, for the first failure in member order
        if set.run.failed[u] != UInt8(0):
            var words = set.run.errors[u].copy()
            member_set_release(set)
            raise Error(words)


def member_set_pack(mut set: MemberSet, u: Int, classes: PythonObject, b: PythonObject) raises -> PythonObject:
    """PACK: member `u`'s fitted estimator (the clone given at admission,
    now holding the attributes its own fit writes); its state is released.
    `classes`: the member's `classes_` (the labels its codes stand for; a
    caller whose members see the ensemble's codes passes the codes
    themselves), or None for a regressor. Labels never cross the ABI: they
    are published here."""
    var estimator = set.estimators[u]
    var member_context = set.contexts[u]
    var fields = set.fields[u]
    try:
        member_pack(estimator, fields, member_context, b, set.run.plans[u], set.run.states[u])
    finally:
        var call = MemberCall(set.run.plans[u].route, set.run.states[u])
        member_free(set.run.plans[u], call)
        set.run.states[u] = 0
    if classes is not None:
        py_setitem(fields, "classes_", classes)
        py_setitem(fields, "n_classes_", PythonObject(len(classes)))
    return estimator

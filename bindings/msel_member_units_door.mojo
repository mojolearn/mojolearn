# SPDX-License-Identifier: Apache-2.0
"""Model selection's units over NATIVE MEMBERS (core/member_abi.mojo): the
(candidate, fold) units of cross_validate, cross_val_score and the searches,
run without a Python estimator when the member's family has published its
member entry.

WHAT THIS REPLACES for such a member. The unit of bindings/msel_cv_door.mojo
(`MselUnitRun.unit`) clones the Python estimator once per unit, sets the
candidate's parameters on the clone, takes the fold's rows as Python arrays
and calls the member's fit and score doors through the Python model, every
step with the GIL held. Here:
  * UNPACK (GIL held, once per CANDIDATE, never per unit): the candidate's
    estimator (the base itself for a plain cross-validation: it is never
    fitted, so it needs no clone) is admitted as a member
    (`member_admit`: route, parameter words, resolved entry); X, y and the
    fold indices are read once as addresses.
  * RUN (GIL released): the units run on `workers` host tasks
    (`host_parallelize` over lanes; a lane takes units lane, lane + workers,
    ...). Each unit gathers its fold's rows into its own buffers, makes its
    own member state (`member_new`), fits, scores and frees it. With one
    worker (a GPU member: its units share one device) the fold gather itself
    is split across the host cores.
  * Results go to typed slots per unit; the first failing unit in unit order
    is raised after the join, as the serial order would raise it.

WHEN IT RUNS (`msel_member_units` answers `ran == False` otherwise and the
caller keeps its Python units): the member's class carries a member mark and
its family's admit fills it; default scoring (the estimator's own score);
`error_score='raise'`; no fitted estimator asked back; a host float32
C-order X with a target; the member answers fit and score and reads dense
host views. Device fold sets, the slice route and learning_curve's test
reuse keep their own route (the caller tests those before calling).

Memory per worker: one fold's train and test rows (at most one copy of X and
y), as a Python unit holds. The worker count is the caller's
(`msel_unit_workers`: the n_jobs bound, the units, one on a GPU).
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.classical_validate_unpack import classical_check_y, classical_estimator_name
from bindings.member_boundary_door import member_admit
from bindings.msel_common_door import msel_is_array, msel_addr, msel_index_i64
from bindings.msel_cv_run import msel_seconds
from bindings.msel_estimator_native_door import (
    msel_estimator_plan, msel_native_clone, msel_native_set_params, msel_clone_parameter,
)
from bindings.native_unit_os_threads_run import native_unit_cores
from bindings.py_balanced import py_call, py_type, py_get
from core.host_parallel import host_parallelize
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_FIT, MEMBER_OP_SCORE, MEMBER_CAN_FIT, MEMBER_CAN_SCORE,
)

#: Rows below which a fold gather is not split (a task costs more than the
#: copy); a cost rule over any shape.
comptime _GATHER_SPLIT_WORDS = 1048576


struct MselMemberScores(Movable):
    """Per-unit results of a native member run (unit = candidate * folds +
    fold). `ran` False: nothing ran and the lists are empty."""
    var ran: Bool
    var test: List[Float64]
    var train: List[Float64]
    var fit_time: List[Float64]
    var score_time: List[Float64]

    def __init__(out self):
        self.ran = False
        self.test = List[Float64]()
        self.train = List[Float64]()
        self.fit_time = List[Float64]()
        self.score_time = List[Float64]()


struct MselMemberRun(Movable):
    """Everything typed the units read. Built by UNPACK; workers read it
    through one address and write only their own unit's slots."""
    var plans: List[MemberPlan]
    # Per fold, four words: train index address, train rows, test index
    # address, test rows (int64 row numbers of X).
    var folds: List[Int]
    var x: Int
    var y: Int
    var cols: Int
    var n_folds: Int
    var n_units: Int
    var workers: Int
    var return_train: Bool
    var test: List[Float64]
    var train: List[Float64]
    var fit_time: List[Float64]
    var score_time: List[Float64]
    var failed: List[UInt8]
    var errors: List[String]

    def __init__(out self, x: Int, y: Int, cols: Int, n_folds: Int, n_units: Int, workers: Int, return_train: Bool):
        self.plans = List[MemberPlan]()
        self.folds = List[Int]()
        self.x = x
        self.y = y
        self.cols = cols
        self.n_folds = n_folds
        self.n_units = n_units
        self.workers = workers
        self.return_train = return_train
        self.test = List[Float64](length=n_units, fill=0.0)
        self.train = List[Float64](length=n_units, fill=0.0)
        self.fit_time = List[Float64](length=n_units, fill=0.0)
        self.score_time = List[Float64](length=n_units, fill=0.0)
        self.failed = List[UInt8](length=n_units, fill=UInt8(0))
        self.errors = List[String](length=n_units, fill=String(""))


def _gather_span(src: Int, index: Int, first: Int, count: Int, cols: Int, dst: Int) raises:
    """Rows `index[first .. first + count]` of the float32 matrix at `src`
    (`cols` wide) into rows `first ..` of `dst`. A host fold gather out of
    host words: a device fold set never takes this route."""
    var rows = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=index)
    var source = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=src)
    var target = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dst)
    for i in range(first, first + count):
        unsafe_memcpy(dest=target + i * cols, src=source + Int(rows[i]) * cols, count=cols)


def _gather(src: Int, index: Int, count: Int, cols: Int, dst: Int, split: Bool) raises:
    """One fold side's rows. `split`: this unit is the only one running, so
    the copy itself is split across the host cores (row chunks are
    independent); otherwise the units are the parallel split."""
    var cores = native_unit_cores()
    if not split or cores <= 1 or count * cols < _GATHER_SPLIT_WORDS:
        _gather_span(src, index, 0, count, cols, dst)
        return
    var chunks = min(cores, count)
    var per = (count + chunks - 1) // chunks

    def _chunk(part: Int) {imm src, imm index, imm count, imm cols, imm dst, imm per}:
        var first = part * per
        if first >= count:
            return
        try:
            _gather_span(src, index, first, min(per, count - first), cols, dst)
        except:
            pass

    host_parallelize(_chunk, chunks)


def _msel_member_unit(run: MutPointer[MselMemberRun, MutUntrackedOrigin], u: Int) raises:
    var c = u // run[].n_folds
    var f = u - c * run[].n_folds
    var cols = run[].cols
    var split = run[].workers <= 1
    var train_index = run[].folds[4 * f]
    var train_rows = run[].folds[4 * f + 1]
    var test_index = run[].folds[4 * f + 2]
    var test_rows = run[].folds[4 * f + 3]
    var t0 = msel_seconds()
    var x_train = List[Float32](length=train_rows * cols, fill=0.0)
    var y_train = List[Float32](length=train_rows, fill=0.0)
    var x_test = List[Float32](length=test_rows * cols, fill=0.0)
    var y_test = List[Float32](length=test_rows, fill=0.0)
    _gather(run[].x, train_index, train_rows, cols, Int(x_train.unsafe_ptr()), split)
    _gather(run[].y, train_index, train_rows, 1, Int(y_train.unsafe_ptr()), split)
    _gather(run[].x, test_index, test_rows, cols, Int(x_test.unsafe_ptr()), split)
    _gather(run[].y, test_index, test_rows, 1, Int(y_test.unsafe_ptr()), split)
    var train_x = MemberMatrix(Int(x_train.unsafe_ptr()), train_rows, cols)
    var train_y = MemberMatrix(Int(y_train.unsafe_ptr()), train_rows, 1)
    var call = member_new(run[].plans[c])
    try:
        call.x = train_x
        call.y = train_y
        _ = member_call(run[].plans[c].member, MEMBER_OP_FIT, call)
        var t1 = msel_seconds()
        call.x = MemberMatrix(Int(x_test.unsafe_ptr()), test_rows, cols)
        call.y = MemberMatrix(Int(y_test.unsafe_ptr()), test_rows, 1)
        _ = member_call(run[].plans[c].member, MEMBER_OP_SCORE, call)
        run[].test[u] = call.value
        if run[].return_train:
            call.x = train_x
            call.y = train_y
            _ = member_call(run[].plans[c].member, MEMBER_OP_SCORE, call)
            run[].train[u] = call.value
        run[].fit_time[u] = t1 - t0
        run[].score_time[u] = msel_seconds() - t1
    finally:
        member_free(run[].plans[c], call)
        # Integer addresses carry no lifetime: the fold buffers live to here.
        _ = len(x_train)
        _ = len(y_train)
        _ = len(x_test)
        _ = len(y_test)


def _msel_member_lane(address: Int, lane: Int):
    """One worker: units lane, lane + workers, ... A failing unit records
    its words in its own slot; the other units still run (each is
    independent), and the caller raises the first in unit order."""
    var run = MutPointer[MselMemberRun, MutUntrackedOrigin](unsafe_from_address=address)
    var u = lane
    while u < run[].n_units:  # driver-loop: over this worker's units
        try:
            _msel_member_unit(run, u)
        except error:
            run[].errors[u] = String(error)
            run[].failed[u] = UInt8(1)
        u += run[].workers


def msel_member_units(base: PythonObject, candidates: PythonObject, y: PythonObject, x: PythonObject,
    folds: PythonObject, return_train: Bool, workers_in: Int, context: PythonObject,
    b: PythonObject) raises -> MselMemberScores:
    """Run every (candidate, fold) unit over native members, or answer
    `ran == False` having changed nothing when this member, data or request
    is not one this route runs (module note). `context` is model selection's
    context, `b` the builtins module, `workers_in` the caller's worker bound."""
    var scores = MselMemberScores()
    var plan = msel_estimator_plan(base, context)
    if py_get(plan, "member") is None or y is None:
        return scores^
    if not msel_is_array(x, context) or not msel_is_array(y, context):
        return scores^
    if String(py=x.dtype) != "<f4" or Int(py=x.ndim) != 2 or String(py=x.order) != "C" or Int(py=x.size) == 0:
        return scores^
    var member_context = plan["context"]
    var n_folds = len(folds)
    var n_candidates = 1 if candidates is None else len(candidates)
    var n_units = n_candidates * n_folds
    if n_units == 0:
        return scores^
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    var workers = max(min(workers_in, n_units), 1)
    # Everything that must stay alive (estimators, route owners, images,
    # index arrays, the admitted target) until the units have joined.
    var pins = Python.list()
    var run = MselMemberRun(msel_addr(x), 0, cols, n_folds, n_units, workers, return_train)
    for c in range(n_candidates):  # driver-loop: over candidates (admission with Python objects; once each)
        var est = base
        if candidates is not None:
            est = msel_native_clone(base, context)
            _ = msel_native_set_params(est, msel_clone_parameter(candidates[c], context), context)
        var fields = py_call(plan["fields_get"], est, py_type(est))
        var owner = member_admit(est, fields, plan, member_context, b)
        if not owner.published:
            return scores^
        if not owner.plan.member.can(MEMBER_CAN_FIT) or not owner.plan.member.can(MEMBER_CAN_SCORE):
            return scores^
        run.plans.append(owner.plan.copy())
        _ = py_call(pins.append, est)
        _ = py_call(pins.append, owner.pins)
    # The target once, as the member's own fit door admits it (present,
    # real, one column, finite, as long as X): float32 words.
    var target = classical_check_y(y, rows, classical_estimator_name(base), member_context)
    _ = py_call(pins.append, target)
    run.y = msel_addr(target)
    for f in range(n_folds):  # driver-loop: over folds (their index arrays, read once)
        var pair = folds[f]
        var train = msel_index_i64(pair[0], context)
        var test = msel_index_i64(pair[1], context)
        _ = py_call(pins.append, train)
        _ = py_call(pins.append, test)
        run.folds.append(msel_addr(train))
        run.folds.append(Int(py=train.size))
        run.folds.append(msel_addr(test))
        run.folds.append(Int(py=test.size))
    var address = Int(UnsafePointer(to=run))
    var lanes = workers

    def _lane(lane: Int) {imm address}:
        _msel_member_lane(address, lane)

    var py = Python()
    try:
        with GILReleased(py):
            host_parallelize(_lane, lanes)
    finally:
        _ = pins
        _ = x
        _ = y
    for u in range(n_units):  # driver-loop: over units, for the first failure in unit order
        if run.failed[u] != UInt8(0):
            raise Error(run.errors[u])
    scores.ran = True
    scores.test = run.test.copy()
    scores.train = run.train.copy()
    scores.fit_time = run.fit_time.copy()
    scores.score_time = run.score_time.copy()
    return scores^

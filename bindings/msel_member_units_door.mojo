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
    ...). X is NOT copied per fold: a fold side that is one run of rows (a
    test fold of an unshuffled KFold, a time-series split's sides) is a view
    of X; a scattered side goes to a member that reads row numbers
    (`MEMBER_CAN_ROW_INDEX`) as the whole X plus the fold's index array.
    Only a scattered side of a family without that bit is gathered into the
    unit's own buffers (with one worker the gather itself is split across
    the host cores). Each unit makes its own member state (`member_new`),
    fits, scores and frees it.
  * Results go to typed slots per unit. If any unit fails, this route
    publishes nothing and answers `ran == False`: the Python units then run
    and raise the estimator's own exception.

THE FALLBACK RULE (MEMBER-ABI-V2.md): a member mark never turns a call that
worked into a refusal. A marked class this route cannot serve (a transformer
or KMeans: no score; an admission the family refuses; a target the linear
check refuses; a classifier's labels outside the CLASSIFIER rule below)
answers `ran == False` silently.

CLASSIFIER MEMBERS (MEMBER-ABI-V2.md item 1). The labels are encoded ONCE per
operation, at UNPACK, by model selection's own encoder (`msel_encode`: the one
label encoder every classifier's fit door calls, so code c is the estimator's
own `classes_[c]`): y reaches the member as int32 codes (`MEMBER_I32`), the
class count rides in every candidate's plan (`plan.classes` -> `call.classes`
at NEW and FIT). Labels never cross the ABI. This route runs a classifier
only when the answer is the one its Python units give:
  * y is a 1-D Array as long as X with at least two classes;
  * float labels are whole finite numbers (a classifier's own fit may refuse
    a continuous target: that refusal is the Python units' to raise);
  * every fold's TRAIN side holds every class. The estimator's own fit on a
    fold that lacks a class learns fewer classes than the codes name, which
    is a different model; such a fold set takes the Python units. The scan
    is one pass per fold over its train rows, in parallel with the fold
    shapes, and stops at the row that completes the classes.

WHEN IT RUNS (`msel_member_units` answers `ran == False` otherwise and the
caller keeps its Python units): the member's class carries a member mark and
its family's admit fills it; default scoring (the estimator's own score);
`error_score='raise'`; no fitted estimator asked back; a host float32
C-order X with a target; the member answers fit and score and reads dense
host views. Device fold sets, the slice route and learning_curve's test
reuse keep their own route (the caller tests those before calling).

Memory per worker: nothing for viewed or indexed sides; for a gathered side
its rows (at most one copy of X and y), as a Python unit holds. The worker count is the caller's
(`msel_unit_workers`: the n_jobs bound, the units, one on a GPU).
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.classical_validate_unpack import classical_check_y, classical_estimator_name
from bindings.member_boundary_door import member_admit_serving
from bindings.msel_common_door import msel_is_array, msel_addr, msel_index_i64, msel_encode
from bindings.msel_cv_run import msel_seconds
from bindings.msel_estimator_native_door import (
    msel_estimator_plan, msel_native_clone, msel_native_set_params, msel_clone_parameter,
)
from bindings.native_unit_os_threads_run import native_unit_cores
from bindings.py_balanced import py_call, py_type, py_get
from core.host_parallel import host_parallelize
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free, member_kind_capabilities, MEMBER_I64,
    MEMBER_F32, MEMBER_I32,
    MEMBER_OP_FIT, MEMBER_OP_SCORE, MEMBER_CAN_FIT, MEMBER_CAN_SCORE, MEMBER_CAN_CLASSIFIER, MEMBER_CAN_ROW_INDEX,
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
    # Per fold side (2 * fold + {0 train, 1 test}): the first row when the
    # side is one run of consecutive rows (a view of X), else -1.
    var starts: List[Int]
    var x: Int
    # The target, one 4-byte word per row: float32 values (a regressor) or
    # int32 class codes (a classifier; `classes` > 0), per `y_dtype`.
    var y: Int
    var y_dtype: Int
    var classes: Int
    # Per fold: 1 when its train side lacks a class (classifier members).
    var short: List[UInt8]
    var rows: Int
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

    def __init__(out self, x: Int, y: Int, rows: Int, cols: Int, n_folds: Int, n_units: Int, workers: Int,
        return_train: Bool):
        self.plans = List[MemberPlan]()
        self.folds = List[Int]()
        self.starts = List[Int](length=2 * n_folds, fill=-1)
        self.x = x
        self.y = y
        self.y_dtype = MEMBER_F32
        self.classes = 0
        self.short = List[UInt8](length=n_folds, fill=UInt8(0))
        self.rows = rows
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


def _fold_run_start(index: Int, count: Int) -> Int:
    """The first row when `index[0 .. count]` is one ascending run of
    consecutive rows (the fold side is then a VIEW of X), else -1."""
    if count <= 0:
        return -1
    var rows = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=index)
    var first = Int(rows[0])
    if Int(rows[count - 1]) - first != count - 1:
        return -1
    for i in range(1, count):
        if Int(rows[i]) != first + i:
            return -1
    return first


def _fold_lacks_class(index: Int, count: Int, codes: Int, classes: Int) -> Bool:
    """True when rows `index[0 .. count]` of the int32 `codes` do not hold
    every code 0 .. classes-1. Stops at the row that completes the classes.
    The codes are the encoder's (0 .. classes-1) and the row numbers the
    splitter's (inside X), as the gathers read them."""
    var seen = List[UInt8](length=classes, fill=UInt8(0))
    var rows = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=index)
    var code = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=codes)
    var found = 0
    for i in range(count):
        var c = Int(code[Int(rows[i])])
        if seen[c] == UInt8(0):
            seen[c] = UInt8(1)
            found += 1
            if found == classes:
                return False
    return True


def _msel_member_fold_shape(address: Int, side: Int):
    """One fold side (2 * fold + {0 train, 1 test}): is it a run of rows?
    A classifier's train side: does it hold every class?"""
    var run = MutPointer[MselMemberRun, MutUntrackedOrigin](unsafe_from_address=address)
    var f = side // 2
    var at = 4 * f + 2 * (side - 2 * f)
    run[].starts[side] = _fold_run_start(run[].folds[at], run[].folds[at + 1])
    if run[].classes > 0 and side == 2 * f:
        if _fold_lacks_class(run[].folds[at], run[].folds[at + 1], run[].y, run[].classes):
            run[].short[f] = UInt8(1)


def _side_mode(start: Int, indexed: Bool) -> Int:
    """How a fold side reaches the member: 0 a view of X (no copy), 1 the
    whole X with the fold's row numbers (`call.row_index`, no copy), 2 rows
    gathered into the unit's own buffer (a family that reads neither)."""
    if start >= 0:
        return 0
    return 1 if indexed else 2


def _side_set(run: MutPointer[MselMemberRun, MutUntrackedOrigin], mode: Int, start: Int, index: Int, count: Int,
    gathered_x: Int, gathered_y: Int, mut call: MemberCall):
    """Point the call's x, y and row_index at one fold side. y is 4-byte
    words either way (float32 values or int32 class codes: `run.y_dtype`)."""
    var cols = run[].cols
    var y_dtype = run[].y_dtype
    call.row_index = MemberMatrix()
    if mode == 0:
        call.x = MemberMatrix(run[].x + start * cols * 4, count, cols)
        call.y = MemberMatrix(run[].y + start * 4, count, 1, y_dtype)
    elif mode == 1:
        call.x = MemberMatrix(run[].x, run[].rows, cols)
        call.y = MemberMatrix(run[].y, run[].rows, 1, y_dtype)
        call.row_index = MemberMatrix(index, count, 1, MEMBER_I64)
    else:
        call.x = MemberMatrix(gathered_x, count, cols)
        call.y = MemberMatrix(gathered_y, count, 1, y_dtype)


def _msel_member_unit(run: MutPointer[MselMemberRun, MutUntrackedOrigin], u: Int) raises:
    var c = u // run[].n_folds
    var f = u - c * run[].n_folds
    var cols = run[].cols
    var split = run[].workers <= 1
    var train_index = run[].folds[4 * f]
    var train_rows = run[].folds[4 * f + 1]
    var test_index = run[].folds[4 * f + 2]
    var test_rows = run[].folds[4 * f + 3]
    var indexed = run[].plans[c].member.can(MEMBER_CAN_ROW_INDEX)
    var train_mode = _side_mode(run[].starts[2 * f], indexed)
    var test_mode = _side_mode(run[].starts[2 * f + 1], indexed)
    var t0 = msel_seconds()
    # X IS NOT COPIED PER FOLD when the side is a run of rows (a view) or the
    # member reads row numbers. Only a scattered side of a family without
    # MEMBER_CAN_ROW_INDEX is gathered, into this unit's own buffers.
    var train_words = train_rows if train_mode == 2 else 0
    var test_words = test_rows if test_mode == 2 else 0
    var x_train = List[Float32](length=train_words * cols, fill=0.0)
    var y_train = List[Float32](length=train_words, fill=0.0)
    var x_test = List[Float32](length=test_words * cols, fill=0.0)
    # The gathered target is copied as 4-byte words: a classifier's int32
    # codes travel through these buffers bit for bit (`run.y_dtype` names them).
    var y_test = List[Float32](length=test_words, fill=0.0)
    if train_mode == 2:
        _gather(run[].x, train_index, train_rows, cols, Int(x_train.unsafe_ptr()), split)
        _gather(run[].y, train_index, train_rows, 1, Int(y_train.unsafe_ptr()), split)
    if test_mode == 2:
        _gather(run[].x, test_index, test_rows, cols, Int(x_test.unsafe_ptr()), split)
        _gather(run[].y, test_index, test_rows, 1, Int(y_test.unsafe_ptr()), split)
    var call = member_new(run[].plans[c])
    try:
        _side_set(run, train_mode, run[].starts[2 * f], train_index, train_rows, Int(x_train.unsafe_ptr()),
                  Int(y_train.unsafe_ptr()), call)
        _ = member_call(run[].plans[c].member, MEMBER_OP_FIT, call)
        var t1 = msel_seconds()
        _side_set(run, test_mode, run[].starts[2 * f + 1], test_index, test_rows, Int(x_test.unsafe_ptr()),
                  Int(y_test.unsafe_ptr()), call)
        _ = member_call(run[].plans[c].member, MEMBER_OP_SCORE, call)
        run[].test[u] = call.value
        if run[].return_train:
            _side_set(run, train_mode, run[].starts[2 * f], train_index, train_rows, Int(x_train.unsafe_ptr()),
                      Int(y_train.unsafe_ptr()), call)
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


def _whole_classes(classes: PythonObject, b: PythonObject) raises -> Bool:
    """False when a class label is a float that is not a whole finite number
    (a continuous target, which a classifier's own fit may refuse). Int, bool
    and text labels pass. One read per CLASS, never per row."""
    for i in range(len(classes)):  # driver-loop: over the distinct classes
        var label = classes[i]
        if py_type(label) is b.float:
            var value = Float64(py=label)
            if not (value - value == 0.0) or value > 9.0e15 or value < -9.0e15 or Float64(Int(value)) != value:
                return False
    return True


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
    var classifier = False
    var run = MselMemberRun(msel_addr(x), 0, rows, cols, n_folds, n_units, workers, return_train)
    for c in range(n_candidates):  # driver-loop: over candidates (admission with Python objects; once each)
        var est = base
        if candidates is not None:
            est = msel_native_clone(base, context)
            _ = msel_native_set_params(est, msel_clone_parameter(candidates[c], context), context)
        var fields = py_call(plan["fields_get"], est, py_type(est))
        # THE FALLBACK RULE: a marked class this route cannot serve (no score:
        # a transformer or KMeans; an admission its family refuses) takes the
        # Python units exactly as an unmarked class does. Never a refusal here.
        var owner = member_admit_serving(est, fields, plan, member_context, b, MEMBER_CAN_FIT | MEMBER_CAN_SCORE)
        if not owner.published:
            return scores^
        # A classifier member reads class codes and a class count; a regressor
        # float32 words. The KIND's own mask decides (the family's union would
        # call SVR, a k-NN or forest regressor a classifier); the kind is the
        # class's, so it is asked for the first candidate only.
        if c == 0:
            var kind_mask = 0
            try:
                kind_mask = member_kind_capabilities(owner.plan)
            except:
                return scores^
            if (kind_mask & MEMBER_CAN_FIT) == 0 or (kind_mask & MEMBER_CAN_SCORE) == 0:
                return scores^
            classifier = (kind_mask & MEMBER_CAN_CLASSIFIER) != 0
        run.plans.append(owner.plan.copy())
        _ = py_call(pins.append, est)
        _ = py_call(pins.append, owner.pins)
    # The target once, as the member's own fit door admits it (present,
    # real, one column, finite, as long as X): float32 words.
    # A target that check refuses (several columns, not real) is not this
    # route's: the Python units admit or refuse it with the member's own words.
    var target = PythonObject(None)
    if classifier:
        # THE LABELS, ONCE PER OPERATION (module note, CLASSIFIER MEMBERS): the
        # int32 codes every unit reads through views, row numbers or gathers,
        # and the class count in every candidate's plan. Anything outside the
        # rule (and anything the encoder refuses) is the Python units' request.
        if Int(py=y.ndim) != 1 or Int(py=y.size) != rows:
            return scores^
        var n_classes = 0
        try:
            var encoded = msel_encode(y, context)
            n_classes = len(encoded[0])
            if n_classes < 2 or not _whole_classes(encoded[0], b):
                return scores^
            target = encoded[1]
        except:
            return scores^
        if Int(py=target.size) != rows:
            return scores^
        run.y_dtype = MEMBER_I32
        run.classes = n_classes
        for c in range(n_candidates):  # driver-loop: over candidates (one word each)
            run.plans[c].classes = n_classes
    else:
        try:
            target = classical_check_y(y, rows, classical_estimator_name(base), member_context)
        except:
            return scores^
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

    def _shape(side: Int) {imm address}:
        _msel_member_fold_shape(address, side)

    var py = Python()
    try:
        with GILReleased(py):
            # The fold sides' shapes first (independent scans of the index
            # arrays), then the units.
            host_parallelize(_shape, 2 * n_folds)
            # A classifier fold whose train side lacks a class is not this
            # route's (module note): nothing has run, the Python units take it.
            var whole = True
            for f in range(n_folds):  # driver-loop: over folds, one flag each
                if run.short[f] != UInt8(0):
                    whole = False
            if not whole:
                lanes = 0
            if lanes > 0:
                host_parallelize(_lane, lanes)
    finally:
        _ = pins
        _ = x
        _ = y
    # A unit the member refused or failed: nothing is published from this
    # route. The Python units run the request and raise the estimator's own
    # exception (type and words) in unit order, as before the member mark. The
    # repeat is paid only by a failing request.
    if lanes == 0:
        return scores^
    for u in range(n_units):  # driver-loop: over units, any failure
        if run.failed[u] != UInt8(0):
            return scores^
    scores.ran = True
    scores.test = run.test.copy()
    scores.train = run.train.copy()
    scores.fit_time = run.fit_time.copy()
    scores.score_time = run.score_time.copy()
    return scores^

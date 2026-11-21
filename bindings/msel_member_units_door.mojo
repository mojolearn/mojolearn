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
    and raise the estimator's own exception. With a numeric `error_score`
    (`tolerate`) a unit whose FIT fails on its data is reported instead and
    the caller writes the value, as the Python unit's failing fit did; a
    failure after the fit still publishes nothing.

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

WHO REFUSES A NON-FINITE X. Model selection holds X with its finite test
deferred to the member doors (bindings/msel_cv_door.mojo), and this route
runs no member door. A member whose KIND has `MEMBER_CAN_TEST_ROWS` refuses
non-finite rows in its own fit and score, from the pass its kernel already
makes: nothing is scanned here; its refusal fails the unit, nothing is
published (or, with a numeric `error_score`, the unit is reported as its
failing fit was) and the Python units raise the estimator's own words. Any
other member reads admitted words: X's verdict is read ONCE per operation at
UNPACK (the held block's recorded verdict, `admitted_held_verify`, which
scans at most once per held X and records it for every later door; an unheld
X is scanned once across the host cores); not finite = `ran == False` and
the Python units' member doors refuse it. y is admitted finite either way
(`classical_check_y`, or the label encode).

RESPONSE ELEMENT TYPE: the member's own OUTPUTS answer
(`member_output_dtype`, once per operation at UNPACK). The placed buffers
are int32 codes and float32 values; a member that writes another type
(int64 codes, float64 probabilities) answers `ran == False` BEFORE any fit
is spent, where its first response call used to refuse after the fits.

WHEN IT RUNS (`msel_member_units` answers `ran == False` otherwise and the
caller keeps its Python units): the member's class carries a member mark and
its family's admit fills it; a host float32 C-order X with a target; the
member's KIND answers fit and every operation the request asks (its score,
or the responses below) and reads dense host views. Device fold sets, the
slice route and learning_curve's test reuse keep their own route (the caller
tests those before calling).

RESPONSES (`responses` not empty: a named scorer, or cross_val_predict). A
unit then fits and, instead of the member's own score, writes
`MEMBER_OP_PREDICT` / `MEMBER_OP_PREDICT_PROBA` of its test rows (and of its
train rows when train scores are asked) into buffers placed at UNPACK: one
float32 Array per (unit, side, response), sized from `member_outputs` before
any state exists, which the unit's worker fills with the GIL released (no
copy afterwards); a classifier's predict writes int32 codes into the unit's
own typed list and PACK decodes them to labels with the classes of the one
encode (`labels_decode_i32_codes`). `scores.outputs[(unit * sides + side) *
len(responses) + k]` is the prediction object (side 0 test, 1 train). The
CALLER runs its metric over them (bindings/msel_cv_door.mojo): this file
knows no scorer. Served only where the width is the plain one: predict one
column, predict_proba one column per class; anything else takes the Python
units.

THE RESIDENT SCOPE (MEMBER-ABI-V2.md item 27, THE PIN RULE at
core/member_abi.mojo `MemberCall.pin`). For a member KIND with
`MEMBER_CAN_SCOPE` the calling thread's scope is read ONCE per operation at
UNPACK (`member_scope_word`; a kind without the bit costs nothing). Inside
`resident()`: every unit's calls carry the scope word, and X's own object is
pinned once per offering call (`member_pin_take`: fit, plus each score or
response call of each side), so the member keeps X on the device across the
units, where it used to put a fold per call. A kind that also reads row
numbers gets EVERY side, a run of rows included, as the whole X plus the
fold's index: one resident base for all folds, each fold gathered on the
device (a view of a run would be a second resident range per fold). A kind
without row numbers offers the pin for its view sides only; a side gathered
into the unit's own buffer never carries a pin. After the units joined the
references no cache took are returned (`member_pin_return`). OUTSIDE
`resident()`: no scope, no pin; the member puts X per call exactly as its own
door does outside a scope. A scope read that raises (a thread-local holding
something else) answers `ran == False`: the Python units' doors refuse it.

A SEARCH'S REFIT (`msel_member_refit`): the same route with one candidate
and one unit whose train side is all of X (a view), fit only; the fitted
state is packed into the route's own clone and held by it.

FITTED ESTIMATORS ASKED BACK (`return_estimator`; `keep` True). A unit then
leaves its fitted state alive; PACK (GIL held, after the units joined) makes
one Python estimator per unit: the native clone of the candidate, the
family's `member_pack` (a classifier: `member_pack_labeled` with the labels
the estimator's own fit publishes), and the state becomes that estimator's
HELD member state (bindings/member_held_door.mojo), so its first predict
imports nothing; where nothing can be held the state is freed. The family
must export (`MEMBER_CAN_EXPORT`), else the Python units run. PACK builds
Python objects: SERIAL BY NATURE under the GIL, one estimator per unit.

Memory per worker: nothing for viewed or indexed sides; for a gathered side
its rows (at most one copy of X and y), as a Python unit holds. The worker count is the caller's
(`msel_unit_workers`: the n_jobs bound, the units, one on a GPU).
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.classical_validate_unpack import classical_check_y, classical_estimator_name
from bindings.member_boundary_door import member_admit_serving, member_pack, member_pack_labeled
from bindings.member_held_door import member_held_adopt
from bindings.member_scope_door import member_scope_word, member_pin_take, member_pin_return
from bindings.classifier_public_lifecycle_door import _public_labels
from bindings.classical_labels_door import labels_decode_i32_codes
from bindings.family_common_pack import family_array
from bindings.msel_common_door import msel_is_array, msel_addr, msel_index_i64, msel_encode
from bindings.msel_cv_run import msel_seconds
from bindings.msel_estimator_native_door import (
    msel_estimator_plan, msel_native_clone, msel_native_set_params, msel_clone_parameter,
)
from bindings.native_unit_os_threads_run import native_unit_cores
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.py_admitted_registry import admitted_held_index
from bindings.admitted_meta_scope_unpack import admitted_held_verify
from core.host_finite_scan import host_all_finite_f32
from core.host_parallel import host_parallelize
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free, member_kind_capabilities, MEMBER_I64,
    MEMBER_F32, MEMBER_I32, MEMBER_CAN_EXPORT, member_outputs, member_output_dtype, MEMBER_CAN_TEST_ROWS,
    MEMBER_OP_PREDICT, MEMBER_OP_PREDICT_PROBA, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA,
    MEMBER_OP_FIT, MEMBER_OP_SCORE, MEMBER_CAN_FIT, MEMBER_CAN_SCORE, MEMBER_CAN_CLASSIFIER, MEMBER_CAN_ROW_INDEX,
    MEMBER_CAN_SCOPE, MEMBER_SCOPE_NONE, member_pin_settle,
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
    # One fitted Python estimator per unit when they were asked back, else None.
    var estimators: PythonObject
    # RESPONSES (module note): the prediction objects, index
    # (unit * sides + side) * responses + k; an empty list without responses.
    var outputs: PythonObject
    var sides: Int
    # `tolerate`: 1 for a unit whose FIT failed (the caller writes its
    # `error_score` into that unit's scores); empty otherwise.
    var fit_failed: List[UInt8]
    # Per candidate: the estimator its units were admitted from (the caller's
    # metric reads its device once per scorer template).
    var members: PythonObject

    def __init__(out self):
        self.ran = False
        self.estimators = PythonObject(None)
        self.outputs = Python.list()
        self.sides = 1
        self.fit_failed = List[UInt8]()
        self.members = Python.list()
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
    # `keep`: a unit leaves its fitted state in `states[u]` for PACK (0: none).
    var keep: Bool
    var states: List[Int]
    var test: List[Float64]
    var train: List[Float64]
    var fit_time: List[Float64]
    var score_time: List[Float64]
    var failed: List[UInt8]
    var errors: List[String]
    # RESPONSES (module note): the operations each unit writes instead of the
    # member's score, each one's columns per row and its 4-byte word type, and
    # the placed buffer of every (unit, side, response): index
    # (unit * sides + side) * len(ops) + k. `codes` owns a classifier's
    # predict words (the other slots are empty lists).
    var ops: List[Int]
    var widths: List[Int]
    var words: List[Int]
    var sides: Int
    var dst: List[Int]
    var codes: List[List[Int32]]
    # A numeric `error_score`: a unit whose FIT fails records it here and
    # runs nothing more; the other units are unaffected.
    var tolerate: Bool
    var fit_failed: List[UInt8]
    # A search's refit: one unit that fits on every row and runs nothing else.
    var fit_only: Bool
    # THE RESIDENT SCOPE (module note): the caller's scope handle word
    # (MEMBER_SCOPE_NONE: none), the pin token of X's own object (0: nothing
    # is offered), whether every side goes as the whole X plus row numbers,
    # and per unit the offered references a cache took.
    var scope: Int
    var pin: Int
    var index_all: Bool
    var taken: List[Int]

    def __init__(out self, x: Int, y: Int, rows: Int, cols: Int, n_folds: Int, n_units: Int, workers: Int,
        return_train: Bool, keep: Bool):
        self.ops = List[Int]()
        self.widths = List[Int]()
        self.words = List[Int]()
        self.sides = 2 if return_train else 1
        self.dst = List[Int]()
        self.codes = List[List[Int32]]()
        self.tolerate = False
        self.fit_only = False
        self.scope = MEMBER_SCOPE_NONE
        self.pin = 0
        self.index_all = False
        self.taken = List[Int](length=n_units, fill=0)
        self.fit_failed = List[UInt8](length=n_units, fill=UInt8(0))
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
        self.keep = keep
        self.states = List[Int](length=n_units, fill=0)
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


def _offered(run: MutPointer[MselMemberRun, MutUntrackedOrigin], u: Int, op: Int, offer: Int,
    mut call: MemberCall) raises:
    """One member call of unit `u` over the side the call points at, offering
    the pin token `offer` (0: the side is the unit's own gathered buffer, or
    the run has no scope). The member's answer about the reference is read
    whether the call returns or raises (THE PIN RULE)."""
    call.pin = offer
    if offer == 0:
        _ = member_call(run[].plans[u // run[].n_folds].member, op, call)
        return
    try:
        _ = member_call(run[].plans[u // run[].n_folds].member, op, call)
    finally:
        run[].taken[u] += member_pin_settle(call)


def _respond(run: MutPointer[MselMemberRun, MutUntrackedOrigin], u: Int, side: Int, count: Int, offer: Int,
    mut call: MemberCall) raises:
    """Write every asked response of the side the call points at into the
    unit's placed buffers (RESPONSES, module note). The fitted state reads
    x (and row_index) as `_side_set` left them."""
    var n_ops = len(run[].ops)
    var at = (u * run[].sides + side) * n_ops
    for k in range(n_ops):  # driver-loop: over the request's responses (at most two)
        call.dst = MemberMatrix(run[].dst[at + k], count, run[].widths[k], run[].words[k])
        _offered(run, u, run[].ops[k], offer, call)
    call.dst = MemberMatrix()


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
    if run[].index_all and indexed and not run[].fit_only:
        # In the caller's resident scope every side is the whole X plus the
        # fold's row numbers: one resident base for all folds (module note).
        train_mode = 1
        test_mode = 1
    # A side gathered into this unit's own buffer never carries the pin.
    var train_offer = run[].pin if train_mode != 2 else 0
    var test_offer = run[].pin if test_mode != 2 else 0
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
    call.scope = run[].scope
    call.pin = 0
    var kept = False
    try:
        _side_set(run, train_mode, run[].starts[2 * f], train_index, train_rows, Int(x_train.unsafe_ptr()),
                  Int(y_train.unsafe_ptr()), call)
        if run[].tolerate:
            try:
                _offered(run, u, MEMBER_OP_FIT, train_offer, call)
            except error:
                run[].errors[u] = String(error)
                run[].fit_failed[u] = UInt8(1)
        else:
            _offered(run, u, MEMBER_OP_FIT, train_offer, call)
        var t1 = msel_seconds()
        if run[].fit_failed[u] != UInt8(0):
            run[].fit_time[u] = t1 - t0
            return
        if run[].fit_only:
            run[].fit_time[u] = t1 - t0
            run[].states[u] = call.state
            kept = True
            return
        var n_ops = len(run[].ops)
        _side_set(run, test_mode, run[].starts[2 * f + 1], test_index, test_rows, Int(x_test.unsafe_ptr()),
                  Int(y_test.unsafe_ptr()), call)
        if n_ops == 0:
            _offered(run, u, MEMBER_OP_SCORE, test_offer, call)
            run[].test[u] = call.value
        else:
            _respond(run, u, 0, test_rows, test_offer, call)
        if run[].return_train:
            _side_set(run, train_mode, run[].starts[2 * f], train_index, train_rows, Int(x_train.unsafe_ptr()),
                      Int(y_train.unsafe_ptr()), call)
            if n_ops == 0:
                _offered(run, u, MEMBER_OP_SCORE, train_offer, call)
                run[].train[u] = call.value
            else:
                _respond(run, u, 1, train_rows, train_offer, call)
        run[].fit_time[u] = t1 - t0
        run[].score_time[u] = msel_seconds() - t1
        if run[].keep:
            # The fitted state outlives the unit: PACK publishes it.
            run[].states[u] = call.state
            kept = True
    finally:
        if not kept:
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


def _free_kept(mut run: MselMemberRun):
    """Release every fitted state the units left for PACK (a route that
    publishes nothing, or a PACK that failed). Never raises."""
    for u in range(run.n_units):  # driver-loop: over units, one release each
        if run.states[u] != 0:
            var call = MemberCall(run.plans[u // run.n_folds].route, run.states[u])
            member_free(run.plans[u // run.n_folds], call)
            run.states[u] = 0


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
    folds: PythonObject, return_train: Bool, keep: Bool, workers_in: Int, context: PythonObject,
    b: PythonObject, responses: List[Int], tolerate: Bool) raises -> MselMemberScores:
    return _member_units(base, candidates, y, x, folds, return_train, keep, workers_in, context, b, responses,
                         tolerate, False)


def msel_member_refit(base: PythonObject, parameters: PythonObject, y: PythonObject, x: PythonObject,
    context: PythonObject, b: PythonObject) raises -> PythonObject:
    """A search's refit over the native member: a clone of `base` with
    `parameters` set, fitted ONCE on every row of X (a view: nothing is
    gathered or copied) with the GIL released, packed by its family's
    `member_pack` and holding its fitted member state
    (bindings/member_held_door.mojo), so a caller that uses the best
    estimator as a member imports nothing. None when this member, data or
    fit is not one the route runs: the caller then fits the estimator through
    its own door, which raises its own exception (THE FALLBACK RULE)."""
    var none = List[Int]()
    var got = _member_units(base, Python.list(parameters), y, x, Python.list(PythonObject(None)), False, True, 1,
                            context, b, none, False, True)
    if not got.ran:
        return PythonObject(None)
    return got.estimators[0]


def _member_units(base: PythonObject, candidates: PythonObject, y: PythonObject, x: PythonObject,
    folds: PythonObject, return_train: Bool, keep: Bool, workers_in: Int, context: PythonObject,
    b: PythonObject, responses: List[Int], tolerate: Bool, fit_only: Bool) raises -> MselMemberScores:
    """Run every (candidate, fold) unit over native members, or answer
    `ran == False` having changed nothing when this member, data or request
    is not one this route runs (module note). `keep`: one fitted estimator
    per unit is asked back (`scores.estimators`). `context` is model selection's
    context, `b` the builtins module, `workers_in` the caller's worker bound.
    `responses`: empty = the member's own score per unit; else the
    `MEMBER_OP_PREDICT` / `MEMBER_OP_PREDICT_PROBA` outputs the caller scores
    or returns (`scores.outputs`). `tolerate` (a numeric `error_score`): a
    unit whose FIT fails is reported in `scores.fit_failed` and the others
    still publish; without it any failing unit publishes nothing."""
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
    var tests_rows = False
    var scope = MEMBER_SCOPE_NONE
    var index_all = False
    var run = MselMemberRun(msel_addr(x), 0, rows, cols, n_folds, n_units, workers, return_train, keep)
    # Per candidate: its estimator and its admission's pins (PACK reads them).
    var members = Python.list()
    var member_pins = Python.list()
    var n_ops = len(responses)
    var asked = MEMBER_CAN_FIT
    if n_ops == 0 and not fit_only:
        asked = asked | MEMBER_CAN_SCORE
    for k in range(n_ops):  # driver-loop: over the request's responses (at most two)
        if responses[k] == MEMBER_OP_PREDICT:
            asked = asked | MEMBER_CAN_PREDICT
        elif responses[k] == MEMBER_OP_PREDICT_PROBA:
            asked = asked | MEMBER_CAN_PREDICT_PROBA
        else:
            return scores^
    var needs = asked
    if keep:
        needs = needs | MEMBER_CAN_EXPORT
    for c in range(n_candidates):  # driver-loop: over candidates (admission with Python objects; once each)
        var est = base
        if candidates is not None:
            est = msel_native_clone(base, context)
            _ = msel_native_set_params(est, msel_clone_parameter(candidates[c], context), context)
        var fields = py_call(plan["fields_get"], est, py_type(est))
        # THE FALLBACK RULE: a marked class this route cannot serve (no score:
        # a transformer or KMeans; an admission its family refuses) takes the
        # Python units exactly as an unmarked class does. Never a refusal here.
        var owner = member_admit_serving(est, fields, plan, member_context, b, needs)
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
            if (kind_mask & asked) != asked:
                return scores^
            classifier = (kind_mask & MEMBER_CAN_CLASSIFIER) != 0
            tests_rows = (kind_mask & MEMBER_CAN_TEST_ROWS) != 0
            if (kind_mask & MEMBER_CAN_SCOPE) != 0:
                # THE RESIDENT SCOPE (module note): read once per operation,
                # only for a kind that can keep X resident. A read that
                # raises is the Python units' doors' to refuse.
                try:
                    scope = member_scope_word(context["family_msel_rows_context"])
                except:
                    return scores^
                index_all = scope >= 0 and (kind_mask & MEMBER_CAN_ROW_INDEX) != 0
        run.plans.append(owner.plan.copy())
        _ = py_call(members.append, est)
        _ = py_call(member_pins.append, owner.pins)
    # X's finite verdict, once per operation, only for a member kind that
    # does not test its rows itself (module note): the held block's recorded
    # verdict (one scan per held X at most, then recorded), else one scan
    # across the host cores. Not finite: the Python units, whose member doors
    # refuse it in the estimator's own words.
    if not tests_rows:
        var finite = False
        try:
            var held_x = admitted_held_index(x)
            if held_x >= 0:
                finite = admitted_held_verify(held_x)
            else:
                finite = host_all_finite_f32(
                    MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=msel_addr(x)), rows * cols)
            _ = x  # keep the Python object alive through the scan above
        except:
            finite = False
        if not finite:
            return scores^
    # The target once, as the member's own fit door admits it (present,
    # real, one column, finite, as long as X): float32 words.
    # A target that check refuses (several columns, not real) is not this
    # route's: the Python units admit or refuse it with the member's own words.
    var target = PythonObject(None)
    var labels = PythonObject(None)
    var classes = PythonObject(None)
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
            classes = encoded[0]
            if keep:
                # The labels as the estimator's own fit publishes them.
                labels = _public_labels(encoded[0], member_context)
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
    if fit_only:
        # The one unit's train side is every row of X: a view from row 0, no
        # index array; it holds every class by construction of the encode.
        run.folds.append(0)
        run.folds.append(rows)
        run.folds.append(0)
        run.folds.append(0)
        run.starts[0] = 0
        run.starts[1] = 0
        run.fit_only = True
    for f in range(0 if fit_only else n_folds):  # driver-loop: over folds (their index arrays, read once)
        var pair = folds[f]
        var train = msel_index_i64(pair[0], context)
        var test = msel_index_i64(pair[1], context)
        _ = py_call(pins.append, train)
        _ = py_call(pins.append, test)
        run.folds.append(msel_addr(train))
        run.folds.append(Int(py=train.size))
        run.folds.append(msel_addr(test))
        run.folds.append(Int(py=test.size))
    # RESPONSES: every unit's output buffers, placed once (module note). The
    # widths are asked before any state exists; a width other than the plain
    # one (or a family that cannot say it before a fit) is the Python units'.
    var outputs = Python.list()
    if n_ops > 0:
        try:
            for k in range(n_ops):  # driver-loop: over the request's responses (at most two)
                var width = member_outputs(run.plans[0], responses[k], cols, 1)
                var coded = classifier and responses[k] == MEMBER_OP_PREDICT
                if responses[k] == MEMBER_OP_PREDICT_PROBA and (not classifier or width != run.classes):
                    return scores^
                if responses[k] == MEMBER_OP_PREDICT and width != 1:
                    return scores^
                # The element type is the member's own answer; one that does
                # not say it is read as the placed type (as before the word
                # existed). Any other type is the Python units' (module note).
                var placed = MEMBER_I32 if coded else MEMBER_F32
                if member_output_dtype(run.plans[0], responses[k], cols, 1, placed) != placed:
                    return scores^
                run.ops.append(responses[k])
                run.widths.append(width)
                run.words.append(placed)
            for u in range(n_units):  # driver-loop: over units (their placed buffers, allocated once each)
                var f = u - (u // n_folds) * n_folds
                for side in range(run.sides):  # driver-loop: test side, then train side when asked
                    var count = run.folds[4 * f + 3] if side == 0 else run.folds[4 * f + 1]
                    for k in range(n_ops):  # driver-loop: over the request's responses
                        if run.words[k] == MEMBER_I32:
                            run.codes.append(List[Int32](length=count, fill=Int32(0)))
                            _ = py_call(outputs.append, PythonObject(None))
                            run.dst.append(0)
                        else:
                            run.codes.append(List[Int32]())
                            var shape = Python.tuple(count) if run.widths[k] == 1 else Python.tuple(count, run.widths[k])
                            var placed = family_array(shape, "<f4", context)
                            _ = py_call(outputs.append, placed)
                            run.dst.append(msel_addr(placed) if count > 0 else 0)
            # The code lists' words are addressed once every list is in place.
            for i in range(len(run.codes)):  # driver-loop: over the placed buffers (one address each)
                if len(run.codes[i]) > 0:
                    run.dst[i] = Int(run.codes[i].unsafe_ptr())
        except:
            return scores^
    run.tolerate = tolerate
    # THE PIN RULE: one reference on X's own object per member call that can
    # offer it (the fit, then one score or each response per side), taken
    # here with the interpreter held; what no cache took comes back below.
    var offers = 0
    if scope >= 0:
        offers = n_units * (1 + (0 if fit_only else run.sides * max(n_ops, 1)))
        run.scope = scope
        run.index_all = index_all
        run.pin = member_pin_take(x, offers)
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
            if not fit_only:
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
        if run.pin != 0:
            var took = 0
            for u in range(n_units):  # driver-loop: over units, the references their caches took
                took += run.taken[u]
            member_pin_return(run.pin, offers - took)
            run.pin = 0
        _ = pins
        _ = members
        _ = member_pins
        _ = outputs
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
            _free_kept(run)
            return scores^
        # A FIT that failed under a numeric error_score takes that value at
        # the caller, as the Python unit's failing fit did. NOT when the words
        # are a refusal of this route and not of the data (an operation or
        # view the member does not publish, a missing image): the estimator's
        # own fit may well succeed, so the Python units run the request. Nor
        # when fitted estimators are asked back (the Python unit hands back
        # its unfitted clone).
        if run.fit_failed[u] != UInt8(0):
            if (keep or run.errors[u].startswith("NotImplementedError") or run.errors[u].startswith("ImportError")
                    or run.errors[u].startswith("member ABI")):
                _free_kept(run)
                return scores^
    if keep:
        # PACK: one fitted Python estimator per unit (module note). A pack the
        # family refuses publishes nothing: the Python units run the request.
        var estimators = Python.list()
        try:
            for u in range(n_units):  # driver-loop: over units (Python estimators, GIL held)
                var c = u // n_folds
                # A refit's member is this route's own clone, never fitted
                # and seen by nobody: it is the estimator handed back.
                var fitted = members[c] if fit_only else msel_native_clone(members[c], context)
                var fitted_fields = py_call(plan["fields_get"], fitted, py_type(fitted))
                if classifier:
                    member_pack_labeled(fitted, fitted_fields, member_context, b, run.plans[c], run.states[u], labels)
                else:
                    member_pack(fitted, fitted_fields, member_context, b, run.plans[c], run.states[u])
                var state = run.states[u]
                run.states[u] = 0
                if not member_held_adopt(fitted, plan, member_context, member_pins[c], run.plans[c], state):
                    var call = MemberCall(run.plans[c].route, state)
                    member_free(run.plans[c], call)
                _ = py_call(estimators.append, fitted)
        except:
            _free_kept(run)
            return scores^
        scores.estimators = estimators
    if n_ops > 0 and classifier:
        # PACK: a classifier's predicted codes as labels, one decode per
        # (unit, side) with the classes of the one encode (code c is the
        # estimator's own `classes_[c]`). SERIAL BY NATURE: Python labels,
        # GIL held; the estimator's own predict made the same decode per unit.
        try:
            for i in range(len(run.codes)):  # driver-loop: over the placed code buffers
                if run.words[i - (i // n_ops) * n_ops] == MEMBER_I32:
                    py_setitem(outputs, PythonObject(i), labels_decode_i32_codes(classes, run.codes[i], member_context))
        except:
            _free_kept(run)
            return scores^
    scores.outputs = outputs
    scores.sides = run.sides
    scores.members = members
    if tolerate:
        scores.fit_failed = run.fit_failed.copy()
    scores.ran = True
    scores.test = run.test.copy()
    scores.train = run.train.copy()
    scores.fit_time = run.fit_time.copy()
    scores.score_time = run.score_time.copy()
    return scores^

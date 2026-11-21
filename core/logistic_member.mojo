# SPDX-License-Identifier: Apache-2.0
"""LogisticRegression as a MEMBER (core/member_abi.mojo, schema v2).

One entry, `logistic_member_v2`, published by each estimators image
(`logistic_member_publish`, beside its QN kernel table). It runs the family's
existing typed RUN (core/logistic_route.mojo `logistic_fit_run` /
`logistic_query_run`) through the QN entry the route holds in `entry` (the
route LogisticRegression's own doors resolve, packed by
`logistic_route_pack`; its flag bits say whether the image has the row-argmax
entry). No kernel is touched; no other route slot is read.

Kind 1, QNRegressor, is core/qn_regressor_member.mojo (its own words, views
and blocks; this entry hands it every kind-1 NEW and every state it minted).

MEMBER_OP_NEW parameters (Float64 words, count 10): [0] kind (0
LogisticRegression), [1] fit_intercept (0 / 1), [2] l1 strength, [3] l2
strength (both as the estimator's own C / penalty / l1_ratio checks produce
them), [4] max_iter, [5] linesearch_max_iter, [6] lbfgs_memory, [7]
penalty_normalized (0 / 1), [8] the gradient tolerance (`tol`), [9] the
change tolerance (`tol * 0.01`, computed by the estimator's own check).

Classifier (MEMBER_CAN_CLASSIFIER): `y` holds the class CODES 0..classes-1 as
a host int64 column (MEMBER_I64; int32 and float32 codes are also read), and
`call.classes` is the class count at FIT (at least two). PREDICT writes int64
codes (the QN entries' own output word) to `dst`, rows x 1. DECISION writes
float32 scores, rows x 1 for two classes, rows x classes for more.
PREDICT_PROBA writes float64 probabilities, rows x 2 for two classes, rows x
classes for more. SCORE is the mean accuracy of the predicted codes against
`y` (the same codes); a weighted score is not published (the caller predicts
and runs its own weighted metric). Views: X dense row-major host float32;
sample weights host float32, one per row. A device-home view, a row or column
index and a nonzero `fit_output` are refused in words. Multiclass with an l1
strength is refused in the estimator's own words.

MEMBER_CAN_TEST_ROWS: every operation that reads X refuses a NaN or an
infinity itself (the QN entries' finite status word, host and device, fit and
query); at FIT the sample weights and float32 codes are scanned here (one
parallel pass each, core/host_finite_scan.mojo), and float32 codes at SCORE.
OUTPUTS answers `out_dtype`: PREDICT int64, DECISION float32, PREDICT_PROBA
float64.

Fitted-state blocks (EXPORT / IMPORT), in index order:
0 `MEMBER_BLOCK_COEF` float32 targets x features (targets = 1 for two
classes, else classes); 1 `MEMBER_BLOCK_INTERCEPT` float32 1 x targets
(zeros without an intercept); 2 `MEMBER_BLOCK_SCALARS` float64 [n_iter,
objective, retcode, features, classes]; 3 `MEMBER_BLOCK_FAMILY` float32
1 x ((features + intercept) * targets), the solver's weight block `_w`. An
import needs the scalars and the weight block; coef and intercept are derived
from the weight block (a parallel copy, at the first export of either).

The fitted state is OWNED here, in a process-wide store of fixed chunks: a
slot never moves once minted, so a fit or predict runs on its slot without
the store's lock; the lock covers only mint, free and the handle lookup.
Serial by nature here: the store's bookkeeping. The code cast, the accuracy
count and the host argmax split their rows across the host's workers.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from core.admitted_refusals import admitted_finite_refuse, admitted_finite_wording
from core.host_finite_scan import host_all_finite_f32
from core.host_helpers_impl import argmax_rows_f32
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.logistic_model_state import LogisticFitLayout, logistic_fit_target_count, logistic_publish_coefficients
from core.logistic_native_api import (
    QnFitRequest, QnQueryRequest, QN_NATIVE_DECISION, QN_NATIVE_BINARY, QN_NATIVE_MULTICLASS, QN_NATIVE_PROBA,
)
from core.logistic_route import LogisticRoute, logistic_route_view, logistic_fit_run, logistic_query_run
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_I64,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_PREDICT_PROBA, MEMBER_OP_DECISION, MEMBER_OP_SCORE, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT,
    MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_DECISION, MEMBER_CAN_SCORE,
    MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT, MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_COEF, MEMBER_CAN_CLASSIFIER,
    MEMBER_CAN_TEST_ROWS,
    MEMBER_BLOCK_COEF, MEMBER_BLOCK_INTERCEPT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX,
    MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_LOGISTIC
from core.qn_regressor_member import qn_regressor_member, qn_regressor_member_owns
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _PARAMS = 10
# linear_model._QN_LOSS_LOGISTIC / _QN_LOSS_SOFTMAX (bindings/logistic_native_door.mojo)
comptime _LOSS_LOGISTIC = 0
comptime _LOSS_SOFTMAX = 2
comptime _ESTIMATOR = "LogisticRegression"

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]


struct LogisticMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var params: List[Float64]
    var fitted: Bool
    var features: Int
    var classes: Int
    var targets: Int
    var weights: List[Float32]
    var coef: List[Float32]
    var intercepts: List[Float32]
    var published: Bool
    var scalars: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.params = List[Float64]()
        self.fitted = False
        self.features = 0
        self.classes = 0
        self.targets = 0
        self.weights = List[Float32]()
        self.coef = List[Float32]()
        self.intercepts = List[Float32]()
        self.published = False
        self.scalars = List[Float64]()

    def intercept(self) -> Bool:
        return self.params[1] != 0.0

    def forget(mut self):
        self.fitted = False
        self.features = 0
        self.classes = 0
        self.targets = 0
        self.weights = List[Float32]()
        self.coef = List[Float32]()
        self.intercepts = List[Float32]()
        self.published = False
        self.scalars = List[Float64]()


struct LogisticMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[LogisticMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[LogisticMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[LogisticMemberStore, "MojolearnLogisticMemberStoreV2", 2]
comptime _SlotP = MutPointer[LogisticMemberSlot, MutUntrackedOrigin]
comptime _CallP = MutPointer[MemberCall, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[LogisticMemberSlot]()


def _mint(var params: List[Float64]) raises -> Int:
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live logistic member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[LogisticMemberSlot](length=_CHUNK, fill=LogisticMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _SlotP(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].params = params^
        slot[].forget()
        return slot[].generation * _INDEX_SPAN + index + 1
    finally:
        shared_slot_unlock(lock)


def _lookup(handle: Int, release: Bool) raises -> Int:
    """The slot address of a live handle. A released handle, a handle of
    another store or a made-up integer is refused: never a stale read."""
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = handle % _INDEX_SPAN - 1
        if handle <= 0 or index < 0 or index >= store[].count:
            raise Error("NotFittedError: this member state was released or never created")
        var address = _slot_address(index)
        var slot = _SlotP(unsafe_from_address=address)
        if not slot[].live or slot[].generation != handle // _INDEX_SPAN:
            raise Error("NotFittedError: this member state was released or never created")
        if release:
            slot[].live = False
            slot[].forget()
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _whole(call: _CallP) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a LogisticRegression member reads whole host views (no row or column index)")


def _dense(view: MemberMatrix, dtype: Int, what: String) raises:
    if not view.dense_host(dtype):
        var word = String("float32") if dtype == MEMBER_F32 else (
            String("float64") if dtype == MEMBER_F64 else String("int64"))
        raise Error("TypeError: a LogisticRegression member reads " + what + " as a dense host " + word + " view")


def _codes_view(view: MemberMatrix, rows: Int, what: String) raises:
    """`y` of a classifier member: one column of host codes (int64, int32 or
    float32), one per row."""
    if not (view.dense_host(MEMBER_I64) or view.dense_host(MEMBER_I32) or view.dense_host(MEMBER_F32)):
        raise Error("TypeError: a LogisticRegression member reads " + what
            + " as dense host class codes (int64, int32 or float32)")
    if view.cols != 1:
        raise Error("ValueError: a LogisticRegression member reads " + what + " as one column of class codes")
    if view.rows != rows:
        raise Error("ValueError: X and y have different numbers of rows")


def _code_cast(view: MemberMatrix, dst: Int, rows: Int):
    """int32 / int64 codes -> float32 (exact below 2**24, as the estimator's
    own `codes.astype('<f4')`), rows split across the host's workers."""
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    var wide = view.dtype == MEMBER_I64
    var source = view.data
    var target = _F32P(unsafe_from_address=dst)

    def task(worker: Int) {imm tasks, imm chunk, imm wide, imm source, imm target, imm rows}:
        var begin = worker * chunk
        var end = min(begin + chunk, rows)
        if wide:
            var codes = _I64P(unsafe_from_address=source)
            for r in range(begin, end):
                target[r] = Float32(codes[r])
        else:
            var codes = _I32P(unsafe_from_address=source)
            for r in range(begin, end):
                target[r] = Float32(codes[r])

    host_parallelize(task, tasks)


def _finite(view: MemberMatrix, what: String) raises:
    """The estimator's finite refusal of a host float32 view (one parallel
    pass); integer codes need none."""
    if view.dtype != MEMBER_F32:
        return
    if not host_all_finite_f32(_F32P(unsafe_from_address=view.data), view.rows * view.cols):
        raise Error("ValueError: " + admitted_finite_wording(_ESTIMATOR, what))


def _features_check(slot: _SlotP, cols: Int) raises:
    if cols != slot[].features:
        raise Error("ValueError: X has " + String(cols) + " features, but LogisticRegression is expecting "
            + String(slot[].features) + " features as input.")


def _fitted(slot: _SlotP) raises:
    if not slot[].fitted:
        raise Error("NotFittedError: this LogisticRegression member is not fitted yet")


def _fit(slot: _SlotP, call: _CallP) raises:
    _whole(call)
    if call[].fit_output != 0:
        member_unsupported(_ESTIMATOR, call[].fit_output)
    _dense(call[].x, MEMBER_F32, "X")
    var rows = call[].x.rows
    var cols = call[].x.cols
    _codes_view(call[].y, rows, "y")
    _finite(call[].y, "y")
    var classes = call[].classes
    var targets = logistic_fit_target_count(classes)
    var p = slot[].params.copy()
    if targets > 1 and p[2] != 0.0:
        raise Error("NotImplementedError: mojolearn LogisticRegression: an l1 penalty with "
            + String(classes) + " classes selects OWL-QN on the softmax objective, "
            + "which runs but has no identity gate (glm/checks/"
            + "multinomial_check.mojo gates l2 and no penalty); refused by "
            + "name until it is gated (CONTRIBUTING.md (Non-default paths))")
    var sample = 0
    if call[].weights.rows > 0:
        _dense(call[].weights, MEMBER_F32, "sample_weight")
        if call[].weights.rows != rows or call[].weights.cols != 1:
            raise Error("ValueError: sample_weight and X have different numbers of rows")
        _finite(call[].weights, "sample_weight")
        sample = call[].weights.data
    var y = call[].y.data
    # The QN solver reads float32 codes; int codes are cast once per fit.
    var codes = List[Float32]()
    if call[].y.dtype != MEMBER_F32:
        codes = List[Float32](length=rows, fill=0.0)
        _code_cast(call[].y, Int(codes.unsafe_ptr()), rows)
        y = Int(codes.unsafe_ptr())
    var intercept = p[1] != 0.0
    var layout = LogisticFitLayout(cols, targets, intercept)
    slot[].forget()
    slot[].weights = List[Float32](length=layout.weights, fill=0.0)
    var info = List[Float32](length=2, fill=Float32(0))
    var status = List[Int32](length=1, fill=Int32(0))
    var request = QnFitRequest(call[].x.data, y, Int(slot[].weights.unsafe_ptr()), Int(info.unsafe_ptr()),
        sample, Int(status.unsafe_ptr()), rows, cols, classes, Int(p[4]), Int(p[5]), Int(p[6]), intercept,
        p[7] != 0.0, _LOSS_LOGISTIC if targets == 1 else _LOSS_SOFTMAX, p[2], p[3], p[8], p[9], 0.0)
    var iterations = logistic_fit_run(logistic_route_view(call[].route), request)
    admitted_finite_refuse(Int(status[0]), _ESTIMATOR)
    _ = codes^
    slot[].features = cols
    slot[].classes = classes
    slot[].targets = targets
    slot[].scalars = [Float64(iterations), Float64(info[0]), Float64(info[1]), Float64(cols), Float64(classes)]
    slot[].fitted = True


def _query(slot: _SlotP, call: _CallP, operation: Int, dst: Int, targets: Int, route: LogisticRoute) raises:
    var status = List[Int32](length=1, fill=Int32(0))
    var request = QnQueryRequest(call[].x.data, Int(slot[].weights.unsafe_ptr()), dst, Int(status.unsafe_ptr()),
        call[].x.rows, call[].x.cols, targets, slot[].intercept(), False)
    logistic_query_run(route, operation, request)
    admitted_finite_refuse(Int(status[0]), _ESTIMATOR)


def _query_x(slot: _SlotP, call: _CallP) raises:
    _fitted(slot)
    _whole(call)
    _dense(call[].x, MEMBER_F32, "X")
    _features_check(slot, call[].x.cols)


def _argmax(scores: Int, rows: Int, cols: Int, dst: Int):
    """First-max row argmax (the host column's own scan), rows split across
    the host's workers; each row is scanned whole by one worker."""
    var tasks = host_predict_task_count_work(rows, rows * cols)
    var chunk = host_predict_chunk(rows, tasks)

    def task(worker: Int) {imm scores, imm rows, imm cols, imm dst, imm chunk}:
        var begin = worker * chunk
        var end = min(begin + chunk, rows)
        if end > begin:
            argmax_rows_f32(_F32P(unsafe_from_address=scores + begin * cols * 4), end - begin, cols,
                _I64P(unsafe_from_address=dst + begin * 8))

    host_parallelize(task, tasks)


def _codes(slot: _SlotP, call: _CallP, dst: Int) raises:
    """Predicted int64 codes of `call.x` into `dst`."""
    var rows = call[].x.rows
    if rows == 0:
        return
    var route = logistic_route_view(call[].route)
    if slot[].targets == 1:
        _query(slot, call, QN_NATIVE_BINARY, dst, 1, route)
    elif route.multiclass:
        _query(slot, call, QN_NATIVE_MULTICLASS, dst, slot[].targets, route)
    else:
        # An image without the row-argmax entry (the host column): its
        # scores, then the same first-maximum rule as the estimator's door.
        var scores = List[Float32](length=rows * slot[].targets, fill=0.0)
        _query(slot, call, QN_NATIVE_DECISION, Int(scores.unsafe_ptr()), slot[].targets, route)
        _argmax(Int(scores.unsafe_ptr()), rows, slot[].targets, dst)
        _ = scores^


def _predict(slot: _SlotP, call: _CallP) raises:
    _query_x(slot, call)
    _dense(call[].dst, MEMBER_I64, "the prediction buffer")
    if call[].dst.rows != call[].x.rows or call[].dst.cols != 1:
        raise Error("ValueError: the prediction buffer holds one code per row of X")
    _codes(slot, call, call[].dst.data)


def _decision(slot: _SlotP, call: _CallP) raises:
    _query_x(slot, call)
    _dense(call[].dst, MEMBER_F32, "the decision buffer")
    if call[].dst.rows != call[].x.rows or call[].dst.cols != slot[].targets:
        raise Error("ValueError: the decision buffer holds " + String(slot[].targets) + " scores per row of X")
    if call[].x.rows == 0:
        return
    _query(slot, call, QN_NATIVE_DECISION, call[].dst.data, slot[].targets, logistic_route_view(call[].route))


def _proba(slot: _SlotP, call: _CallP) raises:
    _query_x(slot, call)
    _dense(call[].dst, MEMBER_F64, "the probability buffer")
    if call[].dst.rows != call[].x.rows or call[].dst.cols != slot[].classes:
        raise Error("ValueError: the probability buffer holds " + String(slot[].classes) + " values per row of X")
    if call[].x.rows == 0:
        return
    _query(slot, call, QN_NATIVE_PROBA, call[].dst.data, slot[].targets, logistic_route_view(call[].route))


def _score(slot: _SlotP, call: _CallP) raises:
    _query_x(slot, call)
    var rows = call[].x.rows
    _codes_view(call[].y, rows, "y")
    _finite(call[].y, "y")
    if call[].weights.rows > 0:
        raise Error("NotImplementedError: mojolearn: a weighted score of a LogisticRegression member is not "
            + "published; the caller predicts and runs its own weighted metric")
    if rows == 0:
        raise Error("ValueError: Found array with 0 sample(s) while a minimum of 1 is required.")
    var predicted = List[Int64](length=rows, fill=0)
    _codes(slot, call, Int(predicted.unsafe_ptr()))
    # The exact count of equal codes (an integer, so the split cannot change it).
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    var counts = List[Int](length=tasks, fill=0)
    var hits = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(counts.unsafe_ptr()))
    var codes = _I64P(unsafe_from_address=Int(predicted.unsafe_ptr()))
    var source = call[].y.data
    var dtype = call[].y.dtype

    def task(worker: Int) {imm hits, imm codes, imm source, imm dtype, imm chunk, imm rows}:
        var begin = worker * chunk
        var end = min(begin + chunk, rows)
        var count = 0
        for r in range(begin, end):
            var truth: Int64
            if dtype == MEMBER_I64:
                truth = _I64P(unsafe_from_address=source)[r]
            elif dtype == MEMBER_I32:
                truth = Int64(_I32P(unsafe_from_address=source)[r])
            else:
                truth = Int64(_F32P(unsafe_from_address=source)[r])
            if truth == codes[r]:
                count += 1
        hits[worker] = count

    host_parallelize(task, tasks)
    var total = 0
    for worker in range(tasks):  # driver-loop: over the workers' partial counts
        total += counts[worker]
    _ = predicted^
    _ = counts^
    call[].value = Float64(total) / Float64(rows)


def _publish(slot: _SlotP) raises:
    """coef / intercept from the weight block, once per fitted state."""
    if slot[].published:
        return
    var layout = LogisticFitLayout(slot[].features, slot[].targets, slot[].intercept())
    slot[].coef = List[Float32](length=layout.coefficients, fill=0.0)
    slot[].intercepts = List[Float32](length=layout.targets, fill=0.0)
    logistic_publish_coefficients(layout, Int(slot[].weights.unsafe_ptr()), Int(slot[].coef.unsafe_ptr()),
        Int(slot[].intercepts.unsafe_ptr()))
    slot[].published = True


def _export(slot: _SlotP, call: _CallP) raises:
    """A view of one fitted block. The slot owns the words; they stay valid
    until this state's next fit, import or free."""
    _fitted(slot)
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > 3:
            raise Error("member ABI: a logistic member has four fitted blocks")
        block = MEMBER_BLOCK_COEF if block == 0 else (MEMBER_BLOCK_INTERCEPT if block == 1 else (
            MEMBER_BLOCK_SCALARS if block == 2 else MEMBER_BLOCK_FAMILY))
    if block == MEMBER_BLOCK_COEF:
        _publish(slot)
        call[].dst = MemberMatrix(Int(slot[].coef.unsafe_ptr()), slot[].targets, slot[].features)
    elif block == MEMBER_BLOCK_INTERCEPT:
        _publish(slot)
        call[].dst = MemberMatrix(Int(slot[].intercepts.unsafe_ptr()), 1, slot[].targets)
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block == MEMBER_BLOCK_FAMILY:
        call[].dst = MemberMatrix(Int(slot[].weights.unsafe_ptr()), 1, len(slot[].weights))
    else:
        raise Error("member ABI: a logistic member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 4


def _import(slot: _SlotP, call: _CallP) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    if call[].block == MEMBER_BLOCK_FAMILY:
        _dense(call[].x, MEMBER_F32, "a fitted block")
        var count = call[].x.rows * call[].x.cols
        var words = List[Float32](length=count, fill=0.0)
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        slot[].fitted = False
        slot[].published = False
        slot[].weights = words^
    elif call[].block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != 5:
            raise Error("member ABI: the logistic scalars block is five host float64 words "
                + "(n_iter, objective, retcode, features, classes)")
        var source = _F64P(unsafe_from_address=call[].x.data)
        slot[].fitted = False
        slot[].scalars = [source[0], source[1], source[2], source[3], source[4]]
    elif call[].block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a logistic member imports its scalars and weight blocks, not block "
            + String(call[].block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) != 5 or len(slot[].weights) == 0:
        raise Error("member ABI: a fitted logistic state needs its scalars and weight blocks")
    var features = Int(slot[].scalars[3])
    var classes = Int(slot[].scalars[4])
    var targets = logistic_fit_target_count(classes)
    var layout = LogisticFitLayout(features, targets, slot[].intercept())
    if layout.weights != len(slot[].weights):
        raise Error("ValueError: the logistic weight block does not match its features, classes and intercept")
    slot[].features = features
    slot[].classes = classes
    slot[].targets = targets
    slot[].published = False
    slot[].fitted = True


def _outputs(call: _CallP) raises:
    """OUTPUTS: the width of `dst` for the operation in `call.flags`, and the
    class count (the state's, else the caller's `call.classes`)."""
    var classes = call[].classes
    if call[].state != 0:
        var slot = _SlotP(unsafe_from_address=_lookup(call[].state, False))
        if slot[].fitted:
            classes = slot[].classes
    var targets = 1 if classes == 2 else classes
    var asked = call[].flags
    if asked == MEMBER_OP_PREDICT:
        call[].out_cols = 1
        call[].out_dtype = MEMBER_I64
    elif asked == MEMBER_OP_DECISION:
        call[].out_cols = targets
        call[].out_dtype = MEMBER_F32
    elif asked == MEMBER_OP_PREDICT_PROBA:
        call[].out_cols = classes
        call[].out_dtype = MEMBER_F64
    else:
        member_unsupported(_ESTIMATOR, asked)
    call[].classes = classes


def _logistic_member(operation: Int, address: Int) raises -> Int:
    var call = _CallP(unsafe_from_address=address)
    # Kind 1, QNRegressor (core/qn_regressor_member.mojo): its NEW words and
    # its tagged states go there; one bit test per call.
    if qn_regressor_member_owns(operation, call):
        return qn_regressor_member(operation, address)
    if operation == MEMBER_OP_CAPABILITIES:
        return (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_DECISION
            | MEMBER_CAN_SCORE | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_CAN_SAMPLE_WEIGHT
            | MEMBER_CAN_COEF | MEMBER_CAN_CLASSIFIER | MEMBER_CAN_TEST_ROWS)
    if operation == MEMBER_OP_NEW:
        if call[].param_count != _PARAMS or not call[].params:
            raise Error("member ABI: a logistic member takes ten parameters")
        var words = _F64P(unsafe_from_address=call[].params)
        if words[0] != 0.0:
            raise Error("member ABI: unknown logistic member kind")
        var params = List[Float64](length=_PARAMS, fill=0.0)
        for i in range(_PARAMS):  # driver-loop: over the ten parameter words
            params[i] = words[i]
        call[].state = _mint(params^)
        return 0
    if operation == MEMBER_OP_FREE:
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        _outputs(call)
        return 0
    var slot = _SlotP(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_PREDICT:
        _predict(slot, call)
    elif operation == MEMBER_OP_DECISION:
        _decision(slot, call)
    elif operation == MEMBER_OP_PREDICT_PROBA:
        _proba(slot, call)
    elif operation == MEMBER_OP_SCORE:
        _score(slot, call)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_ESTIMATOR, operation)
    return 0


def logistic_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of LogisticRegression (schema `MEMBER_ABI`)."""
    try:
        var word = _logistic_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def logistic_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's logistic member entry. One line in the image's
    module initialization, after its QN kernel table is published."""
    member_publish[MEMBER_FAMILY_LOGISTIC](mode, vendor, publisher, logistic_member_v2)

# SPDX-License-Identifier: Apache-2.0
"""The LinearSvm family as MEMBERS (core/member_abi.mojo, schema v2):
LinearSVC and LinearSVR.

One entry, `linear_svm_member_v2`, published by each estimators image
(`linear_svm_member_publish`, beside the logistic member). It runs the
family's existing typed RUN (core/linear_svc_native_api.mojo
`LinearSvcDriver.fit` (one-vs-rest), the QN queries `QN_NATIVE_DECISION /
BINARY / MULTICLASS`, core/qn_single_fit.mojo `QnSingleFitPlan.run` for
LinearSVR) through the entries of the `DeviceRoute` the estimators' own doors
resolve (bindings/linear_svc_boundary_unpack.mojo `linear_svc_route`):
`entry` the `linear_svc_native_api_v1` driver of the estimator's device;
`flags` its LINEAR_SVC_ROUTE_* words (a caller admits only a route with both
bits); `fifth` the x_metrics program entry (the estimator's own score program, core/linear_score_run.mojo `member_r2_score`) of the route's tier and device
(LinearSVR score; without it score is not offered). No kernel is touched.

Kinds (the second word of each class's `_mojolearn_member`): 0 LinearSVC,
1 LinearSVR.

MEMBER_OP_NEW parameters, 11 Float64 words: [0] kind, [1] fit_intercept
(0 / 1), [2] max_iter, [3] linesearch_max_iter, [4] lbfgs_memory,
[5] penalty_normalized (0 / 1), [6] loss (the class's `_LOSSES` code),
[7] l1 strength (1/C for penalty 'l1', else 0), [8] l2 strength (1/C for
'l2', else 0), [9] tol (the change tolerance is 0.1 * tol, as the own fit),
[10] epsilon (LinearSVR; 0 for LinearSVC). The estimator's own checks
(`_linear_check_common`, multi_class, epsilon) run at the caller's admission.

Views: X dense row-major host float32, BORROWED (never copied). LinearSVC:
y the caller's int32 class codes 0..classes-1 (rows x 1), `call.classes`
the class count (>= 2) at FIT and at IMPORT-done; PREDICT int32 codes
rows x 1; DECISION float32 rows x T (T = 1 for two classes, else classes);
SCORE the mean accuracy (an exact count split over the host's workers).
LinearSVR: y host float32 rows x 1; PREDICT float32 rows x 1 (the decision
query); SCORE R^2 of that prediction on the route's fifth entry
(unweighted). Sample weights: host float32 rows x 1, both kinds. A
device-home view, a row or column index and a nonzero `fit_output` are
refused in words; a weighted score is refused in words.

Fitted-state blocks (EXPORT / IMPORT), in index order: 0 `MEMBER_BLOCK_COEF`
float32 fits x features (export only); 1 `MEMBER_BLOCK_INTERCEPT` float32
1 x fits (export only); 2 `MEMBER_BLOCK_SCALARS` float64 [n_iter];
3 `MEMBER_BLOCK_FAMILY` (INFO) float64 fits x 2 [objective, retcode];
4 `MEMBER_BLOCK_FAMILY + 1` (W) float32 width x fits, the `_w` words the
queries read (width = features + fit_intercept; word index fit + fits *
column). fits = 1 for LinearSVR and two classes, else the class count. An
import needs W, SCALARS and INFO; COEF and INTERCEPT are derived from W.

The fitted state is OWNED here, in a process-wide store of fixed chunks (the
linear member's form); the lock covers only mint, free and the handle lookup.
Serial by nature here: the store's bookkeeping.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from core.host_finite_scan import host_all_finite_f32
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.linear_svc_native_api import LinearSvcDriver, LinearSvcOptions, LinearSvcFitRequest, LinearSvcFitPlan, linear_svc_pack
from core.logistic_native_api import QnFitRequest, QnQueryRequest, QN_NATIVE_DECISION, QN_NATIVE_BINARY, QN_NATIVE_MULTICLASS
from core.qn_single_fit import QnSingleFitPlan
from core.member_abi import (
    MEMBER_GIL_FREE,
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_DTYPE_UNSAID,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT, MEMBER_OP_DECISION,
    MEMBER_OP_SCORE, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_DECISION, MEMBER_CAN_SCORE, MEMBER_CAN_EXPORT,
    MEMBER_CAN_IMPORT, MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_COEF, MEMBER_CAN_CLASSIFIER,
    MEMBER_BLOCK_COEF, MEMBER_BLOCK_INTERCEPT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY,
    MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_LINEAR_SVM
from core.linear_score_run import member_r2_score
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_call, native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime LINEAR_SVM_MEMBER_PARAMS = 11
comptime LINEAR_SVM_MEMBER_SVC = 0
comptime LINEAR_SVM_MEMBER_SVR = 1
comptime LINEAR_SVM_BLOCK_INFO = MEMBER_BLOCK_FAMILY
comptime LINEAR_SVM_BLOCK_W = MEMBER_BLOCK_FAMILY + 1
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]


struct LinearSvmMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var params: List[Float64]
    var fitted: Bool
    var features: Int
    var fits: Int
    var classes: Int
    # The `_w` words (width x fits), and the coef_ / intercept_ rows derived from them.
    var w: List[Float32]
    var coef: List[Float32]
    var intercept: List[Float32]
    # [n_iter]
    var scalars: List[Float64]
    # fits x [objective, retcode]
    var info: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.params = List[Float64]()
        self.fitted = False
        self.features = 0
        self.fits = 0
        self.classes = 0
        self.w = List[Float32]()
        self.coef = List[Float32]()
        self.intercept = List[Float32]()
        self.scalars = List[Float64]()
        self.info = List[Float64]()

    def forget(mut self):
        self.fitted = False
        self.features = 0
        self.fits = 0
        self.classes = 0
        self.w = List[Float32]()
        self.coef = List[Float32]()
        self.intercept = List[Float32]()
        self.scalars = List[Float64]()
        self.info = List[Float64]()


struct LinearSvmMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[LinearSvmMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[LinearSvmMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[LinearSvmMemberStore, "MojolearnLinearSvmMemberStoreV2", 2]
comptime _SlotP = MutPointer[LinearSvmMemberSlot, MutUntrackedOrigin]
comptime _CallP = MutPointer[MemberCall, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock; a chunk never grows, so the address stays valid.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[LinearSvmMemberSlot]()


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
                raise Error("RuntimeError: mojolearn: too many live linear SVM member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[LinearSvmMemberSlot](length=_CHUNK, fill=LinearSvmMemberSlot()))
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
    """The slot address of a live handle; a stale or foreign handle is refused."""
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


def _kind(slot: _SlotP) -> Int:
    return Int(slot[].params[0])


def _name(slot: _SlotP) -> String:
    return String("LinearSVR") if _kind(slot) == LINEAR_SVM_MEMBER_SVR else String("LinearSVC")


def _intercept(slot: _SlotP) -> Bool:
    return slot[].params[1] != 0.0


def _capabilities(kind: Int, scored: Bool) -> Int:
    """The bits `kind` honors (kind -1: the family's union); `scored`: the
    route holds the R^2 entry LinearSVR's score reads."""
    # MEMBER_GIL_FREE, every kind (THE GIL RULE; fx-gil-RESULT.md, round 4):
    # every operation is calls of the route's LinearSVC entry (the QN
    # operations and the one-vs-rest fit, into a state this call owns) on
    # per-call buffers, and for LinearSVR's score the metrics R^2 entry. The
    # entry touches no CPython or GIL and reaches only its locked process
    # context; the store has its own lock.
    var bits = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
        | MEMBER_CAN_SAMPLE_WEIGHT | MEMBER_CAN_COEF | MEMBER_GIL_FREE)
    if kind != LINEAR_SVM_MEMBER_SVR:
        bits |= MEMBER_CAN_DECISION | MEMBER_CAN_CLASSIFIER | MEMBER_CAN_SCORE
    elif scored:
        bits |= MEMBER_CAN_SCORE
    return bits


def _dense(view: MemberMatrix, what: String, estimator: String) raises:
    if not view.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + estimator + " member reads " + what + " as a dense host float32 view")


def _whole(call: _CallP, estimator: String) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads whole host views (no row or column index)")


def _codes_view(view: MemberMatrix, rows: Int, what: String, estimator: String) raises:
    if not view.dense_host(MEMBER_I32) or view.cols != 1 or view.rows != rows:
        raise Error("TypeError: a " + estimator + " member reads " + what
            + " as int32 class codes, one per row of X (a dense host view)")


def _from_w(slot: _SlotP):
    """coef_ (fits x features) and intercept_ (fits) from the `_w` words,
    split across the host's workers (the inverse of `linear_svc_pack`)."""
    var fits = slot[].fits
    var features = slot[].features
    var intercept = _intercept(slot)
    var width = features + Int(intercept)
    slot[].coef = List[Float32](length=fits * features, fill=0.0)
    slot[].intercept = List[Float32](length=fits, fill=0.0)
    var source = slot[].w.unsafe_ptr()
    var coef = slot[].coef.unsafe_ptr()
    var bias = slot[].intercept.unsafe_ptr()

    def unpack(index: Int) {imm source, imm coef, imm bias, imm width, imm features, imm fits, imm intercept}:
        var row = index // width
        var column = index % width
        var value = source[row + fits * column]
        if column < features:
            coef[row * features + column] = value
        elif intercept:
            bias[row] = value

    host_parallelize(unpack, fits * width)


def _fit(slot: _SlotP, call: _CallP) raises:
    """The own fit's request words over the borrowed design."""
    var estimator = _name(slot)
    _whole(call, estimator)
    if call[].fit_output != 0:
        member_unsupported(estimator, call[].fit_output)
    _dense(call[].x, "X", estimator)
    var rows = call[].x.rows
    var cols = call[].x.cols
    var sw = 0
    if call[].weights.rows > 0:
        _dense(call[].weights, "sample_weight", estimator)
        if call[].weights.rows != rows or call[].weights.cols != 1:
            raise Error("ValueError: sample_weight and X have different numbers of rows")
        sw = call[].weights.data
    var p = slot[].params.copy()
    var intercept = p[1] != 0.0
    var width = cols + Int(intercept)
    var driver = LinearSvcDriver(call[].route.entry.call)
    slot[].forget()
    if Int(p[0]) == LINEAR_SVM_MEMBER_SVR:
        _dense(call[].y, "y", estimator)
        if call[].y.cols != 1:
            raise Error("mojolearn LinearSVR requires a 1-D y")
        if call[].y.rows != rows:
            raise Error("mojolearn LinearSVR X and y lengths differ")
        var w = List[Float32](length=width, fill=0.0)
        var request = QnFitRequest(call[].x.data, call[].y.data, Int(w.unsafe_ptr()), 0, sw, 0, rows, cols, 1,
            Int(p[2]), Int(p[3]), Int(p[4]), intercept, p[5] != 0.0, Int(p[6]), p[7], p[8], p[9], 0.1 * p[9], p[10])
        var plan = QnSingleFitPlan()
        plan.run(driver.qn(), request^)
        slot[].scalars = [Float64(plan.iterations)]
        slot[].info = [plan.objective(), Float64(plan.retcode())]
        slot[].w = w^
        slot[].features = cols
        slot[].fits = 1
        _from_w(slot)
        slot[].fitted = True
        return
    _codes_view(call[].y, rows, "y", estimator)
    var k = call[].classes
    if k < 2:
        raise Error("mojolearn LinearSVC: y has one class")
    # The original OVR selection: binary fits class 1 once.
    var first = 1 if k == 2 else 0
    var fits = 1 if k == 2 else k
    var options = LinearSvcOptions(rows, cols, first, fits, Int(p[2]), Int(p[3]), Int(p[4]), intercept,
        p[5] != 0.0, Int(p[6]), p[7], p[8], p[9], 0.1 * p[9])
    var plan = LinearSvcFitPlan(fits, width)
    plan.storage[0].prepare_info(fits)
    driver.fit(LinearSvcFitRequest(call[].x.data, call[].y.data, sw, options^), plan)
    slot[].w = List[Float32](length=fits * width, fill=0.0)
    slot[].coef = List[Float32](length=fits * cols, fill=0.0)
    slot[].intercept = List[Float32](length=fits, fill=0.0)
    linear_svc_pack(plan.storage[0], fits, cols, intercept, Int(slot[].w.unsafe_ptr()),
        Int(slot[].coef.unsafe_ptr()), Int(slot[].intercept.unsafe_ptr()))
    slot[].info = List[Float64](length=2 * fits, fill=0.0)
    for fit in range(fits):  # driver-loop: over the fitted machines' two metadata words
        slot[].info[2 * fit] = Float64(plan.storage[0].info[3 * fit])
        slot[].info[2 * fit + 1] = Float64(Int(plan.storage[0].info[3 * fit + 1]))
    slot[].scalars = [Float64(plan.iterations())]
    slot[].features = cols
    slot[].fits = fits
    slot[].classes = k
    _ = plan^
    slot[].fitted = True


def _query(slot: _SlotP, call: _CallP) raises:
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].x.cols != slot[].features:
        raise Error("mojolearn " + estimator + " feature count differs from fit")


def _decision_into(slot: _SlotP, call: _CallP, dst: Int) raises:
    """QN_NATIVE_DECISION: rows x fits float32 scores (the own decision query)."""
    var rows = call[].x.rows
    if rows == 0:
        return
    var driver = LinearSvcDriver(call[].route.entry.call)
    driver.qn().query(QN_NATIVE_DECISION, QnQueryRequest(call[].x.data, Int(slot[].w.unsafe_ptr()), dst, 0, rows,
        slot[].features, slot[].fits, _intercept(slot), False))


def _codes_into(slot: _SlotP, call: _CallP, dst: Int) raises:
    """LinearSVC class codes (int32) of each row: the own binary / multiclass
    query (int64 codes), one parallel cast."""
    var rows = call[].x.rows
    if rows == 0:
        return
    var k = slot[].classes
    var codes = List[Int64](length=rows, fill=0)
    var driver = LinearSvcDriver(call[].route.entry.call)
    driver.qn().query(QN_NATIVE_BINARY if k == 2 else QN_NATIVE_MULTICLASS, QnQueryRequest(call[].x.data,
        Int(slot[].w.unsafe_ptr()), Int(codes.unsafe_ptr()), 0, rows, slot[].features, 1 if k == 2 else k,
        _intercept(slot), False))
    var source = _I64P(unsafe_from_address=Int(codes.unsafe_ptr()))
    var target = _I32P(unsafe_from_address=dst)

    def cast(row: Int) {imm source, imm target}:
        target[row] = Int32(source[row])

    host_parallelize(cast, rows)
    _ = codes^


def _predict(slot: _SlotP, call: _CallP) raises:
    _query(slot, call)
    var rows = call[].x.rows
    if _kind(slot) == LINEAR_SVM_MEMBER_SVR:
        _dense(call[].dst, "the prediction buffer", _name(slot))
        if call[].dst.rows != rows or call[].dst.cols != 1:
            raise Error("ValueError: the prediction buffer holds one value per row of X")
        _decision_into(slot, call, call[].dst.data)
        return
    _codes_view(call[].dst, rows, "the prediction buffer", _name(slot))
    _codes_into(slot, call, call[].dst.data)


def _decision(slot: _SlotP, call: _CallP) raises:
    if _kind(slot) == LINEAR_SVM_MEMBER_SVR:
        member_unsupported(_name(slot), MEMBER_OP_DECISION)
    _query(slot, call)
    _dense(call[].dst, "the decision buffer", _name(slot))
    if call[].dst.rows != call[].x.rows or call[].dst.cols != slot[].fits:
        raise Error("ValueError: the decision buffer holds " + String(slot[].fits) + " values per row of X")
    _decision_into(slot, call, call[].dst.data)


def _score(slot: _SlotP, call: _CallP) raises:
    var estimator = _name(slot)
    _query(slot, call)
    if call[].weights.rows > 0:
        raise Error("NotImplementedError: mojolearn: a weighted score of a " + estimator
            + " member is not published; the caller predicts and runs its own weighted metric")
    var rows = call[].x.rows
    if _kind(slot) == LINEAR_SVM_MEMBER_SVR:
        var route = call[].route
        _dense(call[].y, "y", estimator)
        if call[].y.cols != 1 or call[].y.rows != rows:
            raise Error("ValueError: y must be 1-D with one entry per row of X")
        if not route.has_fifth:
            raise Error("ImportError: mojolearn: the " + estimator + " member's score needs the metrics program entry on its route")
        var prediction = List[Float32](length=rows, fill=0.0)
        _decision_into(slot, call, Int(prediction.unsafe_ptr()))
        call[].value = member_r2_score(route.fifth, call[].y.data, Int(prediction.unsafe_ptr()), rows)
        _ = prediction^
        return
    _codes_view(call[].y, rows, "y", estimator)
    if rows == 0:
        raise Error("ValueError: Found array with 0 sample(s) while a minimum of 1 is required.")
    var predicted = List[Int32](length=rows, fill=0)
    _codes_into(slot, call, Int(predicted.unsafe_ptr()))
    # The exact count of equal codes (an integer, so the split cannot change it).
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    var counts = List[Int](length=tasks, fill=0)
    var hits = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(counts.unsafe_ptr()))
    var codes = _I32P(unsafe_from_address=Int(predicted.unsafe_ptr()))
    var truth = _I32P(unsafe_from_address=call[].y.data)

    def task(worker: Int) {imm hits, imm codes, imm truth, imm chunk, imm rows}:
        var begin = worker * chunk
        var end = min(begin + chunk, rows)
        var count = 0
        for r in range(begin, end):
            if truth[r] == codes[r]:
                count += 1
        hits[worker] = count

    host_parallelize(task, tasks)
    var total = 0
    for worker in range(tasks):  # driver-loop: over the workers' partial counts
        total += counts[worker]
    _ = predicted^
    call[].value = Float64(total) / Float64(rows)


def _export(slot: _SlotP, call: _CallP) raises:
    """A view of one fitted block, owned by the slot until its next fit,
    import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot) + " member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= 5:
            raise Error("member ABI: a " + _name(slot) + " member state has 5 fitted blocks")
        if block == 0:
            block = MEMBER_BLOCK_COEF
        elif block == 1:
            block = MEMBER_BLOCK_INTERCEPT
        elif block == 2:
            block = MEMBER_BLOCK_SCALARS
        elif block == 3:
            block = LINEAR_SVM_BLOCK_INFO
        else:
            block = LINEAR_SVM_BLOCK_W
    var fits = slot[].fits
    if block == MEMBER_BLOCK_COEF:
        call[].dst = MemberMatrix(Int(slot[].coef.unsafe_ptr()), fits, slot[].features)
    elif block == MEMBER_BLOCK_INTERCEPT:
        call[].dst = MemberMatrix(Int(slot[].intercept.unsafe_ptr()), 1, fits)
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, 1, MEMBER_F64)
    elif block == LINEAR_SVM_BLOCK_INFO:
        call[].dst = MemberMatrix(Int(slot[].info.unsafe_ptr()), fits, 2, MEMBER_F64)
    elif block == LINEAR_SVM_BLOCK_W:
        call[].dst = MemberMatrix(Int(slot[].w.unsafe_ptr()), slot[].features + Int(_intercept(slot)), fits)
    else:
        raise Error("member ABI: a " + _name(slot) + " member state holds no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 5
    call[].classes = slot[].classes


def _import(slot: _SlotP, call: _CallP) raises:
    """W (width x fits), SCALARS [n_iter] and INFO (fits x 2), then DONE
    (with `call.classes` for LinearSVC)."""
    var estimator = _name(slot)
    var width_extra = Int(_intercept(slot))
    if call[].block == LINEAR_SVM_BLOCK_W:
        _dense(call[].x, "the W block", estimator)
        var count = call[].x.rows * call[].x.cols
        if call[].x.rows <= width_extra or call[].x.cols < 1:
            raise Error("member ABI: the " + estimator + " W block is (features + fit_intercept) x fits float32 words")
        slot[].fitted = False
        slot[].w = List[Float32](length=count, fill=0.0)
        unsafe_memcpy(dest=slot[].w.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        slot[].features = call[].x.rows - width_extra
        slot[].fits = call[].x.cols
    elif call[].block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != 1:
            raise Error("member ABI: the " + estimator + " scalars block is 1 host float64 word [n_iter]")
        slot[].fitted = False
        slot[].scalars = List[Float64](length=1, fill=0.0)
        unsafe_memcpy(dest=slot[].scalars.unsafe_ptr(), src=_F64P(unsafe_from_address=call[].x.data), count=1)
    elif call[].block == LINEAR_SVM_BLOCK_INFO:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.cols != 2:
            raise Error("member ABI: the " + estimator + " info block is fits x 2 host float64 words")
        var count = call[].x.rows * 2
        slot[].fitted = False
        slot[].info = List[Float64](length=count, fill=0.0)
        unsafe_memcpy(dest=slot[].info.unsafe_ptr(), src=_F64P(unsafe_from_address=call[].x.data), count=count)
    elif call[].block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a " + estimator + " member imports no fitted block " + String(call[].block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].w) == 0 or len(slot[].scalars) != 1 or len(slot[].info) != 2 * slot[].fits:
        raise Error("member ABI: a fitted " + estimator + " state needs its W, scalars and info blocks")
    if _kind(slot) == LINEAR_SVM_MEMBER_SVR:
        if slot[].fits != 1:
            raise Error("member ABI: a fitted LinearSVR state has one machine")
        slot[].classes = 0
    else:
        var classes = call[].classes
        if classes < 2 or (1 if classes == 2 else classes) != slot[].fits:
            raise Error("member ABI: a fitted LinearSVC state needs its class count (one machine for two "
                + "classes, else one per class)")
        slot[].classes = classes
    _from_w(slot)
    slot[].fitted = True


def _linear_svm_member(operation: Int, address: Int) raises -> Int:
    var call = _CallP(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        if call[].state == 0:
            # Without a state the NEW words name the kind when given.
            if call[].params and call[].param_count == LINEAR_SVM_MEMBER_PARAMS:
                return _capabilities(Int(_F64P(unsafe_from_address=call[].params)[0]), call[].route.has_fifth)
            return _capabilities(-1, call[].route.has_fifth)
        var asked = _SlotP(unsafe_from_address=_lookup(call[].state, False))
        return _capabilities(_kind(asked), call[].route.has_fifth)
    if operation == MEMBER_OP_NEW:
        if not call[].params or call[].param_count < 1:
            raise Error("member ABI: a linear SVM member takes its kind as parameter word 0")
        var words = _F64P(unsafe_from_address=call[].params)
        var kind = Int(words[0])
        if Float64(kind) != words[0] or kind < 0 or kind > LINEAR_SVM_MEMBER_SVR:
            raise Error("member ABI: unknown linear SVM member kind")
        if call[].param_count != LINEAR_SVM_MEMBER_PARAMS:
            raise Error("member ABI: a linear SVM member takes " + String(LINEAR_SVM_MEMBER_PARAMS) + " parameters")
        var params = List[Float64](length=LINEAR_SVM_MEMBER_PARAMS, fill=0.0)
        unsafe_memcpy(dest=params.unsafe_ptr(), src=words, count=LINEAR_SVM_MEMBER_PARAMS)
        call[].state = _mint(params^)
        return 0
    if operation == MEMBER_OP_FREE:
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        # With a state the state answers; without one the NEW words (the
        # kind) and the caller's class count do.
        var asked_kind = -1
        var classes = 0
        if call[].state != 0:
            var held = _SlotP(unsafe_from_address=_lookup(call[].state, False))
            asked_kind = _kind(held)
            classes = held[].classes
        elif call[].params and call[].param_count >= 1:
            asked_kind = Int(_F64P(unsafe_from_address=call[].params)[0])
            classes = call[].classes
        var op = call[].flags
        if op != MEMBER_OP_PREDICT and not (op == MEMBER_OP_DECISION and asked_kind != LINEAR_SVM_MEMBER_SVR):
            # An operation the kind lacks is refused, never sized.
            member_unsupported("a linear SVM member", op)
        call[].out_cols = 1
        call[].classes = 0
        # LinearSVR's predict and every decision: float32; LinearSVC's
        # predict: int32 codes.
        call[].out_dtype = MEMBER_F32
        if op == MEMBER_OP_PREDICT and asked_kind == LINEAR_SVM_MEMBER_SVC:
            call[].out_dtype = MEMBER_I32
        elif op == MEMBER_OP_PREDICT and asked_kind < 0:
            call[].out_dtype = MEMBER_DTYPE_UNSAID  # the kind decides (codes or values)
        if asked_kind == LINEAR_SVM_MEMBER_SVC:
            call[].classes = classes
            if op == MEMBER_OP_DECISION:
                call[].out_cols = max(1, 1 if classes == 2 else classes)
        return 0
    var slot = _SlotP(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_PREDICT:
        _predict(slot, call)
    elif operation == MEMBER_OP_DECISION:
        _decision(slot, call)
    elif operation == MEMBER_OP_SCORE:
        _score(slot, call)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_name(slot), operation)
    return 0


def linear_svm_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of LinearSVC / LinearSVR (schema `MEMBER_ABI`)."""
    try:
        var word = _linear_svm_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def linear_svm_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's LinearSvm member entry. One line in the image's
    module initialization, after its operations table is published."""
    member_publish[MEMBER_FAMILY_LINEAR_SVM](mode, vendor, publisher, linear_svm_member_v2)

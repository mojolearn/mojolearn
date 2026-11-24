# SPDX-License-Identifier: Apache-2.0
"""QNRegressor as a MEMBER: kind 1 of the Logistic family
(core/logistic_member.mojo `logistic_member_v2` hands every kind-1 call and
every state minted here to `qn_regressor_member`).

It runs QNRegressor's own typed RUN (glm/qn_door_run.mojo `qn_door_fit_run`,
`qn_door_predict_run`) through the entry of the estimator's own device route
(bindings/qn_boundary_unpack.mojo `qn_route`, the estimators QN table entry
held on its DeviceRouteOwner): `entry` the QN entry of the estimator's
device; `fifth` the x_metrics program entry (the estimator's own score program, core/linear_score_run.mojo `member_r2_score`) of the route's tier and device
(score; without it score is not offered). The resident-fold entry (`second`)
is not read: a member reads host views. No kernel is touched.

MEMBER_OP_NEW parameters, 11 Float64 words: [0] kind (1), [1]
fit_intercept (0 / 1), [2] max_iter, [3] linesearch_max_iter,
[4] lbfgs_memory, [5] penalty_normalized (0 / 1), [6] loss (the
`qn_losses` code of the estimator's `loss`), [7] l1_strength,
[8] l2_strength, [9] tol (the gradient tolerance), [10] the change
tolerance (`delta`, or tol * 0.01).

Views: X dense row-major host float32, BORROWED (the fit entry tests it
for non-finite values through its status word, as the own fit);
y host float32 rows x 1; sample weights host float32 rows x 1. PREDICT
float32 rows x 1 (the decision query); SCORE R^2 of that prediction on the
route's fifth entry (unweighted). A device-home view, a row or column index
and a nonzero `fit_output` are refused in words.

Fitted-state blocks (EXPORT / IMPORT), in index order: 0
`MEMBER_BLOCK_COEF` float32 1 x features (export only); 1
`MEMBER_BLOCK_INTERCEPT` float32 1 x 1 (export only; 0 without an
intercept); 2 `MEMBER_BLOCK_SCALARS` float64 [n_iter, objective, retcode];
3 `MEMBER_BLOCK_FAMILY` float32 1 x (features + fit_intercept), the
solver's `_w` words. An import needs W and SCALARS.

States minted here carry QN_REGRESSOR_MEMBER_TAG in their handle, so the
family entry routes them here without a table read. The store's
bookkeeping is serial by nature (the linear member's form).
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from core.host_finite_scan import host_all_finite_f32
from core.logistic_native_api import QnFitRequest, QnQueryRequest, QnFoldStore
from glm.qn_door_run import QnDoorFitRequest, QnDoorPredictRequest, qn_door_fit_run, qn_door_predict_run
from core.member_abi import (
    MEMBER_GIL_FREE,
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_SCORE, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_SCORE, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_COEF, MEMBER_CAN_ROW_INDEX, MEMBER_CAN_SCOPE, MemberMatrix,
    MEMBER_BLOCK_COEF, MEMBER_BLOCK_INTERCEPT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY,
    MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE, member_unsupported,
)
from core.linear_score_run import member_r2_score
from core.device_route import DEVICE_GPU
from core.member_fold_x import MemberFoldX, member_fold_x_open, member_fold_x_close, member_fold_take_view, member_fold_store_ready
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
# A handle minted here: this bit plus generation * span + index + 1.
comptime QN_REGRESSOR_MEMBER_TAG = 1 << 60
comptime QN_REGRESSOR_MEMBER_KIND = 1
comptime QN_REGRESSOR_MEMBER_PARAMS = 11
comptime QN_REGRESSOR_BLOCK_W = MEMBER_BLOCK_FAMILY
comptime _ESTIMATOR = "QNRegressor"
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]


struct QnRegressorMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var params: List[Float64]
    var fitted: Bool
    var features: Int
    # [coef | intercept] (the solver's words), coef, [intercept]
    var w: List[Float32]
    var coef: List[Float32]
    var intercept: List[Float32]
    # [n_iter, objective, retcode]
    var scalars: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.params = List[Float64]()
        self.fitted = False
        self.features = 0
        self.w = List[Float32]()
        self.coef = List[Float32]()
        self.intercept = List[Float32]()
        self.scalars = List[Float64]()

    def forget(mut self):
        self.fitted = False
        self.features = 0
        self.w = List[Float32]()
        self.coef = List[Float32]()
        self.intercept = List[Float32]()
        self.scalars = List[Float64]()


struct QnRegressorMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[QnRegressorMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[QnRegressorMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[QnRegressorMemberStore, "MojolearnQnRegressorMemberStoreV2", 2]
comptime _SlotP = MutPointer[QnRegressorMemberSlot, MutUntrackedOrigin]
comptime _CallP = MutPointer[MemberCall, MutUntrackedOrigin]


def qn_regressor_member_owns(operation: Int, call: _CallP) -> Bool:
    """True when this call belongs here: a state minted here, or (no state)
    a NEW / OUTPUTS / CAPABILITIES whose parameter word 0 is kind 1."""
    if call[].state != 0:
        return (call[].state & QN_REGRESSOR_MEMBER_TAG) != 0
    if operation != MEMBER_OP_NEW and operation != MEMBER_OP_OUTPUTS and operation != MEMBER_OP_CAPABILITIES:
        return False
    if not call[].params or call[].param_count < 1:
        return False
    return _F64P(unsafe_from_address=call[].params)[0] == Float64(QN_REGRESSOR_MEMBER_KIND)


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock; a chunk never grows, so the address stays valid.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[QnRegressorMemberSlot]()


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
                raise Error("RuntimeError: mojolearn: too many live QNRegressor member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[QnRegressorMemberSlot](length=_CHUNK, fill=QnRegressorMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _SlotP(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].params = params^
        slot[].forget()
        return QN_REGRESSOR_MEMBER_TAG | (slot[].generation * _INDEX_SPAN + index + 1)
    finally:
        shared_slot_unlock(lock)


def _lookup(tagged: Int, release: Bool) raises -> Int:
    """The slot address of a live handle; a stale or foreign handle is refused."""
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var handle = tagged & (QN_REGRESSOR_MEMBER_TAG - 1)
        var index = handle % _INDEX_SPAN - 1
        if (tagged & QN_REGRESSOR_MEMBER_TAG) == 0 or handle <= 0 or index < 0 or index >= store[].count:
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


def _capabilities(scored: Bool) -> Int:
    # MEMBER_GIL_FREE (THE GIL RULE; fx-gil-RESULT.md, round 4): FIT and
    # PREDICT are `qn_door_fit_run` (never its fold arm: `use_fold` False)
    # and `qn_door_predict_run`, one call of the route's QN entry on per-call
    # buffers; SCORE adds the metrics R^2 entry, host arithmetic. The store
    # has its own lock.
    var bits = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
        | MEMBER_CAN_SAMPLE_WEIGHT | MEMBER_CAN_COEF | MEMBER_GIL_FREE)
    if scored:
        bits |= MEMBER_CAN_SCORE
    return bits


def _dense(view: MemberMatrix, what: String) raises:
    if not view.dense_host(MEMBER_F32):
        raise Error("TypeError: a QNRegressor member reads " + what + " as a dense host float32 view")


def _whole(call: _CallP) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a QNRegressor member reads whole host views (no row or column index)")


def _split(slot: _SlotP):
    """coef_ and intercept_ from the solver's [coef | intercept] words."""
    var d = slot[].features
    slot[].coef = List[Float32](length=d, fill=0.0)
    if d > 0:
        unsafe_memcpy(dest=slot[].coef.unsafe_ptr(), src=slot[].w.unsafe_ptr(), count=d)
    slot[].intercept = [slot[].w[d] if slot[].params[1] != 0.0 else Float32(0.0)]


def _fit(slot: _SlotP, call: _CallP) raises:
    """`qn_fit_unpack`'s request over the borrowed design (the host route)."""
    _whole(call)
    if call[].fit_output != 0:
        member_unsupported(_ESTIMATOR, call[].fit_output)
    _dense(call[].x, "X")
    _dense(call[].y, "y")
    var rows = call[].x.rows
    var cols = call[].x.cols
    if call[].y.cols != 1:
        raise Error("mojolearn QNRegressor requires a 1-D y")
    if call[].y.rows != rows:
        raise Error("mojolearn QNRegressor X and y lengths differ")
    var sw = 0
    if call[].weights.rows > 0:
        _dense(call[].weights, "sample_weight")
        if call[].weights.rows != rows or call[].weights.cols != 1:
            raise Error("ValueError: sample_weight and X have different numbers of rows")
        sw = call[].weights.data
    var p = slot[].params.copy()
    var intercept = p[1] != 0.0
    slot[].forget()
    var w = List[Float32](length=cols + Int(intercept), fill=0.0)
    var info = List[Float32](length=2, fill=0.0)
    var status = List[Int32](length=1, fill=0)
    var coef = List[Float32](length=max(cols, 1), fill=0.0)
    var iterations = List[Int64](length=1, fill=0)
    var status_address = Int(status.unsafe_ptr())
    var request = QnFitRequest(call[].x.data, call[].y.data, Int(w.unsafe_ptr()), Int(info.unsafe_ptr()), sw,
        status_address, rows, cols, 1, Int(p[2]), Int(p[3]), Int(p[4]), intercept, p[5] != 0.0, Int(p[6]),
        p[7], p[8], p[9], p[10], 0.0)
    var route = call[].route
    var result = qn_door_fit_run(QnDoorFitRequest(route.entry, route.second, False, QnFoldStore(), request^,
        status_address, Int(coef.unsafe_ptr()), Int(iterations.unsafe_ptr())))
    slot[].scalars = [Float64(iterations[0]), result.objective, Float64(result.retcode)]
    _ = info^
    _ = status^
    _ = coef^
    slot[].w = w^
    slot[].features = cols
    _split(slot)
    slot[].fitted = True


def _predict_into(slot: _SlotP, call: _CallP, dst: Int) raises:
    if not slot[].fitted:
        raise Error("NotFittedError: this QNRegressor member is not fitted yet")
    _whole(call)
    _dense(call[].x, "X")
    var rows = call[].x.rows
    var cols = call[].x.cols
    if cols != slot[].features:
        raise Error("mojolearn QNRegressor feature count differs from fit")
    if rows == 0:
        return
    qn_door_predict_run(QnDoorPredictRequest(call[].route.entry, QnQueryRequest(call[].x.data,
        Int(slot[].w.unsafe_ptr()), dst, 0, rows, cols, 1, slot[].params[1] != 0.0, False)))


def _predict(slot: _SlotP, call: _CallP) raises:
    _dense(call[].dst, "the prediction buffer")
    if call[].dst.rows != call[].x.rows or call[].dst.cols != 1:
        raise Error("ValueError: the prediction buffer holds one value per row of X")
    _predict_into(slot, call, call[].dst.data)


def _score(slot: _SlotP, call: _CallP) raises:
    var route = call[].route
    _dense(call[].y, "y")
    if call[].y.cols != 1 or call[].y.rows != call[].x.rows:
        raise Error("ValueError: y must be 1-D with one entry per row of X")
    if call[].weights.rows > 0:
        raise Error("NotImplementedError: mojolearn: a weighted score of a QNRegressor member is not "
            + "published; the caller predicts and runs its own weighted metric")
    if not route.has_fifth:
        raise Error("ImportError: mojolearn: the QNRegressor member's score needs the metrics program entry on its route")
    var prediction = List[Float32](length=call[].x.rows, fill=0.0)
    _predict_into(slot, call, Int(prediction.unsafe_ptr()))
    call[].value = member_r2_score(route.fifth, call[].y.data, Int(prediction.unsafe_ptr()), call[].x.rows)
    _ = prediction^


def _export(slot: _SlotP, call: _CallP) raises:
    if not slot[].fitted:
        raise Error("NotFittedError: this QNRegressor member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= 4:
            raise Error("member ABI: a QNRegressor member state has 4 fitted blocks")
        if block == 0:
            block = MEMBER_BLOCK_COEF
        elif block == 1:
            block = MEMBER_BLOCK_INTERCEPT
        elif block == 2:
            block = MEMBER_BLOCK_SCALARS
        else:
            block = QN_REGRESSOR_BLOCK_W
    if block == MEMBER_BLOCK_COEF:
        call[].dst = MemberMatrix(Int(slot[].coef.unsafe_ptr()), 1, slot[].features)
    elif block == MEMBER_BLOCK_INTERCEPT:
        call[].dst = MemberMatrix(Int(slot[].intercept.unsafe_ptr()), 1, 1)
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, 3, MEMBER_F64)
    elif block == QN_REGRESSOR_BLOCK_W:
        call[].dst = MemberMatrix(Int(slot[].w.unsafe_ptr()), 1, len(slot[].w))
    else:
        raise Error("member ABI: a QNRegressor member state holds no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 4


def _import(slot: _SlotP, call: _CallP) raises:
    """W (1 x (features + fit_intercept)) and SCALARS [n_iter, objective,
    retcode], then DONE."""
    var extra = 1 if slot[].params[1] != 0.0 else 0
    if call[].block == QN_REGRESSOR_BLOCK_W:
        _dense(call[].x, "the W block")
        var count = call[].x.rows * call[].x.cols
        if count < extra:
            raise Error("member ABI: the QNRegressor W block is features + fit_intercept float32 words")
        slot[].fitted = False
        slot[].w = List[Float32](length=count, fill=0.0)
        if count > 0:
            unsafe_memcpy(dest=slot[].w.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        slot[].features = count - extra
    elif call[].block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != 3:
            raise Error("member ABI: the QNRegressor scalars block is 3 host float64 words [n_iter, objective, retcode]")
        slot[].fitted = False
        slot[].scalars = List[Float64](length=3, fill=0.0)
        unsafe_memcpy(dest=slot[].scalars.unsafe_ptr(), src=_F64P(unsafe_from_address=call[].x.data), count=3)
    elif call[].block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a QNRegressor member imports no fitted block " + String(call[].block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) != 3 or len(slot[].w) != slot[].features + extra:
        raise Error("member ABI: a fitted QNRegressor state needs its W and scalars blocks")
    _split(slot)
    slot[].fitted = True


def qn_regressor_member(operation: Int, address: Int) raises -> Int:
    """One member operation of a QNRegressor state (the Logistic family's
    entry calls this when `qn_regressor_member_owns`)."""
    var call = _CallP(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        if call[].state != 0:
            _ = _lookup(call[].state, False)
        var bits = _capabilities(call[].route.has_fifth)
        # MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE (lane vdf-scope-linear): the QN
        # entry of a GPU image reads THE FOLD X WORD (glm/estimator.mojo); `_fold_op`.
        if call[].route.device == DEVICE_GPU and member_fold_store_ready(call):
            bits |= MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE
        return bits
    if operation == MEMBER_OP_NEW:
        if call[].param_count != QN_REGRESSOR_MEMBER_PARAMS:
            raise Error("member ABI: a QNRegressor member takes " + String(QN_REGRESSOR_MEMBER_PARAMS) + " parameters")
        var params = List[Float64](length=QN_REGRESSOR_MEMBER_PARAMS, fill=0.0)
        unsafe_memcpy(dest=params.unsafe_ptr(), src=_F64P(unsafe_from_address=call[].params),
            count=QN_REGRESSOR_MEMBER_PARAMS)
        call[].state = _mint(params^)
        return 0
    if operation == MEMBER_OP_FREE:
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        if call[].state != 0:
            _ = _lookup(call[].state, False)
        if call[].flags != MEMBER_OP_PREDICT:
            member_unsupported(_ESTIMATOR, call[].flags)
        call[].out_cols = 1
        call[].out_dtype = MEMBER_F32
        call[].classes = 0
        return 0
    var slot = _SlotP(unsafe_from_address=_lookup(call[].state, False))
    if (call[].row_index.rows > 0 and call[].column_index.rows == 0 and call[].route.device == DEVICE_GPU
            and (operation == MEMBER_OP_FIT or operation == MEMBER_OP_PREDICT or operation == MEMBER_OP_SCORE)):
        _fold_op(slot, call, operation)
        return 0
    _op(slot, call, operation)
    return 0


def _op(slot: _SlotP, call: _CallP, operation: Int) raises:
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_PREDICT:
        _predict(slot, call)
    elif operation == MEMBER_OP_SCORE:
        _score(slot, call)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_ESTIMATOR, operation)


def _fold_op(slot: _SlotP, call: _CallP, operation: Int) raises:
    """FIT / PREDICT / SCORE over the rows `call.row_index` of the WHOLE
    host X (lane vdf-scope-linear; core/member_fold_x.mojo): X resident in
    the caller's scope, the fold's rows taken on the device and read by the
    QN entry (THE FOLD X WORD); y and the weights are the base's columns
    taken bitwise through the index on the host. Views restored before this
    returns or raises."""
    var x = call[].x
    var y = call[].y
    var weights = call[].weights
    var index = call[].row_index
    var fold = MemberFoldX()
    var host_y = List[Float32]()
    var host_w = List[Float32]()
    var failure = String("")
    try:
        member_fold_x_open(call, fold, _ESTIMATOR)
        call[].x = MemberMatrix(fold.word(), fold.rows, fold.cols)
        if operation != MEMBER_OP_PREDICT and y.rows > 0:
            _dense(y, "y")
            if y.rows != x.rows:
                raise Error("mojolearn QNRegressor X and y lengths differ")
            host_y = List[Float32](length=max(fold.rows * y.cols, 1), fill=0.0)
            member_fold_take_view(y, fold.index, fold.rows, Int(host_y.unsafe_ptr()))
            call[].y = MemberMatrix(Int(host_y.unsafe_ptr()), fold.rows, y.cols)
        if weights.rows > 0:
            _dense(weights, "sample_weight")
            if weights.rows != x.rows or weights.cols != 1:
                raise Error("ValueError: sample_weight and X have different numbers of rows")
            host_w = List[Float32](length=fold.rows, fill=0.0)
            member_fold_take_view(weights, fold.index, fold.rows, Int(host_w.unsafe_ptr()))
            call[].weights = MemberMatrix(Int(host_w.unsafe_ptr()), fold.rows, 1)
        call[].row_index = MemberMatrix()
        _op(slot, call, operation)
    except error:
        failure = String(error)
    call[].x = x
    call[].y = y
    call[].weights = weights
    call[].row_index = index
    try:
        member_fold_x_close(fold)
    except close_error:
        if failure.byte_length() == 0:
            failure = "RuntimeError: a QNRegressor member's fold slot was not freed: " + String(close_error)
    _ = host_y^
    _ = host_w^
    if failure.byte_length() > 0:
        raise Error(failure)

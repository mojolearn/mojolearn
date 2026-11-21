# SPDX-License-Identifier: Apache-2.0
"""Ridge and LinearRegression as MEMBERS (core/member_abi.mojo): the worked
publication of the member ABI.

One entry, `linear_member_v2`, published by each estimators image
(`linear_member_publish`, beside its kernel table). It runs the family's
existing typed RUN (core/linear_fit_run.mojo, core/linear_ols_run.mojo,
core/linear_boundary_run.mojo, core/linear_score_run.mojo) over the entries
of the `DeviceRoute` it is handed, the same slots the estimator's own doors
read: `entry` the fit table, `second` predict, `third` the weighted /
one-column Ridge solve, `fourth` the decomposition entry of the TSQR
LinearRegression fit, `fifth` the metrics entry of the route's device (the
caller fills it when it will ask for `score`). No kernel is touched.

MEMBER_OP_NEW parameters (Float64 words): [0] kind (0 LinearRegression,
1 Ridge), [1] fit_intercept (0 / 1), [2] alpha (Ridge; ignored for kind 0).
Views: X dense row-major host float32; y and sample weights host float32
vectors (one target). Anything else is refused in words (a device-home
view, a row or column index: the caller hands over gathered host words); the
caller's UNPACK admits the data (finite, shapes) exactly as the estimator's
own door does. `call.fit_output = MEMBER_OP_PREDICT` writes the training
rows' predictions to `dst` in the fit call. A weighted score is not
published (no MEMBER_CAN_WEIGHTED_SCORE): the caller predicts and runs its
own weighted metric.

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), in index order:
`MEMBER_BLOCK_COEF` float32 1 x features; `MEMBER_BLOCK_SCALARS` float64
[intercept, y mean]; `MEMBER_BLOCK_FAMILY` float32 1 x features, the column
means of the fit (zeros without an intercept). An import needs the first
two; the means default to zeros.

The fitted state (coefficients, column means, intercept) is OWNED here, in a
process-wide store of fixed chunks: a slot never moves once minted, so a fit
or predict runs on its slot without the store's lock; the lock covers only
mint, free and the handle lookup (two atomic operations per member call).
Members are independent: a caller runs folds and candidates on its own
workers, one handle each. Serial by nature here: the store's bookkeeping.
"""
from std.sys.info import size_of
from core.device_route import DeviceRoute
from core.entry_table import entry_ref_vendor, entry_ref_mode
from core.linear_boundary_run import linear_predict_run
from core.linear_fit_run import linear_ridge_run
from core.linear_fit_types import LinearFitRequest, LinearFitResult, LinearOlsPlan
from core.linear_model_state import LinearModelState, linear_state_predict_request
from core.linear_ols_run import linear_ols_plan, linear_ols_one_entry, linear_ols_run
from core.linear_score_run import linear_score_run
from std.memory import unsafe_memcpy
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_SCORE, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_SCORE, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_COEF,
    MEMBER_BLOCK_COEF, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_LINEAR
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216


struct LinearMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var ridge: Bool
    var intercept: Bool
    var alpha: Float64
    var state: LinearModelState
    var coef: List[Float32]
    var mean: List[Float32]
    var scalars: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.ridge = False
        self.intercept = True
        self.alpha = 0.0
        self.state = LinearModelState()
        self.coef = List[Float32]()
        self.mean = List[Float32]()
        self.scalars = List[Float64]()


struct LinearMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[LinearMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[LinearMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[LinearMemberStore, "MojolearnLinearMemberStoreV2", 2]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[LinearMemberSlot]()


def _mint(ridge: Bool, intercept: Bool, alpha: Float64) raises -> Int:
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live linear member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[LinearMemberSlot](length=_CHUNK, fill=LinearMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = MutPointer[LinearMemberSlot, MutUntrackedOrigin](unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].ridge = ridge
        slot[].intercept = intercept
        slot[].alpha = alpha
        slot[].state = LinearModelState()
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
        var slot = MutPointer[LinearMemberSlot, MutUntrackedOrigin](unsafe_from_address=address)
        if not slot[].live or slot[].generation != handle // _INDEX_SPAN:
            raise Error("NotFittedError: this member state was released or never created")
        if release:
            slot[].live = False
            slot[].state = LinearModelState()
            slot[].coef = List[Float32]()
            slot[].mean = List[Float32]()
            slot[].scalars = List[Float64]()
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _name(ridge: Bool) -> String:
    return String("Ridge") if ridge else String("LinearRegression")


def _dense(view: MemberMatrix, what: String, estimator: String) raises:
    if not view.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + estimator + " member reads " + what + " as a dense host float32 view")


def _fit(slot: MutPointer[LinearMemberSlot, MutUntrackedOrigin], call: MutPointer[MemberCall, MutUntrackedOrigin]) raises:
    var estimator = _name(slot[].ridge)
    var route = call[].route
    var rows = call[].x.rows
    var cols = call[].x.cols
    _whole(call, estimator)
    if call[].fit_output != 0 and call[].fit_output != MEMBER_OP_PREDICT:
        member_unsupported(estimator, call[].fit_output)
    _dense(call[].x, "X", estimator)
    _dense(call[].y, "y", estimator)
    if call[].y.cols != 1:
        raise Error("ValueError: a " + estimator + " member fits one target column")
    if call[].y.rows != rows:
        raise Error("ValueError: X and y have different numbers of rows")
    var weights = 0
    if call[].weights.rows > 0:
        _dense(call[].weights, "sample_weight", estimator)
        if call[].weights.rows != rows or call[].weights.cols != 1:
            raise Error("ValueError: sample_weight and X have different numbers of rows")
        weights = call[].weights.data
    slot[].state = LinearModelState()
    slot[].coef = List[Float32](length=cols, fill=0.0)
    slot[].mean = List[Float32](length=cols, fill=0.0)
    var request = LinearFitRequest(call[].x.data, call[].y.data, weights, Int(slot[].coef.unsafe_ptr()),
        Int(slot[].mean.unsafe_ptr()), rows, cols, slot[].intercept, slot[].alpha if slot[].ridge else 0.0)
    var result = LinearFitResult()
    if slot[].ridge:
        result = linear_ridge_run(route.entry.call, route.third.call, request)
    else:
        var plan = linear_ols_plan(route.entry.call, rows, cols, estimator)
        var decomp = route.entry.call
        if plan.tsqr:
            if not route.has_fourth:
                raise Error("ImportError: mojolearn: the LinearRegression member needs the decomposition entry on its route")
            decomp = route.fourth.call
        plan.one_entry = linear_ols_one_entry(plan, decomp, slot[].intercept, weights != 0)
        result = linear_ols_run(route.entry.call, decomp, request, plan, route.vendor)
    slot[].state.coef = Int(slot[].coef.unsafe_ptr())
    slot[].state.coef_count = cols
    slot[].state.features = cols
    slot[].state.intercept = result.intercept
    slot[].state.mean = Int(slot[].mean.unsafe_ptr())
    slot[].state.ymean = result.ymean
    slot[].state.has_coef = True
    slot[].state.fitted = True
    slot[].scalars = List[Float64](length=2, fill=0.0)
    slot[].scalars[0] = result.intercept
    slot[].scalars[1] = result.ymean
    if call[].fit_output == MEMBER_OP_PREDICT:
        _predict(slot, call)


def _whole(call: MutPointer[MemberCall, MutUntrackedOrigin], estimator: String) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads whole host views (no row or column index)")


def _export(slot: MutPointer[LinearMemberSlot, MutUntrackedOrigin], call: MutPointer[MemberCall, MutUntrackedOrigin]) raises:
    """A view of one fitted block. The slot owns the words; they stay valid
    until this state's next fit, import or free."""
    if not slot[].state.fitted:
        raise Error("NotFittedError: this " + _name(slot[].ridge) + " member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > 2:
            raise Error("member ABI: a linear member has three fitted blocks")
        block = MEMBER_BLOCK_COEF if block == 0 else (MEMBER_BLOCK_SCALARS if block == 1 else MEMBER_BLOCK_FAMILY)
    if block == MEMBER_BLOCK_COEF:
        call[].dst = MemberMatrix(Int(slot[].coef.unsafe_ptr()), 1, len(slot[].coef))
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block == MEMBER_BLOCK_FAMILY:
        call[].dst = MemberMatrix(Int(slot[].mean.unsafe_ptr()), 1, len(slot[].mean))
    else:
        raise Error("member ABI: a linear member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 3


def _import(slot: MutPointer[LinearMemberSlot, MutUntrackedOrigin], call: MutPointer[MemberCall, MutUntrackedOrigin]) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var estimator = _name(slot[].ridge)
    if call[].block == MEMBER_BLOCK_COEF or call[].block == MEMBER_BLOCK_FAMILY:
        _dense(call[].x, "a fitted block", estimator)
        var count = call[].x.rows * call[].x.cols
        var words = List[Float32](length=count, fill=0.0)
        unsafe_memcpy(dest=words.unsafe_ptr(),
            src=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=call[].x.data), count=count)
        if call[].block == MEMBER_BLOCK_COEF:
            slot[].state = LinearModelState()
            slot[].coef = words^
        else:
            slot[].mean = words^
    elif call[].block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != 2:
            raise Error("member ABI: the linear scalars block is two host float64 words (intercept, y mean)")
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].x.data)
        slot[].scalars = List[Float64](length=2, fill=0.0)
        slot[].scalars[0] = source[0]
        slot[].scalars[1] = source[1]
    elif call[].block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a linear member has no fitted block " + String(call[].block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var cols = len(slot[].coef)
    if cols == 0 or len(slot[].scalars) != 2:
        raise Error("member ABI: a fitted linear state needs its coefficient and scalars blocks")
    if len(slot[].mean) != cols:
        slot[].mean = List[Float32](length=cols, fill=0.0)
    slot[].state.coef = Int(slot[].coef.unsafe_ptr())
    slot[].state.coef_count = cols
    slot[].state.features = cols
    slot[].state.intercept = slot[].scalars[0]
    slot[].state.mean = Int(slot[].mean.unsafe_ptr())
    slot[].state.ymean = slot[].scalars[1]
    slot[].state.has_coef = True
    slot[].state.fitted = True


def _predict(slot: MutPointer[LinearMemberSlot, MutUntrackedOrigin], call: MutPointer[MemberCall, MutUntrackedOrigin]) raises:
    var estimator = _name(slot[].ridge)
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    _dense(call[].dst, "the prediction buffer", estimator)
    if call[].dst.rows != call[].x.rows or call[].dst.cols != 1:
        raise Error("ValueError: the prediction buffer holds one value per row of X")
    var request = linear_state_predict_request(slot[].state, call[].x.data, call[].dst.data,
        call[].x.rows, call[].x.cols, False, estimator)
    _ = linear_predict_run(call[].route.second.call, request)


def _score(slot: MutPointer[LinearMemberSlot, MutUntrackedOrigin], call: MutPointer[MemberCall, MutUntrackedOrigin]) raises:
    var estimator = _name(slot[].ridge)
    var route = call[].route
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    _dense(call[].y, "y", estimator)
    if call[].y.cols != 1:
        raise Error("ValueError: a " + estimator + " member scores one target column")
    if call[].weights.rows > 0:
        raise Error("NotImplementedError: mojolearn: a weighted score of a " + estimator
            + " member is not published; the caller predicts and runs its own weighted metric")
    if not route.has_fifth:
        raise Error("ImportError: mojolearn: the " + estimator + " member's score needs the metrics entry on its route")
    call[].value = linear_score_run(route.second.call, route.fifth.call, slot[].state, call[].x.data,
        call[].x.rows, call[].x.cols, call[].y.data, call[].y.rows, entry_ref_vendor(route.fifth),
        entry_ref_mode(route.fifth), estimator)


def _linear_member(operation: Int, address: Int) raises -> Int:
    var call = MutPointer[MemberCall, MutUntrackedOrigin](unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        return (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_SCORE | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
            | MEMBER_CAN_SAMPLE_WEIGHT | MEMBER_CAN_COEF)
    if operation == MEMBER_OP_NEW:
        if call[].param_count != 3 or not call[].params:
            raise Error("member ABI: a linear member takes three parameters (kind, fit_intercept, alpha)")
        var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)
        if words[0] != 0.0 and words[0] != 1.0:
            raise Error("member ABI: unknown linear member kind")
        if words[0] == 1.0 and words[2] < 0.0:
            raise Error("alpha must be non-negative, got " + String(words[2]))
        call[].state = _mint(words[0] == 1.0, words[1] != 0.0, words[2])
        return 0
    if operation == MEMBER_OP_FREE:
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        call[].out_cols = 1
        call[].classes = 0
        return 0
    var slot = MutPointer[LinearMemberSlot, MutUntrackedOrigin](unsafe_from_address=_lookup(call[].state, False))
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
        member_unsupported(_name(slot[].ridge), operation)
    return 0


def linear_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of Ridge and LinearRegression (schema `MEMBER_ABI`)."""
    try:
        var word = _linear_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def linear_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's linear member entry. One line in the image's
    module initialization, after its linear kernel table is published."""
    member_publish[MEMBER_FAMILY_LINEAR](mode, vendor, publisher, linear_member_v2)

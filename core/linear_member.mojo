# SPDX-License-Identifier: Apache-2.0
"""Ridge and LinearRegression as MEMBERS (core/member_abi.mojo): the worked
publication of the member ABI.

One entry, `linear_member_v1`, published by each estimators image
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
vectors (one target). Anything else is refused in words; the caller's UNPACK
admits the data (finite, shapes) exactly as the estimator's own door does.

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
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_FAMILY_LINEAR,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_SCORE, MEMBER_OP_OUTPUTS, MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_SCORE,
    member_publish, member_call_from_spans, member_unsupported,
)
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

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.ridge = False
        self.intercept = True
        self.alpha = 0.0
        self.state = LinearModelState()
        self.coef = List[Float32]()
        self.mean = List[Float32]()


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


comptime _STORE = SharedSlot[LinearMemberStore, "MojolearnLinearMemberStoreV1", 1]


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


def _predict(slot: MutPointer[LinearMemberSlot, MutUntrackedOrigin], call: MutPointer[MemberCall, MutUntrackedOrigin]) raises:
    var estimator = _name(slot[].ridge)
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
        return MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_SCORE
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
        return 0
    var slot = MutPointer[LinearMemberSlot, MutUntrackedOrigin](unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_PREDICT:
        _predict(slot, call)
    elif operation == MEMBER_OP_SCORE:
        _score(slot, call)
    else:
        member_unsupported(_name(slot[].ridge), operation)
    return 0


def linear_member_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
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
    member_publish[MEMBER_FAMILY_LINEAR](mode, vendor, publisher, linear_member_v1)

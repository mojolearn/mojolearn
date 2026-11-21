# SPDX-License-Identifier: Apache-2.0
"""The XLinear family as MEMBERS (core/member_abi.mojo, schema v2): the
generalized linear regressors PoissonRegressor, GammaRegressor and
TweedieRegressor.

One entry, `xlinear_member_v2`, published by each x_linear image
(`xlinear_member_publish`, beside its operations table). It runs the family's
existing typed RUN (x_linear/door_run.mojo `xl_fit_run`, `xl_decision_run`,
`xl_r2_run`) over the entries of the `DeviceRoute` the estimator's own doors
resolve: `entry` the x_linear operations entry (fit and decision), `fifth`
the metrics image's R^2 entry of the route's tier and device (filled by the
caller's admission when that image is loaded; without it score is refused in
words and the caller predicts and runs its own metric). No kernel is touched.

MEMBER_OP_NEW parameters (Float64 words, count 7): [0] kind (0
PoissonRegressor, 1 GammaRegressor, 2 TweedieRegressor), [1] fit_intercept
(0 / 1), [2] max_iter, [3] link (0 identity, 1 log; Poisson and Gamma: 1),
[4] power (Poisson 1, Gamma 2, Tweedie its own), [5] alpha, [6] tol. The
estimator's own checks (solver, warm_start, alpha >= 0, Tweedie's power and
link words) run at the caller's admission.

Views: X dense row-major host float32; y and sample weights host float32
vectors (one target). A device-home view, a row or column index and a nonzero
`fit_output` are refused in words. PREDICT writes link(X @ coef + intercept)
as float32, rows x 1. SCORE is R^2 of that prediction (unweighted; a weighted
score is not published). Targets outside the loss's range are refused after
the fit, in the estimator's words.

Fitted-state blocks (EXPORT / IMPORT), in index order:
0 `MEMBER_BLOCK_COEF` float32 1 x features; 1 `MEMBER_BLOCK_SCALARS` float64
[intercept, n_iter]. An import needs both.

The fitted state is OWNED here, in a process-wide store of fixed chunks (the
linear member's form); the lock covers only mint, free and the handle lookup.
Serial by nature here: the store's bookkeeping.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from core.entry_table import entry_ref_unresolved
from core.linear_entry_table_ext import XL_OP_DECISION
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_SCORE, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_SCORE, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_COEF,
    MEMBER_BLOCK_COEF, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_XLINEAR
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from x_linear.door_run import (
    ALGO_GLM, LINK_IDENTITY, LINK_EXP, XlFitRequest, XlDecisionRequest, xl_fit_run, xl_decision_run, xl_r2_run,
)

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _PARAMS = 7
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]


struct XLinearMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var params: List[Float64]
    var fitted: Bool
    var features: Int
    # [coef (features) | intercept]: the decision block of one target.
    var block: List[Float32]
    var scalars: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.params = List[Float64]()
        self.fitted = False
        self.features = 0
        self.block = List[Float32]()
        self.scalars = List[Float64]()

    def forget(mut self):
        self.fitted = False
        self.features = 0
        self.block = List[Float32]()
        self.scalars = List[Float64]()


struct XLinearMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[XLinearMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[XLinearMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[XLinearMemberStore, "MojolearnXLinearMemberStoreV2", 2]
comptime _SlotP = MutPointer[XLinearMemberSlot, MutUntrackedOrigin]
comptime _CallP = MutPointer[MemberCall, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock; a chunk never grows, so the address stays valid.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[XLinearMemberSlot]()


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
                raise Error("RuntimeError: mojolearn: too many live x_linear member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[XLinearMemberSlot](length=_CHUNK, fill=XLinearMemberSlot()))
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


def _name(slot: _SlotP) -> String:
    var kind = Int(slot[].params[0])
    return String("PoissonRegressor") if kind == 0 else (
        String("GammaRegressor") if kind == 1 else String("TweedieRegressor"))


def _loss_name(slot: _SlotP) -> String:
    # The class attribute `_loss_name` (Tweedie: None -> "HalfTweedieLoss").
    var kind = Int(slot[].params[0])
    return String("HalfPoissonLoss") if kind == 0 else (
        String("HalfGammaLoss") if kind == 1 else String("HalfTweedieLoss"))


def _dense(view: MemberMatrix, what: String, estimator: String) raises:
    if not view.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + estimator + " member reads " + what + " as a dense host float32 view")


def _whole(call: _CallP, estimator: String) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads whole host views (no row or column index)")


def _fit(slot: _SlotP, call: _CallP) raises:
    var estimator = _name(slot)
    _whole(call, estimator)
    if call[].fit_output != 0:
        member_unsupported(estimator, call[].fit_output)
    _dense(call[].x, "X", estimator)
    _dense(call[].y, "y", estimator)
    var n = call[].x.rows
    var d = call[].x.cols
    if call[].y.cols != 1:
        raise Error("ValueError: a " + estimator + " member fits one target column")
    if call[].y.rows != n:
        raise Error("ValueError: X and y have different numbers of rows")
    # `_with_weights`: the targets, then the weights, in one float32 vector.
    var y = call[].y.data
    var y_size = n
    var has_sw = 0
    var joined = List[Float32]()
    if call[].weights.rows > 0:
        _dense(call[].weights, "sample_weight", estimator)
        if call[].weights.rows != n or call[].weights.cols != 1:
            raise Error("ValueError: sample_weight and X have different numbers of rows")
        joined = List[Float32](length=2 * n, fill=0.0)
        unsafe_memcpy(dest=joined.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].y.data), count=n)
        unsafe_memcpy(dest=joined.unsafe_ptr() + n, src=_F32P(unsafe_from_address=call[].weights.data), count=n)
        y = Int(joined.unsafe_ptr())
        y_size = 2 * n
        has_sw = 1
    var p = slot[].params.copy()
    var m = d + 1
    var ip: List[Int] = [Int(p[2]), 1 if p[1] != 0.0 else 0, Int(p[3]), has_sw]
    var fp: List[Float64] = [p[4], p[5], p[6]]
    slot[].forget()
    var result = List[Float32](length=d + 3, fill=0.0)
    var request = XlFitRequest(call[].route.entry.call, ALGO_GLM, call[].x.data, n * d, n, d, y, y_size, ip^, fp^,
        d + 3, 3 * n + m * m + 3 * m, 1, Int(result.unsafe_ptr()), False, 0, 0, 0)
    _ = xl_fit_run(request)
    _ = joined^
    if result[d + 2] < 0:
        raise Error("Some value(s) of y are out of the valid range of the loss '" + _loss_name(slot) + "'.")
    slot[].block = List[Float32](length=d + 1, fill=0.0)
    unsafe_memcpy(dest=slot[].block.unsafe_ptr(), src=result.unsafe_ptr(), count=d + 1)
    slot[].scalars = [Float64(result[d]), Float64(Int(result[d + 1]))]
    slot[].features = d
    slot[].fitted = True


def _predict_into(slot: _SlotP, call: _CallP, dst: Int) raises:
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].x.cols != slot[].features:
        raise Error("ValueError: X has " + String(call[].x.cols) + " features, but " + estimator
            + " is expecting " + String(slot[].features) + " features as input.")
    if call[].x.rows == 0:
        return
    var link = LINK_EXP if slot[].params[3] == 1.0 else LINK_IDENTITY
    xl_decision_run(XlDecisionRequest(call[].route.entry.call, XL_OP_DECISION, call[].x.data,
        Int(slot[].block.unsafe_ptr()), dst, call[].x.rows, call[].x.cols, 1, link, True, 0, 0, 0, 0))


def _predict(slot: _SlotP, call: _CallP) raises:
    _dense(call[].dst, "the prediction buffer", _name(slot))
    if call[].dst.rows != call[].x.rows or call[].dst.cols != 1:
        raise Error("ValueError: the prediction buffer holds one value per row of X")
    _predict_into(slot, call, call[].dst.data)


def _score(slot: _SlotP, call: _CallP) raises:
    var estimator = _name(slot)
    var route = call[].route
    _dense(call[].y, "y", estimator)
    if call[].y.cols != 1 or call[].y.rows != call[].x.rows:
        raise Error("ValueError: y must be 1-D with one entry per row of X")
    if call[].weights.rows > 0:
        raise Error("NotImplementedError: mojolearn: a weighted score of a " + estimator
            + " member is not published; the caller predicts and runs its own weighted metric")
    if not route.has_fifth:
        raise Error("ImportError: mojolearn: the " + estimator + " member's score needs the metrics R^2 entry on its route")
    var prediction = List[Float32](length=call[].x.rows, fill=0.0)
    _predict_into(slot, call, Int(prediction.unsafe_ptr()))
    call[].value = xl_r2_run(route.fifth.call, entry_ref_unresolved(), call[].y.data, Int(prediction.unsafe_ptr()),
        0, call[].x.rows, True)
    _ = prediction^


def _export(slot: _SlotP, call: _CallP) raises:
    """A view of one fitted block, owned by the slot until its next fit,
    import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot) + " member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > 1:
            raise Error("member ABI: an x_linear GLM member has two fitted blocks")
        block = MEMBER_BLOCK_COEF if block == 0 else MEMBER_BLOCK_SCALARS
    if block == MEMBER_BLOCK_COEF:
        call[].dst = MemberMatrix(Int(slot[].block.unsafe_ptr()), 1, slot[].features)
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    else:
        raise Error("member ABI: an x_linear GLM member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 2


def _import(slot: _SlotP, call: _CallP) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    if call[].block == MEMBER_BLOCK_COEF:
        _dense(call[].x, "a fitted block", _name(slot))
        var count = call[].x.rows * call[].x.cols
        var words = List[Float32](length=count + 1, fill=0.0)
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        slot[].fitted = False
        slot[].features = count
        slot[].block = words^
    elif call[].block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != 2:
            raise Error("member ABI: the x_linear GLM scalars block is two host float64 words (intercept, n_iter)")
        var source = _F64P(unsafe_from_address=call[].x.data)
        slot[].fitted = False
        slot[].scalars = [source[0], source[1]]
    elif call[].block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: an x_linear GLM member has no fitted block " + String(call[].block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].block) == 0 or len(slot[].scalars) != 2:
        raise Error("member ABI: a fitted x_linear GLM state needs its coefficient and scalars blocks")
    slot[].block[slot[].features] = Float32(slot[].scalars[0])
    slot[].fitted = True


def _xlinear_member(operation: Int, address: Int) raises -> Int:
    var call = _CallP(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        return (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_SCORE | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
            | MEMBER_CAN_SAMPLE_WEIGHT | MEMBER_CAN_COEF)
    if operation == MEMBER_OP_NEW:
        if call[].param_count != _PARAMS or not call[].params:
            raise Error("member ABI: an x_linear GLM member takes seven parameters")
        var words = _F64P(unsafe_from_address=call[].params)
        if words[0] != 0.0 and words[0] != 1.0 and words[0] != 2.0:
            raise Error("member ABI: unknown x_linear member kind")
        if words[3] != 0.0 and words[3] != 1.0:
            raise Error("member ABI: an x_linear GLM link word is 0 (identity) or 1 (log)")
        var params = List[Float64](length=_PARAMS, fill=0.0)
        for i in range(_PARAMS):  # driver-loop: over the seven parameter words
            params[i] = words[i]
        call[].state = _mint(params^)
        return 0
    if operation == MEMBER_OP_FREE:
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        call[].out_cols = 1
        call[].classes = 0
        return 0
    var slot = _SlotP(unsafe_from_address=_lookup(call[].state, False))
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
        member_unsupported(_name(slot), operation)
    return 0


def xlinear_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of the x_linear GLM regressors (schema `MEMBER_ABI`)."""
    try:
        var word = _xlinear_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def xlinear_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's x_linear member entry. One line in the image's
    module initialization, after its operations table is published."""
    member_publish[MEMBER_FAMILY_XLINEAR](mode, vendor, publisher, xlinear_member_v2)

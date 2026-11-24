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
vectors (one target). THE 2-D y (lane w3f3-k; Ridge only, MEMBER_CAN_MULTI_OUTPUT
on its kind, as scikit-learn's Ridge): y rows x targets, fitted in ONE solve
for every target (core/linear_fit_run.mojo `linear_ridge_targets_run`, the
estimator's own multi-target route), predict rows x targets, score the
targets' averaged R^2; the COEF block is targets x features and the SCALARS
block the targets' intercepts then the y mean. Anything else is refused in words (a device-home
view, a row or column index: the caller hands over gathered host words); the
caller's UNPACK admits the data (finite, shapes) exactly as the estimator's
own door does (MEMBER_CAN_TEST_ROWS clear: the fit and predict entries do
not test finiteness). OUTPUTS answers PREDICT only: one float32 column
(`out_dtype` MEMBER_F32). `call.fit_output = MEMBER_OP_PREDICT` writes the training
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
from core.linear_fit_run import linear_ridge_targets_run
from core.linear_boundary_run import linear_predict_targets_run, linear_r2_targets
from std.memory import unsafe_memcpy
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_SCORE, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_SCORE, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_COEF, MEMBER_GIL_FREE, MEMBER_CAN_MULTI_OUTPUT,
    MEMBER_CAN_ROW_INDEX, MEMBER_CAN_SCOPE,
    MEMBER_BLOCK_COEF, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_LINEAR
from core.linear_fit_types import LF_FLAGS, LF_HAS_SLOT_X
from core.native_descriptor_call import native_descriptor_call
from core.member_fold_x import (
    MemberFoldX, member_fold_x_open, member_fold_x_close, member_fold_index, member_fold_take_f32,
    member_fold_store_ready,
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
    var scalars: List[Float64]
    # THE 2-D y (lane w3f3-k): the target count (coef is targets x features,
    # scalars the targets' intercepts then the y mean), the intercepts as
    # the multi-target predict reads them, and the score's predictions
    # (grown, never shrunk: no allocation per call).
    var targets: Int
    var intercepts: List[Float64]
    var predictions: List[Float32]

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
        self.targets = 1
        self.intercepts = List[Float64]()
        self.predictions = List[Float32]()


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
        slot[].targets = 1
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
            slot[].intercepts = List[Float64]()
            slot[].predictions = List[Float32]()
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
    if call[].row_index.rows > 0 and call[].column_index.rows == 0:
        _fold_op(slot, call, MEMBER_OP_FIT)
        return
    _fit_whole(slot, call)


def _fit_whole(slot: MutPointer[LinearMemberSlot, MutUntrackedOrigin], call: MutPointer[MemberCall, MutUntrackedOrigin]) raises:
    var estimator = _name(slot[].ridge)
    var route = call[].route
    var rows = call[].x.rows
    var cols = call[].x.cols
    _whole(call, estimator)
    if call[].fit_output != 0 and call[].fit_output != MEMBER_OP_PREDICT:
        member_unsupported(estimator, call[].fit_output)
    _dense(call[].x, "X", estimator)
    _dense(call[].y, "y", estimator)
    if call[].y.cols != 1 and not (slot[].ridge and call[].y.cols > 1):
        raise Error("ValueError: a " + estimator + " member fits one target column")
    if call[].y.rows != rows:
        raise Error("ValueError: X and y have different numbers of rows")
    var weights = 0
    if call[].weights.rows > 0:
        _dense(call[].weights, "sample_weight", estimator)
        if call[].weights.rows != rows or call[].weights.cols != 1:
            raise Error("ValueError: sample_weight and X have different numbers of rows")
        weights = call[].weights.data
    if call[].y.cols > 1:
        _fit_targets(slot, call, weights)
        return
    slot[].targets = 1
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


def _fit_targets(slot: MutPointer[LinearMemberSlot, MutUntrackedOrigin],
                 call: MutPointer[MemberCall, MutUntrackedOrigin], weights: Int) raises:
    """THE 2-D y (lane w3f3-k): Ridge on every target of a rows x targets y
    in one solve (`linear_ridge_targets_run`, the estimator's own
    multi-target route through the route's `third` entry)."""
    var route = call[].route
    var rows = call[].x.rows
    var cols = call[].x.cols
    var targets = call[].y.cols
    if not route.has_third:
        raise Error("ImportError: mojolearn: the Ridge member's multi-target fit needs the weighted solve entry on its route")
    slot[].state = LinearModelState()
    slot[].coef = List[Float32](length=targets * cols, fill=0.0)
    slot[].mean = List[Float32](length=cols, fill=0.0)
    slot[].intercepts = List[Float64](length=targets, fill=0.0)
    var request = LinearFitRequest(call[].x.data, call[].y.data, weights, Int(slot[].coef.unsafe_ptr()),
        Int(slot[].mean.unsafe_ptr()), rows, cols, slot[].intercept, slot[].alpha)
    linear_ridge_targets_run(route.third.call, request, targets, Int(slot[].intercepts.unsafe_ptr()))
    slot[].targets = targets
    _targets_state(slot, cols)
    if call[].fit_output == MEMBER_OP_PREDICT:
        _predict(slot, call)


def _targets_state(slot: MutPointer[LinearMemberSlot, MutUntrackedOrigin], cols: Int):
    """The fitted state of a multi-target slot (lane w3f3-k): the state
    words over its targets x cols coefficients, the SCALARS block (the
    intercepts, then a zero y mean)."""
    var targets = slot[].targets
    slot[].state.coef = Int(slot[].coef.unsafe_ptr())
    slot[].state.coef_count = cols * targets
    slot[].state.features = cols
    slot[].state.targets = targets
    slot[].state.intercept = 0.0
    slot[].state.mean = Int(slot[].mean.unsafe_ptr())
    slot[].state.ymean = 0.0
    slot[].state.has_coef = True
    slot[].state.fitted = True
    slot[].scalars = List[Float64](length=targets + 1, fill=0.0)
    for t in range(targets):  # small-loop(the targets' intercepts)
        slot[].scalars[t] = slot[].intercepts[t]


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
        # targets x features (one row for one target; lane w3f3-k)
        call[].dst = MemberMatrix(Int(slot[].coef.unsafe_ptr()), slot[].targets, len(slot[].coef) // slot[].targets)
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
            # one row per target (lane w3f3-k)
            slot[].targets = max(call[].x.rows, 1)
        else:
            slot[].mean = words^
    elif call[].block == MEMBER_BLOCK_SCALARS:
        var count = call[].x.rows * call[].x.cols
        if not call[].x.dense_host(MEMBER_F64) or count < 2:
            raise Error("member ABI: the linear scalars block is host float64 words (each target's intercept, y mean)")
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].x.data)
        slot[].scalars = List[Float64](length=count, fill=0.0)
        for i in range(count):  # small-loop(the targets' intercepts and the y mean)
            slot[].scalars[i] = source[i]
    elif call[].block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a linear member has no fitted block " + String(call[].block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var targets = slot[].targets
    var cols = len(slot[].coef) // targets
    if cols == 0 or len(slot[].scalars) != targets + 1 or cols * targets != len(slot[].coef):
        raise Error("member ABI: a fitted linear state needs its coefficient and scalars blocks")
    if len(slot[].mean) != cols:
        slot[].mean = List[Float32](length=cols, fill=0.0)
    if targets > 1:
        # THE 2-D y (lane w3f3-k)
        slot[].intercepts = List[Float64](length=targets, fill=0.0)
        for t in range(targets):  # small-loop(the targets' intercepts)
            slot[].intercepts[t] = slot[].scalars[t]
        _targets_state(slot, cols)
        return
    slot[].state.coef = Int(slot[].coef.unsafe_ptr())
    slot[].state.coef_count = cols
    slot[].state.features = cols
    slot[].state.intercept = slot[].scalars[0]
    slot[].state.mean = Int(slot[].mean.unsafe_ptr())
    slot[].state.ymean = slot[].scalars[1]
    slot[].state.has_coef = True
    slot[].state.fitted = True


def _predict(slot: MutPointer[LinearMemberSlot, MutUntrackedOrigin], call: MutPointer[MemberCall, MutUntrackedOrigin]) raises:
    if call[].row_index.rows > 0 and call[].column_index.rows == 0:
        _fold_op(slot, call, MEMBER_OP_PREDICT)
        return
    _predict_whole(slot, call)


def _predict_whole(slot: MutPointer[LinearMemberSlot, MutUntrackedOrigin], call: MutPointer[MemberCall, MutUntrackedOrigin]) raises:
    var estimator = _name(slot[].ridge)
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    _dense(call[].dst, "the prediction buffer", estimator)
    if slot[].targets > 1:
        _predict_targets(slot, call, call[].dst.data)
        return
    if call[].dst.rows != call[].x.rows or call[].dst.cols != 1:
        raise Error("ValueError: the prediction buffer holds one value per row of X")
    var request = linear_state_predict_request(slot[].state, call[].x.data, call[].dst.data,
        call[].x.rows, call[].x.cols, estimator)
    _ = linear_predict_run(call[].route.second.call, request)


def _predict_targets(slot: MutPointer[LinearMemberSlot, MutUntrackedOrigin],
                     call: MutPointer[MemberCall, MutUntrackedOrigin], dst: Int) raises:
    """THE 2-D y's predict (lane w3f3-k): rows x targets float32 at `dst`
    (one call of the route's prediction entry with the T word), with
    the one-target predict's width check."""
    var estimator = _name(slot[].ridge)
    var rows = call[].x.rows
    var cols = call[].x.cols
    var targets = slot[].targets
    if not slot[].state.fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    if cols != slot[].state.features:
        raise Error("X has " + String(cols) + " features, but " + estimator + " is expecting "
            + String(slot[].state.features) + " features as input.")
    if dst == call[].dst.data and (call[].dst.rows != rows or call[].dst.cols != targets):
        raise Error("ValueError: the prediction buffer holds one value per row of X and target")
    _ = linear_predict_targets_run(call[].route.second.call, call[].x.data, Int(slot[].coef.unsafe_ptr()),
        slot[].intercepts, dst, rows, cols, targets)


def _score(slot: MutPointer[LinearMemberSlot, MutUntrackedOrigin], call: MutPointer[MemberCall, MutUntrackedOrigin]) raises:
    if call[].row_index.rows > 0 and call[].column_index.rows == 0:
        _fold_op(slot, call, MEMBER_OP_SCORE)
        return
    _score_whole(slot, call)


def _score_whole(slot: MutPointer[LinearMemberSlot, MutUntrackedOrigin], call: MutPointer[MemberCall, MutUntrackedOrigin]) raises:
    var estimator = _name(slot[].ridge)
    var route = call[].route
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    _dense(call[].y, "y", estimator)
    if slot[].targets > 1:
        # THE 2-D y (lane w3f3-k): r2_score's uniform average over the targets.
        if call[].y.cols != slot[].targets or call[].y.rows != call[].x.rows:
            raise Error("ValueError: a " + estimator + " member scores its " + String(slot[].targets) + " targets")
        if call[].weights.rows > 0:
            raise Error("NotImplementedError: mojolearn: a weighted score of a " + estimator
                + " member is not published; the caller predicts and runs its own weighted metric")
        var count = call[].x.rows * slot[].targets
        if len(slot[].predictions) < count:
            slot[].predictions = List[Float32](length=count, fill=0.0)
        _predict_targets(slot, call, Int(slot[].predictions.unsafe_ptr()))
        call[].value = linear_r2_targets(call[].y.data, Int(slot[].predictions.unsafe_ptr()), call[].x.rows,
            slot[].targets)
        return
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


def _route_slot_x(call: MutPointer[MemberCall, MutUntrackedOrigin]) -> Bool:
    """THE FOLD X WORD on this route (core/fold_x_device.mojo): the image's
    LF_GRAM fit and prediction entry read a device fold slot
    (LF_HAS_SLOT_X) and the image ships the x_prep family store. Asked with
    the capabilities only (once per member class at the caller's UNPACK)."""
    try:
        var flags = native_descriptor_call(call[].route.entry.call, LF_FLAGS, List[Int](), List[Int](), List[Float64]())
        if (flags & LF_HAS_SLOT_X) == 0:
            return False
    except:
        return False
    return member_fold_store_ready(call)


def _fold_device(slot: MutPointer[LinearMemberSlot, MutUntrackedOrigin],
                 call: MutPointer[MemberCall, MutUntrackedOrigin], operation: Int) -> Bool:
    """Whether this operation reads the fold's X on the device: a Ridge
    with one target on a route with the fold word, and for a fit exactly
    the case its Gram route always finishes (no sample weights, alpha > 0,
    more than one row and one column; core/linear_fit_types.mojo
    LF_HAS_SLOT_X). Any other case takes the fold's rows on the host
    (`member_fold_take_f32`, across the host's cores): the entries of
    those routes read host X."""
    if not slot[].ridge or call[].x.cols <= 1:
        return False
    if operation == MEMBER_OP_FIT:
        if call[].weights.rows > 0 or slot[].alpha <= 0.0 or call[].row_index.rows <= 1 or call[].y.cols != 1:
            return False
    elif slot[].targets > 1:
        return False
    return _route_slot_x(call)


def _fold_op(slot: MutPointer[LinearMemberSlot, MutUntrackedOrigin],
             call: MutPointer[MemberCall, MutUntrackedOrigin], operation: Int) raises:
    """FIT / PREDICT / SCORE over the rows `call.row_index` of the WHOLE
    host X (MEMBER_CAN_ROW_INDEX / MEMBER_CAN_SCOPE; core/member_fold_x.mojo):
    X resident in the scope's cache (or this call's slot), the fold's rows
    taken on the device and read there by the entry; y and the weights
    are the base's columns taken through the index on the host. The call's
    views are restored before this returns or raises."""
    var estimator = _name(slot[].ridge)
    var x = call[].x
    var y = call[].y
    var weights = call[].weights
    var index = call[].row_index
    var fold = MemberFoldX()
    var device = _fold_device(slot, call, operation)
    var host_x = List[Float32]()
    var host_y = List[Float32]()
    var host_w = List[Float32]()
    var failure = String("")
    try:
        if device:
            member_fold_x_open(call, fold, estimator)
            call[].x = MemberMatrix(fold.word(), fold.rows, fold.cols)
        else:
            member_fold_index(call, fold, estimator)
            host_x = List[Float32](unsafe_uninit_length=max(fold.rows * fold.cols, 1))
            member_fold_take_f32(x.data, fold.index, fold.rows, fold.cols, Int(host_x.unsafe_ptr()))
            call[].x = MemberMatrix(Int(host_x.unsafe_ptr()), fold.rows, fold.cols)
        if operation != MEMBER_OP_PREDICT and y.rows > 0:
            # the base's targets through the index (the fold's rows of y)
            _dense(y, "y", estimator)
            if y.rows != x.rows:
                raise Error("ValueError: X and y have different numbers of rows")
            host_y = List[Float32](unsafe_uninit_length=max(fold.rows * y.cols, 1))
            member_fold_take_f32(y.data, fold.index, fold.rows, y.cols, Int(host_y.unsafe_ptr()))
            call[].y = MemberMatrix(Int(host_y.unsafe_ptr()), fold.rows, y.cols)
        if weights.rows > 0:
            _dense(weights, "sample_weight", estimator)
            if weights.rows != x.rows or weights.cols != 1:
                raise Error("ValueError: sample_weight and X have different numbers of rows")
            host_w = List[Float32](unsafe_uninit_length=fold.rows)
            member_fold_take_f32(weights.data, fold.index, fold.rows, 1, Int(host_w.unsafe_ptr()))
            call[].weights = MemberMatrix(Int(host_w.unsafe_ptr()), fold.rows, 1)
        call[].row_index = MemberMatrix()
        if operation == MEMBER_OP_FIT:
            _fit_whole(slot, call)
        elif operation == MEMBER_OP_PREDICT:
            _predict_whole(slot, call)
        else:
            _score_whole(slot, call)
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
            failure = "RuntimeError: a " + estimator + " member's fold slot was not freed: " + String(close_error)
    _ = host_x^
    _ = host_y^
    _ = host_w^
    if failure.byte_length() > 0:
        raise Error(failure)


def _linear_member(operation: Int, address: Int) raises -> Int:
    var call = MutPointer[MemberCall, MutUntrackedOrigin](unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        # MEMBER_GIL_FREE (core/member_abi.mojo, THE GIL RULE; the trace is in
        # fx-gil-RESULT.md, round 4): both kinds run on parallel workers with
        # no GIL. The store has its own lock; FIT, PREDICT and SCORE call
        # `linear_ridge_run` / `linear_ols_run`, `linear_predict_run` and
        # `linear_score_run` on per-call buffers; the entries behind the
        # route (the linear fit and predict entries, the weighted solve, the
        # decomposition entry's TSQR and kit, the metrics program) touch no
        # CPython and never the GIL, and every process-wide state they reach
        # has its own lock held for one list step or one first creation (the
        # process contexts, the decomposition pool and stage, the kept TSQR
        # state, which a temporary factor no longer touches). On a GPU route
        # the callers run one worker (several threads enqueueing on one
        # device context is not proven).
        comptime _BOTH = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_SCORE | MEMBER_CAN_EXPORT
            | MEMBER_CAN_IMPORT | MEMBER_CAN_SAMPLE_WEIGHT | MEMBER_CAN_COEF | MEMBER_GIL_FREE)
        # MEMBER_CAN_MULTI_OUTPUT (lane w3f3-k): Ridge's kind (a 2-D y, as
        # scikit-learn's Ridge), asked kind-exact with a live state; the
        # family's union (no state) includes it.
        # MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE (lane vdf-scope-linear): Ridge
        # on a route whose image reads THE FOLD X WORD (`_route_slot_x`): the
        # fold's rows are taken on the device out of X kept resident in the
        # caller's scope (`_fold_op`). LinearRegression keeps whole views (its
        # TSQR route reads host X). The family's union includes them on such a
        # route.
        comptime _FOLD = MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE
        if call[].state == 0:
            return _BOTH | MEMBER_CAN_MULTI_OUTPUT | (_FOLD if _route_slot_x(call) else 0)
        var kind = MutPointer[LinearMemberSlot, MutUntrackedOrigin](unsafe_from_address=_lookup(call[].state, False))
        if kind[].ridge:
            return _BOTH | MEMBER_CAN_MULTI_OUTPUT | (_FOLD if _route_slot_x(call) else 0)
        return _BOTH
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
        # One target column of float32 predictions, with or without a state;
        # an operation this member lacks is refused, never sized.
        if call[].flags != MEMBER_OP_PREDICT:
            member_unsupported("LinearRegression / Ridge", call[].flags)
        # THE 2-D y (lane w3f3-k): a multi-target Ridge's rows x targets;
        # without a state, the shape word `y.cols`.
        var targets = max(call[].y.cols, 1)
        if call[].state != 0:
            targets = MutPointer[LinearMemberSlot, MutUntrackedOrigin](
                unsafe_from_address=_lookup(call[].state, False))[].targets
        call[].out_cols = targets
        call[].out_dtype = MEMBER_F32
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

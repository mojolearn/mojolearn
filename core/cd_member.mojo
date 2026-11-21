# SPDX-License-Identifier: Apache-2.0
"""The Cd family as MEMBERS (core/member_abi.mojo, schema v2): the
coordinate-descent pair ElasticNet and Lasso.

One entry, `cd_member_v2`, published by each solver image
(`cd_member_publish`, beside its operations table). It runs the family's
existing typed RUN (x_linear/door_run.mojo `xl_cd_fit_run`,
`xl_cd_predict_run`, `xl_r2_run`) over the entries of the `DeviceRoute` the
estimators' own doors resolve (`xl_route(..., cd=True)`): `entry` the solver
operations entry (cd_fit, cd_predict), `fifth` the metrics image's R^2 entry
of the route's tier and device (filled by the caller's admission when that
image is loaded; without it score is refused in words and the caller
predicts and runs its own metric). No kernel is touched.

Kinds (the second word of each class's `_mojolearn_member`): 0 ElasticNet,
1 Lasso (ElasticNet with l1_ratio 1).

MEMBER_OP_NEW parameters, 7 Float64 words: [0] kind, [1] fit_intercept
(0 / 1), [2] max_iter (epochs), [3] alpha, [4] l1_ratio, [5] tol,
[6] shuffle (the selection word's code; 'cyclic': 0). The estimator's own
checks (`_xl_cd_validate`: alpha, selection, l1_ratio, solver, positive,
warm_start, precompute, copy_X, random_state) run at the caller's
admission.

Views: X dense row-major host float32, BORROWED at fit (cd_fit reads a
C-order design as it is, `row_major`; a one-column design is the same words
in either order); y and sample weights host float32 rows x 1 (read in
place). A fit needs at least two rows (cuML's floor, in its words).
PREDICT writes float32 rows x 1, X @ coef + intercept. cd_predict reads an
F-order design: a C-order X of more than one row and column is transposed
once into a scratch buffer on the host's workers (the own predict door's
admission makes the same F-order copy of a C-order input); otherwise X is
read in place. SCORE is R^2 of that prediction (unweighted). A device-home
view, a row or column index and a nonzero `fit_output` are refused in
words.

Fitted-state blocks (EXPORT / IMPORT), in index order: 0
`MEMBER_BLOCK_COEF` float32 1 x features; 1 `MEMBER_BLOCK_SCALARS` float64
[intercept, n_iter]. An import needs both.

The fitted state is OWNED here, in a process-wide store of fixed chunks (the
linear member's form); the lock covers only mint, free and the handle lookup.
Serial by nature here: the store's bookkeeping.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from core.entry_table import entry_ref_unresolved
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_SCORE, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_SCORE, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_COEF,
    MEMBER_BLOCK_COEF, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_CD
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from x_linear.door_run import XlCdFitRequest, XlCdPredictRequest, xl_cd_fit_run, xl_cd_predict_run, xl_r2_run

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime CD_MEMBER_PARAMS = 7
comptime CD_MEMBER_ELASTIC_NET = 0
comptime CD_MEMBER_LASSO = 1
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]


struct CdMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var params: List[Float64]
    var fitted: Bool
    var features: Int
    var coef: List[Float32]
    # [intercept (the float32 info word), n_iter]
    var scalars: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.params = List[Float64]()
        self.fitted = False
        self.features = 0
        self.coef = List[Float32]()
        self.scalars = List[Float64]()

    def forget(mut self):
        self.fitted = False
        self.features = 0
        self.coef = List[Float32]()
        self.scalars = List[Float64]()


struct CdMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[CdMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[CdMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[CdMemberStore, "MojolearnCdMemberStoreV2", 2]
comptime _SlotP = MutPointer[CdMemberSlot, MutUntrackedOrigin]
comptime _CallP = MutPointer[MemberCall, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock; a chunk never grows, so the address stays valid.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[CdMemberSlot]()


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
                raise Error("RuntimeError: mojolearn: too many live coordinate-descent member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[CdMemberSlot](length=_CHUNK, fill=CdMemberSlot()))
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
    return String("Lasso") if Int(slot[].params[0]) == CD_MEMBER_LASSO else String("ElasticNet")


def _capabilities() -> Int:
    return (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_SCORE | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
        | MEMBER_CAN_SAMPLE_WEIGHT | MEMBER_CAN_COEF)


def _dense(view: MemberMatrix, what: String, estimator: String) raises:
    if not view.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + estimator + " member reads " + what + " as a dense host float32 view")


def _whole(call: _CallP, estimator: String) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads whole host views (no row or column index)")


def _fit(slot: _SlotP, call: _CallP) raises:
    """`xl_cd_fit_unpack`'s request over the borrowed C-order design."""
    var estimator = _name(slot)
    _whole(call, estimator)
    if call[].fit_output != 0:
        member_unsupported(estimator, call[].fit_output)
    _dense(call[].x, "X", estimator)
    _dense(call[].y, "y", estimator)
    var n = call[].x.rows
    var d = call[].x.cols
    if n < 2:
        raise Error("Found array with " + String(n) + " sample(s) (shape=(" + String(n) + ", "
            + String(d) + ")) while a minimum of 2 is required.")
    if call[].y.cols != 1:
        raise Error("mojolearn ElasticNet currently requires one target")
    if call[].y.rows != n:
        raise Error("mojolearn ElasticNet X and y lengths differ")
    var sw = 0
    if call[].weights.rows > 0:
        _dense(call[].weights, "sample_weight", estimator)
        if call[].weights.rows != n or call[].weights.cols != 1:
            raise Error("ValueError: sample_weight and X have different numbers of rows")
        sw = call[].weights.data
    var p = slot[].params.copy()
    slot[].forget()
    var coef = List[Float32](length=d, fill=0.0)
    var info = List[Float32](length=1, fill=0.0)
    # A C-order design of more than one column is read as it is (the device
    # transposes it); one column is the same words in F order.
    var request = XlCdFitRequest(call[].route.entry.call, call[].x.data, call[].y.data, Int(coef.unsafe_ptr()),
        Int(info.unsafe_ptr()), sw, n, d, p[1] != 0.0, Int(p[2]), p[6] != 0.0, d > 1, p[3], p[4], p[5])
    var n_iter = xl_cd_fit_run(request)
    slot[].scalars = [Float64(info[0]), Float64(n_iter)]
    _ = info^
    slot[].coef = coef^
    slot[].features = d
    slot[].fitted = True


def _f_order(src: Int, dst: Int, rows: Int, cols: Int):
    """The F-order words of a C-order rows x cols float32 design, rows split
    across the host's workers (the own predict admission's one transpose)."""
    var tasks = host_predict_task_count_work(rows, rows * cols)
    var chunk = host_predict_chunk(rows, tasks)
    var source = _F32P(unsafe_from_address=src)
    var target = _F32P(unsafe_from_address=dst)

    def task(worker: Int) {imm chunk, imm source, imm target, imm rows, imm cols}:
        var begin = worker * chunk
        var end = min(begin + chunk, rows)
        for r in range(begin, end):
            for c in range(cols):
                target[c * rows + r] = source[r * cols + c]

    host_parallelize(task, tasks)


def _predict_into(slot: _SlotP, call: _CallP, dst: Int) raises:
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    var rows = call[].x.rows
    var cols = call[].x.cols
    if cols != slot[].features:
        raise Error("mojolearn ElasticNet feature count differs from fit")
    if rows == 0:
        return
    var design = call[].x.data
    var scratch = List[Float32]()
    if rows > 1 and cols > 1:
        scratch = List[Float32](length=rows * cols, fill=0.0)
        _f_order(call[].x.data, Int(scratch.unsafe_ptr()), rows, cols)
        design = Int(scratch.unsafe_ptr())
    xl_cd_predict_run(XlCdPredictRequest(call[].route.entry.call, design, Int(slot[].coef.unsafe_ptr()), dst, rows,
        cols, slot[].scalars[0]))
    _ = scratch^


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
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot) + " member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= 2:
            raise Error("member ABI: a " + _name(slot) + " member state has 2 fitted blocks")
        block = MEMBER_BLOCK_COEF if block == 0 else MEMBER_BLOCK_SCALARS
    if block == MEMBER_BLOCK_COEF:
        call[].dst = MemberMatrix(Int(slot[].coef.unsafe_ptr()), 1, slot[].features)
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, 2, MEMBER_F64)
    else:
        raise Error("member ABI: a " + _name(slot) + " member state holds no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 2


def _import(slot: _SlotP, call: _CallP) raises:
    """COEF (1 x features) and SCALARS [intercept, n_iter], then DONE."""
    var estimator = _name(slot)
    if call[].block == MEMBER_BLOCK_COEF:
        _dense(call[].x, "a fitted block", estimator)
        var count = call[].x.rows * call[].x.cols
        var words = List[Float32](length=count, fill=0.0)
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        slot[].fitted = False
        slot[].features = count
        slot[].coef = words^
    elif call[].block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != 2:
            raise Error("member ABI: the " + estimator + " scalars block is 2 host float64 words [intercept, n_iter]")
        slot[].fitted = False
        slot[].scalars = List[Float64](length=2, fill=0.0)
        unsafe_memcpy(dest=slot[].scalars.unsafe_ptr(), src=_F64P(unsafe_from_address=call[].x.data), count=2)
    elif call[].block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a " + estimator + " member imports no fitted block " + String(call[].block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].coef) == 0 or len(slot[].scalars) != 2:
        raise Error("member ABI: a fitted " + estimator + " state needs its coefficient and scalars blocks")
    slot[].fitted = True


def _cd_member(operation: Int, address: Int) raises -> Int:
    var call = _CallP(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        if call[].state != 0:
            _ = _lookup(call[].state, False)
        return _capabilities()
    if operation == MEMBER_OP_NEW:
        if not call[].params or call[].param_count < 1:
            raise Error("member ABI: a coordinate-descent member takes its kind as parameter word 0")
        var words = _F64P(unsafe_from_address=call[].params)
        var kind = Int(words[0])
        if Float64(kind) != words[0] or kind < 0 or kind > CD_MEMBER_LASSO:
            raise Error("member ABI: unknown coordinate-descent member kind")
        if call[].param_count != CD_MEMBER_PARAMS:
            raise Error("member ABI: a coordinate-descent member takes " + String(CD_MEMBER_PARAMS) + " parameters")
        var params = List[Float64](length=CD_MEMBER_PARAMS, fill=0.0)
        unsafe_memcpy(dest=params.unsafe_ptr(), src=words, count=CD_MEMBER_PARAMS)
        call[].state = _mint(params^)
        return 0
    if operation == MEMBER_OP_FREE:
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        # One target column for every kind, with or without a state.
        if call[].state != 0:
            _ = _lookup(call[].state, False)
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


def cd_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of the coordinate-descent pair (schema `MEMBER_ABI`)."""
    try:
        var word = _cd_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def cd_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's coordinate-descent member entry. One line in the
    image's module initialization, after its operations table is published."""
    member_publish[MEMBER_FAMILY_CD](mode, vendor, publisher, cd_member_v2)

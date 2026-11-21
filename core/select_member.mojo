# SPDX-License-Identifier: Apache-2.0
"""The Select family as MEMBERS (core/member_abi.mojo): VarianceThreshold
(kind 0) so far; SelectKBest and RFE are not published yet.

One entry, `select_member_v2`, published by each preprocessing image beside
the Prep member (`select_member_publish`, bindings/_mojolearn_preprocessing.mojo
and bindings/_mojolearn_preprocessing_host.mojo). VarianceThreshold runs its
door's own x_prep programs (bindings/prep_stateless_native_door.mojo
`_variance_fit`: x_prep/prep_stateless_native_run.mojo `_variance_program`,
column statistics, var_ptp, the binary64 key and compare; `_select_gather_run`:
one gather_cols stage) through the typed RUN of the x_prep transport
(core/prep_member.mojo `_execute`, bindings/x_prep_native_owner_run.mojo
`xprep_execute`) over the transport entry of the `DeviceRoute` it is handed
(`second`), with the same arena layout and input records, so no bit moves.
No kernel is touched. No score, no inverse_transform.

Route slots read: `second` the x_prep transport entry; `flags` the Prep
member's PREP_ROUTE_OUT / SCRATCH / RANGES / DEV_PUT bits (the same admit,
providers/python/member_prep_boundary_door.mojo `_route_admit`).

MEMBER_OP_NEW parameters, VarianceThreshold (Float64 words, 4):
  [0] kind: 0 VarianceThreshold
  [1] threshold (binary64, compared in binary64 on the device as the door)
  [2] the image's identical family switches word (`_idn_fam`, 0 on fast)
  [3] R3 work region 0 / 1 (`xprep_r3("work")`)

Views: X dense row-major host float32 (NaN allowed, as the estimator allows
it); no y (ignored), no sample weights (refused), no row or column index,
no device-home view. The transform width is the kept column count: known
after the fit only (OUTPUTS before a state refuses in words).
`call.fit_output = MEMBER_OP_TRANSFORM` fits, then gathers the kept columns.

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), VarianceThreshold:
  0 `MEMBER_BLOCK_SCALARS` float64 [n_features_in_]
  1 `SELECT_BLOCK_VARIANCES` (MEMBER_BLOCK_FAMILY) float32 1 x features (variances_)
  2 `SELECT_BLOCK_SUPPORT` float32 1 x features, 1 kept / 0 dropped (the support mask)

The fitted state is OWNED here, in a process-wide store of fixed chunks (the
linear member's store): a slot never moves once minted, the lock covers only
mint, free and the handle lookup. Serial by nature here: the store's
bookkeeping; the programs are the family's.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy, bitcast
from bindings.x_prep_native_owner_run import XprepRunInput
from x_prep.prep_stateless_native_run import PrepLayout, OP_GATHER_COLS, _variance_program, _place_scratch
from core.preprocessing_program import prep_program_stage as _stage
from core.prep_member import _execute as prep_execute, _x_input as prep_x_input, PREP_ROUTE_OUT, PREP_ROUTE_SCRATCH
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_OUTPUTS,
    MEMBER_OP_TRANSFORM, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_TRANSFORM, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_SELECT
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime SELECT_KIND_VARIANCE = 0
comptime SELECT_VARIANCE_PARAM_COUNT = 4

comptime SELECT_BLOCK_VARIANCES = MEMBER_BLOCK_FAMILY
comptime SELECT_BLOCK_SUPPORT = MEMBER_BLOCK_FAMILY + 1

comptime _INT32_BOUND = 2147483647
comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216


struct SelectMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var words: List[Float64]
    var fitted: Bool
    var features: Int
    var variances: List[Float32]
    var support: List[Float32]
    # The kept column indices as float32 index words (the door's `_words`),
    # built once per fitted state.
    var columns: List[Float32]
    var scalars: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.words = List[Float64]()
        self.fitted = False
        self.features = 0
        self.variances = List[Float32]()
        self.support = List[Float32]()
        self.columns = List[Float32]()
        self.scalars = List[Float64]()

    def kind(self) -> Int:
        return Int(self.words[0])

    def clear(mut self):
        self.fitted = False
        self.features = 0
        self.variances = List[Float32]()
        self.support = List[Float32]()
        self.columns = List[Float32]()
        self.scalars = List[Float64]()


struct SelectMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[SelectMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[SelectMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[SelectMemberStore, "MojolearnSelectMemberStoreV2", 2]
comptime _Slot = MutPointer[SelectMemberSlot, MutUntrackedOrigin]
comptime _Call = MutPointer[MemberCall, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[SelectMemberSlot]()


def _mint(var words: List[Float64]) raises -> Int:
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live Select member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[SelectMemberSlot](length=_CHUNK, fill=SelectMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _Slot(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].words = words^
        slot[].clear()
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
        var slot = _Slot(unsafe_from_address=address)
        if not slot[].live or slot[].generation != handle // _INDEX_SPAN:
            raise Error("NotFittedError: this member state was released or never created")
        if release:
            slot[].live = False
            slot[].clear()
            slot[].words = List[Float64]()
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _name(slot: _Slot) -> String:
    return String("VarianceThreshold")


def _dense(view: MemberMatrix, what: String, estimator: String) raises:
    if not view.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + estimator + " member reads " + what + " as a dense host float32 view")


def _whole(call: _Call, estimator: String) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads whole host views (no row or column index)")


def _fixed5(value: Float64) -> String:
    """`format(value, ".5f")` for the door's refusal words (finite values;
    anything else as Mojo writes it)."""
    if value != value or value > 1.0e15 or value < -1.0e15:
        return String(value)
    var negative = value < 0.0
    var scaled = Int((-value if negative else value) * 100000.0 + 0.5)
    var whole = scaled // 100000
    var fraction = String(scaled % 100000)
    while fraction.byte_length() < 5:  # small-loop(params: at most five pad digits)
        fraction = "0" + fraction
    return ("-" if negative and scaled > 0 else "") + String(whole) + "." + fraction


def _keep_columns(slot: _Slot):
    # The kept column indices as float32 index words, once per fitted state.
    slot[].columns = List[Float32](capacity=slot[].features)
    for j in range(slot[].features):  # driver-loop: over the fitted support mask (feature metadata)
        if slot[].support[j] != 0:
            slot[].columns.append(Float32(j))


def _fit_variance(slot: _Slot, call: _Call) raises:
    """The door's `_variance_fit` over the caller's X: the same layout,
    program and threshold words."""
    var estimator = _name(slot)
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member takes no sample_weight")
    if call[].fit_output != 0 and call[].fit_output != MEMBER_OP_TRANSFORM:
        member_unsupported(estimator, call[].fit_output)
    var n = call[].x.rows
    var d = call[].x.cols
    var threshold = slot[].words[1]
    var layout = PrepLayout(n * d, slot[].words[3] != 0.0)
    var program = List[Int32]()
    var placed = _variance_program(program, layout, Int(slot[].words[2]), n, d, threshold == 0.0)
    var variance = placed[0]
    var mask = placed[1]
    var limit = placed[2]
    if layout.host + layout.scratch > _INT32_BOUND:
        raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
    _place_scratch(program, layout.host)
    var flags = call[].route.flags
    var device_scratch = layout.scratch > 0 and (flags & (PREP_ROUTE_OUT | PREP_ROUTE_SCRATCH)) != 0
    var arena = List[Float32](length=max(layout.host + (0 if device_scratch else layout.scratch), 1), fill=Float32(0))
    # The threshold as its binary64 word, two 4-byte words (`_f64_input`).
    var limit_words = List[Float32](length=2, fill=Float32(0))
    MutPointer[UInt64, MutUntrackedOrigin](unsafe_from_address=Int(limit_words.unsafe_ptr()))[] = bitcast[DType.uint64](threshold)
    var inputs = List[XprepRunInput](capacity=2)
    inputs.append(prep_x_input(call, n * d))
    inputs.append(XprepRunInput(limit, 2, Int(limit_words.unsafe_ptr()), -1, False, 0, False, 0, 0, 0))
    prep_execute(call, arena, layout.host, layout.scratch, 0, 0, program, inputs)
    _ = limit_words^
    var ap = _F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    slot[].clear()
    slot[].variances = List[Float32](length=d, fill=Float32(0))
    unsafe_memcpy(dest=slot[].variances.unsafe_ptr(), src=ap + variance, count=d)
    slot[].support = List[Float32](length=d, fill=Float32(0))
    var any_kept = False
    for j in range(d):  # driver-loop: over the fitted support mask (feature metadata)
        if ap[mask + j] != 0:
            slot[].support[j] = Float32(1)
            any_kept = True
    _ = arena^
    if not any_kept:
        slot[].clear()
        # scikit-learn's wording, with its one-sample note
        var message = "No feature in X meets the variance threshold " + _fixed5(threshold)
        if n == 1:
            message += " (X contains only one sample)"
        raise Error(message)
    slot[].features = d
    slot[].scalars = [Float64(d)]
    _keep_columns(slot)
    slot[].fitted = True
    if call[].fit_output == MEMBER_OP_TRANSFORM:
        _transform_select(slot, call)


def _transform_select(slot: _Slot, call: _Call) raises:
    """The door's `_select_gather_run`: one gather_cols stage of the kept
    columns."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("RuntimeError: mojolearn: this " + estimator + " instance is not fitted yet")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    _dense(call[].dst, "the output buffer", estimator)
    var n = call[].x.rows
    var d = slot[].features
    if call[].x.cols != d:
        raise Error("mojolearn: X has " + String(call[].x.cols) + " features, " + estimator + " was fitted with "
                    + String(d))
    var k = len(slot[].columns)
    if call[].dst.rows != n or call[].dst.cols != k:
        raise Error("ValueError: the output buffer holds one value per kept column of each row")
    var ko = n * d
    var out = ko + k
    var host = out + n * k
    if host > _INT32_BOUND:
        raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
    var program = List[Int32]()
    _stage(program, OP_GATHER_COLS, n * k, 0, n, d, ko, k, out)
    var inputs = List[XprepRunInput](capacity=2)
    inputs.append(prep_x_input(call, n * d))
    inputs.append(XprepRunInput(ko, k, Int(slot[].columns.unsafe_ptr()), -1, False, 0, False, 0, 0, 0))
    var arena = List[Float32](length=max(host, 1), fill=Float32(0))
    prep_execute(call, arena, host, 0, 0, 0, program, inputs)
    unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data),
        src=_F32P(unsafe_from_address=Int(arena.unsafe_ptr())) + out, count=n * k)
    _ = arena^


def _export(slot: _Slot, call: _Call) raises:
    """A view of one fitted block. The slot owns the words; they stay valid
    until this state's next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot) + " member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > 2:
            raise Error("member ABI: a " + _name(slot) + " member has no fitted block index " + String(block))
        block = MEMBER_BLOCK_SCALARS if block == 0 else (SELECT_BLOCK_VARIANCES if block == 1 else SELECT_BLOCK_SUPPORT)
    if block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block == SELECT_BLOCK_VARIANCES:
        call[].dst = MemberMatrix(Int(slot[].variances.unsafe_ptr()), 1, len(slot[].variances))
    elif block == SELECT_BLOCK_SUPPORT:
        call[].dst = MemberMatrix(Int(slot[].support.unsafe_ptr()), 1, len(slot[].support))
    else:
        raise Error("member ABI: a " + _name(slot) + " member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 3


def _import(slot: _Slot, call: _Call) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var estimator = _name(slot)
    var block = call[].block
    if slot[].fitted:
        slot[].clear()
    if block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != 1:
            raise Error("member ABI: the " + estimator + " scalars block is one host float64 word")
        slot[].scalars = [MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].x.data)[]]
    elif block == SELECT_BLOCK_VARIANCES or block == SELECT_BLOCK_SUPPORT:
        var size = call[].x.rows * call[].x.cols
        var words = List[Float32](length=size, fill=Float32(0))
        if size > 0:
            _dense(call[].x, "a fitted block", estimator)
            unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=size)
        if block == SELECT_BLOCK_VARIANCES:
            slot[].variances = words^
        else:
            slot[].support = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a " + estimator + " member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) != 1:
        raise Error("member ABI: a fitted " + estimator + " state needs its scalars block")
    var d = Int(slot[].scalars[0])
    if d < 1 or len(slot[].support) != d or (len(slot[].variances) != 0 and len(slot[].variances) != d):
        raise Error("member ABI: a fitted " + estimator + " state needs its support mask, one word per feature")
    slot[].features = d
    _keep_columns(slot)
    if len(slot[].columns) == 0:
        raise Error("mojolearn: no features were selected")
    slot[].fitted = True


def _outputs(call: _Call) raises:
    """`call.out_cols` of transform: the kept column count, known after the fit."""
    call[].classes = 0
    if call[].flags != MEMBER_OP_TRANSFORM:
        member_unsupported(String("VarianceThreshold"), call[].flags)
    if call[].state == 0:
        raise Error("NotFittedError: a VarianceThreshold member's output width (its kept columns) is chosen by its fit")
    var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot) + " member's output width is known after its fit")
    call[].out_cols = len(slot[].columns)


def _select_member(operation: Int, address: Int) raises -> Int:
    var call = _Call(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        return MEMBER_CAN_FIT | MEMBER_CAN_TRANSFORM | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
    if operation == MEMBER_OP_NEW:
        if call[].param_count < 1 or not call[].params:
            raise Error("member ABI: a Select member takes its parameter words")
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)
        if source[0] != Float64(SELECT_KIND_VARIANCE):
            raise Error("member ABI: unknown or unpublished Select member kind")
        if call[].param_count != SELECT_VARIANCE_PARAM_COUNT:
            raise Error("member ABI: a VarianceThreshold member takes four parameters")
        var words = List[Float64](length=call[].param_count, fill=0.0)
        for index in range(call[].param_count):  # driver-loop: the parameter words
            words[index] = source[index]
        call[].state = _mint(words^)
        return 0
    if operation == MEMBER_OP_FREE:
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        _outputs(call)
        return 0
    var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_FIT:
        _fit_variance(slot, call)
    elif operation == MEMBER_OP_TRANSFORM:
        _transform_select(slot, call)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_name(slot), operation)
    return 0


def select_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of the Select family (schema `MEMBER_ABI`)."""
    try:
        var word = _select_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def select_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's Select member entry. One line in the image's
    module initialization, after its kernel tables are published."""
    member_publish[MEMBER_FAMILY_SELECT](mode, vendor, publisher, select_member_v2)

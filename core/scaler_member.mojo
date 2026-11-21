# SPDX-License-Identifier: Apache-2.0
"""StandardScaler and MinMaxScaler as MEMBERS (core/member_abi.mojo).

One entry, `scaler_member_v2`, published by each preprocessing image
(`scaler_member_publish`, beside its scaler kernel table). It runs the
family's existing typed RUN, core/scaler_core.mojo (`scaler_standard_fit_core`,
`scaler_minmax_fit_core`, `scaler_transform_rows`, `scaler_transform_core`),
through `ScalerTableCalls` built from the `DeviceRoute` it is handed: `entry`
the scaler kernel entry (prep table PREP_TABLE_SCALER), `second` the x_prep
transport entry (the programs), `flags` bit SCALER_ROUTE_DIRECT: the direct
kernels are used (the image's answer and MOJOLEARN_SCALER_DIRECT, read once
at the caller's UNPACK). The same calls the doors make
(bindings/scaler_native_door.mojo, bindings/minmax_native_door.mojo). No
kernel is touched. No score (neither class has one).

MEMBER_OP_NEW parameters (Float64 words, 6):
  [0] kind: 0 StandardScaler, 1 MinMaxScaler
  [1] StandardScaler with_mean / MinMaxScaler clip (0 / 1)
  [2] StandardScaler with_std (0 / 1)
  [3] MinMaxScaler feature_range lower, [4] upper (Float32 values, lower < upper)
  [5] numeric mode: 1 identical, 0 fast (what PACK writes as numeric_mode_)

Views: X dense row-major host float32 (NaN allowed, as the estimators allow
it: the core fills and restores it); StandardScaler fit reads
`call.weights` (dense host float32, one per row; MEMBER_CAN_SAMPLE_WEIGHT for
kind 0 only: a MinMaxScaler member refuses a given weight). No y, no row or
column index, no device-home view. `call.fit_output = MEMBER_OP_TRANSFORM`
writes the training rows' transform to `dst` (rows x features) after the fit,
in the same call (the fit, then the transform kernel on the same host words).

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), in index order:
  StandardScaler: 0 `MEMBER_BLOCK_SCALARS` float64 [n_features_in_, seen kind
  (core/scaler_core.mojo SCALER_SEEN_*), seen scalar]; 1 `SCALER_BLOCK_STATS`
  (MEMBER_BLOCK_FAMILY) float32 3 x features (mean_, var_, scale_); 2
  `SCALER_BLOCK_SEEN` float32 1 x features (the per-feature seen counts or
  weights; 1 x 0 for a scalar seen kind).
  MinMaxScaler: 0 `MEMBER_BLOCK_SCALARS` float64 [n_features_in_,
  n_samples_seen_]; 1 `SCALER_BLOCK_STATS` float32 5 x features (data_min_,
  data_max_, data_range_, scale_, min_).
An import needs every block of its kind.

The derived transform rows (7 * features + 5 words, `scaler_transform_rows`)
are built once per fitted state, at its first transform, and kept in the
slot. The fitted state is OWNED here, in a process-wide store of fixed chunks
(the linear member's store): a slot never moves once minted, the lock covers
only mint, free and the handle lookup. Serial by nature here: the store's
bookkeeping; the kernels are the family's.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from core.scaler_core import (
    ScalerTableCalls, ScalerStandardFit, ScalerTransformSpec, scaler_standard_fit_core, scaler_minmax_fit_core,
    scaler_transform_rows, scaler_transform_core,
)
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_OUTPUTS,
    MEMBER_OP_TRANSFORM, MEMBER_OP_INVERSE_TRANSFORM, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_TRANSFORM, MEMBER_CAN_INVERSE_TRANSFORM, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_CAN_SAMPLE_WEIGHT,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_SCALER
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime SCALER_KIND_STANDARD = 0
comptime SCALER_KIND_MINMAX = 1
comptime SCALER_PARAM_COUNT = 6
comptime SCALER_BLOCK_STATS = MEMBER_BLOCK_FAMILY
comptime SCALER_BLOCK_SEEN = MEMBER_BLOCK_FAMILY + 1
comptime SCALER_ROUTE_DIRECT = 1

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216


struct ScalerMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var words: List[Float64]
    var fitted: Bool
    var features: Int
    var stats: List[Float32]
    var seen: List[Float32]
    var scalars: List[Float64]
    var rows: List[Float32]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.words = List[Float64]()
        self.fitted = False
        self.features = 0
        self.stats = List[Float32]()
        self.seen = List[Float32]()
        self.scalars = List[Float64]()
        self.rows = List[Float32]()

    def kind(self) -> Int:
        return Int(self.words[0])

    def clear(mut self):
        self.fitted = False
        self.features = 0
        self.stats = List[Float32]()
        self.seen = List[Float32]()
        self.scalars = List[Float64]()
        self.rows = List[Float32]()


struct ScalerMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[ScalerMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[ScalerMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[ScalerMemberStore, "MojolearnScalerMemberStoreV2", 2]
comptime _Slot = MutPointer[ScalerMemberSlot, MutUntrackedOrigin]
comptime _Call = MutPointer[MemberCall, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[ScalerMemberSlot]()


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
                raise Error("RuntimeError: mojolearn: too many live scaler member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[ScalerMemberSlot](length=_CHUNK, fill=ScalerMemberSlot()))
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
    return String("StandardScaler") if slot[].kind() == SCALER_KIND_STANDARD else String("MinMaxScaler")


def _calls(call: _Call) -> ScalerTableCalls:
    var route = call[].route
    return ScalerTableCalls(route.entry, route.second, (route.flags & SCALER_ROUTE_DIRECT) != 0)


def _dense(view: MemberMatrix, what: String, estimator: String) raises:
    if not view.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + estimator + " member reads " + what + " as a dense host float32 view")


def _whole(call: _Call, estimator: String) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads whole host views (no row or column index)")


def _fit(slot: _Slot, call: _Call) raises:
    var estimator = _name(slot)
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].fit_output != 0 and call[].fit_output != MEMBER_OP_TRANSFORM:
        member_unsupported(estimator, call[].fit_output)
    var n = call[].x.rows
    var d = call[].x.cols
    var weight = 0
    if call[].weights.rows > 0:
        if slot[].kind() != SCALER_KIND_STANDARD:
            raise Error("TypeError: MinMaxScaler.fit takes no sample_weight")
        _dense(call[].weights, "sample_weight", estimator)
        if call[].weights.rows != n or call[].weights.cols != 1:
            raise Error("ValueError: sample_weight and X have different numbers of rows")
        weight = call[].weights.data
    var calls = _calls(call)
    var x = _F32P(unsafe_from_address=call[].x.data)
    slot[].clear()
    if slot[].kind() == SCALER_KIND_STANDARD:
        var fit = scaler_standard_fit_core(calls, x, n, d, slot[].words[1] != 0.0, slot[].words[2] != 0.0, weight)
        # Copies of the fit's two short rows (3d and d words): no field is
        # moved out of the result (AGENTS.md part 6).
        slot[].stats = fit.stats.copy()
        slot[].seen = fit.seen.copy()
        slot[].scalars = [Float64(d), Float64(fit.seen_kind), fit.seen_scalar]
        _ = fit^
    else:
        slot[].stats = scaler_minmax_fit_core(calls, x, n, d, Float32(slot[].words[3]), Float32(slot[].words[4]))
        slot[].scalars = [Float64(d), Float64(n)]
    slot[].features = d
    slot[].fitted = True
    if call[].fit_output == MEMBER_OP_TRANSFORM:
        _transform(slot, call, False)


def _transform(slot: _Slot, call: _Call, inverse: Bool) raises:
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    _dense(call[].dst, "the output buffer", estimator)
    var n = call[].x.rows
    var d = slot[].features
    if call[].x.cols != d:
        raise Error(estimator + " input feature count differs from fit")
    if call[].dst.rows != n or call[].dst.cols != d:
        raise Error("ValueError: the output buffer holds one value per element of X")
    var calls = _calls(call)
    var standard = slot[].kind() == SCALER_KIND_STANDARD
    var with_mean = slot[].words[1] != 0.0 if standard else True
    var with_std = slot[].words[2] != 0.0 if standard else True
    if len(slot[].rows) == 0:
        # The derived rows of the fitted pair, once per fitted state.
        var stats = Int(slot[].stats.unsafe_ptr())
        var center = 0
        var spread = 0
        if standard:
            center = stats if (with_mean or with_std) else 0
            spread = stats + 8 * d if with_std else 0
        else:
            center = stats + 16 * d
            spread = stats + 12 * d
        slot[].rows = scaler_transform_rows(calls, center, spread, d, with_mean, with_std)
    var spec = ScalerTransformSpec(False, inverse, with_mean, with_std, False, 0, 0)
    if not standard:
        spec = ScalerTransformSpec(True, inverse, True, True, slot[].words[1] != 0.0, Float32(slot[].words[3]),
            Float32(slot[].words[4]))
    scaler_transform_core(calls, spec, _F32P(unsafe_from_address=Int(slot[].rows.unsafe_ptr())),
        _F32P(unsafe_from_address=call[].x.data), n, d, _F32P(unsafe_from_address=call[].dst.data))


def _export(slot: _Slot, call: _Call) raises:
    """A view of one fitted block. The slot owns the words; they stay valid
    until this state's next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot) + " member is not fitted yet")
    var standard = slot[].kind() == SCALER_KIND_STANDARD
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > (2 if standard else 1):
            raise Error("member ABI: a " + _name(slot) + " member has no fitted block index " + String(block))
        block = MEMBER_BLOCK_SCALARS if block == 0 else (SCALER_BLOCK_STATS if block == 1 else SCALER_BLOCK_SEEN)
    var d = slot[].features
    if block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block == SCALER_BLOCK_STATS:
        call[].dst = MemberMatrix(Int(slot[].stats.unsafe_ptr()), 3 if standard else 5, d)
    elif block == SCALER_BLOCK_SEEN and standard:
        call[].dst = MemberMatrix(Int(slot[].seen.unsafe_ptr()), 1, len(slot[].seen))
    else:
        raise Error("member ABI: a " + _name(slot) + " member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 3 if standard else 2


def _import(slot: _Slot, call: _Call) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var estimator = _name(slot)
    var standard = slot[].kind() == SCALER_KIND_STANDARD
    var block = call[].block
    slot[].fitted = False
    slot[].rows = List[Float32]()
    if block == MEMBER_BLOCK_SCALARS:
        var size = 3 if standard else 2
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != size:
            raise Error("member ABI: the " + estimator + " scalars block is " + String(size) + " host float64 words")
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].x.data)
        slot[].scalars = List[Float64](length=size, fill=0.0)
        for index in range(size):  # driver-loop: two or three fitted scalars
            slot[].scalars[index] = source[index]
    elif block == SCALER_BLOCK_STATS or (block == SCALER_BLOCK_SEEN and standard):
        var count = call[].x.rows * call[].x.cols
        var words = List[Float32](length=count, fill=0.0)
        if count > 0:
            _dense(call[].x, "a fitted block", estimator)
            unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        if block == SCALER_BLOCK_STATS:
            slot[].stats = words^
        else:
            slot[].seen = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a " + estimator + " member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) != (3 if standard else 2):
        raise Error("member ABI: a fitted " + estimator + " state needs its scalars block")
    var d = Int(slot[].scalars[0])
    if d < 1 or len(slot[].stats) != (3 if standard else 5) * d or (standard and len(slot[].seen) != 0
            and len(slot[].seen) != d):
        raise Error("member ABI: a fitted " + estimator + " state needs every fitted block, of the fitted sizes")
    slot[].features = d
    slot[].fitted = True


def _scaler_member(operation: Int, address: Int) raises -> Int:
    var call = _Call(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        var mask = MEMBER_CAN_FIT | MEMBER_CAN_TRANSFORM | MEMBER_CAN_INVERSE_TRANSFORM | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
        if call[].state == 0:
            return mask | MEMBER_CAN_SAMPLE_WEIGHT
        var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
        return mask | (MEMBER_CAN_SAMPLE_WEIGHT if slot[].kind() == SCALER_KIND_STANDARD else 0)
    if operation == MEMBER_OP_NEW:
        if call[].param_count != SCALER_PARAM_COUNT or not call[].params:
            raise Error("member ABI: a scaler member takes six parameters")
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)
        if source[0] != 0.0 and source[0] != 1.0:
            raise Error("member ABI: unknown scaler member kind")
        if source[0] == 1.0 and not Float32(source[3]) < Float32(source[4]):
            raise Error("feature_range must have lower < upper after Float32 conversion")
        var words = List[Float64](length=SCALER_PARAM_COUNT, fill=0.0)
        for index in range(SCALER_PARAM_COUNT):  # driver-loop: six parameter words
            words[index] = source[index]
        call[].state = _mint(words^)
        return 0
    if operation == MEMBER_OP_FREE:
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        # transform and inverse_transform keep the width of X.
        call[].classes = 0
        if call[].flags != MEMBER_OP_TRANSFORM and call[].flags != MEMBER_OP_INVERSE_TRANSFORM:
            member_unsupported("scaler", call[].flags)
        if call[].state == 0:
            raise Error("member ABI: a scaler member's output width is its fitted feature count")
        var held = _Slot(unsafe_from_address=_lookup(call[].state, False))
        if not held[].fitted:
            raise Error("NotFittedError: this " + _name(held) + " member's output width is known after its fit")
        call[].out_cols = held[].features
        return 0
    var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_TRANSFORM:
        _transform(slot, call, False)
    elif operation == MEMBER_OP_INVERSE_TRANSFORM:
        _transform(slot, call, True)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_name(slot), operation)
    return 0


def scaler_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of StandardScaler and MinMaxScaler (schema `MEMBER_ABI`)."""
    try:
        var word = _scaler_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def scaler_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's scaler member entry. One line in the image's
    module initialization, after its scaler kernel table is published."""
    member_publish[MEMBER_FAMILY_SCALER](mode, vendor, publisher, scaler_member_v2)

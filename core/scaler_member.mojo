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
kind 0 only: a MinMaxScaler member refuses a given weight). No y, no column
index, no device-home view. ROW INDEX and SCOPE (lane vdf-scope-neural2;
answered only on a route holding the GPU image's preprocessing member store,
`SCALER_ROUTE_STORE`, preprocessing/prep_member_store.mojo): `call.row_index`
names rows of the WHOLE X (and of the whole sample weights); with the
caller's scope and pin the whole X stays resident in the scope's
"preprocessing" cache, the fold is taken on the device and the direct fit
and transform read it in place (core/scaler_native_api.mojo
SCALER_OP_FIT_SLOT / SCALER_OP_TRANSFORM_SLOT); a NaN X, sample weights or a
fitted state with an all-NaN column run the cores' x_prep programs over an
x_prep copy of the slot (lane vdf-scope-neural6, core/scaler_slot_runs.mojo:
one device-to-device copy, the runner's slot range); a route without a pin
reads the fold's rows on the host (gathered in host row tasks), as the
host-input route. `call.fit_output = MEMBER_OP_TRANSFORM`
writes the training rows' transform to `dst` (rows x features) in the same
call: on a direct route with no weights, the door's ONE combined kernel call
(SCALER_OP_STANDARD_FIT_TRANSFORM / SCALER_OP_MINMAX_FIT_TRANSFORM: one upload
of X); otherwise (NaN in X, weights, the host column) fit then transform, as
the doors do.

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
from std.memory import unsafe_memcpy, bitcast
from core.scaler_core import (
    ScalerTableCalls, ScalerStandardFit, ScalerTransformSpec, scaler_standard_fit_core, scaler_minmax_fit_core,
    scaler_minmax_keep_extrema, scaler_transform_rows, scaler_transform_core, SCALER_SEEN_INT,
    scaler_transform_slot_finish,
)
from core.scaler_slot_runs import scaler_slot_to_xprep, scaler_xprep_free
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
from core.entry_table import entry_ref_call
from core.scaler_core import scaler_transform_rows_valid
from core.scaler_fit_impl import scaler_check_stats
from core.scaler_native_api import SCALER_OP_FIT_SLOT, SCALER_OP_TRANSFORM_SLOT
from core.seq_member_rows import (
    SeqRows, SEQ_ROUTE_STORE, seq_rows_route, member_rows_open, seq_rows_close, seq_rows_count, seq_rows_host,
    seq_rows_target,
)
from core.member_abi import MEMBER_CAN_ROW_INDEX, MEMBER_CAN_SCOPE

comptime SCALER_KIND_STANDARD = 0
comptime SCALER_KIND_MINMAX = 1
comptime SCALER_PARAM_COUNT = 6
comptime SCALER_BLOCK_STATS = MEMBER_BLOCK_FAMILY
comptime SCALER_BLOCK_SEEN = MEMBER_BLOCK_FAMILY + 1
comptime SCALER_ROUTE_DIRECT = 1
#: `route.flags` bit (lane vdf-scope-neural2): `route.fifth` is the GPU
#: image's preprocessing member store (the bit core/seq_member_rows.mojo reads).
comptime SCALER_ROUTE_STORE = SEQ_ROUTE_STORE
#: The scope cache's family prefix of the preprocessing members.
comptime SCALER_STORE_PREFIX = "preprocessing"

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


def _fit(slot: _Slot, call: _Call) raises:
    var estimator = _name(slot)
    _dense(call[].x, "X", estimator)
    if call[].fit_output != 0 and call[].fit_output != MEMBER_OP_TRANSFORM:
        member_unsupported(estimator, call[].fit_output)
    var n = seq_rows_count(call)
    var d = call[].x.cols
    var weight = 0
    var w_fold = List[Int32]()
    if call[].weights.rows > 0:
        if slot[].kind() != SCALER_KIND_STANDARD:
            raise Error("TypeError: MinMaxScaler.fit takes no sample_weight")
        _dense(call[].weights, "sample_weight", estimator)
        if call[].weights.rows != call[].x.rows or call[].weights.cols != 1:
            raise Error("ValueError: sample_weight and X have different numbers of rows")
        # Under a row index the fold's weights (host row tasks), else as given.
        w_fold = seq_rows_target(call, call[].weights, 1)
        weight = Int(w_fold.unsafe_ptr()) if call[].row_index.rows > 0 else call[].weights.data
    var calls = _calls(call)
    slot[].clear()
    if call[].fit_output == MEMBER_OP_TRANSFORM:
        _dense(call[].dst, "the output buffer", estimator)
        if call[].dst.rows != n or call[].dst.cols != d:
            raise Error("ValueError: the output buffer holds one value per element of X")
    # THE RESIDENT ROUTE (core/seq_member_rows.mojo): X in the scope's
    # preprocessing store and the fold taken on the device, or the rows on the
    # host (X as given, or the fold gathered there).
    var held = member_rows_open(call, 0, estimator, String(SCALER_STORE_PREFIX))
    var transformed = False
    try:
        transformed = _fit_rows(slot, call, calls, held, n, d, weight)
        if call[].fit_output == MEMBER_OP_TRANSFORM and not transformed:
            # The training rows' transform reads the same rows (no second take).
            _transform_held(slot, call, held, False)
    finally:
        seq_rows_close(call, held)
    _ = held^  # the gathered words live past the run
    _ = len(w_fold)  # the fold's weights live past the fit


def _fit_slot(slot: _Slot, call: _Call, calls: ScalerTableCalls, store_id: Int, n: Int, d: Int) raises -> Bool:
    """The direct fit of the fold held in the preprocessing member store
    (SCALER_OP_FIT_SLOT, read in place). False (nothing written) when X
    holds a NaN or an infinity: the caller takes the host NaN route."""
    var standard = slot[].kind() == SCALER_KIND_STANDARD
    var stats = List[Float32](length=(3 if standard else 5) * d, fill=0)
    var a: Array[Int, 1] = [Int(stats.unsafe_ptr())]
    var p: Array[Int, 6] = [store_id, n, d, 0 if standard else 1, Int(slot[].words[1]) if standard else 1,
                            Int(slot[].words[2]) if standard else 1]
    var f: Array[Float64, 2] = [Float64(Float32(slot[].words[3])) if not standard else 0.0,
                                Float64(Float32(slot[].words[4])) if not standard else 1.0]
    if entry_ref_call(calls.scaler, SCALER_OP_FIT_SLOT, a, p, f) == 0:
        return False
    if standard:
        # The core's host checks of the three fitted rows (3 d words), as
        # `scaler_standard_fit_core` runs them after the direct kernel.
        var sp = _F32P(unsafe_from_address=Int(stats.unsafe_ptr()))
        scaler_check_stats(sp, sp + d, sp + 2 * d, d)
        slot[].stats = stats^
        slot[].seen = List[Float32]()
        slot[].scalars = [Float64(d), Float64(SCALER_SEEN_INT), Float64(n)]
    else:
        # The door's `_keep_extrema` on a finite X: the c2_mm_keep program.
        slot[].stats = scaler_minmax_keep_extrema(calls, _F32P(unsafe_from_address=Int(stats.unsafe_ptr())), 0, d)
        _ = stats^
        slot[].scalars = [Float64(d), Float64(n)]
    slot[].features = d
    slot[].fitted = True
    return True


def _fit_rows(slot: _Slot, call: _Call, calls: ScalerTableCalls, mut held: SeqRows, n: Int, d: Int,
              weight: Int) raises -> Bool:
    """The fit over the operation's rows `held` (a store slot or host
    words). True when the training rows' transform was written too (the
    door's ONE combined call); otherwise the caller runs the transform."""
    var x_slot = -1
    if held.x < 0:
        if weight == 0 and calls.direct() and _fit_slot(slot, call, calls, -held.x - 1, n, d):
            return False
        # Lane vdf-scope-neural6: the program route (sample weights, or the
        # direct slot fit found a NaN or an infinity) reads the fold from an
        # x_prep copy of the slot (core/scaler_slot_runs.mojo), never from a
        # host gather. Without the direct entries an unweighted fit keeps the
        # host rows (its finite X takes the host kernel, not the program).
        if weight != 0 or calls.direct():
            x_slot = scaler_slot_to_xprep(-held.x - 1, n * d)
    try:
        var x = _F32P(unsafe_from_address=0 if x_slot >= 0 else seq_rows_host(call, held))
        if x_slot < 0 and call[].fit_output == MEMBER_OP_TRANSFORM:
            if calls.direct() and weight == 0 and _fit_transform_once(slot, call, calls, Int(x), n, d):
                return True
        if slot[].kind() == SCALER_KIND_STANDARD:
            var fit = scaler_standard_fit_core(calls, x, n, d, slot[].words[1] != 0.0, slot[].words[2] != 0.0,
                                               weight, x_slot)
            # Copies of the fit's two short rows (3d and d words): no field is
            # moved out of the result (AGENTS.md part 6).
            slot[].stats = fit.stats.copy()
            slot[].seen = fit.seen.copy()
            slot[].scalars = [Float64(d), Float64(fit.seen_kind), fit.seen_scalar]
            _ = fit^
        else:
            slot[].stats = scaler_minmax_fit_core(calls, x, n, d, Float32(slot[].words[3]), Float32(slot[].words[4]),
                                                  x_slot)
            slot[].scalars = [Float64(d), Float64(n)]
    finally:
        if x_slot >= 0:
            scaler_xprep_free(x_slot)
    slot[].features = d
    slot[].fitted = True
    return False


def _fit_transform_once(slot: _Slot, call: _Call, calls: ScalerTableCalls, x: Int, n: Int, d: Int) raises -> Bool:
    """The door's fit_transform: ONE combined kernel call on one upload of
    X (`SCALER_OP_STANDARD_FIT_TRANSFORM` / `SCALER_OP_MINMAX_FIT_TRANSFORM`,
    the exports bindings/scaler_lifecycle_native_door.mojo and
    bindings/minmax_native_door.mojo call), the fitted rows kept as those
    doors keep them. False (nothing written) when X holds a NaN or an
    infinity: the caller runs the two-step fit then transform, as the doors do."""
    var dst = call[].dst.data
    if slot[].kind() == SCALER_KIND_STANDARD:
        var stats = List[Float32](length=3 * d, fill=0)
        var got = calls.standard_fit_transform(x, Int(stats.unsafe_ptr()), dst, n, d,
            slot[].words[1] != 0.0, slot[].words[2] != 0.0)
        if got == 0:
            return False
        slot[].stats = stats^
        slot[].seen = List[Float32]()
        slot[].scalars = [Float64(d), Float64(SCALER_SEEN_INT), Float64(n)]
        slot[].features = d
        slot[].fitted = True
        if got < 0:
            raise Error("StandardScaler transform overflowed in Float32")
        return True
    var raw = List[Float32](length=5 * d, fill=0)
    var got = calls.minmax_fit_transform(x, Int(raw.unsafe_ptr()), dst, n, d,
        Float32(slot[].words[3]), Float32(slot[].words[4]), slot[].words[1] != 0.0)
    if got == 0:
        return False
    # The door's `_keep_extrema` on a finite X: the c2_mm_keep program.
    slot[].stats = scaler_minmax_keep_extrema(calls, _F32P(unsafe_from_address=Int(raw.unsafe_ptr())), 0, d)
    _ = raw^
    slot[].scalars = [Float64(d), Float64(n)]
    slot[].features = d
    slot[].fitted = True
    if got < 0:
        raise Error("MinMaxScaler transform overflowed in Float32")
    return True


def _transform(slot: _Slot, call: _Call, inverse: Bool) raises:
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    var held = member_rows_open(call, 0, estimator, String(SCALER_STORE_PREFIX))
    try:
        _transform_held(slot, call, held, inverse)
    finally:
        seq_rows_close(call, held)
    _ = held^  # the gathered words live past the run


def _transform_held(slot: _Slot, call: _Call, mut held: SeqRows, inverse: Bool) raises:
    """transform / inverse_transform of the operation's rows `held`: a
    store slot read in place by SCALER_OP_TRANSFORM_SLOT, or host words."""
    var estimator = _name(slot)
    _dense(call[].x, "X", estimator)
    _dense(call[].dst, "the output buffer", estimator)
    var n = held.n
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
    var rows = _F32P(unsafe_from_address=Int(slot[].rows.unsafe_ptr()))
    # THE RESIDENT ROUTE: valid rows read the slot in place (the direct
    # transform); lane vdf-scope-neural6: a NaN X's fill and the per-column
    # NaN restore run their programs over an x_prep copy of the slot
    # (`scaler_transform_slot_finish`), and invalid rows take the core's
    # refusals, which read no X.
    if held.x < 0 and calls.direct() and not scaler_transform_rows_valid(rows, d):
        scaler_transform_core(calls, spec, rows, _F32P(unsafe_from_address=0), n, d,
            _F32P(unsafe_from_address=call[].dst.data))
        raise Error(estimator + ": the fitted statistics were refused")  # not reached: the core raised
    if held.x < 0 and calls.direct():
        var a: Array[Int, 3] = [Int(rows + 3 * d) if not standard else Int(rows + 2 * d),
                                Int(rows + 2 * d) if not standard else Int(rows + 3 * d), call[].dst.data]
        var p: Array[Int, 8] = [-held.x - 1, n, d, 0 if standard else 1, 1 if inverse else 0,
                                1 if spec.with_mean else 0, 1 if spec.with_std else 0, 1 if spec.clip else 0]
        var f: Array[Float64, 2] = [Float64(spec.lower) if not standard else 0.0,
                                    Float64(spec.upper) if not standard else 1.0]
        var got = entry_ref_call(calls.scaler, SCALER_OP_TRANSFORM_SLOT, a, p, f)
        if got == 0:
            raise Error(estimator + " transform overflowed in Float32")
        var colnan = bitcast[DType.int32](rows[7 * d]) != 0
        if got == 1 and not colnan:
            return
        # -1: X holds a NaN or an infinity (or a column was all NaN at the
        # fit): the core's NaN fill and restore over an x_prep copy of the slot.
        var x_slot = scaler_slot_to_xprep(-held.x - 1, n * d)
        try:
            scaler_transform_slot_finish(calls, spec, rows, x_slot, n, d, _F32P(unsafe_from_address=call[].dst.data),
                                         got == 1)
        finally:
            scaler_xprep_free(x_slot)
        return
    scaler_transform_core(calls, spec, rows, _F32P(unsafe_from_address=seq_rows_host(call, held)), n, d,
        _F32P(unsafe_from_address=call[].dst.data))


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
        # THE RESIDENT ROUTE: a route holding the image's preprocessing member
        # store reads a fold's rows of the whole X.
        if seq_rows_route(call[].route):
            mask |= MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE
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
        # transform and inverse_transform keep the width of X, float32.
        call[].classes = 0
        call[].out_dtype = MEMBER_F32
        if call[].flags != MEMBER_OP_TRANSFORM and call[].flags != MEMBER_OP_INVERSE_TRANSFORM:
            member_unsupported("scaler", call[].flags)
        if call[].state == 0:
            # Asked before any state (`member_outputs`): the width of the X it
            # will be fitted on (`x.cols`), as a fitted state answers.
            call[].out_cols = call[].x.cols
            return 0
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

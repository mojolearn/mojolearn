# SPDX-License-Identifier: Apache-2.0
"""The Prep family as MEMBERS (core/member_abi.mojo): RobustScaler (kind 0),
MaxAbsScaler (kind 1), Binarizer (kind 2) and Normalizer (kind 3) so far;
the family's other classes are not published yet.

One entry, `prep_member_v2`, published by each preprocessing image beside
the scaler member (`prep_member_publish`, bindings/_mojolearn_preprocessing.mojo
and bindings/_mojolearn_preprocessing_host.mojo). RobustScaler runs the
door's own x_prep programs (x_prep/robust_native_run.mojo
`_robust_fit_program`, the affine stage 3) through the typed RUN of the
x_prep transport (bindings/x_prep_native_owner_run.mojo `xprep_execute`,
`xprep_release`) over the transport entry of the `DeviceRoute` it is handed
(`second`): the same arena layout, input records, range runner, device
scratch and output region the door (bindings/robust_native_pack.mojo
`robust_fit` / `robust_transform`) builds, so no bit moves. No kernel is
touched. MaxAbsScaler runs its door's x_prep programs the same way
(x_prep/maxabs_native_run.mojo `_maxabs_fit_program`, the affine stage 3;
door bindings/maxabs_native_door.mojo `maxabs_fit` / `_maxabs_apply`). No
score, no inverse_transform (no published class has one). Binarizer and
Normalizer are stateless: their fit records X's width only (X is not
uploaded), their transform is the door's one-stage program
(bindings/prep_stateless_native_door.mojo `_transform` binarizer branch,
bindings/prep_normalizer_lifecycle_pack.mojo `_normalizer_apply`).

Route slots read: `second` the x_prep transport entry; `flags`
PREP_ROUTE_OUT (x_prep_run_out), PREP_ROUTE_SCRATCH (x_prep_run_scratch),
PREP_ROUTE_RANGES (x_prep_run_ranges and MOJOLEARN_ARENA_RANGES not 0),
PREP_ROUTE_DEV_PUT (x_prep_dev_put and x_prep_dev_free). The switches the
door reads per call (MOJOLEARN_ARENA_RANGES, MOJOLEARN_XPREP_BLOCKED, the R3
`work` choice) are read once at the caller's UNPACK.

MEMBER_OP_NEW parameters, RobustScaler (Float64 words, 9):
  [0] kind: 0 RobustScaler
  [1] with_centering 0 / 1   [2] with_scaling 0 / 1
  [3] quantile_range lower, [4] upper (percent, checked at admission)
  [5] unit_variance 0 / 1
  [6] radix quantile selection 0 / 1 (the door's `select`: fast, Metal, robust_qselect)
  [7] 2048-row statistics blocks 0 / 1 (MOJOLEARN_XPREP_BLOCKED)
  [8] R3 work region 0 / 1 (`xprep_r3("work")`)

MEMBER_OP_NEW parameters, MaxAbsScaler (Float64 words, 3):
  [0] kind: 1 MaxAbsScaler
  [1] 2048-row statistics blocks 0 / 1 (MOJOLEARN_XPREP_BLOCKED)
  [2] R3 work region 0 / 1 (`xprep_r3("work")`, read only with [1])
MEMBER_OP_NEW parameters, Binarizer (Float64 words, 2): [0] kind 2,
[1] threshold (rounded to float32 at transform, as the door).
MEMBER_OP_NEW parameters, Normalizer (Float64 words, 2): [0] kind 3,
[1] norm: 0 'l1', 1 'l2', 2 'max'.

The Apple fast direct fit entry (`x_prep_maxabs_fit_direct`, opt-in
PREP3_MAXABS) is not a member route: the admit answers False there and the
caller keeps the door.

Views: X dense row-major host float32 (NaN allowed, as the estimator allows
it); no y (ignored), no sample weights (refused), no row or column index,
no device-home view. `call.fit_output = MEMBER_OP_TRANSFORM` writes the
training rows' transform to `dst` (rows x features): in the fit's own
program when rows * features < 2^27 (the door's fused fit_transform), else
the fit then the transform program, as the door.

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), RobustScaler:
  0 `MEMBER_BLOCK_SCALARS` float64 [n_features_in_]
  1 `PREP_BLOCK_CENTER` (MEMBER_BLOCK_FAMILY) float32 1 x features (1 x 0 without centering)
  2 `PREP_BLOCK_SCALE` float32 1 x features (1 x 0 without scaling)
MaxAbsScaler:
  0 `MEMBER_BLOCK_SCALARS` float64 [n_features_in_, n_samples_seen_]
  1 `PREP_BLOCK_MAX_ABS` (= PREP_BLOCK_CENTER) float32 1 x features (max_abs_)
  2 `PREP_BLOCK_SCALE` float32 1 x features (scale_)
Binarizer, Normalizer: 0 `MEMBER_BLOCK_SCALARS` float64 [n_features_in_] only.

The fitted state is OWNED here, in a process-wide store of fixed chunks (the
linear member's store): a slot never moves once minted, the lock covers only
mint, free and the handle lookup. Serial by nature here: the store's
bookkeeping; the programs are the family's.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from bindings.x_prep_native_owner_run import XprepRunInput, XprepRoute, xprep_execute, xprep_release
from x_prep.robust_native_run import _robust_fit_program as robust_fit_program
from x_prep.maxabs_native_run import _maxabs_fit_program as maxabs_fit_program
from core.scaler_fit_impl import scaler_refuse_infinite
from x_prep.prep_stateless_native_run import OP_BINARIZE, OP_NORMALIZE, OUT_MIN_WORDS
from core.preprocessing_program import prep_program_stage as _stage
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_OUTPUTS,
    MEMBER_OP_TRANSFORM, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_TRANSFORM, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_PREP
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime PREP_KIND_ROBUST = 0
comptime PREP_KIND_MAXABS = 1
comptime PREP_ROBUST_PARAM_COUNT = 9
comptime PREP_MAXABS_PARAM_COUNT = 3
comptime PREP_KIND_BINARIZER = 2
comptime PREP_KIND_NORMALIZER = 3
comptime PREP_STATELESS_PARAM_COUNT = 2

comptime PREP_BLOCK_CENTER = MEMBER_BLOCK_FAMILY
comptime PREP_BLOCK_SCALE = MEMBER_BLOCK_FAMILY + 1
#: MaxAbsScaler's max_abs_ sits in the block RobustScaler's center_ uses.
comptime PREP_BLOCK_MAX_ABS = PREP_BLOCK_CENTER

comptime PREP_ROUTE_OUT = 1
comptime PREP_ROUTE_SCRATCH = 2
comptime PREP_ROUTE_RANGES = 4
comptime PREP_ROUTE_DEV_PUT = 8

#: The door's limits: the fused fit_transform and the output region
#: (bindings/robust_native_pack.mojo), the direct put (x_prep_native_owner_door).
comptime _FUSED_LIMIT = 134217728
comptime _PUT_WORDS = 1048576
comptime _INT32_BOUND = 2147483647

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216


struct PrepMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var words: List[Float64]
    var fitted: Bool
    var features: Int
    var center: List[Float32]
    var scale: List[Float32]
    var scalars: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.words = List[Float64]()
        self.fitted = False
        self.features = 0
        self.center = List[Float32]()
        self.scale = List[Float32]()
        self.scalars = List[Float64]()

    def kind(self) -> Int:
        return Int(self.words[0])

    def clear(mut self):
        self.fitted = False
        self.features = 0
        self.center = List[Float32]()
        self.scale = List[Float32]()
        self.scalars = List[Float64]()


struct PrepMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[PrepMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[PrepMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[PrepMemberStore, "MojolearnPrepMemberStoreV2", 2]
comptime _Slot = MutPointer[PrepMemberSlot, MutUntrackedOrigin]
comptime _Call = MutPointer[MemberCall, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[PrepMemberSlot]()


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
                raise Error("RuntimeError: mojolearn: too many live Prep member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[PrepMemberSlot](length=_CHUNK, fill=PrepMemberSlot()))
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


def _kind_name(kind: Int) -> String:
    if kind == PREP_KIND_MAXABS:
        return String("MaxAbsScaler")
    if kind == PREP_KIND_BINARIZER:
        return String("Binarizer")
    if kind == PREP_KIND_NORMALIZER:
        return String("Normalizer")
    return String("RobustScaler")


def _stateless(slot: _Slot) -> Bool:
    return slot[].kind() == PREP_KIND_BINARIZER or slot[].kind() == PREP_KIND_NORMALIZER


def _name(slot: _Slot) -> String:
    return _kind_name(slot[].kind())


def _dense(view: MemberMatrix, what: String, estimator: String) raises:
    if not view.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + estimator + " member reads " + what + " as a dense host float32 view")


def _whole(call: _Call, estimator: String) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads whole host views (no row or column index)")


def _x_input(call: _Call, words: Int) -> XprepRunInput:
    # The door's record of X: copied into the arena by the transport, or put
    # on the device directly when the image has the put and X is large.
    var put = (call[].route.flags & PREP_ROUTE_DEV_PUT) != 0 and words >= _PUT_WORDS
    return XprepRunInput(0, words, call[].x.data, -1, put, words, False, 0, 0, 0)


def _execute(call: _Call, mut arena: List[Float32], host: Int, scratch: Int, output: Int, out_words: Int,
    program: List[Int32], inputs: List[XprepRunInput]) raises:
    """One program run through the route's transport (the door's
    `xprep_run_records` RUN and cleanup, without its Python records)."""
    var flags = call[].route.flags
    var has_out = (flags & PREP_ROUTE_OUT) != 0
    var has_scratch = (flags & PREP_ROUTE_SCRATCH) != 0
    var device_scratch = scratch > 0 and (has_out or has_scratch)
    var ranges = (scratch == 0 or has_out or has_scratch) and (flags & PREP_ROUTE_RANGES) != 0
    var key = call[].route.second
    var owned = List[Int](capacity=2 * len(inputs))
    try:
        xprep_execute(key, Int(arena.unsafe_ptr()), host, scratch, output, out_words, program, inputs, List[Int](),
            XprepRoute(ranges, device_scratch, has_out), owned)
    except error:
        var cleanup = xprep_release(key, owned)
        if cleanup.byte_length() > 0:
            raise Error("RuntimeError: x_prep operation failed: " + String(error) + "; cleanup also failed: " + cleanup)
        raise error^
    var cleanup = xprep_release(key, owned)
    if cleanup.byte_length() > 0:
        raise Error("RuntimeError: x_prep slot cleanup failed: " + cleanup)
    _ = arena


def _fit_robust(slot: _Slot, call: _Call) raises:
    var estimator = _name(slot)
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member takes no sample_weight")
    if call[].fit_output != 0 and call[].fit_output != MEMBER_OP_TRANSFORM:
        member_unsupported(estimator, call[].fit_output)
    var n = call[].x.rows
    var d = call[].x.cols
    var project = call[].fit_output == MEMBER_OP_TRANSFORM
    if project:
        _dense(call[].dst, "the output buffer", estimator)
        if call[].dst.rows != n or call[].dst.cols != d:
            raise Error("ValueError: the output buffer holds one value per element of X")
    var centering = slot[].words[1] != 0.0
    var scaling = slot[].words[2] != 0.0
    var lo = slot[].words[3]
    var hi = slot[].words[4]
    var unit = slot[].words[5] != 0.0
    var select = slot[].words[6] != 0.0
    var blocked = slot[].words[7] != 0.0
    var work = slot[].words[8] != 0.0
    # The door's arena layout (bindings/robust_native_pack.mojo robust_fit).
    var blocks = (n + 2047) // 2048
    var sort_words = 0 if select else n * d
    var part_words = 5 * blocks * d if blocked else 0
    var scratch = sort_words + part_words if work else 0
    var sorted = n * d
    var stats = n * d + (0 if work else sort_words)
    var fractions = stats + 6 * d
    var q = fractions + 3
    var center = q + 3 * d
    var scale = center + d
    var host = scale + d + (0 if work else part_words)
    var fused = project and n * d < _FUSED_LIMIT
    var dest = host
    if fused:
        host += n * d
    var part = scale + d
    if work:
        sorted = host
        part = host + sort_words
    if host + scratch > _INT32_BOUND:
        raise Error("x_prep: the program exceeds the native Int32 indexing bound")
    var flags = call[].route.flags
    var device_scratch = scratch > 0 and (flags & (PREP_ROUTE_OUT | PREP_ROUTE_SCRATCH)) != 0
    var arena = List[Float32](length=max(host + (0 if device_scratch else scratch), 1), fill=Float32(0))
    var fraction_values: List[Float32] = [Float32(lo / 100.0), Float32(0.5), Float32(hi / 100.0)]
    var inputs = List[XprepRunInput](capacity=2)
    inputs.append(_x_input(call, n * d))
    inputs.append(XprepRunInput(fractions, 3, Int(fraction_values.unsafe_ptr()), -1, False, 0, False, 0, 0, 0))
    var program = List[Int32]()
    robust_fit_program(program, select, blocked, unit, fused, centering, scaling, n, d, blocks, sorted, part, stats,
        fractions, q, center, scale, dest)
    _execute(call, arena, host, scratch, 0, 0, program, inputs)
    _ = fraction_values^
    slot[].clear()
    var ap = _F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    if centering:
        slot[].center = List[Float32](length=d, fill=Float32(0))
        unsafe_memcpy(dest=slot[].center.unsafe_ptr(), src=ap + center, count=d)
    if scaling:
        slot[].scale = List[Float32](length=d, fill=Float32(0))
        unsafe_memcpy(dest=slot[].scale.unsafe_ptr(), src=ap + scale, count=d)
    slot[].features = d
    slot[].scalars = [Float64(d)]
    slot[].fitted = True
    if fused:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data), src=ap + dest, count=n * d)
    _ = arena^
    if project and not fused:
        _transform_robust(slot, call)


def _transform_robust(slot: _Slot, call: _Call) raises:
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: This RobustScaler instance is not fitted yet. Call 'fit' with appropriate "
                    + "arguments before using transform.")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    _dense(call[].dst, "the output buffer", estimator)
    var n = call[].x.rows
    var d = slot[].features
    if call[].x.cols != d:
        raise Error("RobustScaler input feature count differs from fit")
    if call[].dst.rows != n or call[].dst.cols != d:
        raise Error("ValueError: the output buffer holds one value per element of X")
    var has_center = len(slot[].center) > 0
    var has_scale = len(slot[].scale) > 0
    # The door's layout (robust_transform).
    var center_at = n * d if has_center else -1
    var scale_at = n * d + (d if has_center else 0) if has_scale else -1
    var dest = n * d + (d if has_center else 0) + (d if has_scale else 0)
    if dest + n * d > _INT32_BOUND:
        raise Error("x_prep: the program exceeds the native Int32 indexing bound")
    var dev_out = n * d >= _FUSED_LIMIT and (call[].route.flags & PREP_ROUTE_OUT) != 0
    var host = dest + (0 if dev_out else n * d)
    var arena = List[Float32](length=max(host, 1), fill=Float32(0))
    var inputs = List[XprepRunInput](capacity=3)
    inputs.append(_x_input(call, n * d))
    if has_center:
        inputs.append(XprepRunInput(center_at, d, Int(slot[].center.unsafe_ptr()), -1, False, 0, False, 0, 0, 0))
    if has_scale:
        inputs.append(XprepRunInput(scale_at, d, Int(slot[].scale.unsafe_ptr()), -1, False, 0, False, 0, 0, 0))
    var program = List[Int32]()
    _stage(program, 3, n * d, 0, n * d, d, center_at, scale_at, dest)
    # A large output comes back through the image's output region straight
    # into the caller's buffer (the door's output store).
    _execute(call, arena, host, 0, call[].dst.data if dev_out else 0, n * d if dev_out else 0, program, inputs)
    if not dev_out:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data),
            src=_F32P(unsafe_from_address=Int(arena.unsafe_ptr())) + dest, count=n * d)
    _ = arena^


def _fit_maxabs(slot: _Slot, call: _Call) raises:
    """The door's `maxabs_fit` (bindings/maxabs_native_door.mojo) over the
    caller's X: the same arena layout, program and fused fit_transform."""
    var estimator = _name(slot)
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member takes no sample_weight")
    if call[].fit_output != 0 and call[].fit_output != MEMBER_OP_TRANSFORM:
        member_unsupported(estimator, call[].fit_output)
    var n = call[].x.rows
    var d = call[].x.cols
    var project = call[].fit_output == MEMBER_OP_TRANSFORM
    if project:
        _dense(call[].dst, "the output buffer", estimator)
        if call[].dst.rows != n or call[].dst.cols != d:
            raise Error("ValueError: the output buffer holds one value per element of X")
    # Existing row blocks are 2048 rows, independent of benchmark shapes.
    var blocked = slot[].words[1] != 0.0
    var blocks = (n + 2047) // 2048
    var work = blocked and slot[].words[2] != 0.0
    var scratch = 5 * blocks * d if work else 0
    var host = n * d + ((2 * d + (0 if work else 5 * blocks * d)) if blocked else 7 * d)
    # fit_transform: the scaled X takes n*d words after the fit's host words
    # (the transform program's affine stage, operands and order).
    var fused = project and n * d < _FUSED_LIMIT
    var dest = host
    if fused:
        host += n * d
    if host + scratch > _INT32_BOUND:
        raise Error("x_prep: the program exceeds the native Int32 indexing bound")
    var flags = call[].route.flags
    var device_scratch = scratch > 0 and (flags & (PREP_ROUTE_OUT | PREP_ROUTE_SCRATCH)) != 0
    var arena = List[Float32](length=max(host + (0 if device_scratch else scratch), 1), fill=Float32(0))
    var sc = n * d
    var program = List[Int32]()
    var ma = maxabs_fit_program(program, blocked, work, fused, n, d, blocks, host, sc, dest)
    var inputs = List[XprepRunInput](capacity=1)
    inputs.append(_x_input(call, n * d))
    _execute(call, arena, host, scratch, 0, 0, program, inputs)
    var ap = _F32P(unsafe_from_address=Int(arena.unsafe_ptr()))
    # scikit-learn refuses an infinite input (NaN alone is allowed); the
    # column maxima of |X| are infinite exactly for those columns.
    scaler_refuse_infinite(ap + ma, d, "ValueError: Input X contains infinity or a value too large for dtype('float64').")
    slot[].clear()
    slot[].center = List[Float32](length=d, fill=Float32(0))
    unsafe_memcpy(dest=slot[].center.unsafe_ptr(), src=ap + ma, count=d)
    slot[].scale = List[Float32](length=d, fill=Float32(0))
    unsafe_memcpy(dest=slot[].scale.unsafe_ptr(), src=ap + sc, count=d)
    slot[].features = d
    slot[].scalars = [Float64(d), Float64(n)]
    slot[].fitted = True
    if fused:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data), src=ap + dest, count=n * d)
    _ = arena^
    if project and not fused:
        _transform_maxabs(slot, call)


def _transform_maxabs(slot: _Slot, call: _Call) raises:
    """The door's `_maxabs_apply`: stage 3 with scale_ after X."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: This MaxAbsScaler instance is not fitted yet. Call 'fit' with appropriate "
                    + "arguments before using transform.")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    _dense(call[].dst, "the output buffer", estimator)
    var n = call[].x.rows
    var d = slot[].features
    if call[].x.cols != d:
        raise Error("MaxAbsScaler input feature count differs from fit")
    if call[].dst.rows != n or call[].dst.cols != d:
        raise Error("ValueError: the output buffer holds one value per element of X")
    var dest = n * d + d
    if dest + n * d > _INT32_BOUND:
        raise Error("x_prep: the program exceeds the native Int32 indexing bound")
    var dev_out = n * d >= _FUSED_LIMIT and (call[].route.flags & PREP_ROUTE_OUT) != 0
    var host = dest + (0 if dev_out else n * d)
    var arena = List[Float32](length=max(host, 1), fill=Float32(0))
    var inputs = List[XprepRunInput](capacity=2)
    inputs.append(_x_input(call, n * d))
    inputs.append(XprepRunInput(n * d, d, Int(slot[].scale.unsafe_ptr()), -1, False, 0, False, 0, 0, 0))
    var program = List[Int32]()
    _stage(program, 3, n * d, 0, n * d, d, -1, n * d, dest)
    # A large output comes back through the image's output region straight
    # into the caller's buffer (the door's output store).
    _execute(call, arena, host, 0, call[].dst.data if dev_out else 0, n * d if dev_out else 0, program, inputs)
    if not dev_out:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data),
            src=_F32P(unsafe_from_address=Int(arena.unsafe_ptr())) + dest, count=n * d)
    _ = arena^


def _fit_stateless(slot: _Slot, call: _Call) raises:
    """Binarizer / Normalizer fit: X's width, as the doors' fit records it
    (nothing is uploaded or computed); fit_transform then transforms."""
    var estimator = _name(slot)
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].weights.rows > 0:
        raise Error("TypeError: a " + estimator + " member takes no sample_weight")
    if call[].fit_output != 0 and call[].fit_output != MEMBER_OP_TRANSFORM:
        member_unsupported(estimator, call[].fit_output)
    slot[].clear()
    slot[].features = call[].x.cols
    slot[].scalars = [Float64(call[].x.cols)]
    slot[].fitted = True
    if call[].fit_output == MEMBER_OP_TRANSFORM:
        _transform_stateless(slot, call)


def _transform_stateless(slot: _Slot, call: _Call) raises:
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
    if call[].dst.rows != n or call[].dst.cols != d:
        raise Error("ValueError: the output buffer holds one value per element of X")
    var words = n * d
    var program = List[Int32]()
    var inputs = List[XprepRunInput](capacity=2)
    inputs.append(_x_input(call, words))
    if slot[].kind() == PREP_KIND_BINARIZER:
        # X, the threshold word, then the output as arena words (the door's
        # binarizer branch; no output region).
        var threshold: List[Float32] = [Float32(slot[].words[1])]
        var out = words + 1
        var host = out + words
        if host > _INT32_BOUND:
            raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
        inputs.append(XprepRunInput(words, 1, Int(threshold.unsafe_ptr()), -1, False, 0, False, 0, 0, 0))
        _stage(program, OP_BINARIZE, words, 0, words, words, out)
        var arena = List[Float32](length=max(host, 1), fill=Float32(0))
        _execute(call, arena, host, 0, 0, 0, program, inputs)
        _ = threshold^
        unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data),
            src=_F32P(unsafe_from_address=Int(arena.unsafe_ptr())) + out, count=words)
        _ = arena^
        return
    # Normalizer (the door's _normalizer_apply): a large output comes back
    # through the image's output region straight into the caller's buffer.
    var dev_out = words >= OUT_MIN_WORDS and (call[].route.flags & PREP_ROUTE_OUT) != 0
    var host = words + (0 if dev_out else words)
    if host + (words if dev_out else 0) > _INT32_BOUND:
        raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
    _stage(program, OP_NORMALIZE, n, 0, n, d, Int(slot[].words[1]), host if dev_out else words)
    var arena = List[Float32](length=max(host, 1), fill=Float32(0))
    _execute(call, arena, host, 0, call[].dst.data if dev_out else 0, words if dev_out else 0, program, inputs)
    if not dev_out:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=call[].dst.data),
            src=_F32P(unsafe_from_address=Int(arena.unsafe_ptr())) + words, count=words)
    _ = arena^


def _export(slot: _Slot, call: _Call) raises:
    """A view of one fitted block. The slot owns the words; they stay valid
    until this state's next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot) + " member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > (0 if _stateless(slot) else 2):
            raise Error("member ABI: a " + _name(slot) + " member has no fitted block index " + String(block))
        block = MEMBER_BLOCK_SCALARS if block == 0 else (PREP_BLOCK_CENTER if block == 1 else PREP_BLOCK_SCALE)
    if block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif _stateless(slot):
        raise Error("member ABI: a " + _name(slot) + " member has no fitted block " + String(block))
    elif block == PREP_BLOCK_CENTER:
        call[].dst = MemberMatrix(Int(slot[].center.unsafe_ptr()), 1, len(slot[].center))
    elif block == PREP_BLOCK_SCALE:
        call[].dst = MemberMatrix(Int(slot[].scale.unsafe_ptr()), 1, len(slot[].scale))
    else:
        raise Error("member ABI: a " + _name(slot) + " member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 1 if _stateless(slot) else 3


def _import(slot: _Slot, call: _Call) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var estimator = _name(slot)
    var block = call[].block
    if slot[].fitted:
        slot[].clear()
    if block == MEMBER_BLOCK_SCALARS:
        var count = 2 if slot[].kind() == PREP_KIND_MAXABS else 1
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != count:
            raise Error("member ABI: the " + estimator + " scalars block is " + String(count) + " host float64 words")
        var given = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].x.data)
        slot[].scalars = List[Float64](length=count, fill=0.0)
        for index in range(count):  # driver-loop: one or two scalar words
            slot[].scalars[index] = given[index]
    elif (block == PREP_BLOCK_CENTER or block == PREP_BLOCK_SCALE) and not _stateless(slot):
        var size = call[].x.rows * call[].x.cols
        var words = List[Float32](length=size, fill=Float32(0))
        if size > 0:
            _dense(call[].x, "a fitted block", estimator)
            unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=size)
        if block == PREP_BLOCK_CENTER:
            slot[].center = words^
        else:
            slot[].scale = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a " + estimator + " member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var maxabs = slot[].kind() == PREP_KIND_MAXABS
    if len(slot[].scalars) != (2 if maxabs else 1):
        raise Error("member ABI: a fitted " + estimator + " state needs its scalars block")
    var d = Int(slot[].scalars[0])
    if maxabs and (len(slot[].center) != d or len(slot[].scale) != d):
        raise Error("member ABI: a fitted MaxAbsScaler state needs max_abs_ and scale_, one value per feature")
    if (d < 1 or (len(slot[].center) != 0 and len(slot[].center) != d)
            or (len(slot[].scale) != 0 and len(slot[].scale) != d)):
        raise Error("member ABI: a fitted " + estimator + " state needs every fitted block, of the fitted sizes")
    slot[].features = d
    slot[].fitted = True


def _outputs(call: _Call) raises:
    """`call.out_cols` of transform: the fitted feature count (X's width)."""
    call[].classes = 0
    if call[].state == 0:
        # Asked before any state (`member_outputs`): a scaler keeps X's width.
        if call[].flags != MEMBER_OP_TRANSFORM:
            var kind = PREP_KIND_ROBUST
            if call[].param_count >= 1 and call[].params:
                kind = Int(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)[])
            member_unsupported(_kind_name(kind), call[].flags)
        call[].out_cols = call[].x.cols
        return
    var slot = _Slot(unsafe_from_address=_lookup(call[].state, False))
    if call[].flags != MEMBER_OP_TRANSFORM:
        member_unsupported(_name(slot), call[].flags)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot) + " member's output width is known after its fit")
    call[].out_cols = slot[].features


def _prep_member(operation: Int, address: Int) raises -> Int:
    var call = _Call(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        # Kind-exact as well: RobustScaler and MaxAbsScaler honor the same
        # operations (no inverse_transform, no weights, no index views).
        return MEMBER_CAN_FIT | MEMBER_CAN_TRANSFORM | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
    if operation == MEMBER_OP_NEW:
        if call[].param_count < 1 or not call[].params:
            raise Error("member ABI: a Prep member takes its parameter words")
        var source = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)
        if source[0] == Float64(PREP_KIND_ROBUST):
            if call[].param_count != PREP_ROBUST_PARAM_COUNT:
                raise Error("member ABI: a RobustScaler member takes nine parameters")
        elif source[0] == Float64(PREP_KIND_MAXABS):
            if call[].param_count != PREP_MAXABS_PARAM_COUNT:
                raise Error("member ABI: a MaxAbsScaler member takes three parameters")
        elif source[0] == Float64(PREP_KIND_BINARIZER) or source[0] == Float64(PREP_KIND_NORMALIZER):
            if call[].param_count != PREP_STATELESS_PARAM_COUNT:
                raise Error("member ABI: a " + _kind_name(Int(source[0])) + " member takes two parameters")
            if source[0] == Float64(PREP_KIND_NORMALIZER) and source[1] != 0.0 and source[1] != 1.0 and source[1] != 2.0:
                raise Error("member ABI: unknown Normalizer norm code")
        else:
            raise Error("member ABI: unknown or unpublished Prep member kind")
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
        if _stateless(slot):
            _fit_stateless(slot, call)
        elif slot[].kind() == PREP_KIND_MAXABS:
            _fit_maxabs(slot, call)
        else:
            _fit_robust(slot, call)
    elif operation == MEMBER_OP_TRANSFORM:
        if _stateless(slot):
            _transform_stateless(slot, call)
        elif slot[].kind() == PREP_KIND_MAXABS:
            _transform_maxabs(slot, call)
        else:
            _transform_robust(slot, call)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_name(slot), operation)
    return 0


def prep_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of the Prep family (schema `MEMBER_ABI`)."""
    try:
        var word = _prep_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def prep_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's Prep member entry. One line in the image's
    module initialization, after its kernel tables are published."""
    member_publish[MEMBER_FAMILY_PREP](mode, vendor, publisher, prep_member_v2)

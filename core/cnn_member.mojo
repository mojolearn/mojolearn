# SPDX-License-Identifier: Apache-2.0
"""CNNClassifier as a MEMBER (core/member_abi.mojo).

One entry, `cnn_member_v2`, published by each `_mojolearn_x_sequence` image
(`cnn_member_publish`, beside the MLP member): CNN_FIT, CNN_OPEN and
CNN_PREDICT run on the x_sequence model descriptor entry of the model's
tier and device, as the estimator's own door
(bindings/xcnn_classifier_native_door.mojo) runs them. It calls the
family's existing typed RUN (x_cnn/classifier_run.mojo: the layer layout,
the seeded Kaiming draws, the descriptor words) and no kernel is touched.

ROUTE SLOTS (filled by providers/python/member_cnn_boundary_door.mojo):
  * `entry`: the model descriptor entry of the route's x_sequence image
    (MODEL_UNIFORM, CNN_FIT, CNN_OPEN, CNN_PREDICT, SESSION_CLOSE); its
    `word` is the address of that provider's retained call workspace.
  No other slot is read; no score is published (the classifier's accuracy
  runs on the metrics owner: the caller predicts and runs its own metric).

MEMBER_OP_NEW parameters (Float64 words, `24 + B` of them):
  [0] kind (0 CNNClassifier); [1] C; [2] H; [3] W (input_shape);
  [4] kernel_size; [5] pool_size; [6] batch_size; [7] max_iter; [8] shuffle
  (0 / 1); [9] optimizer (0 sgd, 1 adam, 2 adamw); [10] nesterov (0 / 1);
  [11] learning_rate; [12] momentum; [13] beta1; [14] beta2; [15] eps;
  [16] weight_decay; [17] dampening; [18] layout seed, low 32 bits;
  [19] layout seed, high 32 bits (`random_state` as a 64-bit integer, 0 when
  None: the seed of the initial Kaiming draws, as the estimator's `_build`);
  [20] fresh seed (1: `random_state` is None, so every FIT draws eight bytes
  of OS entropy for the training stream, as the estimator's own fit does,
  and words 21 and 22 are not read); [21] fit seed, low 32 bits; [22] fit
  seed, high 32 bits (the estimator's seeded stream word); [23] B, the conv
  block count; [24 .. 24 + B) conv_channels.

VIEWS: X dense row-major host float32, rows x C*H*W (each row one image,
C-order); `x.steps` 0, 1 or C (rows x C x H*W reads the same bytes).
y host int32 class codes 0 .. classes-1 (rows x 1), `call.classes` >= 1 at
fit. Predict writes int32 codes (rows x 1); predict_proba float32
rows x classes. No sample weights; a device-home view, a column index and a
nonzero `fit_output` are refused in words. X and y are borrowed for the
call: nothing is copied. ROW INDEX and SCOPE (lane vdf-scope-neural2, as the
MLP member; answered only on a route holding the GPU image's member store,
core/seq_member_rows.mojo): `call.row_index` names rows of the WHOLE X and
y; with the caller's scope and pin the whole X stays resident in the scope's
"x_sequence" cache, the fold is taken on the device and CNN_FIT / CNN_PREDICT
read it in place; without a pin the fold is gathered on the host. The fold's
class codes are taken in host row tasks.

FITTED-STATE BLOCKS, in index order:
  0 `MEMBER_BLOCK_SCALARS` float64 [classes, features, blocks, arrays];
  1 `MEMBER_BLOCK_FAMILY` float32 1 x max_iter: the epoch loss curve;
  2 `MEMBER_BLOCK_FAMILY + 1` float32 1 x (max_iter * steps per epoch):
    every minibatch loss;
  3 + i `MEMBER_BLOCK_FAMILY + 2 + i` float32 1 x count: learned array i in
    descriptor order (per block weight (oc, c, k, k) then bias (oc), then
    the head's weight (classes, flat) and bias (classes)).
An import needs block 0 and every learned array; the two loss blocks are
optional. A curve is empty when max_iter is 0.

RESIDENT PREDICT: the first predict of a fitted state opens one resident
session (CNN_OPEN: every learned array uploaded once, the forward captured);
later predicts upload only the images (CNN_PREDICT), as the estimator's own
predict record does. A refit, an import and FREE close it (SESSION_CLOSE).
The session table is locked (autograd/ops_api.mojo), so members open,
predict and close in parallel workers with the GIL released.

The fitted state is OWNED here, in a process-wide store of fixed chunks: a
slot never moves once minted; the store's lock covers only mint, free and
the handle lookup. SERIAL BY NATURE: the store's bookkeeping and the
per-layer driver loops over model structure; the draws, the training and
the forward are the image's own parallel kernels.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from core.device_route import DeviceRoute
from core.entry_table import EntryRef, entry_ref_unresolved
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_PREDICT_PROBA, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_CAN_CLASSIFIER, MEMBER_CAN_STEPS,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_CNN
from core.mlp_member import _entropy_seed
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from sequence.neural_model_dispatch import CNN_FIT, CNN_PREDICT, CNN_OPEN, SESSION_CLOSE
from sequence.xseq_neural_run import XseqNeuralEntry, xseq_neural_run_call
from core.seq_member_rows import seq_rows_route, seq_rows_open, seq_rows_close, seq_rows_count
from core.member_abi import MEMBER_CAN_ROW_INDEX, MEMBER_CAN_SCOPE
from x_cnn.classifier_run import (
    CnnClfPlan, CnnClfFitSettings, CnnClfLayout, cnnclf_run_layout, cnnclf_run_init, cnnclf_run_check_input_dim,
    cnnclf_run_check_channels, cnnclf_run_check_setting, cnnclf_run_check_nesterov, cnnclf_run_features,
    cnnclf_run_fit_steps, cnnclf_run_fit_addresses, cnnclf_run_fit_ints, cnnclf_run_fit_floats,
    cnnclf_run_open_addresses, cnnclf_run_open_ints, cnnclf_run_check_session, cnnclf_run_predict_ints,
)

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _FIXED_WORDS = 24
comptime _SCALAR_WORDS = 4
comptime _BLOCK_CURVE = MEMBER_BLOCK_FAMILY
comptime _BLOCK_LOSSES = MEMBER_BLOCK_FAMILY + 1
comptime _BLOCK_ARRAYS = MEMBER_BLOCK_FAMILY + 2
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _NAME = "CNNClassifier"


struct CnnMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var fitted: Bool
    # The NEW words (hyperparameters), kept whole, and the conv widths.
    var words: List[Float64]
    var channels: List[Int]
    # The fitted state: classes, the block pool flags of the layout, every
    # learned array (descriptor order) and the loss records.
    var classes: Int
    var pools: List[Int]
    var arrays: List[List[Float32]]
    var curve: List[Float32]
    var losses: List[Float32]
    var scalars: List[Float64]
    # The resident predict session (0: none open): its id, the two scope
    # words CNN_OPEN wrote, and the entry it lives on.
    var sid: Int
    var scope0: Int
    var scope1: Int
    var session_entry: EntryRef

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.fitted = False
        self.words = List[Float64]()
        self.channels = List[Int]()
        self.classes = 0
        self.pools = List[Int]()
        self.arrays = List[List[Float32]]()
        self.curve = List[Float32]()
        self.losses = List[Float32]()
        self.scalars = List[Float64]()
        self.sid = 0
        self.scope0 = 0
        self.scope1 = 0
        self.session_entry = entry_ref_unresolved()


struct CnnMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[CnnMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[CnnMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[CnnMemberStore, "MojolearnCnnMemberStoreV2", 2]

comptime _SlotP = MutPointer[CnnMemberSlot, MutUntrackedOrigin]
comptime _CallP = MutPointer[MemberCall, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[CnnMemberSlot]()


def _mint(var words: List[Float64], var channels: List[Int]) raises -> Int:
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live CNN member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[CnnMemberSlot](length=_CHUNK, fill=CnnMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _SlotP(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].fitted = False
        slot[].words = words^
        slot[].channels = channels^
        return slot[].generation * _INDEX_SPAN + index + 1
    finally:
        shared_slot_unlock(lock)


def _lookup(handle: Int, release: Bool) raises -> Int:
    """The slot address of a live handle. A released handle, a handle of
    another store or a made-up integer is refused: never a stale read. The
    caller closed the slot's session before a release."""
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
            slot[].fitted = False
            slot[].words = List[Float64]()
            slot[].channels = List[Int]()
            slot[].classes = 0
            slot[].pools = List[Int]()
            slot[].arrays = List[List[Float32]]()
            slot[].curve = List[Float32]()
            slot[].losses = List[Float32]()
            slot[].scalars = List[Float64]()
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _dense(view: MemberMatrix, dtype: Int, what: String) raises:
    if not view.dense_host(dtype):
        var kind = String("float32") if dtype == MEMBER_F32 else (
            String("int32") if dtype == MEMBER_I32 else String("float64"))
        raise Error("TypeError: a CNNClassifier member reads " + what + " as a dense host " + kind + " view")


def _word_int(w: _F64P, lo: Int) -> Int:
    # Two 32-bit words back to the 64-bit integer (two's complement kept).
    return Int(UInt64(Int(w[lo])) | (UInt64(Int(w[lo + 1])) << 32))


def _features(slot: _SlotP) -> Int:
    var w = _F64P(unsafe_from_address=Int(slot[].words.unsafe_ptr()))
    return Int(w[1]) * Int(w[2]) * Int(w[3])


def _images(slot: _SlotP, view: MemberMatrix) raises:
    """X as images: dense host float32, one C*H*W row per image."""
    _dense(view, MEMBER_F32, "X")
    var w = _F64P(unsafe_from_address=Int(slot[].words.unsafe_ptr()))
    var features = _features(slot)
    if view.cols != features:
        raise Error("ValueError: images do not match the model input_shape")
    if view.steps > 1 and view.steps != Int(w[1]):
        raise Error("ValueError: a CNNClassifier member reads a row as C x (H*W): x.steps is C")


def _layout(slot: _SlotP, classes: Int) -> CnnClfLayout:
    var w = _F64P(unsafe_from_address=Int(slot[].words.unsafe_ptr()))
    return cnnclf_run_layout(Int(w[1]), Int(w[2]), Int(w[3]), Int(w[4]), Int(w[5]), slot[].channels,
                             classes, _word_int(w, 18))


def _plan(slot: _SlotP, n: Int) -> CnnClfPlan:
    var w = _F64P(unsafe_from_address=Int(slot[].words.unsafe_ptr()))
    return CnnClfPlan(n, Int(w[1]), Int(w[2]), Int(w[3]), slot[].classes, Int(w[4]), Int(w[5]),
                      slot[].channels.copy(), slot[].pools.copy())


def _array_addresses(slot: _SlotP) -> List[Int]:
    var out = List[Int]()
    for i in range(len(slot[].arrays)):  # driver-loop: over the learned arrays (model structure)
        out.append(Int(slot[].arrays[i].unsafe_ptr()))
    return out^


def _close_session(slot: _SlotP) raises:
    """SESSION_CLOSE of the slot's resident predict session, if open."""
    if slot[].sid == 0:
        return
    var sid = slot[].sid
    var scope: List[Int] = [slot[].scope0, slot[].scope1]
    slot[].sid = 0
    slot[].scope0 = 0
    slot[].scope1 = 0
    var entry = XseqNeuralEntry(slot[].session_entry.call, slot[].session_entry.word)
    slot[].session_entry = entry_ref_unresolved()
    var addresses: List[Int] = [sid]
    _ = xseq_neural_run_call(entry, SESSION_CLOSE, addresses, scope, List[Float64]())


def _fit(slot: _SlotP, call: _CallP) raises:
    var route = call[].route
    if call[].fit_output != 0:
        member_unsupported(_NAME, call[].fit_output)
    if call[].weights.rows > 0:
        raise Error("TypeError: a CNNClassifier member fits without sample weights")
    _images(slot, call[].x)
    var n = seq_rows_count(call)
    _dense(call[].y, MEMBER_I32, "y (class codes)")
    if call[].y.cols != 1:
        raise Error("NotImplementedError: multilabel targets are not implemented")
    if call[].y.rows != call[].x.rows:
        raise Error("ValueError: X and y must contain the same number of rows")
    var k = call[].classes
    if k < 1:
        raise Error("ValueError: a CNNClassifier member fit needs its class count")
    var w = _F64P(unsafe_from_address=Int(slot[].words.unsafe_ptr()))
    # A refit publishes new weights: the old resident session goes now.
    _close_session(slot)
    slot[].fitted = False
    var entry = XseqNeuralEntry(route.entry.call, route.entry.word)
    var layout = _layout(slot, k)
    slot[].classes = k
    slot[].arrays = List[List[Float32]]()
    for i in range(len(layout.counts)):  # driver-loop: over the learned arrays (model structure)
        slot[].arrays.append(List[Float32](length=layout.counts[i], fill=0.0))
    var params = _array_addresses(slot)
    # The initial Kaiming draws (`_build`), on the model's image.
    cnnclf_run_init(entry, layout, params)
    slot[].pools = layout.pools.copy()
    var max_iter = Int(w[7])
    var batch = Int(w[6])
    if max_iter > 0:
        var nsteps = cnnclf_run_fit_steps(n, batch)
        slot[].curve = List[Float32](length=max_iter, fill=0.0)
        slot[].losses = List[Float32](length=max_iter * nsteps, fill=0.0)
        var seed: UInt64
        if w[20] != 0.0:
            # random_state=None: a fresh training stream at every fit, so
            # every fold or candidate of a caller draws its own shuffles.
            seed = _entropy_seed()
        else:
            seed = UInt64(Int(w[21])) | (UInt64(Int(w[22])) << 32)
        var settings = CnnClfFitSettings(batch, max_iter, w[8] != 0.0, seed, Int(w[9]), w[10] != 0.0,
            w[11], w[12], w[13], w[14], w[15], w[16], w[17])
        # THE RESIDENT ROUTE (core/seq_member_rows.mojo): X in the scope's
        # store and the fold taken on the device (read in place by CNN_FIT),
        # or the fold gathered on the host; the fold's codes in host row tasks.
        var held = seq_rows_open(call, 1, _NAME)
        try:
            var addresses = cnnclf_run_fit_addresses(held.x, held.y, Int(slot[].curve.unsafe_ptr()),
                Int(slot[].losses.unsafe_ptr()), params)
            _ = xseq_neural_run_call(entry, CNN_FIT, addresses, cnnclf_run_fit_ints(_plan(slot, n), settings),
                cnnclf_run_fit_floats(settings))
        finally:
            seq_rows_close(call, held)
        _ = held^  # the gathered words live past the run
    else:
        slot[].curve = List[Float32]()
        slot[].losses = List[Float32]()
    slot[].scalars = [Float64(k), Float64(_features(slot)), Float64(len(slot[].channels)),
        Float64(len(slot[].arrays))]
    slot[].fitted = True


def _predict_into(slot: _SlotP, call: _CallP, codes: Bool) raises:
    """One CNN_PREDICT on the slot's resident session (opened first)."""
    if not slot[].fitted:
        raise Error("NotFittedError: this CNNClassifier member is not fitted yet")
    if call[].column_index.rows > 0:
        raise Error("TypeError: a CNNClassifier member reads every column of X (no column index)")
    if call[].x.rows == 0:
        # The door returns the empty result before any session (lane
        # fx-audit 17): no open, no CNN_PREDICT, nothing written.
        if call[].x.cols != _features(slot):
            raise Error("ValueError: images do not match the model input_shape")
        return
    _images(slot, call[].x)
    var n = seq_rows_count(call)
    if codes:
        _dense(call[].dst, MEMBER_I32, "the class-code buffer")
        if call[].dst.rows != n or call[].dst.cols != 1:
            raise Error("ValueError: the class-code buffer holds one code per row of X")
    else:
        _dense(call[].dst, MEMBER_F32, "the probability buffer")
        if call[].dst.rows != n or call[].dst.cols != slot[].classes:
            raise Error("ValueError: the probability buffer holds one row of class probabilities per row of X")
    if slot[].sid == 0:
        var route = call[].route
        var entry = XseqNeuralEntry(route.entry.call, route.entry.word)
        var scope = List[Int](length=2, fill=0)
        var addresses = cnnclf_run_open_addresses(Int(scope.unsafe_ptr()), _array_addresses(slot))
        var sid = xseq_neural_run_call(entry, CNN_OPEN, addresses, cnnclf_run_open_ints(_plan(slot, 1)),
            List[Float64]())
        slot[].sid = sid
        slot[].scope0 = scope[0]
        slot[].scope1 = scope[1]
        slot[].session_entry = route.entry
    cnnclf_run_check_session(slot[].sid, 2)
    var entry = XseqNeuralEntry(slot[].session_entry.call, slot[].session_entry.word)
    # THE RESIDENT ROUTE: a store slot X carries its row width as the
    # descriptor's seventh word (sequence/cnn_classifier_program.mojo).
    var held = seq_rows_open(call, 0, _NAME)
    try:
        var addresses: List[Int] = [held.x, call[].dst.data]
        var ints = cnnclf_run_predict_ints(slot[].scope0, slot[].scope1, slot[].sid, n, codes)
        if held.x < 0:
            ints.append(_features(slot))
        _ = xseq_neural_run_call(entry, CNN_PREDICT, addresses, ints, List[Float64]())
    finally:
        seq_rows_close(call, held)
    _ = held^  # the gathered words live past the run


def _export(slot: _SlotP, call: _CallP) raises:
    """A view of one fitted block. The slot owns the words; they stay valid
    until this state's next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this CNNClassifier member is not fitted yet")
    var block = call[].block
    var count = len(slot[].arrays)
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= 3 + count:
            raise Error("member ABI: a CNNClassifier member has " + String(3 + count) + " fitted blocks")
        block = MEMBER_BLOCK_SCALARS if block == 0 else MEMBER_BLOCK_FAMILY + block - 1
    if block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block == _BLOCK_CURVE:
        call[].dst = MemberMatrix(Int(slot[].curve.unsafe_ptr()), 1, len(slot[].curve))
    elif block == _BLOCK_LOSSES:
        call[].dst = MemberMatrix(Int(slot[].losses.unsafe_ptr()), 1, len(slot[].losses))
    elif block >= _BLOCK_ARRAYS and block < _BLOCK_ARRAYS + count:
        var i = block - _BLOCK_ARRAYS
        call[].dst = MemberMatrix(Int(slot[].arrays[i].unsafe_ptr()), 1, len(slot[].arrays[i]))
    else:
        raise Error("member ABI: a CNNClassifier member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 3 + count


def _import(slot: _SlotP, call: _CallP) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var block = call[].block
    if block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        # Any import edits the fitted words: the resident session goes.
        _close_session(slot)
        slot[].fitted = False
    if block == MEMBER_BLOCK_SCALARS:
        _dense(call[].x, MEMBER_F64, "a fitted block")
        if call[].x.rows * call[].x.cols != _SCALAR_WORDS:
            raise Error("member ABI: the CNNClassifier scalars block is four host float64 words")
        var words = List[Float64](length=_SCALAR_WORDS, fill=0.0)
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F64P(unsafe_from_address=call[].x.data), count=_SCALAR_WORDS)
        slot[].scalars = words^
        var arrays = Int(slot[].scalars[3])
        if arrays != 2 * (len(slot[].channels) + 1):
            raise Error("ValueError: the CNNClassifier fitted state does not have this model's layers")
        slot[].arrays = List[List[Float32]](length=arrays, fill=List[Float32]())
    elif block >= _BLOCK_CURVE:
        _dense(call[].x, MEMBER_F32, "a fitted block")
        var count = call[].x.rows * call[].x.cols
        var words = List[Float32](length=count, fill=0.0)
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        if block == _BLOCK_CURVE:
            slot[].curve = words^
        elif block == _BLOCK_LOSSES:
            slot[].losses = words^
        else:
            var i = block - _BLOCK_ARRAYS
            if i >= len(slot[].arrays):
                raise Error("member ABI: import the CNNClassifier scalars block before its learned arrays")
            slot[].arrays[i] = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a CNNClassifier member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) != _SCALAR_WORDS:
        raise Error("member ABI: a fitted CNNClassifier state needs its scalars block")
    var k = Int(slot[].scalars[0])
    if call[].classes != 0:
        k = call[].classes
    if k < 1 or Int(slot[].scalars[1]) != _features(slot):
        raise Error("ValueError: the CNNClassifier fitted state does not have this model's layers")
    var layout = _layout(slot, k)
    if len(layout.counts) != len(slot[].arrays):
        raise Error("ValueError: the CNNClassifier fitted state does not have this model's layers")
    for i in range(len(layout.counts)):  # driver-loop: over the learned arrays (model structure)
        if len(slot[].arrays[i]) != layout.counts[i]:
            raise Error("ValueError: the CNNClassifier fitted state does not have this model's layers")
    slot[].classes = k
    slot[].scalars[0] = Float64(k)
    slot[].pools = layout.pools.copy()
    slot[].fitted = True


def _outputs(call: _CallP, classes: Int) raises:
    """OUTPUTS with or without a state: predict one int32 code column,
    predict_proba float32 x classes; another operation is refused."""
    call[].classes = classes
    if call[].flags == MEMBER_OP_PREDICT:
        call[].out_cols = 1
        call[].out_dtype = MEMBER_I32
    elif call[].flags == MEMBER_OP_PREDICT_PROBA:
        call[].out_cols = classes
        call[].out_dtype = MEMBER_F32
    else:
        member_unsupported("CNNClassifier", call[].flags)


def _capabilities() -> Int:
    return (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_EXPORT
        | MEMBER_CAN_IMPORT | MEMBER_CAN_CLASSIFIER | MEMBER_CAN_STEPS)


def _new(call: _CallP) raises:
    if call[].param_count < _FIXED_WORDS or not call[].params:
        raise Error("member ABI: a CNNClassifier member takes at least twenty-four parameters")
    var w = _F64P(unsafe_from_address=call[].params)
    var blocks = Int(w[23])
    if blocks < 0 or call[].param_count != _FIXED_WORDS + blocks:
        raise Error("member ABI: a CNNClassifier member takes 24 + (conv block count) parameters")
    if w[0] != 0.0:
        raise Error("member ABI: unknown CNN member kind")
    for i in range(1, 4):  # small-loop(the three declared image dimensions)
        cnnclf_run_check_input_dim(Int(w[i]))
    cnnclf_run_check_setting(0, Int(w[4]))
    cnnclf_run_check_setting(1, Int(w[5]))
    cnnclf_run_check_setting(2, Int(w[6]))
    cnnclf_run_check_setting(3, Int(w[7]))
    if w[9] < 0.0 or w[9] > 2.0:
        raise Error("ValueError: optimizer must be sgd, adam or adamw")
    if w[10] != 0.0:
        cnnclf_run_check_nesterov(True, w[12], w[17])
    if w[20] != 0.0 and w[20] != 1.0:
        raise Error("member ABI: a CNNClassifier member's fresh-seed word is 0 or 1")
    var words = List[Float64](length=call[].param_count, fill=0.0)
    unsafe_memcpy(dest=words.unsafe_ptr(), src=w, count=call[].param_count)
    var channels = List[Int]()
    for i in range(blocks):  # driver-loop: over the conv blocks (model structure)
        var width = Int(w[_FIXED_WORDS + i])
        cnnclf_run_check_channels(width)
        channels.append(width)
    call[].state = _mint(words^, channels^)


def _cnn_member(operation: Int, address: Int) raises -> Int:
    var call = _CallP(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        if call[].state != 0:
            _ = _lookup(call[].state, False)
        # THE RESIDENT ROUTE (core/seq_member_rows.mojo): a route holding the
        # image's member store reads a fold's rows of the whole X.
        var rows_bits = (MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE) if seq_rows_route(call[].route) else 0
        return _capabilities() | rows_bits
    if operation == MEMBER_OP_NEW:
        _new(call)
        return 0
    if operation == MEMBER_OP_OUTPUTS and call[].state == 0:
        # No state: answer from the words, as a state fitted on them would.
        _outputs(call, call[].classes)
        return 0
    var slot = _SlotP(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_FREE:
        try:
            _close_session(slot)
        except:
            pass  # FREE never raises; the session table drops it with the image
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_PREDICT:
        _predict_into(slot, call, True)
    elif operation == MEMBER_OP_PREDICT_PROBA:
        _predict_into(slot, call, False)
    elif operation == MEMBER_OP_OUTPUTS:
        _outputs(call, slot[].classes if slot[].fitted else call[].classes)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_NAME, operation)
    return 0


def cnn_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of CNNClassifier (schema `MEMBER_ABI`)."""
    try:
        var word = _cnn_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def cnn_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's CNN member entry. One line in each
    `_mojolearn_x_sequence` image's module initialization, after its glue
    entry is published. A build with no GPU vendor (`vendor` < 0) publishes
    nothing."""
    if vendor < 0:
        return
    member_publish[MEMBER_FAMILY_CNN](mode, vendor, publisher, cnn_member_v2)

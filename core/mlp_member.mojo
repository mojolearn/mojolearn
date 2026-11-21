# SPDX-License-Identifier: Apache-2.0
"""MLPRegressor and MLPClassifier as MEMBERS (core/member_abi.mojo).

One entry, `mlp_member_v2`, published by each `_mojolearn_x_sequence` image
(`mlp_member_publish`, beside its glue entry). It runs the family's existing
typed RUN (sequence/xseq_neural_run.mojo: the Glorot draws, MLP_FIT,
MLP_PREDICT, the threshold / argmax / curve-minimum helpers;
sequence/xseq_score_run.mojo: the R^2 refusals) over the entries of the
`DeviceRoute` it is handed, the same entries the estimator's own door
(bindings/xseq_neural_native_door.mojo) calls. No kernel is touched.

ROUTE SLOTS (filled by providers/python/member_mlp_boundary_door.mojo):
  * `entry`: the model descriptor entry of the route's image (MLP_FIT,
    MLP_PREDICT); its `word` is the address of that provider's retained call
    workspace (the lease the door's calls take; a busy one gives a private
    workspace), not a layout word.
  * `second`: the output-helper entry (the Glorot draws, the p > 0.5
    threshold, the row argmax, the curve minimum), the image the door's
    helpers run on; `word` = its workspace address, as above.
  * `third` (`has_third`): the R^2 entry of the metrics image of the route's
    device (regressor score). Without it a score is refused in words.

MEMBER_OP_NEW parameters (Float64 words, `21 + H` of them):
  [0] kind (0 MLPRegressor, 1 MLPClassifier); [1] activation (0 identity,
  1 logistic, 2 tanh, 3 relu); [2] solver (0 adam, 1 sgd); [3] learning_rate
  (0 constant, 1 invscaling, 2 adaptive); [4] learning_rate_init; [5] beta_1;
  [6] beta_2; [7] epsilon; [8] momentum; [9] power_t; [10] alpha; [11] tol;
  [12] nesterovs_momentum (0 / 1); [13] shuffle (0 / 1); [14] batch_size
  (0 = "auto", else the integer); [15] max_iter; [16] n_iter_no_change;
  [17] seed, low 32 bits; [18] seed, high 32 bits (the 64-bit word of the
  estimator's seeded stream for an integer `random_state`); [19] H, the
  hidden layer count; [20] fresh seed (1: `random_state` is None, so every
  FIT draws its own eight bytes of OS entropy, as the estimator's own fit
  does, and words 17 and 18 are not read; 0: the seed words);
  [21 .. 21 + H) the hidden layer widths.

VIEWS: X dense row-major host float32. Regressor: y host float32, rows x
outputs (multi-output allowed). Classifier: y host int32 class codes
0 .. classes-1 (rows x 1) and `call.classes` >= 2 at fit; predict writes int32
codes (rows x 1), predict_proba float32 rows x classes. No sample weights at
fit (no MEMBER_CAN_SAMPLE_WEIGHT): a given weight is refused. Score: the
regressor's R^2 of one output column, weighted when `call.weights` is given
(host float32, one per row); the classifier publishes no score (its accuracy
runs on the metrics owner, not split from Python): the caller predicts and
runs its own metric. A device-home view, a row or column index and a nonzero
`fit_output` are refused in words.

FITTED-STATE BLOCKS, in index order:
  0 `MEMBER_BLOCK_SCALARS` float64 [n_iter, loss, best_loss, t, output
    activation code (0 identity, 1 logistic, 4 softmax), outputs, classes,
    features];
  1 `MEMBER_BLOCK_FAMILY` float32 1 x P: the flat parameter block, w then b
    of each layer (coefs_[l] row-major, then intercepts_[l]);
  2 `MEMBER_BLOCK_FAMILY + 1` float64 1 x L: the layer widths (input,
    hidden..., output);
  3 `MEMBER_BLOCK_FAMILY + 2` float32 1 x n_iter: the loss curve.
An import needs blocks 0 .. 2; the curve is optional. An exported block is
read-only: the parameter block may be resident on the device (below).

RESIDENT WEIGHTS: every predict, predict_proba and score names the fitted
block (owner = the state's slot address, token = a new number at each fit or
import), so a GPU route uploads it on the first call only and keeps it
resident across calls (sequence/perf3_seq_resident.mojo), as the estimator's
own predict door keeps its packed block; a refit or an import names a new
block, and FREE posts the owner's death so the device copy is freed.

The fitted state is OWNED here, in a process-wide store of fixed chunks: a
slot never moves once minted, so a fit or predict runs on its slot without
the store's lock; the lock covers only mint, free and the handle lookup.
SERIAL BY NATURE: the store's bookkeeping and the per-layer driver loops.
The narrowing of the predicted int64 codes to int32 is split across the
host pool (`host_narrow_codes_i32`). The training and the forward pass are
the image's own parallel kernels.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from std.ffi import external_call
from core.device_route import DeviceRoute
from core.metrics_entry_table import MR_R2, MR_R2_WEIGHTED
from core.native_descriptor_call import native_descriptor_call
from core.member_abi import (
    MEMBER_GIL_DOOR,
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_PREDICT_PROBA, MEMBER_OP_SCORE, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_SCORE, MEMBER_CAN_WEIGHTED_SCORE,
    MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT, MEMBER_CAN_CLASSIFIER, MEMBER_CAN_MULTI_OUTPUT,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_MLP
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.host_parallel import host_narrow_codes_i32
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from sequence.perf3_seq_resident_words import seq_resident_owner_died
from sequence.xseq_native_common_run import xseq_stream_child_seed
from sequence.xseq_neural_run import (
    XseqNeuralEntry, XseqMlpFitRequest, XseqMlpPredictRequest, xseq_run_mlp_sizes, xseq_run_mlp_param_count,
    xseq_run_mlp_glorot, xseq_run_mlp_fit, xseq_run_mlp_predict, xseq_run_threshold, xseq_run_argmax,
    xseq_run_curve_min, xseq_run_mlp_hidden_check, xseq_run_mlp_scalars_check,
)
from sequence.xseq_score_run import xseq_run_vector_finite, xseq_run_weights_check

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _FIXED_WORDS = 21
comptime _SCALAR_WORDS = 8
comptime _BLOCK_PARAMS = MEMBER_BLOCK_FAMILY
comptime _BLOCK_SIZES = MEMBER_BLOCK_FAMILY + 1
comptime _BLOCK_CURVE = MEMBER_BLOCK_FAMILY + 2
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
# Resident-block tokens of member states sit above every token the class
# doors mint (`seq_resident_token`, a counter from 1), so a member's (owner,
# token) never names a door's block even if an owner word were reused.
comptime _TOKEN_BASE = 1 << 40


struct MlpMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var classifier: Bool
    var fitted: Bool
    # The NEW words (hyperparameters), kept whole.
    var words: List[Float64]
    var hidden: List[Int]
    # The fitted state.
    var sizes: List[Int]
    var params: List[Float32]
    var curve: List[Float32]
    var scalars: List[Float64]
    var widths: List[Float64]
    # The device-resident fitted block (sequence/perf3_seq_resident.mojo):
    # the owner word is this slot's address, `token` names the current
    # `params` (0: none yet). `stamp` counts every params change of this
    # slot address across generations (never reset), so a refit, an import
    # or a reuse of the slot names a new block. `named`: a predict named the
    # block, so FREE posts the owner's death to drop its device copy.
    var stamp: Int
    var token: Int
    var named: Bool

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.classifier = False
        self.fitted = False
        self.words = List[Float64]()
        self.hidden = List[Int]()
        self.sizes = List[Int]()
        self.params = List[Float32]()
        self.curve = List[Float32]()
        self.scalars = List[Float64]()
        self.widths = List[Float64]()
        self.stamp = 0
        self.token = 0
        self.named = False


struct MlpMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[MlpMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[MlpMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[MlpMemberStore, "MojolearnMlpMemberStoreV2", 2]

comptime _SlotP = MutPointer[MlpMemberSlot, MutUntrackedOrigin]
comptime _CallP = MutPointer[MemberCall, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock. A chunk's buffer is allocated once and never
    # grows, so the address stays valid after the lock is released.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[MlpMemberSlot]()


def _mint(var words: List[Float64], var hidden: List[Int]) raises -> Int:
    var store = _STORE.get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live MLP member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[MlpMemberSlot](length=_CHUNK, fill=MlpMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _SlotP(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].classifier = words[0] == 1.0
        slot[].fitted = False
        slot[].words = words^
        slot[].hidden = hidden^
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
        var slot = _SlotP(unsafe_from_address=address)
        if not slot[].live or slot[].generation != handle // _INDEX_SPAN:
            raise Error("NotFittedError: this member state was released or never created")
        if release:
            slot[].live = False
            slot[].fitted = False
            slot[].words = List[Float64]()
            slot[].hidden = List[Int]()
            slot[].sizes = List[Int]()
            slot[].params = List[Float32]()
            slot[].curve = List[Float32]()
            slot[].scalars = List[Float64]()
            slot[].widths = List[Float64]()
            slot[].token = 0
            if slot[].named:
                # The device store frees this owner's resident block at its
                # next call, before any lookup (non-raising).
                slot[].named = False
                seq_resident_owner_died(address)
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _name(classifier: Bool) -> String:
    return String("MLPClassifier") if classifier else String("MLPRegressor")


def _dense(view: MemberMatrix, dtype: Int, what: String, estimator: String) raises:
    if not view.dense_host(dtype):
        var kind = String("float32") if dtype == MEMBER_F32 else (
            String("int32") if dtype == MEMBER_I32 else String("float64"))
        raise Error("TypeError: an " + estimator + " member reads " + what + " as a dense host " + kind + " view")


def _whole(call: _CallP, estimator: String) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: an " + estimator + " member reads whole host views (no row or column index)")


def _model_entry(route: DeviceRoute) -> XseqNeuralEntry:
    return XseqNeuralEntry(route.entry.call, route.entry.word)


def _output_entry(route: DeviceRoute) -> XseqNeuralEntry:
    return XseqNeuralEntry(route.second.call, route.second.word)


def _entropy_seed() raises -> UInt64:
    """One 64-bit seed from eight bytes of OS entropy (`getentropy`, the
    source `os.urandom` reads), little-endian as the estimator's own
    `XseqStream` decodes `os.urandom(8)`. Thread-safe: no shared state."""
    var raw = List[UInt8](length=8, fill=0)
    var status = external_call["getentropy", Int32](raw.unsafe_ptr(), Int(8))
    if status != 0:
        raise Error("RuntimeError: OS entropy must return eight bytes")
    var seed = UInt64(0)
    for i in range(8):  # small-loop(one seed word from native entropy)
        seed |= UInt64(raw[i]) << UInt64(8 * i)
    return seed


def _name_params(slot: _SlotP):
    """A new resident-block token for the slot's current `params`."""
    slot[].stamp += 1
    slot[].token = _TOKEN_BASE + slot[].stamp


def _fit(slot: _SlotP, call: _CallP) raises:
    var estimator = _name(slot[].classifier)
    var route = call[].route
    _whole(call, estimator)
    if call[].fit_output != 0:
        member_unsupported(estimator, call[].fit_output)
    if call[].weights.rows > 0:
        raise Error("TypeError: an " + estimator + " member fits without sample weights")
    _dense(call[].x, MEMBER_F32, "X", estimator)
    var rows = call[].x.rows
    var d = call[].x.cols
    if call[].y.rows != rows:
        raise Error("ValueError: X and y have different numbers of rows")
    var w = _F64P(unsafe_from_address=Int(slot[].words.unsafe_ptr()))
    var o: Int
    var n_out: Int
    var out_act: Int
    var loss_code: Int
    if slot[].classifier:
        _dense(call[].y, MEMBER_I32, "y (class codes)", estimator)
        if call[].y.cols != 1:
            raise Error("NotImplementedError: multilabel targets are not implemented")
        var k = call[].classes
        if k < 2:
            raise Error("MLPClassifier needs at least two classes")
        if k == 2:
            o = 1
            n_out = 1
            out_act = 1
            loss_code = 1
        else:
            o = k
            n_out = k
            out_act = 4
            loss_code = 2
    else:
        _dense(call[].y, MEMBER_F32, "y", estimator)
        o = call[].y.cols
        n_out = 0
        out_act = 0
        loss_code = 0
    var sizes = xseq_run_mlp_sizes(d, slot[].hidden, o)
    var total = xseq_run_mlp_param_count(sizes)
    var max_iter = Int(w[15])
    slot[].fitted = False
    slot[].token = 0
    slot[].params = List[Float32](length=total, fill=0.0)
    slot[].curve = List[Float32](length=max_iter, fill=0.0)
    var base = Int(slot[].params.unsafe_ptr())
    # The estimator's seeded stream: Glorot-uniform per layer straight into
    # the flat block the fit trains, then one derived seed for the shuffle.
    var seed: UInt64
    if w[20] != 0.0:
        # random_state=None: a fresh seed at every fit, so every fold or
        # candidate of a caller starts from its own weights.
        seed = _entropy_seed()
    else:
        seed = UInt64(Int(w[17])) | (UInt64(Int(w[18])) << 32)
    var offset = xseq_run_mlp_glorot(_output_entry(route), base, sizes, w[1] == 1.0, seed, 0)
    var shuffle_seed = Int(xseq_stream_child_seed(seed, offset) % UInt64(2147483647))
    var batch = min(200, rows) if w[14] == 0.0 else max(1, min(Int(w[14]), rows))
    var request = XseqMlpFitRequest(rows, d, o, Int(w[1]), out_act, loss_code, Int(w[2]), Int(w[3]),
        w[12] != 0.0, batch, max_iter, w[13] != 0.0, shuffle_seed, Int(w[16]), slot[].hidden.copy(), n_out)
    var floats: List[Float64] = [w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11]]
    var n_iter = xseq_run_mlp_fit(_model_entry(route), request, floats, call[].x.data, call[].y.data, base,
        Int(slot[].curve.unsafe_ptr()))
    if n_iter == 0:
        raise Error("IndexError: list index out of range")
    var best = xseq_run_curve_min(_output_entry(route), Int(slot[].curve.unsafe_ptr()), n_iter)
    slot[].curve.resize(n_iter, 0.0)
    slot[].scalars = List[Float64](length=_SCALAR_WORDS, fill=0.0)
    slot[].scalars[0] = Float64(n_iter)
    slot[].scalars[1] = Float64(slot[].curve[n_iter - 1])
    slot[].scalars[2] = best
    slot[].scalars[3] = Float64(n_iter * rows)
    slot[].scalars[4] = Float64(out_act)
    slot[].scalars[5] = Float64(o)
    slot[].scalars[6] = Float64(call[].classes if slot[].classifier else 0)
    slot[].scalars[7] = Float64(d)
    slot[].widths = List[Float64](length=len(sizes), fill=0.0)
    for l in range(len(sizes)):  # driver-loop: over network layers
        slot[].widths[l] = Float64(sizes[l])
    slot[].sizes = sizes^
    _name_params(slot)
    slot[].fitted = True


def _forward(slot: _SlotP, call: _CallP, proba2: Bool, width: Int, dst: Int) raises:
    """MLP_PREDICT of x into `dst` (rows x width float32)."""
    var estimator = _name(slot[].classifier)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    _whole(call, estimator)
    _dense(call[].x, MEMBER_F32, "X", estimator)
    var features = Int(slot[].scalars[7])
    if call[].x.cols != features:
        raise Error("X has " + String(call[].x.cols) + " features, the model was fitted on " + String(features))
    var hidden = List[Int]()
    for l in range(1, len(slot[].sizes) - 1):  # driver-loop: over hidden layer widths
        hidden.append(slot[].sizes[l])
    var w = _F64P(unsafe_from_address=Int(slot[].words.unsafe_ptr()))
    # The fitted block is named (owner = this slot, token = its params): a
    # GPU route keeps it resident across calls, uploading it on the first
    # predict only, as the estimator's own predict door does; a host
    # executor ignores the three words.
    slot[].named = True
    var request = XseqMlpPredictRequest(call[].x.rows, features, Int(slot[].scalars[5]), Int(w[1]),
        Int(slot[].scalars[4]), hidden^, proba2, width, len(slot[].params), Int(slot), slot[].token)
    xseq_run_mlp_predict(_model_entry(call[].route), request, call[].x.data, Int(slot[].params.unsafe_ptr()), dst)


def _predict(slot: _SlotP, call: _CallP) raises:
    var estimator = _name(slot[].classifier)
    var rows = call[].x.rows
    var outputs = Int(slot[].scalars[5]) if slot[].fitted else 0
    if not slot[].classifier:
        _dense(call[].dst, MEMBER_F32, "the prediction buffer", estimator)
        if call[].dst.rows != rows or call[].dst.cols != outputs:
            raise Error("ValueError: the prediction buffer holds one row of outputs per row of X")
        _forward(slot, call, False, outputs, call[].dst.data)
        return
    _dense(call[].dst, MEMBER_I32, "the class-code buffer", estimator)
    if call[].dst.rows != rows or call[].dst.cols != 1:
        raise Error("ValueError: the class-code buffer holds one code per row of X")
    var scores = List[Float32](length=rows * max(outputs, 1), fill=0.0)
    _forward(slot, call, False, outputs, Int(scores.unsafe_ptr()))
    var codes = List[Int64](length=rows, fill=0)
    if outputs == 1:
        # p > 0.5 per row (NaN takes class 0), as the estimator's own predict
        xseq_run_threshold(_output_entry(call[].route), Int(scores.unsafe_ptr()), Int(codes.unsafe_ptr()), rows)
    else:
        xseq_run_argmax(_output_entry(call[].route), Int(scores.unsafe_ptr()), Int(codes.unsafe_ptr()), rows,
            outputs)
    # Narrowed across the host pool (lane fx-audit, audit-neural finding 1).
    host_narrow_codes_i32(Int(codes.unsafe_ptr()), call[].dst.data, rows)
    _ = scores^
    _ = codes^


def _proba(slot: _SlotP, call: _CallP) raises:
    var estimator = _name(slot[].classifier)
    if not slot[].classifier:
        member_unsupported(estimator, MEMBER_OP_PREDICT_PROBA)
    var classes = Int(slot[].scalars[6]) if slot[].fitted else 0
    _dense(call[].dst, MEMBER_F32, "the probability buffer", estimator)
    if call[].dst.rows != call[].x.rows or call[].dst.cols != classes:
        raise Error("ValueError: the probability buffer holds one row of class probabilities per row of X")
    # A one-output net writes the two-column probability [1 - p, p].
    var outputs = Int(slot[].scalars[5])
    _forward(slot, call, outputs == 1, classes, call[].dst.data)


def _score(slot: _SlotP, call: _CallP) raises:
    var estimator = _name(slot[].classifier)
    if slot[].classifier:
        raise Error("NotImplementedError: mojolearn: the MLPClassifier member publishes no score; the caller "
            + "predicts and runs its own accuracy")
    var route = call[].route
    if not route.has_third:
        raise Error("ImportError: mojolearn: the MLPRegressor member's score needs the R^2 entry on its route")
    _dense(call[].y, MEMBER_F32, "y", estimator)
    var rows = call[].x.rows
    var outputs = Int(slot[].scalars[5]) if slot[].fitted else 0
    if call[].y.cols != 1 or outputs != 1:
        raise Error("NotImplementedError: mojolearn: the MLPRegressor member scores one output column; the caller "
            + "predicts and runs its own multi-output R^2")
    if call[].y.rows != rows:
        raise Error("ValueError: mojolearn metrics: y_true has " + String(call[].y.rows) + " entries and y_pred has "
            + String(rows) + "; they must be the same length")
    var prediction = List[Float32](length=rows, fill=0.0)
    _forward(slot, call, False, 1, Int(prediction.unsafe_ptr()))
    xseq_run_vector_finite(call[].y.data, rows, "y_true")
    xseq_run_vector_finite(Int(prediction.unsafe_ptr()), rows, "y_pred")
    var value = List[Float64](length=1, fill=0.0)
    var words: List[Int] = [rows]
    if call[].weights.rows > 0:
        _dense(call[].weights, MEMBER_F32, "sample_weight", estimator)
        var shape: List[Int] = [call[].weights.rows * call[].weights.cols]
        xseq_run_weights_check(shape, rows, call[].weights.data)
        var addresses: List[Int] = [call[].y.data, Int(prediction.unsafe_ptr()), call[].weights.data,
            Int(value.unsafe_ptr())]
        _ = native_descriptor_call(route.third.call, MR_R2_WEIGHTED, addresses, words, List[Float64]())
    else:
        var addresses: List[Int] = [call[].y.data, Int(prediction.unsafe_ptr()), Int(value.unsafe_ptr())]
        _ = native_descriptor_call(route.third.call, MR_R2, addresses, words, List[Float64]())
    call[].value = value[0]
    _ = prediction^
    _ = value^


def _export(slot: _SlotP, call: _CallP) raises:
    """A view of one fitted block. The slot owns the words; they stay valid
    until this state's next fit, import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot[].classifier) + " member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > 3:
            raise Error("member ABI: an MLP member has four fitted blocks")
        block = MEMBER_BLOCK_SCALARS if block == 0 else (
            _BLOCK_PARAMS if block == 1 else (_BLOCK_SIZES if block == 2 else _BLOCK_CURVE))
    if block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block == _BLOCK_PARAMS:
        call[].dst = MemberMatrix(Int(slot[].params.unsafe_ptr()), 1, len(slot[].params))
    elif block == _BLOCK_SIZES:
        call[].dst = MemberMatrix(Int(slot[].widths.unsafe_ptr()), 1, len(slot[].widths), MEMBER_F64)
    elif block == _BLOCK_CURVE:
        call[].dst = MemberMatrix(Int(slot[].curve.unsafe_ptr()), 1, len(slot[].curve))
    else:
        raise Error("member ABI: an MLP member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 4


def _import(slot: _SlotP, call: _CallP) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var estimator = _name(slot[].classifier)
    var block = call[].block
    if block == _BLOCK_PARAMS or block == _BLOCK_CURVE:
        _dense(call[].x, MEMBER_F32, "a fitted block", estimator)
        var count = call[].x.rows * call[].x.cols
        var words = List[Float32](length=count, fill=0.0)
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        slot[].fitted = False
        if block == _BLOCK_PARAMS:
            slot[].token = 0
            slot[].params = words^
        else:
            slot[].curve = words^
    elif block == MEMBER_BLOCK_SCALARS or block == _BLOCK_SIZES:
        _dense(call[].x, MEMBER_F64, "a fitted block", estimator)
        var count = call[].x.rows * call[].x.cols
        if block == MEMBER_BLOCK_SCALARS and count != _SCALAR_WORDS:
            raise Error("member ABI: the MLP scalars block is eight host float64 words")
        var words = List[Float64](length=count, fill=0.0)
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F64P(unsafe_from_address=call[].x.data), count=count)
        slot[].fitted = False
        if block == MEMBER_BLOCK_SCALARS:
            slot[].scalars = words^
        else:
            slot[].widths = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: an MLP member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) != _SCALAR_WORDS or len(slot[].widths) < 2 or len(slot[].params) == 0:
        raise Error("member ABI: a fitted MLP state needs its scalars, parameter and layer-width blocks")
    var sizes = List[Int]()
    for l in range(len(slot[].widths)):  # driver-loop: over network layers
        var width = Int(slot[].widths[l])
        if width < 1:
            raise Error("ValueError: MLP fitted parameter shapes do not match the layer layout")
        sizes.append(width)
    if (xseq_run_mlp_param_count(sizes) != len(slot[].params) or sizes[0] != Int(slot[].scalars[7])
            or sizes[len(sizes) - 1] != Int(slot[].scalars[5])):
        raise Error("ValueError: MLP fitted parameter shapes do not match the layer layout")
    if slot[].classifier:
        if call[].classes != 0:
            slot[].scalars[6] = Float64(call[].classes)
        if Int(slot[].scalars[6]) < 2:
            raise Error("member ABI: a fitted MLPClassifier state needs its class count")
    slot[].sizes = sizes^
    _name_params(slot)
    slot[].fitted = True


def _outputs(call: _CallP) raises:
    """OUTPUTS with or without a state (item 10): the classifier's predict
    one int32 code column, predict_proba float32 x classes; the regressor's
    predict float32 x targets (the fitted net's outputs, else the caller's
    `y.cols`). Another operation is refused, never sized."""
    var asked = call[].flags
    var classifier: Bool
    var outputs = max(1, call[].y.cols)
    var classes = call[].classes
    if call[].state != 0:
        var slot = _SlotP(unsafe_from_address=_lookup(call[].state, False))
        classifier = slot[].classifier
        if slot[].fitted:
            outputs = Int(slot[].scalars[5])
            classes = Int(slot[].scalars[6])
    elif call[].params and call[].param_count >= 1:
        classifier = _F64P(unsafe_from_address=call[].params)[0] == 1.0
    else:
        raise Error("member ABI: an MLP member's OUTPUTS without a state needs its NEW words")
    call[].classes = classes if classifier else 0
    if classifier:
        if asked == MEMBER_OP_PREDICT:
            call[].out_cols = 1
            call[].out_dtype = MEMBER_I32
        elif asked == MEMBER_OP_PREDICT_PROBA:
            call[].out_cols = classes
            call[].out_dtype = MEMBER_F32
        else:
            member_unsupported(_name(True), asked)
        return
    if asked != MEMBER_OP_PREDICT:
        member_unsupported(_name(False), asked)
    call[].out_cols = outputs
    call[].out_dtype = MEMBER_F32


def _capabilities(classifier: Bool) -> Int:
    # MEMBER_GIL_DOOR (THE GIL RULE, core/member_abi.mojo; fx-gil-RESULT.md,
    # round 4): still one member operation at a time on the public
    # thread; the caller releases the GIL around it, as the family's own
    # door does around the same RUN functions (bindings/xseq_neural_native_door.mojo:
    # `xseq_run_mlp_glorot`, `xseq_run_mlp_fit`, `xseq_run_mlp_predict`, `xseq_run_threshold`,
    # `xseq_run_argmax`, `xseq_run_curve_min`; the R^2 entry is host arithmetic). The helpers
    # the door runs with the GIL held are
    # shape and value checks on the call's own words (no process state),
    # and the store has its own lock. NOT `MEMBER_GIL_FREE`: the images
    # behind the route are not traced for parallel workers.
    var common = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_GIL_DOOR
    if classifier:
        return common | MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_CLASSIFIER
    return common | MEMBER_CAN_SCORE | MEMBER_CAN_WEIGHTED_SCORE | MEMBER_CAN_MULTI_OUTPUT


def _new(call: _CallP) raises:
    if call[].param_count < _FIXED_WORDS or not call[].params:
        raise Error("member ABI: an MLP member takes at least twenty-one parameters")
    var w = _F64P(unsafe_from_address=call[].params)
    var layers = Int(w[19])
    if layers < 0 or call[].param_count != _FIXED_WORDS + layers:
        raise Error("member ABI: an MLP member takes 21 + (hidden layer count) parameters")
    if w[0] != 0.0 and w[0] != 1.0:
        raise Error("member ABI: unknown MLP member kind")
    if w[1] < 0.0 or w[1] > 3.0 or w[2] < 0.0 or w[2] > 1.0 or w[3] < 0.0 or w[3] > 2.0:
        raise Error("member ABI: an MLP member's activation, solver or learning-rate code is out of range")
    if w[20] != 0.0 and w[20] != 1.0:
        raise Error("member ABI: an MLP member's fresh-seed word is 0 or 1")
    if w[14] < 0.0 or w[16] < 1.0:
        raise Error("member ABI: an MLP member's batch_size or n_iter_no_change word is out of range")
    xseq_run_mlp_scalars_check(w[4], w[15], w[10])
    var words = List[Float64](length=call[].param_count, fill=0.0)
    unsafe_memcpy(dest=words.unsafe_ptr(), src=w, count=call[].param_count)
    var hidden = List[Int]()
    for l in range(layers):  # driver-loop: over hidden layer size words
        hidden.append(Int(w[_FIXED_WORDS + l]))
    xseq_run_mlp_hidden_check(hidden)
    call[].state = _mint(words^, hidden^)


def _mlp_member(operation: Int, address: Int) raises -> Int:
    var call = _CallP(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        if call[].state == 0:
            # Kind-exact from the NEW words when given (item 17).
            if call[].params and call[].param_count >= 1:
                return _capabilities(_F64P(unsafe_from_address=call[].params)[0] == 1.0)
            return _capabilities(False) | _capabilities(True)
        var probe = _SlotP(unsafe_from_address=_lookup(call[].state, False))
        return _capabilities(probe[].classifier)
    if operation == MEMBER_OP_NEW:
        _new(call)
        return 0
    if operation == MEMBER_OP_FREE:
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        _outputs(call)
        return 0
    var slot = _SlotP(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_PREDICT:
        _predict(slot, call)
    elif operation == MEMBER_OP_PREDICT_PROBA:
        _proba(slot, call)
    elif operation == MEMBER_OP_SCORE:
        _score(slot, call)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_name(slot[].classifier), operation)
    return 0


def mlp_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of MLPRegressor and MLPClassifier (schema `MEMBER_ABI`)."""
    try:
        var word = _mlp_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def mlp_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's MLP member entry. One line in each
    `_mojolearn_x_sequence` image's module initialization, after its glue
    entry is published. A build with no GPU vendor (`vendor` < 0) publishes
    nothing."""
    if vendor < 0:
        return
    member_publish[MEMBER_FAMILY_MLP](mode, vendor, publisher, mlp_member_v2)

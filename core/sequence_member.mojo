# SPDX-License-Identifier: Apache-2.0
"""The recurrent models (LSTM, GRU and RNN regressors and classifiers) as
MEMBERS (core/member_abi.mojo).

One entry, `sequence_member_v2`, published by each `_mojolearn_x_sequence`
image (`sequence_member_publish`, beside the MLP member). It runs the
family's existing typed RUN (sequence/xseq_neural_run.mojo: the uniform
draws, RNN_FIT, RNN_PREDICT, the row argmax) over the entries of the
`DeviceRoute` it is handed, the same entries the estimator's own door
(bindings/xseq_neural_native_door.mojo `xseq_rnn`) calls. No kernel is
touched.

ROUTE SLOTS (filled by providers/python/member_sequence_boundary_door.mojo):
  * `entry`: the model descriptor entry of the route's image (RNN_FIT,
    RNN_PREDICT); `word` = the address of its provider's retained call
    workspace.
  * `second`: the output-helper entry (the initial uniform draws, the row
    argmax); `word` = its workspace address.
  * `third` (`has_third`): the R^2 entry of the metrics image of the
    route's device (regressor score). Without it a score is refused in words.

MEMBER_OP_NEW parameters (Float64 words, 30 of them):
  [0] kind = task (0 regressor, 1 classifier); [1] cell (0 RNN tanh, 1 RNN
  relu, 2 LSTM, 3 GRU); [2] hidden_size; [3] num_layers; [4] batch_size;
  [5] max_epochs; [6] shuffle (0 / 1); [7] learning_rate (the constant rate
  of every step without an `lr_schedule`); [8]
  predict_chunk; [9] optimizer kind; [10] optimizer flag word; [11 .. 16]
  the optimizer's six float slots (the estimator's own `_rnn_optimizer`
  words, PyTorch's defaults merged with `optimizer_options`); [17] seed,
  low 32 bits; [18] seed, high 32 bits (the estimator's seeded stream of an
  integer `random_state`); [19] fresh seed (1: `random_state` is None, so
  every FIT draws eight bytes of OS entropy, as the estimator's own fit);
  [20] 1: a built-in `lr_schedule` (lane mf-ens6); [21] its class index
  (0 ConstantLR .. 5 OneCycleLR, `schedule_kind`); [22] LrSpec.kind; [23 ..
  25] LrSpec.a, b, c; [26 .. 29] LrSpec.f0 .. f3 (the schedule's admitted
  spec, `schedule_spec_admitted`). With [20] = 1 every FIT fills the rate of
  each of its steps with the schedule's own native series
  (`optimizer_schedule_evaluation(...).fill`, the body `schedule_run`'s
  schedule_values runs for the estimator's own fit), so the fold's step
  count picks the series as the estimator's fit on those rows does.

VIEWS: X dense row-major host float32, rows x (T * F) with `x.steps` = T
>= 1 (MEMBER_CAN_STEPS; `steps` 0, a plain matrix, is refused: the models
read 3-D X). Regressor y host float32 rows x outputs; classifier y host
int32 codes 0 .. classes-1 (rows x 1), `call.classes` >= 2. Predict writes
float32 rows x outputs (regressor) or int32 codes (classifier);
predict_proba float32 rows x classes. Score: the regressor's R^2 of one
output column, weighted when `call.weights` (host float32) is given. No
sample weights at fit; a device-home view, a row or column index and a
nonzero `fit_output` are refused in words. X and a regressor's y are
borrowed; a classifier's codes become the float32 target the recurrent loss
reads (one word per row, as the estimator's own fit converts them).

FITTED-STATE BLOCKS, in index order:
  0 `MEMBER_BLOCK_SCALARS` float64 [n_iter, n_timesteps, features, outputs,
    classes];
  1 `MEMBER_BLOCK_FAMILY` float32 1 x P: `params_` (layout order);
  2 `MEMBER_BLOCK_FAMILY + 1` float32 1 x n_iter: the minibatch loss curve.
An import needs blocks 0 and 1; the curve is optional.

RESIDENT WEIGHTS: every predict names the fitted block (owner = the
state's slot address, token = a new number at each fit or import), so a GPU
route uploads it once and keeps it resident (sequence/perf3_seq_resident.mojo),
as the estimator's own predict record does; FREE posts the owner's death.

The fitted state is OWNED here, in a process-wide store of fixed chunks: a
slot never moves once minted; the store's lock covers only mint, free and
the handle lookup. SERIAL BY NATURE: the store's bookkeeping. The
narrowing of int64 codes to int32 is split across the host pool
(`host_narrow_codes_i32`). The draws, the training and the forward are the
image's own parallel kernels.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from core.metrics_entry_table import MR_R2, MR_R2_WEIGHTED
from core.native_descriptor_call import native_descriptor_call
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_PREDICT_PROBA, MEMBER_OP_SCORE, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_SCORE, MEMBER_CAN_WEIGHTED_SCORE,
    MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT, MEMBER_CAN_CLASSIFIER, MEMBER_CAN_MULTI_OUTPUT, MEMBER_CAN_STEPS,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_SEQUENCE
from core.host_parallel import host_narrow_codes_i32
from core.mlp_member import _entropy_seed
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from core.xseq_marshal import xseq_codes_f32
from sequence.perf3_seq_resident_words import seq_resident_owner_died
from sequence.xseq_native_common_run import xseq_stream_child_seed
from sequence.xseq_neural_run import (
    XseqNeuralEntry, XseqOptimizer, XseqRnnFitRequest, XseqRnnPredictRequest, xseq_run_uniform, xseq_run_argmax,
    xseq_run_rnn_gates, xseq_run_rnn_dims_check, xseq_run_rnn_param_count, xseq_run_rnn_bound, xseq_run_rnn_schedule,
    xseq_run_rnn_fit, xseq_run_rnn_layout, xseq_run_rnn_predict,
)
from sequence.xseq_score_run import xseq_run_vector_finite, xseq_run_weights_check
from sequence.lr_table import LrSpec
from training.optimizer_schedule_plan import optimizer_schedule_evaluation

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _WORDS = 30
comptime _SCALAR_WORDS = 5
comptime _BLOCK_PARAMS = MEMBER_BLOCK_FAMILY
comptime _BLOCK_CURVE = MEMBER_BLOCK_FAMILY + 1
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
# Resident-block tokens above every token the class doors mint (as
# core/mlp_member.mojo; a member's (owner, token) never names a door's block).
comptime _TOKEN_BASE = 1 << 41


struct SequenceMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var classifier: Bool
    var fitted: Bool
    var words: List[Float64]
    var params: List[Float32]
    var curve: List[Float32]
    var scalars: List[Float64]
    # The device-resident fitted block: owner word = this slot's address,
    # `token` names the current `params`; `stamp` never resets; `named`: a
    # predict named the block, so FREE posts the owner's death.
    var stamp: Int
    var token: Int
    var named: Bool

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.classifier = False
        self.fitted = False
        self.words = List[Float64]()
        self.params = List[Float32]()
        self.curve = List[Float32]()
        self.scalars = List[Float64]()
        self.stamp = 0
        self.token = 0
        self.named = False


struct SequenceMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[SequenceMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[SequenceMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[SequenceMemberStore, "MojolearnSequenceMemberStoreV2", 2]

comptime _SlotP = MutPointer[SequenceMemberSlot, MutUntrackedOrigin]
comptime _CallP = MutPointer[MemberCall, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock; a chunk never grows, so the address stays valid.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[SequenceMemberSlot]()


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
                raise Error("RuntimeError: mojolearn: too many live recurrent member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[SequenceMemberSlot](length=_CHUNK, fill=SequenceMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _SlotP(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].classifier = words[0] == 1.0
        slot[].fitted = False
        slot[].words = words^
        return slot[].generation * _INDEX_SPAN + index + 1
    finally:
        shared_slot_unlock(lock)


def _lookup(handle: Int, release: Bool) raises -> Int:
    """The slot address of a live handle; a stale or foreign handle is
    refused, never read."""
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
            slot[].params = List[Float32]()
            slot[].curve = List[Float32]()
            slot[].scalars = List[Float64]()
            slot[].token = 0
            if slot[].named:
                slot[].named = False
                seq_resident_owner_died(address)
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _name(slot: _SlotP) -> String:
    var cell = Int(slot[].words[1])
    var base = String("RNN") if cell < 2 else (String("LSTM") if cell == 2 else String("GRU"))
    return base + (String("Classifier") if slot[].classifier else String("Regressor"))


def _dense(view: MemberMatrix, dtype: Int, what: String, estimator: String) raises:
    if not view.dense_host(dtype):
        var kind = String("float32") if dtype == MEMBER_F32 else (
            String("int32") if dtype == MEMBER_I32 else String("float64"))
        raise Error("TypeError: an " + estimator + " member reads " + what + " as a dense host " + kind + " view")


def _whole(call: _CallP, estimator: String) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: an " + estimator + " member reads whole host views (no row or column index)")


def _sequences(view: MemberMatrix, estimator: String) raises -> Int:
    """X as (rows, T, F): returns T."""
    _dense(view, MEMBER_F32, "X", estimator)
    if view.steps < 1 or view.cols % view.steps != 0:
        raise Error("X must be 3-D: (n_samples, n_timesteps, n_features)")
    return view.steps


def _model_entry(call: _CallP) -> XseqNeuralEntry:
    return XseqNeuralEntry(call[].route.entry.call, call[].route.entry.word)


def _output_entry(call: _CallP) -> XseqNeuralEntry:
    return XseqNeuralEntry(call[].route.second.call, call[].route.second.word)


def _ints(slot: _SlotP) -> List[Int]:
    """`_RecurrentBase._ints()`: cell, D, H, L, O."""
    var out: List[Int] = [Int(slot[].words[1]), Int(slot[].scalars[2]), Int(slot[].words[2]),
        Int(slot[].words[3]), Int(slot[].scalars[3])]
    return out^


def _fit(slot: _SlotP, call: _CallP) raises:
    var estimator = _name(slot)
    _whole(call, estimator)
    if call[].fit_output != 0:
        member_unsupported(estimator, call[].fit_output)
    if call[].weights.rows > 0:
        raise Error("TypeError: an " + estimator + " member fits without sample weights")
    var steps_t = _sequences(call[].x, estimator)
    var n = call[].x.rows
    var d = call[].x.cols // steps_t
    xseq_run_rnn_dims_check(n, steps_t)
    if call[].y.rows != n:
        raise Error("ValueError: X and y have different numbers of rows")
    var w = _F64P(unsafe_from_address=Int(slot[].words.unsafe_ptr()))
    var outputs: Int
    var classes = 0
    var target = call[].y.data
    var codes = List[Float32]()
    if slot[].classifier:
        _dense(call[].y, MEMBER_I32, "y (class codes)", estimator)
        if call[].y.cols != 1:
            raise Error("NotImplementedError: multilabel targets are not implemented")
        classes = call[].classes
        if classes < 2:
            raise Error("classification needs at least two classes")
        outputs = classes
        # The int32 codes as the float32 target the recurrent loss reads.
        codes = List[Float32](length=n, fill=0.0)
        xseq_codes_f32(_I32P(unsafe_from_address=call[].y.data), _F32P(unsafe_from_address=Int(codes.unsafe_ptr())), n)
        target = Int(codes.unsafe_ptr())
    else:
        _dense(call[].y, MEMBER_F32, "y", estimator)
        outputs = call[].y.cols
    var cell = Int(w[1])
    var hidden = Int(w[2])
    var layers = Int(w[3])
    var total = xseq_run_rnn_param_count(xseq_run_rnn_gates(cell), d, hidden, layers, outputs)
    var schedule = xseq_run_rnn_schedule(Int(w[4]), n, Int(w[5]))
    slot[].fitted = False
    slot[].token = 0
    slot[].params = List[Float32](length=total, fill=0.0)
    slot[].curve = List[Float32](length=schedule.steps, fill=0.0)
    var seed: UInt64
    if w[19] != 0.0:
        seed = _entropy_seed()  # random_state=None: its own stream at every fit
    else:
        seed = UInt64(Int(w[17])) | (UInt64(Int(w[18])) << 32)
    # The parameters in layout order, U(-1/sqrt(H), 1/sqrt(H)), one block;
    # then the derived training seed (`XseqStream.child_seed`).
    var bound = xseq_run_rnn_bound(Float64(hidden))
    var offset = xseq_run_uniform(_output_entry(call), Int(slot[].params.unsafe_ptr()), total, seed, 0, -bound, bound)
    var child = xseq_stream_child_seed(seed, offset)
    var rates = List[Float32](length=schedule.steps, fill=Float32(w[7]))
    if w[20] != 0.0:
        # The schedule's series over this fit's steps (`_rnn_rates`).
        var spec = LrSpec(Int(w[22]), Int(w[23]), Int(w[24]), Int(w[25]), w[26], w[27], w[28], w[29])
        optimizer_schedule_evaluation(Int(w[21]), spec, schedule.steps, True).fill(
            _F32P(unsafe_from_address=Int(rates.unsafe_ptr())))
    var optimizer = XseqOptimizer(Int(w[9]), Int(w[10]), [w[11], w[12], w[13], w[14], w[15], w[16]])
    var request = XseqRnnFitRequest(cell, d, hidden, layers, outputs, 1 if slot[].classifier else 0, n, steps_t,
        Int(w[5]), schedule.batch, w[6] != 0.0, child)
    xseq_run_rnn_fit(_model_entry(call), request, optimizer, call[].x.data, target, Int(slot[].params.unsafe_ptr()),
        Int(slot[].curve.unsafe_ptr()), Int(rates.unsafe_ptr()))
    _ = codes^
    _ = rates^
    slot[].scalars = [Float64(schedule.steps), Float64(steps_t), Float64(d), Float64(outputs), Float64(classes)]
    slot[].stamp += 1
    slot[].token = _TOKEN_BASE + slot[].stamp
    slot[].fitted = True


def _forward(slot: _SlotP, call: _CallP, dst: Int) raises:
    """RNN_PREDICT of x into `dst` (rows x outputs float32)."""
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    _whole(call, estimator)
    var steps_t = _sequences(call[].x, estimator)
    var features = Int(slot[].scalars[2])
    if call[].x.cols // steps_t != features:
        raise Error("X has " + String(call[].x.cols // steps_t) + " features, the model was fitted on "
            + String(features))
    var ints = _ints(slot)
    var n_params = xseq_run_rnn_layout(ints, len(slot[].params))
    slot[].named = True
    var seq = List[Float32](length=1, fill=0.0)
    var request = XseqRnnPredictRequest(ints^, n_params, 1 if slot[].classifier else 0, call[].x.rows, steps_t,
        Int(slot[].words[8]), False, Int(slot), slot[].token)
    xseq_run_rnn_predict(_model_entry(call), request, call[].x.data, Int(slot[].params.unsafe_ptr()), dst,
        Int(seq.unsafe_ptr()))
    _ = seq^


def _predict(slot: _SlotP, call: _CallP) raises:
    var estimator = _name(slot)
    var rows = call[].x.rows
    var outputs = Int(slot[].scalars[3]) if slot[].fitted else 0
    if not slot[].classifier:
        _dense(call[].dst, MEMBER_F32, "the prediction buffer", estimator)
        if call[].dst.rows != rows or call[].dst.cols != outputs:
            raise Error("ValueError: the prediction buffer holds one row of outputs per row of X")
        _forward(slot, call, call[].dst.data)
        return
    _dense(call[].dst, MEMBER_I32, "the class-code buffer", estimator)
    if call[].dst.rows != rows or call[].dst.cols != 1:
        raise Error("ValueError: the class-code buffer holds one code per row of X")
    var scores = List[Float32](length=rows * max(outputs, 1), fill=0.0)
    _forward(slot, call, Int(scores.unsafe_ptr()))
    var codes = List[Int64](length=rows, fill=0)
    xseq_run_argmax(_output_entry(call), Int(scores.unsafe_ptr()), Int(codes.unsafe_ptr()), rows, outputs)
    # Narrowed across the host pool (lane fx-audit, audit-neural finding 1).
    host_narrow_codes_i32(Int(codes.unsafe_ptr()), call[].dst.data, rows)
    _ = scores^
    _ = codes^


def _proba(slot: _SlotP, call: _CallP) raises:
    var estimator = _name(slot)
    if not slot[].classifier:
        member_unsupported(estimator, MEMBER_OP_PREDICT_PROBA)
    var classes = Int(slot[].scalars[4]) if slot[].fitted else 0
    _dense(call[].dst, MEMBER_F32, "the probability buffer", estimator)
    if call[].dst.rows != call[].x.rows or call[].dst.cols != classes:
        raise Error("ValueError: the probability buffer holds one row of class probabilities per row of X")
    _forward(slot, call, call[].dst.data)


def _score(slot: _SlotP, call: _CallP) raises:
    var estimator = _name(slot)
    if slot[].classifier:
        raise Error("NotImplementedError: mojolearn: the " + estimator + " member publishes no score; the caller "
            + "predicts and runs its own accuracy")
    var route = call[].route
    if not route.has_third:
        raise Error("ImportError: mojolearn: the " + estimator + " member's score needs the R^2 entry on its route")
    _dense(call[].y, MEMBER_F32, "y", estimator)
    var rows = call[].x.rows
    var outputs = Int(slot[].scalars[3]) if slot[].fitted else 0
    if call[].y.cols != 1 or outputs != 1:
        raise Error("NotImplementedError: mojolearn: the " + estimator + " member scores one output column; the "
            + "caller predicts and runs its own multi-output R^2")
    if call[].y.rows != rows:
        raise Error("ValueError: mojolearn metrics: y_true has " + String(call[].y.rows) + " entries and y_pred has "
            + String(rows) + "; they must be the same length")
    var prediction = List[Float32](length=rows, fill=0.0)
    _forward(slot, call, Int(prediction.unsafe_ptr()))
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
    """A view of one fitted block, valid until this state's next fit,
    import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot) + " member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block > 2:
            raise Error("member ABI: a recurrent member has three fitted blocks")
        block = MEMBER_BLOCK_SCALARS if block == 0 else (_BLOCK_PARAMS if block == 1 else _BLOCK_CURVE)
    if block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block == _BLOCK_PARAMS:
        call[].dst = MemberMatrix(Int(slot[].params.unsafe_ptr()), 1, len(slot[].params))
    elif block == _BLOCK_CURVE:
        call[].dst = MemberMatrix(Int(slot[].curve.unsafe_ptr()), 1, len(slot[].curve))
    else:
        raise Error("member ABI: a recurrent member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 3


def _import(slot: _SlotP, call: _CallP) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var estimator = _name(slot)
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
    elif block == MEMBER_BLOCK_SCALARS:
        _dense(call[].x, MEMBER_F64, "a fitted block", estimator)
        if call[].x.rows * call[].x.cols != _SCALAR_WORDS:
            raise Error("member ABI: the recurrent scalars block is five host float64 words")
        var words = List[Float64](length=_SCALAR_WORDS, fill=0.0)
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F64P(unsafe_from_address=call[].x.data), count=_SCALAR_WORDS)
        slot[].fitted = False
        slot[].scalars = words^
    elif block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a recurrent member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].scalars) != _SCALAR_WORDS or len(slot[].params) == 0:
        raise Error("member ABI: a fitted recurrent state needs its scalars and parameter blocks")
    if slot[].classifier:
        if call[].classes != 0:
            slot[].scalars[4] = Float64(call[].classes)
        if Int(slot[].scalars[4]) < 2:
            raise Error("member ABI: a fitted recurrent classifier state needs its class count")
    _ = xseq_run_rnn_layout(_ints(slot), len(slot[].params))
    slot[].stamp += 1
    slot[].token = _TOKEN_BASE + slot[].stamp
    slot[].fitted = True


def _outputs(call: _CallP) raises:
    """OUTPUTS with or without a state (item 10): the classifier's predict
    one int32 code column, predict_proba float32 x classes; the regressor's
    predict float32 x outputs (the fitted state's, else the caller's
    `y.cols`) and score one value. Another operation is refused."""
    var asked = call[].flags
    var classifier: Bool
    var outputs = max(1, call[].y.cols)
    var classes = call[].classes
    if call[].state != 0:
        var slot = _SlotP(unsafe_from_address=_lookup(call[].state, False))
        classifier = slot[].classifier
        if slot[].fitted:
            outputs = Int(slot[].scalars[3])
            classes = Int(slot[].scalars[4])
    elif call[].param_count == _WORDS and call[].params:
        classifier = _F64P(unsafe_from_address=call[].params)[0] == 1.0
    else:
        raise Error("member ABI: a recurrent member's OUTPUTS without a state needs its NEW words")
    call[].classes = classes if classifier else 0
    if classifier:
        if asked == MEMBER_OP_PREDICT:
            call[].out_cols = 1
            call[].out_dtype = MEMBER_I32
        elif asked == MEMBER_OP_PREDICT_PROBA:
            call[].out_cols = classes
            call[].out_dtype = MEMBER_F32
        else:
            member_unsupported("a recurrent classifier member", asked)
        return
    if asked == MEMBER_OP_SCORE:
        call[].out_cols = 1
    elif asked == MEMBER_OP_PREDICT:
        call[].out_cols = outputs
        call[].out_dtype = MEMBER_F32
    else:
        member_unsupported("a recurrent regressor member", asked)


def _capabilities(classifier: Bool) -> Int:
    var common = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_CAN_STEPS
    if classifier:
        return common | MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_CLASSIFIER
    return common | MEMBER_CAN_SCORE | MEMBER_CAN_WEIGHTED_SCORE | MEMBER_CAN_MULTI_OUTPUT


def _new(call: _CallP) raises:
    if call[].param_count != _WORDS or not call[].params:
        raise Error("member ABI: a recurrent member takes thirty parameters")
    var w = _F64P(unsafe_from_address=call[].params)
    if w[0] != 0.0 and w[0] != 1.0:
        raise Error("member ABI: unknown recurrent member kind")
    if w[1] < 0.0 or w[1] > 3.0 or w[2] < 1.0 or w[3] < 1.0 or w[4] < 1.0 or w[5] < 0.0 or w[8] < 1.0:
        raise Error("member ABI: a recurrent member's cell, size, epoch or chunk word is out of range")
    if w[19] != 0.0 and w[19] != 1.0:
        raise Error("member ABI: a recurrent member's fresh-seed word is 0 or 1")
    if (w[20] != 0.0 and w[20] != 1.0) or (w[20] == 1.0 and (w[21] < 0.0 or w[21] > 5.0)):
        raise Error("member ABI: a recurrent member's schedule words are out of range")
    var words = List[Float64](length=_WORDS, fill=0.0)
    unsafe_memcpy(dest=words.unsafe_ptr(), src=w, count=_WORDS)
    call[].state = _mint(words^)


def _sequence_member(operation: Int, address: Int) raises -> Int:
    var call = _CallP(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        if call[].state == 0:
            # Kind-exact from the NEW words when given (item 17).
            if call[].param_count == _WORDS and call[].params:
                return _capabilities(_F64P(unsafe_from_address=call[].params)[0] == 1.0)
            return _capabilities(False) | _capabilities(True)
        var probe = _SlotP(unsafe_from_address=_lookup(call[].state, False))
        return _capabilities(probe[].classifier)
    if operation == MEMBER_OP_NEW:
        _new(call)
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        _outputs(call)
        return 0
    if operation == MEMBER_OP_FREE:
        _ = _lookup(call[].state, True)
        call[].state = 0
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
        member_unsupported(_name(slot), operation)
    return 0


def sequence_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of the LSTM, GRU and RNN estimators (schema `MEMBER_ABI`)."""
    try:
        var word = _sequence_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def sequence_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's recurrent member entry. One line in each
    `_mojolearn_x_sequence` image's module initialization, after its glue
    entry. A build with no GPU vendor (`vendor` < 0) publishes nothing."""
    if vendor < 0:
        return
    member_publish[MEMBER_FAMILY_SEQUENCE](mode, vendor, publisher, sequence_member_v2)

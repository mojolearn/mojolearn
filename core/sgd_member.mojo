# SPDX-License-Identifier: Apache-2.0
"""The Sgd family as MEMBERS (core/member_abi.mojo, schema v2): the online
linear fits SGDRegressor, PassiveAggressiveRegressor, SGDClassifier,
Perceptron, PassiveAggressiveClassifier and SGDOneClassSVM.

One entry, `sgd_member_v2`, published by each x_linear image
(`sgd_member_publish`, beside its operations table and the XLinear member).
It runs the family's existing typed RUN (x_linear/door_run.mojo
`xl_fit_run`, `xl_class_fit_run`, `xl_decision_run`, `xl_r2_run`) over the
entries of the `DeviceRoute` the estimators' own doors resolve: `entry` the
x_linear operations entry (fit, the class-weight step and the decision),
`fifth` the metrics image's R^2 entry of the route's tier and device (the
regressors' score; filled by the caller's admission when that image is
loaded, else score is refused in words). No kernel is touched.

Kinds (the second word of each class's `_mojolearn_member`): 0
SGDRegressor, 1 PassiveAggressiveRegressor, 2 SGDClassifier, 3 Perceptron,
4 PassiveAggressiveClassifier, 5 SGDOneClassSVM.

MEMBER_OP_NEW parameters, 20 Float64 words for every kind (the integer and
float words of bindings/x_linear_sgd_unpack.mojo `_xl_sgd_build`, after the
kind's fixed choices: PassiveAggressive* take 'pa1' / 'pa2', no penalty,
alpha 1 and eta0 = C; Perceptron the perceptron loss and 'constant';
SGDOneClassSVM the hinge loss, 'l2', alpha = nu):
  [0] kind, [1] loss code (x_linear's SGD loss table: hinge 0, log_loss 1,
  modified_huber 2, squared_hinge 3, perceptron 4, squared_error 10, huber
  11, epsilon_insensitive 12, squared_epsilon_insensitive 13), [2] penalty
  code (None 0, l2 1, l1 2, elasticnet 3), [3] learning-rate code (constant
  0, optimal 1, invscaling 2, adaptive 3, pa1 5, pa2 6), [4] fit_intercept
  (0 / 1), [5] max_iter, [6] n_iter_no_change, [7] shuffle (0 / 1), [8] [9]
  the seed's low and high signed Int32 halves (random_state None: 0, 0),
  [10] batch (batch_size 'auto': minus the auto cap; None: -1; else its
  value), [11] batch_sum (0 / 1, `_sgd_auto_sum`), [12] alpha, [13]
  l1_ratio, [14] eta0, [15] power_t, [16] epsilon, [17] tol (None:
  -3.0e38), [18] class_weight (0 None, 1 'balanced'; a dict is not
  admitted: its labels never cross the ABI), [19] probability (1:
  SGDClassifier with loss='log_loss', which alone publishes
  PREDICT_PROBA).
The estimators' own checks (early_stopping, average, warm_start, the loss,
penalty and learning-rate words, eta0 > 0, alpha > 0 for 'optimal', C > 0,
nu in (0, 1], batch_size) run at the caller's admission.

Views: X dense row-major host float32 (borrowed; never copied). REGRESSORS
(kinds 0, 1): y host float32 rows x 1; PREDICT float32 rows x 1 (X @ coef +
intercept); SCORE R^2 of that prediction (unweighted). CLASSIFIERS (kinds
2..4): y the caller's int32 class codes 0..classes-1 (rows x 1),
`call.classes` the class count (>= 2) at FIT and at IMPORT-done; PREDICT
int32 codes rows x 1 (a binary score picks code 1 when > 0); DECISION
float32 rows x T (T = 1 for two classes, else classes); PREDICT_PROBA
(kind 2, log_loss) float32 rows x classes, `predict_proba`'s rows; SCORE
the mean accuracy (an exact count split over the host's workers).
SGDOneClassSVM (kind 5): y is not read; PREDICT int32 rows x 1 (+1 where
X @ coef - offset >= 0, else -1); DECISION float32 rows x 1 (X @ coef -
offset); SCORE_SAMPLES float32 rows x 1 (X @ coef); no SCORE. Sample
weights (host float32 rows x 1) are read by kinds 0, 2, 3 and 5; a weight
given to kinds 1 and 4 is refused in words (their fit takes none). A
device-home view, a row or column index and a nonzero `fit_output` are
refused in words. A fit whose floating point under- or overflowed is
refused in the estimator's words.

Fitted-state blocks (EXPORT / IMPORT), in index order: 0
`MEMBER_BLOCK_COEF` float32 T x features (T = 1 but for a classifier of
more than two classes); 1 `MEMBER_BLOCK_INTERCEPT` float32 1 x T
(SGDOneClassSVM: its offset_); 2 `MEMBER_BLOCK_SCALARS` float64 [n_iter_,
t_]. An import needs all three (and `call.classes` for a classifier).

The fitted state is OWNED here, in a process-wide store of fixed chunks (the
linear member's form); the lock covers only mint, free and the handle lookup.
Serial by nature here: the store's bookkeeping.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from core.entry_table import entry_ref_unresolved
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.linear_entry_table_ext import XL_OP_DECISION, XL_OP_DECISION_CODES, XL_OP_DECISION_ROWS
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_DTYPE_UNSAID,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_SCORE, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_OP_DECISION,
    MEMBER_OP_PREDICT_PROBA, MEMBER_OP_SCORE_SAMPLES, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_SCORE_SAMPLES,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_SCORE, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_COEF, MEMBER_CAN_DECISION, MEMBER_CAN_CLASSIFIER,
    MEMBER_BLOCK_COEF, MEMBER_BLOCK_INTERCEPT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_SGD
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from x_linear.door_run import (
    ALGO_SGD, LINK_IDENTITY, LINK_SIGMOID, ROWS_SGD_PROBA, XlClassPrepRequest, xl_class_prep_none,
    xl_class_fit_run, xl_run_block, XlFitRequest, XlDecisionRequest, xl_fit_run, xl_decision_run, xl_r2_run,
)

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime SGD_MEMBER_PARAMS = 20
comptime SGD_MEMBER_REGRESSOR = 0
comptime SGD_MEMBER_PA_REGRESSOR = 1
comptime SGD_MEMBER_CLASSIFIER = 2
comptime SGD_MEMBER_PERCEPTRON = 3
comptime SGD_MEMBER_PA_CLASSIFIER = 4
comptime SGD_MEMBER_ONE_CLASS = 5
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]


struct SgdMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var params: List[Float64]
    var fitted: Bool
    var features: Int
    # T: the decision rows (1, or the classes of a multiclass classifier).
    var outputs: Int
    var classes: Int
    # The fit's [coef (T x features) | intercept (T)] words, which the
    # export views; SGDOneClassSVM's intercept word is its offset_.
    var record: List[Float32]
    # The decision block: T rows of [coef | intercept] (SGDOneClassSVM:
    # [coef | -offset]); `plain` [coef | 0] is SGDOneClassSVM's raw score.
    var block: List[Float32]
    var plain: List[Float32]
    # [n_iter_, t_]
    var scalars: List[Float64]
    # The intercept words an import stages until DONE.
    var staged: List[Float32]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.params = List[Float64]()
        self.fitted = False
        self.features = 0
        self.outputs = 1
        self.classes = 0
        self.record = List[Float32]()
        self.block = List[Float32]()
        self.plain = List[Float32]()
        self.scalars = List[Float64]()
        self.staged = List[Float32]()

    def forget(mut self):
        self.fitted = False
        self.features = 0
        self.outputs = 1
        self.classes = 0
        self.record = List[Float32]()
        self.block = List[Float32]()
        self.plain = List[Float32]()
        self.scalars = List[Float64]()
        self.staged = List[Float32]()


struct SgdMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[SgdMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[SgdMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[SgdMemberStore, "MojolearnSgdMemberStoreV2", 2]
comptime _SlotP = MutPointer[SgdMemberSlot, MutUntrackedOrigin]
comptime _CallP = MutPointer[MemberCall, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock; a chunk never grows, so the address stays valid.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[SgdMemberSlot]()


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
                raise Error("RuntimeError: mojolearn: too many live SGD member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[SgdMemberSlot](length=_CHUNK, fill=SgdMemberSlot()))
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


def _kind(slot: _SlotP) -> Int:
    return Int(slot[].params[0])


def _name(slot: _SlotP) -> String:
    var kind = _kind(slot)
    if kind == SGD_MEMBER_REGRESSOR:
        return String("SGDRegressor")
    if kind == SGD_MEMBER_PA_REGRESSOR:
        return String("PassiveAggressiveRegressor")
    if kind == SGD_MEMBER_CLASSIFIER:
        return String("SGDClassifier")
    if kind == SGD_MEMBER_PERCEPTRON:
        return String("Perceptron")
    if kind == SGD_MEMBER_PA_CLASSIFIER:
        return String("PassiveAggressiveClassifier")
    return String("SGDOneClassSVM")


def _classifier(kind: Int) -> Bool:
    return kind >= SGD_MEMBER_CLASSIFIER and kind <= SGD_MEMBER_PA_CLASSIFIER


def _weighted(kind: Int) -> Bool:
    """Whether `kind`'s fit takes sample weights (PassiveAggressive* do not)."""
    return kind != SGD_MEMBER_PA_REGRESSOR and kind != SGD_MEMBER_PA_CLASSIFIER


def _capabilities(kind: Int, probability: Bool) -> Int:
    """The bits `kind` honors; kind -1: the family's union."""
    var bits = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT | MEMBER_CAN_COEF
    if kind != SGD_MEMBER_ONE_CLASS:
        # SGDOneClassSVM has no score method.
        bits |= MEMBER_CAN_SCORE
    if kind < 0 or _weighted(kind):
        bits |= MEMBER_CAN_SAMPLE_WEIGHT
    if kind < 0 or _classifier(kind):
        bits |= MEMBER_CAN_DECISION | MEMBER_CAN_CLASSIFIER
    if kind < 0 or (kind == SGD_MEMBER_CLASSIFIER and probability):
        bits |= MEMBER_CAN_PREDICT_PROBA
    if kind < 0 or kind == SGD_MEMBER_ONE_CLASS:
        bits |= MEMBER_CAN_DECISION | MEMBER_CAN_SCORE_SAMPLES
    return bits


def _dense(view: MemberMatrix, what: String, estimator: String) raises:
    if not view.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + estimator + " member reads " + what + " as a dense host float32 view")


def _whole(call: _CallP, estimator: String) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads whole host views (no row or column index)")


def _codes_view(view: MemberMatrix, rows: Int, what: String, estimator: String) raises:
    if not view.dense_host(MEMBER_I32) or view.cols != 1 or view.rows != rows:
        raise Error("TypeError: a " + estimator + " member reads " + what
            + " as int32 class codes, one per row of X (a dense host view)")


def _codes_f32(src: Int, dst: Int, n: Int):
    """The float32 words of n int32 class codes (`labels_cast`), rows split
    across the host's workers."""
    var tasks = host_predict_task_count_work(n, n)
    var chunk = host_predict_chunk(n, tasks)
    var source = _I32P(unsafe_from_address=src)
    var target = _F32P(unsafe_from_address=dst)

    def task(worker: Int) {imm chunk, imm source, imm target, imm n}:
        var begin = worker * chunk
        var end = min(begin + chunk, n)
        for r in range(begin, end):
            target[r] = source[r].cast[DType.float32]()

    host_parallelize(task, tasks)


def _fit(slot: _SlotP, call: _CallP) raises:
    """One `_sgd_fit`: the request words of `_xl_sgd_build` over the target
    words [targets (class codes as float32; SGDOneClassSVM zeros) | sample
    weights], the class-weight step first when class_weight='balanced'.
    Result words: [coef (P x d) | intercept (P) | n_iter | overflow flag],
    P = classes for a multiclass classifier, else 1."""
    var estimator = _name(slot)
    var kind = _kind(slot)
    _whole(call, estimator)
    if call[].fit_output != 0:
        member_unsupported(estimator, call[].fit_output)
    _dense(call[].x, "X", estimator)
    var n = call[].x.rows
    var d = call[].x.cols
    var k = 0
    if _classifier(kind):
        _codes_view(call[].y, n, "y", estimator)
        k = call[].classes
        if k < 2:
            raise Error("mojolearn " + estimator + ": y has one class")
    elif kind == SGD_MEMBER_ONE_CLASS:
        k = 1
    else:
        _dense(call[].y, "y", estimator)
        if call[].y.cols != 1:
            raise Error("ValueError: a " + estimator + " member fits one target column")
        if call[].y.rows != n:
            raise Error("ValueError: X and y have different numbers of rows")
    var has_sw = 0
    if call[].weights.rows > 0:
        if not _weighted(kind):
            raise Error("TypeError: " + estimator + ".fit() takes no sample_weight")
        _dense(call[].weights, "sample_weight", estimator)
        if call[].weights.rows != n or call[].weights.cols != 1:
            raise Error("ValueError: sample_weight and X have different numbers of rows")
        has_sw = 1
    var p = slot[].params.copy()
    slot[].forget()
    # [targets | weights]: one join copy, as `_xl_with_weights` (a regressor
    # without weights reads y in place).
    var parts = List[Float32]()
    var y = call[].y.data
    if _classifier(kind) or kind == SGD_MEMBER_ONE_CLASS or has_sw == 1:
        parts = List[Float32](length=n * (1 + has_sw), fill=0.0)
        y = Int(parts.unsafe_ptr())
        if _classifier(kind):
            _codes_f32(call[].y.data, y, n)
        elif kind != SGD_MEMBER_ONE_CLASS:
            unsafe_memcpy(dest=parts.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].y.data), count=n)
        if has_sw == 1:
            unsafe_memcpy(dest=parts.unsafe_ptr() + n, src=_F32P(unsafe_from_address=call[].weights.data), count=n)
    var problems = k if k > 2 else 1
    var balanced = _classifier(kind) and p[18] != 0.0
    var ip: List[Int] = [k, Int(p[1]), Int(p[2]), Int(p[3]), Int(p[4]), Int(p[5]), Int(p[6]), Int(p[7]), Int(p[8]),
        Int(p[9]), has_sw, 1 if balanced else 0, Int(p[10]), Int(p[11])]
    var fp: List[Float64] = [p[12], p[13], p[14], p[15], p[16], p[17]]
    var n_out = problems * d + problems + 2
    var result = List[Float32](length=n_out, fill=0.0)
    var request = XlFitRequest(call[].route.entry.call, ALGO_SGD, call[].x.data, n * d, n, d, y, n * (1 + has_sw),
        ip^, fp^, n_out, problems * (n + d + 1), problems * n, Int(result.unsafe_ptr()), False, 0, 0, 0)
    if balanced:
        # 'balanced': the class-weight step writes the k weights the fit
        # takes as its last float parameters (`xl_class_fit_run`).
        var class_weight = List[Float32](length=k, fill=1.0)
        var prep = XlClassPrepRequest(True, call[].y.data, 0, Int(class_weight.unsafe_ptr()), 0, n, k, True, False)
        _ = xl_class_fit_run(request, prep)
        _ = class_weight^
    else:
        _ = xl_fit_run(request)
    _ = parts^
    if result[n_out - 1] != 0.0:
        raise Error("Floating-point under-/overflow occurred. Scaling input data with "
            + "StandardScaler or MinMaxScaler might help.")
    var n_iter = Int(result[n_out - 2])
    var record = List[Float32](length=problems * d + problems, fill=0.0)
    unsafe_memcpy(dest=record.unsafe_ptr(), src=result.unsafe_ptr(), count=problems * d + problems)
    _keep(slot, record^, problems, d, k if _classifier(kind) else 0)
    slot[].scalars = [Float64(n_iter), Float64(n_iter * n + 1)]
    slot[].fitted = True


def _keep(slot: _SlotP, var record: List[Float32], outputs: Int, d: Int, classes: Int):
    """Hold the fit's [coef (T x d) | intercept (T)] words and build the
    decision block (T rows of [coef | intercept]) from them once;
    SGDOneClassSVM's block holds -offset and its raw-score block 0."""
    slot[].block = List[Float32](length=outputs * (d + 1), fill=0.0)
    xl_run_block(Int(slot[].block.unsafe_ptr()), Int(record.unsafe_ptr()), Int(record.unsafe_ptr()) + 4 * outputs * d,
        outputs, d)
    if _kind(slot) == SGD_MEMBER_ONE_CLASS:
        # `Array.from_list([-offset], '<f4')` of the own doors
        slot[].block[d] = -record[d]
        slot[].plain = slot[].block.copy()
        slot[].plain[d] = 0.0
    slot[].record = record^
    slot[].outputs = outputs
    slot[].features = d
    slot[].classes = classes


def _query(slot: _SlotP, call: _CallP) raises:
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].x.cols != slot[].features:
        raise Error("ValueError: X has " + String(call[].x.cols) + " features, but " + estimator
            + " is expecting " + String(slot[].features) + " features as input.")


def _scores_into(slot: _SlotP, call: _CallP, block: Int, dst: Int) raises:
    """float32 X @ coef + intercept of each row and decision row."""
    if call[].x.rows == 0:
        return
    xl_decision_run(XlDecisionRequest(call[].route.entry.call, XL_OP_DECISION, call[].x.data, block, dst,
        call[].x.rows, call[].x.cols, slot[].outputs, LINK_IDENTITY, True, 0, 0, 0, 0))


def _codes_into(slot: _SlotP, call: _CallP, dst: Int) raises:
    """Each row's int32 code: a classifier's class (strict: code 1 only
    when a binary score is > 0), SGDOneClassSVM's +1 / -1 (>= 0)."""
    if call[].x.rows == 0:
        return
    var one_class = _kind(slot) == SGD_MEMBER_ONE_CLASS
    xl_decision_run(XlDecisionRequest(call[].route.entry.call, XL_OP_DECISION_CODES, call[].x.data,
        Int(slot[].block.unsafe_ptr()), dst, call[].x.rows, call[].x.cols, slot[].outputs, LINK_IDENTITY,
        not one_class, -1 if one_class else 0, 1, 0, 0))


def _one_column(slot: _SlotP, call: _CallP, what: String) raises:
    _dense(call[].dst, what, _name(slot))
    if call[].dst.rows != call[].x.rows or call[].dst.cols != 1:
        raise Error("ValueError: " + what + " holds one value per row of X")


def _predict(slot: _SlotP, call: _CallP) raises:
    _query(slot, call)
    var kind = _kind(slot)
    if _classifier(kind) or kind == SGD_MEMBER_ONE_CLASS:
        _codes_view(call[].dst, call[].x.rows, "the prediction buffer", _name(slot))
        _codes_into(slot, call, call[].dst.data)
        return
    _one_column(slot, call, "the prediction buffer")
    _scores_into(slot, call, Int(slot[].block.unsafe_ptr()), call[].dst.data)


def _decision(slot: _SlotP, call: _CallP) raises:
    var kind = _kind(slot)
    if not _classifier(kind) and kind != SGD_MEMBER_ONE_CLASS:
        member_unsupported(_name(slot), MEMBER_OP_DECISION)
    _query(slot, call)
    _dense(call[].dst, "the decision buffer", _name(slot))
    if call[].dst.rows != call[].x.rows or call[].dst.cols != slot[].outputs:
        raise Error("ValueError: the decision buffer holds " + String(slot[].outputs) + " scores per row of X")
    _scores_into(slot, call, Int(slot[].block.unsafe_ptr()), call[].dst.data)


def _score_samples(slot: _SlotP, call: _CallP) raises:
    if _kind(slot) != SGD_MEMBER_ONE_CLASS:
        member_unsupported(_name(slot), MEMBER_OP_SCORE_SAMPLES)
    _query(slot, call)
    _one_column(slot, call, "the score buffer")
    _scores_into(slot, call, Int(slot[].plain.unsafe_ptr()), call[].dst.data)


def _proba(slot: _SlotP, call: _CallP) raises:
    """SGDClassifier.predict_proba (loss='log_loss'): the decision and its
    row link in one call (XL_OP_DECISION_ROWS, LINK_SIGMOID, ROWS_SGD_PROBA),
    rows x classes."""
    if _kind(slot) != SGD_MEMBER_CLASSIFIER:
        member_unsupported(_name(slot), MEMBER_OP_PREDICT_PROBA)
    if slot[].params[19] == 0.0:
        raise Error("AttributeError: probability estimates are not available for this SGDClassifier's loss "
            + "(only loss='log_loss')")
    _query(slot, call)
    var width = 2 if slot[].outputs == 1 else slot[].outputs
    _dense(call[].dst, "the probability buffer", _name(slot))
    if call[].dst.rows != call[].x.rows or call[].dst.cols != width:
        raise Error("ValueError: the probability buffer holds " + String(width) + " values per row of X")
    if call[].x.rows == 0:
        return
    xl_decision_run(XlDecisionRequest(call[].route.entry.call, XL_OP_DECISION_ROWS, call[].x.data,
        Int(slot[].block.unsafe_ptr()), call[].dst.data, call[].x.rows, call[].x.cols, slot[].outputs, LINK_SIGMOID,
        False, 0, 0, ROWS_SGD_PROBA, 0))


def _score(slot: _SlotP, call: _CallP) raises:
    var estimator = _name(slot)
    var kind = _kind(slot)
    if kind == SGD_MEMBER_ONE_CLASS:
        member_unsupported(estimator, MEMBER_OP_SCORE)
    _query(slot, call)
    if call[].weights.rows > 0:
        raise Error("NotImplementedError: mojolearn: a weighted score of a " + estimator
            + " member is not published; the caller predicts and runs its own weighted metric")
    var rows = call[].x.rows
    if not _classifier(kind):
        var route = call[].route
        _dense(call[].y, "y", estimator)
        if call[].y.cols != 1 or call[].y.rows != rows:
            raise Error("ValueError: y must be 1-D with one entry per row of X")
        if not route.has_fifth:
            raise Error("ImportError: mojolearn: the " + estimator
                + " member's score needs the metrics R^2 entry on its route")
        var prediction = List[Float32](length=rows, fill=0.0)
        _scores_into(slot, call, Int(slot[].block.unsafe_ptr()), Int(prediction.unsafe_ptr()))
        call[].value = xl_r2_run(route.fifth.call, entry_ref_unresolved(), call[].y.data,
            Int(prediction.unsafe_ptr()), 0, rows, True)
        _ = prediction^
        return
    _codes_view(call[].y, rows, "y", estimator)
    if rows == 0:
        raise Error("ValueError: Found array with 0 sample(s) while a minimum of 1 is required.")
    var predicted = List[Int32](length=rows, fill=0)
    _codes_into(slot, call, Int(predicted.unsafe_ptr()))
    # The exact count of equal codes (an integer, so the split cannot change it).
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    var counts = List[Int](length=tasks, fill=0)
    var hits = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(counts.unsafe_ptr()))
    var codes = _I32P(unsafe_from_address=Int(predicted.unsafe_ptr()))
    var truth = _I32P(unsafe_from_address=call[].y.data)

    def task(worker: Int) {imm hits, imm codes, imm truth, imm chunk, imm rows}:
        var begin = worker * chunk
        var end = min(begin + chunk, rows)
        var count = 0
        for r in range(begin, end):
            if truth[r] == codes[r]:
                count += 1
        hits[worker] = count

    host_parallelize(task, tasks)
    var total = 0
    for worker in range(tasks):  # driver-loop: over the workers' partial counts
        total += counts[worker]
    _ = predicted^
    call[].value = Float64(total) / Float64(rows)


def _export(slot: _SlotP, call: _CallP) raises:
    """A view of one fitted block, owned by the slot until its next fit,
    import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot) + " member is not fitted yet")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= 3:
            raise Error("member ABI: a " + _name(slot) + " member state has 3 fitted blocks")
        block = MEMBER_BLOCK_COEF if block == 0 else (MEMBER_BLOCK_INTERCEPT if block == 1 else MEMBER_BLOCK_SCALARS)
    var words = Int(slot[].record.unsafe_ptr())
    if block == MEMBER_BLOCK_COEF:
        call[].dst = MemberMatrix(words, slot[].outputs, slot[].features)
    elif block == MEMBER_BLOCK_INTERCEPT:
        call[].dst = MemberMatrix(words + 4 * slot[].outputs * slot[].features, 1, slot[].outputs)
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, 2, MEMBER_F64)
    else:
        raise Error("member ABI: a " + _name(slot) + " member state holds no fitted block " + String(block))
    call[].block = block
    call[].out_cols = 3
    call[].classes = slot[].classes


def _import(slot: _SlotP, call: _CallP) raises:
    """COEF (T x features), INTERCEPT (1 x T) and SCALARS [n_iter_, t_],
    then DONE (with `call.classes` for a classifier)."""
    var estimator = _name(slot)
    var kind = _kind(slot)
    if call[].block == MEMBER_BLOCK_COEF or call[].block == MEMBER_BLOCK_INTERCEPT:
        _dense(call[].x, "a fitted block", estimator)
        var count = call[].x.rows * call[].x.cols
        var words = List[Float32](length=count, fill=0.0)
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        slot[].fitted = False
        if call[].block == MEMBER_BLOCK_COEF:
            # a 1-D coef_ (regressors, SGDOneClassSVM) is one row
            slot[].outputs = call[].x.rows if _classifier(kind) else 1
            slot[].features = count // slot[].outputs if slot[].outputs > 0 else 0
            slot[].block = words^
        else:
            slot[].staged = words^
    elif call[].block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != 2:
            raise Error("member ABI: the " + estimator + " scalars block is 2 host float64 words [n_iter_, t_]")
        slot[].fitted = False
        slot[].scalars = List[Float64](length=2, fill=0.0)
        unsafe_memcpy(dest=slot[].scalars.unsafe_ptr(), src=_F64P(unsafe_from_address=call[].x.data), count=2)
    elif call[].block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a " + estimator + " member imports no fitted block " + String(call[].block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var outputs = slot[].outputs
    var d = slot[].features
    if outputs < 1 or len(slot[].block) != outputs * d or len(slot[].staged) != outputs or len(slot[].scalars) != 2:
        raise Error("member ABI: a fitted " + estimator + " state needs its coefficient (T x features), "
            + "intercept (1 x T) and scalars blocks")
    var classes = 0
    if _classifier(kind):
        classes = call[].classes
        if classes < 2 or (1 if classes == 2 else classes) != outputs:
            raise Error("member ABI: a " + estimator + " state of " + String(outputs) + " decision rows needs its "
                + "class count at import")
    var record = List[Float32](length=outputs * d + outputs, fill=0.0)
    unsafe_memcpy(dest=record.unsafe_ptr(), src=slot[].block.unsafe_ptr(), count=outputs * d)
    unsafe_memcpy(dest=record.unsafe_ptr() + outputs * d, src=slot[].staged.unsafe_ptr(), count=outputs)
    slot[].staged = List[Float32]()
    _keep(slot, record^, outputs, d, classes)
    slot[].fitted = True


def _sgd_member(operation: Int, address: Int) raises -> Int:
    var call = _CallP(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        # A live state answers for its kind and settings; without one the
        # NEW words do when given (kind-exact, item 17), else the union.
        var bits: Int
        var regressor: Bool
        if call[].state == 0:
            if not call[].params or call[].param_count != SGD_MEMBER_PARAMS:
                return _capabilities(-1, True)
            var words = _F64P(unsafe_from_address=call[].params)
            bits = _capabilities(Int(words[0]), words[19] != 0.0)
            regressor = not _classifier(Int(words[0])) and Int(words[0]) != SGD_MEMBER_ONE_CLASS
        else:
            var asked = _SlotP(unsafe_from_address=_lookup(call[].state, False))
            bits = _capabilities(_kind(asked), asked[].params[19] != 0.0)
            regressor = not _classifier(_kind(asked)) and _kind(asked) != SGD_MEMBER_ONE_CLASS
        if regressor and not call[].route.has_fifth:
            # A regressor's score is the metrics image's R^2 entry: refused
            # at RUN without it, so not advertised.
            bits &= ~MEMBER_CAN_SCORE
        return bits
    if operation == MEMBER_OP_NEW:
        if not call[].params or call[].param_count < 1:
            raise Error("member ABI: an SGD member takes its kind as parameter word 0")
        var words = _F64P(unsafe_from_address=call[].params)
        var kind = Int(words[0])
        if Float64(kind) != words[0] or kind < 0 or kind > SGD_MEMBER_ONE_CLASS:
            raise Error("member ABI: unknown SGD member kind")
        if call[].param_count != SGD_MEMBER_PARAMS:
            raise Error("member ABI: an SGD member takes " + String(SGD_MEMBER_PARAMS) + " parameters")
        var params = List[Float64](length=SGD_MEMBER_PARAMS, fill=0.0)
        unsafe_memcpy(dest=params.unsafe_ptr(), src=words, count=SGD_MEMBER_PARAMS)
        call[].state = _mint(params^)
        return 0
    if operation == MEMBER_OP_FREE:
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        # With a state the state answers; without one the NEW words (the
        # kind) and the caller's class count do (MEMBER-ABI-V2 item 10).
        var asked_kind = -1
        var classes = 0
        if call[].state != 0:
            var held = _SlotP(unsafe_from_address=_lookup(call[].state, False))
            asked_kind = _kind(held)
            classes = held[].classes
        elif call[].params and call[].param_count >= 1:
            asked_kind = Int(_F64P(unsafe_from_address=call[].params)[0])
            classes = call[].classes
        var op = call[].flags
        if asked_kind >= 0:
            # An operation the kind lacks is refused, never sized (proba
            # needs loss='log_loss': word 19).
            var probability = True
            if call[].state != 0:
                probability = _SlotP(unsafe_from_address=_lookup(call[].state, False))[].params[19] != 0.0
            elif call[].param_count == SGD_MEMBER_PARAMS:
                probability = _F64P(unsafe_from_address=call[].params)[19] != 0.0
            var can = _capabilities(asked_kind, probability)
            var served = ((op == MEMBER_OP_PREDICT and (can & MEMBER_CAN_PREDICT) != 0)
                or (op == MEMBER_OP_DECISION and (can & MEMBER_CAN_DECISION) != 0)
                or (op == MEMBER_OP_PREDICT_PROBA and (can & MEMBER_CAN_PREDICT_PROBA) != 0)
                or (op == MEMBER_OP_SCORE_SAMPLES and (can & MEMBER_CAN_SCORE_SAMPLES) != 0))
            if not served:
                member_unsupported("an SGD member of kind " + String(asked_kind), op)
        call[].out_cols = 1
        call[].classes = 0
        # Regressor predict, every decision, score_samples and proba:
        # float32; a classifier's or SGDOneClassSVM's predict: int32 codes.
        call[].out_dtype = MEMBER_F32
        if op == MEMBER_OP_PREDICT and asked_kind >= 0 and (_classifier(asked_kind)
                or asked_kind == SGD_MEMBER_ONE_CLASS):
            call[].out_dtype = MEMBER_I32
        elif op == MEMBER_OP_PREDICT and asked_kind < 0:
            call[].out_dtype = MEMBER_DTYPE_UNSAID  # the kind decides (codes or values)
        if asked_kind >= 0 and _classifier(asked_kind):
            call[].classes = classes
            if op == MEMBER_OP_DECISION:
                call[].out_cols = max(1, 1 if classes == 2 else classes)
            elif op == MEMBER_OP_PREDICT_PROBA:
                call[].out_cols = max(1, classes)
        return 0
    var slot = _SlotP(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_PREDICT:
        _predict(slot, call)
    elif operation == MEMBER_OP_DECISION:
        _decision(slot, call)
    elif operation == MEMBER_OP_PREDICT_PROBA:
        _proba(slot, call)
    elif operation == MEMBER_OP_SCORE_SAMPLES:
        _score_samples(slot, call)
    elif operation == MEMBER_OP_SCORE:
        _score(slot, call)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_name(slot), operation)
    return 0


def sgd_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of the SGD family (schema `MEMBER_ABI`)."""
    try:
        var word = _sgd_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def sgd_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's SGD member entry. One line in the image's
    module initialization, after its operations table is published."""
    member_publish[MEMBER_FAMILY_SGD](mode, vendor, publisher, sgd_member_v2)

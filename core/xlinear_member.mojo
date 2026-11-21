# SPDX-License-Identifier: Apache-2.0
"""The XLinear family as MEMBERS (core/member_abi.mojo, schema v2): the
generalized linear regressors PoissonRegressor, GammaRegressor and
TweedieRegressor, HuberRegressor, BayesianRidge and ARDRegression.

One entry, `xlinear_member_v2`, published by each x_linear image
(`xlinear_member_publish`, beside its operations table). It runs the family's
existing typed RUN (x_linear/door_run.mojo `xl_fit_run`, `xl_huber_fit_run`,
`xl_decision_run`, `xl_r2_run`) over the entries of the `DeviceRoute` the
estimator's own doors resolve: `entry` the x_linear operations entry (fit and
decision), `fifth` the metrics image's R^2 entry of the route's tier and
device (filled by the caller's admission when that image is loaded; without
it score is refused in words and the caller predicts and runs its own
metric). No kernel is touched.

Kinds (the second word of each class's `_mojolearn_member`): 0
PoissonRegressor, 1 GammaRegressor, 2 TweedieRegressor, 3 HuberRegressor,
4 BayesianRidge, 5 ARDRegression.

MEMBER_OP_NEW parameters (Float64 words; word 0 is the kind, the count is
the kind's):
  * kinds 0..2, 7 words: [0] kind, [1] fit_intercept (0 / 1), [2] max_iter,
    [3] link (0 identity, 1 log; Poisson and Gamma: 1), [4] power (Poisson 1,
    Gamma 2, Tweedie its own), [5] alpha, [6] tol.
  * kind 3, 6 words: [0] 3, [1] fit_intercept, [2] max_iter, [3] epsilon,
    [4] alpha, [5] tol.
  * kind 4, 10 words: [0] 4, [1] fit_intercept, [2] max_iter, [3] tol,
    [4] alpha_1, [5] alpha_2, [6] lambda_1, [7] lambda_2, [8] alpha_init,
    [9] lambda_init (-1: None).
  * kind 5, 9 words: [0] 5, [1] fit_intercept, [2] max_iter, [3] tol,
    [4] alpha_1, [5] alpha_2, [6] lambda_1, [7] lambda_2, [8] threshold_lambda.
The estimator's own checks (solver, warm_start, alpha >= 0, Tweedie's power
and link words, Huber's epsilon >= 1, BayesianRidge's compute_score) run at the caller's admission.

Views: X dense row-major host float32; y and sample weights host float32
vectors (one target); kind 5 takes no sample weights (a given weight is
refused in words; MEMBER_OP_CAPABILITIES of its state says so). A device-home view, a row or column index and a nonzero
`fit_output` are refused in words. PREDICT writes the prediction as float32,
rows x 1: link(X @ coef + intercept) for kinds 0..2, X @ coef + intercept
for the others. SCORE is R^2 of that prediction (unweighted; a weighted score
is not published). A GLM's targets outside its loss's range are refused after
the fit, in the estimator's words.

Fitted-state blocks (EXPORT / IMPORT), in index order:
0 `MEMBER_BLOCK_COEF` float32 1 x features; 1 `MEMBER_BLOCK_SCALARS` float64
in the kind's order: kinds 0..2 [intercept, n_iter]; kind 3 [intercept,
scale, n_iter]; kind 4 [intercept, alpha_, lambda_, n_iter]; kind 5 [intercept, alpha_,
n_iter]. Kind 5 also holds 2 `XL_MEMBER_BLOCK_LAMBDA` (`MEMBER_BLOCK_FAMILY` +
1) float32 1 x features (`lambda_`). Kind 3 after a fit also holds 2 `XL_MEMBER_BLOCK_OUTLIERS`
(`MEMBER_BLOCK_FAMILY`) int32 1 x training rows, 1 for an outlier row
(`outliers_`). An import needs COEF and SCALARS; a family block is a training
record, optional at import (an imported state then exports two blocks).

The fitted state is OWNED here, in a process-wide store of fixed chunks (the
linear member's form); the lock covers only mint, free and the handle lookup.
Serial by nature here: the store's bookkeeping.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy
from core.entry_table import entry_ref_unresolved
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.linear_entry_table_ext import XL_OP_DECISION
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_SCORE, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_SCORE, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_CAN_COEF,
    MEMBER_BLOCK_COEF, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_XLINEAR
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from x_linear.door_run import (
    ALGO_GLM, ALGO_HUBER, ALGO_BAYES, ALGO_ARD, LINK_IDENTITY, LINK_EXP, XlFitRequest, XlDecisionRequest, xl_fit_run,
    xl_huber_fit_run, xl_decision_run, xl_r2_run, xl_lbfgs_work,
)

comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime XL_MEMBER_HUBER = 3
comptime XL_MEMBER_BAYES = 4
comptime XL_MEMBER_ARD = 5
comptime XL_MEMBER_BLOCK_LAMBDA = MEMBER_BLOCK_FAMILY + 1
comptime XL_MEMBER_BLOCK_OUTLIERS = MEMBER_BLOCK_FAMILY
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _U8P = MutPointer[UInt8, MutUntrackedOrigin]


def xl_member_param_count(kind: Int) -> Int:
    """The MEMBER_OP_NEW word count of `kind` (0: unknown kind)."""
    if kind >= 0 and kind <= 2:
        return 7
    if kind == XL_MEMBER_HUBER:
        return 6
    if kind == XL_MEMBER_BAYES:
        return 10
    if kind == XL_MEMBER_ARD:
        return 9
    return 0


def xl_member_scalar_count(kind: Int) -> Int:
    """The SCALARS block length of `kind`."""
    if kind == XL_MEMBER_HUBER:
        return 3
    if kind == XL_MEMBER_BAYES:
        return 4
    if kind == XL_MEMBER_ARD:
        return 3
    return 2


struct XLinearMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var params: List[Float64]
    var fitted: Bool
    var features: Int
    # [coef (features) | intercept]: the decision block of one target.
    var block: List[Float32]
    var scalars: List[Float64]
    # HuberRegressor's outlier flag of each training row (kernel bytes), and
    # their int32 export words, made once at the first export.
    var flags: List[UInt8]
    var outliers: List[Int32]
    # A per-feature fitted record of the kind (ARDRegression's lambda_).
    var record: List[Float32]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.params = List[Float64]()
        self.fitted = False
        self.features = 0
        self.block = List[Float32]()
        self.scalars = List[Float64]()
        self.flags = List[UInt8]()
        self.outliers = List[Int32]()
        self.record = List[Float32]()

    def forget(mut self):
        self.fitted = False
        self.features = 0
        self.block = List[Float32]()
        self.scalars = List[Float64]()
        self.flags = List[UInt8]()
        self.outliers = List[Int32]()
        self.record = List[Float32]()


struct XLinearMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[XLinearMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[XLinearMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _STORE = SharedSlot[XLinearMemberStore, "MojolearnXLinearMemberStoreV2", 2]
comptime _SlotP = MutPointer[XLinearMemberSlot, MutUntrackedOrigin]
comptime _CallP = MutPointer[MemberCall, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock; a chunk never grows, so the address stays valid.
    var store = _STORE.get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[XLinearMemberSlot]()


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
                raise Error("RuntimeError: mojolearn: too many live x_linear member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[XLinearMemberSlot](length=_CHUNK, fill=XLinearMemberSlot()))
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
    if kind == 0:
        return String("PoissonRegressor")
    if kind == 1:
        return String("GammaRegressor")
    if kind == 2:
        return String("TweedieRegressor")
    if kind == XL_MEMBER_HUBER:
        return String("HuberRegressor")
    if kind == XL_MEMBER_BAYES:
        return String("BayesianRidge")
    return String("ARDRegression")


def _loss_name(slot: _SlotP) -> String:
    # The class attribute `_loss_name` (Tweedie: None -> "HalfTweedieLoss").
    var kind = _kind(slot)
    return String("HalfPoissonLoss") if kind == 0 else (
        String("HalfGammaLoss") if kind == 1 else String("HalfTweedieLoss"))


def _capabilities(kind: Int) -> Int:
    """The bits `kind` honors; kind -1: the family's union."""
    var bits = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_SCORE | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
        | MEMBER_CAN_COEF)
    if _weighted(kind):
        bits |= MEMBER_CAN_SAMPLE_WEIGHT
    return bits


def _weighted(kind: Int) -> Bool:
    """Whether `kind`'s fit takes sample weights (kind -1: some kind does)."""
    return kind != XL_MEMBER_ARD


def _dense(view: MemberMatrix, what: String, estimator: String) raises:
    if not view.dense_host(MEMBER_F32):
        raise Error("TypeError: a " + estimator + " member reads " + what + " as a dense host float32 view")


def _whole(call: _CallP, estimator: String) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a " + estimator + " member reads whole host views (no row or column index)")


def _targets(call: _CallP, kind: Int, n: Int, estimator: String, mut joined: List[Float32], mut has_sw: Int) raises -> Int:
    """`_with_weights`: the target words the fit reads (the targets alone,
    or the targets then the weights in `joined`). Returns their address;
    `has_sw` says which."""
    has_sw = 0
    if call[].weights.rows == 0:
        return call[].y.data
    if not _weighted(kind):
        raise Error("TypeError: " + estimator + ".fit() takes no sample_weight")
    _dense(call[].weights, "sample_weight", estimator)
    if call[].weights.rows != n or call[].weights.cols != 1:
        raise Error("ValueError: sample_weight and X have different numbers of rows")
    joined = List[Float32](length=2 * n, fill=0.0)
    unsafe_memcpy(dest=joined.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].y.data), count=n)
    unsafe_memcpy(dest=joined.unsafe_ptr() + n, src=_F32P(unsafe_from_address=call[].weights.data), count=n)
    has_sw = 1
    return Int(joined.unsafe_ptr())


def _keep_block(slot: _SlotP, result: List[Float32], d: Int):
    slot[].block = List[Float32](length=d + 1, fill=0.0)
    unsafe_memcpy(dest=slot[].block.unsafe_ptr(), src=result.unsafe_ptr(), count=d + 1)
    slot[].features = d


def _fit(slot: _SlotP, call: _CallP) raises:
    var estimator = _name(slot)
    var kind = _kind(slot)
    _whole(call, estimator)
    if call[].fit_output != 0:
        member_unsupported(estimator, call[].fit_output)
    _dense(call[].x, "X", estimator)
    _dense(call[].y, "y", estimator)
    var n = call[].x.rows
    var d = call[].x.cols
    if call[].y.cols != 1:
        raise Error("ValueError: a " + estimator + " member fits one target column")
    if call[].y.rows != n:
        raise Error("ValueError: X and y have different numbers of rows")
    var joined = List[Float32]()
    var has_sw = 0
    var y = _targets(call, kind, n, estimator, joined, has_sw)
    var y_size = n * (1 + has_sw)
    var p = slot[].params.copy()
    var intercept = 1 if p[1] != 0.0 else 0
    slot[].forget()
    if kind == XL_MEMBER_HUBER:
        # The request words of bindings/x_linear_regressors_unpack.mojo
        # `xl_fit_huber_unpack`; the outlier flags come from the same session.
        var width = d + 2 if intercept == 1 else d + 1
        var ip: List[Int] = [Int(p[2]), intercept, has_sw]
        var fp: List[Float64] = [p[3], p[4], p[5]]
        var result = List[Float32](length=d + 4 + width, fill=0.0)
        var flags = List[UInt8](length=n, fill=0)
        var request = XlFitRequest(call[].route.entry.call, ALGO_HUBER, call[].x.data, n * d, n, d, y, y_size, ip^,
            fp^, d + 4 + width, xl_lbfgs_work(width) + n, 1, Int(result.unsafe_ptr()), False, 0, 0, 0)
        xl_huber_fit_run(request, call[].y.data, Int(flags.unsafe_ptr()), LINK_IDENTITY, p[3])
        _ = joined^
        _keep_block(slot, result, d)
        slot[].scalars = [Float64(result[d]), Float64(result[d + 1]), Float64(Int(result[d + 2]))]
        slot[].flags = flags^
        slot[].fitted = True
        return
    if kind >= XL_MEMBER_BAYES:
        _plain_fit(slot, call, kind, p, intercept, n, d, y, y_size, has_sw)
        _ = joined^
        return
    # GLM: the request words of `xl_fit_glm_unpack`.
    var m = d + 1
    var ip: List[Int] = [Int(p[2]), intercept, Int(p[3]), has_sw]
    var fp: List[Float64] = [p[4], p[5], p[6]]
    var result = List[Float32](length=d + 3, fill=0.0)
    var request = XlFitRequest(call[].route.entry.call, ALGO_GLM, call[].x.data, n * d, n, d, y, y_size, ip^, fp^,
        d + 3, 3 * n + m * m + 3 * m, 1, Int(result.unsafe_ptr()), False, 0, 0, 0)
    _ = xl_fit_run(request)
    _ = joined^
    if result[d + 2] < 0:
        raise Error("Some value(s) of y are out of the valid range of the loss '" + _loss_name(slot) + "'.")
    _keep_block(slot, result, d)
    slot[].scalars = [Float64(result[d]), Float64(Int(result[d + 1]))]
    slot[].fitted = True


def _plain_fit(slot: _SlotP, call: _CallP, kind: Int, p: List[Float64], intercept: Int, n: Int, d: Int, y: Int,
    y_size: Int, has_sw: Int) raises:
    """Kinds 4..: one `xl_fit_run` call with the request words of the kind's
    `xl_fit_<kind>_unpack` (bindings/x_linear_regressors_unpack.mojo), then
    the kind's result words into the state."""
    var algo = ALGO_BAYES
    var ip = List[Int]()
    var fp = List[Float64]()
    var n_out = 0
    var n_fw = 0
    var n_iw = 1
    if kind == XL_MEMBER_BAYES:
        ip = [Int(p[2]), intercept, has_sw]
        fp = [p[3], p[4], p[5], p[6], p[7], p[8], p[9]]
        n_out = d + 4
        n_fw = 3 * d * d + 5 * d + n
    elif kind == XL_MEMBER_ARD:
        if n < 2:
            raise Error("mojolearn ARDRegression: at least 2 samples are required")
        algo = ALGO_ARD
        ip = [Int(p[2]), intercept]
        fp = [p[3], p[4], p[5], p[6], p[7], p[8]]
        n_out = 2 * d + 4
        n_fw = 3 * d * d + 4 * d + n
        n_iw = 2 * d
    var result = List[Float32](length=n_out, fill=0.0)
    var request = XlFitRequest(call[].route.entry.call, algo, call[].x.data, n * d, n, d, y, y_size, ip^, fp^,
        n_out, n_fw, n_iw, Int(result.unsafe_ptr()), False, 0, 0, 0)
    _ = xl_fit_run(request)
    _keep_block(slot, result, d)
    if kind == XL_MEMBER_BAYES:
        # [coef | intercept | alpha_ | lambda_ | n_iter]
        slot[].scalars = [Float64(result[d]), Float64(result[d + 1]), Float64(result[d + 2]),
            Float64(Int(result[d + 3]))]
    elif kind == XL_MEMBER_ARD:
        # [coef | intercept | alpha_ | lambda_ (d) | n_iter]
        slot[].scalars = [Float64(result[d]), Float64(result[d + 1]), Float64(Int(result[2 * d + 2]))]
        slot[].record = List[Float32](length=d, fill=0.0)
        unsafe_memcpy(dest=slot[].record.unsafe_ptr(), src=result.unsafe_ptr() + d + 2, count=d)
    slot[].fitted = True


def _predict_into(slot: _SlotP, call: _CallP, dst: Int) raises:
    var estimator = _name(slot)
    if not slot[].fitted:
        raise Error("NotFittedError: this " + estimator + " member is not fitted yet")
    _whole(call, estimator)
    _dense(call[].x, "X", estimator)
    if call[].x.cols != slot[].features:
        raise Error("ValueError: X has " + String(call[].x.cols) + " features, but " + estimator
            + " is expecting " + String(slot[].features) + " features as input.")
    if call[].x.rows == 0:
        return
    var link = LINK_EXP if _kind(slot) <= 2 and slot[].params[3] == 1.0 else LINK_IDENTITY
    xl_decision_run(XlDecisionRequest(call[].route.entry.call, XL_OP_DECISION, call[].x.data,
        Int(slot[].block.unsafe_ptr()), dst, call[].x.rows, call[].x.cols, 1, link, True, 0, 0, 0, 0))


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


def _outlier_words(slot: _SlotP):
    """The int32 words of the outlier flags, made once, rows split across
    the host's workers."""
    var rows = len(slot[].flags)
    if len(slot[].outliers) == rows:
        return
    slot[].outliers = List[Int32](length=rows, fill=0)
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    var source = _U8P(unsafe_from_address=Int(slot[].flags.unsafe_ptr()))
    var target = _I32P(unsafe_from_address=Int(slot[].outliers.unsafe_ptr()))

    def task(worker: Int) {imm chunk, imm source, imm target, imm rows}:
        var begin = worker * chunk
        var end = min(begin + chunk, rows)
        for r in range(begin, end):
            target[r] = Int32(1) if source[r] != 0 else Int32(0)

    host_parallelize(task, tasks)


def _family_block(slot: _SlotP) -> Int:
    """The name of the one family block this state holds after the two
    shared ones, 0 for none (an imported state holds no training record)."""
    if _kind(slot) == XL_MEMBER_HUBER and len(slot[].flags) > 0:
        return XL_MEMBER_BLOCK_OUTLIERS
    if _kind(slot) == XL_MEMBER_ARD and len(slot[].record) > 0:
        return XL_MEMBER_BLOCK_LAMBDA
    return 0


def _export(slot: _SlotP, call: _CallP) raises:
    """A view of one fitted block, owned by the slot until its next fit,
    import or free."""
    if not slot[].fitted:
        raise Error("NotFittedError: this " + _name(slot) + " member is not fitted yet")
    var family = _family_block(slot)
    var count = 2 if family == 0 else 3
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        if block < 0 or block >= count:
            raise Error("member ABI: this " + _name(slot) + " member state has " + String(count) + " fitted blocks")
        block = MEMBER_BLOCK_COEF if block == 0 else (MEMBER_BLOCK_SCALARS if block == 1 else family)
    if block == MEMBER_BLOCK_COEF:
        call[].dst = MemberMatrix(Int(slot[].block.unsafe_ptr()), 1, slot[].features)
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].scalars.unsafe_ptr()), 1, len(slot[].scalars), MEMBER_F64)
    elif block != 0 and block == family and family == XL_MEMBER_BLOCK_OUTLIERS:
        _outlier_words(slot)
        call[].dst = MemberMatrix(Int(slot[].outliers.unsafe_ptr()), 1, len(slot[].outliers), MEMBER_I32)
    elif block != 0 and block == family and family == XL_MEMBER_BLOCK_LAMBDA:
        call[].dst = MemberMatrix(Int(slot[].record.unsafe_ptr()), 1, len(slot[].record))
    else:
        raise Error("member ABI: this " + _name(slot) + " member state holds no fitted block " + String(block))
    call[].block = block
    call[].out_cols = count


def _import(slot: _SlotP, call: _CallP) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE completes the state."""
    var scalar_count = xl_member_scalar_count(_kind(slot))
    if call[].block == MEMBER_BLOCK_COEF:
        _dense(call[].x, "a fitted block", _name(slot))
        var count = call[].x.rows * call[].x.cols
        var words = List[Float32](length=count + 1, fill=0.0)
        unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=call[].x.data), count=count)
        slot[].fitted = False
        slot[].features = count
        slot[].block = words^
    elif call[].block == MEMBER_BLOCK_SCALARS:
        if not call[].x.dense_host(MEMBER_F64) or call[].x.rows * call[].x.cols != scalar_count:
            raise Error("member ABI: the " + _name(slot) + " scalars block is " + String(scalar_count)
                + " host float64 words")
        var source = _F64P(unsafe_from_address=call[].x.data)
        slot[].fitted = False
        slot[].scalars = List[Float64](length=scalar_count, fill=0.0)
        unsafe_memcpy(dest=slot[].scalars.unsafe_ptr(), src=source, count=scalar_count)
    elif call[].block != 0 or (call[].flags & MEMBER_IMPORT_DONE) == 0:
        raise Error("member ABI: a " + _name(slot) + " member imports no fitted block " + String(call[].block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].block) == 0 or len(slot[].scalars) != scalar_count:
        raise Error("member ABI: a fitted " + _name(slot) + " state needs its coefficient and scalars blocks")
    slot[].block[slot[].features] = Float32(slot[].scalars[0])
    slot[].fitted = True


def _xlinear_member(operation: Int, address: Int) raises -> Int:
    var call = _CallP(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        if call[].state == 0:
            return _capabilities(-1)
        return _capabilities(_kind(_SlotP(unsafe_from_address=_lookup(call[].state, False))))
    if operation == MEMBER_OP_NEW:
        if not call[].params or call[].param_count < 1:
            raise Error("member ABI: an x_linear member takes its kind as parameter word 0")
        var words = _F64P(unsafe_from_address=call[].params)
        var kind = Int(words[0])
        var count = xl_member_param_count(kind)
        if Float64(kind) != words[0] or count == 0:
            raise Error("member ABI: unknown x_linear member kind")
        if call[].param_count != count:
            raise Error("member ABI: an x_linear member of kind " + String(kind) + " takes " + String(count)
                + " parameters")
        if kind <= 2 and words[3] != 0.0 and words[3] != 1.0:
            raise Error("member ABI: an x_linear GLM link word is 0 (identity) or 1 (log)")
        var params = List[Float64](length=count, fill=0.0)
        unsafe_memcpy(dest=params.unsafe_ptr(), src=words, count=count)
        call[].state = _mint(params^)
        return 0
    if operation == MEMBER_OP_FREE:
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
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


def xlinear_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of the x_linear regressors (schema `MEMBER_ABI`)."""
    try:
        var word = _xlinear_member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def xlinear_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's x_linear member entry. One line in the image's
    module initialization, after its operations table is published."""
    member_publish[MEMBER_FAMILY_XLINEAR](mode, vendor, publisher, xlinear_member_v2)

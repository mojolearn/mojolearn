# SPDX-License-Identifier: Apache-2.0
"""RUN pieces of the linear expansion door (bindings/x_linear_native_door.mojo):
the typed calls of the x_linear operations entry, the fit request words,
and the byte copies that assemble and slice its float32 words. Typed Mojo
values and addresses only, no Python import (tools/python_free_dirs.sh).
The door's UNPACK admits buffers and allocates the outputs; it releases the
GIL around each operations call exactly where it did before the split; its
PACK publishes the fitted attributes (lane doorsplit-linear, 2026-11-11).
Same operands in the same order, so the bits are unchanged."""
from std.memory import unsafe_memcpy
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from core.linear_entry_table_ext import (
    XL_OP_FIT, XL_OP_FIT_FOLD, XL_OP_DECISION, XL_OP_DECISION_CODES, XL_OP_DECISION_ROWS, XL_OP_SPEARMAN,
    XL_OP_HUBER_OUT, XL_OP_CLASS_PREP, XL_OP_BEST_ALPHA, XL_OP_FLAGS, XL_HAS_FIT_FOLD, XL_OP_BAYES_STD,
)
from core.solver_entry_table import solver_cd_fit_call, solver_cd_predict_call
from core.admitted_refusals import admitted_finite_refuse
from core.admitted_f32_plan import admitted_weights_run
from core.host_finite_scan import host_all_finite_f32
from core.metrics_entry_table import MR_R2, MR_R2_WEIGHTED
from core.entry_table import EntryRef
from core.base_helpers_calls import (
    base_assign_fold_i64, base_check_indices_i64, base_count_fold_hits_i64, base_mark_split_roles_f32,
    base_row_means_f64, base_reduce_stat, base_reduce_stat_int,
)

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]

# The x_linear fit routes (the `algo` word of an XL_OP_FIT request).
comptime ALGO_SGD = 1
comptime ALGO_GLM = 2
comptime ALGO_HUBER = 3
comptime ALGO_BAYES = 4
comptime ALGO_ARD = 5
comptime ALGO_LARS = 6
comptime ALGO_QUANTILE = 7
comptime ALGO_RIDGE = 8
comptime ALGO_ENETCV = 9
comptime ALGO_LOGCV = 10
comptime ALGO_ISOTONIC = 11
comptime ALGO_ISOTONIC_PREDICT = 12
comptime ALGO_RIDGE_KFOLD = 13
# The decision entries' link words.
comptime LINK_IDENTITY = 0
comptime LINK_EXP = 1
comptime LINK_SIGMOID = 2
# L-BFGS history length of the Huber route's work estimate.
comptime XL_LBFGS_M = 10


def xl_fit_fold_held(entry: NativeDescriptorCall) raises -> Bool:
    """Whether the binding behind `entry` fits a model-selection fold from
    rows resident on the device (XL_OP_FLAGS, XL_HAS_FIT_FOLD)."""
    var flags = native_descriptor_call(entry, XL_OP_FLAGS, List[Int](), List[Int](), List[Float64]())
    return (flags & XL_HAS_FIT_FOLD) != 0


def xl_run_fit_ints(header: List[Int], ip: List[Int]) -> List[Int]:
    """A fit request's Int words: its fixed header, then the route's
    integer hyperparameters (core/linear_entry_table_ext.mojo)."""
    var words = header.copy()
    for i in range(len(ip)):  # driver-loop: over the route's integer hyperparameters (model structure)
        words.append(ip[i])
    return words^


def xl_run_fit(entry: NativeDescriptorCall, algo: Int, x_addr: Int, x_size: Int, n: Int, d: Int, y_addr: Int,
               y_size: Int, ip: List[Int], fp: List[Float64], n_out: Int, n_fw: Int, n_iw: Int,
               out_addr: Int) raises -> Int:
    """One XL_OP_FIT call into the `n_out` float32 words at `out_addr`.
    `y_addr` addresses `y_size` 32-bit target words (the caller substitutes
    one zero word when there are none)."""
    var header: List[Int] = [algo, n, d, x_size, y_size, n_out, max(n_fw, 1), max(n_iw, 1), len(ip), len(fp)]
    return native_descriptor_call(entry, XL_OP_FIT, [x_addr, y_addr, out_addr], xl_run_fit_ints(header, ip), fp)


def xl_run_fit_fold(entry: NativeDescriptorCall, algo: Int, base_id: Int, base_rows: Int, index_id: Int, n: Int,
                    d: Int, y_addr: Int, y_size: Int, ip: List[Int], fp: List[Float64], n_out: Int, n_fw: Int,
                    n_iw: Int, out_addr: Int) raises -> Tuple[Int, Int]:
    """One XL_OP_FIT_FOLD call: the fit of a model-selection fold whose
    rows the entry gathers on the device out of the resident base. Returns
    the fold's (finite, range) status words (0 = clear) for
    core/admitted_refusals.mojo door_rows_refuse."""
    var words = List[Int](length=2, fill=0)
    var header: List[Int] = [algo, base_id, base_rows, index_id, n, d, n * d, y_size, n_out, max(n_fw, 1),
                             max(n_iw, 1), len(ip), len(fp)]
    _ = native_descriptor_call(entry, XL_OP_FIT_FOLD, [y_addr, out_addr, Int(words.unsafe_ptr())],
                               xl_run_fit_ints(header, ip), fp)
    var status = (words[0], words[1])
    _ = words^
    return status


def xl_run_status_refuse(word_addr: Int, est: String) raises:
    """Refuse a non-finite X a decision entry found in its status word
    (0: no deferred word)."""
    if word_addr != 0:
        admitted_finite_refuse(Int(_I32P(unsafe_from_address=word_addr)[0]), est)


def xl_run_word(addr: Int, index: Int) -> Float32:
    """The float32 word `index` of a result vector."""
    return _F32P(unsafe_from_address=addr)[index]


def xl_run_copy(dst: Int, src: Int, offset: Int, count: Int):
    """`count` float32 words of `src` from word `offset` into `dst`."""
    if count > 0:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=dst), src=_F32P(unsafe_from_address=src) + offset, count=count)


def xl_run_join(dst: Int, first: Int, first_count: Int, second: Int, second_count: Int):
    """`_concat_f32(a, b)`: a's then b's 32-bit words into `dst`."""
    var o = _F32P(unsafe_from_address=dst)
    if first_count > 0:
        unsafe_memcpy(dest=o, src=_F32P(unsafe_from_address=first), count=first_count)
    if second_count > 0:
        unsafe_memcpy(dest=o + first_count, src=_F32P(unsafe_from_address=second), count=second_count)


def xl_run_parts(dst: Int, addrs: List[Int], counts: List[Int]):
    """`_concat_f32_parts(parts)`: the parts' 32-bit words in order."""
    var o = _F32P(unsafe_from_address=dst)
    var at = 0
    for i in range(len(counts)):  # driver-loop: over the target vector's parts
        if counts[i] > 0:
            unsafe_memcpy(dest=o + at, src=_F32P(unsafe_from_address=addrs[i]), count=counts[i])
        at += counts[i]


def xl_parts_total(counts: List[Int]) -> Int:
    var total = 0
    for i in range(len(counts)):  # driver-loop: over the target vector's parts
        total += counts[i]
    return total


def xl_block_dims(coef_words: Int, intercept_words: Int) -> Tuple[Int, Int]:
    """(k, d) of `_coef_block`: k intercepts, d coefficients per class."""
    var k = intercept_words
    return (k, coef_words // max(k, 1))


def xl_run_block(dst: Int, coef: Int, intercept: Int, k: Int, d: Int):
    """`_coef_block(coef, intercept)`: [coef row | intercept] per class,
    k rows of d + 1 float32 words (byte copies)."""
    var o = _F32P(unsafe_from_address=dst)
    var cp = _F32P(unsafe_from_address=coef)
    var bp = _F32P(unsafe_from_address=intercept)
    for j in range(k):  # driver-loop: over the fitted classes' coefficient rows (model structure)
        unsafe_memcpy(dest=o + j * (d + 1), src=cp + j * d, count=d)
        unsafe_memcpy(dest=o + j * (d + 1) + d, src=bp + j, count=1)


def xl_seed_halves(lo: Int, hi: Int) -> Tuple[Int, Int]:
    """`_seed`: the low and high unsigned 32-bit halves of random_state's
    low 64 bits (UNPACK splits the Python integer) as signed Int32 words."""
    var half = 2147483648
    return (lo - 2 * half if lo >= half else lo, hi - 2 * half if hi >= half else hi)


def xl_lbfgs_work(p: Int) -> Int:
    """The Huber route's L-BFGS float work words for p parameters."""
    return 4 * p + 2 * XL_LBFGS_M * p + 2 * XL_LBFGS_M


def xl_sgd_class_floats(mut fp: List[Float64], class_weight: Int, n_classes: Int):
    """Append the SGD fit's class-weight parameters: (w1, w0) for two
    classes; the k positive weights then k ones for more."""
    var cwp = _F32P(unsafe_from_address=class_weight)
    if n_classes == 2:
        fp.append(Float64(cwp[1]))
        fp.append(Float64(cwp[0]))
    else:
        for c in range(n_classes):  # driver-loop: over the classes' positive weights (model structure)
            fp.append(Float64(cwp[c]))
        for c in range(n_classes):  # driver-loop: over the classes' negative weights (model structure)
            fp.append(1.0)


# ------------------------------------------------------------ cv splits


@fieldwise_init
struct XlSplitWords(Movable):
    """The int64 row indices of each cv split as addresses and counts (an
    address is 0 when its side is empty). UNPACK reads them off the
    split Arrays; the Arrays stay alive in the caller."""
    var train: List[Int]
    var train_sizes: List[Int]
    var test: List[Int]
    var test_sizes: List[Int]


def xl_cv_fold_ids_run(key: EntryRef, splits: XlSplitWords, n: Int, fold_ids: Int) raises -> Bool:
    """`_cv_fold_ids(splits, n)` on the base helper table: write each row's
    int32 fold into the n words at `fold_ids` (filled with -1 by the caller) and
    return True when the test sides partition the rows and every train side
    is the complement of its test side; False otherwise."""
    var covered = 0
    for k in range(len(splits.test)):  # driver-loop: over the cv splits
        var size = splits.test_sizes[k]
        if base_assign_fold_i64(key, splits.test[k] if size > 0 else 0, size, n, k, fold_ids) != 0:
            return False
        covered += size
    if covered != n:
        return False
    for k in range(len(splits.train)):  # driver-loop: over the cv splits
        var tr_size = splits.train_sizes[k]
        if tr_size != n - splits.test_sizes[k]:
            return False
        if tr_size > 0:
            if base_check_indices_i64(key, splits.train[k], tr_size, n) != 0:
                return False
            if base_count_fold_hits_i64(key, splits.train[k], tr_size, fold_ids, n, k) != 0:
                return False
    return True


def xl_cv_roles_limit(n: Int, name: String) raises:
    """Per-split roles are float32 row marks: n + 1 must stay exact."""
    if n + 1 > 16777216:
        raise Error("NotImplementedError: mojolearn " + name + ": cv splits that are not a partition are limited "
                    + "to 16,777,215 rows")


def xl_cv_roles_run(key: EntryRef, splits: XlSplitWords, n: Int, roles: Int, name: String) raises:
    """`_cv_roles_f32`: mark each split's train (1) and test (2) rows in its
    n float32 words of `roles` (x_linear/cv_roles.mojo), refusing an index
    outside the rows, a repeated row, or an empty side."""
    for k in range(len(splits.test)):  # driver-loop: over the cv splits
        for side in range(2):  # driver-loop: over the two sides of one split
            var size = splits.train_sizes[k] if side == 0 else splits.test_sizes[k]
            var rows = splits.train[k] if side == 0 else splits.test[k]
            var status = base_mark_split_roles_f32(key, rows if size > 0 else 0, size, n, k, side + 1, roles)
            if status == 1:
                raise Error("mojolearn " + name + ": a cv split holds an index outside [0, n_samples)")
            if status == 2:
                raise Error("mojolearn " + name + ": a cv split repeats a row, or puts a row on both sides")
        if splits.train_sizes[k] == 0 or splits.test_sizes[k] == 0:
            raise Error("mojolearn " + name + ": every cv split needs training and test rows")


# ------------------------------------------------- Lasso / ElasticNet (cd)
# RUN of the coordinate-descent pair's operations (UNPACK:
# bindings/x_linear_cd_unpack.mojo, PACK: bindings/x_linear_cd_pack.mojo).
# A request is addresses and plain words; the boundary keeps every
# addressed buffer alive and releases the GIL around the run.


struct XlCdFitRequest(Movable):
    """`cd_fit` of one design: `entry` is the solver operations entry of
    the estimator's route; `x` addresses n_rows x n_cols float32 words
    (C order when `row_major`, else F order), `y` n_rows targets, `coef`
    the n_cols result words, `info` one result word (the intercept),
    `sample_weight` n_rows weights or 0."""
    var entry: NativeDescriptorCall
    var x: Int
    var y: Int
    var coef: Int
    var info: Int
    var sample_weight: Int
    var n_rows: Int
    var n_cols: Int
    var fit_intercept: Bool
    var epochs: Int
    var shuffle: Bool
    var row_major: Bool
    var alpha: Float64
    var l1_ratio: Float64
    var tol: Float64

    def __init__(out self, entry: NativeDescriptorCall, x: Int, y: Int, coef: Int, info: Int, sample_weight: Int,
                 n_rows: Int, n_cols: Int, fit_intercept: Bool, epochs: Int, shuffle: Bool, row_major: Bool,
                 alpha: Float64, l1_ratio: Float64, tol: Float64):
        self.entry = entry
        self.x = x
        self.y = y
        self.coef = coef
        self.info = info
        self.sample_weight = sample_weight
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.fit_intercept = fit_intercept
        self.epochs = epochs
        self.shuffle = shuffle
        self.row_major = row_major
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.tol = tol


def xl_cd_fit_run(request: XlCdFitRequest) raises -> Int:
    """ElasticNet.fit / Lasso.fit: one `cd_fit` call; n_iter. The
    coefficients are in `request.coef`, the intercept in `request.info`."""
    return solver_cd_fit_call(request.entry, request.x, request.y, request.coef, request.info,
                              request.sample_weight, request.n_rows, request.n_cols, request.fit_intercept,
                              request.epochs, request.shuffle, request.row_major, request.alpha, request.l1_ratio,
                              request.tol)


struct XlCdPredictRequest(Movable):
    """`cd_predict`: `x` addresses n_rows x n_cols F-order float32 words,
    `coef` n_cols words, `dst` the n_rows result words."""
    var entry: NativeDescriptorCall
    var x: Int
    var coef: Int
    var dst: Int
    var n_rows: Int
    var n_cols: Int
    var intercept: Float64

    def __init__(out self, entry: NativeDescriptorCall, x: Int, coef: Int, dst: Int, n_rows: Int, n_cols: Int,
                 intercept: Float64):
        self.entry = entry
        self.x = x
        self.coef = coef
        self.dst = dst
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.intercept = intercept


def xl_cd_predict_run(request: XlCdPredictRequest) raises:
    """ElasticNet.predict / Lasso.predict: one `cd_predict` call into
    `request.dst`."""
    solver_cd_predict_call(request.entry, request.x, request.coef, request.dst, request.n_rows, request.n_cols,
                           request.intercept)


# ------------------------------------------------------------ decisions
# RUN of the inference operations (predict, decision_function,
# predict_proba, the one-class scores; UNPACK:
# bindings/x_linear_infer_unpack.mojo, PACK: bindings/x_linear_infer_pack.mojo).

# core/py2mojo_rows.mojo row modes of XL_OP_DECISION_ROWS.
comptime ROWS_SGD_PROBA = 2
comptime ROWS_LRCV_PROBA = 3


struct XlDecisionRequest(Movable):
    """One decision call of the x_linear operations entry. `operation` is
    XL_OP_DECISION (link(X @ W^T + b), float32 scores), XL_OP_DECISION_CODES
    (each row's int32 class code; `strict`, `below`, `above`) or
    XL_OP_DECISION_ROWS (the decision and the row link `rows_mode` in one
    call). `x` addresses rows x cols float32 words, `block` k rows of
    [coef | intercept], `dst` the result words; `status` is the deferred
    finite status word of X (0: X was tested at admission)."""
    var entry: NativeDescriptorCall
    var operation: Int
    var x: Int
    var block: Int
    var dst: Int
    var rows: Int
    var cols: Int
    var k: Int
    var link: Int
    var strict: Bool
    var below: Int
    var above: Int
    var rows_mode: Int
    var status: Int

    def __init__(out self, entry: NativeDescriptorCall, operation: Int, x: Int, block: Int, dst: Int, rows: Int,
                 cols: Int, k: Int, link: Int, strict: Bool, below: Int, above: Int, rows_mode: Int, status: Int):
        self.entry = entry
        self.operation = operation
        self.x = x
        self.block = block
        self.dst = dst
        self.rows = rows
        self.cols = cols
        self.k = k
        self.link = link
        self.strict = strict
        self.below = below
        self.above = above
        self.rows_mode = rows_mode
        self.status = status


def xl_decision_run(request: XlDecisionRequest) raises:
    """One decision call into `request.dst` (the same operands, in the same
    order, each operation always took)."""
    var addresses: List[Int] = [request.x, request.block, request.dst]
    var ints: List[Int] = [request.rows, request.cols, request.k]
    if request.operation == XL_OP_DECISION_CODES:
        ints.append(LINK_IDENTITY)
        ints.append(1 if request.strict else 0)
        ints.append(request.below)
        ints.append(request.above)
    elif request.operation == XL_OP_DECISION_ROWS:
        ints.append(request.link)
        ints.append(request.rows_mode)
    else:
        ints.append(request.link)
    ints.append(request.status)
    _ = native_descriptor_call(request.entry, request.operation, addresses, ints, List[Float64]())


@fieldwise_init
struct XlBayesStdRequest(Movable):
    """BayesianRidge / ARDRegression predict(return_std=True)'s standard
    deviations (XL_OP_BAYES_STD; lane mf-iterimp2): X as `rows` x `cols`
    float32 words at `x`, sigma_ (dk x dk float32) at `sigma`, X_offset_
    (cols float32) at `offset`, the dk kept columns' int32 indices at
    `kept` (0: every column, dk == cols), alpha_, the `rows` float32
    results at `dst`."""
    var entry: NativeDescriptorCall
    var x: Int
    var sigma: Int
    var offset: Int
    var kept: Int
    var dst: Int
    var rows: Int
    var cols: Int
    var dk: Int
    var alpha: Float64


def xl_bayes_std_run(request: XlBayesStdRequest) raises:
    """One XL_OP_BAYES_STD call into `request.dst`."""
    var addresses: List[Int] = [request.x, request.sigma, request.offset, request.kept, request.dst]
    var ints: List[Int] = [request.rows, request.cols, request.dk]
    var floats: List[Float64] = [request.alpha]
    _ = native_descriptor_call(request.entry, XL_OP_BAYES_STD, addresses, ints, floats)


def xl_ard_kept_run(lambdas: Int, d: Int, threshold: Float64, dst: Int) -> Int:
    """ARDRegression's kept columns (lambda_ < threshold_lambda in float32,
    the fit's own rule, x_linear/bayes.mojo `ard_update`) as int32 indices
    ascending at `dst` (d words of room). Returns their count."""
    var lam = _F32P(unsafe_from_address=lambdas)
    var out = _I32P(unsafe_from_address=dst)
    var thr = Float32(threshold)
    var dk = 0
    for j in range(d):  # driver-loop: over the fitted features (model structure, one compare each)
        if lam[j] < thr:
            out[dk] = Int32(j)
            dk += 1
    return dk


# ------------------------------------------------------------------ fits
# RUN of the fit operations: one typed request per XL_OP_FIT /
# XL_OP_FIT_FOLD call (UNPACK: bindings/x_linear_common_unpack.mojo
# `xl_fit_build` and each group's bindings/x_linear_<group>_unpack.mojo).


struct XlFitRequest(Movable):
    """One fit call of the x_linear operations entry: the route `algo`, X
    as `x_size` float32 words at `x` (n rows of d), the `y_size` 32-bit
    target words at `y`, the route's integer and float hyperparameters and
    the `n_out` float32 result words at `dst` (`n_fw` / `n_iw`: the float
    and int work words). `fold`: X is a model-selection fold resident on
    the device (`base_id`, `base_rows`, `index_id`; no host words)."""
    var entry: NativeDescriptorCall
    var algo: Int
    var x: Int
    var x_size: Int
    var n: Int
    var d: Int
    var y: Int
    var y_size: Int
    var ip: List[Int]
    var fp: List[Float64]
    var n_out: Int
    var n_fw: Int
    var n_iw: Int
    var dst: Int
    var fold: Bool
    var base_id: Int
    var base_rows: Int
    var index_id: Int

    def __init__(out self, entry: NativeDescriptorCall, algo: Int, x: Int, x_size: Int, n: Int, d: Int, y: Int,
                 y_size: Int, var ip: List[Int], var fp: List[Float64], n_out: Int, n_fw: Int, n_iw: Int, dst: Int,
                 fold: Bool, base_id: Int, base_rows: Int, index_id: Int):
        self.entry = entry
        self.algo = algo
        self.x = x
        self.x_size = x_size
        self.n = n
        self.d = d
        self.y = y
        self.y_size = y_size
        self.ip = ip^
        self.fp = fp^
        self.n_out = n_out
        self.n_fw = n_fw
        self.n_iw = n_iw
        self.dst = dst
        self.fold = fold
        self.base_id = base_id
        self.base_rows = base_rows
        self.index_id = index_id


def xl_fit_run(request: XlFitRequest) raises -> Tuple[Int, Int]:
    """One fit call into `request.dst`. Returns a device fold's (finite,
    range) status words, (0, 0) for a fit from host words."""
    if request.fold:
        return xl_run_fit_fold(request.entry, request.algo, request.base_id, request.base_rows, request.index_id,
                               request.n, request.d, request.y, request.y_size, request.ip, request.fp,
                               request.n_out, request.n_fw, request.n_iw, request.dst)
    _ = xl_run_fit(request.entry, request.algo, request.x, request.x_size, request.n, request.d, request.y,
                   request.y_size, request.ip, request.fp, request.n_out, request.n_fw, request.n_iw, request.dst)
    return (0, 0)


def xl_isotonic_fit_run(request: XlFitRequest, auto: Bool, targets: Int) raises -> Bool:
    """IsotonicRegression.fit: the direction, then the one fit call. With
    increasing='auto' the direction is the sign of Spearman's rho of the
    column at `request.x` and the `request.n` targets at `targets`, exact,
    on the binding (XL_OP_SPEARMAN); it is the fit's first Int word.
    Returns True for increasing."""
    if not auto:
        _ = xl_fit_run(request)
        return request.ip[0] != 0
    var columns: List[Int] = [request.x, targets]
    var count: List[Int] = [request.n]
    var sign = native_descriptor_call(request.entry, XL_OP_SPEARMAN, columns, count, List[Float64]())
    var ip = request.ip.copy()
    ip[0] = 1 if sign >= 0 else 0
    _ = xl_run_fit(request.entry, request.algo, request.x, request.x_size, request.n, request.d, request.y,
                   request.y_size, ip, request.fp, request.n_out, request.n_fw, request.n_iw, request.dst)
    return sign >= 0


def xl_huber_fit_run(request: XlFitRequest, targets: Int, flags: Int, link: Int, epsilon: Float64) raises:
    """HuberRegressor.fit: the fit, then the outlier flag of each training
    row (one byte at `flags`; |y - prediction| > scale * epsilon) from the
    X the fit read, in one device session (XL_OP_HUBER_OUT). The fit's
    first d + 1 result words are [coef | intercept], the one-row
    coefficient block the flags call reads as it is; `scale` is the word
    after them (`float(scale_) * epsilon`, one binary64 product)."""
    _ = xl_fit_run(request)
    var d = request.d
    var thr = Float64(xl_run_word(request.dst, d + 1)) * epsilon
    var addresses: List[Int] = [request.x, request.dst, targets, flags]
    var ints: List[Int] = [request.n, d, link]
    var floats: List[Float64] = [thr]
    _ = native_descriptor_call(request.entry, XL_OP_HUBER_OUT, addresses, ints, floats)


# ---------------------------------------------------------- class weights
# RUN of the class-weight step of the classifier fits (SGDClassifier,
# Perceptron, PassiveAggressiveClassifier, RidgeClassifier,
# LogisticRegressionCV; UNPACK: bindings/x_linear_common_unpack.mojo
# `xl_class_weights_unpack`). It ran inside UNPACK, as its own GIL release,
# until 2026-11-21 (lane dw-xlinear): its results are words of the fit
# request (SGD: float parameters; RidgeClassifier and LogisticRegressionCV:
# row weights among the target words), so the fit RUN makes the call first
# and finishes its own request.


struct XlClassPrepRequest(Movable):
    """One XL_OP_CLASS_PREP call: the n int32 class codes at `codes`, the n
    checked sample weights at `sample_weight` (0: none), the k class
    weights at `class_weight` (read when not `balanced`, written when
    `balanced`), the n row weights written at `rows` (0: not wanted).
    `weighted`: 'balanced' counts the sample weights. `active` False: the
    fit has no class-weight step and RUN makes no call."""
    var active: Bool
    var codes: Int
    var sample_weight: Int
    var class_weight: Int
    var rows: Int
    var n: Int
    var k: Int
    var balanced: Bool
    var weighted: Bool

    def __init__(out self, active: Bool, codes: Int, sample_weight: Int, class_weight: Int, rows: Int, n: Int,
                 k: Int, balanced: Bool, weighted: Bool):
        self.active = active
        self.codes = codes
        self.sample_weight = sample_weight
        self.class_weight = class_weight
        self.rows = rows
        self.n = n
        self.k = k
        self.balanced = balanced
        self.weighted = weighted


def xl_class_prep_none() -> XlClassPrepRequest:
    """The request of a fit without a class-weight step."""
    return XlClassPrepRequest(False, 0, 0, 0, 0, 0, 0, False, False)


def xl_class_prep_run(entry: NativeDescriptorCall, prep: XlClassPrepRequest) raises -> Int:
    """`_class_prep` on the binding (`x_linear_class_prep`): the class
    weights and the row weights land in the request's words; returns the
    largest unweighted class count."""
    var addresses: List[Int] = [prep.codes, prep.sample_weight, prep.class_weight, prep.rows]
    var ints: List[Int] = [prep.n, prep.k, 1 if prep.balanced else 0,
                           1 if (prep.balanced and prep.weighted and prep.sample_weight != 0) else 0]
    return native_descriptor_call(entry, XL_OP_CLASS_PREP, addresses, ints, List[Float64]())


def xl_row_weights_check(address: Int, n: Int) raises:
    """The weight admission's refusals (bindings/admitted_input_unpack.mojo
    `admit_weights`: finiteness, sign, all zero) on n row weights the
    class-weight step wrote."""
    if not host_all_finite_f32(_F32P(unsafe_from_address=address), n):
        raise Error("Input sample_weight contains NaN or infinity.")
    admitted_weights_run(address, address, n, False)


def xl_class_fit_run(request: XlFitRequest, prep: XlClassPrepRequest) raises -> Tuple[Int, Int]:
    """A classifier fit: the class-weight step when the fit has one, then
    the fit call. ALGO_SGD takes the k class weights as its last float
    parameters (`xl_sgd_class_floats`); ALGO_RIDGE reads the n row weights
    the step wrote behind the class codes (`prep.rows` addresses the second
    n target words), checked as the weight admission checks them. Returns
    a device fold's (finite, range) status words, (0, 0) otherwise."""
    if not prep.active:
        return xl_fit_run(request)
    _ = xl_class_prep_run(request.entry, prep)
    if request.algo == ALGO_SGD:
        var fp = request.fp.copy()
        xl_sgd_class_floats(fp, prep.class_weight, prep.k)
        _ = xl_run_fit(request.entry, request.algo, request.x, request.x_size, request.n, request.d, request.y,
                       request.y_size, request.ip, fp, request.n_out, request.n_fw, request.n_iw, request.dst)
        return (0, 0)
    xl_row_weights_check(prep.rows, prep.n)
    return xl_fit_run(request)


def xl_logcv_fit_run(request: XlFitRequest, prep: XlClassPrepRequest, stratified: Bool, folds: Int) raises:
    """LogisticRegressionCV.fit: the class-weight step (the row weights land
    among the fit's target words when the fit is weighted), the stratified
    fold count against the largest class (`_check_stratified_folds`), then
    the one fit call."""
    var largest = xl_class_prep_run(request.entry, prep)
    if stratified and (folds < 2 or folds > largest):
        raise Error("mojolearn: cv must be None or an int in [2, the largest class size]")
    _ = xl_fit_run(request)


# ------------------------------------------------------------------ RidgeCV
# RUN of RidgeCV.fit (UNPACK: bindings/x_linear_cv_unpack.mojo, PACK:
# bindings/x_linear_cv_pack.mojo). One request serves its three paths:
# cv=None (one fit over the alpha grid), the k-fold path (the fold scores on
# the binding, the best alpha, the refit) and the split path (per split and
# alpha a fit, the prediction of the test rows and their R^2; then the mean
# per alpha, the first best, the refit).
#
# The split path's alpha loop is serial by nature: each fit, prediction and
# R^2 is one launch on the estimator's one device (each parallel inside),
# and the test R^2 of an alpha reads that alpha's fit. The splits are driven
# from the door, because a split's rows are gathered by the base binding,
# which has no typed entry for the gather yet.


struct XlRidgeCvRequest(Movable):
    """RidgeCV.fit: X as n rows of d float32 words at `x`; the n targets at
    `y`, followed by the n checked sample weights when `weighted`; the alpha
    grid; `dst`: the result words of the final fit. `folds` and `scores`
    (A float32 words): the k-fold path. `scores` ((A, k) float64 words, row
    per alpha), `means` (A float64 words), `k` (the cv splits), `r2` (the
    R^2 entry of the metrics image of the estimator's device,
    core/metrics_entry_table.mojo) and `helpers` (the base helper table):
    the split path."""
    var entry: NativeDescriptorCall
    var x: Int
    var n: Int
    var d: Int
    var y: Int
    var weighted: Bool
    var fit_intercept: Bool
    var alphas: List[Float64]
    var dst: Int
    var folds: Int
    var scores: Int
    var means: Int
    var k: Int
    var r2: NativeDescriptorCall
    var helpers: EntryRef

    def __init__(out self, entry: NativeDescriptorCall, x: Int, n: Int, d: Int, y: Int, weighted: Bool,
                 fit_intercept: Bool, var alphas: List[Float64], dst: Int, folds: Int, scores: Int, means: Int,
                 k: Int, r2: NativeDescriptorCall, helpers: EntryRef):
        self.entry = entry
        self.x = x
        self.n = n
        self.d = d
        self.y = y
        self.weighted = weighted
        self.fit_intercept = fit_intercept
        self.alphas = alphas^
        self.dst = dst
        self.folds = folds
        self.scores = scores
        self.means = means
        self.k = k
        self.r2 = r2
        self.helpers = helpers


def xl_ridge_out(d: Int, grid: Int) -> Int:
    """The result words of one ridge fit over `grid` alphas: coef (d),
    intercept, the chosen alpha, its score, the grid's scores, status."""
    return d + 1 + 2 + grid + 1


def xl_ridge_singular(dst: Int, d: Int, grid: Int) -> Bool:
    """Status 2: X'X + alpha I does not factor even in float-float (the
    alpha is the float32 word d + 1 of the result)."""
    return xl_run_word(dst, d + 1 + 2 + grid) == 2.0


def _xl_ridge_run(entry: NativeDescriptorCall, x: Int, n: Int, d: Int, y: Int, weighted: Bool,
                  fit_intercept: Bool, alphas: List[Float64], dst: Int) raises:
    """`_ridge_run(est, a, n, d, Y, 1, alphas, sample_weight)`: one target,
    the alphas' Gram solves, into the `xl_ridge_out(d, len(alphas))` words
    at `dst`."""
    var grid = len(alphas)
    var ip: List[Int] = [1, 1 if fit_intercept else 0, grid, 1 if weighted else 0]
    _ = xl_run_fit(entry, ALGO_RIDGE, x, n * d, n, d, y, 2 * n if weighted else n, ip, alphas,
                   xl_ridge_out(d, grid), 3 * d * d + 3 * d + 1 + d + 2 * n, 1, dst)


def xl_ridge_cv_plain_run(request: XlRidgeCvRequest) raises:
    """cv=None: one fit over the alpha grid into `request.dst`."""
    _xl_ridge_run(request.entry, request.x, request.n, request.d, request.y, request.weighted,
                  request.fit_intercept, request.alphas, request.dst)


def xl_ridge_cv_kfold_run(request: XlRidgeCvRequest) raises -> Int:
    """`_fit_kfold`: the k-fold scores of every alpha on the binding
    (ALGO_RIDGE_KFOLD) into `request.scores`, the first best finite score
    (XL_OP_BEST_ALPHA), then the fit of all rows at that alpha into
    `request.dst`. Returns the best alpha's index."""
    var grid = len(request.alphas)
    var n = request.n
    var d = request.d
    var ip: List[Int] = [request.folds, 1 if request.fit_intercept else 0, grid]
    _ = xl_run_fit(request.entry, ALGO_RIDGE_KFOLD, request.x, n * d, n, d, request.y, n, ip, request.alphas, grid,
                   2 * d * d + 3 * d + 1 + grid * (d + 2) + grid * (n // request.folds + 1), 1, request.scores)
    var addresses: List[Int] = [request.scores]
    var counts: List[Int] = [grid]
    var best = native_descriptor_call(request.entry, XL_OP_BEST_ALPHA, addresses, counts, List[Float64]())
    if best < 0:
        raise Error("mojolearn RidgeCV: X'X + alpha I is singular in float32 for every alpha "
                    + "(rescale X, or use larger alphas)")
    var one: List[Float64] = [request.alphas[best]]
    _xl_ridge_run(request.entry, request.x, n, d, request.y, False, request.fit_intercept, one, request.dst)
    return best


def xl_r2_run(r2: NativeDescriptorCall, helpers: EntryRef, y_true: Int, y_pred: Int, weights: Int, n: Int,
              first: Bool) raises -> Float64:
    """`metrics.r2_score(y_true, y_pred, sample_weight=weights)` of n
    float32 words each (`weights` 0: unweighted) as one typed call of the
    metrics image's R^2 entry (core/metrics_entry_table.mojo MR_R2 /
    MR_R2_WEIGHTED: the bodies its Python `r2_score` /
    `r2_score_weighted` entries run), with the refusals the metrics door
    made: NaN or infinity in either vector, weights without a positive
    total. `first` False: `y_true` and `weights` were checked by an earlier
    call over the same words. The weights are rows of weights the fit
    already admitted, so they are finite and not negative."""
    if first and not host_all_finite_f32(_F32P(unsafe_from_address=y_true), n):
        raise Error("mojolearn metrics: y_true contains NaN or infinity; refused "
                    + "here (metrics/README.md HAND-OFF ask 3) rather than letting it "
                    + "reach a kernel whose answer for it is a washed sentinel")
    if not host_all_finite_f32(_F32P(unsafe_from_address=y_pred), n):
        raise Error("mojolearn metrics: y_pred contains NaN or infinity; refused "
                    + "here (metrics/README.md HAND-OFF ask 3) rather than letting it "
                    + "reach a kernel whose answer for it is a washed sentinel")
    var value = List[Float64](length=1, fill=0.0)
    var words: List[Int] = [n]
    if weights != 0:
        if first and not (base_reduce_stat(helpers, weights, 0, n, 1).f > 0):
            raise Error("mojolearn r2_score: sample_weight must have positive total weight")
        var weighted: List[Int] = [y_true, y_pred, weights, Int(value.unsafe_ptr())]
        _ = native_descriptor_call(r2, MR_R2_WEIGHTED, weighted, words, List[Float64]())
    else:
        var plain: List[Int] = [y_true, y_pred, Int(value.unsafe_ptr())]
        _ = native_descriptor_call(r2, MR_R2, plain, words, List[Float64]())
    var result = value[0]
    _ = value^
    return result


struct XlRidgeFoldRequest(Movable):
    """One cv split of RidgeCV's split path, its rows already gathered:
    `x_train` (n_train rows of d words), `y_train` (the n_train targets,
    followed by the n_train weights when the fit is weighted), `x_test`,
    `y_test` and `w_test` (0: unweighted) of n_test rows; scratch words the
    boundary owns: `fit` (`xl_ridge_out(d, 1)` words), `pred` (n_test) and
    the int32 finite `status` word of the prediction; `column`: this
    split's column of the score table."""
    var x_train: Int
    var n_train: Int
    var y_train: Int
    var x_test: Int
    var n_test: Int
    var y_test: Int
    var w_test: Int
    var fit: Int
    var pred: Int
    var status: Int
    var column: Int

    def __init__(out self, x_train: Int, n_train: Int, y_train: Int, x_test: Int, n_test: Int, y_test: Int,
                 w_test: Int, fit: Int, pred: Int, status: Int, column: Int):
        self.x_train = x_train
        self.n_train = n_train
        self.y_train = y_train
        self.x_test = x_test
        self.n_test = n_test
        self.y_test = y_test
        self.w_test = w_test
        self.fit = fit
        self.pred = pred
        self.status = status
        self.column = column


def xl_ridge_cv_fold_run(request: XlRidgeCvRequest, fold: XlRidgeFoldRequest, est: String) raises -> Int:
    """One split: per alpha the ridge fit of the training rows, the
    prediction of the test rows from the fit's own [coef | intercept] words
    (XL_OP_DECISION, identity link) and its R^2 against the test targets,
    written to (alpha, `fold.column`) of the score table. Returns -1, or
    the index of the first alpha whose system does not factor even in
    float-float (its result words are in `fold.fit`; the boundary refuses
    in the fit's words)."""
    var d = request.d
    var words = xl_ridge_out(d, 1)
    var fit = _F32P(unsafe_from_address=fold.fit)
    var table = _F64P(unsafe_from_address=request.scores)
    if request.weighted:
        # the weight admission's all-zero refusal, once for the split's rows
        var w_train = fold.y_train + 4 * fold.n_train
        admitted_weights_run(w_train, w_train, fold.n_train, False)
    for i in range(len(request.alphas)):  # driver-loop: over the alpha grid (serial by nature: see the banner)
        for j in range(words):  # small-loop(the d + 5 result words of one fit)
            fit[j] = Float32(0)
        var one: List[Float64] = [request.alphas[i]]
        _xl_ridge_run(request.entry, fold.x_train, fold.n_train, d, fold.y_train, request.weighted,
                      request.fit_intercept, one, fold.fit)
        if xl_ridge_singular(fold.fit, d, 1):
            return i
        _I32P(unsafe_from_address=fold.status)[0] = Int32(0)
        var addresses: List[Int] = [fold.x_test, fold.fit, fold.pred]
        var ints: List[Int] = [fold.n_test, d, 1, LINK_IDENTITY, fold.status]
        _ = native_descriptor_call(request.entry, XL_OP_DECISION, addresses, ints, List[Float64]())
        xl_run_status_refuse(fold.status, est)
        table[i * request.k + fold.column] = xl_r2_run(request.r2, request.helpers, fold.y_test, fold.pred,
                                                       fold.w_test, fold.n_test, i == 0)
    return -1


def xl_ridge_cv_choose_run(request: XlRidgeCvRequest, undefined: Bool) raises -> Int:
    """The split path's choice and refit: each alpha's mean score over the
    splits (`row_means_f64`: fsum / k) into `request.means`, the first best
    (`reduce_stat` argmax), then the fit of all rows at that alpha into
    `request.dst`. `undefined` (a split with fewer than two test rows):
    every mean is NaN and the first alpha wins. Returns its index."""
    var grid = len(request.alphas)
    var best = 0
    if not undefined:
        if grid > 0 and request.k > 0:
            if base_row_means_f64(request.helpers, request.scores, grid, request.k, request.means) != 0:
                raise Error("-inf + inf in fsum")
        best = base_reduce_stat_int(request.helpers, request.means, 1, grid, 3)
    var one: List[Float64] = [request.alphas[best]]
    _xl_ridge_run(request.entry, request.x, request.n, request.d, request.y, request.weighted,
                  request.fit_intercept, one, request.dst)
    return best

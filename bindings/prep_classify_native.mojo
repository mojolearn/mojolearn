# SPDX-License-Identifier: Apache-2.0
# The native owner of the prep lane's classifiers: GaussianNB, MultinomialNB,
# BernoulliNB, ComplementNB, CategoricalNB, LinearDiscriminantAnalysis and
# QuadraticDiscriminantAnalysis. Each public fit, partial_fit, predict,
# predict_proba, predict_log_proba, predict_joint_log_proba, score,
# decision_function, transform and fit_transform is one call into
# prep_classify_c, which checks the parameters, admits X, encodes the labels,
# builds the x_prep program (the stages, operands and order the Python
# `_Prog` built, bindings/prep_classify_prog.mojo), runs it through the shared
# arena runner and stores the fitted attributes. score predicts and scores in
# the same call; LDA's and QDA's two-class decision function is staged in the
# scoring program instead of a second program over a downloaded block (the same
# matmul stage over the same words).
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem
from bindings.py_typed_call import py_call_typed
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_address, classical_wrap_array, classical_host_f32, classical_typed
from bindings.classical_validate import classical_view
from bindings.prep_lda_admission import (
    lda_native_fields, lda_native_labels, lda_native_context, lda_native_initialize,
    lda_native_get_params, lda_native_set_params, lda_require_fitted,
)
from bindings.classifier_public_lifecycle import classifier_label_list, classifier_after_fit
from bindings.classical_metadata import classical_is_fitted
from bindings.scaler_lifecycle_native import _public_result
from bindings.classical_labels import classical_encode_labels, classical_decode_labels, classical_accuracy, labels_check_sample_weight
from bindings.x_prep_native_inputs import _values
from bindings.arena_cache_release import arena_cache_release_operation
from bindings.prep_classify_prog import ClsProg, NONE, f32_array, i32_array, f32_input
from bindings.prep_classify_common import (
    XIn, CsrIn, PartialCodes, cls_builtins, cls_name, cls_repr, cls_str, cls_kind, cls_mode, cls_entry,
    idn_int, classical_shared, blocked, check_fitted, check_width, cls_x2d, is_sparse, csr_input, csr_dense_x,
    float_list, given_priors, prior_sum, is_real, cls_float, col_stats, cs, class_stats, cls_stats, qda_cov, gram,
    nb_onepass, nb_counts, encode_y, partial_codes, stage_partial_codes, partial_refuse,
    IDN_NB_CSR, XB,
)

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime I64P = MutPointer[Int64, MutUntrackedOrigin]

# x_prep stage codes (_expansion_prep._OPS).
comptime OP_BINARIZE = 12
comptime OP_MATMUL = 13
comptime OP_ROW_SOFTMAX = 14
comptime OP_ROW_ARGMAX = 15
comptime OP_CENTER_ROWS = 17
comptime OP_EIGH = 18
comptime OP_GNB_EPS = 27
comptime OP_GNB_PARAMS = 28
comptime OP_GNB_JLL = 29
comptime OP_CLASS_LOG_PRIOR = 30
comptime OP_MNB_PARAMS = 31
comptime OP_BNB_PARAMS = 32
comptime OP_CNB_PARAMS = 33
comptime OP_CAT_JLL = 35
comptime OP_LDA_PREP = 36
comptime OP_LDA_W = 37
comptime OP_LDA_STAGE2 = 38
comptime OP_LDA_STAGE3 = 39
comptime OP_QDA_PREP = 41
comptime OP_QDA_DEC = 42
comptime OP_GATHER_COLS = 61
comptime OP_LOG = 72
comptime OP_DA_SHRINK = 81
comptime OP_DA_POOL = 82
comptime OP_SYM_FN = 83
comptime OP_DA_INTERCEPT = 84
comptime OP_EVR = 85
comptime OP_ADD_ARRAYS = 90
comptime OP_GNB_MERGE = 91
comptime OP_CAT_COUNTS = 92
comptime OP_CAT_FLP = 93
comptime OP_CAT_HPART = 133
comptime OP_CAT_HFOLD = 134
comptime OP_COLBLOCK = 141
comptime OP_CENTERED_MATMUL = 178
comptime OP_P2M_CCOUNT = 200
comptime OP_P2M_CSCAN = 201
comptime OP_P2M_CSTART = 202
comptime OP_P2M_CWRITE = 203
comptime OP_P2M_RGATHER = 204
comptime OP_Q64_SOFTMAX = 213
# CategoricalNB's block histograms hold at most this many words (_CAT_HIST_WORDS).
comptime CAT_HIST_WORDS = 67108864
# `_p2m_chunks`' chunk cap (_II_CH).
comptime II_CH = 1024
# What a scoring program reads back.
comptime W_PREDICT = 1
comptime W_PROBA = 2
comptime W_LOG = 4


def _put_attr(mut pr: ClsProg, fields: PythonObject, name: String, context: PythonObject) raises -> Int:
    return pr.put(f32_input(fields[PythonObject(name)], context))


def _n_classes(fields: PythonObject) raises -> Int:
    return len(fields["classes_"])


def _zeros(count: Int) -> List[Float64]:
    return List[Float64](length=count, fill=0.0)


def _ones(count: Int) -> List[Float64]:
    return List[Float64](length=count, fill=1.0)


def _pm_one() -> List[Float64]:
    var w = List[Float64]()
    w.append(-1.0)
    w.append(1.0)
    return w^


def _one() -> List[Float64]:
    var w = List[Float64]()
    w.append(1.0)
    return w^


def _weights(sample_weight: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`check_sample_weight(sample_weight, n)` as a float32 Array (None when
    not given): the words `_nb_weights` put."""
    if sample_weight is None:
        return PythonObject(None)
    var m = labels_check_sample_weight(sample_weight, n, context)
    var w = f32_array(n, context)
    memcpy(dest=F32P(unsafe_from_address=Int(py=classical_address(w))), src=F32P(unsafe_from_address=m.addr), count=n)
    _ = w  # keep the Python object alive through the native call above
    _ = m  # keep the Python object alive through the native call above
    _ = m^
    return w


def _put_weights(mut pr: ClsProg, w: PythonObject) raises -> Int:
    if w is None:
        return NONE
    return pr.put(w)


def _copy_block(mut pr: ClsProg, src: Int, rows: Int, cols: Int, context: PythonObject) raises -> Int:
    """`_copy_block`: a bit-for-bit copy (gather_cols over every column)."""
    var out = pr.alloc(rows * cols)
    var index = List[Float64]()
    for j in range(cols):  # driver-loop: over the feature indices (model structure)
        index.append(Float64(j))
    var io = pr.put_list(index, context)
    pr.stage(OP_GATHER_COLS, rows * cols, src, rows, cols, io, cols, out)
    return out


def _binary_difference(mut pr: ClsProg, src: Int, rows: Int, K: Int, d_cols: Int, out_buf: Int, context: PythonObject) raises:
    """`_binary_difference`: the second minus the first column (or row)."""
    var w = pr.put_list(_pm_one(), context)
    if rows == 1:
        pr.stage(OP_MATMUL, d_cols, w, 0, 1, src, d_cols, 1, out_buf, d_cols, 2, NONE, NONE)
    else:
        pr.stage(OP_MATMUL, rows, src, K, 1, w, 1, 0, out_buf, 1, 2, NONE, NONE)


def _set(fields: PythonObject, name: String, value: PythonObject) raises:
    py_setitem(fields, PythonObject(name), value)


# ---------------------------------------------------------------- scoring


@fieldwise_init
struct Offs(Copyable, Movable):
    var n: Int
    var d: Int
    var K: Int
    var jll: Int
    var lp: Int
    var pp: Int
    var am: Int
    var p64: Bool


def _jll_stages(mut pr: ClsProg, kind: String, fields: PythonObject, xo_in: Int, n: Int, d: Int, out_buf: Int,
                context: PythonObject) raises:
    """The subclass's `_jll_stages`: its joint log likelihood stages."""
    var K = _n_classes(fields)
    var xo = xo_in
    if kind == "gnb":
        var th = _put_attr(pr, fields, "theta_", context)
        var va = _put_attr(pr, fields, "var_", context)
        var co = _put_attr(pr, fields, "_const", context)
        pr.stage(OP_GNB_JLL, n * K, xo, n, d, th, va, co, K, out_buf)
    elif kind == "mnb":
        var w = _put_attr(pr, fields, "feature_log_prob_", context)
        var b = _put_attr(pr, fields, "class_log_prior_", context)
        pr.stage(OP_MATMUL, n * K, xo, d, 1, w, 1, d, out_buf, K, d, b, NONE)
    elif kind == "bnb":
        if fields["binarize"] is not None:
            var thr = pr.put_scalar(cls_float(fields["binarize"]), context)
            var xb = pr.work(n * d)
            pr.stage(OP_BINARIZE, n * d, xo, n * d, thr, xb)
            xo = xb
        var w = _put_attr(pr, fields, "_w", context)
        var b = _put_attr(pr, fields, "_bias", context)
        pr.stage(OP_MATMUL, n * K, xo, d, 1, w, 1, d, out_buf, K, d, b, NONE)
    elif kind == "cnb":
        var w = _put_attr(pr, fields, "feature_log_prob_", context)
        var b = _put_attr(pr, fields, "class_log_prior_", context) if K == 1 else NONE
        pr.stage(OP_MATMUL, n * K, xo, d, 1, w, 1, d, out_buf, K, d, b, NONE)
    elif kind == "cat":
        var fo = _put_attr(pr, fields, "_flp", context)
        var co = _put_attr(pr, fields, "class_log_prior_", context)
        pr.stage(OP_CAT_JLL, n * K, xo, n, d, fo, K, Int(py=fields["_cmax"]), co, out_buf)
    elif kind == "lda":
        var c = _put_attr(pr, fields, "_coef", context)
        var b = _put_attr(pr, fields, "_inter", context)
        pr.stage(OP_MATMUL, n * K, xo, d, 1, c, 1, d, out_buf, K, d, b, NONE)
    else:
        var m = _put_attr(pr, fields, "means_", context)
        var r = _put_attr(pr, fields, "_rot", context)
        var lc = _put_attr(pr, fields, "_logc", context)
        pr.stage(OP_QDA_DEC, n * K, xo, n, d, m, r, lc, K, out_buf)


def _proba64_on(mode: PythonObject, context: PythonObject) raises -> Bool:
    """`_proba64_on`: the binding exports x_prep_proba64."""
    return cls_entry(mode, "x_prep_proba64", context) is not None


def _tail(mut pr: ClsProg, fields: PythonObject, kind: String, mode: PythonObject, n: Int, d: Int, K: Int,
          jll: Int, want: Int, chk: Int, context: PythonObject, pair: Bool = False) raises -> Offs:
    """`_score_tail`: softmax and argmax stages, the run, the input refusals.
    `pair`: QuadraticDiscriminantAnalysis' two-class decision function, the
    second score minus the first (`_binary_difference`), staged over the
    scores in this program (its words at `am`)."""
    var lp = pr.alloc(n * K) if (want & W_LOG) != 0 else NONE
    var p64 = (want & W_PROBA) != 0 and _proba64_on(mode, context)
    var pp = pr.alloc((2 if p64 else 1) * n * K) if (want & W_PROBA) != 0 else NONE
    var am = pr.alloc(n) if (want & W_PREDICT) != 0 else NONE
    if p64:
        pr.stage(OP_Q64_SOFTMAX, n, jll, n, K, pp)
        if lp != NONE:
            pr.stage(OP_ROW_SOFTMAX, n, jll, n, K, lp, NONE)
    elif lp != NONE or pp != NONE:
        pr.stage(OP_ROW_SOFTMAX, n, jll, n, K, lp, pp)
    if am != NONE:
        pr.stage(OP_ROW_ARGMAX, n, jll, n, K, am)
    if pair:
        am = pr.alloc(n)
        _binary_difference(pr, jll, n, K, 1, am, context)
    pr.run(mode, context)
    _score_refusals(pr, fields, kind, d, chk)
    return Offs(n, d, K, jll, lp, pp, am, p64)


def _score_refusals(pr: ClsProg, fields: PythonObject, kind: String, d: Int, st: Int) raises:
    """CategoricalNB's `_score_refusals`: a category index outside the fitted range."""
    if kind != "cat" or st == NONE:
        return
    var ncat = I64P(unsafe_from_address=Int(py=classical_address(fields["n_categories_"])))
    for j in range(d):  # driver-loop: over the feature columns' fitted category counts
        if pr.value(st + 3 * d + j) < 0:
            raise Error("IndexError: mojolearn: CategoricalNB got a category index outside the fitted range")
    for j in range(d):  # driver-loop: over the feature columns' fitted category counts
        if Int(pr.value(st + 4 * d + j)) >= Int(ncat[j]):
            raise Error("IndexError: mojolearn: CategoricalNB got a category index outside the fitted range")


def _csr_ready(kind: String, context: PythonObject) raises -> Bool:
    """`_DiscreteNB._nb_csr_ready` (MultinomialNB, ComplementNB)."""
    if kind != "mnb" and kind != "cnb":
        return False
    var mode = cls_mode(context)
    if String(mode) == "identical":
        return (idn_int(mode, context) & IDN_NB_CSR) != 0
    if String(mode) != "fast":
        return False
    return cls_entry(mode, "x_prep_nb_csr_fit", context) is not None


def _base_scores(mut pr: ClsProg, model: PythonObject, fields: PythonObject, kind: String, X: PythonObject,
                 want: Int, csr: CsrIn, has_csr: Bool, context: PythonObject, pair: Bool = False) raises -> Offs:
    """`_Classifier._scores` (`csr`: the IDENTICAL CSR scoring's fallback)."""
    var mode = fields["numeric_mode_"]
    var xo = csr_dense_x(pr, csr, mode, context) if has_csr else -1
    var n = csr.n
    var d = csr.d
    if xo == -1:
        var arr = cls_x2d(py_call(X.toarray) if has_csr else X, context)
        check_width(model, fields, arr.d)
        n = arr.n
        d = arr.d
        xo = pr.put(arr.value)
    var K = _n_classes(fields)
    var chk = NONE
    if kind == "cat":
        chk = pr.alloc(6 * d)
        col_stats(pr, xo, n, d, chk, False)
    # the joint log likelihood stays on the device unless it is the answer
    var jll = pr.work(n * K) if want != 0 else pr.alloc(n * K)
    _jll_stages(pr, kind, fields, xo, n, d, jll, context)
    return _tail(pr, fields, kind, mode, n, d, K, jll, want, chk, context, pair)


def _no_csr(context: PythonObject) raises -> CsrIn:
    return CsrIn(PythonObject(None), PythonObject(None), PythonObject(None), 0, 0, 0)


def _scores(mut pr: ClsProg, model: PythonObject, fields: PythonObject, kind: String, X: PythonObject, want: Int,
            context: PythonObject) raises -> Offs:
    """`_scores` with `_DiscreteNB._scores`' CSR route."""
    check_fitted(model, fields)
    if not (is_sparse(X) and _csr_ready(kind, context)):
        return _base_scores(pr, model, fields, kind, X, want, _no_csr(context), False, context)
    var csr = csr_input(X, context)
    var n = csr.n
    var d = csr.d
    if py_type(fields["n_features_in_"]) is not cls_builtins().int or d != Int(py=fields["n_features_in_"]):
        raise Error("mojolearn: X has " + String(d) + " features, but " + cls_name(model) + " was fitted with " + cls_str(fields["n_features_in_"]))
    var K = _n_classes(fields)
    var mode = fields["numeric_mode_"]
    var jll_csr = cls_entry(mode, "x_prep_nb_csr_jll", context)
    if jll_csr is None:
        raise Error("RuntimeError: the numeric mode's binding has no x_prep_nb_csr_jll")
    var jll_h = classical_typed(Python.tuple(n, K), "<f4", context["output_context"])
    var flp = f32_input(fields["feature_log_prob_"], context)
    var bias = PythonObject(0)
    var bias_keep = PythonObject(None)
    if kind == "mnb" or K == 1:
        bias_keep = f32_input(fields["class_log_prior_"], context)
        bias = classical_address(bias_keep)
        _ = bias_keep  # keep the Python object alive through the native call above
    var dims = Python.list(n, d, K, csr.nnz)
    if String(mode) == "identical":
        # the dense chain without its zero terms; a row whose columns do not
        # ascend strictly takes the dense program, its block built on the device
        var flag = i32_array(1, context)
        _ = py_call_typed(jll_csr, classical_address(csr.indptr), classical_address(csr.indices), classical_address(csr.data),
                          classical_address(flp), bias, dims, classical_address(jll_h), classical_address(flag))
        _ = flag  # keep the Python object alive through the native call above
        _ = flp  # keep the Python object alive through the native call above
        _ = jll_h  # keep the Python object alive through the native call above
        if Int(I32P(unsafe_from_address=Int(py=classical_address(flag)))[0]) != 0:
            return _base_scores(pr, model, fields, kind, X, want, csr, True, context)
        _ = flag  # keep the Python object alive through the native calls above
    else:
        _ = py_call_typed(jll_csr, classical_address(csr.indptr), classical_address(csr.indices), classical_address(csr.data),
                          classical_address(flp), bias, dims, classical_address(jll_h))
        _ = flp  # keep the Python object alive through the native call above
        _ = jll_h  # keep the Python object alive through the native call above
    _ = bias_keep^
    var z = pr.alloc(n * K)
    var jll = pr.alloc(n * K)
    var h = pr.put(jll_h)
    pr.stage(OP_ADD_ARRAYS, n * K, h, z, jll)
    return _tail(pr, fields, kind, mode, n, d, K, jll, want, NONE, context)


def _predict(model: PythonObject, fields: PythonObject, kind: String, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var pr = ClsProg()
    var o = _scores(pr, model, fields, kind, X, W_PREDICT, context)
    return classical_decode_labels(fields["classes_"], pr.get_i32(o.am, o.n, context), context)


def _predict_proba(model: PythonObject, fields: PythonObject, kind: String, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var pr = ClsProg()
    var o = _scores(pr, model, fields, kind, X, W_PROBA, context)
    if o.p64:
        return pr.get_f64(o.pp, o.n, o.K, context)
    return pr.get(o.pp, o.n, o.K, context)


def _predict_log_proba(model: PythonObject, fields: PythonObject, kind: String, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var pr = ClsProg()
    var o = _scores(pr, model, fields, kind, X, W_LOG, context)
    return pr.get(o.lp, o.n, o.K, context)


def _joint_log_proba(model: PythonObject, fields: PythonObject, kind: String, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var pr = ClsProg()
    var o = _scores(pr, model, fields, kind, X, 0, context)
    return pr.get(o.jll, o.n, o.K, context)


def _score(model: PythonObject, fields: PythonObject, kind: String, X: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_Classifier.score`: accuracy of the prediction, scored in the same call."""
    var prediction = _predict(model, fields, kind, X, context)
    return PythonObject(classical_accuracy(y, prediction, PythonObject(None), PythonObject(None), context))


def _decision_function(model: PythonObject, fields: PythonObject, kind: String, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """LinearDiscriminantAnalysis / QuadraticDiscriminantAnalysis decision_function."""
    check_fitted(model, fields)
    var K = _n_classes(fields)
    var mode = fields["numeric_mode_"]
    if kind == "lda" and K == 2:
        # The two-class answer is the coefficient difference's one column
        # (`_pair`, and the C56 route): one program over X; the K scores the
        # old route also computed were discarded.
        var arr = cls_x2d(X, context)
        check_width(model, fields, arr.d)
        var n = arr.n
        var d = arr.d
        var pl = ClsProg()
        var xo = pl.put(arr.value)
        var coef = _put_attr(pl, fields, "coef_", context)
        var inter = _put_attr(pl, fields, "intercept_", context)
        var out = pl.alloc(n)
        pl.stage(OP_MATMUL, n, xo, d, 1, coef, 1, 0, out, 1, d, inter, NONE)
        pl.run(mode, context)
        return pl.get(out, n, 0, context)
    var pr = ClsProg()
    if K != 2:
        var oj = _scores(pr, model, fields, kind, X, 0, context)
        return pr.get(oj.jll, oj.n, K, context)
    # QDA's two-class answer: the difference of the two scores, staged over
    # the scores where they are in the scoring program (the matmul stage and
    # the words the second program used to run over a downloaded block).
    var o = _base_scores(pr, model, fields, kind, X, 0, _no_csr(context), False, context, True)
    return pr.get(o.am, o.n, 0, context)


# ---------------------------------------------------------------- GaussianNB


def _gnb_refuse(fields: PythonObject, context: PythonObject) raises:
    var vs = fields["var_smoothing"]
    var b = cls_builtins()
    if py_type(vs) is b.bool or not is_real(vs, context) or not (cls_float(vs) >= 0):
        raise Error("The 'var_smoothing' parameter of GaussianNB must be a float in the range [0.0, inf). Got " + cls_repr(vs) + " instead.")


def _gnb_stats(mut pr: ClsProg, mode: PythonObject, wo: Int, xo: Int, n: Int, d: Int, yo: Int, K: Int, cnt: Int,
               theta: Int, var_: Int, eps: Int, vs: Int, context: PythonObject) raises:
    """`_gnb_stats`: epsilon and the class counts, means and variances."""
    var st = pr.alloc(6 * d)
    col_stats(pr, xo, n, d, st)
    pr.stage(OP_GNB_EPS, 1, st + 2 * d, d, eps, vs)
    class_stats(pr, wo, K * d, xo, n, d, yo, K, cnt, theta, var_, NONE, mode, context, True)


def _gnb_publish(pr: ClsProg, fields: PythonObject, mode: PythonObject, K: Int, d: Int, theta: Int, var_: Int,
                 cnt: Int, prior: Int, eps: Int, cst: Int, raw: Int, context: PythonObject) raises:
    _set(fields, "theta_", pr.get(theta, K, d, context))
    _set(fields, "var_", pr.get(var_, K, d, context))
    _set(fields, "class_count_", pr.get(cnt, K, 0, context))
    _set(fields, "class_prior_", pr.get(prior, K, 0, context))
    _set(fields, "epsilon_", PythonObject(pr.value(eps)))
    _set(fields, "_const", pr.get(cst, K, 0, context))
    _set(fields, "_raw_var", pr.get(raw, K, d, context))
    _set(fields, "numeric_mode_", mode)
    _set(fields, "n_features_in_", PythonObject(d))


def _gnb_fit(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    _gnb_refuse(fields, context)
    var arr = cls_x2d(args[0], context)
    var n = arr.n
    var d = arr.d
    var codes = encode_y(model, fields, args[1], n, context)
    var K = _n_classes(fields)
    var mode = cls_mode(context)
    var pr = ClsProg()
    var xo = pr.put(arr.value)
    var yo = pr.put_codes(codes)
    var vs = pr.put_scalar(cls_float(fields["var_smoothing"]), context)
    var eps = pr.alloc(1)
    var cnt = pr.alloc(K)
    var theta = pr.alloc(K * d)
    var var_ = pr.alloc(K * d)
    var prior = pr.alloc(K)
    var cst = pr.alloc(K)
    var wo = _put_weights(pr, _weights(args[2], n, context))
    _gnb_stats(pr, mode, wo, xo, n, d, yo, K, cnt, theta, var_, eps, vs, context)
    var raw = _copy_block(pr, var_, K, d, context)
    var given = NONE
    if fields["priors"] is not None:
        given = pr.put_list(given_priors(fields["priors"], K, "GaussianNB", True, context), context)
    elif wo != NONE:
        # class_count_ / class_count_.sum() over the weighted counts
        given = pr.alloc(K)
        var scratch = pr.alloc(d)
        pr.stage(OP_LDA_PREP, 1, cnt, theta, K, d, n, given, scratch, 2, cnt)
    pr.stage(OP_GNB_PARAMS, K, cnt, var_, K, d, n, eps, prior, cst, 1 if given != NONE else 0, given if given != NONE else 0)
    pr.run(mode, context)
    _gnb_publish(pr, fields, mode, K, d, theta, var_, cnt, prior, eps, cst, raw, context)
    return model


def _gnb_partial_fit(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    _gnb_refuse(fields, context)
    var arr = cls_x2d(args[0], context)
    var n = arr.n
    var d = arr.d
    var pc = partial_codes(model, fields, args[1], args[2], n, context)
    var K = _n_classes(fields)
    var mode = cls_mode(context)
    if not pc.first:
        check_width(model, fields, d)
        mode = fields["numeric_mode_"]
    var pr = ClsProg()
    var xo = pr.put(arr.value)
    var coded = stage_partial_codes(pr, pc, context)
    var yo = coded[0]
    var vs = pr.put_scalar(cls_float(fields["var_smoothing"]), context)
    var eps = pr.alloc(1)
    var bc = pr.alloc(K)
    var bm = pr.alloc(K * d)
    var bv = pr.alloc(K * d)
    var wo = _put_weights(pr, _weights(args[3], n, context))
    _gnb_stats(pr, mode, wo, xo, n, d, yo, K, bc, bm, bv, eps, vs, context)
    var oc = pr.put_list(_zeros(K), context) if pc.first else _put_attr(pr, fields, "class_count_", context)
    var om = pr.put_list(_zeros(K * d), context) if pc.first else _put_attr(pr, fields, "theta_", context)
    var ov = pr.put_list(_zeros(K * d), context) if pc.first else _put_attr(pr, fields, "_raw_var", context)
    var cnt = pr.alloc(K)
    var theta = pr.alloc(K * d)
    var raw = pr.alloc(K * d)
    pr.stage(OP_GNB_MERGE, K * d, oc, om, ov, bc, bm, bv, K, d, cnt, theta, raw)
    var var_ = _copy_block(pr, raw, K, d, context)
    var given = NONE
    if fields["priors"] is not None:
        given = pr.put_list(given_priors(fields["priors"], K, "GaussianNB", True, context), context)
    else:
        given = pr.alloc(K)
        var scratch = pr.alloc(d)
        pr.stage(OP_LDA_PREP, 1, cnt, theta, K, d, n, given, scratch, 2, cnt)
    var prior = pr.alloc(K)
    var cst = pr.alloc(K)
    pr.stage(OP_GNB_PARAMS, K, cnt, var_, K, d, 1, eps, prior, cst, 1, given)
    pr.run(mode, context)
    partial_refuse(pr, coded[1], fields, pc, context)
    _gnb_publish(pr, fields, mode, K, d, theta, var_, cnt, prior, eps, cst, raw, context)
    return model


# ---------------------------------------------------------------- discrete naive Bayes


@fieldwise_init
struct Counts(Copyable, Movable):
    """`_fit_counts`' program offsets (pr, mode, n, d, K, st, cnt, fc, clp)."""
    var mode: PythonObject
    var n: Int
    var d: Int
    var K: Int
    var st: Int
    var cnt: Int
    var fc: Int
    var clp: Int


def _check_alpha(model: PythonObject, fields: PythonObject, context: PythonObject) raises:
    var alpha = fields["alpha"]
    if not is_real(alpha, context) or not (cls_float(alpha) > 0):
        raise Error("NotImplementedError: mojolearn: " + cls_name(model) + " needs alpha > 0 (alpha = 0 makes log(0) terms)")


def _prior_stages(mut pr: ClsProg, model: PythonObject, fields: PythonObject, K: Int, cnt: Int, clp: Int,
                  context: PythonObject) raises:
    """`_DiscreteNB._prior_stages`."""
    var given = py_get(fields, "class_prior")
    if given is not None:
        var po = pr.put_list(given_priors(given, K, cls_name(model), False, context), context)
        pr.stage(OP_LOG, K, po, clp)
    elif Bool(py=fields["fit_prior"]):
        pr.stage(OP_CLASS_LOG_PRIOR, K, cnt, K, clp)
    else:
        var ones = pr.put_list(_ones(K), context)
        pr.stage(OP_CLASS_LOG_PRIOR, K, ones, K, clp)


def _needs_min(kind: String) -> Bool:
    return kind == "mnb" or kind == "cnb"


def _binarize(kind: String, fields: PythonObject) raises -> PythonObject:
    if kind != "bnb":
        return PythonObject(None)
    return fields["binarize"]


def _fit_counts(mut pr: ClsProg, model: PythonObject, fields: PythonObject, kind: String, X: PythonObject,
                y: PythonObject, binarize: PythonObject, sample_weight: PythonObject, csr: CsrIn, has_csr: Bool,
                context: PythonObject) raises -> Counts:
    """`_DiscreteNB._fit_counts`."""
    var mode = cls_mode(context)
    var xo = csr_dense_x(pr, csr, mode, context) if has_csr else -1
    var n = csr.n
    var d = csr.d
    if xo == -1:
        var arr = cls_x2d(py_call(X.toarray) if has_csr else X, context)
        n = arr.n
        d = arr.d
        xo = pr.put(arr.value)
    var codes = encode_y(model, fields, y, n, context)
    var K = _n_classes(fields)
    var wo = _put_weights(pr, _weights(sample_weight, n, context))
    if nb_onepass(mode, context):
        # lane idn-int-prep: one unit per (block, column) counts every class,
        # binarizes (BernoulliNB) and flags a negative word
        var yo = pr.put_codes(codes)
        var st = pr.alloc(6 * d)
        var cnt = pr.alloc(K)
        var fc = pr.alloc(K * d)
        var clp = pr.alloc(K)
        var thr = NONE if binarize is None else pr.put_scalar(cls_float(binarize), context)
        nb_counts(pr, wo, xo, n, d, yo, K, cnt, fc, thr, st + 3 * d if _needs_min(kind) else NONE)
        _prior_stages(pr, model, fields, K, cnt, clp, context)
        return Counts(mode, n, d, K, st, cnt, fc, clp)
    if binarize is not None:
        var thr = pr.put_scalar(cls_float(binarize), context)
        var xb = pr.work(n * d)
        pr.stage(OP_BINARIZE, n * d, xo, n * d, thr, xb)
        xo = xb
    var yo = pr.put_codes(codes)
    var st = pr.alloc(6 * d)
    var cnt = pr.alloc(K)
    var fc = pr.alloc(K * d)
    var clp = pr.alloc(K)
    col_stats(pr, xo, n, d, st, False)
    class_stats(pr, wo, K * d, xo, n, d, yo, K, cnt, NONE, NONE, fc, mode, context, False)
    _prior_stages(pr, model, fields, K, cnt, clp, context)
    return Counts(mode, n, d, K, st, cnt, fc, clp)


def _fit_counts_csr(mut pr: ClsProg, model: PythonObject, fields: PythonObject, csr: CsrIn, y: PythonObject,
                    mut done: Bool, context: PythonObject) raises -> Counts:
    """`_DiscreteNB._fit_counts_csr`; `done` False: IDENTICAL refused the
    rows (the caller runs the dense program)."""
    var n = csr.n
    var d = csr.d
    var codes = encode_y(model, fields, y, n, context)
    var K = _n_classes(fields)
    var mode = cls_mode(context)
    var fit_csr = cls_entry(mode, "x_prep_nb_csr_fit", context)
    if fit_csr is None:
        raise Error("RuntimeError: the numeric mode's binding has no x_prep_nb_csr_fit")
    var fc = classical_typed(Python.tuple(K, d), "<f4", context["output_context"])
    var cnt = classical_typed(Python.tuple(K), "<f4", context["output_context"])
    var flag = i32_array(1, context)
    _ = py_call_typed(fit_csr, classical_address(csr.indptr), classical_address(csr.indices), classical_address(csr.data),
                      classical_address(codes), Python.list(n, d, K, csr.nnz),
                      classical_address(fc), classical_address(cnt), classical_address(flag))
    _ = codes  # keep the Python object alive through the native call above
    _ = flag  # keep the Python object alive through the native call above
    _ = cnt  # keep the Python object alive through the native call above
    _ = fc  # keep the Python object alive through the native call above
    var level = Int(I32P(unsafe_from_address=Int(py=classical_address(flag)))[0])
    _ = flag  # keep the Python object alive through the native call above
    if String(mode) == "identical":
        # x_prep/fastnb_csr.mojo: 2 a negative value; 1 a value the CSR route
        # does not take (the caller runs the dense program)
        if level == 1:
            done = False
            return Counts(mode, n, d, K, 0, 0, 0, 0)
        _ = flag  # keep the Python object alive through the native calls above
        level = 1 if level == 2 else 0
        _ = flag  # keep the Python object alive through the native call above
    if level != 0:
        raise Error("mojolearn: Negative values in data passed to " + cls_name(model) + " (input X)")
    _ = flag  # keep the Python object alive through the native calls above
    var st = pr.alloc(6 * d)
    var z = pr.alloc(K * d)
    var cnt_o = pr.alloc(K)
    var fc_o = pr.alloc(K * d)
    var clp = pr.alloc(K)
    var co = pr.put(cnt)
    pr.stage(OP_ADD_ARRAYS, K, co, z, cnt_o)
    var fo = pr.put(fc)
    pr.stage(OP_ADD_ARRAYS, K * d, fo, z, fc_o)
    _prior_stages(pr, model, fields, K, cnt_o, clp, context)
    done = True
    return Counts(mode, n, d, K, st, cnt_o, fc_o, clp)


def _check_nonnegative(pr: ClsProg, st: Int, d: Int, who: String) raises:
    for j in range(d):  # driver-loop: over the per-column minimum statistics (feature count)
        if pr.value(st + 3 * d + j) < 0:
            raise Error("mojolearn: Negative values in data passed to " + who)


def _params(mut pr: ClsProg, model: PythonObject, fields: PythonObject, kind: String, c: Counts, neg: Int,
            pc: PartialCodes, context: PythonObject) raises -> PythonObject:
    """The subclass's `_params`: its parameter stage, the run, the refusals,
    the fitted attributes (`_finish_counts`)."""
    var K = c.K
    var d = c.d
    var a = pr.put_scalar(cls_float(fields["alpha"]), context)
    var flp = pr.alloc(K * d)
    var w = NONE
    var bias = NONE
    if kind == "mnb":
        pr.stage(OP_MNB_PARAMS, K, c.fc, K, d, a, flp)
    elif kind == "bnb":
        w = pr.alloc(K * d)
        bias = pr.alloc(K)
        pr.stage(OP_BNB_PARAMS, K, c.fc, c.cnt, K, d, a, c.clp, flp, w, bias)
    else:
        pr.stage(OP_CNB_PARAMS, K, c.fc, K, d, a, 1 if Bool(py=fields["norm"]) else 0, flp)
    pr.run(c.mode, context)
    partial_refuse(pr, neg, fields, pc, context)
    if kind == "mnb":
        _check_nonnegative(pr, c.st, d, "MultinomialNB (input X)")
    elif kind == "cnb":
        _check_nonnegative(pr, c.st, d, "ComplementNB (input X)")
    _set(fields, "class_count_", pr.get(c.cnt, K, 0, context))
    _set(fields, "feature_count_", pr.get(c.fc, K, d, context))
    _set(fields, "class_log_prior_", pr.get(c.clp, K, 0, context))
    _set(fields, "numeric_mode_", c.mode)
    _set(fields, "n_features_in_", PythonObject(d))
    _set(fields, "feature_log_prob_", pr.get(flp, K, d, context))
    if kind == "bnb":
        _set(fields, "_w", pr.get(w, K, d, context))
        _set(fields, "_bias", pr.get(bias, K, 0, context))
    return model


def _no_partial() -> PartialCodes:
    return PartialCodes(False, False, PythonObject(None), PythonObject(None), 0, 0, PythonObject(None), PythonObject(None))


def _dnb_fit(model: PythonObject, fields: PythonObject, kind: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_DiscreteNB.fit`."""
    _check_alpha(model, fields, context)
    var X = args[0]
    var y = args[1]
    var sample_weight = args[2]
    var binarize = _binarize(kind, fields)
    if sample_weight is None and is_sparse(X) and _csr_ready(kind, context):
        var csr = csr_input(X, context)
        var pc_ = ClsProg()
        var done = False
        var cc = _fit_counts_csr(pc_, model, fields, csr, y, done, context)
        if done:
            return _params(pc_, model, fields, kind, cc, NONE, _no_partial(), context)
        # IDENTICAL only: the dense program, its block built on the device
        # from the CSR arrays (lane idn-all)
        var q = ClsProg()
        var cq = _fit_counts(q, model, fields, kind, X, y, binarize, PythonObject(None), csr, True, context)
        return _params(q, model, fields, kind, cq, NONE, _no_partial(), context)
    var pr = ClsProg()
    var c = _fit_counts(pr, model, fields, kind, X, y, binarize, sample_weight, _no_csr(context), False, context)
    return _params(pr, model, fields, kind, c, NONE, _no_partial(), context)


def _dnb_partial_fit(model: PythonObject, fields: PythonObject, kind: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_DiscreteNB.partial_fit`."""
    _check_alpha(model, fields, context)
    var arr = cls_x2d(args[0], context)
    var n = arr.n
    var d = arr.d
    var pc = partial_codes(model, fields, args[1], args[2], n, context)
    var K = _n_classes(fields)
    var mode = cls_mode(context)
    if not pc.first:
        check_width(model, fields, d)
        mode = fields["numeric_mode_"]
    var pr = ClsProg()
    var xo = pr.put(arr.value)
    var wo = _put_weights(pr, _weights(args[3], n, context))
    var onepass = nb_onepass(mode, context)
    var thr = NONE
    var binarize = _binarize(kind, fields)
    if binarize is not None:
        if onepass:
            thr = pr.put_scalar(cls_float(binarize), context)
        else:
            var xb = pr.work(n * d)
            var to = pr.put_scalar(cls_float(binarize), context)
            pr.stage(OP_BINARIZE, n * d, xo, n * d, to, xb)
            xo = xb
    var coded = stage_partial_codes(pr, pc, context)
    var yo = coded[0]
    var st = pr.alloc(6 * d)
    var cnt = pr.alloc(K)
    var fc = pr.alloc(K * d)
    var clp = pr.alloc(K)
    if not onepass:
        col_stats(pr, xo, n, d, st, False)
    var bc = cnt
    var bf = fc
    if not pc.first:
        bc = pr.alloc(K)
        bf = pr.alloc(K * d)
    if onepass:
        nb_counts(pr, wo, xo, n, d, yo, K, bc, bf, thr, st + 3 * d if _needs_min(kind) else NONE)
    else:
        class_stats(pr, wo, K * d, xo, n, d, yo, K, bc, NONE, NONE, bf, mode, context, False)
    if not pc.first:
        var oc = _put_attr(pr, fields, "class_count_", context)
        pr.stage(OP_ADD_ARRAYS, K, oc, bc, cnt)
        var of = _put_attr(pr, fields, "feature_count_", context)
        pr.stage(OP_ADD_ARRAYS, K * d, of, bf, fc)
    _prior_stages(pr, model, fields, K, cnt, clp, context)
    return _params(pr, model, fields, kind, Counts(mode, n, d, K, st, cnt, fc, clp), coded[1], pc, context)


# ---------------------------------------------------------------- CategoricalNB


def _resident_enter(context: PythonObject) raises -> PythonObject:
    """`with _arena_io.resident():` entered natively (the scope's own native
    enter): the two programs share their big inputs' device copies."""
    var scope = py_call(cls_builtins().object.__new__, context["resident_class"])
    var fields = py_fields(scope)
    py_setitem(fields, "min_words", context["resident_min_words"])
    py_setitem(fields, "_scope", PythonObject(None))
    # the arena rows owner connects its context when model_selection imports
    _ = arena_cache_release_operation(scope, "scope_enter", Python.tuple(), context["arena_io_fields"]["_ROWS_CONTEXT"])
    return scope


def _resident_exit(scope: PythonObject, context: PythonObject) raises:
    _ = arena_cache_release_operation(scope, "scope_exit", Python.tuple(PythonObject(None), PythonObject(None), PythonObject(None)), context["arena_io_fields"]["_ROWS_CONTEXT"])


def _int_list(values: PythonObject, context: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    var floats = float_list(values, context)
    for i in range(len(floats)):  # driver-loop: over a user parameter list (feature-count sized)
        out.append(Int(floats[i]))
    return out^


def _cat_blocks(pr: ClsProg, off: Int, d: Int, K: Int, cmax: Int, ncat: List[Int], context: PythonObject) raises -> PythonObject:
    """Per feature j, the (K, ncat[j]) float32 block of the (d, K, cmax) table."""
    var out = Python.list()
    for j in range(d):  # driver-loop: over the feature columns (model structure)
        var block = classical_typed(Python.tuple(K, ncat[j]), "<f4", context["output_context"]) if ncat[j] > 0 else classical_wrap_array(py_call(context["array_type"], "f"), Python.tuple(K, 0), "<f4", "C", context["output_context"])
        if ncat[j] > 0:
            var dest = F32P(unsafe_from_address=Int(py=classical_address(block)))
            _ = block  # keep the Python object alive through the native call above
            for k in range(K):  # driver-loop: over the classes (model structure)
                memcpy(dest=dest + k * ncat[j], src=F32P(unsafe_from_address=pr.base) + off + (j * K + k) * cmax, count=ncat[j])
                _ = block  # keep the Python object alive through the native call above
        _ = py_call(out.append, block)
    return out


def _cat_fit_programs(model: PythonObject, fields: PythonObject, arr: XIn, pc: PartialCodes, w: PythonObject,
                      mode: PythonObject, merge: Bool, context: PythonObject) raises -> PythonObject:
    var n = arr.n
    var d = arr.d
    var K = _n_classes(fields)
    var pr = ClsProg()
    var xo = pr.put(arr.value)
    var coded = stage_partial_codes(pr, pc, context)
    var yo = coded[0]
    var st = pr.alloc(6 * d)
    var cnt = pr.alloc(K)
    var clp = pr.alloc(K)
    var wo = _put_weights(pr, w)
    col_stats(pr, xo, n, d, st, False)
    if merge:
        var bc = pr.alloc(K)
        class_stats(pr, wo, K, xo, n, 1, yo, K, bc, NONE, NONE, NONE, mode, context, False)
        var oc = _put_attr(pr, fields, "class_count_", context)
        pr.stage(OP_ADD_ARRAYS, K, oc, bc, cnt)
    else:
        class_stats(pr, wo, K, xo, n, 1, yo, K, cnt, NONE, NONE, NONE, mode, context, False)
    if fields["class_prior"] is not None:
        var po = pr.put_list(given_priors(fields["class_prior"], K, "CategoricalNB", False, context), context)
        pr.stage(OP_LOG, K, po, clp)
    else:
        var prior_src = cnt if Bool(py=fields["fit_prior"]) else pr.put_list(_ones(K), context)
        pr.stage(OP_CLASS_LOG_PRIOR, K, prior_src, K, clp)
    pr.run(mode, context)
    partial_refuse(pr, coded[1], fields, pc, context)
    for j in range(d):  # driver-loop: over the per-column minimum statistics (feature count)
        if pr.value(st + 3 * d + j) < 0:
            raise Error("mojolearn: Negative values in data passed to CategoricalNB (input X)")
    var ncat = List[Int]()
    for j in range(d):  # driver-loop: over the per-column maximum statistics (feature count)
        ncat.append(Int(pr.value(st + 4 * d + j)) + 1)
    var mc = fields["min_categories"]
    if mc is not None:
        var mcs = List[Int]()
        if Bool(py=py_call(cls_builtins().isinstance, mc, context["integral_type"])):
            mcs = List[Int](length=d, fill=Int(py=py_call(cls_builtins().int, mc)))
        else:
            mcs = _int_list(mc, context)
        if len(mcs) != d:
            raise Error("mojolearn: 'min_categories' should have shape (" + String(d) + ",) when an array-like is provided. Got " + String(len(mcs)) + " entries instead.")
        for j in range(d):  # driver-loop: over the feature columns (model structure)
            ncat[j] = max(ncat[j], mcs[j])
    if merge:
        var old = I64P(unsafe_from_address=Int(py=classical_address(fields["n_categories_"])))
        for j in range(d):  # driver-loop: over the feature columns (model structure)
            ncat[j] = max(ncat[j], Int(old[j]))
    var cmax = 0
    for j in range(d):  # driver-loop: over the feature columns (model structure)
        cmax = max(cmax, ncat[j])
    var q = ClsProg()
    var xq = q.put(arr.value)
    var qcoded = stage_partial_codes(q, pc, context)
    var yq = qcoded[0]
    var nlist = List[Float64]()
    for j in range(d):  # driver-loop: over the feature columns (model structure)
        nlist.append(Float64(ncat[j]))
    var no = q.put_list(nlist, context)
    var co = q.put(pr.get(cnt, K, 0, context))
    var a = q.put_scalar(cls_float(fields["alpha"]), context)
    var wq = _put_weights(q, w)
    var cc = q.alloc(d * K * cmax)
    if blocked():
        var rows = XB
        while ((n + rows - 1) // rows) * d * K * cmax > CAT_HIST_WORDS and rows < 4194304:  # driver-loop: the histogram block doubles (a function of the shape only)
            rows *= 2
        var nbh = (n + rows - 1) // rows
        var hist = q.work(nbh * d * K * cmax)
        q.stage(OP_CAT_HPART, nbh * d, xq, n, d, yq, K, no, cmax, wq, hist, rows)
        q.stage(OP_CAT_HFOLD, d * K * cmax, hist, nbh, d, K, no, cmax, wq, cc)
    else:
        q.stage(OP_CAT_COUNTS, d * K * cmax, xq, n, d, yq, K, no, cmax, wq, cc)
    if merge:
        var oc = Int(py=fields["_cmax"])
        var pad = q.alloc(d * K * cmax)
        var src = cc
        cc = q.alloc(d * K * cmax)
        if oc > 0:
            var old_cc = _put_attr(q, fields, "_cc", context)
            q.stage(OP_COLBLOCK, d * K * oc, old_cc, oc, pad, cmax, 0)
        q.stage(OP_ADD_ARRAYS, d * K * cmax, pad, src, cc)
    var flp = q.alloc(d * K * cmax)
    q.stage(OP_CAT_FLP, d * K * cmax, cc, K, no, cmax, co, a, flp)
    q.run(mode, context)
    partial_refuse(q, qcoded[1], fields, pc, context)
    _set(fields, "_cc", q.get(cc, d * K * cmax, 0, context))
    _set(fields, "category_count_", _cat_blocks(q, cc, d, K, cmax, ncat, context))
    var counts = classical_typed(Python.tuple(d), "<i8", context["output_context"])
    var cp = I64P(unsafe_from_address=Int(py=classical_address(counts)))
    _ = counts  # keep the Python object alive through the native call above
    for j in range(d):  # driver-loop: over the feature columns (model structure)
        cp[j] = Int64(ncat[j])
        _ = counts  # keep the Python object alive through the native call above
    _set(fields, "n_categories_", counts)
    _set(fields, "_flp", q.get(flp, d * K * cmax, 0, context))
    _set(fields, "_cmax", PythonObject(cmax))
    _set(fields, "feature_log_prob_", _cat_blocks(q, flp, d, K, cmax, ncat, context))
    _set(fields, "class_count_", pr.get(cnt, K, 0, context))
    _set(fields, "class_log_prior_", pr.get(clp, K, 0, context))
    _set(fields, "numeric_mode_", mode)
    _set(fields, "n_features_in_", PythonObject(d))
    return model


def _cat_fit(model: PythonObject, fields: PythonObject, arr: XIn, pc: PartialCodes, sample_weight: PythonObject,
             mode: PythonObject, merge: Bool, context: PythonObject) raises -> PythonObject:
    var w = _weights(sample_weight, arr.n, context)
    var scope = _resident_enter(context)
    try:
        var result = _cat_fit_programs(model, fields, arr, pc, w, mode, merge, context)
        _resident_exit(scope, context)
        return result
    except error:
        try:
            _resident_exit(scope, context)
        except ignored:
            _ = ignored^
        raise error^


def _cat_fit_op(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    _check_alpha(model, fields, context)
    var arr = cls_x2d(args[0], context)
    var codes = encode_y(model, fields, args[1], arr.n, context)
    var pc = PartialCodes(True, False, codes, PythonObject(None), 0, arr.n, PythonObject(None), PythonObject(None))
    return _cat_fit(model, fields, arr, pc, args[2], cls_mode(context), False, context)


def _cat_partial_fit(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    _check_alpha(model, fields, context)
    var arr = cls_x2d(args[0], context)
    var pc = partial_codes(model, fields, args[1], args[2], arr.n, context)
    if pc.first:
        return _cat_fit(model, fields, arr, pc, args[3], cls_mode(context), False, context)
    check_width(model, fields, arr.d)
    return _cat_fit(model, fields, arr, pc, args[3], fields["numeric_mode_"], True, context)


# ---------------------------------------------------------------- discriminant analysis


def _shrinkage(shrinkage: PythonObject, context: PythonObject) raises -> Optional[Float64]:
    """`_shrinkage_value`: None, -1.0 for 'auto', else the float in [0, 1]."""
    var b = cls_builtins()
    if shrinkage is None:
        return None
    if Bool(py=py_call(b.isinstance, shrinkage, b.str)):
        if String(shrinkage) != "auto":
            raise Error("mojolearn: invalid shrinkage " + cls_repr(shrinkage))
        return Optional[Float64](-1.0)
    if not is_real(shrinkage, context) or not (cls_float(shrinkage) >= 0 and cls_float(shrinkage) <= 1):
        raise Error("mojolearn: shrinkage must be 'auto' or a float in [0, 1], got " + cls_repr(shrinkage))
    return Optional[Float64](cls_float(shrinkage))


def _check_estimator_shrinkage(est: PythonObject, shrinkage: PythonObject) raises:
    if est is not None and shrinkage is not None and Bool(py=shrinkage != PythonObject(0)):
        raise Error("mojolearn: covariance_estimator and shrinkage parameters are not None. Only one of the two can be set.")


def _solver_is(fields: PythonObject, name: String) raises -> Bool:
    var solver = fields["solver"]
    return py_type(solver) is cls_builtins().str and String(solver) == name


def _estimator_covs(est: PythonObject, arr: XIn, codes: PythonObject, K: Int, who: String,
                    context: PythonObject) raises -> List[Float64]:
    """`_estimator_covs`: the user covariance estimator fitted on each class's
    float32 rows (grouped on the device by p2m_ccount .. p2m_rgather), its
    covariance_ words as floats."""
    var values = arr.value
    if py_type(values) is context["maxabs_rows_class"]:
        # the user's estimator takes host rows (lane cpu4-misc)
        values = _values(values, context, True)
    var n = arr.n
    var d = arr.d
    var mode = cls_mode(context)
    var blocks = Python.list()
    if codes is not None:
        var pr = ClsProg()
        var xo = pr.put(values)
        var co = pr.put_codes(codes)
        var ch = max(II_CH, (n + II_CH - 1) // II_CH, (n * K + 16777215) // 16777216, 1)
        var nch = max(1, (n + ch - 1) // ch)
        var cnt = pr.alloc(nch * K)
        var tot = pr.alloc(K)
        pr.stage(OP_P2M_CCOUNT, nch, co, n, K, ch, cnt)
        pr.stage(OP_P2M_CSCAN, K, cnt, nch, K, tot)
        var start = pr.alloc(K)
        var ro = pr.alloc(n)
        pr.stage(OP_P2M_CSTART, 1, tot, K, start, NONE)
        pr.stage(OP_P2M_CWRITE, nch, co, n, K, ch, cnt, start, ro)
        var xg = pr.alloc(n * d)
        pr.stage(OP_P2M_RGATHER, n * d, xo, d, ro, xg)
        pr.run(mode, context)
        var totals = I32P(unsafe_from_address=pr.base) + tot
        var at = 0
        for k in range(K):  # driver-loop: over the classes' row blocks (class count)
            var rows = Int(totals[k])
            _ = py_call(blocks.append, pr.get(xg + at * d, rows, d, context))
            at += rows
    else:
        var copy = classical_typed(Python.tuple(n, d), "<f4", context["output_context"])
        memcpy(dest=F32P(unsafe_from_address=Int(py=classical_address(copy))), src=F32P(unsafe_from_address=Int(py=classical_address(values))), count=n * d)
        _ = copy  # keep the Python object alive through the native call above
        _ = values  # keep the Python object alive through the native call above
        _ = py_call(blocks.append, copy)
    var out = List[Float64]()
    var b = cls_builtins()
    for k in range(len(blocks)):  # cpu-route: fits the user covariance estimator once per class
        _ = py_call(est.fit, blocks[k])
        if not Bool(py=py_call(b.hasattr, est, "covariance_")):
            raise Error("mojolearn: " + cls_name(est) + " does not have a covariance_ attribute")
        var view = classical_view(est.covariance_, context, "covariance_")
        if Int(py=view.size) != d * d:
            raise Error("mojolearn: " + who + ": covariance_ of " + cls_name(est) + " is not (" + String(d) + ", " + String(d) + ")")
        var words = classical_host_f32(view, max(1, Int(py=view.ndim)), "covariance_", context)
        var p = F32P(unsafe_from_address=Int(py=classical_address(words)))
        _ = words  # keep the Python object alive through the native call above
        for i in range(d * d):  # cpu-route: reads the user covariance estimator output
            out.append(Float64(p[i]))
            _ = words  # keep the Python object alive through the native call above
        _ = words^
    return out^


def _lda_cov_blocks(mut pr: ClsProg, xo: Int, n: Int, d: Int, yo: Int, K: Int, mean: Int, var_: Int, cnt: Int,
                    shr: Optional[Float64], given: Optional[List[Float64]], context: PythonObject) raises -> Int:
    if given:
        return pr.put_list(given.value(), context)
    var cov = pr.alloc(K * d * d)
    qda_cov(pr, cls_mode(context), xo, n, d, yo, K, mean, cnt, cov, context)
    if shr:
        var so = pr.put_scalar(shr.value(), context)
        var scratch = pr.alloc(K)
        pr.stage(OP_DA_SHRINK, K, xo, n, d, yo, mean, var_, cnt, cov, so, scratch)
    return cov


def _lda_binary(mut pr: ClsProg, coef: Int, inter: Int, K: Int, d: Int, context: PythonObject) raises -> Tuple[Int, Int]:
    var cd = pr.alloc(d)
    var ci = pr.alloc(1)
    if K == 2:
        _binary_difference(pr, coef, 1, K, d, cd, context)
        var w = pr.put_list(_pm_one(), context)
        pr.stage(OP_MATMUL, 1, w, 0, 1, inter, 1, 0, ci, 1, 2, NONE, NONE)
    return (cd, ci)


def _lda_priors(mut pr: ClsProg, fields: PythonObject, K: Int, context: PythonObject) raises -> Tuple[Int, Int]:
    """(gflag, gofs) of the user priors (2 when they are renormalized)."""
    if fields["priors"] is None:
        return (0, 0)
    var pv = given_priors(fields["priors"], K, "LinearDiscriminantAnalysis", False, context)
    var flag = 2 if abs(prior_sum(pv) - 1.0) > 1e-5 else 1
    return (flag, pr.put_list(pv, context))


def _lda_publish_coef(pr: ClsProg, fields: PythonObject, K: Int, d: Int, coef: Int, inter: Int, cd: Int, ci: Int,
                      context: PythonObject) raises:
    var c = pr.get(coef, K, d, context)
    var i = pr.get(inter, K, 0, context)
    _set(fields, "_coef", c)
    _set(fields, "_inter", i)
    if K == 2:
        _set(fields, "coef_", pr.get(cd, 1, d, context))
        _set(fields, "intercept_", pr.get(ci, 1, 0, context))
    else:
        _set(fields, "coef_", c)
        _set(fields, "intercept_", i)


def _lda_fit(model: PythonObject, fields: PythonObject, X: PythonObject, y: PythonObject, context: PythonObject) raises -> XIn:
    """`LinearDiscriminantAnalysis.fit`; returns the admitted X (fit_transform's input)."""
    var b = cls_builtins()
    var solver = fields["solver"]
    if not (_solver_is(fields, "svd") or _solver_is(fields, "lsqr") or _solver_is(fields, "eigen")):
        raise Error("mojolearn: invalid solver " + cls_repr(solver))
    if _solver_is(fields, "svd") and fields["shrinkage"] is not None:
        raise Error("NotImplementedError: mojolearn: shrinkage not supported with 'svd' solver.")
    var est = fields["covariance_estimator"]
    if est is not None and _solver_is(fields, "svd"):
        raise Error("mojolearn: covariance estimator is not supported with svd solver. Try another solver")
    _check_estimator_shrinkage(est, fields["shrinkage"])
    var shr: Optional[Float64] = None
    if est is None:
        shr = _shrinkage(fields["shrinkage"], context)
    var arr = cls_x2d(X, context)
    var n = arr.n
    var d = arr.d
    var codes = encode_y(model, fields, y, n, context)
    var K = _n_classes(fields)
    if K < 2 or n <= K:
        raise Error("mojolearn: LinearDiscriminantAnalysis needs at least two classes and more samples than classes")
    var maxc = min(K - 1, d)
    var nc = fields["n_components"]
    if nc is not None and Bool(py=nc > PythonObject(maxc)):
        raise Error("mojolearn: n_components cannot be larger than min(n_features, n_classes - 1)")
    var mode = cls_mode(context)
    if not _solver_is(fields, "svd"):
        _lda_cov_solver(model, fields, arr, codes, K, d, n, mode, shr, context)
        return arr^
    var pr = ClsProg()
    var xo = pr.put(arr.value)
    var yo = pr.put_codes(codes)
    var cnt = pr.alloc(K)
    var mean = pr.alloc(K * d)
    var priors = pr.alloc(K)
    var xbar = pr.alloc(d)
    var std = pr.alloc(d)
    var w = pr.alloc(d)
    var z2 = pr.work(n * d)
    var g = pr.alloc(d * d)
    var e1 = pr.alloc(d)
    var v1 = pr.alloc(d * d)
    var meta_words = List[Float64]()
    meta_words.append(cls_float(fields["tol"]))
    meta_words.append(0.0)
    meta_words.append(0.0)
    # lda_stage2 writes the ranks into it
    var meta = pr.put_list(meta_words, context, True)
    var scal1 = pr.alloc(d * d)
    var g2 = pr.alloc(d * d)
    var ms = pr.alloc(K * d)
    var e2 = pr.alloc(d)
    var v2 = pr.alloc(d * d)
    var scal = pr.alloc(d * d)
    var coef = pr.alloc(K * d)
    var inter = pr.alloc(K)
    var evr = pr.alloc(d)
    var tmp = pr.alloc(K * d)
    cls_stats(pr, mode, xo, n, d, yo, K, cnt, mean, NONE, NONE, context)
    var gp = _lda_priors(pr, fields, K, context)
    pr.stage(OP_LDA_PREP, 1, cnt, mean, K, d, n, priors, xbar, gp[0], gp[1])
    var z = pr.work(n * d)
    var stz = pr.alloc(6 * d)
    pr.stage(OP_CENTER_ROWS, n * d, xo, n, d, mean, yo, NONE, z)
    cs(pr, mode, z, n, d, stz, context)
    pr.stage(OP_LDA_W, d, stz + 2 * d, d, n, K, std, w)
    pr.stage(OP_CENTER_ROWS, n * d, xo, n, d, mean, yo, w, z2)
    gram(pr, mode, z2, n, d, g, context)
    pr.stage(OP_EIGH, 1, g, d, 0, e1, v1)
    pr.stage(OP_LDA_STAGE2, 1, e1, v1, std, mean, xbar, priors, K, d, n, meta, scal1, g2, ms)
    pr.stage(OP_EIGH, 1, g2, d, 0, e2, v2)
    pr.stage(OP_LDA_STAGE3, 1, e2, v2, scal1, mean, xbar, priors, K, d, meta, scal, coef, inter, evr, tmp)
    var bin = _lda_binary(pr, coef, inter, K, d, context)
    var sw = NONE
    if Bool(py=fields["store_covariance"]):
        var cov = _lda_cov_blocks(pr, xo, n, d, yo, K, mean, NONE, cnt, None, None, context)
        sw = pr.alloc(d * d)
        pr.stage(OP_DA_POOL, d * d, cov, K, d, priors, sw, NONE, NONE)
    pr.run(mode, context)
    if sw != NONE:
        _set(fields, "covariance_", pr.get(sw, d, d, context))
    var rank2 = Int(pr.value(meta + 2))
    _set(fields, "_rank", PythonObject(rank2))
    _set(fields, "means_", pr.get(mean, K, d, context))
    _set(fields, "priors_", pr.get(priors, K, 0, context))
    _set(fields, "xbar_", pr.get(xbar, d, 0, context))
    _set(fields, "_scal_full", pr.get(scal, d, d, context))
    # the rank columns of the (d, d) scalings (Array.__getitem__'s strided copy)
    var scalings = classical_wrap_array(py_call(context["array_type"], "f"), Python.tuple(d, 0), "<f4", "C", context["output_context"])
    if rank2 > 0:
        scalings = classical_typed(Python.tuple(d, rank2), "<f4", context["output_context"])
        var dest = F32P(unsafe_from_address=Int(py=classical_address(scalings)))
        _ = scalings  # keep the Python object alive through the native call above
        var src = F32P(unsafe_from_address=pr.base) + scal
        for i in range(d):  # driver-loop: over the scalings' rows (model structure)
            memcpy(dest=dest + i * rank2, src=src + i * d, count=rank2)
            _ = scalings  # keep the Python object alive through the native call above
    _set(fields, "scalings_", scalings)
    _lda_publish_coef(pr, fields, K, d, coef, inter, bin[0], bin[1], context)
    var keep = maxc if nc is None else Int(py=py_call(b.int, nc))
    _set(fields, "_max_components", PythonObject(keep))
    _set(fields, "explained_variance_ratio_", pr.get(evr, keep, 0, context))
    _set(fields, "numeric_mode_", mode)
    _set(fields, "n_features_in_", PythonObject(d))
    return arr^


def _lda_cov_solver(model: PythonObject, fields: PythonObject, arr: XIn, codes: PythonObject, K: Int, d: Int, n: Int,
                    mode: PythonObject, shr: Optional[Float64], context: PythonObject) raises:
    """solver 'lsqr' / 'eigen' (`_fit_cov_solver`)."""
    var b = cls_builtins()
    var pr = ClsProg()
    var xo = pr.put(arr.value)
    var yo = pr.put_codes(codes)
    var cnt = pr.alloc(K)
    var mean = pr.alloc(K * d)
    var priors = pr.alloc(K)
    var xbar = pr.alloc(d)
    var var_ = pr.alloc(K * d) if Bool(shr) else NONE
    cls_stats(pr, mode, xo, n, d, yo, K, cnt, mean, var_, NONE, context)
    var gp = _lda_priors(pr, fields, K, context)
    pr.stage(OP_LDA_PREP, 1, cnt, mean, K, d, n, priors, xbar, gp[0], gp[1])
    var eigen = _solver_is(fields, "eigen")
    var est = fields["covariance_estimator"]
    var tot = NONE
    if eigen:
        var y0 = pr.put_list(_zeros(n), context)
        var c1 = pr.alloc(1)
        var m1 = pr.alloc(d)
        var v1 = pr.alloc(d) if Bool(shr) else NONE
        cls_stats(pr, mode, xo, n, d, y0, 1, c1, m1, v1, NONE, context)
        var gt: Optional[List[Float64]] = None
        if est is not None:
            gt = _estimator_covs(est, arr, PythonObject(None), 1, "LinearDiscriminantAnalysis", context)
        tot = _lda_cov_blocks(pr, xo, n, d, y0, 1, m1, v1, c1, shr, gt, context)
    var gk: Optional[List[Float64]] = None
    if est is not None:
        gk = _estimator_covs(est, arr, codes, K, "LinearDiscriminantAnalysis", context)
    var cov = _lda_cov_blocks(pr, xo, n, d, yo, K, mean, var_, cnt, shr, gk, context)
    var sw = pr.alloc(d * d)
    pr.stage(OP_DA_POOL, d * d, cov, K, d, priors, sw, NONE, NONE)
    var sb = NONE
    if eigen:
        sb = pr.alloc(d * d)
        var one = pr.put_list(_one(), context)
        var scratch = pr.alloc(d * d)
        pr.stage(OP_DA_POOL, d * d, sw, 1, d, one, scratch, tot, sb)
    var swc = pr.alloc(d * d)
    var one2 = pr.put_list(_one(), context)
    pr.stage(OP_DA_POOL, d * d, sw, 1, d, one2, swc, NONE, NONE)
    var e = pr.alloc(d)
    var v = pr.alloc(d * d)
    pr.stage(OP_EIGH, 1, swc, d, 0, e, v)
    var coef = pr.alloc(K * d)
    var inter = pr.alloc(K)
    var nc = fields["n_components"]
    var mc = min(K - 1, d) if nc is None else Int(py=py_call(b.int, nc))
    var ev = NONE
    var evr = NONE
    if not eigen:
        var P = pr.alloc(d * d)
        pr.stage(OP_SYM_FN, d * d, e, v, d, 0, P)
        pr.stage(OP_MATMUL, K * d, mean, d, 1, P, 1, d, coef, d, d, NONE, NONE)
    else:
        var W = pr.alloc(d * d)
        var T = pr.alloc(d * d)
        var C = pr.alloc(d * d)
        pr.stage(OP_SYM_FN, d * d, e, v, d, 1, W)
        pr.stage(OP_MATMUL, d * d, W, d, 1, sb, d, 1, T, d, d, NONE, NONE)
        pr.stage(OP_MATMUL, d * d, T, d, 1, W, d, 1, C, d, d, NONE, NONE)
        var e2 = pr.alloc(d)
        var y2 = pr.alloc(d * d)
        ev = pr.alloc(d * d)
        evr = pr.alloc(d)
        pr.stage(OP_EIGH, 1, C, d, 0, e2, y2)
        pr.stage(OP_MATMUL, d * d, W, d, 1, y2, d, 1, ev, d, d, NONE, NONE)
        pr.stage(OP_EVR, 1, e2, d, evr)
        var t1 = pr.alloc(K * d)
        pr.stage(OP_MATMUL, K * d, mean, d, 1, ev, d, 1, t1, d, d, NONE, NONE)
        pr.stage(OP_MATMUL, K * d, t1, d, 1, ev, 1, d, coef, d, d, NONE, NONE)
    pr.stage(OP_DA_INTERCEPT, K, mean, coef, priors, d, inter)
    var bin = _lda_binary(pr, coef, inter, K, d, context)
    pr.run(mode, context)
    if eigen:
        for j in range(d):  # driver-loop: over the within-class covariance eigenvalues (feature count)
            if pr.value(e + j) <= 0:
                raise Error("mojolearn: the within-class covariance is not positive definite (the reference's eigh(Sb, Sw) fails); set shrinkage")
    _set(fields, "means_", pr.get(mean, K, d, context))
    _set(fields, "priors_", pr.get(priors, K, 0, context))
    _set(fields, "xbar_", pr.get(xbar, d, 0, context))
    _set(fields, "covariance_", pr.get(sw, d, d, context))
    _lda_publish_coef(pr, fields, K, d, coef, inter, bin[0], bin[1], context)
    _set(fields, "_max_components", PythonObject(mc))
    if eigen:
        var full = pr.get(ev, d, d, context)
        _set(fields, "scalings_", full)
        _set(fields, "_scal_full", full)
        _set(fields, "_rank", PythonObject(d))
        _set(fields, "explained_variance_ratio_", pr.get(evr, mc, 0, context))
    _set(fields, "numeric_mode_", mode)
    _set(fields, "n_features_in_", PythonObject(d))


def _lda_transform(model: PythonObject, fields: PythonObject, arr: XIn, context: PythonObject) raises -> PythonObject:
    """`LinearDiscriminantAnalysis.transform` of an admitted X."""
    check_fitted(model, fields)
    check_width(model, fields, arr.d)
    var n = arr.n
    var d = arr.d
    if _solver_is(fields, "lsqr"):
        raise Error("NotImplementedError: mojolearn: transform not implemented for 'lsqr' solver (use 'svd' or 'eigen').")
    var mc = min(Int(py=fields["_max_components"]), Int(py=fields["_rank"]))
    var mode = fields["numeric_mode_"]
    var eigen = _solver_is(fields, "eigen")
    var pr = ClsProg()
    var xo = pr.put(arr.value)
    var xb = _put_attr(pr, fields, "xbar_", context)
    var sc = _put_attr(pr, fields, "_scal_full", context)
    var out = pr.alloc(n * max(mc, 1))
    if not eigen and (classical_shared(mode, context) & 4) != 0:
        # C04_LDA (-D MOJOLEARN_CLASSICAL_C04_LDA): the same center_rows -> matmul arithmetic in the consumer.
        pr.stage(OP_CENTERED_MATMUL, n * mc, xo, n, d, xb, NONE, NONE, sc, d, 1, out, mc, NONE, NONE)
    else:
        var cen = xo if eigen else pr.alloc(n * d)
        if not eigen:
            pr.stage(OP_CENTER_ROWS, n * d, xo, n, d, xb, NONE, NONE, cen)
        pr.stage(OP_MATMUL, n * mc, cen, d, 1, sc, d, 1, out, mc, d, NONE, NONE)
    pr.run(mode, context)
    return pr.get(out, n, mc, context)


def _qda_fit(model: PythonObject, fields: PythonObject, X: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`QuadraticDiscriminantAnalysis.fit`."""
    var solver = fields["solver"]
    if not (_solver_is(fields, "svd") or _solver_is(fields, "eigen")):
        raise Error("mojolearn: invalid solver " + cls_repr(solver))
    if _solver_is(fields, "svd") and fields["shrinkage"] is not None:
        raise Error("NotImplementedError: mojolearn: shrinkage not supported with 'svd' solver.")
    var est = fields["covariance_estimator"]
    if est is not None and _solver_is(fields, "svd"):
        raise Error("mojolearn: covariance_estimator is not supported with solver='svd'. Try solver='eigen' instead.")
    _check_estimator_shrinkage(est, fields["shrinkage"])
    var shr: Optional[Float64] = None
    if est is None:
        shr = _shrinkage(fields["shrinkage"], context)
    var eigen = _solver_is(fields, "eigen")
    var arr = cls_x2d(X, context)
    var n = arr.n
    var d = arr.d
    var codes = encode_y(model, fields, y, n, context)
    var K = _n_classes(fields)
    if K < 2:
        raise Error("mojolearn: QuadraticDiscriminantAnalysis needs at least two classes")
    var mode = cls_mode(context)
    # without a covariance_estimator the count check reads the program's own
    # class counts right after the run; an estimator fits per class inside
    # the build, so its check stays first (`_class_counts`, a p2m program)
    var late = est is None
    if not late:
        var cp = ClsProg()
        var co = cp.put_codes(codes)
        var tot = 0
        tot = _class_counts(cp, co, n, K)
        cp.run(mode, context)
        var totals = I32P(unsafe_from_address=cp.base) + tot
        for k in range(K):  # driver-loop: over the class counts (class count)
            if Int(totals[k]) < 2:
                raise Error("mojolearn: y has only 1 sample in a class, covariance is ill defined")
    var pr = ClsProg()
    var xo = pr.put(arr.value)
    var yo = pr.put_codes(codes)
    var cnt = pr.alloc(K)
    var mean = pr.alloc(K * d)
    var priors = pr.alloc(K)
    var xbar = pr.alloc(d)
    var cov = pr.alloc(K * d * d)
    var ev = pr.alloc(K * d)
    var evec = pr.alloc(K * d * d)
    var reg = pr.put_scalar(0.0 if eigen else cls_float(fields["reg_param"]), context)
    var rot = pr.alloc(K * d * d)
    var logc = pr.alloc(K)
    var s2 = pr.alloc(K * d)
    var var_ = pr.alloc(K * d) if Bool(shr) else NONE
    # the trailing 1: FAST keeps row-order class sums here
    cls_stats(pr, mode, xo, n, d, yo, K, cnt, mean, var_, NONE, context, 1)
    var gflag = 0
    var gofs = 0
    if fields["priors"] is not None:
        gflag = 1
        gofs = pr.put_list(given_priors(fields["priors"], K, "QuadraticDiscriminantAnalysis", False, context), context)
    pr.stage(OP_LDA_PREP, 1, cnt, mean, K, d, n, priors, xbar, gflag, gofs)
    if est is not None:
        cov = pr.put_list(_estimator_covs(est, arr, codes, K, "QuadraticDiscriminantAnalysis", context), context)
    else:
        qda_cov(pr, mode, xo, n, d, yo, K, mean, cnt, cov, context)
    if shr:
        var so = pr.put_scalar(shr.value(), context)
        var scratch = pr.alloc(K)
        pr.stage(OP_DA_SHRINK, K, xo, n, d, yo, mean, var_, cnt, cov, so, scratch)
    var store = Bool(py=fields["store_covariance"])
    var keep = NONE
    if store and eigen:
        keep = pr.alloc(K * d * d)
        var one = pr.put_list(_one(), context)
        for k in range(K):  # driver-loop: stages one pooled covariance copy per class (class count)
            pr.stage(OP_DA_POOL, d * d, cov + k * d * d, 1, d, one, keep + k * d * d, NONE, NONE)
    pr.stage(OP_EIGH, K, cov, d, d * d, ev, evec)
    pr.stage(OP_QDA_PREP, K, ev, evec, K, d, reg, cnt, n, rot, logc, s2, gflag, gofs)
    if store and not eigen:
        keep = pr.alloc(K * d * d)
        pr.stage(OP_SYM_FN, K * d * d, s2, evec, d, 2, keep)
    pr.run(mode, context)
    if late:
        for k in range(K):  # driver-loop: over the class counts (class count)
            if pr.value(cnt + k) < 2:
                raise Error("mojolearn: y has only 1 sample in a class, covariance is ill defined")
    var tol = cls_float(fields["tol"])
    var classes = fields["classes_"]
    for k in range(K):  # driver-loop: rank check per class before raising (class count)
        var above = 0
        for j in range(d):  # driver-loop: over one class's scalings (feature count)
            if pr.value(s2 + k * d + j) > tol:
                above += 1
        if above < d:
            raise Error("mojolearn: the covariance matrix of class " + cls_repr(classes[k]) + " is not full rank. Increase the value of `" + (String("shrinkage") if eigen else String("reg_param")) + "` to reduce the collinearity.")
    if keep != NONE:
        var covs = Python.list()
        for k in range(K):  # driver-loop: per-class covariance blocks (class count)
            _ = py_call(covs.append, pr.get(keep + k * d * d, d, d, context))
        _set(fields, "covariance_", covs)
    _set(fields, "means_", pr.get(mean, K, d, context))
    _set(fields, "priors_", pr.get(priors, K, 0, context))
    var rotations = Python.list()
    var scalings = Python.list()
    for k in range(K):  # driver-loop: per-class rotation and scaling blocks (class count)
        _ = py_call(rotations.append, pr.get(evec + k * d * d, d, d, context))
        _ = py_call(scalings.append, pr.get(s2 + k * d, d, 0, context))
    _set(fields, "rotations_", rotations)
    _set(fields, "scalings_", scalings)
    _set(fields, "_rot", pr.get(rot, K * d * d, 0, context))
    _set(fields, "_logc", pr.get(logc, K, 0, context))
    _set(fields, "numeric_mode_", mode)
    _set(fields, "n_features_in_", PythonObject(d))
    return model


def _class_counts(mut pr: ClsProg, co: Int, n: Int, K: Int) -> Int:
    """`_class_counts`' p2m stages: the class counts (int32 words) offset."""
    var ch = max(II_CH, (n + II_CH - 1) // II_CH, (n * K + 16777215) // 16777216, 1)
    var nch = max(1, (n + ch - 1) // ch)
    var cnt = pr.alloc(nch * K)
    var tot = pr.alloc(K)
    pr.stage(OP_P2M_CCOUNT, nch, co, n, K, ch, cnt)
    pr.stage(OP_P2M_CSCAN, K, cnt, nch, K, tot)
    return tot


# ---------------------------------------------------------------- the door


def _dispatch(model: PythonObject, operation: String, args: PythonObject, var context: PythonObject) raises -> PythonObject:
    if py_type(py_type(model)) is not py_type(py_type(PythonObject(0))):
        raise Error("TypeError: native classifiers cannot invoke Python metaclass hooks")
    var kind = cls_kind(model, context)
    var fields: PythonObject
    if kind == "lda":
        context = lda_native_context(context)
        if operation == "initialize":
            return lda_native_initialize(model, args, context)
        fields = lda_native_fields(model, context)
        if operation == "is_fitted":
            return classical_is_fitted(model)
        if operation == "get_params":
            return lda_native_get_params(fields, context)
        if operation == "set_params":
            return lda_native_set_params(model, fields, args[0], context)
        if operation == "fit" or operation == "fit_transform" or operation == "score":
            lda_native_labels(args[1], context, operation != "score")
        if operation != "fit" and operation != "fit_transform":
            lda_require_fitted(model, operation, context)
            fields = py_call(Python.import_module("builtins").dict, fields)
            py_setitem(fields, "classes_", classifier_label_list(fields["classes_"], context))
    else:
        fields = py_fields(model)
    if operation == "predict":
        return _predict(model, fields, kind, args[0], context)
    if operation == "predict_proba":
        return _predict_proba(model, fields, kind, args[0], context)
    if operation == "predict_log_proba":
        return _predict_log_proba(model, fields, kind, args[0], context)
    if operation == "predict_joint_log_proba":
        return _joint_log_proba(model, fields, kind, args[0], context)
    if operation == "score":
        return _score(model, fields, kind, args[0], args[1], context)
    if operation == "decision_function" and (kind == "lda" or kind == "qda"):
        return _decision_function(model, fields, kind, args[0], context)
    if operation == "fit":
        if kind == "gnb":
            return _gnb_fit(model, fields, args, context)
        if kind == "mnb" or kind == "bnb" or kind == "cnb":
            return _dnb_fit(model, fields, kind, args, context)
        if kind == "cat":
            return _cat_fit_op(model, fields, args, context)
        if kind == "lda":
            _ = _lda_fit(model, fields, args[0], args[1], context)
            return model
        return _qda_fit(model, fields, args[0], args[1], context)
    if operation == "partial_fit":
        if kind == "gnb":
            return _gnb_partial_fit(model, fields, args, context)
        if kind == "mnb" or kind == "bnb" or kind == "cnb":
            return _dnb_partial_fit(model, fields, kind, args, context)
        if kind == "cat":
            return _cat_partial_fit(model, fields, args, context)
    if kind == "lda" and operation == "transform":
        check_fitted(model, fields)
        return _lda_transform(model, fields, cls_x2d(args[0], context), context)
    if kind == "lda" and operation == "fit_transform":
        # The X converted for the fit is the transform's input.
        var arr = _lda_fit(model, fields, args[0], args[1], context)
        return _lda_transform(model, fields, arr, context)
    raise Error("TypeError: unknown native classifier operation " + operation)


def prep_classify_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        classical_connect_numpy(context)
        var result = _dispatch(model, operation, args, context)
        if py_type(model) is py_get(context, "lda_native_class"):
            if operation == "fit" or operation == "fit_transform":
                classifier_after_fit(py_fields(model), args[0], context)
            result = _public_result(result, context)
        return result.steal_data()
    except error:
        var message = String(error)
        if message == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if message.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if message.startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if message.startswith("IndexError:"):
            return raise_python_exception(error_without_class(error, "IndexError"), ExceptionType("PyExc_IndexError"))
        if message.startswith("ZeroDivisionError:"):
            return raise_python_exception(error_without_class(error, "ZeroDivisionError"), ExceptionType("PyExc_ZeroDivisionError"))
        if message.startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if message.startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if message.startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)

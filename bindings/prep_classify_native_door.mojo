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
from std.memory import unsafe_memcpy
from bindings.native_array_storage_door import native_array_empty
from bindings.native_loaded_module_door import native_loaded_module
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words_unpack import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem
from bindings.py_typed_call import py_call_typed
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_arrays_door import classical_address, classical_wrap_array, classical_typed
from bindings.admitted_input_unpack import admit_x
from bindings.classical_validate_unpack import classical_view
from bindings.prep_lda_admission_door import (
    lda_native_fields, lda_native_labels, lda_native_context, lda_native_initialize,
    lda_native_get_params, lda_native_set_params, lda_require_fitted,
    lda_native_fields_plain, prep_plain_classes,
)
from bindings.classifier_public_lifecycle_door import classifier_label_list, classifier_after_fit
from bindings.public_lifecycle_owner_door import lifecycle_begin, lifecycle_end, lifecycle_complete, lifecycle_result
from bindings.public_lifecycle_owner_run import lifecycle_fitting
from bindings.classical_validate_unpack import classical_estimator_name
from providers.python.linear_metadata_boundary_door import classical_is_fitted
from bindings.classical_pandas_door import classical_admit_value
from bindings.msel_fixed_owner_unpack import fixed_owner_context, fixed_owner_fields
from bindings.msel_fixed_owner_pack import fixed_owner_set_params
from bindings.scaler_lifecycle_native_door import _public_result, _predict_result
from bindings.scaler_route_unpack import scaler_route_context, scaler_route_clear
from bindings.classical_labels_door import classical_encode_labels, classical_decode_labels, classical_score_from_codes
from bindings.admitted_input_unpack import admit_weight_words
from bindings.x_prep_native_inputs_door import _values, _rows_fields
from bindings.arena_cache_release_door import arena_cache_release_operation
from bindings.prep_classify_prog_door import ClsProg, NONE, f32_array, i32_array, f32_input
from bindings.prep_classify_prog_run import (
    cls_gnb_fit_program, cls_score_program, cls_tail_stages,
    ClsDaMode, cls_lda_svd_program, cls_lda_cov_program, cls_lda_transform_program, cls_lda_pair_program,
    cls_qda_fit_program,
    ClsScore, CLS_KIND_GNB, CLS_KIND_MNB, CLS_KIND_BNB, CLS_KIND_CNB, CLS_KIND_CAT, CLS_KIND_LDA, CLS_KIND_QDA,
    ClsCounts, cls_dnb_counts, cls_dnb_params, cls_prior_stages, CLS_PRIOR_GIVEN, CLS_PRIOR_FIT, CLS_PRIOR_UNIFORM,
    cls_gnb_partial_program, cls_dnb_partial_program, cls_cat_count_program, cls_cat_categories, cls_cat_bounds,
    cls_cat_table_program,
)
from bindings.prep_classify_common_door import (
    XIn, CsrIn, PartialCodes, cls_builtins, cls_name, cls_repr, cls_str, cls_kind, cls_mode, cls_requested_mode, cls_entry,
    cls_vendor_used,
    idn_int, classical_shared, blocked, check_fitted, check_width, cls_x2d, is_sparse, csr_input, csr_dense_x,
    float_list, given_priors, is_real, cls_float, cls_da_mode,
    da_shrinkage, da_check_estimator_shrinkage, da_solver_is,
    nb_onepass, encode_y, partial_codes, partial_labels, own_partial_labels, partial_refuse,
    IDN_NB_CSR, idn_fam, IDN_CLASS_ONEPASS, gnb_refuse,
)
from bindings.prep_keys_unpack import (
    prep_keys,
    PREP_OPN_CV_SET_PARAMS, PREP_OPN_DECISION_FUNCTION, PREP_OPN_FIT, PREP_OPN_FIT_TRANSFORM, PREP_OPN_GET_PARAMS,
    PREP_OPN_INITIALIZE, PREP_OPN_IS_FITTED, PREP_OPN_NB_CSR_READY, PREP_OPN_PARTIAL_FIT, PREP_OPN_PREDICT,
    PREP_OPN_PREDICT_JOINT_LOG_PROBA, PREP_OPN_PREDICT_LOG_PROBA, PREP_OPN_PREDICT_PROBA, PREP_OPN_SCORE,
    PREP_OPN_SET_PARAMS, PREP_OPN_TRANSFORM,
)
from bindings.py_keys import py_op_code
from bindings.py_balanced import py_item

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime I64P = MutPointer[Int64, MutUntrackedOrigin]

# x_prep stage codes (_expansion_prep._OPS).
comptime OP_ADD_ARRAYS = 90
comptime OP_P2M_RGATHER = 204
# What a scoring program reads back.
comptime W_PREDICT = 1
comptime W_PROBA = 2
comptime W_LOG = 4


def _n_classes(fields: PythonObject) raises -> Int:
    var nk = prep_keys()
    return len(py_item(fields, nk[].k_classes_))


def _weights(sample_weight: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`check_sample_weight(sample_weight, n)` as a float32 Array (None when
    not given): the words `_nb_weights` put."""
    if sample_weight is None:
        return PythonObject(None)
    var m = admit_weight_words(sample_weight, n, context)
    var w = f32_array(n, context)
    unsafe_memcpy(dest=F32P(unsafe_from_address=Int(py=classical_address(w))), src=F32P(unsafe_from_address=m.address), count=n)
    _ = w  # keep the Python object alive through the native call above
    _ = m  # keep the Python object alive through the native call above
    _ = m^
    return w


def _set(fields: PythonObject, name: String, value: PythonObject) raises:
    py_setitem(fields, PythonObject(name), value)


# ---------------------------------------------------------------- scoring


struct ScoreAttrs(Movable):
    """UNPACK of the fitted blocks one scoring program reads: the class's
    kind word, its blocks as admitted float32 buffers in the order the typed
    builder creates their records (None: absent), BernoulliNB's threshold
    and CategoricalNB's category bound."""
    var kind: Int
    var a: PythonObject
    var b: PythonObject
    var c: PythonObject
    var has_threshold: Bool
    var threshold: Float64
    var cmax: Int

    def __init__(out self, kind: Int):
        self.kind = kind
        self.a = PythonObject(None)
        self.b = PythonObject(None)
        self.c = PythonObject(None)
        self.has_threshold = False
        self.threshold = 0.0
        self.cmax = 0

    def words(self, which: Int) raises -> Int:
        if which == 0:
            return Int(py=self.a.size)
        if which == 1:
            return 0 if self.b is None else Int(py=self.b.size)
        return 0 if self.c is None else Int(py=self.c.size)

    def own(self, mut pr: ClsProg):
        """The records' owners, in the order `cls_jll_stages` created them."""
        pr.own(self.a)
        if self.b is not None:
            pr.own(self.b)
        if self.c is not None:
            pr.own(self.c)


def _attr(fields: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    return f32_input(fields[PythonObject(name)], context)


def _score_attrs(kind: String, fields: PythonObject, K: Int, context: PythonObject) raises -> ScoreAttrs:
    """The subclass's fitted blocks for its `_jll_stages`, read once per
    scoring operation."""
    var nk = prep_keys()
    if kind == "gnb":
        var attrs = ScoreAttrs(CLS_KIND_GNB)
        attrs.a = _attr(fields, "theta_", context)
        attrs.b = _attr(fields, "var_", context)
        attrs.c = _attr(fields, "_const", context)
        return attrs^
    if kind == "mnb":
        var attrs = ScoreAttrs(CLS_KIND_MNB)
        attrs.a = _attr(fields, "feature_log_prob_", context)
        attrs.b = _attr(fields, "class_log_prior_", context)
        return attrs^
    if kind == "bnb":
        var attrs = ScoreAttrs(CLS_KIND_BNB)
        if py_item(fields, nk[].k_binarize) is not None:
            attrs.has_threshold = True
            attrs.threshold = cls_float(py_item(fields, nk[].k_binarize))
        attrs.a = _attr(fields, "_w", context)
        attrs.b = _attr(fields, "_bias", context)
        return attrs^
    if kind == "cnb":
        var attrs = ScoreAttrs(CLS_KIND_CNB)
        attrs.a = _attr(fields, "feature_log_prob_", context)
        if K == 1:
            attrs.b = _attr(fields, "class_log_prior_", context)
        return attrs^
    if kind == "cat":
        var attrs = ScoreAttrs(CLS_KIND_CAT)
        attrs.a = _attr(fields, "_flp", context)
        attrs.b = _attr(fields, "class_log_prior_", context)
        attrs.cmax = Int(py=py_item(fields, nk[].k__cmax))
        return attrs^
    if kind == "lda":
        var attrs = ScoreAttrs(CLS_KIND_LDA)
        attrs.a = _attr(fields, "_coef", context)
        attrs.b = _attr(fields, "_inter", context)
        return attrs^
    var attrs = ScoreAttrs(CLS_KIND_QDA)
    attrs.a = _attr(fields, "means_", context)
    attrs.b = _attr(fields, "_rot", context)
    attrs.c = _attr(fields, "_logc", context)
    return attrs^


def _proba64_on(mode: PythonObject, context: PythonObject) raises -> Bool:
    """`_proba64_on`: the binding exports x_prep_proba64."""
    return cls_entry(mode, "x_prep_proba64", context) is not None


def _tail(mut pr: ClsProg, fields: PythonObject, kind: String, mode: PythonObject, n: Int, d: Int, K: Int,
          jll: Int, want: Int, chk: Int, context: PythonObject) raises -> ClsScore:
    """`_score_tail` over scores the caller staged at `jll` (the CSR
    route): the typed tail stages, the run, the input refusals."""
    var p64 = (want & W_PROBA) != 0 and _proba64_on(mode, context)
    var o = cls_tail_stages(pr.program, n, d, K, jll, want, p64, False, chk)
    pr.run(mode, context)
    _score_refusals(pr, fields, kind, d, chk)
    return o


def _score_refusals(pr: ClsProg, fields: PythonObject, kind: String, d: Int, st: Int) raises:
    """CategoricalNB's `_score_refusals`: a category index outside the fitted range."""
    var nk = prep_keys()
    if kind != "cat" or st == NONE:
        return
    var ncat = I64P(unsafe_from_address=Int(py=classical_address(py_item(fields, nk[].k_n_categories_))))
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
    return _csr_route_ready(context)


def _nb_csr_ready_op(csr_ok: Bool, context: PythonObject) -> Bool:
    """The public classmethod `_DiscreteNB._nb_csr_ready` (operation
    "nb_csr_ready", lane py-msel-prep-cnn: was a Python body over the
    binding probes): False for a class without `_csr_ok`, and False on any
    refusal of the probes, as the Python body's `except Exception` gave."""
    if not csr_ok:
        return False
    try:
        return _csr_route_ready(context)
    except:
        return False


def _csr_route_ready(context: PythonObject) raises -> Bool:
    """FAST with the `x_prep_nb_csr_fit` export, or IDENTICAL with the
    IDN_NB_CSR bit of the device binding."""
    var mode = cls_mode(context)
    if String(mode) == "identical":
        return (idn_int(mode, context) & IDN_NB_CSR) != 0
    if String(mode) != "fast":
        return False
    return cls_entry(mode, "x_prep_nb_csr_fit", context) is not None


def _base_scores(mut pr: ClsProg, model: PythonObject, fields: PythonObject, kind: String, X: PythonObject,
                 want: Int, csr: CsrIn, has_csr: Bool, context: PythonObject, pair: Bool = False) raises -> ClsScore:
    """`_Classifier._scores` (`csr`: the IDENTICAL CSR scoring's fallback).
    UNPACK: X, the fitted blocks, the image's float64-probability word. The
    program is the typed `cls_score_program`. `pair`:
    QuadraticDiscriminantAnalysis' two-class decision function, the second
    score minus the first, staged over the scores in this program."""
    var nk = prep_keys()
    var mode = py_item(fields, nk[].k_numeric_mode_)
    var xo = csr_dense_x(pr, csr, mode, context) if has_csr else NONE
    var n = csr.n
    var d = csr.d
    var x_owner = PythonObject(None)
    if xo == NONE:
        var arr = cls_x2d(py_call(X.toarray) if has_csr else X, context)
        check_width(model, fields, arr.d)
        n = arr.n
        d = arr.d
        x_owner = arr.value
    var K = _n_classes(fields)
    var attrs = _score_attrs(kind, fields, K, context)
    var p64 = (want & W_PROBA) != 0 and _proba64_on(mode, context)
    var o = cls_score_program(pr.program, attrs.kind, xo, n, d, K, attrs.words(0), attrs.words(1), attrs.words(2),
                              want, p64, pair, attrs.kind == CLS_KIND_CAT and blocked(), attrs.has_threshold,
                              attrs.threshold, attrs.cmax)
    if xo == NONE:
        pr.own(x_owner)
    attrs.own(pr)
    pr.run(mode, context)
    _score_refusals(pr, fields, kind, d, o.chk)
    return o


def _no_csr(context: PythonObject) raises -> CsrIn:
    return CsrIn(PythonObject(None), PythonObject(None), PythonObject(None), 0, 0, 0)


def _scores(mut pr: ClsProg, model: PythonObject, fields: PythonObject, kind: String, X: PythonObject, want: Int,
            context: PythonObject) raises -> ClsScore:
    """`_scores` with `_DiscreteNB._scores`' CSR route."""
    var nk = prep_keys()
    check_fitted(model, fields)
    if not (is_sparse(X) and _csr_ready(kind, context)):
        return _base_scores(pr, model, fields, kind, X, want, _no_csr(context), False, context)
    var csr = csr_input(X, context)
    var n = csr.n
    var d = csr.d
    if py_type(py_item(fields, nk[].k_n_features_in_)) is not cls_builtins().int or d != Int(py=py_item(fields, nk[].k_n_features_in_)):
        raise Error("mojolearn: X has " + String(d) + " features, but " + cls_name(model) + " was fitted with " + cls_str(py_item(fields, nk[].k_n_features_in_)))
    var K = _n_classes(fields)
    var mode = py_item(fields, nk[].k_numeric_mode_)
    var jll_csr = cls_entry(mode, "x_prep_nb_csr_jll", context)
    if jll_csr is None:
        raise Error("RuntimeError: the numeric mode's binding has no x_prep_nb_csr_jll")
    var jll_h = classical_typed(Python.tuple(n, K), "<f4", py_item(context, nk[].k_output_context))
    var flp = f32_input(py_item(fields, nk[].k_feature_log_prob_), context)
    var bias = PythonObject(0)
    var bias_keep = PythonObject(None)
    if kind == "mnb" or K == 1:
        bias_keep = f32_input(py_item(fields, nk[].k_class_log_prior_), context)
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
        _ = csr  # read by address inside the call above: named again so it outlives it (lane lastuse-addr-arg)
        _ = flp  # keep the Python object alive through the native call above
        _ = jll_h  # keep the Python object alive through the native call above
    _ = bias_keep^
    var z = pr.alloc(n * K)
    var jll = pr.alloc(n * K)
    var h = pr.put(jll_h)
    pr.stage(OP_ADD_ARRAYS, n * K, h, z, jll)
    return _tail(pr, fields, kind, mode, n, d, K, jll, want, NONE, context)


def _predict(model: PythonObject, fields: PythonObject, kind: String, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = prep_keys()
    var pr = ClsProg(model)
    var o = _scores(pr, model, fields, kind, X, W_PREDICT, context)
    return classical_decode_labels(py_item(fields, nk[].k_classes_), pr.get_i32(o.am, o.n, context), context)


def _predict_proba(model: PythonObject, fields: PythonObject, kind: String, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var pr = ClsProg(model)
    var o = _scores(pr, model, fields, kind, X, W_PROBA, context)
    if o.p64:
        return pr.get_f64(o.pp, o.n, o.K, context)
    return pr.get(o.pp, o.n, o.K, context)


def _predict_log_proba(model: PythonObject, fields: PythonObject, kind: String, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var pr = ClsProg(model)
    var o = _scores(pr, model, fields, kind, X, W_LOG, context)
    return pr.get(o.lp, o.n, o.K, context)


def _joint_log_proba(model: PythonObject, fields: PythonObject, kind: String, X: PythonObject, context: PythonObject) raises -> PythonObject:
    var pr = ClsProg(model)
    var o = _scores(pr, model, fields, kind, X, 0, context)
    return pr.get(o.jll, o.n, o.K, context)


def _score(model: PythonObject, fields: PythonObject, kind: String, X: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_Classifier.score`: accuracy of the predicted codes, scored in the same
    call. y is encoded once against `classes_` and compared with the codes
    (classical_score_from_codes); the predicted labels are never built.
    Lane p2-labels-lists-arrays: the decode of every row to a Python label
    and its re-encode in `classical_accuracy` (str / bool classes: one
    Python object per row each way) are gone."""
    var nk = prep_keys()
    var pr = ClsProg(model)
    var o = _scores(pr, model, fields, kind, X, W_PREDICT, context)
    var codes = pr.get_i32(o.am, o.n, context)
    return PythonObject(classical_score_from_codes(y, py_item(fields, nk[].k_classes_), codes, PythonObject(None),
                                                   PythonObject(None), context))


def _decision_function(model: PythonObject, fields: PythonObject, kind: String, X: PythonObject, context: PythonObject) raises -> PythonObject:
    """LinearDiscriminantAnalysis / QuadraticDiscriminantAnalysis decision_function."""
    var nk = prep_keys()
    check_fitted(model, fields)
    var K = _n_classes(fields)
    var mode = py_item(fields, nk[].k_numeric_mode_)
    if kind == "lda" and K == 2:
        # The two-class answer is the coefficient difference's one column
        # (`_pair`, and the C56 route): one program over X; the K scores the
        # old route also computed were discarded.
        var arr = cls_x2d(X, context)
        check_width(model, fields, arr.d)
        var n = arr.n
        var d = arr.d
        var coef = _sized_attr(fields, "coef_", d, "LinearDiscriminantAnalysis", context)
        var inter = _sized_attr(fields, "intercept_", 1, "LinearDiscriminantAnalysis", context)
        var pl = ClsProg(model)
        var out = cls_lda_pair_program(pl.program, n, d)
        pl.own(arr.value)
        pl.own(coef)
        pl.own(inter)
        pl.run(mode, context)
        return pl.get(out, n, 0, context)
    var pr = ClsProg(model)
    if K != 2:
        var oj = _scores(pr, model, fields, kind, X, 0, context)
        return pr.get(oj.jll, oj.n, K, context)
    # QDA's two-class answer: the difference of the two scores, staged over
    # the scores where they are in the scoring program (the matmul stage and
    # the words the second program used to run over a downloaded block).
    var o = _base_scores(pr, model, fields, kind, X, 0, _no_csr(context), False, context, True)
    return pr.get(o.am, o.n, 0, context)


# ---------------------------------------------------------------- GaussianNB


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
    """GaussianNB.fit. UNPACK: X, the class codes, the weights, the checked
    parameters, the image's mode words. RUN: the typed fit program
    (`cls_gnb_fit_program`) through `ClsProg.run`. PACK: `_gnb_publish`."""
    var nk = prep_keys()
    gnb_refuse(fields, context)
    var arr = cls_x2d(args[0], context)
    var n = arr.n
    var d = arr.d
    var codes = encode_y(model, fields, args[1], n, context)
    var K = _n_classes(fields)
    var mode = cls_requested_mode(fields, context)
    var var_smoothing = cls_float(py_item(fields, nk[].k_var_smoothing))
    var w = _weights(args[2], n, context)
    var has_priors = py_item(fields, nk[].k_priors) is not None
    var priors = List[Float64]()
    if has_priors:
        priors = given_priors(py_item(fields, nk[].k_priors), K, "GaussianNB", True, context)
    var is_blocked = blocked()
    var onepass = is_blocked and (idn_fam(mode, context) & IDN_CLASS_ONEPASS) != 0
    var pr = ClsProg(model)
    var o = cls_gnb_fit_program(pr.program, n, d, K, w is not None, var_smoothing, has_priors, priors, is_blocked,
                                onepass)
    pr.own(arr.value)
    pr.own(codes)
    if w is not None:
        pr.own(w)
    pr.run(mode, context)
    _gnb_publish(pr, fields, mode, K, d, o.theta, o.var_, o.cnt, o.prior, o.eps, o.cst, o.raw, context)
    return model


def _prev_block(fields: PythonObject, name: String, mut words: List[Int], mut owners: List[PythonObject],
                context: PythonObject) raises:
    """UNPACK of one stored fitted block a partial_fit merges into: its
    admitted float32 buffer (`_Prog.put`'s admission) appended to `owners`
    and its word count to `words`."""
    var value = f32_input(fields[PythonObject(name)], context)
    words.append(Int(py=value.size))
    owners.append(value)


def _gnb_partial_fit(model: PythonObject, fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """GaussianNB.partial_fit. UNPACK: X, the batch's labels, the weights,
    the checked parameters, the previous fitted blocks, the image's mode
    words. RUN: the typed program (`cls_gnb_partial_program`) through
    `ClsProg.run`. PACK: the label refusal and `_gnb_publish`."""
    var nk = prep_keys()
    gnb_refuse(fields, context)
    var arr = cls_x2d(args[0], context)
    var n = arr.n
    var d = arr.d
    var pc = partial_codes(model, fields, args[1], args[2], n, context)
    var K = _n_classes(fields)
    var mode = cls_requested_mode(fields, context)
    if not pc.first:
        check_width(model, fields, d)
        mode = py_item(fields, nk[].k_numeric_mode_)
    var labels = partial_labels(pc)
    var var_smoothing = cls_float(py_item(fields, nk[].k_var_smoothing))
    var w = _weights(args[3], n, context)
    var prev = List[Int]()
    var prev_owners = List[PythonObject]()
    if not pc.first:
        _prev_block(fields, "class_count_", prev, prev_owners, context)
        _prev_block(fields, "theta_", prev, prev_owners, context)
        _prev_block(fields, "_raw_var", prev, prev_owners, context)
    var has_priors = py_item(fields, nk[].k_priors) is not None
    var priors = List[Float64]()
    if has_priors:
        priors = given_priors(py_item(fields, nk[].k_priors), K, "GaussianNB", True, context)
    var is_blocked = blocked()
    var onepass = is_blocked and (idn_fam(mode, context) & IDN_CLASS_ONEPASS) != 0
    var pr = ClsProg(model)
    var o = cls_gnb_partial_program(pr.program, n, d, K, labels, var_smoothing, w is not None, prev, has_priors,
                                    priors, is_blocked, onepass)
    pr.own(arr.value)
    own_partial_labels(pr, pc)
    if w is not None:
        pr.own(w)
    for i in range(len(prev_owners)):  # driver-loop: over the three fitted blocks merged into
        pr.own(prev_owners[i])
    pr.run(mode, context)
    partial_refuse(pr, o.neg, fields, pc, context)
    var f = o.fit
    _gnb_publish(pr, fields, mode, K, d, f.theta, f.var_, f.cnt, f.prior, f.eps, f.cst, f.raw, context)
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
    var nk = prep_keys()
    var alpha = py_item(fields, nk[].k_alpha)
    if not is_real(alpha, context) or not (cls_float(alpha) > 0):
        raise Error("NotImplementedError: mojolearn: " + cls_name(model) + " needs alpha > 0 (alpha = 0 makes log(0) terms)")


struct DnbPrior(Movable):
    """UNPACK of the class prior of a discrete naive Bayes fit: the typed
    choice word and the checked `class_prior` list (empty unless given)."""
    var choice: Int
    var priors: List[Float64]

    def __init__(out self, choice: Int, var priors: List[Float64]):
        self.choice = choice
        self.priors = priors^


def _dnb_prior(fields: PythonObject, K: Int, who: String, context: PythonObject) raises -> DnbPrior:
    var nk = prep_keys()
    var given = py_get(fields, nk[].k_class_prior)
    if given is not None:
        return DnbPrior(CLS_PRIOR_GIVEN, given_priors(given, K, who, False, context))
    if Bool(py=py_item(fields, nk[].k_fit_prior)):
        return DnbPrior(CLS_PRIOR_FIT, List[Float64]())
    return DnbPrior(CLS_PRIOR_UNIFORM, List[Float64]())


def _prior_stages(mut pr: ClsProg, model: PythonObject, fields: PythonObject, K: Int, cnt: Int, clp: Int,
                  context: PythonObject) raises:
    """`_DiscreteNB._prior_stages`: UNPACK of the prior, the typed stages
    (`cls_prior_stages`)."""
    var prior = _dnb_prior(fields, K, cls_name(model), context)
    cls_prior_stages(pr.program, prior.choice, prior.priors, K, cnt, clp)


def _kind_word(kind: String) -> Int:
    if kind == "mnb":
        return CLS_KIND_MNB
    if kind == "bnb":
        return CLS_KIND_BNB
    return CLS_KIND_CNB


def _binarize(kind: String, fields: PythonObject) raises -> PythonObject:
    var nk = prep_keys()
    if kind != "bnb":
        return PythonObject(None)
    return py_item(fields, nk[].k_binarize)


def _fit_counts(mut pr: ClsProg, model: PythonObject, fields: PythonObject, kind: String, X: PythonObject,
                y: PythonObject, binarize: PythonObject, sample_weight: PythonObject, csr: CsrIn, has_csr: Bool,
                context: PythonObject) raises -> Counts:
    """`_DiscreteNB._fit_counts`. UNPACK: X (or the block the CSR arrays
    build on the device), the class codes, the weights, the threshold, the
    prior, the image's one-pass word. The stages are the typed
    `cls_dnb_counts`; the owners follow in its record order."""
    var mode = cls_mode(context)
    var xo = csr_dense_x(pr, csr, mode, context) if has_csr else NONE
    var n = csr.n
    var d = csr.d
    var x_owner = PythonObject(None)
    if xo == NONE:
        var arr = cls_x2d(py_call(X.toarray) if has_csr else X, context)
        n = arr.n
        d = arr.d
        x_owner = arr.value
    var codes = encode_y(model, fields, y, n, context)
    var K = _n_classes(fields)
    var w = _weights(sample_weight, n, context)
    var has_threshold = binarize is not None
    var threshold = cls_float(binarize) if has_threshold else 0.0
    var prior = _dnb_prior(fields, K, cls_name(model), context)
    var c = cls_dnb_counts(pr.program, _kind_word(kind), xo, n, d, K, w is not None, has_threshold, threshold,
                           nb_onepass(mode, context), blocked(), prior.choice, prior.priors)
    if xo == NONE:
        pr.own(x_owner)
    if w is not None:
        pr.own(w)
    pr.own(codes)
    return Counts(mode, n, d, K, c.st, c.cnt, c.fc, c.clp)


def _fit_counts_csr(mut pr: ClsProg, model: PythonObject, fields: PythonObject, csr: CsrIn, y: PythonObject,
                    mut done: Bool, context: PythonObject) raises -> Counts:
    """`_DiscreteNB._fit_counts_csr`; `done` False: IDENTICAL refused the
    rows (the caller runs the dense program)."""
    var nk = prep_keys()
    var n = csr.n
    var d = csr.d
    var codes = encode_y(model, fields, y, n, context)
    var K = _n_classes(fields)
    var mode = cls_mode(context)
    var fit_csr = cls_entry(mode, "x_prep_nb_csr_fit", context)
    if fit_csr is None:
        raise Error("RuntimeError: the numeric mode's binding has no x_prep_nb_csr_fit")
    var fc = classical_typed(Python.tuple(K, d), "<f4", py_item(context, nk[].k_output_context))
    var cnt = classical_typed(Python.tuple(K), "<f4", py_item(context, nk[].k_output_context))
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
    var nk = prep_keys()
    var K = c.K
    var d = c.d
    var norm = kind == "cnb" and Bool(py=py_item(fields, nk[].k_norm))
    var o = cls_dnb_params(pr.program, _kind_word(kind), ClsCounts(c.n, d, K, c.st, c.cnt, c.fc, c.clp),
                           cls_float(py_item(fields, nk[].k_alpha)), norm)
    var flp = o.flp
    var w = o.w
    var bias = o.bias
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
        var pc_ = ClsProg(model)
        var done = False
        var cc = _fit_counts_csr(pc_, model, fields, csr, y, done, context)
        if done:
            return _params(pc_, model, fields, kind, cc, NONE, _no_partial(), context)
        # IDENTICAL only: the dense program, its block built on the device
        # from the CSR arrays (lane idn-all)
        var q = ClsProg(model)
        var cq = _fit_counts(q, model, fields, kind, X, y, binarize, PythonObject(None), csr, True, context)
        return _params(q, model, fields, kind, cq, NONE, _no_partial(), context)
    var pr = ClsProg(model)
    var c = _fit_counts(pr, model, fields, kind, X, y, binarize, sample_weight, _no_csr(context), False, context)
    return _params(pr, model, fields, kind, c, NONE, _no_partial(), context)


def _dnb_partial_fit(model: PythonObject, fields: PythonObject, kind: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_DiscreteNB.partial_fit`. UNPACK: X, the batch's labels, the weights,
    the threshold, the previous counts, the prior, the image's one-pass
    word. RUN: the typed counts (`cls_dnb_partial_program`) and parameter
    stage through `ClsProg.run`. PACK: `_params`' refusals and attributes."""
    var nk = prep_keys()
    _check_alpha(model, fields, context)
    var arr = cls_x2d(args[0], context)
    var n = arr.n
    var d = arr.d
    var pc = partial_codes(model, fields, args[1], args[2], n, context)
    var K = _n_classes(fields)
    var mode = cls_mode(context)
    if not pc.first:
        check_width(model, fields, d)
        mode = py_item(fields, nk[].k_numeric_mode_)
    var w = _weights(args[3], n, context)
    var onepass = nb_onepass(mode, context)
    var binarize = _binarize(kind, fields)
    var has_threshold = binarize is not None
    var threshold = cls_float(binarize) if has_threshold else 0.0
    var labels = partial_labels(pc)
    var prev = List[Int]()
    var prev_owners = List[PythonObject]()
    if not pc.first:
        _prev_block(fields, "class_count_", prev, prev_owners, context)
        _prev_block(fields, "feature_count_", prev, prev_owners, context)
    var prior = _dnb_prior(fields, K, cls_name(model), context)
    var pr = ClsProg(model)
    var o = cls_dnb_partial_program(pr.program, _kind_word(kind), n, d, K, labels, w is not None, has_threshold,
                                    threshold, onepass, blocked(), prev, prior.choice, prior.priors)
    pr.own(arr.value)
    if w is not None:
        pr.own(w)
    own_partial_labels(pr, pc)
    for i in range(len(prev_owners)):  # driver-loop: over the two count blocks added to
        pr.own(prev_owners[i])
    var c = o.counts
    return _params(pr, model, fields, kind, Counts(mode, n, d, K, c.st, c.cnt, c.fc, c.clp), o.neg, pc, context)


# ---------------------------------------------------------------- CategoricalNB


def _resident_enter(context: PythonObject) raises -> PythonObject:
    """`with _arena_io.resident():` entered natively (the scope's own native
    enter): the two programs share their big inputs' device copies."""
    var nk = prep_keys()
    return resident_enter_min(context, py_item(context, nk[].k_resident_min_words))


def resident_enter_min(context: PythonObject, min_words: PythonObject) raises -> PythonObject:
    """`with _arena_io.resident(min_words):` entered natively: inputs of at
    least `min_words` words keep one device copy for the scope (RFE's rounds
    pass n * d, so only X is held; lane ne2-b-rounds, N059). Inside an
    enclosing scope the enclosing one is used and this one closes nothing."""
    var nk = prep_keys()
    var scope = py_call(cls_builtins().object.__new__, py_item(context, nk[].k_resident_class))
    # the floor goes to the native scope state with the enter call; no
    # attribute of the `resident` object is written
    # the arena rows owner connects its context when model_selection imports
    _ = arena_cache_release_operation(scope, "scope_enter", Python.tuple(min_words), py_item(py_item(context, nk[].k_arena_io_fields), nk[].k__ROWS_CONTEXT))
    return scope


def _resident_exit(scope: PythonObject, context: PythonObject) raises:
    var nk = prep_keys()
    _ = arena_cache_release_operation(scope, "scope_exit", Python.tuple(PythonObject(None), PythonObject(None), PythonObject(None)), py_item(py_item(context, nk[].k_arena_io_fields), nk[].k__ROWS_CONTEXT))


def _int_list(values: PythonObject, context: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    var floats = float_list(values, context)
    for i in range(len(floats)):  # driver-loop: over a user parameter list (feature-count sized)
        out.append(Int(floats[i]))
    return out^


def _cat_blocks(pr: ClsProg, off: Int, d: Int, K: Int, cmax: Int, ncat: List[Int], context: PythonObject) raises -> PythonObject:
    """Per feature j, the (K, ncat[j]) float32 block of the (d, K, cmax) table."""
    var nk = prep_keys()
    var out = Python.list()
    for j in range(d):  # driver-loop: over the feature columns (model structure)
        var block = classical_typed(Python.tuple(K, ncat[j]), "<f4", py_item(context, nk[].k_output_context)) if ncat[j] > 0 else classical_wrap_array(native_array_empty("f"), Python.tuple(K, 0), "<f4", "C", py_item(context, nk[].k_output_context))
        if ncat[j] > 0:
            var dest = F32P(unsafe_from_address=Int(py=classical_address(block)))
            _ = block  # keep the Python object alive through the native call above
            for k in range(K):  # driver-loop: over the classes (model structure)
                unsafe_memcpy(dest=dest + k * ncat[j], src=F32P(unsafe_from_address=pr.base) + off + (j * K + k) * cmax, count=ncat[j])
                _ = block  # keep the Python object alive through the native call above
        _ = py_call(out.append, block)
    return out


def _cat_fit_programs(model: PythonObject, fields: PythonObject, arr: XIn, pc: PartialCodes, w: PythonObject,
                      mode: PythonObject, merge: Bool, context: PythonObject) raises -> PythonObject:
    """CategoricalNB's fit (`merge`: partial_fit after the first call).
    UNPACK: the labels, the prior, the previous class_count_. RUN: the typed
    first program (`cls_cat_count_program`) through `ClsProg.run`; the typed
    category counts (`cls_cat_categories`, `cls_cat_bounds`, after the
    UNPACK of `min_categories` and the previous n_categories_); the typed
    second program (`cls_cat_table_program`). PACK: the fitted attributes."""
    var nk = prep_keys()
    var n = arr.n
    var d = arr.d
    var K = _n_classes(fields)
    var labels = partial_labels(pc)
    var weighted = w is not None
    var prev_words = List[Int]()
    var prev_owners = List[PythonObject]()
    if merge:
        _prev_block(fields, "class_count_", prev_words, prev_owners, context)
    var prior = _dnb_prior(fields, K, "CategoricalNB", context)
    var is_blocked = blocked()
    var pr = ClsProg(model)
    var o = cls_cat_count_program(pr.program, n, d, K, labels, weighted, prev_words[0] if merge else 0, merge,
                                  prior.choice, prior.priors, is_blocked)
    pr.own(arr.value)
    own_partial_labels(pr, pc)
    if weighted:
        pr.own(w)
    if merge:
        pr.own(prev_owners[0])
    pr.run(mode, context)
    partial_refuse(pr, o.neg, fields, pc, context)
    var ncat = cls_cat_categories(pr.base, o.st, d)
    var mc = py_item(fields, nk[].k_min_categories)
    var has_min = mc is not None
    var mins = List[Int]()
    if has_min:
        if Bool(py=py_call(cls_builtins().isinstance, mc, py_item(context, nk[].k_integral_type))):
            mins = List[Int](length=d, fill=Int(py=py_call(cls_builtins().int, mc)))
        else:
            mins = _int_list(mc, context)
    var prev_ncat = List[Int]()
    if merge:
        var old = I64P(unsafe_from_address=Int(py=classical_address(py_item(fields, nk[].k_n_categories_))))
        for j in range(d):  # driver-loop: over the feature columns (model structure)
            prev_ncat.append(Int(old[j]))
    var cmax = cls_cat_bounds(ncat, has_min, mins, prev_ncat)
    var class_count = pr.get(o.cnt, K, 0, context)
    var prev_cmax = 0
    var table_words = List[Int]()
    var table_owners = List[PythonObject]()
    if merge:
        prev_cmax = Int(py=py_item(fields, nk[].k__cmax))
        if prev_cmax > 0:
            _prev_block(fields, "_cc", table_words, table_owners, context)
    var q = ClsProg(model)
    var t = cls_cat_table_program(q.program, n, d, K, labels, ncat, cmax, cls_float(py_item(fields, nk[].k_alpha)),
                                  weighted, merge, prev_cmax, table_words[0] if len(table_words) > 0 else 0,
                                  is_blocked)
    q.own(arr.value)
    own_partial_labels(q, pc)
    q.own(class_count)
    if weighted:
        q.own(w)
    if len(table_owners) > 0:
        q.own(table_owners[0])
    q.run(mode, context)
    partial_refuse(q, t.neg, fields, pc, context)
    var words = d * K * cmax
    _set(fields, "_cc", q.get(t.cc, words, 0, context))
    _set(fields, "category_count_", _cat_blocks(q, t.cc, d, K, cmax, ncat, context))
    var counts = classical_typed(Python.tuple(d), "<i8", py_item(context, nk[].k_output_context))
    var cp = I64P(unsafe_from_address=Int(py=classical_address(counts)))
    _ = counts  # keep the Python object alive through the native call above
    for j in range(d):  # driver-loop: over the feature columns (model structure)
        cp[j] = Int64(ncat[j])
        _ = counts  # keep the Python object alive through the native call above
    _set(fields, "n_categories_", counts)
    _set(fields, "_flp", q.get(t.flp, words, 0, context))
    _set(fields, "_cmax", PythonObject(cmax))
    _set(fields, "feature_log_prob_", _cat_blocks(q, t.flp, d, K, cmax, ncat, context))
    _set(fields, "class_count_", class_count)
    _set(fields, "class_log_prior_", pr.get(o.clp, K, 0, context))
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
    var nk = prep_keys()
    _check_alpha(model, fields, context)
    var arr = cls_x2d(args[0], context)
    var pc = partial_codes(model, fields, args[1], args[2], arr.n, context)
    if pc.first:
        return _cat_fit(model, fields, arr, pc, args[3], cls_mode(context), False, context)
    check_width(model, fields, arr.d)
    return _cat_fit(model, fields, arr, pc, args[3], py_item(fields, nk[].k_numeric_mode_), True, context)


# ---------------------------------------------------------------- discriminant analysis


def _estimator_covs(est: PythonObject, arr: XIn, codes: PythonObject, K: Int, who: String,
                    context: PythonObject) raises -> List[Float64]:
    """A user covariance estimator would be fitted by its Python `fit` inside
    the runtime, so it is refused by name; LDA and QDA refuse earlier, and
    this guard keeps any remaining caller from calling it."""
    # TOMBSTONE: the per-class Python `est.fit` loop (QDA/LDA
    # covariance_estimator objects) deleted 2026-10-24 by lane
    # fix-round2-host-20261024 (no Python in the runtime); code recoverable at
    # 809a28acb. Restore: none; no compiled covariance-estimator API exists.
    raise Error("TypeError: covariance_estimator cannot invoke Python callbacks; use covariance_estimator=None and the native solver/shrinkage configuration")


def _prior_list(fields: PythonObject, K: Int, who: String, context: PythonObject) raises -> List[Float64]:
    """UNPACK of the checked `priors` parameter of a discriminant fit
    (empty: None)."""
    var nk = prep_keys()
    if py_item(fields, nk[].k_priors) is None:
        return List[Float64]()
    return given_priors(py_item(fields, nk[].k_priors), K, who, False, context)


def _sized_attr(fields: PythonObject, name: String, words: Int, who: String, context: PythonObject) raises -> PythonObject:
    """UNPACK of one fitted block a typed program reads as `words` words."""
    var value = f32_input(fields[PythonObject(name)], context)
    if Int(py=value.size) != words:
        raise Error("ValueError: " + who + " " + name + " does not hold its fitted size")
    return value


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
    """`LinearDiscriminantAnalysis.fit`; returns the admitted X
    (fit_transform's input). UNPACK: the checked parameters, X, the class
    codes, the image's mode words (`cls_da_mode`, once). RUN: the typed fit
    program (`cls_lda_svd_program`, or `_lda_cov_solver`) through
    `ClsProg.run`. PACK: the fitted attributes."""
    var nk = prep_keys()
    var b = cls_builtins()
    var solver = py_item(fields, nk[].k_solver)
    if not (da_solver_is(fields, "svd") or da_solver_is(fields, "lsqr") or da_solver_is(fields, "eigen")):
        raise Error("mojolearn: invalid solver " + cls_repr(solver))
    if da_solver_is(fields, "svd") and py_item(fields, nk[].k_shrinkage) is not None:
        raise Error("NotImplementedError: mojolearn: shrinkage not supported with 'svd' solver.")
    var est = py_item(fields, nk[].k_covariance_estimator)
    if est is not None and da_solver_is(fields, "svd"):
        raise Error("mojolearn: covariance estimator is not supported with svd solver. Try another solver")
    da_check_estimator_shrinkage(est, py_item(fields, nk[].k_shrinkage))
    var shr: Optional[Float64] = None
    if est is None:
        shr = da_shrinkage(py_item(fields, nk[].k_shrinkage), context)
    var arr = cls_x2d(X, context)
    var n = arr.n
    var d = arr.d
    var codes = encode_y(model, fields, y, n, context)
    var K = _n_classes(fields)
    if K < 2 or n <= K:
        raise Error("mojolearn: LinearDiscriminantAnalysis needs at least two classes and more samples than classes")
    var maxc = min(K - 1, d)
    var nc = py_item(fields, nk[].k_n_components)
    if nc is not None and Bool(py=nc > PythonObject(maxc)):
        raise Error("mojolearn: n_components cannot be larger than min(n_features, n_classes - 1)")
    var mode = cls_mode(context)
    var da = cls_da_mode(mode, context)
    if not da_solver_is(fields, "svd"):
        _lda_cov_solver(model, fields, arr, codes, K, d, n, mode, da, shr, context)
        return arr^
    var tol = cls_float(py_item(fields, nk[].k_tol))
    var priors_given = _prior_list(fields, K, "LinearDiscriminantAnalysis", context)
    var pr = ClsProg(model)
    var o = cls_lda_svd_program(pr.program, da, n, d, K, tol, priors_given,
                                Bool(py=py_item(fields, nk[].k_store_covariance)))
    # the buffer records' owners, in record order
    pr.own(arr.value)
    pr.own(codes)
    pr.run(mode, context)
    if o.sw != NONE:
        _set(fields, "covariance_", pr.get(o.sw, d, d, context))
    var rank2 = Int(pr.value(o.meta + 2))
    _set(fields, "_rank", PythonObject(rank2))
    _set(fields, "means_", pr.get(o.mean, K, d, context))
    _set(fields, "priors_", pr.get(o.priors, K, 0, context))
    _set(fields, "xbar_", pr.get(o.xbar, d, 0, context))
    _set(fields, "_scal_full", pr.get(o.scal, d, d, context))
    # the rank columns of the (d, d) scalings (Array.__getitem__'s strided copy)
    var scalings = classical_wrap_array(native_array_empty("f"), Python.tuple(d, 0), "<f4", "C", py_item(context, nk[].k_output_context))
    if rank2 > 0:
        scalings = classical_typed(Python.tuple(d, rank2), "<f4", py_item(context, nk[].k_output_context))
        var dest = F32P(unsafe_from_address=Int(py=classical_address(scalings)))
        _ = scalings  # keep the Python object alive through the native call above
        var src = F32P(unsafe_from_address=pr.base) + o.scal
        for i in range(d):  # driver-loop: over the scalings' rows (model structure)
            unsafe_memcpy(dest=dest + i * rank2, src=src + i * d, count=rank2)
            _ = scalings  # keep the Python object alive through the native call above
    _set(fields, "scalings_", scalings)
    _lda_publish_coef(pr, fields, K, d, o.coef, o.inter, o.cd, o.ci, context)
    var keep = maxc if nc is None else Int(py=py_call(b.int, nc))
    _set(fields, "_max_components", PythonObject(keep))
    _set(fields, "explained_variance_ratio_", pr.get(o.evr, keep, 0, context))
    _set(fields, "numeric_mode_", mode)
    _set(fields, "n_features_in_", PythonObject(d))
    return arr^


def _lda_cov_solver(model: PythonObject, fields: PythonObject, arr: XIn, codes: PythonObject, K: Int, d: Int, n: Int,
                    mode: PythonObject, da: ClsDaMode, shr: Optional[Float64], context: PythonObject) raises:
    """solver 'lsqr' / 'eigen' (`_fit_cov_solver`): the typed
    `cls_lda_cov_program`."""
    var nk = prep_keys()
    var b = cls_builtins()
    var priors_given = _prior_list(fields, K, "LinearDiscriminantAnalysis", context)
    var eigen = da_solver_is(fields, "eigen")
    var est = py_item(fields, nk[].k_covariance_estimator)
    if est is not None:
        # refused by name before any stage (it was refused where its
        # covariances would have been staged; nothing ran either way)
        _ = _estimator_covs(est, arr, codes, K, "LinearDiscriminantAnalysis", context)
    var has_shrinkage = False
    var shrinkage = 0.0
    if shr:
        has_shrinkage = True
        shrinkage = shr.value()
    var pr = ClsProg(model)
    var o = cls_lda_cov_program(pr.program, da, n, d, K, eigen, has_shrinkage, shrinkage, priors_given)
    # the buffer records' owners, in record order
    pr.own(arr.value)
    pr.own(codes)
    if eigen:
        # The one-class code vector: a zeroed float32 Array put as it is.
        pr.own(f32_array(n, context))
    var nc = py_item(fields, nk[].k_n_components)
    var mc = min(K - 1, d) if nc is None else Int(py=py_call(b.int, nc))
    pr.run(mode, context)
    if eigen:
        for j in range(d):  # driver-loop: over the within-class covariance eigenvalues (feature count)
            if pr.value(o.e + j) <= 0:
                raise Error("mojolearn: the within-class covariance is not positive definite (the reference's eigh(Sb, Sw) fails); set shrinkage")
    _set(fields, "means_", pr.get(o.mean, K, d, context))
    _set(fields, "priors_", pr.get(o.priors, K, 0, context))
    _set(fields, "xbar_", pr.get(o.xbar, d, 0, context))
    _set(fields, "covariance_", pr.get(o.sw, d, d, context))
    _lda_publish_coef(pr, fields, K, d, o.coef, o.inter, o.cd, o.ci, context)
    _set(fields, "_max_components", PythonObject(mc))
    if eigen:
        var full = pr.get(o.scal, d, d, context)
        _set(fields, "scalings_", full)
        _set(fields, "_scal_full", full)
        _set(fields, "_rank", PythonObject(d))
        _set(fields, "explained_variance_ratio_", pr.get(o.evr, mc, 0, context))
    _set(fields, "numeric_mode_", mode)
    _set(fields, "n_features_in_", PythonObject(d))


def _lda_transform(model: PythonObject, fields: PythonObject, arr: XIn, context: PythonObject) raises -> PythonObject:
    """`LinearDiscriminantAnalysis.transform` of an admitted X: the typed
    `cls_lda_transform_program`."""
    var nk = prep_keys()
    check_fitted(model, fields)
    check_width(model, fields, arr.d)
    var n = arr.n
    var d = arr.d
    if da_solver_is(fields, "lsqr"):
        raise Error("NotImplementedError: mojolearn: transform not implemented for 'lsqr' solver (use 'svd' or 'eigen').")
    var mc = min(Int(py=py_item(fields, nk[].k__max_components)), Int(py=py_item(fields, nk[].k__rank)))
    var mode = py_item(fields, nk[].k_numeric_mode_)
    var eigen = da_solver_is(fields, "eigen")
    var xbar = _sized_attr(fields, "xbar_", d, "LinearDiscriminantAnalysis", context)
    var scal = _sized_attr(fields, "_scal_full", d * d, "LinearDiscriminantAnalysis", context)
    # C04_LDA (-D MOJOLEARN_CLASSICAL_C04_LDA): the same center_rows -> matmul arithmetic in the consumer.
    var centered = not eigen and (classical_shared(mode, context) & 4) != 0
    var pr = ClsProg(model)
    var dst = cls_lda_transform_program(pr.program, n, d, mc, eigen, centered)
    pr.own(arr.value)
    pr.own(xbar)
    pr.own(scal)
    pr.run(mode, context)
    return pr.get(dst, n, mc, context)


def _qda_fit(model: PythonObject, fields: PythonObject, X: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`QuadraticDiscriminantAnalysis.fit`. UNPACK: the checked parameters,
    X, the class codes, the image's mode words (`cls_da_mode`, once). RUN:
    the typed fit program (`cls_qda_fit_program`) through `ClsProg.run`.
    PACK: the count and rank refusals, the fitted attributes."""
    var nk = prep_keys()
    if py_item(fields, nk[].k_covariance_estimator) is not None:
        # Fitting a user covariance estimator runs its Python `fit` inside
        # the runtime. Refused by name before any other check, as LDA does
        # (a5f9cc738, bindings/prep_lda_admission.mojo).
        raise Error("TypeError: covariance_estimator cannot invoke Python callbacks; use covariance_estimator=None and the native solver/shrinkage configuration")
    var solver = py_item(fields, nk[].k_solver)
    if not (da_solver_is(fields, "svd") or da_solver_is(fields, "eigen")):
        raise Error("mojolearn: invalid solver " + cls_repr(solver))
    if da_solver_is(fields, "svd") and py_item(fields, nk[].k_shrinkage) is not None:
        raise Error("NotImplementedError: mojolearn: shrinkage not supported with 'svd' solver.")
    var shr = da_shrinkage(py_item(fields, nk[].k_shrinkage), context)
    var has_shrinkage = False
    var shrinkage = 0.0
    if shr:
        has_shrinkage = True
        shrinkage = shr.value()
    var eigen = da_solver_is(fields, "eigen")
    var arr = cls_x2d(X, context)
    var n = arr.n
    var d = arr.d
    var codes = encode_y(model, fields, y, n, context)
    var K = _n_classes(fields)
    if K < 2:
        raise Error("mojolearn: QuadraticDiscriminantAnalysis needs at least two classes")
    var mode = cls_mode(context)
    var da = cls_da_mode(mode, context)
    var reg_param = 0.0 if eigen else cls_float(py_item(fields, nk[].k_reg_param))
    var priors_given = _prior_list(fields, K, "QuadraticDiscriminantAnalysis", context)
    var pr = ClsProg(model)
    var o = cls_qda_fit_program(pr.program, da, n, d, K, eigen, reg_param, has_shrinkage, shrinkage, priors_given,
                                Bool(py=py_item(fields, nk[].k_store_covariance)))
    # the buffer records' owners, in record order
    pr.own(arr.value)
    pr.own(codes)
    pr.run(mode, context)
    # the count check reads the program's own class counts right after the run
    for k in range(K):  # driver-loop: over the class counts (class count)
        if pr.value(o.cnt + k) < 2:
            raise Error("mojolearn: y has only 1 sample in a class, covariance is ill defined")
    var tol = cls_float(py_item(fields, nk[].k_tol))
    var classes = py_item(fields, nk[].k_classes_)
    for k in range(K):  # driver-loop: rank check per class before raising (class count)
        var above = 0
        for j in range(d):  # driver-loop: over one class's scalings (feature count)
            if pr.value(o.s2 + k * d + j) > tol:
                above += 1
        if above < d:
            raise Error("mojolearn: the covariance matrix of class " + cls_repr(classes[k]) + " is not full rank. Increase the value of `" + (String("shrinkage") if eigen else String("reg_param")) + "` to reduce the collinearity.")
    if o.keep != NONE:
        var covs = Python.list()
        for k in range(K):  # driver-loop: per-class covariance blocks (class count)
            _ = py_call(covs.append, pr.get(o.keep + k * d * d, d, d, context))
        _set(fields, "covariance_", covs)
    _set(fields, "means_", pr.get(o.mean, K, d, context))
    _set(fields, "priors_", pr.get(o.priors, K, 0, context))
    var rotations = Python.list()
    var scalings = Python.list()
    for k in range(K):  # driver-loop: per-class rotation and scaling blocks (class count)
        _ = py_call(rotations.append, pr.get(o.evec + k * d * d, d, d, context))
        _ = py_call(scalings.append, pr.get(o.s2 + k * d, d, 0, context))
    _set(fields, "rotations_", rotations)
    _set(fields, "scalings_", scalings)
    _set(fields, "_rot", pr.get(o.rot, K * d * d, 0, context))
    _set(fields, "_logc", pr.get(o.logc, K, 0, context))
    _set(fields, "numeric_mode_", mode)
    _set(fields, "n_features_in_", PythonObject(d))
    return model


# ---------------------------------------------------------------- the door


def _kind_of(model: PythonObject, context: PythonObject) raises -> String:
    var nk = prep_keys()
    # `cls_kind`: an exact public class is the MRO's first row, answered by
    # one lookup without the MRO read; a subclass walks the MRO (perf2-prep).
    var own = py_get(py_item(context, nk[].k_cls_kinds), py_type(model))
    if own is not None:
        return String(own)
    return cls_kind(model, context)


def _dispatch(model: PythonObject, op: Int, operation: String, var args: PythonObject, var context: PythonObject,
              known: String = "", given: Optional[PythonObject] = None) raises -> PythonObject:
    var nk = prep_keys()
    # `known`: the kind the caller already resolved for this model (the
    # public naive Bayes / QDA door), so the MRO is not read twice.
    if py_type(py_type(model)) is not py_type(py_type(PythonObject(0))):
        raise Error("TypeError: native classifiers cannot invoke Python metaclass hooks")
    var kind = known if known != "" else _kind_of(model, context)
    var fields: PythonObject
    if kind == "lda":
        context = lda_native_context(context)
        if op == PREP_OPN_INITIALIZE:
            var built = lda_native_initialize(model, args, context)
            # A constructor run again on a live instance: its next data
            # operation resolves the device route from the new `device`.
            scaler_route_clear(model)
            return built
        # One admission pass per fitted state (a per-model record), and the
        # plain classes_ list it read is reused below instead of a second copy.
        var admitted = lda_native_fields_plain(model, context, op == PREP_OPN_IS_FITTED or op == PREP_OPN_GET_PARAMS or op == PREP_OPN_SET_PARAMS)
        fields = admitted[0]
        if op == PREP_OPN_IS_FITTED:
            return classical_is_fitted(model)
        if op == PREP_OPN_GET_PARAMS:
            return lda_native_get_params(fields, context)
        if op == PREP_OPN_SET_PARAMS:
            var updated = lda_native_set_params(model, fields, args[0], context)
            # The device may have changed (checked before it was stored):
            # the next data operation resolves the route again.
            scaler_route_clear(model)
            return updated
        if op == PREP_OPN_FIT or op == PREP_OPN_FIT_TRANSFORM or op == PREP_OPN_SCORE:
            args = Python.tuple(args[0], prep_plain_classes(args[1], context))
            lda_native_labels(args[1], context, op != PREP_OPN_SCORE)
        if op != PREP_OPN_FIT and op != PREP_OPN_FIT_TRANSFORM:
            lda_require_fitted(model, operation, context)
            var plain = admitted[1]
            fields = py_call(native_loaded_module("builtins").dict, fields)
            if plain is None:
                plain = prep_plain_classes(py_item(fields, nk[].k_classes_), context)
            py_setitem(fields, nk[].k_classes_, plain)
    elif kind == "gnb" and py_get(context, nk[].k_cv_fixed_owner) is not None:
        context = fixed_owner_context(context)
        fields = fixed_owner_fields(model, context)
        if op == PREP_OPN_CV_SET_PARAMS:
            return fixed_owner_set_params(model, args[0], context)
        for i in range(len(args)):  # driver-loop: fixed public arguments, including native resident-fold handles
            if i == 0 and py_type(args[i]) is py_item(context, nk[].k_maxabs_rows_class):
                _ = _rows_fields(args[i], context)
            else:
                classical_admit_value(args[i], context)
        if op == PREP_OPN_FIT or op == PREP_OPN_SCORE:
            lda_native_labels(args[1], context, op == PREP_OPN_FIT, "GaussianNB")
    elif given:
        # A read-only public operation's private fields view (classes_ as the
        # plain list), never written back into the model (p3-prep).
        fields = given.value()
    else:
        fields = py_fields(model)
    if op == PREP_OPN_PREDICT:
        return _predict(model, fields, kind, args[0], context)
    if op == PREP_OPN_PREDICT_PROBA:
        return _predict_proba(model, fields, kind, args[0], context)
    if op == PREP_OPN_PREDICT_LOG_PROBA:
        return _predict_log_proba(model, fields, kind, args[0], context)
    if op == PREP_OPN_PREDICT_JOINT_LOG_PROBA:
        return _joint_log_proba(model, fields, kind, args[0], context)
    if op == PREP_OPN_SCORE:
        return _score(model, fields, kind, args[0], args[1], context)
    if op == PREP_OPN_DECISION_FUNCTION and (kind == "lda" or kind == "qda"):
        return _decision_function(model, fields, kind, args[0], context)
    if op == PREP_OPN_FIT:
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
    if op == PREP_OPN_PARTIAL_FIT:
        if kind == "gnb":
            return _gnb_partial_fit(model, fields, args, context)
        if kind == "mnb" or kind == "bnb" or kind == "cnb":
            return _dnb_partial_fit(model, fields, kind, args, context)
        if kind == "cat":
            return _cat_partial_fit(model, fields, args, context)
    if kind == "lda" and op == PREP_OPN_TRANSFORM:
        check_fitted(model, fields)
        return _lda_transform(model, fields, cls_x2d(args[0], context), context)
    if kind == "lda" and op == PREP_OPN_FIT_TRANSFORM:
        # The X converted for the fit is the transform's input.
        var arr = _lda_fit(model, fields, args[0], args[1], context)
        return _lda_transform(model, fields, arr, context)
    raise Error("TypeError: unknown native classifier operation " + operation)


def _public_kind(model: PythonObject, context: PythonObject) raises -> String:
    """The kind of a naive Bayes / QDA model whose public operation runs the
    native lifecycle here ("" for LDA, which keeps its own, and a metaclass
    the door refuses). The caller answers "" itself for the CV fixed owner's
    private calls (its one read of that context key)."""
    var b = native_loaded_module("builtins")
    if py_type(py_type(model)) is not b.type:
        return ""
    var kind = _kind_of(model, context)
    if kind == "gnb" or kind == "mnb" or kind == "bnb" or kind == "cnb" or kind == "cat" or kind == "qda":
        return kind
    return ""


def _classifier_public(model: PythonObject, kind: String, op: Int, operation: String, args: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    """GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB and
    QDA: the former `_mode` wrappers (NotFittedError, X refusals, width and
    feature names, y admission, fit bookkeeping with ClassLabels `classes_`,
    NumPy results outside the package) around the algorithm. The solvers read
    `classes_` as a plain list. A read-only operation (predict*, score,
    decision_function) gets a private copy of the fields dict holding the
    plain list, so the model is never mutated and a concurrent caller never
    sees a swapped state (p3-prep); a fit / partial_fit swaps it in the
    model (its solvers write the fitted fields there) and back after."""
    var nk = prep_keys()
    var source = args[0] if len(args) > 0 else PythonObject(None)
    # MultinomialNB and ComplementNB take CSR X (`_DiscreteNB._csr_ok`).
    var sparse_ok = kind == "mnb" or kind == "cnb"
    var depth = lifecycle_begin(model, operation, source, True, sparse_ok, True, context)
    var fields = py_fields(model)
    var public_labels = py_get(fields, nk[].k_classes_)
    var plain = prep_plain_classes(public_labels, context)
    var fitting = lifecycle_fitting(operation)
    var swapped = plain is not public_labels and not (
        op == PREP_OPN_PREDICT or op == PREP_OPN_PREDICT_PROBA or op == PREP_OPN_PREDICT_LOG_PROBA
        or op == PREP_OPN_PREDICT_JOINT_LOG_PROBA or op == PREP_OPN_SCORE or op == PREP_OPN_DECISION_FUNCTION)
    var result = PythonObject(None)
    try:
        if fitting and len(args) > 1 and depth == 0:
            lda_native_labels(args[1], lda_native_context(context), True, classical_estimator_name(model))
        if swapped:
            py_setitem(fields, nk[].k_classes_, plain)
        var reading = (op == PREP_OPN_PREDICT or op == PREP_OPN_PREDICT_PROBA or op == PREP_OPN_PREDICT_LOG_PROBA
                       or op == PREP_OPN_PREDICT_JOINT_LOG_PROBA or op == PREP_OPN_SCORE
                       or op == PREP_OPN_DECISION_FUNCTION)
        if reading and plain is not public_labels:
            var view = py_call(native_loaded_module("builtins").dict, fields)
            py_setitem(view, nk[].k_classes_, plain)
            result = _dispatch(model, op, operation, args, context, kind, view)
        else:
            result = _dispatch(model, op, operation, args, context, kind)
    finally:
        lifecycle_end(model, operation, depth)
        if swapped:
            var after = py_fields(model)
            if py_get(after, nk[].k_classes_) is plain:
                py_setitem(after, nk[].k_classes_, public_labels)
    lifecycle_complete(model, operation, source, True, context)
    return lifecycle_result(model, operation, result, "classifier", context)


def prep_classify_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var model = call[0]
        var nk = prep_keys()
        # The code of the operation (one lookup with the caller's str); the
        # name String stays for the lifecycle owner and the refusal texts.
        var op = py_op_code(nk[].ops, call[1])
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        if op == PREP_OPN_NB_CSR_READY:
            # a class-level probe: `model` is the class, args[0] its `_csr_ok`
            return PythonObject(_nb_csr_ready_op(Bool(py=args[0]), context)).steal_data()
        # The connected context of the estimator's device route: the
        # owner's own, or the CPU one import connected beside it on a GPU
        # installation (resolved once per estimator; a hit is one record
        # read and one pointer compare). The door reads no device and no
        # process vendor; every image below is selected from this context.
        # Not routed: the CV fixed owner's private calls (its own context;
        # the member's device is that owner's), the constructor (the device
        # is not stored yet) and set_params (a new device is checked against
        # the installation, not the route); the route is cleared after both.
        var fixed = py_get(context, nk[].k_cv_fixed_owner) is not None
        if not fixed and op != PREP_OPN_INITIALIZE and op != PREP_OPN_SET_PARAMS:
            context = scaler_route_context(model, context)
        classical_connect_numpy(context)
        if not fixed and op < 0 and operation == "vendor_used":
            # `vendor_used()`: the image of this estimator's device route,
            # read back (a name outside the operation table: one integer
            # test on a data operation; no input, no lifecycle step).
            return cls_vendor_used(model, context).steal_data()
        var public_kind = String("") if fixed else _public_kind(model, context)
        if public_kind != "":
            return _classifier_public(model, public_kind, op, operation, args, context).steal_data()
        var result = _dispatch(model, op, operation, args, context)
        if py_type(model) is py_get(context, nk[].k_lda_native_class):
            if op == PREP_OPN_FIT or op == PREP_OPN_FIT_TRANSFORM:
                classifier_after_fit(py_fields(model), args[0], context)
            # predict: one result for every caller (lane predict-dtype, 2026-11-22).
            result = _predict_result(result, context) if op == PREP_OPN_PREDICT else _public_result(result, context)
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

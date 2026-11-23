# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Discriminant family: its place in
`bindings/member_boundary_door.mojo` (recipe: MEMBER-ABI-V2.md; worked form
providers/python/member_naive_bayes_boundary_door.mojo). Kind 0 is
LinearDiscriminantAnalysis (`_mojolearn_member = (MEMBER_FAMILY_DISCRIMINANT, 0)`
on the class), kind 1 QuadraticDiscriminantAnalysis; a kind the member entry
does not serve (core/discriminant_member.mojo `discriminant_kind_published`)
is not published (admit answers False and the caller keeps its door).

  * `discriminant_member_admit` (UNPACK, once per member): the estimator's
    own parameter checks in its fit door's words
    (bindings/prep_classify_native_door.mojo `_lda_fit`, `_qda_fit`), the
    process mode its fit runs in, the x_prep image that fit selects (the
    NaiveBayes boundary's `_route`), the image's mode words
    (`cls_da_mode`, the C04 bit) and the parameter words of
    core/discriminant_member.mojo. False (the caller keeps the door): an
    `n_components` below one.
  * `discriminant_member_pack` (PACK): the fitted attributes the
    estimator's own fit writes, from the member state's blocks; the class
    labels through the label hook (`member_labels_take`).
  * `discriminant_member_state` (UNPACK): a member state from a fitted
    estimator.

SERIAL BY NATURE: one member's admission or publication (field reads and
one copy per fitted block); a caller runs its members in parallel.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.member_owner_unpack import MemberOwner, member_labels_take
from bindings.py_balanced import py_call, py_contains, py_item, py_setitem, py_type
from bindings.classical_arrays_door import classical_address, classical_typed, classical_wrap_array
from bindings.native_array_storage_door import native_array_empty
from bindings.native_loaded_module_door import native_loaded_module
from bindings.prep_keys_unpack import prep_keys
from bindings.prep_classify_prog_door import f32_input
from bindings.prep_classify_common_door import (
    cls_mode, cls_float, cls_repr, blocked, float_list, check_priors, classical_shared, cls_da_mode,
    da_shrinkage, da_check_estimator_shrinkage, da_solver_is, lda_n_components,
)
from bindings.prep_classify_prog_run import CLS_LDA_SVD, CLS_LDA_LSQR, CLS_LDA_EIGEN
from bindings.prep_lda_admission_door import lda_native_context
from bindings.x_prep_native_owner_run import xprep_r3
from core.hostptr import f32_ptr
from core.discriminant_member import (
    DA_KIND_LDA, DA_KIND_QDA, DA_PARAM_HEAD, DA_BLOCK_MEANS, DA_BLOCK_PRIORS, DA_BLOCK_XBAR, DA_BLOCK_COEF,
    DA_BLOCK_INTER, DA_BLOCK_SCAL, DA_BLOCK_EVR, DA_BLOCK_COV, DA_BLOCK_PAIR_COEF, DA_BLOCK_PAIR_INTER,
    DA_BLOCK_ROT, DA_BLOCK_LOGC, DA_BLOCK_ROTATIONS, DA_BLOCK_SCALINGS, discriminant_kind_published,
)
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_IMPORT_DONE, MEMBER_F64,
)
from providers.python.member_naive_bayes_boundary_door import _route, _export, _array

comptime _ESTIMATOR_REFUSAL = "TypeError: covariance_estimator cannot invoke Python callbacks; use covariance_estimator=None and the native solver/shrinkage configuration"


def _kind_name(kind: Int) -> String:
    if kind == DA_KIND_QDA:
        return String("QuadraticDiscriminantAnalysis")
    return String("LinearDiscriminantAnalysis")


def discriminant_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. False: this class has no member entry, or
    its settings stay on its door."""
    if not discriminant_kind_published(kind):
        return False
    var nk = prep_keys()
    var who = _kind_name(kind)
    var lda = kind == DA_KIND_LDA
    # LDA's door checks its parameters against the concrete type tuples of
    # its typed context (built once per class context).
    var checks = lda_native_context(context) if lda else context
    var est = py_item(fields, nk[].k_covariance_estimator)
    if not lda and est is not None:
        raise Error(_ESTIMATOR_REFUSAL)
    var svd = da_solver_is(fields, "svd")
    var eigen = da_solver_is(fields, "eigen")
    var lsqr = lda and da_solver_is(fields, "lsqr")
    if not (svd or eigen or lsqr):
        raise Error("mojolearn: invalid solver " + cls_repr(py_item(fields, nk[].k_solver)))
    if svd and py_item(fields, nk[].k_shrinkage) is not None:
        raise Error("NotImplementedError: mojolearn: shrinkage not supported with 'svd' solver.")
    if est is not None and svd:
        raise Error("mojolearn: covariance estimator is not supported with svd solver. Try another solver")
    da_check_estimator_shrinkage(est, py_item(fields, nk[].k_shrinkage))
    if est is not None:
        raise Error(_ESTIMATOR_REFUSAL)
    var shr = da_shrinkage(py_item(fields, nk[].k_shrinkage), checks)
    var sixth = -1.0
    if lda:
        # None: -1; below 1: scikit-learn's own refusal (lane mf-lin6, `lda_n_components`).
        sixth = Float64(lda_n_components(py_item(fields, nk[].k_n_components)))
    elif not eigen:
        sixth = cls_float(py_item(fields, nk[].k_reg_param))
    else:
        sixth = 0.0
    var priors = List[Float64]()
    if py_item(fields, nk[].k_priors) is not None:
        priors = float_list(py_item(fields, nk[].k_priors), checks)
        if len(priors) == 0:
            raise Error("mojolearn: " + who + ": number of priors must match number of classes")
        check_priors(priors, who, False)
    var mode = cls_mode(context)
    _route(model, mode, blocked(), context, owner)
    var da = cls_da_mode(mode, context)
    var words = List[Float64](length=DA_PARAM_HEAD + len(priors), fill=0.0)
    words[0] = Float64(kind)
    words[1] = Float64(CLS_LDA_SVD if svd else (CLS_LDA_EIGEN if eigen else CLS_LDA_LSQR))
    if shr:
        words[2] = 1.0
        words[3] = shr.value()
    words[4] = cls_float(py_item(fields, nk[].k_tol))
    words[5] = 1.0 if Bool(py=py_item(fields, nk[].k_store_covariance)) else 0.0
    words[6] = sixth
    words[7] = 1.0 if da.stats_blocked else 0.0
    words[8] = 1.0 if da.class_onepass else 0.0
    words[9] = Float64(da.gram)
    words[10] = 1.0 if xprep_r3("work") else 0.0
    words[11] = 1.0 if (lda and (classical_shared(mode, context) & 4) != 0) else 0.0
    words[12] = Float64(len(priors))
    for i in range(len(priors)):  # driver-loop: over the user prior list (class-count sized)
        words[DA_PARAM_HEAD + i] = priors[i]
    owner.plan.params = words^
    return True


def _piece(address: Int, rows: Int, cols: Int, matrix: Bool, context: PythonObject) raises -> PythonObject:
    # One class's part of a per-class block, as the door's `ClsProg.get`.
    return _array(MemberMatrix(address, rows, cols), matrix, context)


def _put(fields: PythonObject, name: String, plan: MemberPlan, mut call: MemberCall, block: Int, matrix: Bool,
    context: PythonObject) raises:
    py_setitem(fields, PythonObject(name), _array(_export(plan, call, block), matrix, context))


def discriminant_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the door's fit does (`_lda_fit`, `_lda_cov_solver`, `_qda_fit`)."""
    if not discriminant_kind_published(plan.kind):
        raise Error("ValueError: mojolearn: Discriminant member kind " + String(plan.kind) + " is not published")
    var labels = member_labels_take(fields, b)
    if labels is not None:
        py_setitem(fields, "classes_", labels)
    var call = MemberCall(plan.route, state)
    var scalars = _export(plan, call, MEMBER_BLOCK_SCALARS)
    var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars.data)
    var d = Int(words[0])
    var K = Int(words[1])
    var rank = Int(words[2])
    var components = Int(words[3])
    # The mode word the fit ran in: the route's image word.
    var mode_word = String("identical") if plan.route.mode == 1 else String("fast")
    var solver = Int(plan.params[1])
    if plan.kind == DA_KIND_QDA:
        var kept = _export(plan, call, DA_BLOCK_COV)
        if kept.rows > 0:
            var covs = Python.list()
            for k in range(K):  # driver-loop: per-class covariance blocks (class count)
                _ = py_call(covs.append, _piece(kept.data + 4 * k * d * d, d, d, True, context))
            py_setitem(fields, "covariance_", covs)
        _put(fields, "means_", plan, call, DA_BLOCK_MEANS, True, context)
        _put(fields, "priors_", plan, call, DA_BLOCK_PRIORS, False, context)
        var evec = _export(plan, call, DA_BLOCK_ROTATIONS)
        var s2 = _export(plan, call, DA_BLOCK_SCALINGS)
        var rotations = Python.list()
        var scalings = Python.list()
        for k in range(K):  # driver-loop: per-class rotation and scaling blocks (class count)
            _ = py_call(rotations.append, _piece(evec.data + 4 * k * d * d, d, d, True, context))
            _ = py_call(scalings.append, _piece(s2.data + 4 * k * d, 1, d, False, context))
        py_setitem(fields, "rotations_", rotations)
        py_setitem(fields, "scalings_", scalings)
        _put(fields, "_rot", plan, call, DA_BLOCK_ROT, False, context)
        _put(fields, "_logc", plan, call, DA_BLOCK_LOGC, False, context)
        py_setitem(fields, "numeric_mode_", PythonObject(mode_word))
        py_setitem(fields, "n_features_in_", PythonObject(d))
        py_setitem(fields, "_mojolearn_fitted", True)
        return
    var cov = _export(plan, call, DA_BLOCK_COV)
    if cov.rows > 0:
        py_setitem(fields, "covariance_", _array(cov, True, context))
    if solver == CLS_LDA_SVD:
        py_setitem(fields, "_rank", PythonObject(rank))
    _put(fields, "means_", plan, call, DA_BLOCK_MEANS, True, context)
    _put(fields, "priors_", plan, call, DA_BLOCK_PRIORS, False, context)
    _put(fields, "xbar_", plan, call, DA_BLOCK_XBAR, False, context)
    if solver != CLS_LDA_LSQR:
        var scal = _export(plan, call, DA_BLOCK_SCAL)
        var full = _array(scal, True, context)
        py_setitem(fields, "_scal_full", full)
        if solver == CLS_LDA_EIGEN:
            py_setitem(fields, "scalings_", full)
            py_setitem(fields, "_rank", PythonObject(d))
        else:
            # the rank columns of the (d, d) scalings (the door's strided copy)
            var scalings = classical_wrap_array(native_array_empty("f"), Python.tuple(d, 0), "<f4", "C",
                                                context["output_context"])
            if rank > 0:
                scalings = classical_typed(Python.tuple(d, rank), "<f4", context["output_context"])
                var dest = f32_ptr(Int(py=classical_address(scalings)))
                var src = f32_ptr(scal.data)
                for i in range(d):  # driver-loop: over the scalings' rows (model structure)
                    unsafe_memcpy(dest=dest + i * rank, src=src + i * d, count=rank)
                _ = scalings  # keep the Python object alive through the copies above
            py_setitem(fields, "scalings_", scalings)
    var coef = _array(_export(plan, call, DA_BLOCK_COEF), True, context)
    var inter = _array(_export(plan, call, DA_BLOCK_INTER), False, context)
    py_setitem(fields, "_coef", coef)
    py_setitem(fields, "_inter", inter)
    if K == 2:
        _put(fields, "coef_", plan, call, DA_BLOCK_PAIR_COEF, True, context)
        _put(fields, "intercept_", plan, call, DA_BLOCK_PAIR_INTER, False, context)
    else:
        py_setitem(fields, "coef_", coef)
        py_setitem(fields, "intercept_", inter)
    py_setitem(fields, "_max_components", PythonObject(components))
    if solver != CLS_LDA_LSQR:
        _put(fields, "explained_variance_ratio_", plan, call, DA_BLOCK_EVR, False, context)
    py_setitem(fields, "numeric_mode_", PythonObject(mode_word))
    py_setitem(fields, "n_features_in_", PythonObject(d))
    py_setitem(fields, "_mojolearn_fitted", True)


def _import(plan: MemberPlan, mut call: MemberCall, fields: PythonObject, name: String, block: Int, rows: Int,
    cols: Int, context: PythonObject) raises:
    var who = _kind_name(plan.kind)
    if not py_contains(fields, PythonObject(name)):
        raise Error("NotFittedError: this " + who + " member needs its fitted " + name)
    var value = f32_input(fields[PythonObject(name)], context)
    if Int(py=value.size) != rows * cols:
        raise Error("ValueError: " + who + " " + name + " does not hold its fitted size")
    call.block = block
    call.flags = 0
    call.x = MemberMatrix(Int(py=classical_address(value)), rows, cols)
    _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    _ = value  # the member copied the words in before the buffer is released


def _native_int(fields: PythonObject, name: String, who: String) raises -> Int:
    var bi = native_loaded_module("builtins")
    var value = fields[PythonObject(name)]
    if py_type(value) is not bi.int:
        raise Error("TypeError: " + who + " " + name + " must be a native integer")
    return Int(py=value)


def discriminant_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`.
    The caller frees it (`member_free`)."""
    if not discriminant_kind_published(plan.kind):
        raise Error("ValueError: mojolearn: Discriminant member kind " + String(plan.kind) + " is not published")
    var who = _kind_name(plan.kind)
    var lda = plan.kind == DA_KIND_LDA
    # the block whose size is the class count
    var sized = String("_inter") if lda else String("_logc")
    if not py_contains(fields, "n_features_in_") or not py_contains(fields, PythonObject(sized)):
        raise Error("NotFittedError: This " + who + " instance is not fitted yet. Call 'fit' with appropriate "
                    + "arguments before using this estimator.")
    var d = _native_int(fields, "n_features_in_", who)
    var K = Int(py=f32_input(fields[PythonObject(sized)], context).size)
    if d < 1 or K < 2:
        raise Error("ValueError: " + who + " does not hold a fitted state")
    var solver = Int(plan.params[1])
    var rank = d
    var components = 0
    if lda:
        components = _native_int(fields, "_max_components", who) if py_contains(fields, "_max_components") else 0
        rank = _native_int(fields, "_rank", who) if py_contains(fields, "_rank") else 0
    var scalars: List[Float64] = [Float64(d), Float64(K), Float64(rank), Float64(components)]
    var call = member_new(plan)
    try:
        if lda:
            _import(plan, call, fields, "_coef", DA_BLOCK_COEF, K, d, context)
            _import(plan, call, fields, "_inter", DA_BLOCK_INTER, 1, K, context)
            if K == 2:
                _import(plan, call, fields, "coef_", DA_BLOCK_PAIR_COEF, 1, d, context)
                _import(plan, call, fields, "intercept_", DA_BLOCK_PAIR_INTER, 1, 1, context)
            if solver != CLS_LDA_LSQR:
                _import(plan, call, fields, "xbar_", DA_BLOCK_XBAR, 1, d, context)
                _import(plan, call, fields, "_scal_full", DA_BLOCK_SCAL, d, d, context)
        else:
            _import(plan, call, fields, "means_", DA_BLOCK_MEANS, K, d, context)
            _import(plan, call, fields, "_rot", DA_BLOCK_ROT, K, d * d, context)
            _import(plan, call, fields, "_logc", DA_BLOCK_LOGC, 1, K, context)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(scalars), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = len(scalars)
    return call.state

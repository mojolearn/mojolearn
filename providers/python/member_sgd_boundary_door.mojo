# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the Sgd family; RUN is core/sgd_member.mojo.
Published kinds: 0 SGDRegressor, 1 PassiveAggressiveRegressor,
2 SGDClassifier, 3 Perceptron, 4 PassiveAggressiveClassifier,
5 SGDOneClassSVM (`_mojolearn_member = (MEMBER_FAMILY_SGD, kind)` on each
class).

  * `sgd_member_admit` (UNPACK, once per member): the estimator's own
    refusals (bindings/x_linear_sgd_unpack.mojo: `_xl_sgd_refuse`, the
    kind's loss and C / nu checks, then `_xl_sgd_build`'s penalty,
    learning-rate, eta0, alpha, seed and batch_size words), the estimator's
    own route (bindings/x_linear_route_unpack.mojo `xl_route`: a hit is one
    record read) and, for the regressors, the metrics R^2 entry of the
    process's tier on the route's device in the route's fifth slot when that
    image is loaded; then the 20 parameter words (core/sgd_member.mojo
    docstring). 'balanced' and None are word 18; an exact dict (lane
    mf-adm3) is word 18 = 2 and the class-weight tail from word 20
    (core/member_class_weight.mojo), a pending table the caller fills over
    its labels; any other value is not admitted (False: its own fit refuses
    it in its words).
  * `sgd_member_pack` (PACK): the fitted attributes the kind's
    `xl_fit_*_pack` writes (bindings/x_linear_sgd_pack.mojo), in its order.
  * `sgd_member_state` (UNPACK): a member state from a fitted estimator.

SERIAL BY NATURE: one member's admission or publication; a caller runs its
members in parallel.
"""
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_input_unpack import admit_x
from bindings.classical_arrays_door import classical_array, classical_address
from bindings.member_owner_unpack import MemberOwner, member_labels_take
from bindings.py_balanced import py_get, py_item, py_contains, py_setitem
from bindings.member_class_weight_unpack import member_cw_kind, member_cw_invalid_message, member_cw_kind_accepted
from core.member_class_weight import MEMBER_CW_DICT, member_cw_pending_words, member_cw_invalid_words
from core.member_families import MEMBER_FAMILY_SGD
from bindings.x_linear_common_unpack import (
    _truth, _float, _int, _name, _b, _eq, _gt, _lt, _le, _is_int, _isinstance, _repr, _sorted_repr,
)
from bindings.x_linear_keys_unpack import x_linear_keys
from bindings.x_linear_sgd_unpack import _xl_sgd_refuse, _xl_seed, _xl_sgd_auto_sum
from bindings.x_linear_route_unpack import xl_route, xl_ops_require
from core.entry_table import EntryRef, entry_ref_word
from core.hostptr import f32_ptr
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_COEF, MEMBER_BLOCK_INTERCEPT, MEMBER_BLOCK_SCALARS,
    MEMBER_IMPORT_DONE, MEMBER_F64,
)
from core.sgd_member import (
    SGD_MEMBER_REGRESSOR, SGD_MEMBER_PA_REGRESSOR, SGD_MEMBER_CLASSIFIER, SGD_MEMBER_PERCEPTRON,
    SGD_MEMBER_PA_CLASSIFIER, SGD_MEMBER_ONE_CLASS,
)
from core.metrics_entry_table import MetricsEntryKey, metrics_r2_entry, MR_OP_COUNT
from providers.python.device_route_boundary_door import device_route_value


def _sgd_words(model: PythonObject, kind: Int, loss_code: Int, penalty: PythonObject, lr: PythonObject,
    alpha: PythonObject, l1_ratio: PythonObject, eta0: PythonObject, power_t: PythonObject, epsilon: PythonObject,
    fit_intercept: PythonObject, max_iter: PythonObject, tol: PythonObject, n_iter_no_change: PythonObject,
    shuffle: PythonObject, random_state: PythonObject, batch_size: PythonObject, batch_sum: Bool, auto_cap: Int,
    balanced: Bool, probability: Bool, context: PythonObject) raises -> List[Float64]:
    """`_xl_sgd_build`'s refusals (penalty, learning rate, eta0, alpha, the
    seed, batch_size) in its order and words, then the 20 words."""
    var nk = x_linear_keys()
    var name = _name(model)
    var penalties = py_item(context, nk[].k_xl_sgd_penalty)
    var rates = py_item(context, nk[].k_xl_sgd_lr)
    if not py_contains(penalties, penalty):
        raise Error("mojolearn " + name + ": penalty must be 'l2', 'l1', 'elasticnet' or None")
    if not py_contains(rates, lr):
        raise Error("mojolearn " + name + ": unknown learning_rate " + _repr(lr))
    if py_contains(Python.tuple("constant", "invscaling", "adaptive"), lr) and not _gt(eta0, PythonObject(0)):
        raise Error("mojolearn " + name + ": eta0 must be > 0")
    if _eq(lr, PythonObject("optimal")) and not _gt(alpha, PythonObject(0)):
        raise Error("mojolearn " + name + ": alpha must be > 0 for learning_rate='optimal'")
    var seed = _xl_seed(random_state)
    var auto = _isinstance(batch_size, _b().str) and _eq(batch_size, PythonObject("auto"))
    var batch = -1
    if auto:
        batch = -auto_cap
    elif _truth(batch_size) and (not _is_int(batch_size) or _lt(batch_size, PythonObject(1))):
        raise Error("mojolearn " + name + ": batch_size must be 'auto', 0 or a positive int")
    elif batch_size is not None:
        batch = _int(batch_size)
    return [Float64(kind), Float64(loss_code), Float64(_int(penalties[penalty])), Float64(_int(rates[lr])),
        1.0 if _truth(fit_intercept) else 0.0, Float64(_int(max_iter)), Float64(_int(n_iter_no_change)),
        1.0 if _truth(shuffle) else 0.0, Float64(seed[0]), Float64(seed[1]), Float64(batch), 1.0 if batch_sum else 0.0,
        _float(alpha), _float(l1_ratio), _float(eta0), _float(power_t), _float(epsilon),
        -3.0e38 if tol is None else _float(tol), 1.0 if balanced else 0.0, 1.0 if probability else 0.0]


def sgd_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator. False for a class_weight that is neither
    None, 'balanced' nor an exact dict."""
    var nk = x_linear_keys()
    var balanced = False
    var keyed = False
    var invalid = False
    if kind >= SGD_MEMBER_CLASSIFIER and kind <= SGD_MEMBER_PA_CLASSIFIER:
        # A dict is keyed by labels: its table is read over the caller's
        # labels at the caller's UNPACK (lane mf-adm3, `member_cw_fill`).
        var class_weight = py_item(fields, nk[].k_class_weight)
        if class_weight is not None:
            if member_cw_kind_accepted(MEMBER_FAMILY_SGD, model, class_weight, b) == MEMBER_CW_DICT:
                keyed = True
            elif not (_isinstance(class_weight, _b().str) and _eq(class_weight, PythonObject("balanced"))):
                # A value the fit door refuses (lane mf-left4): the INVALID
                # tail, whose words the FIT caller's UNPACK raises; a value
                # the door accepts that the member does not read: not admitted.
                if len(member_cw_invalid_message(MEMBER_FAMILY_SGD, model, class_weight, b)) == 0:
                    return False
                invalid = True
            else:
                balanced = True
    var words: List[Float64]
    if kind == SGD_MEMBER_REGRESSOR or kind == SGD_MEMBER_PA_REGRESSOR:
        _xl_sgd_refuse(model, py_item(fields, nk[].k_early_stopping), py_item(fields, nk[].k_average),
            py_item(fields, nk[].k_warm_start))
        var losses = py_item(context, nk[].k_xl_sgd_reg_loss)
        var loss = py_item(fields, nk[].k_loss)
        if kind == SGD_MEMBER_PA_REGRESSOR:
            if not py_contains(Python.tuple("epsilon_insensitive", "squared_epsilon_insensitive"), loss):
                raise Error("mojolearn PassiveAggressiveRegressor: loss must be 'epsilon_insensitive' "
                            + "or 'squared_epsilon_insensitive'")
            if not _gt(py_item(fields, nk[].k_C), PythonObject(0)):
                raise Error("mojolearn PassiveAggressiveRegressor: C must be > 0")
            var lr = PythonObject("pa1") if _eq(loss, PythonObject("epsilon_insensitive")) else PythonObject("pa2")
            words = _sgd_words(model, kind, _int(py_item(losses, nk[].k_epsilon_insensitive)), PythonObject(None), lr,
                PythonObject(1.0), PythonObject(0.0), py_item(fields, nk[].k_C), PythonObject(0.5),
                py_item(fields, nk[].k_epsilon), py_item(fields, nk[].k_fit_intercept), py_item(fields, nk[].k_max_iter),
                py_item(fields, nk[].k_tol), py_item(fields, nk[].k_n_iter_no_change), py_item(fields, nk[].k_shuffle),
                py_item(fields, nk[].k_random_state), py_item(fields, nk[].k_batch_size), False, 256, False, False,
                context)
        else:
            if not py_contains(losses, loss):
                raise Error("mojolearn SGDRegressor: loss must be one of " + _sorted_repr(losses))
            words = _sgd_words(model, kind, _int(losses[loss]), py_item(fields, nk[].k_penalty),
                py_item(fields, nk[].k_learning_rate), py_item(fields, nk[].k_alpha), py_item(fields, nk[].k_l1_ratio),
                py_item(fields, nk[].k_eta0), py_item(fields, nk[].k_power_t), py_item(fields, nk[].k_epsilon),
                py_item(fields, nk[].k_fit_intercept), py_item(fields, nk[].k_max_iter), py_item(fields, nk[].k_tol),
                py_item(fields, nk[].k_n_iter_no_change), py_item(fields, nk[].k_shuffle),
                py_item(fields, nk[].k_random_state), py_item(fields, nk[].k_batch_size), False, 4096, False, False,
                context)
    elif kind == SGD_MEMBER_CLASSIFIER or kind == SGD_MEMBER_PERCEPTRON or kind == SGD_MEMBER_PA_CLASSIFIER:
        # `xl_fit_sgd_classifier_unpack`'s refusals, then its three requests.
        var losses = py_item(context, nk[].k_xl_sgd_clf_loss)
        var loss = py_item(fields, nk[].k_loss) if kind != SGD_MEMBER_PERCEPTRON else PythonObject(None)
        if kind == SGD_MEMBER_PERCEPTRON:
            _xl_sgd_refuse(model, py_item(fields, nk[].k_early_stopping), PythonObject(False),
                py_item(fields, nk[].k_warm_start))
        else:
            _xl_sgd_refuse(model, py_item(fields, nk[].k_early_stopping), py_item(fields, nk[].k_average),
                py_item(fields, nk[].k_warm_start))
        if kind == SGD_MEMBER_CLASSIFIER and not py_contains(losses, loss):
            raise Error("mojolearn SGDClassifier: loss must be one of " + _sorted_repr(losses))
        if kind == SGD_MEMBER_PA_CLASSIFIER:
            if not py_contains(Python.tuple("hinge", "squared_hinge"), loss):
                raise Error("mojolearn PassiveAggressiveClassifier: loss must be 'hinge' or 'squared_hinge'")
            if not _gt(py_item(fields, nk[].k_C), PythonObject(0)):
                raise Error("mojolearn PassiveAggressiveClassifier: C must be > 0")
        if kind == SGD_MEMBER_CLASSIFIER:
            var batch_sum = _xl_sgd_auto_sum(py_item(fields, nk[].k_batch_size), loss,
                py_item(fields, nk[].k_learning_rate), context)
            words = _sgd_words(model, kind, _int(losses[loss]), py_item(fields, nk[].k_penalty),
                py_item(fields, nk[].k_learning_rate), py_item(fields, nk[].k_alpha), py_item(fields, nk[].k_l1_ratio),
                py_item(fields, nk[].k_eta0), py_item(fields, nk[].k_power_t), py_item(fields, nk[].k_epsilon),
                py_item(fields, nk[].k_fit_intercept), py_item(fields, nk[].k_max_iter), py_item(fields, nk[].k_tol),
                py_item(fields, nk[].k_n_iter_no_change), py_item(fields, nk[].k_shuffle),
                py_item(fields, nk[].k_random_state), py_item(fields, nk[].k_batch_size), batch_sum, 4096, balanced,
                _eq(loss, PythonObject("log_loss")), context)
        elif kind == SGD_MEMBER_PERCEPTRON:
            words = _sgd_words(model, kind, _int(py_item(losses, nk[].k_perceptron)), py_item(fields, nk[].k_penalty),
                PythonObject("constant"), py_item(fields, nk[].k_alpha), py_item(fields, nk[].k_l1_ratio),
                py_item(fields, nk[].k_eta0), PythonObject(0.5), PythonObject(0.1), py_item(fields, nk[].k_fit_intercept),
                py_item(fields, nk[].k_max_iter), py_item(fields, nk[].k_tol), py_item(fields, nk[].k_n_iter_no_change),
                py_item(fields, nk[].k_shuffle), py_item(fields, nk[].k_random_state), py_item(fields, nk[].k_batch_size),
                True, 256, balanced, False, context)
        else:
            var lr = PythonObject("pa1") if _eq(loss, PythonObject("hinge")) else PythonObject("pa2")
            words = _sgd_words(model, kind, _int(py_item(losses, nk[].k_hinge)), PythonObject(None), lr,
                PythonObject(1.0), PythonObject(0.0), py_item(fields, nk[].k_C), PythonObject(0.5), PythonObject(0.1),
                py_item(fields, nk[].k_fit_intercept), py_item(fields, nk[].k_max_iter), py_item(fields, nk[].k_tol),
                py_item(fields, nk[].k_n_iter_no_change), py_item(fields, nk[].k_shuffle),
                py_item(fields, nk[].k_random_state), py_item(fields, nk[].k_batch_size), False, 256, balanced, False,
                context)
    elif kind == SGD_MEMBER_ONE_CLASS:
        # `xl_fit_one_class_unpack`'s refusals, then its request.
        _xl_sgd_refuse(model, PythonObject(False), py_item(fields, nk[].k_average), py_item(fields, nk[].k_warm_start))
        var nu = py_item(fields, nk[].k_nu)
        if not (_lt(PythonObject(0), nu) and _le(nu, PythonObject(1))):
            raise Error("mojolearn SGDOneClassSVM: nu must be in (0, 1]")
        if py_contains(Python.tuple("pa1", "pa2"), py_item(fields, nk[].k_learning_rate)):
            raise Error("mojolearn SGDOneClassSVM: learning_rate must be constant, optimal, invscaling or adaptive")
        words = _sgd_words(model, kind, _int(py_item(py_item(context, nk[].k_xl_sgd_clf_loss), nk[].k_hinge)),
            PythonObject("l2"), py_item(fields, nk[].k_learning_rate), nu, PythonObject(0.0),
            py_item(fields, nk[].k_eta0), py_item(fields, nk[].k_power_t), PythonObject(0.1),
            py_item(fields, nk[].k_fit_intercept), py_item(fields, nk[].k_max_iter), py_item(fields, nk[].k_tol),
            PythonObject(5), py_item(fields, nk[].k_shuffle), py_item(fields, nk[].k_random_state),
            py_item(fields, nk[].k_batch_size), False, 4096, False, False, context)
    else:
        raise Error("RuntimeError: mojolearn: unknown Sgd member kind " + String(kind))
    var ops = xl_route(model, fields, context, False)
    xl_ops_require(ops)
    var route = device_route_value(ops.owner)
    if (kind == SGD_MEMBER_REGRESSOR or kind == SGD_MEMBER_PA_REGRESSOR) and not route.has_fifth:
        # The metrics image's R^2 entry the estimator's own score reaches
        # (its process tier, the route's device), when that image is loaded.
        var metric_mode = 1 if String(py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)) == "identical" else 0
        try:
            var entry = metrics_r2_entry(MetricsEntryKey(metric_mode, route.vendor))
            route.fifth = EntryRef(entry, entry_ref_word(metric_mode, route.vendor, MR_OP_COUNT))
            route.has_fifth = True
        except problem:
            # Not loaded: score is refused in words; the caller runs its own metric.
            _ = String(problem)
    if keyed:
        words[18] = 2.0
        member_cw_pending_words(words)
    elif invalid:
        words[18] = 2.0
        member_cw_invalid_words(words)
    owner.plan.route = route
    owner.plan.params = words^
    owner.pins = Python.tuple(ops.owner, ops.scope)
    return True


def sgd_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model` in
    the own fit's order (bindings/x_linear_sgd_pack.mojo): a classifier's
    `classes_` first (the caller's labels), then `n_iter_`, `t_`,
    `n_features_in_`, `coef_` ((T, d) for a classifier, else (d,)) and
    `intercept_` (T,) (SGDOneClassSVM: `offset_` (1,))."""
    var nk = x_linear_keys()
    var kind = Int(plan.params[0])
    var classifier = kind >= SGD_MEMBER_CLASSIFIER and kind <= SGD_MEMBER_PA_CLASSIFIER
    var call = MemberCall(plan.route, state)
    if classifier:
        var labels = member_labels_take(fields, b)
        if labels is not None:
            py_setitem(fields, nk[].k_classes_, labels)
    call.block = MEMBER_BLOCK_SCALARS
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var n_iter = Int(scalars[0])
    var t = scalars[1]
    call.block = MEMBER_BLOCK_COEF
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var outputs = call.dst.rows
    var d = call.dst.cols
    var weights = classical_array(outputs, d, context) if classifier else classical_array(d, 0, context)
    unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(weights))), src=f32_ptr(call.dst.data), count=outputs * d)
    call.block = MEMBER_BLOCK_INTERCEPT
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    var offsets = classical_array(outputs, 0, context)
    unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(offsets))), src=f32_ptr(call.dst.data), count=outputs)
    py_setitem(fields, nk[].k_n_iter_, PythonObject(n_iter))
    py_setitem(fields, nk[].k_t_, PythonObject(t))
    py_setitem(fields, nk[].k_n_features_in_, PythonObject(d))
    py_setitem(fields, nk[].k_coef_, weights)
    if kind == SGD_MEMBER_ONE_CLASS:
        py_setitem(fields, nk[].k_offset_, offsets)
    else:
        py_setitem(fields, nk[].k_intercept_, offsets)


def _count_word(fields: PythonObject, key: PythonObject) raises -> Float64:
    var value = py_get(fields, key)
    if value is None:
        return 0.0
    return Float64(py=value)


def sgd_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted attributes of `model`
    (coef_, intercept_ or offset_, n_iter_, t_; a classifier's class count
    from classes_). The caller frees it (`member_free`)."""
    var nk = x_linear_keys()
    var kind = Int(plan.params[0])
    var classifier = kind >= SGD_MEMBER_CLASSIFIER and kind <= SGD_MEMBER_PA_CLASSIFIER
    var coefficient_source = py_get(fields, nk[].k_coef_)
    if coefficient_source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var classes = 0
    if classifier:
        var classes_source = py_get(fields, nk[].k_classes_)
        if classes_source is None:
            raise Error("NotFittedError: this member is not fitted yet")
        classes = len(classes_source)
    var coef = admit_x(coefficient_source, 2 if classifier else 1, "coef_", context)
    var outputs = Int(py=coef.shape[0]) if classifier else 1
    var d = Int(py=coef.shape[1]) if classifier else Int(py=coef.size)
    var intercept_source: PythonObject
    var intercept_name: String
    if kind == SGD_MEMBER_ONE_CLASS:
        intercept_source = py_item(fields, nk[].k_offset_)
        intercept_name = String("offset_")
    else:
        intercept_source = py_item(fields, nk[].k_intercept_)
        intercept_name = String("intercept_")
    var intercept = admit_x(intercept_source, 1, intercept_name, context)
    var scalars: List[Float64] = [_count_word(fields, nk[].k_n_iter_), _count_word(fields, nk[].k_t_)]
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_COEF
        call.x = MemberMatrix(Int(py=classical_address(coef)), outputs, d)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 2, MEMBER_F64)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_INTERCEPT
        call.x = MemberMatrix(Int(py=classical_address(intercept)), 1, Int(py=intercept.size))
        call.classes = classes
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = coef
    _ = intercept
    _ = len(scalars)
    return call.state

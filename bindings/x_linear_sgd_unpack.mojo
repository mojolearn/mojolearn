# SPDX-License-Identifier: Apache-2.0
# UNPACK of the online fits (SGDRegressor, PassiveAggressiveRegressor,
# SGDClassifier, Perceptron, PassiveAggressiveClassifier, SGDOneClassSVM)
# and of RidgeClassifier.fit (python/mojolearn/_expansion_linear.py): the
# refusals in the order and words the Python made them, X through the one
# admission, the targets or the encoded class codes, the checked weights,
# and the typed fit request (x_linear/door_run.mojo `XlFitRequest`), one
# function per public operation. RUN (`xl_fit_run`) holds no Python object;
# PACK is bindings/x_linear_sgd_pack.mojo.
#
# The class weights (`class_weight`) are computed by the binding
# (XL_OP_CLASS_PREP). UNPACK builds that step's typed request
# (`XlClassPrepRequest`: the class-weight words read from a dict, the codes,
# the checked sample weights, where the row weights land) and RUN makes the
# call before the fit (x_linear/door_run.mojo `xl_class_fit_run`): SGD takes
# the k class weights as its last float parameters, RidgeClassifier reads
# the row weights behind its class codes. No native call is made here.
from bindings.x_linear_common_unpack import XlFit, xl_fit_build, xl_fit_build_x, xl_class_weights_unpack, _xl_scope
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_linear.door_run import (
    ALGO_SGD, ALGO_RIDGE, xl_seed_halves, XlClassPrepRequest, xl_class_prep_none, xl_class_fit_run, xl_run_join,
)
from bindings.device_rows_door import door_rows_refuse
from bindings.x_linear_route_unpack import XLinearOps, xl_ops_require
from bindings.x_linear_common_unpack import (
    _b, _op, _truth, _float, _int, _repr, _gt, _ge, _lt, _le, _eq, _isinstance, _is_int, _name, _attr,
    _sorted_repr, _xl_matrix, _xl_vector, _xl_fold_matrix, _xl_with_weights,
)
from bindings.py_balanced import py_call, py_contains
from bindings.admitted_input_unpack import (
    admit_weight_words, admitted_words, admitted_vector, admitted_storage_native,
)
from bindings.admitted_input_run import AdmittedXView
from bindings.x_linear_keys_unpack import (
    x_linear_keys, X_LINEAR_OPN_FIT_PA_CLASSIFIER, X_LINEAR_OPN_FIT_PERCEPTRON, X_LINEAR_OPN_FIT_SGD_CLASSIFIER,
)
from bindings.py_balanced import py_item
from bindings.classical_labels_door import ClassicalLabels, labels_address, labels_cast, classical_encode_labels


struct XlClassFit(Movable):
    """One admitted classifier fit: the fit call over the class codes, the
    sorted class labels PACK publishes as `classes_`, and the class-weight
    step RUN makes before the fit (`prep`; not active without
    `class_weight`) with the objects whose words it addresses: the int32
    class codes, the k class weights and the checked sample weights."""
    var fit: XlFit
    var classes: PythonObject
    var prep: XlClassPrepRequest
    var codes: PythonObject
    var class_weight: PythonObject
    var sample_weight: PythonObject

    def __init__(out self, var fit: XlFit, var classes: PythonObject, var prep: XlClassPrepRequest,
                 var codes: PythonObject, var class_weight: PythonObject, var sample_weight: PythonObject):
        self.fit = fit^
        self.classes = classes^
        self.prep = prep^
        self.codes = codes^
        self.class_weight = class_weight^
        self.sample_weight = sample_weight^


def xl_class_fit_released(call: XlClassFit, model: PythonObject) raises:
    """RUN of one classifier fit with the GIL released (the class-weight
    step, then the fit); then the refusal of a device fold on its status
    words (a host fit has none)."""
    var words: Tuple[Int, Int] = (0, 0)
    var py = Python()
    with GILReleased(py):
        words = xl_class_fit_run(call.fit.request, call.prep)
    if call.fit.request.fold:
        door_rows_refuse(words[0], words[1], _name(model), call.fit.request.base_rows)


def _xl_sgd_refuse(model: PythonObject, early_stopping: PythonObject, average: PythonObject,
                   warm_start: PythonObject) raises:
    """`_sgd_refuse`."""
    var name = _name(model)
    if _truth(early_stopping):
        raise Error("mojolearn " + name + ": early_stopping is not implemented (x_linear/NOT_IMPLEMENTED.tsv)")
    if _truth(average):
        raise Error("mojolearn " + name + ": average is not implemented (x_linear/NOT_IMPLEMENTED.tsv)")
    if _truth(warm_start):
        raise Error("mojolearn " + name + ": warm_start is not implemented (x_linear/NOT_IMPLEMENTED.tsv)")


def _xl_seed(random_state: PythonObject) raises -> Tuple[Int, Int]:
    """`_seed`: the seed's two Int32 halves (None -> 0)."""
    if random_state is None:
        return (0, 0)
    if not _is_int(random_state) or _lt(random_state, PythonObject(0)):
        raise Error("mojolearn: random_state must be None or a non-negative int")
    var op = _op()
    var mask = py_call(op.sub, py_call(op.lshift, PythonObject(1), PythonObject(64)), PythonObject(1))
    var s = py_call(op.and_, random_state, mask)
    var lo = Int(py=py_call(op.and_, s, PythonObject(4294967295)))
    var hi = Int(py=py_call(op.rshift, s, PythonObject(32)))
    return xl_seed_halves(lo, hi)


def _xl_sgd_auto_sum(batch_size: PythonObject, loss: PythonObject, learning_rate: PythonObject,
                     context: PythonObject) raises -> Bool:
    """`_sgd_auto_sum`."""
    var nk = x_linear_keys()
    return (_isinstance(batch_size, _b().str) and _eq(batch_size, PythonObject("auto"))
            and py_contains(py_item(context, nk[].k_xl_sgd_sum_losses), loss)
            and not py_contains(Python.tuple("pa1", "pa2"), learning_rate))


def _xl_sgd_build(model: PythonObject, entries: XLinearOps, x: AdmittedXView, x_keep: PythonObject,
                  y_keep: PythonObject, y_addr: Int,
                  n_classes: Int, loss_code: Int, penalty: PythonObject, lr: PythonObject, alpha: PythonObject,
                  l1_ratio: PythonObject, eta0: PythonObject, power_t: PythonObject, epsilon: PythonObject,
                  fit_intercept: PythonObject, max_iter: PythonObject, tol: PythonObject,
                  n_iter_no_change: PythonObject, shuffle: PythonObject, random_state: PythonObject,
                  sample_weight: PythonObject, class_weight: PythonObject, classes: PythonObject, codes: PythonObject,
                  batch_size: PythonObject, var batch_sum: Bool, auto_cap: Int, mut prep: XlClassPrepRequest,
                  mut class_keep: PythonObject, context: PythonObject) raises -> XlFit:
    """`_sgd_fit`, UNPACK: the refusals, the seed, the integer and float
    hyperparameters, the targets (then the checked weights) and the typed
    fit request. The n float32 targets or class codes are at `y_addr`
    (`y_keep` owns them); 0 is no targets (SGDOneClassSVM: zeros). With
    `class_weight`, `prep` becomes the class-weight step's request
    (XL_OP_CLASS_PREP over the int32 `codes`; `class_keep` owns the k
    class-weight words): RUN makes that call and appends the k weights to
    the request's float parameters (`xl_class_fit_run`)."""
    var nk = x_linear_keys()
    var name = _name(model)
    var n = x.rows
    var d = x.cols
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
    var problems = n_classes if n_classes > 2 else 1
    var ip: List[Int] = [n_classes, loss_code, _int(penalties[penalty]), _int(rates[lr]),
                         1 if _truth(fit_intercept) else 0, _int(max_iter), _int(n_iter_no_change),
                         1 if _truth(shuffle) else 0, seed[0], seed[1]]
    var fp: List[Float64] = [_float(alpha), _float(l1_ratio), _float(eta0), _float(power_t), _float(epsilon),
                             -3.0e38 if tol is None else _float(tol)]
    var targets = y_keep
    var t_addr = y_addr
    if y_addr == 0:
        targets = admitted_vector(n, "<f4", context)
        t_addr = labels_address(targets)
    var has_sw = 0
    var t = admitted_words(targets, t_addr, n, 1)
    var yy = _xl_with_weights(t, sample_weight, n, has_sw, context)
    var has_cw = 0
    if class_weight is not None and n_classes >= 2:
        # k scalars: the fit's last float parameters, appended by RUN after
        # the binding computed them
        var weights = xl_class_weights_unpack(class_weight, classes, context)
        class_keep = weights[0]
        prep = XlClassPrepRequest(True, labels_address(codes), 0, labels_address(class_keep), 0, n, n_classes,
                                  weights[1], False)
        has_cw = 1
    var auto = _isinstance(batch_size, _b().str) and _eq(batch_size, PythonObject("auto"))
    var batch = -1
    if auto:
        batch = -auto_cap
    elif _truth(batch_size) and (not _is_int(batch_size) or _lt(batch_size, PythonObject(1))):
        raise Error("mojolearn " + name + ": batch_size must be 'auto', 0 or a positive int")
    elif batch_size is not None:
        batch = _int(batch_size)
    # TOMBSTONE: MOJOLEARN_SGD_BATCH_SUM (env override, the B arm of MOJOLEARN_SGD_AUTO_STEPS_256_OFF, whose old path was
    # deleted 2026-11-15 by lane/expdel2-21-linear) deleted 2026-11-18 by lane/fx-env-dead; code recoverable at 0c65d61fed.
    # Restore: git apply experiments/removed/MOJOLEARN_SGD_BATCH_SUM.patch
    ip.append(has_sw)
    ip.append(has_cw)
    ip.append(batch)
    ip.append(1 if batch_sum else 0)
    return xl_fit_build(entries, ALGO_SGD, x.address, n * d, n, d, x_keep, yy.address, yy.rows, yy.array, ip^, fp^,
                        problems * d + problems + 2, problems * (n + d + 1), problems * n, context)


def xl_fit_sgd_regressor_unpack(model: PythonObject, fields: PythonObject, args: PythonObject, passive: Bool,
                                entries: XLinearOps, context: PythonObject) raises -> XlFit:
    """SGDRegressor.fit(X, y, sample_weight) and (`passive`)
    PassiveAggressiveRegressor.fit(X, y)."""
    var nk = x_linear_keys()
    _xl_sgd_refuse(model, py_item(fields, nk[].k_early_stopping), py_item(fields, nk[].k_average), py_item(fields, nk[].k_warm_start))
    var losses = py_item(context, nk[].k_xl_sgd_reg_loss)
    var loss = py_item(fields, nk[].k_loss)
    if passive:
        if not py_contains(Python.tuple("epsilon_insensitive", "squared_epsilon_insensitive"), loss):
            raise Error("mojolearn PassiveAggressiveRegressor: loss must be 'epsilon_insensitive' "
                        + "or 'squared_epsilon_insensitive'")
        if not _gt(py_item(fields, nk[].k_C), PythonObject(0)):
            raise Error("mojolearn PassiveAggressiveRegressor: C must be > 0")
    elif not py_contains(losses, loss):
        raise Error("mojolearn SGDRegressor: loss must be one of " + _sorted_repr(losses))
    xl_ops_require(entries)
    var x = _xl_matrix(args[0], _name(model), context)
    var y = _xl_vector(args[1], x.rows, context)
    # a regressor has no class-weight step
    var prep = xl_class_prep_none()
    var class_keep = PythonObject(None)
    if passive:
        var lr = PythonObject("pa1") if _eq(loss, PythonObject("epsilon_insensitive")) else PythonObject("pa2")
        return _xl_sgd_build(model, entries, x.view(), x.array, y.array, y.address, 0, _int(py_item(losses, nk[].k_epsilon_insensitive)),
                             PythonObject(None), lr, PythonObject(1.0), PythonObject(0.0), py_item(fields, nk[].k_C), PythonObject(0.5),
                             py_item(fields, nk[].k_epsilon), py_item(fields, nk[].k_fit_intercept), py_item(fields, nk[].k_max_iter), py_item(fields, nk[].k_tol),
                             py_item(fields, nk[].k_n_iter_no_change), py_item(fields, nk[].k_shuffle), py_item(fields, nk[].k_random_state), PythonObject(None),
                             PythonObject(None), PythonObject(None), PythonObject(None), py_item(fields, nk[].k_batch_size), False, 256,
                             prep, class_keep, context)
    return _xl_sgd_build(model, entries, x.view(), x.array, y.array, y.address, 0, _int(losses[loss]), py_item(fields, nk[].k_penalty),
                         py_item(fields, nk[].k_learning_rate), py_item(fields, nk[].k_alpha), py_item(fields, nk[].k_l1_ratio), py_item(fields, nk[].k_eta0), py_item(fields, nk[].k_power_t),
                         py_item(fields, nk[].k_epsilon), py_item(fields, nk[].k_fit_intercept), py_item(fields, nk[].k_max_iter), py_item(fields, nk[].k_tol),
                         py_item(fields, nk[].k_n_iter_no_change), py_item(fields, nk[].k_shuffle), py_item(fields, nk[].k_random_state), args[2],
                         PythonObject(None), PythonObject(None), PythonObject(None), py_item(fields, nk[].k_batch_size), False, 4096,
                         prep, class_keep, context)


def xl_fit_one_class_unpack(model: PythonObject, fields: PythonObject, args: PythonObject, entries: XLinearOps,
                            context: PythonObject) raises -> XlFit:
    """SGDOneClassSVM.fit(X, y=None, sample_weight=None)."""
    var nk = x_linear_keys()
    _xl_sgd_refuse(model, PythonObject(False), py_item(fields, nk[].k_average), py_item(fields, nk[].k_warm_start))
    var nu = py_item(fields, nk[].k_nu)
    if not (_lt(PythonObject(0), nu) and _le(nu, PythonObject(1))):
        raise Error("mojolearn SGDOneClassSVM: nu must be in (0, 1]")
    if py_contains(Python.tuple("pa1", "pa2"), py_item(fields, nk[].k_learning_rate)):
        raise Error("mojolearn SGDOneClassSVM: learning_rate must be constant, optimal, invscaling or adaptive")
    xl_ops_require(entries)
    var x = _xl_matrix(args[0], _name(model), context)
    # one class: no class-weight step
    var prep = xl_class_prep_none()
    var class_keep = PythonObject(None)
    return _xl_sgd_build(model, entries, x.view(), x.array, PythonObject(None), 0, 1, _int(py_item(py_item(context, nk[].k_xl_sgd_clf_loss), nk[].k_hinge)),
                             PythonObject("l2"), py_item(fields, nk[].k_learning_rate), nu, PythonObject(0.0), py_item(fields, nk[].k_eta0),
                             py_item(fields, nk[].k_power_t), PythonObject(0.1), py_item(fields, nk[].k_fit_intercept), py_item(fields, nk[].k_max_iter), py_item(fields, nk[].k_tol),
                             PythonObject(5), py_item(fields, nk[].k_shuffle), py_item(fields, nk[].k_random_state), args[2], PythonObject(None),
                             PythonObject(None), PythonObject(None), py_item(fields, nk[].k_batch_size), False, 4096, prep,
                             class_keep, context)


def _xl_class_codes(model: PythonObject, source: PythonObject, n: Int, context: PythonObject) raises -> ClassicalLabels:
    """`_class_codes(est, y, n)`: the labels encoded once at entry."""
    var labels = classical_encode_labels(source, context)
    if labels.n != n:
        raise Error("mojolearn " + _name(model) + ": X and y lengths differ")
    if len(labels.classes) < 2:
        raise Error("mojolearn " + _name(model) + ": y has one class")
    return labels^


def xl_fit_sgd_classifier_unpack(model: PythonObject, fields: PythonObject, args: PythonObject, operation: Int,
                                 entries: XLinearOps, context: PythonObject) raises -> XlClassFit:
    """SGDClassifier.fit(X, y, sample_weight) ("sgd"), Perceptron.fit(X, y,
    sample_weight) ("perceptron") and PassiveAggressiveClassifier.fit(X, y)
    ("pa"): one SGD fit over the class codes."""
    var nk = x_linear_keys()
    var losses = py_item(context, nk[].k_xl_sgd_clf_loss)
    if operation == X_LINEAR_OPN_FIT_PERCEPTRON:
        _xl_sgd_refuse(model, py_item(fields, nk[].k_early_stopping), PythonObject(False), py_item(fields, nk[].k_warm_start))
    else:
        _xl_sgd_refuse(model, py_item(fields, nk[].k_early_stopping), py_item(fields, nk[].k_average), py_item(fields, nk[].k_warm_start))
    if operation == X_LINEAR_OPN_FIT_SGD_CLASSIFIER and not py_contains(losses, py_item(fields, nk[].k_loss)):
        raise Error("mojolearn SGDClassifier: loss must be one of " + _sorted_repr(losses))
    if operation == X_LINEAR_OPN_FIT_PA_CLASSIFIER:
        if not py_contains(Python.tuple("hinge", "squared_hinge"), py_item(fields, nk[].k_loss)):
            raise Error("mojolearn PassiveAggressiveClassifier: loss must be 'hinge' or 'squared_hinge'")
        if not _gt(py_item(fields, nk[].k_C), PythonObject(0)):
            raise Error("mojolearn PassiveAggressiveClassifier: C must be > 0")
    xl_ops_require(entries)
    var x = _xl_matrix(args[0], _name(model), context)
    var labels = _xl_class_codes(model, args[1], x.rows, _xl_scope(entries, context))
    var codes = admitted_vector(x.rows, "<f4", context)
    labels_cast(labels_address(labels.codes), "<i4", labels_address(codes), "<f4", x.rows, context)
    var k = len(labels.classes)
    var prep = xl_class_prep_none()
    var class_keep = PythonObject(None)
    if operation == X_LINEAR_OPN_FIT_SGD_CLASSIFIER:
        var batch_sum = _xl_sgd_auto_sum(py_item(fields, nk[].k_batch_size), py_item(fields, nk[].k_loss), py_item(fields, nk[].k_learning_rate), context)
        var sgd = _xl_sgd_build(model, entries, x.view(), x.array, codes, labels_address(codes), k, _int(losses[py_item(fields, nk[].k_loss)]),
                             py_item(fields, nk[].k_penalty), py_item(fields, nk[].k_learning_rate), py_item(fields, nk[].k_alpha), py_item(fields, nk[].k_l1_ratio), py_item(fields, nk[].k_eta0),
                             py_item(fields, nk[].k_power_t), py_item(fields, nk[].k_epsilon), py_item(fields, nk[].k_fit_intercept), py_item(fields, nk[].k_max_iter), py_item(fields, nk[].k_tol),
                             py_item(fields, nk[].k_n_iter_no_change), py_item(fields, nk[].k_shuffle), py_item(fields, nk[].k_random_state), args[2],
                             py_item(fields, nk[].k_class_weight), labels.classes, labels.codes, py_item(fields, nk[].k_batch_size), batch_sum, 4096,
                             prep, class_keep, context)
        return XlClassFit(sgd^, labels.classes, prep^, labels.codes, class_keep, PythonObject(None))
    elif operation == X_LINEAR_OPN_FIT_PERCEPTRON:
        var perceptron = _xl_sgd_build(model, entries, x.view(), x.array, codes, labels_address(codes), k, _int(py_item(losses, nk[].k_perceptron)),
                             py_item(fields, nk[].k_penalty), PythonObject("constant"), py_item(fields, nk[].k_alpha), py_item(fields, nk[].k_l1_ratio), py_item(fields, nk[].k_eta0),
                             PythonObject(0.5), PythonObject(0.1), py_item(fields, nk[].k_fit_intercept), py_item(fields, nk[].k_max_iter), py_item(fields, nk[].k_tol),
                             py_item(fields, nk[].k_n_iter_no_change), py_item(fields, nk[].k_shuffle), py_item(fields, nk[].k_random_state), args[2],
                             py_item(fields, nk[].k_class_weight), labels.classes, labels.codes, py_item(fields, nk[].k_batch_size), True, 256, prep,
                             class_keep, context)
        return XlClassFit(perceptron^, labels.classes, prep^, labels.codes, class_keep, PythonObject(None))
    var lr = PythonObject("pa1") if _eq(py_item(fields, nk[].k_loss), PythonObject("hinge")) else PythonObject("pa2")
    # scikit-learn's PassiveAggressive*.fit takes no sample_weight
    var passive = _xl_sgd_build(model, entries, x.view(), x.array, codes, labels_address(codes), k, _int(py_item(losses, nk[].k_hinge)),
                             PythonObject(None), lr, PythonObject(1.0), PythonObject(0.0), py_item(fields, nk[].k_C), PythonObject(0.5),
                             PythonObject(0.1), py_item(fields, nk[].k_fit_intercept), py_item(fields, nk[].k_max_iter), py_item(fields, nk[].k_tol),
                             py_item(fields, nk[].k_n_iter_no_change), py_item(fields, nk[].k_shuffle), py_item(fields, nk[].k_random_state), PythonObject(None),
                             py_item(fields, nk[].k_class_weight), labels.classes, labels.codes, py_item(fields, nk[].k_batch_size), False, 256, prep,
                             class_keep, context)
    return XlClassFit(passive^, labels.classes, prep^, labels.codes, class_keep, PythonObject(None))


def xl_fit_ridge_classifier_unpack(model: PythonObject, fields: PythonObject, args: PythonObject,
                                   entries: XLinearOps, context: PythonObject) raises -> XlClassFit:
    """RidgeClassifier.fit(X, y, sample_weight): `_ridge_run` over the int32
    class codes (the +-1 targets are built in the binding)."""
    var nk = x_linear_keys()
    var name = _name(model)
    if _truth(_attr(model, "positive", PythonObject(False))):
        raise Error("mojolearn " + name + ": positive=True is not implemented")
    if not py_contains(Python.tuple("auto", "cholesky"), _attr(model, "solver", PythonObject("auto"))):
        raise Error("mojolearn " + name + ": solver must be 'auto' or 'cholesky'")
    var alpha = py_item(fields, nk[].k_alpha)
    if not _ge(alpha, PythonObject(0)):
        raise Error("mojolearn RidgeClassifier: alpha must be >= 0")
    xl_ops_require(entries)
    # lane a8c: a CV fold stays on the device (x_linear_fit_fold)
    var x = _xl_fold_matrix(args[0], name, entries, context)
    var n = x.rows
    var d = x.cols
    var labels = classical_encode_labels(args[1], _xl_scope(entries, context))
    if labels.n != n:
        raise Error("mojolearn RidgeClassifier: X and y lengths differ")
    var k = len(labels.classes)
    if k < 2:
        raise Error("mojolearn RidgeClassifier: y has one class")
    var T = 1 if k == 2 else k
    var sample_weight = args[2]
    var ip: List[Int] = [T, 1 if _truth(py_item(fields, nk[].k_fit_intercept)) else 0, 1, 0, 1]
    var fp: List[Float64] = [_float(alpha)]
    var n_out = T * d + T + 2 + 1 + 1
    var n_fw = 3 * d * d + 3 * d + T + d * T + 2 * n
    if py_item(fields, nk[].k_class_weight) is not None:
        # theirs: sample_weight * compute_sample_weight(class_weight, y),
        # 'balanced' from the unweighted class counts. The targets are the
        # codes' 32-bit words, then the n row weights the class-weight step
        # writes in RUN (`xl_class_fit_run`, which checks them as the weight
        # admission did): one vector, no second copy of the weights.
        var sw_keep = PythonObject(None)
        var sw_addr = 0
        if sample_weight is not None:
            var w = admit_weight_words(sample_weight, n, context)
            sw_keep = w.array
            sw_addr = w.address
            _ = w^
        var weights = xl_class_weights_unpack(py_item(fields, nk[].k_class_weight), labels.classes, context)
        var targets = admitted_storage_native([2 * n], "<f4", context, as_array=True)
        var t_addr = labels_address(targets)
        xl_run_join(t_addr, labels_address(labels.codes), n, 0, 0)
        var prep = XlClassPrepRequest(True, labels_address(labels.codes), sw_addr, labels_address(weights[0]),
                                      t_addr + 4 * n, n, k, weights[1], False)
        ip[3] = 1
        var weighted = xl_fit_build_x(entries, ALGO_RIDGE, x.view(), x.array, t_addr, 2 * n, targets, ip^, fp^,
                                      n_out, n_fw, 1, context)
        return XlClassFit(weighted^, labels.classes, prep^, labels.codes, weights[0], sw_keep)
    # `_ridge_run(..., codes_mode=True)`: the codes' 32-bit words, then the
    # checked weights
    var has_sw = 0
    var y = admitted_words(labels.codes, labels_address(labels.codes), n, 1)
    var yy = _xl_with_weights(y, sample_weight, n, has_sw, context)
    ip[3] = has_sw
    var fit = xl_fit_build_x(entries, ALGO_RIDGE, x.view(), x.array, yy.address, yy.rows, yy.array, ip^, fp^,
                             n_out, n_fw, 1, context)
    return XlClassFit(fit^, labels.classes, xl_class_prep_none(), labels.codes, PythonObject(None), PythonObject(None))

# SPDX-License-Identifier: Apache-2.0
# The native owner of the linear expansion estimators' public operations
# (python/mojolearn/_expansion_linear.py and the coordinate-descent pair in
# python/mojolearn/_solver_impl.py). Each public method is one call into
# x_linear_c, which validates the arguments with the Python wording, encodes
# labels (bindings/classical_labels.mojo), assembles the same int and float
# parameter lists, makes the same `x_linear_fit` / `x_linear_decision` /
# `x_linear_decision_codes` / `x_linear_class_prep` / `py2mojo_rows` /
# `cd_fit` / `cd_predict` calls with the same operands in the same order, and
# writes the fitted attributes the Python wrote (same names, types, dtypes
# and shapes). No kernel changes and no reduction order changes, so the bits
# are the ones the Python path produced.
#
# What the single call removes: the Python between the calls. HuberRegressor
# predicts its training X from the float32 matrix the fit already converted
# (no second conversion); IsotonicRegression.fit_transform transforms the
# column the fit converted; a classifier's score scores the prediction the
# same call decoded; Lasso / ElasticNet score the prediction buffer as it is.
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_wrap_array
from bindings.classical_score import classical_r2_score
from bindings.classical_labels import (
    LabelsMatrix, ClassicalLabels, labels_f32, labels_view, labels_check_rank, labels_f32_view,
    labels_check_sample_weight, labels_vector, labels_empty, labels_full_f32, labels_address, labels_cast, labels_reshape,
    labels_shape_of, labels_env, classical_encode_labels, classical_decode_labels, classical_accuracy,
)

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]

# x_linear/dispatch.mojo algorithm codes (`_expansion_linear.ALGO_*`).
comptime ALGO_SGD = 1
comptime ALGO_GLM = 2
comptime ALGO_HUBER = 3
comptime ALGO_BAYES = 4
comptime ALGO_ARD = 5
comptime ALGO_LARS = 6
comptime ALGO_QUANTILE = 7
comptime ALGO_RIDGE = 8
comptime ALGO_ISOTONIC = 11
comptime ALGO_ISOTONIC_PREDICT = 12
comptime LINK_IDENTITY = 0
comptime LINK_EXP = 1
comptime LINK_SIGMOID = 2
# core/py2mojo_rows.mojo modes (`_ROWS_*`).
comptime _ROWS_SGD_PROBA = 2
comptime _ROWS_LRCV_PROBA = 3
comptime _ROWS_HUBER_OUT = 4
# `_lbfgs_work`'s memory.
comptime _LBFGS_M = 10
# ElasticNet / Lasso refusals by name, word for word.
comptime _CD_COPY_X = "mojolearn ElasticNet: copy_X=False is refused. The design crosses to the device as a copy and the caller's array is never written (DEVIATION 880); there is no in-place arm to select"
comptime _CD_POSITIVE = "mojolearn ElasticNet: positive=True is not supported (cuML raises UnsupportedOnGPU for it too, elastic_net.py:143)"
comptime _CD_PRECOMPUTE = "mojolearn ElasticNet: precompute is not supported (cuML raises UnsupportedOnGPU for it too, elastic_net.py:149)"
comptime _CD_QN = "mojolearn ElasticNet: solver='qn' is not implemented. cuML's solver='auto' picks 'qn' only for SPARSE input (elastic_net.py:243-244), and 'qn' under an l1 penalty is OWL-QN (min_owlqn, qn_solvers.cuh), which glm/NOT_IMPLEMENTED.tsv lists as not implemented. solver='cd' (cuML's 'auto' for dense input) is the implemented arm; see solver/NOT_IMPLEMENTED.tsv"
comptime _CD_RANDOM = "mojolearn ElasticNet: selection='random' is REFUSED BY NAME. cuML seeds std::mt19937(0) and shuffles the coordinate order with std::shuffle each epoch (cd.cuh:198-199, :222), and the C++ standard does not specify std::shuffle's algorithm, so their permutation is a function of the toolchain rather than of the seed and cannot be gated bitwise or certified across vendors. An exact implementation is DEVIATION 611, reserved and not spent; see solver/NOT_IMPLEMENTED.tsv"
comptime _CD_SPARSE = "mojolearn ElasticNet: sparse X is refused. cdFit is dense only; cuML routes sparse input to solver='qn' and raises for solver='cd' (elastic_net.py:265-269), and 'qn' is not implemented"
comptime _CD_WARM = "mojolearn ElasticNet: warm_start=True is not supported (cuML raises UnsupportedOnGPU for it too, elastic_net.py:146). cdFit reads coef as its starting point and solver/estimator.mojo zeroes it on every fit"


# ------------------------------------------------------------ Python values


def _b() raises -> PythonObject:
    return Python.import_module("builtins")


def _op() raises -> PythonObject:
    return Python.import_module("operator")


def _truth(value: PythonObject) raises -> Bool:
    """`bool(value)`."""
    return Bool(py=py_call(_b().bool, value))


def _float(value: PythonObject) raises -> Float64:
    """`float(value)`."""
    return Float64(py=py_call(_b().float, value))


def _int(value: PythonObject) raises -> Int:
    """`int(value)`."""
    return Int(py=py_call(_b().int, value))


def _repr(value: PythonObject) raises -> String:
    return String(py=py_call(_b().repr, value))


def _str(value: PythonObject) raises -> String:
    """`str(value)`: what an f-string prints for it."""
    return String(py=py_call(_b().str, value))


def _gt(a: PythonObject, b: PythonObject) raises -> Bool:
    return _truth(py_call(_op().gt, a, b))


def _ge(a: PythonObject, b: PythonObject) raises -> Bool:
    return _truth(py_call(_op().ge, a, b))


def _lt(a: PythonObject, b: PythonObject) raises -> Bool:
    return _truth(py_call(_op().lt, a, b))


def _le(a: PythonObject, b: PythonObject) raises -> Bool:
    return _truth(py_call(_op().le, a, b))


def _eq(a: PythonObject, b: PythonObject) raises -> Bool:
    return _truth(py_call(_op().eq, a, b))


def _isinstance(value: PythonObject, cls: PythonObject) raises -> Bool:
    return Bool(py=py_call(_b().isinstance, value, cls))


def _is_int(value: PythonObject) raises -> Bool:
    """`isinstance(value, int) and not isinstance(value, bool)`."""
    return _isinstance(value, _b().int) and not _isinstance(value, _b().bool)


def _name(model: PythonObject) raises -> String:
    return String(py=py_type(model).__name__)


def _class_attr(model: PythonObject, name: String) raises -> PythonObject:
    """A class attribute of the model's class (a plain class constant such as
    `_power` or `_loss_name`), read through `getattr` on the class."""
    return py_call(_b().getattr, py_type(model), PythonObject(name))


def _attr(model: PythonObject, name: String, default: PythonObject) raises -> PythonObject:
    """`getattr(model, name, default)`."""
    return py_call(_b().getattr, model, PythonObject(name), default)


def _sorted_repr(table: PythonObject) raises -> String:
    """`f"{sorted(table)}"` of a name table."""
    return _str(py_call(_b().sorted, table))


# ------------------------------------------------------------ bindings


def _xl_mode(fields: PythonObject, context: PythonObject) raises -> String:
    """`_backend.binding`'s mode rule: the estimator's numeric_mode, or the
    process default."""
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not _b().str:
        raise Error("numeric_mode must be fast, identical or None")
    var clean = String(py=py_call(py_call(mode.strip).lower))
    if clean != "identical" and clean != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    return clean


def _xl_entries(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """The estimator's connected x_linear entries (`est._bind(_BINDING)`),
    with the binding's compiled-mode check, or the refusal recorded at
    import ("<Class>: <message>")."""
    var mode = _xl_mode(fields, context)
    var entries = context["xl_bindings"][PythonObject(mode)]
    if py_type(entries) is _b().str:
        raise Error(String(py=entries))
    var reads = py_get(entries, "x_linear_numeric_mode")
    if reads is not None:
        var code = Int(py=py_call(reads))
        if code != (1 if mode == "identical" else 0):
            raise Error("RuntimeError: mojolearn: _mojolearn_x_linear was compiled for "
                        + ("identical" if code == 1 else "fast") + ", but this call requested " + mode
                        + "; rebuild the " + mode + " binding")
    return entries


def _xl_check_fitted(fields: PythonObject, model: PythonObject) raises:
    if not py_contains(fields, "coef_"):
        raise Error("RuntimeError: mojolearn " + _name(model) + ": call fit first")


# ------------------------------------------------------------ arrays


def _xl_word(out: PythonObject, index: Int) raises -> Float32:
    return _F32P(unsafe_from_address=labels_address(out))[index]


def _xl_float(out: PythonObject, index: Int) raises -> PythonObject:
    """`float(vals[index])`: the float32 word as a Python float."""
    return PythonObject(Float64(_xl_word(out, index)))


def _xl_count(out: PythonObject, index: Int) raises -> Int:
    """`int(vals[index])`: the float32 word truncated toward zero."""
    return Int(_xl_word(out, index))


def _xl_slice(out: PythonObject, offset: Int, count: Int, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Array.from_list(vals[offset:offset + count], '<f4')` (shaped as the
    Python list was nested): one byte copy of the float32 words."""
    var array = labels_empty(shape, "<f4", context)
    if count > 0:
        memcpy(dest=_F32P(unsafe_from_address=labels_address(array)),
               src=_F32P(unsafe_from_address=labels_address(out)) + offset, count=count)
    return array


def _xl_join(first: Int, first_count: Int, second: Int, second_count: Int, context: PythonObject) raises -> PythonObject:
    """`_concat_f32(a, b)`: a's then b's 32-bit words in one float32 vector
    (two byte copies)."""
    var out = labels_vector(first_count + second_count, "<f4", context)
    var o = _F32P(unsafe_from_address=labels_address(out))
    if first_count > 0:
        memcpy(dest=o, src=_F32P(unsafe_from_address=first), count=first_count)
    if second_count > 0:
        memcpy(dest=o + first_count, src=_F32P(unsafe_from_address=second), count=second_count)
    return out


def _xl_floats(values: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Array.from_list([float(v) ...], '<f4')` of a Python list of floats:
    `array.array('f', values)` rounds each once, as from_list did."""
    var store = py_call(context["array_type"], PythonObject("f"), values)
    return classical_wrap_array(store, Python.tuple(len(store)), "<f4", "C", context)


def _xl_matrix(source: PythonObject, context: PythonObject) raises -> LabelsMatrix:
    """`_matrix(X)`: `as_f32_c(X, ndim=2, name='X')` and its refusals."""
    return labels_f32(source, 2, "C", "X", context)


def _xl_vector(source: PythonObject, n: Int, context: PythonObject) raises -> LabelsMatrix:
    """`_vector(y, n)`."""
    var y = labels_f32(source, 1, "C", "y", context)
    if y.rows != n:
        raise Error("mojolearn: X and y lengths differ")
    return y^


def _xl_column(source: PythonObject, name: String, context: PythonObject) raises -> LabelsMatrix:
    """IsotonicRegression's `_column`: one feature, 1-D or (n, 1)."""
    var view = labels_view(source, name, context)
    labels_check_rank(view, -1, name)
    var nd = view.ndim()
    if not (nd == 1 or (nd == 2 and view.shape[1] == 1)):
        raise Error("mojolearn IsotonicRegression: " + name + " must be 1-D or of shape (n, 1)")
    return labels_f32_view(view^, "C", context)


def _xl_words(value: PythonObject, name: String, context: PythonObject) raises -> LabelsMatrix:
    """`as_f32_c(value, ndim=None)` of a fitted array, or a Python number as
    one float32 word (`full((1,), float(value), '<f4')`)."""
    if py_type(value) is _b().float or py_type(value) is _b().int:
        var one = labels_full_f32(1, _float(value), context)
        return LabelsMatrix(one, labels_address(one), 1, 1, True)
    return labels_f32(value, -1, "C", name, context)


def _xl_block(coef: PythonObject, intercept: PythonObject, context: PythonObject) raises -> Tuple[PythonObject, Int]:
    """`_coef_block(coef, intercept)`: [coef row | intercept] per class as
    one float32 vector of k rows of d + 1 words (byte copies)."""
    var c = _xl_words(coef, "coef_", context)
    var b = _xl_words(intercept, "intercept_", context)
    var k = b.rows * b.cols
    var d = (c.rows * c.cols) // max(k, 1)
    var out = labels_vector(k * (d + 1), "<f4", context)
    var o = _F32P(unsafe_from_address=labels_address(out))
    var cp = _F32P(unsafe_from_address=c.addr)
    var bp = _F32P(unsafe_from_address=b.addr)
    for j in range(k):  # driver-loop: over the fitted classes' coefficient rows (model structure)
        memcpy(dest=o + j * (d + 1), src=cp + j * d, count=d)
        memcpy(dest=o + j * (d + 1) + d, src=bp + j, count=1)
    _ = c^
    _ = b^
    return (out, k)


def _xl_width(x: LabelsMatrix, fields: PythonObject, model: PythonObject) raises:
    var fitted = fields["n_features_in_"]
    if x.cols != _int(fitted):
        raise Error("mojolearn " + _name(model) + ": X has " + String(x.cols) + " features, the model was fitted with "
                    + _str(fitted))


def _xl_decision(model: PythonObject, x: LabelsMatrix, coef: PythonObject, intercept: PythonObject, link: Int,
                 squeeze: Bool, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_decision(est, X, coef, intercept, link)` over an X already read:
    link(X @ W^T + b) on the binding, (n, k) float32, or (n,) when
    `squeeze` and k == 1 (the `reshape` the callers applied)."""
    var block = _xl_block(coef, intercept, context)
    var wb = block[0]
    var k = block[1]
    var shape = Python.tuple(x.rows) if squeeze and k == 1 else Python.tuple(x.rows, k)
    var out = labels_empty(shape, "<f4", context)
    _ = py_call_typed(entries["x_linear_decision"], PythonObject(x.addr), PythonObject(labels_address(wb)),
                      Python.list(PythonObject(x.rows), PythonObject(x.cols), PythonObject(k), PythonObject(link)),
                      PythonObject(labels_address(out)))
    _ = wb^
    return out


def _xl_codes(model: PythonObject, x: LabelsMatrix, coef: PythonObject, intercept: PythonObject, strict: Bool,
              below: Int, above: Int, entries: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_decision_codes`: each row's int32 class code on the binding."""
    var block = _xl_block(coef, intercept, context)
    var wb = block[0]
    var k = block[1]
    var out = labels_vector(x.rows, "<i4", context)
    _ = py_call_typed(entries["x_linear_decision_codes"], PythonObject(x.addr), PythonObject(labels_address(wb)),
                      Python.list(PythonObject(x.rows), PythonObject(x.cols), PythonObject(k),
                                  PythonObject(LINK_IDENTITY), PythonObject(1 if strict else 0), PythonObject(below),
                                  PythonObject(above)),
                      PythonObject(labels_address(out)))
    _ = wb^
    return out


def _xl_run(entries: PythonObject, algo: Int, x_addr: Int, x_size: Int, n: Int, d: Int, y_addr: Int, y_size: Int,
            ip: PythonObject, fp: PythonObject, n_out: Int, n_fw: Int, n_iw: Int,
            context: PythonObject) raises -> PythonObject:
    """`_run_array`: one `x_linear_fit` call; the flat float32 result. The
    targets are `y_size` 32-bit words at `y_addr` (the caller keeps them
    alive); `y_size` 0 is no targets (one zero word, as `zeros((1,))`)."""
    var out = labels_vector(n_out, "<f4", context)
    var zero = labels_vector(1, "<f4", context)
    var ya = y_addr if y_size > 0 else labels_address(zero)
    _ = py_call_typed(entries["x_linear_fit"], PythonObject(algo), PythonObject(x_addr), PythonObject(ya),
                      Python.list(PythonObject(n), PythonObject(d), PythonObject(x_size), PythonObject(y_size),
                                  PythonObject(n_out), PythonObject(max(n_fw, 1)), PythonObject(max(n_iw, 1)),
                                  PythonObject(len(ip)), PythonObject(len(fp))),
                      ip, fp, PythonObject(labels_address(out)))
    _ = zero^
    return out


def _xl_with_weights(y: LabelsMatrix, sample_weight: PythonObject, n: Int, mut has_sw: Int,
                     context: PythonObject) raises -> LabelsMatrix:
    """`_with_weights(yv, sample_weight, n)`: the targets then the checked
    weights in one float32 vector (has_sw 1) with a sample_weight, the
    targets as they are (has_sw 0) without."""
    if sample_weight is None:
        has_sw = 0
        return LabelsMatrix(y.keep, y.addr, y.rows * y.cols, 1, True)
    var w = labels_check_sample_weight(sample_weight, n, context)
    var joined = _xl_join(y.addr, y.rows * y.cols, w.addr, n, context)
    _ = w^
    has_sw = 1
    return LabelsMatrix(joined, labels_address(joined), y.rows * y.cols + n, 1, True)


def _xl_ints(values: List[Int]) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(values)):  # driver-loop: over the route's integer hyperparameters
        _ = py_call(out.append, PythonObject(values[i]))
    return out


def _xl_reals(values: List[Float64]) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(values)):  # driver-loop: over the route's float hyperparameters
        _ = py_call(out.append, PythonObject(values[i]))
    return out


def _lbfgs_work(p: Int) -> Int:
    return 4 * p + 2 * _LBFGS_M * p + 2 * _LBFGS_M


# ------------------------------------------------------------ class weights


def _xl_class_prep(model: PythonObject, entries: PythonObject, class_weight: PythonObject, classes: PythonObject,
                   codes: PythonObject, sw_addr: Int, weighted: Bool, rows: Bool,
                   context: PythonObject) raises -> Tuple[PythonObject, PythonObject, Int]:
    """`_class_prep(est, class_weight, classes, codes, sample_weight,
    weighted, rows)`: (the k class weights, the n row weights or None, the
    largest unweighted class count), on the binding (`x_linear_class_prep`)."""
    var b = _b()
    var labels = py_call(b.list, classes)
    var k = len(labels)
    var n = Int(py=codes.shape[0])
    var balanced = _eq(class_weight, PythonObject("balanced"))
    var cw = PythonObject(None)
    if balanced:
        cw = labels_vector(k, "<f4", context)
    elif class_weight is None:
        cw = labels_full_f32(k, 1.0, context)
    elif _isinstance(class_weight, b.dict):
        var keys = py_call(b.list, class_weight)
        for i in range(len(keys)):  # driver-loop: over the class_weight entries (named labels)
            if not py_contains(labels, keys[i]):
                raise Error("The classes, " + _str(Python.list(keys[i])) + ", are not in class_weight")
        var weights = Python.list()
        for i in range(k):  # driver-loop: over the fitted classes (model structure)
            _ = py_call(weights.append, PythonObject(_float(py_call(class_weight.get, labels[i], PythonObject(1.0)))))
        cw = _xl_floats(weights, context)
    else:
        raise Error("mojolearn: class_weight must be None, 'balanced' or a dict")
    var out = labels_vector(n, "<f4", context) if rows else PythonObject(None)
    var largest = py_call_typed(entries["x_linear_class_prep"], PythonObject(labels_address(codes)), PythonObject(sw_addr),
                                PythonObject(labels_address(cw)),
                                Python.list(PythonObject(n), PythonObject(k), PythonObject(1 if balanced else 0),
                                            PythonObject(1 if (balanced and weighted and sw_addr != 0) else 0)),
                                PythonObject(0 if out is None else labels_address(out)))
    return (cw, out, _int(largest))


# ------------------------------------------------------------------- SGD


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
    var half = 2147483648
    return (lo - 2 * half if lo >= half else lo, hi - 2 * half if hi >= half else hi)


def _xl_sgd_auto_sum(batch_size: PythonObject, loss: PythonObject, learning_rate: PythonObject,
                     context: PythonObject) raises -> Bool:
    """`_sgd_auto_sum`."""
    return (_isinstance(batch_size, _b().str) and _eq(batch_size, PythonObject("auto"))
            and py_contains(context["xl_sgd_sum_losses"], loss)
            and not py_contains(Python.tuple("pa1", "pa2"), learning_rate))


def _xl_sgd(model: PythonObject, fields: PythonObject, entries: PythonObject, x: LabelsMatrix, y_keep: PythonObject,
            y_addr: Int,
            n_classes: Int, loss_code: Int, penalty: PythonObject, lr: PythonObject, alpha: PythonObject,
            l1_ratio: PythonObject, eta0: PythonObject, power_t: PythonObject, epsilon: PythonObject,
            fit_intercept: PythonObject, max_iter: PythonObject, tol: PythonObject,
            n_iter_no_change: PythonObject, shuffle: PythonObject, random_state: PythonObject,
            sample_weight: PythonObject, class_weight: PythonObject, classes: PythonObject, codes: PythonObject,
            batch_size: PythonObject, var batch_sum: Bool, auto_cap: Int,
            context: PythonObject) raises -> Tuple[PythonObject, PythonObject]:
    """`_sgd_fit`: the SGD fit on the binding; (coef (problems, d),
    intercept (problems,)). The n float32 targets or class codes are at
    `y_addr` (`y_keep` owns them); 0 is no targets (SGDOneClassSVM: zeros)."""
    var name = _name(model)
    var n = x.rows
    var d = x.cols
    var penalties = context["xl_sgd_penalty"]
    var rates = context["xl_sgd_lr"]
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
        targets = labels_vector(n, "<f4", context)
        t_addr = labels_address(targets)
    var has_sw = 0
    var t = LabelsMatrix(targets, t_addr, n, 1, True)
    var yy = _xl_with_weights(t, sample_weight, n, has_sw, context)
    var has_cw = 0
    if class_weight is not None and n_classes >= 2:
        # k scalars: the fit's parameters
        var prep = _xl_class_prep(model, entries, class_weight, classes, codes, 0, False, False, context)
        var cwp = _F32P(unsafe_from_address=labels_address(prep[0]))
        if n_classes == 2:
            fp.append(Float64(cwp[1]))
            fp.append(Float64(cwp[0]))
        else:
            for c in range(n_classes):  # driver-loop: over the classes' positive weights (model structure)
                fp.append(Float64(cwp[c]))
            for c in range(n_classes):  # driver-loop: over the classes' negative weights (model structure)
                fp.append(1.0)
        has_cw = 1
    var auto = _isinstance(batch_size, _b().str) and _eq(batch_size, PythonObject("auto"))
    var batch = -1
    if auto:
        batch = -auto_cap
    elif _truth(batch_size) and (not _is_int(batch_size) or _lt(batch_size, PythonObject(1))):
        raise Error("mojolearn " + name + ": batch_size must be 'auto', 0 or a positive int")
    elif batch_size is not None:
        batch = _int(batch_size)
    var override = labels_env("MOJOLEARN_SGD_BATCH_SUM", "")
    if override == "0" or override == "1":
        batch_sum = override == "1"
    ip.append(has_sw)
    ip.append(has_cw)
    ip.append(batch)
    ip.append(1 if batch_sum else 0)
    var out = _xl_run(entries, ALGO_SGD, x.addr, n * d, n, d, yy.addr, yy.rows, _xl_ints(ip), _xl_reals(fp),
                      problems * d + problems + 2, problems * (n + d + 1), problems * n, context)
    var n_out = problems * d + problems + 2
    if _xl_word(out, n_out - 1) != 0:
        raise Error("Floating-point under-/overflow occurred. Scaling input data with "
                    + "StandardScaler or MinMaxScaler might help.")
    var coef = _xl_slice(out, 0, problems * d, Python.tuple(problems, d), context)
    var intercept = _xl_slice(out, problems * d, problems, Python.tuple(problems), context)
    var n_iter = _xl_count(out, n_out - 2)
    py_setitem(fields, "n_iter_", PythonObject(n_iter))
    py_setitem(fields, "t_", PythonObject(Float64(n_iter * n + 1)))
    py_setitem(fields, "n_features_in_", PythonObject(d))
    _ = yy^
    _ = targets^
    return (coef, intercept)


def xl_fit_sgd_regressor(model: PythonObject, args: PythonObject, passive: Bool, context: PythonObject) raises -> PythonObject:
    """SGDRegressor.fit(X, y, sample_weight) and (`passive`)
    PassiveAggressiveRegressor.fit(X, y)."""
    var fields = py_fields(model)
    _xl_sgd_refuse(model, fields["early_stopping"], fields["average"], fields["warm_start"])
    var losses = context["xl_sgd_reg_loss"]
    var loss = fields["loss"]
    if passive:
        if not py_contains(Python.tuple("epsilon_insensitive", "squared_epsilon_insensitive"), loss):
            raise Error("mojolearn PassiveAggressiveRegressor: loss must be 'epsilon_insensitive' "
                        + "or 'squared_epsilon_insensitive'")
        if not _gt(fields["C"], PythonObject(0)):
            raise Error("mojolearn PassiveAggressiveRegressor: C must be > 0")
    elif not py_contains(losses, loss):
        raise Error("mojolearn SGDRegressor: loss must be one of " + _sorted_repr(losses))
    var entries = _xl_entries(fields, context)
    var x = _xl_matrix(args[0], context)
    var y = _xl_vector(args[1], x.rows, context)
    var fit: Tuple[PythonObject, PythonObject]
    if passive:
        var lr = PythonObject("pa1") if _eq(loss, PythonObject("epsilon_insensitive")) else PythonObject("pa2")
        fit = _xl_sgd(model, fields, entries, x, y.keep, y.addr, 0, _int(losses["epsilon_insensitive"]),
                      PythonObject(None), lr, PythonObject(1.0), PythonObject(0.0), fields["C"], PythonObject(0.5),
                      fields["epsilon"], fields["fit_intercept"], fields["max_iter"], fields["tol"],
                      fields["n_iter_no_change"], fields["shuffle"], fields["random_state"], PythonObject(None),
                      PythonObject(None), PythonObject(None), PythonObject(None), fields["batch_size"], False, 256,
                      context)
    else:
        fit = _xl_sgd(model, fields, entries, x, y.keep, y.addr, 0, _int(losses[loss]), fields["penalty"],
                      fields["learning_rate"], fields["alpha"], fields["l1_ratio"], fields["eta0"], fields["power_t"],
                      fields["epsilon"], fields["fit_intercept"], fields["max_iter"], fields["tol"],
                      fields["n_iter_no_change"], fields["shuffle"], fields["random_state"], args[2],
                      PythonObject(None), PythonObject(None), PythonObject(None), fields["batch_size"], False, 4096,
                      context)
    py_setitem(fields, "coef_", labels_reshape(fit[0], Python.tuple(x.cols), context))
    py_setitem(fields, "intercept_", fit[1])
    _ = x^
    _ = y^
    return model


def xl_fit_one_class(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """SGDOneClassSVM.fit(X, y=None, sample_weight=None)."""
    var fields = py_fields(model)
    _xl_sgd_refuse(model, PythonObject(False), fields["average"], fields["warm_start"])
    var nu = fields["nu"]
    if not (_lt(PythonObject(0), nu) and _le(nu, PythonObject(1))):
        raise Error("mojolearn SGDOneClassSVM: nu must be in (0, 1]")
    if py_contains(Python.tuple("pa1", "pa2"), fields["learning_rate"]):
        raise Error("mojolearn SGDOneClassSVM: learning_rate must be constant, optimal, invscaling or adaptive")
    var entries = _xl_entries(fields, context)
    var x = _xl_matrix(args[0], context)
    var fit = _xl_sgd(model, fields, entries, x, PythonObject(None), 0, 1, _int(context["xl_sgd_clf_loss"]["hinge"]),
                      PythonObject("l2"), fields["learning_rate"], nu, PythonObject(0.0), fields["eta0"],
                      fields["power_t"], PythonObject(0.1), fields["fit_intercept"], fields["max_iter"], fields["tol"],
                      PythonObject(5), fields["shuffle"], fields["random_state"], args[2], PythonObject(None),
                      PythonObject(None), PythonObject(None), fields["batch_size"], False, 4096, context)
    py_setitem(fields, "coef_", labels_reshape(fit[0], Python.tuple(x.cols), context))
    # the binding returns offset_ itself for one class
    py_setitem(fields, "offset_", _xl_slice(fit[1], 0, 1, Python.tuple(1), context))
    _ = x^
    return model


def _xl_offset(fields: PythonObject, context: PythonObject) raises -> Float64:
    """`float(self.offset_[0])`."""
    var off = _xl_words(fields["offset_"], "offset_", context)
    var value = Float64(_F32P(unsafe_from_address=off.addr)[0])
    _ = off^
    return value


def xl_one_class(model: PythonObject, operation: String, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """SGDOneClassSVM.decision_function / score_samples / predict."""
    var fields = py_fields(model)
    _xl_check_fitted(fields, model)
    var entries = _xl_entries(fields, context)
    var x = _xl_matrix(source, context)
    _xl_width(x, fields, model)
    var offset = _xl_offset(fields, context)
    var result = PythonObject(None)
    if operation == "predict":
        # `Array.from_list([-offset], '<f4')`; codes -1 / +1, as int64
        var icpt = labels_full_f32(1, -offset, context)
        var codes = _xl_codes(model, x, fields["coef_"], icpt, False, -1, 1, entries, context)
        var wide = labels_vector(x.rows, "<i8", context)
        labels_cast(labels_address(codes), "<i4", labels_address(wide), "<i8", x.rows, context)
        result = wide
    else:
        var intercept = PythonObject(-offset) if operation == "decision_function" else PythonObject(0.0)
        result = _xl_decision(model, x, fields["coef_"], intercept, LINK_IDENTITY, True, entries, context)
    _ = x^
    return result


# -------------------------------------------------------------- predict


def xl_regressor_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_LinearRegressorMixin.predict(X)` (BayesianRidge and ARDRegression
    pass return_std and refuse first)."""
    var fields = py_fields(model)
    if len(args) > 1:
        _xl_bayes_refuse(model, fields, args[1])
    _xl_check_fitted(fields, model)
    var entries = _xl_entries(fields, context)
    var x = _xl_matrix(args[0], context)
    _xl_width(x, fields, model)
    var link = LINK_IDENTITY
    var stored = py_get(fields, "_link")
    if stored is None:
        stored = _attr(model, "_link", PythonObject(LINK_IDENTITY))
    link = _int(stored)
    var out = _xl_decision(model, x, fields["coef_"], fields["intercept_"], link, True, entries, context)
    _ = x^
    return out


def xl_classifier_decision(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_LinearClassifierMixin.decision_function(X)`."""
    var fields = py_fields(model)
    _xl_check_fitted(fields, model)
    var entries = _xl_entries(fields, context)
    var x = _xl_matrix(source, context)
    _xl_width(x, fields, model)
    var out = _xl_decision(model, x, fields["coef_"], fields["intercept_"], LINK_IDENTITY, True, entries, context)
    _ = x^
    return out


def xl_classifier_predict(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_LinearClassifierMixin.predict(X)`: the codes on the binding, the
    labels decoded natively (`decode_labels(classes_, codes)`)."""
    var fields = py_fields(model)
    _xl_check_fitted(fields, model)
    var entries = _xl_entries(fields, context)
    var x = _xl_matrix(source, context)
    _xl_width(x, fields, model)
    var codes = _xl_codes(model, x, fields["coef_"], fields["intercept_"], True, 0, 1, entries, context)
    _ = x^
    return classical_decode_labels(fields["classes_"], codes, context)


def xl_predict_proba(model: PythonObject, source: PythonObject, lrcv: Bool, context: PythonObject) raises -> PythonObject:
    """SGDClassifier.predict_proba (the sigmoid scores) and
    LogisticRegressionCV.predict_proba (the decision scores), finished per
    row on the binding (`py2mojo_rows`)."""
    var fields = py_fields(model)
    if not lrcv and not _eq(fields["loss"], PythonObject("log_loss")):
        raise Error("AttributeError: probability estimates are not available for loss=" + _repr(fields["loss"]))
    _xl_check_fitted(fields, model)
    var entries = _xl_entries(fields, context)
    var x = _xl_matrix(source, context)
    _xl_width(x, fields, model)
    var scores = _xl_decision(model, x, fields["coef_"], fields["intercept_"],
                              LINK_IDENTITY if lrcv else LINK_SIGMOID, lrcv, entries, context)
    var k = len(fields["intercept_"])
    var n = x.rows
    var out = labels_empty(Python.tuple(n, 2 if k == 1 else k), "<f4", context)
    if n > 0:
        _ = py_call_typed(entries["py2mojo_rows"], PythonObject(_ROWS_LRCV_PROBA if lrcv else _ROWS_SGD_PROBA),
                          PythonObject(labels_address(scores)), PythonObject(labels_address(out)),
                          Python.list(PythonObject(n), PythonObject(k)))
    _ = scores^
    _ = x^
    return out


def xl_classifier_score(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_LinearClassifierMixin.score(X, y)`: predict, then
    `accuracy_fraction(y, prediction)` in the same call."""
    var prediction = xl_classifier_predict(model, args[0], context)
    return PythonObject(classical_accuracy(args[1], prediction, PythonObject(None), PythonObject(None), context))


# ------------------------------------------------------------ regressors


def _xl_set_linear(fields: PythonObject, out: PythonObject, d: Int, context: PythonObject) raises:
    """coef_ = vals[:d] (float32 Array), intercept_ = float(vals[d])."""
    py_setitem(fields, "coef_", _xl_slice(out, 0, d, Python.tuple(d), context))
    py_setitem(fields, "intercept_", _xl_float(out, d))


def _xl_tweedie_link(fields: PythonObject) raises -> Int:
    """TweedieRegressor._link_code."""
    var p = _float(fields["power"])
    if p < 0 or (0 < p and p < 1):
        raise Error("mojolearn TweedieRegressor: power=" + _str(PythonObject(p)) + " is not implemented (power 0 or >= 1)")
    var link = fields["link"]
    if _eq(link, PythonObject("auto")):
        return 0 if p == 0 else 1
    if _eq(link, PythonObject("identity")):
        if p != 0:
            raise Error("mojolearn TweedieRegressor: the identity link is implemented for power=0 only")
        return 0
    if _eq(link, PythonObject("log")):
        return 1
    raise Error("mojolearn TweedieRegressor: link must be 'auto', 'identity' or 'log'")


def xl_fit_glm(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_GLMBase.fit(X, y, sample_weight)`: Poisson, Gamma and Tweedie."""
    var fields = py_fields(model)
    var name = _name(model)
    if not py_contains(Python.tuple("lbfgs", "newton-cholesky"), fields["solver"]):
        raise Error("mojolearn " + name + ": solver must be 'lbfgs' or 'newton-cholesky'")
    if _truth(fields["warm_start"]):
        raise Error("mojolearn " + name + ": warm_start is not implemented")
    if not _ge(fields["alpha"], PythonObject(0)):
        raise Error("mojolearn " + name + ": alpha must be >= 0")
    var entries = _xl_entries(fields, context)
    var x = _xl_matrix(args[0], context)
    var n = x.rows
    var d = x.cols
    var y = _xl_vector(args[1], n, context)
    # TweedieRegressor reads its own power and link; Poisson and Gamma fix
    # them (`_power`, log link)
    var tweedie = py_contains(fields, "power") and py_contains(fields, "link")
    var link = _xl_tweedie_link(fields) if tweedie else 1
    var power = _float(fields["power"]) if tweedie else _float(_class_attr(model, "_power"))
    var m = d + 1
    var has_sw = 0
    var yy = _xl_with_weights(y, args[2], n, has_sw, context)
    var ip: List[Int] = [_int(fields["max_iter"]), 1 if _truth(fields["fit_intercept"]) else 0, link, has_sw]
    var fp: List[Float64] = [power, _float(fields["alpha"]), _float(fields["tol"])]
    var out = _xl_run(entries, ALGO_GLM, x.addr, n * d, n, d, yy.addr, yy.rows, _xl_ints(ip), _xl_reals(fp),
                      d + 3, 3 * n + m * m + 3 * m, 1, context)
    if _xl_word(out, d + 2) < 0:
        var loss_name = _class_attr(model, "_loss_name")
        raise Error("Some value(s) of y are out of the valid range of the loss '"
                    + (_str(loss_name) if _truth(loss_name) else String("HalfTweedieLoss")) + "'.")
    _xl_set_linear(fields, out, d, context)
    py_setitem(fields, "n_iter_", PythonObject(_xl_count(out, d + 1)))
    py_setitem(fields, "n_features_in_", PythonObject(d))
    py_setitem(fields, "_link", PythonObject(LINK_EXP if link == 1 else LINK_IDENTITY))
    _ = x^
    _ = y^
    _ = yy^
    return model


def xl_fit_huber(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """HuberRegressor.fit(X, y, sample_weight): the fit, then the outlier
    mask of the training rows from the X the fit already holds."""
    var fields = py_fields(model)
    var epsilon = fields["epsilon"]
    if not _ge(epsilon, PythonObject(1.0)):
        raise Error("mojolearn HuberRegressor: epsilon must be >= 1.0")
    if _truth(fields["warm_start"]):
        raise Error("mojolearn HuberRegressor: warm_start is not implemented")
    var entries = _xl_entries(fields, context)
    var x = _xl_matrix(args[0], context)
    var n = x.rows
    var d = x.cols
    var y = _xl_vector(args[1], n, context)
    var intercept = _truth(fields["fit_intercept"])
    var p = d + 2 if intercept else d + 1
    var has_sw = 0
    var yy = _xl_with_weights(y, args[2], n, has_sw, context)
    var ip: List[Int] = [_int(fields["max_iter"]), 1 if intercept else 0, has_sw]
    var fp: List[Float64] = [_float(epsilon), _float(fields["alpha"]), _float(fields["tol"])]
    var out = _xl_run(entries, ALGO_HUBER, x.addr, n * d, n, d, yy.addr, yy.rows, _xl_ints(ip), _xl_reals(fp),
                      d + 4 + p, _lbfgs_work(p) + n, 1, context)
    _xl_set_linear(fields, out, d, context)
    var scale = _xl_float(out, d + 1)
    py_setitem(fields, "scale_", scale)
    py_setitem(fields, "n_iter_", PythonObject(_xl_count(out, d + 2)))
    py_setitem(fields, "n_features_in_", PythonObject(d))
    # `self.predict(a)`: the fitted model on the float32 X this call holds
    var link = _int(_attr(model, "_link", PythonObject(LINK_IDENTITY)))
    var pred = _xl_decision(model, x, fields["coef_"], fields["intercept_"], link, True, entries, context)
    var thr = _float(py_call(_op().mul, scale, epsilon))
    # |y - pred| > thr in the binding (core/py2mojo_rows.mojo ROWS_HUBER_OUT)
    var flags = labels_vector(n, "<u1", context)
    _ = py_call_typed(entries["py2mojo_rows"], PythonObject(_ROWS_HUBER_OUT), PythonObject(y.addr),
                      PythonObject(labels_address(flags)),
                      Python.list(PythonObject(n), PythonObject(1), PythonObject(labels_address(pred)), PythonObject(thr)))
    # scikit-learn's bool mask, as a list: one C pass of `bool` over the bytes
    var b = _b()
    py_setitem(fields, "outliers_", py_call(b.list, py_call(b.map, b.bool, py_call(flags._store.tolist))))
    _ = pred^
    _ = flags^
    _ = x^
    _ = y^
    _ = yy^
    return model


def _xl_bayes_refuse(model: PythonObject, fields: PythonObject, return_std: PythonObject) raises:
    """`_bayes_refuse`."""
    var name = _name(model)
    if _truth(fields["compute_score"]):
        raise Error("mojolearn " + name + ": compute_score is not implemented")
    if _truth(return_std):
        raise Error("mojolearn " + name + ": predict(return_std=True) is not implemented")


def _xl_or(value: PythonObject, default: Float64) raises -> Float64:
    return default if value is None else _float(value)


def xl_fit_bayes(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """BayesianRidge.fit(X, y, sample_weight)."""
    var fields = py_fields(model)
    _xl_bayes_refuse(model, fields, PythonObject(False))
    var entries = _xl_entries(fields, context)
    var x = _xl_matrix(args[0], context)
    var n = x.rows
    var d = x.cols
    var y = _xl_vector(args[1], n, context)
    var has_sw = 0
    var yy = _xl_with_weights(y, args[2], n, has_sw, context)
    var ip: List[Int] = [_int(fields["max_iter"]), 1 if _truth(fields["fit_intercept"]) else 0, has_sw]
    var fp: List[Float64] = [_float(fields["tol"]), _float(fields["alpha_1"]), _float(fields["alpha_2"]),
                             _float(fields["lambda_1"]), _float(fields["lambda_2"]),
                             _xl_or(fields["alpha_init"], -1.0), _xl_or(fields["lambda_init"], -1.0)]
    var out = _xl_run(entries, ALGO_BAYES, x.addr, n * d, n, d, yy.addr, yy.rows, _xl_ints(ip), _xl_reals(fp),
                      d + 4, 3 * d * d + 5 * d + n, 1, context)
    _xl_set_linear(fields, out, d, context)
    py_setitem(fields, "alpha_", _xl_float(out, d + 1))
    py_setitem(fields, "lambda_", _xl_float(out, d + 2))
    py_setitem(fields, "n_iter_", PythonObject(_xl_count(out, d + 3)))
    py_setitem(fields, "n_features_in_", PythonObject(d))
    _ = x^
    _ = y^
    _ = yy^
    return model


def xl_fit_ard(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """ARDRegression.fit(X, y)."""
    var fields = py_fields(model)
    _xl_bayes_refuse(model, fields, PythonObject(False))
    var entries = _xl_entries(fields, context)
    var x = _xl_matrix(args[0], context)
    var n = x.rows
    var d = x.cols
    if n < 2:
        raise Error("mojolearn ARDRegression: at least 2 samples are required")
    var y = _xl_vector(args[1], n, context)
    var ip: List[Int] = [_int(fields["max_iter"]), 1 if _truth(fields["fit_intercept"]) else 0]
    var fp: List[Float64] = [_float(fields["tol"]), _float(fields["alpha_1"]), _float(fields["alpha_2"]),
                             _float(fields["lambda_1"]), _float(fields["lambda_2"]), _float(fields["threshold_lambda"])]
    var out = _xl_run(entries, ALGO_ARD, x.addr, n * d, n, d, y.addr, y.rows, _xl_ints(ip), _xl_reals(fp),
                      2 * d + 4, 3 * d * d + 4 * d + n, 2 * d, context)
    _xl_set_linear(fields, out, d, context)
    py_setitem(fields, "alpha_", _xl_float(out, d + 1))
    py_setitem(fields, "lambda_", _xl_slice(out, d + 2, d, Python.tuple(d), context))
    py_setitem(fields, "n_iter_", PythonObject(_xl_count(out, 2 * d + 2)))
    py_setitem(fields, "n_features_in_", PythonObject(d))
    _ = x^
    _ = y^
    return model


def xl_fit_lars(model: PythonObject, args: PythonObject, lasso: Bool, context: PythonObject) raises -> PythonObject:
    """Lars.fit(X, y) and (`lasso`) LassoLars.fit(X, y): `_lars_fit`."""
    var fields = py_fields(model)
    var name = _name(model)
    var max_iter = fields["n_nonzero_coefs"]
    var alpha_min = Float64(0.0)
    if lasso:
        if not _ge(fields["alpha"], PythonObject(0)):
            raise Error("mojolearn LassoLars: alpha must be >= 0")
        max_iter = fields["max_iter"]
        alpha_min = _float(fields["alpha"])
    if _attr(model, "jitter", PythonObject(None)) is not None:
        raise Error("mojolearn " + name + ": jitter is not implemented")
    var entries = _xl_entries(fields, context)
    var x = _xl_matrix(args[0], context)
    var n = x.rows
    var d = x.cols
    var y = _xl_vector(args[1], n, context)
    var ip: List[Int] = [_int(max_iter), 1 if _truth(fields["fit_intercept"]) else 0, 1 if lasso else 0,
                         1 if _truth(_attr(model, "positive", PythonObject(False))) else 0]
    var fp: List[Float64] = [alpha_min]
    var out = _xl_run(entries, ALGO_LARS, x.addr, n * d, n, d, y.addr, y.rows, _xl_ints(ip), _xl_reals(fp),
                      2 * d + 4, 2 * d * d + 8 * d, 2 * d, context)
    _xl_set_linear(fields, out, d, context)
    py_setitem(fields, "n_iter_", PythonObject(_xl_count(out, d + 1)))
    py_setitem(fields, "alpha_", _xl_float(out, d + 2))
    var k = _xl_count(out, d + 3)
    var active = Python.list()
    for i in range(k):  # driver-loop: over the active features (model structure)
        _ = py_call(active.append, PythonObject(_xl_count(out, d + 4 + i)))
    py_setitem(fields, "active_", active)
    py_setitem(fields, "n_features_in_", PythonObject(d))
    _ = x^
    _ = y^
    return model


def xl_fit_quantile(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """QuantileRegressor.fit(X, y, sample_weight)."""
    var fields = py_fields(model)
    var q = fields["quantile"]
    if not (_lt(PythonObject(0), q) and _lt(q, PythonObject(1))):
        raise Error("mojolearn QuantileRegressor: quantile must be strictly between 0 and 1")
    if not _ge(fields["alpha"], PythonObject(0)):
        raise Error("mojolearn QuantileRegressor: alpha must be >= 0")
    var entries = _xl_entries(fields, context)
    var x = _xl_matrix(args[0], context)
    var n = x.rows
    var d = x.cols
    var y = _xl_vector(args[1], n, context)
    var m = d + 1
    var has_sw = 0
    var yy = _xl_with_weights(y, args[2], n, has_sw, context)
    var tol = fields["tol"]
    var ip: List[Int] = [_int(fields["max_iter"]), 1 if _truth(fields["fit_intercept"]) else 0, has_sw]
    var fp: List[Float64] = [_float(q), _float(fields["alpha"]), _float(py_call(_op().truediv, tol, PythonObject(100.0))),
                             _float(tol), 0.0 if labels_env("MOJOLEARN_XQ_ABS_BALANCE", "") == "1" else 1.0]
    var out = _xl_run(entries, ALGO_QUANTILE, x.addr, n * d, n, d, yy.addr, yy.rows, _xl_ints(ip), _xl_reals(fp),
                      d + 3, m * m + 3 * m + 4 * n + 3 * d, 1, context)
    _xl_set_linear(fields, out, d, context)
    py_setitem(fields, "n_iter_", PythonObject(_xl_count(out, d + 1)))
    py_setitem(fields, "n_features_in_", PythonObject(d))
    _ = x^
    _ = y^
    _ = yy^
    return model


# ----------------------------------------------------------- classifiers


def _xl_class_codes(model: PythonObject, source: PythonObject, n: Int, context: PythonObject) raises -> ClassicalLabels:
    """`_class_codes(est, y, n)`: the labels encoded once at entry."""
    var labels = classical_encode_labels(source, context)
    if labels.n != n:
        raise Error("mojolearn " + _name(model) + ": X and y lengths differ")
    if len(labels.classes) < 2:
        raise Error("mojolearn " + _name(model) + ": y has one class")
    return labels^


def xl_fit_sgd_classifier(model: PythonObject, args: PythonObject, kind: String, context: PythonObject) raises -> PythonObject:
    """SGDClassifier.fit(X, y, sample_weight) ("sgd"), Perceptron.fit(X, y,
    sample_weight) ("perceptron") and PassiveAggressiveClassifier.fit(X, y)
    ("pa"): one SGD fit over the class codes."""
    var fields = py_fields(model)
    var losses = context["xl_sgd_clf_loss"]
    if kind == "perceptron":
        _xl_sgd_refuse(model, fields["early_stopping"], PythonObject(False), fields["warm_start"])
    else:
        _xl_sgd_refuse(model, fields["early_stopping"], fields["average"], fields["warm_start"])
    if kind == "sgd" and not py_contains(losses, fields["loss"]):
        raise Error("mojolearn SGDClassifier: loss must be one of " + _sorted_repr(losses))
    if kind == "pa":
        if not py_contains(Python.tuple("hinge", "squared_hinge"), fields["loss"]):
            raise Error("mojolearn PassiveAggressiveClassifier: loss must be 'hinge' or 'squared_hinge'")
        if not _gt(fields["C"], PythonObject(0)):
            raise Error("mojolearn PassiveAggressiveClassifier: C must be > 0")
    var entries = _xl_entries(fields, context)
    var x = _xl_matrix(args[0], context)
    var labels = _xl_class_codes(model, args[1], x.rows, context)
    var codes = labels_vector(x.rows, "<f4", context)
    labels_cast(labels_address(labels.codes), "<i4", labels_address(codes), "<f4", x.rows, context)
    var k = len(labels.classes)
    py_setitem(fields, "classes_", labels.classes)
    var fit: Tuple[PythonObject, PythonObject]
    if kind == "sgd":
        var batch_sum = _xl_sgd_auto_sum(fields["batch_size"], fields["loss"], fields["learning_rate"], context)
        fit = _xl_sgd(model, fields, entries, x, codes, labels_address(codes), k, _int(losses[fields["loss"]]),
                      fields["penalty"], fields["learning_rate"], fields["alpha"], fields["l1_ratio"], fields["eta0"],
                      fields["power_t"], fields["epsilon"], fields["fit_intercept"], fields["max_iter"], fields["tol"],
                      fields["n_iter_no_change"], fields["shuffle"], fields["random_state"], args[2],
                      fields["class_weight"], labels.classes, labels.codes, fields["batch_size"], batch_sum, 4096,
                      context)
    elif kind == "perceptron":
        fit = _xl_sgd(model, fields, entries, x, codes, labels_address(codes), k, _int(losses["perceptron"]),
                      fields["penalty"], PythonObject("constant"), fields["alpha"], fields["l1_ratio"], fields["eta0"],
                      PythonObject(0.5), PythonObject(0.1), fields["fit_intercept"], fields["max_iter"], fields["tol"],
                      fields["n_iter_no_change"], fields["shuffle"], fields["random_state"], args[2],
                      fields["class_weight"], labels.classes, labels.codes, fields["batch_size"], True, 256, context)
    else:
        var lr = PythonObject("pa1") if _eq(fields["loss"], PythonObject("hinge")) else PythonObject("pa2")
        # scikit-learn's PassiveAggressive*.fit takes no sample_weight
        fit = _xl_sgd(model, fields, entries, x, codes, labels_address(codes), k, _int(losses["hinge"]),
                      PythonObject(None), lr, PythonObject(1.0), PythonObject(0.0), fields["C"], PythonObject(0.5),
                      PythonObject(0.1), fields["fit_intercept"], fields["max_iter"], fields["tol"],
                      fields["n_iter_no_change"], fields["shuffle"], fields["random_state"], PythonObject(None),
                      fields["class_weight"], labels.classes, labels.codes, fields["batch_size"], False, 256, context)
    py_setitem(fields, "coef_", fit[0])
    py_setitem(fields, "intercept_", fit[1])
    _ = codes^
    _ = x^
    return model


def xl_fit_ridge_classifier(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """RidgeClassifier.fit(X, y, sample_weight): `_ridge_run` over the int32
    class codes (the +-1 targets are built in the binding)."""
    var fields = py_fields(model)
    var name = _name(model)
    if _truth(_attr(model, "positive", PythonObject(False))):
        raise Error("mojolearn " + name + ": positive=True is not implemented")
    if not py_contains(Python.tuple("auto", "cholesky"), _attr(model, "solver", PythonObject("auto"))):
        raise Error("mojolearn " + name + ": solver must be 'auto' or 'cholesky'")
    var alpha = fields["alpha"]
    if not _ge(alpha, PythonObject(0)):
        raise Error("mojolearn RidgeClassifier: alpha must be >= 0")
    var entries = _xl_entries(fields, context)
    var x = _xl_matrix(args[0], context)
    var n = x.rows
    var d = x.cols
    var labels = classical_encode_labels(args[1], context)
    if labels.n != n:
        raise Error("mojolearn RidgeClassifier: X and y lengths differ")
    var k = len(labels.classes)
    if k < 2:
        raise Error("mojolearn RidgeClassifier: y has one class")
    var T = 1 if k == 2 else k
    var sample_weight = args[2]
    if fields["class_weight"] is not None:
        # theirs: sample_weight * compute_sample_weight(class_weight, y),
        # 'balanced' from the unweighted class counts
        var sw_keep = PythonObject(None)
        var sw_addr = 0
        if sample_weight is not None:
            var w = labels_check_sample_weight(sample_weight, n, context)
            sw_keep = w.keep
            sw_addr = w.addr
        var prep = _xl_class_prep(model, entries, fields["class_weight"], labels.classes, labels.codes, sw_addr,
                                  False, True, context)
        sample_weight = prep[1]
        _ = sw_keep^
    # `_ridge_run(..., codes_mode=True)`: the codes' 32-bit words, then the
    # checked weights
    var has_sw = 0
    var y = LabelsMatrix(labels.codes, labels_address(labels.codes), n, 1, True)
    var yy = _xl_with_weights(y, sample_weight, n, has_sw, context)
    var ip: List[Int] = [T, 1 if _truth(fields["fit_intercept"]) else 0, 1, has_sw, 1]
    var fp: List[Float64] = [_float(alpha)]
    var out = _xl_run(entries, ALGO_RIDGE, x.addr, n * d, n, d, yy.addr, yy.rows, _xl_ints(ip), _xl_reals(fp),
                      T * d + T + 2 + 1 + 1, 3 * d * d + 3 * d + T + d * T + 2 * n, 1, context)
    # status 2: X'X + alpha I does not factor even in float-float
    if _xl_word(out, T * d + T + 2 + 1) == 2.0:
        raise Error("mojolearn " + name + ": X'X + alpha I is singular even in float-float (alpha="
                    + String(py=py_call(_b().format, _xl_float(out, T * d + T), PythonObject("g")))
                    + "); rescale X or use a larger alpha")
    py_setitem(fields, "classes_", labels.classes)
    py_setitem(fields, "coef_", _xl_slice(out, 0, T * d, Python.tuple(T, d), context))
    py_setitem(fields, "intercept_", _xl_slice(out, T * d, T, Python.tuple(T), context))
    py_setitem(fields, "n_features_in_", PythonObject(d))
    _ = x^
    _ = yy^
    return model


# -------------------------------------------------------------- isotonic


def _xl_isotonic_fit(model: PythonObject, fields: PythonObject, entries: PythonObject, x: LabelsMatrix,
                     y_source: PythonObject, sample_weight: PythonObject, context: PythonObject) raises:
    """IsotonicRegression.fit over the column already read: the binding
    sorts the positive-weight rows by (x, y, row) and pools them."""
    var n = x.rows
    var y = _xl_vector(y_source, n, context)
    var increasing = fields["increasing"]
    var rising = False
    if _eq(increasing, PythonObject("auto")):
        # the sign of Spearman's rho, exact, in the binding
        var sign = _int(py_call_typed(entries["x_linear_spearman_sign"], PythonObject(x.addr), PythonObject(y.addr),
                                      PythonObject(n)))
        rising = sign >= 0
    else:
        rising = _truth(increasing)
    py_setitem(fields, "increasing_", PythonObject(rising))
    var has_w = 0
    var yy = _xl_with_weights(y, sample_weight, n, has_w, context)
    var y_min = fields["y_min"]
    var y_max = fields["y_max"]
    var ip: List[Int] = [1 if rising else 0, 0 if y_min is None else 1, 0 if y_max is None else 1, has_w]
    var fp: List[Float64] = [_xl_or(y_min, 0.0), _xl_or(y_max, 0.0)]
    var out = _xl_run(entries, ALGO_ISOTONIC, x.addr, n, n, 1, yy.addr, yy.rows, _xl_ints(ip), _xl_reals(fp),
                      3 + 2 * n, 6 * n, 3 * n, context)
    var k = _xl_count(out, 0)
    py_setitem(fields, "X_min_", _xl_float(out, 1))
    py_setitem(fields, "X_max_", _xl_float(out, 2))
    py_setitem(fields, "X_thresholds_", _xl_slice(out, 3, k, Python.tuple(k), context))
    py_setitem(fields, "y_thresholds_", _xl_slice(out, 3 + n, k, Python.tuple(k), context))
    py_setitem(fields, "n_features_in_", PythonObject(1))
    _ = y^
    _ = yy^


def _xl_isotonic_check(fields: PythonObject) raises:
    if not py_contains(Python.tuple("nan", "clip", "raise"), fields["out_of_bounds"]):
        raise Error("mojolearn IsotonicRegression: out_of_bounds must be 'nan', 'clip' or 'raise'")


def _xl_isotonic_transform(model: PythonObject, fields: PythonObject, entries: PythonObject, t: LabelsMatrix,
                           context: PythonObject) raises -> PythonObject:
    """IsotonicRegression.transform over the column already read: the
    thresholds as two byte copies, the interp1d-linear predict on the
    binding; out_of_bounds='raise' is the flag word after the predictions."""
    var n = t.rows
    var xt = _xl_words(fields["X_thresholds_"], "X_thresholds_", context)
    var yt = _xl_words(fields["y_thresholds_"], "y_thresholds_", context)
    var k = len(fields["X_thresholds_"])
    var thr = _xl_join(xt.addr, xt.rows * xt.cols, yt.addr, yt.rows * yt.cols, context)
    var oob_mode = fields["out_of_bounds"]
    var raise_oob = _eq(oob_mode, PythonObject("raise"))
    var oob = 2 if raise_oob else (1 if _eq(oob_mode, PythonObject("clip")) else 0)
    var ip: List[Int] = [k, oob]
    var fp: List[Float64] = [_float(fields["X_min_"]), _float(fields["X_max_"])]
    var out = _xl_run(entries, ALGO_ISOTONIC_PREDICT, t.addr, n, n, 1, labels_address(thr), Int(py=thr.size),
                      _xl_ints(ip), _xl_reals(fp), n + 1 if raise_oob else n, 1, 1, context)
    _ = xt^
    _ = yt^
    _ = thr^
    if raise_oob:
        if _xl_word(out, n) != 0:
            raise Error("A value in x_new is below/above the interpolation range.")
        return _xl_slice(out, 0, n, Python.tuple(n), context)
    return out


def xl_isotonic(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """IsotonicRegression.fit / fit_transform / transform / predict."""
    var fields = py_fields(model)
    if operation == "fit" or operation == "fit_transform":
        _xl_isotonic_check(fields)
        var entries = _xl_entries(fields, context)
        var x = _xl_column(args[0], "X", context)
        _xl_isotonic_fit(model, fields, entries, x, args[1], args[2], context)
        if operation == "fit":
            _ = x^
            return model
        # `self.fit(X, y, sample_weight).transform(X)`: the column this call
        # already converted
        var out = _xl_isotonic_transform(model, fields, entries, x, context)
        _ = x^
        return out
    if not py_contains(fields, "X_thresholds_"):
        raise Error("RuntimeError: mojolearn IsotonicRegression: call fit first")
    var entries = _xl_entries(fields, context)
    var t = _xl_column(args[0], "T", context)
    var out = _xl_isotonic_transform(model, fields, entries, t, context)
    _ = t^
    return out


# ------------------------------------------------- Lasso / ElasticNet (cd)


def _xl_host_only(model: PythonObject) raises -> Bool:
    """An explicit host inference class (`_HOST_INFERENCE_ONLY` in its class
    dictionaries): it predicts a saved model on the CPU and never fits."""
    var b = _b()
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(model))
    for k in range(len(mro)):  # driver-loop: over the model class's MRO
        var class_fields = py_call(b.type.__dict__["__dict__"].__get__, mro[k])
        if py_get(class_fields, "_HOST_INFERENCE_ONLY") is not None:
            return True
    return False


def _xl_cd_entries(context: PythonObject) raises -> PythonObject:
    var entries = context["cd_solver"]
    if py_type(entries) is _b().str:
        raise Error(String(py=entries))
    return entries


def _xl_cd_validate(fields: PythonObject, context: PythonObject) raises:
    """ElasticNet._validate_params: cuML's guards first, in their order and
    with their messages, then this implementation's refusals by name."""
    var alpha = fields["alpha"]
    var l1_ratio = fields["l1_ratio"]
    var solver = fields["solver"]
    var selection = fields["selection"]
    if _lt(alpha, PythonObject(0.0)):
        raise Error("Expected alpha >= 0, got " + _str(alpha))
    if not py_contains(context["cd_selection"], selection):
        raise Error("selection " + _repr(selection) + " is not supported")
    if _lt(l1_ratio, PythonObject(0.0)) or _gt(l1_ratio, PythonObject(1.0)):
        raise Error("Expected 0.0 <= l1_ratio <= 1.0, got " + _str(l1_ratio))
    if not py_contains(Python.tuple("auto", "cd", "qn"), solver):
        raise Error("solver=" + _repr(solver) + " is not supported")
    if _eq(solver, PythonObject("qn")):
        raise Error("NotImplementedError: " + _CD_QN)
    if _eq(selection, PythonObject("random")):
        raise Error("NotImplementedError: " + _CD_RANDOM)
    if _truth(fields["positive"]):
        raise Error("NotImplementedError: " + _CD_POSITIVE)
    if _truth(fields["warm_start"]):
        raise Error("NotImplementedError: " + _CD_WARM)
    if not (fields["precompute"] is PythonObject(False)):
        raise Error("NotImplementedError: " + _CD_PRECOMPUTE)
    if not _truth(fields["copy_X"]):
        raise Error("NotImplementedError: " + _CD_COPY_X)
    var rs = fields["random_state"]
    var b = _b()
    if rs is not None and (_isinstance(rs, b.bool) or not (_isinstance(rs, b.int)
                                                           or Bool(py=py_call(b.hasattr, rs, PythonObject("randint")))
                                                           or Bool(py=py_call(b.hasattr, rs, PythonObject("integers"))))):
        raise Error("random_state=" + _repr(rs) + ": None, an int or a numpy random state")


def xl_cd_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """ElasticNet.fit / Lasso.fit(X, y, sample_weight): `cd_fit` over the
    design as the Python handed it (a float32 C-order design as it is, the
    device transposing it; any other input as one F-order float32 copy)."""
    var fields = py_fields(model)
    var b = _b()
    if _xl_host_only(model):
        raise Error("ImportError: mojolearn: no CPU implementation of the " + _name(model)
                    + " fit on the host class; the host model predicts a saved model only")
    _xl_cd_validate(fields, context)
    var source = args[0]
    if Bool(py=py_call(b.hasattr, source, PythonObject("toarray"))) or Bool(py=py_call(b.hasattr, source, PythonObject("tocsr"))):
        raise Error("NotImplementedError: " + _CD_SPARSE)
    var view = labels_view(source, "X", context)
    var listed = py_type(source) is b.list or py_type(source) is b.tuple
    var row_major = (view.typestr == "<f4" and view.ndim() == 2 and view.size() > 0 and view.c_order()
                     and not view.both_orders())
    var x: LabelsMatrix
    var copied = listed
    if row_major:
        x = LabelsMatrix(view.keep, view.addr, view.shape[0], view.shape[1], True)
    else:
        labels_check_rank(view, 2, "X")
        var keep = view.keep
        x = labels_f32_view(view^, "F", context)
        copied = copied or not (x.keep is keep)
    py_setitem(fields, "input_copied_", PythonObject(copied))
    py_setitem(fields, "fortran_copied_", PythonObject(copied))
    var n_rows = x.rows
    var n_cols = x.cols
    if n_rows < 2:
        raise Error("Found array with " + String(n_rows) + " sample(s) (shape=(" + String(n_rows) + ", "
                    + String(n_cols) + ")) while a minimum of 2 is required.")
    if len(labels_shape_of(args[1], context)) != 1:
        raise Error("mojolearn ElasticNet currently requires one target")
    var target = labels_f32(args[1], 1, "C", "y", context)
    if target.rows != n_rows:
        raise Error("mojolearn ElasticNet X and y lengths differ")
    var sw_keep = PythonObject(None)
    var sw_addr = 0
    if args[2] is not None:
        var w = labels_check_sample_weight(args[2], n_rows, context)
        sw_keep = w.keep
        sw_addr = w.addr
    var entries = _xl_cd_entries(context)
    var coef = labels_empty(Python.tuple(n_cols), "<f4", context)
    py_setitem(fields, "coef_", coef)
    var info = labels_vector(1, "<f4", context)
    var params = Python.list(PythonObject(n_rows), PythonObject(n_cols),
                             PythonObject(1 if _truth(fields["fit_intercept"]) else 0), PythonObject(_int(fields["max_iter"])),
                             PythonObject(_float(fields["alpha"])), PythonObject(_float(fields["l1_ratio"])),
                             PythonObject(_float(fields["tol"])), context["cd_selection"][fields["selection"]],
                             PythonObject(0 if sw_addr == 0 else 1), PythonObject(1 if row_major else 0))
    if sw_addr != 0:
        _ = py_call(params.append, PythonObject(sw_addr))
    var n_iter = py_call_typed(entries["cd_fit"], PythonObject(x.addr), PythonObject(target.addr),
                               PythonObject(labels_address(coef)), PythonObject(labels_address(info)), params)
    py_setitem(fields, "intercept_", _xl_float(info, 0))
    py_setitem(fields, "n_iter_", PythonObject(_int(n_iter)))
    py_setitem(fields, "n_features_in_", PythonObject(n_cols))
    _ = x^
    _ = target^
    _ = sw_keep^
    _ = info^
    return model


def xl_cd_predict(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """ElasticNet.predict / Lasso.predict(X): `cd_predict` over the F-order
    float32 design."""
    var fields = py_fields(model)
    if not py_contains(fields, "coef_"):
        raise Error("mojolearn ElasticNet: call fit before predict")
    var x = labels_f32(source, 2, "F", "X", context)
    if x.cols != _int(fields["n_features_in_"]):
        raise Error("mojolearn ElasticNet feature count differs from fit")
    var entries = _xl_cd_entries(context)
    var coef = _xl_words(fields["coef_"], "coef_", context)
    var out = labels_vector(x.rows, "<f4", context)
    _ = py_call_typed(entries["cd_predict"], PythonObject(x.addr), PythonObject(coef.addr),
                      PythonObject(labels_address(out)),
                      Python.list(PythonObject(x.rows), PythonObject(x.cols), PythonObject(_float(fields["intercept_"]))))
    _ = x^
    _ = coef^
    return out


def xl_cd_score(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """ElasticNet.score / Lasso.score(X, y): `_r2_host(self.predict(X), y)`.
    The prediction is the R^2 program's input as it is. An explicit host
    class predicts through its own native host entry (`host_infer_predict`)."""
    var prediction = PythonObject(None)
    if _xl_host_only(model):
        var host = context["cd_host_predict"]
        var published = py_call_typed(host[0], model, PythonObject("host_infer_predict"), Python.tuple(args[0]), host[1])
        var cast = labels_f32(published, 1, "C", "predictions", context)
        prediction = _xl_join(cast.addr, cast.rows, 0, 0, context)
        _ = cast^
    else:
        prediction = xl_cd_predict(model, args[0], context)
    return classical_r2_score(prediction, args[1], context)


# ------------------------------------------------------------------ door


def x_linear_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var b = Python.import_module("builtins")
        if len(call) != 4:
            raise Error("TypeError: x_linear_native_call requires model, operation, arguments and connected context")
        if py_type(call[1]) is not b.str or py_type(call[2]) is not b.tuple or py_type(call[3]) is not b.dict:
            raise Error("TypeError: x_linear_native_call requires an operation name, an argument tuple and its context")
        var model = call[0]
        var operation = String(py=call[1])
        var args = call[2]
        var context = call[3]
        classical_connect_numpy(context)
        var result = PythonObject(None)
        if operation == "regressor_predict":
            result = xl_regressor_predict(model, args, context)
        elif operation == "decision_function":
            result = xl_classifier_decision(model, args[0], context)
        elif operation == "classifier_predict":
            result = xl_classifier_predict(model, args[0], context)
        elif operation == "classifier_score":
            result = xl_classifier_score(model, args, context)
        elif operation == "sgd_predict_proba":
            result = xl_predict_proba(model, args[0], False, context)
        elif operation == "lrcv_predict_proba":
            result = xl_predict_proba(model, args[0], True, context)
        elif operation == "fit_sgd_classifier":
            result = xl_fit_sgd_classifier(model, args, "sgd", context)
        elif operation == "fit_perceptron":
            result = xl_fit_sgd_classifier(model, args, "perceptron", context)
        elif operation == "fit_pa_classifier":
            result = xl_fit_sgd_classifier(model, args, "pa", context)
        elif operation == "fit_ridge_classifier":
            result = xl_fit_ridge_classifier(model, args, context)
        elif operation == "fit_sgd_regressor":
            result = xl_fit_sgd_regressor(model, args, False, context)
        elif operation == "fit_pa_regressor":
            result = xl_fit_sgd_regressor(model, args, True, context)
        elif operation == "fit_one_class":
            result = xl_fit_one_class(model, args, context)
        elif operation == "one_class_decision_function":
            result = xl_one_class(model, "decision_function", args[0], context)
        elif operation == "one_class_score_samples":
            result = xl_one_class(model, "score_samples", args[0], context)
        elif operation == "one_class_predict":
            result = xl_one_class(model, "predict", args[0], context)
        elif operation == "fit_glm":
            result = xl_fit_glm(model, args, context)
        elif operation == "fit_huber":
            result = xl_fit_huber(model, args, context)
        elif operation == "fit_bayes":
            result = xl_fit_bayes(model, args, context)
        elif operation == "fit_ard":
            result = xl_fit_ard(model, args, context)
        elif operation == "fit_lars":
            result = xl_fit_lars(model, args, False, context)
        elif operation == "fit_lasso_lars":
            result = xl_fit_lars(model, args, True, context)
        elif operation == "fit_quantile":
            result = xl_fit_quantile(model, args, context)
        elif operation == "isotonic_fit":
            result = xl_isotonic(model, "fit", args, context)
        elif operation == "isotonic_fit_transform":
            result = xl_isotonic(model, "fit_transform", args, context)
        elif operation == "isotonic_transform":
            result = xl_isotonic(model, "transform", args, context)
        elif operation == "cd_fit":
            result = xl_cd_fit(model, args, context)
        elif operation == "cd_predict":
            result = xl_cd_predict(model, args[0], context)
        elif operation == "cd_score":
            result = xl_cd_score(model, args, context)
        else:
            raise Error("unknown native x_linear operation " + operation)
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if String(error).startswith("ZeroDivisionError:"):
            return raise_python_exception(error_without_class(error, "ZeroDivisionError"), ExceptionType("PyExc_ZeroDivisionError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)

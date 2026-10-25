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
from std.math import isinf
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.py_attr import py_attr
from bindings.py_typed_call import py_call_typed
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_wrap_array
from bindings.classical_score import classical_r2_score
from bindings.classical_warnings import classical_warning
from bindings.prep_native_estimator import prep_native_context, prep_native_split
from bindings.msel_inputs import msel_admit
from bindings.classical_labels import (
    LabelsMatrix, ClassicalLabels, labels_f32, labels_view, labels_check_rank, labels_f32_view,
    labels_check_sample_weight, labels_vector, labels_empty, labels_full_f32, labels_address, labels_cast, labels_reshape,
    labels_shape_of, labels_env, classical_encode_labels, classical_decode_labels, classical_accuracy, labels_r2,
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


def _xl_word(out_buf: PythonObject, index: Int) raises -> Float32:
    return _F32P(unsafe_from_address=labels_address(out_buf))[index]


def _xl_float(out_buf: PythonObject, index: Int) raises -> PythonObject:
    """`float(vals[index])`: the float32 word as a Python float."""
    return PythonObject(Float64(_xl_word(out_buf, index)))


def _xl_count(out_buf: PythonObject, index: Int) raises -> Int:
    """`int(vals[index])`: the float32 word truncated toward zero."""
    return Int(_xl_word(out_buf, index))


def _xl_slice(out_buf: PythonObject, offset: Int, count: Int, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Array.from_list(vals[offset:offset + count], '<f4')` (shaped as the
    Python list was nested): one byte copy of the float32 words."""
    var array = labels_empty(shape, "<f4", context)
    if count > 0:
        memcpy(dest=_F32P(unsafe_from_address=labels_address(array)),
               src=_F32P(unsafe_from_address=labels_address(out_buf)) + offset, count=count)
        _ = array  # keep the Python object alive through the native call above
    return array


def _xl_join(first: Int, first_count: Int, second: Int, second_count: Int, context: PythonObject) raises -> PythonObject:
    """`_concat_f32(a, b)`: a's then b's 32-bit words in one float32 vector
    (two byte copies)."""
    var out = labels_vector(first_count + second_count, "<f4", context)
    var o = _F32P(unsafe_from_address=labels_address(out))
    _ = out  # keep the Python object alive through the native call above
    if first_count > 0:
        memcpy(dest=o, src=_F32P(unsafe_from_address=first), count=first_count)
        _ = out  # keep the Python object alive through the native call above
    if second_count > 0:
        memcpy(dest=o + first_count, src=_F32P(unsafe_from_address=second), count=second_count)
        _ = out  # keep the Python object alive through the native call above
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
        var _keep_ret_one = LabelsMatrix(one, labels_address(one), 1, 1, True)
        _ = one  # keep the Python object alive through the native call above
        return _keep_ret_one^
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
    _ = out  # keep the Python object alive through the native call above
    var cp = _F32P(unsafe_from_address=c.addr)
    _ = c  # keep the Python object alive through the native call above
    var bp = _F32P(unsafe_from_address=b.addr)
    _ = b  # keep the Python object alive through the native call above
    for j in range(k):  # driver-loop: over the fitted classes' coefficient rows (model structure)
        memcpy(dest=o + j * (d + 1), src=cp + j * d, count=d)
        _ = out  # keep the Python object alive through the native call above
        _ = c  # keep the Python object alive through the native call above
        memcpy(dest=o + j * (d + 1) + d, src=bp + j, count=1)
        _ = out  # keep the Python object alive through the native call above
        _ = b  # keep the Python object alive through the native call above
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
    _ = wb  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
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
    _ = wb  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
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
    _ = zero  # keep the Python object alive through the native call above
    _ = py_call_typed(entries["x_linear_fit"], PythonObject(algo), PythonObject(x_addr), PythonObject(ya),
                      Python.list(PythonObject(n), PythonObject(d), PythonObject(x_size), PythonObject(y_size),
                                  PythonObject(n_out), PythonObject(max(n_fw, 1)), PythonObject(max(n_iw, 1)),
                                  PythonObject(len(ip)), PythonObject(len(fp))),
                      ip, fp, PythonObject(labels_address(out)))
    _ = zero  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
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
    _ = w  # keep the Python object alive through the native call above
    _ = w^
    has_sw = 1
    var _keep_ret_joined = LabelsMatrix(joined, labels_address(joined), y.rows * y.cols + n, 1, True)
    _ = joined  # keep the Python object alive through the native call above
    return _keep_ret_joined^


def _xl_owned(m: LabelsMatrix, context: PythonObject) raises -> PythonObject:
    """A float32 vector Array of m's words: m's own Array when it is one,
    else one byte copy of the caller's buffer."""
    if py_type(m.keep) is context["array_class"] and labels_address(m.keep) == m.addr:
        return m.keep
    return _xl_join(m.addr, m.rows * m.cols, 0, 0, context)


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
    _ = cw  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
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
        _ = targets  # keep the Python object alive through the native call above
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
        # cwp borrows prep's class-weight Array. Retain its owner until the
        # binary or multiclass parameter reads finish; the pointer owns nothing.
        _ = prep^
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
    _ = yy  # keep the Python object alive through the native call above
    var n_out = problems * d + problems + 2
    if _xl_word(out, n_out - 1) != 0:
        raise Error("Floating-point under-/overflow occurred. Scaling input data with "
                    + "StandardScaler or MinMaxScaler might help.")
    _ = yy  # keep the Python object alive through the native calls above
    var coef = _xl_slice(out, 0, problems * d, Python.tuple(problems, d), context)
    _ = yy  # keep the Python object alive through the native call above
    var intercept = _xl_slice(out, problems * d, problems, Python.tuple(problems), context)
    _ = yy  # keep the Python object alive through the native call above
    var n_iter = _xl_count(out, n_out - 2)
    _ = yy  # keep the Python object alive through the native call above
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
        _ = y  # keep the Python object alive through the native call above
    else:
        fit = _xl_sgd(model, fields, entries, x, y.keep, y.addr, 0, _int(losses[loss]), fields["penalty"],
                      fields["learning_rate"], fields["alpha"], fields["l1_ratio"], fields["eta0"], fields["power_t"],
                      fields["epsilon"], fields["fit_intercept"], fields["max_iter"], fields["tol"],
                      fields["n_iter_no_change"], fields["shuffle"], fields["random_state"], args[2],
                      PythonObject(None), PythonObject(None), PythonObject(None), fields["batch_size"], False, 4096,
                      context)
        _ = y  # keep the Python object alive through the native call above
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
    _ = off  # keep the Python object alive through the native call above
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
        _ = wide  # keep the Python object alive through the native call above
        _ = codes  # keep the Python object alive through the native call above
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
        _ = out  # keep the Python object alive through the native call above
        _ = scores  # keep the Python object alive through the native call above
    _ = scores^
    _ = x^
    return out


def xl_classifier_score(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_LinearClassifierMixin.score(X, y)`: predict, then
    `accuracy_fraction(y, prediction)` in the same call."""
    var prediction = xl_classifier_predict(model, args[0], context)
    return PythonObject(classical_accuracy(args[1], prediction, PythonObject(None), PythonObject(None), context))


# ------------------------------------------------------------ regressors


def _xl_set_linear(fields: PythonObject, out_buf: PythonObject, d: Int, context: PythonObject) raises:
    """coef_ = vals[:d] (float32 Array), intercept_ = float(vals[d])."""
    py_setitem(fields, "coef_", _xl_slice(out_buf, 0, d, Python.tuple(d), context))
    py_setitem(fields, "intercept_", _xl_float(out_buf, d))


def _xl_tweedie_link(fields: PythonObject) raises -> Int:
    """TweedieRegressor._link_code."""
    var p = _float(fields["power"])
    if p < 0 or (p > 0 and p < 1):
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
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    if _xl_word(out, d + 2) < 0:
        var loss_name = _class_attr(model, "_loss_name")
        raise Error("Some value(s) of y are out of the valid range of the loss '"
                    + (_str(loss_name) if _truth(loss_name) else String("HalfTweedieLoss")) + "'.")
    _ = yy  # keep the Python object alive through the native calls above
    _ = x  # keep the Python object alive through the native calls above
    _xl_set_linear(fields, out, d, context)
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "n_iter_", PythonObject(_xl_count(out, d + 1)))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
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
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _xl_set_linear(fields, out, d, context)
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    var scale = _xl_float(out, d + 1)
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "scale_", scale)
    py_setitem(fields, "n_iter_", PythonObject(_xl_count(out, d + 2)))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
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
    _ = y  # keep the Python object alive through the native call above
    _ = flags  # keep the Python object alive through the native call above
    _ = pred  # keep the Python object alive through the native call above
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
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _xl_set_linear(fields, out, d, context)
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "alpha_", _xl_float(out, d + 1))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "lambda_", _xl_float(out, d + 2))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "n_iter_", PythonObject(_xl_count(out, d + 3)))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
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
    _ = y  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _xl_set_linear(fields, out, d, context)
    _ = y  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "alpha_", _xl_float(out, d + 1))
    _ = y  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "lambda_", _xl_slice(out, d + 2, d, Python.tuple(d), context))
    _ = y  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "n_iter_", PythonObject(_xl_count(out, 2 * d + 2)))
    _ = y  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "n_features_in_", PythonObject(d))
    _ = x^
    _ = y^
    return model


def xl_fit_lars(model: PythonObject, args: PythonObject, lasso: Bool, context: PythonObject) raises -> PythonObject:
    """Lars.fit(X, y) and (`lasso`) LassoLars.fit(X, y): `_lars_fit`."""
    var fields = py_fields(model)
    var name = _name(model)
    # LassoLars has max_iter and alpha; Lars has n_nonzero_coefs
    var max_iter: PythonObject
    var alpha_min = Float64(0.0)
    if lasso:
        if not _ge(fields["alpha"], PythonObject(0)):
            raise Error("mojolearn LassoLars: alpha must be >= 0")
        max_iter = fields["max_iter"]
        alpha_min = _float(fields["alpha"])
    else:
        max_iter = fields["n_nonzero_coefs"]
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
    _ = y  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _xl_set_linear(fields, out, d, context)
    _ = y  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "n_iter_", PythonObject(_xl_count(out, d + 1)))
    _ = y  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "alpha_", _xl_float(out, d + 2))
    _ = y  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    var k = _xl_count(out, d + 3)
    _ = y  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    var active = Python.list()
    for i in range(k):  # driver-loop: over the active features (model structure)
        _ = py_call(active.append, PythonObject(_xl_count(out, d + 4 + i)))
        _ = y  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
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
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _xl_set_linear(fields, out, d, context)
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "n_iter_", PythonObject(_xl_count(out, d + 1)))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
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
    _ = codes  # keep the Python object alive through the native call above
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
        _ = codes  # keep the Python object alive through the native call above
    elif kind == "perceptron":
        fit = _xl_sgd(model, fields, entries, x, codes, labels_address(codes), k, _int(losses["perceptron"]),
                      fields["penalty"], PythonObject("constant"), fields["alpha"], fields["l1_ratio"], fields["eta0"],
                      PythonObject(0.5), PythonObject(0.1), fields["fit_intercept"], fields["max_iter"], fields["tol"],
                      fields["n_iter_no_change"], fields["shuffle"], fields["random_state"], args[2],
                      fields["class_weight"], labels.classes, labels.codes, fields["batch_size"], True, 256, context)
        _ = codes  # keep the Python object alive through the native call above
    else:
        var lr = PythonObject("pa1") if _eq(fields["loss"], PythonObject("hinge")) else PythonObject("pa2")
        # scikit-learn's PassiveAggressive*.fit takes no sample_weight
        fit = _xl_sgd(model, fields, entries, x, codes, labels_address(codes), k, _int(losses["hinge"]),
                      PythonObject(None), lr, PythonObject(1.0), PythonObject(0.0), fields["C"], PythonObject(0.5),
                      PythonObject(0.1), fields["fit_intercept"], fields["max_iter"], fields["tol"],
                      fields["n_iter_no_change"], fields["shuffle"], fields["random_state"], PythonObject(None),
                      fields["class_weight"], labels.classes, labels.codes, fields["batch_size"], False, 256, context)
        _ = codes  # keep the Python object alive through the native call above
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
            _ = w  # keep the Python object alive through the native call above
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
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    # status 2: X'X + alpha I does not factor even in float-float
    if _xl_word(out, T * d + T + 2 + 1) == 2.0:
        raise Error("mojolearn " + name + ": X'X + alpha I is singular even in float-float (alpha="
                    + String(py=py_call(_b().format, _xl_float(out, T * d + T), PythonObject("g")))
                    + "); rescale X or use a larger alpha")
        _ = yy  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
    _ = yy  # keep the Python object alive through the native calls above
    _ = x  # keep the Python object alive through the native calls above
    py_setitem(fields, "classes_", labels.classes)
    py_setitem(fields, "coef_", _xl_slice(out, 0, T * d, Python.tuple(T, d), context))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "intercept_", _xl_slice(out, T * d, T, Python.tuple(T), context))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
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
        _ = y  # keep the Python object alive through the native call above
        rising = sign >= 0
        _ = y  # keep the Python object alive through the native call above
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
    _ = yy  # keep the Python object alive through the native call above
    var k = _xl_count(out, 0)
    _ = yy  # keep the Python object alive through the native call above
    py_setitem(fields, "X_min_", _xl_float(out, 1))
    _ = yy  # keep the Python object alive through the native call above
    py_setitem(fields, "X_max_", _xl_float(out, 2))
    _ = yy  # keep the Python object alive through the native call above
    py_setitem(fields, "X_thresholds_", _xl_slice(out, 3, k, Python.tuple(k), context))
    _ = yy  # keep the Python object alive through the native call above
    py_setitem(fields, "y_thresholds_", _xl_slice(out, 3 + n, k, Python.tuple(k), context))
    _ = yy  # keep the Python object alive through the native call above
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
    _ = yt  # keep the Python object alive through the native call above
    _ = xt  # keep the Python object alive through the native call above
    var oob_mode = fields["out_of_bounds"]
    var raise_oob = _eq(oob_mode, PythonObject("raise"))
    var oob = 2 if raise_oob else (1 if _eq(oob_mode, PythonObject("clip")) else 0)
    var ip: List[Int] = [k, oob]
    var fp: List[Float64] = [_float(fields["X_min_"]), _float(fields["X_max_"])]
    var out = _xl_run(entries, ALGO_ISOTONIC_PREDICT, t.addr, n, n, 1, labels_address(thr), Int(py=thr.size),
                      _xl_ints(ip), _xl_reals(fp), n + 1 if raise_oob else n, 1, 1, context)
    _ = thr  # keep the Python object alive through the native call above
    _ = yt  # keep the Python object alive through the native call above
    _ = xt  # keep the Python object alive through the native call above
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
        _ = view  # keep the Python object alive through the native call above
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
        _ = w  # keep the Python object alive through the native call above
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
    _ = target  # keep the Python object alive through the native call above
    _ = info  # keep the Python object alive through the native call above
    _ = coef  # keep the Python object alive through the native call above
    py_setitem(fields, "intercept_", _xl_float(info, 0))
    py_setitem(fields, "n_iter_", PythonObject(_int(n_iter)))
    _ = info  # keep the Python object alive through the native call above
    _ = coef  # keep the Python object alive through the native call above
    _ = target  # keep the Python object alive through the native call above
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
    _ = coef  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
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
        _ = cast  # keep the Python object alive through the native call above
        _ = cast^
    else:
        prediction = xl_cd_predict(model, args[0], context)
    return classical_r2_score(prediction, args[1], context)


# ------------------------------------------------------------ cv helpers
# RidgeCV, LassoCV, ElasticNetCV and LogisticRegressionCV. An int (or None)
# cv takes the same integer fold ids the Python assigned (the base binding's
# `fold_ids` / `select_fold_i64`, or the binding's own StratifiedKFold ids);
# a user splitter object is the user-API boundary: its `split(X, y)` is
# called once, as the Python did, and its (train, test) index arrays are
# read natively.

comptime ALGO_ENETCV = 9
comptime ALGO_LOGCV = 10
comptime ALGO_RIDGE_KFOLD = 13


def _xl_helper(context: PythonObject, name: String) raises -> PythonObject:
    """A base binding helper (`_buffer._native(name)`) connected at import,
    or the refusal recorded then."""
    var entry = py_get(context["xl_helpers"], PythonObject(name))
    if entry is None:
        raise Error("ImportError: mojolearn: the base binding has no `" + name + "`")
    if py_type(entry) is _b().str:
        raise Error(String(py=entry))
    return entry


def _xl_is_int_cv(cv: PythonObject, context: PythonObject) raises -> Bool:
    """`_is_int_cv`: an Integral that is not a bool."""
    return _isinstance(cv, context["labels_integral_type"]) and not _isinstance(cv, _b().bool)


def _xl_index_i64(values: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`as_index_i64(values, name=name)`: row indices as a 1-D int64 Array
    (a list, tuple or range through `array('q', ...)` in C; a buffer through
    the native cast)."""
    var b = _b()
    var cls = py_type(values)
    if cls is b.list or cls is b.tuple or cls is b.range:
        var store = py_call(context["array_type"], PythonObject("q"), values)
        return classical_wrap_array(store, Python.tuple(len(store)), "<i8", "C", context)
    var view = labels_view(values, name, context)
    var count = view.size()
    if view.typestr == "<i8" and (view.c_order() or view.both_orders()):
        if py_type(view.keep) is context["array_class"] and labels_address(view.keep) == view.addr:
            return labels_reshape(view.keep, Python.tuple(count), context)
        _ = view  # keep the Python object alive through the native calls above
        var copy = labels_vector(count, "<i8", context)
        if count > 0:
            memcpy(dest=MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=labels_address(copy)),
                   src=MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=view.addr), count=count)
            _ = copy  # keep the Python object alive through the native call above
            _ = view  # keep the Python object alive through the native call above
        return copy
    var out = labels_vector(count, "<i8", context)
    labels_cast(view.addr, view.typestr, labels_address(out), "<i8", count, context)
    _ = out  # keep the Python object alive through the native call above
    _ = view  # keep the Python object alive through the native call above
    return out


def _xl_cv_splits(cv: PythonObject, x_source: PythonObject, y_source: PythonObject,
                  context: PythonObject) raises -> PythonObject:
    """`_cv_splits(cv, X, y)`: a list of (train, test) int64 Arrays from a
    splitter's `split(X, y)` (called once) or an iterable of pairs."""
    var b = _b()
    var pairs = PythonObject(None)
    if py_type(py_type(cv)) is b.type and py_contains(prep_native_context(context)["family_msel_splitter_layouts"], py_type(cv)):
        pairs = prep_native_split(cv, x_source, y_source, context)
    elif py_type(cv) is b.list or py_type(cv) is b.tuple:
        msel_admit(cv, prep_native_context(context))
        pairs = cv
    else:
        raise Error("TypeError: native linear CV requires a native splitter or list of index pairs; Python callbacks cannot execute")
    var listed = py_call(b.list, pairs)
    var splits = Python.list()
    for k in range(len(listed)):  # driver-loop: over the cv splits
        var pair = py_call(b.tuple, listed[k])
        if len(pair) != 2:
            if len(pair) > 2:
                raise Error("too many values to unpack (expected 2, got " + String(len(pair)) + ")")
            raise Error("not enough values to unpack (expected 2, got " + String(len(pair)) + ")")
        _ = py_call(splits.append, Python.tuple(_xl_index_i64(pair[0], "train", context),
                                                _xl_index_i64(pair[1], "test", context)))
    if len(splits) == 0:
        raise Error("mojolearn: cv produced no (train, test) split")
    return splits


def _xl_default_folds(n: Int, k: Int, context: PythonObject) raises -> PythonObject:
    """`KFold(n_splits=k).split(...)` unshuffled: contiguous folds by the
    core helpers (`fold_ids` with no codes, `select_fold_i64`)."""
    var fold_store = labels_vector(n, "<i4", context)
    var fold_counts = labels_vector(k, "<i8", context)
    var class_counts = labels_vector(1, "<i8", context)
    _ = py_call_typed(_xl_helper(context, "fold_ids"), PythonObject(0), PythonObject(n), PythonObject(0), PythonObject(k),
                      PythonObject(labels_address(class_counts)), PythonObject(labels_address(fold_store)),
                      PythonObject(labels_address(fold_counts)))
    _ = class_counts  # keep the Python object alive through the native call above
    _ = fold_store  # keep the Python object alive through the native call above
    _ = fold_counts  # keep the Python object alive through the native call above
    var select = _xl_helper(context, "select_fold_i64")
    var counts = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=labels_address(fold_counts))
    _ = fold_counts  # keep the Python object alive through the native call above
    var out = Python.list()
    for fold in range(k):  # driver-loop: over the cv folds
        var size = Int(counts[fold])
        _ = fold_counts  # keep the Python object alive through the native call above
        var test = labels_vector(size, "<i8", context)
        var train = labels_vector(n - size, "<i8", context)
        var scratch = labels_vector(n, "<i8", context)
        var got = _int(py_call_typed(select, PythonObject(labels_address(fold_store)), PythonObject(n), PythonObject(fold),
                                     PythonObject(labels_address(test) if size > 0 else labels_address(scratch)),
                                     PythonObject(labels_address(train) if n - size > 0 else labels_address(scratch))))
        _ = fold_store  # keep the Python object alive through the native call above
        _ = test  # keep the Python object alive through the native call above
        _ = scratch  # keep the Python object alive through the native call above
        _ = train  # keep the Python object alive through the native call above
        if got != size:
            raise Error("RuntimeError: mojolearn: select_fold_i64 disagrees with fold_ids")
        _ = scratch  # keep the Python object alive through the native calls above
        _ = train  # keep the Python object alive through the native calls above
        _ = test  # keep the Python object alive through the native calls above
        _ = fold_store  # keep the Python object alive through the native calls above
        _ = py_call(out.append, Python.tuple(train, test))
    return out


def _xl_kfold_ids_f32(n: Int, k: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_kfold_ids_f32(n, k)`: KFold's contiguous fold ids as float32."""
    if not _is_int(k) or _lt(k, PythonObject(2)):
        raise Error("mojolearn: cv must be None or an int in [2, n_samples]")
    var folds = _int(k)
    if folds > n:
        raise Error("Cannot have number of splits n_splits=" + String(folds) + " greater than the "
                    + "number of samples: n_samples=" + String(n) + ".")
    var ids = labels_vector(n, "<i4", context)
    var counts = labels_vector(folds, "<i8", context)
    _ = py_call_typed(_xl_helper(context, "fold_ids"), PythonObject(0), PythonObject(n), PythonObject(0),
                      PythonObject(folds), PythonObject(0), PythonObject(labels_address(ids)),
                      PythonObject(labels_address(counts)))
    _ = ids  # keep the Python object alive through the native call above
    _ = counts  # keep the Python object alive through the native call above
    var out = labels_vector(n, "<f4", context)
    labels_cast(labels_address(ids), "<i4", labels_address(out), "<f4", n, context)
    _ = out  # keep the Python object alive through the native call above
    _ = ids  # keep the Python object alive through the native call above
    return out


def _xl_cv_fold_ids(splits: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_cv_fold_ids(splits, n)`: the int32 fold of every row when the test
    sides partition the rows and every train side is the complement of its
    test side; None otherwise."""
    var fold = _xl_full_i32(n, -1, context)
    var assign = _xl_helper(context, "assign_fold_i64")
    var covered = 0
    for k in range(len(splits)):  # driver-loop: over the cv splits
        var te = splits[k][1]
        var size = Int(py=te.size)
        if _int(py_call_typed(assign, PythonObject(labels_address(te) if size > 0 else 0), PythonObject(size), PythonObject(n),
                              PythonObject(k), PythonObject(labels_address(fold)))) != 0:
            return PythonObject(None)
        _ = te  # keep the Python object alive through the native calls above
        _ = fold  # keep the Python object alive through the native calls above
        covered += size
    if covered != n:
        return PythonObject(None)
    var check = _xl_helper(context, "check_indices_i64")
    var hits = _xl_helper(context, "count_fold_hits_i64")
    for k in range(len(splits)):  # driver-loop: over the cv splits
        var tr = splits[k][0]
        var te = splits[k][1]
        var tr_size = Int(py=tr.size)
        if tr_size != n - Int(py=te.size):
            return PythonObject(None)
        if tr_size > 0:
            if _int(py_call_typed(check, PythonObject(labels_address(tr)), PythonObject(tr_size), PythonObject(n))) != 0:
                return PythonObject(None)
            _ = tr  # keep the Python object alive through the native calls above
            if _int(py_call_typed(hits, PythonObject(labels_address(tr)), PythonObject(tr_size),
                                  PythonObject(labels_address(fold)), PythonObject(n), PythonObject(k))) != 0:
                return PythonObject(None)
            _ = tr  # keep the Python object alive through the native calls above
            _ = fold  # keep the Python object alive through the native calls above
    return fold


def _xl_full_i32(n: Int, value: Int, context: PythonObject) raises -> PythonObject:
    """`full((n,), value, '<i4')`: the store repeated in C."""
    var one = py_call(context["array_type"], PythonObject("i"), Python.list(PythonObject(value)))
    var store = py_call(py_attr(one, "__mul__"), PythonObject(n))
    return classical_wrap_array(store, Python.tuple(n), "<i4", "C", context)


def _xl_cv_fold_ids_f32(splits: PythonObject, n: Int, name: String,
                        context: PythonObject) raises -> Tuple[PythonObject, PythonObject]:
    """`_cv_fold_ids_f32(splits, n, name)`: (fold ids, None) for a
    partition, else (the -(n + 1) words, the per-split roles)."""
    var ids = _xl_cv_fold_ids(splits, n, context)
    if ids is not None:
        var out = labels_vector(n, "<f4", context)
        labels_cast(labels_address(ids), "<i4", labels_address(out), "<f4", n, context)
        _ = out  # keep the Python object alive through the native call above
        _ = ids  # keep the Python object alive through the native call above
        return (out, PythonObject(None))
    # `_cv_roles_f32`: per-split roles (x_linear/cv_roles.mojo)
    if n + 1 > 16777216:
        raise Error("NotImplementedError: mojolearn " + name + ": cv splits that are not a partition are limited "
                    + "to 16,777,215 rows")
    var mark = _xl_helper(context, "mark_split_roles_f32")
    var roles = labels_vector(len(splits) * n, "<f4", context)
    for k in range(len(splits)):  # driver-loop: over the cv splits
        for side in range(2):  # driver-loop: over the two sides of one split
            var rows = splits[k][side]
            var size = Int(py=rows.size)
            var status = _int(py_call_typed(mark, PythonObject(labels_address(rows) if size > 0 else 0), PythonObject(size),
                                            PythonObject(n), PythonObject(k), PythonObject(side + 1),
                                            PythonObject(labels_address(roles))))
            _ = rows  # keep the Python object alive through the native call above
            _ = roles  # keep the Python object alive through the native call above
            if status == 1:
                raise Error("mojolearn " + name + ": a cv split holds an index outside [0, n_samples)")
            _ = rows  # keep the Python object alive through the native calls above
            _ = roles  # keep the Python object alive through the native calls above
            if status == 2:
                raise Error("mojolearn " + name + ": a cv split repeats a row, or puts a row on both sides")
            _ = rows  # keep the Python object alive through the native calls above
            _ = roles  # keep the Python object alive through the native calls above
        if Int(py=splits[k][0].size) == 0 or Int(py=splits[k][1].size) == 0:
            raise Error("mojolearn " + name + ": every cv split needs training and test rows")
    return (labels_full_f32(n, -Float64(n + 1), context), roles)


def _xl_parts(addrs: List[Int], counts: List[Int], context: PythonObject) raises -> PythonObject:
    """`_concat_f32_parts(parts)`: the parts' 32-bit words in order, one
    float32 vector (one byte copy per part)."""
    var total = 0
    for i in range(len(counts)):  # driver-loop: over the target vector's parts
        total += counts[i]
    var out = labels_vector(total, "<f4", context)
    var o = _F32P(unsafe_from_address=labels_address(out))
    _ = out  # keep the Python object alive through the native call above
    var at = 0
    for i in range(len(counts)):  # driver-loop: over the target vector's parts
        if counts[i] > 0:
            memcpy(dest=o + at, src=_F32P(unsafe_from_address=addrs[i]), count=counts[i])
            _ = out  # keep the Python object alive through the native call above
        at += counts[i]
    return out


# ------------------------------------------------------------------ RidgeCV


def _xl_ridge(model: PythonObject, entries: PythonObject, x_addr: Int, n: Int, d: Int, y: LabelsMatrix,
              alphas: List[Float64], sample_weight: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_ridge_run(est, a, n, d, Y, 1, alphas, sample_weight)`: one target,
    the alphas' Gram solves; refused when X'X + alpha I does not factor even
    in float-float."""
    var A = len(alphas)
    var has_sw = 0
    var yy = _xl_with_weights(y, sample_weight, n, has_sw, context)
    var ip: List[Int] = [1, 1 if _truth(py_fields(model)["fit_intercept"]) else 0, A, has_sw]
    var out = _xl_run(entries, ALGO_RIDGE, x_addr, n * d, n, d, yy.addr, yy.rows, _xl_ints(ip), _xl_reals(alphas),
                      d + 1 + 2 + A + 1, 3 * d * d + 3 * d + 1 + d + 2 * n, 1, context)
    _ = yy  # keep the Python object alive through the native call above
    if _xl_word(out, d + 1 + 2 + A) == 2.0:
        raise Error("mojolearn " + _name(model) + ": X'X + alpha I is singular even in float-float (alpha="
                    + String(py=py_call(_b().format, _xl_float(out, d + 1), PythonObject("g")))
                    + "); rescale X or use a larger alpha")
        _ = yy  # keep the Python object alive through the native call above
    _ = yy  # keep the Python object alive through the native calls above
    _ = yy^
    return out


def _xl_gather(src: Int, rows: PythonObject, n: Int, width: Int, context: PythonObject) raises -> PythonObject:
    """`take(src, idx, width)`: rows idx of src (width float32 words a row)."""
    var count = Int(py=rows.size)
    var shape = Python.tuple(count, width) if width > 1 else Python.tuple(count)
    var out = labels_empty(shape, "<f4", context)
    _ = py_call_typed(_xl_helper(context, "gather_rows_bytes"), PythonObject(src), PythonObject(labels_address(out)),
                      PythonObject(labels_address(rows)), PythonObject(n), PythonObject(count), PythonObject(4 * width))
    _ = out  # keep the Python object alive through the native call above
    return out


def _xl_undefined_warning(message: String, context: PythonObject) raises:
    """`_expansion_metrics._undefined_warning(message)`: scikit-learn's
    UndefinedMetricWarning when sklearn.exceptions is loaded, else
    mojolearn's, at the public method's frame."""
    var category = context["xl_undefined_warning"]
    var sk = py_get(context["modules"], "sklearn.exceptions")
    if sk is not None:
        category = py_call(_b().getattr, sk, PythonObject("UndefinedMetricWarning"), category)
    var frame = py_call(Python.import_module("sys")._getframe, PythonObject(0))
    var module = py_get(frame.f_globals, "__name__", "")
    classical_warning(message, category, context, Python.tuple(frame.f_code.co_filename, frame.f_lineno, module),
                      "UndefinedMetricWarning")


def xl_fit_ridge_cv(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """RidgeCV.fit(X, y, sample_weight)."""
    var fields = py_fields(model)
    var b = _b()
    if fields["scoring"] is not None or _truth(fields["alpha_per_target"]):
        raise Error("mojolearn RidgeCV: only scoring=None, alpha_per_target=False are implemented")
    var given = fields["alphas"]
    var listed = given if Bool(py=py_call(b.hasattr, given, PythonObject("__len__"))) else Python.list(given)
    var floats = py_call(b.list, py_call(b.map, b.float, listed))
    var alphas = List[Float64]()
    for i in range(len(floats)):  # driver-loop: over the alpha grid
        alphas.append(Float64(py=floats[i]))
    var refuse = len(alphas) == 0
    for i in range(len(alphas)):  # driver-loop: over the alpha grid
        refuse = refuse or not (alphas[i] > 0)
    if refuse:
        raise Error("mojolearn RidgeCV: alphas must be positive")
    var entries = _xl_entries(fields, context)
    var x = _xl_matrix(args[0], context)
    var n = x.rows
    var d = x.cols
    var y = _xl_vector(args[1], n, context)
    var sample_weight = args[2]
    var cv = fields["cv"]
    var A = len(alphas)
    if cv is None:
        var out = _xl_ridge(model, entries, x.addr, n, d, y, alphas, sample_weight, context)
        _ = x  # keep the Python object alive through the native call above
        _xl_set_linear(fields, out, d, context)
        _ = x  # keep the Python object alive through the native call above
        py_setitem(fields, "alpha_", _xl_float(out, d + 1))
        _ = x  # keep the Python object alive through the native call above
        py_setitem(fields, "best_score_", _xl_float(out, d + 2))
        _ = x  # keep the Python object alive through the native call above
        if _truth(fields["store_cv_results"]):
            py_setitem(fields, "cv_results_", _xl_slice(out, d + 3, A, Python.tuple(A), context))
            _ = x  # keep the Python object alive through the native call above
        py_setitem(fields, "n_features_in_", PythonObject(d))
        _ = x^
        _ = y^
        return model
    var int_cv = _xl_is_int_cv(cv, context)
    if int_cv and sample_weight is None and _int(cv) >= 2 and n // _int(cv) >= 2:
        # `_fit_kfold`: the k-fold scores on the binding, the first best
        # finite score chosen in Mojo
        var folds = _int(cv)
        if not (2 <= folds and folds <= n):
            raise Error("mojolearn RidgeCV: cv must be in [2, n_samples]")
        if _truth(fields["store_cv_results"]):
            raise Error("cv!=None and store_cv_results=True are incompatible")
        var n_fw = 2 * d * d + 3 * d + 1 + A * (d + 2) + A * (n // folds + 1)
        var ip: List[Int] = [folds, 1 if _truth(fields["fit_intercept"]) else 0, A]
        var scores = _xl_run(entries, ALGO_RIDGE_KFOLD, x.addr, n * d, n, d, y.addr, y.rows, _xl_ints(ip),
                             _xl_reals(alphas), A, n_fw, 1, context)
        var best = _int(py_call_typed(entries["x_linear_best_alpha"], PythonObject(labels_address(scores)), PythonObject(A)))
        _ = scores  # keep the Python object alive through the native call above
        if best < 0:
            raise Error("mojolearn RidgeCV: X'X + alpha I is singular in float32 for every alpha "
                        + "(rescale X, or use larger alphas)")
        _ = scores  # keep the Python object alive through the native calls above
        var one: List[Float64] = [alphas[best]]
        _ = scores  # keep the Python object alive through the native call above
        var out = _xl_ridge(model, entries, x.addr, n, d, y, one, PythonObject(None), context)
        _xl_set_linear(fields, out, d, context)
        py_setitem(fields, "alpha_", PythonObject(alphas[best]))
        _ = scores  # keep the Python object alive through the native call above
        py_setitem(fields, "best_score_", _xl_float(scores, best))
        py_setitem(fields, "n_features_in_", PythonObject(d))
        _ = x^
        _ = y^
        return model
    # `_fit_splits`: a native fit, predict and r2_score per split and alpha
    if _truth(fields["store_cv_results"]):
        raise Error("cv!=None and store_cv_results=True are incompatible")
    var splits = PythonObject(None)
    var yv = _xl_owned(y, context)
    if int_cv:
        var folds = _int(cv)
        if not (2 <= folds and folds <= n):
            raise Error("mojolearn RidgeCV: cv must be in [2, n_samples]")
        splits = _xl_default_folds(n, folds, context)
    else:
        splits = _xl_cv_splits(cv, args[0], args[1], context)
    var w_keep = PythonObject(None)
    var w_addr = 0
    if sample_weight is not None:
        var w = labels_check_sample_weight(sample_weight, n, context)
        w_keep = _xl_owned(w, context)
        w_addr = labels_address(w_keep)
        _ = w_keep  # keep the Python object alive through the native call above
    py_setitem(fields, "n_features_in_", PythonObject(d))
    var rows = Python.list()
    for i in range(A):  # driver-loop: over the alpha grid
        _ = py_call(rows.append, Python.list())
    var undefined = False
    for s in range(len(splits)):  # driver-loop: over the cv splits
        var tr = splits[s][0]
        var te = splits[s][1]
        var n_tr = Int(py=tr.size)
        var n_te = Int(py=te.size)
        if n_tr == 0 or n_te == 0:
            raise Error("mojolearn RidgeCV: every cv split needs training and test rows")
        if n_te < 2:
            # scikit-learn's r2_score: NaN, with its warning, for fewer than
            # two test rows; every alpha's mean is then NaN (the first wins)
            _xl_undefined_warning("R^2 score is not well-defined with less than two samples.", context)
            undefined = True
            continue
        var xtr = _xl_gather(x.addr, tr, n, d, context)
        var ytr = _xl_gather(labels_address(yv), tr, n, 1, context)
        _ = yv  # keep the Python object alive through the native call above
        var xte = _xl_gather(x.addr, te, n, d, context)
        var yte = _xl_gather(labels_address(yv), te, n, 1, context)
        _ = yv  # keep the Python object alive through the native call above
        var wtr = PythonObject(None)
        var wte = PythonObject(None)
        if w_addr != 0:
            wtr = _xl_gather(w_addr, tr, n, 1, context)
            wte = _xl_gather(w_addr, te, n, 1, context)
        var ytr_m = LabelsMatrix(ytr, labels_address(ytr), n_tr, 1, True)
        _ = ytr  # keep the Python object alive through the native call above
        _ = yv  # keep the Python object alive through the native call above
        var xte_m = LabelsMatrix(xte, labels_address(xte), n_te, d, True)
        _ = xte  # keep the Python object alive through the native call above
        for i in range(A):  # driver-loop: over the alpha grid
            var one: List[Float64] = [alphas[i]]
            var out = _xl_ridge(model, entries, labels_address(xtr), n_tr, d, ytr_m, one, wtr, context)
            _ = ytr  # keep the Python object alive through the native call above
            _ = xtr  # keep the Python object alive through the native call above
            _ = yv  # keep the Python object alive through the native call above
            var coef = _xl_slice(out, 0, d, Python.tuple(d), context)
            _ = xtr  # keep the Python object alive through the native call above
            var pred = _xl_decision(model, xte_m, coef, _xl_float(out, d), LINK_IDENTITY, True, entries, context)
            _ = xte  # keep the Python object alive through the native call above
            _ = xtr  # keep the Python object alive through the native call above
            var score = labels_r2(yte, pred, wte, PythonObject(None), context)
            _ = yv  # keep the Python object alive through the native call above
            _ = py_call(rows[i].append, PythonObject(score))
        _ = xtr^
        _ = xte^
    var best = 0
    var best_score = py_call(b.float, PythonObject("nan"))
    if not undefined:
        # `_row_means(rows)`: fsum / len per alpha (`row_means_f64`), the
        # first best by `reduce_stat`
        var k = len(rows[0])
        var flat = py_call(context["array_type"], PythonObject("d"))
        for i in range(A):  # driver-loop: over the alpha grid
            _ = py_call(flat.extend, rows[i])
        var means = labels_vector(max(A, 1), "<f8", context)
        if A > 0 and k > 0:
            if _int(py_call_typed(_xl_helper(context, "row_means_f64"), py_call(flat.buffer_info)[0], PythonObject(A),
                                  PythonObject(k), PythonObject(labels_address(means)))) != 0:
                raise Error("-inf + inf in fsum")
            _ = means  # keep the Python object alive through the native calls above
        best = _int(py_call_typed(_xl_helper(context, "reduce_stat"), PythonObject(labels_address(means)), PythonObject(1),
                                  PythonObject(A), PythonObject(3)))
        _ = means  # keep the Python object alive through the native call above
        best_score = PythonObject(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=labels_address(means))[best])
        _ = means  # keep the Python object alive through the native call above
        _ = flat^
        _ = means^
    var one: List[Float64] = [alphas[best]]
    var w_final = PythonObject(None) if w_addr == 0 else w_keep
    var out = _xl_ridge(model, entries, x.addr, n, d, y, one, w_final, context)
    _xl_set_linear(fields, out, d, context)
    py_setitem(fields, "alpha_", PythonObject(alphas[best]))
    py_setitem(fields, "best_score_", best_score)
    _ = w_keep^
    _ = yv^
    _ = x^
    _ = y^
    return model


# ------------------------------------------------------ LassoCV / ElasticNetCV


def _xl_enetcv(model: PythonObject, fields: PythonObject, args: PythonObject, l1_ratios: PythonObject,
               context: PythonObject) raises -> PythonObject:
    """`_enetcv_fit(est, X, y, l1_ratios, sample_weight)`: the alpha grid,
    the folds and the path on the binding; returns the l1_ratios entry
    nearest the best one."""
    var name = _name(model)
    var b = _b()
    if not _eq(fields["selection"], PythonObject("cyclic")):
        raise Error("mojolearn " + name + ": selection='random' is not implemented")
    var L = len(l1_ratios)
    for i in range(L):  # driver-loop: over the l1_ratio grid
        var r = Float64(py=l1_ratios[i])
        if not (r > 0 and r <= 1):
            raise Error("mojolearn " + name + ": l1_ratio must be in (0, 1]")
    var entries = _xl_entries(fields, context)
    var x = _xl_matrix(args[0], context)
    var n = x.rows
    var d = x.cols
    var y = _xl_vector(args[1], n, context)
    var cv = fields["cv"]
    var ids = PythonObject(None)
    var roles = PythonObject(None)
    var folds = 0
    if cv is None or _xl_is_int_cv(cv, context):
        var k = PythonObject(5) if cv is None else py_call(b.int, cv)
        ids = _xl_kfold_ids_f32(n, k, context)
        folds = _int(k)
    else:
        var splits = _xl_cv_splits(cv, args[0], args[1], context)
        folds = len(splits)
        var got = _xl_cv_fold_ids_f32(splits, n, name, context)
        ids = got[0]
        roles = got[1]
    var alphas = fields["alphas"]
    var n_alphas = _attr(model, "n_alphas", PythonObject(None))
    var explicit = False
    var grid = 0
    var values = Python.list()
    if _is_int(alphas):
        grid = _int(alphas)
    elif alphas is None or py_contains(Python.tuple("warn", "deprecated"), alphas):
        grid = _int(n_alphas) if _is_int_plain(n_alphas) else 100
    else:
        explicit = True
        values = py_call(b.sorted, py_call(b.map, b.float, alphas), reverse=PythonObject(True))
        grid = len(values)
    if grid < 1:
        raise Error("mojolearn " + name + ": at least one alpha is required")
    var sample_weight = args[2]
    if _isinstance(sample_weight, context["xl_number_type"]):
        sample_weight = PythonObject(None)
    var addrs = List[Int]()
    var counts = List[Int]()
    addrs.append(y.addr)
    _ = y  # keep the Python object alive through the native call above
    counts.append(n)
    addrs.append(labels_address(ids))
    _ = ids  # keep the Python object alive through the native call above
    counts.append(n)
    var weighted = 0
    var w_keep = PythonObject(None)
    if roles is not None:
        # per-split roles ride the weighted arm, after the weights
        if sample_weight is None:
            w_keep = labels_full_f32(n, 1.0, context)
        else:
            w_keep = _xl_owned(labels_check_sample_weight(sample_weight, n, context), context)
        addrs.append(labels_address(w_keep))
        _ = w_keep  # keep the Python object alive through the native call above
        counts.append(n)
        addrs.append(labels_address(roles))
        _ = roles  # keep the Python object alive through the native call above
        counts.append(Int(py=roles.size))
        weighted = 1
    elif sample_weight is not None:
        w_keep = _xl_owned(labels_check_sample_weight(sample_weight, n, context), context)
        addrs.append(labels_address(w_keep))
        _ = w_keep  # keep the Python object alive through the native call above
        counts.append(n)
        weighted = 1
    var yy = _xl_parts(addrs, counts, context)
    var fp = List[Float64]()
    fp.append(_float(fields["eps"]))
    fp.append(_float(fields["tol"]))
    for i in range(L):  # driver-loop: over the l1_ratio grid
        fp.append(Float64(py=l1_ratios[i]))
    if explicit:
        for i in range(grid):  # driver-loop: over the user alpha grid
            fp.append(Float64(py=values[i]))
    var ip: List[Int] = [_int(fields["max_iter"]), 1 if _truth(fields["fit_intercept"]) else 0, grid, folds, L,
                         1 if explicit else 0, 1 if _truth(fields["positive"]) else 0, weighted]
    var out = _xl_run(entries, ALGO_ENETCV, x.addr, n * d, n, d, labels_address(yy), Int(py=yy.size), _xl_ints(ip),
                      _xl_reals(fp), d + 4 + L * grid + L * grid * folds, d * d + 4 * d + 3 + grid * (d + 2), 1, context)
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _xl_set_linear(fields, out, d, context)
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "alpha_", _xl_float(out, d + 1))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    var l1_best = Float64(_xl_word(out, d + 2))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "n_iter_", PythonObject(_xl_count(out, d + 3)))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    var off = d + 4
    if L == 1:
        py_setitem(fields, "alphas_", _xl_slice(out, off, grid, Python.tuple(grid), context))
        _ = yy  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
        py_setitem(fields, "mse_path_", _xl_slice(out, off + grid, grid * folds, Python.tuple(grid, folds), context))
        _ = yy  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
    else:
        py_setitem(fields, "alphas_", _xl_slice(out, off, L * grid, Python.tuple(L, grid), context))
        _ = yy  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
        py_setitem(fields, "mse_path_", _xl_slice(out, off + L * grid, L * grid * folds, Python.tuple(L, grid, folds), context))
        _ = yy  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "n_features_in_", PythonObject(d))
    # the user's own value (the kernel carried it as float32): the first
    # entry nearest the returned one
    var pick = 0
    var gap = abs(Float64(py=l1_ratios[0]) - l1_best)
    for i in range(1, L):  # driver-loop: over the l1_ratio grid
        var g = abs(Float64(py=l1_ratios[i]) - l1_best)
        if g < gap:
            gap = g
            pick = i
    _ = yy^
    _ = w_keep^
    _ = ids^
    _ = roles^
    _ = x^
    _ = y^
    return l1_ratios[pick]


def _is_int_plain(value: PythonObject) raises -> Bool:
    """`isinstance(value, int)` (bools included, as the Python test was)."""
    return _isinstance(value, _b().int)


def xl_fit_lasso_cv(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """LassoCV.fit(X, y, sample_weight): the l1_ratio grid [1.0]."""
    var fields = py_fields(model)
    _ = _xl_enetcv(model, fields, args, Python.list(PythonObject(1.0)), context)
    return model


def xl_fit_elastic_net_cv(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """ElasticNetCV.fit(X, y, sample_weight): l1_ratio_ is the grid entry
    nearest the best one."""
    var fields = py_fields(model)
    var b = _b()
    var given = fields["l1_ratio"]
    var ratios = py_call(b.list, given) if Bool(py=py_call(b.hasattr, given, PythonObject("__len__"))) else Python.list(given)
    var floats = py_call(b.list, py_call(b.map, b.float, ratios))
    py_setitem(fields, "l1_ratio_", _xl_enetcv(model, fields, args, floats, context))
    return model


# ---------------------------------------------------- LogisticRegressionCV


def _xl_cs_grid(cs: PythonObject, context: PythonObject) raises -> List[Float64]:
    """The Cs grid: `np.logspace(-4, 4, m)` through `_pm.powr` (10 ** y by
    the decimal module at 50 digits, one correctly rounded conversion), or
    the user's values."""
    var grid = List[Float64]()
    if _is_int(cs):
        var m = _int(cs)
        if m <= 1:
            grid.append(1e-4)
            return grid^
        var ten = py_call(context["xl_decimal"], PythonObject(10.0))
        var ctx = context["xl_pow_context"]
        for i in range(m):  # driver-loop: over the Cs grid
            var exponent = -4.0 + Float64(8 * i) / Float64(m - 1)
            var value = Float64(py=py_call(_b().float, py_call(ctx.power, ten, py_call(context["xl_decimal"], PythonObject(exponent)))))
            if isinf(value):
                raise Error("OverflowError: math range error")
            grid.append(value)
        return grid^
    var floats = py_call(_b().list, py_call(_b().map, _b().float, cs))
    for i in range(len(floats)):  # driver-loop: over the user Cs grid
        grid.append(Float64(py=floats[i]))
    return grid^


def xl_fit_logistic_cv(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """LogisticRegressionCV.fit(X, y, sample_weight)."""
    var fields = py_fields(model)
    var b = _b()
    if not _eq(fields["penalty"], PythonObject("l2")) or _truth(fields["dual"]) or fields["l1_ratios"] is not None:
        raise Error("mojolearn LogisticRegressionCV: only penalty='l2' (primal) is implemented")
    if fields["scoring"] is not None or not _truth(fields["refit"]):
        raise Error("mojolearn LogisticRegressionCV: scoring and refit=False are not implemented")
    if not py_contains(Python.tuple("lbfgs", "newton-cg", "newton-cholesky"), fields["solver"]):
        raise Error("mojolearn LogisticRegressionCV: solver must be lbfgs (newton-* run L-BFGS too)")
    var entries = _xl_entries(fields, context)
    var x = _xl_matrix(args[0], context)
    var n = x.rows
    var d = x.cols
    var labels = _xl_class_codes(model, args[1], n, context)
    var K = len(labels.classes)
    var kp = 1 if K == 2 else K
    var Cs = _xl_cs_grid(fields["Cs"], context)
    var cv = fields["cv"]
    var explicit = not (cv is None or _xl_is_int_cv(cv, context))
    var roles = PythonObject(None)
    var fold_ids = PythonObject(None)
    var folds = 0
    if explicit:
        var splits = _xl_cv_splits(cv, args[0], args[1], context)
        folds = len(splits)
        var got = _xl_cv_fold_ids_f32(splits, n, "LogisticRegressionCV", context)
        fold_ids = got[0]
        roles = got[1]
    else:
        folds = 5 if cv is None else _int(cv)
    var sample_weight = args[2]
    var raw = PythonObject(None)
    if sample_weight is not None:
        raw = _xl_owned(labels_check_sample_weight(sample_weight, n, context), context)
    var class_weight = fields["class_weight"]
    var weigh = sample_weight is not None or class_weight is not None or roles is not None
    var prep = _xl_class_prep(model, entries, class_weight, labels.classes, labels.codes,
                              0 if raw is None else labels_address(raw), True, class_weight is not None, context)
    _ = raw  # keep the Python object alive through the native call above
    var fitw = prep[1]
    _ = raw  # keep the Python object alive through the native call above
    var largest = prep[2]
    _ = raw  # keep the Python object alive through the native call above
    if not explicit:
        # `_check_stratified_folds(largest, folds)`
        if folds < 2 or folds > largest:
            raise Error("mojolearn: cv must be None or an int in [2, the largest class size]")
    var codes = labels_vector(n, "<f4", context)
    labels_cast(labels_address(labels.codes), "<i4", labels_address(codes), "<f4", n, context)
    _ = codes  # keep the Python object alive through the native call above
    var zeros = labels_vector(n, "<f4", context)
    var addrs = List[Int]()
    var counts = List[Int]()
    addrs.append(labels_address(codes))
    _ = codes  # keep the Python object alive through the native call above
    counts.append(n)
    addrs.append(labels_address(fold_ids) if explicit else labels_address(zeros))
    _ = zeros  # keep the Python object alive through the native call above
    _ = fold_ids  # keep the Python object alive through the native call above
    counts.append(n)
    var has_sw = 0
    if weigh:
        if raw is None:
            raw = labels_full_f32(n, 1.0, context)
        addrs.append(labels_address(raw) if fitw is None else labels_address(fitw))
        _ = fitw  # keep the Python object alive through the native call above
        _ = raw  # keep the Python object alive through the native call above
        counts.append(n)
        addrs.append(labels_address(raw))
        _ = raw  # keep the Python object alive through the native call above
        counts.append(n)
        has_sw = 1
    if roles is not None:
        addrs.append(labels_address(roles))
        _ = roles  # keep the Python object alive through the native call above
        counts.append(Int(py=roles.size))
    var yy = _xl_parts(addrs, counts, context)
    var p = kp * (d + 1)
    var nc = len(Cs)
    var ip: List[Int] = [_int(fields["max_iter"]), 1 if _truth(fields["fit_intercept"]) else 0, kp, nc, folds, has_sw]
    if explicit:
        ip.append(1)
    var fp = List[Float64]()
    fp.append(_float(fields["tol"]))
    for i in range(nc):  # driver-loop: over the Cs grid
        fp.append(Cs[i])
    var out = _xl_run(entries, ALGO_LOGCV, x.addr, n * d, n, d, labels_address(yy), Int(py=yy.size), _xl_ints(ip),
                      _xl_reals(fp), kp * d + kp + 2 + folds * nc, p + 1 + n * (kp + 1) + _lbfgs_work(p), 5, context)
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "classes_", labels.classes)
    py_setitem(fields, "coef_", _xl_slice(out, 0, kp * d, Python.tuple(kp, d), context))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "intercept_", _xl_slice(out, kp * d, kp, Python.tuple(kp), context))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    # the grid entry nearest the returned best C (the first on a tie)
    var best_word = Float64(_xl_word(out, kp * d + kp))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    var pick = 0
    var gap = abs(Cs[0] - best_word)
    for i in range(1, nc):  # driver-loop: over the Cs grid
        var g = abs(Cs[i] - best_word)
        if g < gap:
            gap = g
            pick = i
    var cs_list = _xl_reals(Cs)
    py_setitem(fields, "Cs_", cs_list)
    var chosen = Python.list()
    for c in range(kp):  # driver-loop: over the fitted problems (model structure)
        _ = py_call(chosen.append, PythonObject(Cs[pick]))
    py_setitem(fields, "C_", chosen)
    py_setitem(fields, "n_iter_", PythonObject(_xl_count(out, kp * d + kp + 1)))
    _ = yy  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    var off = kp * d + kp + 2
    var names = py_call(b.list, labels.classes)
    var scores = Python.dict()
    var first = 1 if kp == 1 else 0
    for c in range(first, len(names)):  # driver-loop: over the class labels (one score table each)
        py_setitem(scores, names[c], _xl_slice(out, off, folds * nc, Python.tuple(folds, nc), context))
        _ = yy  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, "scores_", scores)
    py_setitem(fields, "n_features_in_", PythonObject(d))
    _ = yy^
    _ = codes^
    _ = zeros^
    _ = raw^
    _ = fitw^
    _ = fold_ids^
    _ = roles^
    _ = x^
    return model


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
        elif operation == "isotonic_score":
            # RegressorScoreMixin.score: metrics.r2_score(y, predict(X),
            # sample_weight, numeric_mode=self.numeric_mode) on the float32
            # prediction, which is handed straight to the R^2 kernel.
            var prediction = xl_isotonic(model, "transform", Python.tuple(args[0]), context)
            result = PythonObject(labels_r2(args[1], prediction, args[2],
                                            py_get(py_fields(model), "numeric_mode"), context))
        elif operation == "fit_ridge_cv":
            result = xl_fit_ridge_cv(model, args, context)
        elif operation == "fit_lasso_cv":
            result = xl_fit_lasso_cv(model, args, context)
        elif operation == "fit_elastic_net_cv":
            result = xl_fit_elastic_net_cv(model, args, context)
        elif operation == "fit_logistic_cv":
            result = xl_fit_logistic_cv(model, args, context)
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

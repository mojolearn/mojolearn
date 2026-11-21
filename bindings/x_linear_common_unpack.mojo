# SPDX-License-Identifier: Apache-2.0
# The helpers every operation of the linear expansion door shares
# (bindings/x_linear_native_door.mojo and the per-group
# bindings/x_linear_<group>_unpack.mojo / _pack.mojo files): Python scalar
# reads, the admitted X / y / fitted-array views, the float32 result words
# and slices, the typed fit and decision builds with their released RUN, and
# the class weights' UNPACK. They live here, not in the door, so that a group's boundary file
# imports them without importing the door (which imports the groups).
# Boundary code: Python objects are read and built here; the typed calls are
# x_linear/door_run.mojo's.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.linear_entry_table_ext import XL_OP_DECISION_CODES, XL_OP_DECISION_ROWS
from x_linear.door_run import (
    XlDecisionRequest, xl_decision_run, XlFitRequest, xl_fit_run, xl_fit_fold_held, xl_run_status_refuse,
    xl_run_word, xl_run_copy, xl_run_join, xl_block_dims, xl_run_block,
)
from bindings.x_linear_route_unpack import XLinearOps
from bindings.py_balanced import py_call, py_type, py_contains
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_lists import py_list_f64
from bindings.classical_arrays_door import classical_wrap_array
from bindings.device_rows_door import door_is_device_rows, door_rows_refuse
from bindings.admitted_input_unpack import (
    AdmittedX, admit_x_unrecorded, admit_f32, admit_weight_words, admitted_words, admitted_array, admitted_vector,
    admitted_storage_native, classical_view,
)
from bindings.admitted_input_run import AdmittedXView
from bindings.x_linear_keys_unpack import x_linear_keys
from bindings.py_balanced import py_item
from bindings.classical_labels_door import labels_full_f32, labels_address, labels_type_name

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


# ------------------------------------------------------------ Python values


def _b() raises -> PythonObject:
    # The import-time admitted connection (one stamp compare after the first
    # use), not PyImport_ImportModule's __import__ round trip on every scalar
    # helper call.
    return native_loaded_module("builtins")


def _op() raises -> PythonObject:
    return native_loaded_module("operator")


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
    return labels_type_name(py_type(model))


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



def _xl_scope(entries: XLinearOps, context: PythonObject) raises -> PythonObject:
    """The context a score program of this estimator runs in: the door's own
    context, or the CPU scope of a `device="cpu"` estimator (and of an
    explicit host class) on a GPU installation, whose metrics tables are the
    CPU images (bindings/x_linear_route_unpack.mojo). The scope was connected
    at import, so NumPy loaded since is connected into it here (one native
    dict probe once connected); the default route is one identity compare."""
    if entries.scope is context:
        return context
    var scope = entries.scope
    classical_connect_numpy(scope)
    return scope


def _xl_check_fitted(fields: PythonObject, model: PythonObject) raises:
    var nk = x_linear_keys()
    if not py_contains(fields, nk[].k_coef_):
        raise Error("RuntimeError: mojolearn " + _name(model) + ": call fit first")


# ------------------------------------------------------------ arrays


def _xl_word(out_buf: PythonObject, index: Int) raises -> Float32:
    return xl_run_word(labels_address(out_buf), index)


def _xl_float(out_buf: PythonObject, index: Int) raises -> PythonObject:
    """`float(vals[index])`: the float32 word as a Python float."""
    return PythonObject(Float64(_xl_word(out_buf, index)))


def _xl_count(out_buf: PythonObject, index: Int) raises -> Int:
    """`int(vals[index])`: the float32 word truncated toward zero."""
    return Int(_xl_word(out_buf, index))


def _xl_slice(out_buf: PythonObject, offset: Int, count: Int, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Array.from_list(vals[offset:offset + count], '<f4')` (shaped as the
    Python list was nested): one byte copy of the float32 words."""
    var array = admitted_array(shape, "<f4", context)
    xl_run_copy(labels_address(array), labels_address(out_buf), offset, count)
    _ = out_buf  # keep the Python object alive through the native call above
    return array


def _xl_join(first: Int, first_count: Int, second: Int, second_count: Int, context: PythonObject) raises -> PythonObject:
    """`_concat_f32(a, b)`: a's then b's 32-bit words in one float32 vector
    (two byte copies)."""
    # fx-rows (2026-11-18): the two copies write every word, so the vector
    # is allocated unfilled (`admitted_storage_native`, a mojolearn Array as
    # `admitted_vector` returns); it was zero-filled and then overwritten
    # (per predict for isotonic's knots). An empty join keeps the old form.
    var total = first_count + second_count
    var out = (admitted_storage_native([total], "<f4", context, as_array=True) if total > 0
               else admitted_vector(0, "<f4", context))
    xl_run_join(labels_address(out), first, first_count, second, second_count)
    return out


def _xl_floats(values: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Array.from_list([float(v) ...], '<f4')` of a Python list of floats:
    `array.array('f', values)` rounds each once, as from_list did."""
    var nk = x_linear_keys()
    var store = py_call(py_item(context, nk[].k_array_type), PythonObject("f"), values)
    return classical_wrap_array(store, Python.tuple(len(store)), "<f4", "C", context)


def _xl_matrix(source: PythonObject, est: String, context: PythonObject) raises -> AdmittedX:
    """`_matrix(X)`: X through the one admission (bindings/admitted_input_unpack.mojo
    admit_x_unrecorded) with scikit-learn's refusals: the caller's float32
    C-order words as they are, anything else one parallel cast."""
    # Lane a1b-doors-opt-in-trees-linear: X was labels_f32 (a second cast
    # path, no finite test). Every entry that reads X tests the words it
    # uploads: `x_linear_fit` (x_linear/finite_device.mojo
    # XLIN_IDN_DEV_FINITE, or the entry's own scan) and the decision entries
    # (x_linear_decision / _codes / _rows: the status word, `_xl_status`).
    # So a borrowed X takes no host pass here (defer_finite), a cast X is
    # tested in its cast pass, and a predict refuses through
    # admitted_finite_refuse (before: a predict never tested X).
    return admit_x_unrecorded(source, est, context, defer_finite=True)


def _xl_status(x: AdmittedXView, context: PythonObject) raises -> Tuple[PythonObject, Int]:
    """The deferred finite status word of `x` for a decision entry: a fresh
    int32 word and its address, the request's status word (None and 0 when
    admit_x already tested X)."""
    if x.finite:
        return (PythonObject(None), 0)
    var word = admitted_vector(1, "<i4", context)
    return (word, labels_address(word))


def _xl_status_refuse(word: PythonObject, model: PythonObject) raises:
    """Refuse a non-finite X the decision entry found (scikit-learn's wording)."""
    if word is not None:
        xl_run_status_refuse(labels_address(word), _name(model))
        _ = word  # keep the Python object alive through the native read above


def _xl_vector(source: PythonObject, n: Int, context: PythonObject) raises -> AdmittedX:
    """`_vector(y, n)`: the fit target, tested by the fit entry with X."""
    var y = admit_f32(source, 1, "y", context, defer_finite=True)
    if y.rows != n:
        raise Error("mojolearn: X and y lengths differ")
    return y^


def _xl_column(source: PythonObject, name: String, context: PythonObject) raises -> AdmittedX:
    """IsotonicRegression's `_column`: one feature, 1-D or (n, 1)."""
    var view = classical_view(source, context, name)
    var nd = Int(py=view.ndim)
    if not (nd == 1 or (nd == 2 and Int(py=view.shape[1]) == 1)):
        raise Error("mojolearn IsotonicRegression: " + name + " must be 1-D or of shape (n, 1)")
    # The isotonic fit / predict entries (x_linear_fit) test the words.
    var column = admit_f32(view, nd, name, context, defer_finite=True, admitted=True)
    _ = view^
    return admitted_words(column.array, column.address, column.rows * column.cols, 1)


def _xl_words(value: PythonObject, name: String, context: PythonObject) raises -> AdmittedX:
    """The float32 words of a fitted array (any rank, read as it is), or a
    Python number as one float32 word (`full((1,), float(value), '<f4')`)."""
    if py_type(value) is _b().float or py_type(value) is _b().int:
        var one = labels_full_f32(1, _float(value), context)
        return admitted_words(one, labels_address(one), 1, 1)
    return admit_f32(value, -1, name, context, finite=False)


def xl_intercept_words(intercept: PythonObject) raises -> Int:
    """k of `_xl_block` (`xl_block_dims`: one class row per intercept word)
    read from the fitted `intercept_` alone, before any admission: a Python
    number is one word, an array its `size`. A regressor's score refuses
    k != 1 with this before it predicts (lane fx-audit)."""
    if py_type(intercept) is _b().float or py_type(intercept) is _b().int:
        return 1
    return Int(py=intercept.size)


def _xl_block(coef: PythonObject, intercept: PythonObject, context: PythonObject) raises -> Tuple[PythonObject, Int]:
    """`_coef_block(coef, intercept)`: [coef row | intercept] per class as
    one float32 vector of k rows of d + 1 words (byte copies)."""
    var c = _xl_words(coef, "coef_", context)
    var b = _xl_words(intercept, "intercept_", context)
    var dims = xl_block_dims(c.rows * c.cols, b.rows * b.cols)
    var k = dims[0]
    var d = dims[1]
    var out = admitted_vector(k * (d + 1), "<f4", context)
    xl_run_block(labels_address(out), c.address, b.address, k, d)
    _ = c^
    _ = b^
    return (out, k)


def _xl_width(x: AdmittedXView, fields: PythonObject, model: PythonObject) raises:
    var nk = x_linear_keys()
    var fitted = py_item(fields, nk[].k_n_features_in_)
    if x.cols != _int(fitted):
        raise Error("mojolearn " + _name(model) + ": X has " + String(x.cols) + " features, the model was fitted with "
                    + _str(fitted))


struct XlDecision(Movable):
    """One admitted decision call: the typed request RUN reads
    (x_linear/door_run.mojo `xl_decision_run`) and the objects whose words
    it addresses, kept alive until PACK returns: X, the [coef | intercept]
    block, the deferred finite status word (None when X was tested at
    admission) and the result."""
    var request: XlDecisionRequest
    var x: PythonObject
    var block: PythonObject
    var status: PythonObject
    var result: PythonObject

    def __init__(out self, var request: XlDecisionRequest, var x: PythonObject, var block: PythonObject,
                 var status: PythonObject, var result: PythonObject):
        self.request = request^
        self.x = x^
        self.block = block^
        self.status = status^
        self.result = result^


def xl_decision_build(operation: Int, x: AdmittedXView, x_keep: PythonObject, coef: PythonObject,
                      intercept: PythonObject, link: Int, squeeze: Bool, strict: Bool, below: Int, above: Int,
                      rows_mode: Int, ops: XLinearOps, context: PythonObject) raises -> XlDecision:
    """UNPACK of one decision over an X already read: the coefficient
    block, the result (XL_OP_DECISION: (n, k) float32, or (n,) when
    `squeeze` and k == 1; XL_OP_DECISION_CODES: (n,) int32;
    XL_OP_DECISION_ROWS: (n, 2 if k == 1 else k) float32) and the status
    word. `x_keep` owns X's words (None when the caller keeps them)."""
    var block = _xl_block(coef, intercept, context)
    var wb = block[0]
    var k = block[1]
    var result: PythonObject
    if operation == XL_OP_DECISION_CODES:
        result = admitted_vector(x.rows, "<i4", context)
    elif operation == XL_OP_DECISION_ROWS:
        result = admitted_array(Python.tuple(x.rows, 2 if k == 1 else k), "<f4", context)
    else:
        result = admitted_array(Python.tuple(x.rows) if squeeze and k == 1 else Python.tuple(x.rows, k), "<f4",
                                context)
    var status = _xl_status(x, context)
    var request = XlDecisionRequest(ops.entry, operation, x.address, labels_address(wb), labels_address(result),
                                    x.rows, x.cols, k, link, strict, below, above, rows_mode, status[1])
    return XlDecision(request^, x_keep, wb, status[0], result)


def xl_decision_released(call: XlDecision) raises:
    """RUN of one decision request with the GIL released."""
    var py = Python()
    with GILReleased(py):
        xl_decision_run(call.request)


def xl_decision_refuse(call: XlDecision, model: PythonObject) raises:
    """Refuse a non-finite X the decision entry found (after RUN)."""
    _xl_status_refuse(call.status, model)


def _xl_fold_matrix(source: PythonObject, est: String, ops: XLinearOps, context: PythonObject) raises -> AdmittedX:
    """`_xl_matrix` for a fit whose binding reads a device fold
    (XL_OP_FIT_FOLD, lane a8c-device-x-linear-prep-rfe): a model-selection
    DeviceRows fold keeps its rows on the device (the result's `store` is
    active and no host words exist); any other input, or a binding without
    the fold entry (XL_HAS_FIT_FOLD), is `_xl_matrix`'s."""
    if door_is_device_rows(source, context):
        # The one question a fold admission asks its binding (typed,
        # x_linear/door_run.mojo `xl_fit_fold_held`), with the GIL released;
        # only a DeviceRows input reaches it.
        var held = False
        var py = Python()
        with GILReleased(py):
            held = xl_fit_fold_held(ops.entry)
        if held:
            return admit_x_unrecorded(source, est, context, defer_finite=True, device_x=True)
    return _xl_matrix(source, est, context)


struct XlFit(Movable):
    """One admitted fit call: the typed request RUN reads
    (x_linear/door_run.mojo `xl_fit_run`) and the objects whose words it
    addresses, kept alive until PACK returns: X (None for a device fold or
    when the caller keeps it), the targets, the one zero word a fit without
    targets reads, and the flat float32 result."""
    var request: XlFitRequest
    var x: PythonObject
    var y: PythonObject
    var zero: PythonObject
    var result: PythonObject

    def __init__(out self, var request: XlFitRequest, var x: PythonObject, var y: PythonObject,
                 var zero: PythonObject, var result: PythonObject):
        self.request = request^
        self.x = x^
        self.y = y^
        self.zero = zero^
        self.result = result^


def xl_fit_build(ops: XLinearOps, algo: Int, x_addr: Int, x_size: Int, n: Int, d: Int, x_keep: PythonObject,
                 y_addr: Int, y_size: Int, y_keep: PythonObject, var ip: List[Int], var fp: List[Float64],
                 n_out: Int, n_fw: Int, n_iw: Int, context: PythonObject, written: Int = 0) raises -> XlFit:
    """UNPACK of one XL_OP_FIT call from host words: the result vector and
    the typed request. The targets are `y_size` 32-bit words at `y_addr`
    (`y_keep` owns them); `y_size` 0 is no targets (one zero word, as
    `zeros((1,))`). `written`: the leading result words the entry writes in
    full on every binding (host and device); those are left unfilled and
    only the words after them start as zeros. 0 (the default) zero-fills
    the whole result."""
    # fx-rows (2026-11-18): the result was zero-filled (0.4 MB for 100k
    # isotonic queries) and then overwritten word for word by the entry, and a
    # one-word zero vector was allocated per call even when targets were
    # given. A caller that proves `written` gets an unfilled result with only
    # its tail zeroed; the zero word is allocated only when there are no
    # targets. Timing: isotonic infer on taxi and istella (board), 50k random
    # monotone pairs (non-board). No result word changes.
    var result: PythonObject
    if written > 0 and written <= n_out:
        result = admitted_storage_native([n_out], "<f4", context, as_array=True)
        var tail = _F32P(unsafe_from_address=labels_address(result))
        for i in range(written, n_out):  # small-loop(the status words after the written prefix)
            tail[i] = Float32(0)
    else:
        result = admitted_vector(n_out, "<f4", context)
    var zero = PythonObject(None)
    if y_size <= 0:
        zero = admitted_vector(1, "<f4", context)
    var ya = y_addr if y_size > 0 else labels_address(zero)
    var request = XlFitRequest(ops.entry, algo, x_addr, x_size, n, d, ya, y_size, ip^, fp^, n_out, n_fw, n_iw,
                               labels_address(result), False, 0, 0, 0)
    return XlFit(request^, x_keep, y_keep, zero, result)


def xl_fit_build_x(ops: XLinearOps, algo: Int, x: AdmittedXView, x_keep: PythonObject, y_addr: Int, y_size: Int,
                   y_keep: PythonObject, var ip: List[Int], var fp: List[Float64], n_out: Int, n_fw: Int,
                   n_iw: Int, context: PythonObject) raises -> XlFit:
    """`xl_fit_build` over an admitted X: a device-resident fold
    (`x.store`) is fitted by XL_OP_FIT_FOLD on rows gathered on the device
    (no host gather, no upload of X); any other X by XL_OP_FIT from its
    host words."""
    if not x.store.active():
        return xl_fit_build(ops, algo, x.address, x.rows * x.cols, x.rows, x.cols, x_keep, y_addr, y_size, y_keep,
                            ip^, fp^, n_out, n_fw, n_iw, context)
    var result = admitted_vector(n_out, "<f4", context)
    var zero = admitted_vector(1, "<f4", context)
    var ya = y_addr if y_size > 0 else labels_address(zero)
    var request = XlFitRequest(ops.entry, algo, 0, x.rows * x.cols, x.rows, x.cols, ya, y_size, ip^, fp^, n_out,
                               n_fw, n_iw, labels_address(result), True, x.store.base_id, x.store.base_rows,
                               x.store.index_id)
    return XlFit(request^, x_keep, y_keep, zero, result)


def xl_fit_released(call: XlFit, model: PythonObject) raises:
    """RUN of one fit request with the GIL released; then the refusal of a
    device fold on its status words (a host fit has none)."""
    var words: Tuple[Int, Int] = (0, 0)
    var py = Python()
    with GILReleased(py):
        words = xl_fit_run(call.request)
    if call.request.fold:
        door_rows_refuse(words[0], words[1], _name(model), call.request.base_rows)



def _xl_with_weights(y: AdmittedX, sample_weight: PythonObject, n: Int, mut has_sw: Int,
                     context: PythonObject) raises -> AdmittedX:
    """`_with_weights(yv, sample_weight, n)`: the targets then the checked
    weights in one float32 vector (has_sw 1) with a sample_weight, the
    targets as they are (has_sw 0) without."""
    if sample_weight is None:
        has_sw = 0
        return admitted_words(y.array, y.address, y.rows * y.cols, 1)
    var w = admit_weight_words(sample_weight, n, context)
    var joined = _xl_join(y.address, y.rows * y.cols, w.address, n, context)
    _ = w  # keep the Python object alive through the native call above
    _ = w^
    has_sw = 1
    var _keep_ret_joined = admitted_words(joined, labels_address(joined), y.rows * y.cols + n, 1)
    _ = joined  # keep the Python object alive through the native call above
    return _keep_ret_joined^


def _xl_owned(m: AdmittedX, context: PythonObject) raises -> PythonObject:
    """A float32 vector Array of m's words: m's own Array when it is one,
    else one byte copy of the caller's buffer."""
    var nk = x_linear_keys()
    if py_type(m.array) is py_item(context, nk[].k_array_class) and labels_address(m.array) == m.address:
        return m.array
    return _xl_join(m.address, m.rows * m.cols, 0, 0, context)


def _xl_reals(values: List[Float64]) raises -> PythonObject:
    return py_list_f64(values)


# ------------------------------------------------------------ class weights


def xl_class_weights_unpack(class_weight: PythonObject, classes: PythonObject,
                            context: PythonObject) raises -> Tuple[PythonObject, Bool]:
    """UNPACK of `_class_prep`'s class weights: the k float32 class-weight
    words (the dict's values by fitted class, ones for None, unset for
    'balanced', which the binding computes) and whether they are
    'balanced'; the refusals of a dict that names an unknown class or of
    any other value. The native step itself (XL_OP_CLASS_PREP) is RUN's
    (x_linear/door_run.mojo `xl_class_prep_run`), made by the fit's RUN."""
    var b = _b()
    var labels = py_call(b.list, classes)
    var k = len(labels)
    var balanced = _eq(class_weight, PythonObject("balanced"))
    var cw = PythonObject(None)
    if balanced:
        cw = admitted_vector(k, "<f4", context)
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
    return (cw, balanced)


def _xl_bayes_refuse(model: PythonObject, fields: PythonObject, return_std: PythonObject) raises:
    """`_bayes_refuse`."""
    var nk = x_linear_keys()
    var name = _name(model)
    if _truth(py_item(fields, nk[].k_compute_score)):
        raise Error("mojolearn " + name + ": compute_score is not implemented")
    if _truth(return_std):
        raise Error("mojolearn " + name + ": predict(return_std=True) is not implemented")


def _xl_or(value: PythonObject, default: Float64) raises -> Float64:
    return default if value is None else _float(value)

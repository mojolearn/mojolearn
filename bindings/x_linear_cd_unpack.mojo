# SPDX-License-Identifier: Apache-2.0
# UNPACK of the coordinate-descent pair's operations (Lasso, ElasticNet;
# python/mojolearn/_solver_impl.py): Python objects become the typed
# requests of x_linear/door_run.mojo (`XlCdFitRequest`,
# `XlCdPredictRequest`), one function per public operation, with the
# refusals in the order and words the Python made them. RUN
# (`xl_cd_fit_run`, `xl_cd_predict_run`) holds no Python object; PACK is
# bindings/x_linear_cd_pack.mojo. The door (bindings/x_linear_native_door.mojo)
# calls the three in order and releases the GIL around RUN.
from x_linear.door_run import XlCdFitRequest, XlCdPredictRequest
from std.python import Python, PythonObject
from bindings.x_linear_route_unpack import XLinearOps, xl_ops_require
from bindings.x_linear_common_unpack import (
    _b, _truth, _float, _int, _repr, _str, _gt, _lt, _eq, _isinstance, _name, _xl_join, _xl_words,
)
from bindings.py_balanced import py_call, py_type, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.admitted_input_unpack import (
    admit_x_unrecorded, admit_f32, admit_weight_words, admitted_array, admitted_vector, classical_view,
)
from bindings.classical_arrays_door import classical_c_order, classical_is_f32
from bindings.x_linear_keys_unpack import x_linear_keys
from bindings.py_balanced import py_item
from bindings.classical_labels_door import labels_address, labels_shape_of

# ElasticNet / Lasso refusals by name, word for word.
comptime _CD_COPY_X = "mojolearn ElasticNet: copy_X=False is refused. The design crosses to the device as a copy and the caller's array is never written (DEVIATION 880); there is no in-place arm to select"
comptime _CD_POSITIVE = "mojolearn ElasticNet: positive=True is not supported (cuML raises UnsupportedOnGPU for it too, elastic_net.py:143)"
comptime _CD_PRECOMPUTE = "mojolearn ElasticNet: precompute is not supported (cuML raises UnsupportedOnGPU for it too, elastic_net.py:149)"
comptime _CD_QN = "mojolearn ElasticNet: solver='qn' is not implemented. cuML's solver='auto' picks 'qn' only for SPARSE input (elastic_net.py:243-244), and 'qn' under an l1 penalty is OWL-QN (min_owlqn, qn_solvers.cuh), which glm/NOT_IMPLEMENTED.tsv lists as not implemented. solver='cd' (cuML's 'auto' for dense input) is the implemented arm; see solver/NOT_IMPLEMENTED.tsv"
comptime _CD_RANDOM = "mojolearn ElasticNet: selection='random' is REFUSED BY NAME. cuML seeds std::mt19937(0) and shuffles the coordinate order with std::shuffle each epoch (cd.cuh:198-199, :222), and the C++ standard does not specify std::shuffle's algorithm, so their permutation is a function of the toolchain rather than of the seed and cannot be gated bitwise or certified across vendors. An exact implementation is DEVIATION 611, reserved and not spent; see solver/NOT_IMPLEMENTED.tsv"
comptime _CD_SPARSE = "mojolearn ElasticNet: sparse X is refused. cdFit is dense only; cuML routes sparse input to solver='qn' and raises for solver='cd' (elastic_net.py:265-269), and 'qn' is not implemented"
comptime _CD_WARM = "mojolearn ElasticNet: warm_start=True is not supported (cuML raises UnsupportedOnGPU for it too, elastic_net.py:146). cdFit reads coef as its starting point and solver/estimator.mojo zeroes it on every fit"


struct XlCdFit(Movable):
    """One admitted `cd_fit`: the typed request RUN reads and the objects
    whose words it addresses (kept alive until PACK returns). `copied`:
    the design was copied on the way in (`input_copied_`)."""
    var request: XlCdFitRequest
    var copied: Bool
    var x: PythonObject
    var target: PythonObject
    var weights: PythonObject
    var coef: PythonObject
    var info: PythonObject

    def __init__(out self, var request: XlCdFitRequest, copied: Bool, var x: PythonObject, var target: PythonObject,
                 var weights: PythonObject, var coef: PythonObject, var info: PythonObject):
        self.request = request^
        self.copied = copied
        self.x = x^
        self.target = target^
        self.weights = weights^
        self.coef = coef^
        self.info = info^


struct XlCdPredict(Movable):
    """One admitted `cd_predict`: the typed request and the objects whose
    words it addresses; `prediction` is the (n,) float32 result."""
    var request: XlCdPredictRequest
    var x: PythonObject
    var coef: PythonObject
    var prediction: PythonObject

    def __init__(out self, var request: XlCdPredictRequest, var x: PythonObject, var coef: PythonObject,
                 var prediction: PythonObject):
        self.request = request^
        self.x = x^
        self.coef = coef^
        self.prediction = prediction^


def _xl_cd_validate(fields: PythonObject, context: PythonObject) raises:
    """ElasticNet._validate_params: cuML's guards first, in their order and
    with their messages, then this implementation's refusals by name."""
    var nk = x_linear_keys()
    var alpha = py_item(fields, nk[].k_alpha)
    var l1_ratio = py_item(fields, nk[].k_l1_ratio)
    var solver = py_item(fields, nk[].k_solver)
    var selection = py_item(fields, nk[].k_selection)
    if _lt(alpha, PythonObject(0.0)):
        raise Error("Expected alpha >= 0, got " + _str(alpha))
    if not py_contains(py_item(context, nk[].k_cd_selection), selection):
        raise Error("selection " + _repr(selection) + " is not supported")
    if _lt(l1_ratio, PythonObject(0.0)) or _gt(l1_ratio, PythonObject(1.0)):
        raise Error("Expected 0.0 <= l1_ratio <= 1.0, got " + _str(l1_ratio))
    if not py_contains(Python.tuple("auto", "cd", "qn"), solver):
        raise Error("solver=" + _repr(solver) + " is not supported")
    if _eq(solver, PythonObject("qn")):
        raise Error("NotImplementedError: " + _CD_QN)
    if _eq(selection, PythonObject("random")):
        raise Error("NotImplementedError: " + _CD_RANDOM)
    if _truth(py_item(fields, nk[].k_positive)):
        raise Error("NotImplementedError: " + _CD_POSITIVE)
    if _truth(py_item(fields, nk[].k_warm_start)):
        raise Error("NotImplementedError: " + _CD_WARM)
    if not (py_item(fields, nk[].k_precompute) is PythonObject(False)):
        raise Error("NotImplementedError: " + _CD_PRECOMPUTE)
    if not _truth(py_item(fields, nk[].k_copy_X)):
        raise Error("NotImplementedError: " + _CD_COPY_X)
    var rs = py_item(fields, nk[].k_random_state)
    var b = _b()
    if rs is not None and (_isinstance(rs, b.bool) or not (_isinstance(rs, b.int)
                                                           or Bool(py=py_call(b.hasattr, rs, PythonObject("randint")))
                                                           or Bool(py=py_call(b.hasattr, rs, PythonObject("integers"))))):
        raise Error("random_state=" + _repr(rs) + ": None, an int or a numpy random state")


def xl_cd_fit_unpack(model: PythonObject, fields: PythonObject, args: PythonObject, entries: XLinearOps,
                      context: PythonObject) raises -> XlCdFit:
    """ElasticNet.fit / Lasso.fit(X, y, sample_weight): the refusals in the
    order the fit always made them, the design as the Python handed it (a
    float32 C-order design as it is, the device transposing it; any other
    input as one F-order float32 copy), the targets, the checked weights,
    the result buffers and the settings as plain words."""
    var nk = x_linear_keys()
    var b = _b()
    if entries.host_only:
        raise Error("ImportError: mojolearn: no CPU implementation of the " + _name(model)
                    + " fit on the host class; the host model predicts a saved model only")
    _xl_cd_validate(fields, context)
    var source = args[0]
    if Bool(py=py_call(b.hasattr, source, PythonObject("toarray"))) or Bool(py=py_call(b.hasattr, source, PythonObject("tocsr"))):
        raise Error("NotImplementedError: " + _CD_SPARSE)
    var listed = py_type(source) is b.list or py_type(source) is b.tuple
    # A float32 C-order design is read as it is (the device transposes it);
    # any other input is one F-order float32 copy (admit_x order "F": one
    # parallel cast-and-transpose pass). cd_fit has no finite test: the
    # words are admitted as they are, as before (finite=False).
    var view = classical_view(source, context, "X")
    var row_major = (Int(py=view.ndim) == 2 and classical_is_f32(view, context) and classical_c_order(view, context)
                     and Int(py=view.shape[0]) > 1 and Int(py=view.shape[1]) > 1)
    var x = admit_x_unrecorded(view, _name(model), context, admitted=True, order="C" if row_major else "F",
                               finite=False)
    _ = view^
    var copied = listed or not x.borrowed
    var n_rows = x.rows
    var n_cols = x.cols
    if n_rows < 2:
        raise Error("Found array with " + String(n_rows) + " sample(s) (shape=(" + String(n_rows) + ", "
                    + String(n_cols) + ")) while a minimum of 2 is required.")
    if len(labels_shape_of(args[1], context)) != 1:
        raise Error("mojolearn ElasticNet currently requires one target")
    var target = admit_f32(args[1], 1, "y", context, finite=False)
    if target.rows != n_rows:
        raise Error("mojolearn ElasticNet X and y lengths differ")
    var sw_keep = PythonObject(None)
    var sw_addr = 0
    if args[2] is not None:
        var w = admit_weight_words(args[2], n_rows, context)
        sw_keep = w.array
        sw_addr = w.address
        _ = w^
    xl_ops_require(entries)
    var coef = admitted_array(Python.tuple(n_cols), "<f4", context)
    var info = admitted_vector(1, "<f4", context)
    var fit_intercept = _truth(py_item(fields, nk[].k_fit_intercept))
    var epochs = _int(py_item(fields, nk[].k_max_iter))
    var alpha = _float(py_item(fields, nk[].k_alpha))
    var l1_ratio = _float(py_item(fields, nk[].k_l1_ratio))
    var tol = _float(py_item(fields, nk[].k_tol))
    var shuffle = _int(py_item(context, nk[].k_cd_selection)[py_item(fields, nk[].k_selection)]) != 0
    var request = XlCdFitRequest(entries.entry, x.address, target.address, labels_address(coef),
                                 labels_address(info), sw_addr, n_rows, n_cols, fit_intercept, epochs, shuffle,
                                 row_major, alpha, l1_ratio, tol)
    return XlCdFit(request^, copied, x.array, target.array, sw_keep, coef, info)


def xl_cd_predict_unpack(model: PythonObject, source: PythonObject, entries: XLinearOps,
                          context: PythonObject) raises -> XlCdPredict:
    """ElasticNet.predict / Lasso.predict(X): the F-order float32 design,
    the fitted coefficients' words, the intercept and the result vector."""
    var nk = x_linear_keys()
    var fields = py_fields(model)
    if not py_contains(fields, nk[].k_coef_):
        raise Error("mojolearn ElasticNet: call fit before predict")
    var x = admit_x_unrecorded(source, "ElasticNet", context, order="F", finite=False)
    if x.cols != _int(py_item(fields, nk[].k_n_features_in_)):
        raise Error("mojolearn ElasticNet feature count differs from fit")
    xl_ops_require(entries)
    var coef = _xl_words(py_item(fields, nk[].k_coef_), "coef_", context)
    var prediction = admitted_vector(x.rows, "<f4", context)
    var intercept = _float(py_item(fields, nk[].k_intercept_))
    var request = XlCdPredictRequest(entries.entry, x.address, coef.address, labels_address(prediction), x.rows,
                                     x.cols, intercept)
    return XlCdPredict(request^, x.array, coef.array, prediction)


def xl_cd_host_prediction_unpack(model: PythonObject, source: PythonObject,
                                 context: PythonObject) raises -> PythonObject:
    """The prediction an explicit host class scores: its own native host
    entry (`host_infer_predict`) over X, as one float32 vector."""
    var nk = x_linear_keys()
    var host = py_item(context, nk[].k_cd_host_predict)
    var published = py_call_typed(host[0], model, PythonObject("host_infer_predict"), Python.tuple(source), host[1])
    var cast = admit_f32(published, 1, "predictions", context, finite=False)
    var prediction = _xl_join(cast.address, cast.rows, 0, 0, context)
    _ = cast^
    return prediction

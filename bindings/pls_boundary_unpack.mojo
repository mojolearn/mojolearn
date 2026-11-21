# SPDX-License-Identifier: Apache-2.0
# UNPACK of the PLS family (PLSRegression, PLSCanonical, CCA; the door is
# bindings/pls_native_door.mojo): the settings snapshot and its scalar
# admission, the fitted test, the one admission of the public X and of the
# target, the kit operands (the caller's matrices and the fitted operands,
# resident on a query), the fit's option words and the metrics module of
# `score`. Everything RUN reads leaves here as kit operand numbers and
# plain words (PlsFitUnpacked, PlsOperands); RUN (bindings/pls_run.mojo)
# sees no Python object. Same checks, same order, same words as the door
# held before the split.
from core.mode_answer import MODE_FAMILY_PLS, answer_mode_word
from core.array_metadata import ArrayMetadata
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains, py_item
from bindings.classical_exceptions_door import classical_not_fitted_class
from bindings.classical_validate_run import classical_check_width
from providers.python.estimator_lifecycle_pyapi_unpack import estimator_feature_check
from providers.python.array_metadata_pyapi_door import ArrayPin, array_shape_once, array_metadata_once
from bindings.admitted_input_unpack import admit_f32, admit_x_unrecorded, classical_view
from providers.python.estimator_binding_pyapi_door import estimator_binding_routed
from providers.python.classical_family_admission_pyapi_unpack import classical_family_admit
from bindings.native_scalar_decode_unpack import native_scalar_buffer
from bindings.sequential_admission_door import _one_of
from bindings.classifier_public_lifecycle_door import _feature_names
from bindings.classical_arrays_door import classical_kind
from bindings.host_sparse_native_door import host_sparse_format, host_sparse_dense
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_entry
from bindings.decomp_kit_native_door import (
    DecompDoor, decomp_int, decomp_float, decomp_entries_routed, decomp_is_a, decomp_class_attr,
    decomp_door_snapshot, decomp_door_tested, decomp_door_resident_hit, decomp_door_resident_input,
)
from bindings.decomp_kit_run import DKit
from bindings.pls_run import PlsFitWords
from bindings.pls_keys_unpack import (
    pls_keys,
    PLS_OPN_FIT,
    PLS_OPN_FIT_TRANSFORM,
    PLS_OPN_SCORE,
    PLS_OPN_INVERSE_TRANSFORM,
)

def _name(model: PythonObject) raises -> String:
    return String(py=py_type(model).__name__)



def _pls_scalar(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if value is None or cls is b.bool or cls is b.int or cls is b.float or cls is b.str or cls is b.bytes:
        return value
    var nk = pls_keys()  # after the plain-scalar return: that path reads no key
    var array = cls is py_get(context, nk[].k_ndarray)
    if not array and not _one_of(cls, py_item(context, nk[].k_scalar_types)):
        raise Error("TypeError: PLS " + name + " requires a native scalar without Python conversion hooks")
    if array and Int(py=value.ndim) != 0:
        raise Error("TypeError: PLS " + name + " requires a scalar or a zero-dimensional native numeric array")
    # The raw decoder accepts primitive numeric buffer formats only. For O
    # dtype it refuses before publishing a number, without invoking the stored
    # object's __int__, __float__ or __bool__. No NumPy conversion runs.
    var scalar = native_scalar_buffer(value)
    if scalar.complex:
        raise Error("TypeError: PLS " + name + " requires a native real scalar")
    if name == "scale" or name == "copy" or name == "_predict_1d" or name == "_mojolearn_fitted":
        return PythonObject(scalar.truth())
    return scalar.to_python_number(name != "tol")


def _pls_settings(model: PythonObject, state: PythonObject, operation: Int, operation_name: PythonObject,
                  context: PythonObject) raises -> PythonObject:
    var nk = pls_keys()
    var b = native_loaded_module("builtins")
    # Exact-dict copy captures references atomically while admission holds the
    # GIL. Keep model writes separate; no later numerical stage reads live
    # instance settings or invokes a scalar conversion protocol.
    var fields = py_call(py_item(py_type(Python.dict()).__dict__, nk[].k_copy), state)
    py_setitem(fields, nk[].k__pls_name, PythonObject(_name(model)))
    py_setitem(fields, nk[].k__pls_regression, PythonObject(decomp_is_a(model, py_item(context, nk[].k_pls_regression_class))))
    py_setitem(fields, nk[].k__pls_canonical, PythonObject(decomp_is_a(model, py_item(context, nk[].k_pls_canonical_class))))
    var fitting = operation == PLS_OPN_FIT or operation == PLS_OPN_FIT_TRANSFORM
    var names = Python.tuple("_mojolearn_fitted", "_mojolearn_fitting", "_predict_1d", "n_features_in_")
    if fitting:
        names = Python.tuple("n_components", "scale", "max_iter", "tol", "copy", "_mojolearn_fitting")
    for index in range(len(names)):  # driver-loop: operation-specific scalar metadata snapshot
        if py_contains(fields, names[index]):
            py_setitem(fields, names[index], _pls_scalar(fields[names[index]], String(names[index]), context))
    if not fitting:
        # Fitted detection must precede synthesized default-mode metadata.
        _pls_require_fitted(fields, operation_name, context)
    # Optional, not a PythonObject(None) sentinel: Mojo 1.2.0 fails to lower a loop-carried PythonObject read and reassigned in the loop (modular/modular#7546).
    var default_mode = Optional[PythonObject]()
    var modes = Python.tuple("numeric_mode") if fitting else (Python.tuple("numeric_mode", "numeric_mode_") if operation == PLS_OPN_SCORE else Python.tuple("numeric_mode_"))
    for index in range(len(modes)):  # driver-loop: current/fitted modes actually consumed by this operation
        var mode = py_get(fields, modes[index])
        if mode is None:
            if not default_mode:
                default_mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
            mode = default_mode.value()
        if py_type(mode) is not b.str:
            raise Error("numeric_mode must be fast, identical or None")
        py_setitem(fields, modes[index], mode)
    if fitting:
        var algorithm = py_get(fields, nk[].k_algorithm, PythonObject("nipals"))
        if py_type(algorithm) is not b.str:
            raise Error("PLSCanonical: algorithm must be 'nipals' or 'svd'")
        py_setitem(fields, nk[].k_algorithm, algorithm)
        for key in ["_deflation", "_pmode"]:  # small-loop(fixed class algorithm constants)
            var value = decomp_class_attr(model, key)
            if py_type(value) is not b.str:
                raise Error("TypeError: PLS algorithm constants must remain native strings")
            py_setitem(fields, key, value)
    return fields


def _pls_require_fitted(fields: PythonObject, operation_name: PythonObject, context: PythonObject) raises:
    var nk = pls_keys()
    var fitted = Bool(py=py_get(fields, nk[].k__mojolearn_fitted, PythonObject(False))) or Bool(py=py_get(fields, nk[].k__mojolearn_fitting, PythonObject(False)))
    if not fitted:
        var keys = py_call(native_loaded_module("builtins").list, fields)
        for index in range(len(keys)):  # driver-loop: existing fitted-public-attribute compatibility rule
            var key = String(keys[index])
            if key.endswith("_") and not key.startswith("_") and fields[key] is not None:
                fitted = True
                break
    if not fitted:
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: This " + String(py_item(fields, nk[].k__pls_name)) + " instance is not fitted yet. Call 'fit' with appropriate arguments before using " + String(py=operation_name) + ".")


def _pls_input(source: PythonObject, name: String, context: PythonObject, mut pin: ArrayPin, ndim: Int = 2,
               public_name: String = "", public_target: Bool = False, sparse_allowed: Bool = True) raises -> ArrayMetadata:
    var deferred = False
    return _pls_input_deferred(source, name, context, pin, ndim, public_name, public_target, sparse_allowed, False,
                               deferred)


def _pls_input_deferred(source: PythonObject, name: String, context: PythonObject, mut pin: ArrayPin, ndim: Int,
                        public_name: String,
                        public_target: Bool, sparse_allowed: Bool, defer: Bool, mut deferred: Bool) raises -> ArrayMetadata:
    """A PLS operand through the one admission (bindings/admitted_input_unpack.mojo):
    the public X by admit_x_unrecorded (scikit-learn's refusals and wording),
    a target or fitted operand of rank `ndim` (-1: 1-D or 2-D) by admit_f32;
    then its captured metadata (array_metadata_once) for the typed kit. A
    float32 C-order operand is borrowed, anything else one parallel cast
    with the finite test in the pass."""
    # TOMBSTONE: decomp_input_snapshot (providers/python/decomp_input_pyapi.mojo, PLS's own cast-and-scan copy) deleted 2026-11-09 by lane a1b-doors-opt-in-trees-linear; code recoverable at f59d913b0b.
    # Restore: experiments/removed/A1B_LABELS_ADMISSION_COPY.patch.
    var value = source
    var format = host_sparse_format(source)
    if format.byte_length() > 0:
        if not sparse_allowed:
            raise Error("TypeError: Sparse data was passed for X, but dense data is required. Use '.toarray()' to convert to a dense numpy array. (" + public_name + " takes dense X only.)")
        value = host_sparse_dense(source, format, context)
    var array = PythonObject(None)
    deferred = False
    if public_name != "" and not public_target:
        # TOMBSTONE: A8C_PLS_PUBLIC_X_HOST_SCAN (removed: superseded by the kit upload's device test) deleted 2026-11-09 by 197546da0d; code recoverable at 197546da0d^.
        # Restore: git apply experiments/removed/A8C_PLS_PUBLIC_X_HOST_SCAN.patch
        # Lane a8c-device-x-linear-prep-rfe: with `defer` (the operation's
        # kit tests its upload, DKit.defers_finite) a borrowed X takes
        # no host pass here; `deferred` tells the caller to add it through
        # decomp_door_tested. A cast X is tested in its cast pass.
        var admitted = admit_x_unrecorded(value, public_name, context, defer_finite=defer)
        deferred = not admitted.finite
        array = admitted.array
    else:
        var view = classical_view(value, context, name)
        if classical_kind(view, context) == "c":
            if public_name != "":
                raise Error("Complex data not supported: " + public_name + " got a complex y.")
            raise Error("TypeError: mojolearn: " + name + " has a non-numeric dtype")
        var nd = Int(py=view.ndim)
        if (ndim >= 0 and nd != ndim) or nd == 0 or nd > 2:
            raise Error("mojolearn: " + name + " must be " + String(ndim if ndim >= 0 else 2) + "-D, got "
                        + String(nd) + "-D")
        try:
            array = admit_f32(view, nd, name, context, admitted=True).array
        except error:
            if "contains NaN or infinity" in String(error):
                if public_target and public_name != "":
                    raise Error("Input y contains NaN or infinity (" + public_name + ").")
                raise Error(name + ": input must be finite; NaN/inf are unsupported")
            raise error^
        _ = view^
    var snapshot = array_metadata_once(array, array_shape_once(array, context), name, context, pin)
    # Pin the caller's object with the admitted words for the operation.
    pin.owner = Python.tuple(source, pin.owner)
    return snapshot^


def _matrix(mut door: DecompDoor, mut kit: DKit, source: PythonObject, name: String,
            context: PythonObject) raises -> Int:
    var pin = ArrayPin()
    var input = _pls_input(source, name, context, pin)
    return decomp_door_snapshot(door, kit, input, pin, input.shape[0], input.shape[1])


def _captured(mut door: DecompDoor, mut kit: DKit, captured: ArrayMetadata, pin: ArrayPin, deferred: Bool,
              est: String) raises -> Int:
    """The public X the door admitted (`pls_public_operation`) as a kit
    matrix: tested on its upload (decomp_door_tested) when the door deferred
    its finite test, else borrowed."""
    if deferred:
        if not kit.defers_finite():
            raise Error("RuntimeError: mojolearn " + est + ": X was admitted untested but this binding has no tested upload")
        return decomp_door_tested(door, kit, captured, pin, captured.shape[0], captured.shape[1], est)
    return decomp_door_snapshot(door, kit, captured, pin, captured.shape[0], captured.shape[1])


def _x(mut door: DecompDoor, mut kit: DKit, args: PythonObject, captured_x: Optional[ArrayMetadata],
       x_pin: Optional[ArrayPin], x_deferred: Bool, fields: PythonObject, context: PythonObject) raises -> Int:
    var nk = pls_keys()
    # `x_pin`: the door's pin of `captured_x`.
    if captured_x:
        return _captured(door, kit, captured_x.value(), x_pin.value(), x_deferred, String(py_item(fields, nk[].k__pls_name)))
    return _matrix(door, kit, args[0], "X", context)


def _pls_defers(model: PythonObject, fields: PythonObject, fitting: Bool, context: PythonObject) -> Bool:
    """Whether the operation's kit tests X on its upload (the x_decomp GPU
    binding's kernel-call op XD_DEV_UPLOAD_FINITE, lane a8c): then the door admits a
    borrowed X with no host pass. False when the entries cannot be resolved
    here; the kit then raises its own refusal as before. The entries are the
    ESTIMATOR'S (its device route): the operation's own resolution below is
    then a route hit."""
    try:
        var mode = fields["numeric_mode"] if fitting else py_get(fields, "numeric_mode_")
        var entries = decomp_entries_routed(model, fields, mode, context)
        return py_get(entries, "x_decomp_dev_ew") is not None and py_get(entries, "x_decomp_module") is not None
    except error:
        _ = error^
        return False


def _fitted_matrix(mut door: DecompDoor, mut kit: DKit, fields: PythonObject, name: String,
                   context: PythonObject) raises -> Int:
    # UNPACK a fitted operand: on a query its device copy is kept across
    # calls (the model's record), else borrowed as _matrix.
    var source = fields[name]
    var hit = decomp_door_resident_hit(door, kit, source, name)  # frozen and resident: no re-admission
    if hit >= 0:
        return hit
    var pin = ArrayPin()
    var input = _pls_input(source, name, context, pin)
    var held = decomp_door_resident_input(door, kit, source, name, pin.keep, input.source.addr, pin.owner,
                                          input.shape[0], input.shape[1])
    if held >= 0:
        return held
    return decomp_door_snapshot(door, kit, input, pin, input.shape[0], input.shape[1])


def _fitted_row(mut door: DecompDoor, mut kit: DKit, fields: PythonObject, name: String,
                context: PythonObject) raises -> Int:
    # A fitted 1-D float32 vector as a 1 x n matrix, resident as _fitted_matrix.
    var source = fields[name]
    var hit = decomp_door_resident_hit(door, kit, source, name)  # frozen and resident: no re-admission
    if hit >= 0:
        return hit
    var pin = ArrayPin()
    var input = _pls_input(source, name, context, pin, 1)
    var held = decomp_door_resident_input(door, kit, source, name, pin.keep, input.source.addr, pin.owner, 1,
                                          input.shape[0])
    if held >= 0:
        return held
    return decomp_door_snapshot(door, kit, input, pin, 1, input.shape[0])


def _target(mut door: DecompDoor, mut kit: DKit, source: PythonObject, context: PythonObject,
            captured: Optional[ArrayMetadata] = None,
            captured_pin: Optional[ArrayPin] = None) raises -> Tuple[Int, Bool]:
    # Admit the exact supported input before reading any metadata. A vector
    # keeps the old row-then-transpose execution order, using captured shape
    # and address instead of source.shape, source.reshape or __array__ hooks
    # (the transpose of a row is a view: no operation runs).
    if source is None:
        return (Int(-1), False)
    var pin = ArrayPin()
    var input: ArrayMetadata
    if captured:
        input = captured.value().copy()
        pin = captured_pin.value().copy()
    else:
        input = _pls_input(source, "Y", context, pin, -1)
    if len(input.shape) == 2:
        return (decomp_door_snapshot(door, kit, input, pin, input.shape[0], input.shape[1]), False)
    var row = decomp_door_snapshot(door, kit, input, pin, 1, input.shape[0])
    return (kit.transpose(row), True)


def pls_store(fields: PythonObject, state: PythonObject, key: String, value: PythonObject) raises:
    # Keep this operation's generated values stable even if another caller
    # mutates the exposed model while a later native numerical call runs.
    py_setitem(fields, key, value)
    py_setitem(state, key, value)


# ---------------------------------------------------------------- typed operands


struct PlsFitUnpacked(Copyable, Movable):
    """A fit for RUN: the kit operands of X and of the target (a vector
    target as a column), their widths, and the option words."""
    var x: Int
    var ym: Int
    var p: Int
    var q: Int
    var words: PlsFitWords

    def __init__(out self, x: Int, ym: Int, p: Int, q: Int, words: PlsFitWords):
        self.x = x
        self.ym = ym
        self.p = p
        self.q = q
        self.words = words.copy()


struct PlsOperands(Copyable, Movable):
    """A query for RUN, as kit operand numbers (-1: absent). `a` is the
    public X (transform, predict) or the scores (inverse_transform) with the
    fitted mean, scale and basis it uses (rotations, loadings or
    coefficients); `b` the optional second operand (Y, or the Y scores) with
    its own. predict keeps the target mean in `b_mean`."""
    var a: Int
    var a_mean: Int
    var a_std: Int
    var a_basis: Int
    var b: Int
    var b_mean: Int
    var b_std: Int
    var b_basis: Int

    def __init__(out self):
        self.a = -1
        self.a_mean = -1
        self.a_std = -1
        self.a_basis = -1
        self.b = -1
        self.b_mean = -1
        self.b_std = -1
        self.b_basis = -1


struct PlsPublicUnpacked(Movable):
    """One public call after its admission: the instance state, the private
    settings snapshot every later stage reads, the admitted X (and the
    target of a fit) with the door's pins, and the feature names a fit
    publishes."""
    var state: PythonObject
    var fields: PythonObject
    var name: String
    var fitting: Bool
    var x: ArrayMetadata
    var x_pin: ArrayPin
    var x_deferred: Bool
    var y: Optional[ArrayMetadata]
    var y_pin: Optional[ArrayPin]
    var feature_names: PythonObject

    def __init__(out self, state: PythonObject, fields: PythonObject, name: String, fitting: Bool,
                 var x: ArrayMetadata, var x_pin: ArrayPin, x_deferred: Bool, var y: Optional[ArrayMetadata],
                 var y_pin: Optional[ArrayPin], feature_names: PythonObject):
        self.state = state
        self.fields = fields
        self.name = name
        self.fitting = fitting
        self.x = x^
        self.x_pin = x_pin^
        self.x_deferred = x_deferred
        self.y = y^
        self.y_pin = y_pin^
        self.feature_names = feature_names


def pls_public_unpack(model: PythonObject, operation: Int, operation_name: PythonObject, args: PythonObject,
                      context: PythonObject) raises -> PlsPublicUnpacked:
    """The admission of one public call (the former _track_fit / _needs_fit
    envelope's input half). Import metadata and shared layout admission
    prevent Python descriptors before inspecting model settings; caller
    inputs are admitted before metadata."""
    var nk = pls_keys()
    var state = classical_family_admit(model)
    var fields = _pls_settings(model, state, operation, operation_name, context)
    var b = native_loaded_module("builtins")
    var fitting = operation == PLS_OPN_FIT or operation == PLS_OPN_FIT_TRANSFORM
    var name = String(py_item(fields, nk[].k__pls_name))
    var sparse = Bool(py=py_item(fields, nk[].k__pls_regression))
    # Lane a8c: the X of an operation whose kit tests its upload is admitted
    # with no host pass (`x_deferred`; decomp_door_tested refuses it).
    var x_deferred = False
    var x_pin = ArrayPin()
    var x = _pls_input_deferred(args[0], "X", context, x_pin, 2, name, False, sparse,
                                _pls_defers(model, fields, fitting, context), x_deferred)
    var y = Optional[ArrayMetadata]()
    var y_pin = Optional[ArrayPin]()
    if fitting:
        var target = args[1] if args[1] is not None else args[2]
        if target is None:
            raise Error(name + " requires y to be passed, but the target y is None.")
        var target_pin = ArrayPin()
        y = _pls_input(target, "Y", context, target_pin, -1, name, True)
        y_pin = Optional(target_pin^)
        if x.shape[0] != y.value().shape[0]:
            raise Error("Found input variables with inconsistent numbers of samples: [" + String(x.shape[0]) + ", " + String(y.value().shape[0]) + "]")
    elif operation != PLS_OPN_INVERSE_TRANSFORM:
        var features = py_get(fields, nk[].k_n_features_in_)
        if features is not None:
            if py_type(features) is not b.int:
                raise Error("TypeError: native PLS feature count must be an exact integer")
            classical_check_width(x.shape[1], Int(py=features), name)
        estimator_feature_check(model, fields, args[0], context)
    var feature_names = _feature_names(args[0], context) if fitting else PythonObject(None)
    return PlsPublicUnpacked(state, fields, name, fitting, x^, x_pin^, x_deferred, y^, y_pin^, feature_names)


def pls_algorithm_check(fields: PythonObject) raises:
    """PLSCanonical's `algorithm` value. The settings snapshot already holds
    an exact string (`_pls_settings` refuses any other type in the same
    words), so only the value is tested here."""
    var nk = pls_keys()
    if Bool(py=py_item(fields, nk[].k__pls_canonical)):
        var algorithm = String(py=py_item(fields, nk[].k_algorithm))
        if algorithm != "nipals" and algorithm != "svd":
            raise Error("PLSCanonical: algorithm must be 'nipals' or 'svd'")


def pls_fit_unpack(mut door: DecompDoor, mut kit: DKit, x_source: PythonObject, y_source: PythonObject,
                   context: PythonObject, fields: PythonObject, state: PythonObject,
                   captured_x: Optional[ArrayMetadata], captured_y: Optional[ArrayMetadata], x_deferred: Bool,
                   x_pin: Optional[ArrayPin], y_pin: Optional[ArrayPin]) raises -> PlsFitUnpacked:
    """`_PLS.fit`'s operands, shape rules and option words, from the
    captured settings. `x_pin` / `y_pin` are the door's pins of
    `captured_x` / `captured_y`. Publishes `_predict_1d` where the fit
    always did (after the component bound, before the option words)."""
    var nk = pls_keys()
    var x = _x(door, kit, Python.tuple(x_source), captured_x, x_pin, x_deferred, fields, context)
    var target = _target(door, kit, y_source, context, captured_y, y_pin)
    var ym = target[0]
    if ym < 0:
        raise Error(String(py_item(fields, nk[].k__pls_name)) + " requires y to be passed, but the target y is None.")
    var n = kit.nrows(x)
    var p = kit.ncols(x)
    var q = kit.ncols(ym)
    if kit.nrows(ym) != n:
        raise Error("Found input variables with inconsistent numbers of samples: [" + String(n) + ", "
                    + String(kit.nrows(ym)) + "]")
    if n < 2:
        raise Error("Found array with " + String(n) + " sample(s) (shape=(" + String(n) + ", " + String(p)
                    + ")) while a minimum of 2 is required.")
    var nc = decomp_int(py_item(fields, nk[].k_n_components))
    var deflation = py_item(fields, nk[].k__deflation)
    if py_type(deflation) is not py_type(PythonObject("")):
        raise Error("TypeError: PLS deflation must remain a native string")
    var canonical = String(py=deflation) == "canonical"
    var bound = min(n, min(p, q)) if canonical else p
    if not (1 <= nc and nc <= bound):
        raise Error("n_components == " + String(nc) + ", while 1 <= n_components <= " + String(bound) + " is required")
    pls_store(fields, state, "_predict_1d", PythonObject(target[1]))
    var scale = Bool(py=py_item(fields, nk[].k_scale))
    # An exact string by the settings snapshot (`_pls_settings`); no second type test.
    var svd_algo = String(py=py_get(fields, nk[].k_algorithm, PythonObject("nipals"))) == "svd"
    var pmode = py_item(fields, nk[].k__pmode)
    if py_type(pmode) is not py_type(PythonObject("")):
        raise Error("TypeError: PLS component mode must remain a native string")
    var words = PlsFitWords(nc, decomp_int(py_item(fields, nk[].k_max_iter)), String(py=pmode) == "B", canonical, svd_algo,
                            scale, decomp_float(py_item(fields, nk[].k_tol)))
    return PlsFitUnpacked(x, ym, p, q, words)


def pls_transform_unpack(mut door: DecompDoor, mut kit: DKit, args: PythonObject, y: PythonObject,
                         captured_x: Optional[ArrayMetadata], x_pin: Optional[ArrayPin], x_deferred: Bool,
                         fields: PythonObject, context: PythonObject) raises -> PlsOperands:
    """transform(X, Y): X, Y, then the fitted state they use."""
    var o = PlsOperands()
    o.a = _x(door, kit, args, captured_x, x_pin, x_deferred, fields, context)
    o.b = _target(door, kit, y, context)[0]
    o.a_mean = _fitted_row(door, kit, fields, "_x_mean", context)
    o.a_std = _fitted_row(door, kit, fields, "_x_std", context)
    o.a_basis = _fitted_matrix(door, kit, fields, "x_rotations_", context)
    if o.b >= 0:
        o.b_mean = _fitted_row(door, kit, fields, "_y_mean", context)
        o.b_std = _fitted_row(door, kit, fields, "_y_std", context)
        o.b_basis = _fitted_matrix(door, kit, fields, "y_rotations_", context)
    return o^


def pls_inverse_unpack(mut door: DecompDoor, mut kit: DKit, args: PythonObject, y: PythonObject,
                       captured_x: Optional[ArrayMetadata], x_pin: Optional[ArrayPin], x_deferred: Bool,
                       fields: PythonObject, context: PythonObject) raises -> PlsOperands:
    """inverse_transform(X scores, Y scores): the scores, then the fitted
    loadings, scale and mean, in the order the door always admitted them."""
    var o = PlsOperands()
    o.a = _x(door, kit, args, captured_x, x_pin, x_deferred, fields, context)
    o.a_basis = _fitted_matrix(door, kit, fields, "x_loadings_", context)
    o.a_std = _fitted_row(door, kit, fields, "_x_std", context)
    o.a_mean = _fitted_row(door, kit, fields, "_x_mean", context)
    if y is not None:
        o.b = _matrix(door, kit, y, "Y", context)
        o.b_basis = _fitted_matrix(door, kit, fields, "y_loadings_", context)
        o.b_std = _fitted_row(door, kit, fields, "_y_std", context)
        o.b_mean = _fitted_row(door, kit, fields, "_y_mean", context)
    return o^


def pls_predict_unpack(mut door: DecompDoor, mut kit: DKit, args: PythonObject,
                       captured_x: Optional[ArrayMetadata], x_pin: Optional[ArrayPin], x_deferred: Bool,
                       fields: PythonObject, context: PythonObject) raises -> PlsOperands:
    """predict(X): X, the fitted X mean, the coefficients, the target mean."""
    var o = PlsOperands()
    o.a = _x(door, kit, args, captured_x, x_pin, x_deferred, fields, context)
    o.a_mean = _fitted_row(door, kit, fields, "_x_mean", context)
    o.a_basis = _fitted_matrix(door, kit, fields, "coef_", context)
    o.b_mean = _fitted_row(door, kit, fields, "_y_mean", context)
    return o^


def pls_check_fitted(model: PythonObject, fields: PythonObject) raises:
    var nk = pls_keys()
    if not py_contains(fields, nk[].k_coef_):
        raise Error("RuntimeError: " + String(py_item(fields, nk[].k__pls_name)) + " is not fitted; call fit first")


def pls_score_binding(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = pls_keys()
    # `_metrics_impl._get_binding` belongs to this whole public operation.
    # Use the import-connected provider and preserve its metrics-specific
    # mode refusals/readback before invoking the existing native metric owner.
    var mode = py_get(fields, nk[].k_numeric_mode)
    var b = native_loaded_module("builtins")
    var expected = String("")
    if mode is not None:
        if py_type(mode) is not b.str:
            raise Error("numeric_mode must be 'fast' or 'identical'")
        # One spelling normalization, reused below (it was computed twice).
        expected = String(py=py_call(py_call(mode.strip).lower))
        if expected != "fast" and expected != "identical":
            raise Error("numeric_mode must be 'fast' or 'identical'")
    else:
        mode = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
        if py_type(mode) is not b.str:
            raise Error("numeric_mode must be 'fast' or 'identical'")
        expected = String(py=py_call(py_call(mode.strip).lower))
    answer_mode_word(expected, MODE_FAMILY_PLS)
    # The metrics module of the ESTIMATOR'S device (its binding route), not the process set.
    var module = estimator_binding_routed(model, mode, "_mojolearn_metrics", context, False)
    var reader = neural_descriptor_entry(module, "metrics_numeric_mode", False)
    if reader is None:
        reader = neural_descriptor_entry(module, "umap_numeric_mode", False)
    if reader is not None:
        var word = Int(py=py_call(reader))
        var actual = String("fast") if word == 0 else (String("identical") if word == 1 else String("None"))
        if actual != expected:
            raise Error("RuntimeError: mojolearn metrics: requested " + expected + ", binary reports " + actual + "; rebuild it")
    return module

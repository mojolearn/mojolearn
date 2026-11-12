# SPDX-License-Identifier: Apache-2.0
# The native owner of python/mojolearn/_expansion_decomp.py's PLS family
# (PLSRegression, PLSCanonical, CCA): fit, fit_transform, transform,
# inverse_transform, predict and score are each one call into pls_c. The owner runs
# the steps `_PLS` ran (the centering and scaling, the one
# `x_decomp_pls_fit` component loop, the rotations through `_pinv`, the
# coefficients, the projections) through bindings/decomp_kit_native.mojo:
# the same x_decomp entries with the same words in the same order, so the
# bits are unchanged. fit_transform projects the X and Y the fit already
# holds (one conversion and upload of each).
#
# The fitted centering state is kept as float32 vectors (`_x_mean`,
# `_x_std`, `_y_mean`, `_y_std`, scikit-learn's names), the values the
# Python shell held as device matrices.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words_unpack import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_exceptions_door import classical_not_fitted_class
from bindings.classical_error_boundary_pack import classical_python_error
from bindings.classical_validate_run import classical_check_width
from bindings.parallel_public_result_door import parallel_public_result
from bindings.metrics_native_support_door import metric_warn
from providers.python.estimator_lifecycle_pyapi_unpack import estimator_feature_check
from providers.python.array_metadata_pyapi_door import ArrayMetadata, ArrayPin
from bindings.admitted_input import admit_x
from bindings.admitted_input import admit_f32
from bindings.metrics_cls_native_unpack import metrics_cls_operation
from providers.python.estimator_binding_pyapi_door import estimator_binding_mode
from providers.python.classical_family_admission_pyapi_unpack import classical_family_admit
from bindings.native_scalar_decode_unpack import native_scalar_buffer
from bindings.sequential_admission_door import _one_of
from bindings.classifier_public_lifecycle_door import _feature_names
from bindings.admitted_input import admit_x_unrecorded, classical_view
from bindings.classical_arrays_door import classical_kind
from bindings.host_sparse_native_door import host_sparse_format, host_sparse_dense
from providers.python.array_metadata_pyapi_door import array_shape_once, array_metadata_once
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry
from bindings.decomp_kit_native_door import (
    DecompDoor, decomp_int, decomp_float, decomp_entries, decomp_is_a, decomp_class_attr, decomp_kit_unpack,
    decomp_kit_publish, decomp_kit_out_shape, decomp_door_snapshot, decomp_door_tested, decomp_door_resident_hit,
    decomp_door_resident_input,
)
from bindings.decomp_kit_run import DKit
from bindings.pls_run import PlsFitWords, PlsFit, pls_fit_run, pls_project, pls_rebuild, pls_predict_run
from std.python._cpython import GILReleased

def _name(model: PythonObject) raises -> String:
    return String(py=py_type(model).__name__)



def _pls_scalar(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if value is None or cls is b.bool or cls is b.int or cls is b.float or cls is b.str or cls is b.bytes:
        return value
    var array = cls is py_get(context, "ndarray")
    if not array and not _one_of(cls, context["scalar_types"]):
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


def _pls_settings(model: PythonObject, state: PythonObject, operation: String,
                  context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    # Exact-dict copy captures references atomically while admission holds the
    # GIL. Keep model writes separate; no later numerical stage reads live
    # instance settings or invokes a scalar conversion protocol.
    var fields = py_call(py_type(Python.dict()).__dict__["copy"], state)
    py_setitem(fields, "_pls_name", PythonObject(_name(model)))
    py_setitem(fields, "_pls_regression", PythonObject(decomp_is_a(model, context["pls_regression_class"])))
    py_setitem(fields, "_pls_canonical", PythonObject(decomp_is_a(model, context["pls_canonical_class"])))
    var fitting = operation == "fit" or operation == "fit_transform"
    var names = Python.tuple("_mojolearn_fitted", "_mojolearn_fitting", "_predict_1d", "n_features_in_")
    if fitting:
        names = Python.tuple("n_components", "scale", "max_iter", "tol", "copy", "_mojolearn_fitting")
    for index in range(len(names)):  # driver-loop: operation-specific scalar metadata snapshot
        if py_contains(fields, names[index]):
            py_setitem(fields, names[index], _pls_scalar(fields[names[index]], String(names[index]), context))
    if not fitting:
        # Fitted detection must precede synthesized default-mode metadata.
        _pls_require_fitted(fields, operation, context)
    # Optional, not a PythonObject(None) sentinel: Mojo 1.2.0 fails to lower a loop-carried PythonObject read and reassigned in the loop (modular/modular#7546).
    var default_mode = Optional[PythonObject]()
    var modes = Python.tuple("numeric_mode") if fitting else (Python.tuple("numeric_mode", "numeric_mode_") if operation == "score" else Python.tuple("numeric_mode_"))
    for index in range(len(modes)):  # driver-loop: current/fitted modes actually consumed by this operation
        var mode = py_get(fields, modes[index])
        if mode is None:
            if not default_mode:
                default_mode = context["backend_fields"]["_DEFAULT_MODE"]
            mode = default_mode.value()
        if py_type(mode) is not b.str:
            raise Error("numeric_mode must be fast, identical or None")
        py_setitem(fields, modes[index], mode)
    if fitting:
        var algorithm = py_get(fields, "algorithm", PythonObject("nipals"))
        if py_type(algorithm) is not b.str:
            raise Error("PLSCanonical: algorithm must be 'nipals' or 'svd'")
        py_setitem(fields, "algorithm", algorithm)
        for key in ["_deflation", "_pmode"]:  # small-loop(fixed class algorithm constants)
            var value = decomp_class_attr(model, key)
            if py_type(value) is not b.str:
                raise Error("TypeError: PLS algorithm constants must remain native strings")
            py_setitem(fields, key, value)
    return fields


def _pls_require_fitted(fields: PythonObject, operation: String, context: PythonObject) raises:
    var fitted = Bool(py=py_get(fields, "_mojolearn_fitted", PythonObject(False))) or Bool(py=py_get(fields, "_mojolearn_fitting", PythonObject(False)))
    if not fitted:
        var keys = py_call(native_loaded_module("builtins").list, fields)
        for index in range(len(keys)):  # driver-loop: existing fitted-public-attribute compatibility rule
            var key = String(keys[index])
            if key.endswith("_") and not key.startswith("_") and fields[key] is not None:
                fitted = True
                break
    if not fitted:
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: This " + String(fields["_pls_name"]) + " instance is not fitted yet. Call 'fit' with appropriate arguments before using " + operation + ".")


def _pls_input(source: PythonObject, name: String, context: PythonObject, mut pin: ArrayPin, ndim: Int = 2,
               public_name: String = "", public_target: Bool = False, sparse_allowed: Bool = True) raises -> ArrayMetadata:
    var deferred = False
    return _pls_input_deferred(source, name, context, pin, ndim, public_name, public_target, sparse_allowed, False,
                               deferred)


def _pls_input_deferred(source: PythonObject, name: String, context: PythonObject, mut pin: ArrayPin, ndim: Int,
                        public_name: String,
                        public_target: Bool, sparse_allowed: Bool, defer: Bool, mut deferred: Bool) raises -> ArrayMetadata:
    """A PLS operand through the one admission (bindings/admitted_input.mojo):
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
    # `x_pin`: the door's pin of `captured_x`.
    if captured_x:
        return _captured(door, kit, captured_x.value(), x_pin.value(), x_deferred, String(fields["_pls_name"]))
    return _matrix(door, kit, args[0], "X", context)


def _pls_defers(fields: PythonObject, fitting: Bool, context: PythonObject) -> Bool:
    """Whether the operation's kit tests X on its upload (the x_decomp GPU
    binding's kernel-call op XD_DEV_UPLOAD_FINITE, lane a8c): then the door admits a
    borrowed X with no host pass. False when the entries cannot be resolved
    here; the kit then raises its own refusal as before."""
    try:
        var mode = fields["numeric_mode"] if fitting else py_get(fields, "numeric_mode_")
        var entries = decomp_entries(mode, context)
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


def _pls_store(fields: PythonObject, state: PythonObject, key: String, value: PythonObject) raises:
    # Keep this operation's generated values stable even if another caller
    # mutates the exposed model while a later native numerical call runs.
    py_setitem(fields, key, value)
    py_setitem(state, key, value)


def _pls_fit(model: PythonObject, x_source: PythonObject, y_source: PythonObject, transform: Bool,
             context: PythonObject, fields: PythonObject, state: PythonObject,
             captured_x: Optional[ArrayMetadata] = None, captured_y: Optional[ArrayMetadata] = None,
             x_deferred: Bool = False, x_pin: Optional[ArrayPin] = None,
             y_pin: Optional[ArrayPin] = None) raises -> PythonObject:
    # `_PLS.fit` reads the captured settings and publishes into separate
    # state: one UNPACK, one RUN (bindings/pls_run.mojo), one PACK.
    # `x_pin`/`y_pin` are the door's pins of `captured_x`/`captured_y`.
    var entries = decomp_entries(fields["numeric_mode"], context)
    var door = DecompDoor(entries, context["output_context"])
    var kit = decomp_kit_unpack(entries)
    try:
        var x = _x(door, kit, Python.tuple(x_source), captured_x, x_pin, x_deferred, fields, context)
        var target = _target(door, kit, y_source, context, captured_y, y_pin)
        var ym = target[0]
        if ym < 0:
            raise Error(String(fields["_pls_name"]) + " requires y to be passed, but the target y is None.")
        var n = kit.nrows(x)
        var p = kit.ncols(x)
        var q = kit.ncols(ym)
        if kit.nrows(ym) != n:
            raise Error("Found input variables with inconsistent numbers of samples: [" + String(n) + ", "
                        + String(kit.nrows(ym)) + "]")
        if n < 2:
            raise Error("Found array with " + String(n) + " sample(s) (shape=(" + String(n) + ", " + String(p)
                        + ")) while a minimum of 2 is required.")
        var nc = decomp_int(fields["n_components"])
        var deflation = fields["_deflation"]
        if py_type(deflation) is not py_type(PythonObject("")):
            raise Error("TypeError: PLS deflation must remain a native string")
        var canonical = String(py=deflation) == "canonical"
        var bound = min(n, min(p, q)) if canonical else p
        if not (1 <= nc and nc <= bound):
            raise Error("n_components == " + String(nc) + ", while 1 <= n_components <= " + String(bound) + " is required")
        _pls_store(fields, state, "_predict_1d", PythonObject(target[1]))
        var scale = Bool(py=fields["scale"])
        var algorithm = py_get(fields, "algorithm", PythonObject("nipals"))
        var b = native_loaded_module("builtins")
        var svd_algo = Bool(py=py_call(b.isinstance, algorithm, b.str)) and String(py=algorithm) == "svd"
        var pmode = fields["_pmode"]
        if py_type(pmode) is not py_type(PythonObject("")):
            raise Error("TypeError: PLS component mode must remain a native string")
        var words = PlsFitWords(nc, decomp_int(fields["max_iter"]), String(py=pmode) == "B", canonical, svd_algo,
                                scale, decomp_float(fields["tol"]))
        # RUN: the centering, the one component-loop call, the rotations,
        # the coefficients (and fit_transform's projections), the GIL released.
        var fit = PlsFit()
        var py = Python()
        with GILReleased(py):
            fit = pls_fit_run(kit, x, ym, words, transform)
        # PACK: the warning, the iteration counts and the fitted fields, once.
        if fit.done < nc:
            metric_warn("y residual is constant at iteration " + String(fit.done), b.UserWarning, context)
        var all_iterations = Python.list()
        for index in range(len(fit.iterations)):  # driver-loop: publish completed component iteration counts
            _ = py_call(all_iterations.append, PythonObject(Int(fit.iterations[index])))
        _pls_store(fields, state, "n_iter_", all_iterations)
        _pls_store(fields, state, "x_weights_", decomp_kit_publish(door, kit, fit.outs[0]))
        _pls_store(fields, state, "y_weights_", decomp_kit_publish(door, kit, fit.outs[1]))
        _pls_store(fields, state, "x_loadings_", decomp_kit_publish(door, kit, fit.outs[4]))
        _pls_store(fields, state, "y_loadings_", decomp_kit_publish(door, kit, fit.outs[5]))
        _pls_store(fields, state, "x_rotations_", decomp_kit_publish(door, kit, fit.x_rotations))
        _pls_store(fields, state, "y_rotations_", decomp_kit_publish(door, kit, fit.y_rotations))
        _pls_store(fields, state, "coef_", decomp_kit_publish(door, kit, fit.coef))
        _pls_store(fields, state, "intercept_", decomp_kit_out_shape(door, kit, fit.y_mean, Python.tuple(q)))
        _pls_store(fields, state, "_x_scores", decomp_kit_publish(door, kit, fit.outs[2]))
        _pls_store(fields, state, "_y_scores", decomp_kit_publish(door, kit, fit.outs[3]))
        _pls_store(fields, state, "_x_mean", decomp_kit_out_shape(door, kit, fit.x_mean, Python.tuple(p)))
        _pls_store(fields, state, "_x_std", decomp_kit_out_shape(door, kit, fit.x_std, Python.tuple(p)))
        _pls_store(fields, state, "_y_mean", decomp_kit_out_shape(door, kit, fit.y_mean, Python.tuple(q)))
        _pls_store(fields, state, "_y_std", decomp_kit_out_shape(door, kit, fit.y_std, Python.tuple(q)))
        _pls_store(fields, state, "n_features_in_", PythonObject(p))
        if Bool(py=fields["_pls_regression"]):
            _pls_store(fields, state, "x_scores_", fields["_x_scores"])
            _pls_store(fields, state, "y_scores_", fields["_y_scores"])
        if not transform:
            return model
        # fit_transform: the projections of the X and Y the fit held.
        return Python.tuple(decomp_kit_publish(door, kit, fit.tx), decomp_kit_publish(door, kit, fit.ty))
    finally:
        kit.close()
        door.keep_alive()  # the borrowed owners outlive every RUN read


def _check_fitted(model: PythonObject, fields: PythonObject) raises:
    if not py_contains(fields, "coef_"):
        raise Error("RuntimeError: " + String(fields["_pls_name"]) + " is not fitted; call fit first")


def _score_binding(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    # `_metrics_impl._get_binding` belongs to this whole public operation.
    # Use the import-connected provider and preserve its metrics-specific
    # mode refusals/readback before invoking the existing native metric owner.
    var mode = py_get(fields, "numeric_mode")
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
        mode = context["backend_fields"]["_DEFAULT_MODE"]
        if py_type(mode) is not b.str:
            raise Error("numeric_mode must be 'fast' or 'identical'")
        expected = String(py=py_call(py_call(mode.strip).lower))
    var module = estimator_binding_mode(mode, "_mojolearn_metrics", context, False)
    var reader = neural_descriptor_entry(module, "metrics_numeric_mode", False)
    if reader is None:
        reader = neural_descriptor_entry(module, "umap_numeric_mode", False)
    if reader is not None:
        var word = Int(py=py_call(reader))
        var actual = String("fast") if word == 0 else (String("identical") if word == 1 else String("None"))
        if actual != expected:
            raise Error("RuntimeError: mojolearn metrics: requested " + expected + ", binary reports " + actual + "; rebuild it")
    return module


def pls_operation(model: PythonObject, operation: String, args: PythonObject, context: PythonObject,
                  fields: PythonObject, state: PythonObject, captured_x: Optional[ArrayMetadata] = None,
                  captured_y: Optional[ArrayMetadata] = None, x_deferred: Bool = False,
                  x_pin: Optional[ArrayPin] = None, y_pin: Optional[ArrayPin] = None) raises -> PythonObject:
    if operation == "fit" or operation == "fit_transform":
        # args: (X, y, Y); y wins over Y, as PLSRegression.fit chose.
        var y = args[1] if args[1] is not None else args[2]
        if Bool(py=fields["_pls_canonical"]):
            var algorithm = fields["algorithm"]
            var b = native_loaded_module("builtins")
            var is_str = Bool(py=py_call(b.isinstance, algorithm, b.str))
            if not is_str or (String(py=algorithm) != "nipals" and String(py=algorithm) != "svd"):
                raise Error("PLSCanonical: algorithm must be 'nipals' or 'svd'")
        _pls_store(fields, state, "numeric_mode_", fields["numeric_mode"])
        return _pls_fit(model, args[0], y, operation == "fit_transform", context, fields, state, captured_x,
                        captured_y, x_deferred, x_pin, y_pin)
    if operation == "score":
        # `RegressorScoreMixin.score`: y as float32 (`as_f32_c(y, ndim=None,
        # name="y")`), the native predict, then `metrics.r2_score(y, pred,
        # sample_weight=..., numeric_mode=...)` through the metrics owner
        # (`family_mcls_r2`, the entry r2_score calls, with its arguments);
        # args: (X, y, sample_weight); binding selection remains native.
        var owner = py_get(context, "pls_metric_owner")
        if owner is None:
            raise Error("NotImplementedError: mojolearn: the PLS score needs the native metrics owner "
                        + "(family_mcls_r2); this install does not carry it")
        var binding = _score_binding(fields, owner[1])
        var truth = admit_f32(args[1], -1, "y", context, finite=False)
        var prediction = pls_operation(model, "predict", Python.tuple(args[0]), context, fields, state, captured_x,
                                       None, x_deferred, x_pin)
        var mode = py_get(fields, "numeric_mode")
        var r2 = metrics_cls_operation(PythonObject(None), "family_mcls_r2",
                         Python.tuple(binding, truth.array, prediction, args[2], PythonObject("uniform_average"),
                                      PythonObject(True), mode), owner[1])
        _ = truth^
        return r2
    _check_fitted(model, fields)
    var entries = decomp_entries(py_get(fields, "numeric_mode_"), context)
    var door = DecompDoor(entries, context["output_context"])
    # A query: the fitted operands stay on the device across calls.
    door.owner = model
    var kit = decomp_kit_unpack(entries)
    try:
        var py = Python()
        if operation == "transform":
            # args: (X, y, Y). UNPACK: X, Y, then the fitted state they use.
            var y = args[1] if args[1] is not None else args[2]
            var x = _x(door, kit, args, captured_x, x_pin, x_deferred, fields, context)
            var ym = _target(door, kit, y, context)[0]
            var x_mean = _fitted_row(door, kit, fields, "_x_mean", context)
            var x_std = _fitted_row(door, kit, fields, "_x_std", context)
            var x_rot = _fitted_matrix(door, kit, fields, "x_rotations_", context)
            var y_mean = -1
            var y_std = -1
            var y_rot = -1
            if ym >= 0:
                y_mean = _fitted_row(door, kit, fields, "_y_mean", context)
                y_std = _fitted_row(door, kit, fields, "_y_std", context)
                y_rot = _fitted_matrix(door, kit, fields, "y_rotations_", context)
            var xs = -1
            var ys = -1
            with GILReleased(py):
                xs = pls_project(kit, x, x_mean, x_std, x_rot)
                if ym >= 0:
                    ys = pls_project(kit, ym, y_mean, y_std, y_rot)
            if ym < 0:
                return decomp_kit_publish(door, kit, xs)
            return Python.tuple(decomp_kit_publish(door, kit, xs), decomp_kit_publish(door, kit, ys))
        if operation == "inverse_transform":
            var z = _x(door, kit, args, captured_x, x_pin, x_deferred, fields, context)
            var x_load = _fitted_matrix(door, kit, fields, "x_loadings_", context)
            var x_std = _fitted_row(door, kit, fields, "_x_std", context)
            var x_mean = _fitted_row(door, kit, fields, "_x_mean", context)
            var y = args[1] if args[1] is not None else args[2]
            var w = -1
            var y_load = -1
            var y_std = -1
            var y_mean = -1
            if y is not None:
                w = _matrix(door, kit, y, "Y", context)
                y_load = _fitted_matrix(door, kit, fields, "y_loadings_", context)
                y_std = _fitted_row(door, kit, fields, "_y_std", context)
                y_mean = _fitted_row(door, kit, fields, "_y_mean", context)
            var xr = -1
            var yr = -1
            with GILReleased(py):
                xr = pls_rebuild(kit, z, x_load, x_std, x_mean)
                if w >= 0:
                    yr = pls_rebuild(kit, w, y_load, y_std, y_mean)
            if w < 0:
                return decomp_kit_publish(door, kit, xr)
            return Python.tuple(decomp_kit_publish(door, kit, xr), decomp_kit_publish(door, kit, yr))
        if operation == "predict":
            var x = _x(door, kit, args, captured_x, x_pin, x_deferred, fields, context)
            var mean = _fitted_row(door, kit, fields, "_x_mean", context)
            var coef = _fitted_matrix(door, kit, fields, "coef_", context)
            var y_mean = _fitted_row(door, kit, fields, "_y_mean", context)
            var prediction = -1
            with GILReleased(py):
                prediction = pls_predict_run(kit, x, mean, coef, y_mean)
            if Bool(py=fields["_predict_1d"]):
                return decomp_kit_out_shape(door, kit, prediction, Python.tuple(kit.nrows(prediction)))
            return decomp_kit_publish(door, kit, prediction)
        raise Error("unknown native PLS operation")
    finally:
        kit.close()
        door.keep_alive()  # the borrowed owners outlive every RUN read



def pls_public_operation(model: PythonObject, operation: String, args: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    # Own the former _track_fit/_needs_fit/_public_output envelope. Import
    # metadata and shared layout admission prevent Python descriptors before
    # inspecting model settings; caller inputs are admitted before metadata.
    var state = classical_family_admit(model)
    var fields = _pls_settings(model, state, operation, context)
    var b = native_loaded_module("builtins")
    var fitting = operation == "fit" or operation == "fit_transform"
    var name = String(fields["_pls_name"])
    var sparse = Bool(py=fields["_pls_regression"])
    # Lane a8c: the X of an operation whose kit tests its upload is admitted
    # with no host pass (`x_deferred`; decomp_door_tested refuses it).
    var x_deferred = False
    var x_pin = ArrayPin()
    var x = _pls_input_deferred(args[0], "X", context, x_pin, 2, name, False, sparse,
                                _pls_defers(fields, fitting, context), x_deferred)
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
    elif operation != "inverse_transform":
        var features = py_get(fields, "n_features_in_")
        if features is not None:
            if py_type(features) is not b.int:
                raise Error("TypeError: native PLS feature count must be an exact integer")
            classical_check_width(x.shape[1], Int(py=features), name)
        estimator_feature_check(model, fields, args[0], context)
    var feature_names = _feature_names(args[0], context) if fitting else PythonObject(None)
    var result = PythonObject(None)
    if fitting:
        var depth_value = py_get(fields, "_mojolearn_fitting", PythonObject(0))
        if py_type(depth_value) is not b.int:
            raise Error("TypeError: native fit depth must be an exact integer")
        var depth = Int(py=depth_value)
        py_setitem(state, "_mojolearn_fitting", PythonObject(depth + 1))
        try:
            result = pls_operation(model, operation, args, context, fields, state, x.copy(), y, x_deferred,
                                   Optional(x_pin.copy()), y_pin)
        except error:
            # _sk_validate.reraise annotated failed one-row/one-feature fits.
            var message = String(error)
            var device_fault = message.find("device_context") >= 0 or message.find("CUDA") >= 0 or message.find("HIP") >= 0 or message.find("hipError") >= 0 or message.find("Metal") >= 0 or message.find("MTL") >= 0
            if not device_fault and not message.startswith("TypeError:") and not message.startswith("RuntimeError:") and not message.startswith("NotImplementedError:") and not message.startswith("ImportError:") and message != PYTHON_ERROR_PENDING and not message.startswith("Input X contains NaN or infinity"):
                if x.shape[0] == 1:
                    raise Error(name + " cannot be fit on n_samples=1: " + message)
                if x.shape[1] == 1:
                    raise Error(name + " cannot be fit on n_features=1: " + message)
            raise error^
        finally:
            if depth > 0:
                py_setitem(state, "_mojolearn_fitting", PythonObject(depth))
            else:
                _ = py_call(state.pop, "_mojolearn_fitting", PythonObject(None))
        if feature_names is None:
            _ = py_call(state.pop, "feature_names_in_", PythonObject(None))
        else:
            py_setitem(state, "feature_names_in_", feature_names)
        py_setitem(state, "_mojolearn_fitted", PythonObject(True))
        if operation == "fit":
            return result
    else:
        result = pls_operation(model, operation, args, context, fields, state, x.copy(), None, x_deferred,
                               Optional(x_pin.copy()))
    _ = x_pin^  # the door's pin held X through the operation (the kit holds its own owners)
    return parallel_public_result(result, operation, PythonObject("regressor"), context)


def pls_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        if len(call) != 4:
            raise Error("TypeError: pls_native_call requires model, operation, arguments and connected context")
        var b = native_loaded_module("builtins")
        if py_type(call[1]) is not b.str or py_type(call[2]) is not b.tuple or py_type(call[3]) is not b.dict:
            raise Error("TypeError: native PLS requires an operation, an argument tuple and its context")
        var context = call[3]
        classical_connect_numpy(context)
        var result = pls_public_operation(call[0], String(py=call[1]), call[2], context)
        return result.steal_data()
    except error:
        return classical_python_error(error^, args_ptr)

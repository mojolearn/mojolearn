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
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.classical_error_boundary import classical_python_error
from bindings.classical_validate import classical_check_width
from bindings.parallel_public_result import parallel_public_result
from bindings.metrics_native_support import metric_warn
from providers.python.estimator_lifecycle_pyapi import estimator_feature_check
from providers.python.array_metadata_pyapi import ArrayMetadata
from bindings.classical_arrays import classical_address, classical_host_f32
from bindings.classical_labels import labels_f32
from bindings.metrics_cls_native import metrics_cls_operation
from providers.python.estimator_binding_pyapi import estimator_binding_mode
from providers.python.classical_family_admission_pyapi import classical_family_admit
from bindings.native_scalar_decode import native_scalar_buffer
from bindings.native_array_storage import native_array_zeros
from bindings.sequential_admission import _one_of
from bindings.classifier_public_lifecycle import _feature_names
from providers.python.decomp_input_pyapi import decomp_input_snapshot
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry
from bindings.decomp_kit_native import (
    DecompKit, decomp_int, decomp_float, decomp_entries, decomp_is_a,
    decomp_class_attr, XD_SUB, XD_ADD, XD_MUL, XD_DIV, XD_SCALE, XD_SQ, XD_SQRT, XD_SELECT, XD_RECIP,
)

comptime PLS_F32_EPS = 1.1920928955078125e-07


def _name(model: PythonObject) raises -> String:
    return String(py=py_type(model).__name__)



def _pls_scalar(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
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
    var b = Python.import_module("builtins")
    # Exact-dict copy captures references atomically while admission holds the
    # GIL. Keep model writes separate; no later numerical stage reads live
    # instance settings or invokes a scalar conversion protocol.
    var fields = py_call(b.dict.copy, state)
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
    var default_mode = PythonObject(None)
    var modes = Python.tuple("numeric_mode") if fitting else (Python.tuple("numeric_mode", "numeric_mode_") if operation == "score" else Python.tuple("numeric_mode_"))
    for index in range(len(modes)):  # driver-loop: current/fitted modes actually consumed by this operation
        var mode = py_get(fields, modes[index])
        if mode is None:
            if default_mode is None:
                default_mode = context["backend_fields"]["_DEFAULT_MODE"]
            mode = default_mode
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
        var keys = py_call(Python.import_module("builtins").list, fields)
        for index in range(len(keys)):  # driver-loop: existing fitted-public-attribute compatibility rule
            var key = String(keys[index])
            if key.endswith("_") and not key.startswith("_") and fields[key] is not None:
                fitted = True
                break
    if not fitted:
        _ = classical_not_fitted_class(context)
        raise Error("NotFittedError: This " + String(fields["_pls_name"]) + " instance is not fitted yet. Call 'fit' with appropriate arguments before using " + operation + ".")


def _matrix(mut kit: DecompKit, source: PythonObject, name: String, context: PythonObject) raises -> Int:
    var input = decomp_input_snapshot(source, name, context)
    return kit.input_snapshot(input, input.shape[0], input.shape[1])


def _row(mut kit: DecompKit, source: PythonObject, name: String, context: PythonObject) raises -> Int:
    # A fitted 1-D float32 vector as a 1 x n matrix.
    var input = decomp_input_snapshot(source, name, context, 1)
    return kit.input_snapshot(input, 1, input.shape[0])


def _target(mut kit: DecompKit, source: PythonObject, context: PythonObject,
            captured: Optional[ArrayMetadata] = None) raises -> Tuple[Int, Bool]:
    # Admit the exact supported input before reading any metadata. A vector
    # keeps the old row-then-transpose execution order, using captured shape
    # and address instead of source.shape, source.reshape or __array__ hooks.
    if source is None:
        return (Int(-1), False)
    var input = captured.value().copy() if captured else decomp_input_snapshot(source, "Y", context, -1)
    if len(input.shape) == 2:
        return (kit.input_snapshot(input, input.shape[0], input.shape[1]), False)
    var row = kit.input_snapshot(input, 1, input.shape[0])
    return (kit.transpose(row), True)


def _center_scale(mut kit: DecompKit, a: Int, scale: Bool) raises -> Tuple[Int, Int, Int]:
    # `_PLS._center_scale`: (centered and scaled A, mean, std).
    var n = kit.nrows(a)
    var mean = kit.colmean(a)
    var centered = kit.ew(XD_SUB, a, mean)
    var std: Int
    if scale:
        var squares = kit.ew(XD_SQ, centered)
        var sums = kit.colsum(squares)
        var variance = kit.ew(XD_SCALE, sums, scalar=1.0 / Float64(n - 1))
        var root = kit.ew(XD_SQRT, variance)
        var one = kit.const(1.0)
        std = kit.ew(XD_SELECT, root, root, one, 0.0)
        centered = kit.ew(XD_DIV, centered, std)
    else:
        std = kit.const(1.0, 1, kit.ncols(a))
    return (centered, mean, std)


def _pinv(mut kit: DecompKit, a: Int) raises -> Int:
    # `_pinv` (rcond None): V diag(1/s) U^T through the thin SVD, values at or
    # below max(shape) * float32 eps * s_max dropped.
    var rows = kit.nrows(a)
    var cols = kit.ncols(a)
    var r = min(rows, cols)
    var usv = kit.thin_svd(a, r, True)
    var cond = Float64(max(rows, cols)) * PLS_F32_EPS
    var cut = Float64(0.0)
    if r:
        cut = Float64(Float32(kit.word(usv[1], 0) * cond))
    var zero = kit.zeros(1, 1)
    var kept = kit.ew(XD_SELECT, usv[1], usv[1], zero, cut)
    var inverse = kit.ew(XD_RECIP, kept)
    var scaled = kit.ew(XD_MUL, usv[0], inverse)
    return kit.mm(usv[2], scaled, ta=True, tb=True)


def _rotation(mut kit: DecompKit, weights: Int, loadings: Int) raises -> Int:
    var gram = kit.mm(loadings, weights, ta=True)
    var inverse = _pinv(kit, gram)
    return kit.mm(weights, inverse)


def _pls_store(fields: PythonObject, state: PythonObject, key: String, value: PythonObject) raises:
    # Keep this operation's generated values stable even if another caller
    # mutates the exposed model while a later native numerical call runs.
    py_setitem(fields, key, value)
    py_setitem(state, key, value)


def _pls_fit(model: PythonObject, x_source: PythonObject, y_source: PythonObject, mut kit: DecompKit,
             context: PythonObject, fields: PythonObject, state: PythonObject,
             captured_x: Optional[ArrayMetadata] = None, captured_y: Optional[ArrayMetadata] = None) raises -> Tuple[Int, Int]:
    # `_PLS.fit` reads the captured settings and publishes into separate state.
    var x = kit.input_snapshot(captured_x.value(), captured_x.value().shape[0], captured_x.value().shape[1]) if captured_x else _matrix(kit, x_source, "X", context)
    var target = _target(kit, y_source, context, captured_y)
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
    var xs = _center_scale(kit, x, scale)
    var ys = _center_scale(kit, ym, scale)
    var algorithm = py_get(fields, "algorithm", PythonObject("nipals"))
    var b = Python.import_module("builtins")
    var svd_algo = Bool(py=py_call(b.isinstance, algorithm, b.str)) and String(py=algorithm) == "svd"
    var pmode = fields["_pmode"]
    if py_type(pmode) is not py_type(PythonObject("")):
        raise Error("TypeError: PLS component mode must remain a native string")
    var mode_b = String(py=pmode) == "B"
    # The component loop (NIPALS or the SVD pair, the flips, the deflations)
    # is ONE binding call writing six host blocks.
    var sizes = [p, q, n, n, p, q]
    var outs = List[Int]()
    var addresses = Python.list()
    for k in range(6):  # driver-loop: over the six fitted blocks
        var block = kit.zeros(sizes[k], nc)
        outs.append(block)
        _ = py_call(addresses.append, kit.address(block))
        _ = block  # keep the Python object alive through the native call above
    var its = native_array_zeros("i", nc)
    var iterations_address = Int(py=py_call(its.buffer_info)[0])
    var ax = kit.address(xs[0])
    var ay = kit.address(ys[0])
    var result = py_call(kit.entries["x_decomp_pls_fit"], ax, ay, addresses, PythonObject(iterations_address),
                         Python.list(n, p, q, nc, decomp_int(fields["max_iter"]), 1 if mode_b else 0,
                                     Int(canonical), Int(canonical), Int(svd_algo)),
                         Python.list(decomp_float(fields["tol"])))
    var done = Int(py=result[0])
    var nits = Int(py=result[1])
    for k in range(6):  # driver-loop: over the six fitted blocks
        kit.wrote_host(outs[k])
    if done < nc:
        metric_warn("y residual is constant at iteration " + String(done), b.UserWarning, context)
        for k in range(6):  # driver-loop: over the six fitted blocks
            outs[k] = kit.cols_of(outs[k], 0, done)
    var all_iterations = Python.list()
    # Publish the completed native Int32 counts without Python iterable or
    # slicing work. Match the old list[:nits] bound for the driver's count.
    var stop = max(nc + nits, 0) if nits < 0 else min(nits, nc)
    var iterations = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=iterations_address)
    for index in range(stop):  # driver-loop: publish completed component iteration counts
        _ = py_call(all_iterations.append, PythonObject(Int(iterations[index])))
    _pls_store(fields, state, "n_iter_", all_iterations)
    _ = its
    var x_rotations = _rotation(kit, outs[0], outs[4])
    var y_rotations = _rotation(kit, outs[1], outs[5])
    var coef = kit.mm(x_rotations, outs[5], tb=True)
    var scaled = kit.ew(XD_MUL, coef, ys[2])
    var x_std_t = kit.transpose(xs[2])
    var divided = kit.ew(XD_DIV, scaled, x_std_t)
    coef = kit.transpose(divided)
    _pls_store(fields, state, "x_weights_", kit.publish(outs[0]))
    _pls_store(fields, state, "y_weights_", kit.publish(outs[1]))
    _pls_store(fields, state, "x_loadings_", kit.publish(outs[4]))
    _pls_store(fields, state, "y_loadings_", kit.publish(outs[5]))
    _pls_store(fields, state, "x_rotations_", kit.publish(x_rotations))
    _pls_store(fields, state, "y_rotations_", kit.publish(y_rotations))
    _pls_store(fields, state, "coef_", kit.publish(coef))
    _pls_store(fields, state, "intercept_", kit.out_shape(ys[1], Python.tuple(q)))
    _pls_store(fields, state, "_x_scores", kit.publish(outs[2]))
    _pls_store(fields, state, "_y_scores", kit.publish(outs[3]))
    _pls_store(fields, state, "_x_mean", kit.out_shape(xs[1], Python.tuple(p)))
    _pls_store(fields, state, "_x_std", kit.out_shape(xs[2], Python.tuple(p)))
    _pls_store(fields, state, "_y_mean", kit.out_shape(ys[1], Python.tuple(q)))
    _pls_store(fields, state, "_y_std", kit.out_shape(ys[2], Python.tuple(q)))
    _pls_store(fields, state, "n_features_in_", PythonObject(p))
    if Bool(py=fields["_pls_regression"]):
        _pls_store(fields, state, "x_scores_", fields["_x_scores"])
        _pls_store(fields, state, "y_scores_", fields["_y_scores"])
    return (x, ym)


def _check_fitted(model: PythonObject, fields: PythonObject) raises:
    if not py_contains(fields, "coef_"):
        raise Error("RuntimeError: " + String(fields["_pls_name"]) + " is not fitted; call fit first")


def _project(mut kit: DecompKit, a: Int, mean_name: String, std_name: String, rotations_name: String,
             fields: PythonObject, context: PythonObject) raises -> Int:
    # ((A - mean) / std) @ rotations.
    var mean = _row(kit, fields[mean_name], mean_name, context)
    var std = _row(kit, fields[std_name], std_name, context)
    var rotations = _matrix(kit, fields[rotations_name], rotations_name, context)
    var centered = kit.ew(XD_SUB, a, mean)
    var scaled = kit.ew(XD_DIV, centered, std)
    return kit.mm(scaled, rotations)


def _transform(mut kit: DecompKit, x: Int, ym: Int, fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var xs = _project(kit, x, "_x_mean", "_x_std", "x_rotations_", fields, context)
    if ym < 0:
        return kit.publish(xs)
    var ys = _project(kit, ym, "_y_mean", "_y_std", "y_rotations_", fields, context)
    return Python.tuple(kit.publish(xs), kit.publish(ys))


def _rebuild(mut kit: DecompKit, z: Int, loadings_name: String, std_name: String, mean_name: String,
             fields: PythonObject, context: PythonObject) raises -> Int:
    # (Z @ loadings^T) * std + mean.
    var loadings = _matrix(kit, fields[loadings_name], loadings_name, context)
    var std = _row(kit, fields[std_name], std_name, context)
    var mean = _row(kit, fields[mean_name], mean_name, context)
    var product = kit.mm(z, loadings, tb=True)
    var scaled = kit.ew(XD_MUL, product, std)
    return kit.ew(XD_ADD, scaled, mean)



def _score_binding(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    # `_metrics_impl._get_binding` belongs to this whole public operation.
    # Use the import-connected provider and preserve its metrics-specific
    # mode refusals/readback before invoking the existing native metric owner.
    var mode = py_get(fields, "numeric_mode")
    var b = Python.import_module("builtins")
    if mode is not None:
        if py_type(mode) is not b.str:
            raise Error("numeric_mode must be 'fast' or 'identical'")
        var selected = String(py=py_call(py_call(mode.strip).lower))
        if selected != "fast" and selected != "identical":
            raise Error("numeric_mode must be 'fast' or 'identical'")
    else:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not b.str:
        raise Error("numeric_mode must be 'fast' or 'identical'")
    var expected = String(py=py_call(py_call(mode.strip).lower))
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
                  captured_y: Optional[ArrayMetadata] = None) raises -> PythonObject:
    if operation == "fit" or operation == "fit_transform":
        # args: (X, y, Y); y wins over Y, as PLSRegression.fit chose.
        var y = args[1] if args[1] is not None else args[2]
        if Bool(py=fields["_pls_canonical"]):
            var algorithm = fields["algorithm"]
            var b = Python.import_module("builtins")
            var is_str = Bool(py=py_call(b.isinstance, algorithm, b.str))
            if not is_str or (String(py=algorithm) != "nipals" and String(py=algorithm) != "svd"):
                raise Error("PLSCanonical: algorithm must be 'nipals' or 'svd'")
        var mode = fields["numeric_mode"]
        _pls_store(fields, state, "numeric_mode_", mode)
        var kit = DecompKit(decomp_entries(mode, context), context["output_context"])
        try:
            var held = _pls_fit(model, args[0], y, kit, context, fields, state, captured_x, captured_y)
            if operation == "fit":
                return model
            # transform(X, y): the X and Y the fit converted and uploaded.
            return _transform(kit, held[0], held[1], fields, context)
        finally:
            kit.close()
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
        var truth = labels_f32(args[1], -1, "C", "y", context)
        var prediction = pls_operation(model, "predict", Python.tuple(args[0]), context, fields, state, captured_x)
        var mode = py_get(fields, "numeric_mode")
        var r2 = metrics_cls_operation(PythonObject(None), "family_mcls_r2",
                         Python.tuple(binding, truth.keep, prediction, args[2], PythonObject("uniform_average"),
                                      PythonObject(True), mode), owner[1])
        _ = truth^
        return r2
    _check_fitted(model, fields)
    var kit = DecompKit(decomp_entries(py_get(fields, "numeric_mode_"), context), context["output_context"])
    try:
        if operation == "transform":
            # args: (X, y, Y)
            var y = args[1] if args[1] is not None else args[2]
            var x = kit.input_snapshot(captured_x.value(), captured_x.value().shape[0], captured_x.value().shape[1]) if captured_x else _matrix(kit, args[0], "X", context)
            var target = _target(kit, y, context)
            return _transform(kit, x, target[0], fields, context)
        if operation == "inverse_transform":
            var z = kit.input_snapshot(captured_x.value(), captured_x.value().shape[0], captured_x.value().shape[1]) if captured_x else _matrix(kit, args[0], "X", context)
            var xr = _rebuild(kit, z, "x_loadings_", "_x_std", "_x_mean", fields, context)
            var y = args[1] if args[1] is not None else args[2]
            if y is None:
                return kit.publish(xr)
            var w = _matrix(kit, y, "Y", context)
            var yr = _rebuild(kit, w, "y_loadings_", "_y_std", "_y_mean", fields, context)
            return Python.tuple(kit.publish(xr), kit.publish(yr))
        if operation == "predict":
            var x = kit.input_snapshot(captured_x.value(), captured_x.value().shape[0], captured_x.value().shape[1]) if captured_x else _matrix(kit, args[0], "X", context)
            var mean = _row(kit, fields["_x_mean"], "_x_mean", context)
            var centered = kit.ew(XD_SUB, x, mean)
            var coef = _matrix(kit, fields["coef_"], "coef_", context)
            var product = kit.mm(centered, coef, tb=True)
            var y_mean = _row(kit, fields["_y_mean"], "_y_mean", context)
            var prediction = kit.ew(XD_ADD, product, y_mean)
            if Bool(py=fields["_predict_1d"]):
                return kit.out_shape(prediction, Python.tuple(kit.nrows(prediction)))
            return kit.publish(prediction)
        raise Error("unknown native PLS operation")
    finally:
        kit.close()



def pls_public_operation(model: PythonObject, operation: String, args: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    # Own the former _track_fit/_needs_fit/_public_output envelope. Import
    # metadata and shared layout admission prevent Python descriptors before
    # inspecting model settings; caller inputs are admitted before metadata.
    var state = classical_family_admit(model)
    var fields = _pls_settings(model, state, operation, context)
    var b = Python.import_module("builtins")
    var fitting = operation == "fit" or operation == "fit_transform"
    var name = String(fields["_pls_name"])
    var sparse = Bool(py=fields["_pls_regression"])
    var x = decomp_input_snapshot(args[0], "X", context, 2, name, False, sparse)
    var y = Optional[ArrayMetadata]()
    if fitting:
        var target = args[1] if args[1] is not None else args[2]
        if target is None:
            raise Error(name + " requires y to be passed, but the target y is None.")
        y = decomp_input_snapshot(target, "Y", context, -1, name, True)
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
            result = pls_operation(model, operation, args, context, fields, state, x, y)
        except error:
            # _sk_validate.reraise annotated failed one-row/one-feature fits.
            var message = String(error)
            var device_fault = message.find("device_context") >= 0 or message.find("CUDA") >= 0 or message.find("HIP") >= 0 or message.find("hipError") >= 0 or message.find("Metal") >= 0 or message.find("MTL") >= 0
            if not device_fault and not message.startswith("TypeError:") and not message.startswith("RuntimeError:") and not message.startswith("NotImplementedError:") and not message.startswith("ImportError:") and message != PYTHON_ERROR_PENDING:
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
        result = pls_operation(model, operation, args, context, fields, state, x)
    return parallel_public_result(result, operation, PythonObject("regressor"), context)


def pls_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        if len(call) != 4:
            raise Error("TypeError: pls_native_call requires model, operation, arguments and connected context")
        var b = Python.import_module("builtins")
        if py_type(call[1]) is not b.str or py_type(call[2]) is not b.tuple or py_type(call[3]) is not b.dict:
            raise Error("TypeError: native PLS requires an operation, an argument tuple and its context")
        var context = call[3]
        classical_connect_numpy(context)
        var result = pls_public_operation(call[0], String(py=call[1]), call[2], context)
        return result.steal_data()
    except error:
        return classical_python_error(error^, args_ptr)

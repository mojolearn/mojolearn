# SPDX-License-Identifier: Apache-2.0
# The native TruncatedSVD owner: every public TruncatedSVD operation (fit,
# fit_transform, transform, inverse_transform) is one call into tsvd_c. It
# checks the parameters and converts X once, selects the numeric-mode
# bindings, runs the route the Python shell ran with the same compiled
# entries and arguments in the same order (so the same bits), and stores the
# fitted attributes under the names, types and shapes the shell stored:
#   algorithm='randomized'   x_decomp_tsvd_randomized (x_decomp/tsvd_owner.mojo),
#                            the steps `_randomized_decompose` sequenced
#   FAST TSVD_QFIX, tall X   x_decomp_tsvd_tsqr, the steps `_tsvd_tsqr_components`
#                            sequenced, then tsvd_explained
#   otherwise (the Gram)     tsvd_fit_explained, or tsvd_fit then tsvd_explained
# fit_transform projects the X the fit converted (tsvd_transform), with no
# second conversion and no Python between the two.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_get, py_type, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_address, classical_typed, classical_host_f32
from bindings.decomp_inputs import decomp_input, decomp_copied
from bindings.public_lifecycle_owner import lifecycle_begin, lifecycle_end, lifecycle_complete, lifecycle_result

#: x_decomp/tsqr_core.mojo TS_MAX_N (`_linalg_impl._TS_MAX_N`): the widest X the TSQR takes
comptime _TS_MAX_N = 512
#: x_decomp/qfix.mojo TSVD_QFIX's bit in `x_decomp_qfix_flags`
comptime _QFIX_TSVD = 2


def _repr(value: PythonObject) raises -> String:
    return String(py_call(Python.import_module("builtins").repr, value))


def _admitted(model: PythonObject, context: PythonObject) raises -> Bool:
    # TruncatedSVD itself or a subclass: its MRO, read through type's own
    # descriptor, holds the connected class.
    var b = Python.import_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        return False
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    for k in range(len(mro)):  # driver-loop: over the model class's MRO
        if mro[k] is context["tsvd_class"]:
            return True
    return False


def _host_only(model: PythonObject) raises -> Bool:
    # An explicit host inference class (`_HOST_INFERENCE_ONLY` in its class
    # dictionaries) transforms a saved model; it never fits.
    var b = Python.import_module("builtins")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(model))
    for k in range(len(mro)):  # driver-loop: over the model class's MRO
        var class_fields = py_call(b.type.__dict__["__dict__"].__get__, mro[k])
        if py_get(class_fields, "_HOST_INFERENCE_ONLY") is not None:
            return True
    return False


def _attr(model: PythonObject, fields: PythonObject, name: String) raises -> PythonObject:
    """`model.<name>` for an instance field, with Python's AttributeError."""
    if not py_contains(fields, name):
        raise Error("AttributeError: '" + String(py_type(model).__name__) + "' object has no attribute '" + name + "'")
    return fields[name]


def _int(value: PythonObject) raises -> Int:
    return Int(py=py_call(Python.import_module("builtins").int, value))


def _mode(fields: PythonObject, context: PythonObject) raises -> String:
    # The estimator's numeric mode (None: the process mode), as
    # `_backend.binding` resolves it.
    var b = Python.import_module("builtins")
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    var name = String(py_call(py_call(mode.strip).lower))
    if name != "identical" and name != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    return name


def _entries(mode: String, context: PythonObject) raises -> PythonObject:
    # The connected _mojolearn_estimators entries of `mode`, checked against
    # the selected vendor and mode.
    var entries = context["tsvd_bindings"][PythonObject(mode)]
    if py_type(entries) is Python.import_module("builtins").str:
        raise Error(String(entries))
    if String(entries["estimators_vendor"]()) != String(context["vendor"]):
        raise Error("RuntimeError: TruncatedSVD native vendor differs from selected execution")
    if Int(py=entries["estimators_numeric_mode"]()) != (1 if mode == "identical" else 0):
        raise Error("RuntimeError: TruncatedSVD native mode differs from selected execution")
    return entries


def _decomp(mode: String, context: PythonObject) raises -> PythonObject:
    """The connected _mojolearn_x_decomp entries of `mode` (a dict), or the
    connection's refusal (a str)."""
    return context["tsvd_decomp"][PythonObject(mode)]


def _too_small(rows: Int, cols: Int) -> String:
    # scikit-learn's check_array wording for a block below 2 x 2.
    if rows < 2:
        return ("Found array with " + String(rows) + " sample(s) (shape=(" + String(rows) + ", " + String(cols)
                + ")) while a minimum of 2 is required by TruncatedSVD.")
    return ("Found array with " + String(cols) + " feature(s) (shape=(" + String(rows) + ", " + String(cols)
            + ")) while a minimum of 2 is required by TruncatedSVD.")


def _components(fields: PythonObject, rows: Int, cols: Int) raises -> Int:
    # `_component_count`: n_components (None: the smaller dimension) in [1, cols].
    var requested = fields["n_components"]
    var value = min(rows, cols) if requested is None else _int(requested)
    if value < 1 or value > cols:
        raise Error("mojolearn n_components must be in [1, " + String(cols) + "], got " + String(value))
    return value


def _f32(shape: PythonObject, context: PythonObject) raises -> PythonObject:
    # A zero-filled C-order float32 mojolearn Array (`_buffer.empty`).
    return classical_typed(shape, "<f4", context["output_context"])


def _algorithm(fields: PythonObject) raises -> String:
    """The algorithm name ('' for a non-str value, which no arm accepts)."""
    var value = fields["algorithm"]
    if py_type(value) is Python.import_module("builtins").str:
        return String(value)
    return ""


def _randomized(model: PythonObject, fields: PythonObject, x: PythonObject, rows: Int, cols: Int, nc: Int,
                mode: String, context: PythonObject) raises:
    """algorithm='randomized': `_randomized_decompose(x, nc, center=False,
    ...)` as one x_decomp call, its arrays stored as the shell stored them."""
    var b = Python.import_module("builtins")
    var decomp = _decomp(mode, context)
    if py_type(decomp) is b.str:
        raise Error(String(decomp))
    var normalizer = fields["power_iteration_normalizer"]
    var name = String(normalizer) if py_type(normalizer) is b.str else String("")
    if name != "auto" and name != "QR" and name != "LU" and name != "none":
        raise Error("power_iteration_normalizer must be 'auto', 'QR', 'LU' or 'none'")
    var n_iter = fields["n_iter"]
    var auto = py_type(n_iter) is b.str and String(n_iter) == "auto"
    var iterations = 0 if auto else _int(n_iter)
    var oversamples = _int(fields["n_oversamples"])
    var seed = fields["random_state"]
    if seed is None:
        seed = PythonObject(0)
    elif py_type(seed) is b.bool or not Bool(py=py_call(b.isinstance, seed, b.int)):
        raise Error("TypeError: random_state must be None or an int (the counter-based RNG takes an integer seed)")
    if py_get(decomp, "x_decomp_tsvd_randomized") is None:
        raise Error("NotImplementedError: mojolearn TruncatedSVD: this build of _mojolearn_x_decomp does not export "
                    + "x_decomp_tsvd_randomized; rebuild it with bindings/build_x_decomp.sh")
    # The Philox seed is taken mod 2^32, with Python integer semantics
    # (`_Kit.rand`'s `int(seed) & 0xFFFFFFFF`).
    var word = py_call(b.int.__dict__["__and__"], py_call(b.int, seed), PythonObject(0xFFFFFFFF))
    var components = _f32(Python.tuple(nc, cols), context)
    var singular = _f32(Python.tuple(nc), context)
    var variance = _f32(Python.tuple(nc), context)
    var ratio = _f32(Python.tuple(nc), context)
    _ = py_call_typed(decomp["x_decomp_tsvd_randomized"], classical_address(x), classical_address(components),
                      classical_address(singular), classical_address(variance), classical_address(ratio),
                      Python.list(rows, cols, nc, oversamples, iterations, 1 if auto else 0, word))
    _ = components  # keep the Python object alive through the native call above
    _ = ratio  # keep the Python object alive through the native call above
    _ = variance  # keep the Python object alive through the native call above
    _ = singular  # keep the Python object alive through the native call above
    py_setitem(fields, "components_", components)
    py_setitem(fields, "singular_values_", singular)
    py_setitem(fields, "explained_variance_", variance)
    py_setitem(fields, "explained_variance_ratio_", ratio)


def _tsqr_route(decomp: PythonObject, rows: Int, cols: Int, nc: Int) raises -> Bool:
    """`_tsvd_tsqr_components`'s gate: the binding carries TSVD_QFIX (FAST;
    0 on an IDENTICAL or host binding, or when the kit is unavailable) and
    X is a TSQR shape (`_tsqr_on`: tall, at most _TS_MAX_N wide)."""
    if py_type(decomp) is Python.import_module("builtins").str:
        return False
    var flags_entry = py_get(decomp, "x_decomp_qfix_flags")
    if flags_entry is None:
        return False
    # `_Kit.qfix_flags`: a binding that cannot answer counts as 0
    var flags = 0
    try:
        flags = Int(py=py_call(flags_entry))
    except ignored:
        flags = 0
    if (flags & _QFIX_TSVD) == 0:
        return False
    return cols >= 1 and cols <= _TS_MAX_N and rows >= cols and nc >= 1 and nc <= cols


def _exact(model: PythonObject, fields: PythonObject, x: PythonObject, rows: Int, cols: Int, nc: Int,
           mode: String, entries: PythonObject, context: PythonObject) raises:
    """The exact arm: the TSVD_QFIX components where the shell took them,
    else the Gram; then the explained variance, in this class's binding."""
    var decomp = _decomp(mode, context)
    var variance = _f32(Python.tuple(nc), context)
    var ratio = _f32(Python.tuple(nc), context)
    py_setitem(fields, "explained_variance_", variance)
    py_setitem(fields, "explained_variance_ratio_", ratio)
    var params = Python.list(rows, cols, nc)
    var components = _f32(Python.tuple(nc, cols), context)
    var singular = _f32(Python.tuple(nc), context)
    if _tsqr_route(decomp, rows, cols, nc):
        if py_get(decomp, "x_decomp_tsvd_tsqr") is None:
            raise Error("NotImplementedError: mojolearn TruncatedSVD: this build of _mojolearn_x_decomp does not export "
                        + "x_decomp_tsvd_tsqr; rebuild it with bindings/build_x_decomp.sh")
        _ = py_call_typed(decomp["x_decomp_tsvd_tsqr"], classical_address(x), classical_address(components),
                          classical_address(singular), params)
        _ = components  # keep the Python object alive through the native call above
        _ = singular  # keep the Python object alive through the native call above
        py_setitem(fields, "components_", components)
        py_setitem(fields, "singular_values_", singular)
    elif py_get(entries, "tsvd_fit_explained") is not None:
        # lane fg-pca T1: the Gram fit and its explained variance in one
        # binding call (one upload of X)
        py_setitem(fields, "components_", components)
        py_setitem(fields, "singular_values_", singular)
        _ = py_call_typed(entries["tsvd_fit_explained"], classical_address(x), classical_address(components),
                          classical_address(singular), classical_address(variance), classical_address(ratio),
                          params)
        _ = components  # keep the Python object alive through the native call above
        _ = singular  # keep the Python object alive through the native call above
        _ = ratio  # keep the Python object alive through the native call above
        _ = variance  # keep the Python object alive through the native call above
        return
    else:
        py_setitem(fields, "components_", components)
        py_setitem(fields, "singular_values_", singular)
        _ = py_call_typed(entries["tsvd_fit"], classical_address(x), classical_address(components),
                          classical_address(singular), params)
        _ = components  # keep the Python object alive through the native call above
        _ = singular  # keep the Python object alive through the native call above
    _ = py_call_typed(entries["tsvd_explained"], classical_address(x), classical_address(components),
                      classical_address(variance), classical_address(ratio), params)
    _ = components  # keep the Python object alive through the native call above
    _ = ratio  # keep the Python object alive through the native call above
    _ = variance  # keep the Python object alive through the native call above


def tsvd_fit(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`TruncatedSVD.fit`; returns the converted X (kept for fit_transform)."""
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    if Float64(py=py_call(b.float, _attr(model, fields, "tol"))) != 0.0:
        raise Error("NotImplementedError: mojolearn TruncatedSVD: tol is ARPACK's convergence tolerance "
                    + "and algorithm='arpack' is not implemented; leave tol=0.0")
    var algorithm = _algorithm(fields)
    if algorithm != "randomized" and algorithm != "covariance_eigh" and algorithm != "jacobi":
        raise Error("NotImplementedError: mojolearn TruncatedSVD: algorithm=" + _repr(fields["algorithm"])
                    + " is not implemented; this class runs ('covariance_eigh', 'jacobi', 'randomized'). "
                    + "'arpack' is scipy's svds (implicitly restarted Lanczos), a different algorithm; "
                    + "accepting the name and running the Gram arm would be a silent substitution, which is "
                    + "why this raises (decomposition/NOT_IMPLEMENTED.tsv)")
    var host = _host_only(model)
    var x = decomp_input(source, "TruncatedSVD", context)
    py_setitem(fields, "input_copied_", PythonObject(decomp_copied(source, context)))
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    if rows < 2 or cols < 2:
        raise Error(_too_small(rows, cols))
    var nc = _components(fields, rows, cols)
    var mode = _mode(fields, context)
    # the host class trains on the CPU: the estimators host binding of its
    # `_bind` serves the same entry names (CPU TRAINING IS PUBLIC)
    var entries = (py_call(Python.import_module("builtins").vars, py_call(model._bind, "_mojolearn_estimators"))
                   if host else _entries(mode, context))
    if algorithm == "randomized":
        _randomized(model, fields, x, rows, cols, nc, mode, context)
    else:
        _exact(model, fields, x, rows, cols, nc, mode, entries, context)
    py_setitem(fields, "n_components_", PythonObject(nc))
    py_setitem(fields, "n_features_in_", PythonObject(cols))
    return x


def _project(model: PythonObject, fields: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`transform` of a converted X: X V^T through tsvd_transform."""
    var rows = Int(py=x.shape[0])
    var cols = Int(py=x.shape[1])
    if cols != _int(_attr(model, fields, "n_features_in_")):
        raise Error("mojolearn TruncatedSVD feature count differs from fit")
    var nc = _int(_attr(model, fields, "n_components_"))
    var out = _f32(Python.tuple(rows, nc), context)
    var entries = _entries(_mode(fields, context), context)
    var components = classical_host_f32(fields["components_"], 2, "components_", context)
    _ = py_call_typed(entries["tsvd_transform"], classical_address(x), classical_address(components),
                      classical_address(out), Python.list(rows, cols, nc))
    _ = components  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = components^
    return out


def tsvd_transform(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if not py_contains(fields, "components_"):
        raise Error("mojolearn TruncatedSVD: call fit before transform")
    var x = decomp_input(source, "TruncatedSVD", context)
    var out = _project(model, fields, x, context)
    _ = x^
    return out


def tsvd_inverse_transform(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """X @ components_ (`inverse_transform` without the mean: its pointer is
    the components, a valid float32 address the entry does not read)."""
    var fields = py_fields(model)
    var z = decomp_input(source, "TruncatedSVD", context)
    var rows = Int(py=z.shape[0])
    var nc = _int(_attr(model, fields, "n_components_"))
    if Int(py=z.shape[1]) != nc:
        raise Error("mojolearn TruncatedSVD component count differs from fit")
    var nf = _int(_attr(model, fields, "n_features_in_"))
    var out = _f32(Python.tuple(rows, nf), context)
    var entries = _entries(_mode(fields, context), context)
    var components = classical_host_f32(_attr(model, fields, "components_"), 2, "components_", context)
    _ = py_call_typed(entries["inverse_transform"], classical_address(z), classical_address(components),
                      classical_address(components), classical_address(out), Python.list(rows, nf, nc, 0))
    _ = z  # keep the Python object alive through the native call above
    _ = components  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = components^
    _ = z^
    return out


def tsvd_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        if len(call) != 4:
            raise Error("TypeError: tsvd_native_call requires model, operation, arguments and connected context")
        var model = call[0]
        var b = Python.import_module("builtins")
        if py_type(call[1]) is not b.str or py_type(call[2]) is not b.tuple:
            raise Error("TypeError: native TruncatedSVD operation and argument tuple are required")
        var operation = String(call[1])
        var args = call[2]
        var context = call[3]
        if py_type(context) is not b.dict:
            raise Error("TypeError: native TruncatedSVD requires connected context")
        if not _admitted(model, context):
            raise Error("TypeError: the native TruncatedSVD owner requires a TruncatedSVD instance")
        classical_connect_numpy(context)
        if operation != "fit" and operation != "fit_transform" and operation != "transform" and operation != "inverse_transform":
            raise Error("unknown private TruncatedSVD operation")
        # The whole public lifecycle (the former `_mode` wrappers): fitted
        # check (NotFittedError), X refusals, width and feature-name checks,
        # fit bookkeeping, NumPy results for callers outside the package. A
        # scipy.sparse X is densified by decomp_input (documented).
        var depth = lifecycle_begin(model, operation, args[0], True, True, True, context)
        var result = PythonObject(None)
        try:
            if operation == "fit":
                _ = tsvd_fit(model, args[0], context)
                result = model
            elif operation == "fit_transform":
                # the transform projects the X the fit converted: one conversion
                var x = tsvd_fit(model, args[0], context)
                result = _project(model, py_fields(model), x, context)
                _ = x^
            elif operation == "transform":
                result = tsvd_transform(model, args[0], context)
            else:
                result = tsvd_inverse_transform(model, args[0], context)
        finally:
            lifecycle_end(model, operation, depth)
        lifecycle_complete(model, operation, args[0], False, context)
        result = lifecycle_result(model, operation, result, "", context)
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)

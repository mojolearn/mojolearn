# SPDX-License-Identifier: Apache-2.0
# The native TruncatedSVD owner: every public TruncatedSVD operation (fit,
# fit_transform, transform, inverse_transform) is one call into tsvd_c. It
# checks the parameters and converts X once, selects the numeric-mode
# bindings, runs the route the Python shell ran with the same native
# kernels and typed requests in the same order, and stores the
# fitted attributes under the names, types and shapes the shell stored:
#   algorithm='randomized'   x_decomp_tsvd_randomized (x_decomp/tsvd_owner.mojo),
#                            the steps `_randomized_decompose` sequenced
#   FAST TSVD_QFIX, tall X   x_decomp_tsvd_tsqr, the steps `_tsvd_tsqr_components`
#                            sequenced, then tsvd_explained
#   otherwise (the Gram)     tsvd_fit_explained, or tsvd_fit then tsvd_explained
# fit_transform projects the X the fit converted (tsvd_transform), with no
# second conversion and no Python between the two.
from std.python import Python, PythonObject
from bindings.cpython_table import cpython_table
from bindings.native_loaded_module import native_loaded_module, NativeModules
from bindings.native_class_admission import class_record, class_admit
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_get, py_type, py_fields, py_setitem, py_contains
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_typed
from bindings.native_buffer_address import native_buffer_address
from bindings.admitted_input import admit_x, AdmittedX, admitted_finite_refuse
from bindings.device_rows_door import door_rows_refuse
from bindings.decomp_inputs import decomp_admit, decomp_input_copied
from bindings.public_lifecycle_owner import lifecycle_begin, lifecycle_end, lifecycle_complete, lifecycle_result
from bindings.decomp_scalar_admission import decomp_scalar_admit, decomp_scalar_repr
from core.pca_native_api import PCAFoldStore
from core.tsvd_native_api import (
    TSVDFitRequest, TSVDQueryRequest, TSVD_GRAM, TSVD_FUSED, TSVD_EXPLAINED,
    TSVD_RANDOMIZED, TSVD_TSQR, TSVD_FOLD, TSVD_CAP_FUSED, TSVD_CAP_RANDOMIZED,
    TSVD_CAP_TSQR, TSVD_CAP_RESIDENT,
)
from bindings.tsvd_descriptor_boundary import (
    tsvd_factory, tsvd_connect, tsvd_connection, tsvd_capabilities, tsvd_qfix, tsvd_run_fit, tsvd_run_query,
)
from bindings.tsvd_owner_boundary import tsvd_owner_drop, tsvd_owner_lease, TSVDLeasePin

#: x_decomp/tsqr_core.mojo TS_MAX_N (`_linalg_impl._TS_MAX_N`): the widest X the TSQR takes
comptime _TS_MAX_N = 512
#: x_decomp/qfix.mojo TSVD_QFIX's bit in `x_decomp_qfix_flags`
comptime _QFIX_TSVD = 2


def _admitted(model: PythonObject, context: PythonObject) raises -> Bool:
    # TruncatedSVD itself or a subclass: its MRO, read through type's own
    # descriptor, holds the connected class.
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        return False
    var target = context["tsvd_class"]
    if py_type(target) is not b.type:
        return False
    # CPython's own subtype test walks the same tp_mro by identity in one
    # native call (no MRO tuple fetch or per-row Python indexing).
    # Resolved once per process in the shared table (pass 2), not by name per call.
    var t = cpython_table()
    return t[].PyType_IsSubtype(cls._obj_ptr, target._obj_ptr) != 0


def _host_only(model: PythonObject) raises -> Bool:
    # An explicit host inference class (`_HOST_INFERENCE_ONLY` in its class
    # dictionaries) transforms a saved model; it never fits.
    # Depends only on the class rows: resolved once per class state
    # (native_class_admission re-checks after any class modification).
    var mods = NativeModules()
    var cls = py_type(model)
    var record = class_record(mods, _HOST_ONLY_SLOT, cls)
    if record is not None:
        return Bool(py=record)
    var b = native_loaded_module("builtins")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    var found = False
    for k in range(len(mro)):  # driver-loop: over the model class's MRO
        var class_fields = py_call(b.type.__dict__["__dict__"].__get__, mro[k])
        if py_get(class_fields, "_HOST_INFERENCE_ONLY") is not None:
            found = True
            break
    class_admit(mods, _HOST_ONLY_SLOT, cls, PythonObject(found))
    return found


comptime _HOST_ONLY_SLOT = "tsvd_host_inference_only_v1"


def _attr(model: PythonObject, fields: PythonObject, name: String) raises -> PythonObject:
    """`model.<name>` for an instance field, with Python's AttributeError."""
    if not py_contains(fields, name):
        raise Error("AttributeError: '" + String(py_type(model).__name__) + "' object has no attribute '" + name + "'")
    return fields[name]


def _int(value: PythonObject, context: PythonObject) raises -> Int:
    decomp_scalar_admit(value, context)
    return Int(py=py_call(native_loaded_module("builtins").int, value))


def _mode(fields: PythonObject, context: PythonObject) raises -> String:
    # The estimator's numeric mode (None: the process mode), as
    # `_backend.binding` resolves it.
    var b = native_loaded_module("builtins")
    var mode = py_get(fields, "numeric_mode")
    if mode is None:
        mode = context["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    # The exact spellings answer natively (two Python calls per use before;
    # p3 rank 833); any other text is normalized as before.
    var text = String(py=mode)
    if text == "identical" or text == "fast":
        return text
    var name = String(py_call(py_call(mode.strip).lower))
    if name != "identical" and name != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    return name


def _host_connection(model: PythonObject, fields: PythonObject, context: PythonObject) raises -> Int:
    # Keep _HostBound._bind's late mode refusal without calling that Python
    # method or loading a module during fit. Import setup pins the CPU image.
    var mode = py_get(fields, "numeric_mode")
    if mode is not None and (py_type(mode) is not py_type(PythonObject("")) or String(mode) != "identical"):
        var type_type = native_loaded_module("builtins").type
        var name = String(py_call(type_type.__dict__["__name__"].__get__, py_type(model)))
        raise Error("mojolearn: " + name + " runs IDENTICAL only on the host; this model was saved " + decomp_scalar_repr(mode, context))
    return tsvd_connect(context["tsvd_host_factory"], 1, "cpu")


def _decomp(mode: String, context: PythonObject) raises -> PythonObject:
    return tsvd_factory(context, "tsvd_decomp", mode)


def _too_small(rows: Int, cols: Int) -> String:
    # scikit-learn's check_array wording for a block below 2 x 2.
    if rows < 2:
        return ("Found array with " + String(rows) + " sample(s) (shape=(" + String(rows) + ", " + String(cols)
                + ")) while a minimum of 2 is required by TruncatedSVD.")
    return ("Found array with " + String(cols) + " feature(s) (shape=(" + String(rows) + ", " + String(cols)
            + ")) while a minimum of 2 is required by TruncatedSVD.")


def _components(fields: PythonObject, rows: Int, cols: Int, context: PythonObject) raises -> Int:
    # `_component_count`: n_components (None: the smaller dimension) in [1, cols].
    var requested = fields["n_components"]
    var value = min(rows, cols) if requested is None else _int(requested, context)
    if value < 1 or value > cols:
        raise Error("mojolearn n_components must be in [1, " + String(cols) + "], got " + String(value))
    return value


def _address(value: PythonObject, context: PythonObject) raises -> Int:
    # Only admitted native arrays reach this UNPACK helper. Array owns its
    # address slot; other admitted storage exposes the native buffer directly.
    if py_type(value) is context["array_class"]:
        return Int(py=value._addr)
    return native_buffer_address(value)


def _f32(shape: PythonObject, context: PythonObject) raises -> PythonObject:
    # A zero-filled C-order float32 mojolearn Array (`_buffer.empty`).
    return classical_typed(shape, "<f4", context["output_context"])


def _algorithm(fields: PythonObject) raises -> String:
    """The algorithm name ('' for a non-str value, which no arm accepts)."""
    var value = fields["algorithm"]
    if py_type(value) is native_loaded_module("builtins").str:
        return String(value)
    return ""


def _randomized(model: PythonObject, fields: PythonObject, x: PythonObject, rows: Int, cols: Int, nc: Int,
                mode: String, context: PythonObject, test: Bool) raises:
    """algorithm='randomized': `_randomized_decompose(x, nc, center=False,
    ...)` as one x_decomp call, its arrays stored as the shell stored them."""
    var b = native_loaded_module("builtins")
    var decomp = _decomp(mode, context)
    if py_type(decomp) is b.str:
        raise Error(String(decomp))
    var normalizer = fields["power_iteration_normalizer"]
    var name = String(normalizer) if py_type(normalizer) is b.str else String("")
    if name != "auto" and name != "QR" and name != "LU" and name != "none":
        raise Error("power_iteration_normalizer must be 'auto', 'QR', 'LU' or 'none'")
    var connection = tsvd_connect(decomp, 1 if mode == "identical" else 0, String(context["vendor"]))
    var n_iter = fields["n_iter"]
    var auto = py_type(n_iter) is b.str and String(n_iter) == "auto"
    var iterations = 0 if auto else _int(n_iter, context)
    var oversamples = _int(fields["n_oversamples"], context)
    var seed = fields["random_state"]
    if seed is None:
        seed = PythonObject(0)
    elif py_type(seed) is not b.int:
        raise Error("TypeError: random_state must be None or an int (the counter-based RNG takes an integer seed)")
    if (tsvd_capabilities(connection) & TSVD_CAP_RANDOMIZED) == 0:
        raise Error("NotImplementedError: mojolearn TruncatedSVD: this build of _mojolearn_x_decomp does not export "
                    + "x_decomp_tsvd_randomized; rebuild it with bindings/build_x_decomp.sh")
    # The Philox seed is taken mod 2^32, with Python integer semantics
    # (`_Kit.rand`'s `int(seed) & 0xFFFFFFFF`).
    var word = py_call(b.int.__dict__["__and__"], seed, PythonObject(0xFFFFFFFF))
    var components = _f32(Python.tuple(nc, cols), context)
    var singular = _f32(Python.tuple(nc), context)
    var variance = _f32(Python.tuple(nc), context)
    var ratio = _f32(Python.tuple(nc), context)
    # Lane a8d: `test` appends the finite flag; the driver returns X's
    # finite status word, taken off the variance total it computes anyway.
    var request = TSVDFitRequest(_address(x, context), _address(components, context), _address(singular, context),
        _address(variance, context), _address(ratio, context), rows, cols, nc, oversamples, iterations,
        auto, Int(py=word), test, TSVD_RANDOMIZED, PCAFoldStore())
    var result = tsvd_run_fit(connection, request)
    if test:
        admitted_finite_refuse(result.finite, _TSVD_EST)
    _ = components  # keep the Python object alive through the native call above
    _ = ratio  # keep the Python object alive through the native call above
    _ = variance  # keep the Python object alive through the native call above
    _ = singular  # keep the Python object alive through the native call above
    py_setitem(fields, "components_", components)
    py_setitem(fields, "singular_values_", singular)
    py_setitem(fields, "explained_variance_", variance)
    py_setitem(fields, "explained_variance_ratio_", ratio)


def _tsqr_connection(decomp: PythonObject, mode: String, context: PythonObject,
                     rows: Int, cols: Int, nc: Int) raises -> Int:
    # The old kit treats an unavailable flag entry as zero. Keep that refusal
    # order: an unavailable decomposition image does not disable the Gram fit.
    if py_type(decomp) is native_loaded_module("builtins").str:
        return -1
    var connection = -1
    var flags = 0
    try:
        connection = tsvd_connect(decomp, 1 if mode == "identical" else 0, String(context["vendor"]))
        flags = tsvd_qfix(connection)
    except ignored:
        return -1
    if (flags & _QFIX_TSVD) == 0:
        return -1
    if cols >= 1 and cols <= _TS_MAX_N and rows >= cols and nc >= 1 and nc <= cols:
        return connection
    return -1


comptime _TSVD_EST = "TruncatedSVD"


# TOMBSTONE: _tsvd_scan (the parallel host scan of a deferred X before the TSQR and randomized drivers) deleted 2026-11-09 by lane a8d-device-x-decomp-manifold; code recoverable at 0ce88411d3.
# Both drivers now return X's finite status word over their own upload
# (x_decomp/tsvd_owner.mojo `finite_word`), as the Gram entries do.


def _exact(model: PythonObject, fields: PythonObject, admitted: AdmittedX, rows: Int, cols: Int, nc: Int,
           mode: String, connection: Int, context: PythonObject) raises:
    """Existing TSQR/Gram/fused route, preserving each public publication stage."""
    var x = admitted.array
    var test = not admitted.finite
    var decomp = _decomp(mode, context)
    var variance = _f32(Python.tuple(nc), context)
    var ratio = _f32(Python.tuple(nc), context)
    py_setitem(fields, "explained_variance_", variance)
    py_setitem(fields, "explained_variance_ratio_", ratio)
    var components = _f32(Python.tuple(nc, cols), context)
    var singular = _f32(Python.tuple(nc), context)
    var request = TSVDFitRequest(_address(x, context), _address(components, context), _address(singular, context),
        _address(variance, context), _address(ratio, context), rows, cols, nc, 0, 0, False, 0, test,
        TSVD_GRAM, PCAFoldStore())
    var tsqr = _tsqr_connection(decomp, mode, context, rows, cols, nc)
    if tsqr >= 0:
        if (tsvd_capabilities(tsqr) & TSVD_CAP_TSQR) == 0:
            raise Error("NotImplementedError: mojolearn TruncatedSVD: this build of _mojolearn_x_decomp does not export "
                        + "x_decomp_tsvd_tsqr; rebuild it with bindings/build_x_decomp.sh")
        request.route = TSVD_TSQR
        var result = tsvd_run_fit(tsqr, request)
        if test:
            admitted_finite_refuse(result.finite, _TSVD_EST)
        py_setitem(fields, "components_", components)
        py_setitem(fields, "singular_values_", singular)
    else:
        py_setitem(fields, "components_", components)
        py_setitem(fields, "singular_values_", singular)
        var fused = (tsvd_capabilities(connection) & TSVD_CAP_FUSED) != 0
        request.route = TSVD_FUSED if fused else TSVD_GRAM
        var result = tsvd_run_fit(connection, request)
        if test:
            admitted_finite_refuse(result.finite, _TSVD_EST)
        if fused:
            _ = x
            _ = components
            _ = singular
            _ = variance
            _ = ratio
            return
    request.route = TSVD_EXPLAINED
    request.test_finite = False
    _ = tsvd_run_fit(connection, request)
    _ = x
    _ = components
    _ = singular
    _ = variance
    _ = ratio


def tsvd_fit(model: PythonObject, source: PythonObject, context: PythonObject, fold: Bool = False) raises -> AdmittedX:
    """Return the native admitted view/dimensions for fit_transform reuse.
    Fit-only resident folds retain their native store descriptor, with no host X.
    """
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    # New components replace the old: their device copy goes now (N351).
    tsvd_owner_drop(model)
    var tolerance = _attr(model, fields, "tol")
    decomp_scalar_admit(tolerance, context)
    if Float64(py=py_call(b.float, tolerance)) != 0.0:
        raise Error("NotImplementedError: mojolearn TruncatedSVD: tol is ARPACK's convergence tolerance "
                    + "and algorithm='arpack' is not implemented; leave tol=0.0")
    var algorithm = _algorithm(fields)
    if algorithm != "randomized" and algorithm != "covariance_eigh" and algorithm != "jacobi":
        raise Error("NotImplementedError: mojolearn TruncatedSVD: algorithm=" + decomp_scalar_repr(fields["algorithm"], context)
                    + " is not implemented; this class runs ('covariance_eigh', 'jacobi', 'randomized'). "
                    + "'arpack' is scipy's svds (implicitly restarted Lanczos), a different algorithm; "
                    + "accepting the name and running the Gram arm would be a silent substitution, which is "
                    + "why this raises (decomposition/NOT_IMPLEMENTED.tsv)")
    # A SciPy sparse X (TruncatedSVD's main input) is densified once, and the
    # input_copied_ flag is read off the admission (no second view of X).
    # Lane a7d: a borrowed X is not scanned here (defer_finite); the Gram
    # entries test it over their own upload; the TSQR and randomized drivers
    # return its status word too (lane a8d).
    # Lane a8d: an identical exact (Gram) fit of a cross-validation fold
    # resident on the device is gathered there by the base binding's
    # `tsvd_fit_fold` (no host gather, no upload of X); the TSQR route is
    # `fast` only and the randomized driver uploads X itself, so those (and
    # the host class) get the fold's host rows.
    var host = _host_only(model)
    var mode = _mode(fields, context)
    var fold_entry = py_get(context, "tsvd_fold_factory")
    var fold_ok = fold and fold_entry is not None and not host and mode == "identical" and algorithm != "randomized"
    var admitted = decomp_admit(source, "TruncatedSVD", context, device_x=fold_ok, defer_finite=True)
    py_setitem(fields, "input_copied_", PythonObject(decomp_input_copied(admitted, context)))
    var rows = admitted.rows
    var cols = admitted.cols
    if rows < 2 or cols < 2:
        raise Error(_too_small(rows, cols))
    var nc = _components(fields, rows, cols, context)
    if admitted.store.active():
        var components = _f32(Python.tuple(nc, cols), context)
        var singular = _f32(Python.tuple(nc), context)
        var variance = _f32(Python.tuple(nc), context)
        var ratio = _f32(Python.tuple(nc), context)
        var connection = tsvd_connect(fold_entry, 1, String(context["vendor"]))
        var request = TSVDFitRequest(0, _address(components, context), _address(singular, context),
            _address(variance, context), _address(ratio, context), rows, cols, nc, 0, 0, False, 0, True,
            TSVD_FOLD, PCAFoldStore(admitted.store.base_id, admitted.store.base_rows, admitted.store.index_id))
        var words = tsvd_run_fit(connection, request)
        _ = components
        _ = singular
        _ = variance
        _ = ratio
        door_rows_refuse(words.finite, words.in_range, _TSVD_EST, admitted.store.base_rows)
        py_setitem(fields, "components_", components)
        py_setitem(fields, "singular_values_", singular)
        py_setitem(fields, "explained_variance_", variance)
        py_setitem(fields, "explained_variance_ratio_", ratio)
        py_setitem(fields, "n_components_", PythonObject(nc))
        py_setitem(fields, "n_features_in_", PythonObject(cols))
        return admitted^
    var x = admitted.array
    var connection = _host_connection(model, fields, context) if host else tsvd_connection(mode, context)
    if algorithm == "randomized":
        _randomized(model, fields, x, rows, cols, nc, mode, context, not admitted.finite)
    else:
        _exact(model, fields, admitted, rows, cols, nc, mode, connection, context)
    py_setitem(fields, "n_components_", PythonObject(nc))
    py_setitem(fields, "n_features_in_", PythonObject(cols))
    return admitted^


def _resident_lease(model: PythonObject, connection: Int, stored: PythonObject,
                    components: PythonObject) raises -> TSVDLeasePin:
    # Keep the existing read-only/owning-array rule and resident fallback.
    # The native lease owner pins the image and component storage through RUN.
    if (tsvd_capabilities(connection) & TSVD_CAP_RESIDENT) == 0:
        return TSVDLeasePin(PythonObject(None), 0)
    if components is not stored:
        tsvd_owner_drop(model)
        return TSVDLeasePin(PythonObject(None), 0)
    return tsvd_owner_lease(model, connection, stored)


def _project(model: PythonObject, fields: PythonObject, admitted: AdmittedX, context: PythonObject,
             test: Bool = False) raises -> PythonObject:
    """`transform` of a converted X: X V^T through tsvd_transform. `test`
    (lane a7d): X was admitted with `defer_finite` and not scanned; the
    entry tests it over its own upload and returns the status word."""
    var x = admitted.array
    var rows = admitted.rows
    var cols = admitted.cols
    if cols != _int(_attr(model, fields, "n_features_in_"), context):
        raise Error("mojolearn TruncatedSVD feature count differs from fit")
    var nc = _int(_attr(model, fields, "n_components_"), context)
    var out = _f32(Python.tuple(rows, nc), context)
    var mode = _mode(fields, context)
    var connection = tsvd_connection(mode, context)
    var stored = fields["components_"]
    var components = admit_x(stored, 2, "components_", context)
    var lease = _resident_lease(model, connection, stored, components)
    var request = TSVDQueryRequest(admitted.address, _address(components, context), _address(out, context),
        rows, cols, nc, False, test, lease.lease > 0, lease.lease)
    var status = tsvd_run_query(connection, request)
    if test:
        admitted_finite_refuse(status, _TSVD_EST)
    _ = lease
    _ = x
    _ = components  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = components^
    return out


def tsvd_transform(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if not py_contains(fields, "components_"):
        raise Error("mojolearn TruncatedSVD: call fit before transform")
    # Lane a7d: no host scan of a borrowed X; the transform entry tests it.
    var admitted = decomp_admit(source, _TSVD_EST, context, defer_finite=True)
    var x = admitted.array
    var out = _project(model, fields, admitted, context, not admitted.finite)
    _ = x^
    _ = admitted^
    return out


def tsvd_inverse_transform(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """X @ components_ (`inverse_transform` without the mean: its pointer is
    the components, a valid float32 address the entry does not read)."""
    var fields = py_fields(model)
    var admitted = decomp_admit(source, "TruncatedSVD", context)
    var z = admitted.array
    var rows = admitted.rows
    var nc = _int(_attr(model, fields, "n_components_"), context)
    if admitted.cols != nc:
        raise Error("mojolearn TruncatedSVD component count differs from fit")
    var nf = _int(_attr(model, fields, "n_features_in_"), context)
    var out = _f32(Python.tuple(rows, nf), context)
    var mode = _mode(fields, context)
    var connection = tsvd_connection(mode, context)
    var stored = _attr(model, fields, "components_")
    var components = admit_x(stored, 2, "components_", context)
    var lease = _resident_lease(model, connection, stored, components)
    var request = TSVDQueryRequest(admitted.address, _address(components, context), _address(out, context),
        rows, nf, nc, True, False, lease.lease > 0, lease.lease)
    _ = tsvd_run_query(connection, request)
    _ = lease
    _ = z  # keep the Python object alive through the native call above
    _ = components  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = components^
    _ = z^
    _ = admitted^
    return out


def tsvd_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        if len(call) != 4:
            raise Error("TypeError: tsvd_native_call requires model, operation, arguments and connected context")
        var model = call[0]
        var b = native_loaded_module("builtins")
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
                _ = tsvd_fit(model, args[0], context, True)
                result = model
            elif operation == "fit_transform":
                # the transform projects the X the fit converted: one conversion
                var admitted = tsvd_fit(model, args[0], context)
                result = _project(model, py_fields(model), admitted, context)
                _ = admitted^
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

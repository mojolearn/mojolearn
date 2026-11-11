# SPDX-License-Identifier: Apache-2.0
# GaussianProcessRegressor boundary admission, storage/image pins and public
# result packing. Fit/query/covariance/sample/optimizer/gradient/parameter
# execution uses typed GP requests and the existing numerical providers.
# Kernel packing, resident admission records and StandardScaler normalization
# are staged boundary dependencies, migrated in the following chunk.
from core.gp_native_api import GPRFitRequest, GPRQueryRequest, GPRCovarianceRequest, GPRSampleRequest, GPRGradientRequest, GPROptimizeRequest, GPThetaRequest, GP_CAP_GRADIENT, GP_CAP_OPTIMIZE, GP_CAP_SAMPLE, GP_CAP_THETA, GP_CAP_COVARIANCE
from core.gp_native_storage import GPRFitScratch, GPRGradientScratch, GPRGradientResult, GPOptimizerScratch, GPFreeParameters
from bindings.gp_native_boundary import GPConnection, gp_kernel_view, gp_run, gp_seed_word, gp_require
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.admitted_family import admit_family, admit_family_f64, admit_family_owned
from bindings.native_loaded_module import native_loaded_module
from bindings.metrics_native_support import metric_warn, metric_is_integral, metric_is_bool
from bindings.host_inference_scalars import host_inference_scalar
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.py_attr import py_attr
from bindings.gp_svm_gp_state import gp_kernel_arrays_reused, gp_x_known
from bindings.fitted_resident_door import fitted_resident_lease, fitted_resident_drop
from bindings.classical_arrays import classical_address
from bindings.public_lifecycle_owner import lifecycle_admit_fields
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_validate import classical_view
from bindings.gp_query_x import GpQuery, gp_query, gp_query_address, gp_query_refuse, gp_query_scan
from bindings.family_common import (
    family_repr, family_is_str, family_isinstance, family_call, family_array, family_shape_repr, family_field,
    family_is_real, FAMILY_F32P, FAMILY_F64P,
)
from bindings.scaler_native import scaler_fit
from bindings.scaler_lifecycle_native import scaler_initialize, scaler_transform
from bindings.metrics_reg_native import _reg_new, _reg_run, _arena_double, _RE_DIV
from bindings.gp_common_native import gp_fit_owned
from bindings.gp_common_native import (
    gp_extension, gp_numeric_mode_used, gp_model_kernel, gp_kernel_arrays,
    gp_free, gp_n_dims, gp_with_free_values, gp_f32,
)

comptime _GPR = "mojolearn GaussianProcessRegressor"
comptime _K_PROD = 5


# The prepared-owner slot of a regressor's device-resident fitted state.
comptime _GPR_RESIDENT_SLOT = "gpr_resident_fitted_v1"


def _gpr_ext(model: PythonObject, binding: PythonObject, context: PythonObject) raises -> PythonObject:
    return gp_extension(model, binding, _GPR, "the binary and the directory it sits in disagree, rebuild it with "
                        + "bash bindings/build_gp.sh", context)


def _gpr_default_kernel(context: PythonObject) raises -> PythonObject:
    """`ConstantKernel(1.0, "fixed") * RBF(1.0, "fixed")`, built from fields."""
    var b = native_loaded_module("builtins")
    var classes = context["family_gp_classes"]
    var const = family_call(b.object.__new__, classes["const"])
    var cf = py_fields(const)
    py_setitem(cf, "constant_value", PythonObject(Float64(1.0)))
    py_setitem(cf, "constant_value_bounds", PythonObject("fixed"))
    var rbf = family_call(b.object.__new__, classes["rbf"])
    var rf = py_fields(rbf)
    py_setitem(rf, "length_scale", Python.list(PythonObject(Float64(1.0))))
    py_setitem(rf, "length_scale_bounds", PythonObject("fixed"))
    var pair = family_call(b.object.__new__, classes["combined"])
    var pf = py_fields(pair)
    py_setitem(pf, "a", const)
    py_setitem(pf, "b", rbf)
    py_setitem(pf, "op", PythonObject(_K_PROD))
    py_setitem(pf, "sym", PythonObject("*"))
    return pair


def _is_integral(value: PythonObject, context: PythonObject) raises -> Bool:
    """Native integral scalar admission; never call numbers.Integral hooks."""
    var b = native_loaded_module("builtins")
    return metric_is_integral(value, context) and not metric_is_bool(value, context)


def _gpr_validate(model: PythonObject, context: PythonObject) raises:
    """`GaussianProcessRegressor._validate_params`."""
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var kernel = family_field(fields, "kernel")
    var alpha = family_field(fields, "alpha")
    var optimizer = family_field(fields, "optimizer")
    var n_restarts = family_field(fields, "n_restarts_optimizer")
    var normalize_y = family_field(fields, "normalize_y")
    var copy_x = family_field(fields, "copy_X_train")
    var n_targets = family_field(fields, "n_targets")
    var random_state = family_field(fields, "random_state")
    if kernel is None:
        kernel = _gpr_default_kernel(context)
    if not family_isinstance(kernel, context["family_gp_kernel_base"]):
        raise Error("TypeError: " + String(_GPR) + ": kernel must be a "
                    + "composition of mojolearn's RBF, Matern, ConstantKernel and "
                    + "WhiteKernel (a scikit-learn kernel object carries bounds "
                    + "and a theta this implementation deliberately has no use for; "
                    + "DEVIATION 1761)")
    if Bool(py=family_call(b.callable, optimizer)):
        raise Error("NotImplementedError: " + String(_GPR) + ": a callable optimizer is "
                    + "refused; it would run caller code on host floats whose "
                    + "arithmetic no column can pin. optimizer='fmin_l_bfgs_b' runs "
                    + "the identical bounded quasi-Newton optimizer (DEVIATION 2881)")
    if optimizer is None or (family_is_str(optimizer) and String(family_call(optimizer.lower)) == "none"):
        optimizer = PythonObject(None)
    elif not (family_is_str(optimizer) and String(optimizer) == "fmin_l_bfgs_b"):
        raise Error(String(_GPR) + ": optimizer=" + family_repr(optimizer) + " is "
                    + "not one of None and 'fmin_l_bfgs_b'")
    if not _is_integral(n_restarts, context) or Int(py=family_call(b.int, n_restarts)) < 0:
        raise Error(String(_GPR) + ": n_restarts_optimizer must "
                    + "be a non-negative int, got " + family_repr(n_restarts))
    var restarts = Int(py=family_call(b.int, n_restarts))
    if (not family_isinstance(normalize_y, Python.tuple(b.bool, b.int))
            or not py_contains(Python.tuple(PythonObject(0), PythonObject(1)), normalize_y)):
        raise Error("TypeError: " + String(_GPR) + ": normalize_y must be a bool, "
                    + "got " + family_repr(normalize_y))
    if not Bool(py=family_call(b.bool, copy_x)):
        raise Error("NotImplementedError: " + String(_GPR) + ": copy_X_train=False is "
                    + "refused. sklearn's False stores a REFERENCE to X; this "
                    + "surface always copies -- X crosses to float32 C-order host "
                    + "memory and to the device -- so False has nothing it could "
                    + "mean here, and accepting it would promise a memory saving "
                    + "that cannot be delivered")
    if n_targets is not None:
        raise Error("NotImplementedError: " + String(_GPR) + ": n_targets is refused; "
                    + "it shapes sample_y's draws from the unfitted prior, which is "
                    + "not implemented. y is single-target here")
    if random_state is not None:
        var bad = not _is_integral(random_state, context)
        if not bad:
            var seed = family_call(b.int, random_state)
            bad = not (Bool(py=py_call(py_attr(b.int, "__ge__"), seed, PythonObject(0)))
                       and Bool(py=py_call(py_attr(b.int, "__lt__"), seed, context["family_gp_seed_bound"])))
        if bad:
            raise Error(String(_GPR) + ": random_state must be None "
                        + "or an int in [0, 2**64), got " + family_repr(random_state) + "; it keys the "
                        + "restart draws (DEVIATION 2881). sample_y takes its own "
                        + "random_state argument (DEVIATION 2793)")
    if optimizer is not None and restarts > 0 and random_state is None:
        raise Error(String(_GPR) + ": n_restarts_optimizer > 0 "
                    + "needs random_state to be an int; the restart starting points "
                    + "are position-mapped Philox keyed by it (DEVIATION 2881), and "
                    + "scikit-learn's None (the global NumPy generator) has no "
                    + "identical counterpart")
    py_setitem(fields, "_kernel", kernel)
    py_setitem(fields, "_alpha", family_call(b.float, alpha))
    py_setitem(fields, "_optimizer", optimizer)
    py_setitem(fields, "_n_restarts_optimizer", PythonObject(restarts))
    py_setitem(fields, "_normalize_y", PythonObject(Bool(py=family_call(b.bool, normalize_y))))
    py_setitem(fields, "_random_state", PythonObject(None) if random_state is None else family_call(b.int, random_state))


@fieldwise_init
struct GPRNormalizedTargets(Movable):
    var targets: PythonObject
    var mean: Float64
    var stddev: Float64


def _gpr_normalize(model: PythonObject, binding: PythonObject, targets: PythonObject, n: Int,
                   context: PythonObject) raises -> GPRNormalizedTargets:
    """sklearn `_gpr.py:275-280` through StandardScaler in this estimator's
    tier: `(targets, y_mean, y_std)`, the transformed targets a new (n,)
    float32 Array."""
    var b = native_loaded_module("builtins")
    var scaler_cls = context["family_gp_scaler"][0]
    var scaler_context = context["family_gp_scaler"][1]
    classical_connect_numpy(scaler_context)
    var mode = gp_numeric_mode_used(binding, context)
    var scaler = family_call(b.object.__new__, scaler_cls)
    _ = scaler_initialize(scaler, Python.tuple(PythonObject(True), PythonObject(True), PythonObject(True), mode))
    # `targets.reshape((n_rows, 1))`: the same float32 bytes as one column.
    var column = family_array(Python.tuple(n, 1), "<f4", context)
    memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(column))),
           src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(targets))), count=n)
    _ = column  # keep the Python object alive through the native call above
    _ = scaler_fit(scaler, column, PythonObject(None), scaler_context)
    var sf = py_fields(scaler)
    var y_mean = Float64(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(sf["mean_"])))[0])
    var y_std = Float64(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(sf["scale_"])))[0])
    var scaled = scaler_transform(scaler, column, False, PythonObject(None), scaler_context)
    var flat = family_array(Python.tuple(n), "<f4", context)
    memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(flat))),
           src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(scaled))), count=n)
    _ = flat  # keep the Python object alive through the native call above
    _ = scaled  # keep the Python object alive through the native call above
    _ = scaled^
    _ = column^
    _ = scaler^
    return GPRNormalizedTargets(flat, y_mean, y_std)


def _gpr_lml_grad(model: PythonObject, binding: PythonObject, kernel: PythonObject, x: PythonObject,
                  targets: PythonObject, n_rows: Int, n_cols: Int, context: PythonObject) raises -> GPRGradientResult:
    var arrays = gp_kernel_arrays(kernel, context)
    var free = GPFreeParameters()
    gp_free(kernel, free, context)
    var scratch = GPRGradientScratch(len(free.values))
    var fields = py_fields(model)
    var ext = _gpr_ext(model, binding, context)
    var connection = GPConnection(ext)
    gp_require(connection, GP_CAP_GRADIENT, "gpr_lml_grad")
    var view = gp_kernel_view(arrays)
    var request = GPRGradientRequest(Int(py=classical_address(x)), Int(py=classical_address(targets)),
        Int(free.flags.unsafe_ptr()), Int(scratch.gradient.unsafe_ptr()), Int(scratch.scalars.unsafe_ptr()),
        n_rows, n_cols, view^, Float64(py=fields["_alpha"]))
    var info = gp_run(connection, request)
    lifecycle_admit_fields(fields)
    _ = arrays
    _ = free
    _ = x
    _ = targets
    var likelihood = scratch.scalars[1]
    var gradient = scratch.gradient^
    # Scratch owns one unused word when there are no free parameters; the
    # public gradient has always been empty in that case.
    if len(free.values) == 0:
        gradient = List[Float64]()
    return GPRGradientResult(info, likelihood, gradient^)


def _gpr_optimize(model: PythonObject, binding: PythonObject, x: PythonObject, targets: PythonObject,
                   n_rows: Int, n_cols: Int, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    var base = fields["_kernel"]
    var arrays = gp_kernel_arrays(base, context)
    var free = GPFreeParameters()
    gp_free(base, free, context)
    var n_theta = len(free.values)
    var n_runs = 1 + Int(py=fields["_n_restarts_optimizer"])
    var scratch = GPOptimizerScratch(n_theta, n_runs)
    var seed = PythonObject(0) if fields["_random_state"] is None else fields["_random_state"]
    var b = native_loaded_module("builtins")
    var bits = gp_seed_word(family_call(b.int, seed))
    var ext = _gpr_ext(model, binding, context)
    var connection = GPConnection(ext)
    gp_require(connection, GP_CAP_OPTIMIZE, "gpr_optimize")
    var kernel = gp_kernel_view(arrays)
    var request = GPROptimizeRequest(Int(py=classical_address(x)), Int(py=classical_address(targets)),
        Int(free.flags.unsafe_ptr()), Int(free.bounds.unsafe_ptr()), Int(scratch.theta.unsafe_ptr()),
        Int(scratch.values.unsafe_ptr()), Int(scratch.runs.unsafe_ptr()), n_rows, n_cols, kernel^,
        Float64(py=fields["_alpha"]), n_runs - 1, Int(bits & UInt64(0xFFFFFFFF)), Int(bits >> 32))
    _ = gp_run(connection, request)
    lifecycle_admit_fields(fields)
    _ = arrays
    _ = free
    _ = x
    _ = targets
    var runs = fields["_optimizer_runs"]
    if py_type(runs) is not b.list:
        raise Error("TypeError: native GP optimizer history requires an exact list")
    var stops = context["family_gp_opt_stops"]
    if py_type(stops) is not b.tuple and py_type(stops) is not b.list:
        raise Error("TypeError: native GP optimizer stop metadata requires an exact sequence")
    for run in range(n_runs):  # driver-loop: pack public optimizer restart records
        _ = py_call(runs.append, Python.tuple(PythonObject(Int(scratch.runs[run * 4])),
            PythonObject(Int(scratch.runs[run * 4 + 1])), stops[Int(scratch.runs[run * 4 + 2])],
            PythonObject(-scratch.runs[run * 4 + 3])))
    return gp_with_free_values(base, scratch.values, scratch.theta, context)


def _gpr_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`GaussianProcessRegressor.fit(X, y)`; args = (binding, X, y)."""
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    _gpr_validate(model, context)
    var fields = py_fields(model)
    var source = admit_family(args[1], 2, "X", context).array
    py_setitem(fields, "input_copied_", PythonObject(not (source is args[1])))
    var x = gp_fit_owned(source, args[1], context)
    _ = source^
    var n_rows = Int(py=x.shape[0])
    var n_cols = Int(py=x.shape[1])
    var view = classical_view(args[2], context, "y")
    var rank = Int(py=view.ndim)
    if rank != 1:
        raise Error(String(_GPR) + ": y must be 1-D, got "
                    + String(rank) + "-D shape " + family_shape_repr(view) + "; multi-target GP fits are not "
                    + "implemented (the n_targets refusal in __init__ is this same "
                    + "boundary)")
    # The admitted view goes on to the float32 cast (lane p3-svm-gp-kernel,
    # row 878): y is not viewed and converted a second time.
    var targets = admit_family_owned(admit_family(view, 1, "y", context).array, context)
    _ = view^
    if Int(py=targets.shape[0]) != n_rows:
        raise Error(String(_GPR) + ": y has " + String(Int(py=targets.shape[0]))
                    + " entries, X has " + String(n_rows) + " rows")
    var y_mean = Float64(0.0)
    var y_std = Float64(1.0)
    var normalize_y = Bool(py=fields["_normalize_y"])
    if normalize_y:
        var scaled = _gpr_normalize(model, args[0], targets, n_rows, context)
        targets = scaled.targets
        y_mean = scaled.mean
        y_std = scaled.stddev
    var kernel_fit = fields["_kernel"]
    py_setitem(fields, "_optimizer_runs", Python.list())
    if fields["_optimizer"] is not None and gp_n_dims(fields["_kernel"], context) > 0:
        kernel_fit = _gpr_optimize(model, args[0], x, targets, n_rows, n_cols, context)
    var arrays = gp_kernel_arrays(kernel_fit, context)
    # EVERY SIZE IS A FUNCTION OF (n_rows, n_cols) ALONE.
    var l_out = family_array(Python.tuple(n_rows, n_rows), "<f4", context)
    var dual = family_array(Python.tuple(n_rows), "<f4", context)
    # info, nb, logdet, ydotalpha, lml -- in that order.
    var scratch = GPRFitScratch()
    var ext = _gpr_ext(model, args[0], context)
    var connection = GPConnection(ext)
    var kernel = gp_kernel_view(arrays)
    var info = 0
    var alpha = Float64(py=fields["_alpha"])
    while True:  # driver-loop: over the tenfold ridge retries (alpha up to 1.0)
        # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpr_fit_binding.
        # x, y, kinds, kparams, ls_len, ls, l_out, dual_out, scalars_out;
        # n_train, n_features, n_nodes, n_ls, alpha
        var request = GPRFitRequest(Int(py=classical_address(x)), Int(py=classical_address(targets)),
            Int(py=classical_address(l_out)), Int(py=classical_address(dual)), scratch.address(),
            n_rows, n_cols, kernel.copy(), alpha, (connection.capabilities & 2) == 2)
        info = gp_run(connection, request)
        lifecycle_admit_fields(fields)
        _ = targets  # keep the Python object alive through the native call above
        _ = x  # keep the Python object alive through the native call above
        _ = scratch  # keep the Python object alive through the native call above
        _ = dual  # keep the Python object alive through the native call above
        _ = l_out  # keep the Python object alive through the native call above
        # A ridge too small for float32: a positive alpha is raised tenfold
        # and the fit repeated until the factor exists or the ridge reaches
        # the kernel's unit diagonal, with a warning naming the ridge used.
        if info == 0 or not alpha > 0.0 or alpha >= 1.0:
            break
        var raised = min(alpha * 10.0, 1.0)
        py_setitem(fields, "_alpha", PythonObject(raised))
        var message = (String(_GPR) + ": K + alpha*I is not positive "
                       + "definite in float32 with alpha=" + String(py_call(b.format, PythonObject(alpha), "g"))
                       + "; refitting with alpha=" + String(py_call(b.format, PythonObject(raised), "g"))
                       + ". Pass a larger alpha to skip this step.")
        metric_warn(message, b.RuntimeWarning, context)
        alpha = raised
    var sp = scratch.scalars.unsafe_ptr()
    _ = scratch  # keep the Python object alive through the native call above
    py_setitem(fields, "X_train_", x)
    py_setitem(fields, "y_train_", targets)
    py_setitem(fields, "_y_train_mean", PythonObject(y_mean))
    py_setitem(fields, "_y_train_std", PythonObject(y_std))
    py_setitem(fields, "normalize_y_", PythonObject(normalize_y))
    py_setitem(fields, "kernel_", kernel_fit)
    py_setitem(fields, "n_features_in_", PythonObject(n_cols))
    py_setitem(fields, "L_", l_out)
    py_setitem(fields, "alpha_", dual)
    py_setitem(fields, "info_", PythonObject(info))
    py_setitem(fields, "nb_", PythonObject(Int(sp[1])))
    _ = scratch  # keep the Python object alive through the native call above
    py_setitem(fields, "log_marginal_likelihood_value_", PythonObject(sp[4]))
    _ = scratch  # keep the Python object alive through the native call above
    py_setitem(fields, "_logdet", PythonObject(sp[2]))
    _ = scratch  # keep the Python object alive through the native call above
    py_setitem(fields, "_ydotalpha", PythonObject(sp[3]))
    _ = scratch  # keep the Python object alive through the native call above
    _ = arrays^
    _ = scratch^


def _gpr_fitted(fields: PythonObject, message: String) raises:
    if not py_contains(fields, PythonObject("alpha_")):
        raise Error(String(_GPR) + ": " + message)


def _gpr_query(fields: PythonObject, source: PythonObject, context: PythonObject,
               entry_tests: Bool = False) raises -> GpQuery:
    """`as_f32_c(X, ndim=2, name="X")` with the fitted feature count, as a
    GpQuery (bindings/gp_query_x.mojo; lane a7a): `entry_tests` (the
    resident predict, which returns the query's status words) keeps a
    cross-validation fold on the device. A caller whose entry returns no
    status word calls gp_query_scan."""
    # The public door already admitted this X as C-order float32 and
    # registered it (bindings/gp_svm_records.mojo): no second snapshot.
    var q = gp_query(source, "GaussianProcessRegressor", context, entry_tests)
    var fitted = Int(py=fields["n_features_in_"])
    if q.cols != fitted:
        raise Error(String(_GPR) + ": X has " + String(q.cols) + " "
                    + "features, fit saw " + String(fitted))
    return q^


@fieldwise_init
struct GPRNormalization(Copyable, Movable):
    var enabled: Bool
    var stddev: Float64
    var mean: Float64


def _gpr_normalization(fields: PythonObject) raises -> GPRNormalization:
    var b = native_loaded_module("builtins")
    var enabled = Bool(py=py_call(b.bool, py_get(fields, PythonObject("normalize_y_"), PythonObject(False))))
    if enabled:
        return GPRNormalization(True, Float64(py=family_call(b.float, fields["_y_train_std"])),
            Float64(py=family_call(b.float, fields["_y_train_mean"])))
    return GPRNormalization(False, 1.0, 0.0)


def _gpr_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`predict(X, return_std, return_cov)`; args = (binding, X, return_std, return_cov)."""
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var return_std = Bool(py=family_call(b.bool, args[2]))
    var return_cov = Bool(py=family_call(b.bool, args[3]))
    if return_cov and return_std:
        raise Error("RuntimeError: " + String(_GPR) + ": at most one of return_std "
                    + "or return_cov can be requested (scikit-learn's rule)")
    if return_cov:
        return _gpr_predict_cov(model, args, context)
    _gpr_fitted(fields, "call fit() first (the unfitted-prior arm of sklearn's predict is not implemented)")
    var ext = _gpr_ext(model, args[0], context)
    var connection = GPConnection(ext)
    var caps = connection.capabilities
    var resident = (caps & 4) == 4
    var gq = _gpr_query(fields, args[1], context, resident)
    var q = gq.q
    var n_star = gq.rows
    var xt = gp_f32(fields, "X_train_", 2, context)
    var n_train = Int(py=xt.shape[0])
    var arrays = gp_kernel_arrays_reused(model, gp_model_kernel(fields), context)  # packed once per kernel state
    var mean = family_array(Python.tuple(n_star), "<f4", context)
    # The variance, std and clamp outputs exist only for return_std (lane
    # p3-svm-gp-kernel, row 628): every binding leaves their addresses
    # untouched when return_std is 0, so a mean-only predict passes 0.
    var variance = List[Float32]()
    var std = PythonObject(None)
    var clamped = PythonObject(None)
    var var_address = 0
    var std_address = 0
    var clamped_address = 0
    if return_std:
        variance = List[Float32](length=n_star, fill=0.0)
        std = family_array(Python.tuple(n_star), "<f4", context)
        clamped = family_array(Python.tuple(n_star), "<i4", context)
        var_address = Int(variance.unsafe_ptr())
        std_address = Int(py=classical_address(std))
        clamped_address = Int(py=classical_address(clamped))
    var lf = gp_f32(fields, "L_", -1, context)
    var dual = gp_f32(fields, "alpha_", -1, context)
    var normalization = _gpr_normalization(fields)
    var route = 1 if (caps & 2) == 2 else 0
    var lease = 0
    if not resident:
        # These entries return no status word: the query is tested here.
        gp_query_scan(gq, "GaussianProcessRegressor", context)
    if resident:
        # Device-resident fitted state (lane ne1-a-gp-svm): X_train_, alpha_
        # and L_ cross to the device once per fitted state; the lease word
        # follows the pointer route's params. Lease 0 (an array the binding
        # reads is not the field itself, or cannot be made read-only) uploads
        # per call, as before.
        if xt is fields["X_train_"] and lf is fields["L_"] and dual is fields["alpha_"]:
            lease = fitted_resident_lease(model, _GPR_RESIDENT_SLOT, ext, "gp_resident_lease",
                                          Python.list(xt, dual, lf))
        else:
            fitted_resident_drop(model, _GPR_RESIDENT_SLOT)
        route = 2
    # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpr_predict_binding.
    # xtrain, l, dual, xstar, kinds, kparams, ls_len, ls, mean_out, var_out, std_out, clamped_out;
    # n_train, n_features, n_star, n_nodes, n_ls, return_std, info (+ normalize_y, y_std, y_mean)
    # (+ lease: gpr_predict_resident_binding)
    var kernel = gp_kernel_view(arrays)
    var request = GPRQueryRequest(Int(py=classical_address(xt)), Int(py=classical_address(lf)),
        Int(py=classical_address(dual)), Int(py=gp_query_address(gq)), Int(py=classical_address(mean)),
        var_address, std_address, clamped_address, n_train, Int(py=fields["n_features_in_"]), n_star,
        kernel^, Int(py=fields["info_"]), return_std, normalization.enabled, normalization.stddev,
        normalization.mean, route, lease, gq.store.base_id if gq.store.active() else -1,
        gq.store.base_rows if gq.store.active() else 0, gq.store.index_id if gq.store.active() else -1)
    var answer = gp_run(connection, request)
    lifecycle_admit_fields(fields)
    if resident:
        gp_query_refuse(answer.finite, answer.range_status, gq, "GaussianProcessRegressor")
    var n_clamped = answer.value
    _ = dual  # keep the Python object alive through the native call above
    _ = lf  # keep the Python object alive through the native call above
    _ = xt  # keep the Python object alive through the native call above
    _ = q  # keep the Python object alive through the native call above
    _ = variance  # keep the Python object alive through the native call above
    _ = mean  # keep the Python object alive through the native call above
    _ = clamped  # keep the Python object alive through the native call above
    _ = std  # keep the Python object alive through the native call above
    _ = xt^
    _ = lf^
    _ = dual^
    _ = q^
    _ = arrays^
    _ = variance^
    if return_std:
        py_setitem(fields, "clamped_", clamped)
        py_setitem(fields, "n_clamped_", PythonObject(n_clamped))
        _ = clamped  # keep the Python object alive through the native call above
        _ = std  # keep the Python object alive through the native call above
        _ = mean  # keep the Python object alive through the native call above
        return Python.tuple(mean, std)
    return mean


def _gpr_predict_cov(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`predict(X, return_cov=True)`: (mean, cov) through `gpr_predict_cov`."""
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    _gpr_fitted(fields, "call fit() first (the unfitted-prior arm of sklearn's predict is not implemented)")
    # This entry returns no status word: the query is tested here (lane a7a).
    var gq = _gpr_query(fields, args[1], context)
    gp_query_scan(gq, "GaussianProcessRegressor", context)
    var q = gq.q
    var n_star = Int(py=q.shape[0])
    var xt = gp_f32(fields, "X_train_", 2, context)
    var n_train = Int(py=xt.shape[0])
    var arrays = gp_kernel_arrays_reused(model, gp_model_kernel(fields), context)  # packed once per kernel state
    var ext = _gpr_ext(model, args[0], context)
    var connection = GPConnection(ext)
    gp_require(connection, GP_CAP_COVARIANCE, "gpr_predict_cov")
    # n_star >= 1 (as_f32_c refuses an empty X), so max(n, 1) is n.
    var mean = family_array(Python.tuple(n_star), "<f4", context)
    var cov = family_array(Python.tuple(n_star, n_star), "<f4", context)
    var lf = gp_f32(fields, "L_", -1, context)
    var dual = gp_f32(fields, "alpha_", -1, context)
    var normalization = _gpr_normalization(fields)
    var kernel = gp_kernel_view(arrays)
    var request = GPRCovarianceRequest(Int(py=classical_address(xt)), Int(py=classical_address(lf)),
        Int(py=classical_address(dual)), Int(py=classical_address(q)), Int(py=classical_address(mean)),
        Int(py=classical_address(cov)), n_train, Int(py=fields["n_features_in_"]), n_star,
        kernel^, Int(py=fields["info_"]), normalization.enabled, normalization.stddev, normalization.mean)
    _ = gp_run(connection, request)
    lifecycle_admit_fields(fields)
    _ = dual  # keep the Python object alive through the native call above
    _ = lf  # keep the Python object alive through the native call above
    _ = xt  # keep the Python object alive through the native call above
    _ = q  # keep the Python object alive through the native call above
    _ = cov  # keep the Python object alive through the native call above
    _ = mean  # keep the Python object alive through the native call above
    _ = xt^
    _ = lf^
    _ = dual^
    _ = q^
    _ = arrays^
    return Python.tuple(mean, cov)


def _gpr_sample_y(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`sample_y(X, n_samples, random_state)`; args = (binding, X, n_samples, random_state)."""
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    _gpr_fitted(fields, "call fit() first (sampling the unfitted prior is not implemented)")
    if not _is_integral(args[2], context):
        raise Error("TypeError: " + String(_GPR) + ": n_samples must be an int")
    var seed = args[3]
    var bad = not _is_integral(seed, context)
    if not bad:
        var value = family_call(b.int, seed)
        bad = not (Bool(py=py_call(py_attr(b.int, "__ge__"), value, PythonObject(0)))
                   and Bool(py=py_call(py_attr(b.int, "__lt__"), value, context["family_gp_seed_bound"])))
    if bad:
        raise Error(String(_GPR) + ": sample_y needs random_state to "
                    + "be an int in [0, 2**64), got " + family_repr(seed) + "; it is the key of every "
                    + "draw (DEVIATION 2793)")
    seed = family_call(b.int, seed)
    var n = Int(py=family_call(b.int, args[2]))
    # This entry returns no status word: the query is tested here (lane a7a).
    var gq = _gpr_query(fields, args[1], context)
    gp_query_scan(gq, "GaussianProcessRegressor", context)
    var q = gq.q
    var n_star = Int(py=q.shape[0])
    var xt = gp_f32(fields, "X_train_", 2, context)
    var n_train = Int(py=xt.shape[0])
    var arrays = gp_kernel_arrays_reused(model, gp_model_kernel(fields), context)  # packed once per kernel state
    # Never a zero-length buffer: a refused n still addresses real memory
    # (the binding refuses n_samples < 1 by name).
    var out = family_array(Python.tuple(n_star, n) if n_star * n >= 1 else Python.tuple(1), "<f4", context)
    var lf = gp_f32(fields, "L_", -1, context)
    var dual = gp_f32(fields, "alpha_", -1, context)
    var ext = _gpr_ext(model, args[0], context)
    var seed_bits = gp_seed_word(seed)
    var normalization = _gpr_normalization(fields)
    # Admit the actual native provider capability at this boundary position.
    var connection = GPConnection(ext)
    gp_require(connection, GP_CAP_SAMPLE, "gpr_sample_y")
    var kernel = gp_kernel_view(arrays)
    var request = GPRSampleRequest(Int(py=classical_address(xt)), Int(py=classical_address(lf)),
        Int(py=classical_address(dual)), Int(py=classical_address(q)), Int(py=classical_address(out)),
        n_train, Int(py=fields["n_features_in_"]), n_star, kernel^, Int(py=fields["info_"]),
        n, Int(seed_bits & UInt64(0xFFFFFFFF)), Int(seed_bits >> 32), normalization.enabled,
        normalization.stddev, normalization.mean)
    _ = gp_run(connection, request)
    lifecycle_admit_fields(fields)
    _ = dual  # keep the Python object alive through the native call above
    _ = lf  # keep the Python object alive through the native call above
    _ = xt  # keep the Python object alive through the native call above
    _ = q  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = xt^
    _ = lf^
    _ = dual^
    _ = q^
    _ = arrays^
    if n_star * n < 1:
        raise Error("mojolearn: cannot reshape size 1 into (" + String(n_star) + ", " + String(n) + ")")
    return out


def _gpr_theta(theta: PythonObject, context: PythonObject) raises -> List[Float64]:
    """`[float(t) for t in _flatten(theta)] if _shape_of(theta) else [float(theta)]`."""
    var b = native_loaded_module("builtins")
    var values = List[Float64]()
    if py_type(theta) is b.int or py_type(theta) is b.float:
        values.append(Float64(py=family_call(b.float, theta)))
        return values^
    var view = classical_view(theta, context, "theta")
    if Int(py=view.ndim) == 0:
        host_inference_scalar(theta, context)
        values.append(Float64(py=family_call(b.float, theta)))
        return values^
    var wide = admit_family_f64(view, "theta", context)
    var count = Int(py=wide.size)
    var wp = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(wide)))
    _ = wide  # keep the Python object alive through the native call above
    for i in range(count):  # driver-loop: over the kernel hyperparameters (n_dims entries)
        values.append(wp[i])
        _ = wide  # keep the Python object alive through the native call above
    _ = wide^
    _ = view^
    return values^


def _gpr_log_marginal_likelihood(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`log_marginal_likelihood(theta, eval_gradient, clone_kernel)`;
    args = (binding, theta, eval_gradient, clone_kernel)."""
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    _gpr_fitted(fields, "call fit() first")
    var eval_gradient = args[2]
    if args[1] is not None:
        if not py_contains(Python.tuple(PythonObject(True), PythonObject(False)), eval_gradient):
            raise Error("TypeError: " + String(_GPR) + ": eval_gradient must be a bool")
        var kern = fields["kernel_"]
        var theta = _gpr_theta(args[1], context)
        var n_dims = gp_n_dims(kern, context)
        if len(theta) != n_dims:
            raise Error(String(_GPR) + ": theta has " + String(len(theta)) + " entries, "
                        + "kernel_ has " + String(n_dims) + " free hyperparameters")
        var ext = _gpr_ext(model, args[0], context)
        var connection = GPConnection(ext)
        gp_require(connection, GP_CAP_THETA, "gp_theta_params")
        var params = List[Float64](length=len(theta), fill=0.0)
        var request = GPThetaRequest(Int(theta.unsafe_ptr()), Int(params.unsafe_ptr()), len(theta))
        _ = gp_run(connection, request)
        lifecycle_admit_fields(fields)
        _ = theta
        _ = params
        var xt = gp_f32(fields, "X_train_", 2, context)
        var yt = gp_f32(fields, "y_train_", 1, context)
        var at = gp_with_free_values(kern, params, List[Float64](), context)
        var result = _gpr_lml_grad(model, args[0], at, xt, yt, Int(py=xt.shape[0]), Int(py=xt.shape[1]), context)
        var lml = result.likelihood
        var info = result.info
        var grad = result.gradient^
        if info != 0:
            lml = Float64(py=py_call(b.float, PythonObject("-inf")))
            grad = List[Float64](length=len(theta), fill=0.0)
        _ = xt^
        _ = yt^
        if Bool(py=py_call(b.bool, eval_gradient)):
            var packed = Python.list()
            for i in range(len(grad)):  # driver-loop: pack the public hyperparameter gradient
                _ = py_call(packed.append, PythonObject(grad[i]))
            return Python.tuple(PythonObject(lml), packed)
        return PythonObject(lml)
    if Bool(py=family_call(b.bool, eval_gradient)):
        raise Error(String(_GPR) + ": eval_gradient=True needs a theta, "
                    + "scikit-learn's rule (_gpr.py:582)")
    var info = Int(py=fields["info_"])
    if info != 0:
        raise Error("RuntimeError: " + String(_GPR) + ": the factorization "
                    + "failed (info=" + String(info) + "), so the leading minor of order "
                    + String(info) + " of K + alpha*I was not positive definite and "
                    + "there is no marginal likelihood to report. For a Gaussian "
                    + "process that is a RESULT and not a bug: this kernel and "
                    + "this ridge do not describe these points. The usual causes "
                    + "are duplicate or near-duplicate training rows, a length "
                    + "scale far larger than the spread of the data, and alpha=0. "
                    + "sklearn's optimizer answers -inf here (_gpr.py:593) "
                    + "because it is comparing candidates; there is no optimizer "
                    + "in this lane (DEVIATION 1761), so this refuses instead. "
                    + "DEVIATION 1634")
    return fields["log_marginal_likelihood_value_"]


def _gpr_score(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`score(X, y)`: R^2 from `linear_model._r2_sums`' one program (the y
    mean rounded to float32 on the device, both sums of squares binary64);
    1.0 for a constant target predicted exactly, else 0.0 when SS_tot is 0.
    args = (binding, X, y)."""
    var pred = _gpr_predict(model, Python.tuple(args[0], args[1], PythonObject(False), PythonObject(False)), context)
    var n = Int(py=pred.shape[0])
    var view = classical_view(args[2], context, "y")
    if Int(py=view.ndim) != 1 or Int(py=view.shape[0]) != n:
        raise Error(String(_GPR) + ": y must be 1-D with one "
                    + "entry per row of X, got shape " + family_shape_repr(view) + " against "
                    + String(n) + " predictions")
    if not py_contains(context, PythonObject("family_metrics_ops")):
        raise Error("ImportError: mojolearn GaussianProcessRegressor.score needs the metrics owner (mojolearn._expansion_metrics)")
    # The admitted view goes on to the float32 cast (lane p3-svm-gp-kernel,
    # row 629): y is not viewed and converted a second time.
    var t = admit_family(view, 1, "y", context).array
    _ = view^
    var r = _reg_new(Python.tuple(t, pred, PythonObject(None)), PythonObject("uniform_average"), "score", True, context)
    var sq = Int(py=context["family_metrics_terms"]["sq"])
    r.program(0.0, context)
    var y_off = r.y_off
    var p_off = r.p_off
    var y_sums = r.sums(y_off)
    var y_mean = r.fin32(y_sums, _RE_DIV)
    var res_terms = r.term(sq, y_off, p_off, False)
    var res_sums = r.sums(res_terms)
    var num = r.fin(res_sums, 0)
    var tot_terms = r.term(sq, y_off, y_mean, True)
    var tot_sums = r.sums(tot_terms)
    var den = r.fin(tot_sums, 0)
    _ = r.prog.want(num, 2)
    _ = r.prog.want(den, 2)
    var arena = _reg_run(r.prog, PythonObject(None), context)
    var ss_res = _arena_double(arena, num)
    var ss_tot = _arena_double(arena, den)
    _ = t^
    if ss_tot == 0.0:
        return PythonObject(Float64(1.0) if ss_res == 0.0 else Float64(0.0))
    return PythonObject(Float64(1.0) - ss_res / ss_tot)


def gp_regressor_operation(model: PythonObject, operation: String, args: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    """GaussianProcessRegressor's public operations: args[0] is the resolved binding."""
    if operation == "family_gp_gpr_fit":
        _gpr_fit(model, args, context)
        return model
    if operation == "family_gp_gpr_predict":
        return _gpr_predict(model, args, context)
    if operation == "family_gp_gpr_sample_y":
        return _gpr_sample_y(model, args, context)
    if operation == "family_gp_gpr_lml":
        return _gpr_log_marginal_likelihood(model, args, context)
    if operation == "family_gp_gpr_score":
        return _gpr_score(model, args, context)
    raise Error("RuntimeError: unknown gaussian process regressor operation " + operation)

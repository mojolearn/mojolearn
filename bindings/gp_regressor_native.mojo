# SPDX-License-Identifier: Apache-2.0
# GaussianProcessRegressor `fit`, `predict`, `sample_y`, `log_marginal_likelihood`
# and `score`, one native call each (python/mojolearn/_gp_impl.py). The shell hands
# over the `_mojolearn_gp` binding `_bind` resolved; this owner makes the Python
# methods' refusals in their wording and order, cross-checks the binary's
# compile-time mode, runs normalize_y through StandardScaler's own native fit and
# transform (bindings/scaler_native.mojo, bindings/scaler_lifecycle_native.mojo),
# packs the kernel from its fields, runs the binding's gpr_optimize, gpr_fit(_ptr),
# gpr_predict(_ptr), gpr_predict_cov, gpr_sample_y, gp_theta_params and gpr_lml_grad
# with the argument lists bindings/_mojolearn_gp.mojo documents, in the order the
# Python made them, and stores the fit. score is the same single-output R^2
# program `linear_model._r2_sums` ran (bindings/metrics_reg_native.mojo's `_Reg`).
# The same entries on the same inputs: the bits do not move.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_validate import classical_view
from bindings.family_common import (
    family_repr, family_is_str, family_isinstance, family_call, family_array, family_as_f32,
    family_shape_repr, family_owned, family_field, family_is_real, family_cast_view_f64, FAMILY_F32P, FAMILY_F64P,
)
from bindings.archive_schema_native import _schema_label_store
from bindings.scaler_native import scaler_fit
from bindings.scaler_lifecycle_native import scaler_initialize, scaler_transform
from bindings.metrics_reg_native import _reg_new, _reg_run, _arena_double, _RE_DIV
from bindings.gp_common_native import (
    gp_extension, gp_caps, gp_entry, gp_numeric_mode_used, gp_model_kernel, gp_kernel_arrays,
    gp_free, gp_n_dims, gp_with_free_values, gp_f32,
)

comptime _GPR = "mojolearn GaussianProcessRegressor"
comptime _K_PROD = 5


def _gpr_ext(model: PythonObject, binding: PythonObject, context: PythonObject) raises -> PythonObject:
    return gp_extension(model, binding, _GPR, "the binary and the directory it sits in disagree, rebuild it with "
                        + "bash bindings/build_gp.sh", context)


def _gpr_default_kernel(context: PythonObject) raises -> PythonObject:
    """`ConstantKernel(1.0, "fixed") * RBF(1.0, "fixed")`, built from fields."""
    var b = Python.import_module("builtins")
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


def _is_integral(value: PythonObject) raises -> Bool:
    """`isinstance(value, numbers.Integral) and not isinstance(value, bool)`."""
    var b = Python.import_module("builtins")
    return (family_isinstance(value, Python.import_module("numbers").Integral)
            and not family_isinstance(value, b.bool))


def _gpr_validate(model: PythonObject, context: PythonObject) raises:
    """`GaussianProcessRegressor._validate_params`."""
    var b = Python.import_module("builtins")
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
    if not _is_integral(n_restarts) or Int(py=family_call(b.int, n_restarts)) < 0:
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
        var bad = not _is_integral(random_state)
        if not bad:
            var seed = family_call(b.int, random_state)
            bad = not (Bool(py=py_call(b.int.__ge__, seed, PythonObject(0)))
                       and Bool(py=py_call(b.int.__lt__, seed, context["family_gp_seed_bound"])))
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


def _gpr_normalize(model: PythonObject, binding: PythonObject, targets: PythonObject, n: Int,
                   context: PythonObject) raises -> PythonObject:
    """sklearn `_gpr.py:275-280` through StandardScaler in this estimator's
    tier: `(targets, y_mean, y_std)`, the transformed targets a new (n,)
    float32 Array."""
    var b = Python.import_module("builtins")
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
    _ = scaler_fit(scaler, column, PythonObject(None), scaler_context)
    var sf = py_fields(scaler)
    var y_mean = Float64(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(sf["mean_"])))[0])
    var y_std = Float64(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(sf["scale_"])))[0])
    var scaled = scaler_transform(scaler, column, False, PythonObject(None), scaler_context)
    var flat = family_array(Python.tuple(n), "<f4", context)
    memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(flat))),
           src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(scaled))), count=n)
    _ = scaled^
    _ = column^
    _ = scaler^
    return Python.tuple(flat, PythonObject(y_mean), PythonObject(y_std))


def _gpr_lml_grad(model: PythonObject, binding: PythonObject, kernel: PythonObject, x: PythonObject,
                  targets: PythonObject, n_rows: Int, n_cols: Int, context: PythonObject) raises -> PythonObject:
    """`_lml_grad`: (info, lml, grad list) at `kernel` through `gpr_lml_grad`."""
    var b = Python.import_module("builtins")
    var arrays = gp_kernel_arrays(kernel, context)
    var flags = Python.list()
    var values = Python.list()
    var bounds = Python.list()
    gp_free(kernel, flags, values, bounds, context)
    var free = _schema_label_store(flags, "i", "<i4", context)
    var n_free = len(values)
    var grad = family_array(Python.tuple(max(n_free, 1)), "<f8", context)
    var scalars = family_array(Python.tuple(2), "<f8", context)
    var fields = py_fields(model)
    var ext = _gpr_ext(model, binding, context)
    # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpr_lml_grad_binding.
    # x, y, kinds, kparams, ls_len, ls, free, grad_out, scalars_out;
    # n_train, n_features, n_nodes, n_ls, alpha
    var info = family_call(gp_entry(ext, "gpr_lml_grad"),
        Python.list(classical_address(x), classical_address(targets), classical_address(arrays[0]),
                    classical_address(arrays[1]), classical_address(arrays[2]), classical_address(arrays[3]),
                    classical_address(free), classical_address(grad), classical_address(scalars)),
        Python.list(PythonObject(n_rows), PythonObject(n_cols), PythonObject(len(arrays[0])), arrays[4],
                    fields["_alpha"]))
    var gp = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(grad)))
    var out = Python.list()
    for i in range(n_free):  # driver-loop: over the kernel hyperparameters (n_dims entries)
        _ = py_call(out.append, PythonObject(gp[i]))
    var lml = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(scalars)))[1]
    _ = free^
    _ = arrays^
    return Python.tuple(family_call(b.int, info), PythonObject(lml), out, grad, scalars)


def _gpr_optimize(model: PythonObject, binding: PythonObject, x: PythonObject, targets: PythonObject,
                  n_rows: Int, n_cols: Int, context: PythonObject) raises -> PythonObject:
    """`_optimize`: every start in ONE `gpr_optimize` call; each run recorded
    in `_optimizer_runs` as (n_iter, n_eval, stop, -f); returns the fitted
    kernel, whose theta is the winning run's."""
    var fields = py_fields(model)
    var base = fields["_kernel"]
    var arrays = gp_kernel_arrays(base, context)
    var flags = Python.list()
    var values = Python.list()
    var pairs = Python.list()
    gp_free(base, flags, values, pairs, context)
    var free = _schema_label_store(flags, "i", "<i4", context)
    var n_theta = len(values)
    var flat = Python.list()
    for i in range(len(pairs)):  # driver-loop: over the free hyperparameters' bound pairs
        _ = py_call(flat.append, pairs[i][0])
        _ = py_call(flat.append, pairs[i][1])
    var bounds = _schema_label_store(flat, "f", "<f4", context)
    var n_runs = 1 + Int(py=fields["_n_restarts_optimizer"])
    var theta_out = family_array(Python.tuple(n_theta), "<f8", context)
    var values_out = family_array(Python.tuple(n_theta), "<f8", context)
    var runs_out = family_array(Python.tuple(n_runs * 4), "<f8", context)
    # random_state may be None only with no restarts (checked in
    # _validate_params); the kernel then draws no restart point.
    var seed = PythonObject(0) if fields["_random_state"] is None else fields["_random_state"]
    var b = Python.import_module("builtins")
    var low = py_call(b.int.__and__, seed, PythonObject(0xFFFFFFFF))
    var high = py_call(b.int.__and__, py_call(b.int.__rshift__, seed, PythonObject(32)), PythonObject(0xFFFFFFFF))
    var ext = _gpr_ext(model, binding, context)
    # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpr_optimize_binding.
    # x, y, kinds, kparams, ls_len, ls, free, bounds, theta_out, values_out, runs_out;
    # n_train, n_features, n_nodes, n_ls, alpha, n_restarts, seed_lo, seed_hi
    _ = family_call(gp_entry(ext, "gpr_optimize"),
        Python.list(classical_address(x), classical_address(targets), classical_address(arrays[0]),
                    classical_address(arrays[1]), classical_address(arrays[2]), classical_address(arrays[3]),
                    classical_address(free), classical_address(bounds), classical_address(theta_out),
                    classical_address(values_out), classical_address(runs_out)),
        Python.list(PythonObject(n_rows), PythonObject(n_cols), PythonObject(len(arrays[0])), arrays[4],
                    fields["_alpha"], PythonObject(n_runs - 1), low, high))
    var runs = fields["_optimizer_runs"]
    var stops = context["family_gp_opt_stops"]
    var rp = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(runs_out)))
    for r in range(n_runs):  # driver-loop: over the optimizer restarts
        _ = py_call(runs.append, Python.tuple(PythonObject(Int(rp[r * 4])), PythonObject(Int(rp[r * 4 + 1])),
                                              stops[Int(rp[r * 4 + 2])], PythonObject(-rp[r * 4 + 3])))
    var tp = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(theta_out)))
    var vp = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(values_out)))
    var theta = Python.list()
    var params = Python.list()
    for i in range(n_theta):  # driver-loop: over the kernel hyperparameters (n_dims entries)
        _ = py_call(theta.append, PythonObject(tp[i]))
        _ = py_call(params.append, PythonObject(vp[i]))
    _ = arrays^
    _ = free^
    _ = bounds^
    _ = theta_out^
    _ = values_out^
    _ = runs_out^
    return gp_with_free_values(base, params, theta, context)


def _gpr_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`GaussianProcessRegressor.fit(X, y)`; args = (binding, X, y)."""
    classical_connect_numpy(context)
    var b = Python.import_module("builtins")
    _gpr_validate(model, context)
    var fields = py_fields(model)
    var source = family_as_f32(args[1], 2, "X", context)
    py_setitem(fields, "input_copied_", PythonObject(not (source is args[1])))
    var x = family_owned(source, context)
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
    _ = view^
    var targets = family_owned(family_as_f32(args[2], 1, "y", context), context)
    if Int(py=targets.shape[0]) != n_rows:
        raise Error(String(_GPR) + ": y has " + String(Int(py=targets.shape[0]))
                    + " entries, X has " + String(n_rows) + " rows")
    var y_mean = Float64(0.0)
    var y_std = Float64(1.0)
    if Bool(py=fields["_normalize_y"]):
        var scaled = _gpr_normalize(model, args[0], targets, n_rows, context)
        targets = scaled[0]
        y_mean = Float64(py=scaled[1])
        y_std = Float64(py=scaled[2])
    var kernel_fit = fields["_kernel"]
    py_setitem(fields, "_optimizer_runs", Python.list())
    if fields["_optimizer"] is not None and gp_n_dims(fields["_kernel"], context) > 0:
        kernel_fit = _gpr_optimize(model, args[0], x, targets, n_rows, n_cols, context)
    var arrays = gp_kernel_arrays(kernel_fit, context)
    # EVERY SIZE IS A FUNCTION OF (n_rows, n_cols) ALONE.
    var l_out = family_array(Python.tuple(n_rows, n_rows), "<f4", context)
    var dual = family_array(Python.tuple(n_rows), "<f4", context)
    # info, nb, logdet, ydotalpha, lml -- in that order.
    var scalars = family_array(Python.tuple(5), "<f8", context)
    var ext = _gpr_ext(model, args[0], context)
    var fit_fn = gp_entry(ext, String("gpr_fit_ptr") if (gp_caps(ext) & 2) == 2 else String("gpr_fit"))
    var info = 0
    while True:  # driver-loop: over the tenfold ridge retries (alpha up to 1.0)
        var alpha = Float64(py=fields["_alpha"])
        # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpr_fit_binding.
        # x, y, kinds, kparams, ls_len, ls, l_out, dual_out, scalars_out;
        # n_train, n_features, n_nodes, n_ls, alpha
        info = Int(py=family_call(b.int, family_call(fit_fn,
            Python.list(classical_address(x), classical_address(targets), classical_address(arrays[0]),
                        classical_address(arrays[1]), classical_address(arrays[2]), classical_address(arrays[3]),
                        classical_address(l_out), classical_address(dual), classical_address(scalars)),
            Python.list(PythonObject(n_rows), PythonObject(n_cols), PythonObject(len(arrays[0])), arrays[4],
                        fields["_alpha"]))))
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
        _ = family_call(Python.import_module("warnings").warn, PythonObject(message), b.RuntimeWarning)
    var sp = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(scalars)))
    py_setitem(fields, "X_train_", x)
    py_setitem(fields, "y_train_", targets)
    py_setitem(fields, "_y_train_mean", PythonObject(y_mean))
    py_setitem(fields, "_y_train_std", PythonObject(y_std))
    py_setitem(fields, "normalize_y_", PythonObject(Bool(py=fields["_normalize_y"])))
    py_setitem(fields, "kernel_", kernel_fit)
    py_setitem(fields, "n_features_in_", PythonObject(n_cols))
    py_setitem(fields, "L_", l_out)
    py_setitem(fields, "alpha_", dual)
    py_setitem(fields, "info_", PythonObject(info))
    py_setitem(fields, "nb_", PythonObject(Int(sp[1])))
    py_setitem(fields, "log_marginal_likelihood_value_", PythonObject(sp[4]))
    py_setitem(fields, "_logdet", PythonObject(sp[2]))
    py_setitem(fields, "_ydotalpha", PythonObject(sp[3]))
    _ = arrays^
    _ = scalars^


def _gpr_fitted(fields: PythonObject, message: String) raises:
    if not py_contains(fields, PythonObject("alpha_")):
        raise Error(String(_GPR) + ": " + message)


def _gpr_query(fields: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`as_f32_c(X, ndim=2, name="X")` with the fitted feature count."""
    var q = family_as_f32(source, 2, "X", context)
    var d = Int(py=q.shape[1])
    var fitted = Int(py=fields["n_features_in_"])
    if d != fitted:
        raise Error(String(_GPR) + ": X has " + String(d) + " "
                    + "features, fit saw " + String(fitted))
    return q


def _gpr_tail(fields: PythonObject, params: PythonObject) raises:
    """normalize_y's `[1, y_std, y_mean]` tail: the binding un-normalizes."""
    var b = Python.import_module("builtins")
    if Bool(py=py_call(b.bool, py_get(fields, PythonObject("normalize_y_"), PythonObject(False)))):
        _ = py_call(params.append, PythonObject(1))
        _ = py_call(params.append, family_call(b.float, fields["_y_train_std"]))
        _ = py_call(params.append, family_call(b.float, fields["_y_train_mean"]))


def _gpr_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`predict(X, return_std, return_cov)`; args = (binding, X, return_std, return_cov)."""
    classical_connect_numpy(context)
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var return_std = Bool(py=family_call(b.bool, args[2]))
    var return_cov = Bool(py=family_call(b.bool, args[3]))
    if return_cov and return_std:
        raise Error("RuntimeError: " + String(_GPR) + ": at most one of return_std "
                    + "or return_cov can be requested (scikit-learn's rule)")
    if return_cov:
        return _gpr_predict_cov(model, args, context)
    _gpr_fitted(fields, "call fit() first (the unfitted-prior arm of sklearn's predict is not implemented)")
    var q = _gpr_query(fields, args[1], context)
    var n_star = Int(py=q.shape[0])
    var xt = gp_f32(fields, "X_train_", 2, context)
    var n_train = Int(py=xt.shape[0])
    var arrays = gp_kernel_arrays(gp_model_kernel(fields), context)
    var mean = family_array(Python.tuple(n_star), "<f4", context)
    var var_out = family_array(Python.tuple(n_star), "<f4", context)
    var std = family_array(Python.tuple(n_star), "<f4", context)
    var clamped = family_array(Python.tuple(n_star), "<i4", context)
    var lf = gp_f32(fields, "L_", -1, context)
    var dual = gp_f32(fields, "alpha_", -1, context)
    var ext = _gpr_ext(model, args[0], context)
    var params = Python.list(PythonObject(n_train), fields["n_features_in_"], PythonObject(n_star),
                             PythonObject(len(arrays[0])), arrays[4], PythonObject(1 if return_std else 0),
                             fields["info_"])
    _gpr_tail(fields, params)
    # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpr_predict_binding.
    # xtrain, l, dual, xstar, kinds, kparams, ls_len, ls, mean_out, var_out, std_out, clamped_out;
    # n_train, n_features, n_star, n_nodes, n_ls, return_std, info (+ normalize_y, y_std, y_mean)
    var n_clamped = family_call(gp_entry(ext, String("gpr_predict_ptr") if (gp_caps(ext) & 2) == 2 else String("gpr_predict")),
        Python.list(classical_address(xt), classical_address(lf), classical_address(dual), classical_address(q),
                    classical_address(arrays[0]), classical_address(arrays[1]), classical_address(arrays[2]),
                    classical_address(arrays[3]), classical_address(mean), classical_address(var_out),
                    classical_address(std), classical_address(clamped)),
        params)
    _ = xt^
    _ = lf^
    _ = dual^
    _ = q^
    _ = arrays^
    _ = var_out^
    if return_std:
        py_setitem(fields, "clamped_", clamped)
        py_setitem(fields, "n_clamped_", family_call(b.int, n_clamped))
        return Python.tuple(mean, std)
    return mean


def _gpr_predict_cov(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`predict(X, return_cov=True)`: (mean, cov) through `gpr_predict_cov`."""
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    _gpr_fitted(fields, "call fit() first (the unfitted-prior arm of sklearn's predict is not implemented)")
    var q = _gpr_query(fields, args[1], context)
    var n_star = Int(py=q.shape[0])
    var xt = gp_f32(fields, "X_train_", 2, context)
    var n_train = Int(py=xt.shape[0])
    var arrays = gp_kernel_arrays(gp_model_kernel(fields), context)
    var ext = _gpr_ext(model, args[0], context)
    var entry = py_call(b.getattr, ext, PythonObject("gpr_predict_cov"), PythonObject(None))
    if not Bool(py=py_call(b.callable, entry)):
        raise Error("ImportError: " + String(_GPR) + ": " + String(py_call(b.getattr, ext, PythonObject("__name__")))
                    + " does not export gpr_predict_cov, so return_cov=True cannot run on it; rebuild the GP binding")
    # n_star >= 1 (as_f32_c refuses an empty X), so max(n, 1) is n.
    var mean = family_array(Python.tuple(n_star), "<f4", context)
    var cov = family_array(Python.tuple(n_star, n_star), "<f4", context)
    var lf = gp_f32(fields, "L_", -1, context)
    var dual = gp_f32(fields, "alpha_", -1, context)
    var params = Python.list(PythonObject(n_train), fields["n_features_in_"], PythonObject(n_star),
                             PythonObject(len(arrays[0])), arrays[4], fields["info_"])
    _gpr_tail(fields, params)
    # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpr_predict_cov_binding.
    # xtrain, l, dual, xstar, kinds, kparams, ls_len, ls, mean_out, cov_out;
    # n_train, n_features, n_star, n_nodes, n_ls, info (+ normalize_y, y_std, y_mean)
    _ = family_call(entry,
        Python.list(classical_address(xt), classical_address(lf), classical_address(dual), classical_address(q),
                    classical_address(arrays[0]), classical_address(arrays[1]), classical_address(arrays[2]),
                    classical_address(arrays[3]), classical_address(mean), classical_address(cov)),
        params)
    _ = xt^
    _ = lf^
    _ = dual^
    _ = q^
    _ = arrays^
    return Python.tuple(mean, cov)


def _gpr_sample_y(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`sample_y(X, n_samples, random_state)`; args = (binding, X, n_samples, random_state)."""
    classical_connect_numpy(context)
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    _gpr_fitted(fields, "call fit() first (sampling the unfitted prior is not implemented)")
    if not _is_integral(args[2]):
        raise Error("TypeError: " + String(_GPR) + ": n_samples must be an int")
    var seed = args[3]
    var bad = not _is_integral(seed)
    if not bad:
        var value = family_call(b.int, seed)
        bad = not (Bool(py=py_call(b.int.__ge__, value, PythonObject(0)))
                   and Bool(py=py_call(b.int.__lt__, value, context["family_gp_seed_bound"])))
    if bad:
        raise Error(String(_GPR) + ": sample_y needs random_state to "
                    + "be an int in [0, 2**64), got " + family_repr(seed) + "; it is the key of every "
                    + "draw (DEVIATION 2793)")
    seed = family_call(b.int, seed)
    var n = Int(py=family_call(b.int, args[2]))
    var q = _gpr_query(fields, args[1], context)
    var n_star = Int(py=q.shape[0])
    var xt = gp_f32(fields, "X_train_", 2, context)
    var n_train = Int(py=xt.shape[0])
    var arrays = gp_kernel_arrays(gp_model_kernel(fields), context)
    # Never a zero-length buffer: a refused n still addresses real memory
    # (the binding refuses n_samples < 1 by name).
    var out = family_array(Python.tuple(n_star, n) if n_star * n >= 1 else Python.tuple(1), "<f4", context)
    var lf = gp_f32(fields, "L_", -1, context)
    var dual = gp_f32(fields, "alpha_", -1, context)
    var ext = _gpr_ext(model, args[0], context)
    var params = Python.list(PythonObject(n_train), fields["n_features_in_"], PythonObject(n_star),
                             PythonObject(len(arrays[0])), arrays[4], fields["info_"], PythonObject(n),
                             py_call(b.int.__and__, seed, PythonObject(0xFFFFFFFF)),
                             py_call(b.int.__rshift__, seed, PythonObject(32)))
    _gpr_tail(fields, params)
    # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpr_sample_y_binding.
    # xtrain, l, dual, xstar, kinds, kparams, ls_len, ls, y_out;
    # n_train, n_features, n_star, n_nodes, n_ls, info, n_samples, seed_lo, seed_hi (+ normalize_y tail)
    _ = family_call(gp_entry(ext, "gpr_sample_y"),
        Python.list(classical_address(xt), classical_address(lf), classical_address(dual), classical_address(q),
                    classical_address(arrays[0]), classical_address(arrays[1]), classical_address(arrays[2]),
                    classical_address(arrays[3]), classical_address(out)),
        params)
    _ = xt^
    _ = lf^
    _ = dual^
    _ = q^
    _ = arrays^
    if n_star * n < 1:
        raise Error("mojolearn: cannot reshape size 1 into (" + String(n_star) + ", " + String(n) + ")")
    return out


def _gpr_theta(theta: PythonObject, context: PythonObject) raises -> PythonObject:
    """`[float(t) for t in _flatten(theta)] if _shape_of(theta) else [float(theta)]`."""
    var b = Python.import_module("builtins")
    var out = Python.list()
    if family_is_real(theta):
        _ = py_call(out.append, family_call(b.float, theta))
        return out
    var view = classical_view(theta, context, "theta")
    if Int(py=view.ndim) == 0:
        _ = py_call(out.append, family_call(b.float, theta))
        return out
    var wide = family_cast_view_f64(view, "theta", context)
    var count = Int(py=wide.size)
    var wp = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(wide)))
    for i in range(count):  # driver-loop: over the kernel hyperparameters (n_dims entries)
        _ = py_call(out.append, PythonObject(wp[i]))
    _ = wide^
    _ = view^
    return out


def _gpr_log_marginal_likelihood(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`log_marginal_likelihood(theta, eval_gradient, clone_kernel)`;
    args = (binding, theta, eval_gradient, clone_kernel)."""
    classical_connect_numpy(context)
    var b = Python.import_module("builtins")
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
        var exp_values = family_call(gp_entry(ext, "gp_theta_params"), theta)
        var params = Python.list()
        for i in range(len(exp_values)):  # driver-loop: over the kernel hyperparameters (n_dims entries)
            _ = py_call(params.append, family_call(b.float, exp_values[i]))
        var xt = gp_f32(fields, "X_train_", 2, context)
        var yt = gp_f32(fields, "y_train_", 1, context)
        var at = gp_with_free_values(kern, params, PythonObject(None), context)
        var result = _gpr_lml_grad(model, args[0], at, xt, yt, Int(py=xt.shape[0]), Int(py=xt.shape[1]), context)
        var lml = result[1]
        var grad = result[2]
        if Int(py=result[0]) != 0:
            lml = py_call(b.float, PythonObject("-inf"))
            grad = Python.list()
            for _ in range(len(theta)):  # driver-loop: over the kernel hyperparameters (n_dims entries)
                _ = py_call(grad.append, PythonObject(Float64(0.0)))
        _ = xt^
        _ = yt^
        if Bool(py=py_call(b.bool, eval_gradient)):
            return Python.tuple(lml, grad)
        return lml
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
    _ = view^
    if not py_contains(context, PythonObject("family_metrics_ops")):
        raise Error("ImportError: mojolearn GaussianProcessRegressor.score needs the metrics owner (mojolearn._expansion_metrics)")
    var t = family_as_f32(args[2], 1, "y", context)
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

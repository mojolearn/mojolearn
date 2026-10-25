# SPDX-License-Identifier: Apache-2.0
# GaussianProcessClassifier `fit`, `predict`, `predict_proba` and
# `latent_mean_and_variance`, one native call each (python/mojolearn/_gpc_impl.py).
# The shell hands over the `_mojolearn_gp` binding `_bind` resolved (and, for fit
# and predict, the base binding the label helpers live on); this owner makes the
# Python methods' refusals in their wording and order, encodes the classes
# (bindings/gp_labels_native.mojo), cross-checks the binary's compile-time mode,
# packs the kernel from its fields, runs every binary Laplace fit in one
# gpc_fit_all session (or one gpc_fit per one-vs-rest class where the binding has
# no session entry), predicts with gpc_predict_all (or gpc_predict per class and
# gpc_ovr_combine), and stores the fit as `_set_fitted` did. The same entries on
# the same inputs in the same order: the bits do not move.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.py_attr import py_attr
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import (
    family_repr, family_is_str, family_isinstance, family_call, family_array, family_as_f32,
    family_owned, family_field, FAMILY_F64P,
)
from core.hotpath_impl import reduce_stat, cast_elements, HP_F64, HP_SUM, HP_I32, HP_I64
from bindings.gp_common_native import gp_extension, gp_caps, gp_entry, gp_kernel_arrays, gp_f32
from bindings.gp_labels_native import gp_encode_labels, gp_decode_labels
from bindings.mcls_classify import mcls_accuracy_fraction
from bindings.mcls_common import mc_shape_of

comptime _NAME = "mojolearn GaussianProcessClassifier"
comptime _K_PROD = 5


def _gpc_ext(model: PythonObject, binding: PythonObject, context: PythonObject) raises -> PythonObject:
    return gp_extension(model, binding, _NAME, "rebuild it with bash bindings/build_gp.sh", context)


def _gpc_default_kernel(context: PythonObject) raises -> PythonObject:
    """`ConstantKernel(1.0) * RBF(1.0)` with scikit-learn's default bounds."""
    var b = Python.import_module("builtins")
    var classes = context["family_gp_classes"]
    var pair_bounds = context["family_gp_default_bounds"]
    var const = family_call(b.object.__new__, classes["const"])
    var cf = py_fields(const)
    py_setitem(cf, "constant_value", PythonObject(Float64(1.0)))
    py_setitem(cf, "constant_value_bounds", Python.list(pair_bounds))
    var rbf = family_call(b.object.__new__, classes["rbf"])
    var rf = py_fields(rbf)
    py_setitem(rf, "length_scale", Python.list(PythonObject(Float64(1.0))))
    py_setitem(rf, "length_scale_bounds", Python.list(pair_bounds))
    var pair = family_call(b.object.__new__, classes["combined"])
    var pf = py_fields(pair)
    py_setitem(pf, "a", const)
    py_setitem(pf, "b", rbf)
    py_setitem(pf, "op", PythonObject(_K_PROD))
    py_setitem(pf, "sym", PythonObject("*"))
    return pair


def _gpc_equal(value: PythonObject, text: String) raises -> Bool:
    """`value == text`, Python's comparison."""
    var b = Python.import_module("builtins")
    return Bool(py=family_call(b.bool, family_call(Python.import_module("operator").eq, value, PythonObject(text))))


def gpc_validate(model: PythonObject, context: PythonObject) raises:
    """`GaussianProcessClassifier._validate_params`."""
    var b = Python.import_module("builtins")
    var fields = py_fields(model)
    var kernel = family_field(fields, "kernel")
    var optimizer = family_field(fields, "optimizer")
    var n_restarts = family_field(fields, "n_restarts_optimizer")
    var max_iter = family_field(fields, "max_iter_predict")
    var warm_start = family_field(fields, "warm_start")
    var copy_x = family_field(fields, "copy_X_train")
    var multi_class = family_field(fields, "multi_class")
    var n_jobs = family_field(fields, "n_jobs")
    if kernel is None:
        kernel = _gpc_default_kernel(context)
    if not family_isinstance(kernel, context["family_gp_kernel_base"]):
        raise Error("TypeError: " + String(_NAME) + ": kernel must be a composition of mojolearn's RBF, "
                    + "Matern, ConstantKernel and WhiteKernel (DEVIATION 1761)")
    if not (optimizer is None or (family_is_str(optimizer) and String(family_call(optimizer.lower)) == "none")):
        raise Error("NotImplementedError: " + String(_NAME) + ": optimizer=" + family_repr(optimizer) + " is "
                    + "refused; only None runs "
                    + "(DEVIATION 1761). An optimizer's iteration count is data "
                    + "dependent and its line search, gradient fold and tolerance "
                    + "are not pinned. scikit-learn's default is 'fmin_l_bfgs_b', "
                    + "so this default DIFFERS from theirs and the fitted kernel_ "
                    + "is the kernel you passed")
    if family_isinstance(n_restarts, b.bool) or Int(py=family_call(b.int, n_restarts)) != 0:
        raise Error("NotImplementedError: " + String(_NAME) + ": n_restarts_optimizer=" + family_repr(n_restarts)
                    + " is refused; it restarts the optimizer, which is refused "
                    + "(DEVIATION 1761)")
    if family_isinstance(max_iter, b.bool) or not family_isinstance(max_iter, b.int):
        raise Error("TypeError: " + String(_NAME) + ": max_iter_predict must be an int, got " + family_repr(max_iter))
    if Bool(py=py_call(py_attr(b.int, "__lt__"), max_iter, PythonObject(1))):
        raise Error(String(_NAME) + ": max_iter_predict must be at least 1, got " + String(py_call(b.str, max_iter)))
    if Bool(py=family_call(b.bool, warm_start)):
        raise Error("NotImplementedError: " + String(_NAME) + ": warm_start=True is refused. It starts the Newton "
                    + "loop from the previous fit's latent values (_gpc.py:454-459), "
                    + "so a second fit would depend on the first; every fit here "
                    + "starts from zero")
    if not Bool(py=family_call(b.bool, copy_x)):
        raise Error("NotImplementedError: " + String(_NAME) + ": copy_X_train=False is refused; X always crosses to "
                    + "float32 C-order memory, so there is no reference to keep")
    if _gpc_equal(multi_class, "one_vs_one"):
        raise Error("NotImplementedError: " + String(_NAME) + ": multi_class='one_vs_one' is refused BY NAME; it is "
                    + "not implemented (and the reference gives it no "
                    + "predict_proba). 'one_vs_rest' runs (DEVIATION 2833)")
    if not _gpc_equal(multi_class, "one_vs_rest"):
        raise Error(String(_NAME) + ": multi_class must be 'one_vs_rest' or 'one_vs_one', "
                    + "got " + family_repr(multi_class))
    if n_jobs is not None:
        raise Error("NotImplementedError: " + String(_NAME) + ": n_jobs=" + family_repr(n_jobs) + " is refused; the one-vs-rest "
                    + "binary fits run one after another on one device")
    py_setitem(fields, "_kernel", kernel)
    py_setitem(fields, "_max_iter_predict", family_call(b.int, max_iter))


def _gpc_binary(context: PythonObject, y01: PythonObject, l_out: PythonObject, pi: PythonObject,
                wsr: PythonObject, lml: Float64, n_iter: Int, nb: Int) raises -> PythonObject:
    """A `_BinaryLaplace(y_train, L, pi, W_sr, lml, n_iter, nb)`, its fields set."""
    var b = Python.import_module("builtins")
    var fit = family_call(b.object.__new__, context["family_gp_binary_laplace"])
    var f = py_fields(fit)
    py_setitem(f, "y_train_", y01)
    py_setitem(f, "L_", l_out)
    py_setitem(f, "pi_", pi)
    py_setitem(f, "W_sr_", wsr)
    py_setitem(f, "log_marginal_likelihood_value_", PythonObject(lml))
    py_setitem(f, "n_iter_", PythonObject(n_iter))
    py_setitem(f, "nb_", PythonObject(nb))
    return fit


def _gpc_fit_all(ext: PythonObject, x: PythonObject, codes: PythonObject, arrays: PythonObject,
                 max_iter: PythonObject, columns: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_fit_all`: every binary fit in one `gpc_fit_all` call."""
    var n_rows = Int(py=x.shape[0])
    var n_cols = Int(py=x.shape[1])
    var outs = Python.list()
    # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpc_fit_all_binding.
    var addrs = Python.list(classical_address(x), classical_address(codes), classical_address(arrays[0]),
                            classical_address(arrays[1]), classical_address(arrays[2]), classical_address(arrays[3]))
    for _ in range(len(columns)):  # driver-loop: over the one-vs-rest binary fits
        var y01 = family_array(Python.tuple(n_rows), "<f4", context)
        var l_out = family_array(Python.tuple(n_rows, n_rows), "<f4", context)
        var pi = family_array(Python.tuple(n_rows), "<f4", context)
        var wsr = family_array(Python.tuple(n_rows), "<f4", context)
        var scalars = family_array(Python.tuple(3), "<f8", context)
        _ = py_call(outs.append, Python.tuple(y01, l_out, pi, wsr, scalars))
        _ = py_call(addrs.extend, Python.list(classical_address(y01), classical_address(l_out), classical_address(pi),
                                              classical_address(wsr), classical_address(scalars)))
        _ = pi  # keep the Python object alive through the native call above
        _ = l_out  # keep the Python object alive through the native call above
        _ = y01  # keep the Python object alive through the native call above
        _ = scalars  # keep the Python object alive through the native call above
        _ = wsr  # keep the Python object alive through the native call above
    # n_train, n_features, n_nodes, n_ls, max_iter_predict, n_fits, then the classes
    var params = Python.list(PythonObject(n_rows), PythonObject(n_cols), PythonObject(Int(py=arrays[0].shape[0])), arrays[4],
                             max_iter, PythonObject(len(columns)))
    _ = py_call(params.extend, columns)
    _ = family_call(gp_entry(ext, "gpc_fit_all"), addrs, params)
    var fits = Python.list()
    for i in range(len(outs)):  # driver-loop: over the one-vs-rest binary fits
        var o = outs[i]
        var sp = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(o[4])))
        _ = py_call(fits.append, _gpc_binary(context, o[0], o[1], o[2], o[3], sp[0], Int(sp[1]), Int(sp[2])))
    return fits


def _gpc_fit_binary(ext: PythonObject, x: PythonObject, codes: PythonObject, arrays: PythonObject,
                    max_iter: PythonObject, k: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_fit_binary(..., k=k)`: one `gpc_fit`, the 0/1 targets `code == k`
    built and written back by the binding."""
    var b = Python.import_module("builtins")
    var n_rows = Int(py=x.shape[0])
    var n_cols = Int(py=x.shape[1])
    var y01 = family_array(Python.tuple(n_rows), "<f4", context)
    var l_out = family_array(Python.tuple(n_rows, n_rows), "<f4", context)
    var pi = family_array(Python.tuple(n_rows), "<f4", context)
    var wsr = family_array(Python.tuple(n_rows), "<f4", context)
    # lml, n_iter, nb, in that order.
    var scalars = family_array(Python.tuple(3), "<f8", context)
    # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpc_fit_binding.
    # x, y (the codes), kinds, kparams, ls_len, ls, l_out, pi_out, wsr_out, scalars_out, y_out;
    # n_train, n_features, n_nodes, n_ls, max_iter_predict, k
    var n_iter = family_call(gp_entry(ext, "gpc_fit"),
        Python.list(classical_address(x), classical_address(codes), classical_address(arrays[0]),
                    classical_address(arrays[1]), classical_address(arrays[2]), classical_address(arrays[3]),
                    classical_address(l_out), classical_address(pi), classical_address(wsr),
                    classical_address(scalars), classical_address(y01)),
        Python.list(PythonObject(n_rows), PythonObject(n_cols), PythonObject(Int(py=arrays[0].shape[0])), arrays[4], max_iter, k))
    _ = wsr  # keep the Python object alive through the native call above
    _ = pi  # keep the Python object alive through the native call above
    _ = l_out  # keep the Python object alive through the native call above
    _ = scalars  # keep the Python object alive through the native call above
    _ = y01  # keep the Python object alive through the native call above
    var sp = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(scalars)))
    _ = scalars  # keep the Python object alive through the native call above
    var _keep_ret_scalars_1 = _gpc_binary(context, y01, l_out, pi, wsr, sp[0], Int(py=family_call(b.int, n_iter)), Int(sp[2]))
    _ = scalars  # keep the Python object alive through the native call above
    return _keep_ret_scalars_1^


def _gpc_set_fitted(fields: PythonObject, x: PythonObject, classes: PythonObject, fits: PythonObject,
                    context: PythonObject) raises:
    """`_set_fitted`."""
    var b = Python.import_module("builtins")
    var n_classes = len(classes)
    py_setitem(fields, "X_train_", x)
    py_setitem(fields, "n_features_in_", PythonObject(Int(py=x.shape[1])))
    py_setitem(fields, "classes_", py_call(b.list, classes))
    py_setitem(fields, "n_classes_", PythonObject(n_classes))
    py_setitem(fields, "estimators_", fits)
    py_setitem(fields, "kernel_", fields["_kernel"])
    if n_classes == 2:
        var only = py_fields(fits[0])
        py_setitem(fields, "y_train_", only["y_train_"])
        py_setitem(fields, "L_", only["L_"])
        py_setitem(fields, "pi_", only["pi_"])
        py_setitem(fields, "W_sr_", only["W_sr_"])
        py_setitem(fields, "n_iter_", only["n_iter_"])
        py_setitem(fields, "log_marginal_likelihood_value_", only["log_marginal_likelihood_value_"])
        return
    var iters = Python.list()
    var k = len(fits)
    var values = family_array(Python.tuple(k), "<f8", context)
    var vp = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(values)))
    _ = values  # keep the Python object alive through the native call above
    for i in range(k):  # driver-loop: over the one-vs-rest classes
        var own = py_fields(fits[i])
        _ = py_call(iters.append, own["n_iter_"])
        vp[i] = Float64(py=family_call(b.float, own["log_marginal_likelihood_value_"]))
        _ = values  # keep the Python object alive through the native call above
    py_setitem(fields, "n_iter_", iters)
    # the per-class values summed in storage order by the native reduction
    # (`Array.sum`'s reduce_stat), then divided by the class count
    var total = reduce_stat(Int(py=classical_address(values)), HP_F64, k, HP_SUM).f
    _ = values  # keep the Python object alive through the native call above
    py_setitem(fields, "log_marginal_likelihood_value_", PythonObject(total / Float64(k)))
    _ = values  # keep the Python object alive through the native call above
    _ = values^


def _gpc_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises:
    """`fit(X, y)`; args = (binding, base binding, X, y)."""
    classical_connect_numpy(context)
    gpc_validate(model, context)
    var fields = py_fields(model)
    var source = family_as_f32(args[2], 2, "X", context)
    py_setitem(fields, "input_copied_", PythonObject(not (source is args[2])))
    var x = family_owned(source, context)
    _ = source^
    var n_rows = Int(py=x.shape[0])
    var encoded = gp_encode_labels(args[3], args[1], context)
    var classes = encoded[0]
    var codes = encoded[1]
    var n_codes = Int(py=codes.size)
    if n_codes != n_rows:
        raise Error(String(_NAME) + ": y has " + String(n_codes) + " entries, X has " + String(n_rows) + " rows")
    var n_classes = len(classes)
    if n_classes == 1:
        raise Error(String(_NAME) + " requires 2 or more distinct classes; got 1 class "
                    + "(only class " + family_repr(classes[0]) + " is present)")
    var arrays = gp_kernel_arrays(fields["_kernel"], context)
    var ext = _gpc_ext(model, args[0], context)
    var columns = Python.list()
    if n_classes == 2:
        _ = py_call(columns.append, PythonObject(1))
    else:
        for k in range(n_classes):  # driver-loop: over the one-vs-rest classes
            _ = py_call(columns.append, PythonObject(k))
    var max_iter = fields["_max_iter_predict"]
    var fits: PythonObject
    if (gp_caps(ext) & 1) == 1:
        fits = _gpc_fit_all(ext, x, codes, arrays, max_iter, columns, context)
    else:
        fits = Python.list()
        for i in range(len(columns)):  # driver-loop: one binding call per one-vs-rest class
            _ = py_call(fits.append, _gpc_fit_binary(ext, x, codes, arrays, max_iter, columns[i], context))
    _gpc_set_fitted(fields, x, classes, fits, context)
    _ = arrays^
    _ = codes^


def _gpc_query(fields: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_query(X)`."""
    if not py_contains(fields, PythonObject("estimators_")):
        raise Error(String(_NAME) + ": call fit() or load() first")
    var q = family_as_f32(source, 2, "X", context)
    var d = Int(py=q.shape[1])
    var fitted = Int(py=fields["n_features_in_"])
    if d != fitted:
        raise Error(String(_NAME) + ": X has " + String(d) + " features, fit saw " + String(fitted))
    return q


def _gpc_latent(ext: PythonObject, fields: PythonObject, est: PythonObject, q: PythonObject, want_proba: Bool,
                out_kind: Int, context: PythonObject) raises -> PythonObject:
    return _gpc_latent(ext, fields, est, q, want_proba, out_kind, context, PythonObject(None))


def _gpc_latent(ext: PythonObject, fields: PythonObject, est: PythonObject, q: PythonObject, want_proba: Bool,
                out_kind: Int, context: PythonObject, kernel_arrays: PythonObject) raises -> PythonObject:
    """`_latent(ext, est, q, want_proba, out_kind)`: (mean, var, proba[, out]).
    `kernel_arrays`: the kernel's packed arrays when the caller already has
    them (a multi-GPU shard worker, which holds no kernel object); else
    `gp_kernel_arrays(fields["_kernel"])`."""
    var n_star = Int(py=q.shape[0])
    var xt = gp_f32(fields, "X_train_", 2, context)
    var n_train = Int(py=xt.shape[0])
    var n_features = Int(py=xt.shape[1])
    var arrays = kernel_arrays
    if arrays is None:
        arrays = gp_kernel_arrays(fields["_kernel"], context)
    var mean = family_array(Python.tuple(n_star), "<f4", context)
    var var_out = family_array(Python.tuple(n_star), "<f4", context)
    var proba = family_array(Python.tuple(n_star), "<f8", context)
    var out = PythonObject(None)
    if out_kind == 1:
        out = family_array(Python.tuple(n_star), "<i8", context)
    elif out_kind == 2:
        out = family_array(Python.tuple(n_star, 2), "<f8", context)
    var own = py_fields(est)
    var y01 = gp_f32(own, "y_train_", -1, context)
    var pi = gp_f32(own, "pi_", -1, context)
    var wsr = gp_f32(own, "W_sr_", -1, context)
    var lf = gp_f32(own, "L_", -1, context)
    # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpc_predict_binding.
    # xtrain, y, pi, wsr, l, xstar, kinds, kparams, ls_len, ls, mean_out, var_out, proba_out (+ out);
    # n_train, n_features, n_star, n_nodes, n_ls, want_proba (+ out_kind)
    var addrs = Python.list(classical_address(xt), classical_address(y01), classical_address(pi),
                            classical_address(wsr), classical_address(lf), classical_address(q),
                            classical_address(arrays[0]), classical_address(arrays[1]), classical_address(arrays[2]),
                            classical_address(arrays[3]), classical_address(mean), classical_address(var_out),
                            classical_address(proba))
    _ = pi  # keep the Python object alive through the native call above
    _ = y01  # keep the Python object alive through the native call above
    _ = xt  # keep the Python object alive through the native call above
    _ = lf  # keep the Python object alive through the native call above
    _ = wsr  # keep the Python object alive through the native call above
    _ = var_out  # keep the Python object alive through the native call above
    _ = mean  # keep the Python object alive through the native call above
    _ = proba  # keep the Python object alive through the native call above
    var params = Python.list(PythonObject(n_train), PythonObject(n_features), PythonObject(n_star),
                             PythonObject(Int(py=arrays[0].shape[0])), arrays[4], PythonObject(1 if want_proba else 0))
    if out_kind != 0:
        _ = py_call(addrs.append, classical_address(out))
        _ = lf  # keep the Python object alive through the native call above
        _ = wsr  # keep the Python object alive through the native call above
        _ = pi  # keep the Python object alive through the native call above
        _ = y01  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
        _ = proba  # keep the Python object alive through the native call above
        _ = var_out  # keep the Python object alive through the native call above
        _ = mean  # keep the Python object alive through the native call above
        _ = xt  # keep the Python object alive through the native call above
        _ = py_call(params.append, PythonObject(out_kind))
    _ = family_call(gp_entry(ext, "gpc_predict"), addrs, params)
    _ = lf  # keep the Python object alive through the native call above
    _ = wsr  # keep the Python object alive through the native call above
    _ = pi  # keep the Python object alive through the native call above
    _ = y01  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = proba  # keep the Python object alive through the native call above
    _ = var_out  # keep the Python object alive through the native call above
    _ = mean  # keep the Python object alive through the native call above
    _ = xt  # keep the Python object alive through the native call above
    _ = xt^
    _ = y01^
    _ = pi^
    _ = wsr^
    _ = lf^
    _ = arrays^
    if out_kind != 0:
        return Python.tuple(mean, var_out, proba, out)
    return Python.tuple(mean, var_out, proba)


def _gpc_predict_all(ext: PythonObject, fields: PythonObject, q: PythonObject, out_kind: Int,
                     context: PythonObject) raises -> PythonObject:
    """`_predict_all`: every binary fit against one cross kernel in one
    `gpc_predict_all` session; int64 codes (1) or float64 rows (2)."""
    var n_star = Int(py=q.shape[0])
    var xt = gp_f32(fields, "X_train_", 2, context)
    var n_train = Int(py=xt.shape[0])
    var n_features = Int(py=xt.shape[1])
    var arrays = gp_kernel_arrays(fields["_kernel"], context)
    var fits = fields["estimators_"]
    var k = len(fits)
    var out: PythonObject
    if out_kind == 1:
        out = family_array(Python.tuple(n_star), "<i8", context)
    else:
        out = family_array(Python.tuple(n_star, max(k, 2)), "<f8", context)
    # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpc_predict_all_binding.
    var addrs = Python.list(classical_address(xt), classical_address(q), classical_address(arrays[0]),
                            classical_address(arrays[1]), classical_address(arrays[2]), classical_address(arrays[3]),
                            classical_address(out))
    _ = xt  # keep the Python object alive through the native call above
    var kept = Python.list()
    for i in range(k):  # driver-loop: four model addresses per binary fit
        var own = py_fields(fits[i])
        var parts = Python.list(gp_f32(own, "y_train_", -1, context), gp_f32(own, "pi_", -1, context),
                                gp_f32(own, "W_sr_", -1, context), gp_f32(own, "L_", -1, context))
        _ = py_call(kept.append, parts)
        for j in range(4):  # small-loop(model buffers: four per binary fit)
            _ = py_call(addrs.append, classical_address(parts[j]))
            _ = xt  # keep the Python object alive through the native call above
    # n_train, n_features, n_star, n_nodes, n_ls, out_kind, k
    _ = family_call(gp_entry(ext, "gpc_predict_all"), addrs,
                    Python.list(PythonObject(n_train), PythonObject(n_features), PythonObject(n_star),
                                PythonObject(Int(py=arrays[0].shape[0])), arrays[4], PythonObject(out_kind), PythonObject(k)))
    _ = xt  # keep the Python object alive through the native call above
    _ = kept^
    _ = xt^
    _ = arrays^
    return out


def _gpc_ovr_combine(ext: PythonObject, cols: PythonObject, n_star: Int, context: PythonObject) raises -> PythonObject:
    """`_ovr_combine`: (the (n_star, k) float64 probabilities, the int32 codes)."""
    var k = len(cols)
    var proba = family_array(Python.tuple(n_star, k), "<f8", context)
    var codes = family_array(Python.tuple(n_star), "<i4", context)
    # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpc_ovr_combine_binding.
    var addrs = Python.list(classical_address(proba), classical_address(codes))
    _ = codes  # keep the Python object alive through the native call above
    _ = proba  # keep the Python object alive through the native call above
    for i in range(k):  # driver-loop: one class column address per class
        _ = py_call(addrs.append, classical_address(cols[i]))
        _ = codes  # keep the Python object alive through the native call above
        _ = proba  # keep the Python object alive through the native call above
    _ = family_call(gp_entry(ext, "gpc_ovr_combine"), addrs, Python.list(PythonObject(n_star)))
    _ = codes  # keep the Python object alive through the native call above
    _ = proba  # keep the Python object alive through the native call above
    return Python.tuple(proba, codes)


def _gpc_columns(ext: PythonObject, fields: PythonObject, q: PythonObject, context: PythonObject) raises -> PythonObject:
    """The per-class probability columns (`_latent(...)[2]` per fit)."""
    var cols = Python.list()
    var fits = fields["estimators_"]
    for i in range(len(fits)):  # driver-loop: one latent column per class
        _ = py_call(cols.append, _gpc_latent(ext, fields, fits[i], q, True, 0, context)[2])
    return cols


def _gpc_predict_proba(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`predict_proba(X)`; args = (binding, X)."""
    classical_connect_numpy(context)
    var fields = py_fields(model)
    var q = _gpc_query(fields, args[1], context)
    var ext = _gpc_ext(model, args[0], context)
    if (gp_caps(ext) & 1) == 1:
        return _gpc_predict_all(ext, fields, q, 2, context)
    if Int(py=fields["n_classes_"]) == 2:
        # `[1 - p, p]` from the binding (`gpc_binary_out`)
        return _gpc_latent(ext, fields, fields["estimators_"][0], q, True, 2, context)[3]
    var cols = _gpc_columns(ext, fields, q, context)
    return _gpc_ovr_combine(ext, cols, Int(py=q.shape[0]), context)[0]


def _gpc_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`predict(X)`; args = (binding, base binding, X)."""
    classical_connect_numpy(context)
    var fields = py_fields(model)
    var q = _gpc_query(fields, args[2], context)
    var ext = _gpc_ext(model, args[0], context)
    var codes: PythonObject
    if (gp_caps(ext) & 1) == 1:
        codes = _gpc_predict_all(ext, fields, q, 1, context)
    elif Int(py=fields["n_classes_"]) == 2:
        # `mean > 0` codes from the binding (`gpc_binary_out`)
        codes = _gpc_latent(ext, fields, fields["estimators_"][0], q, False, 1, context)[3]
    else:
        var n_star = Int(py=q.shape[0])
        var cols = _gpc_columns(ext, fields, q, context)
        var codes32 = _gpc_ovr_combine(ext, cols, n_star, context)[1]
        # `codes32.astype("<i8")`: the exact widening
        codes = family_array(Python.tuple(n_star), "<i8", context)
        _ = cast_elements(Int(py=classical_address(codes32)), HP_I32, Int(py=classical_address(codes)), HP_I64, n_star)
        _ = codes32  # keep the Python object alive through the native call above
        _ = codes32^
    return gp_decode_labels(fields["classes_"], codes, args[1], context)


def _gpc_latent_mean_and_variance(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`latent_mean_and_variance(X)`; args = (binding, X)."""
    classical_connect_numpy(context)
    var fields = py_fields(model)
    var q = _gpc_query(fields, args[1], context)
    var n_classes = Int(py=fields["n_classes_"])
    if n_classes > 2:
        raise Error("Returning the mean and variance of the latent function f is "
                    + "only supported for binary classification, received "
                    + String(n_classes) + " classes.")
    var ext = _gpc_ext(model, args[0], context)
    var latent = _gpc_latent(ext, fields, fields["estimators_"][0], q, True, 0, context)
    return Python.tuple(latent[0], latent[1])


def gp_classifier_operation(model: PythonObject, operation: String, args: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    """GaussianProcessClassifier's public operations: args[0] is the resolved binding."""
    if operation == "family_gp_gpc_fit":
        _gpc_fit(model, args, context)
        return model
    if operation == "family_gp_gpc_predict":
        return _gpc_predict(model, args, context)
    if operation == "family_gp_gpc_predict_proba":
        return _gpc_predict_proba(model, args, context)
    if operation == "family_gp_gpc_latent":
        return _gpc_latent_mean_and_variance(model, args, context)
    if operation == "family_gp_gpc_score":
        # `score(X, y)`: predict, the row-count refusal, then
        # `accuracy_fraction(y, pred)`; args = (binding, base binding, X, y).
        var pred = _gpc_predict(model, Python.tuple(args[0], args[1], args[2]), context)
        var shape = mc_shape_of(args[3], context)
        var n_truth = 0
        if Int(py=py_call(Python.import_module("builtins").len, shape)) > 0:
            n_truth = Int(py=py_call(Python.import_module("builtins").int, shape[0]))
        var n_pred = Int(py=py_call(Python.import_module("builtins").len, pred))
        if n_truth != n_pred:
            raise Error("mojolearn GaussianProcessClassifier: y has " + String(n_truth) + " entries for "
                        + String(n_pred) + " rows")
        return mcls_accuracy_fraction(args[3], pred, PythonObject(None), PythonObject(None), context)
    raise Error("RuntimeError: unknown gaussian process classifier operation " + operation)

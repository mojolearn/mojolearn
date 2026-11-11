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
from core.gp_native_api import GPCFitRequest, GPCFitAllRequest, GPCQueryRequest, GPCQueryAllRequest, GPCCombineRequest
from core.native_label_codes import NativeLabelCodes
from core.gp_native_storage import GPCQueryScratch
from core.gp_native_storage import GPCMemberTables, GPCQueryTables
from bindings.gp_score_boundary import GPCPredictionCodes
from bindings.gp_native_boundary import GPConnection, gp_kernel_view, gp_run
from bindings.public_lifecycle_owner import lifecycle_admit_fields
from std.python import Python, PythonObject
from bindings.gp_argument_pins import GPArgumentPins
from bindings.admitted_family import admit_family
from bindings.native_loaded_module import native_loaded_module
from bindings.mode_metadata_native import _fields
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setitem
from bindings.py_attr import py_attr
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import (
    family_repr, family_is_str, family_isinstance, family_call, family_array, family_field, FAMILY_F64P,
)
from core.hotpath_impl import cast_elements, HP_I32, HP_I64
from core.gp_native_fit_summary import GPCFitSummary
from bindings.gp_common_native import gp_extension, gp_caps, gp_kernel_arrays, gp_f32, gp_fit_owned
from bindings.gp_labels_native import gp_encode_labels, gp_decode_labels
from bindings.gp_svm_gp_state import gp_kernel_arrays_reused, gp_x_known
from bindings.gp_query_x import GpQuery, gp_query, gp_query_address, gp_query_refuse, gp_query_scan
from bindings.fitted_resident_door import fitted_resident_drop
from bindings.gp_resident_boundary import gp_resident_lease
from bindings.gp_score_labels_boundary import gpc_score_result
from bindings.gp_labels_native import gp_decode_score_labels
from bindings.mcls_common import mc_shape_of
from bindings.host_inference_scalars import host_inference_scalar

comptime _NAME = "mojolearn GaussianProcessClassifier"
comptime _K_PROD = 5
# The prepared-owner slot of a classifier's device-resident fitted state.
comptime _GPC_RESIDENT_SLOT = "gpc_resident_fitted_v1"


def _gpc_ext(model: PythonObject, binding: PythonObject, context: PythonObject) raises -> PythonObject:
    return gp_extension(model, binding, _NAME, "rebuild it with bash bindings/build_gp.sh", context)


def _gpc_default_kernel(context: PythonObject) raises -> PythonObject:
    """`ConstantKernel(1.0) * RBF(1.0)` with scikit-learn's default bounds."""
    var b = native_loaded_module("builtins")
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
    """Compare an admitted native string option without importing operator."""
    return family_is_str(value) and String(value) == text


def gpc_validate(model: PythonObject, context: PythonObject) raises:
    """`GaussianProcessClassifier._validate_params`."""
    var b = native_loaded_module("builtins")
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


@fieldwise_init
struct GPCBinaryPublication(Movable):
    var member: PythonObject
    var iterations: PythonObject
    var likelihood: Float64


struct GPCFitPublication(Movable):
    # Members and iteration objects are the actual public output pins. The
    # likelihood summary stays native until the final public scalar PACK.
    var members: PythonObject
    var iterations: List[PythonObject]
    var summary: GPCFitSummary

    def __init__(out self, count: Int) raises:
        self.members = Python.list()
        self.iterations = List[PythonObject](capacity=count if count > 1 else 0)
        self.summary = GPCFitSummary(count if count > 1 else 0)

    def append(mut self, value: GPCBinaryPublication) raises:
        _ = py_call(self.members.append, value.member)
        if len(self.summary.values) > 0:
            self.summary.set(len(self.iterations), value.likelihood)
            self.iterations.append(value.iterations)


def _gpc_binary_result(context: PythonObject, y01: PythonObject, l_out: PythonObject, pi: PythonObject,
                wsr: PythonObject, lml: Float64, n_iter: Int, nb: Int) raises -> GPCBinaryPublication:
    """A `_BinaryLaplace(y_train, L, pi, W_sr, lml, n_iter, nb)`, its fields set."""
    var b = native_loaded_module("builtins")
    var fit = family_call(b.object.__new__, context["family_gp_binary_laplace"])
    var f = py_fields(fit)
    py_setitem(f, "y_train_", y01)
    py_setitem(f, "L_", l_out)
    py_setitem(f, "pi_", pi)
    py_setitem(f, "W_sr_", wsr)
    py_setitem(f, "log_marginal_likelihood_value_", PythonObject(lml))
    var iterations = PythonObject(n_iter)
    py_setitem(f, "n_iter_", iterations)
    py_setitem(f, "nb_", PythonObject(nb))
    return GPCBinaryPublication(fit, iterations, lml)



def _gpc_binary(context: PythonObject, y01: PythonObject, l_out: PythonObject, pi: PythonObject,
                wsr: PythonObject, lml: Float64, n_iter: Int, nb: Int) raises -> PythonObject:
    # Explicit worker/interpreter PACK compatibility: no opaque Mojo owner
    # crosses into another interpreter's transport.
    var result = _gpc_binary_result(context, y01, l_out, pi, wsr, lml, n_iter, nb)
    return result.member


@fieldwise_init
struct GPCMemberPins(Movable):
    var targets: PythonObject
    var factor: PythonObject
    var pi: PythonObject
    var wsr: PythonObject


@fieldwise_init
struct GPCLatentResult(Movable):
    var scratch: GPCQueryScratch
    var codes: List[Int64]
    var mean: PythonObject
    var variance: PythonObject
    var probability: PythonObject
    var output: PythonObject


@fieldwise_init
struct GPCCombinedResult(Movable):
    var probabilities: PythonObject
    var codes: NativeLabelCodes


@fieldwise_init
struct GPCPredictAllResult(Movable):
    var probabilities: PythonObject
    var codes: List[Int64]


def _gpc_fit_all(ext: PythonObject, x: PythonObject, codes: NativeLabelCodes, arrays: PythonObject,
                 max_iter: Int, columns: List[Int], context: PythonObject) raises -> GPCFitPublication:
    var n_rows = Int(py=x.shape[0])
    var n_cols = Int(py=x.shape[1])
    var pins = List[GPCMemberPins]()
    var tables = GPCMemberTables(len(columns))
    tables.prepare_scalars()
    for i in range(len(columns)):  # driver-loop: allocate each public OVR member's fitted arrays
        var y01 = family_array(Python.tuple(n_rows), "<f4", context)
        var factor = family_array(Python.tuple(n_rows, n_rows), "<f4", context)
        var pi = family_array(Python.tuple(n_rows), "<f4", context)
        var wsr = family_array(Python.tuple(n_rows), "<f4", context)
        tables.columns[i] = columns[i]
        tables.targets[i] = Int(py=classical_address(y01))
        tables.factors[i] = Int(py=classical_address(factor))
        tables.pis[i] = Int(py=classical_address(pi))
        tables.wsrs[i] = Int(py=classical_address(wsr))
        pins.append(GPCMemberPins(y01, factor, pi, wsr))
    var connection = GPConnection(ext)
    var kernel = gp_kernel_view(arrays)
    var request = GPCFitAllRequest(Int(py=classical_address(x)), codes.address(),
        Int(tables.columns.unsafe_ptr()), Int(tables.targets.unsafe_ptr()), Int(tables.factors.unsafe_ptr()),
        Int(tables.pis.unsafe_ptr()), Int(tables.wsrs.unsafe_ptr()), Int(tables.scalars.unsafe_ptr()),
        n_rows, n_cols, kernel^, max_iter, len(columns))
    _ = gp_run(connection, request)
    _ = x
    _ = codes
    _ = arrays
    _ = pins
    var fits = GPCFitPublication(len(pins))
    for i in range(len(pins)):  # driver-loop: pack the public mutable binary member objects
        var member = _gpc_binary_result(context, pins[i].targets, pins[i].factor, pins[i].pi, pins[i].wsr,
            tables.scalar_storage[3*i], Int(tables.scalar_storage[3*i+1]), Int(tables.scalar_storage[3*i+2]))
        fits.append(member)
    _ = tables
    _ = pins
    return fits^


def _gpc_fit_binary(ext: PythonObject, x: PythonObject, codes: NativeLabelCodes, arrays: PythonObject,
                    max_iter: Int, k: Int, context: PythonObject) raises -> GPCBinaryPublication:
    var answer = _gpc_fit_binary_address(ext, x, codes.address(), arrays, max_iter, k, context)
    _ = codes
    return answer^


def _gpc_fit_binary(ext: PythonObject, x: PythonObject, codes: PythonObject, arrays: PythonObject,
                    max_iter: Int, k: Int, context: PythonObject) raises -> PythonObject:
    # UNPACK of the existing worker transport Array.
    var answer = _gpc_fit_binary_address(ext, x, Int(py=classical_address(codes)), arrays, max_iter, k, context)
    _ = codes
    return answer.member


def _gpc_fit_binary_address(ext: PythonObject, x: PythonObject, codes: Int, arrays: PythonObject,
                    max_iter: Int, k: Int, context: PythonObject) raises -> GPCBinaryPublication:
    # The worker transport enters here too; unpack its exact public storage.
    var n_rows = Int(py=x.shape[0])
    var n_cols = Int(py=x.shape[1])
    var y01 = family_array(Python.tuple(n_rows), "<f4", context)
    var factor = family_array(Python.tuple(n_rows, n_rows), "<f4", context)
    var pi = family_array(Python.tuple(n_rows), "<f4", context)
    var wsr = family_array(Python.tuple(n_rows), "<f4", context)
    var scalars = List[Float64](length=3, fill=0.0)
    var connection = GPConnection(ext)
    var kernel = gp_kernel_view(arrays)
    var request = GPCFitRequest(Int(py=classical_address(x)), codes,
        Int(py=classical_address(factor)), Int(py=classical_address(pi)), Int(py=classical_address(wsr)),
        Int(scalars.unsafe_ptr()), n_rows, n_cols, kernel^, max_iter, True, k,
        Int(py=classical_address(y01)))
    var iterations = gp_run(connection, request)
    _ = x
    _ = codes
    _ = arrays
    _ = y01
    _ = factor
    _ = pi
    _ = wsr
    return _gpc_binary_result(context, y01, factor, pi, wsr, scalars[0], iterations, Int(scalars[2]))


def _gpc_set_fitted(fields: PythonObject, x: PythonObject, classes: PythonObject, fits: GPCFitPublication,
                    context: PythonObject) raises:
    """`_set_fitted`."""
    lifecycle_admit_fields(fields)
    var b = native_loaded_module("builtins")
    var n_classes = len(classes)
    py_setitem(fields, "X_train_", x)
    py_setitem(fields, "n_features_in_", PythonObject(Int(py=x.shape[1])))
    py_setitem(fields, "classes_", py_call(b.list, classes))
    py_setitem(fields, "n_classes_", PythonObject(n_classes))
    py_setitem(fields, "estimators_", fits.members)
    py_setitem(fields, "kernel_", fields["_kernel"])
    if n_classes == 2:
        var only = py_fields(fits.members[0])
        py_setitem(fields, "y_train_", only["y_train_"])
        py_setitem(fields, "L_", only["L_"])
        py_setitem(fields, "pi_", only["pi_"])
        py_setitem(fields, "W_sr_", only["W_sr_"])
        py_setitem(fields, "n_iter_", only["n_iter_"])
        py_setitem(fields, "log_marginal_likelihood_value_", only["log_marginal_likelihood_value_"])
        return
    var iters = Python.list()
    for i in range(len(fits.iterations)):  # driver-loop: public n_iter_ retains each member's exact scalar object
        _ = py_call(iters.append, fits.iterations[i])
    py_setitem(fields, "n_iter_", iters)
    py_setitem(fields, "log_marginal_likelihood_value_", PythonObject(fits.summary.likelihood()))


def _gpc_fit(model: PythonObject, args: GPArgumentPins, context: PythonObject) raises:
    """`fit(X, y)`; args = (binding, base binding, X, y)."""
    classical_connect_numpy(context)
    gpc_validate(model, context)
    var fields = py_fields(model)
    var source = admit_family(args[2], 2, "X", context).array
    py_setitem(fields, "input_copied_", PythonObject(not (source is args[2])))
    var x = gp_fit_owned(source, args[2], context)
    _ = source^
    var n_rows = Int(py=x.shape[0])
    var encoded = gp_encode_labels(args[3], args[1], context)
    var classes = encoded.classes
    var codes = NativeLabelCodes(0)
    swap(codes, encoded.codes)
    var n_codes = codes.count()
    if n_codes != n_rows:
        raise Error(String(_NAME) + ": y has " + String(n_codes) + " entries, X has " + String(n_rows) + " rows")
    var n_classes = len(classes)
    if n_classes == 1:
        raise Error(String(_NAME) + " requires 2 or more distinct classes; got 1 class "
                    + "(only class " + family_repr(classes[0]) + " is present)")
    var arrays = gp_kernel_arrays(fields["_kernel"], context)
    var ext = _gpc_ext(model, args[0], context)
    var columns = List[Int]()
    if n_classes == 2:
        columns.append(1)
    else:
        for k in range(n_classes):  # driver-loop: over the one-vs-rest classes
            columns.append(k)
    var max_iter = Int(py=fields["_max_iter_predict"])
    var fits: GPCFitPublication
    if (gp_caps(ext) & 1) == 1:
        fits = _gpc_fit_all(ext, x, codes, arrays, max_iter, columns, context)
    else:
        fits = GPCFitPublication(len(columns))
        for i in range(len(columns)):  # driver-loop: one binding call per one-vs-rest class
            var member = _gpc_fit_binary(ext, x, codes, arrays, max_iter, columns[i], context)
            fits.append(member)
    _gpc_set_fitted(fields, x, classes, fits, context)
    _ = arrays^
    _ = codes^


def _gpc_query(fields: PythonObject, source: PythonObject, context: PythonObject,
               entry_tests: Bool = False) raises -> GpQuery:
    """`_query(X)` as a GpQuery (bindings/gp_query_x.mojo; lane a7a):
    `entry_tests` (the gpc_predict_all entries, which return the query's
    status words) keeps a cross-validation fold on the device; any other
    route tests the query first (gp_query_scan)."""
    if not py_contains(fields, PythonObject("estimators_")):
        raise Error(String(_NAME) + ": call fit() or load() first")
    # The public door already admitted this X as C-order float32 and
    # registered it (bindings/gp_svm_records.mojo): no second snapshot.
    var q = gp_query(source, "GaussianProcessClassifier", context, entry_tests)
    var fitted = Int(py=fields["n_features_in_"])
    if q.cols != fitted:
        raise Error(String(_NAME) + ": X has " + String(q.cols) + " features, fit saw " + String(fitted))
    if not entry_tests:
        gp_query_scan(q, "GaussianProcessClassifier", context)
    return q^


def _gpc_latent(ext: PythonObject, fields: PythonObject, est: PythonObject, q: PythonObject, want_proba: Bool,
                out_kind: Int, context: PythonObject) raises -> GPCLatentResult:
    return _gpc_latent(ext, fields, est, q, want_proba, out_kind, context, PythonObject(None))


def _gpc_latent(ext: PythonObject, fields: PythonObject, est: PythonObject, q: PythonObject, want_proba: Bool,
                out_kind: Int, context: PythonObject, kernel_arrays: PythonObject, pack: Int = 7, native_codes: Bool = False) raises -> GPCLatentResult:
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
    var exposed = pack if out_kind == 0 else 0
    var scratch = GPCQueryScratch(n_star, exposed)
    var mean = family_array(Python.tuple(n_star), "<f4", context) if (exposed & 1) != 0 else PythonObject(None)
    var var_out = family_array(Python.tuple(n_star), "<f4", context) if (exposed & 2) != 0 else PythonObject(None)
    var proba = family_array(Python.tuple(n_star), "<f8", context) if (exposed & 4) != 0 else PythonObject(None)
    var dst = PythonObject(None)
    var codes = List[Int64]()
    if out_kind == 1:
        if native_codes:
            codes = List[Int64](length=n_star, fill=0)
        else:
            dst = family_array(Python.tuple(n_star), "<i8", context)
    elif out_kind == 2:
        dst = family_array(Python.tuple(n_star, 2), "<f8", context)
    var own = py_fields(est)
    var y01 = gp_f32(own, "y_train_", -1, context)
    var pi = gp_f32(own, "pi_", -1, context)
    var wsr = gp_f32(own, "W_sr_", -1, context)
    var lf = gp_f32(own, "L_", -1, context)
    var connection = GPConnection(ext)
    var kernel = gp_kernel_view(arrays)
    var output_address = 0
    if out_kind != 0:
        output_address = Int(codes.unsafe_ptr()) if native_codes and out_kind == 1 else Int(py=classical_address(dst))
    var mean_address = Int(py=classical_address(mean)) if (exposed & 1) != 0 else Int(scratch.mean.unsafe_ptr())
    var variance_address = Int(py=classical_address(var_out)) if (exposed & 2) != 0 else Int(scratch.variance.unsafe_ptr())
    var probability_address = Int(py=classical_address(proba)) if (exposed & 4) != 0 else Int(scratch.probability.unsafe_ptr())
    var request = GPCQueryRequest(Int(py=classical_address(xt)), Int(py=classical_address(y01)),
        Int(py=classical_address(pi)), Int(py=classical_address(wsr)), Int(py=classical_address(lf)),
        Int(py=classical_address(q)), mean_address, variance_address,
        probability_address, n_train, n_features, n_star, kernel^, want_proba, out_kind, output_address)
    _ = gp_run(connection, request)
    lifecycle_admit_fields(fields)
    lifecycle_admit_fields(own)
    _ = mean
    _ = var_out
    _ = proba
    _ = dst
    _ = q
    _ = xt^
    _ = y01^
    _ = pi^
    _ = wsr^
    _ = lf^
    _ = arrays^
    return GPCLatentResult(scratch^, codes^, mean, var_out, proba, dst)


def _gpc_predict_all(model: PythonObject, ext: PythonObject, fields: PythonObject, gq: GpQuery, out_kind: Int,
                     context: PythonObject, kernel_arrays: PythonObject) raises -> GPCPredictAllResult:
    """`_predict_all`: every binary fit against one cross kernel in one
    `gpc_predict_all` session; int64 codes (1) or float64 rows (2).
    `kernel_arrays`: `gp_kernel_arrays(fields["_kernel"])`. The query `gq`
    is host words or a device fold; the entry returns its status words and
    a flagged query is refused here (lane a7a)."""
    var n_star = gq.rows
    var xt = gp_f32(fields, "X_train_", 2, context)
    var n_train = Int(py=xt.shape[0])
    var n_features = Int(py=xt.shape[1])
    var arrays = kernel_arrays
    var fits = fields["estimators_"]
    var k = len(fits)
    var probabilities = PythonObject(None)
    var codes = List[Int64]()
    var destination: Int
    if out_kind == 1:
        codes = List[Int64](length=n_star, fill=0)
        destination = Int(codes.unsafe_ptr())
    else:
        probabilities = family_array(Python.tuple(n_star, max(k, 2)), "<f8", context)
        destination = Int(py=classical_address(probabilities))
    var tables = GPCQueryTables(k)
    var kept = List[PythonObject]()
    var mirrors = List[PythonObject]()
    var can_mirror = xt is fields["X_train_"]
    if can_mirror:
        mirrors.append(xt)
    var names = Python.tuple("y_train_", "pi_", "W_sr_", "L_")
    for i in range(k):  # driver-loop: four model buffers per fitted OVR member
        var own = py_fields(fits[i])
        var parts = List[PythonObject]()
        parts.append(gp_f32(own, "y_train_", -1, context))
        parts.append(gp_f32(own, "pi_", -1, context))
        parts.append(gp_f32(own, "W_sr_", -1, context))
        parts.append(gp_f32(own, "L_", -1, context))
        tables.targets[i] = Int(py=classical_address(parts[0]))
        tables.pis[i] = Int(py=classical_address(parts[1]))
        tables.wsrs[i] = Int(py=classical_address(parts[2]))
        tables.factors[i] = Int(py=classical_address(parts[3]))
        for j in range(4):  # driver-loop: retain each member buffer through native execution
            kept.append(parts[j])
            if can_mirror:
                if parts[j] is own[names[j]]:
                    mirrors.append(parts[j])
                else:
                    can_mirror = False
    var connection = GPConnection(ext)
    var lease = -1
    if (connection.capabilities & 4) == 4:
        lease = 0
        if can_mirror:
            lease = gp_resident_lease(model, _GPC_RESIDENT_SLOT, connection, mirrors)
        else:
            fitted_resident_drop(model, _GPC_RESIDENT_SLOT)
    var kernel = gp_kernel_view(arrays)
    var request = GPCQueryAllRequest(Int(py=classical_address(xt)), Int(py=gp_query_address(gq)),
        destination, Int(tables.targets.unsafe_ptr()), Int(tables.pis.unsafe_ptr()),
        Int(tables.wsrs.unsafe_ptr()), Int(tables.factors.unsafe_ptr()), n_train, n_features, n_star,
        kernel^, out_kind, k, lease, gq.store.base_id if gq.store.active() else -1,
        gq.store.base_rows if gq.store.active() else 0, gq.store.index_id if gq.store.active() else -1)
    var answer = gp_run(connection, request)
    lifecycle_admit_fields(fields)
    gp_query_refuse(answer.finite, answer.range_status, gq, "GaussianProcessClassifier")
    _ = tables
    _ = kept
    _ = xt
    _ = arrays
    _ = gq
    return GPCPredictAllResult(probabilities, codes^)


def _gpc_combine_addresses(ext: PythonObject, addresses: List[Int], n_star: Int,
                           context: PythonObject, public_probabilities: Bool = True) raises -> GPCCombinedResult:
    var k = len(addresses)
    var proba = PythonObject(None)
    var private_proba = List[Float64]()
    var probability_address: Int
    if public_probabilities:
        proba = family_array(Python.tuple(n_star, k), "<f8", context)
        probability_address = Int(py=classical_address(proba))
    else:
        private_proba = List[Float64](length=n_star*k, fill=0.0)
        probability_address = Int(private_proba.unsafe_ptr())
    var codes = NativeLabelCodes(n_star)
    var connection = GPConnection(ext)
    var request = GPCCombineRequest(Int(addresses.unsafe_ptr()), probability_address,
        codes.address(), n_star, k)
    _ = gp_run(connection, request)
    _ = addresses
    _ = proba
    _ = codes
    _ = private_proba
    return GPCCombinedResult(proba, codes^)


def _gpc_ovr_combine(ext: PythonObject, cols: List[PythonObject], n_star: Int,
                     context: PythonObject, public_probabilities: Bool = True) raises -> GPCCombinedResult:
    # UNPACK of cross-interpreter Array transport.
    var addresses = List[Int](length=len(cols), fill=0)
    for i in range(len(cols)):  # driver-loop: one address per transported class column
        addresses[i] = Int(py=classical_address(cols[i]))
    var answer = _gpc_combine_addresses(ext, addresses, n_star, context, public_probabilities)
    _ = cols
    return answer^


def _gpc_ovr_combine(ext: PythonObject, cols: List[List[Float64]], n_star: Int,
                     context: PythonObject, public_probabilities: Bool = True) raises -> GPCCombinedResult:
    var addresses = List[Int](length=len(cols), fill=0)
    for i in range(len(cols)):  # driver-loop: one native probability span per class
        addresses[i] = Int(cols[i].unsafe_ptr())
    var answer = _gpc_combine_addresses(ext, addresses, n_star, context, public_probabilities)
    _ = cols
    return answer^


def _gpc_columns(ext: PythonObject, fields: PythonObject, q: PythonObject, context: PythonObject,
                 kernel_arrays: PythonObject) raises -> List[List[Float64]]:
    var cols = List[List[Float64]]()
    var fits = fields["estimators_"]
    for i in range(len(fits)):  # driver-loop: preserve the existing non-fused OVR route
        var latent = _gpc_latent(ext, fields, fits[i], q, True, 0, context, kernel_arrays, 0)
        var probability = List[Float64]()
        swap(probability, latent.scratch.probability)
        cols.append(probability^)
    return cols^


def _gpc_predict_proba(model: PythonObject, args: GPArgumentPins, context: PythonObject) raises -> PythonObject:
    """`predict_proba(X)`; args = (binding, X)."""
    classical_connect_numpy(context)
    var fields = py_fields(model)
    var ext = _gpc_ext(model, args[0], context)
    var all_entry = (gp_caps(ext) & 1) == 1
    var gq = _gpc_query(fields, args[1], context, all_entry)
    # Packed once per kernel state of this model (bindings/gp_svm_records.mojo).
    var arrays = gp_kernel_arrays_reused(model, fields["_kernel"], context)
    if all_entry:
        return _gpc_predict_all(model, ext, fields, gq, 2, context, arrays).probabilities
    var q = gq.q
    if Int(py=fields["n_classes_"]) == 2:
        # `[1 - p, p]` from the binding (`gpc_binary_out`)
        return _gpc_latent(ext, fields, fields["estimators_"][0], q, True, 2, context, arrays).output
    var cols = _gpc_columns(ext, fields, q, context, arrays)
    return _gpc_ovr_combine(ext, cols, Int(py=q.shape[0]), context).probabilities


def _gpc_predict(model: PythonObject, args: GPArgumentPins, context: PythonObject) raises -> PythonObject:
    """Public prediction packing; arguments are (binding, base binding, X)."""
    var result = _gpc_prediction_codes(model, args[0], args[2], context)
    return gp_decode_labels(model, result.classes, result.codes, args[1], context)


def _gpc_prediction_codes(model: PythonObject, binding: PythonObject,
                          source: PythonObject, context: PythonObject) raises -> GPCPredictionCodes:
    """The existing prediction routes with typed codes and retained class pins."""
    classical_connect_numpy(context)
    var fields = py_fields(model)
    var ext = _gpc_ext(model, binding, context)
    var all_entry = (gp_caps(ext) & 1) == 1
    var gq = _gpc_query(fields, source, context, all_entry)
    var q = gq.q
    # Packed once per kernel state of this model (bindings/gp_svm_records.mojo).
    var arrays = gp_kernel_arrays_reused(model, fields["_kernel"], context)
    var codes = List[Int64]()
    if all_entry:
        var result = _gpc_predict_all(model, ext, fields, gq, 1, context, arrays)
        swap(codes, result.codes)
    elif Int(py=fields["n_classes_"]) == 2:
        # `mean > 0` codes from the binding (`gpc_binary_out`)
        var result = _gpc_latent(ext, fields, fields["estimators_"][0], q, False, 1, context, arrays, 0, True)
        swap(codes, result.codes)
    else:
        var n_star = Int(py=q.shape[0])
        var cols = _gpc_columns(ext, fields, q, context, arrays)
        var combined = _gpc_ovr_combine(ext, cols, n_star, context, False)
        var codes32 = NativeLabelCodes(0)
        swap(codes32, combined.codes)
        # `codes32.astype("<i8")`: the exact widening
        codes = List[Int64](length=n_star, fill=0)
        _ = cast_elements(codes32.address(), HP_I32, Int(codes.unsafe_ptr()), HP_I64, n_star)
        _ = codes32  # keep the Python object alive through the native call above
        _ = codes32^
    return GPCPredictionCodes(codes^, fields["classes_"])


def _gpc_latent_mean_and_variance(model: PythonObject, args: GPArgumentPins, context: PythonObject) raises -> PythonObject:
    """`latent_mean_and_variance(X)`; args = (binding, X)."""
    classical_connect_numpy(context)
    var fields = py_fields(model)
    var q = _gpc_query(fields, args[1], context).q
    var n_classes = Int(py=fields["n_classes_"])
    if n_classes > 2:
        raise Error("Returning the mean and variance of the latent function f is "
                    + "only supported for binary classification, received "
                    + String(n_classes) + " classes.")
    var ext = _gpc_ext(model, args[0], context)
    var latent = _gpc_latent(ext, fields, fields["estimators_"][0], q, True, 0, context,
                             gp_kernel_arrays_reused(model, fields["_kernel"], context), 3)
    return Python.tuple(latent.mean, latent.variance)


def gp_classifier_operation(model: PythonObject, operation: String, args: GPArgumentPins,
                            context: PythonObject) raises -> PythonObject:
    """GaussianProcessClassifier's public operations: args[0] is the resolved binding."""
    if operation == "family_gp_gpc_lml":
        # The complete public operation: refusal order, fitted state and final
        # scalar are native. Admit truth values before any boolean protocol.
        if args[0] is not None:
            raise Error("NotImplementedError: " + String(_NAME) + ": log_marginal_likelihood(theta) at other "
                        + "hyperparameters is refused; it serves the optimizer, which is "
                        + "refused (DEVIATION 1761)")
        classical_connect_numpy(context)
        host_inference_scalar(args[1], context, True)
        if Bool(py=args[1]):
            raise Error("Gradient can only be evaluated for theta!=None")
        var fields = _fields(model)
        if not py_contains(fields, "estimators_"):
            raise Error(String(_NAME) + ": call fit() first")
        # Return the stored native scalar without calling a descriptor or
        # conversion protocol; clone_kernel remains accepted and unused.
        var result = fields["log_marginal_likelihood_value_"]
        host_inference_scalar(result, context)
        return result
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
        var prediction = _gpc_prediction_codes(model, args[0], args[2], context)
        var pred = gp_decode_score_labels(model, prediction.classes, prediction.codes, args[1], context)
        var shape = mc_shape_of(args[3], context)  # admits y once; the accuracy pair reuses it
        var n_truth = 0
        if Int(py=py_call(native_loaded_module("builtins").len, shape)) > 0:
            n_truth = Int(py=py_call(native_loaded_module("builtins").int, shape[0]))
        var n_pred = pred.rows
        if n_truth != n_pred:
            raise Error("mojolearn GaussianProcessClassifier: y has " + String(n_truth) + " entries for "
                        + String(n_pred) + " rows")
        var result = gpc_score_result(args[3], pred, context)
        _ = prediction
        _ = pred
        return PythonObject(result)
    raise Error("RuntimeError: unknown gaussian process classifier operation " + operation)


def gp_classifier_operation(model: PythonObject, operation: String, args: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    # Compatibility UNPACK for internal archive/model-selection callers.
    # Public GP lifecycle already owns these slots in its native call record.
    var inputs = GPArgumentPins(args)
    return gp_classifier_operation(model, operation, inputs, context)

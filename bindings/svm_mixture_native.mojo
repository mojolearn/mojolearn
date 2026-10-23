# SPDX-License-Identifier: Apache-2.0
# LinearSVC `decision_function`, LinearSVR `predict` and GaussianMixture
# `sample`, one native call each (python/mojolearn/svm.py, mixture.py). The
# shell hands over the binding `_bind` resolved; this owner makes the Python
# methods' refusals, allocates the outputs and runs the binding's
# `qn_decision_function` / `gmm_sample` with the argument lists their
# bindings document. Same entries, same inputs: the bits do not move.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_fields
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import (
    family_repr, family_is_int, family_isinstance, family_call, family_entry, family_array, family_as_f32,
)


def _qn_scores(model: PythonObject, binding: PythonObject, source: PythonObject, n_targets: Int,
               context: PythonObject) raises -> PythonObject:
    """`linear_model._qn_scores`: `qn_decision_function` on the fitted block,
    (n,) for one target, (n, n_targets) for the one-vs-rest block."""
    var b = Python.import_module("builtins")
    var x = family_as_f32(source, 2, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    if d != Int(py=py_call(b.getattr, model, PythonObject("n_features_in_"))):
        raise Error("mojolearn " + String(py_type(model).__name__) + " feature count differs from fit")
    var intercept = 1 if Bool(py=py_call(b.bool, py_call(b.getattr, model, PythonObject("fit_intercept")))) else 0
    var params = Python.list(PythonObject(n), PythonObject(d), PythonObject(intercept))
    var out: PythonObject
    if n_targets > 1:
        _ = py_call(params.append, PythonObject(n_targets))
        out = family_array(Python.tuple(n, n_targets), "<f4", context)
    else:
        out = family_array(Python.tuple(n), "<f4", context)
    var w = py_call(b.getattr, model, PythonObject("_w"))
    _ = family_call(family_entry(binding, "qn_decision_function", context), classical_address(x),
                    classical_address(w), classical_address(out), params)
    _ = x^
    _ = w^
    return out


def _class_count(classes: PythonObject) raises -> Int:
    """`len(classes_)` of a list, a tuple or an array, read natively."""
    var b = Python.import_module("builtins")
    if py_type(classes) is b.list or py_type(classes) is b.tuple:
        return len(classes)
    # an array has a shape; a list subclass (`_labels.ClassLabels`) does not
    if Bool(py=py_call(b.hasattr, classes, PythonObject("shape"))):
        return Int(py=classes.shape[0])
    return len(classes)


def _gmm_sample(model: PythonObject, binding: PythonObject, n_samples: PythonObject,
                context: PythonObject) raises -> PythonObject:
    """`GaussianMixture.sample(n_samples)`: (X, y)."""
    var b = Python.import_module("builtins")
    var operator = Python.import_module("operator")
    if py_call(b.getattr, model, PythonObject("_ext"), PythonObject(None)) is not None:
        raise Error("NotImplementedError: mojolearn GaussianMixture: sample() is the default fit's; a fit with "
                    + "the options routed to x_cluster does not sample (x_cluster/NOT_IMPLEMENTED.tsv)")
    if not Bool(py=family_call(b.hasattr, model, PythonObject("weights_"))):
        raise Error("mojolearn GaussianMixture: call fit before sample")
    if not family_is_int(n_samples):
        raise Error("TypeError: mojolearn GaussianMixture: n_samples must be an int")
    var seed = py_call(b.getattr, model, PythonObject("random_state"))
    var seed_ok = family_is_int(seed)
    if seed_ok:
        seed_ok = Bool(py=family_call(operator.le, PythonObject(0), seed)) and Bool(
            py=family_call(operator.lt, seed, family_call(operator.lshift, PythonObject(1), PythonObject(64))))
    if not seed_ok:
        raise Error("mojolearn GaussianMixture: sample needs random_state to be an int in "
                    + "[0, 2**64), got " + family_repr(seed) + "; it is the key of every draw (DEVIATION 2791)")
    var n = Int(py=family_call(b.int, n_samples))
    var weights = py_call(b.getattr, model, PythonObject("weights_"))
    var k = Int(py=weights.shape[0])
    var d = Int(py=py_call(b.getattr, model, PythonObject("n_features_in_")))
    var means = py_call(b.getattr, model, PythonObject("means_"))
    var covariances = py_call(b.getattr, model, PythonObject("covariances_"))
    var precisions = py_call(b.getattr, model, PythonObject("precisions_cholesky_"))
    var log_det = py_call(b.getattr, model, PythonObject("log_det_chol_"))
    # Never a zero-length buffer: a refused n still addresses real memory.
    var rows = max(n, 1)
    var x = family_array(Python.tuple(rows, d), "<f4", context)
    var y = family_array(Python.tuple(rows), "<i4", context)
    # `_extension`: the binary's own compile-time mode must be the one requested.
    var fields = py_fields(model)
    var want = py_get(fields, PythonObject("numeric_mode"), PythonObject(None))
    if want is None or not Bool(py=py_call(b.bool, want)):
        want = context["backend_fields"]["_DEFAULT_MODE"]
    var getter = py_call(b.getattr, binding, PythonObject("mixture_numeric_mode"), PythonObject(None))
    if getter is not None:
        var got = Int(py=family_call(b.int, family_call(getter)))
        var wanted = py_get(context["family_gmm_mode_codes"], want, PythonObject(None))
        if wanted is None or Int(py=wanted) != got:
            raise Error("RuntimeError: mojolearn GaussianMixture: numeric_mode=" + family_repr(want) + " was requested "
                        + "but " + String(py_call(b.getattr, binding, PythonObject("__name__"))) + " reports compile-time mode code "
                        + String(got) + "; rebuild it with bash bindings/build_mixture.sh")
    var converged = 1 if Bool(py=py_call(b.bool, py_call(b.getattr, model, PythonObject("converged_")))) else 0
    # ORDER MATCHES bindings/_mojolearn_mixture.mojo::gmm_sample_binding:
    # weights, means, covariances, precisions_chol, log_det_chol, X out, y out;
    # k, d, n_iter, converged, lower_bound, n (0: no X is read);
    # n_samples, random_state low 32 bits, random_state high 32 bits.
    _ = family_call(family_entry(binding, "gmm_sample", context),
                    Python.list(classical_address(weights), classical_address(means), classical_address(covariances),
                                classical_address(precisions), classical_address(log_det), classical_address(x),
                                classical_address(y)),
                    Python.list(PythonObject(k), PythonObject(d), py_call(b.getattr, model, PythonObject("n_iter_")),
                                PythonObject(converged), py_call(b.getattr, model, PythonObject("lower_bound_")),
                                PythonObject(0)),
                    Python.list(PythonObject(n), family_call(operator.and_, seed, PythonObject(0xFFFFFFFF)),
                                family_call(operator.rshift, seed, PythonObject(32))))
    _ = weights^
    _ = means^
    _ = covariances^
    _ = precisions^
    _ = log_det^
    return Python.tuple(x, y)


def svm_mixture_operation(model: PythonObject, operation: String, args: PythonObject,
                          context: PythonObject) raises -> PythonObject:
    """args[0] is the resolved binding."""
    classical_connect_numpy(context)
    var b = Python.import_module("builtins")
    if operation == "family_svm_linear_svc_decision":
        if not Bool(py=family_call(b.hasattr, model, PythonObject("_w"))):
            raise Error("mojolearn LinearSVC: call fit first")
        var n_classes = _class_count(py_call(b.getattr, model, PythonObject("classes_")))
        return _qn_scores(model, args[0], args[1], 1 if n_classes == 2 else n_classes, context)
    if operation == "family_svm_linear_svr_predict":
        if not Bool(py=family_call(b.hasattr, model, PythonObject("_w"))):
            raise Error("mojolearn LinearSVR: call fit before predict")
        return _qn_scores(model, args[0], args[1], 1, context)
    if operation == "family_svm_gmm_sample":
        return _gmm_sample(model, args[0], args[1], context)
    raise Error("RuntimeError: unknown svm/mixture operation " + operation)

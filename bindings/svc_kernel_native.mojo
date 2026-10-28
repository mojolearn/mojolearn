# SPDX-License-Identifier: Apache-2.0
# SVC `fit`, `decision_function`, `predict`, `predict_proba`,
# `predict_log_proba`, `score` and SVR `fit`, `predict`, `score`
# (python/mojolearn/_svm_impl.py), one native call each.
#
# This owner is the Python it replaces: `_validate_params` with its refusals
# in the same wording and order, `_as_labels` / `_as_targets`, the exact
# gamma (`_resolve_gamma`: the binding's `scale_gamma_limbs`, then
# core/scale_gamma_exact.mojo's once-rounded rational), the per-row bounds
# of `_c_rows`, and the solver driver: the binary machine, the one-vs-one
# machines over the class pairs (`# driver-loop`), libsvm's Platt
# cross-validation per pair (five folds), then the decision, vote and
# probability epilogues. Every number comes from the same binding entries
# (`svc_fit`, `svc_predict`, `svc_pair_epilogue`, `svc_splitmix_perm`,
# `svc_platt_train`, `svc_ovo_layout`, `svr_fit`, `svr_predict`) called with
# the same parameter lists on the same operands in the same order, so the
# bits do not move. An intermediate one entry writes and the next reads
# (a pair's rows, a fold's held rows, the fold decisions) stays the buffer
# it was written into; the Python's slice copies of their prefixes are not
# made (the entries read only those prefixes).
#
# The binding keeps no device state between calls (DEVIATION 873): `predict`
# hands the fitted support vectors over at every call, as before. There is
# no resident entry for them.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_pandas import classical_admit_value
from bindings.host_inference_scalars import host_inference_scalar
from bindings.native_owner_lifetime import native_release_admit
from bindings.label_admission_native import labels_primitive_kind
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields
from bindings.classical_arrays import classical_address, classical_is_array, classical_wrap_array
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_attributes import _type_mro
from bindings.mode_metadata_native import _class_fields
from bindings.classical_exceptions import classical_require_fitted
from bindings.classical_arrays import classical_typed
from bindings.classical_labels import labels_cast, labels_real_type
from bindings.native_array_view import native_array_ndarray_view
from providers.python.estimator_lifecycle_pyapi import estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete
from bindings.family_common import (
    FAMILY_F32P, FAMILY_F64P, FAMILY_I32P, FAMILY_I64P, family_repr, family_str, family_isinstance,
    family_call, family_entry, family_array, family_sample_weight_snapshot,
)
from core.scale_gamma_exact import scale_gamma_from_limbs
from bindings.svc_common import (
    svc_builtins, svc_field, svc_attr, svc_has, svc_name, svc_float, svc_int, svc_truthy, svc_isfinite,
    svc_round_f32, svc_base, svc_mode, svc_zero_division, svc_prefix, svc_scalar_array, svc_read_f64,
    svc_read_f32, svc_read_i32, svc_f32, svc_snapshot, svc_shape_of, svc_labels_1d, svc_sorted_classes, svc_decode,
    svc_threshold_codes, svc_argmax_rows, svc_r2_sums, svc_accuracy,
)

# `svc_pair_epilogue` modes (svm/impl/svc_rows.mojo), `_svm_impl._EPI_*` and `_GLUE_*`.
comptime _EPI_OVO = 0
comptime _EPI_OVR = 1
comptime _EPI_VOTES = 2
comptime _EPI_PROBA = 3
comptime _EPI_LOG_PROBA = 4
comptime _EPI_BINARY = 5
comptime _GLUE_GATHER = 6
comptime _GLUE_SELECT = 7
comptime _GLUE_C_ROWS = 8
comptime _GLUE_FOLD = 9
comptime _GLUE_FOLD_FINISH = 10
# `_scale_gamma`: the limb words and the py2mojo switch bit.
comptime _SG_SLOTS = 28
comptime _PY2MOJO_SCALE_GAMMA = 1
comptime _MAX_POLY_DEGREE = 32
comptime _DEV636_Y = "mojolearn SVC: y contains a non-finite value (DEVIATION 636: a NaN or inf cannot be fitted; a computed NaN carries a vendor-specific payload and cannot sit in a hashed stage)"


from providers.python.svc_descriptor_pyapi import svc_epilogue_execute, svc_platt_execute, svc_splitmix_execute, svr_execute, svc_descriptor_execute, svc_descriptor_flags, svc_ovo_execute
from providers.python.compiled_binding_snapshot_pyapi import CompiledBindingSnapshot
from providers.python.array_metadata_pyapi import ArrayMetadata, array_shape_once, array_metadata_once
from std.os import getenv
from svm.operation_metadata import SvcOperationSettings, SVC_ABI_NAME
from providers.python.neural_descriptor_pyapi import (
    neural_descriptor_entry, neural_descriptor_connect, neural_descriptor_execute, neural_descriptor_info,
)


def _svc_provider(binding: PythonObject) raises -> PythonObject:
    # One common provider admission per public operation, never a pair/fold
    # cache lookup. Retain its checked module/table until the operation ends.
    if getenv("MOJOLEARN_CLASSICAL_NATIVE_DISPATCH", "0") != "1":
        return PythonObject(None)
    var factory = neural_descriptor_entry(binding, "svm_binary_api_v1", False)
    if factory is None:
        raise Error("SVC native dispatch candidate is unavailable for this binding")
    var native = neural_descriptor_connect(factory, String(SVC_ABI_NAME), 12)
    if neural_descriptor_execute(native, 5, [], [], []) != 1:
        raise Error("SVC native dispatch candidate requires identical mode")
    var flags = neural_descriptor_execute(native, 8, [], [], [])
    return PythonObject(alloc=CompiledBindingSnapshot(native, binding, String(SVC_ABI_NAME), 1, [flags]))


def _svc_publish_dispatch(fields: PythonObject, native: PythonObject, context: PythonObject) raises:
    if native is not None:
        # Plain diagnostic values only: never persist module/capsule owners or
        # make the native table part of a model's pickle/checkpoint state.
        if py_contains(fields, "_native_svc_dispatch"):
            native_release_admit(fields["_native_svc_dispatch"], context)
        py_setitem(fields, "_native_svc_dispatch", neural_descriptor_info(native.downcast_value_ptr[CompiledBindingSnapshot]()[].provider))


def _svc_fit_settings(fields: PythonObject, gamma: Float64, features: Int,
                      context: PythonObject) raises -> SvcOperationSettings:
    return SvcOperationSettings(_kernel_code(fields, context), svc_int(svc_field(fields, "_hp_degree")),
        svc_int(svc_field(fields, "_hp_max_iter")), svc_int(svc_field(fields, "_hp_nochange_steps")),
        features, gamma, svc_float(svc_field(fields, "_hp_C")), svc_float(svc_field(fields, "_hp_tol")),
        svc_float(svc_field(fields, "_hp_coef0")), svc_float(svc_field(fields, "_hp_cache_size")))


def _svc_predict_settings(fields: PythonObject, context: PythonObject) raises -> SvcOperationSettings:
    return SvcOperationSettings(_kernel_code(fields, context), svc_int(svc_field(fields, "_hp_degree")),
        0, 0, svc_int(svc_field(fields, "n_features_in_")), svc_float(svc_field(fields, "_gamma")),
        0.0, 0.0, svc_float(svc_field(fields, "_hp_coef0")), svc_float(svc_field(fields, "_hp_cache_size")))


def _addr(value: PythonObject) raises -> Int:
    return Int(py=classical_address(value))


def _entry(binding: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    return family_entry(binding, name, context)


# ---------------------------------------------------------------- parameters


def _svm_integral(value: PythonObject, context: PythonObject) raises -> Bool:
    return labels_primitive_kind(py_type(value), context) == 3


def _svm_parameter_admission(fields: PythonObject, classifier: Bool, context: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var names = Python.tuple("kernel", "degree", "gamma", "coef0", "C", "tol", "cache_size", "max_iter", "nochange_steps", "verbose", "output_type")
    for i in range(len(names)):  # driver-loop: scalar controls before native conversion, comparison or truth slots
        host_inference_scalar(fields[names[i]], context, True)
    if classifier:
        for name in ["probability", "random_state", "decision_function_shape", "break_ties"]:  # small-loop(fixed classifier controls)
            host_inference_scalar(fields[name], context, True)
        var weights = fields["class_weight"]
        if py_type(weights) is b.dict:
            var items = py_call(b.list, weights.items())
            for i in range(len(items)):  # driver-loop: class-weight hashing/comparison cannot invoke Python scalar protocols
                host_inference_scalar(items[i][0], context)
                host_inference_scalar(items[i][1], context)
        else:
            host_inference_scalar(weights, context)
    else:
        host_inference_scalar(fields["epsilon"], context)
    var replaced = Python.tuple("_hp_C", "_hp_kernel", "_hp_degree", "_hp_gamma", "_hp_coef0", "_hp_tol",
                                "_hp_cache_size", "_hp_max_iter", "_hp_nochange_steps")
    for i in range(len(replaced)):  # driver-loop: only working fields the validator actually overwrites
        if py_contains(fields, replaced[i]):
            native_release_admit(fields[replaced[i]], context)
    if classifier:
        for name in ["_hp_random_state", "_hp_probability", "_hp_break_ties"]:  # small-loop(fixed classifier working state)
            if py_contains(fields, name):
                native_release_admit(fields[name], context)
    elif py_contains(fields, "_hp_epsilon"):
        native_release_admit(fields["_hp_epsilon"], context)


def _svm_validate(model: PythonObject, fields: PythonObject, classifier: Bool, context: PythonObject) raises:
    """`SVC._validate_params` / `SVR._validate_params`: the refusals in the
    Python's order and wording, then the checked `_hp_*` working values."""
    _svm_parameter_admission(fields, classifier, context)
    var b = svc_builtins()
    var operator = native_loaded_module("operator")
    var head = String("mojolearn SVC: ") if classifier else String("mojolearn SVR: ")
    var kernels = context["family_svc_kernels"]
    var refused = context["family_svc_refused" if classifier else "family_svr_refused"]
    var kernel = svc_field(fields, "kernel")
    var degree = svc_field(fields, "degree")
    var gamma = svc_field(fields, "gamma")
    var coef0 = svc_field(fields, "coef0")
    if not family_isinstance(kernel, b.str):
        raise Error(head + "kernel is a name")
    var lower = family_call(kernel.lower)
    var k = kernel if Bool(py=family_call(operator.eq, kernel, lower)) else lower
    if py_contains(refused, k):
        raise Error("NotImplementedError: " + head + "kernel=" + family_repr(kernel) + " is refused; "
                    + String(refused[k]))
    if not py_contains(kernels, k):
        raise Error(head + "kernel=" + family_repr(kernel) + " is not a kernel name; "
                    + "this implementation carries " + family_repr(family_call(b.sorted, kernels)) + " and refuses "
                    + "the others by name (" + family_repr(family_call(b.sorted, refused)) + ")")
    if family_isinstance(gamma, b.str):
        var folded = family_call(gamma.lower)
        var g = gamma if Bool(py=family_call(operator.eq, gamma, folded)) else folded
        if String(g) != "auto" and String(g) != "scale":
            raise Error(head + "gamma=" + family_repr(gamma) + " is not a name; it is "
                        + "'auto', 'scale' or a float")
        gamma = g
    else:
        gamma = family_call(b.float, gamma)
        var value = Float64(py=gamma)
        if not svc_isfinite(value) or value < 0.0:
            raise Error(head + "gamma must be finite and >= 0 for the RBF "
                        + "kernel, got " + family_repr(gamma) + " (DEVIATION 636; scikit-learn's own "
                        + "constraint is gamma >= 0)")
    var name = String(k)
    if name == "poly":
        var boolean = family_isinstance(degree, b.bool)
        if boolean or not _svm_integral(degree, context):
            raise Error("TypeError: " + head + "degree=" + family_repr(degree) + " must be an integer; the "
                        + "polynomial power is an ascending repeated product (DEVIATION 1663)")
        degree = family_call(b.int, degree)
        if (Bool(py=family_call(operator.lt, degree, PythonObject(0)))
                or Bool(py=family_call(operator.gt, degree, PythonObject(_MAX_POLY_DEGREE)))):
            raise Error(head + "degree must be in [0, " + String(_MAX_POLY_DEGREE) + "], got "
                        + family_repr(degree) + " (DEVIATION 1663)")
    elif classifier and name == "sigmoid":
        if Bool(py=family_call(operator.ne, degree, PythonObject(3))):
            raise Error("NotImplementedError: " + head + "degree=" + family_repr(degree) + " is refused with kernel='sigmoid'; "
                        + "it is read only by kernel='poly'")
    elif Bool(py=family_call(operator.ne, degree, PythonObject(3))):
        raise Error("NotImplementedError: " + head + "degree=" + family_repr(degree) + " is refused with kernel="
                    + family_repr(k) + "; it is read only by kernel='poly'. Passing it with another kernel "
                    + "would be a parameter accepted and ignored")
    if name == "poly" or name == "sigmoid":
        coef0 = family_call(b.float, coef0)
        if not svc_isfinite(Float64(py=coef0)):
            raise Error(head + "coef0 must be finite, got " + family_repr(coef0) + " (DEVIATION 636)")
    elif Bool(py=family_call(operator.ne, coef0, PythonObject(0.0))):
        raise Error("NotImplementedError: " + head + "coef0=" + family_repr(coef0) + " is refused with kernel="
                    + family_repr(k) + "; it is read only by kernel='poly' and kernel='sigmoid'")
    var c = family_call(b.float, svc_field(fields, "C"))
    if not svc_isfinite(Float64(py=c)):
        raise Error(head + "C must be finite, got " + family_repr(c) + " (DEVIATION 636)")
    if Float64(py=c) <= 0.0:
        raise Error(head + "C must be positive, got " + family_repr(c))
    var tol = family_call(b.float, svc_field(fields, "tol"))
    if not svc_isfinite(Float64(py=tol)):
        raise Error(head + "tol must be finite, got " + family_repr(tol) + " (DEVIATION 636)")
    if Float64(py=tol) <= 0.0:
        raise Error(head + "tol must be positive, got " + family_repr(tol))
    var epsilon = PythonObject(None)
    if not classifier:
        # converted, not refused: the Mojo fit's own refusals stay reachable
        epsilon = family_call(b.float, svc_field(fields, "epsilon"))
    var cache = family_call(b.float, svc_field(fields, "cache_size"))
    var cache_value = Float64(py=cache)
    if not (cache_value > 0.0) or not svc_isfinite(cache_value):
        raise Error(head + "cache_size is the prediction buffer here "
                    + "(DEVIATION 871) and must be a positive finite MiB, got " + family_repr(cache))
    var max_iter = family_call(b.int, svc_field(fields, "max_iter"))
    if (Bool(py=family_call(operator.eq, max_iter, PythonObject(0)))
            or Bool(py=family_call(operator.lt, max_iter, PythonObject(-1)))):
        raise Error(head + "max_iter is a positive cap or -1 for no "
                    + "limit (cuML's default), got " + family_repr(max_iter))
    var nochange = family_call(b.int, svc_field(fields, "nochange_steps"))
    if Bool(py=family_call(operator.lt, nochange, PythonObject(0))):
        raise Error(head + "nochange_steps cannot be negative, got " + family_repr(nochange))
    if svc_truthy(svc_field(fields, "verbose")):
        raise Error("NotImplementedError: " + head + "verbose is refused; upstream it selects "
                    + "CUML_LOG_DEBUG lines and this implementation prints none, so honoring "
                    + "it is impossible and accepting it would be accepting-and-"
                    + "ignoring")
    if svc_field(fields, "output_type") is not None:
        raise Error("NotImplementedError: " + head + "output_type is a cuML-internal array-type "
                    + "selector; this package returns mojolearn Arrays")
    if not classifier:
        py_setitem(fields, PythonObject("_hp_kernel"), k)
        py_setitem(fields, PythonObject("_hp_degree"), degree)
        py_setitem(fields, PythonObject("_hp_gamma"), gamma)
        py_setitem(fields, PythonObject("_hp_coef0"), coef0)
        py_setitem(fields, PythonObject("_hp_tol"), tol)
        py_setitem(fields, PythonObject("_hp_C"), c)
        py_setitem(fields, PythonObject("_hp_epsilon"), epsilon)
        py_setitem(fields, PythonObject("_hp_cache_size"), cache)
        py_setitem(fields, PythonObject("_hp_max_iter"), max_iter)
        py_setitem(fields, PythonObject("_hp_nochange_steps"), nochange)
        return
    var probability = svc_truthy(svc_field(fields, "probability"))
    var random_state = svc_field(fields, "random_state")
    if not probability:
        # read only by probability's folds: accepted and unused without it
        random_state = PythonObject(None)
    if random_state is not None:
        var bad = family_isinstance(random_state, b.bool) or not _svm_integral(random_state, context)
        if not bad:
            var value = family_call(b.int, random_state)
            bad = (Bool(py=family_call(operator.lt, value, PythonObject(0)))
                   or not Bool(py=family_call(operator.lt, value, family_call(operator.lshift, PythonObject(1), PythonObject(64)))))
        if bad:
            raise Error("mojolearn SVC: random_state must be None or an int in [0, 2**64) "
                        + "(the key of probability's fold shuffle), got " + family_repr(random_state))
    var class_weight = svc_field(fields, "class_weight")
    if class_weight is not None and not family_isinstance(class_weight, Python.tuple(b.dict, b.str)):
        raise Error("mojolearn SVC: class_weight is a dict, 'balanced' or None, "
                    + "got " + String(py_type(class_weight).__name__))
    var shape = svc_field(fields, "decision_function_shape")
    if not py_contains(Python.tuple(PythonObject("ovo"), PythonObject("ovr")), shape):
        raise Error("mojolearn SVC: decision_function_shape is 'ovo' or 'ovr', got " + family_repr(shape))
    var break_ties = svc_truthy(svc_field(fields, "break_ties"))
    if break_ties and Bool(py=family_call(operator.eq, shape, PythonObject("ovo"))):
        raise Error("mojolearn SVC: break_ties must be False when "
                    + "decision_function_shape is 'ovo' (scikit-learn's rule)")
    py_setitem(fields, PythonObject("_hp_C"), c)
    py_setitem(fields, PythonObject("_hp_kernel"), k)
    py_setitem(fields, PythonObject("_hp_degree"), degree)
    py_setitem(fields, PythonObject("_hp_gamma"), gamma)
    py_setitem(fields, PythonObject("_hp_coef0"), coef0)
    py_setitem(fields, PythonObject("_hp_tol"), tol)
    py_setitem(fields, PythonObject("_hp_cache_size"), cache)
    py_setitem(fields, PythonObject("_hp_max_iter"), max_iter)
    py_setitem(fields, PythonObject("_hp_nochange_steps"), nochange)
    py_setitem(fields, PythonObject("_hp_random_state"),
               PythonObject(None) if random_state is None else family_call(b.int, random_state))
    py_setitem(fields, PythonObject("_hp_probability"), PythonObject(probability))
    py_setitem(fields, PythonObject("_hp_break_ties"), PythonObject(break_ties))


def _kernel_code(fields: PythonObject, context: PythonObject) raises -> Int:
    return Int(py=context["family_svc_kernels"][svc_field(fields, "_hp_kernel")])


def _resolve_gamma(fields: PythonObject, binding: PythonObject, x_meta: ArrayMetadata, context: PythonObject, native: PythonObject) raises -> Float64:
    """`SVC._resolve_gamma`: 'auto' 1 / n_cols; 'scale' the exact variance's
    once-rounded reciprocal (`_scale_gamma.scale_gamma_x`); else the float."""
    var g = svc_field(fields, "_hp_gamma")
    var n_cols = x_meta.shape[1]
    if family_isinstance(g, svc_builtins().str):
        if String(g) == "auto":
            return 1.0 / Float64(n_cols)
        if String(g) == "scale":
            var bits = 0
            if native is not None:
                bits = svc_descriptor_flags(native)
            else:
                var flags = py_call(svc_builtins().getattr, binding, PythonObject("py2mojo_linear_flags"), PythonObject(None))
                bits = 0 if flags is None else svc_int(family_call(flags))
            if (bits & _PY2MOJO_SCALE_GAMMA) == 0:
                raise Error("RuntimeError: mojolearn: this binding has no scale_gamma_limbs (an older binary or a "
                            + "-D MOJOLEARN_PY2MOJO_linear_OFF build); rebuild it")
            var n = x_meta.count
            if n == 0:
                raise Error("gamma='scale' needs at least one cell of X")
            var words = family_array(Python.tuple(PythonObject(_SG_SLOTS)), "<i8", context)
            if native is not None:
                _ = svc_descriptor_execute(native, 9, [x_meta.source.addr, _addr(words)], [n], [])
            else:
                _ = family_call(_entry(binding, "scale_gamma_limbs", context), PythonObject(x_meta.source.addr), PythonObject(n),
                                classical_address(words))
            _ = words  # keep the Python object alive through the native call above
            var limbs = List[Int64](length=_SG_SLOTS, fill=Int64(0))
            var wp = FAMILY_I64P(unsafe_from_address=_addr(words))
            _ = words  # keep the Python object alive through the native call above
            for i in range(_SG_SLOTS):  # small-loop(scale_gamma limbs: 28 fixed words)
                limbs[i] = wp[i]
                _ = words  # keep the Python object alive through the native call above
            _ = words^
            return scale_gamma_from_limbs(limbs, n, n_cols)
    return Float64(py=g)


def _c_rows(binding: PythonObject, c_value: Float64, n_rows: Int, sample_weight: PythonObject,
            class_weight: PythonObject, y: PythonObject, who: String, context: PythonObject, native: PythonObject) raises -> PythonObject:
    """`_svm_impl._c_rows`: the per-row bounds `C * class_weight[y_i] *
    sample_weight_i` from the binding's `svc_pair_epilogue` (C_ROWS), or None
    when nothing is weighted."""
    var b = svc_builtins()
    if sample_weight is None and class_weight is None:
        return PythonObject(None)
    var sw = PythonObject(None)
    var sw_owner = PythonObject(None)
    var sw_address = 0
    if sample_weight is not None:
        var weights = family_sample_weight_snapshot(sample_weight, n_rows, False, context, True)
        sw = weights.keep
        sw_owner = Python.tuple(weights.keep, weights.owner)
        sw_address = weights.source.addr
    var codes = PythonObject(None)
    var cwa = PythonObject(None)
    var n_classes = 0
    if class_weight is not None:
        var labels = svc_labels_1d(y, context)[0]
        var encoded = svc_sorted_classes(labels, context)
        var classes = encoded[0]
        codes = encoded[1]
        n_classes = len(classes)
        if Int(py=codes.shape[0]) != n_rows:
            raise Error("mojolearn " + who + ": y has " + String(Int(py=codes.shape[0])) + " entries, X has "
                        + String(n_rows) + " rows")
        if family_isinstance(class_weight, b.str):
            if String(class_weight) != "balanced":
                raise Error("mojolearn " + who + ": class_weight is a dict, 'balanced' or None, got "
                            + family_repr(class_weight))
            var cnt = family_array(Python.tuple(PythonObject(n_classes)), "<i8", context)
            _ = family_call(svc_base("bincount_i64", context), classical_address(codes), PythonObject(2),
                            codes.shape[0], PythonObject(n_classes), classical_address(cnt), PythonObject(0))
            _ = codes  # keep the Python object alive through the native call above
            _ = cnt  # keep the Python object alive through the native call above
            cwa = family_array(Python.tuple(PythonObject(n_classes)), "<f8", context)
            if n_classes > 0 and Int(py=family_call(svc_base("class_ratio_f64", context), classical_address(cnt),
                                                     PythonObject(n_classes), PythonObject(n_rows), PythonObject(0),
                                                     classical_address(cwa))) != 0:
                svc_zero_division(False)
            _ = cnt  # keep the Python object alive through the native calls above
            _ = cwa  # keep the Python object alive through the native calls above
            _ = cnt^
        else:
            cwa = family_array(Python.tuple(PythonObject(n_classes)), "<f8", context)
            var cp = FAMILY_F64P(unsafe_from_address=_addr(cwa))
            _ = cwa  # keep the Python object alive through the native call above
            for i in range(n_classes):  # driver-loop: over the classes (the default weight 1.0)
                cp[i] = 1.0
                _ = cwa  # keep the Python object alive through the native call above
            var items = family_call(b.list, family_call(family_call(b.dict, class_weight).items))
            for i in range(len(items)):  # driver-loop: over the user's class_weight entries
                var key = items[i][0]
                if not py_contains(classes, key):
                    raise Error("mojolearn " + who + ": class_weight names the label " + family_repr(key)
                                + ", which is not in y's classes " + family_repr(classes))
                cp[Int(py=family_call(classes.index, key))] = svc_float(items[i][1])
                _ = cwa  # keep the Python object alive through the native call above
    var out = family_array(Python.tuple(PythonObject(n_rows)), "<f4", context)
    if n_rows > 0:
        _ = svc_epilogue_execute(binding, native,
                        PythonObject(sw_address),
                        PythonObject(0) if cwa is None else classical_address(codes),
                        PythonObject(0) if cwa is None else classical_address(cwa),
                        classical_address(out),
                        Python.list(PythonObject(_GLUE_C_ROWS), PythonObject(n_rows), PythonObject(n_classes),
                                    PythonObject(c_value), PythonObject(0)))
        _ = sw  # keep the Python object alive through the native call above
        _ = codes  # keep the Python object alive through the native call above
        _ = cwa  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
    _ = sw^
    _ = sw_owner
    _ = codes^
    _ = cwa^
    return out


# ---------------------------------------------------------------- the row glue


def _gather(binding: PythonObject, x_addr: Int, n_rows: Int, n_cols: Int, rows_addr: Int, nor: Int,
            cols_addr: Int, noc: Int, context: PythonObject, native: PythonObject) raises -> PythonObject:
    """`_svm_impl._gather`: `x[rows][:, cols]` (an absent index list, address
    0, keeps that axis whole), an exact copy, gathered in the binding."""
    var out = family_array(Python.tuple(PythonObject(nor), PythonObject(noc)), "<f4", context)
    if nor > 0 and noc > 0:
        _ = svc_epilogue_execute(binding, native, PythonObject(x_addr), PythonObject(rows_addr),
                        PythonObject(cols_addr), classical_address(out),
                        Python.list(PythonObject(_GLUE_GATHER), PythonObject(n_rows), PythonObject(n_cols),
                                    PythonObject(nor), PythonObject(noc)))
        _ = out  # keep the Python object alive through the native call above
    return out


def _gather_i32(binding: PythonObject, values_addr: Int, m: Int, rows_addr: Int, count: Int,
                context: PythonObject, native: PythonObject) raises -> PythonObject:
    """`_svm_impl._gather_i32`: `values[rows]`, the binding's 4-byte gather."""
    var out = family_array(Python.tuple(PythonObject(count)), "<i4", context)
    if count > 0:
        _ = svc_epilogue_execute(binding, native, PythonObject(values_addr), PythonObject(rows_addr),
                        PythonObject(0), classical_address(out),
                        Python.list(PythonObject(_GLUE_GATHER), PythonObject(m), PythonObject(1), PythonObject(count),
                                    PythonObject(1)))
        _ = out  # keep the Python object alive through the native call above
    return out


def _i32_array(values: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_svm_impl._i32`: an int32 Array as is, a sequence of ints packed."""
    if classical_is_array(values, context) and String(values.dtype) == "<i4":
        return values
    var b = svc_builtins()
    var store = PythonObject(None)
    try:
        store = py_call(context["array_type"], PythonObject("i"), values)
    except:
        store = family_call(context["array_type"], PythonObject("i"), family_call(b.map, b.int, values))
    return classical_wrap_array(store, Python.tuple(PythonObject(len(store))), "<i4", "C", context)


def _pair_select(binding: PythonObject, codes: PythonObject, ci: Int, cj: Int, c_rows: PythonObject,
                 context: PythonObject, native: PythonObject) raises -> PythonObject:
    """`_svm_impl._pair_select`: (row indices, 0/1 labels, bounds or None, m),
    the buffers uncut: their first m entries are the pair's."""
    var n = Int(py=codes.shape[0])
    var idx = family_array(Python.tuple(PythonObject(n)), "<i4", context)
    var lab = family_array(Python.tuple(PythonObject(n)), "<f4", context)
    var cout = PythonObject(None)
    if c_rows is not None:
        cout = family_array(Python.tuple(PythonObject(n)), "<f4", context)
    var m = 0
    if n > 0:
        m = svc_int(svc_epilogue_execute(binding, native, classical_address(codes),
                                PythonObject(0) if c_rows is None else classical_address(c_rows),
                                classical_address(lab), classical_address(idx),
                                Python.list(PythonObject(_GLUE_SELECT), PythonObject(n), PythonObject(ci), PythonObject(cj),
                                            PythonObject(0) if cout is None else classical_address(cout))))
        _ = lab  # keep the Python object alive through the native call above
        _ = idx  # keep the Python object alive through the native call above
        _ = cout  # keep the Python object alive through the native call above
    return Python.tuple(idx, lab, cout, PythonObject(m))


# ---------------------------------------------------------------- the machines


def _solve(fields: PythonObject, binding: PythonObject, x_addr: Int, n_rows: Int, n_cols: Int, labels_addr: Int,
           gamma: Float64, c_rows_addr: Int, context: PythonObject,
           settings: SvcOperationSettings, native: PythonObject) raises -> PythonObject:
    """`SVC._solve`: ONE binary C-SVC solve; (n_support, dual, support, sv,
    info) with the worst-case buffers uncut. `c_rows_addr` 0 = unweighted."""
    var dual = family_array(Python.tuple(PythonObject(n_rows)), "<f4", context)
    var support = family_array(Python.tuple(PythonObject(n_rows)), "<i4", context)
    var sv = family_array(Python.tuple(PythonObject(n_rows * n_cols)), "<f4", context)
    var info = family_array(Python.tuple(PythonObject(5)), "<f8", context)
    # ORDER MATCHES bindings/_mojolearn_svm.mojo::svc_fit_binding: n_rows,
    # n_features, kernel, gamma, C, tol, max_iter, nochange_steps, degree,
    # coef0[, the per-row bounds' address]
    var n_support = PythonObject(None)
    if native is not None:
        n_support = PythonObject(svc_descriptor_execute(native, 0,
            [x_addr, labels_addr, _addr(dual), _addr(support), _addr(sv), _addr(info), c_rows_addr],
            settings.fit_ints(n_rows, n_cols), settings.fit_floats()))
    else:
        var params = Python.list(n_rows, n_cols, settings.kernel, settings.gamma, settings.c, settings.tol,
            settings.max_iter, settings.nochange_steps, settings.degree, settings.coef0)
        if c_rows_addr != 0:
            _ = py_call(params.append, PythonObject(c_rows_addr))
        n_support = family_call(_entry(binding, "svc_fit", context), PythonObject(x_addr), PythonObject(labels_addr),
            classical_address(dual), classical_address(support), classical_address(sv), classical_address(info), params)
    _ = sv  # keep the Python object alive through the native call above
    _ = support  # keep the Python object alive through the native call above
    _ = dual  # keep the Python object alive through the native call above
    _ = info  # keep the Python object alive through the native call above
    return Python.tuple(PythonObject(svc_int(n_support)), dual, support, sv, info)


def _machine(fields: PythonObject, binding: PythonObject, q: PythonObject, dual: PythonObject, sv: PythonObject,
             n_support: Int, bias: Float64, label0: Float64, label1: Float64, predict_class: Bool,
             support: PythonObject, out_addr: Int, context: PythonObject,
             settings: SvcOperationSettings, native: PythonObject, n_rows: Int, query_cols: Int, query_address: Int) raises:
    """`SVC._machine`: one machine's decisions (or labels) on q, written at
    `out_addr` (n float32)."""
    var n_features = settings.features
    var query = q
    var query_addr = query_address
    var vectors = sv
    var held = PythonObject(None)
    if settings.kernel == 4 and n_support > 0:
        # X is the cross-kernel against the TRAINING rows; the decision
        # reads its columns at the support vectors and no support rows.
        held = _i32_array(support, context)
        query = _gather(binding, query_address, n_rows, query_cols, 0, n_rows, _addr(held), n_support, context, native)
        _ = held  # keep the Python object alive through the native call above
        query_addr = _addr(query)
        n_features = n_support
        vectors = dual
    # ORDER MATCHES bindings/_mojolearn_svm.mojo::svc_predict_binding:
    # n_rows, n_features, n_support, b, classes[0], classes[1], kernel,
    # gamma, predict_class, cache_size_mib, degree, coef0
    if native is not None:
        _ = svc_descriptor_execute(native, 1,
            [query_addr, _addr(dual) if n_support > 0 else 0,
             _addr(vectors) if n_support > 0 else 0, out_addr],
            settings.predict_ints(n_rows, n_features, n_support, predict_class),
            settings.predict_floats(bias, label0, label1))
    else:
        var params = Python.list(n_rows, n_features, n_support, bias, label0, label1, settings.kernel,
            settings.gamma, 1 if predict_class else 0, settings.cache_mib, settings.degree, settings.coef0)
        _ = family_call(_entry(binding, "svc_predict", context), PythonObject(query_addr),
            classical_address(dual) if n_support > 0 else PythonObject(0),
            classical_address(vectors) if n_support > 0 else PythonObject(0), PythonObject(out_addr), params)
    _ = query  # keep the Python object alive through the native call above
    _ = vectors  # keep the Python object alive through the native call above
    _ = dual
    _ = support
    _ = native
    _ = query^
    _ = vectors^
    _ = held^


# ---------------------------------------------------------------- SVC fit


def _as_labels(y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_svm_impl._as_labels`: (float32 labels or None, classes, the label
    pair or the int32 codes when more than two classes)."""
    var b = svc_builtins()
    var pair = svc_labels_1d(y, context)
    var labels = pair[0]
    if labels is None:
        var shape = pair[1]
        raise Error("mojolearn SVC: y must be 1-D, got " + String(len(shape)) + "-D shape " + family_repr(shape))
    var encoded = PythonObject(None)
    try:
        encoded = svc_sorted_classes(labels, context)
    except error:
        if String(error).find("NaN label") < 0:
            raise error
        raise Error(_DEV636_Y)
    var classes = encoded[0]
    var codes = encoded[1]
    if len(classes) < 2:
        raise Error("mojolearn SVC: y has " + String(len(classes)) + " class; at least two are needed")
    if len(classes) > 2:
        return Python.tuple(PythonObject(None), classes, codes)
    var real = True
    for i in range(len(classes)):  # small-loop(classes: the two labels)
        if labels_primitive_kind(py_type(classes[i]), context) < 2:
            real = False
    if not real:
        # non-numeric labels: the solver sees each row's code, 0.0 or 1.0
        var n = Int(py=codes.size)
        var f = family_array(Python.tuple(PythonObject(n)), "<f4", context)
        if n > 0:
            _ = family_call(svc_base("cast_elements", context), classical_address(codes), PythonObject(2),
                            classical_address(f), PythonObject(0), PythonObject(n))
            _ = codes  # keep the Python object alive through the native call above
            _ = f  # keep the Python object alive through the native call above
        _ = codes^
        return Python.tuple(f, classes, Python.tuple(PythonObject(0.0), PythonObject(1.0)))
    var f = svc_f32(y, 1, "y", False, context)[0]
    var n = Int(py=f.size)
    var finite = True
    if n > 0:
        finite = Int(py=family_call(svc_base("all_finite_f32", context), classical_address(f), PythonObject(n))) == 1
        _ = f  # keep the Python object alive through the native call above
    if not finite:
        raise Error(_DEV636_Y)
    return Python.tuple(f, classes, Python.tuple(classes[0], classes[1]))


def _set_binary(fields: PythonObject, solved: PythonObject, n_cols: Int, context: PythonObject) raises:
    """The binary machine's public attributes from its uncut buffers."""
    var n_support = Int(py=solved[0])
    var info = solved[4]
    py_setitem(fields, PythonObject("n_support_"), PythonObject(n_support))
    py_setitem(fields, PythonObject("n_iter_"), PythonObject(Int(svc_read_f64(info, 2))))
    py_setitem(fields, PythonObject("intercept_"), svc_scalar_array(svc_round_f32(svc_read_f64(info, 0)), "<f4", context))
    # compact C-order Arrays detached from the worst-case buffers
    py_setitem(fields, PythonObject("support_"),
               svc_prefix(solved[2], n_support, Python.tuple(PythonObject(n_support)), "<i4", context))
    py_setitem(fields, PythonObject("support_vectors_"),
               svc_prefix(solved[3], n_support * n_cols, Python.tuple(PythonObject(n_support), PythonObject(n_cols)), "<f4", context))
    py_setitem(fields, PythonObject("dual_coef_"),
               svc_prefix(solved[1], n_support, Python.tuple(PythonObject(1), PythonObject(n_support)), "<f4", context))


def _platt(fields: PythonObject, binding: PythonObject, x_meta: ArrayMetadata, codes: PythonObject, gamma: Float64,
           c_rows: PythonObject, k: Int, context: PythonObject,
           settings: SvcOperationSettings, native: PythonObject) raises:
    """`SVC._fit_probability`: libsvm's `svm_binary_svc_probability` per class
    pair, five shuffled folds, then `sigmoid_train`; sets probA_ / probB_."""
    var b = svc_builtins()
    var operator = native_loaded_module("operator")
    if py_contains(fields, PythonObject("_prob_ab")):
        _ = family_call(fields.pop, PythonObject("_prob_ab"), PythonObject(None))
    if not svc_truthy(svc_field(fields, "_hp_probability")):
        return
    var rs = svc_field(fields, "_hp_random_state")
    var seed = PythonObject(0) if rs is None else family_call(b.int, rs)
    var mask = family_call(operator.sub, family_call(operator.lshift, PythonObject(1), PythonObject(64)), PythonObject(1))
    var precomputed = String(svc_field(fields, "_hp_kernel")) == "precomputed"
    var n_all = Int(py=codes.shape[0])
    var x_rows = x_meta.shape[0]
    var x_cols = x_meta.shape[1]
    var ab = Python.list()
    var a_values = family_array(Python.tuple(PythonObject(k * (k - 1) // 2)), "<f8", context)
    var b_values = family_array(Python.tuple(PythonObject(k * (k - 1) // 2)), "<f8", context)
    var pair_no = 0
    # SplitMix64's golden-ratio increment (above the Int64 range, so built
    # as a Python int)
    var golden = family_call(b.int, PythonObject("9E3779B97F4A7C15"), PythonObject(16))
    for ci in range(k):  # driver-loop: over the class pairs (one probability fit each)
        for cj in range(ci + 1, k):  # driver-loop: over the class pairs (one probability fit each)
            # the pair's rows in row order; libsvm's +1 is class ci
            var sel = _pair_select(binding, codes, ci, cj, PythonObject(None), context, native)
            var idx_a = sel[0]
            var lab_a = sel[1]
            var n = Int(py=sel[3])
            var key = family_call(operator.xor, seed,
                                  family_call(operator.and_, family_call(operator.mul, PythonObject(pair_no),
                                                                         golden), mask))
            key = family_call(operator.and_, key, mask)
            var perm = family_array(Python.tuple(PythonObject(max(n, 1))), "<i4", context)
            if n > 0:
                _ = svc_splitmix_execute(binding, native, classical_address(perm), PythonObject(n),
                                family_call(operator.and_, key, PythonObject(0xFFFFFFFF)),
                                family_call(operator.rshift, key, PythonObject(32)))
                _ = perm  # keep the Python object alive through the native call above
            var dec_perm = family_array(Python.tuple(PythonObject(max(n, 1))), "<f4", context)
            var consts = family_array(Python.tuple(PythonObject(10)), "<f4", context)
            var cst = FAMILY_F32P(unsafe_from_address=_addr(consts))
            _ = consts  # keep the Python object alive through the native call above
            for f in range(5):  # small-loop(Platt folds: five per pair)
                var begin = f * n // 5
                var end = (f + 1) * n // 5
                var h = end - begin
                var nt = n - h
                var rows_held = family_array(Python.tuple(PythonObject(max(n, 1))), "<i4", context)
                var lab = family_array(Python.tuple(PythonObject(max(nt, 1))), "<f4", context)
                var n_pos = 0
                if n > 0:
                    n_pos = svc_int(svc_epilogue_execute(binding, native, classical_address(idx_a),
                                                classical_address(perm), classical_address(lab_a),
                                                classical_address(rows_held),
                                                Python.list(PythonObject(_GLUE_FOLD), PythonObject(n), PythonObject(begin),
                                                            PythonObject(end), classical_address(lab))))
                    _ = idx_a  # keep the Python object alive through the native call above
                    _ = perm  # keep the Python object alive through the native call above
                    _ = lab_a  # keep the Python object alive through the native call above
                    _ = rows_held  # keep the Python object alive through the native call above
                    _ = lab  # keep the Python object alive through the native call above
                var n_neg = nt - n_pos
                if n_pos == 0 or n_neg == 0:
                    cst[2 * f] = Float32(1.0)
                    _ = consts  # keep the Python object alive through the native call above
                    cst[2 * f + 1] = Float32(0.0) if nt == 0 else (Float32(1.0) if n_pos > 0 else Float32(-1.0))
                    _ = consts  # keep the Python object alive through the native call above
                    continue
                if h == 0:
                    continue
                var rows_addr = _addr(rows_held)
                _ = rows_held  # keep the Python object alive through the native call above
                var hrows_addr = rows_addr + 4 * nt
                _ = rows_held  # keep the Python object alive through the native call above
                var sub: PythonObject
                if precomputed:
                    sub = _gather(binding, x_meta.source.addr, x_rows, x_cols, rows_addr, nt, rows_addr, nt, context, native)
                    _ = rows_held  # keep the Python object alive through the native call above
                else:
                    sub = _gather(binding, x_meta.source.addr, x_rows, x_cols, rows_addr, nt, 0, x_cols, context, native)
                    _ = rows_held  # keep the Python object alive through the native call above
                var sub_c = PythonObject(None)
                if c_rows is not None:
                    sub_c = _gather(binding, _addr(c_rows), n_all, 1, rows_addr, nt, 0, 1, context, native)
                    _ = rows_held  # keep the Python object alive through the native call above
                var cols = Int(py=sub.shape[1])
                var solved = _solve(fields, binding, _addr(sub), nt, cols, _addr(lab), gamma,
                                    0 if sub_c is None else _addr(sub_c), context, settings, native)
                _ = lab  # keep the Python object alive through the native call above
                _ = sub_c  # keep the Python object alive through the native call above
                var q: PythonObject
                if precomputed:
                    q = _gather(binding, x_meta.source.addr, x_rows, x_cols, hrows_addr, h, rows_addr, nt, context, native)
                    _ = rows_held  # keep the Python object alive through the native call above
                else:
                    q = _gather(binding, x_meta.source.addr, x_rows, x_cols, hrows_addr, h, 0, x_cols, context, native)
                var n_sv = Int(py=solved[0])
                _ = sub_c  # keep the Python object alive through the native call above
                _ = lab  # keep the Python object alive through the native call above
                var dual = svc_prefix(solved[1], n_sv, Python.tuple(PythonObject(1), PythonObject(n_sv)), "<f4", context)
                _ = sub_c  # keep the Python object alive through the native call above
                _ = lab  # keep the Python object alive through the native call above
                var sv = svc_prefix(solved[3], n_sv * cols, Python.tuple(PythonObject(n_sv), PythonObject(cols)), "<f4", context)
                _ = sub_c  # keep the Python object alive through the native call above
                _ = lab  # keep the Python object alive through the native call above
                var support = svc_prefix(solved[2], n_sv, Python.tuple(PythonObject(n_sv)), "<i4", context)
                _ = sub_c  # keep the Python object alive through the native call above
                _ = lab  # keep the Python object alive through the native call above
                _machine(fields, binding, q, dual, sv, n_sv, svc_round_f32(svc_read_f64(solved[4], 0)), 0.0, 1.0, False,
                         support, _addr(dec_perm) + 4 * begin, context, settings, native, h, nt if precomputed else x_cols, _addr(q))
                _ = sub_c  # keep the Python object alive through the native call above
                _ = lab  # keep the Python object alive through the native call above
                _ = dec_perm  # keep the Python object alive through the native call above
                _ = sub^
                _ = sub_c^
                _ = q^
                _ = rows_held^
                _ = lab^
            # libsvm's orientation (positive toward class i) and +1/-1
            # labels, back in pair row order, float64
            var dec = family_array(Python.tuple(PythonObject(max(n, 1))), "<f8", context)
            var labels = family_array(Python.tuple(PythonObject(max(n, 1))), "<f8", context)
            if n > 0:
                _ = svc_epilogue_execute(binding, native, classical_address(dec_perm),
                                classical_address(perm), classical_address(lab_a), classical_address(dec),
                                Python.list(PythonObject(_GLUE_FOLD_FINISH), PythonObject(n), classical_address(consts),
                                            classical_address(labels), PythonObject(0)))
                _ = dec_perm  # keep the Python object alive through the native call above
                _ = dec  # keep the Python object alive through the native call above
                _ = perm  # keep the Python object alive through the native call above
                _ = lab_a  # keep the Python object alive through the native call above
                _ = consts  # keep the Python object alive through the native call above
                _ = labels  # keep the Python object alive through the native call above
            var ab_out = family_array(Python.tuple(PythonObject(2)), "<f8", context)
            _ = svc_platt_execute(binding, native,
                            classical_address(dec) if n > 0 else PythonObject(0),
                            classical_address(labels) if n > 0 else PythonObject(0),
                            classical_address(ab_out), PythonObject(n))
            _ = dec  # keep the Python object alive through the native call above
            _ = labels  # keep the Python object alive through the native call above
            _ = ab_out  # keep the Python object alive through the native call above
            var av = svc_read_f64(ab_out, 0)
            var bv = svc_read_f64(ab_out, 1)
            _ = py_call(ab.append, Python.tuple(PythonObject(av), PythonObject(bv)))
            FAMILY_F64P(unsafe_from_address=_addr(a_values))[pair_no] = av
            _ = a_values  # keep the Python object alive through the native call above
            FAMILY_F64P(unsafe_from_address=_addr(b_values))[pair_no] = bv
            _ = b_values  # keep the Python object alive through the native call above
            pair_no += 1
            _ = idx_a^
            _ = lab_a^
            _ = perm^
            _ = dec_perm^
            _ = consts^
    py_setitem(fields, PythonObject("_prob_ab"), ab)
    py_setitem(fields, PythonObject("probA_"), a_values)
    py_setitem(fields, PythonObject("probB_"), b_values)


def _fit_ovo(fields: PythonObject, binding: PythonObject, x_meta: ArrayMetadata, classes: PythonObject, codes: PythonObject,
             gamma: Float64, c_rows: PythonObject, context: PythonObject,
             settings: SvcOperationSettings, native: PythonObject) raises:
    """`SVC._fit_ovo` and `_set_ovo_native`: one binary machine per class
    pair (i < j, class j the solver's +1), then scikit-learn's multiclass
    layout from `svc_ovo_layout`."""
    var b = svc_builtins()
    var n_rows = x_meta.shape[0]
    var n_cols = x_meta.shape[1]
    var k = len(classes)
    var precomputed = String(svc_field(fields, "_hp_kernel")) == "precomputed"
    var pairs = Python.list()
    var meta = Python.list(PythonObject(k), PythonObject(n_rows), PythonObject(k * (k - 1) // 2))
    var sup_addrs = Python.list()
    var dual_addrs = Python.list()
    var total = 0
    var intercept = family_array(Python.tuple(PythonObject(k * (k - 1) // 2)), "<f4", context)
    var n_iter = family_array(Python.tuple(PythonObject(k * (k - 1) // 2)), "<i4", context)
    var q_no = 0
    for i in range(k):  # driver-loop: over the class pairs (one binary machine each)
        for j in range(i + 1, k):  # driver-loop: over the class pairs (one binary machine each)
            var sel = _pair_select(binding, codes, i, j, c_rows, context, native)
            var idx_a = sel[0]
            var m = Int(py=sel[3])
            var sub: PythonObject
            if precomputed:
                # the pair's kernel matrix: its rows AND its columns
                sub = _gather(binding, x_meta.source.addr, n_rows, n_cols, _addr(idx_a), m, _addr(idx_a), m, context, native)
                _ = idx_a  # keep the Python object alive through the native call above
            else:
                sub = _gather(binding, x_meta.source.addr, n_rows, n_cols, _addr(idx_a), m, 0, n_cols, context, native)
                _ = idx_a  # keep the Python object alive through the native call above
            var sub_c = sel[2]
            var solved = _solve(fields, binding, _addr(sub), m, Int(py=sub.shape[1]), _addr(sel[1]), gamma,
                                0 if sub_c is None else _addr(sub_c), context, settings, native)
            _ = sub_c  # keep the Python object alive through the native call above
            var info = solved[4]
            _ = sub_c  # keep the Python object alive through the native call above
            if svc_round_f32(svc_read_f64(info, 3)) != 0.0 or svc_round_f32(svc_read_f64(info, 4)) != 1.0:
                raise Error("RuntimeError: mojolearn SVC: the solver's label pair for classes ("
                            + family_repr(classes[i]) + ", " + family_repr(classes[j]) + ") is not (0.0, 1.0); "
                            + "the class mapping cannot be trusted")
            var n_sv = Int(py=solved[0])
            _ = sub_c  # keep the Python object alive through the native call above
            var sub_cols = Int(py=sub.shape[1])
            var sv_rows = svc_prefix(solved[3], n_sv * sub_cols, Python.tuple(PythonObject(n_sv), PythonObject(sub_cols)),
                                     "<f4", context)
            _ = sub_c  # keep the Python object alive through the native call above
            # the pair's support in X's row numbers, gathered in the binding
            var pair_support = _gather_i32(binding, _addr(idx_a), m, _addr(solved[2]), n_sv, context, native)
            _ = sub_c  # keep the Python object alive through the native call above
            _ = idx_a  # keep the Python object alive through the native call above
            if precomputed:
                # the support rows of the WHOLE kernel matrix
                sv_rows = _gather(binding, x_meta.source.addr, n_rows, n_cols, _addr(pair_support), n_sv, 0, n_cols, context, native)
                _ = pair_support  # keep the Python object alive through the native call above
                _ = idx_a  # keep the Python object alive through the native call above
            var dual = svc_prefix(solved[1], n_sv, Python.tuple(PythonObject(1), PythonObject(n_sv)), "<f4", context)
            _ = sub_c  # keep the Python object alive through the native call above
            var bias = svc_round_f32(svc_read_f64(info, 0))
            var iters = Int(svc_read_f64(info, 2))
            var pair = Python.dict()
            py_setitem(pair, PythonObject("i"), PythonObject(i))
            py_setitem(pair, PythonObject("j"), PythonObject(j))
            py_setitem(pair, PythonObject("dual"), dual)
            py_setitem(pair, PythonObject("sv"), sv_rows)
            py_setitem(pair, PythonObject("support"), pair_support)
            _ = idx_a  # keep the Python object alive through the native call above
            py_setitem(pair, PythonObject("b"), PythonObject(bias))
            py_setitem(pair, PythonObject("n_iter"), PythonObject(iters))
            _ = py_call(pairs.append, pair)
            _ = py_call(meta.append, PythonObject(i))
            _ = py_call(meta.append, PythonObject(j))
            _ = py_call(meta.append, PythonObject(n_sv))
            _ = py_call(sup_addrs.append, classical_address(pair_support) if n_sv > 0 else PythonObject(0))
            _ = pair_support  # keep the Python object alive through the native call above
            _ = idx_a  # keep the Python object alive through the native call above
            _ = py_call(dual_addrs.append, classical_address(dual) if n_sv > 0 else PythonObject(0))
            _ = dual  # keep the Python object alive through the native call above
            total += n_sv
            FAMILY_F32P(unsafe_from_address=_addr(intercept))[q_no] = Float32(-bias)
            _ = intercept  # keep the Python object alive through the native call above
            FAMILY_I32P(unsafe_from_address=_addr(n_iter))[q_no] = Int32(iters)
            _ = n_iter  # keep the Python object alive through the native call above
            q_no += 1
            _ = sub^
            _ = sel^
            _ = solved^
            _ = sub_c  # keep the Python object alive through the native call above
    # `_set_ovo_native`: the support_ order, dual_coef_ cells and n_support_
    var cap = min(n_rows, total)
    var support = family_array(Python.tuple(PythonObject(max(1, cap))), "<i4", context)
    var n_support = family_array(Python.tuple(PythonObject(k)), "<i4", context)
    var dual_all = family_array(Python.tuple(PythonObject(max(1, (k - 1) * cap))), "<f4", context)
    var n_sv = svc_int(svc_ovo_execute(binding, native, sup_addrs, dual_addrs, meta,
                                   Python.list(classical_address(support), classical_address(n_support),
                                               classical_address(dual_all), PythonObject(cap))))
    _ = n_support  # keep the Python object alive through the native call above
    _ = support  # keep the Python object alive through the native call above
    _ = dual_all  # keep the Python object alive through the native call above
    py_setitem(fields, PythonObject("classes_"), classes)
    py_setitem(fields, PythonObject("n_features_in_"), PythonObject(n_cols))
    py_setitem(fields, PythonObject("_pairs"), pairs)
    if n_sv > 0:
        var sup = svc_prefix(support, n_sv, Python.tuple(PythonObject(n_sv)), "<i4", context)
        _ = dual_all  # keep the Python object alive through the native call above
        _ = n_support  # keep the Python object alive through the native call above
        py_setitem(fields, PythonObject("support_"), sup)
        py_setitem(fields, PythonObject("support_vectors_"),
                   _gather(binding, x_meta.source.addr, n_rows, n_cols, _addr(sup), n_sv, 0, n_cols, context, native))
        _ = sup  # keep the Python object alive through the native call above
        _ = dual_all  # keep the Python object alive through the native call above
        _ = n_support  # keep the Python object alive through the native call above
        _ = support  # keep the Python object alive through the native call above
        py_setitem(fields, PythonObject("dual_coef_"),
                   svc_prefix(dual_all, (k - 1) * n_sv, Python.tuple(PythonObject(k - 1), PythonObject(n_sv)), "<f4", context))
        _ = n_support  # keep the Python object alive through the native call above
        _ = support  # keep the Python object alive through the native call above
    else:
        py_setitem(fields, PythonObject("support_"), family_array(Python.tuple(PythonObject(0)), "<i4", context))
        py_setitem(fields, PythonObject("support_vectors_"),
                   family_array(Python.tuple(PythonObject(0), PythonObject(n_cols)), "<f4", context))
        py_setitem(fields, PythonObject("dual_coef_"),
                   family_array(Python.tuple(PythonObject(k - 1), PythonObject(0)), "<f4", context))
    _ = dual_all  # keep the Python object alive through the native calls above
    _ = n_support  # keep the Python object alive through the native calls above
    _ = support  # keep the Python object alive through the native calls above
    py_setitem(fields, PythonObject("intercept_"), intercept)
    py_setitem(fields, PythonObject("n_support_"), n_support)
    py_setitem(fields, PythonObject("n_iter_"), n_iter)
    py_setitem(fields, PythonObject("_gamma"), PythonObject(gamma))
    py_setitem(fields, PythonObject("_label0"), PythonObject(0.0))
    py_setitem(fields, PythonObject("_label1"), PythonObject(1.0))
    _ = support^
    _ = dual_all^
    _platt(fields, binding, x_meta, codes, gamma, c_rows, k, context, settings, native)


def _svc_fit(model: PythonObject, args: PythonObject, context: PythonObject, native: PythonObject) raises:
    """`SVC.fit(X, y, sample_weight)`; args = (binding, X, y, sample_weight)."""
    var binding = args[0]
    var fields = py_fields(model)
    _svm_validate(model, fields, True, context)
    var x_meta = svc_snapshot(args[1], 2, "X", True, context)
    py_setitem(fields, PythonObject("input_copied_"), PythonObject(x_meta.copied))
    var x = x_meta.keep
    var y = args[2]
    var parsed = _as_labels(y, context)
    var labels = parsed[0]
    var classes = parsed[1]
    var pair = parsed[2]
    var n_rows = x_meta.shape[0]
    var n_cols = x_meta.shape[1]
    var n_y = Int(py=pair.shape[0]) if labels is None else Int(py=labels.shape[0])
    if n_y != n_rows:
        raise Error("mojolearn SVC: y has " + String(n_y) + " entries, X has " + String(n_rows) + " rows")
    if String(svc_field(fields, "_hp_kernel")) == "precomputed" and n_cols != n_rows:
        raise Error("mojolearn SVC: kernel='precomputed' needs the square n x n "
                    + "kernel matrix as X, got " + String(n_rows) + " x " + String(n_cols))
    var gamma = _resolve_gamma(fields, binding, x_meta, context, native)
    var c_rows = _c_rows(binding, svc_float(svc_field(fields, "_hp_C")), n_rows, args[3],
                         py_get(fields, PythonObject("class_weight"), PythonObject(None)), y, "SVC", context, native)
    var settings = _svc_fit_settings(fields, gamma, n_cols, context)
    if labels is None:
        _fit_ovo(fields, binding, x_meta, classes, pair, gamma, c_rows, context, settings, native)
        _svc_publish_dispatch(fields, native, context)
        _ = x^
        _ = x_meta  # retain captured storage owners through the last native use
        _ = c_rows^
        return
    if py_contains(fields, PythonObject("_pairs")):
        _ = family_call(fields.pop, PythonObject("_pairs"), PythonObject(None))
    var solved = _solve(fields, binding, x_meta.source.addr, n_rows, n_cols, _addr(labels), gamma,
                        0 if c_rows is None else _addr(c_rows), context, settings, native)
    _ = labels  # keep the Python object alive through the native call above
    var info = solved[4]
    _ = labels  # keep the Python object alive through the native call above
    var label0 = svc_round_f32(svc_read_f64(info, 3))
    var label1 = svc_round_f32(svc_read_f64(info, 4))
    # THE LABEL PAIR IS CHECKED, NOT ASSUMED (the solver's own host sort).
    var p0 = svc_float(pair[0])
    var p1 = svc_float(pair[1])
    if label0 != svc_round_f32(p0) or label1 != svc_round_f32(p1):
        raise Error("RuntimeError: mojolearn SVC: the solver's label pair "
                    + "(" + family_repr(PythonObject(label0)) + ", " + family_repr(PythonObject(label1)) + ") does not match the "
                    + "host's sorted unique (" + family_repr(PythonObject(p0)) + ", "
                    + family_repr(PythonObject(p1)) + "); the class mapping cannot be trusted")
    py_setitem(fields, PythonObject("classes_"), classes)
    py_setitem(fields, PythonObject("n_features_in_"), PythonObject(n_cols))
    _set_binary(fields, solved, n_cols, context)
    _ = labels  # keep the Python object alive through the native call above
    py_setitem(fields, PythonObject("_gamma"), PythonObject(gamma))
    py_setitem(fields, PythonObject("_label0"), PythonObject(label0))
    py_setitem(fields, PythonObject("_label1"), PythonObject(label1))
    if svc_truthy(svc_field(fields, "_hp_probability")):
        var wide = svc_threshold_codes(labels, label0, True, 0, 1, context)
        var n = Int(py=wide.size)
        var codes = family_array(Python.tuple(PythonObject(n)), "<i4", context)
        if n > 0:
            _ = family_call(svc_base("cast_elements", context), classical_address(wide), PythonObject(3),
                            classical_address(codes), PythonObject(2), PythonObject(n))
            _ = wide  # keep the Python object alive through the native call above
            _ = codes  # keep the Python object alive through the native call above
        _ = wide^
        _platt(fields, binding, x_meta, codes, gamma, c_rows, 2, context, settings, native)
    elif py_contains(fields, PythonObject("_prob_ab")):
        _ = family_call(fields.pop, PythonObject("_prob_ab"), PythonObject(None))
    _svc_publish_dispatch(fields, native, context)
    _ = x^
    _ = x_meta  # retain captured storage owners through the last native use
    _ = labels^
    _ = c_rows^


# ---------------------------------------------------------------- SVC inference


def _query(fields: PythonObject, source: PythonObject, context: PythonObject) raises -> ArrayMetadata:
    """`SVC._query`."""
    if not svc_has(fields, "dual_coef_"):
        raise Error("mojolearn SVC: call fit() first")
    var q = svc_snapshot(source, 2, "X", True, context)
    var d = svc_int(svc_field(fields, "n_features_in_"))
    if q.shape[1] != d:
        raise Error("mojolearn SVC: X has " + String(q.shape[1]) + " features, fit saw " + String(d))
    return q^


def _run(fields: PythonObject, binding: PythonObject, source: PythonObject, predict_class: Bool,
         context: PythonObject, native: PythonObject) raises -> PythonObject:
    """`SVC._run`: the binary machine's (n,) float32 decisions or labels."""
    var q_meta = _query(fields, source, context)
    var q = q_meta.keep
    var shape = q_meta.shape.copy()
    var n_rows = shape[0]
    var query_cols = shape[1]
    var settings = _svc_predict_settings(fields, context)
    var out = family_array(Python.tuple(PythonObject(n_rows)), "<f4", context)
    _machine(fields, binding, q, svc_field(fields, "dual_coef_"), svc_field(fields, "support_vectors_"),
             svc_int(svc_field(fields, "n_support_")), svc_read_f32(svc_field(fields, "intercept_"), 0),
             svc_float(svc_field(fields, "_label0")), svc_float(svc_field(fields, "_label1")), predict_class,
             svc_field(fields, "support_"), _addr(out), context, settings, native, n_rows, query_cols, q_meta.source.addr)
    _ = out  # keep the Python object alive through the native call above
    _ = q^
    _ = q_meta  # retain captured storage owners through the last native use
    _svc_publish_dispatch(fields, native, context)
    return out


def _pair_decisions(fields: PythonObject, binding: PythonObject, source: PythonObject,
                    context: PythonObject, native: PythonObject) raises -> PythonObject:
    """`SVC._pair_decisions`: each pair machine's raw decision, pair-major,
    in ONE float32 (n_pairs, n) Array."""
    var q_meta = _query(fields, source, context)
    var q = q_meta.keep
    var pairs = svc_field(fields, "_pairs")
    var shape = q_meta.shape.copy()
    var n_rows = shape[0]
    var query_cols = shape[1]
    var settings = _svc_predict_settings(fields, context)
    var dec = family_array(Python.tuple(PythonObject(len(pairs)), PythonObject(n_rows)), "<f4", context)
    var base = _addr(dec)
    _ = dec  # keep the Python object alive through the native call above
    for pno in range(len(pairs)):  # driver-loop: over the class pairs (one machine each)
        var p = pairs[pno]
        var dual = p["dual"]
        _machine(fields, binding, q, dual, p["sv"], Int(py=dual.shape[1]), svc_float(p["b"]), 0.0, 1.0, False,
                 p["support"], base + 4 * n_rows * pno, context, settings, native, n_rows, query_cols, q_meta.source.addr)
        _ = dec  # keep the Python object alive through the native call above
    _ = q^
    _ = q_meta  # retain captured storage owners through the last native use
    _svc_publish_dispatch(fields, native, context)
    return dec


def _pair_codes(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """The (i, j) class codes of the pairs, as int32 words."""
    var pairs = svc_field(fields, "_pairs")
    var out = family_array(Python.tuple(PythonObject(max(1, 2 * len(pairs)))), "<i4", context)
    var op = FAMILY_I32P(unsafe_from_address=_addr(out))
    _ = out  # keep the Python object alive through the native call above
    for i in range(len(pairs)):  # driver-loop: over the class pairs
        op[2 * i] = Int32(Int(py=pairs[i]["i"]))
        _ = out  # keep the Python object alive through the native call above
        op[2 * i + 1] = Int32(Int(py=pairs[i]["j"]))
        _ = out  # keep the Python object alive through the native call above
    return out


def _binary_codes(context: PythonObject) raises -> PythonObject:
    var out = family_array(Python.tuple(PythonObject(2)), "<i4", context)
    FAMILY_I32P(unsafe_from_address=_addr(out))[1] = 1
    _ = out  # keep the Python object alive through the native call above
    return out


def _epilogue(fields: PythonObject, binding: PythonObject, mode: Int, dec: PythonObject, n_pairs: Int, n_rows: Int,
              codes: PythonObject, shape: PythonObject, typestr: String, ab: PythonObject, label1: Float64,
              context: PythonObject, native: PythonObject) raises -> PythonObject:
    """`SVC._epilogue`: the per-row epilogues in the binding."""
    var out = family_array(shape, typestr, context)
    if n_rows > 0:
        var abuf = PythonObject(None)
        if ab is not None and len(ab) > 0:
            abuf = family_array(Python.tuple(PythonObject(2 * len(ab))), "<f8", context)
            var ap = FAMILY_F64P(unsafe_from_address=_addr(abuf))
            _ = abuf  # keep the Python object alive through the native call above
            for i in range(len(ab)):  # driver-loop: over the class pairs (their sigmoid parameters)
                ap[2 * i] = svc_float(ab[i][0])
                _ = abuf  # keep the Python object alive through the native call above
                ap[2 * i + 1] = svc_float(ab[i][1])
                _ = abuf  # keep the Python object alive through the native call above
        _ = svc_epilogue_execute(binding, native, classical_address(dec),
                        classical_address(codes) if n_pairs > 0 else PythonObject(0),
                        PythonObject(0) if abuf is None else classical_address(abuf), classical_address(out),
                        Python.list(PythonObject(mode), PythonObject(n_rows), PythonObject(n_pairs),
                                    PythonObject(len(svc_field(fields, "classes_"))), PythonObject(label1)))
        _ = abuf  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
        _ = abuf^
    return out


def _svc_decision(fields: PythonObject, binding: PythonObject, source: PythonObject,
                  context: PythonObject, native: PythonObject) raises -> PythonObject:
    """`SVC.decision_function`."""
    var pairs = py_get(fields, PythonObject("_pairs"), PythonObject(None))
    if pairs is None:
        return _run(fields, binding, source, False, context, native)
    var dec = _pair_decisions(fields, binding, source, context, native)
    var n_pairs = Int(py=dec.shape[0])
    var n_rows = Int(py=dec.shape[1])
    var codes = _pair_codes(fields, context)
    if String(svc_field(fields, "decision_function_shape")) == "ovo":
        return _epilogue(fields, binding, _EPI_OVO, dec, n_pairs, n_rows, codes,
                         Python.tuple(PythonObject(n_rows), PythonObject(n_pairs)), "<f4", PythonObject(None), 0.0, context, native)
    var k = len(svc_field(fields, "classes_"))
    return _epilogue(fields, binding, _EPI_OVR, dec, n_pairs, n_rows, codes,
                     Python.tuple(PythonObject(n_rows), PythonObject(k)), "<f8", PythonObject(None), 0.0, context, native)


def _svc_predict(fields: PythonObject, binding: PythonObject, source: PythonObject,
                 context: PythonObject, native: PythonObject) raises -> PythonObject:
    """`SVC.predict`: the binary epilogue's codes, or the one-vs-one vote
    (the 'ovr' argmax with break_ties)."""
    var pairs = py_get(fields, PythonObject("_pairs"), PythonObject(None))
    if pairs is None:
        var raw = _run(fields, binding, source, True, context, native)
        var n_rows = Int(py=raw.shape[0])
        var codes = _epilogue(fields, binding, _EPI_BINARY, raw, 1, n_rows, _binary_codes(context),
                              Python.tuple(PythonObject(n_rows)), "<i8", PythonObject(None),
                              svc_float(svc_field(fields, "_label1")), context, native)
        return svc_decode(svc_field(fields, "classes_"), codes, context)
    var dec = _pair_decisions(fields, binding, source, context, native)
    var n_pairs = Int(py=dec.shape[0])
    var n_rows = Int(py=dec.shape[1])
    var classes = svc_field(fields, "classes_")
    var k = len(classes)
    var pc = _pair_codes(fields, context)
    if svc_truthy(svc_field(fields, "_hp_break_ties")):
        var scores = _epilogue(fields, binding, _EPI_OVR, dec, n_pairs, n_rows, pc,
                               Python.tuple(PythonObject(n_rows), PythonObject(k)), "<f8", PythonObject(None), 0.0, context, native)
        return svc_decode(classes, svc_argmax_rows(scores, context), context)
    return svc_decode(classes, _epilogue(fields, binding, _EPI_VOTES, dec, n_pairs, n_rows, pc,
                                         Python.tuple(PythonObject(n_rows)), "<i8", PythonObject(None), 0.0, context, native),
                      context)


def _svc_proba(fields: PythonObject, binding: PythonObject, source: PythonObject, mode: Int,
               context: PythonObject, native: PythonObject) raises -> PythonObject:
    """`SVC.predict_proba` / `predict_log_proba` (`_proba`)."""
    var ab = py_get(fields, PythonObject("_prob_ab"), PythonObject(None))
    if ab is None:
        raise Error("AttributeError: mojolearn SVC: predict_proba is not available when "
                    + "probability=False (scikit-learn's rule); fit with probability=True")
    var k = len(svc_field(fields, "classes_"))
    var dec: PythonObject
    var codes: PythonObject
    var n_pairs: Int
    var n_rows: Int
    if py_get(fields, PythonObject("_pairs"), PythonObject(None)) is None:
        dec = _run(fields, binding, source, False, context, native)
        n_pairs = 1
        n_rows = Int(py=dec.shape[0])
        codes = _binary_codes(context)
    else:
        dec = _pair_decisions(fields, binding, source, context, native)
        n_pairs = Int(py=dec.shape[0])
        n_rows = Int(py=dec.shape[1])
        codes = _pair_codes(fields, context)
    return _epilogue(fields, binding, mode, dec, n_pairs, n_rows, codes,
                     Python.tuple(PythonObject(n_rows), PythonObject(k)), "<f8", ab, 0.0, context, native)


def _pred_count(pred: PythonObject) raises -> Int:
    if py_type(pred) is svc_builtins().list:
        return len(pred)
    return Int(py=pred.shape[0])


# ---------------------------------------------------------------- SVR


def _kernel_tail(fields: PythonObject, params: PythonObject) raises:
    """degree and coef0 as the optional trailing params, only for 'poly'
    and 'sigmoid'."""
    var k = String(svc_field(fields, "_hp_kernel"))
    if k == "poly" or k == "sigmoid":
        _ = py_call(params.append, family_call(svc_builtins().int, svc_field(fields, "_hp_degree")))
        _ = py_call(params.append, family_call(svc_builtins().float, svc_field(fields, "_hp_coef0")))


def _svr_fit(model: PythonObject, args: PythonObject, context: PythonObject, native: PythonObject) raises:
    """`SVR.fit(X, y, sample_weight)`; args = (binding, X, y, sample_weight)."""
    var binding = args[0]
    var fields = py_fields(model)
    _svm_validate(model, fields, False, context)
    var x_meta = svc_snapshot(args[1], 2, "X", True, context)
    py_setitem(fields, PythonObject("input_copied_"), PythonObject(x_meta.copied))
    var x = x_meta.keep
    var n_rows = x_meta.shape[0]
    var n_cols = x_meta.shape[1]
    # `_as_targets`: no finiteness check here (svr_fit names the index)
    var shape = svc_shape_of(args[2], context)
    if len(shape) != 1:
        raise Error("mojolearn SVR: y must be 1-D, got " + String(len(shape)) + "-D shape " + family_repr(shape))
    var targets = svc_f32(args[2], 1, "y", False, context)[0]
    if Int(py=targets.shape[0]) != n_rows:
        raise Error("mojolearn SVR: y has " + String(Int(py=targets.shape[0])) + " entries, X has " + String(n_rows) + " rows")
    if String(svc_field(fields, "_hp_kernel")) == "precomputed" and n_cols != n_rows:
        raise Error("mojolearn SVR: kernel='precomputed' needs the square n x n "
                    + "kernel matrix as X, got " + String(n_rows) + " x " + String(n_cols))
    var gamma = _resolve_gamma(fields, binding, x_meta, context, native)
    var c_rows = _c_rows(binding, svc_float(svc_field(fields, "_hp_C")), n_rows, args[3], PythonObject(None),
                         PythonObject(None), "SVR", context, native)
    # WORST-CASE OUTPUT BUFFERS: at most n_rows coefficients (DEVIATION 873)
    var dual = family_array(Python.tuple(PythonObject(n_rows)), "<f4", context)
    var support = family_array(Python.tuple(PythonObject(n_rows)), "<i4", context)
    var sv = family_array(Python.tuple(PythonObject(n_rows * n_cols)), "<f4", context)
    var info = family_array(Python.tuple(PythonObject(3)), "<f8", context)
    # ORDER MATCHES bindings/_mojolearn_svm.mojo::svr_fit_binding: n_rows,
    # n_features, kernel, gamma, C, epsilon, tol, max_iter,
    # nochange_steps[, the per-row bounds' address[, degree, coef0]]
    var params = Python.list(PythonObject(n_rows), PythonObject(n_cols), PythonObject(_kernel_code(fields, context)),
                             PythonObject(gamma), svc_field(fields, "_hp_C"), svc_field(fields, "_hp_epsilon"),
                             svc_field(fields, "_hp_tol"), svc_field(fields, "_hp_max_iter"),
                             svc_field(fields, "_hp_nochange_steps"))
    var k = String(svc_field(fields, "_hp_kernel"))
    if c_rows is not None or k == "poly" or k == "sigmoid":
        _ = py_call(params.append, PythonObject(0) if c_rows is None else classical_address(c_rows))
        _ = c_rows  # keep the Python object alive through the native call above
        _kernel_tail(fields, params)
        _ = c_rows  # keep the Python object alive through the native call above
    var n_support = svc_int(svr_execute(binding, native, True, Python.list(PythonObject(x_meta.source.addr), classical_address(targets),
                                        classical_address(dual), classical_address(support), classical_address(sv),
                                        classical_address(info)), params))
    _ = targets  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = sv  # keep the Python object alive through the native call above
    _ = support  # keep the Python object alive through the native call above
    _ = dual  # keep the Python object alive through the native call above
    _ = info  # keep the Python object alive through the native call above
    _ = c_rows  # keep the Python object alive through the native call above
    py_setitem(fields, PythonObject("n_features_in_"), PythonObject(n_cols))
    py_setitem(fields, PythonObject("n_support_"), PythonObject(n_support))
    _ = info  # keep the Python object alive through the native call above
    _ = sv  # keep the Python object alive through the native call above
    _ = support  # keep the Python object alive through the native call above
    _ = dual  # keep the Python object alive through the native call above
    _ = targets  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, PythonObject("n_iter_"), PythonObject(Int(svc_read_f64(info, 2))))
    py_setitem(fields, PythonObject("intercept_"), svc_scalar_array(svc_round_f32(svc_read_f64(info, 0)), "<f4", context))
    py_setitem(fields, PythonObject("support_"), svc_prefix(support, n_support, Python.tuple(PythonObject(n_support)), "<i4", context))
    _ = info  # keep the Python object alive through the native call above
    _ = sv  # keep the Python object alive through the native call above
    _ = dual  # keep the Python object alive through the native call above
    _ = targets  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, PythonObject("support_vectors_"),
               svc_prefix(sv, n_support * n_cols, Python.tuple(PythonObject(n_support), PythonObject(n_cols)), "<f4", context))
    _ = info  # keep the Python object alive through the native call above
    _ = support  # keep the Python object alive through the native call above
    _ = dual  # keep the Python object alive through the native call above
    _ = targets  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, PythonObject("dual_coef_"),
               svc_prefix(dual, n_support, Python.tuple(PythonObject(1), PythonObject(n_support)), "<f4", context))
    _ = info  # keep the Python object alive through the native call above
    _ = sv  # keep the Python object alive through the native call above
    _ = support  # keep the Python object alive through the native call above
    _ = targets  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    py_setitem(fields, PythonObject("_gamma"), PythonObject(gamma))
    _ = x^
    _ = x_meta  # retain captured storage owners through the last native use
    _ = targets^
    _ = c_rows^


def _svr_predict(fields: PythonObject, binding: PythonObject, source: PythonObject,
                 context: PythonObject, native: PythonObject) raises -> PythonObject:
    """`SVR.predict`."""
    if not svc_has(fields, "dual_coef_"):
        raise Error("mojolearn SVR: call fit() first")
    var q_meta = svc_snapshot(source, 2, "X", True, context)
    var q = q_meta.keep
    var d = svc_int(svc_field(fields, "n_features_in_"))
    if q_meta.shape[1] != d:
        raise Error("mojolearn SVR: X has " + String(q_meta.shape[1]) + " features, fit saw " + String(d))
    var n_rows = q_meta.shape[0]
    var out = family_array(Python.tuple(PythonObject(n_rows)), "<f4", context)
    var dual = svc_field(fields, "dual_coef_")
    var sv = svc_field(fields, "support_vectors_")
    var n_support = svc_int(svc_field(fields, "n_support_"))
    var n_features = d
    var query = q
    var query_address = q_meta.source.addr
    var held = PythonObject(None)
    if String(svc_field(fields, "_hp_kernel")) == "precomputed" and n_support > 0:
        # the cross-kernel's columns at the support vectors (SVC's rule)
        held = _i32_array(svc_field(fields, "support_"), context)
        query = _gather(binding, q_meta.source.addr, n_rows, q_meta.shape[1], 0, n_rows, _addr(held), Int(py=held.shape[0]), context, native)
        _ = held  # keep the Python object alive through the native call above
        _ = q  # keep the Python object alive through the native call above
        query_address = _addr(query)
        n_features = n_support
        sv = dual
    # ORDER MATCHES bindings/_mojolearn_svm.mojo::svr_predict_binding:
    # n_rows, n_features, n_support, b, kernel, gamma, cache_size_mib
    var params = Python.list(PythonObject(n_rows), PythonObject(n_features), PythonObject(n_support),
                             PythonObject(svc_read_f32(svc_field(fields, "intercept_"), 0)),
                             PythonObject(_kernel_code(fields, context)), svc_field(fields, "_gamma"),
                             svc_field(fields, "_hp_cache_size"))
    _kernel_tail(fields, params)
    _ = svr_execute(binding, native, False, Python.list(PythonObject(query_address),
                    classical_address(dual) if n_support > 0 else PythonObject(0),
                    classical_address(sv) if n_support > 0 else PythonObject(0),
                    classical_address(out)), params)
    _ = query  # keep the Python object alive through the native call above
    _ = dual  # keep the Python object alive through the native call above
    _ = sv  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = query^
    _ = held^
    _ = q^
    _ = q_meta  # retain captured storage owners through the last native use
    return out


def _svr_score(fields: PythonObject, binding: PythonObject, source: PythonObject, y: PythonObject,
               context: PythonObject, native: PythonObject) raises -> PythonObject:
    """`SVR.score`: R^2, 1.0 for a perfect constant fit, 0.0 otherwise."""
    var pred = _svr_predict(fields, binding, source, context, native)
    var shape = svc_shape_of(y, context)
    if len(shape) != 1:
        raise Error("mojolearn SVR: y must be 1-D, got " + String(len(shape)) + "-D shape " + family_repr(shape))
    var n = Int(py=pred.shape[0])
    if Int(py=shape[0]) != n:
        raise Error("mojolearn SVR: y has " + family_str(shape[0]) + " entries, X has " + String(n) + " rows")
    var sums = svc_r2_sums(pred, y, context)
    var ss_res = Float64(py=sums[0])
    var ss_tot = Float64(py=sums[1])
    if ss_tot == 0.0:
        return PythonObject(1.0 if ss_res == 0.0 else 0.0)
    return PythonObject(1.0 - ss_res / ss_tot)


def _coef_snapshot(value: PythonObject, name: String, context: PythonObject) raises -> ArrayMetadata:
    if py_type(value) is not context["array_class"] and py_type(value) is not py_get(context, "ndarray"):
        raise Error("TypeError: " + name + " requires an admitted native array")
    classical_admit_value(value, context)
    var shape = array_shape_once(value, context)
    var captured = array_metadata_once(value, shape^, name, context)
    if len(captured.shape) != 2 or not captured.f32_c() or not captured.native_byte_order:
        raise Error("TypeError: " + name + " requires a contiguous native float32 matrix")
    return captured^


def _linear_coef(fields: PythonObject, binding: PythonObject, classifier: Bool,
                 context: PythonObject, native: PythonObject) raises -> PythonObject:
    # Public coef_ and composed RFE use the same admitted metadata and GEMV.
    var b = native_loaded_module("builtins")
    var name = String("SVC" if classifier else "SVR")
    if not svc_has(fields, "dual_coef_"):
        raise Error("AttributeError: mojolearn " + name + ": call fit() first")
    if py_type(fields["_hp_kernel"]) is not b.str:
        raise Error("TypeError: fitted SVM kernel requires an exact native string")
    if String(fields["_hp_kernel"]) != "linear":
        raise Error("AttributeError: mojolearn " + name + ": coef_ is only available for kernel='linear'")
    host_inference_scalar(fields["n_features_in_"], context)
    var d = Int(py=fields["n_features_in_"])
    if d < 0:
        raise Error("ValueError: SVM feature count cannot be negative")
    var pairs = py_get(fields, "_pairs") if classifier else PythonObject(None)
    if pairs is not None and py_type(pairs) is not b.list and py_type(pairs) is not b.tuple:
        raise Error("TypeError: SVM pair state requires an exact list or tuple")
    var count = len(pairs) if pairs is not None else 1
    var duals = List[ArrayMetadata]()
    var supports = List[ArrayMetadata]()
    for row in range(count):  # driver-loop: capture each fitted buffer exactly once before execution
        var pair = pairs[row] if pairs is not None else fields
        if py_type(pair) is not b.dict:
            raise Error("TypeError: SVM pair state requires an exact dictionary")
        var keys = py_call(b.list, pair)
        for i in range(len(keys)):  # driver-loop: reject key equality/hash hooks before field lookups
            if py_type(keys[i]) is not b.str:
                raise Error("TypeError: SVM fitted state requires exact string keys")
        var dual = _coef_snapshot(pair["dual" if pairs is not None else "dual_coef_"], "dual_coef_", context)
        var sv = _coef_snapshot(pair["sv" if pairs is not None else "support_vectors_"], "support_vectors_", context)
        if dual.shape[0] != 1 or dual.shape[1] != sv.shape[0] or sv.shape[1] != d:
            raise Error("ValueError: SVM coefficient and support-vector shapes disagree")
        duals.append(dual^)
        supports.append(sv^)
    var out = family_array(Python.tuple(count, d), "<f4", context)
    var address = Int(py=classical_address(out))
    var entry = neural_descriptor_entry(binding, "svc_dual_gemv", False) if native is None else PythonObject(None)
    for row in range(count):  # driver-loop: one existing GEMV per fitted pair; captured owners remain live
        var n = supports[row].shape[0]
        if d > 0 and n > 0:
            if native is not None:
                _ = svc_descriptor_execute(native, 11, [duals[row].source.addr, supports[row].source.addr, address + 4 * row * d],
                    [n, d, 1 if pairs is not None else 0], [])
            else:
                _ = family_call(entry, PythonObject(duals[row].source.addr), PythonObject(supports[row].source.addr),
                    Python.list(n, d, 1 if pairs is not None else 0), PythonObject(address + 4 * row * d))
    _ = duals
    _ = supports
    return out


# ---------------------------------------------------------------- dispatch


def _svm_lifecycle_admit(model: PythonObject, fields: PythonObject, context: PythonObject) raises:
    """Fit bookkeeping read by the public lifecycle: exact native values,
    and no Python completion callback a native fit could be asked to run."""
    var b = native_loaded_module("builtins")
    if py_type(py_get(fields, "_mojolearn_fitting", PythonObject(0))) is not b.int:
        raise Error("TypeError: SVM fitting depth requires an exact integer")
    var fitted = py_get(fields, "_mojolearn_fitted", PythonObject(False))
    if py_type(fitted) is not b.bool and py_type(fitted) is not b.int:
        raise Error("TypeError: SVM fitted state requires a native boolean or integer")
    var ancestry = _type_mro(py_type(model))
    for i in range(len(ancestry)):  # driver-loop: no completion callback may execute from native fit
        if py_contains(_class_fields(ancestry[i]), "_mojolearn_after_fit"):
            raise Error("TypeError: SVM fitting cannot invoke Python completion callbacks")
    if py_contains(fields, "_mojolearn_after_fit"):
        raise Error("TypeError: SVM fitting cannot invoke Python completion callbacks")


def svc_kernel_operation(model: PythonObject, operation: String, args: PythonObject,
                         context: PythonObject) raises -> PythonObject:
    """The kernel machines' whole public operation: argument admission, the
    fitted and feature-name checks, fit bookkeeping (depth, fitted flag,
    feature names, public class labels), the algorithm through one admitted
    provider, and NumPy results for callers outside the package."""
    classical_connect_numpy(context)
    var b = native_loaded_module("builtins")
    if py_type(args) is not b.tuple:
        raise Error("TypeError: SVM public arguments require an exact tuple")
    var fields = py_fields(model)
    var classifier = operation.startswith("family_svc_svc_")
    if operation == "family_svc_svc_validate" or operation == "family_svc_svr_validate":
        _svm_validate(model, fields, classifier, context)
        return PythonObject(None)
    var fitting = operation == "family_svc_svc_fit" or operation == "family_svc_svr_fit"
    var public = (operation == "family_svc_svc_decision" or operation == "family_svc_svc_predict"
                  or operation == "family_svc_svc_proba" or operation == "family_svc_svc_log_proba"
                  or operation == "family_svc_svc_score" or operation == "family_svc_svr_predict"
                  or operation == "family_svc_svr_score")
    if not fitting and not public:
        # coef_ (the native property schema) and composed callers: algorithm only.
        var native_only = _svc_provider(args[0])
        var plain = _svc_kernel_operation(model, operation, args, context, fields, native_only)
        _svc_publish_dispatch(fields, native_only, context)
        return plain
    _svm_lifecycle_admit(model, fields, context)
    var method = String("fit")
    if operation.endswith("_decision"):
        method = "decision_function"
    elif operation.endswith("_log_proba"):
        method = "predict_log_proba"
    elif operation.endswith("_proba"):
        method = "predict_proba"
    elif operation.endswith("_predict"):
        method = "predict"
    elif operation.endswith("_score"):
        method = "score"
    var depth = Int(py=py_get(fields, "_mojolearn_fitting", PythonObject(0)))
    if public:
        classical_require_fitted(model, False, method, context)
        if depth == 0:
            estimator_feature_check(model, fields, args[1], context)
    var native = _svc_provider(args[0])
    var initial_depth = estimator_fit_begin(fields) if fitting else 0
    var result = PythonObject(None)
    try:
        result = _svc_kernel_operation(model, operation, args, context, fields, native)
    finally:
        if fitting:
            estimator_fit_end(model, initial_depth)
    # A nested set_params may have replaced the dictionary's contents.
    fields = py_fields(model)
    _svc_publish_dispatch(fields, native, context)
    if fitting:
        estimator_fit_complete(model, args[1], classifier, context)
        return result
    if (method == "score" or not estimator_outside() or py_get(context, "ndarray") is None
            or py_type(result) is not context["array_class"]):
        return result
    if not classifier and method == "predict":
        # scikit-learn's regression predictions are float64: widened exactly.
        var wide = classical_typed(result.shape, "<f8", context)
        labels_cast(Int(py=classical_address(result)), "<f4", Int(py=classical_address(wide)), "<f8",
                    Int(py=result.size), context)
        _ = result^
        return wide
    return native_array_ndarray_view(result, context)


def _svc_kernel_operation(model: PythonObject, operation: String, args: PythonObject,
                          context: PythonObject, fields: PythonObject, native: PythonObject) raises -> PythonObject:
    if operation == "family_svc_svc_coef" or operation == "family_svc_svr_coef":
        return _linear_coef(fields, args[0], operation == "family_svc_svc_coef", context, native)
    if operation == "family_svc_svc_fit":
        _svc_fit(model, args, context, native)
        return model
    if operation == "family_svc_svc_decision":
        return _svc_decision(fields, args[0], args[1], context, native)
    if operation == "family_svc_svc_predict":
        return _svc_predict(fields, args[0], args[1], context, native)
    if operation == "family_svc_svc_proba":
        return _svc_proba(fields, args[0], args[1], _EPI_PROBA, context, native)
    if operation == "family_svc_svc_log_proba":
        return _svc_proba(fields, args[0], args[1], _EPI_LOG_PROBA, context, native)
    if operation == "family_svc_svc_score":
        var pred = _svc_predict(fields, args[0], args[1], context, native)
        return PythonObject(svc_accuracy(args[2], pred, _pred_count(pred), context))
    if operation == "family_svc_svr_fit":
        _svr_fit(model, args, context, native)
        return model
    if operation == "family_svc_svr_predict":
        return _svr_predict(fields, args[0], args[1], context, native)
    if operation == "family_svc_svr_score":
        return _svr_score(fields, args[0], args[1], args[2], context, native)
    raise Error("RuntimeError: unknown kernel SVM operation " + operation)

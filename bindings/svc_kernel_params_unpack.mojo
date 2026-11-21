# SPDX-License-Identifier: Apache-2.0
# UNPACK of the kernel SVMs' parameters (SVC and SVR;
# bindings/svc_kernel_native_door.mojo sequences the operations):
# `_validate_params` with its refusals in the same wording and order, the
# kernel code, gamma ('auto', 'scale' from the exact limbs, or the float)
# and the per-row bounds C * class_weight * sample_weight.
from std.python import Python, PythonObject
from bindings.base_helpers_key_unpack import base_helpers_key
from bindings.base_helpers_released_unpack import base_bincount_i64
from bindings.admitted_family_door import admit_family_weights
from bindings.native_loaded_module_door import native_loaded_module
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.native_owner_lifetime_door import native_release_admit
from bindings.label_admission_native_unpack import labels_primitive_kind
from bindings.py_balanced import py_call, py_type, py_contains, py_setitem, py_item
from bindings.classical_arrays_door import classical_address
from bindings.family_common_unpack import FAMILY_F64P, family_repr, family_isinstance
from bindings.family_common_pack import family_call, family_array
from bindings.svc_common_unpack import svc_builtins, svc_field, svc_float, svc_int, svc_truthy, svc_base, svc_zero_division
from bindings.svc_common_run import svc_isfinite
from providers.python.array_metadata_pyapi_door import ArrayMetadata, ArrayPin
from bindings.svm_keys_unpack import svm_keys
from svm.operation_metadata import SvcOperationSettings
from core.svc_native_api import SvcDrivers
from bindings.svc_kernel_rows_unpack import _addr

# `_scale_gamma`: the py2mojo switch bit.
comptime _PY2MOJO_SCALE_GAMMA = 1
comptime _MAX_POLY_DEGREE = 32


# ---------------------------------------------------------------- parameters


def _svm_integral(value: PythonObject, context: PythonObject) raises -> Bool:
    return labels_primitive_kind(py_type(value), context) == 3


def _svm_parameter_admission(fields: PythonObject, classifier: Bool, context: PythonObject) raises:
    var b = native_loaded_module("builtins")
    var names = Python.tuple("kernel", "degree", "gamma", "coef0", "C", "tol", "cache_size", "max_iter", "nochange_steps", "verbose", "output_type")
    for i in range(len(names)):  # driver-loop: scalar controls before native conversion, comparison or truth slots
        host_inference_scalar(fields[names[i]], context, True)
    var nk = svm_keys()
    if classifier:
        for name in ["probability", "random_state", "decision_function_shape", "break_ties"]:  # small-loop(fixed classifier controls)
            host_inference_scalar(fields[name], context, True)
        var weights = py_item(fields, nk[].k_class_weight)
        if py_type(weights) is b.dict:
            var items = py_call(b.list, weights.items())
            for i in range(len(items)):  # driver-loop: class-weight hashing/comparison cannot invoke Python scalar protocols
                host_inference_scalar(items[i][0], context)
                host_inference_scalar(items[i][1], context)
        else:
            host_inference_scalar(weights, context)
    else:
        host_inference_scalar(py_item(fields, nk[].k_epsilon), context)
    var replaced = Python.tuple("_hp_C", "_hp_kernel", "_hp_degree", "_hp_gamma", "_hp_coef0", "_hp_tol",
                                "_hp_cache_size", "_hp_max_iter", "_hp_nochange_steps")
    for i in range(len(replaced)):  # driver-loop: only working fields the validator actually overwrites
        if py_contains(fields, replaced[i]):
            native_release_admit(fields[replaced[i]], context)
    if classifier:
        for name in ["_hp_random_state", "_hp_probability", "_hp_break_ties"]:  # small-loop(fixed classifier working state)
            if py_contains(fields, name):
                native_release_admit(fields[name], context)
    elif py_contains(fields, nk[].k__hp_epsilon):
        native_release_admit(py_item(fields, nk[].k__hp_epsilon), context)


def _svm_validate(model: PythonObject, fields: PythonObject, classifier: Bool, context: PythonObject) raises:
    """`SVC._validate_params` / `SVR._validate_params`: the refusals in the
    Python's order and wording, then the checked `_hp_*` working values."""
    _svm_parameter_admission(fields, classifier, context)
    var b = svc_builtins()
    var operator = native_loaded_module("operator")
    var head = String("mojolearn SVC: ") if classifier else String("mojolearn SVR: ")
    var nk = svm_keys()
    var kernels = py_item(context, nk[].k_family_svc_kernels)
    var refused = context["family_svc_refused" if classifier else "family_svr_refused"]
    var kernel = svc_field(fields, nk[].k_kernel, "kernel")
    var degree = svc_field(fields, nk[].k_degree, "degree")
    var gamma = svc_field(fields, nk[].k_gamma, "gamma")
    var coef0 = svc_field(fields, nk[].k_coef0, "coef0")
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
    var c = family_call(b.float, svc_field(fields, nk[].k_C, "C"))
    if not svc_isfinite(Float64(py=c)):
        raise Error(head + "C must be finite, got " + family_repr(c) + " (DEVIATION 636)")
    if Float64(py=c) <= 0.0:
        raise Error(head + "C must be positive, got " + family_repr(c))
    var tol = family_call(b.float, svc_field(fields, nk[].k_tol, "tol"))
    if not svc_isfinite(Float64(py=tol)):
        raise Error(head + "tol must be finite, got " + family_repr(tol) + " (DEVIATION 636)")
    if Float64(py=tol) <= 0.0:
        raise Error(head + "tol must be positive, got " + family_repr(tol))
    var epsilon = PythonObject(None)
    if not classifier:
        # converted, not refused: the Mojo fit's own refusals stay reachable
        epsilon = family_call(b.float, svc_field(fields, nk[].k_epsilon, "epsilon"))
    var cache = family_call(b.float, svc_field(fields, nk[].k_cache_size, "cache_size"))
    var cache_value = Float64(py=cache)
    if not (cache_value > 0.0) or not svc_isfinite(cache_value):
        raise Error(head + "cache_size is the prediction buffer here "
                    + "(DEVIATION 871) and must be a positive finite MiB, got " + family_repr(cache))
    var max_iter = family_call(b.int, svc_field(fields, nk[].k_max_iter, "max_iter"))
    if (Bool(py=family_call(operator.eq, max_iter, PythonObject(0)))
            or Bool(py=family_call(operator.lt, max_iter, PythonObject(-1)))):
        raise Error(head + "max_iter is a positive cap or -1 for no "
                    + "limit (cuML's default), got " + family_repr(max_iter))
    var nochange = family_call(b.int, svc_field(fields, nk[].k_nochange_steps, "nochange_steps"))
    if Bool(py=family_call(operator.lt, nochange, PythonObject(0))):
        raise Error(head + "nochange_steps cannot be negative, got " + family_repr(nochange))
    if svc_truthy(svc_field(fields, nk[].k_verbose, "verbose")):
        raise Error("NotImplementedError: " + head + "verbose is refused; upstream it selects "
                    + "CUML_LOG_DEBUG lines and this implementation prints none, so honoring "
                    + "it is impossible and accepting it would be accepting-and-"
                    + "ignoring")
    if svc_field(fields, nk[].k_output_type, "output_type") is not None:
        raise Error("NotImplementedError: " + head + "output_type is a cuML-internal array-type "
                    + "selector; this package returns mojolearn Arrays")
    if not classifier:
        py_setitem(fields, nk[].k__hp_kernel, k)
        py_setitem(fields, nk[].k__hp_degree, degree)
        py_setitem(fields, nk[].k__hp_gamma, gamma)
        py_setitem(fields, nk[].k__hp_coef0, coef0)
        py_setitem(fields, nk[].k__hp_tol, tol)
        py_setitem(fields, nk[].k__hp_C, c)
        py_setitem(fields, nk[].k__hp_epsilon, epsilon)
        py_setitem(fields, nk[].k__hp_cache_size, cache)
        py_setitem(fields, nk[].k__hp_max_iter, max_iter)
        py_setitem(fields, nk[].k__hp_nochange_steps, nochange)
        return
    var probability = svc_truthy(svc_field(fields, nk[].k_probability, "probability"))
    var random_state = svc_field(fields, nk[].k_random_state, "random_state")
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
    var class_weight = svc_field(fields, nk[].k_class_weight, "class_weight")
    if class_weight is not None and not family_isinstance(class_weight, Python.tuple(b.dict, b.str)):
        raise Error("mojolearn SVC: class_weight is a dict, 'balanced' or None, "
                    + "got " + String(py_type(class_weight).__name__))
    var shape = svc_field(fields, nk[].k_decision_function_shape, "decision_function_shape")
    if not py_contains(Python.tuple(PythonObject("ovo"), PythonObject("ovr")), shape):
        raise Error("mojolearn SVC: decision_function_shape is 'ovo' or 'ovr', got " + family_repr(shape))
    var break_ties = svc_truthy(svc_field(fields, nk[].k_break_ties, "break_ties"))
    if break_ties and Bool(py=family_call(operator.eq, shape, PythonObject("ovo"))):
        raise Error("mojolearn SVC: break_ties must be False when "
                    + "decision_function_shape is 'ovo' (scikit-learn's rule)")
    py_setitem(fields, nk[].k__hp_C, c)
    py_setitem(fields, nk[].k__hp_kernel, k)
    py_setitem(fields, nk[].k__hp_degree, degree)
    py_setitem(fields, nk[].k__hp_gamma, gamma)
    py_setitem(fields, nk[].k__hp_coef0, coef0)
    py_setitem(fields, nk[].k__hp_tol, tol)
    py_setitem(fields, nk[].k__hp_cache_size, cache)
    py_setitem(fields, nk[].k__hp_max_iter, max_iter)
    py_setitem(fields, nk[].k__hp_nochange_steps, nochange)
    py_setitem(fields, nk[].k__hp_random_state,
               PythonObject(None) if random_state is None else family_call(b.int, random_state))
    py_setitem(fields, nk[].k__hp_probability, PythonObject(probability))
    py_setitem(fields, nk[].k__hp_break_ties, PythonObject(break_ties))


def _kernel_code(fields: PythonObject, context: PythonObject) raises -> Int:
    var nk = svm_keys()
    return Int(py=py_item(context, nk[].k_family_svc_kernels)[svc_field(fields, nk[].k__hp_kernel, "_hp_kernel")])


def svm_scale_gamma_admit(hp: SvmSettings, drivers: SvcDrivers, x_meta: ArrayMetadata) raises:
    """The refusals of gamma='scale' on host words, before RUN resolves it
    (svm/svc_kernel_fit_run.mojo svc_gamma_run): the binding has the exact
    limbs, and X has a cell."""
    if hp.gamma_kind == SVM_GAMMA_SCALE:
        if (drivers.flags() & _PY2MOJO_SCALE_GAMMA) == 0:
            raise Error("RuntimeError: mojolearn: this binding has no scale_gamma_limbs (an older binary or a "
                        + "-D MOJOLEARN_PY2MOJO_linear_OFF build); rebuild it")
        if x_meta.count == 0:
            raise Error("gamma='scale' needs at least one cell of X")


struct SvmWeightsUnpacked(Movable):
    """The operands of the per-row bounds C * class_weight[y_i] *
    sample_weight_i (svm/svc_kernel_fit_run.mojo svc_bounds_run): the
    float64 sample weights and the float64 class weights (address 0: that
    factor is absent), with their owners. `weighted`: any factor present."""
    var owners: PythonObject
    var sample_weights: Int
    var class_weights: Int
    var weighted: Bool

    def __init__(out self, owners: PythonObject, sample_weights: Int, class_weights: Int, weighted: Bool):
        self.owners = owners
        self.sample_weights = sample_weights
        self.class_weights = class_weights
        self.weighted = weighted


def svm_weights_unpack(n_rows: Int, sample_weight: PythonObject, class_weight: PythonObject, classes: PythonObject,
                       codes: PythonObject, who: String, context: PythonObject) raises -> SvmWeightsUnpacked:
    """`_svm_impl._c_rows` up to its launch: sample_weight admitted, then
    the class-weight vector ('balanced' from the class counts, or the
    user's dict over 1.0). `classes` and `codes` are the fit's one label
    encoding (None for SVR, which has no class weights)."""
    var b = svc_builtins()
    var owners = Python.list()
    if sample_weight is None and class_weight is None:
        return SvmWeightsUnpacked(owners, 0, 0, False)
    var sw_address = 0
    if sample_weight is not None:
        var weights_pin = ArrayPin()
        var weights = admit_family_weights(sample_weight, n_rows, False, context, weights_pin, True)
        _ = py_call(owners.append, weights_pin.keep)
        _ = py_call(owners.append, weights_pin.owner)
        sw_address = weights.source.addr
    var cw_address = 0
    if class_weight is not None:
        var n_classes = len(classes)
        var cwa = family_array(Python.tuple(PythonObject(n_classes)), "<f8", context)
        if family_isinstance(class_weight, b.str):
            if String(class_weight) != "balanced":
                raise Error("mojolearn " + who + ": class_weight is a dict, 'balanced' or None, got "
                            + family_repr(class_weight))
            var cnt = family_array(Python.tuple(PythonObject(n_classes)), "<i8", context)
            var nk = svm_keys()
            _ = base_bincount_i64(base_helpers_key(py_item(context, nk[].k_family_svc_base_helpers)),
                                  Int(py=classical_address(codes)), 2, Int(py=codes.shape[0]), n_classes,
                                  Int(py=classical_address(cnt)), False)
            if n_classes > 0 and Int(py=family_call(svc_base("class_ratio_f64", context), classical_address(cnt),
                                                     PythonObject(n_classes), PythonObject(n_rows), PythonObject(0),
                                                     classical_address(cwa))) != 0:
                svc_zero_division(False)
            _ = cnt  # keep the Python object alive through the native calls above
            _ = codes
        else:
            var cp = FAMILY_F64P(unsafe_from_address=_addr(cwa))
            for i in range(n_classes):  # driver-loop: over the classes (the default weight 1.0)
                cp[i] = 1.0
            var items = family_call(b.list, family_call(family_call(b.dict, class_weight).items))
            for i in range(len(items)):  # driver-loop: over the user's class_weight entries
                var key = items[i][0]
                if not py_contains(classes, key):
                    raise Error("mojolearn " + who + ": class_weight names the label " + family_repr(key)
                                + ", which is not in y's classes " + family_repr(classes))
                cp[Int(py=family_call(classes.index, key))] = svc_float(items[i][1])
        _ = py_call(owners.append, cwa)
        cw_address = _addr(cwa)
    return SvmWeightsUnpacked(owners, sw_address, cw_address, True)


# ---------------------------------------------------------------- the settings record


comptime SVM_GAMMA_VALUE = 0
comptime SVM_GAMMA_AUTO = 1
comptime SVM_GAMMA_SCALE = 2


@fieldwise_init
struct SvmSettings(Copyable, Movable):
    """The checked parameters of one kernel SVM operation as plain words,
    read ONCE after validation: no later step of the operation reads a
    `_hp_*` field again. `gamma_kind` is SVM_GAMMA_VALUE (then `gamma` is
    the float), SVM_GAMMA_AUTO or SVM_GAMMA_SCALE; `epsilon` is SVR's (0.0
    for SVC); `probability` is SVC's (False for SVR)."""
    var kernel: Int
    var precomputed: Bool
    var degree: Int
    var coef0: Float64
    var gamma_kind: Int
    var gamma: Float64
    var c: Float64
    var tol: Float64
    var epsilon: Float64
    var max_iter: Int
    var nochange_steps: Int
    var cache_mib: Float64
    var probability: Bool

    def device_gamma(self, n_cols: Int) -> Tuple[Float64, Bool]:
        """`_resolve_gamma` for the device-X entry: 'auto' 1 / n_cols, a
        float as given, and 'scale' (True) resolved by the entry from its
        own device copy of X (the same exact limbs, so the same gamma; 1.0
        is the placeholder the entry replaces)."""
        if self.gamma_kind == SVM_GAMMA_AUTO:
            return (1.0 / Float64(n_cols), False)
        if self.gamma_kind == SVM_GAMMA_SCALE:
            return (1.0, True)
        return (self.gamma, False)

    def fit_operation(self, gamma: Float64, features: Int) -> SvcOperationSettings:
        """The binary solver's operation words with the gamma the fit resolved."""
        return SvcOperationSettings(self.kernel, self.degree, self.max_iter, self.nochange_steps, features, gamma,
                                    self.c, self.tol, self.coef0, self.cache_mib)


def svm_settings_unpack(model: PythonObject, fields: PythonObject, classifier: Bool,
                        context: PythonObject) raises -> SvmSettings:
    """`_validate_params` (its refusals, then the `_hp_*` working values it
    publishes), then those values converted once. SVC converts degree and
    coef0 for every kernel; SVR only for 'poly' and 'sigmoid' (3 and 0.0
    otherwise), as each fit always did."""
    _svm_validate(model, fields, classifier, context)
    var nk = svm_keys()
    var name = String(svc_field(fields, nk[].k__hp_kernel, "_hp_kernel"))
    var kernel = _kernel_code(fields, context)
    var degree = 3
    var coef0 = Float64(0)
    if classifier or name == "poly" or name == "sigmoid":
        degree = svc_int(svc_field(fields, nk[].k__hp_degree, "_hp_degree"))
        coef0 = svc_float(svc_field(fields, nk[].k__hp_coef0, "_hp_coef0"))
    var g = svc_field(fields, nk[].k__hp_gamma, "_hp_gamma")
    var gamma_kind = SVM_GAMMA_VALUE
    var gamma = 0.0
    if family_isinstance(g, svc_builtins().str):
        # validation left only these two names
        gamma_kind = SVM_GAMMA_AUTO if String(g) == "auto" else SVM_GAMMA_SCALE
    else:
        gamma = Float64(py=g)
    var epsilon = 0.0
    var probability = False
    if classifier:
        probability = svc_truthy(svc_field(fields, nk[].k__hp_probability, "_hp_probability"))
    else:
        epsilon = Float64(py=svc_field(fields, nk[].k__hp_epsilon, "_hp_epsilon"))
    return SvmSettings(kernel, name == "precomputed", degree, coef0, gamma_kind, gamma,
                       svc_float(svc_field(fields, nk[].k__hp_C, "_hp_C")),
                       svc_float(svc_field(fields, nk[].k__hp_tol, "_hp_tol")), epsilon,
                       svc_int(svc_field(fields, nk[].k__hp_max_iter, "_hp_max_iter")),
                       svc_int(svc_field(fields, nk[].k__hp_nochange_steps, "_hp_nochange_steps")),
                       svc_float(svc_field(fields, nk[].k__hp_cache_size, "_hp_cache_size")), probability)

# SPDX-License-Identifier: Apache-2.0
# UNPACK part of the GaussianMixture (python/mojolearn/mixture.py) and
# BayesianGaussianMixture (python/mojolearn/_expansion_cluster.py) door: the
# Python reads. Model settings and fitted state are read here, once per
# operation and in the order the Python methods refused them, into typed
# values (`GmmFitSettings`, `BgmmEntryFit`, the scoring requests); the public
# admission and the per-model records are here too, and the two typed
# provider connections with their calls under GIL release. Every refusal that
# compares numbers is RUN's (mixture/mixture_door_run.mojo), called at the
# place the compare stood. PACK and the operators are
# bindings/mixture_public_native_pack.mojo.
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.admitted_family_door import admit_family, admit_family_refuse_empty
from bindings.admitted_input_unpack import classical_view
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_item
from bindings.native_loaded_module_door import native_loaded_module
from bindings.family_raw_attr_unpack import family_raw_has
from bindings.family_raw_attr_pack import family_raw_get, family_raw_get_or
from bindings.classical_arrays_door import (
    classical_address, classical_is_array, classical_is_f32, classical_c_order, classical_cast_kind,
)
from bindings.family_common_unpack import (
    FAMILY_F32P,
    family_repr,
    family_str,
    family_nonfinite,
    family_is_int,
    family_is_real,
    family_is_str,
    family_type_name,
)
from bindings.family_common_pack import (
    family_call,
    family_array,
)
from core.admitted_input import admitted_cast_f32
from bindings.classical_pandas_door import classical_admit_value
from bindings.host_inference_scalars_door import host_inference_scalar
from bindings.native_owner_lifetime_door import native_release_admit
from providers.python.estimator_binding_pyapi_door import estimator_binding, _estimator_mode_word
from bindings.cluster_side_records_unpack import (
    cluster_refuse_fit_callbacks,
    cluster_record_put,
)
from bindings.cluster_side_records_pack import (
    cluster_record_key,
    cluster_record_get,
)
from providers.python.neural_descriptor_pyapi_door import (
    NeuralDescriptorProvider, neural_descriptor_entry, neural_descriptor_connect_factory_once,
)
from core.bgmm_native_api import (
    BGMM_NATIVE_ABI, BGMM_NATIVE_COUNT, BgmmDriver, BgmmFitRequest, BgmmResultSlot,
)
from core.kmeans_variants_api import VariantsSpan
from core.mixture_native_api import (
    MIXTURE_NATIVE_ABI, MIXTURE_NATIVE_COUNT, GmmDriver, GmmScoreRequest,
)
from bindings.kmeans_variants_boundary_door import variants_span
from bindings.boundary_read_door import read_int_like, read_float_like, read_bool
from bindings.mode_answer_door import fast_answers_compiled_code
from core.mode_answer import MODE_FAMILY_MIXTURE
from bindings.svm_keys_unpack import svm_keys
from mixture.mixture_door_run import (
    MIX_GMM, MIX_REFUSE_N_INIT, MIX_REFUSE_WARM, MIX_REFUSE_INIT_TAIL,
    gmm_components_positive, mixture_rows_cover, bgmm_count_at_least_one, mixture_weights_size,
    mixture_means_shape, mixture_precision_values, mixture_precisions_finite, mixture_values_finite,
    mixture_mean_prior_size, mixture_square_prior_refused, mixture_diag_prior_size,
)

# Per-model records (bindings/cluster_side_records.mojo, lane perf2-cluster):
# the public admission of the parameters and fitted state passed for this key,
# and the scoring operands of the default model read for this key.
comptime _GMM_ADMIT_SLOT = "cluster_gmm_admitted_v1"
comptime _GMM_OPERANDS_SLOT = "cluster_gmm_operands_v1"


# ------------------------------------------------------------------ helpers
# ------------------------------------------------------------------ helpers
def _get(model: PythonObject, name: String) raises -> PythonObject:
    """`getattr(model, name)`, its AttributeError unchanged, through the
    generic lookup: never the class's Python `__getattribute__` hook."""
    return family_raw_get(model, name)


def _get_or(model: PythonObject, name: String, default: PythonObject) raises -> PythonObject:
    """`getattr(model, name, default)` through the generic lookup."""
    return family_raw_get_or(model, name, default)


def _int(value: PythonObject) raises -> PythonObject:
    """`int(value)`, its exception unchanged."""
    return family_call(native_loaded_module("builtins").int, value)


def _float(value: PythonObject) raises -> PythonObject:
    """`float(value)`, its exception unchanged."""
    return family_call(native_loaded_module("builtins").float, value)


def _f32_ready(source: PythonObject, context: PythonObject) raises -> Bool:
    """`source` is an exact Array or ndarray, non-empty, float32 in C order:
    `as_f32_c` would admit it and return its own storage. Perf (lane
    perf2-cluster): the BayesianGaussianMixture door and the fitted-state
    reads pass such arrays (the public door's admitted X, the fitted blocks)
    on every query; they skip a second view admission."""
    var cls = py_type(source)
    var nk = svm_keys()
    if cls is not py_item(context, nk[].k_array_class) and cls is not py_get(context, nk[].k_ndarray):
        return False
    return Int(py=source.size) > 0 and classical_is_f32(source, context) and classical_c_order(source, context)


def _x_finite(source: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_expansion_cluster._f32(source, name)`: as_f32_c (2-D, non-empty)
    and every value finite, in that refusal's wording."""
    if _f32_ready(source, context) and Int(py=source.ndim) == 2:
        # One native scan of the admitted float32 words: the same refusal
        # the cast's scan makes, for the same values.
        mixture_values_finite(Int(py=classical_address(source)), Int(py=source.size), name)
        _ = source  # keep the Python object alive through the native scan above
        return source
    var x = PythonObject(None)
    try:
        x = admit_family(source, 2, name, context, finite=True).array
    except error:
        if family_nonfinite(error, name):
            raise Error("mojolearn: " + name + " contains NaN or infinity")
        raise Error(String(error))
    return x


def _any_f32(source: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_buffer.as_f32_c(source, ndim=None, name=name)`: any rank, the input
    itself when it already is float32 in C order, else one cast copy."""
    var kept = False
    return _any_f32_kept(source, name, context, kept)


def _any_f32_kept(source: PythonObject, name: String, context: PythonObject, mut kept: Bool) raises -> PythonObject:
    """`_any_f32`; `kept` stays True only when the result is the storage of
    `source` itself (an exact Array or ndarray already float32 in C order, no
    cast copy), so a later in-place edit of `source` is seen through it."""
    if _f32_ready(source, context):
        # The input itself, as the view admission below returns it (its own
        # storage: `kept` stays as it is).
        return source
    # The one admission path: a rank 0-2 input goes through admit_family
    # (its rank and emptiness refusals and its parallel cast); a higher rank
    # (a stack of initial matrices) gets the same emptiness refusal and the
    # slab casts below (family_view deleted).
    var view = classical_view(source, context, name)
    var shape = List[Int]()
    var count = 1
    for axis in range(Int(py=view.ndim)):  # small-loop(rank: the axes of one array)
        shape.append(Int(py=view.shape[axis]))
        count *= shape[axis]
    if count == 0:
        admit_family_refuse_empty(name, shape)
    var result = _any_f32_view(view, name, context)
    if kept:
        var cls = py_type(source)
        var nk = svm_keys()
        kept = result is view and (cls is py_item(context, nk[].k_array_class) or cls is py_get(context, nk[].k_ndarray))
    _ = view^
    return result


def _any_f32_view(view: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var nd = Int(py=view.ndim)
    if nd <= 2:
        return admit_family(view, -1, name, context, admitted=True, empty=True).array
    if not classical_is_array(view, context) and not Bool(py=view.dtype.isnative):
        raise Error("TypeError: " + name + " must be in native byte order")
    if classical_is_f32(view, context) and classical_c_order(view, context):
        return view
    var b = native_loaded_module("builtins")
    var result = family_array(py_call(b.tuple, view.shape), "<f4", context)
    var rows = Int(py=view.shape[nd - 2])
    var cols = Int(py=view.shape[nd - 1])
    var s0 = Int(py=view.strides[nd - 2])
    var s1 = Int(py=view.strides[nd - 1])
    var item = Int(py=view.itemsize)
    var kind = classical_cast_kind(view, item, name, context)
    var slabs = 1
    for a in range(nd - 2):  # small-loop(rank: the leading axes of one array): bounded by its rank
        slabs *= Int(py=view.shape[a])
    var base = Int(py=classical_address(view))
    var out = Int(py=classical_address(result))
    _ = result  # keep the Python object alive through the native call above
    for s in range(slabs):  # driver-loop: over the leading-axis slabs (components) of an initial array
        var offset = 0
        var rest = s
        for a in range(nd - 3, -1, -1):  # small-loop(rank: the leading axes of one array): bounded by its rank
            var dim = Int(py=view.shape[a])
            offset += (rest % dim) * Int(py=view.strides[a])
            rest //= dim
        _ = admitted_cast_f32(base + offset, s0, s1, rows, cols, item, kind,
                         FAMILY_F32P(unsafe_from_address=out + s * rows * cols * 4))
        _ = result  # keep the Python object alive through the native call above
    _ = view  # keep the Python object alive through the native casts above
    return result


def _aux(parts: PythonObject) raises -> List[Float32]:
    """`_expansion_cluster._aux(*parts)`: the float32 parts (C order, from
    `_any_f32`) end to end, one byte copy each, now into a Mojo-owned
    buffer the typed BGMM request names (no Python Array is allocated for
    it). Empty when the parts hold no value (the former None)."""
    var total = 0
    for p in range(len(parts)):  # small-loop(parts: the few prior, warm-state and model blocks)
        total += Int(py=parts[p].size)
    var out = List[Float32](length=total, fill=0)
    var offset = 0
    for p in range(len(parts)):  # small-loop(parts: the few prior, warm-state and model blocks)
        var size = Int(py=parts[p].size)
        if size:
            unsafe_memcpy(dest=out.unsafe_ptr() + offset,
                   src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(parts[p]))), count=size)
        offset += size
    _ = parts  # the parts stay owned through the copies above
    return out^


def _seed(random_state: PythonObject) raises -> PythonObject:
    """`_expansion_cluster._seed`: None is 0; a non-negative int keeps its
    low 63 bits."""
    if random_state is None:
        return PythonObject(0)
    var operator = native_loaded_module("operator")
    if not family_is_int(random_state) or Bool(py=family_call(operator.lt, random_state, PythonObject(0))):
        raise Error("mojolearn: random_state must be None or a non-negative int "
                    + "(the lane's seeded host stream; a RandomState object is not taken)")
    return family_call(operator.and_, _int(random_state), PythonObject(0x7FFFFFFFFFFFFFFF))


@fieldwise_init
struct _BgmmConnection(Movable):
    var provider: PythonObject
    var driver: BgmmDriver


def _bgmm_connection(binding: PythonObject) raises -> _BgmmConnection:
    """The typed BGMM provider of the x_cluster `binding`
    (`bgmm_native_api_v1`), connected once per process."""
    var factory = neural_descriptor_entry(binding, "bgmm_native_api_v1")
    var provider = neural_descriptor_connect_factory_once(factory, String(BGMM_NATIVE_ABI), BGMM_NATIVE_COUNT)
    var driver = BgmmDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry)
    return _BgmmConnection(provider, driver^)


def _bgmm_fit_call(binding: PythonObject, x: PythonObject, aux: List[Float32], var ints: List[Int],
                   var floats: List[Float64]) raises -> BgmmResultSlot:
    """ENTRY_BGMM on the typed transport: the words the former
    `_XCluster._call(ENTRY_BGMM, x, aux, ints, floats)` packed into Python
    lists, as Mojo values; the GIL is released around the entry as
    `x_cluster_call` released it."""
    var connection = _bgmm_connection(binding)
    var request = BgmmFitRequest(variants_span(x), VariantsSpan(Int(aux.unsafe_ptr()), len(aux)), ints^, floats^)
    var result = BgmmResultSlot()
    try:
        with GILReleased(Python()):
            connection.driver.fit(request, result)
    finally:
        _ = connection.provider
        _ = x
        _ = aux
    return result^


def _bgmm_score_call(binding: PythonObject, x: PythonObject, aux: List[Float32], n: Int, d: Int, k: Int,
                     flags: Int) raises -> BgmmResultSlot:
    """ENTRY_BGMM_SCORE on the typed transport (ints [n, d, k, flags])."""
    var connection = _bgmm_connection(binding)
    var source = variants_span(x)
    var result = BgmmResultSlot()
    try:
        with GILReleased(Python()):
            connection.driver.score(source, VariantsSpan(Int(aux.unsafe_ptr()), len(aux)), n, d, k, flags, result)
    finally:
        _ = connection.provider
        _ = x
        _ = aux
    return result^


def _sorted_repr(table: PythonObject) raises -> String:
    return family_repr(py_call(native_loaded_module("builtins").sorted, table))


# ----------------------------------------------------- GaussianMixture, default
@fieldwise_init
struct _GmmConnection(Movable):
    var provider: PythonObject
    var driver: GmmDriver


def _gmm_connection(binding: PythonObject) raises -> _GmmConnection:
    """The typed GaussianMixture provider of the mixture `binding`
    (`mixture_native_api_v1`), connected once per process."""
    var factory = neural_descriptor_entry(binding, "mixture_native_api_v1")
    var provider = neural_descriptor_connect_factory_once(factory, String(MIXTURE_NATIVE_ABI), MIXTURE_NATIVE_COUNT)
    var driver = GmmDriver(provider.downcast_value_ptr[NeuralDescriptorProvider]()[].record[0].entry)
    return _GmmConnection(provider, driver^)

def _gmm_score_run(binding: PythonObject, operation: Int, request: GmmScoreRequest) raises:
    """One scoring entry on the typed transport, the GIL released around it
    (the Python entries released it around the body only). The caller keeps
    every array the request addresses alive past this call."""
    var connection = _gmm_connection(binding)
    try:
        with GILReleased(Python()):
            connection.driver.score(operation, request)
    finally:
        _ = connection.provider


def _check_mode(binding: PythonObject, model: PythonObject, context: PythonObject) raises:
    """`GaussianMixture._extension`: the binary's own compile-time mode must
    be the one requested."""
    var want = _get_or(model, "numeric_mode", PythonObject(None))
    var nk = svm_keys()
    if not read_bool(want):
        want = py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    var getter = neural_descriptor_entry(binding, "mixture_numeric_mode", False)
    if getter is None:
        return
    # Perf (lane perf2-cluster): the compiled mode word is a constant of the
    # loaded binary, read once per (module, getter) for the process
    # (`_estimator_mode_word`, as estimator_checked_mode reads it), not
    # called on every fit and query.
    var got = _estimator_mode_word(binding, getter)
    var wanted = py_get(py_item(context, nk[].k_family_gmm_mode_codes), want, PythonObject(None))
    if wanted is None or Int(py=wanted) != got:
        if fast_answers_compiled_code(want, py_item(context, nk[].k_family_gmm_mode_codes), got, MODE_FAMILY_MIXTURE):
            return
        raise Error("RuntimeError: " + MIX_GMM + "numeric_mode=" + family_repr(want) + " was requested "
                    + "but " + String(py_item(py_fields(binding), nk[].k___name__))
                    + " reports compile-time mode code " + String(got)
                    + "; rebuild it with bash bindings/build_mixture.sh")


def _real(value: PythonObject, name: String) raises -> Float64:
    if not family_is_real(value):
        raise Error("TypeError: " + MIX_GMM + name + " must be a real number, got " + family_type_name(value))
    return read_float_like(value)


def _gmm_needs_ext(model: PythonObject) raises -> Bool:
    """`_expansion_cluster._gmm_needs_ext`: an option the mixture binding
    does not carry (short-circuit order kept)."""
    var operator = native_loaded_module("operator")
    if Bool(py=family_call(operator.ne, _get(model, "covariance_type"), PythonObject("full"))):
        return True
    if Bool(py=family_call(operator.contains, Python.tuple(PythonObject("k-means++"), PythonObject("random_from_data")),
                           _get(model, "init_params"))):
        return True
    if Bool(py=family_call(operator.ne, _int(_get(model, "n_init")), PythonObject(1))):
        return True
    if read_bool(_get(model, "warm_start")):
        return True
    return (_get(model, "weights_init") is not None or _get(model, "means_init") is not None
            or _get(model, "precisions_init") is not None)


@fieldwise_init
struct GmmFitSettings(Movable):
    """The default fit's settings as Mojo values, read once."""
    var k: Int
    var max_iter: Int
    var tol: Float64
    var reg_covar: Float64
    var covariance_name: String
    var init_name: String


def gmm_default_refuse_options(model: PythonObject) raises:
    """`_refuse_knobs`: the options `GmmParams` does not carry, by name."""
    var operator = native_loaded_module("operator")
    if Bool(py=family_call(operator.ne, _get(model, "n_init"), PythonObject(1))):
        raise Error(MIX_GMM + MIX_REFUSE_N_INIT)
    if read_bool(_get(model, "warm_start")):
        raise Error(MIX_GMM + MIX_REFUSE_WARM)
    if _get(model, "means_init") is not None:
        raise Error(MIX_GMM + "means_init" + MIX_REFUSE_INIT_TAIL)
    if _get(model, "weights_init") is not None:
        raise Error(MIX_GMM + "weights_init" + MIX_REFUSE_INIT_TAIL)
    if _get(model, "precisions_init") is not None:
        raise Error(MIX_GMM + "precisions_init" + MIX_REFUSE_INIT_TAIL)


def gmm_fit_settings(model: PythonObject) raises -> GmmFitSettings:
    """The default fit's settings after X was admitted, in the order the
    Python method read and refused them. The two names cross as str reads
    (both checked to be str here) and are decoded by the binding's own
    covariance_type_from_name / init_params_from_name."""
    var n_components = _get(model, "n_components")
    if not family_is_int(n_components):
        raise Error("TypeError: " + MIX_GMM + "n_components must be an int")
    var covariance_type = _get(model, "covariance_type")
    var init_params = _get(model, "init_params")
    if not family_is_str(covariance_type) or not family_is_str(init_params):
        raise Error("TypeError: " + MIX_GMM + "covariance_type and init_params are "
                    + "scikit-learn's strings; they are decoded, and refused, by name on "
                    + "the Mojo host")
    var k = read_int_like(n_components)
    gmm_components_positive(k)
    var tol = _real(_get(model, "tol"), "tol")
    var reg_covar = _real(_get(model, "reg_covar"), "reg_covar")
    var max_iter = read_int_like(_get(model, "max_iter"))
    return GmmFitSettings(k, max_iter, tol, reg_covar, String(py=covariance_type), String(py=init_params))


def gmm_fit_random_state(model: PythonObject) raises -> Int:
    """Read after the mode check, where the Python method read it."""
    return read_int_like(_get(model, "random_state"))


def _gmm_score_call(model: PythonObject, binding: PythonObject, x: PythonObject, operation: Int, what: String,
                    out_address: Int, context: PythonObject, key: PythonObject) raises:
    """`_model_lists` then the scoring `operation` (MIXTURE_NATIVE_*; lane
    bw2-svm: the callers pass the word, the former entry name and its
    four-compare map `_gmm_operation` are gone) writing the caller's output
    at `out_address` (an array or a Mojo buffer the caller keeps alive past
    this call): the fitted check, the feature count, the mode check and the
    call.

    Perf (lane perf2-cluster): with a record `key` (a public query whose
    admission key was built), the fitted operands read here (nine generic
    attribute reads, three array admissions, the mode check) are kept per
    model while the key matches and the binding is the same module. They are
    recorded only when every operand is the fitted array's own storage (no
    cast copy) read from the instance dictionary, so an in-place edit of a
    fitted array is still seen by the next query."""
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var held = cluster_record_get(_GMM_OPERANDS_SLOT, model, key)
    if held is not None and held[0] is binding:
        var width = held[6]
        if Bool(py=family_call(native_loaded_module("operator").ne, PythonObject(d), width)):
            raise Error(MIX_GMM + "X has " + String(d) + " features, the fit had " + family_str(width))
        # ORDER MATCHES bindings/_mojolearn_mixture.mojo::_rebuild_model.
        _gmm_score_run(binding, operation, GmmScoreRequest(
            Int(py=classical_address(held[1])), Int(py=classical_address(held[2])),
            Int(py=classical_address(held[3])), Int(py=classical_address(held[4])),
            Int(py=classical_address(held[5])), Int(py=classical_address(x)), out_address,
            Int(py=held[7]), d, Int(py=held[8]), Int(py=held[9]) != 0, n, Float64(py=held[10])))
        _ = held^  # the recorded operands stay owned through the native call above
        _ = x
        return
    if not family_raw_has(model, "weights_"):
        raise Error(MIX_GMM + "call fit before " + what)
    var fitted = _get(model, "n_features_in_")
    if Bool(py=family_call(native_loaded_module("operator").ne, PythonObject(d), fitted)):
        raise Error(MIX_GMM + "X has " + String(d) + " features, the fit had " + family_str(fitted))
    var weights = _get(model, "weights_")
    var k = Int(py=weights.shape[0])
    var kept = key is not None
    var means = _get(model, "means_")
    var flat_m = _any_f32_kept(means, "means_", context, kept)
    var covariances = _get(model, "covariances_")
    var flat_c = _any_f32_kept(covariances, "covariances_", context, kept)
    var cholesky = _get(model, "precisions_cholesky_")
    var flat_p = _any_f32_kept(cholesky, "precisions_cholesky_", context, kept)
    var log_det = _get(model, "log_det_chol_")
    var converged_ = _get(model, "converged_")
    var converged = 1 if read_bool(converged_) else 0
    var n_iter = _get(model, "n_iter_")
    var lower = _get(model, "lower_bound_")
    # ORDER MATCHES bindings/_mojolearn_mixture.mojo::_rebuild_model.
    # weights, means, covariances, precisions_chol, log_det_chol, x, out;
    # k, d, n_iter, converged, lower_bound, n (each read once, natively,
    # after the mode check, as the Python entry read them).
    _check_mode(binding, model, context)
    var request = GmmScoreRequest(
        Int(py=classical_address(weights)), Int(py=classical_address(flat_m)), Int(py=classical_address(flat_c)),
        Int(py=classical_address(flat_p)), Int(py=classical_address(log_det)), Int(py=classical_address(x)),
        out_address, k, d, Int(py=n_iter), converged != 0, n, Float64(py=lower))
    _gmm_score_run(binding, operation, request)
    _ = x
    _ = log_det  # keep the Python object alive through the native call above
    _ = flat_p  # keep the Python object alive through the native call above
    _ = flat_c  # keep the Python object alive through the native call above
    _ = flat_m  # keep the Python object alive through the native call above
    _ = weights  # keep the Python object alive through the native call above
    if kept:
        # Every operand must be the value the instance dictionary stores (no
        # class attribute or descriptor in between) and a native array, so
        # the record's identities cover exactly what was read.
        var nk = svm_keys()
        var fields = py_fields(model)
        var missing = Python.list()
        var array_class = py_item(context, nk[].k_array_class)
        var ndarray = py_get(context, nk[].k_ndarray)
        var weights_class = py_type(weights)
        var log_det_class = py_type(log_det)
        kept = (weights_class is array_class or weights_class is ndarray) and (
            log_det_class is array_class or log_det_class is ndarray)
        var names = Python.tuple("n_features_in_", "weights_", "means_", "covariances_", "precisions_cholesky_",
                                 "log_det_chol_", "converged_", "n_iter_", "lower_bound_")
        var read = Python.tuple(fitted, weights, means, covariances, cholesky, log_det, converged_, n_iter, lower)
        for i in range(len(names)):  # small-loop(nine fitted fields read above)
            kept = kept and py_get(fields, names[i], missing) is read[i]
        if kept:
            cluster_record_put(_GMM_OPERANDS_SLOT, model, key,
                               Python.tuple(binding, weights, flat_m, flat_c, flat_p, log_det, fitted,
                                            PythonObject(k), n_iter, PythonObject(converged), lower))
    _ = weights^
    _ = flat_m^
    _ = flat_c^
    _ = flat_p^
    _ = log_det^


# ------------------------------------------- GaussianMixture, option parity
@fieldwise_init
struct BgmmEntryFit(Movable):
    """One ENTRY_BGMM fit as typed values: the plain GaussianMixture solver
    (`plain`, the three init flags) or BayesianGaussianMixture (`process`,
    the priors). `x` is the admitted float32 X, pinned through the fit;
    `key` is (k, d, covariance type), the warm-state key PACK stores; `aux`
    holds the prior, init and warm-state words end to end."""
    var x: PythonObject
    var key: PythonObject
    var aux: List[Float32]
    var n: Int
    var d: Int
    var k: Int
    var ct: Int
    var process: Bool
    var plain: Bool
    var mean_prior: Bool
    var cov_mode: Int
    var warm: Bool
    var flag_w: Int
    var flag_m: Int
    var flag_p: Int
    var prior_weight: Float64
    var prior_mean_precision: Float64
    var prior_freedom: Float64
    var reg_covar: Float64
    var tol: Float64
    var lower: Float64
    var max_iter: Int
    var n_init: Int
    var init_code: Int
    var seed: Int


def gmm_ext_fit_stage(model: PythonObject, source: PythonObject, context: PythonObject) raises -> BgmmEntryFit:
    """`_expansion_cluster._gmm_ext_fit`, its reads: scikit-learn's
    GaussianMixture on the cluster lane's mixture driver (ENTRY_BGMM, plain
    mode). Same reads and refusals, in the same order."""
    var operator = native_loaded_module("operator")
    var nk = svm_keys()
    var ctypes = py_item(context, nk[].k_family_mix_ctypes)
    var inits = py_item(context, nk[].k_family_mix_inits)
    var covariance_type = _get(model, "covariance_type")
    if not py_contains(ctypes, covariance_type):
        raise Error("covariance_type must be one of " + _sorted_repr(ctypes) + ", got " + family_repr(covariance_type))
    var init_params = _get(model, "init_params")
    if not py_contains(inits, init_params):
        raise Error("init_params must be one of " + _sorted_repr(inits) + ", got " + family_repr(init_params))
    var x = _x_finite(source, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var k = read_int_like(_get(model, "n_components"))
    mixture_rows_cover(k, n)
    var ct = Int(py=ctypes[covariance_type])
    var key = Python.tuple(PythonObject(k), PythonObject(d), covariance_type)
    # the warm state and the init arrays as byte copies (`_aux`); the
    # binding expands tied, diag and spherical precisions_init to d x d per
    # component (`x_cluster/entries.mojo::bgmm_entry`, ip[15] = 2, 3, 4)
    var parts = Python.list()
    var prev = _get_or(model, "_ext", PythonObject(None))
    var warm = read_bool(_get(model, "warm_start")) and prev is not None
    if warm:
        warm = Bool(py=family_call(operator.eq, py_item(prev, nk[].k_key), key))
    if warm:
        _ = py_call(parts.append, py_item(prev, nk[].k_state))
    var flag_w = 0
    var flag_m = 0
    var flag_p = 0
    var weights_init = _get(model, "weights_init")
    if weights_init is not None and not warm:
        var w = _any_f32(weights_init, "weights_init", context)
        mixture_weights_size(Int(py=w.size), k)
        _ = py_call(parts.append, w)
        flag_w = 1
    var means_init = _get(model, "means_init")
    if means_init is not None and not warm:
        var m = _x_finite(means_init, "means_init", context)
        mixture_means_shape(Int(py=m.shape[0]), Int(py=m.shape[1]), k, d)
        _ = py_call(parts.append, m)
        flag_m = 1
    var precisions_init = _get(model, "precisions_init")
    if precisions_init is not None and not warm:
        var pi = _any_f32(precisions_init, "precisions_init", context)
        var want = mixture_precision_values(ct, k, d)
        var size = Int(py=pi.size)
        if size != want:
            raise Error("The parameter '" + family_str(covariance_type) + " precision' has " + String(size)
                        + " values, expected " + String(want))
        mixture_precisions_finite(Int(py=classical_address(pi)), size)
        _ = pi  # keep the Python object alive through the native scan above
        _ = py_call(parts.append, pi)
        flag_p = ct + 1
    var aux = _aux(parts)
    var lower = 0.0
    if warm:
        lower = read_float_like(_get_or(model, "lower_bound_", PythonObject(0.0)))
    # The entry's words, read in the order the former Python lists read
    # them: floats, then ints (RUN lays them out).
    var reg_covar = read_float_like(_get(model, "reg_covar"))
    var tol = read_float_like(_get(model, "tol"))
    var max_iter = read_int_like(_get(model, "max_iter"))
    var n_init = read_int_like(_get(model, "n_init"))
    var init_code = read_int_like(inits[init_params])
    var seed = Int(py=_seed(_get(model, "random_state")))
    _ = parts^  # the parts stay owned through the copies of `_aux`
    return BgmmEntryFit(x=x, key=key, aux=aux^, n=n, d=d, k=k, ct=ct, process=False, plain=True,
                        mean_prior=False, cov_mode=0, warm=warm, flag_w=flag_w, flag_m=flag_m, flag_p=flag_p,
                        prior_weight=-1.0, prior_mean_precision=-1.0, prior_freedom=-1.0,
                        reg_covar=reg_covar, tol=tol, lower=lower, max_iter=max_iter, n_init=n_init,
                        init_code=init_code, seed=seed)


def _gmm_ext_score(model: PythonObject, x: PythonObject, context: PythonObject) raises -> BgmmResultSlot:
    """`_expansion_cluster._gmm_ext_score` on a checked float32 X: (f, i, s)
    of ENTRY_BGMM_SCORE with flags 1 | 2 (predict_proba's exp and score's
    fold in the binding), on the typed transport."""
    var ext = _get(model, "_ext")
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var fitted = _get(model, "n_features_in_")
    if Bool(py=family_call(native_loaded_module("operator").ne, PythonObject(d), fitted)):
        raise Error(MIX_GMM + "X has " + String(d) + " features, the fit had " + family_str(fitted))
    var means = _get(model, "means_")
    var k = Int(py=means.shape[0])
    var nk = svm_keys()
    var aux = _aux(Python.list(_any_f32(means, "aux", context), _any_f32(py_item(ext, nk[].k_pchol), "aux", context),
                               _any_f32(py_item(ext, nk[].k_consts), "aux", context)))
    # The image of the estimator's binding route (its device and mode).
    return _bgmm_score_call(estimator_binding(model, "_mojolearn_x_cluster", context), x, aux, n, d, k, 3)


# ------------------------------------------------- BayesianGaussianMixture
def bgmm_fit_stage(model: PythonObject, source: PythonObject, context: PythonObject) raises -> BgmmEntryFit:
    """`BayesianGaussianMixture._fit`, its reads (ENTRY_BGMM). Same reads and
    refusals, in the same order."""
    var operator = native_loaded_module("operator")
    var nk = svm_keys()
    var ctypes = py_item(context, nk[].k_family_mix_ctypes)
    var inits = py_item(context, nk[].k_family_mix_inits)
    var covariance_type = _get(model, "covariance_type")
    if not py_contains(ctypes, covariance_type):
        raise Error("covariance_type must be one of " + _sorted_repr(ctypes) + ", got " + family_repr(covariance_type))
    var ct = Int(py=ctypes[covariance_type])
    var init_params = _get(model, "init_params")
    if not py_contains(inits, init_params):
        raise Error("init_params must be one of " + _sorted_repr(inits) + ", got " + family_repr(init_params))
    var prior_type = _get(model, "weight_concentration_prior_type")
    if not py_contains(Python.tuple(PythonObject("dirichlet_process"), PythonObject("dirichlet_distribution")),
                       prior_type):
        raise Error("weight_concentration_prior_type must be 'dirichlet_process' or "
                    + "'dirichlet_distribution'")
    var process = Bool(py=family_call(operator.eq, prior_type, PythonObject("dirichlet_process")))
    var x = _x_finite(source, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var k = read_int_like(_get(model, "n_components"))
    mixture_rows_cover(k, n)
    bgmm_count_at_least_one(read_int_like(_get(model, "n_init")))
    bgmm_count_at_least_one(read_int_like(_get(model, "max_iter")))
    # the priors and any warm state as byte copies (`_aux`); the binding
    # expands a diag or spherical covariance prior to d x d
    # (`x_cluster/entries.mojo::bgmm_entry`, ip[9] = 2 or 3)
    var parts = Python.list()
    var cov_mode = 0
    var mean_prior = _get(model, "mean_prior")
    if mean_prior is not None:
        var mp = _any_f32(mean_prior, "mean_prior", context)
        mixture_mean_prior_size(Int(py=mp.size), d)
        _ = py_call(parts.append, mp)
    var covariance_prior = _get(model, "covariance_prior")
    if covariance_prior is not None:
        var cp: PythonObject
        if ct == 0 or ct == 1:
            cp = _x_finite(covariance_prior, "covariance_prior", context)
            if mixture_square_prior_refused(Int(py=cp.shape[0]), Int(py=cp.shape[1]), d):
                raise Error("The parameter '" + family_str(covariance_type) + " covariance prior' should have the "
                            + "shape of (" + String(d) + ", " + String(d) + ")")
            cov_mode = 1
        elif ct == 2:
            cp = _any_f32(covariance_prior, "covariance_prior", context)
            mixture_diag_prior_size(Int(py=cp.size), d)
            cov_mode = 2
        else:
            var value = read_float_like(covariance_prior)
            cp = family_array(Python.tuple(1), "<f4", context)
            var word = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(cp)))
            word[0] = Float32(value)
            cov_mode = 3
        _ = py_call(parts.append, cp)
    var key = Python.tuple(PythonObject(k), PythonObject(d), covariance_type)
    var prev = _get_or(model, "_warm", PythonObject(None))
    var warm = read_bool(_get(model, "warm_start")) and prev is not None
    if warm:
        warm = Bool(py=family_call(operator.eq, prev[0], key))
    if warm:
        _ = py_call(parts.append, _any_f32(_get(model, "_warm")[1], "aux", context))
    var aux = _aux(parts)
    # The entry's words, read in the order the former Python lists read
    # them: floats, then ints (RUN lays them out).
    var wcp = _get(model, "weight_concentration_prior")
    var prior_weight = -1.0
    if wcp is not None:
        prior_weight = read_float_like(wcp)
    var mpp = _get(model, "mean_precision_prior")
    var prior_mean_precision = -1.0
    if mpp is not None:
        prior_mean_precision = read_float_like(mpp)
    var dfp = _get(model, "degrees_of_freedom_prior")
    var prior_freedom = -1.0
    if dfp is not None:
        prior_freedom = read_float_like(dfp)
    var reg_covar = read_float_like(_get(model, "reg_covar"))
    var tol = read_float_like(_get(model, "tol"))
    var lower = 0.0
    if warm:
        lower = read_float_like(_get_or(model, "lower_bound_", PythonObject(0.0)))
    var max_iter = read_int_like(_get(model, "max_iter"))
    var n_init = read_int_like(_get(model, "n_init"))
    var init_code = read_int_like(inits[init_params])
    var seed = Int(py=_seed(_get(model, "random_state")))
    _ = parts^  # the parts stay owned through the copies of `_aux`
    return BgmmEntryFit(x=x, key=key, aux=aux^, n=n, d=d, k=k, ct=ct, process=process, plain=False,
                        mean_prior=mean_prior is not None, cov_mode=cov_mode, warm=warm, flag_w=0, flag_m=0,
                        flag_p=0, prior_weight=prior_weight, prior_mean_precision=prior_mean_precision,
                        prior_freedom=prior_freedom, reg_covar=reg_covar, tol=tol, lower=lower,
                        max_iter=max_iter, n_init=n_init, init_code=init_code, seed=seed)


def _bgmm_score(model: PythonObject, binding: PythonObject, source: PythonObject, flags: Int,
                context: PythonObject, admitted: Bool, mut n: Int, mut k: Int) raises -> BgmmResultSlot:
    """`BayesianGaussianMixture._score`: (f, i, s) of ENTRY_BGMM_SCORE on the
    typed transport; `n` and `k` receive the row and component counts.
    `admitted`: `source` is the X the public cluster door admitted
    (classical_check_x: 2-D, non-empty, finite, float32 in C order), used
    as is."""
    var means = _get_or(model, "means_", PythonObject(None))
    if means is None:
        raise Error("RuntimeError: this " + String(py_type(model).__name__) + " is not fitted yet")
    # Perf (lane pass2-cluster-nbr): the door already refused a non-finite X;
    # the second native scan of the same words is skipped for its X only.
    var x = source if admitted else _x_finite(source, "X", context)
    n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var fitted = _get(model, "n_features_in_")
    if d != Int(py=fitted):
        raise Error("mojolearn: X has " + String(d) + " features, " + String(py_type(model).__name__)
                    + " was fitted with " + family_str(fitted))
    k = Int(py=means.shape[0])
    # the model block as byte copies (`_aux`)
    var aux = _aux(Python.list(_any_f32(means, "aux", context), _any_f32(_get(model, "_full_pchol"), "aux", context),
                               _any_f32(_get(model, "_log_consts"), "aux", context)))
    var result = _bgmm_score_call(binding, x, aux, n, d, k, flags)
    _ = x^
    return result^


def _gmm_public_ext(ext: PythonObject, context: PythonObject, releasing: Bool) raises:
    if ext is None:
        return
    var b = native_loaded_module("builtins")
    if py_type(ext) is not b.dict:
        raise Error("TypeError: GaussianMixture extension state requires its native dictionary")
    var keys = py_call(b.list, ext)
    for i in range(len(keys)):  # driver-loop: exact extension-state keys before hashing or indexing
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: GaussianMixture state keys require exact strings")
    var nk = svm_keys()
    for i in range(len(keys)):  # driver-loop: native scalar and buffer state, never an arbitrary input protocol
        classical_admit_value(ext[keys[i]], context)
        if releasing:
            native_release_admit(ext[keys[i]], context)
    var key = py_item(ext, nk[].k_key)
    if py_type(key) is not b.tuple or len(key) != 3:
        raise Error("TypeError: GaussianMixture extension key requires its native scalar tuple")
    for i in range(len(key)):  # driver-loop: component count, feature count and covariance spelling
        host_inference_scalar(key[i], context)


def _gmm_fitted_names() raises -> PythonObject:
    """The stored fields the public admission reads besides the parameters."""
    return Python.tuple("n_features_in_", "n_iter_", "converged_", "lower_bound_", "_mojolearn_fitted",
                        "_mojolearn_fitting", "weights_", "means_", "covariances_", "precisions_cholesky_",
                        "log_det_chol_", "_ext", "feature_names_in_")


def _gmm_public_admit(model: PythonObject, context: PythonObject, fitting: Bool,
                      query: Bool = False) raises -> PythonObject:
    """The public admission; returns (fields, record key or None, recorded).

    Perf (lane perf2-cluster): a query (`query`) builds the identity key of
    the parameters and fitted fields first; when this model's admission
    already passed for an identical key, the per-value admissions (every
    parameter, scalar, array and the extension state) are not repeated. The
    flag checks and the callback refusal run on every call."""
    var b = native_loaded_module("builtins")
    var fields = py_fields(model)
    var nk = svm_keys()
    var parameters = py_item(context, nk[].k_gmm_public_parameters)
    var key = PythonObject(None)
    var recorded = False
    if query and not fitting:
        key = cluster_record_key(model, fields, parameters, _gmm_fitted_names())
        recorded = cluster_record_get(_GMM_ADMIT_SLOT, model, key) is not None
    if not recorded:
        for i in range(len(parameters)):  # driver-loop: constructor settings read through the admitted native owner
            var name = String(parameters[i])
            var value = _get_or(model, name, PythonObject(None)) if name == "numeric_mode" else _get(model, name)
            classical_admit_value(value, context)
            if name != "weights_init" and name != "means_init" and name != "precisions_init":
                host_inference_scalar(value, context, True)
        var scalars = Python.tuple("n_features_in_", "n_iter_", "converged_", "lower_bound_", "_mojolearn_fitted", "_mojolearn_fitting")
        var arrays = Python.tuple("weights_", "means_", "covariances_", "precisions_cholesky_", "log_det_chol_")
        for i in range(len(scalars)):  # driver-loop: consumed scalar model state
            if py_contains(fields, scalars[i]):
                host_inference_scalar(fields[scalars[i]], context)
                if fitting:
                    native_release_admit(fields[scalars[i]], context)
        for i in range(len(arrays)):  # driver-loop: retained fitted buffer owners
            if py_contains(fields, arrays[i]):
                classical_admit_value(fields[arrays[i]], context)
                if fitting:
                    native_release_admit(fields[arrays[i]], context)
    var fitted = py_get(fields, nk[].k__mojolearn_fitted, PythonObject(False))
    if py_type(fitted) is not b.bool and py_type(fitted) is not b.int:
        raise Error("TypeError: native fitted flags cannot invoke Python boolean hooks")
    var depth = py_get(fields, nk[].k__mojolearn_fitting, PythonObject(0))
    if py_type(depth) is not b.int:
        raise Error("TypeError: native fitting depth requires an exact integer")
    if fitting and py_contains(fields, nk[].k_feature_names_in_):
        native_release_admit(py_item(fields, nk[].k_feature_names_in_), context)
    # The ancestry walk once per class state (bindings/cluster_side_records.mojo).
    cluster_refuse_fit_callbacks(py_type(model), "TypeError: GaussianMixture cannot invoke Python fit callbacks")
    if not recorded and py_contains(fields, nk[].k__ext):
        _gmm_public_ext(py_item(fields, nk[].k__ext), context, fitting)
    if py_contains(fields, nk[].k__mojolearn_after_fit):
        raise Error("TypeError: GaussianMixture cannot invoke Python fit callbacks")
    return Python.tuple(fields, key, PythonObject(recorded))

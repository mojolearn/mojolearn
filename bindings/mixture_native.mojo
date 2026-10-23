# SPDX-License-Identifier: Apache-2.0
# The compiled owner of GaussianMixture (python/mojolearn/mixture.py) and
# BayesianGaussianMixture (python/mojolearn/_expansion_cluster.py) fit,
# fit_predict, predict, predict_proba, score_samples, score, bic and aic: one
# native call per public method, reached from bindings/family_native.mojo for
# every `family_mix_*` operation.
#
# The shells hand over the bindings `_bind` resolved for the model's numeric
# mode (GaussianMixture.fit also the `_mojolearn_x_cluster` binding its
# option-parity solver runs on). This owner makes the Python methods'
# refusals in their order and wording, prepares the same inputs, calls the
# same compiled entries with the same argument lists (`gmm_fit`,
# `gmm_score_samples`, `gmm_predict_proba`, `gmm_predict`,
# `gmm_score_bic_aic`, `x_cluster_call` ENTRY_BGMM / ENTRY_BGMM_SCORE) and
# writes the fitted attributes with the names, shapes and dtypes the Python
# set. EM iterations and n_init restarts already run inside those entries
# (one call per fit), so no driver loop is added here. Same entries, same
# inputs, same order: the bits do not move.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setattr
from bindings.classical_arrays import (
    classical_address, classical_wrap_array, classical_is_array, classical_is_f32, classical_c_order,
    classical_cast_kind,
)
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import (
    FAMILY_F32P, FAMILY_F64P, family_repr, family_str, family_call, family_entry, family_array, family_f32,
    family_as_f32, family_view, family_cast_view, family_nonfinite, family_is_int, family_is_real, family_is_str,
    family_type_name,
)
from core.family_cast_impl import family_cast_rows, family_all_finite_f32
from checks.numerics import pinned_mul_f64, portable_log64

# x_cluster/entries.mojo: the entry numbers of `x_cluster_call`
comptime _E_BGMM = 8
comptime _E_BGMM_SCORE = 9

comptime _GMM = "mojolearn GaussianMixture: "
# The by-name refusals of the options `GmmParams` does not carry
# (DEVIATION 1734), in the former `_REFUSED_KNOBS` wording.
comptime _REFUSE_N_INIT = ("n_init > 1 picks the best restart by a float comparison of "
                           + "lower bounds (_base.py:282), a second data-dependent branch "
                           + "with no identity argument yet (DEVIATION 1734)")
comptime _REFUSE_WARM = ("warm_start makes the fit depend on the object's history "
                         + "rather than on its inputs (DEVIATION 1734)")
comptime _REFUSE_INIT_TAIL = " is not a field of GmmParams (mixture/NOT_IMPLEMENTED.tsv)"


# ------------------------------------------------------------------ helpers
def _get(model: PythonObject, name: String) raises -> PythonObject:
    """`getattr(model, name)`, its AttributeError unchanged."""
    return family_call(Python.import_module("builtins").getattr, model, PythonObject(name))


def _get_or(model: PythonObject, name: String, default: PythonObject) raises -> PythonObject:
    """`getattr(model, name, default)`."""
    return family_call(Python.import_module("builtins").getattr, model, PythonObject(name), default)


def _truthy(value: PythonObject) raises -> Bool:
    """`bool(value)`."""
    return Bool(py=family_call(Python.import_module("builtins").bool, value))


def _int(value: PythonObject) raises -> PythonObject:
    """`int(value)`, its exception unchanged."""
    return family_call(Python.import_module("builtins").int, value)


def _float(value: PythonObject) raises -> PythonObject:
    """`float(value)`, its exception unchanged."""
    return family_call(Python.import_module("builtins").float, value)


def _from_flat(flat: PythonObject, shape: PythonObject, typestr: String, code: String,
               context: PythonObject) raises -> PythonObject:
    """`Array._from_flat(flat, shape, typestr)`: a copy of the entry's
    `array.array` result as an owned C-order Array."""
    var store = py_call(context["array_type"], PythonObject(code), flat)
    return classical_wrap_array(store, shape, typestr, "C", context)


def _x_finite(source: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_expansion_cluster._f32(source, name)`: as_f32_c (2-D, non-empty)
    and every value finite, in that refusal's wording."""
    var x = PythonObject(None)
    try:
        x = family_f32(source, 2, name, context)
    except error:
        if family_nonfinite(error, name):
            raise Error("mojolearn: " + name + " contains NaN or infinity")
        raise Error(String(error))
    return x


def _any_f32(source: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_buffer.as_f32_c(source, ndim=None, name=name)`: any rank, the input
    itself when it already is float32 in C order, else one cast copy."""
    var view = family_view(source, -1, name, context)
    var nd = Int(py=view.ndim)
    if nd <= 2:
        var cast = family_cast_view(view, name, context)
        _ = view^
        return cast
    if not classical_is_array(view, context) and not Bool(py=view.dtype.isnative):
        raise Error("TypeError: " + name + " must be in native byte order")
    if classical_is_f32(view, context) and classical_c_order(view, context):
        return view
    var b = Python.import_module("builtins")
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
    for s in range(slabs):  # driver-loop: over the leading-axis slabs (components) of an initial array
        var offset = 0
        var rest = s
        for a in range(nd - 3, -1, -1):  # small-loop(rank: the leading axes of one array): bounded by its rank
            var dim = Int(py=view.shape[a])
            offset += (rest % dim) * Int(py=view.strides[a])
            rest //= dim
        family_cast_rows(base + offset, s0, s1, rows, cols, item, kind,
                         FAMILY_F32P(unsafe_from_address=out + s * rows * cols * 4))
    _ = view^
    return result


def _concat(parts: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_expansion_cluster._aux(*parts)`: the float32 parts (C order, from
    `_any_f32`) end to end as one float32 Array, byte copies; None when the
    parts hold no value."""
    var total = 0
    for p in range(len(parts)):  # small-loop(parts: the few prior, warm-state and model blocks)
        total += Int(py=parts[p].size)
    if total == 0:
        return PythonObject(None)
    var out = family_array(Python.tuple(total), "<f4", context)
    var dst = Int(py=classical_address(out))
    var offset = 0
    for p in range(len(parts)):  # small-loop(parts: the few prior, warm-state and model blocks)
        var size = Int(py=parts[p].size)
        if size:
            memcpy(dest=FAMILY_F32P(unsafe_from_address=dst + 4 * offset),
                   src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(parts[p]))), count=size)
        offset += size
    return out


def _state(blocks: PythonObject, zeros_at: Int, zeros: Int, context: PythonObject) raises -> PythonObject:
    """The warm-start state: the entry's float blocks end to end (with
    `zeros` zeros in place of block `zeros_at` when it is empty, or always
    when `zeros_at` < 0 names an inserted block) as one float32 Array, the
    bytes the former `list(...) + ...` concatenation cast to float32."""
    var store = py_call(context["array_type"], PythonObject("f"))
    var b = Python.import_module("builtins")
    for j in range(len(blocks)):  # small-loop(blocks: the eight fitted parameter blocks)
        var block = blocks[j]
        if block is None or (j == zeros_at and len(block) == 0):
            _ = py_call(store.frombytes, py_call(b.bytes, PythonObject(4 * zeros)))
        else:
            _ = py_call(store.extend, block)
    return classical_wrap_array(store, Python.tuple(len(store)), "<f4", "C", context)


def _seed(random_state: PythonObject) raises -> PythonObject:
    """`_expansion_cluster._seed`: None is 0; a non-negative int keeps its
    low 63 bits."""
    if random_state is None:
        return PythonObject(0)
    var operator = Python.import_module("operator")
    if not family_is_int(random_state) or Bool(py=family_call(operator.lt, random_state, PythonObject(0))):
        raise Error("mojolearn: random_state must be None or a non-negative int "
                    + "(the lane's seeded host stream; a RandomState object is not taken)")
    return family_call(operator.and_, _int(random_state), PythonObject(0x7FFFFFFFFFFFFFFF))


def _xc_call(binding: PythonObject, which: Int, x: PythonObject, aux: PythonObject, ints: PythonObject,
             floats: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_XCluster._call(which, x, aux, ints, floats)`: (f, i, s)."""
    var entry = family_entry(binding, "x_cluster_call", context)
    var aux_address = PythonObject(0)
    var aux_size = 0
    if aux is not None:
        aux_size = Int(py=aux.size)
        if aux_size:
            aux_address = classical_address(aux)
    var result = family_call(entry, PythonObject(which), classical_address(x), PythonObject(Int(py=x.size)),
                             aux_address, PythonObject(aux_size), ints, floats)
    _ = aux^
    return result


def _by_type(flat: PythonObject, k: Int, d: Int, ct: Int, context: PythonObject) raises -> PythonObject:
    """`BayesianGaussianMixture._by_type`: the entry's by-type values (full
    k x d x d, tied d x d, diag k x d, spherical k) under the covariance
    type's shape."""
    if ct == 0:
        return _from_flat(flat, Python.tuple(k, d, d), "<f4", "f", context)
    if ct == 1:
        return _from_flat(flat, Python.tuple(d, d), "<f4", "f", context)
    if ct == 2:
        return _from_flat(flat, Python.tuple(k, d), "<f4", "f", context)
    return _from_flat(flat, Python.tuple(k), "<f4", "f", context)


def _sorted_repr(table: PythonObject) raises -> String:
    return family_repr(py_call(Python.import_module("builtins").sorted, table))


# ----------------------------------------------------- GaussianMixture, default
def _check_mode(binding: PythonObject, model: PythonObject, context: PythonObject) raises:
    """`GaussianMixture._extension`: the binary's own compile-time mode must
    be the one requested."""
    var b = Python.import_module("builtins")
    var want = _get_or(model, "numeric_mode", PythonObject(None))
    if not _truthy(want):
        want = context["backend_fields"]["_DEFAULT_MODE"]
    var getter = py_call(b.getattr, binding, PythonObject("mixture_numeric_mode"), PythonObject(None))
    if getter is None:
        return
    var got = Int(py=family_call(b.int, family_call(getter)))
    var wanted = py_get(context["family_gmm_mode_codes"], want, PythonObject(None))
    if wanted is None or Int(py=wanted) != got:
        raise Error("RuntimeError: " + _GMM + "numeric_mode=" + family_repr(want) + " was requested "
                    + "but " + String(py_call(b.getattr, binding, PythonObject("__name__")))
                    + " reports compile-time mode code " + String(got)
                    + "; rebuild it with bash bindings/build_mixture.sh")


def _real(value: PythonObject, name: String) raises -> PythonObject:
    if not family_is_real(value):
        raise Error("TypeError: " + _GMM + name + " must be a real number, got " + family_type_name(value))
    return _float(value)


def _gmm_needs_ext(model: PythonObject) raises -> Bool:
    """`_expansion_cluster._gmm_needs_ext`: an option the mixture binding
    does not carry (short-circuit order kept)."""
    var operator = Python.import_module("operator")
    if Bool(py=family_call(operator.ne, _get(model, "covariance_type"), PythonObject("full"))):
        return True
    if Bool(py=family_call(operator.contains, Python.tuple(PythonObject("k-means++"), PythonObject("random_from_data")),
                           _get(model, "init_params"))):
        return True
    if Bool(py=family_call(operator.ne, _int(_get(model, "n_init")), PythonObject(1))):
        return True
    if _truthy(_get(model, "warm_start")):
        return True
    return (_get(model, "weights_init") is not None or _get(model, "means_init") is not None
            or _get(model, "precisions_init") is not None)


def _gmm_default_fit(model: PythonObject, binding: PythonObject, source: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """The default fit (`gmm_fit`); returns the float32 X it read."""
    var b = Python.import_module("builtins")
    var operator = Python.import_module("operator")
    py_setattr(model, "_ext", PythonObject(None))
    # `_refuse_knobs`
    if Bool(py=family_call(operator.ne, _get(model, "n_init"), PythonObject(1))):
        raise Error(_GMM + _REFUSE_N_INIT)
    if _truthy(_get(model, "warm_start")):
        raise Error(_GMM + _REFUSE_WARM)
    if _get(model, "means_init") is not None:
        raise Error(_GMM + "means_init" + _REFUSE_INIT_TAIL)
    if _get(model, "weights_init") is not None:
        raise Error(_GMM + "weights_init" + _REFUSE_INIT_TAIL)
    if _get(model, "precisions_init") is not None:
        raise Error(_GMM + "precisions_init" + _REFUSE_INIT_TAIL)
    var x = family_as_f32(source, 2, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var n_components = _get(model, "n_components")
    if not family_is_int(n_components):
        raise Error("TypeError: " + _GMM + "n_components must be an int")
    var covariance_type = _get(model, "covariance_type")
    var init_params = _get(model, "init_params")
    if not family_is_str(covariance_type) or not family_is_str(init_params):
        raise Error("TypeError: " + _GMM + "covariance_type and init_params are "
                    + "scikit-learn's strings; they are decoded, and refused, by name on "
                    + "the Mojo host")
    var k = Int(py=_int(n_components))
    if k < 1:
        raise Error(_GMM + "n_components must be positive, got " + String(k))
    var tol = _real(_get(model, "tol"), "tol")
    var reg_covar = _real(_get(model, "reg_covar"), "reg_covar")
    var max_iter = _int(_get(model, "max_iter"))
    var weights = family_array(Python.tuple(k), "<f4", context)
    var means = family_array(Python.tuple(k, d), "<f4", context)
    var covariances = family_array(Python.tuple(k, d, d), "<f4", context)
    var precisions = family_array(Python.tuple(k, d, d), "<f4", context)
    var log_det = family_array(Python.tuple(k), "<f4", context)
    var scalars = family_array(Python.tuple(3), "<f8", context)
    _check_mode(binding, model, context)
    var entry = family_entry(binding, "gmm_fit", context)
    # ORDER MATCHES bindings/_mojolearn_mixture.mojo::gmm_fit_binding.
    # x, weights_out, means_out, covariances_out, precisions_chol_out, log_det_chol_out, scalars_out;
    # n, d, n_components, covariance_type, tol, reg_covar, max_iter, init_params, random_state
    var addrs = Python.list(classical_address(x), classical_address(weights), classical_address(means),
                            classical_address(covariances), classical_address(precisions),
                            classical_address(log_det), classical_address(scalars))
    var params = Python.list(PythonObject(n), PythonObject(d), PythonObject(k), covariance_type, tol, reg_covar,
                             max_iter, init_params, _int(_get(model, "random_state")))
    var n_iter = family_call(entry, addrs, params)
    var words = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(scalars)))
    py_setattr(model, "n_features_in_", PythonObject(d))
    py_setattr(model, "weights_", weights)
    py_setattr(model, "means_", means)
    py_setattr(model, "covariances_", covariances)
    py_setattr(model, "precisions_cholesky_", precisions)
    py_setattr(model, "log_det_chol_", log_det)
    py_setattr(model, "n_iter_", _int(n_iter))
    py_setattr(model, "converged_", PythonObject(Int(words[1]) != 0))
    py_setattr(model, "lower_bound_", PythonObject(words[2]))
    _ = scalars^
    return x


def _gmm_score_call(model: PythonObject, binding: PythonObject, x: PythonObject, name: String, what: String,
                    out_buf: PythonObject, context: PythonObject) raises:
    """`_model_lists` then the scoring entry `name` writing `out`: the
    fitted check, the feature count, the mode check and the call."""
    var b = Python.import_module("builtins")
    if not Bool(py=family_call(b.hasattr, model, PythonObject("weights_"))):
        raise Error(_GMM + "call fit before " + what)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var fitted = _get(model, "n_features_in_")
    if Bool(py=family_call(Python.import_module("operator").ne, PythonObject(d), fitted)):
        raise Error(_GMM + "X has " + String(d) + " features, the fit had " + family_str(fitted))
    var weights = _get(model, "weights_")
    var k = Int(py=weights.shape[0])
    var flat_m = _any_f32(_get(model, "means_"), "means_", context)
    var flat_c = _any_f32(_get(model, "covariances_"), "covariances_", context)
    var flat_p = _any_f32(_get(model, "precisions_cholesky_"), "precisions_cholesky_", context)
    var log_det = _get(model, "log_det_chol_")
    var converged = 1 if _truthy(_get(model, "converged_")) else 0
    # ORDER MATCHES bindings/_mojolearn_mixture.mojo::_rebuild_model.
    # weights, means, covariances, precisions_chol, log_det_chol, x, out;
    # k, d, n_iter, converged, lower_bound, n
    var addrs = Python.list(classical_address(weights), classical_address(flat_m), classical_address(flat_c),
                            classical_address(flat_p), classical_address(log_det), classical_address(x),
                            classical_address(out_buf))
    var params = Python.list(PythonObject(k), PythonObject(d), _get(model, "n_iter_"), PythonObject(converged),
                             _get(model, "lower_bound_"), PythonObject(n))
    _check_mode(binding, model, context)
    _ = family_call(family_entry(binding, name, context), addrs, params)
    _ = weights^
    _ = flat_m^
    _ = flat_c^
    _ = flat_p^
    _ = log_det^


# ------------------------------------------- GaussianMixture, option parity
def _gmm_ext_fit(model: PythonObject, xc_binding: PythonObject, source: PythonObject,
                 context: PythonObject) raises -> PythonObject:
    """`_expansion_cluster._gmm_ext_fit`: scikit-learn's GaussianMixture on
    the cluster lane's mixture driver (ENTRY_BGMM, plain mode); returns the
    float32 X it read."""
    var b = Python.import_module("builtins")
    var operator = Python.import_module("operator")
    var ctypes = context["family_mix_ctypes"]
    var inits = context["family_mix_inits"]
    var covariance_type = _get(model, "covariance_type")
    if not py_contains(ctypes, covariance_type):
        raise Error("covariance_type must be one of " + _sorted_repr(ctypes) + ", got " + family_repr(covariance_type))
    var init_params = _get(model, "init_params")
    if not py_contains(inits, init_params):
        raise Error("init_params must be one of " + _sorted_repr(inits) + ", got " + family_repr(init_params))
    var x = _x_finite(source, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var k = Int(py=_int(_get(model, "n_components")))
    if k < 1 or n < k:
        raise Error("Expected n_samples >= n_components but got n_components = " + String(k)
                    + ", n_samples = " + String(n))
    var ct = Int(py=ctypes[covariance_type])
    var key = Python.tuple(PythonObject(k), PythonObject(d), covariance_type)
    # the warm state and the init arrays as byte copies (`_aux`); the
    # binding expands tied, diag and spherical precisions_init to d x d per
    # component (`x_cluster/entries.mojo::bgmm_entry`, ip[15] = 2, 3, 4)
    var parts = Python.list()
    var prev = _get_or(model, "_ext", PythonObject(None))
    var warm = _truthy(_get(model, "warm_start")) and prev is not None
    if warm:
        warm = Bool(py=family_call(operator.eq, prev["key"], key))
    if warm:
        _ = py_call(parts.append, prev["state"])
    var flag_w = 0
    var flag_m = 0
    var flag_p = 0
    var weights_init = _get(model, "weights_init")
    if weights_init is not None and not warm:
        var w = _any_f32(weights_init, "weights_init", context)
        if Int(py=w.size) != k:
            raise Error("The parameter 'weights' should have the shape of (" + String(k) + ",)")
        _ = py_call(parts.append, w)
        flag_w = 1
    var means_init = _get(model, "means_init")
    if means_init is not None and not warm:
        var m = _x_finite(means_init, "means_init", context)
        if Int(py=m.shape[0]) != k or Int(py=m.shape[1]) != d:
            raise Error("The parameter 'means' should have the shape of (" + String(k) + ", " + String(d) + ")")
        _ = py_call(parts.append, m)
        flag_m = 1
    var precisions_init = _get(model, "precisions_init")
    if precisions_init is not None and not warm:
        var pi = _any_f32(precisions_init, "precisions_init", context)
        var want: Int
        if ct == 0:
            want = k * d * d
        elif ct == 1:
            want = d * d
        elif ct == 2:
            want = k * d
        else:
            want = k
        var size = Int(py=pi.size)
        if size != want:
            raise Error("The parameter '" + family_str(covariance_type) + " precision' has " + String(size)
                        + " values, expected " + String(want))
        if not family_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(pi))), size):
            raise Error("precisions_init must be finite")
        _ = py_call(parts.append, pi)
        flag_p = ct + 1
    var aux = _concat(parts, context)
    var lower = PythonObject(0.0)
    if warm:
        lower = _float(_get_or(model, "lower_bound_", PythonObject(0.0)))
    var floats = Python.list(PythonObject(-1.0), PythonObject(-1.0), PythonObject(-1.0),
                             _float(_get(model, "reg_covar")), _float(_get(model, "tol")), lower)
    var ints = Python.list(PythonObject(n), PythonObject(d), PythonObject(k), PythonObject(0),
                           _int(_get(model, "max_iter")), _int(_get(model, "n_init")), _int(inits[init_params]),
                           _seed(_get(model, "random_state")), PythonObject(0), PythonObject(0), PythonObject(ct),
                           PythonObject(1 if warm else 0), PythonObject(1), PythonObject(flag_w), PythonObject(flag_m),
                           PythonObject(flag_p))
    var result = _xc_call(xc_binding, _E_BGMM, x, aux, ints, floats, context)
    _ = parts^
    var f = result[0]
    var s = result[2]
    py_setattr(model, "n_features_in_", PythonObject(d))
    py_setattr(model, "weights_", _from_flat(f[0], Python.tuple(k), "<f4", "f", context))
    py_setattr(model, "means_", _from_flat(f[1], Python.tuple(k, d), "<f4", "f", context))
    py_setattr(model, "covariances_", _by_type(f[13], k, d, ct, context))
    py_setattr(model, "precisions_cholesky_", _by_type(f[14], k, d, ct, context))
    py_setattr(model, "log_det_chol_", _from_flat(f[12], Python.tuple(k), "<f4", "f", context))   # x_cluster bgmm_entry
    py_setattr(model, "n_iter_", _int(s[1]))
    py_setattr(model, "converged_", PythonObject(_truthy(s[2])))
    py_setattr(model, "lower_bound_", _float(s[0]))
    # The fit's own state: the x_cluster binding it ran on (its scoring
    # entry), the full per-component Cholesky block and log constants, and
    # the state a warm_start refit resumes from.
    var ext = Python.dict()
    ext["key"] = key
    ext["binding"] = xc_binding
    ext["pchol"] = _from_flat(f[3], Python.tuple(k, d, d), "<f4", "f", context)
    ext["consts"] = _from_flat(f[8], Python.tuple(k), "<f4", "f", context)
    ext["state"] = _state(Python.list(f[11], f[4], PythonObject(None), f[6], f[1], f[7], f[2], f[3]), -1, k, context)
    py_setattr(model, "_ext", ext)
    return x


def _gmm_ext_score(model: PythonObject, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_expansion_cluster._gmm_ext_score` on a checked float32 X: (f, i, s)
    of ENTRY_BGMM_SCORE with flags 1 | 2 (predict_proba's exp and score's
    fold in the binding)."""
    var ext = _get(model, "_ext")
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var fitted = _get(model, "n_features_in_")
    if Bool(py=family_call(Python.import_module("operator").ne, PythonObject(d), fitted)):
        raise Error(_GMM + "X has " + String(d) + " features, the fit had " + family_str(fitted))
    var means = _get(model, "means_")
    var k = Int(py=means.shape[0])
    var aux = _concat(Python.list(_any_f32(means, "aux", context), _any_f32(ext["pchol"], "aux", context),
                                  _any_f32(ext["consts"], "aux", context)), context)
    return _xc_call(ext["binding"], _E_BGMM_SCORE, x, aux,
                    Python.list(PythonObject(n), PythonObject(d), PythonObject(k), PythonObject(3)), Python.list(),
                    context)


def _gmm_bic_aic(model: PythonObject, binding: PythonObject, source: PythonObject,
                 context: PythonObject) raises -> List[Float64]:
    """`GaussianMixture._score_bic_aic`: (score, bic, aic)."""
    var b = Python.import_module("builtins")
    var result = List[Float64]()
    if _get_or(model, "_ext", PythonObject(None)) is not None:
        # `_gmm_ext_bic_aic`: scikit-learn's `_n_parameters`, `bic` and
        # `aic`; the mean log-likelihood is the binding's float-float fold
        # over the rows divided by n.
        var x = _x_finite(source, "X", context)
        var n = Int(py=x.shape[0])
        var got = _gmm_ext_score(model, x, context)
        var score = Float64(py=got[2][0]) / Float64(n)
        var means = _get(model, "means_")
        var k = Int(py=means.shape[0])
        var d = Int(py=means.shape[1])
        var ct = Int(py=context["family_mix_ctypes"][_get(model, "covariance_type")])
        # The covariance parameter counts are whole numbers (d * (d + 1) is
        # even), so the integer sums equal the former float ones exactly.
        var cov: Int
        if ct == 0:
            cov = k * d * (d + 1) // 2
        elif ct == 2:
            cov = k * d
        elif ct == 1:
            cov = d * (d + 1) // 2
        else:
            cov = k
        var p = cov + k * d + k - 1
        var fit_term = pinned_mul_f64(pinned_mul_f64(-2.0, score), Float64(n))
        result.append(score)
        result.append(fit_term + pinned_mul_f64(Float64(p), portable_log64(Float64(n))))
        result.append(fit_term + Float64(2 * p))
        _ = x^
        return result^
    var x = family_as_f32(source, 2, "X", context)
    var out = family_array(Python.tuple(3), "<f8", context)
    _gmm_score_call(model, binding, x, "gmm_score_bic_aic", "score", out, context)
    var words = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(out)))
    result.append(words[0])
    result.append(words[1])
    result.append(words[2])
    _ = out^
    _ = x^
    return result^


def _gmm_infer(model: PythonObject, binding: PythonObject, source: PythonObject, which: Int,
               context: PythonObject) raises -> PythonObject:
    """score_samples (0), predict_proba (1) or predict (2)."""
    var ext = _get_or(model, "_ext", PythonObject(None))
    if ext is not None:
        var xe = _x_finite(source, "X", context)
        return _gmm_ext_labels(model, xe, which, context)
    var x = family_as_f32(source, 2, "X", context)
    return _gmm_default_infer(model, binding, x, which, context)


def _gmm_ext_labels(model: PythonObject, x: PythonObject, which: Int, context: PythonObject) raises -> PythonObject:
    var n = Int(py=x.shape[0])
    var got = _gmm_ext_score(model, x, context)
    var k = Int(py=_get(model, "means_").shape[0])
    if which == 0:
        return _from_flat(got[0][1], Python.tuple(n), "<f4", "f", context)
    if which == 1:
        return _from_flat(got[0][2], Python.tuple(n, k), "<f4", "f", context)
    return _from_flat(got[1][0], Python.tuple(n), "<i4", "i", context)


def _gmm_default_infer(model: PythonObject, binding: PythonObject, x: PythonObject, which: Int,
                       context: PythonObject) raises -> PythonObject:
    var b = Python.import_module("builtins")
    var n = Int(py=x.shape[0])
    if which == 0:
        var out = family_array(Python.tuple(n), "<f4", context)
        _gmm_score_call(model, binding, x, "gmm_score_samples", "score_samples", out, context)
        return out
    if which == 1:
        var k = 0
        if Bool(py=family_call(b.hasattr, model, PythonObject("weights_"))):
            k = Int(py=_get(model, "weights_").shape[0])
        var out = family_array(Python.tuple(n, k), "<f4", context)
        _gmm_score_call(model, binding, x, "gmm_predict_proba", "predict_proba", out, context)
        return out
    var out = family_array(Python.tuple(n), "<i4", context)
    _gmm_score_call(model, binding, x, "gmm_predict", "predict", out, context)
    return out


def _gmm_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`GaussianMixture.fit`: args (binding, x_cluster binding, X); returns
    the float32 X it read."""
    if _gmm_needs_ext(model):
        return _gmm_ext_fit(model, args[1], args[2], context)
    return _gmm_default_fit(model, args[0], args[2], context)


# ------------------------------------------------- BayesianGaussianMixture
def _bgmm_fit(model: PythonObject, binding: PythonObject, source: PythonObject,
              context: PythonObject) raises -> PythonObject:
    """`BayesianGaussianMixture._fit`: ENTRY_BGMM; returns the labels."""
    var b = Python.import_module("builtins")
    var operator = Python.import_module("operator")
    var ctypes = context["family_mix_ctypes"]
    var inits = context["family_mix_inits"]
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
    var k = Int(py=_int(_get(model, "n_components")))
    if k < 1 or n < k:
        raise Error("Expected n_samples >= n_components but got n_components = " + String(k)
                    + ", n_samples = " + String(n))
    if Int(py=_int(_get(model, "n_init"))) < 1 or Int(py=_int(_get(model, "max_iter"))) < 1:
        raise Error("n_init and max_iter must be >= 1")
    # the priors and any warm state as byte copies (`_aux`); the binding
    # expands a diag or spherical covariance prior to d x d
    # (`x_cluster/entries.mojo::bgmm_entry`, ip[9] = 2 or 3)
    var parts = Python.list()
    var cov_mode = 0
    var mean_prior = _get(model, "mean_prior")
    if mean_prior is not None:
        var mp = _any_f32(mean_prior, "mean_prior", context)
        if Int(py=mp.size) != d:
            raise Error("The parameter 'means' should have the shape of (" + String(d) + ",)")
        _ = py_call(parts.append, mp)
    var covariance_prior = _get(model, "covariance_prior")
    if covariance_prior is not None:
        var cp: PythonObject
        if ct == 0 or ct == 1:
            cp = _x_finite(covariance_prior, "covariance_prior", context)
            if Int(py=cp.shape[0]) != d or Int(py=cp.shape[1]) != d:
                raise Error("The parameter '" + family_str(covariance_type) + " covariance prior' should have the "
                            + "shape of (" + String(d) + ", " + String(d) + ")")
            cov_mode = 1
        elif ct == 2:
            cp = _any_f32(covariance_prior, "covariance_prior", context)
            if Int(py=cp.size) != d:
                raise Error("The parameter 'diag covariance prior' should have the shape of (" + String(d) + ",)")
            cov_mode = 2
        else:
            var value = Float64(py=_float(covariance_prior))
            cp = family_array(Python.tuple(1), "<f4", context)
            var word = FAMILY_F32P(unsafe_from_address=Int(py=classical_address(cp)))
            word[0] = Float32(value)
            cov_mode = 3
        _ = py_call(parts.append, cp)
    var key = Python.tuple(PythonObject(k), PythonObject(d), covariance_type)
    var prev = _get_or(model, "_warm", PythonObject(None))
    var warm = _truthy(_get(model, "warm_start")) and prev is not None
    if warm:
        warm = Bool(py=family_call(operator.eq, prev[0], key))
    if warm:
        _ = py_call(parts.append, _any_f32(_get(model, "_warm")[1], "aux", context))
    var aux = _concat(parts, context)
    var floats = Python.list()
    var wcp = _get(model, "weight_concentration_prior")
    _ = py_call(floats.append, PythonObject(-1.0) if wcp is None else _float(wcp))
    var mpp = _get(model, "mean_precision_prior")
    _ = py_call(floats.append, PythonObject(-1.0) if mpp is None else _float(mpp))
    var dfp = _get(model, "degrees_of_freedom_prior")
    _ = py_call(floats.append, PythonObject(-1.0) if dfp is None else _float(dfp))
    _ = py_call(floats.append, _float(_get(model, "reg_covar")))
    _ = py_call(floats.append, _float(_get(model, "tol")))
    _ = py_call(floats.append, _float(_get_or(model, "lower_bound_", PythonObject(0.0))) if warm else PythonObject(0.0))
    var ints = Python.list(PythonObject(n), PythonObject(d), PythonObject(k), PythonObject(1 if process else 0),
                           _int(_get(model, "max_iter")), _int(_get(model, "n_init")), _int(inits[init_params]),
                           _seed(_get(model, "random_state")), PythonObject(1 if mean_prior is not None else 0),
                           PythonObject(cov_mode), PythonObject(ct), PythonObject(1 if warm else 0))
    var result = _xc_call(binding, _E_BGMM, x, aux, ints, floats, context)
    _ = parts^
    var f = result[0]
    var i = result[1]
    var s = result[2]
    py_setattr(model, "weights_", _from_flat(f[0], Python.tuple(k), "<f4", "f", context))
    py_setattr(model, "means_", _from_flat(f[1], Python.tuple(k, d), "<f4", "f", context))
    py_setattr(model, "_full_pchol", _from_flat(f[3], Python.tuple(k, d, d), "<f4", "f", context))
    # f[13], f[14], f[15]: the binding's by-type forms (bgmm_entry)
    py_setattr(model, "covariances_", _by_type(f[13], k, d, ct, context))
    py_setattr(model, "precisions_cholesky_", _by_type(f[14], k, d, ct, context))
    if process:
        py_setattr(model, "weight_concentration_",
                   Python.tuple(_from_flat(f[4], Python.tuple(k), "<f4", "f", context),
                                _from_flat(f[5], Python.tuple(k), "<f4", "f", context)))
    else:
        py_setattr(model, "weight_concentration_", _from_flat(f[4], Python.tuple(k), "<f4", "f", context))
    py_setattr(model, "mean_precision_", _from_flat(f[6], Python.tuple(k), "<f4", "f", context))
    if ct == 1:
        py_setattr(model, "degrees_of_freedom_", _float(f[7][0]))
    else:
        py_setattr(model, "degrees_of_freedom_", _from_flat(f[7], Python.tuple(k), "<f4", "f", context))
    py_setattr(model, "_log_consts", _from_flat(f[8], Python.tuple(k), "<f4", "f", context))
    py_setattr(model, "mean_prior_", _from_flat(f[9], Python.tuple(d), "<f4", "f", context))
    var cpf = f[15]
    if ct == 2:
        py_setattr(model, "covariance_prior_", _from_flat(cpf, Python.tuple(d), "<f4", "f", context))
    elif ct == 3:
        py_setattr(model, "covariance_prior_", _float(cpf[0]))
    else:
        py_setattr(model, "covariance_prior_", _from_flat(cpf, Python.tuple(d, d), "<f4", "f", context))
    py_setattr(model, "lower_bound_", _float(s[0]))
    py_setattr(model, "n_iter_", _int(s[1]))
    py_setattr(model, "converged_", PythonObject(_truthy(s[2])))
    # the state a warm_start refit resumes from (full per-component
    # matrices); an empty f[5] (Dirichlet distribution) is k zeros
    py_setattr(model, "_warm", Python.tuple(key, _state(Python.list(f[11], f[4], f[5], f[6], f[1], f[7], f[2], f[3]),
                                                        2, k, context)))
    py_setattr(model, "weight_concentration_prior_", _float(s[3]))
    py_setattr(model, "mean_precision_prior_", _float(s[4]))
    py_setattr(model, "degrees_of_freedom_prior_", _float(s[5]))
    py_setattr(model, "n_features_in_", PythonObject(d))
    return _from_flat(i[0], Python.tuple(n), "<i4", "i", context)


def _bgmm_score(model: PythonObject, binding: PythonObject, source: PythonObject, flags: Int,
                context: PythonObject) raises -> PythonObject:
    """`BayesianGaussianMixture._score`: (f, i, s) of ENTRY_BGMM_SCORE plus
    the row count, as a tuple (result, n, k)."""
    var means = _get_or(model, "means_", PythonObject(None))
    if means is None:
        raise Error("RuntimeError: this " + String(py_type(model).__name__) + " is not fitted yet")
    var x = _x_finite(source, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var fitted = _get(model, "n_features_in_")
    if d != Int(py=fitted):
        raise Error("mojolearn: X has " + String(d) + " features, " + String(py_type(model).__name__)
                    + " was fitted with " + family_str(fitted))
    var k = Int(py=means.shape[0])
    var ints = Python.list(PythonObject(n), PythonObject(d), PythonObject(k))
    if flags:
        _ = py_call(ints.append, PythonObject(flags))
    # the model block as byte copies (`_aux`)
    var aux = _concat(Python.list(_any_f32(means, "aux", context), _any_f32(_get(model, "_full_pchol"), "aux", context),
                                  _any_f32(_get(model, "_log_consts"), "aux", context)), context)
    var result = _xc_call(binding, _E_BGMM_SCORE, x, aux, ints, Python.list(), context)
    _ = x^
    return Python.tuple(result, PythonObject(n), PythonObject(k))


# ------------------------------------------------------------------- door
def mixture_operation(model: PythonObject, operation: String, args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    """args[0] is the resolved binding; GaussianMixture fit and fit_predict
    also take the `_mojolearn_x_cluster` binding as args[1]; X is last."""
    classical_connect_numpy(context)
    if operation == "family_mix_gmm_fit":
        _ = _gmm_fit(model, args, context)
        return model
    if operation == "family_mix_gmm_fit_predict":
        # `fit(X).predict(X)` on the float32 X the fit read.
        var x = _gmm_fit(model, args, context)
        if _get_or(model, "_ext", PythonObject(None)) is not None:
            return _gmm_ext_labels(model, x, 2, context)
        return _gmm_default_infer(model, args[0], x, 2, context)
    if operation == "family_mix_gmm_score_samples":
        return _gmm_infer(model, args[0], args[1], 0, context)
    if operation == "family_mix_gmm_predict_proba":
        return _gmm_infer(model, args[0], args[1], 1, context)
    if operation == "family_mix_gmm_predict":
        return _gmm_infer(model, args[0], args[1], 2, context)
    if operation == "family_mix_gmm_score":
        return PythonObject(_gmm_bic_aic(model, args[0], args[1], context)[0])
    if operation == "family_mix_gmm_bic":
        return PythonObject(_gmm_bic_aic(model, args[0], args[1], context)[1])
    if operation == "family_mix_gmm_aic":
        return PythonObject(_gmm_bic_aic(model, args[0], args[1], context)[2])
    if operation == "family_mix_bgmm_fit":
        _ = _bgmm_fit(model, args[0], args[1], context)
        return model
    if operation == "family_mix_bgmm_fit_predict":
        return _bgmm_fit(model, args[0], args[1], context)
    if operation == "family_mix_bgmm_predict":
        # the labels come from the device score (`bodies.argmax_row`: the
        # first largest log responsibility, the lowest index on a tie)
        var got = _bgmm_score(model, args[0], args[1], 0, context)
        return _from_flat(got[0][1][0], Python.tuple(got[1]), "<i4", "i", context)
    if operation == "family_mix_bgmm_predict_proba":
        # the exp of every log responsibility in the binding (bodies.exp_cell)
        var got = _bgmm_score(model, args[0], args[1], 1, context)
        return _from_flat(got[0][0][2], Python.tuple(got[1], got[2]), "<f4", "f", context)
    if operation == "family_mix_bgmm_score_samples":
        var got = _bgmm_score(model, args[0], args[1], 0, context)
        return _from_flat(got[0][0][1], Python.tuple(got[1]), "<f4", "f", context)
    if operation == "family_mix_bgmm_score":
        # the float-float fold of log_prob_norm in the binding
        var got = _bgmm_score(model, args[0], args[1], 2, context)
        return PythonObject(Float64(py=got[0][2][0]) / Float64(Int(py=got[1])))
    raise Error("RuntimeError: unknown family operation " + operation)

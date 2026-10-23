# SPDX-License-Identifier: Apache-2.0
# The compiled owner of the fits of MiniBatchKMeans (fit, partial_fit),
# BisectingKMeans, MeanShift, OPTICS and AffinityPropagation
# (python/mojolearn/_expansion_cluster.py, with the centers' fit_predict and
# fit_transform) and of HDBSCAN (python/mojolearn/hdbscan.py): one native call
# per public method, reached from bindings/family_native.mojo for every
# `family_cfa_*` operation.
#
# Each fit makes the Python method's refusals (same wording, same order),
# builds the same integer and float words, calls the same binding entry once
# (`x_cluster_call`, or HDBSCAN's `hdbscan_fit` and
# `hdbscan_generate_prediction_data`) with the same inputs, and stores the
# attributes the Python set, with the same names, dtypes and shapes. Every
# iteration, batch, bisection and seed loop already runs inside the entry,
# which uploads X once per fit. Same entries, same operands, same order: the
# bits do not move.
from std.math import isinf
from std.memory import memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_fields, py_setattr, py_setitem
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.family_common import (
    FAMILY_F32P, FAMILY_I32P, family_repr, family_str, family_is_str, family_is_int, family_isinstance,
    family_type_name, family_call, family_call_kw, family_entry, family_array, family_view, family_cast_view,
    family_f32, family_as_f32, family_nonfinite, family_sample_weight, family_shape_repr,
)
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, PENDING_PYTHON_ERROR
from bindings.cluster_native import cluster_operation, xc_call_floats, _xc_array

# x_cluster/entries.mojo: the entry numbers of `x_cluster_call`
comptime _E_MINIBATCH = 2
comptime _E_BISECT = 3
comptime _E_MEANSHIFT = 5
comptime _E_OPTICS = 6
comptime _E_AFFINITY = 7
comptime _E_MINIBATCH_PARTIAL = 10
# OPTICS metric codes (x_cluster/optics.mojo)
comptime _OPTICS_PRECOMPUTED = 5


# ---------------------------------------------------------------- helpers

def _attr(model: PythonObject, name: String) raises -> PythonObject:
    """`getattr(model, name)`; its AttributeError is restored unchanged."""
    return family_call(Python.import_module("builtins").getattr, model, PythonObject(name))


def _attr_or_none(model: PythonObject, name: String) raises -> PythonObject:
    """`getattr(model, name, None)`."""
    return family_call(Python.import_module("builtins").getattr, model, PythonObject(name), PythonObject(None))


def _truthy(value: PythonObject) raises -> Bool:
    """`bool(value)`."""
    return Bool(py=family_call(Python.import_module("builtins").bool, value))


def _int(value: PythonObject) raises -> PythonObject:
    """`int(value)`."""
    return family_call(Python.import_module("builtins").int, value)


def _float(value: PythonObject) raises -> PythonObject:
    """`float(value)`."""
    return family_call(Python.import_module("builtins").float, value)


def _eq(value: PythonObject, other: PythonObject) raises -> Bool:
    """`bool(value == other)`."""
    return _truthy(family_call(Python.import_module("operator").eq, value, other))


def _one_of(value: PythonObject, a: String, b: String) raises -> Bool:
    """`value in (a, b)`."""
    return py_contains(Python.tuple(PythonObject(a), PythonObject(b)), value)


def _set(model: PythonObject, var name: String, value: PythonObject) raises:
    """`setattr(model, name, value)`."""
    py_setattr(model, name^, value)


def _pending(index: Int) raises -> PythonObject:
    """A new reference to word `index` (0 type, 1 value) of the Python
    exception `family_call` stashed, or None. The stash is kept."""
    var pending = PENDING_PYTHON_ERROR.get_or_create_ptr()
    var address = pending[].words[index]
    if address == 0:
        return PythonObject(None)
    ref cpy = Python().cpython()
    var pointer = cpy.lib.get_function[PyObjectPtr]("Py_NewRef")(address)
    return PythonObject(from_owned=pointer)


def _drop_pending() raises:
    """Release the stashed exception: the caller raises its own instead."""
    ref cpy = Python().cpython()
    var release = cpy.lib.get_function[NoneType]("Py_DecRef")
    var pending = PENDING_PYTHON_ERROR.get_or_create_ptr()
    for t in range(3):  # small-loop(the stashed exception's three references)
        var address = pending[].words[t]
        if address:
            release(address)
            pending[].words[t] = 0


def _pending_is(error: Error, classes: PythonObject) raises -> Bool:
    """True when `error` stands for a stashed exception of one of `classes`
    (`except classes:`)."""
    if String(error) != PYTHON_ERROR_PENDING:
        return False
    var kind = _pending(0)
    if kind is None:
        return False
    return Bool(py=py_call(Python.import_module("builtins").issubclass, kind, classes))


def _pending_text(error: Error) raises -> String:
    """`str(e)` of the stashed exception (`error` itself otherwise)."""
    if String(error) != PYTHON_ERROR_PENDING:
        return String(error)
    var value = _pending(1)
    if value is None:
        return String("")
    return String(py=py_call(Python.import_module("builtins").str, value))


def _f32(source: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_expansion_cluster._f32(source, name)`: `as_f32_c(ndim=2)` and the
    by-name NaN/inf refusal."""
    try:
        return family_f32(source, 2, name, context)
    except error:
        if family_nonfinite(error, name):
            raise Error("mojolearn: " + name + " contains NaN or infinity")
        raise Error(String(error))


def _seed(random_state: PythonObject) raises -> Int:
    """`_expansion_cluster._seed`: None is 0; a non-negative int keeps its
    low 63 bits."""
    if random_state is None:
        return 0
    var b = Python.import_module("builtins")
    if (family_isinstance(random_state, b.bool) or not family_isinstance(random_state, b.int)
            or Bool(py=py_call(Python.import_module("operator").lt, random_state, PythonObject(0)))):
        raise Error("mojolearn: random_state must be None or a non-negative int "
                    + "(the lane's seeded host stream; a RandomState object is not taken)")
    return Int(py=py_call(Python.import_module("operator").and_, random_state,
                          PythonObject(9223372036854775807)))


def _weights(sample_weight: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_expansion_cluster._weights`: None stays None; otherwise
    `check_sample_weight(sample_weight, n, allow_negative=True)`."""
    if sample_weight is None:
        return PythonObject(None)
    return family_sample_weight(sample_weight, n, True, context)


def _aux(parts: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_expansion_cluster._aux(*parts)`: the parts (None skipped) end to
    end as one float32 Array, one byte copy per part; None when empty."""
    var arrays = Python.list()
    var total = 0
    for t in range(len(parts)):  # small-loop(the aux block's parts): at most three
        var part = parts[t]
        if part is not None:
            var a = family_as_f32(part, -1, "aux", context)
            total += Int(py=a.size)
            _ = py_call(arrays.append, a)
    if total == 0:
        return PythonObject(None)
    var out = family_array(Python.tuple(total), "<f4", context)
    var base = Int(py=classical_address(out))
    _ = out  # keep the Python object alive through the native call above
    var offset = 0
    for t in range(len(arrays)):  # small-loop(the aux block's parts): one copy each
        var a = arrays[t]
        var size = Int(py=a.size)
        if size:
            memcpy(dest=FAMILY_F32P(unsafe_from_address=base + 4 * offset),
                   src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(a))), count=size)
            _ = a  # keep the Python object alive through the native call above
        offset += size
    _ = arrays^
    return out


def _callable_init(init: PythonObject, source: PythonObject, k: Int, random_state: PythonObject) raises -> PythonObject:
    """scikit-learn's callable `init` (`init(X, n_clusters,
    random_state=random_state)`), called once with every row of the input:
    a NumPy array and a RandomState seeded by `random_state` when NumPy is
    installed, the input and the int otherwise."""
    var b = Python.import_module("builtins")
    var data = source
    var state = random_state
    var numpy = PythonObject(None)
    try:
        numpy = Python.import_module("numpy")
    except:
        numpy = PythonObject(None)
    if numpy is not None:
        try:
            data = family_call(numpy.asarray, source)
            state = family_call(numpy.random.RandomState, random_state)
        except error:
            if not _pending_is(error, Python.tuple(b.ImportError, b.AttributeError)):
                raise Error(String(error))
            _drop_pending()
    var keywords = Python.dict()
    py_setitem(keywords, PythonObject("random_state"), state)
    return family_call_kw(init, keywords, data, PythonObject(k))


def _init_array(model: PythonObject, source: PythonObject, init: PythonObject, k: Int, d: Int,
                context: PythonObject) raises -> PythonObject:
    """MiniBatchKMeans' array or callable `init` as a checked (k, d) float32
    array."""
    var b = Python.import_module("builtins")
    var given = init
    if Bool(py=py_call(b.callable, init)):
        given = _callable_init(init, source, k, _attr(model, "random_state"))
    var arr = _f32(given, "init", context)
    if Int(py=arr.shape[0]) != k or Int(py=arr.shape[1]) != d:
        raise Error("The shape of the initial centers " + family_shape_repr(arr) + " does not match "
                    + "the number of clusters " + String(k) + " and features " + String(d) + ".")
    return arr


def _flag(value: Bool) raises -> PythonObject:
    return PythonObject(1 if value else 0)


def _rows(model: PythonObject, k: Int, n: Int) raises:
    if k < 1 or k > n:
        raise Error("n_samples=" + String(n) + " should be >= n_clusters=" + String(k) + ".")


# ---------------------------------------------------------------- MiniBatchKMeans

def _mbk_devscan(binding: PythonObject) raises -> Bool:
    """`MiniBatchKMeans._devscan`: whether the binding scans X for NaN/inf
    itself (MBK_FAST_DEVSCAN), so the host scan is skipped; False on any
    failure."""
    var b = Python.import_module("builtins")
    try:
        var getter = py_call(b.getattr, binding, PythonObject("x_cluster_mbk_devscan"))
        return Int(py=py_call(b.int, py_call(getter))) == 1
    except:
        return False


def _mbk_fit(model: PythonObject, binding: PythonObject, source: PythonObject, sample_weight: PythonObject,
             context: PythonObject) raises:
    """`MiniBatchKMeans.fit(X, sample_weight)`."""
    var devscan = _mbk_devscan(binding)
    var x: PythonObject
    if devscan:
        x = family_as_f32(source, 2, "X", context)
    else:
        x = _f32(source, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var k = Int(py=_int(_attr(model, "n_clusters")))
    _rows(model, k, n)
    var weights = _weights(sample_weight, n, context)
    var init = _attr(model, "init")
    var init_arr = PythonObject(None)
    var init_random = False
    if family_is_str(init):
        if not _one_of(init, "k-means++", "random"):
            raise Error("init must be 'k-means++', 'random' or an array, got " + family_repr(init))
        init_random = _eq(init, PythonObject("random"))
    else:
        init_arr = _init_array(model, source, init, k, d, context)
    var n_init_field = _attr(model, "n_init")
    var n_init: Int
    if _eq(n_init_field, PythonObject("auto")):
        n_init = 3 if (init_arr is None and init_random) else 1
    else:
        n_init = Int(py=_int(n_init_field))
    if n_init < 1:
        raise Error("n_init must be >= 1")
    var mni_field = _attr(model, "max_no_improvement")
    var mni = PythonObject(-1) if mni_field is None else _int(mni_field)
    var max_iter = _int(_attr(model, "max_iter"))
    var batch_size = _int(_attr(model, "batch_size"))
    var init_size_field = _attr(model, "init_size")
    var init_size = PythonObject(0) if init_size_field is None else _int(init_size_field)
    var seed = _seed(_attr(model, "random_state"))
    # ORDER MATCHES x_cluster/entries.mojo::minibatch_entry: n, d, k,
    # max_iter, batch_size, max_no_improvement (-1 None), init_size (0
    # default), n_init, has_init, seed, init_random, has_weights.
    var ints = Python.list(PythonObject(n), PythonObject(d), PythonObject(k), max_iter, batch_size, mni, init_size,
                           PythonObject(n_init), _flag(init_arr is not None), PythonObject(seed),
                           _flag(init_arr is None and init_random), _flag(weights is not None))
    var aux = _aux(Python.list(init_arr, weights), context)
    if devscan:
        _ = py_call(ints.append, PythonObject(1))  # ints[12]: the binding owes the NaN/inf scan
    var floats = Python.list(_float(_attr(model, "tol")), _float(_attr(model, "reassignment_ratio")))
    var result = PythonObject(None)
    try:
        result = xc_call_floats(binding, _E_MINIBATCH, x, aux, ints, floats, context)
    except error:
        if devscan and _pending_text(error).find("contains NaN or infinity") >= 0:
            _drop_pending()
            raise Error("mojolearn: X contains NaN or infinity")
        raise Error(String(error))
    _ = x^
    _ = aux^
    _ = init_arr^
    _ = weights^
    var f = result[0]
    var i = result[1]
    var s = result[2]
    _set(model, "cluster_centers_", _xc_array(f[0], Python.tuple(k, d), "<f4", "f", context))
    _set(model, "counts_", _xc_array(f[1], Python.tuple(k), "<f4", "f", context))
    _set(model, "labels_", _xc_array(i[0], Python.tuple(n), "<i4", "i", context))
    _set(model, "inertia_", _float(s[0]))
    _set(model, "n_steps_", _int(s[1]))
    _set(model, "n_iter_", _int(s[2]))
    _set(model, "n_features_in_", PythonObject(d))


def _mbk_partial_fit(model: PythonObject, binding: PythonObject, source: PythonObject,
                     sample_weight: PythonObject, context: PythonObject) raises:
    """`MiniBatchKMeans.partial_fit(X, sample_weight)`: one step on all of X;
    the random stream and the reassignment counter carry over."""
    var b = Python.import_module("builtins")
    var x = _f32(source, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var k = Int(py=_int(_attr(model, "n_clusters")))
    var first = _attr_or_none(model, "cluster_centers_") is None
    if not first and not Bool(py=family_call(b.hasattr, model, PythonObject("_partial_state"))):
        # a model from fit(): continue from its centers and counts, as sklearn
        var n0 = min(Int(py=_int(_attr(model, "batch_size"))), n)
        _set(model, "_partial_batch", PythonObject(n0))
        _set(model, "_partial_init", Python.tuple(PythonObject(0), PythonObject(0)))
        var seed0 = _seed(_attr(model, "random_state"))
        _set(model, "_partial_state",
             Python.tuple(Python.tuple(PythonObject(seed0 >> 32), PythonObject(seed0 & 0xFFFFFFFF)), PythonObject(0)))
    if not first:
        var fitted = _attr(model, "n_features_in_")
        if _truthy(family_call(Python.import_module("operator").ne, PythonObject(d), fitted)):
            raise Error("X has " + String(d) + " features, MiniBatchKMeans was fitted with " + family_str(fitted))
    var parts = Python.list()
    var mode = PythonObject(0)
    var init_size = PythonObject(0)
    var batch = PythonObject(0)
    var state0 = PythonObject(0)
    var state1 = PythonObject(0)
    var since = PythonObject(0)
    if first:
        _rows(model, k, n)
        var batch_eff = min(Int(py=_int(_attr(model, "batch_size"))), n)
        var size_field = _attr(model, "init_size")
        var size = 3 * batch_eff if size_field is None else Int(py=_int(size_field))
        if size < k:
            size = 3 * k
        size = min(size, n)
        var init = _attr(model, "init")
        var code: Int
        if family_is_str(init):
            if not _one_of(init, "k-means++", "random"):
                raise Error("init must be 'k-means++', 'random' or an array, got " + family_repr(init))
            code = 0 if _eq(init, PythonObject("k-means++")) else 1
        else:
            _ = py_call(parts.append, _init_array(model, source, init, k, d, context))
            code = 2
        var seed = _seed(_attr(model, "random_state"))
        state0 = PythonObject(seed >> 32)
        state1 = PythonObject(seed & 0xFFFFFFFF)
        since = PythonObject(0)
        mode = PythonObject(code)
        init_size = PythonObject(size)
        batch = PythonObject(batch_eff)
        _set(model, "_partial_batch", batch)
        _set(model, "_partial_init", Python.tuple(mode, init_size))
        _set(model, "n_steps_", PythonObject(0))
    else:
        var pinit = _attr(model, "_partial_init")
        mode = pinit[0]
        init_size = pinit[1]
        batch = _attr(model, "_partial_batch")
        var pstate = _attr(model, "_partial_state")
        var words = pstate[0]
        since = pstate[1]
        state0 = words[0]
        state1 = words[1]
        _ = py_call(parts.append, _attr(model, "cluster_centers_"))
        _ = py_call(parts.append, _attr(model, "counts_"))
    _ = py_call(parts.append, _weights(sample_weight, n, context))
    # ORDER MATCHES x_cluster/entries.mojo::minibatch_partial_entry: n, d,
    # k, first, init mode, init_size, batch, steps since reassignment, the
    # stream's two words, has_weights.
    var ints = Python.list(PythonObject(n), PythonObject(d), PythonObject(k), _flag(first), _int(mode),
                           _int(init_size), _int(batch), _int(since), _int(state0), _int(state1),
                           _flag(sample_weight is not None))
    var aux = _aux(parts, context)
    var floats = Python.list(_float(_attr(model, "reassignment_ratio")))
    var result = xc_call_floats(binding, _E_MINIBATCH_PARTIAL, x, aux, ints, floats, context)
    _ = x^
    _ = aux^
    _ = parts^
    var f = result[0]
    var i = result[1]
    var s = result[2]
    _set(model, "cluster_centers_", _xc_array(f[0], Python.tuple(k, d), "<f4", "f", context))
    _set(model, "counts_", _xc_array(f[1], Python.tuple(k), "<f4", "f", context))
    _set(model, "labels_", _xc_array(i[0], Python.tuple(n), "<i4", "i", context))
    _set(model, "inertia_", _float(s[0]))
    _set(model, "_partial_state", Python.tuple(Python.tuple(_int(s[2]), _int(s[3])), _int(s[1])))
    var steps = family_call(b.getattr, model, PythonObject("n_steps_"), PythonObject(0))
    _set(model, "n_steps_", family_call(Python.import_module("operator").add, steps, PythonObject(1)))
    _set(model, "n_features_in_", PythonObject(d))


# ---------------------------------------------------------------- BisectingKMeans

def _bisect_fit(model: PythonObject, binding: PythonObject, source: PythonObject, sample_weight: PythonObject,
                context: PythonObject) raises:
    """`BisectingKMeans.fit(X, sample_weight)`."""
    var init = _attr(model, "init")
    if not _one_of(init, "random", "k-means++"):
        raise Error("NotImplementedError: mojolearn BisectingKMeans: init=" + family_repr(init)
                    + " is not implemented; 'random' or 'k-means++'")
    var strategy = _attr(model, "bisecting_strategy")
    if not _one_of(strategy, "biggest_inertia", "largest_cluster"):
        raise Error("bisecting_strategy must be 'biggest_inertia' or 'largest_cluster', got " + family_repr(strategy))
    var algorithm = _attr(model, "algorithm")
    if not _one_of(algorithm, "lloyd", "elkan"):
        raise Error("algorithm must be 'lloyd' or 'elkan', got " + family_repr(algorithm))
    var x = _f32(source, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var k = Int(py=_int(_attr(model, "n_clusters")))
    _rows(model, k, n)
    var n_init = _int(_attr(model, "n_init"))
    if Int(py=n_init) < 1:
        raise Error("n_init must be >= 1")
    # ORDER MATCHES x_cluster/entries.mojo::bisect_entry: n, d, k, n_init,
    # init (1 random), max_iter, seed, largest_cluster.
    var random_init = _flag(_eq(init, PythonObject("random")))
    var max_iter = _int(_attr(model, "max_iter"))
    var seed = _seed(_attr(model, "random_state"))
    var ints = Python.list(PythonObject(n), PythonObject(d), PythonObject(k), n_init, random_init, max_iter,
                           PythonObject(seed), _flag(_eq(strategy, PythonObject("largest_cluster"))))
    var w = PythonObject(None)
    if sample_weight is not None:
        w = _weights(sample_weight, n, context)
    var floats = Python.list(_float(_attr(model, "tol")))
    var result = xc_call_floats(binding, _E_BISECT, x, w, ints, floats, context)
    _ = x^
    _ = w^
    var f = result[0]
    var i = result[1]
    var s = result[2]
    var b = Python.import_module("builtins")
    _set(model, "cluster_centers_", _xc_array(f[0], Python.tuple(k, d), "<f4", "f", context))
    var m = Int(py=py_call(b.len, i[1])) // 3
    _set(model, "_tree_centers", _xc_array(f[1], Python.tuple(m, d), "<f4", "f", context))
    _set(model, "_tree_nodes", _xc_array(i[1], Python.tuple(m, 3), "<i4", "i", context))
    _set(model, "labels_", _xc_array(i[0], Python.tuple(n), "<i4", "i", context))
    _set(model, "inertia_", _float(s[0]))
    _set(model, "n_features_in_", PythonObject(d))


def _centers_fit(model: PythonObject, binding: PythonObject, source: PythonObject,
                 context: PythonObject) raises:
    """`self.fit(X)` of a centers estimator (fit_predict and fit_transform
    pass no sample_weight)."""
    if family_isinstance(model, context["family_cfa_minibatch"]):
        _mbk_fit(model, binding, source, PythonObject(None), context)
    else:
        _bisect_fit(model, binding, source, PythonObject(None), context)


# ---------------------------------------------------------------- MeanShift

def _meanshift_fit(model: PythonObject, binding: PythonObject, source: PythonObject,
                   context: PythonObject) raises:
    """`MeanShift.fit(X)`."""
    var b = Python.import_module("builtins")
    var x = _f32(source, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var seeds = PythonObject(None)
    var seeds_field = _attr(model, "seeds")
    if seeds_field is not None:
        seeds = _f32(seeds_field, "seeds", context)
        if Int(py=seeds.shape[1]) != d:
            raise Error("seeds have " + String(Int(py=seeds.shape[1])) + " features, X has " + String(d))
    var bw = PythonObject(0.0)
    var bandwidth = _attr(model, "bandwidth")
    if bandwidth is not None:
        bw = _float(bandwidth)
        if not Float64(py=bw) > 0:
            raise Error("bandwidth needs to be greater than zero or None, got "
                        + String(py=py_call(b.format, bw, PythonObject("f"))))
    # ORDER MATCHES x_cluster/entries.mojo::meanshift_entry: n, d, n_seeds,
    # cluster_all, max_iter, bin_seeding, min_bin_freq.
    var n_seeds = 0 if seeds is None else Int(py=seeds.shape[0])
    var cluster_all = _flag(_truthy(_attr(model, "cluster_all")))
    var max_iter = _int(_attr(model, "max_iter"))
    var bin_seeding = _flag(_truthy(_attr(model, "bin_seeding")))
    var ints = Python.list(PythonObject(n), PythonObject(d), PythonObject(n_seeds), cluster_all, max_iter,
                           bin_seeding, _int(_attr(model, "min_bin_freq")))
    var result = xc_call_floats(binding, _E_MEANSHIFT, x, seeds, ints, Python.list(bw), context)
    _ = x^
    _ = seeds^
    var f = result[0]
    var i = result[1]
    var s = result[2]
    var kc = Int(py=_int(s[2]))
    _set(model, "cluster_centers_", _xc_array(f[0], Python.tuple(kc, d), "<f4", "f", context))
    _set(model, "labels_", _xc_array(i[0], Python.tuple(n), "<i4", "i", context))
    _set(model, "bandwidth_", _float(s[0]))
    _set(model, "n_iter_", _int(s[1]))
    _set(model, "n_features_in_", PythonObject(d))


# ---------------------------------------------------------------- OPTICS

def _optics_metric(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`OPTICS._metric_code`: (code, p); euclidean at p=2 keeps the
    squared-distance route (-1)."""
    var b = Python.import_module("builtins")
    var metric = _attr(model, "metric")
    var params = _attr(model, "metric_params")
    var p_field = _attr(model, "p")
    var p: PythonObject
    if _truthy(params):
        p = _float(family_call(params.get, PythonObject("p"), p_field))
    else:
        p = _float(p_field) if p_field is not None else PythonObject(2.0)
    var pv = Float64(py=p)
    if _eq(metric, PythonObject("minkowski")):
        if not pv >= 1:
            raise Error("p must be >= 1 for minkowski, got " + family_str(p))
        if pv == 2:
            return Python.tuple(PythonObject(-1), PythonObject(2.0))
        if pv == 1:
            return Python.tuple(PythonObject(1), PythonObject(1.0))
        if isinf(pv) and pv > 0:
            return Python.tuple(PythonObject(2), p)
        return Python.tuple(PythonObject(3), p)
    if _one_of(metric, "euclidean", "l2"):
        return Python.tuple(PythonObject(-1), PythonObject(2.0))
    var table = context["family_cfa_optics_metrics"]
    if not py_contains(table, metric):
        var names = py_call(Python.import_module("operator").add, py_call(b.sorted, table),
                            Python.list(PythonObject("minkowski")))
        raise Error("NotImplementedError: mojolearn OPTICS: metric=" + family_repr(metric)
                    + " is not implemented; one of " + family_repr(names) + " (x_cluster/NOT_IMPLEMENTED.tsv)")
    return Python.tuple(table[metric], p)


def _optics_size(value: PythonObject, n: Int, name: String) raises -> Int:
    """`OPTICS._size`: an int count in [2, n], or a fraction in (0, 1] of n
    (at least 2)."""
    var b = Python.import_module("builtins")
    var op = Python.import_module("operator")
    if family_isinstance(value, b.bool) or value is None:
        raise Error(name + " must be an int >= 2 or a float in (0, 1]")
    if family_isinstance(value, b.int) or (family_isinstance(value, b.float)
                                           and _truthy(family_call(op.gt, value, PythonObject(1)))):
        if (_truthy(family_call(op.ne, _int(value), value)) or _truthy(family_call(op.lt, value, PythonObject(2)))
                or _truthy(family_call(op.gt, value, PythonObject(n)))):
            raise Error(name + " must be no greater than the number of samples (" + String(n) + ") and >= 2, got "
                        + family_str(value))
        return Int(py=_int(value))
    if not (_truthy(family_call(op.lt, PythonObject(0), value)) and _truthy(family_call(op.le, value, PythonObject(1)))):
        raise Error(name + " must be in (0, 1], got " + family_str(value))
    return max(2, Int(py=_int(family_call(op.mul, value, PythonObject(n)))))


def _optics_fit(model: PythonObject, binding: PythonObject, source: PythonObject,
                context: PythonObject) raises:
    """`OPTICS.fit(X)`."""
    var code = _optics_metric(model, context)
    var metric = Int(py=code[0])
    var pw = code[1]
    var method = _attr(model, "cluster_method")
    if not _one_of(method, "xi", "dbscan"):
        raise Error("cluster_method must be 'xi' or 'dbscan', got " + family_repr(method))
    var x = _f32(source, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var ms = _optics_size(_attr(model, "min_samples"), n, "min_samples")
    var mcs_field = _attr(model, "min_cluster_size")
    var mcs = ms if mcs_field is None else _optics_size(mcs_field, n, "min_cluster_size")
    var max_eps = _float(_attr(model, "max_eps"))
    var eps_field = _attr(model, "eps")
    var eps = max_eps if eps_field is None else _float(eps_field)
    var is_xi = _eq(method, PythonObject("xi"))
    if _eq(method, PythonObject("dbscan")) and Float64(py=eps) > Float64(py=max_eps):
        raise Error("Specify an epsilon smaller than " + family_str(max_eps) + ". Got " + family_str(eps) + ".")
    var xi = _float(_attr(model, "xi"))
    if not (Float64(py=xi) >= 0 and Float64(py=xi) <= 1):
        raise Error("xi must be in [0, 1]")
    if metric == _OPTICS_PRECOMPUTED and n != d:
        raise Error("metric='precomputed' needs a square distance matrix, got (" + String(n) + ", " + String(d) + ")")
    # ORDER MATCHES x_cluster/entries.mojo::optics_entry: n, d, min_samples,
    # min_cluster_size, method (0 xi), predecessor_correction, metric;
    # max_eps, xi, eps, p.
    var ints = Python.list(PythonObject(n), PythonObject(d), PythonObject(ms), PythonObject(mcs),
                           PythonObject(0 if is_xi else 1),
                           _flag(_truthy(_attr(model, "predecessor_correction"))), PythonObject(metric))
    var floats = Python.list(max_eps, _float(_attr(model, "xi")), eps, pw)
    var result = xc_call_floats(binding, _E_OPTICS, x, PythonObject(None), ints, floats, context)
    _ = x^
    var f = result[0]
    var i = result[1]
    _set(model, "core_distances_", _xc_array(f[0], Python.tuple(n), "<f4", "f", context))
    _set(model, "reachability_", _xc_array(f[1], Python.tuple(n), "<f4", "f", context))
    _set(model, "ordering_", _xc_array(i[0], Python.tuple(n), "<i4", "i", context))
    _set(model, "predecessor_", _xc_array(i[1], Python.tuple(n), "<i4", "i", context))
    _set(model, "labels_", _xc_array(i[2], Python.tuple(n), "<i4", "i", context))
    if is_xi:
        var rows = Int(py=py_call(Python.import_module("builtins").len, i[3])) // 2
        _set(model, "cluster_hierarchy_", _xc_array(i[3], Python.tuple(rows, 2), "<i4", "i", context))
    _set(model, "n_features_in_", PythonObject(d))


# ---------------------------------------------------------------- AffinityPropagation

def _affinity_fit(model: PythonObject, binding: PythonObject, source: PythonObject,
                  context: PythonObject) raises:
    """`AffinityPropagation.fit(X)`."""
    var b = Python.import_module("builtins")
    var affinity = _attr(model, "affinity")
    if not _one_of(affinity, "euclidean", "precomputed"):
        raise Error("affinity must be 'euclidean' or 'precomputed', got " + family_repr(affinity))
    var damping_field = _attr(model, "damping")
    var damping = Float64(py=_float(damping_field))
    if not (0.5 <= damping and damping < 1):
        raise Error("damping must be in [0.5, 1), got " + family_str(damping_field))
    if Int(py=_int(_attr(model, "max_iter"))) < 1 or Int(py=_int(_attr(model, "convergence_iter"))) < 1:
        raise Error("max_iter and convergence_iter must be >= 1")
    var x = _f32(source, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var pre = _eq(affinity, PythonObject("precomputed"))
    if pre and n != d:
        raise Error("The matrix of similarities must be a square array. Got (" + String(n) + ", " + String(d)
                    + ") instead.")
    var pref_arr = PythonObject(None)
    var mode = 0
    var scalar = PythonObject(0.0)
    var preference = _attr(model, "preference")
    if preference is not None:
        if family_isinstance(preference, Python.tuple(b.int, b.float)):
            mode = 1
            scalar = _float(preference)
        else:
            pref_arr = _f32(Python.list(family_call(b.list, preference)), "preference", context)
            if Int(py=pref_arr.shape[1]) != n:
                raise Error("preference must be a scalar or have one value per sample")
            mode = 2
    # ORDER MATCHES x_cluster/entries.mojo::affinity_entry: n, d,
    # precomputed, preference mode, max_iter, convergence_iter, seed;
    # damping, the scalar preference.
    var max_iter = _int(_attr(model, "max_iter"))
    var conv = _int(_attr(model, "convergence_iter"))
    var seed = _seed(_attr(model, "random_state"))
    var ints = Python.list(PythonObject(n), PythonObject(d), _flag(pre), PythonObject(mode), max_iter, conv,
                           PythonObject(seed))
    var floats = Python.list(_float(damping_field), scalar)
    var result = xc_call_floats(binding, _E_AFFINITY, x, pref_arr, ints, floats, context)
    _ = x^
    _ = pref_arr^
    var f = result[0]
    var i = result[1]
    var s = result[2]
    var k = Int(py=py_call(b.len, i[0]))
    _set(model, "cluster_centers_indices_", _xc_array(i[0], Python.tuple(k), "<i4", "i", context))
    _set(model, "labels_", _xc_array(i[1], Python.tuple(n), "<i4", "i", context))
    _set(model, "affinity_matrix_", _xc_array(f[0], Python.tuple(n, n), "<f4", "f", context))
    # the final availability and responsibility diagonals (a diagnostic the
    # verifier lanes hash; scikit-learn keeps no such attribute)
    _set(model, "_message_diag", _xc_array(f[1], Python.tuple(Int(py=py_call(b.len, f[1]))), "<f4", "f", context))
    _set(model, "n_iter_", _int(s[0]))
    _set(model, "n_features_in_", PythonObject(d))
    if not pre:
        _set(model, "cluster_centers_", _xc_array(f[2], Python.tuple(k, d), "<f4", "f", context))


# ---------------------------------------------------------------- HDBSCAN

def _i32_at(array: PythonObject, index: Int) raises -> Int:
    return Int(FAMILY_I32P(unsafe_from_address=Int(py=classical_address(array)))[index])


def _head(array: PythonObject, count: Int, typestr: String, context: PythonObject) raises -> PythonObject:
    """The first `count` elements of a 1-D 4-byte Array as a new Array
    (`array[:count]`, same values)."""
    var out = family_array(Python.tuple(count), typestr, context)
    if count:
        memcpy(dest=FAMILY_I32P(unsafe_from_address=Int(py=classical_address(out))),
               src=FAMILY_I32P(unsafe_from_address=Int(py=classical_address(array))), count=count)
        _ = out  # keep the Python object alive through the native call above
    return out


def _hdbscan_int_param(model: PythonObject, name: String) raises -> PythonObject:
    var value = _attr(model, name)
    if not family_is_int(value):
        raise Error("TypeError: mojolearn HDBSCAN: " + name + " must be an int, got " + family_type_name(value))
    return value


def _hdbscan_mode(model: PythonObject, binding: PythonObject, context: PythonObject) raises:
    """`HDBSCAN._extension`: the binary's own compile-time mode must be the
    one requested."""
    var b = Python.import_module("builtins")
    var want = _attr_or_none(model, "numeric_mode")
    if want is None or not _truthy(want):
        want = context["backend_fields"]["_DEFAULT_MODE"]
    var getter = py_call(b.getattr, binding, PythonObject("hdbscan_numeric_mode"), PythonObject(None))
    if getter is not None:
        var got = Int(py=_int(family_call(getter)))
        var wanted = py_get(context["family_cfa_hdbscan_mode_codes"], want, PythonObject(None))
        if wanted is None or Int(py=wanted) != got:
            raise Error("RuntimeError: mojolearn HDBSCAN: numeric_mode=" + family_repr(want) + " was requested but "
                        + String(py=py_call(b.getattr, binding, PythonObject("__name__"))) + " reports compile-time "
                        + "mode code " + String(got) + "; rebuild it with bash bindings/build_hdbscan.sh")


def _hdbscan_fit(model: PythonObject, binding: PythonObject, source: PythonObject,
                 context: PythonObject) raises:
    """`HDBSCAN.fit(X)`."""
    var b = Python.import_module("builtins")
    var view = family_view(source, 2, "X", context)
    var x = family_cast_view(view, "X", context)
    # `as_f32_c`'s `copied`: the cast made new storage.
    var copied = Int(py=classical_address(x)) != Int(py=classical_address(view))
    _ = x  # keep the Python object alive through the native call above
    _ = view  # keep the Python object alive through the native call above
    _ = view^
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    if n < 2:
        # scikit-learn's check_array wording (ensure_min_samples=2); the
        # mutual-reachability tree needs two points
        raise Error("Found array with " + String(n) + " sample(s) (shape=(" + String(n) + ", " + String(d)
                    + ")) while a minimum of 2 is required.")
    var min_cluster_size = _hdbscan_int_param(model, "min_cluster_size")
    var max_cluster_size = _hdbscan_int_param(model, "max_cluster_size")
    var ms_field = _attr(model, "min_samples")
    var min_samples: PythonObject
    if ms_field is None:
        min_samples = _int(min_cluster_size)
    elif not family_is_int(ms_field):
        raise Error("TypeError: mojolearn HDBSCAN: min_samples must be an int or None")
    else:
        min_samples = _int(ms_field)
    var metric_field = _attr(model, "metric")
    var metric: PythonObject
    if family_is_str(metric_field):
        var metrics = context["family_cfa_hdbscan_metrics"]
        if not py_contains(metrics, metric_field):
            raise Error("mojolearn HDBSCAN: metric must be 'euclidean' (cuML's L2SqrtExpanded, the one "
                        + "fit_hdbscan accepts), got " + family_repr(metric_field))
        metric = metrics[metric_field]
    else:
        metric = _int(metric_field)
    var method_field = _attr(model, "cluster_selection_method")
    var method: PythonObject
    if family_is_str(method_field):
        var selections = context["family_cfa_hdbscan_selection"]
        if not py_contains(selections, method_field):
            raise Error("mojolearn HDBSCAN: cluster_selection_method must be 'eom' or 'leaf', got "
                        + family_repr(method_field))
        method = selections[method_field]
    else:
        method = _int(method_field)
    var alpha_field = _attr(model, "alpha")
    if not family_isinstance(alpha_field, Python.tuple(b.int, b.float)) or family_isinstance(alpha_field, b.bool):
        raise Error("TypeError: mojolearn HDBSCAN: alpha must be a real number")
    var cse_field = _attr(model, "cluster_selection_epsilon")
    if not family_isinstance(cse_field, Python.tuple(b.int, b.float)) or family_isinstance(cse_field, b.bool):
        raise Error("TypeError: mojolearn HDBSCAN: cluster_selection_epsilon must be a real number")
    var cse = Float64(py=_float(cse_field))
    if not (cse >= 0.0) or (isinf(cse) and cse > 0):
        raise Error("mojolearn HDBSCAN: cluster_selection_epsilon must be a finite number >= 0, got "
                    + family_repr(cse_field) + " (scikit-learn's range)")
    var want_pd_field = _attr(model, "prediction_data")
    if not family_isinstance(want_pd_field, b.bool):
        raise Error("TypeError: mojolearn HDBSCAN: prediction_data must be a bool")
    var want_pd = Bool(py=want_pd_field)
    var labels = family_array(Python.tuple(n), "<i4", context)
    var core = family_array(Python.tuple(n), "<f4", context)
    var info = family_array(Python.tuple(5 if want_pd else 4), "<i4", context)
    # ORDER MATCHES bindings/_mojolearn_hdbscan.mojo::hdbscan_fit_binding.
    # x, labels_out, core_dists_out, info_out
    var addrs = Python.list(classical_address(x), classical_address(labels), classical_address(core),
                            classical_address(info))
    _ = core  # keep the Python object alive through the native call above
    _ = labels  # keep the Python object alive through the native call above
    _ = x  # keep the Python object alive through the native call above
    _ = info  # keep the Python object alive through the native call above
    var probs = family_array(Python.tuple(n), "<f4", context)
    var t_par = PythonObject(None)
    var t_ch = PythonObject(None)
    var t_lam = PythonObject(None)
    var t_sz = PythonObject(None)
    var t_inv = PythonObject(None)
    if want_pd:
        # tree_parents_out, tree_children_out, tree_lambdas_out,
        # tree_sizes_out (2 * n each), inverse_label_map_out (n)
        t_par = family_array(Python.tuple(2 * n), "<i4", context)
        t_ch = family_array(Python.tuple(2 * n), "<i4", context)
        t_lam = family_array(Python.tuple(2 * n), "<f4", context)
        t_sz = family_array(Python.tuple(2 * n), "<i4", context)
        t_inv = family_array(Python.tuple(n), "<i4", context)
        _ = py_call(addrs.append, classical_address(t_par))
        _ = t_par  # keep the Python object alive through the native call above
        _ = py_call(addrs.append, classical_address(t_ch))
        _ = t_ch  # keep the Python object alive through the native call above
        _ = py_call(addrs.append, classical_address(t_lam))
        _ = t_lam  # keep the Python object alive through the native call above
        _ = py_call(addrs.append, classical_address(t_sz))
        _ = t_sz  # keep the Python object alive through the native call above
        _ = py_call(addrs.append, classical_address(t_inv))
        _ = t_inv  # keep the Python object alive through the native call above
    # probabilities_out, always LAST (DEVIATION 5116)
    _ = py_call(addrs.append, classical_address(probs))
    _ = probs  # keep the Python object alive through the native call above
    _hdbscan_mode(model, binding, context)
    # n, d, min_samples, min_cluster_size, max_cluster_size, alpha,
    # allow_single_cluster, cluster_selection_method, cluster_selection_epsilon, metric
    var params = Python.list(PythonObject(n), PythonObject(d), min_samples, _int(min_cluster_size),
                             _int(max_cluster_size), _float(alpha_field),
                             _flag(_truthy(_attr(model, "allow_single_cluster"))), method, _float(cse_field), metric)
    var n_clusters = family_call(family_entry(binding, "hdbscan_fit", context), addrs, params)
    _set(model, "n_features_in_", PythonObject(d))
    _set(model, "labels_", labels)
    _set(model, "core_distances_", core)
    _set(model, "probabilities_", probs)
    _set(model, "n_clusters_", _int(n_clusters))
    _set(model, "n_outliers_", PythonObject(_i32_at(info, 1)))
    _set(model, "n_boruvka_rounds_", PythonObject(_i32_at(info, 2)))
    _set(model, "n_condensed_clusters_", PythonObject(_i32_at(info, 3)))
    _set(model, "_prediction_data", PythonObject(None))
    _set(model, "_raw_data", PythonObject(None))
    if want_pd:
        _set(model, "_prediction_data",
             _hdbscan_prediction_data(model, binding, n, _i32_at(info, 4), t_par, t_ch, t_lam, t_sz, t_inv, context))
        # cuML keeps the training matrix as `_raw_data` (hdbscan.pyx:1323);
        # a borrowed input is copied so a caller's later write cannot move it.
        var raw = x
        if not copied:
            raw = family_array(Python.tuple(n, d), "<f4", context)
            memcpy(dest=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(raw))),
                   src=FAMILY_F32P(unsafe_from_address=Int(py=classical_address(x))), count=n * d)
            _ = raw  # keep the Python object alive through the native call above
            _ = x  # keep the Python object alive through the native call above
        _set(model, "_raw_data", raw)
    _ = x^
    _ = info^


def _hdbscan_prediction_data(model: PythonObject, binding: PythonObject, n: Int, n_edges: Int,
                             t_par: PythonObject, t_ch: PythonObject, t_lam: PythonObject, t_sz: PythonObject,
                             t_inv: PythonObject, context: PythonObject) raises -> PythonObject:
    """cuML `generate_prediction_data` (`hdbscan.pyx:394-427`,
    `prediction_data.cu:92-239`) through the binding's host function."""
    var n_cond = Int(py=_attr(model, "n_condensed_clusters_"))
    var n_sel = Int(py=_attr(model, "n_clusters_"))
    var pd = Python.dict()
    py_setitem(pd, PythonObject("n_edges"), PythonObject(n_edges))
    py_setitem(pd, PythonObject("n_condensed"), PythonObject(n_cond))
    var parents = _head(t_par, n_edges, "<i4", context)
    var children = _head(t_ch, n_edges, "<i4", context)
    var lambdas = _head(t_lam, n_edges, "<f4", context)
    var sizes = _head(t_sz, n_edges, "<i4", context)
    var inverse = _head(t_inv, max(n_sel, 1), "<i4", context)
    var deaths = family_array(Python.tuple(n_cond), "<f4", context)
    var selected = family_array(Python.tuple(max(n_sel, 1)), "<i4", context)
    var exemplar_idx = family_array(Python.tuple(n), "<i4", context)
    var offsets = family_array(Python.tuple(n_sel + 1), "<i4", context)
    var index_into = family_array(Python.tuple(n_edges + 1), "<i4", context)
    py_setitem(pd, PythonObject("parents"), parents)
    py_setitem(pd, PythonObject("children"), children)
    py_setitem(pd, PythonObject("lambdas"), lambdas)
    py_setitem(pd, PythonObject("sizes"), sizes)
    py_setitem(pd, PythonObject("inverse_label_map"), inverse)
    py_setitem(pd, PythonObject("deaths"), deaths)
    py_setitem(pd, PythonObject("selected_clusters"), selected)
    py_setitem(pd, PythonObject("exemplar_idx"), exemplar_idx)
    py_setitem(pd, PythonObject("exemplar_label_offsets"), offsets)
    py_setitem(pd, PythonObject("index_into_children"), index_into)
    var labels = _attr(model, "labels_")
    # labels, parents, children, lambdas, sizes, inverse_label_map,
    # deaths_out, selected_clusters_out, exemplar_idx_out,
    # exemplar_label_offsets_out, index_into_children_out;
    # n_leaves, n_edges, n_clusters, n_selected
    var n_ex = family_call(
        family_entry(binding, "hdbscan_generate_prediction_data", context),
        Python.list(classical_address(labels), classical_address(parents), classical_address(children),
                    classical_address(lambdas), classical_address(sizes), classical_address(inverse),
                    classical_address(deaths), classical_address(selected), classical_address(exemplar_idx),
                    classical_address(offsets), classical_address(index_into)),
        Python.list(PythonObject(n), PythonObject(n_edges), PythonObject(n_cond), PythonObject(n_sel)),
    )
    _ = labels  # keep the Python object alive through the native call above
    _ = children  # keep the Python object alive through the native call above
    _ = parents  # keep the Python object alive through the native call above
    _ = inverse  # keep the Python object alive through the native call above
    _ = sizes  # keep the Python object alive through the native call above
    _ = lambdas  # keep the Python object alive through the native call above
    _ = exemplar_idx  # keep the Python object alive through the native call above
    _ = selected  # keep the Python object alive through the native call above
    _ = deaths  # keep the Python object alive through the native call above
    _ = index_into  # keep the Python object alive through the native call above
    _ = offsets  # keep the Python object alive through the native call above
    var n_exemplars = Int(py=_int(n_ex))
    py_setitem(pd, PythonObject("n_exemplars"), PythonObject(n_exemplars))
    py_setitem(pd, PythonObject("exemplar_idx"), _head(exemplar_idx, max(n_exemplars, 1), "<i4", context))
    _ = labels^
    return pd


# ---------------------------------------------------------------- door

def cluster_fit_a_operation(model: PythonObject, operation: String, args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    """args[0] is the resolved binding, args[1] X, args[2] sample_weight
    where the method takes one."""
    classical_connect_numpy(context)
    var binding = args[0]
    if operation == "family_cfa_mbk_fit":
        _mbk_fit(model, binding, args[1], args[2], context)
        return model
    if operation == "family_cfa_mbk_partial_fit":
        _mbk_partial_fit(model, binding, args[1], args[2], context)
        return model
    if operation == "family_cfa_bisect_fit":
        _bisect_fit(model, binding, args[1], args[2], context)
        return model
    if operation == "family_cfa_fit_predict":
        _centers_fit(model, binding, args[1], context)
        return _attr(model, "labels_")
    if operation == "family_cfa_fit_transform":
        _centers_fit(model, binding, args[1], context)
        return cluster_operation(model, "family_xc_transform", Python.tuple(binding, args[1]), context)
    if operation == "family_cfa_meanshift_fit":
        _meanshift_fit(model, binding, args[1], context)
        return model
    if operation == "family_cfa_meanshift_fit_predict":
        _meanshift_fit(model, binding, args[1], context)
        return _attr(model, "labels_")
    if operation == "family_cfa_optics_fit":
        _optics_fit(model, binding, args[1], context)
        return model
    if operation == "family_cfa_optics_fit_predict":
        _optics_fit(model, binding, args[1], context)
        return _attr(model, "labels_")
    if operation == "family_cfa_ap_fit":
        _affinity_fit(model, binding, args[1], context)
        return model
    if operation == "family_cfa_ap_fit_predict":
        _affinity_fit(model, binding, args[1], context)
        return _attr(model, "labels_")
    if operation == "family_cfa_hdbscan_fit":
        _hdbscan_fit(model, binding, args[1], context)
        return model
    if operation == "family_cfa_hdbscan_fit_predict":
        _hdbscan_fit(model, binding, args[1], context)
        return _attr(model, "labels_")
    raise Error("RuntimeError: unknown family operation " + operation)

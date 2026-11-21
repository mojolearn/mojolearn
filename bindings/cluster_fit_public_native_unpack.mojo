# SPDX-License-Identifier: Apache-2.0
# UNPACK part of the x_cluster fit door: the fits of MiniBatchKMeans (fit,
# partial_fit), BisectingKMeans, MeanShift, OPTICS and AffinityPropagation
# (python/mojolearn/_expansion_cluster.py). The model's settings and X are
# read here, once per fit and in the order the Python method refused them,
# into one typed stage per fit (X and the admitted auxiliary arrays pinned,
# every setting a Mojo value); the typed provider connections' calls under
# GIL release are here too. Every refusal that compares numbers and every
# request is RUN's (cluster/cluster_fit_door_run.mojo), called at the place
# the compare stood. PACK and the operator are
# bindings/cluster_fit_public_native_pack.mojo. Same entries, same operands,
# same order: the bits do not move.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.admitted_family_door import admit_family, admit_family_weights
from providers.python.array_metadata_pyapi_door import ArrayPin
from bindings.native_loaded_module_door import native_loaded_module
from bindings.family_raw_attr_pack import family_raw_get, family_raw_get_or
from bindings.py_balanced import py_call, py_type, py_contains, py_setattr, py_item
from bindings.family_common_unpack import (
    family_repr,
    family_str,
    family_isinstance,
    family_nonfinite,
    family_shape_repr,
)
from bindings.family_common_pack import family_call
from bindings.py_cluster_resident import CRES_X_CLUSTER, cres_forget
from core.kmeans_variants_api import VariantsSpan
from core.cluster_centroid_fit_api import CentroidFitResultSlot
from core.optics_fit_api import OpticsMetric, OpticsFitResultSlot
from bindings.optics_fit_boundary_door import optics_fit_connection
from bindings.cluster_keys_unpack import cluster_keys
from bindings.cluster_centroid_fit_boundary_door import centroid_fit_connection
from bindings.kmeans_variants_boundary_door import variants_span
from cluster.cluster_fit_door_run import (
    meanshift_seed_features, meanshift_bandwidth_refused, meanshift_fit_request,
    optics_minkowski_refused, optics_minkowski_metric, optics_euclidean_metric, optics_eps_refused,
    optics_xi_range, optics_precomputed_square, optics_fit_request,
)


# ---------------------------------------------------------------- helpers

def _attr(model: PythonObject, name: String) raises -> PythonObject:
    """`getattr(model, name)`; its AttributeError is restored unchanged."""
    return family_raw_get(model, name)


def _attr_or_none(model: PythonObject, name: String) raises -> PythonObject:
    """`getattr(model, name, None)`."""
    return family_raw_get_or(model, name, PythonObject(None))


def _truthy(value: PythonObject) raises -> Bool:
    """`bool(value)`."""
    return Bool(py=family_call(native_loaded_module("builtins").bool, value))


def _int(value: PythonObject) raises -> PythonObject:
    """`int(value)`."""
    return family_call(native_loaded_module("builtins").int, value)


def _float(value: PythonObject) raises -> PythonObject:
    """`float(value)`."""
    return family_call(native_loaded_module("builtins").float, value)


def _eq(value: PythonObject, other: PythonObject) raises -> Bool:
    """`bool(value == other)`."""
    return _truthy(family_call(native_loaded_module("operator").eq, value, other))


def _one_of(value: PythonObject, a: String, b: String) raises -> Bool:
    """`value in (a, b)`."""
    return py_contains(Python.tuple(PythonObject(a), PythonObject(b)), value)


def _set(model: PythonObject, var name: String, value: PythonObject) raises:
    """`setattr(model, name, value)`: PACK's publication of one fitted
    attribute. It is defined here because `partial_fit` writes its stream
    state between its reads (minibatch_partial_stage), exactly where the
    Python method did. A fit storing new `cluster_centers_`
    releases the model's resident device copy of the old ones at once
    (bindings/cluster_resident_table_run.mojo, N024); the next query uploads
    the new object once."""
    if name == "cluster_centers_":
        cres_forget(model, CRES_X_CLUSTER)
    py_setattr(model, name^, value)


def _f32(source: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_expansion_cluster._f32(source, name)`: `as_f32_c(ndim=2)` and the
    by-name NaN/inf refusal."""
    try:
        return admit_family(source, 2, name, context, finite=True).array
    except error:
        if family_nonfinite(error, name):
            raise Error("mojolearn: " + name + " contains NaN or infinity")
        raise Error(String(error))


def _seed(random_state: PythonObject) raises -> Int:
    """`_expansion_cluster._seed`: None is 0; a non-negative int keeps its
    low 63 bits."""
    if random_state is None:
        return 0
    var b = native_loaded_module("builtins")
    if (py_type(random_state) is not b.int
            or Bool(py=py_call(native_loaded_module("operator").lt, random_state, PythonObject(0)))):
        raise Error("mojolearn: random_state must be None or a non-negative int "
                    + "(the lane's seeded host stream; a RandomState object is not taken)")
    return Int(py=py_call(native_loaded_module("operator").and_, random_state,
                          PythonObject(9223372036854775807)))


def _weights(sample_weight: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_expansion_cluster._weights`: None stays None; otherwise
    `check_sample_weight(sample_weight, n, allow_negative=True)`."""
    if sample_weight is None:
        return PythonObject(None)
    var pin = ArrayPin()
    _ = admit_family_weights(sample_weight, n, True, context, pin)
    return pin.keep


def _init_array(init: PythonObject, k: Int, d: Int, context: PythonObject) raises -> PythonObject:
    """Admit explicit initial centers; model execution never invokes init callbacks."""
    ref cpy = Python().cpython()
    if cpy.lib.get_function[Int32]("PyCallable_Check")(init._obj_ptr) != 0:
        raise Error("TypeError: Python init callbacks are not accepted; pass explicit initial centers or use 'k-means++' or 'random'")
    var arr = _f32(init, "init", context)
    if Int(py=arr.shape[0]) != k or Int(py=arr.shape[1]) != d:
        raise Error("The shape of the initial centers " + family_shape_repr(arr) + " does not match "
                    + "the number of clusters " + String(k) + " and features " + String(d) + ".")
    return arr


# ---------------------------------------------------------------- MeanShift
@fieldwise_init
struct MeanShiftFit(Movable):
    """`MeanShift.fit(X)` as typed values. `x` is the admitted float32 X and
    `seeds` the admitted explicit seeds (read only when `has_seeds`), both
    pinned through the fit; `bandwidth` 0.0 asks the entry to estimate it."""
    var x: PythonObject
    var seeds: PythonObject
    var has_seeds: Bool
    var n: Int
    var d: Int
    var n_seeds: Int
    var cluster_all: Bool
    var max_iter: Int
    var bin_seeding: Bool
    var min_bin_freq: Int
    var bandwidth: Float64


def meanshift_fit_stage(model: PythonObject, source: PythonObject, context: PythonObject) raises -> MeanShiftFit:
    """The reads of `MeanShift.fit`, in its order: X, seeds, bandwidth,
    cluster_all, max_iter, bin_seeding, min_bin_freq."""
    var x = _f32(source, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var seeds = PythonObject(None)
    var has_seeds = False
    var n_seeds = 0
    var seeds_field = _attr(model, "seeds")
    if seeds_field is not None:
        seeds = _f32(seeds_field, "seeds", context)
        has_seeds = True
        meanshift_seed_features(Int(py=seeds.shape[1]), d)
    var bw = Float64(0.0)
    var bandwidth = _attr(model, "bandwidth")
    if bandwidth is not None:
        bw = Float64(py=_float(bandwidth))
        if meanshift_bandwidth_refused(bw):
            raise Error("bandwidth needs to be greater than zero or None, got "
                        + String(py=py_call(native_loaded_module("builtins").format, PythonObject(bw),
                                            PythonObject("f"))))
    if has_seeds:
        n_seeds = Int(py=seeds.shape[0])
    var cluster_all = _truthy(_attr(model, "cluster_all"))
    var max_iter = _int(_attr(model, "max_iter"))
    var bin_seeding = _truthy(_attr(model, "bin_seeding"))
    var min_bin_freq = _int(_attr(model, "min_bin_freq"))
    return MeanShiftFit(x=x, seeds=seeds, has_seeds=has_seeds, n=n, d=d, n_seeds=n_seeds,
                        cluster_all=cluster_all, max_iter=Int(py=max_iter), bin_seeding=bin_seeding,
                        min_bin_freq=Int(py=min_bin_freq), bandwidth=bw)


def meanshift_fit_call(binding: PythonObject, stage: MeanShiftFit, mut result: CentroidFitResultSlot) raises:
    """The MeanShift entry on the typed transport, the GIL released around
    it."""
    var connection = centroid_fit_connection(binding)
    var seeds = variants_span(stage.seeds) if stage.has_seeds else VariantsSpan(0, 0)
    var request = meanshift_fit_request(variants_span(stage.x), seeds, stage.n, stage.d, stage.n_seeds,
        stage.cluster_all, stage.max_iter, stage.bin_seeding, stage.min_bin_freq, stage.bandwidth)
    try:
        with GILReleased(Python()):
            connection.driver.meanshift(request, result)
    finally:
        _ = connection.provider
        _ = stage.x
        _ = stage.seeds


# ---------------------------------------------------------------- OPTICS
def _optics_metric(model: PythonObject, context: PythonObject) raises -> OpticsMetric:
    """`OPTICS._metric_code`: (code, p); euclidean at p=2 keeps the
    squared-distance route (-1)."""
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
        if optics_minkowski_refused(pv):
            raise Error("p must be >= 1 for minkowski, got " + family_str(p))
        return optics_minkowski_metric(pv)
    if _one_of(metric, "euclidean", "l2"):
        return optics_euclidean_metric()
    var nk = cluster_keys()
    var table = py_item(context, nk[].k_family_cfa_optics_metrics)
    if not py_contains(table, metric):
        var names = py_call(native_loaded_module("operator").add,
                            py_call(native_loaded_module("builtins").sorted, table),
                            Python.list(PythonObject("minkowski")))
        raise Error("NotImplementedError: mojolearn OPTICS: metric=" + family_repr(metric)
                    + " is not implemented; one of " + family_repr(names) + " (x_cluster/NOT_IMPLEMENTED.tsv)")
    return OpticsMetric(Int(py=table[metric]), pv)


def _optics_size(value: PythonObject, n: Int, name: String) raises -> Int:
    """`OPTICS._size`: an int count in [2, n], or a fraction in (0, 1] of n
    (at least 2)."""
    var b = native_loaded_module("builtins")
    var op = native_loaded_module("operator")
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


@fieldwise_init
struct OpticsFit(Movable):
    """`OPTICS.fit(X)` as typed values. `x` is the admitted float32 X,
    pinned through the fit; `metric` is (code, p); the two sizes are row
    counts; `xi` is the value of the model's second read."""
    var x: PythonObject
    var n: Int
    var d: Int
    var min_samples: Int
    var min_cluster_size: Int
    var is_xi: Bool
    var predecessor_correction: Bool
    var metric: OpticsMetric
    var max_eps: Float64
    var xi: Float64
    var eps: Float64


def optics_fit_stage(model: PythonObject, source: PythonObject, context: PythonObject) raises -> OpticsFit:
    """The reads of `OPTICS.fit`, in its order: metric, cluster_method, X,
    the two sizes, max_eps, eps, xi, predecessor_correction."""
    var code = _optics_metric(model, context)
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
    if optics_eps_refused(_eq(method, PythonObject("dbscan")), Float64(py=eps), Float64(py=max_eps)):
        raise Error("Specify an epsilon smaller than " + family_str(max_eps) + ". Got " + family_str(eps) + ".")
    optics_xi_range(Float64(py=_float(_attr(model, "xi"))))
    optics_precomputed_square(code.code, n, d)
    var predecessor_correction = _truthy(_attr(model, "predecessor_correction"))
    # Preserve the second xi read at its original stage after the flag.
    var repeated_xi = Float64(py=_float(_attr(model, "xi")))
    return OpticsFit(x=x, n=n, d=d, min_samples=ms, min_cluster_size=mcs, is_xi=is_xi,
                     predecessor_correction=predecessor_correction, metric=code.copy(),
                     max_eps=Float64(py=max_eps), xi=repeated_xi, eps=Float64(py=eps))


def optics_fit_call(binding: PythonObject, stage: OpticsFit, mut result: OpticsFitResultSlot) raises:
    """The OPTICS entry on the typed transport, the GIL released around it."""
    var connection = optics_fit_connection(binding)
    var request = optics_fit_request(variants_span(stage.x), stage.n, stage.d, stage.min_samples,
        stage.min_cluster_size, stage.is_xi, stage.predecessor_correction, stage.metric, stage.max_eps,
        stage.xi, stage.eps)
    try:
        with GILReleased(Python()):
            connection.driver.fit(request, result)
    finally:
        _ = connection.provider
        _ = stage.x

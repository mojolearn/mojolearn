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
    family_is_str,
    family_isinstance,
    family_nonfinite,
    family_shape_repr,
)
from bindings.family_common_pack import family_call
from bindings.py_cluster_resident import CRES_X_CLUSTER, cres_forget
from core.kmeans_variants_api import VariantsSpan, VariantsResultSlot, VARIANTS_CAP_DEVSCAN
from core.cluster_centroid_fit_api import CentroidFitResultSlot
from core.optics_fit_api import OpticsMetric, OpticsFitResultSlot
from core.affinity_preference_plan import AffinityPreferencePlan
from bindings.affinity_preference_boundary_door import affinity_preference_prepare
from bindings.optics_fit_boundary_door import optics_fit_connection
from bindings.cluster_keys_unpack import cluster_keys
from bindings.cluster_centroid_fit_boundary_door import centroid_fit_connection
from bindings.kmeans_variants_boundary_door import (
    KMeansVariantsConnection, VariantsAuxiliaryPins, variants_connection, variants_span,
)
from bindings.cluster_pending_error_unpack import _pending_text, _drop_pending
from cluster.cluster_fit_door_run import (
    cluster_rows_cover, cluster_n_init_positive, bisect_fit_request,
    minibatch_auto_n_init, minibatch_fit_request,
    meanshift_seed_features, meanshift_bandwidth_refused, meanshift_fit_request,
    optics_minkowski_refused, optics_minkowski_metric, optics_euclidean_metric, optics_eps_refused,
    optics_xi_range, optics_precomputed_square, optics_fit_request,
    affinity_damping_refused, affinity_iterations_positive, affinity_similarity_square,
    affinity_preference_rows, affinity_fit_request,
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


# ---------------------------------------------------------------- BisectingKMeans
@fieldwise_init
struct BisectFit(Movable):
    """`BisectingKMeans.fit(X, sample_weight)` as typed values. `x` is the
    admitted float32 X and `weights` the admitted sample weights (None when
    the fit takes none), both pinned through the fit."""
    var x: PythonObject
    var weights: PythonObject
    var n: Int
    var d: Int
    var k: Int
    var n_init: Int
    var random_init: Bool
    var max_iter: Int
    var seed: Int
    var largest_cluster: Bool
    var tolerance: Float64


def bisect_fit_stage(model: PythonObject, source: PythonObject, sample_weight: PythonObject,
                     context: PythonObject) raises -> BisectFit:
    """The reads of `BisectingKMeans.fit`, in its order: init,
    bisecting_strategy, algorithm, X, n_clusters, n_init, max_iter,
    random_state, sample_weight, tol."""
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
    cluster_rows_cover(k, n)
    var n_init = Int(py=_int(_attr(model, "n_init")))
    cluster_n_init_positive(n_init)
    var random_init = _eq(init, PythonObject("random"))
    var max_iter = _int(_attr(model, "max_iter"))
    var seed = _seed(_attr(model, "random_state"))
    var largest_cluster = _eq(strategy, PythonObject("largest_cluster"))
    var w = PythonObject(None)
    if sample_weight is not None:
        w = _weights(sample_weight, n, context)
    var tolerance = _float(_attr(model, "tol"))
    return BisectFit(x=x, weights=w, n=n, d=d, k=k, n_init=n_init, random_init=random_init,
                     max_iter=Int(py=max_iter), seed=seed, largest_cluster=largest_cluster,
                     tolerance=Float64(py=tolerance))


def bisect_fit_call(binding: PythonObject, stage: BisectFit, mut result: VariantsResultSlot) raises:
    """The bisecting entry on the typed transport, the GIL released around
    it."""
    var connection = variants_connection(binding)
    var weight_span = variants_span(stage.weights)
    var source_span = variants_span(stage.x)
    var request = bisect_fit_request(source_span, weight_span, stage.n, stage.d, stage.k, stage.n_init,
        stage.random_init, stage.max_iter, stage.seed, stage.largest_cluster, stage.tolerance)
    try:
        with GILReleased(Python()):
            connection.driver.bisect_fit(request, result)
    finally:
        _ = connection.provider
        _ = stage.x
        _ = stage.weights


# ---------------------------------------------------------------- AffinityPropagation
@fieldwise_init
struct AffinityFit(Movable):
    """`AffinityPropagation.fit(X)` as typed values. `x` is the admitted
    float32 X, pinned through the fit; `preference` owns the per-sample
    preferences as float32 words (mode 2); `scalar` is the scalar preference
    (mode 1); mode 0 takes the median similarity."""
    var x: PythonObject
    var preference: AffinityPreferencePlan
    var n: Int
    var d: Int
    var precomputed: Bool
    var mode: Int
    var max_iter: Int
    var convergence_iter: Int
    var seed: Int
    var damping: Float64
    var scalar: Float64


def affinity_fit_stage(model: PythonObject, source: PythonObject, context: PythonObject) raises -> AffinityFit:
    """The reads of `AffinityPropagation.fit`, in its order: affinity,
    damping, the two iteration counts, X, preference, then max_iter,
    convergence_iter, random_state and damping again."""
    var affinity = _attr(model, "affinity")
    if not _one_of(affinity, "euclidean", "precomputed"):
        raise Error("affinity must be 'euclidean' or 'precomputed', got " + family_repr(affinity))
    var damping_field = _attr(model, "damping")
    if affinity_damping_refused(Float64(py=_float(damping_field))):
        raise Error("damping must be in [0.5, 1), got " + family_str(damping_field))
    affinity_iterations_positive(Int(py=_int(_attr(model, "max_iter"))))
    affinity_iterations_positive(Int(py=_int(_attr(model, "convergence_iter"))))
    var x = _f32(source, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var pre = _eq(affinity, PythonObject("precomputed"))
    affinity_similarity_square(pre, n, d)
    var preference_plan = AffinityPreferencePlan()
    var mode = 0
    var scalar = Float64(0.0)
    var preference = _attr(model, "preference")
    if preference is not None:
        var b = native_loaded_module("builtins")
        if family_isinstance(preference, Python.tuple(b.int, b.float)):
            mode = 1
            scalar = Float64(py=_float(preference))
        else:
            preference_plan = affinity_preference_prepare(preference, context)
            affinity_preference_rows(preference_plan.count, n)
            mode = 2
    var max_iter = _int(_attr(model, "max_iter"))
    var conv = _int(_attr(model, "convergence_iter"))
    var seed = _seed(_attr(model, "random_state"))
    var repeated_damping = _float(damping_field)
    return AffinityFit(x=x, preference=preference_plan^, n=n, d=d, precomputed=pre, mode=mode,
                       max_iter=Int(py=max_iter), convergence_iter=Int(py=conv), seed=seed,
                       damping=Float64(py=repeated_damping), scalar=scalar)


def affinity_fit_call(binding: PythonObject, stage: AffinityFit, mut result: CentroidFitResultSlot) raises:
    """The AffinityPropagation entry on the typed transport, the GIL
    released around it."""
    var connection = centroid_fit_connection(binding)
    var request = affinity_fit_request(variants_span(stage.x), stage.preference.span(), stage.n, stage.d,
        stage.precomputed, stage.mode, stage.max_iter, stage.convergence_iter, stage.seed, stage.damping,
        stage.scalar)
    try:
        with GILReleased(Python()):
            connection.driver.affinity(request, result)
    finally:
        _ = connection.provider
        _ = stage.x
        _ = stage.preference.count  # the preference words stay alive through the entry


# ---------------------------------------------------------------- MiniBatchKMeans
@fieldwise_init
struct MiniBatchFit(Movable):
    """`MiniBatchKMeans.fit(X, sample_weight)` as typed values. `x` is the
    admitted float32 X; `auxiliary` owns the explicit initial centers and the
    sample weights as float32 words and pins their admitted arrays; `init`
    and `weights` are those arrays (None when absent), pinned through the
    fit. `scan_input`: the entry refuses a non-finite X on the device."""
    var x: PythonObject
    var init: PythonObject
    var weights: PythonObject
    var auxiliary: VariantsAuxiliaryPins
    var n: Int
    var d: Int
    var k: Int
    var max_iter: Int
    var batch_size: Int
    var max_no_improvement: Int
    var init_size: Int
    var n_init: Int
    var has_init: Bool
    var seed: Int
    var init_random: Bool
    var has_weights: Bool
    var scan_input: Bool
    var tolerance: Float64
    var reassignment_ratio: Float64


def minibatch_fit_stage(model: PythonObject, source: PythonObject, sample_weight: PythonObject,
                        capabilities: Int, context: PythonObject) raises -> MiniBatchFit:
    """The reads of `MiniBatchKMeans.fit`, in its order: X, n_clusters,
    sample_weight, init, n_init, max_no_improvement, max_iter, batch_size,
    init_size, random_state, the auxiliary words, tol, reassignment_ratio.
    `capabilities` is the connected provider's: with VARIANTS_CAP_DEVSCAN the
    finite refusal of X is the entry's."""
    var devscan = (capabilities & VARIANTS_CAP_DEVSCAN) != 0
    var x: PythonObject
    if devscan:
        x = admit_family(source, 2, "X", context).array
    else:
        x = _f32(source, "X", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var k = Int(py=_int(_attr(model, "n_clusters")))
    cluster_rows_cover(k, n)
    var weights = _weights(sample_weight, n, context)
    var has_weights = weights is not None
    var init = _attr(model, "init")
    var init_arr = PythonObject(None)
    var has_init = False
    var init_random = False
    if family_is_str(init):
        if not _one_of(init, "k-means++", "random"):
            raise Error("init must be 'k-means++', 'random' or an array, got " + family_repr(init))
        init_random = _eq(init, PythonObject("random"))
    else:
        init_arr = _init_array(init, k, d, context)
        has_init = True
    var n_init_field = _attr(model, "n_init")
    var n_init: Int
    if _eq(n_init_field, PythonObject("auto")):
        n_init = minibatch_auto_n_init(has_init, init_random)
    else:
        n_init = Int(py=_int(n_init_field))
    cluster_n_init_positive(n_init)
    var mni_field = _attr(model, "max_no_improvement")
    var mni = PythonObject(-1) if mni_field is None else _int(mni_field)
    var max_iter = _int(_attr(model, "max_iter"))
    var batch_size = _int(_attr(model, "batch_size"))
    var init_size_field = _attr(model, "init_size")
    var init_size = PythonObject(0) if init_size_field is None else _int(init_size_field)
    var seed = _seed(_attr(model, "random_state"))
    var parts: List[PythonObject] = [init_arr, weights]
    var auxiliary = VariantsAuxiliaryPins(parts, context)
    var tolerance = _float(_attr(model, "tol"))
    var ratio = _float(_attr(model, "reassignment_ratio"))
    # Preserve native-word narrowing after auxiliary and float admission.
    return MiniBatchFit(x=x, init=init_arr, weights=weights, auxiliary=auxiliary^, n=n, d=d, k=k,
                        max_iter=Int(py=max_iter), batch_size=Int(py=batch_size),
                        max_no_improvement=Int(py=mni), init_size=Int(py=init_size), n_init=n_init,
                        has_init=has_init, seed=seed, init_random=init_random, has_weights=has_weights,
                        scan_input=devscan, tolerance=Float64(py=tolerance),
                        reassignment_ratio=Float64(py=ratio))


def minibatch_fit_call(connection: KMeansVariantsConnection, stage: MiniBatchFit,
                       mut result: VariantsResultSlot) raises:
    """The mini-batch entry on the typed transport, the GIL released around
    it. The entry's own non-finite refusal of X (`scan_input`) is restated
    in the public wording."""
    var source_span = variants_span(stage.x)
    var request = minibatch_fit_request(source_span, stage.auxiliary.plan.span(), stage.n, stage.d, stage.k,
        stage.max_iter, stage.batch_size, stage.max_no_improvement, stage.init_size, stage.n_init,
        stage.has_init, stage.seed, stage.init_random, stage.has_weights, stage.scan_input, stage.tolerance,
        stage.reassignment_ratio)
    try:
        with GILReleased(Python()):
            connection.driver.minibatch_fit(request, result)
    except error:
        if stage.scan_input and _pending_text(error).find("contains NaN or infinity") >= 0:
            _drop_pending()
            raise Error("mojolearn: X contains NaN or infinity")
        raise Error(String(error))
    finally:
        _ = connection.provider
        _ = len(stage.auxiliary.owners)  # the auxiliary words and their arrays outlive the entry
        _ = stage.x
        _ = stage.init
        _ = stage.weights

# SPDX-License-Identifier: Apache-2.0
"""The member boundary of the XCluster family (x_cluster estimators). Recipe:
MEMBER-ABI-V2.md; RUN: core/xcluster_member.mojo.

Published kinds: 0 MiniBatchKMeans, 1 BisectingKMeans, 2 MeanShift,
3 OPTICS (lane mf-member8: fit and export; `_optics_admit` reads
`optics_fit_stage`'s settings in its order without X, the row-count refusals
run at fit; `optics_fit`'s attributes are packed), 4 AffinityPropagation,
5 BayesianGaussianMixture (`_mojolearn_member = (25, kind)` on each class).
Settings whose value is DATA (lane mf-clu6) are
read once here and travel as the state's data words
(core/cluster_member_words.mojo): a MiniBatchKMeans array `init`, MeanShift
`seeds`, an AffinityPropagation per-sample preference (and
affinity='precomputed', a flag word), a BayesianGaussianMixture array
`mean_prior` / `covariance_prior` and the warm state a `warm_start=True`
refit resumes from. Their shape refusals, which need X's feature or row
count, run in RUN at fit in the door's order. AffinityPropagation reads `affinity_fit_stage`'s
parameters in its order and packs `affinity_fit`'s attributes. MeanShift's route also holds the binding's centroid-fit
entry in `route.second` (`meanshift_fit_stage`'s reads: seeds, bandwidth,
cluster_all, max_iter, bin_seeding, min_bin_freq; its pack: `meanshift_fit`'s
cluster_centers_, labels_, bandwidth_, n_iter_, n_features_in_).

  * `xcluster_member_admit` (UNPACK, once per member): the estimator's own
    parameter reads and refusals in its fit's order
    (bindings/cluster_fit_public_native_unpack.mojo `minibatch_fit_stage`:
    n_clusters, init, n_init ('auto' by `minibatch_auto_n_init`),
    max_no_improvement, max_iter, batch_size, init_size, random_state, tol,
    reassignment_ratio; `bisect_fit_stage`:
    init, bisecting_strategy, algorithm, n_clusters, n_init, max_iter,
    random_state, tol; the X-dependent refusals run in RUN at fit), its
    binding through the estimator's own binding route (`device=` and the
    numeric mode, `estimator_binding(model, "_mojolearn_x_cluster")`), and
    that binding's KMeans-variants entry as the route's `entry` with its
    capability word in `route.flags`.
  * `xcluster_member_pack` (PACK): the attributes the class's fit writes
    (bindings/cluster_fit_public_native_pack.mojo `minibatch_fit`:
    cluster_centers_, counts_, labels_, inertia_, n_steps_, n_iter_,
    n_features_in_; `bisect_fit`: cluster_centers_,
    _tree_centers, tree_nodes_, labels_, inertia_, n_features_in_; the
    model's resident centers released by `_set`).
  * `xcluster_member_state` (UNPACK): a member state from a fitted model.

SERIAL BY NATURE: one member's admission or publication (a few attribute
reads and block copies); a caller runs its members in parallel.
"""
from std.memory import bitcast, unsafe_memcpy
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family
from bindings.classical_arrays_door import classical_address, classical_c_bytes
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.cluster_fit_public_native_unpack import _attr, _int, _float, _eq, _one_of, _seed, _set, _truthy
from bindings.cluster_fit_public_native_unpack import _optics_metric
from bindings.optics_fit_boundary_door import optics_fit_connection
from core.optics_fit_api import OPTICS_FIT_COUNT
from cluster.cluster_fit_door_run import optics_eps_refused, optics_xi_range
from bindings.cluster_centroid_fit_boundary_door import centroid_fit_connection
from bindings.native_loaded_module_door import native_loaded_module
from core.cluster_centroid_fit_api import CENTROID_FIT_COUNT
from bindings.family_common_pack import family_array
from bindings.family_common_unpack import family_repr, family_is_str, family_str, family_isinstance
from bindings.kmeans_variants_boundary_door import variants_connection
from bindings.member_owner_unpack import MemberOwner
from bindings.native_buffer_address_unpack import native_buffer_address
from bindings.py_balanced import py_call, py_get
from cluster.cluster_fit_door_run import (
    cluster_n_init_positive, minibatch_auto_n_init, meanshift_bandwidth_refused, affinity_damping_refused,
    affinity_iterations_positive,
)
from cluster.cluster_door_run import cluster_node_words
from core.device_route import DeviceRoute, DEVICE_CPU, DEVICE_GPU, device_vendor_code
from core.entry_table import EntryRef, entry_ref_word
from core.hostptr import f32_ptr
from core.kmeans_variants_api import VARIANTS_MODE, KMEANS_VARIANTS_COUNT
from core.bgmm_native_api import BGMM_NATIVE_COUNT
from bindings.boundary_read_door import read_int_like, read_float_like, read_bool
from bindings.mixture_public_native_unpack import _bgmm_connection, _any_f32, _x_finite, _get_or, bgmm_score_words
from bindings.affinity_preference_boundary_door import affinity_preference_prepare
from core.affinity_preference_plan import AffinityPreferencePlan
from core.cluster_member_words import member_words_append_f32
from core.host_finite_scan import host_all_finite_f32
from core.member_families import MEMBER_FAMILY_XCLUSTER
from bindings.cluster_side_records_unpack import cluster_record_drop
from bindings.mixture_public_native_unpack import _GMM_ADMIT_SLOT, _GMM_OPERANDS_SLOT
from mixture.mixture_door_run import bgmm_count_at_least_one, mixture_by_type_shape
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_new, member_free,
    MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_IMPORT_DONE,
    MEMBER_F64, MEMBER_I32,
)
from providers.python.estimator_binding_pyapi_door import estimator_binding

comptime _KIND_MINIBATCH = 0
comptime _KIND_BISECT = 1
comptime _KIND_MEANSHIFT = 2
comptime _KIND_OPTICS = 3
comptime _KIND_AFFINITY = 4
comptime _KIND_BGMM = 5
comptime _BLOCK_BGMM = MEMBER_BLOCK_FAMILY + 8
comptime _BGMM_SCALAR_WORDS = 6
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]


def _variants_route(model: PythonObject, context: PythonObject, mut owner: MemberOwner, centroid: Bool = False) raises:
    """The estimator's x_cluster binding and its variants entry -> the route;
    `centroid`: also its centroid-fit entry -> `route.second`."""
    var binding = estimator_binding(model, "_mojolearn_x_cluster", context)
    var connection = variants_connection(binding)
    var mode = connection.driver.metadata(VARIANTS_MODE)
    var vendor = device_vendor_code(String(py_call(binding.x_cluster_vendor)))
    if vendor < 0:
        raise Error("ImportError: mojolearn: the x_cluster binding names no vendor")
    var route = DeviceRoute(DEVICE_CPU if vendor == 0 else DEVICE_GPU)
    route.mode = mode
    route.vendor = vendor
    route.entry = EntryRef(connection.driver.entry, entry_ref_word(mode, vendor, KMEANS_VARIANTS_COUNT))
    route.flags = connection.capabilities
    if centroid:
        var fit = centroid_fit_connection(binding)
        route.second = EntryRef(fit.driver.entry, entry_ref_word(mode, vendor, CENTROID_FIT_COUNT))
        route.has_second = True
        owner.plan.route = route
        owner.pins = Python.tuple(binding, connection.provider, fit.provider)
        return
    owner.plan.route = route
    owner.pins = Python.tuple(binding, connection.provider)


def _data_words(array: PythonObject, mut words: List[Float64]) raises:
    """An admitted float32 C-order 2-D array as data words: rows, columns,
    then its words packed (one copy)."""
    var rows = Int(py=array.shape[0])
    var cols = Int(py=array.shape[1])
    words.append(Float64(rows))
    words.append(Float64(cols))
    member_words_append_f32(words, Int(py=classical_address(array)), rows * cols)
    _ = array  # the admitted words stay alive through the copy above


def _init_admitted(init: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_init_array`'s callback refusal and admission (2-D, float32); its
    finiteness and shape refusals run in RUN at fit, in its order."""
    ref cpy = Python().cpython()
    if cpy.lib.get_function[Int32]("PyCallable_Check")(init._obj_ptr) != 0:
        raise Error("TypeError: Python init callbacks are not accepted; pass explicit initial centers or use 'k-means++' or 'random'")
    return admit_family(init, 2, "init", context).array


def _minibatch_admit(model: PythonObject, context: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """`minibatch_fit_stage`'s reads, in its order, without X and weights;
    explicit starting centers as data words."""
    var k = Int(py=_int(_attr(model, "n_clusters")))
    if k < 1:
        raise Error("ValueError: n_clusters must be at least 1, got " + String(k))
    var init = _attr(model, "init")
    var has_init = not family_is_str(init)
    var init_random = False
    var start = List[Float64]()
    if has_init:
        _data_words(_init_admitted(init, context), start)
    else:
        if not _one_of(init, "k-means++", "random"):
            raise Error("init must be 'k-means++', 'random' or an array, got " + family_repr(init))
        init_random = _eq(init, PythonObject("random"))
    var n_init_field = _attr(model, "n_init")
    var n_init: Int
    if _eq(n_init_field, PythonObject("auto")):
        n_init = minibatch_auto_n_init(has_init, init_random)
    else:
        n_init = Int(py=_int(n_init_field))
    cluster_n_init_positive(n_init)
    var mni_field = _attr(model, "max_no_improvement")
    var mni = -1 if mni_field is None else Int(py=_int(mni_field))
    var max_iter = Int(py=_int(_attr(model, "max_iter")))
    var batch_size = Int(py=_int(_attr(model, "batch_size")))
    var init_size_field = _attr(model, "init_size")
    var init_size = 0 if init_size_field is None else Int(py=_int(init_size_field))
    var seed = _seed(_attr(model, "random_state"))
    var tolerance = Float64(py=_float(_attr(model, "tol")))
    var ratio = Float64(py=_float(_attr(model, "reassignment_ratio")))
    _variants_route(model, context, owner)
    var words: List[Float64] = [Float64(_KIND_MINIBATCH), Float64(k), 1.0 if init_random else 0.0,
        Float64(n_init), Float64(max_iter), tolerance, bitcast[DType.float64](Int64(seed)), Float64(batch_size),
        Float64(mni), Float64(init_size), ratio]
    words.extend(start^)
    owner.plan.params = words^
    return True


def _meanshift_admit(model: PythonObject, context: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """`meanshift_fit_stage`'s reads, in its order, without X; explicit
    seeds as data words (the door's `_f32` admission; its finiteness and
    feature-count refusals run in RUN at fit)."""
    var seeds = List[Float64]()
    var seeds_field = _attr(model, "seeds")
    if seeds_field is not None:
        _data_words(admit_family(seeds_field, 2, "seeds", context).array, seeds)
    var bw = Float64(0.0)
    var bandwidth = _attr(model, "bandwidth")
    if bandwidth is not None:
        bw = Float64(py=_float(bandwidth))
        if meanshift_bandwidth_refused(bw):
            raise Error("bandwidth needs to be greater than zero or None, got "
                        + String(py=py_call(native_loaded_module("builtins").format, PythonObject(bw),
                                            PythonObject("f"))))
    var cluster_all = _truthy(_attr(model, "cluster_all"))
    var max_iter = Int(py=_int(_attr(model, "max_iter")))
    var bin_seeding = _truthy(_attr(model, "bin_seeding"))
    var min_bin_freq = Int(py=_int(_attr(model, "min_bin_freq")))
    _variants_route(model, context, owner, True)
    var words: List[Float64] = [Float64(_KIND_MEANSHIFT), bw, 1.0 if cluster_all else 0.0, Float64(max_iter),
        1.0 if bin_seeding else 0.0, Float64(min_bin_freq)]
    words.extend(seeds^)
    owner.plan.params = words^
    return True


def _optics_size_words(value: PythonObject, name: String, mut words: List[Float64]) raises:
    """`OPTICS._size`'s reads without the row count (bindings/
    cluster_fit_public_native_unpack.mojo `_optics_size`): its refusals that
    need no n, in its words, then (mode, value, float word) for the fit."""
    var b = native_loaded_module("builtins")
    var op = native_loaded_module("operator")
    if family_isinstance(value, b.bool) or value is None:
        raise Error(name + " must be an int >= 2 or a float in (0, 1]")
    var is_float = family_isinstance(value, b.float)
    if family_isinstance(value, b.int) or (is_float and _truthy(py_call(op.gt, value, PythonObject(1)))):
        # A count: its range against n is checked at fit, in these words.
        words.append(0.0)
        words.append(Float64(py=_float(value)))
        words.append(1.0 if is_float else 0.0)
        return
    if not (_truthy(py_call(op.lt, PythonObject(0), value)) and _truthy(py_call(op.le, value, PythonObject(1)))):
        raise Error(name + " must be in (0, 1], got " + family_str(value))
    words.append(1.0)
    words.append(Float64(py=_float(value)))
    words.append(1.0)


def _optics_admit(model: PythonObject, context: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """Lane mf-member8: `optics_fit_stage`'s reads, in its order, without X
    (metric, cluster_method, the two sizes, max_eps, eps, xi,
    predecessor_correction, the second xi read); the row-count refusals and
    the precomputed square check run at fit. The binding's OPTICS entry is
    `route.third`."""
    var code = _optics_metric(model, context)
    var method = _attr(model, "cluster_method")
    if not _one_of(method, "xi", "dbscan"):
        raise Error("cluster_method must be 'xi' or 'dbscan', got " + family_repr(method))
    var words: List[Float64] = [Float64(_KIND_OPTICS)]
    _optics_size_words(_attr(model, "min_samples"), "min_samples", words)
    var mcs_field = _attr(model, "min_cluster_size")
    if mcs_field is None:
        words.append(-1.0)
        words.append(0.0)
        words.append(0.0)
    else:
        _optics_size_words(mcs_field, "min_cluster_size", words)
    var max_eps = _float(_attr(model, "max_eps"))
    var eps_field = _attr(model, "eps")
    var eps = max_eps if eps_field is None else _float(eps_field)
    var is_xi = _eq(method, PythonObject("xi"))
    if optics_eps_refused(_eq(method, PythonObject("dbscan")), Float64(py=eps), Float64(py=max_eps)):
        raise Error("Specify an epsilon smaller than " + family_str(max_eps) + ". Got " + family_str(eps) + ".")
    optics_xi_range(Float64(py=_float(_attr(model, "xi"))))
    var predecessor_correction = _truthy(_attr(model, "predecessor_correction"))
    var repeated_xi = Float64(py=_float(_attr(model, "xi")))
    words.append(0.0 if is_xi else 1.0)
    words.append(1.0 if predecessor_correction else 0.0)
    words.append(Float64(code.code))
    words.append(code.power)
    words.append(Float64(py=max_eps))
    words.append(repeated_xi)
    words.append(Float64(py=eps))
    _variants_route(model, context, owner)
    var binding = owner.pins[0]
    var optics = optics_fit_connection(binding)
    owner.plan.route.third = EntryRef(optics.driver.entry, entry_ref_word(owner.plan.route.mode,
        owner.plan.route.vendor, OPTICS_FIT_COUNT))
    owner.plan.route.has_third = True
    owner.pins = Python.tuple(owner.pins[0], owner.pins[1], optics.provider)
    owner.plan.params = words^
    return True


def _affinity_admit(model: PythonObject, context: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """`affinity_fit_stage`'s reads, in its order, without X;
    affinity='precomputed' as a flag word and a per-sample preference as
    data words (`affinity_preference_prepare`, read once; the square-matrix
    and one-per-sample refusals run in RUN at fit)."""
    var affinity = _attr(model, "affinity")
    if not _one_of(affinity, "euclidean", "precomputed"):
        raise Error("affinity must be 'euclidean' or 'precomputed', got " + family_repr(affinity))
    var damping_field = _attr(model, "damping")
    if affinity_damping_refused(Float64(py=_float(damping_field))):
        raise Error("damping must be in [0.5, 1), got " + family_str(damping_field))
    affinity_iterations_positive(Int(py=_int(_attr(model, "max_iter"))))
    affinity_iterations_positive(Int(py=_int(_attr(model, "convergence_iter"))))
    var precomputed = _eq(affinity, PythonObject("precomputed"))
    var mode = 0
    var scalar = Float64(0.0)
    var per_sample = AffinityPreferencePlan()
    var preference = _attr(model, "preference")
    if preference is not None:
        var builtins = native_loaded_module("builtins")
        if family_isinstance(preference, Python.tuple(builtins.int, builtins.float)):
            mode = 1
            scalar = Float64(py=_float(preference))
        else:
            per_sample = affinity_preference_prepare(preference, context)
            mode = 2
    var max_iter = Int(py=_int(_attr(model, "max_iter")))
    var convergence = Int(py=_int(_attr(model, "convergence_iter")))
    var seed = _seed(_attr(model, "random_state"))
    var damping = Float64(py=_float(damping_field))
    _variants_route(model, context, owner, True)
    var words: List[Float64] = [Float64(_KIND_AFFINITY), damping, Float64(max_iter), Float64(convergence),
        bitcast[DType.float64](Int64(seed)), Float64(mode), scalar]
    if precomputed or mode == 2:
        words.append(1.0 if precomputed else 0.0)
        var span = per_sample.span()
        words.append(Float64(span.count))
        member_words_append_f32(words, span.address, span.count)
    _ = per_sample^  # the preference words stay alive through the copy above
    owner.plan.params = words^
    return True


def _choice(value: PythonObject, names: List[String], what: String) raises -> Int:
    """The code of `value` in `names` (the estimator's own option table:
    python/mojolearn/mixture.py family_mix_ctypes / family_mix_inits), or the
    door's refusal with the sorted names."""
    for i in range(len(names)):  # driver-loop: at most four option names
        if family_is_str(value) and _eq(value, PythonObject(names[i])):
            return i
    var listed = String("")
    var order = List[Int]()
    for i in range(len(names)):  # driver-loop: at most four option names, sorted
        var at = 0
        while at < len(order) and names[order[at]] < names[i]:
            at += 1
        order.insert(at, i)
    for i in range(len(order)):  # driver-loop: at most four option names
        listed += (", " if i else "") + "'" + names[order[i]] + "'"
    raise Error(what + " must be one of [" + listed + "], got " + family_repr(value))


def _bgmm_admit(model: PythonObject, context: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """`bgmm_fit_stage`'s reads and refusals, in its order, without X (the
    rows-cover, prior-shape and warm-state feature-count checks run in RUN
    at fit). An array mean_prior or covariance_prior and the warm state of
    a `warm_start=True` refit are data words (lane mf-clu6)."""
    var ctypes: List[String] = ["full", "tied", "diag", "spherical"]
    var inits: List[String] = ["kmeans", "random", "k-means++", "random_from_data"]
    var ct = _choice(_attr(model, "covariance_type"), ctypes, "covariance_type")
    var init_code = _choice(_attr(model, "init_params"), inits, "init_params")
    var prior_type = _attr(model, "weight_concentration_prior_type")
    if not _one_of(prior_type, "dirichlet_process", "dirichlet_distribution"):
        raise Error("weight_concentration_prior_type must be 'dirichlet_process' or "
                    + "'dirichlet_distribution'")
    var process = _eq(prior_type, PythonObject("dirichlet_process"))
    var k = read_int_like(_attr(model, "n_components"))
    bgmm_count_at_least_one(read_int_like(_attr(model, "n_init")))
    bgmm_count_at_least_one(read_int_like(_attr(model, "max_iter")))
    var data = _bgmm_data(model, k, ct, context)
    var wcp = _attr(model, "weight_concentration_prior")
    var mpp = _attr(model, "mean_precision_prior")
    var dfp = _attr(model, "degrees_of_freedom_prior")
    var prior_weight = -1.0 if wcp is None else read_float_like(wcp)
    var prior_mean_precision = -1.0 if mpp is None else read_float_like(mpp)
    var prior_freedom = -1.0 if dfp is None else read_float_like(dfp)
    var reg_covar = read_float_like(_attr(model, "reg_covar"))
    var tol = read_float_like(_attr(model, "tol"))
    var max_iter = read_int_like(_attr(model, "max_iter"))
    var n_init = read_int_like(_attr(model, "n_init"))
    var seed = _seed(_attr(model, "random_state"))
    _variants_route(model, context, owner)
    # The binding's typed BGMM entry (`bgmm_native_api_v1`) -> route.third.
    var binding = owner.pins[0]
    var mixture = _bgmm_connection(binding)
    owner.plan.route.third = EntryRef(mixture.driver.entry,
        entry_ref_word(owner.plan.route.mode, owner.plan.route.vendor, BGMM_NATIVE_COUNT))
    owner.plan.route.has_third = True
    owner.pins = Python.tuple(owner.pins[0], owner.pins[1], mixture.provider)
    var words: List[Float64] = [Float64(_KIND_BGMM), Float64(k), Float64(ct), Float64(init_code),
        1.0 if process else 0.0, prior_weight, prior_mean_precision, prior_freedom, reg_covar, tol,
        Float64(max_iter), Float64(n_init), bitcast[DType.float64](Int64(seed))]
    words.extend(data^)
    owner.plan.params = words^
    return True


def _bgmm_data(model: PythonObject, k: Int, ct: Int, context: PythonObject) raises -> List[Float64]:
    """The data words of a BayesianGaussianMixture (core/xcluster_member.mojo
    kind 5, words 13 ..): the door's admissions of `mean_prior` (`_any_f32`)
    and `covariance_prior` (`_x_finite` for full / tied, `_any_f32` for diag,
    a float for spherical), and the warm state `bgmm_fit_stage` resumes
    from (`_warm` with the same n_components and covariance_type; its
    feature count is checked at fit). Empty when none is present."""
    var words = List[Float64]()
    var mean = PythonObject(None)
    var mean_source = _attr(model, "mean_prior")
    if mean_source is not None:
        mean = _any_f32(mean_source, "mean_prior", context)
    var cov = PythonObject(None)
    var cov_mode = 0
    var cov_rows = 0
    var cov_cols = 0
    var spherical = List[Float32]()
    var cov_source = _attr(model, "covariance_prior")
    if cov_source is not None:
        if ct == 0 or ct == 1:
            cov = _x_finite(cov_source, "covariance_prior", context)
            cov_mode = 1
            cov_rows = Int(py=cov.shape[0])
            cov_cols = Int(py=cov.shape[1])
        elif ct == 2:
            cov = _any_f32(cov_source, "covariance_prior", context)
            cov_mode = 2
            cov_rows = Int(py=cov.size)
            cov_cols = 1
        else:
            spherical.append(Float32(read_float_like(cov_source)))
            cov_mode = 3
            cov_rows = 1
            cov_cols = 1
    var warm = PythonObject(None)
    var warm_d = 0
    var lower = 0.0
    if read_bool(_attr(model, "warm_start")):
        var prev = _get_or(model, "_warm", PythonObject(None))
        if prev is not None:
            var operator = native_loaded_module("operator")
            var key = prev[0]
            if (Bool(py=py_call(operator.eq, key[0], PythonObject(k)))
                    and Bool(py=py_call(operator.eq, key[2], _attr(model, "covariance_type")))):
                warm = _any_f32(prev[1], "aux", context)
                warm_d = Int(py=key[1])
                lower = read_float_like(_get_or(model, "lower_bound_", PythonObject(0.0)))
    if mean is None and cov_mode == 0 and warm is None:
        return words^
    var mean_size = -1 if mean is None else Int(py=mean.size)
    var warm_size = 0 if warm is None else Int(py=warm.size)
    # [21 ..] the GaussianMixture words (core/xcluster_member.mojo): not plain, no starts.
    var head: List[Float64] = [Float64(mean_size), Float64(cov_mode), Float64(cov_rows), Float64(cov_cols),
        0.0 if warm is None else 1.0, Float64(warm_d), lower, Float64(warm_size), 0.0, -1.0, -1.0, 0.0, -1.0, 1.0]
    words = head^
    if mean_size > 0:
        member_words_append_f32(words, Int(py=classical_address(mean)), mean_size)
    if cov_mode == 3:
        member_words_append_f32(words, Int(spherical.unsafe_ptr()), 1)
    elif cov_mode > 0:
        member_words_append_f32(words, Int(py=classical_address(cov)), cov_rows * cov_cols)
    if warm_size > 0:
        member_words_append_f32(words, Int(py=classical_address(warm)), warm_size)
    # the admitted words stay alive through the copies above
    _ = mean
    _ = cov
    _ = warm
    _ = len(spherical)
    return words^


def xcluster_gmm_plain_admit(model: PythonObject, context: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """A GaussianMixture whose options need the option-parity fit
    (`_gmm_needs_ext`), admitted as this family's kind 5 in plain mode (lane
    mf-clu6): `gmm_ext_fit_stage`'s reads and refusals without X, in its
    order (covariance_type, init_params, n_components, the warm state of
    `_ext` with the same n_components and covariance_type, the weights,
    means and precisions starts), then the floats and ints the entry takes.
    The starts' shape checks against X run in RUN at fit. Its family and
    kind become MEMBER_FAMILY_XCLUSTER / 5, so the caller resolves this
    family's entry and PACK publishes what `_gmm_ext_fit` publishes."""
    classical_connect_numpy(context)
    var ctypes: List[String] = ["full", "tied", "diag", "spherical"]
    var inits: List[String] = ["kmeans", "random", "k-means++", "random_from_data"]
    var covariance_type = _attr(model, "covariance_type")
    var ct = _choice(covariance_type, ctypes, "covariance_type")
    var init_code = _choice(_attr(model, "init_params"), inits, "init_params")
    var k = read_int_like(_attr(model, "n_components"))
    var warm = PythonObject(None)
    var warm_d = 0
    var lower = 0.0
    var prev = _get_or(model, "_ext", PythonObject(None))
    if read_bool(_attr(model, "warm_start")) and prev is not None:
        var operator = native_loaded_module("operator")
        var key = py_get(prev, PythonObject("key"), PythonObject(None))
        if (key is not None and Bool(py=py_call(operator.eq, key[0], PythonObject(k)))
                and Bool(py=py_call(operator.eq, key[2], covariance_type))):
            warm = _any_f32(py_get(prev, PythonObject("state"), PythonObject(None)), "aux", context)
            warm_d = Int(py=key[1])
            lower = read_float_like(_get_or(model, "lower_bound_", PythonObject(0.0)))
    var weights = PythonObject(None)
    var weights_source = _attr(model, "weights_init")
    if weights_source is not None:
        weights = _any_f32(weights_source, "weights_init", context)
    var means = PythonObject(None)
    var means_source = _attr(model, "means_init")
    if means_source is not None:
        means = _x_finite(means_source, "means_init", context)
    var precisions = PythonObject(None)
    var precisions_finite = True
    var precisions_source = _attr(model, "precisions_init")
    if precisions_source is not None:
        precisions = _any_f32(precisions_source, "precisions_init", context)
        precisions_finite = host_all_finite_f32(f32_ptr(Int(py=classical_address(precisions))),
            Int(py=precisions.size))
    var reg_covar = read_float_like(_attr(model, "reg_covar"))
    var tol = read_float_like(_attr(model, "tol"))
    var max_iter = read_int_like(_attr(model, "max_iter"))
    var n_init = read_int_like(_attr(model, "n_init"))
    var seed = _seed(_attr(model, "random_state"))
    _variants_route(model, context, owner)
    # The binding's typed BGMM entry (`bgmm_native_api_v1`) -> route.third.
    var binding = owner.pins[0]
    var mixture = _bgmm_connection(binding)
    owner.plan.route.third = EntryRef(mixture.driver.entry,
        entry_ref_word(owner.plan.route.mode, owner.plan.route.vendor, BGMM_NATIVE_COUNT))
    owner.plan.route.has_third = True
    owner.pins = Python.tuple(owner.pins[0], owner.pins[1], mixture.provider)
    var warm_size = 0 if warm is None else Int(py=warm.size)
    var weights_size = -1 if weights is None else Int(py=weights.size)
    var means_rows = -1 if means is None else Int(py=means.shape[0])
    var means_cols = 0 if means is None else Int(py=means.shape[1])
    var precisions_size = -1 if precisions is None else Int(py=precisions.size)
    var words: List[Float64] = [Float64(_KIND_BGMM), Float64(k), Float64(ct), Float64(init_code), 0.0,
        -1.0, -1.0, -1.0, reg_covar, tol, Float64(max_iter), Float64(n_init), bitcast[DType.float64](Int64(seed)),
        -1.0, 0.0, 0.0, 0.0, 0.0 if warm is None else 1.0, Float64(warm_d), lower, Float64(warm_size),
        1.0, Float64(weights_size), Float64(means_rows), Float64(means_cols), Float64(precisions_size),
        1.0 if precisions_finite else 0.0]
    if warm_size > 0:
        member_words_append_f32(words, Int(py=classical_address(warm)), warm_size)
    if weights_size > 0:
        member_words_append_f32(words, Int(py=classical_address(weights)), weights_size)
    if means_rows > 0 and means_cols > 0:
        member_words_append_f32(words, Int(py=classical_address(means)), means_rows * means_cols)
    if precisions_size > 0:
        member_words_append_f32(words, Int(py=classical_address(precisions)), precisions_size)
    # the admitted words stay alive through the copies above
    _ = warm
    _ = weights
    _ = means
    _ = precisions
    owner.plan.params = words^
    owner.plan.family = MEMBER_FAMILY_XCLUSTER
    owner.plan.kind = _KIND_BGMM
    return True


def _bisect_admit(model: PythonObject, context: PythonObject, mut owner: MemberOwner) raises:
    """`bisect_fit_stage`'s reads, in its order, without X."""
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
    var k = Int(py=_int(_attr(model, "n_clusters")))
    if k < 1:
        raise Error("ValueError: n_clusters must be at least 1, got " + String(k))
    var n_init = Int(py=_int(_attr(model, "n_init")))
    cluster_n_init_positive(n_init)
    var random_init = _eq(init, PythonObject("random"))
    var max_iter = Int(py=_int(_attr(model, "max_iter")))
    var seed = _seed(_attr(model, "random_state"))
    var largest_cluster = _eq(strategy, PythonObject("largest_cluster"))
    var tolerance = Float64(py=_float(_attr(model, "tol")))
    _variants_route(model, context, owner)
    var words: List[Float64] = [Float64(_KIND_BISECT), Float64(k), 0.0 if random_init else 1.0, Float64(n_init),
        Float64(max_iter), tolerance, bitcast[DType.float64](Int64(seed)), 1.0 if largest_cluster else 0.0]
    owner.plan.params = words^


def xcluster_member_admit(model: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
    b: PythonObject, mut owner: MemberOwner) raises -> Bool:
    """UNPACK: fill `owner.plan.route`, `owner.plan.params` and `owner.pins`
    from the unfitted estimator: True, or the class's own refusal (an
    unmarked kind is the member's own NEW refusal)."""
    if kind == _KIND_MINIBATCH:
        classical_connect_numpy(context)
        return _minibatch_admit(model, context, owner)
    if kind == _KIND_MEANSHIFT:
        classical_connect_numpy(context)
        return _meanshift_admit(model, context, owner)
    if kind == _KIND_AFFINITY:
        classical_connect_numpy(context)
        return _affinity_admit(model, context, owner)
    if kind == _KIND_OPTICS:
        classical_connect_numpy(context)
        return _optics_admit(model, context, owner)
    if kind == _KIND_BGMM:
        classical_connect_numpy(context)
        return _bgmm_admit(model, context, owner)
    if kind != _KIND_BISECT:
        # Kinds 0 to 5 are marked (python/mojolearn/_expansion_cluster.py);
        # any other mark is the member's own NEW refusal, raised here (lane mf-clu6).
        raise Error("member ABI: x_cluster member kind " + String(kind) + " is not published")
    classical_connect_numpy(context)
    _bisect_admit(model, context, owner)
    return True


def _export(plan: MemberPlan, mut call: MemberCall, block: Int) raises:
    call.block = block
    call.flags = 0
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)


def _f32_array(view: MemberMatrix, context: PythonObject) raises -> PythonObject:
    return _f32_shaped(view, Python.tuple(view.rows, view.cols), context)


def _f32_shaped(view: MemberMatrix, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    var array = family_array(shape, "<f4", context)
    if view.rows * view.cols > 0:
        unsafe_memcpy(dest=f32_ptr(Int(py=classical_address(array))), src=f32_ptr(view.data),
            count=view.rows * view.cols)
    return array


def _i32_array(view: MemberMatrix, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    var array = family_array(shape, "<i4", context)
    if view.rows * view.cols > 0:
        unsafe_memcpy(dest=_I32P(unsafe_from_address=Int(py=classical_address(array))),
            src=_I32P(unsafe_from_address=view.data), count=view.rows * view.cols)
    return array


def _shape(values: List[Int]) raises -> PythonObject:
    var shape = Python.list()
    for i in range(len(values)):  # driver-loop: one to three axis lengths
        _ = py_call(shape.append, PythonObject(values[i]))
    return py_call(native_loaded_module("builtins").tuple, shape)


def _bgmm_pack(model: PythonObject, context: PythonObject, plan: MemberPlan, mut call: MemberCall) raises:
    """`_bgmm_fit`'s publications (bindings/mixture_public_native_pack.mojo),
    the same names, dtypes, shapes and order, from the state's blocks."""
    var k = Int(plan.params[1])
    var ct = Int(plan.params[2])
    var process = plan.params[4] != 0.0
    _export(plan, call, _BLOCK_BGMM + 1)
    var d = call.dst.cols
    var means = _f32_array(call.dst, context)
    _export(plan, call, _BLOCK_BGMM + 0)
    _set(model, "weights_", _f32_shaped(call.dst, Python.tuple(k), context))
    _set(model, "means_", means)
    _export(plan, call, _BLOCK_BGMM + 3)
    var pchol = _f32_shaped(call.dst, Python.tuple(k, d, d), context)
    _set(model, "_full_pchol", pchol)
    _export(plan, call, _BLOCK_BGMM + 13)
    _set(model, "covariances_", _f32_shaped(call.dst, _shape(mixture_by_type_shape(ct, k, d)), context))
    _export(plan, call, _BLOCK_BGMM + 14)
    _set(model, "precisions_cholesky_", _f32_shaped(call.dst, _shape(mixture_by_type_shape(ct, k, d)), context))
    _export(plan, call, _BLOCK_BGMM + 4)
    var wc0 = _f32_shaped(call.dst, Python.tuple(k), context)
    if process:
        _export(plan, call, _BLOCK_BGMM + 5)
        _set(model, "weight_concentration_", Python.tuple(wc0, _f32_shaped(call.dst, Python.tuple(k), context)))
    else:
        _set(model, "weight_concentration_", wc0)
    _export(plan, call, _BLOCK_BGMM + 6)
    _set(model, "mean_precision_", _f32_shaped(call.dst, Python.tuple(k), context))
    _export(plan, call, _BLOCK_BGMM + 7)
    if ct == 1:
        _set(model, "degrees_of_freedom_", PythonObject(Float64(f32_ptr(call.dst.data)[0])))
    else:
        _set(model, "degrees_of_freedom_", _f32_shaped(call.dst, Python.tuple(k), context))
    _export(plan, call, _BLOCK_BGMM + 8)
    var consts = _f32_shaped(call.dst, Python.tuple(k), context)
    _set(model, "_log_consts", consts)
    # the class's score door reads these words (laid out once per state)
    _set(model, "_score_words", bgmm_score_words(means, pchol, consts, context))
    _export(plan, call, _BLOCK_BGMM + 9)
    _set(model, "mean_prior_", _f32_shaped(call.dst, Python.tuple(d), context))
    _export(plan, call, _BLOCK_BGMM + 15)
    if ct == 2:
        _set(model, "covariance_prior_", _f32_shaped(call.dst, Python.tuple(d), context))
    elif ct == 3:
        _set(model, "covariance_prior_", PythonObject(Float64(f32_ptr(call.dst.data)[0])))
    else:
        _set(model, "covariance_prior_", _f32_shaped(call.dst, Python.tuple(d, d), context))
    _export(plan, call, MEMBER_BLOCK_SCALARS)
    var s = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var lower = s[0]
    var iterations = Int(s[1])
    var converged = s[2] != 0.0
    var prior_weight = s[3]
    var prior_mean_precision = s[4]
    var prior_freedom = s[5]
    _set(model, "lower_bound_", PythonObject(lower))
    _set(model, "n_iter_", PythonObject(iterations))
    _set(model, "converged_", PythonObject(converged))
    # The state a warm_start refit resumes from: blocks [11, 4, 5, 6, 1, 7,
    # 2, 3] end to end, an empty block 5 (Dirichlet distribution) as k
    # zeros, the bytes `_state` lays out (one allocation, one copy a block).
    var order: List[Int] = [11, 4, 5, 6, 1, 7, 2, 3]
    var sizes = List[Int](length=len(order), fill=0)
    var total = 0
    for j in range(len(order)):  # driver-loop: the eight warm-state blocks
        _export(plan, call, _BLOCK_BGMM + order[j])
        sizes[j] = call.dst.rows * call.dst.cols
        if order[j] == 5 and sizes[j] == 0:
            sizes[j] = -k
        total += sizes[j] if sizes[j] >= 0 else k
    var warm = family_array(Python.tuple(total), "<f4", context)
    var at = Int(py=classical_address(warm))
    for j in range(len(order)):  # driver-loop: the eight warm-state blocks
        if sizes[j] < 0:
            var zero = f32_ptr(at)
            for t in range(k):  # driver-loop: k zero words of an absent block
                zero[t] = 0.0
            at += 4 * k
            continue
        _export(plan, call, _BLOCK_BGMM + order[j])
        if sizes[j] > 0:
            unsafe_memcpy(dest=f32_ptr(at), src=f32_ptr(call.dst.data), count=sizes[j])
        at += 4 * sizes[j]
    var key = Python.tuple(PythonObject(k), PythonObject(d), _attr(model, "covariance_type"))
    _set(model, "_warm", Python.tuple(key, warm))
    _set(model, "weight_concentration_prior_", PythonObject(prior_weight))
    _set(model, "mean_precision_prior_", PythonObject(prior_mean_precision))
    _set(model, "degrees_of_freedom_prior_", PythonObject(prior_freedom))
    _set(model, "n_features_in_", PythonObject(d))
    _ = means
    _ = warm


def _plain(plan: MemberPlan) -> Bool:
    """A GaussianMixture's words (`xcluster_gmm_plain_admit`)."""
    return len(plan.params) > 21 and plan.params[21] != 0.0


def _gmm_plain_pack(model: PythonObject, context: PythonObject, plan: MemberPlan, mut call: MemberCall) raises:
    """`_gmm_ext_fit`'s publications (bindings/mixture_public_native_pack.mojo),
    the same names, dtypes, shapes and order, from the state's blocks, after
    dropping the model's admission and operand records as the class's own
    fit does."""
    cluster_record_drop(_GMM_ADMIT_SLOT, model)
    cluster_record_drop(_GMM_OPERANDS_SLOT, model)
    var k = Int(plan.params[1])
    var ct = Int(plan.params[2])
    _export(plan, call, _BLOCK_BGMM + 1)
    var d = call.dst.cols
    var means = _f32_shaped(call.dst, Python.tuple(k, d), context)
    _export(plan, call, _BLOCK_BGMM + 0)
    var weights = _f32_shaped(call.dst, Python.tuple(k), context)
    _set(model, "n_features_in_", PythonObject(d))
    _set(model, "weights_", weights)
    _set(model, "means_", means)
    _export(plan, call, _BLOCK_BGMM + 13)
    _set(model, "covariances_", _f32_shaped(call.dst, _shape(mixture_by_type_shape(ct, k, d)), context))
    _export(plan, call, _BLOCK_BGMM + 14)
    _set(model, "precisions_cholesky_", _f32_shaped(call.dst, _shape(mixture_by_type_shape(ct, k, d)), context))
    _export(plan, call, _BLOCK_BGMM + 12)
    _set(model, "log_det_chol_", _f32_shaped(call.dst, Python.tuple(k), context))
    _export(plan, call, MEMBER_BLOCK_SCALARS)
    var s = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var lower = s[0]
    var iterations = Int(s[1])
    var converged = s[2] != 0.0
    _set(model, "n_iter_", PythonObject(iterations))
    _set(model, "converged_", PythonObject(converged))
    _set(model, "lower_bound_", PythonObject(lower))
    _export(plan, call, _BLOCK_BGMM + 3)
    var pchol = _f32_shaped(call.dst, Python.tuple(k, d, d), context)
    _export(plan, call, _BLOCK_BGMM + 8)
    var consts = _f32_shaped(call.dst, Python.tuple(k), context)
    # The state a warm_start refit resumes from (`_state` with order
    # [11, 4, -1, 6, 1, 7, 2, 3]: -1 is k zeros), one allocation, one copy a block.
    var order: List[Int] = [11, 4, -1, 6, 1, 7, 2, 3]
    var sizes = List[Int](length=len(order), fill=0)
    var total = 0
    for j in range(len(order)):  # driver-loop: the eight warm-state blocks
        if order[j] < 0:
            sizes[j] = -k
            total += k
            continue
        _export(plan, call, _BLOCK_BGMM + order[j])
        sizes[j] = call.dst.rows * call.dst.cols
        total += sizes[j]
    var warm = family_array(Python.tuple(total), "<f4", context)
    var at = Int(py=classical_address(warm))
    for j in range(len(order)):  # driver-loop: the eight warm-state blocks
        if sizes[j] < 0:
            var zero = f32_ptr(at)
            for t in range(k):  # driver-loop: k zero words of the inserted block
                zero[t] = 0.0
            at += 4 * k
            continue
        _export(plan, call, _BLOCK_BGMM + order[j])
        if sizes[j] > 0:
            unsafe_memcpy(dest=f32_ptr(at), src=f32_ptr(call.dst.data), count=sizes[j])
        at += 4 * sizes[j]
    var ext = Python.dict()
    ext[PythonObject("key")] = Python.tuple(PythonObject(k), PythonObject(d), _attr(model, "covariance_type"))
    ext[PythonObject("pchol")] = pchol
    ext[PythonObject("consts")] = consts
    ext[PythonObject("score_words")] = bgmm_score_words(means, pchol, consts, context)
    ext[PythonObject("state")] = warm
    _set(model, "_ext", ext)
    _ = means
    _ = warm


def _gmm_plain_state(model: PythonObject, context: PythonObject, plan: MemberPlan) raises -> Int:
    """A GaussianMixture state from a model fitted by the option-parity fit:
    the score entry's model blocks (means_, `_ext`'s pchol and consts) and
    the three fitted scalars."""
    var ext = _get_or(model, "_ext", PythonObject(None))
    var means_source = _get_or(model, "means_", PythonObject(None))
    if ext is None or means_source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var pchol_source = py_get(ext, PythonObject("pchol"), PythonObject(None))
    var consts_source = py_get(ext, PythonObject("consts"), PythonObject(None))
    if pchol_source is None or consts_source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var means = _any_f32(means_source, "means_", context)
    var pchol = _any_f32(pchol_source, "pchol", context)
    var consts = _any_f32(consts_source, "consts", context)
    var names: List[String] = ["lower_bound_", "n_iter_", "converged_"]
    var scalars = List[Float64](length=_BGMM_SCALAR_WORDS, fill=0.0)
    for i in range(len(names)):  # driver-loop: three fitted scalar attributes
        var value = _get_or(model, names[i], PythonObject(None))
        if value is not None:
            scalars[i] = read_float_like(value)
    var call = member_new(plan)
    try:
        call.block = _BLOCK_BGMM + 1
        call.x = MemberMatrix(Int(py=classical_address(means)), Int(py=means.shape[0]), Int(py=means.shape[1]))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = _BLOCK_BGMM + 3
        call.x = MemberMatrix(Int(py=classical_address(pchol)), 1, Int(py=pchol.size))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = _BLOCK_BGMM + 8
        call.x = MemberMatrix(Int(py=classical_address(consts)), 1, Int(py=consts.size))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, _BGMM_SCALAR_WORDS, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = means
    _ = pchol
    _ = consts
    _ = len(scalars)
    return call.state


def xcluster_member_pack(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan, state: Int) raises:
    """PACK: write the fitted attributes of state `state` into `model`, as
    the estimator's own fit does (the same names, dtypes and shapes)."""
    var kind = Int(plan.params[0])
    if (kind != _KIND_BISECT and kind != _KIND_MINIBATCH and kind != _KIND_MEANSHIFT and kind != _KIND_AFFINITY
            and kind != _KIND_BGMM and kind != _KIND_OPTICS):
        raise Error("RuntimeError: mojolearn: this x_cluster member kind is not published")
    classical_connect_numpy(context)
    var call = MemberCall(plan.route, state)
    if kind == _KIND_OPTICS:
        # Lane mf-member8: `optics_fit`'s publications, the same names,
        # dtypes, shapes and order (core/xcluster_member.mojo blocks 0 .. 5).
        _export(plan, call, MEMBER_BLOCK_FAMILY + 40)
        var n = call.dst.cols
        var core_distances = _f32_shaped(call.dst, Python.tuple(n), context)
        _export(plan, call, MEMBER_BLOCK_FAMILY + 41)
        var reachability = _f32_shaped(call.dst, Python.tuple(n), context)
        _export(plan, call, MEMBER_BLOCK_FAMILY + 42)
        var ordering = _i32_array(call.dst, Python.tuple(n), context)
        _export(plan, call, MEMBER_BLOCK_FAMILY + 43)
        var predecessor = _i32_array(call.dst, Python.tuple(n), context)
        _export(plan, call, MEMBER_BLOCK_FAMILY + 1)
        var op_labels = _i32_array(call.dst, Python.tuple(n), context)
        _set(model, "core_distances_", core_distances)
        _set(model, "reachability_", reachability)
        _set(model, "ordering_", ordering)
        _set(model, "predecessor_", predecessor)
        _set(model, "labels_", op_labels)
        if plan.params[7] == 0.0:
            _export(plan, call, MEMBER_BLOCK_FAMILY + 44)
            _set(model, "cluster_hierarchy_", _i32_array(call.dst, Python.tuple(call.dst.rows, 2), context))
        # The fit's X width (SCALARS [n_features_in_]), as `optics_fit` publishes it.
        _export(plan, call, MEMBER_BLOCK_SCALARS)
        var op_features = Int(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)[0])
        _set(model, "n_features_in_", PythonObject(op_features))
        return
    if kind == _KIND_BGMM and _plain(plan):
        _gmm_plain_pack(model, context, plan, call)
        return
    if kind == _KIND_BGMM:
        _bgmm_pack(model, context, plan, call)
        return
    if kind == _KIND_AFFINITY:
        _export(plan, call, MEMBER_BLOCK_FAMILY)
        var ap_d = call.dst.cols
        var ap_centers = _f32_array(call.dst, context)
        _export(plan, call, MEMBER_BLOCK_FAMILY + 5)
        var exemplars = _i32_array(call.dst, Python.tuple(call.dst.cols), context)
        _export(plan, call, MEMBER_BLOCK_FAMILY + 1)
        var ap_labels = _i32_array(call.dst, Python.tuple(call.dst.cols), context)
        _export(plan, call, MEMBER_BLOCK_FAMILY + 6)
        var matrix = _f32_array(call.dst, context)
        _export(plan, call, MEMBER_BLOCK_FAMILY + 7)
        var messages = _f32_shaped(call.dst, Python.tuple(call.dst.cols), context)
        _export(plan, call, MEMBER_BLOCK_SCALARS)
        var ap_iterations = Int(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)[0])
        _set(model, "cluster_centers_indices_", exemplars)
        _set(model, "labels_", ap_labels)
        _set(model, "affinity_matrix_", matrix)
        _set(model, "message_diag_", messages)
        _set(model, "n_iter_", PythonObject(ap_iterations))
        # affinity='precomputed' (data word 7, lane mf-clu6): X was the n x n
        # similarity matrix, so the feature count is the row count and the
        # class publishes no centers (`affinity_fit`).
        var ap_precomputed = len(plan.params) > 7 and plan.params[7] != 0.0
        _set(model, "n_features_in_", PythonObject(Int(py=ap_labels.shape[0]) if ap_precomputed else ap_d))
        if not ap_precomputed:
            _set(model, "cluster_centers_", ap_centers)
        return
    if kind == _KIND_MEANSHIFT:
        _export(plan, call, MEMBER_BLOCK_FAMILY)
        var ms_d = call.dst.cols
        var ms_centers = _f32_array(call.dst, context)
        _export(plan, call, MEMBER_BLOCK_FAMILY + 1)
        var ms_labels = _i32_array(call.dst, Python.tuple(call.dst.cols), context)
        _export(plan, call, MEMBER_BLOCK_SCALARS)
        var ms = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
        var found_bandwidth = ms[0]
        var ms_iterations = Int(ms[1])
        _set(model, "cluster_centers_", ms_centers)
        _set(model, "labels_", ms_labels)
        _set(model, "bandwidth_", PythonObject(found_bandwidth))
        _set(model, "n_iter_", PythonObject(ms_iterations))
        _set(model, "n_features_in_", PythonObject(ms_d))
        return
    if kind == _KIND_MINIBATCH:
        _export(plan, call, MEMBER_BLOCK_FAMILY)
        var cols = call.dst.cols
        var mb_centers = _f32_array(call.dst, context)
        _export(plan, call, MEMBER_BLOCK_FAMILY + 4)
        var counts = _f32_shaped(call.dst, Python.tuple(call.dst.cols), context)
        _export(plan, call, MEMBER_BLOCK_FAMILY + 1)
        var mb_labels = _i32_array(call.dst, Python.tuple(call.dst.cols), context)
        _export(plan, call, MEMBER_BLOCK_SCALARS)
        var mb = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
        var mb_inertia = mb[0]
        var steps = Int(mb[1])
        var iterations = Int(mb[2])
        _set(model, "cluster_centers_", mb_centers)
        _set(model, "counts_", counts)
        _set(model, "labels_", mb_labels)
        _set(model, "inertia_", PythonObject(mb_inertia))
        _set(model, "n_steps_", PythonObject(steps))
        _set(model, "n_iter_", PythonObject(iterations))
        _set(model, "n_features_in_", PythonObject(cols))
        return
    _export(plan, call, MEMBER_BLOCK_FAMILY)
    var d = call.dst.cols
    var centers = _f32_array(call.dst, context)
    _export(plan, call, MEMBER_BLOCK_FAMILY + 2)
    var tree_centers = _f32_array(call.dst, context)
    _export(plan, call, MEMBER_BLOCK_FAMILY + 3)
    var tree_nodes = _i32_array(call.dst, Python.tuple(call.dst.rows, 3), context)
    _export(plan, call, MEMBER_BLOCK_FAMILY + 1)
    var labels = _i32_array(call.dst, Python.tuple(call.dst.cols), context)
    _export(plan, call, MEMBER_BLOCK_SCALARS)
    var inertia = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)[0]
    _set(model, "cluster_centers_", centers)
    _set(model, "_tree_centers", tree_centers)
    _set(model, "tree_nodes_", tree_nodes)
    _set(model, "labels_", labels)
    _set(model, "inertia_", PythonObject(inertia))
    _set(model, "n_features_in_", PythonObject(d))


def _centers_state(fields: PythonObject, plan: MemberPlan, centers: PythonObject, names: List[String]) raises -> Int:
    """A state of a centers kind: the centers block, then the kind's scalar
    attributes in its SCALARS order."""
    var count = len(names)
    var scalars = List[Float64](length=count, fill=0.0)
    for i in range(count):  # driver-loop: one to three fitted scalar attributes
        var value = py_get(fields, PythonObject(names[i]), PythonObject(None))
        if value is not None:
            scalars[i] = Float64(py=value)
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_FAMILY
        call.x = MemberMatrix(Int(py=classical_address(centers)), Int(py=centers.shape[0]), Int(py=centers.shape[1]))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, count, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = centers
    _ = len(scalars)
    return call.state


def _bgmm_state(fields: PythonObject, context: PythonObject, plan: MemberPlan) raises -> Int:
    """A BayesianGaussianMixture state from a fitted model: the score
    entry's model blocks (means_, _full_pchol, _log_consts, the words
    `_bgmm_score` reads) and the six fitted scalars."""
    var means_source = py_get(fields, PythonObject("means_"), PythonObject(None))
    var pchol_source = py_get(fields, PythonObject("_full_pchol"), PythonObject(None))
    var consts_source = py_get(fields, PythonObject("_log_consts"), PythonObject(None))
    if means_source is None or pchol_source is None or consts_source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var means = _any_f32(means_source, "means_", context)
    var pchol = _any_f32(pchol_source, "_full_pchol", context)
    var consts = _any_f32(consts_source, "_log_consts", context)
    var names: List[String] = ["lower_bound_", "n_iter_", "converged_", "weight_concentration_prior_",
        "mean_precision_prior_", "degrees_of_freedom_prior_"]
    var scalars = List[Float64](length=len(names), fill=0.0)
    for i in range(len(names)):  # driver-loop: six fitted scalar attributes
        var value = py_get(fields, PythonObject(names[i]), PythonObject(None))
        if value is not None:
            scalars[i] = read_float_like(value)
    var call = member_new(plan)
    try:
        call.block = _BLOCK_BGMM + 1
        call.x = MemberMatrix(Int(py=classical_address(means)), Int(py=means.shape[0]), Int(py=means.shape[1]))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = _BLOCK_BGMM + 3
        call.x = MemberMatrix(Int(py=classical_address(pchol)), 1, Int(py=pchol.size))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = _BLOCK_BGMM + 8
        call.x = MemberMatrix(Int(py=classical_address(consts)), 1, Int(py=consts.size))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, len(names), MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = means
    _ = pchol
    _ = consts
    _ = len(scalars)
    return call.state


def _optics_state(fields: PythonObject, context: PythonObject, plan: MemberPlan) raises -> Int:
    """Round b (lane mf-member8): a member state from a fitted OPTICS (a
    meta-estimator's held estimator): its five per-row arrays, the xi
    hierarchy and n_features_in_, as `optics_fit` published them."""
    var f32_names: List[String] = ["core_distances_", "reachability_"]
    var f32_blocks: List[Int] = [MEMBER_BLOCK_FAMILY + 40, MEMBER_BLOCK_FAMILY + 41]
    var i32_names: List[String] = ["ordering_", "predecessor_", "labels_", "cluster_hierarchy_"]
    var i32_blocks: List[Int] = [MEMBER_BLOCK_FAMILY + 42, MEMBER_BLOCK_FAMILY + 43, MEMBER_BLOCK_FAMILY + 1,
        MEMBER_BLOCK_FAMILY + 44]
    var width = py_get(fields, PythonObject("n_features_in_"), PythonObject(None))
    if width is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var scalars: List[Float64] = [Float64(py=width)]
    var held = Python.list()
    var call = member_new(plan)
    try:
        for j in range(2):  # driver-loop: two float32 fitted arrays
            var source = py_get(fields, PythonObject(f32_names[j]), PythonObject(None))
            if source is None:
                raise Error("NotFittedError: this member is not fitted yet")
            var values = admit_family(source, 1, f32_names[j], context).array
            _ = py_call(held.append, values)
            call.block = f32_blocks[j]
            call.x = MemberMatrix(Int(py=classical_address(values)), 1, Int(py=values.size))
            call.flags = 0
            _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        for j in range(4):  # driver-loop: four int32 fitted arrays (the hierarchy only for 'xi')
            var source = py_get(fields, PythonObject(i32_names[j]), PythonObject(None))
            if source is None:
                if j == 3:
                    continue
                raise Error("NotFittedError: this member is not fitted yet")
            # The raw C-order int32 words, as the class's own readers take them.
            var words = classical_c_bytes(source, i32_names[j], context)
            _ = py_call(held.append, words)
            var count = len(words) // 4
            if count == 0:
                continue
            call.block = i32_blocks[j]
            call.x = MemberMatrix(native_buffer_address(words), count // 2 if j == 3 else 1,
                                  2 if j == 3 else count, MEMBER_I32)
            call.flags = 0
            _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 1, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = held
    _ = len(scalars)
    return call.state


def xcluster_member_state(model: PythonObject, fields: PythonObject, context: PythonObject, b: PythonObject,
    plan: MemberPlan) raises -> Int:
    """UNPACK: a new member state holding the fitted leaf centers, tree and
    inertia of `model` (labels_ are not needed to predict and are not
    imported). A MiniBatchKMeans state holds its centers and
    [inertia_, n_steps_, n_iter_]. The caller frees it (`member_free`)."""
    var kind = Int(plan.params[0])
    if (kind != _KIND_BISECT and kind != _KIND_MINIBATCH and kind != _KIND_MEANSHIFT and kind != _KIND_AFFINITY
            and kind != _KIND_BGMM and kind != _KIND_OPTICS):
        raise Error("RuntimeError: mojolearn: this x_cluster member kind is not published")
    classical_connect_numpy(context)
    if kind == _KIND_OPTICS:
        return _optics_state(fields, context, plan)
    if kind == _KIND_BGMM and _plain(plan):
        return _gmm_plain_state(model, context, plan)
    if kind == _KIND_BGMM:
        return _bgmm_state(fields, context, plan)
    if kind == _KIND_AFFINITY and len(plan.params) > 7 and plan.params[7] != 0.0:
        # A precomputed fit has no centers to predict from: scikit-learn's
        # own refusal (sklearn/cluster/_affinity_propagation.py `predict`).
        raise Error("Predict method is not supported when affinity='precomputed'.")
    var source = py_get(fields, PythonObject("cluster_centers_"), PythonObject(None))
    if source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var centers = admit_family(source, 2, "cluster_centers_", context, finite=True).array
    if kind == _KIND_MINIBATCH:
        var mb_names: List[String] = ["inertia_", "n_steps_", "n_iter_"]
        return _centers_state(fields, plan, centers, mb_names)
    if kind == _KIND_MEANSHIFT:
        var ms_names: List[String] = ["bandwidth_", "n_iter_"]
        return _centers_state(fields, plan, centers, ms_names)
    if kind == _KIND_AFFINITY:
        var ap_names: List[String] = ["n_iter_"]
        return _centers_state(fields, plan, centers, ap_names)
    var tree_source = py_get(fields, PythonObject("_tree_centers"), PythonObject(None))
    var node_source = py_get(fields, PythonObject("tree_nodes_"), PythonObject(None))
    if tree_source is None or node_source is None:
        raise Error("NotFittedError: this member is not fitted yet")
    var tree = admit_family(tree_source, 2, "_tree_centers", context, finite=True).array
    # The tree's raw C-order int32 words, read as the estimator's own predict reads them.
    var nodes = classical_c_bytes(node_source, "tree_nodes_", context)
    var words = cluster_node_words(len(nodes))
    var scalars: List[Float64] = [0.0]
    var value = py_get(fields, PythonObject("inertia_"), PythonObject(None))
    if value is not None:
        scalars[0] = Float64(py=value)
    var call = member_new(plan)
    try:
        call.block = MEMBER_BLOCK_FAMILY
        call.x = MemberMatrix(Int(py=classical_address(centers)), Int(py=centers.shape[0]), Int(py=centers.shape[1]))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_FAMILY + 2
        call.x = MemberMatrix(Int(py=classical_address(tree)), Int(py=tree.shape[0]), Int(py=tree.shape[1]))
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_FAMILY + 3
        call.x = MemberMatrix(native_buffer_address(nodes) if words else 0, words // 3, 3, MEMBER_I32)
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
        call.block = MEMBER_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 1, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(plan.member, MEMBER_OP_IMPORT, call)
    except error:
        member_free(plan, call)
        raise error
    _ = centers
    _ = source
    _ = tree
    _ = tree_source
    _ = nodes
    _ = node_source
    _ = len(scalars)
    return call.state

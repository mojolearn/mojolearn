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
from bindings.py_balanced import py_call, py_type, py_contains, py_setattr
from bindings.family_common_unpack import (
    family_nonfinite,
    family_shape_repr,
)
from bindings.family_common_pack import family_call
from bindings.py_cluster_resident import CRES_X_CLUSTER, cres_forget
from core.kmeans_variants_api import VariantsSpan
from core.cluster_centroid_fit_api import CentroidFitResultSlot
from bindings.cluster_centroid_fit_boundary_door import centroid_fit_connection
from bindings.kmeans_variants_boundary_door import variants_span
from cluster.cluster_fit_door_run import (
    meanshift_seed_features, meanshift_bandwidth_refused, meanshift_fit_request,
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

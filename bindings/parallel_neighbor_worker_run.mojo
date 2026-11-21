# SPDX-License-Identifier: Apache-2.0
"""Typed rules of the neighbor / IVF shard protocol (RUN): the count, shape,
radius and mode refusals of the worker's handlers, the query method codes
and their keywords, the segmented sort's network shape and the class-set
check of a reference-sharded vote. `bindings/parallel_neighbor_worker_door.mojo`
unpacks the Python values into these arguments, calls the neighbor family's
compiled entries and packs the results; no Python object reaches this file.
A few words per shard request: nothing to split across cores."""
from std.math import isfinite

# A query-rows method, by its public name.
comptime NEIGHBOR_METHOD_OTHER = 0
comptime NEIGHBOR_METHOD_KNEIGHBORS = 1
comptime NEIGHBOR_METHOD_RADIUS = 2
comptime NEIGHBOR_METHOD_PREDICT = 3
comptime NEIGHBOR_METHOD_PREDICT_PROBA = 4
comptime NEIGHBOR_METHOD_SCORE_SAMPLES = 5


def neighbor_method_code(method: String) -> Int:
    if method == "kneighbors":
        return NEIGHBOR_METHOD_KNEIGHBORS
    if method == "radius_neighbors":
        return NEIGHBOR_METHOD_RADIUS
    if method == "predict":
        return NEIGHBOR_METHOD_PREDICT
    if method == "predict_proba":
        return NEIGHBOR_METHOD_PREDICT_PROBA
    if method == "score_samples":
        return NEIGHBOR_METHOD_SCORE_SAMPLES
    return NEIGHBOR_METHOD_OTHER


def neighbor_keyword_admit(method: Int, method_name: String, exact: Bool, key: String) raises:
    """The keywords a query method takes: `kneighbors` n_neighbors and
    return_distance; `radius_neighbors` radius, return_distance and
    sort_results; every other method none. `exact`: the key is an exact str."""
    var allowed = False
    if exact and method == NEIGHBOR_METHOD_KNEIGHBORS:
        allowed = key == "n_neighbors" or key == "return_distance"
    elif exact and method == NEIGHBOR_METHOD_RADIUS:
        allowed = key == "radius" or key == "return_distance" or key == "sort_results"
    if not allowed:
        raise Error("TypeError: unsupported keyword for " + method_name)


def neighbor_features_admit(columns: Int, fitted: Int) raises:
    if columns != fitted:
        raise Error("ValueError: input feature count differs from fit")


def neighbor_k_admit(k: Int, reference_rows: Int) raises:
    if k < 1 or k > reference_rows:
        raise Error("ValueError: n_neighbors must be in [1, reference rows]")


def neighbor_filled_admit(lowest_id: Int) raises:
    if lowest_id < 0:
        raise Error("RuntimeError: RBC returned an unfilled neighbor slot")


def neighbor_radius_admit(radius: Float64) raises:
    if not isfinite(radius) or radius <= 0:
        raise Error("ValueError: radius must be positive and finite")


def neighbor_sort_admit(sort: Bool, distances: Bool) raises:
    if sort and not distances:
        raise Error("ValueError: sort_results=True requires return_distance=True")


# The fill-count refusal and the sort network's shape of the radius query are
# neighbors/radius_run.mojo (radius_fill_run, radius_sort_run).


def neighbor_radius_offsets_admit(offsets: Int) raises -> Int:
    """Query rows of a radius reply from its row-offset count."""
    if offsets < 1:
        raise Error("RuntimeError: radius worker returned no row offsets")
    return offsets - 1


def neighbor_class_sets_admit(unique: Int, fitted: List[Int]) raises:
    """The vote kernel's class ids (int32 at `unique`) are the fitted ones,
    output after output."""
    var native = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=unique)
    for i in range(len(fitted)):  # small-loop(class ids of the fitted outputs)
        if Int(native[i]) != fitted[i]:
            raise Error("RuntimeError: native and fitted class sets differ")


def neighbor_ivf_features_admit(columns: Int, fitted: Int) raises:
    if columns != fitted:
        raise Error("ValueError: query features differ from index")


def neighbor_ivf_mode_admit(exact: Bool, word: String) raises:
    if not exact or word != "identical":
        raise Error("ValueError: distributed IVF requires IDENTICAL numeric mode")


def neighbor_ivf_metric_admit(requested: Int, fitted: Int) raises:
    if requested != fitted:
        raise Error("ValueError: metric differs from the fitted index")


def neighbor_ivf_counts_admit(probes: Int, lists: Int, neighbors: Int) raises:
    if probes < 1 or probes > lists or neighbors < 1:
        raise Error("ValueError: invalid IVF probe or neighbor count")


def neighbor_ivf_binding_admit(mode_word: Int) raises:
    """The IVF binding's compile-time mode word: 1 is identical."""
    if mode_word != 1:
        raise Error("RuntimeError: IVF binding reports another compile-time mode")


def neighbor_ivf_data_admit(rank: Int, rows: Int, columns: Int, n: Int, dim: Int) raises:
    """`rows` / `columns` are read only for a rank-2 block."""
    if rank != 2 or rows != n or columns != dim:
        raise Error("ValueError: IVF stored data shape differs from index metadata")


def neighbor_ivf_lists_admit(ids: Int, n: Int, offsets: Int, lists: Int) raises:
    if ids != n:
        raise Error("ValueError: IVF original IDs must be a permutation of stored row IDs")
    if offsets != lists + 1:
        raise Error("ValueError: invalid global IVF list offsets")

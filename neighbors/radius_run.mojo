# SPDX-License-Identifier: Apache-2.0
"""RadiusNeighbors RUN: the typed radius query, no Python.

A radius query's output size is not a function of its inputs, so RUN is two
stages around one allocation: `radius_count_run` (pass one: the row starts
and the edge count), then `radius_fill_run` (pass two, the optional row sort,
and the int64 ids). The boundary (bindings/parallel_neighbor_worker_door.mojo
`_radius`) admits the public arguments into `RadiusRequest`, allocates the
edge buffers between the stages, releases the GIL for each stage and packs
the result. The passes are the neighbors operations table's
(core/neighbors_entry_table.mojo), the sort is the x_neighbors driver's.
"""
from core.native_descriptor_abi import NativeDescriptorCall
from core.neighbors_entry_table import nb_radius_count_call, nb_radius_fill_call
from core.neighbors_native_driver import XnDriver
from core.admitted_cast_elements import admitted_cast_elements


struct RadiusRequest(Copyable, Movable):
    # Addresses of admitted storage the boundary keeps alive: the fitted
    # float32 index rows, the float32 query rows (the index itself for a
    # query of the index) and the int32 row starts (`n_queries + 1` words).
    # `device_status`: the count pass tests the uploaded queries and returns
    # their finite word (a borrowed query on a GPU image).
    var index: Int
    var queries: Int
    var row_starts: Int
    var n_index: Int
    var n_queries: Int
    var n_features: Int
    var metric: Int
    var radius: Float64
    var metric_arg: Float64
    var device_status: Bool

    def __init__(out self, index: Int, queries: Int, row_starts: Int, n_index: Int, n_queries: Int,
                 n_features: Int, metric: Int, radius: Float64, metric_arg: Float64, device_status: Bool):
        self.index = index
        self.queries = queries
        self.row_starts = row_starts
        self.n_index = n_index
        self.n_queries = n_queries
        self.n_features = n_features
        self.metric = metric
        self.radius = radius
        self.metric_arg = metric_arg
        self.device_status = device_status


struct RadiusCount(Copyable, Movable):
    # `edges` is the edge count of pass one; `status` the finite word of the
    # queries (0 = every word finite, or not tested by this pass).
    var edges: Int
    var status: Int

    def __init__(out self):
        self.edges = 0
        self.status = 0


# The radius, option and feature-count refusals of the query are the typed
# admits of bindings/parallel_neighbor_worker_run.mojo (neighbor_radius_admit,
# neighbor_sort_admit, neighbor_features_admit): one set of checks.


def radius_count_run(entry: NativeDescriptorCall, request: RadiusRequest) raises -> RadiusCount:
    """Pass one: the row starts at `request.row_starts` and the edge count."""
    var result = RadiusCount()
    var status = List[Int32](length=1, fill=Int32(0))
    result.edges = nb_radius_count_call(entry, request.index, request.queries, request.row_starts,
        Int(status.unsafe_ptr()) if request.device_status else 0,
        request.n_index, request.n_queries, request.n_features, request.metric,
        request.radius, request.metric_arg)
    result.status = Int(status[0])
    _ = status^
    return result^


def radius_fill_run(entry: NativeDescriptorCall, request: RadiusRequest, edges: Int,
                    ids: Int, distances: Int) raises:
    """Pass two: `edges` int32 ids (ascending within a row) and float32
    distances, in CSR order. Nothing runs for an empty result."""
    if edges <= 0:
        return
    var got = nb_radius_fill_call(entry, request.index, request.queries, request.row_starts, ids, distances,
        request.n_index, request.n_queries, request.n_features, edges, request.metric,
        request.radius, request.metric_arg)
    if got != edges:
        raise Error("RuntimeError: radius counting and filling passes disagree")


def radius_sort_run(driver: XnDriver, request: RadiusRequest, edges: Int, ids: Int, distances: Int,
                    sorted_ids: Int, sorted_distances: Int) raises:
    """Each row's edges by ascending distance, into the sorted buffers: the
    x_neighbors segmented row sort (`p2m_row_sort`) over the padded power."""
    var power = 1
    var lg = 0
    while power < edges:  # driver-loop: segmented sorting network shape
        power *= 2
        lg += 1
    var a: Array[Int, 5] = [request.row_starts, ids, distances, sorted_ids, sorted_distances]
    var p: Array[Int, 4] = [request.n_queries, edges, power, lg * (lg + 1) // 2]
    var f: Array[Float64, 0] = []
    _ = driver.run("p2m_row_sort", a, p, f)


def radius_widen_run(ids: Int, wide: Int, edges: Int) raises:
    """The int32 ids as the public int64 ids (an exact widening)."""
    # parallel: admitted_cast_elements splits the element range across the host's cores
    if admitted_cast_elements(ids, 2, wide, 3, edges) != 0:
        raise Error("TypeError: mojolearn: an element does not fit <i8")

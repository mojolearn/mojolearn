# SPDX-License-Identifier: Apache-2.0
"""Typed AgglomerativeClustering requests over its existing native fit ABIs."""
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_fixed_call import native_descriptor_call_fixed
from core.neighbors_native_driver import XnDriver
from hierarchy.solver_descriptor_schema import HIERARCHY_SOLVER_LINKAGE
from hierarchy.cluster_descriptor_schema import HIERARCHY_CLUSTER_FIT
from core.cfb_impl import cfb_any_positive_i32
from core.dbscan_native_api import DBSCANReferenceRequest, DBSCANQueryRequest, DBSCANResidentQuery


def agglo_reference_request(refs: Int, labels: Int, rows: Int, features: Int) -> DBSCANReferenceRequest:
    # Agglomerative ties use the fitted cluster label, then reference-row index.
    # The shared labeled-reference query uses code 0 for Euclidean distance.
    return DBSCANReferenceRequest(refs, labels, labels, rows, features, 0)


def agglo_query_request(refs: Int, labels: Int, query: Int, dst: Int, chosen: Int,
    rows: Int, queries: Int, features: Int) -> DBSCANQueryRequest:
    return DBSCANQueryRequest(refs, labels, labels, query, dst, chosen,
        rows, queries, features, 0, False, 0.0)


def agglo_resident_query(query: Int, dst: Int, handle: Int, queries: Int, features: Int) -> DBSCANResidentQuery:
    return DBSCANResidentQuery(query, dst, handle, queries, features, 0, False, 0.0)


@fieldwise_init
struct AggloSolverRequest(Copyable, Movable):
    var source: Int
    var children: Int
    var labels: Int
    var info: Int
    var rows: Int
    var features: Int
    var clusters: Int
    var metric: Int
    var use_knn: Bool


@fieldwise_init
struct AggloClusterRequest(Copyable, Movable):
    var source: Int
    var connectivity: Int
    var children: Int
    var labels: Int
    var distances: Int
    var scalars: Int
    var source_count: Int
    var connectivity_count: Int
    var rows: Int
    var features: Int
    var linkage: Int
    var metric: Int
    var edges: Int
    var merges: Int
    var edge_mode: Int
    var cut: Int
    var use_threshold: Bool
    var minkowski: Float64
    var threshold: Float64


@fieldwise_init
struct AggloZeroRowsRequest(Copyable, Movable):
    var source: Int
    var flags: Int
    var rows: Int
    var features: Int


@fieldwise_init
struct AggloSolverDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def fit(self, r: AggloSolverRequest) raises -> Int:
        var a: InlineArray[Int, 4] = [r.source, r.children, r.labels, r.info]
        var p: InlineArray[Int, 5] = [r.rows, r.features, r.clusters, r.metric, Int(r.use_knn)]
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, HIERARCHY_SOLVER_LINKAGE, a, p, f)


@fieldwise_init
struct AggloClusterDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def fit(self, r: AggloClusterRequest) raises:
        var a: InlineArray[Int, 6] = [r.source, r.connectivity, r.children, r.labels, r.distances, r.scalars]
        var p: InlineArray[Int, 11] = [r.source_count, r.connectivity_count, r.rows, r.features,
            r.linkage, r.metric, r.edges, r.merges, r.edge_mode, r.cut, Int(r.use_threshold)]
        var f: InlineArray[Float64, 2] = [r.minkowski, r.threshold]
        _ = native_descriptor_call_fixed(self.entry, HIERARCHY_CLUSTER_FIT, a, p, f)


@fieldwise_init
struct AggloZeroRowsDriver(Copyable, Movable):
    var native: XnDriver

    def run(self, r: AggloZeroRowsRequest) raises:
        var a: InlineArray[Int, 2] = [r.source, r.flags]
        var p: InlineArray[Int, 2] = [r.rows, r.features]
        var f: InlineArray[Float64, 0] = []
        _ = self.native.run("row_all_zero", a, p, f)


struct AggloSolverScratch(Movable):
    var info: List[Int32]

    def __init__(out self):
        self.info = List[Int32](length=2, fill=0)

    def address(mut self) -> Int:
        return Int(self.info.unsafe_ptr())

    def rounds(self) -> Int:
        return Int(self.info[0])

    def components(self) -> Int:
        return Int(self.info[1])


struct AggloClusterScratch(Movable):
    var scalars: List[Float64]

    def __init__(out self):
        self.scalars = List[Float64](length=2, fill=0.0)

    def address(mut self) -> Int:
        return Int(self.scalars.unsafe_ptr())

    def components(self) -> Int:
        return Int(self.scalars[0])

    def clusters(self) -> Int:
        return Int(self.scalars[1])


struct AggloZeroRowsScratch(Movable):
    var flags: List[Int32]

    def __init__(out self, rows: Int):
        self.flags = List[Int32](length=rows, fill=0)

    def address(mut self) -> Int:
        return Int(self.flags.unsafe_ptr())

    def any_zero(mut self) -> Bool:
        return cfb_any_positive_i32(self.flags.unsafe_ptr(), len(self.flags))

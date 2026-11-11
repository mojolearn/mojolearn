# SPDX-License-Identifier: Apache-2.0
"""Original CPU query algorithms and ownership transfers; no Python calls."""
from core.hostptr import f32_ptr, i32_ptr
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.hdbscan_query_api import HDBSCAN_QUERY_ABI, HdbscanQueryState, HDBSCAN_QUERY_APPROXIMATE, HDBSCAN_QUERY_ALL_POINTS
from hdbscan.native_query_dispatch import HdbscanQueryBackend, hdbscan_query_dispatch
from hdbscan.host.hdbscan_door_host_impl import approximate_predict_host_run, membership_vector_host_run, all_points_membership_host_run
from checks.numerics import GLOBAL_NUMERIC_MODE


struct HdbscanQueryHostBackend(HdbscanQueryBackend):
    @staticmethod
    def mode() -> Int:
        return GLOBAL_NUMERIC_MODE

    @staticmethod
    def host() -> Bool:
        return True

    @staticmethod
    def run(mut state: HdbscanQueryState) raises -> Int:
        var r = state.request.copy()
        var d = r.dimensions.copy()
        var b = r.buffers.copy()
        if state.operation == HDBSCAN_QUERY_APPROXIMATE:
            approximate_predict_host_run(state.training, d.rows, d.features,
                f32_ptr(b.core_distances.address), i32_ptr(b.labels.address), f32_ptr(b.lambdas.address),
                f32_ptr(b.deaths.address), i32_ptr(b.selected.address), i32_ptr(b.index_into.address),
                state.query, r.query_rows, d.neighbors, i32_ptr(b.output.address), f32_ptr(b.probabilities.address))
            return r.query_rows
        # These five snapshots were owned arguments of the original host
        # door. Swap each into an empty owner so state remains initialized;
        # no tensor allocation or copy is added by the ownership transfer.
        var deaths = List[Float32]()
        swap(deaths, state.deaths)
        var selected = List[Int32]()
        swap(selected, state.selected)
        var index_into = List[Int32]()
        swap(index_into, state.index_into)
        var exemplars = List[Int32]()
        swap(exemplars, state.exemplars)
        var offsets = List[Int32]()
        swap(offsets, state.offsets)
        if state.operation == HDBSCAN_QUERY_ALL_POINTS:
            all_points_membership_host_run(f32_ptr(b.training.address), d.rows, d.features,
                state.parents, state.lambdas, d.edges, d.condensed, deaths^, selected^,
                index_into^, exemplars^, offsets^, r.row_start, r.query_rows, d.clusters, f32_ptr(b.output.address))
        else:
            membership_vector_host_run(state.training, d.rows, d.features,
                f32_ptr(b.core_distances.address), i32_ptr(b.labels.address), state.parents,
                state.lambdas, d.edges, d.condensed, deaths^, selected^, index_into^,
                exemplars^, offsets^, state.query, r.query_rows, d.neighbors, d.clusters, f32_ptr(b.output.address))
        return r.query_rows


def hdbscan_query_host_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return hdbscan_query_dispatch[HdbscanQueryHostBackend](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct HdbscanQueryHostAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(HDBSCAN_QUERY_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(hdbscan_query_host_v1)


comptime HDBSCAN_QUERY_HOST_API = SharedSlot[HdbscanQueryHostAPI,
    ("MojolearnHdbscanQueryHostIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnHdbscanQueryHostFastV1")]

# The fit-host and inference-only bindings are separate provider images.
# Each capsule must retain its own image's entry, never a pointer first
# published by the other image under a shared process slot identity.
comptime HDBSCAN_QUERY_INFER_HOST_API = SharedSlot[HdbscanQueryHostAPI,
    ("MojolearnHdbscanQueryInferHostIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnHdbscanQueryInferHostFastV1")]

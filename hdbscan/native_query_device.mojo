# SPDX-License-Identifier: Apache-2.0
"""Original GPU query doors and batch boundaries; no host algorithm fallback."""
from core.hostptr import f32_ptr, i32_ptr, read_f32
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.hdbscan_query_api import HDBSCAN_QUERY_ABI, HdbscanQueryState, HDBSCAN_QUERY_APPROXIMATE, HDBSCAN_QUERY_ALL_POINTS
from core.hostptr import read_i32
from core.identity_trace import IdentityTrace
from core.neural_context import process_ctx
from hdbscan.impl.hdbscan_door_impl import HDB_DEVCTX_SLOT
from hdbscan.impl.prediction_data import PredictionData, refuse_soft_clustering_inputs
from hdbscan.impl.detail.resident import (
    HdbscanResident, hdbscan_resident_publish, hdbscan_resident_release, hdbscan_resident_view,
)
from hdbscan.impl.detail.predict import approximate_predict_snapshot
from hdbscan.impl.detail.soft_clustering import membership_vector_snapshot, all_points_snapshot
from hdbscan.native_query_dispatch import HdbscanQueryBackend, hdbscan_query_dispatch
from hdbscan.native_query_device_doors import _hdbscan_predict_run, _hdbscan_membership_run, _hdbscan_all_points_run
from checks.numerics import GLOBAL_NUMERIC_MODE


struct HdbscanQueryDeviceBackend(HdbscanQueryBackend):
    @staticmethod
    def mode() -> Int:
        return GLOBAL_NUMERIC_MODE

    @staticmethod
    def host() -> Bool:
        return False

    @staticmethod
    def prepare(mut state: HdbscanQueryState) raises -> Int:
        """Lane vdf-misc4: the fitted arrays of the plan's spans, ONCE onto
        the device (hdbscan/impl/detail/resident.mojo), for the query kind
        `state.kind`; the soft kinds' admission of the tree runs here once
        (was: per call). Returns the snapshot's handle."""
        var r = state.request.copy()
        var d = r.dimensions.copy()
        var b = r.buffers.copy()
        var approximate = state.kind == HDBSCAN_QUERY_APPROXIMATE
        var all_points = state.kind == HDBSCAN_QUERY_ALL_POINTS
        if not approximate:
            var name = String("hdbscan.all_points_membership_vectors") if all_points else String("hdbscan.membership_vector")
            var pd = PredictionData(
                d.rows, d.edges, d.condensed, d.clusters, d.exemplars,
                read_f32(b.deaths.address, d.condensed), read_i32(b.exemplars.address, d.exemplars),
                read_i32(b.offsets.address, d.clusters + 1), read_i32(b.selected.address, d.clusters),
                read_i32(b.index_into.address, d.edges + 1),
            )
            refuse_soft_clustering_inputs(
                read_i32(b.parents.address, d.edges), read_f32(b.lambdas.address, d.edges), pd, d.rows, name,
            )
        var ctx = process_ctx[HDB_DEVCTX_SLOT]()
        var snapshot = HdbscanResident(
            ctx, b.training.address, d.rows, d.features,
            0 if all_points else b.core_distances.address, 0 if all_points else b.labels.address,
            0 if approximate else b.parents.address, b.lambdas.address, d.edges,
            b.deaths.address, d.condensed, b.selected.address, d.clusters, b.index_into.address,
            0 if approximate else b.exemplars.address, d.exemplars,
            0 if approximate else b.offsets.address,
        )
        var handle = hdbscan_resident_publish(snapshot^)
        # DEVIATION 1946: the context dies LAST, after every value built on it.
        _ = ctx^
        return handle

    @staticmethod
    def release(handle: Int) raises:
        hdbscan_resident_release(handle)

    @staticmethod
    def run(mut state: HdbscanQueryState) raises -> Int:
        var r = state.request.copy()
        var d = r.dimensions.copy()
        var b = r.buffers.copy()
        if state.handle != 0:
            return _run_snapshot(state)
        if state.operation == HDBSCAN_QUERY_APPROXIMATE:
            _hdbscan_predict_run(state.training, d.rows, d.features, state.core_distances,
                state.labels, state.lambdas, d.edges, d.condensed, state.deaths, state.selected,
                state.index_into, state.query, r.query_rows, d.neighbors,
                i32_ptr(b.output.address), f32_ptr(b.probabilities.address))
            return r.query_rows
        var first = 0
        while first < r.query_rows:  # driver-loop: original GPU query/training-row batch boundaries
            var count = min(r.batch, r.query_rows - first)
            if state.operation == HDBSCAN_QUERY_ALL_POINTS:
                _hdbscan_all_points_run(state.training, d.rows, d.features, state.parents,
                    state.lambdas, d.edges, d.condensed, state.deaths, state.selected,
                    state.index_into, state.exemplars, state.offsets, r.row_start + first, count,
                    f32_ptr(b.output.address + first * d.clusters * 4))
            else:
                # Original membership wrapper copied each query slice inside
                # the released-GIL batch loop; fitted snapshots remain shared.
                var query = read_f32(b.query.address + first * d.features * 4, count * d.features)
                _hdbscan_membership_run(state.training, d.rows, d.features, state.core_distances,
                    state.labels, state.parents, state.lambdas, d.edges, d.condensed, state.deaths,
                    state.selected, state.index_into, state.exemplars, state.offsets,
                    query, count, d.neighbors, f32_ptr(b.output.address + first * d.clusters * 4))
            first += count
        return r.query_rows


def _run_snapshot(mut state: HdbscanQueryState) raises -> Int:
    """A query on the fitted snapshot `state.handle` (lane vdf-misc4): the
    training rows, the tree and the exemplar rows are read on the device in
    place; per call only the query rows go up (per batch, once each) and the
    results come home once (per batch, into their slice of the output)."""
    var r = state.request.copy()
    var d = r.dimensions.copy()
    var b = r.buffers.copy()
    var e = hdbscan_resident_view(state.handle)
    if (e[].m != d.rows or e[].n != d.features or e[].n_edges != d.edges
            or e[].n_clusters != d.condensed or e[].n_selected != d.clusters):
        raise Error("HDBSCAN query ABI: the fitted snapshot was taken at another shape")
    var ctx = process_ctx[HDB_DEVCTX_SLOT]()
    var trace = IdentityTrace()
    if state.operation == HDBSCAN_QUERY_APPROXIMATE:
        approximate_predict_snapshot(ctx, trace, e[], b.query.address, r.query_rows, d.neighbors,
                                     b.output.address, b.probabilities.address)
    else:
        var first = 0
        while first < r.query_rows:  # driver-loop: original GPU query/training-row batch boundaries
            var count = min(r.batch, r.query_rows - first)
            var out_address = b.output.address + first * d.clusters * 4
            if state.operation == HDBSCAN_QUERY_ALL_POINTS:
                all_points_snapshot(ctx, trace, e[], r.row_start + first, count, out_address)
            else:
                membership_vector_snapshot(ctx, trace, e[], b.query.address + first * d.features * 4, count,
                                           d.neighbors, out_address)
            first += count
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = e^
    _ = ctx^
    return r.query_rows


def hdbscan_query_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return hdbscan_query_dispatch[HdbscanQueryDeviceBackend](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct HdbscanQueryDeviceAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(HDBSCAN_QUERY_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(hdbscan_query_device_v1)


comptime HDBSCAN_QUERY_DEVICE_API = SharedSlot[HdbscanQueryDeviceAPI,
    ("MojolearnHdbscanQueryDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnHdbscanQueryDeviceFastV1")]

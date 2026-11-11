# SPDX-License-Identifier: Apache-2.0
"""Original GPU query doors and batch boundaries; no host algorithm fallback."""
from core.hostptr import f32_ptr, i32_ptr, read_f32
from core.shared_slot import SharedSlot
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.hdbscan_query_api import HDBSCAN_QUERY_ABI, HdbscanQueryState, HDBSCAN_QUERY_APPROXIMATE, HDBSCAN_QUERY_ALL_POINTS
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
    def run(mut state: HdbscanQueryState) raises -> Int:
        var r = state.request.copy()
        var d = r.dimensions.copy()
        var b = r.buffers.copy()
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

# SPDX-License-Identifier: Apache-2.0
"""The existing CPU base image's COO and precomputed-neighbor helpers."""
from core.spectral_preparation_api import SpectralPreparationBackend, SpectralCOOFillRequest, SpectralKNNRequest
from core.hostptr import f32_ptr, f64_ptr, i32_ptr
from core.core_host_impl import nonzero_f64_count, nonzero_f64_fill
from core.dense_coo_host import knn_affinity_f32


struct SpectralPreparationHost(SpectralPreparationBackend):
    @staticmethod
    def count_wide(source: Int, count: Int) raises -> Int:
        return nonzero_f64_count(f64_ptr(source), count)

    @staticmethod
    def fill_wide(r: SpectralCOOFillRequest) raises -> Int:
        var source = f64_ptr(r.source)
        var rows = i32_ptr(r.row_output)
        var columns = i32_ptr(r.column_output)
        var values = f32_ptr(r.value_output)
        return nonzero_f64_fill(source, r.rows, r.columns, rows, columns, values, r.capacity)

    @staticmethod
    def knn(r: SpectralKNNRequest) raises:
        # The existing host door substitutes admitted destination pointers for
        # unused operands. Preserve that convention; the kernel ignores them.
        var affinity = f32_ptr(r.affinity)
        var status = i32_ptr(r.status)
        var dense = affinity
        var rows = status
        var columns = status
        var values = affinity
        if r.sparse:
            if r.nonzeros > 0:
                rows = i32_ptr(r.row_indices)
                columns = i32_ptr(r.column_indices)
                values = f32_ptr(r.values)
        else:
            dense = f32_ptr(r.dense)
        knn_affinity_f32(dense, rows, columns, values, r.nonzeros, r.sparse,
            r.rows, r.neighbors, affinity, status)

# SPDX-License-Identifier: Apache-2.0
"""The existing base image's COO helpers and device kNN affinity kernel."""
from core.spectral_preparation_api import SpectralPreparationBackend, SpectralCOOFillRequest, SpectralKNNRequest
from core.hostptr import f32_ptr, f64_ptr, i32_ptr
from core.mojolearn_shell_impl import nonzero_f64_count_host, nonzero_f64_fill_host
from core.dense_coo_device import knn_affinity_f32_device
from core.neural_context import process_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime _SPECTRAL_PREP_CONTEXT = "MojolearnCoreContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnCoreContextFastV3"


struct SpectralPreparationDevice(SpectralPreparationBackend):
    @staticmethod
    def count_wide(source: Int, count: Int) raises -> Int:
        if source == 0:
            raise Error("mojolearn: null buffer address")
        return nonzero_f64_count_host(f64_ptr(source), count)

    @staticmethod
    def fill_wide(r: SpectralCOOFillRequest) raises -> Int:
        if r.source == 0:
            raise Error("mojolearn: null buffer address")
        var source = f64_ptr(r.source)
        var rows = i32_ptr(r.row_output)
        var columns = i32_ptr(r.column_output)
        if r.value_output == 0:
            raise Error("mojolearn: null buffer address")
        var values = f32_ptr(r.value_output)
        var count = nonzero_f64_fill_host(source, r.rows, r.columns, r.capacity, rows, columns, values)
        if count < 0:
            raise Error("nonzero_f64_fill: more than " + String(r.capacity)
                + " nonzero entries; count first with nonzero_f64_count")
        return count

    @staticmethod
    def knn(r: SpectralKNNRequest) raises:
        var ctx = process_ctx[_SPECTRAL_PREP_CONTEXT]()
        knn_affinity_f32_device(ctx, r.dense, r.row_indices, r.column_indices, r.values,
            r.nonzeros, r.sparse, r.rows, r.neighbors, r.affinity, r.status)

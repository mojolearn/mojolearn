# SPDX-License-Identifier: Apache-2.0
"""Device spectral adapter: the existing numerical entries and their image context."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.spectral_native_api import (
    SpectralFitRequest, SpectralEmbeddingRequest, SpectralQueryShape,
    SPECTRAL_CAP_DENSE, SPECTRAL_CAP_RESIDENT,
)
from core.spectral_native_execution import (
    SpectralBackend, SpectralKernels, SpectralFitResult, SpectralEmbeddingResult,
)
from spectral.impl.spectral_predict_common import SpectralPredictionState, SpectralPrediction
from spectral.estimator import (
    spectral_fit_predict_dataset_host_keep, spectral_fit_predict_graph_host_keep,
    spectral_fit_predict_dense_host_keep, spectral_embedding_dataset_host,
    spectral_embedding_graph_host, spectral_embedding_dense_host, spectral_predict_host,
)
from spectral.impl.spectral_predict_resident import (
    spectral_resident_prepare, spectral_resident_predict, spectral_resident_release,
    spectral_resident_count,
)


struct SpectralDeviceBackend(SpectralBackend):
    @staticmethod
    def mode() raises -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def capabilities() raises -> Int:
        return SPECTRAL_CAP_DENSE | SPECTRAL_CAP_RESIDENT

    @staticmethod
    def dataset_copy_count(count: Int) -> Int:
        # Preserve this image's existing low-level negative-count refusal.
        return max(0, count)

    @staticmethod
    def check_query_output() -> Bool:
        return True

    @staticmethod
    def fit_dataset(r: SpectralFitRequest, source: List[Float32], mut result: SpectralFitResult) raises:
        # The non-keep entry is a wrapper calling this same function with False.
        result.columns = spectral_fit_predict_dataset_host_keep(
            source, r.input.rows, r.input.features, r.options.clusters,
            r.options.components, r.options.initializations, r.options.neighbors,
            Float32(r.options.tolerance), UInt64(r.options.seed), result.labels,
            result.embedding, result.state, r.options.keep_state,
        )

    @staticmethod
    def fit_graph(r: SpectralFitRequest, rows: List[Int32], columns: List[Int32],
                  values: List[Float32], mut result: SpectralFitResult) raises:
        result.columns = spectral_fit_predict_graph_host_keep(
            rows, columns, values, r.input.rows, r.options.clusters, r.options.components,
            r.options.initializations, r.options.neighbors, Float32(r.options.tolerance),
            UInt64(r.options.seed), result.labels, result.embedding, result.state,
            r.options.keep_state,
        )

    @staticmethod
    def fit_dense(r: SpectralFitRequest, mut result: SpectralFitResult) raises:
        result.columns = spectral_fit_predict_dense_host_keep(
            r.input.source, r.input.rows, r.options.clusters, r.options.components,
            r.options.initializations, r.options.neighbors, Float32(r.options.tolerance),
            UInt64(r.options.seed), result.labels, result.embedding, result.state,
            r.options.keep_state,
        )

    @staticmethod
    def embed_dataset(r: SpectralEmbeddingRequest, source: List[Float32], mut result: SpectralEmbeddingResult) raises:
        result.columns = spectral_embedding_dataset_host(
            source, r.input.rows, r.input.features, r.options.lanczos_components,
            r.options.neighbors, r.options.normalize_laplacian, r.options.drop_first,
            UInt64(r.options.seed), result.embedding, Float32(r.options.tolerance),
        )

    @staticmethod
    def embed_graph(r: SpectralEmbeddingRequest, rows: List[Int32], columns: List[Int32],
                    values: List[Float32], mut result: SpectralEmbeddingResult) raises:
        result.columns = spectral_embedding_graph_host(
            rows, columns, values, r.input.rows, r.options.lanczos_components,
            r.options.normalize_laplacian, r.options.drop_first, UInt64(r.options.seed),
            result.embedding, Float32(r.options.tolerance),
        )

    @staticmethod
    def embed_dense(r: SpectralEmbeddingRequest, mut result: SpectralEmbeddingResult) raises:
        result.columns = spectral_embedding_dense_host(
            r.input.source, r.input.rows, r.options.lanczos_components,
            r.options.normalize_laplacian, r.options.drop_first, UInt64(r.options.seed),
            result.embedding, Float32(r.options.tolerance),
        )

    @staticmethod
    def query(source: List[Float32], training: List[Float32], shape: SpectralQueryShape,
              state: SpectralPredictionState) raises -> SpectralPrediction:
        return spectral_predict_host(
            source, training, shape.training_rows, shape.rows, shape.features,
            shape.components, shape.clusters, shape.neighbors, shape.affinity, state,
        )

    @staticmethod
    def prepare(var training: List[Float32], shape: SpectralQueryShape,
                state: SpectralPredictionState) raises -> Int:
        return spectral_resident_prepare(training^, shape.training_rows, shape.features,
                                         shape.components, shape.clusters, shape.affinity, state)

    @staticmethod
    def query_resident(handle: Int, source: List[Float32], shape: SpectralQueryShape) raises -> SpectralPrediction:
        return spectral_resident_predict(handle, source, shape.training_rows, shape.rows,
                                         shape.features, shape.components, shape.clusters,
                                         shape.neighbors, shape.affinity)

    @staticmethod
    def release(handle: Int) raises:
        spectral_resident_release(handle)

    @staticmethod
    def resident_count() raises -> Int:
        return spectral_resident_count()


comptime SpectralDeviceKernels = SpectralKernels[SpectralDeviceBackend]

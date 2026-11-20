# SPDX-License-Identifier: Apache-2.0
"""Host spectral adapter: the existing numerical entries and their image context."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.spectral_native_api import (
    SpectralFitRequest, SpectralEmbeddingRequest, SpectralQueryShape,
    SPECTRAL_CAP_DENSE, SPECTRAL_CAP_RESIDENT,
)
from core.spectral_native_execution import (
    SpectralBackend, SpectralKernels, SpectralFitResult, SpectralEmbeddingResult,
)
from spectral.impl.spectral_predict_common import SpectralPredictionState, SpectralPrediction
from spectral.host.spectral_oracle import (
    host_spectral_fit_predict_dataset_keep, host_spectral_fit_predict_coo_keep,
    host_spectral_embedding_dataset, host_spectral_embedding_coo,
)
from spectral.host.spectral_predict_host import host_spectral_predict


struct SpectralHostBackend(SpectralBackend):
    @staticmethod
    def mode() raises -> Int:
        return Int(GLOBAL_NUMERIC_MODE)

    @staticmethod
    def capabilities() raises -> Int:
        return 0

    @staticmethod
    def dataset_copy_count(count: Int) -> Int:
        # Preserve this image's existing low-level negative-count refusal.
        return count

    @staticmethod
    def check_query_output() -> Bool:
        return False

    @staticmethod
    def fit_dataset(r: SpectralFitRequest, source: List[Float32], mut result: SpectralFitResult) raises:
        # The non-keep entry is a wrapper calling this same function with False.
        result.columns = host_spectral_fit_predict_dataset_keep(
            source, r.input.rows, r.input.features, r.options.clusters,
            r.options.components, r.options.initializations, r.options.neighbors,
            Float32(r.options.tolerance), UInt64(r.options.seed), result.labels,
            result.embedding, result.state, r.options.keep_state,
        )

    @staticmethod
    def fit_graph(r: SpectralFitRequest, rows: List[Int32], columns: List[Int32],
                  values: List[Float32], mut result: SpectralFitResult) raises:
        result.columns = host_spectral_fit_predict_coo_keep(
            rows, columns, values, r.input.rows, r.options.clusters, r.options.components,
            r.options.initializations, r.options.neighbors, Float32(r.options.tolerance),
            UInt64(r.options.seed), result.labels, result.embedding, result.state,
            r.options.keep_state,
        )

    @staticmethod
    def fit_dense(r: SpectralFitRequest, mut result: SpectralFitResult) raises:
        raise Error("Spectral native API: dense fit is unavailable in the host image")

    @staticmethod
    def embed_dataset(r: SpectralEmbeddingRequest, source: List[Float32], mut result: SpectralEmbeddingResult) raises:
        result.columns = host_spectral_embedding_dataset(
            source, r.input.rows, r.input.features, r.options.lanczos_components,
            r.options.neighbors, r.options.normalize_laplacian, r.options.drop_first,
            UInt64(r.options.seed), result.embedding, Float32(r.options.tolerance),
        )

    @staticmethod
    def embed_graph(r: SpectralEmbeddingRequest, rows: List[Int32], columns: List[Int32],
                    values: List[Float32], mut result: SpectralEmbeddingResult) raises:
        result.columns = host_spectral_embedding_coo(
            rows, columns, values, r.input.rows, r.options.lanczos_components,
            r.options.normalize_laplacian, r.options.drop_first, UInt64(r.options.seed),
            result.embedding, Float32(r.options.tolerance),
        )

    @staticmethod
    def embed_dense(r: SpectralEmbeddingRequest, mut result: SpectralEmbeddingResult) raises:
        raise Error("Spectral native API: dense embedding is unavailable in the host image")

    @staticmethod
    def query(source: List[Float32], training: List[Float32], shape: SpectralQueryShape,
              state: SpectralPredictionState) raises -> SpectralPrediction:
        return host_spectral_predict(
            source, training, shape.training_rows, shape.rows, shape.features,
            shape.components, shape.clusters, shape.neighbors, shape.affinity, state,
        )

    @staticmethod
    def prepare(var training: List[Float32], shape: SpectralQueryShape,
                state: SpectralPredictionState) raises -> Int:
        raise Error("Spectral native API: resident prediction is unavailable in the host image")

    @staticmethod
    def query_resident(handle: Int, source: List[Float32], shape: SpectralQueryShape) raises -> SpectralPrediction:
        raise Error("Spectral native API: resident prediction is unavailable in the host image")

    @staticmethod
    def release(handle: Int) raises:
        raise Error("Spectral native API: resident prediction is unavailable in the host image")

    @staticmethod
    def resident_count() raises -> Int:
        return 0


comptime SpectralHostKernels = SpectralKernels[SpectralHostBackend]

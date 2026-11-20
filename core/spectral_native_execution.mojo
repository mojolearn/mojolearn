# SPDX-License-Identifier: Apache-2.0
"""Native spectral marshalling and result ownership, shared by boundary adapters.

The backend selects the established image's numerical functions. All loads,
output room guards and copies are the existing binding work; this module adds
no solver, numerical data loop, transfer or algorithm selection threshold.
"""
from std.memory import unsafe_memcpy
from core.hostptr import f32_ptr, i32_ptr, read_f32, read_i32
from core.spectral_native_api import (
    SpectralInput, SpectralStateView, SpectralFitRequest, SpectralEmbeddingRequest,
    SpectralQueryShape, SpectralQueryRequest, SpectralPrepareRequest,
    SpectralResidentQueryRequest, SpectralNativeKernels,
    SPECTRAL_DATASET, SPECTRAL_GRAPH, SPECTRAL_DENSE,
)
from spectral.impl.spectral_predict_common import (
    SpectralPredictionState, SpectralPrediction, spectral_predict_validate,
    spectral_predict_check_state,
)


struct SpectralFitResult(Movable):
    var columns: Int
    var labels: List[Int32]
    var embedding: List[Float32]
    var state: SpectralPredictionState

    def __init__(out self):
        self.columns = 0
        self.labels = List[Int32]()
        self.embedding = List[Float32]()
        self.state = SpectralPredictionState()


struct SpectralEmbeddingResult(Movable):
    var columns: Int
    var embedding: List[Float32]

    def __init__(out self):
        self.columns = 0
        self.embedding = List[Float32]()


def spectral_guard_fit(labels: List[Int32], embedding: List[Float32], rows: Int, columns: Int) raises:
    if len(labels) != rows:
        raise Error("spectral clustering: the kernel returned " + String(len(labels))
                    + " labels for " + String(rows)
                    + " samples; the output buffer was sized for n_samples")
    if len(embedding) != rows * columns:
        raise Error("spectral clustering: the kernel returned " + String(len(embedding))
                    + " embedding floats, but the output buffer was sized for n_samples * n_components = "
                    + String(rows * columns))


def spectral_guard_embedding(embedding: List[Float32], rows: Int, columns: Int, room: Int) raises:
    if columns != room or len(embedding) != rows * room:
        raise Error("spectral embedding: the kernel returned " + String(len(embedding))
                    + " floats in " + String(columns) + " columns, but the output buffer"
                    " was sized for " + String(rows) + " x " + String(room))


def spectral_write_state(state: SpectralPredictionState, view: SpectralStateView,
                         rows: Int, components: Int, clusters: Int) raises:
    spectral_predict_check_state(state, rows, components, clusters)
    var eigenvalues = f32_ptr(view.eigenvalues)
    var eigenvectors = f32_ptr(view.eigenvectors)
    var diagonal = f32_ptr(view.diagonal)
    var centroids = f32_ptr(view.centroids)
    unsafe_memcpy(dest=eigenvalues, src=state.eigenvalues.unsafe_ptr(), count=len(state.eigenvalues))
    unsafe_memcpy(dest=eigenvectors, src=state.eigenvectors.unsafe_ptr(), count=len(state.eigenvectors))
    unsafe_memcpy(dest=diagonal, src=state.diag.unsafe_ptr(), count=len(state.diag))
    unsafe_memcpy(dest=centroids, src=state.centroids.unsafe_ptr(), count=len(state.centroids))


def spectral_read_state(view: SpectralStateView, shape: SpectralQueryShape) raises -> SpectralPredictionState:
    var state = SpectralPredictionState()
    state.eigenvalues = read_f32(view.eigenvalues, max(0, shape.components))
    state.eigenvectors = read_f32(view.eigenvectors, max(0, shape.training_rows * shape.components))
    state.diag = read_f32(view.diagonal, max(0, shape.training_rows))
    state.centroids = read_f32(view.centroids, max(0, shape.clusters * shape.components))
    return state^


def spectral_write_fit(request: SpectralFitRequest, result: SpectralFitResult) raises -> Int:
    if request.input.route == SPECTRAL_DENSE and result.columns < 0:
        return result.columns
    spectral_guard_fit(result.labels, result.embedding, request.input.rows, request.options.components)
    if request.options.keep_state:
        spectral_write_state(result.state, request.state, request.input.rows,
                             request.options.components, request.options.clusters)
    unsafe_memcpy(dest=i32_ptr(request.labels), src=result.labels.unsafe_ptr(), count=request.input.rows)
    unsafe_memcpy(dest=f32_ptr(request.embedding), src=result.embedding.unsafe_ptr(), count=len(result.embedding))
    _ = result
    return result.columns


def spectral_write_embedding(request: SpectralEmbeddingRequest,
                             result: SpectralEmbeddingResult) raises -> Int:
    if request.input.route == SPECTRAL_DENSE and result.columns < 0:
        return result.columns
    spectral_guard_embedding(result.embedding, request.input.rows, result.columns,
                             request.options.output_components)
    unsafe_memcpy(dest=f32_ptr(request.embedding), src=result.embedding.unsafe_ptr(), count=len(result.embedding))
    _ = result
    return result.columns


def spectral_write_prediction(result: SpectralPrediction, labels: Int, embedding: Int,
                              shape: SpectralQueryShape, check_output: Bool) raises:
    if check_output and (len(result.labels) != shape.rows or len(result.embedding) != shape.rows * shape.components):
        raise Error("spectral_predict: the kernel returned the wrong number of outputs")
    unsafe_memcpy(dest=i32_ptr(labels), src=result.labels.unsafe_ptr(), count=shape.rows)
    unsafe_memcpy(dest=f32_ptr(embedding), src=result.embedding.unsafe_ptr(), count=shape.rows * shape.components)
    _ = result


trait SpectralBackend:
    @staticmethod
    def mode() raises -> Int: ...
    @staticmethod
    def capabilities() raises -> Int: ...
    @staticmethod
    def dataset_copy_count(count: Int) -> Int: ...
    @staticmethod
    def check_query_output() -> Bool: ...
    @staticmethod
    def fit_dataset(request: SpectralFitRequest, source: List[Float32], mut result: SpectralFitResult) raises: ...
    @staticmethod
    def fit_graph(request: SpectralFitRequest, rows: List[Int32], columns: List[Int32],
                  values: List[Float32], mut result: SpectralFitResult) raises: ...
    @staticmethod
    def fit_dense(request: SpectralFitRequest, mut result: SpectralFitResult) raises: ...
    @staticmethod
    def embed_dataset(request: SpectralEmbeddingRequest, source: List[Float32], mut result: SpectralEmbeddingResult) raises: ...
    @staticmethod
    def embed_graph(request: SpectralEmbeddingRequest, rows: List[Int32], columns: List[Int32],
                    values: List[Float32], mut result: SpectralEmbeddingResult) raises: ...
    @staticmethod
    def embed_dense(request: SpectralEmbeddingRequest, mut result: SpectralEmbeddingResult) raises: ...
    @staticmethod
    def query(source: List[Float32], training: List[Float32], shape: SpectralQueryShape,
              state: SpectralPredictionState) raises -> SpectralPrediction: ...
    @staticmethod
    def prepare(var training: List[Float32], shape: SpectralQueryShape,
                state: SpectralPredictionState) raises -> Int: ...
    @staticmethod
    def query_resident(handle: Int, source: List[Float32], shape: SpectralQueryShape) raises -> SpectralPrediction: ...
    @staticmethod
    def release(handle: Int) raises: ...
    @staticmethod
    def resident_count() raises -> Int: ...


struct SpectralKernels[B: SpectralBackend](SpectralNativeKernels):
    @staticmethod
    def mode() raises -> Int:
        return Self.B.mode()

    @staticmethod
    def capabilities() raises -> Int:
        return Self.B.capabilities()

    @staticmethod
    def fit_result(r: SpectralFitRequest) raises -> SpectralFitResult:
        var result = SpectralFitResult()
        if r.input.route == SPECTRAL_DATASET:
            var source = read_f32(r.input.source, Self.B.dataset_copy_count(r.input.rows * r.input.features))
            _ = i32_ptr(r.labels)
            _ = f32_ptr(r.embedding)
            Self.B.fit_dataset(r, source, result)
            _ = source
        elif r.input.route == SPECTRAL_GRAPH:
            var rows = read_i32(r.input.graph_rows, max(0, r.input.nonzeros))
            var columns = read_i32(r.input.graph_columns, max(0, r.input.nonzeros))
            var values = read_f32(r.input.graph_values, max(0, r.input.nonzeros))
            _ = i32_ptr(r.labels)
            _ = f32_ptr(r.embedding)
            Self.B.fit_graph(r, rows, columns, values, result)
            _ = rows
            _ = columns
            _ = values
        elif r.input.route == SPECTRAL_DENSE:
            _ = f32_ptr(r.input.source)
            _ = i32_ptr(r.labels)
            _ = f32_ptr(r.embedding)
            Self.B.fit_dense(r, result)
        else:
            raise Error("Spectral native ABI: unknown fit input route")
        return result^

    @staticmethod
    def fit(r: SpectralFitRequest) raises -> Int:
        var result = Self.fit_result(r)
        return spectral_write_fit(r, result)

    @staticmethod
    def embedding_result(r: SpectralEmbeddingRequest) raises -> SpectralEmbeddingResult:
        var result = SpectralEmbeddingResult()
        if r.input.route == SPECTRAL_DATASET:
            var source = read_f32(r.input.source, Self.B.dataset_copy_count(r.input.rows * r.input.features))
            _ = f32_ptr(r.embedding)
            Self.B.embed_dataset(r, source, result)
            _ = source
        elif r.input.route == SPECTRAL_GRAPH:
            var rows = read_i32(r.input.graph_rows, max(0, r.input.nonzeros))
            var columns = read_i32(r.input.graph_columns, max(0, r.input.nonzeros))
            var values = read_f32(r.input.graph_values, max(0, r.input.nonzeros))
            _ = f32_ptr(r.embedding)
            Self.B.embed_graph(r, rows, columns, values, result)
            _ = rows
            _ = columns
            _ = values
        elif r.input.route == SPECTRAL_DENSE:
            _ = f32_ptr(r.input.source)
            _ = f32_ptr(r.embedding)
            Self.B.embed_dense(r, result)
        else:
            raise Error("Spectral native ABI: unknown embedding input route")
        return result^

    @staticmethod
    def embed(r: SpectralEmbeddingRequest) raises -> Int:
        var result = Self.embedding_result(r)
        return spectral_write_embedding(r, result)

    @staticmethod
    def query(r: SpectralQueryRequest) raises -> Int:
        var width = r.shape.training_rows if r.shape.affinity == 1 else r.shape.features
        var source = read_f32(r.source, max(0, r.shape.rows * width))
        var training = List[Float32]()
        if r.shape.affinity == 0:
            training = read_f32(r.training, max(0, r.shape.training_rows * r.shape.features))
        var state = spectral_read_state(r.state, r.shape)
        _ = i32_ptr(r.labels)
        _ = f32_ptr(r.embedding)
        var result = Self.B.query(source, training, r.shape, state)
        spectral_write_prediction(result, r.labels, r.embedding, r.shape, Self.B.check_query_output())
        _ = source
        _ = training
        _ = state
        return 0

    @staticmethod
    def prepare(r: SpectralPrepareRequest) raises -> Int:
        spectral_predict_validate(r.shape.training_rows, r.shape.rows, r.shape.features,
                                  r.shape.components, r.shape.clusters, r.shape.neighbors, r.shape.affinity)
        var training = List[Float32]()
        if r.shape.affinity == 0:
            training = read_f32(r.training, max(0, r.shape.training_rows * r.shape.features))
        var state = spectral_read_state(r.state, r.shape)
        return Self.B.prepare(training^, r.shape, state)

    @staticmethod
    def query_resident(r: SpectralResidentQueryRequest) raises -> Int:
        var width = r.shape.training_rows if r.shape.affinity == 1 else r.shape.features
        var source = read_f32(r.source, max(0, r.shape.rows * width))
        _ = i32_ptr(r.labels)
        _ = f32_ptr(r.embedding)
        var result = Self.B.query_resident(r.handle, source, r.shape)
        spectral_write_prediction(result, r.labels, r.embedding, r.shape, True)
        _ = source
        return 0

    @staticmethod
    def release(handle: Int) raises:
        Self.B.release(handle)

    @staticmethod
    def resident_count() raises -> Int:
        return Self.B.resident_count()

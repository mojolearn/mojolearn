# SPDX-License-Identifier: Apache-2.0
"""Typed spectral requests over the existing synchronous native descriptor ABI.

Addresses borrow caller-owned storage for the duration of the call. DATASET,
GRAPH and DENSE name existing implementations; capability bits preserve the
image's dense and resident support. Numerical validation, defaults, output
room checks and dense refusal codes remain in the original provider route.
This ABI does not select a replacement algorithm or copy tensor data.
"""
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.native_descriptor_fixed_call import native_descriptor_call_fixed

comptime SPECTRAL_NATIVE_ABI = "mojolearn.spectral.native.v1"
comptime SPECTRAL_MODE = 0
comptime SPECTRAL_CAPABILITIES = 1
comptime SPECTRAL_FIT = 2
comptime SPECTRAL_EMBED = 3
comptime SPECTRAL_QUERY = 4
comptime SPECTRAL_PREPARE = 5
comptime SPECTRAL_QUERY_RESIDENT = 6
comptime SPECTRAL_RELEASE = 7
comptime SPECTRAL_RESIDENT_COUNT = 8
comptime SPECTRAL_NATIVE_COUNT = 9
comptime SPECTRAL_CAP_DENSE = 1
comptime SPECTRAL_CAP_RESIDENT = 2
comptime SPECTRAL_DATASET = 0
comptime SPECTRAL_GRAPH = 1
comptime SPECTRAL_DENSE = 2

@fieldwise_init
struct SpectralInput(ImplicitlyCopyable, Movable):
    var source: Int
    var graph_rows: Int
    var graph_columns: Int
    var graph_values: Int
    var route: Int
    var rows: Int
    var features: Int
    var nonzeros: Int

@fieldwise_init
struct SpectralStateView(ImplicitlyCopyable, Movable):
    var eigenvalues: Int
    var eigenvectors: Int
    var diagonal: Int
    var centroids: Int

@fieldwise_init
struct SpectralFitOptions(ImplicitlyCopyable, Movable):
    var clusters: Int
    var components: Int
    var initializations: Int
    var neighbors: Int
    var tolerance: Float64
    var seed: Int
    var keep_state: Bool

@fieldwise_init
struct SpectralFitRequest(ImplicitlyCopyable, Movable):
    var input: SpectralInput
    var labels: Int
    var embedding: Int
    var state: SpectralStateView
    var options: SpectralFitOptions

@fieldwise_init
struct SpectralEmbeddingOptions(ImplicitlyCopyable, Movable):
    var lanczos_components: Int
    var output_components: Int
    var neighbors: Int
    var normalize_laplacian: Bool
    var drop_first: Bool
    var seed: Int
    var tolerance: Float64

@fieldwise_init
struct SpectralEmbeddingRequest(ImplicitlyCopyable, Movable):
    var input: SpectralInput
    var embedding: Int
    var options: SpectralEmbeddingOptions

@fieldwise_init
struct SpectralQueryShape(ImplicitlyCopyable, Movable):
    var training_rows: Int
    var rows: Int
    var features: Int
    var components: Int
    var clusters: Int
    var neighbors: Int
    var affinity: Int

@fieldwise_init
struct SpectralQueryRequest(ImplicitlyCopyable, Movable):
    var source: Int
    var training: Int
    var state: SpectralStateView
    var labels: Int
    var embedding: Int
    var shape: SpectralQueryShape

@fieldwise_init
struct SpectralPrepareRequest(ImplicitlyCopyable, Movable):
    var training: Int
    var state: SpectralStateView
    var shape: SpectralQueryShape

@fieldwise_init
struct SpectralResidentQueryRequest(ImplicitlyCopyable, Movable):
    var handle: Int
    var source: Int
    var labels: Int
    var embedding: Int
    var shape: SpectralQueryShape

trait SpectralNativeKernels:
    @staticmethod
    def mode() raises -> Int: ...
    @staticmethod
    def capabilities() raises -> Int: ...
    @staticmethod
    def resident_count() raises -> Int: ...
    @staticmethod
    def fit(request: SpectralFitRequest) raises -> Int: ...
    @staticmethod
    def embed(request: SpectralEmbeddingRequest) raises -> Int: ...
    @staticmethod
    def query(request: SpectralQueryRequest) raises -> Int: ...
    @staticmethod
    def prepare(request: SpectralPrepareRequest) raises -> Int: ...
    @staticmethod
    def query_resident(request: SpectralResidentQueryRequest) raises -> Int: ...
    @staticmethod
    def release(handle: Int) raises: ...

def spectral_native_dispatch[K: SpectralNativeKernels](operation: Int,
    aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    try:
        if result == 0:
            raise Error("Spectral native ABI: missing result span")
        var value = 0
        if operation == SPECTRAL_MODE or operation == SPECTRAL_CAPABILITIES or operation == SPECTRAL_RESIDENT_COUNT:
            if na != 0 or np != 0 or nf != 0:
                raise Error("Spectral native ABI: metadata takes no arguments")
            if operation == SPECTRAL_MODE:
                value = K.mode()
            elif operation == SPECTRAL_CAPABILITIES:
                value = K.capabilities()
            else:
                value = K.resident_count()
        elif operation == SPECTRAL_FIT:
            if na != 10 or np != 10 or nf != 1 or aa == 0 or pp == 0 or ff == 0:
                raise Error("Spectral native ABI: fit requires 10 addresses, 10 integers and 1 floats")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
            var request = SpectralFitRequest(SpectralInput(a[0], a[1], a[2], a[3], p[0], p[1], p[2], p[3]), a[4], a[5], SpectralStateView(a[6], a[7], a[8], a[9]), SpectralFitOptions(p[4], p[5], p[6], p[7], f[0], p[8], p[9] != 0))
            value = K.fit(request)
        elif operation == SPECTRAL_EMBED:
            if na != 5 or np != 10 or nf != 1 or aa == 0 or pp == 0 or ff == 0:
                raise Error("Spectral native ABI: embed requires 5 addresses, 10 integers and 1 floats")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
            var request = SpectralEmbeddingRequest(SpectralInput(a[0], a[1], a[2], a[3], p[0], p[1], p[2], p[3]), a[4], SpectralEmbeddingOptions(p[4], p[5], p[6], p[7] != 0, p[8] != 0, p[9], f[0]))
            value = K.embed(request)
        elif operation == SPECTRAL_QUERY:
            if na != 8 or np != 7 or nf != 0 or aa == 0 or pp == 0:
                raise Error("Spectral native ABI: query requires 8 addresses, 7 integers and 0 floats")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var request = SpectralQueryRequest(a[0], a[1], SpectralStateView(a[2], a[3], a[4], a[5]), a[6], a[7], SpectralQueryShape(p[0], p[1], p[2], p[3], p[4], p[5], p[6]))
            value = K.query(request)
        elif operation == SPECTRAL_PREPARE:
            if na != 5 or np != 7 or nf != 0 or aa == 0 or pp == 0:
                raise Error("Spectral native ABI: prepare requires 5 addresses, 7 integers and 0 floats")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var request = SpectralPrepareRequest(a[0], SpectralStateView(a[1], a[2], a[3], a[4]), SpectralQueryShape(p[0], p[1], p[2], p[3], p[4], p[5], p[6]))
            value = K.prepare(request)
        elif operation == SPECTRAL_QUERY_RESIDENT:
            if na != 3 or np != 8 or nf != 0 or aa == 0 or pp == 0:
                raise Error("Spectral native ABI: query_resident requires 3 addresses, 8 integers and 0 floats")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var request = SpectralResidentQueryRequest(p[0], a[0], a[1], a[2], SpectralQueryShape(p[1], p[2], p[3], p[4], p[5], p[6], p[7]))
            value = K.query_resident(request)
        elif operation == SPECTRAL_RELEASE:
            if na != 0 or np != 1 or nf != 0 or pp == 0:
                raise Error("Spectral native ABI: release requires one integer")
            K.release(Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)[])
        else:
            raise Error("Spectral native ABI: unknown operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


@fieldwise_init
struct SpectralDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def metadata(self, operation: Int) raises -> Int:
        var a: InlineArray[Int, 0] = []
        var p: InlineArray[Int, 0] = []
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, operation, a, p, f)

    def fit(self, r: SpectralFitRequest) raises -> Int:
        var a: InlineArray[Int, 10] = [r.input.source, r.input.graph_rows, r.input.graph_columns, r.input.graph_values, r.labels, r.embedding, r.state.eigenvalues, r.state.eigenvectors, r.state.diagonal, r.state.centroids]
        var p: InlineArray[Int, 10] = [r.input.route, r.input.rows, r.input.features, r.input.nonzeros, r.options.clusters, r.options.components, r.options.initializations, r.options.neighbors, r.options.seed, Int(r.options.keep_state)]
        var f: InlineArray[Float64, 1] = [r.options.tolerance]
        return native_descriptor_call_fixed(self.entry, SPECTRAL_FIT, a, p, f)

    def embed(self, r: SpectralEmbeddingRequest) raises -> Int:
        var a: InlineArray[Int, 5] = [r.input.source, r.input.graph_rows, r.input.graph_columns, r.input.graph_values, r.embedding]
        var p: InlineArray[Int, 10] = [r.input.route, r.input.rows, r.input.features, r.input.nonzeros, r.options.lanczos_components, r.options.output_components, r.options.neighbors, Int(r.options.normalize_laplacian), Int(r.options.drop_first), r.options.seed]
        var f: InlineArray[Float64, 1] = [r.options.tolerance]
        return native_descriptor_call_fixed(self.entry, SPECTRAL_EMBED, a, p, f)

    def query(self, r: SpectralQueryRequest) raises -> Int:
        var a: InlineArray[Int, 8] = [r.source, r.training, r.state.eigenvalues, r.state.eigenvectors, r.state.diagonal, r.state.centroids, r.labels, r.embedding]
        var p: InlineArray[Int, 7] = [r.shape.training_rows, r.shape.rows, r.shape.features, r.shape.components, r.shape.clusters, r.shape.neighbors, r.shape.affinity]
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, SPECTRAL_QUERY, a, p, f)

    def prepare(self, r: SpectralPrepareRequest) raises -> Int:
        var a: InlineArray[Int, 5] = [r.training, r.state.eigenvalues, r.state.eigenvectors, r.state.diagonal, r.state.centroids]
        var p: InlineArray[Int, 7] = [r.shape.training_rows, r.shape.rows, r.shape.features, r.shape.components, r.shape.clusters, r.shape.neighbors, r.shape.affinity]
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, SPECTRAL_PREPARE, a, p, f)

    def query_resident(self, r: SpectralResidentQueryRequest) raises -> Int:
        var a: InlineArray[Int, 3] = [r.source, r.labels, r.embedding]
        var p: InlineArray[Int, 8] = [r.handle, r.shape.training_rows, r.shape.rows, r.shape.features, r.shape.components, r.shape.clusters, r.shape.neighbors, r.shape.affinity]
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, SPECTRAL_QUERY_RESIDENT, a, p, f)

    def release(self, handle: Int) raises:
        var a: InlineArray[Int, 0] = []
        var p: InlineArray[Int, 1] = [handle]
        var f: InlineArray[Float64, 0] = []
        _ = native_descriptor_call_fixed(self.entry, SPECTRAL_RELEASE, a, p, f)

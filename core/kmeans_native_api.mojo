# SPDX-License-Identifier: Apache-2.0
"""Typed KMeans requests across compiled images; no Python values in RUN."""
from std.memory import bitcast
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.native_descriptor_fixed_call import native_descriptor_call_fixed

comptime KMEANS_NATIVE_ABI = "mojolearn.kmeans.native.v1"
comptime KMEANS_NATIVE_COUNT = 6
comptime KMEANS_MODE = 0
comptime KMEANS_CAPABILITIES = 1
comptime KMEANS_FIT = 2
comptime KMEANS_QUERY = 3
comptime KMEANS_PREPARE = 4
comptime KMEANS_RELEASE = 5
comptime KMEANS_CAP_JOINT = 1
comptime KMEANS_CAP_RESIDENT = 2
comptime KMEANS_CAP_FOLD = 4
comptime KMEANS_CAP_HOST = 8


struct KMeansFoldStore(ImplicitlyCopyable, Movable):
    var base: Int
    var rows: Int
    var indices: Int

    def __init__(out self, base: Int = -1, rows: Int = 0, indices: Int = -1):
        self.base = base
        self.rows = rows
        self.indices = indices

    def active(self) -> Bool:
        return self.base >= 0 and self.indices >= 0


@fieldwise_init
struct KMeansFitRequest(Copyable, Movable):
    var source: Int
    var centers: Int
    var labels: Int
    var weights: Int
    var distances: Int
    var rows: Int
    var features: Int
    var clusters: Int
    var weight_count: Int
    var max_iter: Int
    var seed: Int
    var n_init: Int
    var init: Int
    var metric: Int
    var test_finite: Bool
    var tolerance: Float64
    var oversampling: Float64
    var fold: KMeansFoldStore


@fieldwise_init
struct KMeansFitResult(Copyable, Movable):
    var inertia: Float64
    var iterations: Int
    var sum_scale: Float64
    var weight_scale: Float64
    var finite: Int
    var range_error: Int


@fieldwise_init
struct KMeansQueryRequest(Copyable, Movable):
    var source: Int
    var centers: Int
    var dst: Int
    var rows: Int
    var features: Int
    var clusters: Int
    var metric: Int
    var test_finite: Bool
    var transform: Bool
    var handle: Int
    var fold: KMeansFoldStore


@fieldwise_init
struct KMeansQueryResult(Copyable, Movable):
    var finite: Int
    var range_error: Int


trait KMeansNativeKernels:
    @staticmethod
    def mode() -> Int: ...
    @staticmethod
    def capabilities() -> Int: ...
    @staticmethod
    def fit(request: KMeansFitRequest) raises -> KMeansFitResult: ...
    @staticmethod
    def query(request: KMeansQueryRequest) raises -> KMeansQueryResult: ...
    @staticmethod
    def prepare(centers: Int, clusters: Int, features: Int) raises -> Int: ...
    @staticmethod
    def release(handle: Int) raises: ...


def kmeans_native_dispatch[K: KMeansNativeKernels](operation: Int,
    aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    try:
        if result == 0:
            raise Error("KMeans native ABI: missing result span")
        var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
        var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
        var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
        var value = 0
        if operation == KMEANS_MODE or operation == KMEANS_CAPABILITIES:
            if na != 0 or np != 0 or nf != 0:
                raise Error("KMeans native ABI: metadata takes no arguments")
            value = K.mode() if operation == KMEANS_MODE else K.capabilities()
        elif operation == KMEANS_FIT:
            if na != 6 or np != 13 or nf != 2 or aa == 0 or pp == 0 or ff == 0:
                raise Error("KMeans native ABI: fit requires 6 addresses, 13 integers and 2 floats")
            if a[5] == 0:
                raise Error("KMeans native ABI: missing fit output span")
            var answer = K.fit(KMeansFitRequest(a[0], a[1], a[2], a[3], a[4],
                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9] != 0,
                f[0], f[1], KMeansFoldStore(p[10], p[11], p[12])))
            var dst = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=a[5])
            dst[0] = bitcast[DType.int64](answer.inertia)
            dst[1] = Int64(answer.iterations)
            dst[2] = bitcast[DType.int64](answer.sum_scale)
            dst[3] = bitcast[DType.int64](answer.weight_scale)
            dst[4] = Int64(answer.finite)
            dst[5] = Int64(answer.range_error)
        elif operation == KMEANS_QUERY:
            if na != 4 or np != 10 or nf != 0 or aa == 0 or pp == 0:
                raise Error("KMeans native ABI: query requires 4 addresses and 10 integers")
            if a[3] == 0:
                raise Error("KMeans native ABI: missing query output span")
            var answer = K.query(KMeansQueryRequest(a[0], a[1], a[2], p[0], p[1], p[2], p[3],
                p[4] != 0, p[5] != 0, p[6], KMeansFoldStore(p[7], p[8], p[9])))
            var dst = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=a[3])
            dst[0] = Int64(answer.finite)
            dst[1] = Int64(answer.range_error)
        elif operation == KMEANS_PREPARE:
            if na != 1 or np != 2 or nf != 0 or aa == 0 or pp == 0:
                raise Error("KMeans native ABI: prepare requires 1 address and 2 integers")
            value = K.prepare(a[0], p[0], p[1])
        elif operation == KMEANS_RELEASE:
            if na != 0 or np != 1 or nf != 0 or pp == 0:
                raise Error("KMeans native ABI: release requires 1 integer")
            K.release(p[0])
        else:
            raise Error("KMeans native ABI: unknown operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


@fieldwise_init
struct KMeansNativeDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def metadata(self, operation: Int) raises -> Int:
        var a: InlineArray[Int, 0] = []
        var p: InlineArray[Int, 0] = []
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, operation, a, p, f)

    def fit(self, r: KMeansFitRequest) raises -> KMeansFitResult:
        var words = List[Int64](length=6, fill=0)
        var a: InlineArray[Int, 6] = [r.source, r.centers, r.labels, r.weights, r.distances, Int(words.unsafe_ptr())]
        var p: InlineArray[Int, 13] = [r.rows, r.features, r.clusters, r.weight_count, r.max_iter,
            r.seed, r.n_init, r.init, r.metric, Int(r.test_finite), r.fold.base, r.fold.rows, r.fold.indices]
        var f: InlineArray[Float64, 2] = [r.tolerance, r.oversampling]
        _ = native_descriptor_call_fixed(self.entry, KMEANS_FIT, a, p, f)
        return KMeansFitResult(bitcast[DType.float64](words[0]), Int(words[1]),
            bitcast[DType.float64](words[2]), bitcast[DType.float64](words[3]), Int(words[4]), Int(words[5]))

    def query(self, r: KMeansQueryRequest) raises -> KMeansQueryResult:
        var words = List[Int64](length=2, fill=0)
        var a: InlineArray[Int, 4] = [r.source, r.centers, r.dst, Int(words.unsafe_ptr())]
        var p: InlineArray[Int, 10] = [r.rows, r.features, r.clusters, r.metric, Int(r.test_finite),
            Int(r.transform), r.handle, r.fold.base, r.fold.rows, r.fold.indices]
        var f: InlineArray[Float64, 0] = []
        _ = native_descriptor_call_fixed(self.entry, KMEANS_QUERY, a, p, f)
        return KMeansQueryResult(Int(words[0]), Int(words[1]))

    def prepare(self, centers: Int, clusters: Int, features: Int) raises -> Int:
        var a: InlineArray[Int, 1] = [centers]
        var p: InlineArray[Int, 2] = [clusters, features]
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, KMEANS_PREPARE, a, p, f)

    def release(self, handle: Int) raises:
        var a: InlineArray[Int, 0] = []
        var p: InlineArray[Int, 1] = [handle]
        var f: InlineArray[Float64, 0] = []
        _ = native_descriptor_call_fixed(self.entry, KMEANS_RELEASE, a, p, f)

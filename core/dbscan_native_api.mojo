# SPDX-License-Identifier: Apache-2.0
"""DBSCAN fixed native protocol: existing fit, core gather and reference kernels."""
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.native_descriptor_fixed_call import native_descriptor_call_fixed

comptime DBSCAN_NATIVE_ABI = "mojolearn.dbscan.native.v1"
comptime DBSCAN_NATIVE_COUNT = 10
comptime DBSCAN_MODE = 0
comptime DBSCAN_VENDOR = 1
comptime DBSCAN_CAPABILITIES = 2
comptime DBSCAN_CAP_RESIDENT = 1
comptime DBSCAN_FIT = 3
comptime DBSCAN_CORE_COUNT = 4
comptime DBSCAN_CORE_GATHER = 5
comptime DBSCAN_PREPARE = 6
comptime DBSCAN_RELEASE = 7
comptime DBSCAN_QUERY = 8
comptime DBSCAN_RESIDENT_QUERY = 9


@fieldwise_init
struct DBSCANFitRequest(Copyable, Movable):
    var x: Int
    var labels: Int
    var weights: Int
    var core: Int
    var rows: Int
    var features: Int
    var min_samples: Int
    var budget_mb: Int
    var max_iter: Int
    var method: Int
    var metric: Int
    var eps: Float64


@fieldwise_init
struct DBSCANCoreRequest(Copyable, Movable):
    var x: Int
    var labels: Int
    var core: Int
    var rows: Int
    var features: Int


@fieldwise_init
struct DBSCANGatherRequest(Copyable, Movable):
    var x: Int
    var labels: Int
    var core: Int
    var indices: Int
    var components: Int
    var core_labels: Int
    var rows: Int
    var features: Int
    var count: Int


@fieldwise_init
struct DBSCANReferenceRequest(Copyable, Movable):
    var refs: Int
    var keys: Int
    var labels: Int
    var rows: Int
    var features: Int
    var metric: Int


@fieldwise_init
struct DBSCANQueryRequest(Copyable, Movable):
    var refs: Int
    var keys: Int
    var labels: Int
    var query: Int
    var dst: Int
    var chosen: Int
    var rows: Int
    var queries: Int
    var features: Int
    var metric: Int
    var has_threshold: Bool
    var eps: Float64


@fieldwise_init
struct DBSCANResidentQuery(Copyable, Movable):
    var query: Int
    var dst: Int
    var handle: Int
    var queries: Int
    var features: Int
    var metric: Int
    var has_threshold: Bool
    var eps: Float64


struct DBSCANFitWorkspace(Movable):
    """The private core mask was a Python Array; it is native scratch now."""
    var core: List[UInt8]

    def __init__(out self, rows: Int):
        self.core = List[UInt8](length=rows, fill=0)

    def address(mut self) -> Int:
        return Int(self.core.unsafe_ptr())


trait DBSCANNativeKernels:
    @staticmethod
    def mode() -> Int: ...
    @staticmethod
    def vendor() -> Int: ...
    @staticmethod
    def capabilities() -> Int: ...
    @staticmethod
    def fit(r: DBSCANFitRequest) raises -> Int: ...
    @staticmethod
    def core_count(r: DBSCANCoreRequest) raises -> Int: ...
    @staticmethod
    def core_gather(r: DBSCANGatherRequest) raises -> Int: ...
    @staticmethod
    def prepare(r: DBSCANReferenceRequest) raises -> Int: ...
    @staticmethod
    def release(handle: Int) raises -> Int: ...
    @staticmethod
    def query(r: DBSCANQueryRequest) raises -> Int: ...
    @staticmethod
    def resident(r: DBSCANResidentQuery) raises -> Int: ...


def dbscan_native_dispatch[K: DBSCANNativeKernels](operation: Int, aa: Int, na: Int,
    pp: Int, np: Int, ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    try:
        if result == 0:
            raise Error("DBSCAN native ABI: missing result span")
        if (na != 0 and aa == 0) or (np != 0 and pp == 0) or (nf != 0 and ff == 0):
            raise Error("DBSCAN native ABI: missing argument span")
        var value = 0
        if operation >= DBSCAN_MODE and operation <= DBSCAN_CAPABILITIES:
            if na != 0 or np != 0 or nf != 0:
                raise Error("DBSCAN native ABI: metadata takes no arguments")
            if operation == DBSCAN_MODE:
                value = K.mode()
            elif operation == DBSCAN_VENDOR:
                value = K.vendor()
            else:
                value = K.capabilities()
        elif operation == DBSCAN_FIT:
            if na != 4 or np != 7 or nf != 1:
                raise Error("DBSCAN native ABI: fit span length mismatch")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
            var request = DBSCANFitRequest(a[0], a[1], a[2], a[3], p[0], p[1], p[2], p[3], p[4], p[5], p[6], f[0])
            value = K.fit(request)
        elif operation == DBSCAN_CORE_COUNT:
            if na != 3 or np != 2 or nf != 0:
                raise Error("DBSCAN native ABI: core_count span length mismatch")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var request = DBSCANCoreRequest(a[0], a[1], a[2], p[0], p[1])
            value = K.core_count(request)
        elif operation == DBSCAN_CORE_GATHER:
            if na != 6 or np != 3 or nf != 0:
                raise Error("DBSCAN native ABI: core_gather span length mismatch")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var request = DBSCANGatherRequest(a[0], a[1], a[2], a[3], a[4], a[5], p[0], p[1], p[2])
            value = K.core_gather(request)
        elif operation == DBSCAN_PREPARE:
            if na != 3 or np != 3 or nf != 0:
                raise Error("DBSCAN native ABI: prepare span length mismatch")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var request = DBSCANReferenceRequest(a[0], a[1], a[2], p[0], p[1], p[2])
            value = K.prepare(request)
        elif operation == DBSCAN_RELEASE:
            if na != 0 or np != 1 or nf != 0:
                raise Error("DBSCAN native ABI: release span length mismatch")
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            value = K.release(p[0])
        elif operation == DBSCAN_QUERY:
            if na != 6 or np != 5 or nf != 1:
                raise Error("DBSCAN native ABI: query span length mismatch")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
            var request = DBSCANQueryRequest(a[0], a[1], a[2], a[3], a[4], a[5], p[0], p[1], p[2], p[3], Bool(p[4]), f[0])
            value = K.query(request)
        elif operation == DBSCAN_RESIDENT_QUERY:
            if na != 2 or np != 5 or nf != 1:
                raise Error("DBSCAN native ABI: resident span length mismatch")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
            var request = DBSCANResidentQuery(a[0], a[1], p[0], p[1], p[2], p[3], Bool(p[4]), f[0])
            value = K.resident(request)
        else:
            raise Error("DBSCAN native ABI: unknown operation")
        Pointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = value
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def dbscan_native_v1[K: DBSCANNativeKernels](operation: Int, aa: Int, na: Int,
    pp: Int, np: Int, ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return dbscan_native_dispatch[K](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct DBSCANProviderAPI[K: DBSCANNativeKernels](Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(DBSCAN_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(dbscan_native_v1[Self.K])


@fieldwise_init
struct DBSCANNativeDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def metadata(self, operation: Int) raises -> Int:
        var a: InlineArray[Int, 0] = []
        var p: InlineArray[Int, 0] = []
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, operation, a, p, f)

    def fit(self, r: DBSCANFitRequest) raises -> Int:
        var a: InlineArray[Int, 4] = [r.x, r.labels, r.weights, r.core]
        var p: InlineArray[Int, 7] = [r.rows, r.features, r.min_samples, r.budget_mb, r.max_iter, r.method, r.metric]
        var f: InlineArray[Float64, 1] = [r.eps]
        return native_descriptor_call_fixed(self.entry, DBSCAN_FIT, a, p, f)

    def core_count(self, r: DBSCANCoreRequest) raises -> Int:
        var a: InlineArray[Int, 3] = [r.x, r.labels, r.core]
        var p: InlineArray[Int, 2] = [r.rows, r.features]
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, DBSCAN_CORE_COUNT, a, p, f)

    def core_gather(self, r: DBSCANGatherRequest) raises -> Int:
        var a: InlineArray[Int, 6] = [r.x, r.labels, r.core, r.indices, r.components, r.core_labels]
        var p: InlineArray[Int, 3] = [r.rows, r.features, r.count]
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, DBSCAN_CORE_GATHER, a, p, f)

    def prepare(self, r: DBSCANReferenceRequest) raises -> Int:
        var a: InlineArray[Int, 3] = [r.refs, r.keys, r.labels]
        var p: InlineArray[Int, 3] = [r.rows, r.features, r.metric]
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, DBSCAN_PREPARE, a, p, f)

    def release(self, handle: Int) raises -> Int:
        var a: InlineArray[Int, 0] = []
        var p: InlineArray[Int, 1] = [handle]
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, DBSCAN_RELEASE, a, p, f)

    def query(self, r: DBSCANQueryRequest) raises -> Int:
        var a: InlineArray[Int, 6] = [r.refs, r.keys, r.labels, r.query, r.dst, r.chosen]
        var p: InlineArray[Int, 5] = [r.rows, r.queries, r.features, r.metric, Int(r.has_threshold)]
        var f: InlineArray[Float64, 1] = [r.eps]
        return native_descriptor_call_fixed(self.entry, DBSCAN_QUERY, a, p, f)

    def resident(self, r: DBSCANResidentQuery) raises -> Int:
        var a: InlineArray[Int, 2] = [r.query, r.dst]
        var p: InlineArray[Int, 5] = [r.handle, r.queries, r.features, r.metric, Int(r.has_threshold)]
        var f: InlineArray[Float64, 1] = [r.eps]
        return native_descriptor_call_fixed(self.entry, DBSCAN_RESIDENT_QUERY, a, p, f)

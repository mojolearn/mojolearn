# SPDX-License-Identifier: Apache-2.0
"""KernelDensity requests and fixed native descriptor protocol."""
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.native_descriptor_fixed_call import native_descriptor_call_fixed

comptime KDE_NATIVE_ABI = "mojolearn.kde.native.v1"
comptime KDE_NATIVE_COUNT = 7
comptime KDE_MODE = 0
comptime KDE_VENDOR = 1
comptime KDE_CAPABILITIES = 2
comptime KDE_PREPARE = 3
comptime KDE_RELEASE = 4
comptime KDE_QUERY = 5
comptime KDE_RESIDENT_QUERY = 6
comptime KDE_CAP_QUERY = 1
comptime KDE_CAP_RESIDENT = 2
comptime KDE_CAP_IMMUTABLE = 4


@fieldwise_init
struct KDEPrepareRequest(Copyable, Movable):
    var train: Int
    var weights: Int
    var rows: Int
    var features: Int
    var bandwidth: Float64
    var has_weights: Bool
    var finite_tested: Bool
    var kernel: String
    var metric: String


@fieldwise_init
struct KDEQueryRequest(Copyable, Movable):
    var train: Int
    var query: Int
    var weights: Int
    var dst: Int
    var train_rows: Int
    var query_rows: Int
    var features: Int
    var bandwidth: Float64
    var has_weights: Bool
    var want_total: Bool
    var kernel: String
    var metric: String


@fieldwise_init
struct KDEResidentQuery(Copyable, Movable):
    var handle: Int
    var query: Int
    var dst: Int
    var status: Int
    var rows: Int
    var features: Int
    var bandwidth: Float64
    var want_total: Bool
    var kernel: String
    var metric: String


trait KDENativeKernels:
    @staticmethod
    def mode() -> Int: ...
    @staticmethod
    def vendor() -> Int: ...
    @staticmethod
    def capabilities() -> Int: ...
    @staticmethod
    def prepare(request: KDEPrepareRequest) raises -> Int: ...
    @staticmethod
    def release(handle: Int) raises: ...
    @staticmethod
    def query(request: KDEQueryRequest) raises -> Float64: ...
    @staticmethod
    def resident(request: KDEResidentQuery) raises -> Float64: ...


def _kde_name(address: Int, count: Int) raises -> String:
    if count < 0 or (count != 0 and address == 0):
        raise Error("KDE native ABI: invalid name span")
    if count == 0:
        return String("")
    var bytes = Span(unsafe_ptr=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=address), length=count)
    # Length-delimited UTF-8 keeps embedded NULs and arbitrary invalid-name
    # diagnostics intact. No C-string truncation or Python name object.
    return String(from_utf8=bytes)


def kde_native_dispatch[K: KDENativeKernels](operation: Int, aa: Int, na: Int,
    pp: Int, np: Int, ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    try:
        if result == 0:
            raise Error("KDE native ABI: missing result span")
        var value = 0
        if operation >= KDE_MODE and operation <= KDE_CAPABILITIES:
            if na != 0 or np != 0 or nf != 0:
                raise Error("KDE native ABI: metadata takes no arguments")
            if operation == KDE_MODE:
                value = K.mode()
            elif operation == KDE_VENDOR:
                value = K.vendor()
            else:
                value = K.capabilities()
        elif operation == KDE_RELEASE:
            if na != 0 or np != 1 or nf != 0 or pp == 0:
                raise Error("KDE native ABI: release requires one integer")
            K.release(Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)[])
        elif operation == KDE_PREPARE:
            if na != 4 or np != 6 or nf != 1 or aa == 0 or pp == 0 or ff == 0:
                raise Error("KDE native ABI: prepare requires 4 addresses, 6 integers and one scalar")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
            var request = KDEPrepareRequest(a[0], a[1], p[0], p[1], f[0], p[2] != 0, p[3] != 0,
                _kde_name(a[2], p[4]), _kde_name(a[3], p[5]))
            value = K.prepare(request)
        elif operation == KDE_QUERY:
            if na != 7 or np != 7 or nf != 1 or aa == 0 or pp == 0 or ff == 0:
                raise Error("KDE native ABI: query requires 7 addresses, 7 integers and one scalar")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
            if a[6] == 0:
                raise Error("KDE native ABI: missing total output")
            var request = KDEQueryRequest(a[0], a[1], a[2], a[3], p[0], p[1], p[2], f[0], p[3] != 0, p[4] != 0,
                _kde_name(a[4], p[5]), _kde_name(a[5], p[6]))
            MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=a[6])[] = K.query(request)
        elif operation == KDE_RESIDENT_QUERY:
            if na != 6 or np != 6 or nf != 1 or aa == 0 or pp == 0 or ff == 0:
                raise Error("KDE native ABI: resident query requires 6 addresses, 6 integers and one scalar")
            var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
            var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
            var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
            if a[5] == 0:
                raise Error("KDE native ABI: missing total output")
            var request = KDEResidentQuery(p[0], a[0], a[1], a[2], p[1], p[2], f[0], p[3] != 0,
                _kde_name(a[3], p[4]), _kde_name(a[4], p[5]))
            MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=a[5])[] = K.resident(request)
        else:
            raise Error("KDE native ABI: unknown operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def kde_native_v1[K: KDENativeKernels](operation: Int, aa: Int, na: Int,
    pp: Int, np: Int, ff: Int, nf: Int, result: Int, context: Int,
    sink: NativeDescriptorError) abi("C") -> Int:
    return kde_native_dispatch[K](operation, aa, na, pp, np, ff, nf, result, context, sink)


struct KDEProviderAPI[K: KDENativeKernels](Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(KDE_NATIVE_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(kde_native_v1[Self.K])


@fieldwise_init
struct KDENativeDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def metadata(self, operation: Int) raises -> Int:
        var a: Array[Int, 0] = []
        var p: Array[Int, 0] = []
        var f: Array[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, operation, a, p, f)

    def prepare(self, r: KDEPrepareRequest) raises -> Int:
        var kb = r.kernel.as_bytes()
        var mb = r.metric.as_bytes()
        var a: Array[Int, 4] = [r.train, r.weights, Int(kb.unsafe_ptr()), Int(mb.unsafe_ptr())]
        var p: Array[Int, 6] = [r.rows, r.features, Int(r.has_weights), Int(r.finite_tested), len(kb), len(mb)]
        var f: Array[Float64, 1] = [r.bandwidth]
        try:
            return native_descriptor_call_fixed(self.entry, KDE_PREPARE, a, p, f)
        finally:
            _ = kb
            _ = mb
            _ = r

    def release(self, handle: Int) raises:
        var a: Array[Int, 0] = []
        var p: Array[Int, 1] = [handle]
        var f: Array[Float64, 0] = []
        _ = native_descriptor_call_fixed(self.entry, KDE_RELEASE, a, p, f)

    def query(self, r: KDEQueryRequest) raises -> Float64:
        var kb = r.kernel.as_bytes()
        var mb = r.metric.as_bytes()
        var total = List[Float64](length=1, fill=0.0)
        var a: Array[Int, 7] = [r.train, r.query, r.weights, r.dst, Int(kb.unsafe_ptr()), Int(mb.unsafe_ptr()), Int(total.unsafe_ptr())]
        var p: Array[Int, 7] = [r.train_rows, r.query_rows, r.features, Int(r.has_weights), Int(r.want_total), len(kb), len(mb)]
        var f: Array[Float64, 1] = [r.bandwidth]
        try:
            _ = native_descriptor_call_fixed(self.entry, KDE_QUERY, a, p, f)
            return total[0]
        finally:
            _ = total
            _ = kb
            _ = mb
            _ = r

    def resident(self, r: KDEResidentQuery) raises -> Float64:
        var kb = r.kernel.as_bytes()
        var mb = r.metric.as_bytes()
        var total = List[Float64](length=1, fill=0.0)
        var a: Array[Int, 6] = [r.query, r.dst, r.status, Int(kb.unsafe_ptr()), Int(mb.unsafe_ptr()), Int(total.unsafe_ptr())]
        var p: Array[Int, 6] = [r.handle, r.rows, r.features, Int(r.want_total), len(kb), len(mb)]
        var f: Array[Float64, 1] = [r.bandwidth]
        try:
            _ = native_descriptor_call_fixed(self.entry, KDE_RESIDENT_QUERY, a, p, f)
            return total[0]
        finally:
            _ = total
            _ = kb
            _ = mb
            _ = r

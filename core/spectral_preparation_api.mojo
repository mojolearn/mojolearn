# SPDX-License-Identifier: Apache-2.0
"""Native Spectral COO and precomputed-neighbor preparation requests."""
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.native_descriptor_fixed_call import native_descriptor_call_fixed
from core.hostptr import f32_ptr, i32_ptr
from core.dense_coo import nonzero_f32_count, nonzero_f32_fill

comptime SPECTRAL_PREPARATION_ABI = "mojolearn.spectral.preparation.v1"
comptime SPECTRAL_PREPARATION_COUNT = 5


@fieldwise_init
struct SpectralCOOCountRequest(ImplicitlyCopyable, Movable):
    var source: Int
    var count: Int
    var wide: Bool


@fieldwise_init
struct SpectralCOOFillRequest(ImplicitlyCopyable, Movable):
    var source: Int
    var row_output: Int
    var column_output: Int
    var value_output: Int
    var rows: Int
    var columns: Int
    var capacity: Int
    var wide: Bool


@fieldwise_init
struct SpectralKNNRequest(ImplicitlyCopyable, Movable):
    var dense: Int
    var row_indices: Int
    var column_indices: Int
    var values: Int
    var affinity: Int
    var status: Int
    var rows: Int
    var neighbors: Int
    var nonzeros: Int
    var sparse: Bool


struct SpectralAffinityStatus(Movable):
    var words: List[Int32]

    def __init__(out self):
        self.words = List[Int32](length=3, fill=0)

    def address(self) -> Int:
        return Int(self.words.unsafe_ptr())


trait SpectralPreparationBackend:
    @staticmethod
    def count_wide(source: Int, count: Int) raises -> Int: ...
    @staticmethod
    def fill_wide(request: SpectralCOOFillRequest) raises -> Int: ...
    @staticmethod
    def knn(request: SpectralKNNRequest) raises: ...


def spectral_coo_count[B: SpectralPreparationBackend](r: SpectralCOOCountRequest) raises -> Int:
    if r.count < 0:
        raise Error("nonzero_f" + String(64 if r.wide else 32) + "_count: n must be non-negative, got " + String(r.count))
    if r.count == 0:
        return 0
    if r.wide:
        return B.count_wide(r.source, r.count)
    return nonzero_f32_count(f32_ptr(r.source), r.count)


def spectral_coo_fill[B: SpectralPreparationBackend](r: SpectralCOOFillRequest) raises -> Int:
    if r.wide:
        if r.rows < 0 or r.columns < 0:
            raise Error("nonzero_f64_fill: rows and cols must be non-negative, got "
                + String(r.rows) + " x " + String(r.columns))
        if r.capacity < 0:
            raise Error("nonzero_f64_fill: capacity must be non-negative, got " + String(r.capacity))
    elif r.rows < 0 or r.columns < 0 or r.capacity < 0:
        raise Error("nonzero_f32_fill: rows, cols and capacity must be non-negative")
    if r.rows == 0 or r.columns == 0:
        return 0
    if r.wide:
        return B.fill_wide(r)
    var source = f32_ptr(r.source)
    var rows = i32_ptr(r.row_output)
    var columns = i32_ptr(r.column_output)
    var values = f32_ptr(r.value_output)
    return nonzero_f32_fill(source, r.rows, r.columns, rows, columns, values, r.capacity)


def spectral_knn_admit(r: SpectralKNNRequest) raises -> SpectralKNNRequest:
    if r.rows < 1 or r.neighbors < 0 or r.nonzeros < 0:
        raise Error("knn_affinity_f32: n must be positive, k and nnz non-negative")
    var affinity = Int(f32_ptr(r.affinity))
    var status = Int(i32_ptr(r.status))
    var dense = 0
    var rows = 0
    var columns = 0
    var values = 0
    if r.sparse:
        if r.nonzeros > 0:
            rows = Int(i32_ptr(r.row_indices))
            columns = Int(i32_ptr(r.column_indices))
            values = Int(f32_ptr(r.values))
    else:
        dense = Int(f32_ptr(r.dense))
    return SpectralKNNRequest(dense, rows, columns, values, affinity, status,
        r.rows, r.neighbors, r.nonzeros, r.sparse)


def spectral_preparation_native_v1[B: SpectralPreparationBackend](operation: Int,
    aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int, result: Int,
    context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if result == 0 or nf != 0 or aa == 0 or pp == 0:
            raise Error("Spectral preparation native ABI: missing request/output storage")
        var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
        var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
        var value = 0
        if operation == 0 or operation == 1:
            if na != 1 or np != 1:
                raise Error("Spectral preparation native ABI: count requires one address and one integer")
            value = spectral_coo_count[B](SpectralCOOCountRequest(a[0], p[0], operation == 1))
        elif operation == 2 or operation == 3:
            if na != 4 or np != 3:
                raise Error("Spectral preparation native ABI: fill requires four addresses and three integers")
            value = spectral_coo_fill[B](SpectralCOOFillRequest(a[0], a[1], a[2], a[3], p[0], p[1], p[2], operation == 3))
        elif operation == 4:
            if na != 6 or np != 4:
                raise Error("Spectral preparation native ABI: neighbors require six addresses and four integers")
            B.knn(spectral_knn_admit(SpectralKNNRequest(a[0], a[1], a[2], a[3], a[4], a[5], p[0], p[1], p[2], p[3] != 0)))
        else:
            raise Error("Spectral preparation native ABI: unknown operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = value
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


@fieldwise_init
struct SpectralPreparationDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def count(self, r: SpectralCOOCountRequest) raises -> Int:
        var a: InlineArray[Int, 1] = [r.source]
        var p: InlineArray[Int, 1] = [r.count]
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, 1 if r.wide else 0, a, p, f)

    def fill(self, r: SpectralCOOFillRequest) raises -> Int:
        var a: InlineArray[Int, 4] = [r.source, r.row_output, r.column_output, r.value_output]
        var p: InlineArray[Int, 3] = [r.rows, r.columns, r.capacity]
        var f: InlineArray[Float64, 0] = []
        return native_descriptor_call_fixed(self.entry, 3 if r.wide else 2, a, p, f)

    def knn(self, r: SpectralKNNRequest) raises:
        var a: InlineArray[Int, 6] = [r.dense, r.row_indices, r.column_indices, r.values, r.affinity, r.status]
        var p: InlineArray[Int, 4] = [r.rows, r.neighbors, r.nonzeros, Int(r.sparse)]
        var f: InlineArray[Float64, 0] = []
        _ = native_descriptor_call_fixed(self.entry, 4, a, p, f)


struct SpectralPreparationAPI[B: SpectralPreparationBackend](Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(SPECTRAL_PREPARATION_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(spectral_preparation_native_v1[Self.B])

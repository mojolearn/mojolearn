# SPDX-License-Identifier: Apache-2.0
"""KernelPCA typed options and scratch plans over the unchanged Xn operations."""
from core.neighbors_native_driver import XnDriver
from core.native_array_layout import NativeArrayLayout

comptime KPCA_PRECOMPUTED = 100


def kpca_shape(rows: Int, cols: Int, vector: Bool = False) -> List[Int]:
    var shape = List[Int](length=1, fill=rows)
    if not vector:
        shape.append(cols)
    return shape^


@fieldwise_init
struct KPCAKernelOptions(Copyable, Movable):
    var kind: Int
    var degree: Int
    var gamma: Float64
    var coef0: Float64


struct KPCAFitRoute(Copyable, Movable):
    var lanczos: Bool
    var resident: Bool
    var room: Int

    def __init__(out self, rows: Int, requested: Int, kind: Int, auto: Bool, resident_available: Bool):
        # Preserve the existing public auto/top-k route. This refactor does
        # not introduce a new threshold or promote a different algorithm.
        self.lanczos = auto and rows > 200 and requested > 0 and requested < 10
        self.resident = kind == 2 and self.lanczos and resident_available
        self.room = rows if requested < 0 else min(rows, requested)


struct KPCAScratch(Movable):
    var rows: Int
    var cols: Int
    var values: List[Float32]

    def __init__(out self, rows: Int, cols: Int) raises:
        var layout = NativeArrayLayout(kpca_shape(rows, cols), 4)
        self.rows = rows
        self.cols = cols
        self.values = List[Float32](length=layout.count, fill=Float32(0))

    def address(self) -> Int:
        return Int(self.values.unsafe_ptr()) if len(self.values) else 0


struct KPCAEigenRoom(Movable):
    var values: List[Float32]
    var vectors: List[Float32]

    def __init__(out self, rows: Int, room: Int):
        # Same rooms, zero fill and prefix copies as the prior eigen owner.
        self.values = List[Float32](length=max(rows, 1), fill=Float32(0))
        self.vectors = List[Float32](length=max(rows * room, 1), fill=Float32(0))

    def values_address(self) -> Int:
        return Int(self.values.unsafe_ptr())

    def vectors_address(self) -> Int:
        return Int(self.vectors.unsafe_ptr())


@fieldwise_init
struct KPCAAlphaRequest(Copyable, Movable):
    var vectors: Int
    var values: Int
    var dst: Int
    var rows: Int
    var count: Int
    var divide: Bool


@fieldwise_init
struct KPCAQueryRequest(Copyable, Movable):
    var source: Int
    var fitted: Int
    var columns: Int
    var total: Int
    var alphas: Int
    var dst: Int
    var rows: Int
    var fitted_rows: Int
    var features: Int
    var count: Int
    var options: KPCAKernelOptions


@fieldwise_init
struct KPCANeighborDriver(Copyable, Movable):
    var native: XnDriver

    def kernel(self, source: Int, other: Int, dst: Int, rows: Int, cols: Int, features: Int,
               options: KPCAKernelOptions) raises:
        var a: InlineArray[Int, 3] = [source, other, dst]
        var p: InlineArray[Int, 5] = [rows, cols, features, options.kind, options.degree]
        var f: InlineArray[Float64, 2] = [Float64(Float32(options.gamma)), Float64(Float32(options.coef0))]
        _ = self.native.run("kernel", a, p, f)

    def colsum(self, source: Int, dst: Int, rows: Int, cols: Int) raises:
        var a: InlineArray[Int, 2] = [source, dst]
        var p: InlineArray[Int, 2] = [rows, cols]
        var f: InlineArray[Float64, 0] = []
        _ = self.native.run("colsum", a, p, f)

    def rowsum(self, source: Int, dst: Int, rows: Int, cols: Int) raises:
        var a: InlineArray[Int, 2] = [source, dst]
        var p: InlineArray[Int, 2] = [rows, cols]
        var f: InlineArray[Float64, 0] = []
        _ = self.native.run("rowsum", a, p, f)

    def divide(self, source: Int, dst: Int, count: Int, divisor: Float64) raises:
        var a: InlineArray[Int, 2] = [source, dst]
        var p: InlineArray[Int, 1] = [count]
        var f: InlineArray[Float64, 1] = [divisor]
        _ = self.native.run("scale_div", a, p, f)

    def center(self, source: Int, columns: Int, rows_mean: Int, total: Int, dst: Int,
               rows: Int, cols: Int) raises:
        var a: InlineArray[Int, 5] = [source, columns, rows_mean, total, dst]
        var p: InlineArray[Int, 2] = [rows, cols]
        var f: InlineArray[Float64, 0] = []
        _ = self.native.run("kpca_center", a, p, f)

    def alpha(self, r: KPCAAlphaRequest) raises:
        var a: InlineArray[Int, 3] = [r.vectors, r.values, r.dst]
        var p: InlineArray[Int, 3] = [r.rows, r.count, Int(r.divide)]
        var f: InlineArray[Float64, 0] = []
        _ = self.native.run("kpca_alpha_scale", a, p, f)

    def matmul(self, left: Int, right: Int, dst: Int, rows: Int, width: Int, cols: Int) raises:
        var a: InlineArray[Int, 3] = [left, right, dst]
        var p: InlineArray[Int, 3] = [rows, width, cols]
        var f: InlineArray[Float64, 0] = []
        _ = self.native.run("matmul", a, p, f)

    def fused_query(self, r: KPCAQueryRequest) raises:
        var pre = r.options.kind == KPCA_PRECOMPUTED
        var a: InlineArray[Int, 6] = [r.source, r.source if pre else r.fitted,
            r.columns, r.total, r.alphas, r.dst]
        var p: InlineArray[Int, 6] = [r.rows, r.fitted_rows, r.features, r.count, r.options.kind, r.options.degree]
        var f: InlineArray[Float64, 3] = [0.0 if pre else Float64(Float32(r.options.gamma)),
            0.0 if pre else Float64(Float32(r.options.coef0)), Float64(r.fitted_rows)]
        _ = self.native.run("kpca_transform", a, p, f)


struct KPCACenteredQuery(Movable):
    var centered: KPCAScratch

    def __init__(out self, driver: KPCANeighborDriver, r: KPCAQueryRequest) raises:
        # The old-items route is retained as its original sequence. All
        # private host stores are native owners, with unchanged Xn copies.
        var kernel: Optional[KPCAScratch] = None
        var address = r.source
        if r.options.kind != KPCA_PRECOMPUTED:
            kernel = Optional[KPCAScratch](KPCAScratch(r.rows, r.fitted_rows))
            address = kernel.value().address()
            driver.kernel(r.source, r.fitted, address, r.rows, r.fitted_rows, r.features, r.options)
        var sums = KPCAScratch(r.rows, 1)
        driver.rowsum(address, sums.address(), r.rows, r.fitted_rows)
        var means = KPCAScratch(r.rows, 1)
        driver.divide(sums.address(), means.address(), r.rows, Float64(r.fitted_rows))
        var centered = KPCAScratch(r.rows, r.fitted_rows)
        driver.center(address, r.columns, means.address(), r.total, centered.address(), r.rows, r.fitted_rows)
        _ = kernel
        _ = sums
        _ = means
        self.centered = centered^

    def finish(self, driver: KPCANeighborDriver, alphas: Int, dst: Int, count: Int) raises:
        driver.matmul(self.centered.address(), alphas, dst, self.centered.rows, self.centered.cols, count)
        _ = self.centered

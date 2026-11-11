# SPDX-License-Identifier: Apache-2.0
"""OneClassSVM native storage stages over the existing x_neighbors operations.

The boundary controls admission/publication stages. No kernel, solver, stopping
rule or reduction is implemented here; fixed native requests preserve the
existing compaction, Gram, resident, fused and unfused routes.
"""
from std.memory import memcpy
from core.native_array_layout import NativeArrayLayout
from core.neighbors_native_driver import XnDriver
from core.admitted_weights import admitted_fill
from core.ocsvm_resident_api import OcResidentDriver, OcResidentRequest


def oc_f32_address(values: List[Float32]) -> Int:
    return Int(values.unsafe_ptr()) if len(values) else 0


def oc_i32_address(values: List[Int32]) -> Int:
    return Int(values.unsafe_ptr()) if len(values) else 0


def oc_count(rows: Int, columns: Int) raises -> Int:
    var shape = NativeArrayLayout([rows, columns], 4)
    return shape.count


struct OcMatrix(Movable):
    var rows: Int
    var columns: Int
    var borrowed: Int
    var storage: List[Float32]

    def __init__(out self, rows: Int, columns: Int, borrowed: Int = 0, allocate: Bool = False) raises:
        self.rows = rows
        self.columns = columns
        self.borrowed = borrowed
        self.storage = List[Float32](length=oc_count(rows, columns), fill=0) if allocate else List[Float32]()

    def address(self) -> Int:
        return self.borrowed if self.borrowed else oc_f32_address(self.storage)


def oc_take_rows(driver: XnDriver, x: Int, source_rows: Int, columns: Int,
    indices: Int, count: Int) raises -> OcMatrix:
    var result = OcMatrix(count, columns, 0, True)
    oc_take_rows_into(driver, x, source_rows, columns, indices, count, result.address())
    _ = result
    return result^


def oc_take_rows_into(driver: XnDriver, x: Int, source_rows: Int, columns: Int,
    indices: Int, count: Int, dst: Int) raises:
    var a: Array[Int, 3] = [x, indices, dst]
    var p: Array[Int, 3] = [count, columns, source_rows]
    var f: Array[Float64, 0] = []
    _ = driver.run("take_rows", a, p, f)


def oc_take_cols(driver: XnDriver, x: Int, rows: Int, source_columns: Int,
    indices: Int, count: Int) raises -> OcMatrix:
    var result = OcMatrix(rows, count, 0, True)
    var a: Array[Int, 3] = [x, indices, result.address()]
    var p: Array[Int, 3] = [rows, source_columns, count]
    var f: Array[Float64, 0] = []
    _ = driver.run("take_cols", a, p, f)
    _ = result
    return result^


def oc_kernel(driver: XnDriver, lhs: Int, rhs: Int, rows: Int, references: Int,
    features: Int, kernel: Int, degree: Int, gamma: Float64, coef0: Float64) raises -> OcMatrix:
    var result = OcMatrix(rows, references, 0, True)
    var a: Array[Int, 3] = [lhs, rhs, result.address()]
    var p: Array[Int, 5] = [rows, references, features, kernel, degree]
    var f: Array[Float64, 2] = [Float64(Float32(gamma)), Float64(Float32(coef0))]
    _ = driver.run("kernel", a, p, f)
    _ = result
    return result^


def oc_variance(driver: XnDriver, source: Int, count: Int) raises -> Float64:
    var result = List[Float32](length=1, fill=0)
    var a: Array[Int, 2] = [source, Int(result.unsafe_ptr())]
    var p: Array[Int, 1] = [count]
    var f: Array[Float64, 0] = []
    _ = driver.run("variance", a, p, f)
    return Float64(result[0])


struct OcWeights(Movable):
    var rows: List[Int32]
    var values: List[Float32]
    var count: Int
    var selected: Bool

    def __init__(out self):
        self.rows = List[Int32]()
        self.values = List[Float32]()
        self.count = 0
        self.selected = False

    def unit(mut self, count: Int):
        self.rows = List[Int32]()
        self.values = List[Float32](length=count, fill=0)
        admitted_fill[DType.float32](MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(self.values.unsafe_ptr())), count, Float32(1))
        self.count = count
        self.selected = False
        _ = self

    def compact(mut self, driver: XnDriver, source: Int, count: Int) raises:
        var rows = List[Int32](length=count, fill=0)
        var values = List[Float32](length=count, fill=0)
        var info = List[Int32](length=1, fill=0)
        var a: Array[Int, 4] = [source, oc_i32_address(rows), oc_f32_address(values), Int(info.unsafe_ptr())]
        var p: Array[Int, 1] = [count]
        var f: Array[Float64, 0] = []
        _ = driver.run("pos_compact", a, p, f)
        var kept = Int(info[0])
        if kept == 0:
            raise Error("Invalid input - all samples have zero or negative weights.")
        # Preserve the two former private prefix copies, in their order.
        self.rows = List[Int32](length=kept, fill=0)
        memcpy(dest=self.rows.unsafe_ptr(), src=rows.unsafe_ptr(), count=kept)
        self.values = List[Float32](length=kept, fill=0)
        memcpy(dest=self.values.unsafe_ptr(), src=values.unsafe_ptr(), count=kept)
        self.count = kept
        self.selected = True
        _ = rows
        _ = values
        _ = info


struct OcFitState(Movable):
    var count: Int
    var alpha: List[Float32]
    var info: List[Float32]
    var iterations: List[Int32]
    var support_rows: List[Int32]
    var support_values: List[Float32]
    var support_info: List[Int32]
    var supports: Int

    def __init__(out self, count: Int):
        self.count = count
        self.alpha = List[Float32](length=count, fill=0)
        self.info = List[Float32]()
        self.iterations = List[Int32]()
        self.support_rows = List[Int32]()
        self.support_values = List[Float32]()
        self.support_info = List[Int32]()
        self.supports = 0

    def initialize(mut self, driver: XnDriver, bounds: Int, nu: Float64) raises:
        var a: Array[Int, 2] = [bounds, oc_f32_address(self.alpha)]
        var p: Array[Int, 1] = [self.count]
        var f: Array[Float64, 1] = [nu]
        _ = driver.run("ocsvm_alpha_init", a, p, f)
        _ = self

    def prepare_metadata(mut self):
        # The old info/iteration allocations followed alpha initialization.
        self.info = List[Float32](length=1, fill=0)
        self.iterations = List[Int32](length=1, fill=0)

    def solve_gram(mut self, driver: XnDriver, gram: Int, bounds: Int,
        cap: Int, tolerance: Float64) raises:
        var a: Array[Int, 5] = [gram, bounds, oc_f32_address(self.alpha),
            oc_f32_address(self.info), oc_i32_address(self.iterations)]
        var p: Array[Int, 2] = [self.count, cap]
        var f: Array[Float64, 1] = [Float64(Float32(tolerance))]
        _ = driver.run("ocsvm", a, p, f)
        _ = self

    def solve_resident(mut self, driver: OcResidentDriver, x: Int, bounds: Int,
        features: Int, kernel: Int, degree: Int, cap: Int, fast: Bool,
        gamma: Float64, coef0: Float64, tolerance: Float64) raises:
        driver.run(OcResidentRequest(x, bounds, oc_f32_address(self.alpha), oc_f32_address(self.info),
            oc_i32_address(self.iterations), self.count, features, kernel, degree, cap, fast,
            gamma, Float64(Float32(coef0)), Float64(Float32(tolerance))))
        _ = self

    def solve_descriptor(mut self, driver: XnDriver, x: Int, bounds: Int,
        features: Int, kernel: Int, degree: Int, cap: Int, gamma: Float64,
        coef0: Float64, tolerance: Float64) raises:
        # The original candidate remains on its original resident descriptor.
        var a: Array[Int, 5] = [x, bounds, oc_f32_address(self.alpha),
            oc_f32_address(self.info), oc_i32_address(self.iterations)]
        var p: Array[Int, 5] = [self.count, features, kernel, degree, cap]
        var f: Array[Float64, 3] = [gamma, Float64(Float32(coef0)), Float64(Float32(tolerance))]
        _ = driver.run("ocsvm_resident", a, p, f)
        _ = self

    def compact_support(mut self, driver: XnDriver) raises:
        self.support_rows = List[Int32](length=max(self.count, 1), fill=0)
        self.support_values = List[Float32](length=max(self.count, 1), fill=0)
        self.support_info = List[Int32](length=1, fill=0)
        if self.count > 0:
            var a: Array[Int, 4] = [oc_f32_address(self.alpha), oc_i32_address(self.support_rows),
                oc_f32_address(self.support_values), oc_i32_address(self.support_info)]
            var p: Array[Int, 1] = [self.count]
            var f: Array[Float64, 0] = []
            _ = driver.run("pos_compact", a, p, f)
        self.supports = Int(self.support_info[0]) if self.count > 0 else 0
        _ = self


def oc_kernel_product(driver: XnDriver, query: Int, references: Int, coefficients: Int, dst: Int,
    rows: Int, supports: Int, features: Int, kernel: Int, degree: Int,
    gamma: Float64, coef0: Float64) raises:
    var a: Array[Int, 4] = [query, references, coefficients, dst]
    var p: Array[Int, 6] = [rows, supports, features, 1, kernel, degree]
    var f: Array[Float64, 2] = [Float64(Float32(gamma)), Float64(Float32(coef0))]
    _ = driver.run("kernel_matmul", a, p, f)


def oc_fused_labels(driver: XnDriver, query: Int, references: Int, coefficients: Int, dst: Int,
    rows: Int, supports: Int, features: Int, kernel: Int, degree: Int,
    gamma: Float64, coef0: Float64, offset: Float64) raises:
    var a: Array[Int, 4] = [query, references, coefficients, dst]
    var p: Array[Int, 6] = [rows, supports, features, kernel, degree, 2]
    var f: Array[Float64, 5] = [Float64(Float32(gamma)), Float64(Float32(coef0)), 1, -offset, 0]
    _ = driver.run("ocsvm_labels", a, p, f)


def oc_product(driver: XnDriver, gram: Int, coefficients: Int, dst: Int, rows: Int, supports: Int) raises:
    var a: Array[Int, 3] = [gram, coefficients, dst]
    var p: Array[Int, 3] = [rows, supports, 1]
    var f: Array[Float64, 0] = []
    _ = driver.run("matmul", a, p, f)


def oc_decision(driver: XnDriver, source: Int, dst: Int, count: Int, offset: Float64) raises:
    var a: Array[Int, 2] = [source, dst]
    var p: Array[Int, 2] = [count, 6]
    var f: Array[Float64, 2] = [1, -offset]
    _ = driver.run("unary", a, p, f)


def oc_sign(driver: XnDriver, source: Int, dst: Int, count: Int) raises:
    var a: Array[Int, 2] = [source, dst]
    var p: Array[Int, 2] = [count, 2]
    var f: Array[Float64, 1] = [0]
    _ = driver.run("p2m_sign_label", a, p, f)


struct OcQueryState(Movable):
    var count: Int
    var labels: Bool
    var external: Int
    var scores: List[Float32]
    var label_words: List[Int32]

    def __init__(out self, count: Int, labels: Bool, allocate: Bool) raises:
        self.count = count
        self.labels = labels
        self.external = 0
        self.scores = List[Float32]()
        self.label_words = List[Int32]()
        if allocate:
            var size = oc_count(count, 1)
            if labels:
                self.label_words = List[Int32](length=size, fill=0)
            else:
                self.scores = List[Float32](length=size, fill=0)

    def address(self) -> Int:
        if self.external:
            return self.external
        return oc_i32_address(self.label_words) if self.labels else oc_f32_address(self.scores)

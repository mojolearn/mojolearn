# SPDX-License-Identifier: Apache-2.0
"""Native LocalOutlierFactor scratch and staged execution plans.

Public fit arrays and requested results are borrowed output descriptors. All
private distances, indices, query densities and decision buffers are Mojo
values. Stages keep finite refusal and public field reads at their old boundary
positions. The existing fused/tiled/unfused operations do all data computation.
"""
from std.math import floor
from core.neighbors_native_driver import XnDriver
from core.admitted_cast_elements import admitted_cast_elements
from core.native_array_layout import NativeArrayLayout
from checks.numerics import pinned_mul_f64


def lof_f32_address(values: List[Float32]) -> Int:
    # Empty Array scratch used address zero. Preserve the descriptor's null
    # refusal instead of depending on an empty List's allocation policy.
    return Int(values.unsafe_ptr()) if len(values) else 0


def lof_i32_address(values: List[Int32]) -> Int:
    return Int(values.unsafe_ptr()) if len(values) else 0


def lof_power_steps(count: Int) -> Tuple[Int, Int]:
    var padded = 1
    var bits = 0
    while padded < count:  # metadata: at most 63 power-of-two doublings
        padded *= 2
        bits += 1
    return (padded, bits * (bits + 1) // 2)


def lof_matrix_count(rows: Int, columns: Int) raises -> Int:
    # Reuse Array's shape/overflow refusal before replacing its private
    # storage with List. Mutable fitted k still reaches these same checks.
    var layout = NativeArrayLayout([rows, columns], 4)
    return layout.count


struct LofPlan(Movable):
    var rows: Int
    var reference_rows: Int
    var features: Int
    var neighbors: Int
    var exclude_self: Bool
    var unfused: Bool
    var old_items: Bool
    var squared: List[Float32]
    var indices: List[Int32]
    var distances: List[Float32]
    var densities: List[Float32]
    var scores: List[Float32]
    var decisions: List[Float32]

    def __init__(out self, rows: Int, reference_rows: Int, features: Int,
        neighbors: Int, exclude_self: Bool, unfused: Bool = False, old_items: Bool = False):
        self.rows = rows
        self.reference_rows = reference_rows
        self.features = features
        self.neighbors = neighbors
        self.exclude_self = exclude_self
        self.unfused = unfused
        self.old_items = old_items
        self.squared = List[Float32]()
        self.indices = List[Int32]()
        self.distances = List[Float32]()
        self.densities = List[Float32]()
        self.scores = List[Float32]()
        self.decisions = List[Float32]()

    def select(mut self, driver: XnDriver, query: Int, reference: Int) raises -> Int:
        var status = List[Int32](length=1, fill=0)
        var f: Array[Float64, 0] = []
        if self.unfused:
            var matrix = List[Float32](length=lof_matrix_count(self.rows, self.reference_rows), fill=0)
            var a: Array[Int, 3] = [query, reference, lof_f32_address(matrix)]
            var p: Array[Int, 4] = [self.rows, self.reference_rows, self.features, Int(status.unsafe_ptr())]
            _ = driver.run("sqdist", a, p, f)
            self.squared = List[Float32](length=lof_matrix_count(self.rows, self.neighbors), fill=0)
            self.indices = List[Int32](length=self.rows * self.neighbors, fill=0)
            var aa: Array[Int, 3] = [lof_f32_address(matrix), lof_f32_address(self.squared), lof_i32_address(self.indices)]
            var pp: Array[Int, 4] = [self.rows, self.reference_rows, self.neighbors, Int(self.exclude_self)]
            _ = driver.run("knn_select", aa, pp, f)
            _ = matrix^
        else:
            self.squared = List[Float32](length=lof_matrix_count(self.rows, self.neighbors), fill=0)
            self.indices = List[Int32](length=self.rows * self.neighbors, fill=0)
            var a: Array[Int, 4] = [query, reference, lof_f32_address(self.squared), lof_i32_address(self.indices)]
            var p: Array[Int, 6] = [self.rows, self.reference_rows, self.features,
                self.neighbors, Int(self.exclude_self), Int(status.unsafe_ptr())]
            _ = driver.run("knn_sq" if self.old_items else "knn_sq_tiled", a, p, f)
        return Int(status[0])

    def roots(mut self, driver: XnDriver, dst: Int) raises:
        var a: Array[Int, 2] = [lof_f32_address(self.squared), dst]
        var p: Array[Int, 2] = [self.rows * self.neighbors, 2]
        var f: Array[Float64, 2] = [1.0, 0.0]
        _ = driver.run("unary", a, p, f)
        self.squared = List[Float32]()

    def query_neighbors(mut self, driver: XnDriver, query: Int, reference: Int) raises -> Int:
        var status = self.select(driver, query, reference)
        self.distances = List[Float32](length=self.rows * self.neighbors, fill=0)
        var distance_address = lof_f32_address(self.distances)
        self.roots(driver, distance_address)
        return status

    def density(self, driver: XnDriver, distances: Int, fit_distances: Int, dst: Int, fit_rows: Int) raises:
        var a: Array[Int, 4] = [distances, lof_i32_address(self.indices), fit_distances, dst]
        var p: Array[Int, 3] = [self.neighbors, self.rows, fit_rows]
        var f: Array[Float64, 0] = []
        _ = driver.run("lof_lrd", a, p, f)

    def score(self, driver: XnDriver, fit_densities: Int, densities: Int, dst: Int, fit_rows: Int) raises:
        var a: Array[Int, 4] = [lof_i32_address(self.indices), fit_densities, densities, dst]
        var p: Array[Int, 3] = [self.neighbors, self.rows, fit_rows]
        var f: Array[Float64, 0] = []
        _ = driver.run("lof_score", a, p, f)

    def query_density(mut self, driver: XnDriver, fit_distances: Int, fit_rows: Int) raises:
        self.densities = List[Float32](length=self.rows, fill=0)
        self.density(driver, lof_f32_address(self.distances), fit_distances, lof_f32_address(self.densities), fit_rows)

    def query_score(mut self, driver: XnDriver, fit_densities: Int, fit_rows: Int) raises:
        self.scores = List[Float32](length=self.rows, fill=0)
        self.score(driver, fit_densities, lof_f32_address(self.densities), lof_f32_address(self.scores), fit_rows)

    def decision(self, driver: XnDriver, dst: Int, shift: Float64) raises:
        var a: Array[Int, 2] = [lof_f32_address(self.scores), dst]
        var p: Array[Int, 2] = [self.rows, 6]
        var f: Array[Float64, 2] = [1.0, shift]
        _ = driver.run("unary", a, p, f)

    def query_decision(mut self, driver: XnDriver, shift: Float64) raises:
        self.decisions = List[Float32](length=self.rows, fill=0)
        self.decision(driver, lof_f32_address(self.decisions), shift)


def lof_percentile(driver: XnDriver, score: Int, count: Int, q: Float64) raises -> Float64:
    """Existing row-sort and linear two-order-statistic interpolation."""
    var offsets: List[Int32] = [0, Int32(count)]
    var columns = List[Int32](length=count, fill=0)
    var sorted_columns = List[Int32](length=count, fill=0)
    var sorted_scores = List[Float32](length=count, fill=0)
    var network = lof_power_steps(count)
    var a: Array[Int, 5] = [Int(offsets.unsafe_ptr()), lof_i32_address(columns), score,
        lof_i32_address(sorted_columns), lof_f32_address(sorted_scores)]
    var p: Array[Int, 4] = [1, count, network[0], network[1]]
    var f: Array[Float64, 0] = []
    _ = driver.run("p2m_row_sort", a, p, f)
    _ = offsets^
    _ = columns^
    _ = sorted_columns^
    var position = pinned_mul_f64(q / 100.0, Float64(count - 1))
    var lo = Int(floor(position))
    var hi = min(lo + 1, count - 1)
    var t = position - Float64(lo)
    var low = Float64(sorted_scores[lo])
    var high = Float64(sorted_scores[hi])
    var difference = high - low
    if t >= 0.5:
        return high - pinned_mul_f64(difference, 1.0 - t)
    return low + pinned_mul_f64(difference, t)


def lof_offset(driver: XnDriver, score: Int, count: Int, contamination: Float64) raises -> Float64:
    var q = pinned_mul_f64(100.0, contamination)
    return Float64(Float32(lof_percentile(driver, score, count, q)))


struct LofLabelsPlan(Movable):
    var labels: List[Int32]

    def __init__(out self, count: Int):
        self.labels = List[Int32](length=count, fill=0)

    def classify(mut self, driver: XnDriver, score: Int, mode: Int, threshold: Float64) raises:
        if len(self.labels) == 0:
            return
        var a: Array[Int, 2] = [score, lof_i32_address(self.labels)]
        var p: Array[Int, 2] = [len(self.labels), mode]
        var f: Array[Float64, 1] = [threshold]
        _ = driver.run("p2m_sign_label", a, p, f)

    def write_i64(self, dst: Int) raises:
        if admitted_cast_elements(lof_i32_address(self.labels), 2, dst, 3, len(self.labels)) != 0:
            raise Error("TypeError: mojolearn: an element does not fit <i8")

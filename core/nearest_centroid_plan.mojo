# SPDX-License-Identifier: Apache-2.0
"""Typed NearestCentroid orchestration over the existing neighbor kernels.

Public buffers are borrowed descriptors; only private scratch lives here. Each
request uses fixed native words. The fused/split kernel choices are unchanged.
"""
from std.math import isinf
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_fixed_call import native_descriptor_call_fixed
from core.admitted_cast_elements import admitted_cast_elements
from core.array_helpers_impl import fsum_f64
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from x_neighbors.descriptor_schema import xn_operation_id

comptime _F32 = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64 = MutPointer[Float64, MutUntrackedOrigin]
comptime _I64 = Pointer[Int64, ImmUntrackedOrigin]


def nc_scratch_address(values: List[Float32]) -> Int:
    # Private Array scratch exposed address zero for an empty buffer. Keep
    # that descriptor contract independent of Mojo's empty List pointer.
    return Int(values.unsafe_ptr()) if len(values) else 0


def nc_close(a: Float64, b: Float64) -> Bool:
    if a == b:
        return True
    if isinf(a) or isinf(b):
        return False
    var difference = abs(b - a)
    return difference <= abs(1e-5 * b) or difference <= abs(1e-5 * a) or difference <= 1e-8


def nc_power_steps(count: Int) -> Tuple[Int, Int]:
    var padded = 1
    var bits = 0
    while padded < count:  # metadata: at most 63 power-of-two doublings
        padded *= 2
        bits += 1
    return (padded, bits * (bits + 1) // 2)


def nc_uniform(prior: Int, count: Int) raises -> Bool:
    if count == 0:
        return True
    var p = _F64(unsafe_from_address=prior)
    var tasks = host_predict_task_count_work(count, 1)
    var chunk = host_predict_chunk(count, tasks)
    var flags = List[Int](length=tasks, fill=0)
    var fp = flags.unsafe_ptr()
    var target = 1.0 / Float64(count)
    def check(worker: Int) {imm p, imm count, imm chunk, imm fp, imm target}:
        var end = min(count, (worker + 1) * chunk)
        for i in range(worker * chunk, end):
            if not nc_close(p[i], target):
                fp[worker] = 1
                break
    host_parallelize(check, tasks)
    var result = True
    for task in range(tasks):  # metadata: one result per worker
        result = result and flags[task] == 0
    return result


def nc_narrow_priors(src: Int, dst: Int, count: Int) raises:
    if count == 0:
        return
    var source = _F64(unsafe_from_address=src)
    var target = _F32(unsafe_from_address=dst)
    var tasks = host_predict_task_count_work(count, 1)
    var chunk = host_predict_chunk(count, tasks)
    def narrow(worker: Int) {imm source, imm target, imm count, imm chunk}:
        for i in range(worker * chunk, min(count, (worker + 1) * chunk)):
            target[i] = Float32(source[i])
    host_parallelize(narrow, tasks)


def nc_normalize_priors(address: Int, count: Int) raises:
    """The existing negative check, Shewchuk fsum and optional rescale.

    The fsum dependency is serial by its arithmetic definition; independent
    validation and rescaling ranges run across the host workers.
    """
    if count == 0:
        raise Error("ZeroDivisionError: float division by zero")
    var p = _F64(unsafe_from_address=address)
    var tasks = host_predict_task_count_work(count, 1)
    var chunk = host_predict_chunk(count, tasks)
    var flags = List[Int](length=tasks, fill=0)
    var fp = flags.unsafe_ptr()
    def check(worker: Int) {imm p, imm count, imm chunk, imm fp}:
        for i in range(worker * chunk, min(count, (worker + 1) * chunk)):
            if p[i] < 0:
                fp[worker] = 1
                break
    host_parallelize(check, tasks)
    for task in range(tasks):  # metadata: one refusal summary per worker
        if flags[task]:
            raise Error("priors must be non-negative")
    var total = List[Float64](length=1, fill=0)
    var status = fsum_f64(p, _F64(unsafe_from_address=Int(total.unsafe_ptr())), count)
    var value = total[0]
    if status != 0:
        if value != 0.0:
            raise Error("OverflowError: math range error")
        raise Error("-inf + inf in fsum")
    if nc_close(value, 1.0):
        return
    if value == 0.0:
        raise Error("ZeroDivisionError: float division by zero")
    def rescale(worker: Int) {imm p, imm count, imm chunk, imm value}:
        for i in range(worker * chunk, min(count, (worker + 1) * chunk)):
            p[i] = p[i] / value
    host_parallelize(rescale, tasks)


def nc_class_offsets(counts: Int, count: Int) raises -> List[Int32]:
    """Parallel block prefix of the existing integer class-count metadata."""
    var result = List[Int32](length=count + 1, fill=0)
    if count == 0:
        return result^
    var source = _I64(unsafe_from_address=counts)
    var dst = result.unsafe_ptr()
    var tasks = host_predict_task_count_work(count, 1)
    var chunk = host_predict_chunk(count, tasks)
    var sums = List[Int64](length=tasks, fill=0)
    var sp = sums.unsafe_ptr()
    def sum_block(worker: Int) {imm source, imm count, imm chunk, imm sp}:
        var total = Int64(0)
        for i in range(worker * chunk, min(count, (worker + 1) * chunk)):
            total += source[i]
        sp[worker] = total
    host_parallelize(sum_block, tasks)
    var offset = Int64(0)
    for task in range(tasks):  # metadata: prefix of worker totals
        var total = sums[task]
        sums[task] = offset
        offset += total
    def write_block(worker: Int) {imm source, imm dst, imm count, imm chunk, imm sp}:
        var total = sp[worker]
        for i in range(worker * chunk, min(count, (worker + 1) * chunk)):
            total += source[i]
            dst[i + 1] = Int32(total)
    host_parallelize(write_block, tasks)
    _ = sums^
    return result^


@fieldwise_init
struct NcDriver(Copyable, Movable):
    var entry: NativeDescriptorCall

    def run[NA: Int, NP: Int, NF: Int](self, name: String,
        a: InlineArray[Int, NA], p: InlineArray[Int, NP], f: InlineArray[Float64, NF],
    ) raises -> Int:
        return native_descriptor_call_fixed(self.entry, xn_operation_id(name), a, p, f)


struct NcFitPlan(Movable):
    var rows: Int
    var features: Int
    var classes: Int
    var euclidean: Bool
    var split: Bool
    var counts: List[Float32]
    var wide_counts: List[Int64]
    var centroids: List[Float32]
    var dataset_centroid: List[Float32]

    def __init__(out self, rows: Int, features: Int, classes: Int, euclidean: Bool, split: Bool):
        self.rows = rows
        self.features = features
        self.classes = classes
        self.euclidean = euclidean
        self.split = split
        self.counts = List[Float32](length=classes, fill=0)
        self.wide_counts = List[Int64](length=classes, fill=0)
        self.centroids = List[Float32]()
        self.dataset_centroid = List[Float32]()

    def count(mut self, driver: NcDriver, labels: Int) raises:
        var info = List[Int32](length=1, fill=0)
        var a: InlineArray[Int, 3] = [labels, Int(self.counts.unsafe_ptr()), Int(info.unsafe_ptr())]
        var p: InlineArray[Int, 2] = [self.rows, self.classes]
        var f: InlineArray[Float64, 0] = []
        _ = driver.run("p2m_class_counts", a, p, f)
        _ = admitted_cast_elements(Int(self.counts.unsafe_ptr()), 0, Int(self.wide_counts.unsafe_ptr()), 3, self.classes)
        _ = info^

    def statistics(mut self, driver: NcDriver, x: Int, labels: Int, stats: Int) raises -> Int:
        # Priors have already validated and class_prior_ has been published.
        # Rejected priors never allocate the potentially large centroid scratch.
        self.centroids = List[Float32](length=self.classes * self.features, fill=0)
        var f: InlineArray[Float64, 0] = []
        if self.euclidean:
            if self.split:
                var a: InlineArray[Int, 3] = [x, labels, nc_scratch_address(self.centroids)]
                var p: InlineArray[Int, 3] = [self.rows, self.features, self.classes]
                _ = driver.run("group_mean", a, p, f)
        else:
            var offsets = nc_class_offsets(Int(self.wide_counts.unsafe_ptr()), self.classes)
            var steps = nc_power_steps(self.rows)
            var a: InlineArray[Int, 4] = [x, labels, Int(offsets.unsafe_ptr()), nc_scratch_address(self.centroids)]
            var p: InlineArray[Int, 5] = [self.rows, self.features, self.classes, steps[0], steps[1]]
            _ = driver.run("nc_median", a, p, f)
            _ = offsets^
        var status = List[Int32](length=1, fill=0)
        var p: InlineArray[Int, 4] = [self.rows, self.features, self.classes, Int(status.unsafe_ptr())]
        if self.euclidean and not self.split:
            self.dataset_centroid = List[Float32](length=self.features, fill=0)
            var a: InlineArray[Int, 6] = [x, labels, Int(self.counts.unsafe_ptr()),
                nc_scratch_address(self.centroids), stats, nc_scratch_address(self.dataset_centroid)]
            _ = driver.run("nc_stats", a, p, f)
        else:
            var a: InlineArray[Int, 4] = [x, labels, nc_scratch_address(self.centroids), stats]
            _ = driver.run("nc_std", a, p, f)
        return Int(status[0])

    def median_std(self, driver: NcDriver, stats: Int) raises -> Tuple[Float64, Bool]:
        var result = List[Float32](length=2, fill=0)
        var steps = nc_power_steps(self.features)
        var a: InlineArray[Int, 2] = [stats, Int(result.unsafe_ptr())]
        var p: InlineArray[Int, 3] = [self.features, steps[0], steps[1]]
        var f: InlineArray[Float64, 0] = []
        _ = driver.run("nc_med_std", a, p, f)
        return (Float64(result[0]), result[1] != 0)

    def constant(self, driver: NcDriver, x: Int) raises -> Bool:
        var flag = List[Int32](length=1, fill=0)
        var a: InlineArray[Int, 2] = [x, Int(flag.unsafe_ptr())]
        var p: InlineArray[Int, 2] = [self.rows, self.features]
        var f: InlineArray[Float64, 0] = []
        _ = driver.run("p2m_const_cols", a, p, f)
        return flag[0] == 0

    def shrink(mut self, driver: NcDriver, x: Int, stats: Int, dst: Int, deviations: Int,
               median: Float64, threshold: Float64) raises:
        var fused = self.euclidean and not self.split
        var a: InlineArray[Int, 6] = [nc_scratch_address(self.dataset_centroid) if fused else x,
            nc_scratch_address(self.centroids), Int(self.counts.unsafe_ptr()), stats, dst, deviations]
        var p: InlineArray[Int, 4] = [self.rows, self.features, self.classes, 1 if threshold != 0 else 0]
        var f: InlineArray[Float64, 2] = [median, threshold]
        _ = driver.run("nc_shrink_d" if fused else "nc_shrink", a, p, f)


@fieldwise_init
struct NcQueryPlan(Copyable, Movable):
    var query: Int
    var centroids: Int
    var std: Int
    var prior: Int
    var rows: Int
    var features: Int
    var classes: Int

    def decision(self, driver: NcDriver, dst: Int, status: Int = 0) raises:
        var a: InlineArray[Int, 5] = [self.query, self.centroids, self.std, self.prior, dst]
        var f: InlineArray[Float64, 0] = []
        if status:
            var p: InlineArray[Int, 4] = [self.rows, self.features, self.classes, status]
            _ = driver.run("nc_decision", a, p, f)
        else:
            var p: InlineArray[Int, 3] = [self.rows, self.features, self.classes]
            _ = driver.run("nc_decision", a, p, f)

    def fused(self, driver: NcDriver, name: String, dst: Int, mode: Int) raises -> Int:
        var status = List[Int32](length=1, fill=0)
        var a: InlineArray[Int, 5] = [self.query, self.centroids, self.std, self.prior, dst]
        var p: InlineArray[Int, 5] = [self.rows, self.features, self.classes, mode, Int(status.unsafe_ptr())]
        var f: InlineArray[Float64, 0] = []
        _ = driver.run(name, a, p, f)
        return Int(status[0])

    def distance_labels(self, driver: NcDriver, dst: Int, euclidean: Bool) raises:
        var distances = List[Float32](length=self.rows * self.classes, fill=0)
        var selected = List[Float32](length=self.rows, fill=0)
        var f: InlineArray[Float64, 0] = []
        var a: InlineArray[Int, 3] = [self.query, self.centroids, nc_scratch_address(distances)]
        var p: InlineArray[Int, 3] = [self.rows, self.classes, self.features]
        _ = driver.run("sqdist" if euclidean else "l1dist", a, p, f)
        var aa: InlineArray[Int, 3] = [nc_scratch_address(distances), nc_scratch_address(selected), dst]
        var pp: InlineArray[Int, 4] = [self.rows, self.classes, 1, 0]
        _ = driver.run("knn_select", aa, pp, f)
        _ = selected^
        _ = distances^

    def score_labels(self, driver: NcDriver, dst: Int) raises:
        var scores = List[Float32](length=self.rows * self.classes, fill=0)
        self.decision(driver, nc_scratch_address(scores))
        if self.rows:
            var a: InlineArray[Int, 2] = [nc_scratch_address(scores), dst]
            var p: InlineArray[Int, 2] = [self.rows, self.classes]
            var f: InlineArray[Float64, 0] = []
            _ = driver.run("row_argmax", a, p, f)
        _ = scores^

    def probabilities(self, driver: NcDriver, dst: Int, mode: Int) raises:
        var scores = List[Float32](length=self.rows * self.classes, fill=0)
        self.decision(driver, nc_scratch_address(scores))
        var a: InlineArray[Int, 2] = [nc_scratch_address(scores), dst]
        var p: InlineArray[Int, 2] = [self.rows, self.classes]
        var f: InlineArray[Float64, 0] = []
        _ = driver.run("log_softmax" if mode == 1 else ("softmax64" if mode == 2 else "softmax"), a, p, f)
        _ = scores^


def nc_binary_decision(driver: NcDriver, scores: Int, dst: Int, rows: Int) raises:
    var signs = List[Float32](length=2, fill=1)
    signs[0] = -1
    var a: InlineArray[Int, 3] = [scores, Int(signs.unsafe_ptr()), dst]
    var p: InlineArray[Int, 3] = [rows, 2, 1]
    var f: InlineArray[Float64, 0] = []
    _ = driver.run("matmul", a, p, f)
    _ = signs^

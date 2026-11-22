# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""HDBSCAN fit, approximate_predict and soft-membership relations through the
typed entries (ported from the arms `arm_fit`, `arm_predict` and `arm_soft` of
`python/mojolearn/tests/test_hdbscan_surface.py`).

Data. The Python file drew two uniform blobs of 60 rows in [0, 0.5)^2 and
[6, 6.5)^2 and a duplicated grid (40 + 40 rows on a 1/16 grid, each row twice)
from NumPy PCG64. Every relation here holds for any data of that shape, so the
rows are regenerated with a counter-based splitmix64 uniform (`_uniform`).

Relations already gated elsewhere are not repeated here:
`checks/native_tests/hdbscan_native.mojo` (typed entries, both columns) holds
the planted two-cluster partition, exact core distances, the leaf selection
switch, a far query predicted as noise with probability 0, exemplar queries
predicting their label, and one-hot exemplar rows of all_points; the kernel
gates of `hdbscan/checks/hdbscan_check.mojo` (`pixi run check-hdbscan`) hold
labels, outlier count and core distances against the oracle and the leaf rule.

Bit relations (repeat fits, rows alone, batch_size, reversed query order) are
the `identical` contract; the check columns build `identical`.
"""
from std.sys.compile import is_defined
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases
from std.math import isfinite
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.kmeans_variants_api import VariantsSpan
from core.hdbscan_fit_api import (
    HdbscanFitDriver,
    HdbscanFitOptions,
    HdbscanFitRequest,
    HdbscanFitPlan,
    HdbscanPredictionRequest,
    HdbscanPredictionPlan,
    HDBSCAN_FIT_HOST,
)
from core.hdbscan_query_api import (
    HdbscanQueryDriver,
    HdbscanQueryDimensions,
    HdbscanQueryBuffers,
    HdbscanQueryRequest,
    HdbscanQueryPlan,
    HDBSCAN_QUERY_HOST,
    HDBSCAN_QUERY_APPROXIMATE,
    HDBSCAN_QUERY_MEMBERSHIP,
    HDBSCAN_QUERY_ALL_POINTS,
)
from hdbscan.native_fit_host import hdbscan_fit_host_v1
from hdbscan.native_query_host import hdbscan_query_host_v1
from hdbscan.native_fit_device import hdbscan_fit_device_v1
from hdbscan.native_query_device import hdbscan_query_device_v1


# The column this binary was built for (`-D MOJOLEARN_COLUMN_CPU=1` builds the host column).
comptime CPU_COLUMN_BUILD = is_defined["MOJOLEARN_COLUMN_CPU"]()


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


comptime HDBSCAN_SURFACE_CASES = 7
comptime SELECTION_EOM = 0
comptime METRIC_L2 = 1
comptime DEFAULT_BATCH = 4096
comptime BLOB_HALF = 60
comptime DUPE_HALF = 40


# ---- data --------------------------------------------------------------

def _uniform(seed: UInt64, index: Int) -> Float32:
    """splitmix64 of (seed, index), top 24 bits as a float in [0, 1)."""
    var z = seed * UInt64(0xD1B54A32D192ED03) + UInt64(index + 1) * UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    z = z ^ (z >> 31)
    return (z >> 40).cast[DType.float32]() * Float32(1.0 / 16777216.0)


def _blobs() -> List[Float32]:
    """120 x 2: rows 0-59 uniform in [0, 0.5)^2, rows 60-119 in [6, 6.5)^2."""
    var n = 2 * BLOB_HALF
    var x = List[Float32](length=n * 2, fill=0)
    var xp = x.unsafe_ptr()
    def cell(i: Int) {imm xp}:
        var shift = Float32(6.0) if i >= BLOB_HALF * 2 else Float32(0.0)
        xp[i] = _uniform(0, i) * Float32(0.5) + shift
    host_parallelize(cell, n * 2)
    return x^


def _dupes() -> List[Float32]:
    """160 x 2: [a, a, b, b], a on the 1/16 grid of [0, 0.5]^2, b = a' + 5;
    every row duplicated and k-NN distances tie many ways."""
    var n = 4 * DUPE_HALF
    var x = List[Float32](length=n * 2, fill=0)
    var xp = x.unsafe_ptr()
    def cell(i: Int) {imm xp}:
        var row = i // 2
        var col = i % 2
        var block = row // DUPE_HALF  # 0, 1 -> a; 2, 3 -> b
        var r = row % DUPE_HALF
        var source = (0 if block < 2 else 2 * DUPE_HALF) + r * 2 + col
        var k = Int(_uniform(2, source) * Float32(8.0) + Float32(0.5))
        xp[i] = Float32(k) / Float32(16.0) + (Float32(5.0) if block >= 2 else Float32(0.0))
    host_parallelize(cell, n * 2)
    return x^


def _reversed_rows(x: List[Float32], n: Int, d: Int) -> List[Float32]:
    var y = List[Float32](length=n * d, fill=0)
    var xp = x.unsafe_ptr()
    var yp = y.unsafe_ptr()
    def row(i: Int) {imm xp, imm yp, imm n, imm d}:
        for j in range(d):  # metadata: two features per row
            yp[i * d + j] = xp[(n - 1 - i) * d + j]
    host_parallelize(row, n)
    _ = x
    return y^


def _row_slice(x: List[Float32], start: Int, rows: Int, d: Int) -> List[Float32]:
    var y = List[Float32](length=rows * d, fill=0)
    for i in range(rows * d):  # metadata: one or two query rows copied
        y[i] = x[start * d + i]
    return y^


def _same_f32(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):  # metadata: verdict over a check output
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return False
    return True


def _same_i32(a: List[Int32], b: List[Int32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):  # metadata: verdict over a check output
        if a[i] != b[i]:
            return False
    return True


# ---- one fit through the typed fit entry ---------------------------------

struct Fitted(Movable):
    """One HDBSCAN fit, plus its prediction data when asked, as the public
    door (`bindings/hdbscan_fit_boundary_pack.mojo` `hdbscan_fit` and
    `_hdbscan_prediction_data`) builds them."""
    var n: Int
    var d: Int
    var neighbors: Int
    var source: List[Float32]
    var labels: List[Int32]
    var core: List[Float32]
    var probabilities: List[Float32]
    var fit: HdbscanFitPlan
    var clusters: Int
    var edges: Int
    var condensed: Int
    var exemplars: Int
    var deaths: List[Float32]
    var selected: List[Int32]
    var offsets: List[Int32]
    var index_into: List[Int32]
    var prediction: HdbscanPredictionPlan

    def __init__(out self, driver: HdbscanFitDriver, var source: List[Float32], n: Int, d: Int,
                 min_samples: Int, min_cluster_size: Int, method: Int, prediction_data: Bool) raises:
        self.n = n
        self.d = d
        self.neighbors = min_samples
        self.source = source^
        self.labels = List[Int32](length=n, fill=-9)
        self.core = List[Float32](length=n, fill=-9)
        self.probabilities = List[Float32](length=n, fill=-9)
        self.fit = HdbscanFitPlan(prediction_data)
        self.clusters = 0
        self.edges = 0
        self.condensed = 0
        self.exemplars = 0
        self.deaths = List[Float32]()
        self.selected = List[Int32]()
        self.offsets = List[Int32]()
        self.index_into = List[Int32]()
        self.prediction = HdbscanPredictionPlan(n if prediction_data else 0)
        var options = HdbscanFitOptions(n, d, min_samples, min_cluster_size, 0, False,
            method, METRIC_L2, prediction_data, 1.0, 0.0)
        var request = HdbscanFitRequest(VariantsSpan(Int(self.source.unsafe_ptr()), n * d),
            VariantsSpan(Int(self.labels.unsafe_ptr()), n), VariantsSpan(Int(self.core.unsafe_ptr()), n),
            VariantsSpan(Int(self.probabilities.unsafe_ptr()), n), options.copy())
        if prediction_data:
            self.fit.storage[0].prepare_tree(n)
        if driver.metadata(HDBSCAN_FIT_HOST) != 0:
            self.fit.capture_host_source(request)
        self.clusters = driver.fit(request, self.fit)
        self.fit.storage[0].host_source = List[Float32]()
        if not prediction_data:
            return
        self.edges = Int(self.fit.storage[0].info[4])
        self.condensed = Int(self.fit.storage[0].info[3])
        var c = self.clusters
        self.deaths = List[Float32](length=self.condensed, fill=0)
        self.selected = List[Int32](length=max(c, 1), fill=0)
        self.offsets = List[Int32](length=c + 1, fill=0)
        self.index_into = List[Int32](length=self.edges + 1, fill=0)
        var t = self.edges
        var preparation = HdbscanPredictionRequest(VariantsSpan(Int(self.labels.unsafe_ptr()), n),
            VariantsSpan(Int(self.fit.storage[0].tree_parents.unsafe_ptr()), t),
            VariantsSpan(Int(self.fit.storage[0].tree_children.unsafe_ptr()), t),
            VariantsSpan(Int(self.fit.storage[0].tree_lambdas.unsafe_ptr()), t),
            VariantsSpan(Int(self.fit.storage[0].tree_sizes.unsafe_ptr()), t),
            VariantsSpan(Int(self.fit.storage[0].inverse.unsafe_ptr()), c),
            VariantsSpan(Int(self.deaths.unsafe_ptr()), self.condensed),
            VariantsSpan(Int(self.selected.unsafe_ptr()), c),
            VariantsSpan(Int(self.offsets.unsafe_ptr()), c + 1),
            VariantsSpan(Int(self.index_into.unsafe_ptr()), t + 1), n, t, self.condensed, c)
        self.prediction.storage[0].capture(preparation)
        self.exemplars = driver.prediction(preparation, self.prediction)
        self.prediction.storage[0].release_inputs()

    def outliers(self) -> Int:
        return Int(self.fit.storage[0].info[1])


@fieldwise_init
struct Approx(Movable):
    var labels: List[Int32]
    var probabilities: List[Float32]


def _query(driver: HdbscanQueryDriver, f: Fitted, operation: Int, query: List[Float32], nq: Int,
           batch: Int, mut labels_out: List[Int32], mut values_out: List[Float32]) raises:
    """One query through the typed query entry, with the spans the public
    door (`providers/python/hdbscan_public_lifecycle_pyapi_unpack.mojo` `hdbscan_query_buffers`) passes."""
    var host = driver.metadata(HDBSCAN_QUERY_HOST) != 0
    var approximate = operation == HDBSCAN_QUERY_APPROXIMATE
    var c = f.clusters
    var t = f.edges
    var query_span = VariantsSpan(0, 0) if operation == HDBSCAN_QUERY_ALL_POINTS else VariantsSpan(Int(query.unsafe_ptr()), nq * f.d)
    var output_span = VariantsSpan(Int(labels_out.unsafe_ptr()), nq) if approximate else VariantsSpan(Int(values_out.unsafe_ptr()), nq * c)
    var probability_span = VariantsSpan(Int(values_out.unsafe_ptr()), nq) if approximate else VariantsSpan(0, 0)
    var buffers = HdbscanQueryBuffers(VariantsSpan(Int(f.source.unsafe_ptr()), f.n * f.d),
        VariantsSpan(Int(f.core.unsafe_ptr()), f.n), VariantsSpan(Int(f.labels.unsafe_ptr()), f.n),
        VariantsSpan(Int(f.fit.storage[0].tree_parents.unsafe_ptr()), t),
        VariantsSpan(Int(f.fit.storage[0].tree_lambdas.unsafe_ptr()), t),
        VariantsSpan(Int(f.deaths.unsafe_ptr()), f.condensed),
        VariantsSpan(Int(f.selected.unsafe_ptr()), c),
        VariantsSpan(Int(f.index_into.unsafe_ptr()), t + 1),
        VariantsSpan(Int(f.prediction.storage[0].exemplars.unsafe_ptr()), f.exemplars),
        VariantsSpan(Int(f.offsets.unsafe_ptr()), c + 1),
        query_span, output_span, probability_span)
    var dimensions = HdbscanQueryDimensions(f.n, f.d, t, f.condensed, f.exemplars, c, f.neighbors)
    var request = HdbscanQueryRequest(dimensions.copy(), buffers.copy(), nq, 0, batch)
    var plan = HdbscanQueryPlan(request, operation, host)
    var consumed = driver.run(plan)
    _ = plan
    _ = query
    _ = labels_out
    _ = values_out
    require(consumed == nq, "query returns its consumed row count")


def approximate(driver: HdbscanQueryDriver, f: Fitted, query: List[Float32], nq: Int) raises -> Approx:
    var labels = List[Int32](length=nq, fill=-9)
    var probabilities = List[Float32](length=nq, fill=-9)
    _query(driver, f, HDBSCAN_QUERY_APPROXIMATE, query, nq, 1, labels, probabilities)
    return Approx(labels^, probabilities^)


def membership(driver: HdbscanQueryDriver, f: Fitted, query: List[Float32], nq: Int, batch: Int) raises -> List[Float32]:
    var unused = List[Int32]()
    var values = List[Float32](length=nq * f.clusters, fill=-9)
    _query(driver, f, HDBSCAN_QUERY_MEMBERSHIP, query, nq, batch, unused, values)
    return values^


def all_points(driver: HdbscanQueryDriver, f: Fitted, batch: Int) raises -> List[Float32]:
    var unused = List[Int32]()
    var none = List[Float32]()
    var values = List[Float32](length=f.n * f.clusters, fill=-9)
    _query(driver, f, HDBSCAN_QUERY_ALL_POINTS, none, f.n, batch, unused, values)
    return values^


# ---- soft-membership row facts ------------------------------------------

def _row_sum64(v: List[Float32], row: Int, c: Int) -> Float64:
    var s = Float64(0)
    for j in range(c):  # metadata: one row's cluster columns
        s += Float64(v[row * c + j])
    return s


def _row_sum32(v: List[Float32], row: Int, c: Int) -> Float32:
    var s = Float32(0)
    for j in range(c):  # metadata: one row's cluster columns
        s += v[row * c + j]
    return s


def _row_argmax(v: List[Float32], row: Int, c: Int) -> Int:
    var best = 0
    for j in range(1, c):  # metadata: one row's cluster columns (first maximum wins, as numpy)
        if v[row * c + j] > v[row * c + best]:
            best = j
    return best


def _soft_rows(v: List[Float32], f: Fitted, where: String) raises:
    """Finite, non-negative, row sums (float64) at most 1 + 1e-5, and on two
    planted blobs a clustered training row's largest entry is its own label."""
    var c = f.clusters
    for i in range(len(v)):  # metadata: verdict over a check output
        require(isfinite(v[i]) and v[i] >= 0, where + ": finite and non-negative (cell " + String(i) + ")")
    for row in range(f.n):  # metadata: verdict over a check output
        require(_row_sum64(v, row, c) <= 1.0 + 1e-5, where + ": row " + String(row) + " sums to at most 1")
        if f.labels[row] >= 0:
            require(_row_argmax(v, row, c) == Int(f.labels[row]),
                where + ": clustered training row " + String(row) + " is most likely its own label")


# ---- the cases ------------------------------------------------------------

def _case_fit(fit_driver: HdbscanFitDriver, query_driver: HdbscanQueryDriver) raises:
    var n = 2 * BLOB_HALF
    # FIT: n_outliers equals the -1 count, the tree counts are carried,
    # and two fits of the same input agree bit for bit.
    var m = Fitted(fit_driver, _blobs(), n, 2, 8, 8, SELECTION_EOM, False)
    require(m.clusters == 2, "two planted blobs give two clusters, got " + String(m.clusters))
    var noise = 0
    for i in range(n):  # metadata: verdict over a check output
        if m.labels[i] == -1:
            noise += 1
    require(m.outliers() == noise, "n_outliers " + String(m.outliers()) + " counts the -1 labels " + String(noise))
    require(Int(m.fit.storage[0].info[2]) >= 1 and Int(m.fit.storage[0].info[3]) >= 1,
        "Boruvka rounds and condensed cluster count carried")
    var m2 = Fitted(fit_driver, _blobs(), n, 2, 8, 8, SELECTION_EOM, False)
    require(_same_i32(m.labels, m2.labels) and _same_f32(m.core, m2.core),
        "two fits of the same input agree bit for bit")


def _case_prediction_data_fit(fit_driver: HdbscanFitDriver, query_driver: HdbscanQueryDriver) raises:
    var n = 2 * BLOB_HALF
    # prediction_data=True fits the same labels and core distance bytes.
    var plain = Fitted(fit_driver, _blobs(), n, 2, 8, 8, SELECTION_EOM, False)
    var m = Fitted(fit_driver, _blobs(), n, 2, 8, 8, SELECTION_EOM, True)
    require(_same_i32(plain.labels, m.labels) and _same_f32(plain.core, m.core),
        "prediction_data=True fits the same labels and core distance bytes")


def _case_approximate(fit_driver: HdbscanFitDriver, query_driver: HdbscanQueryDriver) raises:
    var n = 2 * BLOB_HALF
    # approximate_predict on the training rows.
    var x = _blobs()
    var m = Fitted(fit_driver, x.copy(), n, 2, 8, 8, SELECTION_EOM, True)
    require(m.clusters == 2, "two planted blobs give two clusters")
    var a = approximate(query_driver, m, x, n)
    for i in range(n):  # metadata: verdict over a check output
        var p = a.probabilities[i]
        require(p >= 0 and p <= 1, "probability in [0, 1] (row " + String(i) + ")")
        if a.labels[i] == -1:
            require(p == 0, "a -1 label has probability 0 (row " + String(i) + ")")
        require(a.labels[i] == m.labels[i], "training row " + String(i) + " predicts its fitted label")
    var again = approximate(query_driver, m, x, n)
    require(_same_i32(again.labels, a.labels) and _same_f32(again.probabilities, a.probabilities),
        "two calls agree bit for bit")
    for i in range(5):  # metadata: five single-row queries
        var one = approximate(query_driver, m, _row_slice(x, i, 1, 2), 1)
        require(one.labels[0] == a.labels[i]
            and bitcast[DType.uint32](one.probabilities[0]) == bitcast[DType.uint32](a.probabilities[i]),
            "row " + String(i) + " alone equals its row in the batch")
    _ = x


def _case_approximate_dupes(fit_driver: HdbscanFitDriver, query_driver: HdbscanQueryDriver) raises:
    # Duplicated rows and tied distances: reversing the query order
    # reverses the approximate_predict answer bytes.
    var nd = 4 * DUPE_HALF
    var dx = _dupes()
    var dm = Fitted(fit_driver, dx.copy(), nd, 2, 4, 6, SELECTION_EOM, True)
    require(dm.clusters >= 1, "the duplicated fixture finds a cluster")
    var d1 = approximate(query_driver, dm, dx, nd)
    var d2 = approximate(query_driver, dm, _reversed_rows(dx, nd, 2), nd)
    for i in range(nd):  # metadata: verdict over a check output
        require(d1.labels[nd - 1 - i] == d2.labels[i]
            and bitcast[DType.uint32](d1.probabilities[nd - 1 - i]) == bitcast[DType.uint32](d2.probabilities[i]),
            "reversed query row " + String(i) + " reverses the answer bytes")
    _ = dx


def _case_membership(fit_driver: HdbscanFitDriver, query_driver: HdbscanQueryDriver) raises:
    var n = 2 * BLOB_HALF
    # membership_vector on the training rows and on far points.
    var x = _blobs()
    var m = Fitted(fit_driver, x.copy(), n, 2, 8, 8, SELECTION_EOM, True)
    require(m.clusters == 2, "two planted blobs give two clusters")
    var c = m.clusters
    var mv = membership(query_driver, m, x, n, DEFAULT_BATCH)
    _soft_rows(mv, m, "membership_vector")
    var kept_min = Float64(1e30)
    for row in range(n):  # metadata: verdict over a check output
        if m.labels[row] >= 0:
            kept_min = min(kept_min, _row_sum64(mv, row, c))
    var far: List[Float32] = [1000.0, -1000.0, -500.0, 700.0]
    var fv = membership(query_driver, m, far, 2, DEFAULT_BATCH)
    for row in range(2):  # metadata: two far queries
        var fs = _row_sum64(fv, row, c)
        for j in range(c):  # metadata: one row's cluster columns
            require(isfinite(fv[row * c + j]), "far row finite")
        require(fs < 0.05 and fs < kept_min,
            "far point " + String(row) + " row sum " + String(fs) + " below 0.05 and below every clustered row (" + String(kept_min) + ")")
    var again = membership(query_driver, m, x, n, DEFAULT_BATCH)
    require(_same_f32(again, mv), "two membership_vector calls agree bit for bit")
    for i in range(5):  # metadata: five single-row queries
        var one = membership(query_driver, m, _row_slice(x, i, 1, 2), 1, DEFAULT_BATCH)
        for j in range(c):  # metadata: one row's cluster columns
            require(bitcast[DType.uint32](one[j]) == bitcast[DType.uint32](mv[i * c + j]),
                "membership row " + String(i) + " alone equals its row in the batch")
    var small = membership(query_driver, m, x, n, 7)
    require(_same_f32(small, mv), "membership_vector batch_size=7 changes no byte")
    _ = x


def _case_all_points(fit_driver: HdbscanFitDriver, query_driver: HdbscanQueryDriver) raises:
    var n = 2 * BLOB_HALF
    # all_points_membership_vectors.
    var m = Fitted(fit_driver, _blobs(), n, 2, 8, 8, SELECTION_EOM, True)
    require(m.clusters == 2, "two planted blobs give two clusters")
    var ap = all_points(query_driver, m, DEFAULT_BATCH)
    _soft_rows(ap, m, "all_points_membership_vectors")
    var ap_small = all_points(query_driver, m, 13)
    require(_same_f32(ap_small, ap), "all_points batch_size=13 changes no byte")


def _case_dupes_soft(fit_driver: HdbscanFitDriver, query_driver: HdbscanQueryDriver) raises:
    # Duplicated rows (zero distances, FLT_MAX lambdas) through both soft
    # calls: finite rows, sums at most 1, reversed order reverses the bytes.
    var nd = 4 * DUPE_HALF
    var dx = _dupes()
    var dm = Fitted(fit_driver, dx.copy(), nd, 2, 4, 6, SELECTION_EOM, True)
    require(dm.clusters >= 1, "the duplicated fixture finds a cluster")
    var c = dm.clusters
    var d1 = membership(query_driver, dm, dx, nd, DEFAULT_BATCH)
    var d2 = membership(query_driver, dm, _reversed_rows(dx, nd, 2), nd, DEFAULT_BATCH)
    var dap = all_points(query_driver, dm, DEFAULT_BATCH)
    for i in range(nd * c):  # metadata: verdict over a check output
        require(isfinite(d1[i]) and isfinite(dap[i]), "duplicated rows give finite membership (cell " + String(i) + ")")
    for row in range(nd):  # metadata: verdict over a check output
        # float32 row sums, as the Python gate's default-dtype sum.
        require(_row_sum32(d1, row, c) <= Float32(1.0 + 1e-5) and _row_sum32(dap, row, c) <= Float32(1.0 + 1e-5),
            "duplicated rows: row " + String(row) + " sums to at most 1")
        for j in range(c):  # metadata: one row's cluster columns
            require(bitcast[DType.uint32](d1[(nd - 1 - row) * c + j]) == bitcast[DType.uint32](d2[row * c + j]),
                "reversed query row " + String(row) + " reverses the membership bytes")
    _ = dx


def fixture(fit_driver: HdbscanFitDriver, query_driver: HdbscanQueryDriver, unit: Int) raises:
    if unit == 0:
        _case_fit(fit_driver, query_driver)
    elif unit == 1:
        _case_prediction_data_fit(fit_driver, query_driver)
    elif unit == 2:
        _case_approximate(fit_driver, query_driver)
    elif unit == 3:
        _case_approximate_dupes(fit_driver, query_driver)
    elif unit == 4:
        _case_membership(fit_driver, query_driver)
    elif unit == 5:
        _case_all_points(fit_driver, query_driver)
    else:
        _case_dupes_soft(fit_driver, query_driver)


def _host_fixture(unit: Int) raises:
    fixture(HdbscanFitDriver(hdbscan_fit_host_v1), HdbscanQueryDriver(hdbscan_query_host_v1), unit)


def _device_fixture(unit: Int) raises:
    # Device column only: every module function is instantiated by TestSuite discovery, so the host build
    # must not reach the device route.
    comptime if not CPU_COLUMN_BUILD:
        fixture(HdbscanFitDriver(hdbscan_fit_device_v1), HdbscanQueryDriver(hdbscan_query_device_v1), unit)


def test_fixtures() raises:
    comptime if CPU_COLUMN_BUILD:
        check_cases("fixture", HDBSCAN_SURFACE_CASES, _host_fixture)
    else:
        for unit in range(HDBSCAN_SURFACE_CASES):  # driver-loop: ordered shared-context numerical cases, as hdbscan_native
            try:
                _device_fixture(unit)
            except problem:
                raise Error("fixture row " + String(unit) + ": " + String(problem))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

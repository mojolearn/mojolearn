# SPDX-License-Identifier: Apache-2.0
"""KMeans predict and transform through the actual KMeans provider.

Ported from `python/mojolearn/tests/test_kmeans_predict.py` (arms EQUAL,
TIES, HELDOUT) and `test_kmeans_transform.py` (arms SHAPE without the dtype
and order conversions, MIN, REFERENCE, EXACT, HELDOUT). The dtype, order,
nested-list and refusal arms stay in Python (public boundary).

Inputs keep the original shapes, cluster counts, seeds (`random_state`),
metrics, weights and tolerances. The data come from a SplitMix64 stream
regenerated here (uniform [0, 4) and an integer grid in {0, 1, 2, 3}): the
claims hold for any such data, so NumPy's PCG64 stream is not needed. The
fits keep the public defaults (k-means++ init, max_iter 300, tol 1e-4, n_init
1, oversampling 2). Every claim is a relation the original asserted between
outputs or against a float64 reference computed here, never another call of
the implementation as its own oracle.

Claims checked in IDENTICAL only in the original (rows alone and split
batches give the whole batch's bits) are unconditional here: the pixi tasks
build with `MOJOLEARN_NUMERIC_IDENTICAL=1`.
"""
from core.kmeans_native_api import (
    KMeansNativeDriver, KMeansFoldStore, KMeansFitRequest, KMeansQueryRequest,
)
from checks.numerics import identical_sqrt

comptime KMEANS_QUERY_FIXTURES = 11
# cluster/impl/kmeans_params.mojo: INIT_KMEANS_PLUS_PLUS = 0,
# METRIC_L2_EXPANDED = 0 (the public 'euclidean'), METRIC_L2_SQRT_EXPANDED = 1.
comptime _INIT_KMEANS_PLUS_PLUS = 0
comptime _L2 = 0
comptime _L2_SQRT = 1


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _next(mut state: UInt64) -> UInt64:
    """SplitMix64."""
    state += UInt64(0x9E3779B97F4A7C15)
    var z = state
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def uniform4(seed: Int, rows: Int, features: Int) -> List[Float32]:
    """Uniform [0, 4) float32 rows (24 random bits each, exact)."""
    var state = UInt64(seed) * UInt64(0x2545F4914F6CDD1D) + UInt64(1)
    var values = List[Float32](length=rows * features, fill=0)
    for i in range(rows * features):
        var bits = Int(_next(state) >> 40)
        values[i] = Float32(bits) * Float32(1.0 / 16777216.0) * Float32(4.0)
    return values^


def grid(seed: Int, rows: Int, features: Int) -> List[Float32]:
    """Integer grid rows in {0, 1, 2, 3}: many rows tie between centers."""
    var state = UInt64(seed) * UInt64(0x2545F4914F6CDD1D) + UInt64(1)
    var values = List[Float32](length=rows * features, fill=0)
    for i in range(rows * features):
        values[i] = Float32(Int(_next(state) >> 62))
    return values^


def fit(driver: KMeansNativeDriver, x: List[Float32], rows: Int, features: Int, clusters: Int,
        metric: Int, seed: Int, weights: List[Float32], mut centers: List[Float32],
        mut labels: List[UInt32], mut distances: List[Float32], want_distances: Bool) raises:
    var weight_address = Int(weights.unsafe_ptr()) if len(weights) > 0 else Int(x.unsafe_ptr())
    var result = driver.fit(KMeansFitRequest(Int(x.unsafe_ptr()), Int(centers.unsafe_ptr()),
        Int(labels.unsafe_ptr()), weight_address,
        Int(distances.unsafe_ptr()) if want_distances else 0,
        rows, features, clusters, len(weights), 300, seed, 1, _INIT_KMEANS_PLUS_PLUS, metric,
        True, 1e-4, 2.0, KMeansFoldStore()))
    require(result.finite == 0 and result.range_error == 0 and result.iterations > 0, "KMeans fit status")
    _ = x
    _ = weights


def query(driver: KMeansNativeDriver, source: Int, centers: List[Float32], dst: Int, rows: Int,
          features: Int, clusters: Int, metric: Int, transform: Bool) raises:
    var result = driver.query(KMeansQueryRequest(source, Int(centers.unsafe_ptr()), dst,
        rows, features, clusters, metric, True, transform, -1, KMeansFoldStore()))
    require(result.finite == 0 and result.range_error == 0, "KMeans query status")
    _ = centers


def _labels(rows: Int) -> List[UInt32]:
    return List[UInt32](length=rows, fill=UInt32(0xFFFFFFFF))


def _cells(count: Int) -> List[Float32]:
    return List[Float32](length=count, fill=Float32(-99.0))


def same_labels(left: List[UInt32], right: List[UInt32], what: String) raises:
    require(len(left) == len(right), what + ": extents")
    var differ = 0
    for i in range(len(left)):
        differ += Int(left[i] != right[i])
    require(differ == 0, what + ": " + String(differ) + " labels differ")


def same_bits(left: List[Float32], right: List[Float32], what: String) raises:
    require(len(left) == len(right), what + ": extents")
    var differ = 0
    for i in range(len(left)):
        differ += Int(left[i].to_bits() != right[i].to_bits())
    require(differ == 0, what + ": " + String(differ) + " cells differ bit for bit")


def fitted(driver: KMeansNativeDriver, x: List[Float32], rows: Int, features: Int, clusters: Int,
           metric: Int, seed: Int, mut centers: List[Float32], mut labels: List[UInt32]) raises:
    var none = List[Float32]()
    var unused = List[Float32]()
    fit(driver, x, rows, features, clusters, metric, seed, none, centers, labels, unused, False)


def case_equal(driver: KMeansNativeDriver, unit: Int) raises:
    """EQUAL: predict(X) is the fit's labels_ bit for bit, on uniform and
    integer-grid data under both metrics, and after a weighted fit."""
    var weighted = unit == 4
    var on_grid = unit == 2 or unit == 3
    var features = 3 if on_grid else 4
    var x = grid(2, 512, 3) if on_grid else uniform4(4 if weighted else 0, 512, 4)
    var clusters = 6 if weighted else 5
    var metric = _L2 if weighted else unit % 2
    var weights = List[Float32]()
    if weighted:
        # np.linspace(0.5, 1.5, 512, dtype=float32): float64 steps, one rounding.
        for i in range(512):
            weights.append(Float32(0.5 + Float64(i) * (1.0 / 511.0)))
    var centers = _cells(clusters * features)
    var labels = _labels(512)
    var unused = List[Float32]()
    fit(driver, x, 512, features, clusters, metric, 1 if weighted else 3, weights, centers, labels, unused, False)
    var predicted = _labels(512)
    query(driver, Int(x.unsafe_ptr()), centers, Int(predicted.unsafe_ptr()), 512, features, clusters, metric, False)
    same_labels(predicted, labels, "EQUAL fixture " + String(unit) + ": predict(X) is labels_")
    _ = x^
    _ = weights^
    _ = centers^
    _ = labels^
    _ = predicted^


def case_ties_exact(driver: KMeansNativeDriver) raises:
    """TIES and EXACT: (1, 0) is at squared distance 1 from (0, 0) and from
    (2, 0), which the expanded form reproduces exactly; the lowest center
    index wins the tie, a row on a center takes that center and clamps to 0."""
    var x: List[Float32] = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 5.0]
    for order in range(2):
        var centers: List[Float32] = [0.0, 0.0, 2.0, 0.0]
        if order == 1:
            centers = [2.0, 0.0, 0.0, 0.0]
        var p = _labels(4)
        query(driver, Int(x.unsafe_ptr()), centers, Int(p.unsafe_ptr()), 4, 2, 2, _L2, False)
        require(p[0] == 0 and p[3] == 0, "TIES: equidistant rows take center 0")
        require(p[1] == UInt32(0 if order == 0 else 1) and p[2] == UInt32(1 if order == 0 else 0),
            "TIES: rows on a center take that center")
        if order == 0:
            var t = _cells(8)
            query(driver, Int(x.unsafe_ptr()), centers, Int(t.unsafe_ptr()), 4, 2, 2, _L2, True)
            var want: List[Float32] = [1.0, 1.0, 0.0, 4.0, 4.0, 0.0, 26.0, 26.0]
            same_bits(t, want, "EXACT: small integer rows give the exact squared distances")
            _ = t^
        _ = centers^
        _ = p^
    _ = x^


def case_heldout(driver: KMeansNativeDriver) raises:
    """HELDOUT (predict and transform): every held-out label is the float64
    argmin up to float32 round-off; each of the first 16 rows alone and the
    split 1, 7, rest give the whole batch's bits."""
    var x = uniform4(6, 2048, 4)
    var xh = uniform4(7, 777, 4)
    var centers = _cells(32)
    var labels = _labels(2048)
    fitted(driver, x, 2048, 4, 8, _L2, 3, centers, labels)
    var whole = _labels(777)
    var cells = _cells(777 * 8)
    query(driver, Int(xh.unsafe_ptr()), centers, Int(whole.unsafe_ptr()), 777, 4, 8, _L2, False)
    query(driver, Int(xh.unsafe_ptr()), centers, Int(cells.unsafe_ptr()), 777, 4, 8, _L2, True)
    var bad = 0
    for row in range(777):
        var best = Float64(1.0e300)
        var chosen = Float64(0)
        for c in range(8):
            var d2 = Float64(0)
            for f in range(4):
                var delta = Float64(xh[row * 4 + f]) - Float64(centers[c * 4 + f])
                d2 += delta * delta
            if d2 < best:
                best = d2
            if c == Int(whole[row]):
                chosen = d2
        if chosen > best + 1e-4 * max(best, 1.0):
            bad += 1
    require(bad == 0, "HELDOUT: " + String(bad) + " held-out labels are not the float64 argmin")
    var base = Int(xh.unsafe_ptr())
    for row in range(16):
        var alone = _labels(1)
        var alone_cells = _cells(8)
        query(driver, base + row * 16, centers, Int(alone.unsafe_ptr()), 1, 4, 8, _L2, False)
        query(driver, base + row * 16, centers, Int(alone_cells.unsafe_ptr()), 1, 4, 8, _L2, True)
        require(alone[0] == whole[row], "HELDOUT: held-out row " + String(row) + " alone is its batch label")
        for c in range(8):
            require(alone_cells[c].to_bits() == cells[row * 8 + c].to_bits(),
                "HELDOUT: held-out row " + String(row) + " alone is its batch transform row")
        _ = alone^
        _ = alone_cells^
    var split = _labels(777)
    var split_cells = _cells(777 * 8)
    var starts: List[Int] = [0, 1, 8]
    var ends: List[Int] = [1, 8, 777]
    for part in range(3):
        var a = starts[part]
        var n = ends[part] - a
        query(driver, base + a * 16, centers, Int(split.unsafe_ptr()) + a * 4, n, 4, 8, _L2, False)
        query(driver, base + a * 16, centers, Int(split_cells.unsafe_ptr()) + a * 32, n, 4, 8, _L2, True)
    same_labels(split, whole, "HELDOUT: the split 1, 7, rest concatenates to the whole batch labels")
    same_bits(split_cells, cells, "HELDOUT: the split 1, 7, rest concatenates to the whole batch transform")
    _ = x^
    _ = xh^
    _ = centers^
    _ = labels^
    _ = whole^
    _ = cells^
    _ = split^
    _ = split_cells^


def case_shape(driver: KMeansNativeDriver) raises:
    """SHAPE: every transform distance is finite and non-negative, and
    fit_transform (the fit's own distances) is fit then transform."""
    var x = uniform4(0, 512, 4)
    var centers = _cells(20)
    var labels = _labels(512)
    var fit_cells = _cells(512 * 5)
    var none = List[Float32]()
    fit(driver, x, 512, 4, 5, _L2, 3, none, centers, labels, fit_cells, True)
    var t = _cells(512 * 5)
    query(driver, Int(x.unsafe_ptr()), centers, Int(t.unsafe_ptr()), 512, 4, 5, _L2, True)
    var bad = 0
    for i in range(512 * 5):
        var v = t[i]
        if not (v >= 0.0 and v <= Float32.MAX):
            bad += 1
    require(bad == 0, "SHAPE: " + String(bad) + " distances are negative or non-finite")
    same_bits(fit_cells, t, "SHAPE: fit_transform is fit then transform")
    _ = x^
    _ = centers^
    _ = labels^
    _ = fit_cells^
    _ = t^


def case_min(driver: KMeansNativeDriver, metric: Int) raises:
    """MIN: the transform at labels_ and at predict(X) is each row's minimum,
    bit for bit."""
    var x = uniform4(1, 512, 4)
    var centers = _cells(24)
    var labels = _labels(512)
    fitted(driver, x, 512, 4, 6, metric, 3, centers, labels)
    var t = _cells(512 * 6)
    var p = _labels(512)
    query(driver, Int(x.unsafe_ptr()), centers, Int(t.unsafe_ptr()), 512, 4, 6, metric, True)
    query(driver, Int(x.unsafe_ptr()), centers, Int(p.unsafe_ptr()), 512, 4, 6, metric, False)
    var at_label = 0
    var at_predict = 0
    for row in range(512):
        var low = t[row * 6]
        for c in range(1, 6):
            low = min(low, t[row * 6 + c])
        at_label += Int(t[row * 6 + Int(labels[row])].to_bits() != low.to_bits())
        at_predict += Int(t[row * 6 + Int(p[row])].to_bits() != low.to_bits())
    require(at_label == 0, "MIN metric " + String(metric) + ": transform at labels_ is not the row minimum on "
        + String(at_label) + " rows")
    require(at_predict == 0, "MIN metric " + String(metric) + ": transform at predict(X) is not the row minimum on "
        + String(at_predict) + " rows")
    _ = x^
    _ = centers^
    _ = labels^
    _ = t^
    _ = p^


def case_reference(driver: KMeansNativeDriver) raises:
    """REFERENCE: squared distances equal the float64 reference up to float32
    round-off; l2_sqrt_expanded on the same centers is the correctly rounded
    root of the euclidean cell, bit for bit."""
    var x = uniform4(2, 2048, 4)
    var xh = uniform4(3, 333, 4)
    var centers = _cells(32)
    var labels = _labels(2048)
    fitted(driver, x, 2048, 4, 8, _L2, 3, centers, labels)
    var t = _cells(333 * 8)
    var ts = _cells(333 * 8)
    query(driver, Int(xh.unsafe_ptr()), centers, Int(t.unsafe_ptr()), 333, 4, 8, _L2, True)
    query(driver, Int(xh.unsafe_ptr()), centers, Int(ts.unsafe_ptr()), 333, 4, 8, _L2_SQRT, True)
    var bad = 0
    var unrooted = 0
    for row in range(333):
        for c in range(8):
            var d2 = Float64(0)
            for f in range(4):
                var delta = Float64(xh[row * 4 + f]) - Float64(centers[c * 4 + f])
                d2 += delta * delta
            var cell = t[row * 8 + c]
            if abs(Float64(cell) - d2) > 1e-4 * max(d2, 1.0):
                bad += 1
            unrooted += Int(ts[row * 8 + c].to_bits() != identical_sqrt(cell).to_bits())
    require(bad == 0, "REFERENCE: " + String(bad) + " cells differ from the float64 reference")
    require(unrooted == 0, "REFERENCE: " + String(unrooted)
        + " l2_sqrt_expanded cells are not the correctly rounded root of the euclidean cell")
    _ = x^
    _ = xh^
    _ = centers^
    _ = labels^
    _ = t^
    _ = ts^


def fixture(driver: KMeansNativeDriver, unit: Int) raises:
    if unit < 5:
        case_equal(driver, unit)
    elif unit == 5:
        case_ties_exact(driver)
    elif unit == 6:
        case_heldout(driver)
    elif unit == 7:
        case_shape(driver)
    elif unit == 8:
        case_min(driver, _L2)
    elif unit == 9:
        case_min(driver, _L2_SQRT)
    else:
        case_reference(driver)

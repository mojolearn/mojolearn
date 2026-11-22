# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""fowlkes_mallows_score against scikit-learn, bit for bit (lane py2mojo-pilot, 2026-11-22; ported from
python/mojolearn/tests/test_fowlkes_mallows_score.py `test_empty_is_zero_like_sklearn`,
`test_matches_sklearn_bitwise_on_small_cases` and `test_matches_sklearn_bitwise_on_a_random_pair`).

The public call maps both label vectors onto `0..k-1` over their sorted union (bindings/mcls_cluster_door.mojo
`_cluster_codes`) and hands the codes and the range `[0, k-1]` to the metrics binding; this check does the same mapping
in its oracle and calls the binding bodies: the host column `host_fowlkes_mallows` (metrics/host/metrics_oracle.mojo,
what bindings/_mojolearn_metrics_host.mojo runs), the device column `fowlkes_mallows_score_host`
(metrics/estimator.mojo, the device contingency matrix, what bindings/_mojolearn_metrics.mojo runs).

    unit 0      no samples: 0.0 (the door answers it before the binding; the binding body gives 0.0 too)
    unit 1..7   the original's seven literal pairs, negative and sparse labels included
    unit 8      the original's random pair (`np.random.default_rng(3)`, 5000 labels in [0, 6) and [-2, 9)), and the
                score is symmetric in its arguments

References are scikit-learn's binary64 values, stored in `fixtures/metrics_fm/` by `tools/py2mojo_native_fixtures.py`
(command and versions in its docstring); the random pair's labels are stored there too. Comparison is of the binary64
words. The refusals, the sparse= keyword and the manifest stay in the Python file (public boundary, source text).
"""
from std.memory import bitcast
from metrics.host.metrics_oracle import host_fowlkes_mallows
from metrics.estimator import fowlkes_mallows_score_host
from checks.native_tests.tsa_metrics_fixture_words import fixture_i32
from checks.native_tests.py2mojo_fixture_words import METRICS_FM_DIR, fixture_f64
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases
from std.sys.compile import is_defined

comptime FM_UNITS = 9
comptime FM_RAND_N = 5000


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def small_pair(index: Int) -> Tuple[List[Int32], List[Int32]]:
    """test_matches_sklearn_bitwise_on_small_cases, the parametrized pairs in order."""
    var t: List[Int32]
    var p: List[Int32]
    if index == 0:
        t = [0, 0, 1, 1]
        p = [0, 0, 1, 1]
    elif index == 1:
        t = [0, 0, 1, 1]
        p = [1, 1, 0, 0]
    elif index == 2:
        t = [0, 0, 0, 0]
        p = [0, 1, 2, 3]
    elif index == 3:
        t = [0, 0, 0, 0]
        p = [0, 0, 0, 0]
    elif index == 4:
        t = [5]
        p = [9]
    elif index == 5:
        t = [0, 1]
        p = [0, 0]
    else:
        t = [-3, 7, 7, 100000, -3]
        p = [2, 2, 2, 1, 1]
    return (t^, p^)


def union_codes(a: List[Int32], b: List[Int32]) -> Tuple[List[Int32], List[Int32], Int]:
    """The oracle's label mapping: both vectors onto 0..k-1 over their sorted union."""
    var uniq = List[Int32]()
    for i in range(len(a) + len(b)):  # small-loop(oracle: at most 10000 labels, a sorted insert of the new ones)
        var v = a[i] if i < len(a) else b[i - len(a)]
        var at = _find(uniq, v)
        if at < len(uniq) and uniq[at] == v:
            continue
        uniq.insert(at, v)
    var ca = List[Int32](length=len(a), fill=0)
    var cb = List[Int32](length=len(b), fill=0)
    for i in range(len(a)):  # small-loop(oracle: binary search per label)
        ca[i] = Int32(_find(uniq, a[i]))
    for i in range(len(b)):  # small-loop(oracle: binary search per label)
        cb[i] = Int32(_find(uniq, b[i]))
    return (ca^, cb^, len(uniq))


def _find(uniq: List[Int32], v: Int32) -> Int:
    """The first index whose label is not below v (len(uniq) when none is)."""
    var lo = 0
    var hi = len(uniq)
    while lo < hi:
        var mid = (lo + hi) // 2
        if uniq[mid] < v:
            lo = mid + 1
        else:
            hi = mid
    return lo


trait FowlkesCall:
    @staticmethod
    def score(y_true: List[Int32], y_pred: List[Int32], n: Int, lower: Int32, upper: Int32) raises -> Float64:
        ...


struct HostFowlkes(FowlkesCall):
    @staticmethod
    def score(y_true: List[Int32], y_pred: List[Int32], n: Int, lower: Int32, upper: Int32) raises -> Float64:
        return host_fowlkes_mallows(y_true, y_pred, n, lower, upper)


struct DeviceFowlkes(FowlkesCall):
    @staticmethod
    def score(y_true: List[Int32], y_pred: List[Int32], n: Int, lower: Int32, upper: Int32) raises -> Float64:
        return fowlkes_mallows_score_host(y_true, y_pred, n, lower, upper)


def scored[K: FowlkesCall](a: List[Int32], b: List[Int32]) raises -> Float64:
    var mapped = union_codes(a, b)
    var k = mapped[2]
    return K.score(mapped[0], mapped[1], len(a), 0, Int32(max(k - 1, 0)))


def same_words(got: Float64, want: Float64, label: String) raises:
    require(bitcast[DType.uint64](got) == bitcast[DType.uint64](want),
        label + ": " + String(got) + " scikit-learn " + String(want))


def fowlkes_unit[K: FowlkesCall](unit: Int) raises:
    if unit == 0:
        var empty = List[Int32]()
        same_words(K.score(empty, empty, 0, 0, 0), 0.0, "no samples")
    elif unit <= 7:
        var pair = small_pair(unit - 1)
        var want = fixture_f64(METRICS_FM_DIR + "fm-small-want.f64", 7)
        same_words(scored[K](pair[0], pair[1]), want[unit - 1], "small pair " + String(unit - 1))
    else:
        var yt = fixture_i32(METRICS_FM_DIR + "fm-rand-yt.i32", FM_RAND_N)
        var yp = fixture_i32(METRICS_FM_DIR + "fm-rand-yp.i32", FM_RAND_N)
        var want = fixture_f64(METRICS_FM_DIR + "fm-rand-want.f64", 1)
        var got = scored[K](yt, yp)
        same_words(got, want[0], "random pair")
        same_words(scored[K](yp, yt), got, "random pair, arguments swapped")


def _host_unit(unit: Int) raises:
    fowlkes_unit[HostFowlkes](unit)


def test_fowlkes_mallows() raises:
    comptime if is_defined["MOJOLEARN_COLUMN_CPU"]():
        check_cases("fowlkes_mallows", FM_UNITS, _host_unit)
    else:
        # the units share the process's one metrics device context: in order
        var failed = String("")
        var count = 0
        for unit in range(FM_UNITS):  # driver-loop: over the units on the one device context
            try:
                fowlkes_unit[DeviceFowlkes](unit)
            except error:
                count += 1
                failed += "\n      fowlkes_mallows row " + String(unit) + ": " + String(error)
        if count > 0:
            raise Error("fowlkes_mallows: " + String(count) + " of " + String(FM_UNITS) + " cases failed" + failed)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

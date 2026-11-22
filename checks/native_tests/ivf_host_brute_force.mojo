# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The host IVF-Flat index probing every list is brute force, and its euclidean metric is the root of the squared one
(lane py2mojo-D, 2026-11-22; ported from python/mojolearn/tests/test_cpu_training_embedding_ivf.py
`test_ivf_runs_on_the_host_when_built`, its numeric half).

The build and search are the host column's (ivf/host/ivf_host.mojo `host_ivf_build`, `host_ivf_search`; the build
words the `_mojolearn_ivf_host` build entry hands it, ivf/ops_entry_host_build.mojo): 300 x 6 rows and 9 queries
(the original's NumPy draws, seed 5, stored in `fixtures/ivf_host/` by `tools/py2mojo_native_fixtures_b7.py`),
n_lists 4, n_probes 4, n_neighbors 5, kmeans_n_iters 20 (the public default), random_state 1.

    unit 0  sqeuclidean (metric 0): every query counts all 300 candidates; the ids are the stable argsort of the
            float64 squared distances (the stored NumPy reference)
    unit 1  euclidean (metric 1): the same ids as sqeuclidean, and every distance is the root of the squared one
            within numpy.allclose(rtol=1e-6) (atol 1e-8)

The n_probes > n_lists refusal and the sabotage word stay in the Python test. The two units are independent and run
across host cores; the build and search keep their own parallelism.
"""
from std.math import sqrt
from ivf.host.ivf_host import IvfHostResult, host_ivf_build, host_ivf_search
from checks.native_tests.tsa_metrics_fixture_words import fixture_f32, fixture_i32
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases

comptime IVF_DIR = "checks/native_tests/fixtures/ivf_host/"
comptime N = 300
comptime DIM = 6
comptime Q = 9
comptime K = 5
comptime LISTS = 4
comptime KMEANS_ITERS = 20
comptime METRIC_SQ = 0  # python/mojolearn/_ivf_impl.py _METRICS: sqeuclidean is L2Expanded
comptime METRIC_SQRT = 1  # euclidean is L2SqrtExpanded
comptime IVF_HOST_UNITS = 2


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def search(metric: Int) raises -> IvfHostResult:
    var x = fixture_f32(IVF_DIR + "x.f32", N * DIM)
    var q = fixture_f32(IVF_DIR + "q.f32", Q * DIM)
    var index = host_ivf_build(x, N, DIM, LISTS, KMEANS_ITERS, metric, UInt64(1))
    return host_ivf_search(index, q, Q, K, LISTS)


def run_unit(unit: Int) raises:
    var sq = search(METRIC_SQ)
    if unit == 0:
        var want = fixture_i32(IVF_DIR + "want-ids.i32", Q * K)
        for qi in range(Q):  # driver-loop: over the fixture's queries
            require(Int(sq.n_candidates[qi]) == N, "query " + String(qi) + " counts " + String(sq.n_candidates[qi])
                    + " candidates, expected " + String(N))
            for s in range(K):
                require(Int(sq.indices[qi * K + s]) == Int(want[qi * K + s]), "query " + String(qi) + " slot "
                        + String(s) + ": id " + String(sq.indices[qi * K + s]) + ", brute force "
                        + String(want[qi * K + s]))
    elif unit == 1:
        var root = search(METRIC_SQRT)
        for c in range(Q * K):  # driver-loop: over the fixture's result slots
            require(root.indices[c] == sq.indices[c], "slot " + String(c) + ": euclidean id "
                    + String(root.indices[c]) + ", sqeuclidean id " + String(sq.indices[c]))
            var a = Float64(root.distances[c])
            var e = sqrt(Float64(sq.distances[c]))
            require(abs(a - e) <= 1e-8 + 1e-6 * abs(e), "slot " + String(c) + ": euclidean distance " + String(a)
                    + ", root of the squared " + String(e))
    else:
        raise Error("unknown ivf host unit")


def test_ivf_host_brute_force() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    check_cases("ivf_host_brute_force", IVF_HOST_UNITS, run_unit)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Louvain's reported modularity against the modularity of the labels it returns, and its quality against networkx
(lane py2mojo-D6, 2026-11-22; ported from python/mojolearn/tests/test_x_neighbors_sanity.py `test_louvain`).

The fit is the one Louvain's door runs on a dense adjacency (neighbors/graph_fit_run.mojo `louvain_fit_run`: the
native symmetry check, the aggregation levels, the renumbering) on the host x_neighbors image, with the estimator's
defaults (resolution 1.0, threshold 1e-7, max_level None: 0, no limit). Input: the original `_graph(80)` (seed 0,
weights 1..3, symmetric, zero diagonal), stored in fixtures/xn_louvain/ by tools/py2mojo_native_fixtures_b6.py with
the reference: networkx 3.7's best modularity of `louvain_communities` over seeds 0..4.

    the modularity of the returned labels, (1 / 2m) sum_ij [A_ij - k_i k_j / 2m] [c_i == c_j] (networkx's
    `community.modularity` with resolution 1), equals `modularity_` to 1e-4, and exceeds networkx's best less 0.03

The networkx call itself (the reference) and the public attributes stay in Python. The modularity oracle's rows run
across the host cores.
"""
from core.host_parallel import host_parallelize
from core.neighbors_native_driver import XnDriver
from neighbors.graph_fit_run import LouvainRequest, louvain_fit_run
from x_neighbors.native_descriptor_host import xn_native_v1
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import require, fixture_f32

comptime XN_LOUVAIN_DIR = "checks/native_tests/fixtures/xn_louvain/"


def modularity(a: List[Float32], labels: List[Int32], n: Int) -> Float64:
    """networkx `community.modularity` (resolution 1) of the partition `labels` of the weighted graph `a`."""
    var degree = List[Float64](length=n, fill=0)
    var ap = a.unsafe_ptr()
    var dp = degree.unsafe_ptr()
    def row_degree(i: Int) {imm ap, imm dp, imm n}:
        var total = Float64(0)
        for j in range(n):  # the one row's weighted degree
            total += Float64(ap[i * n + j])
        dp[i] = total
    host_parallelize(row_degree, n)
    var two_m = Float64(0)
    for i in range(n):  # small-loop: the total weight 2m over 80 degrees
        two_m += degree[i]
    var terms = List[Float64](length=n, fill=0)
    var lp = labels.unsafe_ptr()
    var tp = terms.unsafe_ptr()
    def row_term(i: Int) {imm ap, imm dp, imm lp, imm tp, imm n, imm two_m}:
        var total = Float64(0)
        for j in range(n):  # the one row's same-community terms
            if lp[i] == lp[j]:
                total += Float64(ap[i * n + j]) - dp[i] * dp[j] / two_m
        tp[i] = total
    host_parallelize(row_term, n)
    var q = Float64(0)
    for i in range(n):  # small-loop: the sum of 80 row terms
        q += terms[i]
    _ = degree^
    _ = terms^
    return q / two_m


def test_louvain() raises:
    var n = 80
    var a = fixture_f32(XN_LOUVAIN_DIR + "adjacency-80.f32", n * n)
    var q_ref = fixture_f64(XN_LOUVAIN_DIR + "q-ref.f64", 1)[0]
    var labels = List[Int32](length=n, fill=-1)
    var outcome = louvain_fit_run(XnDriver(xn_native_v1), LouvainRequest(False, 0, 0, 0, 0, Int(a.unsafe_ptr()),
        Int(labels.unsafe_ptr()), n, 0, 1.0, 1.0e-7))
    require(outcome.communities > 0, "Louvain found no community")
    for i in range(n):  # small-loop: every node labelled
        require(labels[i] >= 0 and Int(labels[i]) < outcome.communities,
                "Louvain label " + String(i) + ": " + String(labels[i]) + " of " + String(outcome.communities))
    var q_ours = modularity(a, labels, n)
    require(abs(q_ours - outcome.modularity) < 1e-4, "modularity_ " + String(outcome.modularity)
            + " differs from the modularity of the returned labels " + String(q_ours))
    require(q_ours > q_ref - 0.03, "Louvain modularity " + String(q_ours) + " not within 0.03 of networkx's best "
            + String(q_ref))
    _ = a^
    _ = labels^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

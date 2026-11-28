# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Leiden (lane leiden-umap-1, 2026-11-28) through the door's RUN (neighbors/graph_fit_run.mojo `leiden_fit_run`:
the native symmetry check, x_neighbors/graph_par.mojo `ld_drive` / `ld_drive_csr`, the renumbering) on the host
x_neighbors image.

    a hand-checkable graph: two triangles joined by one edge are two communities, modularity 2 (3/7 - (7/14)^2)
    the partition is valid: every label in [0, n_communities_), numbered by first occurrence
    every community is connected (the refinement's guarantee), on the modularity and the CPM fits
    quality_ is the quality of the returned labels (modularity: networkx's formula; CPM: e_c / m - res N_c^2 / 2m)
    the modularity is not below Louvain's on the same graph (random_state 0: Louvain's colouring)
    the CSR entry returns the dense entry's labels and quality bit for bit; until-stable is not below one iteration

Input: the Louvain check's `_graph(80)` fixture (checks/native_tests/fixtures/xn_louvain/adjacency-80.f32). The
oracles' rows and communities run across the host cores.
"""
from core.host_parallel import host_parallelize
from core.neighbors_native_driver import XnDriver
from neighbors.graph_fit_run import LouvainRequest, louvain_fit_run, LeidenRequest, LeidenOutcome, leiden_fit_run
from neighbors.neighbors_graph_run import XN_LEIDEN_MODULARITY, XN_LEIDEN_CPM
from x_neighbors.native_descriptor_host import xn_native_v1
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import require, fixture_f32

comptime XN_LOUVAIN_DIR = "checks/native_tests/fixtures/xn_louvain/"


def quality(a: List[Float32], labels: List[Int32], n: Int, res: Float64, cpm: Bool) -> Float64:
    """The quality of the partition `labels` of the weighted graph `a`: networkx's modularity with resolution res,
    or igraph's CPM (1 / 2m) sum_ij [A_ij - res] [c_i == c_j] over i != j plus the self pairs' -res."""
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
    for i in range(n):  # small-loop: the total weight 2m over the degrees
        two_m += degree[i]
    var terms = List[Float64](length=n, fill=0)
    var lp = labels.unsafe_ptr()
    var tp = terms.unsafe_ptr()
    def row_term(i: Int) {imm ap, imm dp, imm lp, imm tp, imm n, imm two_m, imm res, imm cpm}:
        var total = Float64(0)
        for j in range(n):  # the one row's same-community terms
            if lp[i] == lp[j]:
                var expected_null = res if cpm else res * dp[i] * dp[j] / two_m
                total += Float64(ap[i * n + j]) - expected_null
        tp[i] = total
    host_parallelize(row_term, n)
    var q = Float64(0)
    for i in range(n):  # small-loop: the sum of the row terms
        q += terms[i]
    _ = degree^
    _ = terms^
    return q / two_m


def check_partition(a: List[Float32], labels: List[Int32], n: Int, communities: Int, what: String) raises:
    """Labels in range and numbered by first occurrence; every community connected (a breadth-first search from
    its first node inside the community reaches all its members, one community per host task)."""
    require(communities > 0, what + ": no community")
    var next_new = 0
    for i in range(n):  # small-loop: the first-occurrence numbering over the nodes
        var c = Int(labels[i])
        require(c >= 0 and c < communities, what + ": label " + String(i) + " = " + String(c))
        require(c <= next_new, what + ": labels not numbered by first occurrence at node " + String(i))
        if c == next_new:
            next_new += 1
    require(next_new == communities, what + ": " + String(next_new) + " labels used of " + String(communities))
    var reached = List[Int32](length=communities, fill=0)
    var sizes = List[Int32](length=communities, fill=0)
    for i in range(n):  # small-loop: the community sizes
        sizes[Int(labels[i])] += 1
    var ap = a.unsafe_ptr()
    var lp = labels.unsafe_ptr()
    var rp = reached.unsafe_ptr()
    def walk(c: Int) {imm ap, imm lp, imm rp, imm n}:
        var seen = List[Bool](length=n, fill=False)
        var stack = List[Int]()
        for i in range(n):  # the community's first node
            if Int(lp[i]) == c:
                stack.append(i)
                seen[i] = True
                break
        var count = 0
        while len(stack) > 0:
            var u = stack.pop()
            count += 1
            for v in range(n):  # u's neighbours inside the community
                if not seen[v] and Int(lp[v]) == c and ap[u * n + v] != Float32(0):
                    seen[v] = True
                    stack.append(v)
        rp[c] = Int32(count)
    host_parallelize(walk, communities)
    for c in range(communities):  # small-loop: one verdict per community
        require(reached[c] == sizes[c], what + ": community " + String(c) + " is not connected ("
                + String(reached[c]) + " of " + String(sizes[c]) + " nodes reached)")
    _ = reached^
    _ = sizes^


def leiden_dense(a: List[Float32], mut labels: List[Int32], n: Int, n_iterations: Int, which: Int, seed: Int,
                 res: Float64) raises -> LeidenOutcome:
    return leiden_fit_run(XnDriver(xn_native_v1), LeidenRequest(False, 0, 0, 0, 0, Int(a.unsafe_ptr()),
        Int(labels.unsafe_ptr()), n, n_iterations, which, seed, res))


def test_leiden_two_triangles() raises:
    var n = 6
    var a = List[Float32](length=n * n, fill=0)
    var edges: List[Int] = [0, 1, 0, 2, 1, 2, 3, 4, 3, 5, 4, 5, 2, 3]
    for e in range(7):  # small-loop: the seven edges, both directions
        a[edges[2 * e] * n + edges[2 * e + 1]] = 1.0
        a[edges[2 * e + 1] * n + edges[2 * e]] = 1.0
    var labels = List[Int32](length=n, fill=-1)
    var got = leiden_dense(a, labels, n, -1, XN_LEIDEN_MODULARITY, 0, 1.0)
    require(got.communities == 2, "two triangles: " + String(got.communities) + " communities")
    for i in range(n):  # small-loop: the expected labels
        require(Int(labels[i]) == (0 if i < 3 else 1), "two triangles: node " + String(i) + " in " + String(labels[i]))
    var expected = 2.0 * (3.0 / 7.0 - 0.25)
    require(abs(got.quality - expected) < 1e-5, "two triangles: quality_ " + String(got.quality) + ", expected "
            + String(expected))
    _ = a^
    _ = labels^


def test_leiden_modularity_against_louvain() raises:
    var n = 80
    var a = fixture_f32(XN_LOUVAIN_DIR + "adjacency-80.f32", n * n)
    var labels = List[Int32](length=n, fill=-1)
    var got = leiden_dense(a, labels, n, -1, XN_LEIDEN_MODULARITY, 0, 1.0)
    check_partition(a, labels, n, got.communities, "Leiden modularity")
    var q = quality(a, labels, n, 1.0, False)
    require(abs(q - got.quality) < 1e-4, "quality_ " + String(got.quality) + " differs from the labels' modularity "
            + String(q))
    require(got.iterations >= 1.0, "Leiden ran no iteration")
    var lv = List[Int32](length=n, fill=-1)
    _ = louvain_fit_run(XnDriver(xn_native_v1), LouvainRequest(False, 0, 0, 0, 0, Int(a.unsafe_ptr()),
        Int(lv.unsafe_ptr()), n, 0, 1.0, 1.0e-7))
    var q_lv = quality(a, lv, n, 1.0, False)
    require(q >= q_lv - 1e-6, "Leiden modularity " + String(q) + " below Louvain's " + String(q_lv))
    var one = List[Int32](length=n, fill=-1)
    var got_one = leiden_dense(a, one, n, 1, XN_LEIDEN_MODULARITY, 0, 1.0)
    check_partition(a, one, n, got_one.communities, "Leiden one iteration")
    require(got.quality >= got_one.quality, "until stable " + String(got.quality) + " below one iteration "
            + String(got_one.quality))
    require(got_one.iterations == 1.0, "n_iterations 1 ran " + String(got_one.iterations))
    var seeded = List[Int32](length=n, fill=-1)
    var got_seeded = leiden_dense(a, seeded, n, -1, XN_LEIDEN_MODULARITY, 7, 0.5)
    check_partition(a, seeded, n, got_seeded.communities, "Leiden random_state 7, resolution 0.5")
    var q_seeded = quality(a, seeded, n, 0.5, False)
    require(abs(q_seeded - got_seeded.quality) < 1e-4, "resolution 0.5: quality_ " + String(got_seeded.quality)
            + " differs from " + String(q_seeded))
    _ = a^
    _ = labels^
    _ = lv^
    _ = one^
    _ = seeded^


def test_leiden_cpm_and_csr() raises:
    var n = 80
    var a = fixture_f32(XN_LOUVAIN_DIR + "adjacency-80.f32", n * n)
    var labels = List[Int32](length=n, fill=-1)
    var res = 0.5
    var got = leiden_dense(a, labels, n, -1, XN_LEIDEN_CPM, 0, res)
    check_partition(a, labels, n, got.communities, "Leiden CPM")
    var q = quality(a, labels, n, res, True)
    require(abs(q - got.quality) < 1e-4, "CPM quality_ " + String(got.quality) + " differs from the labels' CPM "
            + String(q))
    # the same graph as CSR (rows ascending, columns ascending, its weights)
    var indptr = List[Int32](length=n + 1, fill=0)
    var indices = List[Int32]()
    var vals = List[Float32]()
    for i in range(n):  # small-loop: the CSR rows of the 80-node fixture
        for j in range(n):
            if a[i * n + j] != Float32(0):
                indices.append(Int32(j))
                vals.append(a[i * n + j])
        indptr[i + 1] = Int32(len(indices))
    var csr_labels = List[Int32](length=n, fill=-1)
    var got_csr = leiden_fit_run(XnDriver(xn_native_v1), LeidenRequest(True, Int(indptr.unsafe_ptr()),
        Int(indices.unsafe_ptr()), Int(vals.unsafe_ptr()), len(indices), 0, Int(csr_labels.unsafe_ptr()), n, -1,
        XN_LEIDEN_CPM, 0, res))
    require(got_csr.quality == got.quality, "CSR quality_ " + String(got_csr.quality) + " is not the dense "
            + String(got.quality))
    for i in range(n):  # small-loop: the same labels
        require(csr_labels[i] == labels[i], "CSR label " + String(i) + " differs from the dense entry's")
    _ = a^
    _ = labels^
    _ = indptr^
    _ = indices^
    _ = vals^
    _ = csr_labels^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

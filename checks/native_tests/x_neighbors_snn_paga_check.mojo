# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The shared-nearest-neighbor graph and PAGA (lane sc-graph-1, 2026-11-28) through the door's RUN
(neighbors/graph_fit_run.mojo `snn_run`, `paga_run`; x_neighbors/graph_par.mojo `sn_drive`, `pg_drive`) on the host
x_neighbors image.

    shared_neighbors_graph, hand-checked: four nodes in two overlapping triples give the weights 1 and 1/2 of the
    Jaccard index; prune 0.6 keeps the 1s only; a CSR with rows of unequal length gives 2/3 and 1/3; a node listed
    twice and an index out of range are refused by name
    shared_neighbors_graph against a float64 brute force on 200 nodes x 10 neighbors: the same edge set, every weight
    within one float32 rounding, the result symmetric bit for bit, columns ascending; Louvain takes it as CSR
    PAGA, hand-checked: six nodes, three groups, the counts, the v1.2 connectivities 1/4.2, 1/1.8, min(1, 2/1) and
    the spanning tree with its tie between two pairs of one edge each (the pair of lower indices kept); the dense
    entry returns the CSR entry's words bit for bit
    labels by community size (Louvain / Leiden label_order="size"): the largest community 0, ties to the community
    whose first node comes first

Nothing is read from a fixture pack: every graph is written here. The brute force's rows run across the host cores.
"""
from core.host_parallel import host_parallelize
from core.neighbors_native_driver import XnDriver
from neighbors.graph_fit_run import (
    SnnRequest, SnnOutcome, snn_run, PagaRequest, PagaOutcome, paga_run, LouvainRequest, louvain_fit_run,
    size_relabel_run,
)
from x_neighbors.native_descriptor_host import xn_native_v1
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import require


def snn_index(idx: List[Int32], n: Int, k: Int, prune: Float64, mut indptr: List[Int32]) raises -> SnnOutcome:
    """shared_neighbors_graph of an (n, k) index array on the host image."""
    indptr = List[Int32](length=n + 1, fill=-1)
    return snn_run(XnDriver(xn_native_v1), SnnRequest(0, Int(idx.unsafe_ptr()), n * k, k, Int(indptr.unsafe_ptr()),
                                                      n, prune))


def snn_csr(ip: List[Int32], ix: List[Int32], n: Int, prune: Float64, mut indptr: List[Int32]) raises -> SnnOutcome:
    indptr = List[Int32](length=n + 1, fill=-1)
    return snn_run(XnDriver(xn_native_v1), SnnRequest(Int(ip.unsafe_ptr()), Int(ix.unsafe_ptr()), len(ix), 0,
                                                      Int(indptr.unsafe_ptr()), n, prune))


def weight_at(indptr: List[Int32], got: SnnOutcome, i: Int, j: Int) -> Float32:
    """Entry (i, j) of the result, -1 when absent."""
    for e in range(Int(indptr[i]), Int(indptr[i + 1])):  # small-loop: one row of the result
        if Int(got.indices[e]) == j:
            return got.data[e]
    return Float32(-1)


def test_snn_hand_checked() raises:
    # N0 = N1 = {0, 1, 2}, N2 = N3 = {1, 2, 3}: J = 1 inside a pair, 2 / (3 + 3 - 2) = 1/2 across
    var idx: List[Int32] = [0, 1, 2, 1, 0, 2, 2, 1, 3, 3, 2, 1]
    var indptr = List[Int32]()
    var got = snn_index(idx, 4, 3, 1.0 / 15.0, indptr)
    require(got.nnz == 16, "prune 1/15: " + String(got.nnz) + " entries, expected 16")
    for i in range(4):  # small-loop: the 4 x 4 weights
        for j in range(4):
            var want = Float32(1) if (i < 2) == (j < 2) else Float32(0.5)
            var w = weight_at(indptr, got, i, j)
            require(w == want, "prune 1/15: weight (" + String(i) + ", " + String(j) + ") = " + String(w)
                    + ", expected " + String(want))
    var pruned_ip = List[Int32]()
    var pruned = snn_index(idx, 4, 3, 0.6, pruned_ip)
    require(pruned.nnz == 8, "prune 0.6: " + String(pruned.nnz) + " entries, expected the 8 weights of 1")
    for i in range(4):  # small-loop: the kept columns of each row
        require(Int(pruned_ip[i + 1]) - Int(pruned_ip[i]) == 2, "prune 0.6: row " + String(i) + " length")
        var lo = Int(pruned_ip[i])
        var first = 0 if i < 2 else 2
        require(Int(pruned.indices[lo]) == first and Int(pruned.indices[lo + 1]) == first + 1,
                "prune 0.6: row " + String(i) + " columns")
    # rows of unequal length: N0 = {0, 1}, N1 = {1, 0, 2}, N2 = {2}
    var ip: List[Int32] = [0, 2, 5, 6]
    var ix: List[Int32] = [0, 1, 1, 0, 2, 2]
    var uneven_ip = List[Int32]()
    var uneven = snn_csr(ip, ix, 3, 0.0, uneven_ip)
    require(uneven.nnz == 7, "unequal rows: " + String(uneven.nnz) + " entries, expected 7 (no 0 - 2 pair)")
    require(weight_at(uneven_ip, uneven, 0, 1) == Float32(2.0) / Float32(3.0), "unequal rows: J(0, 1) is not 2/3")
    require(weight_at(uneven_ip, uneven, 1, 2) == Float32(1.0) / Float32(3.0), "unequal rows: J(1, 2) is not 1/3")
    require(weight_at(uneven_ip, uneven, 0, 2) == Float32(-1), "unequal rows: 0 and 2 share no neighbor")
    for i in range(3):  # small-loop: the self weights
        require(weight_at(uneven_ip, uneven, i, i) == Float32(1), "unequal rows: self weight " + String(i))
    # refusals
    var dup: List[Int32] = [0, 1, 1, 1, 0, 2, 2, 1, 0]
    var dup_ip = List[Int32]()
    var refused = False
    try:
        _ = snn_index(dup, 3, 3, 0.0, dup_ip)
    except error:
        refused = String(error).find("appears twice") >= 0
    require(refused, "a node listed twice in one row was not refused by name")
    var out_of_range: List[Int32] = [0, 3, 1, 0, 2, 1]
    var oor_ip = List[Int32]()
    refused = False
    try:
        _ = snn_index(out_of_range, 3, 2, 0.0, oor_ip)
    except error:
        refused = String(error).find("outside [0, n)") >= 0
    require(refused, "a neighbor index outside [0, n) was not refused by name")


def test_snn_against_brute_force() raises:
    var n = 200
    var k = 10
    # neighbor j of node i: i itself, then (i + h(i, m)) mod n for a fixed integer hash, distinct within the row
    var idx = List[Int32](length=n * k, fill=0)
    for i in range(n):  # small-loop: the fixture rows
        idx[i * k] = Int32(i)
        var m = 1
        var step = 1
        while m < k:
            var j = (i + step * step + (i * 7 + step * 13) % 5) % n
            var seen = False
            for c in range(m):  # small-loop: the row's earlier entries
                if Int(idx[i * k + c]) == j:
                    seen = True
            if not seen:
                idx[i * k + m] = Int32(j)
                m += 1
            step += 1
    var indptr = List[Int32]()
    var got = snn_index(idx, n, k, 1.0 / 15.0, indptr)
    # the float64 Jaccard of every pair, one row per host task
    var member = List[Int32](length=n * n, fill=0)
    for i in range(n):  # small-loop: the membership matrix
        for c in range(k):
            member[i * n + Int(idx[i * k + c])] = 1
    var want = List[Float64](length=n * n, fill=0)
    var mp = member.unsafe_ptr()
    var wp = want.unsafe_ptr()
    def row(i: Int) {imm mp, imm wp, imm n, imm k}:
        for j in range(n):  # the one row's pairs
            var s = 0
            for v in range(n):
                if mp[i * n + v] != 0 and mp[j * n + v] != 0:
                    s += 1
            wp[i * n + j] = Float64(s) / Float64(2 * k - s) if s > 0 else -1.0
    host_parallelize(row, n)
    var count = 0
    for i in range(n):  # small-loop: the expected kept entries
        for j in range(n):
            if want[i * n + j] >= 1.0 / 15.0:
                count += 1
    require(got.nnz == count, "brute force: " + String(got.nnz) + " entries, expected " + String(count))
    for i in range(n):  # small-loop: every entry of the result against the brute force
        var prev = -1
        for e in range(Int(indptr[i]), Int(indptr[i + 1])):
            var j = Int(got.indices[e])
            require(j > prev, "row " + String(i) + ": columns not strictly ascending")
            prev = j
            var w = Float64(got.data[e])
            require(abs(w - want[i * n + j]) <= 6e-8 * want[i * n + j], "weight (" + String(i) + ", " + String(j)
                    + ") = " + String(w) + ", expected " + String(want[i * n + j]))
            require(weight_at(indptr, got, j, i) == got.data[e], "the mirror of (" + String(i) + ", " + String(j)
                    + ") differs")
    # the form Louvain's CSR loader takes: sorted, mirrored, weighted
    var labels = List[Int32](length=n, fill=-1)
    _ = louvain_fit_run(XnDriver(xn_native_v1), LouvainRequest(True, Int(indptr.unsafe_ptr()),
        Int(got.indices.unsafe_ptr()), Int(got.data.unsafe_ptr()), got.nnz, 0, Int(labels.unsafe_ptr()), n, 0, 1.0,
        1.0e-7))
    for i in range(n):  # small-loop: every node labelled
        require(Int(labels[i]) >= 0 and Int(labels[i]) < n, "Louvain on the SNN graph: label " + String(i))
    _ = got^  # read by the Louvain call above through its pointers
    _ = indptr^
    _ = labels^
    _ = idx^
    _ = member^
    _ = want^


def test_paga_hand_checked() raises:
    # six nodes, groups [0 0 0 1 1 2]; stored edges 0->1 0->5 1->0 1->2 2->1 2->3 3->4 4->3 4->5 5->4
    # counts: inside 0: 4, 0->1: 1, 0->2: 1, inside 1: 2, 1->2: 1, 2->1: 1; es = [6, 3, 1], sizes [3, 2, 1]
    # E01 = (6*2 + 3*3)/5 = 4.2, E02 = (6*1 + 1*3)/5 = 1.8, E12 = (3*1 + 1*2)/5 = 1
    # v01 = 1, v02 = 1, v12 = 2: connectivities 1/4.2, 1/1.8, min(1, 2); the tree keeps 1-2 and, of the tie
    # between 0-1 and 0-2 (one edge each), the pair of lower indices 0-1
    var n = 6
    var ip: List[Int32] = [0, 2, 4, 6, 7, 9, 10]
    var ix: List[Int32] = [1, 5, 0, 2, 1, 3, 4, 3, 5, 4]
    var groups: List[Int32] = [0, 0, 0, 1, 1, 2]
    var got = paga_run(XnDriver(xn_native_v1), PagaRequest(True, Int(ip.unsafe_ptr()), Int(ix.unsafe_ptr()), len(ix),
                                                           0, Int(groups.unsafe_ptr()), n))
    require(got.groups == 3, "PAGA: " + String(got.groups) + " groups, expected 3")
    require(Int(got.sizes[0]) == 3 and Int(got.sizes[1]) == 2 and Int(got.sizes[2]) == 1, "PAGA: group sizes")
    var want = List[Float32](length=9, fill=0)
    want[1] = Float32(1) / Float32(4.2)
    want[3] = want[1]
    want[2] = Float32(1) / Float32(1.8)
    want[6] = want[2]
    want[5] = Float32(1)
    want[7] = Float32(1)
    for t in range(9):  # small-loop: the 3 x 3 connectivities
        require(abs(Float64(got.connectivities[t]) - Float64(want[t])) <= 1e-6, "PAGA: connectivity "
                + String(t) + " = " + String(got.connectivities[t]) + ", expected " + String(want[t]))
    for t in range(9):  # small-loop: the tree
        var on_tree = t == 1 or t == 3 or t == 5 or t == 7
        require((got.tree[t] == got.connectivities[t]) if on_tree else (got.tree[t] == Float32(0)),
                "PAGA: tree cell " + String(t) + " = " + String(got.tree[t]))
    # the dense entry: the same words
    var a = List[Float32](length=n * n, fill=0)
    for r in range(n):  # small-loop: the dense adjacency of the same edges
        for e in range(Int(ip[r]), Int(ip[r + 1])):
            a[r * n + Int(ix[e])] = Float32(1)
    var dense = paga_run(XnDriver(xn_native_v1), PagaRequest(False, 0, 0, 0, Int(a.unsafe_ptr()),
                                                             Int(groups.unsafe_ptr()), n))
    require(dense.groups == 3, "PAGA dense: groups")
    for t in range(9):  # small-loop: bit for bit
        require(dense.connectivities[t] == got.connectivities[t] and dense.tree[t] == got.tree[t],
                "PAGA dense: cell " + String(t) + " differs from the CSR entry's")
    # refusals: a negative label, a label above n - 1
    var negative: List[Int32] = [0, 0, -1, 1, 1, 2]
    var refused = False
    try:
        _ = paga_run(XnDriver(xn_native_v1), PagaRequest(True, Int(ip.unsafe_ptr()), Int(ix.unsafe_ptr()), len(ix),
                                                         0, Int(negative.unsafe_ptr()), n))
    except error:
        refused = String(error).find("negative") >= 0
    require(refused, "PAGA: a negative label was not refused by name")
    var large: List[Int32] = [0, 0, 0, 1, 1, 9]
    refused = False
    try:
        _ = paga_run(XnDriver(xn_native_v1), PagaRequest(True, Int(ip.unsafe_ptr()), Int(ix.unsafe_ptr()), len(ix),
                                                         0, Int(large.unsafe_ptr()), n))
    except error:
        refused = String(error).find("numbered from 0") >= 0
    require(refused, "PAGA: a label above n - 1 was not refused by name")
    _ = a^  # every list below was read through its pointer by a call above
    _ = ip^
    _ = ix^
    _ = groups^
    _ = negative^
    _ = large^


def test_labels_by_size() raises:
    # sizes [2, 3, 1]: community 1 becomes 0, 0 becomes 1, 2 stays 2
    var labels: List[Int32] = [0, 0, 1, 1, 1, 2]
    size_relabel_run(XnDriver(xn_native_v1), Int(labels.unsafe_ptr()), 6, 3)
    var want: List[Int32] = [1, 1, 0, 0, 0, 2]
    for i in range(6):  # small-loop: the expected labels
        require(labels[i] == want[i], "by size: node " + String(i) + " has " + String(labels[i]))
    # a tie (sizes [2, 2, 1]): the community whose first node comes first keeps the lower label
    var tie: List[Int32] = [0, 1, 1, 0, 2]
    size_relabel_run(XnDriver(xn_native_v1), Int(tie.unsafe_ptr()), 5, 3)
    var want_tie: List[Int32] = [0, 1, 1, 0, 2]
    for i in range(5):  # small-loop: unchanged
        require(tie[i] == want_tie[i], "by size, tie: node " + String(i) + " has " + String(tie[i]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seams DEVIATION 5314 and 5315 of the decomp lane: the undirected edge
weight of the shortest-path rows (5314; the visiting order's tie cannot
reach the output, since each distance is the exact minimum over sums
computed the same way, which is why the oracle here is a different
algorithm) and the barycenter weights' regularization (5315).

    tools/with_identical_mode.sh pixi run mojo run -I . x_decomp/checks/graph_check.mojo
"""
from core.identity_trace import IdentityTrace
from x_decomp.checks.xd_oracles import oracle_barycenter, oracle_dijkstra
from x_decomp.checks.seam_util import count_diff_f32, lcg_unit, positive_fixture, ptr, require_separates, same, seam_fixture, zeros
from x_decomp.device import DevExec
from x_decomp.host import HostExec


def main() raises:
    var tr = IdentityTrace()
    tr.header("x_decomp graph_check (DEVIATIONS 5314, 5315)")
    # ---- 5314 an ASYMMETRIC sparse weight matrix, one unreachable node
    var n = 19
    var st = UInt64(5)
    var W = zeros(n * n)
    for i in range(n - 1):
        for j in range(n - 1):
            var u = lcg_unit(st)
            if i != j and u < Float32(0.25):
                W[i * n + j] = Float32(0.5) + u * Float32(8)
    var want = oracle_dijkstra(W, n)
    require_separates("5314 undirected edge weight", count_diff_f32(want, oracle_dijkstra(W, n, 1)))
    var dd = zeros(n * n)
    var rr = zeros(n)
    DevExec.dijkstra_rows(ptr(W), ptr(dd), ptr(rr), n)
    same("5314 dijkstra device", count_diff_f32(dd, want))
    var dh = zeros(n * n)
    HostExec.dijkstra_rows(ptr(W), ptr(dh), ptr(rr), n)
    same("5314 dijkstra host", count_diff_f32(dh, want))
    var rh = zeros(n)
    DevExec.dijkstra_rows(ptr(W), ptr(dd), ptr(rr), n)
    HostExec.dijkstra_rows(ptr(W), ptr(dh), ptr(rh), n)
    same("5314 dijkstra reached counts host", count_diff_f32(rh, rr))
    # the host's heap spelling (x_decomp/host_graph.mojo) on a larger graph
    # with EXACT distance ties (integer weights, many equal-length paths)
    # and two components
    var n2 = 150
    var W2 = zeros(n2 * n2)
    var st2 = UInt64(11)
    for i in range(n2):
        for j in range(n2):
            var u = lcg_unit(st2)
            if i != j and (i < 100) == (j < 100) and u < Float32(0.06):
                W2[i * n2 + j] = Float32(1 + Int(u * Float32(50)) % 3)
    var want2 = oracle_dijkstra(W2, n2)
    var d2 = zeros(n2 * n2)
    var r2 = zeros(n2)
    DevExec.dijkstra_rows(ptr(W2), ptr(d2), ptr(r2), n2)
    same("5314 dijkstra (ties) device", count_diff_f32(d2, want2))
    var h2 = zeros(n2 * n2)
    var hr2 = zeros(n2)
    HostExec.dijkstra_rows(ptr(W2), ptr(h2), ptr(hr2), n2)
    same("5314 dijkstra (ties) host", count_diff_f32(h2, want2))
    same("5314 dijkstra (ties) reached counts host", count_diff_f32(hr2, r2))
    tr.record_list_f32("x_decomp.dijkstra", dd)
    # ---- 5315 barycenter weights
    var nq = 13
    var ny = 30
    var d = 6
    var k = 5
    var X = positive_fixture(nq, d, 3)
    var Y = positive_fixture(ny, d, 4)
    var nbr = List[Int]()
    var nbrf = List[Float32]()
    for i in range(nq):
        for a in range(k):
            var j = (i * 7 + a * 3) % ny
            nbr.append(j)
            nbrf.append(Float32(j))
    var reg = Float32(0.001)
    var wb = oracle_barycenter(X, Y, nbr, nq, d, k, reg)
    require_separates("5315 barycenter regularization", count_diff_f32(wb, oracle_barycenter(X, Y, nbr, nq, d, k, reg, 1)))
    var bd = zeros(nq * k)
    var fl = zeros(nq)
    DevExec.barycenter_rows(ptr(X), ptr(Y), ptr(nbrf), ptr(bd), ptr(fl), nq, ny, d, k, reg)
    same("5315 barycenter device", count_diff_f32(bd, wb))
    var bh = zeros(nq * k)
    HostExec.barycenter_rows(ptr(X), ptr(Y), ptr(nbrf), ptr(bh), ptr(fl), nq, ny, d, k, reg)
    same("5315 barycenter host", count_diff_f32(bh, wb))
    tr.record_list_f32("x_decomp.barycenter", bd)
    print("PASS x_decomp graph_check")

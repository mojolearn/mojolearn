# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seams DEVIATION 5117 (the Lance-Williams update: Float64 from the Float32 matrix, ward's three pinned products summed left to right, minus last, one quotient; average's two pinned products, one quotient) and 5118 (the merge order: the lowest live pair, the lowest i then j on a tie).

    tools/with_identical_mode.sh pixi run mojo run -I . x_cluster/checks/agglo_check.mojo

Pass 2, lane/algos-cluster. The fixture is first shown to SEPARATE the pinned
spelling from the alternative (scipy's reciprocal spelling of ward and
average; the LAST lowest pair on a tie), VACUOUS otherwise; then the device
column (`DeviceOps`) and the CPU column (`HostOps`) running
`x_cluster/agglo.mojo::agglo_tree` must each equal the host oracle
(`x_cluster/checks/oracles.mojo::oracle_agglo`) bit for bit under IDENTICAL,
in children and merge values, for all four linkages, unconstrained and
under a two-component chain graph (the component join reached) with a
partial tree. With MOJOLEARN_IDENTITY_TRACE set, the stage lands on the
card."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, identical_sqrt
from core.identity_trace import IdentityTrace
from x_cluster.agglo import agglo_tree
from x_cluster.checks.oracles import *
from x_cluster.checks.seam_util import count_diff_f32, count_diff_i32, require_equal, require_separates, seam_fixture
from x_cluster.device_ops import DeviceOps
from x_cluster.host.host_ops import HostOps
from x_cluster.ops import ClusterOps


def _same(seam: String, differing: Int) raises:
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        require_equal(seam, differing)
    else:
        print("  " + seam + ": FAST, " + String(differing) + " cells differ from the oracle (no claim)")


def _chain_edges(n: Int, cut: Int) -> List[Float32]:
    """i -- i + 1 for every i but `cut`: two components."""
    var e = List[Float32]()
    for i in range(n - 1):
        if i != cut:
            e.append(Float32(i))
            e.append(Float32(i + 1))
    return e^


def _joined_mask(dist: List[Float32], n: Int, cut: Int, rooted: Bool) -> List[Bool]:
    """The chain's mask plus scikit-learn's component join, restated: the
    components are 0..cut and cut+1..n-1 (found from vertex 0 in that order);
    the pair (a in the second, b in the first) at the lowest distance, the
    first in (a, b) ascending order on a tie."""
    var adj = List[Bool](length=n * n, fill=False)
    for i in range(n - 1):
        if i != cut:
            adj[i * n + i + 1] = True
            adj[(i + 1) * n + i] = True
    var ba = -1
    var bb = -1
    var bv = Float32(0)
    for a in range(cut + 1, n):
        for b in range(cut + 1):
            var v = identical_sqrt(dist[a * n + b]) if rooted else dist[a * n + b]
            if ba < 0 or v < bv:
                ba = a
                bb = b
                bv = v
    adj[ba * n + bb] = True
    adj[bb * n + ba] = True
    return adj^


def run[O: ClusterOps](
    mut ops: O, x: List[Float32], n: Int, d: Int, linkage: Int, edges: List[Float32], n_edges: Int, n_merges: Int,
    mut children: List[Int32],
) raises -> List[Float32]:
    var dist = List[Float32]()
    var n_cc = 0
    agglo_tree(ops, x, n, d, linkage, -1, Float32(2), edges, n_edges, n_merges, children, dist, n_cc)
    if n_edges >= 0 and n_cc != 2:
        raise Error("agglo_check: the chain graph has 2 components, the driver found " + String(n_cc))
    return dist^


def main() raises:
    var n = 40
    var d = 5
    var cut = 17
    var x = seam_fixture(n, d, 31)
    # rows 5 and 6 copy row 1 (rows 1 and 2 already tie): several pairs at
    # distance 0, so the merge order's tie rule decides the first merges
    for f in range(d):
        x[5 * d + f] = x[1 * d + f]
        x[6 * d + f] = x[1 * d + f]
    var sq = oracle_sqdist(x, n, x, n, d)
    var rt = List[Float32](capacity=n * n)
    for v in sq:
        rt.append(identical_sqrt(v if v > Float32(0) else Float32(0)))
    var edges = _chain_edges(n, cut)
    var n_edges = len(edges) // 2
    var none = List[Bool]()
    # the separation first: over all eight arms the pinned spelling and the
    # alternative must differ somewhere
    var sep_lw = 0
    var sep_tie = 0
    for linkage in range(4):
        var dm = sq.copy() if linkage == 0 else rt.copy()
        for constrained in range(2):
            var mask = _joined_mask(dm, n, cut, linkage == 0) if constrained == 1 else none.copy()
            var m = n - 3 if constrained == 1 else n - 1
            var want_c = List[Int32]()
            var want_d = List[Float32]()
            oracle_agglo(dm, n, linkage, mask, m, want_c, want_d)
            var alt_c = List[Int32]()
            var alt_d = List[Float32]()
            oracle_agglo(dm, n, linkage, mask, m, alt_c, alt_d, scipy_spelling=True)
            sep_lw += count_diff_f32(want_d, alt_d)
            oracle_agglo(dm, n, linkage, mask, m, alt_c, alt_d, last_on_tie=True)
            sep_tie += count_diff_i32(want_c, alt_c)
    require_separates("5117 Lance-Williams spelling (ward, average)", sep_lw)
    require_separates("5118 merge order on a tie", sep_tie)
    var tr = IdentityTrace()
    for linkage in range(4):
        var dm = sq.copy() if linkage == 0 else rt.copy()
        for constrained in range(2):
            var mask = _joined_mask(dm, n, cut, linkage == 0) if constrained == 1 else none.copy()
            var m = n - 3 if constrained == 1 else n - 1
            var want_c = List[Int32]()
            var want_d = List[Float32]()
            oracle_agglo(dm, n, linkage, mask, m, want_c, want_d)
            var tag = "linkage " + String(linkage) + (" constrained" if constrained == 1 else "")
            var ne = n_edges if constrained == 1 else -1
            var dev = DeviceOps()
            var dc = List[Int32]()
            var dd = run(dev, x, n, d, linkage, edges, ne, m, dc)
            _same("5117/5118 agglo device children " + tag, count_diff_i32(dc, want_c))
            _same("5117/5118 agglo device distances " + tag, count_diff_f32(dd, want_d))
            var host = HostOps()
            var hc = List[Int32]()
            var hd = run(host, x, n, d, linkage, edges, ne, m, hc)
            _same("5117/5118 agglo host children " + tag, count_diff_i32(hc, want_c))
            _same("5117/5118 agglo host distances " + tag, count_diff_f32(hd, want_d))
            tr.record_list_i32("x_cluster.agglo.children", dc)
            tr.record_list_f32("x_cluster.agglo.distances", dd)
    print("PASS x_cluster agglo_check")

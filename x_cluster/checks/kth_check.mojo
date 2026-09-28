# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seams DEVIATION 5103 and 5120: the row order statistic, exact whatever the order of the row (ties and -0.0 planted). 5103 is the host's bisection on the float bits; 5120 the device's block radix select, which must give the same value on a long row (many values per thread, one high byte shared by thousands, an +inf and a repeated run).

    tools/with_identical_mode.sh pixi run mojo run -I . x_cluster/checks/kth_check.mojo

Pass 2, lane/algos-cluster. The fixture is first shown to SEPARATE the pinned
spelling from the alternative (VACUOUS otherwise); then the device column
(`DeviceOps`) and the CPU column (`HostOps`) must each equal the host oracle
(`x_cluster/checks/oracles.mojo`) bit for bit under IDENTICAL (FAST: reported,
no claim). With MOJOLEARN_IDENTITY_TRACE set, the stage lands on the card."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.identity_trace import IdentityTrace
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


def run[O: ClusterOps](mut ops: O, m: List[Float32], rows: Int, cols: Int, k: Int) raises -> List[Float32]:
    var sm = ops.put(m)
    var so = ops.zeros(rows)
    ops.kth(sm, rows, cols, k, so)
    return ops.get(so, rows)


def main() raises:
    var d = 4
    var a = seam_fixture(40, d, 5)
    var m = oracle_sqdist(a, 40, a, 40, d)
    m[3] = Float32(-0.0)
    var ks: List[Int] = [1, 2, 7, 20, 40]
    for k in ks:
        var want = oracle_kth(m, 40, 40, k)
        var off = oracle_kth(m, 40, 40, k + 1 if k < 40 else k - 1)
        require_separates("5103 kth k=" + String(k), count_diff_f32(want, off))
        var dev = DeviceOps()
        var got = run(dev, m, 40, 40, k)
        _same("5103 kth device k=" + String(k), count_diff_f32(got, want))
        var host = HostOps()
        _same("5103 kth host k=" + String(k), count_diff_f32(run(host, m, 40, 40, k), want))
        var tr = IdentityTrace()
        tr.record_list_f32("x_cluster.kth", got)
    # 5120: the long rows (more values than the block has threads; the
    # pairwise squares of 80 fixture rows, 6400 per row, three rows)
    var b = seam_fixture(80, d, 11)
    var big = oracle_sqdist(b, 80, b, 80, d)
    var rows = 3
    var cols = 80 * 80 // rows
    big[5] = Float32.MAX * Float32(2)
    for j in range(100, 400):
        big[cols + j] = Float32(0.5)
    var lks: List[Int] = [60, 150, 400, cols // 2, cols // 2 + 1, cols - 1, cols]
    for k in lks:
        var want = oracle_kth(big, rows, cols, k)
        var off = oracle_kth(big, rows, cols, k + 1 if k < cols else k - 1)
        require_separates("5120 kth long k=" + String(k), count_diff_f32(want, off))
        var dev = DeviceOps()
        var got = run(dev, big, rows, cols, k)
        _same("5120 kth device long k=" + String(k), count_diff_f32(got, want))
        var host = HostOps()
        _same("5120 kth host long k=" + String(k), count_diff_f32(run(host, big, rows, cols, k), want))
        var tr = IdentityTrace()
        tr.record_list_f32("x_cluster.kth_long", got)
    # the host's selection (cluster-cpu lane, 2026-09-28): rows long enough
    # to split over tasks, ties from duplicated rows
    var nr = 300
    var c = seam_fixture(nr, d, 6)
    for f in range(d):
        c[(nr - 1) * d + f] = c[f]
        c[(nr - 2) * d + f] = c[d + f]
    var mc = oracle_sqdist(c, nr, c, nr, d)
    mc[3] = Float32(-0.0)
    var ks2: List[Int] = [1, 3, 10, 150, 299, 300]
    for k in ks2:
        var want = oracle_kth(mc, nr, nr, k)
        var off = oracle_kth(mc, nr, nr, k + 1 if k < nr else k - 1)
        require_separates("5103 kth n=300 k=" + String(k), count_diff_f32(want, off))
        var dev = DeviceOps()
        var got = run(dev, mc, nr, nr, k)
        _same("5103 kth device n=300 k=" + String(k), count_diff_f32(got, want))
        var host = HostOps()
        _same("5103 kth host n=300 k=" + String(k), count_diff_f32(run(host, mc, nr, nr, k), want))
        var tr = IdentityTrace()
        tr.record_list_f32("x_cluster.kth_n300", got)
    print("PASS x_cluster kth_check")

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seam DEVIATION 5103: the row order statistic by a bisection on the float bits, exact whatever the order of the row (ties and -0.0 planted).

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
    print("PASS x_cluster kth_check")

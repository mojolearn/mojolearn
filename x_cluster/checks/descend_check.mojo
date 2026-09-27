# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seam DEVIATION 5107: BisectingKMeans predict descends to the LEFT child on an exact distance tie.

    tools/with_identical_mode.sh pixi run mojo run -I . x_cluster/checks/descend_check.mojo

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


def run[O: ClusterOps](mut ops: O, x: List[Float32], n: Int, d: Int, c: List[Float32], nodes: List[Int32]) raises -> List[Int32]:
    var sx = ops.put(x)
    var sc = ops.put(c)
    var sn = ops.put_i(nodes)
    var sl = ops.zeros_i(n)
    ops.descend(sx, n, d, sc, sn, sl)
    return ops.get_i(sl, n)


def main() raises:
    var d = 2
    var n = 64
    var x = seam_fixture(n, d, 11)
    # root 0 -> (1, 2); node 1 -> (3, 4); leaves 2, 3, 4 labelled 0, 1, 2.
    # Children of each split mirror each other through the origin in the first
    # feature, so every row with a zero first feature ties (column d - 1 is zero)
    var c = List[Float32]()
    var cs: List[Float32] = [0, 0, 5, 0, -5, 0, 0, 7, 0, -7]
    for v in cs:
        c.append(v)
    var nodes: List[Int32] = [1, 2, -1, 3, 4, -1, -1, -1, 0, -1, -1, 1, -1, -1, 2]
    for i in range(n):
        x[i * d + 0] = Float32(0)
    var want = oracle_descend(x, n, d, c, nodes)
    require_separates("5107 descend tie", count_diff_i32(want, oracle_descend(x, n, d, c, nodes, True)))
    var dev = DeviceOps()
    var got = run(dev, x, n, d, c, nodes)
    _same("5107 descend device", count_diff_i32(got, want))
    var host = HostOps()
    _same("5107 descend host", count_diff_i32(run(host, x, n, d, c, nodes), want))
    var tr = IdentityTrace()
    tr.record_list_i32("x_cluster.descend", got)
    print("PASS x_cluster descend_check")

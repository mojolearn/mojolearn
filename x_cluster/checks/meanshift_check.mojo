# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seam DEVIATION 5104: the mean-shift seed loop, the flat-kernel fold over the rows ascending, one quotient per feature, the rooted shift test.

    tools/with_identical_mode.sh pixi run mojo run -I . x_cluster/checks/meanshift_check.mojo

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


def run[O: ClusterOps](mut ops: O, x: List[Float32], n: Int, d: Int, bw: Float32, stop: Float32) raises -> List[Float32]:
    var sx = ops.put(x)
    var sc = ops.put(x)
    var ss = ops.zeros(n * d)
    var si = ops.zeros_i(n)
    var st = ops.zeros_i(n)
    ops.meanshift(sx, n, d, bw, stop, 300, sc, n, ss, si, st)
    return ops.get(sc, n * d)


def main() raises:
    var n = 96
    var d = 3
    var x = seam_fixture(n, d, 7)
    # a bandwidth that takes in the 1e3-scale column's neighbors
    var bw = Float32(600)
    var stop = Float32(0.6)
    var want = oracle_meanshift(x, n, d, bw, stop, 300, x, n)
    require_separates("5104 meanshift fold order", count_diff_f32(want, oracle_meanshift(x, n, d, bw, stop, 300, x, n, True)))
    var dev = DeviceOps()
    var got = run(dev, x, n, d, bw, stop)
    _same("5104 meanshift device", count_diff_f32(got, want))
    var host = HostOps()
    _same("5104 meanshift host", count_diff_f32(run(host, x, n, d, bw, stop), want))
    var tr = IdentityTrace()
    tr.record_list_f32("x_cluster.meanshift", got)
    print("PASS x_cluster meanshift_check")

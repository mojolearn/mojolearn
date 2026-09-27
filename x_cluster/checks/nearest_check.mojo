# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seam DEVIATION 5102: the nearest-row argmin keeps the LOWEST index on an exact tie.

    tools/with_identical_mode.sh pixi run mojo run -I . x_cluster/checks/nearest_check.mojo

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


def run[O: ClusterOps](mut ops: O, a: List[Float32], b: List[Float32], d: Int) raises -> List[Int32]:
    var sa = ops.put(a)
    var sb = ops.put(b)
    var sl = ops.zeros_i(64)
    var sd = ops.zeros(64)
    ops.nearest(sa, 64, sb, 12, d, sl, sd)
    return ops.get_i(sl, 64)


def main() raises:
    var d = 5
    var a = seam_fixture(64, d, 3)
    # centers: rows of a, the first six repeated, so every row near them ties exactly
    var b = List[Float32]()
    for r in range(12):
        for f in range(d):
            b.append(a[(r % 6) * d + f])
    var dist = oracle_sqdist(a, 64, b, 12, d)
    var want = oracle_nearest(dist, 64, 12)
    require_separates("5102 argmin tie", count_diff_i32(want, oracle_nearest(dist, 64, 12, True)))
    var dev = DeviceOps()
    var got = run(dev, a, b, d)
    _same("5102 nearest device", count_diff_i32(got, want))
    var host = HostOps()
    _same("5102 nearest host", count_diff_i32(run(host, a, b, d), want))
    var tr = IdentityTrace()
    tr.record_list_i32("x_cluster.nearest", got)
    print("PASS x_cluster nearest_check")

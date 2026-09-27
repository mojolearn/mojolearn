# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seam DEVIATION 5100/5101: the squared distance, features folded ascending with the pinned product (no fused multiply-add).

    tools/with_identical_mode.sh pixi run mojo run -I . x_cluster/checks/dist_check.mojo

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


def run[O: ClusterOps](mut ops: O, a: List[Float32], b: List[Float32], d: Int) raises -> List[Float32]:
    var sa = ops.put(a)
    var sb = ops.put(b)
    var so = ops.zeros(64 * 9)
    ops.sqdist(sa, 64, sb, 9, d, so)
    return ops.get(so, 64 * 9)


def main() raises:
    var d = 7
    var a = seam_fixture(64, d, 1)
    var b = seam_fixture(9, d, 2)
    var want = oracle_sqdist(a, 64, b, 9, d)
    require_separates("5100 distance fold order", count_diff_f32(want, oracle_sqdist(a, 64, b, 9, d, 1)))
    require_separates("5101 distance contraction", count_diff_f32(want, oracle_sqdist(a, 64, b, 9, d, 2)))
    var dev = DeviceOps()
    var got = run(dev, a, b, d)
    _same("5100/5101 sqdist device", count_diff_f32(got, want))
    var host = HostOps()
    _same("5100/5101 sqdist host", count_diff_f32(run(host, a, b, d), want))
    var tr = IdentityTrace()
    tr.record_list_f32("x_cluster.sqdist", got)
    print("PASS x_cluster dist_check")

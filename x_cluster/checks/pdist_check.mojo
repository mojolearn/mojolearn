# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seam DEVIATION 5111: the non-euclidean metrics (manhattan, chebyshev, minkowski p, cosine), every fold over the features ascending, the portable pow and sqrt, cosine's zero-norm rows at distance 1, the clamp at +0.

    tools/with_identical_mode.sh pixi run mojo run -I . x_cluster/checks/pdist_check.mojo

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


def run[O: ClusterOps](mut ops: O, a: List[Float32], b: List[Float32], d: Int, metric: Int, p: Float32) raises -> List[Float32]:
    var sa = ops.put(a)
    var sb = ops.put(b)
    var so = ops.zeros(48 * 11)
    ops.pdist(sa, 48, sb, 11, d, metric, p, so)
    return ops.get(so, 48 * 11)


def main() raises:
    var d = 7
    var a = seam_fixture(48, d, 21)
    var b = seam_fixture(11, d, 22)
    for i in range(d):
        b[3 * d + i] = Float32(0)
    var ms: List[Int] = [1, 3, 4]
    for metric in ms:
        var p = Float32(3)
        var want = oracle_pdist(a, 48, b, 11, d, metric, p)
        require_separates("5111 pdist fold order metric " + String(metric), count_diff_f32(want, oracle_pdist(a, 48, b, 11, d, metric, p, True)))
        var dev = DeviceOps()
        var got = run(dev, a, b, d, metric, p)
        _same("5111 pdist device metric " + String(metric), count_diff_f32(got, want))
        var host = HostOps()
        _same("5111 pdist host metric " + String(metric), count_diff_f32(run(host, a, b, d, metric, p), want))
        var tr = IdentityTrace()
        tr.record_list_f32("x_cluster.pdist", got)
    # chebyshev is order-free by construction: equality alone
    var want2 = oracle_pdist(a, 48, b, 11, d, 2, Float32(3))
    var dev2 = DeviceOps()
    _same("5111 pdist device chebyshev", count_diff_f32(run(dev2, a, b, d, 2, Float32(3)), want2))
    print("PASS x_cluster pdist_check")

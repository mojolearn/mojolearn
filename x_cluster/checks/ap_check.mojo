# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seams DEVIATION 5105/5106: affinity propagation's damped updates as two pinned products and one add, and the availability column fold ascending.

    tools/with_identical_mode.sh pixi run mojo run -I . x_cluster/checks/ap_check.mojo

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


def run[O: ClusterOps](mut ops: O, s: List[Float32], n: Int, iters: Int) raises -> List[Float32]:
    var ss = ops.put(s)
    var sa = ops.zeros(n * n)
    var sr = ops.zeros(n * n)
    for _t in range(iters):
        ops.ap_r(ss, sa, sr, n, Float32(0.7))
        ops.ap_a(sr, sa, n, Float32(0.7))
    var out = ops.get(sa, n * n)
    for v in ops.get(sr, n * n):
        out.append(v)
    return out^


def oracle_run(s: List[Float32], n: Int, iters: Int, alt: Int) -> List[Float32]:
    var a = List[Float32](length=n * n, fill=Float32(0))
    var r = List[Float32](length=n * n, fill=Float32(0))
    for _t in range(iters):
        oracle_ap_step(s, a, r, n, Float32(0.7), alt)
    var out = a.copy()
    for v in r:
        out.append(v)
    return out^


def main() raises:
    var n = 48
    var d = 4
    var x = seam_fixture(n, d, 9)
    var s = oracle_sqdist(x, n, x, n, d)
    for t in range(n * n):
        s[t] = -s[t]
    for i in range(n):
        s[i * n + i] = Float32(-2.5e5)
    var want = oracle_run(s, n, 6, 0)
    require_separates("5105 AP damping contraction", count_diff_f32(want, oracle_run(s, n, 6, 1)))
    require_separates("5106 AP availability fold order", count_diff_f32(want, oracle_run(s, n, 6, 2)))
    var dev = DeviceOps()
    var got = run(dev, s, n, 6)
    _same("5105/5106 AP device", count_diff_f32(got, want))
    var host = HostOps()
    _same("5105/5106 AP host", count_diff_f32(run(host, s, n, 6), want))
    var tr = IdentityTrace()
    tr.record_list_f32("x_cluster.ap_a_r", got)
    _block_case()
    print("PASS x_cluster ap_check")


def _block_case() raises:
    """The device's block-per-row responsibility kernel past one block's
    width (n = 600 > 256 threads, strided) with PLANTED TIES: rows 2, 5 and
    300 copy row 1, so row 1's max and second max are each a tie among
    exact duplicates (-0.0 distances), and the lowest index must win."""
    var n = 600
    var d = 3
    var x = seam_fixture(n, d, 11)
    for r in [2, 5, 300]:
        for f in range(d):
            x[r * d + f] = x[1 * d + f]
    var s = oracle_sqdist(x, n, x, n, d)
    for t in range(n * n):
        s[t] = -s[t]
    for i in range(n):
        s[i * n + i] = Float32(-2.5e5)
    var want = oracle_run(s, n, 3, 0)
    var dev = DeviceOps()
    var got = run(dev, s, n, 3)
    _same("5105 AP block-per-row device (ties, n > block)", count_diff_f32(got, want))
    var host = HostOps()
    _same("5105 AP block-per-row host", count_diff_f32(run(host, s, n, 3), want))
    var tr = IdentityTrace()
    tr.record_list_f32("x_cluster.ap_a_r.block", got)

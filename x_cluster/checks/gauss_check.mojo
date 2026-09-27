# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seams DEVIATION 5108/5109: the Mahalanobis fold against the upper-triangular precision factor (difference first, ascending) and the E-step log-sum-exp (row max, ascending sum of the portable exp, portable log).

    tools/with_identical_mode.sh pixi run mojo run -I . x_cluster/checks/gauss_check.mojo

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


def run[O: ClusterOps](
    mut ops: O, x: List[Float32], n: Int, d: Int, m: List[Float32], p: List[Float32], c: List[Float32], kc: Int
) raises -> List[Float32]:
    var sx = ops.put(x)
    var sm = ops.put(m)
    var sp = ops.put(p)
    var sc = ops.put(c)
    var sq = ops.zeros(n * kc)
    var sl = ops.zeros(n)
    ops.gauss_q(sx, n, d, sm, sp, kc, sq)
    var q = ops.get(sq, n * kc)
    ops.resp(sq, sc, n, kc, sl)
    for v in ops.get(sq, n * kc):
        q.append(v)
    return q^


def main() raises:
    var n = 80
    var d = 4
    var kc = 5
    var x = seam_fixture(n, d, 13)
    var m = seam_fixture(kc, d, 14)
    var p = List[Float32](length=kc * d * d, fill=Float32(0))
    for k in range(kc):
        for a in range(d):
            for j in range(a, d):
                # scaled against the column's magnitude (seam_fixture: 1, 1e3, 1e-3)
                # so every term of the inner fold is comparable and its order shows
                var inv = Float32(1) if a % 3 == 0 else (Float32(0.001) if a % 3 == 1 else Float32(1000))
                p[k * d * d + a * d + j] = Float32(0.0003) * inv * Float32(1 + a + 2 * j + k) if a != j else Float32(0.0007) * inv * Float32(1 + k)
    var c: List[Float32] = [0.1, 0.35, -0.2, -0.0, 0.3]
    var q = oracle_gauss_q(x, n, d, m, p, kc)
    require_separates("5108 gauss fold (difference first)", count_diff_f32(q, oracle_gauss_q(x, n, d, m, p, kc, True)))
    require_separates("5108 gauss fold order", count_diff_f32(q, oracle_gauss_q(x, n, d, m, p, kc, False, True)))
    var lr = oracle_resp(q, c, n, kc)
    require_separates("5109 log-sum-exp order", count_diff_f32(lr, oracle_resp(q, c, n, kc, True)))
    var want = q.copy()
    for v in lr:
        want.append(v)
    var dev = DeviceOps()
    var got = run(dev, x, n, d, m, p, c, kc)
    _same("5108/5109 gauss+resp device", count_diff_f32(got, want))
    var host = HostOps()
    _same("5108/5109 gauss+resp host", count_diff_f32(run(host, x, n, d, m, p, c, kc), want))
    var tr = IdentityTrace()
    tr.record_list_f32("x_cluster.gauss_q_resp", got)
    print("PASS x_cluster gauss_check")

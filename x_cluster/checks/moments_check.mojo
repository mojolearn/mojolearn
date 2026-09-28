# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seam DEVIATION 5110: the mixture M-step moments (nk, means, covariances), every fold over the rows ascending.

    tools/with_identical_mode.sh pixi run mojo run -I . x_cluster/checks/moments_check.mojo

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


def run[O: ClusterOps](mut ops: O, resp: List[Float32], x: List[Float32], n: Int, d: Int, kc: Int) raises -> List[Float32]:
    var sr = ops.put(resp)
    var sx = ops.put(x)
    var sn = ops.zeros(kc)
    var sm = ops.zeros(kc * d)
    var sc = ops.zeros(kc * d * d)
    ops.moments(sr, sx, n, d, kc, Float32(1e-6), sn, sm, sc)
    var out = ops.get(sn, kc)
    for v in ops.get(sm, kc * d):
        out.append(v)
    for v in ops.get(sc, kc * d * d):
        out.append(v)
    return out^


def _case(n: Int, d: Int, kc: Int, seed: UInt64, tag: String) raises:
    var x = seam_fixture(n, d, seed)
    var u = seam_fixture(n, kc, seed + 1)
    var resp = List[Float32](capacity=n * kc)
    for i in range(n):
        var s = Float32(0)
        for k in range(kc):
            s = s + abs(u[i * kc + k]) + Float32(0.01)
        for k in range(kc):
            resp.append((abs(u[i * kc + k]) + Float32(0.01)) / s)
    var want = oracle_moments(resp, x, n, d, kc, Float32(1e-6))
    require_separates(tag + " moments fold order", count_diff_f32(want, oracle_moments(resp, x, n, d, kc, Float32(1e-6), True)))
    var dev = DeviceOps()
    var got = run(dev, resp, x, n, d, kc)
    _same(tag + " moments device", count_diff_f32(got, want))
    var host = HostOps()
    _same(tag + " moments host", count_diff_f32(run(host, resp, x, n, d, kc), want))
    var tr = IdentityTrace()
    tr.record_list_f32("x_cluster.moments." + tag, got)


def main() raises:
    _case(200, 3, 4, 15, "5110")
    # 5121: the device's tiled kernel over several row tiles (T = 256 rows
    # at d = 5), more covariance chains than the block has threads (d = 17:
    # 289), and the one-thread-per-cell fallback past MOM_MAX_D (d = 65)
    _case(700, 5, 3, 21, "5121-tiles")
    _case(600, 17, 2, 23, "5121-chains")
    _case(40, 65, 2, 25, "5121-fallback")
    # d = 19: the host's vector lanes (two groups of eight) and its scalar
    # tail (cluster-cpu lane, 2026-09-28); n large enough to split tasks
    _case(3000, 19, 3, 17, "host-d19")
    print("PASS x_cluster moments_check")

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


def _shape(n: Int, d: Int, kc: Int, sx: UInt64, su: UInt64) raises:
    var x = seam_fixture(n, d, sx)
    var u = seam_fixture(n, kc, su)
    var resp = List[Float32](capacity=n * kc)
    for i in range(n):
        var s = Float32(0)
        for k in range(kc):
            s = s + abs(u[i * kc + k]) + Float32(0.01)
        for k in range(kc):
            resp.append((abs(u[i * kc + k]) + Float32(0.01)) / s)
    var want = oracle_moments(resp, x, n, d, kc, Float32(1e-6))
    var tag = " d=" + String(d)
    require_separates("5110 moments fold order" + tag, count_diff_f32(want, oracle_moments(resp, x, n, d, kc, Float32(1e-6), True)))
    var dev = DeviceOps()
    var got = run(dev, resp, x, n, d, kc)
    _same("5110 moments device" + tag, count_diff_f32(got, want))
    var host = HostOps()
    _same("5110 moments host" + tag, count_diff_f32(run(host, resp, x, n, d, kc), want))
    var tr = IdentityTrace()
    tr.record_list_f32("x_cluster.moments" if d == 3 else "x_cluster.moments_d" + String(d), got)


def main() raises:
    _shape(200, 3, 4, 15, 16)
    # d = 19: the host's vector lanes (two groups of eight) and its scalar
    # tail (cluster-cpu lane, 2026-09-28); n large enough to split tasks.
    _shape(3000, 19, 3, 17, 18)
    print("PASS x_cluster moments_check")

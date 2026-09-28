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


def _flag_case() raises:
    """The speculative chain's fallback (device_ops `_moments_pass_kernel`):
    feature 0 alternates +1.0000001e-37 and -1e-37 (each addend 0.5x,
    normal) so every pair's partial sum is subnormal, which `chain_add`
    flushes to zero and a plain add keeps; feature 1 alternates +1 and -1,
    an exact cancellation to zero from nonzero operands in rows 0 and 1,
    then mixed-scale values. A chain that skipped the fallback would leave a
    subnormal mean where the oracle has zero (on a vendor that keeps
    subnormals); arm 5121_moments_flag re-adds a flagged tile in reverse, so
    it bites wherever the flag fires, every vendor."""
    var n = 600
    var d = 2
    var kc = 2
    var mixed = seam_fixture(n, 1, 27)
    var x = List[Float32](capacity=n * d)
    var resp = List[Float32](capacity=n * kc)
    for i in range(n):
        x.append(Float32(1.0000001e-37) if i % 2 == 0 else Float32(-1e-37))
        # rows 0 and 1 cancel exactly (flagged on every vendor), the rest
        # are mixed-scale, so a re-add in another order moves bits
        if i == 0:
            x.append(Float32(1))
        elif i == 1:
            x.append(Float32(-1))
        else:
            x.append(mixed[i])
        resp.append(Float32(0.5))
        resp.append(Float32(0.5))
    var want = oracle_moments(resp, x, n, d, kc, Float32(1e-6))
    # the fixture separates: a plain (unflushed) fold of feature 0 leaves a
    # nonzero sum where the pinned one has zero
    var plain = Float32(0)
    for i in range(n):
        plain = plain + Float32(0.5) * x[i * d]
    require_separates("5121-flag plain vs flushed chain", count_diff_f32([plain], [Float32(0)]))
    var dev = DeviceOps()
    var got = run(dev, resp, x, n, d, kc)
    _same("5121-flag moments device", count_diff_f32(got, want))
    var host = HostOps()
    _same("5121-flag moments host", count_diff_f32(run(host, resp, x, n, d, kc), want))
    var tr = IdentityTrace()
    tr.record_list_f32("x_cluster.moments.5121-flag", got)


def main() raises:
    _case(200, 3, 4, 15, "5110")
    # 5121: the device's tiled kernel over several row tiles (T = 256 rows
    # at d = 5), more covariance chains than the block has threads (d = 17:
    # 289), and the one-thread-per-cell fallback past MOM_MAX_D (d = 65)
    _case(700, 5, 3, 21, "5121-tiles")
    _case(600, 17, 2, 23, "5121-chains")
    _case(40, 65, 2, 25, "5121-fallback")
    # the speculative chain's flagged re-add
    _flag_case()
    print("PASS x_cluster moments_check")

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Seam DEVIATION 5122: AffinityPropagation's tie noise by the draw's counter, each cell on its own thread, the float32 of the unit draw rounded from the integer (no Float64).

    tools/with_identical_mode.sh pixi run mojo run -I . x_cluster/checks/ap_noise_check.mojo

Pass 2, lane/algos-cluster. The oracle is the stream as the driver drew it
before 5122 (`oracles.oracle_ap_noise`: one sequential SplitMix64, each unit
a Float64 narrowed by the compiler). The fixture is first shown to SEPARATE
the pinned rounding from truncation (VACUOUS otherwise: a Float64 draw
narrowed to float32 rounds UP about half the time); then the device column
(`DeviceOps`) and the CPU column (`HostOps`) must each equal the oracle bit
for bit under IDENTICAL (FAST: reported, no claim). 40,000 cells, seeds 0,
1 and 2^62 + 7 (a seed whose sums wrap)."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.identity_trace import IdentityTrace
from x_cluster.bodies import ap_noise_cell
from x_cluster.checks.oracles import *
from x_cluster.checks.seam_util import count_diff_f32, require_equal, require_separates, seam_fixture
from x_cluster.device_ops import DeviceOps
from x_cluster.host.host_ops import HostOps
from x_cluster.ops import ClusterOps


def _same(seam: String, differing: Int) raises:
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        require_equal(seam, differing)
    else:
        print("  " + seam + ": FAST, " + String(differing) + " cells differ from the oracle (no claim)")


def run[O: ClusterOps](mut ops: O, s: List[Float32], seed: UInt64) raises -> List[Float32]:
    var ss = ops.put(s)
    ops.ap_noise(ss, len(s), seed)
    return ops.get(ss, len(s))


def main() raises:
    var m = 40000
    var x = seam_fixture(200, 200, 31)
    var s = List[Float32](capacity=m)
    for t in range(m):
        s.append(-abs(x[t]) - Float32(0.25))
    var seeds: List[UInt64] = [UInt64(0), UInt64(1), (UInt64(1) << 62) + UInt64(7)]
    for seed in seeds:
        var want = oracle_ap_noise(s, seed)
        var trunc = s.copy()
        for t in range(m):
            ap_noise_cell[True](trunc.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), seed, t)
        require_separates("5122 AP noise rounding seed=" + String(seed), count_diff_f32(want, trunc))
        var dev = DeviceOps()
        var got = run(dev, s, seed)
        _same("5122 AP noise device seed=" + String(seed), count_diff_f32(got, want))
        var host = HostOps()
        _same("5122 AP noise host seed=" + String(seed), count_diff_f32(run(host, s, seed), want))
        var tr = IdentityTrace()
        tr.record_list_f32("x_cluster.ap_noise", got)
    print("PASS x_cluster ap_noise_check")

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The sparse arm past the dense bound: the GPU column against the CPU
column on one box, and a card of hashes to compare across boxes.

    tools/with_identical_mode.sh pixi run mojo run -I . hdbscan/checks/sparse_mr_card.mojo

DEVIATION 1620 at a size the dense arm refuses (default 70,000 rows, past
46,340 and past the 65,536 a 16-bit edge packing holds). Fits the fixture
with `fit_hdbscan` on the device and `hdbh_fit` on the CPU, both at the
default graph (auto takes the sparse arm here), and prints one FNV-1a
hash per output: labels, core distances, the condensed tree, the round
count, the probabilities. Under IDENTICAL the device and CPU hashes must
be equal (exit 1 otherwise); the `CARD` lines are what a second box's run
is compared with, line for line. MOJOLEARN_SMR_ROWS overrides the rows.
"""

from std.os import getenv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.identity_trace import IdentityTrace
from hdbscan.checks.sparse_mr_check import FIX_BLOBS, FIX_DUPS, fixture
from hdbscan.host.hdbscan_host_oracle import hdbh_fit
from hdbscan.impl.detail.extract import probabilities_from_labels_host
from hdbscan.impl.detail.select import CLUSTER_SELECTION_EOM
from hdbscan.impl.runner import GRAPH_BUILD_BRUTE_FORCE_KNN, HDBSCANParams, fit_hdbscan
from hierarchy.impl.cluster.detail.connectivities import DISTANCE_L2_SQRT_EXPANDED


def _fnv_i32(h0: UInt64, v: List[Int32]) -> UInt64:
    var h = h0
    for i in range(len(v)):
        var u = UInt64(Int(v[i]) & 0xFFFFFFFF)
        for b in range(4):
            h = (h ^ ((u >> UInt64(8 * b)) & 0xFF)) * 0x100000001B3
    return h


def _fnv_f32(h0: UInt64, v: List[Float32]) -> UInt64:
    var h = h0
    for i in range(len(v)):
        var u = UInt64(Int(rebind[UInt32](v[i].to_bits())))
        for b in range(4):
            h = (h ^ ((u >> UInt64(8 * b)) & 0xFF)) * 0x100000001B3
    return h


comptime FNV0: UInt64 = 0xCBF29CE484222325


def _hex64(v: UInt64) -> String:
    var digits = String("0123456789abcdef")
    var out = String("0x")
    var shift = 60
    while shift >= 0:
        var nib = Int((v >> UInt64(shift)) & UInt64(0xF))
        out += String(digits[byte=nib])
        shift -= 4
    return out


def main() raises:
    var m = 70000
    var env = getenv("MOJOLEARN_SMR_ROWS")
    if env != "":
        m = Int(env)
    var d = 6
    var ms = 10
    var mcs = 100
    var mode = "IDENTICAL" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "FAST"
    print("sparse_mr_card " + mode + " rows=" + String(m) + " d=" + String(d)
          + " min_samples=" + String(ms) + " min_cluster_size=" + String(mcs))
    # blobs, then a duplicated block at the end: zero-weight ties past 2^16
    var x = fixture(FIX_BLOBS, 5, m, d)
    var dup = fixture(FIX_DUPS, 9, 4000, d)
    for i in range(4000 * d):
        x[(m - 4000) * d + i] = dup[i]

    var ctx = DeviceContext()
    var params = HDBSCANParams(ms, mcs, 0, Float32(0.0), False, Float32(1.0),
                               CLUSTER_SELECTION_EOM, GRAPH_BUILD_BRUTE_FORCE_KNN)
    var xh = x.copy()
    var xd = ctx.enqueue_create_buffer[DType.float32](m * d)
    ctx.enqueue_copy(dst_buf=xd, src_ptr=x.unsafe_ptr())
    ctx.synchronize()
    var trace = IdentityTrace.disabled()
    var t0 = Int(perf_counter_ns())
    var g = fit_hdbscan(ctx, trace, xh, xd, m, d, DISTANCE_L2_SQRT_EXPANDED, params)
    var t1 = Int(perf_counter_ns())
    var gp = probabilities_from_labels_host(g.condensed, g.labels, g.inverse_label_map, m)
    print("device fit s=" + String(Float64(t1 - t0) / 1.0e9) + " clusters=" + String(g.n_clusters)
          + " noise=" + String(g.n_outliers) + " rounds=" + String(g.n_boruvka_rounds))

    t0 = Int(perf_counter_ns())
    var c = hdbh_fit(x, m, d, ms, mcs, 0, Float32(1.0), False, CLUSTER_SELECTION_EOM, Float32(0.0),
                     DISTANCE_L2_SQRT_EXPANDED)
    t1 = Int(perf_counter_ns())
    var cp = probabilities_from_labels_host(c.tree, c.labels, c.inverse_label_map, m)
    print("cpu fit s=" + String(Float64(t1 - t0) / 1.0e9) + " clusters=" + String(c.n_clusters)
          + " noise=" + String(c.n_outliers) + " rounds=" + String(c.n_boruvka_rounds))

    var names = List[String]()
    var gh = List[UInt64]()
    var ch = List[UInt64]()
    names.append("labels"); gh.append(_fnv_i32(FNV0, g.labels)); ch.append(_fnv_i32(FNV0, c.labels))
    names.append("core"); gh.append(_fnv_f32(FNV0, g.core_dists)); ch.append(_fnv_f32(FNV0, c.core_dists))
    names.append("tree"); gh.append(_fnv_f32(_fnv_i32(_fnv_i32(FNV0, g.condensed.parents), g.condensed.children), g.condensed.lambdas))
    ch.append(_fnv_f32(_fnv_i32(_fnv_i32(FNV0, c.tree.parents), c.tree.children), c.tree.lambdas))
    names.append("probabilities"); gh.append(_fnv_f32(FNV0, gp)); ch.append(_fnv_f32(FNV0, cp))
    var rg = List[Int32](); rg.append(Int32(g.n_boruvka_rounds)); rg.append(Int32(g.n_clusters))
    var rc = List[Int32](); rc.append(Int32(c.n_boruvka_rounds)); rc.append(Int32(c.n_clusters))
    names.append("rounds_clusters"); gh.append(_fnv_i32(FNV0, rg)); ch.append(_fnv_i32(FNV0, rc))
    var bad = 0
    for i in range(len(names)):
        var same = gh[i] == ch[i]
        print("CARD gpu " + names[i] + " " + _hex64(gh[i]))
        print("CARD cpu " + names[i] + " " + _hex64(ch[i]) + (" (== gpu)" if same else " (DIFFERS from gpu)"))
        if not same:
            bad += 1
    if bad != 0 and GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        raise Error("sparse_mr_card: " + String(bad) + " output(s) differ between the GPU and the CPU column")
    print("sparse_mr_card: GPU == CPU on every output" if bad == 0 else "sparse_mr_card: FAST, differences reported")

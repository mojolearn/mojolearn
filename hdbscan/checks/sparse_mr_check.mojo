# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The sparse mutual reachability MST equals the dense one, bit for bit.

    tools/with_identical_mode.sh pixi run mojo run -I . hdbscan/checks/sparse_mr_check.mojo

DEVIATION 1620's seam (`hdbscan/impl/detail/sparse_mr.mojo`). On inputs
small enough for the dense m x m graph, for every fixture:

  1. DEVICE LINKAGE. `build_mr_linkage` with graph=DENSE and graph=SPARSE:
     every MST edge (lo, hi) in the same slot, every weight's bits, the
     round count, the dendrogram's children, deltas and sizes.
  2. DEVICE SPARSE, MANY LAUNCHES. `sparse_mr_mst` with the launch bound
     cut to a few thousand multiply-adds, so every round takes many
     launches and many slices: the same edges and bits as (1).
  3. DEVICE FIT. `fit_hdbscan` both ways: labels, raw labels, core
     distances, the condensed tree, stabilities, the round count and the
     probabilities (`probabilities_from_labels`, both bindings' function).
  4. CPU. `hdbh_sparse_prim`'s edges against the DEVICE DENSE edges (the
     CPU column against the GPU column at the seam), and `hdbh_fit` with
     graph=DENSE against graph=SPARSE, output for output.

Fixtures: four seeds of hashed blobs (tie-free in practice), every row
duplicated (zero-weight ties), a small-integer grid (many equal weights),
and a grid at alpha = 1.3 with leaf selection. A tie fixture must hold
two tree edges of EQUAL weight or it is reported VACUOUS and fails: the
tie rule is the thing the sabotage arms break.

Under IDENTICAL every comparison is an assertion. Under FAST the dense
distances come from a vendor matmul and the sparse arm's from the pinned
chain, so (1)-(4) are REPORTED, not asserted (no cross-arm claim in FAST).

SABOTAGE (tools/identity_lanes/cluster.checks): the device kernel's
strict `<` made `<=` (a tie keeps the HIGHER j), and the CPU walk's
per-vertex update taking an equal key from ANY later tree vertex (the
triple dropped for `<=` on the key alone). Each must fail this driver.
(A first CPU arm swapped (lo, hi) to (hi, lo) and was INERT, measured on
the A40: for one vertex u both read its edges {c, u} in ascending c.)
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, identical_div
from core.identity_trace import IdentityTrace
from hdbscan.checks.hdbscan_sabotage import HDB_SAB_NONE
from hdbscan.host.hdbscan_host_oracle import (
    HDBH_GRAPH_DENSE,
    HDBH_GRAPH_SPARSE,
    hdbh_core_distances,
    hdbh_fit,
    hdbh_sparse_prim,
)
from hdbscan.impl.cluster.detail.single_linkage import (
    MR_GRAPH_DENSE,
    MR_GRAPH_SPARSE,
    build_mr_linkage,
)
from hdbscan.impl.cluster.detail.sparse_mr_mst import sparse_mr_mst
from hdbscan.impl.detail.extract import probabilities_from_labels
from hdbscan.impl.detail.reachability import compute_core_dists
from hdbscan.impl.detail.select import (
    CLUSTER_SELECTION_EOM,
    CLUSTER_SELECTION_LEAF,
)
from hdbscan.impl.runner import (
    GRAPH_BUILD_BRUTE_FORCE_KNN,
    HDBSCANParams,
    effective_min_samples,
    fit_hdbscan,
)
from hierarchy.impl.cluster.detail.connectivities import DISTANCE_L2_SQRT_EXPANDED


comptime IDENTICAL_BUILD = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

comptime FIX_BLOBS = 0
comptime FIX_DUPS = 1
comptime FIX_GRID = 2
comptime FIX_GRID_LEAF = 3


def _hash01(i: Int, f: Int, salt: Int) -> Float32:
    var h = UInt64(i) * 0x9E3779B97F4A7C15 + UInt64(f) * 0xBF58476D1CE4E5B9 + UInt64(salt) * 0x94D049BB133111EB
    h = (h ^ (h >> 31)) * 0xD6E8FEB86659FD93
    h = h ^ (h >> 32)
    return Float32(Int(h & UInt64(0xFFFF))) / Float32(65536.0)


def fixture(kind: Int, seed: Int, m: Int, d: Int) -> List[Float32]:
    """Deterministic rows. BLOBS: five centers, hashed offsets. DUPS: blobs
    with every row twice. GRID: small integers, many equal distances."""
    var x = List[Float32](capacity=m * d)
    for i in range(m):
        for f in range(d):
            if kind == FIX_BLOBS:
                var c = i % 5
                x.append(Float32(c * 3) * _hash01(c, f, seed + 101) * Float32(4.0)
                         + _hash01(i, f, seed) * Float32(2.0) - Float32(1.0))
            elif kind == FIX_DUPS:
                var r = i // 2
                var c = r % 4
                x.append(Float32(c * 5) + _hash01(r, f, seed) * Float32(3.0))
            else:
                x.append(Float32(Int(_hash01(i, f, seed) * Float32(5.0))) + Float32((i % 3) * 7))
    return x^


def _hex(v: Float32) -> String:
    var b = rebind[UInt32](v.to_bits())
    var digits = String("0123456789abcdef")
    var out = String("0x")
    var shift = 28
    while shift >= 0:
        var nib = Int((b >> UInt32(shift)) & UInt32(0xF))
        out += String(digits[byte=nib])
        shift -= 4
    return out


@fieldwise_init
struct Linkage(Movable):
    var lo: List[Int32]
    var hi: List[Int32]
    var w: List[Float32]
    var rounds: Int
    var children: List[Int32]
    var deltas: List[Float32]
    var sizes: List[Int32]


def _down_i32(ctx: DeviceContext, buf: DeviceBuffer[DType.int32], n: Int) raises -> List[Int32]:
    var h = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf.create_sub_buffer[DType.int32](0, n))
    ctx.synchronize()
    var out = List[Int32](capacity=n)
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    return out^


def _down_f32(ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf.create_sub_buffer[DType.float32](0, n))
    ctx.synchronize()
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    return out^


def _upload(ctx: DeviceContext, vals: List[Float32]) raises -> DeviceBuffer[DType.float32]:
    var dev = ctx.enqueue_create_buffer[DType.float32](len(vals))
    ctx.enqueue_copy(dst_buf=dev, src_ptr=vals.unsafe_ptr())
    ctx.synchronize()
    return dev^


def _linkage(
    ctx: DeviceContext, vals: List[Float32], m: Int, d: Int, k: Int,
    alpha: Float32, graph: Int,
) raises -> Linkage:
    var x_host = vals.copy()
    var x = _upload(ctx, vals)
    var trace = IdentityTrace.disabled()
    var ne = m - 1
    var core = ctx.enqueue_create_buffer[DType.float32](m)
    var rows = ctx.enqueue_create_buffer[DType.int32](ne)
    var cols = ctx.enqueue_create_buffer[DType.int32](ne)
    var wts = ctx.enqueue_create_buffer[DType.float32](ne)
    var ch = ctx.enqueue_create_buffer[DType.int32](ne * 2)
    var dl = ctx.enqueue_create_buffer[DType.float32](ne)
    var sz = ctx.enqueue_create_buffer[DType.int32](ne)
    ctx.synchronize()
    var rounds = build_mr_linkage(
        ctx, trace, x_host, x, m, d, k, alpha, DISTANCE_L2_SQRT_EXPANDED,
        core, rows, cols, wts, ch, dl, sz, graph=graph,
    )
    var out = Linkage(
        _down_i32(ctx, rows, ne), _down_i32(ctx, cols, ne), _down_f32(ctx, wts, ne),
        rounds, _down_i32(ctx, ch, ne * 2), _down_f32(ctx, dl, ne), _down_i32(ctx, sz, ne),
    )
    _ = x^
    return out^


def _say(ok: Bool, what: String) -> Int:
    """Prints a verdict; returns 1 when it counts as a failure."""
    if ok:
        print("  PASS " + what)
        return 0
    comptime if IDENTICAL_BUILD:
        print("  FAIL " + what)
        return 1
    else:
        print("  REPORT (FAST, not asserted) " + what)
        return 0


def _edges_diff(
    alo: List[Int32], ahi: List[Int32], aw: List[Float32],
    blo: List[Int32], bhi: List[Int32], bw: List[Float32],
) -> String:
    """Empty when equal slot for slot and bit for bit; else the first slot."""
    if len(alo) != len(blo):
        return "length " + String(len(alo)) + " vs " + String(len(blo))
    for e in range(len(alo)):
        if alo[e] != blo[e] or ahi[e] != bhi[e] or aw[e].to_bits() != bw[e].to_bits():
            return (
                "slot " + String(e) + ": (" + String(alo[e]) + "," + String(ahi[e]) + ","
                + _hex(aw[e]) + ") vs (" + String(blo[e]) + "," + String(bhi[e]) + ","
                + _hex(bw[e]) + ")"
            )
    return ""


def _i32_eq(a: List[Int32], b: List[Int32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _f32_eq(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i].to_bits() != b[i].to_bits():
            return False
    return True


def _equal_weight_pairs(w: List[Float32]) -> Int:
    var n = 0
    for e in range(1, len(w)):
        if w[e].to_bits() == w[e - 1].to_bits():
            n += 1
    return n


def check_one(
    ctx: DeviceContext, name: String, kind: Int, seed: Int, m: Int, d: Int,
    min_samples: Int, mcs: Int, alpha: Float32, method: Int,
) raises -> Int:
    print("fixture " + name + " m=" + String(m) + " d=" + String(d)
          + " min_samples=" + String(min_samples) + " alpha=" + String(alpha))
    var bad = 0
    var x = fixture(kind, seed, m, d)
    var k = effective_min_samples(min_samples, m)

    # 1. device linkage, dense vs sparse
    var dn = _linkage(ctx, x, m, d, k, alpha, MR_GRAPH_DENSE)
    var sp = _linkage(ctx, x, m, d, k, alpha, MR_GRAPH_SPARSE)
    var diff = _edges_diff(dn.lo, dn.hi, dn.w, sp.lo, sp.hi, sp.w)
    bad += _say(diff == "", "device MST edges and weight bits, sparse == dense " + diff)
    bad += _say(dn.rounds == sp.rounds, "device round count " + String(sp.rounds) + " == dense " + String(dn.rounds))
    bad += _say(
        _i32_eq(dn.children, sp.children) and _f32_eq(dn.deltas, sp.deltas) and _i32_eq(dn.sizes, sp.sizes),
        "device dendrogram children, deltas, sizes",
    )
    var ties = _equal_weight_pairs(dn.w)
    if kind != FIX_BLOBS:
        if ties < 2:
            print("  FAIL VACUOUS: " + String(ties) + " equal-weight tree edge pairs; the tie rule is untested here")
            bad += 1
        else:
            print("  ties: " + String(ties) + " adjacent equal-weight tree edge pairs")

    # 2. device sparse, many launches and slices
    var x_host = x.copy()
    var xd = _upload(ctx, x)
    var core = ctx.enqueue_create_buffer[DType.float32](m)
    var kd = ctx.enqueue_create_buffer[DType.float32](m * k)
    var ki = ctx.enqueue_create_buffer[DType.int32](m * k)
    var trace = IdentityTrace.disabled()
    compute_core_dists(ctx, trace, xd, core, m, d, DISTANCE_L2_SQRT_EXPANDED, k, kd, ki)
    var inv_alpha = identical_div(Float32(1.0), alpha)
    var small = sparse_mr_mst(ctx, x_host, xd, core, m, d, inv_alpha, HDB_SAB_NONE, launch_macs=4096)
    diff = _edges_diff(dn.lo, dn.hi, dn.w, small.lo, small.hi, small.w)
    bad += _say(diff == "" and small.rounds == dn.rounds,
                "device sparse at a 4096-MAC launch bound == dense " + diff)
    _ = xd^
    _ = kd^
    _ = ki^

    # 3. device fit, dense vs sparse
    var params = HDBSCANParams(
        min_samples, mcs, 0, Float32(0.0), False, alpha, method, GRAPH_BUILD_BRUTE_FORCE_KNN,
    )
    var xh1 = x.copy()
    var xd1 = _upload(ctx, x)
    var t1 = IdentityTrace.disabled()
    var fd = fit_hdbscan(ctx, t1, xh1, xd1, m, d, DISTANCE_L2_SQRT_EXPANDED, params.copy(), graph=MR_GRAPH_DENSE)
    var xh2 = x.copy()
    var xd2 = _upload(ctx, x)
    var t2 = IdentityTrace.disabled()
    var fs = fit_hdbscan(ctx, t2, xh2, xd2, m, d, DISTANCE_L2_SQRT_EXPANDED, params.copy(), graph=MR_GRAPH_SPARSE)
    var pd = probabilities_from_labels(fd.condensed, fd.labels, fd.inverse_label_map, m)
    var ps = probabilities_from_labels(fs.condensed, fs.labels, fs.inverse_label_map, m)
    var fit_ok = (
        _i32_eq(fd.labels, fs.labels) and _i32_eq(fd.raw_labels, fs.raw_labels)
        and _f32_eq(fd.core_dists, fs.core_dists)
        and _i32_eq(fd.condensed.parents, fs.condensed.parents)
        and _i32_eq(fd.condensed.children, fs.condensed.children)
        and _f32_eq(fd.condensed.lambdas, fs.condensed.lambdas)
        and _i32_eq(fd.condensed.sizes, fs.condensed.sizes)
        and _f32_eq(fd.stabilities, fs.stabilities)
        and fd.n_boruvka_rounds == fs.n_boruvka_rounds
        and fd.n_clusters == fs.n_clusters
    )
    bad += _say(fit_ok, "device fit labels, tree, stabilities, rounds (clusters "
                + String(fs.n_clusters) + ", noise " + String(fs.n_outliers) + ")")
    bad += _say(_f32_eq(pd, ps), "device probabilities bit for bit")
    _ = xd1^
    _ = xd2^

    # 4. CPU: sparse Prim edges vs the DEVICE dense edges; host fit both ways
    var hcore = hdbh_core_distances(x, m, d, k)
    var hp = hdbh_sparse_prim(x, m, d, hcore, alpha)
    diff = _edges_diff(dn.lo, dn.hi, dn.w, hp.src, hp.dst, hp.weights)
    bad += _say(diff == "" and hp.rounds == dn.rounds,
                "CPU sparse (Prim) edges == device dense edges, rounds " + String(hp.rounds) + " " + diff)
    var hd = hdbh_fit(x, m, d, min_samples, mcs, 0, alpha, False, method, Float32(0.0),
                      DISTANCE_L2_SQRT_EXPANDED, HDBH_GRAPH_DENSE)
    var hs = hdbh_fit(x, m, d, min_samples, mcs, 0, alpha, False, method, Float32(0.0),
                      DISTANCE_L2_SQRT_EXPANDED, HDBH_GRAPH_SPARSE)
    var hpd = probabilities_from_labels(hd.tree, hd.labels, hd.inverse_label_map, m)
    var hps = probabilities_from_labels(hs.tree, hs.labels, hs.inverse_label_map, m)
    bad += _say(
        _i32_eq(hd.labels, hs.labels) and _f32_eq(hd.core_dists, hs.core_dists)
        and hd.n_boruvka_rounds == hs.n_boruvka_rounds and hd.n_clusters == hs.n_clusters
        and _f32_eq(hpd, hps) and _i32_eq(hd.tree.children, hs.tree.children)
        and _f32_eq(hd.tree.lambdas, hs.tree.lambdas),
        "CPU fit dense == sparse: labels, core, rounds, tree, probabilities",
    )
    bad += _say(_i32_eq(hs.labels, fs.labels) and _f32_eq(hps, ps),
                "CPU sparse fit == device sparse fit (labels, probabilities)")
    return bad


def main() raises:
    comptime if IDENTICAL_BUILD:
        print("sparse_mr_check: IDENTICAL build (assertions)")
    else:
        print("sparse_mr_check: FAST build (reports only)")
    var ctx = DeviceContext()
    var bad = 0
    for seed in range(4):
        bad += check_one(ctx, "blobs seed " + String(seed), FIX_BLOBS, seed, 600 + 37 * seed, 5,
                         5, 10, Float32(1.0), CLUSTER_SELECTION_EOM)
    bad += check_one(ctx, "dups", FIX_DUPS, 7, 520, 4, 4, 8, Float32(1.0), CLUSTER_SELECTION_EOM)
    bad += check_one(ctx, "grid", FIX_GRID, 3, 700, 3, 5, 10, Float32(1.0), CLUSTER_SELECTION_EOM)
    bad += check_one(ctx, "grid alpha 1.3 leaf", FIX_GRID, 11, 450, 2, 3, 6, Float32(1.3), CLUSTER_SELECTION_LEAF)
    if bad != 0:
        raise Error("sparse_mr_check: " + String(bad) + " comparison(s) FAILED")
    print("sparse_mr_check: ALL PASS")

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""UMAP spectral initialization with connected components (lane
sc-ann-1-spectral, 2026-11-27): native checks on hand-sized graphs.

1. TWO COMPONENTS (the even and the odd vertices of a 32-vertex graph, so
   the grouping is exercised): the report says two components, both solved,
   none fallen back; every cell finite; the components' means in the first
   coordinate are apart (centroids at +-e_0); the host column
   (`host_spectral_initialize_report`) gives the same bits.
2. ONE COMPONENT (the same graph with one bridging edge pair): the report
   says one component; the embedding has the bits of the previous route
   (`transform_device_coo_device` at cuVS's bound, then the post-pass), so a
   connected graph that converges inside the restart bound is unchanged;
   the host column agrees.
3. FALLBACK, under `-D MOJOLEARN_UMAP_SPECTRAL_SABOTAGE_SOLVE`: every solve
   counts as failed, `n_fallback` equals the solvable components, the
   message is set, every cell finite, and the host column agrees. Without
   the define, `n_fallback == 0` is asserted instead.

Run (CPU column and any GPU):
  mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . umap/checks/spectral_components_check.mojo
  mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_UMAP_SPECTRAL_SABOTAGE_SOLVE=1 -I . umap/checks/spectral_components_check.mojo
Each prints SPECTRAL_COMPONENTS_PASS <case> and exits non-zero on a failure.
"""

from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.math import isfinite

from core.identity_trace import IdentityTrace
from spectral.checks.device_io import upload_f32, upload_i32
from spectral.impl.sparse.coo import CooGraph
from spectral.impl.preprocessing.detail.spectral_embedding import transform_device_coo_device
from umap.host.spectral_components_host import host_spectral_initialize_report
from umap.spectral_components_math import SC_SABOTAGE_SOLVE
from umap.spectral_init import (
    _sc_params,
    _spectral_post_pass_device,
    spectral_initialize_device_coo_report,
)


def _weight(a: Int, b: Int) -> Float32:
    """A symmetric, uneven positive weight (no two rows alike)."""
    var lo = a if a < b else b
    var hi = b if a < b else a
    return Float32(1.0) / Float32(1 + ((lo * 7 + hi * 3) % 5))


def _graph(n_half: Int, bridge: Bool) -> CooGraph:
    """Component A = even vertices, B = odd vertices; inside each, local
    vertex `l` links to `l + 1`, `l + 2`, `l + 3` (both directions). With
    `bridge`, vertices 0 and 1 are linked too. Row-major, columns ascending."""
    var n = 2 * n_half
    var adj = List[Float32](length=n * n, fill=Float32(0.0))
    for parity in range(2):
        for l in range(n_half):
            for step in range(1, 4):
                var m = l + step
                if m >= n_half:
                    continue
                var a = 2 * l + parity
                var b = 2 * m + parity
                var w = _weight(a, b)
                adj[a * n + b] = w
                adj[b * n + a] = w
    if bridge:
        adj[0 * n + 1] = Float32(0.25)
        adj[1 * n + 0] = Float32(0.25)
    var rows = List[Int32]()
    var cols = List[Int32]()
    var vals = List[Float32]()
    for r in range(n):
        for c in range(n):
            if adj[r * n + c] > Float32(0.0):
                rows.append(Int32(r))
                cols.append(Int32(c))
                vals.append(adj[r * n + c])
    return CooGraph(n, rows^, cols^, vals^)


def _same_bits(a: List[Float32], b: List[Float32], tag: String) raises:
    if len(a) != len(b):
        raise Error("spectral components: length differs " + tag)
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            raise Error("spectral components: bits differ " + tag + " at " + String(i))


def _all_finite(a: List[Float32], tag: String) raises:
    for i in range(len(a)):
        if not isfinite(a[i]):
            raise Error("spectral components: non-finite cell " + tag + " at " + String(i))


def two_components(ctx: DeviceContext) raises:
    var dim = 2
    var seed = UInt64(23)
    var g = _graph(16, False)
    var rep = spectral_initialize_device_coo_report(
        ctx, g.n, g.nnz(), upload_i32(ctx, g.rows), upload_i32(ctx, g.cols), upload_f32(ctx, g.vals),
        dim, 15, seed,
    )
    if rep.n_graph_components != 2:
        raise Error("two components: found " + String(rep.n_graph_components))
    comptime if SC_SABOTAGE_SOLVE:
        if rep.n_fallback != 2 or rep.n_solved != 0 or rep.message.byte_length() == 0:
            raise Error("two components: the sabotaged solves did not fall back")
    else:
        if rep.n_solved != 2 or rep.n_fallback != 0:
            raise Error("two components: solved " + String(rep.n_solved) + " fallback " + String(rep.n_fallback))
    _all_finite(rep.embedding, "two")
    var mean_a = Float32(0.0)
    var mean_b = Float32(0.0)
    for v in range(g.n):
        if v % 2 == 0:
            mean_a += rep.embedding[v * dim]
        else:
            mean_b += rep.embedding[v * dim]
    mean_a /= Float32(16)
    mean_b /= Float32(16)
    var gap = mean_a - mean_b
    if gap < Float32(0.0):
        gap = -gap
    if not (gap > Float32(2.0)):
        raise Error("two components: not separated, mean gap " + String(gap))
    var host = host_spectral_initialize_report(g, dim, seed)
    _same_bits(rep.embedding, host.embedding, "two.host")
    print("SPECTRAL_COMPONENTS_PASS two_components gap", gap)


def one_component(ctx: DeviceContext) raises:
    var dim = 2
    var seed = UInt64(23)
    var g = _graph(16, True)
    var rep = spectral_initialize_device_coo_report(
        ctx, g.n, g.nnz(), upload_i32(ctx, g.rows), upload_i32(ctx, g.cols), upload_f32(ctx, g.vals),
        dim, 15, seed,
    )
    if rep.n_graph_components != 1:
        raise Error("one component: found " + String(rep.n_graph_components))
    _all_finite(rep.embedding, "one")
    comptime if SC_SABOTAGE_SOLVE:
        if rep.n_fallback != 1 or rep.message.byte_length() == 0:
            raise Error("one component: the sabotaged solve did not fall back")
    else:
        if rep.n_fallback != 0:
            raise Error("one component: fell back")
        # the previous route: the whole-graph solve at cuVS's bound, the post-pass
        var cp = _sc_params(dim, 15, seed)
        var trace = IdentityTrace.disabled()
        var d_emb = ctx.enqueue_create_buffer[DType.float32](1)
        var n_out = transform_device_coo_device(
            ctx, cp, g.n, g.nnz(), upload_i32(ctx, g.rows), upload_i32(ctx, g.cols), upload_f32(ctx, g.vals),
            d_emb, trace,
        )
        if n_out != dim:
            raise Error("one component: previous route shape")
        var before = _spectral_post_pass_device(ctx, d_emb, g.n, dim)
        _same_bits(rep.embedding, before, "one.previous")
    var host = host_spectral_initialize_report(g, dim, seed)
    _same_bits(rep.embedding, host.embedding, "one.host")
    print("SPECTRAL_COMPONENTS_PASS one_component")


def main() raises:
    var ctx = DeviceContext()
    two_components(ctx)
    one_component(ctx)
    print("SPECTRAL_COMPONENTS_PASS all")

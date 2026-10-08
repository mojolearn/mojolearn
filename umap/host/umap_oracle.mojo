# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""The IDENTICAL UMAP fit and transform on the HOST, with no device (lane
lane/cpu-training-umap-b, 2026-09-14).

WHAT THIS IS. `bindings/_mojolearn_metrics.mojo::umap_fit_transform_binding`
runs `umap/estimator.mojo::fit_transform` ->
`umap/sparse_estimator.mojo::sparse_fit_transform`, and
`umap_transform_binding` runs `umap/transform.mojo::transform`. Both reach a
DeviceContext: the exact k-NN (`neighbors/estimator.mojo::knn_search`), the
spectral initialization (`spectral/impl/spectral_embedding.mojo::
transform_connectivity`, the device Laplacian and Lanczos) and, under
IDENTICAL, the layout optimizer (kernel-matrix row
`umap_device_optimizer_for`, `umap/optimizer_identical_device.mojo`). This
file is a SECOND spelling of that path in the device's order, so the metrics
CPU host binding can serve `UMAP` on a CPU-only install.

THE HOST LOOP IS NOT THE CONTRACT. `umap/sparse_optimizer.mojo::
optimize_sparse_layout_identical_reference` is a Gauss-Seidel sweep; the device fold is
Jacobi over an epoch snapshot and produces DIFFERENT bits
(`checks/kernel_matrix.mojo::umap_device_optimizer_for`). The IDENTICAL
contract on every GPU column is the device fold, so `host_umap_vertex` below
restates `umap_identical_epoch_kernel` vertex by vertex: the snapshot is a
separate buffer, vertex `v` reads only the snapshot and writes only
`destination[v]`, so a serial loop over `v` in any order is the same
function as one thread per vertex.

THE STAGES, EACH WITH THE DEVICE CODE IT MIRRORS

    shape, finiteness   sparse_estimator.mojo:31-40 (params.validate, the
                        shape and finiteness refusals)
    k-NN self-join      knn_search at its defaults (L2, sqrt, the IDENTICAL
                        tiled arm) -> core/knn_host_predict.mojo::
                        host_knn_search at KNN_HOST_METRIC_FROM_IS_SQRT,
                        return_sqrt True (the knn lanes' restatement)
    self first          umap/graph.mojo::canonicalize_self_neighbors
                        (host code, imported; no device)
    fuzzy graph         umap/host/sparse_graph_host.mojo::sparse_fuzzy_simplicial_graph
                        (host code, imported; no device)
    spectral init       sparse_estimator.mojo:67-88 (the positive row-major
                        COO) -> umap/spectral_init.mojo:72-125
                        (MLSpectralEmbeddingParams(n_components + 1,
                        norm_laplacian, drop_first, seed), cuVS's default
                        tolerance 1e-5, then the sign and scale post-pass)
                        -> spectral/host/spectral_oracle.mojo::
                        oracle_embedding (the spectral lanes' restatement of
                        transform_graph)
    curve               umap/curve.mojo::fit_umap_curve (host code, imported)
    optimizer checks    sparse_optimizer.mojo:175-214 (the scalar refusals,
                        validate_sparse_weights, finiteness), then
                        optimizer_identical_device.mojo:387-424 (positive
                        non-self compaction, the symmetry refusal, the
                        flushed scaled weight)
    epoch loop          optimizer_identical_device.mojo:269-363 (neg2ab and
                        rep2b once, the flushed initial upload, alpha per
                        epoch in Float64, the two buffers swapped by parity)
    one vertex          optimizer_identical_device.mojo:116-237
                        (umap_identical_epoch_kernel; the snapshot fold by
                        default, DEVIATION 2668's live row behind the same
                        kernel-matrix row)
    negatives           core/philox.mojo:19-51 (philox4x32_10), counter
                        (e lo, e hi, epoch, slot // 4), key (seed lo, seed hi)
    transform           umap/transform.mojo (the host k-NN above for
                        knn_search_resident; the per-row statements of the
                        device prepare and epoch kernels are
                        umap/transform_rows.mojo, which this column calls
                        itself: tr_row_memberships, tr_row_initialize,
                        tr_row_refine_prep, tr_row_refine_epoch; the epoch
                        count and the curve as in transform())

THE SABOTAGE (-D MOJOLEARN_HOST_SABOTAGE=1, the routed set's negative
control). Every negative draw keys its Philox counter with `epoch + 1`
instead of `epoch` (`UMAP_ORACLE_HOST_SABOTAGE`), so every repulsive move
of every fit samples other vertices and every embedding cell moves; the
transform refinement's SplitMix64 counter takes the same `epoch + 1`. The
k-NN restatement under the same define carries its own arms
(`core/knn_host_predict.mojo`).
"""

from std.math import isfinite
from std.memory import bitcast
from std.sys.compile import is_defined

from checks.kernel_matrix import TARGET_COLUMN, umap_device_optimizer_live_row_for
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_div,
    identical_exp64,
    identical_log2_64,
    identical_mul,
    identical_mul_add,
    identical_pow,
    identical_pow64,
)
from core.knn_host_predict import KNN_HOST_METRIC_FROM_IS_SQRT, host_knn_search
from spectral.host.spectral_oracle import oracle_embedding
from spectral.impl.sparse.coo import CooGraph
from umap.curve import fit_umap_curve
from umap.graph import canonicalize_self_neighbors
from umap.params import UMAPParams
from umap.sparse_graph_cells import SparseFuzzySimplicialGraph
from umap.host.sparse_graph_host import (
    host_categorical_intersection,
    host_general_intersection,
    sparse_fuzzy_simplicial_graph,
)
from umap.host.spectral_post_pass_host import host_spectral_post_pass
from umap.transform_rows import (
    TR_F32P,
    TR_FLAGS,
    TR_OK,
    TR_U32P,
    TR_U64P,
    tr_alpha,
    tr_neg2ab,
    tr_raise,
    tr_rep2b,
    tr_row_initialize,
    tr_row_memberships,
    tr_row_refine_epoch,
    tr_row_refine_prep,
    tr_target,
)


#: The gate's negative control (module docstring). Read back by
#: `metrics_host_sabotage`.
comptime UMAP_ORACLE_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

#: `UMAP_IDENTICAL_GRAD_CLIP`, `optimizer_identical_device.mojo:71`, and
#: `UMAP_GRAD_CLIP`, `umap/optimizer.mojo:16` (both 4).
comptime UMAP_HOST_GRAD_CLIP = Float32(4.0)

#: `UMAP_LIVE_ROW` and `UMAP_LIVE_BOTH_ARM`,
#: `optimizer_identical_device.mojo:96-99`, the same row and define.
comptime UMAP_HOST_LIVE_ROW = umap_device_optimizer_live_row_for[
    TARGET_COLUMN, GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
]()
comptime UMAP_HOST_LIVE_BOTH_ARM = is_defined["MOJOLEARN_UMAP_LIVE_BOTH_ARM"]()

#: `PHILOX_W32_*`, `PHILOX_M4X32_*`, `core/philox.mojo:12-15`.
comptime UMAP_PHILOX_W32_0: UInt32 = 0x9E3779B9
comptime UMAP_PHILOX_W32_1: UInt32 = 0xBB67AE85
comptime UMAP_PHILOX_M4X32_0: UInt32 = 0xD2511F53
comptime UMAP_PHILOX_M4X32_1: UInt32 = 0xCD9E8D57


# ---------------------------------------------------------------------------
# Small helpers, restated (their files import device modules)
# ---------------------------------------------------------------------------


def _finite(v: Float32) -> Bool:
    """`_finite`, `optimizer_identical_device.mojo:111-113`."""
    var bits = bitcast[DType.uint32](v)
    return ((bits >> UInt32(23)) & UInt32(0xFF)) != UInt32(0xFF)


def _clip(value: Float32) -> Float32:
    """`_clip`, `optimizer_identical_device.mojo:102-108` (and
    `umap/optimizer.mojo:31-36`, the same function)."""
    if value > UMAP_HOST_GRAD_CLIP:
        return UMAP_HOST_GRAD_CLIP
    if value < -UMAP_HOST_GRAD_CLIP:
        return -UMAP_HOST_GRAD_CLIP
    return value


def _splitmix64(value: UInt64) -> UInt64:
    """`_splitmix64`, `umap/optimizer.mojo:24-28`."""
    var z = value + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _mulhilo32(a: UInt32, b: UInt32) -> Tuple[UInt32, UInt32]:
    """`_mulhilo32`, `core/philox.mojo:19-25`, returning `(hi, lo)`."""
    var p = (a.cast[DType.uint64]() & 0xFFFFFFFF) * (b.cast[DType.uint64]() & 0xFFFFFFFF)
    return (UInt32((p >> 32) & 0xFFFFFFFF), UInt32(p & 0xFFFFFFFF))


def _philox4x32_round(c: SIMD[DType.uint32, 4], k: SIMD[DType.uint32, 2]) -> SIMD[DType.uint32, 4]:
    """`_philox4x32_round`, `core/philox.mojo:28-37`."""
    var r0 = _mulhilo32(UMAP_PHILOX_M4X32_0, c[0])
    var r1 = _mulhilo32(UMAP_PHILOX_M4X32_1, c[2])
    return SIMD[DType.uint32, 4](r1[0] ^ c[1] ^ k[0], r1[1], r0[0] ^ c[3] ^ k[1], r0[1])


def host_philox4x32_10(ctr: SIMD[DType.uint32, 4], key: SIMD[DType.uint32, 2]) -> SIMD[DType.uint32, 4]:
    """`philox4x32_10`, `core/philox.mojo:40-51`: ten rounds, nine key bumps."""
    var c = ctr
    var k = key
    for _ in range(9):
        c = _philox4x32_round(c, k)
        k[0] = k[0] + UMAP_PHILOX_W32_0
        k[1] = k[1] + UMAP_PHILOX_W32_1
    return _philox4x32_round(c, k)


# ---------------------------------------------------------------------------
# The fit's graph and initialization
# ---------------------------------------------------------------------------


def host_umap_fuzzy_graph(
    x_rowmajor: List[Float32], n_samples: Int, n_features: Int, params: UMAPParams
) raises -> SparseFuzzySimplicialGraph:
    """`sparse_fuzzy_graph_from_data`, `sparse_estimator.mojo:24-64`, with
    the device k-NN replaced by its host restatement."""
    params.validate(n_samples)
    if n_features < 1 or len(x_rowmajor) != n_samples * n_features:
        raise Error("UMAP input does not match its declared shape")
    for i in range(len(x_rowmajor)):
        if not isfinite(x_rowmajor[i]):
            raise Error("UMAP input coordinates must be finite")
    var distances = List[Float32](length=n_samples * params.n_neighbors, fill=Float32(0.0))
    var indices = List[UInt32](length=n_samples * params.n_neighbors, fill=UInt32(0))
    host_knn_search(
        x_rowmajor, n_samples, x_rowmajor, n_samples, n_features, params.n_neighbors,
        KNN_HOST_METRIC_FROM_IS_SQRT if params.metric == -1 else params.metric, True, distances, indices,
        params.metric_arg,
    )
    canonicalize_self_neighbors(indices, distances, n_samples, params.n_neighbors)
    return sparse_fuzzy_simplicial_graph(
        indices^, distances^, n_samples, params.n_neighbors, params.set_op_mix_ratio,
        params.local_connectivity,
    )


def host_validate_sparse_weights(graph: SparseFuzzySimplicialGraph) raises -> Float32:
    """`validate_sparse_weights`, `sparse_optimizer.mojo:21-44`, verbatim."""
    var n = graph.n_samples
    if n < 2 or len(graph.offsets) != n + 1 or len(graph.indices) != len(graph.values):
        raise Error("UMAP sparse graph shape mismatch")
    if graph.offsets[0] != 0 or graph.offsets[n] != len(graph.values):
        raise Error("UMAP sparse graph terminal offset mismatch")
    var max_weight = Float32(0.0)
    for row in range(n):
        var begin = graph.offsets[row]
        var end = graph.offsets[row + 1]
        if begin < 0 or end < begin or end > len(graph.values):
            raise Error("UMAP sparse graph offset out of range")
        var previous = -1
        for edge in range(begin, end):
            var col = Int(graph.indices[edge])
            var weight = graph.values[edge]
            if col <= previous or col >= n or col == row:
                raise Error("UMAP sparse graph needs unique sorted nonself columns")
            previous = col
            if not _finite(weight) or weight < Float32(0.0):
                raise Error("UMAP sparse graph weight is invalid")
            if weight > max_weight:
                max_weight = weight
    return max_weight


def host_umap_spectral_initialize(
    graph: SparseFuzzySimplicialGraph, n_components: Int, seed: UInt64
) raises -> List[Float32]:
    """`sparse_spectral_initialize`, `sparse_estimator.mojo:67-88`, then
    `spectral_initialize_coo`, `spectral_init.mojo:72-125`, over the host
    `transform_graph` (`oracle_embedding` at cuVS's default tolerance)."""
    _ = host_validate_sparse_weights(graph)
    var rows = List[Int32]()
    var cols = List[Int32]()
    var vals = List[Float32]()
    for row in range(graph.n_samples):
        for edge in range(graph.offsets[row], graph.offsets[row + 1]):
            var value = graph.values[edge]
            if value > Float32(0.0):
                rows.append(Int32(row))
                cols.append(Int32(graph.indices[edge]))
                vals.append(value)
    if len(vals) == 0:
        raise Error("UMAP spectral graph has no edges")
    var n_samples = graph.n_samples
    var coo = CooGraph(n_samples, rows^, cols^, vals^)
    if n_components < 1 or n_components > 32:
        raise Error("UMAP spectral initialization supports 1 to 32 output dimensions")
    if n_samples < 2 * n_components + 4:
        raise Error("UMAP spectral initialization has too few samples")
    if coo.n != n_samples:
        raise Error("UMAP spectral COO shape disagrees with n_samples")
    # transform_graph's value refusals, spectral_embedding.mojo:328-339.
    for i in range(coo.nnz()):
        var v = coo.vals[i]
        if not isfinite(v):
            raise Error(
                "spectral: connectivity_graph has a non-finite value at entry "
                + String(i) + " -- refused by name"
            )
        if v < Float32(0.0):
            raise Error(
                "spectral: connectivity_graph has a negative value at entry "
                + String(i) + " -- refused by name (sqrt of a negative degree is NaN in theirs)"
            )
    var res = oracle_embedding[DType.float32](
        coo, n_components + 1, True, True, Float32(1e-5), seed
    )
    var n_out = res.n_out
    var embedding = List[Float32]()
    for i in range(len(res.embedding)):
        embedding.append(res.embedding[i])
    if n_out != n_components or len(embedding) != n_samples * n_components:
        raise Error("UMAP spectral solver returned the wrong shape")
    # The device post-pass's host column (identical_div / identical_mul + ftz,
    # lane cpu4-umap 2026-10-04): umap/host/spectral_post_pass_host.mojo.
    host_spectral_post_pass(embedding, n_samples, n_components)
    return embedding^


# ---------------------------------------------------------------------------
# The device optimizer on the host
# ---------------------------------------------------------------------------


def _csr_weight_at(
    offsets: List[Int], indices: List[UInt32], values: List[Float32], row: Int, col: Int
) -> Float32:
    """`_csr_weight_at`, `optimizer_identical_device.mojo:370-384`."""
    var lo = offsets[row]
    var hi = offsets[row + 1]
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if Int(indices[mid]) < col:
            lo = mid + 1
        else:
            hi = mid
    if lo < offsets[row + 1] and Int(indices[lo]) == col:
        return values[lo]
    return Float32(0.0)


def host_umap_vertex[C: Int](
    source: List[Float32],
    row_offsets: List[UInt32],
    tails: List[UInt32],
    scaled: List[Float32],
    mut destination: List[Float32],
    v: Int,
    n: Int,
    epoch: Int,
    alpha: Float32,
    rate: Int,
    neg2ab: Float32,
    rep2b: Float32,
    a: Float32,
    b: Float32,
    seed: UInt64,
):
    """`umap_identical_epoch_kernel[C]` at thread `v`,
    `optimizer_identical_device.mojo:116-237`, in the same statement order."""
    var epoch_f = Float32(epoch)
    var next_f = Float32(epoch + 1)
    var draw_epoch = epoch
    comptime if UMAP_ORACLE_HOST_SABOTAGE:
        # THE SABOTAGE ARM: every negative draw keyed one epoch late. Wrong
        # on purpose; see UMAP_ORACLE_HOST_SABOTAGE.
        draw_epoch = epoch + 1
    var epoch_u = UInt32(draw_epoch)
    var n_u = UInt32(n)
    var key = SIMD[DType.uint32, 2](UInt32(seed & 0xFFFFFFFF), UInt32((seed >> 32) & 0xFFFFFFFF))
    var x = SIMD[DType.float32, 4](0.0)
    var acc = SIMD[DType.float32, 4](0.0)
    comptime for c in range(C):
        x[c] = source[v * C + c]
    var begin = Int(row_offsets[v])
    var end = Int(row_offsets[v + 1])
    for e in range(begin, end):
        var s = scaled[e]
        if Int(next_f * s) <= Int(epoch_f * s):
            continue
        var u = Int(tails[e])
        var delta = SIMD[DType.float32, 4](0.0)
        var d2 = Float32(0.0)
        comptime for c in range(C):
            delta[c] = ftz(x[c] - source[u * C + c])
            d2 = ftz(identical_mul_add(delta[c], delta[c], d2))
        if d2 > Float32(0.0):
            var dp = identical_pow(d2, b)
            var coeff = identical_div(
                identical_mul(neg2ab, identical_div(dp, d2)),
                ftz(identical_mul_add(a, dp, Float32(1.0))),
            )
            comptime for c in range(C):
                var g = ftz(identical_mul(alpha, _clip(ftz(identical_mul(coeff, delta[c])))))
                comptime if UMAP_HOST_LIVE_BOTH_ARM:
                    x[c] = ftz(x[c] + g)
                    x[c] = ftz(x[c] + g)
                elif UMAP_HOST_LIVE_ROW:
                    x[c] = ftz(x[c] + g)
                    acc[c] = ftz(acc[c] + g)
                else:
                    acc[c] = ftz(acc[c] + g)
                    acc[c] = ftz(acc[c] + g)
        var draw = SIMD[DType.uint32, 4](0)
        for j in range(rate):
            var lane = j & 3
            if lane == 0:
                draw = host_philox4x32_10(
                    SIMD[DType.uint32, 4](
                        UInt32(e & 0xFFFFFFFF),
                        UInt32((e >> 32) & 0xFFFFFFFF),
                        epoch_u,
                        UInt32(j >> 2),
                    ),
                    key,
                )
            var other = Int(draw[lane] % n_u)
            if other == v:
                continue
            var nd = SIMD[DType.float32, 4](0.0)
            var n2 = Float32(0.0)
            comptime for c in range(C):
                nd[c] = ftz(x[c] - source[other * C + c])
                n2 = ftz(identical_mul_add(nd[c], nd[c], n2))
            if n2 > Float32(0.0):
                var np_ = identical_pow(n2, b)
                var coeff = identical_div(
                    rep2b,
                    identical_mul(
                        ftz(Float32(0.001) + n2),
                        ftz(identical_mul_add(a, np_, Float32(1.0))),
                    ),
                )
                comptime for c in range(C):
                    comptime if UMAP_HOST_LIVE_ROW or UMAP_HOST_LIVE_BOTH_ARM:
                        x[c] = ftz(x[c] + ftz(identical_mul(alpha, _clip(ftz(identical_mul(coeff, nd[c]))))))
                    else:
                        acc[c] = ftz(acc[c] + ftz(identical_mul(alpha, _clip(ftz(identical_mul(coeff, nd[c]))))))
    comptime for c in range(C):
        destination[v * C + c] = ftz(x[c] + acc[c])


def host_umap_vertex_rt(
    source: List[Float32],
    row_offsets: List[UInt32],
    tails: List[UInt32],
    scaled: List[Float32],
    mut destination: List[Float32],
    v: Int,
    n: Int,
    epoch: Int,
    alpha: Float32,
    rate: Int,
    neg2ab: Float32,
    rep2b: Float32,
    a: Float32,
    b: Float32,
    seed: UInt64,
    C: Int,
):
    """`umap_identical_epoch_kernel_rt` at thread `v` (DEVIATION 5322): the
    comptime vertex with the dimension read at run time, the rows in local
    arrays of UMAP_RT_WIDTH."""
    var epoch_f = Float32(epoch)
    var next_f = Float32(epoch + 1)
    var draw_epoch = epoch
    comptime if UMAP_ORACLE_HOST_SABOTAGE:
        # THE SABOTAGE ARM: every negative draw keyed one epoch late. Wrong
        # on purpose; see UMAP_ORACLE_HOST_SABOTAGE.
        draw_epoch = epoch + 1
    var epoch_u = UInt32(draw_epoch)
    var n_u = UInt32(n)
    var key = SIMD[DType.uint32, 2](UInt32(seed & 0xFFFFFFFF), UInt32((seed >> 32) & 0xFFFFFFFF))
    var x = InlineArray[Float32, 32](fill=Float32(0.0))
    var acc = InlineArray[Float32, 32](fill=Float32(0.0))
    for c in range(C):
        x[c] = source[v * C + c]
    var begin = Int(row_offsets[v])
    var end = Int(row_offsets[v + 1])
    for e in range(begin, end):
        var s = scaled[e]
        if Int(next_f * s) <= Int(epoch_f * s):
            continue
        var u = Int(tails[e])
        var delta = InlineArray[Float32, 32](fill=Float32(0.0))
        var d2 = Float32(0.0)
        for c in range(C):
            delta[c] = ftz(x[c] - source[u * C + c])
            d2 = ftz(identical_mul_add(delta[c], delta[c], d2))
        if d2 > Float32(0.0):
            var dp = identical_pow(d2, b)
            var coeff = identical_div(
                identical_mul(neg2ab, identical_div(dp, d2)),
                ftz(identical_mul_add(a, dp, Float32(1.0))),
            )
            for c in range(C):
                var g = ftz(identical_mul(alpha, _clip(ftz(identical_mul(coeff, delta[c])))))
                comptime if UMAP_HOST_LIVE_BOTH_ARM:
                    x[c] = ftz(x[c] + g)
                    x[c] = ftz(x[c] + g)
                elif UMAP_HOST_LIVE_ROW:
                    x[c] = ftz(x[c] + g)
                    acc[c] = ftz(acc[c] + g)
                else:
                    acc[c] = ftz(acc[c] + g)
                    acc[c] = ftz(acc[c] + g)
        var draw = SIMD[DType.uint32, 4](0)
        for j in range(rate):
            var lane = j & 3
            if lane == 0:
                draw = host_philox4x32_10(
                    SIMD[DType.uint32, 4](
                        UInt32(e & 0xFFFFFFFF),
                        UInt32((e >> 32) & 0xFFFFFFFF),
                        epoch_u,
                        UInt32(j >> 2),
                    ),
                    key,
                )
            var other = Int(draw[lane] % n_u)
            if other == v:
                continue
            var nd = InlineArray[Float32, 32](fill=Float32(0.0))
            var n2 = Float32(0.0)
            for c in range(C):
                nd[c] = ftz(x[c] - source[other * C + c])
                n2 = ftz(identical_mul_add(nd[c], nd[c], n2))
            if n2 > Float32(0.0):
                var np_ = identical_pow(n2, b)
                var coeff = identical_div(
                    rep2b,
                    identical_mul(
                        ftz(Float32(0.001) + n2),
                        ftz(identical_mul_add(a, np_, Float32(1.0))),
                    ),
                )
                for c in range(C):
                    comptime if UMAP_HOST_LIVE_ROW or UMAP_HOST_LIVE_BOTH_ARM:
                        x[c] = ftz(x[c] + ftz(identical_mul(alpha, _clip(ftz(identical_mul(coeff, nd[c]))))))
                    else:
                        acc[c] = ftz(acc[c] + ftz(identical_mul(alpha, _clip(ftz(identical_mul(coeff, nd[c]))))))
    for c in range(C):
        destination[v * C + c] = ftz(x[c] + acc[c])


def host_optimize_csr_layout(
    initial: List[Float32],
    row_offsets: List[UInt32],
    tails: List[UInt32],
    scaled: List[Float32],
    n_samples: Int,
    n_components: Int,
    n_epochs: Int,
    learning_rate: Float32,
    negative_rate: Int,
    repulsion: Float32,
    a: Float32,
    b: Float32,
    seed: UInt64,
) raises -> List[Float32]:
    """`optimize_csr_layout_identical_device`,
    `optimizer_identical_device.mojo:269-363`: the refusals, the flushed
    upload, one host loop over every vertex per epoch into the other buffer."""
    if n_samples < 2 or (n_components < 1 or n_components > 32):
        raise Error("UMAP device optimizer supports 1 to 32 dimensions")
    if len(initial) != n_samples * n_components or len(row_offsets) != n_samples + 1:
        raise Error("UMAP device optimizer input shape mismatch")
    if len(tails) != len(scaled) or len(tails) == 0:
        raise Error("UMAP device optimizer graph has no non-self edges")
    if n_samples > 2147483647 or n_epochs > 2147483647 or negative_rate > 2147483647:
        raise Error("UMAP device optimizer scalar exceeds kernel Int32 range")
    if len(tails) > 4294967295:
        raise Error("UMAP device optimizer edge count exceeds CSR UInt32 range")
    if n_epochs < 1 or not (learning_rate > Float32(0.0)) or negative_rate < 0:
        raise Error("UMAP device optimizer parameters are invalid")
    var neg2ab = -Float32(2.0) * a * b
    var rep2b = Float32(2.0) * repulsion * b
    var first = List[Float32]()
    for i in range(len(initial)):
        first.append(ftz(initial[i]))
    var second = List[Float32](length=len(initial), fill=Float32(0.0))
    for epoch in range(n_epochs):
        var alpha = learning_rate * Float32(Float64(n_epochs - epoch) / Float64(n_epochs))
        for v in range(n_samples):
            if epoch % 2 == 0:
                if n_components == 2:
                    host_umap_vertex[2](
                        first, row_offsets, tails, scaled, second, v, n_samples,
                        epoch, alpha, negative_rate, neg2ab, rep2b, a, b, seed,
                    )
                elif n_components == 3:
                    host_umap_vertex[3](
                        first, row_offsets, tails, scaled, second, v, n_samples,
                        epoch, alpha, negative_rate, neg2ab, rep2b, a, b, seed,
                    )
                else:
                    host_umap_vertex_rt(
                        first, row_offsets, tails, scaled, second, v, n_samples,
                        epoch, alpha, negative_rate, neg2ab, rep2b, a, b, seed, n_components,
                    )
            else:
                if n_components == 2:
                    host_umap_vertex[2](
                        second, row_offsets, tails, scaled, first, v, n_samples,
                        epoch, alpha, negative_rate, neg2ab, rep2b, a, b, seed,
                    )
                elif n_components == 3:
                    host_umap_vertex[3](
                        second, row_offsets, tails, scaled, first, v, n_samples,
                        epoch, alpha, negative_rate, neg2ab, rep2b, a, b, seed,
                    )
                else:
                    host_umap_vertex_rt(
                        second, row_offsets, tails, scaled, first, v, n_samples,
                        epoch, alpha, negative_rate, neg2ab, rep2b, a, b, seed, n_components,
                    )
    if n_epochs % 2 == 0:
        return first^
    return second^


def host_optimize_sparse_layout(
    initial_embedding: List[Float32],
    graph: SparseFuzzySimplicialGraph,
    n_samples: Int,
    n_components: Int,
    n_epochs: Int,
    initial_learning_rate: Float32,
    negative_sample_rate: Int,
    repulsion_strength: Float32,
    a: Float32,
    b: Float32,
    seed: UInt64,
) raises -> List[Float32]:
    """`optimize_sparse_layout_identical_on_device`,
    `sparse_optimizer.mojo:175-214` (the refusals in its order), then
    `optimize_sparse_layout_identical_device`,
    `optimizer_identical_device.mojo:387-424` (the compaction)."""
    if n_samples < 2 or (n_components < 1 or n_components > 32):
        raise Error("UMAP optimizer supports at least two samples in 1 to 32 dimensions")
    if len(initial_embedding) != n_samples * n_components or graph.n_samples != n_samples:
        raise Error("UMAP optimizer input shape mismatch")
    if not isfinite(initial_learning_rate) or not isfinite(repulsion_strength) or (
        not isfinite(a) or not isfinite(b)
    ):
        raise Error("UMAP optimizer scalar parameters must be finite")
    if n_epochs < 1 or not (initial_learning_rate > Float32(0.0)):
        raise Error("UMAP optimizer needs positive epochs and learning rate")
    if negative_sample_rate < 0 or repulsion_strength < Float32(0.0):
        raise Error("UMAP optimizer negative sampling parameters are invalid")
    if not (a > Float32(0.0)) or not (b > Float32(0.0)):
        raise Error("UMAP optimizer curve parameters must be positive")
    var max_weight = host_validate_sparse_weights(graph)
    if not (max_weight > Float32(0.0)):
        raise Error("UMAP optimizer graph has no positive edges")
    for i in range(len(initial_embedding)):
        if not _finite(initial_embedding[i]):
            raise Error("UMAP optimizer initialization is not finite")
    var row_offsets = List[UInt32]()
    var tails = List[UInt32]()
    var scaled = List[Float32]()
    row_offsets.append(UInt32(0))
    for head in range(n_samples):
        for edge in range(graph.offsets[head], graph.offsets[head + 1]):
            var tail = Int(graph.indices[edge])
            var weight = graph.values[edge]
            if head == tail or not (weight > Float32(0.0)):
                continue
            if weight != _csr_weight_at(graph.offsets, graph.indices, graph.values, tail, head):
                raise Error("UMAP device optimizer requires symmetric weights")
            tails.append(UInt32(tail))
            scaled.append(ftz(Float32(Float64(weight) / Float64(max_weight))))
        row_offsets.append(UInt32(len(tails)))
    return host_optimize_csr_layout(
        initial_embedding, row_offsets, tails, scaled, n_samples, n_components,
        n_epochs, initial_learning_rate, negative_sample_rate, repulsion_strength,
        a, b, seed,
    )


def host_supervise_graph(
    graph: SparseFuzzySimplicialGraph, target: List[Float32], n_samples: Int,
    target_kind: Int, target_dims: Int, target_n_neighbors: Int, target_weight: Float32, seed: UInt64,
) raises -> SparseFuzzySimplicialGraph:
    """`supervise_graph`, `sparse_estimator.mojo`, with the target's k-NN on
    the host (the set operations are the host column's `host_*` in
    `umap/host/sparse_graph_host.mojo`, over the per-cell statements the
    device kernels share)."""
    if target_kind == 1:
        var far = Float64(1.0e12)
        if target_weight < Float32(1.0):
            far = Float64(2.5) * (Float64(1.0) / (Float64(1.0) - Float64(target_weight)))
        return host_categorical_intersection(graph, target, far)
    var tp = UMAPParams(n_neighbors=target_n_neighbors, n_components=2, random_seed=seed)
    var tgraph = host_umap_fuzzy_graph(target, n_samples, target_dims, tp)
    return host_general_intersection(graph, tgraph, target_weight)


def host_umap_fit_transform(
    x_rowmajor: List[Float32], n_samples: Int, n_features: Int, params: UMAPParams,
    initial_given: List[Float32] = List[Float32](),
    target: List[Float32] = List[Float32](),
    target_kind: Int = 0,
    target_dims: Int = 1,
    target_n_neighbors: Int = 0,
    target_weight: Float32 = Float32(0.5),
) raises -> List[Float32]:
    """`sparse_fit_transform`, `sparse_estimator.mojo`, on the host."""
    params.validate(n_samples)
    if n_features < 1 or len(x_rowmajor) != n_samples * n_features:
        raise Error("UMAP input does not match its declared shape")
    if params.n_components < 1 or params.n_components > 32:
        raise Error("UMAP fit_transform supports 1 to 32 output dimensions")
    var given = len(initial_given) > 0
    if given and len(initial_given) != n_samples * params.n_components:
        raise Error("UMAP initial embedding does not match n_samples x n_components")
    if not given and n_samples < 2 * params.n_components + 4:
        raise Error("UMAP fit_transform has too few samples for spectral init")
    var graph = host_umap_fuzzy_graph(x_rowmajor, n_samples, n_features, params)
    if target_kind != 0:
        var tk = target_n_neighbors if target_n_neighbors > 0 else params.n_neighbors
        graph = host_supervise_graph(
            graph, target, n_samples, target_kind, target_dims, tk, target_weight, params.random_seed,
        )
    var initial: List[Float32]
    if given:
        initial = initial_given.copy()
    else:
        initial = host_umap_spectral_initialize(graph.copy(), params.n_components, params.random_seed)
    var epochs = params.n_epochs
    if epochs == 0:
        epochs = 200
    var ab = params.curve()
    return host_optimize_sparse_layout(
        initial^, graph, n_samples, params.n_components, epochs,
        params.learning_rate, params.negative_sample_rate, params.repulsion_strength,
        ab[0], ab[1], params.random_seed,
    )


# ---------------------------------------------------------------------------
# The transform (umap/transform.mojo, restated)
# ---------------------------------------------------------------------------


def host_transform_memberships(distances: List[Float32], rows: Int, k: Int) raises -> List[Float32]:
    """The transform's memberships, `umap/transform_rows.mojo::tr_row_memberships`
    per row (the statements the device prepare kernel runs). The lowest
    refusal code over the rows raises, the device's flag order."""
    if rows < 1 or k < 2 or len(distances) != rows * k:
        raise Error("UMAP transform neighbor shape mismatch")
    var weights = List[Float32](length=rows * k, fill=Float32(0.0))
    var dp = rebind[TR_F32P](distances.unsafe_ptr())
    var wp = rebind[TR_F32P](weights.unsafe_ptr())
    var target = tr_target(k)
    var worst = TR_FLAGS
    for row in range(rows):
        var status = tr_row_memberships(dp, wp, row, k, target)
        if status != TR_OK and status < worst:
            worst = status
    if worst < TR_FLAGS:
        tr_raise(worst)
    return weights^


def host_initialize_transform(
    indices: List[UInt32], weights: List[Float32], training: List[Float32],
    rows: Int, n_train: Int, k: Int, components: Int,
) raises -> List[Float32]:
    """The starting coordinates, `transform_rows.mojo::tr_row_initialize` per
    row (the device prepare kernel's statements)."""
    if rows < 1 or n_train < 2 or k < 2 or k > n_train or components < 1 or components > 32:
        raise Error("UMAP transform initialization dimensions are unsupported")
    if len(indices) != rows * k or len(weights) != rows * k or len(training) != n_train * components:
        raise Error("UMAP transform initialization shape mismatch")
    for value in training:
        if not isfinite(value):
            raise Error("UMAP transform training embedding must be finite")
    var result = List[Float32](length=rows * components, fill=Float32(0.0))
    var ip = rebind[TR_U32P](indices.unsafe_ptr())
    var wp = rebind[TR_F32P](weights.unsafe_ptr())
    var ep = rebind[TR_F32P](training.unsafe_ptr())
    var op = rebind[TR_F32P](result.unsafe_ptr())
    var worst = TR_FLAGS
    for row in range(rows):
        var status = tr_row_initialize(ip, wp, ep, op, row, k, n_train, components)
        if status != TR_OK and status < worst:
            worst = status
    if worst < TR_FLAGS:
        tr_raise(worst)
    return result^


def host_refine_transform(
    initial: List[Float32], training: List[Float32], indices: List[UInt32],
    weights: List[Float32], rows: Int, n_train: Int, k: Int, components: Int,
    epochs: Int, a: Float32, b: Float32, seed: UInt64,
    learning_rate: Float32 = Float32(1),
    repulsion_strength: Float32 = Float32(1),
    negative_sample_rate: Int = 5,
) raises -> List[Float32]:
    """The refinement, `transform_rows.mojo::tr_row_refine_prep` and
    `tr_row_refine_epoch` per row (the device kernels' statements), epochs
    outermost as the device launches them; but for the sabotage arm's
    counter epoch."""
    if epochs < 1 or not isfinite(a) or not isfinite(b) or a <= Float32(0) or b <= Float32(0):
        raise Error("UMAP transform refinement requires positive finite parameters")
    if not isfinite(learning_rate) or learning_rate <= Float32(0) or (
        not isfinite(repulsion_strength) or repulsion_strength < Float32(0)
    ) or negative_sample_rate < 0 or negative_sample_rate > 2147483647:
        raise Error("UMAP transform optimizer controls are invalid")
    if len(initial) != rows * components:
        raise Error("UMAP transform refinement initialization shape mismatch")
    _ = host_initialize_transform(indices, weights, training, rows, n_train, k, components)
    var result = initial.copy()
    for value in result:
        if not isfinite(value):
            raise Error("UMAP transform refinement initialization must be finite")
    var scaled = List[UInt64](length=rows * k, fill=UInt64(0))
    var keys = List[UInt64](length=rows, fill=UInt64(0))
    var ip = rebind[TR_U32P](indices.unsafe_ptr())
    var wp = rebind[TR_F32P](weights.unsafe_ptr())
    var ep = rebind[TR_F32P](training.unsafe_ptr())
    var rp = rebind[TR_F32P](result.unsafe_ptr())
    var sp = rebind[TR_U64P](scaled.unsafe_ptr())
    var kp = rebind[TR_U64P](keys.unsafe_ptr())
    var worst = TR_FLAGS
    for row in range(rows):
        var status = tr_row_refine_prep(ip, wp, sp, kp, row, k)
        if status != TR_OK and status < worst:
            worst = status
    if worst < TR_FLAGS:
        tr_raise(worst)
    var b_word = bitcast[DType.uint64](Float64(b))
    var neg2ab = tr_neg2ab(a, b)
    var rep2b = tr_rep2b(repulsion_strength, b)
    for epoch in range(epochs):
        var alpha = tr_alpha(epoch, epochs, learning_rate)
        var draw_epoch = epoch
        comptime if UMAP_ORACLE_HOST_SABOTAGE:
            # THE SABOTAGE ARM, as in host_umap_vertex.
            draw_epoch = epoch + 1
        for row in range(rows):
            var status = tr_row_refine_epoch(
                rp, ep, ip, sp, keys[row], row, k, components, n_train,
                epoch, draw_epoch, alpha, a, b_word, neg2ab, rep2b, seed,
                negative_sample_rate,
            )
            if status != TR_OK and status < worst:
                worst = status
    if worst < TR_FLAGS:
        tr_raise(worst)
    for value in result:
        if not isfinite(value):
            raise Error("UMAP transform returned non-finite coordinates")
    return result^


def host_umap_transform(
    training_data: List[Float32], training_embedding: List[Float32],
    queries: List[Float32], n_train: Int, n_queries: Int, n_features: Int,
    params: UMAPParams,
) raises -> List[Float32]:
    """`transform`, `transform.mojo:181-228`, with the device k-NN replaced by
    its host restatement."""
    params.validate(n_train)
    if n_queries < 1 or n_features < 1 or (params.n_components < 1 or params.n_components > 32):
        raise Error("UMAP transform supports nonempty dense queries and 1 to 32 output dimensions")
    if len(training_data) != n_train * n_features or len(queries) != n_queries * n_features or len(training_embedding) != n_train * params.n_components:
        raise Error("UMAP transform input shape mismatch")
    for value in training_data:
        if not isfinite(value):
            raise Error("UMAP transform training input must be finite")
    for value in queries:
        if not isfinite(value):
            raise Error("UMAP transform queries must be finite")
    for value in training_embedding:
        if not isfinite(value):
            raise Error("UMAP transform training embedding must be finite")
    var k = params.n_neighbors
    var distances = List[Float32](length=n_queries * k, fill=Float32(0.0))
    var indices = List[UInt32](length=n_queries * k, fill=UInt32(0))
    host_knn_search(
        training_data, n_train, queries, n_queries, n_features, k,
        KNN_HOST_METRIC_FROM_IS_SQRT if params.metric == -1 else params.metric, True, distances, indices,
        params.metric_arg,
    )
    var weights = host_transform_memberships(distances, n_queries, k)
    var initial = host_initialize_transform(indices, weights, training_embedding, n_queries, n_train, k, params.n_components)
    var epochs = max(1, params.n_epochs // 3)
    if params.n_epochs == 0:
        # BATCH INVARIANCE. This read `100 if n_queries <= 10000 else 30`, so
        # one extra row in a request of ten thousand cut every other row's
        # refinement from 100 epochs to 30 and moved a row by 1.36 on a map
        # whose clusters sit about 11 apart, while the same one-row growth at
        # 9,999 moved no bit at all. The count no longer reads the request
        # size. The cost of that was measured, not asserted.
        epochs = 100
    var ab = params.curve()
    return host_refine_transform(
        initial, training_embedding, indices, weights, n_queries, n_train, k,
        params.n_components, epochs, ab[0], ab[1], params.random_seed,
        params.learning_rate, params.repulsion_strength, params.negative_sample_rate,
    )

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
optimize_sparse_layout_identical` is a Gauss-Seidel sweep; the device fold is
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
    fuzzy graph         umap/sparse_graph.mojo::sparse_fuzzy_simplicial_graph
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
    transform           umap/transform.mojo:33-228 (the host k-NN above for
                        knn_search, then transform_memberships,
                        initialize_transform, the epoch count, the curve and
                        refine_transform, which are host code in that file
                        and are restated here verbatim because the file
                        imports max.gpu.host)

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
from umap.sparse_graph import SparseFuzzySimplicialGraph, sparse_fuzzy_simplicial_graph


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
        KNN_HOST_METRIC_FROM_IS_SQRT, True, distances, indices,
    )
    canonicalize_self_neighbors(indices, distances, n_samples, params.n_neighbors)
    return sparse_fuzzy_simplicial_graph(
        indices^, distances^, n_samples, params.n_neighbors, params.set_op_mix_ratio,
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
    if n_components != 2 and n_components != 3:
        raise Error("UMAP spectral initialization supports only 2D or 3D")
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
    for c in range(n_components):
        var pivot = 0
        var peak = Float32(0.0)
        for i in range(n_samples):
            var value = embedding[i * n_components + c]
            if not _finite(value):
                raise Error("UMAP spectral solver returned a non-finite value")
            var magnitude = value if value >= Float32(0.0) else -value
            if magnitude > peak:
                peak = magnitude
                pivot = i
        if not (peak > Float32(0.0)):
            raise Error("UMAP spectral solver returned a zero component")
        var scale = Float32(10.0) / peak
        if embedding[pivot * n_components + c] < Float32(0.0):
            scale = -scale
        for i in range(n_samples):
            embedding[i * n_components + c] *= scale
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
    if n_samples < 2 or (n_components != 2 and n_components != 3):
        raise Error("UMAP device optimizer supports 2D/3D layouts")
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
                else:
                    host_umap_vertex[3](
                        first, row_offsets, tails, scaled, second, v, n_samples,
                        epoch, alpha, negative_rate, neg2ab, rep2b, a, b, seed,
                    )
            else:
                if n_components == 2:
                    host_umap_vertex[2](
                        second, row_offsets, tails, scaled, first, v, n_samples,
                        epoch, alpha, negative_rate, neg2ab, rep2b, a, b, seed,
                    )
                else:
                    host_umap_vertex[3](
                        second, row_offsets, tails, scaled, first, v, n_samples,
                        epoch, alpha, negative_rate, neg2ab, rep2b, a, b, seed,
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
    if n_samples < 2 or (n_components != 2 and n_components != 3):
        raise Error("UMAP optimizer supports at least two samples in 2D/3D")
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


def host_umap_fit_transform(
    x_rowmajor: List[Float32], n_samples: Int, n_features: Int, params: UMAPParams
) raises -> List[Float32]:
    """`sparse_fit_transform`, `sparse_estimator.mojo:91-129`, on the host."""
    params.validate(n_samples)
    if n_features < 1 or len(x_rowmajor) != n_samples * n_features:
        raise Error("UMAP input does not match its declared shape")
    if params.n_components != 2 and params.n_components != 3:
        raise Error("UMAP fit_transform currently supports only 2D or 3D")
    if n_samples < 2 * params.n_components + 4:
        raise Error("UMAP fit_transform has too few samples for spectral init")
    var graph = host_umap_fuzzy_graph(x_rowmajor, n_samples, n_features, params)
    var initial = host_umap_spectral_initialize(graph.copy(), params.n_components, params.random_seed)
    var epochs = params.n_epochs
    if epochs == 0:
        epochs = 200
    var curve = fit_umap_curve(params.min_dist, params.spread)
    return host_optimize_sparse_layout(
        initial^, graph, n_samples, params.n_components, epochs,
        params.learning_rate, params.negative_sample_rate, params.repulsion_strength,
        curve.a, curve.b, params.random_seed,
    )


# ---------------------------------------------------------------------------
# The transform (umap/transform.mojo, restated)
# ---------------------------------------------------------------------------


def host_transform_memberships(distances: List[Float32], rows: Int, k: Int) raises -> List[Float32]:
    """`transform_memberships`, `transform.mojo:33-75`, verbatim."""
    if rows < 1 or k < 2 or len(distances) != rows * k:
        raise Error("UMAP transform neighbor shape mismatch")
    for i in range(len(distances)):
        if not isfinite(distances[i]) or distances[i] < Float32(0):
            raise Error("UMAP transform neighbor distances must be finite and nonnegative")
        if i % k > 0 and distances[i] < distances[i - 1]:
            raise Error("UMAP transform neighbors must be distance-sorted")
    var target = identical_log2_64(Float64(k))
    var weights = List[Float32]()
    for row in range(rows):
        # BATCH INVARIANCE. The sigma floor's mean is THIS ROW's own k
        # neighbor distances and never the whole request's, so a batch of N
        # is the concatenation of N batches of one. Measured effectively
        # inert on ordinary data by lane/umap-batch-determinism, and
        # repaired anyway, because an inert coupling is still a coupling.
        var mean = Float64(0)
        for j in range(k):
            mean += Float64(distances[row * k + j])
        mean /= Float64(k)
        var lo = Float64(0)
        var hi = Float64(-1)
        var sigma = Float64(1)
        for iteration in range(64):
            var total = Float64(0)
            for j in range(1, k):
                var distance = Float64(distances[row * k + j])
                total += Float64(1) if distance == 0 else identical_exp64(-distance / sigma)
            if abs(total - target) <= Float64(1.0e-5):
                continue
            if total > target:
                hi = sigma
                sigma = (lo + hi) * Float64(0.5)
            else:
                lo = sigma
                sigma = sigma * Float64(2) if hi < 0 else (lo + hi) * Float64(0.5)
        sigma = max(sigma, Float64(0.001) * mean)
        var row_sum = Float32(0)
        for j in range(k):
            var distance = distances[row * k + j]
            var weight = Float32(1) if distance == Float32(0) else Float32(identical_exp64(-Float64(distance) / sigma))
            weights.append(weight)
            row_sum += weight
        if not isfinite(row_sum) or row_sum <= Float32(0):
            raise Error("UMAP transform query has no positive memberships")
    return weights^


def host_initialize_transform(
    indices: List[UInt32], weights: List[Float32], training: List[Float32],
    rows: Int, n_train: Int, k: Int, components: Int,
) raises -> List[Float32]:
    """`initialize_transform`, `transform.mojo:78-114`, verbatim."""
    if rows < 1 or n_train < 2 or k < 2 or k > n_train or (components != 2 and components != 3):
        raise Error("UMAP transform initialization dimensions are unsupported")
    if len(indices) != rows * k or len(weights) != rows * k or len(training) != n_train * components:
        raise Error("UMAP transform initialization shape mismatch")
    for value in training:
        if not isfinite(value):
            raise Error("UMAP transform training embedding must be finite")
    for i in range(rows * k):
        if indices[i] >= UInt32(n_train) or not isfinite(weights[i]) or weights[i] < Float32(0) or weights[i] > Float32(1):
            raise Error("UMAP transform membership or neighbor index is invalid")
    var result = List[Float32]()
    for row in range(rows):
        var total = Float32(0)
        var exact = -1
        for j in range(k):
            var w = weights[row * k + j]
            total += w
            if exact < 0 and w == Float32(1):
                exact = Int(indices[row * k + j])
        if not isfinite(total) or total <= Float32(0):
            raise Error("UMAP transform query has no positive memberships")
        for c in range(components):
            var value = Float32(0)
            if exact >= 0:
                value = training[exact * components + c]
            else:
                for j in range(k):
                    var tail = Int(indices[row * k + j])
                    # ONE rounding: the default (contract=fast) build fused this
                    # product into the accumulate; explicit so no build mode can
                    # change it (lane/explicit-fma-contract-proof, 2026-09-26)
                    value = identical_mul_add(weights[row * k + j] / total, training[tail * components + c], value)
            if not isfinite(value):
                raise Error("UMAP transform initialization is not finite")
            result.append(value)
    return result^


def host_refine_transform(
    initial: List[Float32], training: List[Float32], indices: List[UInt32],
    weights: List[Float32], rows: Int, n_train: Int, k: Int, components: Int,
    epochs: Int, a: Float32, b: Float32, seed: UInt64,
    learning_rate: Float32, repulsion_strength: Float32, negative_sample_rate: Int,
) raises -> List[Float32]:
    """`refine_transform`, `transform.mojo:117-178`, verbatim but for the
    sabotage arm's counter epoch."""
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
    # BATCH INVARIANCE, the two couplings that lived in this function. Each
    # row's edge schedule is scaled by THAT ROW's largest membership rather
    # than by the whole request's, and each row's negative-sample counter is
    # keyed on a hash of that row's own k neighbor INDICES rather than on
    # `row * k + j`, which was a position in the request and not a property
    # of the query. The memberships are deliberately NOT in the key: hashing
    # them was tried and made a 4-ULP change to one input feature move the
    # output by 0.146 where the shipped code needed 16,384 ULPs to move it by
    # 0.005, which trades a batch coupling for an input discontinuity. The
    # indices are integers and do not wobble. Both are precomputed once per
    # row, so the epoch loop below is otherwise unchanged.
    var row_max = List[Float32]()
    var row_key = List[UInt64]()
    for row in range(rows):
        var maximum = Float32(0)
        var key = UInt64(0x9E3779B97F4A7C15)
        for j in range(k):
            var edge = row * k + j
            maximum = max(maximum, weights[edge])
            key = _splitmix64(key ^ UInt64(indices[edge]))
        if not isfinite(maximum) or maximum <= Float32(0):
            raise Error("UMAP transform query has no positive memberships")
        row_max.append(maximum)
        row_key.append(key)
    for epoch in range(epochs):
        var alpha = (Float32(0.25) * learning_rate) * Float32(Float64(epochs - epoch) / Float64(epochs))
        var draw_epoch = epoch
        comptime if UMAP_ORACLE_HOST_SABOTAGE:
            # THE SABOTAGE ARM, as in host_umap_vertex.
            draw_epoch = epoch + 1
        for row in range(rows):
            for j in range(k):
                var edge = row * k + j
                var edge_key = row_key[row] ^ (UInt64(j) * UInt64(0x9E3779B97F4A7C15))
                var scaled = Float64(weights[edge]) / Float64(row_max[row])
                if Int(Float64(epoch + 1) * scaled) <= Int(Float64(epoch) * scaled):
                    continue
                var tail = Int(indices[edge])
                for slot in range(negative_sample_rate + 1):
                    var other = tail
                    if slot > 0:
                        var counter = seed ^ (UInt64(draw_epoch) * UInt64(0xD1B54A32D192ED03)) ^ (edge_key * UInt64(0x94D049BB133111EB)) ^ UInt64(slot - 1)
                        other = Int(_splitmix64(counter) % UInt64(n_train))
                    var distance = Float32(0)
                    for c in range(components):
                        var delta = result[row * components + c] - training[other * components + c]
                        distance = identical_mul_add(delta, delta, distance)
                    if not isfinite(distance):
                        raise Error("UMAP transform refinement distance is not finite")
                    if distance <= Float32(0):
                        continue
                    var powered = Float32(identical_pow64(Float64(distance), Float64(b)))
                    var coeff = Float32(0)
                    if slot == 0:
                        coeff = -Float32(2) * a * b * (powered / distance) / identical_mul_add(a, powered, Float32(1))
                    else:
                        coeff = Float32(2) * repulsion_strength * b / ((Float32(0.001) + distance) * identical_mul_add(a, powered, Float32(1)))
                    if not isfinite(coeff):
                        raise Error("UMAP transform gradient is not finite")
                    for c in range(components):
                        var delta = result[row * components + c] - training[other * components + c]
                        result[row * components + c] = identical_mul_add(alpha, _clip(coeff * delta), result[row * components + c])
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
    if n_queries < 1 or n_features < 1 or (params.n_components != 2 and params.n_components != 3):
        raise Error("UMAP transform supports nonempty dense queries and 2D/3D output")
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
        KNN_HOST_METRIC_FROM_IS_SQRT, True, distances, indices,
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
    var curve = fit_umap_curve(params.min_dist, params.spread)
    return host_refine_transform(
        initial, training_embedding, indices, weights, n_queries, n_train, k,
        params.n_components, epochs, curve.a, curve.b, params.random_seed,
        params.learning_rate, params.repulsion_strength, params.negative_sample_rate,
    )

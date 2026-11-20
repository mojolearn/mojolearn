# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of Mojolearn, https://doi.org/10.5281/zenodo.22068632
"""BRUTE-FORCE k-NN DISTANCES ON THE MATRIX UNIT: THE NEIGHBORS ROUTE OF `fast`
(lane fx-fast-nvidia-routes, 2026-11-19).

EXPERIMENT `-D MOJOLEARN_NEIGHBORS_FAST_MATRIX_UNITS`, default OFF, A/B owed with quality beside time
(docs/apple-fast/EXPERIMENTS.md). Unverified: written without a compile. A family route of the matrix-units row of
checks/mode_policy.mojo (`ROUTE_NEIGHBORS_MATRIX_UNITS`): legal only in a `fast` build
(`-D MOJOLEARN_MODE_FAST_ROWS`); a compile error in a build that promises identical bits. With the define off
nothing here is compiled into a launch.

WHAT `identical` PAYS HERE. The expanded L2 distance is one FMA chain per (query, index) cell over the feature
axis, in one fixed order, with the epilogue in the same kernel (IDENTITY_PATHS rows 19, 23, 24), plus the
exponent-admission passes of the exact chain. The vendor product is closed to it because a k-split is a
summation order.

WHAT THE ROUTE DOES. The tiled arm's distance step becomes the reference algorithm's own two launches: the product
Q . Y^T on MAX `linalg.matmul` (NVIDIA: tensor cores, TF32 operands, FP32 accumulation, which is cuVS's default for
this product; AMD: fp32 MFMA), then `expand_distances_kernel` (`qn + yn - 2 dot`, the clamp, the optional root) in
FP32. The norms stay exact FP32. Selection, the index-axis tiling and the merge are unchanged, so ties are still
broken by index. One source for NVIDIA and AMD; on Apple the distance already has matrix-unit kernels
(neighbors/checks/apple_mma_distance.mojo, neighbors/impl/detail/fast_mma_knn.mojo), so the row points at those and
this route is off there; the host column is exact.

THE RULE (`knn_distance_units_takes`): sizes and the hardware only.
  1. L2Expanded or L2SqrtExpanded, the device selectors (not the vendor top-k arm).
  2. n_features >= 256. The fused kernels the route replaces never store the distance matrix; the route writes
     and re-reads rows x cols cells (8 bytes of traffic a cell). At the scalar chain's rate (about 1e13
     multiply-adds a second on the L40S) against a matrix unit several times that, the stored matrix costs about
     as much as 200 to 250 chain steps a cell, so under that many features the fused exact kernel is the faster
     one and keeps the request. Reasoned, not measured: the owed A/B times both sides of it.
  3. `vendor_matmul_size_rule` on one full tile (edges >= 64, tile work >= 2^27).
  4. The caller's distance tile holds query_tile x index_tile cells (the tiled arm's own requirement).

QUALITY. The dot of a cell differs from FP32 by at most `matrix_units_cell_bound` (NVIDIA: 1.95e-3 of
sum |q_k y_k|, typically 4e-4 / sqrt(K) of it); the distance by twice that, an error relative to the NORMS, not to
the distance. Neighbours whose distances differ by less than that can change order or membership. The check
(checks/native_tests/knn_matrix_units.mojo) holds every distance to the bound and the recall against the exact
distances to its floor; the owed A/B records recall@k against the zero-row fast set and the estimator's own
metric.
"""
from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.mode_policy import ROUTE_NEIGHBORS_MATRIX_UNITS
from gemm.fast_matrix_units import vendor_matmul_size_rule

#: The route: the tiled brute-force k-NN distance product on the matrix unit in an NVIDIA or AMD `fast` build.
comptime KNN_DISTANCE_MATRIX_UNITS = ROUTE_NEIGHBORS_MATRIX_UNITS

#: rule 2: the feature count at which the stored distance matrix is paid for by the faster product.
comptime KNN_UNITS_MIN_FEATURES = 256


@always_inline
def knn_distance_units_takes(
    query_tile: Int,
    index_tile: Int,
    n_queries: Int,
    n_index: Int,
    n_features: Int,
    l2_expanded: Bool,
    vendor_topk: Bool,
    tile_cells: Int,
) -> Bool:
    """Whether the route takes this request (the rule of the module docstring). `tile_cells` is the length of
    the caller's distance tile. Always False without the route."""
    comptime if not KNN_DISTANCE_MATRIX_UNITS:
        return False
    if not l2_expanded or vendor_topk:
        return False
    if n_features < KNN_UNITS_MIN_FEATURES:
        return False
    if tile_cells < query_tile * index_tile:
        return False
    return vendor_matmul_size_rule(min(query_tile, n_queries), min(index_tile, n_index), n_features)


def knn_distance_units_product(
    ctx: DeviceContext,
    mut dots: DeviceBuffer[DType.float32],
    mut q: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    rows: Int,
    cols: Int,
    n_features: Int,
) raises:
    """`dots[rows x cols] = q[rows x d] . y[cols x d]^T` on the matrix unit: one asynchronous launch, no
    workspace, no wait. The caller runs `expand_distances_kernel` on `dots` next and keeps the three buffers
    alive past its own synchronize."""
    comptime if KNN_DISTANCE_MATRIX_UNITS:
        var tz = TileTensor(dots, row_major(rows, cols))
        var tq = TileTensor(q, row_major(rows, n_features))
        var ty = TileTensor(y, row_major(cols, n_features))
        matmul[transpose_b=True, target="gpu"](tz, tq, ty, ctx)

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""cuML's `ML::linkage::single_linkage`, the C++ entry.

PORT OF `cuml/cpp/src/hierarchy/linkage.cu` and
`cuml/cpp/include/cuml/cluster/linkage.hpp`, cuML `00094f7` (the
`cuml-v26.08.00` checkout carries the same two files). cuML's file is a
forwarder: it builds the mdspan views and calls
`cuvs::cluster::agglomerative::single_linkage` with
`Linkage::KNN_GRAPH` when `use_knn` and `Linkage::PAIRWISE` otherwise, and
`c` only when `use_knn` (`linkage.cu:16-42`). So is this.

THE DEFAULTS, AND WHICH IS PORTED. `linkage.hpp:51-52` defaults
`use_knn = false, c = 15`, and that PAIRWISE arm is rung 1, ported and
gated here. cuML's PYTHON layer defaults the other way --
`AgglomerativeClustering(connectivity="knn")`, `agglomerative.pyx:123` --
so their Python default path is the KNN_GRAPH arm, which is rung 2 and is
REFUSED BY NAME by `get_distance_graph`. A Python surface over this lane
must default `connectivity="pairwise"` and say so, or refuse; see the
README's HAND-OFF.

PARAMETERS, EACH HONORED OR REFUSED BY NAME:
  X, n_rows, n_cols   honored (dense row-major Float32 on the device)
  n_clusters          honored (1 <= n_clusters <= n_rows, else raises)
  metric              L2SqrtExpanded (1) and L2Expanded (0) honored; L1 (3),
                      CosineExpanded (2) and every other code raise by name
                      in `pairwise_distances`
  children, labels    honored, `(n_rows - 1) x 2` and `n_rows` Int32
  use_knn             `false` honored; `true` raises by name (rung 2)
  c                   forwarded only with `use_knn`, as theirs (`:41`)
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from hierarchy.original.edge_order import LINK_SAB_NONE
from hierarchy.derived.cluster.detail.agglomerative import EXTRACT_TPB
from hierarchy.derived.cluster.detail.connectivities import (
    LINKAGE_KNN_GRAPH,
    LINKAGE_PAIRWISE,
)
from hierarchy.derived.cluster.detail.single_linkage import (
    SingleLinkageOutput,
    single_linkage as cuvs_single_linkage,
)
from neighbors.original.pinned_distance_tile import PINNED_TILE_TPB


comptime LINKAGE_DEFAULT_C = 15
"""`linkage.hpp:52` `int c = 15`."""


def single_linkage(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    n_clusters: Int,
    metric: Int,
    mut children: DeviceBuffer[DType.int32],
    mut labels: DeviceBuffer[DType.int32],
    use_knn: Bool = False,
    c: Int = LINKAGE_DEFAULT_C,
    tile_tpb: Int = PINNED_TILE_TPB,
    mst_tpb: Int = 256,
    extract_tpb: Int = EXTRACT_TPB,
    sabotage: Int32 = LINK_SAB_NONE,
) raises -> SingleLinkageOutput:
    """`linkage.cu:16-42`."""
    if len(children) < (n_rows - 1) * 2:
        raise Error(
            "hierarchy.single_linkage: children holds " + String(len(children))
            + " < (n_rows - 1) * 2 = " + String((n_rows - 1) * 2)
        )
    if len(labels) < n_rows:
        raise Error(
            "hierarchy.single_linkage: labels holds " + String(len(labels))
            + " < n_rows = " + String(n_rows)
        )
    var linkage = LINKAGE_KNN_GRAPH if use_knn else LINKAGE_PAIRWISE
    return cuvs_single_linkage(
        ctx, x, n_rows, n_cols, metric, children, labels,
        c if use_knn else 0, n_clusters, linkage,
        tile_tpb, mst_tpb, extract_tpb, sabotage,
    )

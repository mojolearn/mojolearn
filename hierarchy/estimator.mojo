# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host-pointer surface for the hierarchy section: single-linkage agglomerative
clustering.

**THIS IS THE ENTRY THE PYTHON PACKAGE USES.**
`bindings/_mojolearn_solver.mojo` calls `linkage_fit_host` and
`python/mojolearn/_hierarchy_impl.py` calls that, so everything below is what
a `mojolearn.AgglomerativeClustering().fit(X)` actually runs. The shape is
`dbscan/estimator.mojo`'s: host pointers in, device buffers owned here for
exactly one call, results read back, nothing retained.

The ported entry is `hierarchy/impl/hierarchy/linkage.mojo::single_linkage`
(cuML `cpp/src/hierarchy/linkage.cu`, forwarding to cuVS
`cluster/detail/single_linkage.cuh` and RAFT's Boruvka MST). The lane's
README, `DERIVATION_MAP.tsv` and `NOT_IMPLEMENTED.tsv` are the record of what is and is
not in it, and this file re-decides none of it.

WHICH TREE THE LINE NUMBERS BELOW COME FROM. cuML `upstream/cuml-v26.08.00`
(265b9da) and cuVS `upstream/cuvs-v26.08.00` (6ba2ce2), read 2026-08-24.
`hierarchy/DERIVATION_MAP.tsv` pins cuVS at `94c2819` and RAFT at `661a3b8`,
which are the UNTAGGED default-branch checkouts rather than the release
tags; the code is the same in both trees but the LINE NUMBERS are not, so a
citation here and one in the lane's own files can disagree by a few dozen
lines and both be right about the same statement.

X IS ROW-MAJOR, unlike `solver/estimator.mojo`'s. cuML's
`AgglomerativeClustering.fit` calls `check_inputs(..., order="C")`
(`agglomerative.pyx:144-152`) and `pairwise_distances` reads the dense
design row by row, so the layout the Python `_arrays.as_f32_c` already
produces is the layout this takes. No copy beyond the host-to-device one.

==========================================================================
DEVIATION 881 -- THE CONNECTIVITY DEFAULT IS THE C++ ONE, NOT THE PYTHON ONE
==========================================================================
WHAT THEIRS DOES: cuML has two defaults for the same knob. The C++ entry
defaults `use_knn = false` (`linkage.hpp:43-44`), the PAIRWISE arm; their
Python estimator defaults `connectivity="knn"` (`agglomerative.pyx:123`),
the KNN_GRAPH arm.

WHAT OURS DOES: `use_knn` defaults False here and `connectivity` defaults
`"pairwise"` at the Python surface, because the `Linkage::KNN_GRAPH`
specialization (`connectivities.cuh:49`), the cross-component fix-up it
needs (`connect_knn_graph`'s two overloads, `mst.cuh:67` and `:131`) and
`merge_msts` (`mst.cuh:32`) are rung 2 and NOT PORTED. `use_knn=True` is
not quietly downgraded: it reaches `get_distance_graph`, which REFUSES IT
BY NAME. So the default here is cuML's C++ default and scikit-learn's dense
graph, and their Python default is the arm that raises.

Recorded as a deviation rather than left implicit because it is a DEFAULT
that differs from the upstream estimator this class mirrors, and a caller
porting a cuML script gets a different graph without asking for one.

THE PYTHON PATH EMITS NO IDENTITY CARD, AND THAT IS OWED
--------------------------------------------------------
`solver/estimator.mojo` hands `cd_fit_traced` a live `IdentityTrace`, so a
Python coordinate-descent fit writes the same card its Mojo driver does.
There is no equivalent here: the ported `single_linkage` entry takes no
trace, and `hierarchy/linkage_main.mojo` builds its eight-stage card by
RE-RUNNING `pairwise_distances` and `build_sorted_mst` beside the fit. A
card written from this file would either duplicate that work on every fit
or carry a different stage list from the lane's, and a card that cannot be
diffed against `linkage.*` is worse than none. So: the certified card is
`hierarchy/linkage_main.mojo`'s, this path runs the same `single_linkage`
entry, and no claim is made here that a Python fit was carded. Closing that
gap means threading an `IdentityTrace` through `single_linkage.mojo`, which
belongs to the hierarchy lane, not to this surface.

WHAT IS REFUSED, AND WHERE
--------------------------
All of it is the ported code's, reached from here: `use_knn=True` (rung 2,
`get_distance_graph`), every metric but L2SqrtExpanded (1) and L2Expanded
(0) (`pairwise_distances`), `n_rows < 2` and `n_rows > 46340`, `n_clusters
< 1` and `n_clusters > n_rows` (`single_linkage.cuh`), and the `children` /
`labels` size checks in `linkage.mojo` itself.

ON THE 46340 BOUND, because the lane's own justification names the wrong
arm. `hierarchy/NOT_IMPLEMENTED.tsv` and `connectivities.mojo` both say "their
`int nnz = m * m` (`connectivities.cuh:145`)". In BOTH cuVS trees that
declaration belongs to the `Linkage::KNN_GRAPH` specialization, which is
the arm that is NOT ported; the PAIRWISE arm declares `size_t nnz = m * m`
(`connectivities.cuh:191` in the tag, `:197` at `94c2819`). The BOUND still
stands -- `m` is their `int` `value_idx`, so the product overflows before
it is widened to `size_t` -- but the citation points at the other
specialization. Reported, not edited: those two files belong to the
hierarchy lane.
"""

from max.gpu.host import DeviceContext

from hierarchy.impl.cluster.detail.connectivities import (
    DISTANCE_L2_SQRT_EXPANDED,
)
from hierarchy.impl.hierarchy.linkage import LINKAGE_DEFAULT_C, single_linkage


def linkage_fit_host(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    children_ptr: MutPointer[Int32, MutUntrackedOrigin],
    labels_ptr: MutPointer[Int32, MutUntrackedOrigin],
    info_ptr: MutPointer[Int32, MutUntrackedOrigin],
    n_rows: Int,
    n_cols: Int,
    n_clusters: Int,
    metric: Int = DISTANCE_L2_SQRT_EXPANDED,
    use_knn: Bool = False,
    c: Int = LINKAGE_DEFAULT_C,
) raises -> Int:
    """Cluster host-resident ROW-MAJOR data. Returns the Boruvka round count.

    `x_ptr` is `n_rows x n_cols` float32, row-major.
    `children_ptr` receives `(n_rows - 1) * 2` Int32 -- the dendrogram, one
    merge per row, in cuVS's orientation (`src` is the vertex that added the
    edge; see the lane README's honesty note about scikit-learn's rows).
    `labels_ptr` receives `n_rows` Int32 in `0..n_clusters-1`, numbered by
    cuVS's `extract_flattened_clusters` (descending root index), which is NOT
    scikit-learn's numbering.
    `info_ptr` receives two Int32: `[0]` the Boruvka round count (the card's
    `linkage.mst.rounds`, an integer stage), `[1]`
    `n_connected_components`.

    `n_connected_components` COMES FROM THE PORT, NOT FROM HERE. `single_
    linkage.mojo:156` returns the literal 1, mirroring
    `single_linkage.cuh:301`, which is sound on the PAIRWISE arm because
    the graph is complete and Boruvka finishes in one component. It is read
    back rather than restated in Python so that if the port ever computes it
    the surface follows without an edit.
    """
    if n_rows < 2:
        raise Error(
            "linkage_fit_host needs n_rows >= 2, got " + String(n_rows)
        )
    if n_cols < 1:
        raise Error(
            "linkage_fit_host needs n_cols >= 1, got " + String(n_cols)
        )

    var n_children = (n_rows - 1) * 2
    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_cols)
    var children = ctx.enqueue_create_buffer[DType.int32](n_children)
    var labels = ctx.enqueue_create_buffer[DType.int32](n_rows)
    ctx.synchronize()

    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.synchronize()

    var out = single_linkage(
        ctx, x, n_rows, n_cols, n_clusters, metric, children, labels,
        use_knn, c,
    )

    var hc = ctx.enqueue_create_host_buffer[DType.int32](n_children)
    var hl = ctx.enqueue_create_host_buffer[DType.int32](n_rows)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hc.unsafe_ptr(), src_buf=children)
    ctx.enqueue_copy(dst_ptr=hl.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()

    for i in range(n_children):
        children_ptr.unsafe_store(i, hc.unsafe_ptr().unsafe_load(i))
    for i in range(n_rows):
        labels_ptr.unsafe_store(i, hl.unsafe_ptr().unsafe_load(i))
    info_ptr.unsafe_store(0, Int32(out.n_boruvka_rounds))
    info_ptr.unsafe_store(1, Int32(out.n_connected_components))

    # [[mojo-buffer-freed-at-last-use]]: every buffer outlives the queue.
    _ = hc^
    _ = hl^
    _ = x^
    _ = children^
    _ = labels^
    return out.n_boruvka_rounds

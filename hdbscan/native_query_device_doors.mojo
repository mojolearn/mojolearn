# SPDX-License-Identifier: Apache-2.0
"""Existing GIL-free HDBSCAN GPU query doors, moved without body changes."""
from core.neural_context import process_ctx
from hdbscan.impl.hdbscan_door_impl import HDB_DEVCTX_SLOT
from hdbscan.estimator import hdbscan_approximate_predict_host, hdbscan_membership_vector_host, hdbscan_all_points_membership_vectors_host

comptime _DEVCTX_SLOT = HDB_DEVCTX_SLOT


def _hdbscan_predict_run(
    x: List[Float32],
    m: Int,
    d: Int,
    core: List[Float32],
    labels: List[Int32],
    lambdas: List[Float32],
    n_edges: Int,
    n_clusters: Int,
    deaths: List[Float32],
    selected: List[Int32],
    iic: List[Int32],
    q: List[Float32],
    nq: Int,
    min_samples: Int,
    lo: MutPointer[Int32, MutUntrackedOrigin],
    po: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    var ctx = process_ctx[_DEVCTX_SLOT]()
    hdbscan_approximate_predict_host(
        ctx, x, m, d, core, labels, lambdas, n_edges, n_clusters, deaths,
        selected, iic, q, nq, min_samples, lo, po,
    )
    ctx.synchronize()
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^


def _hdbscan_membership_run(
    x: List[Float32], m: Int, d: Int, core: List[Float32], labels: List[Int32],
    parents: List[Int32], lambdas: List[Float32], n_edges: Int,
    n_clusters: Int, deaths: List[Float32], selected: List[Int32],
    iic: List[Int32], ex_idx: List[Int32], offsets: List[Int32],
    q: List[Float32], nq: Int, min_samples: Int,
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    var ctx = process_ctx[_DEVCTX_SLOT]()
    hdbscan_membership_vector_host(
        ctx, x, m, d, core, labels, parents, lambdas, n_edges, n_clusters,
        deaths, selected, iic, ex_idx, offsets, q, nq, min_samples, out_ptr,
    )
    ctx.synchronize()
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^


def _hdbscan_all_points_run(
    x: List[Float32], m: Int, d: Int, parents: List[Int32],
    lambdas: List[Float32], n_edges: Int, n_clusters: Int,
    deaths: List[Float32], selected: List[Int32], iic: List[Int32],
    ex_idx: List[Int32], offsets: List[Int32], row0: Int, count: Int,
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    var ctx = process_ctx[_DEVCTX_SLOT]()
    hdbscan_all_points_membership_vectors_host(
        ctx, x, m, d, parents, lambdas, n_edges, n_clusters, deaths, selected,
        iic, ex_idx, offsets, row0, count, out_ptr,
    )
    ctx.synchronize()
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^

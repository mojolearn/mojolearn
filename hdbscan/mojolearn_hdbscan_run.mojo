# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RUN bodies of `bindings/_mojolearn_hdbscan.mojo` (wave 8 run split).

Typed Mojo only. The binding keeps the address and parameter reads (UNPACK),
releases the GIL around each call here exactly where it did before, and builds
the returned count (PACK).
"""

from core.hostptr import f32_ptr, i32_ptr, read_f32
from hdbscan.native_query_device_doors import _hdbscan_membership_run, _hdbscan_all_points_run


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    return i32_ptr(addr)


def hdbscan_membership_batches_run(
    x: List[Float32], m: Int, d: Int, core: List[Float32], labels: List[Int32],
    parents: List[Int32], lambdas: List[Float32], n_edges: Int,
    n_clusters: Int, deaths: List[Float32], selected: List[Int32],
    iic: List[Int32], ex_idx: List[Int32], offsets: List[Int32],
    nq: Int, min_samples: Int, batch: Int, n_selected: Int,
    q_addr: Int, out_addr: Int,
) raises:
    """The query-batch loop of `hdbscan_membership_vector` (called with the GIL released)."""
    var first = 0
    while first < nq:  # driver-loop: over query batches inside the one public call
        var count = min(batch, nq - first)
        var q = read_f32(q_addr + first * d * 4, count * d)
        _hdbscan_membership_run(
            x, m, d, core, labels, parents, lambdas, n_edges, n_clusters,
            deaths, selected, iic, ex_idx, offsets, q, count, min_samples,
            _f32_ptr(out_addr + first * n_selected * 4),
        )
        first += count


def hdbscan_all_points_batches_run(
    x: List[Float32], m: Int, d: Int, parents: List[Int32],
    lambdas: List[Float32], n_edges: Int, n_clusters: Int,
    deaths: List[Float32], selected: List[Int32], iic: List[Int32],
    ex_idx: List[Int32], offsets: List[Int32], row0: Int, count: Int,
    batch: Int, n_selected: Int, out_addr: Int,
) raises:
    """The training-row batch loop of `hdbscan_all_points_membership_vectors`
    (called with the GIL released)."""
    var done = 0
    while done < count:  # driver-loop: over training-row batches inside the one public call
        var part = min(batch, count - done)
        _hdbscan_all_points_run(
            x, m, d, parents, lambdas, n_edges, n_clusters, deaths, selected,
            iic, ex_idx, offsets, row0 + done, part,
            _f32_ptr(out_addr + done * n_selected * 4),
        )
        done += part

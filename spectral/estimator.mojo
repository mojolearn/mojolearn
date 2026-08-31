# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host-list surface for spectral clustering: what `bindings/` calls.

Shaped like `kde/estimator.mojo` and `metrics/estimator.mojo`. Host lists in,
host lists out; no `DeviceBuffer` and no `DeviceContext` crosses this
boundary. `spectral/README.md` section 5 (HAND-OFF) names what belongs to a
Python caller and this file performs none of it: the `n_components` default,
`n_neighbors=10`, `n_init=10`, the handling of `eigen_tol`, and the refusals
of `affinity` outside `{nearest_neighbors, precomputed}` and
`assign_labels != 'kmeans'`, are all `python/mojolearn/_spectral_impl.py`'s.

**THAT README SECTION IS WRONG ABOUT ONE OF THEM AND IS NOT THIS LANE'S FILE
TO EDIT.** It says `eigen_tol='auto'` means "cuVS's `1e-5`". At the 26.08
pin it does not: cuML's `spectral_clustering.pyx:346-347` maps `'auto'` to
`0.0`, and `cuvs::cluster::spectral::detail::fit_predict` then assigns
`spectral_embedding_config.tolerance = config.tolerance` unconditionally
(`cpp/src/cluster/detail/spectral.cuh:36`), OVERWRITING the `1e-5` struct
default at `preprocessing/spectral_embedding.hpp:59`. `'auto'` therefore
means a ZERO tolerance on the clustering path. See DEVIATION 890 below.

Two entries, because cuVS has two: `fit_predict` on a DATASET (which builds
a kNN connectivity graph first, cuML's `affinity='nearest_neighbors'`) and
`fit_predict` on a connectivity GRAPH given as COO triples (cuML's
`affinity='precomputed'`). Both land in
`spectral/derived/spectral/spectral_clustering.mojo`, which is cuML's own
surface, so nothing here reimplements a forward.

**THE IDENTITY CARD IS EMITTED FROM HERE WHEN THE ENVIRONMENT ASKS FOR
ONE**, unlike `metrics/estimator.mojo`, and for a reason: the ported
`fit_predict` takes an `IdentityTrace` as a REQUIRED argument and records
exactly the stage list `IDENTICAL_SPECTRAL_CONTRACT.md` section 8 freezes.
So `MOJOLEARN_IDENTITY_TRACE=<path>` around a Python `fit_predict` produces
the same card `spectral/spectral_main.mojo` does for the same shape, and
nothing partial or invented. `IdentityTrace()` reads that variable once and
is disabled when it is unset, which is the shipping state.

WHAT THIS FILE DOES NOT DO. It does not validate the DATA beyond lengths:
`transform_graph` refuses a non-finite or negative affinity BY NAME, and
`create_connectivity_graph` refuses a non-finite dataset value BY NAME,
before any recorded stage. It does not sort a caller's COO either, because
it must not: `compute_graph_laplacian` sorts it itself through
`_mark_and_insert_diagonal` and then refuses a repeated `(row, col)` key by
name (DEVIATIONS 775/777), which is the behavior a caller has to see.

DEVIATIONS 890-899 are this surface's, shared with `metrics/estimator.mojo`.
Two are spent, both on the SURFACE and neither on arithmetic:

  * **890**, on `_config` below: a non-positive `eigen_tol` is refused by
    name instead of mirroring cuML's `'auto' -> 0.0`, which overwrites
    cuVS's own `1e-5` struct default with a zero that disables the
    convergence test outright.
  * **891**, in `python/mojolearn/_spectral_impl.py`: `random_state=None`
    means cuVS's struct default SEED 0 and is deterministic, where cuML's
    Python draws a fresh random seed and scikit-learn's None means an
    entropy draw. The no-seed arm of the Lanczos start vector is refused
    outright by DEVIATION 772, so None cannot mean "no seed" here.

Nothing else in this file computes.
"""

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from spectral.derived.spectral.spectral_clustering import (
    MLSpectralClusteringParams,
    fit_predict,
    fit_predict_connectivity,
)
from spectral.derived.sparse.coo import CooGraph


def _config(
    n_clusters: Int,
    n_components: Int,
    n_init: Int,
    n_neighbors: Int,
    eigen_tol: Float32,
    seed: UInt64,
) raises -> MLSpectralClusteringParams:
    """`ML::SpectralClustering::params` (`spectral_clustering.hpp:19-32`),
    with the checks that would otherwise surface as a confusing error from
    three layers down.

    `n_init` and `n_neighbors` are checked here; `n_clusters` is checked by
    `fit_predict_graph`, `n_components` by the Lanczos entry (`1 <= k < n`
    and `n - k > 0`).

    **DEVIATION 890: a non-positive `eigen_tol` is REFUSED BY NAME.** The
    field itself is cuVS's plumbed `tolerance`, verbatim theirs and not a
    choice of ours (contract section 4, C3 STRUCK). The refusal is ours, and
    it is here because cuML 26.08's Python maps `eigen_tol='auto'` to
    `0.0` (`spectral_clustering.pyx:346-347`) and nothing downstream turns
    that back into a default: `cuvs::cluster::spectral::detail::fit_predict`
    assigns `spectral_embedding_config.tolerance = config.tolerance`
    unconditionally, so the `1e-5` struct default at
    `preprocessing/spectral_embedding.hpp:59` is OVERWRITTEN WITH ZERO.
    A zero tolerance disables the convergence test (`lanczos_smallest`'s
    loop condition is `res > tol AND iter < maxIter`), so the solve runs to
    `max_iterations = 10 * n_samples` and an exactly converged problem walks
    into DEVIATION 774's restart-breakdown refusal. Refusing names the
    choice; silently substituting `1e-5` would run a number the caller did
    not ask for. `python/mojolearn/_spectral_impl.py` carries the same
    refusal where a caller can read it, and defaults to `1e-5`."""
    if n_init < 1:
        raise Error(
            "spectral clustering: n_init must be at least 1, got "
            + String(n_init)
        )
    if n_neighbors < 1:
        raise Error(
            "spectral clustering: n_neighbors must be at least 1, got "
            + String(n_neighbors)
        )
    if not (eigen_tol > Float32(0.0)):
        raise Error(
            "spectral clustering: eigen_tol must be positive, got "
            + String(eigen_tol)
        )
    return MLSpectralClusteringParams(
        n_clusters=n_clusters,
        n_components=n_components,
        n_init=n_init,
        n_neighbors=n_neighbors,
        eigen_tol=eigen_tol,
        seed=seed,
    )


def spectral_fit_predict_dataset_host(
    dataset: List[Float32],
    n_samples: Int,
    n_features: Int,
    n_clusters: Int,
    n_components: Int,
    n_init: Int,
    n_neighbors: Int,
    eigen_tol: Float32,
    seed: UInt64,
    mut labels: List[Int32],
    mut embedding: List[Float32],
) raises -> Int:
    """`ML::SpectralClustering::fit_predict(handle, config, dataset,
    labels)` (`spectral_clustering.cu:35-41`), which is cuVS's dataset
    overload (`cluster/detail/spectral.cuh:64-80`): the kNN connectivity
    graph, then the embedding, then k-means.

    `dataset` is row-major `n_samples x n_features`. `labels` comes back
    with `n_samples` ids in `[0, n_clusters)`; `embedding` comes back
    row-major `n_samples x n_out` and the return value is `n_out`.

    `n_out == n_components` on this path, not `n_components - 1`: cuVS's
    clustering overload sets `drop_first = false` (`:36`) and
    `norm_laplacian = true`, so the trivial eigenvector is KEPT and k-means
    runs on all `n_components` columns. That is theirs, not a choice, and it
    is why this embedding is not the same object as a `SpectralEmbedding`
    transform, which drops it."""
    if n_samples <= 0 or n_features <= 0:
        raise Error(
            "spectral clustering: X must be n_samples x n_features with both"
            " positive, got " + String(n_samples) + " x " + String(n_features)
        )
    if len(dataset) < n_samples * n_features:
        raise Error(
            "spectral clustering: X holds " + String(len(dataset))
            + " floats, needs " + String(n_samples * n_features)
        )
    var config = _config(
        n_clusters, n_components, n_init, n_neighbors, eigen_tol, seed
    )
    var ctx = DeviceContext()
    var trace = IdentityTrace()
    trace.header(
        "spectral clustering (dataset): n_samples=" + String(n_samples)
        + " n_features=" + String(n_features)
        + " n_clusters=" + String(n_clusters)
        + " n_components=" + String(n_components)
        + " n_init=" + String(n_init)
        + " n_neighbors=" + String(n_neighbors)
        + " eigen_tol=" + String(eigen_tol)
        + " seed=" + String(seed)
    )
    fit_predict(
        ctx, config, dataset, n_samples, n_features, labels, embedding, trace
    )
    return len(embedding) // n_samples


def spectral_fit_predict_graph_host(
    rows: List[Int32],
    cols: List[Int32],
    vals: List[Float32],
    n_samples: Int,
    n_clusters: Int,
    n_components: Int,
    n_init: Int,
    n_neighbors: Int,
    eigen_tol: Float32,
    seed: UInt64,
    mut labels: List[Int32],
    mut embedding: List[Float32],
) raises -> Int:
    """`ML::SpectralClustering::fit_predict(handle, config, rows, cols,
    vals, labels)` (`spectral_clustering.cu:51-64`), cuVS's graph overload
    (`cluster/detail/spectral.cuh:17-62`): the affinity graph is GIVEN, so
    no kNN runs and `n_neighbors` is carried in the config and read by
    nobody on this path.

    `(rows, cols, vals)` is one COO triple per nonzero over `n_samples x
    n_samples`. It need not be sorted -- `compute_graph_laplacian` sorts it,
    inserts a zero diagonal where a row lacks one, and then refuses a
    repeated `(row, col)` key by name (DEVIATIONS 775/777; RAFT's
    `coo_reduce_duplicates`, which would SUM them instead, is not ported).
    A non-finite or negative value is refused by name by `transform_graph`.

    Returns `n_out`, the number of embedding columns."""
    if n_samples <= 0:
        raise Error(
            "spectral clustering: n_samples must be positive, got "
            + String(n_samples)
        )
    var nnz = len(vals)
    if nnz <= 0:
        raise Error(
            "spectral clustering: the connectivity graph has no entries"
        )
    if len(rows) != nnz or len(cols) != nnz:
        raise Error(
            "spectral clustering: rows, cols and vals must be the same"
            " length, got " + String(len(rows)) + ", " + String(len(cols))
            + ", " + String(nnz)
        )
    var config = _config(
        n_clusters, n_components, n_init, n_neighbors, eigen_tol, seed
    )
    var ctx = DeviceContext()
    var trace = IdentityTrace()
    trace.header(
        "spectral clustering (precomputed graph): n_samples="
        + String(n_samples) + " nnz=" + String(nnz)
        + " n_clusters=" + String(n_clusters)
        + " n_components=" + String(n_components)
        + " n_init=" + String(n_init)
        + " eigen_tol=" + String(eigen_tol)
        + " seed=" + String(seed)
    )
    var r = rows.copy()
    var c = cols.copy()
    var v = vals.copy()
    var graph = CooGraph(n_samples, r^, c^, v^)
    fit_predict_connectivity(ctx, config, graph, labels, embedding, trace)
    return len(embedding) // n_samples

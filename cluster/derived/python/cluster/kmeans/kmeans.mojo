# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""The Python surface, mirrored so the names a user types are cuVS's names.

PORT OF `cuvs/python/cuvs/cuvs/cluster/kmeans/kmeans.pyx` at cuVS
`94c2819`. Partial. Do not improve.

There are no Python bindings in this tree yet, so this file is not a binding.
It is the SHAPE of one, written in Mojo, and it exists for the reason the
rest of the mirror exists: when bindings do get written, "what should the
Python API be" must not be a fresh design question with our taste in it. It
is `cuvs.cluster.kmeans`, and the answer is already decided.

Their public surface is exactly four names
(`python/cuvs/cuvs/cluster/kmeans/__init__.py`):

    KMeansParams, fit, predict, cluster_cost

`KMeansParams` there is a `cdef class` wrapping the C++ `params` struct with
one read-only property per field, which `cluster/kmeans_params.mojo` already
transliterates. What this file adds is the two things the pyx does that the
C++ layer does not:

1. **`fit` returns `(centroids, inertia, n_iter)` as a tuple**, so inertia is
   not an out-parameter. The Mojo counterpart is `FitResult` plus the
   centroids buffer the caller owns.
2. **The pyx allocates the output centroids for you** when `centroids=None`
   (`kmeans.pyx:229-231`), so a Python caller never sizes the buffer. Here the
   caller owns and sizes every buffer, which is the single biggest difference
   a Python user would notice. Note their pyx does NOT accept host arrays: it
   calls `_check_input_array` on a wrapped CUDA-array-interface object
   (`kmeans.pyx:218-219`), so device residency is theirs too, not a
   simplification of ours.

`cluster_cost` is theirs and is worth keeping visible: it is inertia for an
arbitrary centroid set, which is how you score a model you did not fit, and
it is the function a validation harness against scikit-learn actually calls.
"""

from cluster.derived.cluster.detail.kmeans import FitResult
from cluster.derived.cluster.kmeans_params import (
    INIT_ARRAY,
    INIT_KMEANS_PLUS_PLUS,
    INIT_RANDOM,
    KMeansParams,
)


def kmeans_params_from_python(
    n_clusters: Int = 8,
    init_method: String = String("KMeansPlusPlus"),
    max_iter: Int = 300,
    tol: Float64 = 1e-4,
    n_init: Int = 1,
    oversampling_factor: Float64 = 2.0,
    batch_samples: Int = 1 << 15,
    batch_centroids: Int = 0,
    seed: UInt64 = 0,
) raises -> KMeansParams:
    """`KMeansParams.__init__` (`kmeans.pyx:98-129`), keyword-only.

    Their `__init__` is keyword-only (`def __init__(self, *, ...)`), which is
    a deliberate API choice: no caller can pass `n_clusters` positionally and
    then be wrong about the second argument. Kept.

    Their keyword list is `metric, n_clusters, init_method, max_iter, tol,
    n_init, oversampling_factor, hierarchical, hierarchical_n_iters` and
    NOTHING else: `batch_samples`, `batch_centroids`, `inertia_check` and the
    seed are C++-only fields with no Python setter. `batch_samples`,
    `batch_centroids` and `seed` are kept here because this tree's fit takes
    them; `hierarchical`/`hierarchical_n_iters` select their balanced k-means
    (`kmeans_balanced.cuh`), which is NOT PORTED, so they are absent rather
    than accepted and ignored.
    """
    var p = KMeansParams.default()
    p.n_clusters = n_clusters
    p.max_iter = max_iter
    p.tol = tol
    p.n_init = n_init
    p.oversampling_factor = oversampling_factor
    p.batch_samples = batch_samples
    p.batch_centroids = batch_centroids
    p.seed = seed

    if init_method == String("KMeansPlusPlus"):
        p.init = INIT_KMEANS_PLUS_PLUS
    elif init_method == String("Random"):
        p.init = INIT_RANDOM
    elif init_method == String("Array"):
        p.init = INIT_ARRAY
    else:
        raise Error("unknown init method: " + init_method)

    p.validate()
    return p

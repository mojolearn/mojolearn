# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python surface, shaped so the names a user types are cuVS's names.

Reference: `cuvs/python/cuvs/cuvs/cluster/kmeans/kmeans.pyx` (cuVS
`94c2819`). Partial.

There are no Python bindings in this tree yet, so this file is not a binding.
It is the SHAPE of one, written in Mojo: when bindings do get written, "what
should the Python API be" must not be a fresh design question. It
is `cuvs.cluster.kmeans`, and the answer is already decided.

The reference public surface is exactly four names
(`python/cuvs/cuvs/cluster/kmeans/__init__.py`):

    KMeansParams, fit, predict, cluster_cost

`KMeansParams` there is a `cdef class` wrapping the C++ `params` struct with
one read-only property per field, which `cluster/kmeans_params.mojo` already
covers. What this file adds is the two things the pyx does that the
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

from cluster.impl.detail.kmeans import FitResult
from cluster.impl.kmeans_params import (
    INIT_ARRAY,
    INIT_KMEANS_PLUS_PLUS,
    INIT_RANDOM,
    KMeansParams,
)


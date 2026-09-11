# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""k-means on the GPU, mirroring cuVS."""

from . import _mojolearn
from ._buffer import addr, addr_ro, as_f32_c, empty, zeros
from ._mode import NumericModeMixin

INIT_KMEANS_PLUS_PLUS = 0
INIT_RANDOM = 1
INIT_ARRAY = 2

METRIC_L2_EXPANDED = 0
METRIC_L2_SQRT_EXPANDED = 1
METRIC_COSINE_EXPANDED = 2

_INIT_NAMES = {
    "k-means++": INIT_KMEANS_PLUS_PLUS,
    "random": INIT_RANDOM,
    "array": INIT_ARRAY,
}


class KMeans(NumericModeMixin):
    """k-means, mirroring cuVS's `kmeans::fit_predict`.

    **THE DEFAULTS ARE cuVS'S, NOT scikit-learn's**, and one of them changes
    results rather than just speed:

        n_init      cuVS 1        scikit-learn 10
        max_iter    cuVS 300      scikit-learn 300
        tol         cuVS 1e-4     scikit-learn 1e-4
        init        k-means++     k-means++

    `n_init=1` means ONE restart. scikit-learn runs ten and keeps the best,
    so a like-for-like comparison must set them equal; comparing this class's
    default against scikit-learn's default compares one restart to ten and is
    not a hardware result.

    Parameters
    ----------
    n_clusters : int, default 8
    init : {'k-means++', 'random', 'array'}, default 'k-means++'
        'array' takes the starting centroids from `init_centroids`.
    n_init : int, default 1
        cuVS's default. Restarts share one seeded host RNG, so restart 2
        draws different starting centroids than restart 1; the best
        post-loop inertia wins (`tools/e2u_matrix_fit.py` measures that
        `n_init=3` moves the answer on its fixture).
    max_iter : int, default 300
    tol : float, default 1e-4
    random_state : int, default 0
        cuVS's `seed`.

    Attributes
    ----------
    cluster_centers_ : Array (n_clusters, n_features) float32
    labels_ : Array (n_samples,) int32
        The assignment against the FINAL centroids, not the last iteration's.
        A fit that returned the latter would be off by one iteration in a way
        no aggregate metric would reveal; cuVS and scikit-learn both run the
        extra pass and so does this.
    inertia_ : float
        The weighted sum of squared distances to the FINAL centroids, from
        the one fresh assignment cuVS runs after the loop
        (`detail/kmeans.cuh:516-535`); with `n_init > 1` it is the best
        restart's and decides which restart is kept. **This docstring used
        to say "0.0 MEANS NEVER COMPUTED", and that was false** (corrected
        2026-08-23, measured: a 256 x 4 fit reports 55.44). What cuVS's
        `inertia_check=False` turns off is the IN-LOOP cost, which would
        make the cost ratio a second stopping rule; the only convergence
        criterion is the centroid shift, and the post-loop inertia is
        always formed. This mirrors them.
    n_iter_ : int
    sum_scale_, weight_scale_ : float
        The fixed-point accumulator multipliers chosen for your data. Exposed
        because a wrong answer in this algorithm comes from these two, and
        reproducing a result needs them.

    NUMPY-FREE SINCE DEVIATION 2369: the two attributes above are
    `_array.Array`s where they were ndarrays; `np.asarray(model.labels_)`
    is a zero-copy view for a caller who has NumPy.
    """

    #: This family's binding, for `NumericModeMixin._bind`.
    _BINDING = "_mojolearn"

    def __init__(
        self,
        n_clusters=8,
        init="k-means++",
        n_init=1,
        max_iter=300,
        tol=1e-4,
        random_state=0,
        init_centroids=None,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init_centroids = init_centroids

    def fit(self, X, y=None, sample_weight=None):
        """Fit, and set `labels_` from a pass against the final centroids."""
        if isinstance(self.init, str):
            if self.init not in _INIT_NAMES:
                raise ValueError(
                    f"mojolearn: init must be one of "
                    f"{sorted(_INIT_NAMES)}, got {self.init!r}"
                )
            init_code = _INIT_NAMES[self.init]
        else:
            init_code = int(self.init)

        x, _ = as_f32_c(X, ndim=2, name="X")
        n, d = x.shape
        if self.n_clusters > n:
            raise ValueError(
                f"mojolearn: n_clusters={self.n_clusters} exceeds "
                f"n_samples={n}"
            )

        if init_code == INIT_ARRAY:
            if self.init_centroids is None:
                raise ValueError(
                    "mojolearn: init='array' needs init_centroids"
                )
            c0, _ = as_f32_c(self.init_centroids, ndim=2,
                             name="init_centroids")
            if c0.shape != (self.n_clusters, d):
                raise ValueError(
                    f"mojolearn: init_centroids must be "
                    f"({self.n_clusters}, {d}), got {c0.shape}"
                )
            # A COPY, on purpose: the kernel writes the centroids in place
            # and `c0` may be a zero-copy borrow of the caller's array.
            centers = c0.copy()
        else:
            centers = zeros((self.n_clusters, d), "<f4")
        # DEVIATION 2672: ALLOCATED AS THE ATTRIBUTE'S OWN DTYPE. The kernel
        # writes uint32 cluster ids and every id is below 2**31, so int32 is
        # the same bytes; allocating int32 here lets the kernel write the
        # array the caller keeps. It used to be `<u4` and `labels_` was then
        # `frombytes(labels.tobytes(), "<i4", (n,))`, two host copies of the
        # label vector per fit (16 MB each at 4,000,000 rows).
        labels = empty((n,), "<i4")

        if sample_weight is None:
            n_weights = 0
            # Never read when n_weights is 0. Passing X's address avoids
            # allocating an array of ones the Mojo side would ignore.
            w = x
        else:
            # DEVIATION 2369: a 1-D vector by contract (`ndim=1`); the
            # NumPy-era `.ravel()` also accepted an (n, 1) column.
            w, _ = as_f32_c(sample_weight, ndim=1, name="sample_weight")
            if w.shape[0] != n:
                raise ValueError(
                    f"mojolearn: sample_weight has {w.shape[0]} entries, X "
                    f"has {n} rows"
                )
            n_weights = n

        inertia, n_iter, sum_scale, weight_scale = self._bind("_mojolearn").kmeans_fit(
            addr_ro(x, name="X"),
            addr(centers, name="cluster_centers_"),
            addr(labels, name="labels_"),
            addr_ro(w, name="sample_weight"),
            # ORDER MATCHES bindings/_mojolearn.mojo::kmeans_fit_binding.
            # n_samples, n_features, n_clusters, n_weights, max_iter,
            # tol, seed, n_init, init, metric
            [
                n, d, self.n_clusters, n_weights, self.max_iter,
                float(self.tol), int(self.random_state), self.n_init,
                init_code, METRIC_L2_EXPANDED,
            ],
        )

        self.cluster_centers_ = centers
        # The kernel writes uint32 cluster ids; scikit-learn's attribute is
        # signed. Every id is below 2**31, so int32 is the SAME BYTES, and
        # since DEVIATION 2672 the array the kernel wrote IS that int32
        # array (it was `<u4` plus `frombytes(labels.tobytes(), "<i4")`,
        # two host copies per fit; before that a `labels.astype(np.int32)`
        # Python loop, DEVIATION 2369).
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = n_iter
        self.sum_scale_ = sum_scale
        self.weight_scale_ = weight_scale
        self.n_features_in_ = d
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        return self.fit(X, sample_weight=sample_weight).labels_

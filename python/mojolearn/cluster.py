"""k-means on the GPU, mirroring cuVS."""

import numpy as np

from . import _mojolearn
from ._arrays import _addr, _addr_ro, as_f32_c

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


class KMeans:
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
    cluster_centers_ : ndarray (n_clusters, n_features)
    labels_ : ndarray (n_samples,)
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
    """

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

        x, _ = as_f32_c(X, "X")
        n, d = x.shape
        if self.n_clusters > n:
            raise ValueError(
                f"mojolearn: n_clusters={self.n_clusters} exceeds "
                f"n_samples={n}"
            )

        centers = np.zeros((self.n_clusters, d), dtype=np.float32)
        if init_code == INIT_ARRAY:
            if self.init_centroids is None:
                raise ValueError(
                    "mojolearn: init='array' needs init_centroids"
                )
            c0, _ = as_f32_c(self.init_centroids, "init_centroids")
            if c0.shape != (self.n_clusters, d):
                raise ValueError(
                    f"mojolearn: init_centroids must be "
                    f"({self.n_clusters}, {d}), got {c0.shape}"
                )
            centers[:] = c0
        labels = np.empty(n, dtype=np.uint32)

        if sample_weight is None:
            n_weights = 0
            # Never read when n_weights is 0. Passing X's address avoids
            # allocating an array of ones the Mojo side would ignore.
            w = x
        else:
            w = np.ascontiguousarray(sample_weight, dtype=np.float32).ravel()
            if w.size != n:
                raise ValueError(
                    f"mojolearn: sample_weight has {w.size} entries, X has "
                    f"{n} rows"
                )
            n_weights = n

        inertia, n_iter, sum_scale, weight_scale = _mojolearn.kmeans_fit(
            _addr_ro(x),
            _addr(centers),
            _addr(labels),
            _addr_ro(w),
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
        self.labels_ = labels.astype(np.int32)
        self.inertia_ = inertia
        self.n_iter_ = n_iter
        self.sum_scale_ = sum_scale
        self.weight_scale_ = weight_scale
        self.n_features_in_ = d
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        return self.fit(X, sample_weight=sample_weight).labels_

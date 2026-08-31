# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Spectral clustering on the GPU, mirroring cuML's `SpectralClustering`.

The estimator is `SpectralClustering`. It is backed by `spectral/`, which
ports cuML 26.08's `ML::SpectralClustering::fit_predict` down through
cuVS's `cluster::spectral::detail::fit_predict` (the kNN connectivity
graph, the normalized graph Laplacian, RAFT's thick-restart Lanczos, and
`cluster/`'s already-ported k-means) with every closed vendor library along
that path replaced by a named, numbered stand-in (DEVIATIONS 770-781).

**THIS LANE HAS RUN ON ONE APPLE M4 AND NOWHERE ELSE.** Read that before
reading anything else here. `spectral/IDENTICAL_SPECTRAL_CONTRACT.md`
section 10 says it in the lane's own words -- "no cross-vendor result of
any kind (nothing has run anywhere but one M4)" -- and the artifacts agree:
`spectral` is not one of the lanes `tools/e1_bootstrap.sh` phase 8 runs, it
has no card in either leg-11 vendor directory, and
`tools/e3_round_judge.sh` section 7 does not name it. The class docstring
below says what the profile PINS; a pinned convention is not a measured
result, and nothing in this module claims one.

The binding lives in `bindings/_mojolearn_metrics.mojo` alongside the
metrics functions and is loaded through `_metrics_impl._get_binding`.
"""

import numpy as np

from ._arrays import _addr, _addr_ro, as_f32_c
from ._metrics_impl import _get_binding

__all__ = ["SpectralClustering"]

_AFFINITIES = ("nearest_neighbors", "precomputed")

#: cuVS's own struct default for the eigensolver tolerance
#: (`cuvs/preprocessing/spectral_embedding.hpp:59`, `tolerance{1e-5f}`).
DEFAULT_EIGEN_TOL = 1e-5

#: cuVS's own struct default seed (`cuvs/cluster/spectral.hpp`,
#: `raft::random::RngState rng_state{0}`).
DEFAULT_SEED = 0


def _coo_triples(A):
    """A precomputed affinity matrix as `(rows, cols, vals, n)` int32/float32.

    Mirrors cuML's `spectral_clustering.pyx:306-312`, which accepts scipy or
    cupy sparse in COO/CSR/CSC and calls `sp.coo_matrix(X)` on a dense
    input. Converting a dense matrix through COO DROPS EXACT ZEROS, which is
    theirs and is also what makes a dense input usable at all.
    """
    try:
        import scipy.sparse as sp
    except ImportError:
        sp = None
    if sp is not None and sp.issparse(A):
        coo = A.tocoo()
        n = coo.shape[0]
        if coo.shape[0] != coo.shape[1]:
            raise ValueError(
                "mojolearn SpectralClustering: a precomputed affinity matrix "
                f"must be square, got shape {coo.shape}"
            )
        rows = np.ascontiguousarray(coo.row, dtype=np.int32)
        cols = np.ascontiguousarray(coo.col, dtype=np.int32)
        vals = np.ascontiguousarray(coo.data, dtype=np.float32)
        return rows, cols, vals, int(n)
    dense = np.asarray(A)
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        raise ValueError(
            "mojolearn SpectralClustering: with affinity='precomputed', X "
            "must be a square affinity matrix or a scipy sparse matrix, got "
            f"shape {dense.shape}"
        )
    nz = np.nonzero(dense)
    rows = np.ascontiguousarray(nz[0], dtype=np.int32)
    cols = np.ascontiguousarray(nz[1], dtype=np.int32)
    vals = np.ascontiguousarray(dense[nz], dtype=np.float32)
    return rows, cols, vals, int(dense.shape[0])


class SpectralClustering:
    """Spectral clustering, mirroring cuML 26.08's `SpectralClustering`.

    The pipeline is cuVS's: build a kNN connectivity graph (or take a
    precomputed affinity matrix), form the NORMALIZED graph Laplacian,
    negate it, take the `n_components` largest algebraic eigenpairs of the
    negation with a thick-restart Lanczos, and run k-means on the resulting
    embedding.

    WHAT IS PINNED, AND WHAT IS ONLY MEASURED ON ONE MACHINE
    --------------------------------------------------------
    An eigenproblem has no unique answer. If `v` is a unit eigenvector then
    so is `-v`; if two eigenvalues are equal then every orthonormal basis of
    their shared subspace is equally correct. RAFT pins neither convention,
    because RAFT ships one backend and has never had to. This port pins
    both, and `spectral/IDENTICAL_SPECTRAL_CONTRACT.md` (profile
    `mojolearn.identical.spectral.fp32.v1`) is where they live.

    **THE SIGN RULE (DEVIATION 770).** After the projected eigenproblem is
    solved and its eigenvalues sorted, every column of the `ncv x ncv`
    PROJECTED eigenvector matrix is negated if and only if its first
    nonzero component in ascending row index is negative. "Nonzero" is
    `x != 0.0`, which is false for both `+0.0` and `-0.0`, so a leading
    signed zero is SKIPPED rather than consulted for its sign bit; an
    all-zero column is left alone. **The rule is applied there and nowhere
    else. THERE IS NO SIGN PIN ON THE EMBEDDING THIS CLASS RETURNS.** The
    embedding's signs follow from the pinned start vector (DEVIATION 772, a
    splitmix64 hashed uniform that is an exact function of `(seed, n)`), not
    from a second rule. Adding one would be a different profile.

    **THE ORDERING AND TIE RULES (DEVIATION 778).** The projected solver
    returns its eigenvalues ASCENDING, eigenvector `c` in column `c`, which
    is cuSOLVER `syevd`'s convention and what RAFT's slicing assumes. This
    path uses `which = LA` on the NEGATED Laplacian and takes the LAST `k`,
    so index `k-1` is the Laplacian's smallest eigenvalue; the reversed
    gather then puts the Laplacian's second-smallest in embedding column 0.
    Ties are broken by ORIGINAL INDEX: the ascending sort is an insertion
    sort with the strict comparison `d[order[j]] > d[key]`, so equal
    eigenvalues keep the order they had on the Jacobi's diagonal, which is
    itself a pure function of the input bits and the pinned sweep schedule.
    `+0.0` and `-0.0` compare EQUAL, so two zero eigenvalues of opposite
    sign are ordered by index and not by sign bit, deliberately. The `LM`
    and `SM` solver modes are REFUSED by name rather than ported, because
    theirs sorts by magnitude with `thrust::sort`, which is not stable, and
    porting it would mean inventing a tie-break.

    **WHAT THE PROFILE CLAIMS ABOUT DEGENERACY, exactly.** Given the same
    input bits and the same profile, the same eigenvector bits come out,
    INCLUDING inside a degenerate subspace, because the Jacobi sweep that
    produced them is the same deterministic sequence of rotations. It does
    NOT claim that the basis inside a degenerate subspace is STABLE under
    perturbation, and it is not: a one-ulp change in one Laplacian entry can
    rotate a degenerate pair's basis arbitrarily. A graph with `c` connected
    components has eigenvalue zero with multiplicity `c`, so a
    `n_components <= c` embedding of such a graph lies entirely inside a
    degenerate subspace. That is a property of spectral embedding, not of
    this port, and it is the shape most likely to surprise you.

    **WHAT COVERAGE THAT SITS ON, stated as narrowly as the evidence.**

      * NO CROSS-VENDOR RESULT OF ANY KIND. Every gate in this lane ran on
        one Apple M4 on 2026-08-23. `spectral` is not in
        `tools/e1_bootstrap.sh` phase 8, has no card in either leg-11
        vendor lane directory, and is not judged by
        `tools/e3_round_judge.sh` section 7. The metrics functions in
        `mojolearn`'s sibling module ARE certified on three vendors; this
        class is not, and the two must not be read together.
      * THE DEVICE-SIDE INSTRUMENT FOR THE SIGN RULE RESTS ON ONE FIXTURE.
        The `SIGN_FLIP` sabotage bites on the `hashed` fixture and is inert
        on `path`, `ring`, `hashed_unnorm` and `blobs`, because after a
        restart the re-canonicalization absorbs the flip on those four
        (`CARD_GAPS.md:78`; the contract's section 9 table). The rule has
        two host instruments the sabotage does not target, so the clause is
        not undefended, but the DEVICE path's coverage is one fixture wide.
      * A WHOLE-COLUMN SIGN DISAGREEMENT BETWEEN TWO VENDORS WOULD BE
        INVISIBLE TO THE CARD. `spectral.ritz.vectors` is recorded AFTER
        the sign pin is applied, so the pin re-canonicalizes such a
        divergence away before it is hashed (`CARD_GAPS.md`, "Three more
        cards that hash on the far side of a washer"). That is an open gap,
        not a solved problem.
      * SEAM J4's CERTIFICATE GATE IS OWED. The fused `ROTATE` (DEVIATION
        781) lives in a solver SHARED by the device arm and the oracle, so
        a device-equals-oracle compare cannot see a change to it. Its
        sabotage does bite, on the ring fixture's degenerate pair, but the
        hash-against-a-literal check the lane wants is written down and not
        done (`spectral/NOT_IMPLEMENTED.tsv`, "this lane's own gaps").

    **ON DEVIATION 780, so nobody re-inflates it.** That deviation once
    claimed five constants as this port's own: `ncv = min(n - k, max(2k+1,
    20))`, `max_iterations = 10 * n_samples`, the plumbed `tolerance`, the
    Jacobi sweep cap, and the `ncv` admissibility bound. THREE OF THE FIVE
    WERE STRUCK on 2026-08-23. They are VERBATIM cuVS 26.08, down to the
    message string of the `RAFT_EXPECTS`; the claim was made while only a
    cuVS 25.08 checkout existed on the machine, and 25.08 spells them as
    literals. What remains ours is the host Jacobi's 60-sweep cap and this
    lane's own `ncv` admissibility guard, both in code that stands where a
    CLOSED vendor library does. Reading the wrong tree invents ORIGINALITY
    a port does not have, which is exactly as bad as missing a real
    deviation.

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY
    -----------------------------------------
        n_clusters     honored   cuML's default 8. Must be in
                                 [1, n_samples].
        n_components   honored   None (the default) means `n_clusters`,
                                 which is cuML's rule. The Lanczos requires
                                 `1 <= n_components < n_samples` and
                                 `n_samples - n_components > 0`, and
                                 refuses otherwise by name.
        n_neighbors    honored   cuML's default 10. Ignored with
                                 affinity='precomputed', as theirs is.
        n_init         honored   cuML's default 10 (k-means restarts). NOTE
                                 that `mojolearn.KMeans` defaults to cuVS's
                                 1; this path takes cuVS's clustering
                                 default of 10, which is what
                                 `cluster/detail/spectral.cuh` passes.
        eigen_tol      honored   a positive float; the default 1e-5 is
                                 cuVS's own struct default
                                 (`spectral_embedding.hpp:59`).
                                 **'auto' IS REFUSED BY NAME (DEVIATION
                                 890).** cuML 26.08's Python maps 'auto' to
                                 0.0 (`spectral_clustering.pyx:346-347`),
                                 and 0.0 does not select a default anywhere
                                 downstream -- it is passed through as the
                                 tolerance, which disables the convergence
                                 test entirely (the loop condition is
                                 `res > tol`), runs to `max_iterations =
                                 10 * n_samples`, and drives an exactly
                                 converged problem into DEVIATION 774's
                                 restart-breakdown refusal. Substituting
                                 1e-5 silently would be a value cuML does
                                 not use; refusing names the choice.
        random_state   honored   an int in [0, 2**32) as cuML requires.
                                 **None means SEED 0 (DEVIATION 891)**, not
                                 a fresh entropy draw: cuVS's struct
                                 default is `rng_state{0}` and the no-seed
                                 arm of the Lanczos start vector is refused
                                 outright (DEVIATION 772, because
                                 `std::random_device` is not reproducible).
                                 So `random_state=None` here is
                                 DETERMINISTIC, which is neither
                                 scikit-learn's meaning nor cuML's Python
                                 one, and this line is why.
        affinity       honored   'nearest_neighbors' (default) or
                                 'precomputed'. 'rbf',
                                 'precomputed_nearest_neighbors',
                                 'nearest_neighbors' with a callable, and
                                 every scikit-learn kernel name are
                                 REFUSED by name: cuML's own surface
                                 accepts exactly these two.
        assign_labels  honored   'kmeans' only; 'discretize' and
                                 'cluster_qr' are refused by name, as
                                 cuML's `spectral_clustering.pyx:165-167`
                                 refuses them.
        eigen_solver   refused   scikit-learn's choice of arpack / lobpcg /
                                 amg. There is one solver here, RAFT's
                                 thick-restart Lanczos, and it is the only
                                 thing the profile pins.
        gamma, degree, refused   parameters of the RBF and polynomial
        coef0,                   affinities, which are not in cuML's
        kernel_params            affinity set at all.
        n_jobs, verbose refused  no host thread pool and nothing prints.
        affinity_matrix_ absent  cuVS builds the connectivity graph inside
                                 the fit and does not hand it back; this
                                 binding has no arm that returns it.
        predict()      absent    spectral clustering is transductive.
                                 scikit-learn has no `predict` either.

    NORMALIZATION IS NOT A PARAMETER HERE, and that is theirs.
    `cluster/detail/spectral.cuh:35-36` hard-codes `norm_laplacian = true`
    and `drop_first = false` on the clustering path. So the trivial
    (roughly constant) eigenvector is KEPT and k-means runs on all
    `n_components` columns. `embedding_` is therefore not the same object a
    spectral EMBEDDING transform returns, which drops it.

    Attributes
    ----------
    labels_ : ndarray (n_samples,) int32
        Cluster ids in `[0, n_clusters)`, k-means's numbering.
    embedding_ : ndarray (n_samples, n_components) float32
        The row-major spectral embedding k-means was run on. Exposed
        because the lane's gates read it and because it is the thing every
        clause above is about.
    n_features_in_ : int
        Only set for affinity='nearest_neighbors'.
    """

    def __init__(
        self,
        n_clusters=8,
        *,
        n_components=None,
        random_state=None,
        n_neighbors=10,
        n_init=10,
        eigen_tol=DEFAULT_EIGEN_TOL,
        affinity="nearest_neighbors",
        assign_labels="kmeans",
        eigen_solver=None,
        gamma=None,
        degree=None,
        coef0=None,
        kernel_params=None,
        n_jobs=None,
        verbose=False,
    ):
        if affinity not in _AFFINITIES:
            raise ValueError(
                f"mojolearn SpectralClustering: affinity={affinity!r} is "
                f"refused; it must be one of {list(_AFFINITIES)}. cuML's own "
                "surface accepts exactly these two, and 'rbf' / "
                "'precomputed_nearest_neighbors' / a callable are different "
                "affinity constructions that nothing here ports."
            )
        if assign_labels != "kmeans":
            raise NotImplementedError(
                f"mojolearn SpectralClustering: assign_labels="
                f"{assign_labels!r} is refused; the k-means arm is the only "
                "one cuVS has (cuML's spectral_clustering.pyx:165-167 "
                "refuses the same)"
            )
        if eigen_solver is not None:
            raise NotImplementedError(
                f"mojolearn SpectralClustering: eigen_solver={eigen_solver!r} "
                "is refused; there is one solver here, RAFT's thick-restart "
                "Lanczos, and it is the one the identity profile pins"
            )
        for name, value in (
            ("gamma", gamma),
            ("degree", degree),
            ("coef0", coef0),
            ("kernel_params", kernel_params),
        ):
            if value is not None:
                raise NotImplementedError(
                    f"mojolearn SpectralClustering: {name} is refused; it "
                    "parameterizes an RBF or polynomial affinity, and "
                    "affinity is restricted to 'nearest_neighbors' and "
                    "'precomputed'"
                )
        if n_jobs is not None:
            raise NotImplementedError(
                "mojolearn SpectralClustering: n_jobs is refused; the work "
                "is on the GPU and there is no host thread pool to size"
            )
        if verbose:
            raise NotImplementedError(
                "mojolearn SpectralClustering: verbose is refused; nothing "
                "in this path prints. Set MOJOLEARN_IDENTITY_TRACE=<path> "
                "for the identity card instead"
            )
        if isinstance(eigen_tol, str):
            raise NotImplementedError(
                f"mojolearn SpectralClustering: eigen_tol={eigen_tol!r} is "
                "refused (DEVIATION 890). cuML 26.08's Python maps 'auto' to "
                "0.0, which is passed straight through as the eigensolver "
                "tolerance and disables the convergence test (the loop "
                "condition is res > tol), so the solve runs to "
                "max_iterations = 10 * n_samples and an exactly converged "
                "problem hits the restart-breakdown refusal. Pass a positive "
                f"float; {DEFAULT_EIGEN_TOL} is cuVS's own struct default "
                "and is this class's default."
            )
        eigen_tol = float(eigen_tol)
        if not (eigen_tol > 0.0) or eigen_tol != eigen_tol:
            raise ValueError(
                "mojolearn SpectralClustering: eigen_tol must be positive, "
                f"got {eigen_tol!r} (DEVIATION 890; see the 'auto' note)"
            )
        if random_state is None:
            seed = DEFAULT_SEED
        else:
            seed = int(random_state)
            if seed < 0 or seed >= 2**32:
                raise ValueError(
                    "mojolearn SpectralClustering: random_state must satisfy "
                    f"0 <= random_state < 2**32, got {random_state!r} (cuML's "
                    "check_random_seed refuses the same range)"
                )
        if int(n_clusters) < 1:
            raise ValueError(
                "mojolearn SpectralClustering: n_clusters must be at least 1"
            )
        if int(n_init) < 1:
            raise ValueError(
                "mojolearn SpectralClustering: n_init must be at least 1"
            )
        if int(n_neighbors) < 1:
            raise ValueError(
                "mojolearn SpectralClustering: n_neighbors must be at least 1"
            )
        if n_components is not None and int(n_components) < 1:
            raise ValueError(
                "mojolearn SpectralClustering: n_components must be at least "
                "1, or None to follow n_clusters"
            )
        self.n_clusters = int(n_clusters)
        self.n_components = n_components
        self.random_state = random_state
        self.n_neighbors = int(n_neighbors)
        self.n_init = int(n_init)
        self.eigen_tol = eigen_tol
        self.affinity = affinity
        self.assign_labels = "kmeans"
        self._seed = seed

    def _n_components(self):
        """cuML's rule: `n_components` defaults to `n_clusters` when None
        (`spectral_clustering.pyx`)."""
        return self.n_clusters if self.n_components is None else int(
            self.n_components
        )

    def fit(self, X, y=None):
        k = self._n_components()
        labels_out = None
        if self.affinity == "precomputed":
            rows, cols, vals, n = _coo_triples(X)
            if vals.size == 0:
                raise ValueError(
                    "mojolearn SpectralClustering: the precomputed affinity "
                    "matrix has no nonzero entries"
                )
            if not np.isfinite(vals).all():
                raise ValueError(
                    "mojolearn SpectralClustering: the precomputed affinity "
                    "matrix has a non-finite entry (the ported path refuses "
                    "it by name, because sqrt of a non-finite degree is a "
                    "NaN and no NaN may reach a recorded value)"
                )
            if (vals < 0).any():
                raise ValueError(
                    "mojolearn SpectralClustering: the precomputed affinity "
                    "matrix has a negative entry (refused by name: sqrt of a "
                    "negative degree is a NaN in cuVS too)"
                )
            self._check_shape(n, k)
            labels = np.empty(n, dtype=np.int32)
            embedding = np.empty((n, k), dtype=np.float32)
            # ORDER MATCHES bindings/_mojolearn_metrics.mojo::
            # spectral_fit_predict_graph_binding.
            # n_samples, nnz, n_clusters, n_components, n_init, n_neighbors,
            # eigen_tol, seed
            n_out = int(
                _get_binding().spectral_fit_predict_graph(
                    _addr_ro(rows),
                    _addr_ro(cols),
                    _addr_ro(vals),
                    _addr(labels),
                    _addr(embedding),
                    [
                        n,
                        int(vals.shape[0]),
                        self.n_clusters,
                        k,
                        self.n_init,
                        self.n_neighbors,
                        self.eigen_tol,
                        self._seed,
                    ],
                )
            )
            labels_out = labels
        else:
            x, self.input_copied_ = as_f32_c(X, "X")
            if not np.isfinite(x).all():
                raise ValueError(
                    "mojolearn SpectralClustering: X contains NaN or "
                    "infinity (the ported path refuses it by name before any "
                    "recorded stage)"
                )
            n = int(x.shape[0])
            self._check_shape(n, k)
            if self.n_neighbors > n:
                raise ValueError(
                    f"mojolearn SpectralClustering: n_neighbors="
                    f"{self.n_neighbors} exceeds n_samples={n}"
                )
            labels = np.empty(n, dtype=np.int32)
            embedding = np.empty((n, k), dtype=np.float32)
            # ORDER MATCHES bindings/_mojolearn_metrics.mojo::
            # spectral_fit_predict_dataset_binding.
            # n_samples, n_features, n_clusters, n_components, n_init,
            # n_neighbors, eigen_tol, seed
            n_out = int(
                _get_binding().spectral_fit_predict_dataset(
                    _addr_ro(x),
                    _addr(labels),
                    _addr(embedding),
                    [
                        n,
                        int(x.shape[1]),
                        self.n_clusters,
                        k,
                        self.n_init,
                        self.n_neighbors,
                        self.eigen_tol,
                        self._seed,
                    ],
                )
            )
            labels_out = labels
            self.n_features_in_ = int(x.shape[1])
        if n_out != k:
            # Cannot fire today: the clustering path hard-codes
            # `drop_first = false`, so `n_out == n_components`. The binding
            # checks the same thing before it writes anything, and this is
            # the reader-facing half of that guard.
            raise RuntimeError(
                f"mojolearn SpectralClustering: the kernel returned {n_out} "
                f"embedding columns where {k} were expected"
            )
        self.labels_ = labels_out
        self.embedding_ = embedding
        self.n_components_ = k
        return self

    def _check_shape(self, n, k):
        if self.n_clusters > n:
            raise ValueError(
                f"mojolearn SpectralClustering: n_clusters={self.n_clusters} "
                f"exceeds n_samples={n}"
            )
        if k >= n:
            raise ValueError(
                f"mojolearn SpectralClustering: n_components={k} must be "
                f"below n_samples={n}; the Lanczos requires "
                "1 <= n_components < n_samples and n_samples - n_components "
                "> 0 (cuVS's own RAFT_EXPECTS)"
            )

    def fit_predict(self, X, y=None):
        return self.fit(X, y=y).labels_

    def predict(self, X):
        raise NotImplementedError(
            "mojolearn SpectralClustering: predict() does not exist. "
            "Spectral clustering is transductive -- the labels belong to the "
            "rows that were fit -- and scikit-learn has no predict either. "
            "Use fit_predict(X)."
        )

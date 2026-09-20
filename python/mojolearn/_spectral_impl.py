# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Spectral clustering on the GPU. Reference: cuML's `SpectralClustering`.

The estimator is `SpectralClustering`. It is backed by `spectral/`, which
implements cuML 26.08's `ML::SpectralClustering::fit_predict` down through
cuVS's `cluster::spectral::detail::fit_predict` (the kNN connectivity
graph, the normalized graph Laplacian, RAFT's thick-restart Lanczos, and
`cluster/`'s already-implemented k-means) with every closed vendor library along
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

from . import _serialize
from ._array import Array
from ._buffer import (
    _native, addr, addr_ro, all_finite, as_f32_c, as_f64_c, as_i32_c, empty,
)
from ._metrics_impl import _get_binding
from .density import _check_queries
from .linear_model import _check_saved_by, _restore_mode, _saved_mode, _shape_of

__all__ = ["SpectralClustering", "SpectralEmbedding", "spectral_embedding"]

#: `SpectralClustering.save`'s format tag (lane/spectral-predict,
#: 2026-09-15): the prediction data of a `prediction_data=True` fit.
_SPECTRAL_FORMAT = "mojolearn-spectral-1"

#: The prediction data a `prediction_data=True` fit keeps.
_PREDICTION_ATTRS = ("_pd_eigenvalues", "_pd_eigenvectors", "_pd_diag", "_pd_centroids", "_fit_X")

_AFFINITIES = ("nearest_neighbors", "precomputed")

#: cuVS's own struct default for the eigensolver tolerance
#: (`cuvs/preprocessing/spectral_embedding.hpp:59`, `tolerance{1e-5f}`).
DEFAULT_EIGEN_TOL = 1e-5

#: cuVS's own struct default seed (`cuvs/cluster/spectral.hpp`,
#: `raft::random::RngState rng_state{0}`).
DEFAULT_SEED = 0


def _coo_triples(A, who="SpectralClustering"):
    """A precomputed affinity matrix as `(rows, cols, vals, n)` int32/float32.

    Same behavior as cuML's `spectral_clustering.pyx:306-312`, which accepts scipy or
    cupy sparse in COO/CSR/CSC and calls `sp.coo_matrix(X)` on a dense
    input. Converting a dense matrix through COO DROPS EXACT ZEROS, which is
    theirs and is also what makes a dense input usable at all.

    A sparse input is recognized by DUCK TYPE, not by importing SciPy: any
    object with a `tocoo()` whose result carries `shape`, `row`, `col` and
    `data` is taken as sparse, and those three arrays are read through the
    buffer protocol like every other input (DEVIATION 2489). SciPy, CuPy
    (after `.get()`), and anything else shaped like them qualify; nothing
    is imported to ask.
    """
    tocoo = getattr(A, "tocoo", None)
    if callable(tocoo):
        coo = tocoo()
        shape = tuple(coo.shape)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(
                f"mojolearn {who}: a precomputed affinity matrix "
                f"must be square, got shape {shape}"
            )
        rows, _ = as_i32_c(coo.row, ndim=1, name="rows")
        cols, _ = as_i32_c(coo.col, ndim=1, name="cols")
        vals, _ = as_f32_c(coo.data, ndim=1, name="vals")
        return rows, cols, vals, int(shape[0])
    shape = _shape_of(A)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(
            f"mojolearn {who}: with affinity='precomputed', X "
            "must be a square affinity matrix or a sparse matrix with "
            f"`tocoo()`, got shape {shape}"
        )
    # The nonzero test is made in float64, the widest dtype the boundary
    # carries, so an entry that is nonzero in a float64 input and rounds to
    # 0.0f is KEPT as an explicit zero, exactly as `np.nonzero` on the
    # source dtype kept it; the values are narrowed to float32 afterwards,
    # one rounding each. -0.0 is a zero, NaN is not (it compares unequal).
    dense, _ = as_f64_c(A, ndim=2, name="X")
    n = int(shape[0])
    # DEVIATION 2489: the scan in Mojo, two calls, count then fill into
    # buffers this side allocates. The Python loop this replaced
    # (DEVIATION 2373) lives on only as the oracle in
    # tests/test_native_nonzero.py, which holds the two to byte equality.
    nnz = int(_native("nonzero_f64_count")(addr_ro(dense, name="X"), dense.size)) if dense.size else 0
    rows = empty((nnz,), "<i4")
    cols = empty((nnz,), "<i4")
    vals = empty((nnz,), "<f4")
    if nnz == 0:
        # Nothing to write, and an empty Array has no address to hand the
        # binding (its pointer helpers refuse a null by design).
        return rows, cols, vals, n
    wrote = int(_native("nonzero_f64_fill")(
        addr_ro(dense, name="X"), n, n,
        [addr(rows, name="rows"), addr(cols, name="cols"),
         addr(vals, name="vals")],
        nnz,
    ))
    if wrote != nnz:
        raise RuntimeError(
            f"mojolearn {who}: nonzero_f64_fill wrote "
            f"{wrote} of {nnz} entries"
        )
    return rows, cols, vals, n


class SpectralClustering:
    """Spectral clustering. Reference: cuML 26.08's `SpectralClustering`.

    The pipeline (as in cuVS): build a kNN connectivity graph (or take a
    precomputed affinity matrix), form the NORMALIZED graph Laplacian,
    negate it, take the `n_components` largest algebraic eigenpairs of the
    negation with a thick-restart Lanczos, and run k-means on the resulting
    embedding.

    WHAT IS PINNED, AND WHAT IS ONLY MEASURED ON ONE MACHINE
    --------------------------------------------------------
    An eigenproblem has no unique answer. If `v` is a unit eigenvector then
    so is `-v`; if two eigenvalues are equal then every orthonormal basis of
    their shared subspace is equally correct. RAFT pins neither convention,
    because RAFT ships one backend and has never had to. This implementation pins
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
    and `SM` solver modes are REFUSED by name rather than implemented, because
    theirs sorts by magnitude with `thrust::sort`, which is not stable, and
    implementing it would mean inventing a tie-break.

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
    this implementation, and it is the shape most likely to surprise you.

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
        (`archive/plans/CARD_GAPS.md:78`; the contract's section 9 table). The rule has
        two host instruments the sabotage does not target, so the clause is
        not undefended, but the DEVICE path's coverage is one fixture wide.
      * A WHOLE-COLUMN SIGN DISAGREEMENT BETWEEN TWO VENDORS WOULD BE
        INVISIBLE TO THE CARD. `spectral.ritz.vectors` is recorded AFTER
        the sign pin is applied, so the pin re-canonicalizes such a
        divergence away before it is hashed (`archive/plans/CARD_GAPS.md`, "Three more
        cards that hash on the far side of a washer"). That is an open gap,
        not a solved problem.
      * SEAM J4's CERTIFICATE GATE IS OWED. The fused `ROTATE` (DEVIATION
        781) lives in a solver SHARED by the device arm and the oracle, so
        a device-equals-oracle compare cannot see a change to it. Its
        sabotage does bite, on the ring fixture's degenerate pair, but the
        hash-against-a-literal check the lane wants is written down and not
        done (`spectral/NOT_IMPLEMENTED.tsv`, "this lane's own gaps").

    **ON DEVIATION 780, so nobody re-inflates it.** That deviation once
    claimed five constants as this implementation's own: `ncv = min(n - k, max(2k+1,
    20))`, `max_iterations = 10 * n_samples`, the plumbed `tolerance`, the
    Jacobi sweep cap, and the `ncv` admissibility bound. THREE OF THE FIVE
    WERE STRUCK on 2026-08-23. They are IDENTICAL to cuVS 26.08, down to the
    message string of the `RAFT_EXPECTS`; the claim was made while only a
    cuVS 25.08 checkout existed on the machine, and 25.08 spells them as
    literals. What remains ours is the host Jacobi's 60-sweep cap and this
    lane's own `ncv` admissibility guard, both in code that stands where a
    CLOSED vendor library does. Reading the wrong tree invents ORIGINALITY
    an implementation does not have, which is exactly as bad as missing a real
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
        prediction_data honored  False (the default) fits exactly as before.
                                 True also copies out of the fit the Ritz
                                 values, the Ritz vectors before the degree
                                 division, the degree scaling and the final
                                 k-means centroids (and, for
                                 'nearest_neighbors', keeps the training
                                 rows), which `predict` and `save` need. The
                                 copies add no arithmetic; the fit's labels
                                 and embedding are the same bytes
        predict()      NEW       DEVIATION 2860, the Nystrom out-of-sample
                                 extension (Bengio et al., NIPS 2003) and the
                                 fit's own k-means assignment. Neither cuML
                                 nor scikit-learn has one. See `predict`

    NORMALIZATION IS NOT A PARAMETER HERE, and that is theirs.
    `cluster/detail/spectral.cuh:35-36` hard-codes `norm_laplacian = true`
    and `drop_first = false` on the clustering path. So the trivial
    (roughly constant) eigenvector is KEPT and k-means runs on all
    `n_components` columns. `embedding_` is therefore not the same object a
    spectral EMBEDDING transform returns, which drops it.

    Attributes
    ----------
    labels_ : Array (n_samples,) int32
        Cluster ids in `[0, n_clusters)`, k-means's numbering.
    embedding_ : Array (n_samples, n_components) float32
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
        prediction_data=False,
    ):
        if affinity not in _AFFINITIES:
            raise ValueError(
                f"mojolearn SpectralClustering: affinity={affinity!r} is "
                f"refused; it must be one of {list(_AFFINITIES)}. cuML's own "
                "surface accepts exactly these two, and 'rbf' / "
                "'precomputed_nearest_neighbors' / a callable are different "
                "affinity constructions that nothing here implements."
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
        self.prediction_data = prediction_data

    #: The binding `predict` asks; a host subclass answers the metrics host
    #: binding, which ships in the inference wheel.
    _BINDING = "_mojolearn_metrics"

    def _bind(self, name=None):
        return _get_binding(getattr(self, "numeric_mode", None))

    def _state_arrays(self, n, k):
        return (empty((k,), "<f4"), empty((n, k), "<f4"), empty((n,), "<f4"),
                empty((self.n_clusters, k), "<f4"))

    def _n_components(self):
        """cuML's rule: `n_components` defaults to `n_clusters` when None
        (`spectral_clustering.pyx`)."""
        return self.n_clusters if self.n_components is None else int(
            self.n_components
        )

    def fit(self, X, y=None):
        if not isinstance(self.prediction_data, bool):
            raise TypeError(
                "mojolearn SpectralClustering: prediction_data must be a bool, "
                f"got {type(self.prediction_data).__name__}"
            )
        for name in _PREDICTION_ATTRS:
            self.__dict__.pop(name, None)
        k = self._n_components()
        labels_out = None
        state = None
        if self.affinity == "precomputed":
            rows, cols, vals, n = _coo_triples(X)
            if vals.size == 0:
                raise ValueError(
                    "mojolearn SpectralClustering: the precomputed affinity "
                    "matrix has no nonzero entries"
                )
            if not all_finite(vals):
                raise ValueError(
                    "mojolearn SpectralClustering: the precomputed affinity "
                    "matrix has a non-finite entry (the implemented path refuses "
                    "it by name, because sqrt of a non-finite degree is a "
                    "NaN and no NaN may reach a recorded value)"
                )
            if vals.min() < 0:
                raise ValueError(
                    "mojolearn SpectralClustering: the precomputed affinity "
                    "matrix has a negative entry (refused by name: sqrt of a "
                    "negative degree is a NaN in cuVS too)"
                )
            self._check_shape(n, k)
            labels = empty((n,), "<i4")
            embedding = empty((n, k), "<f4")
            # ORDER MATCHES bindings/_mojolearn_metrics.mojo::
            # spectral_fit_predict_graph_binding.
            # n_samples, nnz, n_clusters, n_components, n_init, n_neighbors,
            # eigen_tol, seed
            params = [
                n,
                int(vals.shape[0]),
                self.n_clusters,
                k,
                self.n_init,
                self.n_neighbors,
                self.eigen_tol,
                self._seed,
            ]
            if self.prediction_data:
                state = self._state_arrays(n, k)
                # spectral_fit_predict_graph_state_binding: the same call,
                # plus copies of the prediction data.
                n_out = int(_get_binding().spectral_fit_predict_graph_state(
                    [addr_ro(rows, name="rows"), addr_ro(cols, name="cols"),
                     addr_ro(vals, name="vals"), addr(labels, name="labels"),
                     addr(embedding, name="embedding")]
                    + [addr(a, name="prediction data") for a in state],
                    params,
                ))
            else:
                n_out = int(
                    _get_binding().spectral_fit_predict_graph(
                        addr_ro(rows, name="rows"),
                        addr_ro(cols, name="cols"),
                        addr_ro(vals, name="vals"),
                        addr(labels, name="labels"),
                        addr(embedding, name="embedding"),
                        params,
                    )
                )
            labels_out = labels
        else:
            x, self.input_copied_ = as_f32_c(X, ndim=2, name="X")
            if not all_finite(x):
                raise ValueError(
                    "mojolearn SpectralClustering: X contains NaN or "
                    "infinity (the implemented path refuses it by name before any "
                    "recorded stage)"
                )
            n = int(x.shape[0])
            self._check_shape(n, k)
            if self.n_neighbors > n:
                raise ValueError(
                    f"mojolearn SpectralClustering: n_neighbors="
                    f"{self.n_neighbors} exceeds n_samples={n}"
                )
            labels = empty((n,), "<i4")
            embedding = empty((n, k), "<f4")
            # ORDER MATCHES bindings/_mojolearn_metrics.mojo::
            # spectral_fit_predict_dataset_binding.
            # n_samples, n_features, n_clusters, n_components, n_init,
            # n_neighbors, eigen_tol, seed
            params = [
                n,
                int(x.shape[1]),
                self.n_clusters,
                k,
                self.n_init,
                self.n_neighbors,
                self.eigen_tol,
                self._seed,
            ]
            if self.prediction_data:
                state = self._state_arrays(n, k)
                n_out = int(_get_binding().spectral_fit_predict_dataset_state(
                    [addr_ro(x, name="x"), addr(labels, name="labels"),
                     addr(embedding, name="embedding")]
                    + [addr(a, name="prediction data") for a in state],
                    params,
                ))
                # A borrowed input is copied so a caller's later write cannot
                # move what predict reads.
                self._fit_X = x if self.input_copied_ else x.copy()
            else:
                n_out = int(
                    _get_binding().spectral_fit_predict_dataset(
                        addr_ro(x, name="x"),
                        addr(labels, name="labels"),
                        addr(embedding, name="embedding"),
                        params,
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
        if state is not None:
            (self._pd_eigenvalues, self._pd_eigenvectors, self._pd_diag,
             self._pd_centroids) = state
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
        """Label NEW rows under the fitted clustering. NEW CAPABILITY
        (DEVIATION 2860): neither scikit-learn's nor cuML's
        SpectralClustering has a `predict`, and this is not either library's
        behavior.

        THE METHOD is the Nystrom out-of-sample extension of Bengio,
        Paiement, Vincent, Delalleau, Le Roux and Ouimet, "Out-of-Sample
        Extensions for LLE, Isomap, MDS, Eigenmaps, and Spectral Clustering"
        (NIPS 2003), as Fowlkes et al. (TPAMI 2004) use it for spectral
        grouping, followed by the fit's own k-means assignment. For a new row
        `x` and each embedding column `c`,

            e_c(x) = (1 / mu_c) * sum_i Ktilde(x, x_i) u_c[i] / sqrt(d(x))
            Ktilde(x, x_i) = K(x, x_i) / (sqrt(d(x)) * sqrt(d_i))

        where `u_c` is the fit's unit eigenvector before its degree division,
        `mu_c = 1 + theta_c` the normalized affinity's eigenvalue (`theta_c`
        the Ritz value of the negated normalized Laplacian), `d_i` the fit's
        degree and `d(x) = sum_i K(x, x_i)`. The last division by
        `sqrt(d(x))` is the fit's own row scaling. The row is then assigned
        to the nearest fitted k-means centroid, ties to the lowest centroid
        index.

        THE AFFINITY OF A NEW ROW.
          'nearest_neighbors'  its `n_neighbors` nearest training rows under
                               the fit's own k-NN (L2, ties by the search's
                               own order), each with weight 0.5: the fit's
                               symmetrization `0.5 * (a + b)` of an edge with
                               no reverse edge, since a new row has none. The
                               training degrees are not changed.
          'precomputed'        X is the affinity between the new rows and
                               the training rows, shape (n_new, n_train),
                               finite and non-negative (a sparse matrix is
                               densified with `toarray()`).
        Every fold (the degree, each projection) runs over the training rows
        in ascending index, seeded +0.0, one query at a time, so a row's
        label does not depend on the batch, and the GPU binding and the CPU
        host binding compute the same bytes.

        THE THRESHOLD. Nothing is dropped (the clustering fit keeps every
        column, the trivial one included). A column whose `|mu_c|` is below
        1e-3 would be amplified by more than 1000x, so predict REFUSES BY
        NAME, naming the column.

        ON THE TRAINING ROWS the extension reproduces `embedding_` only in
        exact arithmetic and only for the fit's own affinity: asked as a
        query, a training row sees itself at 0.5 where the fit's graph had
        1.0, has no reverse edges, and the eigenpairs hold to `eigen_tol`.
        `predict(X_train) == labels_` is therefore not promised; how often it
        holds is measured in
        bench/results/identity_break/2026-09-15_spectral-predict/.

        Requires `prediction_data=True` at fit; refused by name otherwise.
        Returns int32 labels, the dtype and numbering of `labels_`.
        """
        return self._predict_embedding(X)[0]

    def _predict_embedding(self, X):
        """`predict`'s labels and the extended embedding (n_new x
        n_components float32)."""
        if not hasattr(self, "labels_"):
            raise ValueError(
                "mojolearn SpectralClustering.predict: this instance is not "
                "fitted yet; call fit first"
            )
        if getattr(self, "_pd_eigenvalues", None) is None:
            raise ValueError(
                "mojolearn SpectralClustering.predict: prediction data was not "
                "stored. Fit with SpectralClustering(prediction_data=True), which "
                "keeps the eigenpairs, degrees and centroids the Nystrom extension "
                "needs (DEVIATION 2860)"
            )
        if self.affinity not in _AFFINITIES:
            raise ValueError(
                f"mojolearn SpectralClustering.predict: affinity={self.affinity!r} "
                "has no out-of-sample rule; it must be one of "
                f"{list(_AFFINITIES)}"
            )
        k = int(self.n_components_)
        n_train = int(self._pd_diag.shape[0])
        if self.affinity == "precomputed":
            A = X.toarray() if callable(getattr(X, "toarray", None)) else X
            shape = _shape_of(A)
            if len(shape) != 2 or shape[1] != n_train:
                raise ValueError(
                    "mojolearn SpectralClustering.predict: with affinity="
                    "'precomputed', X must be the affinity between the new rows "
                    f"and the {n_train} training rows, shape (n_new, {n_train}); "
                    f"got shape {shape}"
                )
            q, _ = as_f32_c(A, ndim=2, name="X")
            if q.shape[0] < 1:
                raise ValueError("mojolearn SpectralClustering.predict: X has no rows; refused by name")
            if not all_finite(q):
                raise ValueError(
                    "mojolearn SpectralClustering.predict: the affinity has a "
                    "non-finite entry; refused by name"
                )
            if q.min() < 0:
                raise ValueError(
                    "mojolearn SpectralClustering.predict: the affinity has a "
                    "negative entry; refused by name, as the fit refuses one"
                )
            train_addr, n_features, affinity = 0, 0, 1
        else:
            q = _check_queries(X, self.n_features_in_, "SpectralClustering.predict")
            train_addr = addr_ro(self._fit_X, name="training rows")
            n_features, affinity = int(self.n_features_in_), 0
        nq = int(q.shape[0])
        labels = empty((nq,), "<i4")
        emb = empty((nq, k), "<f4")
        self._bind().spectral_predict(
            # ORDER MATCHES bindings/_mojolearn_metrics.mojo::spectral_predict_binding.
            [addr_ro(q, name="X"), train_addr,
             addr_ro(self._pd_eigenvalues, name="eigenvalues"),
             addr_ro(self._pd_eigenvectors, name="eigenvectors"),
             addr_ro(self._pd_diag, name="degree scaling"),
             addr_ro(self._pd_centroids, name="centroids"),
             addr(labels, name="labels"), addr(emb, name="embedding")],
            # n_train, n_queries, n_features, n_components, n_clusters,
            # n_neighbors, affinity
            [n_train, nq, n_features, k, int(self.n_clusters), int(self.n_neighbors), affinity],
        )
        return labels, emb

    def save(self, path):
        """Write the prediction data of a `prediction_data=True` fit to
        `path` as an npz: `eigenvalues` `<f4` (k), `eigenvectors` `<f4`
        (n_train, k), `diag` `<f4` (n_train), `centroids` `<f4`
        (n_clusters, k), `labels` `<i4`, `x` `<f4` (the training rows,
        'nearest_neighbors' only), `affinity`, `numeric_mode`, `eigen_tol`
        `<f8` and `meta` `<i8` [n_clusters, n_components_, n_neighbors,
        n_init, seed, random_state_is_none, n_train, n_features_in_ (0 for
        precomputed)]. A loaded model predicts; it does not refit. On a
        CPU-only install `SpectralClustering.load(path).predict(X)` runs
        through `_mojolearn_metrics_host.spectral_predict`."""
        if not hasattr(self, "labels_"):
            raise RuntimeError("this estimator is not fitted yet")
        if getattr(self, "_pd_eigenvalues", None) is None:
            raise ValueError(
                "mojolearn SpectralClustering.save: prediction data was not stored; "
                "fit with SpectralClustering(prediction_data=True). A model without "
                "it can label no new row, so there is nothing a saved file could serve"
            )
        nn = self.affinity == "nearest_neighbors"
        n_train = int(self._pd_diag.shape[0])
        arrays = {
            "format": _SPECTRAL_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": _saved_mode(self),
            "affinity": str(self.affinity),
            "eigenvalues": self._pd_eigenvalues,
            "eigenvectors": self._pd_eigenvectors,
            "diag": self._pd_diag,
            "centroids": self._pd_centroids,
            "labels": self.labels_,
            "eigen_tol": Array.from_list([float(self.eigen_tol)], "<f8"),
            "meta": Array.from_list(
                [int(self.n_clusters), int(self.n_components_), int(self.n_neighbors),
                 int(self.n_init), int(self._seed), 1 if self.random_state is None else 0,
                 n_train, int(self.n_features_in_) if nn else 0],
                "<i8",
            ),
        }
        if nn:
            arrays["x"] = self._fit_X
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load a model saved by `save`. The result predicts; every array is
        read at its saved dtype and never cast."""
        arrays = _serialize.read_npz(path, _SPECTRAL_FORMAT)
        _check_saved_by(arrays, path, cls)
        meta = _serialize.exact(arrays, "meta", "<i8")
        if meta.size != 8:
            raise ValueError(f"mojolearn: {path!r} meta holds {meta.size} fields, 8 are needed")
        n_clusters, k, n_neighbors, n_init, seed, seed_none, n_train, nf = (int(v) for v in meta.tolist())
        tol = _serialize.exact(arrays, "eigen_tol", "<f8")
        if tol.size != 1:
            raise ValueError(f"mojolearn: {path!r} eigen_tol must hold one value")
        affinity = _serialize.scalar_str(arrays, "affinity")
        obj = cls(
            n_clusters=n_clusters, n_components=k, n_neighbors=n_neighbors, n_init=n_init,
            random_state=None if seed_none else seed, eigen_tol=float(tol[0]),
            affinity=affinity, prediction_data=True,
        )
        _restore_mode(obj, arrays)
        ev = _serialize.exact(arrays, "eigenvalues", "<f4")
        evec = _serialize.exact(arrays, "eigenvectors", "<f4")
        dg = _serialize.exact(arrays, "diag", "<f4")
        cent = _serialize.exact(arrays, "centroids", "<f4")
        labels = _serialize.exact(arrays, "labels", "<i4")
        if (ev.size != k or evec.ndim != 2 or tuple(evec.shape) != (n_train, k) or dg.size != n_train
                or cent.ndim != 2 or tuple(cent.shape) != (n_clusters, k) or labels.size != n_train):
            raise ValueError(
                f"mojolearn: {path!r} prediction data shapes do not match n_train={n_train}, "
                f"n_components={k}, n_clusters={n_clusters}"
            )
        obj._fit_X = None
        if affinity == "nearest_neighbors":
            x = _serialize.exact(arrays, "x", "<f4")
            if x.ndim != 2 or tuple(x.shape) != (n_train, nf):
                raise ValueError(f"mojolearn: {path!r} x shape {tuple(x.shape)} is not ({n_train}, {nf})")
            obj._fit_X = x
            obj.n_features_in_ = nf
        obj._pd_eigenvalues = ev
        obj._pd_eigenvectors = evec
        obj._pd_diag = dg
        obj._pd_centroids = cent
        obj.labels_ = labels
        obj.n_components_ = k
        return obj


class SpectralEmbedding:
    """Laplacian eigenmaps. Reference: cuML's `manifold.SpectralEmbedding`.

    The kNN connectivity graph (or a precomputed affinity), the normalized
    graph Laplacian and RAFT's thick-restart Lanczos, the same stages
    `SpectralClustering` runs before its k-means; the trivial eigenvector is
    dropped, so `embedding_` is `n_samples x n_components`.

    HONORED: `n_components`, `affinity` in {'nearest_neighbors',
    'precomputed'}, `random_state`, `n_neighbors` (None means
    `max(n_samples // 10, 1)`, the reference's rule). `random_state=None`
    means seed 0 and is deterministic (DEVIATION 891). The eigensolver
    tolerance is cuVS's struct default `1e-5` and is not a parameter, as in
    the reference.

    REFUSED BY NAME: any other `affinity`, `gamma`, `eigen_solver`,
    `eigen_tol`, `n_jobs`, `verbose`. A precomputed affinity with a repeated
    `(row, col)` entry, a negative entry or a non-finite entry is refused;
    its diagonal entries are dropped, as the reference drops them.

    There is no `transform`: the embedding is of the fitted rows only, in
    the reference and in scikit-learn alike.
    """

    _drop_first = True
    _norm_laplacian = True

    def __init__(
        self,
        n_components=2,
        *,
        affinity="nearest_neighbors",
        random_state=None,
        n_neighbors=None,
        gamma=None,
        eigen_solver=None,
        eigen_tol=None,
        n_jobs=None,
        verbose=False,
    ):
        if affinity not in _AFFINITIES:
            raise ValueError(
                f"mojolearn SpectralEmbedding: affinity={affinity!r} is "
                f"refused; it must be one of {list(_AFFINITIES)}. 'rbf', "
                "'precomputed_nearest_neighbors' and a callable are different "
                "affinity constructions that nothing here implements."
            )
        if gamma is not None:
            raise NotImplementedError(
                "mojolearn SpectralEmbedding: gamma is refused; it "
                "parameterizes an RBF affinity, and affinity is restricted to "
                "'nearest_neighbors' and 'precomputed'"
            )
        if eigen_solver is not None:
            raise NotImplementedError(
                f"mojolearn SpectralEmbedding: eigen_solver={eigen_solver!r} "
                "is refused; there is one solver here, RAFT's thick-restart "
                "Lanczos, and it is the one the identity profile pins"
            )
        if eigen_tol is not None:
            raise NotImplementedError(
                f"mojolearn SpectralEmbedding: eigen_tol={eigen_tol!r} is "
                "refused; the tolerance is cuVS's struct default "
                f"{DEFAULT_EIGEN_TOL} and the reference exposes no parameter "
                "for it on this class"
            )
        if n_jobs is not None:
            raise NotImplementedError(
                "mojolearn SpectralEmbedding: n_jobs is refused; there is no "
                "host thread pool to size"
            )
        if verbose:
            raise NotImplementedError(
                "mojolearn SpectralEmbedding: verbose is refused; nothing in "
                "this path prints. Set MOJOLEARN_IDENTITY_TRACE=<path> for "
                "the identity card instead"
            )
        if int(n_components) < 1:
            raise ValueError(
                "mojolearn SpectralEmbedding: n_components must be at least 1"
            )
        if n_neighbors is not None and int(n_neighbors) < 1:
            raise ValueError(
                "mojolearn SpectralEmbedding: n_neighbors must be at least 1, "
                "or None for max(n_samples // 10, 1)"
            )
        if random_state is None:
            seed = DEFAULT_SEED
        else:
            seed = int(random_state)
            if seed < 0 or seed >= 2**32:
                raise ValueError(
                    "mojolearn SpectralEmbedding: random_state must satisfy "
                    f"0 <= random_state < 2**32, got {random_state!r}"
                )
        self.n_components = int(n_components)
        self.affinity = affinity
        self.random_state = random_state
        self.n_neighbors = None if n_neighbors is None else int(n_neighbors)
        self._seed = seed

    def _resolved_neighbors(self, n):
        return self.n_neighbors if self.n_neighbors is not None else max(n // 10, 1)

    def fit(self, X, y=None):
        k = self.n_components
        drop_first = bool(self._drop_first)
        norm_laplacian = bool(self._norm_laplacian)
        # The one piece of arithmetic the reference keeps in Python
        # (spectral_embedding.pyx:294): the transform receives this number.
        n_lanczos = k + 1 if drop_first else k
        if self.affinity == "precomputed":
            rows, cols, vals, n = _coo_triples(X, "SpectralEmbedding")
            if vals.size == 0:
                raise ValueError(
                    "mojolearn SpectralEmbedding: the precomputed affinity "
                    "matrix has no nonzero entries"
                )
            if not all_finite(vals):
                raise ValueError(
                    "mojolearn SpectralEmbedding: the precomputed affinity "
                    "matrix has a non-finite entry"
                )
            if vals.min() < 0:
                raise ValueError(
                    "mojolearn SpectralEmbedding: the precomputed affinity "
                    "matrix has a negative entry (refused by name: sqrt of a "
                    "negative degree is a NaN)"
                )
            self._check_shape(n, n_lanczos)
            self.n_neighbors_ = self._resolved_neighbors(n)
            embedding = empty((n, k), "<f4")
            # ORDER MATCHES bindings/_mojolearn_metrics.mojo::
            # spectral_embedding_graph_binding.
            params = [n, int(vals.shape[0]), n_lanczos, k,
                      int(norm_laplacian), int(drop_first), self._seed]
            n_out = int(_get_binding().spectral_embedding_graph(
                addr_ro(rows, name="rows"), addr_ro(cols, name="cols"),
                addr_ro(vals, name="vals"), addr(embedding, name="embedding"),
                params,
            ))
        else:
            x, self.input_copied_ = as_f32_c(X, ndim=2, name="X")
            if not all_finite(x):
                raise ValueError(
                    "mojolearn SpectralEmbedding: X contains NaN or infinity"
                )
            n = int(x.shape[0])
            self._check_shape(n, n_lanczos)
            n_neighbors = self._resolved_neighbors(n)
            if n_neighbors > n:
                raise ValueError(
                    f"mojolearn SpectralEmbedding: n_neighbors={n_neighbors} "
                    f"exceeds n_samples={n}"
                )
            self.n_neighbors_ = n_neighbors
            embedding = empty((n, k), "<f4")
            # ORDER MATCHES bindings/_mojolearn_metrics.mojo::
            # spectral_embedding_dataset_binding.
            params = [n, int(x.shape[1]), n_lanczos, k, n_neighbors,
                      int(norm_laplacian), int(drop_first), self._seed]
            n_out = int(_get_binding().spectral_embedding_dataset(
                addr_ro(x, name="x"), addr(embedding, name="embedding"), params,
            ))
            self.n_features_in_ = int(x.shape[1])
        if n_out != k:
            raise RuntimeError(
                f"mojolearn SpectralEmbedding: the kernel returned {n_out} "
                f"embedding columns where {k} were expected"
            )
        self.embedding_ = embedding
        return self

    def _check_shape(self, n, n_lanczos):
        if n_lanczos >= n:
            raise ValueError(
                f"mojolearn SpectralEmbedding: n_components={self.n_components} "
                f"needs {n_lanczos} eigenpairs, which must be below "
                f"n_samples={n}"
            )

    def fit_transform(self, X, y=None):
        return self.fit(X, y).embedding_


def spectral_embedding(
    A,
    *,
    n_components=8,
    affinity="nearest_neighbors",
    random_state=None,
    n_neighbors=None,
    norm_laplacian=True,
    drop_first=True,
):
    """`SpectralEmbedding(...).fit_transform(A)` with the two switches the
    class fixes: `norm_laplacian` (the symmetric normalized Laplacian) and
    `drop_first` (drop the trivial eigenvector). Reference: cuML's
    `manifold.spectral_embedding`."""
    for name, value in (("norm_laplacian", norm_laplacian), ("drop_first", drop_first)):
        if not isinstance(value, bool):
            raise TypeError(
                f"mojolearn spectral_embedding: {name} must be a bool, got "
                f"{type(value).__name__}"
            )
    model = SpectralEmbedding(
        n_components=n_components, affinity=affinity,
        random_state=random_state, n_neighbors=n_neighbors,
    )
    model._drop_first = drop_first
    model._norm_laplacian = norm_laplacian
    return model.fit_transform(A)

# SPDX-License-Identifier: Apache-2.0
"""Frozen b32103c00 Python paths of PageRank.fit, Louvain.fit/fit_predict,
connected_components and SVGP.fit/predict/score, used only as an independent
test oracle for their one-call native doors (lane native-neighbors-20261024)."""
import struct

from mojolearn._buffer import addr, addr_ro, as_f32_c, empty, zeros
from mojolearn._lazy_out import _empty_out
from mojolearn._expansion_neighbors import (
    _XNeighbors, _f32, _f32_1d, _i32, _f32_scalar, _p2m_relabel, _OLD_ITEMS, _U_IDENTITY,
)


def _adjacency(A):
    A = _f32(A, "adjacency")
    if A.shape[0] != A.shape[1]:
        raise ValueError("the adjacency matrix must be square")
    return A


def pagerank_unit(self, v, n, what):
    """A caller's length-n non-negative vector divided by its sum (IEEE
    double, rounded once to float32), as networkx normalizes its dicts."""
    # lane/fam2-neighbors: the sum, the sign test and the n divisions by
    # the binding (`x_neighbors_unit_ff`, device kernels; registered in
    # IDENTICAL unless -D MOJOLEARN_IDN_XN_UNIT_DEV_OFF)
    # (lane py-runtime-b: the Python fsum-and-divide route behind the
    # _OFF define is deleted; that binary refuses a weight vector)
    unit_fn = getattr(self._bind(), "x_neighbors_unit_ff", None)
    if unit_fn is None:
        raise NotImplementedError(
            f"mojolearn PageRank {what}: this binary lacks x_neighbors_unit_ff "
            "(rebuild without -D MOJOLEARN_IDN_XN_UNIT_DEV_OFF)")
    wv = _f32_1d(v, what)
    if wv.shape[0] != n:
        raise ValueError(f"{what} must be n non-negative values, not all zero")
    out = _empty_out((n,), "<f4")
    uinfo = empty((2,), "<i4")
    unit_fn([addr_ro(wv, name="xn_unit v"), addr(out, name="xn_unit out"),
             addr(uinfo, name="xn_unit info")], [n], [])
    neg, zero = uinfo.tolist()
    if neg or zero:
        raise ValueError(f"{what} must be n non-negative values, not all zero")
    return out


def pagerank_fit(self, A, y=None):
    """Compute the PageRank scores of the graph.

    Parameters
    ----------
    A : array-like of shape (n, n), scipy.sparse matrix, or tuple
        Weighted adjacency matrix (``A[i, j]`` is the weight of i -> j): a
        dense square array, a scipy.sparse matrix, or an
        ``(indptr, indices, n)`` CSR tuple.
    y : None
        Ignored.

    Returns
    -------
    self : PageRank
        The fitted estimator, with ``pagerank_`` and ``n_iter_``.

    Raises
    ------
    RuntimeError
        If the iteration does not converge within ``max_iter`` steps.
    """
    binary = self.weight is None or self.weight is False
    csr = _csr_graph_of(A)
    if csr is not None:
        # lane gap-graph (2026-10-08, docs/plans/gaps-2026-10-08.md 5.2):
        # a scipy.sparse matrix or an (indptr, indices, n) tuple goes to
        # the device as it is (`xn_pr_csr`): the column lists are built
        # by one sort on the device, the vectors the caller left to the
        # default are filled there, no dense matrix and no separate
        # fill calls. Glue only: pointer handoff and flags.
        indptr, indices, vals, n = csr
    else:
        A = _adjacency(A)
        n = A.shape[0]
    # the iteration over the column lists of the adjacency, built
    # and iterated on the device (lane hr-graph,
    # x_neighbors/graph_par.mojo): the dense chains with their zero
    # terms left out, the dangling mass and |x' - x| blocked folds

    def uniform():
        # [1/n] * n: 1/n in IEEE double rounded once to float32 (the
        # binding's float argument), filled on the device
        u = empty((n,), "<f4")
        if n:
            self._op("p2m_fill", [(u, 1)], (n,), (1.0 / n,))
        return u

    info = empty((2,), "<i4")
    thr = struct.unpack("<Q", struct.pack("<d", n * float(self.tol)))[0]
    if csr is not None:
        p_uniform = self.personalization is None
        p = empty((n,), "<f4") if p_uniform else pagerank_unit(self, self.personalization, n, "personalization")
        dw_uniform = self.dangling is None and p_uniform
        dw = p if self.dangling is None else pagerank_unit(self, self.dangling, n, "dangling")
        x_uniform = self.nstart is None
        x = empty((n,), "<f4") if x_uniform else pagerank_unit(self, self.nstart, n, "nstart").copy()
        has_vals = vals is not None
        if not has_vals:
            vals = empty((1,), "<f4")
        self._op("pr_csr", [(indptr, 0), (indices, 0), (vals, 0), (x, 1), (p, 0), (dw, 0), (info, 1)],
                 (n, indices.shape[0], 1 if has_vals else 0, int(self.max_iter), thr >> 32,
                  thr & 0xFFFFFFFF, 1 if binary else 0, 1 if x_uniform else 0, 1 if p_uniform else 0,
                  1 if dw_uniform else 0),
                 (_f32_scalar(self.alpha),))
    else:
        if self.personalization is None:
            p = uniform()
        else:
            p = pagerank_unit(self, self.personalization, n, "personalization")
        dw = p if self.dangling is None else pagerank_unit(self, self.dangling, n, "dangling")
        x = uniform() if self.nstart is None else pagerank_unit(self, self.nstart, n, "nstart")
        x = x.copy()
        self._op("pr_iterate_sparse", [(A, 0), (x, 1), (p, 0), (dw, 0), (info, 1)],
                 (n, int(self.max_iter), thr >> 32, thr & 0xFFFFFFFF, 1 if binary else 0),
                 (_f32_scalar(self.alpha),))
    it, ok = info.tolist()
    if ok:
        self.pagerank_ = x
        self.n_iter_ = int(it)
        return self
    raise RuntimeError(f"PageRank: power iteration failed to converge within {self.max_iter} iterations")


def connected_components(A, directed=True, connection="weak", return_labels=True, numeric_mode=None):
    """Connected components of a graph.

    Follows ``scipy.sparse.csgraph.connected_components``. An edge is any
    nonzero entry of the adjacency matrix. Weak connectivity ignores edge
    direction. Components are numbered in order of their lowest node, so
    component 0 holds node 0, as scipy numbers them.

    Parameters
    ----------
    A : array-like of shape (n, n), scipy.sparse matrix, or tuple
        Adjacency matrix: a dense square array, a scipy.sparse matrix (any
        format), or an ``(indptr, indices, n)`` CSR tuple.
    directed : bool, default True
        Whether the graph is directed.
    connection : {'weak', 'strong'}, default 'weak'
        Connectivity type. ``'strong'`` on a directed graph raises
        ``NotImplementedError``; on an undirected graph it equals
        ``'weak'``. Other values raise ``ValueError``.
    return_labels : bool, default True
        Also return the component label of each node.
    numeric_mode : {'identical', 'fast'} or None, default None
        Numeric mode for the call (``'fast'`` is Apple only). None uses the
        process default.

    Returns
    -------
    n_components : int
        Number of connected components.
    labels : Array of shape (n,), int32
        Component of each node; returned only when ``return_labels`` is True.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> A = (np.random.default_rng(0).random((500, 500)) < 0.002).astype(np.float32)
    >>> n_components, labels = mojolearn.connected_components(A, directed=False)
    """
    if directed and connection == "strong":
        raise NotImplementedError("connected_components: connection='strong' is not implemented")
    if connection not in ("weak", "strong"):
        raise ValueError("connection must be 'weak' or 'strong'")
    est = _XNeighbors()
    est.numeric_mode = numeric_mode
    csr = _csr_of(A)
    if csr is not None:
        # lane/neural-pass69 (2026-10-01): a sparse graph (scipy.sparse CSR, or
        # (indptr, indices, n)) walks its edge lists directly: the same rounds
        # and labels as the dense matrix's, without building or scanning it
        indptr, indices, n = csr
        lab = _p2m_iota(est, n)
        # TOMBSTONE: MOJOLEARN_CC_FAST_OFF (removed) deleted 2026-10-08 by 2851a6627; code recoverable at 2851a6627^.
        # Restore: git apply experiments/removed/MOJOLEARN_CC_FAST_OFF.patch; record in docs/TOMBSTONES.md.
        # lane gap-graph (2026-10-08, docs/plans/gaps-2026-10-08.md 5.3): the
        # GPU binding on every vendor and tier runs the batched rounds and
        # compacts the labels itself in order of first appearance
        # (`_cc_relabel`'s integers; lane/apple-fast-graph's CC_FAST
        # promoted) and puts n_components + 1 in info[1]; info[1] == 0
        # means the host binding ran (it never writes info[1]) and `lab`
        # still needs `_cc_relabel` (zeros, not empty, for that reason).
        info = zeros((2,), "<i4")
        est._op("cc_iterate_csr", [(indptr, 0), (indices, 0), (lab, 1), (info, 1)], (n, indices.shape[0]))
        k1 = int(info.tolist()[1])
        if k1 > 0:
            return (k1 - 1, lab) if return_labels else k1 - 1
        return _cc_relabel(lab, return_labels, est)
    A = _adjacency(A)
    n = A.shape[0]
    lab = _p2m_iota(est, n)
    # the min-label rounds as ONE resident op, A uploaded once
    info = empty((1,), "<i4")
    est._op("cc_iterate", [(A, 0), (lab, 1), (info, 1)], (n,))
    return _cc_relabel(lab, return_labels, est)


def _p2m_iota(est, n):
    """[0, 1, ..., n - 1] as int32 (`xn_p2m_iota` on the device)."""
    lab = empty((max(n, 0),), "<i4")
    if n > 0:
        est._op("p2m_iota", [(lab, 1)], (n,))
    return lab


def _cc_relabel(lab, return_labels, est):
    """(count, labels) renumbered by first occurrence (`xn_p2m_relabel`)."""
    if not lab.shape[0]:
        return (0, lab) if return_labels else 0
    got = _p2m_relabel(est, lab)
    if got is None:
        raise RuntimeError("connected_components: a component label fell outside [0, n)")
    cnt, labels = got
    return (cnt, labels) if return_labels else cnt


def _csr_of(A):
    """(indptr, indices, n) as int32 Arrays for a scipy.sparse matrix (any
    format: converted to CSR) or a tuple (indptr, indices, n); None for a
    dense input. The edge weights do not matter to connectivity."""
    if isinstance(A, tuple) and len(A) == 3:
        indptr, indices, n = A
        n = int(n)
    elif hasattr(A, "tocsr") and hasattr(A, "shape"):
        if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("the adjacency matrix must be square")
        M = A.tocsr()
        indptr, indices, n = M.indptr, M.indices, int(M.shape[0])
    else:
        return None
    ip = _i32(indptr, "indptr")
    ix = _i32(indices, "indices")
    if ip.shape[0] != n + 1:
        raise ValueError("connected_components: indptr must hold n + 1 entries")
    return ip, ix, n


def _csr_graph_of(A):
    """(indptr, indices, vals, n) for PageRank and Louvain (lane gap-graph):
    `_csr_of`'s lists plus the weights as a float32 Array (a scipy matrix's
    `data`), or None for the weights when the caller gave an (indptr,
    indices, n) tuple (every edge then weighs 1.0 on the device); None for
    a dense input. Glue only: the device checks the column order and the
    symmetry, never Python."""
    if isinstance(A, tuple) and len(A) == 3:
        indptr, indices, n = A
        n = int(n)
        vals = None
    elif hasattr(A, "tocsr") and hasattr(A, "shape"):
        if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("the adjacency matrix must be square")
        M = A.tocsr()
        indptr, indices, n = M.indptr, M.indices, int(M.shape[0])
        vals = _f32_1d(M.data, "data")
    else:
        return None
    ip = _i32(indptr, "indptr")
    ix = _i32(indices, "indices")
    if ip.shape[0] != n + 1:
        raise ValueError("the CSR adjacency's indptr must hold n + 1 entries")
    if int(ip[n]) != ix.shape[0]:  # argument check: one scalar, the entry count
        raise ValueError("the CSR adjacency's indices must hold indptr[n] entries")
    if vals is not None and vals.shape[0] != ix.shape[0]:
        raise ValueError("the CSR adjacency's data must hold one weight per entry")
    return ip, ix, vals, n


def louvain_fit(self, A, y=None):
    """Detect communities in the graph.

    Parameters
    ----------
    A : array-like of shape (n, n), scipy.sparse matrix, or tuple
        Symmetric weighted adjacency matrix: a dense square array, a
        scipy.sparse matrix, or an ``(indptr, indices, n)`` CSR tuple.
    y : None
        Ignored.

    Returns
    -------
    self : Louvain
        The fitted estimator.

    Raises
    ------
    ValueError
        If ``max_level`` is not a positive integer or None, the matrix is not
        symmetric, or the graph has no edges.
    """
    ml = 0 if self.max_level is None else int(self.max_level)
    if self.max_level is not None and ml < 1:
        raise ValueError("max_level must be a positive integer or None")
    csr = _csr_graph_of(A)
    if csr is not None:
        # lane gap-graph (2026-10-08, docs/plans/gaps-2026-10-08.md 5.1):
        # a scipy.sparse matrix or an (indptr, indices, n) tuple goes to
        # the device as it is (`xn_louvain_csr`): the graph slot is
        # filled from the lists, the symmetry, column-order and no-edge
        # checks run on the device (the op raises their ValueError
        # text), no dense matrix. Glue only: pointer handoff.
        indptr, indices, vals, n = csr
        has_vals = vals is not None
        if not has_vals:
            vals = empty((1,), "<f4")
        labels = empty((n,), "<i4")
        info = _empty_out((2,), "<f4")
        try:
            self._op("louvain_csr", [(indptr, 0), (indices, 0), (vals, 0), (labels, 1), (info, 1)],
                     (n, indices.shape[0], 1 if has_vals else 0, ml),
                     (_f32_scalar(self.resolution), _f32_scalar(self.threshold)))
        except Exception as e:  # the device checks' messages as ValueErrors
            msg = str(e)
            if "Louvain:" in msg:
                raise ValueError(msg[msg.index("Louvain:"):]) from None
            raise
    else:
        A = _adjacency(A)
        n = A.shape[0]
        # The symmetry and no-edge checks run natively (`xn_graph_symmetry`,
        # lane neural-pass14): over `A.tolist()` they were a 400-million-cell
        # Python scan at the board's 20,000 nodes, most of the race's minute.
        flags = _empty_out((2,), "<i4")
        self._op("graph_symmetry", [(A, 0), (flags, 1)], (n,))
        flags = flags.tolist()
        if flags[0]:
            raise ValueError("Louvain: the adjacency matrix must be symmetric (an undirected graph)")
        if not flags[1]:
            raise ValueError("Louvain: the graph has no edges")
        labels = empty((n,), "<i4")
        info = _empty_out((2,), "<f4")
        self._op("louvain", [(A, 0), (labels, 1), (info, 1)], (n, ml),
                 (_f32_scalar(self.resolution), _f32_scalar(self.threshold)))
    got = _p2m_relabel(self, labels)
    if got is None:
        raise RuntimeError("Louvain: a community label fell outside [0, n)")
    self.n_communities_, self.labels_ = got
    info = info.tolist()
    self.modularity_ = info[0]
    self.n_levels_ = int(info[1])
    return self


def louvain_fit_predict(self, A, y=None):
    return louvain_fit(self, A).labels_


def svgp_fit(self, X, y):
    """Fit the variational distribution for the given hyperparameters.

    Parameters
    ----------
    X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
        Training inputs.
    y : array-like of shape (n_samples,)
        Training targets.

    Returns
    -------
    self : SVGP
        The fitted model.

    Raises
    ------
    ValueError
        If X and y have different numbers of rows, or the inducing-point
        system is not positive definite (raise ``jitter`` or
        ``noise_variance``).
    """
    X = _f32(X)
    n, d = X.shape
    yv = _f32_1d(y, "y")
    if len(yv) != n:
        raise ValueError("X and y have different numbers of rows")
    if self.inducing_points is not None:
        Z = _f32(self.inducing_points, "inducing_points")
    else:
        M = min(int(self.n_inducing), n)
        Z = self._take_rows(X, [i * n // M for i in range(M)])  # glue: evenly spaced inducing row indices (M sized)
    M = Z.shape[0]
    # one resident device chain (lane/cgr-kernel, `xn_svgp_fit_ff`):
    # Kuu, then B = Kuf Kfu and b = Kuf y in float-float per Kfu row tile
    # (Kuf is Kfu^T bit for bit, never formed), then the float-float
    # solve (x_neighbors/svgp_ff.mojo). Taxi's Sigma (eigenvalues 1e-6 ..
    # 9e5) defeats float32: B's float32 accumulation error alone exceeds
    # its smallest eigenvalue. Kuu, B and b never leave the device.
    alpha = _empty_out((M,), "<f4")
    C = _empty_out((M, M), "<f4")
    qmu = _empty_out((M,), "<f4")
    qsqrt = _empty_out((M, M), "<f4")
    info = _empty_out((2,), "<f4")
    self._op("svgp_fit_ff", [(X, 0), (Z, 0), (yv, 0), (alpha, 1), (C, 1), (qmu, 1), (qsqrt, 1), (info, 1)],
             (n, M, d),
             (_f32_scalar(self._gamma_value()), _f32_scalar(self.kernel_variance),
              _f32_scalar(self.noise_variance), _f32_scalar(self.jitter), _f32_scalar(self.kernel_variance)))
    elbo, ok = info.tolist()
    self.precision_ = "float-float"
    if ok == 0:
        raise ValueError("SVGP: the inducing system is not positive definite; raise jitter or noise_variance")
    self.Z_, self._alpha, self._C = Z, alpha, C
    self.q_mu_ = qmu
    self.q_sqrt_ = qsqrt
    self.elbo_ = elbo
    self.n_features_in_ = d
    return self


def svgp_predict_f(self, X):
    """Mean and variance of the latent function at X.

    Parameters
    ----------
    X : array-like of shape (n_queries, n_features)
        Query inputs.

    Returns
    -------
    mean : Array of shape (n_queries,), float32
    var : Array of shape (n_queries,), float32
    """
    Q = _f32(X)
    M = self.Z_.shape[0]
    if not _OLD_ITEMS and Q.shape[0]:
        # the fused chain (lane/py-dn-kern): Ksu, the mean and the
        # variance per row tile on the device (`xn_svgp_predict`)
        nq = Q.shape[0]
        mean = _empty_out((nq,), "<f4")
        var = _empty_out((nq,), "<f4")
        self._op("svgp_predict", [(Q, 0), (self.Z_, 0), (self._alpha, 0), (self._C, 0), (mean, 1), (var, 1)],
                 (nq, M, Q.shape[1]),
                 (_f32_scalar(self._gamma_value()), _f32_scalar(self.kernel_variance),
                  _f32_scalar(self.kernel_variance)))
        return mean, var
    Ksu = self._k(Q, self.Z_)
    mean = self._matmul(Ksu, self._alpha.reshape((M, 1))).reshape((Q.shape[0],))
    var = _empty_out((Q.shape[0],), "<f4")
    self._op("svgp_var", [(Ksu, 0), (self._C, 0), (var, 1)], (Q.shape[0], M), (_f32_scalar(self.kernel_variance),))
    return mean, var


def svgp_predict(self, X):
    return svgp_predict_f(self, X)[0]


def svgp_score(self, X, y, sample_weight=None):
    """`RegressorScoreMixin.score` as SVGP inherited it."""
    from mojolearn._metrics_impl import r2_score
    truth, _ = as_f32_c(y, ndim=None, name="y")
    pred, _ = as_f32_c(svgp_predict(self, X), ndim=None, name="prediction")
    return r2_score(truth, pred, sample_weight=sample_weight,
                    numeric_mode=getattr(self, "numeric_mode", None))

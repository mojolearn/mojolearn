# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE DECOMP LANE'S PUBLIC DOOR (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).

Owned by the `decomp` expansion lane. `mojolearn/__init__.py` imports this
module and makes every name in `__all__` public as `mojolearn.<name>`; a name
that is already public is refused at import.

HOW THE ARITHMETIC IS SPLIT. Every floating-point operation on data runs in
Mojo, in the cells of `x_decomp/cells.mojo`, through `_mojolearn_x_decomp`
(the GPU binding: one thread per output) or, on a CPU-only install, its host
binding (the same cells in a host loop). This file holds CONTROL FLOW and
DATA MOVEMENT only: slicing, stacking, transposing, sign flips (exact) and
comparisons of float32 values the binding returned. Scalars handed to a cell
are Python floats rounded once to float32 at the binding boundary, and are
computed here only with IEEE +, -, *, / and sqrt, which CPython performs
correctly rounded on every platform. So the CPU column and every GPU column
run the same arithmetic in the same order, and the result is the same bits.
"""
import array
import math

from . import _backend
from ._buffer import as_f32_c, as_i32_c, frombytes

__all__ = ["IncrementalPCA", "GaussianRandomProjection", "SparseRandomProjection", "johnson_lindenstrauss_min_dim",
           "NMF", "FastICA", "FactorAnalysis",
           "lu_factor", "lu_solve", "solve", "lstsq", "randomized_svd"]

_BINDING = "_mojolearn_x_decomp"

# x_decomp/cells.mojo op codes
_OP = dict(
    muz=36,
    add=0, sub=1, mul=2, div=3, axpy=4, maxs=5, mu=6, sqrt=7, sq=8, exp=9, logs=10, tanh=11,
    onemsq=12, abs=13, scale=14, fma=15, recip=16, soft=17, submul=18, mins=19, copyb=20,
    sqdiff=21, adds=22, gts=23, digamma=24, expg=25, expgp=26, cube=27, cubep=28, max=30,
    min=31, sign=33, le=34, select=35,
)


def _f32(x):
    """x rounded once to float32, as the binding would round it."""
    return array.array("f", [x])[0]


class _M:
    """A row-major float32 matrix held in an `array.array('f')`."""
    __slots__ = ("s", "r", "c")

    def __init__(self, s, r, c):
        if len(s) != r * c:
            raise ValueError("x_decomp: matrix store does not match its shape")
        self.s, self.r, self.c = s, r, c

    @classmethod
    def zeros(cls, r, c):
        return cls(array.array("f", bytes(4 * r * c)), r, c)

    @classmethod
    def of(cls, values, r, c):
        return cls(array.array("f", values), r, c)

    @classmethod
    def from_input(cls, X, name="X"):
        a = as_f32_c(X, ndim=2, name=name)[0]
        if a.ndim != 2 or min(a.shape) == 0:
            raise ValueError(f"{name}: a nonempty two-dimensional input is required")
        s = array.array("f")
        s.frombytes(a.tobytes())
        for v in s:
            if v != v or v in (math.inf, -math.inf):
                raise ValueError(f"{name}: input must be finite; NaN/inf are unsupported")
        return cls(s, a.shape[0], a.shape[1])

    @property
    def addr(self):
        return self.s.buffer_info()[0] if len(self.s) else _M._one.s.buffer_info()[0]

    def copy(self):
        return _M(array.array("f", self.s), self.r, self.c)

    def out(self, shape=None):
        """A public float32 Array (C order) of this matrix."""
        return frombytes(self.s.tobytes(), "<f4", shape or (self.r, self.c))

    def at(self, i, j=0):
        return self.s[i * self.c + j]

    def row(self, i):
        return self.s[i * self.c:(i + 1) * self.c]

    def rows(self, a, b):
        return _M(self.s[a * self.c:b * self.c], b - a, self.c)

    def take_rows(self, idx):
        out = array.array("f")
        for i in idx:
            out.extend(self.s[i * self.c:(i + 1) * self.c])
        return _M(out, len(idx), self.c)

    def cols(self, a, b):
        out = array.array("f")
        for i in range(self.r):
            out.extend(self.s[i * self.c + a:i * self.c + b])
        return _M(out, self.r, b - a)

    def take_cols(self, idx):
        out = array.array("f")
        for i in range(self.r):
            base = i * self.c
            out.extend(self.s[base + j] for j in idx)
        return _M(out, self.r, len(idx))

    @property
    def T(self):
        out = array.array("f")
        for j in range(self.c):
            out.extend(self.s[j::self.c])
        return _M(out, self.c, self.r)

    def reshape(self, r, c):
        return _M(self.s, r, c)

    def neg_rows(self, flags):
        """Exact sign flip of the rows whose flag is set."""
        out = array.array("f", self.s)
        for i, f in enumerate(flags):
            if f:
                for j in range(i * self.c, (i + 1) * self.c):
                    out[j] = -out[j]
        return _M(out, self.r, self.c)

    def neg_cols(self, flags):
        out = array.array("f", self.s)
        for j, f in enumerate(flags):
            if f:
                for i in range(self.r):
                    out[i * self.c + j] = -out[i * self.c + j]
        return _M(out, self.r, self.c)

    def list(self):
        return list(self.s)


_M._one = _M(array.array("f", [0.0]), 1, 1)


def _vstack(*ms):
    s = array.array("f")
    for m in ms:
        s.extend(m.s)
    return _M(s, sum(m.r for m in ms), ms[0].c)


def _hstack(*ms):
    s = array.array("f")
    for i in range(ms[0].r):
        for m in ms:
            s.extend(m.s[i * m.c:(i + 1) * m.c])
    return _M(s, ms[0].r, sum(m.c for m in ms))


class _Kit:
    """The binding's cells, called on `_M` matrices."""

    def __init__(self, mode, binding=None):
        self.mode = mode
        self.b = _backend.binding("_mojolearn_x_decomp", mode) if binding is None else binding

    # ---- elementwise
    def ew(self, op, A, B=None, C=None, s=0.0):
        def mode_of(X):
            if X is None:
                return _M._one, 3
            if X.r == A.r and X.c == A.c:
                return X, 0
            if X.r == 1 and X.c == A.c:
                return X, 1
            if X.c == 1 and X.r == A.r:
                return X, 2
            if X.r == 1 and X.c == 1:
                return X, 3
            raise ValueError(f"x_decomp: cannot broadcast {X.r}x{X.c} against {A.r}x{A.c}")
        Bm, bm = mode_of(B)
        Cm, cm = mode_of(C)
        out = _M.zeros(A.r, A.c)
        if A.r * A.c:
            self.b.x_decomp_ew(A.addr, Bm.addr, Cm.addr, out.addr,
                               [_OP[op], A.r * A.c, A.c, len(Bm.s), bm, len(Cm.s), cm], float(s))
        return out

    def const(self, v, r=1, c=1):
        return _M.of([v] * (r * c), r, c)

    # ---- reductions and products
    def mm(self, A, B, ta=False, tb=False):
        m, k = (A.c, A.r) if ta else (A.r, A.c)
        k2, n = (B.c, B.r) if tb else (B.r, B.c)
        if k != k2:
            raise ValueError(f"x_decomp: gemm inner dimensions {k} and {k2} differ")
        out = _M.zeros(m, n)
        if m * n:
            self.b.x_decomp_gemm(A.addr, B.addr, out.addr, [m, k, n, int(ta), int(tb)])
        return out

    def colsum(self, A):
        out = _M.zeros(1, A.c)
        self.b.x_decomp_colsum(A.addr, out.addr, [A.r, A.c])
        return out

    def rowsum(self, A):
        out = _M.zeros(A.r, 1)
        self.b.x_decomp_rowsum(A.addr, out.addr, [A.r, A.c])
        return out

    def total(self, A):
        """The sum of every entry (1 x 1): rows ascending, then the row sums ascending."""
        return self.colsum(self.rowsum(A))

    def sqdist(self, A, B):
        out = _M.zeros(A.r, B.r)
        self.b.x_decomp_sqdist(A.addr, B.addr, out.addr, [A.r, B.r, A.c])
        return out

    def rand(self, r, c, seed, stream, kind):
        out = _M.zeros(r, c)
        if r * c:
            self.b.x_decomp_rand(out.addr, [r * c, int(seed) & 0xFFFFFFFF, int(stream) & 0xFFFFFFFF, kind])
        return out

    # ---- small dense linear algebra
    def eigh(self, A):
        """Ascending eigenvalues (1 x n) and eigenvectors in COLUMNS (n x n)."""
        n = A.r
        w, v = _M.zeros(1, n), _M.zeros(n, n)
        self.b.x_decomp_eigh(A.addr, w.addr, v.addr, [n])
        return w, v

    def lu(self, A):
        n = A.r
        lu = A.copy()
        piv = array.array("i", [0] * n)
        info = _M.zeros(1, 1)
        self.b.x_decomp_lu(lu.addr, piv.buffer_info()[0], info.addr, [n])
        return lu, piv, int(info.s[0])

    def lu_solve(self, lu, piv, B):
        out = B.copy()
        self.b.x_decomp_lu_solve(lu.addr, piv.buffer_info()[0], out.addr, [lu.r, B.c])
        return out

    def cd_rows(self, W, HHt, XHt, perm):
        """One sklearn `_update_cdnmf_fast` sweep over every row of W, in
        place; returns the total violation (rows ascending)."""
        n, kc = W.r, W.c
        viol = _M.zeros(n, 1)
        p = array.array("i", perm)
        self.b.x_decomp_cd_rows(W.addr, HHt.addr, XHt.addr, p.buffer_info()[0], viol.addr, [n, kc])
        return self.total(viol).s[0]

    def orth(self, A):
        """A copy of A with its columns orthonormalized (MGS2)."""
        Q = A.copy()
        self.b.x_decomp_orth(Q.addr, [A.r, A.c])
        return Q

    def chol(self, A):
        L = A.copy()
        info = _M.zeros(1, 1)
        self.b.x_decomp_chol(L.addr, info.addr, [A.r])
        return L, int(info.s[0])

    # ---- composites (cells only)
    def colmean(self, A):
        return self.ew("scale", self.colsum(A), s=1.0 / A.r)

    def center(self, A, mean):
        return self.ew("sub", A, mean)


def _mode(numeric_mode):
    return _backend.default_mode() if numeric_mode is None else numeric_mode


def _svd_flip_v(Vt):
    """sklearn `svd_flip(u_based_decision=False)`: each row of Vt signed so its
    largest-|.| entry (first on a tie) is positive. Exact sign flips."""
    flags = []
    for i in range(Vt.r):
        row = Vt.row(i)
        best, arg = -1.0, 0
        for j, v in enumerate(row):
            if abs(v) > best:
                best, arg = abs(v), j
        flags.append(row[arg] < 0)
    return Vt.neg_rows(flags)


def _gram_svd(k, Z):
    """Thin SVD of Z (m x d) through the eigendecomposition of Z^T Z:
    S descending (1 x d), Vt (d x d, rows = right singular vectors)."""
    w, V = k.eigh(k.mm(Z, Z, ta=True))
    d = w.c
    order = list(range(d - 1, -1, -1))
    S = k.ew("sqrt", w.take_cols(order))
    Vt = V.take_cols(order).T
    return S, _svd_flip_v(Vt)


class _Base:
    _parameters = ()

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in self._parameters}

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self._parameters:
                raise ValueError(f"Invalid {type(self).__name__} parameter: {k}")
            setattr(self, k, v)
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def _kit(self):
        return _Kit(self.numeric_mode_)

    def _check(self, attr="components_"):
        if not hasattr(self, attr):
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit first")


# ================================================================ IncrementalPCA
class IncrementalPCA(_Base):
    """sklearn.decomposition.IncrementalPCA (reference: scikit-learn
    `decomposition/_incremental_pca.py`, `partial_fit`; the running mean and
    variance are `utils/extmath.py::_incremental_mean_and_var`).

    Each batch's SVD is the thin SVD of the stacked matrix sklearn builds,
    computed as the eigendecomposition of its d x d Gram matrix (the Jacobi
    eigh of decomposition/, device or host), then sklearn's `svd_flip` with
    `u_based_decision=False`. Not carried: sparse input."""
    _parameters = ("n_components", "whiten", "copy", "batch_size", "numeric_mode")

    def __init__(self, n_components=None, *, whiten=False, copy=True, batch_size=None, numeric_mode=None):
        self.n_components, self.whiten, self.copy = n_components, whiten, copy
        self.batch_size, self.numeric_mode = batch_size, numeric_mode

    def fit(self, X, y=None):
        self.numeric_mode_ = _mode(self.numeric_mode)
        for a in ("components_", "n_samples_seen_"):
            if hasattr(self, a):
                delattr(self, a)
        M = _M.from_input(X)
        n, d = M.r, M.c
        self.batch_size_ = 5 * d if self.batch_size is None else int(self.batch_size)
        mb = self.n_components or 0
        start = 0
        for _ in range(n // self.batch_size_):
            end = start + self.batch_size_
            if end + mb > n:
                continue
            self._partial(M.rows(start, end))
            start = end
        if start < n:
            self._partial(M.rows(start, n))
        return self

    def partial_fit(self, X, y=None):
        if not hasattr(self, "numeric_mode_"):
            self.numeric_mode_ = _mode(self.numeric_mode)
        self._partial(_M.from_input(X))
        return self

    def _partial(self, Xb):
        k = self._kit()
        n, d = Xb.r, Xb.c
        if self.n_components is None:
            nc = d if not hasattr(self, "components_") else self.components_m_.r
        else:
            nc = int(self.n_components)
        if not 1 <= nc <= d:
            raise ValueError(f"n_components={nc} must be between 1 and n_features={d}")
        if nc > n:
            raise ValueError(f"n_components={nc} must be less or equal to the batch number of samples {n}")
        if hasattr(self, "components_") and self.components_m_.c != d:
            raise ValueError("Number of input features has changed from the previous batch")
        seen = getattr(self, "n_samples_seen_", 0)
        if seen == 0:
            last_mean, last_var = _M.zeros(1, d), _M.zeros(1, d)
        else:
            last_mean, last_var = self.mean_m_, self.var_m_
        # _incremental_mean_and_var
        new_sum = k.colsum(Xb)
        total = seen + n
        last_sum = k.ew("scale", last_mean, s=float(seen))
        upd_mean = k.ew("scale", k.ew("add", last_sum, new_sum), s=1.0 / total)
        bmean = k.ew("scale", new_sum, s=1.0 / n)
        new_unnorm = k.colsum(k.ew("sqdiff", Xb, bmean))
        if seen == 0:
            upd_unnorm = new_unnorm
        else:
            last_over_new = seen / n
            last_unnorm = k.ew("scale", last_var, s=float(seen))
            t = k.ew("sub", k.ew("scale", last_sum, s=1.0 / last_over_new), new_sum)
            corr = k.ew("scale", k.ew("sq", t), s=last_over_new / total)
            upd_unnorm = k.ew("add", k.ew("add", last_unnorm, new_unnorm), corr)
        upd_var = k.ew("scale", upd_unnorm, s=1.0 / total)
        # the stacked matrix
        if seen == 0:
            Z = k.ew("sub", Xb, upd_mean)
        else:
            Xc = k.ew("sub", Xb, bmean)
            mc = k.ew("scale", k.ew("sub", self.mean_m_, bmean), s=math.sqrt((seen / total) * n))
            prev = k.ew("mul", self.components_m_, self.singular_values_m_.T)
            Z = _vstack(prev, Xc, mc)
        S, Vt = _gram_svd(k, Z)
        ev = k.ew("scale", k.ew("sq", S), s=1.0 / (total - 1))
        tot_var = k.total(k.ew("scale", upd_var, s=float(total)))
        evr = k.ew("div", k.ew("sq", S), tot_var)
        self.n_samples_seen_ = total
        self.components_m_ = Vt.rows(0, nc)
        self.singular_values_m_ = S.cols(0, nc)
        self.mean_m_, self.var_m_ = upd_mean, upd_var
        self.explained_variance_m_ = ev.cols(0, nc)
        if nc < min(d, Z.r):
            rest = ev.cols(nc, d)
            nv = k.ew("scale", k.total(rest), s=1.0 / rest.c)
        else:
            nv = _M.zeros(1, 1)
        self.n_components_ = nc
        self.n_features_in_ = d
        self.components_ = self.components_m_.out()
        self.singular_values_ = self.singular_values_m_.out((nc,))
        self.mean_ = upd_mean.out((d,))
        self.var_ = upd_var.out((d,))
        self.explained_variance_ = self.explained_variance_m_.out((nc,))
        self.explained_variance_ratio_ = evr.cols(0, nc).out((nc,))
        self.noise_variance_ = nv.s[0]

    def transform(self, X):
        self._check()
        k = self._kit()
        M = _M.from_input(X)
        if M.c != self.n_features_in_:
            raise ValueError(f"X has {M.c} features, but IncrementalPCA is expecting {self.n_features_in_}")
        Y = k.mm(k.ew("sub", M, self.mean_m_), self.components_m_, tb=True)
        if self.whiten:
            Y = k.ew("div", Y, k.ew("sqrt", self.explained_variance_m_))
        return Y.out()

    def inverse_transform(self, X):
        self._check()
        k = self._kit()
        Y = _M.from_input(X)
        C = self.components_m_
        if self.whiten:
            C = k.ew("mul", C, k.ew("sqrt", self.explained_variance_m_).T)
        return k.ew("add", k.mm(Y, C), self.mean_m_).out()


# ================================================================ random projection
def johnson_lindenstrauss_min_dim(n_samples, *, eps=0.1):
    """sklearn `random_projection.johnson_lindenstrauss_min_dim` (scalar form)."""
    if not 0 < eps < 1:
        raise ValueError("The JL bound is defined for eps in ]0, 1[")
    if n_samples <= 0:
        raise ValueError("The JL bound is defined for n_samples greater than zero")
    denominator = (eps ** 2 / 2) - (eps ** 3 / 3)
    return int(4 * math.log(n_samples) / denominator)


def _seed_of(random_state):
    if random_state is None:
        return 0
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise TypeError("random_state must be None or an int (the counter-based RNG takes an integer seed)")
    return random_state


def _pinv_rows(k, C):
    """Pseudo-inverse of a full-row-rank C (k x d, k <= d): C^T (C C^T)^-1,
    through the LU solve of the k x k Gram (d x k result)."""
    G = k.mm(C, C, tb=True)
    lu, piv, info = k.lu(G)
    if info:
        raise ValueError("compute_inverse_components: the components are rank deficient")
    return k.lu_solve(lu, piv, C).T


class _RandomProjection(_Base):
    """sklearn `random_projection.py::BaseRandomProjection`. The matrix is
    drawn from the lane's counter-based Philox stream (x_decomp/cells.mojo
    `rand_cell`), not numpy's generator: the same `random_state` gives the
    same matrix on every box, and a different one than sklearn's."""

    def fit(self, X, y=None):
        self.numeric_mode_ = _mode(self.numeric_mode)
        M = _M.from_input(X)
        n, d = M.r, M.c
        if self.n_components == "auto":
            kc = johnson_lindenstrauss_min_dim(n, eps=self.eps)
            if kc <= 0:
                raise ValueError(f"eps={self.eps} and n_samples={n} lead to a target dimension of {kc} which is invalid")
            if kc > d:
                raise ValueError(f"eps={self.eps} and n_samples={n} lead to a target dimension of {kc} which is larger than the original space with n_features={d}")
        else:
            kc = int(self.n_components)
            if kc <= 0:
                raise ValueError(f"n_components must be greater than 0, got {kc}")
        k = self._kit()
        self.n_components_ = kc
        self.n_features_in_ = d
        self.components_m_ = self._make(k, kc, d, _seed_of(self.random_state))
        self.components_ = self.components_m_.out()
        if self.compute_inverse_components:
            self.inverse_m_ = _pinv_rows(k, self.components_m_)
            self.inverse_components_ = self.inverse_m_.out()
        return self

    def transform(self, X):
        self._check()
        M = _M.from_input(X)
        if M.c != self.n_features_in_:
            raise ValueError(f"X has {M.c} features, but {type(self).__name__} is expecting {self.n_features_in_}")
        return self._kit().mm(M, self.components_m_, tb=True).out()

    def inverse_transform(self, X):
        self._check()
        if not hasattr(self, "inverse_m_"):
            raise ValueError("inverse_transform needs compute_inverse_components=True")
        M = _M.from_input(X)
        return self._kit().mm(M, self.inverse_m_, tb=True).out()


class GaussianRandomProjection(_RandomProjection):
    """sklearn.random_projection.GaussianRandomProjection: components drawn
    N(0, 1/n_components) (Box-Muller on the Philox stream)."""
    _parameters = ("n_components", "eps", "compute_inverse_components", "random_state", "numeric_mode")

    def __init__(self, n_components="auto", *, eps=0.1, compute_inverse_components=False, random_state=None,
                 numeric_mode=None):
        self.n_components, self.eps = n_components, eps
        self.compute_inverse_components, self.random_state = compute_inverse_components, random_state
        self.numeric_mode = numeric_mode

    def _make(self, k, kc, d, seed):
        return k.ew("scale", k.rand(kc, d, seed, 1, 1), s=1.0 / math.sqrt(kc))


class SparseRandomProjection(_RandomProjection):
    """sklearn.random_projection.SparseRandomProjection (Achlioptas / Li et
    al.): each entry is +-sqrt(1/density)/sqrt(n_components) with probability
    density/2 each, else 0. `density='auto'` is 1/sqrt(n_features). The
    components are held DENSE (`dense_output` only chooses sklearn's output
    container; the values are the same)."""
    _parameters = ("n_components", "density", "eps", "dense_output", "compute_inverse_components",
                   "random_state", "numeric_mode")

    def __init__(self, n_components="auto", *, density="auto", eps=0.1, dense_output=False,
                 compute_inverse_components=False, random_state=None, numeric_mode=None):
        self.n_components, self.density, self.eps, self.dense_output = n_components, density, eps, dense_output
        self.compute_inverse_components, self.random_state = compute_inverse_components, random_state
        self.numeric_mode = numeric_mode

    def _make(self, k, kc, d, seed):
        dens = 1.0 / math.sqrt(d) if self.density == "auto" else float(self.density)
        if not 0 < dens <= 1:
            raise ValueError(f"Expected density in range ]0, 1], got: {dens}")
        self.density_ = dens
        u = k.rand(kc, d, seed, 2, 0)
        sgn = k.ew("scale", k.rand(kc, d, seed, 3, 2), s=math.sqrt(1.0 / dens) / math.sqrt(kc))
        if dens == 1:
            return sgn
        # u < density keeps the signed value, else 0 (select: x > s -> y else z)
        return k.ew("select", u, _M.zeros(1, 1), sgn, s=dens - 2.0 ** -25)


# ================================================================ graph helpers
def _knn_order(D, i, k, include_self=True):
    """The k smallest entries of row i of a distance matrix, ascending, ties
    broken by the LOWER column index (comparisons of float32 values only)."""
    row = D.row(i)
    idx = sorted(range(len(row)), key=lambda j: (row[j], j))
    if not include_self:
        idx = [j for j in idx if j != i]
    return idx[:k]


def _knn_connectivity(k, X, n_neighbors, include_self=True):
    """sklearn `kneighbors_graph(mode='connectivity')` as a dense n x n 0/1 matrix."""
    D = k.sqdist(X, X)
    n = X.r
    A = array.array("f", bytes(4 * n * n))
    for i in range(n):
        for j in _knn_order(D, i, n_neighbors, include_self):
            A[i * n + j] = 1.0
    return _M(A, n, n), D


def _symmetrize(k, A):
    """0.5 * (A + A^T)."""
    return k.ew("scale", k.ew("add", A, A.T), s=0.5)


def _normed_laplacian(k, A):
    """scipy `csgraph.laplacian(normed=True, return_diag=True)` on a dense
    adjacency: the diagonal is ignored, dd = sqrt(degree) (1 for an isolated
    node), L = I - D^-1/2 A D^-1/2 with the diagonal set to 1 (sklearn
    `_set_diag`)."""
    n = A.r
    A0 = A.copy()
    for i in range(n):
        A0.s[i * n + i] = 0.0
    deg = k.rowsum(A0)
    dd = k.ew("sqrt", deg)
    dd = _M.of([v if v > 0 else 1.0 for v in dd.s], n, 1)
    scaled = k.ew("div", k.ew("div", A0, dd), dd.T)
    L = k.ew("scale", scaled, s=-1.0)
    for i in range(n):
        L.s[i * n + i] = 1.0
    return L, dd


def _sign_flip_rows(U):
    """sklearn `_deterministic_vector_sign_flip`: each row signed so its
    largest-|.| entry (first on a tie) is positive."""
    return _svd_flip_v(U)


# ================================================================ thin SVD
def _thin_svd(k, X, nc, u_based=True):
    """(U n x nc, S 1 x nc, Vt nc x d) of X through the eigendecomposition of
    the smaller Gram matrix, singular values descending, then sklearn's
    `svd_flip` (u_based: the largest-|.| entry of each COLUMN of U positive;
    else of each ROW of Vt). U (or Vt) is recovered as X V / S (X^T U / S);
    a zero singular value gives a zero vector."""
    n, d = X.r, X.c
    if d <= n:
        S, Vt = _gram_svd(k, X)
        S, Vt = S.cols(0, nc), Vt.rows(0, nc)
        U = k.ew("div", k.mm(X, Vt, tb=True), S)
    else:
        S, Ut = _gram_svd(k, X.T)
        S, Ut = S.cols(0, nc), Ut.rows(0, nc)
        U = Ut.T
        Vt = k.ew("div", k.mm(U, X, ta=True), S.T)
    if u_based:
        Ut = U.T
        fl = []
        for i in range(Ut.r):
            row = Ut.row(i)
            best, arg = -1.0, 0
            for j, v in enumerate(row):
                if abs(v) > best:
                    best, arg = abs(v), j
            fl.append(row[arg] < 0)
        U, Vt = U.neg_cols(fl), Vt.neg_rows(fl)
    else:
        fl = []
        for i in range(Vt.r):
            row = Vt.row(i)
            best, arg = -1.0, 0
            for j, v in enumerate(row):
                if abs(v) > best:
                    best, arg = abs(v), j
            fl.append(row[arg] < 0)
        U, Vt = U.neg_cols(fl), Vt.neg_rows(fl)
    return U, S, Vt


def _norm(k, v):
    return k.ew("sqrt", k.total(k.ew("sq", v))).s[0]


# ================================================================ NMF
_F32_EPS = 1.1920928955078125e-07


class NMF(_Base):
    """sklearn.decomposition.NMF (reference: scikit-learn
    `decomposition/_nmf.py`: `_initialize_nmf`, `_fit_multiplicative_update`
    with `_multiplicative_update_w/_h` for beta_loss='frobenius', and
    `_fit_coordinate_descent` with `_cdnmf_fast.pyx::_update_cdnmf_fast`).

    solver 'cd' (the default) and 'mu'; init 'random' (the lane's Philox
    stream, not numpy's), 'nndsvd', 'nndsvda', 'nndsvdar' (the SVD is exact,
    through the Gram eigh, where sklearn calls randomized_svd), 'custom'.
    REFUSED BY NAME: beta_loss other than 'frobenius', shuffle=True."""
    _parameters = ("n_components", "init", "solver", "beta_loss", "tol", "max_iter", "random_state",
                   "alpha_W", "alpha_H", "l1_ratio", "verbose", "shuffle", "numeric_mode")

    def __init__(self, n_components="auto", *, init=None, solver="cd", beta_loss="frobenius", tol=1e-4,
                 max_iter=200, random_state=None, alpha_W=0.0, alpha_H="same", l1_ratio=0.0, verbose=0,
                 shuffle=False, numeric_mode=None):
        self.n_components, self.init, self.solver, self.beta_loss = n_components, init, solver, beta_loss
        self.tol, self.max_iter, self.random_state = tol, max_iter, random_state
        self.alpha_W, self.alpha_H, self.l1_ratio = alpha_W, alpha_H, l1_ratio
        self.verbose, self.shuffle, self.numeric_mode = verbose, shuffle, numeric_mode

    # ---- setup
    def _validate(self, M):
        if self.beta_loss not in ("frobenius", 2, 2.0):
            raise ValueError("beta_loss other than 'frobenius' is not carried")
        if self.solver not in ("cd", "mu"):
            raise ValueError(f"Invalid solver parameter: got {self.solver!r} instead of one of {{'cd', 'mu'}}")
        if self.shuffle:
            raise ValueError("shuffle=True is not carried (the coordinate order is the identity)")
        for v in M.s:
            if v < 0:
                raise ValueError("Negative values in data passed to NMF (input X)")

    def _reg(self, n, d):
        aH = self.alpha_W if self.alpha_H == "same" else self.alpha_H
        aW = self.alpha_W
        return (d * aW * self.l1_ratio, n * aH * self.l1_ratio,
                d * aW * (1.0 - self.l1_ratio), n * aH * (1.0 - self.l1_ratio))

    def _init(self, k, M, nc, init):
        n, d = M.r, M.c
        seed = _seed_of(self.random_state)
        xmean = k.ew("scale", k.total(M), s=1.0 / (n * d)).s[0]
        if init == "random":
            avg = math.sqrt(xmean / nc)
            H = k.ew("scale", k.ew("abs", k.rand(nc, d, seed, 10, 1)), s=avg)
            W = k.ew("scale", k.ew("abs", k.rand(n, nc, seed, 11, 1)), s=avg)
            return W, H
        U, S, Vt = _thin_svd(k, M, nc, u_based=(n >= d))
        Wc, Hr = [], []
        for j in range(nc):
            x, y = U.cols(j, j + 1), Vt.rows(j, j + 1)
            sj = S.s[j]
            if j == 0:
                r = math.sqrt(sj)
                Wc.append(k.ew("scale", k.ew("abs", x), s=r))
                Hr.append(k.ew("scale", k.ew("abs", y), s=r))
                continue
            xp, yp = k.ew("maxs", x, s=0.0), k.ew("maxs", y, s=0.0)
            xn, yn = k.ew("abs", k.ew("mins", x, s=0.0)), k.ew("abs", k.ew("mins", y, s=0.0))
            xpn, ypn, xnn, ynn = _norm(k, xp), _norm(k, yp), _norm(k, xn), _norm(k, yn)
            mp = _f32(_f32(xpn) * _f32(ypn))
            mn = _f32(_f32(xnn) * _f32(ynn))
            if mp > mn:
                u, v, sigma = k.ew("scale", xp, s=1.0 / xpn if xpn else 0.0), k.ew("scale", yp, s=1.0 / ypn if ypn else 0.0), mp
            else:
                u, v, sigma = k.ew("scale", xn, s=1.0 / xnn if xnn else 0.0), k.ew("scale", yn, s=1.0 / ynn if ynn else 0.0), mn
            lbd = math.sqrt(_f32(sj * sigma))
            Wc.append(k.ew("scale", u, s=lbd))
            Hr.append(k.ew("scale", v, s=lbd))
        W, H = _hstack(*Wc), _vstack(*Hr)
        z = _M.zeros(1, 1)
        W = k.ew("select", W, W, z, s=1e-6 - 1e-13)
        H = k.ew("select", H, H, z, s=1e-6 - 1e-13)
        if init == "nndsvda":
            W = k.ew("select", W, W, k.const(xmean), s=0.0)
            H = k.ew("select", H, H, k.const(xmean), s=0.0)
        elif init == "nndsvdar":
            rw = k.ew("scale", k.ew("abs", k.rand(n, nc, seed, 12, 1)), s=abs(xmean) / 100)
            rh = k.ew("scale", k.ew("abs", k.rand(nc, d, seed, 13, 1)), s=abs(xmean) / 100)
            W = k.ew("select", W, W, rw, s=0.0)
            H = k.ew("select", H, H, rh, s=0.0)
        return W, H

    def _err(self, k, M, W, H):
        return k.ew("sqrt", k.total(k.ew("sqdiff", M, k.mm(W, H)))).s[0]

    # ---- solvers
    def _mu(self, k, M, W, H, update_H, regs):
        l1W, l1H, l2W, l2H = regs
        err0 = self._err(k, M, W, H)
        prev = err0
        it = 0
        for it in range(1, self.max_iter + 1):
            num = k.mm(M, H, tb=True)
            den = k.mm(W, k.mm(H, H, tb=True))
            if l1W > 0:
                den = k.ew("adds", den, s=l1W)
            if l2W > 0:
                den = k.ew("axpy", den, W, s=l2W)
            W = k.ew("muz", W, num, den, s=_F32_EPS)
            if update_H:
                num = k.mm(W, M, ta=True)
                den = k.mm(k.mm(W, W, ta=True), H)
                if l1H > 0:
                    den = k.ew("adds", den, s=l1H)
                if l2H > 0:
                    den = k.ew("axpy", den, H, s=l2H)
                H = k.ew("muz", H, num, den, s=_F32_EPS)
            if self.tol > 0 and it % 10 == 0:
                err = self._err(k, M, W, H)
                if (prev - err) / err0 < self.tol:
                    break
                prev = err
        return W, H, it

    def _cd_side(self, k, M, W, Ht, l1, l2, perm, trans):
        HHt = k.mm(Ht, Ht, ta=True)
        XHt = k.mm(M, Ht, ta=trans)
        if l2:
            HHt = HHt.copy()
            for t in range(HHt.r):
                HHt.s[t * HHt.c + t] = _f32(HHt.s[t * HHt.c + t] + l2)
        if l1:
            XHt = k.ew("adds", XHt, s=-l1)
        return k.cd_rows(W, HHt, XHt, perm)

    def _cd(self, k, M, W, H, update_H, regs):
        l1W, l1H, l2W, l2H = regs
        perm = list(range(W.c))
        Ht = H.T
        W = W.copy()
        v_init = None
        it = 0
        for it in range(1, self.max_iter + 1):
            viol = self._cd_side(k, M, W, Ht, l1W, l2W, perm, False)
            if update_H:
                viol += self._cd_side(k, M, Ht, W, l1H, l2H, perm, True)
            if v_init is None:
                v_init = viol
            if v_init == 0:
                break
            if viol / v_init <= self.tol:
                break
        return W, Ht.T if update_H else H, it

    def _fit_transform(self, M, H=None, update_H=True):
        k = self._kit()
        n, d = M.r, M.c
        regs = self._reg(n, d)
        if update_H:
            nc = self.n_components
            if nc in (None, "auto"):
                nc = d if self.init != "custom" else None
            nc = int(nc)
            init = self.init or ("nndsvda" if nc <= min(n, d) else "random")
            if init not in ("random", "nndsvd", "nndsvda", "nndsvdar"):
                raise ValueError(f"init={init!r} is not supported here (random, nndsvd, nndsvda, nndsvdar)")
            if init.startswith("nndsvd") and nc > min(n, d):
                raise ValueError("init = 'nndsvd' can only be used when n_components <= min(n_samples, n_features)")
            W, H = self._init(k, M, nc, init)
        else:
            nc = H.r
            if self.solver == "mu":
                xmean = k.ew("scale", k.total(M), s=1.0 / (n * d)).s[0]
                W = k.const(math.sqrt(xmean / nc), n, nc)
            else:
                W = _M.zeros(n, nc)
        solve = self._cd if self.solver == "cd" else self._mu
        W, H, it = solve(k, M, W, H, update_H, regs)
        return W, H, it, nc

    def fit_transform(self, X, y=None, W=None, H=None):
        self.numeric_mode_ = _mode(self.numeric_mode)
        M = _M.from_input(X)
        self._validate(M)
        if self.init == "custom":
            k = self._kit()
            Wm, Hm = _M.from_input(W, "W"), _M.from_input(H, "H")
            regs = self._reg(M.r, M.c)
            solve = self._cd if self.solver == "cd" else self._mu
            Wm, Hm, it = solve(k, M, Wm, Hm, True, regs)
            nc = Hm.r
        else:
            Wm, Hm, it, nc = self._fit_transform(M)
        self.components_m_ = Hm
        self.components_ = Hm.out()
        self.n_components_ = nc
        self.n_iter_ = it
        self.n_features_in_ = M.c
        self.reconstruction_err_ = self._err(self._kit(), M, Wm, Hm)
        return Wm.out()

    def fit(self, X, y=None, **params):
        self.fit_transform(X, **params)
        return self

    def transform(self, X):
        self._check()
        M = _M.from_input(X)
        if M.c != self.n_features_in_:
            raise ValueError(f"X has {M.c} features, but NMF is expecting {self.n_features_in_}")
        self._validate(M)
        W, _, _, _ = self._fit_transform(M, H=self.components_m_, update_H=False)
        return W.out()

    def inverse_transform(self, X):
        self._check()
        return self._kit().mm(_M.from_input(X, "W"), self.components_m_).out()


# ================================================================ FastICA
def _sym_decorrelation(k, W):
    """sklearn `_fastica.py::_sym_decorrelation`: (W W^T)^(-1/2) W through
    eigh, eigenvalues clipped at float32 tiny."""
    w, u = k.eigh(k.mm(W, W, tb=True))
    w = k.ew("maxs", w, s=1.1754943508222875e-38)
    ui = k.ew("mul", u, k.ew("recip", k.ew("sqrt", w)))
    return k.mm(k.mm(ui, u, tb=True), W)


class FastICA(_Base):
    """sklearn.decomposition.FastICA (reference: scikit-learn
    `decomposition/_fastica.py`: `_fit_transform`, `_ica_par`, `_ica_def`,
    `_sym_decorrelation`, `_gs_decorrelation`, `_logcosh`, `_exp`, `_cube`).

    The whitening SVD is the eigendecomposition of X^T X (both of sklearn's
    whiten_solver values take this route); a singular value below 10 float32
    eps is clamped there, as sklearn's 'eigh' solver clamps it. w_init
    defaults to a draw from the lane's Philox stream, not numpy's.
    REFUSED BY NAME: a callable `fun`."""
    _parameters = ("n_components", "algorithm", "whiten", "fun", "fun_args", "max_iter", "tol", "w_init",
                   "whiten_solver", "random_state", "numeric_mode")

    def __init__(self, n_components=None, *, algorithm="parallel", whiten="unit-variance", fun="logcosh",
                 fun_args=None, max_iter=200, tol=1e-4, w_init=None, whiten_solver="svd", random_state=None,
                 numeric_mode=None):
        self.n_components, self.algorithm, self.whiten, self.fun = n_components, algorithm, whiten, fun
        self.fun_args, self.max_iter, self.tol, self.w_init = fun_args, max_iter, tol, w_init
        self.whiten_solver, self.random_state, self.numeric_mode = whiten_solver, random_state, numeric_mode

    def _g(self, k, Y):
        if self.fun == "logcosh":
            alpha = (self.fun_args or {}).get("alpha", 1.0)
            if not 1 <= alpha <= 2:
                raise ValueError("alpha must be in [1,2]")
            gx = k.ew("tanh", k.ew("scale", Y, s=alpha))
            gp = k.ew("scale", k.ew("onemsq", gx), s=alpha)
        elif self.fun == "exp":
            gx, gp = k.ew("expg", Y), k.ew("expgp", Y)
        elif self.fun == "cube":
            gx, gp = k.ew("cube", Y), k.ew("cubep", Y)
        else:
            raise ValueError("fun must be 'logcosh', 'exp' or 'cube' (a callable is not carried)")
        return gx, k.ew("scale", k.rowsum(gp), s=1.0 / Y.c)

    def _par(self, k, X1, W):
        W = _sym_decorrelation(k, W)
        p = X1.c
        it = 0
        for it in range(1, self.max_iter + 1):
            gx, gp = self._g(k, k.mm(W, X1))
            W1 = _sym_decorrelation(k, k.ew("sub", k.ew("scale", k.mm(gx, X1, tb=True), s=1.0 / p),
                                            k.ew("mul", W, gp)))
            dots = k.rowsum(k.ew("mul", W1, W))
            lim = max(k.ew("abs", k.ew("adds", k.ew("abs", dots), s=-1.0)).s)
            W = W1
            if lim < self.tol:
                break
        return W, it

    def _def(self, k, X1, Winit):
        nc = Winit.r
        p = X1.c
        rows = []
        its = []
        for j in range(nc):
            w = Winit.rows(j, j + 1)
            if rows:
                Wp = _vstack(*rows)
                w = k.ew("sub", w, k.mm(k.mm(w, Wp, tb=True), Wp))
            w = k.ew("scale", w, s=1.0 / _norm(k, w) if _norm(k, w) else 0.0)
            it = 0
            for it in range(1, self.max_iter + 1):
                gx, gp = self._g(k, k.mm(w, X1))
                w1 = k.ew("sub", k.ew("scale", k.mm(gx, X1, tb=True), s=1.0 / p), k.ew("mul", w, gp))
                if rows:
                    Wp = _vstack(*rows)
                    w1 = k.ew("sub", w1, k.mm(k.mm(w1, Wp, tb=True), Wp))
                nw = _norm(k, w1)
                w1 = k.ew("scale", w1, s=1.0 / nw if nw else 0.0)
                lim = abs(abs(k.total(k.ew("mul", w1, w)).s[0]) - 1)
                w = w1
                if lim < self.tol:
                    break
            its.append(it)
            rows.append(w)
        return _vstack(*rows), max(its)

    def fit_transform(self, X, y=None):
        return self._fit(X, True).out()

    def fit(self, X, y=None):
        self._fit(X, False)
        return self

    def _fit(self, X, sources):
        self.numeric_mode_ = _mode(self.numeric_mode)
        k = self._kit()
        M = _M.from_input(X)
        n, d = M.r, M.c
        if self.algorithm not in ("parallel", "deflation"):
            raise ValueError("algorithm must be 'parallel' or 'deflation'")
        whiten = self.whiten
        if whiten not in (False, "unit-variance", "arbitrary-variance"):
            raise ValueError(f"whiten={whiten!r} is not supported")
        nc = self.n_components
        if not whiten and nc is not None:
            nc = None
        if nc is None:
            nc = min(n, d)
        if nc > min(n, d):
            nc = min(n, d)
        if whiten:
            mean = k.colmean(M)
            Xc = k.ew("sub", M, mean)
            ev, u = k.eigh(k.mm(Xc, Xc, ta=True))
            order = list(range(d - 1, -1, -1))
            ev = k.ew("maxs", ev.take_cols(order), s=_F32_EPS * 10)
            sv = k.ew("sqrt", ev)
            u = u.take_cols(order)
            u = u.neg_cols([v < 0 for v in u.row(0)])
            K = k.ew("div", u, sv).T.rows(0, nc)
            X1 = k.ew("scale", k.mm(K, Xc, tb=True), s=math.sqrt(n))
        else:
            Xc = M
            X1 = M.T
        if self.w_init is None:
            Winit = k.rand(nc, nc, _seed_of(self.random_state), 20, 1)
        else:
            Winit = _M.from_input(self.w_init, "w_init")
            if (Winit.r, Winit.c) != (nc, nc):
                raise ValueError(f"w_init has invalid shape -- should be {(nc, nc)}")
        W, it = (self._par if self.algorithm == "parallel" else self._def)(k, X1, Winit)
        self.n_iter_ = it
        S = None
        if whiten:
            WK = k.mm(W, K)
            S = k.mm(Xc, WK, tb=True)
            if whiten == "unit-variance":
                smean = k.colmean(S)
                std = k.ew("sqrt", k.ew("scale", k.colsum(k.ew("sqdiff", S, smean)), s=1.0 / n))
                S = k.ew("div", S, std)
                W = k.ew("div", W, std.T)
            comp = k.mm(W, K)
            self.whitening_m_ = K
            self.whitening_ = K.out()
            self.mean_m_ = mean
            self.mean_ = mean.out((d,))
        else:
            S = k.mm(M, W, tb=True)
            comp = W
        self.components_m_ = comp
        self.components_ = comp.out()
        self.mixing_m_ = _pinv_rows(k, comp)
        self.mixing_ = self.mixing_m_.out()
        self._whiten = whiten
        self.n_features_in_ = d
        return S

    def transform(self, X, copy=True):
        self._check()
        k = self._kit()
        M = _M.from_input(X)
        if self._whiten:
            M = k.ew("sub", M, self.mean_m_)
        return k.mm(M, self.components_m_, tb=True).out()

    def inverse_transform(self, X, copy=True):
        self._check()
        k = self._kit()
        Y = k.mm(_M.from_input(X), self.mixing_m_, tb=True)
        if self._whiten:
            Y = k.ew("add", Y, self.mean_m_)
        return Y.out()


# ================================================================ FactorAnalysis
_LOG_2PI = 1.8378770664093453


def _dsum(values):
    """Sequential float64 sum (IEEE adds, ascending): the same on every box."""
    t = 0.0
    for v in values:
        t += v
    return t


def _inv(k, A):
    """A^-1 through the LU solve against the identity (getrf + getrs)."""
    lu, piv, info = k.lu(A)
    if info:
        raise ValueError("singular matrix")
    return k.lu_solve(lu, piv, _eye(A.r))


def _eye(n):
    E = _M.zeros(n, n)
    for i in range(n):
        E.s[i * n + i] = 1.0
    return E


def _logdet(k, A):
    """log|det A| from the LU diagonal (sum ascending); sign ignored."""
    lu, piv, info = k.lu(A)
    diag = _M.of([lu.s[i * A.r + i] for i in range(A.r)], 1, A.r)
    return k.total(k.ew("logs", k.ew("abs", diag), s=1.1754943508222875e-38)).s[0]


def _polar(k, A):
    """U V^T of the SVD of a square A, and the sum of its singular values:
    A V S^-1 V^T through the eigh of A^T A."""
    w, V = k.eigh(k.mm(A, A, ta=True))
    sv = k.ew("sqrt", w)
    AV = k.ew("div", k.mm(A, V), sv)
    return k.mm(AV, V, tb=True), k.total(sv).s[0]


def _ortho_rotation(k, C, method, tol=1e-6, max_iter=100):
    """sklearn `_factor_analysis.py::_ortho_rotation`; C is n_features x n_components."""
    nrow, ncol = C.r, C.c
    R = _eye(ncol)
    var = 0.0
    for _ in range(max_iter):
        cr = k.mm(C, R)
        if method == "varimax":
            tmp = k.ew("mul", cr, k.ew("scale", k.colsum(k.ew("sq", cr)), s=1.0 / nrow))
            target = k.ew("sub", k.ew("cube", cr), tmp)
        else:
            target = k.ew("cube", cr)
        R, var_new = _polar(k, k.mm(C, target, ta=True))
        if var != 0 and var_new < var * (1 + tol):
            break
        var = var_new
    return k.mm(C, R).T


class FactorAnalysis(_Base):
    """sklearn.decomposition.FactorAnalysis (reference: scikit-learn
    `decomposition/_factor_analysis.py`: `fit`, `transform`,
    `get_covariance`, `get_precision`, `score_samples`, `_ortho_rotation`).

    Each iteration's SVD of the scaled data is the eigh of its d x d Gram
    matrix: the squared singular values sklearn uses ARE its eigenvalues.
    svd_method='randomized' takes the same exact route (sklearn's randomized
    SVD is an approximation of it). `rotation` in {None, 'varimax',
    'quartimax'}."""
    _parameters = ("n_components", "tol", "copy", "max_iter", "noise_variance_init", "svd_method",
                   "iterated_power", "rotation", "random_state", "numeric_mode")

    def __init__(self, n_components=None, *, tol=1e-2, copy=True, max_iter=1000, noise_variance_init=None,
                 svd_method="randomized", iterated_power=3, rotation=None, random_state=0, numeric_mode=None):
        self.n_components, self.tol, self.copy, self.max_iter = n_components, tol, copy, max_iter
        self.noise_variance_init, self.svd_method, self.iterated_power = noise_variance_init, svd_method, iterated_power
        self.rotation, self.random_state, self.numeric_mode = rotation, random_state, numeric_mode

    def fit(self, X, y=None):
        self.numeric_mode_ = _mode(self.numeric_mode)
        if self.svd_method not in ("lapack", "randomized"):
            raise ValueError("svd_method must be 'lapack' or 'randomized'")
        if self.rotation not in (None, "varimax", "quartimax"):
            raise ValueError("rotation must be None, 'varimax' or 'quartimax'")
        k = self._kit()
        M = _M.from_input(X)
        n, d = M.r, M.c
        nc = self.n_components or d
        mean = k.colmean(M)
        Xc = k.ew("sub", M, mean)
        nsqrt = math.sqrt(n)
        llconst = d * _LOG_2PI + nc
        var = k.ew("scale", k.colsum(k.ew("sq", Xc)), s=1.0 / n)
        if self.noise_variance_init is None:
            psi = k.const(1.0, 1, d)
        else:
            psi = _M.of([float(v) for v in self.noise_variance_init], 1, len(self.noise_variance_init))
            if psi.c != d:
                raise ValueError(f"noise_variance_init dimension does not match the number of features : {psi.c} != {d}")
        SMALL = 1e-12
        old_ll = -math.inf
        loglike = []
        it = 0
        W = None
        for it in range(1, self.max_iter + 1):
            sqrt_psi = k.ew("adds", k.ew("sqrt", psi), s=SMALL)
            Z = k.ew("scale", k.ew("div", Xc, sqrt_psi), s=1.0 / nsqrt)
            ev, V = k.eigh(k.mm(Z, Z, ta=True))
            order = list(range(d - 1, -1, -1))
            s2 = k.ew("maxs", ev.take_cols(order), s=0.0)
            Vt = V.take_cols(order).T.rows(0, nc)
            sk = s2.cols(0, nc)
            # the log-likelihood is accumulated in Python float64 (IEEE adds,
            # sequential): at float32 its step falls under tol=1e-2 early
            unexp = _dsum(s2.cols(nc, d).s) if nc < d else 0.0
            W = k.ew("mul", Vt, k.ew("sqrt", k.ew("maxs", k.ew("adds", sk, s=-1.0), s=0.0)).T)
            W = k.ew("mul", W, sqrt_psi)
            slog = _dsum(k.ew("logs", sk, s=1.1754943508222875e-38).s)
            plog = _dsum(k.ew("logs", psi, s=1.1754943508222875e-38).s)
            ll = (llconst + slog + unexp + plog) * (-n / 2.0)
            loglike.append(ll)
            if (ll - old_ll) < self.tol:
                break
            old_ll = ll
            psi = k.ew("maxs", k.ew("sub", var, k.colsum(k.ew("sq", W))), s=SMALL)
        if self.rotation is not None:
            W = _ortho_rotation(k, W.T, self.rotation).rows(0, nc)
        self.components_m_ = W
        self.components_ = W.out()
        self.noise_variance_m_ = psi
        self.noise_variance_ = psi.out((d,))
        self.mean_m_ = mean
        self.mean_ = mean.out((d,))
        self.loglike_ = loglike
        self.n_iter_ = it
        self.n_features_in_ = d
        return self

    def transform(self, X):
        self._check()
        k = self._kit()
        M = k.ew("sub", _M.from_input(X), self.mean_m_)
        W = self.components_m_
        Wpsi = k.ew("div", W, self.noise_variance_m_)
        cov_z = _inv(k, k.ew("add", _eye(W.r), k.mm(Wpsi, W, tb=True)))
        return k.mm(k.mm(M, Wpsi, tb=True), cov_z).out()

    def _cov(self, k):
        W = self.components_m_
        C = k.mm(W, W, ta=True)
        d = C.r
        for i in range(d):
            C.s[i * d + i] = _f32(C.s[i * d + i] + self.noise_variance_m_.s[i])
        return C

    def get_covariance(self):
        self._check()
        return self._cov(self._kit()).out()

    def get_precision(self):
        self._check()
        k = self._kit()
        return _inv(k, self._cov(k)).out()

    def score_samples(self, X):
        v = self._ss(X)
        return v.out((v.r,))

    def _ss(self, X):
        self._check()
        k = self._kit()
        Xr = k.ew("sub", _M.from_input(X), self.mean_m_)
        C = self._cov(k)
        P = _inv(k, C)
        ld = _logdet(k, P)
        q = k.rowsum(k.ew("mul", Xr, k.mm(Xr, P)))
        return k.ew("adds", k.ew("scale", q, s=-0.5), s=-0.5 * (Xr.c * _LOG_2PI - ld))

    def score(self, X, y=None):
        k = self._kit()
        v = self._ss(X)
        return k.ew("scale", k.total(v), s=1.0 / v.r).s[0]


# ================================================================ LU
def lu_factor(a, *, numeric_mode=None):
    """scipy.linalg.lu_factor (LAPACK getrf semantics): `(lu, piv)` with L
    unit-lower and U in one n x n float32 matrix and `piv` the 0-based row
    interchanges, applied in order. Partial pivoting on the largest |a[i, k]|,
    ties broken by the LOWEST row index. A zero pivot is kept (getrf's
    info > 0) and warned about, as scipy warns."""
    k = _Kit(_mode(numeric_mode))
    A = _M.from_input(a, "a")
    if A.r != A.c:
        raise ValueError(f"expected a square matrix, got {A.r} x {A.c}")
    lu, piv, info = k.lu(A)
    if info:
        import warnings
        warnings.warn(f"Diagonal number {info} is exactly zero. Singular matrix.", RuntimeWarning, stacklevel=2)
    return lu.out(), frombytes(piv.tobytes(), "<i4", (A.r,))


def lu_solve(lu_and_piv, b, *, trans=0, numeric_mode=None):
    """scipy.linalg.lu_solve (LAPACK getrs, trans=0 only): solve A x = b
    from `lu_factor`'s pair. `b` is n or n x nrhs; a zero pivot yields 0 in
    that component (never inf or NaN). REFUSED BY NAME: trans != 0."""
    if trans != 0:
        raise NotImplementedError("lu_solve: trans != 0 is not carried")
    lu, piv = lu_and_piv
    k = _Kit(_mode(numeric_mode))
    L = _M.from_input(lu, "lu")
    n = L.r
    pa = as_i32_c(piv, ndim=1, name="piv")[0]
    pv = array.array("i")
    pv.frombytes(pa.tobytes())
    if len(pv) != n or any(not 0 <= p < n for p in pv):
        raise ValueError("piv must hold one row index in [0, n) per row")
    vec = len(getattr(b, "shape", ())) == 1 or (not hasattr(b, "shape") and not isinstance(b[0], (list, tuple)))
    B = _M.from_input(_row_of(b), "b").T if vec else _M.from_input(b, "b")
    if B.r != n:
        raise ValueError(f"b has {B.r} rows, the factorization has {n}")
    X = k.lu_solve(L, pv, B)
    return X.out((n,)) if vec else X.out()


def solve(a, b, *, numeric_mode=None):
    """numpy.linalg.solve through lu_factor + lu_solve (gesv)."""
    return lu_solve(lu_factor(a, numeric_mode=numeric_mode), b, numeric_mode=numeric_mode)


def _row_of(b):
    """A 1-D input as a 1 x n float32 matrix input, without a float64 cast."""
    shape = getattr(b, "shape", None)
    if shape is not None and hasattr(b, "reshape"):
        return b.reshape(1, -1)
    return [list(b)]


# ================================================================ lstsq / randomized SVD
def _orthonormal_cols(k, A):
    """An orthonormal basis of A's columns (m x l, m >= l): modified
    Gram-Schmidt with one re-orthogonalization pass (x_decomp/cells.mojo
    `orth_serial`), stable where a Cholesky QR of an ill-conditioned A is not."""
    return k.orth(A)


def _flip_u(U, Vt):
    """sklearn svd_flip(u_based_decision=True): each column of U signed so its
    largest-|.| entry (first on a tie) is positive; Vt's rows follow."""
    fl = []
    Ut = U.T
    for i in range(Ut.r):
        row = Ut.row(i)
        best, arg = -1.0, 0
        for j, v in enumerate(row):
            if abs(v) > best:
                best, arg = abs(v), j
        fl.append(row[arg] < 0)
    return U.neg_cols(fl), Vt.neg_rows(fl)


def randomized_svd(M, n_components, *, n_oversamples=10, n_iter="auto", power_iteration_normalizer="auto",
                   transpose="auto", flip_sign=True, random_state=None, numeric_mode=None):
    """sklearn.utils.extmath.randomized_svd (Halko et al.; RAFT
    `linalg/rsvd.cuh` is the same scheme). The Gaussian test matrix is the
    lane's Philox stream (random_state None means 0). Every power iteration
    re-orthonormalizes by MGS2 whatever `power_iteration_normalizer`
    says: 'LU', 'QR' and 'none' span the same subspace, only the rounding of
    the basis differs. The small SVD of Q^T M is exact (Gram eigh).
    Returns (U, s, Vt)."""
    k = _Kit(_mode(numeric_mode))
    A = _M.from_input(M, "M")
    n, d = A.r, A.c
    if power_iteration_normalizer not in ("auto", "QR", "LU", "none"):
        raise ValueError("power_iteration_normalizer must be 'auto', 'QR', 'LU' or 'none'")
    nr = n_components + n_oversamples
    if n_iter == "auto":
        n_iter = 7 if n_components < 0.1 * min(n, d) else 4
    if transpose == "auto":
        transpose = n < d
    if transpose:
        A = A.T
        n, d = d, n
    if not 1 <= n_components <= min(n, d):
        raise ValueError("n_components must be in [1, min(n_samples, n_features)]")
    nr = min(nr, d, n)
    Q = k.rand(d, nr, _seed_of(random_state), 30, 1)
    for _ in range(int(n_iter)):
        Q = _orthonormal_cols(k, k.mm(A, Q))
        Q = _orthonormal_cols(k, k.mm(A, Q, ta=True))
    Q = _orthonormal_cols(k, k.mm(A, Q))
    B = k.mm(Q, A, ta=True)
    Uh, S, Vt = _thin_svd(k, B, min(B.r, B.c), u_based=True)
    U = k.mm(Q, Uh)
    if flip_sign:
        if not transpose:
            U, Vt = _flip_u(U, Vt)
        else:
            U, Vt = _flip_u(Vt.T, U.T)
            U, Vt = Vt.T, U.T
    kc = n_components
    if transpose:
        return Vt.rows(0, kc).T.out(), S.cols(0, kc).out((kc,)), U.cols(0, kc).T.out()
    return U.cols(0, kc).out(), S.cols(0, kc).out((kc,)), Vt.rows(0, kc).out()


def lstsq(a, b, rcond=None, *, numeric_mode=None):
    """numpy.linalg.lstsq: (x, residuals, rank, s) through the SVD of a
    (the Gram eigh of the smaller side, singular values descending), singular
    values at or below rcond * s_max treated as zero (rcond None: float32
    eps * max(M, N)). residuals are the squared column norms of b - a x when
    rank == N < M, else empty."""
    k = _Kit(_mode(numeric_mode))
    A = _M.from_input(a, "a")
    m, nn = A.r, A.c
    vec = len(getattr(b, "shape", ())) == 1 or (not hasattr(b, "shape") and not isinstance(b[0], (list, tuple)))
    B = _M.from_input(_row_of(b), "b").T if vec else _M.from_input(b, "b")
    if B.r != m:
        raise ValueError("Incompatible dimensions")
    r = min(m, nn)
    U, S, Vt = _thin_svd(k, A, r, u_based=True)
    if rcond is None:
        rcond = _F32_EPS * max(m, nn)
    cut = _f32(S.s[0] * rcond) if r else 0.0
    rank = sum(1 for v in S.s if v > cut)
    inv = k.ew("recip", k.ew("select", S, S, _M.zeros(1, 1), s=cut))
    X = k.mm(Vt, k.ew("mul", k.mm(U, B, ta=True), inv.T), ta=True)
    if rank == nn and m > nn:
        res = k.colsum(k.ew("sq", k.ew("sub", B, k.mm(A, X))))
        resid = res.out((B.c,))
    else:
        resid = _M.zeros(1, 0).out((0,))
    return (X.out((nn,)) if vec else X.out()), resid, rank, S.out((r,))

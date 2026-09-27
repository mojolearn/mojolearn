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
from ._buffer import as_f32_c, frombytes

__all__ = ["IncrementalPCA"]

_BINDING = "_mojolearn_x_decomp"

# x_decomp/cells.mojo op codes
_OP = dict(
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

    def __init__(self, mode):
        self.mode = mode
        self.b = _backend.binding(_BINDING, mode)

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

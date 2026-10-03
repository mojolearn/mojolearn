# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE DECOMP LANE'S PUBLIC DOOR.

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
import heapq as _heapq
import ctypes
from . import _portable_math as math
import sys

from . import _backend
from . import _portable_math as _pm
from ._array import Array
from ._buffer import addr_ro, as_f32_c, as_i32_c, frombytes

__all__ = ["IncrementalPCA", "GaussianRandomProjection", "SparseRandomProjection", "johnson_lindenstrauss_min_dim",
           "NMF", "FastICA", "FactorAnalysis",
           "lu_factor", "lu_solve", "solve", "lstsq", "randomized_svd",
           "PLSRegression", "PLSCanonical", "CCA",
           "DictionaryLearning", "MiniBatchDictionaryLearning", "SparsePCA", "MiniBatchSparsePCA", "sparse_encode", "SparseCoder",
           "LatentDirichletAllocation", "Isomap", "MDS", "ClassicalMDS", "LocallyLinearEmbedding",
           "MinCovDet", "EllipticEnvelope", "AlternatingLeastSquares"]

_BINDING = "_mojolearn_x_decomp"

# x_decomp/cells.mojo op codes
_OP = dict(
    muz=36, lgamma=37,
    add=0, sub=1, mul=2, div=3, axpy=4, maxs=5, mu=6, sqrt=7, sq=8, exp=9, logs=10, tanh=11,
    onemsq=12, abs=13, scale=14, fma=15, recip=16, soft=17, submul=18, mins=19, copyb=20,
    sqdiff=21, adds=22, gts=23, digamma=24, expg=25, expgp=26, cube=27, cubep=28, max=30,
    min=31, sign=33, le=34, select=35, p2scale=38,
)


def _f32(x):
    """x rounded once to float32, as the binding would round it."""
    return array.array("f", [x])[0]


def _is_sparse(X):
    """A scipy.sparse matrix or array, by duck type (no scipy import)."""
    return hasattr(X, "toarray") and hasattr(X, "nnz") and hasattr(X, "format")


class _DevBuf:
    """A device-resident matrix of the GPU binding (x_decomp/resident.mojo):
    a pooled buffer id, returned to the pool when this object dies."""
    __slots__ = ("b", "id", "n")

    def __init__(self, b, n):
        self.b, self.n = b, n
        self.id = b.x_decomp_dev_alloc(max(n, 1))

    def __del__(self):
        try:
            self.b.x_decomp_dev_free(self.id)
        except Exception:     # interpreter shutdown: the pool dies with the process
            pass


# THE BUFFER POOL (lane decomp-cpu, 2026-09-28). Every cell call writes a
# fresh output matrix, and a large fresh `array.array` is fresh pages from the
# OS: on a 10M-element elementwise op the page faults cost more than the
# threaded cell itself. A large store whose last holder (an `_M`) is dropped
# goes back here instead, and `_M.zeros` of the same length takes it and
# zeroes it (one memset). Data movement only: a pooled store is handed out
# only when nothing else references it (its reference count says so), and
# it is zeroed exactly as a fresh one is. Bounded: stores of at least
# _POOL_MIN elements, at most _POOL_PER_SIZE per length, _POOL_CAP bytes held.
_POOL = {}
_POOL_HELD = [0]
_POOL_MIN = 1 << 18
_POOL_PER_SIZE = 8
_POOL_CAP = 512 << 20


def _pool_take(n):
    got = _POOL.get(n)
    if not got:
        return None
    s = got.pop()
    _POOL_HELD[0] -= 4 * n
    ctypes.memset(s.buffer_info()[0], 0, 4 * n)
    return s


def _pool_give(s):
    n = len(s)
    if n < _POOL_MIN or _POOL_HELD[0] + 4 * n > _POOL_CAP:
        return
    got = _POOL.setdefault(n, [])
    if len(got) < _POOL_PER_SIZE:
        got.append(s)
        _POOL_HELD[0] += 4 * n


def _holders(s):
    return sys.getrefcount(s)


class _Probe:
    """Measures, once, what `_holders(self.s)` reads inside `__del__` when
    the dying object is the store's only holder (the count differs between
    Python versions, so it is measured, never assumed)."""
    __slots__ = ("s",)
    seen = []

    def __del__(self):
        _Probe.seen.append(_holders(self.s))


_p = _Probe()
_p.s = array.array("f")
del _p
_SOLE_HOLDER = _Probe.seen[0] if _Probe.seen else -1



def _host_all_finite(a):
    """True/False from the base binding's `all_finite_f32`, or None when this
    install has no such helper (the caller then runs the cell form)."""
    try:
        from ._buffer import all_finite
        return bool(all_finite(a))
    except (ImportError, AttributeError, OSError, TypeError):
        return None


class _M:
    """A row-major float32 matrix held in an `array.array('f')`, or on the
    device (lane decomp-apple): a GPU kit's elementwise, product, fold and
    distance results stay in a device buffer (`_d`) until Python reads `s`
    (or `addr`), which downloads them once and makes the host store the
    matrix again. A host matrix handed to those entries is uploaded once
    and MOVES to the device (its value is the one at upload)."""
    __slots__ = ("_s", "r", "c", "_d")

    def __init__(self, s, r, c):
        if len(s) != r * c:
            raise ValueError("x_decomp: matrix store does not match its shape")
        self._s, self.r, self.c, self._d = s, r, c, None

    @classmethod
    def _on_device(cls, d, r, c):
        m = cls.__new__(cls)
        m._s, m.r, m.c, m._d = None, r, c, d
        return m

    @property
    def s(self):
        if self._s is None:
            d = self._d
            n = self.r * self.c
            a = _pool_take(n) if n >= _POOL_MIN else None
            if a is None:
                a = array.array("f", [0.0]) * n
            if len(a):
                d.b.x_decomp_dev_download(d.id, a.buffer_info()[0], len(a))
            self._s = a
        self._d = None        # the host store may now change
        return self._s

    @s.setter
    def s(self, v):
        self._s, self._d = v, None

    def __getstate__(self):
        return (self.s, self.r, self.c)

    def __setstate__(self, st):
        self._s, self.r, self.c = st
        self._d = None

    def __del__(self):
        try:        # the host store only (a device-resident matrix frees its buffer through _DevBuf)
            if self._s is not None and len(self._s) >= _POOL_MIN and _holders(self._s) == _SOLE_HOLDER:
                _pool_give(self._s)
        except Exception:
            pass

    @classmethod
    def zeros(cls, r, c):
        s = _pool_take(r * c) if r * c >= _POOL_MIN else None
        return cls(s if s is not None else array.array("f", [0.0]) * (r * c), r, c)

    @classmethod
    def of(cls, values, r, c):
        return cls(array.array("f", values), r, c)

    @classmethod
    def shape_of_input(cls, X, name="X"):
        """(rows, cols) of an input `from_input` would accept, with the same
        refusals (shape, dtype, finiteness), and no store built; the cell
        form of the finiteness check runs where the base binding lacks the
        host helper."""
        if _is_sparse(X):
            X = X.toarray()
        a = as_f32_c(X, ndim=2, name=name)[0]
        if a.ndim != 2 or min(a.shape) == 0:
            raise ValueError(f"{name}: a nonempty two-dimensional input is required")
        fin = _host_all_finite(a)
        if fin is False:
            raise ValueError(f"{name}: input must be finite; NaN/inf are unsupported")
        if fin is None:
            cls.from_input(a, name)
        return a.shape[0], a.shape[1]

    @classmethod
    def from_input(cls, X, name="X"):
        if _is_sparse(X):
            X = X.toarray()          # a scipy.sparse matrix/array: densified (exact)
        a = as_f32_c(X, ndim=2, name=name)[0]
        if a.ndim != 2 or min(a.shape) == 0:
            raise ValueError(f"{name}: a nonempty two-dimensional input is required")
        s = array.array("f")
        mv = getattr(a, "_mv", None)
        # one copy: the Array's own buffer straight into the store (its
        # tobytes() was a second full copy of every input)
        # (a 2-D float memoryview is not bytes-like to array.frombytes: cast
        # it to a flat byte view first, as _array.Array does; same bytes)
        s.frombytes(mv.cast("B") if isinstance(mv, memoryview) and mv.c_contiguous else a.tobytes())
        # The finiteness refusal on the host when the base binding has its
        # helper (lane neural-pass27, 2026-10-01): the cell form below
        # uploaded the whole input to the device and freed it again just to
        # sum x * 0 (880 MB and ~250 ms a call at the board's 1M x 220 on
        # the M4); the predicate is the same, no recorded bit depends on it.
        fin = _host_all_finite(a)
        if fin is False:
            raise ValueError(f"{name}: input must be finite; NaN/inf are unsupported")
        m = cls(s, a.shape[0], a.shape[1])
        if fin is None:
            m._check_finite(name)
        return m

    def _check_finite(self, name):
        """NaN/inf refused, through the cells: every x * 0 summed is 0 for a
        finite input and NaN if any value is NaN or infinite."""
        if not len(self.s):
            return
        k = _Kit(_backend.default_mode())
        t = k.total(k.ew("scale", self, s=0.0)).s[0]
        if t != t:
            raise ValueError(f"{name}: input must be finite; NaN/inf are unsupported")

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
        return self.take_cols(range(a, b))

    def take_cols(self, idx):
        """Data movement by strided slices: one C-level copy per column."""
        idx = list(idx)
        w = len(idx)
        out = array.array("f", [0.0]) * (self.r * w)
        for t, j in enumerate(idx):
            out[t::w] = self.s[j::self.c]
        return _M(out, self.r, w)

    @property
    def T(self):
        if self.r == 1 or self.c == 1:
            return _M(self.s, self.c, self.r)
        out = array.array("f")
        for j in range(self.c):
            out.extend(self.s[j::self.c])
        return _M(out, self.c, self.r)

    def reshape(self, r, c):
        return _M(self.s, r, c)

    def neg_rows(self, flags):
        """Exact sign flip of the rows whose flag is set (a sign bit only)."""
        if not any(flags):
            return self
        k = _Kit(_backend.default_mode())
        return k.ew("mul", self, _M.of([-1.0 if f else 1.0 for f in flags], self.r, 1))

    def neg_cols(self, flags):
        if not any(flags):
            return self
        k = _Kit(_backend.default_mode())
        return k.ew("mul", self, _M.of([-1.0 if f else 1.0 for f in flags], 1, self.c))

    def list(self):
        return list(self.s)


_M._one = _M(array.array("f", [0.0]), 1, 1)
_DEV_ONE = {}
#: the fewest values for which a host-only kit call goes resident (_Kit._use)
#: (cgr-decomp: the MOJOLEARN_XD_RES_MIN override is deleted).
#: 1 (always) measured fastest on m4pro-a once the pool was O(1) and capped
#: (1790588268075: MiniBatchDictionaryLearning 0.66 s at 1 against 1.56 s at
#: 1024 and 1.76 s at 16384; MinCovDet 48.4 / 55.0 / 51.1 s; FastICA 0.089 /
#: 0.129 / 0.124 s)
import os as _os
_RES_MIN = 1


def _dev_one(kit):
    """The unused broadcast operand (_M._one, a 0) on the device, one per
    binding, never moved off the host `_M._one`."""
    raw = kit._raw()
    d = _DEV_ONE.get(id(raw))
    if d is None or d.b is not raw:
        d = _DevBuf(raw, 1)
        z = array.array("f", [0.0])
        kit.b.x_decomp_dev_upload(d.id, z.buffer_info()[0], 1)
        _DEV_ONE[id(raw)] = d
    return d.id


_M._dev_one = staticmethod(_dev_one)


def _any_negative(M):
    """Whether any value is < 0, through the cells (x < 0 counted as ones)."""
    if not len(M.s):
        return False
    k = _Kit(_backend.default_mode())
    return k.total(k.ew("gts", k.ew("scale", M, s=-1.0), s=0.0)).s[0] > 0


def _vstack(*ms):
    s = array.array("f")
    for m in ms:
        s.extend(m.s)
    return _M(s, sum(m.r for m in ms), ms[0].c)


def _hstack(*ms):
    r = ms[0].r
    w = sum(m.c for m in ms)
    s = array.array("f", [0.0]) * (r * w)
    off = 0
    for m in ms:
        for j in range(m.c):
            s[off + j::w] = m.s[j::m.c]
        off += m.c
    return _M(s, r, w)


def _kit_vendor(kit):
    """The vendor the kit's binding was compiled for ("metal", "cuda", "hip",
    ...), "" when the binding cannot say; asked once per kit."""
    v = kit.__dict__.get("_vendor")
    if v is None:
        try:
            v = str(kit._raw().x_decomp_vendor())
        except Exception:
            v = ""
        kit._vendor = v
    return v


class _Kit:
    """The binding's cells, called on `_M` matrices."""

    def __init__(self, mode, binding=None):
        self.mode = mode
        self.b = _backend.binding("_mojolearn_x_decomp", mode) if binding is None else binding

    # ---- device-resident path (GPU binding only; x_decomp/resident.mojo)
    def _raw(self):
        try:        # through bench/decomp_speed.py's profiler proxy (never the binding's own __getattr__)
            return object.__getattribute__(self.b, "_b")
        except AttributeError:
            return self.b

    def _res(self):
        """Whether this binding has the resident entries (the GPU binding; a
        missing name on the host binding's proxy raises ImportError)."""
        r = self.__dict__.get("_res_ok")
        if r is None:
            try:
                getattr(self._raw(), "x_decomp_dev_ew")
                r = True
            except Exception:
                r = False
            self._res_ok = r
        return r

    def _use(self, *ms):
        """The resident path for this call: the GPU binding, and an operand
        already on the device or one of at least _RES_MIN values. A small
        host-only call keeps the synchronous host-address path (a device
        result Python reads at once would pay an upload, a launch and a
        download where one call did)."""
        if not self._res():
            return False
        big = False
        for m in ms:
            if m is None:
                continue
            if m._d is not None:
                return True
            if m.r * m.c >= _RES_MIN:
                big = True
        return big

    def _did(self, M):
        """M's device id on this binding, uploading a host matrix (it moves)."""
        raw = self._raw()
        d = M._d
        if d is not None and d.b is raw:
            return d.id
        a = M.s
        d = _DevBuf(raw, len(a))
        if len(a):
            self.b.x_decomp_dev_upload(d.id, a.buffer_info()[0], len(a))
        M._s, M._d = None, d
        return d.id

    def _dout(self, r, c):
        return _M._on_device(_DevBuf(self._raw(), r * c), r, c)

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
        if A.r * A.c and self._use(A, B, C):
            out = self._dout(A.r, A.c)
            one = _M._dev_one(self) if (Bm is _M._one or Cm is _M._one) else None
            ib = one if Bm is _M._one else self._did(Bm)
            ic = one if Cm is _M._one else self._did(Cm)
            self.b.x_decomp_dev_ew(self._did(A), ib, ic, out._d.id,
                                   [_OP[op], A.r * A.c, A.c, Bm.r * Bm.c, bm, Cm.r * Cm.c, cm], float(s))
            return out
        out = _M.zeros(A.r, A.c)
        if A.r * A.c:
            self.b.x_decomp_ew(A.addr, Bm.addr, Cm.addr, out.addr,
                               [_OP[op], A.r * A.c, A.c, len(Bm.s), bm, len(Cm.s), cm], float(s))
        return out

    def const(self, v, r=1, c=1):
        return _M.of([v] * (r * c), r, c)

    def diag_mask(self, n):
        """The n x n identity as a device matrix (1 on the diagonal, 0 off
        it), made on the device by ONE `copyb` launch: the n * n buffer read
        as rows of n + 1, whose first column is exactly the diagonal, takes
        e = [1, 0, ..., 0] (length n + 1) as a row-vector broadcast. None on
        a binding without the resident entries."""
        if n < 1 or not self._res():
            return None
        out = self._dout(n, n)
        e = _M.of([1.0] + [0.0] * n, 1, n + 1)
        one = _M._dev_one(self)
        did = out._d.id
        self.b.x_decomp_dev_ew(did, self._did(e), one, did,
                               [_OP["copyb"], n * n, n + 1, n + 1, 1, 1, 3], 0.0)
        return out

    # ---- reductions and products
    def mm(self, A, B, ta=False, tb=False):
        m, k = (A.c, A.r) if ta else (A.r, A.c)
        k2, n = (B.c, B.r) if tb else (B.r, B.c)
        if k != k2:
            raise ValueError(f"x_decomp: gemm inner dimensions {k} and {k2} differ")
        if m * n and m * k and k * n and self._use(A, B):
            out = self._dout(m, n)
            self.b.x_decomp_dev_gemm(self._did(A), self._did(B), out._d.id, [m, k, n, int(ta), int(tb)])
            return out
        out = _M.zeros(m, n)
        if m * n:
            self.b.x_decomp_gemm(A.addr, B.addr, out.addr, [m, k, n, int(ta), int(tb)])
        return out

    def colsum(self, A):
        if A.r * A.c and self._use(A):
            out = self._dout(1, A.c)
            self.b.x_decomp_dev_colsum(self._did(A), out._d.id, [A.r, A.c])
            return out
        out = _M.zeros(1, A.c)
        self.b.x_decomp_colsum(A.addr, out.addr, [A.r, A.c])
        return out

    def rowsum(self, A):
        if A.r * A.c and self._use(A):
            out = self._dout(A.r, 1)
            self.b.x_decomp_dev_rowsum(self._did(A), out._d.id, [A.r, A.c])
            return out
        out = _M.zeros(A.r, 1)
        self.b.x_decomp_rowsum(A.addr, out.addr, [A.r, A.c])
        return out

    def total(self, A):
        """The sum of every entry (1 x 1): rows ascending, then the row sums ascending."""
        return self.colsum(self.rowsum(A))

    def sqdist(self, A, B):
        if A.r * B.r and A.c and self._use(A, B):
            out = self._dout(A.r, B.r)
            self.b.x_decomp_dev_sqdist(self._did(A), self._did(B), out._d.id, [A.r, B.r, A.c])
            return out
        out = _M.zeros(A.r, B.r)
        self.b.x_decomp_sqdist(A.addr, B.addr, out.addr, [A.r, B.r, A.c])
        return out

    def pdist(self, A, B, kind, pw=2.0):
        """A non-Euclidean distance matrix (x_decomp/cells.mojo pdist_cell,
        DEVIATION 5319): kind 1 manhattan, 2 chebyshev, 3 minkowski pw,
        4 cosine."""
        if A.r * B.r and A.c and self._use(A, B):
            out = self._dout(A.r, B.r)
            self.b.x_decomp_dev_sqdist(self._did(A), self._did(B), out._d.id, [A.r, B.r, A.c, int(kind), float(pw)])
            return out
        out = _M.zeros(A.r, B.r)
        self.b.x_decomp_sqdist(A.addr, B.addr, out.addr, [A.r, B.r, A.c, int(kind), float(pw)])
        return out

    def rand(self, r, c, seed, stream, kind):
        out = _M.zeros(r, c)
        if r * c:
            self.b.x_decomp_rand(out.addr, [r * c, int(seed) & 0xFFFFFFFF, int(stream) & 0xFFFFFFFF, kind])
        return out

    # ---- small dense linear algebra
    def eigh(self, A, uplo=0):
        """Ascending eigenvalues (1 x n) and eigenvectors in COLUMNS (n x n):
        the round-robin Jacobi at every size. uplo 1 / 2 reads only the
        lower / upper triangle (numpy's UPLO, mirrored by the binding on the
        device), 0 the whole matrix."""
        n = A.r
        w, v = _M.zeros(1, n), _M.zeros(n, n)
        self.b.x_decomp_eigh(A.addr, w.addr, v.addr, [n, int(uplo)])
        return w, v

    def eigh_batch(self, A, batch, n):
        """`eigh` of `batch` n x n problems stacked in A (batch n x n), one
        device block each (x_decomp/rr_batch.mojo): (W batch x n ascending,
        V batch n x n, problem b's vectors in the COLUMNS of rows b n ..
        (b + 1) n)."""
        W, V = _M.zeros(batch, n), _M.zeros(batch * n, n)
        if batch:
            self.b.x_decomp_eigh_batch(A.addr, W.addr, V.addr, [int(batch), int(n)])
        return W, V

    def lu(self, A):
        n = A.r
        lu = A.copy()
        piv = array.array("i", [0] * n)
        info = _M.zeros(1, 1)
        self.b.x_decomp_lu(lu.addr, piv.buffer_info()[0], info.addr, [n])
        return lu, piv, int(info.s[0])

    def trisolve(self, lu, idx, B, trans=0):
        """The right-looking triangular solves on an LU factor (x_decomp/
        cells.mojo `trisolve_serial`): trans 0, U^-1 L^-1 B[idx] (F0^-1 B
        with idx the pivots' row order); trans 1, (L^-T U^-T B)[idx]
        (F0^-T B with idx that order's inverse). idx is an n x 1 matrix of
        row numbers."""
        n, w = lu.r, B.c
        if n * w and self._use(lu, B):
            out = self._dout(n, w)
            self.b.x_decomp_dev_trisolve(self._did(lu), self._did(idx), self._did(B), out._d.id, [n, w, int(trans)])
            return out
        out = _M.zeros(n, w)
        if n * w:
            self.b.x_decomp_trisolve(lu.addr, idx.addr, B.addr, out.addr, [n, w, int(trans)])
        return out

    def knn_select(self, D, k, exclude_self):
        """(indices as floats, values) of each row's k smallest entries of D,
        ascending by (value, column), column r skipped in row r when
        exclude_self (x_decomp/cells.mojo `knn_select_row`)."""
        n, m = D.r, D.c
        if n * k and self._use(D):
            I, V = self._dout(n, k), self._dout(n, k)
            self.b.x_decomp_dev_knn_select(self._did(D), V._d.id, I._d.id, [n, m, k, int(bool(exclude_self))])
            return I, V
        I, V = _M.zeros(n, k), _M.zeros(n, k)
        if n * k:
            self.b.x_decomp_knn_select(D.addr, V.addr, I.addr, [n, m, k, int(bool(exclude_self))])
        return I, V

    def lu_solve(self, lu, piv, B, trans=0):
        out = B.copy()
        p = [lu.r, B.c, trans] if trans else [lu.r, B.c]
        self.b.x_decomp_lu_solve(lu.addr, piv.buffer_info()[0], out.addr, p)
        return out

    def cd_rows(self, W, HHt, XHt, perm):
        """One sklearn `_update_cdnmf_fast` sweep over every row of W, in
        place; returns the total violation (rows ascending)."""
        n, kc = W.r, W.c
        viol = _M.zeros(n, 1)
        p = array.array("i", perm)
        self.b.x_decomp_cd_rows(W.addr, HHt.addr, XHt.addr, p.buffer_info()[0], viol.addr, [n, kc])
        return self.total(viol).s[0]

    def svd(self, A):
        """(S 1 x n DESCENDING, Vt n x n) of a tall A (m >= n): Householder QR
        then the one-sided Jacobi SVD of R (decomposition/'s full-PCA route),
        values sorted descending with ties to the lower index."""
        m, n = A.r, A.c
        s, v = _M.zeros(1, n), _M.zeros(n, n)
        self.b.x_decomp_svd(A.addr, s.addr, v.addr, [m, n])
        order = sorted(range(n), key=lambda j: (-s.s[j], j))
        return s.take_cols(order), v.take_cols(order).T

    def orth(self, A):
        """A copy of A with its columns orthonormalized: two passes of the
        Householder R and a row-parallel A R^-1 (DEVIATION 5309)."""
        if A.r * A.c and self._use(A):
            Q = self._dout(A.r, A.c)
            self.b.x_decomp_dev_orth(self._did(A), Q._d.id, [A.r, A.c])
            return Q
        Q = A.copy()
        self.b.x_decomp_orth(Q.addr, [A.r, A.c])
        return Q

    def orth_diag(self, A):
        """`orth`, and the list of the two passes' R-diagonal products (A.c
        floats): the sign of entry j orients Q's column j along A's, 0 marks
        a dependent column (`orth_diag_cell`, lane neural-pass17)."""
        diag = _M.zeros(1, A.c)
        if A.r * A.c and self._use(A):
            Q = self._dout(A.r, A.c)
            self.b.x_decomp_dev_orth_diag(self._did(A), Q._d.id, diag.addr, [A.r, A.c])
            return Q, list(diag.s)
        Q = A.copy()
        self.b.x_decomp_orth_diag(Q.addr, diag.addr, [A.r, A.c])
        return Q, list(diag.s)

    def lasso_rows(self, G, Q, W, alpha, max_iter, tol, positive):
        """Row-parallel Lasso CD on the Gram (x_decomp/cells.mojo `lasso_row`),
        W (n x k) the warm start, updated in place."""
        its = _M.zeros(Q.r, 1)
        self.b.x_decomp_lasso_rows(G.addr, Q.addr, W.addr, its.addr, [Q.r, Q.c, int(max_iter), int(positive)],
                                   [float(alpha), float(tol)])
        return W

    def lle_apply(self, Wb, idm, E):
        """out (nq x nc) = sum_a Wb[i, a] E[idm[i, a]] (LLE transform)."""
        nq, nn, nc = Wb.r, Wb.c, E.c
        out = _M.zeros(nq, nc)
        if nq * nc:
            self.b.x_decomp_lle_apply(Wb.addr, idm.addr, E.addr, out.addr, [nq, E.r, nn, nc])
        return out

    def lle_local(self, M, idm, method, nn, nc, tol):
        """LocallyLinearEmbedding's stacked factor B (x_decomp/lle_local.mojo):
        method 0 LTSA (n nn x n), 1 Hessian (n (nn - 1 - nc) x n), 2 modified
        (n nn x n); every per-sample step (the local Gram, its eigh, the
        assembly) a cell on the device, the local eigensolves batched."""
        n, d = M.r, M.c
        rows = n * (nn - 1 - nc) if method == 1 else n * nn
        B = _M.zeros(rows, n)
        self.b.x_decomp_lle_local(M.addr, idm.addr, B.addr, [int(method), n, d, int(nn), int(nc)], [float(tol)])
        return B

    def lu_aux(self, lu, piv, clamp=False):
        """An LU factor's companions, computed by the binding (on the device
        where there is one; x_decomp/cells.mojo `lu_aux_*`): (stats = [max
        |u_ii|, zero pivots, negative pivots, swaps], diag 1 x n, pm n x 1
        the swaps' row order, im n x 1 its inverse). clamp floors the pivots
        under eps * max |u_ii| in `lu` itself."""
        n = lu.r
        pm, im, diag, st = _M.zeros(n, 1), _M.zeros(n, 1), _M.zeros(1, n), _M.zeros(1, 4)
        self.b.x_decomp_lu_aux(lu.addr, piv.buffer_info()[0], pm.addr, im.addr, diag.addr, st.addr,
                               [n, int(bool(clamp))])
        return [float(v) for v in st.s], diag, pm, im

    def lars_rows(self, G, Q, m, nnz):
        """Row-parallel Lars on the Gram (x_decomp/cells.mojo `lars_row`): the
        n x k coefficients, m the samples of each row's problem."""
        W = _M.zeros(Q.r, Q.c)
        na = _M.zeros(Q.r, 1)
        if Q.r * Q.c:
            self.b.x_decomp_lars_rows(G.addr, Q.addr, W.addr, na.addr, [Q.r, Q.c, int(m), int(nnz)])
        return W

    def omp_rows(self, G, Q, nnz):
        W = _M.zeros(Q.r, Q.c)
        na = _M.zeros(Q.r, 1)
        self.b.x_decomp_omp_rows(G.addr, Q.addr, W.addr, na.addr, [Q.r, Q.c, int(nnz)])
        return W

    def rand_gamma(self, r, c, seed, stream, shape):
        out = _M.zeros(r, c)
        if r * c:
            self.b.x_decomp_rand_gamma(out.addr, [r * c, int(seed) & 0xFFFFFFFF, int(stream) & 0xFFFFFFFF],
                                       float(shape))
        return out

    def lda_rows(self, X, EW, Dt, Et, prior, max_iter, tol):
        """Row-parallel `_update_doc_distribution` (x_decomp/cells.mojo
        `lda_doc_row`); Dt and Et (n x k) are updated in place."""
        if X.r and self._use(X, EW, Dt, Et):
            # resident (lane/py-decomp-nbrs): X, EW, Dt and Et stay on the
            # device; Dt and Et are updated in place there
            self.b.x_decomp_dev_lda_rows(self._did(X), self._did(EW), self._did(Dt), self._did(Et),
                                         [X.r, EW.r, X.c, int(max_iter)], [float(prior), float(tol)])
            return Dt, Et
        its = _M.zeros(X.r, 1)
        self.b.x_decomp_lda_rows(X.addr, EW.addr, Dt.addr, Et.addr, its.addr,
                                 [X.r, EW.r, X.c, int(max_iter)], [float(prior), float(tol)])
        return Dt, Et

    def dijkstra(self, W):
        """All-pairs shortest paths on a dense undirected graph (0 = no edge),
        one source per thread; -1 marks an unreachable pair. The GPU binding
        keeps W and the result resident and compresses the arcs on the
        device (x_decomp/graph_device.mojo)."""
        n = W.r
        if self._res():
            dist = self._dout(n, n)
            self.b.x_decomp_dev_graph_dijkstra(self._did(W), dist._d.id, [n])
            return dist
        dist = _M.zeros(n, n)
        self.b.x_decomp_graph_dijkstra(W.addr, dist.addr, [n])
        return dist

    # ---- neighbor graphs (x_decomp/graph_cells.mojo, lane hr2-graph-embed):
    # resident on the GPU binding, the same cells on the host binding
    def graph_knn(self, D, nn, exclude_self):
        """(idx, dst), n x nn each: the nn smallest of every row of D, ties
        to the lower column, `exclude_self` dropping column i; idx holds the
        column numbers as exact floats."""
        n, m = D.r, D.c
        p = [n, m, nn, int(bool(exclude_self))]
        if self._res():
            idx, dst = self._dout(n, nn), self._dout(n, nn)
            if n * nn:
                self.b.x_decomp_dev_graph_knn(self._did(D), idx._d.id, dst._d.id, p)
            return idx, dst
        idx, dst = _M.zeros(n, nn), _M.zeros(n, nn)
        if n * nn:
            self.b.x_decomp_graph_knn(D.addr, idx.addr, dst.addr, p)
        return idx, dst

    def graph_knn_dense(self, idx, w, n):
        """The dense n x n graph W[i, idx[i, a]] = w[i, a] (1e-10 for 0)."""
        nn = idx.c
        if self._res():
            out = self._dout(n, n)
            self.b.x_decomp_dev_graph_knn_dense(self._did(idx), self._did(w), out._d.id, [n, nn])
            return out
        out = _M.zeros(n, n)
        self.b.x_decomp_graph_knn_dense(idx.addr, w.addr, out.addr, [n, nn])
        return out

    def graph_radius(self, D, r):
        """W[i, j] = D[i, j] (1e-10 for 0) where j != i and D[i, j] <= r."""
        n = D.r
        if self._res():
            out = self._dout(n, n)
            self.b.x_decomp_dev_graph_radius(self._did(D), out._d.id, [n], float(r))
            return out
        out = _M.zeros(n, n)
        self.b.x_decomp_graph_radius(D.addr, out.addr, [n], float(r))
        return out

    def graph_radius_geo(self, Dq, D, r):
        """Isomap's radius transform: G[i, c] = min over j (ascending) with
        Dq[i, j] <= r of D[j, c] + Dq[i, j] (x_decomp/graph_cells.mojo
        `radius_geo_cell`; 0 for a row with no such j)."""
        nq, n = Dq.r, Dq.c
        if self._res():
            out = self._dout(nq, n)
            if nq * n:
                self.b.x_decomp_dev_graph_radius_geo(self._did(Dq), self._did(D), out._d.id, [nq, n], float(r))
            return out
        out = _M.zeros(nq, n)
        if nq * n:
            self.b.x_decomp_graph_radius_geo(Dq.addr, D.addr, out.addr, [nq, n], float(r))
        return out

    def graph_lle_iw(self, idx, wb, n):
        """LLE's dense I - W from the kNN lists and barycenter weights."""
        nn = idx.c
        if self._res():
            out = self._dout(n, n)
            self.b.x_decomp_dev_graph_lle_iw(self._did(idx), self._did(wb), out._d.id, [n, nn])
            return out
        out = _M.zeros(n, n)
        self.b.x_decomp_graph_lle_iw(idx.addr, wb.addr, out.addr, [n, nn])
        return out

    def graph_components(self, W):
        """(comp n x 1, C): the weak components of W (nonzero either way),
        numbered by their lowest node."""
        n = W.r
        if self._res():
            comp = self._dout(n, 1)
            c = self.b.x_decomp_dev_graph_components(self._did(W), comp._d.id, [n])
            return comp, int(c)
        comp = _M.zeros(n, 1)
        c = self.b.x_decomp_graph_components(W.addr, comp.addr, [n])
        return comp, int(c)

    def graph_join(self, W, D, comp, C):
        """sklearn `_fix_connected_components`, W changed in place."""
        n = W.r
        if self._res():
            self.b.x_decomp_dev_graph_join(self._did(W), self._did(D), self._did(comp), [n, C])
            return W
        self.b.x_decomp_graph_join(W.addr, D.addr, comp.addr, [n, C])
        return W

    def barycenter(self, X, Y, nbr, reg):
        """sklearn barycenter_weights: (n x k) weights of each row of X on
        its k neighbors in Y (`nbr`: n lists of k indices)."""
        if isinstance(nbr, _M):
            n, k, idx = X.r, nbr.c, nbr
        else:
            n, k = X.r, len(nbr[0])
            idx = _M.of([float(j) for row in nbr for j in row], n, k)
        W, flags = _M.zeros(n, k), _M.zeros(n, 1)
        self.b.x_decomp_barycenter_rows(X.addr, Y.addr, idx.addr, W.addr, flags.addr, [n, Y.r, X.c, k], float(reg))
        return W

    def als_resident(self, C):
        """Whether `als` runs on the device with C resident (the GPU binding
        with x_decomp_dev_als_rows): the item half-sweep then reads C through
        strides (`trans=True`), so no transpose is built."""
        return self._use(C) and hasattr(self._raw(), "x_decomp_dev_als_rows")

    def als(self, C, Y, reg, trans=False):
        """One implicit least_squares half-sweep: every row's factor from the
        confidences C (n x m; with trans, the rows are C's columns) and the
        other side's factors Y (m x f). trans needs `als_resident(C)`."""
        if trans or self.als_resident(C):
            n, m = (C.c, C.r) if trans else (C.r, C.c)
            f = Y.c
            YtY = self.mm(Y, Y, ta=True)
            X, flags = self._dout(n, f), self._dout(n, 1)
            su, si = (1, C.c) if trans else (C.c, 1)
            self.b.x_decomp_dev_als_rows(self._did(C), self._did(Y), self._did(YtY), X._d.id, flags._d.id,
                                         [n, m, f, su, si, C.r * C.c], float(reg))
            return X
        n, f = C.r, Y.c
        YtY = self.mm(Y, Y, ta=True)
        X, flags = _M.zeros(n, f), _M.zeros(n, 1)
        self.b.x_decomp_als_rows(C.addr, Y.addr, YtY.addr, X.addr, flags.addr, [n, C.c, f], float(reg))
        return X

    def geqrf(self, A):
        """(h, tau): LAPACK geqrf's factored form of A (m x n, any shape): R on
        and above the diagonal, the reflectors' tails below it, tau
        (1 x min(m, n)) their scalars (x_decomp/qr_sliced.mojo's order,
        DEVIATION 5320)."""
        h = A.copy()
        kk = min(A.r, A.c)
        tau = _M.zeros(1, kk)
        self.b.x_decomp_geqrf(h.addr, tau.addr, [A.r, A.c])
        return h, tau

    def orgqr(self, h, tau, qc):
        """The first qc columns of Q = H_0 ... H_{k-1} (m x qc) from geqrf's
        (h, tau), in x_decomp/qr_sliced.mojo's order."""
        Q = _M.zeros(h.r, qc)
        self.b.x_decomp_orgqr(h.addr, tau.addr, Q.addr, [h.r, h.c, tau.c, qc])
        return Q

    def als_cg(self, C, Y, X0, reg, cg_steps):
        """implicit's `_least_squares_cg` half-sweep: every row of X0 (n x f)
        moved by cg_steps conjugate-gradient steps (x_decomp/cells.mojo
        `als_cg_row`, DEVIATION 5321); returns the new factors."""
        n, f = C.r, Y.c
        YtY = self.mm(Y, Y, ta=True)
        X, steps = X0.copy(), _M.zeros(n, 1)
        self.b.x_decomp_als_cg_rows(C.addr, Y.addr, YtY.addr, X.addr, steps.addr, [n, C.c, f, int(cg_steps)],
                                    float(reg))
        return X

    def qr_r(self, A):
        """R (n x n) of the Householder QR of a tall A (decomposition/'s TSQR)."""
        R = _M.zeros(A.c, A.c)
        self.b.x_decomp_qr_r(A.addr, R.addr, [A.r, A.c])
        return R

    def absmax_flags(self, A, by_col):
        """Per column (by_col) or row of A: True when its largest-|.| entry
        (ties to the lower index) is negative (x_decomp/cells.mojo
        `absmax_sign_cell`, DEVIATION 5317)."""
        cnt = A.c if by_col else A.r
        if cnt and A.r * A.c and self._use(A):
            out = self._dout(1, cnt)
            self.b.x_decomp_dev_absmax(self._did(A), out._d.id, [A.r, A.c, 1 if by_col else 0])
            return [v < 0 for v in out.s]
        out = _M.zeros(1, cnt)
        if cnt and len(A.s):
            self.b.x_decomp_absmax_sign(A.addr, out.addr, [A.r, A.c, 1 if by_col else 0])
        return [v < 0 for v in out.s]

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
    return Vt.neg_rows(_Kit(_backend.default_mode()).absmax_flags(Vt, False))


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
    `u_based_decision=False`. A scipy.sparse X is densified one batch at a
    time in fit (sklearn's route), whole in partial_fit and transform."""
    _parameters = ("n_components", "whiten", "copy", "batch_size", "numeric_mode")

    def __init__(self, n_components=None, *, whiten=False, copy=True, batch_size=None, numeric_mode=None):
        self.n_components, self.whiten, self.copy = n_components, whiten, copy
        self.batch_size, self.numeric_mode = batch_size, numeric_mode

    def fit(self, X, y=None):
        self.numeric_mode_ = _mode(self.numeric_mode)
        for a in ("components_", "n_samples_seen_"):
            if hasattr(self, a):
                delattr(self, a)
        if _is_sparse(X):
            n, d = X.shape
            Xc = X.tocsr()
            if n == 0 or d == 0:
                raise ValueError("X: a nonempty two-dimensional input is required")

            def rows(a, b):
                return _M.from_input(Xc[a:b])
        else:
            M = _M.from_input(X)
            n, d = M.r, M.c
            rows = M.rows
        self.batch_size_ = 5 * d if self.batch_size is None else int(self.batch_size)
        mb = self.n_components or 0
        start = 0
        for _ in range(n // self.batch_size_):
            end = start + self.batch_size_
            if end + mb > n:
                continue
            self._partial(rows(start, end))
            start = end
        if start < n:
            self._partial(rows(start, n))
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
    # DEVIATION 6900: eps ** 2, eps ** 3 correctly rounded and the pinned log,
    # not the platform pow / log (a host's last bit can move the floor)
    denominator = (_pm.powi(eps, 2) / 2) - (_pm.powi(eps, 3) / 3)
    return int(4 * _pm.log(n_samples) / denominator)


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


def _rp_rand(k, r, c, seed, stream, kind, dev):
    """`k.rand`'s matrix; with dev, drawn into a device matrix by the same
    kernel (enqueued, nothing downloaded)."""
    if not (dev and r * c):
        return k.rand(r, c, seed, stream, kind)
    out = k._dout(r, c)
    k.b.x_decomp_dev_rand(out._d.id, [r * c, int(seed) & 0xFFFFFFFF, int(stream) & 0xFFFFFFFF, kind])
    return out


def _grp_cls2(k):
    """lane/apple-fast-gap-cls2: the random projections' FAST Apple fit
    switches compiled into the kit's binding (x_decomp/resident.mojo
    `grp_cls2_py`: bit 1 NOSCAN, 2 DEVSCAN, 4 LAZY); 0 when it has none."""
    fn = getattr(k.b, "x_decomp_grp_cls2", None)
    return int(fn()) if fn is not None else 0


def _sparse_rp_device(mode):
    """lane/apple-fast-kapprox: whether SparseRandomProjection.fit draws its
    matrix in one x_neighbors launch and skips the host finiteness pass over
    X (FAST + Apple, `-D MOJOLEARN_SPARSE_RP_DEVICE`), read back from that
    binding's compile-time constant (no env read). The binding, or None."""
    try:
        b = _backend.binding("_mojolearn_x_neighbors", mode)
    except Exception:
        return None
    fn = getattr(b, "x_neighbors_sparse_rp_device", None)
    return b if fn is not None and int(fn()) != 0 else None


class _RandomProjection(_Base):
    """sklearn `random_projection.py::BaseRandomProjection`. The matrix is
    drawn from the lane's counter-based Philox stream (x_decomp/cells.mojo
    `rand_cell`), not numpy's generator: the same `random_state` gives the
    same matrix on every box, and a different one than sklearn's."""

    def _device_fit(self, X):
        """lane/apple-fast-kapprox: the FAST + Apple fit of the sparse
        projection (SparseRandomProjection overrides); None takes main's path."""
        return None

    @property
    def components_(self):
        """The (n_components, n_features) matrix. Under lane/apple-fast-gap-cls2's
        LAZY switch the fit leaves it on the device and the first read
        downloads it once (the same words)."""
        dct = self.__dict__
        v = dct.get("_rp_components")
        if v is None and "_rp_lazy" in dct:
            k, C, kc, d = dct.pop("_rp_lazy")
            res = array.array("f", [0.0]) * (kc * d)
            k.b.x_decomp_dev_download(C._d.id, res.buffer_info()[0], kc * d)
            v = dct["_rp_components"] = Array._owned(res, (kc, d), "<f4", "C")
        if v is None:
            raise AttributeError("components_")
        return v

    @components_.setter
    def components_(self, v):
        self.__dict__.pop("_rp_lazy", None)
        self.__dict__["_rp_components"] = v

    def _check(self, attr="components_"):
        if attr == "components_" and "_rp_lazy" in self.__dict__:
            return
        super()._check(attr)

    def __getstate__(self):
        if "_rp_lazy" in self.__dict__:
            _ = self.components_
        return dict(self.__dict__)

    def fit(self, X, y=None):
        self.numeric_mode_ = _mode(self.numeric_mode)
        got = self._device_fit(X)
        if got is not None:
            return got
        # fit reads only the input's shape and refuses a non-finite input;
        # the store (a copy of the whole input) is transform's to build
        # (lane neural-pass27: fit_transform converted the 880 MB input
        # twice at the board's shape)
        k = self._kit()
        cls2 = _grp_cls2(k) if self.numeric_mode_ == "fast" else 0
        if cls2 & 3 and not _is_sparse(X):
            # lane/apple-fast-gap-cls2 (FAST + Apple, x_decomp/resident.mojo
            # GRP_CLS2_*): no host walk over X. NOSCAN: the shape only
            # (transform's device projection refuses a non-finite X);
            # DEVSCAN: the refusal as one device scan of X
            a = as_f32_c(X, ndim=2, name="X")[0]
            if a.ndim != 2 or min(a.shape) == 0:
                raise ValueError("X: a nonempty two-dimensional input is required")
            if cls2 & 2 and int(k.b.x_decomp_dev_first_nonfinite(addr_ro(a, name="X"), a.size)) >= 0:
                raise ValueError("X: input must be finite; NaN/inf are unsupported")
            n, d = a.shape
        else:
            n, d = _M.shape_of_input(X)
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
        self.n_components_ = kc
        self.n_features_in_ = d
        # lane gap-nb-maxabs-grp: on the GPU binding the matrix is drawn on
        # the device (`x_decomp_dev_rand`, the same Philox words as
        # `x_decomp_rand`); cgr-decomp deleted the MOJOLEARN_XD_RP_TILED switch
        dev = k._res()
        self.components_m_ = self._make(k, kc, d, _seed_of(self.random_state), dev)
        if dev and self.components_m_._d is not None:
            # the matrix was drawn on the device and
            # stays there for transform; components_ is a copy of its words
            # (one download, the device matrix kept)
            C = self.components_m_
            if cls2 & 4 and not self.compute_inverse_components:
                # lane/apple-fast-gap-cls2 LAZY: downloaded on first read
                self.__dict__.pop("_rp_components", None)
                self.__dict__["_rp_lazy"] = (k, C, kc, d)
            else:
                res = array.array("f", [0.0]) * (kc * d)
                k.b.x_decomp_dev_download(C._d.id, res.buffer_info()[0], kc * d)
                self.components_ = Array._owned(res, (kc, d), "<f4", "C")
        else:
            self.components_ = self.components_m_.out()
        if self.compute_inverse_components:
            self.inverse_m_ = _pinv_rows(k, self.components_m_)
            self.inverse_components_ = self.inverse_m_.out()
        return self

    def transform(self, X):
        self._check()
        if not _is_sparse(X):
            out = self._project(X)
            if out is not None:
                return out
        M = _M.from_input(X)
        if M.c != self.n_features_in_:
            raise ValueError(f"X has {M.c} features, but {type(self).__name__} is expecting {self.n_features_in_}")
        return self._kit().mm(M, self.components_m_, tb=True).out()

    def _project(self, X):
        """transform on the GPU binding (lane gap-nb-maxabs-grp): X goes up from
        its own buffer (no host copy into a store, no host finiteness pass),
        x_decomp_dev_project's tiled kernel computes `mm(X, components, tb)`'s
        words and flags a non-finite entry of X on the device, and the result
        comes down once into the returned array. None on the host binding
        (or a column without the tile): the caller takes the path above."""
        k = self._kit()
        if not k._res():
            return None
        a = as_f32_c(X, ndim=2, name="X")[0]
        if a.ndim != 2 or min(a.shape) == 0:
            raise ValueError("X: a nonempty two-dimensional input is required")
        m, d = a.shape
        if d != self.n_features_in_:
            raise ValueError(f"X has {d} features, but {type(self).__name__} is expecting {self.n_features_in_}")
        B = self.components_m_
        nc = B.r
        A = _M._on_device(_DevBuf(k._raw(), a.size), m, d)
        k.b.x_decomp_dev_upload(A._d.id, addr_ro(a, name="X"), a.size)
        C, flag = k._dout(m, nc), k._dout(1, 1)
        if int(k.b.x_decomp_dev_project(A._d.id, k._did(B), C._d.id, flag._d.id, [m, d, nc])) < 0:
            return None
        res = array.array("f", [0.0]) * (m * nc)
        k.b.x_decomp_dev_download(C._d.id, res.buffer_info()[0], m * nc)
        bad = array.array("f", [0.0])
        k.b.x_decomp_dev_download(flag._d.id, bad.buffer_info()[0], 1)
        if bad[0] != 0:
            raise ValueError("X: input must be finite; NaN/inf are unsupported")
        return Array._owned(res, (m, nc), "<f4", "C")

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

    def _make(self, k, kc, d, seed, dev=False):
        return k.ew("scale", _rp_rand(k, kc, d, seed, 1, 1, dev), s=1.0 / math.sqrt(kc))


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

    def _make(self, k, kc, d, seed, dev=False):
        dens = 1.0 / math.sqrt(d) if self.density == "auto" else float(self.density)
        if not 0 < dens <= 1:
            raise ValueError(f"Expected density in range ]0, 1], got: {dens}")
        self.density_ = dens
        u = _rp_rand(k, kc, d, seed, 2, 0, dev)
        sgn = k.ew("scale", _rp_rand(k, kc, d, seed, 3, 2, dev), s=math.sqrt(1.0 / dens) / math.sqrt(kc))
        if dens == 1:
            return sgn
        # u < density keeps the signed value, else 0 (select: x > s -> y else z)
        return k.ew("select", u, _M.zeros(1, 1), sgn, s=dens - 2.0 ** -25)

    def _device_fit(self, X):
        """lane/apple-fast-kapprox (FAST + Apple, -D MOJOLEARN_SPARSE_RP_DEVICE):
        fit reads the input's shape from its own buffer (no host finiteness
        pass over every cell: transform's tiled device projection flags a
        non-finite X, as it does on main) and draws the kc x d matrix in ONE
        x_neighbors launch (counter-based uniforms of the same law as the
        kit's four-launch draw, not its Philox words). The matrix comes down
        once as components_; transform's `_project` uploads it once and
        keeps it on the device, as main does."""
        if self.n_components == "auto" or self.compute_inverse_components or _is_sparse(X):
            return None
        b = _sparse_rp_device(self.numeric_mode_)
        if b is None:
            return None
        a = as_f32_c(X, ndim=2, name="X")[0]
        if a.ndim != 2 or min(a.shape) == 0:
            raise ValueError("X: a nonempty two-dimensional input is required")
        n, d = a.shape
        kc = int(self.n_components)
        if kc <= 0:
            raise ValueError(f"n_components must be greater than 0, got {kc}")
        dens = 1.0 / math.sqrt(d) if self.density == "auto" else float(self.density)
        if not 0 < dens <= 1:
            raise ValueError(f"Expected density in range ]0, 1], got: {dens}")
        self.density_ = dens
        res = array.array("f", [0.0]) * (kc * d)
        b.xn_kapprox_sparse_rp([res.buffer_info()[0]], [kc, d, _seed_of(self.random_state) & 0x7FFFFFFF],
                               [float(dens), math.sqrt(1.0 / dens) / math.sqrt(kc)])
        self.n_components_ = kc
        self.n_features_in_ = d
        self.components_m_ = _M(res, kc, d)
        self.components_ = Array._owned(array.array("f", res), (kc, d), "<f4", "C")
        return self


# ================================================================ thin SVD
def _thin_svd(k, X, nc, u_based=True):
    """(U n x nc, S 1 x nc, Vt nc x d) of X through the QR + one-sided Jacobi
    SVD of X (or of X^T when X is wide), singular values descending, then sklearn's
    `svd_flip` (u_based: the largest-|.| entry of each COLUMN of U positive;
    else of each ROW of Vt). U (or Vt) is recovered as X V / S (X^T U / S);
    a zero singular value gives a zero vector."""
    n, d = X.r, X.c
    if d <= n:
        S, Vt = k.svd(X)
        S, Vt = S.cols(0, nc), Vt.rows(0, nc)
        U = k.ew("div", k.mm(X, Vt, tb=True), S)
    else:
        S, Ut = k.svd(X.T)
        S, Ut = S.cols(0, nc), Ut.rows(0, nc)
        U = Ut.T
        Vt = k.ew("div", k.mm(U, X, ta=True), S.T)
    fl = k.absmax_flags(U, True) if u_based else k.absmax_flags(Vt, False)
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
    shuffle=True permutes the coordinates of every CD sweep with the Philox
    stream. beta_loss 'frobenius', 'kullback-leibler' and 'itakura-saito'
    (the last two with solver='mu'). REFUSED BY NAME: a general float beta."""
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
    def _beta(self):
        b = {"frobenius": 2.0, "kullback-leibler": 1.0, "itakura-saito": 0.0}.get(self.beta_loss, self.beta_loss)
        if b not in (0.0, 1.0, 2.0):
            raise ValueError("beta_loss: 'frobenius', 'kullback-leibler' and 'itakura-saito' are carried; "
                             "a general float beta is not")
        return float(b)

    def _validate(self, M):
        beta = self._beta()
        if beta != 2.0 and self.solver != "mu":
            raise ValueError("Invalid beta_loss parameter: solver 'cd' does not handle beta_loss other than 'frobenius'")
        if self.solver not in ("cd", "mu"):
            raise ValueError(f"Invalid solver parameter: got {self.solver!r} instead of one of {{'cd', 'mu'}}")
        if _any_negative(M):
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
        """sklearn `_beta_divergence(..., square_root=True)`."""
        beta = self._beta()
        if beta == 2.0:
            return k.ew("sqrt", k.total(k.ew("sqdiff", M, k.mm(W, H)))).s[0]
        WH = k.ew("maxs", k.mm(W, H), s=_F32_EPS)
        keep = k.ew("gts", M, s=_F32_EPS)                       # X > EPSILON
        div = k.ew("div", M, WH)
        if beta == 1.0:
            sum_wh = k.mm(k.colsum(W), k.rowsum(H)).s[0]
            xlog = k.total(k.ew("mul", k.ew("mul", M, k.ew("logs", div, s=1.1754943508222875e-38)), keep)).s[0]
            xs = k.total(k.ew("mul", M, keep)).s[0]
            res = xlog + sum_wh - xs
        else:
            dsum = k.total(k.ew("mul", div, keep)).s[0]
            lsum = k.total(k.ew("mul", k.ew("logs", div, s=1.1754943508222875e-38), keep)).s[0]
            res = dsum - M.r * M.c - lsum
        return math.sqrt(2 * res) if res > 0 else 0.0

    def _mu_ratio(self, k, M, W, H, beta):
        """(X / WH) for KL, (X / WH^2) for IS, WH clamped at EPSILON; and WH^(beta-1)."""
        WH = k.ew("maxs", k.mm(W, H), s=_F32_EPS)
        if beta == 1.0:
            return k.ew("div", M, WH), None
        return k.ew("div", k.ew("div", M, WH), WH), k.ew("recip", WH)

    # ---- solvers
    def _mu(self, k, M, W, H, update_H, regs):
        l1W, l1H, l2W, l2H = regs
        beta = self._beta()
        if beta != 2.0:
            return self._mu_beta(k, M, W, H, update_H, regs, beta)
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

    def _mu_beta(self, k, M, W, H, update_H, regs, beta):
        """sklearn `_multiplicative_update_w/_h` for beta 1 (KL) and 0 (IS):
        the ratio matrix through H^T (W^T), the denominator H's row sums (W's
        column sums, a zero replaced by 1) for KL or WH^-1 H^T (W^T WH^-1)
        for IS, a zero denominator replaced by EPSILON, and for IS the
        update's square root (gamma = 1 / (2 - beta))."""
        l1W, l1H, l2W, l2H = regs
        err0 = self._err(k, M, W, H)
        prev = err0
        it = 0
        for it in range(1, self.max_iter + 1):
            R, P = self._mu_ratio(k, M, W, H, beta)
            num = k.mm(R, H, tb=True)
            den = k.rowsum(H).T if beta == 1.0 else k.mm(P, H, tb=True)
            if beta == 1.0:
                den = k.ew("add", _M.zeros(W.r, W.c), den)
            if l1W > 0:
                den = k.ew("adds", den, s=l1W)
            if l2W > 0:
                den = k.ew("axpy", den, W, s=l2W)
            den = k.ew("select", k.ew("abs", den), den, k.const(_F32_EPS), s=0.0)
            delta = k.ew("div", num, den)
            if beta == 0.0:
                delta = k.ew("sqrt", delta)
            W = k.ew("mul", W, delta)
            if update_H:
                R, P = self._mu_ratio(k, M, W, H, beta)
                num = k.mm(W, R, ta=True)
                if beta == 1.0:
                    ws = k.colsum(W)
                    ws = k.ew("select", k.ew("abs", ws), ws, k.const(1.0), s=0.0)
                    den = k.ew("add", _M.zeros(H.r, H.c), ws.T)
                else:
                    den = k.mm(W, P, ta=True)
                if l1H > 0:
                    den = k.ew("adds", den, s=l1H)
                if l2H > 0:
                    den = k.ew("axpy", den, H, s=l2H)
                den = k.ew("select", k.ew("abs", den), den, k.const(_F32_EPS), s=0.0)
                delta = k.ew("div", num, den)
                if beta == 0.0:
                    delta = k.ew("sqrt", delta)
                H = k.ew("mul", H, delta)
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

    def _perm(self, k, kc):
        """The coordinate order of one `_update_coordinate_descent` call: the
        identity, or with shuffle=True a Philox permutation (a sort of draws,
        ties to the lower index; DEVIATION 5306)."""
        if not self.shuffle:
            return list(range(kc))
        self._draws = getattr(self, "_draws", 0) + 1
        u = k.rand(1, kc, _seed_of(self.random_state), 200 + self._draws, 0).s
        return sorted(range(kc), key=lambda i: (u[i], i))

    def _cd(self, k, M, W, H, update_H, regs):
        l1W, l1H, l2W, l2H = regs
        self._draws = 0
        Ht = H.T
        W = W.copy()
        v_init = None
        it = 0
        for it in range(1, self.max_iter + 1):
            viol = self._cd_side(k, M, W, Ht, l1W, l2W, self._perm(k, W.c), False)
            if update_H:
                viol += self._cd_side(k, M, Ht, W, l1H, l2H, self._perm(k, W.c), True)
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
    """The n x n identity, built whole (row major, a 1 every n + 1 words):
    no per-row loop. The resident kit's `diag_mask` makes it on the device."""
    if n < 1:
        return _M.zeros(0, 0)
    return _M(array.array("f", [1.0] + [0.0] * n) * (n - 1) + array.array("f", [1.0]), n, n)


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

    Each iteration's SVD of the scaled data is the QR + one-sided Jacobi SVD
    of decomposition/ (the eigh of the d x d Gram when n < d).
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
        # Xc = Q R once; the scaled data Xc D / sqrt(n) then has the singular
        # values and right vectors of the d x d R D / sqrt(n) (Q orthogonal)
        Rx = k.qr_r(Xc) if n >= d else None
        old_ll = -math.inf
        loglike = []
        it = 0
        W = None
        for it in range(1, self.max_iter + 1):
            sqrt_psi = k.ew("adds", k.ew("sqrt", psi), s=SMALL)
            if n >= d:
                sv, Vt = k.svd(k.ew("scale", k.ew("div", Rx, sqrt_psi), s=1.0 / nsqrt))
                s2 = k.ew("sq", sv)
            else:
                Z = k.ew("scale", k.ew("div", Xc, sqrt_psi), s=1.0 / nsqrt)
                ev, V = k.eigh(k.mm(Z, Z, ta=True))
                order = list(range(d - 1, -1, -1))
                s2 = k.ew("maxs", ev.take_cols(order), s=0.0)
                Vt = V.take_cols(order).T
            Vt = Vt.rows(0, nc)
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


# ================================================================ TruncatedSVD explained variance
def _exact_eigen_solver(name, who):
    """Isomap's and LLE's eigen_solver: 'auto' and 'dense' name an exact
    solve, which is what runs (LLE's 'auto' past 200 rows: shift-invert
    subspace iteration to convergence, `_lle_smallest`). 'arpack' (a
    Lanczos) is a different algorithm and is refused by name rather than run
    as another solve (a silent substitution; x_decomp/NOT_IMPLEMENTED.tsv)."""
    if name == "arpack":
        raise NotImplementedError(
            f"{who}: eigen_solver='arpack' is not implemented; the exact solve "
            "runs for 'auto' and 'dense', and running it under ARPACK's name "
            "would be a silent substitution (x_decomp/NOT_IMPLEMENTED.tsv)")
    if name not in ("auto", "dense"):
        raise ValueError(f"{who}: eigen_solver must be 'auto', 'dense' or 'arpack', got {name!r}")


# ================================================================ PCA n_components='mle'
_LOG2, _LOGPI, _LOG2PI_D = 0.6931471805599453, 1.1447298858494002, 1.8378770664093453


def _pca_mle_rank(spectrum, n_samples, mode):
    """scikit-learn `_pca.py::_infer_dimension` / `_assess_dimension`
    (Minka's MLE of the PCA rank) over the full explained-variance spectrum.
    Every log and gammaln of data runs in the cells (float32 results, one
    call per rank); the sums and products are IEEE double in sklearn's order
    and the constants log 2, log pi and log 2 pi are literals, so the rank is
    the same on every column. The argmax takes the first maximum, as numpy."""
    k = _Kit(mode)
    d = len(spectrum)
    sp = [float(v) for v in spectrum]
    tiny = 1.1754943508222875e-38

    def logs(values):
        return k.ew("logs", _M.of(values, 1, len(values)), s=tiny).s if values else []

    lsp = logs(sp)
    lg = k.ew("lgamma", _M.of([(d - i + 1) / 2.0 for i in range(1, d + 1)], 1, d)).s
    logn = logs([float(n_samples)])[0]
    best, arg = -math.inf, 0
    for rank in range(1, d):
        if sp[rank - 1] < 1e-15:
            continue
        pu = -rank * _LOG2
        for i in range(1, rank + 1):
            pu += lg[i - 1] - _LOGPI * (d - i + 1) / 2.0
        pl = -_dsum(lsp[:rank]) * n_samples / 2.0
        v = max(1e-15, _dsum(sp[rank:]) / (d - rank))
        logv = logs([v])[0]
        pv = -logv * n_samples * (d - rank) / 2.0
        m = d * rank - rank * (rank + 1.0) / 2.0
        pp = _LOG2PI_D * (m + rank) / 2.0
        spv = sp[:rank] + [v] * (d - rank)
        terms = [(sp[i] - sp[j]) * (1.0 / spv[j] - 1.0 / spv[i]) for i in range(rank) for j in range(i + 1, d)]
        pa = 0.0
        for t in logs(terms):
            pa += t + logn
        ll = pu + pl + pv + pp - pa / 2.0 - rank * logn / 2.0
        if ll > best:
            best, arg = ll, rank
    return arg


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
    """scipy.linalg.lu_solve (LAPACK getrs): solve A x = b (trans=0) or
    A^T x = b (trans=1, and 2, which is the same for a real matrix) from
    `lu_factor`'s pair. `b` is n or n x nrhs; a zero pivot yields 0 in that
    component (never inf or NaN)."""
    if trans not in (0, 1, 2):
        raise ValueError("trans must be 0, 1 or 2")
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
    X = k.lu_solve(L, pv, B, trans=1 if trans else 0)
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
    """An orthonormal basis of A's columns (m x l, m >= l): two passes of the
    Householder QR's R (decomposition/'s TSQR) and Q = A R^-1 one row per
    thread (x_decomp/cells.mojo `trsm_row`)."""
    return k.orth(A)


def _flip_u(U, Vt):
    """sklearn svd_flip(u_based_decision=True): each column of U signed so its
    largest-|.| entry (first on a tie) is positive; Vt's rows follow."""
    fl = _Kit(_backend.default_mode()).absmax_flags(U, True)
    return U.neg_cols(fl), Vt.neg_rows(fl)


def randomized_svd(M, n_components, *, n_oversamples=10, n_iter="auto", power_iteration_normalizer="auto",
                   transpose="auto", flip_sign=True, random_state=None, numeric_mode=None):
    """sklearn.utils.extmath.randomized_svd (Halko et al.; RAFT
    `linalg/rsvd.cuh` is the same scheme). The Gaussian test matrix is the
    lane's Philox stream (random_state None means 0). Every power iteration
    re-orthonormalizes by the two-pass Householder-R solve whatever `power_iteration_normalizer`
    says: 'LU', 'QR' and 'none' span the same subspace, only the rounding of
    the basis differs. The small SVD of Q^T M is exact (Gram eigh).
    Returns (U, s, Vt)."""
    k = _Kit(_mode(numeric_mode))
    U, S, Vt = _rsvd_core(k, _M.from_input(M, "M"), n_components, n_oversamples, n_iter,
                          power_iteration_normalizer, transpose, flip_sign, random_state)
    kc = n_components
    return U.out(), S.out((kc,)), Vt.out()


def _rsvd_core(k, A, n_components, n_oversamples, n_iter, power_iteration_normalizer, transpose, flip_sign,
               random_state):
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
    # The orth rank guard (DEVIATION 5318) leaves a numerically dependent
    # column of the range exactly 0. Q^T M then has a zero row, and the
    # one-sided Jacobi SVD of a rank-deficient matrix rotates the rounding
    # noise of the column it drives to zero forever, so the small SVD runs on
    # the live columns only; the components past the numerical rank are 0
    # (singular value 0, zero vectors), as the zero Q column already says.
    live = [j for j, v in enumerate(k.colsum(k.ew("abs", Q)).s) if v != 0.0]
    if not live:
        raise ValueError("randomized_svd: M is numerically zero")
    full = Q.c
    if len(live) < full:
        Q = Q.take_cols(live)
    B = k.mm(Q, A, ta=True)
    Uh, S, Vt = _thin_svd(k, B, min(B.r, B.c), u_based=True)
    if S.c < full:
        pad = full - S.c
        S = _M(S.s + array.array("f", [0.0]) * pad, 1, full)
        Vt = _M(Vt.s + array.array("f", [0.0]) * (pad * Vt.c), full, Vt.c)
        Uh = Uh.T
        Uh = _M(Uh.s + array.array("f", [0.0]) * (pad * Uh.c), full, Uh.c).T
    U = k.mm(Q, Uh)
    if flip_sign:
        if not transpose:
            U, Vt = _flip_u(U, Vt)
        else:
            U, Vt = _flip_u(Vt.T, U.T)
            U, Vt = Vt.T, U.T
    kc = n_components
    if transpose:
        return Vt.rows(0, kc).T, S.cols(0, kc), U.cols(0, kc).T
    return U.cols(0, kc), S.cols(0, kc), Vt.rows(0, kc)


#: x_decomp/tsqr_core.mojo TS_MAX_N: the widest [A | B] the TSQR takes
_TS_MAX_N = 512


def _f32_input(X, name, ndim):
    """A C-contiguous float32 Array of X (no copy when it already is one),
    NaN/inf refused as `_M.from_input` refuses them."""
    if _is_sparse(X):
        X = X.toarray()
    a = as_f32_c(X, ndim=ndim, name=name)[0]
    if min(a.shape) == 0:
        raise ValueError(f"{name}: a nonempty input is required")
    fin = _host_all_finite(a)
    if fin is False:
        raise ValueError(f"{name}: input must be finite; NaN/inf are unsupported")
    if fin is None:
        _M.from_input(a if a.ndim == 2 else a.reshape(a.shape[0], 1), name)
    return a


def _tsqr_lstsq_on(m, nn, nrhs):
    import os
    return (os.environ.get("MOJOLEARN_LINALG_TSQR", "1") != "0"
            and nn >= 1 and nrhs >= 1 and nn + nrhs <= _TS_MAX_N and m >= nn + nrhs)


def _tsqr_lstsq_core(k, a_arr, b_arr, m, nn, nrhs, rcond, equilibrate=False):
    """(X nn x nrhs, residuals _M 1 x nrhs or None, rank, S 1 x nn) of
    min ||A X - B|| through the blocked TSQR of [A | B] (lane
    neural-pass140; x_decomp/tsqr_core.mojo): R_aug = [[R, C], [0, R22]]
    with C = Q^T B, so no pass over the rows after the factorization and no
    Gram matrix. The minimum-norm solution is the SVD of the small R
    (`_svd_tall`: S descending, U_R orthonormal): X = V diag(1/s) U_R^T C,
    singular values at or below rcond * s_max dropped, as before; the
    residuals (rank == nn < m only) are the squared column norms of R22,
    which is ||b - a x||^2 at the least-squares solution.

    `equilibrate` (LinearRegression, lane apple-fast-tsqr): before the SVD
    every column j of R is multiplied by the exact power of two s_j that
    DEVIATION 2620 picks for the Gram diagonal ||R e_j||^2 = ||A e_j||^2
    (op p2scale, x_decomp/cells.mojo), so the rcond cutoff sees the design
    in balanced units as the normal equations route did, and X is
    multiplied by s after. A power of two scales without rounding, so
    R S is what the TSQR of A S gives; every step stays on the binding's
    cells (the same words on every column)."""
    from ._linalg_impl import _svd_tall
    from ._buffer import addr_ro
    n = nn + nrhs
    Ra = _M.zeros(n, n)
    # ORDER MATCHES x_decomp/api.mojo tsqr_r_py: (a, b, r_out), (m, d, nrhs, keep)
    k.b.x_decomp_tsqr_r(addr_ro(a_arr, name="a"), addr_ro(b_arr, name="b"), Ra.addr,
                        [int(m), int(nn), int(nrhs), 0])
    top = Ra.rows(0, nn)
    R, C = top.cols(0, nn), top.cols(nn, n)
    sc = None
    if equilibrate:
        sc = k.ew("p2scale", k.colsum(k.ew("sq", R)))
        R = k.ew("mul", R, sc)
    Ur, S, Vt = _svd_tall(k, R, False)
    cut = _f32(S.s[0] * rcond)
    rank = sum(1 for v in S.s if v > cut)
    inv = k.ew("recip", k.ew("select", S, S, _M.zeros(1, 1), s=cut))
    X = k.mm(Vt, k.ew("mul", k.mm(Ur, C, ta=True), inv.T), ta=True)
    if sc is not None:
        X = k.ew("mul", X, sc.T)
    res = None
    if rank == nn and m > nn:
        res = k.colsum(k.ew("sq", Ra.rows(nn, n).cols(nn, n)))
    return X, res, rank, S


def _lstsq_tsqr(k, a, b, vec, rcond):
    """`lstsq` through the blocked TSQR when it serves the shape (tall,
    [A | B] at most _TS_MAX_N wide), else None (the SVD route below)."""
    a_arr = _f32_input(a, "a", 2)
    m, nn = a_arr.shape
    b_arr = _f32_input(b, "b", 1 if vec else 2)
    nrhs = 1 if vec else b_arr.shape[1]
    if b_arr.shape[0] != m:
        raise ValueError("Incompatible dimensions")
    if not _tsqr_lstsq_on(m, nn, nrhs):
        return None
    if rcond is None:
        rcond = _F32_EPS * max(m, nn)
    X, res, rank, S = _tsqr_lstsq_core(k, a_arr, b_arr, m, nn, nrhs, rcond)
    resid = res.out((nrhs,)) if res is not None else _M.zeros(1, 0).out((0,))
    return (X.out((nn,)) if vec else X.out()), resid, rank, S.out((nn,))


def lstsq(a, b, rcond=None, *, numeric_mode=None):
    """numpy.linalg.lstsq: (x, residuals, rank, s), singular values
    descending, those at or below rcond * s_max treated as zero (rcond None:
    float32 eps * max(M, N)); residuals are the squared column norms of
    b - a x when rank == N < M, else empty. A tall a with N + nrhs <= 512
    takes the blocked TSQR of [a | b] and the SVD of its small R
    (`_tsqr_lstsq_core`, lane neural-pass140; MOJOLEARN_LINALG_TSQR=0 keeps
    the route below); any other shape the QR + one-sided Jacobi SVD of a (or
    of a^T)."""
    k = _Kit(_mode(numeric_mode))
    vec = len(getattr(b, "shape", ())) == 1 or (not hasattr(b, "shape") and not isinstance(b[0], (list, tuple)))
    got = _lstsq_tsqr(k, a, b, vec, rcond)
    if got is not None:
        return got
    A = _M.from_input(a, "a")
    m, nn = A.r, A.c
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


# ================================================================ PLS / CCA
def _pinv(k, A, rcond=None):
    """Moore-Penrose pseudo-inverse through the thin SVD (Gram eigh):
    V diag(1/s) U^T, singular values at or below cond * s_max dropped
    (cond = max(shape) * float32 eps, sklearn `_pinv2_old`'s rule)."""
    r = min(A.r, A.c)
    U, S, Vt = _thin_svd(k, A, r, u_based=True)
    cond = (max(A.r, A.c) * _F32_EPS) if rcond is None else rcond
    cut = _f32(S.s[0] * cond) if r else 0.0
    inv = k.ew("recip", k.ew("select", S, S, _M.zeros(1, 1), s=cut))
    return k.mm(Vt, k.ew("mul", U, inv), ta=True, tb=True)


def _dot(k, a, b):
    """a . b for two column vectors (n x 1): one gemm cell, rows ascending."""
    return k.mm(a, b, ta=True)


class _PLS(_Base):
    """sklearn `cross_decomposition/_pls.py::_PLS` (NIPALS): `fit`,
    `_get_first_singular_vectors_power_method`, `_center_scale_xy`,
    `_get_first_singular_vectors_svd` (PLSCanonical algorithm='svd'),
    `_svd_flip_1d`, `transform`, `inverse_transform`, `predict`."""
    _parameters = ("n_components", "scale", "max_iter", "tol", "copy", "numeric_mode")
    _deflation = "regression"
    _pmode = "A"

    def __init__(self, n_components=2, *, scale=True, max_iter=500, tol=1e-06, copy=True, numeric_mode=None):
        self.n_components, self.scale, self.max_iter, self.tol, self.copy = n_components, scale, max_iter, tol, copy
        self.numeric_mode = numeric_mode

    def _xy(self, X, Y):
        M = _M.from_input(X)
        vec = Y is not None and (len(getattr(Y, "shape", ())) == 1 or
                                 (not hasattr(Y, "shape") and not isinstance(Y[0], (list, tuple))))
        Ym = None
        if Y is not None:
            Ym = _M.from_input(_row_of(Y), "Y").T if vec else _M.from_input(Y, "Y")
        return M, Ym, vec

    def _center_scale(self, k, A):
        n = A.r
        mean = k.colmean(A)
        Ac = k.ew("sub", A, mean)
        if self.scale:
            std = k.ew("sqrt", k.ew("scale", k.colsum(k.ew("sq", Ac)), s=1.0 / (n - 1)))
            std = k.ew("select", std, std, k.const(1.0), s=0.0)
            Ac = k.ew("div", Ac, std)
        else:
            std = k.const(1.0, 1, A.c)
        return Ac, mean, std

    def _power(self, k, X, Y, norm_y):
        eps = _F32_EPS
        y_score = None
        for j in range(Y.c):
            col = Y.cols(j, j + 1)
            if any(abs(v) > eps for v in col.s):
                y_score = col
                break
        if y_score is None:
            raise StopIteration("y residual is constant")
        xw_old = None
        if self._pmode == "B":
            Xp, Yp = _pinv(k, X), _pinv(k, Y)
        it = 0
        for it in range(1, self.max_iter + 1):
            if self._pmode == "B":
                xw = k.mm(Xp, y_score)
            else:
                xw = k.ew("div", k.mm(X, y_score, ta=True), _dot(k, y_score, y_score))
            xw = k.ew("div", xw, k.ew("adds", k.ew("sqrt", _dot(k, xw, xw)), s=eps))
            x_score = k.mm(X, xw)
            if self._pmode == "B":
                yw = k.mm(Yp, x_score)
            else:
                yw = k.ew("div", k.mm(Y, x_score, ta=True), _dot(k, x_score, x_score))
            if norm_y:
                yw = k.ew("div", yw, k.ew("adds", k.ew("sqrt", _dot(k, yw, yw)), s=eps))
            y_score = k.ew("div", k.mm(Y, yw), k.ew("adds", _dot(k, yw, yw), s=eps))
            if Y.c == 1:
                break
            if xw_old is not None:
                diff = k.ew("sub", xw, xw_old)
                if _dot(k, diff, diff).s[0] < self.tol:
                    break
            else:
                diff = k.ew("adds", xw, s=-100.0)
                if _dot(k, diff, diff).s[0] < self.tol:
                    break
            xw_old = xw
        return xw, yw, it

    def fit(self, X, Y):
        self.numeric_mode_ = _mode(self.numeric_mode)
        k = self._kit()
        M, Ym, vec = self._xy(X, Y)
        if Ym is None:
            raise ValueError("Y is required")
        n, p, q = M.r, M.c, Ym.c
        nc = int(self.n_components)
        bound = p if self._deflation == "regression" else min(n, p, q)
        if not 1 <= nc <= bound:
            raise ValueError(f"n_components == {nc}, while 1 <= n_components <= {bound} is required")
        self._predict_1d = vec
        Xk, self._x_mean, self._x_std = self._center_scale(k, M)
        Yk, self._y_mean, self._y_std = self._center_scale(k, Ym)
        norm_y = self._deflation == "canonical"
        xw_c, yw_c, xs_c, ys_c, xl_c, yl_c = [], [], [], [], [], []
        self.n_iter_ = []
        thr = 10 * _F32_EPS
        for _c in range(nc):
            # Yk columns that are all below 10 eps are set to zero: per
            # column, the count of |y| >= thr (-|y| <= -thr, exact) on the
            # device; q values come back
            live = k.colsum(k.ew("le", k.ew("scale", k.ew("abs", Yk), s=-1.0), _M.of([-thr], 1, 1)))
            dead = [v == 0.0 for v in live.s]
            if any(dead):
                Yk = k.ew("mul", Yk, _M.of([0.0 if d else 1.0 for d in dead], 1, q))
            try:
                if getattr(self, "algorithm", "nipals") == "svd":
                    # `_get_first_singular_vectors_svd`: the first singular pair of X^T Y
                    Cxy = k.mm(Xk, Yk, ta=True)
                    U, _, Vt = _thin_svd(k, Cxy, 1, u_based=True)
                    xw, yw, it = U.cols(0, 1), Vt.rows(0, 1).T, 0
                else:
                    xw, yw, it = self._power(k, Xk, Yk, norm_y)
            except StopIteration:
                import warnings
                warnings.warn(f"y residual is constant at iteration {_c}", stacklevel=2)
                break
            if getattr(self, "algorithm", "nipals") != "svd":
                self.n_iter_.append(it)
            # _svd_flip_1d: the largest-|.| entry of x_weights positive
            best, arg = -1.0, 0
            for j, v in enumerate(xw.s):
                if abs(v) > best:
                    best, arg = abs(v), j
            if xw.s[arg] < 0:
                xw, yw = xw.neg_cols([True]), yw.neg_cols([True])
            x_scores = k.mm(Xk, xw)
            y_ss = k.const(1.0) if norm_y else _dot(k, yw, yw)
            y_scores = k.ew("div", k.mm(Yk, yw), y_ss)
            x_load = k.ew("div", k.mm(Xk, x_scores, ta=True), _dot(k, x_scores, x_scores))
            Xk = k.ew("sub", Xk, k.mm(x_scores, x_load, tb=True))
            if self._deflation == "canonical":
                y_load = k.ew("div", k.mm(Yk, y_scores, ta=True), _dot(k, y_scores, y_scores))
                Yk = k.ew("sub", Yk, k.mm(y_scores, y_load, tb=True))
            else:
                y_load = k.ew("div", k.mm(Yk, x_scores, ta=True), _dot(k, x_scores, x_scores))
                Yk = k.ew("sub", Yk, k.mm(x_scores, y_load, tb=True))
            xw_c.append(xw); yw_c.append(yw); xs_c.append(x_scores); ys_c.append(y_scores)
            xl_c.append(x_load); yl_c.append(y_load)
        self.x_weights_m_ = _hstack(*xw_c)
        self.y_weights_m_ = _hstack(*yw_c)
        self.x_loadings_m_ = _hstack(*xl_c)
        self.y_loadings_m_ = _hstack(*yl_c)
        self._x_scores_m, self._y_scores_m = _hstack(*xs_c), _hstack(*ys_c)
        self.x_rotations_m_ = k.mm(self.x_weights_m_, _pinv(k, k.mm(self.x_loadings_m_, self.x_weights_m_, ta=True)))
        self.y_rotations_m_ = k.mm(self.y_weights_m_, _pinv(k, k.mm(self.y_loadings_m_, self.y_weights_m_, ta=True)))
        coef = k.mm(self.x_rotations_m_, self.y_loadings_m_, tb=True)          # p x q
        coef = k.ew("div", k.ew("mul", coef, self._y_std), self._x_std.T).T     # q x p
        self.coef_m_ = coef
        for name in ("x_weights", "y_weights", "x_loadings", "y_loadings", "x_rotations", "y_rotations"):
            setattr(self, name + "_", getattr(self, name + "_m_").out())
        self.coef_ = coef.out()
        self.intercept_ = self._y_mean.out((q,))
        self._x_scores, self._y_scores = self._x_scores_m.out(), self._y_scores_m.out()
        self.n_features_in_ = p
        self.components_m_ = self.x_rotations_m_
        return self

    def _check_fitted(self):
        if not hasattr(self, "coef_m_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit first")

    def transform(self, X, y=None, Y=None, copy=True):
        self._check_fitted()
        Y = y if y is not None else Y
        k = self._kit()
        M, Ym, vec = self._xy(X, Y)
        xs = k.mm(k.ew("div", k.ew("sub", M, self._x_mean), self._x_std), self.x_rotations_m_)
        if Ym is None:
            return xs.out()
        ys = k.mm(k.ew("div", k.ew("sub", Ym, self._y_mean), self._y_std), self.y_rotations_m_)
        return xs.out(), ys.out()

    def inverse_transform(self, X, y=None, Y=None):
        self._check_fitted()
        k = self._kit()
        Z = _M.from_input(X)
        Xr = k.ew("add", k.ew("mul", k.mm(Z, self.x_loadings_m_, tb=True), self._x_std), self._x_mean)
        Y = y if y is not None else Y
        if Y is None:
            return Xr.out()
        W = _M.from_input(Y, "Y")
        Yr = k.ew("add", k.ew("mul", k.mm(W, self.y_loadings_m_, tb=True), self._y_std), self._y_mean)
        return Xr.out(), Yr.out()

    def predict(self, X, copy=True):
        self._check_fitted()
        k = self._kit()
        M = _M.from_input(X)
        P = k.ew("add", k.mm(k.ew("sub", M, self._x_mean), self.coef_m_, tb=True), self._y_mean)
        return P.out((P.r,)) if self._predict_1d else P.out()

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)


class PLSRegression(_PLS):
    """sklearn.cross_decomposition.PLSRegression: NIPALS, mode A, regression deflation."""
    _deflation, _pmode = "regression", "A"

    def fit(self, X, y=None, Y=None):
        super().fit(X, y if y is not None else Y)
        self.x_scores_, self.y_scores_ = self._x_scores, self._y_scores
        return self


class PLSCanonical(_PLS):
    """sklearn.cross_decomposition.PLSCanonical (algorithm='nipals'): mode A,
    canonical deflation."""
    _deflation, _pmode = "canonical", "A"
    _parameters = ("n_components", "scale", "algorithm", "max_iter", "tol", "copy", "numeric_mode")

    def __init__(self, n_components=2, *, scale=True, algorithm="nipals", max_iter=500, tol=1e-06, copy=True,
                 numeric_mode=None):
        if algorithm not in ("nipals", "svd"):
            raise ValueError("PLSCanonical: algorithm must be 'nipals' or 'svd'")
        self.algorithm = algorithm
        super().__init__(n_components, scale=scale, max_iter=max_iter, tol=tol, copy=copy, numeric_mode=numeric_mode)


class CCA(_PLS):
    """sklearn.cross_decomposition.CCA: NIPALS, mode B (the pseudo-inverses of
    X and Y), canonical deflation."""
    _deflation, _pmode = "canonical", "B"


# ================================================================ dictionary learning / sparse PCA
_SPARSE_ALGOS = ("lasso_lars", "lasso_cd", "lars", "omp", "threshold")


def _sparse_encode(k, X, D, algorithm, alpha=None, n_nonzero_coefs=None, init=None, max_iter=1000, positive=False):
    """sklearn `_dict_learning.py::_sparse_encode` + `_sparse_encode_precomputed`.
    'lasso_lars' solves the same Lasso as 'lasso_cd' (by coordinate descent
    from zero: the Lasso optimum does not depend on the path taken to it);
    'lasso_cd' warm-starts from `init`; 'omp' and 'threshold' as sklearn.
    'lars' is sklearn's Lars(fit_intercept=False, n_nonzero_coefs) of each
    row on the dictionary's columns, on the Gram, every row at once
    (x_decomp/cells.mojo `lars_row`, x_linear/lars.mojo's lar path)."""
    if algorithm not in _SPARSE_ALGOS:
        raise ValueError(f"algorithm={algorithm!r} is not carried; one of {_SPARSE_ALGOS}")
    n, m = X.r, X.c
    kc = D.r
    if algorithm in ("omp", "lars"):
        reg = n_nonzero_coefs if n_nonzero_coefs is not None else min(max(m / 10, 1), kc)
    else:
        reg = alpha if alpha is not None else 1.0
    if algorithm == "lars":
        # cgr-decomp (2026-10-03): every row's Lars on the Gram at once, one
        # device thread a row (x_decomp/cells.mojo `lars_row`), in place of
        # the linear lane's Lars fitted one row at a time from Python
        return k.lars_rows(k.mm(D, D, tb=True), k.mm(X, D, tb=True), m, int(reg))
    Q = k.mm(X, D, tb=True)                      # n x k: row i is D x_i
    if algorithm == "threshold":
        code = k.ew("soft", Q, s=reg)
        return k.ew("maxs", code, s=0.0) if positive else code
    G = k.mm(D, D, tb=True)
    if algorithm == "omp":
        return k.omp_rows(G, Q, int(reg))
    W = init.copy() if (algorithm == "lasso_cd" and init is not None) else _M.zeros(n, kc)
    return k.lasso_rows(G, Q, W, float(reg), max_iter, 1e-8, positive)


def sparse_encode(X, dictionary, *, algorithm="lasso_lars", n_nonzero_coefs=None, alpha=None, init=None,
                  max_iter=1000, positive=False, numeric_mode=None, **_ignored):
    """sklearn.decomposition.sparse_encode (see `_sparse_encode`)."""
    k = _Kit(_mode(numeric_mode))
    init_m = None if init is None else _M.from_input(init, "init")
    return _sparse_encode(k, _M.from_input(X), _M.from_input(dictionary, "dictionary"), algorithm, alpha,
                          n_nonzero_coefs, init_m, max_iter, positive).out()


def _resample_atom(k, Y, seed, counter):
    """sklearn `_update_dict`'s unused-atom branch on the Philox stream: a
    row of Y chosen uniformly, plus N(0, 0.01 * std(row) or 0.01) noise."""
    u = k.rand(1, 1, seed, 40 + 2 * counter, 0).s[0]
    idx = min(int(u * Y.r), Y.r - 1)
    row = Y.rows(idx, idx + 1)
    mean = k.ew("scale", k.total(row), s=1.0 / row.c)
    std = k.ew("sqrt", k.ew("scale", k.total(k.ew("sqdiff", row, mean)), s=1.0 / row.c)).s[0]
    noise = k.ew("scale", k.rand(1, row.c, seed, 41 + 2 * counter, 1), s=0.01 * (std or 1.0))
    return k.ew("add", row, noise)


def _update_dict(k, D, Y, code, A=None, B=None, positive=False, seed=0, counter=None):
    """sklearn `_dict_learning.py::_update_dict`: block coordinate descent over
    the atoms in order, each projected onto the unit ball. Returns (D, code)."""
    if A is None:
        A = k.mm(code, code, ta=True)
    if B is None:
        B = k.mm(Y, code, ta=True)
    rows = [D.rows(j, j + 1) for j in range(D.r)]
    zero_cols = []
    for j in range(D.r):
        ajj = A.s[j * A.c + j]
        if ajj > 1e-6:
            Dcur = _vstack(*rows)
            upd = k.ew("sub", B.cols(j, j + 1).T, k.mm(A.rows(j, j + 1), Dcur))
            rows[j] = k.ew("add", rows[j], k.ew("div", upd, k.const(ajj)))
        else:
            c = counter[0] if counter is not None else 0
            rows[j] = _resample_atom(k, Y, seed, c)
            if counter is not None:
                counter[0] += 1
            zero_cols.append(j)
        if positive:
            rows[j] = k.ew("maxs", rows[j], s=0.0)
        nrm = k.ew("sqrt", k.total(k.ew("sq", rows[j])))
        rows[j] = k.ew("div", rows[j], k.ew("maxs", nrm, s=1.0))
    if zero_cols:
        code = code.copy()
        for i in range(code.r):
            for j in zero_cols:
                code.s[i * code.c + j] = 0.0
    return _vstack(*rows), code


def _cost(k, X, code, D, alpha):
    r = k.total(k.ew("sqdiff", X, k.mm(code, D))).s[0]
    l1 = k.total(k.ew("abs", code)).s[0]
    return 0.5 * r + alpha * l1


def _dict_learning(k, X, nc, alpha, max_iter, tol, method, seed, code_init=None, dict_init=None,
                   positive_dict=False, positive_code=False, method_max_iter=1000):
    """sklearn `_dict_learning.py::_dict_learning`: returns (code, D, errors, n_iter)."""
    if code_init is not None and dict_init is not None:
        code, D = code_init, dict_init
    else:
        r = min(X.r, X.c)
        U, S, Vt = _thin_svd(k, X, r, u_based=True)
        code, D = U, k.ew("mul", Vt, S.T)
    r = D.r
    if nc <= r:
        code, D = code.cols(0, nc), D.rows(0, nc)
    else:
        code = _hstack(code, _M.zeros(code.r, nc - r))
        D = _vstack(D, _M.zeros(nc - r, D.c))
    errors = []
    ii = 0
    counter = [0]
    for ii in range(1, max_iter + 1):
        code = _sparse_encode(k, X, D, method, alpha=alpha, init=code, max_iter=method_max_iter,
                              positive=positive_code)
        D, code = _update_dict(k, D, X, code, positive=positive_dict, seed=seed, counter=counter)
        errors.append(_cost(k, X, code, D, alpha))
        if len(errors) > 1:
            if errors[-2] - errors[-1] < tol * errors[-1]:
                break
    return code, D, errors, ii


class _SparseCoding(_Base):
    def _encode(self, X):
        self._check()
        k = self._kit()
        M = _M.from_input(X)
        ta = getattr(self, "transform_alpha", None)
        if ta is None:
            ta = getattr(self, "alpha", 1.0)
        code = _sparse_encode(k, M, self.components_m_, self.transform_algorithm,
                              alpha=ta, n_nonzero_coefs=self.transform_n_nonzero_coefs,
                              max_iter=self.transform_max_iter, positive=self.positive_code)
        if self.split_sign:
            code = _hstack(k.ew("maxs", code, s=0.0), k.ew("scale", k.ew("mins", code, s=0.0), s=-1.0))
        return code

    def transform(self, X):
        return self._encode(X).out()

    def inverse_transform(self, X):
        self._check()
        k = self._kit()
        C = _M.from_input(X, "code")
        D = self.components_m_
        if self.split_sign:
            h = D.r
            C = k.ew("sub", C.cols(0, h), C.cols(h, 2 * h))
        return k.mm(C, D).out()


def _check_sparse_algos(fit_algorithm, transform_algorithm):
    if fit_algorithm not in ("lars", "cd"):
        raise ValueError("fit_algorithm must be 'lars' or 'cd'")
    if transform_algorithm not in _SPARSE_ALGOS:
        raise ValueError(f"transform_algorithm must be one of {_SPARSE_ALGOS}")


class DictionaryLearning(_SparseCoding):
    """sklearn.decomposition.DictionaryLearning (reference: scikit-learn
    `decomposition/_dict_learning.py`: `_dict_learning`, `_update_dict`,
    `_sparse_encode_precomputed`, `_BaseSparseCoding._transform`). The
    initial SVD is exact (QR + one-sided Jacobi); a resampled unused atom is
    drawn from the Philox stream. fit_algorithm 'lars' solves its Lasso by
    coordinate descent (the same optimum). REFUSED BY NAME: callback."""
    _parameters = ("n_components", "alpha", "max_iter", "tol", "fit_algorithm", "transform_algorithm",
                   "transform_n_nonzero_coefs", "transform_alpha", "split_sign", "random_state",
                   "positive_code", "positive_dict", "transform_max_iter", "numeric_mode")

    def __init__(self, n_components=None, *, alpha=1, max_iter=1000, tol=1e-8, fit_algorithm="lars",
                 transform_algorithm="omp", transform_n_nonzero_coefs=None, transform_alpha=None, n_jobs=None,
                 code_init=None, dict_init=None, callback=None, verbose=False, split_sign=False, random_state=None,
                 positive_code=False, positive_dict=False, transform_max_iter=1000, numeric_mode=None):
        self.n_components, self.alpha, self.max_iter, self.tol = n_components, alpha, max_iter, tol
        self.fit_algorithm, self.transform_algorithm = fit_algorithm, transform_algorithm
        self.transform_n_nonzero_coefs, self.transform_alpha = transform_n_nonzero_coefs, transform_alpha
        self.n_jobs, self.code_init, self.dict_init, self.callback = n_jobs, code_init, dict_init, callback
        self.verbose, self.split_sign, self.random_state = verbose, split_sign, random_state
        self.positive_code, self.positive_dict = positive_code, positive_dict
        self.transform_max_iter, self.numeric_mode = transform_max_iter, numeric_mode

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        self.numeric_mode_ = _mode(self.numeric_mode)
        _check_sparse_algos(self.fit_algorithm, self.transform_algorithm)
        if self.callback is not None:
            raise NotImplementedError("callback is not carried")
        k = self._kit()
        M = _M.from_input(X)
        nc = self.n_components if self.n_components is not None else M.c
        ci = None if self.code_init is None else _M.from_input(self.code_init, "code_init")
        di = None if self.dict_init is None else _M.from_input(self.dict_init, "dict_init")
        code, D, errors, it = _dict_learning(
            k, M, nc, float(self.alpha), self.max_iter, self.tol, "lasso_" + self.fit_algorithm,
            _seed_of(self.random_state), ci, di, self.positive_dict, self.positive_code, self.transform_max_iter)
        self.components_m_ = D
        self.components_ = D.out()
        self.error_ = errors
        self.n_iter_ = it
        self.n_features_in_ = M.c
        if self.split_sign:
            code = _hstack(k.ew("maxs", code, s=0.0), k.ew("scale", k.ew("mins", code, s=0.0), s=-1.0))
        return code.out()


class MiniBatchDictionaryLearning(_SparseCoding):
    """sklearn.decomposition.MiniBatchDictionaryLearning (reference:
    `_dict_learning.py::MiniBatchDictionaryLearning`: `_initialize_dict`,
    `_minibatch_step`, `_update_inner_stats`, `_check_convergence`). The
    shuffle is a Philox permutation (sort of counter draws, ties to the lower
    index) and the initial dictionary the exact SVD (sklearn: randomized)."""
    _parameters = ("n_components", "alpha", "max_iter", "fit_algorithm", "batch_size", "shuffle",
                   "transform_algorithm", "transform_n_nonzero_coefs", "transform_alpha", "split_sign",
                   "random_state", "positive_code", "positive_dict", "transform_max_iter", "tol",
                   "max_no_improvement", "numeric_mode")

    def __init__(self, n_components=None, *, alpha=1, max_iter=1000, fit_algorithm="lars", n_jobs=None,
                 batch_size=256, shuffle=True, dict_init=None, transform_algorithm="omp",
                 transform_n_nonzero_coefs=None, transform_alpha=None, verbose=False, split_sign=False,
                 random_state=None, positive_code=False, positive_dict=False, transform_max_iter=1000,
                 callback=None, tol=1e-3, max_no_improvement=10, numeric_mode=None):
        self.n_components, self.alpha, self.max_iter, self.fit_algorithm = n_components, alpha, max_iter, fit_algorithm
        self.n_jobs, self.batch_size, self.shuffle, self.dict_init = n_jobs, batch_size, shuffle, dict_init
        self.transform_algorithm, self.transform_n_nonzero_coefs = transform_algorithm, transform_n_nonzero_coefs
        self.transform_alpha, self.verbose, self.split_sign = transform_alpha, verbose, split_sign
        self.random_state, self.positive_code, self.positive_dict = random_state, positive_code, positive_dict
        self.transform_max_iter, self.callback, self.tol = transform_max_iter, callback, tol
        self.max_no_improvement, self.numeric_mode = max_no_improvement, numeric_mode

    def fit(self, X, y=None):
        self.numeric_mode_ = _mode(self.numeric_mode)
        _check_sparse_algos(self.fit_algorithm, self.transform_algorithm)
        if self.callback is not None:
            raise NotImplementedError("callback is not carried")
        k = self._kit()
        M = _M.from_input(X)
        n, m = M.r, M.c
        nc = self.n_components if self.n_components is not None else m
        bs = min(int(self.batch_size), n)
        seed = _seed_of(self.random_state)
        if self.dict_init is not None:
            D = _M.from_input(self.dict_init, "dict_init")
        else:
            r = min(n, m)
            _, S, Vt = _thin_svd(k, M, r, u_based=True)
            D = k.ew("mul", Vt, S.T)
        if nc <= D.r:
            D = D.rows(0, nc)
        else:
            D = _vstack(D, _M.zeros(nc - D.r, m))
        if self.shuffle:
            u = k.rand(1, n, seed, 50, 0).s
            perm = sorted(range(n), key=lambda i: (u[i], i))
            Xt = M.take_rows(perm)
        else:
            Xt = M
        A, B = _M.zeros(nc, nc), _M.zeros(m, nc)
        steps_per_iter = -(-n // bs)
        n_steps = self.max_iter * steps_per_iter
        batches = []
        start = 0
        for _ in range(n // bs):
            batches.append((start, start + bs))
            start += bs
        if start < n:
            batches.append((start, n))
        ewa, ewa_min, no_imp = None, None, 0
        counter = [0]
        step = -1
        for step in range(n_steps):
            a, b = batches[step % len(batches)]
            Xb = Xt.rows(a, b)
            b_n = Xb.r
            code = _sparse_encode(k, Xb, D, "lasso_" + self.fit_algorithm, alpha=float(self.alpha),
                                  max_iter=self.transform_max_iter, positive=self.positive_code)
            cost = _cost(k, Xb, code, D, float(self.alpha)) / b_n
            theta = (step + 1) * b_n if step < b_n - 1 else b_n ** 2 + step + 1 - b_n
            beta = (theta + 1 - b_n) / (theta + 1)
            A = k.ew("add", k.ew("scale", A, s=beta), k.ew("scale", k.mm(code, code, ta=True), s=1.0 / b_n))
            B = k.ew("add", k.ew("scale", B, s=beta), k.ew("scale", k.mm(Xb, code, ta=True), s=1.0 / b_n))
            old = D
            D, code = _update_dict(k, D, Xb, code, A, B, self.positive_dict, seed, counter)
            # _check_convergence
            s1 = step + 1
            if s1 <= min(100, n / b_n):
                continue
            if ewa is None:
                ewa = cost
            else:
                al = min(b_n / (n + 1), 1)
                ewa = ewa * (1 - al) + cost * al
            diff = math.sqrt(k.total(k.ew("sqdiff", D, old)).s[0]) / nc
            if self.tol > 0 and diff <= self.tol:
                break
            if ewa_min is None or ewa < ewa_min:
                no_imp, ewa_min = 0, ewa
            else:
                no_imp += 1
            if self.max_no_improvement is not None and no_imp >= self.max_no_improvement:
                break
        self.n_steps_ = step + 1
        self.n_iter_ = -(-self.n_steps_ // steps_per_iter)
        self.components_m_ = D
        self.components_ = D.out()
        self.n_features_in_ = m
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class _BaseSparsePCA(_Base):
    def _normalize(self, k, C):
        nrm = k.ew("sqrt", k.rowsum(k.ew("sq", C)))
        nrm = k.ew("select", nrm, nrm, k.const(1.0), s=0.0)
        return k.ew("div", C, nrm)

    def transform(self, X):
        """sklearn `_BaseSparsePCA.transform`: ridge_regression(components_.T,
        (X - mean_).T, ridge_alpha, solver='cholesky')."""
        self._check()
        k = self._kit()
        Xc = k.ew("sub", _M.from_input(X), self.mean_m_)
        C = self.components_m_
        G = k.mm(C, C, tb=True)
        G = G.copy()
        for i in range(G.r):
            G.s[i * G.c + i] = _f32(G.s[i * G.c + i] + self.ridge_alpha)
        lu, piv, _ = k.lu(G)
        U = k.lu_solve(lu, piv, k.mm(C, Xc, tb=True)).T
        return U.out()

    def inverse_transform(self, X):
        self._check()
        k = self._kit()
        return k.ew("add", k.mm(_M.from_input(X), self.components_m_), self.mean_m_).out()


class SparsePCA(_BaseSparsePCA):
    """sklearn.decomposition.SparsePCA (reference: scikit-learn
    `decomposition/_sparse_pca.py`): dictionary learning on the centered X^T,
    the code's transpose the components, each normalized to unit norm."""
    _parameters = ("n_components", "alpha", "ridge_alpha", "max_iter", "tol", "method", "random_state",
                   "numeric_mode")

    def __init__(self, n_components=None, *, alpha=1, ridge_alpha=0.01, max_iter=1000, tol=1e-8, method="lars",
                 n_jobs=None, U_init=None, V_init=None, verbose=False, random_state=None, numeric_mode=None):
        self.n_components, self.alpha, self.ridge_alpha = n_components, alpha, ridge_alpha
        self.max_iter, self.tol, self.method, self.n_jobs = max_iter, tol, method, n_jobs
        self.U_init, self.V_init, self.verbose, self.random_state = U_init, V_init, verbose, random_state
        self.numeric_mode = numeric_mode

    def fit(self, X, y=None):
        self.numeric_mode_ = _mode(self.numeric_mode)
        if self.method not in ("lars", "cd"):
            raise ValueError("method must be 'lars' or 'cd'")
        k = self._kit()
        M = _M.from_input(X)
        nc = self.n_components if self.n_components is not None else M.c
        self.mean_m_ = k.colmean(M)
        Xc = k.ew("sub", M, self.mean_m_)
        ci = None if self.V_init is None else _M.from_input(self.V_init, "V_init").T
        di = None if self.U_init is None else _M.from_input(self.U_init, "U_init").T
        code, D, errors, it = _dict_learning(k, Xc.T, nc, float(self.alpha), self.max_iter, self.tol,
                                             "lasso_" + self.method, _seed_of(self.random_state), ci, di)
        self.components_m_ = self._normalize(k, code.T)
        self.components_ = self.components_m_.out()
        self.n_components_ = self.components_m_.r
        self.error_ = errors
        self.n_iter_ = it
        self.mean_ = self.mean_m_.out((M.c,))
        self.n_features_in_ = M.c
        return self


class MiniBatchSparsePCA(_BaseSparsePCA):
    """sklearn.decomposition.MiniBatchSparsePCA: MiniBatchDictionaryLearning
    on the centered X^T, the components its lasso_lars transform of X^T,
    each normalized to unit norm."""
    _parameters = ("n_components", "alpha", "ridge_alpha", "max_iter", "batch_size", "shuffle", "method",
                   "random_state", "tol", "max_no_improvement", "numeric_mode")

    def __init__(self, n_components=None, *, alpha=1, ridge_alpha=0.01, max_iter=1000, callback=None,
                 batch_size=3, verbose=False, shuffle=True, n_jobs=None, method="lars", random_state=None,
                 tol=1e-3, max_no_improvement=10, numeric_mode=None):
        self.n_components, self.alpha, self.ridge_alpha, self.max_iter = n_components, alpha, ridge_alpha, max_iter
        self.callback, self.batch_size, self.verbose, self.shuffle = callback, batch_size, verbose, shuffle
        self.n_jobs, self.method, self.random_state, self.tol = n_jobs, method, random_state, tol
        self.max_no_improvement, self.numeric_mode = max_no_improvement, numeric_mode

    def fit(self, X, y=None):
        self.numeric_mode_ = _mode(self.numeric_mode)
        k = self._kit()
        M = _M.from_input(X)
        nc = self.n_components if self.n_components is not None else M.c
        self.mean_m_ = k.colmean(M)
        Xc = k.ew("sub", M, self.mean_m_)
        est = MiniBatchDictionaryLearning(
            n_components=nc, alpha=self.alpha, max_iter=self.max_iter, batch_size=self.batch_size,
            shuffle=self.shuffle, fit_algorithm=self.method, random_state=self.random_state,
            transform_algorithm="lasso_lars", transform_alpha=self.alpha, tol=self.tol,
            max_no_improvement=self.max_no_improvement, numeric_mode=self.numeric_mode_)
        XT = Xc.T
        est.fit(XT.out())
        self.components_m_ = self._normalize(k, est._encode(XT.out()).T)
        self.components_ = self.components_m_.out()
        self.n_components_ = self.components_m_.r
        self.n_iter_ = est.n_iter_
        self.mean_ = self.mean_m_.out((M.c,))
        self.n_features_in_ = M.c
        return self


# ================================================================ LatentDirichletAllocation
_F64_EPS = 2.220446049250313e-16


def _dirichlet_expectation_2d(k, A):
    """psi(A) - psi(rowsum(A)) (sklearn `_online_lda_fast.pyx`)."""
    return k.ew("sub", k.ew("digamma", A), k.ew("digamma", k.rowsum(A)))


class LatentDirichletAllocation(_Base):
    """sklearn.decomposition.LatentDirichletAllocation (reference:
    scikit-learn `decomposition/_lda.py` with `_online_lda_fast.pyx`:
    `_init_latent_vars`, `_e_step`, `_update_doc_distribution`, `_em_step`,
    `_approx_bound`, `transform`, `perplexity`, `score`), variational EM in a
    fixed order: every document's inner loop is one thread, the sufficient
    statistics one gemm. The Gamma(100, 1/100) draws are Marsaglia-Tsang on
    the Philox stream. learning_method 'batch' and 'online' (online: the
    mini-batches in row order, as sklearn with no shuffle does).
    A scipy.sparse count matrix is densified (exact); the per-document cell
    already skips zero counts. evaluate_every with n_jobs: no host pool, so
    n_jobs changes nothing."""
    _parameters = ("n_components", "doc_topic_prior", "topic_word_prior", "learning_method", "learning_decay",
                   "learning_offset", "max_iter", "batch_size", "evaluate_every", "total_samples", "perp_tol",
                   "mean_change_tol", "max_doc_update_iter", "random_state", "numeric_mode")

    def __init__(self, n_components=10, *, doc_topic_prior=None, topic_word_prior=None, learning_method="batch",
                 learning_decay=0.7, learning_offset=10.0, max_iter=10, batch_size=128, evaluate_every=-1,
                 total_samples=1e6, perp_tol=1e-1, mean_change_tol=1e-3, max_doc_update_iter=100, n_jobs=None,
                 verbose=0, random_state=None, numeric_mode=None):
        self.n_components, self.doc_topic_prior, self.topic_word_prior = n_components, doc_topic_prior, topic_word_prior
        self.learning_method, self.learning_decay, self.learning_offset = learning_method, learning_decay, learning_offset
        self.max_iter, self.batch_size, self.evaluate_every = max_iter, batch_size, evaluate_every
        self.total_samples, self.perp_tol, self.mean_change_tol = total_samples, perp_tol, mean_change_tol
        self.max_doc_update_iter, self.n_jobs, self.verbose = max_doc_update_iter, n_jobs, verbose
        self.random_state, self.numeric_mode = random_state, numeric_mode

    def _check_X(self, X, whom):
        M = _M.from_input(X)
        if _any_negative(M):
            raise ValueError(f"Negative values in data passed to {whom}")
        return M

    def _init(self, k, d):
        nc = self.n_components
        self.doc_topic_prior_ = 1.0 / nc if self.doc_topic_prior is None else self.doc_topic_prior
        self.topic_word_prior_ = 1.0 / nc if self.topic_word_prior is None else self.topic_word_prior
        self._seed = _seed_of(self.random_state)
        self._draw = 0
        self.n_batch_iter_ = 1
        self.n_iter_ = 0
        self.components_m_ = k.ew("scale", k.rand_gamma(nc, d, self._seed, 60, 100.0), s=0.01)
        self._exp_dir = k.ew("exp", _dirichlet_expectation_2d(k, self.components_m_))

    def _e_step(self, k, X, cal_sstats, random_init):
        n, nc = X.r, self.components_m_.r
        if random_init:
            self._draw += 1
            Dt = k.ew("scale", k.rand_gamma(n, nc, self._seed, 61 + self._draw, 100.0), s=0.01)
        else:
            Dt = k.const(1.0, n, nc)
        Et = k.ew("exp", _dirichlet_expectation_2d(k, Dt))
        ss = None
        if cal_sstats:
            ss = self._fused_estep_ss(k, X, Dt, Et)     # FAST + Apple + define only, else None
        if ss is None:
            Dt, Et = k.lda_rows(X, self._exp_dir, Dt, Et, self.doc_topic_prior_, self.max_doc_update_iter,
                                self.mean_change_tol)
        if cal_sstats and ss is None:
            norm_phi = k.ew("adds", k.mm(Et, self._exp_dir), s=_F64_EPS)
            R = k.ew("div", X, norm_phi)
            ss = k.ew("mul", k.mm(Et, R, ta=True), self._exp_dir)
        return Dt, ss

    def _fused_estep_ss(self, k, X, Dt, Et):
        """FAST only (lane apple-fast-nb): the E-step's document loop and the
        sufficient statistics in one launch, `x_decomp_dev_lda_estep_ss`
        (x_decomp/lda_fast.mojo), exported by the GPU binding only when built
        FAST on Apple (default; off with -D MOJOLEARN_LDA_FUSED_SS_OFF). Dt and Et are updated in
        place on the device as `lda_rows` does. None (the caller runs main's
        chain) under IDENTICAL, without the export, off the resident path, or
        past the kernel's caps."""
        if k.mode != "fast" or not X.r or not k._use(X, self._exp_dir, Dt, Et):
            return None
        try:
            fn = getattr(k._raw(), "x_decomp_dev_lda_estep_ss")
        except Exception:       # the host proxy raises ImportError for an absent export
            return None
        nc, v = self._exp_dir.r, self._exp_dir.c
        ss = k._dout(nc, v)
        ok = fn(k._did(X), k._did(self._exp_dir), k._did(Dt), k._did(Et), ss._d.id,
                [X.r, nc, v, int(self.max_doc_update_iter)],
                [float(self.doc_topic_prior_), float(self.mean_change_tol)])
        return ss if int(ok) else None

    def _em_step(self, k, X, total_samples, batch_update):
        _, ss = self._e_step(k, X, True, True)
        if batch_update:
            self.components_m_ = k.ew("adds", ss, s=self.topic_word_prior_)
        else:
            # (offset + n_batch_iter)^(-decay) through the cells' log and exp
            # (Python's ** is the platform libm's pow)
            weight = k.ew("exp", k.ew("scale", k.ew("logs", k.const(self.learning_offset + self.n_batch_iter_),
                                                    s=1e-30), s=-self.learning_decay)).s[0]
            doc_ratio = float(total_samples) / X.r
            upd = k.ew("adds", k.ew("scale", ss, s=doc_ratio), s=self.topic_word_prior_)
            self.components_m_ = k.ew("add", k.ew("scale", self.components_m_, s=1 - weight),
                                      k.ew("scale", upd, s=weight))
        self._exp_dir = k.ew("exp", _dirichlet_expectation_2d(k, self.components_m_))
        self.n_batch_iter_ += 1

    def _online_pass(self, k, M, total_samples):
        """`for a in range(0, n, batch_size): self._em_step(k, M.rows(a, b),
        total_samples, False)` in ONE binding call (x_decomp/lda_online.mojo,
        lane/py-decomp-nbrs): the same cells, draws and float32 scalars per
        mini-batch (cgr-decomp: the MOJOLEARN_XD_LDA_PYTHON loop arm is
        deleted)."""
        bs = self.batch_size
        if isinstance(bs, bool) or not isinstance(bs, int) or bs < 1:
            raise ValueError("batch_size must be a positive integer")
        C, E = self.components_m_, self._exp_dir
        nc, v = C.r, C.c
        self._draw, self.n_batch_iter_ = k.b.x_decomp_lda_online(
            M.addr, C.addr, E.addr,
            [M.r, v, nc, bs, int(self.max_doc_update_iter), int(self._seed) & 0xFFFFFFFF, self._draw,
             self.n_batch_iter_],
            [float(self.doc_topic_prior_), float(self.topic_word_prior_), float(self.learning_offset),
             float(self.learning_decay), float(self.mean_change_tol), float(total_samples)])

    def fit(self, X, y=None):
        self.numeric_mode_ = _mode(self.numeric_mode)
        if self.learning_method not in ("batch", "online"):
            raise ValueError("learning_method must be 'batch' or 'online'")
        k = self._kit()
        M = self._check_X(X, "LatentDirichletAllocation.fit")
        n, d = M.r, M.c
        self._init(k, d)
        last_bound = None
        it = 0
        for it in range(1, self.max_iter + 1):
            if self.learning_method == "online":
                self._online_pass(k, M, n)
            else:
                self._em_step(k, M, n, True)
            if self.evaluate_every > 0 and it % self.evaluate_every == 0:
                Dt, _ = self._e_step(k, M, False, False)
                bound = self._perplexity(k, M, Dt)
                if last_bound is not None and abs(last_bound - bound) < self.perp_tol:
                    break
                last_bound = bound
        self.n_iter_ = it
        Dt, _ = self._e_step(k, M, False, False)
        self.bound_ = self._perplexity(k, M, Dt)
        self.components_ = self.components_m_.out()
        self.exp_dirichlet_component_ = self._exp_dir.out()
        self.n_features_in_ = d
        return self

    def partial_fit(self, X, y=None):
        if not hasattr(self, "components_m_"):
            self.numeric_mode_ = _mode(self.numeric_mode)
        k = self._kit()
        M = self._check_X(X, "LatentDirichletAllocation.partial_fit")
        if not hasattr(self, "components_m_"):
            self._init(k, M.c)
        self._online_pass(k, M, self.total_samples)
        self.components_ = self.components_m_.out()
        self.exp_dirichlet_component_ = self._exp_dir.out()
        self.n_features_in_ = M.c
        return self

    def _unnormalized(self, X):
        self._check()
        k = self._kit()
        M = self._check_X(X, "LatentDirichletAllocation.transform")
        return k, M, self._e_step(k, M, False, False)[0]

    def transform(self, X, *, normalize=True):
        k, M, Dt = self._unnormalized(X)
        if normalize:
            Dt = k.ew("div", Dt, k.rowsum(Dt))
        return Dt.out()

    def fit_transform(self, X, y=None, *, normalize=True):
        return self.fit(X).transform(X, normalize=normalize)

    def _loglik(self, k, prior, distr, dirichlet, size):
        s1 = k.total(k.ew("mul", k.ew("scale", k.ew("adds", distr, s=-prior), s=-1.0), dirichlet)).s[0]
        s2 = k.total(k.ew("adds", k.ew("lgamma", distr), s=-k.ew("lgamma", k.const(prior)).s[0])).s[0]
        lg_ps = k.ew("lgamma", k.const(prior * size)).s[0]
        s3 = k.total(k.ew("scale", k.ew("adds", k.ew("lgamma", k.rowsum(distr)), s=-lg_ps), s=-1.0)).s[0]
        return s1 + s2 + s3

    def _approx_bound(self, k, M, Dt, sub_sampling=False):
        n, v = M.r, M.c
        nc = self.components_m_.r
        ddt = _dirichlet_expectation_2d(k, Dt)             # n x k
        dcomp = _dirichlet_expectation_2d(k, self.components_m_)   # k x v
        floor = 1.1754943508222875e-38
        if n * v and k._use(M, ddt, dcomp) and hasattr(k._raw(), "x_decomp_dev_lda_bound"):
            # lane gap-lda-als: the same ew cells per (i, w) in one kernel,
            # no n x v term matrix per topic (the L40S text-block OOM)
            P = k._dout(n, v)
            k.b.x_decomp_dev_lda_bound(k._did(M), k._did(ddt), k._did(dcomp), P._d.id, [n, nc, v], [floor])
            score = k.total(P).s[0]
        else:
            zero = _M.zeros(n, v)
            terms = [k.ew("add", k.ew("add", zero, ddt.cols(t, t + 1)), dcomp.rows(t, t + 1)) for t in range(nc)]
            mx = terms[0]
            for t in range(1, nc):
                mx = k.ew("max", mx, terms[t])
            acc = _M.zeros(n, v)
            for t in range(nc):
                acc = k.ew("add", acc, k.ew("exp", k.ew("sub", terms[t], mx)))
            lse = k.ew("add", k.ew("logs", acc, s=floor), mx)
            score = k.total(k.ew("mul", M, lse)).s[0]
        score += self._loglik(k, self.doc_topic_prior_, Dt, ddt, nc)
        if sub_sampling:
            score *= float(self.total_samples) / n
        score += self._loglik(k, self.topic_word_prior_, self.components_m_, dcomp, v)
        return score

    def _perplexity(self, k, M, Dt, sub_sampling=False):
        bound = self._approx_bound(k, M, Dt, sub_sampling)
        word_cnt = k.total(M).s[0]
        if sub_sampling:
            word_cnt *= float(self.total_samples) / M.r
        if not word_cnt:
            return math.inf
        return self._kit().ew("exp", k.const(-bound / word_cnt)).s[0]

    def score(self, X, y=None):
        k, M, Dt = self._unnormalized(X)
        return self._approx_bound(k, M, Dt)

    def perplexity(self, X, sub_sampling=False):
        k, M, Dt = self._unnormalized(X)
        return self._perplexity(k, M, Dt, sub_sampling)


# ================================================================ manifold: Isomap, MDS, LLE
_PD_KIND = {"manhattan": 1, "cityblock": 1, "l1": 1, "chebyshev": 2, "infinity": 2, "cosine": 4}


def _metric_spec(metric, p=2, metric_params=None, who="metric"):
    """sklearn's metric name (and minkowski p) as (kind, p) for `_dist`:
    kind 0 is Euclidean (sqrt of the squared distance cell), the rest
    x_decomp/cells.mojo `pdist_cell` (DEVIATION 5319). A callable or any
    other name is refused by name."""
    if metric_params:
        extra = set(metric_params) - {"p"}
        if extra:
            raise NotImplementedError(f"{who}: metric_params {sorted(extra)} are not carried")
        p = metric_params.get("p", p)
    if metric in ("euclidean", "l2"):
        return 0, 2.0
    if metric == "minkowski":
        p = float(p)
        if not p > 0:
            raise ValueError(f"{who}: minkowski p must be > 0")
        if p == 2.0:
            return 0, 2.0
        if p == 1.0:
            return 1, 1.0
        if p == math.inf:
            return 2, 0.0
        return 3, p
    if isinstance(metric, str) and metric in _PD_KIND:
        return _PD_KIND[metric], 0.0
    raise NotImplementedError(f"{who}: metric={metric!r} is not carried (euclidean, minkowski p, manhattan, "
                              "chebyshev, cosine)")


def _dist(k, A, B, kind, pw, same=False):
    """The distance matrix of `_metric_spec`'s (kind, p). `same` (A is B):
    sklearn's pairwise_distances zeroes the cosine diagonal."""
    if kind == 0:
        return k.ew("sqrt", k.sqdist(A, B))
    D = k.pdist(A, B, kind, pw)
    if same and kind == 4:
        for i in range(A.r):
            D.s[i * B.r + i] = 0.0
    return D


def _knn_mats(k, Q, X, n_neighbors, exclude_self, kind=0, pw=2.0):
    """`_knn_lists` as two matrices (indices as exact floats, distances),
    n x n_neighbors, selected by the `graph_knn` cell (resident on the GPU
    binding)."""
    D = k.sqdist(Q, X) if kind == 0 else _dist(k, Q, X, kind, pw, same=exclude_self)
    return k.graph_knn(D, n_neighbors, exclude_self)


def _knn_lists(k, Q, X, n_neighbors, exclude_self, kind=0, pw=2.0):
    """(indices, distances) of the n_neighbors nearest rows of X for every
    row of Q, ascending, ties to the lower index; `exclude_self` drops the
    query's own index (queries ARE the training rows). kind 0 returns
    SQUARED Euclidean distances (the callers take the root); any other kind
    the `_dist` distances themselves."""
    im, dm = _knn_mats(k, Q, X, n_neighbors, exclude_self, kind, pw)
    nn = n_neighbors
    iv, dv = im.s, dm.s
    return ([[int(iv[i * nn + a]) for a in range(nn)] for i in range(Q.r)],
            [list(dv[i * nn:(i + 1) * nn]) for i in range(Q.r)])


def _center_kernel(k, K):
    """sklearn KernelCenterer.fit_transform: K - row means - column means + total mean.
    Returns (Kc, column means (1 x n), total mean (1 x 1))."""
    n = K.r
    col = k.ew("scale", k.colsum(K), s=1.0 / n)
    allm = k.ew("scale", k.total(col), s=1.0 / n)
    row = k.ew("scale", k.rowsum(K), s=1.0 / K.c)
    Kc = k.ew("add", k.ew("sub", k.ew("sub", K, col), row), allm)
    return Kc, col, allm


#: FAST's Lanczos route for the top eigenpairs (lane/decomp-apple2): taken
#: under sklearn's own ARPACK policy for eigen_solver='auto' (KernelPCA /
#: Isomap: n > 200 and fewer than 10 components), FAST mode only. ON by
#: default ON APPLE since lane/decomp-apple3 (opt-in elsewhere,
#: MOJOLEARN_XD_LANCZOS=1, where it has no quality check): its paired check
#: (bench/decomp_fast_quality.py) ran on a GPU with a FAST binding (m4-a
#: 1790626766529, 600 rows, 2 datasets x 2 seeds, Isomap and ClassicalMDS:
#: 8/8 PASS, every eigenvalue and eigenvector error UNDER the exact dense
#: solve's against the float64 reference, fits 0.03 to 0.17 s against 1.8 to
#: 4.1 s). MOJOLEARN_XD_LANCZOS=0 keeps the exact dense solve.
_LANCZOS_MIN_N = 200
_LANCZOS_MAX_NC = 10
#: Ritz residual bound, relative to the largest |Ritz value|, and the basis cap
#: past which the route gives up and runs the exact dense solve instead.
_LANCZOS_TOL = 1e-7
_LANCZOS_MAX_M = 600


def _kdot(k, a, b):
    return float(k.mm(a, b, ta=True).s[0])


def _lanczos_top(k, A, nc):
    """The nc largest eigenpairs of symmetric A by Lanczos with full
    reorthogonalization (classical Gram-Schmidt, twice), every product on
    the kit: A q, the basis projections and the updates. The basis starts at
    max(2 nc + 1, 20) vectors (ARPACK's ncv) and doubles until every wanted
    Ritz pair's residual estimate beta_m |y_m| is at most _LANCZOS_TOL times
    the largest |Ritz value|. Returns None when that has not happened by
    _LANCZOS_MAX_M vectors (the caller then runs the exact solve), so the
    route never returns a less converged answer than it promises. The start
    vector is a seeded uniform draw centred at 0 (a constant vector is
    orthogonal to a centred kernel's spectrum)."""
    n = A.r
    q = k.ew("adds", k.rand(n, 1, 0x1A2C05, 91, 0), s=-0.5)
    q = k.ew("scale", q, s=1.0 / math.sqrt(_kdot(k, q, q)))
    QT = array.array("f")
    alphas, betas = [], []
    m = min(n, max(2 * nc + 1, 20))
    j = 0
    stop = False
    while True:
        while j < m and not stop:
            QT.extend(q.s)
            w = k.mm(A, q)
            Qj = _M(QT[:], j + 1, n)
            c = k.mm(Qj, w)
            a = float(c.s[j])
            w = k.ew("sub", w, k.mm(Qj, c, ta=True))
            c = k.mm(Qj, w)
            a += float(c.s[j])
            w = k.ew("sub", w, k.mm(Qj, c, ta=True))
            alphas.append(a)
            b = math.sqrt(max(_kdot(k, w, w), 0.0))
            betas.append(b)
            j += 1
            if b <= 1e-30 * max(1.0, abs(a)) or j == n:
                stop = True
                break
            q = k.ew("scale", w, s=1.0 / b)
        T = [0.0] * (j * j)
        for i in range(j):
            T[i * j + i] = alphas[i]
            if i + 1 < j:
                T[i * j + i + 1] = T[(i + 1) * j + i] = betas[i]
        th, Y = k.eigh(_M.of(T, j, j))
        top = list(range(j - 1, max(j - 1 - nc, -1), -1))
        if len(top) < nc:
            return None
        big = max(abs(th.s[i]) for i in top) or 1.0
        res = [abs(betas[j - 1] * Y.s[(j - 1) * j + i]) for i in top]
        if stop or max(res) <= _LANCZOS_TOL * big:
            break
        if m >= min(n, _LANCZOS_MAX_M):
            return None
        m = min(n, 2 * m, _LANCZOS_MAX_M)
    Yt = Y.take_cols(top)
    V = k.mm(_M(QT[:j * n], j, n), Yt, ta=True)
    return th.take_cols(top), V


def _top_eig(k, A, nc, topk=False):
    """The nc LARGEST eigenpairs of symmetric A (descending), vectors in
    columns, each column signed by sklearn's svd_flip(u_based_decision=True).
    topk: the Lanczos route under sklearn's ARPACK policy (_lanczos_top), the
    exact dense solve otherwise.

    lane/neural-pass105 (Andrew, 2026-10-01: Isomap / ClassicalMDS /
    KernelPCA-style top-k uses take a deterministic top-k solver): the route
    is on in every mode and on every vendor. Every step is a kit primitive
    (the IDENTICAL GEMM, element-wise ops, the small eigh) or Python float64
    scalar arithmetic, in a fixed order with a fixed seeded start, so the
    words agree across vendors. MOJOLEARN_XD_LANCZOS=0 keeps the exact solve."""
    n = A.r
    got = None
    if topk and n > _LANCZOS_MIN_N and nc < _LANCZOS_MAX_NC and _os.environ.get(
            "MOJOLEARN_XD_LANCZOS", "1") == "1":
        got = _lanczos_top(k, A, nc)
    if got is not None:
        w, V = got
    else:
        w, V = k.eigh(A)
        order = list(range(n - 1, n - 1 - nc, -1))
        w, V = w.take_cols(order), V.take_cols(order)
    return w, V.neg_cols(k.absmax_flags(V, True))


class Isomap(_Base):
    """sklearn.manifold.Isomap (reference: scikit-learn `manifold/_isomap.py`
    with `KernelPCA(kernel='precomputed')` and `utils/graph.py`): the
    Euclidean kNN distance graph (ties to the lower index), all-pairs
    shortest paths by Dijkstra (one source per thread), the kernel
    -0.5 * D^2 centered, its top eigenpairs (Jacobi eigh). `radius` (with
    n_neighbors=None) takes every other row within the radius (closed, the
    float32 distance compared exactly). path_method 'FW' runs the same
    Dijkstra (x_decomp/NOT_IMPLEMENTED.tsv: DELIBERATELY DIVERGENT). metric:
    euclidean, minkowski p (p = 1, 2 and inf by name), manhattan, chebyshev,
    cosine (DEVIATION 5319). REFUSED BY NAME: a callable or any other
    metric."""
    _parameters = ("n_neighbors", "radius", "n_components", "eigen_solver", "path_method", "metric", "p", "numeric_mode")

    def __init__(self, *, n_neighbors=5, radius=None, n_components=2, eigen_solver="auto", tol=0, max_iter=None,
                 path_method="auto", neighbors_algorithm="auto", n_jobs=None, metric="minkowski", p=2,
                 metric_params=None, numeric_mode=None):
        self.n_neighbors, self.radius, self.n_components = n_neighbors, radius, n_components
        self.eigen_solver, self.tol, self.max_iter, self.path_method = eigen_solver, tol, max_iter, path_method
        self.neighbors_algorithm, self.n_jobs, self.metric, self.p = neighbors_algorithm, n_jobs, metric, p
        self.metric_params, self.numeric_mode = metric_params, numeric_mode

    def fit(self, X, y=None):
        self.numeric_mode_ = _mode(self.numeric_mode)
        if self.n_neighbors is not None and self.radius is not None:
            raise ValueError(f"Both n_neighbors and radius are provided. Use Isomap(radius={self.radius}, "
                             "n_neighbors=None) if intended to use radius-based neighbors")
        if self.n_neighbors is None and self.radius is None:
            raise ValueError("Isomap: one of n_neighbors and radius must be set")
        if self.radius is not None and not float(self.radius) >= 0:
            raise ValueError("Isomap: radius must be >= 0")
        if self.path_method not in ("auto", "FW", "D"):
            raise ValueError("path_method must be 'auto', 'FW' or 'D'")
        _exact_eigen_solver(self.eigen_solver, "Isomap")
        kind, pw = _metric_spec(self.metric, self.p, self.metric_params, "Isomap")
        self._kind, self._pw = kind, pw
        k = self._kit()
        M = _M.from_input(X)
        n = M.r
        Wg, nn = self._graph(k, M, n, kind, pw)
        D = k.dijkstra(Wg)
        self.dist_matrix_m_ = D
        self.dist_matrix_ = D.out()
        G = k.ew("scale", k.ew("sq", D), s=-0.5)
        Kc, self._k_col, self._k_all = _center_kernel(k, G)
        w, V = _top_eig(k, Kc, int(self.n_components), topk=self.eigen_solver in ("auto", "arpack"))
        self.eigenvalues_m_ = w
        self.eigenvectors_m_ = V
        self.embedding_m_ = k.ew("mul", V, k.ew("sqrt", w))
        self.embedding_ = self.embedding_m_.out()
        self._fit_X, self._knn = M, nn
        self.n_features_in_ = M.c
        self._Kc = Kc
        return self

    def _graph(self, k, M, n, kind, pw):
        """The neighbor graph, its components joined (sklearn
        `_fix_connected_components`), all as graph cells (lane
        hr2-graph-embed): resident on the GPU binding."""
        if self.radius is not None:
            D = _dist(k, M, M, kind, pw, same=True)
            Wg = k.graph_radius(D, _f32(float(self.radius)))
            nn = None
        else:
            nn = int(self.n_neighbors)
            idx, sq = _knn_mats(k, M, M, nn, True, kind, pw)
            if kind == 0:
                sq = k.ew("sqrt", sq)
            Wg = k.graph_knn_dense(idx, sq, n)
        comp, C = k.graph_components(Wg)
        self.n_connected_components_ = C
        if C > 1:
            import warnings
            warnings.warn(f"The number of connected components of the neighbors graph is {C} > 1. "
                          "Completing the graph to fit Isomap might be slow.", stacklevel=3)
            D = _dist(k, M, M, kind, pw, same=True)
            k.graph_join(Wg, D, comp, C)
        return Wg, nn

    def fit_transform(self, X, y=None):
        return self.fit(X).embedding_

    def transform(self, X):
        """sklearn Isomap.transform: geodesic distance of each query through
        its k nearest training points, then KernelPCA.transform."""
        self._check("embedding_m_")
        k = self._kit()
        Q = _M.from_input(X)
        n = self._fit_X.r
        D = self.dist_matrix_m_
        if self._knn is None:
            G = self._radius_geodesic(k, Q)
        else:
            idx, dst = _knn_lists(k, Q, self._fit_X, self._knn, False, self._kind, self._pw)
            sq = _M.of([v for row in dst for v in row], Q.r, self._knn)
            if self._kind == 0:
                sq = k.ew("sqrt", sq)
            G = None
            for a in range(self._knn):
                rows = D.take_rows([idx[i][a] for i in range(Q.r)])
                cand = k.ew("add", rows, sq.cols(a, a + 1))
                G = cand if G is None else k.ew("min", G, cand)
        G = k.ew("scale", k.ew("sq", G), s=-0.5)
        row = k.ew("scale", k.rowsum(G), s=1.0 / n)
        Kc = k.ew("add", k.ew("sub", k.ew("sub", G, self._k_col), row), self._k_all)
        V = k.ew("div", self.eigenvectors_m_, k.ew("sqrt", self.eigenvalues_m_))
        return k.mm(Kc, V).out()

    def _radius_geodesic(self, k, Q):
        """sklearn's radius transform: for each query, the minimum over the
        training rows within the radius (distance 0 included) of
        dist_matrix_[j] + d(q, j): one cell per (query, column) on the
        device, the candidates in ascending j (`graph_radius_geo`)."""
        r = _f32(float(self.radius))
        Dq = _dist(k, Q, self._fit_X, self._kind, self._pw)
        D = self.dist_matrix_m_
        # every query has a training row within the radius: the per-row
        # counts and their total on the device, one scalar read
        hits = k.ew("gts", k.rowsum(k.ew("le", Dq, _M.of([r], 1, 1))), s=0.5)
        if Q.r and k.total(hits).s[0] != Q.r:
            i = list(hits.s).index(0.0)
            raise ValueError(f"Isomap.transform: query row {i} has no training row within radius {self.radius}")
        return k.graph_radius_geo(Dq, D, r)

    def reconstruction_error(self):
        self._check("embedding_m_")
        k = self._kit()
        # a difference of two nearly equal sums: accumulated in float64
        # (sequential IEEE adds of exact float32 squares) or it cancels
        t = _dsum(v * v for v in self._Kc.s) - _dsum(v * v for v in self.eigenvalues_m_.s)
        return math.sqrt(t) / self._Kc.r if t > 0 else 0.0


class ClassicalMDS(_Base):
    """sklearn.manifold.ClassicalMDS (reference: scikit-learn
    `manifold/_classical_mds.py`): double-centred -0.5 * D^2, its top
    eigenpairs, embedding = V sqrt(max(w, 0))."""
    _parameters = ("n_components", "metric", "numeric_mode")

    def __init__(self, n_components=2, *, metric="euclidean", metric_params=None, numeric_mode=None):
        self.n_components, self.metric, self.metric_params, self.numeric_mode = n_components, metric, metric_params, numeric_mode

    def fit(self, X, y=None):
        self.numeric_mode_ = _mode(self.numeric_mode)
        k = self._kit()
        M = _M.from_input(X)
        if self.metric == "precomputed":
            D2 = k.ew("sq", M)
            self.dissimilarity_matrix_ = M.out()
        elif self.metric == "euclidean":
            D2 = k.sqdist(M, M)
            self.dissimilarity_matrix_ = k.ew("sqrt", D2).out()
        else:
            kind, pw = _metric_spec(self.metric, 2, self.metric_params, "ClassicalMDS")
            Dm = _dist(k, M, M, kind, pw, same=True)
            D2 = k.ew("sq", Dm)
            self.dissimilarity_matrix_ = Dm.out()
        B, _, _ = _center_kernel(k, k.ew("scale", D2, s=-0.5))
        w, V = _top_eig(k, B, int(self.n_components), topk=True)
        self.eigenvalues_ = w.out((w.c,))
        self.embedding_m_ = k.ew("mul", V, k.ew("sqrt", w))
        self.embedding_ = self.embedding_m_.out()
        self.n_features_in_ = M.c
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X).embedding_


class MDS(_Base):
    """sklearn.manifold.MDS (reference: scikit-learn `manifold/_mds.py`,
    `_smacof_single`, `smacof`): metric SMACOF with the Guttman transform,
    `n_init` starts (random ones from the Philox stream), the lowest stress
    kept. init 'random' (the default the FutureWarning names), 'classical_mds'
    or an array. metric_mds=False is Kruskal's non-metric SMACOF: the
    disparities are the isotonic regression of the distances on the
    dissimilarities (the linear lane's IsotonicRegression), rescaled."""
    _parameters = ("n_components", "metric_mds", "n_init", "init", "max_iter", "eps", "random_state", "metric",
                   "normalized_stress", "numeric_mode")

    def __init__(self, n_components=2, *, metric_mds=True, n_init=1, init="random", max_iter=300, verbose=0,
                 eps=1e-6, n_jobs=None, random_state=None, metric="euclidean", metric_params=None,
                 normalized_stress="auto", numeric_mode=None):
        self.n_components, self.metric_mds, self.n_init, self.init = n_components, metric_mds, n_init, init
        self.max_iter, self.verbose, self.eps, self.n_jobs = max_iter, verbose, eps, n_jobs
        self.random_state, self.metric, self.metric_params = random_state, metric, metric_params
        self.normalized_stress, self.numeric_mode = normalized_stress, numeric_mode

    def _dist(self, k, Y):
        return k.ew("sqrt", k.sqdist(Y, Y))

    def _nm_native(self, k, Dis, n):
        """The non-metric SMACOF bookkeeping in native calls (lane/py-decomp-nbrs):
        Python gathered n (n - 1) / 2 pairs, fed them to IsotonicRegression as
        lists (a lambda sort of every pair) and scattered and mirrored them back
        one element at a time, every iteration (about 1 s per iteration at
        n = 3000). The same positions, the same stable (x, y) order, the same
        x_linear isotonic fit and predict calls and the same scatter, as
        buffers. Returns the per-iteration disparity function."""
        from . import _expansion_linear as _xlin
        b = k.b
        cap = max(n * (n - 1) // 2, 1)
        pos, mir = array.array("i", [0]) * cap, array.array("i", [0]) * cap
        m = int(b.x_decomp_triu_nonzero(Dis.addr, n, pos.buffer_info()[0], mir.buffer_info()[0]))
        pa, ma = pos.buffer_info()[0], mir.buffer_info()[0]
        dis_w = array.array("f", [0.0]) * max(m, 1)
        b.x_decomp_gather(Dis.addr, pa, m, dis_w.buffer_info()[0])
        xorder = array.array("i", [0]) * max(m, 1)
        b.x_decomp_argsort_f32(dis_w.buffer_info()[0], m, xorder.buffer_info()[0])
        ir = _xlin.IsotonicRegression(out_of_bounds="clip", numeric_mode=self.numeric_mode_)
        lin = ir._bind(_xlin._BINDING)
        ones = array.array("f", [1.0]) * m

        def call(algo, X, Y, rows, ip, fp, n_out, n_fw, n_iw):
            # _expansion_linear._run's one x_linear_fit call, its parameter
            # list verbatim, the output kept as a buffer (no Python list)
            out = array.array("f", [0.0]) * max(n_out, 1)
            lin.x_linear_fit(int(algo), X.buffer_info()[0], Y.buffer_info()[0],
                             [rows, 1, rows, len(Y), n_out, max(n_fw, 1), max(n_iw, 1), len(ip), len(fp)],
                             [int(v) for v in ip], [float(v) for v in fp], out.buffer_info()[0])
            return out

        def disparities(d, first):
            # the closure holds pos and mir themselves: their addresses alone
            # would let the arrays die when _nm_native returns
            pa, ma = pos.buffer_info()[0], mir.buffer_info()[0]
            if first:
                flat = dis_w
            else:
                ds = array.array("f", [0.0]) * max(m, 1)
                b.x_decomp_gather(d.addr, pa, m, ds.buffer_info()[0])
                order = array.array("i", [0]) * max(m, 1)
                b.x_decomp_iso_order(dis_w.buffer_info()[0], ds.buffer_info()[0], xorder.buffer_info()[0], m,
                                     order.buffer_info()[0])
                ob = order.buffer_info()[0]
                xa = array.array("f", [0.0]) * max(m, 1)
                b.x_decomp_gather(dis_w.buffer_info()[0], ob, m, xa.buffer_info()[0])
                yy = array.array("f", [0.0]) * m
                b.x_decomp_gather(ds.buffer_info()[0], ob, m, yy.buffer_info()[0])
                yy.extend(ones)
                # IsotonicRegression(out_of_bounds='clip').fit(dis_w, ds): increasing, no bounds
                vals = call(_xlin.ALGO_ISOTONIC, xa, yy, m, [1, 0, 0], [0.0, 0.0], 3 + 2 * m, 3 * m, m)
                kk = int(vals[0])
                thr = vals[3:3 + kk] + vals[3 + m:3 + m + kk]
                # .transform(dis_w): clip, the thresholds and bounds above
                flat = call(_xlin.ALGO_ISOTONIC_PREDICT, dis_w, thr, m, [kk, 1],
                            [float(vals[1]), float(vals[2])], m, 1, 1)
            P = _M.zeros(n, n)
            b.x_decomp_scatter(P.addr, pa, m, flat.buffer_info()[0])
            ss = k.total(k.ew("sq", P)).s[0]
            P = k.ew("scale", P, s=math.sqrt((n * (n - 1) / 2) / ss))
            tmp = array.array("f", [0.0]) * max(m, 1)
            b.x_decomp_gather(P.addr, pa, m, tmp.buffer_info()[0])
            b.x_decomp_scatter(P.addr, ma, m, tmp.buffer_info()[0])
            return P

        return disparities

    def _single(self, k, Dis, Y, run):
        n = Dis.r
        # cgfin-c-decomp: non-metric SMACOF's bookkeeping is the native
        # calls only (`_nm_native`); the Python pair lists and the
        # MOJOLEARN_XD_MDS_PYTHON switch are deleted
        native = not self.metric_mds
        if native:
            if n * n > 2147483647:
                raise ValueError(f"MDS(metric_mds=False): {n} rows exceed the 32-bit pair index (n * n <= 2**31 - 1)")
            nm = self._nm_native(k, Dis, n)
        disp = Dis
        d = self._dist(k, Y)
        old = None
        it = 0
        # The Guttman transform's diagonal (lane hr2-mds-agglo): B's diagonal
        # gets the row sums by fma(I, rs, B) (1 * rs + B, one rounding;
        # 0 * rs + B is B off it), so the n x n matrix stays resident. The
        # host binding takes the same fma on a host identity.
        eye = k.diag_mask(n)
        if eye is None:
            eye = _eye(n)
        floor = k.const(1e-5)
        for it in range(1, self.max_iter + 1):
            if native:
                disp = nm(d, it == 1)
            dz = k.ew("select", d, d, floor, s=0.0)
            ratio = k.ew("div", disp, dz)
            B = k.ew("scale", ratio, s=-1.0)
            rs = k.rowsum(ratio)
            B = k.ew("fma", eye, rs, B)
            Y = k.ew("scale", k.mm(B, Y), s=1.0 / n)
            d = self._dist(k, Y)
            stress = k.total(k.ew("sqdiff", d, disp)).s[0] / 2
            if old is not None:
                ssd = k.total(k.ew("sq", d)).s[0]
                if (old - stress) / (ssd / 2) < self.eps:
                    break
            old = stress
        if self._norm:
            ssd = k.total(k.ew("sq", d)).s[0]
            stress = math.sqrt(stress / (ssd / 2)) if ssd else 0.0
        return Y, stress, it

    def fit_transform(self, X, y=None, init=None):
        self.numeric_mode_ = _mode(self.numeric_mode)
        k = self._kit()
        M = _M.from_input(X)
        if self.metric == "precomputed":
            Dis = M
        elif self.metric == "euclidean":
            Dis = self._dist(k, M)
        else:
            kind, pw = _metric_spec(self.metric, 2, self.metric_params, "MDS")
            Dis = _dist(k, M, M, kind, pw, same=True)
        self.dissimilarity_matrix_ = Dis.out()
        self._norm = (self.normalized_stress is True) or (self.normalized_stress == "auto" and not self.metric_mds)
        n, nc = Dis.r, int(self.n_components)
        seed = _seed_of(self.random_state)
        if init is not None:
            starts = [_M.from_input(init, "init")]
        elif self.init == "classical_mds":
            starts = [ClassicalMDS(nc, metric="precomputed", numeric_mode=self.numeric_mode_).fit(Dis.out()).embedding_m_]
        elif self.init == "random":
            starts = [k.rand(n, nc, seed, 70 + r, 0) for r in range(int(self.n_init))]
        else:
            raise ValueError("init must be 'random', 'classical_mds' or an array")
        best = None
        for r, Y0 in enumerate(starts):
            Y, stress, it = self._single(k, Dis, Y0, r)
            if best is None or stress < best[1]:
                best = (Y, stress, it)
        Y, self.stress_, self.n_iter_ = best
        self.embedding_ = Y.out()
        self.n_features_in_ = M.c
        return self.embedding_

    def fit(self, X, y=None, init=None):
        self.fit_transform(X, init=init)
        return self


#: LocallyLinearEmbedding's iterative route for eigen_solver='auto', taken
#: under sklearn's own ARPACK policy (n > 200 and n_components + 1 < 10).
#: The dense route is the one-sided Jacobi SVD of the whole n x n factor:
#: O(n^3) per sweep in ONE threadgroup on the GPU (M2 Pro: 3.2 s at 250
#: rows, 22 s at 500, 199 s at 1,000; past the board's 600 s ceiling at
#: 2,000 on the M3 Ultra). method='standard' only (its factor I - W is
#: square); 'ltsa', 'hessian' and 'modified' keep the dense route.
_LLE_ITER_MIN_N = 200
_LLE_ITER_MAX_K = 10
#: Converged: the sine of the largest principal angle between two successive
#: wanted Ritz subspaces is at most _LLE_SUBSPACE_TOL, or, under
#: _LLE_STALL_TOL, it stopped shrinking (float32's floor for this factor).
_LLE_SUBSPACE_TOL = 1e-5
_LLE_STALL_TOL = 1e-2
#: The deflation guard: F's rms row sum |F 1| / sqrt(n) (float32's rounding
#: of the weights' unit sums) must be at most this fraction of its rms row
#: norm, or the constant is not F's null vector and the dense route runs.
_LLE_NULL_GUARD = 1e-3
#: A wanted Ritz singular value at most this many float32 epsilons times the
#: rms row norm is numerically zero: the kNN graph has several components
#: (taxi's near-duplicate rows at 2,000), M's null space is wider than the
#: constant, and any basis of it is the answer (sklearn's ARPACK returns
#: its own); the iteration stops there from its third step.
_LLE_NULL_FLOOR = 8.0


def _lle_orth(k, Z):
    """Z's columns orthonormalized by the Householder QR (geqrf, orgqr):
    unconditionally orthonormal, never a zeroed column (the kit's orth zeroes
    a column it judges dependent, DEVIATION 5318, and subspace iteration's
    columns lean together). Each column is first scaled by its 1-norm (the
    shift-invert operator reaches 1 / sigma^2, 1e18 on a null space at
    float32 resolution, whose squares overflow)."""
    nr = [float(v) for v in k.colsum(k.ew("abs", Z)).s]
    sc = _M.of([1.0 / v if v > 0.0 and math.isfinite(v) else 1.0 for v in nr], 1, Z.c)
    h, tau = k.geqrf(k.ew("mul", Z, sc))
    return k.orgqr(h, tau, Z.c)


def _lle_smallest(k, F, nc, max_iter, seed=0):
    """The nc smallest right singular pairs of the square LLE factor F
    (n x n, M = F^T F) past its null vector, the constant: (V n x nc, unit
    columns, ascending singular value, each signed so its largest-|.| entry
    is positive; S 1 x nc). None when the route does not apply (the caller
    runs the dense SVD).

    The constant is deflated exactly: H, the Householder reflector taking
    u = 1 / sqrt(n) to e_{n-1}, and F^ = F H[:, :n-1], F restricted to the
    constant's complement (its last column F u is dropped; F u is float32
    rounding of the weights' unit sums, and the guard below checks it is
    far under its rms row norm). F0 = [F^ | u] is square and, when
    u is not orthogonal to z (the unit left null vector of F^), nonsingular:
    ONE LU (the device's parallel right-looking getrf; a pivot under float32
    resolution is set to eps times the largest); every solve against it is `trisolve` (right-looking substitution
    on the device, no inverse formed). z = F0^-T e_{n-1}, normalized. For x in R^{n-1},
    (F^T F^)^-1 x = the first n - 1 entries of F0^-1 (P_z F0^-T [x; 0]),
    P_z = I - z z^T: F0^-T [x; 0] solves F^^T y = x, P_z takes its
    minimum-norm part (in range(F^)), and F0^-1 solves F^ w = that exactly.
    Subspace iteration on that operator (sklearn's shift-invert at
    sigma = 0) with p = max(2 nc + 1, 20) columns (ARPACK's ncv), each step
    in two halves (F^+T, then F^+), each half orthonormalized (`_lle_orth`),
    then rotated to the Ritz vectors of F^
    (the one-sided Jacobi SVD of F^ X, n x p), until the wanted Ritz
    subspace settles (_LLE_SUBSPACE_TOL, or stalled under _LLE_STALL_TOL).
    Every product is a kit cell and the stopping test reads their outputs:
    the same iterations on every vendor. Not settled in max_iter steps
    raises (an unconverged answer is not returned as one). Wanted singular
    values under _LLE_NULL_FLOOR float32 epsilons (a null space wider than
    the constant) stop it from the third step on (each step has shrunk
    the rest by (sigma / sigma_next)^2): any basis of that space is the
    answer, to float32's resolution of |F^ x|."""
    n = F.c
    if F.r != n:
        return None
    n1 = n - 1
    p = min(n1, max(2 * nc + 1, 20))
    if p <= nc:
        return None
    rn = 1.0 / math.sqrt(n)
    hv = [rn] * n
    hv[n - 1] = rn - 1.0
    coef = 2.0 / _dsum(v * v for v in hv)
    h = _M.of(hv, n, 1)
    un = _M.of([rn] * n, n, 1)
    hrow = _M.of([rn] * n1, 1, n1)
    Fh = k.mm(F, h)
    Fhat = k.ew("sub", F.cols(0, n1), k.ew("scale", k.mm(Fh, hrow), s=coef))
    Fu = k.mm(F, un)
    g = math.sqrt(_dsum(float(v) * float(v) for v in Fu.s))
    rms = math.sqrt(max(float(k.total(k.ew("sq", F)).s[0]), 0.0) / n)
    if not (g <= _LLE_NULL_GUARD * rms):
        return None
    floor = _LLE_NULL_FLOOR * _F32_EPS * rms
    F0 = _hstack(Fhat, un)
    lu, piv, _ = k.lu(F0)
    # a pivot under float32 resolution (an exactly zero one skipped its
    # step) is set to eps times the largest: inverse iteration's usual
    # perturbation (LAPACK's stein/hsein); the factor is only the spectral
    # transform, the Rayleigh-Ritz step below uses F^ itself. The floor and
    # the swaps' row order (and its inverse) are cells (`lu_aux`), no host
    # loop over the rows.
    st, _, pm, im = k.lu_aux(lu, piv, clamp=True)
    big = st[0]
    if not (big > 0.0 and math.isfinite(big)):
        return None

    def solve(B):               # F0^-1 B = U^-1 L^-1 P B
        return k.trisolve(lu, pm, B)

    def solve_t(B):             # F0^-T B = P^T L^-T U^-T B
        return k.trisolve(lu, im, B, 1)

    en = _M.zeros(n, 1)
    en.s[n - 1] = 1.0
    z = solve_t(en)
    zn = math.sqrt(_dsum(float(v) * float(v) for v in z.s))
    if not zn > 0.0 or not math.isfinite(zn):
        return None
    z = k.ew("scale", z, s=1.0 / zn)

    X = _lle_orth(k, k.ew("adds", k.rand(n1, p, seed, 0x11E, 0), s=-0.5))
    want = list(range(p - 1, p - 1 - nc, -1))
    prev, e_prev = None, float("inf")
    for it in range(max(1, int(max_iter))):
        # (F^T F^)^-1 X in two orthonormalized halves: F^+T X = P_z F0^-T
        # [X; 0] (range(F^), n x p), then F^+ of that = the first n - 1 rows
        # of F0^-1; each half stretches the block by 1 / sigma, not
        # 1 / sigma^2, so the columns stay far from float32 dependence
        Y = solve_t(_M(array.array("f", X.s) + array.array("f", [0.0]) * X.c, n, X.c))
        Y = _lle_orth(k, k.ew("sub", Y, k.mm(z, k.mm(z, Y, ta=True))))
        X = _lle_orth(k, solve(Y).rows(0, n1))
        S, Vt = k.svd(k.mm(Fhat, X))
        X = k.mm(X, Vt, tb=True)
        Y = X.take_cols(want)
        if it >= 2 and max(float(v) for v in S.take_cols(want).s) <= floor:
            break
        if prev is not None:
            E = k.ew("sub", Y, k.mm(prev, k.mm(prev, Y, ta=True)))
            e = math.sqrt(max(float(k.total(k.ew("sq", E)).s[0]), 0.0))
            if e <= _LLE_SUBSPACE_TOL or (e <= _LLE_STALL_TOL and e >= e_prev):
                break
            e_prev = e
        prev = Y
    else:
        raise RuntimeError(
            f"LocallyLinearEmbedding: the shift-invert subspace iteration did not settle in {max_iter} "
            f"iterations (last subspace change {e_prev:.3g}); pass eigen_solver='dense' for the full SVD. "
            "An unconverged embedding is not returned as if it were one.")
    sv = S.take_cols(want)
    # back to R^n: H [Y; 0] = [Y; 0] - coef h (h^T [Y; 0])
    t = k.mm(hrow, Y)
    full = _M(array.array("f", Y.s) + array.array("f", [0.0]) * nc, n, nc)
    V = k.ew("sub", full, k.ew("scale", k.mm(h, t), s=coef))
    # the columns are unit vectors or the solve is not an answer (an
    # overflow, a dropped launch): refuse rather than return them
    sq = k.colsum(k.ew("sq", V)).s
    if not all(0.9 <= float(v) <= 1.1 for v in sq) or not all(math.isfinite(float(v)) for v in sv.s):
        raise RuntimeError(
            "LocallyLinearEmbedding: the shift-invert subspace iteration returned columns of squared norm "
            f"{[float(v) for v in sq]} (not unit); pass eigen_solver='dense' for the full SVD.")
    return V.neg_cols(k.absmax_flags(V, True)), sv


class LocallyLinearEmbedding(_Base):
    """sklearn.manifold.LocallyLinearEmbedding, method='standard' (reference:
    scikit-learn `manifold/_locally_linear.py`: `barycenter_kneighbors_graph`,
    `barycenter_weights`, `null_space` (here the smallest right singular
    vectors of I - W, which are M's smallest eigenvectors),
    `locally_linear_embedding`, `transform`). M = (I - W)^T (I - W), its
    n_components + 1 smallest eigenvectors, the first dropped.
    method='ltsa', 'hessian' and 'modified' build sklearn's M as a stacked
    factor (M = B^T B) and take the same SVD route. eigen_solver='auto'
    with method='standard' past 200 rows (sklearn's ARPACK policy) takes
    `_lle_smallest`, shift-invert subspace iteration on the deflated factor;
    'dense' (and every other case) the full one-sided Jacobi SVD."""
    _parameters = ("n_neighbors", "n_components", "reg", "eigen_solver", "method", "random_state", "numeric_mode")

    def __init__(self, *, n_neighbors=5, n_components=2, reg=1e-3, eigen_solver="auto", tol=1e-6, max_iter=100,
                 method="standard", hessian_tol=1e-4, modified_tol=1e-12, neighbors_algorithm="auto",
                 random_state=None, n_jobs=None, numeric_mode=None):
        self.n_neighbors, self.n_components, self.reg, self.eigen_solver = n_neighbors, n_components, reg, eigen_solver
        self.tol, self.max_iter, self.method, self.hessian_tol = tol, max_iter, method, hessian_tol
        self.modified_tol, self.neighbors_algorithm, self.random_state = modified_tol, neighbors_algorithm, random_state
        self.n_jobs, self.numeric_mode = n_jobs, numeric_mode

    def fit(self, X, y=None):
        self.numeric_mode_ = _mode(self.numeric_mode)
        if self.method not in ("standard", "hessian", "modified", "ltsa"):
            raise ValueError(f"LocallyLinearEmbedding: unrecognized method {self.method!r}")
        _exact_eigen_solver(self.eigen_solver, "LocallyLinearEmbedding")
        k = self._kit()
        M = _M.from_input(X)
        n = M.r
        nn = int(self.n_neighbors)
        nc = int(self.n_components)
        if nn >= n:
            raise ValueError("Expected n_neighbors < n_samples")
        if self.method == "hessian" and nn <= nc + nc * (nc + 1) // 2:
            raise ValueError("for method='hessian', n_neighbors must be greater than "
                             "[n_components * (n_components + 3) / 2]")
        if self.method == "modified" and nn < nc:
            raise ValueError("modified LLE requires n_neighbors >= n_components")
        # lane hr2-graph-embed: the kNN, the barycenter weights and I - W
        # as cells, I - W resident on the GPU binding; cgr-decomp: the LTSA,
        # Hessian and modified factors as cells too (x_decomp/lle_local.mojo,
        # the local eigensolves batched), no Python loop over the samples
        idm, _ = _knn_mats(k, M, M, nn, True)
        if self.method == "standard":
            IW = k.graph_lle_iw(idm, k.barycenter(M, M, idm, self.reg), n)
        elif self.method == "ltsa":
            IW = k.lle_local(M, idm, 0, nn, nc, 0.0)
        elif self.method == "hessian":
            IW = k.lle_local(M, idm, 1, nn, nc, float(self.hessian_tol))
        else:
            IW = k.lle_local(M, idm, 2, nn, nc, float(self.modified_tol))
        # The eigenvectors of M = (I - W)^T (I - W) for its smallest
        # eigenvalues are the right singular vectors of I - W for its
        # smallest singular values. Those eigenvalues sit near 1e-7, under
        # float32 resolution next to M's largest, so the dense eigh of M
        # cannot order them; the one-sided Jacobi SVD of I - W resolves its
        # small singular values to high RELATIVE accuracy.
        got = None
        if (self.method == "standard" and self.eigen_solver == "auto" and n > _LLE_ITER_MIN_N
                and nc + 1 < _LLE_ITER_MAX_K):
            got = _lle_smallest(k, IW, nc, int(self.max_iter), _seed_of(self.random_state))
        if got is not None:
            self.embedding_m_, sv = got
        else:
            # eigen_solver='dense', a small n, or the iterative route's guard
            # refused: the full SVD of I - W (O(n^3) per Jacobi sweep)
            S, Vt = k.svd(IW)
            rows = list(range(n - 2, n - 2 - nc, -1))
            self.embedding_m_ = Vt.take_rows(rows).T
            sv = S.take_cols(rows)
        self.embedding_ = self.embedding_m_.out()
        self.reconstruction_error_ = _dsum(v * v for v in sv.s)
        self._fit_X, self._knn = M, nn
        self.n_features_in_ = M.c
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X).embedding_

    def transform(self, X):
        self._check("embedding_m_")
        k = self._kit()
        Q = _M.from_input(X)
        idm, _ = _knn_mats(k, Q, self._fit_X, self._knn, False)
        Wb = k.barycenter(Q, self._fit_X, idm, self.reg)
        # X_new[i] = sum_a W[i, a] * embedding[idx[i][a]]: one cell per output
        # (x_decomp/lle_local.mojo `lle_apply_cell`), no loop over the queries
        return k.lle_apply(Wb, idm, self.embedding_m_).out()


# ================================================================ robust covariance
def _pinvh(k, A):
    """scipy.linalg.pinvh: V diag(1/w) V^T over |w| > max|w| * n * float32 eps."""
    w, V = k.eigh(A)
    wmax = max(abs(v) for v in w.s) if w.c else 0.0
    cut = _f32(wmax * A.r * _F32_EPS)
    keep = k.ew("recip", w)
    inv = _M.of([keep.s[j] if abs(w.s[j]) > cut else 0.0 for j in range(w.c)], 1, w.c)
    return k.mm(k.ew("mul", V, inv), V, tb=True)


def _slogdet(k, A):
    """(sign, log|det|) from the LU factorization (getrf; sums ascending)."""
    lu, piv, info = k.lu(A)
    st, diag, _, _ = k.lu_aux(lu, piv)
    if st[1] > 0:
        return 0.0, -math.inf
    neg = int(st[2]) + int(st[3])
    ld = k.total(k.ew("logs", k.ew("abs", diag), s=1.1754943508222875e-38)).s[0]
    return (-1.0 if neg % 2 else 1.0), ld


def _fast_logdet(k, A):
    sign, ld = _slogdet(k, A)
    return ld if sign > 0 else -math.inf


def _emp_cov(k, Xs, assume_centered=False):
    """sklearn empirical_covariance: (X - mean)^T (X - mean) / n."""
    if assume_centered:
        return k.ew("scale", k.mm(Xs, Xs, ta=True), s=1.0 / Xs.r)
    Xc = k.ew("sub", Xs, k.colmean(Xs))
    return k.ew("scale", k.mm(Xc, Xc, ta=True), s=1.0 / Xs.r)


def _f32_below(x):
    """The largest float32 strictly below the float x: v < x is v <= this for
    every float32 v (the kit's `le` cell takes the same decision)."""
    f = _f32(x)
    if f < x:
        return f
    b = array.array("f", [f])
    u = array.array("I", b.tobytes())
    if f > 0.0:
        u[0] -= 1
    elif f == 0.0:
        u[0] = 0x80000001          # the least negative subnormal
    else:
        u[0] += 1
    return array.array("f", u.tobytes())[0]


def _masked_cov(k, M, m, assume_centered):
    """sklearn's location and empirical covariance of the rows of M whose
    mask entry (m, n x 1, 0 or 1) is 1, on the device: the masked rows are
    exact zeros in the products, the count is the mask's total."""
    cnt = k.total(m).s[0]
    Xm = k.ew("mul", M, m)
    if assume_centered:
        loc = _M.zeros(1, M.c)
        Xc = Xm
    else:
        loc = k.ew("scale", k.colsum(Xm), s=1.0 / cnt)
        Xc = k.ew("mul", k.ew("sub", M, loc), m)
    return loc, k.ew("scale", k.mm(Xc, Xc, ta=True), s=1.0 / cnt)


def _mahal(k, X, loc, P):
    Xc = k.ew("sub", X, loc)
    return k.rowsum(k.ew("mul", k.mm(Xc, P), Xc))


def _chi2_cdf(k, dof, m):
    """P(chi2_dof <= m): the regularized lower incomplete gamma P(dof/2, m/2),
    its series summed in float64 and its prefactor x^a e^-x / Gamma(a + 1)
    through the cells' exp, log and lgamma."""
    a = dof / 2.0
    x = m / 2.0
    if x <= 0:
        return 0.0
    pref = k.ew("exp", k.const(_f32(a * k.ew("logs", k.const(x), s=1e-30).s[0] - x
                                     - k.ew("lgamma", k.const(a + 1)).s[0]))).s[0]
    term, tot, n = 1.0, 1.0, 1
    while n < 4000:
        term *= x / (a + n)
        tot += term
        if term < 1e-17 * tot:
            break
        n += 1
    return pref * tot


def _chi2_quantile(k, dof, upper):
    """The point m with P(chi2_dof > m) = upper (scipy chi2.isf), by bisection."""
    target = 1.0 - upper
    lo, hi = 0.0, max(1.0, 4.0 * dof + 40.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _chi2_cdf(k, dof, mid) < target:
            lo = mid
        else:
            hi = mid
    return _f32(0.5 * (lo + hi))


def _consistency_factor(k, p, alpha):
    """sklearn `_robust_covariance.py::_consistency_factor` (Pison 2002):
    alpha / chi2.cdf(chi2.ppf(alpha, p), p + 2)."""
    q = _chi2_quantile(k, p, 1.0 - alpha)
    return _f32(alpha / _chi2_cdf(k, p + 2, q))


class MinCovDet(_Base):
    """sklearn.covariance.MinCovDet (reference: scikit-learn
    `covariance/_robust_covariance.py`: `c_step`/`_c_step`,
    `select_candidates`, `fast_mcd`, `MinCovDet.fit`, `correct_covariance`,
    `reweight_covariance`). The random subsets are Philox permutations (a
    sort of counter draws, ties to the lower index) and every argsort breaks
    ties by index. chi2 quantiles by bisection of the incomplete gamma; one
    feature takes sklearn's 1-D shortcut."""
    _parameters = ("store_precision", "assume_centered", "support_fraction", "random_state", "numeric_mode")

    def __init__(self, *, store_precision=True, assume_centered=False, support_fraction=None, random_state=None,
                 numeric_mode=None):
        self.store_precision, self.assume_centered = store_precision, assume_centered
        self.support_fraction, self.random_state, self.numeric_mode = support_fraction, random_state, numeric_mode

    # ---- randomness
    def _perm(self, k, n):
        self._draws += 1
        u = k.rand(1, n, self._seed, 1000 + self._draws, 0).s
        return sorted(range(n), key=lambda i: (u[i], i))

    # ---- the C-step
    def _c_step(self, k, X, h, iters, init=None):
        n = X.r
        dist = None
        if init is None:
            sel = self._perm(k, n)[:h]
        else:
            loc0, cov0 = init
            P0 = _pinvh(k, cov0)
            dist = _mahal(k, X, loc0, P0)
            sel = sorted(range(n), key=lambda i: (dist.s[i], i))[:h]
        sel = sorted(sel)
        Xs = X.take_rows(sel)
        loc = k.colmean(Xs)
        cov = _emp_cov(k, Xs)
        det = _fast_logdet(k, cov)
        P = _pinvh(k, cov) if det == -math.inf else None
        prev_det = math.inf
        prev = None
        while det < prev_det and iters > 0 and det != -math.inf:
            prev = (loc, cov, det, sel, dist)
            prev_det = det
            P = _pinvh(k, cov)
            dist = _mahal(k, X, loc, P)
            sel = sorted(sorted(range(n), key=lambda i: (dist.s[i], i))[:h])
            Xs = X.take_rows(sel)
            loc = k.colmean(Xs)
            cov = _emp_cov(k, Xs)
            det = _fast_logdet(k, cov)
            iters -= 1
        prev_dist = dist
        dist = _mahal(k, X, loc, P)
        # sklearn's four checks in its order, the LAST one that fires wins
        res = (loc, cov, det, sel, dist)
        if prev is not None and det > prev_det:
            res = (prev[0], prev[1], prev[2], prev[3], prev_dist)
        if iters == 0:
            res = (loc, cov, det, sel, dist)
        return res

    def _select(self, k, X, h, trials, select, n_iter=30):
        if isinstance(trials, int):
            est = [self._c_step(k, X, h, n_iter) for _ in range(trials)]
        else:
            est = [self._c_step(k, X, h, n_iter, init=t) for t in trials]
        order = sorted(range(len(est)), key=lambda j: (est[j][2], j))[:select]
        return [est[j] for j in order]

    def _mcd_1d(self, k, X, h):
        """sklearn fast_mcd's one-feature shortcut: the shortest window of h
        sorted values (every tie of the minimum width kept), the location the
        mean of their midpoints, the support the h values nearest it (ties to
        the lower index), the variance of the support."""
        n = X.r
        order = sorted(range(n), key=lambda i: (X.s[i], i))
        xs = _M.of([X.s[i] for i in order], n, 1)
        if h < n:
            diff = k.ew("sub", xs.rows(h, n), xs.rows(0, n - h))
            dmin = min(diff.s)
            starts = [i for i, v in enumerate(diff.s) if v == dmin]
            mids = k.ew("scale", k.ew("add", xs.take_rows([h + i for i in starts]), xs.take_rows(starts)), s=0.5)
            loc = k.colmean(mids)
            cen = k.ew("abs", k.ew("sub", X, loc))
            sel = sorted(sorted(range(n), key=lambda i: (cen.s[i], i))[:h])
        else:
            sel = list(range(n))
            loc = k.colmean(X)
        Xs = X.take_rows(sel)
        cov = _emp_cov(k, Xs)
        P = _pinvh(k, cov)
        support = [False] * n
        for i in sel:
            support[i] = True
        return loc, cov, support, _mahal(k, X, loc, P)

    def _fast_mcd_native(self, k, X, h):
        """fast_mcd for two or more features in ONE binding call
        (x_decomp/mcd.mojo, lane/py-decomp-nbrs): the same C-steps, cells,
        draws and (value, index) orders as the Python driver below, which
        made 12 to 40 kit calls per C-step. Only the O(1) plan integers are
        computed here, with the float expressions they always used."""
        n, p = X.r, X.c
        plan = [0] * 7
        if n > 500:
            n_sub = n // 300
            n_ss = n // n_sub
            n_m = min(1500, n)
            plan = [n_sub, n_ss, int(math.ceil(n_ss * (h / float(n)))), max(10, 500 // n_sub),
                    n_m, int(math.ceil(n_m * (h / float(n)))), 10 if n > 1500 else 1]
        loc, cov, dist = _M.zeros(1, p), _M.zeros(p, p), _M.zeros(n, 1)
        sup = array.array("i", [0]) * n
        k.b.x_decomp_mcd(X.addr, loc.addr, cov.addr, sup.buffer_info()[0], dist.addr,
                         [n, p, h, int(self._seed) & 0xFFFFFFFF] + plan)
        return loc, cov, [v != 0 for v in sup], dist

    def _fast_mcd(self, k, X):
        n, p = X.r, X.c
        h = int(math.ceil(0.5 * (n + p + 1))) if self.support_fraction is None else int(self.support_fraction * n)
        if p == 1:
            return self._mcd_1d(k, X, h)
        if _os.environ.get("MOJOLEARN_XD_MCD_PYTHON") != "1":
            return self._fast_mcd_native(k, X, h)
        # THE REFERENCE ARM (MOJOLEARN_XD_MCD_PYTHON=1, timing and A/B only):
        # the same search driven from Python one kit call at a time.
        if n > 500:
            n_sub = n // 300
            n_ss = n // n_sub
            shuf = self._perm(k, n)
            h_sub = int(math.ceil(n_ss * (h / float(n))))
            n_trials = max(10, 500 // n_sub)
            pool = []
            for i in range(n_sub):
                cur = X.take_rows(shuf[i * n_ss:(i + 1) * n_ss])
                pool += [(e[0], e[1]) for e in self._select(k, cur, h_sub, n_trials, 10, n_iter=2)]
            n_m = min(1500, n)
            h_m = int(math.ceil(n_m * (h / float(n))))
            n_best_m = 10 if n > 1500 else 1
            selection = self._perm(k, n)[:n_m]
            merged = self._select(k, X.take_rows(selection), h_m, pool, n_best_m)
            if n < 1500:
                loc, cov, _, sup_sel, d = merged[0]
                support = [False] * n
                dist = [0.0] * n
                for a, idx in enumerate(selection):
                    dist[idx] = d.s[a]
                for a in sup_sel:
                    support[selection[a]] = True
                return loc, cov, support, _M.of(dist, n, 1)
            full = self._select(k, X, h, [(e[0], e[1]) for e in merged], 1)
        else:
            best = self._select(k, X, h, 30, 10, n_iter=2)
            full = self._select(k, X, h, [(e[0], e[1]) for e in best], 1)
        loc, cov, _, sup_sel, d = full[0]
        support = [False] * n
        for a in sup_sel:
            support[a] = True
        return loc, cov, support, d

    def fit(self, X, y=None):
        self.numeric_mode_ = _mode(self.numeric_mode)
        k = self._kit()
        M = _M.from_input(X)
        n, p = M.r, M.c
        self._seed = _seed_of(self.random_state)
        self._draws = 0
        loc, cov, support, dist = self._fast_mcd(k, M)
        if self.assume_centered:
            loc = _M.zeros(1, p)
            sm = _M.of([1.0 if v else 0.0 for v in support], n, 1)
            _, cov = _masked_cov(k, M, sm, True)
            dist = k.rowsum(k.ew("mul", k.mm(M, _pinvh(k, cov)), M))
        self.raw_location_m_, self.raw_covariance_m_ = loc, cov
        self.raw_location_ = loc.out((p,))
        self.raw_covariance_ = cov.out()
        self.raw_support_ = support
        # correct_covariance: consistency at the normal model (the corrected
        # matrix is returned by sklearn and not kept; dist_ is rescaled)
        n_support = sum(1 for v in support if v)
        corr = _consistency_factor(k, p, n_support / n)
        dist = k.ew("scale", dist, s=1.0 / corr)
        # reweight_covariance
        thr = _chi2_quantile(k, p, 0.025)
        # the reweighting mask dist < thr on the device (`le` against the
        # float32 just below thr: the same decision), then the masked moments
        mm = k.ew("le", dist, _M.of([_f32_below(thr)], 1, 1))
        locr, covr = _masked_cov(k, M, mm, self.assume_centered)
        covr = k.ew("scale", covr, s=_consistency_factor(k, p, 0.975))
        mask = [v != 0.0 for v in mm.s]
        self.location_m_, self.covariance_m_ = locr, covr
        self.precision_m_ = _pinvh(k, covr)
        self.location_ = locr.out((p,))
        self.covariance_ = covr.out()
        self.precision_ = self.precision_m_.out() if self.store_precision else None
        self.support_ = mask
        self.dist_m_ = _mahal(k, M, locr, self.precision_m_)
        self.dist_ = self.dist_m_.out((n,))
        self.n_features_in_ = p
        return self

    def mahalanobis(self, X):
        if not hasattr(self, "precision_m_"):
            raise RuntimeError("MinCovDet is not fitted; call fit first")
        k = self._kit()
        d = _mahal(k, _M.from_input(X), self.location_m_, self.precision_m_)
        return d.out((d.r,))

    def get_precision(self):
        return self.precision_m_.out()

    def score(self, X, y=None):
        """sklearn EmpiricalCovariance.score: the Gaussian log-likelihood of
        X under location_ and covariance_ (X's own empirical covariance)."""
        k = self._kit()
        M = _M.from_input(X)
        cov = _emp_cov(k, k.ew("sub", M, self.location_m_), assume_centered=True)
        tr = k.total(k.ew("mul", cov, self.precision_m_)).s[0]
        _, ld = _slogdet(k, self.precision_m_)
        p = M.c
        return -(p * _LOG_2PI) / 2.0 - 0.5 * tr + 0.5 * ld


class EllipticEnvelope(MinCovDet):
    """sklearn.covariance.EllipticEnvelope: MinCovDet, then offset_ the
    `contamination` percentile of -dist_ (linear interpolation, float64)."""
    _parameters = ("store_precision", "assume_centered", "support_fraction", "contamination", "random_state",
                   "numeric_mode")

    def __init__(self, *, store_precision=True, assume_centered=False, support_fraction=None, contamination=0.1,
                 random_state=None, numeric_mode=None):
        super().__init__(store_precision=store_precision, assume_centered=assume_centered,
                         support_fraction=support_fraction, random_state=random_state, numeric_mode=numeric_mode)
        self.contamination = contamination

    def fit(self, X, y=None):
        if not 0 < self.contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")
        super().fit(X)
        v = sorted(-d for d in self.dist_m_.s)
        q = 100.0 * self.contamination / 100.0 * (len(v) - 1)
        lo = int(math.floor(q))
        hi = min(lo + 1, len(v) - 1)
        self.offset_ = v[lo] + (v[hi] - v[lo]) * (q - lo)
        return self

    def score_samples(self, X):
        k = self._kit()
        d = _mahal(k, _M.from_input(X), self.location_m_, self.precision_m_)
        return k.ew("scale", d, s=-1.0).out((d.r,))

    def decision_function(self, X):
        k = self._kit()
        d = _mahal(k, _M.from_input(X), self.location_m_, self.precision_m_)
        return k.ew("adds", k.ew("scale", d, s=-1.0), s=-self.offset_).out((d.r,))

    def predict(self, X):
        vals = self.decision_function(X)
        from ._buffer import frombytes as _fb
        out = array.array("i", [1 if v >= 0 else -1 for v in vals])
        return _fb(out.tobytes(), "<i4", (len(out),))

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)

    def score(self, X, y, sample_weight=None):
        """sklearn OutlierMixin/ClassifierMixin.score: accuracy_score(y,
        predict(X), sample_weight), the (weighted) share of exact label
        matches as an IEEE double (weights summed in order)."""
        pred = self.predict(X).tolist()
        yl = [int(v) for v in (y.tolist() if hasattr(y, "tolist") else list(y))]
        if len(yl) != len(pred):
            raise ValueError("y and X have different numbers of rows")
        if sample_weight is None:
            return sum(1 for a, b in zip(yl, pred) if a == b) / len(pred)
        w = [float(v) for v in (sample_weight.tolist() if hasattr(sample_weight, "tolist") else list(sample_weight))]
        tw = _dsum(w)
        if tw == 0:
            raise ZeroDivisionError("Weights sum to zero, can't be normalized")
        return _dsum(wi for wi, a, b in zip(w, yl, pred) if a == b) / tw


# ================================================================ implicit ALS
class AlternatingLeastSquares(_Base):
    """Implicit-feedback matrix factorization by alternating least squares
    (Hu, Koren and Volinsky 2008; reference: the `implicit` library's
    `als.py` and `cpu/_als.pyx::least_squares`, the exact solver): the
    confidence matrix is alpha * user_items, factors start as uniform draws
    * 0.01 (the Philox stream), and each iteration solves every user row
    and then every item row as one batched Cholesky per row (one thread per
    row, sums in a fixed order). The input is a DENSE users x items array
    (0 = no interaction). calculate_training_loss=True records
    `training_loss_` after every iteration: sum over observed (u, i) of
    c_ui (1 - x_u . y_i)^2, over the rest of (x_u . y_i)^2, plus
    regularization (|X|^2 + |Y|^2), divided by (total confidence + the
    unobserved count) (implicit's calculate_loss), through the cells.
    use_cg=True (implicit's default) is implicit's conjugate-gradient solver
    (`_least_squares_cg`): each half-sweep moves every row from its previous
    factor by `cg_steps` (3) CG steps, stopping when r.r < 1e-20
    (x_decomp/cells.mojo `als_cg_row`, DEVIATION 5321); use_cg=False is the
    exact Cholesky solve. OUR DEFAULT STAYS use_cg=False, the exact solve,
    so existing fits keep their bits."""
    _parameters = ("factors", "regularization", "alpha", "iterations", "use_cg", "cg_steps",
                   "calculate_training_loss", "random_state", "numeric_mode")

    def __init__(self, factors=100, regularization=0.01, alpha=1.0, iterations=15, use_cg=False,
                 calculate_training_loss=False, random_state=None, numeric_mode=None, cg_steps=3):
        self.factors, self.regularization, self.alpha, self.iterations = factors, regularization, alpha, iterations
        self.use_cg, self.calculate_training_loss = use_cg, calculate_training_loss
        self.random_state, self.numeric_mode, self.cg_steps = random_state, numeric_mode, cg_steps

    def fit(self, user_items, show_progress=False):
        self.numeric_mode_ = _mode(self.numeric_mode)
        k = self._kit()
        R = _M.from_input(user_items, "user_items")
        n, m = R.r, R.c
        C = R if self.alpha == 1.0 else k.ew("scale", R, s=self.alpha)
        # resident (lane gap-lda-als): C stays on the device and the item
        # half-sweep reads it through strides; the host column transposes
        res = not self.use_cg and k.als_resident(C)
        Ct = None if res else C.T
        seed = _seed_of(self.random_state)
        f = int(self.factors)
        X = k.ew("scale", k.rand(n, f, seed, 80, 0), s=0.01)
        Y = k.ew("scale", k.rand(m, f, seed, 81, 0), s=0.01)
        losses = []
        if self.use_cg and int(self.cg_steps) < 1:
            raise ValueError("AlternatingLeastSquares: cg_steps must be >= 1")
        for _ in range(int(self.iterations)):
            if self.use_cg:
                X = k.als_cg(C, Y, X, self.regularization, self.cg_steps)
                Y = k.als_cg(Ct, X, Y, self.regularization, self.cg_steps)
            else:
                X = k.als(C, Y, self.regularization)
                Y = k.als(C, X, self.regularization, trans=True) if res else k.als(Ct, X, self.regularization)
            if self.calculate_training_loss:
                losses.append(self._loss(k, C, X, Y))
        if self.calculate_training_loss:
            self.training_loss_ = losses
        self.user_factors_m_, self.item_factors_m_ = X, Y
        self.user_factors, self.item_factors = X.out(), Y.out()
        # sklearn-style fitted names (the board harness and docs read these)
        self.user_factors_, self.item_factors_ = self.user_factors, self.item_factors
        self.components_m_ = Y
        return self

    def _loss(self, k, C, X, Y):
        P = k.mm(X, Y, tb=True)
        seen = k.ew("gts", k.ew("abs", C), s=0.0)                     # 1 where c_ui != 0
        obs = k.ew("mul", C, k.ew("sq", k.ew("adds", k.ew("scale", P, s=-1.0), s=1.0)))
        term = k.ew("select", seen, obs, k.ew("sq", P), s=0.5)
        reg = k.ew("add", k.total(k.ew("sq", X)), k.total(k.ew("sq", Y)))
        tot = k.ew("add", k.total(term), k.ew("scale", reg, s=self.regularization)).s[0]
        nnz = k.total(seen).s[0]
        conf = k.total(k.ew("mul", C, seen)).s[0]
        return tot / (conf + (C.r * C.c - nnz))

    def _check_fit(self):
        if not hasattr(self, "user_factors_m_"):
            raise RuntimeError("AlternatingLeastSquares is not fitted; call fit first")

    def recommend(self, userid, user_items, N=10, filter_already_liked_items=True):
        """(ids, scores) of the N best items for one user: x_u . y_i, ties to
        the lower item id; items the user interacted with (row `userid` of
        user_items, or user_items itself when it is one row) are skipped."""
        self._check_fit()
        k = self._kit()
        U = self.user_factors_m_.rows(userid, userid + 1)
        sc = k.mm(U, self.item_factors_m_, tb=True).s
        liked = set()
        if filter_already_liked_items and user_items is not None:
            R = _M.from_input(user_items, "user_items")
            row = R.row(0 if R.r == 1 else userid)
            liked = {i for i, v in enumerate(row) if v != 0}
        order = sorted((i for i in range(len(sc)) if i not in liked), key=lambda i: (-sc[i], i))[:N]
        from ._buffer import frombytes as _fb
        return (_fb(array.array("i", order).tobytes(), "<i4", (len(order),)),
                _fb(array.array("f", [sc[i] for i in order]).tobytes(), "<f4", (len(order),)))

    def similar_items(self, itemid, N=10):
        """(ids, scores) of the N items whose factors have the largest cosine
        with item `itemid` (itself included, as implicit returns it)."""
        self._check_fit()
        k = self._kit()
        Y = self.item_factors_m_
        nrm = k.ew("sqrt", k.rowsum(k.ew("sq", Y)))
        Yn = k.ew("div", Y, nrm)
        sc = k.mm(Yn.rows(itemid, itemid + 1), Yn, tb=True).s
        order = sorted(range(len(sc)), key=lambda i: (-sc[i], i))[:N]
        from ._buffer import frombytes as _fb
        return (_fb(array.array("i", order).tobytes(), "<i4", (len(order),)),
                _fb(array.array("f", [sc[i] for i in order]).tobytes(), "<f4", (len(order),)))


class SparseCoder(_SparseCoding):
    """sklearn.decomposition.SparseCoder: sparse coding against a FIXED
    dictionary (`_BaseSparseCoding._transform`), the encoders of `_sparse_encode`."""
    _parameters = ("dictionary", "transform_algorithm", "transform_n_nonzero_coefs", "transform_alpha",
                   "split_sign", "positive_code", "transform_max_iter", "numeric_mode")

    def __init__(self, dictionary, *, transform_algorithm="omp", transform_n_nonzero_coefs=None, transform_alpha=None,
                 split_sign=False, n_jobs=None, positive_code=False, transform_max_iter=1000, numeric_mode=None):
        self.dictionary, self.transform_algorithm = dictionary, transform_algorithm
        self.transform_n_nonzero_coefs, self.transform_alpha = transform_n_nonzero_coefs, transform_alpha
        self.split_sign, self.n_jobs, self.positive_code = split_sign, n_jobs, positive_code
        self.transform_max_iter, self.numeric_mode = transform_max_iter, numeric_mode
        _check_sparse_algos("lars", transform_algorithm)
        self.numeric_mode_ = _mode(numeric_mode)
        self.components_m_ = _M.from_input(dictionary, "dictionary")
        self.components_ = self.components_m_.out()
        self.n_components_, self.n_features_in_ = self.components_m_.r, self.components_m_.c

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)


# ================================================================ randomized arms of PCA / TruncatedSVD
def _randomized_decompose(X, nc, *, center, n_oversamples, n_iter, power_iteration_normalizer, random_state,
                          numeric_mode, binding=None):
    """The randomized arm of `decomposition.PCA` (sklearn `_pca.py::_fit_truncated`,
    center=True) and `decomposition.TruncatedSVD` (`_truncated_svd.py`,
    center=False): `randomized_svd` (flip_sign=False) on the (centered) data,
    then `svd_flip(u_based_decision=False)`. Returns a dict of float32
    Arrays and the float noise variance."""
    k = _Kit(_mode(numeric_mode), binding)
    M = _M.from_input(X)
    n, d = M.r, M.c
    mean = k.colmean(M)
    A = k.ew("sub", M, mean) if center else M
    Um, Sm, Vm = _rsvd_core(k, A, nc, n_oversamples, n_iter, power_iteration_normalizer, "auto", False,
                            random_state)
    # svd_flip(u_based_decision=False): each row of Vt, U's columns follow
    fl = k.absmax_flags(Vm, False)
    Vm, Um = Vm.neg_rows(fl), Um.neg_cols(fl)
    out = dict(components=Vm.out(), singular_values=Sm.out((nc,)), mean=mean.out((d,)))
    if center:
        ev = k.ew("scale", k.ew("sq", Sm), s=1.0 / (n - 1))
        tot = k.ew("scale", k.total(k.ew("sq", A)), s=1.0 / (n - 1))
        out["explained_variance"] = ev.out((nc,))
        out["explained_variance_ratio"] = k.ew("div", ev, tot).out((nc,))
        r = min(n, d)
        out["noise_variance"] = (k.ew("scale", k.ew("sub", tot, k.total(ev)), s=1.0 / (r - nc)).s[0]
                                 if nc < r else 0.0)
    else:
        Xt = k.ew("mul", Um, Sm)
        mt = k.colmean(Xt)
        ev = k.ew("scale", k.colsum(k.ew("sqdiff", Xt, mt)), s=1.0 / n)
        full = k.total(k.ew("scale", k.colsum(k.ew("sqdiff", M, mean)), s=1.0 / n))
        out["explained_variance"] = ev.out((nc,))
        out["explained_variance_ratio"] = k.ew("div", ev, full).out((nc,))
    return out

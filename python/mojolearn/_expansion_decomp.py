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
from ._buffer import addr_ro, as_f32_c, as_f64_c, as_i32_c, frombytes
from ._buffer import addr as _addr, empty as _empty

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
        if a.ndim != 2 or min(a.shape) == 0:  # glue: smaller of two shape dims
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
        if a.ndim != 2 or min(a.shape) == 0:  # glue: smaller of two shape dims
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
        if self._d is not None and 0 <= a < b <= self.r and _res_moves(self, (b - a) * self.c):
            # lane fam2-decomp: rows a .. b - 1 gathered on the device
            # (TAKE_ROWS, the row numbers as exact floats); lane
            # cpu2-l8-decomp: at every size and in every mode
            return _dev_gather(self, _MV_TAKE_ROWS, range(a, b))
        return _M(self.s[a * self.c:b * self.c], b - a, self.c)

    def take_rows(self, idx):
        k = _host_kit()
        if self.r >= _F32_INDEX_MAX:
            raise ValueError("x_decomp: take_rows exceeds the float32 row-index bound (2^24 rows)")
        if not isinstance(idx, _M):
            a = array.array("f", idx)
            idx = _M(a, len(a), 1)
        I = idx
        m = I.r * I.c
        if m * self.c and k._res():
            # lane cpu3-python: the kit's device TAKE_ROWS move on a GPU
            # binding (the operand goes up once if it is a host matrix)
            return k.take_rows(self, I)
        out = _M.zeros(m, self.c)
        if m * self.c:
            k.b.x_decomp_move(self.addr, I.addr, out.addr,  # cpu-route: host binding only (CPU-only installs), the GPU binding moved above
                              [_MV_TAKE_ROWS, m * self.c, self.c, 0, 0, 1, 0, len(self.s), m * self.c, m])
        return out

    def cols(self, a, b):
        return self.take_cols(range(a, b))

    def take_cols(self, idx):
        """Data movement by strided slices: one C-level copy per column."""
        idx = list(idx)
        w = len(idx)
        in_range = all(0 <= j < self.c for j in idx)  # glue: range check of the caller's column-index argument
        if self._d is not None and in_range and w and _res_moves(self, self.r * w):
            # lane fam2-decomp: the columns gathered on the device (TAKE_COLS);
            # lane cpu2-l8-decomp: at every size and in every mode
            return _dev_gather(self, _MV_TAKE_COLS, idx)
        out = array.array("f", [0.0]) * (self.r * w)
        for t, j in enumerate(idx):  # glue: one C-level slice copy per selected column on the host matrix (idx-sized: selected column indices)
            out[t::w] = self.s[j::self.c]
        return _M(out, self.r, w)

    @property
    def T(self):
        d = self._d
        n = self.r * self.c
        if d is None and n and self.r != 1 and self.c != 1 and n < _I32_MAX:
            # lane cpu3-python: a host matrix on a GPU binding goes up once
            # (it moves) and is transposed there; the move has no index operand
            k = _host_kit()
            if k._res():
                k._did(self)
                d = self._d
        if d is not None and (self.r == 1 or self.c == 1):
            # lane cpu2-l8-decomp: a vector's transpose is the same words, a
            # view of the same device buffer (as the host form shares its store)
            return _M._on_device(d, self.c, self.r)
        if d is not None and n < _I32_MAX:
            # lane fam2-decomp: transposed on the device (the TRANSPOSE move's
            # index functions; the index operand is unused, so the source's
            # own id stands in for it)
            out = _M._on_device(_DevBuf(d.b, n), self.c, self.r)
            d.b.x_decomp_dev_move(d.id, d.id, out._d.id, [_MV_TRANSPOSE, n, self.r, self.c, 0, 0, 0, n, n, 1])
            return out
        if self.r == 1 or self.c == 1:
            return _M(self.s, self.c, self.r)
        k = _host_kit()
        out = _M.zeros(self.c, self.r)
        n = self.r * self.c
        k.b.x_decomp_move(self.addr, _M._one.addr, out.addr, [_MV_TRANSPOSE, n, self.r, self.c, 0, 0, 0, n, n, 1])  # cpu-route: host binding only (CPU-only installs); a GPU binding transposed above
        return out

    def reshape(self, r, c):
        if self._d is not None and r * c == self.r * self.c:
            return _M._on_device(self._d, r, c)     # lane cpu2-l8-decomp: a view, no download
        return _M(self.s, r, c)

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


#: lane apple-fast-py2mojo-decomp (2026-10-03): x_decomp/moves.mojo `move`
#: ops and the float32 row-index bound (indices travel as exact floats).
_MV_TAKE_ROWS, _MV_TRANSPOSE, _MV_PLACE_COLS, _MV_FILL0 = 0, 1, 2, 3
_MV_TAKE_COLS = 4
_F32_INDEX_MAX = 1 << 24
_I32_MAX = (1 << 31) - 1    # the move entries' int32 counts


def _res_moves(M, count):
    """Whether a move of the device matrix M with `count` result values runs
    on the device (shape facts only). Lane cpu2-l8-decomp (re-audit L8, the
    `_M.take_cols` row): every size and every mode; the IDENTICAL-only
    binding bit and the small-result floor are gone, so a resident matrix
    is never downloaded whole to slice a part of it."""
    return count > 0 and M.r < _F32_INDEX_MAX and M.c < _F32_INDEX_MAX


def _dev_gather(M, op, idx):
    """Rows (TAKE_ROWS) or columns (TAKE_COLS) `idx` of the device matrix M
    as a device matrix: the indices go up once as exact floats (range
    checked by the caller), one thread a moved value."""
    raw = M._d.b
    ia = array.array("f", idx)
    w = len(ia)
    di = _DevBuf(raw, w)
    raw.x_decomp_dev_upload(di.id, ia.buffer_info()[0], w)
    n = M.r * M.c
    if op == _MV_TAKE_ROWS:
        count = w * M.c
        out = _M._on_device(_DevBuf(raw, count), w, M.c)
        raw.x_decomp_dev_move(M._d.id, di.id, out._d.id, [op, count, M.c, 0, 0, 1, 0, n, count, w])
    else:
        count = M.r * w
        out = _M._on_device(_DevBuf(raw, count), M.r, w)
        raw.x_decomp_dev_move(M._d.id, di.id, out._d.id, [op, count, w, M.c, 0, 0, 0, n, count, w])
    return out


def _host_kit():
    """The default-mode kit (for the `_M` host moves, which use only its
    host-address entries). The Python data path and its A/B define are gone
    (Python = glue only, Oct 3 2026): every move is x_decomp/moves.mojo."""
    return _Kit(_backend.default_mode())


def _any_negative(M):
    """Whether any value is < 0, through the cells (x < 0 counted as ones)."""
    if not len(M.s):
        return False
    k = _Kit(_backend.default_mode())
    return k.total(k.ew("gts", k.ew("scale", M, s=-1.0), s=0.0)).s[0] > 0


def _vstack(*ms):
    s = array.array("f")
    for m in ms:  # glue: one C-level copy per argument matrix
        s.extend(m.s)
    return _M(s, sum(m.r for m in ms), ms[0].c)  # glue: row count of argument matrices


def _hstack(*ms):
    r = ms[0].r
    w = sum(m.c for m in ms)  # glue: column count of argument matrices
    k = _host_kit()
    out = _M.zeros(r, w)
    off = 0
    for m in ms:  # glue: one Mojo move per argument matrix
        if m.r * m.c:
            k.b.x_decomp_move(m.addr, _M._one.addr, out.addr,  # cpu-route: host binding only, _Kit.hstack moves on a GPU binding
                              [_MV_PLACE_COLS, m.r * m.c, m.c, w, off, 0, 0, m.r * m.c, r * w, 1])
        off += m.c
    return out


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


def _kit_fast_define(kit, name):
    """lane/apple-fast-rec-decomp (2026-10-04; from
    lane/apple-fast-decomp-linalg@74d52352b): whether the kit's binding is a
    FAST Metal build compiled with `-D <name>` (asked of the binding's
    `x_decomp_fast_defines` once per kit; no env read). False for every
    IDENTICAL kit, another vendor and a binding without the entry, so the
    default never takes one of these routes."""
    if kit.mode != "fast" or _kit_vendor(kit) != "metal":
        return False
    d = kit.__dict__.get("_fast_defines")
    if d is None:
        try:
            d = str(kit._raw().x_decomp_fast_defines()).split(",")
        except Exception:
            d = []
        kit._fast_defines = d
    return name in d


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

    def w4_flags(self):
        """lane/apple-fast-w4-decomp: the binding's compiled w4 candidates
        (`x_decomp_w4_flags`: bit 1 LLE_FAST_DEV_LU, bit 2
        RSVD_FAST_DIRECT_IN; 0 on a binding without the entry)."""
        f = self.__dict__.get("_w4_flags")
        if f is None:
            try:
                f = int(getattr(self._raw(), "x_decomp_w4_flags")())
            except Exception:
                f = 0
            self._w4_flags = f
        return f

    def s_flags(self):
        """lane/apple-fast-s-linalg: the binding's compiled speed candidates
        (`x_decomp_s_flags`, x_decomp/s_linalg_fast.mojo: bit 1
        RSVD_FAST_DEVSCAN, bit 2 DECOMP_FAST_ORTH_WS; 0 on a binding without
        the entry)."""
        f = self.__dict__.get("_s_flags")
        if f is None:
            try:
                f = int(getattr(self._raw(), "x_decomp_s_flags")())
            except Exception:
                f = 0
            self._s_flags = f
        return f

    def qfix_flags(self):
        """lane/apple-fast-q-linalg: the binding's FAST quality repairs
        (`x_decomp_qfix_flags`, x_decomp/qfix.mojo: bit 1 SVD_QFIX, bit 2
        TSVD_QFIX, bit 4 LU_QFIX; each off with its -D MOJOLEARN_*_QOLD; 0 on
        an IDENTICAL or host binding, which keep the old routes)."""
        f = self.__dict__.get("_qfix_flags")
        if f is None:
            try:
                f = int(getattr(self._raw(), "x_decomp_qfix_flags")())
            except Exception:
                f = 0
            self._qfix_flags = f
        return f

    def lu_resid(self, A, X, B):
        """B - A X (n x nrhs), the sum folded in float-float on the device
        (x_decomp/qfix.mojo `lu_resid_ff_kernel`) and rounded once."""
        R = _M.zeros(B.r, B.c)
        self.b.x_decomp_lu_resid(A.addr, X.addr, B.addr, R.addr, [A.r, B.c])
        return R

    def lu_dev_aux(self, A, clamp=False):
        """lane/apple-fast-w4-decomp LLE_FAST_DEV_LU: `lu` then `lu_aux` on
        the device (x_decomp/w4_fast.mojo `dev_lu_aux_py`, the same launches
        on the same words), A untouched. Returns (lu n x n, pm n x 1, im
        n x 1, stats as four floats), lu / pm / im device-resident."""
        n = A.r
        lu, pm, im = self._dout(n, n), self._dout(n, 1), self._dout(n, 1)
        diag, st = self._dout(1, n), self._dout(1, 4)
        self.b.x_decomp_dev_lu_aux(self._did(A), lu._d.id, pm._d.id, im._d.id, diag._d.id, st._d.id,
                                   [n, int(bool(clamp))])
        return lu, pm, im, [float(v) for v in st.s]  # glue: the four-field lu_aux status

    def _use(self, *ms):
        """The resident path for this call: the GPU binding, and an operand
        already on the device or one of at least _RES_MIN values. A small
        host-only call keeps the synchronous host-address path (a device
        result Python reads at once would pay an upload, a launch and a
        download where one call did)."""
        if not self._res():
            return False
        big = False
        for m in ms:  # glue: walks the matrix arguments of one call
            if m is None:
                continue
            if m._d is not None:
                return True
            if m.r * m.c >= _RES_MIN:
                big = True
        return big

    def _opt_dev(self, name):
        """Whether this binding exports the optional resident entry `name`
        (lane fam-decomp: entries compiled only into an IDENTICAL GPU build
        whose _OFF define is not set); asked once per kit and name."""
        got = self.__dict__.setdefault("_opt_fns", {})
        r = got.get(name)
        if r is None:
            r = False
            if self._res():
                try:
                    getattr(self._raw(), name)
                    r = True
                except Exception:
                    r = False
            got[name] = r
        return r

    def _dict_dev(self):
        """`x_decomp_dev_dict_update` when this binding exports it (a FAST
        GPU build without -D MOJOLEARN_DECOMP_FAST_DICT_DEV_OFF), else None."""
        if "_dd_fn" not in self.__dict__:
            fn = None
            if self._res():
                try:
                    fn = getattr(self._raw(), "x_decomp_dev_dict_update")
                except Exception:
                    fn = None
            self._dd_fn = fn
        return self._dd_fn

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

    def vstack_dev(self, ms):
        """`_vstack` on the device: each matrix's rows placed after the
        previous ones by the PLACE_COLS move (a3 = the row offset times the
        width), copies only, so the same values as the host stack."""
        c = ms[0].c
        R = sum(m.r for m in ms)  # glue: row count of argument matrices
        out = self._dout(R, c)
        one = _M._dev_one(self)
        off = 0
        for m in ms:  # glue: one Mojo move per argument matrix
            cnt = m.r * m.c
            if cnt:
                self.b.x_decomp_dev_move(self._did(m), one, out._d.id,
                                         [_MV_PLACE_COLS, cnt, c, c, off * c, 0, 0, cnt, R * c, 1])
            off += m.r
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

    def _idn_flags(self):
        """The binding's `x_decomp_idn_flags` (0 when it has none); asked once
        per kit."""
        f = self.__dict__.get("_idn_bits")
        if f is None:
            try:
                f = int(getattr(self._raw(), "x_decomp_idn_flags")())
            except Exception:
                f = 0
            self._idn_bits = f
        return f

    def rand(self, r, c, seed, stream, kind):
        if r * c >= 1024 and self._res():   # a small draw is read at once: one call
            # lane fam-decomp: drawn into a device matrix by the same kernel,
            # downloaded only if Python reads it (lane cpu2-l8-decomp: in
            # every mode, no longer behind the IDENTICAL-only binding bit)
            out = self._dout(r, c)
            self.b.x_decomp_dev_rand(out._d.id, [r * c, int(seed) & 0xFFFFFFFF, int(stream) & 0xFFFFFFFF, kind])
            return out
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
        if n >= 1 and A.c == n and A._d is not None and A._d.b is self._raw() and self._opt_dev("x_decomp_dev_eigh"):
            # lane fam-decomp: an operand already on the device is solved on
            # a device copy (an IDENTICAL GPU build without
            # -D MOJOLEARN_IDN_EIGH_RESIDENT_OFF); a host operand keeps the
            # host-address call (one upload either way)
            self.b.x_decomp_dev_eigh(A._d.id, w.addr, v.addr, [n, int(uplo)])
            return w, v
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
        piv = array.array("i", [0] * n)
        info = _M.zeros(1, 1)
        if n >= 1 and A.c == n and self._opt_dev("x_decomp_dev_lu") and self._use(A):
            # lane fam-decomp: the factor made in a device matrix from a
            # device copy of A (an IDENTICAL GPU build without
            # -D MOJOLEARN_IDN_LU_RESIDENT_OFF); pivots and info come down
            lu = self._dout(n, n)
            self.b.x_decomp_dev_lu(self._did(A), lu._d.id, piv.buffer_info()[0], info.addr, [n])
            return lu, piv, int(info.s[0])
        lu = A.copy()
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

    # ---- data movement and orders (lane apple-fast-py2mojo-decomp, x_decomp/moves.mojo)

    def take_rows(self, A, idx, m=None, ist=1, ioff=0, radd=0):
        """Rows Int(idx[a * ist + ioff]) + radd of A for a < m (idx an _M of
        exact float row numbers; m defaults to all of idx): one thread per
        copied value on the device, the checked host loop otherwise."""
        if m is None:
            m = idx.r * idx.c
        count = m * A.c
        p = [_MV_TAKE_ROWS, count, A.c, radd, 0, ist, ioff, A.r * A.c, count, idx.r * idx.c]
        if count and self._use(A, idx):
            out = self._dout(m, A.c)
            self.b.x_decomp_dev_move(self._did(A), self._did(idx), out._d.id, p)
            return out
        out = _M.zeros(m, A.c)
        if count:
            self.b.x_decomp_move(A.addr, idx.addr, out.addr, p)  # cpu-route: host binding only (CPU-only installs); a GPU binding takes the device branch (_RES_MIN 1)
        return out

    def copy(self, A):
        """An exact copy of A, on the device when A is there (the PLACE_COLS
        move of all of A into a matrix of its own shape)."""
        n = A.r * A.c
        if n and A._d is not None and self._use(A):
            out = self._dout(A.r, A.c)
            self.b.x_decomp_dev_move(self._did(A), _M._dev_one(self), out._d.id,
                                     [_MV_PLACE_COLS, n, A.c, A.c, 0, 0, 0, n, n, 1])
            return out
        return A.copy()

    def fill0(self, A, start, stride, count):
        """A[start + t * stride] = 0 for t < count, in place."""
        if not count:
            return A
        p = [_MV_FILL0, count, start, stride, 0, 0, 0, A.r * A.c, A.r * A.c, 1]
        if self._use(A):
            did = self._did(A)
            self.b.x_decomp_dev_move(did, _M._dev_one(self), did, p)
            return A
        self.b.x_decomp_move(A.addr, _M._one.addr, A.addr, p)  # cpu-route: host binding only (CPU-only installs); a GPU binding takes the device branch (_RES_MIN 1)
        return A

    # ---- lane cpu2-l8-decomp (2026-10-04, re-audit L8): scalar decisions
    # and small moves without a download of the operand. Each has one host
    # form (host-address entries of both bindings) and one device form (the
    # resident entries) with the same words: select reductions and moves
    # are exact, and the elementwise steps are the kit's own cells.
    _SEL_MAXABS, _SEL_MAX, _SEL_MIN = 0, 1, 2

    def reduce(self, A, op):
        """max |A| (op 0), max A (1) or min A (2) as a 1 x 1 matrix
        (x_decomp/select_ops.mojo; a NaN anywhere gives NaN)."""
        n = A.r * A.c
        if not n:
            raise ValueError("x_decomp: reduce of an empty matrix")
        if self._use(A):
            out = self._dout(1, 1)
            self.b.x_decomp_dev_reduce(self._did(A), out._d.id, [n, int(op)])
            return out
        out = _M.zeros(1, 1)
        self.b.x_decomp_reduce(A.addr, out.addr, [n, int(op)])  # cpu-route: host binding only (CPU-only installs); a GPU binding takes the device branch (_RES_MIN 1)
        return out

    def word(self, A, i=0):
        """A's value at flat position i as a float: one word down (the
        element gathered first when A is on the device)."""
        if A._d is not None and A.r * A.c > 1:
            return float(self.strided(A, 1, 1, i).s[0])
        return float(A.s[i])

    def strided(self, A, count, stride, off=0):
        """A's values at off + t * stride (t < count) as a 1 x count matrix:
        the TAKE_COLS move with one column index, off (off < stride, or
        count 1). Exact copies."""
        if count == 1:
            stride = off + 1
        idx = _M.of([float(off)], 1, 1)
        n = A.r * A.c
        p = [_MV_TAKE_COLS, count, 1, stride, 0, 0, 0, n, count, 1]
        if count and self._use(A):
            out = self._dout(1, count)
            self.b.x_decomp_dev_move(self._did(A), self._did(idx), out._d.id, p)
            return out
        out = _M.zeros(1, count)
        if count:
            self.b.x_decomp_move(A.addr, idx.addr, out.addr, p)  # cpu-route: host binding only (CPU-only installs); a GPU binding takes the device branch (_RES_MIN 1)
        return out

    def place_strided(self, D, V, stride, off=0):
        """D[off + t * stride] = V[t] for every t of V, in place (the
        PLACE_COLS move with width 1). Returns D."""
        cnt = V.r * V.c
        if not cnt:
            return D
        p = [_MV_PLACE_COLS, cnt, 1, stride, off, 0, 0, cnt, D.r * D.c, 1]
        if self._use(D, V):
            self.b.x_decomp_dev_move(self._did(V), _M._dev_one(self), self._did(D), p)
            return D
        self.b.x_decomp_move(V.addr, _M._one.addr, D.addr, p)  # cpu-route: host binding only (CPU-only installs); a GPU binding takes the device branch (_RES_MIN 1)
        return D

    def place_rows(self, D, V, row):
        """V (r x D.c) into rows row .. row + r - 1 of D, in place."""
        return self.place_strided(D, V.reshape(1, V.r * V.c), 1, row * D.c) if V.r * V.c else D

    def diag(self, A):
        """The diagonal of square A as 1 x n."""
        return self.strided(A, A.r, A.r + 1, 0)

    def diag_add(self, A, v):
        """A copy of square A with v (1 x 1, or 1 x n) added to its
        diagonal: one add per diagonal value, the rest copied."""
        return self.place_strided(self.copy(A), self.ew("add", self.diag(A), v), A.r + 1, 0)

    def order_small(self, A):
        """The stable ascending order of A's values (ties to the lower
        index, NaN last) as an n x 1 matrix of exact floats, on the device
        for a resident kit (x_decomp/select_*.mojo; n <= 65536)."""
        n = A.r * A.c
        if n and self._use(A):
            out = self._dout(n, 1)
            self.b.x_decomp_dev_order_small(self._did(A), out._d.id, [n])
            return out
        out = _M.zeros(n, 1)
        if n:
            self.b.x_decomp_order_small(A.addr, out.addr, [n])  # cpu-route: host binding only (CPU-only installs); a GPU binding takes the device branch (_RES_MIN 1)
        return out

    def take_cols_m(self, A, idx, w):
        """Columns idx[0 .. w) of A (idx an _M of exact floats, which may
        live on the device)."""
        p = [_MV_TAKE_COLS, A.r * w, w, A.c, 0, 0, 0, A.r * A.c, A.r * w, w]
        if A.r * w and self._use(A, idx):
            out = self._dout(A.r, w)
            self.b.x_decomp_dev_move(self._did(A), self._did(idx), out._d.id, p)
            return out
        out = _M.zeros(A.r, w)
        if A.r * w:
            self.b.x_decomp_move(A.addr, idx.addr, out.addr, p)  # cpu-route: host binding only (CPU-only installs); a GPU binding takes the device branch (_RES_MIN 1)
        return out

    def count_gt(self, A, s):
        """How many values of A are > s (an exact count; one word down)."""
        if not A.r * A.c:
            return 0
        return int(self.total(self.ew("gts", A, s=float(s))).s[0])

    def vstack(self, ms):
        """`_vstack` on the device for a resident kit (copies only)."""
        if self._res() and sum(m.r * m.c for m in ms):  # glue: value count of argument matrices
            return self.vstack_dev(ms)
        return _vstack(*ms)

    def hstack(self, ms):
        """`_hstack` with the device PLACE_COLS moves for a resident kit."""
        r = ms[0].r
        w = sum(m.c for m in ms)  # glue: column count of argument matrices
        if not (self._res() and r * w):
            return _hstack(*ms)
        out = self._dout(r, w)
        one = _M._dev_one(self)
        off = 0
        for m in ms:  # glue: one Mojo move per argument matrix
            if m.r * m.c:
                self.b.x_decomp_dev_move(self._did(m), one, out._d.id,
                                         [_MV_PLACE_COLS, m.r * m.c, m.c, w, off, 0, 0, m.r * m.c, r * w, 1])
            off += m.c
        return out

    def absmax_signs(self, A, by_col):
        """Per column (by_col, 1 x c) or row (r x 1) of A: -1 where its
        largest-|.| entry (ties to the lower index) is negative, else +1
        (`absmax_sign_cell`, DEVIATION 5317, then one select cell)."""
        cnt = A.c if by_col else A.r
        if cnt and A.r * A.c and self._use(A):
            out = self._dout(1, cnt)
            self.b.x_decomp_dev_absmax(self._did(A), out._d.id, [A.r, A.c, 1 if by_col else 0])
        else:
            out = _M.zeros(1, cnt)
            if cnt and A.r * A.c:
                self.b.x_decomp_absmax_sign(A.addr, out.addr, [A.r, A.c, 1 if by_col else 0])
        return self.neg_signs(out if by_col else out.reshape(cnt, 1))

    def neg_signs(self, V):
        """-1 where V < 0, else +1 (NaN: +1), elementwise: select(-V > 0)."""
        return self.ew("select", self.ew("scale", V, s=-1.0), _M.of([-1.0], 1, 1), _M.of([1.0], 1, 1), s=0.0)

    def _refuse_nan(self, A):
        """A device operand's NaN refusal (the host forms' `f32_key`): its max
        is NaN when any value is, one word down."""
        v = self.word(self.reduce(A, self._SEL_MAX))
        if v != v:
            raise ValueError("x_decomp: a NaN has no order (refused)")

    def _order_dev(self, A, neg=0, skip=None):
        """(order, count) of `x_decomp_dev_order_f` on a resident kit: the
        n x 1 device order and the number of positions not skipped."""
        n = A.r * A.c
        if n >= _F32_INDEX_MAX:
            raise ValueError("x_decomp: order_f exceeds the float32 index bound")
        self._refuse_nan(A)
        out = self._dout(n, 1)
        cnt = self._dout(1, 1)
        sid = self._did(skip) if skip is not None else self._did(A)
        self.b.x_decomp_dev_order_f(self._did(A), sid, out._d.id, cnt._d.id,
                                    [n, int(neg), 1 if skip is not None else 0])
        return out, cnt

    def order(self, A):
        """The stable ascending order of A's values (ties to the lower index)
        as an n x 1 _M of exact floats; NaN refused. Lane cpu3-python: a
        radix sort on the device for a resident kit."""
        n = A.r * A.c
        if n and self._use(A):
            return self._order_dev(A)[0]
        out = _M.zeros(n, 1)
        if n:
            self.b.x_decomp_order_f(A.addr, n, out.addr)  # cpu-route: host binding only (CPU-only installs); a GPU binding sorts on the device above
        return out

    def select_smallest(self, A, h):
        """(sel, mask): the h smallest values' positions by (value, index),
        ascending by position (h x 1 exact floats), and the int32 membership
        of every position."""
        n = A.r * A.c
        if n and self._use(A):
            # lane cpu3-python: two stable radix sorts on the device; the
            # membership is an n x 1 device matrix of 0 / 1
            if n >= _F32_INDEX_MAX or h < 0 or h > n:
                raise ValueError("x_decomp: select_smallest out of range")
            self._refuse_nan(A)
            sel, mask = self._dout(h, 1), self._dout(n, 1)
            self.b.x_decomp_dev_select_smallest(self._did(A), sel._d.id, mask._d.id, [n, h])
            return sel, mask
        sel = _M.zeros(h, 1)
        mask = array.array("i", [0]) * n
        if n:
            self.b.x_decomp_select_smallest(A.addr, [n, h], sel.addr, mask.buffer_info()[0])  # cpu-route: host binding only (CPU-only installs); a GPU binding selects on the device above
        return sel, mask

    def argmin_all(self, A):
        """Every position (exact floats, c x 1) whose value equals the minimum."""
        n = A.r * A.c
        if n and self._use(A):
            # lane cpu3-python: the min, an equality key and a stable sort on
            # the device; the count is one word down
            if n >= _F32_INDEX_MAX:
                raise ValueError("x_decomp: argmin_all exceeds the float32 index bound")
            out, cnt = self._dout(n, 1), self._dout(1, 1)
            self.b.x_decomp_dev_argmin_all(self._did(A), out._d.id, cnt._d.id, [n])
            c = int(self.word(cnt))
            return out.rows(0, c) if c else _M.zeros(0, 1)
        buf = _M.zeros(max(n, 1), 1)
        c = int(self.b.x_decomp_argmin_all(A.addr, n, buf.addr)) if n else 0  # cpu-route: host binding only (CPU-only installs); a GPU binding finds them on the device above
        return _M(buf.s[:c], c, 1)

    def lu_solve(self, lu, piv, B, trans=0):
        out = B.copy()
        p = [lu.r, B.c, trans] if trans else [lu.r, B.c]
        self.b.x_decomp_lu_solve(lu.addr, piv.buffer_info()[0], out.addr, p)
        return out

    def cd_rows(self, W, HHt, XHt, perm):
        """One sklearn `_update_cdnmf_fast` sweep over every row of W, in
        place; returns the total violation (rows ascending)."""
        n, kc = W.r, W.c
        p = array.array("i", perm)
        if "_cd_fn" not in self.__dict__:
            # lane fam-decomp: `x_decomp_dev_cd_rows` when this binding
            # exports it (an IDENTICAL GPU build without
            # -D MOJOLEARN_IDN_CD_RESIDENT_OFF), else None
            fn = None
            if self._res():
                try:
                    fn = getattr(self._raw(), "x_decomp_dev_cd_rows")
                except Exception:
                    fn = None
            self._cd_fn = fn
        if self._cd_fn is not None and n * kc and self._use(W, HHt, XHt):
            # W swept where it lives (it has moved to the device: no host
            # store to go stale), the violations folded there, one float read
            viol = self._dout(n, 1)
            self.b.x_decomp_dev_cd_rows(self._did(W), self._did(HHt), self._did(XHt), p.buffer_info()[0],
                                        viol._d.id, [n, kc])
            return self.total(viol).s[0]
        viol = _M.zeros(n, 1)
        self.b.x_decomp_cd_rows(W.addr, HHt.addr, XHt.addr, p.buffer_info()[0], viol.addr, [n, kc])
        return self.total(viol).s[0]

    def svd(self, A):
        """(S 1 x n DESCENDING, Vt n x n) of a tall A (m >= n): Householder QR
        then the one-sided Jacobi SVD of R (decomposition/'s full-PCA route),
        values sorted descending with ties to the lower index."""
        m, n = A.r, A.c
        s, v = _M.zeros(1, n), _M.zeros(n, n)
        if m >= n >= 1 and self._opt_dev("x_decomp_dev_svd") and self._use(A):
            # lane fam-decomp: the solve on a device copy of the resident
            # operand (an IDENTICAL GPU build without
            # -D MOJOLEARN_IDN_SVD_RESIDENT_OFF); A stays on the device
            self.b.x_decomp_dev_svd(self._did(A), s.addr, v.addr, [m, n])
        else:
            self.b.x_decomp_svd(A.addr, s.addr, v.addr, [m, n])
        # descending by value, ties to the lower index: the stable ascending
        # order of -s (an exact negation), and the gathers, in Mojo
        o = self.order(self.ew("scale", s, s=-1.0))
        return self.take_rows(s.T, o).T, self.take_rows(v.T, o)

    def orth(self, A):
        """A copy of A with its columns orthonormalized: two passes of the
        Householder R and a row-parallel A R^-1 (DEVIATION 5309)."""
        if A.r * A.c and self._use(A):
            Q = self._dout(A.r, A.c)
            # DECOMP_FAST_ORTH_WS (FAST + Apple default, rollback _OFF): the same passes
            # with pooled device work buffers (x_decomp/s_linalg_fast.mojo)
            fn = self.b.x_decomp_dev_orth_ws if self.s_flags() & 2 else self.b.x_decomp_dev_orth
            fn(self._did(A), Q._d.id, [A.r, A.c])
            return Q
        Q = A.copy()
        self.b.x_decomp_orth(Q.addr, [A.r, A.c])
        return Q

    def orth_diag(self, A):
        """`orth`, and the two passes' R-diagonal products (a 1 x A.c
        matrix): the sign of entry j orients Q's column j along A's, 0 marks
        a dependent column (`orth_diag_cell`, lane neural-pass17)."""
        diag = _M.zeros(1, A.c)
        if A.r * A.c and self._use(A):
            Q = self._dout(A.r, A.c)
            self.b.x_decomp_dev_orth_diag(self._did(A), Q._d.id, diag.addr, [A.r, A.c])
            return Q, diag
        Q = A.copy()
        self.b.x_decomp_orth_diag(Q.addr, diag.addr, [A.r, A.c])
        return Q, diag

    def lasso_rows(self, G, Q, W, alpha, max_iter, tol, positive):
        """Row-parallel Lasso CD on the Gram (x_decomp/cells.mojo `lasso_row`),
        W (n x k) the warm start, updated in place."""
        if Q.r * Q.c and self._opt_dev("x_decomp_dev_code_rows") and self._use(G, Q, W):
            # lane fam-decomp: the Gram, Q and the codes stay on the device
            # (an IDENTICAL GPU build without -D MOJOLEARN_IDN_CODE_RESIDENT_OFF)
            self.b.x_decomp_dev_code_rows(self._did(G), self._did(Q), self._did(W),
                                          [0, Q.r, Q.c, int(max_iter), int(positive)], [float(alpha), float(tol)])
            return W
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
        if n >= 1 and lu._d is not None and lu._d.b is self._raw() and self._opt_dev("x_decomp_dev_lu"):
            # lane fam-decomp: read (and clamped) where the factor lives
            self.b.x_decomp_dev_lu_aux(lu._d.id, piv.buffer_info()[0], pm.addr, im.addr, diag.addr, st.addr,
                                       [n, int(bool(clamp))])
            return [float(v) for v in st.s], diag, pm, im  # glue: the four-field lu_aux status
        self.b.x_decomp_lu_aux(lu.addr, piv.buffer_info()[0], pm.addr, im.addr, diag.addr, st.addr,
                               [n, int(bool(clamp))])
        return [float(v) for v in st.s], diag, pm, im  # glue: the four-field lu_aux status

    def lars_rows(self, G, Q, m, nnz):
        """Row-parallel Lars on the Gram (x_decomp/cells.mojo `lars_row`): the
        n x k coefficients, m the samples of each row's problem."""
        if Q.r * Q.c and self._opt_dev("x_decomp_dev_code_rows") and self._use(G, Q):
            W = self._dout(Q.r, Q.c)       # lane fam-decomp: see lasso_rows
            self.b.x_decomp_dev_code_rows(self._did(G), self._did(Q), W._d.id,
                                          [1, Q.r, Q.c, int(m), int(nnz)], [0.0, 0.0])
            return W
        W = _M.zeros(Q.r, Q.c)
        na = _M.zeros(Q.r, 1)
        if Q.r * Q.c:
            self.b.x_decomp_lars_rows(G.addr, Q.addr, W.addr, na.addr, [Q.r, Q.c, int(m), int(nnz)])
        return W

    def omp_rows(self, G, Q, nnz):
        if Q.r * Q.c and self._opt_dev("x_decomp_dev_code_rows") and self._use(G, Q):
            W = self._dout(Q.r, Q.c)       # lane fam-decomp: see lasso_rows
            self.b.x_decomp_dev_code_rows(self._did(G), self._did(Q), W._d.id,
                                          [2, Q.r, Q.c, int(nnz), 0], [0.0, 0.0])
            return W
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
        its k neighbors in Y (`nbr`: an n x k _M of exact float indices)."""
        n, k, idx = X.r, nbr.c, nbr
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

    def qr_r(self, A, consume=False):
        """R (n x n) of the Householder QR of a tall A (decomposition/'s TSQR).

        consume=True: the caller gives A up (it is not read again). A device
        A may then be factored in place (lane fix-d1-decomp,
        IDN_QR_R_INPLACE: no second m x n device buffer); A is left empty
        (0 x 0) when the binding consumed it."""
        R = _M.zeros(A.c, A.c)
        if A.r >= A.c >= 1 and self._opt_dev("x_decomp_dev_qr_r") and self._use(A):
            # lane fam-decomp: the QR of a device copy of the resident
            # operand (an IDENTICAL GPU build without
            # -D MOJOLEARN_IDN_QR_R_RESIDENT_OFF); only R comes down
            used = self.b.x_decomp_dev_qr_r(self._did(A), R.addr, [A.r, A.c, 1 if consume else 0])
            if consume and int(used) == 1:
                # its buffer holds the destroyed factorization: drop it
                # (back to the pool) and leave A an empty matrix
                A._s, A._d, A.r, A.c = array.array("f"), None, 0, 0
            return R
        self.b.x_decomp_qr_r(A.addr, R.addr, [A.r, A.c])
        return R

    def pad_zero_row(self, X):
        """[X; 0]: X ((r) x c) over one zero row. A device X stays on the
        device (lane fix-d1-decomp, `x_decomp_idn_flags` bit 8,
        IDN_LLE_PAD_DEV: a PLACE_COLS move and a FILL0 move, copies only);
        else the host stack."""
        r, c = X.r, X.c
        if c and X._d is not None and self._use(X):     # lane cpu2-l8-decomp: every mode
            tot = (r + 1) * c
            out = self._dout(r + 1, c)
            one = _M._dev_one(self)
            cnt = r * c
            if cnt:
                self.b.x_decomp_dev_move(self._did(X), one, out._d.id,
                                         [_MV_PLACE_COLS, cnt, c, c, 0, 0, 0, cnt, tot, 1])
            did = out._d.id
            self.b.x_decomp_dev_move(did, one, did, [_MV_FILL0, c, cnt, 1, 0, 0, 0, tot, tot, 1])
            return out
        return _M(array.array("f", X.s) + array.array("f", [0.0]) * c, r + 1, c)

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
    k = _Kit(_backend.default_mode())
    return k.ew("mul", Vt, k.absmax_signs(Vt, False))


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
        return {name: getattr(self, name) for name in self._parameters}  # glue: estimator parameter names (get_params)

    def set_params(self, **params):
        for k, v in params.items():  # glue: estimator keyword arguments (set_params)
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
def _ipca_dev_on(k):
    """lane/apple-fast-gap-linalg2-kpca: whether this kit's binding was built
    with -D MOJOLEARN_IPCA_FAST_DEV (FAST + Apple): IncrementalPCA.fit stacks
    each batch's matrix on the device (no download and re-upload of the
    centered batch) and reads its public arrays once after the last batch.
    Lane cpu2-l8-decomp: every GPU kit in every mode (the stack is copies,
    the same words; the IDENTICAL and NVIDIA/AMD FAST fits downloaded each
    centered batch to stack it on the host)."""
    return k._res()


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
        for a in ("components_", "n_samples_seen_"):  # glue: drops two fitted attribute names
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
        dev = _ipca_dev_on(self._kit())
        start = 0
        for _ in range(n // self.batch_size_):  # glue: drives one device partial fit per batch (batch_size_-sized: minibatches)
            end = start + self.batch_size_
            if end + mb > n:
                continue
            self._partial(rows(start, end), publish=not dev, dev=dev)
            start = end
        if start < n:
            self._partial(rows(start, n), publish=not dev, dev=dev)
        if dev and hasattr(self, "_pending"):
            self._publish(*self._pending)
        return self

    def partial_fit(self, X, y=None):
        if not hasattr(self, "numeric_mode_"):
            self.numeric_mode_ = _mode(self.numeric_mode)
        self._partial(_M.from_input(X))
        return self

    def _partial(self, Xb, publish=True, dev=False):
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
            Z = k.vstack([prev, Xc, mc])
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
        if publish:
            self._publish(nc, d, evr, nv)
        else:
            # lane/apple-fast-gap-linalg2-kpca (-D MOJOLEARN_IPCA_FAST_DEV):
            # fit publishes the public arrays once, after its last batch
            self._pending = (nc, d, evr, nv)

    def _publish(self, nc, d, evr, nv):
        """The public attributes from the running device/host matrices."""
        self.__dict__.pop("_pending", None)
        self.components_ = self.components_m_.out()
        self.singular_values_ = self.singular_values_m_.out((nc,))
        self.mean_ = self.mean_m_.out((d,))
        self.var_ = self.var_m_.out((d,))
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


def _srp_grid_count(thr):
    """The number of uniforms k * 2^-24 (k in [0, 2^24)) at or below
    float32(thr): main's sparse keep rate on the draw grid, as an integer
    (MOJOLEARN_XD_FAST_SRP_STRAT). Glue: one float32 rounding and a floor."""
    t32 = array.array("f", [thr])[0]
    if t32 < 0:
        return 0
    return min(int(t32 * 16777216.0) + 1, 16777216)  # exact: t32 * 2^24 is a scaled float32


def _grp_cls2(k):
    """lane/apple-fast-gap-cls2: the random projections' FAST Apple fit
    switches compiled into the kit's binding (x_decomp/resident.mojo
    `grp_cls2_py`: bit 1 NOSCAN, 2 DEVSCAN, 4 LAZY); 0 when it has none."""
    fn = getattr(k.b, "x_decomp_grp_cls2", None)
    return int(fn()) if fn is not None else 0


class _RandomProjection(_Base):
    """sklearn `random_projection.py::BaseRandomProjection`. The matrix is
    drawn from the lane's counter-based Philox stream (x_decomp/cells.mojo
    `rand_cell`), not numpy's generator: the same `random_state` gives the
    same matrix on every box, and a different one than sklearn's."""

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
        # fit reads only the input's shape and refuses a non-finite input;
        # the store (a copy of the whole input) is transform's to build
        # (lane neural-pass27: fit_transform converted the 880 MB input
        # twice at the board's shape)
        k = self._kit()
        # lane/idn-gates: the IDENTICAL GPU binding compiles DEVSCAN (bit 2) too
        cls2 = _grp_cls2(k)
        if cls2 & 8 and not _is_sparse(X) and self._fused_fit(k, X):
            return self
        if cls2 & 3 and not _is_sparse(X):
            # lane/apple-fast-gap-cls2 (FAST + Apple, x_decomp/resident.mojo
            # GRP_CLS2_*): no host walk over X. NOSCAN: the shape only
            # (transform's device projection refuses a non-finite X);
            # DEVSCAN: the refusal as one device scan of X
            a = as_f32_c(X, ndim=2, name="X")[0]
            if a.ndim != 2 or min(a.shape) == 0:  # glue: shape check of the input argument
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

    def _fused_fit(self, k, X):
        """lane apple-fast-gap-kapprox2 MOJOLEARN_XD_FAST_GRP_FUSED (bit 8 of
        `x_decomp_grp_cls2`): the scan, the matrix and its download in one
        binding call with one wait (x_decomp/resident.mojo
        `grp_fit_fused_py`); the same words and the same fit-time refusal as
        the DEVSCAN route, Gaussian and sparse. False (nothing done) where
        it does not apply: 'auto' n_components or
        compute_inverse_components."""
        if self.n_components == "auto" or self.compute_inverse_components or not k._res():
            return False
        a = as_f32_c(X, ndim=2, name="X")[0]
        if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] == 0:
            raise ValueError("X: a nonempty two-dimensional input is required")
        n, d = a.shape
        kc = int(self.n_components)
        if kc <= 0:
            raise ValueError(f"n_components must be greater than 0, got {kc}")
        extra = []
        if isinstance(self, SparseRandomProjection):
            dens = 1.0 / math.sqrt(d) if self.density == "auto" else float(self.density)
            if not 0 < dens <= 1:
                raise ValueError(f"Expected density in range ]0, 1], got: {dens}")
            mode, sc, thr = (2 if dens == 1 else 1), math.sqrt(1.0 / dens) / math.sqrt(kc), dens - 2.0 ** -25
            if mode == 1 and _grp_cls2(k) & 16:
                # lane apple-fast-w2-kfeat MOJOLEARN_XD_FAST_SRP_STRAT
                # (x_decomp/resident.mojo `srp_strat_kernel`): the
                # column-stratified pattern; D = the count of 2^-24 grid
                # points main's `u <= float32(thr)` keeps, so every entry is
                # nonzero with main's probability
                mode, extra = 3, [d, _srp_grid_count(thr)]
        else:
            dens, mode, sc, thr = None, 0, 1.0 / math.sqrt(kc), 0.0
        C = k._dout(kc, d)
        res = array.array("f", [0.0]) * (kc * d)
        bad = int(k.b.x_decomp_grp_fit_fused(addr_ro(a, name="X"), a.size, C._d.id, res.buffer_info()[0],
                                             [kc * d, int(_seed_of(self.random_state)) & 0xFFFFFFFF, mode] + extra,
                                             [sc, thr]))
        if bad >= 0:
            raise ValueError("X: input must be finite; NaN/inf are unsupported")
        if dens is not None:
            self.density_ = dens
        self.n_components_ = kc
        self.n_features_in_ = d
        self.components_m_ = C
        self.components_ = Array._owned(res, (kc, d), "<f4", "C")
        return True

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
        if a.ndim != 2 or min(a.shape) == 0:  # glue: smaller of two shape dims
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
    # the signs as a +-1 vector on the device (lane cpu2-l8-decomp): exact
    # multiplies, no flag list on the host
    if u_based:
        sg = k.absmax_signs(U, True)
        U, Vt = k.ew("mul", U, sg), k.ew("mul", Vt, sg.T)
    else:
        sg = k.absmax_signs(Vt, False)
        U, Vt = k.ew("mul", U, sg.T), k.ew("mul", Vt, sg)
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
        # the per-component factor pairs in ONE binding call (lane
        # py-runtime-b: x_decomp/nmf.mojo `nmf_nndsvd`, the same cells per
        # component and the same float32 products of the norms; on resident
        # device matrices in the GPU binding)
        W, H = _M.zeros(n, nc), _M.zeros(nc, d)
        k.b.x_decomp_nmf_nndsvd(U.addr, S.addr, Vt.addr, W.addr, H.addr, [n, d, S.r * S.c, nc])
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

    # ---- solvers
    def _solve(self, k, M, W, H, update_H, regs):
        """`_cd` (solver 'cd') or `_mu` (frobenius) / `_mu_beta` (KL, IS) in
        ONE binding call (lane py-runtime-b: x_decomp/nmf.mojo on the host
        column, x_decomp/nmf_dev.mojo on resident device matrices): the same
        cells, broadcast modes and float32 scalars as the Python drivers it
        replaces, their float64 convergence tests in the same order, the
        shuffle orders from the same Philox streams (200 + draw). W and H
        are copied; the binding replaces the copies."""
        l1W, l1H, l2W, l2H = regs
        W, H = W.copy(), H.copy()
        p = [M.r, M.c, W.c, 0 if self.solver == "cd" else 1, int(bool(update_H)), int(self.max_iter),
             int(bool(self.shuffle)), _seed_of(self.random_state) & 0xFFFFFFFF]
        f = [self._beta(), float(self.tol), float(l1W), float(l1H), float(l2W), float(l2H)]
        it = int(k.b.x_decomp_nmf_solve(M.addr, W.addr, H.addr, p, f))
        if it < 0:
            raise ZeroDivisionError("float division by zero")
        return W, H, it

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
        W, H, it = self._solve(k, M, W, H, update_H, regs)
        return W, H, it, nc

    def fit_transform(self, X, y=None, W=None, H=None):
        self.numeric_mode_ = _mode(self.numeric_mode)
        M = _M.from_input(X)
        self._validate(M)
        if self.init == "custom":
            k = self._kit()
            Wm, Hm = _M.from_input(W, "W"), _M.from_input(H, "H")
            regs = self._reg(M.r, M.c)
            Wm, Hm, it = self._solve(k, M, Wm, Hm, True, regs)
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

    def _solve(self, k, X1, Winit):
        """`_par` (with `_sym_decorrelation`) or `_def` in ONE binding call
        (lane py-runtime-b: x_decomp/ica.mojo on the host column,
        x_decomp/ica_dev.mojo on resident device matrices): the same cells,
        broadcast modes and float32 scalars as the Python loops it replaces,
        their float64 limits and norms in the same order. Returns (W, it)."""
        alpha = 1.0
        if self.fun == "logcosh":
            alpha = (self.fun_args or {}).get("alpha", 1.0)
            if not 1 <= alpha <= 2:
                raise ValueError("alpha must be in [1,2]")
        elif self.fun not in ("exp", "cube"):
            raise ValueError("fun must be 'logcosh', 'exp' or 'cube' (a callable is not carried)")
        fun = {"logcosh": 0, "exp": 1, "cube": 2}[self.fun]
        W = Winit.copy()
        p = [W.r, X1.r, X1.c, 0 if self.algorithm == "parallel" else 1, fun, int(self.max_iter)]
        it = int(k.b.x_decomp_ica_solve(X1.addr, W.addr, p, [float(alpha), float(self.tol)]))
        return W, it

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
            u = k.ew("mul", u, k.neg_signs(u.rows(0, 1)))   # row 0's signs, on the device
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
        W, it = self._solve(k, X1, Winit)
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


def _f32_store(a):
    """An `array.array('f')` holding a float32 Array's bytes (one memcpy)."""
    st = array.array("f")
    st.frombytes(a.tobytes())
    return st


def _support_of(mask, n):
    """A 0/1 int32 membership store as a public '<u1' Array (the Mojo cast helper)."""
    from ._buffer import frombytes as _fb
    return _fb(mask.tobytes(), "<i4", (n,)).astype("<u1")


def _dsum_sq(k, M):
    """The binary64 sum of the squares of M's float32 values (each square
    exact), DSUM_CHUNK values per partial then the partials folded the same
    way: on the device in software binary64 for a resident kit (lane
    cpu3-python), the same words on the host binding (`dsum_sq_host`)."""
    n = M.r * M.c
    if not n:
        return 0.0
    if k._use(M):
        out = k._dout(1, 2)
        k.b.x_decomp_dev_dsum_sq(k._did(M), out._d.id, [n])
        w = array.array("d")
        w.frombytes(out.s.tobytes())   # glue: the binary64 result's two words, as bytes
        return w[0]
    return float(k.b.x_decomp_dsum_sq(M.addr, n))  # cpu-route: host binding only (CPU-only installs); a GPU binding sums on the device above


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
    diag = k.diag(lu)       # lane cpu2-l8-decomp: gathered where the LU lives
    return k.total(k.ew("logs", k.ew("abs", diag), s=1.1754943508222875e-38)).s[0]


def _polar(k, A):
    """U V^T of the SVD of a square A, and the sum of its singular values:
    A V S^-1 V^T through the eigh of A^T A.

    A is first scaled by a power of two, 2^-e with 2^(e-1) <= max|A| < 2^e,
    and the singular-value sum scaled back by 2^e. The polar factor is
    scale-invariant and a power-of-two scale is exact, so a finite A^T A
    gives the same words as before; what changes is that A^T A can no longer
    overflow. Varimax on large loadings (FactorAnalysis on the identity
    reference's `wide` fixture, columns up to 1e4) cubed them into an A whose
    A^T A was inf in float32, and eigh refused it (DEVIATION 590) on every
    column. A is n_components x n_components: the max is a k x k host read."""
    # lane fix-d1-decomp: one word down; lane cpu2-l8-decomp: in every mode
    m = k.word(k.reduce(A, k._SEL_MAXABS)) if A.r * A.c else 0.0
    # clamped so 2^-e stays a normal float32 in the device's scale
    e = max(-120, min(120, math.frexp(m)[1])) if m > 0.0 and math.isfinite(m) else 0
    if e:
        A = k.ew("scale", A, s=math.ldexp(1.0, -e))
    w, V = k.eigh(k.mm(A, A, ta=True))
    sv = k.ew("sqrt", w)
    AV = k.ew("div", k.mm(A, V), sv)
    return k.mm(AV, V, tb=True), math.ldexp(k.total(sv).s[0], e)


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


#: x_decomp/fa_fast.mojo's limits (kernel-derived): FA_MAX_D, FA_TR_MAXK, FA_TR_FLOATS
_FA_MAX_D = 256
_FA_TR_MAXK = 16
_FA_TR_FLOATS = 4096


def _fa_fast_defines(k):
    """The FactorAnalysis FAST defines this binding was built with (a FAST
    Apple build registers `x_decomp_fa_defines`; every other binding has no
    such entry: the empty set). Asked once per kit; no env read."""
    got = k.__dict__.get("_fa_defs")
    if got is None:
        got = frozenset()
        if k._res():
            try:
                got = frozenset(filter(None, str(getattr(k._raw(), "x_decomp_fa_defines")()).split(",")))
            except Exception:
                got = frozenset()
        k._fa_defs = got
    return got


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
        # lane/apple-fast-quality-glmfa (2026-10-03): the two-pass mean. The
        # float32 blocked column sum of a million rows carries a relative
        # error near 1e-6, so a constant column c had mean c(1 + e), centred
        # residuals c e and a variance (c e)^2 where theirs (float64) is 0:
        # its psi stopped near (c e)^2 instead of the 1e-12 floor, and the
        # mean_ the score subtracts was off by c e. The second pass adds the
        # mean of the residuals (exact for a constant column: c is recovered
        # to the word, the residuals are 0). Device launches only.
        mean = k.colmean(M)
        mean = k.ew("add", mean, k.colmean(k.ew("sub", M, mean)))
        llconst = d * _LOG_2PI + nc
        # FAST on Apple (lane/apple-fast-fa, recovered 2026-10-04): taken only
        # when the binding reports MOJOLEARN_FA_GRAM_ONCE or
        # MOJOLEARN_FA_ITER_DEVICE (x_decomp/fa_fast.mojo; ITER_DEVICE is the
        # FAST + Apple default since 2026-10-04, rollback
        # -D MOJOLEARN_FA_ITER_DEVICE_OFF); an IDENTICAL
        # binding registers no FA entry, so this never runs there
        fdefs = _fa_fast_defines(k)
        if (("MOJOLEARN_FA_GRAM_ONCE" in fdefs or "MOJOLEARN_FA_ITER_DEVICE" in fdefs)
                and d <= _FA_MAX_D and 1 <= nc <= d and self.max_iter >= 1):
            return self._fit_fast(k, M, mean, n, d, nc, llconst, fdefs)
        Xc = k.ew("sub", M, mean)
        M = None     # not read again: the input's device copy goes back to the pool before the QR
        nsqrt = math.sqrt(n)
        psi = self._psi_init(k, d)
        SMALL = 1e-12
        # Xc = Q R once; the scaled data Xc D / sqrt(n) then has the singular
        # values and right vectors of the d x d R D / sqrt(n) (Q orthogonal)
        # (n >= d never reads Xc again, so the QR may consume it: one copy of X on the device)
        Rx = k.qr_r(Xc, consume=True) if n >= d else None
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
            Vfull = Vt
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
            # lane/apple-fast-quality-glmfa (2026-10-03): psi without the
            # cancellation. Theirs is var - sum_k W_kj^2; both terms are near
            # var for a column the factors explain, and in float32 their
            # difference has a floor near var * 1e-7 (theirs, float64, keeps
            # falling toward the 1e-12 floor: Istella's held-out
            # log-likelihood 89.0 against 98.1). With q = (sqrt psi + SMALL)^2
            # the scaled data's column norms are var_j / q_j = sum_i V_ij^2 s2_i
            # (every singular vector), so var_j - sum_k W_kj^2 = q_j (sum_i
            # V_ij^2 w_i), w_i = min(s2_i, 1) for the nc kept components and
            # s2_i past them: a sum of nonnegative terms, relative accuracy at
            # any psi. The same value in exact arithmetic; d x d device ops.
            dfull = Vfull.r
            keep = _M.of([1.0] * nc + [0.0] * (dfull - nc), 1, dfull)
            drop = _M.of([0.0] * nc + [1.0] * (dfull - nc), 1, dfull)
            wts = k.ew("add", k.ew("mul", k.ew("mins", s2, s=1.0), keep), k.ew("mul", s2, drop))
            share = k.mm(wts, k.ew("sq", Vfull))
            psi = k.ew("maxs", k.ew("mul", k.ew("sq", sqrt_psi), share), s=SMALL)
        return self._fit_store(k, W, psi, mean, loglike, it, d, nc)

    def _psi_init(self, k, d):
        if self.noise_variance_init is None:
            return k.const(1.0, 1, d)
        psi = _M.of([float(v) for v in self.noise_variance_init], 1, len(self.noise_variance_init))  # glue: converts the noise_variance_init argument
        if psi.c != d:
            raise ValueError(f"noise_variance_init dimension does not match the number of features : {psi.c} != {d}")
        return psi

    def _fit_fast(self, k, M, mean, n, d, nc, llconst, fdefs):
        """FAST on Apple (lane/apple-fast-fa@3efbce2af; x_decomp/fa_fast.mojo):
        the centred Gram G (d x d) in ONE pass over the resident X
        (MOJOLEARN_FA_GRAM_ONCE: X D / sqrt(n) has the spectrum and right
        vectors of D G D / n), then the EM loop on G only, either here (the
        eigh of D G D / n, main's psi update) or as ONE binding call
        (MOJOLEARN_FA_ITER_DEVICE, `fa_em_py`)."""
        psi = self._psi_init(k, d)
        # MOJOLEARN_FA_GRAM_DF (lane/apple-fast-fa-quality): the binding forms
        # G in double-float, hi words then lo words, so G's buffer is 2 d x d
        G = k._dout(2 * d if "MOJOLEARN_FA_GRAM_DF" in fdefs else d, d)
        var = k._dout(1, d)
        k.b.x_decomp_fa_gram(k._did(M), k._did(mean), G._d.id, var._d.id, [n, d])
        if "MOJOLEARN_FA_ITER_DEVICE" in fdefs:
            p0 = array.array("f", psi.s)
            wa = array.array("f", [0.0]) * (nc * d)
            pa = array.array("f", [0.0]) * d
            la = array.array("d", [0.0]) * self.max_iter
            it = int(k.b.x_decomp_fa_em(G._d.id, p0.buffer_info()[0], wa.buffer_info()[0], pa.buffer_info()[0],
                                        la.buffer_info()[0], [d, nc, n, self.max_iter], float(self.tol)))
            return self._fit_store(k, _M(wa, nc, d), _M(pa, 1, d), mean, list(la[:it]), it, d, nc)
        SMALL = 1e-12
        old_ll = -math.inf
        loglike = []
        order = list(range(d - 1, -1, -1))
        keep = _M.of([1.0] * nc + [0.0] * (d - nc), 1, d)
        drop = _M.of([0.0] * nc + [1.0] * (d - nc), 1, d)
        it = 0
        W = None
        for it in range(1, self.max_iter + 1):  # glue: EM iteration driver; every step is device launches (main's loop form)
            sqrt_psi = k.ew("adds", k.ew("sqrt", psi), s=SMALL)
            ev, V = k.eigh(k.ew("scale", k.ew("div", k.ew("div", G, sqrt_psi), sqrt_psi.T), s=1.0 / n))
            s2 = k.ew("maxs", ev.take_cols(order), s=0.0)
            # the nc leading right vectors signed as main's SVD route signs them
            Vfull = V.take_cols(order).T
            Vt = _svd_flip_v(Vfull.rows(0, nc))
            sk = s2.cols(0, nc)
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
            # main's cancellation-free psi update (lane/apple-fast-quality-glmfa)
            wts = k.ew("add", k.ew("mul", k.ew("mins", s2, s=1.0), keep), k.ew("mul", s2, drop))
            share = k.mm(wts, k.ew("sq", Vfull))
            psi = k.ew("maxs", k.ew("mul", k.ew("sq", sqrt_psi), share), s=SMALL)
        return self._fit_store(k, W, psi, mean, loglike, it, d, nc)

    def _fit_store(self, k, W, psi, mean, loglike, it, d, nc):
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
        W = self.components_m_
        if ("MOJOLEARN_FA_TRANSFORM_FUSED" in _fa_fast_defines(k)
                and W.r <= _FA_TR_MAXK and W.c * (W.r + 1) <= _FA_TR_FLOATS):
            # FAST on Apple (lane/apple-fast-fa, recovered): (X - mean) P in
            # ONE launch over rows, P = (W / psi)^T cov_z (d x nc) in
            # threadgroup memory (x_decomp/fa_fast.mojo fa_transform_kernel)
            Xm = _M.from_input(X)
            if Xm.c != W.c:
                raise ValueError(f"x_decomp: cannot broadcast 1x{W.c} against {Xm.r}x{Xm.c}")
            Wpsi = k.ew("div", W, self.noise_variance_m_)
            cov_z = _inv(k, k.ew("add", _eye(W.r), k.mm(Wpsi, W, tb=True)))
            P = k.mm(Wpsi, cov_z, ta=True)
            out = k._dout(Xm.r, W.r)
            k.b.x_decomp_fa_transform(k._did(Xm), k._did(self.mean_m_), k._did(P), out._d.id, [Xm.r, Xm.c, W.r])
            return out.out()
        M = k.ew("sub", _M.from_input(X), self.mean_m_)
        Wpsi = k.ew("div", W, self.noise_variance_m_)
        cov_z = _inv(k, k.ew("add", _eye(W.r), k.mm(Wpsi, W, tb=True)))
        return k.mm(k.mm(M, Wpsi, tb=True), cov_z).out()

    def _cov(self, k):
        W = self.components_m_
        # lane cpu2-l8-decomp: the noise variances added to the diagonal
        # where C lives (one add a diagonal value; no d x d round trip)
        return k.diag_add(k.mm(W, W, ta=True), self.noise_variance_m_)

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
    (Minka's MLE of the PCA rank) over the full explained-variance spectrum,
    in software binary64 on the device on every vendor (x_decomp/pca_mle.mojo,
    lane cpu3-python): the spectrum goes up once as binary64 words, the rank
    comes back as one word. The host binding (CPU-only installs, the host
    column) runs the same phases on the host. The argmax takes the first
    maximum, as numpy."""
    k = _Kit(mode)
    d = len(spectrum)
    if d < 2:
        return 0
    words = array.array("f")
    words.frombytes(array.array("d", spectrum).tobytes())   # glue: the binary64 words, as bytes
    if k._res():
        raw = k._raw()
        sp = _DevBuf(raw, 2 * d)
        raw.x_decomp_dev_upload(sp.id, words.buffer_info()[0], 2 * d)
        out = _DevBuf(raw, 1)
        raw.x_decomp_dev_pca_mle_rank(sp.id, out.id, [d, int(n_samples)])
        got = array.array("f", [0.0])
        raw.x_decomp_dev_download(out.id, got.buffer_info()[0], 1)
        return int(got[0])
    return int(k.b.x_decomp_pca_mle_rank(words.buffer_info()[0], [d, int(n_samples)]))  # cpu-route: host column binding, CPU-only installs without device entries


# ================================================================ LU
def lu_factor(a, *, numeric_mode=None):
    """scipy.linalg.lu_factor (LAPACK getrf semantics): `(lu, piv)` with L
    unit-lower and U in one n x n float32 matrix and `piv` the 0-based row
    interchanges, applied in order. Partial pivoting on the largest |a[i, k]|,
    ties broken by the LOWEST row index. A zero pivot is kept (getrf's
    info > 0) and warned about, as scipy warns."""
    k = _Kit(_mode(numeric_mode))
    if k.s_flags() & 4 and k.qfix_flags() & 4 and not _is_sparse(a):
        # LU_FAST_RESIDENT (FAST + Apple default, rollback _OFF; x_decomp/s_linalg_fast.mojo)
        return _lu_factor_resident(k, a)
    A = _M.from_input(a, "a")
    if A.r != A.c:
        raise ValueError(f"expected a square matrix, got {A.r} x {A.c}")
    lu, piv, info = k.lu(A)
    if info:
        import warnings
        warnings.warn(f"Diagonal number {info} is exactly zero. Singular matrix.", RuntimeWarning, stacklevel=2)
    pair = (lu.out(), frombytes(piv.tobytes(), "<i4", (A.r,)))
    if not info and k.qfix_flags() & 4:
        # LU_QFIX (FAST default; -D MOJOLEARN_LU_QOLD off): the pair carries
        # this call's own copy of A so `lu_solve` can refine (see _LUPair)
        return _LUPair(pair, A)
    return pair


def _lu_factor_resident(k, a):
    """lane/apple-fast-s-linalg LU_FAST_RESIDENT: `lu_factor` with A uploaded
    once from the caller's buffer, the NaN/inf refusal and the factor on the
    device, LU and piv downloaded once into the returned arrays; A, LU and
    piv stay on the device in the returned pair for `lu_solve`."""
    a_arr = as_f32_c(a, ndim=2, name="a")[0]
    if a_arr.ndim != 2 or min(a_arr.shape) == 0:  # glue: smaller of two shape dims
        raise ValueError("a: a nonempty two-dimensional input is required")
    n, c = a_arr.shape
    if n != c:
        raise ValueError(f"expected a square matrix, got {n} x {c}")
    raw = k._raw()
    dA, dLU, dP = _DevBuf(raw, n * n), _DevBuf(raw, n * n), _DevBuf(raw, n)
    lu, piv, info = _empty((n, n), "<f4"), _empty((n,), "<i4"), _empty((1,), "<f4")
    rc = int(k.b.x_decomp_dev_lu_factor(addr_ro(a_arr, name="a"), _addr(lu, name="lu"), _addr(piv, name="piv"),
                                        _addr(info, name="info"), dA.id, dLU.id, dP.id, [n]))
    if rc < 0:
        raise ValueError("a: input must be finite; NaN/inf are unsupported")
    inf = int(array.array("f", info.tobytes())[0])
    if inf:
        import warnings
        warnings.warn(f"Diagonal number {inf} is exactly zero. Singular matrix.", RuntimeWarning, stacklevel=3)
        return (lu, piv)
    return _LUPair((lu, piv), None, dev=(dA, dLU, dP, lu, piv))


def _lu_solve_resident(k, dev, b):
    """LU_FAST_RESIDENT's `lu_solve` (trans 0) on the pair's device A, LU and
    piv: B up, the solve, LU_QFIX's refinement step and X down."""
    dA, dLU, dP, lu, _ = dev
    n = lu.shape[0]
    vec = len(getattr(b, "shape", ())) == 1 or (not hasattr(b, "shape") and not isinstance(b[0], (list, tuple)))
    bb = as_f32_c(b, ndim=1 if vec else 2, name="b")[0]
    if bb.shape[0] != n:
        raise ValueError(f"b has {bb.shape[0]} rows, the factorization has {n}")
    nrhs = 1 if vec else bb.shape[1]
    if not n or not nrhs:
        return None
    if _host_all_finite(bb) is False:
        raise ValueError("b: input must be finite; NaN/inf are unsupported")
    out = _empty((n,) if vec else (n, nrhs), "<f4")
    k.b.x_decomp_dev_lu_solve(dLU.id, dP.id, dA.id, addr_ro(bb, name="b"), _addr(out, name="x"), [n, nrhs, 1])
    return out


class _LUPair(tuple):
    """`lu_factor`'s (lu, piv), unpacked and indexed as the plain tuple
    scipy returns, plus `_qfix_a`: the float32 copy of A the factor came
    from (lane/apple-fast-q-linalg LU_QFIX). `lu_solve` uses it for one
    step of iterative refinement, x += LU^-1 (B - A x), the residual in
    float-float on the device. #: audit 2026-10-04: lu-solve / lu-factor
    synthetic relative_residual 3.26e-06 vs numpy 3.26e-08, torch-gpu
    8.23e-07; -D MOJOLEARN_LU_QOLD restores the unrefined solve. KEPT for
    quality 2026-10-04 (rab5-lu: 2.59e-6 -> 3.26e-8, +14-15% time)."""

    def __new__(cls, pair, a_m, dev=None):
        obj = super().__new__(cls, pair)
        obj._qfix_a = a_m
        # LU_FAST_RESIDENT: (device A, device LU, device piv, the lu and piv
        # arrays returned); `lu_solve` uses them only for this same lu / piv
        obj._dev = dev
        return obj


def lu_solve(lu_and_piv, b, *, trans=0, numeric_mode=None):
    """scipy.linalg.lu_solve (LAPACK getrs): solve A x = b (trans=0) or
    A^T x = b (trans=1, and 2, which is the same for a real matrix) from
    `lu_factor`'s pair. `b` is n or n x nrhs; a zero pivot yields 0 in that
    component (never inf or NaN)."""
    if trans not in (0, 1, 2):
        raise ValueError("trans must be 0, 1 or 2")
    lu, piv = lu_and_piv
    k = _Kit(_mode(numeric_mode))
    dev = getattr(lu_and_piv, "_dev", None)
    if dev is not None and not trans and lu is dev[3] and piv is dev[4] and k.s_flags() & 4:
        # LU_FAST_RESIDENT (FAST + Apple default, rollback _OFF): the pair's device factor
        X = _lu_solve_resident(k, dev, b)
        if X is not None:
            return X
    L = _M.from_input(lu, "lu")
    n = L.r
    pa = as_i32_c(piv, ndim=1, name="piv")[0]
    pv = array.array("i")
    pv.frombytes(pa.tobytes())
    if len(pv) != n or any(not 0 <= p < n for p in pv):  # glue: validates the piv argument's row indices
        raise ValueError("piv must hold one row index in [0, n) per row")
    vec = len(getattr(b, "shape", ())) == 1 or (not hasattr(b, "shape") and not isinstance(b[0], (list, tuple)))
    B = _M.from_input(_row_of(b), "b").T if vec else _M.from_input(b, "b")
    if B.r != n:
        raise ValueError(f"b has {B.r} rows, the factorization has {n}")
    X = k.lu_solve(L, pv, B, trans=1 if trans else 0)
    A0 = getattr(lu_and_piv, "_qfix_a", None)
    if A0 is not None and not trans and A0.r == n and A0.c == n and k.qfix_flags() & 4:
        # LU_QFIX: one refinement step (lane/apple-fast-q-linalg; see _LUPair)
        X = k.ew("add", X, k.lu_solve(L, pv, k.lu_resid(A0, X, B)))
    return X.out((n,)) if vec else X.out()


def solve(a, b, *, numeric_mode=None):
    """numpy.linalg.solve through lu_factor + lu_solve (gesv). Lane
    idn-dense-linalg: one binding entry (`x_decomp_lu_gesv`) when the
    binding routes there (`x_decomp_idn_flags` bit 1): the same launches
    with the factor and the pivots resident, so only A and B go up and X
    comes down; the same words, warning and refusals."""
    k = _Kit(_mode(numeric_mode))
    try:
        flags = int(getattr(k._raw(), "x_decomp_idn_flags")())
    except (ImportError, AttributeError):
        flags = 0
    if not flags & 2:
        return lu_solve(lu_factor(a, numeric_mode=numeric_mode), b, numeric_mode=numeric_mode)
    A = _M.from_input(a, "a")
    if A.r != A.c:
        raise ValueError(f"expected a square matrix, got {A.r} x {A.c}")
    n = A.r
    vec = len(getattr(b, "shape", ())) == 1 or (not hasattr(b, "shape") and not isinstance(b[0], (list, tuple)))
    B = _M.from_input(_row_of(b), "b").T if vec else _M.from_input(b, "b")
    if B.r != n:
        raise ValueError(f"b has {B.r} rows, the factorization has {n}")
    X = B          # from_input's own store: solved in place
    info = _M.zeros(1, 1)
    # ORDER MATCHES x_decomp/api.mojo lu_gesv_py: (a, b in/out, info), (n, nrhs)
    k.b.x_decomp_lu_gesv(A.addr, X.addr, info.addr, [n, B.c])
    if int(info.s[0]):
        import warnings
        warnings.warn(f"Diagonal number {int(info.s[0])} is exactly zero. Singular matrix.", RuntimeWarning, stacklevel=2)
    return X.out((n,)) if vec else X.out()


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
    k = _Kit(_backend.default_mode())
    sg = k.absmax_signs(U, True)
    return k.ew("mul", U, sg), k.ew("mul", Vt, sg.T)


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
    A = _rsvd_direct_input(k, M, transpose) if k.w4_flags() & 2 else None
    U, S, Vt = _rsvd_core(k, A if A is not None else _M.from_input(M, "M"), n_components, n_oversamples, n_iter,
                          power_iteration_normalizer, transpose, flip_sign, random_state)
    kc = n_components
    return U.out(), S.out((kc,)), Vt.out()


def _rsvd_direct_input(k, M, transpose):
    """lane/apple-fast-w4-decomp RSVD_FAST_DIRECT_IN (FAST + Apple default;
    -D MOJOLEARN_RSVD_FAST_DIRECT_IN_OFF rolls back): M up from its own buffer into a pooled device matrix, as
    KernelPCA's resident route and the random projections' transform do,
    instead of `_M.from_input`'s copy into a fresh host store that the first
    product uploads (880 MB of fresh host pages at the board's 1M x 220).
    The same refusals in the same order (2-D nonempty, then the host
    finiteness scan); the same words reach the device. None (the caller
    takes `_M.from_input`) for sparse input, a binding without the resident
    entries, or a wide input (`_rsvd_core` transposes it on the host)."""
    if _is_sparse(M) or not k._res():
        return None
    a = as_f32_c(M, ndim=2, name="M")[0]
    if a.ndim != 2 or min(a.shape) == 0:  # glue: smaller of two shape dims
        raise ValueError("M: a nonempty two-dimensional input is required")
    if transpose is True or (transpose == "auto" and a.shape[0] < a.shape[1]):
        return None     # `_rsvd_core` transposes on the host: main's route
    if k.s_flags() & 1:
        # RSVD_FAST_DEVSCAN (FAST + Apple default, rollback _OFF; x_decomp/s_linalg_fast.mojo):
        # upload, then the finiteness scan on the device copy (the same refusal)
        A = _M._on_device(_DevBuf(k._raw(), a.size), a.shape[0], a.shape[1])
        if int(k.b.x_decomp_dev_upload_scan(A._d.id, addr_ro(a, name="M"), a.size)) >= 0:
            raise ValueError("M: input must be finite; NaN/inf are unsupported")
        return A
    fin = _host_all_finite(a)
    if fin is None:
        return None
    if fin is False:
        raise ValueError("M: input must be finite; NaN/inf are unsupported")
    A = _M._on_device(_DevBuf(k._raw(), a.size), a.shape[0], a.shape[1])
    k.b.x_decomp_dev_upload(A._d.id, addr_ro(a, name="M"), a.size)
    return A


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
    for _ in range(int(n_iter)):  # glue: drives the fixed randomized-SVD power iterations on the device (n_iter-sized: power iterations)
        Q = _orthonormal_cols(k, k.mm(A, Q))
        Q = _orthonormal_cols(k, k.mm(A, Q, ta=True))
    Q = _orthonormal_cols(k, k.mm(A, Q))
    # The orth rank guard (DEVIATION 5318) leaves a numerically dependent
    # column of the range exactly 0. Q^T M then has a zero row, and the
    # one-sided Jacobi SVD of a rank-deficient matrix rotates the rounding
    # noise of the column it drives to zero forever, so the small SVD runs on
    # the live columns only; the components past the numerical rank are 0
    # (singular value 0, zero vectors), as the zero Q column already says.
    # lane cpu2-l8-decomp: the live count is one word and the live columns
    # are compacted where Q lives (the stable order of the 0 (live) / 1
    # (dead) keys puts them first, in column order)
    cs = k.colsum(k.ew("abs", Q))
    nlive = k.count_gt(k.ew("abs", cs), 0.0)
    if not nlive:
        raise ValueError("randomized_svd: M is numerically zero")
    full = Q.c
    if nlive < full:
        dead = k.ew("adds", k.ew("scale", k.ew("gts", k.ew("abs", cs), s=0.0), s=-1.0), s=1.0)
        Q = k.take_cols_m(Q, k.order_small(dead), nlive)
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
    if min(a.shape) == 0:  # glue: smaller of two shape dims
        raise ValueError(f"{name}: a nonempty input is required")
    fin = _host_all_finite(a)
    if fin is False:
        raise ValueError(f"{name}: input must be finite; NaN/inf are unsupported")
    if fin is None:
        _M.from_input(a if a.ndim == 2 else a.reshape(a.shape[0], 1), name)
    return a


def _tsqr_lstsq_on(m, nn, nrhs):
    return nn >= 1 and nrhs >= 1 and nn + nrhs <= _TS_MAX_N and m >= nn + nrhs


def _tsqr_lstsq_core(k, a_arr, b_arr, m, nn, nrhs, rcond, equilibrate=False, Ra=None):
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
    cells (the same words on every column).

    `Ra` (lane idn-dense-linalg): R_aug already factored by the caller
    (`linear_model._ols_tsqr_centered`'s one entry); a_arr and b_arr are
    then not read."""
    from ._linalg_impl import _svd_tall
    n = nn + nrhs
    if Ra is None:
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
    cut = _f32(k.word(S, 0) * rcond)
    rank = k.count_gt(S, cut)       # lane cpu2-l8-decomp: two words, not S
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
    (`_tsqr_lstsq_core`, lane neural-pass140); any other shape the QR + one-sided Jacobi SVD of a (or
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
    cut = _f32(k.word(S, 0) * rcond) if r else 0.0
    rank = k.count_gt(S, cut)       # lane cpu2-l8-decomp: two words, not S
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
        # the first column of Y with some |y| > eps (lane cpu2-l8-decomp,
        # re-audit L8 `_PLS._power`): per-column counts on the device, one
        # word for "any", and the column found by the stable order of the
        # 0 (live) / 1 (dead) keys, gathered where Y lives (no column download)
        cnt = k.colsum(k.ew("gts", k.ew("abs", Y), s=eps))
        if k.count_gt(cnt, 0.0) == 0:
            raise StopIteration("y residual is constant")
        first = k.order_small(k.ew("adds", k.ew("scale", k.ew("gts", cnt, s=0.0), s=-1.0), s=1.0))
        y_score = k.take_cols_m(Y, first, 1)
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
            # lane cpu2-l8-decomp: one word ("is any column dead?") and the
            # 0/1 mask made on the device (live count > 0), not q flags
            if k.count_gt(live, 0.0) < q:
                Yk = k.ew("mul", Yk, k.ew("gts", live, s=0.0))
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
            # _svd_flip_1d: the largest-|.| entry of x_weights (the first on
            # a tie) positive; its sign as a 1 x 1 device value (lane
            # cpu2-l8-decomp: absmax_sign_cell over xw's one column)
            sg = k.absmax_signs(xw, True)
            xw, yw = k.ew("mul", xw, sg), k.ew("mul", yw, sg)
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
        self.x_weights_m_ = k.hstack(xw_c)
        self.y_weights_m_ = k.hstack(yw_c)
        self.x_loadings_m_ = k.hstack(xl_c)
        self.y_loadings_m_ = k.hstack(yl_c)
        self._x_scores_m, self._y_scores_m = k.hstack(xs_c), k.hstack(ys_c)
        self.x_rotations_m_ = k.mm(self.x_weights_m_, _pinv(k, k.mm(self.x_loadings_m_, self.x_weights_m_, ta=True)))
        self.y_rotations_m_ = k.mm(self.y_weights_m_, _pinv(k, k.mm(self.y_loadings_m_, self.y_weights_m_, ta=True)))
        coef = k.mm(self.x_rotations_m_, self.y_loadings_m_, tb=True)          # p x q
        coef = k.ew("div", k.ew("mul", coef, self._y_std), self._x_std.T).T     # q x p
        self.coef_m_ = coef
        for name in ("x_weights", "y_weights", "x_loadings", "y_loadings", "x_rotations", "y_rotations"):  # glue: loops over six attribute names
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
    # lane/apple-fast-gap-clus3: the atom loop in one fused device entry when
    # the build exports it (FAST on every vendor since lane cpu2-l8-decomp;
    # off: -D MOJOLEARN_DECOMP_FAST_DICT_DEV_OFF, x_decomp/dict_fast.mojo); an
    # unused atom (the Philox resample) or positive_dict keep the loop below
    nc = D.r
    dg = k.diag(A)                       # A's diagonal, 1 x nc, where A lives
    used = k.count_gt(dg, 1e-6)          # one word: every atom used?
    fn = k._dict_dev() if not positive and D.r * D.c else None
    if fn is not None and used == nc:
        Dn = k._dout(D.r, D.c)
        fn(k._did(D), k._did(A), k._did(B), Dn._d.id, [D.r, D.c])
        return Dn, code
    # lane cpu2-l8-decomp (re-audit L8, `_update_dict`): the loop keeps the
    # dictionary as ONE matrix where the kit keeps it (on the device for a
    # GPU kit) and writes atom j's new row into it in place: the same values
    # the per-atom host stack of the rows gave (rows before j updated), with
    # no download of an atom, of A or of B. A[j, j] is a 1 x 1 operand of the
    # division (the same cell as the old host constant); the per-atom
    # used/unused decision is one word, read only when some atom is unused.
    Dm = k.copy(D)
    zero_cols = []
    for j in range(nc):  # glue: sklearn's atom order, each atom a chain of kit cells on the resident D
        ajj = dg.cols(j, j + 1)
        if used == nc or k.word(ajj) > 1e-6:
            upd = k.ew("sub", B.cols(j, j + 1).T, k.mm(A.rows(j, j + 1), Dm))
            row = k.ew("add", Dm.rows(j, j + 1), k.ew("div", upd, ajj))
        else:
            c = counter[0] if counter is not None else 0
            row = _resample_atom(k, Y, seed, c)
            if counter is not None:
                counter[0] += 1
            zero_cols.append(j)
        if positive:
            row = k.ew("maxs", row, s=0.0)
        nrm = k.ew("sqrt", k.total(k.ew("sq", row)))
        row = k.ew("div", row, k.ew("maxs", nrm, s=1.0))
        k.place_rows(Dm, row, j)
    if zero_cols:
        code = k.copy(code)
        for j in zero_cols:
            k.fill0(code, j, code.c, code.r)
    return Dm, code


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
        code = k.hstack([code, _M.zeros(code.r, nc - r)])
        D = k.vstack([D, _M.zeros(nc - r, D.c)])
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
            code = k.hstack([k.ew("maxs", code, s=0.0), k.ew("scale", k.ew("mins", code, s=0.0), s=-1.0)])
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
            code = k.hstack([k.ew("maxs", code, s=0.0), k.ew("scale", k.ew("mins", code, s=0.0), s=-1.0)])
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
            D = k.vstack([D, _M.zeros(nc - D.r, m)])
        if self.shuffle:
            Xt = k.take_rows(M, k.order(k.rand(1, n, seed, 50, 0)))
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
        # lane cpu2-l8-decomp: ridge_alpha added to the diagonal where G lives
        G = k.diag_add(k.mm(C, C, tb=True), _M.of([float(self.ridge_alpha)], 1, 1))
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
        k.fill0(D, 0, B.r + 1, min(A.r, B.r))
    return D


def _knn_mats(k, Q, X, n_neighbors, exclude_self, kind=0, pw=2.0):
    """`_knn_lists` as two matrices (indices as exact floats, distances),
    n x n_neighbors, selected by the `graph_knn` cell (resident on the GPU
    binding)."""
    D = k.sqdist(Q, X) if kind == 0 else _dist(k, Q, X, kind, pw, same=exclude_self)
    return k.graph_knn(D, n_neighbors, exclude_self)


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


def _lanczos_batch(run, ab, n, j, m, cap, alphas, betas):
    """Steps j .. m-1 (`run`), then their alphas and betas appended up to
    the first stop (`_lanczos_top`'s breakdown test). Returns (j, stop)."""
    run(j, m)
    for jj in range(j, m):  # glue: per-step scalars, at most _LANCZOS_MAX_M
        a, b = float(ab[jj]), float(ab[cap + jj])
        alphas.append(a)
        betas.append(b)
        if b <= 1e-30 * max(1.0, abs(a)) or jj + 1 == n:  # glue: two scalars
            return jj + 1, True
    return m, False


#: the Lanczos basis cap in floats ((cap + 1) * n): past it the exact dense solve runs
_LANCZOS_DEV_MAX_FLOATS = 1 << 26


def _lanczos_top(k, A, nc):
    """The nc largest eigenpairs of symmetric A by Lanczos with full
    reorthogonalization (classical Gram-Schmidt, twice). The basis starts at
    max(2 nc + 1, 20) vectors (ARPACK's ncv) and doubles until every wanted
    Ritz pair's residual estimate beta_m |y_m| is at most _LANCZOS_TOL times
    the largest |Ritz value|. Returns None when that has not happened by
    _LANCZOS_MAX_M vectors (the caller then runs the exact solve), so the
    route never returns a less converged answer than it promises. The start
    vector is a seeded uniform draw centred at 0 (a constant vector is
    orthogonal to a centred kernel's spectrum).

    Lane cpu2-l8-decomp (re-audit L8, `_lanczos_top`): the steps run in Mojo
    in every mode, the basis in ONE matrix that never leaves where the kit
    keeps it: on a GPU kit `x_decomp_dev_lanczos` (x_decomp/lanczos_dev.mojo,
    enqueued, the alphas and betas read once a batch), on the host column
    `x_decomp_lanczos` (x_decomp/lanczos_host.mojo, the same steps and
    words). The Python step that downloaded q and re-uploaded the basis twice
    a step is deleted, and the Ritz test reads two words (max |theta| and
    max |beta y|, reduced where they live)."""
    n = A.r
    cap = min(n, _LANCZOS_MAX_M)
    if (cap + 1) * n > _LANCZOS_DEV_MAX_FLOATS:
        return None
    q = k.ew("adds", k.rand(n, 1, 0x1A2C05, 91, 0), s=-0.5)
    q = k.ew("scale", q, s=1.0 / math.sqrt(_kdot(k, q, q)))
    ab = array.array("f", [0.0]) * (2 * cap)
    if k._res():
        Qd = k._dout(cap + 1, n)
        k.place_rows(Qd, q.reshape(1, n), 0)
        ABd = k._dout(1, 2 * cap)
        aid = k._did(A)

        def run(j0, j1):
            k.b.x_decomp_dev_lanczos(aid, Qd._d.id, ABd._d.id, [n, j0, j1, cap])
            k.b.x_decomp_dev_download(ABd._d.id, ab.buffer_info()[0], 2 * cap)
    else:
        Qd = _M.zeros(cap + 1, n)
        k.place_rows(Qd, q.reshape(1, n), 0)

        def run(j0, j1):
            k.b.x_decomp_lanczos(A.addr, Qd.addr, ab.buffer_info()[0], [n, j0, j1, cap])
    alphas, betas = [], []
    m = min(n, max(2 * nc + 1, 20))
    j = 0
    stop = False
    while True:
        if j < m and not stop:
            j, stop = _lanczos_batch(run, ab, n, j, m, cap, alphas, betas)
        # the tridiagonal T from the per-step scalars the batch read (m-sized
        # scalar control, at most _LANCZOS_MAX_M squared words)
        T = [0.0] * (j * j)
        for i in range(j):  # glue: the batch's own alphas and betas into T
            T[i * j + i] = alphas[i]
            if i + 1 < j:
                T[i * j + i + 1] = T[(i + 1) * j + i] = betas[i]
        th, Y = k.eigh(_M.of(T, j, j))
        top = list(range(j - 1, max(j - 1 - nc, -1), -1))
        if len(top) < nc:
            return None
        big = k.word(k.reduce(th.take_cols(top), k._SEL_MAXABS)) or 1.0
        res = k.word(k.reduce(k.ew("scale", Y.rows(j - 1, j).take_cols(top), s=betas[j - 1]), k._SEL_MAXABS))
        if stop or res <= _LANCZOS_TOL * big:
            break
        if m >= cap:
            return None
        m = min(n, 2 * m, _LANCZOS_MAX_M)
    Yt = Y.take_cols(top)
    QM = _M._on_device(Qd._d, j, n) if Qd._d is not None else _M(Qd.s[:j * n], j, n)
    V = k.mm(QM, Yt, ta=True)
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
    words agree across vendors."""
    n = A.r
    got = None
    if topk and n > _LANCZOS_MIN_N and nc < _LANCZOS_MAX_NC:
        got = _lanczos_top(k, A, nc)
    if got is not None:
        w, V = got
    else:
        w, V = k.eigh(A)
        order = list(range(n - 1, n - 1 - nc, -1))
        w, V = w.take_cols(order), V.take_cols(order)
    return w, k.ew("mul", V, k.absmax_signs(V, True))


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
            G = None
            # the neighbour matrices as they come (indices as exact
            # floats), each neighbour's rows of D gathered on the device
            im, sq = _knn_mats(k, Q, self._fit_X, self._knn, False, self._kind, self._pw)
            if self._kind == 0:
                sq = k.ew("sqrt", sq)
            for a in range(self._knn):  # glue: one device gather-and-min step per neighbor rank (_knn-sized: neighbor ranks)
                rows = k.take_rows(D, im, m=Q.r, ist=self._knn, ioff=a)
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
        # a difference of two nearly equal sums: accumulated in binary64
        # (exact float32 squares, `_dsum_sq`'s chunked adds) or it cancels
        Kc, ev = self._Kc, self.eigenvalues_m_
        t = _dsum_sq(k, Kc) - _dsum_sq(k, ev)
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
        """Non-metric SMACOF's disparity function (lane cpu2-l8-decomp,
        re-audit L8 `MDS._nm_native`): one Mojo entry a fit for the setup and
        one an iteration for the disparities (x_decomp/mds_iso.mojo, the
        device form x_decomp/mds_iso_dev.mojo, the same words), the pairs
        and the isotonic fit never leaving where the kit keeps them. The old
        bookkeeping (host-address triu/gather/sort helpers and x_linear's
        isotonic fit and predict on host buffers, every iteration) is
        deleted. Returns disparities(d, first) -> the symmetric n x n P,
        normalized to sum of squares n (n - 1) / 2 over the upper triangle."""
        if k._res():
            N2 = 1 << max(0, (n * n - 1).bit_length())
            keys, idx, gid = k._dout(1, N2), k._dout(1, N2), k._dout(1, N2)
            tmp, gst, word = k._dout(1, N2), k._dout(1, N2 + 1), k._dout(1, 1)
            m, G = k.b.x_decomp_dev_mds_setup(
                k._did(Dis), [keys._d.id, idx._d.id, gid._d.id, tmp._d.id, gst._d.id, word._d.id], [n, N2])
            del tmp, word
            m, G = int(m), int(G)
            # sm, wt, end, prv, last, hf, hd0, hd1, gv: G values each
            work = [k._dout(1, max(G, 1)) for _ in range(9)]  # glue: nine scratch matrices
            hold = (keys, idx, gid, gst, work)
            ids = [keys._d.id, idx._d.id, gid._d.id, gst._d.id] + [w._d.id for w in work]  # glue: the scratch buffer ids

            def run(d, first):
                _ = hold                # the buffers live as long as this function
                P = k._dout(n, n)
                k.b.x_decomp_dev_mds_disp(P._d.id if first else k._did(d), P._d.id, ids,
                                          [n, m, G, 1 if first else 0])
                return P
        else:
            cap = max(n * (n - 1) // 2, 1)
            keys = array.array("f", [0.0]) * cap
            idx, gid = array.array("i", [0]) * cap, array.array("i", [0]) * cap
            gst = array.array("i", [0]) * (cap + 1)
            m, G = k.b.x_decomp_mds_setup(Dis.addr, keys.buffer_info()[0], idx.buffer_info()[0],
                                          gid.buffer_info()[0], gst.buffer_info()[0], [n])
            m, G = int(m), int(G)
            gw = max(G, 1)
            ints = [array.array("i", [0]) * gw for _ in range(5)]  # glue: five int32 scratch buffers
            work = [array.array("f", [0.0]) * gw] + ints + [array.array("f", [0.0]) * gw]
            addrs = [a.buffer_info()[0] for a in [keys, idx, gid, gst] + work]  # glue: the scratch buffer addresses

            def run(d, first):
                _ = (keys, idx, gid, gst, work)     # alive as long as this function
                P = _M.zeros(n, n)
                k.b.x_decomp_mds_disp(P.addr if first else d.addr, P.addr, addrs, [n, m, G, 1 if first else 0])
                return P

        def disparities(d, first):
            P = run(d, first)
            ss = k.total(k.ew("sq", P)).s[0]
            P = k.ew("scale", P, s=math.sqrt((n * (n - 1) / 2) / ss))
            return k.ew("add", P, P.T)          # the mirror: u + 0 above, 0 + u below

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
            starts = [k.rand(n, nc, seed, 70 + r, 0) for r in range(int(self.n_init))]  # glue: draws one device start per restart (n_init-sized: random restarts)
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
#: lane/apple-fast-gap-manprep: `_lle_smallest` builds F0 in cells (FAST +
#: Apple A/B gmp-lle-devf0-*: taxi 3,827 -> 2,570 ms, istella 3,895 ->
#: 2,653 ms, trustworthiness the same); every mode and vendor since lane
#: cpu2-l8-decomp (the env switch and the host route are gone)
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
#: LLE_FAST_NULL_CANON (x_decomp/w4_fast.mojo): a Ritz value at most
#: _LLE_CANON_NULL floors is in the null space N; one in (_LLE_CANON_NULL,
#: _LLE_CANON_GAP] floors means N has no clear edge (no canonical answer);
#: when every column is null, p widens 4x up to _LLE_CANON_MAX_P columns.
_LLE_CANON_NULL = 4.0
_LLE_CANON_GAP = 64.0
_LLE_CANON_MAX_P = 160


def _lle_orth(k, Z):
    """Z's columns orthonormalized by the Householder QR (geqrf, orgqr):
    unconditionally orthonormal, never a zeroed column (the kit's orth zeroes
    a column it judges dependent, DEVIATION 5318, and subspace iteration's
    columns lean together). Each column is first scaled by its 1-norm (the
    shift-invert operator reaches 1 / sigma^2, 1e18 on a null space at
    float32 resolution, whose squares overflow)."""
    # lane cpu2-l8-decomp: the scale vector made where Z lives: 1 / norm
    # (a zero norm gives 0, an infinite one 0), and 1 where that is not > 0
    rc = k.ew("recip", k.colsum(k.ew("abs", Z)))
    sc = k.ew("select", rc, rc, _M.of([1.0], 1, 1), s=0.0)
    h, tau = k.geqrf(k.ew("mul", Z, sc))
    return k.orgqr(h, tau, Z.c)


def _lle_smallest(k, F, nc, max_iter, seed=0, Xd=None):
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
    # |h|^2 in closed form (n - 1 entries rn, one rn - 1): a scalar, no fold over n
    coef = 2.0 / ((n - 1) * (rn * rn) + (rn - 1.0) * (rn - 1.0))
    h = _M.of(hv, n, 1)
    un = _M.of([rn] * n, n, 1)
    hrow = _M.of([rn] * n1, 1, n1)
    Fh = k.mm(F, h)
    # lane fam-decomp / apple-fast-gap-manprep: F0 in cells; lane
    # cpu2-l8-decomp: in every mode on every column (device and host binding
    # alike: one arithmetic), no longer behind the IDENTICAL binding bit or
    # FAST's Metal-only switch (the F.cols + host stack route is gone)
    dev_f0 = True
    if dev_f0:
        # lane/apple-fast-gap-manprep (2026-10-03), FAST + Apple default: F0 = [F^ | u] built on the device in three cells
        # instead of F.cols (F downloaded, then one strided Python slice per
        # column, 10,000 at the board's n) and _hstack (F^ downloaded and
        # moved on the host, then uploaded again for the LU). The same words:
        # the rank-one product has one term per entry, the mask multiply is
        # exact (x * 1 + 0, 0 * x + rn in one fused rounding).
        hfull = _M.of([rn] * n1 + [0.0], 1, n)
        mask = _M.of([1.0] * n1 + [0.0], 1, n)
        last = _M.of([0.0] * n1 + [rn], 1, n)
        D = k.ew("sub", F, k.ew("scale", k.mm(Fh, hfull), s=coef))
        F0 = k.ew("fma", D, mask, last)
        Fhat = None
    else:
        Fhat = k.ew("sub", F.cols(0, n1), k.ew("scale", k.mm(Fh, hrow), s=coef))
    Fu = k.mm(F, un)
    g = math.sqrt(_dsum_sq(k, Fu))
    rms = math.sqrt(max(float(k.total(k.ew("sq", F)).s[0]), 0.0) / n)
    if not (g <= _LLE_NULL_GUARD * rms):
        return None
    floor = _LLE_NULL_FLOOR * _F32_EPS * rms
    if not dev_f0:
        F0 = k.hstack([Fhat, un])
    if dev_f0 and k.w4_flags() & 1:
        # lane/apple-fast-w4-decomp LLE_FAST_DEV_LU (-D MOJOLEARN_LLE_FAST_DEV_LU,
        # FAST + Apple): F0 factored where it lives; the same launches as the
        # two calls below, without F0 / the factor crossing to the host 5 times
        lu, pm, im, st = k.lu_dev_aux(F0, clamp=True)
    else:
        lu, piv, _ = k.lu(F0)
        st, _, pm, im = k.lu_aux(lu, piv, clamp=True)
    # a pivot under float32 resolution (an exactly zero one skipped its
    # step) is set to eps times the largest: inverse iteration's usual
    # perturbation (LAPACK's stein/hsein); the factor is only the spectral
    # transform, the Rayleigh-Ritz step below uses F^ itself. The floor and
    # the swaps' row order (and its inverse) are cells (`lu_aux`), no host
    # loop over the rows.
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
    zn = math.sqrt(_dsum_sq(k, z))
    if not zn > 0.0 or not math.isfinite(zn):
        return None
    z = k.ew("scale", z, s=1.0 / zn)

    # lane/apple-fast-s-shap LLE_FAST_NULL_CANON (-D MOJOLEARN_LLE_FAST_NULL_CANON,
    # FAST + Apple, x_decomp/w4_fast.mojo): a null space wider than nc gets a
    # canonical answer (a function of N and Xd) instead of the one the LU's
    # last-bit rounding picks inside N
    canon = Xd is not None and bool(k.w4_flags() & 4)

    def back(Yc):           # back to R^n: H [Yc; 0] = [Yc; 0] - coef h (h^T [Yc; 0])
        t = k.mm(hrow, Yc)
        full = k.pad_zero_row(Yc)
        return k.ew("sub", full, k.ew("scale", k.mm(h, t), s=coef))

    while True:
        X, Y, S, null = _lle_iterate(k, F0, Fhat, dev_f0, solve, solve_t, z, n, n1, nc, p, max_iter, seed, floor)
        if not (canon and null):
            break
        # N = the Ritz vectors whose values are numerically zero (under
        # _LLE_CANON_NULL floors). A value in the band up to _LLE_CANON_GAP
        # floors leaves N's edge unclear: the iteration's answer stays.
        sl = [float(v) for v in S.s]  # glue: the p Ritz values
        if any(_LLE_CANON_NULL * floor < v <= _LLE_CANON_GAP * floor for v in sl):  # glue: p <= 160 scalars
            break
        idx = [j for j in range(p) if sl[j] <= _LLE_CANON_NULL * floor]  # glue: p <= 160 column numbers
        if len(idx) == p and p < min(n1, _LLE_CANON_MAX_P):
            p = min(n1, 4 * p, _LLE_CANON_MAX_P)    # every column null: N may be wider, iterate wider
            continue
        if nc < len(idx) < p:
            # all of N (c orthonormal columns): its nc directions along which
            # the data varies most, the top left singular vectors of
            # V_N^T Xd (c x d), from the c x c Gram's SVD (descending)
            Z = X.take_cols(idx)
            C = k.mm(back(Z), Xd, ta=True)
            _, Ut = k.svd(k.mm(C, C, tb=True))
            Y = k.mm(Z, Ut.rows(0, nc), tb=True)
        break
    sv = S.take_cols(list(range(p - 1, p - 1 - nc, -1)))
    V = back(Y)
    # the columns are unit vectors or the solve is not an answer (an
    # overflow, a dropped launch): refuse rather than return them
    # the unit-norm and finiteness guard as three words (lane cpu2-l8-decomp:
    # min and max of the squared norms, max |sv|, reduced where they live)
    sq = k.colsum(k.ew("sq", V))
    lo, hi = k.word(k.reduce(sq, k._SEL_MIN)), k.word(k.reduce(sq, k._SEL_MAX))
    if not (0.9 <= lo and hi <= 1.1) or not math.isfinite(k.word(k.reduce(sv, k._SEL_MAXABS))):
        raise RuntimeError(
            "LocallyLinearEmbedding: the shift-invert subspace iteration returned columns of squared norm "
            f"{[float(v) for v in sq.s]} (not unit); pass eigen_solver='dense' for the full SVD.")
    return k.ew("mul", V, k.absmax_signs(V, True)), sv


def _lle_iterate(k, F0, Fhat, dev_f0, solve, solve_t, z, n, n1, nc, p, max_iter, seed, floor):
    """`_lle_smallest`'s subspace iteration with p columns: (X n1 x p the Ritz
    vectors by descending Ritz value, Y n1 x nc the wanted ones, S 1 x p
    their values, whether it stopped on the null floor)."""
    X = _lle_orth(k, k.ew("adds", k.rand(n1, p, seed, 0x11E, 0), s=-0.5))
    want = list(range(p - 1, p - 1 - nc, -1))
    prev, e_prev = None, float("inf")
    for it in range(max(1, int(max_iter))):
        # (F^T F^)^-1 X in two orthonormalized halves: F^+T X = P_z F0^-T
        # [X; 0] (range(F^), n x p), then F^+ of that = the first n - 1 rows
        # of F0^-1; each half stretches the block by 1 / sigma, not
        # 1 / sigma^2, so the columns stay far from float32 dependence
        Y = solve_t(k.pad_zero_row(X))      # [X; 0] (lane fix-d1-decomp: on the device, IDN_LLE_PAD_DEV)
        Y = _lle_orth(k, k.ew("sub", Y, k.mm(z, k.mm(z, Y, ta=True))))
        X = _lle_orth(k, solve(Y).rows(0, n1))
        if dev_f0:          # F^ X = F0 [X; 0] (the last column of F0 meets a zero row)
            S, Vt = k.svd(k.mm(F0, k.pad_zero_row(X)))
        else:
            S, Vt = k.svd(k.mm(Fhat, X))
        X = k.mm(X, Vt, tb=True)
        Y = X.take_cols(want)
        if it >= 2 and k.word(k.reduce(S.take_cols(want), k._SEL_MAX)) <= floor:
            return X, Y, S, True
        if prev is not None:
            E = k.ew("sub", Y, k.mm(prev, k.mm(prev, Y, ta=True)))
            e = math.sqrt(max(float(k.total(k.ew("sq", E)).s[0]), 0.0))
            if e <= _LLE_SUBSPACE_TOL or (e <= _LLE_STALL_TOL and e >= e_prev):
                return X, Y, S, False
            e_prev = e
        prev = Y
    else:
        raise RuntimeError(
            f"LocallyLinearEmbedding: the shift-invert subspace iteration did not settle in {max_iter} "
            f"iterations (last subspace change {e_prev:.3g}); pass eigen_solver='dense' for the full SVD. "
            "An unconverged embedding is not returned as if it were one.")


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
            got = _lle_smallest(k, IW, nc, int(self.max_iter), _seed_of(self.random_state), M)
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
        self.reconstruction_error_ = _dsum_sq(k, sv)
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
    # lane cpu2-l8-decomp: max |w| is one word; the cutoff mask is a select
    # cell where w lives
    wmax = k.word(k.reduce(w, k._SEL_MAXABS)) if w.c else 0.0
    cut = _f32(wmax * A.r * _F32_EPS)
    inv = k.ew("select", k.ew("abs", w), k.ew("recip", w), _M.zeros(1, 1), s=cut)
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
    # Experiment: only the MCD masked covariance seam; binding presence is
    # compile-gated FAST+Apple. Python schedules device buffers, no host math.
    ordered = getattr(k.b, "x_decomp_dev_mcd_cov", None)
    if ordered is not None and k._use(Xc) and Xc.r > 0 and Xc.c > 0:
        gram = k._dout(Xc.c, Xc.c)
        ordered(k._did(Xc), gram._d.id, [Xc.r, Xc.c])
    else:
        gram = k.mm(Xc, Xc, ta=True)
    return loc, k.ew("scale", gram, s=1.0 / cnt)


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

    def _mcd_1d(self, k, X, h):
        """sklearn fast_mcd's one-feature shortcut: the shortest window of h
        sorted values (every tie of the minimum width kept), the location the
        mean of their midpoints, the support the h values nearest it (ties to
        the lower index), the variance of the support."""
        return self._mcd_1d_native(k, X, h)

    def _mcd_1d_native(self, k, X, h):
        """`_mcd_1d` with its orders, gathers, window minimum and support in
        x_decomp/moves.mojo (lane apple-fast-py2mojo-decomp): the same
        (value, index) orders, the same kit cells on the same values."""
        n = X.r
        xs = k.take_rows(X, k.order(X))
        if h < n:
            diff = k.ew("sub", xs.rows(h, n), xs.rows(0, n - h))
            starts = k.argmin_all(diff)
            mids = k.ew("scale", k.ew("add", k.take_rows(xs, starts, radd=h), k.take_rows(xs, starts)), s=0.5)
            loc = k.colmean(mids)
            cen = k.ew("abs", k.ew("sub", X, loc))
            sel, mask = k.select_smallest(cen, h)
            Xs = k.take_rows(X, sel)
            # the device form's membership is a 0 / 1 float matrix (lane cpu3-python)
            support = mask.out((n,)).astype("<u1") if isinstance(mask, _M) else _support_of(mask, n)
        else:
            loc = k.colmean(X)
            Xs = X
            support = _support_of(array.array("i", [1]) * n, n)
        cov = _emp_cov(k, Xs)
        P = _pinvh(k, cov)
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
        return loc, cov, _support_of(sup, n), dist

    def _fast_mcd(self, k, X):
        n, p = X.r, X.c
        h = int(math.ceil(0.5 * (n + p + 1))) if self.support_fraction is None else int(self.support_fraction * n)
        if p == 1:
            return self._mcd_1d(k, X, h)
        return self._fast_mcd_native(k, X, h)

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
            sm = _M(_f32_store(support.astype("<f4")), n, 1)
            _, cov = _masked_cov(k, M, sm, True)
            dist = k.rowsum(k.ew("mul", k.mm(M, _pinvh(k, cov)), M))
        self.raw_location_m_, self.raw_covariance_m_ = loc, cov
        self.raw_location_ = loc.out((p,))
        self.raw_covariance_ = cov.out()
        self.raw_support_ = support
        # correct_covariance: consistency at the normal model (the corrected
        # matrix is returned by sklearn and not kept; dist_ is rescaled)
        n_support = int(support.sum())  # glue: Array.sum is the Mojo reduce helper
        corr = _consistency_factor(k, p, n_support / n)
        dist = k.ew("scale", dist, s=1.0 / corr)
        # reweight_covariance
        thr = _chi2_quantile(k, p, 0.025)
        # the reweighting mask dist < thr on the device (`le` against the
        # float32 just below thr: the same decision), then the masked moments
        mm = k.ew("le", dist, _M.of([_f32_below(thr)], 1, 1))
        locr, covr = _masked_cov(k, M, mm, self.assume_centered)
        covr = k.ew("scale", covr, s=_consistency_factor(k, p, 0.975))
        mask = mm.out((n,)).astype("<u1")
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
        k = self._kit()
        # the two order statistics the interpolation reads: sorted(-d)[i]
        # is -(d in ascending order)[n - 1 - i] (negation is exact)
        dm = self.dist_m_
        nv = dm.r * dm.c
        q = 100.0 * self.contamination / 100.0 * (nv - 1)
        lo = int(math.floor(q))
        hi = min(lo + 1, nv - 1)
        # lane cpu3-python: the kit's order (a device radix sort on a GPU
        # binding), then the two order statistics one word each
        o = k.order(dm)
        vlo = -k.word(dm, int(k.word(o, nv - 1 - lo)))
        vhi = -k.word(dm, int(k.word(o, nv - 1 - hi)))
        self.offset_ = vlo + (vhi - vlo) * (q - lo)
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
        from ._buffer import frombytes as _fb
        k = self._kit()
        d = _mahal(k, _M.from_input(X), self.location_m_, self.precision_m_)
        dec = k.ew("adds", k.ew("scale", d, s=-1.0), s=-self.offset_)
        if not d.r:
            return _fb(b"", "<i4", (0,))
        # lane cpu3-python: 1 where dec >= 0 (-dec <= 0), else -1 (NaN: -1),
        # in the kit's cells on either binding
        ge = k.ew("le", k.ew("scale", dec, s=-1.0), _M.of([0.0], 1, 1))
        lab = k.ew("select", ge, _M.of([1.0], 1, 1), _M.of([-1.0], 1, 1), s=0.5)
        return lab.out((d.r,)).astype("<i4")

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)

    def score(self, X, y, sample_weight=None):
        """sklearn OutlierMixin/ClassifierMixin.score: accuracy_score(y,
        predict(X), sample_weight), the (weighted) share of exact label
        matches. Lane cpu4-python: the x_metrics device program
        (`accuracy_fraction`: exact match counts, the weight total its
        PairSum, the ratio binary64 on the device; the host column runs the
        same units) replaces the host loop `x_decomp_accuracy`."""
        from ._expansion_metrics import accuracy_fraction
        pa = self.predict(X)
        n = len(pa)
        ya = as_f64_c(y, ndim=1, name="y")[0]
        if len(ya) != n:
            raise ValueError("y and X have different numbers of rows")
        if sample_weight is not None and len(as_f64_c(sample_weight, ndim=1, name="sample_weight")[0]) != n:
            raise ValueError("sample_weight and X have different numbers of rows")
        # both sides Float64 labels: one label kind for the encoder
        try:
            return accuracy_fraction(ya, pa.astype("<f8"), sample_weight, self.numeric_mode_)
        except ZeroDivisionError:
            raise ZeroDivisionError("Weights sum to zero, can't be normalized") from None


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

    def _topn(self, k, S, N, skip):
        """(ids, scores) of the N best entries of the score row S by (-score,
        id), skipping the nonzero entries of the float32 row at address
        `skip` (0: none): x_decomp/moves.mojo `topn_desc` and `gather`."""
        from ._buffer import frombytes as _fb
        m = S.r * S.c
        n = max(min(N, m), 0)
        if n and k._use(S):
            # lane cpu3-python: the (-score, id) order on the device, the
            # skipped items last; the N first ids and their scores gathered there
            sk = None
            if skip:
                sk = _M.zeros(1, m)
                ctypes.memmove(sk.addr, skip, 4 * m)   # glue: the caller's skip row, one C copy
            o, cnt = k._order_dev(S, neg=1, skip=sk)
            c = min(n, int(k.word(cnt)))
            if not c:
                return (_fb(b"", "<i4", (0,)), _fb(b"", "<f4", (0,)))
            ids = o.rows(0, c)
            vals = k.take_cols_m(S.reshape(1, m), ids, c)
            return (ids.out((c,)).astype("<i4"), vals.out((c,)))
        order = array.array("i", [0]) * max(n, 1)
        c = int(k.b.x_decomp_topn_desc(S.addr, [m, n], skip, order.buffer_info()[0])) if n else 0  # cpu-route: host binding only (CPU-only installs); a GPU binding orders on the device above
        vals = array.array("f", [0.0]) * max(c, 1)
        if c:
            k.b.x_decomp_gather(S.addr, order.buffer_info()[0], c, vals.buffer_info()[0])  # cpu-route: host binding only, the device route returned above
        return (_fb(order[:c].tobytes(), "<i4", (c,)), _fb(vals[:c].tobytes(), "<f4", (c,)))

    def recommend(self, userid, user_items, N=10, filter_already_liked_items=True):
        """(ids, scores) of the N best items for one user: x_u . y_i, ties to
        the lower item id; items the user interacted with (row `userid` of
        user_items, or user_items itself when it is one row) are skipped."""
        self._check_fit()
        k = self._kit()
        U = self.user_factors_m_.rows(userid, userid + 1)
        S = k.mm(U, self.item_factors_m_, tb=True)
        if int(N) < 0:
            raise ValueError(f"recommend: N must be non-negative, got {N}")
        skip = 0
        if filter_already_liked_items and user_items is not None:
            R = _M.from_input(user_items, "user_items")
            if R.c != S.c or not (R.r == 1 or 0 <= userid < R.r):
                raise ValueError(f"recommend: user_items of shape ({R.r}, {R.c}) has no row {userid} "
                                 f"over the {S.c} items")
            skip = R.addr + 4 * (0 if R.r == 1 else int(userid)) * R.c
        return self._topn(k, S, int(N), skip)

    def similar_items(self, itemid, N=10):
        """(ids, scores) of the N items whose factors have the largest cosine
        with item `itemid` (itself included, as implicit returns it)."""
        self._check_fit()
        k = self._kit()
        Y = self.item_factors_m_
        nrm = k.ew("sqrt", k.rowsum(k.ew("sq", Y)))
        Yn = k.ew("div", Y, nrm)
        S = k.mm(Yn.rows(itemid, itemid + 1), Yn, tb=True)
        if int(N) < 0:
            raise ValueError(f"similar_items: N must be non-negative, got {N}")
        return self._topn(k, S, int(N), 0)


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
    sg = k.absmax_signs(Vm, False)
    Vm, Um = k.ew("mul", Vm, sg), k.ew("mul", Um, sg.T)
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

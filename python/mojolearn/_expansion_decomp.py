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

from . import _backend, _native_family_bridge
from . import _portable_math as _pm
from ._array import Array
from ._buffer import addr_ro, as_f32_c, as_f64_c, as_i32_c, frombytes
from ._buffer import addr as _addr, empty as _empty
from ._mode import SklearnTagsMixin, available_if
from ._mode import RegressorScoreMixin


def _min_samples(n, d, who, minimum=2):
    """scikit-learn's check_array(ensure_min_samples=...) refusal and wording."""
    if n < minimum:
        raise ValueError(f"Found array with {n} sample(s) (shape=({n}, {d})) while a minimum of "
                         f"{minimum} is required by {who}.")

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
            k.b.x_decomp_move(self.addr, I.addr, out.addr,  # cpu-route: host matrix (_M buffer) read once by x_decomp_move on a CPU-only install
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
        k.b.x_decomp_move(self.addr, _M._one.addr, out.addr, [_MV_TRANSPOSE, n, self.r, self.c, 0, 0, 0, n, n, 1])  # cpu-route: host matrix (_M buffer) read once by x_decomp_move on a CPU-only install
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
            k.b.x_decomp_move(m.addr, _M._one.addr, out.addr,  # cpu-route: host matrix (_M buffer) read once by x_decomp_move on a CPU-only install
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

    def enable_lle_precision(self):
        """LLE's near-null solve needs ordered products and accurate LU updates.

        The optional entries exist only in Apple FAST bindings; all arithmetic
        remains on the GPU. Older bindings and other modes keep their entries.
        """
        if self.mode != "fast" or _kit_vendor(self) != "metal":
            return
        try:
            mm = getattr(self.b, "x_decomp_lle_gemm_ordered")
            lu = getattr(self.b, "x_decomp_lle_lu_ordered")
        except (AttributeError, ImportError):
            return
        self._lle_mm, self._lle_lu = mm, lu

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
        RSVD_FAST_DIRECT_IN on FAST + Apple or RSVD_IDN_DIRECT_IN on
        IDENTICAL, lane fg-pca R1; 0 on a binding without the entry)."""
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
        (`x_decomp_qfix_flags`, x_decomp/qfix.mojo: bit 2 TSVD_QFIX, bit 4
        LU_QFIX (bit 1, SVD_QFIX, deleted 2026-10-09); each off with its -D MOJOLEARN_*_QOLD; 0 on
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
        precise = getattr(self, "_lle_mm", None)
        if precise is not None:
            out = _M.zeros(m, n)
            precise(A.addr, B.addr, out.addr, [m, k, n, int(ta), int(tb)])
            return out
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
        precise = getattr(self, "_lle_lu", None)
        if precise is not None:
            lu = A.copy()
            precise(lu.addr, piv.buffer_info()[0], info.addr, [n])
            return lu, piv, int(info.s[0])
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
            self.b.x_decomp_move(A.addr, idx.addr, out.addr, p)  # cpu-route: host matrix (_M buffer) read once by x_decomp_move on a CPU-only install
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
        self.b.x_decomp_move(A.addr, _M._one.addr, A.addr, p)  # cpu-route: host matrix (_M buffer) read once by x_decomp_move on a CPU-only install
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
        self.b.x_decomp_reduce(A.addr, out.addr, [n, int(op)])  # cpu-route: host matrix (_M buffer) read once by x_decomp_reduce on a CPU-only install
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
            self.b.x_decomp_move(A.addr, idx.addr, out.addr, p)  # cpu-route: host matrix (_M buffer) read once by x_decomp_move on a CPU-only install
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
        self.b.x_decomp_move(V.addr, _M._one.addr, D.addr, p)  # cpu-route: host matrix (_M buffer) read once by x_decomp_move on a CPU-only install
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
            self.b.x_decomp_order_small(A.addr, out.addr, [n])  # cpu-route: host vector (_M buffer) read once by x_decomp_order_small on a CPU-only install
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
            self.b.x_decomp_move(A.addr, idx.addr, out.addr, p)  # cpu-route: host matrix (_M buffer) read once by x_decomp_move on a CPU-only install
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
            self.b.x_decomp_order_f(A.addr, n, out.addr)  # cpu-route: host vector (_M buffer) read once by x_decomp_order_f on a CPU-only install
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
            self.b.x_decomp_select_smallest(A.addr, [n, h], sel.addr, mask.buffer_info()[0])  # cpu-route: host vector (_M buffer) read once by x_decomp_select_smallest on a CPU-only install
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
        c = int(self.b.x_decomp_argmin_all(A.addr, n, buf.addr)) if n else 0  # cpu-route: host vector (_M buffer) read once by x_decomp_argmin_all on a CPU-only install
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


class _Base(SklearnTagsMixin):
    _parameters = ()

    def get_params(self, deep=True):
        """Return this estimator's constructor parameters.

        Parameters
        ----------
        deep : bool, default True
            Accepted for scikit-learn compatibility; it has no effect.

        Returns
        -------
        dict
            Parameter name to current value.
        """
        return {name: getattr(self, name) for name in self._parameters}  # glue: estimator parameter names (get_params)

    def set_params(self, **params):
        """Set constructor parameters by keyword.

        Raises
        ------
        ValueError
            If a keyword is not a parameter of this estimator.

        Returns
        -------
        self
        """
        for k, v in params.items():  # glue: estimator keyword arguments (set_params)
            if k not in self._parameters:
                raise ValueError(f"Invalid {type(self).__name__} parameter: {k}")
            setattr(self, k, v)
        return self

    # Only a class with a `transform` has `fit_transform` (scikit-learn's
    # TransformerMixin): a classifier or detector on this base has neither.
    @available_if(lambda self: hasattr(self, "transform"))
    def fit_transform(self, X, y=None):
        """Fit to X, then return ``transform(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return self.fit(X, y).transform(X)

    def _kit(self):
        return _Kit(self.numeric_mode_)

    def _check(self, attr="components_"):
        if not hasattr(self, attr):
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit first")


# ================================================================ IncrementalPCA
# Every public IncrementalPCA operation is one call into the compiled owner
# (bindings/ipca_native.mojo, drivers x_decomp/ipca_owner.mojo).
# Reference: scikit-learn `decomposition/_incremental_pca.py`, `partial_fit`;
# the running mean and variance are `utils/extmath.py::_incremental_mean_and_var`.
# Each batch's SVD is the thin SVD of the stacked matrix sklearn builds,
# computed as the eigendecomposition of its d x d Gram matrix (the Jacobi
# eigh of decomposition/, device or host), then sklearn's `svd_flip` with
# `u_based_decision=False`.
class IncrementalPCA(_Base):
    """Incremental principal component analysis, fitted batch by batch.

    Follows scikit-learn's ``IncrementalPCA``: the mean, variance and
    components are updated one batch at a time, so a dataset larger than
    memory can be fitted through ``partial_fit``. Component signs follow
    scikit-learn's ``svd_flip`` on the rows of the components. A
    ``scipy.sparse`` input is densified one batch at a time in ``fit`` and
    whole in ``partial_fit`` and ``transform``.

    Parameters
    ----------
    n_components : int or None, default None
        Number of components to keep. None keeps ``n_features`` (or, in a
        later ``partial_fit``, the number already fitted). Must be between 1
        and ``n_features`` and at most the number of rows in each batch.
    whiten : bool, default False
        Divide the projected data by the square root of each component's
        explained variance.
    copy : bool, default True
        Accepted for scikit-learn compatibility; the input is never modified.
    batch_size : int or None, default None
        Rows per batch in ``fit``. None uses ``5 * n_features``.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes, sorted by decreasing explained variance.
    explained_variance_ : ndarray of shape (n_components,)
    explained_variance_ratio_ : ndarray of shape (n_components,)
    singular_values_ : ndarray of shape (n_components,)
    mean_ : ndarray of shape (n_features,)
    var_ : ndarray of shape (n_features,)
    noise_variance_ : float
    n_components_ : int
    n_samples_seen_ : int
    n_features_in_ : int
    batch_size_ : int

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> model = mojolearn.IncrementalPCA(n_components=3, batch_size=200).fit(X)
    >>> model.transform(X).shape
    (1000, 3)
    """
    _parameters = ("n_components", "whiten", "copy", "batch_size", "numeric_mode")

    def __init__(self, n_components=None, *, whiten=False, copy=True, batch_size=None, numeric_mode=None):
        self.n_components, self.whiten, self.copy = n_components, whiten, copy
        self.batch_size, self.numeric_mode = batch_size, numeric_mode

    def fit(self, X, y=None):
        """Fit the model on X in batches of ``batch_size`` rows.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Training data. Any earlier fit is discarded.
        y : ignored

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X is empty or ``n_components`` is out of range.
        """
        return _IPCA_CALL(self, "fit", (X,), _IPCA_CONTEXT)

    def partial_fit(self, X, y=None):
        """Update the model with one more batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The batch. ``n_features`` must match earlier batches and
            ``n_samples`` must be at least ``n_components``.
        y : ignored

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the number of features changed or ``n_components`` is out of
            range for this batch.
        """
        return _IPCA_CALL(self, "partial_fit", (X,), _IPCA_CONTEXT)

    def transform(self, X):
        """Project X onto the fitted components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        ValueError
            If X has a different number of features than the fitted data.
        """
        return _IPCA_CALL(self, "transform", (X,), _IPCA_CONTEXT)

    def fit_transform(self, X, y=None):
        """Fit to X, then return ``transform(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _IPCA_CALL(self, "fit_transform", (X,), _IPCA_CONTEXT)

    def inverse_transform(self, X):
        """Map projected data back to the original feature space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)

        Returns
        -------
        ndarray of shape (n_samples, n_features), float32
        """
        return _IPCA_CALL(self, "inverse_transform", (X,), _IPCA_CONTEXT)


# ================================================================ random projection
def johnson_lindenstrauss_min_dim(n_samples, *, eps=0.1):
    """Minimum dimension for a random projection that keeps distances within eps.

    Follows scikit-learn's ``johnson_lindenstrauss_min_dim`` for a single
    ``n_samples`` and ``eps`` (scalars only): ``4 log(n_samples) /
    (eps**2 / 2 - eps**3 / 3)``, truncated to an int.

    Parameters
    ----------
    n_samples : int
        Number of samples; must be greater than zero.
    eps : float, default 0.1
        Maximum distortion, strictly between 0 and 1.

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If ``eps`` is not in (0, 1) or ``n_samples`` is not positive.

    Examples
    --------
    >>> import mojolearn
    >>> d = mojolearn.johnson_lindenstrauss_min_dim(1000, eps=0.5)
    """
    # DEVIATION 6900: eps ** 2, eps ** 3 correctly rounded and the pinned log,
    # not the platform pow / log (a host's last bit can move the floor), in
    # the compiled owner (bindings/jl_native.mojo)
    return _JL_CALL(None, "min_dim", (n_samples, eps), _JL_CONTEXT)


def _seed_of(random_state):
    if random_state is None:
        return 0
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise TypeError("random_state must be None or an int (the counter-based RNG takes an integer seed)")
    return random_state


class _RandomProjection(_Base):
    """sklearn `random_projection.py::BaseRandomProjection`. The matrix is
    drawn from the lane's counter-based Philox stream (x_decomp/cells.mojo
    `rand_cell`), not numpy's generator: the same `random_state` gives the
    same matrix on every box, and a different one than sklearn's."""

    # Under lane/apple-fast-gap-cls2's LAZY switch the fit leaves the matrix
    # on the device and the first read downloads it once (the same words).
    @property
    def components_(self):
        """The random projection matrix, shape (n_components, n_features)."""
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
        """Draw the projection matrix for X's shape.

        Only X's shape is used (and X must be finite); the values are not.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If X is empty or non-finite, ``n_components`` is not positive, or
            ``n_components="auto"`` gives a dimension larger than
            ``n_features``.
        TypeError
            If ``random_state`` is not None or an int.
        """
        return _DECOMP_CALL(self, "fit", (X,), _DECOMP_CONTEXT)

    def transform(self, X):
        """Project X: ``X @ components_.T``.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
            Always a dense array.

        Raises
        ------
        ValueError
            If X has the wrong number of features or is non-finite.
        """
        return _DECOMP_CALL(self, "transform", (X,), _DECOMP_CONTEXT)

    def fit_transform(self, X, y=None):
        """Fit to X, then return ``transform(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DECOMP_CALL(self, "fit_transform", (X,), _DECOMP_CONTEXT)

    def inverse_transform(self, X):
        """Map projected data back with the pseudo-inverse of the components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)

        Returns
        -------
        ndarray of shape (n_samples, n_features), float32

        Raises
        ------
        ValueError
            If the model was fitted without ``compute_inverse_components=True``.
        """
        return _DECOMP_CALL(self, "inverse_transform", (X,), _DECOMP_CONTEXT)


# Box-Muller on the Philox stream.
class GaussianRandomProjection(_RandomProjection):
    """Reduce dimension with a Gaussian random matrix.

    Follows scikit-learn's ``GaussianRandomProjection``: each component entry
    is drawn from N(0, 1 / n_components).

    Parameters
    ----------
    n_components : int or "auto", default "auto"
        Target dimension. ``"auto"`` uses
        ``johnson_lindenstrauss_min_dim(n_samples, eps=eps)``.
    eps : float, default 0.1
        Distortion bound used by ``n_components="auto"`` through
        :func:`johnson_lindenstrauss_min_dim`.
    compute_inverse_components : bool, default False
        Compute ``inverse_components_`` (the pseudo-inverse of the
        components) in ``fit``, needed by ``inverse_transform``. Raises
        ValueError if the components are rank deficient.
    random_state : int or None, default None
        Seed of mojolearn's counter-based random generator; None means 0.
        The same seed gives the same matrix on every device, but not the
        matrix scikit-learn draws for that seed.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
    inverse_components_ : ndarray of shape (n_features, n_components)
        Only with ``compute_inverse_components=True``.
    n_components_ : int
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 64), dtype=np.float32)
    >>> model = mojolearn.GaussianRandomProjection(n_components=8, random_state=0).fit(X)
    >>> model.transform(X).shape
    (1000, 8)
    """
    _parameters = ("n_components", "eps", "compute_inverse_components", "random_state", "numeric_mode")

    def __init__(self, n_components="auto", *, eps=0.1, compute_inverse_components=False, random_state=None,
                 numeric_mode=None):
        self.n_components, self.eps = n_components, eps
        self.compute_inverse_components, self.random_state = compute_inverse_components, random_state
        self.numeric_mode = numeric_mode


# Achlioptas / Li et al.
class SparseRandomProjection(_RandomProjection):
    """Reduce dimension with a sparse random matrix.

    Follows scikit-learn's ``SparseRandomProjection``: each entry is
    ``+sqrt(1/density)/sqrt(n_components)`` or its negative with probability
    ``density/2`` each, and 0 otherwise. The components are stored as a dense
    array, and ``transform`` always returns a dense array.

    Parameters
    ----------
    n_components : int or "auto", default "auto"
        Target dimension. ``"auto"`` uses
        ``johnson_lindenstrauss_min_dim(n_samples, eps=eps)``.
    density : float or "auto", default "auto"
        Fraction of nonzero entries, in (0, 1]. ``"auto"`` is
        ``1 / sqrt(n_features)``. Out-of-range values raise ValueError.
    dense_output : bool, default False
        Accepted for scikit-learn compatibility; the output is always dense.
    eps : float, default 0.1
        Distortion bound used by ``n_components="auto"`` through
        :func:`johnson_lindenstrauss_min_dim`.
    compute_inverse_components : bool, default False
        Compute ``inverse_components_`` (the pseudo-inverse of the
        components) in ``fit``, needed by ``inverse_transform``. Raises
        ValueError if the components are rank deficient.
    random_state : int or None, default None
        Seed of mojolearn's counter-based random generator; None means 0.
        The same seed gives the same matrix on every device, but not the
        matrix scikit-learn draws for that seed.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
    inverse_components_ : ndarray of shape (n_features, n_components)
        Only with ``compute_inverse_components=True``.
    density_ : float
    n_components_ : int
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 64), dtype=np.float32)
    >>> model = mojolearn.SparseRandomProjection(n_components=8, random_state=0).fit(X)
    >>> model.transform(X).shape
    (1000, 8)
    """
    _parameters = ("n_components", "density", "eps", "dense_output", "compute_inverse_components",
                   "random_state", "numeric_mode")

    def __init__(self, n_components="auto", *, density="auto", eps=0.1, dense_output=False,
                 compute_inverse_components=False, random_state=None, numeric_mode=None):
        self.n_components, self.density, self.eps, self.dense_output = n_components, density, eps, dense_output
        self.compute_inverse_components, self.random_state = compute_inverse_components, random_state
        self.numeric_mode = numeric_mode


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


# Reference: scikit-learn `decomposition/_nmf.py`: `_initialize_nmf`,
# `_fit_multiplicative_update` with `_multiplicative_update_w/_h` for
# beta_loss='frobenius', and `_fit_coordinate_descent` with
# `_cdnmf_fast.pyx::_update_cdnmf_fast`. The NNDSVD SVD is exact (through
# the Gram eigh) where sklearn calls randomized_svd. 'random' init and
# shuffle use the lane's Philox stream, not numpy's. A general float beta
# is refused by name.
class NMF(_Base):
    """Non-negative matrix factorization ``X ~ W @ H``.

    Follows scikit-learn's ``NMF``: finds non-negative ``W`` (n_samples,
    n_components) and ``H`` (n_components, n_features) minimizing the chosen
    beta divergence plus optional L1/L2 penalties. The NNDSVD initializations
    use an exact SVD rather than scikit-learn's randomized one, and random
    draws come from mojolearn's own generator, so results differ slightly
    from scikit-learn's for the same seed.

    Parameters
    ----------
    n_components : int, "auto" or None, default "auto"
        Number of components. ``"auto"`` or None uses ``n_features`` (with
        ``init="custom"``, the number of rows of the given ``H``).
    init : {"random", "nndsvd", "nndsvda", "nndsvdar", "custom"} or None, default None
        Initialization. None uses ``"nndsvda"`` when ``n_components <=
        min(n_samples, n_features)``, else ``"random"``. The NNDSVD variants
        need ``n_components <= min(n_samples, n_features)``. ``"custom"``
        takes ``W`` and ``H`` from ``fit_transform``.
    solver : {"cd", "mu"}, default "cd"
        Coordinate descent or multiplicative update.
    beta_loss : {"frobenius", "kullback-leibler", "itakura-saito"}, default "frobenius"
        Divergence to minimize; the last two need ``solver="mu"``. The
        numbers 2, 1 and 0 are accepted for the same three; any other float
        beta raises ValueError.
    tol : float, default 1e-4
        Stopping tolerance.
    max_iter : int, default 200
        Maximum number of iterations.
    random_state : int or None, default None
        Seed for ``"random"`` and ``"nndsvdar"`` initialization and for
        ``shuffle``; None means 0. Not scikit-learn's random stream.
    alpha_W : float, default 0.0
        Regularization strength on ``W``.
    alpha_H : float or "same", default "same"
        Regularization strength on ``H``; ``"same"`` uses ``alpha_W``.
    l1_ratio : float, default 0.0
        Mix of L1 (1.0) and L2 (0.0) regularization.
    verbose : int, default 0
        Accepted for scikit-learn compatibility; nothing is printed.
    shuffle : bool, default False
        Visit coordinates in random order in each ``"cd"`` sweep.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The factor ``H``.
    n_components_ : int
    n_iter_ : int
    reconstruction_err_ : float
        Beta divergence between X and ``W @ H`` (square-rooted, as in
        scikit-learn).
    n_features_in_ : int

    Raises
    ------
    ValueError
        If X has negative values, or an unsupported solver, init or
        beta_loss is requested.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> model = mojolearn.NMF(n_components=3, random_state=0)
    >>> W = model.fit_transform(X)
    >>> H = model.components_
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _sklearn_input_tags = {"positive_only": True}
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
    # ---- solvers
    def fit_transform(self, X, y=None, W=None, H=None):
        """Fit the model and return ``W``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Non-negative data.
        y : ignored
        W : array-like of shape (n_samples, n_components), optional
            Starting ``W``; used only with ``init="custom"``.
        H : array-like of shape (n_components, n_features), optional
            Starting ``H``; used only with ``init="custom"``.

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DECOMP_CALL(self, "fit_transform", (X, W, H), _DECOMP_CONTEXT)

    def fit(self, X, y=None, **params):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Non-negative data.
        y : ignored
        **params
            ``W`` and ``H`` for ``init="custom"``, as in ``fit_transform``.

        Returns
        -------
        self
        """
        return _DECOMP_CALL(self, "fit", (X, params), _DECOMP_CONTEXT)

    def transform(self, X):
        """Solve for ``W`` with the fitted ``H`` held fixed.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Non-negative data.

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DECOMP_CALL(self, "transform", (X,), _DECOMP_CONTEXT)

    def inverse_transform(self, X):
        """Return ``X @ components_``, the data reconstructed from ``W``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            A ``W`` matrix.

        Returns
        -------
        ndarray of shape (n_samples, n_features), float32
        """
        return _DECOMP_CALL(self, "inverse_transform", (X,), _DECOMP_CONTEXT)


# ================================================================ FastICA
# Reference: scikit-learn `decomposition/_fastica.py`: `_fit_transform`,
# `_ica_par`, `_ica_def`, `_sym_decorrelation`, `_gs_decorrelation`,
# `_logcosh`, `_exp`, `_cube`. The whitening SVD is the eigendecomposition
# of X^T X (both of sklearn's whiten_solver values take this route); a
# singular value below 10 float32 eps is clamped there, as sklearn's 'eigh'
# solver clamps it. w_init defaults to a draw from the lane's Philox
# stream, not numpy's. A callable `fun` is refused by name.
class FastICA(_Base):
    """Independent component analysis with the FastICA algorithm.

    Follows scikit-learn's ``FastICA``. Whitening always uses the
    eigendecomposition of ``X.T @ X`` (``whiten_solver`` is accepted but both
    values take this route), and the default starting matrix is drawn from
    mojolearn's own generator, so results can differ from scikit-learn's for
    the same seed.

    Parameters
    ----------
    n_components : int or None, default None
        Number of components; None, or a value above ``min(n_samples,
        n_features)``, uses ``min(n_samples, n_features)``. Ignored when
        ``whiten=False``.
    algorithm : {"parallel", "deflation"}, default "parallel"
        Estimate all components together, or one at a time.
    whiten : {"unit-variance", "arbitrary-variance"} or False, default "unit-variance"
        Whitening strategy; False assumes X is already whitened.
    fun : {"logcosh", "exp", "cube"}, default "logcosh"
        Contrast function. A callable is not supported.
    fun_args : dict or None, default None
        For ``"logcosh"``, ``{"alpha": a}`` with ``1 <= a <= 2`` (default 1.0).
    max_iter : int, default 200
        Maximum number of iterations.
    tol : float, default 1e-4
        Convergence tolerance.
    w_init : array-like of shape (n_components, n_components) or None, default None
        Starting unmixing matrix; None draws one from ``random_state``.
    whiten_solver : {"svd", "eigh"}, default "svd"
        Accepted for scikit-learn compatibility; see above.
    random_state : int or None, default None
        Seed for the default ``w_init``; None means 0.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Unmixing matrix applied by ``transform``.
    mixing_ : ndarray of shape (n_features, n_components)
        Pseudo-inverse of ``components_``.
    mean_ : ndarray of shape (n_features,)
        Only when whitening.
    whitening_ : ndarray of shape (n_components, n_features)
        Only when whitening.
    n_iter_ : int
    n_features_in_ : int

    Raises
    ------
    ValueError
        For an unsupported ``algorithm``, ``whiten`` or ``fun``, ``alpha``
        outside [1, 2], or a ``w_init`` of the wrong shape.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> S = mojolearn.FastICA(n_components=3, random_state=0).fit_transform(X)
    """
    _parameters = ("n_components", "algorithm", "whiten", "fun", "fun_args", "max_iter", "tol", "w_init",
                   "whiten_solver", "random_state", "numeric_mode")

    def __init__(self, n_components=None, *, algorithm="parallel", whiten="unit-variance", fun="logcosh",
                 fun_args=None, max_iter=200, tol=1e-4, w_init=None, whiten_solver="svd", random_state=None,
                 numeric_mode=None):
        self.n_components, self.algorithm, self.whiten, self.fun = n_components, algorithm, whiten, fun
        self.fun_args, self.max_iter, self.tol, self.w_init = fun_args, max_iter, tol, w_init
        self.whiten_solver, self.random_state, self.numeric_mode = whiten_solver, random_state, numeric_mode

    def fit_transform(self, X, y=None):
        """Fit the model and return the estimated sources.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DECOMP_CALL(self, "fit_transform", (X,), _DECOMP_CONTEXT)

    def fit(self, X, y=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        return _DECOMP_CALL(self, "fit", (X,), _DECOMP_CONTEXT)

    def transform(self, X, copy=True):
        """Recover the sources from X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        copy : bool, default True
            Accepted for scikit-learn compatibility; X is never modified.

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DECOMP_CALL(self, "transform", (X,), _DECOMP_CONTEXT)

    def inverse_transform(self, X, copy=True):
        """Map sources back to the feature space with ``mixing_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
        copy : bool, default True
            Accepted for scikit-learn compatibility; X is never modified.

        Returns
        -------
        ndarray of shape (n_samples, n_features), float32
        """
        return _DECOMP_CALL(self, "inverse_transform", (X,), _DECOMP_CONTEXT)


# ================================================================ FactorAnalysis
_LOG_2PI = 1.8378770664093453


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
    return float(k.b.x_decomp_dsum_sq(M.addr, n))  # cpu-route: host matrix (_M buffer) read once by x_decomp_dsum_sq on a CPU-only install


# Every public FactorAnalysis operation is one call into the compiled owner
# (bindings/fa_native.mojo, driver x_decomp/fa_owner.mojo).
# Reference: scikit-learn `decomposition/_factor_analysis.py`: `fit`,
# `transform`, `get_covariance`, `get_precision`, `score_samples`,
# `_ortho_rotation`. Each iteration's SVD of the scaled data is the QR +
# one-sided Jacobi SVD of decomposition/ (the eigh of the d x d Gram when
# n < d). svd_method='randomized' takes the same exact route (sklearn's
# randomized SVD is an approximation of it).
class FactorAnalysis(_Base):
    """Factor analysis: a linear Gaussian latent variable model.

    Follows scikit-learn's ``FactorAnalysis``, fitted by EM. Both
    ``svd_method`` values use an exact SVD at every iteration (scikit-learn's
    ``"randomized"`` is an approximation of it), so ``iterated_power`` and
    ``random_state`` have no effect.

    Parameters
    ----------
    n_components : int or None, default None
        Dimension of the latent space; None uses ``n_features``.
    tol : float, default 1e-2
        Stop when the log-likelihood gain falls below this.
    copy : bool, default True
        Accepted for scikit-learn compatibility; X is never modified.
    max_iter : int, default 1000
        Maximum number of EM iterations.
    noise_variance_init : array-like of shape (n_features,) or None, default None
        Starting noise variances; None uses ones.
    svd_method : {"lapack", "randomized"}, default "randomized"
        Accepted for scikit-learn compatibility; both run the exact SVD.
    iterated_power : int, default 3
        Accepted for scikit-learn compatibility; unused.
    rotation : {"varimax", "quartimax"} or None, default None
        Rotation applied to the fitted components.
    random_state : int or None, default 0
        Accepted for scikit-learn compatibility; unused.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Factor loadings.
    noise_variance_ : ndarray of shape (n_features,)
    mean_ : ndarray of shape (n_features,)
    loglike_ : list of float
        Log-likelihood at each iteration.
    n_iter_ : int
    n_features_in_ : int

    Raises
    ------
    ValueError
        For an unsupported ``svd_method`` or ``rotation``, or a
        ``noise_variance_init`` of the wrong length.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> fa = mojolearn.FactorAnalysis(n_components=2).fit(X)
    >>> Z = fa.transform(X)
    """
    _parameters = ("n_components", "tol", "copy", "max_iter", "noise_variance_init", "svd_method",
                   "iterated_power", "rotation", "random_state", "numeric_mode")

    def __init__(self, n_components=None, *, tol=1e-2, copy=True, max_iter=1000, noise_variance_init=None,
                 svd_method="randomized", iterated_power=3, rotation=None, random_state=0, numeric_mode=None):
        self.n_components, self.tol, self.copy, self.max_iter = n_components, tol, copy, max_iter
        self.noise_variance_init, self.svd_method, self.iterated_power = noise_variance_init, svd_method, iterated_power
        self.rotation, self.random_state, self.numeric_mode = rotation, random_state, numeric_mode

    def fit(self, X, y=None):
        """Fit the model with EM.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        return _DECOMP_CALL(self, "fit", (X,), _DECOMP_CONTEXT)

    def transform(self, X):
        """Return the expected latent variables for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DECOMP_CALL(self, "transform", (X,), _DECOMP_CONTEXT)

    def fit_transform(self, X, y=None):
        """Fit to X, then return ``transform(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DECOMP_CALL(self, "fit_transform", (X,), _DECOMP_CONTEXT)

    def get_covariance(self):
        """Return the model covariance ``components_.T @ components_ + diag(noise_variance_)``.

        Returns
        -------
        ndarray of shape (n_features, n_features), float32
        """
        return _DECOMP_CALL(self, "get_covariance", (), _DECOMP_CONTEXT)

    def get_precision(self):
        """Return the inverse of the model covariance.

        Returns
        -------
        ndarray of shape (n_features, n_features), float32

        Raises
        ------
        ValueError
            If the covariance is singular.
        """
        return _DECOMP_CALL(self, "get_precision", (), _DECOMP_CONTEXT)

    def score_samples(self, X):
        """Return the log-likelihood of each sample under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,), float32
        """
        return _DECOMP_CALL(self, "score_samples", (X,), _DECOMP_CONTEXT)

    def score(self, X, y=None):
        """Return the average log-likelihood of the samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        float
        """
        return _DECOMP_CALL(self, "score", (X,), _DECOMP_CONTEXT)


# ================================================================ TruncatedSVD explained variance
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
    return int(k.b.x_decomp_pca_mle_rank(words.buffer_info()[0], [d, int(n_samples)]))  # cpu-route: array('d') eigenvalue words read once by x_decomp_pca_mle_rank on a CPU-only install


# ================================================================ LU
# LAPACK getrf semantics. Partial pivoting on the largest |a[i, k]|, ties
# broken by the lowest row index. A zero pivot is kept (getrf's info > 0).
def lu_factor(a, *, numeric_mode=None):
    """LU factorization of a square matrix with partial pivoting.

    Follows ``scipy.linalg.lu_factor``: returns ``(lu, piv)``, where ``lu``
    holds the unit lower-triangular L (diagonal not stored) and U in one
    float32 matrix, and ``piv`` holds the 0-based row interchanges, applied
    in order. Pivoting picks the largest absolute value in the column; ties
    go to the lowest row. A zero pivot is kept and a ``RuntimeWarning`` is
    issued, as scipy does.

    Parameters
    ----------
    a : array-like of shape (n, n)
        The matrix to factor; must be finite.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this call. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Returns
    -------
    lu : ndarray of shape (n, n), float32
    piv : ndarray of shape (n,), int32
        Pass the pair to :func:`lu_solve`.

    Raises
    ------
    ValueError
        If ``a`` is not square, is empty, or contains NaN or infinity.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> A = np.random.default_rng(0).random((8, 8), dtype=np.float32) + 8 * np.eye(8, dtype=np.float32)
    >>> lu, piv = mojolearn.lu_factor(A)
    >>> x = mojolearn.lu_solve((lu, piv), np.ones(8, dtype=np.float32))
    """
    k = _Kit(_mode(numeric_mode))
    sf = k.s_flags()
    if ((sf & 4 and k.qfix_flags() & 4) or sf & 8) and not _is_sparse(a):
        # LU_FAST_RESIDENT (bit 4, FAST + Apple default, rollback _OFF) or
        # IDN_LU_RESIDENT_PAIR (bit 8, IDENTICAL default, rollback -D MOJOLEARN_IDN_LU_RESIDENT_PAIR_OFF);
        # x_decomp/s_linalg_fast.mojo
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


def _lu_solve_resident(k, dev, b, refine=True):
    """LU_FAST_RESIDENT's `lu_solve` (trans 0) on the pair's device A, LU and
    piv: B up, the solve, LU_QFIX's refinement step (refine; FAST only, off
    on IDN_LU_RESIDENT_PAIR) and X down."""
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
    k.b.x_decomp_dev_lu_solve(dLU.id, dP.id, dA.id, addr_ro(bb, name="b"), _addr(out, name="x"), [n, nrhs, 1 if refine else 0])
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


# LAPACK getrs semantics.
def lu_solve(lu_and_piv, b, *, trans=0, numeric_mode=None):
    """Solve a linear system from an LU factorization.

    Follows ``scipy.linalg.lu_solve``: solves ``A x = b`` (``trans=0``) or
    ``A.T x = b`` (``trans=1`` or ``2``; the same for a real matrix) with the
    pair returned by :func:`lu_factor`. A zero pivot gives 0 in that
    component of x, never infinity or NaN.

    Parameters
    ----------
    lu_and_piv : tuple (lu, piv)
        The output of :func:`lu_factor`.
    b : array-like of shape (n,) or (n, nrhs)
        Right-hand side.
    trans : {0, 1, 2}, default 0
        Which system to solve.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this call. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Returns
    -------
    x : ndarray of shape (n,) or (n, nrhs), float32
        Same shape as ``b``.

    Raises
    ------
    ValueError
        If ``trans`` is not 0, 1 or 2, ``piv`` holds an index outside
        [0, n), ``b`` has the wrong number of rows, or ``b`` is not finite.
    """
    if trans not in (0, 1, 2):
        raise ValueError("trans must be 0, 1 or 2")
    lu, piv = lu_and_piv
    k = _Kit(_mode(numeric_mode))
    dev = getattr(lu_and_piv, "_dev", None)
    if dev is not None and not trans and lu is dev[3] and piv is dev[4] and k.s_flags() & 12:
        # LU_FAST_RESIDENT (bit 4, FAST + Apple default, rollback _OFF) or
        # IDN_LU_RESIDENT_PAIR (bit 8, IDENTICAL, no refinement): the pair's device factor
        X = _lu_solve_resident(k, dev, b, refine=bool(k.s_flags() & 4))
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


# lu_factor + lu_solve (gesv). Lane idn-dense-linalg: one binding entry
# (`x_decomp_lu_gesv`) when the binding routes there (`x_decomp_idn_flags`
# bit 1): the same launches with the factor and the pivots resident, so
# only A and B go up and X comes down; the same words, warning and refusals.
def solve(a, b, *, numeric_mode=None):
    """Solve the square linear system ``a @ x = b``.

    Follows ``numpy.linalg.solve``, computed by LU factorization with partial
    pivoting (as :func:`lu_factor` and :func:`lu_solve`). A singular matrix
    gives a ``RuntimeWarning`` rather than an exception.

    Parameters
    ----------
    a : array-like of shape (n, n)
    b : array-like of shape (n,) or (n, nrhs)
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this call. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Returns
    -------
    x : ndarray of shape (n,) or (n, nrhs), float32

    Raises
    ------
    ValueError
        If ``a`` is not square or ``b`` has the wrong number of rows.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> A = np.random.default_rng(0).random((8, 8), dtype=np.float32) + 8 * np.eye(8, dtype=np.float32)
    >>> x = mojolearn.solve(A, np.ones(8, dtype=np.float32))
    """
    return _LINALG_CALL(None, "solve", (a, b, numeric_mode), _LINALG_CONTEXT)


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


# Halko et al.; RAFT `linalg/rsvd.cuh` is the same scheme. The Gaussian
# test matrix is the lane's Philox stream. Every power iteration
# re-orthonormalizes by the two-pass Householder-R solve whatever
# `power_iteration_normalizer` says: 'LU', 'QR' and 'none' span the same
# subspace, only the rounding of the basis differs. The small SVD of
# Q^T M is exact (Gram eigh).
def randomized_svd(M, n_components, *, n_oversamples=10, n_iter="auto", power_iteration_normalizer="auto",
                   transpose="auto", flip_sign=True, random_state=None, numeric_mode=None):
    """Truncated SVD by randomized range finding.

    Follows scikit-learn's ``utils.extmath.randomized_svd`` (Halko et al.).
    Every power iteration re-orthonormalizes with a QR step whatever
    ``power_iteration_normalizer`` says (all choices span the same
    subspace), and the random test matrix comes from mojolearn's own
    generator, so results differ slightly from scikit-learn's for the same
    seed.

    Parameters
    ----------
    M : array-like or scipy.sparse matrix of shape (n_samples, n_features)
    n_components : int
        Number of singular values and vectors; between 1 and
        ``min(n_samples, n_features)``.
    n_oversamples : int, default 10
        Extra random vectors used to sample the range.
    n_iter : int or "auto", default "auto"
        Power iterations. ``"auto"`` uses 7 when ``n_components < 0.1 *
        min(M.shape)``, else 4.
    power_iteration_normalizer : {"auto", "QR", "LU", "none"}, default "auto"
        Accepted for scikit-learn compatibility; see above.
    transpose : bool or "auto", default "auto"
        Work on ``M.T``; ``"auto"`` does so when M has more columns than rows.
    flip_sign : bool, default True
        Make the largest-magnitude entry of each column of U positive, as
        scikit-learn's ``svd_flip``.
    random_state : int or None, default None
        Seed of the random test matrix; None means 0.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this call. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Returns
    -------
    U : ndarray of shape (n_samples, n_components), float32
    s : ndarray of shape (n_components,), float32
        Singular values, descending.
    Vt : ndarray of shape (n_components, n_features), float32

    Raises
    ------
    ValueError
        If ``n_components`` is out of range, ``power_iteration_normalizer`` is
        unknown, or M is numerically zero.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 50), dtype=np.float32)
    >>> U, s, Vt = mojolearn.randomized_svd(X, 5, random_state=0)
    """
    return _LINALG_CALL(None, "randomized_svd", (M, n_components, n_oversamples, n_iter, power_iteration_normalizer,
                                                    transpose, flip_sign, random_state, numeric_mode), _LINALG_CONTEXT)


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


# A tall a with N + nrhs <= 512 takes the blocked TSQR of [a | b] and the
# SVD of its small R (`_tsqr_lstsq_core`); any other shape the QR +
# one-sided Jacobi SVD of a (or of a^T).
def lstsq(a, b, rcond=None, *, numeric_mode=None):
    """Least-squares solution of ``a @ x = b``.

    Follows ``numpy.linalg.lstsq``, solved through the SVD of ``a``.
    Singular values at or below ``rcond * s.max()`` are treated as zero.

    Parameters
    ----------
    a : array-like of shape (M, N)
    b : array-like of shape (M,) or (M, K)
    rcond : float or None, default None
        Relative cutoff for small singular values. None uses float32
        machine epsilon times ``max(M, N)``.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this call. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Returns
    -------
    x : ndarray of shape (N,) or (N, K), float32
    residuals : ndarray of shape (K,) or (0,), float32
        Squared column norms of ``b - a @ x`` when ``rank == N < M``;
        otherwise empty.
    rank : int
        Effective rank of ``a``.
    s : ndarray of shape (min(M, N),), float32
        Singular values of ``a``, descending.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` have different numbers of rows.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> A = rng.random((100, 4), dtype=np.float32)
    >>> x, res, rank, s = mojolearn.lstsq(A, rng.random(100, dtype=np.float32))
    """
    return _LINALG_CALL(None, "lstsq", (a, b, rcond, numeric_mode), _LINALG_CONTEXT)


# ================================================================ PLS / CCA
def _dot(k, a, b):
    """a . b for two column vectors (n x 1): one gemm cell, rows ascending."""
    return k.mm(a, b, ta=True)


class _PLS(RegressorScoreMixin, _Base):
    """sklearn `cross_decomposition/_pls.py::_PLS` (NIPALS): `fit`,
    `_get_first_singular_vectors_power_method`, `_center_scale_xy`,
    `_get_first_singular_vectors_svd` (PLSCanonical algorithm='svd'),
    `_svd_flip_1d`, `transform`, `inverse_transform`, `predict`."""
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "regressor"
    _sklearn_target_tags = {"multi_output": True}
    _parameters = ("n_components", "scale", "max_iter", "tol", "copy", "numeric_mode")
    _deflation = "regression"
    _pmode = "A"

    # fit, fit_transform, transform, inverse_transform and predict are one
    # call each into the native PLS owner (bindings/pls_native.mojo), which
    # reads `_deflation` / `_pmode` from the class.
    def __init__(self, n_components=2, *, scale=True, max_iter=500, tol=1e-06, copy=True, numeric_mode=None):
        self.n_components, self.scale, self.max_iter, self.tol, self.copy = n_components, scale, max_iter, tol, copy
        self.numeric_mode = numeric_mode

    def __sklearn_tags__(self):
        # scikit-learn's _PLS tags: a projection model, not scored as a
        # regressor by its estimator checks (poor_score)
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        return tags

    def fit(self, X, Y):
        """Fit the model to X and Y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples,) or (n_samples, n_targets)

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If Y is missing or ``n_components`` is out of range.
        """
        return _PLS_CALL(self, "fit", (X, Y, None), _PLS_CONTEXT)

    def transform(self, X, y=None, Y=None, copy=True):
        """Project X (and optionally Y) onto the fitted rotations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y, Y : array-like of shape (n_samples, n_targets), optional
            Targets to project as well (either name).
        copy : bool, default True
            Accepted for scikit-learn compatibility; inputs are never modified.

        Returns
        -------
        x_scores : ndarray of shape (n_samples, n_components), float32
        y_scores : ndarray of shape (n_samples, n_components), float32
            Returned only when Y is given.
        """
        return _PLS_CALL(self, "transform", (X, y, Y), _PLS_CONTEXT)

    def inverse_transform(self, X, y=None, Y=None):
        """Map scores back to the original X (and optionally Y) space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            X scores.
        y, Y : array-like of shape (n_samples, n_components), optional
            Y scores (either name).

        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_features), float32
        Y_reconstructed : ndarray of shape (n_samples, n_targets), float32
            Returned only when Y scores are given.
        """
        return _PLS_CALL(self, "inverse_transform", (X, y, Y), _PLS_CONTEXT)

    def predict(self, X, copy=True):
        """Predict targets for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        copy : bool, default True
            Accepted for scikit-learn compatibility; X is never modified.

        Returns
        -------
        ndarray of shape (n_samples,) or (n_samples, n_targets), float32
            One-dimensional when the model was fitted with a 1-D Y.
        """
        return _PLS_CALL(self, "predict", (X,), _PLS_CONTEXT)

    def fit_transform(self, X, y=None):
        """Fit the model, then return ``transform(X, y)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, n_targets)

        Returns
        -------
        x_scores, y_scores : ndarray of shape (n_samples, n_components), float32
        """
        return _PLS_CALL(self, "fit_transform", (X, y, None), _PLS_CONTEXT)

    def score(self, X, y, sample_weight=None):
        """Coefficient of determination R^2 of ``predict(X)`` against ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True targets.
        sample_weight : array-like of shape (n_samples,), default None
            Sample weights.

        Returns
        -------
        float
            R^2, averaged uniformly over outputs; 1.0 is a perfect fit.
        """
        return _PLS_CALL(self, "score",
                         (X, y, sample_weight, _get_binding(getattr(self, "numeric_mode", None))), _PLS_CONTEXT)


# NIPALS, mode A, regression deflation.
class PLSRegression(_PLS):
    """Partial least squares regression.

    Follows scikit-learn's ``PLSRegression`` (NIPALS algorithm). If Y's
    residual becomes constant before ``n_components`` are found, a warning
    is issued and fewer components are kept.

    Parameters
    ----------
    n_components : int, default 2
        Number of components to keep, between 1 and ``n_features``.
    scale : bool, default True
        Scale X and Y to unit variance.
    max_iter : int, default 500
        Maximum NIPALS iterations per component.
    tol : float, default 1e-06
        NIPALS convergence tolerance.
    copy : bool, default True
        Accepted for scikit-learn compatibility; inputs are never modified.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    x_weights_, y_weights_ : ndarray of shape (n_features, n_components), (n_targets, n_components)
    x_loadings_, y_loadings_ : ndarray of shape (n_features, n_components), (n_targets, n_components)
    x_rotations_, y_rotations_ : ndarray of shape (n_features, n_components), (n_targets, n_components)
    coef_ : ndarray of shape (n_targets, n_features)
    intercept_ : ndarray of shape (n_targets,)
    n_iter_ : list of int
        NIPALS iterations for each component.
    n_features_in_ : int
    x_scores_, y_scores_ : ndarray of shape (n_samples, n_components)
        Scores of the training data.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X @ rng.random(8, dtype=np.float32)
    >>> pred = mojolearn.PLSRegression(n_components=2).fit(X, y).predict(X)
    """
    _deflation, _pmode = "regression", "A"

    def fit(self, X, y=None, Y=None):
        """Fit the model to X and the targets (``y`` or ``Y``).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y, Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Targets (either name).

        Returns
        -------
        self
        """
        return _PLS_CALL(self, "fit", (X, y, Y), _PLS_CONTEXT)


# Mode A, canonical deflation.
class PLSCanonical(_PLS):
    """Partial least squares in canonical form.

    Follows scikit-learn's ``PLSCanonical``: finds paired directions in X
    and Y with maximal covariance, deflating both blocks symmetrically.

    Parameters
    ----------
    n_components : int, default 2
        Number of components to keep, between 1 and
        ``min(n_samples, n_features, n_targets)``.
    scale : bool, default True
        Scale X and Y to unit variance.
    algorithm : {"nipals", "svd"}, default "nipals"
        How the first singular vectors of each step are found. Any other
        value raises ValueError at construction.
    max_iter : int, default 500
        Maximum NIPALS iterations per component.
    tol : float, default 1e-06
        NIPALS convergence tolerance.
    copy : bool, default True
        Accepted for scikit-learn compatibility; inputs are never modified.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    x_weights_, y_weights_ : ndarray of shape (n_features, n_components), (n_targets, n_components)
    x_loadings_, y_loadings_ : ndarray of shape (n_features, n_components), (n_targets, n_components)
    x_rotations_, y_rotations_ : ndarray of shape (n_features, n_components), (n_targets, n_components)
    coef_ : ndarray of shape (n_targets, n_features)
    intercept_ : ndarray of shape (n_targets,)
    n_iter_ : list of int
        NIPALS iterations for each component.
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X, Y = rng.random((1000, 8), dtype=np.float32), rng.random((1000, 3), dtype=np.float32)
    >>> xs, ys = mojolearn.PLSCanonical(n_components=2).fit(X, Y).transform(X, Y)
    """
    _deflation, _pmode = "canonical", "A"
    _parameters = ("n_components", "scale", "algorithm", "max_iter", "tol", "copy", "numeric_mode")

    def __init__(self, n_components=2, *, scale=True, algorithm="nipals", max_iter=500, tol=1e-06, copy=True,
                 numeric_mode=None):
        self.algorithm = algorithm
        super().__init__(n_components, scale=scale, max_iter=max_iter, tol=tol, copy=copy, numeric_mode=numeric_mode)


# NIPALS, mode B (the pseudo-inverses of X and Y), canonical deflation.
class CCA(_PLS):
    """Canonical correlation analysis.

    Follows scikit-learn's ``CCA`` (NIPALS algorithm): finds paired
    directions in X and Y with maximal correlation.

    Parameters
    ----------
    n_components : int, default 2
        Number of components to keep, between 1 and
        ``min(n_samples, n_features, n_targets)``.
    scale : bool, default True
        Scale X and Y to unit variance.
    max_iter : int, default 500
        Maximum NIPALS iterations per component.
    tol : float, default 1e-06
        NIPALS convergence tolerance.
    copy : bool, default True
        Accepted for scikit-learn compatibility; inputs are never modified.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    x_weights_, y_weights_ : ndarray of shape (n_features, n_components), (n_targets, n_components)
    x_loadings_, y_loadings_ : ndarray of shape (n_features, n_components), (n_targets, n_components)
    x_rotations_, y_rotations_ : ndarray of shape (n_features, n_components), (n_targets, n_components)
    coef_ : ndarray of shape (n_targets, n_features)
    intercept_ : ndarray of shape (n_targets,)
    n_iter_ : list of int
        NIPALS iterations for each component.
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X, Y = rng.random((1000, 8), dtype=np.float32), rng.random((1000, 3), dtype=np.float32)
    >>> xs, ys = mojolearn.CCA(n_components=2).fit(X, Y).transform(X, Y)
    """
    _deflation, _pmode = "canonical", "B"


# ================================================================ dictionary learning / sparse PCA
def sparse_encode(X, dictionary, *, algorithm="lasso_lars", n_nonzero_coefs=None, alpha=None, init=None,
                  max_iter=1000, positive=False, numeric_mode=None, **_ignored):
    """Sparse-code each row of X against a fixed dictionary.

    Follows scikit-learn's ``sparse_encode``. ``"lasso_lars"`` and
    ``"lasso_cd"`` solve the same Lasso problem (both by coordinate descent;
    the optimum does not depend on the path). Other scikit-learn keywords
    (such as ``n_jobs`` or ``check_input``) are accepted and ignored.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    dictionary : array-like of shape (n_components, n_features)
        Dictionary atoms, one per row.
    algorithm : {"lasso_lars", "lasso_cd", "lars", "omp", "threshold"}, default "lasso_lars"
        Coding method.
    n_nonzero_coefs : int or None, default None
        Nonzero coefficients per row for ``"lars"`` and ``"omp"``. None uses
        ``min(max(n_features / 10, 1), n_components)``.
    alpha : float or None, default None
        Penalty for the Lasso methods, or threshold for ``"threshold"``.
        None uses 1.0.
    init : array-like of shape (n_samples, n_components) or None, default None
        Starting code for ``"lasso_cd"``.
    max_iter : int, default 1000
        Maximum iterations for the Lasso methods.
    positive : bool, default False
        Force the code to be non-negative (Lasso methods and ``"threshold"``).
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this call. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Returns
    -------
    ndarray of shape (n_samples, n_components), float32

    Raises
    ------
    ValueError
        For an unknown ``algorithm``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> D = rng.random((5, 8), dtype=np.float32)
    >>> code = mojolearn.sparse_encode(rng.random((100, 8), dtype=np.float32), D, algorithm="omp", n_nonzero_coefs=2)
    """
    return _DL_CALL(None, "sparse_encode", (X, dictionary, algorithm, n_nonzero_coefs, alpha, init, max_iter,
                                            positive, numeric_mode), _DL_CONTEXT)


class _SparseCoding(_Base):
    def transform(self, X):
        """Encode X as a sparse combination of the dictionary atoms.

        Uses ``transform_algorithm`` and its settings.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
            Twice as many columns when ``split_sign=True``.
        """
        return _DL_CALL(self, "coding_transform", (X,), _DL_CONTEXT)

    def inverse_transform(self, X):
        """Rebuild data from a code: ``code @ components_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            A code, as returned by ``transform`` (with ``2 * n_components``
            columns when ``split_sign=True``).

        Returns
        -------
        ndarray of shape (n_samples, n_features), float32
        """
        return _DL_CALL(self, "coding_inverse", (X,), _DL_CONTEXT)


# Reference: scikit-learn `decomposition/_dict_learning.py`:
# `_dict_learning`, `_update_dict`, `_sparse_encode_precomputed`,
# `_BaseSparseCoding._transform`. The initial SVD is exact (QR + one-sided
# Jacobi); a resampled unused atom is drawn from the Philox stream.
# fit_algorithm 'lars' solves its Lasso by coordinate descent (the same
# optimum). callback is refused by name.
class DictionaryLearning(_SparseCoding):
    """Learn a dictionary in which the data has a sparse representation.

    Follows scikit-learn's ``DictionaryLearning``: alternates sparse coding
    of X with updates of the dictionary atoms. The starting dictionary comes
    from an exact SVD of X, and both ``fit_algorithm`` values solve the same
    Lasso problem by coordinate descent.

    Parameters
    ----------
    n_components : int or None, default None
        Number of atoms; None uses ``n_features``.
    alpha : float, default 1
        Sparsity penalty used while fitting.
    max_iter : int, default 1000
        Maximum number of iterations.
    tol : float, default 1e-8
        Convergence tolerance.
    fit_algorithm : {"lars", "cd"}, default "lars"
        Lasso solver name used while fitting (both give the same optimum).
    transform_algorithm : {"omp", "lasso_lars", "lasso_cd", "lars", "threshold"}, default "omp"
        Method used by ``transform``; see :func:`sparse_encode`.
    transform_n_nonzero_coefs : int or None, default None
        Nonzero coefficients for ``"omp"`` and ``"lars"`` in ``transform``.
    transform_alpha : float or None, default None
        Penalty or threshold used by ``transform``; None uses ``alpha``.
    n_jobs : ignored
    code_init : array-like of shape (n_samples, n_components) or None, default None
        Starting code; used only together with ``dict_init``.
    dict_init : array-like of shape (n_components, n_features) or None, default None
        Starting dictionary; used only together with ``code_init``.
    callback : None
        Must be None; a callable raises NotImplementedError.
    verbose : ignored
    split_sign : bool, default False
        Return the positive and negative parts of the code as separate
        columns.
    random_state : int or None, default None
        Seed used to resample unused atoms; None means 0.
    positive_code : bool, default False
        Force the code to be non-negative.
    positive_dict : bool, default False
        Force the dictionary to be non-negative.
    transform_max_iter : int, default 1000
        Maximum iterations of the Lasso methods during fit and transform.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The dictionary atoms.
    error_ : list of float
        Objective value at each iteration.
    n_iter_ : int
    n_features_in_ : int

    Raises
    ------
    ValueError
        For an unknown ``fit_algorithm`` or ``transform_algorithm``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((500, 8), dtype=np.float32)
    >>> dl = mojolearn.DictionaryLearning(n_components=5, max_iter=20, random_state=0)
    >>> code = dl.fit_transform(X)
    """
    _parameters = ("n_components", "alpha", "max_iter", "tol", "fit_algorithm", "transform_algorithm",
                   "transform_n_nonzero_coefs", "transform_alpha", "split_sign", "random_state",
                   "positive_code", "positive_dict", "transform_max_iter", "numeric_mode",
                   "callback", "code_init", "dict_init", "n_jobs", "verbose")

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
        """Fit the dictionary to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        return _DL_CALL(self, "dl_fit", (X,), _DL_CONTEXT)

    def fit_transform(self, X, y=None):
        """Fit the dictionary and return the code found while fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
            Twice as many columns when ``split_sign=True``.
        """
        return _DL_CALL(self, "dl_fit_transform", (X,), _DL_CONTEXT)


# Reference: `_dict_learning.py::MiniBatchDictionaryLearning`:
# `_initialize_dict`, `_minibatch_step`, `_update_inner_stats`,
# `_check_convergence`. The shuffle is a Philox permutation (sort of
# counter draws, ties to the lower index) and the initial dictionary the
# exact SVD (sklearn: randomized).
class MiniBatchDictionaryLearning(_SparseCoding):
    """Learn a sparse-coding dictionary from mini-batches.

    Follows scikit-learn's ``MiniBatchDictionaryLearning``. The starting
    dictionary comes from an exact SVD (scikit-learn uses a randomized one)
    and the shuffle uses mojolearn's own generator, so results differ
    slightly from scikit-learn's for the same seed.

    Parameters
    ----------
    n_components : int or None, default None
        Number of atoms; None uses ``n_features``.
    alpha : float, default 1
        Sparsity penalty used while fitting.
    max_iter : int, default 1000
        Maximum passes over the data.
    fit_algorithm : {"lars", "cd"}, default "lars"
        Lasso solver name used while fitting.
    n_jobs : ignored
    batch_size : int, default 256
        Rows per mini-batch (capped at ``n_samples``).
    shuffle : bool, default True
        Shuffle the rows once before fitting.
    dict_init : array-like of shape (n_components, n_features) or None, default None
        Starting dictionary.
    transform_algorithm : {"omp", "lasso_lars", "lasso_cd", "lars", "threshold"}, default "omp"
        Method used by ``transform``; see :func:`sparse_encode`.
    transform_n_nonzero_coefs : int or None, default None
        Nonzero coefficients for ``"omp"`` and ``"lars"`` in ``transform``.
    transform_alpha : float or None, default None
        Penalty or threshold used by ``transform``; None uses ``alpha``.
    verbose : ignored
    split_sign : bool, default False
        Return the positive and negative parts of the code as separate
        columns.
    random_state : int or None, default None
        Seed for the shuffle; None means 0.
    positive_code : bool, default False
        Force the code to be non-negative.
    positive_dict : bool, default False
        Force the dictionary to be non-negative.
    transform_max_iter : int, default 1000
        Maximum iterations of the Lasso methods during fit and transform.
    callback : None
        Must be None; a callable raises NotImplementedError.
    tol : float, default 1e-3
        Stop when the dictionary changes less than this between steps.
    max_no_improvement : int or None, default 10
        Stop after this many mini-batches without improvement of the
        smoothed cost; None disables this test.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The dictionary atoms.
    n_iter_ : int
        Passes over the data.
    n_steps_ : int
        Mini-batches processed.
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((1000, 8), dtype=np.float32)
    >>> dl = mojolearn.MiniBatchDictionaryLearning(n_components=5, max_iter=5, random_state=0).fit(X)
    >>> code = dl.transform(X)
    """
    _parameters = ("n_components", "alpha", "max_iter", "fit_algorithm", "batch_size", "shuffle",
                   "transform_algorithm", "transform_n_nonzero_coefs", "transform_alpha", "split_sign",
                   "random_state", "positive_code", "positive_dict", "transform_max_iter", "tol",
                   "max_no_improvement", "numeric_mode",
                   "callback", "dict_init", "n_jobs", "verbose")

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
        """Fit the dictionary to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        return _DL_CALL(self, "mbdl_fit", (X,), _DL_CONTEXT)

    def fit_transform(self, X, y=None):
        """Fit the dictionary, then return ``transform(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DL_CALL(self, "mbdl_fit_transform", (X,), _DL_CONTEXT)


class _BaseSparsePCA(_Base):
    # fit, fit_transform, transform and inverse_transform are one call each
    # into the native dictionary-learning owner (bindings/dict_learning_native.mojo),
    # which tells SparsePCA and MiniBatchSparsePCA apart by class.
    def fit_transform(self, X, y=None):
        """Fit the sparse components, then return ``transform(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DL_CALL(self, "spca_fit_transform", (X,), _DL_CONTEXT)

    # sklearn `_BaseSparsePCA.transform`: ridge_regression(components_.T,
    # (X - mean_).T, ridge_alpha, solver='cholesky').
    def transform(self, X):
        """Project X onto the sparse components by ridge regression.

        Solves a least-squares problem with penalty ``ridge_alpha`` for each
        centered row, as scikit-learn does.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DL_CALL(self, "spca_transform", (X,), _DL_CONTEXT)

    def inverse_transform(self, X):
        """Map projected data back: ``X @ components_ + mean_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)

        Returns
        -------
        ndarray of shape (n_samples, n_features), float32
        """
        return _DL_CALL(self, "spca_inverse", (X,), _DL_CONTEXT)


# Reference: scikit-learn `decomposition/_sparse_pca.py`: dictionary
# learning on the centered X^T, the code's transpose the components.
class SparsePCA(_BaseSparsePCA):
    """Sparse principal component analysis.

    Follows scikit-learn's ``SparsePCA``: finds components with few nonzero
    loadings by dictionary learning on the centered, transposed data. Each
    component is scaled to unit norm.

    Parameters
    ----------
    n_components : int or None, default None
        Number of components; None uses ``n_features``.
    alpha : float, default 1
        Sparsity penalty; larger values give sparser components.
    ridge_alpha : float, default 0.01
        Ridge penalty used by ``transform``.
    max_iter : int, default 1000
        Maximum number of iterations.
    tol : float, default 1e-8
        Convergence tolerance.
    method : {"lars", "cd"}, default "lars"
        Lasso solver name (both give the same optimum).
    n_jobs : ignored
    U_init : array-like of shape (n_samples, n_components) or None, default None
        Starting sample loadings; used only together with ``V_init``.
    V_init : array-like of shape (n_components, n_features) or None, default None
        Starting components; used only together with ``U_init``.
    verbose : ignored
    random_state : int or None, default None
        Seed used to resample unused atoms; None means 0.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Sparse components, each scaled to unit norm.
    n_components_ : int
    mean_ : ndarray of shape (n_features,)
    n_iter_ : int
    n_features_in_ : int
    error_ : list of float
        Objective value at each iteration.

    Raises
    ------
    ValueError
        If ``method`` is not ``"lars"`` or ``"cd"``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((500, 8), dtype=np.float32)
    >>> spca = mojolearn.SparsePCA(n_components=3, max_iter=20, random_state=0).fit(X)
    >>> Z = spca.transform(X)
    """
    _parameters = ("n_components", "alpha", "ridge_alpha", "max_iter", "tol", "method", "random_state",
                   "numeric_mode",
                   "U_init", "V_init", "n_jobs", "verbose")

    def __init__(self, n_components=None, *, alpha=1, ridge_alpha=0.01, max_iter=1000, tol=1e-8, method="lars",
                 n_jobs=None, U_init=None, V_init=None, verbose=False, random_state=None, numeric_mode=None):
        self.n_components, self.alpha, self.ridge_alpha = n_components, alpha, ridge_alpha
        self.max_iter, self.tol, self.method, self.n_jobs = max_iter, tol, method, n_jobs
        self.U_init, self.V_init, self.verbose, self.random_state = U_init, V_init, verbose, random_state
        self.numeric_mode = numeric_mode

    def fit(self, X, y=None):
        """Fit the sparse components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        return _DL_CALL(self, "spca_fit", (X,), _DL_CONTEXT)


# MiniBatchDictionaryLearning on the centered X^T, the components its
# lasso_lars transform of X^T. callback is not forwarded (never called).
class MiniBatchSparsePCA(_BaseSparsePCA):
    """Sparse principal component analysis fitted on mini-batches.

    Follows scikit-learn's ``MiniBatchSparsePCA``: runs
    :class:`MiniBatchDictionaryLearning` on the centered, transposed data.
    Each component is scaled to unit norm.

    Parameters
    ----------
    n_components : int or None, default None
        Number of components; None uses ``n_features``.
    alpha : float, default 1
        Sparsity penalty.
    ridge_alpha : float, default 0.01
        Ridge penalty used by ``transform``.
    max_iter : int, default 1000
        Maximum passes over the data.
    callback : ignored
    batch_size : int, default 3
        Features per mini-batch (the data is transposed).
    verbose : ignored
    shuffle : bool, default True
        Shuffle once before fitting.
    n_jobs : ignored
    method : {"lars", "cd"}, default "lars"
        Lasso solver name.
    random_state : int or None, default None
        Seed for the shuffle; None means 0.
    tol : float, default 1e-3
        Convergence tolerance on the dictionary change.
    max_no_improvement : int or None, default 10
        Stop after this many mini-batches without improvement; None disables
        this test.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Sparse components, each scaled to unit norm.
    n_components_ : int
    mean_ : ndarray of shape (n_features,)
    n_iter_ : int
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((500, 8), dtype=np.float32)
    >>> spca = mojolearn.MiniBatchSparsePCA(n_components=3, max_iter=5, random_state=0).fit(X)
    """
    _parameters = ("n_components", "alpha", "ridge_alpha", "max_iter", "batch_size", "shuffle", "method",
                   "random_state", "tol", "max_no_improvement", "numeric_mode",
                   "callback", "n_jobs", "verbose")

    def __init__(self, n_components=None, *, alpha=1, ridge_alpha=0.01, max_iter=1000, callback=None,
                 batch_size=3, verbose=False, shuffle=True, n_jobs=None, method="lars", random_state=None,
                 tol=1e-3, max_no_improvement=10, numeric_mode=None):
        self.n_components, self.alpha, self.ridge_alpha, self.max_iter = n_components, alpha, ridge_alpha, max_iter
        self.callback, self.batch_size, self.verbose, self.shuffle = callback, batch_size, verbose, shuffle
        self.n_jobs, self.method, self.random_state, self.tol = n_jobs, method, random_state, tol
        self.max_no_improvement, self.numeric_mode = max_no_improvement, numeric_mode

    def fit(self, X, y=None):
        """Fit the sparse components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        return _DL_CALL(self, "spca_fit", (X,), _DL_CONTEXT)


# ================================================================ LatentDirichletAllocation


# Reference: scikit-learn `decomposition/_lda.py` with
# `_online_lda_fast.pyx`: `_init_latent_vars`, `_e_step`,
# `_update_doc_distribution`, `_em_step`, `_approx_bound`, `transform`,
# `perplexity`, `score`. Variational EM in a fixed order: every document's
# inner loop is one thread, the sufficient statistics one gemm. The
# Gamma(100, 1/100) draws are Marsaglia-Tsang on the Philox stream. Online
# mini-batches go in row order, as sklearn with no shuffle does. A
# scipy.sparse count matrix is densified (exact); the per-document cell
# already skips zero counts.
class LatentDirichletAllocation(_Base):
    """Latent Dirichlet allocation topic model, fitted by variational EM.

    Follows scikit-learn's ``LatentDirichletAllocation``. X is a
    document-term count matrix (non-negative; a ``scipy.sparse`` matrix is
    densified). Random draws come from mojolearn's own generator, so topics
    differ from scikit-learn's for the same seed. Online learning takes the
    mini-batches in row order.

    Parameters
    ----------
    n_components : int, default 10
        Number of topics.
    doc_topic_prior : float or None, default None
        Prior on the document-topic distribution; None uses
        ``1 / n_components``.
    topic_word_prior : float or None, default None
        Prior on the topic-word distribution; None uses ``1 / n_components``.
    learning_method : {"batch", "online"}, default "batch"
        Use all data in each update, or mini-batches.
    learning_decay : float, default 0.7
        Online learning rate decay.
    learning_offset : float, default 10.0
        Online learning rate offset (down-weights early iterations).
    max_iter : int, default 10
        Maximum passes over the data.
    batch_size : int, default 128
        Documents per mini-batch for online learning; must be a positive int.
    evaluate_every : int, default -1
        Compute perplexity every this many iterations and stop once it
        improves by less than ``perp_tol``; 0 or negative never evaluates.
    total_samples : float, default 1e6
        Total number of documents, used by ``partial_fit`` and
        ``perplexity(sub_sampling=True)``.
    perp_tol : float, default 0.1
        Perplexity tolerance for ``evaluate_every``.
    mean_change_tol : float, default 1e-3
        Stopping tolerance of the per-document E-step.
    max_doc_update_iter : int, default 100
        Maximum E-step iterations per document.
    n_jobs : ignored
    verbose : ignored
    random_state : int or None, default None
        Seed; None means 0.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Variational topic-word parameters.
    exp_dirichlet_component_ : ndarray of shape (n_components, n_features)
    doc_topic_prior_ : float
    topic_word_prior_ : float
    bound_ : float
        Final perplexity on the training data.
    n_iter_ : int
    n_batch_iter_ : int
    n_features_in_ : int

    Raises
    ------
    ValueError
        If X has negative values, or ``learning_method`` or ``batch_size``
        is invalid.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).integers(0, 5, (200, 30)).astype(np.float32)
    >>> lda = mojolearn.LatentDirichletAllocation(n_components=4, random_state=0).fit(X)
    >>> doc_topics = lda.transform(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _sklearn_input_tags = {"positive_only": True}
    _parameters = ("n_components", "doc_topic_prior", "topic_word_prior", "learning_method", "learning_decay",
                   "learning_offset", "max_iter", "batch_size", "evaluate_every", "total_samples", "perp_tol",
                   "mean_change_tol", "max_doc_update_iter", "random_state", "numeric_mode",
                   "n_jobs", "verbose")

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

    def fit(self, X, y=None):
        """Fit the topic model to a document-term matrix.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Non-negative word counts.
        y : ignored

        Returns
        -------
        self
        """
        return _DECOMP_CALL(self, "fit", (X,), _DECOMP_CONTEXT)

    def partial_fit(self, X, y=None):
        """Update the model with one batch of documents (online learning).

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Non-negative word counts, split into mini-batches of
            ``batch_size``.
        y : ignored

        Returns
        -------
        self
        """
        return _DECOMP_CALL(self, "partial_fit", (X,), _DECOMP_CONTEXT)

    def transform(self, X, *, normalize=True):
        """Return the topic distribution of each document.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
            Non-negative word counts.
        normalize : bool, default True
            Normalize each row to sum to 1.

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DECOMP_CALL(self, "transform", (X, normalize), _DECOMP_CONTEXT)

    def fit_transform(self, X, y=None, *, normalize=True):
        """Fit the model, then return ``transform(X, normalize=normalize)``.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
        y : ignored
        normalize : bool, default True

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DECOMP_CALL(self, "fit_transform", (X, normalize), _DECOMP_CONTEXT)

    def score(self, X, y=None):
        """Return the approximate log-likelihood (variational bound) of X.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        float
        """
        return _DECOMP_CALL(self, "score", (X,), _DECOMP_CONTEXT)

    def perplexity(self, X, sub_sampling=False):
        """Return the approximate perplexity of X.

        Parameters
        ----------
        X : array-like or scipy.sparse matrix of shape (n_samples, n_features)
        sub_sampling : bool, default False
            Scale the bound as if X were a sample of ``total_samples``
            documents.

        Returns
        -------
        float
            Infinity when X has no words.
        """
        return _DECOMP_CALL(self, "perplexity", (X, sub_sampling), _DECOMP_CONTEXT)


# ================================================================ manifold: Isomap, MDS, LLE
_PD_KIND = {"manhattan": 1, "cityblock": 1, "l1": 1, "chebyshev": 2, "infinity": 2, "cosine": 4}


# The manifold estimators' metric rule (`_metric_spec`), distances, kNN
# matrices, kernel centering and top eigenpairs run in their compiled owner
# (bindings/mani_decomp_native.mojo); `_PD_KIND` is its table.


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
#: (The route's n > 200 and n_components < 10 policy is in bindings/mani_decomp_native.mojo.)
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


# Reference: scikit-learn `manifold/_isomap.py` with
# `KernelPCA(kernel='precomputed')` and `utils/graph.py`: the kNN distance
# graph (ties to the lower index), all-pairs shortest paths by Dijkstra
# (one source per thread), the kernel -0.5 * D^2 centered, its top
# eigenpairs (Jacobi eigh). path_method 'FW' runs the same Dijkstra
# (x_decomp/NOT_IMPLEMENTED.tsv: deliberately divergent). Metrics per
# DEVIATION 5319. A callable or any other metric is refused by name.
class Isomap(_Base):
    """Isomap embedding: kernel PCA on geodesic distances.

    Follows scikit-learn's ``Isomap``: builds a nearest-neighbor (or radius)
    graph, computes shortest-path distances on it, and embeds the centered
    kernel ``-0.5 * D**2`` by its top eigenvectors. If the graph has more
    than one connected component, a warning is issued and the components are
    joined. Shortest paths always use Dijkstra's algorithm, and the
    eigenproblem is always solved exactly.

    Parameters
    ----------
    n_neighbors : int or None, default 5
        Neighbors per point. Set to None to use ``radius`` instead.
    radius : float or None, default None
        Connect every point within this distance (inclusive). Only one of
        ``n_neighbors`` and ``radius`` may be set.
    n_components : int, default 2
        Embedding dimension.
    eigen_solver : {"auto", "dense"}, default "auto"
        Both solve exactly. ``"arpack"`` raises NotImplementedError.
    tol : ignored
    max_iter : ignored
    path_method : {"auto", "FW", "D"}, default "auto"
        Accepted for scikit-learn compatibility; every value runs Dijkstra.
    neighbors_algorithm : ignored
    n_jobs : ignored
    metric : str, default "minkowski"
        Distance metric: ``"euclidean"``, ``"l2"``, ``"minkowski"`` (with ``p``), ``"manhattan"``
        (``"cityblock"``, ``"l1"``), ``"chebyshev"`` (``"infinity"``) or
        ``"cosine"``; a callable or any other name raises
        NotImplementedError.
    p : float, default 2
        Minkowski power (must be positive).
    metric_params : dict or None, default None
        Only ``{"p": ...}`` is supported; other keys raise
        NotImplementedError.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
    dist_matrix_ : ndarray of shape (n_samples, n_samples)
        Geodesic distances between training points.
    n_connected_components_ : int
    n_features_in_ : int

    Raises
    ------
    ValueError
        If both or neither of ``n_neighbors`` and ``radius`` are set,
        ``radius`` is negative, or ``path_method`` or ``eigen_solver`` is
        unknown.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((500, 8), dtype=np.float32)
    >>> Y = mojolearn.Isomap(n_neighbors=10, n_components=2).fit_transform(X)
    """
    _parameters = ("n_neighbors", "radius", "n_components", "eigen_solver", "path_method", "metric", "p", "numeric_mode",
                   "max_iter", "metric_params", "n_jobs", "neighbors_algorithm", "tol")

    def __init__(self, *, n_neighbors=5, radius=None, n_components=2, eigen_solver="auto", tol=0, max_iter=None,
                 path_method="auto", neighbors_algorithm="auto", n_jobs=None, metric="minkowski", p=2,
                 metric_params=None, numeric_mode=None):
        self.n_neighbors, self.radius, self.n_components = n_neighbors, radius, n_components
        self.eigen_solver, self.tol, self.max_iter, self.path_method = eigen_solver, tol, max_iter, path_method
        self.neighbors_algorithm, self.n_jobs, self.metric, self.p = neighbors_algorithm, n_jobs, metric, p
        self.metric_params, self.numeric_mode = metric_params, numeric_mode

    def fit(self, X, y=None):
        """Compute the embedding of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        return _NATIVE(self, "family_mani_iso_fit",
                       (_manifold_binding(self, "numeric_mode"), _backend.binding(_BINDING), X), _NATIVE_CONTEXT)

    def fit_transform(self, X, y=None):
        """Fit the model and return ``embedding_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _NATIVE(self, "family_mani_iso_fit_transform",
                       (_manifold_binding(self, "numeric_mode"), _backend.binding(_BINDING), X), _NATIVE_CONTEXT)

    def transform(self, X):
        """Embed new points.

        Each query's geodesic distance to the training points goes through
        its nearest training neighbors (or, with ``radius``, every training
        point within the radius), then is projected as in kernel PCA.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        ndarray of shape (n_queries, n_components), float32

        Raises
        ------
        ValueError
            With ``radius``, if a query has no training point within it.
        """
        return _NATIVE(self, "family_mani_iso_transform",
                       (_manifold_binding(self, "numeric_mode_"), _backend.binding(_BINDING), X), _NATIVE_CONTEXT)

    def reconstruction_error(self):
        """Return the reconstruction error of the embedding.

        Computed as in scikit-learn: the Frobenius norm of the difference
        between the centered geodesic kernel and its rank-``n_components``
        approximation, divided by ``n_samples``.

        Returns
        -------
        float
        """
        return _NATIVE(self, "family_mani_iso_reconstruction_error",
                       (_manifold_binding(self, "numeric_mode_"), _backend.binding(_BINDING)), _NATIVE_CONTEXT)
# Reference: scikit-learn `manifold/_classical_mds.py`.
class ClassicalMDS(_Base):
    """Classical (Torgerson) multidimensional scaling.

    Follows scikit-learn's ``ClassicalMDS``: double-centers ``-0.5 * D**2``
    for the pairwise distances D and embeds by its top eigenvectors scaled
    by the square roots of the eigenvalues.

    Parameters
    ----------
    n_components : int, default 2
        Embedding dimension.
    metric : str, default "euclidean"
        ``"precomputed"`` (X is a distance matrix) or a distance metric:
        ``"euclidean"``, ``"l2"``, ``"minkowski"`` (``p`` from ``metric_params``), ``"manhattan"``
        (``"cityblock"``, ``"l1"``), ``"chebyshev"`` (``"infinity"``) or
        ``"cosine"``; a callable or any other name raises
        NotImplementedError.
    metric_params : dict or None, default None
        Only ``{"p": ...}`` is supported.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
    eigenvalues_ : ndarray of shape (n_components,)
    dissimilarity_matrix_ : ndarray of shape (n_samples, n_samples)
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((500, 8), dtype=np.float32)
    >>> Y = mojolearn.ClassicalMDS(n_components=2).fit_transform(X)
    """
    _parameters = ("n_components", "metric", "numeric_mode",
                   "metric_params")

    def __init__(self, n_components=2, *, metric="euclidean", metric_params=None, numeric_mode=None):
        self.n_components, self.metric, self.metric_params, self.numeric_mode = n_components, metric, metric_params, numeric_mode

    def fit(self, X, y=None):
        """Compute the embedding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
            Data, or a distance matrix with ``metric="precomputed"``.
        y : ignored

        Returns
        -------
        self
        """
        return _NATIVE(self, "family_mani_cmds_fit",
                       (_manifold_binding(self, "numeric_mode"), _backend.binding(_BINDING), X), _NATIVE_CONTEXT)

    def fit_transform(self, X, y=None):
        """Compute the embedding and return ``embedding_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _NATIVE(self, "family_mani_cmds_fit_transform",
                       (_manifold_binding(self, "numeric_mode"), _backend.binding(_BINDING), X), _NATIVE_CONTEXT)
# Reference: scikit-learn `manifold/_mds.py`, `_smacof_single`, `smacof`.
# Random starts come from the Philox stream. init='random' is the default
# sklearn's FutureWarning names. Non-metric: the disparities are the
# isotonic regression of the distances on the dissimilarities (the linear
# lane's IsotonicRegression), rescaled.
class MDS(_Base):
    """Multidimensional scaling by SMACOF.

    Follows scikit-learn's ``MDS``: minimizes stress with the Guttman
    transform from ``n_init`` starts and keeps the start with the lowest
    stress. ``metric_mds=False`` runs Kruskal's non-metric SMACOF. Random
    starts come from mojolearn's own generator.

    Parameters
    ----------
    n_components : int, default 2
        Embedding dimension.
    metric_mds : bool, default True
        Metric (True) or non-metric (False) MDS.
    n_init : int, default 1
        Number of random starts (with ``init="random"``).
    init : {"random", "classical_mds"} or array-like, default "random"
        Starting configuration: random, the :class:`ClassicalMDS`
        embedding, or an array of shape (n_samples, n_components).
    max_iter : int, default 300
        Maximum SMACOF iterations per start.
    verbose : ignored
    eps : float, default 1e-6
        Convergence tolerance on the stress.
    n_jobs : ignored
    random_state : int or None, default None
        Seed for random starts; None means 0.
    metric : str, default "euclidean"
        ``"precomputed"`` (X is a dissimilarity matrix) or a distance
        metric: ``"euclidean"``, ``"l2"``, ``"minkowski"`` (``p`` from ``metric_params``), ``"manhattan"``
        (``"cityblock"``, ``"l1"``), ``"chebyshev"`` (``"infinity"``) or
        ``"cosine"``; a callable or any other name raises
        NotImplementedError.
    metric_params : dict or None, default None
        Only ``{"p": ...}`` is supported.
    normalized_stress : bool or "auto", default "auto"
        Report normalized stress (Kruskal's stress-1). ``"auto"`` is True
        for non-metric MDS and False otherwise.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
    stress_ : float
    n_iter_ : int
    dissimilarity_matrix_ : ndarray of shape (n_samples, n_samples)
    n_features_in_ : int

    Raises
    ------
    ValueError
        For an unknown ``init``, or with ``metric_mds=False`` when
        ``n_samples**2`` exceeds ``2**31 - 1``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((300, 8), dtype=np.float32)
    >>> Y = mojolearn.MDS(n_components=2, random_state=0).fit_transform(X)
    """
    _parameters = ("n_components", "metric_mds", "n_init", "init", "max_iter", "eps", "random_state", "metric",
                   "normalized_stress", "numeric_mode",
                   "metric_params", "n_jobs", "verbose")

    def __init__(self, n_components=2, *, metric_mds=True, n_init=1, init="random", max_iter=300, verbose=0,
                 eps=1e-6, n_jobs=None, random_state=None, metric="euclidean", metric_params=None,
                 normalized_stress="auto", numeric_mode=None):
        self.n_components, self.metric_mds, self.n_init, self.init = n_components, metric_mds, n_init, init
        self.max_iter, self.verbose, self.eps, self.n_jobs = max_iter, verbose, eps, n_jobs
        self.random_state, self.metric, self.metric_params = random_state, metric, metric_params
        self.normalized_stress, self.numeric_mode = normalized_stress, numeric_mode

    def fit_transform(self, X, y=None, init=None):
        """Compute the embedding and return it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
            Data, or dissimilarities with ``metric="precomputed"``.
        y : ignored
        init : array-like of shape (n_samples, n_components) or None, default None
            Starting configuration; overrides the ``init`` parameter.

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _NATIVE(self, "family_mani_mds_fit_transform",
                       (_manifold_binding(self, "numeric_mode"), _backend.binding(_BINDING), X, init), _NATIVE_CONTEXT)

    def fit(self, X, y=None, init=None):
        """Compute the embedding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
        y : ignored
        init : array-like of shape (n_samples, n_components) or None, default None
            Starting configuration; overrides the ``init`` parameter.

        Returns
        -------
        self
        """
        return _NATIVE(self, "family_mani_mds_fit",
                       (_manifold_binding(self, "numeric_mode"), _backend.binding(_BINDING), X, init), _NATIVE_CONTEXT)
# LocallyLinearEmbedding's shift-invert route (`_lle_smallest`) and its
# policy run in the compiled owner (bindings/mani_decomp_native.mojo).


# Reference: scikit-learn `manifold/_locally_linear.py`:
# `barycenter_kneighbors_graph`, `barycenter_weights`, `null_space` (here
# the smallest right singular vectors of I - W, which are M's smallest
# eigenvectors), `locally_linear_embedding`, `transform`. M = (I - W)^T
# (I - W), its n_components + 1 smallest eigenvectors, the first dropped.
# method='ltsa', 'hessian' and 'modified' build sklearn's M as a stacked
# factor (M = B^T B) and take the same SVD route. eigen_solver='auto' with
# method='standard' past 200 rows (sklearn's ARPACK policy) takes
# `_lle_smallest`, shift-invert subspace iteration on the deflated factor;
# 'dense' (and every other case) the full one-sided Jacobi SVD.
class LocallyLinearEmbedding(_Base):
    """Locally linear embedding.

    Follows scikit-learn's ``LocallyLinearEmbedding``: reconstructs each
    point from its Euclidean nearest neighbors and finds the low-dimensional
    embedding that best preserves those reconstruction weights. Supports the
    standard, LTSA, Hessian and modified variants.

    Parameters
    ----------
    n_neighbors : int, default 5
        Neighbors per point; must be less than ``n_samples``.
    n_components : int, default 2
        Embedding dimension.
    reg : float, default 1e-3
        Regularization of the local weight solves.
    eigen_solver : {"auto", "dense"}, default "auto"
        ``"dense"`` always runs the full exact solve. ``"auto"`` with
        ``method="standard"``, more than 200 samples and ``n_components + 1 <
        10`` uses an iterative solver for the smallest eigenvectors;
        otherwise it runs the full solve. ``"arpack"`` raises
        NotImplementedError.
    tol : ignored
    max_iter : int, default 100
        Iteration limit of the iterative solver.
    method : {"standard", "ltsa", "hessian", "modified"}, default "standard"
        LLE variant. ``"hessian"`` needs ``n_neighbors > n_components *
        (n_components + 3) / 2``; ``"modified"`` needs ``n_neighbors >=
        n_components``.
    hessian_tol : float, default 1e-4
        Tolerance for ``method="hessian"``.
    modified_tol : float, default 1e-12
        Tolerance for ``method="modified"``.
    neighbors_algorithm : ignored
    random_state : int or None, default None
        Seed of the iterative solver's start; None means 0.
    n_jobs : ignored
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
    reconstruction_error_ : float
    n_features_in_ : int

    Raises
    ------
    ValueError
        For an unknown ``method`` or ``eigen_solver``, or ``n_neighbors``
        out of range for the method.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).random((500, 8), dtype=np.float32)
    >>> Y = mojolearn.LocallyLinearEmbedding(n_neighbors=10, n_components=2).fit_transform(X)
    """
    _parameters = ("n_neighbors", "n_components", "reg", "eigen_solver", "method", "random_state", "numeric_mode",
                   "hessian_tol", "max_iter", "modified_tol", "n_jobs", "neighbors_algorithm", "tol")

    def __init__(self, *, n_neighbors=5, n_components=2, reg=1e-3, eigen_solver="auto", tol=1e-6, max_iter=100,
                 method="standard", hessian_tol=1e-4, modified_tol=1e-12, neighbors_algorithm="auto",
                 random_state=None, n_jobs=None, numeric_mode=None):
        self.n_neighbors, self.n_components, self.reg, self.eigen_solver = n_neighbors, n_components, reg, eigen_solver
        self.tol, self.max_iter, self.method, self.hessian_tol = tol, max_iter, method, hessian_tol
        self.modified_tol, self.neighbors_algorithm, self.random_state = modified_tol, neighbors_algorithm, random_state
        self.n_jobs, self.numeric_mode = n_jobs, numeric_mode

    def fit(self, X, y=None):
        """Compute the embedding of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        return _NATIVE(self, "family_mani_lle_fit",
                       (_manifold_binding(self, "numeric_mode"), _backend.binding(_BINDING), X), _NATIVE_CONTEXT)

    def fit_transform(self, X, y=None):
        """Compute the embedding and return ``embedding_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _NATIVE(self, "family_mani_lle_fit_transform",
                       (_manifold_binding(self, "numeric_mode"), _backend.binding(_BINDING), X), _NATIVE_CONTEXT)

    def transform(self, X):
        """Embed new points.

        Each query is written as a weighted combination of its nearest
        training points, and the same weights are applied to their
        embeddings.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        ndarray of shape (n_queries, n_components), float32
        """
        return _NATIVE(self, "family_mani_lle_transform",
                       (_manifold_binding(self, "numeric_mode_"), _backend.binding(_BINDING), X), _NATIVE_CONTEXT)


def _manifold_binding(model, field):
    """The decomp binding of a manifold estimator's numeric mode (`field`:
    "numeric_mode" for a fit, the fitted "numeric_mode_" after one; None is
    the default mode). Binding acquisition only."""
    return _backend.binding(_BINDING, getattr(model, field, None))


# The compiled family owner, connected once at import: every public method
# of Isomap, ClassicalMDS, MDS and LocallyLinearEmbedding is one native call
# (bindings/mani_decomp_native.mojo). Its tables: the kit's matrix and device
# buffer classes and the per-binding broadcast zero (`_dev_one`'s cache), and
# the metric kinds.
from . import _native_runtime_manifold_bridge
_NATIVE, _NATIVE_CONTEXT = _native_runtime_manifold_bridge.connect()
_native_family_bridge.tables(family_mani_M=_M, family_mani_devbuf=_DevBuf, family_mani_dev_one=_DEV_ONE,
                             family_mani_pd_kind=dict(_PD_KIND))


# ================================================================ robust covariance
# Reference: scikit-learn `covariance/_robust_covariance.py`:
# `c_step`/`_c_step`, `select_candidates`, `fast_mcd`, `MinCovDet.fit`,
# `correct_covariance`, `reweight_covariance`. The random subsets are
# Philox permutations (a sort of counter draws, ties to the lower index)
# and every argsort breaks ties by index. chi2 quantiles by bisection of
# the incomplete gamma; one feature takes sklearn's 1-D shortcut.
class MinCovDet(_Base):
    """Minimum covariance determinant: a robust covariance estimator.

    Follows scikit-learn's ``MinCovDet`` (FastMCD), including the
    consistency correction and the reweighting step. Random subsets come
    from mojolearn's own generator and ties are broken by row index, so
    results can differ from scikit-learn's for the same seed.

    Parameters
    ----------
    store_precision : bool, default True
        Store ``precision_``; when False it is None.
    assume_centered : bool, default False
        Treat the data as centered (location fixed at zero).
    support_fraction : float or None, default None
        Fraction of points in the raw support. None uses
        ``ceil((n_samples + n_features + 1) / 2)`` points.
    random_state : int or None, default None
        Seed for the random subsets; None means 0.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Robust location (reweighted).
    covariance_ : ndarray of shape (n_features, n_features)
        Robust covariance (reweighted).
    precision_ : ndarray of shape (n_features, n_features) or None
    support_ : ndarray of shape (n_samples,), uint8
        1 for points used in the reweighted estimate.
    raw_location_ : ndarray of shape (n_features,)
    raw_covariance_ : ndarray of shape (n_features, n_features)
    raw_support_ : ndarray of shape (n_samples,), uint8
    dist_ : ndarray of shape (n_samples,)
        Squared Mahalanobis distances of the training points.
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).normal(size=(500, 3)).astype(np.float32)
    >>> mcd = mojolearn.MinCovDet(random_state=0).fit(X)
    >>> d = mcd.mahalanobis(X)
    """
    _parameters = ("store_precision", "assume_centered", "support_fraction", "random_state", "numeric_mode")

    def __init__(self, *, store_precision=True, assume_centered=False, support_fraction=None, random_state=None,
                 numeric_mode=None):
        self.store_precision, self.assume_centered = store_precision, assume_centered
        self.support_fraction, self.random_state, self.numeric_mode = support_fraction, random_state, numeric_mode

    # Every operation is one call into the native MinCovDet owner
    # (bindings/mcd_native.mojo). A covariance estimator has no transform,
    # so it has no fit_transform either (scikit-learn's TransformerMixin).
    @available_if(lambda self: hasattr(self, "transform"))
    def fit_transform(self, X, y=None):
        """Not available: this estimator has no ``transform``."""
        raise AttributeError(f"'{type(self).__name__}' object has no attribute 'fit_transform'")

    def fit(self, X, y=None):
        """Fit the robust location and covariance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        return _MCD_CALL(self, "fit", (X,), _MCD_CONTEXT)

    def mahalanobis(self, X):
        """Return squared Mahalanobis distances under the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,), float32

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        return _MCD_CALL(self, "mahalanobis", (X,), _MCD_CONTEXT)

    def get_precision(self):
        """Return the precision matrix (pseudo-inverse of ``covariance_``).

        Available even when ``store_precision=False``.

        Returns
        -------
        ndarray of shape (n_features, n_features), float32
        """
        return _MCD_CALL(self, "get_precision", (), _MCD_CONTEXT)

    def score(self, X, y=None):
        """Return the Gaussian log-likelihood of X under the fitted model.

        As scikit-learn's ``EmpiricalCovariance.score``: uses X's empirical
        covariance around ``location_`` and the fitted precision.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        float
        """
        return _MCD_CALL(self, "score", (X,), _MCD_CONTEXT)


class EllipticEnvelope(MinCovDet):
    """Outlier detection for Gaussian-distributed data.

    Follows scikit-learn's ``EllipticEnvelope``: fits a :class:`MinCovDet`
    and sets ``offset_`` at the ``contamination`` percentile of the negated
    training distances, so that about that fraction of training points is
    labeled an outlier.

    Parameters
    ----------
    store_precision : bool, default True
        Store ``precision_``.
    assume_centered : bool, default False
        Treat the data as centered.
    support_fraction : float or None, default None
        Fraction of points in the raw support; see :class:`MinCovDet`.
    contamination : float, default 0.1
        Expected fraction of outliers, in (0, 0.5].
    random_state : int or None, default None
        Seed for the random subsets; None means 0.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    offset_ : float
        Threshold on the decision function.
    location_, covariance_, precision_, support_, dist_
        As in :class:`MinCovDet`.

    Raises
    ------
    ValueError
        If ``contamination`` is not in (0, 0.5].

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> X = np.random.default_rng(0).normal(size=(500, 3)).astype(np.float32)
    >>> labels = mojolearn.EllipticEnvelope(contamination=0.05, random_state=0).fit(X).predict(X)
    """
    # scikit-learn tags, read by `_mode.sklearn_tags`
    _estimator_type = "outlier_detector"
    _parameters = ("store_precision", "assume_centered", "support_fraction", "contamination", "random_state",
                   "numeric_mode")

    def __init__(self, *, store_precision=True, assume_centered=False, support_fraction=None, contamination=0.1,
                 random_state=None, numeric_mode=None):
        super().__init__(store_precision=store_precision, assume_centered=assume_centered,
                         support_fraction=support_fraction, random_state=random_state, numeric_mode=numeric_mode)
        self.contamination = contamination

    def fit(self, X, y=None):
        """Fit the robust model and set ``offset_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        return _MCD_CALL(self, "fit", (X,), _MCD_CONTEXT)

    def score_samples(self, X):
        """Return the negated squared Mahalanobis distance of each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,), float64
        """
        return _MCD_CALL(self, "score_samples", (X,), _MCD_CONTEXT)

    def decision_function(self, X):
        """Return ``score_samples(X) - offset_``; negative means outlier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,), float64
        """
        return _MCD_CALL(self, "decision_function", (X,), _MCD_CONTEXT)

    def predict(self, X):
        """Label each sample 1 (inlier) or -1 (outlier).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,), int32
        """
        return _MCD_CALL(self, "predict", (X,), _MCD_CONTEXT)

    def fit_predict(self, X, y=None):
        """Fit the model, then return ``predict(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples,), int32
        """
        return _MCD_CALL(self, "fit_predict", (X,), _MCD_CONTEXT)

    # Lane cpu4-python: the x_metrics device program (`accuracy_fraction`:
    # exact match counts, the weight total its PairSum, the ratio binary64
    # on the device; the host column runs the same units) replaces the host
    # loop `x_decomp_accuracy`.
    def score(self, X, y, sample_weight=None):
        """Return the accuracy of ``predict(X)`` against labels y.

        As scikit-learn's ``EllipticEnvelope.score``: the (weighted) fraction
        of exact label matches.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            True labels (1 inlier, -1 outlier).
        sample_weight : array-like of shape (n_samples,) or None, default None

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If y or sample_weight has a different length than X.
        ZeroDivisionError
            If the weights sum to zero.
        """
        return _MCD_CALL(self, "elliptic_score", (X, y, sample_weight), _MCD_CONTEXT)


# ================================================================ implicit ALS
# Reference: the `implicit` library's `als.py` and
# `cpu/_als.pyx::least_squares` (exact solver) and `_least_squares_cg`
# (use_cg=True; x_decomp/cells.mojo `als_cg_row`, DEVIATION 5321). Factors
# start as Philox uniform draws * 0.01; each iteration solves every user row
# then every item row as one batched Cholesky per row (one thread per row,
# sums in a fixed order). The default stays use_cg=False, the exact solve,
# so existing fits keep their bits.
class AlternatingLeastSquares(_Base):
    """Implicit-feedback matrix factorization by alternating least squares.

    The method of Hu, Koren and Volinsky (2008), following the ``implicit``
    library: confidence is ``alpha * user_items``, and each iteration solves
    for every user's factors and then every item's. Unlike ``implicit``,
    ``use_cg`` defaults to False (the exact solve), and ``user_items`` is
    held dense (0 means no interaction; a ``scipy.sparse`` matrix is
    densified).

    Parameters
    ----------
    factors : int, default 100
        Number of latent factors.
    regularization : float, default 0.01
        L2 penalty on the factors.
    alpha : float, default 1.0
        Confidence scale applied to ``user_items``.
    iterations : int, default 15
        Number of alternating sweeps.
    use_cg : bool, default False
        Use ``implicit``'s conjugate-gradient solver (``cg_steps`` steps per
        row per half-sweep) instead of the exact Cholesky solve.
    calculate_training_loss : bool, default False
        Record ``training_loss_`` after every iteration (``implicit``'s loss).
    random_state : int or None, default None
        Seed of the starting factors; None means 0.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.
    cg_steps : int, default 3
        Conjugate-gradient steps per row with ``use_cg=True``; must be at
        least 1.

    Attributes
    ----------
    user_factors, user_factors_ : ndarray of shape (n_users, factors)
    item_factors, item_factors_ : ndarray of shape (n_items, factors)
    training_loss_ : list of float
        Only with ``calculate_training_loss=True``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> R = (np.random.default_rng(0).random((200, 50)) < 0.05).astype(np.float32)
    >>> als = mojolearn.AlternatingLeastSquares(factors=8, iterations=5, random_state=0).fit(R)
    >>> ids, scores = als.recommend(0, R, N=5)
    """
    _parameters = ("factors", "regularization", "alpha", "iterations", "use_cg", "cg_steps",
                   "calculate_training_loss", "random_state", "numeric_mode")

    def __init__(self, factors=100, regularization=0.01, alpha=1.0, iterations=15, use_cg=False,
                 calculate_training_loss=False, random_state=None, numeric_mode=None, cg_steps=3):
        self.factors, self.regularization, self.alpha, self.iterations = factors, regularization, alpha, iterations
        self.use_cg, self.calculate_training_loss = use_cg, calculate_training_loss
        self.random_state, self.numeric_mode, self.cg_steps = random_state, numeric_mode, cg_steps

    # fit, recommend and similar_items are one call each into the native
    # AlternatingLeastSquares owner (bindings/als_native.mojo).

    def fit(self, user_items, show_progress=False):
        """Fit user and item factors.

        Parameters
        ----------
        user_items : array-like of shape (n_users, n_items)
            Interaction strengths; 0 means no interaction. A
            ``scipy.sparse`` matrix is densified.
        show_progress : bool, default False
            Accepted for ``implicit`` compatibility; nothing is shown.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``use_cg=True`` and ``cg_steps < 1``.
        """
        return _ALS_CALL(self, "fit", (user_items,), _ALS_CONTEXT)

    # A recommender has no transform, so it has no fit_transform either.
    @available_if(lambda self: hasattr(self, "transform"))
    def fit_transform(self, X, y=None):
        """Not available: this estimator has no ``transform``."""
        raise AttributeError(f"'{type(self).__name__}' object has no attribute 'fit_transform'")

    def recommend(self, userid, user_items, N=10, filter_already_liked_items=True):
        """Return the top N items for one user.

        Items are ranked by the dot product of user and item factors, ties
        going to the lower item id.

        Parameters
        ----------
        userid : int
            Row of the user.
        user_items : array-like of shape (n_users, n_items) or (1, n_items), or None
            Interactions used to filter items the user already has: row
            ``userid``, or the single row given.
        N : int, default 10
            Number of items to return; must be non-negative.
        filter_already_liked_items : bool, default True
            Skip items with a nonzero entry in the user's row.

        Returns
        -------
        ids : ndarray of shape (<= N,), int32
        scores : ndarray of shape (<= N,), float32

        Raises
        ------
        ValueError
            If ``N`` is negative or ``user_items`` has no matching row.
        """
        return _ALS_CALL(self, "recommend", (userid, user_items, N, filter_already_liked_items), _ALS_CONTEXT)

    def similar_items(self, itemid, N=10):
        """Return the N items most similar to one item.

        Similarity is the cosine of the item factors. The item itself is
        included, as ``implicit`` returns it.

        Parameters
        ----------
        itemid : int
        N : int, default 10
            Must be non-negative.

        Returns
        -------
        ids : ndarray of shape (<= N,), int32
        scores : ndarray of shape (<= N,), float32
        """
        return _ALS_CALL(self, "similar_items", (itemid, N), _ALS_CONTEXT)


class SparseCoder(_SparseCoding):
    """Sparse coding against a fixed, precomputed dictionary.

    Follows scikit-learn's ``SparseCoder``; ``transform`` encodes each row
    with :func:`sparse_encode`. Nothing is learned, so ``fit`` does nothing.

    Parameters
    ----------
    dictionary : array-like of shape (n_components, n_features)
        Dictionary atoms, one per row.
    transform_algorithm : {"omp", "lasso_lars", "lasso_cd", "lars", "threshold"}, default "omp"
        Coding method. An unknown value raises ValueError at ``fit`` or
        ``transform``.
    transform_n_nonzero_coefs : int or None, default None
        Nonzero coefficients for ``"omp"`` and ``"lars"``.
    transform_alpha : float or None, default None
        Penalty or threshold; None uses 1.0.
    split_sign : bool, default False
        Return the positive and negative parts of the code as separate
        columns.
    n_jobs : ignored
    positive_code : bool, default False
        Force the code to be non-negative.
    transform_max_iter : int, default 1000
        Maximum iterations of the Lasso methods.
    numeric_mode : {"identical", "fast"} or None, default None
        Numeric mode for this estimator. None uses the package default
        (``"identical"`` unless changed). In ``identical`` mode, mojolearn returns
        the same bits on Apple, NVIDIA and AMD GPUs. ``"fast"`` is available on
        Apple GPUs only.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
    n_components_ : int
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> coder = mojolearn.SparseCoder(rng.random((5, 8), dtype=np.float32), transform_n_nonzero_coefs=2)
    >>> code = coder.transform(rng.random((100, 8), dtype=np.float32))
    """
    # stateless: the dictionary is a constructor parameter, so transform needs no fit
    _sklearn_requires_fit = False
    _parameters = ("dictionary", "transform_algorithm", "transform_n_nonzero_coefs", "transform_alpha",
                   "split_sign", "positive_code", "transform_max_iter", "numeric_mode",
                   "n_jobs")

    def __init__(self, dictionary, *, transform_algorithm="omp", transform_n_nonzero_coefs=None, transform_alpha=None,
                 split_sign=False, n_jobs=None, positive_code=False, transform_max_iter=1000, numeric_mode=None):
        self.dictionary, self.transform_algorithm = dictionary, transform_algorithm
        self.transform_n_nonzero_coefs, self.transform_alpha = transform_n_nonzero_coefs, transform_alpha
        self.split_sign, self.n_jobs, self.positive_code = split_sign, n_jobs, positive_code
        self.transform_max_iter, self.numeric_mode = transform_max_iter, numeric_mode

    # Nothing is learned: `transform` works without `fit` (scikit-learn tags
    # SparseCoder requires_fit=False). `__init__` stores the parameters
    # verbatim; every operation checks them and reads the dictionary, in
    # the native dictionary-learning owner.
    def transform(self, X):
        """Encode X as a sparse combination of the dictionary atoms.

        Uses ``transform_algorithm`` and its settings.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
            Twice as many columns when ``split_sign=True``.
        """
        return _DL_CALL(self, "coder_transform", (X,), _DL_CONTEXT)

    def inverse_transform(self, X):
        """Rebuild data from a code: ``code @ dictionary``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            A code, as returned by ``transform`` (with ``2 * n_components``
            columns when ``split_sign=True``).

        Returns
        -------
        ndarray of shape (n_samples, n_features), float32
        """
        return _DL_CALL(self, "coder_inverse", (X,), _DL_CONTEXT)

    def fit(self, X, y=None):
        """Check the parameters, record the dictionary's shape and return self
        (the dictionary is fixed; nothing is learned).

        Returns
        -------
        self
        """
        return _DL_CALL(self, "coder_fit", (X,), _DL_CONTEXT)

    def fit_transform(self, X, y=None):
        """Return ``transform(X)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples, n_components), float32
        """
        return _DL_CALL(self, "coder_transform", (X,), _DL_CONTEXT)


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


# The compiled IncrementalPCA owner, connected once at import.
from . import _native_ipca_bridge  # noqa: E402
_IPCA_CALL, _IPCA_CONTEXT = _native_ipca_bridge.connect_public(IncrementalPCA)
# The compiled johnson_lindenstrauss_min_dim owner.
from . import _native_jl_bridge  # noqa: E402
_JL_CALL, _JL_CONTEXT = _native_jl_bridge.connect_public()


# The native dictionary-learning owner (bindings/dict_learning_native.mojo):
# DictionaryLearning, MiniBatchDictionaryLearning, SparseCoder, SparsePCA,
# MiniBatchSparsePCA and sparse_encode, one call per public operation.
from ._native_decomp_kit_bridge import connect_public as _connect_decomp_kit  # noqa: E402
_DL_CALL, _DL_CONTEXT = _connect_decomp_kit("dict_learning_native_call", minibatch_sparse_pca_class=MiniBatchSparsePCA)


# The native PLS owner (bindings/pls_native.mojo): PLSRegression, PLSCanonical, CCA.
# `score` runs R^2 through the native metrics owner (family_mcls_r2) inside
# the same call; `_get_binding` resolves the metrics binding, as r2_score did.
from ._metrics_impl import _get_binding  # noqa: E402
_PLS_CALL, _PLS_CONTEXT = _connect_decomp_kit("pls_native_call", pls_regression_class=PLSRegression,
                                              pls_canonical_class=PLSCanonical,
                                              pls_metric_owner=_native_family_bridge.connect())


# The native MinCovDet owner (bindings/mcd_native.mojo): MinCovDet, EllipticEnvelope.
_MCD_CALL, _MCD_CONTEXT = _connect_decomp_kit("mcd_native_call", elliptic_envelope_class=EllipticEnvelope)


# The native AlternatingLeastSquares owner (bindings/als_native.mojo).
_ALS_CALL, _ALS_CONTEXT = _connect_decomp_kit("als_native_call")


# The native linear-algebra owner (bindings/linalg_decomp_native.mojo):
# randomized_svd and solve.
_LINALG_CALL, _LINALG_CONTEXT = _connect_decomp_kit("linalg_decomp_native_call")


# The compiled owner of the decomposition estimators whose public operations
# are one native call each (bindings/decomp_native.mojo): FastICA, ...
from . import _native_decomp_bridge  # noqa: E402
_DECOMP_CALL, _DECOMP_CONTEXT = _native_decomp_bridge.connect_public({
    "FastICA": FastICA, "NMF": NMF, "FactorAnalysis": FactorAnalysis,
    "LatentDirichletAllocation": LatentDirichletAllocation,
    "GaussianRandomProjection": GaussianRandomProjection, "SparseRandomProjection": SparseRandomProjection})

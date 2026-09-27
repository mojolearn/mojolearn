# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE PREP LANE'S PUBLIC DOOR (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).

Owned by the `prep` expansion lane. `mojolearn/__init__.py` imports this
module and makes every name in `__all__` public as `mojolearn.<name>`; a name
that is already public is refused at import. Put the classes here, or import
them here from the lane's own modules. Two optional hooks route a SAVED model
to the CPU (`mojolearn.host_model`); `_classical_host.py` merges them when it is
first imported, after the package, so both may rely on every module existing:

  CLASSICAL_HOST_BASENAMES     {"_mojolearn_x_prep": "_mojolearn_x_prep_host"}
  def classical_host_formats() -> {"mojolearn-<format>-1": {"Estimator": HostEstimator}}
                               import `_classical_host` INSIDE it and subclass
                               its `_HostBound`; never import it at module level

HOW THE CLASSES COMPUTE. Every number these estimators produce comes out of
ONE binding entry, `x_prep_run`, which runs a PROGRAM of units
(x_prep/common.mojo) over one float32 arena: the GPU binding
(`_mojolearn_x_prep`) launches a thread per unit, the host binding
(`_mojolearn_x_prep_host`, what `_backend.binding` returns on a CPU-only
install) runs the same units in a loop. Python here only lays out the arena,
lists the stages and reads results back; the only arithmetic it does is
integer bookkeeping and IEEE basic operations on scalar parameters.
"""
import array
import ctypes
import numbers

from . import _backend
from ._array import Array
from ._buffer import as_f32_c, addr_ro
from ._labels import encode_labels, decode_labels

__all__ = ["RobustScaler", "MaxAbsScaler"]

_BINDING = "_mojolearn_x_prep"

#: op name -> id; x_prep/units.mojo `run_unit` holds the same table.
_OPS = dict(
    sort_cols=0, col_stats=1, quantile=2, affine=3, scale_params=4, unique_cols=5, mode_cols=6,
    lookup=7, count_neg=8, onehot=9, i2f=10, f2i=11, binarize=12, matmul=13, row_softmax=14,
    row_argmax=15, class_stats=16, center_rows=17, eigh=18, where_neg=19,
)
_PARAMS = 14
_NONE = -1


def _prep_binding(mode):
    return _backend.binding("_mojolearn_x_prep", mode)


class _Prog:
    """One program: an arena layout, the inputs copied into it, and stages."""

    def __init__(self):
        self.size = 0
        self._inputs = []
        self._stages = []
        self.arena = None

    def alloc(self, n):
        off = self.size
        self.size += max(int(n), 0)
        return off

    def put(self, arr):
        """A float32 C-contiguous Array (or anything as_f32_c takes) -> offset."""
        if not (isinstance(arr, Array) and arr.dtype == "<f4" and arr._has_order("C")):
            arr = as_f32_c(arr, ndim=None, name="input")[0]
        off = self.alloc(arr.size)
        self._inputs.append((off, arr, "f"))
        return off

    def put_list(self, values):
        return self.put(Array.from_list([float(v) for v in values] or [0.0], "<f4"))

    def put_scalar(self, value):
        return self.put_list([value])

    def put_codes(self, codes):
        """int32 codes -> offset of their float values (an i2f stage)."""
        if not (isinstance(codes, Array) and codes.dtype == "<i4"):
            codes = Array.from_list([int(c) for c in codes], "<i4")
        bits = self.alloc(codes.size)
        self._inputs.append((bits, codes, "i"))
        out = self.alloc(codes.size)
        self.stage("i2f", codes.size, bits, out)
        return out

    def stage(self, op, total, *params):
        if len(params) > _PARAMS:
            raise ValueError("x_prep: too many stage parameters")
        self._stages.append([_OPS[op], int(total)] + [int(v) for v in params] + [0] * (_PARAMS - len(params)))

    def run(self, mode):
        arena = array.array("f", bytes(4 * max(self.size, 1)))
        base = arena.buffer_info()[0]
        for off, arr, _ in self._inputs:
            if arr.size:
                ctypes.memmove(base + 4 * off, addr_ro(arr, name="input"), 4 * arr.size)
        prog = array.array("i", [v for s in self._stages for v in s] or [0])
        _prep_binding(mode).x_prep_run(base, self.size, prog.buffer_info()[0], len(self._stages))
        self.arena = arena
        return self

    def get(self, off, shape):
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (int(shape),)
        n = 1
        for s in shape:
            n *= s
        return Array._owned(self.arena[off:off + n], shape, "<f4", "C")

    def get_i32(self, off, shape):
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (int(shape),)
        n = 1
        for s in shape:
            n *= s
        store = array.array("i")
        store.frombytes(self.arena[off:off + n].tobytes())
        return Array._owned(store, shape, "<i4", "C")

    def values(self, off, n):
        """Python floats of n arena entries (for integer bookkeeping)."""
        return list(self.arena[off:off + n])


def _mode():
    return _backend.default_mode()


def _x2d(X, name="X"):
    arr = as_f32_c(X, ndim=2, name=name)[0]
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"mojolearn: {name} must be a nonempty two-dimensional array")
    if arr.size > 2 ** 31 - 1:
        raise ValueError(f"mojolearn: {name} exceeds the native Int32 indexing bound")
    return arr


class _PrepBase:
    """sklearn's parameter protocol and fit_transform for the lane's classes."""
    _parameters = ()

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in self._parameters}

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self._parameters:
                raise ValueError(f"mojolearn: invalid parameter {k!r} for {type(self).__name__}")
            setattr(self, k, v)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)

    def _check_fitted(self):
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError(f"mojolearn: this {type(self).__name__} instance is not fitted yet")

    def _check_width(self, arr):
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(f"mojolearn: X has {arr.shape[1]} features, {type(self).__name__} "
                             f"was fitted with {self.n_features_in_}")


def _affine(mode, X, center, scale):
    """(X - center) / scale on the device (either may be None)."""
    arr = _x2d(X)
    n, d = arr.shape
    pr = _Prog()
    xo = pr.put(arr)
    co = pr.put(center) if center is not None else _NONE
    so = pr.put(scale) if scale is not None else _NONE
    out = pr.alloc(n * d)
    pr.stage("affine", n * d, xo, n * d, d, co, so, out)
    return pr.run(mode).get(out, (n, d))


# ---------------------------------------------------------------- scalers
class RobustScaler(_PrepBase):
    """sklearn.preprocessing.RobustScaler: center by the median, scale by the
    quantile range (numpy's linear percentile over the non-NaN entries; NaN is
    ignored in fit and kept in transform). Float32 throughout; a scale below
    10 * float32 eps is one (`_handle_zeros_in_scale`). `unit_variance=True`
    is refused by name."""
    _parameters = ("with_centering", "with_scaling", "quantile_range", "copy", "unit_variance")

    def __init__(self, *, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True,
                 unit_variance=False):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy
        self.unit_variance = unit_variance

    def fit(self, X, y=None):
        if self.unit_variance:
            raise NotImplementedError("mojolearn: RobustScaler(unit_variance=True) is not implemented")
        lo, hi = (float(v) for v in self.quantile_range)
        if not 0 <= lo <= hi <= 100:
            raise ValueError(f"mojolearn: invalid quantile range {self.quantile_range!r}")
        arr = _x2d(X)
        n, d = arr.shape
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        so = pr.alloc(n * d)
        st = pr.alloc(6 * d)
        qf = pr.put_list([lo / 100.0, 0.5, hi / 100.0])
        q = pr.alloc(3 * d)
        center = pr.alloc(d)
        scale = pr.alloc(d)
        pr.stage("sort_cols", d, xo, n, d, so, 0)
        pr.stage("col_stats", d, xo, n, d, st)
        pr.stage("quantile", 3 * d, so, n, d, qf, 3, q, st)
        pr.stage("scale_params", d, q, 3, d, center, scale, 0, 0, 2, 1)
        pr.run(mode)
        self.center_ = pr.get(center, d) if self.with_centering else None
        self.scale_ = pr.get(scale, d) if self.with_scaling else None
        self.numeric_mode_, self.n_features_in_ = mode, d
        return self

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        return _affine(self.numeric_mode_, arr, self.center_, self.scale_)


class MaxAbsScaler(_PrepBase):
    """sklearn.preprocessing.MaxAbsScaler: X / max|X| per column over the
    non-NaN entries (NaN kept in transform); a zero column scales by one."""
    _parameters = ("copy",)

    def __init__(self, *, copy=True):
        self.copy = copy

    def fit(self, X, y=None):
        arr = _x2d(X)
        n, d = arr.shape
        mode = _mode()
        pr = _Prog()
        xo = pr.put(arr)
        st = pr.alloc(6 * d)
        scale = pr.alloc(d)
        pr.stage("col_stats", d, xo, n, d, st)
        pr.stage("scale_params", d, st + 5 * d, 1, d, _NONE, scale, 1)
        pr.run(mode)
        self.max_abs_ = pr.get(st + 5 * d, d)
        self.scale_ = pr.get(scale, d)
        self.numeric_mode_, self.n_features_in_, self.n_samples_seen_ = mode, d, n
        return self

    def transform(self, X):
        self._check_fitted()
        arr = _x2d(X)
        self._check_width(arr)
        return _affine(self.numeric_mode_, arr, None, self.scale_)

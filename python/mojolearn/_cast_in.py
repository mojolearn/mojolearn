# SPDX-License-Identifier: Apache-2.0
"""The input descriptor of a user's array for the autograd entries."""
# Engineering note (lane ag-thin-glue, 2026-10-19): the Sequential and
# ops.Session entries take the user's array as it is. Python checks the
# arguments and passes [address, NumPy dtype number, item size, rows, cols,
# row stride, column stride]; Mojo casts to float32 on the executor
# (core/cast_in.mojo, autograd/cast_ops.mojo). No astype, ascontiguousarray
# or other copy happens here.
from ._optional_numpy import require_numpy
np = require_numpy('_cast_in')


def as_input(x):
    """`x` itself when it is an ndarray; otherwise the ndarray NumPy builds
    from it (a list or a scalar has no buffer to pass), with no dtype cast."""
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)  # cpu-route: a Python sequence or scalar becomes a buffer, no cast


def describe(x, name):
    """The descriptor of ndarray `x` (0-, 1- or 2-D, numeric, C or Fortran
    order). A 1-D array of n is (n, 1), a scalar (1, 1). The caller keeps
    `x` alive across the call into Mojo."""
    dt = x.dtype
    if dt.kind not in "biuf" or (dt.kind == "f" and dt.itemsize not in (4, 8)):
        raise TypeError(f"{name} must be a numeric array (bool, integer, float32 or float64), "
                        f"got dtype {dt}")
    if not dt.isnative:
        raise TypeError(f"{name} must be in native byte order, got dtype {dt.str}")
    if x.ndim > 2:
        raise ValueError(f"{name} must have at most 2 dimensions, got {x.ndim}")
    if x.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not (x.flags.c_contiguous or x.flags.f_contiguous):
        raise ValueError(f"{name} must be C- or Fortran-contiguous; pass numpy.ascontiguousarray({name})")
    size = dt.itemsize
    if x.ndim == 0:
        rows, cols, s0, s1 = 1, 1, size, size
    elif x.ndim == 1:
        rows, cols, s0, s1 = x.shape[0], 1, x.strides[0], size
    else:
        rows, cols = x.shape
        s0, s1 = x.strides
    return [x.ctypes.data, dt.num, size, rows, cols, s0, s1]

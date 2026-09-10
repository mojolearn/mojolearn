# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEPRECATED compatibility shim over `_buffer.py` (DEVIATION 2309).

This module was the NumPy-era boundary. Its job moved to `_buffer.py`
(addresses, conversions) and `_array.py` (the `Array` container) on branch
numpy-free-0.7; the old names are kept here, with their old positional
signatures and their old messages, so every `_*_impl.py` keeps importing
`from ._arrays import _addr, _addr_ro, as_f32_c, as_f32_colmajor` during the
migration. New code imports `_buffer` directly. Delete this file when the
last importer is gone (`grep -n '_arrays' python/mojolearn/*.py`).

Differences a migrating caller sees:
- `as_f32_c` / `as_f32_colmajor` return `Array`s, not ndarrays; `np.asarray`
  over one is zero-copy for a caller that has NumPy.
- `as_f32_colmajor` still returns the 3-tuple `(array, flat, copied)`;
  `flat` is the 1-D storage-order view `_buffer.as_f32_colmajor` no longer
  hands out separately (it is `array._flat()`).
"""

from . import _buffer
from ._array import Array  # noqa: F401  (re-exported for migrating callers)


def _addr(a):
    """The address of a buffer about to be WRITTEN; a read-only buffer is
    refused with the message the old module used."""
    return _buffer.addr(a, name="output buffer")


def _addr_ro(a):
    """The address of a buffer that will only be read."""
    return _buffer.addr_ro(a, name="input buffer")


def as_f32_c(x, name):
    """`(Array, copied)`: a C-contiguous float32 2-D Array of `x`."""
    return _buffer.as_f32_c(x, ndim=2, name=name)


def as_f32_colmajor(x, name):
    """`(Array, flat, copied)`: an F-order float32 2-D Array of `x`, its
    flat column-major view over the SAME buffer (`flat[f * n_rows + r] ==
    array[r, f]`), and whether a copy was taken. Holding either keeps the
    buffer alive."""
    a, copied = _buffer.as_f32_colmajor(x, name=name)
    return a, a._flat(), copied

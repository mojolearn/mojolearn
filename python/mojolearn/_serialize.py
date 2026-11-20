# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Deterministic npz save and load for fitted models, without NumPy
(DEVIATION 2302).

Every helper here is one call into the compiled archive codec
(`bindings/archive_npz_door.mojo`), the same native ZIP/NPY reader and writer
the public `save` / `load` methods use. Floats travel as raw bytes, never as
decimal text; `write_npz` pins member timestamps to the zip epoch and member
order to sorted names, so bit-identical arrays give a bit-identical file that
`np.load` reads unchanged.

`read_npz` returns a dict whose numeric members are `Array`s. A string member
(`'<U..'`) comes back as a `str` when it holds exactly one element and as a
(nested) list of `str` otherwise; a bytes member (`'|S..'`) likewise as
`bytes` / list of `bytes`; a bool member (`'|b1'`) as a `'<u1'` Array of 0/1
(DEVIATION 2308). `exact()` refuses dtype mismatches, and `scalar_str()` reads
a one-element string field.
"""

from __future__ import annotations

from ._array import normalize_dtype
from . import _native_archive_bridge


def _npz(verb, *args):
    # glue: one native codec call; connect() is cached after the first call.
    entry, context = _native_archive_bridge.connect()
    return entry(None, "archive_npz", (verb, *args), context)


def encode_npy(value, *, c_order=False):
    """The complete npy bytes for `value`: what `numpy.lib.format.write_array`
    would write, including `fortran_order: True` for an F-order block
    unless `c_order` asks for the `write_npz` C-order linearization."""
    return _npz("encode", value, c_order)


def write_npy(fp, value, *, c_order=False):
    """Write `encode_npy(value)` to the binary file `fp` (header and payload
    as two writes, no joined copy)."""
    _npz("write_npy", fp, value, c_order)


def _parse_header(data):
    """`(descr, fortran_order, shape, data_offset)` from the start of an
    npy byte string. Versions 1.0, 2.0 and 3.0 are read."""
    return _npz("header", data)


def decode_npy(data):
    """An Array (numeric), str / list of str (`'<U'`), or bytes / list of
    bytes (`'|S'`) from complete npy bytes."""
    return _npz("decode", data)


def write_npz(path, arrays):
    """Write `arrays` (a dict of name to array-like) to `path` as an
    uncompressed npz whose bytes depend only on the array contents."""
    return _npz("write", path, arrays)


def read_npz(path, expected_format):
    """Load an npz written by `write_npz` (or `np.savez`) into a dict,
    checking its `format` tag against `expected_format` (a str or a tuple of
    the formats the caller reads). Pickle never enters."""
    return _npz("read", path, expected_format)


def peek_npz(path, names):
    """Decode only named members, for loader routing, not model loading.
    The selected loader must still call :func:`read_npz`."""
    return _npz("peek", path, names)


def scalar_str(arrays, name):
    """A string field as a plain str. Scalars are stored as one-element
    arrays because the npy writer promotes 0-d to 1-d."""
    return _npz("scalar_str", arrays, name)


def exact(arrays, name, dtype):
    """`arrays[name]` with its dtype REQUIRED to match, never cast. A cast
    on load could silently change bits, which is the one failure a model
    file must not have. `dtype` is a typestr (`'<i4'`), a NumPy dtype or
    scalar type when the caller has NumPy, or `float` / `int`. Returns a
    C-order Array."""
    return _npz("exact", arrays, name, normalize_dtype(dtype))

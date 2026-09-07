# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Readings of a `_buffer.Buf` that several surfaces share (DEVIATION 2437).

`_buffer.view(obj)` hands back a `Buf` -- format, shape, strides, readonly,
c_contiguous -- for ANY object that supports the buffer protocol: a NumPy
array, an `array.array`, a `mojolearn.Array`, a `bytes`. The neural,
time-series and linalg surfaces all need the same few pure-Python readings
of it that `_buffer.py` does not carry:

    is this format native float32 / a 4-byte integer / any integer
    what dtype NAME to print in a refusal ("float64", "bfloat16", ...)
    how many elements a shape holds
    a byte-moving primitive on raw addresses (`memcopy`, `memzero`)
    the LITTLE-ENDIAN bytes of a buffer, for the on-disk checkpoint forms
    a flat native memoryview over a contiguous buffer, for the one
    permitted per-element operation (a memoryview slice copy)

Nothing here imports numpy. A `Buf` is a context manager around
`PyObject_GetBuffer` / `PyBuffer_Release`, so holding one past its `with`
block pins the exporter (a NumPy array with a live export cannot be
resized and cannot have its `writeable` flag flipped); `probe` therefore
copies the six fields out and releases the buffer before returning, and
every surface reads the copy.

The natural home for all of this is `_buffer.py`. It lives here so that six
modules do not carry six copies while that file belongs to another lane;
folding it in is a one-move refactor with no behavior change.
"""

import array
import ctypes
import sys
from collections import namedtuple

from ._array import Array
from ._buffer import addr_ro, view


def _storage_view(obj):
    """A typed memoryview over `obj`'s storage. An `Array` is read through
    its own storage view (`_mv`, the way `_buffer.all_finite` reads it):
    `memoryview(Array)` needs `__buffer__`, which CPython consults only
    from 3.12, and this package's floor is 3.10. Every other exporter goes
    through `memoryview` itself."""
    if isinstance(obj, Array):
        return obj._mv
    return memoryview(obj)

#: The byte-order marks that name the HOST's order, and so leave a struct
#: format code meaning what its bare letter means.
_NATIVE_MARKS = ("@", "=", "<" if sys.byteorder == "little" else ">")

#: struct code -> dtype name, for refusal messages. `l`/`L` depend on the
#: platform's `long` and are resolved by itemsize in `dtype_name`.
_FORMAT_NAMES = {
    "f": "float32", "d": "float64", "e": "float16", "g": "longdouble",
    "?": "bool", "b": "int8", "B": "uint8", "h": "int16", "H": "uint16",
    "i": "int32", "I": "uint32", "q": "int64", "Q": "uint64",
    "n": "intp", "N": "uintp", "c": "bytes",
}

#: `Array.dtype` typestr -> dtype name.
_TYPESTR_NAMES = {
    "<f4": "float32", "<f8": "float64", "<f2": "float16", "<i4": "int32",
    "<i8": "int64", "<u4": "uint32", "<u1": "uint8", "<i2": "int16",
    "<u2": "uint16", ">f4": "float32", ">f8": "float64", ">i4": "int32",
    ">i8": "int64", ">u4": "uint32", "|u1": "uint8", "|i1": "int8",
}

_INTEGER_CODES = "bBhHiIlLqQnN"

Probe = namedtuple(
    "Probe", "ndim shape format itemsize readonly c_contiguous nbytes")


def probe(obj):
    """The six facts about `obj`'s buffer, with the buffer RELEASED before
    this returns. Raises `TypeError` (from `_buffer.view`) when `obj` does
    not support the buffer protocol."""
    with view(obj) as b:
        return Probe(int(b.ndim), tuple(int(s) for s in b.shape),
                     str(b.format), int(b.itemsize), bool(b.readonly),
                     bool(b.c_contiguous), int(b.nbytes))


def base_format(fmt):
    """`fmt` with a byte-order mark that names the host's order removed.
    A mark naming the OTHER order is kept, so the result never reads as
    native by accident."""
    if len(fmt) > 1 and fmt[0] in _NATIVE_MARKS:
        return fmt[1:]
    return fmt


def is_native_f32(fmt):
    return base_format(fmt) == "f"


def is_integer(fmt):
    return base_format(fmt) in _INTEGER_CODES


def is_int32(fmt, itemsize):
    """A 4-byte signed integer: `i`, or `l` where `long` is 4 bytes (the
    format NumPy reports for int32 on such platforms)."""
    return base_format(fmt) in ("i", "l") and itemsize == 4


def dtype_name(obj, pb):
    """The name to print in a refusal: NumPy's own `dtype.name` when the
    object carries one (so an ml_dtypes `bfloat16` is named as such), the
    `Array` typestr's name, else the struct code's."""
    dt = getattr(obj, "dtype", None)
    name = getattr(dt, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(dt, str) and dt in _TYPESTR_NAMES:
        return _TYPESTR_NAMES[dt]
    code = base_format(pb.format)
    if code in ("l", "L"):
        signed = code == "l"
        return ("int" if signed else "uint") + str(8 * pb.itemsize)
    return _FORMAT_NAMES.get(code, repr(pb.format))


def nelems(shape):
    n = 1
    for s in shape:
        n *= int(s)
    return n


def memcopy(dst_addr, src_addr, nbytes):
    """`memmove` on raw addresses. Both buffers must be alive and
    contiguous for `nbytes`; the caller holds the owning objects."""
    if nbytes:
        ctypes.memmove(int(dst_addr), int(src_addr), int(nbytes))


def memzero(dst_addr, nbytes):
    """All-zero bytes, which is `+0.0` for float32 and `0` for int32."""
    if nbytes:
        ctypes.memset(int(dst_addr), 0, int(nbytes))


def le_bytes(obj, typecode):
    """The C-order bytes of `obj` in LITTLE-ENDIAN order whatever the host:
    the JSON/hex checkpoint forms store `<f4` / `<i4`, exactly the bytes
    `np.asarray(value, dtype='<f4', order='C').tobytes()` used to emit.
    On a little-endian host this is the buffer's bytes unchanged; on a
    big-endian host every 4-byte group is swapped (`typecode` is the
    `array.array` code of the element, 'f' or 'i')."""
    raw = _storage_view(obj).tobytes()
    if sys.byteorder == "big":
        a = array.array(typecode)
        a.frombytes(raw)
        a.byteswap()
        raw = a.tobytes()
    return raw


def flat_view(obj, code):
    """A flat 1-D memoryview of native single-letter format `code` over a
    C-contiguous buffer, whatever shape or format its exporter reports:
    the memoryview is cast to bytes and back, which the buffer protocol
    permits only for contiguous data. Slices of it (with a step) are the
    contract's permitted per-element operation. Only C-order storage is
    ever handed here (an F-order Array's storage view is column-major and
    would silently give the wrong flat order)."""
    mv = _storage_view(obj)
    if mv.ndim == 1 and mv.format == code:
        return mv
    return mv.cast("B").cast(code)


def ctypes_view(obj, ctype, count, *, name):
    """A ctypes array over `obj`'s memory, for the rare exporter whose
    memoryview cannot be cast (a non-native format string). Slicing with
    a step yields a list; slice assignment accepts one."""
    return (ctype * int(count)).from_address(addr_ro(obj, name=name))


def strided_rows(src, offset, count, n_rows, row_len, dst):
    """Un-interleave a TIME-MAJOR float32 block into `dst`, a C-contiguous
    `(n_rows, row_len)` float32 buffer: element `[s + i * n_rows]` of
    `src[offset : offset + count]` lands at `dst[s, i]`. That is exactly
    cuML's `.reshape((n_rows, row_len), order="F")`, done as `n_rows`
    strided memoryview slice copies (C-level element loops) rather than a
    Python element loop. Falls back to ctypes slicing when an exporter's
    format cannot be cast."""
    try:
        s_mv = flat_view(src, "f")
        d_mv = flat_view(dst, "f")
        for s in range(n_rows):
            d_mv[s * row_len:(s + 1) * row_len] = \
                s_mv[offset + s:offset + count:n_rows]
    except (TypeError, ValueError, NotImplementedError):
        s_ct = ctypes_view(src, ctypes.c_float, offset + count, name="src")
        d_ct = ctypes_view(dst, ctypes.c_float, n_rows * row_len, name="dst")
        for s in range(n_rows):
            d_ct[s * row_len:(s + 1) * row_len] = \
                s_ct[offset + s:offset + count:n_rows]
    return dst

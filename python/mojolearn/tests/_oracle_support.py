# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Local copies of the small product utilities the test oracles used: a TESTING TOOL, never product execution.

The oracle helpers under tests/ (`_legacy_*`, `_mamba_block_reference`, `_typed_input`, ...) imported these from
product-internal modules (`mojolearn._buffer`, `_lazy_out`, `_optional_numpy`, `_portable_math`, `_bufcheck`,
`_array`), so every internal rename or deletion broke every user at once (26 of the 69 smoke-2 failures of
2026-11-15). This module depends only on the standard library, NumPy and the public `mojolearn.Array`. Each copy
keeps the product's arithmetic and its refusal wording; where the product called a native helper whose result is
exactly defined (a finiteness scan, a correctly rounded sum, a fill, a byte copy) the copy computes the same result
with the standard library or NumPy.

Nothing in the package imports this module.
"""
import array
import ctypes
import math as _stdmath
import sys
from collections import namedtuple
from pathlib import Path

import mojolearn
from mojolearn import Array

# typestr -> array.array typecode of an Array's backing store (the product's `_array._CODE`).
CODE = {
    "<f4": "f", "<f8": "d", "<i4": "i", "<i8": "q",
    "<u4": "I", "<u1": "B", "<f2": "H",
    "<u2": "H", "<i1": "b",
}

# typestr -> the native dtype code the array helpers take (the product's `_array._NATIVE_CODE`).
NATIVE_CODE = {"<f4": 0, "<f8": 1, "<i4": 2, "<i8": 3, "<u4": 4, "<u1": 5}


# ------------------------------------------------------------ optional NumPy


def require_numpy(feature):
    """NumPy, or the product's ModuleNotFoundError wording (`mojolearn._optional_numpy.require_numpy`)."""
    try:
        import numpy
    except ModuleNotFoundError as exc:
        if exc.name != "numpy":
            raise
        raise ModuleNotFoundError(
            f"mojolearn: {feature} requires optional NumPy support. "
            'Install it with: python -m pip install "mojolearn[numpy]" '
            '("mojolearn[verify]" also includes NumPy).', name="numpy"
        ) from exc
    return numpy


# ------------------------------------------------------------ buffer views


class _PyBuffer(ctypes.Structure):
    """`Py_buffer` from `Include/cpython/object.h` (3.10 through 3.14)."""

    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.c_void_p),
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_char_p),
        ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
        ("internal", ctypes.c_void_p),
    ]


_PyBUF_FULL_RO = 0x11C
_PyBUF_FULL = 0x11D
# Private function pointers (`PyDLL.__getitem__` builds an uncached one), so no other library's argtypes reach them.
_get_buffer = ctypes.pythonapi["PyObject_GetBuffer"]
_get_buffer.argtypes = [ctypes.py_object, ctypes.POINTER(_PyBuffer), ctypes.c_int]
_get_buffer.restype = ctypes.c_int
_release_buffer = ctypes.pythonapi["PyBuffer_Release"]
_release_buffer.argtypes = [ctypes.POINTER(_PyBuffer)]
_release_buffer.restype = None


def _contiguity(shape, strides, itemsize):
    """(c_contiguous, f_contiguous) from strides, NumPy's rules."""
    if not shape:
        return True, True
    n = 1
    for s in shape:
        n *= s
    if n <= 1:
        return True, True
    if strides is None:
        return True, False
    c_ok, expect = True, itemsize
    for k in range(len(shape) - 1, -1, -1):
        if shape[k] != 1 and strides[k] != expect:
            c_ok = False
            break
        expect *= shape[k]
    f_ok, expect = True, itemsize
    for k in range(len(shape)):
        if shape[k] != 1 and strides[k] != expect:
            f_ok = False
            break
        expect *= shape[k]
    return c_ok, f_ok


class Buf:
    """A held buffer export (the product's `_buffer.Buf`): address, format, shape, strides, contiguity and the
    read-only flag, valid until `release` / `__exit__`. An `Array` is read through its public
    `__array_interface__` and `flags`."""

    def __init__(self, obj, *, writable=False, name="object"):
        self._released = True
        self._view = None
        self._obj = obj
        if isinstance(obj, Array):
            iface = obj.__array_interface__
            flags = obj.flags
            self.addr = int(iface["data"][0])
            self.nbytes = int(obj.nbytes)
            self.itemsize = int(obj.itemsize)
            self.ndim = int(obj.ndim)
            self.shape = tuple(obj.shape)
            self.strides = tuple(obj.strides)
            self.format = CODE[obj.dtype]
            self.readonly = not flags["WRITEABLE"]
            self.c_contiguous = bool(flags["C_CONTIGUOUS"])
            self.f_contiguous = bool(flags["F_CONTIGUOUS"])
        else:
            for attr in ("__cuda_array_interface__",):
                if hasattr(obj, attr):
                    raise TypeError(
                        "mojolearn: %s is a DEVICE array (it exports %s). mojolearn reads HOST memory only. Copy it "
                        "to the host first: cupy `x.get()`, torch `x.cpu()`, numba `x.copy_to_host()`."
                        % (name, attr))
            view = _PyBuffer()
            try:
                _get_buffer(obj, ctypes.byref(view), _PyBUF_FULL if writable else _PyBUF_FULL_RO)
            except TypeError:
                raise TypeError(f"mojolearn: {name} does not support the buffer protocol") from None
            except BufferError:
                _get_buffer(obj, ctypes.byref(view), _PyBUF_FULL_RO)
            self._view = view
            self._released = False
            self.addr = int(view.buf or 0)
            self.nbytes = int(view.len)
            self.itemsize = int(view.itemsize)
            self.ndim = int(view.ndim)
            fmt = view.format
            self.format = fmt.decode("ascii") if fmt else "B"
            self.readonly = bool(view.readonly)
            if self.ndim == 0:
                self.shape, self.strides = (), ()
            else:
                self.shape = (tuple(int(view.shape[k]) for k in range(self.ndim)) if view.shape
                              else (self.nbytes // max(self.itemsize, 1),))
                self.strides = tuple(int(view.strides[k]) for k in range(self.ndim)) if view.strides else None
            self.c_contiguous, self.f_contiguous = _contiguity(self.shape, self.strides, self.itemsize)
        if writable and self.readonly:
            self.release()
            raise ValueError(f"mojolearn: {name} is read-only, refusing to write to it")

    def release(self):
        if not self._released:
            self._released = True
            view, self._view = self._view, None
            if view is not None:
                _release_buffer(ctypes.byref(view))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


def view(obj, *, writable=False, name="object"):
    """A `Buf` over `obj` (the product's `_buffer.view`)."""
    return Buf(obj, writable=writable, name=name)


def addr(obj, *, name="output buffer"):
    """The address of a buffer a kernel WRITES; a read-only buffer is refused. The caller keeps `obj` alive."""
    with Buf(obj, writable=True, name=name) as b:
        return b.addr


def addr_ro(obj, *, name="input buffer"):
    """The address of a buffer a kernel only reads. The caller keeps `obj` alive."""
    with Buf(obj, writable=False, name=name) as b:
        return b.addr


_SIGNED = "bhilqn"
_UNSIGNED = "BHILQNP"


def typestr_of(buf):
    """A NumPy-style typestr for a `Buf` (the product's `_buffer.typestr_of`)."""
    fmt = buf.format
    size = buf.itemsize
    if fmt[:1] in "@=<>!|":
        if fmt[0] == ">" and size > 1:
            raise TypeError(f"mojolearn: big-endian buffer format {fmt!r} is not supported")
        fmt = fmt[1:]
    if fmt in ("f", "d", "e"):
        return {"f": "<f4", "d": "<f8", "e": "<f2"}[fmt]
    if len(fmt) == 1 and fmt in _SIGNED:
        return "<i1" if size == 1 else f"<i{size}"
    if len(fmt) == 1 and fmt in _UNSIGNED:
        return "<u1" if size == 1 else f"<u{size}"
    if fmt == "?":
        return "|b1"
    if fmt == "c":
        return "|S1"
    if fmt.endswith("s") and fmt[:-1].isdigit():
        return f"|S{fmt[:-1]}"
    if fmt.endswith("w") and fmt[:-1].isdigit():
        return f"<U{fmt[:-1]}"
    if fmt[:1] == "Z":
        raise ValueError(f"Complex data not supported: buffer format {fmt!r} is complex")
    raise TypeError(f"mojolearn: unsupported buffer format {fmt!r}")


# The product's `_bufcheck` probe and refusal names.
_NATIVE_MARKS = ("@", "=", "<" if sys.byteorder == "little" else ">")
_FORMAT_NAMES = {
    "f": "float32", "d": "float64", "e": "float16", "g": "longdouble",
    "?": "bool", "b": "int8", "B": "uint8", "h": "int16", "H": "uint16",
    "i": "int32", "I": "uint32", "q": "int64", "Q": "uint64",
    "n": "intp", "N": "uintp", "c": "bytes",
}
_TYPESTR_NAMES = {
    "<f4": "float32", "<f8": "float64", "<f2": "float16", "<i4": "int32",
    "<i8": "int64", "<u4": "uint32", "<u1": "uint8", "<i2": "int16",
    "<u2": "uint16", ">f4": "float32", ">f8": "float64", ">i4": "int32",
    ">i8": "int64", ">u4": "uint32", "|u1": "uint8", "|i1": "int8",
}
Probe = namedtuple("Probe", "ndim shape format itemsize readonly c_contiguous nbytes")


def probe(obj):
    """The facts about `obj`'s buffer, released before this returns; TypeError for a non-buffer."""
    with view(obj) as b:
        return Probe(int(b.ndim), tuple(int(s) for s in b.shape), str(b.format), int(b.itemsize),
                     bool(b.readonly), bool(b.c_contiguous), int(b.nbytes))


def base_format(fmt):
    if len(fmt) > 1 and fmt[0] in _NATIVE_MARKS:
        return fmt[1:]
    return fmt


def is_native_f32(fmt):
    return base_format(fmt) == "f"


def dtype_name(obj, pb):
    """The dtype name a refusal prints (the product's `_bufcheck.dtype_name`)."""
    dt = getattr(obj, "dtype", None)
    name = getattr(dt, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(dt, str) and dt in _TYPESTR_NAMES:
        return _TYPESTR_NAMES[dt]
    code = base_format(pb.format)
    if code in ("l", "L"):
        return ("int" if code == "l" else "uint") + str(8 * pb.itemsize)
    return _FORMAT_NAMES.get(code, repr(pb.format))


# ------------------------------------------------------------ constructors


def empty(shape, dtype):
    """A new zero-filled C-order Array (the product's `_buffer.empty`; also its `_lazy_out._empty_out`, whose
    opt-in memory map held the same zeros)."""
    return Array(shape, dtype)


zeros = empty
empty_out = empty


def _shape_tuple(shape):
    return tuple(int(v) for v in (shape if isinstance(shape, (tuple, list)) else (shape,)))


def _from_store(store, dtype, shape):
    shape = _shape_tuple(shape)
    if dtype in ("<f2", "<u2") and CODE[dtype] == "H":
        np = require_numpy("the oracle float16 / uint16 constructors")
        return Array.from_buffer(np.frombuffer(store, dtype=dtype).reshape(shape).copy())
    return Array.from_buffer(store).reshape(shape)


def full(shape, value, dtype):
    """A new C-order Array of `shape` holding `value` (the product's `_buffer.full`)."""
    shape = _shape_tuple(shape)
    n = 1
    for v in shape:
        n *= v
    store = array.array(CODE[dtype], [value]) * n if n else array.array(CODE[dtype])
    return _from_store(store, dtype, shape)


def frombytes(raw, dtype, shape):
    """An owned C-order Array holding a COPY of `raw` (the product's `_buffer.frombytes`)."""
    store = array.array(CODE[dtype])
    store.frombytes(bytes(raw))
    return _from_store(store, dtype, shape)


def all_finite(arr):
    """True when every element of a float32/float64 Array is finite (the product's `_buffer.all_finite`)."""
    if not isinstance(arr, Array) or arr.dtype not in ("<f4", "<f8"):
        raise TypeError("mojolearn: all_finite takes a float32 or float64 Array, got "
                        f"{getattr(arr, 'dtype', type(arr).__name__)!r}")
    if arr.size == 0:
        return True
    np = require_numpy("the oracle finiteness scan")
    return bool(np.isfinite(np.asarray(arr)).all())


# ------------------------------------------------------------ host math


class PortableMath:
    """The product's `_portable_math` (the repository-owned binary64 log/exp/sqrt of the shipped
    `libMojolearnMath`, loaded from the installed package directory), so an oracle's host arithmetic keeps the
    product's bits. `fsum` is the correctly rounded sum, as the product's `fsum_f64` is."""

    inf = float("inf")
    nan = float("nan")
    pi = float.fromhex("0x1.921fb54442d18p+1")

    def __init__(self):
        self._lib = None

    def _load(self):
        if self._lib is None:
            root = Path(mojolearn.__file__).resolve().parent
            path = (root / ".dylibs/libMojolearnMath.dylib" if sys.platform == "darwin"
                    else root / ".libs/libMojolearnMath.so")
            lib = ctypes.CDLL(str(path))
            for operation in ("sqrt", "log", "log2", "log10", "exp"):
                fn = getattr(lib, "mojolearn_" + operation)
                fn.argtypes = [ctypes.c_double]
                fn.restype = ctypes.c_double
            self._lib = lib
        return self._lib

    def _native(self, name, x):
        return getattr(self._load(), "mojolearn_" + name)(x)

    @staticmethod
    def isfinite(x):
        return _stdmath.isfinite(x)

    @staticmethod
    def isinf(x):
        return _stdmath.isinf(x)

    @staticmethod
    def isnan(x):
        return _stdmath.isnan(x)

    def sqrt(self, x):
        x = float(x)
        if x < 0.0:
            raise ValueError("math domain error")
        return self._native("sqrt", x)

    def log(self, x, base=None):
        x = float(x)
        if x <= 0.0:
            raise ValueError("math domain error")
        result = self._native("log", x)
        return result if base is None else result / self.log(base)

    def log2(self, x):
        x = float(x)
        if x <= 0.0:
            raise ValueError("math domain error")
        return self._native("log2", x)

    def exp(self, x):
        x = float(x)
        result = self._native("exp", x)
        if _stdmath.isinf(result) and _stdmath.isfinite(x):
            raise OverflowError("math range error")
        return result

    @staticmethod
    def fsum(values):
        return _stdmath.fsum(values)


portable_math = PortableMath()

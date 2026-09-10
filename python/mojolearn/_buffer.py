# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Buffer handling at the boundary, without NumPy (DEVIATION 2300).

The Mojo side takes RAW ADDRESSES. It borrows, never owns, and it holds
nothing after a call returns. That is only sound if the Python object owning
the memory is alive for the whole call, so every converter here returns the
`Array` alongside its address and every caller keeps that `Array` in a local
for the duration. `addr` and `addr_ro` are the one place an address is taken
and the one place to look when that contract is broken.

HOW AN ADDRESS IS TAKEN. `PyObject_GetBuffer` through `ctypes.pythonapi`,
with a `Py_buffer` struct laid out here field for field, and ALWAYS a
`PyBuffer_Release`. That is the same buffer protocol NumPy, `bytes`,
`bytearray`, `array.array`, `memoryview` and ctypes objects all export, so
one path serves every input. Contiguity is computed from the strides the
exporter reports, not trusted from a flag.

FLOAT32, NOT FLOAT64, AND THAT IS NOT NEGOTIABLE HERE. Metal has no float64
and neither does the Mojo side; every kernel in this library is float32. A
float64 input is converted, which COPIES. The copy is reported by the
`copied` flag rather than hidden, because on a 4,000,000 x 32 matrix it is
512 MB the caller did not ask for.

HOW A CONVERSION IS DONE. `array.array('f', <memoryview of doubles>)`: the
item setter is a C `(float)` cast of each double, which is one
round-to-nearest-even per element, exactly what NumPy's `astype(float32)`
does (the DEVIATION 1887 argument holds unchanged). It is a C loop over
Python float objects, roughly 40-80 ns per element, so it is slower than
NumPy's vectorized cast; it is never a per-element PYTHON loop. The one
place Python iterates is the outer loop of a layout change (one iteration
per column of a matrix, see `_array._reorder`). Integer inputs reach
float32 through a double first: exact for every |v| <= 2**53 (all of
int32, and every int64 a label or count could be), and possibly one ulp
off a direct cast beyond that (DEVIATION 2306, documented, not hidden).
"""

from __future__ import annotations

import array
import ctypes
import math
import mmap
import sys

from ._array import Array, _CODE, normalize_dtype

# ------------------------------------------------------------- Py_buffer


class _PyBuffer(ctypes.Structure):
    """`Py_buffer` from `Include/cpython/object.h` (identical on every
    CPython this library supports, 3.10 through 3.14)."""

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


PyBUF_WRITABLE = 0x0001
PyBUF_FORMAT = 0x0004
PyBUF_ND = 0x0008
PyBUF_STRIDES = 0x0010 | PyBUF_ND
PyBUF_INDIRECT = 0x0100 | PyBUF_STRIDES
PyBUF_FULL_RO = PyBUF_INDIRECT | PyBUF_FORMAT          # 0x11c
PyBUF_FULL = PyBUF_FULL_RO | PyBUF_WRITABLE            # 0x11d
_PyBUF_READ = 0x100
_PyBUF_WRITE = 0x200

_api = ctypes.pythonapi
_get_buffer = _api.PyObject_GetBuffer
_get_buffer.argtypes = [ctypes.py_object, ctypes.POINTER(_PyBuffer), ctypes.c_int]
_get_buffer.restype = ctypes.c_int
_release_buffer = _api.PyBuffer_Release
_release_buffer.argtypes = [ctypes.POINTER(_PyBuffer)]
_release_buffer.restype = None
_memoryview_from_memory = _api.PyMemoryView_FromMemory
_memoryview_from_memory.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int]
_memoryview_from_memory.restype = ctypes.py_object


def memory_at(addr, nbytes, *, writable):
    """A flat `'B'` memoryview over `nbytes` at `addr`. It holds NO
    reference to whatever owns that memory: the caller must."""
    if nbytes == 0:
        return memoryview(bytearray(0) if writable else b"")
    flag = _PyBUF_WRITE if writable else _PyBUF_READ
    return _memoryview_from_memory(addr, nbytes, flag)


class Buf:
    """A held buffer export: context manager around `PyObject_GetBuffer`
    and `PyBuffer_Release`. Fields are plain attributes read once at
    acquisition; the export (and so the address) is valid until `release`,
    `__exit__` or garbage collection, whichever comes first.
    """

    __slots__ = (
        "_view", "_obj", "_released", "addr", "nbytes", "itemsize", "ndim",
        "shape", "strides", "format", "readonly", "c_contiguous",
        "f_contiguous",
    )

    def __init__(self, obj, *, writable=False, name="object"):
        self._released = True
        self._obj = obj
        if isinstance(obj, Array):
            self._from_array(obj, name)
            return
        view = _PyBuffer()
        flags = PyBUF_FULL if writable else PyBUF_FULL_RO
        try:
            _get_buffer(obj, ctypes.byref(view), flags)
        except TypeError:
            raise TypeError(
                f"mojolearn: {name} does not support the buffer protocol"
            ) from None
        except BufferError:
            # The exporter refused the writable request; re-acquire read-only
            # so the failure below is the library's own message.
            _get_buffer(obj, ctypes.byref(view), PyBUF_FULL_RO)
        self._view = view
        self._released = False
        try:
            self._fill(view)
        except BaseException:
            self.release()
            raise
        if writable and self.readonly:
            self.release()
            raise ValueError(
                f"mojolearn: {name} is read-only, refusing to write to it"
            )

    def _fill(self, view):
        if view.suboffsets:
            for k in range(view.ndim):
                if view.suboffsets[k] >= 0:
                    raise TypeError(
                        "mojolearn: indirect (suboffset) buffers are not "
                        "supported"
                    )
        self.addr = int(view.buf or 0)
        self.nbytes = int(view.len)
        self.itemsize = int(view.itemsize)
        self.ndim = int(view.ndim)
        fmt = view.format
        self.format = fmt.decode("ascii") if fmt else "B"
        self.readonly = bool(view.readonly)
        if self.ndim == 0:
            self.shape = ()
            self.strides = ()
        else:
            if view.shape:
                self.shape = tuple(int(view.shape[k]) for k in range(self.ndim))
            else:
                # ndim 1 without a shape array: a simple byte-like buffer
                self.shape = (self.nbytes // max(self.itemsize, 1),)
            if view.strides:
                self.strides = tuple(int(view.strides[k]) for k in range(self.ndim))
            else:
                self.strides = None
        self.c_contiguous, self.f_contiguous = _contiguity(
            self.shape, self.strides, self.itemsize
        )

    def _from_array(self, arr, name):
        # Pure-Python classes cannot export the buffer protocol before
        # 3.12, so an Array is read from its backing store on every
        # supported Python (DEVIATION 2305). Pinning the store's memoryview
        # keeps an owned `array.array` from being resized underneath us.
        self._view = memoryview(arr._store) if arr._pin is None else arr._pin
        self._released = False
        self.addr = arr._addr
        self.nbytes = arr.nbytes
        self.itemsize = arr.itemsize
        self.ndim = arr.ndim
        self.shape = arr.shape
        self.strides = arr.strides
        self.format = _CODE[arr.dtype]
        self.readonly = arr._readonly
        flags = arr.flags
        self.c_contiguous = flags["C_CONTIGUOUS"]
        self.f_contiguous = flags["F_CONTIGUOUS"]

    def release(self):
        if not self._released:
            self._released = True
            view = self._view
            self._view = None
            if isinstance(view, _PyBuffer):
                _release_buffer(ctypes.byref(view))
            elif view is not None and isinstance(self._obj, Array) and self._obj._pin is None:
                view.release()

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

    def __repr__(self):
        return (f"Buf(addr=0x{self.addr:x}, format={self.format!r}, "
                f"shape={self.shape}, strides={self.strides}, "
                f"readonly={self.readonly})")


def _contiguity(shape, strides, itemsize):
    """(c_contiguous, f_contiguous) from strides, NumPy's rules: an empty
    or single-element block is both; a dimension of extent 1 does not
    constrain; `strides is None` means C-contiguous by definition."""
    if not shape:
        return True, True
    n = 1
    for s in shape:
        n *= s
    if n <= 1:
        return True, True
    if strides is None:
        return True, False
    c_ok = True
    expect = itemsize
    for k in range(len(shape) - 1, -1, -1):
        if shape[k] != 1 and strides[k] != expect:
            c_ok = False
            break
        expect *= shape[k]
    f_ok = True
    expect = itemsize
    for k in range(len(shape)):
        if shape[k] != 1 and strides[k] != expect:
            f_ok = False
            break
        expect *= shape[k]
    return c_ok, f_ok


# ---------------------------------------------------------------- formats

_SIGNED = "bhilqn"
_UNSIGNED = "BHILQNP"


def typestr_of(buf):
    """A NumPy-style typestr for a `Buf`, from its PEP 3118 format and its
    itemsize (the letter alone does not fix the width: NumPy exports int64
    as `'l'` on LP64 platforms and `'q'` elsewhere)."""
    fmt = buf.format
    size = buf.itemsize
    if fmt[:1] in "@=<>!|":
        if fmt[0] == ">" and size > 1:
            raise TypeError(
                f"mojolearn: big-endian buffer format {fmt!r} is not supported"
            )
        fmt = fmt[1:]
    if fmt in ("f", "d", "e"):
        return {"f": "<f4", "d": "<f8", "e": "<f2"}[fmt]
    if len(fmt) == 1 and fmt in _SIGNED:
        return ("|i1" if size == 1 else f"<i{size}")
    if len(fmt) == 1 and fmt in _UNSIGNED:
        return ("<u1" if size == 1 else f"<u{size}")
    if fmt == "?":
        return "|b1"
    if fmt == "c":
        return "|S1"
    if fmt.endswith("s") and fmt[:-1].isdigit():
        return f"|S{fmt[:-1]}"
    if fmt.endswith("w") and fmt[:-1].isdigit():
        return f"<U{fmt[:-1]}"
    raise TypeError(f"mojolearn: unsupported buffer format {fmt!r}")


def _iter_format(typestr):
    """The memoryview cast letter that makes a flat buffer of `typestr`
    iterate as Python scalars, or None when it cannot."""
    if typestr in _CODE:
        return _CODE[typestr] if typestr != "<f2" else "e"
    return {
        "|i1": "b", "<i2": "h", "<u2": "H", "<u8": "Q", "|b1": "?",
    }.get(typestr)


# ------------------------------------------------------------------ views


def view(obj, *, writable=False, name="object"):
    """A `Buf` over `obj`. Raises TypeError("mojolearn: <name> does not
    support the buffer protocol") for a non-buffer, and, with
    `writable=True`, ValueError("mojolearn: <name> is read-only, refusing
    to write to it") for a read-only one. Use as a context manager."""
    return Buf(obj, writable=writable, name=name)


def addr(obj, *, name):
    """The address of a buffer the callee WRITES. A read-only buffer is
    refused here with a ValueError, because handing it to a kernel is a
    segfault rather than an exception. The caller keeps `obj` alive."""
    if isinstance(obj, Array):
        if obj._readonly:
            raise ValueError(
                f"mojolearn: {name} is read-only, refusing to write to it"
            )
        return obj._addr
    with Buf(obj, writable=True, name=name) as b:
        return b.addr


def addr_ro(obj, *, name):
    """The address of a buffer that will only be read."""
    if isinstance(obj, Array):
        return obj._addr
    with Buf(obj, writable=False, name=name) as b:
        return b.addr


# ------------------------------------------------------------ conversion


def _materialize(obj, name):
    """`obj` as an Array (zero-copy when it already is one or exports a
    contiguous supported buffer), plus whether that copied."""
    if isinstance(obj, Array):
        return obj, False
    if isinstance(obj, (list, tuple)) or not _has_buffer(obj):
        if isinstance(obj, (str, bytes)):
            raise TypeError(
                f"mojolearn: {name} is a {type(obj).__name__}, not an array"
            )
        # Nested lists of Python scalars: infer like NumPy (any float ->
        # float64, else int64), so a later cast is the SAME single
        # rounding NumPy would do from the same intermediate.
        from ._array import _flatten
        shape, flat = _flatten(obj)
        if any(isinstance(v, float) for v in flat):
            return Array._from_flat(flat, shape, "<f8"), True
        for v in flat:
            if not isinstance(v, (int, bool)):
                raise TypeError(
                    f"mojolearn: {name} holds a {type(v).__name__}, not a number"
                )
        return Array._from_flat(flat, shape, "<i8"), True
    with Buf(obj, name=name) as b:
        ts = typestr_of(b)
        if b.ndim >= 1 and (ts in _CODE) and (b.c_contiguous or b.f_contiguous):
            pass  # zero-copy path below
        else:
            # Not directly representable (a strided view, an unusual
            # width, a bool buffer): copy to a C-order block of Python
            # scalars' natural dtype. `memoryview.tobytes()` linearizes
            # any strides in C order, in C.
            letter = _iter_format(ts)
            if letter is None:
                raise TypeError(
                    f"mojolearn: {name} has buffer format {b.format!r}, "
                    "which is not a numeric array"
                )
            raw = memoryview(obj).tobytes()  # C order whatever the strides
            shape = b.shape if b.ndim else (1,)
            target = _NATURAL.get(ts, ts)
            if ts == "<f2":
                # no memoryview.cast('e') before 3.12: struct does it
                import struct
                n = len(raw) // 2
                values = struct.unpack("<%de" % n, raw) if n else ()
                return Array._from_flat(list(values), shape, "<f2"), True
            flat = memoryview(raw).cast(letter)
            store = array.array(_CODE[target], flat)
            return Array._owned(store, shape, target, "C"), True
    return Array.from_buffer(obj), False


# Buffer dtypes that are not Array dtypes land on the narrowest Array dtype
# that holds every value exactly; a uint64 above 2**63 overflows int64 and
# is refused by array.array rather than wrapped.
_NATURAL = {"|b1": "<u1", "|i1": "<i4", "<i2": "<i4", "<u2": "<u4", "<u8": "<i8"}


def _has_buffer(obj):
    try:
        memoryview(obj).release()
        return True
    except TypeError:
        return False


def _convert(a, dtype, order):
    """`a` itself when it already has `dtype` and `order`, else a converted
    copy. Returns `(Array, copied)`.

    A float64 -> float32 conversion goes through the base binding's host
    converters when the binding is built (DEVIATION 2470, 2471): the flat
    cast, or ONE fused cast-and-transpose when a C-order source wants
    column-major output. Without the binding the pure-Python path below
    runs and produces the same bytes; the native path is a speed choice,
    never a different answer, and `tests/test_native_convert.py` holds the
    two to byte equality.
    """
    if a.dtype == dtype and a._has_order(order):
        # a (1, n) or (n, 1) block is both C- and F-contiguous; relabeling
        # it is free and is not a copy
        return a._as_order(order), False
    if a.dtype == "<f8" and dtype == "<f4" and a.size:
        out = _native_f64_to_f32(a, order)
        if out is not None:
            return out, True
    if a.dtype != dtype:
        a = a.astype(dtype)  # keeps a's order; the cast is done once
    return a._as_order(order), True


def _native_f64_to_f32(a, order):
    """`a` (float64) as a float32 Array in `order` through the native
    converters, or None when the binding is not built (the caller then
    takes the pure-Python path). The result always owns fresh storage."""
    if order == "F" and a.order == "C" and a.ndim == 2 and not a._both_orders():
        fn = _native("cast_colmajor_f64_to_f32")
        if fn is None:
            return None
        store = _output_store("f", a.size)
        rows, cols = a.shape
        fn(a._addr, store.buffer_info()[0], rows, cols)
        return Array._owned(store, a.shape, "<f4", "F")
    fn = _native("cast_f64_to_f32")
    if fn is None:
        return None
    # the flat cast keeps a's storage order; a relabel or reorder follows
    store = _output_store("f", a.size)
    fn(a._addr, store.buffer_info()[0], a.size)
    return Array._owned(store, a.shape, "<f4", a.order)._as_order(order)


class _AnonStore(mmap.mmap):
    """An owned block the native converters write into, backed by an
    ANONYMOUS MAPPING instead of an `array.array`.

    Every way of allocating an `array.array` from Python writes every byte
    once (a zero fill or a copy), and at 2,000,000 x 20 that pass costs
    about as much as the cast itself: 9.8 ms of a 16 ms total on the M4,
    against NumPy's 11.6 ms, which allocates uninitialized memory and
    touches each page for the first time inside its own copy loop. An
    anonymous mapping gets the same treatment from the kernel: the pages
    are zero by construction and materialize on the converter's first
    write, so there is no separate fill pass. Measured at 40,000,000
    elements: 13.0 ms end to end against NumPy's 12.8 ms.

    It quacks like the `array.array` the `Array` class expects: a typed
    buffer (`__buffer__`, Python 3.12+) and `buffer_info()`. Only the
    native converters build one; nothing else needs to know.
    """

    __slots__ = ("_code", "_n")

    def __new__(cls, code, n):
        return super().__new__(cls, -1, max(1, n * array.array(code).itemsize))

    def __init__(self, code, n):
        self._code = code
        self._n = n

    def __buffer__(self, flags):
        return mmap.mmap.__buffer__(self, flags).cast(self._code)

    def buffer_info(self):
        return _AnonStore._address(self), self._n

    @staticmethod
    def _address(m):
        raw = mmap.mmap.__buffer__(m, 0)
        try:
            return ctypes.addressof(ctypes.c_char.from_buffer(raw))
        finally:
            raw.release()


def _output_store(code, n):
    """Fresh storage for `n` elements of `code` for a native converter to
    fill: an `_AnonStore` where the interpreter lets a Python class export
    a buffer (3.12+), else the zero-filled `array.array` every other Array
    uses. The bytes written are identical either way; only the cost of
    getting the block differs."""
    if sys.version_info >= (3, 12):
        return _AnonStore(code, n)
    from ._array import _new_store
    return _new_store(code, n)


def _as_typed(obj, dtype, order, ndim, name):
    a, copied = _materialize(obj, name)
    if ndim is not None and a.ndim != ndim:
        raise ValueError(
            f"mojolearn: {name} must be {ndim}-D, got {a.ndim}-D shape {a.shape}"
        )
    if a.size == 0:
        raise ValueError(f"mojolearn: {name} is empty, shape {a.shape}")
    a, c2 = _convert(a, dtype, order)
    return a, copied or c2


def as_f32_c(obj, *, ndim=2, name):
    """A C-contiguous float32 Array of `obj`, and whether that cost a copy.

    Returns `(array, copied)`. The caller MUST keep `array` alive across
    the Mojo call; that is the whole reason this returns the array rather
    than just an address. `ndim=None` accepts any rank.
    """
    return _as_typed(obj, "<f4", "C", ndim, name)


def as_f32_colmajor(obj, *, name):
    """A COLUMN-MAJOR float32 2-D Array of `obj`, at most one copy.

    Returns `(array, copied)`; `array.order == 'F'` and its storage is the
    column-major flat (`array._flat()[f * n_rows + r] == array[r, f]`).

    DEVIATION 1887 holds: a float64 or C-order input costs ONE copy (the
    cast happens in the source layout, the transpose in float32), and a
    float32 F-contiguous input is a ZERO-COPY borrow. The float64 ->
    float32 cast is the same one round-to-nearest-even per element either
    way, so the bytes equal `numpy.asfortranarray(x, dtype=float32)`'s.
    """
    return _as_typed(obj, "<f4", "F", 2, name)


def as_i32_c(obj, *, ndim=1, name):
    return _as_typed(obj, "<i4", "C", ndim, name)


def as_i64_c(obj, *, ndim=1, name):
    return _as_typed(obj, "<i8", "C", ndim, name)


def as_f64_c(obj, *, ndim=1, name):
    return _as_typed(obj, "<f8", "C", ndim, name)


# ------------------------------------------------------------ constructors


def empty(shape, dtype):
    """A new owned C-order Array. The memory is zero-filled: there is no
    uninitialized allocation in `array.array`, and a defined value costs
    nothing that matters next to what the kernels do."""
    return Array(shape, dtype)


def zeros(shape, dtype):
    return Array(shape, dtype)


def full(shape, value, dtype):
    dtype = normalize_dtype(dtype)
    a = Array.from_list([value], dtype)  # one conversion of `value`
    from ._array import _check_shape, _prod
    shape = _check_shape(shape)
    size = _prod(shape)
    store = a._store * size if size else array.array(a._store.typecode)
    return Array._owned(store, shape, dtype, "C")


def frombytes(raw, dtype, shape):
    """An owned C-order Array holding a COPY of `raw`."""
    dtype = normalize_dtype(dtype)
    from ._array import _check_shape, _prod, _ITEMSIZE
    shape = _check_shape(shape)
    need = _prod(shape) * _ITEMSIZE[dtype]
    if len(raw) != need:
        raise ValueError(
            f"mojolearn: {len(raw)} bytes do not fill shape {shape} of "
            f"{dtype} ({need} bytes)"
        )
    store = array.array(_CODE[dtype])
    store.frombytes(raw)
    return Array._owned(store, shape, dtype, "C")


# --------------------------------------------------------------- scanning


def all_finite(arr):
    """True when every element of a float32/float64 Array is finite.

    Uses the base binding's `all_finite_f32` / `all_finite_f64` when it
    loads (DEVIATION 2303's helpers); otherwise `math.fsum` over the
    storage memoryview, a C loop, whose result is finite only if every
    input was, with the Python loop run ONLY to confirm a non-finite or
    overflowed sum (DEVIATION 2307).
    """
    if not isinstance(arr, Array) or arr.dtype not in ("<f4", "<f8"):
        raise TypeError(
            "mojolearn: all_finite takes a float32 or float64 Array, got "
            f"{getattr(arr, 'dtype', type(arr).__name__)!r}"
        )
    if arr.size == 0:
        return True
    fn = _native_all_finite(arr.dtype)
    if fn is not None:
        return int(fn(arr._addr, arr.size)) == 1
    try:
        total = math.fsum(arr._mv)
    except (OverflowError, ValueError):
        total = math.nan
    if math.isfinite(total):
        return True
    from ._array import isfinite_all_py
    return isfinite_all_py(arr._mv)


_NATIVE = {}


def _native(key):
    """The base binding's host helper `key`, cached, or None when the
    binding is not built, is the wrong tier, or lacks the symbol. Setting
    `_NATIVE[key] = None` forces the pure-Python path for that helper,
    which is how the tests and the timing script compare the two arms."""
    if key in _NATIVE:
        return _NATIVE[key]
    fn = None
    try:
        from . import _backend
        fn = getattr(_backend.binding("_mojolearn"), key)
    except Exception:  # not built, wrong tier, no device: the host path
        fn = None
    _NATIVE[key] = fn
    return fn


def _native_all_finite(dtype):
    return _native("all_finite_f32" if dtype == "<f4" else "all_finite_f64")

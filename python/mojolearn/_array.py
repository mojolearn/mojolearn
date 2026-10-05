# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`Array`: the NumPy-free container every public estimator returns.

DEVIATION 2301 -- the class `python/mojolearn/NUMPY_FREE_CONTRACT.md`
specifies. It is a typed, shaped, C- or F-contiguous block of memory that
OWNS its bytes (an `array.array`) or, through `from_buffer`, borrows them
from another buffer-protocol object it keeps alive. Its address never moves
for the whole lifetime of the object: the backing `array.array` is never
resized, so `buffer_info()[0]` taken at construction is the address for
good, which is what the Mojo side's borrow-an-address contract needs.

WHAT THIS IS NOT. No arithmetic operators, no broadcasting, no fancy
indexing. `min`, `max`, `sum` and `argmax` are HOST reductions in Python
floats, in element order, and are NOT part of any identity claim; the
kernels never use them. They exist so a caller without NumPy can look at a
result.

ZERO-COPY INTO NUMPY. `__array_interface__` (version 3) hands NumPy the
address, so `numpy.asarray(a)` is a view, never a copy, whenever a caller
has NumPy. `__buffer__` does the same for `memoryview(a)` on Python 3.12+;
on 3.10 and 3.11 a pure-Python class cannot export the buffer protocol at
all, so `memoryview(a)` raises TypeError there and the library's own code
never relies on it: `_buffer.view` recognizes `Array` and reads its backing
store directly on every supported Python (recorded in DEVIATION 2305).

FLOAT16 IS STORED, NOT COMPUTED. `'<f2'` is backed by an `array.array('H')`
and converted with `struct` on the way to Python floats; nothing here does
arithmetic in half precision.
"""

from __future__ import annotations

import array
import struct
import sys
from itertools import chain

# NumPy before 2.4 treats a null interface pointer as a scalar conversion,
# even for an empty shape. Keep a stable, non-null address for empty exports.
_EMPTY_ARRAY_STORAGE = array.array("B", [0] * 8)

# typestr -> array.array typecode of the backing store
_CODE = {
    "<f4": "f", "<f8": "d", "<i4": "i", "<i8": "q",
    "<u4": "I", "<u1": "B", "<f2": "H",
    # lane/identical-lowbit-inference (2026-09-17): bf16 bits travel as
    # uint16 and int8 codes as int8 (gemm/IDENTICAL_LOWBIT_CONTRACT.md).
    "<u2": "H", "<i1": "b",
}
_ITEMSIZE = {"<f4": 4, "<f8": 8, "<i4": 4, "<i8": 8, "<u4": 4, "<u1": 1, "<f2": 2,
             "<u2": 2, "<i1": 1}
_FLOAT = {"<f4", "<f8", "<f2"}
_INT = {"<i4", "<i8", "<u4", "<u1", "<u2", "<i1"}
SUPPORTED_DTYPES = tuple(_CODE)

# The store typecodes must have the sizes the typestrs promise. `'i'` is 4
# and `'q'` is 8 on every platform this library builds for; checked once at
# import rather than assumed.
for _ts, _c in _CODE.items():
    if array.array(_c).itemsize != _ITEMSIZE[_ts]:
        raise ImportError(
            f"mojolearn: array.array typecode {_c!r} is "
            f"{array.array(_c).itemsize} bytes on this platform, expected "
            f"{_ITEMSIZE[_ts]} for {_ts!r}"
        )


# lane/python-hotpath (2026-09-17, DEVIATIONS 3100-3102). dtype codes of the
# core helpers in `bindings/hotpath_helpers.mojo`; float16, bf16 bits and the
# 8/16-bit integers have no code and keep the Python routines. lane
# apple-fast-py2mojo-core (2026-10-03): every non-empty block of a coded
# dtype takes the helper, whatever its size (the `_NATIVE_MIN` cut that kept
# blocks under 256 elements in Python is gone from `_helper`; `_block_store`
# keeps it for its memcpy), and the integer sum is the helper's too
# (`_REDUCE_ISUM`, an exact 128-bit accumulator). The Python routines stay
# as the helpers' definitions: `_native_optional` answers None only for a
# binary that lacks the helper, the state the differential test's reference
# arm (`tests/test_hotpath_native.py`) plants.
_NATIVE_CODE = {"<f4": 0, "<f8": 1, "<i4": 2, "<i8": 3, "<u4": 4, "<u1": 5}
_NATIVE_MIN = 256
_REDUCE_MIN, _REDUCE_MAX, _REDUCE_SUM, _REDUCE_ARGMAX, _REDUCE_INTEGRAL, _REDUCE_ISUM = range(6)
_SCALAR_TYPES = frozenset((float, int, bool))
_ROW_TYPES = frozenset((list, tuple))
_NOT = bytes((1, 0)) + bytes(range(2, 256))


def _helper(key, size):
    """The core helper `key` for a non-empty block, or None for an empty
    block (whose Python routine answers or refuses without a loop) or a
    binary that lacks the helper."""
    if size <= 0:
        return None
    from . import _buffer
    return _buffer._native_optional(key)


_INT_BOUNDS = {"<i4": (-(1 << 31), (1 << 31) - 1), "<i8": (-(1 << 63), (1 << 63) - 1),
               "<u4": (0, (1 << 32) - 1), "<u1": (0, 255)}


def _exact_in(value, dtype):
    """`value` (a Python int, float or bool) as an element of `dtype` that
    compares equal to it EXACTLY under Python's `==`, or None when no
    element of `dtype` can equal it."""
    if isinstance(value, bool):
        value = int(value)
    if dtype in _INT_BOUNDS:
        if isinstance(value, float):
            if value != value or value in (float("inf"), float("-inf")) or value != int(value):
                return None
            value = int(value)
        lo, hi = _INT_BOUNDS[dtype]
        return value if lo <= value <= hi else None
    # float dtypes: a float element x equals value iff float(x) == value
    if isinstance(value, int):
        try:
            f = float(value)
        except OverflowError:
            return None
        if f != value:
            return None  # an int no double holds: no float element equals it
        value = f
    if value != value:
        return None  # NaN equals nothing
    if dtype == "<f4":
        try:
            g = struct.unpack("<f", struct.pack("<f", value))[0]
        except OverflowError:
            return None
        if g != value:
            return None
        return g
    return value


def normalize_dtype(dtype):
    """A typestr in `SUPPORTED_DTYPES` from a typestr, a NumPy dtype or
    scalar type (when the caller has NumPy; nothing is imported here) or a
    Python `float` / `int` / `bool` type. Unknown -> ValueError."""
    if isinstance(dtype, str):
        ts = _ALIAS.get(dtype, dtype)
        if ts in _CODE:
            return ts
        raise ValueError(
            f"mojolearn: unsupported dtype {dtype!r}; supported: "
            f"{', '.join(SUPPORTED_DTYPES)}"
        )
    if dtype is float:
        return "<f8"
    if dtype is int:
        return "<i8"
    if dtype is bool:
        return "<u1"
    # numpy.dtype instances carry `.str`; numpy scalar types carry
    # `__name__` ('float32'). Neither needs numpy imported to be read.
    s = getattr(dtype, "str", None)
    if isinstance(s, str):
        return normalize_dtype(s)
    n = getattr(dtype, "__name__", None)
    if isinstance(n, str) and n in _ALIAS:
        return _ALIAS[n]
    raise ValueError(f"mojolearn: unsupported dtype {dtype!r}")


_ALIAS = {
    "float32": "<f4", "float64": "<f8", "int32": "<i4", "int64": "<i8",
    "uint32": "<u4", "uint8": "<u1", "float16": "<f2",
    "uint16": "<u2", "int8": "<i1", "|i1": "<i1", "=u2": "<u2", "u2": "<u2", "i1": "<i1",
    "|u1": "<u1", "=f4": "<f4", "=f8": "<f8", "=i4": "<i4", "=i8": "<i8",
    "=u4": "<u4", "=f2": "<f2", "f4": "<f4", "f8": "<f8", "i4": "<i4",
    "i8": "<i8", "u4": "<u4", "u1": "<u1", "f2": "<f2",
    "float": "<f8", "int": "<i8", "bool": "<u1", "bool_": "<u1",
}


def _c_strides(shape, itemsize):
    strides = []
    acc = itemsize
    for n in reversed(shape):  # glue: one stride per array axis
        strides.append(acc)
        acc *= n
    return tuple(reversed(strides))


def _f_strides(shape, itemsize):
    strides = []
    acc = itemsize
    for n in shape:  # glue: one stride per array axis
        strides.append(acc)
        acc *= n
    return tuple(strides)


def _prod(shape):
    n = 1
    for s in shape:  # glue: product of shape dimensions
        n *= s
    return n


def _check_shape(shape):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(int(s) for s in shape)  # glue: validates the shape argument
    for s in shape:  # glue: validates the shape argument
        if s < 0:
            raise ValueError(f"mojolearn: negative dimension in shape {shape}")
    return shape


def _new_store(code, size):
    """A zero-filled `array.array` of `size` elements, allocated once."""
    if size == 0:
        return array.array(code)
    return array.array(code, [0]) * size


def _to_py_values(mv, dtype):
    """Python scalars for a flat store memoryview."""
    if dtype == "<f2":
        n = len(mv)
        return list(struct.unpack("<%de" % n, mv.tobytes())) if n else []
    return mv.tolist()


def _strided_copy(src_addr, dst_addr, itemsize, shape, src_strides, dst_strides,
                  src_base=0, dst_base=0):
    """Element (i_0..i_k) of `shape`, in C order, moves from
    src[src_base + sum i_j * src_strides[j]] to dst[dst_base + sum i_j *
    dst_strides[j]] (strides and bases in elements, bits unchanged), in Mojo
    (`strided_copy_bytes`, `bindings/array_helpers.mojo`; lane pyglue-sweep:
    the per-row and per-column Python loops it replaces are gone). The GPU
    base binding runs it on the device (lane cpu2-l1-input,
    `core/input_device.mojo`); the host loop is the host column."""
    from . import _buffer
    dims = array.array("q", (*shape, *src_strides, *dst_strides, src_base, dst_base))
    _buffer._native("strided_copy_bytes")(src_addr, dst_addr, dims.buffer_info()[0],
                                          len(shape), itemsize)


def _reorder(a, src_order, out):
    """Copy the elements of `a` (laid out in `src_order`) into the store
    `out` laid out in the other order: one Mojo strided copy."""
    shape = a.shape
    if len(shape) <= 1 or a.size == 0:
        memoryview(out)[:] = a._mv
        return
    cs = _c_strides(shape, 1)
    fs = _f_strides(shape, 1)
    src, dst = (cs, fs) if src_order == "C" else (fs, cs)
    _strided_copy(a._addr, out.buffer_info()[0], a.itemsize, shape, src, dst)


def _restore_array(raw, shape, dtype, order, readonly):
    """Pickle restores values and layout, never a borrowed pointer or GPU handle."""
    if readonly:
        flat = Array.from_buffer(memoryview(raw).cast(_CODE[dtype]))
        return Array._view_of(flat, shape, order)
    store = array.array(_CODE[dtype])
    store.frombytes(raw)
    return Array._owned(store, shape, dtype, order)


class _ArrayInterface:
    """Carries only an Array's `__array_interface__`, so `Array.__array__`
    can hand NumPy a zero-copy view without recursing into itself. NumPy
    keeps this object (and through it the Array) alive as the view's base."""

    __slots__ = ("_owner", "__array_interface__")

    def __init__(self, owner):
        self._owner = owner
        self.__array_interface__ = owner.__array_interface__


class Array:
    """A typed, shaped, contiguous block of memory. See the module docstring.

    `Array(shape, dtype)` allocates zeros; `empty`, `zeros`, `full`,
    `frombytes`, `from_list` and `from_buffer` are the usual constructors.
    """

    __slots__ = (
        "_store", "_base", "_pin", "_mv", "_addr", "_readonly",
        "shape", "dtype", "ndim", "size", "nbytes", "itemsize", "strides",
        "order",
    )

    def __init__(self, shape, dtype="<f4"):
        dtype = normalize_dtype(dtype)
        shape = _check_shape(shape)
        store = _new_store(_CODE[dtype], _prod(shape))
        self._init_owned(store, shape, dtype, "C")

    # ------------------------------------------------------------ construction
    def _init_owned(self, store, shape, dtype, order):
        self._store = store
        self._base = None
        self._pin = None
        self._mv = memoryview(store)
        if self._mv.format != _CODE[dtype]:
            self._mv = self._mv.cast("B").cast(_CODE[dtype])
        self._addr = store.buffer_info()[0]
        self._readonly = False
        self._set_meta(shape, dtype, order)
        return self

    def _set_meta(self, shape, dtype, order):
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)
        self.size = _prod(shape)
        self.itemsize = _ITEMSIZE[dtype]
        self.nbytes = self.size * self.itemsize
        self.order = order
        if order == "C":
            self.strides = _c_strides(shape, self.itemsize)
        else:
            self.strides = _f_strides(shape, self.itemsize)
        if len(self._mv) != self.size:
            raise ValueError(
                f"mojolearn: buffer holds {len(self._mv)} elements, shape "
                f"{shape} needs {self.size}"
            )

    @classmethod
    def _owned(cls, store, shape, dtype, order="C"):
        self = cls.__new__(cls)
        return self._init_owned(store, shape, dtype, order)

    @classmethod
    def _view_of(cls, other, shape, order="C"):
        """Another shape over the SAME memory (no copy, no ownership
        transfer): `other` stays alive through `_base`."""
        self = cls.__new__(cls)
        self._store = other._store
        self._base = other
        self._pin = other._pin
        self._mv = other._mv
        self._addr = other._addr
        self._readonly = other._readonly
        self._set_meta(shape, other.dtype, order)
        return self

    @classmethod
    def from_buffer(cls, obj):
        """A zero-copy view over any contiguous buffer-protocol object.

        `obj` is kept alive by the result and NOT owned; writes through the
        result reach `obj`. A non-contiguous or unsupported-format buffer
        is refused (ValueError / TypeError) rather than copied: copying is
        `_buffer.as_*_c`'s job and it reports the copy.
        """
        from . import _buffer  # lazy: _buffer imports this module

        if isinstance(obj, Array):
            return cls._view_of(obj, obj.shape, obj.order)
        with _buffer.view(obj, name="object") as b:
            dtype = _buffer.typestr_of(b)
            if dtype not in _CODE:
                raise TypeError(
                    f"mojolearn: buffer format {b.format!r} (itemsize "
                    f"{b.itemsize}) is not a supported Array dtype"
                )
            if b.c_contiguous:
                order = "C"
            elif b.f_contiguous:
                order = "F"
            else:
                raise ValueError(
                    "mojolearn: from_buffer needs a contiguous buffer; "
                    f"got shape {b.shape} strides {b.strides}"
                )
            self = cls.__new__(cls)
            self._store = obj
            self._base = None
            # a memoryview of the exporter pins its buffer for our lifetime
            self._pin = memoryview(obj)
            self._addr = b.addr
            self._readonly = b.readonly
            self._mv = _buffer.memory_at(
                b.addr, b.nbytes, writable=not b.readonly
            ).cast(_CODE[dtype])
            self._set_meta(b.shape, dtype, order)
            return self

    @classmethod
    def from_list(cls, nested, dtype="<f8"):
        """An owned C-order Array from a (nested) list of Python scalars.

        Shape is inferred and must be rectangular. Float -> `'<f4'` is
        round-to-nearest-even (the `array.array('f')` item setter is a C
        `(float)` cast); float -> int dtypes are refused rather than
        truncated, ints wider than the dtype overflow rather than wrap.
        """
        dtype = normalize_dtype(dtype)
        shape, flat = _flatten(nested)
        return cls._from_flat(flat, shape, dtype)

    @classmethod
    def _from_flat(cls, flat, shape, dtype):
        code = _CODE[dtype]
        if dtype == "<f2":
            store = array.array("H")
            if len(flat):
                store.frombytes(struct.pack("<%de" % len(flat), *flat))
        else:
            try:
                store = array.array(code, flat)
            except (TypeError, OverflowError) as e:
                raise TypeError(
                    f"mojolearn: cannot build a {dtype} Array from these "
                    f"values: {e}"
                ) from None
        return cls._owned(store, shape, dtype, "C")

    # ------------------------------------------------------------- interface
    def _both_orders(self):
        """True when C and F layouts coincide: at most one dimension is
        larger than 1 (NumPy's rule), so either label is free."""
        return sum(1 for s in self.shape if s != 1) <= 1  # glue: counts non-unit shape dimensions

    def _has_order(self, order):
        return self.order == order or self._both_orders()

    @property
    def flags(self):
        both = self._both_orders()
        return {
            "C_CONTIGUOUS": self.order == "C" or both,
            "F_CONTIGUOUS": self.order == "F" or both,
            "WRITEABLE": not self._readonly,
        }

    @property
    def __array_interface__(self):
        # Version 3. `strides` None means C-contiguous; an F-order block
        # reports its real strides so NumPy views it without copying.
        return {
            "shape": self.shape,
            "typestr": self.dtype,
            "data": (self._addr if self.size or self._addr else
                     _EMPTY_ARRAY_STORAGE.buffer_info()[0], bool(self._readonly)),
            "strides": None if self.order == "C" else self.strides,
            "version": 3,
        }

    def __array__(self, dtype=None, copy=None):
        """`numpy.asarray(a)` through the `__array__` protocol.

        NumPy itself reads `__array_interface__` above and never gets here;
        this exists for the libraries that test `hasattr(x, "__array__")`
        before converting. scikit-learn's `type_of_target` (reached by
        `accuracy_score`, `f1_score`, `confusion_matrix`, ...) refused a
        classifier's `predict` output as "Expected array-like" without it
        (Sep 22 pip smoke). Only NumPy calls this method, so NumPy is
        already loaded: it is read from `sys.modules`, never imported, and
        the module keeps its no-NumPy import contract.
        """
        np = sys.modules.get("numpy")
        if np is None:
            raise TypeError("mojolearn: Array.__array__ needs NumPy loaded")
        out = np.asarray(_ArrayInterface(self))
        if dtype is not None and out.dtype != np.dtype(dtype):
            if copy is False:
                raise ValueError("mojolearn: a dtype change needs a copy")
            return out.astype(dtype)
        return out.copy() if copy else out

    def __buffer__(self, flags):
        """Python 3.12+ buffer protocol: a memoryview of the backing store
        with this Array's shape.

        An F-order block of rank 2+ and a `'<f2'` block RAISE BufferError:
        a memoryview built from Python cannot carry column-major strides or
        the `'e'` format, and handing out the flat store instead would be
        wrong in a way that matters. NumPy consults the buffer protocol
        BEFORE `__array_interface__` (measured on numpy 2.4 / CPython
        3.14), so a flat view here would make `numpy.asarray` return a 1-D
        array for a matrix; on the BufferError it falls through to
        `__array_interface__`, which describes both cases exactly and is
        still zero-copy. `tobytes()` and `_flat()` reach the raw store.
        """
        if self.dtype == "<f2":
            raise BufferError(
                "mojolearn: a float16 Array has no memoryview format; use "
                "numpy.asarray (zero-copy) or tolist()"
            )
        if self.ndim == 0:
            return self._mv
        if self.order != "C" and self.ndim > 1:
            raise BufferError(
                "mojolearn: an F-order Array cannot be exported as a "
                "memoryview; use numpy.asarray (zero-copy), tobytes() or "
                "reshape() for a C-order copy"
            )
        if self.size == 0:
            # CPython refuses `cast(fmt, shape)` with a zero in the shape.
            # A 1-D empty Array exports as a typed empty view (so `bytes()`
            # and `memoryview()` work); an empty matrix raises BufferError
            # so `numpy.asarray` falls through to `__array_interface__`,
            # which carries the (0, k) shape exactly (found 2026-09-10 by
            # the DEVIATION 2489 gate on an all-zero affinity matrix).
            if self.ndim == 1:
                return self._mv.cast("B").cast(_CODE[self.dtype])
            raise BufferError(
                "mojolearn: an empty matrix has no memoryview shape; use "
                "numpy.asarray (zero-copy), tobytes() or reshape(-1)"
            )
        return self._mv.cast("B").cast(_CODE[self.dtype], self.shape)

    def _flat(self):
        """A 1-D Array over the storage IN STORAGE ORDER, no copy. For an
        F-order matrix that is the column-major flat, `flat[f * rows + r]
        == a[r, f]`; the shape used by the gbdt quantizer."""
        return Array._view_of(self, (self.size,), "C")

    # ------------------------------------------------------------- basic ops
    def __reduce__(self):
        return (_restore_array, (self.tobytes(), self.shape, self.dtype,
                                 self.order, self._readonly))

    def tobytes(self):
        """The bytes of the storage, in storage order (C for a C-order
        Array, column-major for an F-order one)."""
        return self._mv.tobytes()

    def _values(self):
        return _to_py_values(self._mv, self.dtype)

    def _as_c(self):
        """Self if C-order, else a C-order copy."""
        if self.order == "C":
            return self
        if self._both_orders():
            return Array._view_of(self, self.shape, "C")
        out = _new_store(_CODE[self.dtype], self.size)
        _reorder(self, "F", out)
        return Array._owned(out, self.shape, self.dtype, "C")

    def _as_order(self, order):
        """Self when already laid out as `order`, a relabeled view when
        both layouts coincide, else a copy in `order`."""
        if order == self.order:
            return self
        if self._both_orders():
            return Array._view_of(self, self.shape, order)
        if order == "C":
            return self._as_c()
        out = _new_store(_CODE[self.dtype], self.size)
        _reorder(self, "C", out)
        return Array._owned(out, self.shape, self.dtype, "F")

    def tolist(self):
        """Nested Python lists by shape (a scalar for a 0-d Array)."""
        a = self._as_c()
        if self.ndim == 0:
            return a._values()[0]
        return _nest(a)

    def copy(self):
        """An owned copy, same dtype, shape and order."""
        store = array.array(_CODE[self.dtype])
        store.frombytes(self._mv.cast("B"))  # one memcpy
        return Array._owned(store, self.shape, self.dtype, self.order)

    def astype(self, dtype):
        """A converted copy (always a copy, even for the same dtype).

        Float -> `'<f4'` is one round-to-nearest-even per element (the C
        cast); float -> `'<f8'` and every int -> float within 2**53 is
        exact. Float -> int truncates toward zero like NumPy and is a
        per-element Python loop: it is for labels, not for matrices. Order
        is preserved.
        """
        dtype = normalize_dtype(dtype)
        if dtype == self.dtype:
            return self.copy()
        src = self.dtype
        code = _CODE[dtype]
        fast = self._native_astype(dtype)
        if fast is not None:
            return fast
        if src == "<f2" or dtype == "<f2":
            values = self._values()
            if dtype in _INT:
                values = [int(v) for v in values]
            out = Array._from_flat(values, self.shape, dtype)
        elif dtype in _FLOAT:
            # C-level loop: array.array's item setter converts each element
            # exactly as `(float)` / `(double)` would.
            out = Array._owned(
                array.array(code, self._mv), self.shape, dtype, "C"
            )
        elif src in _INT:
            try:
                store = array.array(code, self._mv)
            except OverflowError as e:
                raise OverflowError(
                    f"mojolearn: value does not fit {dtype}: {e}"
                ) from None
            out = Array._owned(store, self.shape, dtype, "C")
        else:
            values = [int(v) for v in self._mv]
            out = Array._from_flat(values, self.shape, dtype)
        out.order = self.order
        out._set_meta(self.shape, dtype, self.order)
        return out

    def _native_astype(self, dtype):
        """DEVIATION 3100: `astype` through the core helper `cast_elements`,
        or None. The routine below it is `array.array(code, <memoryview>)`,
        a C loop that builds one Python object per element (22 to 31 ns);
        the helper performs the same single conversion per element with no
        object. When the helper meets an element the routine below REFUSES
        (an integer outside the target, a NaN or infinity headed for an
        integer dtype) it reports so and the routine below runs and raises
        its own words."""
        sc = _NATIVE_CODE.get(self.dtype)
        dc = _NATIVE_CODE.get(dtype)
        if sc is None or dc is None:
            return None
        fn = _helper("cast_elements", self.size)
        if fn is None:
            return None
        from . import _buffer
        store = _buffer._output_store(_CODE[dtype], self.size)
        if int(fn(self._addr, sc, store.buffer_info()[0], dc, self.size)):
            return None
        return Array._owned(store, self.shape, dtype, self.order)

    def reshape(self, shape):
        """A C-order view over the same buffer (one `-1` allowed). An
        F-order Array is copied to C order first, as NumPy would."""
        if isinstance(shape, int):
            shape = (shape,)
        shape = [int(s) for s in shape]  # glue: validates the shape argument
        if shape.count(-1) > 1:
            raise ValueError("mojolearn: reshape allows one -1")
        if -1 in shape:
            known = _prod(s for s in shape if s != -1)  # glue: product of the known dimensions
            if known == 0 or self.size % known:
                raise ValueError(
                    f"mojolearn: cannot reshape size {self.size} into {tuple(shape)}"
                )
            shape[shape.index(-1)] = self.size // known
        shape = tuple(shape)
        if _prod(shape) != self.size:
            raise ValueError(
                f"mojolearn: cannot reshape size {self.size} into {shape}"
            )
        return Array._view_of(self._as_c(), shape, "C")

    def ravel(self):
        return self.reshape((-1,))

    def __len__(self):
        if self.ndim == 0:
            raise TypeError("len() of a 0-d Array")
        return self.shape[0]

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d Array")
        if self.ndim == 1:
            return iter(self._values())
        return _RowIter(self._as_c())

    def _row_view(self, i, shape, step):
        """Row `i` of a C-order Array as a zero-copy view (`shape` = the
        trailing dimensions, `step` = their element count): the same
        memory, kept alive through `_base`, as NumPy's iteration gives."""
        v = Array.__new__(Array)
        v._store = self._store
        v._base = self
        v._pin = self._pin
        off = i * step
        v._mv = self._mv[off:off + step]
        v._addr = self._addr + off * self.itemsize
        v._readonly = self._readonly
        v._set_meta(shape, self.dtype, "C")
        return v

    def __repr__(self):
        return f"Array(shape={self.shape}, dtype={self.dtype!r})"

    def __bool__(self):
        if self.size != 1:
            raise ValueError(
                "mojolearn: the truth value of an Array with more than one "
                "element is ambiguous"
            )
        return bool(self._values()[0])

    # ------------------------------------------------------------- indexing
    def __getitem__(self, key):
        """Int / slice / tuple of them. Full integer indexing returns a
        Python scalar; anything else a COPY as a C-order Array."""
        a = self._as_c()
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) > a.ndim:
            raise IndexError(
                f"mojolearn: too many indices ({len(key)}) for shape {a.shape}"
            )
        dims = []  # (start, step, count) or int
        for axis, k in enumerate(key):  # glue: one index entry per array axis
            n = a.shape[axis]
            if isinstance(k, slice):
                start, stop, step = k.indices(n)
                count = len(range(start, stop, step))
                dims.append((start, step, count))
            elif isinstance(k, int) or hasattr(k, "__index__"):
                k = k.__index__()
                if k < -n or k >= n:
                    raise IndexError(
                        f"mojolearn: index {k} out of range for axis {axis} "
                        f"of size {n}"
                    )
                dims.append(k % n if n else k)
            else:
                raise TypeError(
                    "mojolearn: Array indices are ints, slices or tuples of "
                    f"them, not {type(k).__name__}"
                )
        for axis in range(len(key), a.ndim):  # glue: one entry per remaining array axis
            dims.append((0, 1, a.shape[axis]))
        cs = _c_strides(a.shape, 1)
        # The selection as one strided copy (shape, strides, base offset in
        # elements), done in Mojo; a contiguous block is one memcpy.
        base_off = 0
        sel_shape = []
        sel_strides = []
        for axis, d in enumerate(dims):  # glue: one entry per array axis
            if isinstance(d, tuple):
                base_off += d[0] * cs[axis]
                sel_shape.append(d[2])
                sel_strides.append(d[1] * cs[axis])
            else:
                base_off += d * cs[axis]
        out_shape = tuple(sel_shape)
        if not out_shape:
            return _to_py_values(a._mv[base_off:base_off + 1], a.dtype)[0]
        block = _contiguous_block(dims, a.shape, cs)
        store = _block_store(a, block[0], block[1]) if block is not None else None
        if store is None:
            total = _prod(out_shape)
            store = _new_store(_CODE[a.dtype], total)
            if total:
                _strided_copy(a._addr, store.buffer_info()[0], a.itemsize, out_shape,
                              sel_strides, _c_strides(out_shape, 1), base_off, 0)
        return Array._owned(store, out_shape, a.dtype, "C")

    def __eq__(self, other):
        """Elementwise equality against an Array of the same shape or a
        scalar, as a `'<u1'` Array of 0/1."""
        if isinstance(other, Array) and other.shape == self.shape:
            fast = self._native_eq(other)
            if fast is not None:
                return fast
        elif isinstance(other, (int, float, bool)):
            fast = self._native_eq_scalar(other)
            if fast is not None:
                return fast
        mine = self._as_c()._values()
        if isinstance(other, Array):
            if other.shape != self.shape:
                raise ValueError(
                    f"mojolearn: shapes {self.shape} and {other.shape} differ"
                )
            theirs = other._as_c()._values()
            bits = [1 if x == y else 0 for x, y in zip(mine, theirs)]
        elif isinstance(other, (int, float, bool)):
            bits = [1 if x == other else 0 for x in mine]
        else:
            return NotImplemented
        return Array._owned(array.array("B", bits), self.shape, "<u1", "C")

    def _native_eq(self, other):
        """DEVIATION 3102: elementwise equality of two same-shape Arrays of
        ONE dtype through the core helper `equal_elements`, or None. IEEE
        equality is Python's float equality; two dtypes (an int against a
        float compares exactly in Python) keep the Python routine."""
        code = _NATIVE_CODE.get(self.dtype)
        if code is None or other.dtype != self.dtype:
            return None
        fn = _helper("equal_elements", self.size)
        if fn is None:
            return None
        a = self._as_c()
        b = other._as_c()
        store = array.array("B", bytes(self.size))
        fn(a._addr, b._addr, code, self.size, store.buffer_info()[0])
        return Array._owned(store, self.shape, "<u1", "C")

    def _native_eq_scalar(self, other):
        """lane apple-fast-py2mojo-core: equality against a Python scalar
        through `equal_elements`, the scalar repeated in C as this dtype, or
        None. Python compares exactly (an int against a float included), so
        the scalar is first taken to this dtype EXACTLY; a scalar no element
        can equal (NaN, a non-integer against an integer dtype, a value out
        of the dtype's range or not representable in float32) answers all
        zeros without a pass."""
        code = _NATIVE_CODE.get(self.dtype)
        if code is None:
            return None
        fn = _helper("equal_elements", self.size)
        if fn is None:
            return None
        v = _exact_in(other, self.dtype)
        store = array.array("B", bytes(self.size))
        if v is not None:
            a = self._as_c()
            fill = array.array(_CODE[self.dtype], [v]) * self.size
            fn(a._addr, fill.buffer_info()[0], code, self.size, store.buffer_info()[0])
        return Array._owned(store, self.shape, "<u1", "C")

    def __ne__(self, other):
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return eq
        # `bytes.translate` flips 0 and 1 in C (was a comprehension per element)
        return Array._owned(
            array.array("B", eq._mv.tobytes().translate(_NOT)), self.shape, "<u1", "C"
        )

    __hash__ = None

    # ------------------------------------------------------------ reductions
    # HOST reductions in Python scalars, sequential in storage order. They
    # are NOT part of any identity claim and no kernel path uses them.
    def _reduce_values(self, what):
        if self.size == 0:
            raise ValueError(f"mojolearn: {what} of an empty Array")
        return self._values()

    def _native_reduce(self, what):
        """DEVIATION 3101: `what` through the core helper `reduce_stat`, or
        None. Sequential in storage order with Python's own comparison, so a
        NaN or a signed zero answers as `min(list)` / `max(list)` does."""
        code = _NATIVE_CODE.get(self.dtype)
        if code is None:
            return None
        fn = _helper("reduce_stat", self.size)
        if fn is None:
            return None
        return fn(self._addr, code, self.size, what)

    def min(self):
        fast = self._native_reduce(_REDUCE_MIN)
        if fast is not None:
            return fast
        return min(self._reduce_values("min"))

    def max(self):
        fast = self._native_reduce(_REDUCE_MAX)
        if fast is not None:
            return fast
        return max(self._reduce_values("max"))

    def sum(self):
        """Sequential accumulation: exact `int` for int dtypes, a Python
        float summed left to right in storage order for float dtypes."""
        if self.dtype in ("<f4", "<f8"):
            fast = self._native_reduce(_REDUCE_SUM)
            if fast is not None:
                return fast
        elif self.dtype in _NATIVE_CODE:
            # lane apple-fast-py2mojo-core: the exact integer sum in the
            # helper, (hi, lo >> 32, lo & 0xffffffff) of a 128-bit total
            parts = self._native_reduce(_REDUCE_ISUM)
            if parts is not None:
                hi, mid, low = (int(v) for v in parts)  # glue: three words of a 128-bit sum
                return (hi << 64) + (mid << 32) + low
        values = self._values()
        if self.dtype in _INT:
            return sum(values)
        acc = 0.0
        for v in values:
            acc += v
        return acc

    def argmax(self):
        """Flat index of the first maximum (first-max-wins), in storage
        order, over the C-order view."""
        fast = self._as_c()._native_reduce(_REDUCE_ARGMAX)
        if fast is not None:
            return fast
        values = self._as_c()._reduce_values("argmax")
        best = 0
        best_v = values[0]
        for i in range(1, len(values)):
            v = values[i]
            if v > best_v:
                best = i
                best_v = v
        return best


class _RowIter:
    """Iteration over the rows of a C-order Array of rank >= 2: each step
    hands out a zero-copy view of the next row (no element is read)."""

    __slots__ = ("_a", "_i", "_n", "_shape", "_step")

    def __init__(self, a):
        self._a = a
        self._i = 0
        self._n = a.shape[0]
        self._shape = a.shape[1:]
        self._step = _prod(self._shape)

    def __iter__(self):
        return self

    def __next__(self):
        i = self._i
        if i >= self._n:
            raise StopIteration
        self._i = i + 1
        return self._a._row_view(i, self._shape, self._step)

    def __length_hint__(self):
        return self._n - self._i


def _contiguous_block(dims, shape, cs):
    """`(offset, length)` in elements when the selection `dims` of a C-order
    block is ONE contiguous run whose C-order walk is the run itself:
    leading integer indices, then at most one step-1 slice, then only whole
    axes. None otherwise (a stride, a slice after a partial slice)."""
    offset = 0
    axis = 0
    n = len(dims)
    while axis < n and not isinstance(dims[axis], tuple):
        offset += dims[axis] * cs[axis]
        axis += 1
    if axis == n:
        return None
    start, step, count = dims[axis]
    if step != 1:
        return None
    for later in range(axis + 1, n):  # glue: one check per trailing array axis
        d = dims[later]
        if not isinstance(d, tuple) or d != (0, 1, shape[later]):
            return None
    return offset + start * cs[axis], count * cs[axis]


def _block_store(a, offset, length):
    """DEVIATION 3105: an owned store holding `length` elements of `a` from
    `offset`, equal byte for byte to the per-run `array.array(code,
    <memoryview slice>)` copies it replaces, or None when that needs the
    helper and the helper is absent. One memcpy instead of one Python object
    per element (0.2 ns against 22); see `_buffer._same_dtype_store` for the
    one float32 subtlety."""
    if length < _NATIVE_MIN:
        return None  # a short run: the per-element copy is as fast as the detour
    from . import _buffer
    raw = a._mv[offset:offset + length].cast("B")
    return _buffer._same_dtype_store(raw, a.dtype)


def _flatten_fast(nested):
    """DEVIATION 3106: `_flatten` for the two shapes every estimator input
    takes, a flat list of Python scalars and a list of equal-length rows of
    them, with the leaf test and the row walk done by C builtins instead of
    one recursive call per leaf (1.5 s per 10,000,000 leaves). None for
    anything else (a ragged block, a NumPy scalar, a nested Array, rank 3),
    which `_flatten` then walks and, where it must, refuses in its words."""
    if type(nested) not in _ROW_TYPES or not nested:
        return None
    kinds = set(map(type, nested))
    if kinds <= _SCALAR_TYPES:
        return (len(nested),), list(nested)
    if not kinds <= _ROW_TYPES:
        return None
    widths = set(map(len, nested))
    if len(widths) != 1:
        return None
    flat = list(chain.from_iterable(nested))
    if not set(map(type, flat)) <= _SCALAR_TYPES:
        return None
    return (len(nested), widths.pop()), flat


def _flatten(nested):
    """`(shape, flat_list)` for a rectangular nested sequence; a scalar
    gives shape ()."""
    if isinstance(nested, Array):
        return nested.shape, nested._as_c()._values()
    fast = _flatten_fast(nested)
    if fast is not None:
        return fast
    if isinstance(nested, (str, bytes)) or not hasattr(nested, "__len__"):
        return (), [nested]
    seq = list(nested)
    if not seq:
        return (0,), []
    first_shape, flat = _flatten(seq[0])
    for item in seq[1:]:
        shape, vals = _flatten(item)
        if shape != first_shape:
            raise ValueError(
                "mojolearn: nested sequence is not rectangular "
                f"({shape} vs {first_shape})"
            )
        flat.extend(vals)
    return (len(seq),) + first_shape, flat


def _nest(a):
    """Nested Python lists of a C-order Array, built by `memoryview.tolist`
    (CPython walks the rows in C; a float16 block goes through its exact
    float64 values first)."""
    if a.size == 0:
        return _empty_nest(a.shape)
    if a.dtype == "<f2":
        flat = array.array("d", a._values())
        return memoryview(flat).cast("B").cast("d", a.shape).tolist()
    return a._mv.cast("B").cast(_CODE[a.dtype], a.shape).tolist()


def _empty_nest(shape):
    """The nested lists of an empty block: no elements, only the list
    structure up to the first zero dimension."""
    if not shape or shape[0] == 0:
        return []
    return [_empty_nest(shape[1:]) for _ in range(shape[0])]  # glue: list structure of an empty block, no elements

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python definitions of `mojolearn.Array`'s public operations: the test
oracles of `bindings/array_native_door.mojo` (test_hotpath_native.py,
test_native_array_operations.py, tools/test_native_array_connection.py).

Moved out of the product package (lane p5-cpu-route-relabel-and-dead-code,
2026-11-09): every public Array operation is one `array_native_call`, so these
definitions and the core-helper wrappers only they used (`_native_astype`,
`_native_eq`, `_native_eq_scalar`, `_native_reduce`, `_no_reduce`,
`_reduce_values`, `_astype_reference`, `_RowIter` and the `_reference()` seam)
live with the tests. `_REFERENCE` is True only inside test_hotpath_native.py's
reference arm: the Python definitions answer there and nowhere else.
"""
import array
import copyreg
import ctypes
import itertools
from itertools import chain

from mojolearn import Array
# The address the product's `__array_interface__` reports for an empty Array; the reference must report the same one.
from mojolearn._array import _EMPTY_ARRAY_STORAGE  # oracle subject: the product's empty-export storage, whose address no public route gives
import struct
import sys

from mojolearn.tests import _typed_input
from mojolearn.tests._oracle_support import CODE as _CODE


# Local copies of `mojolearn._array`'s tables and small helpers (the product's values and wording), so an internal
# rename does not break this oracle.
_ITEMSIZE = {"<f4": 4, "<f8": 8, "<i4": 4, "<i8": 8, "<u4": 4, "<u1": 1, "<f2": 2,
             "<u2": 2, "<i1": 1}
SUPPORTED_DTYPES = tuple(_CODE)
_NATIVE_CODE = {"<f4": 0, "<f8": 1, "<i4": 2, "<i8": 3, "<u4": 4, "<u1": 5,
                "<u2": 6, "<i1": 7, "<f2": 8}
#: equal_elements' exact int64-against-float64 code
_EQ_I64_F64 = 9
_NATIVE_MIN = 256
_REDUCE_MIN, _REDUCE_MAX, _REDUCE_SUM, _REDUCE_ARGMAX, _REDUCE_INTEGRAL, _REDUCE_ISUM = range(6)
_SCALAR_TYPES = frozenset((float, int, bool))
_ROW_TYPES = frozenset((list, tuple))
_NOT = bytes((1, 0)) + bytes(range(2, 256))
_INT_BOUNDS = {"<i4": (-(1 << 31), (1 << 31) - 1), "<i8": (-(1 << 63), (1 << 63) - 1),
               "<u4": (0, (1 << 32) - 1), "<u1": (0, 255), "<u2": (0, 65535), "<i1": (-128, 127)}
_ALIAS = {
    "float32": "<f4", "float64": "<f8", "int32": "<i4", "int64": "<i8",
    "uint32": "<u4", "uint8": "<u1", "float16": "<f2",
    "uint16": "<u2", "int8": "<i1", "|i1": "<i1", "=u2": "<u2", "u2": "<u2", "i1": "<i1",
    "|u1": "<u1", "=f4": "<f4", "=f8": "<f8", "=i4": "<i4", "=i8": "<i8",
    "=u4": "<u4", "=f2": "<f2", "f4": "<f4", "f8": "<f8", "i4": "<i4",
    "i8": "<i8", "u4": "<u4", "u1": "<u1", "f2": "<f2",
    "float": "<f8", "int": "<i8", "bool": "<u1", "bool_": "<u1",
}


def normalize_dtype(dtype):
    """A typestr in `SUPPORTED_DTYPES` from a typestr, a NumPy dtype or
    scalar type or a Python `float` / `int` / `bool` type. Unknown -> ValueError."""
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
    s = getattr(dtype, "str", None)
    if isinstance(s, str):
        return normalize_dtype(s)
    n = getattr(dtype, "__name__", None)
    if isinstance(n, str) and n in _ALIAS:
        return _ALIAS[n]
    raise ValueError(f"mojolearn: unsupported dtype {dtype!r}")


def _to_py_values(mv, dtype):
    """Python scalars for a flat store memoryview."""
    if dtype == "<f2":
        n = len(mv)
        return list(struct.unpack("<%de" % n, mv.tobytes())) if n else []
    return mv.tolist()


class _ArrayInterface:
    """Carries only an Array's `__array_interface__`, so `__array__` can hand
    NumPy a zero-copy view without recursing into itself. NumPy keeps this
    object (and through it the Array) alive as the view's base."""

    __slots__ = ("_owner", "__array_interface__")

    def __init__(self, owner):
        self._owner = owner
        self.__array_interface__ = owner.__array_interface__


def _flatten_fast(nested):
    """`_flatten` for a flat list of Python scalars and a list of equal-length
    rows of them; None for anything else."""
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
    # the general walk (a ragged or mixed nested list, Arrays inside it) is the product's native nested-list read
    from mojolearn import _user_objects  # oracle subject: the native nested-list reader (user_objects "nested"), no public route returns its raw flat values
    shape, flat = _user_objects.call("nested", nested, Array)
    return tuple(shape), flat


def _empty_nest(shape):
    """The nested lists of an empty block: the list structure up to the first zero dimension."""
    if not shape or shape[0] == 0:
        return []
    return [_empty_nest(shape[1:]) for _ in range(shape[0])]


def _nest(a):
    """Nested Python lists of a C-order Array, built by `memoryview.tolist`
    (a float16 block goes through its exact float64 values first)."""
    if a.size == 0:
        return _empty_nest(a.shape)
    if a.dtype == "<f2":
        flat = array.array("d", a._values())
        return memoryview(flat).cast("B").cast("d", a.shape).tolist()
    return a._mv.cast("B").cast(_CODE[a.dtype], a.shape).tolist()


# The Python definitions of the shape, stride and layout helpers whose product
# copies moved to bindings/array_native_door.mojo (lane pyout-runtime-boundary,
# 2026-11-11): oracles of `_c_strides`, `_f_strides`, `_prod`,
# `_checked_shape`, `_new_store`, `_strided_copy`, `_set_meta`, `_init_owned`,
# `_both_orders` and the `row` operation there.
def _c_strides(shape, itemsize):
    strides = []
    acc = itemsize
    for n in reversed(shape):
        strides.append(acc)
        acc *= n
    return tuple(reversed(strides))


def _f_strides(shape, itemsize):
    strides = []
    acc = itemsize
    for n in shape:
        strides.append(acc)
        acc *= n
    return tuple(strides)


def _prod(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def _check_shape(shape):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(int(s) for s in shape)
    for s in shape:
        if s < 0:
            raise ValueError(f"mojolearn: negative dimension in shape {shape}")
    return shape


def _new_store(code, size):
    if size == 0:
        return array.array(code)
    return array.array(code, [0]) * size


_WORD = {1: ctypes.c_uint8, 2: ctypes.c_uint16, 4: ctypes.c_uint32, 8: ctypes.c_uint64}


def _span(base, shape, strides):
    """The lowest and highest element offset a strided walk touches."""
    lo = hi = base
    for n, s in zip(shape, strides):
        if s >= 0:
            hi += (n - 1) * s
        else:
            lo += (n - 1) * s
    return lo, hi


def _strided_copy(src_addr, dst_addr, itemsize, shape, src_strides, dst_strides,
                  src_base=0, dst_base=0):
    """The core helper `strided_copy_bytes` (core/array_helpers_impl.mojo `_strided_copy`) in the standard library:
    walking `shape` in C order, element (i_0..i_k) moves from src[src_base + sum i_j * s_j] to
    dst[dst_base + sum i_j * d_j] (offsets and strides in elements), as an unsigned word of `itemsize` bytes."""
    if _prod(shape) == 0:
        return
    word = _WORD[itemsize]
    slo, shi = _span(src_base, shape, src_strides)
    dlo, dhi = _span(dst_base, shape, dst_strides)
    src = (word * (shi - slo + 1)).from_address(src_addr + slo * itemsize)
    dst = (word * (dhi - dlo + 1)).from_address(dst_addr + dlo * itemsize)
    for idx in itertools.product(*(range(n) for n in shape)):
        so = src_base - slo
        do = dst_base - dlo
        for i, s, d in zip(idx, src_strides, dst_strides):
            so += i * s
            do += i * d
        dst[do] = src[so]


def _set_meta(a, shape, dtype, order):
    a.shape = shape
    a.dtype = dtype
    a.ndim = len(shape)
    a.size = _prod(shape)
    a.itemsize = _ITEMSIZE[dtype]
    a.nbytes = a.size * a.itemsize
    a.order = order
    a.strides = _c_strides(shape, a.itemsize) if order == "C" else _f_strides(shape, a.itemsize)
    if len(a._mv) != a.size:
        raise ValueError(
            f"mojolearn: buffer holds {len(a._mv)} elements, shape "
            f"{shape} needs {a.size}"
        )


def _init_owned(a, store, shape, dtype, order):
    a._store = store
    a._base = None
    a._pin = None
    a._mv = memoryview(store)
    if a._mv.format != _CODE[dtype]:
        a._mv = a._mv.cast("B").cast(_CODE[dtype])
    a._addr = store.buffer_info()[0]
    a._readonly = False
    _set_meta(a, shape, dtype, order)
    return a


def _both_orders(a):
    return sum(1 for s in a.shape if s != 1) <= 1


def _row_view(a, i, shape, step):
    v = Array.__new__(Array)
    v._store = a._store
    v._base = a
    v._pin = a._pin
    off = i * step
    v._mv = a._mv[off:off + step]
    v._addr = a._addr + off * a.itemsize
    v._readonly = a._readonly
    _set_meta(v, shape, a.dtype, "C")
    return v

_FLOAT = {"<f4", "<f8", "<f2"}
_INT = {"<i4", "<i8", "<u4", "<u1", "<u2", "<i1"}

#: True only inside tests/test_hotpath_native.py's reference arm.
_REFERENCE = False


def _reference():
    """True only inside the differential test's reference arm."""
    return _REFERENCE


def _rebuild(key):
    return RuntimeError(f"mojolearn: this binary has no `{key}` helper (an older build); rebuild it")


def _helper(key, size):
    """The core helper `key` for a non-empty block, or None for an empty
    block (whose Python routine answers or refuses without a loop) or a
    binary that lacks the helper."""
    if size <= 0:
        return None
    from mojolearn import _buffer  # oracle subject: `_buffer._native_optional`, the helper seam tests/test_hotpath_native.py plants and removes (`_buffer._NATIVE`) for its reference arm; the native core helpers have no public route
    return _buffer._native_optional(key)


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
    if dtype in ("<f4", "<f2"):
        f = "<f" if dtype == "<f4" else "<e"
        try:
            g = struct.unpack(f, struct.pack(f, value))[0]
        except OverflowError:
            return None
        if g != value:
            return None
        return g
    return value


def _astype_reference(self, dtype):
    """`astype`'s Python definition (the differential test's reference)."""
    src = self.dtype
    code = _CODE[dtype]
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
    _set_meta(out, self.shape, dtype, self.order)
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
    store = _typed_input.output_store(_CODE[dtype], self.size)
    if int(fn(self._addr, sc, store.buffer_info()[0], dc, self.size)):
        return None
    return Array._owned(store, self.shape, dtype, self.order)

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

def _no_reduce(self, what):
    """An empty Array refuses `what`; else the helper is missing: the
    reference arm answers in Python, a production binary is rebuilt."""
    if self.size == 0:
        raise ValueError(f"mojolearn: {what} of an empty Array")
    if not _reference():
        raise _rebuild("reduce_stat")


class _ArrayReference:
    """The Python definitions of Array's public operations, kept ONLY as the
    test oracles of `bindings/array_native_door.mojo` (test_hotpath_native.py,
    test_native_array_operations.py). The product never calls them: every
    public Array operation is one `array_native_call`. Called as
    `_ArrayReference.<name>(array, ...)`; never instantiated."""

    def __init__(self, shape, dtype="<f4"):
        dtype = normalize_dtype(dtype)
        shape = _check_shape(shape)
        store = _new_store(_CODE[dtype], _prod(shape))
        _init_owned(self, store, shape, dtype, "C")

    def flags(self):
        """``{"C_CONTIGUOUS", "F_CONTIGUOUS", "WRITEABLE"}`` as booleans."""
        both = _both_orders(self)
        return {
            "C_CONTIGUOUS": self.order == "C" or both,
            "F_CONTIGUOUS": self.order == "F" or both,
            "WRITEABLE": not self._readonly,
        }

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
            # A 0-d view (shape ()): NumPy reads the buffer before
            # __array_interface__, and the flat store made a 0-d Array 1-D.
            return self._mv.cast("B").cast(_CODE[self.dtype], ())
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

    def __reduce__(self):
        return (copyreg.__newobj__, (type(self),),
                (self.tobytes(), self.shape, self.dtype, self.order, self._readonly))

    def tobytes(self):
        """The raw bytes, in storage order (row-major for a C-order Array, column-major for an F-order one)."""
        return self._mv.tobytes()

    def tolist(self):
        """The values as nested Python lists (a scalar for a 0-d Array)."""
        a = self._as_c()
        if self.ndim == 0:
            return a._values()[0]
        return _nest(a)

    def copy(self):
        """A new Array with the same dtype, shape, order and values."""
        store = array.array(_CODE[self.dtype])
        store.frombytes(self._mv.cast("B"))  # one memcpy
        return Array._owned(store, self.shape, self.dtype, self.order)

    def astype(self, dtype):
        """A converted copy (always a copy, even for the same dtype).

        Float to ``"<f4"`` rounds to nearest even; float to ``"<f8"`` and integers
        up to 2**53 to float are exact. Float to integer truncates toward zero, like
        NumPy. The memory order is kept.

        Parameters
        ----------
        dtype : str
            Target element type.

        Returns
        -------
        Array
        """
        dtype = normalize_dtype(dtype)
        if dtype == self.dtype:
            return _ArrayReference.copy(self)
        src = self.dtype
        code = _CODE[dtype]
        if self.size == 0:
            return Array._owned(_new_store(code, 0), self.shape, dtype, self.order)
        fast = _native_astype(self, dtype)
        if fast is not None:
            return fast
        if _reference():
            return _astype_reference(self, dtype)
        if _helper("cast_elements", self.size) is None:
            raise _rebuild("cast_elements")
        # the helper refused an element (a NaN, an infinity or a value
        # outside the target): the Python definition raises its own words
        # (it never returns here: every element it accepts, the helper did)
        return _astype_reference(self, dtype)

    def reshape(self, shape):
        """A C-order view with a new shape (one dimension may be ``-1``).

        An F-order Array is copied to C order first, as NumPy does.

        Parameters
        ----------
        shape : int or tuple of int

        Returns
        -------
        Array
        """
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
        """A 1-D view, same as ``reshape(-1)``."""
        return _ArrayReference.reshape(self, (-1,))

    def __len__(self):
        # PYTHON COMPATIBILITY EXCEPTION (Andrew, 2026-10-25): native
        # Voting/Stacking/MultiOutput/Calibrated meta-estimator paths still
        # call this owned metadata method through len(Array). It only reads
        # dimensional metadata; it does not compute model results. Retained
        # under the owner's fallback authorization after Codex's Daybreak
        # screening interrupted that lane. This is an incomplete native
        # migration, not a demonstrated Mojo limitation. Verification records
        # array-length-metadata explicitly and never calls it Python-free.
        # See _runtime_fallbacks.py and Modular issue 7399 for the context.
        # The owner also permits Python runtime calls where a documented Mojo
        # bug requires them, including calls from Mojo. That policy does not
        # turn this retained length callback into a confirmed Mojo bug.
        if self.ndim == 0:
            raise TypeError("len() of a 0-d Array")
        return self.shape[0]

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d Array")
        if self.ndim == 1:
            return iter(self._values())
        return _RowIter(self._as_c())

    def __repr__(self):
        return f"Array(shape={self.shape}, dtype={self.dtype!r})"

    def __bool__(self):
        if self.size != 1:
            raise ValueError(
                "mojolearn: the truth value of an Array with more than one "
                "element is ambiguous"
            )
        return bool(self._values()[0])

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
                if a.dtype == "<f4":
                    # The contiguous block above (`_typed_input.same_dtype_store`)
                    # quiets a float32 signaling NaN, as the item setter does;
                    # a strided selection gets the same float32 round trip
                    # (the in-place `cast_elements` float32 -> float32 it
                    # stands for) so a copy's bits do not depend on the
                    # selection's layout.
                    store = array.array("f", store.tolist())
        return Array._owned(store, out_shape, a.dtype, "C")

    def min(self):
        """Smallest element, in storage order; ValueError for an empty Array."""
        fast = _native_reduce(self, _REDUCE_MIN)
        if fast is not None:
            return fast
        _no_reduce(self, "min")
        return min(_reduce_values(self, "min"))

    def max(self):
        """Largest element, in storage order; ValueError for an empty Array."""
        fast = _native_reduce(self, _REDUCE_MAX)
        if fast is not None:
            return fast
        _no_reduce(self, "max")
        return max(_reduce_values(self, "max"))

    def sum(self):
        """Sum of all elements: an exact ``int`` for integer dtypes, a float summed in storage order for float dtypes."""
        if self.size == 0:
            return 0 if self.dtype in _INT else 0.0
        if self.dtype in ("<f4", "<f8", "<f2"):
            fast = _native_reduce(self, _REDUCE_SUM)
            if fast is not None:
                return fast
        elif self.dtype in _NATIVE_CODE:
            # lane apple-fast-py2mojo-core: the exact integer sum in the
            # helper, (hi, lo >> 32, lo & 0xffffffff) of a 128-bit total
            parts = _native_reduce(self, _REDUCE_ISUM)
            if parts is not None:
                hi, mid, low = (int(v) for v in parts)  # glue: three words of a 128-bit sum
                return (hi << 64) + (mid << 32) + low
        if not _reference():
            raise _rebuild("reduce_stat")
        values = self._values()
        if self.dtype in _INT:
            return sum(values)
        acc = 0.0
        for v in values:
            acc += v
        return acc

    def argmax(self):
        """Flat (C-order) index of the first maximum."""
        fast = _native_reduce(self._as_c(), _REDUCE_ARGMAX)
        if fast is not None:
            return fast
        _no_reduce(self, "argmax")
        values = _reduce_values(self._as_c(), "argmax")
        best = 0
        best_v = values[0]
        for i in range(1, len(values)):
            v = values[i]
            if v > best_v:
                best = i
                best_v = v
        return best

    def __eq__(self, other):
        """Elementwise equality against an Array of the same shape or a
        scalar, as a `'<u1'` Array of 0/1."""
        if isinstance(other, Array) and other.shape == self.shape:
            fast = _native_eq(self, other)
            if fast is not None:
                return fast
        elif isinstance(other, (int, float, bool)):
            if self.size == 0:
                return Array._owned(array.array("B"), self.shape, "<u1", "C")
            fast = _native_eq_scalar(self, other)
            if fast is not None:
                return fast
            if not _reference():
                raise _rebuild("equal_elements")
        if isinstance(other, Array) and other.shape == self.shape and not _reference():
            if self.size == 0:
                return Array._owned(array.array("B"), self.shape, "<u1", "C")
            if other.dtype == self.dtype:
                raise _rebuild("equal_elements")
            # two dtypes (lane py-runtime round 2): both widened EXACTLY to one
            # dtype by the cast helper, then one native comparison. int64
            # against a float is the one pair no exact widening covers.
            a_int, b_int = self.dtype in _INT, other.dtype in _INT
            wide = "<i8" if (a_int and b_int) else ("<f8" if "<i8" not in (self.dtype, other.dtype) else None)
            if wide is not None:
                return self.astype(wide).__eq__(other.astype(wide))
            # int64 against a float (lane py-runtime round 3): the float side
            # widened exactly to float64, then the helper's exact int == float
            fn = _helper("equal_elements", self.size)
            if fn is None:
                raise _rebuild("equal_elements")
            ints, flts = (self, other) if self.dtype in _INT else (other, self)
            ia = ints.astype("<i8")._as_c() if ints.dtype != "<i8" else ints._as_c()
            fa = flts.astype("<f8")._as_c() if flts.dtype != "<f8" else flts._as_c()
            store = array.array("B", bytes(self.size))
            fn(ia._addr, fa._addr, _EQ_I64_F64, self.size, store.buffer_info()[0])
            return Array._owned(store, self.shape, "<u1", "C")
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

    def __ne__(self, other):
        eq = _ArrayReference.__eq__(self, other)
        if eq is NotImplemented:
            return eq
        # `bytes.translate` flips 0 and 1 in C (was a comprehension per element)
        return Array._owned(
            array.array("B", eq._mv.tobytes().translate(_NOT)), self.shape, "<u1", "C"
        )

    def from_list(cls, nested, dtype="<f8"):
        """A new C-order Array from a (nested) list of Python numbers.

        The shape is inferred and must be rectangular. Floats converted to
        ``"<f4"`` round to nearest even. Floats are not converted to integer dtypes
        (an error is raised instead of truncating), and integers too large for
        the dtype raise an error rather than wrap.

        Parameters
        ----------
        nested : list
            Nested lists of numbers.
        dtype : str, default "<f8"
            Element type.

        Returns
        -------
        Array
        """
        dtype = normalize_dtype(dtype)
        shape, flat = _flatten(nested)
        return cls._from_flat(flat, shape, dtype)



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
        return _row_view(self._a, i, self._shape, self._step)

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
    <memoryview slice>)` copies it replaces, or None for a short run. One memcpy instead of one Python object
    per element (0.2 ns against 22); see `_typed_input.same_dtype_store` for the
    one float32 subtlety."""
    if length < _NATIVE_MIN:
        return None  # a short run: the per-element copy is as fast as the detour
    raw = a._mv[offset:offset + length].cast("B")
    return _typed_input.same_dtype_store(raw, a.dtype)

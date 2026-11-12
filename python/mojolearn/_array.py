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
import copyreg
import mmap as _array_mmap
import functools
import operator
import struct
import sys
from itertools import chain
from types import BuiltinFunctionType

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
# kept it for its memcpy), and the integer sum is the helper's too
# (`_REDUCE_ISUM`, an exact 128-bit accumulator). The Python routines stay
# as the helpers' definitions: `_native_optional` answers None only for a
# binary that lacks the helper, the state the differential test's reference
# arm (`tests/test_hotpath_native.py`) plants. Those Python definitions and
# their helper wrappers live in python/mojolearn/tests/_array_reference.py
# (moved out of the product, lane p5-cpu-route-relabel-and-dead-code).
_NATIVE_CODE = {"<f4": 0, "<f8": 1, "<i4": 2, "<i8": 3, "<u4": 4, "<u1": 5,
                # lane py-runtime round 2: bf16 bits, int8 and float16 too
                "<u2": 6, "<i1": 7, "<f2": 8}
#: equal_elements' exact int64-against-float64 code (lane py-runtime round 3)
_EQ_I64_F64 = 9


_NATIVE_MIN = 256
_REDUCE_MIN, _REDUCE_MAX, _REDUCE_SUM, _REDUCE_ARGMAX, _REDUCE_INTEGRAL, _REDUCE_ISUM = range(6)
_SCALAR_TYPES = frozenset((float, int, bool))
_ROW_TYPES = frozenset((list, tuple))
_NOT = bytes((1, 0)) + bytes(range(2, 256))


_INT_BOUNDS = {"<i4": (-(1 << 31), (1 << 31) - 1), "<i8": (-(1 << 63), (1 << 63) - 1),
               "<u4": (0, (1 << 32) - 1), "<u1": (0, 255), "<u2": (0, 65535), "<i1": (-128, 127)}


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


# The shape, stride, size and layout arithmetic of an Array (C and F strides,
# the element count, `_check_shape`, the zero-filled store and the C <-> F
# reorder) is `bindings/array_native.mojo` (`_c_strides`, `_f_strides`,
# `_prod`, `_checked_shape`, `_new_store`, `_reorder`); the Python copies are
# the test oracle in python/mojolearn/tests/_array_reference.py (lane
# pyout-runtime-boundary, 2026-11-11).


def _to_py_values(mv, dtype):
    """Python scalars for a flat store memoryview."""
    if dtype == "<f2":
        n = len(mv)
        return list(struct.unpack("<%de" % n, mv.tobytes())) if n else []
    return mv.tolist()


class _ArrayInterface:
    """Carries only an Array's `__array_interface__`, so `Array.__array__`
    can hand NumPy a zero-copy view without recursing into itself. NumPy
    keeps this object (and through it the Array) alive as the view's base."""

    __slots__ = ("_owner", "__array_interface__")

    def __init__(self, owner):
        self._owner = owner
        self.__array_interface__ = owner.__array_interface__


# A typed, shaped, contiguous block of memory. See the module docstring.
#
# `Array(shape, dtype)` allocates zeros; `empty`, `zeros`, `full`,
# `frombytes`, `from_list` and `from_buffer` are the usual constructors.
class Array:
    """The array type mojolearn returns: typed, shaped, contiguous memory.

    mojolearn does not require NumPy, so estimators return ``Array`` objects.
    An ``Array`` is a C- or F-contiguous block that either owns its memory or,
    through :meth:`from_buffer`, borrows it from another buffer it keeps alive.
    Its memory address never changes during its lifetime.

    With NumPy installed, ``numpy.asarray(a)`` is a zero-copy view (through
    ``__array_interface__``). On Python 3.12 and later ``memoryview(a)`` is
    zero-copy too.

    ``Array`` is a container, not a NumPy replacement: there are no arithmetic
    operators, no broadcasting and no fancy indexing. Indexing with ints and
    slices is supported (a full integer index returns a Python scalar, anything
    else a C-order copy), as are :meth:`tolist`, :meth:`reshape`,
    :meth:`astype`, ``==`` and the reductions :meth:`min`, :meth:`max`,
    :meth:`sum` and :meth:`argmax`, so you can inspect results without NumPy.
    ``'<f2'`` (float16) values are stored, not computed with.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the new, zero-filled array.
    dtype : str, default "<f4"
        One of ``"<f4"``, ``"<f8"``, ``"<f2"``, ``"<i4"``, ``"<i8"``, ``"<i1"``,
        ``"<u4"``, ``"<u2"``, ``"<u1"``.

    Attributes
    ----------
    shape : tuple of int
    dtype : str
    ndim, size, nbytes, itemsize : int
    strides : tuple of int
    order : {"C", "F"}
    flags : dict
        ``C_CONTIGUOUS``, ``F_CONTIGUOUS`` and ``WRITEABLE``.

    Examples
    --------
    >>> from mojolearn import Array
    >>> a = Array.from_list([[1.0, 2.0], [3.0, 4.0]], "<f4")
    >>> a.shape, a.sum()
    ((2, 2), 10.0)
    >>> import numpy as np
    >>> np.asarray(a).dtype
    dtype('float32')
    """

    __slots__ = (
        "_store", "_base", "_pin", "_mv", "_addr", "_readonly",
        "shape", "dtype", "ndim", "size", "nbytes", "itemsize", "strides",
        "order",
    )

    def __init__(self, shape, dtype="<f4"):
        return _ARRAY_NATIVE("init", self, _ARRAY_CONTEXT, shape, dtype)

    # ------------------------------------------------------------ construction
    @classmethod
    def _owned(cls, store, shape, dtype, order="C"):
        """A `cls` that owns `store` (an `array.array`) as `shape`, `dtype`
        and `order` (the shape, stride and size check are native)."""
        return _ARRAY_NATIVE("owned", cls, _ARRAY_CONTEXT, store, shape, dtype, order)

    def _view_of(self, shape, order="C"):
        """Another shape over the SAME memory (no copy, no ownership
        transfer): `self` stays alive through the view's `_base`. Called as
        `Array._view_of(other, shape, order)`."""
        return _ARRAY_NATIVE("view_of", self, _ARRAY_CONTEXT, shape, order)

    @classmethod
    def from_buffer(cls, obj):
        """A zero-copy view of any contiguous buffer-protocol object.

        The result keeps ``obj`` alive but does not own it; writes through the
        result reach ``obj``.

        Parameters
        ----------
        obj : buffer-protocol object
            For example a NumPy array, ``array.array``, ``bytearray`` or
            ``memoryview``.

        Returns
        -------
        Array

        Raises
        ------
        ValueError, TypeError
            For a non-contiguous buffer or an unsupported format (no copy is made).
        """
        return _ARRAY_NATIVE("from_buffer", cls, _ARRAY_CONTEXT, obj)

    @classmethod
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
        return _ARRAY_NATIVE("from_list", cls, _ARRAY_CONTEXT, nested, dtype)

    @classmethod
    def _from_flat(cls, flat, shape, dtype):
        """A C-order `cls` of the Python values `flat` (native store build)."""
        return _ARRAY_NATIVE("from_flat", cls, _ARRAY_CONTEXT, flat, shape, dtype)

    # ------------------------------------------------------------- interface
    def _has_order(self, order):
        """True when the block is laid out as `order`, or when at most one
        dimension is larger than 1 so either label is free (native)."""
        # PERF FIX (perf-audit-3, 2026-11-11): the label compare the Python
        # method answered on before the native move, kept ahead of the
        # binding crossing (the common case of every public call).
        if self.order == order:  # glue: one attribute compare, no data
            return True
        return _ARRAY_NATIVE("has_order", self, _ARRAY_CONTEXT, order)

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
        return _ARRAY_NATIVE("array", self, _ARRAY_CONTEXT, dtype, copy)

    def _flat(self):
        """A 1-D Array over the storage IN STORAGE ORDER, no copy. For an
        F-order matrix that is the column-major flat, `flat[f * rows + r]
        == a[r, f]`; the shape used by the gbdt quantizer."""
        return Array._view_of(self, (self.size,), "C")

    # ------------------------------------------------------------- basic ops
    # The protocol methods (`__len__`, `__getitem__`, `__eq__`, `__ne__`,
    # `__repr__`, `__bool__`, `__iter__`, `__reduce__`, `__setstate__`,
    # `__buffer__`) and the `flags` / `__array_interface__` properties are
    # set below the class: each is a C `property` over
    # `functools.partial(array_native_call, op, context)`, so `len(a)`,
    # `a[i]`, `a == b`, `repr(a)`, pickling and NumPy's interface read reach
    # the native entry without entering a Python frame.

    def tobytes(self):
        """The raw bytes, in storage order (row-major for a C-order Array, column-major for an F-order one)."""
        return _ARRAY_NATIVE("tobytes", self, _ARRAY_CONTEXT)

    def _values(self):
        return _to_py_values(self._mv, self.dtype)

    def _as_c(self):
        """Self if C-order, a relabeled view when both layouts coincide,
        else a C-order copy (native)."""
        # PERF FIX (perf-audit-3): self on the label compare, no crossing.
        if self.order == "C":  # glue: one attribute compare, no data
            return self
        return _ARRAY_NATIVE("as_c", self, _ARRAY_CONTEXT)

    def _as_order(self, order):
        """Self when already laid out as `order`, a relabeled view when
        both layouts coincide, else a copy in `order` (native)."""
        # PERF FIX (perf-audit-3): self on the label compare, no crossing.
        if order == self.order:  # glue: one attribute compare, no data
            return self
        return _ARRAY_NATIVE("as_order", self, _ARRAY_CONTEXT, order)

    def tolist(self):
        """The values as nested Python lists (a scalar for a 0-d Array)."""
        return _ARRAY_NATIVE("tolist", self, _ARRAY_CONTEXT)

    def copy(self):
        """A new Array with the same dtype, shape, order and values."""
        return _ARRAY_NATIVE("copy", self, _ARRAY_CONTEXT)

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
        return _ARRAY_NATIVE("astype", self, _ARRAY_CONTEXT, dtype)

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
        return _ARRAY_NATIVE("reshape", self, _ARRAY_CONTEXT, shape)

    def ravel(self):
        """A 1-D view, same as ``reshape(-1)``."""
        return _ARRAY_NATIVE("ravel", self, _ARRAY_CONTEXT)

    __hash__ = None

    def min(self):
        """Smallest element, in storage order; ValueError for an empty Array."""
        return _ARRAY_NATIVE("min", self, _ARRAY_CONTEXT)

    def max(self):
        """Largest element, in storage order; ValueError for an empty Array."""
        return _ARRAY_NATIVE("max", self, _ARRAY_CONTEXT)

    def sum(self):
        """Sum of all elements: an exact ``int`` for integer dtypes, a float summed in storage order for float dtypes."""
        return _ARRAY_NATIVE("sum", self, _ARRAY_CONTEXT)

    def argmax(self):
        """Flat (C-order) index of the first maximum."""
        return _ARRAY_NATIVE("argmax", self, _ARRAY_CONTEXT)


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
    # the general walk (a ragged or mixed nested list, Arrays inside it) reads
    # the user's list once natively (bindings/user_objects_native.mojo nested)
    from . import _user_objects
    shape, flat = _user_objects.call("nested", nested, Array)
    return tuple(shape), flat


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


# Every public Array operation is one call of this native entry
# (bindings/array_native.mojo, in the tokenizer host binding). `_buffer`
# imports this module, so the binding is loaded here directly; the core
# helpers the operations share (cast_elements, reduce_stat,
# strided_copy_bytes) are connected by `_buffer` once it is defined.
def _array_binding():
    from . import _backend
    return _backend.load_host_module("_mojolearn_tokenizer_host")  # import-time: host-memory Array container operations, binding loaded once


_ARRAY_NATIVE = _array_binding().array_native_call
if type(_ARRAY_NATIVE) is not BuiltinFunctionType:
    raise TypeError("mojolearn: Array operations require a compiled native entry")
_ARRAY_CONTEXT = dict(
    array_class=Array, interface_class=_ArrayInterface, array_type=array.array,
    codes=dict(_CODE), itemsizes=dict(_ITEMSIZE), native_codes=dict(_NATIVE_CODE),
    aliases=dict(_ALIAS), supported=SUPPORTED_DTYPES, native_min=_NATIVE_MIN,
    empty_address=_EMPTY_ARRAY_STORAGE.buffer_info()[0], empty_storage=_EMPTY_ARRAY_STORAGE,
    struct_pack=struct.pack, struct_unpack=struct.unpack, operator=operator,
    partial=functools.partial, sys_modules=sys.modules, modules=sys.modules, newobj=copyreg.__newobj__,
    entry=_ARRAY_NATIVE, helpers={}, int_bounds=dict(_INT_BOUNDS), infinity=float("inf"),
    minus_infinity=float("-inf"), scalar_types=(int, float, bool), eq_i64_f64=_EQ_I64_F64,
    not_implemented=NotImplemented, not_table=_NOT, scalar_type_set=_SCALAR_TYPES,
    row_type_set=_ROW_TYPES, chain_from_iterable=chain.from_iterable,
    user_objects=_array_binding().user_objects_native_call,
    from_buffer_mmap=_array_mmap.mmap,
    from_buffer_mmap_hooks={name: vars(_array_mmap.mmap).get(name) for name in ("__buffer__", "__release_buffer__")},  # glue: capture native mmap protocol identities at import
    from_buffer_slots={name: Array.__dict__[name] for name in Array.__slots__})  # glue: retain fixed native Array member descriptor identities


# glue: retain canonical native owner identities once, after Array and its
# storage context exist; fitted-buffer replacement admission reads this table.
from ._native_owner_lifetime import connect_owner_context as _connect_array_owners
_connect_array_owners(_ARRAY_CONTEXT)


def _bound(op):
    """`a.<op>(*args)` as `array_native_call(op, _ARRAY_CONTEXT, a, *args)`:
    a C `property` whose getter is a C `functools.partial` building another
    one, so the protocol slots and attribute reads enter no Python frame."""
    return property(functools.partial(functools.partial, _ARRAY_NATIVE, op, _ARRAY_CONTEXT))


Array.__len__ = _bound("len")
Array.__iter__ = _bound("iter")
Array.__repr__ = _bound("repr")
Array.__bool__ = _bound("bool")
Array.__getitem__ = _bound("getitem")
Array.__eq__ = _bound("eq")
Array.__ne__ = _bound("ne")
Array.__reduce__ = _bound("reduce")
Array.__setstate__ = _bound("setstate")
Array.__buffer__ = _bound("buffer")
Array.flags = property(functools.partial(_ARRAY_NATIVE, "flags", _ARRAY_CONTEXT),
                       doc='``{"C_CONTIGUOUS", "F_CONTIGUOUS", "WRITEABLE"}`` as booleans.')
Array.__array_interface__ = property(functools.partial(_ARRAY_NATIVE, "interface", _ARRAY_CONTEXT))


def connect_helpers(resolve):
    """Capture the shared core helpers once (`_buffer._native_optional`);
    an absent helper (None) is refused by name by the operation that needs it.
    The names are literal so the lane selector sees exactly these exports."""
    found = (("cast_elements", resolve("cast_elements")), ("reduce_stat", resolve("reduce_stat")),
             ("strided_copy_bytes", resolve("strided_copy_bytes")),
             ("equal_elements", resolve("equal_elements")))
    _ARRAY_CONTEXT["helpers"].update(item for item in found if item[1] is not None)  # glue: four fixed helper names
    # glue: once, the base helper table key the native Array operations read
    # (lane calltable-base-helpers); a binary without one leaves it absent and
    # the operation that needs it refuses by name
    from ._buffer import base_native_key
    try:
        _ARRAY_CONTEXT["helpers"]["base_native_key"] = base_native_key()
    except Exception:  # noqa: BLE001 - absent key: the native operation refuses
        pass


# SmallMLP's native strided-Array admission consumes the same helper owner.
# Helpers added during core connection mutate this dictionary in place.
from ._native_loaded_modules import connect_module as _connect_native_module
_connect_native_module(__name__, sys.modules[__name__], ('_ARRAY_CONTEXT',))

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
import math
import struct

# typestr -> array.array typecode of the backing store
_CODE = {
    "<f4": "f", "<f8": "d", "<i4": "i", "<i8": "q",
    "<u4": "I", "<u1": "B", "<f2": "H",
}
_ITEMSIZE = {"<f4": 4, "<f8": 8, "<i4": 4, "<i8": 8, "<u4": 4, "<u1": 1, "<f2": 2}
_FLOAT = {"<f4", "<f8", "<f2"}
_INT = {"<i4", "<i8", "<u4", "<u1"}
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
    "|u1": "<u1", "=f4": "<f4", "=f8": "<f8", "=i4": "<i4", "=i8": "<i8",
    "=u4": "<u4", "=f2": "<f2", "f4": "<f4", "f8": "<f8", "i4": "<i4",
    "i8": "<i8", "u4": "<u4", "u1": "<u1", "f2": "<f2",
    "float": "<f8", "int": "<i8", "bool": "<u1", "bool_": "<u1",
}


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


def _reorder(src_mv, shape, src_order, dst_mv):
    """Copy the elements of `src_mv` (flat, laid out in `src_order`) into
    `dst_mv` (flat, the other order). Both are typed memoryviews of the
    same format and length.

    The loop is over the OUTER product of dimensions; every iteration moves
    `shape[0]` (C -> F) or `shape[-1]` (F -> C) elements with one strided
    memoryview slice assignment, which CPython performs in C. For a 2-D
    matrix that is one Python iteration per column, not per element. The
    only per-element Python work is the index arithmetic of the outer loop.
    """
    ndim = len(shape)
    if ndim <= 1 or len(src_mv) == 0:
        dst_mv[:] = src_mv
        return
    cs = _c_strides(shape, 1)
    fs = _f_strides(shape, 1)
    if src_order == "C":
        # destination F: fastest axis 0; source axis-0 stride in elements
        run = shape[0]
        step = cs[0]
        outer = shape[1:]
        outer_src = cs[1:]
        outer_dst = fs[1:]
    else:
        run = shape[-1]
        step = fs[-1]
        outer = shape[:-1]
        outer_src = fs[:-1]
        outer_dst = cs[:-1]
    n_outer = _prod(outer)
    idx = [0] * len(outer)
    for _ in range(n_outer):
        so = 0
        do = 0
        for k, i in enumerate(idx):
            so += i * outer_src[k]
            do += i * outer_dst[k]
        dst_mv[do:do + run] = src_mv[so:so + run * step:step]
        # increment the mixed-radix counter (row-major over `outer`)
        for k in range(len(outer) - 1, -1, -1):
            idx[k] += 1
            if idx[k] < outer[k]:
                break
            idx[k] = 0


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
        return sum(1 for s in self.shape if s != 1) <= 1

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
            "data": (self._addr, bool(self._readonly)),
            "strides": None if self.order == "C" else self.strides,
            "version": 3,
        }

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
        return self._mv.cast("B").cast(_CODE[self.dtype], self.shape)

    def _flat(self):
        """A 1-D Array over the storage IN STORAGE ORDER, no copy. For an
        F-order matrix that is the column-major flat, `flat[f * rows + r]
        == a[r, f]`; the shape used by the gbdt quantizer."""
        return Array._view_of(self, (self.size,), "C")

    # ------------------------------------------------------------- basic ops
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
        _reorder(self._mv, self.shape, "F", memoryview(out))
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
        _reorder(self._mv, self.shape, "C", memoryview(out))
        return Array._owned(out, self.shape, self.dtype, "F")

    def tolist(self):
        """Nested Python lists by shape (a scalar for a 0-d Array)."""
        values = self._as_c()._values()
        if self.ndim == 0:
            return values[0]
        return _nest(values, self.shape)

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

    def reshape(self, shape):
        """A C-order view over the same buffer (one `-1` allowed). An
        F-order Array is copied to C order first, as NumPy would."""
        if isinstance(shape, int):
            shape = (shape,)
        shape = [int(s) for s in shape]
        if shape.count(-1) > 1:
            raise ValueError("mojolearn: reshape allows one -1")
        if -1 in shape:
            known = _prod(s for s in shape if s != -1)
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
        return (self[i] for i in range(self.shape[0]))

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
        for axis, k in enumerate(key):
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
        for axis in range(len(key), a.ndim):
            dims.append((0, 1, a.shape[axis]))
        out_shape = tuple(d[2] for d in dims if isinstance(d, tuple))
        cs = _c_strides(a.shape, 1)
        # Every outer combination is one Python iteration; the innermost
        # run is one memoryview slice (strided or not), copied in C.
        outer = [d for d in dims[:-1]]
        last = dims[-1] if dims else (0, 1, 1)
        if isinstance(last, tuple):
            l_start, l_step, l_count = last
        else:
            l_start, l_step, l_count = last, 1, 1
        base_off = l_start
        outer_ranges = []
        for axis, d in enumerate(outer):
            if isinstance(d, tuple):
                outer_ranges.append([(d[0] + i * d[1]) * cs[axis] for i in range(d[2])])
            else:
                base_off += d * cs[axis]
        src = a._mv
        code = _CODE[a.dtype]
        if not outer_ranges:
            # the constructor copies a (possibly strided) typed memoryview
            # in C; `frombytes` would insist on a byte-format buffer
            store = array.array(
                code, _run(src, base_off, l_count, l_step)
            ) if l_count else array.array(code)
        else:
            total = _prod(len(r) for r in outer_ranges) * l_count
            store = _new_store(code, total)
            store_mv = memoryview(store)
            pos = 0
            for off in _offsets(outer_ranges):
                store_mv[pos:pos + l_count] = _run(src, base_off + off, l_count, l_step)
                pos += l_count
        if not out_shape:
            return _to_py_values(memoryview(store), a.dtype)[0]
        return Array._owned(store, out_shape, a.dtype, "C")

    def __eq__(self, other):
        """Elementwise equality against an Array of the same shape or a
        scalar, as a `'<u1'` Array of 0/1."""
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
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return eq
        return Array._owned(
            array.array("B", [1 - b for b in eq._mv]), self.shape, "<u1", "C"
        )

    __hash__ = None

    # ------------------------------------------------------------ reductions
    # HOST reductions in Python scalars, sequential in storage order. They
    # are NOT part of any identity claim and no kernel path uses them.
    def _reduce_values(self, what):
        if self.size == 0:
            raise ValueError(f"mojolearn: {what} of an empty Array")
        return self._values()

    def min(self):
        return min(self._reduce_values("min"))

    def max(self):
        return max(self._reduce_values("max"))

    def sum(self):
        """Sequential accumulation: exact `int` for int dtypes, a Python
        float summed left to right in storage order for float dtypes."""
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
        values = self._as_c()._reduce_values("argmax")
        best = 0
        best_v = values[0]
        for i in range(1, len(values)):
            v = values[i]
            if v > best_v:
                best = i
                best_v = v
        return best


def _run(mv, start, count, step):
    """`count` elements of a flat memoryview from `start` with `step`,
    which may be negative; a computed stop below zero would mean "the
    end" to a slice, so it is spelled as None."""
    stop = start + count * step
    if step < 0 and stop < 0:
        stop = None
    return mv[start:stop:step]


def _offsets(ranges):
    """Every sum of one element from each list in `ranges`, in row-major
    order; `ranges` is short (one list per outer axis)."""
    if len(ranges) == 1:
        yield from ranges[0]
        return
    for head in ranges[0]:
        for rest in _offsets(ranges[1:]):
            yield head + rest


def _flatten(nested):
    """`(shape, flat_list)` for a rectangular nested sequence; a scalar
    gives shape ()."""
    if isinstance(nested, Array):
        return nested.shape, nested._as_c()._values()
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


def _nest(values, shape):
    if len(shape) == 1:
        return list(values)
    n = len(shape)
    step = _prod(shape[1:])
    return [_nest(values[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]


def isfinite_all_py(values):
    """Reference finiteness test, a Python loop; `_buffer.all_finite` uses
    it only as the confirmation path."""
    for v in values:
        if not math.isfinite(v):
            return False
    return True

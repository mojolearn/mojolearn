# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python definitions of `mojolearn.Array`'s public operations: the test
oracles of `bindings/array_native.mojo` (test_hotpath_native.py,
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

from mojolearn._array import (
    Array, _ArrayInterface, _EMPTY_ARRAY_STORAGE, _CODE, _NATIVE_CODE, _NATIVE_MIN, _EQ_I64_F64,
    _REDUCE_MIN, _REDUCE_MAX, _REDUCE_SUM, _REDUCE_ARGMAX, _REDUCE_ISUM, _NOT,
    _INT_BOUNDS, normalize_dtype, _check_shape, _new_store, _prod,
    _nest, _c_strides, _to_py_values,
    _strided_copy, _flatten,
)
import struct
import sys

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
    from mojolearn import _buffer
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
    from mojolearn import _buffer
    store = _buffer._output_store(_CODE[dtype], self.size)
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
    test oracles of `bindings/array_native.mojo` (test_hotpath_native.py,
    test_native_array_operations.py). The product never calls them: every
    public Array operation is one `array_native_call`. Called as
    `_ArrayReference.<name>(array, ...)`; never instantiated."""

    def __init__(self, shape, dtype="<f4"):
        dtype = normalize_dtype(dtype)
        shape = _check_shape(shape)
        store = _new_store(_CODE[dtype], _prod(shape))
        self._init_owned(store, shape, dtype, "C")

    def flags(self):
        """``{"C_CONTIGUOUS", "F_CONTIGUOUS", "WRITEABLE"}`` as booleans."""
        both = self._both_orders()
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
                    # The contiguous block above (`_buffer._same_dtype_store`)
                    # quiets a float32 signaling NaN, as the item setter does;
                    # a strided selection gets the same in-place
                    # `cast_elements` float32 -> float32 so a copy's bits do
                    # not depend on the selection's layout.
                    from mojolearn import _buffer
                    fn = _buffer._native_optional("cast_elements")
                    if fn is not None:
                        at = store.buffer_info()[0]
                        fn(at, 0, at, 0, total)  # in place: slot i reads and writes slot i only
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
    from mojolearn import _buffer
    raw = a._mv[offset:offset + length].cast("B")
    return _buffer._same_dtype_store(raw, a.dtype)

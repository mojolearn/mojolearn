# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed Arrays of test values: a TESTING TOOL, never product execution.

The product's Python conversion layer (`_buffer.as_f32_c` and friends) was
deleted on 2026-11-15 (lane py-tail-2): every public call admits its inputs
natively (`bindings/admitted_family_door.mojo`). Fixtures and the legacy
oracle copies under tests/ still need a typed, laid-out Array of a test
value; these build it from the native Array operations (`from_list`,
`from_buffer`, `astype`, `reshape`, `order` / `flags`), with NumPy (the test
extra) for the inputs no Array constructor borrows (strided views, odd widths,
object or `__array__`-only exporters, sparse matrices) and for a column-major
copy. Only the public Array API and `_oracle_support` are used, so an internal
rename does not break the oracles that import this module.

Each converter returns `(Array, copied)`, the shape the deleted helpers had,
so the oracle copies read unchanged.
"""
import array
import ctypes
import sys
from functools import lru_cache

from mojolearn import Array
from mojolearn.tests._oracle_support import CODE as _CODE, require_numpy


def _numpy():
    return sys.modules.get("numpy")


def materialize(obj, name="X"):
    """`obj` as an Array of its own dtype (lists: float64 when any leaf is a
    float, else int64), and whether that copied."""
    if isinstance(obj, Array):
        return obj, False
    if isinstance(obj, range):
        # the deleted `_buffer` materialized a range as the sequence it is (5c1d770282^ _buffer.py:1065)
        obj = list(obj)
    if isinstance(obj, (list, tuple)):
        try:
            return Array.from_list(obj, "<i8"), True
        except (TypeError, ValueError, OverflowError):
            return Array.from_list(obj, "<f8"), True
    np = _numpy()
    if np is not None:
        if hasattr(obj, "toarray") and hasattr(obj, "nnz"):
            raise TypeError(f"mojolearn: Sparse data was passed for {name}, but dense data is required. "
                            "Use '.toarray()' to convert to a dense numpy array.")
        if not isinstance(obj, np.ndarray) and callable(getattr(obj, "__array__", None)):
            obj = np.asarray(obj)
        if isinstance(obj, np.ndarray):
            copied = False
            # The deleted `_buffer` refusals (5c1d770282^ _buffer.py:478, :554), which the product keeps
            # (bindings/host_iforest_fit_native_door.mojo:216,222).
            if obj.dtype.kind in "US":
                raise TypeError(f"mojolearn: {name} has buffer format {memoryview(obj).format!r}, "
                                "which is not a numeric array")
            if obj.dtype == object:
                if any(isinstance(v, str) for v in obj.ravel().tolist()):
                    raise TypeError(f"mojolearn: {name} holds a str, not a number")
                obj = np.asarray(obj.tolist(), dtype=np.float64)
                copied = True
            if obj.dtype.kind == "u" and obj.dtype.itemsize == 8 and obj.size and int(obj.max()) > 2 ** 63 - 1:
                # the product refuses uint64 storage past int64 (core/host_sparse_impl.mojo host_sparse_uint64_bound)
                raise OverflowError("int too big to convert")
            if obj.dtype.str not in _CODE:
                natural = {"|b1": "<u1", "|i1": "<i4", "<i2": "<i4", "<u2": "<u4", "<u8": "<i8"}
                obj = obj.astype(natural.get(obj.dtype.newbyteorder("<").str, obj.dtype.newbyteorder("<").str))
                copied = True
            if not (obj.flags["C_CONTIGUOUS"] or obj.flags["F_CONTIGUOUS"]):
                obj = np.ascontiguousarray(obj)
                copied = True
            return Array.from_buffer(obj), copied
    try:
        return Array.from_buffer(obj), False
    except (TypeError, ValueError):
        view = memoryview(obj)
        return Array.from_list(view.tolist(), "<f8" if view.format in ("f", "d", "e") else "<i8"), True


def as_typed(obj, dtype, order="C", ndim=None, name="X"):
    a, copied = materialize(obj, name)
    if ndim is not None and a.ndim != ndim:
        # the product's rank refusal (bindings/admitted_family_door.mojo): a 2-D request below rank 2 adds
        # scikit-learn's reshape hint
        hint = (". Reshape your data either using array.reshape(-1, 1) if your data has a single feature or "
                "array.reshape(1, -1) if it contains a single sample.") if ndim == 2 and a.ndim < 2 else ""
        raise ValueError(f"mojolearn: {name} must be {ndim}-D, got {a.ndim}-D shape {a.shape}{hint}")
    if a.size == 0:
        what = "feature(s)" if a.ndim == 2 and a.shape[0] != 0 else "sample(s)"
        raise ValueError(f"mojolearn: {name} is empty, shape {a.shape}: Found array with 0 {what} "
                         f"(shape={a.shape}) while a minimum of 1 is required.")
    if a.dtype != dtype:
        a = a.astype(dtype)
        copied = True
    laid, flipped = _in_order(a, order)
    return laid, copied or flipped


def _in_order(a, order):
    """`a` laid out as `order` ("C" or "F") and whether that copied: `a` itself when its label is `order` or both
    layouts coincide (at most one dimension above 1, `flags` reports both), else a copy with the same values in
    `order`: the layout and values of the former private `Array._as_order`, which relabeled a coinciding layout
    (same bytes) where this returns it unchanged."""
    if a.order == order:
        return a, False
    flags = a.flags
    if flags["C_CONTIGUOUS"] and flags["F_CONTIGUOUS"]:
        return a, False
    if order == "C":
        return a.reshape(a.shape), True  # public reshape copies an F-order Array to C order
    np = require_numpy("the oracle column-major layout")
    return Array.from_buffer(np.asfortranarray(np.asarray(a))), True


def as_f32_c(obj, *, ndim=2, name="X"):
    return as_typed(obj, "<f4", "C", ndim, name)


def as_f32_dense_c(obj, *, ndim=2, name="X"):
    if hasattr(obj, "toarray") and hasattr(obj, "nnz") and hasattr(obj, "tocsr"):
        obj = obj.toarray()
    return as_f32_c(obj, ndim=ndim, name=name)


def f32_dense(X, name="X"):
    """The deleted `_expansion_neighbors._f32` (ec78e70324): a dense 2-D float32 Array."""
    return as_f32_dense_c(X, ndim=2, name=name)[0]


def f32_1d(x, name):
    """The deleted `_expansion_neighbors._f32_1d` (ec78e70324)."""
    return as_f32_c(x, ndim=1, name=name)[0]


def i32_1d(x, name):
    """The deleted `_expansion_neighbors._i32` (ec78e70324)."""
    return as_i32_c(x, ndim=1, name=name)[0]


def as_f32_colmajor(obj, *, name="X"):
    return as_typed(obj, "<f4", "F", 2, name)


def as_i32_c(obj, *, ndim=1, name="X"):
    return as_typed(obj, "<i4", "C", ndim, name)


def as_i64_c(obj, *, ndim=1, name="X"):
    return as_typed(obj, "<i8", "C", ndim, name)


def as_f64_c(obj, *, ndim=1, name="X"):
    return as_typed(obj, "<f8", "C", ndim, name)


def has_buffer(obj):
    """`obj` exports the buffer protocol."""
    try:
        memoryview(obj).release()
        return True
    except TypeError:
        return False


# The raw stores the deleted `_buffer._output_store` / `_raw_store_type` /
# `_same_dtype_store` built, for the oracle copies (`_array_reference`,
# `_prep_prog_oracle`, `_legacy_host_inference`) that still lay Arrays over
# them. Uninitialized storage: the caller fills every byte.
_RAW_MALLOC = ctypes.pythonapi.PyMem_RawMalloc
_RAW_MALLOC.argtypes = [ctypes.c_size_t]
_RAW_MALLOC.restype = ctypes.c_void_p
_RAW_FREE = ctypes.pythonapi.PyMem_RawFree
_RAW_FREE.argtypes = [ctypes.c_void_p]
_RAW_FREE.restype = None


class _RawAllocation:
    __slots__ = ("addr", "_free", "__weakref__")

    def __init__(self, nbytes):
        self.addr = None
        self._free = _RAW_FREE
        self.addr = _RAW_MALLOC(max(1, nbytes))
        if not self.addr:
            raise MemoryError("mojolearn tests: cannot allocate a raw store")

    def __del__(self):
        if self.addr:
            self._free(self.addr)
            self.addr = None


@lru_cache(maxsize=128)
def raw_store_type(code, n):
    scalar = {"f": ctypes.c_float, "d": ctypes.c_double,
              "i": ctypes.c_int32, "q": ctypes.c_int64,
              "I": ctypes.c_uint32, "H": ctypes.c_uint16,
              "B": ctypes.c_uint8, "b": ctypes.c_int8}[code]

    class Store(ctypes.Array):
        _type_ = scalar
        _length_ = n

        def buffer_info(self):
            return ctypes.addressof(self), n

    return Store


def output_store(code, n):
    if n < 0:
        raise ValueError("mojolearn tests: negative store size")
    store_type = raw_store_type(code, n)
    owner = _RawAllocation(ctypes.sizeof(store_type))
    store = store_type.from_address(owner.addr)
    store._allocation = owner
    return store


def same_dtype_store(raw, dtype):
    """An `array.array` of the elements `raw` (bytes of `dtype`) holds, equal
    to `array.array(code, memoryview(raw).cast(code))` (the float32 object
    round trip quiets a signaling NaN, as the core helper `cast_elements`
    float32 -> float32 did in place; here the round trip itself, stdlib only)."""
    store = array.array(_CODE[dtype])
    store.frombytes(raw)
    if dtype != "<f4" or not len(store):
        return store
    return array.array("f", store.tolist())

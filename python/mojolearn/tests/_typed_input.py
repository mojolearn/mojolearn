# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed Arrays of test values: a TESTING TOOL, never product execution.

The product's Python conversion layer (`_buffer.as_f32_c` and friends) was
deleted on 2026-11-15 (lane py-tail-2): every public call admits its inputs
natively (`bindings/admitted_family_door.mojo`). Fixtures and the legacy
oracle copies under tests/ still need a typed, laid-out Array of a test
value; these build it from the native Array operations (`from_list`,
`from_buffer`, `astype`, `_as_order`), with NumPy (the test extra) for the
inputs no Array constructor borrows (strided views, odd widths, object or
`__array__`-only exporters, sparse matrices).

Each converter returns `(Array, copied)`, the shape the deleted helpers had,
so the oracle copies read unchanged.
"""
import array
import ctypes
import sys
from functools import lru_cache

from mojolearn._array import Array, _CODE


def _numpy():
    return sys.modules.get("numpy")


def materialize(obj, name="X"):
    """`obj` as an Array of its own dtype (lists: float64 when any leaf is a
    float, else int64), and whether that copied."""
    if isinstance(obj, Array):
        return obj, False
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
            if obj.dtype == object:
                obj = np.asarray(obj.tolist(), dtype=np.float64)
                copied = True
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
        raise ValueError(f"mojolearn: {name} must be {ndim}-D, got {a.ndim}-D shape {a.shape}")
    if a.size == 0:
        raise ValueError(f"mojolearn: {name} is empty, shape {a.shape}")
    if a.dtype != dtype:
        a = a.astype(dtype)
        copied = True
    if not a._has_order(order):
        copied = True
    return a._as_order(order), copied


def as_f32_c(obj, *, ndim=2, name="X"):
    return as_typed(obj, "<f4", "C", ndim, name)


def as_f32_dense_c(obj, *, ndim=2, name="X"):
    if hasattr(obj, "toarray") and hasattr(obj, "nnz") and hasattr(obj, "tocsr"):
        obj = obj.toarray()
    return as_f32_c(obj, ndim=ndim, name=name)


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
    round trip quiets a signaling NaN; the core helper `cast_elements` does
    the same in place), or None without that helper."""
    from mojolearn._buffer import _native_optional
    store = array.array(_CODE[dtype])
    store.frombytes(raw)
    if dtype != "<f4" or not len(store):
        return store
    fn = _native_optional("cast_elements")
    if fn is None:
        return None
    at = store.buffer_info()[0]
    fn(at, 0, at, 0, len(store))
    return store

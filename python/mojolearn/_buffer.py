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

CONVERSIONS ARE NATIVE. Casting, layout changes and the admission of lists,
sparse and device inputs are done by the compiled doors
(`bindings/admitted_family_door.mojo`); this module only takes addresses,
allocates and scans.
"""

from __future__ import annotations

import array
import ctypes

from ._array import Array, _ARRAY_CONTEXT, _ARRAY_NATIVE, _CODE

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
# PRIVATE function-pointer objects, on purpose (2026-09-11, H100 istella leg).
# `ctypes.pythonapi` is a process-wide singleton and `_api.PyObject_GetBuffer`
# (attribute access) is CACHED on it, so every library that spells it that
# way shares ONE `_FuncPtr` and the last `argtypes` assignment wins for all
# of them. `treelite.model` (imported by cuML) assigns its own `_PyBuffer`
# class there; after `import cuml` our calls failed with "argument 2:
# expected LP__PyBuffer instance instead of pointer to _PyBuffer" and every
# RF fit in a process that also held cuML was refused. `PyDLL.__getitem__`
# builds a fresh, uncached `_FuncPtr` each time, so nobody else's argtypes
# can reach these three.
_get_buffer = _api["PyObject_GetBuffer"]
_get_buffer.argtypes = [ctypes.py_object, ctypes.POINTER(_PyBuffer), ctypes.c_int]
_get_buffer.restype = ctypes.c_int
_release_buffer = _api["PyBuffer_Release"]
_release_buffer.argtypes = [ctypes.POINTER(_PyBuffer)]
_release_buffer.restype = None
_memoryview_from_memory = _api["PyMemoryView_FromMemory"]
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
        # DEVIATION 2692, defence in depth: the chokepoint for `view`, `addr`
        # and `addr_ro`, so a caller cannot get "does not support the buffer
        # protocol" for memory whose real problem is that it lives on a GPU.
        # ONE definition of the test.
        device_kind = _device_array_kind(obj)
        if device_kind is not None:
            raise TypeError(
                "mojolearn: %s is a DEVICE array (it exports %s). mojolearn "
                "reads HOST memory only. Copy it to the host first: cupy "
                "`x.get()`, torch `x.cpu()`, numba `x.copy_to_host()`."
                % (name, device_kind)
            )
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
            for k in range(view.ndim):  # glue: one check per buffer axis
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
                self.shape = tuple(int(view.shape[k]) for k in range(self.ndim))  # glue: one entry per buffer axis
            else:
                # ndim 1 without a shape array: a simple byte-like buffer
                self.shape = (self.nbytes // max(self.itemsize, 1),)
            if view.strides:
                self.strides = tuple(int(view.strides[k]) for k in range(self.ndim))  # glue: one entry per buffer axis
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
    for s in shape:  # glue: product of shape dimensions
        n *= s
    if n <= 1:
        return True, True
    if strides is None:
        return True, False
    c_ok = True
    expect = itemsize
    for k in range(len(shape) - 1, -1, -1):  # glue: one stride check per axis
        if shape[k] != 1 and strides[k] != expect:
            c_ok = False
            break
        expect *= shape[k]
    f_ok = True
    expect = itemsize
    for k in range(len(shape)):  # glue: one stride check per axis
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
        # lane/neural-net-experiment (2026-09-30): a signed one-byte buffer
        # is the Array's own int8 dtype ("<i1", `array.array('b')`), so a
        # contiguous int8 exporter (NumPy int8 codes) is a zero-copy view.
        # It was spelled "|i1", which no Array dtype matched, so the former
        # Python converter widened it to int32 one Python object per element
        # and cast it back: 1.25 s per 4096 x 4096 code array on
        # the L40S, the whole of the board's gemm-int8 cell (2.46 s against a
        # 6 ms device side).
        return ("<i1" if size == 1 else f"<i{size}")
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
    if fmt[:1] == "Z":
        # scikit-learn's wording and type (`check_array`): complex input is
        # a value error, not an unsupported container
        raise ValueError(f"Complex data not supported: buffer format {fmt!r} is complex")
    raise TypeError(f"mojolearn: unsupported buffer format {fmt!r}")


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


# TOMBSTONE: _buffer.flat_bytes (no product caller; tests read Array.tobytes) deleted 2026-11-14 by lane
# py-admission; code recoverable at 1fac46234f.


# ------------------------------------------------------------ conversion


_DLPACK_CPU_DEVICES = (1, 3, 10)  # kDLCPU, kDLCUDAHost, kDLCUDAManaged


def _device_array_kind(obj):
    """The protocol by which `obj` announces it lives on a device, or None.

    DEVIATION 2692. Detection only: no pointer is read and no device is
    touched. A CPU tensor that exports DLPack is NOT a device array, so the
    DLPack arm asks `__dlpack_device__` for the device type and accepts only
    a non-CPU one -- otherwise a host torch tensor, which also exports the
    buffer protocol, would lose its existing zero-copy path.
    """
    if hasattr(obj, "__cuda_array_interface__"):
        return "__cuda_array_interface__"
    if hasattr(obj, "__dlpack_device__"):
        try:
            device_type = tuple(obj.__dlpack_device__())[0]
        except Exception:
            return None
        if device_type not in _DLPACK_CPU_DEVICES:
            return "__dlpack__ (device type %r)" % (device_type,)
    return None


# TOMBSTONE: _buffer._materialize, _materialize_buffer, materialize_f32_lists, _same_dtype_store, _NATURAL,
# _object_buffer, _has_buffer, _convert, _native_to_f32, _RawAllocation, _raw_store_type, _output_store, _as_typed,
# as_f32_c, as_f32_dense_c, as_f32_colmajor, as_i32_c, as_i64_c, as_u16_c, as_i8_c, as_f64_c and _iter_format (the
# Python buffer conversion layer; its last product caller, RegressorScoreMixin.score, became the native
# family_mcls_regressor_score; every door admits natively through bindings/admitted_family_door.mojo) deleted
# 2026-11-15 by lane py-tail-2; code recoverable at 1a9d8c01bd.


# TOMBSTONE: _buffer.as_f32_forest_layout (no caller; bindings/trees_public_inputs_door.mojo owns the
# forest layout) deleted 2026-11-14 by lane py-admission; code recoverable at 1fac46234f.


# ------------------------------------------------------------ constructors


def empty(shape, dtype):
    """A new owned C-order Array. The memory is zero-filled: there is no
    uninitialized allocation in `array.array`, and a defined value costs
    nothing that matters next to what the kernels do."""
    return Array(shape, dtype)


def zeros(shape, dtype):
    return Array(shape, dtype)


def full(shape, value, dtype):
    """A new owned C-order Array of `shape` holding `value` (native)."""
    return _ARRAY_NATIVE("full", Array, _ARRAY_CONTEXT, shape, value, dtype)


def frombytes(raw, dtype, shape):
    """An owned C-order Array holding a COPY of `raw` (native)."""
    return _ARRAY_NATIVE("frombytes", Array, _ARRAY_CONTEXT, raw, dtype, shape)


# --------------------------------------------------------------- scanning


# TOMBSTONE: _buffer.shape_of (no product caller; the doors read shapes natively, the oracle copy is
# tests/_legacy_neighbors_knn.py) deleted 2026-11-15 by lane py-buffer; code recoverable at 8579360a2e.


def all_finite(arr):
    """True when every element of a float32/float64 Array is finite.

    The base binding's `all_finite_f32` / `all_finite_f64` (DEVIATION
    2303's helpers). In the GPU base binding the scan runs on the device
    and one status word comes back (lane cpu2-l1-input,
    `core/input_device.mojo`); the host loop is the host column (CPU-only
    installs) and the fallback. The `math.fsum` fallback of DEVIATION 2307 was
    removed 2026-09-10 with every other Python copy of a native helper.
    """
    if not isinstance(arr, Array) or arr.dtype not in ("<f4", "<f8"):
        raise TypeError(
            "mojolearn: all_finite takes a float32 or float64 Array, got "
            f"{getattr(arr, 'dtype', type(arr).__name__)!r}"
        )
    if arr.size == 0:
        return True
    fn = _native("all_finite_f32" if arr.dtype == "<f4" else "all_finite_f64")
    return int(fn(arr._addr, arr.size)) == 1


_NATIVE = {}
_NATIVE_MISSING = set()

# TOMBSTONE: _buffer.hotpath_enabled (always True, no caller) deleted 2026-11-15 by lane py-buffer;
# code recoverable at 8579360a2e.


def _native_optional(key):
    """`_native(key)`, or None when this install's binary does not carry
    `key`.

    FOR THE lane/python-hotpath HELPERS ONLY, and the reason it may return
    None where `_native` raises: each of those helpers stands in for a
    Python routine that STAYS in the package as its definition, so an
    install whose binary predates the helper (a GPU base binding not yet
    rebuilt, a core host binding from before 2026-09-17) computes the same
    answer more slowly instead of refusing to run. The miss is remembered,
    so a missing symbol costs one lookup per process."""
    fn = _NATIVE.get(key)
    if fn is not None:
        return fn
    if key in _NATIVE_MISSING:
        return None
    try:
        return _native(key)
    except (ImportError, AttributeError):
        _NATIVE_MISSING.add(key)
        return None


def _native(key):
    """The base binding's host helper `key`, resolved once per process
    from the IDENTICAL `_mojolearn` binary and cached in `_NATIVE` (a test
    may plant a callable there to observe a call).

    ALWAYS THE IDENTICAL BINARY, WHATEVER TIER THE CALLER RUNS. These
    helpers are a float64-to-float32 cast, a transpose and a finiteness
    predicate: byte copies with no tier semantics (on the device in the GPU
    base binding since lane cpu2-l1-input: bit moves and the soft-f64
    narrowing, the same bytes on every vendor), and every estimator in
    every tier funnels its input through them. Since DEVIATION 2490
    (2026-09-10) the base binding is built in the identical tier alone
    (only the tree lanes have a fast binary), so
    resolving through the running tier here would refuse every FAST tree
    fit at its first input conversion. Found by the macOS release smoke
    under MOJOLEARN_NUMERIC_MODE=fast the day the rule landed.

    There is NO Python fallback. The package refuses to import without its
    binding, so the only way this symbol can be missing is a binary on
    disk older than the Python beside it, and running a slower copy of the
    same arithmetic would hide that. It raises by name with the rebuild
    command instead (fallback arms removed 2026-09-10)."""
    fn = _NATIVE.get(key)
    if fn is not None:
        return fn
    from . import _backend
    try:
        fn = getattr(_backend.binding("_mojolearn", mode="identical"), key)
    except Exception as exc:
        fn = _host_native(key)
        if fn is None:
            if _backend._CPU_ONLY is not None and isinstance(exc, ImportError):
                # A CPU-ONLY INSTALL (the CPU training lane, 2026-09-13). The
                # base binding is not old, there is none, and the stub has
                # already said so BY NAME ("no CPU implementation of
                # _mojolearn.<key> yet"). That sentence is the one an
                # identity_break cell must record, so it is re-raised as is.
                raise
            raise ImportError(
                f"mojolearn: the base binding has no `{key}`; the compiled "
                "_mojolearn extension is older than this Python layer, or the "
                "identical set is not built. Rebuild it with\n    "
                "MOJOLEARN_NUMERIC_MODE=identical sh bindings/build.sh"
            ) from exc
    _NATIVE[key] = fn
    return fn


#: The two door binaries `_host_native` resolves helpers from (loaded by
#: `_backend.load_door_host`, the same loader the doors use).
_BYTE_LM_HOST_EXTENSION = "_mojolearn_byte_lm_host"
_FOREST_HOST_EXTENSION = "_mojolearn_forest_host"
#: DEVIATION 2614: the helpers the CPU inference binding also exports, and the
#: only ones `_native` may resolve from it.
_HOST_NATIVE_KEYS = frozenset({"all_finite_f32", "all_finite_f64", "cast_f64_to_f32"})
#: The forest host binding (the forest host lane, 2026-09-13,
#: `bindings/host_helpers_pack.mojo`) exports those three and the four
#: `_labels.argmax_rows` and `_labels.decode_labels` resolve, so a saved
#: forest's class labels decode on a CPU-only install through the same native
#: helper the GPU path uses.
_FOREST_HOST_NATIVE_KEYS = _HOST_NATIVE_KEYS | frozenset(
    {"argmax_rows_f32", "argmax_rows_f64", "gather_i64", "gather_f64"})


def _host_native(key):
    """`key` from a CPU inference binding, on a box whose GPU base binding
    is not built (DEVIATION 2614). The same native helper compiled into a
    second binary, not a Python copy, so the no-Python-fallback rule above
    holds. The byte LM host binding is asked first for the three it carries,
    then the forest host binding for its seven. None when `key` is not one of
    them or neither binary is built; a refused sabotage build raises rather
    than hiding."""
    # THE LOADERS, NOT THE DOORS (lane/lazy-imports, 2026-10-10). Importing
    # `_byte_lm_host` / `_forest_host` here for their `_load()` put the byte LM
    # door (lowbit, linalg, the decomp and metrics expansions) and the forest
    # door (`_classical_host` and every classical door) in every lane's import
    # closure, since every lane loads this file: 251 files selected every lane
    # in tools/test_lane_select.py. `_backend` holds the loaders now.
    if key in _HOST_NATIVE_KEYS:
        from . import _backend
        try:
            return getattr(_backend.load_door_host(_BYTE_LM_HOST_EXTENSION), key)
        except (ImportError, AttributeError):
            pass
    if key in _FOREST_HOST_NATIVE_KEYS:
        from . import _backend
        try:
            return getattr(_backend.load_door_host(_FOREST_HOST_EXTENSION), key)
        except (ImportError, AttributeError):
            pass
    return None


#: The base helper table's vendor words (core/base_entry_table.mojo
#: `base_entry_vendor`), the mapping the binaries apply at publication.
_BASE_NATIVE_VENDORS = {"cpu": 0, "metal": 1, "cuda": 2, "hip": 3}


def base_native_key():
    """`(mode, vendor)`: the base helper table key of the binary `_native`
    resolves (lane calltable-base-helpers), which the Mojo doors read
    instead of calling the helpers through Python. Read once at import by
    the bridges that hand the helpers to a door; raises like `_native` when
    the binary has no probes."""
    mode = int(_native("mojolearn_numeric_mode")())  # import-time: the binary's compiled tier
    vendor = _BASE_NATIVE_VENDORS.get(str(_native("mojolearn_vendor")()), -1)  # import-time: its vendor
    if mode not in (0, 1) or vendor < 0:
        raise ImportError("mojolearn: the base binding publishes no native helper table "
                          f"(mode {mode!r}, vendor {vendor!r})")
    return (mode, vendor)


def as_index_i64(values, *, name="indices"):
    """Row indices (an Array, any buffer or a list of ints) as a 1-D int64
    Array: the native cast of a contiguous numeric buffer; a list goes in
    through `array('q', ...)`, in C. No per-index Python arithmetic."""
    if not isinstance(values, Array):
        if isinstance(values, (list, tuple, range)):
            store = array.array("q", values)
            return Array._owned(store, (len(store),), "<i8", "C")
        # Any other contiguous buffer is borrowed zero-copy (native `from_buffer`);
        # the int64 cast below is the native `astype`.
        values = Array.from_buffer(values)
    values = values.reshape((values.size,))
    return values if values.dtype == "<i8" else values.astype("<i8")

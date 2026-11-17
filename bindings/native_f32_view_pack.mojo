# SPDX-License-Identifier: Apache-2.0
# PACK helper: an admitted C-order float32 buffer as a zero-copy mojolearn
# Array view (fb-boost-copy, 2026-11-17). The doors that hand a caller's
# float32 words to a member fit, a predict or a program runner used to copy
# them into a new zero-filled Array on every call (a serial fill of the whole
# buffer and its page faults, then a copy: ~1.5 GB/s, per element of the
# input, so every shape pays it). The view reads the same words in place, as
# 0.8.55's `as_f32_c` did (a borrowed buffer was wrapped, never copied).
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_setattr
from bindings.native_buffer_owner_pack import native_buffer_view, native_storage_allocate


def native_f32_view_array(owner: PythonObject, address: Int, shape: PythonObject, rows: Int, cols: Int,
                          array_class: PythonObject) raises -> PythonObject:
    """An Array of `array_class` reading the C-contiguous float32 words of
    `owner` (any object exporting a buffer that starts at `address`: an
    ndarray, an `array.array`, a cast's fresh storage) as `shape`, rank 1
    (`cols` 1) or 2, with no copy. The Array pins the export and holds
    `owner` for as long as it lives; a read-only export stays read-only
    (`_readonly`). Callers only read the words; a caller that writes keeps
    its own copy."""
    var rank = len(shape)
    var count = rows * cols
    var pin = native_buffer_view(owner)
    var array = native_storage_allocate(array_class)
    py_setattr(array, "_store", owner)
    py_setattr(array, "_base", PythonObject(None))
    py_setattr(array, "_pin", pin)
    py_setattr(array, "_mv", py_call(py_call(pin.cast, "B").cast, "f"))
    py_setattr(array, "_addr", PythonObject(address))
    py_setattr(array, "_readonly", pin.readonly)
    py_setattr(array, "shape", shape)
    py_setattr(array, "dtype", PythonObject("<f4"))
    py_setattr(array, "ndim", PythonObject(rank))
    py_setattr(array, "size", PythonObject(count))
    py_setattr(array, "nbytes", PythonObject(count * 4))
    py_setattr(array, "itemsize", PythonObject(4))
    py_setattr(array, "strides", Python.tuple(cols * 4, 4) if rank == 2 else Python.tuple(4))
    py_setattr(array, "order", PythonObject("C"))
    return array

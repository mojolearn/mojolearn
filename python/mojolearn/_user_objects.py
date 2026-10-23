# SPDX-License-Identifier: Apache-2.0
"""The door of the user-object reader (bindings/user_objects_native.mojo).

A Python object that cannot cross into Mojo as a buffer (a list of label
objects, a ragged nested list, a user's index pairs or group ids) is handed
to ONE native call, which reads it once and returns typed stores or the
objects placed where they go. This module only calls it and raises a
refusal it returns, with the exception type and words the native read
chose."""
# Engineering note: lane purge-user-objects (2026-10-22). The reader is an
# entry of the tokenizer host binding (`user_objects_native_call`), the
# binding every install loads for its host shells.
from __future__ import annotations

_CALL = []
_REFUSALS = {"TypeError": TypeError, "ValueError": ValueError, "IndexError": IndexError,
             "OverflowError": OverflowError}

# ("kinds", values) counters, bindings/user_objects_native.mojo KIND_*
KIND_STR, KIND_BOOL, KIND_EXACT_INT, KIND_INTEGRAL, KIND_REAL = 0, 1, 2, 3, 4
KIND_INTVAL, KIND_SEQ, KIND_INT_OR_BOOL, KIND_PLAIN_INT, KIND_REAL_NONBOOL = 5, 6, 7, 8, 9


def call(operation, *args):
    """One user-object read: ``operation`` and its arguments as the native
    entry takes them. Raises the refusal the read returns."""
    if not _CALL:
        from .tokenizer import _binding
        _CALL.append(_binding()._module.user_objects_native_call)
    got = _CALL[0]((operation,) + args)
    if type(got) is tuple and len(got) == 3 and got[0] == "__refuse__":
        raise _REFUSALS[got[1]](got[2])
    return got


def kinds(values):
    """The kind counts of a list of objects (KIND_* above)."""
    return call("kinds", values)


def take(values, indices):
    """``[values[i] for i in indices]`` for a Python list or tuple and an
    int32 or int64 index Array."""
    import array
    from ._array import Array
    if isinstance(indices, array.array) and indices.typecode in ("q", "i") and indices.itemsize in (4, 8):
        return call("take", values, indices.buffer_info()[0] if len(indices) else 0, len(indices),
                    indices.itemsize)
    if not isinstance(indices, Array) or indices.dtype not in ("<i4", "<i8"):
        from ._buffer import as_index_i64
        indices = as_index_i64(indices, name="indices")
    idx = indices._as_c()
    width = 4 if idx.dtype == "<i4" else 8
    return call("take", values, idx._addr if idx.size else 0, int(idx.size), width)


def scatter(rows, indices, values):
    """``rows[i] = v`` for each index of ``indices`` (an int64 Array) and
    value of ``values``, in order, up to the shorter of the two."""
    from ._buffer import as_index_i64
    idx = as_index_i64(indices, name="indices")._as_c()
    call("scatter", rows, idx._addr if idx.size else 0, int(idx.size), values)

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Class labels without NumPy: the `classes_` ORDER RULE, the code map, the
argmax over class columns, and the flat host view every validator reads.

DEVIATION 2340 -- `classes_` IS A PYTHON LIST HOLDING THE CALLER'S LABEL
OBJECTS, and its order is defined HERE, once, for every classifier in the
package that maps labels onto dense codes (`randomforest.py`,
`extratrees.py`). `np.unique(y, return_inverse=True)` used to define it by
way of NumPy's dtype promotion and sort; this is the rule that replaces it,
stated so a caller can predict the order from the labels alone:

  1. Every label must be a NUMBER (`numbers.Real`, which is every Python
     and NumPy int and float, plus bool) or a `str`. Any other object
     (bytes, None, tuples) is refused with a ValueError. `np.unique` would
     have cast bytes to a `|S` array; there is no such array here.
  2. All labels must be of ONE of those two kinds. A `y` that mixes
     strings and numbers is refused. `np.unique` would have promoted every
     number to its string spelling and sorted the spellings; that order
     was never what anyone meant.
  3. Numbers are grouped by NUMERIC EQUALITY and sorted by VALUE, exactly
     as Python compares them (int against float is exact, so a label of
     2**60 + 1 does not collide with 2**60). Therefore `1`, `1.0` and
     `True` are ONE class, and `-0.0` and `0.0` are ONE class: the
     representative kept in `classes_` is the FIRST such object seen in
     `y`. `np.unique` collapsed the same pairs after promoting to one
     dtype, but reported the promoted value (`1.0`) rather than the
     caller's object (`1`); this keeps the caller's.
  4. NaN is REFUSED with a ValueError. `np.unique` sorted NaN last and,
     since NaN != NaN, could report one NaN class per NaN row on older
     versions and one on newer ones; a class nobody can name is not a
     class.
  5. Strings sort by Python's `str` ordering (code point order), which is
     the order `np.unique` gave a `<U` array.

The code map is a dict keyed by the label object, so encoding `y` is one
dict lookup per row: O(rows) Python, the label-encoding loop the contract
explicitly permits (NUMPY_FREE_CONTRACT.md, Rules). Decoding a prediction
is the same loop in the other direction.

`flat_view` is the ONE place a host-side scan takes a memoryview over an
`Array`'s STORAGE (zero-copy, storage order, through the address the
binding itself would read). It is here rather than in `_buffer` because it
exists for the label routines and the input validators that share this
module, and because it depends only on the contract's `addr_ro`.
"""

import ctypes
import math
import numbers

from ._array import Array
from ._buffer import addr_ro

_CTYPES = {
    "f": ctypes.c_float,
    "d": ctypes.c_double,
    "i": ctypes.c_int32,
    "I": ctypes.c_uint32,
    "q": ctypes.c_int64,
    "B": ctypes.c_uint8,
}

_FORMAT_OF_DTYPE = {
    "<f4": "f", "<f8": "d", "<i4": "i", "<u4": "I", "<i8": "q", "<u1": "B",
}


def flat_view(arr, fmt=None):
    """A writable 1-D memoryview over `arr`'s STORAGE, in storage order.

    For a C-order `Array` that is row-major; for the F-order `Array`
    `as_f32_colmajor` returns it is column-major, so column `f` of an
    `[n_rows, n_cols]` matrix is the slice `[f * n_rows:(f + 1) * n_rows]`.
    `fmt` is the struct format ('f', 'd', 'i', 'I', 'q', 'B'); omitted,
    it follows `arr.dtype`. The caller MUST keep `arr` alive while the
    view is used: nothing here owns the memory.
    """
    if fmt is None:
        fmt = _FORMAT_OF_DTYPE[arr.dtype]
    n = int(arr.size)
    buf = (_CTYPES[fmt] * n).from_address(addr_ro(arr, name="host view"))
    return memoryview(buf).cast("B").cast(fmt)


def is_bool(value):
    """Python's `bool` or NumPy's `bool_`, without importing NumPy."""
    return isinstance(value, bool) or type(value).__name__ == "bool_"


def flatten_labels(y):
    """`y` as a flat Python list of label objects. Anything with
    `tolist()` (an `Array`, an ndarray, an `array.array`) is unpacked
    through it; nested lists and tuples are flattened, which is what
    `np.asarray(y).ravel()` did for a column vector."""
    if hasattr(y, "tolist") and not isinstance(y, (list, tuple)):
        y = y.tolist()
    if isinstance(y, (str, bytes)):
        raise ValueError("mojolearn: y must be a sequence of labels")
    out = []
    stack = [y]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(reversed(item))
        else:
            out.append(item)
    return out


def sorted_classes(labels):
    """`(classes, codes)`: the class list under the ORDER RULE above, and
    one dense code per label, `classes[codes[i]] == labels[i]` under
    Python equality."""
    if not labels:
        raise ValueError("mojolearn: y is empty")
    first = {}
    numeric = strings = 0
    for v in labels:
        if isinstance(v, str):
            strings += 1
        elif isinstance(v, numbers.Real) or is_bool(v):
            numeric += 1
            if v != v:
                raise ValueError(
                    "mojolearn: y contains a NaN label; NaN is not a class"
                )
        else:
            raise ValueError(
                f"mojolearn: label {v!r} is neither a number nor a str"
            )
        if v not in first:
            first[v] = v
    if numeric and strings:
        raise ValueError(
            "mojolearn: y mixes numeric and str labels; encode one kind "
            "before fitting"
        )
    classes = sorted(first)
    code = {c: i for i, c in enumerate(classes)}
    return classes, [code[v] for v in labels]


def label_kind(classes):
    """'int' if every class is an integer (bools excluded), 'float' if
    every class is a real number, else 'object'."""
    if all(isinstance(c, numbers.Integral) and not is_bool(c) for c in classes):
        return "int"
    if all(isinstance(c, numbers.Real) and not is_bool(c) for c in classes):
        return "float"
    return "object"


def decode_labels(classes, codes):
    """Predicted codes back to labels. Integer classes give an int64
    `Array`, real classes a float64 `Array`; str or bool classes give a
    Python list of the label objects, since no `Array` dtype holds them
    (DEVIATION 2340: an ndarray of labels used to come back)."""
    values = [classes[int(c)] for c in codes]
    kind = label_kind(classes)
    try:
        if kind == "int":
            return Array.from_list([int(v) for v in values], "<i8")
        if kind == "float":
            return Array.from_list([float(v) for v in values], "<f8")
    except OverflowError:
        pass
    return values


def classes_member(classes):
    """`classes_` as the npz member `save` writes: int64 for integer
    labels, float64 for real labels, the list of str for str labels
    (`_serialize.write_npz` encodes a list of str as a `<U` member, the
    dtype `np.asarray` gave them). Bool labels are stored as int64 and
    come back as ints from `load` (DEVIATION 2340)."""
    kind = label_kind(classes)
    if kind == "int" or all(is_bool(c) for c in classes):
        return Array.from_list([int(c) for c in classes], "<i8")
    if kind == "float":
        return Array.from_list([float(c) for c in classes], "<f8")
    if all(isinstance(c, str) for c in classes):
        return list(classes)
    raise ValueError("mojolearn: classes_ holds labels no model file can carry")


def classes_from_member(member):
    """The inverse of `classes_member`, also reading a 0.6.x file's
    `classes` member (int, float, bool or `<U` from `np.unique`)."""
    if isinstance(member, str):
        return [member]
    if isinstance(member, (list, tuple)):
        return [str(v) for v in flatten_labels(list(member))]
    dtype = str(getattr(member, "dtype", ""))
    kind = dtype.lstrip("<>|=")[:1]
    values = flatten_labels(member)
    if kind in ("i", "u"):
        return [int(v) for v in values]
    if kind == "f":
        return [float(v) for v in values]
    if kind == "b":
        return [bool(v) for v in values]
    return [str(v) for v in values]


def argmax_rows(scores):
    """Row-wise first-max-wins argmax of an `[n_rows, n_cols]` `Array`
    of float32 or float64 scores, as an int64 `Array`. O(rows * classes)
    Python: the argmax over class counts the contract permits."""
    n_rows, n_cols = scores.shape
    view = flat_view(scores)
    out = []
    for r in range(n_rows):
        base = r * n_cols
        best = 0
        best_value = view[base]
        for c in range(1, n_cols):
            value = view[base + c]
            if value > best_value:
                best, best_value = c, value
        out.append(best)
    return Array.from_list(out, "<i8")


def finite_integer_codes(arr):
    """Sorted distinct values of a float32 label `Array` as ints, or None
    when any value is non-finite, negative or not an integer. The
    finiteness test is native (`_buffer.all_finite`); the distinct set is
    built by `set()` over the storage view, O(rows) at C speed with no
    Python loop until the loop over DISTINCT values."""
    from ._buffer import all_finite

    if not all_finite(arr):
        return None
    distinct = set(flat_view(arr, "f"))
    if not distinct or min(distinct) < 0:
        return None
    for v in distinct:
        if v != math.floor(v):
            return None
    return sorted(int(v) for v in distinct)

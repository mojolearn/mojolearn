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
from . import _portable_math as math

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
    `np.asarray(y).ravel()` did for a column vector.

    One native call (bindings/classical_labels_door.mojo `_flatten_labels`):
    exact containers are read by pointer from their C storage, depth first."""
    from ._native_labels_bridge import labels_entry, LABELS_OP_FLATTEN
    entry, context = labels_entry()
    return entry(LABELS_OP_FLATTEN, (y,), context)


# TOMBSTONE: _labels._plain_labels / _classes_and_codes / _sorted_plain_classes /
# _kind_of_type / _first_true / _check_label_kinds (the Python ORDER RULE walk)
# deleted 2026-11-09 by lane p2-labels-lists-arrays: `sorted_classes` is one
# native call (bindings/classical_labels_door.mojo `_sorted_classes`); the oracle
# copies live in tests/test_hotpath_native.py.


# TOMBSTONE: _labels.str_labels (its one product caller, unique_inverse, is native) deleted
# 2026-11-14 by lane py-admission; the oracle copy is tests/_former_prep_labels.py.


def sorted_classes(labels):
    """`(classes, codes)`: the class list under the ORDER RULE above, and
    one dense code per label, `classes[codes[i]] == labels[i]` under
    Python equality."""
    classes, codes = _sorted_native(labels)
    return classes, codes.tolist()


def _sorted_native(labels):
    """`sorted_classes` with the codes as the int32 `Array` the native ORDER
    RULE writes (one native call; exact label types only, by pointer)."""
    from ._native_labels_bridge import labels_entry, LABELS_OP_SORTED
    if type(labels) not in (list, tuple):
        labels = list(labels)
    entry, context = labels_entry()
    return entry(LABELS_OP_SORTED, (labels,), context)


# DEVIATION 2500 (2026-09-10): the ORDER RULE for ONE numeric buffer, in
# compiled code. Since lane migrate-shared-labels every numeric label vector
# (a buffer, or a flat list of exact ints or exact floats) is encoded by ONE
# native call, the base binding's `prepare_targets` (`bindings/_mojolearn.mojo`
# on a GPU install, the core host binding's twin on a CPU-only install): the
# capped encoder `encode_labels_<dtype>`, then `unique_inverse` past
# `_NATIVE_ENCODE_MAX_CLASSES` classes, the same helpers this file called one
# by one before. Only str and other Python label objects take the label input
# step `sorted_classes(flatten_labels(y))`; `tests/test_labels_native.py`
# holds the native answer equal to it.
#: The dtype codes `prepare_targets` encodes (`_array._NATIVE_CODE` 0..5).
_NATIVE_ENCODE = {"<f4": 0, "<f8": 1, "<i4": 2, "<i8": 3, "<u4": 4, "<u1": 5}
#: `bindings/_mojolearn.mojo::PT_ENCODE_CAP`: past this many classes the
#: entry answers through `unique_inverse`.
_NATIVE_ENCODE_MAX_CLASSES = 4096

_ALLOW_NEGATIVE = 1
_ALLOW_ALL_ZERO = 2


def prepare_targets(labels, weights, n, *, allow_negative=False, allow_all_zero=False):
    """The shared argument-and-label entry: ONE native call.

    `labels` (a C-contiguous vector `Array` of a `_NATIVE_ENCODE` dtype, or
    None) is encoded under the ORDER RULE; `weights` (a float32 or float64
    `Array` of `n` entries, or None) is checked: non-finite, negative
    (unless `allow_negative`) and all-zero (unless `allow_all_zero`)
    vectors are refused with scikit-learn's wording. Returns `(classes,
    codes)`: the sorted classes as a float64 (float labels) or int64
    (integer and bool labels) `Array`, one int32 code per row; `(None,
    None)` without labels."""
    # Lane py-admission: the stores, dtype codes, weight refusals and the cut
    # of the class table are bindings/labels_admission_unpack.mojo
    # `labels_prepare_targets_op`.
    from ._native_labels_bridge import labels_entry, LABELS_OP_PREPARE_TARGETS
    entry, context = labels_entry()
    flags = (_ALLOW_NEGATIVE if allow_negative else 0) | (_ALLOW_ALL_ZERO if allow_all_zero else 0)
    return entry(LABELS_OP_PREPARE_TARGETS, (labels, weights, int(n), flags), context)


def encode_labels(y):
    """`(classes, codes)` under the ORDER RULE, `codes` an int32 `Array`
    with one dense code per label."""
    # Lane ne1-h-labels (plan NATIVE_ENTRY_POINTS_AFTER_0866 H, N413): ONE
    # native call reads y (buffer, Array, list, tuple, Series, str ndarray,
    # or any nested label sequence) and encodes it
    # (bindings/classical_labels_door.mojo `classical_encode_labels`, the same
    # classes, codes and refusals as the former shell steps here:
    # `_encode_labels_native`, `flatten_labels`, the list and str encoders,
    # then `sorted_classes`). The shell no longer walks or copies y.
    from ._native_labels_bridge import labels_entry, LABELS_OP_ENCODE, LABELS_KIND_ORDER_RULE
    entry, context = labels_entry()
    return entry(LABELS_OP_ENCODE, (y, LABELS_KIND_ORDER_RULE), context)


def _encode_labels_native(y):
    """`(classes, codes)` for one numeric label vector through
    `prepare_targets`, or None for what only the label input step reads
    (lists, str or object buffers, a matrix, an empty vector). One native
    call (bindings/labels_admission_unpack.mojo `labels_encode_numeric_op`)."""
    from ._native_labels_bridge import labels_entry, LABELS_OP_ENCODE_NUMERIC
    entry, context = labels_entry()
    return entry(LABELS_OP_ENCODE_NUMERIC, (y,), context)


def decode_labels(classes, codes):
    """Predicted codes back to labels. Integer classes give an int64
    `Array`, real classes a float64 `Array`; str or bool classes give a
    Python list of the label objects, since no `Array` dtype holds them
    (DEVIATION 2340: an ndarray of labels used to come back)."""
    # ONE native call (bindings/classical_labels_door.mojo `labels_decode_labels`):
    # the class kind, the gathers and, for str / bool / object classes, one
    # list filled in C. Codes that are not a 1-D int32 / int64 Array are first
    # made one natively (bindings/labels_admission_unpack.mojo
    # `labels_decode_any_op`, lane py-admission).
    from ._native_labels_bridge import labels_entry, LABELS_OP_DECODE
    entry, context = labels_entry()
    return entry(LABELS_OP_DECODE, (classes, codes), context)


# TOMBSTONE: _labels.label_kind / classes_from_member / threshold_codes (no product caller:
# bindings/classical_labels_door.mojo labels_label_kind, the forest archive reader and
# bindings/svc_common_door.mojo own them) deleted 2026-11-14 by lane py-admission; code
# recoverable at 1fac46234f; the test oracles' copy of label_kind is tests/_former_prep_labels.py.


# TOMBSTONE: _labels._decode_list_native / _decode_labels_native (the Python
# decode seams; every decode is one native call now) deleted 2026-11-09 by lane
# p2-labels-lists-arrays; the frozen copies the legacy test oracle reads live in
# tests/_legacy_host_inference.py.


class ClassLabels(list):
    """`classes_` of a fitted classifier: the class list of the ORDER RULE
    above (a `list`, so every internal use stays a list use), which also
    takes an integer array as an index, as an ndarray `classes_` does:
    `clf.classes_[codes]` gives the label of each code, the idiom
    scikit-learn's callers use to turn `decision_function` columns or an
    argmax back into labels. The lookup is `decode_labels`: the native
    gather for int and float classes, the label output step for str."""

    __slots__ = ()

    def __getitem__(self, key):
        try:
            return list.__getitem__(self, key)
        except TypeError:
            if key is None or isinstance(key, (str, bytes, float, tuple, dict)):
                raise
        # the integer array index: its refusals, widening and decode are one
        # native call (bindings/labels_admission_unpack.mojo `labels_class_index_op`)
        from ._native_labels_bridge import labels_entry, LABELS_OP_CLASS_INDEX
        entry, context = labels_entry()
        return entry(LABELS_OP_CLASS_INDEX, (self, key), context)

    # Pickle and copy take object.__reduce_ex__'s C path: NEWOBJ with no
    # arguments, then the list items appended. `__slotnames__` is filled
    # here so CPython never calls copyreg._slotnames (Python) for the class.
    # The former Python `__reduce__` ran once per pickled classes_ list.
    __slotnames__ = []


def argmax_rows(scores):
    """Row-wise first-max-wins argmax of an `[n_rows, n_cols]` block of
    scores, as an int64 `Array`: the base binding's scan (DEVIATION 2500,
    strict `>` from column 0, so ties keep the lowest column and a NaN
    never replaces). Every input takes it: a non-`Array` buffer is read in
    C order, and integer scores are widened to float64 (exact below 2**53,
    so the order and the ties are the same). One native call
    (bindings/labels_admission_unpack.mojo `labels_argmax_op`)."""
    from ._native_labels_bridge import labels_entry, LABELS_OP_ARGMAX_ROWS
    entry, context = labels_entry()
    return entry(LABELS_OP_ARGMAX_ROWS, (scores,), context)


def unique_inverse(y):
    """`(classes, codes)` for one NUMERIC label vector, both `Array`s:
    the sorted distinct values (float64 for real labels, int64 for integer
    and bool labels) and one int32 code per row, `classes[codes] == y`.
    This is `np.unique(y, return_inverse=True)` computed by ONE call of the
    shared entry `prepare_targets` (on a GPU install a device sort, a
    flag/scan compaction and a gather, `core/label_encode_device.mojo`; on a
    CPU-only install the core host twin, `core/label_encode.mojo`).
    `-0.0` and `0.0` are one class (the first row's spelling kept); a NaN
    label, an empty y and a y that is not a vector are refused. str labels
    have no device sort: they take the str encoder or the ORDER RULE and
    `classes` comes back a list. One native call
    (bindings/labels_admission_unpack.mojo `labels_unique_inverse_op`)."""
    from ._native_labels_bridge import labels_entry, LABELS_OP_UNIQUE_INVERSE
    entry, context = labels_entry()
    return entry(LABELS_OP_UNIQUE_INVERSE, (y,), context)

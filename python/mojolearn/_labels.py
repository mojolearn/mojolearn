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
from itertools import chain
from . import _portable_math as math
import numbers
import operator

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


def _is_nested(item):
    return isinstance(item, (list, tuple))


def _level(item):
    """One nesting level of a label list: a list or tuple as is, a label
    as a one-element tuple."""
    return item if isinstance(item, (list, tuple)) else (item,)


def flatten_labels(y):
    """`y` as a flat Python list of label objects. Anything with
    `tolist()` (an `Array`, an ndarray, an `array.array`) is unpacked
    through it; nested lists and tuples are flattened, which is what
    `np.asarray(y).ravel()` did for a column vector.

    The explicit label input step for Python objects (str labels have no
    Mojo type): one C-level `chain` pass per nesting LEVEL, so the work is
    the builtins', and depth-first order is kept (a level at a time
    expands every list in place)."""
    if hasattr(y, "tolist") and not isinstance(y, (list, tuple)):
        y = y.tolist()
    if isinstance(y, (str, bytes)):
        raise ValueError("mojolearn: y must be a sequence of labels")
    if type(y) in (list, tuple) and _plain_labels(y) is not None:
        # DEVIATION 3103: nothing to flatten (the type set, read in C)
        return list(y)
    items = [y]
    while any(map(_is_nested, items)):  # glue: one pass per nesting level (depth-sized)
        items = list(chain.from_iterable(map(_level, items)))  # cpu-route: Python label objects, the explicit label input step
    return items


_PLAIN_NUMBERS = frozenset((int, float, bool))
_PLAIN_STR = frozenset((str,))


def _plain_labels(labels):
    """'number' or 'str' when EVERY label is exactly a Python int, float or
    bool, or exactly a str (the type set, read in C); None for anything
    else, an empty sequence included."""
    if not labels:
        return None
    kinds = set(map(type, labels))
    if kinds <= _PLAIN_NUMBERS:
        return "number"
    if kinds == _PLAIN_STR:
        return "str"
    return None


def _classes_and_codes(labels):
    """The ORDER RULE over labels already checked: the FIRST of equal keys
    is the one `dict.fromkeys` keeps, the classes sort by Python's own
    comparison, and each label's code is one dict lookup, driven by `map`."""
    classes = sorted(dict.fromkeys(labels))  # cpu-route: Python label objects (str or Python numbers), the explicit label input step
    code = {c: i for i, c in enumerate(classes)}  # glue: class index table (classes-sized: distinct classes)
    return classes, list(map(code.__getitem__, labels))  # cpu-route: Python label objects, the explicit label input step


def _sorted_plain_classes(labels, kind):
    """DEVIATION 3103: `sorted_classes` for labels `_plain_labels` vouched
    for: one kind (no mixing) of exact Python types, so only a float can be
    a NaN."""
    if kind == "number":
        # `v != v` in C: True only for a NaN float (an int or bool never,
        # at any size)
        if any(map(operator.ne, labels, labels)):  # cpu-route: NaN test of a plain Python label list, the explicit label input step
            raise ValueError(
                "mojolearn: y contains a NaN label; NaN is not a class"
            )
    return _classes_and_codes(labels)


def _kind_of_type(t):
    """'str', 'number' or None (refused) for one label TYPE: the per-label
    tests of the ORDER RULE (rules 1 and 2), asked once per distinct type."""
    if issubclass(t, str):
        return "str"
    if issubclass(t, numbers.Real) or t.__name__ == "bool_":
        return "number"
    return None


def _first_true(flags):
    """Index of the first true flag, or None."""
    try:
        return operator.indexOf(flags, True)
    except ValueError:
        return None


def _check_label_kinds(labels):
    """Rules 1, 2 and 4 of the ORDER RULE over a list of label objects, with
    the refusals in walk order: the first label that is neither a number
    nor a str, or a NaN before it, then a mix of the two kinds."""
    kinds = {t: _kind_of_type(t) for t in set(map(type, labels))}  # glue: over the distinct label types (types-sized)
    refused = frozenset(t for t, kind in kinds.items() if kind is None)  # glue: over the distinct label types (types-sized)
    stop = _first_true(map(refused.__contains__, map(type, labels))) if refused else None
    head = labels if stop is None else labels[:stop]
    if _first_true(map(operator.ne, head, head)) is not None:  # cpu-route: NaN test of Python label objects, the explicit label input step
        raise ValueError(
            "mojolearn: y contains a NaN label; NaN is not a class"
        )
    if stop is not None:
        raise ValueError(
            f"mojolearn: label {labels[stop]!r} is neither a number nor a str"
        )
    found = set(kinds.values())
    if "str" in found and "number" in found:
        raise ValueError(
            "mojolearn: y mixes numeric and str labels; encode one kind "
            "before fitting"
        )


_STR_LABELS = []


def str_labels(operation, *args):
    """The native str-label encoder (bindings/str_labels_native.mojo): ONE
    call over a flat list of label objects. ("encode", labels) gives
    `(classes, int32 codes Array)`; ("lookup", classes, values, as_float)
    gives `(codes Array, first missing labels, missing count)`. None when a
    label is not a str: numbers and mixed lists take `sorted_classes`, whose
    ORDER RULE refusals they keep."""
    if not _STR_LABELS:
        from .tokenizer import _binding
        _STR_LABELS.append(_binding()._module.str_labels_native_call)
    got = _STR_LABELS[0]((operation,) + args)
    if got is None:
        return None
    if operation == "encode":
        classes, store = got
        return classes, Array._owned(store, (len(store),), "<i4", "C")
    store, missing, count = got
    return Array._owned(store, (len(store),), "<f4" if args[2] else "<i4", "C"), missing, count


def sorted_classes(labels):
    """`(classes, codes)`: the class list under the ORDER RULE above, and
    one dense code per label, `classes[codes[i]] == labels[i]` under
    Python equality."""
    if not labels:
        raise ValueError("mojolearn: y is empty")
    if type(labels) is not list:
        labels = list(labels)
    kind = _plain_labels(labels)
    if kind is not None:
        return _sorted_plain_classes(labels, kind)
    _check_label_kinds(labels)
    return _classes_and_codes(labels)


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

#: scikit-learn's own message for an all-zero weight vector; its
#: check_all_zero_sample_weights_error matches "weight" ... "zero".
ZERO_WEIGHTS_MSG = "Sample weights must contain at least one non-zero number."
NEGATIVE_WEIGHTS_MSG = "Negative values in data passed to `sample_weight`"
#: `prepare_targets`' weight statuses (`PT_W_*`) and scikit-learn's wording.
_WEIGHT_REFUSALS = {
    -3: "Input sample_weight contains NaN or infinity.",
    -4: NEGATIVE_WEIGHTS_MSG,
    -5: ZERO_WEIGHTS_MSG,
}
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
    from ._buffer import _native, _output_store

    if labels is not None:
        wide = "<f8" if labels.dtype in ("<f4", "<f8") else "<i8"
        fmt = "d" if wide == "<f8" else "q"
        classes_store = _output_store(fmt, n)
        codes_store = _output_store("i", n)
        y_addr, y_code = labels._addr, _NATIVE_ENCODE[labels.dtype]
        classes_addr, codes_addr = classes_store.buffer_info()[0], codes_store.buffer_info()[0]
    else:
        y_addr = y_code = classes_addr = codes_addr = 0
    w_addr, w_code = (0, 0) if weights is None else (weights._addr, _NATIVE_ENCODE[weights.dtype])
    flags = (_ALLOW_NEGATIVE if allow_negative else 0) | (_ALLOW_ALL_ZERO if allow_all_zero else 0)
    try:
        k = int(_native("prepare_targets")(y_addr, y_code, n, classes_addr, codes_addr,
                                           w_addr, w_code, flags))
    except Exception as exc:  # a Mojo Error crosses as a bare Exception
        if "NaN label" in str(exc):
            raise ValueError(str(exc)) from None
        raise
    if k < 0:
        raise ValueError(_WEIGHT_REFUSALS[k])
    if labels is None:
        return None, None
    # the first k slots hold the classes: one memmove into a k-slot store
    if k < n:
        cut = _output_store(fmt, k)
        ctypes.memmove(cut.buffer_info()[0], classes_store.buffer_info()[0], 8 * k)
        classes_store = cut
    return (Array._owned(classes_store, (k,), wide, "C"),
            Array._owned(codes_store, (n,), "<i4", "C"))


def encode_labels(y):
    """`(classes, codes)` under the ORDER RULE, `codes` an int32 `Array`
    with one dense code per label. One native call for numeric labels;
    `sorted_classes` (the label input step) for str and object labels."""
    fast = _encode_labels_native(y)
    if fast is not None:
        return fast
    flat = flatten_labels(y)
    fast = _encode_label_list_native(flat)
    if fast is not None:
        return fast
    native = str_labels("encode", flat) if type(flat) is list else None
    if native is not None:
        return native
    classes, codes = sorted_classes(flat)  # cpu-route: mixed or non-str object labels, the explicit label input step
    return classes, Array.from_list(codes, "<i4")


def _encode_label_list_native(y):
    """DEVIATION 3103: a FLAT Python list or tuple whose every label is
    EXACTLY an `int`, or exactly a `float`, through the native encoder.

    `sorted_classes` keeps the first-seen OBJECT of each class, which only
    matters when equal labels can be different objects (`1`, `1.0`, `True`).
    With one exact type they cannot: equal ints are indistinguishable, and
    for floats the one distinguishable pair, `0.0` and `-0.0`, is kept
    first-seen by the native encoder too (DEVIATION 2500). So the type set,
    read in C, decides; a bool, a subclass, a str or a mix takes the label
    input step, as does an int outside int64. NaN raises the same
    ValueError. Every length takes it (the 256-label floor that kept short
    lists in Python is gone with the Python encoder)."""
    import array

    if type(y) not in (list, tuple) or not y:
        return None
    kinds = set(map(type, y))
    if kinds == {int}:
        code, dtype = "q", "<i8"
    elif kinds == {float}:
        code, dtype = "d", "<f8"
    else:
        return None
    try:
        store = array.array(code, y)
    except OverflowError:
        return None  # an int outside int64: a class only Python can hold
    return _encode_labels_native(Array._owned(store, (len(store),), dtype, "C"))


def _encode_labels_native(y):
    """`(classes, codes)` for one numeric label vector through
    `prepare_targets`, or None for what only the label input step reads
    (lists, str or object buffers, a matrix, an empty vector)."""
    from ._buffer import Buf, _has_buffer, _materialize, typestr_of

    if isinstance(y, (list, tuple, str, bytes)):
        return None
    if not isinstance(y, Array) and not _has_buffer(y):
        return None
    bool_source = False
    if not isinstance(y, Array):
        try:
            with Buf(y, name="y") as b:
                bool_source = typestr_of(b) == "|b1"
        except (TypeError, ValueError):
            # an object-dtype buffer ('O': str labels in a NumPy object
            # array) or a format no Array holds: the label input step reads
            # it through `tolist()` and names any refusal
            return None
    try:
        arr, _ = _materialize(y, "y")
    except (TypeError, ValueError):
        return None  # the label input step names the refusal
    if arr.dtype not in _NATIVE_ENCODE or arr.size == 0:
        return None
    # `flatten_labels` reads the LOGICAL order; storage order equals it only
    # for a vector (rank 1, or every other axis of length 1).
    if arr.size != max(arr.shape):  # glue: largest axis of the shape
        return None
    classes, codes = prepare_targets(arr, None, int(arr.size))
    del arr
    values = classes.tolist()
    if bool_source:
        values = list(map(bool, values))  # glue: the k class values
    return values, codes


def label_kind(classes):
    """'int' if every class is an integer (bools excluded), 'float' if
    every class is a real number, else 'object'."""
    if all(isinstance(c, numbers.Integral) and not is_bool(c) for c in classes):  # glue: kind test of the class labels (classes-sized: distinct classes)
        return "int"
    if all(isinstance(c, numbers.Real) and not is_bool(c) for c in classes):  # glue: kind test of the class labels (classes-sized: distinct classes)
        return "float"
    return "object"


def decode_labels(classes, codes):
    """Predicted codes back to labels. Integer classes give an int64
    `Array`, real classes a float64 `Array`; str or bool classes give a
    Python list of the label objects, since no `Array` dtype holds them
    (DEVIATION 2340: an ndarray of labels used to come back)."""
    kind = label_kind(classes)
    fast = _decode_labels_native(classes, codes, kind)
    if fast is not None:
        return fast
    values = list(map(classes.__getitem__, map(int, codes)))  # cpu-route: str or bool classes decode to Python objects, the explicit label output step
    try:
        if kind == "int":
            return Array.from_list(list(map(int, values)), "<i8")  # cpu-route: int classes past the native decode range, the explicit label output step
        if kind == "float":
            return Array.from_list(list(map(float, values)), "<f8")  # cpu-route: float classes past the native decode range, the explicit label output step
    except (OverflowError, TypeError):
        # `Array.from_list` reports an int outside int64 as a TypeError
        # (pre-existing: only OverflowError was caught, so a 2**70 class
        # raised instead of returning the label objects; fixed 2026-09-10)
        pass
    return values


def _decode_labels_native(classes, codes, kind):
    """DEVIATION 2500: `classes[code]` per row through the base binding's
    `gather_i64` / `gather_f64` when `codes` is an int64 `Array` (what
    `argmax_rows` returns) and the classes are all int or all float. The
    dtype of the answer is the one `decode_labels` documents; a class that
    does not fit int64 leaves the Python arm to raise and fall back."""
    from ._buffer import _native, _output_store

    if kind not in ("int", "float") or not isinstance(codes, Array):
        return None
    if codes.dtype == "<i4" and codes.ndim == 1 and classes:
        # DEVIATION 3103: int32 codes (`GradientBoostingClassifier.predict`
        # hands these over) used to take the Python arm, four passes over
        # the rows. Widened by `Array.astype` they take the gather. A code
        # the gather refuses (negative: Python indexes from the END, so the
        # Python arm ANSWERS it) sends the call back to the Python arm.
        try:
            return _decode_labels_native(classes, codes.astype("<i8"), kind)
        except (ImportError, AttributeError):
            raise
        except Exception:
            return None
    if codes.dtype != "<i8" or codes.ndim != 1 or not classes:
        return None
    try:
        table = (Array.from_list([int(c) for c in classes], "<i8") if kind == "int"  # glue: packs the class table for the native decode (classes-sized: distinct classes)
                 else Array.from_list([float(c) for c in classes], "<f8"))  # glue: packs the class table for the native decode (classes-sized: distinct classes)
    except (OverflowError, TypeError):
        return None  # a class outside int64: the Python arm returns the objects
    n = int(codes.size)
    store = _output_store("q" if kind == "int" else "d", n)
    fn = _native("gather_i64" if kind == "int" else "gather_f64")
    fn(table._addr, len(classes), codes._addr, n, store.buffer_info()[0])
    return Array._owned(store, (n,), "<i8" if kind == "int" else "<f8", "C")


_INDEX_DTYPES = ("<i8", "<i4", "<u4", "<u2", "<u1", "<i1")


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
        from ._buffer import _materialize
        codes, _ = _materialize(key, "index")
        if codes.dtype not in _INDEX_DTYPES:
            raise TypeError("classes_ takes an integer array as an index, "
                            f"not an array of {codes.dtype}")
        if codes.ndim != 1:
            raise TypeError("classes_ takes a 1-D integer array as an index, "
                            f"not one of shape {codes.shape}")
        if codes.dtype not in ("<i8", "<i4"):
            codes = codes.astype("<i8")
        return decode_labels(list(self), codes)

    def __reduce__(self):
        return (ClassLabels, (list(self),))


def as_class_labels(classes):
    """`classes` as a `ClassLabels` when it is a plain list, else unchanged."""
    if type(classes) is list:
        return ClassLabels(classes)
    return classes


def classes_member(classes):
    """`classes_` as the npz member `save` writes: int64 for integer
    labels, float64 for real labels, the list of str for str labels
    (`_serialize.write_npz` encodes a list of str as a `<U` member, the
    dtype `np.asarray` gave them). Bool labels are stored as int64 and
    come back as ints from `load` (DEVIATION 2340)."""
    kind = label_kind(classes)
    if kind == "int" or all(is_bool(c) for c in classes):  # glue: kind test of the class labels (classes-sized: distinct classes)
        return Array.from_list([int(c) for c in classes], "<i8")  # glue: packs the class table (classes-sized: distinct classes)
    if kind == "float":
        return Array.from_list([float(c) for c in classes], "<f8")  # glue: packs the class table (classes-sized: distinct classes)
    if all(isinstance(c, str) for c in classes):  # glue: kind test of the class labels (classes-sized: distinct classes)
        return list(classes)
    raise ValueError("mojolearn: classes_ holds labels no model file can carry")


def classes_from_member(member):
    """The inverse of `classes_member`, also reading a 0.6.x file's
    `classes` member (int, float, bool or `<U` from `np.unique`)."""
    if isinstance(member, str):
        return [member]
    if isinstance(member, (list, tuple)):
        return [str(v) for v in flatten_labels(list(member))]  # glue: converts a classes_ member of str labels
    dtype = str(getattr(member, "dtype", ""))
    kind = dtype.lstrip("<>|=")[:1]
    values = flatten_labels(member)
    if kind in ("i", "u"):
        return [int(v) for v in values]  # glue: converts a classes_ member (values-sized: distinct classes)
    if kind == "f":
        return [float(v) for v in values]  # glue: converts a classes_ member (values-sized: distinct classes)
    if kind == "b":
        return [bool(v) for v in values]  # glue: converts a classes_ member (values-sized: distinct classes)
    return [str(v) for v in values]  # glue: converts a classes_ member (values-sized: distinct classes)


def argmax_rows(scores):
    """Row-wise first-max-wins argmax of an `[n_rows, n_cols]` block of
    scores, as an int64 `Array`: the base binding's scan (DEVIATION 2500,
    strict `>` from column 0, so ties keep the lowest column and a NaN
    never replaces). Every input takes it: a non-`Array` buffer is wrapped,
    an F-order block is laid out in C order, and integer scores are widened
    to float64 (exact below 2**53, so the order and the ties are the same).
    The per-row Python loop is gone (cpu-gpu-cleanup c-core)."""
    if not isinstance(scores, Array):
        from ._buffer import _materialize
        scores, _ = _materialize(scores, "scores")
    n_rows, n_cols = scores.shape
    if not n_rows:
        return Array.from_list([], "<i8")
    if not n_cols:
        raise ValueError("argmax_rows: scores have no columns")
    if scores.dtype not in ("<f4", "<f8"):
        scores = scores.astype("<f8")
    scores = scores._as_c()
    from ._buffer import _native, _output_store
    fn = _native("argmax_rows_f32" if scores.dtype == "<f4" else "argmax_rows_f64")
    store = _output_store("q", n_rows)
    fn(scores._addr, n_rows, n_cols, store.buffer_info()[0])
    return Array._owned(store, (n_rows,), "<i8", "C")


def threshold_codes(scores, threshold=0.0, *, strict=True, below=0, above=1):
    """`above` where a score is > threshold (`strict`) or >= threshold, else
    `below` (NaN takes `below`), as an int64 `Array`, by the base binding's
    `threshold_labels_i64` (lane cgr4-py-compute: the predict label maps
    that were per-row Python comprehensions)."""
    if not isinstance(scores, Array):
        from ._buffer import _materialize
        scores, _ = _materialize(scores, "scores")
    if scores.dtype not in ("<f4", "<f8"):
        scores = scores.astype("<f8")
    scores = scores._as_c()
    n = scores.size
    if not n:
        return Array.from_list([], "<i8")
    from ._buffer import _native, _output_store
    store = _output_store("q", n)
    _native("threshold_labels_i64")(scores._addr, 0 if scores.dtype == "<f4" else 1, n, float(threshold),
                                    int(bool(strict)), int(below), int(above), store.buffer_info()[0])
    return Array._owned(store, (n,), "<i8", "C")


def finite_integer_codes(arr):
    """Sorted distinct values of a float32 label `Array` as ints, or None
    when any value is non-finite, negative or not an integer.

    Lane cpu2-l2-labels: ONE pass over the rows, the device `unique_inverse`
    (sort, flag/scan, gather; it refuses a NaN, which answers None here).
    Every other test reads only the k sorted classes: finiteness and the sign
    are the two ends (the classes are sorted), integrality is the core helper
    `reduce_stat` over the k classes. The host `all_finite`, the integral
    test and the minimum over all n rows that ran before are gone; Python
    only converts the k class values."""
    from ._array import _NATIVE_CODE, _REDUCE_INTEGRAL
    from ._buffer import _native

    if arr.size == 0:
        return None
    flat = arr._as_c()
    try:
        classes, _ = unique_inverse(flat.reshape((flat.size,)))
    except ValueError:
        return None  # a NaN label
    k = int(classes.size)
    ends = flat_view(classes)
    lo, hi = ends[0], ends[k - 1]
    if lo < 0:
        return None
    if classes.dtype == "<f8":
        if not (hi < float("inf")):
            return None
        if not int(_native("reduce_stat")(classes._addr, _NATIVE_CODE["<f8"], k, _REDUCE_INTEGRAL)):
            return None
    del ends
    return [int(v) for v in classes.tolist()]  # glue: reads the class table (classes-sized: distinct classes)


def unique_inverse(y):
    """`(classes, codes)` for one NUMERIC label vector, both `Array`s:
    the sorted distinct values (float64 for real labels, int64 for integer
    and bool labels) and one int32 code per row, `classes[codes] == y`.
    This is `np.unique(y, return_inverse=True)` computed by ONE call of the
    shared entry `prepare_targets` (its capped encoder, then the base
    binding's `unique_inverse`, cpu-gpu-cleanup w2-pyglue, 2026-10-02): on a
    GPU install a device sort, a flag/scan compaction and a gather
    (`core/label_encode_device.mojo`); on a CPU-only install the core host
    binding's twin of the same statements (`core/label_encode.mojo`).
    `-0.0` and `0.0` are one class (the first row's spelling kept); a NaN
    label is refused. str labels have no device sort: they take
    `sorted_classes` (the ORDER RULE) and `classes` comes back a list."""
    from ._buffer import _materialize

    try:
        arr, _ = _materialize(y, "y")
    except (TypeError, ValueError):
        arr = None
    if arr is None or arr.dtype.lstrip("<>|=")[:1] not in ("f", "i", "u", "b"):
        # str or object labels: no device sort orders text, so these take
        # the ORDER RULE's own encoding (the label input step)
        flat = flatten_labels(y)
        native = str_labels("encode", flat) if type(flat) is list else None
        if native is not None:
            return native
        classes, codes = sorted_classes(flat)  # cpu-route: mixed or non-str object labels, the explicit label input step
        return classes, Array.from_list(codes, "<i4")
    if arr.size == 0:
        raise ValueError("mojolearn: y is empty")
    if arr.size != max(arr.shape):  # glue: largest axis of the shape
        raise ValueError("mojolearn: y must be a vector of labels")
    if arr.dtype not in _NATIVE_ENCODE:
        # float16, int16, int8, ...: widened exactly, as the 64-bit sort read them
        arr = arr.astype("<f8" if arr.dtype.lstrip("<>|=")[:1] == "f" else "<i8")
    return prepare_targets(arr, None, int(arr.size))

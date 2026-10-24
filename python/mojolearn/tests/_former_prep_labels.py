# SPDX-License-Identifier: Apache-2.0
"""The Python routes of LabelEncoder, LabelBinarizer and MultiLabelBinarizer
as they stood before the native label owner (bindings/prep_labels_native.mojo),
kept verbatim as the oracle of tests/test_native_prep_labels_door.py: the
native door must return the same bits and fitted attributes. Test-only code;
the package never imports it."""
import array
import ctypes
import itertools
import numbers

from mojolearn._array import Array
from mojolearn._buffer import as_f32_c, full
from mojolearn import _labels
from mojolearn._labels import flatten_labels, sorted_classes, label_kind
from mojolearn._expansion_prep import (
    _PrepBase, _Prog, _NONE, _CAT_R, _CAT_CH, _mode, _x2d, _codes, _fit_categories, _label_chunk,
    _LABEL_KIND, _LabelBuf, _label_buffer, _class_words, _prep_binding, _optional_prep_entry, _native_helper)

def _block_argmax(pr, arr, widths, drops, check):
    """One code per (row, block): the block_argmax stage over a (n, sum(widths)) input."""
    n, W = arr.shape
    d = len(widths)
    xo = pr.put(arr)
    so = pr.put_list([sum(widths[:j]) for j in range(d)])
    wo = pr.put_list(widths)
    do = pr.put_list(drops) if drops is not None else _NONE
    codes = pr.alloc(n * d)
    pr.stage("block_argmax", n * d, xo, n, W, d, so, wo, do, 1 if check else 0, codes)
    return codes


_INT_FLOAT = frozenset((int, float))
_FLOAT_ONLY = frozenset((float,))
_INT_ONLY = frozenset((int,))


def _numeric_labels(values):
    """The labels as floats when every one is a real number that float32
    holds exactly (so the device's categories are the labels themselves),
    else None (str labels, ints beyond 2**24: the Python route)."""
    # every label read once natively (bindings/user_objects_native.mojo f32_exact)
    from . import _user_objects
    return _user_objects.call("f32_exact", values)


def _label_classes(mode, values):
    """(classes list in the reference's order, device categories Array or
    None). Numeric labels: a device sort and run scan; else sorted()."""
    nums = _numeric_labels(values)
    if nums is None or not nums:
        classes, _ = sorted_classes(values)
        return classes, None
    cats = _fit_categories(mode, Array._from_flat(nums, (len(nums), 1), "<f4"))[0]
    from . import _user_objects
    ints = _user_objects.kinds(values)[_user_objects.KIND_INTEGRAL] == len(values)
    classes = [int(c) if ints else float(c) for c in cats.tolist()]  # glue: class list from the device categories (cats-sized: distinct class values)
    return classes, cats


def _label_codes(pr, values, cats):
    """Stages: each label's index among `cats` (or -1). Returns the codes
    offset and the unknown-count offset."""
    from . import _user_objects
    words = _user_objects.call("floats", values)
    arr = Array._owned(words, (len(words), 1), "<f8", "C").astype("<f4")
    return _codes(pr, arr, [cats])


def _classes_array(classes):
    kind = label_kind(classes)
    if kind == "int":
        return Array.from_list(classes, "<i8")
    if kind == "float":
        return Array.from_list(classes, "<f8")
    return list(classes)


def _multilabel_indicator(y):
    """y as a float32 (n, K) Array when it is the reference's
    `multilabel-indicator` (`type_of_target`): two-dimensional, more than one
    column, at most two distinct values, all integral. A 2-D y of more
    distinct values is `multiclass-multioutput`, refused as the reference
    refuses it; anything else (1-D, one column) is None."""
    shape = getattr(y, "shape", None)
    if shape is not None:
        if len(shape) != 2:
            return None
        return _multilabel_buffer(y, shape)
    from . import _user_objects
    if not (isinstance(y, (list, tuple)) and y and _user_objects.kinds(y)[_user_objects.KIND_SEQ] == len(y)):
        return None
    if len(y[0]) < 2:
        return None
    # the rows read once natively (bindings/user_objects_native.mojo indicator)
    status, words, width = _user_objects.call("indicator", y)
    if status == 1:
        raise ValueError("mojolearn: y rows have different lengths")
    if status == 2:
        raise ValueError("mojolearn: Multioutput target data is not supported with label binarization")
    return _x2d(Array._owned(words, (len(y), width), "<f4", "C"), "y")


def _multilabel_buffer(y, shape):
    """`_multilabel_indicator` of a 2-D buffer (lane py-runtime round 2: no
    `tolist` walk): every value finite and integral (the core helper's
    integral test), at most two distinct values (native min, max and two
    equality counts), then float32 as the list route's `float(v)` made it."""
    if not shape[0] or shape[1] < 2:
        return None
    if "bool" in str(getattr(y, "dtype", "")):
        raise ValueError("mojolearn: Multioutput target data is not supported with label binarization")
    from ._array import _NATIVE_CODE, _REDUCE_INTEGRAL
    from ._buffer import as_f64_c
    Y, _ = as_f64_c(y, ndim=2, name="y")
    refuse = ValueError("mojolearn: Multioutput target data is not supported with label binarization")
    if not int(Y._native_reduce(_REDUCE_INTEGRAL)):
        raise refuse
    lo, hi = Y.min(), Y.max()
    hits = (Y == lo).sum() + ((Y == hi).sum() if hi != lo else 0)  # glue: native equality and native integer sums (Array helpers)
    if hits != Y.size:
        raise refuse
    return _x2d(Y.astype("<f4"), "y")


# lane neural-pass137: a numeric label BUFFER (an ndarray, an Array) takes the
# device from its own bytes. The route it replaces built Python objects for
# every label five times over (tolist, the float32 exactness test, the float
# list, the float32 Array, the type sets) in fit and again in transform, then
# ran two programs whose unique scan and unknown count were ONE thread
# walking every row. Here: one upload of the raw words, `lab_load` (the
# float32 word the old route would have built, NaN where float32 cannot hold
# the label, which sends the call back to the old route), the device sort, a
# chunked run scan, and for fit_transform the codes in the same program.
# The old route stays for labels no numeric buffer holds (str, bool, lists).


def _label_load(pr, lb):
    x = pr.work(lb.n)
    pr.stage("lab_load", lb.n, pr.put_words(lb.words), lb.kind, x)
    return x


def _label_unique(pr, x, n):
    """Stages: the sorted distinct words of x (sort_cols + a chunked run
    scan: `unique_cols`' words). Returns (U offset, count offset)."""
    s = pr.work(n)
    pr.stage("sort_cols", 1, x, n, 1, s, 1)
    ch = _label_chunk(n)
    nch = -(-n // ch)
    cnt, off = pr.work(nch), pr.work(nch)
    u, k = pr.alloc(n), pr.alloc(1)
    pr.stage("uniq_count", nch, s, n, ch, cnt)
    pr.stage("uniq_scan", 1, cnt, nch, off, k)
    pr.stage("uniq_write", nch, s, n, ch, off, u)
    return u, k


def _label_classes_of(pr, lb, u, k):
    """(classes, cats) read back after the run, or None when a label has no
    exact float32 word (NaN sorts last): the caller takes the old route."""
    K = int(pr.values(k, 1)[0])
    cats = pr.get(u, K)
    vals = cats.tolist()
    if not vals or vals[-1] != vals[-1]:
        return None
    classes = [float(c) for c in vals] if lb.is_float else [int(c) for c in vals]  # glue: class list from the device categories (vals-sized: distinct class values)
    return classes, cats


_LABEL_PRESENT = {}


def _label_present_enabled(mode):
    # FAST + Apple (LABEL_DIRECT) and, lane idn-int-prep, IDENTICAL on every
    # vendor and the host column (IDN_LABEL): the binding's export is the switch
    if mode not in ("fast", "identical"):
        return False
    binding = _prep_binding(mode)
    key = id(binding)
    if key not in _LABEL_PRESENT:
        fn = _optional_prep_entry(binding, "x_prep_label_present")
        _LABEL_PRESENT[key] = bool(fn()) if fn is not None else False
    return _LABEL_PRESENT[key]


def _label_fit_present(mode, lb, codes):
    """Exact small-integer distinct labels on GPU, avoiding an n-row sort
    and n-word download. A GPU BAD flag rejects values outside [0, 4096),
    including fractional/inexact labels. The caller then uses its unchanged
    general GPU sort. No host scan of the input or target values."""
    n, R, ch = lb.n, _CAT_R, _CAT_CH
    pr = _Prog()
    x = _label_load(pr, lb)
    fl, bad = pr.work(R), pr.alloc(1)
    u, k = pr.alloc(R), pr.alloc(1)
    nch = -(-R // ch)
    cnt, off = pr.work(nch), pr.work(nch)
    pr.stage("cat_zero", R, fl)
    pr.stage("cat_present", n, x, n, 1, R, fl, bad)
    pr.stage("pres_count", nch, fl, R, ch, cnt)
    pr.stage("uniq_scan", 1, cnt, nch, off, k)
    pr.stage("pres_write", nch, fl, R, ch, off, u)
    out = None
    if codes:
        c = pr.work(n)
        pr.stage("lookup", n, x, n, 1, u, R, k, c)
        out = pr.output(n, "i")
        pr.stage("f2i", n, c, out)
    pr.run(mode)
    if pr.get_i32(bad, 1).tolist()[0]:
        return None
    got = _label_classes_of(pr, lb, u, k)
    if got is None:
        return None
    return got[0], got[1], (pr.get_i32(out, n) if codes else None)


# TOMBSTONE: MOJOLEARN_IDN_LAB_UNIQUE_HIST, MOJOLEARN_IDN_LAB_CODES_DIRECT, MOJOLEARN_IDN_LAB_ONE_ENTRY (losers: label-encoder 2.1-2.6x, label-binarizer istella 1.15-1.39x slower, NV + AMD) deleted 2026-10-11 by de9ce7b39; code recoverable at de9ce7b39^.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_LAB_UNIQUE_HIST_CODES_DIRECT_ONE_ENTRY.patch
def _label_fit_device(mode, lb, codes=False):
    """One program: (classes, cats, int32 codes Array or None), or None for
    the old route."""
    if _label_present_enabled(mode):
        got = _label_fit_present(mode, lb, codes)
        if got is not None:
            return got
    n = lb.n
    pr = _Prog()
    x = _label_load(pr, lb)
    u, k = _label_unique(pr, x, n)
    out = None
    if codes:
        c = pr.work(n)
        pr.stage("lookup", n, x, n, 1, u, n, k, c)
        out = pr.output(n, "i")
        pr.stage("f2i", n, c, out)
    pr.run(mode)
    got = _label_classes_of(pr, lb, u, k)
    if got is None:
        return None
    return got[0], got[1], (pr.get_i32(out, n) if codes else None)


def _label_codes_device(pr, lb, cats):
    """Stages: each label's index among `cats` (-1 when absent). Returns the
    float codes offset."""
    x = _label_load(pr, lb)
    K = cats.size
    c = pr.work(lb.n)
    pr.stage("lookup", lb.n, x, lb.n, 1, pr.put(cats), K, pr.put_scalar(K), c)
    return c


# Numeric labels run on the device (sort + run scan for classes_, a binary
# search for transform); a numeric label buffer crosses as its raw words
# (`_label_buffer`), so fit_transform is one program. str labels take Python.
class LabelEncoder(_PrepBase):
    """Encode target labels as integers ``0`` to ``n_classes - 1``.

    Follows scikit-learn's ``LabelEncoder``. ``classes_`` are the sorted distinct labels and
    ``transform`` returns each label's index as int32. Numeric and string labels are supported. A label
    not seen in fit raises ``ValueError`` in ``transform`` and ``inverse_transform``.

    Attributes
    ----------
    classes_ : array or list
        Sorted distinct labels: an int64 array for integer labels, a float64 array for real labels,
        otherwise a list.
    numeric_mode_ : str
        Numeric mode at fit time; transform runs in the same mode.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> enc = mojolearn.LabelEncoder().fit(np.array([10, 30, 20, 10]))
    >>> enc.transform(np.array([30, 10]))
    >>> enc.inverse_transform(np.array([0, 2]))
    """
    # scikit-learn tags: transforms a 1-D label vector, not an X matrix
    _sklearn_input_tags = {"one_d_array": True, "two_d_array": False}
    _sklearn_target_tags = {"one_d_labels": True}
    _parameters = ()

    def fit(self, y):
        self.numeric_mode_ = _mode()
        lb = _label_buffer(y)
        got = _label_fit_device(self.numeric_mode_, lb) if lb is not None else None
        if got is not None:
            self._classes, self._cats = got[0], got[1]
        else:
            values = flatten_labels(y)
            self._classes, self._cats = _label_classes(self.numeric_mode_, values)
        self.classes_ = _classes_array(self._classes)
        return self

    def fit_transform(self, y):
        # a numeric buffer: classes and codes in ONE program (lane neural-pass137)
        lb = _label_buffer(y)
        if lb is not None:
            self.numeric_mode_ = _mode()
            got = _label_fit_device(self.numeric_mode_, lb, codes=True)
            if got is not None:
                self._classes, self._cats = got[0], got[1]
                self.classes_ = _classes_array(self._classes)
                return got[2]
        return self.fit(y).transform(y)

    def _check_fitted(self):
        if not hasattr(self, "_classes"):
            raise RuntimeError("mojolearn: this LabelEncoder instance is not fitted yet")

    def transform(self, y):
        self._check_fitted()
        lb = _label_buffer(y) if self._cats is not None else None
        if lb is not None:
            n = lb.n
            pr = _Prog()
            codes = _label_codes_device(pr, lb, self._cats)
            out = pr.output(n, "i")
            pr.stage("f2i", n, codes, out)
            ch = _label_chunk(n)
            nch = -(-n // ch)
            neg = pr.alloc(nch)
            pr.stage("chunk_neg", nch, codes, n, ch, neg)
            pr.run(self.numeric_mode_)
            if not any(pr.get_i32(neg, nch).tolist()):
                return pr.get_i32(out, n)
            # an unknown label: the old route names the refusal
        values = flatten_labels(y)
        if not values:
            return Array((0,), "<i4")
        if self._cats is None or _numeric_labels(values) is None:
            native = _labels.str_labels("lookup", list(self._classes), values, False)
            if native is not None:
                codes, missing, count = native
                if count:
                    raise ValueError(f"mojolearn: y contains previously unseen labels: {missing}")
                return codes
            index = {c: i for i, c in enumerate(self._classes)}  # cpu-route: non-str object labels, the explicit label input step
            missing = [v for v in values if v not in index]  # cpu-route: non-str object labels, the explicit label input step
            if missing:
                raise ValueError(f"mojolearn: y contains previously unseen labels: {missing[:5]}")
            return Array.from_list([index[v] for v in values], "<i4")  # cpu-route: non-str object labels, the explicit label input step
        n = len(values)
        pr = _Prog()
        codes, neg = _label_codes(pr, values, self._cats)
        out = pr.alloc(n)
        pr.stage("f2i", n, codes, out)
        pr.run(self.numeric_mode_)
        if pr.values(neg, 1)[0] > 0:
            raise ValueError("mojolearn: y contains previously unseen labels")
        return pr.get_i32(out, n)

    def _inverse_device(self, y):
        """An integer or float code buffer over numeric classes, one program
        on every tier (lane fam2-prep-metrics; lane cpu2-l3-prep: FAST and
        float code buffers too): each code checked (an integer in [0, K)) and
        its class written as the int64 / float64 word returned
        (f2_code_gather, x_prep/fam2.mojo). None: str classes or a code list
        (the explicit input-prep route)."""
        if self._cats is None:
            return None
        lb = _label_buffer(y)
        if lb is None:
            return None
        n = lb.n
        ints = label_kind(self._classes) == "int"
        pr = _Prog()
        out, neg = _stage_class_gather(pr, _label_load(pr, lb), n, self._cats, ints)
        pr.run(self.numeric_mode_)
        if pr.values(neg, 1)[0] > 0:
            raise ValueError("mojolearn: y contains previously unseen labels")
        return _class_words(pr, out, n, ints)

    def inverse_transform(self, y):
        self._check_fitted()
        got = self._inverse_device(y)
        if got is not None:
            return got
        # str classes or a Python code list: the explicit input-prep route (G5)
        from . import _user_objects
        # the codes read once natively, range-checked by the base helper
        codes = _user_objects.call("ints", flatten_labels(y))
        if len(codes) and int(_native_helper("check_indices_i64")(
                codes.buffer_info()[0], len(codes), len(self._classes))) == 1:
            raise ValueError("mojolearn: y contains previously unseen labels")
        return _classes_array(_user_objects.take(self._classes, codes))


def _stage_class_gather(pr, codes, n, cats, ints):
    """Stages f2_code_gather (x_prep/fam2.mojo): the n float codes at `codes`
    to their classes among `cats` as 64-bit words (int64 when ints, else
    binary64), plus the count of codes that are not an integer in
    [0, cats.size). Returns (words offset, count offset)."""
    out, bad, neg = pr.alloc(2 * n), pr.work(n), pr.alloc(1)
    pr.stage("f2_code_gather", n, codes, cats.size, pr.put(cats), 0 if ints else 1, out, bad)
    pr.stage("count_neg", 1, bad, n, 1, neg)
    return out, neg


# Numeric labels take the device route, str labels Python's.
class LabelBinarizer(_PrepBase):
    """Binarize labels in a one-vs-all fashion.

    Follows scikit-learn's ``LabelBinarizer``. ``transform`` returns an int32 matrix with one column per
    class holding ``pos_label`` for the row's class and ``neg_label`` elsewhere; with two classes it
    returns a single column (positive for the second class), and with one class a single column of
    ``neg_label``. A label not seen in fit becomes a row of ``neg_label``. Numeric and string labels are
    supported.

    A 2-D ``y`` with more than one column and at most two distinct integer values is treated as a
    multilabel indicator, as in scikit-learn: ``classes_`` are the column indices, and each nonzero
    entry becomes ``pos_label`` and each zero ``neg_label``.

    Parameters
    ----------
    neg_label : int, default 0
        Value for negative entries.
    pos_label : int, default 1
        Value for positive entries; must be greater than ``neg_label`` (``ValueError`` otherwise).
    sparse_output : bool, default False
        Must be False; True raises ``NotImplementedError``.

    Attributes
    ----------
    classes_ : array or list
        Class labels.
    y_type_ : str
        ``'binary'``, ``'multiclass'`` or ``'multilabel-indicator'``.
    numeric_mode_ : str
        Numeric mode at fit time; transform runs in the same mode.

    Notes
    -----
    ``inverse_transform(Y, threshold=None)`` takes the largest column for multiclass output and applies
    ``threshold`` (by default halfway between ``neg_label`` and ``pos_label``) for binary and multilabel
    output, as in scikit-learn.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> lb = mojolearn.LabelBinarizer().fit(np.array([1, 2, 6, 4, 2]))
    >>> lb.transform(np.array([1, 6]))
    """
    # scikit-learn tags: transforms a 1-D label vector, not an X matrix
    _sklearn_input_tags = {"one_d_array": True, "two_d_array": False}
    _sklearn_target_tags = {"one_d_labels": True}
    _parameters = ("neg_label", "pos_label", "sparse_output")

    def __init__(self, *, neg_label=0, pos_label=1, sparse_output=False):
        self.neg_label = neg_label
        self.pos_label = pos_label
        self.sparse_output = sparse_output

    def fit(self, y):
        if self.sparse_output:
            raise NotImplementedError("mojolearn: LabelBinarizer(sparse_output=True) is not implemented")
        if not (isinstance(self.neg_label, numbers.Integral) and isinstance(self.pos_label, numbers.Integral)
                and self.neg_label < self.pos_label):
            raise ValueError("mojolearn: neg_label must be an integer below pos_label")
        self.numeric_mode_ = _mode()
        ind = _multilabel_indicator(y)
        if ind is not None:
            self._classes, self._cats = list(range(ind.shape[1])), None
            self.classes_ = Array.from_list(self._classes, "<i8")
            self.y_type_ = "multilabel-indicator"
            return self
        lb = _label_buffer(y)
        got = _label_fit_device(self.numeric_mode_, lb) if lb is not None else None
        if got is not None:
            self._classes, self._cats = got[0], got[1]
        else:
            values = flatten_labels(y)
            self._classes, self._cats = _label_classes(self.numeric_mode_, values)
        self.classes_ = _classes_array(self._classes)
        self.y_type_ = "binary" if len(self._classes) <= 2 else "multiclass"
        return self

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def _check_fitted(self):
        if not hasattr(self, "_classes"):
            raise RuntimeError("mojolearn: this LabelBinarizer instance is not fitted yet")

    def transform(self, y):
        self._check_fitted()
        ind = _multilabel_indicator(y)
        if ind is not None or self.y_type_ == "multilabel-indicator":
            if self.y_type_ != "multilabel-indicator":
                raise ValueError("mojolearn: The object was not fitted with multilabel input.")
            if ind is None:
                raise ValueError("mojolearn: y is not a multilabel indicator; the binarizer was fitted with one")
            n, K = ind.shape
            if K != len(self._classes):
                raise ValueError(f"mojolearn: classes {self._classes} mismatch with the {K} label columns in y")
            pr = _Prog()
            out = pr.output(n * K, "i")
            pr.stage("indicator", n * K, pr.put(ind), n * K, 0, _NONE, int(self.neg_label), int(self.pos_label),
                     out)
            pr.run(self.numeric_mode_)
            return pr.get_i32(out, (n, K))
        K = len(self._classes)
        binary = K <= 2
        W = 1 if binary else K
        lb = _label_buffer(y) if self._cats is not None and K > 1 else None
        if lb is not None:
            # lane neural-pass137: the raw words up, the codes and the dense
            # output on the device, no Python object per label
            n = lb.n
            pr = _Prog()
            codes = _label_codes_device(pr, lb, self._cats)
            out = pr.output(n * W, "i")
            pr.stage("label_binarize", n * W, codes, n, K, 1 if binary else 0, int(self.neg_label),
                     int(self.pos_label), W, out)
            pr.run(self.numeric_mode_)
            return pr.get_i32(out, (n, W))
        values = flatten_labels(y)
        n = len(values)
        pr = _Prog()
        if self._cats is None or _numeric_labels(values) is None:
            native = _labels.str_labels("lookup", list(self._classes), values, True) if values else None
            if native is not None:
                codes = pr.put(native[0])
            else:
                index = {c: i for i, c in enumerate(self._classes)}  # cpu-route: non-str object labels, the explicit label input step
                codes = pr.put_list([index.get(v, -1) for v in values])  # cpu-route: non-str object labels, the explicit label input step
        else:
            codes, _neg = _label_codes(pr, values, self._cats)
        if K == 1:
            codes = pr.put_list([-1] * n)
        out = pr.output(n * W, "i")
        pr.stage("label_binarize", n * W, codes, n, K, 1 if binary else 0, int(self.neg_label),
                 int(self.pos_label), W, out)
        pr.run(self.numeric_mode_)
        return pr.get_i32(out, (n, W))

    def inverse_transform(self, Y, threshold=None):
        self._check_fitted()
        arr = _x2d(Y, "Y")
        n, W = arr.shape
        K = len(self._classes)
        pr = _Prog()
        if self.y_type_ == "multilabel-indicator":
            if W != K:
                raise ValueError(f"mojolearn: Y has {W} columns, expected {K}")
            if threshold is None:
                threshold = (self.pos_label + self.neg_label) / 2.0
            out = pr.alloc(n * W)
            pr.stage("indicator", n * W, pr.put(arr), n * W, 1, pr.put_scalar(threshold), 0, 1, out)
            pr.run(self.numeric_mode_)
            return pr.get_i32(out, (n, W))
        if self.y_type_ == "multiclass":
            if W != K:
                raise ValueError(f"mojolearn: Y has {W} columns, expected {K}")
            codes = _block_argmax(pr, arr, [K], None, False)
        else:
            if W > 2:
                raise ValueError("mojolearn: output_type='binary', but y.shape = " + str((n, W)))
            if threshold is None:
                threshold = (self.pos_label + self.neg_label) / 2.0
            # the last column against the threshold, one code per row (c2_bin_code)
            codes = pr.alloc(n)
            pr.stage("c2_bin_code", n, pr.put(arr), n, W, pr.put_scalar(threshold), codes)
        if self._cats is None:
            # str classes: the codes come back for the explicit object-label decode (G5)
            pr.run(self.numeric_mode_)
            from . import _user_objects
            # one class object per code, placed natively (user_objects take)
            idx = _user_objects.call("ints", pr.values(codes, n)) if K != 1 else array.array("q", bytes(8 * n))
            return _classes_array(_user_objects.take(self._classes, idx))
        # numeric classes (lane cpu2-l3-prep): each code's class gathered on the device
        # (f2_code_gather); one class: both codes name it
        ints = label_kind(self._classes) == "int"
        cats = self._cats if K > 1 else Array._from_flat(self._cats.tolist() * 2, (2,), "<f4")
        out, _neg = _stage_class_gather(pr, codes, n, cats, ints)
        pr.run(self.numeric_mode_)
        return _class_words(pr, out, n, ints)


#: lane gap-prep2 (2026-10-02): MultiLabelBinarizer's int label sets cross as
#: ONE int64 buffer (two C-speed passes over the sets) and the rest is the
#: device's: lab_load + the run scan for classes_ (LabelEncoder's route),
#: lookup for the codes and one unit per row (`row_ones`) for the indicator.
#: The Python route flattened the sets three times, converted every label to
#: a float twice and built a float row-owner vector on the host. The same
#: classes and the same int32 words. MOJOLEARN_MLB_DEVICE=0: the Python route.
_MLB_ROW_TYPES = frozenset((list, tuple, set, frozenset))


class _MLBFlat:
    __slots__ = ("n", "lb", "offs")

    def __init__(self, n, lb, offs):
        self.n, self.lb, self.offs = n, lb, offs


def _mlb_flat(y):
    """y's label sets as one int64 label buffer and int32 row offsets, or
    None (the Python route): rows that are not lists, tuples or sets, a
    label that is not a plain int (bool, float, str, numpy scalars), no
    label at all (lane pyglue-numeric deleted the MOJOLEARN_MLB_DEVICE
    switch). glue: the walks below convert the caller's Python containers
    into one buffer, in C (map, chain, array)."""
    rows = y if isinstance(y, (list, tuple)) else None
    if not rows or not set(map(type, rows)) <= _MLB_ROW_TYPES:
        return None
    if set(map(type, itertools.chain.from_iterable(rows))) != _INT_ONLY:
        return None
    try:
        flat = array.array("q", itertools.chain.from_iterable(rows))
    except OverflowError:
        return None
    if not flat or len(flat) > 2 ** 30:
        return None
    lb = _label_buffer(flat)
    if lb is None:
        return None
    offs = Array._owned(array.array("i", itertools.accumulate(map(len, rows), initial=0)), (len(rows) + 1,),  # cpu-route: offsets of the user's Python label rows, the explicit label input step
                        "<i4", "C")
    return _MLBFlat(len(rows), lb, offs)


# Numeric labels take the device route (a sort, a binary search, a scatter
# of ones), str labels Python's.
class MultiLabelBinarizer(_PrepBase):
    """Convert sets of labels to a binary indicator matrix.

    Follows scikit-learn's ``MultiLabelBinarizer``. ``y`` is an iterable of label collections, one per
    sample. ``transform`` returns an int32 matrix with a 1 where the sample has the class. A label not
    seen in fit is ignored (scikit-learn also warns). Numeric and string labels are supported.

    Parameters
    ----------
    classes : array-like, default None
        Class labels in the order of the output columns. None uses the sorted union of all labels in
        fit.
    sparse_output : bool, default False
        Must be False; True raises ``NotImplementedError``.

    Attributes
    ----------
    classes_ : array or list
        Class labels: an int64 array for integer labels, a float64 array for real labels, otherwise a
        list.
    numeric_mode_ : str
        Numeric mode at fit time; transform runs in the same mode.

    Notes
    -----
    ``inverse_transform`` returns a list with one tuple of labels per row.

    Examples
    --------
    >>> import mojolearn
    >>> mlb = mojolearn.MultiLabelBinarizer()
    >>> mlb.fit_transform([[1, 2], [3], [2, 3]])
    """
    # scikit-learn tags: transforms label collections, not an X matrix
    _sklearn_input_tags = {"two_d_array": False}
    _sklearn_target_tags = {"two_d_labels": True}
    _parameters = ("classes", "sparse_output")

    def __init__(self, *, classes=None, sparse_output=False):
        self.classes = classes
        self.sparse_output = sparse_output

    def fit(self, y):
        if self.sparse_output:
            raise NotImplementedError("mojolearn: MultiLabelBinarizer(sparse_output=True) is not implemented")
        self.numeric_mode_ = _mode()
        if self.classes is not None:
            self._classes = list(self.classes)
            nums = _numeric_labels(self._classes)
            self._given = True
            self._cats = None
        elif not self._fit_flat(_mlb_flat(y)):
            flat = [v for row in y for v in row]  # cpu-route: Python iterables of labels, the explicit label input step
            self._classes, self._cats = _label_classes(self.numeric_mode_, flat) if flat else ([], None)
            self._given = False
        self.classes_ = _classes_array(self._classes)
        return self

    def _fit_flat(self, fl):
        """classes_ from the device route (`_mlb_flat`); False when it does
        not apply (the caller takes the Python route)."""
        if fl is None:
            return False
        got = _label_fit_device(self.numeric_mode_, fl.lb)
        if got is None:
            return False
        self._classes, self._cats = got[0], got[1]
        self._given = False
        return True

    def _transform_flat(self, fl):
        """The int32 indicator from the device route: one program."""
        n, K = fl.n, len(self._classes)
        pr = _Prog()
        codes = _label_codes_device(pr, fl.lb, self._cats)
        offs = pr.put_words(fl.offs)
        out = pr.output(n * max(K, 1), "i")
        pr.stage("row_ones", n, codes, offs, K, out)
        pr.run(self.numeric_mode_)
        return pr.get_i32(out, (n, K))

    def fit_transform(self, y):
        if self.classes is None and not self.sparse_output:
            fl = _mlb_flat(y)
            if fl is not None:
                self.numeric_mode_ = _mode()
                if self._fit_flat(fl):
                    self.classes_ = _classes_array(self._classes)
                    return self._transform_flat(fl)
        y = [list(row) for row in y]  # cpu-route: Python iterables of labels, the explicit label input step
        return self.fit(y).transform(y)

    def _check_fitted(self):
        if not hasattr(self, "_classes"):
            raise RuntimeError("mojolearn: this MultiLabelBinarizer instance is not fitted yet")

    def transform(self, y):
        self._check_fitted()
        if self._cats is not None and self._classes:
            fl = _mlb_flat(y)
            if fl is not None:
                return self._transform_flat(fl)
        rows = [list(r) for r in y]  # cpu-route: Python list-of-lists y, the explicit label input step
        n, K = len(rows), len(self._classes)
        flat = [v for r in rows for v in r]  # cpu-route: Python iterables of labels, the explicit label input step
        owner = [i for i, r in enumerate(rows) for _ in r]  # cpu-route: Python iterables of labels, the explicit label input step
        pr = _Prog()
        if not flat:
            out = pr.alloc(n * max(K, 1))
            pr.run(self.numeric_mode_)
            return pr.get_i32(out, (n, K))
        if self._cats is None or _numeric_labels(flat) is None:
            native = _labels.str_labels("lookup", list(self._classes), flat, True)
            if native is not None:
                codes = pr.put(native[0])
            else:
                index = {c: i for i, c in enumerate(self._classes)}  # cpu-route: non-str object labels, the explicit label input step
                codes = pr.put_list([index.get(v, -1) for v in flat])  # cpu-route: Python iterables of labels, the explicit label input step
        else:
            codes, _neg = _label_codes(pr, flat, self._cats)
        ro = pr.put_list(owner)
        out = pr.output(n * max(K, 1), "i")
        pr.stage("scatter_ones", len(flat), codes, ro, K, out)
        pr.run(self.numeric_mode_)
        return pr.get_i32(out, (n, K))

    def inverse_transform(self, yt):
        self._check_fitted()
        rows = yt.tolist() if hasattr(yt, "tolist") else list(yt)
        return [tuple(self._classes[j] for j, v in enumerate(r) if v) for r in rows]  # cpu-route: builds the user list of Python label tuples (object output)
